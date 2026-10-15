import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import inspect
from .layer import RotaryPositionEmbedding, apply_rotary_pos_emb
from .topology import (
    DynamicTopologyRouter,
    compute_expected_block_distance,
    get_dynamic_ultrametric_mask,
)

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x):
        x_f32 = x.float()
        gamma = x_f32.abs().mean(dim=-1, keepdim=True)
        threshold = 0.5 * gamma
        
        x_ternary = torch.zeros_like(x_f32)
        x_ternary[x_f32 > threshold] = 1.0
        x_ternary[x_f32 < -threshold] = -1.0
        
        ctx.save_for_backward(x_f32, threshold)
        return x_ternary * gamma

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        return grad_output

def pack_ternary(x_ternary: torch.Tensor) -> torch.Tensor:
    """
    Packs a float tensor containing {-1.0, 0.0, 1.0} into int32.
    """
    assert x_ternary.shape[-1] % 16 == 0, "Last dimension must be multiple of 16 for packing"
    
    mapped = torch.zeros_like(x_ternary, dtype=torch.int32)
    mapped[x_ternary == -1.0] = 2
    mapped[x_ternary == 1.0] = 1
    
    shape = list(mapped.shape)
    shape[-1] = shape[-1] // 16
    shape.append(16)
    
    mapped = mapped.view(shape)
    packed = torch.zeros(shape[:-1], dtype=torch.int32, device=x_ternary.device)
    
    for i in range(16):
        packed |= (mapped[..., i] << (2 * i))
        
    return packed

def unpack_ternary(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Unpacks an int32 tensor back to float tensor containing {-1.0, 0.0, 1.0}.
    """
    shape = list(packed.shape)
    shape.append(16)
    
    unpacked = torch.zeros(shape, dtype=torch.int32, device=packed.device)
    for i in range(16):
        val = (packed >> (2 * i)) & 3
        unpacked[..., i] = val
        
    unpacked = unpacked.view(original_shape).float()
    
    res = torch.zeros_like(unpacked)
    res[unpacked == 2.0] = -1.0
    res[unpacked == 1.0] = 1.0
    
    return res

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equiv to torch.repeat_interleave(x, dim=1, repeats=n_rep)
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class SurgeryLossRamp(nn.Module):
    """
    Computes auxiliary loss for Llama Surgery to encourage sparsity.
    """
    def __init__(self, lambda_init: float = 0.0, lambda_max: float = 1.0, ramp_steps: int = 1000):
        super().__init__()
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.ramp_steps = ramp_steps

    def get_lambda(self, step: int) -> float:
        if step >= self.ramp_steps:
            return self.lambda_max
        return self.lambda_init + (self.lambda_max - self.lambda_init) * (step / self.ramp_steps)

    def forward(self, g_h: torch.Tensor, step: int) -> torch.Tensor:
        """
        g_h: Tensor of shape (num_heads,)
        Loss penalizes dense execution (g_h \approx 0).
        """
        lambda_t = self.get_lambda(step)
        # Average over heads
        sparsity_penalty = (1.0 - g_h).mean()
        return lambda_t * sparsity_penalty

def _cfg_attr(cfg, attr: str, default=None, layer_idx: Optional[int] = None):
    """Safely read config attributes on both homogeneous and heterogeneous (Gemma 4) configs."""
    if layer_idx is not None:
        per_layer = getattr(cfg, "per_layer_config", None)
        if per_layer is not None and 0 <= layer_idx < len(per_layer):
            try:
                val = getattr(per_layer[layer_idx], attr, None)
                if val is not None:
                    return val
            except Exception:
                pass
    try:
        val = getattr(cfg, attr, default)
        return default if val is None else val
    except Exception:
        d = getattr(cfg, "__dict__", {})
        if attr in d and d[attr] is not None:
            return d[attr]
        return default


class SurgicalLlamaAttention(nn.Module):
    """
    Llama Attention modified for Continuous Sparsification (Surgery).
    Integrates the per-head routing gate to interpolate between dense and sparse topology.
    Supports V3 QK-Mode post-RoPE block routing, Multi-Prime Adèlic GQA head groups,
    Shifted-Window (Swin) Ultrametric Alternation, Cross-Layer Routing Stride, and
    Block-Level Tiled KL Distillation with Dual Lagrangian Density Control.
    """
    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        head_dim_override: Optional[int] = None,
        num_heads_override: Optional[int] = None,
        num_kv_heads_override: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.root_config = None
        self.layer_idx = layer_idx
        self.patched_layer_idx = layer_idx

        # Gemma 4 KV-sharing & layer-type metadata (populated by patch_llama_model if applicable)
        self.is_kv_shared_layer = False
        self.store_full_length_kv = False
        self.layer_type = None
        self.is_sliding = False
        self.sliding_window = None
        self.is_causal = True

        self.embed_dim = _cfg_attr(config, "hidden_size", 4096, layer_idx=layer_idx)
        self.num_heads = (
            num_heads_override
            if num_heads_override is not None
            else _cfg_attr(config, "num_attention_heads", 32, layer_idx=layer_idx)
        )
        self.head_dim = (
            head_dim_override
            if head_dim_override is not None
            else _cfg_attr(config, "head_dim", self.embed_dim // max(1, self.num_heads), layer_idx=layer_idx)
        )

        # GQA compliance
        self.num_key_value_heads = (
            num_kv_heads_override
            if num_kv_heads_override is not None
            else _cfg_attr(config, "num_key_value_heads", self.num_heads, layer_idx=layer_idx)
        )
        self.num_key_value_groups = max(1, self.num_heads // max(1, self.num_key_value_heads))

        # Get surgery-specific config params, default to some values if not present
        self.p = _cfg_attr(config, "surgical_p", 2)
        self.alpha = _cfg_attr(config, "surgical_alpha", 10000.0)
        self.qk_mode = bool(_cfg_attr(config, "surgical_qk_mode", False))
        self.prime_arities = _cfg_attr(config, "surgical_prime_arities", None)
        self.swin_shift = bool(_cfg_attr(config, "surgical_swin_shift", False))
        self.route_stride = int(_cfg_attr(config, "surgical_route_stride", 1))
        self.distill_kl = bool(_cfg_attr(config, "surgical_distill_kl", False))
        self.target_density = float(_cfg_attr(config, "surgical_target_density", 0.25))

        self.scale = 1.0 / math.sqrt(self.head_dim)

        attn_bias = bool(_cfg_attr(config, "attention_bias", False, layer_idx=layer_idx))
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=attn_bias)

        # Optional per-head RMSNorm / LayerNorm preserved from modern architectures (Qwen3, Gemma3/4)
        self.q_norm = None
        self.k_norm = None
        self.v_norm = None

        max_pos_embeddings = _cfg_attr(config, "max_position_embeddings", 8192, layer_idx=layer_idx)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_pos_embeddings)
        self.attn_dropout = nn.Dropout(_cfg_attr(config, "attention_dropout", 0.0, layer_idx=layer_idx))

        init_mode = _cfg_attr(config, "surgical_init_mode", "collapse")
        self.triad_mode = bool(_cfg_attr(config, "surgical_triad_mode", False))
        self.d_shell = int(_cfg_attr(config, "surgical_d_shell", 8))
        self.router = DynamicTopologyRouter(
            embed_dim=self.embed_dim,
            seq_len=max_pos_embeddings,
            num_heads=self.num_heads,
            p=self.p,
            init_mode=init_mode,
            levels=_cfg_attr(config, "surgical_tree_depth", None),
            tree_mode=_cfg_attr(config, "surgical_tree_mode", False),
            qk_mode=self.qk_mode,
            head_dim=self.head_dim,
            prime_arities=self.prime_arities,
            triad_mode=self.triad_mode,
            d_shell=self.d_shell,
        )
        self.p = self.router.p

        # Dual Lagrangian multiplier for target attention density control
        self.register_buffer("density_lambda", torch.tensor(5.0, dtype=torch.float32), persistent=False)

        # Cross-layer routing cache dict reference (populated by patch_llama_model)
        self._route_cache_ref = None
        self._cached_assignments = None
        self._cached_block_assignments = None
        self._cached_q_blk = None
        self._cached_k_blk = None

    # ------------------------------------------------------------------
    # Valid-length masked mean + peak-norm block pooling (Fixes Trap 3)
    # ------------------------------------------------------------------
    @staticmethod
    def _pool_tensor_blocks(x: torch.Tensor, route_block: int) -> torch.Tensor:
        """
        Pools a 4D tensor (B, H, S, D) into blocks of size `route_block` -> (B, H, NB, D)
        using valid-length masked mean + peak-norm token pooling so tail blocks or
        seq_len == 1 decode steps are never diluted by zero-padding.
        """
        B, H, S, D = x.shape
        RB = max(1, int(route_block))
        if RB == 1 or S == 1:
            return x

        NB = -(-S // RB)
        pad_len = NB * RB - S
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len), value=0.0)
        else:
            x_pad = x
        x_blk = x_pad.view(B, H, NB, RB, D)

        # Valid token mask per block
        idx = torch.arange(NB * RB, device=x.device).view(1, 1, NB, RB, 1)
        valid_bool = idx < S
        valid_f = valid_bool.to(x.dtype)
        valid_counts = valid_f.sum(dim=3).clamp(min=1.0)  # (1, 1, NB, 1)

        # 1. Valid-length masked mean
        mean_pool = (x_blk * valid_f).sum(dim=3) / valid_counts  # (B, H, NB, D)

        # 2. Peak-norm token pooling among valid tokens
        norms = torch.norm(x_blk.float(), p=2, dim=-1)  # (B, H, NB, RB)
        norms = norms.masked_fill(~valid_bool.squeeze(-1), float("-inf"))
        peak_idx = norms.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(B, H, NB, 1, D)
        peak_pool = torch.gather(x_blk, dim=3, index=peak_idx).squeeze(3)  # (B, H, NB, D)

        return 0.5 * (mean_pool + peak_pool)

    def _compute_qk_routing(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        tau: float,
        past_key_value=None,
    ):
        """
        Computes block-level and token-expanded routing assignments from post-RoPE Q and K.
        Handles Swin shift on odd layers, valid-length tail pooling, and cross-layer stride caching.
        """
        B, H, seq_len, D = q.shape
        L = k.shape[-2]
        RB = max(1, int(getattr(self.config, "surgical_route_block", 128)))
        stride = max(1, int(getattr(self.config, "surgical_route_stride", self.route_stride)))
        p_idx = getattr(self, "patched_layer_idx", None)
        if p_idx is None:
            p_idx = self.layer_idx

        # Check cross-layer routing reuse when stride > 1 (keyed by (H, D, anchor_idx) for heterogeneous safety)
        if (
            stride > 1
            and p_idx is not None
            and (p_idx % stride != 0)
            and self._route_cache_ref is not None
        ):
            anchor_key = (H, D, (p_idx // stride) * stride)
            cached = self._route_cache_ref.get(anchor_key, self._route_cache_ref.get((p_idx // stride) * stride, None))
            if cached is not None and cached["L"] == L and cached["B"] == B and cached.get("H", H) == H and cached.get("D", D) == D:
                self._cached_block_assignments = cached["block_assignments"]
                self._cached_assignments = cached["assignments"]
                self._cached_q_blk = cached["q_blk"]
                self._cached_k_blk = cached["k_blk"]
                return cached["assignments"], cached["block_assignments"], cached["load_balance_loss"]

        swin_active = (
            bool(getattr(self.config, "surgical_swin_shift", self.swin_shift))
            and p_idx is not None
            and (p_idx % 2 == 1)
            and RB > 1
        )
        shift = (RB // 2) if swin_active else 0

        if past_key_value is None or seq_len > 1 or self._cached_block_assignments is None:
            # Prefill or full sequence routing
            k_curr = k[:, :, -seq_len:, :] if L >= seq_len else k
            q_in = torch.roll(q, shifts=shift, dims=-2) if (shift > 0 and seq_len > 1) else q
            k_in = torch.roll(k_curr, shifts=shift, dims=-2) if (shift > 0 and k_curr.shape[-2] > 1) else k_curr

            q_blk = self._pool_tensor_blocks(q_in, RB)  # (B, H, NB, D)
            k_blk = self._pool_tensor_blocks(k_in, RB)  # (B, H, NB, D)
            qk_desc = torch.cat([q_blk, k_blk], dim=-1)  # (B, H, NB, 2 * D)

            block_assignments, load_balance_loss = self.router(qk_desc, tau_override=tau)
            self._cached_block_assignments = block_assignments
            self._cached_q_blk = q_blk
            self._cached_k_blk = k_blk

            # Expand block assignments to token length seq_len for eager masks
            if RB > 1:
                token_assign = block_assignments.repeat_interleave(RB, dim=2)[:, :, :seq_len, :, :]
                if shift > 0 and seq_len > 1:
                    token_assign = torch.roll(token_assign, shifts=-shift, dims=2)
            else:
                token_assign = block_assignments[:, :, :seq_len, :, :]

            if L > seq_len and self._cached_assignments is not None and self._cached_assignments.shape[2] == L - seq_len:
                assignments = torch.cat([self._cached_assignments, token_assign], dim=2)
            else:
                assignments = token_assign
            self._cached_assignments = assignments
        else:
            # Decode step: seq_len == 1 with existing cache of total length L
            NB_total = -(-L // RB)
            tail_start = (NB_total - 1) * RB
            k_tail = k[:, :, tail_start:L, :]  # (B, H, tail_len, D)
            k_tail_blk = self._pool_tensor_blocks(k_tail, RB)[:, :, :1, :]  # (B, H, 1, D)
            q_tail_blk = q  # (B, H, 1, D) — single query token, zero padding dilution
            qk_tail_desc = torch.cat([q_tail_blk, k_tail_blk], dim=-1)  # (B, H, 1, 2 * D)

            tail_block_assign, load_balance_loss = self.router(qk_tail_desc, tau_override=tau)

            prev_nb = self._cached_block_assignments.shape[2]
            if NB_total > prev_nb:
                block_assignments = torch.cat([self._cached_block_assignments, tail_block_assign], dim=2)
            else:
                block_assignments = torch.cat([self._cached_block_assignments[:, :, :-1], tail_block_assign], dim=2)
            self._cached_block_assignments = block_assignments

            if self._cached_assignments is not None:
                assignments = torch.cat([self._cached_assignments, tail_block_assign], dim=2)
            else:
                assignments = block_assignments.repeat_interleave(RB, dim=2)[:, :, :L, :, :]
            self._cached_assignments = assignments

        if stride > 1 and p_idx is not None and (p_idx % stride == 0) and self._route_cache_ref is not None:
            entry = {
                "B": B,
                "L": L,
                "H": H,
                "D": D,
                "assignments": assignments,
                "block_assignments": block_assignments,
                "load_balance_loss": load_balance_loss,
                "q_blk": self._cached_q_blk,
                "k_blk": self._cached_k_blk,
            }
            self._route_cache_ref[(H, D, p_idx)] = entry
            if self.layer_idx is not None and (self.layer_idx % stride == 0):
                self._route_cache_ref[self.layer_idx] = entry

        return assignments, block_assignments, load_balance_loss

    # ------------------------------------------------------------------
    # Block-Level Tiled KL Distillation + Dual Lagrangian Density Control
    # ------------------------------------------------------------------
    def _compute_block_kl_lagrangian_loss(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        assignments: torch.Tensor,
        block_assignments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes O(NB^2 * D) block-level KL divergence between pooled teacher QK scores
        and differentiable ultrametric cophenetic distances, plus Dual Lagrangian penalty
        for target attention density control. Never materializes an N x N token matrix.
        """
        RB = max(1, int(getattr(self.config, "surgical_route_block", 128)))
        q_blk = self._cached_q_blk if (self.qk_mode and self._cached_q_blk is not None) else self._pool_tensor_blocks(q, RB)
        k_blk = self._cached_k_blk if (self.qk_mode and self._cached_k_blk is not None) else self._pool_tensor_blocks(k, RB)

        if block_assignments is None:
            # Pool token assignments (B, H, S, levels, p) into (B, H, NB, levels, p)
            B, H, S, levels, p_dim = assignments.shape
            NB = q_blk.shape[2]
            if S == NB:
                block_assignments = assignments
            else:
                pad_len = NB * RB - S
                a_pad = F.pad(assignments, (0, 0, 0, 0, 0, pad_len), value=0.0) if pad_len > 0 else assignments
                a_blk = a_pad.view(B, H, NB, RB, levels, p_dim)
                idx = torch.arange(NB * RB, device=assignments.device).view(1, 1, NB, RB, 1, 1)
                valid_f = (idx < S).to(assignments.dtype)
                counts = valid_f.sum(dim=3).clamp(min=1.0)
                block_assignments = (a_blk * valid_f).sum(dim=3) / counts

        NB = block_assignments.shape[2]
        levels = block_assignments.shape[-2]
        surgical_req_depth = getattr(self.config, "surgical_req_depth", None)
        max_dist = getattr(
            self.config,
            "surgical_max_dist",
            (levels - surgical_req_depth) if surgical_req_depth is not None else max(levels // 2, 1),
        )
        if max_dist is None:
            max_dist = max(levels // 2, 1)
        tau = max(float(getattr(self.config, "surgical_tau", 1.0)), 1e-4)

        idx_b = torch.arange(NB, device=q.device)
        causal_blk = idx_b.unsqueeze(1) >= idx_b.unsqueeze(0)  # (NB, NB): j <= i
        causal_mask_4d = causal_blk.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            teacher_scores = torch.matmul(q_blk.float(), k_blk.float().transpose(-2, -1)) * self.scale
            teacher_scores = teacher_scores.masked_fill(~causal_mask_4d, float("-inf"))
            teacher_probs = F.softmax(teacher_scores, dim=-1).nan_to_num(0.0)

        expected_block_dist = compute_expected_block_distance(block_assignments.float(), p=self.p)  # (B, H, NB, NB)
        student_logits = (max_dist - expected_block_dist) / tau
        student_logits_masked = student_logits.masked_fill(~causal_mask_4d, float("-inf"))
        student_log_probs = F.log_softmax(student_logits_masked, dim=-1)
        # Mask non-causal entries to 0.0 before F.kl_div to prevent 0 * (-inf) = NaN in IEEE 754
        student_log_probs_safe = student_log_probs.masked_fill(~causal_mask_4d, 0.0)

        kl_loss = F.kl_div(student_log_probs_safe, teacher_probs, reduction="batchmean") / max(
            1, block_assignments.shape[1] * NB
        )

        # Dual Lagrangian density control on causal block mask
        soft_block_mask = torch.sigmoid((max_dist - expected_block_dist) / 0.5)
        causal_f = causal_mask_4d.float()
        num_causal_pairs = causal_f.sum().clamp(min=1.0)
        measured_density = (soft_block_mask * causal_f).sum(dim=(-2, -1)).mean() / num_causal_pairs

        target_density = float(getattr(self.config, "surgical_target_density", self.target_density))
        lambda_val = self.density_lambda.detach().clone().to(measured_density.dtype)
        lagrangian_penalty = lambda_val * (
            F.relu(measured_density - target_density) ** 2
        )

        if self.training:
            with torch.no_grad():
                lambda_lr = float(getattr(self.config, "surgical_lambda_lr", 0.05))
                self.density_lambda.add_(lambda_lr * (measured_density.detach() - target_density)).clamp_(
                    min=0.0, max=100.0
                )

        self.last_measured_block_density = float(measured_density.detach().item())
        return kl_loss + lagrangian_penalty

    # ------------------------------------------------------------------
    # Opt-in fast prefill backend (config.surgical_attention_backend = "block_list")
    # ------------------------------------------------------------------
    def _block_list_eligible(self, hidden_states, attention_mask, past_key_value) -> bool:
        """True if this call can use the block-list prefill/training path.

        Requires: backend opt-in, a fresh prefill/training sequence (no cached
        keys yet), CUDA fp16/bf16 (or TRITON_INTERPRET=1), no mask override,
        and no padding (batch 1 or no attention_mask), since the kernel applies
        only the causal mask.
        """
        if getattr(self.config, "surgical_attention_backend", "eager") != "block_list":
            return False
        if hidden_states.shape[1] <= 1:
            return False
        if torch.is_grad_enabled() and not getattr(self.config, "surgical_block_list_train", True):
            return False
        import os
        interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"
        if not interpret and (not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16)):
            return False
        if getattr(self.config, "surgical_mask_override", None) is not None:
            return False
        if attention_mask is not None and hidden_states.shape[0] != 1:
            return False
        if past_key_value is not None:
            past_len = 0
            if hasattr(past_key_value, "get_seq_length"):
                try:
                    past_len = int(past_key_value.get_seq_length(self.layer_idx))
                except TypeError:
                    past_len = int(past_key_value.get_seq_length())
            elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                past_len = past_key_value[0].shape[-2]
            if past_len != 0:
                return False
        return True

    def _update_darksnail_shield(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Compute and cache DarkSnail Beltrami + Query-Tail Outlier Shield tokens during prefill.

        Combines:
        1. Post-RoPE Beltrami adjacent key jump flux ||k_j - k_{j-1}||_2 with kernel-9
           max_pool1d neighborhood shielding (protecting multi-token needles & factual clauses).
        2. Boundary Query-Tail resonance max_{t in [S-W_q, S)} <q_t, k_j> * scale with
           kernel-13 max_pool1d span expansion so answer spans adjacent to matched query
           entities (e.g. 'is spherical coordinates.' or QASPER citations) are preserved intact.
        """
        B, H, S, _ = k.shape
        local_window = int(getattr(self.config, "surgical_local_window", 16))
        shield_k = int(getattr(self.config, "surgical_darksnail_shield_k", 96))
        w_q = min(48, S)

        with torch.no_grad():
            k_f = k.to(torch.float32)
            q_tail = q[:, :, -w_q:, :].to(torch.float32)

            # 1. Beltrami adjacent key jump flux ||k_j - k_{j-1}||_2
            dk = torch.zeros((B, H, S), dtype=torch.float32, device=k.device)
            if S > 1:
                dk[:, :, 1:] = torch.norm(k_f[:, :, 1:, :] - k_f[:, :, :-1, :], p=2, dim=-1)
            flux_norm = dk / (dk.mean(dim=-1, keepdim=True) + 1e-6)
            flux_shield = F.max_pool1d(
                flux_norm.reshape(B * H, 1, S), kernel_size=9, stride=1, padding=4
            ).reshape(B, H, S)

            # 2. Boundary Query-Tail resonance (max over final w_q prompt tokens)
            res_scores = torch.matmul(q_tail, k_f.transpose(-2, -1)) * self.scale  # (B, H, w_q, S)
            res_max = res_scores.max(dim=-2).values                                # (B, H, S)
            res_shield = F.max_pool1d(
                res_max.reshape(B * H, 1, S), kernel_size=13, stride=1, padding=6
            ).reshape(B, H, S)

            # Standardize and fuse Query-Tail resonance + Beltrami flux
            res_z = (res_shield - res_shield.mean(dim=-1, keepdim=True)) / (
                res_shield.std(dim=-1, keepdim=True) + 1e-6
            )
            flux_z = (flux_shield - flux_shield.mean(dim=-1, keepdim=True)) / (
                flux_shield.std(dim=-1, keepdim=True) + 1e-6
            )
            salience = torch.maximum(res_z, 0.85 * flux_z) + 0.35 * (res_z + flux_z)  # (B, H, S)
            self._darksnail_token_salience = salience

            # Mask out already-unconditional positions: sink (0) and local window [S - local_window, S)
            masked_sal = salience.clone()
            masked_sal[:, :, 0] = float("-inf")
            if local_window > 0 and S > local_window:
                masked_sal[:, :, max(1, S - local_window):] = float("-inf")

            avail = max(1, S - max(1, local_window) - 1)
            actual_k = min(shield_k, avail, S)
            if actual_k > 0:
                _, top_idx = masked_sal.topk(actual_k, dim=-1)  # (B, H, actual_k)
                self._darksnail_shield_tokens = top_idx
            else:
                self._darksnail_shield_tokens = None

    def _apply_darksnail_block_bridge(self, bl, RB: int, S: int, local_blocks: int):
        """Measure-preserving Callosal Block Bridge for _block_list_prefill.

        Re-targets the interior active key blocks of the final question-tail route blocks
        (NB - 1 and NB - 2) to the highest-salience Beltrami + Query-Tail context blocks,
        donating slots from earlier full-length filler rows when NB - 1 is under-provisioned
        at deep sparsity (r >= 3) so that:
          - bl.counts.sum() (block density) is <= routed@r
          - last_mean_allowed_keys (causal key budget) is <= routed@r
          - ascending block order in bl.lists is strictly preserved.
        """
        from ultrametric.block_list import BlockLists

        salience = getattr(self, "_darksnail_token_salience", None)
        if salience is None:
            return bl

        lists = bl.lists.clone()
        counts = bl.counts.clone()
        max_count = bl.max_count
        B, H, NB = counts.shape
        if NB <= local_blocks + 3 or max_count <= 2:
            return bl

        bridge_k = int(getattr(self.config, "surgical_darksnail_bridge_blocks", 4))
        pad_len = NB * RB - S
        sal_pad = F.pad(salience, (0, pad_len), value=float("-inf")) if pad_len > 0 else salience[:, :, : NB * RB]
        blk_sal = sal_pad.view(B, H, NB, RB).max(dim=-1).values  # (B, H, NB)

        # Work on CPU int32 views for fast index manipulation, then copy back once
        lists_cpu = lists.cpu()
        counts_cpu = counts.cpu()
        sal_cpu = blk_sal.cpu()

        for b_i in range(B):
            for h_i in range(H):
                # 1. Optionally upgrade NB - 2 existing interior blocks (zero count change)
                if NB >= local_blocks + 5:
                    qb2 = NB - 2
                    int_end2 = qb2 - local_blocks
                    cnt2 = int(counts_cpu[b_i, h_i, qb2].item())
                    n_loc2 = qb2 - int_end2 + 1
                    n_int2 = max(0, cnt2 - 1 - n_loc2)
                    if n_int2 > 0 and int_end2 > 1:
                        top2 = torch.argsort(sal_cpu[b_i, h_i, 1:int_end2], descending=True)[:n_int2] + 1
                        top2_sorted, _ = torch.sort(top2.to(torch.int32))
                        lists_cpu[b_i, h_i, qb2, 1 : 1 + n_int2] = top2_sorted

                # 2. Upgrade final question block qb = NB - 1, donating from filler rows if needed
                qb = NB - 1
                int_end = qb - local_blocks
                if int_end <= 1:
                    continue
                m_int = int_end - 1
                cnt = int(counts_cpu[b_i, h_i, qb].item())
                n_loc = qb - int_end + 1
                n_int = max(0, cnt - 1 - n_loc)
                target_int = max(n_int, min(bridge_k, m_int, max_count - 1 - n_loc))

                if target_int > n_int:
                    need = target_int - n_int
                    donated = 0
                    # Search earlier full-length rows for surplus interior block slots to donate
                    for d_qb in range(local_blocks + 2, NB - 2):
                        if donated >= need:
                            break
                        d_cnt = int(counts_cpu[b_i, h_i, d_qb].item())
                        d_int_end = d_qb - local_blocks
                        d_n_loc = d_qb - d_int_end + 1
                        d_n_int = d_cnt - 1 - d_n_loc
                        while d_n_int > 0 and donated < need and (cnt + 1) <= max_count:
                            # Remove slot 1 (first interior block) from donor row d_qb
                            lists_cpu[b_i, h_i, d_qb, 1 : d_cnt - 1] = lists_cpu[b_i, h_i, d_qb, 2:d_cnt].clone()
                            lists_cpu[b_i, h_i, d_qb, d_cnt - 1] = 0
                            d_cnt -= 1
                            d_n_int -= 1
                            counts_cpu[b_i, h_i, d_qb] = d_cnt
                            cnt += 1
                            n_int += 1
                            counts_cpu[b_i, h_i, qb] = cnt
                            donated += 1

                if n_int > 0:
                    top_int = torch.argsort(sal_cpu[b_i, h_i, 1:int_end], descending=True)[:n_int] + 1
                    top_sorted, _ = torch.sort(top_int.to(torch.int32))
                    lists_cpu[b_i, h_i, qb, 0] = 0
                    lists_cpu[b_i, h_i, qb, 1 : 1 + n_int] = top_sorted
                    lists_cpu[b_i, h_i, qb, 1 + n_int : cnt] = torch.arange(
                        int_end, qb + 1, dtype=torch.int32
                    )

        return BlockLists(lists_cpu.to(lists.device), counts_cpu.to(counts.device), max_count)

    def _block_list_prefill(self, q, k, v, assignments, r, block_assignments=None):
        """Causal prefill / training with BLOCK-granular routing via the Triton block-list kernel.

        Each route block (config.surgical_route_block tokens, default 128) gets one
        routing vector: the per-level majority vote of its tokens' argmax branches
        (first r levels), or direct block routing when assignments already have length NB.
        When grad is enabled, gathers a zero-valued STE soft_block_bias over active blocks
        so exact language-modeling gradients flow into self.router without an O(N^2) matrix.
        """
        from ultrametric.block_list import block_list_attention, build_block_lists

        RB = int(getattr(self.config, "surgical_route_block", 128))
        B, H, S, _ = q.shape
        NB = -(-S // RB)

        if block_assignments is not None and block_assignments.shape[2] == NB:
            blk_assign = block_assignments
            block_route = blk_assign[:, :, :NB, :r, :].argmax(dim=-1).to(torch.int32)
        elif assignments.shape[2] == NB and NB != S:
            blk_assign = assignments[:, :, :NB]
            block_route = blk_assign[:, :, :NB, :r, :].argmax(dim=-1).to(torch.int32)
        else:
            idx = assignments[:, :, :S, :r, :].argmax(dim=-1)                 # (B, H, S, r)
            votes = F.one_hot(idx, self.p).to(torch.float32)                  # (B, H, S, r, p)
            if NB * RB != S:
                votes = F.pad(votes, (0, 0, 0, 0, 0, NB * RB - S))
            block_route = votes.view(B, H, NB, RB, r, self.p).sum(3).argmax(-1).to(torch.int32)  # (B, H, NB, r)
            blk_assign = None

        local_window = int(getattr(self.config, "surgical_local_window", 16))
        local_blocks = -(-local_window // RB) if local_window > 0 else 0

        bl_kwargs = dict(
            is_causal=True,
            sink_block=True,
            local_blocks=local_blocks,
            arity=self.p,
        )
        if self.router.prime_arities is not None:
            sig = inspect.signature(build_block_lists)
            if "head_arities" in sig.parameters:
                bl_kwargs["head_arities"] = self.router.head_arities

        bl = build_block_lists(block_route, r, **bl_kwargs)

        if (
            getattr(self.config, "surgical_darksnail_mode", False)
            and (not self.training or not torch.is_grad_enabled())
            and getattr(self.config, "surgical_mask_override", None) is None
        ):
            bl = self._apply_darksnail_block_bridge(bl, RB=RB, S=S, local_blocks=local_blocks)

        soft_block_bias = None
        if torch.is_grad_enabled() and assignments is not None and assignments.requires_grad:
            if blk_assign is None:
                levels_dim, p_dim = assignments.shape[-2], assignments.shape[-1]
                pad_len = NB * RB - S
                a_pad = F.pad(assignments, (0, 0, 0, 0, 0, pad_len), value=0.0) if pad_len > 0 else assignments
                a_blk = a_pad.view(B, H, NB, RB, levels_dim, p_dim)
                idx_valid = torch.arange(NB * RB, device=assignments.device).view(1, 1, NB, RB, 1, 1)
                valid_f = (idx_valid < S).to(assignments.dtype)
                blk_assign = (a_blk * valid_f).sum(dim=3) / valid_f.sum(dim=3).clamp(min=1.0)

            levels_dim = blk_assign.shape[-2]
            max_dist = getattr(
                self.config,
                "surgical_max_dist",
                (levels_dim - r) if r is not None else max(levels_dim // 2, 1),
            )
            expected_block_dist = compute_expected_block_distance(blk_assign.float(), p=self.p)  # (B, H, NB, NB)
            soft_logits_full = (max_dist - expected_block_dist) / 0.5
            safe_lists = bl.lists.clamp(min=0, max=NB - 1).to(torch.int64)
            gathered_logits = torch.gather(soft_logits_full, dim=-1, index=safe_lists)
            # Zero-valued STE bias: forward value is 0.0, backward gradient is 1.0 w.r.t. gathered_logits
            soft_block_bias = gathered_logits - gathered_logits.detach()

        if getattr(self.config, "surgical_collect_stats", False):
            lens = torch.full((NB,), RB, dtype=torch.float32, device=q.device)
            lens[-1] = S - (NB - 1) * RB
            c = bl.counts.to(torch.float32)
            allowed = ((c - 1) * RB * lens + lens * (lens + 1) / 2).sum(-1)  # (B, H)
            self.last_mean_allowed_keys = (allowed.mean() / S).item()
            self.last_mean_causal_keys = (S + 1) / 2
            self.last_block_density = (c.sum() / (B * H * NB * (NB + 1) / 2)).item()

        return block_list_attention(
            q, k, v, bl, route_block=RB, is_causal=True, sm_scale=self.scale, soft_block_bias=soft_block_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        **kwargs,
    ):
        # Sync runtime surgical flags from root model.config when self.config is a nested text_config (e.g. Gemma 4)
        if getattr(self, "root_config", None) is not None and self.root_config is not self.config:
            for _k, _v in self.root_config.__dict__.items():
                if _k.startswith("surgical_") or _k in (
                    "preserve_sinks",
                    "use_triton_sparse_attention",
                    "quantize_kv_cache",
                    "kv_cache_bits",
                ):
                    setattr(self.config, _k, _v)

        batch_size, seq_len, _ = hidden_states.size()
        past_key_value = kwargs.get("past_key_value", past_key_values)
        shared_kv_states = kwargs.get("shared_kv_states", None)

        use_block_list = self._block_list_eligible(hidden_states, attention_mask, past_key_value)
        req_depth_cfg = getattr(self.config, "surgical_req_depth", None)
        qk_mode = bool(getattr(self.config, "surgical_qk_mode", self.qk_mode))
        tau = getattr(self.config, "surgical_tau", 1.0)
        p_idx = getattr(self, "patched_layer_idx", None)
        if p_idx is None:
            p_idx = self.layer_idx

        block_assignments = None
        if not qk_mode:
            if use_block_list and req_depth_cfg == 0:
                self.current_penalty = hidden_states.new_zeros(())
                self._cached_assignments = None
                assignments = None
            else:
                stride = max(1, int(getattr(self.config, "surgical_route_stride", self.route_stride)))
                anchor_key = (self.num_heads, self.head_dim, (p_idx // stride) * stride) if p_idx is not None else None
                if (
                    stride > 1
                    and p_idx is not None
                    and (p_idx % stride != 0)
                    and self._route_cache_ref is not None
                    and (anchor_key in self._route_cache_ref or ((p_idx // stride) * stride) in self._route_cache_ref)
                ):
                    anchor = self._route_cache_ref.get(anchor_key, self._route_cache_ref.get((p_idx // stride) * stride))
                    assignments = anchor["assignments"]
                    self.current_penalty = anchor["load_balance_loss"]
                    self._cached_assignments = assignments
                else:
                    curr_assignments, load_balance_loss = self.router(hidden_states, tau_override=tau)
                    self.current_penalty = load_balance_loss

                    if past_key_value is None or seq_len > 1:
                        self._cached_assignments = curr_assignments
                        assignments = curr_assignments
                    else:
                        if hasattr(self, '_cached_assignments') and self._cached_assignments is not None:
                            assignments = torch.cat([self._cached_assignments, curr_assignments], dim=2)
                            self._cached_assignments = assignments
                        else:
                            assignments = curr_assignments
                            self._cached_assignments = assignments

                    if stride > 1 and p_idx is not None and (p_idx % stride == 0) and self._route_cache_ref is not None:
                        entry = {
                            "B": batch_size,
                            "L": assignments.shape[2],
                            "H": self.num_heads,
                            "D": self.head_dim,
                            "assignments": assignments,
                            "block_assignments": None,
                            "load_balance_loss": load_balance_loss,
                            "q_blk": None,
                            "k_blk": None,
                        }
                        self._route_cache_ref[(self.num_heads, self.head_dim, p_idx)] = entry
                        if self.layer_idx is not None and (self.layer_idx % stride == 0):
                            self._route_cache_ref[self.layer_idx] = entry

        # Project Q, K, V (with support for Qwen 3.5 / 3.6 gated q_proj and Gemma 4 k_eq_v + KV sharing)
        q_proj_out = self.q_proj(hidden_states)
        if q_proj_out.shape[-1] == 2 * self.num_heads * self.head_dim:
            q_raw, attn_gate = torch.chunk(q_proj_out, 2, dim=-1)
        else:
            q_raw, attn_gate = q_proj_out, None

        q = q_raw.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if getattr(self, "q_norm", None) is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)

        is_kv_shared = bool(getattr(self, "is_kv_shared_layer", False)) and (
            shared_kv_states is not None and getattr(self, "layer_type", None) in shared_kv_states
        )

        if not is_kv_shared:
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            if self.v_proj is not None:
                v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            else:
                v = k.clone()

            if getattr(self, "k_norm", None) is not None:
                k = self.k_norm(k)
            if getattr(self, "v_norm", None) is not None:
                v = self.v_norm(v)

            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply KV cache fake quantization if enabled (QAT)
            if getattr(self.config, "quantize_kv_cache", False):
                from .qat import FakeQuantizeSTE
                kv_bits = getattr(self.config, "kv_cache_bits", 4)
                q_min = - (2 ** (kv_bits - 1))
                q_max = (2 ** (kv_bits - 1)) - 1
                k_scale = k.abs().max(dim=-1, keepdim=True).values / q_max
                v_scale = v.abs().max(dim=-1, keepdim=True).values / q_max
                k = FakeQuantizeSTE.apply(k, kv_bits, k_scale, q_min, q_max)
                v = FakeQuantizeSTE.apply(v, kv_bits, v_scale, q_min, q_max)

        # Apply RoPE
        skip_rope = getattr(self.config, "surgical_kolibri_rope", False) and getattr(self, "layer_idx", -1) % 5 == 0
        if skip_rope:
            pass
        elif position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            def rotate_half(x):
                x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            rot_dim = cos.shape[-1]
            if rot_dim < q.shape[-1]:
                q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
                q = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
                if not is_kv_shared:
                    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
                    k = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
            else:
                q = (q * cos) + (rotate_half(q) * sin)
                if not is_kv_shared:
                    k = (k * cos) + (rotate_half(k) * sin)
        else:
            cos, sin = self.rope(q)
            if is_kv_shared:
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if is_kv_shared:
            k, v = shared_kv_states[self.layer_type]
            k = k.to(q.device)
            v = v.to(q.device)
        else:
            if past_key_value is not None:
                if hasattr(past_key_value, "update"):
                    cache_kwargs = {"sin": getattr(self, "_dummy", None), "cos": getattr(self, "_dummy", None)}
                    k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
                elif isinstance(past_key_value, tuple):
                    k = torch.cat([past_key_value[0], k], dim=-2)
                    v = torch.cat([past_key_value[1], v], dim=-2)
                    past_key_value = (k, v)
            if getattr(self, "store_full_length_kv", False) and shared_kv_states is not None and getattr(self, "layer_type", None) is not None:
                shared_kv_states[self.layer_type] = (k, v)

        # Broadcast KV for GQA before attention computation (and before QK routing!)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Adaptive Context Gate (for hybrid architectures like Gemma 4):
        # Keep global layers dense when total KV length is below surgical_min_sparse_seq_len
        min_sparse_len = int(getattr(self.config, "surgical_min_sparse_seq_len", 0) or 0)
        if (
            min_sparse_len > 0
            and k.shape[-2] < min_sparse_len
            and (not self.training)
            and getattr(self.config, "surgical_mask_override", None) is None
        ):
            req_depth_cfg = 0

        # Post-RoPE QK Routing (Milestone 2.1)
        if qk_mode:
            if use_block_list and req_depth_cfg == 0:
                self.current_penalty = hidden_states.new_zeros(())
                self._cached_assignments = None
                self._cached_block_assignments = None
                assignments = None
            else:
                assignments, block_assignments, load_balance_loss = self._compute_qk_routing(
                    q, k, tau=tau, past_key_value=past_key_value
                )
                self.current_penalty = load_balance_loss

        if (
            getattr(self.config, "surgical_darksnail_mode", False)
            and seq_len > 1
            and (not self.training or not torch.is_grad_enabled())
            and getattr(self.config, "surgical_mask_override", None) is None
        ):
            self._update_darksnail_shield(q, k)

        # [Opt-in] Fast prefill & training backend: block-granular routing + Triton block-list
        if use_block_list:
            if self.training and assignments is not None and getattr(self.config, "surgical_distill_kl", self.distill_kl):
                self.current_distill_loss = self._compute_block_kl_lagrangian_loss(
                    q, k, assignments, block_assignments=block_assignments
                )
            else:
                self.current_distill_loss = None

            if req_depth_cfg == 0:
                if getattr(self.config, "surgical_collect_stats", False):
                    S = q.shape[-2]
                    self.last_mean_allowed_keys = (S + 1) / 2
                    self.last_mean_causal_keys = (S + 1) / 2
                    self.last_block_density = 1.0
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            else:
                r = req_depth_cfg if req_depth_cfg is not None else assignments.shape[-2] // 2
                out = self._block_list_prefill(
                    q, k, v, assignments, min(r, assignments.shape[-2]), block_assignments=block_assignments
                )
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            if attn_gate is not None:
                out = out * torch.sigmoid(attn_gate)
            return self.o_proj(out), None

        use_triton = getattr(self.config, "use_triton_sparse_attention", False)
        req_depth = getattr(self.config, "surgical_req_depth", 2)
        if use_triton and seq_len > 1 and q.dtype == torch.float16 and not q.requires_grad:
            from .kernel import routing_to_block_indices, ultrametric_attention_triton
            router_indices = routing_to_block_indices(assignments, seq_len=seq_len, block_size=128)
            out = ultrametric_attention_triton(q, k, v, router_indices, local_window=128, req_depth=req_depth, p=self.p)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            if attn_gate is not None:
                out = out * torch.sigmoid(attn_gate)
            out = self.o_proj(out)
            return out, None

        # Raw attention scores
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * self.scale

        L_kv = k.shape[-2]
        if attention_mask is not None:
            mask_slice = attention_mask[..., :seq_len, :L_kv]
            if mask_slice.dtype == torch.bool:
                scores = scores.masked_fill(~mask_slice, float("-inf"))
            else:
                scores = scores + mask_slice
        if seq_len == L_kv and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, L_kv, dtype=torch.bool, device=hidden_states.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        elif seq_len > 1:
            idx_q = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
            idx_k = torch.arange(L_kv, device=hidden_states.device)
            causal_mask = idx_k.unsqueeze(0) > idx_q.unsqueeze(1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Dynamic Sparsification
        L = k.shape[-2]
        local_window = getattr(self.config, "surgical_local_window", 16)
        levels = assignments.shape[-2] if assignments is not None else 5
        surgical_req_depth = req_depth_cfg if req_depth_cfg == 0 else getattr(self.config, "surgical_req_depth", None)

        # FAST INFERENCE PATH (Zero memory churn, O(1) integer prefix comparison)
        is_inference = (not self.training) or (not torch.is_grad_enabled())
        if is_inference:
            if surgical_req_depth == 0:
                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
                attn_weights.nan_to_num_(0.0)
                attn_weights = attn_weights.to(v.dtype)
                attn_weights = self.attn_dropout(attn_weights)
                out = torch.matmul(attn_weights, v)
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                if attn_gate is not None:
                    out = out * torch.sigmoid(attn_gate)
                return self.o_proj(out), attn_weights

            r = min(surgical_req_depth if surgical_req_depth is not None else (levels // 2), levels)
            indices = assignments.argmax(dim=-1)[..., :r]  # (B, H, S_total, r)
            powers = (self.p ** torch.arange(r, device=indices.device))
            branch_id = (indices * powers).sum(dim=-1)      # (B, H, S_total)

            if seq_len == 1 and L > 1:
                curr_branch = branch_id[:, :, -1:].unsqueeze(-1)    # (B, H, 1, 1)
                past_branches = branch_id[:, :, :L].unsqueeze(-2)  # (B, H, 1, L)
                um_mask_bool = (curr_branch == past_branches)      # (B, H, 1, L)
                if local_window > 0:
                    um_mask_bool[..., :, max(0, L - local_window):] = True
            else:
                curr_b = branch_id[:, :, -seq_len:]                # (B, H, seq_len)
                past_b = branch_id[:, :, :L]                       # (B, H, L)
                um_mask_bool = (curr_b.unsqueeze(-1) == past_b.unsqueeze(-2)) # (B, H, seq_len, L)
                if local_window > 0:
                    idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                    idx_k = torch.arange(L, device=hidden_states.device)
                    band = torch.abs(idx_q.unsqueeze(1) - idx_k.unsqueeze(0)) <= local_window
                    um_mask_bool = um_mask_bool | band.unsqueeze(0).unsqueeze(0)

            # Attention Sink (Token 0)
            um_mask_bool[..., :, 0] = True

            if (
                getattr(self.config, "surgical_darksnail_mode", False)
                and getattr(self.config, "surgical_mask_override", None) is None
            ):
                sh_tokens = getattr(self, "_darksnail_shield_tokens", None)
                if seq_len == 1 and L > 1:
                    if sh_tokens is not None:
                        sh_idx = sh_tokens.unsqueeze(2).clamp(0, L - 1)
                        um_mask_bool.scatter_(-1, sh_idx, True)
                    # Live decode-step neighborhood span shield
                    decode_k = min(int(getattr(self.config, "surgical_darksnail_decode_k", 64)), L)
                    if decode_k > 0:
                        live_pool = F.max_pool1d(
                            scores.squeeze(2).reshape(batch_size * self.num_heads, 1, L),
                            kernel_size=9,
                            stride=1,
                            padding=4,
                        ).reshape(batch_size, self.num_heads, 1, L)
                        _, live_top = live_pool.topk(decode_k, dim=-1)
                        um_mask_bool.scatter_(-1, live_top, True)
                elif seq_len > 1 and sh_tokens is not None:
                    w_q = min(16, seq_len)
                    tail_slice = um_mask_bool[:, :, -w_q:, :]
                    before_cnt = tail_slice.sum(dim=(-2, -1))  # (B, H)
                    sh_exp = sh_tokens.unsqueeze(2).expand(-1, -1, w_q, -1).clamp(0, L - 1)
                    tail_slice.scatter_(-1, sh_exp, True)
                    after_cnt = tail_slice.sum(dim=(-2, -1))
                    added = (after_cnt - before_cnt).clamp(min=0)
                    # Measure-preserving donor pruning on preceding filler query rows
                    donor_end = seq_len - w_q
                    donor_start = max(1 + local_window, donor_end - 32)
                    if donor_end > donor_start and int(added.max().item()) > 0:
                        for b_i in range(batch_size):
                            for h_i in range(self.num_heads):
                                rem = int(added[b_i, h_i].item())
                                if rem <= 0:
                                    continue
                                for r_d in range(donor_start, donor_end):
                                    if rem <= 0:
                                        break
                                    k_lim = max(1, (L - seq_len + r_d) - local_window)
                                    if k_lim <= 1:
                                        continue
                                    row_d = um_mask_bool[b_i, h_i, r_d, 1:k_lim]
                                    true_pos = torch.nonzero(row_d, as_tuple=False).squeeze(-1)
                                    if true_pos.numel() > 0:
                                        take = min(rem, int(true_pos.numel()))
                                        row_d[true_pos[:take]] = False
                                        rem -= take

            if getattr(self.config, "surgical_mask_override", None) == "window":
                W = int(getattr(self.config, "surgical_window_size", 256))
                idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                idx_k = torch.arange(L, device=hidden_states.device)
                band = (idx_k.unsqueeze(0) > idx_q.unsqueeze(1) - W)
                um_mask_bool = band.unsqueeze(0).unsqueeze(0).expand_as(um_mask_bool).clone()
                um_mask_bool[..., :, 0] = True
            elif getattr(self.config, "surgical_mask_override", None) == "oracle":
                W = int(getattr(self.config, "surgical_window_size", 256))
                recent_w = min(128, W)
                top_k = max(0, W - recent_w)

                idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                idx_k = torch.arange(L, device=hidden_states.device)
                band = (idx_k.unsqueeze(0) > idx_q.unsqueeze(1) - recent_w).unsqueeze(0).unsqueeze(0)

                um_mask_bool = band.expand_as(um_mask_bool).clone()

                if top_k > 0:
                    oracle_scores = scores.clone()
                    oracle_scores.masked_fill_(band, float('-inf'))
                    if L > seq_len:
                        causal = (idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
                        oracle_scores.masked_fill_(~causal, float('-inf'))

                    actual_top_k = min(top_k, L)
                    _, topk_indices = oracle_scores.topk(actual_top_k, dim=-1)
                    um_mask_bool.scatter_(-1, topk_indices, True)

                um_mask_bool[..., :, 0] = True

            if getattr(self.config, "surgical_collect_stats", False) and seq_len > 1:
                idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                idx_k = torch.arange(L, device=hidden_states.device)
                causal = idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)
                allowed = (um_mask_bool & causal).sum(-1).float()
                self.last_mean_allowed_keys = allowed.mean().item()
                self.last_mean_causal_keys = causal.sum(-1).float().mean().item()

            sparse_scores = scores.masked_fill(~um_mask_bool, float('-inf'))
            is_all_neg_inf = (sparse_scores == float('-inf')).all(dim=-1, keepdim=True)
            sparse_scores.masked_fill_(is_all_neg_inf, 0.0)

            attn_weights = F.softmax(sparse_scores, dim=-1, dtype=torch.float32)
            attn_weights.nan_to_num_(0.0)
            attn_weights = attn_weights.to(v.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            if attn_gate is not None:
                out = out * torch.sigmoid(attn_gate)
            return self.o_proj(out), attn_weights

        # TRAINING PATH (Differentiable Straight-Through Estimator)
        max_dist = getattr(
            self.config,
            "surgical_max_dist",
            (levels - surgical_req_depth) if surgical_req_depth is not None else None
        )
        full_mask = get_dynamic_ultrametric_mask(
            assignments, p=self.p, max_dist=max_dist, local_window=local_window
        ).to(hidden_states.device)

        # Distillation Loss: Block-Level Tiled KL + Dual Lagrangian (Milestone 1.3) or Legacy Oracle BCE
        if self.training:
            if getattr(self.config, "surgical_distill_kl", self.distill_kl):
                self.current_distill_loss = self._compute_block_kl_lagrangian_loss(
                    q, k, assignments, block_assignments=block_assignments
                )
            else:
                with torch.no_grad():
                    W = max(1, L // 4)
                    recent_w = min(128, W)
                    top_k = max(0, W - recent_w)

                    idx_q = torch.arange(L - seq_len, L, device=hidden_states.device)
                    idx_k = torch.arange(L, device=hidden_states.device)
                    band = (idx_k.unsqueeze(0) > idx_q.unsqueeze(1) - recent_w).unsqueeze(0).unsqueeze(0)
                    oracle_mask = band.expand(batch_size, self.num_heads, seq_len, L).clone()

                    causal = (idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
                    if top_k > 0:
                        oracle_scores = scores.clone()
                        oracle_scores.masked_fill_(band, float('-inf'))
                        oracle_scores.masked_fill_(~causal, float('-inf'))

                        actual_top_k = min(top_k, L)
                        _, topk_indices = oracle_scores.topk(actual_top_k, dim=-1)
                        oracle_mask.scatter_(-1, topk_indices, True)

                    oracle_mask[..., :, 0] = True
                    oracle_mask_float = oracle_mask.float()

                causal_bool = causal.expand(batch_size, self.num_heads, seq_len, L)
                self.current_distill_loss = F.binary_cross_entropy(
                    full_mask[causal_bool].clamp(0.0, 1.0),
                    oracle_mask_float[causal_bool]
                )
        else:
            self.current_distill_loss = None

        um_mask_bool = full_mask > 0.5  # Shape: (B, H, S_full, L) or (B, H, L, L)

        if seq_len == 1 and L > 1:
            um_mask_bool = um_mask_bool[:, :, -1:, :]
            full_mask = full_mask[:, :, -1:, :]

        sparse_scores = scores.masked_fill(~um_mask_bool, float('-inf'))

        is_all_neg_inf = (sparse_scores == float('-inf')).all(dim=-1, keepdim=True)
        sparse_scores.masked_fill_(is_all_neg_inf, 0.0)

        attn_weights = F.softmax(sparse_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.nan_to_num(0.0)

        # Multiply by the differentiable soft mask for training
        attn_weights = attn_weights * full_mask
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        attn_weights = attn_weights.to(v.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        if attn_gate is not None:
            out = out * torch.sigmoid(attn_gate)

        out = self.o_proj(out)

        return out, attn_weights

