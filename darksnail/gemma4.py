"""
darksnail_gemma4: Project DarkSnail V2 Adaptation for Google Gemma 4 31B (arXiv:2607.02770).

Adapts the bidirectional Project DarkSnail architecture (Ultrametric Bruhat-Tits
Dyadic Shell Hydrodynamics x Bi-Hemispheric Lateralized Neuromorphic Reasoning)
from `Meta-Llama-3.1-8B-Instruct` (32 homogeneous layers, d_model=4096, head_dim=128)
to `google/gemma-4-31B` (60 heterogeneous 5:1 Local/Global layers, d_model=5376,
global_head_dim=512, num_global_key_value_heads=4, attention_k_eq_v=True, 25% p-RoPE).

Key Architectural Adaptations & Audit Fixes for Gemma 4 31B (arXiv:2607.02770):
1. Selective 5:1 Global Layer Surgery (`GLOBAL_LAYER_INDICES = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`):
   Leaves the 50 Local Sliding-Window (`W = 1024`, `head_dim = 256`, `H_kv = 16`) layers
   untouched and restricts surgical hooks and Bruhat-Tits tree routing strictly to the
   10 Global Full-Attention layers (`head_dim = 512`, `H_kv = 4`, `H_q = 32`, `scale = 1.0`).
2. Dual-Layout Proportional RoPE (`p-RoPE`, `partial_rotary_factor = 0.25`, `theta = 1,000,000.0`)
   Subspace Separation (`ProportionalRoPESubspaceSplitter`) & 3D/Position-Indexed `apply_gemma4_prope`:
   Separates the 25% rotary phase-entangled channels (`d_rope = 128`) from the 75%
   position-invariant semantic content channels (`d_nope = 384`) under both HuggingFace's
   `rotate_half` split-half layout (`[0:64] U [256:320]` vs. `[64:256] U [320:512]`) and
   contiguous prefix layout (`[0:128]` vs. `[128:512]`). Properly handles 3D `(B, S, D)`
   `position_embeddings` from `Gemma4TextRotaryEmbedding` as well as `position_ids` offsets
   during single-token autoregressive decoding (`seq_len == 1, past_len > 0`).
3. Zero-Shot Haar-DCT Spectral Hyperplane Routing (`init_mode="haar_dct"`) + Optional Block-KL
   Lagrangian Router Warmup (`warmup_gemma4_routers_kl`):
   Prevents untrained `init_mode="collapse"` (`+5.0` branch-0 bias) from degenerating into
   100% dense masks when no pre-trained Gemma 4 `.safetensors` sidecar is present, while
   supporting fast self-supervised block-KL distillation warmup on `google/gemma-4-31B`.
4. Full Prefill + Autoregressive Decode (`seq_len == 1`) Parity in `SurgicalGemma4Attention`:
   Includes tail-block decode routing cache updates, live decode-step neighborhood span
   shielding (`decode_k = 64`), measure-preserving donor pruning during prefill,
   `surgical_mask_override in ("window", "oracle")` for iso-budget baselines, and
   `_block_list_prefill` + `_apply_darksnail_block_bridge`.
5. Zero-Copy Submodule Reuse (`skip_proj_init=True`) & 4-Bit `bitsandbytes` Compatibility:
   Reuses `old_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` directly without allocating
   ~7.5 GB of temporary FP32 projection matrices, reading `in_features` / `out_features`
   so 4-bit quantized (`Params4bit` / NF4) checkpoints patch cleanly on 24-40 GB GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Gemma 4 31B Architecture Specification (arXiv:2607.02770)
# ============================================================================

GLOBAL_LAYER_INDICES: Tuple[int, ...] = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
SLIDING_LAYER_INDICES: Tuple[int, ...] = tuple(
    i for i in range(60) if i not in GLOBAL_LAYER_INDICES
)


@dataclass(frozen=True)
class Gemma4_31B_Spec:
    """Ground-truth architectural specification for `google/gemma-4-31B` (arXiv:2607.02770)."""

    model_id: str = "google/gemma-4-31B"
    arxiv_ref: str = "arXiv:2607.02770"
    vocab_size: int = 262144
    d_model: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    # Local Sliding-Window Layers (50 layers: indices i % 6 != 5)
    num_key_value_heads_sliding: int = 16
    head_dim_sliding: int = 256
    sliding_window: int = 1024
    rope_theta_sliding: float = 10000.0
    # Global Full-Attention Layers (10 layers: indices [5, 11, 17, 23, 29, 35, 41, 47, 53, 59])
    num_key_value_heads_global: int = 4
    head_dim_global: int = 512
    rope_theta_global: float = 1000000.0
    partial_rotary_factor: float = 0.25
    attention_k_eq_v: bool = True
    attention_scale: float = 1.0
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    # DarkSnail V2 Surgery & Callosal Bridge Defaults
    global_layer_indices: Tuple[int, ...] = GLOBAL_LAYER_INDICES
    mid_global_hook_layer: int = 29  # 5th Global Layer (30/60 = 50% network depth)
    bridge_dim: int = 64
    callosal_cond_dim: int = 4096  # Frozen brain-ai CallosalConditioning input contract

    @property
    def d_rope_global(self) -> int:
        return int(round(self.head_dim_global * self.partial_rotary_factor))

    @property
    def d_nope_global(self) -> int:
        return self.head_dim_global - self.d_rope_global

    @staticmethod
    def is_global_layer(layer_idx: int, num_layers: int = 60) -> bool:
        return (layer_idx % 6 == 5) or (layer_idx == num_layers - 1)

    @staticmethod
    def snap_to_nearest_global_layer(
        layer_idx: int,
        global_indices: Sequence[int] = GLOBAL_LAYER_INDICES,
    ) -> int:
        """Snaps any requested layer index (e.g., Llama's 16) to the closest Gemma 4 Global layer."""
        if not global_indices:
            return layer_idx
        if layer_idx in global_indices:
            return int(layer_idx)
        if layer_idx == 16 and 29 in global_indices:
            return 29
        return int(min(global_indices, key=lambda g: abs(int(g) - int(layer_idx))))


GEMMA4_31B_SPEC = Gemma4_31B_Spec()


# ============================================================================
# 2. Gemma 4 RMSNorm & Proportional RoPE (p-RoPE) Subspace Geometry
# ============================================================================


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm with direct learnable scale (`weight * x / rms(x)`)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.pow(x_f.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return (normed * self.weight.float()).to(x.dtype)


class Gemma4RMSNormNoScale(nn.Module):
    """Scale-free RMSNorm (`with_scale=False`) used for `v_norm` in Gemma 4 Attention."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.pow(x_f.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return normed.to(x.dtype)


def build_gemma4_prope_cache(
    seq_len: int,
    head_dim: int = 512,
    theta: float = 1000000.0,
    partial_rotary_factor: float = 0.25,
    device: Optional[Union[str, torch.device]] = None,
    layout: Literal["hf_split_half", "contiguous_prefix"] = "hf_split_half",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the Gemma 4 Proportional RoPE (`p-RoPE`) cos/sin cache of shape `(seq_len, head_dim)`.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    nope_angles = (head_dim // 2) - rope_angles

    inv_freq_rot = 1.0 / (
        theta ** (torch.arange(0, 2 * rope_angles, 2, device=dev, dtype=torch.float32) / float(head_dim))
    )
    positions = torch.arange(seq_len, device=dev, dtype=torch.float32)

    if layout == "hf_split_half":
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq_rot, torch.zeros(nope_angles, device=dev, dtype=torch.float32)], dim=-1
            )
        else:
            inv_freq = inv_freq_rot
        freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)
        return emb.cos(), emb.sin()
    elif layout == "contiguous_prefix":
        freqs_rot = torch.outer(positions, inv_freq_rot)  # (seq_len, rope_angles)
        emb_rot = torch.cat([freqs_rot, freqs_rot], dim=-1)  # (seq_len, 2 * rope_angles = d_rope)
        if nope_angles > 0:
            zeros_nope = torch.zeros((seq_len, 2 * nope_angles), device=dev, dtype=torch.float32)
            emb = torch.cat([emb_rot, zeros_nope], dim=-1)
        else:
            emb = emb_rot
        return emb.cos(), emb.sin()
    else:
        raise ValueError(f"Unsupported p-RoPE layout: {layout}")


def apply_gemma4_prope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: Literal["hf_split_half", "contiguous_prefix"] = "hf_split_half",
    position_ids: Optional[torch.Tensor] = None,
    position_offset: int = 0,
) -> torch.Tensor:
    """
    Applies Gemma 4 Proportional RoPE to `x` of shape `(B, H, S, head_dim)`.

    Audit Fix #2:
    Properly handles:
      - 3D `(B, S, head_dim)` `cos/sin` tensors passed directly by HuggingFace `Gemma4TextRotaryEmbedding`
        (without erroneously slicing `cos[:S]` along the batch axis `dim=0`!),
      - 4D `(B, 1, S, head_dim)` pre-unsqueezed `cos/sin` tensors, and
      - 2D `(max_seq_len, head_dim)` cached tables indexed by `position_ids` (`(B, S)`) or
        `position_offset` (`past_len` during `seq_len == 1` autoregressive decoding).
    """
    S = x.shape[-2]
    head_dim = x.shape[-1]
    cos_t = cos.to(device=x.device, dtype=x.dtype)
    sin_t = sin.to(device=x.device, dtype=x.dtype)

    if cos_t.dim() == 2:
        if position_ids is not None:
            c = cos_t[position_ids].unsqueeze(1)  # (B, 1, S, D)
            s = sin_t[position_ids].unsqueeze(1)
        else:
            start = max(0, int(position_offset))
            c = cos_t[start : start + S].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
            s = sin_t[start : start + S].unsqueeze(0).unsqueeze(0)
    elif cos_t.dim() == 3:
        # Shape from HF Gemma4TextRotaryEmbedding: (B, S_cos, D)
        if cos_t.shape[1] > S:
            c = cos_t[:, -S:, :].unsqueeze(1)
            s = sin_t[:, -S:, :].unsqueeze(1)
        else:
            c = cos_t.unsqueeze(1)
            s = sin_t.unsqueeze(1)
    elif cos_t.dim() == 4:
        c = cos_t[:, :, -S:, :] if cos_t.shape[2] > S else cos_t
        s = sin_t[:, :, -S:, :] if sin_t.shape[2] > S else sin_t
    else:
        raise ValueError(f"Unexpected cos tensor rank {cos_t.dim()} with shape {tuple(cos_t.shape)}")

    if layout == "hf_split_half" or c.shape[-1] == head_dim and layout != "contiguous_prefix":
        half = head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        x_rot = torch.cat([-x2, x1], dim=-1)
        return (x * c + x_rot * s).to(x.dtype)
    elif layout == "contiguous_prefix":
        if c.shape[-1] < head_dim:
            d_rope = c.shape[-1]
        else:
            d_rope = int((s.abs().amax(dim=(0, 1, 2)) > 0).sum().item())
            if d_rope == 0:
                d_rope = head_dim // 4
        x_r, x_n = x[..., :d_rope], x[..., d_rope:]
        c_r, s_r = c[..., :d_rope], s[..., :d_rope]
        half_r = d_rope // 2
        xr1, xr2 = x_r[..., :half_r], x_r[..., half_r:]
        xr_rot = torch.cat([-xr2, xr1], dim=-1)
        x_r_out = x_r * c_r + xr_rot * s_r
        return torch.cat([x_r_out, x_n], dim=-1).to(x.dtype)
    else:
        raise ValueError(f"Unsupported p-RoPE layout: {layout}")


class ProportionalRoPESubspaceSplitter(nn.Module):
    """
    Exact subspace projector splitting Gemma 4 Global `head_dim = 512` into:
      - `x_rope` (`d_rope = 128`, 25%): 1M-theta phase-rotated positional channels
      - `x_nope` (`d_nope = 384`, 75%): Pure position-invariant semantic content channels
    """

    def __init__(
        self,
        head_dim: int = 512,
        p_rope_ratio: float = 0.25,
        layout: Literal["contiguous_prefix", "hf_split_half"] = "contiguous_prefix",
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.p_rope_ratio = float(p_rope_ratio)
        self.rope_angles = int(round(self.head_dim * self.p_rope_ratio)) // 2
        self.d_rope = 2 * self.rope_angles
        self.d_nope = self.head_dim - self.d_rope
        self.layout = layout

    def split(
        self,
        x: torch.Tensor,
        layout_override: Optional[Literal["contiguous_prefix", "hf_split_half"]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = layout_override or self.layout
        if mode == "contiguous_prefix":
            return x[..., : self.d_rope], x[..., self.d_rope :]
        elif mode == "hf_split_half":
            half = self.head_dim // 2
            ra = self.rope_angles
            x_rope = torch.cat([x[..., :ra], x[..., half : half + ra]], dim=-1)
            x_nope = torch.cat([x[..., ra:half], x[..., half + ra :]], dim=-1)
            return x_rope, x_nope
        else:
            raise ValueError(f"Unknown p-RoPE layout '{mode}'")

    def merge(
        self,
        x_rope: torch.Tensor,
        x_nope: torch.Tensor,
        layout_override: Optional[Literal["contiguous_prefix", "hf_split_half"]] = None,
    ) -> torch.Tensor:
        mode = layout_override or self.layout
        if mode == "contiguous_prefix":
            return torch.cat([x_rope, x_nope], dim=-1)
        elif mode == "hf_split_half":
            ra = self.rope_angles
            nh = self.d_nope // 2
            r1, r2 = x_rope[..., :ra], x_rope[..., ra:]
            n1, n2 = x_nope[..., :nh], x_nope[..., nh:]
            return torch.cat([r1, n1, r2, n2], dim=-1)
        else:
            raise ValueError(f"Unknown p-RoPE layout '{mode}'")


# ============================================================================
# 3. DarkSnailGemma4GlobalHook (Upgraded from gemini-code-1792152508835.py)
# ============================================================================


class DarkSnailGemma4GlobalHook(nn.Module):
    """
    Hooks strictly into the 10 Global p-RoPE layers of a frozen Gemma 4 31B backbone
    (`GLOBAL_LAYER_INDICES = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`).
    Leaves the 50 Local Sliding-Window (1024-token) layers untouched.
    """

    def __init__(
        self,
        head_dim: int = 512,
        d_model: int = 5376,
        bridge_dim: int = 64,
        p_rope_ratio: float = 0.25,
        prope_layout: Literal["contiguous_prefix", "hf_split_half"] = "contiguous_prefix",
        target_keep_ratio: float = 0.50,
        local_window: int = 32,
        d_shell: int = 16,
        shield_radius: int = 2,
        preserve_sphere_norm: bool = True,
        zero_init_up_for_homotopy: bool = True,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.d_model = int(d_model)
        self.bridge_dim = int(bridge_dim)
        self.p_rope_ratio = float(p_rope_ratio)
        self.d_rope = int(round(self.head_dim * self.p_rope_ratio))
        self.d_nope = self.head_dim - self.d_rope
        self.prope_layout = prope_layout
        self.target_keep_ratio = float(target_keep_ratio)
        self.local_window = max(0, int(local_window))
        self.d_shell = max(2, int(d_shell))
        self.shield_radius = max(0, int(shield_radius))
        self.preserve_sphere_norm = bool(preserve_sphere_norm)

        self.splitter = ProportionalRoPESubspaceSplitter(
            head_dim=self.head_dim,
            p_rope_ratio=self.p_rope_ratio,
            layout=self.prope_layout,
        )

        # 1. brain-ai Lateralized Callosal Bridge (5376-D <-> 64-D continuous manifold)
        self.callosal_down = nn.Linear(self.d_model, self.bridge_dim, bias=False)
        self.callosal_up = nn.Linear(self.bridge_dim, self.d_model, bias=False)

        # 2. Skew-Symmetric Triad Operator (J = A - A^T in 64-D continuous manifold)
        self.triad_gen = nn.Parameter(torch.randn(self.bridge_dim, self.bridge_dim) * 0.02)

        # 3. Deterministic orthogonal coarse/fine shell bases on k_nope (384-D) for Beltrami flux
        p_c, p_f = self._build_nope_shell_bases(self.d_nope, self.d_shell)
        self.register_buffer("nope_proj_coarse", p_c, persistent=False)
        self.register_buffer("nope_proj_fine", p_f, persistent=False)

        with torch.no_grad():
            w_orth = torch.empty(self.bridge_dim, self.d_model, dtype=torch.float32)
            nn.init.orthogonal_(w_orth, gain=1.0)
            self.callosal_down.weight.copy_(w_orth.to(self.callosal_down.weight.dtype))
            if zero_init_up_for_homotopy:
                nn.init.zeros_(self.callosal_up.weight)

        self.last_keep_mask: Optional[torch.Tensor] = None
        self.last_spectral_salience: Optional[torch.Tensor] = None
        self.last_beltrami_flux: Optional[torch.Tensor] = None
        self.last_tangent_ortho_residual: float = 0.0
        self.last_sphere_norm_drift: float = 0.0

    @staticmethod
    def _build_nope_shell_bases(d_nope: int, d_shell: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(d_nope, dtype=torch.float32).unsqueeze(0)
        k_c = torch.arange(1, d_shell + 1, dtype=torch.float32).unsqueeze(1)
        k_f = torch.arange(d_shell + 1, 2 * d_shell + 1, dtype=torch.float32).unsqueeze(1)

        raw_c = torch.cos(math.pi * (idx + 0.5) * k_c / float(d_nope))
        haar = torch.where((torch.arange(d_nope) % 2) == 0, 1.0, -1.0).float().unsqueeze(0)
        raw_f = torch.sin(math.pi * (idx + 0.5) * k_f / float(d_nope)) * haar

        stacked = torch.cat([raw_c, raw_f], dim=0).t()
        if d_nope >= 2 * d_shell:
            q, _ = torch.linalg.qr(stacked, mode="reduced")
            return q[:, :d_shell].t().contiguous(), q[:, d_shell : 2 * d_shell].t().contiguous()
        return F.normalize(raw_c, p=2, dim=-1), F.normalize(raw_f, p=2, dim=-1)

    def get_skew_triad(self) -> torch.Tensor:
        gen_f = self.triad_gen.float()
        return gen_f - gen_f.transpose(-1, -2)

    def compute_nope_beltrami_salience(
        self,
        k_nope: torch.Tensor,
        q_global: Optional[torch.Tensor] = None,
        k_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_nf = k_nope.float()
        B, H_kv, N, D_n = k_nf.shape
        spectral_energy = torch.norm(k_nf, p=2, dim=-1).mean(dim=1)  # [B, N]

        if N <= 1:
            self.last_beltrami_flux = torch.zeros_like(spectral_energy)
            return spectral_energy

        k_mean = k_nf.mean(dim=1)  # [B, N, D_n]
        if D_n == self.nope_proj_coarse.shape[-1]:
            p_c = self.nope_proj_coarse.to(k_nf.device)
            p_f = self.nope_proj_fine.to(k_nf.device)
        else:
            p_c, p_f = self._build_nope_shell_bases(D_n, self.d_shell)
            p_c, p_f = p_c.to(k_nf.device), p_f.to(k_nf.device)

        u_c = F.linear(k_mean, p_c)
        u_f = F.linear(k_mean, p_f)
        if N >= 3:
            u_c = F.avg_pool1d(u_c.transpose(1, 2), kernel_size=3, stride=1, padding=1).transpose(1, 2)

        u_hat_c = F.normalize(u_c, p=2, dim=-1)
        u_hat_f = F.normalize(u_f, p=2, dim=-1)
        cos_cf = (u_hat_c * u_hat_f).sum(dim=-1).clamp(min=-1.0, max=1.0)
        curl = torch.sqrt(torch.clamp(1.0 - cos_cf * cos_cf, min=0.0))

        dk = torch.zeros((B, H_kv, N), dtype=torch.float32, device=k_nf.device)
        diff = torch.norm(k_nf[:, :, 1:, :] - k_nf[:, :, :-1, :], p=2, dim=-1)
        dk[:, :, 1:] = diff
        dk[:, :, :-1] = torch.maximum(dk[:, :, :-1], diff)
        jump_flux = dk.mean(dim=1) * (0.25 + 0.75 * curl)

        if self.shield_radius > 0 and N > 1:
            k_sz = 2 * self.shield_radius + 1
            jump_flux = F.max_pool1d(
                jump_flux.unsqueeze(1), kernel_size=k_sz, stride=1, padding=self.shield_radius
            ).squeeze(1)[:, :N]

        self.last_beltrami_flux = jump_flux.detach()

        e_z = (spectral_energy - spectral_energy.mean(dim=-1, keepdim=True)) / (
            spectral_energy.std(dim=-1, keepdim=True, correction=0) + 1e-6
        )
        f_z = (jump_flux - jump_flux.mean(dim=-1, keepdim=True)) / (
            jump_flux.std(dim=-1, keepdim=True, correction=0) + 1e-6
        )
        salience = spectral_energy * (1.0 + 0.35 * torch.sigmoid(f_z + 0.5 * e_z))

        if q_global is not None and k_global is not None and q_global.shape[-2] >= 1:
            w_q = min(32, q_global.shape[-2])
            q_tail = q_global[:, :, -w_q:, :].float()
            k_full = k_global.float()
            if q_tail.shape[1] != k_full.shape[1]:
                groups = max(1, q_tail.shape[1] // max(1, k_full.shape[1]))
                k_full = k_full.repeat_interleave(groups, dim=1)
            res = torch.matmul(q_tail, k_full.transpose(-2, -1)).amax(dim=(1, 2))
            res_z = (res - res.mean(dim=-1, keepdim=True)) / (
                res.std(dim=-1, keepdim=True, correction=0) + 1e-6
            )
            salience = salience + 0.25 * spectral_energy.mean(dim=-1, keepdim=True) * torch.sigmoid(res_z)

        return salience

    def compute_ultrametric_mask(
        self,
        spectral_salience: torch.Tensor,
        keep_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        sal_f = spectral_salience.float()
        if sal_f.dim() == 1:
            sal_f = sal_f.unsqueeze(0)
        B, N = sal_f.shape
        device = sal_f.device
        ratio = float(keep_ratio if keep_ratio is not None else self.target_keep_ratio)
        ratio = max(0.05, min(1.0, ratio))

        hard_mask = torch.zeros((B, N), dtype=torch.float32, device=device)
        if N == 0:
            return hard_mask.to(spectral_salience.dtype)

        hard_mask[:, 0] = 1.0
        loc_w = min(self.local_window, N)
        if loc_w > 0:
            hard_mask[:, N - loc_w :] = 1.0

        far_end = max(1, N - loc_w)
        if far_end > 1:
            far_len = far_end - 1
            total_far_keep = max(1, min(far_len, int(round(far_len * ratio))))
            outlier_k = max(1, total_far_keep // 2)
            medoid_c = max(0, total_far_keep - outlier_k)

            far_sal = sal_f[:, 1:far_end]
            _, top_idx = torch.topk(far_sal, k=min(outlier_k, far_len), dim=-1)
            hard_mask.scatter_(1, top_idx + 1, 1.0)

            if medoid_c > 0 and far_len >= medoid_c:
                boundaries = [1 + int((c * far_len) // medoid_c) for c in range(medoid_c + 1)]
                boundaries[-1] = far_end
                for c in range(medoid_c):
                    s_c, e_c = boundaries[c], max(boundaries[c] + 1, boundaries[c + 1])
                    if s_c < far_end:
                        e_c = min(e_c, far_end)
                        blk_peak = sal_f[:, s_c:e_c].argmax(dim=-1, keepdim=True) + s_c
                        hard_mask.scatter_(1, blk_peak, 1.0)

        if sal_f.requires_grad:
            mu = sal_f.mean(dim=-1, keepdim=True)
            std = sal_f.std(dim=-1, keepdim=True, correction=0).clamp(min=1e-6)
            soft_gate = torch.sigmoid((sal_f - mu) / (0.5 * std))
            mask_out = hard_mask.detach() - soft_gate.detach() + soft_gate
        else:
            mask_out = hard_mask

        return mask_out.to(spectral_salience.dtype)

    @staticmethod
    def _project_tangent_orthogonal(z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        z_f = z.float()
        dz_f = dz.float()
        denom = (z_f * z_f).sum(dim=-1, keepdim=True).clamp(min=1e-30)
        proj1 = ((z_f * dz_f).sum(dim=-1, keepdim=True) / denom) * z_f
        dz_1 = dz_f - proj1
        proj2 = ((z_f * dz_1).sum(dim=-1, keepdim=True) / denom) * z_f
        return dz_1 - proj2

    @staticmethod
    def _retract_to_sphere(z_updated: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        ref_norm = torch.norm(z_ref.float(), p=2, dim=-1, keepdim=True)
        new_norm = torch.norm(z_updated.float(), p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.where(ref_norm > 0.0, z_updated.float() * (ref_norm / new_norm), z_updated.float())

    def forward_global_kv_hook(
        self,
        k_global: torch.Tensor,
        hidden_states: torch.Tensor,
        dt: float = 0.1,
        steps: int = 4,
        v_global: Optional[torch.Tensor] = None,
        q_global: Optional[torch.Tensor] = None,
        prope_layout: Optional[Literal["contiguous_prefix", "hf_split_half"]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_k_dtype = k_global.dtype
        orig_h_dtype = hidden_states.dtype

        _, k_nope = self.splitter.split(k_global, layout_override=prope_layout)

        spectral_salience = self.compute_nope_beltrami_salience(
            k_nope, q_global=q_global, k_global=k_global
        )
        keep_mask = self.compute_ultrametric_mask(spectral_salience)
        self.last_spectral_salience = spectral_salience.detach()
        self.last_keep_mask = keep_mask.detach()

        mask_4d = keep_mask.to(orig_k_dtype).unsqueeze(1).unsqueeze(-1)
        k_pruned = k_global * mask_4d
        if v_global is None:
            v_pruned = k_pruned
        else:
            v_pruned = v_global * mask_4d

        h_f = hidden_states.float()
        w_down = self.callosal_down.weight.float()
        z0 = F.linear(h_f, w_down)
        z = z0
        J = self.get_skew_triad()
        mask_3d = keep_mask.float().unsqueeze(-1)

        max_ortho = 0.0
        for _ in range(max(1, int(steps))):
            dz_raw = torch.matmul(z, J)
            dz_orth = self._project_tangent_orthogonal(z, dz_raw) * mask_3d
            with torch.no_grad():
                ortho_err = float((z * dz_orth).sum(dim=-1).abs().amax().item())
                if ortho_err > max_ortho:
                    max_ortho = ortho_err
            z_step = z + float(dt) * torch.tanh(dz_orth)
            if self.preserve_sphere_norm:
                z = torch.where(mask_3d > 0.0, self._retract_to_sphere(z_step, z0), z0)
            else:
                z = z_step

        with torch.no_grad():
            self.last_tangent_ortho_residual = max_ortho
            self.last_sphere_norm_drift = float(
                (torch.norm(z, p=2, dim=-1) - torch.norm(z0, p=2, dim=-1)).abs().amax().item()
            )

        w_up = self.callosal_up.weight.float()
        delta_h = F.linear(z, w_up).to(orig_h_dtype)
        return k_pruned, v_pruned, hidden_states + delta_h, keep_mask

    def forward(
        self,
        k_global: torch.Tensor,
        hidden_states: torch.Tensor,
        dt: float = 0.1,
        steps: int = 4,
        v_global: Optional[torch.Tensor] = None,
        q_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_global_kv_hook(
            k_global=k_global,
            hidden_states=hidden_states,
            dt=dt,
            steps=steps,
            v_global=v_global,
            q_global=q_global,
        )


# ============================================================================
# 4. Gemma 4 Beltrami Callosal Shell (Global Layer 29 + 5376 -> 4096 Bridge)
# ============================================================================


class Gemma4BeltramiCallosalShell(nn.Module):
    """
    Gemma 4 31B Left-Hemisphere Adelic + Beltrami Outlier-Shielded Callosal Shell Extractor.
    Hooks Global Layer 29 (50% network depth) and bridges `(B, 32, 5376)` -> `(B, 32, 4096)`
    for frozen `brain-ai` `CallosalConditioning` compatibility.
    """

    def __init__(
        self,
        d_model: int = 5376,
        callosal_cond_dim: int = 4096,
        target_tokens: int = 32,
        outlier_budget: int = 16,
        condensed_budget: int = 16,
        d_shell: int = 16,
        shield_radius: int = 2,
        viscous_nu: float = 0.05,
        hook_layer: int = 29,
        bridge_to_callosal_cond: bool = True,
        sidecar_path: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.callosal_cond_dim = int(callosal_cond_dim)
        self.target_tokens = int(target_tokens)
        self.outlier_budget = int(outlier_budget)
        self.condensed_budget = int(condensed_budget)
        self.d_shell = int(d_shell)
        self.shield_radius = int(shield_radius)
        self.viscous_nu = float(viscous_nu)
        self.hook_layer = Gemma4_31B_Spec.snap_to_nearest_global_layer(hook_layer)
        self.bridge_to_callosal_cond = bool(bridge_to_callosal_cond)

        try:
            from .shell import BeltramiCallosalShell
        except ImportError:
            from darksnail.shell import BeltramiCallosalShell

        self.inner_shell = BeltramiCallosalShell(
            d_model=self.d_model,
            target_tokens=self.target_tokens,
            outlier_budget=self.outlier_budget,
            condensed_budget=self.condensed_budget,
            d_shell=self.d_shell,
            shield_radius=self.shield_radius,
            viscous_nu=self.viscous_nu,
            hook_layer=self.hook_layer,
            sidecar_path=sidecar_path,
        )

        p_bridge = self._build_isometric_callosal_bridge(self.d_model, self.callosal_cond_dim)
        self.register_buffer("callosal_bridge_proj", p_bridge, persistent=False)

    @staticmethod
    def _build_isometric_callosal_bridge(d_in: int, d_out: int) -> torch.Tensor:
        if d_in == d_out:
            return torch.eye(d_out, dtype=torch.float32)
        proj = torch.zeros((d_out, d_in), dtype=torch.float32)
        if d_in > d_out:
            rem = d_in - d_out
            direct = d_out - rem
            if direct > 0:
                idx_d = torch.arange(direct)
                proj[idx_d, idx_d] = 1.0
            idx_r = torch.arange(rem)
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            proj[direct + idx_r, direct + idx_r] = inv_sqrt2
            proj[direct + idx_r, direct + rem + idx_r] = inv_sqrt2
        else:
            idx = torch.arange(d_in)
            proj[idx, idx] = 1.0
        return proj.contiguous()

    def project_to_callosal_4096(self, shell_5376: torch.Tensor) -> torch.Tensor:
        if shell_5376.shape[-1] == self.callosal_cond_dim:
            return shell_5376.float()
        proj = self.callosal_bridge_proj.to(device=shell_5376.device, dtype=torch.float32)
        rms_scale = math.sqrt(float(self.d_model) / float(self.callosal_cond_dim))
        return F.linear(shell_5376.float(), proj) * rms_scale

    def lift_from_callosal_4096(self, shell_4096: torch.Tensor) -> torch.Tensor:
        if shell_4096.shape[-1] == self.d_model:
            return shell_4096.float()
        proj = self.callosal_bridge_proj.to(device=shell_4096.device, dtype=torch.float32)
        inv_scale = math.sqrt(float(self.callosal_cond_dim) / float(self.d_model))
        return torch.matmul(shell_4096.float() * inv_scale, proj)

    def forward(
        self,
        raw_latents: torch.Tensor,
        return_diagnostics: bool = False,
        bridge_to_callosal: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        use_bridge = self.bridge_to_callosal_cond if bridge_to_callosal is None else bool(bridge_to_callosal)
        shell_native, diag = self.inner_shell(raw_latents, return_diagnostics=True)
        diag["hook_layer"] = self.hook_layer
        diag["native_d_model"] = self.d_model
        diag["shell_native_5376"] = shell_native

        if use_bridge:
            shell_out = self.project_to_callosal_4096(shell_native)
            diag["bridged_d_model"] = self.callosal_cond_dim
        else:
            shell_out = shell_native
            diag["bridged_d_model"] = self.d_model

        if return_diagnostics:
            return shell_out, diag
        return shell_out


# ============================================================================
# 5. SurgicalGemma4Attention (Gemma 4 Global Layer Ultrametric + DarkSnail Surgery)
# ============================================================================


def _get_topology_module() -> Any:
    """
    Audit Fix #1: Loads `llama_surgery.topology` without triggering `llama_surgery/__init__.py`
    (which eagerly imports `transformers.Trainer`), with zero recursive self-calls or temp paths.
    """
    import importlib.util
    import sys

    if "_darksnail_ls_topology" in sys.modules:
        return sys.modules["_darksnail_ls_topology"]

    here = Path(__file__).resolve().parent
    search_roots = [here.parent, here, Path.cwd()] + [Path(sp) for sp in sys.path if sp]
    for root in search_roots:
        for sub in ("llama_surgery", "ultrametric"):
            cand = root / sub / "topology.py"
            if cand.exists():
                spec = importlib.util.spec_from_file_location("_darksnail_ls_topology", str(cand))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["_darksnail_ls_topology"] = mod
                    spec.loader.exec_module(mod)
                    return mod

    from llama_surgery import topology as mod
    return mod


def _init_router_haar_dct_spectral_hash(router: nn.Module, seed: int = 0) -> None:
    """
    Audit Fix #3: Deterministic Orthonormal Haar-DCT Spectral Hyperplane Initialization
    for zero-shot / sidecar-free Bruhat-Tits routing on `google/gemma-4-31B`.

    Why this is essential:
    `DynamicTopologyRouter(init_mode="collapse")` sets `route_heads.weight = 0` and
    `route_heads.bias = [+5.0, -5.0]` so that 100% of tokens collapse into branch 0 at
    Step 0 of training. If evaluated without a pre-trained `.safetensors` sidecar,
    `"collapse"` assigns every token `branch_id == 0`, producing a 100% dense mask at
    every depth `r in {2, 3, 4, 5}`!
    In contrast, `"haar_dct"` initializes:
      - `backbone` as a symmetric `0.5 * (q_nope + k_nope)` locality-sensitive projector
        (so queries and keys in the same semantic subspace map to the same tree path!),
      - `route_heads.weight` as QR-orthonormalized multi-scale Haar/DCT spectral hyperplanes
        with zero bias (`route_heads.bias = 0`), and
      - `triad_readout` with small orthogonal weights (`gain = 0.25 / sqrt(d_shell)`),
    yielding genuine hierarchical Bruhat-Tits tree partitions (`~1 / p^r` branch collision)
    immediately at Step 0 even before KL distillation warmup.
    """
    with torch.no_grad():
        in_dim = int(router.in_dim)
        half_d = in_dim // 2
        # Make backbone symmetric across [q_nope, k_nope] halves so Q and K share a common metric space
        w_bb = torch.zeros((in_dim, in_dim), dtype=torch.float32)
        if 2 * half_d == in_dim and half_d > 0:
            eye_h = torch.eye(half_d, dtype=torch.float32) * (1.0 / math.sqrt(2.0))
            w_bb[:half_d, :half_d] = eye_h
            w_bb[:half_d, half_d:] = eye_h
            w_bb[half_d:, :half_d] = eye_h
            w_bb[half_d:, half_d:] = -eye_h
        else:
            nn.init.orthogonal_(w_bb, gain=1.0)
        router.backbone.weight.copy_(w_bb.to(router.backbone.weight.dtype))
        if router.backbone.bias is not None:
            nn.init.zeros_(router.backbone.bias)

        # Build deterministic DCT/Haar hyperplane projections per head
        total_out = router.route_heads.weight.shape[0]
        idx = torch.arange(in_dim, dtype=torch.float32).unsqueeze(0)
        k_freq = torch.arange(1, total_out + 1, dtype=torch.float32).unsqueeze(1)
        haar = torch.where(((torch.arange(in_dim) + int(seed)) % 2) == 0, 1.0, -1.0).float().unsqueeze(0)
        raw = torch.cos(math.pi * (idx + 0.5) * k_freq / float(in_dim)) * haar
        # Row-normalize so logits have unit variance
        w_rh = F.normalize(raw, p=2, dim=-1) * 1.5
        router.route_heads.weight.copy_(w_rh.to(router.route_heads.weight.dtype))
        if router.route_heads.bias is not None:
            nn.init.zeros_(router.route_heads.bias)

        if getattr(router, "triad_readout", None) is not None:
            M = int(router.d_shell)
            nn.init.normal_(router.triad_readout, mean=0.0, std=0.15 / math.sqrt(max(1, M)))


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SurgicalGemma4Attention(nn.Module):
    """
    Gemma 4 31B Global Full-Attention layer upgraded with V4 Triad Ultrametric Routing,
    p-RoPE Subspace Separation, Unified `K = V` (`attention_k_eq_v = True`) Handling,
    Query-Tail Beltrami Outlier Shielding, Measure-Preserving Donor Pruning / Block Bridge,
    and optional inline `DarkSnailGemma4GlobalHook`.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int = 5,
        head_dim_override: Optional[int] = None,
        num_heads_override: Optional[int] = None,
        num_kv_heads_override: Optional[int] = None,
        enable_global_hook: bool = False,
        prope_nope_routing: bool = True,
        prope_layout: Literal["contiguous_prefix", "hf_split_half"] = "hf_split_half",
        skip_proj_init: bool = False,
    ):
        super().__init__()
        top_mod = _get_topology_module()
        DynamicTopologyRouter = top_mod.DynamicTopologyRouter
        self._compute_expected_block_distance = top_mod.compute_expected_block_distance
        self._get_dynamic_ultrametric_mask = top_mod.get_dynamic_ultrametric_mask

        self.config = config
        self.root_config = None
        self.layer_idx = int(layer_idx)
        self.patched_layer_idx = int(layer_idx)
        self.is_sliding = False
        self.layer_type = "full_attention"
        self.is_kv_shared_layer = False
        self.kv_shared_layer_index: Optional[int] = None
        self.store_full_length_kv = False
        self.is_causal = True

        self.embed_dim = int(getattr(config, "hidden_size", 5376))
        self.num_heads = int(
            num_heads_override
            if num_heads_override is not None
            else getattr(config, "num_attention_heads", 32)
        )
        self.head_dim = int(
            head_dim_override
            if head_dim_override is not None
            else getattr(config, "global_head_dim", getattr(config, "head_dim", 512))
        )
        self.num_key_value_heads = int(
            num_kv_heads_override
            if num_kv_heads_override is not None
            else getattr(config, "num_global_key_value_heads", getattr(config, "num_key_value_heads", 4))
        )
        self.num_key_value_groups = max(1, self.num_heads // max(1, self.num_key_value_heads))

        self.k_eq_v = bool(getattr(config, "attention_k_eq_v", True))
        self.scale = float(getattr(config, "surgical_attn_scale", 1.0))
        self.p_rope_ratio = float(getattr(config, "partial_rotary_factor", 0.25))
        self.prope_layout = prope_layout
        self.prope_nope_routing = bool(prope_nope_routing)

        self.splitter = ProportionalRoPESubspaceSplitter(
            head_dim=self.head_dim,
            p_rope_ratio=self.p_rope_ratio,
            layout=self.prope_layout,
        )

        attn_bias = bool(getattr(config, "attention_bias", False))
        if skip_proj_init:
            # Zero-copy placeholder attributes populated from old_attn in patch_gemma4_model
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = None
            self.q_norm = None
            self.k_norm = None
            self.v_norm = None
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=attn_bias)
            self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=attn_bias)
            if not self.k_eq_v:
                self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=attn_bias)
            else:
                self.v_proj = None
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=attn_bias)

            eps = float(getattr(config, "rms_norm_eps", 1e-6))
            self.q_norm = Gemma4RMSNorm(self.head_dim, eps=eps)
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=eps)
            self.v_norm = Gemma4RMSNormNoScale(self.head_dim, eps=eps)

        self.p = int(getattr(config, "surgical_p", 2))
        self.qk_mode = bool(getattr(config, "surgical_qk_mode", True))
        self.prime_arities = getattr(config, "surgical_prime_arities", [2, 3])
        self.swin_shift = bool(getattr(config, "surgical_swin_shift", True))
        self.route_stride = int(getattr(config, "surgical_route_stride", 2))
        self.distill_kl = bool(getattr(config, "surgical_distill_kl", True))
        self.target_density = float(getattr(config, "surgical_target_density", 0.25))
        self.triad_mode = bool(getattr(config, "surgical_triad_mode", True))
        self.d_shell = int(getattr(config, "surgical_d_shell", 8))

        init_mode = str(getattr(config, "surgical_init_mode", "haar_dct"))
        base_init_mode = "collapse" if init_mode in ("collapse", "haar_dct") else init_mode
        router_head_dim = self.splitter.d_nope if (self.qk_mode and self.prope_nope_routing) else self.head_dim
        max_pos = int(getattr(config, "max_position_embeddings", 262144))
        self.router = DynamicTopologyRouter(
            embed_dim=self.embed_dim,
            seq_len=min(max_pos, 65536),
            num_heads=self.num_heads,
            p=self.p,
            init_mode=base_init_mode,
            levels=getattr(config, "surgical_tree_depth", 5),
            tree_mode=getattr(config, "surgical_tree_mode", False),
            qk_mode=self.qk_mode,
            head_dim=router_head_dim,
            prime_arities=self.prime_arities,
            triad_mode=self.triad_mode,
            d_shell=self.d_shell,
        )
        if init_mode == "haar_dct":
            _init_router_haar_dct_spectral_hash(self.router, seed=self.layer_idx)
        self.p = self.router.p

        if enable_global_hook:
            self.global_hook: Optional[DarkSnailGemma4GlobalHook] = DarkSnailGemma4GlobalHook(
                head_dim=self.head_dim,
                d_model=self.embed_dim,
                bridge_dim=int(getattr(config, "surgical_bridge_dim", 64)),
                p_rope_ratio=self.p_rope_ratio,
                prope_layout=self.prope_layout,
            )
        else:
            self.global_hook = None

        self.register_buffer("density_lambda", torch.tensor(5.0, dtype=torch.float32), persistent=False)
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._route_cache_ref: Optional[Dict[Any, Any]] = None
        self._cached_assignments: Optional[torch.Tensor] = None
        self._cached_block_assignments: Optional[torch.Tensor] = None
        self._cached_q_blk: Optional[torch.Tensor] = None
        self._cached_k_blk: Optional[torch.Tensor] = None
        self._darksnail_shield_tokens: Optional[torch.Tensor] = None
        self._darksnail_token_salience: Optional[torch.Tensor] = None
        self.last_mean_allowed_keys: Optional[float] = None
        self.last_mean_causal_keys: Optional[float] = None
        self.last_block_density: Optional[float] = None
        self.current_penalty: Optional[torch.Tensor] = None
        self.current_distill_loss: Optional[torch.Tensor] = None

    @staticmethod
    def _pool_tensor_blocks(x: torch.Tensor, route_block: int) -> torch.Tensor:
        B, H, S, D = x.shape
        RB = max(1, int(route_block))
        if RB == 1 or S == 1:
            return x
        NB = -(-S // RB)
        pad_len = NB * RB - S
        x_pad = F.pad(x, (0, 0, 0, pad_len), value=0.0) if pad_len > 0 else x
        x_blk = x_pad.view(B, H, NB, RB, D)

        idx = torch.arange(NB * RB, device=x.device).view(1, 1, NB, RB, 1)
        valid_bool = idx < S
        valid_f = valid_bool.to(x.dtype)
        valid_counts = valid_f.sum(dim=3).clamp(min=1.0)

        mean_pool = (x_blk * valid_f).sum(dim=3) / valid_counts
        norms = torch.norm(x_blk.float(), p=2, dim=-1).masked_fill(~valid_bool.squeeze(-1), float("-inf"))
        peak_idx = norms.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(B, H, NB, 1, D)
        peak_pool = torch.gather(x_blk, dim=3, index=peak_idx).squeeze(3)
        return 0.5 * (mean_pool + peak_pool)

    def _extract_routing_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prope_nope_routing and self.router.in_dim == 2 * self.splitter.d_nope:
            _, q_nope = self.splitter.split(q)
            _, k_nope = self.splitter.split(k)
            return q_nope, k_nope
        return q, k

    def _compute_qk_routing(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        tau: float,
        past_key_value: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Audit Fix #4a: Full Prefill (`seq_len > 1`) AND Autoregressive Decode (`seq_len == 1, L > 1`)
        post-p-RoPE `k_nope` routing with cross-layer stride cache and tail-block updates.
        """
        q_r, k_r = self._extract_routing_qk(q, k)
        B, H, seq_len, D = q_r.shape
        L = k_r.shape[-2]
        cfg_rb = int(getattr(self.config, "surgical_route_block", 128))
        RB = max(1, min(cfg_rb, max(1, L // 8)) if L >= 8 else 1)
        stride = max(1, int(getattr(self.config, "surgical_route_stride", self.route_stride)))
        p_idx = self.patched_layer_idx

        if (
            stride > 1
            and p_idx is not None
            and (p_idx % stride != 0)
            and self._route_cache_ref is not None
        ):
            anchor_key = (H, D, (p_idx // stride) * stride)
            cached = self._route_cache_ref.get(anchor_key)
            if cached is not None and cached["L"] == L and cached["B"] == B:
                self._cached_block_assignments = cached["block_assignments"]
                self._cached_assignments = cached["assignments"]
                self._cached_q_blk = cached.get("q_blk")
                self._cached_k_blk = cached.get("k_blk")
                return cached["assignments"], cached["block_assignments"], cached["load_balance_loss"]

        swin_active = (
            bool(getattr(self.config, "surgical_swin_shift", self.swin_shift))
            and p_idx is not None
            and (p_idx % 2 == 1)
            and RB > 1
        )
        shift = (RB // 2) if swin_active else 0

        if past_key_value is None or seq_len > 1 or self._cached_block_assignments is None:
            k_curr = k_r[:, :, -seq_len:, :] if L >= seq_len else k_r
            q_in = torch.roll(q_r, shifts=shift, dims=-2) if (shift > 0 and seq_len > 1) else q_r
            k_in = torch.roll(k_curr, shifts=shift, dims=-2) if (shift > 0 and k_curr.shape[-2] > 1) else k_curr

            q_blk = self._pool_tensor_blocks(q_in, RB)
            k_blk = self._pool_tensor_blocks(k_in, RB)
            qk_desc = torch.cat([q_blk, k_blk], dim=-1)

            block_assignments, load_balance_loss = self.router(qk_desc, tau_override=tau)
            self._cached_block_assignments = block_assignments
            self._cached_q_blk = q_blk
            self._cached_k_blk = k_blk

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
            # Decode step: seq_len == 1 with cached history of total length L
            NB_total = -(-L // RB)
            tail_start = (NB_total - 1) * RB
            k_tail = k_r[:, :, tail_start:L, :]
            k_tail_blk = self._pool_tensor_blocks(k_tail, RB)[:, :, :1, :]
            q_tail_blk = q_r  # (B, H, 1, D)
            qk_tail_desc = torch.cat([q_tail_blk, k_tail_blk], dim=-1)

            tail_block_assign, load_balance_loss = self.router(qk_tail_desc, tau_override=tau)
            prev_nb = self._cached_block_assignments.shape[2]
            if NB_total > prev_nb:
                block_assignments = torch.cat([self._cached_block_assignments, tail_block_assign], dim=2)
            else:
                block_assignments = torch.cat([self._cached_block_assignments[:, :, :-1], tail_block_assign], dim=2)
            self._cached_block_assignments = block_assignments

            if self._cached_assignments is not None and self._cached_assignments.shape[2] == L - 1:
                assignments = torch.cat([self._cached_assignments, tail_block_assign], dim=2)
            else:
                assignments = block_assignments.repeat_interleave(RB, dim=2)[:, :, :L, :, :]
            self._cached_assignments = assignments

        if stride > 1 and p_idx is not None and (p_idx % stride == 0) and self._route_cache_ref is not None:
            self._route_cache_ref[(H, D, p_idx)] = {
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

        return assignments, block_assignments, load_balance_loss

    def _compute_block_kl_lagrangian_loss(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        assignments: torch.Tensor,
        block_assignments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes `O(NB^2 * D)` block-level KL divergence between teacher QK scores (`scale = 1.0`)
        and differentiable ultrametric cophenetic distances + Dual Lagrangian density penalty.
        """
        if block_assignments is not None and q.shape[2] == block_assignments.shape[2]:
            q_blk = q
            k_blk = k
        else:
            RB = max(1, int(getattr(self.config, "surgical_route_block", 128)))
            q_blk = self._pool_tensor_blocks(q, RB)
            k_blk = self._pool_tensor_blocks(k, RB)
            if block_assignments is None:
                block_assignments = assignments[:, :, : q_blk.shape[2]]

        NB = block_assignments.shape[2]
        levels = block_assignments.shape[-2]
        surgical_req_depth = getattr(self.config, "surgical_req_depth", None)
        eff_req_depth = (
            int(surgical_req_depth)
            if (surgical_req_depth is not None and int(surgical_req_depth) > 0)
            else max(levels // 2, 1)
        )
        max_dist = getattr(self.config, "surgical_max_dist", None)
        if max_dist is None:
            max_dist = max(levels - eff_req_depth, 1)
        tau = max(float(getattr(self.config, "surgical_tau", 1.0)), 1e-4)

        idx_b = torch.arange(NB, device=q.device)
        causal_blk = idx_b.unsqueeze(1) >= idx_b.unsqueeze(0)
        causal_mask_4d = causal_blk.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            teacher_scores = torch.matmul(q_blk.float(), k_blk.float().transpose(-2, -1)) * self.scale
            teacher_scores = teacher_scores.masked_fill(~causal_mask_4d, float("-inf"))
            teacher_probs = F.softmax(teacher_scores, dim=-1).nan_to_num(0.0)

        expected_block_dist = self._compute_expected_block_distance(block_assignments.float(), p=self.p)
        student_logits = (max_dist - expected_block_dist) / tau
        student_logits_masked = student_logits.masked_fill(~causal_mask_4d, float("-inf"))
        student_log_probs = F.log_softmax(student_logits_masked, dim=-1).masked_fill(~causal_mask_4d, 0.0)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") / max(
            1, block_assignments.shape[1] * NB
        )

        soft_block_mask = torch.sigmoid((max_dist - expected_block_dist) / 0.5)
        causal_f = causal_mask_4d.float()
        measured_density = (soft_block_mask * causal_f).sum(dim=(-2, -1)).mean() / causal_f.sum().clamp(min=1.0)
        target_density = float(getattr(self.config, "surgical_target_density", self.target_density))
        lambda_val = self.density_lambda.detach().clone().to(measured_density.dtype)
        lagrangian_penalty = lambda_val * (F.relu(measured_density - target_density) ** 2)

        if self.training or self.router.training:
            with torch.no_grad():
                lambda_lr = float(getattr(self.config, "surgical_lambda_lr", 0.05))
                self.density_lambda.add_(lambda_lr * (measured_density.detach() - target_density)).clamp_(0.0, 100.0)

        self.last_block_density = float(measured_density.detach().item())
        return kl_loss + lagrangian_penalty

    def _update_darksnail_shield(self, q: torch.Tensor, k: torch.Tensor) -> None:
        B, H, S, _ = k.shape
        local_window = int(getattr(self.config, "surgical_local_window", 16))
        shield_k = int(getattr(self.config, "surgical_darksnail_shield_k", 96))
        w_q = min(48, S)

        with torch.no_grad():
            _, k_nope = self.splitter.split(k)
            k_nf = k_nope.to(torch.float32)
            k_f = k.to(torch.float32)
            q_tail = q[:, :, -w_q:, :].to(torch.float32)

            dk = torch.zeros((B, H, S), dtype=torch.float32, device=k.device)
            if S > 1:
                dk[:, :, 1:] = torch.norm(k_nf[:, :, 1:, :] - k_nf[:, :, :-1, :], p=2, dim=-1)
            flux_norm = dk / (dk.mean(dim=-1, keepdim=True) + 1e-6)
            flux_shield = F.max_pool1d(
                flux_norm.reshape(B * H, 1, S), kernel_size=9, stride=1, padding=4
            ).reshape(B, H, S)

            res_scores = torch.matmul(q_tail, k_f.transpose(-2, -1)) * self.scale
            res_max = res_scores.max(dim=-2).values
            res_shield = F.max_pool1d(
                res_max.reshape(B * H, 1, S), kernel_size=13, stride=1, padding=6
            ).reshape(B, H, S)

            res_z = (res_shield - res_shield.mean(dim=-1, keepdim=True)) / (
                res_shield.std(dim=-1, keepdim=True) + 1e-6
            )
            flux_z = (flux_shield - flux_shield.mean(dim=-1, keepdim=True)) / (
                flux_shield.std(dim=-1, keepdim=True) + 1e-6
            )
            salience = torch.maximum(res_z, 0.85 * flux_z) + 0.35 * (res_z + flux_z)
            self._darksnail_token_salience = salience

            masked_sal = salience.clone()
            masked_sal[:, :, 0] = float("-inf")
            if local_window > 0 and S > local_window:
                masked_sal[:, :, max(1, S - local_window) :] = float("-inf")

            avail = max(1, S - max(1, local_window) - 1)
            actual_k = min(shield_k, max(1, avail // 4), avail, S)
            if actual_k > 0:
                _, top_idx = masked_sal.topk(actual_k, dim=-1)
                self._darksnail_shield_tokens = top_idx
            else:
                self._darksnail_shield_tokens = None

    def _block_list_eligible(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor], past_key_value: Any
    ) -> bool:
        if getattr(self.config, "surgical_attention_backend", "eager") != "block_list":
            return False
        if hidden_states.shape[1] <= 1:
            return False
        interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"
        if not interpret and (
            not getattr(hidden_states, "is_cuda", False)
            or hidden_states.dtype not in (getattr(torch, "float16", None), getattr(torch, "bfloat16", None))
        ):
            return False
        if getattr(self.config, "surgical_mask_override", None) is not None:
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

    def _apply_darksnail_block_bridge(self, bl: Any, RB: int, S: int, local_blocks: int) -> Any:
        from ultrametric.block_list import BlockLists

        salience = self._darksnail_token_salience
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
        blk_sal = sal_pad.view(B, H, NB, RB).max(dim=-1).values

        lists_cpu = lists.cpu()
        counts_cpu = counts.cpu()
        sal_cpu = blk_sal.cpu()

        for b_i in range(B):
            for h_i in range(H):
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
                    for d_qb in range(local_blocks + 2, NB - 2):
                        if donated >= need:
                            break
                        d_cnt = int(counts_cpu[b_i, h_i, d_qb].item())
                        d_int_end = d_qb - local_blocks
                        d_n_loc = d_qb - d_int_end + 1
                        d_n_int = d_cnt - 1 - d_n_loc
                        while d_n_int > 0 and donated < need and (cnt + 1) <= max_count:
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
                    lists_cpu[b_i, h_i, qb, 1 + n_int : cnt] = torch.arange(int_end, qb + 1, dtype=torch.int32)

        return BlockLists(lists_cpu.to(lists.device), counts_cpu.to(counts.device), max_count)

    def _block_list_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        assignments: torch.Tensor,
        r: int,
        block_assignments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from ultrametric.block_list import block_list_attention, build_block_lists

        RB = int(getattr(self.config, "surgical_route_block", 128))
        B, H, S, _ = q.shape
        NB = -(-S // RB)

        if block_assignments is not None and block_assignments.shape[2] == NB:
            block_route = block_assignments[:, :, :NB, :r, :].argmax(dim=-1).to(torch.int32)
        else:
            idx = assignments[:, :, :S, :r, :].argmax(dim=-1)
            votes = F.one_hot(idx, self.p).to(torch.float32)
            if NB * RB != S:
                votes = F.pad(votes, (0, 0, 0, 0, 0, NB * RB - S))
            block_route = votes.view(B, H, NB, RB, r, self.p).sum(3).argmax(-1).to(torch.int32)

        local_window = int(getattr(self.config, "surgical_local_window", 16))
        local_blocks = -(-local_window // RB) if local_window > 0 else 0

        bl_kwargs: Dict[str, Any] = dict(
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
            and getattr(self.config, "surgical_mask_override", None) is None
        ):
            bl = self._apply_darksnail_block_bridge(bl, RB=RB, S=S, local_blocks=local_blocks)

        if getattr(self.config, "surgical_collect_stats", False):
            lens = torch.full((NB,), RB, dtype=torch.float32, device=q.device)
            lens[-1] = S - (NB - 1) * RB
            c = bl.counts.to(torch.float32)
            allowed = ((c - 1) * RB * lens + lens * (lens + 1) / 2).sum(-1)
            self.last_mean_allowed_keys = float((allowed.mean() / S).item())
            self.last_mean_causal_keys = float((S + 1) / 2.0)
            self.last_block_density = float((c.sum() / (B * H * NB * (NB + 1) / 2)).item())

        return block_list_attention(
            q, k, v, bl, route_block=RB, is_causal=True, sm_scale=self.scale, soft_block_bias=None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if getattr(self, "root_config", None) is not None and self.root_config is not self.config:
            for _k, _v in self.root_config.__dict__.items():
                if _k.startswith("surgical_") or _k in ("preserve_sinks",):
                    setattr(self.config, _k, _v)

        batch_size, seq_len, _ = hidden_states.size()
        past_key_value = kwargs.get("past_key_value", past_key_values)
        shared_kv_states = kwargs.get("shared_kv_states", None)
        position_ids = kwargs.get("position_ids", None)

        req_depth_cfg = getattr(self.config, "surgical_req_depth", None)
        mask_override = getattr(self.config, "surgical_mask_override", None)
        darksnail_mode = bool(getattr(self.config, "surgical_darksnail_mode", False))

        # Audit Fix #5: Only run inline 64-D Callosal Bridge hook when explicitly enabled AND in darksnail_mode
        run_inline_hook = (
            bool(getattr(self.config, "surgical_enable_global_hook", False))
            and darksnail_mode
            and (req_depth_cfg is None or req_depth_cfg > 0)
            and mask_override is None
        )

        # Estimate past_len for fallback 2D RoPE cache during decode
        past_len = 0
        if past_key_value is not None:
            if hasattr(past_key_value, "get_seq_length"):
                try:
                    past_len = int(past_key_value.get_seq_length(self.layer_idx))
                except TypeError:
                    past_len = int(past_key_value.get_seq_length())
            elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                past_len = int(past_key_value[0].shape[-2])

        # --- Step 1: Gemma 4 Q, K, V Projections & Per-Head RMSNorms ---
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)

        # Audit Fix #6: Support both HuggingFace 5.5 `past_key_value.shared_layers[kv_shared_layer_index]`
        # and dict-based `shared_kv_states[layer_type]`
        has_hf_shared_kv = (
            bool(self.is_kv_shared_layer)
            and past_key_value is not None
            and hasattr(past_key_value, "shared_layers")
            and self.kv_shared_layer_index in getattr(past_key_value, "shared_layers", {})
        )
        has_dict_shared_kv = (
            bool(self.is_kv_shared_layer)
            and shared_kv_states is not None
            and self.layer_type in shared_kv_states
        )
        is_kv_shared = has_hf_shared_kv or has_dict_shared_kv

        if not is_kv_shared:
            k_raw = self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_key_value_heads, self.head_dim
            )
            if self.v_proj is not None:
                v_raw = self.v_proj(hidden_states).view(
                    batch_size, seq_len, self.num_key_value_heads, self.head_dim
                )
            else:
                v_raw = k_raw

            k = self.k_norm(k_raw) if self.k_norm is not None else k_raw
            v = self.v_norm(v_raw) if self.v_norm is not None else v_raw

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            q = q.transpose(1, 2)

        # --- Step 2: Gemma 4 Proportional RoPE (25% rotated, 75% unrotated) ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = apply_gemma4_prope(q, cos, sin, layout=self.prope_layout)
            if not is_kv_shared:
                k = apply_gemma4_prope(k, cos, sin, layout=self.prope_layout)
        else:
            needed_len = past_len + seq_len
            if self._rope_cache is None or self._rope_cache[0].shape[0] < needed_len:
                self._rope_cache = build_gemma4_prope_cache(
                    max(needed_len, 2048),
                    head_dim=self.head_dim,
                    theta=1000000.0,
                    partial_rotary_factor=self.p_rope_ratio,
                    device=hidden_states.device,
                    layout=self.prope_layout,
                )
            cos, sin = self._rope_cache
            q = apply_gemma4_prope(
                q, cos, sin, layout=self.prope_layout, position_ids=position_ids, position_offset=past_len
            )
            if not is_kv_shared:
                k = apply_gemma4_prope(
                    k, cos, sin, layout=self.prope_layout, position_ids=position_ids, position_offset=past_len
                )

        # --- Step 3: Optional Inline 64-D Callosal Bridge Residual Conditioning ---
        delta_h_bridge: Optional[torch.Tensor] = None
        if not is_kv_shared and self.global_hook is not None and run_inline_hook:
            _, _, h_bridged, _ = self.global_hook.forward_global_kv_hook(
                k_global=k,
                hidden_states=hidden_states,
                v_global=v,
                q_global=q,
                prope_layout=self.prope_layout,
            )
            delta_h_bridge = h_bridged - hidden_states

        # --- Step 4: KV Cache & Gemma 4 Shared-KV State Propagation ---
        if has_hf_shared_kv:
            k, v = past_key_value.shared_layers[self.kv_shared_layer_index]
            k = k.to(q.device)
            v = v.to(q.device)
        elif has_dict_shared_kv:
            k, v = shared_kv_states[self.layer_type]
            k = k.to(q.device)
            v = v.to(q.device)
        else:
            if past_key_value is not None:
                if hasattr(past_key_value, "update"):
                    try:
                        k, v = past_key_value.update(k, v, self.layer_idx)
                    except TypeError:
                        k, v = past_key_value.update(k, v, self.layer_idx, {})
                elif isinstance(past_key_value, tuple):
                    k = torch.cat([past_key_value[0], k], dim=-2)
                    v = torch.cat([past_key_value[1], v], dim=-2)
                if self.store_full_length_kv:
                    if not hasattr(past_key_value, "shared_layers"):
                        try:
                            past_key_value.shared_layers = {}
                        except Exception:
                            pass
                    if hasattr(past_key_value, "shared_layers"):
                        past_key_value.shared_layers[self.layer_idx] = (k, v)
            if self.store_full_length_kv and shared_kv_states is not None:
                shared_kv_states[self.layer_type] = (k, v)

        # Broadcast 4 KV heads -> 32 Query heads (GQA group factor = 8)
        k_exp = _repeat_kv(k, self.num_key_value_groups)
        v_exp = _repeat_kv(v, self.num_key_value_groups)

        # --- Step 5: Post-p-RoPE `k_nope` Ultrametric Routing & DarkSnail Outlier Shield ---
        use_block_list = self._block_list_eligible(hidden_states, attention_mask, past_key_value)
        tau = float(getattr(self.config, "surgical_tau", 1.0))

        if use_block_list and req_depth_cfg == 0:
            self.current_penalty = hidden_states.new_zeros(()) if hasattr(hidden_states, "new_zeros") else torch.zeros(1)
            self._cached_assignments = None
            self._cached_block_assignments = None
            assignments = None
            block_assignments = None
        else:
            assignments, block_assignments, load_balance_loss = self._compute_qk_routing(
                q, k_exp, tau=tau, past_key_value=past_key_value
            )
            self.current_penalty = load_balance_loss

        if darksnail_mode and seq_len > 1 and mask_override is None:
            self._update_darksnail_shield(q, k_exp)

        if use_block_list:
            if req_depth_cfg == 0:
                if getattr(self.config, "surgical_collect_stats", False):
                    S = q.shape[-2]
                    self.last_mean_allowed_keys = float((S + 1) / 2.0)
                    self.last_mean_causal_keys = float((S + 1) / 2.0)
                    self.last_block_density = 1.0
                out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True, scale=self.scale)
            else:
                r = req_depth_cfg if req_depth_cfg is not None else assignments.shape[-2] // 2
                out = self._block_list_prefill(
                    q, k_exp, v_exp, assignments, min(r, assignments.shape[-2]), block_assignments=block_assignments
                )
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            proj_out = self.o_proj(out)
            if delta_h_bridge is not None:
                proj_out = proj_out + delta_h_bridge
            return proj_out, None

        # --- Step 6: Eager Attention Score Matrix & Ultrametric + Beltrami Masking ---
        scores = torch.matmul(q.to(torch.float32), k_exp.to(torch.float32).transpose(-2, -1)) * self.scale
        L_kv = k_exp.shape[-2]

        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask[:, :L_kv] == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, float("-inf"))
            elif attention_mask.dim() >= 4:
                mask_slice = attention_mask[..., :seq_len, :L_kv]
                if mask_slice.dtype == torch.bool:
                    scores = scores.masked_fill(~mask_slice, float("-inf"))
                else:
                    scores = scores + mask_slice
        if seq_len == L_kv and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, L_kv, dtype=torch.bool, device=hidden_states.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        elif seq_len > 1:
            idx_q = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
            idx_k = torch.arange(L_kv, device=hidden_states.device)
            causal_mask = idx_k.unsqueeze(0) > idx_q.unsqueeze(1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        local_window = int(getattr(self.config, "surgical_local_window", 16))
        levels = assignments.shape[-2] if assignments is not None else 5
        surgical_req_depth = req_depth_cfg if req_depth_cfg == 0 else getattr(self.config, "surgical_req_depth", None)

        if surgical_req_depth == 0 and mask_override is None:
            if getattr(self.config, "surgical_collect_stats", False) and seq_len > 1:
                self.last_mean_allowed_keys = float((L_kv + 1) / 2.0)
                self.last_mean_causal_keys = float((L_kv + 1) / 2.0)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).nan_to_num(0.0).to(v_exp.dtype)
            out = torch.matmul(attn_weights, v_exp)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            proj_out = self.o_proj(out)
            if delta_h_bridge is not None:
                proj_out = proj_out + delta_h_bridge
            return proj_out, attn_weights

        r = min(surgical_req_depth if surgical_req_depth is not None else (levels // 2), levels)
        indices = assignments.argmax(dim=-1)[..., :r]
        powers = self.p ** torch.arange(r, device=indices.device)
        branch_id = (indices * powers).sum(dim=-1)

        if seq_len == 1 and L_kv > 1:
            curr_branch = branch_id[:, :, -1:].unsqueeze(-1)     # (B, H, 1, 1)
            past_branches = branch_id[:, :, :L_kv].unsqueeze(-2)  # (B, H, 1, L_kv)
            um_mask_bool = curr_branch == past_branches
            if local_window > 0:
                um_mask_bool[..., :, max(0, L_kv - local_window) :] = True
        else:
            curr_b = branch_id[:, :, -seq_len:]
            past_b = branch_id[:, :, :L_kv]
            um_mask_bool = curr_b.unsqueeze(-1) == past_b.unsqueeze(-2)
            if local_window > 0:
                idx_q = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
                idx_k = torch.arange(L_kv, device=hidden_states.device)
                band = torch.abs(idx_q.unsqueeze(1) - idx_k.unsqueeze(0)) <= local_window
                um_mask_bool = um_mask_bool | band.unsqueeze(0).unsqueeze(0)

        # Always keep Token 0 (Attention Sink)
        um_mask_bool[..., :, 0] = True

        # Audit Fix #4b: Full DarkSnail Decode + Prefill Measure-Preserving Donor Pruning
        if darksnail_mode and mask_override is None:
            sh_tokens = self._darksnail_shield_tokens
            if seq_len == 1 and L_kv > 1:
                if sh_tokens is not None:
                    sh_idx = sh_tokens.unsqueeze(2).clamp(0, L_kv - 1)
                    um_mask_bool.scatter_(-1, sh_idx, True)
                decode_k = min(int(getattr(self.config, "surgical_darksnail_decode_k", 64)), L_kv)
                if decode_k > 0:
                    live_pool = F.max_pool1d(
                        scores.squeeze(2).reshape(batch_size * self.num_heads, 1, L_kv),
                        kernel_size=9,
                        stride=1,
                        padding=4,
                    ).reshape(batch_size, self.num_heads, 1, L_kv)
                    _, live_top = live_pool.topk(decode_k, dim=-1)
                    um_mask_bool.scatter_(-1, live_top, True)
            elif seq_len > 1 and sh_tokens is not None:
                w_q = min(16, max(1, seq_len // 8))
                idx_q_all = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
                idx_k_all = torch.arange(L_kv, device=hidden_states.device)
                causal_all = (idx_k_all.unsqueeze(0) <= idx_q_all.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
                before_causal = (um_mask_bool & causal_all).sum(dim=(-2, -1))

                # Snapshot tail slice before scatter in case donor rows cannot cover all additions
                tail_start = seq_len - w_q
                tail_backup = um_mask_bool[:, :, tail_start:, :].clone()
                sh_exp = sh_tokens.unsqueeze(2).expand(-1, -1, w_q, -1).clamp(0, L_kv - 1)
                um_mask_bool[:, :, tail_start:, :].scatter_(-1, sh_exp, True)

                after_causal = (um_mask_bool & causal_all).sum(dim=(-2, -1))
                added = (after_causal - before_causal).clamp(min=0)

                # Measure-preserving donor pruning on preceding filler query rows (batched on CPU to avoid CUDA syncs)
                donor_end = tail_start
                donor_start = max(1 + local_window, donor_end - 96)
                added_cpu = added.cpu()
                if int(added_cpu.max().item()) > 0:
                    win_slice = um_mask_bool[:, :, donor_start:seq_len, :].cpu()
                    tail_bak_cpu = tail_backup.cpu()
                    n_donor = max(0, donor_end - donor_start)
                    for b_i in range(batch_size):
                        for h_i in range(self.num_heads):
                            add_bh = int(added_cpu[b_i, h_i].item())
                            if add_bh <= 0:
                                continue
                            rem = add_bh + max(8, int(round(0.55 * float(add_bh))))
                            for d_rel in range(n_donor - 1, -1, -1):
                                if rem <= 0:
                                    break
                                r_d = donor_start + d_rel
                                k_lim = max(1, (L_kv - seq_len + r_d) - local_window)
                                if k_lim <= 1:
                                    continue
                                row_d = win_slice[b_i, h_i, d_rel, 1:k_lim]
                                true_pos = torch.nonzero(row_d, as_tuple=False).reshape(-1)
                                if true_pos.numel() > 0:
                                    take = min(rem, int(true_pos.numel()))
                                    win_slice[b_i, h_i, d_rel, 1 + true_pos[:take]] = False
                                    rem -= take
                            # If short sequence had insufficient donor capacity, prune excess from non-final tail rows
                            if rem > 0:
                                for r_t in range(max(1, w_q - 1)):
                                    if rem <= 0:
                                        break
                                    q_abs = L_kv - seq_len + tail_start + r_t
                                    k_lim_t = max(1, q_abs - local_window)
                                    if k_lim_t <= 1:
                                        continue
                                    row_t = win_slice[b_i, h_i, n_donor + r_t, 1:k_lim_t]
                                    true_pos_t = torch.nonzero(row_t, as_tuple=False).reshape(-1)
                                    if true_pos_t.numel() > 0:
                                        take_t = min(rem, int(true_pos_t.numel()))
                                        win_slice[b_i, h_i, n_donor + r_t, 1 + true_pos_t[:take_t]] = False
                                        rem -= take_t
                    um_mask_bool[:, :, donor_start:seq_len, :] = win_slice.to(um_mask_bool.device)

        # Audit Fix #4c: Support matched-budget `window@r` and `oracle@r` overrides
        if mask_override == "window":
            W = int(getattr(self.config, "surgical_window_size", 256) or 256)
            idx_q = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
            idx_k = torch.arange(L_kv, device=hidden_states.device)
            band = idx_k.unsqueeze(0) > (idx_q.unsqueeze(1) - W)
            um_mask_bool = band.unsqueeze(0).unsqueeze(0).expand(*um_mask_bool.shape).clone()
            um_mask_bool[..., :, 0] = True
        elif mask_override == "oracle":
            W = int(getattr(self.config, "surgical_window_size", 256) or 256)
            recent_w = min(128, W)
            top_k = max(0, W - recent_w)
            idx_q = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
            idx_k = torch.arange(L_kv, device=hidden_states.device)
            band = (idx_k.unsqueeze(0) > (idx_q.unsqueeze(1) - recent_w)).unsqueeze(0).unsqueeze(0)
            um_mask_bool = band.expand(*um_mask_bool.shape).clone()
            if top_k > 0:
                oracle_scores = scores.clone().masked_fill_(band, float("-inf"))
                if L_kv > seq_len:
                    causal = (idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
                    oracle_scores.masked_fill_(~causal, float("-inf"))
                _, topk_indices = oracle_scores.topk(min(top_k, L_kv), dim=-1)
                um_mask_bool.scatter_(-1, topk_indices, True)
            um_mask_bool[..., :, 0] = True

        if getattr(self.config, "surgical_collect_stats", False) and seq_len > 1:
            idx_q = torch.arange(L_kv - seq_len, L_kv, device=hidden_states.device)
            idx_k = torch.arange(L_kv, device=hidden_states.device)
            causal = idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)
            allowed = (um_mask_bool & causal).sum(-1).float()
            self.last_mean_allowed_keys = float(allowed.mean().item())
            self.last_mean_causal_keys = float(causal.sum(-1).float().mean().item())

        sparse_scores = scores.masked_fill(~um_mask_bool, float("-inf"))
        is_all_neg_inf = (sparse_scores == float("-inf")).all(dim=-1, keepdim=True)
        sparse_scores.masked_fill_(is_all_neg_inf, 0.0)

        attn_weights = F.softmax(sparse_scores, dim=-1, dtype=torch.float32).nan_to_num(0.0).to(v_exp.dtype)
        out = torch.matmul(attn_weights, v_exp)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        proj_out = self.o_proj(out)
        if delta_h_bridge is not None:
            proj_out = proj_out + delta_h_bridge
        return proj_out, attn_weights


# ============================================================================
# 6. Model Patching, Layer-29 Beltrami Hook, KL Warmup & Sidecar I/O
# ============================================================================


def discover_gemma4_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Discovers the 60-layer decoder `ModuleList` across `Gemma4ForConditionalGeneration`,
    `Gemma4ForCausalLM`, `Gemma4TextModel`, or standalone PyTorch `Gemma4TextModel`.
    """
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        if hasattr(m, "layers"):
            return m.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Could not locate decoder `.layers` on model of type {type(model).__name__}")


def is_gemma4_global_layer(layer: nn.Module, layer_idx: int, total_layers: int = 60) -> bool:
    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        if hasattr(attn, "is_sliding"):
            return not bool(getattr(attn, "is_sliding"))
        l_type = getattr(attn, "layer_type", getattr(layer, "attention_type", None))
        if l_type is not None:
            return l_type == "full_attention" or l_type == "full"
    return Gemma4_31B_Spec.is_global_layer(layer_idx, num_layers=total_layers)


def patch_gemma4_model(
    model: nn.Module,
    tree_depth: int = 5,
    arity: int = 2,
    prime_arities: Sequence[int] = (2, 3),
    swin_shift: bool = True,
    route_stride: int = 2,
    d_shell: int = 8,
    init_mode: str = "haar_dct",
    enable_global_hook: bool = False,
    prope_nope_routing: bool = True,
    prope_layout: Literal["contiguous_prefix", "hf_split_half"] = "hf_split_half",
    only_global_layers: bool = True,
    **kwargs: Any,
) -> nn.Module:
    text_config = getattr(model.config, "text_config", model.config)
    configs = [model.config] if text_config is model.config else [model.config, text_config]

    for cfg in configs:
        setattr(cfg, "surgical_p", max(prime_arities) if prime_arities else arity)
        setattr(cfg, "surgical_tree_depth", tree_depth)
        setattr(cfg, "surgical_init_mode", init_mode)
        setattr(cfg, "surgical_qk_mode", True)
        setattr(cfg, "surgical_prime_arities", list(prime_arities) if prime_arities else None)
        setattr(cfg, "surgical_swin_shift", swin_shift)
        setattr(cfg, "surgical_route_stride", route_stride)
        setattr(cfg, "surgical_triad_mode", True)
        setattr(cfg, "surgical_d_shell", d_shell)
        setattr(cfg, "surgical_enable_global_hook", enable_global_hook)
        for k_extra, v_extra in kwargs.items():
            setattr(cfg, f"surgical_{k_extra}", v_extra)

    model._surgical_route_cache = {}
    layers = discover_gemma4_decoder_layers(model)
    total_layers = len(layers)
    patched_count = 0

    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue
        if only_global_layers and not is_gemma4_global_layer(layer, i, total_layers=total_layers):
            continue

        old_attn = layer.self_attn
        head_dim = int(getattr(old_attn, "head_dim", getattr(text_config, "global_head_dim", 512)))

        # Audit Fix #5: Read `in_features` / `out_features` before `.weight.shape` so 4-bit
        # `bitsandbytes` (`Params4bit` where `.weight.shape[1] == 1`) never miscomputes head counts!
        o_in = getattr(old_attn.o_proj, "in_features", None)
        if o_in is None:
            o_in = old_attn.o_proj.weight.shape[1]
        num_heads = int(
            getattr(
                old_attn,
                "num_heads",
                getattr(old_attn, "num_q_heads", getattr(text_config, "num_attention_heads", o_in // max(1, head_dim))),
            )
        )

        if hasattr(old_attn, "k_proj") and old_attn.k_proj is not None:
            k_out = getattr(old_attn.k_proj, "out_features", None)
            if k_out is None:
                k_out = old_attn.k_proj.weight.shape[0]
            num_kv_heads = int(k_out // max(1, head_dim))
        else:
            num_kv_heads = int(
                getattr(
                    old_attn,
                    "num_key_value_heads",
                    getattr(old_attn, "num_kv_heads", getattr(text_config, "num_global_key_value_heads", 4)),
                )
            )

        new_attn = SurgicalGemma4Attention(
            text_config,
            layer_idx=i,
            head_dim_override=head_dim,
            num_heads_override=num_heads,
            num_kv_heads_override=num_kv_heads,
            enable_global_hook=enable_global_hook,
            prope_nope_routing=prope_nope_routing,
            prope_layout=prope_layout,
            skip_proj_init=True,
        )
        new_attn.root_config = model.config
        new_attn.patched_layer_idx = patched_count
        new_attn._route_cache_ref = model._surgical_route_cache
        new_attn.is_kv_shared_layer = bool(getattr(old_attn, "is_kv_shared_layer", False))
        new_attn.kv_shared_layer_index = getattr(old_attn, "kv_shared_layer_index", None)
        new_attn.store_full_length_kv = bool(getattr(old_attn, "store_full_length_kv", False))
        new_attn.k_eq_v = getattr(old_attn, "v_proj", None) is None

        # Zero-copy submodule transfer (compatible with bfloat16, float16, and 4-bit bitsandbytes)
        new_attn.q_proj = old_attn.q_proj
        new_attn.k_proj = getattr(old_attn, "k_proj", None)
        new_attn.v_proj = getattr(old_attn, "v_proj", None)
        new_attn.o_proj = old_attn.o_proj
        new_attn.q_norm = getattr(old_attn, "q_norm", None)
        new_attn.k_norm = getattr(old_attn, "k_norm", None)
        new_attn.v_norm = getattr(old_attn, "v_norm", None)

        dev = old_attn.q_proj.weight.device
        new_attn.router.to(device=dev).float()
        new_attn.density_lambda = new_attn.density_lambda.to(device=dev)
        if new_attn.global_hook is not None:
            new_attn.global_hook.to(device=dev).float()

        layer.self_attn = new_attn
        patched_count += 1

    return model


def attach_gemma4_beltrami_hook(
    model: nn.Module,
    layer_idx: int = 29,
    bridge_to_callosal_4096: bool = True,
) -> Gemma4BeltramiCallosalShell:
    layers = discover_gemma4_decoder_layers(model)
    total_layers = len(layers)
    global_indices = [
        i for i in range(total_layers) if is_gemma4_global_layer(layers[i], i, total_layers=total_layers)
    ]
    target_idx = Gemma4_31B_Spec.snap_to_nearest_global_layer(layer_idx, global_indices=global_indices)
    text_cfg = getattr(model.config, "text_config", model.config)
    d_model = int(getattr(text_cfg, "hidden_size", 5376))

    if hasattr(model, "_darksnail_gemma4_hook_handle") and model._darksnail_gemma4_hook_handle is not None:
        model._darksnail_gemma4_hook_handle.remove()

    dev = next(model.parameters()).device
    shell = Gemma4BeltramiCallosalShell(
        d_model=d_model,
        callosal_cond_dim=4096,
        target_tokens=32,
        outlier_budget=16,
        condensed_budget=16,
        d_shell=16,
        shield_radius=2,
        hook_layer=target_idx,
        bridge_to_callosal_cond=bridge_to_callosal_4096,
    ).to(dev)
    shell.eval()

    model._darksnail_beltrami_shell = shell
    model._darksnail_layer29_telemetry = {"zeno_flux": 0.0, "lca_variance": 0.0, "hook_layer": target_idx}
    model._darksnail_layer16_telemetry = model._darksnail_layer29_telemetry

    def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > 1:
            with torch.no_grad():
                h_win = h[:, -min(256, h.shape[1]) :, :].detach()
                super_tokens, telemetry = shell(h_win, return_diagnostics=True)
                z_val = telemetry.get("zeno_flux", 0.0)
                l_val = telemetry.get("lca_variance", 0.0)
                info = {
                    "zeno_flux": float(z_val.mean().item() if isinstance(z_val, torch.Tensor) else z_val),
                    "lca_variance": float(l_val.mean().item() if isinstance(l_val, torch.Tensor) else l_val),
                    "hook_layer": target_idx,
                    "callosal_shell_shape": tuple(super_tokens.shape),
                }
                model._darksnail_layer29_telemetry = info
                model._darksnail_layer16_telemetry = info

    model._darksnail_gemma4_hook_handle = layers[target_idx].register_forward_hook(_hook)
    return shell


def save_gemma4_router_sidecar(model: nn.Module, path: Union[str, Path]) -> Dict[str, Any]:
    """Saves the 10 Global `SurgicalGemma4Attention` routers to a `.safetensors` sidecar."""
    path_str = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
    layers = discover_gemma4_decoder_layers(model)
    tensors: Dict[str, torch.Tensor] = {}
    patched_indices: List[int] = []

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "router"):
            continue
        patched_indices.append(i)
        for k, v in attn.router.state_dict().items():
            tensors[f"layers.{i}.self_attn.router.{k}"] = v.detach().contiguous().cpu()
        if hasattr(attn, "density_lambda") and isinstance(attn.density_lambda, torch.Tensor):
            tensors[f"layers.{i}.self_attn.density_lambda"] = attn.density_lambda.detach().contiguous().cpu()

    meta = {
        "format": "sparse-ai-router-sidecar-v4-gemma4",
        "model_type": "gemma4_text",
        "patched_layers": json.dumps(patched_indices),
    }
    try:
        from safetensors.torch import save_file

        save_file(tensors, path_str, metadata=meta)
    except Exception:
        torch.save({"tensors": tensors, "metadata": meta}, path_str)
    return meta


def load_gemma4_router_sidecar(model: nn.Module, path: Union[str, Path], strict: bool = False) -> int:
    """Loads pre-warmed router weights into the 10 Global `SurgicalGemma4Attention` layers."""
    path_str = str(path)
    try:
        from safetensors.torch import load_file

        tensors = load_file(path_str, device="cpu")
    except Exception:
        ckpt = torch.load(path_str, map_location="cpu", weights_only=False)
        tensors = ckpt.get("tensors", ckpt)

    layers = discover_gemma4_decoder_layers(model)
    loaded = 0
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "router"):
            continue
        prefix = f"layers.{i}.self_attn.router."
        sd = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
        if sd:
            try:
                attn.router.load_state_dict(sd, strict=strict)
                loaded += 1
            except Exception:
                pass
    return loaded


def warmup_gemma4_routers_kl(
    model: nn.Module,
    tokenizer: Any = None,
    calibration_texts: Optional[Sequence[str]] = None,
    *,
    calibration_input_ids: Optional[Sequence[torch.Tensor]] = None,
    steps: Optional[int] = None,
    num_steps: int = 40,
    lr: float = 2e-3,
    target_density: float = 0.25,
    max_seq_len: int = 2048,
    save_sidecar_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Self-supervised Block-KL + Dual Lagrangian warmup for the 10 Global `DynamicTopologyRouter`
    modules on `google/gemma-4-31B` (`O(NB^2 * D)` block distillation, zero backbone gradients).

    Supports both pre-tokenized `calibration_input_ids` (Colab Cell 7 contract) and raw
    `calibration_texts` + `tokenizer`, as well as both `steps` and `num_steps` kwargs.
    """
    total_steps = int(steps if steps is not None else num_steps)
    layers = discover_gemma4_decoder_layers(model)
    global_attns: List[SurgicalGemma4Attention] = [
        layer.self_attn for layer in layers if isinstance(getattr(layer, "self_attn", None), SurgicalGemma4Attention)
    ]
    if not global_attns:
        raise RuntimeError("No SurgicalGemma4Attention layers found on model.")

    tcfg = getattr(model.config, "text_config", model.config)
    configs = [model.config] if tcfg is model.config else [model.config, tcfg]
    prev_cfg_state = [
        (
            cfg,
            getattr(cfg, "surgical_req_depth", None),
            getattr(cfg, "surgical_route_stride", 2),
            getattr(cfg, "surgical_target_density", 0.25),
            getattr(cfg, "surgical_attention_backend", "eager"),
        )
        for cfg in configs
    ]
    for cfg in configs:
        cfg.surgical_req_depth = 0
        cfg.surgical_route_stride = 1
        cfg.surgical_target_density = float(target_density)
        cfg.surgical_attention_backend = "eager"
    for attn in global_attns:
        attn.target_density = float(target_density)
        attn.config.surgical_target_density = float(target_density)

    dev = next(model.parameters()).device

    # Build list of calibration input_ids tensors
    cal_batches: List[torch.Tensor] = []
    if calibration_input_ids is not None and len(calibration_input_ids) > 0:
        for item in calibration_input_ids:
            t_ids = item if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.long)
            if t_ids.dim() == 1:
                t_ids = t_ids.unsqueeze(0)
            cal_batches.append(t_ids[:, :max_seq_len].to(dev))
    else:
        if not calibration_texts:
            from darksnail.longbench import HAYSTACK_BASE

            calibration_texts = [HAYSTACK_BASE * 24]
        if tokenizer is None:
            raise ValueError("Either `calibration_input_ids` or `tokenizer` must be provided to warmup_gemma4_routers_kl.")
        for txt in calibration_texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_seq_len)
            t_ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "__getitem__") else enc
            cal_batches.append(t_ids.to(dev))

    # Cache exact dense teacher (q_blk, k_blk) per Global layer across calibration sequences (1 pass per unique sequence)
    cached_teacher_qk: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    with torch.no_grad():
        for ids in cal_batches:
            if hasattr(model, "_surgical_route_cache") and isinstance(model._surgical_route_cache, dict):
                model._surgical_route_cache.clear()
            try:
                model(input_ids=ids, use_cache=False)
            except TypeError:
                model(ids)
            sample_qk: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for attn in global_attns:
                if attn._cached_q_blk is not None and attn._cached_k_blk is not None:
                    sample_qk.append((attn._cached_q_blk.detach().clone(), attn._cached_k_blk.detach().clone()))
            if len(sample_qk) == len(global_attns):
                cached_teacher_qk.append(sample_qk)

    router_params: List[nn.Parameter] = []
    for attn in global_attns:
        attn.router.train()
        for p in attn.router.parameters():
            p.requires_grad_(True)
            router_params.append(p)

    optimizer = torch.optim.AdamW(router_params, lr=lr, weight_decay=0.01)

    losses: List[float] = []
    for step in range(total_steps):
        optimizer.zero_grad()
        tau = max(0.35, 1.5 - 1.15 * (step / max(1, total_steps - 1)))
        step_loss = torch.zeros((), dtype=torch.float32, device=dev)

        if cached_teacher_qk:
            sample_qk = cached_teacher_qk[step % len(cached_teacher_qk)]
            for attn, (q_b, k_b) in zip(global_attns, sample_qk):
                qk_desc = torch.cat([q_b, k_b], dim=-1)
                blk_assign, lb_loss = attn.router(qk_desc, tau_override=tau)
                kl_loss = attn._compute_block_kl_lagrangian_loss(q_b, k_b, blk_assign, block_assignments=blk_assign)
                step_loss = step_loss + kl_loss + 0.01 * lb_loss
        else:
            ids = cal_batches[step % len(cal_batches)]
            with torch.no_grad():
                try:
                    model(input_ids=ids, use_cache=False)
                except TypeError:
                    model(ids)
            for attn in global_attns:
                if attn._cached_q_blk is not None and attn._cached_k_blk is not None:
                    q_b = attn._cached_q_blk.detach()
                    k_b = attn._cached_k_blk.detach()
                    qk_desc = torch.cat([q_b, k_b], dim=-1)
                    blk_assign, lb_loss = attn.router(qk_desc, tau_override=tau)
                    kl_loss = attn._compute_block_kl_lagrangian_loss(q_b, k_b, blk_assign, block_assignments=blk_assign)
                    step_loss = step_loss + kl_loss + 0.01 * lb_loss

        if step_loss.requires_grad:
            step_loss.backward()
            optimizer.step()
            losses.append(float(step_loss.detach().item()))

    densities = [float(attn.last_block_density) for attn in global_attns if attn.last_block_density is not None]
    mean_density = sum(densities) / len(densities) if densities else float(target_density)

    for attn in global_attns:
        attn.router.eval()
        for p in attn.router.parameters():
            p.requires_grad_(False)

    for cfg, prev_depth, prev_stride, prev_td, prev_backend in prev_cfg_state:
        cfg.surgical_req_depth = prev_depth
        cfg.surgical_route_stride = prev_stride
        cfg.surgical_target_density = prev_td
        cfg.surgical_attention_backend = prev_backend

    if save_sidecar_path is not None:
        save_gemma4_router_sidecar(model, save_sidecar_path)

    return {
        "steps": total_steps,
        "num_steps": total_steps,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "final_mean_block_density": mean_density,
    }


def prepare_darksnail_gemma4_model(
    model: nn.Module,
    sidecar_path: Optional[Union[str, Path]] = None,
    attention_backend: str = "eager",
    enable_global_hook: bool = False,
    attach_callosal_shell_hook: bool = True,
    hook_layer: int = 29,
    router_init_mode: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Drop-in Gemma 4 31B equivalent of `prepare_darksnail_llama_model`:
      1. Surgically patches only the 10 Global Full-Attention layers (`[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`)
         with `init_mode="haar_dct"` (or `"collapse"` if a trained Gemma 4 sidecar exists).
      2. Loads optional `gemma4_31b_router_v4.safetensors` sidecar if present.
      3. Attaches the Global Layer-29 `Gemma4BeltramiCallosalShell` (`5376 -> 4096`) hook.
    """
    has_sidecar = sidecar_path is not None and Path(sidecar_path).exists()
    if router_init_mode is not None:
        init_mode = "collapse" if has_sidecar else str(router_init_mode)
    else:
        init_mode = "collapse" if has_sidecar else "haar_dct"

    layers = discover_gemma4_decoder_layers(model)
    already_patched = any(isinstance(getattr(l, "self_attn", None), SurgicalGemma4Attention) for l in layers)
    if not already_patched:
        model = patch_gemma4_model(
            model,
            tree_depth=5,
            arity=2,
            prime_arities=(2, 3),
            swin_shift=True,
            route_stride=2,
            d_shell=8,
            init_mode=init_mode,
            enable_global_hook=enable_global_hook,
            prope_nope_routing=True,
            only_global_layers=True,
            **kwargs,
        )

    if has_sidecar:
        loaded = load_gemma4_router_sidecar(model, str(sidecar_path), strict=False)
        if loaded == 0:
            # If the provided sidecar had incompatible shapes (e.g. Llama 128-D vs Gemma 384-D),
            # re-initialize with haar_dct spectral hyperplanes so routing is never stuck at collapse!
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if isinstance(attn, SurgicalGemma4Attention):
                    _init_router_haar_dct_spectral_hash(attn.router, seed=attn.layer_idx)

    tcfg = getattr(model.config, "text_config", model.config)
    for cfg in ([model.config] if tcfg is model.config else [model.config, tcfg]):
        cfg.use_cache = True
        cfg.surgical_attention_backend = attention_backend
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    if attach_callosal_shell_hook:
        attach_gemma4_beltrami_hook(model, layer_idx=hook_layer, bridge_to_callosal_4096=True)

    model.eval()
    return model


__all__ = [
    "GLOBAL_LAYER_INDICES",
    "SLIDING_LAYER_INDICES",
    "Gemma4_31B_Spec",
    "GEMMA4_31B_SPEC",
    "Gemma4RMSNorm",
    "Gemma4RMSNormNoScale",
    "build_gemma4_prope_cache",
    "apply_gemma4_prope",
    "ProportionalRoPESubspaceSplitter",
    "DarkSnailGemma4GlobalHook",
    "Gemma4BeltramiCallosalShell",
    "SurgicalGemma4Attention",
    "discover_gemma4_decoder_layers",
    "is_gemma4_global_layer",
    "patch_gemma4_model",
    "attach_gemma4_beltrami_hook",
    "save_gemma4_router_sidecar",
    "load_gemma4_router_sidecar",
    "warmup_gemma4_routers_kl",
    "prepare_darksnail_gemma4_model",
]
