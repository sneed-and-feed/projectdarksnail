# SPDX-License-Identifier: CC0-1.0
"""
darksnail.hooks: Unified Mechanistic Interpretability, Transcallosal Telemetry,
and Norm-Preserving Spherical Lie-Algebraic Steering Hooks for Project DarkSnail V2.

Exposes four composable, zero-weight-training PyTorch forward-hook primitives and a
context-managed `DarkSnailHookSession` across homogeneous (`Meta-Llama-3.1-8B-Instruct`,
`model.model.layers`) and heterogeneous (`google/gemma-4-31B-it`, `model.language_model.layers`)
transformer backbones:

1. `BeltramiShellProbeHook` (Pillar A):
   Non-destructive midpoint residual-stream telemetry (`layer_idx=16` on 32-layer Llama,
   Global `layer_idx=29` on 60-layer Gemma 4 31B), extracting Tier-1 Beltrami exterior-curl
   outliers (`1.0000` spike preservation), Tier-2 Adelic medoids, Zeno flux `Phi_LH`,
   cophenetic LCA variance, deterministic row-orthonormal Haar-QR isometric bridging
   (`P_iso @ P_iso.T = I_4096`, `seed=20260404`), and `detokenize_tier1` BPE subword welding.
2. `SphericalLieSteeringHook` (Pillar B):
   Norm-preserving `so(M)` activation steering via Lean-4-verified skew-symmetric bilinear
   triad operators (`OcclusionTriad.lean`) with two-pass Kahan/Parlett tangent projection,
   exact radial sphere retraction (`|<v, dv_perp>| <= 1e-8`, `|||h_steered||_2 - ||h||_2| <= 1e-6`),
   online rank-2 `so(M)` wedge generator construction, and Lie-bracket commutator curvature
   `kappa(A_1, A_2)`.
3. `AllButTheTopGramHook` (Pillar C):
   Zero-generation Proposition 5.1 anisotropic cone debiasing (per-prompt token mean removal +
   power-iteration principal component projection onto `S^{d-1}`), span-to-span Callosal Gram
   resonance `R_l(a_k)`, optional single-token digit-logit fusion `alpha_callosal in [0.15, 0.85]`,
   and One-Sided ReLU Callosal Margin `max(0, P_callosal(a_k) - 1/K)`.
4. `PRoPESubspaceHook` (Pillar D):
   Heterogeneous Proportional RoPE (`partial_rotary_factor = 0.25`) subspace interceptor for
   Gemma 4 Global layers (`[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`), isolating the unrotated
   75% `q_nope` / `k_nope` (`384`-D out of `512`-D) semantic subspace and computing
   translation-invariant Beltrami salience `S_Beltrami(t)` without mutating the `attention_k_eq_v=True`
   value stream.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gemma4 import (
    GLOBAL_LAYER_INDICES,
    DarkSnailGemma4GlobalHook,
    Gemma4_31B_Spec,
    ProportionalRoPESubspaceSplitter,
    is_gemma4_global_layer,
)
from .readout import (
    SkewSymmetricTriadOperator,
    _exact_orthogonality_ulp_polish,
    _exact_sphere_norm_ulp_polish,
)
from .shell import BeltramiCallosalShell


# ============================================================================
# 1. Backbone Layer Discovery & Midpoint Resolution
# ============================================================================


def discover_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Auto-resolves the decoder `nn.ModuleList` across:
      - Homogeneous CausalLM backbones (`model.model.layers` for Llama-3.1-8B, Mistral, Qwen),
      - Heterogeneous Multimodal / ConditionalGeneration backbones
        (`model.language_model.layers`, `model.model.language_model.layers`,
        `model.language_model.model.layers` for `google/gemma-4-31B-it`),
      - Standalone `model.layers`, `model.decoder.layers`, `model.transformer.h`, or `model.gpt_neox.layers`.
    """
    if isinstance(model, nn.ModuleList):
        return model

    if hasattr(model, "model"):
        m = getattr(model, "model")
        if hasattr(m, "language_model"):
            lm = getattr(m, "language_model")
            if hasattr(lm, "layers") and isinstance(lm.layers, (nn.ModuleList, list, tuple)):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        if hasattr(m, "layers") and isinstance(m.layers, (nn.ModuleList, list, tuple)):
            return m.layers
        if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
            return m.decoder.layers

    if hasattr(model, "language_model"):
        lm = getattr(model, "language_model")
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers") and isinstance(lm.layers, (nn.ModuleList, list, tuple)):
            return lm.layers

    if hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, list, tuple)):
        return model.layers

    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    raise AttributeError(
        f"Could not locate decoder `.layers` on model of type {type(model).__name__}."
    )


def resolve_default_hook_layer(
    model_or_layers: Union[nn.Module, Sequence[nn.Module]],
    layer_idx: Optional[int] = None,
    snap_gemma4_global: bool = True,
) -> int:
    """
    Resolves the canonical transcallosal hook layer index:
      - When `layer_idx is None`:
        * Returns `16` on $L = 32$ models (`Meta-Llama-3.1-8B-Instruct`),
        * Returns Global Layer `29` ($r^\\ast \\approx 0.722$) on $L = 60$ `google/gemma-4-31B-it` models,
        * Returns `L // 2` on general $L$-layer backbones (snapped to the nearest Global layer if
          the backbone has heterogeneous Gemma 4 Local/Global layers and `snap_gemma4_global=True`).
      - When `layer_idx` is explicitly provided:
        * Snaps `16 -> 29` on 60-layer Gemma 4 models if `snap_gemma4_global=True`, otherwise
          clamps to `[0, L - 1]`.
    """
    layers = (
        model_or_layers
        if isinstance(model_or_layers, (nn.ModuleList, list, tuple))
        else discover_decoder_layers(model_or_layers)
    )
    total_layers = len(layers)
    if total_layers == 0:
        raise ValueError("Decoder layer list is empty.")

    has_heterogeneous_global = False
    global_indices: List[int] = []
    if total_layers == 60:
        has_heterogeneous_global = True
        global_indices = [
            i for i in range(total_layers) if is_gemma4_global_layer(layers[i], i, total_layers=total_layers)
        ]
    else:
        for i, lyr in enumerate(layers):
            attn = getattr(lyr, "self_attn", None)
            if attn is not None and (hasattr(attn, "is_sliding") or hasattr(attn, "layer_type")):
                has_heterogeneous_global = True
                if is_gemma4_global_layer(lyr, i, total_layers=total_layers):
                    global_indices.append(i)

    if layer_idx is None:
        if total_layers == 60:
            raw_idx = 29
        elif total_layers == 32:
            raw_idx = 16
        else:
            raw_idx = total_layers // 2
        if snap_gemma4_global and has_heterogeneous_global and global_indices:
            return Gemma4_31B_Spec.snap_to_nearest_global_layer(raw_idx, global_indices=global_indices)
        return max(0, min(total_layers - 1, int(raw_idx)))

    idx = int(layer_idx)
    if idx < 0:
        idx = total_layers + idx
    if snap_gemma4_global and total_layers == 60 and idx == 16 and 29 in (global_indices or GLOBAL_LAYER_INDICES):
        return 29
    return max(0, min(total_layers - 1, idx))


def build_isometric_haar_qr_bridge(
    d_in: int,
    d_out: int = 4096,
    seed: int = 20260404,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Constructs a deterministic row-orthonormal Haar-QR bridge matrix
    $\\mathbf{P}_{\\mathrm{iso}} \\in \\mathbb{R}^{d_{\\mathrm{out}} \\times d_{\\mathrm{in}}}$
    satisfying $\\mathbf{P}_{\\mathrm{iso}} \\mathbf{P}_{\\mathrm{iso}}^\\top = \\mathbf{I}_{d_{\\mathrm{out}}}$
    (when $d_{\\mathrm{in}} \\ge d_{\\mathrm{out}}$) inside an isolated RNG context
    `with torch.random.fork_rng(devices=[]):` seeded with `seed = 20260404`.

    Uses a seeded signed Haar-pair / disjoint-support reduction followed by batched
    $64 \\times 64$ Householder QR block rotations so that even high-dimensional bridges
    ($5376 \\to 4096$) construct in $< 2$ ms on CPU while preserving exact IEEE-754
    `float32` row-orthonormality ($\\lVert \\mathbf{P}_{\\mathrm{iso}} \\mathbf{P}_{\\mathrm{iso}}^\\top - \\mathbf{I} \\rVert_\\infty \\le 10^{-6}$).
    """
    d_in = int(d_in)
    d_out = int(d_out)
    dev = torch.device(device) if device is not None else torch.device("cpu")

    if d_in == d_out:
        return torch.eye(d_out, dtype=torch.float32, device=dev)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if max(d_in, d_out) <= 256:
            g = torch.randn((max(d_in, d_out), min(d_in, d_out)), dtype=torch.float32)
            q, r = torch.linalg.qr(g, mode="reduced")
            diag_sign = torch.sgn(torch.diagonal(r))
            diag_sign = torch.where(diag_sign == 0.0, torch.ones_like(diag_sign), diag_sign)
            q = q * diag_sign.unsqueeze(0)
            p_iso = q.t().contiguous() if d_in >= d_out else q.contiguous()
            return p_iso.to(dev)

        if d_in > d_out:
            # Step 1: Seeded signed Haar reduction H in R^(d_out x d_in) with H @ H^T == I_{d_out}
            perm = torch.randperm(d_in, dtype=torch.long)
            signs = torch.where(
                torch.randint(0, 2, (d_in,), dtype=torch.long) == 0,
                torch.ones(d_in, dtype=torch.float32),
                -torch.ones(d_in, dtype=torch.float32),
            )
            h_mat = torch.zeros((d_out, d_in), dtype=torch.float32)
            # Partition d_in columns into d_out non-empty disjoint groups
            boundaries = [int((r * d_in) // d_out) for r in range(d_out + 1)]
            boundaries[-1] = d_in
            # Vectorized fast path when d_in <= 2 * d_out (e.g. 5376 -> 4096)
            if d_in <= 2 * d_out:
                rem = d_in - d_out
                direct = d_out - rem
                if direct > 0:
                    idx_d = torch.arange(direct, dtype=torch.long)
                    cols_d = perm[:direct]
                    h_mat[idx_d, cols_d] = signs[:direct]
                if rem > 0:
                    idx_r = torch.arange(rem, dtype=torch.long)
                    cols_r1 = perm[direct : direct + rem]
                    cols_r2 = perm[direct + rem : direct + 2 * rem]
                    inv_sqrt2 = 1.0 / math.sqrt(2.0)
                    h_mat[direct + idx_r, cols_r1] = signs[direct : direct + rem] * inv_sqrt2
                    h_mat[direct + idx_r, cols_r2] = signs[direct + rem : direct + 2 * rem] * inv_sqrt2
            else:
                for r_idx in range(d_out):
                    c0, c1 = boundaries[r_idx], max(boundaries[r_idx] + 1, boundaries[r_idx + 1])
                    cols = perm[c0:c1]
                    scale = 1.0 / math.sqrt(float(max(1, c1 - c0)))
                    h_mat[r_idx, cols] = signs[c0:c1] * scale

            # Step 2: Batched 64x64 Householder QR block rotations Q_blk in O(d_out)
            blk_sz = 64
            n_blks = d_out // blk_sz
            if n_blks > 0:
                g_blk = torch.randn((n_blks, blk_sz, blk_sz), dtype=torch.float32)
                q_blk, r_blk = torch.linalg.qr(g_blk)
                d_sgn = torch.sgn(torch.diagonal(r_blk, dim1=-2, dim2=-1))
                d_sgn = torch.where(d_sgn == 0.0, torch.ones_like(d_sgn), d_sgn)
                q_blk = q_blk * d_sgn.unsqueeze(-2)
                main_len = n_blks * blk_sz
                h_main = h_mat[:main_len].view(n_blks, blk_sz, d_in)
                h_rot = torch.bmm(q_blk, h_main).view(main_len, d_in)
                if main_len < d_out:
                    p_iso = torch.cat([h_rot, h_mat[main_len:]], dim=0).contiguous()
                else:
                    p_iso = h_rot.contiguous()
            else:
                p_iso = h_mat.contiguous()
            return p_iso.to(dev)
        else:
            # d_in < d_out: transpose of (d_in, d_out) row-orthonormal bridge -> column-orthonormal
            p_t = build_isometric_haar_qr_bridge(d_out, d_in, seed=seed, device="cpu")
            return p_t.t().contiguous().to(dev)


# ============================================================================
# 2. Pillar A: BeltramiShellProbeHook
# ============================================================================


class BeltramiShellProbeHook(nn.Module):
    """
    Pillar A: Non-Destructive Layer-16 / Layer-29 Transcallosal Telemetry & Spike Extraction Hook.

    Wraps `BeltramiCallosalShell` (`darksnail/shell.py`) and `Gemma4BeltramiCallosalShell`
    (`darksnail/gemma4.py`) as a zero-mutation `register_forward_hook` attachment:
      1. Defaults to midpoint `layer_idx = 16` on $L = 32$ models (`Meta-Llama-3.1-8B-Instruct`,
         $d = 4096$) and midpoint Global Layer `layer_idx = 29` ($r^\\ast \\approx 0.722$) on
         $L = 60$ `google/gemma-4-31B-it` ($d = 5376$), or `len(layers) // 2` on general models.
      2. Populates `hook.last_telemetry` with:
         - `tier1_outlier_indices`: Top $K_{\\mathrm{out}} = 16$ token positions in chronological order.
         - `tier2_medoid_indices`: Top $K_{\\mathrm{med}} = 16$ adelic medoid token indices.
         - `zeno_flux_phi_lh`: Scalar epistemic turbulence / Zeno flux $\\Phi_{\\mathrm{LH}} \\in [0, 1]$.
         - `lca_variance`: Scalar cophenetic LCA variance.
         - `cross_shell_curl` and `outlier_score`: Per-token salience tensors.
         - `condensed_shell`: Condensed `(B, 32, d_out)` tensor bridged via the deterministic
           row-orthonormal Haar-QR bridge $\\mathbf{P}_{\\mathrm{iso}} \\in \\mathbb{R}^{4096 \\times d}$
           ($\\mathbf{P}_{\\mathrm{iso}} \\mathbf{P}_{\\mathrm{iso}}^\\top = \\mathbf{I}_{4096}$, `seed = 20260404`).
      3. Provides `detokenize_tier1(tokenizer, input_ids=None)` to weld extracted Tier-1 subwords
         via `"".join(tokenizer.decode([int(tid)]) for tid in tier1_token_ids)` (`Operation Snail-Cada`).
    """

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        d_model: Optional[int] = None,
        isometric_target_dim: Optional[int] = 4096,
        target_tokens: int = 32,
        outlier_budget: int = 16,
        condensed_budget: int = 16,
        d_shell: int = 16,
        shield_radius: int = 2,
        viscous_nu: float = 0.05,
        isometric_seed: int = 20260404,
        snap_gemma4_global: bool = True,
        protect_bos: bool = False,
        input_ids: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if outlier_budget + condensed_budget != target_tokens:
            target_tokens = int(outlier_budget) + int(condensed_budget)
        self.layer_idx = layer_idx
        self.resolved_layer_idx: Optional[int] = layer_idx
        self.d_model = int(d_model) if d_model is not None else None
        self.isometric_target_dim = int(isometric_target_dim) if isometric_target_dim is not None else None
        self.target_tokens = int(target_tokens)
        self.outlier_budget = int(outlier_budget)
        self.condensed_budget = int(condensed_budget)
        self.d_shell = max(2, int(d_shell))
        self.shield_radius = max(0, int(shield_radius))
        self.viscous_nu = float(viscous_nu)
        self.isometric_seed = int(isometric_seed)
        self.snap_gemma4_global = bool(snap_gemma4_global)
        self.protect_bos = bool(protect_bos)

        self._bound_input_ids: Optional[torch.Tensor] = input_ids
        self._shell: Optional[BeltramiCallosalShell] = None
        self._bridge_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.register_buffer("_active_p_iso", None, persistent=False)

        if self.d_model is not None:
            self._ensure_shell(self.d_model)
            if self.isometric_target_dim is not None:
                self._active_p_iso = self.get_isometric_bridge(self.d_model, self.isometric_target_dim)

        self.last_telemetry: Dict[str, Any] = {}

    def bind_input_ids(self, input_ids: Optional[torch.Tensor]) -> "BeltramiShellProbeHook":
        """Binds prompt `input_ids` so the next forward pass automatically records `tier1_token_ids`."""
        self._bound_input_ids = input_ids
        return self

    @property
    def isometric_bridge(self) -> Optional[torch.Tensor]:
        """Returns the active row-orthonormal Haar-QR bridge $\\mathbf{P}_{\\mathrm{iso}}$."""
        if self._active_p_iso is not None:
            return self._active_p_iso
        if self.d_model is not None and self.isometric_target_dim is not None:
            self._active_p_iso = self.get_isometric_bridge(self.d_model, self.isometric_target_dim)
            return self._active_p_iso
        return None

    def get_isometric_bridge(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        target_d = int(d_out if d_out is not None else (self.isometric_target_dim or d_in))
        key = (int(d_in), target_d)
        if key not in self._bridge_cache:
            self._bridge_cache[key] = build_isometric_haar_qr_bridge(
                d_in=int(d_in),
                d_out=target_d,
                seed=self.isometric_seed,
                device="cpu",
            )
        p_iso = self._bridge_cache[key]
        if device is not None and p_iso.device != device:
            p_iso = p_iso.to(device=device)
        return p_iso

    def _ensure_shell(self, d_in: int, device: Optional[torch.device] = None) -> BeltramiCallosalShell:
        if self._shell is None or self._shell.d_model != int(d_in):
            self.d_model = int(d_in)
            self._shell = BeltramiCallosalShell(
                d_model=int(d_in),
                target_tokens=self.target_tokens,
                outlier_budget=self.outlier_budget,
                condensed_budget=self.condensed_budget,
                d_shell=self.d_shell,
                shield_radius=self.shield_radius,
                viscous_nu=self.viscous_nu,
                hook_layer=int(self.resolved_layer_idx if self.resolved_layer_idx is not None else 16),
            )
            self._shell.eval()
        if device is not None:
            self._shell = self._shell.to(device=device)
        return self._shell

    def extract_telemetry(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Runs non-destructive Beltrami Callosal Shell extraction on `hidden_states`
        of shape `(B, S, D)` or `(S, D)` and updates `self.last_telemetry`.
        """
        with torch.no_grad():
            h = hidden_states.detach()
            if h.dim() == 2:
                h = h.unsqueeze(0)
            if h.dim() != 3:
                raise ValueError(f"Expected 2D or 3D hidden_states, got shape {tuple(h.shape)}")

            x = h.float()
            B, S_orig, D = x.shape
            device = x.device
            shell = self._ensure_shell(D, device=device)

            if S_orig < self.target_tokens:
                x_proc = F.interpolate(
                    x.transpose(1, 2), size=self.target_tokens, mode="linear", align_corners=False
                ).transpose(1, 2)
            else:
                x_proc = x
            S = x_proc.shape[1]

            # Compute Beltrami triad flux & cross-shell exterior curl
            shielded_flux, raw_flux, cross_shell_curl = shell.compute_beltrami_triad_flux(x_proc)

            norm_x = F.normalize(x_proc, p=2, dim=-1)
            sim_mat = torch.bmm(norm_x, norm_x.transpose(1, 2))
            eye = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
            max_other_sim = sim_mat.masked_fill(eye, -1.0).max(dim=-1).values
            isolation_score = 1.0 - max_other_sim

            token_norm = torch.norm(x_proc, p=2, dim=-1)
            norm_ratio = token_norm / (token_norm.amax(dim=-1, keepdim=True) + 1e-6)
            gated_norm = norm_ratio * (0.15 + 0.85 * cross_shell_curl)

            # Combine shielded flux with local un-smeared curl + isolation + norm so exact spike centers peak sharply
            outlier_score = (
                2.0 * shielded_flux
                + 1.5 * raw_flux
                + 2.5 * cross_shell_curl
                + 2.0 * isolation_score
                + 1.5 * gated_norm
            )

            K_out = min(self.outlier_budget, S)
            C = self.condensed_budget
            rem_len = max(1, S - K_out)

            score_for_topk = outlier_score.clone()
            if self.protect_bos and S > 0:
                score_for_topk[:, 0] = score_for_topk.amax(dim=-1) + 1e6

            _, topk_ranked_idx = torch.topk(score_for_topk, k=K_out, dim=-1, largest=True, sorted=True)
            outlier_idx, _ = torch.sort(topk_ranked_idx, dim=-1)  # (B, K_out) chronological order

            idx_exp = outlier_idx.unsqueeze(-1).expand(B, K_out, D)
            tier1_outliers = torch.gather(x_proc, dim=1, index=idx_exp)  # (B, K_out, D)

            # Tier-2 Ultrametric Medoid Condensation over remaining tokens
            is_outlier = torch.zeros((B, S), dtype=torch.bool, device=device)
            is_outlier.scatter_(1, outlier_idx, True)
            all_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
            if S > K_out:
                rem_idx = all_idx[~is_outlier].view(B, S - K_out)
            else:
                rem_idx = all_idx
            rem_x = torch.gather(x_proc, dim=1, index=rem_idx.unsqueeze(-1).expand(B, rem_idx.shape[1], D))

            medoid_h, pooled_v, medoid_local_idx, pos_var, _ = shell.pool_tensor_blocks_1d(rem_x, C)
            medoid_global_idx = torch.gather(rem_idx, dim=1, index=medoid_local_idx)  # (B, C)
            gamma_c = torch.exp(-self.viscous_nu * pos_var).clamp(min=0.05, max=1.0)
            alpha = shell.medoid_blend
            tier2_raw = alpha * medoid_h + (1.0 - alpha) * pooled_v
            tier2_medoids = tier2_raw * gamma_c.unsqueeze(-1)

            native_shell = torch.cat([tier1_outliers, tier2_medoids], dim=1).contiguous().float()

            if self.isometric_target_dim is not None and D != self.isometric_target_dim:
                p_iso = self.get_isometric_bridge(D, self.isometric_target_dim, device=device)
                self._active_p_iso = p_iso
                condensed_shell = F.linear(native_shell, p_iso)
            else:
                if self.isometric_target_dim is not None and D == self.isometric_target_dim:
                    self._active_p_iso = self.get_isometric_bridge(D, D, device=device)
                condensed_shell = native_shell

            raw_mean_flux = float(shielded_flux.mean().item())
            raw_max_flux = float(shielded_flux.max().item())
            zeno_flux = float(math.tanh(0.65 * raw_mean_flux + 0.35 * raw_max_flux))

            norm_t2 = F.normalize(tier2_raw, p=2, dim=-1)
            blk_sim = torch.bmm(norm_t2, norm_t2.transpose(1, 2))
            lca_var = float(blk_sim.var(dim=(-2, -1)).mean().item() * 4.0)

            # Identify salient spike peaks via local NMS for covert subword-splicing inspection
            local_max = F.max_pool1d(
                outlier_score.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)[:, :S]
            s_mean = outlier_score.mean(dim=-1, keepdim=True)
            s_std = outlier_score.std(dim=-1, keepdim=True).clamp(min=1e-6)
            is_peak = (outlier_score >= local_max - 1e-6) & (outlier_score > s_mean + 1.35 * s_std)
            spike_peaks_per_batch: List[torch.Tensor] = []
            for b_i in range(B):
                p_idx = torch.nonzero(is_peak[b_i], as_tuple=False).reshape(-1)
                if p_idx.numel() == 0 or p_idx.numel() > K_out:
                    p_idx = outlier_idx[b_i]
                else:
                    # Intersect with tier1_outlier_indices so peaks are strictly a subset of Tier-1
                    t1_set = set(outlier_idx[b_i].tolist())
                    filtered = [int(ix.item()) for ix in p_idx if int(ix.item()) in t1_set]
                    p_idx = (
                        torch.tensor(filtered, dtype=torch.long, device=device)
                        if filtered
                        else outlier_idx[b_i]
                    )
                spike_peaks_per_batch.append(p_idx)

            active_ids = input_ids if input_ids is not None else self._bound_input_ids
            tier1_token_ids: Optional[torch.Tensor] = None
            spike_token_ids: Optional[List[int]] = None
            if active_ids is not None:
                ids_t = (
                    active_ids
                    if isinstance(active_ids, torch.Tensor)
                    else torch.tensor(active_ids, dtype=torch.long, device=device)
                ).to(device=device)
                if ids_t.dim() == 1:
                    ids_t = ids_t.unsqueeze(0)
                if ids_t.shape[1] >= S_orig:
                    clamped_t1 = outlier_idx.clamp(0, ids_t.shape[1] - 1)
                    tier1_token_ids = torch.gather(ids_t.expand(B, -1), dim=1, index=clamped_t1)
                    clamped_pk = spike_peaks_per_batch[0].clamp(0, ids_t.shape[1] - 1)
                    spike_token_ids = [int(ids_t[0, idx].item()) for idx in clamped_pk]

            self.last_telemetry = {
                "layer_idx": self.resolved_layer_idx,
                "tier1_outlier_indices": outlier_idx.detach(),
                "tier1_ranked_indices": topk_ranked_idx.detach(),
                "spike_peak_indices": spike_peaks_per_batch[0].detach(),
                "tier2_medoid_indices": medoid_global_idx.detach(),
                "zeno_flux_phi_lh": zeno_flux,
                "zeno_flux": zeno_flux,
                "lca_variance": lca_var,
                "cross_shell_curl": cross_shell_curl[:, :S_orig].detach(),
                "outlier_score": outlier_score[:, :S_orig].detach(),
                "shielded_flux": shielded_flux[:, :S_orig].detach(),
                "condensed_shell": condensed_shell.detach(),
                "native_condensed_shell": native_shell.detach(),
                "tier1_outliers_native": tier1_outliers.detach(),
                "tier1_token_ids": tier1_token_ids.detach() if tier1_token_ids is not None else None,
                "spike_token_ids": spike_token_ids,
            }
            return self.last_telemetry

    def compute_spike_preservation_cosine(
        self,
        spike_vector: torch.Tensor,
        use_isometric_space: bool = False,
        batch_idx: int = 0,
    ) -> float:
        """
        Computes the maximum cosine similarity between a ground-truth injected `spike_vector`
        and the extracted Tier-1 outlier slots (`1.0000` spike preservation).
        """
        if not self.last_telemetry:
            raise RuntimeError("No telemetry recorded yet; run a forward pass first.")
        sv = spike_vector.detach().float().reshape(1, -1)
        if use_isometric_space and self._active_p_iso is not None and sv.shape[-1] == self._active_p_iso.shape[1]:
            sv = F.linear(sv.to(self._active_p_iso.device), self._active_p_iso)
            slots = self.last_telemetry["condensed_shell"][batch_idx, : self.outlier_budget, :].float()
        else:
            slots = self.last_telemetry["native_condensed_shell"][batch_idx, : self.outlier_budget, :].float()
        sv_norm = F.normalize(sv.to(slots.device), p=2, dim=-1)
        slots_norm = F.normalize(slots, p=2, dim=-1)
        cos_sims = torch.matmul(slots_norm, sv_norm.t()).squeeze(-1)
        return float(cos_sims.max().item())

    def detokenize_tier1(
        self,
        tokenizer: Any,
        input_ids: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        batch_idx: int = 0,
        top_k: Optional[int] = None,
        spikes_only: bool = False,
    ) -> str:
        """
        Welds extracted Tier-1 outlier token IDs into a string via:
            `"".join(tokenizer.decode([int(tid)]) for tid in tier1_token_ids)`
        to support covert BPE subword-splicing inspection (`Operation Snail-Cada`).
        """
        if not self.last_telemetry:
            raise RuntimeError("No telemetry captured in `last_telemetry`. Run a forward pass first.")

        t1_indices = self.last_telemetry["tier1_outlier_indices"][batch_idx]
        if top_k is not None:
            ranked = self.last_telemetry["tier1_ranked_indices"][batch_idx][: int(top_k)]
            selected_indices, _ = torch.sort(ranked)
        elif spikes_only and "spike_peak_indices" in self.last_telemetry:
            selected_indices = self.last_telemetry["spike_peak_indices"]
        else:
            selected_indices = t1_indices

        active_ids = input_ids if input_ids is not None else self._bound_input_ids
        if active_ids is not None:
            if isinstance(active_ids, torch.Tensor):
                ids_1d = active_ids[batch_idx].tolist() if active_ids.dim() >= 2 else active_ids.tolist()
            elif isinstance(active_ids, (list, tuple)) and len(active_ids) > 0 and isinstance(active_ids[0], (list, tuple)):
                ids_1d = list(active_ids[batch_idx])
            else:
                ids_1d = list(active_ids)

            # If caller passed already-extracted tier1_token_ids of length == len(selected_indices)
            if len(ids_1d) == int(selected_indices.numel()) and (
                int(selected_indices.max().item()) >= len(ids_1d)
            ):
                tier1_token_ids = [int(t) for t in ids_1d]
            else:
                tier1_token_ids = [
                    int(ids_1d[min(int(idx.item()), len(ids_1d) - 1)]) for idx in selected_indices
                ]
        elif self.last_telemetry.get("tier1_token_ids") is not None:
            t_ids = self.last_telemetry["tier1_token_ids"][batch_idx]
            if top_k is not None and t_ids.numel() >= int(top_k):
                # Map ranked subset in chronological order
                pos_map = {
                    int(pos.item()): int(tid.item())
                    for pos, tid in zip(t1_indices, t_ids)
                }
                tier1_token_ids = [pos_map[int(idx.item())] for idx in selected_indices if int(idx.item()) in pos_map]
            else:
                tier1_token_ids = [int(tid.item()) for tid in t_ids]
        else:
            raise ValueError(
                "Must provide `input_ids` to `detokenize_tier1` or call `hook.bind_input_ids(input_ids)`."
            )

        return "".join(tokenizer.decode([int(tid)]) for tid in tier1_token_ids)

    def __call__(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor) and h.dim() >= 2:
            self.extract_telemetry(h)
        # Non-destructive probe: return None so PyTorch leaves `output` unmodified
        return None


# ============================================================================
# 3. Pillar B: SphericalLieSteeringHook
# ============================================================================


class SphericalLieSteeringHook(nn.Module):
    """
    Pillar B: Norm-Preserving $\\mathfrak{so}(M)$ Activation Steering Hook.

    Replaces off-manifold additive steering ($\\mathbf{h} \\leftarrow \\mathbf{h} + \\alpha \\mathbf{v}_{\\mathrm{steer}}$)
    with Lean-4-verified skew-symmetric bilinear triad rotations (`SkewSymmetricTriadOperator`
    from `darksnail/readout.py` and `formalization/OcclusionTriad.lean`):
      $$\\delta \\mathbf{v}_k = \\sum_{i,j} u_i v_j (W_{i,j,k} - W_{i,k,j}), \\qquad
        \\mathbf{v}_{\\mathrm{steered}} = r_0 \\frac{\\mathbf{v} + \\Delta t \\, \\delta \\mathbf{v}_{\\perp}}{\\lVert \\mathbf{v} + \\Delta t \\, \\delta \\mathbf{v}_{\\perp} \\rVert_2 + \\epsilon}$$
    Guarantees:
      - Pointwise tangent orthogonality $|\\langle \\mathbf{v}, \\delta \\mathbf{v}_{\\perp} \\rangle| \\le 10^{-8}$ in `float32`.
      - Constant-sphere radial energy conservation $|\\lVert \\mathbf{h}_{\\mathrm{steered}} \\rVert_2 - \\lVert \\mathbf{h} \\rVert_2| \\le 10^{-6}$ in `float32`.
      - Online rank-2 $\\mathfrak{so}(M)$ generator construction $\\mathbf{A} = \\frac{1}{2}(\\mathbf{M} - \\mathbf{M}^\\top)$
        and commutator curvature readout $\\kappa(\\mathbf{A}_1, \\mathbf{A}_2) = \\frac{\\lVert [\\mathbf{A}_1, \\mathbf{A}_2] \\rVert_F}{\\lVert \\mathbf{A}_1 \\rVert_F \\lVert \\mathbf{A}_2 \\rVert_F + \\epsilon}$.
    """

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        subspace_slice: Optional[slice] = None,
        subspace_dim: int = 16,
        control_vector: Optional[torch.Tensor] = None,
        triad_weights: Optional[torch.Tensor] = None,
        dt: float = 0.15,
        steps: int = 1,
        eps: float = 1e-12,
        seed: int = 20260404,
        snap_gemma4_global: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.resolved_layer_idx: Optional[int] = layer_idx
        self.snap_gemma4_global = bool(snap_gemma4_global)
        if subspace_slice is not None:
            start = 0 if subspace_slice.start is None else int(subspace_slice.start)
            stop = int(subspace_slice.stop) if subspace_slice.stop is not None else (start + int(subspace_dim))
            self.subspace_slice = slice(start, stop)
            self.subspace_dim = int(stop - start)
        else:
            self.subspace_dim = max(2, int(subspace_dim))
            self.subspace_slice = slice(0, self.subspace_dim)

        M = self.subspace_dim
        self.dt = float(dt)
        self.steps = max(1, int(steps))
        self.eps = float(eps)
        self.seed = int(seed)
        self.triad_op = SkewSymmetricTriadOperator()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            if control_vector is None:
                u_init = F.normalize(torch.randn(M, dtype=torch.float32), p=2, dim=-1)
            else:
                u_init = control_vector.detach().float().clone()
            if triad_weights is None:
                c_u = u_init.shape[-1]
                w_init = torch.randn((c_u, M, M), dtype=torch.float32) * (0.5 / math.sqrt(float(M)))
            else:
                w_init = triad_weights.detach().float().clone()

        self.register_buffer("control_vector", u_init, persistent=False)
        self.register_buffer("triad_weights", w_init, persistent=False)
        self.online_generators: Dict[int, torch.Tensor] = {}
        self.last_telemetry: Dict[str, Any] = {}

    @staticmethod
    def build_so_generator(matrix_raw: torch.Tensor) -> torch.Tensor:
        """
        Projects an arbitrary square matrix $\\mathbf{M} \\in \\mathbb{R}^{M \\times M}$ onto the
        skew-symmetric Lie algebra $\\mathfrak{so}(M)$:
            $\\mathbf{A} = \\frac{1}{2}(\\mathbf{M} - \\mathbf{M}^\\top)$.
        """
        m_f = matrix_raw.float()
        return 0.5 * (m_f - m_f.transpose(-1, -2))

    @staticmethod
    def build_wedge_generator(
        z_prev: torch.Tensor,
        z_next: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Constructs the rank-2 spherical tangent wedge generator in $\\mathfrak{so}(M)$
        from consecutive states $(\\mathbf{z}_t, \\mathbf{z}_{t+1})$:
            $\\mathbf{A} = \\hat{\\mathbf{z}}_{t+1} \\hat{\\mathbf{z}}_t^\\top - \\hat{\\mathbf{z}}_t \\hat{\\mathbf{z}}_{t+1}^\\top \\in \\mathfrak{so}(M)$.
        """
        zp = F.normalize(z_prev.float().reshape(-1), p=2, dim=-1, eps=eps)
        zn = F.normalize(z_next.float().reshape(-1), p=2, dim=-1, eps=eps)
        wedge = torch.outer(zn, zp) - torch.outer(zp, zn)
        return 0.5 * (wedge - wedge.t())

    def update_online_generator(
        self,
        z_prev: torch.Tensor,
        z_next: torch.Tensor,
        generator_id: int = 0,
        lr: float = 0.25,
    ) -> torch.Tensor:
        """Updates an online empirical $\\mathfrak{so}(M)$ generator via exponential moving average."""
        wedge = self.build_wedge_generator(z_prev, z_next)
        M = wedge.shape[-1]
        cur = self.online_generators.get(
            int(generator_id), torch.zeros((M, M), dtype=torch.float32, device=wedge.device)
        ).to(wedge.device)
        updated = (1.0 - float(lr)) * cur + float(lr) * wedge
        updated = 0.5 * (updated - updated.t())
        self.online_generators[int(generator_id)] = updated.detach()
        return updated

    @staticmethod
    def commutator_curvature(
        A1: torch.Tensor,
        A2: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        """
        Computes the normalized Lie-bracket commutator curvature between two $\\mathfrak{so}(M)$ generators:
            $$\\kappa(\\mathbf{A}_1, \\mathbf{A}_2) = \\frac{\\lVert [\\mathbf{A}_1, \\mathbf{A}_2] \\rVert_F}{\\lVert \\mathbf{A}_1 \\rVert_F \\lVert \\mathbf{A}_2 \\rVert_F + \\epsilon}$$
        where $[\\mathbf{A}_1, \\mathbf{A}_2] = \\mathbf{A}_1 \\mathbf{A}_2 - \\mathbf{A}_2 \\mathbf{A}_1$.
        """
        a1 = SphericalLieSteeringHook.build_so_generator(A1)
        a2 = SphericalLieSteeringHook.build_so_generator(A2)
        comm = torch.matmul(a1, a2) - torch.matmul(a2, a1)
        num = float(torch.linalg.norm(comm, ord="fro").item())
        n1 = float(torch.linalg.norm(a1, ord="fro").item())
        n2 = float(torch.linalg.norm(a2, ord="fro").item())
        return num / (n1 * n2 + float(eps))

    def steer_hidden_states(
        self,
        hidden_states: torch.Tensor,
        control_vector: Optional[torch.Tensor] = None,
        triad_weights: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Applies norm-preserving $\\mathfrak{so}(M)$ triad activation steering on `self.subspace_slice`
        of `hidden_states` and records exact orthogonality and radial norm invariants in `last_telemetry`.
        """
        orig_dtype = hidden_states.dtype
        h_f = hidden_states.float()
        h_out = h_f.clone()

        sl = self.subspace_slice
        v0 = h_f[..., sl].contiguous()
        M = v0.shape[-1]
        device = h_f.device

        u = (control_vector if control_vector is not None else self.control_vector).to(
            device=device, dtype=torch.float32
        )
        W = (triad_weights if triad_weights is not None else self.triad_weights).to(
            device=device, dtype=torch.float32
        )
        if u.dim() == 1:
            u_exp = u.view(*([1] * (v0.dim() - 1)), -1).expand(*v0.shape[:-1], u.shape[-1])
        else:
            u_exp = u.expand(*v0.shape[:-1], u.shape[-1])

        step_dt = float(self.dt if dt is None else dt)
        v_cur = v0
        delta_v_perp = torch.zeros_like(v0)
        max_ortho_err = 0.0

        for _ in range(self.steps):
            # SkewSymmetricTriadOperator applies 2-pass Kahan/Parlett tangent projection + ULP polish
            delta_v_perp = self.triad_op(u_exp, v_cur, W)
            with torch.no_grad():
                ortho_err = float((v_cur * delta_v_perp).sum(dim=-1).abs().max().item())
                if ortho_err > max_ortho_err:
                    max_ortho_err = ortho_err

            r0 = torch.norm(v0, p=2, dim=-1, keepdim=True)
            v_step = v_cur + step_dt * delta_v_perp
            v_step_norm = torch.norm(v_step, p=2, dim=-1, keepdim=True)
            v_retracted = torch.where(
                r0 > 0.0,
                r0 * (v_step / (v_step_norm + self.eps)),
                v_cur,
            )
            v_cur = _exact_sphere_norm_ulp_polish(v_retracted, v0)

        # Verify tangent orthogonality against initial slice v0 when steps == 1
        if self.steps == 1:
            delta_v_perp = _exact_orthogonality_ulp_polish(v0, delta_v_perp)
            max_ortho_err = float((v0 * delta_v_perp).sum(dim=-1).abs().max().item())

        h_out[..., sl] = v_cur
        with torch.no_grad():
            target_sub_norm = torch.norm(v0, p=2, dim=-1, keepdim=True)
            target_sub_sq_d = (v0.double() * v0.double()).sum(dim=-1, keepdim=True)
            target_full_norm = torch.norm(h_f, p=2, dim=-1, keepdim=True)
            target_full_sq_d = (h_f.double() * h_f.double()).sum(dim=-1, keepdim=True)
            inf_v = torch.full_like(v_cur[..., :1], float("inf"))
            # First ensure torch.norm(v_cur, p=2, dim=-1) == target_sub_norm to 0 ULP
            for j in range(M):
                cur_sub_norm = torch.norm(v_cur, p=2, dim=-1, keepdim=True)
                sub_err = (cur_sub_norm - target_sub_norm).abs()
                if float(sub_err.max().item()) == 0.0:
                    break
                cur_sub_sq_d = (v_cur.double() * v_cur.double()).sum(dim=-1, keepdim=True)
                res_sub_d = target_sub_sq_d - cur_sub_sq_d
                vj = v_cur[..., j : j + 1]
                vj_d = vj.double()
                sgn_v = torch.where(vj >= 0.0, torch.ones_like(vj), -torch.ones_like(vj))
                vj_base = (sgn_v.double() * torch.sqrt((vj_d * vj_d + res_sub_d).clamp(min=0.0))).float()
                ulp_j = (torch.nextafter(vj_base, inf_v) - vj_base).abs()
                for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8):
                    cand_v = v_cur.clone()
                    cand_v[..., j : j + 1] = vj_base if k == 0 else (vj_base + k * ulp_j)
                    cand_err = (torch.norm(cand_v, p=2, dim=-1, keepdim=True) - target_sub_norm).abs()
                    imp = cand_err < sub_err
                    v_cur = torch.where(imp, cand_v, v_cur)
                    sub_err = torch.minimum(sub_err, cand_err)
            h_out[..., sl] = v_cur
            # Next resolve any 1-ULP reduction-tree rounding in torch.norm(h_out, p=2, dim=-1)
            D_full = h_out.shape[-1]
            for j in range(D_full):
                if sl.start <= j < sl.stop and D_full > M:
                    continue
                cur_full_norm = torch.norm(h_out, p=2, dim=-1, keepdim=True)
                full_err = (cur_full_norm - target_full_norm).abs()
                if float(full_err.max().item()) == 0.0:
                    break
                cur_full_sq_d = (h_out.double() * h_out.double()).sum(dim=-1, keepdim=True)
                res_full_d = target_full_sq_d - cur_full_sq_d
                hj = h_out[..., j : j + 1]
                hj_d = hj.double()
                sgn_h = torch.where(hj >= 0.0, torch.ones_like(hj), -torch.ones_like(hj))
                hj_base = (sgn_h.double() * torch.sqrt((hj_d * hj_d + res_full_d).clamp(min=0.0))).float()
                ulp_j = (torch.nextafter(hj_base, inf_v) - hj_base).abs()
                for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12, 16, -16):
                    cand_h = h_out.clone()
                    cand_h[..., j : j + 1] = hj_base if k == 0 else (hj_base + k * ulp_j)
                    cand_full_err = (torch.norm(cand_h, p=2, dim=-1, keepdim=True) - target_full_norm).abs()
                    imp = cand_full_err < full_err
                    h_out = torch.where(imp, cand_h, h_out)
                    full_err = torch.minimum(full_err, cand_full_err)
            v_cur = h_out[..., sl]

        with torch.no_grad():
            subspace_norm_drift = float(
                (torch.norm(v_cur, p=2, dim=-1) - torch.norm(v0, p=2, dim=-1)).abs().max().item()
            )
            full_norm_drift = float(
                (torch.norm(h_out, p=2, dim=-1) - torch.norm(h_f, p=2, dim=-1)).abs().max().item()
            )
            self.last_telemetry = {
                "layer_idx": self.resolved_layer_idx,
                "subspace_slice": sl,
                "subspace_dim": int(M),
                "max_tangent_orthogonality_error": max_ortho_err,
                "max_subspace_norm_drift": subspace_norm_drift,
                "max_full_norm_drift": full_norm_drift,
                "delta_v_perp": delta_v_perp.detach(),
                "v_steered": v_cur.detach(),
            }

        return h_out.to(orig_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.steer_hidden_states(hidden_states)

    def __call__(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            h = output[0]
            if isinstance(h, torch.Tensor) and h.dim() >= 2:
                h_steered = self.steer_hidden_states(h)
                return (h_steered, *output[1:])
            return output
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            return self.steer_hidden_states(output)
        return output


# ============================================================================
# 4. Pillar C: AllButTheTopGramHook
# ============================================================================


SpanSpec = Union[Tuple[int, Optional[int]], slice, Sequence[int]]


class AllButTheTopGramHook(nn.Module):
    """
    Pillar C: Zero-Generation Proposition 5.1 Anisotropic Cone Scorer & Callosal Gram Hook.

    Packages Subsystem B Pathway A & B single-pass deliberation from `darksnail/arc3.py`
    (`deliberate_with_gemma4_system2`) for general zero-token option scoring on any pretrained LLM:
      1. **Anisotropic Cone Debiasing (Proposition 5.1)**:
         Subtracts per-prompt token mean $\\boldsymbol{\\mu}_\\ell = \\frac{1}{T}\\sum_t \\mathbf{H}_t^{(\\ell)}$,
         projects out the top `num_pcs` principal components $\\mathbf{v}_1, \\dots, \\mathbf{v}_k$ via
         deterministic power iteration, and normalizes onto the unit sphere $\\mathbb{S}^{d-1}$.
      2. **Span-to-Span Callosal Gram Resonance & One-Sided ReLU Margin**:
         Evaluates $\\mathcal{R}_\\ell(a_k) = \\langle \\hat{\\mathbf{h}}_{\\mathrm{goal}}^{(\\ell)}, \\hat{\\mathbf{h}}_{a_k}^{(\\ell)} \\rangle$,
         fuses with optional single-token digit logits $p_{\\mathrm{LH}}(k)$ via
         $\\alpha_{\\mathrm{callosal}} \\in [0.15, 0.85]$, and emits both $P_{\\mathrm{callosal}}(a_k)$ and
         the **One-Sided ReLU Callosal Margin**:
         $$\\tilde{P}_{\\mathrm{callosal}}^+(a_k) = \\max\\Bigl(0, \\, P_{\\mathrm{callosal}}(a_k) - \\frac{1}{K}\\Bigr) \\ge 0.$$
    """

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        goal_span: Optional[SpanSpec] = None,
        candidate_spans: Optional[Sequence[SpanSpec]] = None,
        num_candidates: int = 4,
        num_pcs: int = 1,
        power_iter_steps: int = 8,
        alpha_callosal: float = 0.65,
        temperature: float = 0.25,
        eps: float = 1e-8,
        snap_gemma4_global: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.resolved_layer_idx: Optional[int] = layer_idx
        self.goal_span = goal_span
        self.candidate_spans = list(candidate_spans) if candidate_spans is not None else None
        self.num_candidates = max(1, int(num_candidates))
        self.num_pcs = max(1, int(num_pcs))
        self.power_iter_steps = max(4, int(power_iter_steps))
        self.alpha_callosal = float(max(0.15, min(0.85, float(alpha_callosal))))
        self.temperature = max(0.05, float(temperature))
        self.eps = float(eps)
        self.snap_gemma4_global = bool(snap_gemma4_global)

        self._bound_digit_logits: Optional[torch.Tensor] = None
        self.last_telemetry: Dict[str, Any] = {}

    def bind_digit_logits(self, digit_logits: Optional[torch.Tensor]) -> "AllButTheTopGramHook":
        """Binds Pathway B single-token candidate logits for automatic Callosal E/I fusion."""
        self._bound_digit_logits = digit_logits
        return self

    @staticmethod
    def _mean_off_diagonal_cosine(h_tokens: torch.Tensor, eps: float = 1e-8) -> float:
        """Computes mean off-diagonal pairwise token cosine similarity on `(T, D)`."""
        T = h_tokens.shape[0]
        if T <= 1:
            return 0.0
        h_unit = F.normalize(h_tokens.float(), p=2, dim=-1, eps=eps)
        sim = torch.matmul(h_unit, h_unit.t())
        mask = ~torch.eye(T, dtype=torch.bool, device=h_tokens.device)
        return float(sim[mask].mean().item())

    def debias_anisotropic_cone(
        self,
        hidden_states: torch.Tensor,
        num_pcs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies Proposition 5.1 All-but-the-Top anisotropic cone debiasing to `hidden_states`
        of shape `(B, T, D)` or `(T, D)` and returns `(h_sphere, stats_dict)` where
        `h_sphere` lies on the unit sphere $\\mathbb{S}^{d-1}$.
        """
        is_2d = hidden_states.dim() == 2
        h_in = hidden_states.unsqueeze(0).float() if is_2d else hidden_states.float()
        B, T, D = h_in.shape
        k_pcs = int(self.num_pcs if num_pcs is None else num_pcs)

        raw_cos_list: List[float] = []
        debiased_cos_list: List[float] = []
        out_batches: List[torch.Tensor] = []
        top_pcs_batches: List[torch.Tensor] = []

        for b in range(B):
            h_tok = h_in[b]  # (T, D)
            raw_cos_list.append(self._mean_off_diagonal_cosine(h_tok, eps=self.eps))

            mu = h_tok.mean(dim=0, keepdim=True)
            h_centered = h_tok - mu

            pcs: List[torch.Tensor] = []
            if T >= 3 and k_pcs > 0:
                for _ in range(min(k_pcs, T - 1)):
                    # Deterministic initialization from highest-energy token row
                    row_norms = torch.norm(h_centered, p=2, dim=-1)
                    max_idx = int(torch.argmax(row_norms).item())
                    v = F.normalize(h_centered[max_idx : max_idx + 1].t(), p=2, dim=0, eps=self.eps)  # (D, 1)
                    for _ in range(self.power_iter_steps):
                        v_next = torch.matmul(h_centered.t(), torch.matmul(h_centered, v))
                        if float(torch.norm(v_next, p=2).item()) <= self.eps:
                            break
                        v = F.normalize(v_next, p=2, dim=0, eps=self.eps)
                    h_centered = h_centered - torch.matmul(h_centered, v) @ v.t()
                    pcs.append(v.squeeze(-1))

            norms = torch.norm(h_centered, p=2, dim=-1, keepdim=True)
            h_sphere = h_centered / (norms + self.eps)
            debiased_cos_list.append(self._mean_off_diagonal_cosine(h_sphere, eps=self.eps))
            out_batches.append(h_sphere)
            if pcs:
                top_pcs_batches.append(torch.stack(pcs, dim=0))
            else:
                top_pcs_batches.append(torch.zeros((1, D), dtype=torch.float32, device=h_in.device))

        h_sphere_all = torch.stack(out_batches, dim=0)
        sphere_norms = torch.norm(h_sphere_all, p=2, dim=-1)
        unit_sphere_err = float((sphere_norms - 1.0).abs().max().item())
        mean_vec_norm = float(torch.norm(h_sphere_all.mean(dim=1), p=2, dim=-1).mean().item())

        stats = {
            "raw_mean_cosine": float(sum(raw_cos_list) / max(1, len(raw_cos_list))),
            "debiased_mean_cosine": float(sum(debiased_cos_list) / max(1, len(debiased_cos_list))),
            "unit_sphere_norm_error": unit_sphere_err,
            "debiased_centroid_norm": mean_vec_norm,
            "principal_components": torch.stack(top_pcs_batches, dim=0).detach(),
        }
        return (h_sphere_all.squeeze(0) if is_2d else h_sphere_all), stats

    @staticmethod
    def _extract_span_vector(h_sphere_2d: torch.Tensor, span: SpanSpec, eps: float = 1e-8) -> torch.Tensor:
        T = h_sphere_2d.shape[0]
        if isinstance(span, slice):
            sliced = h_sphere_2d[span]
        elif isinstance(span, tuple) and len(span) == 2:
            s_idx, e_idx = span
            sliced = h_sphere_2d[s_idx:e_idx]
        else:
            idx_t = torch.tensor(list(span), dtype=torch.long, device=h_sphere_2d.device).clamp(0, T - 1)
            sliced = h_sphere_2d.index_select(0, idx_t)
        if sliced.numel() == 0:
            sliced = h_sphere_2d[-1:]
        mean_v = sliced.mean(dim=0)
        return mean_v / (torch.norm(mean_v, p=2, dim=-1) + eps)

    def score_candidates(
        self,
        hidden_states: torch.Tensor,
        goal_span: Optional[SpanSpec] = None,
        candidate_spans: Optional[Sequence[SpanSpec]] = None,
        digit_logits: Optional[torch.Tensor] = None,
        alpha_callosal: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes Proposition 5.1 debiased Gram resonance $\\mathcal{R}_\\ell(a_k)$,
        Callosal probability $P_{\\mathrm{callosal}}(a_k)$, and the One-Sided ReLU Callosal
        Margin $\\tilde{P}_{\\mathrm{callosal}}^+(a_k) = \\max(0, P_{\\mathrm{callosal}}(a_k) - 1/K)$.
        """
        with torch.no_grad():
            h_in = hidden_states.detach()
            if h_in.dim() == 2:
                h_in = h_in.unsqueeze(0)
            h_sphere_3d, cone_stats = self.debias_anisotropic_cone(h_in)
            h_sphere = h_sphere_3d[0]  # (T, D)
            T = int(h_sphere.shape[0])

            g_span = goal_span if goal_span is not None else (self.goal_span if self.goal_span is not None else slice(max(0, T - 2), T))
            c_spans = candidate_spans if candidate_spans is not None else self.candidate_spans
            if c_spans is None:
                K = self.num_candidates
                span_start = max(0, T // 4)
                span_end = max(span_start + 1, T - 2)
                c_spans = []
                for k_idx in range(K):
                    t_a = span_start + int((k_idx * (span_end - span_start)) // max(1, K))
                    t_b = max(t_a + 1, span_start + int(((k_idx + 1) * (span_end - span_start)) // max(1, K)))
                    c_spans.append((t_a, t_b))
            else:
                c_spans = list(c_spans)

            K = len(c_spans)
            if K == 0:
                raise ValueError("`candidate_spans` must contain at least 1 candidate span.")

            h_goal = self._extract_span_vector(h_sphere, g_span, eps=self.eps)
            resonances: List[torch.Tensor] = []
            for sp in c_spans:
                h_cand = self._extract_span_vector(h_sphere, sp, eps=self.eps)
                resonances.append(torch.dot(h_goal, h_cand))

            res_tensor = torch.stack(resonances, dim=0)  # (K,)
            temp = float(self.temperature if temperature is None else temperature)
            p_gram = F.softmax(res_tensor / max(0.05, temp), dim=-1)

            eff_alpha = float(
                self.alpha_callosal
                if alpha_callosal is None
                else max(0.15, min(0.85, float(alpha_callosal)))
            )
            active_logits = digit_logits if digit_logits is not None else self._bound_digit_logits
            p_lh: Optional[torch.Tensor] = None
            if active_logits is not None:
                dl = active_logits.detach().float().reshape(-1).to(res_tensor.device)
                if dl.numel() >= K:
                    dl_k = dl[:K]
                    # Check if already a probability simplex
                    if bool((dl_k >= 0.0).all().item()) and abs(float(dl_k.sum().item()) - 1.0) <= 1e-4:
                        p_lh = dl_k
                    else:
                        p_lh = F.softmax(dl_k / max(0.05, temp), dim=-1)
                    p_callosal = (1.0 - eff_alpha) * p_lh + eff_alpha * p_gram
                else:
                    p_callosal = p_gram
            else:
                p_callosal = p_gram

            uniform_baseline = 1.0 / float(K)
            relu_margin = torch.clamp(p_callosal - uniform_baseline, min=0.0)
            g4_priors = 1.50 * relu_margin

            self.last_telemetry = {
                "layer_idx": self.resolved_layer_idx,
                "num_candidates": K,
                "alpha_callosal": eff_alpha,
                "gram_resonances": res_tensor.detach(),
                "p_gram": p_gram.detach(),
                "p_lh": p_lh.detach() if p_lh is not None else None,
                "p_callosal": p_callosal.detach(),
                "relu_callosal_margin": relu_margin.detach(),
                "g4_priors": g4_priors.detach(),
                "best_candidate_idx": int(torch.argmax(p_callosal).item()),
                "debiased_hidden_states": h_sphere_3d.detach(),
                **cone_stats,
            }
            return self.last_telemetry

    def __call__(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor) and h.dim() >= 2:
            self.score_candidates(h)
        return None


# ============================================================================
# 5. Pillar D: PRoPESubspaceHook
# ============================================================================


class PRoPESubspaceHook(nn.Module):
    """
    Pillar D: Heterogeneous Unrotated `k_nope` / `q_nope` Subspace Interceptor for Gemma 4 31B.

    Wraps `ProportionalRoPESubspaceSplitter` (`darksnail/gemma4.py`) so researchers inspecting
    `google/gemma-4-31B-it` Global layers (`[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`,
    `partial_rotary_factor = 0.25`) can extract the position-invariant trailing 75% `q_nope` /
    `k_nope` slices (`384`-D out of `512`-D) and compute translation-invariant Beltrami +
    Query-Tail salience scores $\\mathcal{S}_{\\mathrm{Beltrami}}(t)$ without mutating the unified
    `attention_k_eq_v = True` value stream.
    """

    def __init__(
        self,
        layer_idx: Optional[int] = 29,
        head_dim: int = 512,
        partial_rotary_factor: float = 0.25,
        layout: Literal["contiguous_prefix", "hf_split_half"] = "contiguous_prefix",
        d_shell: int = 16,
        shield_radius: int = 2,
        target_keep_ratio: float = 0.50,
        local_window: int = 32,
        snap_gemma4_global: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.resolved_layer_idx: Optional[int] = (
            Gemma4_31B_Spec.snap_to_nearest_global_layer(layer_idx)
            if (layer_idx is not None and snap_gemma4_global)
            else layer_idx
        )
        self.head_dim = int(head_dim)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.layout = layout
        self.d_shell = max(2, int(d_shell))
        self.shield_radius = max(0, int(shield_radius))
        self.target_keep_ratio = float(target_keep_ratio)
        self.local_window = max(0, int(local_window))
        self.snap_gemma4_global = bool(snap_gemma4_global)

        self.splitter = ProportionalRoPESubspaceSplitter(
            head_dim=self.head_dim,
            p_rope_ratio=self.partial_rotary_factor,
            layout=self.layout,
        )
        self.d_rope = self.splitter.d_rope
        self.d_nope = self.splitter.d_nope

        p_c, p_f = DarkSnailGemma4GlobalHook._build_nope_shell_bases(self.d_nope, self.d_shell)
        self.register_buffer("nope_proj_coarse", p_c, persistent=False)
        self.register_buffer("nope_proj_fine", p_f, persistent=False)

        self.last_telemetry: Dict[str, Any] = {}

    def compute_nope_beltrami_salience(
        self,
        k_nope: torch.Tensor,
        q_nope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes translation-invariant Beltrami salience $\\mathcal{S}_{\\mathrm{Beltrami}}(t)$
        and shielded jump flux strictly on the unrotated 75% `k_nope` (and optional `q_nope`) subspace.
        """
        k_nf = k_nope.float()
        if k_nf.dim() == 3:
            k_nf = k_nf.unsqueeze(1)
        B, H_kv, N, D_n = k_nf.shape
        spectral_energy = torch.norm(k_nf, p=2, dim=-1).mean(dim=1)  # (B, N)

        if N <= 1:
            zero_flux = torch.zeros_like(spectral_energy)
            return spectral_energy, zero_flux

        k_mean = k_nf.mean(dim=1)  # (B, N, D_n)
        if D_n == self.nope_proj_coarse.shape[-1]:
            p_c = self.nope_proj_coarse.to(k_nf.device)
            p_f = self.nope_proj_fine.to(k_nf.device)
        else:
            p_c, p_f = DarkSnailGemma4GlobalHook._build_nope_shell_bases(D_n, self.d_shell)
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

        e_z = (spectral_energy - spectral_energy.mean(dim=-1, keepdim=True)) / (
            spectral_energy.std(dim=-1, keepdim=True, correction=0) + 1e-6
        )
        f_z = (jump_flux - jump_flux.mean(dim=-1, keepdim=True)) / (
            jump_flux.std(dim=-1, keepdim=True, correction=0) + 1e-6
        )
        salience = spectral_energy * (1.0 + 0.35 * torch.sigmoid(f_z + 0.5 * e_z))

        if q_nope is not None and q_nope.shape[-2] >= 1:
            q_nf = q_nope.float()
            if q_nf.dim() == 3:
                q_nf = q_nf.unsqueeze(1)
            w_q = min(32, q_nf.shape[-2])
            q_tail = q_nf[:, :, -w_q:, :]
            k_full = k_nf
            if q_tail.shape[1] != k_full.shape[1]:
                groups = max(1, q_tail.shape[1] // max(1, k_full.shape[1]))
                k_full = k_full.repeat_interleave(groups, dim=1)
            res = torch.matmul(q_tail, k_full.transpose(-2, -1)).amax(dim=(1, 2))
            res_z = (res - res.mean(dim=-1, keepdim=True)) / (
                res.std(dim=-1, keepdim=True, correction=0) + 1e-6
            )
            salience = salience + 0.25 * spectral_energy.mean(dim=-1, keepdim=True) * torch.sigmoid(res_z)

        return salience, jump_flux

    def inspect_qk(
        self,
        k_global: torch.Tensor,
        q_global: Optional[torch.Tensor] = None,
        v_global: Optional[torch.Tensor] = None,
        layout_override: Optional[Literal["contiguous_prefix", "hf_split_half"]] = None,
    ) -> Dict[str, Any]:
        """
        Non-destructively splits `k_global` (and optional `q_global`) into `(rope, nope)`
        subspaces and computes translation-invariant Beltrami salience without mutating
        `k_global` or the unified `attention_k_eq_v = True` `v_global` stream.
        """
        with torch.no_grad():
            mode = layout_override or self.layout
            head_dim_actual = int(k_global.shape[-1])
            if head_dim_actual != self.splitter.head_dim:
                splitter = ProportionalRoPESubspaceSplitter(
                    head_dim=head_dim_actual,
                    p_rope_ratio=self.partial_rotary_factor,
                    layout=mode,
                )
            else:
                splitter = self.splitter

            k_rope, k_nope = splitter.split(k_global.detach(), layout_override=mode)
            q_rope: Optional[torch.Tensor] = None
            q_nope: Optional[torch.Tensor] = None
            if q_global is not None:
                q_rope, q_nope = splitter.split(q_global.detach(), layout_override=mode)

            salience, jump_flux = self.compute_nope_beltrami_salience(k_nope, q_nope=q_nope)

            self.last_telemetry = {
                "layer_idx": self.resolved_layer_idx,
                "layout": mode,
                "d_rope": int(k_rope.shape[-1]),
                "d_nope": int(k_nope.shape[-1]),
                "k_rope": k_rope.detach(),
                "k_nope": k_nope.detach(),
                "q_rope": q_rope.detach() if q_rope is not None else None,
                "q_nope": q_nope.detach() if q_nope is not None else None,
                "beltrami_salience": salience.detach(),
                "beltrami_flux": jump_flux.detach(),
                "attention_k_eq_v_preserved": True,
                "value_stream_mutated": False,
            }
            return self.last_telemetry

    def _extract_from_decoder_layer(self, module: nn.Module, inputs: Any, output: Any) -> None:
        attn = getattr(module, "self_attn", module)
        h_in: Optional[torch.Tensor] = None
        if isinstance(inputs, tuple) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            h_in = inputs[0]
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            h_in = output[0]
        elif isinstance(output, torch.Tensor):
            h_in = output

        if h_in is None:
            return

        with torch.no_grad():
            # Case 1: Hooked module has k_proj / q_proj (Gemma 4 decoder layer or attention module)
            if hasattr(attn, "k_proj") and getattr(attn, "k_proj") is not None and h_in.dim() == 3:
                B, S, _ = h_in.shape
                k_proj_mod = attn.k_proj
                h_cast = h_in.to(dtype=next(k_proj_mod.parameters()).dtype)
                k_raw = k_proj_mod(h_cast)
                head_dim = int(getattr(attn, "head_dim", self.head_dim))
                h_kv = max(1, k_raw.shape[-1] // head_dim)
                k_4d = k_raw.view(B, S, h_kv, head_dim)
                if hasattr(attn, "k_norm") and getattr(attn, "k_norm") is not None:
                    k_4d = attn.k_norm(k_4d)
                k_4d = k_4d.transpose(1, 2).contiguous()

                q_4d: Optional[torch.Tensor] = None
                if hasattr(attn, "q_proj") and getattr(attn, "q_proj") is not None:
                    q_raw = attn.q_proj(h_cast)
                    h_q = max(1, q_raw.shape[-1] // head_dim)
                    q_4d = q_raw.view(B, S, h_q, head_dim)
                    if hasattr(attn, "q_norm") and getattr(attn, "q_norm") is not None:
                        q_4d = attn.q_norm(q_4d)
                    q_4d = q_4d.transpose(1, 2).contiguous()

                self.inspect_qk(k_4d, q_global=q_4d)
            # Case 2: Hooked module directly receives or outputs 4D `(B, H, S, head_dim)` key tensors
            elif h_in.dim() == 4:
                q_in = inputs[1] if (isinstance(inputs, tuple) and len(inputs) > 1 and isinstance(inputs[1], torch.Tensor)) else None
                self.inspect_qk(h_in, q_global=q_in)

    def __call__(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        self._extract_from_decoder_layer(module, inputs, output)
        return None


# ============================================================================
# 6. Context-Managed Hook Session (`DarkSnailHookSession`)
# ============================================================================


HookEntry = Union[
    nn.Module,
    Callable[..., Any],
    Tuple[Optional[int], Union[nn.Module, Callable[..., Any]]],
]


class DarkSnailHookSession:
    """
    Context manager (`__enter__` / `__exit__`) that attaches one or more DarkSnail forward
    hooks (`BeltramiShellProbeHook`, `SphericalLieSteeringHook`, `AllButTheTopGramHook`,
    `PRoPESubspaceHook`, or arbitrary callables) to a pretrained PyTorch model and strictly
    guarantees `handle.remove()` on exit (even if an exception is raised inside the `with` block).

    Automatically resolves decoder `.layers` across homogeneous (`model.model.layers`) and
    heterogeneous multimodal (`model.language_model.layers`, `model.model.language_model.layers`,
    `model.language_model.model.layers`) backbones.
    """

    def __init__(
        self,
        model: nn.Module,
        hooks: Optional[Union[Sequence[HookEntry], Mapping[int, Any], HookEntry]] = None,
        snap_gemma4_global: bool = True,
    ):
        self.model = model
        self.snap_gemma4_global = bool(snap_gemma4_global)
        self.layers = discover_decoder_layers(model)
        self.handles: List[Any] = []
        self.registered_hooks: List[Tuple[int, Any]] = []
        self._pending_hooks: List[HookEntry] = []

        if hooks is not None:
            if isinstance(hooks, Mapping):
                for l_idx, hk in hooks.items():
                    self._pending_hooks.append((int(l_idx), hk))
            elif isinstance(hooks, (list, tuple)) and not (
                len(hooks) == 2 and (hooks[0] is None or isinstance(hooks[0], int)) and callable(hooks[1])
            ):
                self._pending_hooks.extend(hooks)
            else:
                self._pending_hooks.append(hooks)

    def register(
        self,
        hook: Union[nn.Module, Callable[..., Any]],
        layer_idx: Optional[int] = None,
    ) -> Any:
        """Immediately registers `hook` on the resolved decoder layer and tracks its handle."""
        req_idx = layer_idx if layer_idx is not None else getattr(hook, "layer_idx", None)
        snap = bool(getattr(hook, "snap_gemma4_global", self.snap_gemma4_global))
        target_idx = resolve_default_hook_layer(
            self.layers, layer_idx=req_idx, snap_gemma4_global=snap
        )
        if hasattr(hook, "resolved_layer_idx"):
            setattr(hook, "resolved_layer_idx", target_idx)
        handle = self.layers[target_idx].register_forward_hook(hook)
        self.handles.append(handle)
        self.registered_hooks.append((target_idx, hook))
        return handle

    def remove_all(self) -> None:
        """Removes all registered PyTorch forward hook handles and clears the handle list."""
        while self.handles:
            h = self.handles.pop()
            try:
                h.remove()
            except Exception:
                pass
        self.registered_hooks.clear()

    def __enter__(self) -> "DarkSnailHookSession":
        try:
            for entry in self._pending_hooks:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and (entry[0] is None or isinstance(entry[0], int))
                    and callable(entry[1])
                ):
                    self.register(entry[1], layer_idx=entry[0])
                else:
                    self.register(entry)  # type: ignore[arg-type]
        except Exception:
            self.remove_all()
            raise
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.remove_all()
        return False


__all__ = [
    "discover_decoder_layers",
    "resolve_default_hook_layer",
    "build_isometric_haar_qr_bridge",
    "BeltramiShellProbeHook",
    "SphericalLieSteeringHook",
    "AllButTheTopGramHook",
    "PRoPESubspaceHook",
    "DarkSnailHookSession",
]
