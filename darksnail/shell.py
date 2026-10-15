"""
darksnail.shell: Adelic Tree + Beltrami Outlier-Shielded Callosal Shell Extraction.

Implements Upgrade 1 of Project DarkSnail V2:
- Replaces uniform `F.adaptive_avg_pool1d(raw_latents, 32)` in `brain_ai.models.lh_cache`
  with `BeltramiCallosalShell(nn.Module)`, preserving the exact `(B, 32, 4096)` float32
  contract expected by `CallosalConditioning` while protecting high-salience rule transitions
  and foreground tokens via Tier-1 Beltrami Outlier Shielding (16 slots) and Tier-2
  Ultrametric Medoid Super-Tokens (16 slots).
- Provides `compute_grid_prompt_zeno_flux(demo_pairs, test_inputs)` for sub-millisecond
  deterministic multi-scale dyadic/triadic shell flux (Phi_LH) and cophenetic LCA depth
  variance (sigma_LCA^2) directly from ARC task demonstration and test input grids.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeltramiCallosalShell(nn.Module):
    """
    Adelic Tree + Beltrami Outlier-Shielded Callosal Shell Extractor.

    Compresses variable-length Left Hemisphere (Layer 16) residual stream states
    of shape `(B, S, D)` into `target_tokens = 32` super-tokens of shape `(B, 32, D)`
    in `float32`:
      - Tier-1 (`outlier_budget = 16` slots): Top-16 Beltrami outlier-shielded tokens
        in preserved chronological order.
      - Tier-2 (`condensed_budget = 16` slots): 16 contiguous ultrametric tree blocks
        over the remaining tokens, selecting the block Medoid token
        `t* = argmin_{t in S_c} ||h_t - mean(h_{S_c})||_2`, combining with
        `0.5 * (mean_pool + peak_pool)`, and attenuating by Adelic positional
        variance `gamma_c = exp(-nu * Var(P_{S_c}))`.
    """

    def __init__(
        self,
        d_model: int = 4096,
        target_tokens: int = 32,
        outlier_budget: int = 16,
        condensed_budget: int = 16,
        d_shell: int = 16,
        shield_radius: int = 2,
        viscous_nu: float = 0.05,
        viscous_p: int = 2,
        medoid_blend: float = 0.5,
        router_sidecar: Optional[nn.Module] = None,
        sidecar_path: Optional[str] = None,
        hook_layer: int = 16,
        num_outliers: Optional[int] = None,
        num_medoids: Optional[int] = None,
        num_clusters: Optional[int] = None,
    ):
        super().__init__()
        if num_outliers is not None:
            outlier_budget = num_outliers
        if num_medoids is not None:
            condensed_budget = num_medoids
        if num_clusters is not None:
            condensed_budget = num_clusters
        if num_outliers is not None or num_medoids is not None or num_clusters is not None:
            target_tokens = int(outlier_budget) + int(condensed_budget)
        if outlier_budget + condensed_budget != target_tokens:
            raise ValueError(
                f"outlier_budget ({outlier_budget}) + condensed_budget ({condensed_budget}) "
                f"must equal target_tokens ({target_tokens})"
            )
        self.d_model = int(d_model)
        self.target_tokens = int(target_tokens)
        self.outlier_budget = int(outlier_budget)
        self.condensed_budget = int(condensed_budget)
        self.d_shell = max(2, int(d_shell))
        self.shield_radius = max(0, int(shield_radius))
        self.viscous_nu = float(viscous_nu)
        self.viscous_p = max(2, int(viscous_p))
        self.medoid_blend = float(medoid_blend)
        self.hook_layer = int(hook_layer)
        self.router_sidecar = router_sidecar

        # Build deterministic orthogonal multi-scale Haar/DCT shell projection buffers
        # Used when no V4 TriadTopologyRouter sidecar is attached or when input is full d_model
        p_coarse, p_fine = self._build_deterministic_shell_projections(self.d_model, self.d_shell)
        self.register_buffer("proj_coarse", p_coarse, persistent=False)
        self.register_buffer("proj_fine", p_fine, persistent=False)

        if sidecar_path is not None and os.path.exists(sidecar_path):
            self.load_sidecar_projections(sidecar_path, layer_idx=self.hook_layer)

        # Telemetry recorded on the most recent forward pass
        self.last_token_flux: Optional[torch.Tensor] = None
        self.last_shielded_flux: Optional[torch.Tensor] = None
        self.last_cross_shell_curl: Optional[torch.Tensor] = None
        self.last_outlier_indices: Optional[torch.Tensor] = None
        self.last_medoid_indices: Optional[torch.Tensor] = None
        self.last_gamma: Optional[torch.Tensor] = None
        self.last_pos_var: Optional[torch.Tensor] = None
        self.last_zeno_flux: float = 0.0
        self.last_lca_variance: float = 0.0

    @staticmethod
    def _build_deterministic_shell_projections(
        d_model: int, d_shell: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs deterministic orthogonal low-frequency (coarse) and high-frequency (fine)
        multi-scale Haar/DCT shell basis matrices in R^(d_shell x d_model).
        """
        idx = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)  # (1, D)
        k_coarse = torch.arange(1, d_shell + 1, dtype=torch.float32).unsqueeze(1)  # (M, 1)
        k_fine = torch.arange(d_shell + 1, 2 * d_shell + 1, dtype=torch.float32).unsqueeze(1)

        # Low-frequency cosine shell (coarse laminar envelope)
        raw_coarse = torch.cos(math.pi * (idx + 0.5) * k_coarse / float(d_model))
        # High-frequency alternating Haar/Walsh-modulated sine shell (fine local innovation)
        haar_sign = torch.where((torch.arange(d_model) % 2) == 0, 1.0, -1.0).float().unsqueeze(0)
        raw_fine = torch.sin(math.pi * (idx + 0.5) * k_fine / float(d_model)) * haar_sign

        # Orthonormalize rows via QR decomposition on the stacked (2*d_shell, d_model)^T
        stacked = torch.cat([raw_coarse, raw_fine], dim=0).t()  # (D, 2*M)
        if d_model >= 2 * d_shell:
            q, _ = torch.linalg.qr(stacked, mode="reduced")  # (D, 2*M)
            p_coarse = q[:, :d_shell].t().contiguous()
            p_fine = q[:, d_shell : 2 * d_shell].t().contiguous()
        else:
            p_coarse = F.normalize(raw_coarse, p=2, dim=-1)
            p_fine = F.normalize(raw_fine, p=2, dim=-1)
        return p_coarse, p_fine

    def load_sidecar_projections(self, sidecar_path: str, layer_idx: int = 16) -> bool:
        """
        Loads V4 TriadTopologyRouter `shell_inject` weights from a safetensors sidecar
        at `layer_idx` if present.
        """
        try:
            from safetensors.torch import load_file

            tensors = load_file(sidecar_path, device="cpu")
            target_key = None
            for k in tensors:
                if f"layers.{layer_idx}." in k and "shell_inject" in k:
                    target_key = k
                    break
            if target_key is None:
                for k in tensors:
                    if "shell_inject" in k:
                        target_key = k
                        break
            if target_key is not None:
                shell_inj = tensors[target_key].float()  # (H, L, M, in_dim)
                if shell_inj.dim() == 4 and shell_inj.shape[1] >= 2:
                    # Average or flatten across heads to build coarse (level 0) and fine (level -1) projections
                    coarse_w = shell_inj[:, 0, :, :].reshape(-1, shell_inj.shape[-1])
                    fine_w = shell_inj[:, -1, :, :].reshape(-1, shell_inj.shape[-1])
                    self.register_buffer("sidecar_coarse_w", coarse_w, persistent=False)
                    self.register_buffer("sidecar_fine_w", fine_w, persistent=False)
                    return True
        except Exception:
            pass
        return False

    def compute_shell_projections(
        self, raw_latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes coarse (`u_coarse`) and fine (`u_fine`) shell states of shape `(B, S, M)`
        in `float32` from `raw_latents` of shape `(B, S, D)`.
        """
        x = raw_latents.float()
        B, S, D = x.shape

        # 1. If an active V4 TriadTopologyRouter sidecar is attached and compatible
        if self.router_sidecar is not None and getattr(self.router_sidecar, "shell_inject", None) is not None:
            shell_inj = self.router_sidecar.shell_inject.float()  # (H, L, M, in_dim)
            in_dim = shell_inj.shape[-1]
            if D == in_dim:
                x_h = x.unsqueeze(1).expand(B, shell_inj.shape[0], S, in_dim)
                u_c = torch.einsum("bhnd,hmd->bhnm", x_h, shell_inj[:, 0]).mean(dim=1)
                u_f = torch.einsum("bhnd,hmd->bhnm", x_h, shell_inj[:, -1]).mean(dim=1)
                return u_c, u_f
            elif D >= in_dim:
                # Project coarse from first in_dim channels and fine from last in_dim channels
                x_c = x[..., :in_dim].unsqueeze(1).expand(B, shell_inj.shape[0], S, in_dim)
                x_f = x[..., -in_dim:].unsqueeze(1).expand(B, shell_inj.shape[0], S, in_dim)
                u_c = torch.einsum("bhnd,hmd->bhnm", x_c, shell_inj[:, 0]).mean(dim=1)
                u_f = torch.einsum("bhnd,hmd->bhnm", x_f, shell_inj[:, -1]).mean(dim=1)
                return u_c, u_f

        # 2. If safetensors sidecar weights were loaded
        if hasattr(self, "sidecar_coarse_w") and hasattr(self, "sidecar_fine_w"):
            sc_w = self.sidecar_coarse_w.to(x.device)
            sf_w = self.sidecar_fine_w.to(x.device)
            in_dim = sc_w.shape[-1]
            if D >= in_dim:
                u_c = F.linear(x[..., :in_dim], sc_w[: self.d_shell])
                u_f = F.linear(x[..., -in_dim:], sf_w[: self.d_shell])
                return u_c, u_f

        # 3. Deterministic orthogonal multi-scale shell projections + temporal dyadic smoothing
        if D == self.proj_coarse.shape[-1]:
            p_c = self.proj_coarse.to(x.device)
            p_f = self.proj_fine.to(x.device)
        else:
            p_c, p_f = self._build_deterministic_shell_projections(D, self.d_shell)
            p_c = p_c.to(x.device)
            p_f = p_f.to(x.device)

        u_c_raw = F.linear(x, p_c)  # (B, S, M)
        u_f_raw = F.linear(x, p_f)  # (B, S, M)

        # Apply dyadic temporal low-pass filter to u_coarse so it captures multi-token block context
        if S >= 3:
            u_c = F.avg_pool1d(
                u_c_raw.transpose(1, 2), kernel_size=3, stride=1, padding=1
            ).transpose(1, 2)
        else:
            u_c = u_c_raw

        return u_c, u_f_raw

    def compute_beltrami_triad_flux(
        self, raw_latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the Beltrami Triad Flux per token:
            Phi_tilde_t = sqrt(1 - <u_hat_coarse,t, u_hat_fine,t>^2) * ||u_fine,t - u_fine,t-1||_2
        and applies a kernel-(2*shield_radius + 1) `F.max_pool1d` neighborhood shield.

        Returns:
            shielded_flux: `(B, S)` in `float32`
            raw_flux: `(B, S)` in `float32`
            cross_shell_curl: `(B, S)` in `float32`
        """
        u_coarse, u_fine = self.compute_shell_projections(raw_latents)
        u_hat_c = F.normalize(u_coarse.float(), p=2, dim=-1)
        u_hat_f = F.normalize(u_fine.float(), p=2, dim=-1)

        cos_cf = (u_hat_c * u_hat_f).sum(dim=-1).clamp(min=-1.0, max=1.0)
        cross_shell_curl = torch.sqrt(torch.clamp(1.0 - cos_cf * cos_cf, min=0.0))  # (B, S)

        B, S, _ = u_fine.shape
        fine_innovation = torch.zeros_like(cross_shell_curl)
        if S > 1:
            diff = torch.norm(u_fine[:, 1:, :] - u_fine[:, :-1, :], p=2, dim=-1)
            fine_innovation[:, 1:] = diff
            fine_innovation[:, :-1] = torch.maximum(fine_innovation[:, :-1], diff)

        raw_flux = cross_shell_curl * fine_innovation  # (B, S)

        if self.shield_radius > 0 and S > 1:
            k_size = 2 * self.shield_radius + 1
            shielded_flux = F.max_pool1d(
                raw_flux.unsqueeze(1),
                kernel_size=k_size,
                stride=1,
                padding=self.shield_radius,
            ).squeeze(1)[:, :S]
        else:
            shielded_flux = raw_flux

        return shielded_flux, raw_flux, cross_shell_curl

    @staticmethod
    def pool_tensor_blocks_1d(x: torch.Tensor, num_blocks: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Partitions `x` of shape `(B, S_rem, D)` and `positions` `(B, S_rem)` into
        `num_blocks` contiguous ultrametric tree blocks.
        For each block `S_c`:
          1. Selects the Medoid token `t* = argmin_{t in S_c} ||h_t - mean(h_{S_c})||_2`
          2. Pools block values via `0.5 * (mean_pool + peak_pool)` (from `SurgicalLlamaAttention._pool_tensor_blocks`)
          3. Computes positional variance `Var(P_{S_c})`

        Returns:
            medoid_tokens: `(B, C, D)`
            pooled_values: `(B, C, D)`
            medoid_indices_in_rem: `(B, C)`
            pos_var: `(B, C)`
            block_counts: `(B, C)`
        """
        B, S_rem, D = x.shape
        C = int(num_blocks)
        device = x.device
        x_f = x.float()

        # Partition [0, S_rem) into C contiguous non-empty blocks
        # Use exact integer boundaries floor(c * S_rem / C) .. floor((c+1) * S_rem / C)
        boundaries = [int((c * S_rem) // C) for c in range(C + 1)]
        boundaries[-1] = S_rem

        medoid_list = []
        pooled_list = []
        medoid_idx_list = []
        pos_var_list = []
        counts_list = []

        pos_scale = max(float(S_rem) / 10.0, 1.0)

        for c in range(C):
            start = boundaries[c]
            end = max(start + 1, min(boundaries[c + 1], S_rem))
            if start >= S_rem:
                start = S_rem - 1
                end = S_rem

            blk = x_f[:, start:end, :]  # (B, blk_len, D)
            blk_len = blk.shape[1]

            # 1. Block mean
            mean_h = blk.mean(dim=1, keepdim=True)  # (B, 1, D)

            # 2. Medoid token: t* = argmin_{t in S_c} ||h_t - mean_h||_2
            dists = torch.norm(blk - mean_h, p=2, dim=-1)  # (B, blk_len)
            local_medoid_idx = dists.argmin(dim=-1)  # (B,)
            global_medoid_idx = local_medoid_idx + start  # (B,)
            gather_med = local_medoid_idx.view(B, 1, 1).expand(B, 1, D)
            medoid_h = torch.gather(blk, dim=1, index=gather_med).squeeze(1)  # (B, D)

            # 3. Peak-norm token in block (`SurgicalLlamaAttention._pool_tensor_blocks`)
            norms = torch.norm(blk, p=2, dim=-1)  # (B, blk_len)
            peak_local_idx = norms.argmax(dim=-1).view(B, 1, 1).expand(B, 1, D)
            peak_h = torch.gather(blk, dim=1, index=peak_local_idx).squeeze(1)  # (B, D)
            pooled_v = 0.5 * (mean_h.squeeze(1) + peak_h)  # (B, D)

            # 4. Feature-weighted positional variance Var(P_{S_c}) in block
            if blk_len > 1:
                pos_c = (
                    torch.arange(start, end, device=device, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(B, blk_len)
                    / pos_scale
                )
                # Weight positions by token energy so non-uniform blocks reflect true phase spread
                tok_weights = (norms + 1e-6) / (norms.sum(dim=-1, keepdim=True) + 1e-6 * blk_len)
                mean_pos = (pos_c * tok_weights).sum(dim=-1, keepdim=True)
                var_c = ((pos_c - mean_pos) ** 2 * tok_weights).sum(dim=-1) * (
                    float(blk_len) / float(blk_len - 1)
                )
            else:
                var_c = torch.zeros(B, device=device, dtype=torch.float32)

            medoid_list.append(medoid_h)
            pooled_list.append(pooled_v)
            medoid_idx_list.append(global_medoid_idx)
            pos_var_list.append(var_c)
            counts_list.append(torch.full((B,), float(blk_len), device=device, dtype=torch.float32))

        medoid_tokens = torch.stack(medoid_list, dim=1)  # (B, C, D)
        pooled_values = torch.stack(pooled_list, dim=1)  # (B, C, D)
        medoid_indices = torch.stack(medoid_idx_list, dim=1)  # (B, C)
        pos_var = torch.stack(pos_var_list, dim=1)  # (B, C)
        block_counts = torch.stack(counts_list, dim=1)  # (B, C)

        return medoid_tokens, pooled_values, medoid_indices, pos_var, block_counts

    def forward(
        self,
        raw_latents: torch.Tensor,
        return_diagnostics: bool = False,
        return_stats: bool = False,
        return_telemetry: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Extracts `(B, target_tokens, D)` float32 Adelic super-tokens from `raw_latents`.

        Args:
            raw_latents: `(B, S, D)` or `(S, D)` residual stream tensor.
            return_diagnostics: If True, returns `(super_tokens, diagnostics_dict)`.
            return_stats: Alias for `return_diagnostics`.
        """
        if raw_latents.dim() == 2:
            raw_latents = raw_latents.unsqueeze(0)
        if raw_latents.dim() != 3:
            raise ValueError(f"Expected 2D or 3D raw_latents, got shape {tuple(raw_latents.shape)}")

        x = raw_latents.float()
        B, S, D = x.shape
        device = x.device

        # Ensure sequence length is at least target_tokens so Tier-1 (16) + Tier-2 (16) are well-defined
        if S < self.target_tokens:
            x = F.interpolate(
                x.transpose(1, 2), size=self.target_tokens, mode="linear", align_corners=False
            ).transpose(1, 2)
            S = self.target_tokens

        # Step 1: Compute Beltrami Triad Flux with kernel-5 max-pool neighborhood shield
        shielded_flux, raw_flux, cross_shell_curl = self.compute_beltrami_triad_flux(x)

        # Combine shielded Beltrami flux with semantic isolation score
        norm_x = F.normalize(x, p=2, dim=-1)
        sim_mat = torch.bmm(norm_x, norm_x.transpose(1, 2))  # (B, S, S)
        eye = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
        max_other_sim = sim_mat.masked_fill(eye, -1.0).max(dim=-1).values
        isolation_score = 1.0 - max_other_sim  # (B, S)

        token_norm = torch.norm(x, p=2, dim=-1)
        gated_norm = (token_norm / (token_norm.amax(dim=-1, keepdim=True) + 1e-6)) * (
            0.15 + 0.85 * cross_shell_curl
        )

        outlier_score = 3.5 * shielded_flux + 2.0 * isolation_score + gated_norm

        # Step 2: Tier-1 Outlier-Shielded Selection (K_out = 16 slots in temporal order)
        K_out = self.outlier_budget
        C = self.condensed_budget
        rem_len = S - K_out

        # Protect token 0 (BOS / task header sink) and select top K_out tokens
        score_for_topk = outlier_score.clone()
        score_for_topk[:, 0] = score_for_topk.amax(dim=-1) + 1e6
        _, topk_idx = torch.topk(score_for_topk, k=K_out, dim=-1, largest=True, sorted=False)
        outlier_idx, _ = torch.sort(topk_idx, dim=-1)  # (B, K_out) chronological order

        idx_exp = outlier_idx.unsqueeze(-1).expand(B, K_out, D)
        tier1_outliers = torch.gather(x, dim=1, index=idx_exp)  # (B, K_out, D)

        # Step 3: Extract remaining background/hay tokens for Tier-2 Ultrametric Medoid Condensation
        is_outlier = torch.zeros((B, S), dtype=torch.bool, device=device)
        is_outlier.scatter_(1, outlier_idx, True)
        all_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        rem_idx = all_idx[~is_outlier].view(B, rem_len)  # (B, rem_len)
        rem_x = torch.gather(x, dim=1, index=rem_idx.unsqueeze(-1).expand(B, rem_len, D))

        # Partition remaining tokens into C = 16 contiguous ultrametric tree blocks
        medoid_h, pooled_v, medoid_local_idx, pos_var, _ = self.pool_tensor_blocks_1d(rem_x, C)
        medoid_global_idx = torch.gather(rem_idx, dim=1, index=medoid_local_idx)  # (B, C)

        # Adelic variance attenuation factor: gamma_c = exp(-nu * Var(P_{S_c}))
        gamma_c = torch.exp(-self.viscous_nu * pos_var).clamp(min=0.05, max=1.0)  # (B, C)

        # Combine Medoid token (phase-invariant anchor) with 0.5 * (mean_pool + peak_pool)
        alpha = self.medoid_blend
        tier2_raw = alpha * medoid_h + (1.0 - alpha) * pooled_v
        tier2_medoids = tier2_raw * gamma_c.unsqueeze(-1)  # (B, C, D)

        # Concatenate [tier1_outliers, tier2_medoids] -> (B, 32, D) in float32
        super_tokens = torch.cat([tier1_outliers, tier2_medoids], dim=1).contiguous().float()

        # Compute summary Zeno flux in [0, 1] and cophenetic LCA variance
        raw_mean_flux = float(shielded_flux.mean().item())
        raw_max_flux = float(shielded_flux.max().item())
        zeno_flux = float(math.tanh(0.65 * raw_mean_flux + 0.35 * raw_max_flux))

        # Estimate cophenetic LCA variance from block similarity hierarchy
        norm_t2 = F.normalize(tier2_raw, p=2, dim=-1)
        blk_sim = torch.bmm(norm_t2, norm_t2.transpose(1, 2))
        lca_var = float(blk_sim.var(dim=(-2, -1)).mean().item() * 4.0)

        self.last_token_flux = raw_flux.detach()
        self.last_shielded_flux = shielded_flux.detach()
        self.last_cross_shell_curl = cross_shell_curl.detach()
        self.last_outlier_indices = outlier_idx.detach()
        self.last_medoid_indices = medoid_global_idx.detach()
        self.last_gamma = gamma_c.detach()
        self.last_pos_var = pos_var.detach()
        self.last_zeno_flux = zeno_flux
        self.last_lca_variance = lca_var

        if return_diagnostics or return_stats or return_telemetry:
            diag = {
                "zeno_flux": zeno_flux,
                "lca_variance": lca_var,
                "outlier_indices": outlier_idx.detach(),
                "medoid_indices": medoid_global_idx.detach(),
                "gamma": gamma_c.detach(),
                "pos_var": pos_var.detach(),
                "shielded_flux": shielded_flux.detach(),
                "cross_shell_curl": cross_shell_curl.detach(),
            }
            return super_tokens, diag

        return super_tokens


def _grid_to_feature_tokens(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """
    Encodes a 2D ARC grid (values 0..9) into a 1D sequence of 16D token feature vectors
    for fast (<1 ms) multi-scale dyadic/triadic shell flux analysis.
    """
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.zeros((1, 16), dtype=np.float32)

    h, w = arr.shape
    clipped = np.clip(arr, 0, 9)
    # Build sequence with row delimiters so 2D spatial structure is preserved in 1D stream
    n_tokens = h * (w + 1)
    feats = np.zeros((n_tokens, 16), dtype=np.float32)

    r_idx = np.repeat(np.arange(h, dtype=np.float32) / max(1.0, float(h)), w + 1)
    c_seq = np.tile(np.arange(w + 1, dtype=np.float32) / max(1.0, float(w)), h)
    feats[:, 10] = r_idx
    feats[:, 11] = c_seq

    pos = 0
    for r in range(h):
        row_colors = clipped[r]
        for c in range(w):
            color = int(row_colors[c])
            feats[pos, color] = 1.0
            feats[pos, 12] = 1.0 if color > 0 else 0.0  # foreground indicator
            # Local 2D gradient indicator
            up_diff = 1.0 if (r > 0 and clipped[r - 1, c] != color) else 0.0
            left_diff = 1.0 if (c > 0 and clipped[r, c - 1] != color) else 0.0
            feats[pos, 13] = 0.5 * (up_diff + left_diff)
            pos += 1
        # Row delimiter token
        feats[pos, 14] = 1.0
        pos += 1

    return feats


def _compute_prime_shell_flux_and_lca(seq_feats: np.ndarray, p: int) -> Tuple[float, float, float]:
    """
    Computes multi-scale shell Beltrami flux and cophenetic LCA depth variance
    for prime arity `p` (e.g. p=2 dyadic or p=3 triadic) on `seq_feats` of shape `(T, D)`.
    """
    T, D = seq_feats.shape
    if T < 2:
        return 0.0, 0.0, 0.0

    # Fine shell: local token representation + 1-step difference
    u_fine = seq_feats.astype(np.float32)
    fine_norms = np.linalg.norm(u_fine, axis=-1, keepdims=True)
    u_hat_fine = u_fine / np.maximum(fine_norms, 1e-6)

    # Coarse shell: block-averaged representation at scale p^2
    blk_size = max(2, p * p)
    n_blocks = int(math.ceil(T / float(blk_size)))
    pad_len = n_blocks * blk_size - T
    if pad_len > 0:
        padded = np.pad(u_fine, ((0, pad_len), (0, 0)), mode="edge")
    else:
        padded = u_fine
    blk_means = padded.reshape(n_blocks, blk_size, D).mean(axis=1)  # (n_blocks, D)
    u_coarse = np.repeat(blk_means, blk_size, axis=0)[:T]
    coarse_norms = np.linalg.norm(u_coarse, axis=-1, keepdims=True)
    u_hat_coarse = u_coarse / np.maximum(coarse_norms, 1e-6)

    # Cross-shell exterior curl: sqrt(1 - <u_hat_coarse, u_hat_fine>^2)
    cos_cf = np.clip(np.sum(u_hat_coarse * u_hat_fine, axis=-1), -1.0, 1.0)
    curl = np.sqrt(np.maximum(0.0, 1.0 - cos_cf * cos_cf))

    # Fine-shell innovation: ||u_hat_fine(t) - u_hat_fine(t-1)||_2
    diff = np.linalg.norm(u_hat_fine[1:] - u_hat_fine[:-1], axis=-1)
    innov = np.zeros(T, dtype=np.float32)
    innov[1:] = diff
    innov[:-1] = np.maximum(innov[:-1], diff)

    beltrami_flux = curl * innov
    mean_flux = float(np.mean(beltrami_flux))
    max_spike = float(np.max(beltrami_flux))

    # Cophenetic LCA depth variance across p-adic tree levels
    levels = max(2, int(math.ceil(math.log(max(n_blocks, 2), p))))
    blk_norms = np.linalg.norm(blk_means, axis=-1, keepdims=True)
    blk_unit = blk_means / np.maximum(blk_norms, 1e-6)

    # Soft branch assignments across levels l = 0 .. levels-1
    # Project block descriptors onto p orthogonal directions per level
    cophenetic_agree = np.zeros((n_blocks, n_blocks, levels), dtype=np.float32)
    for ell in range(levels):
        # Deterministic frequency slice per level
        cols = [(ell * p + b) % D for b in range(p)]
        logits = blk_unit[:, cols] * (2.5 * (ell + 1))
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.maximum(np.sum(exp_l, axis=-1, keepdims=True), 1e-8)  # (NB, p)
        cophenetic_agree[:, :, ell] = probs @ probs.T

    # Cumulative minimum from root (level 0) to leaf (level L-1), matching compute_expected_block_distance
    prefix_min = np.minimum.accumulate(cophenetic_agree, axis=-1)
    expected_dist = float(levels) - np.sum(prefix_min, axis=-1)  # (NB, NB)
    lca_var = float(np.var(expected_dist))

    return mean_flux, max_spike, lca_var


def compute_grid_prompt_zeno_flux(
    demo_pairs: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
    test_inputs: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """
    Computes deterministic multi-scale dyadic (p=2) and triadic (p=3) shell flux (`zeno_flux`)
    and cophenetic LCA depth variance (`lca_variance`) directly from ARC demonstration pairs
    and test input grids in <1 ms (without requiring Llama-3.1-8B in GPU memory).

    STRICT LEAKAGE GUARD: Only demonstration (`train`) pairs and test `input` grids are
    inspected; test `output` grids are never accessed.

    Args:
        demo_pairs: Either a list of `{"input": grid, "output": grid}` dicts, or a full
                    ARC task dictionary `{"train": [...], "test": [...]}`.
        test_inputs: Optional list of test input 2D grids (or `[{"input": grid}]` dicts).

    Returns:
        Dictionary containing:
          - `zeno_flux`: Normalized Beltrami triad turbulence in `[0.0, 1.0]`
          - `lca_variance`: Multi-prime cophenetic LCA depth variance `>= 0.0`
          - `dyadic_flux_p2`: Raw dyadic (p=2) Beltrami flux
          - `triadic_flux_p3`: Raw triadic (p=3) Beltrami flux
          - `max_beltrami_spike`: Peak kernel-shielded Beltrami spike across the prompt
          - `rule_complexity`: Structural input-to-output rule complexity score in `[0.0, 1.0]`
          - `num_tokens`: Total serialized grid tokens analyzed
    """
    if isinstance(demo_pairs, Mapping):
        train_list = demo_pairs.get("train", demo_pairs.get("train_pairs", []))
        if test_inputs is None and "test" in demo_pairs:
            test_inputs = [
                tp["input"] if isinstance(tp, Mapping) and "input" in tp else tp
                for tp in demo_pairs["test"]
            ]
    else:
        train_list = list(demo_pairs) if demo_pairs is not None else []

    token_blocks: List[np.ndarray] = []
    rule_scores: List[float] = []

    for pair in train_list:
        inp = pair.get("input", []) if isinstance(pair, Mapping) else []
        out = pair.get("output", []) if isinstance(pair, Mapping) else []
        inp_tok = _grid_to_feature_tokens(inp)
        out_tok = _grid_to_feature_tokens(out)

        # Transition delimiter token between [Input Grid] and [Output Grid]
        delim = np.zeros((1, 16), dtype=np.float32)
        delim[0, 15] = 1.0
        token_blocks.extend([inp_tok, delim, out_tok, delim])

        # Measure structural transformation complexity between demo input and output
        inp_arr = np.asarray(inp, dtype=np.int32)
        out_arr = np.asarray(out, dtype=np.int32)
        if inp_arr.size > 0 and out_arr.size > 0:
            shape_changed = 1.0 if inp_arr.shape != out_arr.shape else 0.0
            inp_colors = set(np.unique(inp_arr).tolist()) - {0}
            out_colors = set(np.unique(out_arr).tolist()) - {0}
            new_colors = 1.0 if not out_colors.issubset(inp_colors) else 0.0
            fg_in = float(np.mean(inp_arr > 0))
            fg_out = float(np.mean(out_arr > 0))
            fg_delta = abs(fg_out - fg_in)
            if inp_arr.shape == out_arr.shape:
                cell_change_rate = float(np.mean(inp_arr != out_arr))
            else:
                cell_change_rate = 0.65
            rule_scores.append(
                0.35 * cell_change_rate + 0.25 * shape_changed + 0.20 * new_colors + 0.20 * fg_delta
            )

    if test_inputs is not None:
        # Handle a single 2D grid passed directly vs a list of grids
        if (
            isinstance(test_inputs, (list, tuple))
            and len(test_inputs) > 0
            and isinstance(test_inputs[0], (list, tuple))
            and (len(test_inputs[0]) == 0 or isinstance(test_inputs[0][0], (int, np.integer)))
        ):
            test_grid_list = [test_inputs]
        else:
            test_grid_list = list(test_inputs)

        for tp in test_grid_list:
            g = tp.get("input", []) if isinstance(tp, Mapping) else tp
            if g is not None and len(g) > 0:
                t_tok = _grid_to_feature_tokens(g)
                delim = np.zeros((1, 16), dtype=np.float32)
                delim[0, 15] = 1.0
                token_blocks.extend([t_tok, delim])

    if not token_blocks:
        return {
            "zeno_flux": 0.0,
            "lca_variance": 0.0,
            "dyadic_flux_p2": 0.0,
            "triadic_flux_p3": 0.0,
            "max_beltrami_spike": 0.0,
            "rule_complexity": 0.0,
            "num_tokens": 0,
        }

    full_seq = np.concatenate(token_blocks, axis=0)
    flux_p2, spike_p2, lca_p2 = _compute_prime_shell_flux_and_lca(full_seq, p=2)
    flux_p3, spike_p3, lca_p3 = _compute_prime_shell_flux_and_lca(full_seq, p=3)

    mean_raw_flux = 0.5 * (flux_p2 + flux_p3)
    max_spike = max(spike_p2, spike_p3)
    rule_comp = float(np.mean(rule_scores)) if rule_scores else 0.25

    # Calibrated composite Zeno flux in [0.0, 1.0]
    zeno_flux = float(
        np.clip(0.55 * math.tanh(2.2 * mean_raw_flux) + 0.45 * rule_comp, 0.0, 1.0)
    )
    lca_variance = float(0.5 * (lca_p2 + lca_p3))

    return {
        "zeno_flux": zeno_flux,
        "lca_variance": lca_variance,
        "dyadic_flux_p2": flux_p2,
        "triadic_flux_p3": flux_p3,
        "max_beltrami_spike": max_spike,
        "rule_complexity": rule_comp,
        "num_tokens": int(full_seq.shape[0]),
    }


def extract_beltrami_lh_latents(
    prompt: Union[str, torch.Tensor],
    model: Optional[nn.Module] = None,
    tokenizer: Any = None,
    shell_extractor: Optional[BeltramiCallosalShell] = None,
    hook_layer: int = 16,
    target_tokens: int = 32,
    d_model: int = 4096,
    device: str = "cpu",
    mock_mode: bool = False,
    return_stats: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Drop-in Beltrami-Adelic upgrade for `brain_ai.models.lh_cache.extract_lh_latents`.
    Extracts Layer-16 residual stream states `(1, S, d_model)` and condenses them into
    `(1, 32, d_model)` via `BeltramiCallosalShell` instead of uniform `adaptive_avg_pool1d`.
    Also accepts a pre-extracted hidden-state `torch.Tensor` directly as `prompt`.
    """
    if isinstance(prompt, torch.Tensor):
        raw_latents = prompt.unsqueeze(0) if prompt.dim() == 2 else prompt
        d_model = int(raw_latents.size(-1))
        target_dev = str(raw_latents.device)
        if shell_extractor is None:
            shell_extractor = BeltramiCallosalShell(
                d_model=d_model,
                target_tokens=target_tokens,
                outlier_budget=target_tokens // 2,
                condensed_budget=target_tokens - (target_tokens // 2),
                hook_layer=hook_layer,
            ).to(raw_latents.device)
        with torch.no_grad():
            super_tokens, diag = shell_extractor(raw_latents.float(), return_diagnostics=True)
        return (super_tokens.to(target_dev), diag) if return_stats else super_tokens.to(target_dev)

    if shell_extractor is None:
        shell_extractor = BeltramiCallosalShell(
            d_model=d_model,
            target_tokens=target_tokens,
            outlier_budget=target_tokens // 2,
            condensed_budget=target_tokens - (target_tokens // 2),
            hook_layer=hook_layer,
        ).to(device)

    if mock_mode or model is None or getattr(model, "mock_mode", False):
        h = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], byteorder="big")
        g = torch.Generator().manual_seed(seed)
        seq_len = max(target_tokens * 4, min(512, max(64, len(prompt) // 4)))
        raw_latents = torch.randn(1, seq_len, d_model, generator=g, dtype=torch.float32)
        raw_latents = F.layer_norm(raw_latents, (d_model,)).to(device)
    else:
        try:
            model_device = next(model.parameters()).device
        except (StopIteration, Exception):
            model_device = torch.device(device)

        model.eval()
        if tokenizer is not None:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(model_device)
            attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model_device)
        else:
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=model_device)
            attn_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            if hasattr(model, "extract_residual_stream"):
                raw_latents = model.extract_residual_stream(input_ids, attention_mask=attn_mask)
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                layer_idx = min(hook_layer + 1, len(outputs.hidden_states) - 1)
                raw_latents = outputs.hidden_states[layer_idx]
            else:
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                raw_latents = out.get(
                    "residual_latent", out.logits if hasattr(out, "logits") else out[0]
                )

        if raw_latents.dim() == 2:
            raw_latents = raw_latents.unsqueeze(0)
        raw_latents = raw_latents.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        super_tokens, diag = shell_extractor(raw_latents, return_diagnostics=True)

    return (super_tokens.to(device), diag) if return_stats else super_tokens.to(device)
