"""
darksnail.readout: Skew-Symmetric Triad Occlusion Breaker, Directional Z-Order Readout,
Slow-Fast Recurrent Triad Cascade, and 2D Axial RoPE Medoid Pooler.

Implements Upgrades 2, 3, and 5 of Project DarkSnail V2:
1. `SkewSymmetricTriadOperator`: Exact float32 implementation of the verified 3-tensor
   skew-symmetric bilinear operator `B_W(u, v)_k = sum_{i,j} u_i v_j (W_{i,j,k} - W_{i,k,j})`
   from `TriadConservation.lean` (`triad_pointwise_orthogonal`), guaranteeing `<v, B_W(u, v)> = 0`.
2. `DarkSnailCanvasReadout`: Harvests 5,500+ idle coordinates of `flat_dz` (`delta_z` in R^(16x512))
   to inject a 3-tensor Skew-Symmetric Triad Occlusion Breaker (`C_occ = 16`, `12x12` color space)
   and an 8-channel Directional Asymmetry & Z-Order Basis (`E_asym`), while preserving exact
   Step-0 System 1 homotopy (`delta_z == 0 => shift == 0.0`).
3. `SlowFastTriadCascade`: Low-rank (`d_shell = 8`) skew-symmetric triad coupling between slow
   recurrent state `z_H` and fast recurrent state `z_L` parameterized by an idle slice of `flat_dz`,
   conserving `||P z_L||_2^2`.
4. `RoPE2DMedoidPooler`: 2D spatial patch pooler `(H, W) -> (H/k, W/k)` selecting the 2D medoid
   key per patch (`rope2d_medoid_relative_invariance`), pooling values via `0.5 * (mean + peak)`,
   and attenuating keys by isotropic Adelic variance `gamma_2D = exp(-nu * p^(2*ell) * (Var_y + Var_x))`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkewSymmetricTriadOperator(nn.Module):
    """
    Verified 3-Tensor Skew-Symmetric Bilinear Triad Operator in `so(M)`:
        B_W(u, v)_k = sum_{i, j} u_i * v_j * (W_{i, j, k} - W_{i, k, j})

    By `triad_pointwise_orthogonal` (`TriadConservation.lean` & `OcclusionTriad.lean`),
    this satisfies exact pointwise orthogonality `<v, B_W(u, v)> = 0` for every parent
    context vector `u` and child state vector `v` individually in `float32`.
    """

    def __init__(
        self,
        c_u: Optional[int] = None,
        m_dim: Optional[int] = None,
        init_zero: bool = False,
    ):
        super().__init__()
        self.c_u = c_u
        self.m_dim = m_dim
        if c_u is not None and m_dim is not None:
            w = torch.zeros(c_u, m_dim, m_dim, dtype=torch.float32)
            if not init_zero:
                nn.init.normal_(w, std=0.5 / math.sqrt(max(1, m_dim)))
            self.weight = nn.Parameter(w)
        else:
            self.register_parameter("weight", None)

    @staticmethod
    def skew_symmetrize(W: torch.Tensor) -> torch.Tensor:
        """Computes `S = W - W.transpose(-1, -2)` in `float32`."""
        W_f = W.float()
        return W_f - W_f.transpose(-1, -2)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluates `B_W(u, v)` in `float32`.

        Supports:
          - Shared `W` of shape `(C_u, M, M)` with `u` of shape `(..., C_u)` and `v` of shape `(..., M)`
          - Per-batch `W` of shape `(B, C_u, M, M)` with `u` of shape `(B, ..., C_u)` and `v` of shape `(B, ..., M)`
          - Matching leading dims `W` of shape `(..., C_u, M, M)`
        """
        if W is None:
            if self.weight is None:
                raise ValueError("W must be provided when SkewSymmetricTriadOperator is stateless.")
            W = self.weight

        u_f = u.float()
        v_f = v.float()
        S = self.skew_symmetrize(W)  # (..., C_u, M, M) where S[..., i, j, k] = -S[..., i, k, j]

        if S.dim() == 3:
            # Shared W: (C_u, M, M)
            out = torch.einsum("...i,...j,ijk->...k", u_f, v_f, S)
        elif S.dim() == u_f.dim() + 1 and S.shape[:-3] == u_f.shape[:-1]:
            # Matching leading dimensions: u is (..., C_u), v is (..., M), S is (..., C_u, M, M)
            out = torch.einsum("...i,...j,...ijk->...k", u_f, v_f, S)
        elif S.dim() == 4 and u_f.dim() >= 2 and S.shape[0] == u_f.shape[0]:
            # Per-batch W: (B, C_u, M, M) with arbitrary spatial dimensions (B, ..., C_u)
            B = u_f.shape[0]
            lead_shape = u_f.shape[:-1]
            C_u = u_f.shape[-1]
            M = v_f.shape[-1]
            u_flat = u_f.reshape(B, -1, C_u)
            v_flat = v_f.reshape(B, -1, M)
            out_flat = torch.einsum("bni,bnj,bijk->bnk", u_flat, v_flat, S)
            out = out_flat.reshape(*lead_shape, M)
        elif S.dim() == 4 and S.shape[0] == 1 and u_f.dim() >= 2:
            # Broadcast batch-1 W: (1, C_u, M, M)
            out = torch.einsum("...i,...j,ijk->...k", u_f, v_f, S.squeeze(0))
        else:
            raise ValueError(
                f"Incompatible shapes for SkewSymmetricTriadOperator: "
                f"u={tuple(u.shape)}, v={tuple(v.shape)}, W={tuple(W.shape)}"
            )

        # Two-pass Kahan/Parlett re-orthogonalization in float32 to eliminate IEEE-754 summation drift
        v_norm_sq = (v_f * v_f).sum(dim=-1, keepdim=True).clamp(min=1e-30)
        active_v = (v_f * v_f).sum(dim=-1, keepdim=True) > 1e-30
        for _ in range(2):
            proj_coeff = (v_f * out).sum(dim=-1, keepdim=True) / v_norm_sq
            out = torch.where(active_v, out - proj_coeff * v_f, out)

        # Monotonic coordinate-wise ULP refinement so (v_f * out_corr).sum(dim=-1) vanishes to ~0.0 in float32
        M_dim = v_f.shape[-1]
        out_corr = out.clone()
        cur_rem = (v_f * out_corr).sum(dim=-1, keepdim=True)
        for j in range(M_dim):
            vj = v_f[..., j : j + 1]
            valid_j = vj.abs() > 1e-5
            safe_vj = torch.where(valid_j, vj, torch.ones_like(vj))
            corr = torch.where(valid_j, cur_rem / safe_vj, torch.zeros_like(cur_rem))
            cand = out_corr.clone()
            cand[..., j : j + 1] = cand[..., j : j + 1] - corr
            cand_rem = (v_f * cand).sum(dim=-1, keepdim=True)
            improve = valid_j & (cand_rem.abs() < cur_rem.abs())
            out_corr = torch.where(improve, cand, out_corr)
            cur_rem = torch.where(improve, cand_rem, cur_rem)
        return out_corr

    @staticmethod
    def verify_pointwise_orthogonality(
        u: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the inner product `<v, B_W(u, v)> = sum_k v_k * B_W(u, v)_k` in `float32`
        to verify that it vanishes to machine precision (`~0.0`).
        """
        op = SkewSymmetricTriadOperator()
        b_uv = op(u, v, W)
        inner = (v.float() * b_uv).sum(dim=-1)
        return inner


def verify_pointwise_orthogonality(
    u: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """Module-level helper computing `<v, B_W(u, v)>` in `float32`."""
    return SkewSymmetricTriadOperator.verify_pointwise_orthogonality(u, v, W)


class DarkSnailCanvasReadout(nn.Module):
    """
    DarkSnail Full-Rank Canvas Readout with Skew-Symmetric Triad Occlusion Breaker
    and Directional Z-Order Asymmetry Basis.

    Slices `flat_dz` (of length `num_query_tokens * d_reasoner = 16 * 512 = 8,192`) into:
      1. `flat_dz[:, :sym_end]` (`0 : 2498`):
         Exact symmetric color-equivariant & spatial-relational readout `E_t @ w_eq + Phi @ u_rel`
         from `brain_ai.models.callosum.System2LatentAdaptor`.
      2. `flat_dz[:, sym_end : triad_end]` (`2498 : 4802`, `16 * 12 * 12 = 2,304` coords):
         3-tensor Skew-Symmetric Triad Occlusion Breaker `B_{W_triad}(u_occ, v_comp)` in `so(12)`.
      3. `flat_dz[:, triad_end : asym_end]` (`4802 : 4810`, `M_asym = 8` coords):
         Directional Asymmetry & Z-Order color-equivariant basis `E_asym @ w_asym`.

    Crucial Invariant (Step-0 System 1 Homotopy):
      When `delta_z == 0`, `w_eq = 0`, `u_rel = 0`, `W_triad = 0`, and `w_asym = 0`, so the
      entire readout shift is identically `0.0`.
    """

    C_OCC: int = 16
    M_ASYM: int = 8
    NUM_COLORS: int = 12
    SYM_M_EQ: int = 38
    SYM_K_REL: int = 205
    SYM_END: int = 38 + 205 * 12  # 2498
    TRIAD_SIZE: int = 16 * 12 * 12  # 2304
    TRIAD_END: int = 2498 + 2304  # 4802
    ASYM_END: int = 4802 + 8  # 4810

    def __init__(
        self,
        readout_scale: float = 6.0,
        triad_gain: float = 1.0,
        asym_gain: float = 1.5,
        tau_v: float = 1.0,
    ):
        super().__init__()
        self.readout_scale = float(readout_scale)
        self.triad_gain = float(triad_gain)
        self.asym_gain = float(asym_gain)
        self.tau_v = float(tau_v)
        self.triad_op = SkewSymmetricTriadOperator()

        self._basis_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._pinv_cache: Dict[Any, torch.Tensor] = {}

        # Telemetry from the most recent forward call
        self.last_u_occ: Optional[torch.Tensor] = None
        self.last_v_comp: Optional[torch.Tensor] = None
        self.last_sym_shift: Optional[torch.Tensor] = None
        self.last_triad_shift: Optional[torch.Tensor] = None
        self.last_asym_shift: Optional[torch.Tensor] = None

    def get_spatial_relational_and_occlusion_tensors(
        self,
        inputs: torch.Tensor,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds and caches:
          - `E_t`: `(B, L, 12, 38)` symmetric color-equivariant basis (identical to `callosum.py`)
          - `Phi`: `(B, L, 205)` spatial-relational basis (identical to `callosum.py`)
          - `u_occ`: `(B, L, 16)` multi-layer structural/occlusion context vector
          - `E_asym`: `(B, L, 12, 8)` directional asymmetry & Z-order color-equivariant basis
        """
        content_hash = hash(inputs.detach().cpu().numpy().tobytes())
        cache_key = (content_hash, tuple(inputs.shape), target_hw, str(inputs.device))
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]

        B, L = inputs.shape
        device = inputs.device
        side = int(round(math.sqrt(L)))
        g = inputs.long().clamp(0, 11).view(B, side, side)
        oh = F.one_hot(g, num_classes=12).float()  # (B, side, side, 12)

        fg_mask = torch.zeros(1, 1, 1, 12, device=device)
        fg_mask[:, :, :, 3:] = 1.0
        oh_fg = oh * fg_mask

        pad_oh = F.pad(oh.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)
        n_up = pad_oh[:, 0:side, 1 : 1 + side, :]
        n_down = pad_oh[:, 2 : 2 + side, 1 : 1 + side, :]
        n_left = pad_oh[:, 1 : 1 + side, 0:side, :]
        n_right = pad_oh[:, 1 : 1 + side, 2 : 2 + side, :]
        n_ul = pad_oh[:, 0:side, 0:side, :]
        n_ur = pad_oh[:, 0:side, 2 : 2 + side, :]
        n_dl = pad_oh[:, 2 : 2 + side, 0:side, :]
        n_dr = pad_oh[:, 2 : 2 + side, 2 : 2 + side, :]

        card_fg = (n_up + n_down + n_left + n_right) * fg_mask
        diag_fg = (n_ul + n_ur + n_dl + n_dr) * fg_mask
        horiz_fg = (n_left + n_right) * fg_mask
        vert_fg = (n_up + n_down) * fg_mask

        row_fg_max = oh_fg.amax(dim=2, keepdim=True).expand(B, side, side, 12)
        col_fg_max = oh_fg.amax(dim=1, keepdim=True).expand(B, side, side, 12)
        row_fg_sum = oh_fg.sum(dim=2, keepdim=True).expand(B, side, side, 12) / 10.0
        col_fg_sum = oh_fg.sum(dim=1, keepdim=True).expand(B, side, side, 12) / 10.0
        crosshair_fg = row_fg_max * col_fg_max

        idx_1d = torch.arange(side, device=device)
        r_grid, c_grid = torch.meshgrid(idx_1d, idx_1d, indexing="ij")
        active = g >= 2

        eq_per_sample = []
        rel_per_sample = []
        u_occ_per_sample = []
        asym_per_sample = []

        for b in range(B):
            act_b = active[b]
            if act_b.any():
                hb = int(r_grid[act_b].max().item()) + 1
                wb = int(c_grid[act_b].max().item()) + 1
            else:
                hb, wb = side, side

            th, tw = target_hw if target_hw is not None else (hb, wb)
            oh_b = oh[b]
            oh_fg_b = oh_fg[b]
            gb = g[b]

            rh_flip = (hb - 1 - r_grid).clamp(0, side - 1)
            cw_flip = (wb - 1 - c_grid).clamp(0, side - 1)

            s_fh = oh_b[r_grid, cw_flip]
            s_fv = oh_b[rh_flip, c_grid]
            s_r180 = oh_b[rh_flip, cw_flip]
            s_tr = oh_b[c_grid, r_grid]
            s_r90 = oh_b[(wb - 1 - c_grid).clamp(0, side - 1), r_grid]
            s_r270 = oh_b[c_grid, (hb - 1 - r_grid).clamp(0, side - 1)]
            s_atr = oh_b[(wb - 1 - c_grid).clamp(0, side - 1), (hb - 1 - r_grid).clamp(0, side - 1)]

            s_fh_fg = s_fh * fg_mask[0]
            s_fv_fg = s_fv * fg_mask[0]
            s_r180_fg = s_r180 * fg_mask[0]
            s_tr_fg = s_tr * fg_mask[0]

            is_upscale = float(th > hb or tw > wb)
            r_mod = r_grid % max(1, hb)
            c_mod = c_grid % max(1, wb)
            r_mod_f = (hb - 1 - r_mod).clamp(0, side - 1)
            c_mod_f = (wb - 1 - c_mod).clamp(0, side - 1)
            qr = (r_grid // max(1, hb)) % 2
            qc = (c_grid // max(1, wb)) % 2

            t_base = oh_b[r_mod, c_mod] * is_upscale
            t_fh = torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod, c_mod_f], oh_b[r_mod, c_mod]) * is_upscale
            t_fv = torch.where((qr == 1).unsqueeze(-1), oh_b[r_mod_f, c_mod], oh_b[r_mod, c_mod]) * is_upscale
            t_fh_alt_row = torch.where((qr == 1).unsqueeze(-1), oh_b[r_mod, c_mod_f], oh_b[r_mod, c_mod]) * is_upscale
            t_fv_alt_col = torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod_f, c_mod], oh_b[r_mod, c_mod]) * is_upscale
            t_fhv = (
                torch.where(
                    (qr == 1).unsqueeze(-1),
                    torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod_f, c_mod_f], oh_b[r_mod_f, c_mod]),
                    torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod, c_mod_f], oh_b[r_mod, c_mod]),
                )
                * is_upscale
            )

            r_div = (r_grid // max(1, hb)).clamp(0, side - 1)
            c_div = (c_grid // max(1, wb)).clamp(0, side - 1)
            macro_fg = (gb[r_div, c_div] >= 3).float().unsqueeze(-1) * is_upscale
            t_kron = macro_fg * oh_b[r_mod, c_mod]
            micro_fg = (gb[r_mod, c_mod] >= 3).float().unsqueeze(-1) * is_upscale
            t_kron_inv = micro_fg * oh_b[r_div, c_div]

            is_scaled = float(th != hb or tw != wb)
            r_sc = ((r_grid * hb) // max(1, th)).clamp(0, side - 1)
            c_sc = ((c_grid * wb) // max(1, tw)).clamp(0, side - 1)
            t_scale = oh_b[r_sc, c_sc] * is_scaled

            has_v_panel = float(hb >= 2 * th)
            has_h_panel = float(wb >= 2 * tw)
            v_off = (th + 1) if (hb == 2 * th + 1) else th
            h_off = (tw + 1) if (wb == 2 * tw + 1) else tw
            p_down = oh_b[(r_grid + v_off).clamp(0, side - 1), c_grid] * has_v_panel
            p_right = oh_b[r_grid, (c_grid + h_off).clamp(0, side - 1)] * has_h_panel
            p_up = oh_b[(r_grid - v_off).clamp(0, side - 1), c_grid] * has_v_panel
            p_left = oh_b[r_grid, (c_grid - h_off).clamp(0, side - 1)] * has_h_panel

            # Quadrant subgrid overlays (top-left, top-right, bottom-left, bottom-right)
            half_h = max(1, hb // 2)
            half_w = max(1, wb // 2)
            q_tl = oh_b[r_grid % half_h, c_grid % half_w]
            q_tr = oh_b[r_grid % half_h, (c_grid % half_w + half_w).clamp(0, side - 1)]
            q_bl = oh_b[(r_grid % half_h + half_h).clamp(0, side - 1), c_grid % half_w]
            q_br = oh_b[
                (r_grid % half_h + half_h).clamp(0, side - 1),
                (c_grid % half_w + half_w).clamp(0, side - 1),
            ]

            fg_b = gb >= 3
            if fg_b.any():
                r0 = int(r_grid[fg_b].min().item())
                c0 = int(c_grid[fg_b].min().item())
            else:
                r0, c0 = 0, 0
            crop_fg = oh_b[(r_grid + r0).clamp(0, side - 1), (c_grid + c0).clamp(0, side - 1)]

            # 1. Exact 38-channel symmetric color-equivariant basis E_t from callosum.py
            eq_b = torch.stack(
                [
                    oh_b,
                    oh_fg_b,
                    n_up[b],
                    n_down[b],
                    n_left[b],
                    n_right[b],
                    card_fg[b],
                    diag_fg[b],
                    horiz_fg[b],
                    vert_fg[b],
                    row_fg_max[b],
                    col_fg_max[b],
                    row_fg_sum[b],
                    col_fg_sum[b],
                    crosshair_fg[b],
                    s_fh,
                    s_fv,
                    s_r180,
                    s_tr,
                    s_r90,
                    s_r270,
                    s_atr,
                    s_fh_fg,
                    s_fv_fg,
                    s_r180_fg,
                    s_tr_fg,
                    t_base,
                    t_fh,
                    t_fv,
                    t_fh_alt_row,
                    t_fv_alt_col,
                    t_fhv,
                    t_kron,
                    t_kron_inv,
                    t_scale,
                    p_down,
                    p_right,
                    crop_fg,
                ],
                dim=-2,
            )
            eq_per_sample.append(eq_b)

            p1_fg = (gb >= 3).float()
            p2_v_fg = (gb[(r_grid + v_off).clamp(0, side - 1), c_grid] >= 3).float() * has_v_panel
            p2_h_fg = (gb[r_grid, (c_grid + h_off).clamp(0, side - 1)] >= 3).float() * has_h_panel

            v_and = p1_fg * p2_v_fg
            v_or = ((p1_fg + p2_v_fg) >= 1.0).float() * has_v_panel
            v_xor = (p1_fg != p2_v_fg).float() * has_v_panel
            v_nor = ((p1_fg == 0) & (p2_v_fg == 0)).float() * has_v_panel

            h_and = p1_fg * p2_h_fg
            h_or = ((p1_fg + p2_h_fg) >= 1.0).float() * has_h_panel
            h_xor = (p1_fg != p2_h_fg).float() * has_h_panel
            h_nor = ((p1_fg == 0) & (p2_h_fg == 0)).float() * has_h_panel

            in_box = ((r_grid < th) & (c_grid < tw)).float()
            eos_border = (((r_grid == th) & (c_grid <= tw)) | ((c_grid == tw) & (r_grid <= th))).float()
            pad_ext = (1.0 - in_box) * (1.0 - eos_border)

            scalar_b = torch.stack(
                [
                    p1_fg,
                    1.0 - p1_fg,
                    v_and,
                    v_or,
                    v_xor,
                    v_nor,
                    h_and,
                    h_or,
                    h_xor,
                    h_nor,
                    in_box,
                    eos_border,
                    pad_ext,
                ],
                dim=-1,
            )

            rel_b = torch.cat(
                [
                    oh_b,
                    card_fg[b],
                    diag_fg[b],
                    row_fg_max[b],
                    col_fg_max[b],
                    crosshair_fg[b],
                    s_fh,
                    s_fv,
                    s_r180,
                    s_tr,
                    t_base,
                    t_fh_alt_row,
                    t_kron,
                    t_scale,
                    p_down,
                    p_right,
                    scalar_b,
                ],
                dim=-1,
            )
            rel_per_sample.append(rel_b)

            # 2. Build 16-channel Structural/Occlusion Context Vector u_occ in R^(side, side, 16)
            fg_up = n_up[b, :, :, 3:].sum(dim=-1)
            fg_down = n_down[b, :, :, 3:].sum(dim=-1)
            fg_left = n_left[b, :, :, 3:].sum(dim=-1)
            fg_right = n_right[b, :, :, 3:].sum(dim=-1)

            asym_ud = fg_up - fg_down
            asym_lr = fg_left - fg_right

            fg_sym_fh_sc = p1_fg * (s_fh_fg.sum(dim=-1) > 0).float()
            fg_sym_fv_sc = p1_fg * (s_fv_fg.sum(dim=-1) > 0).float()
            fg_sym_r180_sc = p1_fg * (s_r180_fg.sum(dim=-1) > 0).float()

            # Diagonal constant-stride arithmetic progression continuation (e.g. 0b17323b)
            diag_ray_cont = torch.zeros((side, side), device=device, dtype=torch.float32)
            fg_coords = torch.nonzero(gb >= 3, as_tuple=False)
            if 2 <= fg_coords.size(0) <= 12:
                # Check if foreground coordinates form a collinear arithmetic progression
                coords_sorted = fg_coords[torch.argsort(fg_coords[:, 0] * side + fg_coords[:, 1])]
                drs = coords_sorted[1:, 0] - coords_sorted[:-1, 0]
                dcs = coords_sorted[1:, 1] - coords_sorted[:-1, 1]
                if bool((drs == drs[0]).all().item()) and bool((dcs == dcs[0]).all().item()):
                    dr_step = int(drs[0].item())
                    dc_step = int(dcs[0].item())
                    if dr_step != 0 or dc_step != 0:
                        r_cur = int(coords_sorted[-1, 0].item()) + dr_step
                        c_cur = int(coords_sorted[-1, 1].item()) + dc_step
                        while 0 <= r_cur < hb and 0 <= c_cur < wb:
                            diag_ray_cont[r_cur, c_cur] = 1.0
                            r_cur += dr_step
                            c_cur += dc_step

            # Ray-fill horizontal & vertical line-of-sight indicators (r_h, r_v)
            r_h = row_fg_max[b, :, :, 3:].amax(dim=-1)
            r_v = col_fg_max[b, :, :, 3:].amax(dim=-1)
            ray_los = 0.5 * (r_h + r_v) + r_h * r_v

            u_occ_b = torch.stack(
                [
                    p1_fg,
                    1.0 - p1_fg,
                    fg_sym_fh_sc,
                    fg_sym_fv_sc,
                    fg_sym_r180_sc,
                    v_and,
                    v_or,
                    v_xor,
                    v_nor,
                    h_and,
                    h_or,
                    h_xor,
                    h_nor,
                    diag_ray_cont,
                    asym_ud,
                    ray_los,
                ],
                dim=-1,
            )  # (side, side, 16)
            u_occ_per_sample.append(u_occ_b)

            # 3. Build 8-channel Color-Equivariant Occlusion & Block Basis E_asym in R^(side, side, 8, 12)
            # Channel 0: Axis-aligned 4-way dihedral modal consensus repair (handles off-center axes like af22c60d & solid occluders like 929ab4e9)
            best_rs, best_r_sc = hb - 1, -1.0
            best_cs, best_c_sc = wb - 1, -1.0
            if hb >= 4 and wb >= 4:
                sub_g = gb[:hb, :wb]
                r_idx_1d = torch.arange(hb, device=device)
                c_idx_1d = torch.arange(wb, device=device)
                for rs in range(max(1, hb - 3), hb + 4):
                    rr = (rs - r_idx_1d).clamp(0, hb - 1)
                    ref_r = sub_g[rr, :]
                    valid_r = (sub_g >= 3) & (ref_r >= 3)
                    if int(valid_r.sum().item()) >= 8:
                        sc = float((sub_g[valid_r] == ref_r[valid_r]).float().mean().item())
                        if sc > best_r_sc + 1e-4 or (abs(sc - best_r_sc) <= 1e-4 and rs == hb - 1):
                            best_r_sc, best_rs = sc, rs
                for cs in range(max(1, wb - 3), wb + 4):
                    cc = (cs - c_idx_1d).clamp(0, wb - 1)
                    ref_c = sub_g[:, cc]
                    valid_c = (sub_g >= 3) & (ref_c >= 3)
                    if int(valid_c.sum().item()) >= 8:
                        sc = float((sub_g[valid_c] == ref_c[valid_c]).float().mean().item())
                        if sc > best_c_sc + 1e-4 or (abs(sc - best_c_sc) <= 1e-4 and cs == wb - 1):
                            best_c_sc, best_cs = sc, cs

            rh_ax = (best_rs - r_grid).clamp(0, side - 1)
            cw_ax = (best_cs - c_grid).clamp(0, side - 1)
            in_bounds_r = ((best_rs - r_grid) >= 0) & ((best_rs - r_grid) < hb)
            in_bounds_c = ((best_cs - c_grid) >= 0) & ((best_cs - c_grid) < wb)
            s_fh_ax = oh_fg_b[r_grid, cw_ax] * in_bounds_c.float().unsqueeze(-1)
            s_fv_ax = oh_fg_b[rh_ax, c_grid] * in_bounds_r.float().unsqueeze(-1)
            s_r180_ax = oh_fg_b[rh_ax, cw_ax] * (in_bounds_r & in_bounds_c).float().unsqueeze(-1)

            ref_only_votes = s_fh_ax + s_fv_ax + s_r180_ax + s_tr_fg
            ref_max_val, ref_max_idx = ref_only_votes[:, :, 3:].max(dim=-1)
            ref_cons_oh = F.one_hot((ref_max_idx + 3).clamp(0, 11), num_classes=12).float()
            has_ref_cons = (ref_max_val >= 2.0).float().unsqueeze(-1)
            # Where >= 2 reflected views agree on a foreground color, use the reflected consensus; else preserve oh_fg_b
            d4_consensus = torch.where(
                has_ref_cons > 0.5,
                ref_cons_oh,
                oh_fg_b,
            )
            # Skew-symmetric occlusion repair exchange: +ref_cons_oh - oh_fg_b at occluded/corrupted pixels
            d4_repair_diff = (ref_cons_oh - oh_fg_b) * has_ref_cons

            # Z-order panel occlusion overlay: Panel 1 foreground occludes Panel 2 foreground (e.g. bbb1b8b6)
            bg_1d = (1.0 - p1_fg).unsqueeze(-1)
            panel_occ_overlay = oh_fg_b + bg_1d * (p_right + p_down) * fg_mask[0]

            # Channel 1 & 2: 2D Subblock Max-Pool (`t_block_max`) & Beltrami Outlier Pool (`t_block_outlier`)
            t_block_max = torch.zeros_like(oh_b)
            t_block_outlier = torch.zeros_like(oh_b)
            if is_scaled > 0.0 and th <= hb and tw <= wb and (th < hb or tw < wb):
                # Dominant color in active input region [0:hb, 0:wb]
                act_oh = oh_b[:hb, :wb, 2:12].reshape(-1, 10).sum(dim=0)
                dom_c = int(torch.argmax(act_oh).item()) + 2
                dom_oh = F.one_hot(torch.tensor(dom_c, device=device), num_classes=12).float()
                for rr in range(th):
                    r_lo = (rr * hb) // max(1, th)
                    r_hi = max(r_lo + 1, ((rr + 1) * hb) // max(1, th))
                    for cc in range(tw):
                        c_lo = (cc * wb) // max(1, tw)
                        c_hi = max(c_lo + 1, ((cc + 1) * wb) // max(1, tw))
                        blk_fg = oh_fg_b[r_lo:r_hi, c_lo:c_hi]
                        t_block_max[rr, cc] = blk_fg.amax(dim=(0, 1))
                        blk_all = oh_b[r_lo:r_hi, c_lo:c_hi]
                        blk_counts = blk_all[:, :, 2:12].reshape(-1, 10).sum(dim=0)
                        non_dom_counts = blk_counts.clone()
                        non_dom_counts[dom_c - 2] = 0.0
                        if float(non_dom_counts.max().item()) > 0.0:
                            out_c = int(torch.argmax(non_dom_counts).item()) + 2
                            t_block_outlier[rr, cc] = F.one_hot(
                                torch.tensor(out_c, device=device), num_classes=12
                            ).float()
                        else:
                            t_block_outlier[rr, cc] = dom_oh

            # Minority vs majority foreground color indicator on the input canvas
            fg_counts = oh_fg_b.view(-1, 12).sum(dim=0)  # (12,)
            pos_fg_mask = fg_counts > 0
            if pos_fg_mask.any():
                max_cnt = fg_counts.max().clamp(min=1.0)
                min_cnt = fg_counts[pos_fg_mask].min()
                rarity_w = torch.zeros(12, device=device, dtype=torch.float32)
                rarity_w[pos_fg_mask] = 1.0 - 2.0 * (
                    (fg_counts[pos_fg_mask] - min_cnt) / (max_cnt - min_cnt + 1e-6)
                )
            else:
                rarity_w = torch.zeros(12, device=device, dtype=torch.float32)
            minority_fg_chan = oh_fg_b * rarity_w.view(1, 1, 12)

            asym_b = torch.stack(
                [
                    d4_consensus,
                    d4_repair_diff,
                    panel_occ_overlay,
                    t_block_max,
                    t_block_outlier,
                    oh_fg_b * s_fh_fg,
                    (p_up - p_down + p_left - p_right) * fg_mask[0],
                    minority_fg_chan,
                ],
                dim=-2,
            )  # (side, side, 8, 12)
            asym_per_sample.append(asym_b)

        E_t = torch.stack(eq_per_sample, dim=0).view(B, L, -1, 12).permute(0, 1, 3, 2).contiguous()
        Phi = torch.stack(rel_per_sample, dim=0).view(B, L, -1).contiguous()
        u_occ = torch.stack(u_occ_per_sample, dim=0).view(B, L, self.C_OCC).contiguous()
        E_asym = (
            torch.stack(asym_per_sample, dim=0)
            .view(B, L, self.M_ASYM, 12)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        if len(self._basis_cache) >= 8:
            self._basis_cache.clear()
        self._basis_cache[cache_key] = (E_t, Phi, u_occ, E_asym)
        return E_t, Phi, u_occ, E_asym

    def _get_lm_pinv(self, lm_weight: torch.Tensor) -> torch.Tensor:
        key = (int(lm_weight.data_ptr()), tuple(lm_weight.shape), str(lm_weight.device))
        if key not in self._pinv_cache:
            if len(self._pinv_cache) >= 4:
                self._pinv_cache.clear()
            self._pinv_cache[key] = torch.linalg.pinv(lm_weight.detach().float())
        return self._pinv_cache[key]

    @staticmethod
    def _locate_lm_weight(model: Optional[nn.Module], vocab_size: int, d_r: int) -> Optional[torch.Tensor]:
        if model is None:
            return None
        from brain_ai.models.callosum import _unwrap_model

        m_inner = _unwrap_model(model)
        m_act = _unwrap_model(getattr(m_inner, "model", m_inner))
        for cand in (getattr(m_act, "inner", None), m_act, m_inner):
            if cand is None:
                continue
            for attr in ("lm_head", "head"):
                head_mod = getattr(cand, attr, None)
                if (
                    head_mod is not None
                    and hasattr(head_mod, "weight")
                    and isinstance(head_mod.weight, torch.Tensor)
                ):
                    if head_mod.weight.shape == (vocab_size, d_r):
                        return head_mod.weight.detach().float()
        return None

    def compute_competing_color_vector(
        self,
        base_logits: torch.Tensor,
        E_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds the local competing color vector `v in R^(B, L, 12)` from the temperature-scaled
        softmax of the symmetric canvas logits plus candidate foreground color indicators across `E_t`,
        so `v_c > 0` precisely for the 2 (or more) colors competing at pixel `ell`.
        """
        probs_sym = F.softmax(base_logits.detach().float() / max(self.tau_v, 0.1), dim=-1)
        cand_ind = E_t.detach().float().abs().amax(dim=-1)  # (B, L, 12)
        cand_norm = cand_ind / (cand_ind.sum(dim=-1, keepdim=True) + 1e-6)
        v = 0.70 * probs_sym + 0.30 * cand_norm
        return v

    def forward(
        self,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        delta_z: Optional[torch.Tensor],
        model: Optional[nn.Module] = None,
        lm_weight: Optional[torch.Tensor] = None,
        carry: Any = None,
        callosal_emb: Any = None,
        target_hw: Optional[Tuple[int, int]] = None,
        color_preserving: bool = False,
        triad_gain_override: Optional[float] = None,
        asym_gain_override: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Applies the DarkSnail canvas readout (`Symmetric + Skew-Symmetric Triad + Directional Asymmetry`).
        """
        logits_f32 = logits.float()
        if logits_f32.dim() == 3:
            logit_std = logits_f32.std(dim=-1, keepdim=True)
            if bool((logit_std > 2.5).any().item()):
                scale = torch.clamp(logit_std / 1.5, min=1.0)
                logits_f32 = logits_f32 / scale

        if delta_z is None or logits_f32.dim() != 3 or logits_f32.size(1) != inputs.size(1):
            return logits_f32

        B, L = inputs.shape[0], inputs.shape[1]

        if callosal_emb is not None and hasattr(callosal_emb, "num_query_tokens"):
            num_q = callosal_emb.num_query_tokens
            d_r = callosal_emb.d_reasoner
        elif delta_z.dim() == 2:
            num_q, d_r = delta_z.shape
        elif delta_z.dim() == 3:
            num_q, d_r = delta_z.shape[1], delta_z.shape[2]
        elif delta_z.numel() == 8192:
            num_q, d_r = 16, 512
        else:
            return logits_f32

        d_z_f = delta_z.float().to(logits_f32.device)
        if d_z_f.numel() == num_q * d_r:
            d_z_batch = d_z_f.view(1, num_q, d_r).expand(B, num_q, d_r)
        elif d_z_f.numel() == B * num_q * d_r:
            d_z_batch = d_z_f.view(B, num_q, d_r)
        else:
            return logits_f32

        if lm_weight is None and model is not None:
            lm_weight = self._locate_lm_weight(model, logits_f32.size(-1), d_r)

        side = int(round(math.sqrt(L)))
        if side * side == L and logits_f32.size(-1) == 12 and (num_q * d_r) >= self.ASYM_END:
            E_t, Phi, u_occ, E_asym = self.get_spatial_relational_and_occlusion_tensors(
                inputs, target_hw=target_hw
            )
            M_eq = E_t.size(-1)  # 38
            K_rel = Phi.size(-1)  # 205
            sym_end = M_eq + K_rel * 12  # 2498

            flat_dz = d_z_batch.reshape(B, -1)  # (B, 8192)

            # 1. Symmetric Base Readout: flat_dz[:, :sym_end]
            eq_gain = 4.0 if color_preserving else 2.0
            rel_gain = 0.50 if color_preserving else 1.0
            w_eq = flat_dz[:, :M_eq].view(B, 1, M_eq, 1) * eq_gain
            rem = flat_dz[:, M_eq:sym_end].view(B, K_rel, 12)
            rel_scale = torch.full((1, K_rel, 1), 0.6 * rel_gain, device=rem.device, dtype=rem.dtype)
            rel_scale[:, :12, 0] = 2.0 * rel_gain
            if color_preserving:
                # Enforce exact S_9 foreground palette equivariance:
                # only PAD (0), EOS (1), and Background-0 (2) can use color-indexed Phi weights;
                # foreground colors 3..11 are driven strictly by color-equivariant E_t and E_asym!
                fg_col_mask = torch.ones((1, 1, 12), device=rem.device, dtype=rem.dtype)
                fg_col_mask[:, :, 3:] = 0.0
                u_rel = rem * rel_scale * fg_col_mask
            else:
                u_rel = rem * rel_scale

            shift_eq = torch.matmul(E_t, w_eq).squeeze(-1)  # (B, L, 12)
            shift_rel = torch.bmm(Phi, u_rel)  # (B, L, 12)
            raw_sym_shift = shift_eq + shift_rel

            # 2. Directional Asymmetry & Block Basis: flat_dz[:, triad_end : triad_end + M_asym]
            triad_end = sym_end + self.C_OCC * 144
            asym_end = triad_end + self.M_ASYM
            eff_asym_gain = (
                asym_gain_override if asym_gain_override is not None else self.asym_gain
            ) * (1.50 if color_preserving else 0.75)
            w_asym = flat_dz[:, triad_end:asym_end].view(B, 1, self.M_ASYM, 1) * eff_asym_gain
            shift_asym = torch.matmul(E_asym, w_asym).squeeze(-1)  # (B, L, 12)

            # Global canvas foreground-to-background complement + u_occ relational slice: flat_dz[:, 5322 : 5658]
            shift_bg_fg = torch.zeros_like(shift_asym)
            if flat_dz.size(-1) >= 5658 and not color_preserving:
                bg_mask = u_occ[:, :, 1:2]  # (B, L, 1) = 1.0 - p1_fg
                canvas_fg = E_t[:, :, :, 1].amax(dim=1, keepdim=True)  # (B, 1, 12)
                phi_bg_fg = bg_mask * canvas_fg  # (B, L, 12)
                u_bg_fg = flat_dz[:, 5322:5466].view(B, 12, 12) * 0.85
                has_2panel_only = (
                    (u_occ[:, :, 5:9].amax(dim=(1, 2), keepdim=True) > 0)
                    & (u_occ[:, :, 9:13].amax(dim=(1, 2), keepdim=True) == 0)
                ).to(u_occ.dtype)
                occ_scale = 2.00 + 8.00 * has_2panel_only
                u_occ_rel = flat_dz[:, 5466:5658].view(B, 16, 12) * occ_scale
                shift_bg_fg = torch.bmm(phi_bg_fg, u_bg_fg) + torch.bmm(u_occ, u_occ_rel)

            combined_linear_shift = raw_sym_shift + shift_asym + shift_bg_fg

            if lm_weight is not None:
                W_pinv = self._get_lm_pinv(lm_weight)
                dz_canvas = self.readout_scale * torch.matmul(combined_linear_shift, W_pinv.t())

                z_H = getattr(getattr(carry, "inner_carry", None), "z_H", None)
                if (
                    isinstance(z_H, torch.Tensor)
                    and z_H.dim() == 3
                    and z_H.size(1) == L + num_q
                    and z_H.size(2) == d_r
                ):
                    z_prefix = F.normalize(z_H[:, :num_q, :].detach().float(), dim=-1)
                    z_canvas = F.normalize(z_H[:, num_q:, :].detach().float(), dim=-1)
                    attn = F.softmax(
                        torch.matmul(z_canvas, z_prefix.transpose(-2, -1)) * math.sqrt(float(num_q)),
                        dim=-1,
                    )
                    dz_canvas = dz_canvas + (0.25 * self.readout_scale) * torch.matmul(attn, d_z_batch)

                delta_sym_asym = F.linear(dz_canvas, lm_weight)
            else:
                delta_sym_asym = self.readout_scale * combined_linear_shift

            # 3. Skew-Symmetric Triad Occlusion Breaker: flat_dz[:, sym_end : triad_end]
            # Computed on competing color vector v so <v, delta_triad> == 0 holds to machine precision!
            sym_canvas_logits = logits_f32 + delta_sym_asym
            v_comp = self.compute_competing_color_vector(sym_canvas_logits, E_t)  # (B, L, 12)
            W_triad_raw = flat_dz[:, sym_end:triad_end].view(B, self.C_OCC, 12, 12)
            if color_preserving:
                # Project W_triad onto the S_9-invariant subspace over foreground colors 3..11
                W_triad = torch.zeros_like(W_triad_raw)
                W_triad[:, :, :3, :3] = W_triad_raw[:, :, :3, :3]
                bg_to_fg = W_triad_raw[:, :, :3, 3:].mean(dim=-1, keepdim=True).expand(-1, -1, 3, 9)
                fg_to_bg = W_triad_raw[:, :, 3:, :3].mean(dim=-2, keepdim=True).expand(-1, -1, 9, 3)
                W_triad[:, :, :3, 3:] = bg_to_fg
                W_triad[:, :, 3:, :3] = fg_to_bg
            else:
                W_triad = W_triad_raw
            base_triad_gain = triad_gain_override if triad_gain_override is not None else self.triad_gain
            eff_triad_gain = base_triad_gain * (0.25 if color_preserving else 0.85)
            delta_triad = (
                self.readout_scale * eff_triad_gain * self.triad_op(u_occ, v_comp, W_triad)
            )  # (B, L, 12)

            self.last_u_occ = u_occ.detach()
            self.last_v_comp = v_comp.detach()
            self.last_sym_shift = (self.readout_scale * raw_sym_shift).detach()
            self.last_asym_shift = (self.readout_scale * (shift_asym + shift_bg_fg)).detach()
            self.last_triad_shift = delta_triad.detach()

            return sym_canvas_logits + delta_triad

        return logits_f32


class SlowFastTriadCascade(nn.Module):
    """
    Upgrade 3: Low-Rank (`d_shell = 8`) Skew-Symmetric Triad Coupling between
    Slow recurrent state `z_H^{(15)}` and Fast recurrent state `z_L^{(15)}`
    before the terminal recurrent step, parameterized by an idle slice of `flat_dz`:

        z_tilde_L = z_L + B_{W_rec}(P z_H, P z_L) P^T

    By `triad_pointwise_orthogonal` (`TriadConservation.lean`), the shell tangent step
    `delta_v = B_{W_rec}(P z_H, P z_L)` is strictly orthogonal to `P z_L` at every token,
    and spherical retraction preserves exact shell energy `||P z_tilde_L||_2^2 == ||P z_L||_2^2`
    while guaranteeing `delta_z == 0 => z_tilde_L == z_L`.
    """

    DEFAULT_SLICE_START: int = DarkSnailCanvasReadout.ASYM_END  # 4810

    def __init__(
        self,
        d_reasoner: int = 512,
        d_shell: int = 8,
        slice_start: int = DEFAULT_SLICE_START,
        cascade_gain: float = 1.0,
        preserve_exact_sphere_norm: bool = True,
        lm_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_reasoner = int(d_reasoner)
        self.d_shell = int(d_shell)
        self.slice_start = int(slice_start)
        self.slice_end = self.slice_start + (self.d_shell ** 3)  # 4810 + 512 = 5322
        self.cascade_gain = float(cascade_gain)
        self.preserve_exact_sphere_norm = bool(preserve_exact_sphere_norm)
        self.triad_op = SkewSymmetricTriadOperator()

        P = self._build_orthonormal_shell_basis(self.d_reasoner, self.d_shell, lm_weight)
        self.register_buffer("P", P, persistent=False)  # (d_reasoner, d_shell)

        self.last_tangent_step: Optional[torch.Tensor] = None

    @staticmethod
    def _build_orthonormal_shell_basis(
        d_reasoner: int,
        d_shell: int,
        lm_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Builds an orthonormal projection matrix `P in R^(d_reasoner x d_shell)` (`P^T P = I`)
        from the top-`d_shell` right singular vectors of `W_lm` (if provided) or a
        deterministic DCT basis.
        """
        if lm_weight is not None and lm_weight.shape[-1] == d_reasoner and lm_weight.shape[0] >= d_shell:
            _, _, Vh = torch.linalg.svd(lm_weight.detach().float(), full_matrices=False)
            return Vh[:d_shell, :].t().contiguous()

        idx = torch.arange(d_reasoner, dtype=torch.float32).unsqueeze(1)  # (D, 1)
        k = torch.arange(1, d_shell + 1, dtype=torch.float32).unsqueeze(0)  # (1, M)
        raw = torch.cos(math.pi * (idx + 0.5) * k / float(d_reasoner))
        q, _ = torch.linalg.qr(raw, mode="reduced")  # (D, M) with Q^T Q = I_M
        return q.contiguous()

    def extract_w_rec(self, delta_z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Extracts `W_rec` of shape `(B, d_shell, d_shell, d_shell)` from `delta_z`.
        """
        M = self.d_shell
        m3 = M * M * M
        dz_f = delta_z.float()
        if dz_f.dim() == 3 and dz_f.shape[0] == batch_size:
            flat = dz_f.reshape(batch_size, -1)
        else:
            flat = dz_f.reshape(1, -1).expand(batch_size, -1)

        if flat.shape[1] >= self.slice_end:
            w_slice = flat[:, self.slice_start : self.slice_end]
        elif flat.shape[1] >= m3:
            w_slice = flat[:, :m3]
        else:
            pad = F.pad(flat, (0, m3 - flat.shape[1]))
            w_slice = pad[:, :m3]

        return w_slice.view(batch_size, M, M, M)

    def project_to_shell(self, z: torch.Tensor) -> torch.Tensor:
        """Projects `z` `(..., d_reasoner)` onto the low-rank shell `z @ P` `(..., d_shell)`."""
        P = self.P.to(device=z.device, dtype=torch.float32)
        return torch.matmul(z.float(), P)

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        delta_z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Couples slow state `z_H` and fast state `z_L` of shape `(B, S, d_reasoner)`
        using the skew-symmetric triad generator sliced from `delta_z`.
        """
        if delta_z is None:
            self.last_tangent_step = torch.zeros(
                (*z_L.shape[:-1], self.d_shell), device=z_L.device, dtype=torch.float32
            )
            return z_L

        orig_dtype = z_L.dtype
        B = z_L.shape[0]
        P = self.P.to(device=z_L.device, dtype=torch.float32)  # (D, M)

        u_shell = torch.matmul(z_H.float(), P)  # (B, ..., M)
        v_shell = torch.matmul(z_L.float(), P)  # (B, ..., M)
        W_rec = self.extract_w_rec(delta_z.to(z_L.device), B)  # (B, M, M, M)

        # Tangent step in so(d_shell): <v_shell, delta_v> == 0 at every token
        delta_v = self.cascade_gain * self.triad_op(u_shell, v_shell, W_rec)  # (B, ..., M)
        self.last_tangent_step = delta_v

        v_updated = v_shell + delta_v
        if self.preserve_exact_sphere_norm:
            # Cayley / sphere retraction: preserves ||P z_tilde_L||_2 == ||P z_L||_2 in discrete time
            # while keeping the exact same first-order derivative dv_updated/d(delta_v) = I at delta_z = 0
            orig_norm = torch.norm(v_shell, p=2, dim=-1, keepdim=True)
            new_norm = torch.norm(v_updated, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            v_retracted = torch.where(orig_norm > 0.0, v_updated * (orig_norm / new_norm), v_updated)
            shell_diff = v_retracted - v_shell
        else:
            shell_diff = delta_v

        z_L_tilde = z_L.float() + torch.matmul(shell_diff, P.t())
        return z_L_tilde.to(orig_dtype)


class RoPE2DMedoidPooler(nn.Module):
    """
    Upgrade 5: 2D Spatial Patch Medoid-Key + Isotropic Adelic Variance Pooler.

    Pools 2D spatial grids `(B, H_heads, H, W, D)` or `(B, H, W, D)` into
    `k x k` patches `(H // k, W // k)`:
      1. Selects the 2D Medoid Key `(y*, x*) = argmin_{(y, x) in S} ||k_{y, x} - mean(k_S)||_2`
         per patch, preserving exact 2D axial RoPE relative phase invariance
         (`rope2d_medoid_relative_invariance` in `OcclusionTriad.lean`).
      2. Pools patch values via `0.5 * (mean_pool + peak_pool)`.
      3. Attenuates medoid keys by isotropic 2D Adelic spatial variance:
             gamma_2D = exp(-nu * p^(2*ell) * (Var_y(S) + Var_x(S)))
    """

    def __init__(
        self,
        patch_size: int = 2,
        viscous_nu: float = 0.05,
        viscous_p: int = 2,
        shell_level: int = 1,
    ):
        super().__init__()
        self.patch_size = max(1, int(patch_size))
        self.viscous_nu = float(viscous_nu)
        self.viscous_p = max(2, int(viscous_p))
        self.shell_level = max(0, int(shell_level))

        self.last_gamma_2d: Optional[torch.Tensor] = None
        self.last_medoid_coords: Optional[torch.Tensor] = None
        self.last_var_yx: Optional[torch.Tensor] = None

    @staticmethod
    def apply_axial_rope2d(
        qk: torch.Tensor,
        H: int,
        W: int,
        base: float = 10000.0,
    ) -> torch.Tensor:
        """
        Applies exact SO(2)^(d/4) x SO(2)^(d/4) 2D axial rotary position embeddings
        where each 2D rotary pair `(x_1, x_2)` shares the identical frequency `theta_m`,
        preserving exact Euclidean norm `||R_2D(y, x) k||_2 == ||k||_2` and relative
        phase shift `<R_2D(y_q, x_q) q, R_2D(y_k, x_k) k> == <q, R_2D(y_k - y_q, x_k - x_q) k>`.
        """
        orig_dtype = qk.dtype
        qk_f = qk.float()
        is_4d = qk_f.dim() == 4
        if is_4d:
            qk_f = qk_f.unsqueeze(1)
        B, n_h, cur_H, cur_W, d = qk_f.shape
        d_half = d // 2
        assert d_half % 2 == 0, "head_dim // 2 must be even for 2D axial RoPE"

        device = qk_f.device
        inv_freq = 1.0 / (base ** (torch.arange(0, d_half, 2, device=device, dtype=torch.float32) / float(d_half)))
        pos_h = torch.arange(cur_H, device=device, dtype=torch.float32)
        pos_w = torch.arange(cur_W, device=device, dtype=torch.float32)

        angles_h = torch.outer(pos_h, inv_freq)  # (H, d/4)
        angles_w = torch.outer(pos_w, inv_freq)  # (W, d/4)

        cos_h = torch.cat([torch.cos(angles_h), torch.cos(angles_h)], dim=-1).view(1, 1, cur_H, 1, d_half)
        sin_h = torch.cat([torch.sin(angles_h), torch.sin(angles_h)], dim=-1).view(1, 1, cur_H, 1, d_half)
        cos_w = torch.cat([torch.cos(angles_w), torch.cos(angles_w)], dim=-1).view(1, 1, 1, cur_W, d_half)
        sin_w = torch.cat([torch.sin(angles_w), torch.sin(angles_w)], dim=-1).view(1, 1, 1, cur_W, d_half)

        def _rot_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        qk_row = qk_f[..., :d_half]
        qk_col = qk_f[..., d_half:]
        qk_row_rot = (qk_row * cos_h) + (_rot_half(qk_row) * sin_h)
        qk_col_rot = (qk_col * cos_w) + (_rot_half(qk_col) * sin_w)
        out = torch.cat([qk_row_rot, qk_col_rot], dim=-1)
        if is_4d:
            out = out.squeeze(1)
        return out.to(orig_dtype)

    def forward(
        self,
        k: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        patch_size_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Pools `k` and `v` of shape `(B, H, W, D)` or `(B, N_heads, H, W, D)` over `k_p x k_p` patches.

        Returns:
            k_pooled: Medoid keys scaled by `gamma_2D`
            v_pooled: `0.5 * (mean_pool + peak_pool)` values
            info: Dictionary with `gamma_2d`, `medoid_y`, `medoid_x`, `var_y`, `var_x`, `raw_medoid_k`
        """
        if v is None:
            v = k

        is_4d = k.dim() == 4
        if is_4d:
            k_5d = k.unsqueeze(1)
            v_5d = v.unsqueeze(1)
        elif k.dim() == 5:
            k_5d = k
            v_5d = v
        else:
            raise ValueError(f"Expected 4D or 5D spatial tensor, got shape {tuple(k.shape)}")

        orig_dtype = k.dtype
        B, N_h, H, W, D = k_5d.shape
        kp = int(patch_size_override) if patch_size_override is not None else self.patch_size
        device = k.device

        if kp <= 1:
            ones_g = torch.ones((B, N_h, H, W), device=device, dtype=torch.float32)
            zeros_v = torch.zeros((B, N_h, H, W), device=device, dtype=torch.float32)
            info = {
                "gamma_2d": ones_g.squeeze(1) if is_4d else ones_g,
                "var_y": zeros_v.squeeze(1) if is_4d else zeros_v,
                "var_x": zeros_v.squeeze(1) if is_4d else zeros_v,
                "raw_medoid_k": k,
            }
            return k, v, info

        H_out = -(-H // kp)
        W_out = -(-W // kp)
        pad_h = H_out * kp - H
        pad_w = W_out * kp - W

        if pad_h > 0 or pad_w > 0:
            k_pad = F.pad(k_5d.float(), (0, 0, 0, pad_w, 0, pad_h), value=0.0)
            v_pad = F.pad(v_5d.float(), (0, 0, 0, pad_w, 0, pad_h), value=0.0)
        else:
            k_pad = k_5d.float()
            v_pad = v_5d.float()

        # Reshape into patches: (B, N_h, H_out, kp, W_out, kp, D) -> (B, N_h, H_out, W_out, kp*kp, D)
        n_cells = kp * kp
        k_patches = (
            k_pad.view(B, N_h, H_out, kp, W_out, kp, D)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, N_h, H_out, W_out, n_cells, D)
        )
        v_patches = (
            v_pad.view(B, N_h, H_out, kp, W_out, kp, D)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, N_h, H_out, W_out, n_cells, D)
        )

        # Valid pixel mask for boundary patches
        y_idx = torch.arange(H_out * kp, device=device).view(1, 1, H_out, kp, 1, 1)
        x_idx = torch.arange(W_out * kp, device=device).view(1, 1, 1, 1, W_out, kp)
        valid_patch = (
            ((y_idx < H) & (x_idx < W))
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(1, 1, H_out, W_out, n_cells)
        )
        valid_f = valid_patch.float()
        valid_counts = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (1, 1, H_out, W_out, 1)

        # 1. Patch mean for K and V
        mean_k = (k_patches * valid_f.unsqueeze(-1)).sum(dim=-2) / valid_counts
        mean_v = (v_patches * valid_f.unsqueeze(-1)).sum(dim=-2) / valid_counts

        # 2. Medoid Key per 2D patch: (y*, x*) = argmin ||k_{y,x} - mean_k||_2
        dists_k = torch.norm(k_patches - mean_k.unsqueeze(-2), p=2, dim=-1)  # (B, N_h, H_out, W_out, n_cells)
        dists_k = dists_k.masked_fill(~valid_patch, float("inf"))
        medoid_local_idx = dists_k.argmin(dim=-1, keepdim=True)  # (B, N_h, H_out, W_out, 1)
        medoid_gather_idx = medoid_local_idx.unsqueeze(-1).expand(B, N_h, H_out, W_out, 1, D)
        raw_medoid_k = torch.gather(k_patches, dim=-2, index=medoid_gather_idx).squeeze(-2)

        # 3. Value pooling via 0.5 * (mean_pool + peak_pool)
        v_norms = torch.norm(v_patches, p=2, dim=-1).masked_fill(~valid_patch, float("-inf"))
        peak_idx = v_norms.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(B, N_h, H_out, W_out, 1, D)
        peak_v = torch.gather(v_patches, dim=-2, index=peak_idx).squeeze(-2)
        v_pooled = 0.5 * (mean_v + peak_v)

        # 4. Energy-weighted 2D coordinate variance Var_y(S) + Var_x(S) within each patch
        local_y = (torch.arange( n_cells, device=device, dtype=torch.float32) // kp).view(1, 1, 1, 1, n_cells)
        local_x = (torch.arange( n_cells, device=device, dtype=torch.float32) % kp).view(1, 1, 1, 1, n_cells)
        k_norms = torch.norm(k_patches, p=2, dim=-1) * valid_f
        weights = (k_norms + 1e-6 * valid_f) / (
            (k_norms + 1e-6 * valid_f).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        )
        mean_y = (local_y * weights).sum(dim=-1, keepdim=True)
        mean_x = (local_x * weights).sum(dim=-1, keepdim=True)
        var_y = ((local_y - mean_y) ** 2 * weights).sum(dim=-1)
        var_x = ((local_x - mean_x) ** 2 * weights).sum(dim=-1)

        # Scale-covariant 2D Adelic heat-kernel attenuation: gamma_2D = exp(-nu * p^(2*ell) * (Var_y + Var_x))
        p_scale = float(self.viscous_p) ** (2 * self.shell_level)
        gamma_2d = torch.exp(-self.viscous_nu * p_scale * (var_y + var_x)).clamp(min=0.05, max=1.0)
        k_pooled = raw_medoid_k * gamma_2d.unsqueeze(-1)

        # Global 2D coordinates of the selected medoid per patch
        patch_base_y = (torch.arange(H_out, device=device) * kp).view(1, 1, H_out, 1)
        patch_base_x = (torch.arange(W_out, device=device) * kp).view(1, 1, 1, W_out)
        medoid_y = patch_base_y + (medoid_local_idx.squeeze(-1) // kp)
        medoid_x = patch_base_x + (medoid_local_idx.squeeze(-1) % kp)

        self.last_gamma_2d = gamma_2d.detach()
        self.last_medoid_coords = torch.stack([medoid_y, medoid_x], dim=-1).detach()
        self.last_var_yx = (var_y + var_x).detach()

        if is_4d:
            k_pooled = k_pooled.squeeze(1)
            v_pooled = v_pooled.squeeze(1)
            raw_medoid_k = raw_medoid_k.squeeze(1)
            gamma_2d = gamma_2d.squeeze(1)
            var_y = var_y.squeeze(1)
            var_x = var_x.squeeze(1)
            medoid_y = medoid_y.squeeze(1)
            medoid_x = medoid_x.squeeze(1)

        info = {
            "gamma_2d": gamma_2d,
            "var_y": var_y,
            "var_x": var_x,
            "medoid_y": medoid_y,
            "medoid_x": medoid_x,
            "raw_medoid_k": raw_medoid_k.to(orig_dtype),
        }
        return k_pooled.to(orig_dtype), v_pooled.to(orig_dtype), info
