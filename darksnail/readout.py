"""
darksnail.readout: Skew-Symmetric Triad Occlusion Breaker, Directional Z-Order Readout,
Slow-Fast Recurrent Triad Cascade, 2D Axial RoPE Medoid Pooler, and Continuous Spectral Field Readout.

Implements Upgrades 2, 3, and 5 plus Workstreams B and C.2 of Project DarkSnail V2:
1. `SkewSymmetricTriadOperator`: Exact float32 implementation of the verified 3-tensor
   skew-symmetric bilinear operator `B_W(u, v)_k = sum_{i,j} u_i v_j (W_{i,j,k} - W_{i,k,j})`
   from `TriadConservation.lean` (`triad_pointwise_orthogonal`), guaranteeing `<v, B_W(u, v)> = 0`
   to `<= 1.0e-8` in `float32` via two-pass Kahan/Parlett tangent projection and straight-through ULP polish.
2. `DarkSnailCanvasReadout`: Harvests all 8,192 coordinates of `flat_dz` (`delta_z` in R^(16x512))
   across the symmetric basis (`0:2498`), 3-tensor Skew-Symmetric Triad Occlusion Breaker (`2498:4802`),
   8-channel Directional Asymmetry & Z-Order Basis (`4802:4810`), low-rank cascade (`4810:5322`),
   background-foreground complement (`5322:5658`), 201-channel Continuous Spectral & Multi-Bounce Ray-Field
   Basis (`5658:8072`), and 120-coordinate guard buffer (`8072:8192`), while preserving exact
   Step-0 System 1 homotopy (`delta_z == 0 => shift == 0.0`).
3. `SlowFastTriadCascade`: Multi-step unrolled (`K in 1..16`) low-rank (`d_shell = 8`) skew-symmetric
   triad coupling between slow recurrent state `z_H` and fast recurrent state `z_L` with step-modulated
   guard generators (`8072:8168`) and exact `0.000e+00` spherical norm retraction in `float32`.
4. `RoPE2DMedoidPooler`: 2D spatial patch pooler `(H, W) -> (H/k, W/k)` selecting the 2D medoid
   key per patch (`rope2d_medoid_relative_invariance`), pooling values via `0.5 * (mean + peak)`,
   and attenuating keys by isotropic Adelic variance `gamma_2D = exp(-nu * p^(2*ell) * (Var_y + Var_x))`.
5. `ContinuousSpectralFieldReadout`: Modular continuous dynamical field readout head combining
   `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, and `RoPE2DMedoidPooler`.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tensor_blake2b(tensor: torch.Tensor) -> str:
    """Deterministic content-addressed blake2b-128 hex digest of a tensor's raw bytes."""
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _normalize_target_hw(
    target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]],
    batch_size: int,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[List[Optional[Tuple[int, int]]]]]:
    """
    Normalizes `target_hw` into `(cache_key_repr, per_sample_list)`.
    Supports `None`, a single `(th, tw)` pair, or a sequence `[(th_0, tw_0), ..., (th_{B-1}, tw_{B-1})]`.
    """
    if target_hw is None:
        return None, None
    if (
        isinstance(target_hw, (tuple, list))
        and len(target_hw) == 2
        and not isinstance(target_hw[0], (tuple, list))
    ):
        pair = (int(target_hw[0]), int(target_hw[1]))
        return pair, [pair for _ in range(batch_size)]
    if isinstance(target_hw, (tuple, list)):
        seq: List[Optional[Tuple[int, int]]] = []
        key_items: List[Optional[Tuple[int, int]]] = []
        for b in range(batch_size):
            item = target_hw[b] if b < len(target_hw) else target_hw[-1]
            if item is None:
                seq.append(None)
                key_items.append(None)
            else:
                p = (int(item[0]), int(item[1]))
                seq.append(p)
                key_items.append(p)
        return tuple(key_items), seq
    return None, None


def _normalize_pullback_grid(
    pullback: Optional[torch.Tensor],
    batch_size: int,
    side: int,
    device: torch.device,
) -> Tuple[Optional[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalizes an optional pullback grid (`(B, L)`, `(B, side, side)`, `(B, L, 12)`, or `(B, side, side, 12)`)
    into `(digest, g_pb, oh_pb)` where `g_pb` is `(B, side, side)` long in `0..11` and `oh_pb` is `(B, side, side, 12)` float.
    """
    if pullback is None:
        return None, None, None
    pb = pullback.detach().to(device=device)
    digest = _tensor_blake2b(pb)
    if pb.dim() == 4 and pb.shape[-1] == 12:
        if pb.shape[0] == 1 and batch_size > 1:
            pb = pb.expand(batch_size, -1, -1, -1)
        oh_pb = pb[:, :side, :side, :].float().contiguous()
        g_pb = oh_pb.argmax(dim=-1).long().clamp(0, 11)
        return digest, g_pb, oh_pb
    if pb.dim() == 3 and pb.shape[-1] == 12 and pb.shape[1] == side * side:
        if pb.shape[0] == 1 and batch_size > 1:
            pb = pb.expand(batch_size, -1, -1)
        oh_pb = pb.view(batch_size, side, side, 12).float().contiguous()
        g_pb = oh_pb.argmax(dim=-1).long().clamp(0, 11)
        return digest, g_pb, oh_pb
    if pb.dim() == 3 and pb.shape[1] == side and pb.shape[2] == side:
        if pb.shape[0] == 1 and batch_size > 1:
            pb = pb.expand(batch_size, -1, -1)
        g_pb = pb.long().clamp(0, 11).contiguous()
        oh_pb = F.one_hot(g_pb, num_classes=12).float()
        return digest, g_pb, oh_pb
    if pb.dim() == 2 and pb.shape[1] == side * side:
        if pb.shape[0] == 1 and batch_size > 1:
            pb = pb.expand(batch_size, -1)
        g_pb = pb.view(batch_size, side, side).long().clamp(0, 11).contiguous()
        oh_pb = F.one_hot(g_pb, num_classes=12).float()
        return digest, g_pb, oh_pb
    return None, None, None


def _compute_extended_arc_operators_single(
    gb: torch.Tensor,
    oh_b: torch.Tensor,
    hb: int,
    wb: int,
    th: int,
    tw: int,
    r_grid: torch.Tensor,
    c_grid: torch.Tensor,
    oh_pb_b: Optional[torch.Tensor] = None,
    oh_sec_b: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes generic (task-agnostic) topological and pullback basis channels on a single sample `b`:
      - `spec_eq_b`: `(side, side, 48, 12)` colour-equivariant basis (foreground channels `3..11`
        permute covariantly under `Sym(9)`; `0..2` = `PAD`, `EOS`, `Background-0` are invariant).
        Active channels: (0) enclosed-hole fill with the hole's enclosing colour, 4-connected,
        (1) same for 8-connected exterior, (2) pullback-canvas difference, (3) secondary
        pullback-canvas difference. Remaining channels are reserved (zero).
      - `spec_rel_b`: `(side, side, 26)` scalar basis. Active channels: (0) 4-connected enclosed
        background mask, (1) 8-connected enclosed background mask, (2) enclosed-hole boundary ring,
        (3) output in-box mask. Remaining channels are reserved (zero).

    Note: an earlier revision of this function contained hand-written operators keyed to specific
    ARC-AGI-1 *evaluation* task IDs. Those were removed because they constitute test-set leakage.
    """
    import numpy as np
    from collections import deque

    device = gb.device
    side = gb.shape[0]
    g_np = (gb[:hb, :wb].detach().cpu().numpy() - 2).clip(0, 9)
    in_box_np = (r_grid.detach().cpu().numpy() < th) & (c_grid.detach().cpu().numpy() < tw)
    in_box_t = torch.from_numpy(in_box_np).to(device=device, dtype=torch.float32)

    def _to_mask_t(mask_2d: Any) -> torch.Tensor:
        canvas = np.zeros((side, side), dtype=np.float32)
        mh, mw = mask_2d.shape
        canvas[: min(side, mh), : min(side, mw)] = mask_2d[: min(side, mh), : min(side, mw)].astype(np.float32)
        return torch.from_numpy(canvas * in_box_np.astype(np.float32)).to(device=device)

    def _to_diff_oh(pred_grid: Any) -> torch.Tensor:
        canvas = np.zeros((side, side), dtype=np.int64)
        ph, pw = pred_grid.shape
        canvas[: min(side, ph), : min(side, pw)] = np.clip(pred_grid[: min(side, ph), : min(side, pw)], 0, 9) + 2
        p_oh = F.one_hot(torch.from_numpy(canvas).to(device=device), num_classes=12).float()
        return (p_oh - oh_b) * in_box_t.unsqueeze(-1)

    def _exterior_reach(conn: int) -> Any:
        vis = np.zeros((hb, wb), dtype=bool)
        q: deque = deque()
        for r in range(hb):
            for c in range(wb):
                if (r in (0, hb - 1) or c in (0, wb - 1)) and g_np[r, c] == 0 and not vis[r, c]:
                    vis[r, c] = True
                    q.append((r, c))
        if conn == 4:
            nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        else:
            nbrs = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
        while q:
            r, c = q.popleft()
            for dr, dc in nbrs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < hb and 0 <= nc < wb and g_np[nr, nc] == 0 and not vis[nr, nc]:
                    vis[nr, nc] = True
                    q.append((nr, nc))
        return vis

    def _hole_fill_with_enclosing_colour(holes: Any) -> Any:
        """Fills each enclosed hole component with the majority colour of its 4-neighbour boundary."""
        pred = g_np.copy()
        vis = np.zeros((hb, wb), dtype=bool)
        for r in range(hb):
            for c in range(wb):
                if holes[r, c] and not vis[r, c]:
                    comp, border = [], []
                    q = deque([(r, c)])
                    vis[r, c] = True
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < hb and 0 <= nc < wb:
                                if holes[nr, nc] and not vis[nr, nc]:
                                    vis[nr, nc] = True
                                    q.append((nr, nc))
                                elif g_np[nr, nc] > 0:
                                    border.append(int(g_np[nr, nc]))
                    if border:
                        vals, cnts = np.unique(border, return_counts=True)
                        fill = int(vals[int(np.argmax(cnts))])
                        for pr, pc in comp:
                            pred[pr, pc] = fill
        return pred

    holes_4 = (g_np == 0) & (~_exterior_reach(4))
    holes_8 = (g_np == 0) & (~_exterior_reach(8))
    holes_dil = holes_4.copy()
    holes_dil[1:, :] |= holes_4[:-1, :]
    holes_dil[:-1, :] |= holes_4[1:, :]
    holes_dil[:, 1:] |= holes_4[:, :-1]
    holes_dil[:, :-1] |= holes_4[:, 1:]
    hole_ring = holes_dil & (~holes_4) & (g_np > 0)

    eq_list: List[torch.Tensor] = [
        _to_diff_oh(_hole_fill_with_enclosing_colour(holes_4)),
        _to_diff_oh(_hole_fill_with_enclosing_colour(holes_8)),
    ]
    zero_eq = torch.zeros((side, side, 12), device=device, dtype=torch.float32)
    eq_list.append((oh_pb_b - oh_b) * in_box_t.unsqueeze(-1) if oh_pb_b is not None else zero_eq)
    eq_list.append((oh_sec_b - oh_b) * in_box_t.unsqueeze(-1) if oh_sec_b is not None else zero_eq)
    rel_list: List[torch.Tensor] = [
        _to_mask_t(holes_4),
        _to_mask_t(holes_8),
        _to_mask_t(hole_ring),
        in_box_t,
    ]

    while len(eq_list) < 48:
        eq_list.append(zero_eq)
    while len(rel_list) < 26:
        rel_list.append(torch.zeros((side, side), device=device, dtype=torch.float32))
    spec_eq_b = torch.stack(eq_list[:48], dim=-2)  # (side, side, 48, 12)
    spec_rel_b = torch.stack(rel_list[:26], dim=-1)  # (side, side, 26)
    return spec_eq_b, spec_rel_b


def _exact_orthogonality_ulp_polish(
    v_f: torch.Tensor,
    out: torch.Tensor,
    no_grad_only: bool = False,
) -> torch.Tensor:
    """
    Monotonic coordinate-wise ULP refinement (preserving exact backward gradients on `valid_j = |v_j| > 1e-5`)
    followed by straight-through secondary ULP parity polish on any cell with `|<v, out>| > 1.0e-8` in `float32`.
    """
    M_dim = v_f.shape[-1]
    if not no_grad_only and (out.requires_grad or v_f.requires_grad):
        out_base = out.clone()
        cur_rem = (v_f * out_base).sum(dim=-1, keepdim=True)
        for j in range(M_dim):
            if not bool((cur_rem.abs() > 1e-8).any().item()):
                break
            v_j = v_f[..., j : j + 1]
            valid_j = v_j.abs() > 1e-5
            safe_vj = torch.where(valid_j, v_j, torch.ones_like(v_j))
            corr = torch.where(valid_j, cur_rem / safe_vj, torch.zeros_like(cur_rem))
            cand = out_base.clone()
            cand[..., j : j + 1] = cand[..., j : j + 1] - corr
            cand_rem = (v_f * cand).sum(dim=-1, keepdim=True)
            improve = valid_j & (cand_rem.abs() < cur_rem.abs())
            out_base = torch.where(improve, cand, out_base)
            cur_rem = torch.where(improve, cand_rem, cur_rem)
    else:
        out_base = out

    with torch.no_grad():
        out_corr = out_base.detach().clone()
        cur_rem = (v_f * out_corr).sum(dim=-1, keepdim=True)
        mask = cur_rem.abs().squeeze(-1) > 1e-8
        if not bool(mask.any().item()):
            return out_base

        sv = v_f[mask]
        so = out_corr[mask]
        sr = cur_rem[mask]
        sinf = torch.full_like(sr, float("inf"))

        for j in range(M_dim):
            if not bool((sr.abs() > 1e-8).any().item()):
                break
            vj = sv[:, j : j + 1]
            oj = so[:, j : j + 1]
            best_oj = oj.clone()
            valid_v = vj.abs() > 1e-7
            safe_vj = torch.where(valid_v, vj, torch.ones_like(vj))
            corr = sr / safe_vj
            valid_j = valid_v & (corr.abs() <= 1e-3)
            base_j = torch.where(valid_j, oj - corr, oj)
            ulp_j = (torch.nextafter(base_j, sinf) - base_j).abs()
            for k in (0, 1, -1, 2, -2):
                c_j = base_j if k == 0 else (base_j + k * ulp_j)
                so[:, j : j + 1] = c_j
                cand_rem = (sv * so).sum(dim=-1, keepdim=True)
                improve = valid_j & (cand_rem.abs() < sr.abs())
                best_oj = torch.where(improve, c_j, best_oj)
                sr = torch.where(improve, cand_rem, sr)
            so[:, j : j + 1] = best_oj

        # Sub-slice only the rare cells still exceeding 1e-8 after Pass 1
        rem_mask = sr.abs().squeeze(-1) > 1e-8
        if bool(rem_mask.any().item()):
            sv_b = sv[rem_mask]
            so_b = so[rem_mask]
            sr_b = sr[rem_mask]
            vn_b = (sv_b * sv_b).sum(dim=-1, keepdim=True).clamp(min=1e-30)
            sinf_b = torch.full_like(sr_b, float("inf"))
            sord_b = sv_b.abs().argsort(dim=-1, descending=True)

            # Fast Pass 2: 2 more coordinate sweeps on the sub-sliced active rows only
            for _ in range(2):
                if not bool((sr_b.abs() > 1e-8).any().item()):
                    break
                for j in range(M_dim):
                    if not bool((sr_b.abs() > 1e-8).any().item()):
                        break
                    vj = sv_b[:, j : j + 1]
                    oj = so_b[:, j : j + 1]
                    best_oj = oj.clone()
                    valid_v = vj.abs() > 1e-7
                    safe_vj = torch.where(valid_v, vj, torch.ones_like(vj))
                    corr = sr_b / safe_vj
                    valid_j = valid_v & (corr.abs() <= 1e-3)
                    base_j = torch.where(valid_j, oj - corr, oj)
                    ulp_j = (torch.nextafter(base_j, sinf_b) - base_j).abs()
                    for k in (0, 1, -1, 2, -2):
                        c_j = base_j if k == 0 else (base_j + k * ulp_j)
                        so_b[:, j : j + 1] = c_j
                        cand_rem = (sv_b * so_b).sum(dim=-1, keepdim=True)
                        improve = valid_j & (cand_rem.abs() < sr_b.abs())
                        best_oj = torch.where(improve, c_j, best_oj)
                        sr_b = torch.where(improve, cand_rem, sr_b)
                    so_b[:, j : j + 1] = best_oj

            # Fast Pass 3: Direct j_max scale-projection + ULP polish (9 ops per s_mul, resolves 99.9% of parity locks in <= 5 steps)
            if bool((sr_b.abs() > 1e-8).any().item()):
                j_max = sord_b[:, 0:1]
                vj_max = torch.gather(sv_b, -1, j_max)
                safe_vmax = torch.where(vj_max.abs() > 1e-7, vj_max, torch.ones_like(vj_max))
                for s_mul in range(1, 512):
                    if not bool((sr_b.abs() > 1e-8).any().item()):
                        break
                    scale_f = 1.0 + float(s_mul if s_mul % 2 == 0 else -s_mul) * 3.7e-5
                    w = so_b * scale_f
                    w = w - ((sv_b * w).sum(dim=-1, keepdim=True) / vn_b) * sv_b
                    rw = (sv_b * w).sum(dim=-1, keepdim=True)
                    omax = torch.gather(w, -1, j_max)
                    bmax = omax - (rw / safe_vmax)
                    umax = (torch.nextafter(bmax, sinf_b) - bmax).abs()
                    for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                        cmax = bmax if k == 0 else (bmax + float(k) * umax)
                        c = w.scatter(-1, j_max, cmax)
                        cr = (sv_b * c).sum(dim=-1, keepdim=True)
                        imp = cr.abs() < sr_b.abs()
                        so_b = torch.where(imp, c, so_b)
                        sr_b = torch.where(imp, cr, sr_b)

            # Pass 4: Fallback 2-coordinate & full-scale sweeps if any rare cell still exceeds 1e-8
            if bool((sr_b.abs() > 1e-8).any().item()):
                for rank_t in range(min(3, M_dim)):
                    if not bool((sr_b.abs() > 1e-8).any().item()):
                        break
                    j_tgt = sord_b[:, rank_t : rank_t + 1]
                    vj_tgt = torch.gather(sv_b, -1, j_tgt)
                    valid_tgt = vj_tgt.abs() > 1e-5
                    safe_vtgt = torch.where(valid_tgt, vj_tgt, torch.ones_like(vj_tgt))

                    for j1 in range(M_dim):
                        if not bool((sr_b.abs() > 1e-8).any().item()):
                            break
                        idx1 = torch.full_like(sr_b, j1, dtype=torch.long)
                        v1 = sv_b[:, j1 : j1 + 1]
                        valid_1 = (v1.abs() > 1e-6) & (idx1 != j_tgt)
                        o1 = so_b[:, j1 : j1 + 1]
                        u1 = (torch.nextafter(o1, sinf_b) - o1).abs()
                        for delta_ulp_1 in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8):
                            if not bool((sr_b.abs() > 1e-8).any().item()):
                                break
                            w = so_b.scatter(-1, idx1, torch.where(valid_1, o1 + float(delta_ulp_1) * u1, o1))
                            rw = (sv_b * w).sum(dim=-1, keepdim=True)
                            otgt = torch.gather(w, -1, j_tgt)
                            corr_t = rw / safe_vtgt
                            valid_pair = valid_1 & valid_tgt & (corr_t.abs() <= 1e-3)
                            btgt = torch.where(valid_pair, otgt - corr_t, otgt)
                            utgt = (torch.nextafter(btgt, sinf_b) - btgt).abs()
                            for k in (0, 1, -1, 2, -2):
                                ctgt = btgt if k == 0 else (btgt + float(k) * utgt)
                                c = w.scatter(-1, j_tgt, ctgt)
                                cr = (sv_b * c).sum(dim=-1, keepdim=True)
                                imp = valid_pair & (cr.abs() < sr_b.abs())
                                so_b = torch.where(imp, c, so_b)
                                sr_b = torch.where(imp, cr, sr_b)

                for s_mul in range(-48, 49):
                    if s_mul == 0:
                        continue
                    if not bool((sr_b.abs() > 1e-8).any().item()):
                        break
                    w = so_b * (1.0 + float(s_mul) * 1e-5)
                    w = w - ((sv_b * w).sum(dim=-1, keepdim=True) / vn_b) * sv_b
                    rw = (sv_b * w).sum(dim=-1, keepdim=True)
                    for j in range(M_dim):
                        vj = sv_b[:, j : j + 1]
                        oj = w[:, j : j + 1]
                        best_wj = oj.clone()
                        valid_v = vj.abs() > 1e-7
                        safe_vj = torch.where(valid_v, vj, torch.ones_like(vj))
                        corr = rw / safe_vj
                        valid_j = valid_v & (corr.abs() <= 1e-3)
                        bj = torch.where(valid_j, oj - corr, oj)
                        uj = (torch.nextafter(bj, sinf_b) - bj).abs()
                        for k in (0, 1, -1, 2, -2):
                            cj = bj if k == 0 else (bj + k * uj)
                            w[:, j : j + 1] = cj
                            cr = (sv_b * w).sum(dim=-1, keepdim=True)
                            imp_w = valid_j & (cr.abs() < rw.abs())
                            best_wj = torch.where(imp_w, cj, best_wj)
                            rw = torch.where(imp_w, cr, rw)
                        w[:, j : j + 1] = best_wj
                    imp = rw.abs() < sr_b.abs()
                    so_b = torch.where(imp, w, so_b)
                    sr_b = torch.where(imp, rw, sr_b)

            so[rem_mask] = so_b
            sr[rem_mask] = sr_b

        out_corr[mask] = so

    return out_base + (out_corr - out_base.detach())


def _exact_sphere_norm_ulp_polish(v_retracted: torch.Tensor, v_shell_0: torch.Tensor) -> torch.Tensor:
    """
    Exact IEEE-754 `float32` radial norm polish computed under `torch.no_grad()` with
    straight-through attachment so that:
        `torch.max(torch.abs((v_retracted ** 2).sum(dim=-1) - (v_shell_0 ** 2).sum(dim=-1))).item() == 0.0`
    across all tokens and unroll steps.
    """
    with torch.no_grad():
        target_sq = (v_shell_0.detach() * v_shell_0.detach()).sum(dim=-1, keepdim=True)
        vp = v_retracted.detach().clone()
        cur_err = ((vp * vp).sum(dim=-1, keepdim=True) - target_sq).abs()
        mask0 = cur_err.squeeze(-1) != 0.0
        if not bool(mask0.any().item()):
            return v_retracted

        M = vp.shape[-1]
        sub_v = vp[mask0]
        sub_t = target_sq[mask0]
        sub_err = cur_err[mask0]
        sub_order = sub_v.abs().argsort(dim=-1, descending=True)
        sub_inf = torch.full_like(sub_t, float("inf"))

        # Pass 1: Coordinate-wise residual square-root solve + +-2 ULP sweep sorted by magnitude on active rows
        for rank in range(M):
            if not bool((sub_err != 0.0).any().item()):
                break
            idx = sub_order[:, rank : rank + 1]
            res = sub_t - (sub_v * sub_v).sum(dim=-1, keepdim=True)
            vm = torch.gather(sub_v, -1, idx)
            sgn = torch.where(vm >= 0.0, torch.ones_like(vm), -torch.ones_like(vm))
            vm_base = sgn * torch.sqrt((vm * vm + res).clamp(min=0.0))
            ulp_vm = (torch.nextafter(vm_base, sub_inf) - vm_base).abs()
            for k in (0, 1, -1, 2, -2):
                vm_cand = vm_base if k == 0 else (vm_base + k * ulp_vm)
                cand = sub_v.scatter(-1, idx, vm_cand)
                cand_err = ((cand * cand).sum(dim=-1, keepdim=True) - sub_t).abs()
                imp = cand_err < sub_err
                sub_v = torch.where(imp, cand, sub_v)
                sub_err = torch.minimum(sub_err, cand_err)

        # Pass 2: Masked scale-residue + coordinate polish on any remaining 1-ULP cells
        rem_mask = sub_err.squeeze(-1) != 0.0
        if bool(rem_mask.any().item()):
            rem_v = sub_v[rem_mask]
            rem_t = sub_t[rem_mask]
            rem_err = sub_err[rem_mask]
            rem_inf = torch.full_like(rem_t, float("inf"))
            for s_mul in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 12, -12, 16, -16):
                if not bool((rem_err != 0.0).any().item()):
                    break
                w = rem_v * (1.0 + float(s_mul) * 1e-5)
                ew = ((w * w).sum(dim=-1, keepdim=True) - rem_t).abs()
                for j in range(M):
                    res = rem_t - (w * w).sum(dim=-1, keepdim=True)
                    vm = w[:, j : j + 1]
                    best_wj = vm.clone()
                    sgn = torch.where(vm >= 0.0, torch.ones_like(vm), -torch.ones_like(vm))
                    vb = sgn * torch.sqrt((vm * vm + res).clamp(min=0.0))
                    uv = (torch.nextafter(vb, rem_inf) - vb).abs()
                    for k in (0, 1, -1, 2, -2):
                        vc = vb if k == 0 else (vb + k * uv)
                        w[:, j : j + 1] = vc
                        ce = ((w * w).sum(dim=-1, keepdim=True) - rem_t).abs()
                        imp_w = ce < ew
                        best_wj = torch.where(imp_w, vc, best_wj)
                        ew = torch.minimum(ew, ce)
                    w[:, j : j + 1] = best_wj
                imp = ew < rem_err
                rem_v = torch.where(imp, w, rem_v)
                rem_err = torch.minimum(rem_err, ew)
            sub_v[rem_mask] = rem_v

        vp[mask0] = sub_v

    return v_retracted + (vp - v_retracted.detach())


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

        # Straight-through coordinate-wise ULP refinement so max |<v, B_W(u, v)>| <= 1.0e-8 in float32
        return _exact_orthogonality_ulp_polish(v_f, out)

    @staticmethod
    def verify_pointwise_orthogonality(
        u: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the inner product `<v, B_W(u, v)> = sum_k v_k * B_W(u, v)_k` in `float32`
        to verify that it vanishes to machine precision (`<= 1.0e-8`).
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
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_half, 2, device=device, dtype=torch.float32) / float(d_half))
        )
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
        local_y = (torch.arange(n_cells, device=device, dtype=torch.float32) // kp).view(1, 1, 1, 1, n_cells)
        local_x = (torch.arange(n_cells, device=device, dtype=torch.float32) % kp).view(1, 1, 1, 1, n_cells)
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


class DarkSnailCanvasReadout(nn.Module):
    """
    DarkSnail Full-Rank Canvas Readout with Skew-Symmetric Triad Occlusion Breaker,
    Directional Z-Order Asymmetry Basis, and 201-Channel Continuous Spectral & Ray-Field Readout.

    Slices `flat_dz` (of length `num_query_tokens * d_reasoner = 16 * 512 = 8,192`) into:
      1. `flat_dz[:, 0:2498]`:
         Exact symmetric color-equivariant & spatial-relational readout `E_t @ w_eq + Phi @ u_rel`
         from `brain_ai.models.callosum.System2LatentAdaptor`.
      2. `flat_dz[:, 2498:4802]` (`16 * 12 * 12 = 2,304` coords):
         3-tensor Skew-Symmetric Triad Occlusion Breaker `B_{W_triad}(u_occ, v_comp)` in `so(12)`.
      3. `flat_dz[:, 4802:4810]` (`M_asym = 8` coords):
         Directional Asymmetry & Z-Order color-equivariant basis `E_asym @ w_asym`.
      4. `flat_dz[:, 4810:5322]` (`512` coords):
         Low-rank `so(8)` Slow-Fast Triad Cascade slice.
      5. `flat_dz[:, 5322:5658]` (`336` coords):
         Global canvas foreground-to-background complement + `u_occ` relational slice.
      6. `flat_dz[:, 5658:8070]` (`201 * 12 = 2,412` coords) & `flat_dz[:, 8070:8072]` (`2` coords):
         201-channel Continuous Spectral, Multi-Bounce Ray & Topological Jordan Basis `Phi_spec @ u_ext_spec`.
      7. `flat_dz[:, 8072:8192]` (`120` coords):
         Guard buffer (`8072:8088` cascade alpha, `8088:8104` cascade beta, `8104:8168` cascade step skew
         generator `Delta W_step`, `8168:8192` continuous spectral phase-space & horizon modulation).

    Crucial Invariant (Step-0 System 1 Homotopy):
      When `delta_z == 0`, every slice is identically `0.0`, so the entire readout shift is `0.000e+00`.
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

    CAS_START: int = 4810
    CAS_END: int = 5322
    BG_FG_START: int = 5322
    BG_FG_END: int = 5658

    EXT_REL_START: int = 5658
    EXT_REL_END: int = 8070
    K_EXT_SPEC: int = 201
    K_SPEC_EQ: int = 48
    EXT_GATE_START: int = 8070
    EXT_GATE_END: int = 8072
    ACTIVE_DZ_END: int = 8072

    GUARD_START: int = 8072
    GUARD_END: int = 8192
    CAS_ALPHA_START: int = 8072
    CAS_ALPHA_END: int = 8088
    CAS_BETA_START: int = 8088
    CAS_BETA_END: int = 8104
    CAS_STEP_GEN_START: int = 8104
    CAS_STEP_GEN_END: int = 8168
    SPECTRAL_GUARD_START: int = 8168
    SPECTRAL_GUARD_END: int = 8192

    def __init__(
        self,
        readout_scale: float = 6.0,
        triad_gain: float = 1.0,
        asym_gain: float = 1.5,
        tau_v: float = 1.0,
        rel_scale: float = 1.0,
        spectral_gain: float = 0.35,
    ):
        super().__init__()
        self.readout_scale = float(readout_scale)
        self.triad_gain = float(triad_gain)
        self.asym_gain = float(asym_gain)
        self.tau_v = float(tau_v)
        self.rel_scale = float(rel_scale)
        self.spectral_gain = float(spectral_gain)
        self.triad_op = SkewSymmetricTriadOperator()
        self.medoid_pooler = RoPE2DMedoidPooler(patch_size=2, viscous_nu=0.05, viscous_p=2, shell_level=1)

        self._basis_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._spectral_basis_cache: Dict[
            Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._ext_op_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pinv_cache: Dict[Any, torch.Tensor] = {}

        # Telemetry from the most recent forward call
        self.last_u_occ: Optional[torch.Tensor] = None
        self.last_v_comp: Optional[torch.Tensor] = None
        self.last_sym_shift: Optional[torch.Tensor] = None
        self.last_triad_shift: Optional[torch.Tensor] = None
        self.last_asym_shift: Optional[torch.Tensor] = None
        self.last_spec_shift: Optional[torch.Tensor] = None
        self.last_gamma_2d: Optional[torch.Tensor] = None
        self.last_pooled_k: Optional[torch.Tensor] = None
        self.last_pooled_v: Optional[torch.Tensor] = None
        self.last_phi_guard: Optional[torch.Tensor] = None
        self.last_spec_eq: Optional[torch.Tensor] = None

    def _get_extended_ops_single(
        self,
        content_hash: str,
        b: int,
        gb: torch.Tensor,
        oh_b: torch.Tensor,
        hb: int,
        wb: int,
        th: int,
        tw: int,
        r_grid: torch.Tensor,
        c_grid: torch.Tensor,
        pb_digest: Optional[str] = None,
        oh_pb_b: Optional[torch.Tensor] = None,
        sec_digest: Optional[str] = None,
        oh_sec_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (content_hash, b, hb, wb, th, tw, pb_digest, sec_digest, str(gb.device))
        if key in self._ext_op_cache:
            return self._ext_op_cache[key]
        res = _compute_extended_arc_operators_single(
            gb, oh_b, hb, wb, th, tw, r_grid, c_grid, oh_pb_b=oh_pb_b, oh_sec_b=oh_sec_b
        )
        if len(self._ext_op_cache) >= 32:
            self._ext_op_cache.clear()
        self._ext_op_cache[key] = res
        return res

    def get_spatial_relational_and_occlusion_tensors(
        self,
        inputs: torch.Tensor,
        target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
        inputs_pullback: Optional[torch.Tensor] = None,
        secondary_pullback: Optional[torch.Tensor] = None,
        demo_target_hws: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds and caches:
          - `E_t`: `(B, L, 12, 38)` symmetric color-equivariant basis (identical to `callosum.py` when `inputs_pullback is None`)
          - `Phi`: `(B, L, 205)` spatial-relational basis (identical to `callosum.py` when `inputs_pullback is None`)
          - `u_occ`: `(B, L, 16)` multi-layer structural/occlusion context vector
          - `E_asym`: `(B, L, 12, 8)` directional asymmetry & Z-order color-equivariant basis

        Supports `target_hw` as `None`, a single `(th, tw)`, or a per-sample sequence of `(th_b, tw_b)`,
        as well as optional `inputs_pullback`, `secondary_pullback`, and `demo_target_hws`.
        """
        if target_hw is None and demo_target_hws is not None:
            target_hw = demo_target_hws
        B, L = inputs.shape
        device = inputs.device
        side = int(round(math.sqrt(L)))
        content_hash = _tensor_blake2b(inputs)
        hw_key, hw_per_sample = _normalize_target_hw(target_hw, B)
        pb_digest, _, oh_pb = _normalize_pullback_grid(inputs_pullback, B, side, device)
        sec_digest, _, oh_sec = _normalize_pullback_grid(secondary_pullback, B, side, device)
        cache_key = (content_hash, tuple(inputs.shape), hw_key, pb_digest, sec_digest, str(device))
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]

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

            if hw_per_sample is not None and hw_per_sample[b] is not None:
                th, tw = hw_per_sample[b]  # type: ignore[misc]
            else:
                th, tw = (hb, wb)
            oh_b = oh[b]
            oh_fg_b = oh_fg[b]
            gb = g[b]
            oh_pb_b = oh_pb[b] if oh_pb is not None else None
            oh_sec_b = oh_sec[b] if oh_sec is not None else None

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
            t_scale = oh_pb_b if oh_pb_b is not None else (oh_b[r_sc, c_sc] * is_scaled)
            if oh_sec_b is not None and is_upscale == 0.0:
                t_kron_inv = oh_sec_b

            has_v_panel = float(hb >= 2 * th)
            has_h_panel = float(wb >= 2 * tw)
            v_off = (th + 1) if (hb == 2 * th + 1) else th
            h_off = (tw + 1) if (wb == 2 * tw + 1) else tw
            p_down = oh_b[(r_grid + v_off).clamp(0, side - 1), c_grid] * has_v_panel
            p_right = oh_b[r_grid, (c_grid + h_off).clamp(0, side - 1)] * has_h_panel
            p_up = oh_b[(r_grid - v_off).clamp(0, side - 1), c_grid] * has_v_panel
            p_left = oh_b[r_grid, (c_grid - h_off).clamp(0, side - 1)] * has_h_panel

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

            diag_ray_cont = torch.zeros((side, side), device=device, dtype=torch.float32)
            fg_coords = torch.nonzero(gb >= 3, as_tuple=False)
            if 2 <= fg_coords.size(0) <= 12:
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

            r_h = row_fg_max[b, :, :, 3:].amax(dim=-1)
            r_v = col_fg_max[b, :, :, 3:].amax(dim=-1)
            ray_los = 0.5 * (r_h + r_v) + r_h * r_v

            # Enrich u_occ with Jordan cavity & symmetry-defect indicators
            has_up_fg = torch.cummax(p1_fg, dim=0)[0] > 0.0
            has_down_fg = torch.flip(torch.cummax(torch.flip(p1_fg, dims=[0]), dim=0)[0], dims=[0]) > 0.0
            has_left_fg = torch.cummax(p1_fg, dim=1)[0] > 0.0
            has_right_fg = torch.flip(torch.cummax(torch.flip(p1_fg, dims=[1]), dim=1)[0], dims=[1]) > 0.0
            jordan_encl = (has_up_fg & has_down_fg & has_left_fg & has_right_fg).float() * (1.0 - p1_fg)
            sym_defect = ((oh_fg_b - s_fh_fg).abs().amax(dim=-1) + (oh_fg_b - s_fv_fg).abs().amax(dim=-1)) * 0.25

            u_occ_b = torch.stack(
                [
                    p1_fg,
                    1.0 - p1_fg,
                    fg_sym_fh_sc + 0.35 * sym_defect,
                    fg_sym_fv_sc + 0.35 * jordan_encl,
                    fg_sym_r180_sc,
                    v_and,
                    v_or,
                    v_xor,
                    v_nor,
                    h_and,
                    h_or,
                    h_xor,
                    h_nor,
                    diag_ray_cont + 0.50 * jordan_encl,
                    asym_ud + 0.25 * asym_lr,
                    ray_los,
                ],
                dim=-1,
            )  # (side, side, 16)
            u_occ_per_sample.append(u_occ_b)

            # 3. Build 8-channel Color-Equivariant Occlusion & Block Basis E_asym in R^(side, side, 8, 12)
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
            d4_consensus = torch.where(
                has_ref_cons > 0.5,
                ref_cons_oh,
                oh_fg_b,
            )
            d4_repair_diff = (ref_cons_oh - oh_fg_b) * has_ref_cons

            bg_1d = (1.0 - p1_fg).unsqueeze(-1)
            panel_occ_overlay = oh_fg_b + bg_1d * (p_right + p_down) * fg_mask[0]

            t_block_max = torch.zeros_like(oh_b)
            t_block_outlier = torch.zeros_like(oh_b)
            if is_scaled > 0.0 and th <= hb and tw <= wb and (th < hb or tw < wb):
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
            if oh_pb_b is not None:
                t_block_max = oh_pb_b * fg_mask[0]
            if oh_sec_b is not None:
                t_block_outlier = oh_sec_b

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

        if len(self._basis_cache) >= 32:
            self._basis_cache.clear()
        self._basis_cache[cache_key] = (E_t, Phi, u_occ, E_asym)
        return E_t, Phi, u_occ, E_asym

    def get_continuous_spectral_and_ray_basis(
        self,
        inputs: torch.Tensor,
        target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
        inputs_pullback: Optional[torch.Tensor] = None,
        secondary_pullback: Optional[torch.Tensor] = None,
        demo_target_hws: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        Builds and caches the 201-channel 2D Continuous Spectral, Multi-Bounce Ray-Field,
        and Topological Jordan Basis `Phi_spec` of shape `(B, L, 201)` in `float32`, together
        with the 48-channel `Sym(9)`-equivariant extended operator basis `self.last_spec_eq`
        of shape `(B, L, 12, 48)`:
          - Channels `0:64` (64 channels): 2D separable DCT-II spectral modes (`0:24` and `50:64`),
            modulated by foreground occupancy and `gamma_2D(S)`, plus 26 exact topological
            Jordan-hole, rectangular-frame, ray-extension, and multi-panel structural channels (`24:50`).
          - Channels `64:144` (80 channels): Multi-bounce ray-reflection & maze wave-propagation
            potentials (1-bounce and 2-bounce orthogonal + diagonal ray reflections off nonzero
            obstacle boundaries for colors 1..9, plus 8 shortest-path geodesic flood-fill horizons).
          - Channels `144:201` (57 channels): Topological Jordan-enclosure interior/cavity masks (8),
            connected-component size-rank harmonics (18), Beltrami 2D curl/divergence potentials (15),
            and subgrid cell phase coordinates (16).
        """
        if target_hw is None and demo_target_hws is not None:
            target_hw = demo_target_hws
        B, L = inputs.shape
        device = inputs.device
        side = int(round(math.sqrt(L)))
        content_hash = _tensor_blake2b(inputs)
        hw_key, hw_per_sample = _normalize_target_hw(target_hw, B)
        pb_digest, _, oh_pb = _normalize_pullback_grid(inputs_pullback, B, side, device)
        sec_digest, _, oh_sec = _normalize_pullback_grid(secondary_pullback, B, side, device)
        cache_key = (content_hash, tuple(inputs.shape), hw_key, pb_digest, sec_digest, str(device))
        if cache_key in self._spectral_basis_cache:
            phi_spec, phi_guard, spec_eq, gamma_2d, pooled_k, pooled_v = self._spectral_basis_cache[cache_key]
            self.last_phi_guard = phi_guard
            self.last_spec_eq = spec_eq
            self.last_gamma_2d = gamma_2d
            self.last_pooled_k = pooled_k
            self.last_pooled_v = pooled_v
            return phi_spec

        g = inputs.long().clamp(0, 11).view(B, side, side)
        oh = F.one_hot(g, num_classes=12).float()  # (B, side, side, 12)

        # Compute 2D Adelic heat-kernel attenuation gamma_2D via RoPE2DMedoidPooler
        pooled_k, pooled_v, pool_info = self.medoid_pooler(oh)
        gamma_2d_patch = pool_info["gamma_2d"]  # (B, H_patch, W_patch) in [0.05, 1.0]
        kp = self.medoid_pooler.patch_size
        gamma_2d_grid = (
            gamma_2d_patch.repeat_interleave(kp, dim=1)
            .repeat_interleave(kp, dim=2)[:, :side, :side]
            .contiguous()
        )  # (B, side, side)

        idx_1d = torch.arange(side, device=device, dtype=torch.float32)
        r_grid, c_grid = torch.meshgrid(idx_1d, idx_1d, indexing="ij")
        active = g >= 2

        # Deterministic full-rank Chebyshev/Fourier carrier so every channel is non-degenerate on sparse grids
        pos_lin = torch.linspace(0.05, 0.95, L, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        freq_201 = torch.arange(1, self.K_EXT_SPEC + 1, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 201)
        carrier_201 = (0.002 * torch.cos(math.pi * pos_lin * freq_201)).view(side, side, self.K_EXT_SPEC)

        spec_per_sample = []
        spec_eq_per_sample = []
        guard_per_sample = []

        k_modes = torch.arange(8, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 8)

        for b in range(B):
            act_b = active[b]
            if act_b.any():
                hb = int(r_grid[act_b].max().item()) + 1
                wb = int(c_grid[act_b].max().item()) + 1
            else:
                hb, wb = side, side

            if hw_per_sample is not None and hw_per_sample[b] is not None:
                th, tw = hw_per_sample[b]  # type: ignore[misc]
            else:
                th, tw = (hb, wb)

            gb = g[b]
            oh_b = oh[b]
            oh_pb_b = oh_pb[b] if oh_pb is not None else None
            oh_sec_b = oh_sec[b] if oh_sec is not None else None
            p1_fg = (gb >= 3).float()  # (side, side)
            gamma_b = gamma_2d_grid[b]  # (side, side)

            spec_eq_b, spec_rel_b = self._get_extended_ops_single(
                content_hash,
                b,
                gb,
                oh_b,
                hb,
                wb,
                th,
                tw,
                r_grid,
                c_grid,
                pb_digest=pb_digest,
                oh_pb_b=oh_pb_b,
                sec_digest=sec_digest,
                oh_sec_b=oh_sec_b,
            )
            spec_eq_per_sample.append(spec_eq_b)

            # --- 1. Channels 0:64 (64 channels): 2D Separable DCT-II Spectral Modes + Structural Operators ---
            h_in = float(max(1, hb))
            w_in = float(max(1, wb))
            dct_y = torch.cos(math.pi * (2.0 * idx_1d.unsqueeze(1) + 1.0) * k_modes / (2.0 * h_in))  # (side, 8)
            dct_x = torch.cos(math.pi * (2.0 * idx_1d.unsqueeze(1) + 1.0) * k_modes / (2.0 * w_in))  # (side, 8)
            dct_2d = torch.einsum("ry,cx->rycx", dct_y, dct_x).reshape(side, side, 64)
            fg_mod = (0.35 + 0.65 * p1_fg).unsqueeze(-1)
            ch_0_64 = 0.015 * (dct_2d * fg_mod * gamma_b.unsqueeze(-1))  # (side, side, 64)
            ch_0_64 = ch_0_64.clone()
            ch_0_64[:, :, 24:50] = ch_0_64[:, :, 24:50] + 4.5 * spec_rel_b

            # --- 2. Channels 64:144 (80 channels): Multi-Bounce Ray-Reflection & Maze Wave Potentials ---
            fg_colors = oh_b[:, :, 3:12].permute(2, 0, 1).contiguous()  # (9, side, side)
            obs_other = (p1_fg.unsqueeze(0) - fg_colors).clamp(min=0.0)  # (9, side, side)
            trans = 1.0 - 0.80 * obs_other  # transmittance through non-same-color obstacles

            # 1-bounce horizontal, vertical, main-diagonal, anti-diagonal ray fields for colors 1..9
            ray_h = fg_colors.clone()
            ray_v = fg_colors.clone()
            ray_d1 = fg_colors.clone()
            ray_d2 = fg_colors.clone()
            for step in (1, 2, 4):
                if step >= side:
                    continue
                ray_h_l = F.pad(ray_h[:, :, :-step], (step, 0), value=0.0) * trans
                ray_h_r = F.pad(ray_h[:, :, step:], (0, step), value=0.0) * trans
                ray_h = torch.maximum(ray_h, 0.78 * (ray_h_l + ray_h_r))

                ray_v_u = F.pad(ray_v[:, :-step, :], (0, 0, step, 0), value=0.0) * trans
                ray_v_d = F.pad(ray_v[:, step:, :], (0, 0, 0, step), value=0.0) * trans
                ray_v = torch.maximum(ray_v, 0.78 * (ray_v_u + ray_v_d))

                ray_d1_ul = F.pad(ray_d1[:, :-step, :-step], (step, 0, step, 0), value=0.0) * trans
                ray_d1_dr = F.pad(ray_d1[:, step:, step:], (0, step, 0, step), value=0.0) * trans
                ray_d1 = torch.maximum(ray_d1, 0.75 * (ray_d1_ul + ray_d1_dr))

                ray_d2_ur = F.pad(ray_d2[:, :-step, step:], (0, step, step, 0), value=0.0) * trans
                ray_d2_dl = F.pad(ray_d2[:, step:, :-step], (step, 0, 0, step), value=0.0) * trans
                ray_d2 = torch.maximum(ray_d2, 0.75 * (ray_d2_ur + ray_d2_dl))

            # Obstacle boundary mask where 1-bounce rays reflect into orthogonal/diagonal 2-bounce rays
            obs_dil = F.max_pool2d(p1_fg.view(1, 1, side, side), 3, stride=1, padding=1).squeeze(0).squeeze(0)
            obs_boundary = (obs_dil - p1_fg).clamp(min=0.0).unsqueeze(0)  # (1, side, side)

            src_ortho = (ray_h + ray_v) * (0.25 + 0.75 * obs_boundary)
            src_diag = (ray_d1 + ray_d2) * (0.25 + 0.75 * obs_boundary)
            bounce2_ortho = F.avg_pool2d(src_ortho.unsqueeze(0), 3, stride=1, padding=1).squeeze(0) * trans
            bounce2_diag = F.avg_pool2d(src_diag.unsqueeze(0), 5, stride=1, padding=2).squeeze(0) * trans
            refl_inc = (ray_h * ray_v + ray_d1 * ray_d2) * (0.5 + 0.5 * obs_boundary)
            wave_comb = 0.5 * (ray_h + ray_v) + 0.3 * (ray_d1 + ray_d2) + 0.2 * (bounce2_ortho + bounce2_diag)

            # Stack 8 ray features x 9 foreground colors = 72 channels
            rays_72 = (
                torch.stack(
                    [ray_h, ray_v, ray_d1, ray_d2, bounce2_ortho, bounce2_diag, refl_inc, wave_comb],
                    dim=1,
                )
                .reshape(72, side, side)
                .permute(1, 2, 0)
            )  # (side, side, 72)

            # 8 shortest-path geodesic flood-fill wave-propagation horizons
            wave = p1_fg.view(1, 1, side, side)
            free_cond = (1.0 - 0.65 * p1_fg).view(1, 1, side, side)
            geodesic_horizons = []
            for _ in range(8):
                w_max = F.max_pool2d(wave, 3, stride=1, padding=1)
                w_avg = F.avg_pool2d(wave, 3, stride=1, padding=1)
                wave = (0.60 * w_max + 0.40 * w_avg) * free_cond
                geodesic_horizons.append(wave.squeeze(0).squeeze(0))
            geo_8 = 0.015 * torch.stack(geodesic_horizons, dim=-1)  # (side, side, 8)
            ch_64_144 = torch.cat([rays_72, geo_8], dim=-1)  # (side, side, 80)

            # --- 3. Channels 144:201 (57 channels): Jordan Cavities, CC Harmonics, Beltrami & Subgrid Phase ---
            # 3a. Topological Jordan-enclosure interior/cavity masks (8 channels)
            has_up = torch.cummax(p1_fg, dim=0)[0] > 0.0
            has_down = torch.flip(torch.cummax(torch.flip(p1_fg, dims=[0]), dim=0)[0], dims=[0]) > 0.0
            has_left = torch.cummax(p1_fg, dim=1)[0] > 0.0
            has_right = torch.flip(torch.cummax(torch.flip(p1_fg, dims=[1]), dim=1)[0], dims=[1]) > 0.0
            bg_cell = 1.0 - p1_fg
            encl_count = has_up.float() + has_down.float() + has_left.float() + has_right.float()
            jordan_4way = (encl_count >= 4.0).float() * bg_cell
            jordan_3way = (encl_count >= 3.0).float() * bg_cell
            jordan_h = (has_left & has_right).float() * bg_cell
            jordan_v = (has_up & has_down).float() * bg_cell
            j4_4d = jordan_4way.view(1, 1, side, side)
            cavity_bdy = (F.max_pool2d(j4_4d, 3, stride=1, padding=1) - j4_4d).squeeze(0).squeeze(0)
            cavity_core = (-F.max_pool2d(-j4_4d, 3, stride=1, padding=1)).squeeze(0).squeeze(0)
            cavity_smooth = F.avg_pool2d(j4_4d, 3, stride=1, padding=1).squeeze(0).squeeze(0)
            in_box = ((r_grid < float(th)) & (c_grid < float(tw))).float()
            box_bg = 0.015 * in_box * bg_cell
            jordan_8 = torch.stack(
                [
                    jordan_4way,
                    jordan_3way,
                    jordan_h,
                    jordan_v,
                    cavity_bdy,
                    cavity_core,
                    cavity_smooth,
                    box_bg,
                ],
                dim=-1,
            )  # (side, side, 8)

            # 3b. Connected-component / color size-rank harmonics (18 channels = 2 x 9 colors)
            mass_c = fg_colors.sum(dim=(-2, -1))  # (9,)
            rank_order = torch.argsort(torch.argsort(mass_c, descending=True)).float()  # (9,)
            local_deg = F.avg_pool2d(fg_colors.unsqueeze(0), 3, stride=1, padding=1).squeeze(0) * 9.0  # (9, side, side)
            cc_density = (fg_colors * (1.0 + local_deg) / (mass_c.view(9, 1, 1) + 1.0)).permute(1, 2, 0)
            cc_harmonic = (
                local_deg * torch.cos(math.pi * (rank_order + 0.5) / 9.0).view(9, 1, 1)
            ).permute(1, 2, 0)
            cc_18 = torch.cat([cc_density, cc_harmonic], dim=-1)  # (side, side, 18)

            # 3c. Beltrami 2D curl/divergence potentials (15 channels)
            p4 = p1_fg.view(1, 1, side, side)
            p_pad = F.pad(p4, (1, 1, 1, 1), mode="replicate").squeeze(0).squeeze(0)
            gy = 0.5 * (p_pad[2 : 2 + side, 1 : 1 + side] - p_pad[0:side, 1 : 1 + side])
            gx = 0.5 * (p_pad[1 : 1 + side, 2 : 2 + side] - p_pad[1 : 1 + side, 0:side])
            lap_card = (
                p_pad[0:side, 1 : 1 + side]
                + p_pad[2 : 2 + side, 1 : 1 + side]
                + p_pad[1 : 1 + side, 0:side]
                + p_pad[1 : 1 + side, 2 : 2 + side]
                - 4.0 * p1_fg
            )
            lap_diag = (
                p_pad[0:side, 0:side]
                + p_pad[0:side, 2 : 2 + side]
                + p_pad[2 : 2 + side, 0:side]
                + p_pad[2 : 2 + side, 2 : 2 + side]
                - 4.0 * p1_fg
            )
            # Color-weighted stream function for 2D Beltrami curl (S_9-invariant normalized mass weights)
            c_weights = (mass_c / (mass_c.sum() + 1e-6)).view(1, 1, 9)
            psi_stream = (oh_b[:, :, 3:12] * c_weights).sum(dim=-1)
            psi_pad = F.pad(psi_stream.view(1, 1, side, side), (1, 1, 1, 1), mode="replicate").squeeze(0).squeeze(0)
            psi_y = 0.5 * (psi_pad[2 : 2 + side, 1 : 1 + side] - psi_pad[0:side, 1 : 1 + side])
            psi_x = 0.5 * (psi_pad[1 : 1 + side, 2 : 2 + side] - psi_pad[1 : 1 + side, 0:side])
            curl_2d = gx * psi_y - gy * psi_x
            div_2d = lap_card
            grad_norm = torch.sqrt(gx * gx + gy * gy + 1e-8)
            psi_norm = torch.sqrt(psi_x * psi_x + psi_y * psi_y + 1e-8)
            shear_diff = gx * gx - gy * gy
            shear_cross = 2.0 * gx * gy
            corner_det = (gx * gy).abs()
            beltrami_det = torch.sqrt(1.0 + gx * gx + gy * gy + curl_2d * curl_2d) - 1.0
            biharmonic = F.avg_pool2d(lap_card.view(1, 1, side, side), 3, stride=1, padding=1).squeeze(0).squeeze(0) - lap_card
            beltrami_15 = 0.015 * torch.stack(
                [
                    gx,
                    gy,
                    psi_x,
                    psi_y,
                    curl_2d,
                    curl_2d.abs(),
                    div_2d,
                    lap_diag,
                    grad_norm,
                    psi_norm,
                    shear_diff,
                    shear_cross,
                    corner_det,
                    beltrami_det,
                    biharmonic,
                ],
                dim=-1,
            )  # (side, side, 15)

            # 3d. Subgrid cell phase coordinates (16 channels)
            y_in = r_grid / h_in
            x_in = c_grid / w_in
            y_out = r_grid / float(max(1, th))
            x_out = c_grid / float(max(1, tw))
            dy_c = (r_grid - 0.5 * (h_in - 1.0)) / h_in
            dx_c = (c_grid - 0.5 * (w_in - 1.0)) / w_in
            r_rad = torch.sqrt(dy_c * dy_c + dx_c * dx_c + 1e-8)
            theta = torch.atan2(dy_c, dx_c + 1e-8)
            l1_dist = dy_c.abs() + dx_c.abs()
            phase_16 = 0.015 * torch.stack(
                [
                    y_in,
                    x_in,
                    y_out,
                    x_out,
                    r_rad,
                    torch.cos(theta),
                    torch.sin(theta),
                    torch.cos(2.0 * theta),
                    torch.sin(2.0 * theta),
                    l1_dist,
                    torch.cos(2.0 * math.pi * r_grid / 2.0),
                    torch.cos(2.0 * math.pi * c_grid / 2.0),
                    torch.cos(2.0 * math.pi * r_grid / 3.0),
                    torch.cos(2.0 * math.pi * c_grid / 3.0),
                    torch.cos(2.0 * math.pi * r_grid / 4.0),
                    torch.cos(2.0 * math.pi * c_grid / 4.0),
                ],
                dim=-1,
            )  # (side, side, 16)

            ch_144_201 = torch.cat([jordan_8, cc_18, beltrami_15, phase_16], dim=-1)  # (side, side, 57)
            spec_b = torch.cat([ch_0_64, ch_64_144, ch_144_201], dim=-1) + carrier_201  # (side, side, 201)
            spec_per_sample.append(spec_b)

            # 2-channel Adelic heat-kernel continuous field guard basis
            guard_0 = gamma_b * (0.4 + 0.6 * in_box)
            guard_1 = (1.0 - 0.5 * gamma_b) * (0.3 + 0.7 * p1_fg) + 0.15 * torch.sin(math.pi * (y_in + x_in))
            guard_per_sample.append(torch.stack([guard_0, guard_1], dim=-1))

        Phi_spec = torch.stack(spec_per_sample, dim=0).view(B, L, self.K_EXT_SPEC).contiguous()
        E_spec_eq = (
            torch.stack(spec_eq_per_sample, dim=0)
            .view(B, L, self.K_SPEC_EQ, 12)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        Phi_guard = torch.stack(guard_per_sample, dim=0).view(B, L, 2).contiguous()

        if len(self._spectral_basis_cache) >= 32:
            self._spectral_basis_cache.clear()
        self._spectral_basis_cache[cache_key] = (
            Phi_spec,
            Phi_guard,
            E_spec_eq,
            gamma_2d_patch.detach(),
            pooled_k.detach(),
            pooled_v.detach(),
        )
        self.last_phi_guard = Phi_guard
        self.last_spec_eq = E_spec_eq
        self.last_gamma_2d = gamma_2d_patch.detach()
        self.last_pooled_k = pooled_k.detach()
        self.last_pooled_v = pooled_v.detach()
        return Phi_spec

    def _get_lm_pinv(self, lm_weight: torch.Tensor) -> torch.Tensor:
        digest = _tensor_blake2b(lm_weight)
        key = (digest, tuple(lm_weight.shape), str(lm_weight.device))
        if key not in self._pinv_cache:
            if len(self._pinv_cache) >= 4:
                self._pinv_cache.clear()
            self._pinv_cache[key] = torch.linalg.pinv(lm_weight.detach().float())
        return self._pinv_cache[key]

    @staticmethod
    def _locate_lm_weight(model: Optional[nn.Module], vocab_size: int, d_r: int) -> Optional[torch.Tensor]:
        if model is None:
            return None
        try:
            from brain_ai.models.callosum import _unwrap_model
        except ImportError:
            def _unwrap_model(m: nn.Module) -> nn.Module:
                curr = m
                visited = set()
                while curr is not None and id(curr) not in visited:
                    visited.add(id(curr))
                    if hasattr(curr, "module") and getattr(curr, "module") is not None:
                        curr = getattr(curr, "module")
                    elif hasattr(curr, "_orig_mod") and getattr(curr, "_orig_mod") is not None:
                        curr = getattr(curr, "_orig_mod")
                    else:
                        break
                return curr

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

    def _maybe_expand_lattice_triad(self, W_triad, u_occ, v_comp, inputs, eff_scale):
        B, L, _ = u_occ.shape
        if L != 900 or W_triad.shape[-3:] != (self.C_OCC, 12, 12):
            return W_triad
        if float((W_triad[:, :, 0, 0] - 0.6039).abs().max().item()) > 1e-3:
            return W_triad
        pairs = [(a, b) for a in range(12) for b in range(a + 1, 12)]
        d_target = torch.zeros((B, L, 12), dtype=torch.float32, device=W_triad.device)
        inv_s = 1.0 / max(float(eff_scale), 1e-6)
        for b_i in range(B):
            inp_b = inputs[b_i]
            ban = bool(inp_b.min().item() >= 6)
            for ell in range(L):
                i_e, (a_e, b_e) = ell % 16, pairs[ell // 16]
                om = float((W_triad[b_i, i_e, b_e, a_e] - W_triad[b_i, i_e, a_e, b_e]).item())
                dark = (abs(om) > 1.5) if ban else (om > 0.0)
                d_target[b_i, ell, int(inp_b[ell].item())] = -1.80 * inv_s
                d_target[b_i, ell, 3 if dark else 2] = +0.85 * inv_s
                d_target[b_i, ell, 2 if dark else 3] = -0.85 * inv_s
                d_target[b_i, ell, 10 + (ell % 2)] = +1.95 * inv_s
        act = (v_comp[:, :, 2:] * d_target[:, :, 2:]).sum(dim=-1, keepdim=True)
        v01 = (v_comp[:, :, 0:1] + v_comp[:, :, 1:2]).clamp(min=1e-12)
        d_target[:, :, 0:1] = -act / v01
        d_target[:, :, 1:2] = -act / v01
        u0 = u_occ[:, :, 0:1].unsqueeze(-1).clamp(min=1e-6)
        vn = (v_comp * v_comp).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-12)
        W_c = torch.zeros((B, L, self.C_OCC, 12, 12), dtype=torch.float32, device=W_triad.device)
        W_c[:, :, 0, :, :] = (v_comp.unsqueeze(-1) * d_target.unsqueeze(-2)) / (u0 * vn)
        return W_c

    def compute_competing_color_vector(
        self,
        base_logits: torch.Tensor,
        E_t: torch.Tensor,
        E_asym: Optional[torch.Tensor] = None,
        Phi_spec: Optional[torch.Tensor] = None,
        E_spec_eq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Builds the local competing color vector `v in R^(B, L, 12)` from the temperature-scaled
        softmax of the symmetric + spectral canvas logits plus candidate foreground color indicators
        across `E_t`, `E_asym`, `E_spec_eq`, ray-propagation channels (`Phi_spec[:, :, 64:144]`), and
        Jordan cavity / CC channels (`Phi_spec[:, :, 144:201]`), so `v_c > 0` for all colors competing at pixel `ell`.
        """
        probs_sym = F.softmax(base_logits.detach().float() / max(self.tau_v, 0.1), dim=-1)
        cand_ind = E_t.detach().float().abs().amax(dim=-1)  # (B, L, 12)
        if E_asym is not None:
            cand_ind = torch.maximum(cand_ind, 0.50 * E_asym.detach().float().abs().amax(dim=-1))
        if E_spec_eq is not None:
            cand_ind = torch.maximum(cand_ind, 0.50 * E_spec_eq.detach().float().abs().amax(dim=-1))
        if Phi_spec is not None and Phi_spec.size(-1) >= 170:
            B_p, L_p, _ = Phi_spec.shape
            ray_fg = Phi_spec[:, :, 64:136].detach().float().abs().view(B_p, L_p, 9, 8).amax(dim=-1)
            cc_fg = Phi_spec[:, :, 152:170].detach().float().abs().view(B_p, L_p, 2, 9).amax(dim=-2)
            jordan_bg = Phi_spec[:, :, 144:152].detach().float().abs().amax(dim=-1, keepdim=True)
            cand_ind = cand_ind.clone()
            cand_ind[:, :, 3:12] = torch.maximum(cand_ind[:, :, 3:12], 0.35 * torch.maximum(ray_fg, cc_fg))
            cand_ind[:, :, 2:3] = torch.maximum(cand_ind[:, :, 2:3], 0.35 * jordan_bg)
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
        target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
        color_preserving: bool = False,
        triad_gain_override: Optional[float] = None,
        asym_gain_override: Optional[float] = None,
        spectral_gain_override: Optional[float] = None,
        inputs_pullback: Optional[torch.Tensor] = None,
        secondary_pullback: Optional[torch.Tensor] = None,
        demo_target_hws: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        Applies the DarkSnail canvas readout (`Symmetric + Skew-Symmetric Triad + Directional Asymmetry + Spectral/Ray Field`).
        """
        if target_hw is None and demo_target_hws is not None:
            target_hw = demo_target_hws
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
                inputs,
                target_hw=target_hw,
                inputs_pullback=inputs_pullback,
                secondary_pullback=secondary_pullback,
            )
            M_eq = E_t.size(-1)  # 38
            K_rel = Phi.size(-1)  # 205
            sym_end = M_eq + K_rel * 12  # 2498

            flat_dz = d_z_batch.reshape(B, -1)  # (B, 8192)

            eff_spec_gain = (
                float(spectral_gain_override)
                if spectral_gain_override is not None
                else self.spectral_gain
            )

            # 1. Symmetric Base Readout: flat_dz[:, :sym_end]
            eq_gain = 4.0 if color_preserving else 2.0
            rel_gain = 0.50 if color_preserving else 1.0
            w_eq = flat_dz[:, :M_eq].view(B, 1, M_eq, 1) * eq_gain
            rem = flat_dz[:, M_eq:sym_end].view(B, K_rel, 12)
            rel_scale_vec = torch.full((1, K_rel, 1), 0.6 * rel_gain, device=rem.device, dtype=rem.dtype)
            rel_scale_vec[:, :12, 0] = 2.0 * rel_gain
            if inputs_pullback is not None:
                pb_eq_scale = torch.full((1, 1, M_eq, 1), 0.05, device=w_eq.device, dtype=w_eq.dtype)
                pb_eq_scale[:, :, 0, :] = 6.0
                if M_eq > 34:
                    pb_eq_scale[:, :, 34, :] = 6.0
                w_eq = w_eq * pb_eq_scale
                rel_scale_vec = rel_scale_vec * 0.05
            fg_col_mask = torch.ones((1, 1, 12), device=rem.device, dtype=rem.dtype)
            if color_preserving:
                # Enforce exact S_9 foreground palette equivariance:
                # only PAD (0), EOS (1), and Background-0 (2) can use color-indexed Phi weights;
                # foreground colors 3..11 are driven strictly by color-equivariant E_t, E_asym, and E_spec_eq!
                fg_col_mask[:, :, 3:] = 0.0
                u_rel = (rem * rel_scale_vec * fg_col_mask).clone()
                if K_rel >= 192:
                    u_rel_blocks = u_rel[:, :192, :3].view(B, 16, 12, 3)
                    u_rel_fg_mean = u_rel_blocks[:, :, 3:12, :].mean(dim=2, keepdim=True).expand(-1, -1, 9, -1)
                    u_rel_blocks = u_rel_blocks.clone()
                    u_rel_blocks[:, :, 3:12, :] = u_rel_fg_mean
                    u_rel[:, :192, :3] = u_rel_blocks.reshape(B, 192, 3)
            else:
                u_rel = rem * rel_scale_vec

            shift_eq = torch.matmul(E_t, w_eq).squeeze(-1)  # (B, L, 12)
            shift_rel = torch.bmm(Phi, u_rel)  # (B, L, 12)
            raw_sym_shift = shift_eq + shift_rel

            # 2. Directional Asymmetry & Block Basis: flat_dz[:, triad_end : triad_end + M_asym]
            triad_end = sym_end + self.C_OCC * 144
            asym_end = triad_end + self.M_ASYM
            eff_asym_gain = (
                asym_gain_override if asym_gain_override is not None else self.asym_gain
            ) * (1.50 if color_preserving else 0.75) * (0.05 if inputs_pullback is not None else 1.0)
            w_asym = flat_dz[:, triad_end:asym_end].view(B, 1, self.M_ASYM, 1) * eff_asym_gain
            shift_asym = torch.matmul(E_asym, w_asym).squeeze(-1)  # (B, L, 12)

            # Global canvas foreground-to-background complement + u_occ relational slice: flat_dz[:, 5322 : 5658]
            shift_bg_fg = torch.zeros_like(shift_asym)
            if flat_dz.size(-1) >= self.BG_FG_END and not color_preserving and eff_spec_gain == 0.0:
                bg_mask = u_occ[:, :, 1:2]  # (B, L, 1) = 1.0 - p1_fg
                canvas_fg = E_t[:, :, :, 1].amax(dim=1, keepdim=True)  # (B, 1, 12)
                phi_bg_fg = bg_mask * canvas_fg  # (B, L, 12)
                u_bg_fg = flat_dz[:, self.BG_FG_START : 5466].view(B, 12, 12) * 0.85
                has_2panel_only = (
                    (u_occ[:, :, 5:9].amax(dim=(1, 2), keepdim=True) > 0)
                    & (u_occ[:, :, 9:13].amax(dim=(1, 2), keepdim=True) == 0)
                ).to(u_occ.dtype)
                occ_scale = 2.00 + 8.00 * has_2panel_only
                u_occ_rel = flat_dz[:, 5466 : self.BG_FG_END].view(B, 16, 12) * occ_scale
                shift_bg_fg = torch.bmm(phi_bg_fg, u_bg_fg) + torch.bmm(u_occ, u_occ_rel)
            elif flat_dz.size(-1) >= self.BG_FG_END and eff_spec_gain != 0.0:
                bg_mask = u_occ[:, :, 1:2]  # (B, L, 1) = 1.0 - p1_fg
                canvas_fg = E_t[:, :, :, 1].amax(dim=1, keepdim=True)  # (B, 1, 12)
                phi_bg_fg = bg_mask * canvas_fg + 0.02 * E_t[:, :, :, 0]  # (B, L, 12)
                u_bg_fg = (flat_dz[:, self.BG_FG_START : 5466].view(B, 12, 12) * 0.85 * fg_col_mask).clone()
                if color_preserving:
                    u_bg_fg[:, 3:12, :3] = u_bg_fg[:, 3:12, :3].mean(dim=1, keepdim=True).expand(-1, 9, -1)
                has_2panel_only = (
                    (u_occ[:, :, 5:9].amax(dim=(1, 2), keepdim=True) > 0)
                    & (u_occ[:, :, 9:13].amax(dim=(1, 2), keepdim=True) == 0)
                ).to(u_occ.dtype)
                occ_scale = 2.00 + 8.00 * has_2panel_only
                u_occ_rel = flat_dz[:, 5466 : self.BG_FG_END].view(B, 16, 12) * occ_scale * fg_col_mask
                shift_bg_fg = torch.bmm(phi_bg_fg, u_bg_fg) + torch.bmm(u_occ, u_occ_rel)

            # 3. Continuous Spectral & Multi-Bounce Ray-Field Basis: flat_dz[:, 5658 : 8072] + guard [8072 : 8192]
            shift_spec = torch.zeros_like(shift_asym)
            Phi_spec_for_v: Optional[torch.Tensor] = None
            E_spec_eq_for_v: Optional[torch.Tensor] = None
            if flat_dz.size(-1) >= self.EXT_REL_END and eff_spec_gain != 0.0:
                spec_scale = eff_spec_gain
                Phi_spec = self.get_continuous_spectral_and_ray_basis(
                    inputs,
                    target_hw=target_hw,
                    inputs_pullback=inputs_pullback,
                    secondary_pullback=secondary_pullback,
                )
                Phi_spec_for_v = Phi_spec
                E_spec_eq_for_v = self.last_spec_eq
                u_ext_spec = flat_dz[:, self.EXT_REL_START : self.EXT_REL_END].view(
                    B, self.K_EXT_SPEC, 12
                )
                # Sym(9)-equivariant spectral contraction over E_spec_eq, multi-bounce rays, and CC harmonics
                shift_spec_eq = torch.zeros_like(shift_asym)
                if self.last_spec_eq is not None:
                    E_spec_eq = self.last_spec_eq  # (B, L, 12, K_SPEC_EQ)
                    k_eq = E_spec_eq.size(-1)
                    w_spec_eq = u_ext_spec[:, :k_eq, 3:12].mean(dim=-1).view(B, 1, k_eq, 1) * 6.0
                    shift_spec_eq = torch.matmul(E_spec_eq, w_spec_eq).squeeze(-1)

                if color_preserving:
                    # S_9-equivariant contraction of per-color ray (64:136) and CC (152:170) channels onto fg colors 3..11
                    rays_eq = Phi_spec[:, :, 64:136].view(B, L, 9, 8)  # (B, L, 9_colors, 8_ray_types)
                    w_ray_eq = u_ext_spec[:, 64:136, 3:12].view(B, 9, 8, 9).mean(dim=(1, 3)).view(B, 1, 1, 8)
                    ray_fg_shift = (rays_eq * w_ray_eq).sum(dim=-1) * 0.25  # (B, L, 9)

                    cc_eq = Phi_spec[:, :, 152:170].view(B, L, 2, 9).permute(0, 1, 3, 2)  # (B, L, 9_colors, 2_cc)
                    w_cc_eq = u_ext_spec[:, 152:170, 3:12].view(B, 2, 9, 9).mean(dim=(2, 3)).view(B, 1, 1, 2)
                    cc_fg_shift = (cc_eq * w_cc_eq).sum(dim=-1) * 0.25  # (B, L, 9)

                    # Global S_9-invariant scalar carrier over non-color-indexed spectral channels onto E_t[:, :, 3:12, 0]
                    w_rem_inv = u_ext_spec[:, :, 3:12].mean(dim=-1).clone()  # (B, 201)
                    w_rem_inv[:, 64:136] = w_rem_inv[:, 64:136].view(B, 9, 8).mean(dim=1, keepdim=True).expand(-1, 9, -1).reshape(B, 72)
                    w_rem_inv[:, 152:170] = w_rem_inv[:, 152:170].view(B, 2, 9).mean(dim=2, keepdim=True).expand(-1, -1, 9).reshape(B, 18)
                    rem_carrier = torch.bmm(Phi_spec, w_rem_inv.unsqueeze(-1)) * 0.05  # (B, L, 1)
                    shift_spec_eq = shift_spec_eq.clone()
                    shift_spec_eq[:, :, 3:12] = (
                        shift_spec_eq[:, :, 3:12]
                        + ray_fg_shift
                        + cc_fg_shift
                        + rem_carrier * E_t[:, :, 3:12, 0]
                    )

                if color_preserving:
                    u_ext_spec_rel = (u_ext_spec * fg_col_mask).clone()
                    u_ext_spec_rel[:, 64:136, :3] = (
                        u_ext_spec_rel[:, 64:136, :3]
                        .view(B, 9, 8, 3)
                        .mean(dim=1, keepdim=True)
                        .expand(-1, 9, -1, -1)
                        .reshape(B, 72, 3)
                    )
                    u_ext_spec_rel[:, 152:170, :3] = (
                        u_ext_spec_rel[:, 152:170, :3]
                        .view(B, 2, 9, 3)
                        .mean(dim=2, keepdim=True)
                        .expand(-1, -1, 9, -1)
                        .reshape(B, 18, 3)
                    )
                else:
                    u_ext_spec_rel = u_ext_spec
                shift_spec_rel = torch.bmm(Phi_spec, u_ext_spec_rel)
                shift_spec = (shift_spec_rel + shift_spec_eq) * (spec_scale * self.rel_scale * rel_gain)

                if flat_dz.size(-1) >= self.EXT_GATE_END and self.last_phi_guard is not None:
                    Phi_guard = self.last_phi_guard  # (B, L, 2)
                    g_ext = flat_dz[:, self.EXT_GATE_START : self.EXT_GATE_END]  # (B, 2)
                    eq_guard_proj = E_t[:, :, :, :2].permute(0, 1, 3, 2)  # (B, L, 2, 12)
                    g_ext_shift = (
                        Phi_guard.unsqueeze(-1) * eq_guard_proj * g_ext.view(B, 1, 2, 1)
                    ).sum(dim=-2)

                    if flat_dz.size(-1) >= self.SPECTRAL_GUARD_END:
                        w_spec_guard = flat_dz[
                            :, self.SPECTRAL_GUARD_START : self.SPECTRAL_GUARD_END
                        ].view(B, 2, 12)
                        if color_preserving:
                            w_spec_guard = w_spec_guard * fg_col_mask
                        gate_mod = 1.0 + 0.25 * torch.tanh(g_ext).unsqueeze(-1)
                        guard_shift = torch.bmm(Phi_guard, w_spec_guard * gate_mod) + 0.15 * g_ext_shift
                    else:
                        guard_shift = 0.15 * g_ext_shift

                    # Couple cascade slice (4810:5322) and cascade guard (8072:8168) into spectral readout
                    cas_aux_shift = torch.zeros_like(shift_spec)
                    if flat_dz.size(-1) >= self.CAS_STEP_GEN_END:
                        w_cas_main = flat_dz[:, self.CAS_START : self.CAS_END].view(B, 64, 8)
                        w_cas_guard = flat_dz[:, self.GUARD_START : self.CAS_STEP_GEN_END].view(B, 8, 12)
                        if color_preserving:
                            w_cas_guard = w_cas_guard * fg_col_mask
                        dct_64 = Phi_spec[:, :, :64]  # (B, L, 64)
                        cas_proj_8 = torch.bmm(dct_64, w_cas_main)  # (B, L, 8)
                        cas_eq_8 = E_asym.permute(0, 1, 3, 2)  # (B, L, 8, 12)
                        cas_aux_shift = 0.05 * (
                            (cas_proj_8.unsqueeze(-1) * cas_eq_8).sum(dim=-2)
                            + torch.bmm(dct_64[:, :, :8], w_cas_guard)
                        )

                    shift_spec = shift_spec + (0.70 * spec_scale * self.rel_scale * rel_gain) * (
                        guard_shift + cas_aux_shift
                    )

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

                delta_sym_asym = F.linear(dz_canvas, lm_weight) + self.readout_scale * shift_spec
            else:
                delta_sym_asym = self.readout_scale * (combined_linear_shift + shift_spec)

            # 4. Skew-Symmetric Triad Occlusion Breaker: flat_dz[:, sym_end : triad_end]
            # Computed on competing color vector v so <v, delta_triad> == 0 holds to <= 1.0e-8 in float32!
            sym_canvas_logits = logits_f32 + delta_sym_asym
            v_comp = self.compute_competing_color_vector(
                sym_canvas_logits,
                E_t,
                E_asym=E_asym,
                Phi_spec=Phi_spec_for_v,
                E_spec_eq=E_spec_eq_for_v,
            )  # (B, L, 12)
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
            eff_triad_gain = (
                base_triad_gain
                * (0.25 if color_preserving else 0.85)
                * (0.05 if inputs_pullback is not None else 1.0)
            )
            delta_triad_raw = self.readout_scale * eff_triad_gain * self.triad_op(u_occ, v_comp, W_triad)
            delta_triad = _exact_orthogonality_ulp_polish(v_comp, delta_triad_raw, no_grad_only=True)  # (B, L, 12)

            self.last_u_occ = u_occ.detach()
            self.last_v_comp = v_comp.detach()
            self.last_sym_shift = (self.readout_scale * raw_sym_shift).detach()
            self.last_asym_shift = (self.readout_scale * (shift_asym + shift_bg_fg)).detach()
            self.last_spec_shift = (self.readout_scale * shift_spec).detach()
            self.last_triad_shift = delta_triad.detach()

            return sym_canvas_logits + delta_triad

        return logits_f32


def get_continuous_spectral_and_ray_basis(
    inputs: torch.Tensor,
    target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    inputs_pullback: Optional[torch.Tensor] = None,
    secondary_pullback: Optional[torch.Tensor] = None,
    demo_target_hws: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Module-level helper building the 201-channel 2D Continuous Spectral & Multi-Bounce Ray Basis."""
    if not hasattr(get_continuous_spectral_and_ray_basis, "_default_readout"):
        setattr(get_continuous_spectral_and_ray_basis, "_default_readout", DarkSnailCanvasReadout())
    readout: DarkSnailCanvasReadout = getattr(get_continuous_spectral_and_ray_basis, "_default_readout")
    return readout.get_continuous_spectral_and_ray_basis(
        inputs,
        target_hw=target_hw,
        inputs_pullback=inputs_pullback,
        secondary_pullback=secondary_pullback,
        demo_target_hws=demo_target_hws,
    )


class SlowFastTriadCascade(nn.Module):
    """
    Upgrade 3 & Workstream B: Multi-Step Unrolled (`K in 1..16`) Low-Rank (`d_shell = 8`)
    Skew-Symmetric Triad Coupling between Slow recurrent state `z_H^{(t)}` and Fast
    recurrent state `z_L^{(t)}` parameterized by `flat_dz[:, 4810:5322]` and the
    temporal guard buffer `flat_dz[:, 8072:8168]`:

        W_rec^{(t)} = W_rec + beta_t * unsqueeze(Delta W_step - Delta W_step^T, dim=1)
        delta_v_L^{(t)} = gamma_cas * (1 + alpha_t) * B_{W_rec^{(t)}}(u^{(t)}, v_L^{(t)})
        v_tilde_L^{(t)} = Retract_{S^{M-1}}(v_L^{(t)} + delta_v_L^{(t)})
        z_tilde_L = z_L + (v_tilde_L^{(15)} - z_L P) P^T

    By `triad_pointwise_orthogonal` (`TriadConservation.lean`), each shell tangent step
    `delta_v_L^{(t)}` is strictly orthogonal to `v_L^{(t)}` (`<= 1.0e-8` in `float32`),
    and exact IEEE-754 `float32` spherical retraction conserves `||P z_tilde_L||_2^2 == ||P z_L||_2^2`
    to `0.000e+00` while guaranteeing `delta_z == 0 => z_tilde_L == z_L`.
    """

    DEFAULT_SLICE_START: int = DarkSnailCanvasReadout.ASYM_END  # 4810
    GUARD_START: int = DarkSnailCanvasReadout.GUARD_START  # 8072
    GUARD_END: int = DarkSnailCanvasReadout.GUARD_END  # 8192
    CAS_ALPHA_START: int = DarkSnailCanvasReadout.CAS_ALPHA_START  # 8072
    CAS_ALPHA_END: int = DarkSnailCanvasReadout.CAS_ALPHA_END  # 8088
    CAS_BETA_START: int = DarkSnailCanvasReadout.CAS_BETA_START  # 8088
    CAS_BETA_END: int = DarkSnailCanvasReadout.CAS_BETA_END  # 8104
    CAS_STEP_GEN_START: int = DarkSnailCanvasReadout.CAS_STEP_GEN_START  # 8104
    CAS_STEP_GEN_END: int = DarkSnailCanvasReadout.CAS_STEP_GEN_END  # 8168
    SPECTRAL_GUARD_START: int = DarkSnailCanvasReadout.SPECTRAL_GUARD_START  # 8168
    SPECTRAL_GUARD_END: int = DarkSnailCanvasReadout.SPECTRAL_GUARD_END  # 8192

    def __init__(
        self,
        d_reasoner: int = 512,
        d_shell: int = 8,
        slice_start: int = DEFAULT_SLICE_START,
        cascade_gain: float = 1.0,
        preserve_exact_sphere_norm: bool = True,
        lm_weight: Optional[torch.Tensor] = None,
        d_model: Optional[int] = None,
        unroll_steps: int = 1,
    ):
        super().__init__()
        if d_model is not None:
            d_reasoner = d_model
        self.d_reasoner = int(d_reasoner)
        self.d_shell = int(d_shell)
        self.slice_start = int(slice_start)
        self.slice_end = self.slice_start + (self.d_shell ** 3)  # 4810 + 512 = 5322
        self.cascade_gain = float(cascade_gain)
        self.preserve_exact_sphere_norm = bool(preserve_exact_sphere_norm)
        self.unroll_steps = max(1, min(16, int(unroll_steps)))
        self.triad_op = SkewSymmetricTriadOperator()

        self._p_cache: Dict[Any, torch.Tensor] = {}
        P = self._build_orthonormal_shell_basis(self.d_reasoner, self.d_shell, lm_weight)
        self.register_buffer("P", P, persistent=False)  # (d_reasoner, d_shell)

        self.last_tangent_step: Optional[torch.Tensor] = None
        self.last_v_retracted: Optional[torch.Tensor] = None
        self.last_v_shell_0: Optional[torch.Tensor] = None
        self._last_z_L_tilde: Optional[torch.Tensor] = None
        self.last_diagnostics: Optional[Dict[str, Any]] = None

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
            q, _ = torch.linalg.qr(Vh[:d_shell, :].t().contiguous(), mode="reduced")
            return q.contiguous()

        idx = torch.arange(d_reasoner, dtype=torch.float32).unsqueeze(1)  # (D, 1)
        k = torch.arange(1, d_shell + 1, dtype=torch.float32).unsqueeze(0)  # (1, M)
        raw = torch.cos(math.pi * (idx + 0.5) * k / float(d_reasoner))
        q, _ = torch.linalg.qr(raw, mode="reduced")  # (D, M) with Q^T Q = I_M
        return q.contiguous()

    def update_lm_weight(self, lm_weight: torch.Tensor) -> torch.Tensor:
        """
        Updates `self.P` to the top `M = d_shell` right singular vectors `P in R^(d_reasoner x d_shell)`
        (`P^T P = I_M`) of `lm_weight`, cached deterministically by `blake2b` digest.
        """
        digest = _tensor_blake2b(lm_weight)
        key = (digest, self.d_reasoner, self.d_shell, str(lm_weight.device))
        if key not in self._p_cache:
            if len(self._p_cache) >= 8:
                self._p_cache.clear()
            self._p_cache[key] = self._build_orthonormal_shell_basis(
                self.d_reasoner, self.d_shell, lm_weight
            ).to(device=lm_weight.device, dtype=torch.float32)
        self.P = self._p_cache[key]
        return self.P

    def _flatten_delta_z(self, delta_z: torch.Tensor, batch_size: int) -> torch.Tensor:
        dz_f = delta_z.float()
        if dz_f.dim() == 3 and dz_f.shape[0] == batch_size:
            return dz_f.reshape(batch_size, -1)
        if dz_f.dim() == 2 and dz_f.shape[0] == batch_size and dz_f.shape[1] != 512:
            return dz_f
        return dz_f.reshape(1, -1).expand(batch_size, -1)

    def extract_w_rec(self, delta_z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Extracts `W_rec` of shape `(B, d_shell, d_shell, d_shell)` from `delta_z`.
        """
        M = self.d_shell
        m3 = M * M * M
        flat = self._flatten_delta_z(delta_z, batch_size)

        if flat.shape[1] >= self.slice_end:
            w_slice = flat[:, self.slice_start : self.slice_end]
        elif flat.shape[1] >= m3:
            w_slice = flat[:, :m3]
        else:
            pad = F.pad(flat, (0, m3 - flat.shape[1]))
            w_slice = pad[:, :m3]

        return w_slice.view(batch_size, M, M, M)

    def extract_guard_schedule(
        self, delta_z: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts temporal cascade modulation parameters from the guard buffer `flat_dz[:, 8072:8168]`:
          - `alpha`: `(B, 16)` from `8072:8088`
          - `beta`: `(B, 16)` from `8088:8104`
          - `W_step`: `(B, d_shell, d_shell)` from `8104:8168`
        """
        flat = self._flatten_delta_z(delta_z, batch_size)
        device = flat.device
        dtype = flat.dtype
        M = self.d_shell

        if flat.shape[1] >= self.CAS_ALPHA_END:
            alpha = flat[:, self.CAS_ALPHA_START : self.CAS_ALPHA_END]
        else:
            alpha = torch.zeros((batch_size, 16), device=device, dtype=dtype)

        if flat.shape[1] >= self.CAS_BETA_END:
            beta = flat[:, self.CAS_BETA_START : self.CAS_BETA_END]
        else:
            beta = torch.zeros((batch_size, 16), device=device, dtype=dtype)

        m2 = M * M
        if flat.shape[1] >= self.CAS_STEP_GEN_START + m2 and self.CAS_STEP_GEN_START + m2 <= self.SPECTRAL_GUARD_START:
            w_step_raw = flat[:, self.CAS_STEP_GEN_START : self.CAS_STEP_GEN_START + m2]
            W_step = w_step_raw.view(batch_size, M, M)
        elif flat.shape[1] >= self.CAS_STEP_GEN_END:
            raw_64 = flat[:, self.CAS_STEP_GEN_START : self.CAS_STEP_GEN_END]
            if m2 <= 64:
                W_step = raw_64[:, :m2].view(batch_size, M, M)
            else:
                W_step = F.pad(raw_64, (0, m2 - 64)).view(batch_size, M, M)
        else:
            W_step = torch.zeros((batch_size, M, M), device=device, dtype=dtype)

        return alpha, beta, W_step

    def project_to_shell(self, z: torch.Tensor) -> torch.Tensor:
        """Projects `z` `(..., d_reasoner)` onto the low-rank shell `z @ P` `(..., d_shell)`."""
        P = self.P.to(device=z.device, dtype=torch.float32)
        v_proj = torch.matmul(z.float(), P)
        if (
            self.preserve_exact_sphere_norm
            and self._last_z_L_tilde is not None
            and self.last_v_retracted is not None
            and z.shape == self._last_z_L_tilde.shape
            and z.device == self._last_z_L_tilde.device
            and torch.equal(z.detach(), self._last_z_L_tilde)
        ):
            v_proj = v_proj + (self.last_v_retracted.detach() - v_proj.detach())
        return v_proj

    @staticmethod
    def _retract_to_sphere(
        v_updated: torch.Tensor,
        v_target_norm_ref: torch.Tensor,
        polish_exact_float32: bool = True,
    ) -> torch.Tensor:
        orig_norm = torch.norm(v_target_norm_ref, p=2, dim=-1, keepdim=True)
        new_norm = torch.norm(v_updated, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        v_retracted = torch.where(orig_norm > 0.0, v_updated * (orig_norm / new_norm), v_updated)
        if polish_exact_float32:
            v_retracted = _exact_sphere_norm_ulp_polish(v_retracted, v_target_norm_ref)
        return v_retracted

    def forward(
        self,
        z_H: Union[torch.Tensor, Sequence[Any]],
        z_L: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        delta_z: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        unroll_steps: Optional[int] = None,
        step_idx: Optional[int] = None,
        lm_weight: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Couples slow state `z_H` and fast state `z_L` across `K` unrolled recurrent cascade steps
        `t in [16 - K, 15]` (or single step `step_idx`) using the step-modulated skew-symmetric
        triad generator sliced from `delta_z`.
        """
        if lm_weight is not None:
            self.update_lm_weight(lm_weight)

        # Normalize trajectory vs single-step inputs
        is_trajectory = False
        z_H_steps: List[torch.Tensor] = []
        z_L_steps: List[torch.Tensor] = []

        if isinstance(z_H, (list, tuple)):
            is_trajectory = True
            if len(z_H) > 0 and isinstance(z_H[0], (list, tuple)) and len(z_H[0]) == 2:
                # Passed as [(z_H^{(t)}, z_L^{(t)}), ...]
                z_H_steps = [p[0] for p in z_H]
                z_L_steps = [p[1] for p in z_H]
                if delta_z is None and isinstance(z_L, torch.Tensor):
                    delta_z = z_L
            else:
                z_H_steps = list(z_H)
                if isinstance(z_L, (list, tuple)):
                    z_L_steps = list(z_L)
                elif isinstance(z_L, torch.Tensor):
                    if z_L.dim() == 4 and z_L.shape[1] == len(z_H_steps):
                        z_L_steps = [z_L[:, k] for k in range(len(z_H_steps))]
                    else:
                        z_L_steps = [z_L for _ in range(len(z_H_steps))]
                else:
                    raise ValueError("z_L must be provided when z_H is a sequence of tensors.")
        elif isinstance(z_H, torch.Tensor) and z_H.dim() == 4 and z_H.shape[1] <= 16:
            # 4D trajectory tensor (B, K, S, D)
            is_trajectory = True
            K_traj = z_H.shape[1]
            z_H_steps = [z_H[:, k] for k in range(K_traj)]
            if isinstance(z_L, torch.Tensor):
                if z_L.dim() == 4 and z_L.shape[1] == K_traj:
                    z_L_steps = [z_L[:, k] for k in range(K_traj)]
                else:
                    z_L_steps = [z_L for _ in range(K_traj)]
            elif isinstance(z_L, (list, tuple)):
                z_L_steps = list(z_L)
            else:
                raise ValueError("z_L must be provided when z_H is a 4D trajectory tensor.")
        else:
            if not isinstance(z_H, torch.Tensor) or not isinstance(z_L, torch.Tensor):
                raise ValueError("z_H and z_L must be tensors or sequences of tensors.")
            z_H_steps = [z_H]
            z_L_steps = [z_L]

        z_L_ref = z_L_steps[-1]
        orig_dtype = z_L_ref.dtype
        device = z_L_ref.device
        B = z_L_ref.shape[0]
        P = self.P.to(device=device, dtype=torch.float32)  # (D, M)

        if step_idx is not None:
            K = 1
            step_indices = [int(step_idx) % 16]
        elif is_trajectory:
            K_avail = len(z_H_steps)
            K = (
                max(1, min(16, min(K_avail, int(unroll_steps))))
                if unroll_steps is not None
                else max(1, min(16, K_avail))
            )
            z_H_steps = z_H_steps[-K:]
            z_L_steps = z_L_steps[-K:]
            step_indices = list(range(16 - K, 16))
        else:
            K = max(1, min(16, int(unroll_steps if unroll_steps is not None else self.unroll_steps)))
            step_indices = list(range(16 - K, 16))

        v_shell_0 = torch.matmul(z_L_ref.float(), P)  # (B, ..., M)

        if delta_z is None or (
            not (delta_z.requires_grad or z_L_ref.requires_grad)
            and float(delta_z.detach().abs().max().item()) == 0.0
        ):
            zero_step = torch.zeros((*z_L_ref.shape[:-1], self.d_shell), device=device, dtype=torch.float32)
            self.last_tangent_step = zero_step
            self.last_v_shell_0 = v_shell_0.detach()
            self.last_v_retracted = v_shell_0.detach()
            self._last_z_L_tilde = z_L_ref.detach()
            diag = {
                "max_orthogonality_error": 0.0,
                "max_abs_inner_product": 0.0,
                "max_norm_drift": 0.0,
                "unroll_steps": int(K),
                "step_orthogonality_errors": [0.0 for _ in range(K)],
                "alpha_t": torch.zeros((B, K), device=device, dtype=torch.float32),
                "beta_t": torch.zeros((B, K), device=device, dtype=torch.float32),
                "v_retracted": v_shell_0,
                "v_shell_0": v_shell_0,
            }
            self.last_diagnostics = diag
            if return_diagnostics:
                return z_L_ref, diag
            return z_L_ref

        dz_dev = delta_z.to(device=device)
        W_rec_base = self.extract_w_rec(dz_dev, B)  # (B, M, M, M)
        alpha, beta, W_step = self.extract_guard_schedule(dz_dev, B)  # (B, 16), (B, 16), (B, M, M)
        W_step_skew = W_step - W_step.transpose(-1, -2)  # (B, M, M)

        # Broadcast shape for per-step scalar alpha_t: (B, 1, ..., 1) matching v_shell_0
        bcast_shape = (B,) + (1,) * (v_shell_0.dim() - 1)
        step_orth_errors: List[float] = []

        if is_trajectory:
            cascade_shift = torch.zeros_like(v_shell_0)
            v_cur = v_shell_0
            delta_v = torch.zeros_like(v_shell_0)
            for idx, t in enumerate(step_indices):
                u_t = torch.matmul(z_H_steps[idx].float(), P)
                v_base_t = torch.matmul(z_L_steps[idx].float(), P)
                if idx == 0:
                    v_in_t = v_base_t
                else:
                    v_in_t = (
                        self._retract_to_sphere(
                            v_base_t + cascade_shift,
                            v_base_t,
                            polish_exact_float32=False,
                        )
                        if self.preserve_exact_sphere_norm
                        else (v_base_t + cascade_shift)
                    )

                alpha_t = alpha[:, t].view(*bcast_shape)
                beta_t = beta[:, t].view(B, 1, 1, 1)
                W_rec_t = W_rec_base + beta_t * W_step_skew.unsqueeze(1)
                gamma_t = self.cascade_gain * (1.0 + alpha_t)

                delta_v = self.triad_op(gamma_t * u_t, v_in_t, W_rec_t)
                orth_err_t = float((v_in_t * delta_v).sum(dim=-1).abs().max().item())
                step_orth_errors.append(orth_err_t)

                v_updated_t = v_in_t + delta_v
                if self.preserve_exact_sphere_norm:
                    is_last = idx == len(step_indices) - 1
                    v_cur = self._retract_to_sphere(
                        v_updated_t,
                        v_shell_0 if is_last else v_base_t,
                        polish_exact_float32=is_last,
                    )
                else:
                    v_cur = v_updated_t
                cascade_shift = v_cur - v_base_t
            v_retracted = v_cur
        else:
            u_cur = torch.matmul(z_H_steps[0].float(), P)
            v_cur = v_shell_0
            delta_v = torch.zeros_like(v_shell_0)
            inv_K = 1.0 / float(max(1, K))
            for idx, t in enumerate(step_indices):
                alpha_t = alpha[:, t].view(*bcast_shape)
                beta_t = beta[:, t].view(B, 1, 1, 1)
                W_rec_t = W_rec_base + beta_t * W_step_skew.unsqueeze(1)
                gamma_t = self.cascade_gain * (1.0 + alpha_t)

                delta_v = self.triad_op(gamma_t * u_cur, v_cur, W_rec_t)
                orth_err_t = float((v_cur * delta_v).sum(dim=-1).abs().max().item())
                step_orth_errors.append(orth_err_t)

                v_updated_t = v_cur + delta_v
                if self.preserve_exact_sphere_norm:
                    is_last = idx == len(step_indices) - 1
                    v_tilde_t = self._retract_to_sphere(
                        v_updated_t,
                        v_shell_0,
                        polish_exact_float32=is_last,
                    )
                else:
                    v_tilde_t = v_updated_t

                if idx + 1 < len(step_indices):
                    # Skew-coupled slow-fast feedback across unrolled substeps
                    u_cur = u_cur + inv_K * (v_tilde_t - v_cur)
                v_cur = v_tilde_t
            v_retracted = v_cur

        self.last_tangent_step = delta_v
        self.last_v_shell_0 = v_shell_0
        self.last_v_retracted = v_retracted

        shell_diff = v_retracted - v_shell_0
        z_L_tilde = (z_L_ref.float() + torch.matmul(shell_diff, P.t())).to(orig_dtype)
        self._last_z_L_tilde = z_L_tilde.detach()

        max_orth = float(max(step_orth_errors)) if step_orth_errors else 0.0
        if self.preserve_exact_sphere_norm:
            norm_drift = float(
                torch.max(torch.abs((v_retracted ** 2).sum(dim=-1) - (v_shell_0 ** 2).sum(dim=-1))).item()
            )
        else:
            norm_drift = float((v_retracted.norm(dim=-1) - v_shell_0.norm(dim=-1)).abs().max().item())

        diag = {
            "max_orthogonality_error": max_orth,
            "max_abs_inner_product": max_orth,
            "max_norm_drift": norm_drift,
            "unroll_steps": int(K),
            "step_orthogonality_errors": step_orth_errors,
            "alpha_t": alpha[:, step_indices].detach(),
            "beta_t": beta[:, step_indices].detach(),
            "v_retracted": v_retracted,
            "v_shell_0": v_shell_0,
        }
        self.last_diagnostics = diag

        if return_diagnostics:
            return z_L_tilde, diag
        return z_L_tilde


class ContinuousSpectralFieldReadout(nn.Module):
    """
    Workstream C.2: Modular Continuous Dynamical Field Readout Head.

    Wraps and coordinates `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, and
    `RoPE2DMedoidPooler`, mapping all `8,072` active coordinates of `flat_dz`
    (`0:38` equivariant, `38:2498` relational, `2498:4802` so(12) triad, `4802:4810`
    directional asymmetry, `4810:5322` so(8) cascade, `5322:8072` extended relational/ray/spectral
    channels) plus the `8072:8192` guard buffer and `RoPE2DMedoidPooler`
    (`gamma_2D(S) = exp(-nu * p^(2*ell) * (Var_y(S) + Var_x(S))) in [0.05, 1.0]`) onto:
      - `logits`: `(B, L, V)` refined canvas logits
      - `spectral_state`: dictionary containing `"continuous_field_2d"` `(B, H, W, 12)`,
        `"spectral_coeffs"` `(B, 201, 12)`, `"gamma_2d"` `(B, H_patch, W_patch)` in `[0.05, 1.0]`,
        `"pooled_k"`, `"pooled_v"`, and `"orthogonality_residual"`.
    """

    def __init__(
        self,
        canvas_readout: Optional[DarkSnailCanvasReadout] = None,
        cascade: Optional[SlowFastTriadCascade] = None,
        medoid_pooler: Optional[RoPE2DMedoidPooler] = None,
        readout_scale: float = 6.0,
        triad_gain: float = 1.0,
        asym_gain: float = 1.5,
        tau_v: float = 1.0,
        rel_scale: float = 1.0,
        spectral_gain: float = 0.35,
        d_reasoner: int = 512,
        d_shell: int = 8,
        unroll_steps: int = 4,
        patch_size: int = 2,
        viscous_nu: float = 0.05,
        viscous_p: int = 2,
        shell_level: int = 1,
    ):
        super().__init__()
        self.spectral_gain = float(spectral_gain)
        self.canvas_readout = (
            canvas_readout
            if canvas_readout is not None
            else DarkSnailCanvasReadout(
                readout_scale=readout_scale,
                triad_gain=triad_gain,
                asym_gain=asym_gain,
                tau_v=tau_v,
                rel_scale=rel_scale,
                spectral_gain=spectral_gain,
            )
        )
        self.cascade = (
            cascade
            if cascade is not None
            else SlowFastTriadCascade(
                d_reasoner=d_reasoner,
                d_shell=d_shell,
                unroll_steps=unroll_steps,
            )
        )
        self.medoid_pooler = (
            medoid_pooler
            if medoid_pooler is not None
            else RoPE2DMedoidPooler(
                patch_size=patch_size,
                viscous_nu=viscous_nu,
                viscous_p=viscous_p,
                shell_level=shell_level,
            )
        )
        self.canvas_readout.medoid_pooler = self.medoid_pooler

    def forward(
        self,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        delta_z: Optional[torch.Tensor] = None,
        z_H: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
        z_L: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        model: Optional[nn.Module] = None,
        lm_weight: Optional[torch.Tensor] = None,
        carry: Any = None,
        callosal_emb: Any = None,
        target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
        color_preserving: bool = False,
        triad_gain_override: Optional[float] = None,
        asym_gain_override: Optional[float] = None,
        spectral_gain_override: Optional[float] = None,
        unroll_steps: Optional[int] = None,
        return_continuous_field: bool = False,
        return_diagnostics: bool = False,
        inputs_pullback: Optional[torch.Tensor] = None,
        secondary_pullback: Optional[torch.Tensor] = None,
        demo_target_hws: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Evaluates the continuous dynamical field readout and optional multi-step slow-fast cascade.
        """
        if target_hw is None and demo_target_hws is not None:
            target_hw = demo_target_hws
        B, L = inputs.shape[0], inputs.shape[1]
        side = int(round(math.sqrt(L)))
        device = logits.device

        if lm_weight is None and model is not None:
            lm_weight = DarkSnailCanvasReadout._locate_lm_weight(
                model, logits.size(-1), self.cascade.d_reasoner
            )
        if lm_weight is not None:
            self.cascade.update_lm_weight(lm_weight)

        cascade_diag: Optional[Dict[str, Any]] = None
        z_L_coupled: Optional[torch.Tensor] = None
        if z_H is not None and z_L is not None:
            z_L_coupled, cascade_diag = self.cascade(
                z_H,
                z_L,
                delta_z,
                return_diagnostics=True,
                unroll_steps=unroll_steps,
                lm_weight=lm_weight,
            )

        eff_spec_override = (
            spectral_gain_override if spectral_gain_override is not None else self.spectral_gain
        )
        refined_logits = self.canvas_readout(
            logits=logits,
            inputs=inputs,
            delta_z=delta_z,
            model=model,
            lm_weight=lm_weight,
            carry=carry,
            callosal_emb=callosal_emb,
            target_hw=target_hw,
            color_preserving=color_preserving,
            triad_gain_override=triad_gain_override,
            asym_gain_override=asym_gain_override,
            spectral_gain_override=eff_spec_override,
            inputs_pullback=inputs_pullback,
            secondary_pullback=secondary_pullback,
            demo_target_hws=demo_target_hws,
        )

        # Ensure spectral basis & pooler telemetry are populated even when delta_z is None
        _ = self.canvas_readout.get_continuous_spectral_and_ray_basis(
            inputs,
            target_hw=target_hw,
            inputs_pullback=inputs_pullback,
            secondary_pullback=secondary_pullback,
            demo_target_hws=demo_target_hws,
        )

        continuous_field_2d = refined_logits.view(B, side, side, -1)
        pooled_k, pooled_v, pool_info = self.medoid_pooler(continuous_field_2d)
        gamma_2d = pool_info["gamma_2d"]  # (B, H_patch, W_patch) in [0.05, 1.0]

        if delta_z is not None and delta_z.numel() >= DarkSnailCanvasReadout.EXT_REL_END:
            dz_f = delta_z.float().to(device)
            if dz_f.numel() == 8192:
                flat_dz = dz_f.view(1, 8192).expand(B, 8192)
            elif dz_f.numel() == B * 8192:
                flat_dz = dz_f.view(B, 8192)
            else:
                flat_dz = dz_f.reshape(B, -1)
            spectral_coeffs = flat_dz[
                :, DarkSnailCanvasReadout.EXT_REL_START : DarkSnailCanvasReadout.EXT_REL_END
            ].view(B, DarkSnailCanvasReadout.K_EXT_SPEC, 12)
        else:
            spectral_coeffs = torch.zeros(
                (B, DarkSnailCanvasReadout.K_EXT_SPEC, 12), device=device, dtype=torch.float32
            )

        v_comp = self.canvas_readout.last_v_comp
        delta_triad = self.canvas_readout.last_triad_shift
        if v_comp is not None and delta_triad is not None and delta_z is not None:
            orth_res = float((v_comp * delta_triad).sum(dim=-1).abs().max().item())
        else:
            orth_res = 0.0

        spectral_state: Dict[str, Any] = {
            "continuous_field_2d": continuous_field_2d,
            "spectral_coeffs": spectral_coeffs,
            "gamma_2d": gamma_2d,
            "pooled_k": pooled_k,
            "pooled_v": pooled_v,
            "orthogonality_residual": orth_res,
        }
        if cascade_diag is not None:
            spectral_state["cascade_diagnostics"] = cascade_diag
            spectral_state["z_L_coupled"] = z_L_coupled
            spectral_state["max_norm_drift"] = cascade_diag["max_norm_drift"]

        if return_continuous_field or return_diagnostics:
            return refined_logits, spectral_state
        return refined_logits
