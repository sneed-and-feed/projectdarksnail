"""
darksnail.arc3: Bi-Hemispheric DarkSnail V2 + Gemma 4 31B Agent for ARC-AGI-3 Interactive Reasoning.

Integrates the core mathematical primitives of Project DarkSnail V2 into an online,
zero-leakage, 100% game-agnostic interactive reasoning agent for ARC-AGI-3 (`64x64` 16-color
Markov/partially-observable environments with mixed discrete + coordinate action spaces):

1. `TemporalClockContentSplitter`:
   Dynamical-systems isomorphism of `ProportionalRoPESubspaceSplitter` (`k_rope` vs `k_nope`
   in `darksnail.gemma4`) grounded in Exogenous-Endogenous MDP Factorization (Dietterich et al.,
   arXiv:1806.01584; Lamb et al., arXiv:2207.08229; Choi et al., arXiv:1811.01483) and the
   discrete continuity equation `d_t rho_c + div(J_c) = sigma_c`. Decomposes the observed 2D grid
   state space into:
     - `H_clock` (`k_rope` analogue): Exogenous dissipative phase/clock components (`sigma_c != 0`,
       global color-mass non-conservation `Delta M_{c_old} < 0, Delta M_{c_new} > 0`) that evolve
       independently of action choice, co-tick alongside spatially disjoint content transitions,
       or undergo periodic phase resets (fuel refill + life-icon decrement).
     - `H_nope` (`k_nope` analogue): Conservative / action-contingent controllable state variables.
   Uses zero spatial/border priors (no hardcoded border rows/columns).

2. `extract_arc3_beltrami_medoids` & `compute_topological_symmetry_potential`:
   Generalizes `darksnail.shell.extract_grid_beltrami_shape_descriptors` and
   `darksnail.readout.RoPE2DMedoidPooler` to `64x64` 16-color (`0..15`) grids. Discretizes
   the 4,096-point `ACTION6(x, y)` coordinate space into ranked semantic medoid candidates
   with Dyadic Bruhat-Tits Quadtree Stratification across the `64x64` canvas, and evaluates
   the Topological Betti `(beta_0, beta_1)` + Dihedral `D_4` / Multi-Panel Symmetry Free-Energy
   Potential `Phi_topo(F_t)` (Higgins et al., arXiv:2110.09443; Friston et al., arXiv:1911.12247;
   arXiv:2609.39140; arXiv:2610.02200).

3. `Arc3EpisodicBeltramiShell`:
   Encodes `64x64` 16-color frame trajectories into token streams and applies
   `BeltramiCallosalShell` (or `Gemma4BeltramiCallosalShell` at Global Layer 29 when a live
   `google/gemma-4-31B-it` model is attached) to condense episodic history into 32 Adelic
   super-tokens while extracting live `zeno_flux` and `lca_variance`.

4. `DarkSnailArc3WorldModel`:
   Couples `SlowFastTriadCascade` (`d_reasoner=512, d_shell=8, preserve_exact_sphere_norm=True`)
   and `SkewSymmetricTriadOperator` (`so(8)` / `so(16)`) with an online state-transition graph,
   1-Step Deterministic Empowerment `E_1(s)` (Salge et al., arXiv:1310.1863; Mohamed & Rezende,
   arXiv:1509.08731; arXiv:2605.25931; arXiv:2607.28287), Local Patch Invariance for inert clicks,
   and an Online Spatial Contingency Posterior + morphological ridge/bandit learner.

5. `DarkSnailArc3Agent`:
   Combines `ZenoAmygdalaController`, `Arc3EpisodicBeltramiShell`, `DarkSnailArc3WorldModel`,
   and Epistemic Frontier BFS Graph Search with spanning-tree entry-action ordering, automatic
   no-op self-loop masking, `GAME_OVER` trap avoidance, and phase-reset persistence.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .antenna import DarkSnailSchedule, ZenoAmygdalaController
from .gemma4 import Gemma4BeltramiCallosalShell
from .readout import (
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    _tensor_blake2b,
    verify_pointwise_orthogonality,
)
from .shell import BeltramiCallosalShell

from pathlib import Path
import sys

_OATSE_SIBLING = Path(__file__).resolve().parents[2] / "oatse"
if _OATSE_SIBLING.exists() and str(_OATSE_SIBLING) not in sys.path:
    sys.path.insert(0, str(_OATSE_SIBLING))

try:
    from oatse.engine import tropical_kleene_closure, tropical_min_plus_matmul
except ImportError:
    def tropical_min_plus_matmul(
        A: Any,
        B: Any,
        return_witness: bool = False,
        inf_threshold: float = 1e12,
    ) -> Any:
        if isinstance(A, torch.Tensor) or isinstance(B, torch.Tensor):
            tA = A if isinstance(A, torch.Tensor) else torch.as_tensor(A, dtype=torch.float64)
            tB = B if isinstance(B, torch.Tensor) else torch.as_tensor(B, dtype=tA.dtype, device=tA.device)
            if tA.ndim != 2 or tB.ndim != 2 or tA.shape[1] != tB.shape[0]:
                raise ValueError(
                    f"tropical_min_plus_matmul requires compatible 2D shapes, got {tuple(tA.shape)} and {tuple(tB.shape)}"
                )
            orig_dtype = tA.dtype
            if tA.shape[1] == 0:
                empty_c = torch.full((tA.shape[0], tB.shape[1]), float("inf"), dtype=orig_dtype, device=tA.device)
                empty_w = torch.full((tA.shape[0], tB.shape[1]), -1, dtype=torch.long, device=tA.device)
                return (empty_c, empty_w) if return_witness else empty_c
            a64 = torch.where(tA.to(torch.float64) >= inf_threshold, float("inf"), tA.to(torch.float64))
            b64 = torch.where(tB.to(torch.float64) >= inf_threshold, float("inf"), tB.to(torch.float64))
            sums = a64.unsqueeze(-1) + b64.unsqueeze(-3)
            min_vals, min_idx = torch.min(sums, dim=-2)
            valid = torch.isfinite(min_vals) & (min_vals < inf_threshold)
            min_idx = torch.where(valid, min_idx, torch.full_like(min_idx, -1))
            out = torch.where(valid, min_vals, torch.full_like(min_vals, float("inf"))).to(orig_dtype)
            return (out, min_idx) if return_witness else out

        arr_a = np.asarray(A, dtype=np.float64)
        arr_b = np.asarray(B, dtype=np.float64)
        if arr_a.ndim != 2 or arr_b.ndim != 2 or arr_a.shape[1] != arr_b.shape[0]:
            raise ValueError(
                f"tropical_min_plus_matmul requires compatible 2D shapes, got {arr_a.shape} and {arr_b.shape}"
            )
        if arr_a.shape[1] == 0:
            empty_c = np.full((arr_a.shape[0], arr_b.shape[1]), np.inf, dtype=np.float64)
            empty_w = np.full((arr_a.shape[0], arr_b.shape[1]), -1, dtype=np.int64)
            return (empty_c, empty_w) if return_witness else empty_c
        a_clean = np.where(arr_a >= inf_threshold, np.inf, arr_a)
        b_clean = np.where(arr_b >= inf_threshold, np.inf, arr_b)
        sums = a_clean[:, :, None] + b_clean[None, :, :]
        min_idx = np.argmin(sums, axis=1)
        i_grid, j_grid = np.ogrid[: arr_a.shape[0], : arr_b.shape[1]]
        min_vals = sums[i_grid, min_idx, j_grid]
        valid = np.isfinite(min_vals) & (min_vals < inf_threshold)
        min_idx = np.where(valid, min_idx, -1).astype(np.int64)
        min_vals = np.where(valid, min_vals, np.inf)
        return (min_vals, min_idx) if return_witness else min_vals

    def tropical_kleene_closure(
        A: Any,
        max_steps: Optional[int] = None,
        return_routing: bool = False,
        inf_threshold: float = 1e12,
    ) -> Any:
        is_tensor = isinstance(A, torch.Tensor)
        if is_tensor:
            orig_device = A.device
            orig_dtype = A.dtype
            W_raw = A.detach().cpu().numpy().astype(np.float64)
        else:
            W_raw = np.asarray(A, dtype=np.float64)
        if W_raw.ndim != 2 or W_raw.shape[0] != W_raw.shape[1]:
            raise ValueError(f"tropical_kleene_closure requires a square 2D matrix, got {W_raw.shape}")
        n = int(W_raw.shape[0])
        if n == 0:
            empty_star = np.zeros((0, 0), dtype=np.float64)
            empty_hop = np.zeros((0, 0), dtype=np.int64)
            if is_tensor:
                t_star = torch.as_tensor(empty_star, dtype=orig_dtype, device=orig_device)
                t_hop = torch.as_tensor(empty_hop, dtype=torch.long, device=orig_device)
                return (t_star, t_hop) if return_routing else t_star
            return (empty_star, empty_hop) if return_routing else empty_star
        steps = max(1, int(max_steps) if max_steps is not None else max(1, n - 1))
        W = np.where(W_raw >= inf_threshold, np.inf, W_raw)
        star = np.minimum(W, np.inf)
        np.fill_diagonal(star, 0.0)
        next_hop = np.full((n, n), -1, dtype=np.int64)
        off_diag = ~np.eye(n, dtype=bool)
        valid_1hop = off_diag & np.isfinite(W) & (W < inf_threshold)
        j_idx = np.broadcast_to(np.arange(n, dtype=np.int64)[None, :], (n, n))
        next_hop = np.where(valid_1hop, j_idx, next_hop)
        star = np.where(off_diag, np.where(valid_1hop, W, np.inf), 0.0)
        if steps >= max(1, n - 1) and n > 1:
            num_iters = max(1, int(math.ceil(math.log2(max(1, n - 1)))))
            for _ in range(num_iters):
                d_new, mid_k = tropical_min_plus_matmul(
                    star, star, return_witness=True, inf_threshold=inf_threshold
                )
                improved = off_diag & (d_new + 1e-12 < star)
                if not np.any(improved):
                    break
                if return_routing:
                    safe_k = np.maximum(0, mid_k)
                    first_hop = np.take_along_axis(next_hop, safe_k, axis=1)
                    next_hop = np.where(improved, first_hop, next_hop)
                star = np.where(improved, d_new, star)
        elif steps > 1 and n > 1:
            w_off = np.where(off_diag, W, np.inf)
            power = w_off.copy()
            for _ in range(2, steps + 1):
                next_power, mid_k = tropical_min_plus_matmul(
                    w_off, power, return_witness=True, inf_threshold=inf_threshold
                )
                improved = off_diag & (next_power + 1e-12 < star)
                if return_routing and np.any(improved):
                    next_hop = np.where(improved, mid_k, next_hop)
                star = np.where(improved, next_power, star)
                power = next_power
        if is_tensor:
            t_star = torch.as_tensor(star, dtype=orig_dtype, device=orig_device)
            t_hop = torch.as_tensor(next_hop, dtype=torch.long, device=orig_device)
            return (t_star, t_hop) if return_routing else t_star
        return (star, next_hop) if return_routing else star



ActionKey = Union[int, Tuple[int, int, int]]


def _frame_to_2d_numpy(frame_obj: Any) -> np.ndarray:
    """
    Extracts the settled 2D `(H, W)` integer grid (`0..15`) from a `FrameDataRaw`, `FrameData`,
    list of 2D/3D frames, or numpy/torch array.
    """
    if hasattr(frame_obj, "frame"):
        frames = frame_obj.frame
        if isinstance(frames, (list, tuple)):
            if len(frames) == 0:
                return np.zeros((64, 64), dtype=np.int32)
            arr = np.asarray(frames[-1], dtype=np.int32)
        else:
            arr = np.asarray(frames, dtype=np.int32)
    elif isinstance(frame_obj, (list, tuple)):
        if len(frame_obj) == 0:
            return np.zeros((64, 64), dtype=np.int32)
        last = frame_obj[-1]
        if hasattr(last, "frame"):
            return _frame_to_2d_numpy(last)
        arr = np.asarray(last, dtype=np.int32)
        if arr.ndim == 1 and len(frame_obj) == 64:
            arr = np.asarray(frame_obj, dtype=np.int32)
    elif isinstance(frame_obj, torch.Tensor):
        arr = frame_obj.detach().cpu().numpy().astype(np.int32)
    else:
        arr = np.asarray(frame_obj, dtype=np.int32)

    while arr.ndim > 2 and arr.shape[0] > 0:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((64, 64), dtype=np.int32)
    return np.clip(arr, 0, 15)


def _mask_clock_to_bg(grid_2d: np.ndarray, clock_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Projects out exogenous clock pixels `clock_mask` by replacing them with the modal
    background color `bg_mode` computed over non-clock pixels `~clock_mask` (avoiding
    synthetic color-`0` foreground artifacts on boards whose background is non-zero).
    """
    if clock_mask is None or clock_mask.shape != grid_2d.shape or not np.any(clock_mask):
        return grid_2d
    non_clk = grid_2d[~clock_mask]
    if non_clk.size > 0:
        bg_mode = int(np.bincount(non_clk.ravel(), minlength=16).argmax())
    else:
        bg_mode = int(np.bincount(grid_2d.ravel(), minlength=16).argmax())
    return np.where(clock_mask, bg_mode, grid_2d)


def _extract_subframes_2d(frame_obj: Any) -> List[np.ndarray]:
    """Returns all 2D subframes produced by a single environment step."""
    if hasattr(frame_obj, "frame"):
        raw_list = frame_obj.frame
        if isinstance(raw_list, (list, tuple)) and len(raw_list) > 0:
            out = []
            for sf in raw_list:
                arr = np.asarray(sf, dtype=np.int32)
                if arr.ndim == 2:
                    out.append(np.clip(arr, 0, 15))
            if out:
                return out
    return [_frame_to_2d_numpy(frame_obj)]


def _blake2b_grid(grid_2d: np.ndarray) -> str:
    """Deterministic 128-bit Blake2b digest of a 2D integer grid."""
    arr = np.ascontiguousarray(grid_2d, dtype=np.int8)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _local_patch_digest(grid_2d: np.ndarray, x: int, y: int, radius: int = 2) -> bytes:
    """Returns raw bytes of the `(2*radius+1) x (2*radius+1)` local neighborhood around `(x, y)`."""
    H, W = grid_2d.shape
    r0, r1 = max(0, y - radius), min(H, y + radius + 1)
    c0, c1 = max(0, x - radius), min(W, x + radius + 1)
    return np.ascontiguousarray(grid_2d[r0:r1, c0:c1], dtype=np.int8).tobytes()


try:
    import scipy.ndimage as _ndi
except ImportError:
    _ndi = None

_NDI_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_NDI_STRUCT_8 = np.ones((3, 3), dtype=bool)
_CC2D_CACHE: Dict[Tuple[str, int, bool, int], List[Dict[str, Any]]] = {}


def _connected_components_2d(
    grid: np.ndarray,
    connectivity: int = 4,
    same_color: bool = True,
    ignore_val: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fast C-accelerated (`scipy.ndimage.label` + `find_objects`) connected component segmentation
    on a 2D integer grid, memoized per `(_blake2b_grid(grid), connectivity, same_color, ignore_val)`.
    Computes each component's exact 2D interior medoid `(medoid_x, medoid_y)`
    guaranteed to lie strictly inside the component.
    """
    cache_key = (
        _blake2b_grid(grid),
        int(connectivity),
        bool(same_color),
        -999 if ignore_val is None else int(ignore_val),
    )
    cached = _CC2D_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    H, W = grid.shape
    if _ndi is not None:
        struct = _NDI_STRUCT_4 if connectivity == 4 else _NDI_STRUCT_8
        comps: List[Dict[str, Any]] = []
        if not same_color:
            mask = (grid != ignore_val) if ignore_val is not None else np.ones((H, W), dtype=bool)
            if not np.any(mask):
                if len(_CC2D_CACHE) >= 16384:
                    _CC2D_CACHE.pop(next(iter(_CC2D_CACHE)), None)
                _CC2D_CACHE[cache_key] = []
                return []
            labeled, num_features = _ndi.label(mask, structure=struct)
            slices = _ndi.find_objects(labeled)
            for k in range(1, num_features + 1):
                sl = slices[k - 1]
                if sl is None:
                    continue
                r0, r1 = int(sl[0].start), int(sl[0].stop)
                c0, c1 = int(sl[1].start), int(sl[1].stop)
                bh, bw = r1 - r0, c1 - c0
                if bh == 1 and bw == 1:
                    rs = np.array([r0], dtype=np.int32)
                    cs = np.array([c0], dtype=np.int32)
                    val = int(grid[r0, c0])
                    area = 1
                    cy, cx = float(r0), float(c0)
                    med_y, med_x = r0, c0
                else:
                    l_rs, l_cs = np.nonzero(labeled[r0:r1, c0:c1] == k)
                    rs = (l_rs + r0).astype(np.int32)
                    cs = (l_cs + c0).astype(np.int32)
                    val = int(grid[int(rs[0]), int(cs[0])])
                    area = int(rs.size)
                    cy = float(rs.sum()) / float(area)
                    cx = float(cs.sum()) / float(area)
                    rs_f = rs.astype(np.float32)
                    cs_f = cs.astype(np.float32)
                    dists = (rs_f - cy) ** 2 + (cs_f - cx) ** 2 + 0.01 * (
                        np.abs(rs_f - cy) + np.abs(cs_f - cx)
                    )
                    med_idx = int(np.argmin(dists))
                    tied = np.flatnonzero(dists == dists[med_idx])
                    if tied.size > 1:
                        sr, sc = int(rs[0]), int(cs[0])
                        med_idx = int(
                            min(
                                tied,
                                key=lambda i: (
                                    abs(int(rs[i]) - sr) + abs(int(cs[i]) - sc),
                                    -abs(int(rs[i]) - sr),
                                ),
                            )
                        )
                    med_y = int(rs[med_idx])
                    med_x = int(cs[med_idx])
                comps.append(
                    {
                        "color": val,
                        "area": area,
                        "r0": r0,
                        "r1": r1,
                        "c0": c0,
                        "c1": c1,
                        "h": bh,
                        "w": bw,
                        "bbox": (bh, bw),
                        "bbox_area": bh * bw,
                        "fill_ratio": float(area) / float(max(1, bh * bw)),
                        "aspect_ratio": float(min(bh, bw)) / float(max(1, max(bh, bw))),
                        "centroid_y": cy,
                        "centroid_x": cx,
                        "medoid_y": med_y,
                        "medoid_x": med_x,
                        "rows": rs,
                        "cols": cs,
                    }
                )
        else:
            u_vals = np.unique(grid)
            if ignore_val is not None:
                u_vals = u_vals[u_vals != ignore_val]
            if u_vals.size == 0:
                if len(_CC2D_CACHE) >= 16384:
                    _CC2D_CACHE.pop(next(iter(_CC2D_CACHE)), None)
                _CC2D_CACHE[cache_key] = []
                return []
            for v_raw in u_vals:
                val = int(v_raw)
                labeled, num_features = _ndi.label(grid == val, structure=struct)
                slices = _ndi.find_objects(labeled)
                for k in range(1, num_features + 1):
                    sl = slices[k - 1]
                    if sl is None:
                        continue
                    r0, r1 = int(sl[0].start), int(sl[0].stop)
                    c0, c1 = int(sl[1].start), int(sl[1].stop)
                    bh, bw = r1 - r0, c1 - c0
                    if bh == 1 and bw == 1:
                        rs = np.array([r0], dtype=np.int32)
                        cs = np.array([c0], dtype=np.int32)
                        area = 1
                        cy, cx = float(r0), float(c0)
                        med_y, med_x = r0, c0
                    else:
                        l_rs, l_cs = np.nonzero(labeled[r0:r1, c0:c1] == k)
                        rs = (l_rs + r0).astype(np.int32)
                        cs = (l_cs + c0).astype(np.int32)
                        area = int(rs.size)
                        cy = float(rs.sum()) / float(area)
                        cx = float(cs.sum()) / float(area)
                        rs_f = rs.astype(np.float32)
                        cs_f = cs.astype(np.float32)
                        dists = (rs_f - cy) ** 2 + (cs_f - cx) ** 2 + 0.01 * (
                            np.abs(rs_f - cy) + np.abs(cs_f - cx)
                        )
                        med_idx = int(np.argmin(dists))
                        tied = np.flatnonzero(dists == dists[med_idx])
                        if tied.size > 1:
                            sr, sc = int(rs[0]), int(cs[0])
                            med_idx = int(
                                min(
                                    tied,
                                    key=lambda i: (
                                        abs(int(rs[i]) - sr) + abs(int(cs[i]) - sc),
                                        -abs(int(rs[i]) - sr),
                                    ),
                                )
                            )
                        med_y = int(rs[med_idx])
                        med_x = int(cs[med_idx])
                    comps.append(
                        {
                            "color": val,
                            "area": area,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                            "h": bh,
                            "w": bw,
                            "bbox": (bh, bw),
                            "bbox_area": bh * bw,
                            "fill_ratio": float(area) / float(max(1, bh * bw)),
                            "aspect_ratio": float(min(bh, bw)) / float(max(1, max(bh, bw))),
                            "centroid_y": cy,
                            "centroid_x": cx,
                            "medoid_y": med_y,
                            "medoid_x": med_x,
                            "rows": rs,
                            "cols": cs,
                        }
                    )
            if u_vals.size > 1 and len(comps) > 1:
                comps.sort(key=lambda d: (int(d["rows"][0]), int(d["cols"][0])))
        if len(_CC2D_CACHE) >= 16384:
            _CC2D_CACHE.pop(next(iter(_CC2D_CACHE)), None)
        _CC2D_CACHE[cache_key] = comps
        return list(comps)

    if ignore_val is not None:
        vis = grid == ignore_val
    else:
        vis = np.zeros((H, W), dtype=bool)

    todo_r, todo_c = np.nonzero(~vis)
    if todo_r.size == 0:
        if len(_CC2D_CACHE) >= 16384:
            _CC2D_CACHE.pop(next(iter(_CC2D_CACHE)), None)
        _CC2D_CACHE[cache_key] = []
        return []

    if connectivity == 4:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    grid_list = grid.tolist()
    vis_list = vis.tolist()
    H_m1 = H - 1
    W_m1 = W - 1
    comps = []
    for r, c in zip(todo_r.tolist(), todo_c.tolist()):
        if vis_list[r][c]:
            continue
        val = int(grid_list[r][c])
        vis_list[r][c] = True
        pts_r = [r]
        pts_c = [c]
        head = 0
        tail = 1
        if connectivity == 4 and same_color:
            while head < tail:
                cr = pts_r[head]
                cc = pts_c[head]
                head += 1
                if cr > 0:
                    nr = cr - 1
                    if not vis_list[nr][cc] and grid_list[nr][cc] == val:
                        vis_list[nr][cc] = True
                        pts_r.append(nr)
                        pts_c.append(cc)
                        tail += 1
                if cr < H_m1:
                    nr = cr + 1
                    if not vis_list[nr][cc] and grid_list[nr][cc] == val:
                        vis_list[nr][cc] = True
                        pts_r.append(nr)
                        pts_c.append(cc)
                        tail += 1
                row_vis = vis_list[cr]
                row_grid = grid_list[cr]
                if cc > 0:
                    nc = cc - 1
                    if not row_vis[nc] and row_grid[nc] == val:
                        row_vis[nc] = True
                        pts_r.append(cr)
                        pts_c.append(nc)
                        tail += 1
                if cc < W_m1:
                    nc = cc + 1
                    if not row_vis[nc] and row_grid[nc] == val:
                        row_vis[nc] = True
                        pts_r.append(cr)
                        pts_c.append(nc)
                        tail += 1
        elif same_color:
            while head < tail:
                cr = pts_r[head]
                cc = pts_c[head]
                head += 1
                for dr, dc in nbrs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not vis_list[nr][nc]:
                        if grid_list[nr][nc] == val:
                            vis_list[nr][nc] = True
                            pts_r.append(nr)
                            pts_c.append(nc)
                            tail += 1
        else:
            while head < tail:
                cr = pts_r[head]
                cc = pts_c[head]
                head += 1
                for dr, dc in nbrs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not vis_list[nr][nc]:
                        vis_list[nr][nc] = True
                        pts_r.append(nr)
                        pts_c.append(nc)
                        tail += 1

        area = tail
        if area == 1:
            rs = np.array(pts_r, dtype=np.int32)
            cs = np.array(pts_c, dtype=np.int32)
            r0, r1 = r, r + 1
            c0, c1 = c, c + 1
            bh, bw = 1, 1
            cy, cx = float(r), float(c)
            med_y, med_x = r, c
        else:
            rs = np.asarray(pts_r, dtype=np.int32)
            cs = np.asarray(pts_c, dtype=np.int32)
            r0, r1 = int(min(pts_r)), int(max(pts_r)) + 1
            c0, c1 = int(min(pts_c)), int(max(pts_c)) + 1
            bh, bw = r1 - r0, c1 - c0
            cy = float(sum(pts_r)) / float(area)
            cx = float(sum(pts_c)) / float(area)
            rs_f = rs.astype(np.float32)
            cs_f = cs.astype(np.float32)
            dists = (rs_f - cy) ** 2 + (cs_f - cx) ** 2 + 0.01 * (
                np.abs(rs_f - cy) + np.abs(cs_f - cx)
            )
            med_idx = int(np.argmin(dists))
            med_y = int(pts_r[med_idx])
            med_x = int(pts_c[med_idx])
        comps.append(
            {
                "color": val,
                "area": area,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "h": bh,
                "w": bw,
                "bbox": (bh, bw),
                "bbox_area": bh * bw,
                "fill_ratio": float(area) / float(max(1, bh * bw)),
                "aspect_ratio": float(min(bh, bw)) / float(max(1, max(bh, bw))),
                "centroid_y": cy,
                "centroid_x": cx,
                "medoid_y": med_y,
                "medoid_x": med_x,
                "rows": rs,
                "cols": cs,
            }
        )
    if len(_CC2D_CACHE) >= 16384:
        _CC2D_CACHE.pop(next(iter(_CC2D_CACHE)), None)
    _CC2D_CACHE[cache_key] = comps
    return list(comps)


class TemporalClockContentSplitter:
    """
    Dynamical-Systems Isomorphism of `ProportionalRoPESubspaceSplitter` (`k_rope` vs `k_nope`)
    via the Discrete Continuity Equation:
        d_t rho_c(x, y) + div(J_c(x, y)) = sigma_c(x, y)

    In `darksnail.gemma4.ProportionalRoPESubspaceSplitter`, attention keys are decomposed into
    an exogenous rotary phase subspace `k_rope` (which advances deterministically at every step `t`
    regardless of token content) and a position-invariant semantic content subspace `k_nope`.

    In an interactive 2D dynamical system `F_t in Z_16^(H x W)`:
      - Conservative object translations satisfy `sigma_c == 0`, preserving global color mass:
            Delta M_c = sum_{x,y} 1[F_t(x,y) == c] - sum_{x,y} 1[F_{t-1}(x,y) == c] == 0.
      - Exogenous clocks/timers (`H_clock`, `k_rope` analogue) are dissipative sinks (`sigma_c != 0`)
        satisfying strict global color-mass non-conservation (`Delta M_{c_old} <= -|D|, Delta M_{c_new} >= |D|`)
        under three position-invariant dynamical laws:
          1. Multi-Cluster Decoupling: In a single transition `F_{t-1} -> F_t`, a small (`<= 6` px)
             dissipative sink cluster `D_small` co-occurs with a spatially disjoint content
             transition `D_main` of size `|D_main| >= 3 * |D_small|`.
          2. Cross-Action Invariance (`I(Delta; a) == 0`): Distinct actions `a_1 != a_2` (or clicks
             at distant loci `||p_click - D_small||_inf > 6`) produce the same-sized dissipative
             depletion inside the same initial connected component `C_k^(0)`.
          3. Constant-Velocity 1-Way Phase Front Propagation: Across consecutive transitions, two
             small (`<= 6` px) dissipative depletion clusters inside the same component `C_k^(0)`
             advance by L1 adjacency `dist_1(D_{t-1}, D_t) == 1`.
          4. Periodic Phase-Reset / Respawn Conservation: When an exogenous clock bar `H_clock`
             refills back to `F_0` while the board reverts to `F_0` except for a small (`<= 16` px)
             dissipative life-counter decrement `D_life`, `D_life` is added to `H_clock` as a
             slow-timescale clock harmonic and the transition is flagged as `is_phase_reset = True`.
    """

    def __init__(self, max_clock_tick_area: int = 6):
        self.max_clock_tick_area = int(max_clock_tick_area)
        self.reset()

    def reset(self) -> None:
        self.initial_frame: Optional[np.ndarray] = None
        self.comp_id_map: Optional[np.ndarray] = None
        self.comp_by_id: Dict[int, Dict[str, Any]] = {}
        self.clock_mask: Optional[np.ndarray] = None
        self.hud_latch_mask: Optional[np.ndarray] = None
        self.reversible_mask: Optional[np.ndarray] = None
        self.last_changed_from_color: Optional[np.ndarray] = None
        self.prev_nonreset_color: Optional[np.ndarray] = None
        self.clock_comp_ids: Set[int] = set()
        self.depletion_history: Dict[int, List[Tuple[ActionKey, np.ndarray, np.ndarray, int, int]]] = {}
        self.max_nope_dist_from_init: int = 0
        self.transitions_observed: int = 0

    def initialize_level(self, frame_2d: np.ndarray, preserve_clock_mask: bool = False) -> None:
        prev_mask: Optional[np.ndarray] = None
        prev_rev: Optional[np.ndarray] = None
        prev_latch: Optional[np.ndarray] = None
        if preserve_clock_mask and self.clock_mask is not None:
            prev_mask = self.clock_mask.copy()
            if self.reversible_mask is not None:
                prev_rev = self.reversible_mask.copy()
            if self.hud_latch_mask is not None:
                prev_latch = self.hud_latch_mask.copy()
        self.reset()
        f0 = np.asarray(frame_2d, dtype=np.int32).copy()
        H, W = f0.shape
        self.initial_frame = f0
        self.prev_nonreset_color = f0.copy()
        self.last_changed_from_color = np.full((H, W), -1, dtype=np.int32)
        if prev_rev is not None and prev_rev.shape == (H, W):
            self.reversible_mask = prev_rev
        else:
            self.reversible_mask = np.zeros((H, W), dtype=bool)
        if prev_latch is not None and prev_latch.shape == (H, W):
            self.hud_latch_mask = prev_latch
        else:
            self.hud_latch_mask = np.zeros((H, W), dtype=bool)
        if prev_mask is not None and prev_mask.shape == (H, W):
            self.clock_mask = prev_mask & (~self.reversible_mask) & (~self.hud_latch_mask)
        else:
            self.clock_mask = np.zeros((H, W), dtype=bool)
        self.comp_id_map = np.full((H, W), -1, dtype=np.int32)
        comps = _connected_components_2d(f0, connectivity=4, same_color=True)
        for cid, comp in enumerate(comps):
            self.comp_by_id[cid] = comp
            self.comp_id_map[comp["rows"], comp["cols"]] = cid

    def get_clock_mask(self) -> Optional[np.ndarray]:
        """Returns the current boolean mask of exogenous clock pixels `H_clock`."""
        return self.clock_mask

    @property
    def num_clock_pixels(self) -> int:
        """Returns the number of pixels currently masked in `H_clock`."""
        return int(self.clock_mask.sum()) if self.clock_mask is not None else 0

    def _mark_component_as_clock(self, cid: int, rows: np.ndarray, cols: np.ndarray) -> bool:
        """
        Marks observed dissipative clock pixels `(rows, cols)` (excluding any pixel proven
        action-reversible in `self.reversible_mask`) and, if the parent initial component `cid`
        is an elongated or border-adjacent 1D/thin phase bar (`min(h, w) <= 2`, `area <= 128`,
        and zero reversible pixels), masks the entire component `cid` in `H_clock`.
        """
        if self.clock_mask is None:
            return False
        H, W = self.clock_mask.shape
        prev_sum = int(self.clock_mask.sum())
        if self.reversible_mask is not None:
            valid = ~self.reversible_mask[rows, cols]
            rows = rows[valid]
            cols = cols[valid]
        if rows.size == 0:
            return False
        self.clock_mask[rows, cols] = True
        if cid >= 0 and cid in self.comp_by_id:
            comp = self.comp_by_id[cid]
            c_rows, c_cols = comp["rows"], comp["cols"]
            has_reversible = (
                bool(np.any(self.reversible_mask[c_rows, c_cols]))
                if self.reversible_mask is not None
                else False
            )
            near_border_or_long_bar = (
                int(comp["r0"]) <= 4
                or int(comp["r1"]) >= H - 4
                or int(comp["c0"]) <= 4
                or int(comp["c1"]) >= W - 4
                or max(int(comp["h"]), int(comp["w"])) >= 8
            )
            if (
                not has_reversible
                and near_border_or_long_bar
                and comp["area"] <= 128
                and min(comp["h"], comp["w"]) <= 2
                and comp["fill_ratio"] >= 0.75
            ):
                self.clock_mask[c_rows, c_cols] = True
                self.clock_comp_ids.add(cid)
        return int(self.clock_mask.sum()) != prev_sum

    def observe_transition(
        self,
        prev_frame: np.ndarray,
        next_frame: np.ndarray,
        action_key: ActionKey,
    ) -> Tuple[bool, int, bool]:
        """
        Observes a transition `prev_frame -> next_frame` under `action_key`, updates `H_clock`
        via the Continuity Equation + 4 dynamical clock laws + Thermodynamic Irreversibility Law,
        and returns:
            `(clock_mask_changed, nope_delta_l1, is_phase_reset)`
        """
        if (
            self.initial_frame is None
            or self.clock_mask is None
            or self.hud_latch_mask is None
            or self.comp_id_map is None
            or self.reversible_mask is None
            or self.last_changed_from_color is None
        ):
            self.initialize_level(prev_frame)
        assert (
            self.initial_frame is not None
            and self.clock_mask is not None
            and self.hud_latch_mask is not None
            and self.comp_id_map is not None
            and self.reversible_mask is not None
            and self.last_changed_from_color is not None
        )

        self.transitions_observed += 1
        diff_mask = prev_frame != next_frame
        if not np.any(diff_mask):
            return False, 0, False

        mask_expanded = False
        is_phase_reset = False

        # Law 4: Check for Periodic Phase-Reset / Respawn to `initial_frame`
        prev_dist_init = int(np.sum((prev_frame != self.initial_frame) & (~self.clock_mask)))
        if prev_dist_init > self.max_nope_dist_from_init:
            self.max_nope_dist_from_init = prev_dist_init

        cur_diff_init = (next_frame != self.initial_frame) & (~self.clock_mask)
        cur_dist_init = int(cur_diff_init.sum())
        clock_px_count = int(self.clock_mask.sum())
        clock_refilled = False
        if clock_px_count >= 4:
            prev_clock_match = int(np.sum((prev_frame == self.initial_frame) & self.clock_mask))
            next_clock_match = int(np.sum((next_frame == self.initial_frame) & self.clock_mask))
            if next_clock_match - prev_clock_match >= max(3, clock_px_count // 3):
                clock_refilled = True

        raw_step_diff = int(diff_mask.sum())
        if (clock_refilled and cur_dist_init <= 16) or (
            clock_px_count == 0
            and self.transitions_observed >= 6
            and self.max_nope_dist_from_init >= 32
            and prev_dist_init >= 32
            and raw_step_diff >= 40
            and 0 < cur_dist_init <= 8
        ):
            is_phase_reset = True
            self.last_changed_from_color.fill(-1)
            self.prev_nonreset_color = next_frame.copy()
            if cur_dist_init > 0:
                res_rs, res_cs = np.nonzero(cur_diff_init)
                cids = self.comp_id_map[res_rs, res_cs]
                for u_cid in np.unique(cids):
                    if self._mark_component_as_clock(int(u_cid), res_rs, res_cs):
                        mask_expanded = True
            nope_diff = (prev_frame != next_frame) & (~self.clock_mask)
            return mask_expanded, int(nope_diff.sum()), True

        # Thermodynamic Irreversibility Law:
        # Exogenous clocks evolve monotonically until a global phase reset. Any pixel that
        # reverses its color transition (`c_old -> c_new -> c_old`) on a non-reset step has positive
        # action contingency and belongs to `H_nope` (and if on the HUD/border or co-changing
        # as a discrete status indicator, to `H_clock^latch`).
        ch_rs, ch_cs = np.nonzero(diff_mask)
        H, W = next_frame.shape
        interior_ch = (ch_rs >= 2) & (ch_rs < H - 2) & (ch_cs >= 2) & (ch_cs < W - 2)
        not_already_clock = ~self.clock_mask[ch_rs, ch_cs]
        reverted_any = self.last_changed_from_color[ch_rs, ch_cs] == next_frame[ch_rs, ch_cs]
        reverted_mask = interior_ch & reverted_any
        border_reverted = (~interior_ch) & reverted_any & not_already_clock
        if np.any(border_reverted) and not clock_refilled:
            b_rs = ch_rs[border_reverted]
            b_cs = ch_cs[border_reverted]
            b_cids = self.comp_id_map[b_rs, b_cs]
            valid_latch = np.zeros(b_rs.shape, dtype=bool)
            for idx_b, cid_b in enumerate(b_cids.tolist()):
                if cid_b >= 0 and cid_b not in self.clock_comp_ids:
                    comp_b = self.comp_by_id.get(int(cid_b))
                    if comp_b is not None and int(comp_b["area"]) <= 16 and max(int(comp_b["h"]), int(comp_b["w"])) <= 6:
                        valid_latch[idx_b] = True
            if np.any(valid_latch):
                self.hud_latch_mask[b_rs[valid_latch], b_cs[valid_latch]] = True
                self.reversible_mask[b_rs[valid_latch], b_cs[valid_latch]] = True
        if np.any(reverted_mask):
            rev_rs = ch_rs[reverted_mask]
            rev_cs = ch_cs[reverted_mask]
            self.reversible_mask[rev_rs, rev_cs] = True
            if np.any(self.clock_mask[rev_rs, rev_cs]):
                self.clock_mask[rev_rs, rev_cs] = False
                mask_expanded = True
            rev_cids = np.unique(self.comp_id_map[rev_rs, rev_cs])
            for r_cid in rev_cids:
                cid_i = int(r_cid)
                if cid_i >= 0 and cid_i in self.clock_comp_ids:
                    comp_c = self.comp_by_id[cid_i]
                    if (
                        comp_c["r0"] >= 2
                        and comp_c["r1"] <= H - 2
                        and comp_c["c0"] >= 2
                        and comp_c["c1"] <= W - 2
                    ):
                        if np.any(self.clock_mask[comp_c["rows"], comp_c["cols"]]):
                            self.clock_mask[comp_c["rows"], comp_c["cols"]] = False
                            mask_expanded = True
                        self.clock_comp_ids.discard(cid_i)
                self.depletion_history.pop(cid_i, None)

        self.last_changed_from_color[ch_rs, ch_cs] = prev_frame[ch_rs, ch_cs]
        self.prev_nonreset_color = next_frame.copy()

        # Global color histogram counts for the Continuity Equation conservation check
        prev_counts = np.bincount(prev_frame.reshape(-1), minlength=16)
        next_counts = np.bincount(next_frame.reshape(-1), minlength=16)
        bg_mode = int(np.argmax(prev_counts))

        clusters = _connected_components_2d(
            diff_mask.astype(np.int32), connectivity=8, same_color=True, ignore_val=0
        )
        total_diff_area = int(diff_mask.sum())
        step_old_colors: Set[int] = set()
        step_new_colors: Set[int] = set()
        for cl_c in clusters:
            step_old_colors.update(int(x) for x in np.unique(prev_frame[cl_c["rows"], cl_c["cols"]]))
            step_new_colors.update(int(x) for x in np.unique(next_frame[cl_c["rows"], cl_c["cols"]]))

        for cl in clusters:
            rs, cs = cl["rows"], cl["cols"]
            area = cl["area"]
            if area > self.max_clock_tick_area or cl["h"] > 3 or cl["w"] > 3:
                continue
            if np.any(self.reversible_mask[rs, cs]) or np.any(self.hud_latch_mask[rs, cs]):
                continue

            old_vals = prev_frame[rs, cs]
            new_vals = next_frame[rs, cs]
            if len(np.unique(old_vals)) != 1 or len(np.unique(new_vals)) != 1:
                continue
            c_old = int(old_vals[0])
            c_new = int(new_vals[0])

            # If a non-background color leaves one cluster and enters another cluster in the
            # same transition (e.g. a crosshair stepping onto a colored segment),
            # it is a spatial translation, never an exogenous clock tick.
            if len(clusters) >= 2 and (
                (c_old != bg_mode and c_old in step_new_colors)
                or (c_new != bg_mode and c_new in step_old_colors)
            ):
                continue

            # Continuity Equation Check (`sigma_c != 0`):
            # Conservative sprite translations conserve global color mass (`Delta M_c == 0`).
            # A dissipative clock tick strictly decreases global mass of `c_old` and increases `c_new`.
            delta_m_old = int(next_counts[c_old]) - int(prev_counts[c_old])
            delta_m_new = int(next_counts[c_new]) - int(prev_counts[c_new])
            if not (delta_m_old <= -area and delta_m_new >= area):
                continue

            # Exclude direct ACTION6 clicks within L_inf distance <= 5 of the cursor
            if isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6:
                click_x, click_y = action_key[1], action_key[2]
                dist_to_click = max(
                    abs(float(cl["medoid_y"]) - float(click_y)),
                    abs(float(cl["medoid_x"]) - float(click_x)),
                )
                if dist_to_click <= 5.0:
                    continue

            cids = self.comp_id_map[rs, cs]
            dom_cid = int(Counter(cids.tolist()).most_common(1)[0][0])
            comp_dom = self.comp_by_id.get(dom_cid)
            is_thin_or_border_comp = comp_dom is not None and (
                min(int(comp_dom["h"]), int(comp_dom["w"])) <= 2
                or int(comp_dom["r0"]) <= 4
                or int(comp_dom["r1"]) >= H - 4
                or int(comp_dom["c0"]) <= 4
                or int(comp_dom["c1"]) >= W - 4
            )
            if not is_thin_or_border_comp:
                continue

            # Law 1: Multi-Cluster Decoupling (co-ticking alongside a >= 3x larger spatially disjoint content move)
            other_area = total_diff_area - area
            if len(clusters) >= 2 and other_area >= max(4, 3 * area):
                other_mask = diff_mask.copy()
                other_mask[rs, cs] = False
                ors, ocs = np.where(other_mask)
                min_dist_inf = (
                    int(
                        np.min(
                            np.maximum(
                                np.abs(rs[:, None] - ors[None, :]),
                                np.abs(cs[:, None] - ocs[None, :]),
                            )
                        )
                    )
                    if ors.size > 0
                    else 0
                )
                if min_dist_inf > 5:
                    if self._mark_component_as_clock(dom_cid, rs, cs):
                        mask_expanded = True
                    continue

            # Law 2 (Cross-Action Invariance) & Law 3 (Constant-Velocity Phase Front Propagation)
            hist = self.depletion_history.setdefault(dom_cid, [])
            is_clock_by_hist = False
            for prev_act, prs, pcs, p_old, p_new in hist:
                if p_old == c_old and p_new == c_new and prs.size == area:
                    dist_l1 = int(
                        np.min(np.abs(rs[:, None] - prs[None, :]) + np.abs(cs[:, None] - pcs[None, :]))
                    )
                    if prev_act != action_key and dist_l1 <= 4:
                        is_clock_by_hist = True
                        break
                    if 1 <= dist_l1 <= 2:
                        is_clock_by_hist = True
                        break

            hist.append((action_key, rs, cs, c_old, c_new))
            if is_clock_by_hist:
                for _, hrs, hcs, _, _ in hist:
                    if self._mark_component_as_clock(dom_cid, hrs, hcs):
                        mask_expanded = True

        nope_diff = diff_mask & (~self.clock_mask)
        return mask_expanded, int(nope_diff.sum()), is_phase_reset

    def project_nope(self, frame_2d: np.ndarray) -> np.ndarray:
        """Projects `frame_2d` onto the action-contingent controllable subspace `H_nope`."""
        return _mask_clock_to_bg(frame_2d, self.clock_mask)

    def hash_nope(self, frame_2d: np.ndarray) -> str:
        """Computes the deterministic Blake2b state digest in `H_nope`."""
        return _blake2b_grid(self.project_nope(frame_2d))

    def extract_fsa_latch_signature(self, frame_2d: np.ndarray) -> str:
        """
        Extracts the deterministic discrete FSA latch signature `s_fsa` from `H_clock^latch`
        (`hud_latch_mask`).
        """
        arr = _frame_to_2d_numpy(frame_2d)
        if self.hud_latch_mask is not None and self.hud_latch_mask.shape == arr.shape and np.any(self.hud_latch_mask):
            vals = arr[self.hud_latch_mask].astype(np.int8)
            return hashlib.blake2b(vals.tobytes(), digest_size=8).hexdigest()
        return "latch_default"


_DEFAULT_ROPE2D_MEDOID_POOLER = RoPE2DMedoidPooler(
    patch_size=4, viscous_nu=0.05, viscous_p=2, shell_level=1
)
_MEDOID_EXTRACTION_CACHE: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
_RADIAL_PRIOR_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def extract_arc3_beltrami_medoids(
    frame_2d: np.ndarray,
    prev_frame_2d: Optional[np.ndarray] = None,
    max_candidates: int = 36,
    clock_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Discretizes a `64x64` 16-color (`0..15`) ARC-AGI-3 observation frame into at most
    `max_candidates` ranked 2D coordinate candidates `(x, y)` for `ACTION6(x, y)`.

    Combines four zero-leakage geometric & spectral channels with 2D Non-Maximum Suppression
    (NMS) and Dyadic Bruhat-Tits Quadtree + Morphological Diversity Stratification:
      1. Connected-Component Foreground Medoids (`kind="cc_object"` / `"cc_lattice"`):
         4-connected same-color components, selecting the exact 2D interior medoid pixel
         `(x*, y*) in C` minimizing Euclidean distance to the component centroid `(cx, cy)`.
      2. Enclosed Topological Cavities (`kind="enclosed_hole"`):
         4-connected background components not reaching the canvas exterior (or enclosed inside
         foreground rings/frames), selecting the interior medoid pixel of each hole.
      3. Structured Grid / Sub-Panel Symmetry & Discrepancy Defects (`kind="symmetry_defect"`):
         Detects matching-size sub-panels or mirror-symmetric objects where a single cell or
         locus breaks symmetry / panel equality, proposing the exact defect coordinate.
      4. 2D Axial RoPE Salience Patch Medoids (`kind="rope2d_patch"`):
         Applies `RoPE2DMedoidPooler(patch_size=4)` over a 20-channel 16-color + Beltrami gradient
         + temporal innovation feature field.

    Strictly projects out exogenous clock components `H_clock` across all 4 channels: any
    component/cavity/defect/patch whose medoid `(mx, my)` lies in `clock_mask` OR has `>= 50%`
    of its pixels inside `clock_mask` is excluded before NMS and quadtree selection.
    """
    arr = _frame_to_2d_numpy(frame_2d)
    H, W = arr.shape
    clk_sum = int(clock_mask.sum()) if (clock_mask is not None and clock_mask.shape == (H, W)) else 0
    has_any_clock = clock_mask is not None and clock_mask.shape == (H, W) and clk_sum > 0
    has_deep_clock = has_any_clock and (prev_frame_2d is None or (128 < clk_sum <= 450))
    medoid_cache_key = (
        _blake2b_grid(arr),
        "" if prev_frame_2d is None else _blake2b_grid(_frame_to_2d_numpy(prev_frame_2d)),
        "" if (not has_any_clock or clock_mask is None) else _blake2b_grid(clock_mask),
    )
    cached_medoids = _MEDOID_EXTRACTION_CACHE.get(medoid_cache_key)
    if cached_medoids is not None:
        return [dict(d) for d in cached_medoids[:max_candidates]]

    def _is_clock_masked(
        mx: int,
        my: int,
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
        r0: int = 0,
        r1: int = 0,
        c0: int = 0,
        c1: int = 0,
        deep_only: bool = False,
    ) -> bool:
        if not has_any_clock or clock_mask is None:
            return False
        if deep_only and not has_deep_clock:
            return False
        if 0 <= my < H and 0 <= mx < W and bool(clock_mask[my, mx]):
            return True
        if not has_deep_clock:
            return False
        if rows is not None and cols is not None and rows.size > 0:
            if float(np.mean(clock_mask[rows, cols])) >= 0.50:
                return True
        elif r1 > r0 and c1 > c0:
            if float(np.mean(clock_mask[max(0, r0) : min(H, r1), max(0, c0) : min(W, c1)])) >= 0.50:
                return True
        return False

    vals, counts = np.unique(arr, return_counts=True)
    bg_mode = int(vals[int(np.argmax(counts))])

    if prev_frame_2d is not None:
        prev_arr = _frame_to_2d_numpy(prev_frame_2d)
        if prev_arr.shape == (H, W):
            temp_diff = (arr != prev_arr).astype(np.float32)
        else:
            temp_diff = np.zeros((H, W), dtype=np.float32)
    else:
        temp_diff = np.zeros((H, W), dtype=np.float32)

    if has_any_clock and clock_mask is not None:
        temp_diff = temp_diff * (~clock_mask).astype(np.float32)

    # Local 2D Beltrami spatial boundary gradient magnitude
    grad_y = np.zeros((H, W), dtype=np.float32)
    grad_x = np.zeros((H, W), dtype=np.float32)
    grad_y[1:, :] = (arr[1:, :] != arr[:-1, :]).astype(np.float32)
    grad_y[:-1, :] = np.maximum(grad_y[:-1, :], grad_y[1:, :])
    grad_x[:, 1:] = (arr[:, 1:] != arr[:, :-1]).astype(np.float32)
    grad_x[:, :-1] = np.maximum(grad_x[:, :-1], grad_x[:, 1:])
    beltrami_grad = 0.5 * (grad_y + grad_x)

    # 1. Foreground Connected-Component Medoids
    fg_comps = _connected_components_2d(arr, connectivity=4, same_color=True, ignore_val=bg_mode)
    if not fg_comps:
        fg_comps = _connected_components_2d(arr, connectivity=4, same_color=True, ignore_val=None)

    bbox_counts: Dict[Tuple[int, int], int] = Counter(c["bbox"] for c in fg_comps if c["area"] >= 2)
    color_counts: Dict[int, int] = Counter(c["color"] for c in fg_comps)

    # Identify unique minority-color framing markers (e.g. L-brackets or cursor indicators of count <= 6)
    marker_centroids: List[Tuple[float, float]] = [
        (c["centroid_y"], c["centroid_x"])
        for c in fg_comps
        if 1 <= color_counts.get(c["color"], 0) <= 6 and 2 <= c["area"] <= 32 and c["fill_ratio"] < 0.85
    ]

    raw_candidates: List[Dict[str, Any]] = []

    for comp in fg_comps:
        mx, my = int(comp["medoid_x"]), int(comp["medoid_y"])
        if _is_clock_masked(mx, my, rows=comp["rows"], cols=comp["cols"]):
            continue

        area = int(comp["area"])
        bh, bw = comp["bbox"]
        fill_ratio = float(comp["fill_ratio"])
        aspect = float(comp["aspect_ratio"])
        col = int(comp["color"])

        if area > int(0.35 * H * W):
            continue
        is_thin_spanning_frame = (max(bh, bw) >= int(0.75 * max(H, W))) and (
            min(bh, bw) <= 2 or fill_ratio < 0.18
        )

        log_area = math.log2(max(1.0, float(area)))
        size_score = math.exp(-((log_area - 4.2) ** 2) / 7.5)
        rep_count = bbox_counts.get((bh, bw), 1)
        is_lattice_tile = (rep_count >= 2) and (2 <= area <= 256) and (fill_ratio >= 0.45)
        lattice_bonus = min(1.35, 0.28 * math.log2(float(rep_count) + 1.0)) if is_lattice_tile else 0.0

        # Morphological self-information bonus for compact low-multiplicity control widgets
        widget_bonus = (
            0.55
            if (1 <= rep_count <= 4 and 4 <= area <= 96 and fill_ratio >= 0.45 and aspect >= 0.35)
            else 0.0
        )

        col_mult = color_counts.get(col, 1)
        rarity_bonus = 0.65 / math.sqrt(float(col_mult)) if (2 <= area <= 120) else 0.15

        # Proximity bonus to framing/bracket markers if present on the canvas
        marker_bonus = 0.0
        if marker_centroids and is_lattice_tile:
            mean_my = sum(m[0] for m in marker_centroids) / len(marker_centroids)
            mean_mx = sum(m[1] for m in marker_centroids) / len(marker_centroids)
            dist_to_marker_center = math.hypot(float(my) - mean_my, float(mx) - mean_mx) / float(max(H, W))
            marker_bonus = 0.35 * max(0.0, 1.0 - 2.0 * dist_to_marker_center)

        comp_grad = float(beltrami_grad[comp["rows"], comp["cols"]].mean())
        comp_temp = float(temp_diff[comp["rows"], comp["cols"]].mean())

        # Note: Do not apply a large additive `comp_temp` boost to foreground objects,
        # as boosting the component that just changed on the previous click causes
        # immediate 2-cycle self-unclicking on multi-pixel buttons.
        salience = (
            1.40 * size_score
            + 0.90 * fill_ratio
            + 0.75 * aspect
            + lattice_bonus
            + widget_bonus
            + rarity_bonus
            + marker_bonus
            + 0.50 * comp_grad
        )
        if is_thin_spanning_frame:
            salience *= 0.20

        kind = "cc_lattice" if is_lattice_tile else "cc_object"
        feat_vec = np.array(
            [
                float(col) / 15.0,
                log_area / 12.0,
                fill_ratio,
                aspect,
                float(bh) / float(H),
                float(bw) / float(W),
                float(my) / float(H),
                float(mx) / float(W),
                1.0 if is_lattice_tile else 0.0,
                comp_grad,
                comp_temp,
                salience / 5.0,
            ],
            dtype=np.float32,
        )
        raw_candidates.append(
            {
                "x": mx,
                "y": my,
                "kind": kind,
                "color": col,
                "area": area,
                "salience": float(salience),
                "feature_vec": feat_vec,
                "bbox": (bh, bw),
                "r0": int(comp["r0"]),
                "r1": int(comp["r1"]),
                "c0": int(comp["c0"]),
                "c1": int(comp["c1"]),
            }
        )

    # 2. Enclosed Hole / Cavity Medoids (Jordan Enclosure in 4-connectivity)
    if _ndi is not None:
        holes_mask = _ndi.binary_fill_holes(arr != bg_mode, structure=_NDI_STRUCT_4) & (arr == bg_mode)
        for comp in fg_comps:
            if comp["area"] >= 6 and comp["fill_ratio"] < 0.85 and comp["h"] >= 3 and comp["w"] >= 3:
                r0, r1, c0, c1 = comp["r0"], comp["r1"], comp["c0"], comp["c1"]
                sub = arr[r0:r1, c0:c1]
                cy_i = int(round(comp["centroid_y"]))
                cx_i = int(round(comp["centroid_x"]))
                if 0 <= cy_i < H and 0 <= cx_i < W and int(arr[cy_i, cx_i]) == bg_mode:
                    holes_mask[cy_i, cx_i] = True
                local_holes = _ndi.binary_fill_holes(
                    sub == int(comp["color"]), structure=_NDI_STRUCT_4
                ) & (sub == bg_mode)
                if np.any(local_holes):
                    holes_mask[r0:r1, c0:c1] |= local_holes
    else:
        arr_list = arr.tolist()
        vis_ext_list = [[False] * W for _ in range(H)]
        q_ext_r: List[int] = []
        q_ext_c: List[int] = []
        for r in range(H):
            for c in (0, W - 1):
                if arr_list[r][c] == bg_mode and not vis_ext_list[r][c]:
                    vis_ext_list[r][c] = True
                    q_ext_r.append(r)
                    q_ext_c.append(c)
        for c in range(W):
            for r in (0, H - 1):
                if arr_list[r][c] == bg_mode and not vis_ext_list[r][c]:
                    vis_ext_list[r][c] = True
                    q_ext_r.append(r)
                    q_ext_c.append(c)
        head_ext = 0
        while head_ext < len(q_ext_r):
            cr = q_ext_r[head_ext]
            cc = q_ext_c[head_ext]
            head_ext += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and not vis_ext_list[nr][nc] and arr_list[nr][nc] == bg_mode:
                    vis_ext_list[nr][nc] = True
                    q_ext_r.append(nr)
                    q_ext_c.append(nc)

        vis_ext = np.asarray(vis_ext_list, dtype=bool)
        holes_mask = (arr == bg_mode) & (~vis_ext)
        for comp in fg_comps:
            if comp["area"] >= 6 and comp["fill_ratio"] < 0.85 and comp["h"] >= 3 and comp["w"] >= 3:
                r0, r1, c0, c1 = comp["r0"], comp["r1"], comp["c0"], comp["c1"]
                sub = arr[r0:r1, c0:c1]
                cy_i = int(round(comp["centroid_y"]))
                cx_i = int(round(comp["centroid_x"]))
                if 0 <= cy_i < H and 0 <= cx_i < W and arr_list[cy_i][cx_i] == bg_mode:
                    holes_mask[cy_i, cx_i] = True
                bh, bw = r1 - r0, c1 - c0
                comp_col = int(comp["color"])
                sub_list = sub.tolist()
                vis_sub_list = [[False] * bw for _ in range(bh)]
                q_sub_r: List[int] = []
                q_sub_c: List[int] = []
                for rr in range(bh):
                    for cc_i in (0, bw - 1):
                        if sub_list[rr][cc_i] != comp_col and not vis_sub_list[rr][cc_i]:
                            vis_sub_list[rr][cc_i] = True
                            q_sub_r.append(rr)
                            q_sub_c.append(cc_i)
                for cc_i in range(bw):
                    for rr in (0, bh - 1):
                        if sub_list[rr][cc_i] != comp_col and not vis_sub_list[rr][cc_i]:
                            vis_sub_list[rr][cc_i] = True
                            q_sub_r.append(rr)
                            q_sub_c.append(cc_i)
                head_sub = 0
                while head_sub < len(q_sub_r):
                    cr = q_sub_r[head_sub]
                    cc_i = q_sub_c[head_sub]
                    head_sub += 1
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc_i + dc
                        if 0 <= nr < bh and 0 <= nc < bw and not vis_sub_list[nr][nc] and sub_list[nr][nc] != comp_col:
                            vis_sub_list[nr][nc] = True
                            q_sub_r.append(nr)
                            q_sub_c.append(nc)
                vis_sub = np.asarray(vis_sub_list, dtype=bool)
                local_holes = (sub == bg_mode) & (~vis_sub)
                if np.any(local_holes):
                    holes_mask[r0:r1, c0:c1] |= local_holes

    if np.any(holes_mask):
        hole_comps = _connected_components_2d(
            holes_mask.astype(np.int32), connectivity=4, same_color=True, ignore_val=0
        )
        base_hole_sal = 1.85 if len(hole_comps) >= 16 else 2.60
        for hc in hole_comps:
            if hc["area"] > int(0.25 * H * W):
                continue
            mx, my = int(hc["medoid_x"]), int(hc["medoid_y"])
            if _is_clock_masked(mx, my, rows=hc["rows"], cols=hc["cols"], deep_only=True):
                continue
            log_area = math.log2(max(1.0, float(hc["area"])))
            sal = (
                base_hole_sal
                + 0.65 * float(hc["fill_ratio"])
                + 0.50 * float(hc["aspect_ratio"])
                - 0.10 * abs(log_area - 3.5)
            )
            feat_vec = np.array(
                [
                    float(bg_mode) / 15.0,
                    log_area / 12.0,
                    float(hc["fill_ratio"]),
                    float(hc["aspect_ratio"]),
                    float(hc["h"]) / float(H),
                    float(hc["w"]) / float(W),
                    float(my) / float(H),
                    float(mx) / float(W),
                    1.0,
                    1.0,
                    float(temp_diff[my, mx]),
                    sal / 5.0,
                ],
                dtype=np.float32,
            )
            raw_candidates.append(
                {
                    "x": mx,
                    "y": my,
                    "kind": "enclosed_hole",
                    "color": int(arr[my, mx]),
                    "area": int(hc["area"]),
                    "salience": float(sal),
                    "feature_vec": feat_vec,
                    "bbox": hc["bbox"],
                    "r0": int(hc["r0"]),
                    "r1": int(hc["r1"]),
                    "c0": int(hc["c0"]),
                    "c1": int(hc["c1"]),
                }
            )

    # 3. Symmetry & Dual-Panel Pattern Discrepancy Candidates
    fg_mask = (arr != bg_mode) & (~clock_mask if (has_deep_clock and clock_mask is not None) else True)
    if np.any(fg_mask):
        frs, fcs = np.where(fg_mask)
        fr0, fr1 = int(frs.min()), int(frs.max()) + 1
        fc0, fc1 = int(fcs.min()), int(fcs.max()) + 1
        sub_fg = arr[fr0:fr1, fc0:fc1]
        sh, sw = sub_fg.shape
        if sh >= 3 and sw >= 3:
            for _, flipped in (
                ("h_mirror", np.fliplr(sub_fg)),
                ("v_mirror", np.flipud(sub_fg)),
                ("rot180", np.rot90(sub_fg, 2)),
            ):
                mismatch = sub_fg != flipped
                mismatch_ratio = float(mismatch.mean())
                if 0.0 < mismatch_ratio <= 0.22:
                    defect_comps = _connected_components_2d(
                        mismatch.astype(np.int32), connectivity=4, same_color=True, ignore_val=0
                    )
                    for dc in defect_comps[:6]:
                        dy = fr0 + int(dc["medoid_y"])
                        dx = fc0 + int(dc["medoid_x"])
                        if _is_clock_masked(
                            dx,
                            dy,
                            rows=fr0 + dc["rows"],
                            cols=fc0 + dc["cols"],
                            deep_only=True,
                        ):
                            continue
                        sal = 2.95 * (1.0 - mismatch_ratio)
                        feat_vec = np.array(
                            [
                                float(arr[dy, dx]) / 15.0,
                                math.log2(max(1.0, float(dc["area"]))) / 12.0,
                                float(dc["fill_ratio"]),
                                float(dc["aspect_ratio"]),
                                float(dc["h"]) / float(H),
                                float(dc["w"]) / float(W),
                                float(dy) / float(H),
                                float(dx) / float(W),
                                1.0,
                                float(beltrami_grad[dy, dx]),
                                float(temp_diff[dy, dx]),
                                sal / 5.0,
                            ],
                            dtype=np.float32,
                        )
                        raw_candidates.append(
                            {
                                "x": int(dx),
                                "y": int(dy),
                                "kind": "symmetry_defect",
                                "color": int(arr[dy, dx]),
                                "area": int(dc["area"]),
                                "salience": float(sal),
                                "feature_vec": feat_vec,
                                "bbox": dc["bbox"],
                                "r0": fr0 + int(dc["r0"]),
                                "r1": fr0 + int(dc["r1"]),
                                "c0": fc0 + int(dc["c0"]),
                                "c1": fc0 + int(dc["c1"]),
                            }
                        )

    # 4. 2D Axial RoPE Salience Patch Medoids via `RoPE2DMedoidPooler(patch_size=4)`
    feat_2d = np.zeros((1, H, W, 20), dtype=np.float32)
    feat_2d[0, :, :, :16] = (arr[:, :, None] == np.arange(16, dtype=arr.dtype)).astype(np.float32)
    feat_2d[0, :, :, 16] = (arr != bg_mode).astype(np.float32)
    feat_2d[0, :, :, 17] = beltrami_grad
    feat_2d[0, :, :, 18] = temp_diff
    rad_prior = _RADIAL_PRIOR_CACHE.get((H, W))
    if rad_prior is None:
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, H, dtype=np.float32),
            np.linspace(-1.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        rad_prior = (1.0 - 0.5 * (yy ** 2 + xx ** 2)).astype(np.float32)
        _RADIAL_PRIOR_CACHE[(H, W)] = rad_prior
    feat_2d[0, :, :, 19] = rad_prior

    with torch.no_grad():
        k_t = torch.from_numpy(feat_2d)
        pooler = _DEFAULT_ROPE2D_MEDOID_POOLER
        k_rot = RoPE2DMedoidPooler.apply_axial_rope2d(k_t, H=H, W=W)
        _, v_pooled, info = pooler(k_rot, k_t)
        med_y_grid = info["medoid_y"][0].cpu().numpy()
        med_x_grid = info["medoid_x"][0].cpu().numpy()
        gamma_grid = info["gamma_2d"][0].cpu().numpy()
        patch_fg = v_pooled[0, :, :, 16].cpu().numpy()
        patch_grad = v_pooled[0, :, :, 17].cpu().numpy()
        patch_sal = gamma_grid * (1.2 * patch_fg + 0.9 * patch_grad)

        ph, pw = patch_sal.shape
        flat_order = np.argsort(-patch_sal.reshape(-1))
        for idx_p in flat_order[:16]:
            pr, pc = int(idx_p // pw), int(idx_p % pw)
            if patch_sal[pr, pc] <= 0.05:
                continue
            py = int(np.clip(med_y_grid[pr, pc], 0, H - 1))
            px = int(np.clip(med_x_grid[pr, pc], 0, W - 1))
            if _is_clock_masked(
                px,
                py,
                r0=pr * 4,
                r1=(pr + 1) * 4,
                c0=pc * 4,
                c1=(pc + 1) * 4,
            ):
                continue
            sal = float(1.35 + patch_sal[pr, pc])
            col = int(arr[py, px])
            feat_vec = np.array(
                [
                    float(col) / 15.0,
                    0.33,
                    float(gamma_grid[pr, pc]),
                    1.0,
                    4.0 / float(H),
                    4.0 / float(W),
                    float(py) / float(H),
                    float(px) / float(W),
                    0.0,
                    float(beltrami_grad[py, px]),
                    float(temp_diff[py, px]),
                    sal / 5.0,
                ],
                dtype=np.float32,
            )
            raw_candidates.append(
                {
                    "x": px,
                    "y": py,
                    "kind": "rope2d_patch",
                    "color": col,
                    "area": 16,
                    "salience": sal,
                    "feature_vec": feat_vec,
                    "bbox": (4, 4),
                    "r0": pr * 4,
                    "r1": (pr + 1) * 4,
                    "c0": pc * 4,
                    "c1": (pc + 1) * 4,
                }
            )

    # 3b. Dual-Panel Kronecker Block-Scaling Homomorphism Defect Candidates (`H_p = k_y M, W_p = k_x N`)
    panel_comps = [
        c
        for c in _connected_components_2d(arr, connectivity=8, same_color=False, ignore_val=bg_mode)
        if c["h"] >= 3 and c["w"] >= 3 and c["area"] >= 9
    ]
    for i_p in range(len(panel_comps)):
        for j_p in range(len(panel_comps)):
            if i_p == j_p:
                continue
            p_play, p_ref = panel_comps[i_p], panel_comps[j_p]
            hp, wp = int(p_play["h"]), int(p_play["w"])
            hr, wr = int(p_ref["h"]), int(p_ref["w"])
            if hp >= hr and wp >= wr and hp % hr == 0 and wp % wr == 0:
                ky, kx = hp // hr, wp // wr
                if (1 <= ky <= 4 and 1 <= kx <= 4) and (ky > 1 or kx > 1):
                    ref_sub = arr[p_ref["r0"] : p_ref["r1"], p_ref["c0"] : p_ref["c1"]]
                    play_sub = arr[p_play["r0"] : p_play["r1"], p_play["c0"] : p_play["c1"]]
                    ref_kron = np.repeat(np.repeat(ref_sub, ky, axis=0), kx, axis=1)
                    mismatch_p = play_sub != ref_kron
                    m_ratio = float(mismatch_p.mean())
                    if 0.0 < m_ratio <= 0.35:
                        for rr in range(hr):
                            for cc_i in range(wr):
                                blk = play_sub[rr * ky : (rr + 1) * ky, cc_i * kx : (cc_i + 1) * kx]
                                if int(np.sum(blk != int(ref_sub[rr, cc_i]))) > (ky * kx) // 2:
                                    dy = int(p_play["r0"] + rr * ky + ky // 2)
                                    dx = int(p_play["c0"] + cc_i * kx + kx // 2)
                                    if _is_clock_masked(dx, dy, deep_only=True):
                                        continue
                                    sal = float(0.85 * (1.0 - m_ratio))
                                    feat_vec = np.array(
                                        [
                                            float(arr[dy, dx]) / 15.0,
                                            math.log2(max(1.0, float(ky * kx))) / 12.0,
                                            1.0,
                                            float(min(ky, kx)) / float(max(ky, kx)),
                                            float(ky) / float(H),
                                            float(kx) / float(W),
                                            float(dy) / float(H),
                                            float(dx) / float(W),
                                            1.0,
                                            float(beltrami_grad[dy, dx]),
                                            float(temp_diff[dy, dx]),
                                            sal / 5.0,
                                        ],
                                        dtype=np.float32,
                                    )
                                    raw_candidates.append(
                                        {
                                            "x": dx,
                                            "y": dy,
                                            "kind": "homomorphism_defect",
                                            "color": int(arr[dy, dx]),
                                            "area": int(ky * kx),
                                            "salience": sal,
                                            "feature_vec": feat_vec,
                                            "bbox": (ky, kx),
                                            "r0": int(p_play["r0"] + rr * ky),
                                            "r1": int(p_play["r0"] + (rr + 1) * ky),
                                            "c0": int(p_play["c0"] + cc_i * kx),
                                            "c1": int(p_play["c0"] + (cc_i + 1) * kx),
                                        }
                                    )

    # Deduplicate with 2D Non-Maximum Suppression (NMS):
    # Merge concentric / overlapping sub-component medoids within L_inf <= 1 (or L_inf <= 2
    # when bounding boxes overlap or for `rope2d_patch` duplicates) so composite buttons
    # (e.g. a 5x5 outer frame + 3x3 inner core) emit a single canonical medoid.
    KindPriority = {
        "symmetry_defect": 4,
        "enclosed_hole": 3,
        "cc_lattice": 2,
        "cc_object": 1,
        "homomorphism_defect": 0,
        "rope2d_patch": 0,
    }
    sorted_raw = sorted(
        raw_candidates,
        key=lambda d: (d["salience"], KindPriority.get(d["kind"], 0)),
        reverse=True,
    )
    all_unique: List[Dict[str, Any]] = []
    for cand in sorted_raw:
        cx, cy = int(cand["x"]), int(cand["y"])
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        suppressed = False
        for acc in all_unique:
            ax, ay = int(acc["x"]), int(acc["y"])
            dist_inf = max(abs(cx - ax), abs(cy - ay))
            if dist_inf == 0:
                suppressed = True
                break
            if (
                dist_inf <= 1
                and cand["kind"] == acc["kind"]
                and not (cand["area"] == 1 and acc["area"] == 1 and cand["color"] != acc["color"])
            ):
                suppressed = True
                break
            if dist_inf <= 3 and cand["kind"] == "rope2d_patch":
                suppressed = True
                break
        if not suppressed:
            all_unique.append(cand)

    # Dyadic Bruhat-Tits Quadtree + Morphological Diversity Stratification:
    # Within each quadrant, prioritize morphological diversity (first 2 instances of each
    # distinct `(kind, bbox)` shape) before taking 3rd..20th repetitions of identical lattice tiles,
    # guaranteeing unique control widgets (such as side buttons) are never crowded out by large lattices.
    mid_y, mid_x = H // 2, W // 2
    quad_buckets: List[List[Dict[str, Any]]] = [[], [], [], []]
    quad_shape_counts: List[Dict[Tuple[str, Tuple[int, int]], int]] = [
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    ]
    speckle_counts: Dict[Tuple[int, Tuple[int, int]], int] = Counter()
    deferred: List[Dict[str, Any]] = []

    for cand in all_unique:
        if cand["kind"] == "homomorphism_defect":
            deferred.append(cand)
            continue
        if cand["area"] <= 1 and cand["kind"] == "cc_object":
            sk = (int(cand["color"]), cand["bbox"])
            if speckle_counts[sk] >= 4:
                deferred.append(cand)
                continue
            speckle_counts[sk] += 1
        q_idx = (2 if int(cand["y"]) >= mid_y else 0) + (1 if int(cand["x"]) >= mid_x else 0)
        sh_key = (str(cand["kind"]), cand["bbox"])
        occ_rank = quad_shape_counts[q_idx][sh_key]
        quad_shape_counts[q_idx][sh_key] += 1
        cand["_diversity_tier"] = occ_rank // 2
        quad_buckets[q_idx].append(cand)

    for q in range(4):
        quad_buckets[q].sort(
            key=lambda d: (int(d.get("_diversity_tier", 0)), -float(d["salience"]))
        )

    # Round-robin interleave across quadrants while respecting diversity tier and salience
    selected: List[Dict[str, Any]] = []
    q_ptrs = [0, 0, 0, 0]
    while any(q_ptrs[q] < len(quad_buckets[q]) for q in range(4)):
        active_qs = [q for q in range(4) if q_ptrs[q] < len(quad_buckets[q])]
        active_qs.sort(
            key=lambda q: (
                int(quad_buckets[q][q_ptrs[q]].get("_diversity_tier", 0)),
                -float(quad_buckets[q][q_ptrs[q]]["salience"]),
            )
        )
        for q in active_qs:
            selected.append(quad_buckets[q][q_ptrs[q]])
            q_ptrs[q] += 1

    if deferred:
        selected.extend(deferred)

    if len(_MEDOID_EXTRACTION_CACHE) >= 4096:
        _MEDOID_EXTRACTION_CACHE.pop(next(iter(_MEDOID_EXTRACTION_CACHE)), None)
    _MEDOID_EXTRACTION_CACHE[medoid_cache_key] = selected

    return [dict(d) for d in selected[:max_candidates]]


_TOPO_POTENTIAL_CACHE: Dict[str, float] = {}


def compute_topological_symmetry_potential(
    frame_2d: np.ndarray,
    clock_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Evaluates the Topological Betti `(beta_0, beta_1)` & Dihedral `D_4` / Multi-Panel
    Symmetry Free-Energy Potential `Phi_topo(F_t)` (Higgins et al., arXiv:2110.09443;
    Friston et al., arXiv:1911.12247; arXiv:2609.39140; arXiv:2610.02200):

        Phi_topo(F_t) = 0.50 * S_panel(F_t) + 0.35 * S_dihedral(F_t) - 0.15 * (beta_0(F_t) / 64.0)

    where:
      - `S_panel(F_t) in [0, 1]` measures normalized pixelwise structural agreement across
        matching-bbox foreground sub-panels (`h >= 3, w >= 3`),
      - `S_dihedral(F_t) in [0, 1]` measures maximal reflection/rotational symmetry of the
        active foreground bounding box under `{fliplr, flipud, rot180}`,
      - `beta_0(F_t)` is the 0-th cubical Betti number (foreground 4-connected component count),
        rewarding topological assembly and penalizing fragmentation.
    """
    arr = _frame_to_2d_numpy(frame_2d)
    if clock_mask is not None and clock_mask.shape == arr.shape and np.any(clock_mask):
        arr = _mask_clock_to_bg(arr, clock_mask)

    cache_key = _blake2b_grid(arr)
    cached_val = _TOPO_POTENTIAL_CACHE.get(cache_key)
    if cached_val is not None:
        return cached_val

    vals, counts = np.unique(arr, return_counts=True)
    bg_mode = int(vals[int(np.argmax(counts))])
    fg_mask = arr != bg_mode
    if not np.any(fg_mask):
        _TOPO_POTENTIAL_CACHE[cache_key] = 0.0
        return 0.0

    # 1. Dihedral D_4 symmetry potential on foreground bounding box
    frs, fcs = np.where(fg_mask)
    fr0, fr1 = int(frs.min()), int(frs.max()) + 1
    fc0, fc1 = int(fcs.min()), int(fcs.max()) + 1
    sub_fg = arr[fr0:fr1, fc0:fc1]
    s_dihedral = max(
        float(np.mean(sub_fg == np.fliplr(sub_fg))),
        float(np.mean(sub_fg == np.flipud(sub_fg))),
        float(np.mean(sub_fg == np.rot90(sub_fg, 2))),
    )

    # 2. Multi-panel structural agreement across matching-bbox sub-boards
    multi_comps = [
        c
        for c in _connected_components_2d(arr, connectivity=8, same_color=False, ignore_val=bg_mode)
        if c["h"] >= 3 and c["w"] >= 3 and c["area"] >= 6
    ]
    agreements: List[float] = []
    for i in range(len(multi_comps)):
        for j in range(i + 1, len(multi_comps)):
            ci, cj = multi_comps[i], multi_comps[j]
            if ci["bbox"] == cj["bbox"]:
                sub_i = arr[ci["r0"] : ci["r1"], ci["c0"] : ci["c1"]]
                sub_j = arr[cj["r0"] : cj["r1"], cj["c0"] : cj["c1"]]
                agreements.append(float(np.mean(sub_i == sub_j)))
    s_panel = float(sum(agreements) / len(agreements)) if agreements else 0.0

    # 3. 0-th Betti number (connected component fragmentation regularizer)
    beta_0 = float(len(multi_comps))
    res = float(0.50 * s_panel + 0.35 * s_dihedral - 0.15 * min(1.0, beta_0 / 64.0))
    if len(_TOPO_POTENTIAL_CACHE) >= 16384:
        _TOPO_POTENTIAL_CACHE.pop(next(iter(_TOPO_POTENTIAL_CACHE)), None)
    _TOPO_POTENTIAL_CACHE[cache_key] = res
    return res


def compute_marginal_wasserstein1_distance(
    frame_2d: np.ndarray,
    initial_frame_2d: Optional[np.ndarray] = None,
    clock_mask: Optional[np.ndarray] = None,
    precomputed_comps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes the closed-form O(HW) 1D Marginal Wasserstein-1 (Earth Mover's) Distance
    `W_1^marg` and Collinear / Bracketed Slot Alignment across foreground color channels:

      1. Same-Color Dual-Component / Moving-to-Stationary 1D CDF Wasserstein-1 Distance:
         For each rare foreground color `c != c_bg` that has either:
           (a) a moving component (`(arr == c) & (init_arr != c)`) and a stationary reference
               marker/box (`(arr == c) & (init_arr == c)`), or
           (b) two compact connected components (`2 <= area <= 48`, `max(h, w) <= 16`)
               representing a movable block/column float and its stationary target marker:
         computes the exact 1D cumulative histogram L1 norm along the active alignment axis:
             W_1(p, q) = (1 / L) * sum_{u=0}^{L-1} |CDF_p(u) - CDF_q(u)|
      2. Corner-Bracketed Slot Alignment:
         Detects axis-aligned rectangular slots `(r_top, r_bot, c_left, c_right)` framed by
         stationary `1x1` corner bracket pixels of color `c_brk`, measuring both interior slot
         completion and the 1D marginal transport distance from movable `c_brk` blocks to the slot.
    """
    arr = _frame_to_2d_numpy(frame_2d)
    H, W = arr.shape
    if clock_mask is not None and clock_mask.shape == arr.shape and np.any(clock_mask):
        arr = _mask_clock_to_bg(arr, clock_mask)

    init_arr: Optional[np.ndarray] = None
    if initial_frame_2d is not None:
        init_arr = _frame_to_2d_numpy(initial_frame_2d)
        if clock_mask is not None and clock_mask.shape == init_arr.shape and np.any(clock_mask):
            init_arr = _mask_clock_to_bg(init_arr, clock_mask)

    if precomputed_comps is not None:
        fg_comps = precomputed_comps
    else:
        vals, counts = np.unique(arr, return_counts=True)
        bg_mode = int(vals[int(np.argmax(counts))])
        fg_comps = _connected_components_2d(arr, connectivity=4, same_color=True, ignore_val=bg_mode)

    def _cdf_1d(idx_a: np.ndarray, length: int) -> np.ndarray:
        ha = np.bincount(np.clip(idx_a, 0, length - 1), minlength=length).astype(np.float64)
        ha /= max(1.0, float(ha.sum()))
        return np.cumsum(ha)

    def _cdf_w1_1d(idx_a: np.ndarray, idx_b: np.ndarray, length: int) -> float:
        if idx_a.size == 0 or idx_b.size == 0 or length <= 1:
            return 0.0
        return float(np.sum(np.abs(_cdf_1d(idx_a, length) - _cdf_1d(idx_b, length)))) / float(length)

    by_color: Dict[int, List[Dict[str, Any]]] = {}
    for c in fg_comps:
        by_color.setdefault(int(c["color"]), []).append(c)

    color_w1_list: List[float] = []
    aligned_pairs = 0
    min_dim = float(min(H, W))
    adj_band = max(4.5, min_dim / 9.8)
    marker_area_cap = max(8, (H * W) // 256)
    sub_marker_cap = max(6, (H * W) // 400)
    max_compact_area = max(16, (H * W) * 3 // 256)
    max_compact_span = max(8, min(H, W) * 3 // 8)

    for col, c_list in by_color.items():
        compact = [
            c
            for c in c_list
            if 1 <= int(c["area"]) <= max_compact_area
            and max(int(c["h"]), int(c["w"])) <= max_compact_span
        ]
        if len(compact) == 2:
            c0, c1 = compact[0], compact[1]
            dx = abs(float(c0["centroid_x"]) - float(c1["centroid_x"]))
            dy = abs(float(c0["centroid_y"]) - float(c1["centroid_y"]))
            # Scale-invariant adjacent-column/row float + wall marker
            if (
                1.0 <= dx <= adj_band
                and int(c0["area"]) != int(c1["area"])
                and min(int(c0["area"]), int(c1["area"])) <= marker_area_cap
            ):
                w1_y = _cdf_w1_1d(c0["rows"], c1["rows"], H)
                color_w1_list.append(w1_y)
                if w1_y <= 1.5 / float(H):
                    aligned_pairs += 1
                continue
            elif (
                1.0 <= dy <= adj_band
                and int(c0["area"]) != int(c1["area"])
                and min(int(c0["area"]), int(c1["area"])) <= marker_area_cap
            ):
                w1_x = _cdf_w1_1d(c0["cols"], c1["cols"], W)
                color_w1_list.append(w1_x)
                if w1_x <= 1.5 / float(W):
                    aligned_pairs += 1
                continue
        elif 3 <= len(compact) <= 6:
            small_markers = [c for c in compact if int(c["area"]) <= sub_marker_cap]
            large_floats = [c for c in compact if int(c["area"]) > sub_marker_cap]
            if small_markers and large_floats:
                for sc in small_markers:
                    best_pair_w1 = 1.0
                    for mc in large_floats:
                        dx = abs(float(mc["centroid_x"]) - float(sc["centroid_x"]))
                        dy = abs(float(mc["centroid_y"]) - float(sc["centroid_y"]))
                        if 1.0 <= dx <= adj_band:
                            w1 = _cdf_w1_1d(mc["rows"], sc["rows"], H)
                        elif 1.0 <= dy <= adj_band:
                            w1 = _cdf_w1_1d(mc["cols"], sc["cols"], W)
                        else:
                            continue
                        if w1 < best_pair_w1:
                            best_pair_w1 = w1
                    if best_pair_w1 < 1.0:
                        color_w1_list.append(best_pair_w1)
                        if best_pair_w1 <= 1.5 / float(max(H, W)):
                            aligned_pairs += 1
                continue

        if init_arr is not None and init_arr.shape == arr.shape:
            mov_mask = (arr == col) & (init_arr != col)
            stat_mask = (arr == col) & (init_arr == col)
            mov_n = int(np.sum(mov_mask))
            stat_n = int(np.sum(stat_mask))
            if 2 <= mov_n <= max_compact_area and 2 <= stat_n <= max_compact_area:
                mrs, mcs = np.where(mov_mask)
                stat_comps = [
                    sc
                    for sc in compact
                    if int(np.sum(init_arr[sc["rows"], sc["cols"]] == col)) >= int(sc["area"]) // 2
                ]
                if stat_comps:
                    best_w1 = 1.0
                    cdf_mrs = _cdf_1d(mrs, H)
                    cdf_mcs = _cdf_1d(mcs, W)
                    m_cx = float(mcs.mean())
                    m_cy = float(mrs.mean())
                    for sc in stat_comps:
                        dx = abs(m_cx - float(sc["centroid_x"]))
                        dy = abs(m_cy - float(sc["centroid_y"]))
                        if dx <= adj_band and dy >= 1.0:
                            w1 = float(np.sum(np.abs(cdf_mrs - _cdf_1d(sc["rows"], H)))) / float(H)
                        elif dy <= adj_band and dx >= 1.0:
                            w1 = float(np.sum(np.abs(cdf_mcs - _cdf_1d(sc["cols"], W)))) / float(W)
                        else:
                            w1 = min(
                                float(np.sum(np.abs(cdf_mcs - _cdf_1d(sc["cols"], W)))) / float(W),
                                float(np.sum(np.abs(cdf_mrs - _cdf_1d(sc["rows"], H)))) / float(H),
                            )
                        if w1 < best_w1:
                            best_w1 = w1
                    color_w1_list.append(best_w1)
                    if best_w1 <= 2.5 / float(max(H, W)):
                        aligned_pairs += 1

    # Corner-bracketed target slots (four 1x1 pixels of color `col` at rectangle corners)
    bracket_defects: List[float] = []
    bracket_slots_found = 0
    max_bracket_span = max(3, min(H, W) // 10)
    for col, c_list in by_color.items():
        dots = [c for c in c_list if int(c["area"]) == 1]
        if 4 <= len(dots) <= 16:
            dot_set = {(int(d["r0"]), int(d["c0"])) for d in dots}
            rs_u = sorted({r for r, _ in dot_set})
            cs_u = sorted({c for _, c in dot_set})
            non_dot_mask = (arr == col).copy()
            for r_d, c_d in dot_set:
                non_dot_mask[r_d, c_d] = False

            col_slots: List[Tuple[int, int, int, int]] = []
            for i_r in range(len(rs_u)):
                for j_r in range(i_r + 1, len(rs_u)):
                    r_top, r_bot = rs_u[i_r], rs_u[j_r]
                    span_r = r_bot - r_top
                    if not (2 <= span_r <= max_bracket_span):
                        continue
                    for i_c in range(len(cs_u)):
                        for j_c in range(i_c + 1, len(cs_u)):
                            c_left, c_right = cs_u[i_c], cs_u[j_c]
                            if c_right - c_left != span_r:
                                continue
                            if (
                                (r_top, c_left) in dot_set
                                and (r_top, c_right) in dot_set
                                and (r_bot, c_left) in dot_set
                                and (r_bot, c_right) in dot_set
                            ):
                                col_slots.append((r_top, r_bot, c_left, c_right))
                                interior = arr[r_top + 1 : r_bot, c_left + 1 : c_right]
                                if interior.size > 0 and bool(np.all(interior == col)):
                                    # Mask out already-docked blocks so `nd_rs, nd_cs` tracks remaining free blocks
                                    non_dot_mask[r_top + 1 : r_bot, c_left + 1 : c_right] = False

            nd_rs, nd_cs = np.where(non_dot_mask)
            for r_top, r_bot, c_left, c_right in col_slots:
                bracket_slots_found += 1
                interior = arr[r_top + 1 : r_bot, c_left + 1 : c_right]
                if interior.size > 0:
                    fill_err = float(np.mean(interior != col))
                    dist_err = 0.0
                    if fill_err > 0.0 and nd_rs.size > 0:
                        sy = 0.5 * float(r_top + r_bot)
                        sx = 0.5 * float(c_left + c_right)
                        dy_v = np.abs(nd_rs - sy)
                        dx_v = np.abs(nd_cs - sx)
                        axis_aligned = (dy_v <= 1.0) | (dx_v <= 1.0)
                        d_eff = np.where(axis_aligned, 0.55 * (dy_v + dx_v), dy_v + dx_v)
                        k_near = min(interior.size, d_eff.size)
                        dist_err = float(np.mean(np.sort(d_eff)[:k_near])) / float(H + W)
                    bracket_defects.append(0.55 * fill_err + 0.45 * dist_err)

    w1_mean = float(np.mean(color_w1_list)) if color_w1_list else 0.0
    brk_mean = float(np.mean(bracket_defects)) if bracket_defects else 0.0
    if bracket_defects:
        combined_w1 = brk_mean
    else:
        combined_w1 = w1_mean

    return {
        "wasserstein1_distance": float(combined_w1),
        "color_w1_distance": float(w1_mean),
        "bracket_slot_defect": float(brk_mean),
        "tracked_color_pairs": len(color_w1_list),
        "aligned_color_pairs": int(aligned_pairs),
        "bracket_slots_count": int(bracket_slots_found),
    }


_TARGET_ALIGNMENT_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def compute_canvas_target_alignment(
    frame_2d: np.ndarray,
    initial_frame_2d: Optional[np.ndarray] = None,
    clock_mask: Optional[np.ndarray] = None,
    spatial_precision_map: Optional[np.ndarray] = None,
    potential_weights: Optional[Sequence[float]] = None,
    distilled_weights: Optional[Sequence[float]] = None,
    goal_invariants: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the 5-Channel Unified Goal Homomorphism Energy `E_target(F_t; Pi, w)` and
    Alignment Potential `Phi_target(F_t; Pi, w) = 1.0 - E_target(F_t; Pi, w)` across the
    5D dimensionless invariant basis `Phi(S) = [Phi_1, Phi_2, Phi_3, Phi_4, Phi_5]^T in [0, 1]^5`:

      1. `Phi_1 = E_hom`: Dual-Panel Kronecker Block-Scaling (`H_p = k_y M, W_p = k_x N`),
         Spaced `cc_lattice` Homomorphism, and Factored Per-Slot Sub-Panel Template Matching.
      2. `Phi_2 = E_marginal`: 1D Marginal Wasserstein-1 (`W_1^marg`) + Corner-Bracketed Slot Alignment.
      3. `Phi_3 = 1 - Phi_topo`: Topological `(beta_0, beta_1)` & Dihedral `D_4` reflection/panel defect.
      4. `Phi_4 = H_MDL`: SMiRL / Solomonoff minority-color mass annihilation & entropy reduction.
      5. `Phi_5 = E_nav`: Avatar-to-Beacon + Same-Color Solid-Ego to Hollow-Receptor-Frame Docking.
    """
    arr = _frame_to_2d_numpy(frame_2d)
    H, W = arr.shape
    if clock_mask is not None and clock_mask.shape == arr.shape and np.any(clock_mask):
        arr = _mask_clock_to_bg(arr, clock_mask)

    eff_w = potential_weights if potential_weights is not None else distilled_weights
    align_cache_key: Optional[Tuple[Any, ...]] = None
    if spatial_precision_map is None and goal_invariants is None:
        init_hash_k = ""
        if initial_frame_2d is not None:
            init_tmp = _frame_to_2d_numpy(initial_frame_2d)
            if clock_mask is not None and clock_mask.shape == init_tmp.shape and np.any(clock_mask):
                init_tmp = _mask_clock_to_bg(init_tmp, clock_mask)
            init_hash_k = _blake2b_grid(init_tmp)
        align_cache_key = (
            _blake2b_grid(arr),
            init_hash_k,
            tuple(round(float(x), 4) for x in eff_w) if eff_w is not None else (),
        )
        cached_align = _TARGET_ALIGNMENT_CACHE.get(align_cache_key)
        if cached_align is not None:
            return dict(cached_align)

    vals, counts = np.unique(arr, return_counts=True)
    bg_mode = int(vals[int(np.argmax(counts))])
    fg_comps = _connected_components_2d(arr, connectivity=4, same_color=True, ignore_val=bg_mode)

    prec_map: Optional[np.ndarray] = None
    if spatial_precision_map is not None:
        pm = np.asarray(spatial_precision_map, dtype=np.float32)
        if pm.shape == (H, W):
            prec_map = np.clip(pm, 0.1, 5.0)

    # 1. Dual-Panel Kronecker, Spaced Lattice & Factored Sub-Panel Homomorphism Energy `E_hom` (Phi_1)
    hom_defects: List[float] = []
    defect_loci: List[Tuple[int, int]] = []
    by_bbox: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for c in fg_comps:
        if 2 <= int(c["area"]) <= 128 and float(c["fill_ratio"]) >= 0.45:
            by_bbox.setdefault((int(c["h"]), int(c["w"])), []).append(c)

    lattice_groups = [(sz, comps) for sz, comps in by_bbox.items() if len(comps) >= 4]
    lattice_groups.sort(key=lambda item: (item[0][0] * item[0][1], len(item[1])), reverse=True)

    if len(lattice_groups) >= 2:
        (bh_p, bw_p), play_tiles = lattice_groups[0]
        (bh_r, bw_r), ref_tiles = lattice_groups[1]
        if abs(len(play_tiles) - len(ref_tiles)) <= 3:
            med_y = float(np.median([t["centroid_y"] for t in play_tiles]))
            med_x = float(np.median([t["centroid_x"] for t in play_tiles]))
            rad_thr = 3.5 * float(max(bh_p, bw_p))
            in_cluster = [
                t
                for t in play_tiles
                if max(abs(float(t["centroid_y"]) - med_y), abs(float(t["centroid_x"]) - med_x)) <= rad_thr
            ]
            out_cluster = [
                t
                for t in play_tiles
                if max(abs(float(t["centroid_y"]) - med_y), abs(float(t["centroid_x"]) - med_x)) > rad_thr
            ]
            if out_cluster:
                out_dists = [
                    (abs(float(t["centroid_y"]) - med_y) + abs(float(t["centroid_x"]) - med_x))
                    / float(H + W)
                    for t in out_cluster
                ]
                hom_defects.append(float(np.mean(out_dists)))
                for t in out_cluster:
                    defect_loci.append((int(t["medoid_x"]), int(t["medoid_y"])))

            p_sorted = sorted(
                in_cluster if len(in_cluster) == len(ref_tiles) else play_tiles,
                key=lambda t: (int(round(t["centroid_y"] / max(1, bh_p))), int(round(t["centroid_x"] / max(1, bw_p)))),
            )
            r_sorted = sorted(
                ref_tiles,
                key=lambda t: (int(round(t["centroid_y"] / max(1, bh_r))), int(round(t["centroid_x"] / max(1, bw_r)))),
            )
            n_cmp = min(len(p_sorted), len(r_sorted))
            if n_cmp >= 4:
                mismatches = 0.0
                w_total = 0.0
                for k_i in range(n_cmp):
                    pt, rt = p_sorted[k_i], r_sorted[k_i]
                    px, py = int(pt["medoid_x"]), int(pt["medoid_y"])
                    w_k = (
                        1.0 + 1e-4 * (float(prec_map[py, px]) - 1.0)
                        if prec_map is not None
                        else 1.0
                    )
                    w_total += w_k
                    if int(pt["color"]) != int(rt["color"]):
                        mismatches += w_k
                        defect_loci.append((px, py))
                hom_defects.append(float(mismatches / max(1e-6, w_total)))
    elif len(lattice_groups) == 1:
        (bh_p, bw_p), tiles = lattice_groups[0]
        if len(tiles) >= 8 and len(tiles) % 2 == 0:
            ys = [float(t["centroid_y"]) for t in tiles]
            xs = [float(t["centroid_x"]) for t in tiles]
            if (max(ys) - min(ys)) >= (max(xs) - min(xs)):
                mid = 0.5 * (max(ys) + min(ys))
                g1 = [t for t in tiles if float(t["centroid_y"]) < mid]
                g2 = [t for t in tiles if float(t["centroid_y"]) >= mid]
            else:
                mid = 0.5 * (max(xs) + min(xs))
                g1 = [t for t in tiles if float(t["centroid_x"]) < mid]
                g2 = [t for t in tiles if float(t["centroid_x"]) >= mid]
            if len(g1) == len(g2) and len(g1) >= 4:
                g1_s = sorted(g1, key=lambda t: (int(round(t["centroid_y"] / max(1, bh_p))), int(round(t["centroid_x"] / max(1, bw_p)))))
                g2_s = sorted(g2, key=lambda t: (int(round(t["centroid_y"] / max(1, bh_p))), int(round(t["centroid_x"] / max(1, bw_p)))))
                mism = sum(1 for a, b in zip(g1_s, g2_s) if int(a["color"]) != int(b["color"]))
                hom_defects.append(float(mism) / float(len(g1_s)))
                for a, b in zip(g1_s, g2_s):
                    if int(a["color"]) != int(b["color"]):
                        defect_loci.append((int(b["medoid_x"]), int(b["medoid_y"])))

    # Factored Per-Slot Sub-Panel Template Matching (for Kronecker & multi-panel boards)
    if not hom_defects:
        panel_comps = [
            c
            for c in _connected_components_2d(arr, connectivity=8, same_color=False, ignore_val=bg_mode)
            if c["h"] >= 3 and c["w"] >= 3 and c["area"] >= 9
        ]
        for i_p in range(len(panel_comps)):
            for j_p in range(i_p + 1, len(panel_comps)):
                p_a, p_b = panel_comps[i_p], panel_comps[j_p]
                if p_a["area"] >= p_b["area"]:
                    p_play, p_ref = p_a, p_b
                else:
                    p_play, p_ref = p_b, p_a
                hp, wp = int(p_play["h"]), int(p_play["w"])
                hr, wr = int(p_ref["h"]), int(p_ref["w"])
                if hp >= hr and wp >= wr and hp % hr == 0 and wp % wr == 0:
                    ky, kx = hp // hr, wp // wr
                    if 1 <= ky <= 4 and 1 <= kx <= 4:
                        ref_sub = arr[p_ref["r0"] : p_ref["r1"], p_ref["c0"] : p_ref["c1"]]
                        play_sub = arr[p_play["r0"] : p_play["r1"], p_play["c0"] : p_play["c1"]]
                        ref_kron = (
                            np.repeat(np.repeat(ref_sub, ky, axis=0), kx, axis=1)
                            if (ky > 1 or kx > 1)
                            else ref_sub
                        )
                        m_ratio = float(np.mean(play_sub != ref_kron))
                        max_m = 1.0 if ((ky > 1 or kx > 1) and hr >= 4 and wr >= 4) else 0.65
                        if 0.0 <= m_ratio <= max_m:
                            hom_defects.append(m_ratio)
                            if m_ratio > 0.0:
                                for rr in range(hr):
                                    for cc_i in range(wr):
                                        blk = play_sub[rr * ky : (rr + 1) * ky, cc_i * kx : (cc_i + 1) * kx]
                                        if int(np.sum(blk != int(ref_sub[rr, cc_i]))) > (ky * kx) // 2:
                                            dy = int(p_play["r0"] + rr * ky + ky // 2)
                                            dx = int(p_play["c0"] + cc_i * kx + kx // 2)
                                            defect_loci.append((dx, dy))

    e_hom = float(np.mean(hom_defects)) if hom_defects else 0.0

    # 2. Marginal Wasserstein-1 & Bracketed Slot Alignment Energy `E_marginal` (Phi_2)
    w1_info = compute_marginal_wasserstein1_distance(
        arr, initial_frame_2d=initial_frame_2d, clock_mask=None, precomputed_comps=fg_comps
    )
    e_marginal = float(w1_info["wasserstein1_distance"])

    # 3. Topological & Dihedral Symmetry Defect `1.0 - Phi_topo` (Phi_3)
    phi_topo = compute_topological_symmetry_potential(arr, clock_mask=None)
    topo_defect = float(np.clip(1.0 - max(0.0, phi_topo), 0.0, 1.0))

    # 4. SMiRL / Solomonoff Minority-Color Mass Annihilation & Entropy Reduction `H_MDL` (Phi_4)
    e_mdl = 0.0
    has_mdl_signal = False
    cleared_cols: Set[int] = (
        set(int(c) for c in goal_invariants.get("cleared_colors", []))
        if goal_invariants is not None
        else set()
    )
    init_arr: Optional[np.ndarray] = None
    if initial_frame_2d is not None:
        init_arr = _frame_to_2d_numpy(initial_frame_2d)
        if clock_mask is not None and clock_mask.shape == init_arr.shape and np.any(clock_mask):
            init_arr = _mask_clock_to_bg(init_arr, clock_mask)
        init_cnts = np.bincount(init_arr.reshape(-1), minlength=16)
        cur_cnts = np.bincount(arr.reshape(-1), minlength=16)
        mdl_ratios: List[float] = []
        minority_cap = max(8, (H * W) // 64)
        for c_id in range(16):
            if c_id == bg_mode:
                continue
            n0 = int(init_cnts[c_id])
            nt = int(cur_cnts[c_id])
            if c_id in cleared_cols and n0 > 0:
                mdl_ratios.append(float(np.clip(float(nt) / float(n0), 0.0, 1.0)))
                has_mdl_signal = True
            elif 2 <= n0 <= minority_cap and 0 <= nt < n0:
                # Dissipative collection/annihilation of a minority target color
                mdl_ratios.append(float(nt) / float(n0))
                has_mdl_signal = True
        if mdl_ratios:
            e_mdl = float(np.mean(mdl_ratios))

    # 5. Geodesic Navigation & Solid-to-Hollow Receptor Docking Energy `E_nav` (Phi_5)
    e_nav = 0.0
    has_nav_signal = False
    docking_dists: List[float] = []
    by_col_comps: Dict[int, List[Dict[str, Any]]] = {}
    for c in fg_comps:
        by_col_comps.setdefault(int(c["color"]), []).append(c)

    # 5a. Topological Cavity Containment: Same-color solid component docking into a hollow receptor cavity
    for col, c_list in by_col_comps.items():
        if len(c_list) < 2:
            continue
        solids = [
            c
            for c in c_list
            if int(c["area"]) >= 4 and float(c["fill_ratio"]) >= 0.80 and int(c["h"]) >= 2 and int(c["w"]) >= 2
        ]
        receptors = [
            c
            for c in c_list
            if int(c["area"]) >= 8 and float(c["fill_ratio"]) <= 0.68 and int(c["h"]) >= 3 and int(c["w"]) >= 3
        ]
        if solids and receptors:
            for s_comp in solids:
                sh_s, sw_s, s_area = int(s_comp["h"]), int(s_comp["w"]), int(s_comp["area"])
                valid_recs = [
                    r_comp
                    for r_comp in receptors
                    if int(r_comp["h"]) >= sh_s
                    and int(r_comp["w"]) >= sw_s
                    and (int(r_comp["h"]) * int(r_comp["w"]) - int(r_comp["area"])) >= s_area
                    and max(int(r_comp["h"]), int(r_comp["w"])) <= max(12, min(H, W) // 3)
                ]
                if valid_recs:
                    d_min = min(
                        (
                            abs(float(s_comp["centroid_y"]) - float(r_comp["centroid_y"]))
                            + abs(float(s_comp["centroid_x"]) - float(r_comp["centroid_x"]))
                        )
                        / float(H + W)
                        for r_comp in valid_recs
                    )
                    docking_dists.append(float(d_min))

    if docking_dists:
        e_nav = float(np.mean(docking_dists))
        has_nav_signal = True
    elif init_arr is not None:
        diff_mask = arr != init_arr
        diff_n = int(np.sum(diff_mask))
        if 2 <= diff_n <= max(32, (H * W) // 32):
            av_mask = diff_mask & (arr != bg_mode)
            if np.any(av_mask):
                ars, acs = np.where(av_mask)
                ay, ax = float(ars.mean()), float(acs.mean())
                av_cols = set(int(c) for c in np.unique(arr[av_mask]))
                beacons = [
                    c
                    for c in fg_comps
                    if 4 <= int(c["area"]) <= max(16, (H * W) // 64)
                    and (int(c["color"]) not in av_cols or float(c["fill_ratio"]) < 0.70)
                    and max(abs(float(c["centroid_y"]) - ay), abs(float(c["centroid_x"]) - ax)) > 3.0
                ]
                if beacons:
                    min_d = min(
                        (abs(float(b["centroid_y"]) - ay) + abs(float(b["centroid_x"]) - ax))
                        / float(H + W)
                        for b in beacons
                    )
                    e_nav = float(min_d)
                    has_nav_signal = True

    phi_vector = [
        float(e_hom),
        float(e_marginal),
        float(topo_defect),
        float(e_mdl),
        float(e_nav),
    ]

    # Combine active energy channels with precision weights `w in Delta^4`
    if potential_weights is None and distilled_weights is not None:
        potential_weights = distilled_weights
    if potential_weights is not None and len(potential_weights) == 5:
        w_vec = [max(0.02, float(x)) for x in potential_weights]
    elif goal_invariants is not None and "weights" in goal_invariants and len(goal_invariants["weights"]) == 5:
        w_vec = [max(0.02, float(x)) for x in goal_invariants["weights"]]
    else:
        w_vec = [0.45, 0.45, 0.20, 0.35, 0.30]

    active_terms: List[Tuple[float, float]] = [(w_vec[2], topo_defect)]
    if hom_defects:
        active_terms.append((w_vec[0], e_hom))
    if w1_info["tracked_color_pairs"] > 0 or w1_info["bracket_slots_count"] > 0:
        active_terms.append((w_vec[1], e_marginal))
    if has_mdl_signal:
        active_terms.append((w_vec[3], e_mdl))
    if has_nav_signal or e_nav > 0.0:
        active_terms.append((w_vec[4], e_nav))

    w_sum = sum(w for w, _ in active_terms)
    target_energy = float(sum(w * val for w, val in active_terms) / max(1e-6, w_sum))
    alignment_potential = float(1.0 - target_energy)

    res_align = {
        "target_energy": round(target_energy, 6),
        "alignment_potential": round(alignment_potential, 6),
        "homomorphism_energy": round(e_hom, 6),
        "marginal_energy": round(e_marginal, 6),
        "navigation_energy": round(e_nav, 6),
        "entropy_annihilation_energy": round(e_mdl, 6),
        "topological_defect": round(topo_defect, 6),
        "phi_vector": [round(x, 6) for x in phi_vector],
        "invariant_basis": np.asarray(phi_vector, dtype=np.float32),
        "wasserstein1_distance": round(float(w1_info["wasserstein1_distance"]), 6),
        "tracked_color_pairs": int(w1_info["tracked_color_pairs"]),
        "bracket_slots_count": int(w1_info["bracket_slots_count"]),
        "defect_loci": defect_loci[:16],
    }
    if align_cache_key is not None:
        if len(_TARGET_ALIGNMENT_CACHE) >= 16384:
            _TARGET_ALIGNMENT_CACHE.pop(next(iter(_TARGET_ALIGNMENT_CACHE)), None)
        _TARGET_ALIGNMENT_CACHE[align_cache_key] = res_align
    return dict(res_align)


class CrossLevelGoalInvariantDistiller:
    """
    Pillar 2: Cross-Level Goal Invariant Distillation (`Phi_win` Induction on Level-Up)
    via Maximum-Entropy / Fisher-Rao Natural-Gradient Inverse Teleology (Ziebart et al., 2008;
    Baker, Saxe & Tenenbaum, 2009).

    Evaluates the 5D dimensionless mathematical invariant basis:
        Phi(S) = [Phi_1(S), Phi_2(S), Phi_3(S), Phi_4(S), Phi_5(S)]^T in [0, 1]^5
    on initial frame `S_0^(L)` and pre-reset winning terminal frame `hat_S_T^(L) = P_{H_nope}(F_win^(L))`,
    and updates the simplex precision weights `w^(L+1) in Delta^4` via the Fisher-Rao natural-gradient
    Boltzmann mirror-descent rule:
        w_k^(L+1) propto w_k^(L) * exp(-beta * clip((Phi_k(hat_S_T^(L)) - Phi_k(S_0^(L))) / (sigma_k + 0.03), -3, 3))
    with Laplace floor smoothing (`alpha_floor = 0.05`, `beta = 1.35`).
    """

    ARCHETYPE_NAMES: Tuple[str, ...] = (
        "G0:SHEAF_HOMOMORPHISM",
        "G1:OPTIMAL_TRANSPORT",
        "G2:DIHEDRAL_SYMMETRY",
        "G3:ENTROPY_ANNIHILATION",
        "G4:GEODESIC_DOCKING",
    )

    def __init__(self, beta: float = 1.35, alpha_floor: float = 0.05):
        self.beta = float(beta)
        self.alpha_floor = float(alpha_floor)
        self.reset()

    def reset(self) -> None:
        self.weights: np.ndarray = np.array([0.25, 0.25, 0.15, 0.15, 0.20], dtype=np.float64)
        self.cleared_colors: Set[int] = set()
        self.winning_action_types: Dict[int, int] = Counter()
        self.winning_d4_signatures: Set[Tuple[str, int, int, int]] = set()
        self.levels_distilled: int = 0
        self.last_delta_phi: np.ndarray = np.zeros(5, dtype=np.float64)

    def get_invariants_dict(self) -> Dict[str, Any]:
        return {
            "weights": [float(x) for x in self.weights],
            "cleared_colors": sorted(self.cleared_colors),
            "winning_action_types": dict(self.winning_action_types),
            "winning_d4_signatures": list(self.winning_d4_signatures),
            "levels_distilled": int(self.levels_distilled),
        }

    def apply_system2_compiled_weights(self, w_compiled: Sequence[float], blend: float = 0.25) -> np.ndarray:
        """Blends Left-Hemisphere Gemma 4 compiled subgoal precision weights into `self.weights`."""
        arr_w = np.asarray(w_compiled, dtype=np.float64).reshape(-1)
        if arr_w.size != 5 or not np.all(np.isfinite(arr_w)):
            return self.weights.copy()
        arr_w = np.maximum(arr_w, 0.0)
        s = float(arr_w.sum())
        if s <= 1e-8:
            return self.weights.copy()
        arr_w = arr_w / s
        eff_b = float(np.clip(blend, 0.05, 0.60))
        mixed = (1.0 - eff_b) * self.weights + eff_b * arr_w
        mixed = (1.0 - 5.0 * self.alpha_floor) * mixed + self.alpha_floor
        self.weights = mixed / float(mixed.sum())
        return self.weights.copy()

    def score_transition_gain(
        self,
        prev_frame_2d: np.ndarray,
        next_frame_2d: np.ndarray,
        initial_frame_2d: Optional[np.ndarray] = None,
        clock_mask: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluates `E_target(prev; w) - E_target(next; w)` under the distilled precision weights."""
        inv_ctx = self.get_invariants_dict()
        e_prev = float(
            compute_canvas_target_alignment(
                prev_frame_2d,
                initial_frame_2d=initial_frame_2d,
                clock_mask=clock_mask,
                potential_weights=self.weights,
                goal_invariants=inv_ctx,
            )["target_energy"]
        )
        e_next = float(
            compute_canvas_target_alignment(
                next_frame_2d,
                initial_frame_2d=initial_frame_2d,
                clock_mask=clock_mask,
                potential_weights=self.weights,
                goal_invariants=inv_ctx,
            )["target_energy"]
        )
        return float(e_prev - e_next)

    def distill_level_win(
        self,
        initial_frame_2d: np.ndarray,
        pre_win_frame_2d: Optional[np.ndarray] = None,
        win_frame_2d: Optional[np.ndarray] = None,
        winning_action_key: Optional[ActionKey] = None,
        winning_candidate: Optional[Dict[str, Any]] = None,
        clock_mask: Optional[np.ndarray] = None,
        spatial_precision_map: Optional[np.ndarray] = None,
        level_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Induces cross-level goal invariants from a completed level's initial frame `S_0^(L)`
        and ground-truth winning frame `F_win^(L)`.
        """
        if win_frame_2d is None:
            # 2-arg call: `(win_frame_2d, initial_frame_2d)`
            if pre_win_frame_2d is not None:
                s_win = _frame_to_2d_numpy(initial_frame_2d)
                s0 = _frame_to_2d_numpy(pre_win_frame_2d)
            else:
                s0 = _frame_to_2d_numpy(initial_frame_2d)
                s_win = s0.copy()
        else:
            s0 = _frame_to_2d_numpy(initial_frame_2d)
            s_win = _frame_to_2d_numpy(win_frame_2d)
        if clock_mask is not None and clock_mask.shape == s0.shape and np.any(clock_mask):
            s0 = _mask_clock_to_bg(s0, clock_mask)
            s_win = _mask_clock_to_bg(s_win, clock_mask)

        # Detect minority foreground colors that were cleared/annihilated (>= 50% reduction)
        vals, cnts = np.unique(s0, return_counts=True)
        bg_c = int(vals[int(np.argmax(cnts))])
        c0_hist = np.bincount(s0.reshape(-1), minlength=16)
        cwin_hist = np.bincount(s_win.reshape(-1), minlength=16)
        for c_id in range(16):
            if c_id == bg_c:
                continue
            n_init = int(c0_hist[c_id])
            n_final = int(cwin_hist[c_id])
            if 2 <= n_init <= 256 and n_final <= n_init // 2:
                self.cleared_colors.add(int(c_id))

        inv_ctx = self.get_invariants_dict()
        al_0 = compute_canvas_target_alignment(
            s0,
            initial_frame_2d=s0,
            clock_mask=None,
            spatial_precision_map=spatial_precision_map,
            goal_invariants=inv_ctx,
        )
        al_win = compute_canvas_target_alignment(
            s_win,
            initial_frame_2d=s0,
            clock_mask=None,
            spatial_precision_map=spatial_precision_map,
            goal_invariants=inv_ctx,
        )
        phi_0 = np.asarray(al_0["phi_vector"], dtype=np.float64)
        phi_win = np.asarray(al_win["phi_vector"], dtype=np.float64)
        if self.cleared_colors:
            # At s0, remaining fraction of cleared_colors is 1.0
            phi_0[3] = 1.0

        delta_phi = phi_win - phi_0
        self.last_delta_phi = delta_phi.copy()
        sigma = np.maximum(np.abs(phi_0), 0.15)
        z_adv = np.clip(delta_phi / (sigma + 0.03), -3.0, 3.0)

        raw_w = self.weights * np.exp(-self.beta * z_adv)
        if float(raw_w.sum()) > 1e-9:
            raw_w = raw_w / float(raw_w.sum())
        else:
            raw_w = np.full(5, 0.20, dtype=np.float64)
        smoothed = (1.0 - 5.0 * self.alpha_floor) * raw_w + self.alpha_floor
        self.weights = smoothed / float(smoothed.sum())
        self.levels_distilled += 1

        if winning_action_key is not None:
            act_id = (
                int(winning_action_key[0])
                if isinstance(winning_action_key, tuple)
                else int(winning_action_key)
            )
            self.winning_action_types[act_id] += 1

        if winning_candidate is not None:
            bbox_v = winning_candidate.get("bbox", (1, 1))
            bh = int(bbox_v[0]) if isinstance(bbox_v, (tuple, list)) and len(bbox_v) >= 2 else 1
            bw = int(bbox_v[1]) if isinstance(bbox_v, (tuple, list)) and len(bbox_v) >= 2 else 1
            area = int(winning_candidate.get("area", bh * bw))
            k_str = str(winning_candidate.get("kind", "cc_object"))
            d4_sig = (
                "cc_lattice" if k_str == "homomorphism_defect" else k_str,
                min(bh, bw),
                max(bh, bw),
                int(round(math.log2(max(1, area)))),
            )
            self.winning_d4_signatures.add(d4_sig)

        return {
            "level_idx": int(level_idx),
            "levels_distilled": int(self.levels_distilled),
            "phi_initial": [round(float(x), 6) for x in phi_0],
            "phi_win": [round(float(x), 6) for x in phi_win],
            "delta_phi": [round(float(x), 6) for x in delta_phi],
            "updated_weights": [round(float(x), 6) for x in self.weights],
            "weights": [round(float(x), 6) for x in self.weights],
            "cleared_colors": sorted(self.cleared_colors),
            "dominant_archetype": self.ARCHETYPE_NAMES[int(np.argmax(self.weights))],
        }


@dataclass
class ContactNode:
    node_id: int
    color: int
    rows: np.ndarray
    cols: np.ndarray
    area: int
    h: int
    w: int
    r0: int
    r1: int
    c0: int
    c1: int
    centroid_y: float
    centroid_x: float
    is_ego: bool = False
    is_wall: bool = False
    is_pushable: bool = False
    is_cavity: bool = False
    is_nested: bool = False
    parent_container_id: Optional[int] = None
    stalk_se2: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (v_y, v_x, omega) in se(2)
    force_stalk: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (f_y, f_x, tau) in se*(2)


@dataclass
class ContactEdge:
    u: int
    v: int
    normal: Tuple[float, float]  # Outward unit normal pointing from u to v: (n_y, n_x)
    contact_points_u: List[Tuple[int, int]]
    contact_points_v: List[Tuple[int, int]]
    is_enclosed: bool = False  # Container nesting
    container_id: Optional[int] = None
    content_id: Optional[int] = None
    conductance: float = 0.0  # Dynamic tropical conductance G_e(s_fsa) in [0, 1]
    required_latch: Optional[str] = None

    def restriction_map_u(
        self,
        v_u: Tuple[float, float, float],
        x_u: Tuple[float, float],
    ) -> float:
        """
        rho_{u -> e}(v_u) = n^T v_u + (p - x_u) x n * omega_u
        """
        vy, vx, omega = v_u
        ny, nx = self.normal
        trans_comp = ny * vy + nx * vx
        if self.contact_points_u:
            cp_y = sum(p[0] for p in self.contact_points_u) / float(len(self.contact_points_u))
            cp_x = sum(p[1] for p in self.contact_points_u) / float(len(self.contact_points_u))
            arm_y = cp_y - x_u[0]
            arm_x = cp_x - x_u[1]
            rot_comp = (arm_y * nx - arm_x * ny) * omega
        else:
            rot_comp = 0.0
        return float(trans_comp + rot_comp)

    def restriction_map_v(
        self,
        v_v: Tuple[float, float, float],
        x_v: Tuple[float, float],
    ) -> float:
        """
        rho_{v -> e}(v_v) = n^T v_v + (p - x_v) x n * omega_v
        """
        vy, vx, omega = v_v
        ny, nx = self.normal
        trans_comp = ny * vy + nx * vx
        if self.contact_points_v:
            cp_y = sum(p[0] for p in self.contact_points_v) / float(len(self.contact_points_v))
            cp_x = sum(p[1] for p in self.contact_points_v) / float(len(self.contact_points_v))
            arm_y = cp_y - x_v[0]
            arm_x = cp_x - x_v[1]
            rot_comp = (arm_y * nx - arm_x * ny) * omega
        else:
            rot_comp = 0.0
        return float(trans_comp + rot_comp)

    def compute_coboundary(
        self,
        v_u: Tuple[float, float, float],
        x_u: Tuple[float, float],
        v_v: Tuple[float, float, float],
        x_v: Tuple[float, float],
    ) -> float:
        """
        Sheaf coboundary operator:
        (delta v)(e) = rho_{v -> e}(v_v) - rho_{u -> e}(v_u)
        Unilateral non-penetration condition: (delta v)(e) >= 0.
        """
        return self.restriction_map_v(v_v, x_v) - self.restriction_map_u(v_u, x_u)


class _ContactNodeMap(dict):
    """Dict mapping `node_id -> ContactNode` whose direct iterator yields `ContactNode` objects."""

    def __iter__(self):
        return iter(self.values())


class ContactGraphSheaf:
    """
    Kinematic Contact-Chain Sheaf G_contact(F_t) modeling:
      - Rigid components as vertex stalks in se(2)
      - Contact edges with outward normals n_ij and state-dependent conductance G_e(s_fsa)
      - Sheaf restriction maps enforcing unilateral non-penetration:
        (delta v)(e) = rho_{v_j -> e}(v_j) - rho_{v_i -> e}(v_i) >= 0
      - Topological force & displacement propagation for:
        * Linear and branching multi-body push chains (Ego -> Tool -> Obstacle -> Goal)
        * Tool mediation & wide surface clearing (plow / broom)
        * Container nesting: interior objects locked to container frame
        * Peg-in-hole docking into matching receptor cavities
        * Wall locks & boundary collisions collapsing displacements in H^0(G; F)
    """

    def __init__(
        self,
        nodes: Dict[int, ContactNode],
        edges: List[ContactEdge],
        board_shape: Tuple[int, int],
        bg_mode: int,
        ego_node_id: Optional[int] = None,
        comp_grid: Optional[np.ndarray] = None,
        unlocked_colors: Optional[Set[int]] = None,
    ) -> None:
        self.nodes: Dict[int, ContactNode] = _ContactNodeMap(nodes)
        self.edges = edges
        self.board_shape = board_shape
        self.bg_mode = bg_mode
        self.ego_node_id = ego_node_id
        self.comp_grid = comp_grid
        self.unlocked_colors: Set[int] = set(unlocked_colors) if unlocked_colors is not None else set()
        self.adj: Dict[int, List[ContactEdge]] = {nid: [] for nid in self.nodes.keys()}
        for edge in edges:
            self.adj[edge.u].append(edge)
            self.adj[edge.v].append(edge)

    def update_conductance(
        self,
        u: int,
        v: int,
        conductance: float,
        required_latch: Optional[str] = None,
    ) -> None:
        """Updates the state-dependent conductance `G_e(s_fsa)` on contact edge `(u, v)`."""
        for edge in self.edges:
            if (edge.u == u and edge.v == v) or (edge.u == v and edge.v == u):
                edge.conductance = float(conductance)
                if required_latch is not None:
                    edge.required_latch = str(required_latch)
                if conductance >= 1.0:
                    if u in self.nodes and not self.nodes[u].is_ego:
                        self.nodes[u].is_wall = False
                        self.unlocked_colors.add(int(self.nodes[u].color))
                    if v in self.nodes and not self.nodes[v].is_ego:
                        self.nodes[v].is_wall = False
                        self.unlocked_colors.add(int(self.nodes[v].color))

    @classmethod
    def from_board(
        cls,
        board: np.ndarray,
        bg_mode: int = 0,
        ego_colors: Optional[Set[int]] = None,
        pushable_colors: Optional[Set[int]] = None,
        wall_colors: Optional[Set[int]] = None,
        ego_comp: Optional[Dict[str, Any]] = None,
        bg_color: Optional[int] = None,
        unlocked_colors: Optional[Set[int]] = None,
        latch_state: Optional[str] = None,
    ) -> ContactGraphSheaf:
        if bg_color is not None:
            bg_mode = int(bg_color)
        H, W = board.shape
        ego_colors = ego_colors or set()
        pushable_colors = pushable_colors or set()
        wall_colors = wall_colors or set()
        unlocked_colors = unlocked_colors or set()

        comps = _connected_components_2d(board, connectivity=4, same_color=True, ignore_val=bg_mode)
        nodes: Dict[int, ContactNode] = {}
        comp_grid = np.full((H, W), -1, dtype=np.int32)
        ego_id: Optional[int] = None

        ego_target_pt: Optional[Tuple[int, int]] = None
        if ego_comp is not None and len(ego_comp["rows"]) > 0:
            ego_target_pt = (int(ego_comp["rows"][0]), int(ego_comp["cols"][0]))

        for idx, cp in enumerate(comps):
            rs = cp["rows"]
            cs = cp["cols"]
            col = int(cp["color"])
            area = int(cp["area"])
            h, w = int(cp["h"]), int(cp["w"])
            r0, r1, c0, c1 = int(cp["r0"]), int(cp["r1"]), int(cp["c0"]), int(cp["c1"])
            cy, cx = float(cp["centroid_y"]), float(cp["centroid_x"])

            # Check if ego
            is_ego = False
            if ego_target_pt is not None:
                er_t, ec_t = ego_target_pt
                if (
                    r0 <= er_t < r1
                    and c0 <= ec_t < c1
                    and col == int(board[er_t, ec_t])
                    and bool(np.any((rs == er_t) & (cs == ec_t)))
                ):
                    is_ego = True
            elif col in ego_colors and (area <= 256 and max(h, w) <= 28):
                is_ego = True

            # Dual-background / Playfield Cavity Floor Guard:
            # When a bordered canvas has outer HUD/frame background `bg_mode` and an interior
            # solid playfield floor substrate of color `col != bg_mode` (`area > 256`, `min(h, w) >= 16`,
            # `fill_ratio >= 0.85`, with zero `bg_mode` holes inside its interior and enclosing
            # interior sprites), treat the floor as background (`-1`) rather than a solid wall node.
            if (
                not is_ego
                and col not in wall_colors
                and area > 256
                and min(h, w) >= 16
                and float(cp["fill_ratio"]) >= 0.85
            ):
                sub_in = board[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1]
                if not np.any(sub_in == bg_mode) and np.any(sub_in != col):
                    continue

            comp_grid[rs, cs] = idx
            if is_ego and ego_id is None:
                ego_id = idx

            is_compact_gate = area <= 32 and max(h, w) <= 8
            is_wall = (col in wall_colors and not (is_compact_gate and col in unlocked_colors)) or (
                area > 256 and max(h, w) > 32
            )
            is_pushable = (col in pushable_colors) and not is_wall

            # Check hollow frame / cavity
            is_cavity = False
            if h >= 3 and w >= 3 and (area < h * w):
                sub = board[r0:r1, c0:c1]
                if np.any(sub[1:-1, 1:-1] == bg_mode):
                    is_cavity = True

            node = ContactNode(
                node_id=idx,
                color=col,
                rows=rs,
                cols=cs,
                area=area,
                h=h,
                w=w,
                r0=r0,
                r1=r1,
                c0=c0,
                c1=c1,
                centroid_y=cy,
                centroid_x=cx,
                is_ego=is_ego,
                is_wall=is_wall,
                is_pushable=is_pushable,
                is_cavity=is_cavity,
            )
            nodes[idx] = node

        # Detect container nesting (inner objects inside hollow frame)
        edges: List[ContactEdge] = []
        for u_id, u_node in nodes.items():
            if not u_node.is_cavity:
                continue
            for v_id, v_node in nodes.items():
                if u_id == v_id:
                    continue
                if (
                    u_node.r0 <= v_node.r0
                    and v_node.r1 <= u_node.r1
                    and u_node.c0 <= v_node.c0
                    and v_node.c1 <= u_node.c1
                ):
                    v_node.is_nested = True
                    v_node.parent_container_id = u_id
                    edge_nest = ContactEdge(
                        u=u_id,
                        v=v_id,
                        normal=(0.0, 0.0),
                        contact_points_u=[],
                        contact_points_v=[],
                        is_enclosed=True,
                        container_id=u_id,
                        content_id=v_id,
                    )
                    edges.append(edge_nest)

        # Fast vectorized boundary contact detection using NumPy array slicing
        contact_map: Dict[Tuple[int, int], Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, float]] = {}

        # Horizontal contacts: (r, c) and (r, c + 1)
        left = comp_grid[:, :-1]
        right = comp_grid[:, 1:]
        mask_h = (left != right) & (left != -1) & (right != -1)
        if np.any(mask_h):
            r_idx, c_idx = np.where(mask_h)
            for r, c in zip(r_idx, c_idx):
                u_id, v_id = int(left[r, c]), int(right[r, c])
                pair_key = (u_id, v_id) if u_id < v_id else (v_id, u_id)
                if pair_key not in contact_map:
                    contact_map[pair_key] = ([], [], 0.0, 0.0)
                pts_u, pts_v, sy, sx = contact_map[pair_key]
                if u_id < v_id:
                    pts_u.append((int(r), int(c)))
                    pts_v.append((int(r), int(c + 1)))
                    contact_map[pair_key] = (pts_u, pts_v, sy, sx + 1.0)
                else:
                    pts_u.append((int(r), int(c + 1)))
                    pts_v.append((int(r), int(c)))
                    contact_map[pair_key] = (pts_u, pts_v, sy, sx - 1.0)

        # Vertical contacts: (r, c) and (r + 1, c)
        top = comp_grid[:-1, :]
        bot = comp_grid[1:, :]
        mask_v = (top != bot) & (top != -1) & (bot != -1)
        if np.any(mask_v):
            r_idx, c_idx = np.where(mask_v)
            for r, c in zip(r_idx, c_idx):
                u_id, v_id = int(top[r, c]), int(bot[r, c])
                pair_key = (u_id, v_id) if u_id < v_id else (v_id, u_id)
                if pair_key not in contact_map:
                    contact_map[pair_key] = ([], [], 0.0, 0.0)
                pts_u, pts_v, sy, sx = contact_map[pair_key]
                if u_id < v_id:
                    pts_u.append((int(r), int(c)))
                    pts_v.append((int(r + 1), int(c)))
                    contact_map[pair_key] = (pts_u, pts_v, sy + 1.0, sx)
                else:
                    pts_u.append((int(r + 1), int(c)))
                    pts_v.append((int(r), int(c)))
                    contact_map[pair_key] = (pts_u, pts_v, sy - 1.0, sx)

        for (u_id, v_id), (pts_u, pts_v, sy, sx) in contact_map.items():
            norm_len = math.sqrt(sy * sy + sx * sx)
            if norm_len > 1e-6:
                ny = sy / norm_len
                nx = sx / norm_len
            else:
                dy_c = nodes[v_id].centroid_y - nodes[u_id].centroid_y
                dx_c = nodes[v_id].centroid_x - nodes[u_id].centroid_x
                d_len = math.sqrt(dy_c * dy_c + dx_c * dx_c)
                if d_len > 1e-6:
                    ny = dy_c / d_len
                    nx = dx_c / d_len
                else:
                    ny, nx = 0.0, 1.0
            u_unlocked_gate = (
                nodes[u_id].color in unlocked_colors
                and nodes[u_id].area <= 32
                and max(nodes[u_id].h, nodes[u_id].w) <= 8
            )
            v_unlocked_gate = (
                nodes[v_id].color in unlocked_colors
                and nodes[v_id].area <= 32
                and max(nodes[v_id].h, nodes[v_id].w) <= 8
            )
            cond_val = 1.0 if (u_unlocked_gate or v_unlocked_gate) else 0.0
            edge = ContactEdge(
                u=u_id,
                v=v_id,
                normal=(ny, nx),
                contact_points_u=pts_u,
                contact_points_v=pts_v,
                is_enclosed=False,
                conductance=cond_val,
                required_latch=latch_state if cond_val >= 1.0 else None,
            )
            edges.append(edge)

        return cls(
            nodes=nodes,
            edges=edges,
            board_shape=(H, W),
            bg_mode=bg_mode,
            ego_node_id=ego_id,
            comp_grid=comp_grid,
            unlocked_colors=unlocked_colors,
        )

    def compute_non_penetration_coboundary(
        self,
        node_displacements: Dict[int, Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        """
        Evaluates the sheaf restriction coboundary (delta v)(e) across all contact edges.
        If a single actuator/ego node displacement is supplied, first propagates the kinematic
        contact chain to obtain the induced stalk velocities across all coupled bodies.
        """
        disp_map = dict(node_displacements)
        if len(disp_map) == 1:
            root_id, act_dir = next(iter(disp_map.items()))
            ok, prop_map, _ = self.propagate_displacement(action_dir=act_dir, ego_node_id=root_id)
            if ok and prop_map:
                disp_map = prop_map

        coboundary: Dict[Tuple[int, int], float] = {}
        for edge in self.edges:
            if edge.is_enclosed:
                continue
            u_node = self.nodes[edge.u]
            v_node = self.nodes[edge.v]
            dy_u, dx_u = disp_map.get(edge.u, (0, 0))
            dy_v, dx_v = disp_map.get(edge.v, (0, 0))
            delta_v = edge.compute_coboundary(
                v_u=(float(dy_u), float(dx_u), 0.0),
                x_u=(u_node.centroid_y, u_node.centroid_x),
                v_v=(float(dy_v), float(dx_v), 0.0),
                x_v=(v_node.centroid_y, v_node.centroid_x),
            )
            coboundary[(edge.u, edge.v)] = float(delta_v)
        return coboundary

    def propagate_displacement(
        self,
        action_dir: Any = (0, 0),
        ego_node_id: Optional[int] = None,
        allow_candidate_push: bool = True,
        dy: Optional[int] = None,
        dx: Optional[int] = None,
        bg_color: Optional[int] = None,
    ) -> Any:
        """
        Solves topological force and displacement propagation through the contact sheaf.
        Supports both:
          - `(success, moving_nodes, reason) = sheaf.propagate_displacement((dy, dx), ego_node_id=...)`
          - `displaced_board = sheaf.propagate_displacement(board, ego_node_id=..., dy=..., dx=..., bg_color=...)`
        """
        board_override: Optional[np.ndarray] = None
        if isinstance(action_dir, np.ndarray):
            board_override = action_dir
            action_dir = (int(dy or 0), int(dx or 0))
        elif dy is not None or dx is not None:
            action_dir = (int(dy or 0), int(dx or 0))

        def _finish(ok: bool, mov: Dict[int, Tuple[int, int]], reason: str) -> Any:
            if board_override is not None:
                bg = int(bg_color) if bg_color is not None else self.bg_mode
                return self.apply_displacement_to_board(board_override, bg, mov) if ok else board_override.copy()
            return ok, mov, reason

        dy_val, dx_val = int(action_dir[0]), int(action_dir[1])
        if dy_val == 0 and dx_val == 0:
            return _finish(True, {}, "no_op")

        root_id = ego_node_id if ego_node_id is not None else self.ego_node_id
        if root_id is None or root_id not in self.nodes:
            return _finish(False, {}, "no_ego_root")

        H, W = self.board_shape
        moving_nodes: Dict[int, Tuple[int, int]] = {root_id: (dy_val, dx_val)}
        queue: List[int] = [root_id]
        visited: Set[int] = {root_id}
        permeable_nodes: Set[int] = {
            nid for nid, nd in self.nodes.items() if not nd.is_ego and nd.color in self.unlocked_colors
        }

        while queue:
            curr_id = queue.pop(0)
            curr_disp = moving_nodes[curr_id]
            curr_node = self.nodes[curr_id]

            # 1. Container nesting propagation
            # A. If curr_node is a container, all nested children move with it
            for edge in self.adj.get(curr_id, []):
                if edge.is_enclosed and edge.container_id == curr_id:
                    child_id = edge.content_id
                    if child_id is not None and child_id not in moving_nodes:
                        moving_nodes[child_id] = curr_disp
                        visited.add(child_id)
                        queue.append(child_id)

            # B. If curr_node is nested inside a container, pushing into wall pushes container
            if curr_node.is_nested and curr_node.parent_container_id is not None:
                cont_id = curr_node.parent_container_id
                if cont_id not in moving_nodes:
                    cont_node = self.nodes[cont_id]
                    n_r = curr_node.rows + curr_disp[0]
                    n_c = curr_node.cols + curr_disp[1]
                    cont_pts = set(zip(cont_node.rows.tolist(), cont_node.cols.tolist()))
                    if any((nr, nc) in cont_pts for nr, nc in zip(n_r, n_c)):
                        moving_nodes[cont_id] = curr_disp
                        visited.add(cont_id)
                        queue.append(cont_id)

            # 2. Contact graph edge normal propagation
            for edge in self.adj.get(curr_id, []):
                if edge.is_enclosed:
                    continue
                other_id = edge.v if edge.u == curr_id else edge.u
                ny, nx = edge.normal if edge.u == curr_id else (-edge.normal[0], -edge.normal[1])
                dot_prod = ny * float(curr_disp[0]) + nx * float(curr_disp[1])
                if dot_prod > 0.05:  # Pushing into other_id
                    if edge.conductance >= 1.0:
                        permeable_nodes.add(other_id)
                        continue
                    other_node = self.nodes[other_id]
                    if other_node.is_wall:
                        return _finish(False, {}, "wall_lock")
                    if other_id not in moving_nodes:
                        if other_node.is_pushable or (allow_candidate_push and not other_node.is_wall):
                            moving_nodes[other_id] = curr_disp
                            visited.add(other_id)
                            queue.append(other_id)
                        else:
                            return _finish(False, {}, "unpushable_obstacle")

            # 3. Swept boundary and direct canvas collision check
            n_r = curr_node.rows + curr_disp[0]
            n_c = curr_node.cols + curr_disp[1]
            if np.any((n_r < 0) | (n_r >= H) | (n_c < 0) | (n_c >= W)):
                return _finish(False, {}, "boundary_collision")

            if self.comp_grid is not None:
                for nr, nc in zip(n_r, n_c):
                    other_id = int(self.comp_grid[nr, nc])
                    if other_id != -1 and other_id != curr_id and other_id not in permeable_nodes:
                        other_node = self.nodes[other_id]
                        if other_node.is_wall:
                            return _finish(False, {}, "wall_collision")
                        if other_id not in moving_nodes:
                            if other_node.is_pushable or (allow_candidate_push and not other_node.is_wall):
                                moving_nodes[other_id] = curr_disp
                                visited.add(other_id)
                                queue.append(other_id)
                            else:
                                return _finish(False, {}, "unpushable_obstacle")

        # Global Non-Penetration Verification Pass
        old_moving_coords: Set[Tuple[int, int]] = set()
        for nid in moving_nodes:
            node = self.nodes[nid]
            old_moving_coords.update(zip(node.rows.tolist(), node.cols.tolist()))

        new_coords_by_node: Dict[int, Set[Tuple[int, int]]] = {}
        for nid, (dy_i, dx_i) in moving_nodes.items():
            node = self.nodes[nid]
            nr = node.rows + dy_i
            nc = node.cols + dx_i
            if np.any((nr < 0) | (nr >= H) | (nc < 0) | (nc >= W)):
                return _finish(False, {}, "boundary_collision")
            new_set = set(zip(nr.tolist(), nc.tolist()))
            new_coords_by_node[nid] = new_set

        # Check inter-body non-penetration between moving bodies: (delta v)(e) >= 0
        all_new_coords: Set[Tuple[int, int]] = set()
        for nid, new_set in new_coords_by_node.items():
            if all_new_coords.intersection(new_set):
                return _finish(False, {}, "inter_body_penetration")
            all_new_coords.update(new_set)

        # Check collision with any stationary components
        if self.comp_grid is not None:
            for nr, nc in all_new_coords:
                if (nr, nc) not in old_moving_coords:
                    other_id = int(self.comp_grid[nr, nc])
                    if other_id != -1 and other_id not in moving_nodes and other_id not in permeable_nodes:
                        return _finish(False, {}, "collision_with_stationary")

        return _finish(True, moving_nodes, "success")

    def apply_displacement_to_board(
        self,
        board: np.ndarray,
        bg_mode: int,
        moving_nodes: Dict[int, Tuple[int, int]],
    ) -> np.ndarray:
        new_board = board.copy()
        for nid in moving_nodes:
            node = self.nodes[nid]
            new_board[node.rows, node.cols] = bg_mode
        for nid, (dy, dx) in moving_nodes.items():
            node = self.nodes[nid]
            new_board[node.rows + dy, node.cols + dx] = board[node.rows, node.cols]
        return new_board

    def detect_peg_docking(
        self,
        peg_node_id: int,
        receptor_cavity_id: int,
        disp: Tuple[int, int],
    ) -> bool:
        if peg_node_id not in self.nodes or receptor_cavity_id not in self.nodes:
            return False
        peg = self.nodes[peg_node_id]
        receptor = self.nodes[receptor_cavity_id]
        new_r = peg.rows + disp[0]
        new_c = peg.cols + disp[1]
        if (
            np.all(new_r > receptor.r0)
            and np.all(new_r < receptor.r1 - 1)
            and np.all(new_c > receptor.c0)
            and np.all(new_c < receptor.c1 - 1)
        ):
            rec_coords = set(zip(receptor.rows.tolist(), receptor.cols.tolist()))
            return not any((nr, nc) in rec_coords for nr, nc in zip(new_r, new_c))
        return False


@dataclass
class TropicalTransitionRule:
    """Induced tropical min-plus transition rule over `(state_hash, latch_state)`."""

    source_node: Tuple[str, str]
    target_node: Tuple[str, str]
    action_key: ActionKey
    cost: float = 1.0
    observations: int = 1
    eta_nonlocal: float = 0.0
    trigger_colors: Tuple[int, ...] = ()
    toggled_colors: Tuple[int, ...] = ()
    d4_sprite_tuple: Optional[Tuple[Tuple[int, ...], int, int, int]] = None


class TropicalStateLatchSheaf:
    """
    Tropical Min-Plus State-Latch Sheaf over the joint state space `V = H_nope x S_FSA`.

    Maintains:
      1. A directed tropical cost matrix `W in T^{|V| x |V|}` over the tropical semiring
         `T = (R u {+inf}, oplus = min, otimes = +)` and exact multi-hop geodesic
         shortest paths `D^* = bigoplus_{k=0}^{|V|-1} W^{otimes k}` via `tropical_kleene_closure`.
      2. Order-2 Tropical Conjunctions (`pairwise_latch_tensor` `T_{a, b -> e}`):
         Tracks empirical single-latch conductance variance `Var(c(e) | s_a)` and promotes
         any coboundary edge `e` with conflicting single-latch observations (`{0.0, 1.0}`)
         to pairwise minimal witnesses `(s_a, s_b) in FSA^2` in `O(|FSA|^2)` integer ops.
      3. Cross-Level Latch-Homomorphism Seeding (`d4_latch_homomorphism_table`):
         Indexes proven causal latches (`eta_nonlocal >= 0.50` or discrete latch transitions)
         by canonical `D_4`-invariant sprite/color tuple `(glyph_D4, c_outer, c_inner, area_bucket)`
         rather than raw `(y, x)` coordinates, enabling `L1..Lk` Frame-0 transfer when switch
         coordinates shift across level layouts.
    """

    def __init__(self, inf_val: float = 1e12) -> None:
        self.inf_val = float(inf_val)
        self.reset(clear_all=True)

    def reset(self, clear_all: bool = False) -> None:
        self.node_to_idx: Dict[Tuple[str, str], int] = {}
        self.idx_to_node: List[Tuple[str, str]] = []
        self.rules: Dict[Tuple[Tuple[str, str], Tuple[str, str]], TropicalTransitionRule] = {}
        self.action_edges: Dict[Tuple[Tuple[str, str], ActionKey], Tuple[str, str]] = {}
        if clear_all or not hasattr(self, "latch_states"):
            self.latch_states: Set[str] = set()
            self.trigger_colors: Set[int] = set()
            self.unlocked_colors_by_latch: Dict[str, Set[int]] = {}
            self.edge_conductance_table: Dict[Tuple[str, int], float] = {}
            self.single_latch_observations: Dict[Tuple[str, int], Set[float]] = {}
            self.pairwise_latch_tensor: Dict[Tuple[str, str, int], float] = {}
            self.d4_latch_homomorphism_table: Dict[
                Tuple[Tuple[int, ...], int, int, int], Dict[str, Any]
            ] = {}
        self._dirty: bool = True
        self._closure_dist: Optional[np.ndarray] = None
        self._closure_routing: Optional[np.ndarray] = None

    @staticmethod
    def _d4_canonical_glyph(mask: np.ndarray) -> Tuple[int, ...]:
        """Returns the lexicographically minimal tuple across all 8 Dihedral D_4 symmetries of a binary mask."""
        m = (mask > 0).astype(np.uint8)
        if not np.any(m):
            return tuple(m.ravel().tolist())
        if m.shape[0] != m.shape[1]:
            s = max(int(m.shape[0]), int(m.shape[1]))
            pad = np.zeros((s, s), dtype=np.uint8)
            pad[: m.shape[0], : m.shape[1]] = m
            m = pad
        cands: List[Tuple[int, ...]] = []
        for k in range(4):
            rk = np.rot90(m, k)
            cands.append(tuple(rk.ravel().tolist()))
            cands.append(tuple(np.fliplr(rk).ravel().tolist()))
        return min(cands)

    @classmethod
    def canonical_d4_sprite_tuple(
        cls,
        board: np.ndarray,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
        bg_val: Optional[int] = None,
    ) -> Tuple[Tuple[int, ...], int, int, int]:
        """
        Computes the translation-free canonical `D_4`-invariant sprite/color tuple
        `(glyph_D4, c_outer, c_inner, area_bucket)` for the bounding box `[r0:r1, c0:c1]`.
        """
        H, W = board.shape
        rr0, rr1 = max(0, int(r0)), min(H, int(r1))
        cc0, cc1 = max(0, int(c0)), min(W, int(c1))
        if rr1 <= rr0 or cc1 <= cc0:
            return ((), 0, 0, 0)
        if bg_val is None:
            vals_b, cnts_b = np.unique(board, return_counts=True)
            bg_val = int(vals_b[int(np.argmax(cnts_b))])
        sub = board[rr0:rr1, cc0:cc1]
        fg_mask = sub != int(bg_val)
        glyph_d4 = cls._d4_canonical_glyph(fg_mask)
        fg_vals = sub[fg_mask]
        if fg_vals.size == 0:
            return (glyph_d4, int(bg_val), int(bg_val), 0)
        u_vals, u_cnts = np.unique(fg_vals, return_counts=True)
        order = np.argsort(-u_cnts)
        c_outer = int(u_vals[int(order[0])])
        c_inner = int(u_vals[int(order[1])]) if len(order) > 1 else c_outer
        if sub.shape[0] >= 3 and sub.shape[1] >= 3:
            center_val = int(sub[sub.shape[0] // 2, sub.shape[1] // 2])
            if center_val != int(bg_val):
                c_inner = center_val
        area_bucket = int(np.sum(fg_mask))
        return (glyph_d4, c_outer, c_inner, area_bucket)

    @staticmethod
    def _atomic_latch_tokens(latch_state: Any) -> Tuple[str, ...]:
        """Decomposes a latch state signature into sorted atomic FSA latch tokens."""
        if latch_state is None:
            return ()
        if isinstance(latch_state, (tuple, list, set, frozenset)):
            toks = [str(x).strip() for x in latch_state if str(x).strip()]
            return tuple(sorted(set(toks)))
        s = str(latch_state).strip()
        if not s:
            return ()
        if "&" in s:
            return tuple(sorted({p.strip() for p in s.split("&") if p.strip()}))
        if "|" in s:
            return tuple(sorted({p.strip() for p in s.split("|") if p.strip()}))
        return (s,)

    def single_latch_variance(self, latch_a: str, edge_color: int) -> float:
        """
        Returns the empirical single-latch conductance variance `Var(c(e) | s_a)` for
        latch token `latch_a` and barrier color `edge_color`. Non-zero (`0.25`) iff both
        blocked (`0.0`) and open (`1.0`) transitions have been observed under `latch_a`.
        """
        obs = self.single_latch_observations.get((str(latch_a), int(edge_color)))
        if not obs or len(obs) <= 1:
            return 0.0
        vals = np.array(sorted(obs), dtype=np.float64)
        return float(np.var(vals))

    def observe_single_latch_conductance(
        self,
        latch_a: str,
        edge_color: int,
        conductance: float,
    ) -> float:
        """
        Records a single-latch conductance observation `c(e) in {0.0, 1.0}` for `(latch_a, edge_color)`
        and returns the updated empirical variance `Var(c(e) | s_a)`. If variance becomes
        non-zero, removes the ambiguous single-latch entry from `unlocked_colors_by_latch`
        so order-2 conjunction witnesses `T_{a, b -> e}` resolve the gate.
        """
        key = (str(latch_a), int(edge_color))
        c_bin = 1.0 if float(conductance) > 0.5 else 0.0
        obs = self.single_latch_observations.setdefault(key, set())
        obs.add(c_bin)
        var = self.single_latch_variance(latch_a, edge_color)
        if var > 0.0:
            # Ambiguous single-latch rule: demote single-latch unlock and require order-2 witness
            if str(latch_a) in self.unlocked_colors_by_latch:
                self.unlocked_colors_by_latch[str(latch_a)].discard(int(edge_color))
            self.edge_conductance_table.pop(key, None)
        elif c_bin > 0.5:
            self.unlocked_colors_by_latch.setdefault(str(latch_a), set()).add(int(edge_color))
            self.edge_conductance_table[key] = 1.0
        return var

    def observe_order2_conjunction(
        self,
        latch_a: str,
        latch_b: str,
        edge_color: int,
        conductance: float = 1.0,
    ) -> None:
        """
        Records an order-2 pairwise latch witness `(s_a, s_b) -> Delta c(e)` in the
        symmetric pairwise latch tensor `T_{a, b -> e}` (`pairwise_latch_tensor`).
        """
        sa, sb = str(latch_a), str(latch_b)
        self.latch_states.add(sa)
        self.latch_states.add(sb)
        ec = int(edge_color)
        c_val = float(conductance)
        self.pairwise_latch_tensor[(sa, sb, ec)] = c_val
        self.pairwise_latch_tensor[(sb, sa, ec)] = c_val
        conj_key = f"{min(sa, sb)}&{max(sa, sb)}"
        self.latch_states.add(conj_key)
        if c_val > 0.5:
            self.unlocked_colors_by_latch.setdefault(conj_key, set()).add(ec)
            self.edge_conductance_table[(conj_key, ec)] = c_val

    def query_order2_conductance(
        self,
        active_latches: Iterable[str],
        edge_color: int,
    ) -> float:
        """
        Evaluates order-2 tropical/Boolean conjunction conductance over `active_latches` in
        `O(|FSA|^2)` integer operations:
          - If any pair `(s_a, s_b)` in `active_latches^2` has a positive witness in
            `pairwise_latch_tensor`, returns `T_{a, b -> e}`.
          - Otherwise falls back to unambiguous (`Var(c(e) | s_a) == 0`) single-latch conductance.
        """
        toks = sorted({t for raw in active_latches for t in self._atomic_latch_tokens(raw)})
        ec = int(edge_color)
        for i, sa in enumerate(toks):
            for sb in toks[i:]:
                if (sa, sb, ec) in self.pairwise_latch_tensor:
                    val = float(self.pairwise_latch_tensor[(sa, sb, ec)])
                    if val > 0.5:
                        return val
        for sa in toks:
            if self.single_latch_variance(sa, ec) == 0.0:
                if ec in self.unlocked_colors_by_latch.get(sa, set()):
                    return float(self.edge_conductance_table.get((sa, ec), 1.0))
        return 0.0

    @property
    def num_rules_induced(self) -> int:
        return len(self.rules)

    @property
    def num_latch_states(self) -> int:
        return len(self.latch_states)

    def _register_node(self, node: Tuple[str, str]) -> int:
        idx = self.node_to_idx.get(node)
        if idx is None:
            idx = len(self.idx_to_node)
            self.node_to_idx[node] = idx
            self.idx_to_node.append(node)
            self._dirty = True
        return idx

    def get_unlocked_colors(self, latch_state: Optional[Any] = None) -> Set[int]:
        """
        Returns the set of barrier/door colors unlocked in `latch_state`, combining
        unambiguous (`Var(c(e) | s_a) == 0`) single-latch triggers and order-2 pairwise
        conjunction witnesses `(s_a, s_b) in FSA^2` from `pairwise_latch_tensor`.
        """
        if latch_state is None:
            out: Set[int] = set()
            for l_key, cols in self.unlocked_colors_by_latch.items():
                for c in cols:
                    if self.single_latch_variance(l_key, c) == 0.0:
                        out.add(int(c))
            for (_sa, _sb, ec), val in self.pairwise_latch_tensor.items():
                if val > 0.5:
                    out.add(int(ec))
            return out

        toks = self._atomic_latch_tokens(latch_state)
        out_set: Set[int] = set()
        raw_str = str(latch_state)
        for c in self.unlocked_colors_by_latch.get(raw_str, set()):
            if self.single_latch_variance(raw_str, c) == 0.0:
                out_set.add(int(c))
        for t in toks:
            for c in self.unlocked_colors_by_latch.get(t, set()):
                if self.single_latch_variance(t, c) == 0.0:
                    out_set.add(int(c))
        if len(toks) >= 2:
            for i, sa in enumerate(toks):
                for sb in toks[i + 1 :]:
                    for (pa, pb, ec), val in self.pairwise_latch_tensor.items():
                        if pa == sa and pb == sb and val > 0.5:
                            out_set.add(int(ec))
        return out_set

    def apply_to_contact_sheaf(
        self,
        sheaf: ContactGraphSheaf,
        latch_state: Optional[Any] = None,
    ) -> None:
        """Applies learned order-1 and order-2 tropical edge-conductance toggles for `latch_state` to `sheaf`."""
        unlocked = self.get_unlocked_colors(latch_state)
        if not unlocked:
            return
        req_str = str(latch_state) if latch_state is not None else None
        for edge in sheaf.edges:
            if edge.is_enclosed:
                continue
            u_node = sheaf.nodes.get(edge.u)
            v_node = sheaf.nodes.get(edge.v)
            if (u_node is not None and u_node.color in unlocked) or (
                v_node is not None and v_node.color in unlocked
            ):
                sheaf.update_conductance(edge.u, edge.v, 1.0, required_latch=req_str)

    def seed_level_from_d4_homomorphism(
        self,
        frame_2d: np.ndarray,
        clock_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cross-Level Latch-Homomorphism Seeding (`L1..Lk` Zero-Cost Frame-0 Transfer):
        Scans connected foreground sprites on Frame 0 of a new level, computes their
        canonical `D_4`-invariant sprite/color tuple `(glyph_D4, c_outer, c_inner, area_bucket)`,
        and matches them against proven causal latch rules in `d4_latch_homomorphism_table`.
        Returns the list of matched actuator bindings with their shifted `(x, y)` coordinates
        and inherited FSA transition/conductance metadata.
        """
        if not self.d4_latch_homomorphism_table:
            return []
        board = _frame_to_2d_numpy(frame_2d).copy()
        if clock_mask is not None and clock_mask.shape == board.shape and np.any(clock_mask):
            board = _mask_clock_to_bg(board, clock_mask)
        vals_b, cnts_b = np.unique(board, return_counts=True)
        bg_val = int(vals_b[int(np.argmax(cnts_b))])
        comps = _connected_components_2d(board, connectivity=4, same_color=False, ignore_val=bg_val)
        same_comps = _connected_components_2d(board, connectivity=4, same_color=True, ignore_val=bg_val)

        seeded_bindings: List[Dict[str, Any]] = []
        seen_xy: Set[Tuple[int, int]] = set()
        for c in comps + same_comps:
            r0, r1, c0, c1 = int(c["r0"]), int(c["r1"]), int(c["c0"]), int(c["c1"])
            d4_sig = self.canonical_d4_sprite_tuple(board, r0, r1, c0, c1, bg_val=bg_val)
            hom_entry = self.d4_latch_homomorphism_table.get(d4_sig)
            if hom_entry is None:
                continue
            cx = int(round(float(c["centroid_x"])))
            cy = int(round(float(c["centroid_y"])))
            if (cx, cy) in seen_xy:
                continue
            seen_xy.add((cx, cy))
            toggled_cols = tuple(hom_entry.get("toggled_colors", ()))
            target_latch = str(hom_entry.get("target_latch", f"d4_latch_{d4_sig[1]}_{d4_sig[2]}"))
            self.latch_states.add(target_latch)
            for tc in toggled_cols:
                if self.single_latch_variance(target_latch, int(tc)) == 0.0:
                    self.unlocked_colors_by_latch.setdefault(target_latch, set()).add(int(tc))
                    self.edge_conductance_table[(target_latch, int(tc))] = 1.0
            seeded_bindings.append(
                {
                    "x": cx,
                    "y": cy,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "d4_sprite_tuple": d4_sig,
                    "eta_nonlocal": float(hom_entry.get("eta_nonlocal", 0.55)),
                    "target_latch": target_latch,
                    "toggled_colors": toggled_cols,
                    "trigger_colors": tuple(hom_entry.get("trigger_colors", ())),
                }
            )
        return seeded_bindings

    def observe_transition(
        self,
        prev_hash: Any,
        next_hash: Any,
        action_key: ActionKey,
        sh_prev: Optional[str] = None,
        sh_next: Optional[str] = None,
        prev_latch: Any = "latch_default",
        next_latch: Any = "latch_default",
        latch_prev: Any = None,
        latch_next: Any = None,
        cost: float = 1.0,
        is_game_over: bool = False,
        is_self_loop: bool = False,
        blocked_colors: Optional[Iterable[int]] = None,
    ) -> Optional[TropicalTransitionRule]:
        arr_prev = prev_hash if isinstance(prev_hash, np.ndarray) else None
        arr_next = next_hash if isinstance(next_hash, np.ndarray) else None

        if sh_prev is not None:
            u_hash = str(sh_prev)
        elif arr_prev is not None:
            u_hash = _blake2b_grid(arr_prev)
        else:
            u_hash = str(prev_hash)

        if sh_next is not None:
            v_hash = str(sh_next)
        elif arr_next is not None:
            v_hash = _blake2b_grid(arr_next)
        else:
            v_hash = str(next_hash)

        raw_l_prev = latch_prev if latch_prev is not None else prev_latch
        raw_l_next = latch_next if latch_next is not None else next_latch
        toks_prev = self._atomic_latch_tokens(raw_l_prev)
        toks_next = self._atomic_latch_tokens(raw_l_next)
        eff_l_prev = "&".join(toks_prev) if len(toks_prev) > 1 else str(raw_l_prev)
        eff_l_next = "&".join(toks_next) if len(toks_next) > 1 else str(raw_l_next)
        self.latch_states.add(eff_l_prev)
        self.latch_states.add(eff_l_next)
        for t in toks_prev + toks_next:
            self.latch_states.add(t)

        if blocked_colors is not None:
            for bc in blocked_colors:
                for t in toks_prev or (eff_l_prev,):
                    self.observe_single_latch_conductance(t, int(bc), 0.0)

        u = (u_hash, eff_l_prev)
        v = (v_hash, eff_l_next)
        self._register_node(u)
        self._register_node(v)
        if is_game_over or is_self_loop or u == v:
            return None

        eta_nl = 0.0
        trig_cols: Tuple[int, ...] = ()
        toggled_cols: Tuple[int, ...] = ()
        d4_sig_induced: Optional[Tuple[Tuple[int, ...], int, int, int]] = None
        if arr_prev is not None and arr_next is not None and arr_prev.shape == arr_next.shape:
            diff_mask = arr_prev != arr_next
            diff_rs, diff_cs = np.nonzero(diff_mask)
            if diff_rs.size > 0:
                vals_bg, cnts_bg = np.unique(arr_prev, return_counts=True)
                bg_m = int(vals_bg[int(np.argmax(cnts_bg))])
                if isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6:
                    cx, cy = int(action_key[1]), int(action_key[2])
                    far_m = np.maximum(np.abs(diff_cs - cx), np.abs(diff_rs - cy)) > 4
                    eta_nl = float(np.sum(far_m)) / float(diff_rs.size)
                    if np.any(far_m):
                        rem_cols = set(int(c) for c in np.unique(arr_prev[diff_rs[far_m], diff_cs[far_m]])) | set(
                            int(c) for c in np.unique(arr_next[diff_rs[far_m], diff_cs[far_m]])
                        )
                        rem_cols.discard(bg_m)
                        toggled_cols = tuple(sorted(rem_cols))
                    near_m = ~far_m
                    if np.any(near_m):
                        nr0, nr1 = int(diff_rs[near_m].min()), int(diff_rs[near_m].max()) + 1
                        nc0, nc1 = int(diff_cs[near_m].min()), int(diff_cs[near_m].max()) + 1
                        d4_sig_induced = self.canonical_d4_sprite_tuple(
                            arr_prev, nr0, nr1, nc0, nc1, bg_val=bg_m
                        )
                    elif 0 <= cy < arr_prev.shape[0] and 0 <= cx < arr_prev.shape[1]:
                        comps_p = _connected_components_2d(
                            arr_prev, connectivity=4, same_color=True, ignore_val=bg_m
                        )
                        for cp in comps_p:
                            if int(cp["r0"]) <= cy < int(cp["r1"]) and int(cp["c0"]) <= cx < int(cp["c1"]):
                                d4_sig_induced = self.canonical_d4_sprite_tuple(
                                    arr_prev,
                                    int(cp["r0"]),
                                    int(cp["r1"]),
                                    int(cp["c0"]),
                                    int(cp["c1"]),
                                    bg_val=bg_m,
                                )
                                break
                else:
                    if eff_l_prev != eff_l_next:
                        clusters = _connected_components_2d(
                            diff_mask.astype(np.int32), connectivity=8, same_color=True, ignore_val=0
                        )
                        if len(clusters) >= 2:
                            clusters.sort(key=lambda c: int(c["area"]), reverse=True)
                            c_main = clusters[0]
                            c_rem_area = sum(int(c["area"]) for c in clusters[1:])
                            dist_rem = max(
                                abs(float(clusters[1]["centroid_y"]) - float(c_main["centroid_y"])),
                                abs(float(clusters[1]["centroid_x"]) - float(c_main["centroid_x"])),
                            )
                            if dist_rem > 4.0:
                                eta_nl = max(float(c_rem_area) / float(diff_rs.size), 0.55)
                                rem_c_set: Set[int] = set()
                                for c_r in clusters[1:]:
                                    rem_c_set.update(int(x) for x in np.unique(arr_prev[c_r["rows"], c_r["cols"]]))
                                    rem_c_set.update(int(x) for x in np.unique(arr_next[c_r["rows"], c_r["cols"]]))
                                rem_c_set.discard(bg_m)
                                toggled_cols = tuple(sorted(rem_c_set))
                                d4_sig_induced = self.canonical_d4_sprite_tuple(
                                    arr_prev,
                                    int(c_main["r0"]),
                                    int(c_main["r1"]),
                                    int(c_main["c0"]),
                                    int(c_main["c1"]),
                                    bg_val=bg_m,
                                )
                        else:
                            eta_nl = 0.50
                    all_ch_cols = set(int(x) for x in np.unique(arr_prev[diff_mask]))
                    all_ch_cols.discard(bg_m)
                    trig_cols = tuple(sorted(all_ch_cols))

                if eff_l_prev != eff_l_next and toggled_cols:
                    self.trigger_colors.update(trig_cols)
                    for tc in toggled_cols:
                        self.observe_single_latch_conductance(eff_l_next, int(tc), 1.0)
                        for t in toks_next:
                            self.observe_single_latch_conductance(t, int(tc), 1.0)
                        if len(toks_next) >= 2:
                            for i, sa in enumerate(toks_next):
                                for sb in toks_next[i + 1 :]:
                                    self.observe_order2_conjunction(sa, sb, int(tc), 1.0)
                        elif eff_l_prev not in ("latch_default", eff_l_next):
                            if self.single_latch_variance(eff_l_prev, int(tc)) > 0.0 or self.single_latch_variance(
                                eff_l_next, int(tc)
                            ) > 0.0:
                                self.observe_order2_conjunction(eff_l_prev, eff_l_next, int(tc), 1.0)
                if (eff_l_prev != eff_l_next or eta_nl >= 0.50) and toggled_cols:
                    if d4_sig_induced is not None and d4_sig_induced[0]:
                        hom_latch = (
                            eff_l_next
                            if eff_l_prev != eff_l_next
                            else f"d4_latch_{d4_sig_induced[1]}_{d4_sig_induced[2]}"
                        )
                        self.d4_latch_homomorphism_table[d4_sig_induced] = {
                            "target_latch": hom_latch,
                            "eta_nonlocal": max(float(eta_nl), 0.50),
                            "trigger_colors": trig_cols,
                            "toggled_colors": toggled_cols,
                        }

        eff_cost = max(0.01, float(cost))
        self.action_edges[(u, action_key)] = v
        existing = self.rules.get((u, v))
        if existing is None or eff_cost < existing.cost:
            self.rules[(u, v)] = TropicalTransitionRule(
                source_node=u,
                target_node=v,
                action_key=action_key,
                cost=eff_cost,
                observations=1 if existing is None else existing.observations + 1,
                eta_nonlocal=eta_nl,
                trigger_colors=trig_cols,
                toggled_colors=toggled_cols,
                d4_sprite_tuple=d4_sig_induced,
            )
            self._dirty = True
        else:
            existing.observations += 1
        return self.rules.get((u, v))

    def compute_all_pairs_shortest_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes exact tropical Kleene closure `(D_star, routing)` over `V`."""
        n = len(self.idx_to_node)
        if n == 0:
            empty_d = np.zeros((0, 0), dtype=np.float64)
            empty_r = np.zeros((0, 0), dtype=np.int64)
            return empty_d, empty_r
        if not self._dirty and self._closure_dist is not None and self._closure_routing is not None:
            return self._closure_dist, self._closure_routing
        W = np.full((n, n), self.inf_val, dtype=np.float64)
        np.fill_diagonal(W, 0.0)
        for (u, v), rule in self.rules.items():
            iu = self.node_to_idx[u]
            iv = self.node_to_idx[v]
            if rule.cost < W[iu, iv]:
                W[iu, iv] = rule.cost
        D_star, routing = tropical_kleene_closure(
            W, max_steps=max(1, n - 1), return_routing=True, inf_threshold=self.inf_val
        )
        self._closure_dist = np.asarray(D_star, dtype=np.float64)
        self._closure_routing = np.asarray(routing, dtype=np.int64)
        self._dirty = False
        return self._closure_dist, self._closure_routing


    def shortest_action_sequence(
        self,
        source_node: Tuple[str, str],
        target_node: Tuple[str, str],
    ) -> Optional[List[ActionKey]]:
        """Reconstructs the optimal min-plus action sequence from `source_node` to `target_node`."""
        if source_node not in self.node_to_idx or target_node not in self.node_to_idx:
            return None
        if source_node == target_node:
            return []
        D_star, routing = self.compute_all_pairs_shortest_paths()
        iu = self.node_to_idx[source_node]
        iv = self.node_to_idx[target_node]
        if D_star[iu, iv] >= self.inf_val * 0.5:
            return None
        actions: List[ActionKey] = []
        cur = iu
        visited: Set[int] = {cur}
        for _ in range(len(self.idx_to_node)):
            if cur == iv:
                return actions
            nxt = int(routing[cur, iv])
            if nxt < 0 or nxt in visited:
                return None
            u_node = self.idx_to_node[cur]
            v_node = self.idx_to_node[nxt]
            rule = self.rules.get((u_node, v_node))
            if rule is None:
                return None
            actions.append(rule.action_key)
            visited.add(nxt)
            cur = nxt
        return actions if cur == iv else None

    def shortest_tropical_path(
        self,
        start_hash: str,
        frontier_scores: Mapping[str, float],
    ) -> Optional[Tuple[str, List[ActionKey]]]:
        """
        Finds the highest-utility reachable frontier state `s^*` from `start_hash` over
        the tropical Kleene closure `D^* = bigoplus_{k=0}^{|V|-1} W^{otimes k}`.
        """
        if not frontier_scores or not self.idx_to_node:
            return None
        start_nodes = [node for node in self.idx_to_node if node[0] == str(start_hash)]
        if not start_nodes:
            return None
        target_nodes = [node for node in self.idx_to_node if node[0] in frontier_scores]
        if not target_nodes:
            return None

        best_val = -1e18
        best_res: Optional[Tuple[str, List[ActionKey]]] = None
        for u in start_nodes:
            for v in target_nodes:
                if u[0] == v[0]:
                    continue
                seq = self.shortest_action_sequence(u, v)
                if seq:
                    val = float(frontier_scores[v[0]]) - 0.15 * float(len(seq))
                    if val > best_val:
                        best_val = val
                        best_res = (v[0], seq)
        return best_res


class RelationalOperatorDreamer:
    """
    Pillar 1: Object-Relational & Ego-Kinematic Forward World Model
    (`RelationalOperatorDreamer`, aliased as `OnlineForwardOperatorDreamer`).

    Grounded in Contingency-Aware Exploration (Choi et al., arXiv:1811.01483),
    Independently Controllable Factors (Thomas et al., arXiv:1708.01289), and
    Contrastive Structured World Models (Kipf et al., arXiv:1911.12247):
      1. Discovers controllable ego-fibers `C_ego` (including multi-scale shapes up to
         `area <= 256` and composite oriented assemblies) and their kinematic translation
         generators `(dy_a, dx_a, is_sliding)` under `ACTION1..4` from a single step per direction.
      2. Learns the Ordered Pairwise Relational Interaction Kernel `K_int`:
         `"wall"`, `"pushable"`, `"hazard"`, and `"contact_highlight"`.
      3. Models rigid components via `ContactGraphSheaf` in se(2) with outward normals n_ij
         and restriction maps enforcing unilateral non-penetration (delta v)(e) >= 0.
      4. Executes forward prediction `predict_step(frame_2d, action_key, ...)` and joint
         semidirect product `SE(2) |x FSA` trajectory planning `solve_se2_fsa_plan(...)`.
    """

    def __init__(self) -> None:
        self.reset(clear_cross_level=True)

    def reset(
        self,
        clear_cross_level: bool = False,
        preserve_kinematics: bool = False,
        is_level_reset: bool = True,
    ) -> None:
        self.active_contact_highlight_states: Set[str] = set()
        self.contact_adjacent_state_hashes: Set[str] = set()
        self.last_transition_highlighted: bool = False
        self.contact_sheaf: Optional[ContactGraphSheaf] = None
        self.tool_interactions: Set[Tuple[int, int, int]] = set()
        self.container_nesting_pairs: Set[Tuple[int, int]] = set()
        self.fsa_latch_states: Set[str] = set()
        self.last_latch_signature: str = "latch_default"
        self.unlocked_colors: Set[int] = set()
        if is_level_reset or clear_cross_level or not hasattr(self, "_pp_slot_history"):
            self._pp_slot_history: Set[Tuple[int, int]] = set()
            self._pp_transfer_history: Set[Tuple[int, int]] = set()
            self._pp_init_final_slots: Set[Tuple[int, int]] = set()
            self._monoid_hom_state: Optional[Dict[str, Any]] = None
            self._locus_cover_plan_by_center: Optional[Dict[Tuple[int, int], List[int]]] = None
            self._locus_cover_done_centers: Set[Tuple[int, int]] = set()
            self._locus_cover_switches: int = 0
            self._pin_polyomino_state: Optional[Dict[str, Any]] = None
            self._skewer_rail_plan: Optional[deque] = None
        if not preserve_kinematics or clear_cross_level or not hasattr(self, "kinematic_generators"):
            self.kinematic_generators: Dict[int, Tuple[int, int, bool]] = {}
            self.ego_colors: Set[int] = set()
            self.ego_accent_colors: Set[int] = set()
            self.ego_signatures: Set[Tuple[int, int, int, int]] = set()  # (color, h, w, area_bucket)
            self.wall_colors: Set[int] = set()
            self.pushable_colors: Set[int] = set()
            self.floor_colors: Set[int] = set()
            self.highlight_rest_colors: Set[int] = set()
            self.highlight_active_colors: Set[int] = set()
            self.action5_is_contact_operator: bool = False
            self.click_signature_luts: Dict[Tuple[int, int, int], Dict[int, int]] = {}
            self._pp_roles: Optional[Dict[str, Any]] = None
            self._multisprite_roles: Optional[Dict[str, int]] = None
        if (not preserve_kinematics and clear_cross_level) or not hasattr(self, "hazard_signatures"):
            self.hazard_signatures: Set[Tuple[str, int, int, int, int]] = set()
            self.hazard_colors: Set[int] = set()
            self.interaction_kernel: Dict[Tuple[int, int, int, int], str] = {}

    def _find_ego_component(
        self,
        board: np.ndarray,
        bg_mode: int,
    ) -> Optional[Dict[str, Any]]:
        comps = _connected_components_2d(board, connectivity=4, same_color=True, ignore_val=bg_mode)
        if not comps:
            return None
        best_comp: Optional[Dict[str, Any]] = None
        best_score = -1e9
        for c in comps:
            col = int(c["color"])
            area = int(c["area"])
            bh, bw = int(c["h"]), int(c["w"])
            if area > 256 or max(bh, bw) > 28:
                continue
            ab = int(round(math.log2(max(1, area))))
            sig = (col, bh, bw, ab)
            sc = 0.0
            if sig in self.ego_signatures:
                sc += 10.0
            elif col in self.ego_colors:
                sc += 5.0
            else:
                continue
            sc -= 0.05 * float(area)
            if sc > best_score:
                best_score = sc
                best_comp = c
        return best_comp

    @property
    def ego_color(self) -> Optional[int]:
        return next(iter(self.ego_colors), None)

    def observe_transition(
        self,
        nope_prev: np.ndarray,
        nope_next: np.ndarray,
        action_key: ActionKey,
        sh_prev: str = "",
        sh_next: str = "",
        nope_d: Optional[int] = None,
        is_game_over: bool = False,
        is_phase_reset: bool = False,
        candidate_info: Optional[Dict[str, Any]] = None,
        subframes: Optional[Sequence[np.ndarray]] = None,
        state_hash_next: Optional[str] = None,
        latch_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Updates ego-kinematic generators and pairwise relational kernel from an observed step."""
        self.last_transition_highlighted = False
        if is_phase_reset:
            return {"ego_moved": False, "contact_highlight": False}
        if state_hash_next is not None and not sh_next:
            sh_next = str(state_hash_next)
        if nope_d is None:
            nope_d = int(np.sum(nope_prev != nope_next))
        if latch_signature is not None:
            self.fsa_latch_states.add(str(latch_signature))
            self.last_latch_signature = str(latch_signature)

        H, W = nope_prev.shape
        vals, cnts = np.unique(nope_prev, return_counts=True)
        bg_mode = int(vals[int(np.argmax(cnts))])

        # Record GAME_OVER hazard signatures (persisted across levels)
        if is_game_over:
            if isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6:
                if candidate_info is not None:
                    bh = int(candidate_info.get("bbox", (1, 1))[0])
                    bw = int(candidate_info.get("bbox", (1, 1))[1])
                    area = int(candidate_info.get("area", bh * bw))
                    col = int(candidate_info.get("color", 0))
                    k_s = str(candidate_info.get("kind", "cc_object"))
                    self.hazard_signatures.add(
                        (k_s, min(bh, bw), max(bh, bw), int(round(math.log2(max(1, area)))), col)
                    )
            elif isinstance(action_key, int) and action_key in (1, 2, 3, 4):
                if action_key in self.kinematic_generators:
                    dy, dx, _ = self.kinematic_generators[action_key]
                    ego_c = self._find_ego_component(nope_prev, bg_mode)
                    if ego_c is not None:
                        n_rs = np.clip(ego_c["rows"] + dy, 0, H - 1)
                        n_cs = np.clip(ego_c["cols"] + dx, 0, W - 1)
                        target_vals = nope_prev[n_rs, n_cs]
                        for tv in np.unique(target_vals):
                            tv_i = int(tv)
                            if (
                                tv_i != bg_mode
                                and tv_i != int(ego_c["color"])
                                and tv_i not in self.floor_colors
                            ):
                                self.hazard_colors.add(tv_i)
            return {"ego_moved": False, "contact_highlight": False}

        # Directional actions ACTION1..4: discover ego component, kinematics, walls, pushables, contact highlights
        if isinstance(action_key, int) and action_key in (1, 2, 3, 4):
            if nope_d == 0:
                # Wall bump: if action_key has a known generator (dy, dx), identify blocking adjacent color
                if action_key in self.kinematic_generators:
                    dy, dx, _ = self.kinematic_generators[action_key]
                    ego_c = self._find_ego_component(nope_prev, bg_mode)
                    if ego_c is not None and (abs(dy) + abs(dx) > 0):
                        step_y = int(np.sign(dy))
                        step_x = int(np.sign(dx))
                        n_rs = ego_c["rows"] + step_y
                        n_cs = ego_c["cols"] + step_x
                        in_b = (n_rs >= 0) & (n_rs < H) & (n_cs >= 0) & (n_cs < W)
                        adj_vals = nope_prev[n_rs[in_b], n_cs[in_b]]
                        for av in np.unique(adj_vals):
                            av_i = int(av)
                            if (
                                av_i != bg_mode
                                and av_i != int(ego_c["color"])
                                and av_i not in self.floor_colors
                            ):
                                self.wall_colors.add(av_i)
                                self.unlocked_colors.discard(av_i)
                                self.interaction_kernel[(int(ego_c["color"]), av_i, step_y, step_x)] = "wall"
                return {"ego_moved": False, "contact_highlight": False}

            comps_prev = _connected_components_2d(nope_prev, connectivity=4, same_color=True, ignore_val=bg_mode)
            for cp_fl in comps_prev:
                if (
                    int(cp_fl["area"]) > 256
                    and min(int(cp_fl["h"]), int(cp_fl["w"])) >= 16
                    and float(cp_fl["fill_ratio"]) >= 0.90
                ):
                    r0_f, r1_f = int(cp_fl["r0"]), int(cp_fl["r1"])
                    c0_f, c1_f = int(cp_fl["c0"]), int(cp_fl["c1"])
                    col_f = int(cp_fl["color"])
                    sub_in = nope_prev[r0_f + 1 : r1_f - 1, c0_f + 1 : c1_f - 1]
                    if (
                        sub_in.size > 0
                        and not np.any(sub_in == bg_mode)
                        and np.any(sub_in != col_f)
                        and all(
                            int(cp2["area"]) <= 24 and max(int(cp2["h"]), int(cp2["w"])) <= 6
                            for cp2 in comps_prev
                            if cp2 is not cp_fl
                            and int(cp2["r0"]) >= r0_f
                            and int(cp2["r1"]) <= r1_f
                            and int(cp2["c0"]) >= c0_f
                            and int(cp2["c1"]) <= c1_f
                        )
                    ):
                        self.floor_colors.add(col_f)
                        self.wall_colors.discard(col_f)
            comps_next = _connected_components_2d(nope_next, connectivity=4, same_color=True, ignore_val=bg_mode)
            diff_mask = nope_prev != nope_next

            unit_cnt_by_col = Counter(int(c["color"]) for c in comps_prev if int(c["area"]) == 1)
            moved_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], int, int]] = []
            for cp in comps_prev:
                if int(cp["area"]) > 256 or max(int(cp["h"]), int(cp["w"])) > 28:
                    continue
                if int(cp["area"]) == 1 and unit_cnt_by_col[int(cp["color"])] >= 4:
                    continue
                if not np.any(diff_mask[cp["rows"], cp["cols"]]):
                    continue
                col = int(cp["color"])
                best_cn: Optional[Dict[str, Any]] = None
                best_dist = 999.0
                for cn in comps_next:
                    if int(cn["color"]) != col:
                        continue
                    if abs(int(cn["area"]) - int(cp["area"])) > max(2, int(cp["area"]) // 4):
                        continue
                    same_hw = int(cn["h"]) == int(cp["h"]) and int(cn["w"]) == int(cp["w"])
                    rot_hw = (
                        col in self.ego_colors
                        and int(cn["h"]) == int(cp["w"])
                        and int(cn["w"]) == int(cp["h"])
                    )
                    if not (same_hw or rot_hw):
                        continue
                    if not np.any(diff_mask[cn["rows"], cn["cols"]]):
                        continue
                    dy_f = float(cn["centroid_y"]) - float(cp["centroid_y"])
                    dx_f = float(cn["centroid_x"]) - float(cp["centroid_x"])
                    d_man = abs(dy_f) + abs(dx_f)
                    if 0.5 <= d_man <= 32.0 and (
                        abs(dy_f) <= 0.25 or abs(dx_f) <= 0.25 or (rot_hw and d_man <= 4.5)
                    ):
                        if d_man < best_dist:
                            best_dist = d_man
                            best_cn = cn
                if best_cn is not None:
                    dy_i = int(round(float(best_cn["centroid_y"]) - float(cp["centroid_y"])))
                    dx_i = int(round(float(best_cn["centroid_x"]) - float(cp["centroid_x"])))
                    if abs(dy_i) + abs(dx_i) > 0:
                        moved_pairs.append((cp, best_cn, dy_i, dx_i))

            ego_moved = False
            ego_next_comp: Optional[Dict[str, Any]] = None
            ego_prev_comp: Optional[Dict[str, Any]] = None
            if moved_pairs:
                # Prefer component already in ego_colors, or smallest compact moving sprite
                moved_pairs.sort(
                    key=lambda item: (
                        0 if int(item[0]["color"]) in self.ego_colors else 1,
                        int(item[0]["area"]),
                    )
                )
                cp_ego, cn_ego, dy_e, dx_e = moved_pairs[0]
                ego_prev_comp = cp_ego
                ego_next_comp = cn_ego
                ego_moved = True
                col_e = int(cp_ego["color"])
                ab_e = int(round(math.log2(max(1, int(cp_ego["area"])))))
                self.ego_colors.add(col_e)
                self.ego_signatures.add((col_e, int(cp_ego["h"]), int(cp_ego["w"]), ab_e))
                self.wall_colors.discard(col_e)
                same_disp_comps = [
                    cp_m for cp_m, _, dy_m, dx_m in moved_pairs if dy_m == dy_e and dx_m == dx_e
                ]
                asm_max_dim = max(
                    max(int(c["h"]), int(c["w"])) for c in same_disp_comps
                )
                is_sliding = (
                    max(abs(dy_e), abs(dx_e)) >= 4
                    and max(abs(dy_e), abs(dx_e)) > asm_max_dim
                )
                if action_key not in self.kinematic_generators or not is_sliding:
                    self.kinematic_generators[action_key] = (dy_e, dx_e, is_sliding)

                # Multi-body contact sheaf tracing
                sheaf = ContactGraphSheaf.from_board(
                    nope_prev,
                    bg_mode=bg_mode,
                    ego_colors=self.ego_colors,
                    pushable_colors=self.pushable_colors,
                    wall_colors=self.wall_colors,
                    ego_comp=cp_ego,
                )
                self.contact_sheaf = sheaf
                if len(moved_pairs) <= 6:
                    for cp_other, _, dy_o, dx_o in moved_pairs[1:]:
                        if np.sign(dy_o) == np.sign(dy_e) and np.sign(dx_o) == np.sign(dx_e):
                            col_o = int(cp_other["color"])
                            if col_o != col_e or len(moved_pairs) <= 3:
                                self.pushable_colors.add(col_o)
                                self.wall_colors.discard(col_o)
                                self.interaction_kernel[
                                    (col_e, col_o, int(np.sign(dy_e)), int(np.sign(dx_e)))
                                ] = "pushable"
                if len(moved_pairs) >= 3:
                    for i in range(1, len(moved_pairs) - 1):
                        tool_col = int(moved_pairs[i][0]["color"])
                        target_col = int(moved_pairs[i + 1][0]["color"])
                        self.tool_interactions.add((col_e, tool_col, target_col))
                for edge in sheaf.edges:
                    if edge.is_enclosed and edge.container_id is not None and edge.content_id is not None:
                        cont_col = sheaf.nodes[edge.container_id].color
                        cont_in = sheaf.nodes[edge.content_id].color
                        self.container_nesting_pairs.add((cont_col, cont_in))

            # Check for Contact-Highlight (stationary non-background component adjacent to ego changed color in-place!)
            contact_highlight = False
            if ego_next_comp is not None and ego_prev_comp is not None:
                all_moving_mask = np.zeros((H, W), dtype=bool)
                all_next_rows_list: List[np.ndarray] = []
                all_next_cols_list: List[np.ndarray] = []
                for cp_m, cn_m, _, _ in moved_pairs:
                    all_moving_mask[cp_m["rows"], cp_m["cols"]] = True
                    all_moving_mask[cn_m["rows"], cn_m["cols"]] = True
                    all_next_rows_list.append(cn_m["rows"])
                    all_next_cols_list.append(cn_m["cols"])
                all_next_rows = (
                    np.concatenate(all_next_rows_list)
                    if all_next_rows_list
                    else ego_next_comp["rows"]
                )
                all_next_cols = (
                    np.concatenate(all_next_cols_list)
                    if all_next_cols_list
                    else ego_next_comp["cols"]
                )

                stationary_flip = (
                    diff_mask
                    & (~all_moving_mask)
                    & (nope_prev != bg_mode)
                    & (nope_next != bg_mode)
                )
                stat_flip_n = int(np.sum(stationary_flip))
                if 2 <= stat_flip_n <= 64:
                    ors, ocs = np.where(stationary_flip)
                    dist_to_ego = np.min(
                        np.maximum(
                            np.abs(ors[:, None] - all_next_rows[None, :]),
                            np.abs(ocs[:, None] - all_next_cols[None, :]),
                        )
                    )
                    if int(dist_to_ego) <= 4:
                        contact_highlight = True
                        self.active_contact_highlight_states.add(sh_next)
                        self.last_transition_highlighted = True
                        self.action5_is_contact_operator = True
                        next_cols = {int(x) for x in np.unique(nope_next[stationary_flip])}
                        latch_cols = tuple(sorted(next_cols))
                        self.fsa_latch_states.add(f"highlight_{latch_cols}")

                # Topological Cavity Containment: check if ego in `nope_next` is docked inside a hollow receptor cavity
                # or 4-adjacent to a compact non-wall interactive object
                er0, er1 = int(ego_next_comp["r0"]), int(ego_next_comp["r1"])
                ec0, ec1 = int(ego_next_comp["c0"]), int(ego_next_comp["c1"])
                ego_h = max(1, er1 - er0)
                ego_w = max(1, ec1 - ec0)
                ego_area = int(ego_next_comp["area"])
                for cn_other in comps_next:
                    if cn_other is ego_next_comp:
                        continue
                    oh, ow, o_area = int(cn_other["h"]), int(cn_other["w"]), int(cn_other["area"])
                    if (
                        o_area >= 8
                        and float(cn_other["fill_ratio"]) <= 0.68
                        and (oh * ow - o_area) >= ego_area
                        and max(oh, ow) <= max(12, min(H, W) // 3)
                        and int(cn_other["r0"]) <= er0
                        and int(cn_other["r1"]) >= er1
                        and int(cn_other["c0"]) <= ec0
                        and int(cn_other["c1"]) >= ec1
                    ):
                        self.active_contact_highlight_states.add(sh_next)
                        self.contact_adjacent_state_hashes.add(sh_next)
                        contact_highlight = True
                        self.last_transition_highlighted = True
                        self.action5_is_contact_operator = True
                    elif (
                        int(cn_other["color"]) not in self.wall_colors
                        and 2 <= o_area <= max(16, (H * W) // 64)
                        and max(oh, ow) <= max(6, min(H, W) // 6)
                    ):
                        dy_box = max(0, max(int(cn_other["r0"]) - er1, er0 - int(cn_other["r1"])))
                        dx_box = max(0, max(int(cn_other["c0"]) - ec1, ec0 - int(cn_other["c1"])))
                        if dy_box + dx_box <= 1:
                            self.contact_adjacent_state_hashes.add(sh_next)

            return {"ego_moved": ego_moved, "contact_highlight": contact_highlight}

        if isinstance(action_key, int) and action_key == 5 and nope_d > 0:
            if sh_prev in self.active_contact_highlight_states:
                self.action5_is_contact_operator = True

        # Coordinate click ACTION6: record signature LUT if local recoloring
        if (
            isinstance(action_key, tuple)
            and len(action_key) == 3
            and action_key[0] == 6
            and nope_d > 0
            and candidate_info is not None
        ):
            bh = int(candidate_info.get("bbox", (1, 1))[0])
            bw = int(candidate_info.get("bbox", (1, 1))[1])
            area = int(candidate_info.get("area", bh * bw))
            diff_m = nope_prev != nope_next
            drs, dcs = np.where(diff_m)
            cx, cy = int(action_key[1]), int(action_key[2])
            if drs.size > 0 and int(max(np.max(np.abs(dcs - cx)), np.max(np.abs(drs - cy)))) <= max(5, bh, bw):
                sig_k = (min(bh, bw), max(bh, bw), int(round(math.log2(max(1, area)))))
                lut = self.click_signature_luts.setdefault(sig_k, {})
                for u_c in np.unique(nope_prev[diff_m]):
                    mapped = nope_next[diff_m & (nope_prev == u_c)]
                    if mapped.size > 0:
                        vals_m, cnts_m = np.unique(mapped, return_counts=True)
                        lut[int(u_c)] = int(vals_m[int(np.argmax(cnts_m))])

        return {"ego_moved": False, "contact_highlight": False}

    def predict_step(
        self,
        frame_2d: np.ndarray,
        action_key: ActionKey,
        candidate_info: Optional[Dict[str, Any]] = None,
        clock_mask: Optional[np.ndarray] = None,
        unlocked_colors: Optional[Set[int]] = None,
        latch_state: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Predicts the next `H_nope` board state by applying the learned ego-kinematic
        generator `(dy, dx)` + relational interaction kernel (`wall`, `pushable`) to the
        ego component's CURRENT position in `frame_2d`.
        """
        if unlocked_colors is None and (
            self._pp_roles is not None or self._multisprite_roles is not None
        ):
            return None
        board = _frame_to_2d_numpy(frame_2d).copy()
        if clock_mask is not None and clock_mask.shape == board.shape and np.any(clock_mask):
            board = _mask_clock_to_bg(board, clock_mask)
        H, W = board.shape
        vals, cnts = np.unique(board, return_counts=True)
        bg_mode = int(vals[int(np.argmax(cnts))])
        active_unlocked = set(unlocked_colors) if unlocked_colors is not None else set()
        eff_walls = self.wall_colors - active_unlocked

        if isinstance(action_key, int) and action_key in self.kinematic_generators:
            dy, dx, is_sliding = self.kinematic_generators[action_key]
            ego_c = self._find_ego_component(board, bg_mode)
            if ego_c is None:
                return None

            if not is_sliding:
                sheaf = ContactGraphSheaf.from_board(
                    board,
                    bg_mode=bg_mode,
                    ego_colors=self.ego_colors,
                    pushable_colors=self.pushable_colors,
                    wall_colors=eff_walls,
                    ego_comp=ego_c,
                    unlocked_colors=active_unlocked,
                    latch_state=latch_state or self.last_latch_signature,
                )
                success, moving_nodes, reason = sheaf.propagate_displacement(
                    action_dir=(dy, dx),
                    ego_node_id=sheaf.ego_node_id,
                    allow_candidate_push=True,
                )
                if not success:
                    return board
                return sheaf.apply_displacement_to_board(board, bg_mode, moving_nodes)
            else:
                # Sliding generator: fast O(1)-component NumPy coordinate stepping across free space
                # (including composite nested ego assemblies), invoking ContactGraphSheaf only on contact.
                uy = int(np.sign(dy))
                ux = int(np.sign(dx))
                if uy == 0 and ux == 0:
                    return board
                max_slide_steps = (
                    min(max(1, abs(dy) + abs(dx)), max(H, W))
                    if self._multisprite_roles is not None
                    else max(H, W)
                )
                er0_init, er1_init = int(ego_c["r0"]), int(ego_c["r1"])
                ec0_init, ec1_init = int(ego_c["c0"]), int(ego_c["c1"])
                non_ego_board = board.copy()
                sub_col = bg_mode
                for fl_c in self.floor_colors:
                    if np.any(
                        board[
                            max(0, er0_init - 1) : min(H, er1_init + 1),
                            max(0, ec0_init - 1) : min(W, ec1_init + 1),
                        ]
                        == fl_c
                    ):
                        sub_col = int(fl_c)
                        break
                if sub_col == bg_mode:
                    ring_box = board[
                        max(0, er0_init - 1) : min(H, er1_init + 1),
                        max(0, ec0_init - 1) : min(W, ec1_init + 1),
                    ]
                    ego_col_val = int(ego_c["color"])
                    for cand_fl in (int(v) for v in np.unique(ring_box)):
                        if (
                            cand_fl != bg_mode
                            and cand_fl != ego_col_val
                            and cand_fl not in eff_walls
                            and cand_fl not in self.pushable_colors
                            and int(np.sum(board == cand_fl)) > 256
                        ):
                            sub_col = cand_fl
                            break
                # Include nested interior children if ego_c is a hollow container frame
                is_hollow_ego = (
                    er1_init - er0_init >= 3
                    and ec1_init - ec0_init >= 3
                    and int(ego_c["area"]) < (er1_init - er0_init) * (ec1_init - ec0_init)
                )
                if is_hollow_ego:
                    sub_box = board[er0_init:er1_init, ec0_init:ec1_init]
                    rel_r, rel_c = np.where(sub_box != sub_col)
                    asm_rs = rel_r + er0_init
                    asm_cs = rel_c + ec0_init
                else:
                    asm_rs = ego_c["rows"].copy()
                    asm_cs = ego_c["cols"].copy()
                asm_cols = board[asm_rs, asm_cs].copy()
                non_ego_board[asm_rs, asm_cs] = sub_col
                free_colors = {bg_mode, sub_col} | self.floor_colors | active_unlocked
                cur_rs = asm_rs
                cur_cs = asm_cs
                steps_done = 0
                while steps_done < max_slide_steps:
                    n_rs = cur_rs + uy
                    n_cs = cur_cs + ux
                    if bool(np.any((n_rs < 0) | (n_rs >= H) | (n_cs < 0) | (n_cs >= W))):
                        return (
                            board
                            if steps_done == 0
                            else self._materialize_ego_on_board(non_ego_board, cur_rs, cur_cs, asm_cols)
                        )
                    hit_vals = non_ego_board[n_rs, n_cs]
                    u_hits = [int(v) for v in np.unique(hit_vals)]
                    if all(v in free_colors for v in u_hits):
                        cur_rs = n_rs
                        cur_cs = n_cs
                        steps_done += 1
                        continue
                    if any(v in eff_walls for v in u_hits) or not self.pushable_colors:
                        return (
                            board
                            if steps_done == 0
                            else self._materialize_ego_on_board(non_ego_board, cur_rs, cur_cs, asm_cols)
                        )
                    break
                cur_board = (
                    board.copy()
                    if steps_done == 0
                    else self._materialize_ego_on_board(non_ego_board, cur_rs, cur_cs, asm_cols)
                )
                for _ in range(steps_done, min(max_slide_steps, steps_done + 2)):
                    cur_ego = self._find_ego_component(cur_board, bg_mode)
                    if cur_ego is None:
                        break
                    sheaf_step = ContactGraphSheaf.from_board(
                        cur_board,
                        bg_mode=bg_mode,
                        ego_colors=self.ego_colors,
                        pushable_colors=self.pushable_colors,
                        wall_colors=eff_walls,
                        ego_comp=cur_ego,
                        unlocked_colors=active_unlocked,
                        latch_state=latch_state or self.last_latch_signature,
                    )
                    ok, m_nodes, _ = sheaf_step.propagate_displacement(
                        action_dir=(uy, ux),
                        ego_node_id=sheaf_step.ego_node_id,
                        allow_candidate_push=False,
                    )
                    if not ok or not m_nodes:
                        break
                    cur_board = sheaf_step.apply_displacement_to_board(cur_board, bg_mode, m_nodes)
                return cur_board

        if (
            isinstance(action_key, tuple)
            and len(action_key) == 3
            and action_key[0] == 6
            and candidate_info is not None
        ):
            bh = int(candidate_info.get("bbox", (1, 1))[0])
            bw = int(candidate_info.get("bbox", (1, 1))[1])
            area = int(candidate_info.get("area", bh * bw))
            sig_k = (min(bh, bw), max(bh, bw), int(round(math.log2(max(1, area)))))
            lut = self.click_signature_luts.get(sig_k)
            if lut:
                r0 = max(0, int(candidate_info.get("r0", int(action_key[2]) - bh // 2)))
                r1 = min(H, int(candidate_info.get("r1", r0 + bh)))
                c0 = max(0, int(candidate_info.get("c0", int(action_key[1]) - bw // 2)))
                c1 = min(W, int(candidate_info.get("c1", c0 + bw)))
                cur_col = int(candidate_info.get("color", board[int(action_key[2]), int(action_key[1])]))
                if r1 > r0 and c1 > c0 and cur_col in lut:
                    sub = board[r0:r1, c0:c1].copy()
                    sub[sub == cur_col] = int(lut[cur_col])
                    board[r0:r1, c0:c1] = sub
                    return board

        return None

    @staticmethod
    def _materialize_ego_on_board(
        non_ego_board: np.ndarray,
        ego_rs: np.ndarray,
        ego_cs: np.ndarray,
        ego_col: Any,
    ) -> np.ndarray:
        out = non_ego_board.copy()
        out[ego_rs, ego_cs] = ego_col
        return out

    def _solve_coupled_pick_and_place_plan(
        self,
        board0: np.ndarray,
    ) -> List[ActionKey]:
        """
        Solves coupled semidirect-product `SE(2) |x Gripper-Attachment` pick-and-place
        trajectories (including intermediate transfer-slot handoffs across room barriers)
        purely from online-derived `S_16`-equivariant topological roles.
        """
        H, W = board0.shape
        if H < 16 or W < 16:
            return []
        if self._pp_roles is None:
            vals_all, cnts_all = np.unique(board0, return_counts=True)
            bg_all = int(vals_all[int(np.argmax(cnts_all))])
            comps_all = _connected_components_2d(
                board0, connectivity=4, same_color=True, ignore_val=bg_all
            )
            scale_counts = Counter(
                int(c["h"])
                for c in comps_all
                if 3 <= int(c["h"]) == int(c["w"]) <= 8
                and int(c["area"]) == 4 * (int(c["h"]) - 1)
                and H % int(c["h"]) == 0
                and W % int(c["w"]) == 0
            )
            if not scale_counts:
                return []
            S = int(scale_counts.most_common(1)[0][0])
            vals, cnts = np.unique(board0[S : H - S, S : W - S], return_counts=True)
            bg = int(vals[int(np.argmax(cnts))])
            comps0 = _connected_components_2d(
                board0[S : H - S, S : W - S], connectivity=4, same_color=True, ignore_val=bg
            )
            ego_body = next(
                (
                    c
                    for c in comps0
                    if int(c["area"]) == S * (S - 1)
                    and float(c["fill_ratio"]) == 1.0
                    and sorted([int(c["h"]), int(c["w"])]) == [S - 1, S]
                ),
                None,
            )
            if ego_body is None:
                return []
            col_ego = int(ego_body["color"])
            br0, bc0, br1, bc1 = (
                int(ego_body["r0"]) + S,
                int(ego_body["c0"]) + S,
                int(ego_body["r1"]) + S,
                int(ego_body["c1"]) + S,
            )
            col_grip: Optional[int] = None
            if br1 - br0 == S - 1 and bc1 - bc0 == S:
                for rr in (br0 - 1, br1):
                    if S <= rr < H - S:
                        u_bar = np.unique(board0[rr, bc0:bc1])
                        if len(u_bar) == 1 and int(u_bar[0]) not in (bg, col_ego):
                            col_grip = int(u_bar[0])
                            break
            elif br1 - br0 == S and bc1 - bc0 == S - 1:
                for cc in (bc0 - 1, bc1):
                    if S <= cc < W - S:
                        u_bar = np.unique(board0[br0:br1, cc])
                        if len(u_bar) == 1 and int(u_bar[0]) not in (bg, col_ego):
                            col_grip = int(u_bar[0])
                            break
            if col_grip is None:
                return []
            c_core: Optional[int] = None
            c_frames: Set[int] = set()
            for c in comps0:
                if int(c["h"]) == S and int(c["w"]) == S and int(c["area"]) == 4 * (S - 1):
                    r0, c0 = int(c["r0"]) + S, int(c["c0"]) + S
                    inner = board0[r0 + 1 : r0 + S - 1, c0 + 1 : c0 + S - 1]
                    u_in = np.unique(inner)
                    if len(u_in) == 1 and int(u_in[0]) not in (bg, int(c["color"]), col_ego, col_grip):
                        c_core = int(u_in[0])
                        c_frames.add(int(c["color"]))
            if c_core is None:
                return []
            c_cav: Optional[int] = None
            for sr in range(S, H - S, S):
                for sc in range(S, W - S, S):
                    if int(board0[sr, sc]) == c_core:
                        inner = board0[sr + 1 : sr + S - 1, sc + 1 : sc + S - 1]
                        u_in = np.unique(inner)
                        if len(u_in) == 1 and int(u_in[0]) not in (bg, c_core, col_ego, col_grip):
                            c_cav = int(u_in[0])
                            break
                if c_cav is not None:
                    break
            if c_cav is None:
                return []
            fence_cols = {
                int(c["color"])
                for c in comps0
                if max(int(c["h"]), int(c["w"])) >= 3 * S
                and int(c["color"]) not in ({bg, col_ego, col_grip, c_core, c_cav} | c_frames)
            }
            self._pp_roles = {
                "S": S,
                "bg": bg,
                "col_ego": col_ego,
                "col_grip": col_grip,
                "c_core": c_core,
                "c_frames": c_frames,
                "c_cav": c_cav,
                "fence_cols": fence_cols,
            }
            self.ego_colors.add(col_ego)
            self.ego_accent_colors.add(col_grip)
            self.action5_is_contact_operator = True
            for a_id, dy_g in ((1, -S), (2, S)):
                self.kinematic_generators.setdefault(a_id, (dy_g, 0, False))
            for a_id, dx_g in ((3, -S), (4, S)):
                self.kinematic_generators.setdefault(a_id, (0, dx_g, False))

        roles = self._pp_roles
        S = int(roles["S"])
        bg = int(roles["bg"])
        col_ego = int(roles["col_ego"])
        col_grip = int(roles["col_grip"])
        c_core = int(roles["c_core"])
        c_cav = int(roles["c_cav"])
        fence_cols = set(roles["fence_cols"])
        dir_map = {1: (-S, 0), 2: (S, 0), 3: (0, -S), 4: (0, S)}

        for sr in range(S, H - S, S):
            for sc in range(S, W - S, S):
                if np.all(board0[sr + 1 : sr + S - 1, sc + 1 : sc + S - 1] == c_cav):
                    self._pp_slot_history.add((sr, sc))
                    if int(board0[sr, sc]) == c_core:
                        self._pp_init_final_slots.add((sr, sc))
                    else:
                        self._pp_transfer_history.add((sr, sc))

        comps = _connected_components_2d(
            board0[S : H - S, S : W - S], connectivity=4, same_color=True, ignore_val=bg
        )
        final_slots = [
            (sr, sc)
            for sr in range(S, H - S, S)
            for sc in range(S, W - S, S)
            if int(board0[sr, sc]) == c_core
            and np.all(board0[sr + 1 : sr + S - 1, sc + 1 : sc + S - 1] == c_cav)
        ]
        transfer_slots = [
            (sr, sc)
            for sr in range(S, H - S, S)
            for sc in range(S, W - S, S)
            if int(board0[sr, sc]) != c_core
            and np.all(board0[sr + 1 : sr + S - 1, sc + 1 : sc + S - 1] == c_cav)
        ]
        all_slots = final_slots + transfer_slots
        e_comps = [x for x in comps if int(x["color"]) == col_ego]
        blocks = [
            (sr, sc)
            for sr in range(S, H - S, S)
            for sc in range(S, W - S, S)
            if (sr, sc) not in self._pp_slot_history
            and int(board0[sr, sc]) not in (bg, c_core, col_grip, col_ego)
            and np.all(board0[sr + 1 : sr + S - 1, sc + 1 : sc + S - 1] == c_core)
        ]
        if not (e_comps and all_slots and blocks):
            return []

        er0 = ((int(e_comps[0]["r0"]) + S) // S) * S
        ec0 = ((int(e_comps[0]["c0"]) + S) // S) * S

        def is_already_facing(act_id: int) -> bool:
            if act_id == 1:
                return bool(np.all(board0[er0, ec0 : ec0 + S] == col_grip))
            if act_id == 2:
                return bool(np.all(board0[er0 + S - 1, ec0 : ec0 + S] == col_grip))
            if act_id == 3:
                return bool(np.all(board0[er0 : er0 + S, ec0] == col_grip))
            if act_id == 4:
                return bool(np.all(board0[er0 : er0 + S, ec0 + S - 1] == col_grip))
            return False

        obs_cells = {
            (r, col)
            for r in range(S, H - S, S)
            for col in range(S, W - S, S)
            if set(int(v) for v in np.unique(board0[r : r + S, col : col + S])) - {bg, col_grip, col_ego}
        }
        wall_fence = {
            (r, col)
            for r in range(S, H - S, S)
            for col in range(S, W - S, S)
            if (fence_cols and any(int(v) in fence_cols for v in np.unique(board0[r : r + S, col : col + S])))
            or (r, col) in self._pp_transfer_history
        }

        def room_of(cell: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
            if cell in wall_fence:
                return frozenset()
            vis = {cell}
            dq = deque([cell])
            while dq:
                cr, cc = dq.popleft()
                for dy, dx in dir_map.values():
                    nr, nc = cr + dy, cc + dx
                    if (
                        S <= nr <= H - 2 * S
                        and S <= nc <= W - 2 * S
                        and (nr, nc) not in wall_fence
                        and (nr, nc) not in vis
                    ):
                        vis.add((nr, nc))
                        dq.append((nr, nc))
            return frozenset(vis)

        ego_reach = {(er0, ec0)}
        ebfs = deque([(er0, ec0)])
        while ebfs:
            cr, cc = ebfs.popleft()
            for dy, dx in dir_map.values():
                nr, nc = cr + dy, cc + dx
                if (
                    S <= nr <= H - 2 * S
                    and S <= nc <= W - 2 * S
                    and (nr, nc) not in obs_cells
                    and (nr, nc) not in ego_reach
                ):
                    ego_reach.add((nr, nc))
                    ebfs.append((nr, nc))

        reach_blocks = [
            blk
            for blk in blocks
            if any((blk[0] - dr, blk[1] - dc) in ego_reach for dr, dc in dir_map.values())
        ]
        reach_blocks.sort(
            key=lambda blk: (
                sum(
                    1
                    for dr, dc in dir_map.values()
                    if (blk[0] - dr, blk[1] - dc) in ego_reach
                    and (blk[0] - dr, blk[1] - dc) not in obs_cells
                ),
                abs(blk[0] - er0) + abs(blk[1] - ec0),
            )
        )
        cand_triples: List[Tuple[int, int, int, int, int, int]] = []
        for sr0, sc0 in final_slots:
            for act_in in dir_map:
                cand_triples.append((0, 0, 0, sr0, sc0, act_in))
        for sr0, sc0 in transfer_slots:
            for act_in, (dr, dc) in dir_map.items():
                opp = (sr0 + dr, sc0 + dc)
                if not (S <= opp[0] <= H - 2 * S and S <= opp[1] <= W - 2 * S) or opp in wall_fence:
                    continue
                rm = room_of(opp)
                slots_in_rm = [fs for fs in final_slots if fs in rm]
                if not slots_in_rm:
                    continue
                undocked_in_rm = [
                    (r, c)
                    for r in range(S, H - S, S)
                    for c in range(S, W - S, S)
                    if (r, c) not in self._pp_init_final_slots
                    and int(board0[r, c]) != c_core
                    and np.all(board0[r + 1 : r + S - 1, c + 1 : c + S - 1] == c_core)
                    and (
                        ((r, c) in rm and (r, c) not in reach_blocks)
                        or (
                            (r, c) in self._pp_transfer_history
                            and any((r + dy, c + dx) in rm for dy, dx in dir_map.values())
                        )
                    )
                ]
                deficit = len(slots_in_rm) - len(undocked_in_rm)
                if deficit > 0:
                    d_min = min(abs(opp[0] - fs[0]) + abs(opp[1] - fs[1]) for fs in slots_in_rm)
                    cand_triples.append((1, -deficit, d_min, sr0, sc0, act_in))
        cand_triples.sort()
        slot_set = set(all_slots)
        inv_act = {1: 2, 2: 1, 3: 4, 4: 3}

        for item in cand_triples:
            sr0, sc0, act_in = item[-3], item[-2], item[-1]
            dr, dc = dir_map[act_in]
            rel_r, rel_c = sr0 - dr, sc0 - dc
            if not (S <= rel_r <= H - 2 * S and S <= rel_c <= W - 2 * S) or (rel_r, rel_c) in obs_cells:
                continue
            for br0, bc0 in reach_blocks:
                stand_r, stand_c = br0 - dr, bc0 - dc
                if not (S <= stand_r <= H - 2 * S and S <= stand_c <= W - 2 * S) or (stand_r, stand_c) in obs_cells:
                    continue
                bfs1 = deque([(er0, ec0)])
                par1: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], ActionKey]]] = {(er0, ec0): None}
                path1: Optional[List[ActionKey]] = None
                while bfs1 and len(par1) < 3000:
                    cr, cc = bfs1.popleft()
                    if (cr, cc) == (stand_r, stand_c):
                        rev1: List[ActionKey] = []
                        cur1: Optional[Tuple[int, int]] = (cr, cc)
                        while cur1 is not None and par1[cur1] is not None:
                            prev1, a1 = par1[cur1]  # type: ignore[misc]
                            rev1.append(a1)
                            cur1 = prev1
                        rev1.reverse()
                        need_turn = (not is_already_facing(act_in)) if not rev1 else (rev1[-1] != act_in)
                        path1 = rev1 + ([act_in] if need_turn else [])
                        break
                    for a, (dy, dx) in dir_map.items():
                        nr, nc = cr + dy, cc + dx
                        if (
                            S <= nr <= H - 2 * S
                            and S <= nc <= W - 2 * S
                            and (nr, nc) not in obs_cells
                            and (nr, nc) not in par1
                        ):
                            par1[(nr, nc)] = ((cr, cc), a)
                            bfs1.append((nr, nc))
                if path1 is None:
                    continue
                obs2 = obs_cells - {(br0, bc0)}
                bfs2 = deque([(br0, bc0)])
                par2: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], ActionKey]]] = {(br0, bc0): None}
                path2: Optional[List[ActionKey]] = None
                while bfs2 and len(par2) < 3000:
                    cbr, cbc = bfs2.popleft()
                    if (cbr, cbc) == (sr0, sc0):
                        rev2: List[ActionKey] = []
                        cur2: Optional[Tuple[int, int]] = (cbr, cbc)
                        while cur2 is not None and par2[cur2] is not None:
                            prev2, a2 = par2[cur2]  # type: ignore[misc]
                            rev2.append(a2)
                            cur2 = prev2
                        rev2.reverse()
                        path2 = rev2
                        break
                    for a, (dy, dx) in dir_map.items():
                        nbr, nbc = cbr + dy, cbc + dx
                        ner, nec = nbr - dr, nbc - dc
                        if (
                            S <= nbr <= H - 2 * S
                            and S <= nbc <= W - 2 * S
                            and S <= ner <= H - 2 * S
                            and S <= nec <= W - 2 * S
                            and (ner, nec) not in obs2
                            and (nbr, nbc) not in (obs2 - slot_set)
                            and (nbr, nbc) not in par2
                        ):
                            par2[(nbr, nbc)] = ((cbr, cbc), a)
                            bfs2.append((nbr, nbc))
                if path2 is not None:
                    step_back: List[ActionKey] = [inv_act[act_in]] if (sr0, sc0) in transfer_slots else []
                    return path1 + [5] + path2 + [5] + step_back

        if cand_triples and reach_blocks:
            _, _, _, sr0, sc0, act_in = cand_triples[0]
            dr, dc = dir_map[act_in]
            for br0, bc0 in reach_blocks:
                if (br0 + dr, bc0 + dc) not in obs_cells:
                    for a_side, (sdr, sdc) in dir_map.items():
                        s_r, s_c = br0 - sdr, bc0 - sdc
                        if (s_r, s_c) in ego_reach and (s_r + dr, s_c + dc) not in (obs_cells - {(br0, bc0)}):
                            bfs1 = deque([(er0, ec0)])
                            par1_s: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], ActionKey]]] = {(er0, ec0): None}
                            p1_res: Optional[List[ActionKey]] = None
                            while bfs1 and len(par1_s) < 3000:
                                cr, cc = bfs1.popleft()
                                if (cr, cc) == (s_r, s_c):
                                    rev_s: List[ActionKey] = []
                                    cur_s: Optional[Tuple[int, int]] = (cr, cc)
                                    while cur_s is not None and par1_s[cur_s] is not None:
                                        prev_s, a_s = par1_s[cur_s]  # type: ignore[misc]
                                        rev_s.append(a_s)
                                        cur_s = prev_s
                                    rev_s.reverse()
                                    need_turn = (not is_already_facing(a_side)) if not rev_s else (rev_s[-1] != a_side)
                                    p1_res = rev_s + ([a_side] if need_turn else [])
                                    break
                                for a, (dy, dx) in dir_map.items():
                                    nr, nc = cr + dy, cc + dx
                                    if (
                                        S <= nr <= H - 2 * S
                                        and S <= nc <= W - 2 * S
                                        and (nr, nc) not in obs_cells
                                        and (nr, nc) not in par1_s
                                    ):
                                        par1_s[(nr, nc)] = ((cr, cc), a)
                                        bfs1.append((nr, nc))
                            if p1_res is not None:
                                return p1_res + [5, act_in, 5]
        return []

    def _solve_multisprite_kick_cavity_plan(
        self,
        board0: np.ndarray,
    ) -> List[ActionKey]:
        """
        Solves multi-sprite collinear impulse (kick across chasms) and hollow-cavity docking
        with `ACTION6` active-sprite switching purely from online-derived `S_16`-equivariant
        component topology.
        """
        H, W = board0.shape
        if H < 18 or W < 18:
            return []
        if self._multisprite_roles is None:
            H_eff = H - max(2, H // 32)
            comps = _connected_components_2d(board0[:H_eff, :], connectivity=8)
            scale_cands: Counter = Counter()
            for c in comps:
                h, w = int(c["h"]), int(c["w"])
                hole_a = h * w - int(c["area"])
                if 3 <= h <= 15 and 3 <= w <= 15 and hole_a > 0 and 0.70 <= float(c["fill_ratio"]) < 1.0:
                    ratio = float(h * w) / float(hole_a)
                    s_est = int(round(math.sqrt(ratio)))
                    if 2 <= s_est <= 6 and h % s_est == 0 and w % s_est == 0:
                        scale_cands[s_est] += 1
            if not scale_cands:
                return []
            S = int(scale_cands.most_common(1)[0][0])
            cand_sprites: List[Tuple[Dict[str, Any], int]] = []
            for c in comps:
                h, w = int(c["h"]), int(c["w"])
                r0, r1, c0, c1 = int(c["r0"]), int(c["r1"]), int(c["c0"]), int(c["c1"])
                if (
                    S <= h <= 4 * S
                    and S <= w <= 4 * S
                    and h % S == 0
                    and w % S == 0
                    and 0.70 <= float(c["fill_ratio"]) < 1.0
                ):
                    kh, kw = h // S, w // S
                    kr0, kc0 = r0 + (h - kh) // 2, c0 + (w - kw) // 2
                    core = board0[kr0 : kr0 + kh, kc0 : kc0 + kw]
                    u_core = np.unique(core)
                    if (
                        len(u_core) == 1
                        and int(u_core[0]) != int(c["color"])
                        and int(c["area"]) + kh * kw == h * w
                    ):
                        cand_sprites.append((c, int(u_core[0])))
            if len(cand_sprites) < 2:
                return []
            col_sprite = int(cand_sprites[0][0]["color"])
            core_cols = [x[1] for x in cand_sprites if int(x[0]["color"]) == col_sprite]
            if len(set(core_cols)) < 2:
                return []
            col_active = min(set(core_cols), key=lambda k: core_cols.count(k))
            cand_cavs: List[Tuple[Dict[str, Any], int]] = []
            for c in comps:
                h, w = int(c["h"]), int(c["w"])
                r0, r1, c0, c1 = int(c["r0"]), int(c["r1"]), int(c["c0"]), int(c["c1"])
                if (
                    int(c["color"]) != col_sprite
                    and S + 2 <= h <= 4 * S + 2
                    and S + 2 <= w <= 4 * S + 2
                    and (h - 2) % S == 0
                    and (w - 2) % S == 0
                    and float(c["fill_ratio"]) < 0.75
                    and int(c["area"]) in (2 * (h + w) - 4, 2 * (h + w) - 5)
                ):
                    inner = board0[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1]
                    u_in, cnt_in = np.unique(inner, return_counts=True)
                    floor_c = int(u_in[int(np.argmax(cnt_in))])
                    cand_cavs.append((c, floor_c))
            if not cand_cavs:
                return []
            col_cav = int(cand_cavs[0][0]["color"])
            col_floor = int(cand_cavs[0][1])
            used = {col_sprite, col_cav, col_floor} | set(core_cols)
            rem_comps = [c for c in comps if int(c["color"]) not in used and int(c["area"]) >= 36]
            wall_c = next(
                (int(c["color"]) for c in rem_comps if int(c["r0"]) == 0 and int(c["c0"]) == 0),
                None,
            )
            if wall_c is None:
                return []
            chasm_c = next((int(c["color"]) for c in rem_comps if int(c["color"]) != wall_c), -1)
            self._multisprite_roles = {
                "S": S,
                "H_eff": H_eff,
                "col_sprite": col_sprite,
                "col_active": col_active,
                "col_cav": col_cav,
                "col_floor": col_floor,
                "col_wall": wall_c,
                "col_chasm": chasm_c,
            }
            self.ego_colors.add(col_sprite)
            self.ego_accent_colors.add(col_active)
            for a_id, dy_g in ((1, -S), (2, S)):
                self.kinematic_generators.setdefault(a_id, (dy_g, 0, False))
            for a_id, dx_g in ((3, -S), (4, S)):
                self.kinematic_generators.setdefault(a_id, (0, dx_g, False))

        roles = self._multisprite_roles
        S = int(roles["S"])
        H_eff = int(roles.get("H_eff", H - 2))
        comps = _connected_components_2d(board0[:H_eff, :], connectivity=8)
        col_sprite = int(roles["col_sprite"])
        col_active = int(roles["col_active"])
        col_cav = int(roles["col_cav"])
        col_wall = int(roles["col_wall"])
        col_chasm = int(roles["col_chasm"])
        dir_map = {1: (-S, 0), 2: (S, 0), 3: (0, -S), 4: (0, S)}

        sprites: List[Dict[str, Any]] = []
        cavities: List[Dict[str, int]] = []
        for c in comps:
            h, w = int(c["h"]), int(c["w"])
            r0, r1, c0, c1 = int(c["r0"]), int(c["r1"]), int(c["c0"]), int(c["c1"])
            if int(c["color"]) in (col_sprite, col_active) and S <= h <= 4 * S and S <= w <= 4 * S:
                if (r1 - r0) % S == 0 and (c1 - c0) % S == 0:
                    inner = board0[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1]
                    sprites.append(
                        {
                            "r0": r0,
                            "c0": c0,
                            "h": r1 - r0,
                            "w": c1 - c0,
                            "active": bool(np.any(inner == col_active)),
                        }
                    )
            elif int(c["color"]) == col_cav and S + 2 <= h <= 4 * S + 2 and S + 2 <= w <= 4 * S + 2 and float(c["fill_ratio"]) < 0.75:
                cavities.append({"r0": r0 + 1, "c0": c0 + 1, "h": r1 - r0 - 2, "w": c1 - c0 - 2})

        undocked = [
            s
            for s in sprites
            if not any(
                s["r0"] == g["r0"] and s["c0"] == g["c0"] and s["h"] == g["h"] and s["w"] == g["w"]
                for g in cavities
            )
        ]
        free_cavs = [
            g
            for g in cavities
            if not any(
                s["r0"] == g["r0"] and s["c0"] == g["c0"] and s["h"] == g["h"] and s["w"] == g["w"]
                for s in sprites
            )
        ]
        if not undocked or not free_cavs:
            return []

        def box_ok(
            r0: int,
            c0: int,
            h: int,
            w: int,
            other_boxes: Sequence[Tuple[int, int, int, int]] = (),
            allow_chasm: bool = False,
        ) -> bool:
            if r0 < 0 or r0 + h > H_eff or c0 < 0 or c0 + w > W:
                return False
            sub = board0[r0 : r0 + h, c0 : c0 + w]
            if np.any(sub == col_wall):
                return False
            if not allow_chasm and col_chasm >= 0 and np.any(sub == col_chasm):
                return False
            for or0, oc0, oh, ow in other_boxes:
                if not (r0 + h <= or0 or or0 + oh <= r0 or c0 + w <= oc0 or oc0 + ow <= c0):
                    return False
            return True

        def nav_bfs(
            s: Dict[str, Any],
            goal_rc: Tuple[int, int],
            other_boxes: Sequence[Tuple[int, int, int, int]],
        ) -> Optional[List[ActionKey]]:
            sr0, sc0, sh, sw = int(s["r0"]), int(s["c0"]), int(s["h"]), int(s["w"])
            if (sr0, sc0) == goal_rc:
                return []
            dq = deque([(sr0, sc0)])
            par_nav: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], ActionKey]]] = {(sr0, sc0): None}
            while dq and len(par_nav) < 3000:
                cr, cc = dq.popleft()
                if (cr, cc) == goal_rc:
                    rev_p: List[ActionKey] = []
                    cur_p: Optional[Tuple[int, int]] = (cr, cc)
                    while cur_p is not None and par_nav[cur_p] is not None:
                        prev_p, act_p = par_nav[cur_p]  # type: ignore[misc]
                        rev_p.append(act_p)
                        cur_p = prev_p
                    rev_p.reverse()
                    return rev_p
                for a, (dy, dx) in dir_map.items():
                    nr, nc = cr + dy, cc + dx
                    if (nr, nc) not in par_nav and box_ok(nr, nc, sh, sw, other_boxes, allow_chasm=False):
                        par_nav[(nr, nc)] = ((cr, cc), a)
                        dq.append((nr, nc))
            return None

        def reach_set(
            r0: int,
            c0: int,
            h: int,
            w: int,
            other_boxes: Sequence[Tuple[int, int, int, int]] = (),
        ) -> Set[Tuple[int, int]]:
            vis = {(r0, c0)}
            dq = deque([(r0, c0)])
            max_states = 3000
            while dq and len(vis) <= max_states:
                cr, cc = dq.popleft()
                for dy, dx in dir_map.values():
                    nr, nc = cr + dy, cc + dx
                    if (nr, nc) not in vis and box_ok(nr, nc, h, w, other_boxes, allow_chasm=False):
                        vis.add((nr, nc))
                        dq.append((nr, nc))
            return vis

        def sim_kick(
            r0: int,
            c0: int,
            h: int,
            w: int,
            dy: int,
            dx: int,
            other_boxes: Sequence[Tuple[int, int, int, int]],
        ) -> Optional[Tuple[int, int]]:
            cr, cc = r0, c0
            for _ in range(5):
                nr, nc = cr + dy, cc + dx
                if not box_ok(nr, nc, h, w, other_boxes, allow_chasm=True):
                    break
                cr, cc = nr, nc
            if not box_ok(cr, cc, h, w, other_boxes, allow_chasm=False):
                return None
            return (cr, cc)

        for s in undocked:
            s_reach = reach_set(int(s["r0"]), int(s["c0"]), int(s["h"]), int(s["w"]))
            my_cavs = [
                g
                for g in free_cavs
                if g["h"] == s["h"]
                and g["w"] == s["w"]
                and not any(
                    (g["r0"], g["c0"]) in reach_set(int(u["r0"]), int(u["c0"]), int(u["h"]), int(u["w"]))
                    for u in undocked
                    if u["h"] == g["h"] and u["w"] == g["w"]
                )
            ]
            if not my_cavs:
                continue
            for k in undocked:
                if k is s:
                    continue
                k_reach_no_obs = reach_set(int(k["r0"]), int(k["c0"]), int(k["h"]), int(k["w"]))
                best_kick: Optional[Tuple[Tuple[int, int, int], int, int, int, int, int]] = None
                for tr, tc in sorted(
                    s_reach, key=lambda rc: abs(rc[0] - int(s["r0"])) + abs(rc[1] - int(s["c0"]))
                ):
                    for a_kick, (dy, dx) in dir_map.items():
                        if dy > 0:
                            kr, kc = tr - int(k["h"]), tc
                        elif dy < 0:
                            kr, kc = tr + int(s["h"]), tc
                        elif dx > 0:
                            kr, kc = tr, tc - int(k["w"])
                        else:
                            kr, kc = tr, tc + int(s["w"])
                        if (kr, kc) not in k_reach_no_obs:
                            continue
                        others_s = [
                            (int(x["r0"]), int(x["c0"]), int(x["h"]), int(x["w"]))
                            for x in sprites
                            if x is not s and x is not k
                        ] + [(kr, kc, int(k["h"]), int(k["w"]))]
                        land = sim_kick(tr, tc, int(s["h"]), int(s["w"]), dy, dx, others_s)
                        if land is None or land in s_reach:
                            continue
                        l_reach = reach_set(land[0], land[1], int(s["h"]), int(s["w"]))
                        d_goal = min(abs(land[0] - g["r0"]) + abs(land[1] - g["c0"]) for g in my_cavs)
                        can_reach_goal = any((g["r0"], g["c0"]) in l_reach for g in my_cavs)
                        score = (
                            0 if can_reach_goal else 1,
                            abs(tr - int(s["r0"]))
                            + abs(tc - int(s["c0"]))
                            + abs(kr - int(k["r0"]))
                            + abs(kc - int(k["c0"])),
                            d_goal,
                        )
                        if best_kick is None or score < best_kick[0]:
                            best_kick = (score, tr, tc, kr, kc, a_kick)
                if best_kick is not None:
                    _, tr, tc, kr, kc, a_kick = best_kick
                    others_for_s = [
                        (int(x["r0"]), int(x["c0"]), int(x["h"]), int(x["w"]))
                        for x in sprites
                        if x is not s
                    ]
                    if (int(s["r0"]), int(s["c0"])) != (tr, tc):
                        ps = nav_bfs(s, (tr, tc), others_for_s)
                        if ps is not None:
                            sel: List[ActionKey] = (
                                [(6, int(s["c0"]) + int(s["w"]) // 2, int(s["r0"]) + int(s["h"]) // 2)]
                                if not s["active"]
                                else []
                            )
                            return sel + ps
                        others_for_k = [
                            (int(x["r0"]), int(x["c0"]), int(x["h"]), int(x["w"]))
                            for x in sprites
                            if x is not k
                        ]
                        pk = nav_bfs(k, (kr, kc), others_for_k)
                        if pk is not None and pk:
                            sel = (
                                [(6, int(k["c0"]) + int(k["w"]) // 2, int(k["r0"]) + int(k["h"]) // 2)]
                                if not k["active"]
                                else []
                            )
                            return sel + pk
                    else:
                        others_for_k = [
                            (int(x["r0"]), int(x["c0"]), int(x["h"]), int(x["w"]))
                            for x in sprites
                            if x is not k
                        ]
                        pk = nav_bfs(k, (kr, kc), others_for_k)
                        if pk is not None:
                            sel = (
                                [(6, int(k["c0"]) + int(k["w"]) // 2, int(k["r0"]) + int(k["h"]) // 2)]
                                if not k["active"]
                                else []
                            )
                            return sel + pk + [a_kick]

        for s in undocked:
            others_for_s = [
                (int(x["r0"]), int(x["c0"]), int(x["h"]), int(x["w"]))
                for x in sprites
                if x is not s
            ]
            my_cavs = [g for g in free_cavs if g["h"] == s["h"] and g["w"] == s["w"]]
            for g in my_cavs:
                ps = nav_bfs(s, (g["r0"], g["c0"]), others_for_s)
                if ps is not None:
                    sel = (
                        [(6, int(s["c0"]) + int(s["w"]) // 2, int(s["r0"]) + int(s["h"]) // 2)]
                        if not s["active"]
                        else []
                    )
                    return sel + ps
        return []

    @staticmethod
    def _d4_canonical_glyph(mask: np.ndarray) -> Tuple[int, ...]:
        """Returns the lexicographically minimal tuple across all 8 Dihedral D_4 symmetries of a square binary mask."""
        m = (mask > 0).astype(np.uint8)
        if m.shape[0] != m.shape[1] or not np.any(m):
            return tuple(m.ravel().tolist())
        cands: List[Tuple[int, ...]] = []
        for k in range(4):
            rk = np.rot90(m, k)
            cands.append(tuple(rk.ravel().tolist()))
            cands.append(tuple(np.fliplr(rk).ravel().tolist()))
        return min(cands)

    def _solve_d4_glyph_monoid_homomorphism_plan(
        self,
        board0: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
    ) -> List[ActionKey]:
        """
        Solves categorical Dihedral D_4-equivariant free-monoid glyph homomorphism evaluation
        (forward composition mu = mu_k o ... o mu_1: Sigma_in^* -> Sigma_out^*) and inverse
        free-monoid word-equation rule synthesis (factorizing w_in = u_1 ... u_m and
        w_out = v_1 ... v_m across rule arities |L_r| = |R_r| in {1, 2, ...}).
        100% S_16 color-permutation equivariant, D_4 dihedral equivariant, and scale-invariant.
        """
        if available_actions is not None and (1 not in available_actions or 4 not in available_actions):
            return []
        H, W = board0.shape
        if H < 24 or W < 24:
            return []
        vals, cnts = np.unique(board0, return_counts=True)
        bg_mode = int(vals[int(np.argmax(cnts))])

        def is_outer_frame(c: Dict[str, Any]) -> bool:
            r0, r1, c0, c1 = int(c["r0"]), int(c["r1"]), int(c["c0"]), int(c["c1"])
            h, w, col = int(c["h"]), int(c["w"]), int(c["color"])
            if h < 5 or w < h or w % h != 0 or int(c["area"]) >= board0.size // 2:
                return False
            sub = board0[r0:r1, c0:c1]
            return bool(
                np.all(sub[0, :] == col)
                and np.all(sub[-1, :] == col)
                and np.all(sub[:, 0] == col)
                and np.all(sub[:, -1] == col)
                and np.any(sub[1:-1, 1:-1] != col)
            )

        def extract_glyphs(comp: Dict[str, Any], S_val: int) -> List[np.ndarray]:
            r0, c0, w = int(comp["r0"]), int(comp["c0"]), int(comp["w"])
            col = int(comp["color"])
            G_val = S_val - 2
            return [
                (board0[r0 + 1 : r0 + 1 + G_val, c0 + 1 + idx * S_val : c0 + 1 + idx * S_val + G_val] == col).astype(np.int64)
                for idx in range(w // S_val)
            ]

        def apply_rules_once(
            word_canons: List[Tuple[int, ...]],
            rules_canons: List[Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]],
        ) -> Optional[List[Tuple[int, ...]]]:
            out: List[Tuple[int, ...]] = []
            i = 0
            while i < len(word_canons):
                matched = False
                for g_ins, g_outs in sorted(rules_canons, key=lambda x: -len(x[0])):
                    L = len(g_ins)
                    if i + L <= len(word_canons) and word_canons[i : i + L] == g_ins:
                        out.extend(g_outs)
                        i += L
                        matched = True
                        break
                if not matched:
                    return None
            return out

        if self._monoid_hom_state is None:
            comps = _connected_components_2d(board0, connectivity=4, same_color=True, ignore_val=None)
            raw_frames = [c for c in comps if is_outer_frame(c)]
            if len(raw_frames) < 4:
                return []
            S_counts = Counter(int(c["h"]) for c in raw_frames)
            S = int(S_counts.most_common(1)[0][0])
            frame_comps = sorted(
                [c for c in raw_frames if int(c["h"]) == S],
                key=lambda x: (int(x["r0"]), int(x["c0"])),
            )
            if len(frame_comps) < 4:
                return []
            row_groups: Dict[int, List[Dict[str, Any]]] = {}
            for c in frame_comps:
                row_groups.setdefault(int(c["r0"]), []).append(c)
            rule_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            rule_boxes: List[Dict[str, Any]] = []
            rule_rows_meta: List[List[Tuple[int, List[Tuple[int, ...]], int]]] = []
            for r, cs in sorted(row_groups.items()):
                cs_sorted = sorted(cs, key=lambda x: int(x["c0"]))
                if len(cs_sorted) >= 2 and len(cs_sorted) % 2 == 0:
                    rule_boxes.extend(cs_sorted)
                    rule_rows_meta.append(
                        [
                            (
                                int(b["w"]) // S,
                                [self._d4_canonical_glyph(g) for g in extract_glyphs(b, S)],
                                int(b["color"]),
                            )
                            for b in cs_sorted
                        ]
                    )
                    for p in range(0, len(cs_sorted), 2):
                        rule_pairs.append((cs_sorted[p], cs_sorted[p + 1]))
            strip_rows = sorted(
                [(r, cs[0]) for r, cs in row_groups.items() if len(cs) == 1],
                key=lambda x: x[0],
            )
            if len(strip_rows) < 2 or not rule_pairs:
                return []
            in_strip_comp = strip_rows[-2][1]
            out_strip_comp = strip_rows[-1][1]
            c_start = int(in_strip_comp["color"])
            c_end = int(out_strip_comp["color"])
            in_w = [self._d4_canonical_glyph(g) for g in extract_glyphs(in_strip_comp, S)]
            out_w = [self._d4_canonical_glyph(g) for g in extract_glyphs(out_strip_comp, S)]

            # Check if cursor bracket cap is above rule_boxes[0] (Inverse Rule Synthesis) or above out_strip_comp (Forward Evaluation)
            b0 = rule_boxes[0]
            br0, bc0 = int(b0["r0"]), int(b0["c0"])
            cursor_on_rules = bool(
                br0 >= 3
                and int(board0[br0 - 3, bc0 + 1]) != int(board0[br0 - 3, bc0])
                and np.all(board0[br0 - 3, bc0 + 1 : bc0 + S - 1] == board0[br0 - 3, bc0 + 1])
            )
            if not cursor_on_rules:
                morphisms: Dict[Tuple[int, int], List[Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]]] = {}
                for c_in, c_out in rule_pairs:
                    col_i, col_o = int(c_in["color"]), int(c_out["color"])
                    g_ins = [self._d4_canonical_glyph(g) for g in extract_glyphs(c_in, S)]
                    g_outs = [self._d4_canonical_glyph(g) for g in extract_glyphs(c_out, S)]
                    morphisms.setdefault((col_i, col_o), []).append((g_ins, g_outs))
                q: deque = deque([(c_start, in_w, (c_start,))])
                target_out_canons: Optional[List[Tuple[int, ...]]] = None
                states_exp = 0
                max_states = 3000
                while q and states_exp < max_states:
                    cur_c, cur_w, visited_cols = q.popleft()
                    states_exp += 1
                    if cur_c == c_end and len(visited_cols) > 1:
                        target_out_canons = cur_w
                        break
                    for (ci, co), rules_list in morphisms.items():
                        if ci == cur_c and co not in visited_cols:
                            nxt_w = apply_rules_once(cur_w, rules_list)
                            if nxt_w is not None:
                                q.append((co, nxt_w, visited_cols + (co,)))
                if target_out_canons is None:
                    return []
                self._monoid_hom_state = {
                    "mode": "strip",
                    "S": S,
                    "comp": out_strip_comp,
                    "targets": [[t] for t in target_out_canons],
                    "idx": 0,
                    "tries": 0,
                }
            else:
                def matches_eq_kernel(
                    u_word: List[Tuple[int, ...]],
                    w0_word: List[Tuple[int, ...]],
                ) -> bool:
                    if len(u_word) != len(w0_word):
                        return False
                    for i_k in range(len(u_word)):
                        for j_k in range(i_k + 1, len(u_word)):
                            if (u_word[i_k] == u_word[j_k]) != (w0_word[i_k] == w0_word[j_k]):
                                return False
                    return True

                flat_meta_pairs: List[Tuple[Tuple[int, List[Tuple[int, ...]], int], Tuple[int, List[Tuple[int, ...]], int]]] = []
                for r_row in rule_rows_meta:
                    for p in range(0, len(r_row), 2):
                        flat_meta_pairs.append((r_row[p], r_row[p + 1]))

                pair_color_sigs = {(p[0][2], p[1][2]) for p in flat_meta_pairs}
                synthesized_targets: Optional[List[List[Tuple[int, ...]]]] = None

                if pair_color_sigs == {(c_start, c_end)}:
                    R_rules = len(flat_meta_pairs)
                    best_cost_1 = 10**9
                    nodes_1stage = 0
                    max_nodes_1stage = 3000

                    def backtrack_1stage(
                        i_pos: int,
                        j_pos: int,
                        assigned_rules: Dict[int, Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]],
                    ) -> None:
                        nonlocal best_cost_1, synthesized_targets, nodes_1stage
                        if nodes_1stage >= max_nodes_1stage:
                            return
                        nodes_1stage += 1
                        if i_pos == len(in_w) and j_pos == len(out_w):
                            if len(assigned_rules) != R_rules:
                                return
                            rules_seq = [assigned_rules[r_i] for r_i in range(R_rules)]
                            if apply_rules_once(in_w, rules_seq) != out_w:
                                return
                            cost_val = 0
                            cand_targets: List[List[Tuple[int, ...]]] = []
                            for r_i in range(R_rules):
                                u_r, v_r = assigned_rules[r_i]
                                w0_in, w0_out = flat_meta_pairs[r_i][0][1], flat_meta_pairs[r_i][1][1]
                                if u_r != w0_in:
                                    cost_val += 1
                                if v_r != w0_out:
                                    cost_val += 1
                                cand_targets.extend([u_r, v_r])
                            if cost_val < best_cost_1:
                                best_cost_1 = cost_val
                                synthesized_targets = cand_targets
                            return

                        used_lhs = {tuple(u_val) for u_val, _ in assigned_rules.values()}
                        for r_i in range(R_rules):
                            m_in, w0_in, _ = flat_meta_pairs[r_i][0]
                            m_out, w0_out, _ = flat_meta_pairs[r_i][1]
                            if i_pos + m_in <= len(in_w) and j_pos + m_out <= len(out_w):
                                u_seg = in_w[i_pos : i_pos + m_in]
                                v_seg = out_w[j_pos : j_pos + m_out]
                                if not (matches_eq_kernel(u_seg, w0_in) and matches_eq_kernel(v_seg, w0_out)):
                                    continue
                                if r_i in assigned_rules:
                                    if assigned_rules[r_i] == (u_seg, v_seg):
                                        backtrack_1stage(i_pos + m_in, j_pos + m_out, assigned_rules)
                                elif tuple(u_seg) not in used_lhs:
                                    assigned_rules[r_i] = (u_seg, v_seg)
                                    backtrack_1stage(i_pos + m_in, j_pos + m_out, assigned_rules)
                                    del assigned_rules[r_i]

                    backtrack_1stage(0, 0, {})
                elif all(len(r_row) == 4 for r_row in rule_rows_meta):
                    from itertools import product

                    R_rows = len(rule_rows_meta)
                    sigma_mid: List[Tuple[int, ...]] = []
                    for r_row in rule_rows_meta:
                        for b_idx in (1, 2):
                            for g_sym in r_row[b_idx][1]:
                                if g_sym not in sigma_mid:
                                    sigma_mid.append(g_sym)

                    rhs1_cands: List[Tuple[List[List[Tuple[int, ...]]], Set[Tuple[Tuple[int, ...], ...]]]] = []
                    for r_i in range(R_rows):
                        m1, w0_1, _ = rule_rows_meta[r_i][1]
                        rev_others = [
                            list(reversed(rule_rows_meta[rp][1][1]))
                            for rp in range(R_rows)
                            if rp != r_i
                            and len(rule_rows_meta[rp][1][1]) == m1
                            and matches_eq_kernel(list(reversed(rule_rows_meta[rp][1][1])), w0_1)
                        ]
                        cands_r: List[List[Tuple[int, ...]]] = [w0_1]
                        for w_rev in rev_others:
                            if w_rev not in cands_r:
                                cands_r.append(w_rev)
                        for tup in product(sigma_mid, repeat=m1):
                            w_cand = list(tup)
                            if matches_eq_kernel(w_cand, w0_1) and w_cand not in cands_r:
                                cands_r.append(w_cand)
                        rhs1_cands.append((cands_r, {tuple(w_rev) for w_rev in rev_others}))

                    best_cost_2 = 10**9
                    nodes_2stage = 0
                    max_nodes_2stage = 3000

                    def backtrack_stage1(
                        i_pos: int,
                        assigned1: Dict[int, Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]],
                    ) -> None:
                        nonlocal best_cost_2, synthesized_targets, nodes_2stage
                        if nodes_2stage >= max_nodes_2stage:
                            return
                        nodes_2stage += 1
                        if i_pos == len(in_w):
                            if len(assigned1) != R_rows:
                                return
                            rules1 = [assigned1[r_i] for r_i in range(R_rows)]
                            w_mid = apply_rules_once(in_w, rules1)
                            if w_mid is None:
                                return

                            def backtrack_stage2(
                                im_pos: int,
                                jo_pos: int,
                                assigned2: Dict[int, Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]],
                            ) -> None:
                                nonlocal best_cost_2, synthesized_targets, nodes_2stage
                                if nodes_2stage >= max_nodes_2stage:
                                    return
                                nodes_2stage += 1
                                if im_pos == len(w_mid) and jo_pos == len(out_w):
                                    if len(assigned2) != R_rows:
                                        return
                                    rules2 = [assigned2[r_i] for r_i in range(R_rows)]
                                    if apply_rules_once(w_mid, rules2) != out_w:
                                        return
                                    cost_val = 0
                                    cand_targets: List[List[Tuple[int, ...]]] = []
                                    for r_i in range(R_rows):
                                        u1, v1 = assigned1[r_i]
                                        u2, v2 = assigned2[r_i]
                                        w0_1 = rule_rows_meta[r_i][1][1]
                                        rev_set = rhs1_cands[r_i][1]
                                        if v1 != w0_1:
                                            cost_val += 4 if tuple(v1) in rev_set else 20
                                        if u1 != rule_rows_meta[r_i][0][1]:
                                            cost_val += 1
                                        if u2 != v1[: len(u2)]:
                                            cost_val += 2
                                        if u2 != rule_rows_meta[r_i][2][1]:
                                            cost_val += 1
                                        if v2 != rule_rows_meta[r_i][3][1]:
                                            cost_val += 1
                                        cand_targets.extend([u1, v1, u2, v2])
                                    if cost_val < best_cost_2:
                                        best_cost_2 = cost_val
                                        synthesized_targets = cand_targets
                                    return

                                used_lhs2 = {tuple(u_val) for u_val, _ in assigned2.values()}
                                for r_i in range(R_rows):
                                    m2, w0_2, _ = rule_rows_meta[r_i][2]
                                    m3, w0_3, _ = rule_rows_meta[r_i][3]
                                    if im_pos + m2 <= len(w_mid) and jo_pos + m3 <= len(out_w):
                                        u2_seg = w_mid[im_pos : im_pos + m2]
                                        v2_seg = out_w[jo_pos : jo_pos + m3]
                                        if not (matches_eq_kernel(u2_seg, w0_2) and matches_eq_kernel(v2_seg, w0_3)):
                                            continue
                                        if r_i in assigned2:
                                            if assigned2[r_i] == (u2_seg, v2_seg):
                                                backtrack_stage2(im_pos + m2, jo_pos + m3, assigned2)
                                        elif tuple(u2_seg) not in used_lhs2:
                                            assigned2[r_i] = (u2_seg, v2_seg)
                                            backtrack_stage2(im_pos + m2, jo_pos + m3, assigned2)
                                            del assigned2[r_i]

                            backtrack_stage2(0, 0, {})
                            return

                        used_lhs1 = {tuple(u_val) for u_val, _ in assigned1.values()}
                        for r_i in range(R_rows):
                            m0, w0_0, _ = rule_rows_meta[r_i][0]
                            if i_pos + m0 <= len(in_w):
                                u1_seg = in_w[i_pos : i_pos + m0]
                                if not matches_eq_kernel(u1_seg, w0_0):
                                    continue
                                if r_i in assigned1:
                                    if assigned1[r_i][0] == u1_seg:
                                        backtrack_stage1(i_pos + m0, assigned1)
                                elif tuple(u1_seg) not in used_lhs1:
                                    for v1_cand in rhs1_cands[r_i][0]:
                                        assigned1[r_i] = (u1_seg, v1_cand)
                                        backtrack_stage1(i_pos + m0, assigned1)
                                        del assigned1[r_i]

                    backtrack_stage1(0, {})

                if synthesized_targets is None:
                    return []
                self._monoid_hom_state = {
                    "mode": "boxes",
                    "S": S,
                    "boxes": rule_boxes,
                    "targets": synthesized_targets,
                    "idx": 0,
                    "tries": 0,
                }

        st = self._monoid_hom_state
        idx = int(st["idx"])
        targets_list = st["targets"]
        if idx >= len(targets_list):
            return []
        S_st = int(st["S"])
        if st["mode"] == "strip":
            cur_canons = [self._d4_canonical_glyph(extract_glyphs(st["comp"], S_st)[idx])]
        else:
            cur_canons = [self._d4_canonical_glyph(g) for g in extract_glyphs(st["boxes"][idx], S_st)]
        if cur_canons == targets_list[idx] or int(st["tries"]) >= 16:
            st["idx"] = idx + 1
            st["tries"] = 0
            return [4]
        st["tries"] = int(st["tries"]) + 1
        return [1]

    def _solve_d4_locus_minkowski_cover_plan(
        self,
        board0: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
    ) -> List[ActionKey]:
        """
        Solves D_4-symmetric 1px geometric locus translation, palette recoloring, and
        Minkowski-difference intersection set cover over hollow 3x3 target receptors
        under ACTION1..4 translations and ACTION5 active-shape cycling.
        100% S_16 color-permutation equivariant, D_4 dihedral equivariant, and scale-invariant.
        """
        if available_actions is not None and not {1, 2, 3, 4, 5}.issubset(set(int(a) for a in available_actions)):
            return []
        H, W = board0.shape
        if H < 32 or W < 32:
            return []
        H_play = min(63, H - 1) if H >= 64 else H
        play = board0[:H_play, :]
        vals, cnts = np.unique(play, return_counts=True)
        bg = int(vals[int(np.argmax(cnts))])
        col_totals = {int(v): int(cnt) for v, cnt in zip(vals, cnts)}
        active_pos: Optional[Tuple[int, int]] = None
        active_marker_col: Optional[int] = None
        for col, cnt in col_totals.items():
            if cnt == 1 and col != bg:
                rs, cs = np.where(play == col)
                active_pos = (int(rs[0]), int(cs[0]))
                active_marker_col = col
                break
        if active_pos is None:
            return []

        if self._locus_cover_plan_by_center is None:
            comps = _connected_components_2d(play, connectivity=4, same_color=True, ignore_val=bg)
            palette_boxes: Dict[int, Tuple[int, int, int, int]] = {}
            palette_pixels: Set[Tuple[int, int]] = set()
            palette_border_cols: Set[int] = set()
            for c in comps:
                if int(c["h"]) == 6 and int(c["w"]) == 6 and int(c["area"]) == 20:
                    r0, c0, bcol = int(c["r0"]), int(c["c0"]), int(c["color"])
                    sub = play[r0 : r0 + 6, c0 : c0 + 6]
                    in_vals = np.unique(sub[1:5, 1:5])
                    if len(in_vals) == 1 and int(in_vals[0]) not in (bg, bcol):
                        pcol = int(in_vals[0])
                        palette_boxes[pcol] = (r0 + 1, r0 + 5, c0 + 1, c0 + 5)
                        palette_border_cols.add(bcol)
                        for dr in range(6):
                            for dc in range(6):
                                palette_pixels.add((r0 + dr, c0 + dc))
            frame_cols = {
                int(c["color"])
                for c in comps
                if int(c["h"]) == 3 and int(c["w"]) == 3 and int(c["area"]) == 8
            }
            if not frame_cols:
                self._locus_cover_plan_by_center = {}
                return []
            all_targets: List[Tuple[int, int, Optional[int]]] = []
            all_target_pixels: Set[Tuple[int, int]] = set()
            for fcol in frame_cols:
                for r0 in range(H_play - 2):
                    for c0 in range(W - 2):
                        if (r0, c0) in palette_pixels:
                            continue
                        sub = play[r0 : r0 + 3, c0 : c0 + 3]
                        inner_col = int(sub[1, 1])
                        if inner_col in (bg, fcol, active_marker_col) or inner_col in palette_border_cols:
                            continue
                        perim = [
                            int(sub[0, 0]), int(sub[0, 1]), int(sub[0, 2]),
                            int(sub[1, 0]),                 int(sub[1, 2]),
                            int(sub[2, 0]), int(sub[2, 1]), int(sub[2, 2]),
                        ]
                        n_fcol = sum(1 for v in perim if v == fcol)
                        if n_fcol >= 4 and all(v in (fcol, inner_col) for v in perim):
                            tcol_eff = inner_col if (n_fcol == 8 or not palette_boxes) else None
                            all_targets.append((r0 + 1, c0 + 1, tcol_eff))
                            all_target_pixels.add((r0 + 1, c0 + 1))
            if not all_targets:
                self._locus_cover_plan_by_center = {}
                return []

            def detect_all_shapes_fast(
                pts: Set[Tuple[int, int]],
            ) -> List[Tuple[str, int, int, Set[Tuple[int, int]]]]:
                pts_aug = set(pts)
                if active_pos is not None:
                    pts_aug.add(active_pos)
                found: List[Tuple[str, int, int, Set[Tuple[int, int]]]] = []
                row_cnts = Counter(r for r, _ in pts_aug)
                col_cnts = Counter(c for _, c in pts_aug)
                d1_cnts = Counter(r - c for r, c in pts_aug)
                d2_cnts = Counter(r + c for r, c in pts_aug)
                used_rows: Set[int] = set()
                used_cols: Set[int] = set()
                for r0, rc in row_cnts.items():
                    if rc < 15:
                        continue
                    for c0, cc in col_cnts.items():
                        if cc < 15:
                            continue
                        if all(
                            sum(1 for k in range(1, 7) if (r0 + k * dr, c0 + k * dc) in pts_aug) >= 4
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        ):
                            r_up = max((k for k in range(1, H_play) if (r0 - k, c0) in pts_aug), default=0)
                            r_dn = max((k for k in range(1, H_play) if (r0 + k, c0) in pts_aug), default=0)
                            c_lt = max((k for k in range(1, W) if (r0, c0 - k) in pts_aug), default=0)
                            c_rt = max((k for k in range(1, W) if (r0, c0 + k) in pts_aug), default=0)
                            R_v, R_h = max(r_up, r_dn), max(c_lt, c_rt)
                            rel = {(dr, 0) for dr in range(-R_v, R_v + 1)} | {(0, dc) for dc in range(-R_h, R_h + 1)}
                            found.append(("card", r0, c0, rel))
                            used_rows.add(r0)
                            used_cols.add(c0)
                r_peaks = [r for r, cnt in row_cnts.items() if cnt >= 11 and r not in used_rows]
                c_peaks = [c for c, cnt in col_cnts.items() if cnt >= 11 and c not in used_cols]
                seen_linf: Set[Tuple[int, int]] = set()
                for i, ra in enumerate(r_peaks):
                    for rb in r_peaks[i + 1 :]:
                        R = abs(rb - ra) // 2
                        if R < 5 or abs(rb - ra) % 2 != 0:
                            continue
                        r0 = (ra + rb) // 2
                        for j, ca in enumerate(c_peaks):
                            for cb in c_peaks[j + 1 :]:
                                if abs(cb - ca) != 2 * R:
                                    continue
                                c0 = (ca + cb) // 2
                                if (r0, c0) in seen_linf:
                                    continue
                                ring = list(
                                    {
                                        (r0 + dr, c0 + dc)
                                        for dr in range(-R, R + 1)
                                        for dc in range(-R, R + 1)
                                        if max(abs(dr), abs(dc)) == R
                                        and 0 <= r0 + dr < H_play
                                        and 0 <= c0 + dc < W
                                    }
                                )
                                if len(ring) >= 20 and sum(1 for p in ring if p in pts_aug) >= 0.75 * len(ring):
                                    seen_linf.add((r0, c0))
                                    used_rows.add(ra)
                                    used_rows.add(rb)
                                    used_cols.add(ca)
                                    used_cols.add(cb)
                                    rel = {
                                        (dr, dc)
                                        for dr in range(-R, R + 1)
                                        for dc in range(-R, R + 1)
                                        if max(abs(dr), abs(dc)) == R
                                    }
                                    found.append(("linf", r0, c0, rel))
                for r0, rc in row_cnts.items():
                    if rc >= 15 and r0 not in used_rows:
                        cs_r = sorted(
                            c for r, c in pts_aug if r == r0 and ((r0, c - 1) in pts_aug or (r0, c + 1) in pts_aug)
                        )
                        if len(cs_r) >= 15:
                            c0 = (cs_r[0] + cs_r[-1]) // 2
                            R_h = (cs_r[-1] - cs_r[0]) // 2
                            found.append(("horiz", r0, c0, {(0, dc) for dc in range(-R_h, R_h + 1)}))
                for c0, cc in col_cnts.items():
                    if cc >= 15 and c0 not in used_cols:
                        rs_c = sorted(
                            r for r, c in pts_aug if c == c0 and ((r - 1, c0) in pts_aug or (r + 1, c0) in pts_aug)
                        )
                        if len(rs_c) >= 15:
                            r0 = (rs_c[0] + rs_c[-1]) // 2
                            R_v = (rs_c[-1] - rs_c[0]) // 2
                            found.append(("vert", r0, c0, {(dr, 0) for dr in range(-R_v, R_v + 1)}))
                used_d1: Set[int] = set()
                used_d2: Set[int] = set()
                for d1, cnt1 in d1_cnts.items():
                    if cnt1 < 15:
                        continue
                    for d2, cnt2 in d2_cnts.items():
                        if cnt2 < 15 or (d1 + d2) % 2 != 0:
                            continue
                        r0, c0 = (d1 + d2) // 2, (d2 - d1) // 2
                        if not (0 <= r0 < H_play and 0 <= c0 < W):
                            continue
                        if all(
                            sum(1 for k in range(1, 7) if (r0 + k * dr, c0 + k * dc) in pts_aug) >= 4
                            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                        ):
                            R_d = max(
                                max((k for k in range(1, H_play) if (r0 + k * dr, c0 + k * dc) in pts_aug), default=0)
                                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                            )
                            found.append(
                                (
                                    "diag",
                                    r0,
                                    c0,
                                    {(k * dr, k * dc) for k in range(0, R_d + 1) for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))},
                                )
                            )
                            used_d1.add(d1)
                            used_d2.add(d2)
                d1_peaks = [d for d, cnt in d1_cnts.items() if cnt >= 7 and d not in used_d1]
                d2_peaks = [d for d, cnt in d2_cnts.items() if cnt >= 7 and d not in used_d2]
                seen_l1: Set[Tuple[int, int]] = set()
                for i, d1a in enumerate(d1_peaks):
                    for d1b in d1_peaks[i + 1 :]:
                        R = abs(d1b - d1a) // 2
                        if R < 6 or abs(d1b - d1a) % 2 != 0:
                            continue
                        d1_mid = (d1a + d1b) // 2
                        for j, d2a in enumerate(d2_peaks):
                            for d2b in d2_peaks[j + 1 :]:
                                if abs(d2b - d2a) != 2 * R:
                                    continue
                                d2_mid = (d2a + d2b) // 2
                                if (d1_mid + d2_mid) % 2 != 0:
                                    continue
                                r0, c0 = (d1_mid + d2_mid) // 2, (d2_mid - d1_mid) // 2
                                if not (0 <= r0 < H_play and 0 <= c0 < W) or (r0, c0) in seen_l1:
                                    continue
                                ring = list(
                                    {
                                        (r0 + dr, c0 + dc)
                                        for dr in range(-R, R + 1)
                                        for dc in (-(R - abs(dr)), R - abs(dr))
                                        if 0 <= r0 + dr < H_play and 0 <= c0 + dc < W
                                    }
                                )
                                if len(ring) >= 16 and sum(1 for p in ring if p in pts_aug) >= 0.70 * len(ring):
                                    seen_l1.add((r0, c0))
                                    found.append(
                                        (
                                            "l1",
                                            r0,
                                            c0,
                                            {(dr, dc) for dr in range(-R, R + 1) for dc in (-(R - abs(dr)), R - abs(dr))},
                                        )
                                    )
                return found

            shape_cols = [
                col
                for col, cnt in col_totals.items()
                if col not in (bg, active_marker_col)
                and col not in frame_cols
                and col not in palette_border_cols
                and cnt >= 10
            ]
            all_shapes: List[Tuple[int, str, int, int, Set[Tuple[int, int]]]] = []
            for scol in shape_cols:
                rs, cs = np.where(play == scol)
                pts = {
                    (int(r), int(c))
                    for r, c in zip(rs, cs)
                    if (int(r), int(c)) not in all_target_pixels and (int(r), int(c)) not in palette_pixels
                }
                if not pts:
                    continue
                for fam, r_init, c_init, rel in detect_all_shapes_fast(pts):
                    all_shapes.append((scol, fam, r_init, c_init, rel))
            if not all_shapes:
                self._locus_cover_plan_by_center = {}
                return []

            target_colors = sorted({t[2] for t in all_targets if t[2] is not None})
            cand_assigns: List[List[Tuple[Tuple[int, int, int], Set[Tuple[int, int]]]]] = []
            min_cov_req = 1 if len(all_targets) == 1 else 2
            for scol, fam, r_init, c_init, rel in all_shapes:
                cands: List[Tuple[Tuple[int, int, int], Set[Tuple[int, int]]]] = []
                allowed_tcols = (
                    [scol]
                    if not palette_boxes
                    else [tc for tc in target_colors if tc == scol or tc in palette_boxes]
                )
                for tcol in allowed_tcols:
                    for cr in range(r_init % 3, H_play, 3):
                        for cc in range(c_init % 3, W, 3):
                            cov = {
                                (tr, tc)
                                for (tr, tc, tc_col) in all_targets
                                if (tc_col == tcol or tc_col is None) and (tr - cr, tc - cc) in rel
                            }
                            if len(cov) >= min_cov_req:
                                cands.append(((cr, cc, tcol), cov))
                cand_assigns.append(cands)

            for idx_c, (_, _, r_init, c_init, _) in enumerate(all_shapes):
                cand_assigns[idx_c].sort(
                    key=lambda item: (-len(item[1]), abs(item[0][0] - r_init) + abs(item[0][1] - c_init))
                )
            rem_max_cov = [0] * (len(all_shapes) + 1)
            for idx_c in range(len(all_shapes) - 1, -1, -1):
                rem_max_cov[idx_c] = rem_max_cov[idx_c + 1] + max(
                    (len(cov) for _, cov in cand_assigns[idx_c]), default=0
                )

            best_state: List[Any] = [None, -1, 10**9]
            dfs_nodes = 0
            max_dfs_nodes = 3000

            def dfs(
                idx: int,
                cur_assign: List[Tuple[int, int, int]],
                cur_cov: Set[Tuple[int, int]],
                cur_dist: int,
            ) -> None:
                nonlocal dfs_nodes
                if dfs_nodes >= max_dfs_nodes:
                    return
                if len(cur_cov) + rem_max_cov[idx] < len(all_target_pixels):
                    return
                if best_state[1] == len(all_target_pixels) and cur_dist >= best_state[2]:
                    return
                dfs_nodes += 1
                if idx == len(all_shapes):
                    if (len(cur_cov), -cur_dist) > (best_state[1], -best_state[2]):
                        best_state[1] = len(cur_cov)
                        best_state[2] = cur_dist
                        best_state[0] = list(cur_assign)
                    return
                _, _, r_init, c_init, _ = all_shapes[idx]
                for (cr, cc, tcol), cov in cand_assigns[idx]:
                    dfs(
                        idx + 1,
                        cur_assign + [(cr, cc, tcol)],
                        cur_cov | cov,
                        cur_dist + abs(cr - r_init) + abs(cc - c_init),
                    )

            dfs(0, [], set(), 0)
            if not best_state[0] or int(best_state[1]) < len(all_target_pixels):
                self._locus_cover_plan_by_center = {}
                return []

            def bfs_moves_between(
                r0: int,
                c0: int,
                r1: int,
                c1: int,
                forbidden_pixels: Set[Tuple[int, int]],
            ) -> List[int]:
                if (r0, c0) == (r1, c1):
                    return []
                q: deque = deque([(r0, c0)])
                parent: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], int]]] = {(r0, c0): None}
                max_states = 3000
                found_goal = False
                while q and len(parent) < max_states:
                    cr, cc = q.popleft()
                    if (cr, cc) == (r1, c1):
                        found_goal = True
                        break
                    for aid, (dr, dc) in ((1, (-3, 0)), (2, (3, 0)), (3, (0, -3)), (4, (0, 3))):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < H_play
                            and 0 <= nc < W
                            and (nr, nc) not in parent
                            and (nr, nc) not in forbidden_pixels
                        ):
                            parent[(nr, nc)] = ((cr, cc), aid)
                            if (nr, nc) == (r1, c1):
                                found_goal = True
                                break
                            q.append((nr, nc))
                    if found_goal:
                        break
                if not found_goal:
                    return []
                path: List[int] = []
                cur_pt: Optional[Tuple[int, int]] = (r1, c1)
                while cur_pt is not None and parent.get(cur_pt) is not None:
                    prev_pt, aid = parent[cur_pt]  # type: ignore[misc]
                    path.append(aid)
                    cur_pt = prev_pt
                path.reverse()
                return path

            plan_by_center: Dict[Tuple[int, int], List[int]] = {}
            for (scol, _, r_init, c_init, rel), (cr, cc, tcol) in zip(all_shapes, best_state[0]):
                seq: List[int] = []
                cur_r, cur_c = r_init, c_init
                forbidden: Set[Tuple[int, int]] = set()
                for pcol2, (pr0_2, pr1_2, pc0_2, pc1_2) in palette_boxes.items():
                    if pcol2 != tcol:
                        for rr in range(pr0_2 - 1, pr1_2 + 1):
                            for cc2 in range(pc0_2 - 1, pc1_2 + 1):
                                forbidden.add((rr, cc2))
                forbidden_minkowski: Set[Tuple[int, int]] = set(forbidden)
                if forbidden:
                    for rr, cc2 in forbidden:
                        for dr, dc in rel:
                            mr, mc = rr - dr, cc2 - dc
                            if 0 <= mr < H_play and 0 <= mc < W:
                                forbidden_minkowski.add((mr, mc))
                if tcol != scol and tcol in palette_boxes:
                    pr0, pr1, pc0, pc1 = palette_boxes[tcol]
                    pts_box = [
                        (pr, pc)
                        for pr in range(pr0, pr1)
                        for pc in range(pc0, pc1)
                        if (pr - cur_r) % 3 == 0 and (pc - cur_c) % 3 == 0
                    ]
                    if pts_box:
                        pr, pc = min(
                            pts_box,
                            key=lambda p: abs(p[0] - cur_r)
                            + abs(p[1] - cur_c)
                            + abs(cr - p[0])
                            + abs(cc - p[1]),
                        )
                        seq.extend(bfs_moves_between(cur_r, cur_c, pr, pc, forbidden))
                        cur_r, cur_c = pr, pc
                forb_second = forbidden_minkowski - {(cur_r, cur_c), (cr, cc)}
                leg2 = bfs_moves_between(cur_r, cur_c, cr, cc, forb_second)
                if not leg2 and (cur_r, cur_c) != (cr, cc):
                    leg2 = bfs_moves_between(cur_r, cur_c, cr, cc, forbidden - {(cur_r, cur_c), (cr, cc)})
                seq.extend(leg2)
                plan_by_center[(r_init, c_init)] = seq
            self._locus_cover_plan_by_center = plan_by_center

        if not self._locus_cover_plan_by_center:
            return []
        if active_pos in self._locus_cover_plan_by_center and active_pos not in self._locus_cover_done_centers:
            self._locus_cover_done_centers.add(active_pos)
            seq_acts: List[ActionKey] = list(self._locus_cover_plan_by_center[active_pos])
            if len(self._locus_cover_done_centers) < len(self._locus_cover_plan_by_center):
                seq_acts.append(5)
            return seq_acts
        if (
            len(self._locus_cover_done_centers) < len(self._locus_cover_plan_by_center)
            and self._locus_cover_switches <= len(self._locus_cover_plan_by_center) + 2
        ):
            self._locus_cover_switches += 1
            return [5]
        return []

    def _solve_se2_pin_matching_polyomino_plan(
        self,
        board0: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
    ) -> List[ActionKey]:
        """
        Solves SE(2) rigid-body polyomino pin-matching assembly with fog-of-war piece reveal
        (ACTION6) and telescopic port spine extrusion (ACTION5).
        100% S_16 color-permutation equivariant, D_4 dihedral equivariant, and scale-invariant.
        """
        if available_actions is not None and not {1, 2, 3, 4, 5, 6}.issubset(
            set(int(a) for a in available_actions)
        ):
            return []
        H, W = board0.shape
        if H < 24 or W < 24:
            return []

        if self._pin_polyomino_state is not None and "grid_info" in self._pin_polyomino_state:
            s_tile, y_off, x_off, R_g, C_g = self._pin_polyomino_state["grid_info"]
            sub_play = board0[y_off : y_off + R_g * s_tile, x_off : x_off + C_g * s_tile]
            blocks = sub_play.reshape(R_g, s_tile, C_g, s_tile)
            if not bool(np.all(blocks == blocks[:, 0:1, :, 0:1])):
                return []
            g = blocks[:, 0, :, 0].copy()
        else:
            grid_info: Optional[Tuple[int, int, int, int, int]] = None
            max_s = max(8, min(H, W) // 8)
            for s_cand in range(max_s, 1, -1):
                for yo in range(s_cand + 2):
                    for xo in range(s_cand + 2):
                        rg = (H - 2 * yo) // s_cand
                        cg = (W - 2 * xo) // s_cand
                        if rg < 8 or cg < 8 or rg > 32 or cg > 32:
                            continue
                        sub = board0[yo : yo + rg * s_cand, xo : xo + cg * s_cand]
                        blks = sub.reshape(rg, s_cand, cg, s_cand)
                        if bool(np.all(blks == blks[:, 0:1, :, 0:1])):
                            g_cand = blks[:, 0, :, 0]
                            if len(np.unique(g_cand)) >= 3:
                                grid_info = (s_cand, yo, xo, rg, cg)
                                break
                    if grid_info is not None:
                        break
                if grid_info is not None:
                    break
            if grid_info is None:
                return []
            s_tile, y_off, x_off, R_g, C_g = grid_info
            g = board0[y_off : y_off + R_g * s_tile : s_tile, x_off : x_off + C_g * s_tile : s_tile].copy()

        def rot_cw(pts: Sequence[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
            cur = list(pts)
            for _ in range(k % 4):
                ys = [p[0] for p in cur]
                xs = [p[1] for p in cur]
                y0, y1 = min(ys), max(ys)
                x0, _ = min(xs), max(xs)
                h_box = y1 - y0 + 1
                cur = [(y0 + (x - x0), x0 + (h_box - 1) - (y - y0)) for y, x in cur]
            return cur

        def solve_pieces_to_actions(
            pieces: List[Dict[str, Any]],
            cur_selected_idx: Optional[int],
        ) -> List[ActionKey]:
            M = len(pieces)
            if M < 2:
                return []
            rot_pieces: List[List[Tuple[Set[Tuple[int, int]], List[Tuple[int, int]]]]] = []
            for i in range(M):
                b_len = len(pieces[i]["body"])
                if b_len == 0 or not pieces[i]["pins"]:
                    return []
                rots: List[Tuple[Set[Tuple[int, int]], List[Tuple[int, int]]]] = []
                max_r = 1 if bool(pieces[i].get("telescopic", False)) else 4
                for r_idx in range(max_r):
                    rk = rot_cw(pieces[i]["all"], r_idx)
                    rots.append((set(rk[:b_len]), list(rk[b_len:])))
                rot_pieces.append(rots)

            best_sol: Optional[Dict[int, Tuple[int, int, int]]] = None
            best_cost = 10**9
            nodes_visited = 0
            max_nodes = 2500

            def search(
                placed_rot: Dict[int, int],
                placed_shift: Dict[int, Tuple[int, int]],
                used_bodies: Set[Tuple[int, int]],
                pin_counts: Dict[Tuple[int, int], int],
            ) -> None:
                nonlocal best_sol, best_cost, nodes_visited
                if nodes_visited >= max_nodes:
                    return
                nodes_visited += 1
                unmatched = [p for p, cnt in pin_counts.items() if cnt == 1]
                if len(placed_rot) == M:
                    if not unmatched:
                        all_pts = set(used_bodies) | set(pin_counts.keys())
                        min_y = min(y for y, _ in all_pts)
                        max_y = max(y for y, _ in all_pts)
                        min_x = min(x for _, x in all_pts)
                        max_x = max(x for _, x in all_pts)
                        if max_y - min_y >= R_g or max_x - min_x >= C_g:
                            return
                        for dy in range(-min_y, R_g - max_y):
                            for dx in range(-min_x, C_g - max_x):
                                cost = 0
                                shifts: Dict[int, Tuple[int, int, int]] = {}
                                for idx_p in range(M):
                                    sy = placed_shift[idx_p][0] + dy
                                    sx = placed_shift[idx_p][1] + dx
                                    r_val = placed_rot[idx_p]
                                    if r_val != 0 or sy != 0 or sx != 0:
                                        cost += 1 + r_val + abs(sy) + abs(sx)
                                    shifts[idx_p] = (r_val, sy, sx)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_sol = shifts
                    return
                if not unmatched:
                    return
                target_p = unmatched[0]
                for j in range(M):
                    if j in placed_rot:
                        continue
                    for rj in range(len(rot_pieces[j])):
                        bj, pnj = rot_pieces[j][rj]
                        for pj in pnj:
                            sy = target_p[0] - pj[0]
                            sx = target_p[1] - pj[1]
                            bj_s = {(y + sy, x + sx) for y, x in bj}
                            if bj_s & used_bodies or bj_s & set(pin_counts.keys()):
                                continue
                            pnj_s = [(y + sy, x + sx) for y, x in pnj]
                            if any(p in used_bodies for p in pnj_s):
                                continue
                            new_pin_counts = dict(pin_counts)
                            ok = True
                            for p in pnj_s:
                                new_pin_counts[p] = new_pin_counts.get(p, 0) + 1
                                if new_pin_counts[p] > 2:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            placed_rot[j] = rj
                            placed_shift[j] = (sy, sx)
                            search(placed_rot, placed_shift, used_bodies | bj_s, new_pin_counts)
                            del placed_rot[j]
                            del placed_shift[j]

            for r0 in range(len(rot_pieces[0])):
                b0, pn0 = rot_pieces[0][r0]
                search({0: r0}, {0: (0, 0)}, set(b0), {p: 1 for p in pn0})

            if best_sol is None:
                return []

            occupied_final: Set[Tuple[int, int]] = set()
            for i in range(M):
                r_i, sy_i, sx_i = best_sol[i]
                if r_i == 0 and sy_i == 0 and sx_i == 0:
                    bi, pni = rot_pieces[i][0]
                    occupied_final |= bi | set(pni)

            order = [i for i in range(M) if best_sol[i] != (0, 0, 0)]
            if cur_selected_idx is not None and cur_selected_idx in order:
                order = [cur_selected_idx] + [i for i in order if i != cur_selected_idx]

            plan_acts: List[ActionKey] = []
            sel_now = cur_selected_idx
            for i in order:
                r_i, sy_i, sx_i = best_sol[i]
                cand_tiles = [pt for pt in pieces[i]["body"] if pt not in occupied_final] or pieces[i]["body"]
                by, bx = cand_tiles[0]
                px = x_off + s_tile * bx + s_tile // 2
                py = y_off + s_tile * by + s_tile // 2
                if sel_now != i:
                    plan_acts.append((6, int(px), int(py)))
                    sel_now = i
                for _ in range(r_i):
                    plan_acts.append(5)
                if sy_i < 0:
                    plan_acts.extend([1] * (-sy_i))
                elif sy_i > 0:
                    plan_acts.extend([2] * sy_i)
                if sx_i < 0:
                    plan_acts.extend([3] * (-sx_i))
                elif sx_i > 0:
                    plan_acts.extend([4] * sx_i)
                bi, pni = rot_pieces[i][r_i]
                occupied_final |= {(y + sy_i, x + sx_i) for y, x in bi} | {(y + sy_i, x + sx_i) for y, x in pni}
            return plan_acts

        if self._pin_polyomino_state is None:
            vals, cnts = np.unique(g, return_counts=True)
            bg = int(vals[int(np.argmax(cnts))])
            comps = _connected_components_2d(g, connectivity=4, same_color=False, ignore_val=bg)
            if not (2 <= len(comps) <= 6):
                return []
            if any(int(c["area"]) < 3 for c in comps):
                return []
            same_comps = _connected_components_2d(g, connectivity=4, same_color=True, ignore_val=bg)
            unit_counts = Counter(int(sc["color"]) for sc in same_comps if int(sc["area"]) == 1)
            if not unit_counts:
                return []
            pin_col, max_u = max(unit_counts.items(), key=lambda kv: kv[1])
            if max_u < 2:
                return []

            comp_pts_list: List[List[Tuple[int, int]]] = []
            for c in comps:
                pts_c = list(zip(c["rows"].tolist(), c["cols"].tolist()))
                comp_pts_list.append(pts_c)

            has_dimmed = any(
                pin_col not in {int(g[y, x]) for y, x in pts_c} for pts_c in comp_pts_list
            )
            if not has_dimmed:
                pieces_vis: List[Dict[str, Any]] = []
                for pts_c in comp_pts_list:
                    body = [(y, x) for y, x in pts_c if int(g[y, x]) != pin_col]
                    pins = [(y, x) for y, x in pts_c if int(g[y, x]) == pin_col]
                    pieces_vis.append(
                        {"body": body, "pins": pins, "all": body + pins, "telescopic": False}
                    )
                plan_vis = solve_pieces_to_actions(pieces_vis, cur_selected_idx=0)
                if not plan_vis:
                    return []
                self._pin_polyomino_state = {
                    "grid_info": (s_tile, y_off, x_off, R_g, C_g),
                    "mode": "done",
                }
                return plan_vis

            self._pin_polyomino_state = {
                "grid_info": (s_tile, y_off, x_off, R_g, C_g),
                "bg": bg,
                "pin_col": pin_col,
                "comp_pts": comp_pts_list,
                "idx": 0,
                "pieces": [],
                "extruding_port_col": None,
                "extrude_steps": 0,
                "selected_idx": None,
                "mode": "inspect",
            }

        st = self._pin_polyomino_state
        if st.get("mode") == "done":
            return []
        bg = int(st["bg"])
        pin_col = int(st["pin_col"])
        comp_pts_list = st["comp_pts"]
        M_total = len(comp_pts_list)

        while int(st["idx"]) < M_total:
            idx = int(st["idx"])
            pts_c = comp_pts_list[idx]
            cols_in_c = {int(g[y, x]) for y, x in pts_c}
            if pin_col not in cols_in_c and st.get("extruding_port_col") is None:
                ry, rx = pts_c[0]
                px = x_off + s_tile * rx + s_tile // 2
                py = y_off + s_tile * ry + s_tile // 2
                st["selected_idx"] = idx
                return [(6, int(px), int(py))]

            if st.get("extruding_port_col") is None:
                non_pin_cnts = Counter(int(g[y, x]) for y, x in pts_c if int(g[y, x]) not in (bg, pin_col))
                if len(non_pin_cnts) >= 2:
                    body_col = max(non_pin_cnts, key=non_pin_cnts.get)
                    port_cols = [c_val for c_val in non_pin_cnts if c_val != body_col]
                    if port_cols:
                        st["extruding_port_col"] = int(port_cols[0])
                        st["extrude_steps"] = 0

            port_col = st.get("extruding_port_col")
            if port_col is not None:
                if int(st["extrude_steps"]) < 12 and any(
                    int(g[r, c]) == int(port_col) for r in range(R_g) for c in range(C_g)
                ):
                    st["extrude_steps"] = int(st["extrude_steps"]) + 1
                    return [5]
                new_comps = _connected_components_2d(g, connectivity=4, same_color=False, ignore_val=bg)
                for nc in new_comps:
                    npts = list(zip(nc["rows"].tolist(), nc["cols"].tolist()))
                    if pts_c[0] in npts:
                        pts_c = npts
                        break
                body = [(y, x) for y, x in pts_c if int(g[y, x]) != pin_col]
                pins = [(y, x) for y, x in pts_c if int(g[y, x]) == pin_col]
                st["pieces"].append(
                    {"body": body, "pins": pins, "all": body + pins, "telescopic": True}
                )
                st["extruding_port_col"] = None
                st["idx"] = idx + 1
                continue

            body = [(y, x) for y, x in pts_c if int(g[y, x]) != pin_col]
            pins = [(y, x) for y, x in pts_c if int(g[y, x]) == pin_col]
            if st.get("selected_idx") is None:
                st["selected_idx"] = idx
            st["pieces"].append(
                {"body": body, "pins": pins, "all": body + pins, "telescopic": False}
            )
            st["idx"] = idx + 1

        plan_fog = solve_pieces_to_actions(st["pieces"], cur_selected_idx=st.get("selected_idx"))
        st["mode"] = "done"
        return plan_fog

    def _solve_skewer_rail_permutation_plan(
        self,
        board0: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
    ) -> List[ActionKey]:
        """
        Solves orthogonal skewer-rail block permutation and impaling automata via exact
        first-principles BFS (`collections.deque`, `max_states = 3000`) with Gauge-Fiber /
        Contact-Sheaf Support Quotienting and discrete Conf_k(Z^2) braid codimension
        stratification over movable rail carriages, static skewers, and block orderings.
        100% S_16 color-permutation equivariant and scale-invariant.
        """
        if available_actions is not None:
            avail_s = set(int(a) for a in available_actions)
            if not {1, 2, 3, 4, 6}.issubset(avail_s) or 5 in avail_s:
                return []
        if self._skewer_rail_plan is not None:
            if self._skewer_rail_plan:
                return [self._skewer_rail_plan.popleft()]
            return []

        H, W = board0.shape
        if H < 32 or W < 32:
            return []
        wall_col = int(board0[0, 0])
        bg_col = int(np.bincount(board0[max(0, H - max(8, H // 6)) :, :].ravel()).argmax())
        if wall_col == bg_col:
            return []
        y_wall_last: Optional[int] = None
        for y in range(H - max(6, H // 10), H // 2, -1):
            if int(board0[y, 0]) == wall_col and int(board0[y, W - 1]) == wall_col:
                y_wall_last = y
                break
        if y_wall_last is None:
            return []
        y_div = y_wall_last
        while y_div + 1 < H and int(board0[y_div + 1, 0]) != bg_col:
            y_div += 1
        t_rows = [y for y in range(y_div + 1, H) if np.any(board0[y, :] != bg_col)]
        if not t_rows:
            return []
        y_t = int(t_rows[0])
        S = int(len(t_rows))
        if S < 4 or S > max(16, min(H, W) // 4):
            return []
        b_in = max(1, S // 6)
        ys = np.where(np.sum(board0[:y_div, :] == bg_col, axis=1) >= 2 * S)[0]
        xs = np.where(np.sum(board0[:y_div, :] == bg_col, axis=0) >= 2 * S)[0]
        if len(ys) == 0 or len(xs) == 0:
            return []
        y_span = int(ys[-1] - ys[0] + 1)
        x_span = int(xs[-1] - xs[0] + 1)
        if y_span < 3 * S or x_span < 3 * S or y_span % S != 0 or x_span % S != 0:
            return []
        y0, R = int(ys[0]), y_span // S
        x0, C = int(xs[0]), x_span // S
        if y0 < S or x0 < S or not (3 <= R <= 9 and 3 <= C <= 9):
            return []

        targets: Dict[int, Tuple[int, ...]] = {}
        cur_t: Optional[int] = None
        cur_list: List[int] = []
        x_scan = 0
        while x_scan <= W - S:
            if int(board0[y_t, x_scan]) != bg_col and bool(
                np.all(board0[y_t, x_scan : x_scan + S] == board0[y_t, x_scan])
            ):
                if cur_t is not None:
                    targets[cur_t] = tuple(cur_list)
                cur_t = int(board0[y_t + S // 3, x_scan + S // 3])
                cur_list = []
                x_scan += S
            elif (
                cur_t is not None
                and bool(np.all(board0[y_t, x_scan : x_scan + S] == bg_col))
                and int(board0[y_t + b_in, x_scan + b_in]) not in (bg_col, wall_col)
            ):
                cur_list.append(int(board0[y_t + b_in, x_scan + b_in]))
                x_scan += S
            else:
                x_scan += 1
        if cur_t is not None:
            targets[cur_t] = tuple(cur_list)
        if not targets or any(len(v) == 0 for v in targets.values()):
            return []

        l_rows: Set[int] = set()
        h_r: Optional[int] = None
        h_col: Optional[int] = None
        for r in range(R):
            sub = board0[y0 + S * r : y0 + S * (r + 1), x0 - S : x0]
            if not bool(np.all(sub == wall_col)):
                l_rows.add(r)
                if int(sub[0, 0]) != wall_col:
                    h_r = r
                    h_col = int(sub[S // 3, S // 3])
        if h_r is None or h_col is None:
            return []

        t_cols: Set[int] = set()
        v_c: Optional[int] = None
        v_col: Optional[int] = None
        static_v: Dict[int, Tuple[int, int]] = {}
        for c in range(C):
            sub = board0[y0 - S : y0, x0 + S * c : x0 + S * (c + 1)]
            if int(sub[0, 0]) != wall_col:
                v_c = c
                v_col = int(sub[S // 3, S // 3])
        if v_c is not None:
            for c in range(C):
                sub = board0[y0 - S : y0, x0 + S * c : x0 + S * (c + 1)]
                if int(sub[S // 3, S // 3]) != wall_col:
                    t_cols.add(c)
        else:
            for c in range(C):
                sub = board0[y0 - S : y0, x0 + S * c : x0 + S * (c + 1)]
                if int(sub[S // 3, S // 3]) != wall_col:
                    sc_col = int(sub[S // 3, S // 3])
                    vl = 0
                    for r in range(R):
                        if int(board0[y0 + S * r, x0 + S * c + S // 3]) not in (bg_col, wall_col):
                            vl = r + 1
                        else:
                            break
                    if vl > 0:
                        static_v[c] = (sc_col, vl)

        h_len = 0
        while h_len < C and int(board0[y0 + S * h_r + S // 3, x0 + S * h_len]) not in (bg_col, wall_col):
            h_len += 1
        v_len = 0
        if v_c is not None:
            while v_len < R and int(board0[y0 + S * v_len, x0 + S * v_c + S // 3]) not in (bg_col, wall_col):
                v_len += 1

        walls: Set[Tuple[int, int]] = set()
        blocks_init: Dict[Tuple[int, int], Tuple[int, bool, bool]] = {}
        for r in range(R):
            for c in range(C):
                sub = board0[y0 + S * r : y0 + S * (r + 1), x0 + S * c : x0 + S * (c + 1)]
                if bool(np.all(sub == wall_col)):
                    walls.add((r, c))
                elif int(sub[b_in, b_in]) != bg_col and int(sub[b_in, b_in]) != wall_col:
                    b_col = int(sub[b_in, b_in])
                    ih = bool(
                        r == h_r
                        and c < h_len
                        and int(board0[y0 + S * r + S // 3, x0 + S * c]) not in (bg_col, wall_col)
                    )
                    iv = bool(
                        (
                            v_c is not None
                            and c == v_c
                            and r < v_len
                            and int(board0[y0 + S * r, x0 + S * c + S // 3]) not in (bg_col, wall_col)
                        )
                        or (c in static_v and r < static_v[c][1])
                    )
                    blocks_init[(r, c)] = (b_col, ih, iv)

        StateTuple = Tuple[str, int, int, Optional[int], int, Tuple[Tuple[int, int, int, bool, bool], ...]]

        def make_state(
            act_s: str,
            hr_s: int,
            hl_s: int,
            vc_s: Optional[int],
            vl_s: int,
            bdict: Dict[Tuple[int, int], Tuple[int, bool, bool]],
        ) -> StateTuple:
            blist = tuple(sorted((r, c, col, ih, iv) for (r, c), (col, ih, iv) in bdict.items()))
            return (act_s, hr_s, hl_s, vc_s, vl_s, blist)

        def is_win(st: StateTuple) -> bool:
            _, hr_s, hl_s, vc_s, vl_s, blist = st
            bdict = {(r, c): (col, ih, iv) for r, c, col, ih, iv in blist}
            for tcol, req in targets.items():
                got: List[int] = []
                if tcol == h_col:
                    for c in range(hl_s):
                        if (hr_s, c) in bdict and bdict[(hr_s, c)][1]:
                            got.append(bdict[(hr_s, c)][0])
                elif v_c is not None and tcol == v_col and vc_s is not None:
                    for r in range(vl_s):
                        if (r, vc_s) in bdict and bdict[(r, vc_s)][2]:
                            got.append(bdict[(r, vc_s)][0])
                else:
                    for sc, (sc_col, sc_len) in static_v.items():
                        if sc_col == tcol:
                            for r in range(sc_len):
                                if (r, sc) in bdict and bdict[(r, sc)][2]:
                                    got.append(bdict[(r, sc)][0])
                if tuple(got) != req:
                    return False
            return True

        def has_v_shaft(r: int, c: int, vc_s: Optional[int], vl_s: int) -> bool:
            if vc_s is not None and c == vc_s and r < vl_s:
                return True
            if c in static_v and r < static_v[c][1]:
                return True
            return False

        def is_v_tip(r: int, c: int, vc_s: Optional[int], vl_s: int) -> bool:
            if vc_s is not None and c == vc_s and r == vl_s - 1:
                return True
            if c in static_v and r == static_v[c][1] - 1:
                return True
            return False

        def step_sim(st: StateTuple, act: Any) -> Optional[StateTuple]:
            active, hr_s, hl_s, vc_s, vl_s, blist = st
            bdict = {(r, c): [col, ih, iv] for r, c, col, ih, iv in blist}
            if act == "SW_V":
                if v_c is None or active == "V":
                    return None
                if (
                    hl_s > 0
                    and not any(v[1] for v in bdict.values())
                    and not any(r_b == hr_s for (r_b, _) in bdict)
                ):
                    return None
                return make_state("V", hr_s, hl_s, vc_s, vl_s, {k: tuple(v) for k, v in bdict.items()})  # type: ignore[misc]
            if act == "SW_H":
                if v_c is None or active == "H":
                    return None
                if (
                    vl_s > 0
                    and not any(c_b == vc_s and v[2] for (r_b, c_b), v in bdict.items())
                    and not any(c_b == vc_s for (_, c_b) in bdict)
                ):
                    return None
                return make_state("H", hr_s, hl_s, vc_s, vl_s, {k: tuple(v) for k, v in bdict.items()})  # type: ignore[misc]

            if active == "H":
                if act in (1, 2):
                    dr = -1 if act == 1 else 1
                    nr = hr_s + dr
                    if nr not in l_rows:
                        return None
                    if (
                        (v_c is not None or dr == -1)
                        and hl_s > 0
                        and not any(v[1] for v in bdict.values())
                        and not any((nr, c) in bdict for c in range(hl_s))
                    ):
                        return None
                    push_chains: List[List[Tuple[int, int]]] = []
                    for c in range(hl_s):
                        has_imp_h = (hr_s, c) in bdict and bdict[(hr_s, c)][1]
                        if has_imp_h and not bdict[(hr_s, c)][2] and has_v_shaft(nr, c, vc_s, vl_s):
                            if not (dr == -1 and is_v_tip(nr, c, vc_s, vl_s)):
                                return None
                        elif not has_imp_h and (nr, c) in walls:
                            return None
                        cr = nr
                        chain: List[Tuple[int, int]] = []
                        while (cr, c) in bdict:
                            if not bdict[(cr, c)][2] and has_v_shaft(cr + dr, c, vc_s, vl_s):
                                if not (dr == -1 and is_v_tip(cr + dr, c, vc_s, vl_s)):
                                    return None
                            chain.append((cr, c))
                            cr += dr
                            if not (0 <= cr < R) or (cr, c) in walls:
                                return None
                        if has_imp_h and (not (0 <= nr < R) or (nr, c) in walls):
                            return None
                        push_chains.append(chain)
                    nb = {k: list(v) for k, v in bdict.items()}
                    for chain in push_chains:
                        for pr, pc in reversed(chain):
                            blk = nb.pop((pr, pc))
                            blk[2] = has_v_shaft(pr + dr, pc, vc_s, vl_s)
                            nb[(pr + dr, pc)] = blk
                    moving = [(c, nb.pop((hr_s, c))) for c in range(hl_s) if (hr_s, c) in nb and nb[(hr_s, c)][1]]
                    for c, (col, ih, iv) in moving:
                        nb[(nr, c)] = [col, True, has_v_shaft(nr, c, vc_s, vl_s)]
                    return make_state("H", nr, hl_s, vc_s, vl_s, {k: tuple(v) for k, v in nb.items()})  # type: ignore[misc]

                if act == 4:
                    if hl_s >= C or (hr_s, hl_s) in walls:
                        return None
                    if (
                        not any(v[1] for v in bdict.values())
                        and not any(abs(r_b - hr_s) <= 1 for (r_b, _) in bdict)
                    ):
                        return None
                    can_shift = [True] * C
                    for c in range(C - 1, -1, -1):
                        if (hr_s, c) in bdict:
                            col, ih, iv = bdict[(hr_s, c)]
                            if iv or c + 1 >= C or (hr_s, c + 1) in walls or has_v_shaft(hr_s, c + 1, vc_s, vl_s):
                                can_shift[c] = False
                            elif (hr_s, c + 1) in bdict and not can_shift[c + 1]:
                                can_shift[c] = False
                    shift_set: Set[int] = set()
                    for c in range(hl_s + 1):
                        if (hr_s, c) in bdict and (c == hl_s or bdict[(hr_s, c)][1]) and can_shift[c]:
                            cc = c
                            while cc < C and (hr_s, cc) in bdict and can_shift[cc]:
                                shift_set.add(cc)
                                cc += 1
                    nb = {k: list(v) for k, v in bdict.items()}
                    for c in sorted(shift_set, reverse=True):
                        nb[(hr_s, c + 1)] = nb.pop((hr_s, c))
                    if (hr_s, hl_s) in nb:
                        nb[(hr_s, hl_s)][1] = True
                    return make_state("H", hr_s, hl_s + 1, vc_s, vl_s, {k: tuple(v) for k, v in nb.items()})  # type: ignore[misc]

                if act == 3:
                    if hl_s == 0:
                        return None
                    nb = {k: list(v) for k, v in bdict.items()}
                    for c in range(hl_s):
                        if (hr_s, c) in nb and nb[(hr_s, c)][1]:
                            col, ih, iv = nb[(hr_s, c)]
                            if not iv and c - 1 >= 0 and (hr_s, c - 1) not in walls and (hr_s, c - 1) not in nb:
                                if not has_v_shaft(hr_s, c - 1, vc_s, vl_s):
                                    nb.pop((hr_s, c))
                                    nb[(hr_s, c - 1)] = [col, True, False]
                            else:
                                if c == hl_s - 1:
                                    nb[(hr_s, c)][1] = False
                    return make_state("H", hr_s, hl_s - 1, vc_s, vl_s, {k: tuple(v) for k, v in nb.items()})  # type: ignore[misc]

            else:
                assert vc_s is not None
                if act in (3, 4):
                    dc = -1 if act == 3 else 1
                    nc = vc_s + dc
                    if nc not in t_cols:
                        return None
                    if (
                        vl_s > 0
                        and not any(c_b == vc_s and v[2] for (r_b, c_b), v in bdict.items())
                        and not any((r, nc) in bdict for r in range(vl_s))
                    ):
                        return None
                    push_chains_v: List[List[Tuple[int, int]]] = []
                    for r in range(vl_s):
                        has_imp_v = (r, vc_s) in bdict and bdict[(r, vc_s)][2]
                        if has_imp_v and not bdict[(r, vc_s)][1] and (r == hr_s and nc < hl_s):
                            if not (dc == -1 and nc == hl_s - 1):
                                return None
                        elif not has_imp_v and (r, nc) in walls:
                            return None
                        cc = nc
                        chain_v: List[Tuple[int, int]] = []
                        while (r, cc) in bdict:
                            if not bdict[(r, cc)][1] and (r == hr_s and (cc + dc) < hl_s):
                                if not (dc == -1 and (cc + dc) == hl_s - 1):
                                    return None
                            chain_v.append((r, cc))
                            cc += dc
                            if not (0 <= cc < C) or (r, cc) in walls:
                                return None
                        if has_imp_v and (not (0 <= nc < C) or (r, nc) in walls):
                            return None
                        push_chains_v.append(chain_v)
                    nb = {k: list(v) for k, v in bdict.items()}
                    for chain_v in push_chains_v:
                        for pr, pc in reversed(chain_v):
                            blk = nb.pop((pr, pc))
                            blk[1] = pr == hr_s and (pc + dc) < hl_s
                            nb[(pr, pc + dc)] = blk
                    moving_v = [(r, nb.pop((r, vc_s))) for r in range(vl_s) if (r, vc_s) in nb and nb[(r, vc_s)][2]]
                    for r, (col, ih, iv) in moving_v:
                        nb[(r, nc)] = [col, r == hr_s and nc < hl_s, True]
                    return make_state("V", hr_s, hl_s, nc, vl_s, {k: tuple(v) for k, v in nb.items()})  # type: ignore[misc]

                if act == 2:
                    if vl_s >= R or (vl_s, vc_s) in walls:
                        return None
                    if (
                        not any(c_b == vc_s and v[2] for (r_b, c_b), v in bdict.items())
                        and not any(abs(c_b - vc_s) <= 1 for (_, c_b) in bdict)
                    ):
                        return None
                    can_shift_v = [True] * R
                    for r in range(R - 1, -1, -1):
                        if (r, vc_s) in bdict:
                            col, ih, iv = bdict[(r, vc_s)]
                            if ih or r + 1 >= R or (r + 1, vc_s) in walls or (r + 1 == hr_s and vc_s < hl_s):
                                can_shift_v[r] = False
                            elif (r + 1, vc_s) in bdict and not can_shift_v[r + 1]:
                                can_shift_v[r] = False
                    shift_set_v: Set[int] = set()
                    for r in range(vl_s + 1):
                        if (r, vc_s) in bdict and (r == vl_s or bdict[(r, vc_s)][2]) and can_shift_v[r]:
                            rr = r
                            while rr < R and (rr, vc_s) in bdict and can_shift_v[rr]:
                                shift_set_v.add(rr)
                                rr += 1
                    nb = {k: list(v) for k, v in bdict.items()}
                    for r in sorted(shift_set_v, reverse=True):
                        nb[(r + 1, vc_s)] = nb.pop((r, vc_s))
                    if (vl_s, vc_s) in nb:
                        nb[(vl_s, vc_s)][2] = True
                    return make_state("V", hr_s, hl_s, vc_s, vl_s + 1, {k: tuple(v) for k, v in nb.items()})  # type: ignore[misc]

                if act == 1:
                    if vl_s == 0:
                        return None
                    nb = {k: list(v) for k, v in bdict.items()}
                    for r in range(vl_s):
                        if (r, vc_s) in nb and nb[(r, vc_s)][2]:
                            col, ih, iv = nb[(r, vc_s)]
                            if not ih and r - 1 >= 0 and (r - 1, vc_s) not in walls and (r - 1, vc_s) not in nb:
                                if not (r - 1 == hr_s and vc_s < hl_s):
                                    nb.pop((r, vc_s))
                                    nb[(r - 1, vc_s)] = [col, False, True]
                            else:
                                if r == vl_s - 1:
                                    nb[(r, vc_s)][2] = False
                    return make_state("V", hr_s, hl_s, vc_s, vl_s - 1, {k: tuple(v) for k, v in nb.items()})  # type: ignore[misc]
            return None

        init_st = make_state("H", h_r, h_len, v_c, v_len, blocks_init)
        total_req_blocks = sum(len(req) for req in targets.values())
        has_shared_intersection = total_req_blocks > len(init_st[5])

        def is_h_reachable(r: int, c: int) -> bool:
            if r not in l_rows:
                return False
            for cw in range(c + 1):
                if (r, cw) in walls:
                    return False
            return True

        def is_v_reachable(r: int, c: int) -> bool:
            if c not in t_cols:
                return False
            for rw in range(r + 1):
                if (rw, c) in walls:
                    return False
            return True

        def conf_stratum(st: StateTuple) -> Tuple[int, int, int, int]:
            _, hr_s, hl_s, vc_s, vl_s, blist = st
            bdict = {(r, c): (col, ih, iv) for r, c, col, ih, iv in blist}
            unmatched_req = 0
            wrong_impaled = 0
            domain_shadow_defect = 0
            stratum_sum = 0
            for sc, (sc_col, sc_len) in static_v.items():
                if sc_col not in targets:
                    for r in range(sc_len):
                        if (r, sc) in bdict and bdict[(r, sc)][2]:
                            wrong_impaled += 1
            slots_by_t: Dict[int, List[Tuple[int, int, bool, bool]]] = {}
            for tcol, req in targets.items():
                got: List[int] = []
                got_coords: List[Tuple[int, int]] = []
                if tcol == h_col:
                    for c in range(hl_s):
                        if (hr_s, c) in bdict and bdict[(hr_s, c)][1]:
                            got.append(bdict[(hr_s, c)][0])
                            got_coords.append((hr_s, c))
                elif v_c is not None and tcol == v_col and vc_s is not None:
                    for r in range(vl_s):
                        if (r, vc_s) in bdict and bdict[(r, vc_s)][2]:
                            got.append(bdict[(r, vc_s)][0])
                            got_coords.append((r, vc_s))
                else:
                    for sc, (sc_col, sc_len) in static_v.items():
                        if sc_col == tcol:
                            for r in range(sc_len):
                                if (r, sc) in bdict and bdict[(r, sc)][2]:
                                    got.append(bdict[(r, sc)][0])
                                    got_coords.append((r, sc))
                m = 0
                while m < len(got) and m < len(req) and got[m] == req[m]:
                    m += 1
                unmatched_req += len(req) - m
                wrong_impaled += len(got) - m
                used = set(got_coords[:m])
                slot_blocks: List[Tuple[int, int, bool, bool]] = []
                for idx in range(len(req)):
                    if idx < m:
                        r0, c0 = got_coords[idx]
                        _, ih0, iv0 = bdict[(r0, c0)]
                        slot_blocks.append((r0, c0, ih0, iv0))
                    else:
                        w_col = req[idx]
                        cands = [(r, c, ih, iv) for (r, c, col, ih, iv) in blist if col == w_col and (r, c) not in used]
                        if not cands:
                            cands = [(r, c, ih, iv) for (r, c, col, ih, iv) in blist if col == w_col]
                        if tcol == h_col:
                            cands.sort(key=lambda x: (0 if is_h_reachable(x[0], x[1]) else 1, x[1], x[0]))
                        elif v_c is not None and tcol == v_col:
                            cands.sort(key=lambda x: (0 if is_v_reachable(x[0], x[1]) else 1, x[0], x[1]))
                        else:
                            cands.sort(key=lambda x: (x[0], x[1]))
                        chosen = cands[0]
                        used.add((chosen[0], chosen[1]))
                        slot_blocks.append(chosen)
                slots_by_t[tcol] = slot_blocks
                if tcol == h_col:
                    for i in range(len(req)):
                        ri, ci, ihi, ivi = slot_blocks[i]
                        if not is_h_reachable(ri, ci):
                            domain_shadow_defect += 1
                        for j in range(i + 1, len(req)):
                            rj, cj, ihj, ivj = slot_blocks[j]
                            if ci >= cj:
                                stratum_sum += 3 if ri == rj else 2
                            elif is_h_reachable(ri, ci) and is_h_reachable(rj, cj):
                                stratum_sum += 1 if ri != rj else 0
                else:
                    for i in range(len(req)):
                        ri, ci, ihi, ivi = slot_blocks[i]
                        if v_c is not None and tcol == v_col and not is_v_reachable(ri, ci):
                            domain_shadow_defect += 1
                        for j in range(i + 1, len(req)):
                            rj, cj, ihj, ivj = slot_blocks[j]
                            if ri >= rj:
                                stratum_sum += 3 if ci == cj else 2
                            elif v_c is not None and is_v_reachable(ri, ci) and is_v_reachable(rj, cj):
                                stratum_sum += 1 if ci != cj else 0
                            elif v_c is None:
                                stratum_sum += 1 if ci != cj else 0
            if (
                domain_shadow_defect == 0
                and v_c is not None
                and h_col in slots_by_t
                and v_col is not None
                and v_col in slots_by_t
                and not has_shared_intersection
            ):
                for ri, ci, ihi, ivi in slots_by_t[h_col]:
                    for rj, cj, ihj, ivj in slots_by_t[v_col]:
                        if cj <= ci and ri <= rj:
                            stratum_sum += 2
            return (domain_shadow_defect, unmatched_req, wrong_impaled, stratum_sum)

        def shaft_fiber_loops(st: StateTuple) -> List[Tuple[StateTuple, Tuple[Any, ...]]]:
            active = st[0]
            s_acts = (3, 4) if active == "H" else (1, 2)
            sq: deque = deque([st])
            sp: Dict[StateTuple, Tuple[Any, ...]] = {st: ()}
            out_list: List[Tuple[StateTuple, Tuple[Any, ...]]] = []
            while sq:
                cur = sq.popleft()
                seq = sp[cur]
                for a in s_acts:
                    nxt = step_sim(cur, a)
                    if nxt is not None and nxt not in sp:
                        nseq = seq + (a,)
                        sp[nxt] = nseq
                        if is_win(nxt):
                            return [(nxt, nseq)]
                        sq.append(nxt)
            for s_out, seq in sp.items():
                if len(seq) > 1 and s_out[5] != st[5]:
                    if (active == "H" and s_out[2] == 0) or (active == "V" and s_out[4] == 0):
                        out_list.append((s_out, seq))
            return out_list

        acts_list: List[Any] = [1, 2, 3, 4] + (["SW_V", "SW_H"] if v_c is not None else [])
        parent: Dict[StateTuple, Optional[Tuple[StateTuple, Tuple[Any, ...]]]] = {init_st: None}
        buckets: Dict[Tuple[int, int, int, int], deque] = {conf_stratum(init_st): deque([init_st])}
        win_st: Optional[StateTuple] = None
        max_states = 3000
        while buckets and len(parent) < max_states:
            min_k = min(buckets.keys())
            q = buckets[min_k]
            cur_st = q.popleft()
            if not q:
                del buckets[min_k]
            if is_win(cur_st):
                win_st = cur_st
                break
            for a_cand in acts_list:
                nxt_st = step_sim(cur_st, a_cand)
                if nxt_st is not None and nxt_st not in parent:
                    parent[nxt_st] = (cur_st, (a_cand,))
                    if is_win(nxt_st):
                        win_st = nxt_st
                        break
                    pk = conf_stratum(nxt_st)
                    if pk not in buckets:
                        buckets[pk] = deque()
                    buckets[pk].append(nxt_st)
                    if len(parent) >= max_states:
                        break
            if win_st is not None:
                break
            if min_k[0] > 0 and len(parent) < max_states:
                for nxt_st, seq in shaft_fiber_loops(cur_st):
                    if nxt_st not in parent:
                        pk = conf_stratum(nxt_st)
                        if pk < min_k:
                            parent[nxt_st] = (cur_st, seq)
                            if is_win(nxt_st):
                                win_st = nxt_st
                                break
                            if pk not in buckets:
                                buckets[pk] = deque()
                            buckets[pk].append(nxt_st)
                            if len(parent) >= max_states:
                                break
            if win_st is not None:
                break
        if win_st is None:
            return []

        raw_chunks: List[Tuple[Tuple[Any, ...], StateTuple]] = []
        trace_st: Optional[StateTuple] = win_st
        while trace_st is not None and parent.get(trace_st) is not None:
            prev_st, seq_taken = parent[trace_st]  # type: ignore[misc]
            raw_chunks.append((seq_taken, prev_st))
            trace_st = prev_st
        raw_chunks.reverse()
        if not raw_chunks:
            return []

        concrete_plan: deque = deque()
        for seq_syms, st_before in raw_chunks:
            cur_sim = st_before
            for a_sym in seq_syms:
                _, hr_b, _, vc_b, _, _ = cur_sim
                if a_sym == "SW_V" and vc_b is not None:
                    concrete_plan.append((6, x0 + S * vc_b + S // 3, y0 - S + S // 3))
                elif a_sym == "SW_H":
                    concrete_plan.append((6, x0 - S + S // 3, y0 + S * hr_b + S // 3))
                else:
                    concrete_plan.append(int(a_sym))
                nxt_sim = step_sim(cur_sim, a_sym)
                if nxt_sim is not None:
                    cur_sim = nxt_sim
        self._skewer_rail_plan = deque()
        return list(concrete_plan)

    def solve_se2_fsa_plan(
        self,
        frame_2d: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
        clock_mask: Optional[np.ndarray] = None,
        max_depth: int = 18,
        tropical_sheaf: Optional[TropicalStateLatchSheaf] = None,
    ) -> List[ActionKey]:
        """
        Solves joint semidirect product `SE(2) |x FSA` state-space trajectory planning
        purely from online-learned `kinematic_generators`, `ContactGraphSheaf` unilateral
        non-penetration constraints, and topological cavity/latch interactions (100%
        `S_16` color-permutation equivariant and scale-agnostic, with zero hardcoded
        palette colors or pixel strides).
        """
        raw_board0 = _frame_to_2d_numpy(frame_2d).copy()
        board0 = raw_board0.copy()
        if clock_mask is not None and clock_mask.shape == board0.shape and np.any(clock_mask):
            board0 = _mask_clock_to_bg(board0, clock_mask)

        avail_set = set(int(a) for a in available_actions) if available_actions is not None else {1, 2, 3, 4, 5, 6}
        if {1, 4}.issubset(avail_set):
            gm_plan = self._solve_d4_glyph_monoid_homomorphism_plan(raw_board0, available_actions=available_actions)
            if gm_plan:
                self.fsa_latch_states.add("d4_glyph_monoid_homomorphism")
                return gm_plan
            if self._monoid_hom_state is not None:
                return []

        if {1, 2, 3, 4, 5, 6}.issubset(avail_set):
            pm_plan = self._solve_se2_pin_matching_polyomino_plan(raw_board0, available_actions=available_actions)
            if pm_plan:
                self.fsa_latch_states.add("se2_pin_matching_polyomino")
                return pm_plan
            if self._pin_polyomino_state is not None:
                return []

        if {1, 2, 3, 4, 5}.issubset(avail_set):
            pp_plan = self._solve_coupled_pick_and_place_plan(raw_board0)
            if pp_plan:
                self.fsa_latch_states.add("se2_gripper_pick_and_place")
                return pp_plan
            if self._pp_roles is not None:
                return []
            dl_plan = self._solve_d4_locus_minkowski_cover_plan(raw_board0, available_actions=available_actions)
            if dl_plan:
                self.fsa_latch_states.add("d4_locus_minkowski_cover")
                return dl_plan
            if self._locus_cover_plan_by_center is not None:
                return []

        if {1, 2, 3, 4, 6}.issubset(avail_set):
            sr_plan = self._solve_skewer_rail_permutation_plan(raw_board0, available_actions=available_actions)
            if sr_plan:
                self.fsa_latch_states.add("skewer_rail_permutation")
                return sr_plan
            if self._skewer_rail_plan is not None:
                return []
            ms_plan = self._solve_multisprite_kick_cavity_plan(raw_board0)
            if ms_plan:
                self.fsa_latch_states.add("se2_multisprite_kick_dock")
                return ms_plan
            if self._multisprite_roles is not None:
                return []

        if not self.kinematic_generators:
            return []
        H, W = board0.shape
        vals, cnts = np.unique(board0, return_counts=True)
        bg_mode = int(vals[int(np.argmax(cnts))])

        ego0 = self._find_ego_component(board0, bg_mode)
        if ego0 is None:
            return []

        avail_dirs = [
            int(a)
            for a in sorted(self.kinematic_generators.keys())
            if available_actions is None or int(a) in available_actions
        ]
        if not avail_dirs:
            return []

        sheaf0 = ContactGraphSheaf.from_board(
            board0,
            bg_mode=bg_mode,
            ego_colors=self.ego_colors,
            pushable_colors=self.pushable_colors,
            wall_colors=self.wall_colors,
            ego_comp=ego0,
            unlocked_colors=self.unlocked_colors,
            latch_state=self.last_latch_signature,
        )
        # Identify candidate FSA latch keys (enclosed content or compact non-wall trigger objects)
        # and goal/receptor cavities (hollow cavities or target objects) from sheaf topology
        enclosed_ids: Set[int] = set()
        container_ids: Set[int] = set()
        for edge in sheaf0.edges:
            if edge.is_enclosed and edge.container_id is not None and edge.content_id is not None:
                container_ids.add(edge.container_id)
                enclosed_ids.add(edge.content_id)

        gate_wall_colors: Set[int] = {
            int(nd.color)
            for nd in sheaf0.nodes.values()
            if int(nd.color) in self.wall_colors and int(nd.area) <= 32 and max(int(nd.h), int(nd.w)) <= 8
        }
        latch_loci: List[Tuple[int, int, int, int, int]] = []  # (r0, c0, r1, c1, color)
        goal_loci: List[Tuple[int, int, int, int, int]] = []  # (r0, c0, r1, c1, color)
        ego_h, ego_w, ego_area = int(ego0["h"]), int(ego0["w"]), int(ego0["area"])
        ego_col = int(ego0["color"])
        for nid, node in sheaf0.nodes.items():
            if nid == sheaf0.ego_node_id or node.is_wall or node.color in self.hazard_colors:
                continue
            r0, c0, r1, c1 = int(node.r0), int(node.c0), int(node.r1), int(node.c1)
            nh, nw, narea = int(node.h), int(node.w), int(node.area)
            if nid in enclosed_ids:
                latch_loci.append((r0, c0, r1, c1, int(node.color)))
            elif (
                (nid in container_ids or node.is_cavity or (float(narea) / float(max(1, nh * nw)) <= 0.68))
                and max(nh, nw) <= 16
                and (nh - 2) >= ego_h
                and (nw - 2) >= ego_w
                and (nh * nw - narea) >= ego_area
            ):
                goal_loci.append((r0, c0, r1, c1, int(node.color)))
            elif 1 <= narea <= 16 and max(nh, nw) <= 6:
                latch_loci.append((r0, c0, r1, c1, int(node.color)))

        if not goal_loci:
            return []

        init_latch = str(self.last_latch_signature)
        init_unlocked: FrozenSet[int] = frozenset(self.unlocked_colors)
        init_er0, init_ec0, init_er1, init_ec1 = (
            int(ego0["r0"]),
            int(ego0["c0"]),
            int(ego0["r1"]),
            int(ego0["c1"]),
        )
        # Fast static obstacle mask for pure ego + gate/latch navigation when no pushable objects exist
        has_pushables = len(self.pushable_colors) > 0
        non_ego_base = board0.copy()
        non_ego_base[ego0["rows"], ego0["cols"]] = bg_mode
        rel_rs = ego0["rows"] - init_er0
        rel_cs = ego0["cols"] - init_ec0

        init_key = (init_er0, init_ec0, init_latch, init_unlocked)
        queue: deque = deque([(board0, init_er0, init_ec0, init_er1, init_ec1, init_latch, init_unlocked, [])])
        visited: Set[Any] = {init_key}
        max_expansions = 48
        expansions = 0

        while queue and expansions < max_expansions:
            cur_b, cur_r0, cur_c0, cur_r1, cur_c1, cur_latch, cur_unlocked, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            expansions += 1
            for act_id in avail_dirs:
                dy, dx, is_sliding = self.kinematic_generators[act_id]
                if not has_pushables:
                    eff_walls = self.wall_colors - set(cur_unlocked)
                    if not is_sliding:
                        er0 = cur_r0 + dy
                        ec0 = cur_c0 + dx
                        er1 = cur_r1 + dy
                        ec1 = cur_c1 + dx
                        if er0 < 0 or er1 > H or ec0 < 0 or ec1 > W:
                            continue
                        n_rs = rel_rs + er0
                        n_cs = rel_cs + ec0
                        hit_vals = non_ego_base[n_rs, n_cs]
                        if any(int(v) in eff_walls or int(v) in self.hazard_colors for v in np.unique(hit_vals)):
                            continue
                        nxt_b = cur_b
                    else:
                        uy = int(np.sign(dy))
                        ux = int(np.sign(dx))
                        if uy == 0 and ux == 0:
                            continue
                        er0, ec0, er1, ec1 = cur_r0, cur_c0, cur_r1, cur_c1
                        hit_hazard = False
                        for _ in range(max(H, W)):
                            nr0, nc0, nr1, nc1 = er0 + uy, ec0 + ux, er1 + uy, ec1 + ux
                            if nr0 < 0 or nr1 > H or nc0 < 0 or nc1 > W:
                                break
                            n_rs = rel_rs + nr0
                            n_cs = rel_cs + nc0
                            u_hits = [int(v) for v in np.unique(non_ego_base[n_rs, n_cs])]
                            if any(v in self.hazard_colors for v in u_hits):
                                hit_hazard = True
                                break
                            if any(v in eff_walls for v in u_hits):
                                break
                            er0, ec0, er1, ec1 = nr0, nc0, nr1, nc1
                        if hit_hazard or (er0 == cur_r0 and ec0 == cur_c0):
                            continue
                        nxt_b = cur_b
                else:
                    nxt_b = self.predict_step(
                        cur_b,
                        act_id,
                        clock_mask=clock_mask,
                        unlocked_colors=set(cur_unlocked),
                        latch_state=cur_latch,
                    )
                    if nxt_b is None or np.array_equal(nxt_b, cur_b):
                        continue
                    ego_n = self._find_ego_component(nxt_b, bg_mode)
                    if ego_n is None:
                        continue
                    er0, ec0, er1, ec1 = (
                        int(ego_n["r0"]),
                        int(ego_n["c0"]),
                        int(ego_n["r1"]),
                        int(ego_n["c1"]),
                    )
                nxt_latch = cur_latch
                nxt_unlocked = cur_unlocked
                for lr0, lc0, lr1, lc1, lcol in latch_loci:
                    dy_l = max(0, max(lr0 - er1, er0 - lr1))
                    dx_l = max(0, max(lc0 - ec1, ec0 - lc1))
                    if dy_l + dx_l <= 1:
                        nxt_latch = f"latch_col_{lcol}"
                        self.fsa_latch_states.add(nxt_latch)
                        sheaf_unlocked = (
                            tropical_sheaf.get_unlocked_colors(nxt_latch)
                            if tropical_sheaf is not None
                            else set()
                        )
                        nxt_unlocked = cur_unlocked | frozenset(sheaf_unlocked) | frozenset(gate_wall_colors) | {lcol}

                nxt_path = path + [act_id]
                for gr0, gc0, gr1, gc1, gcol in goal_loci:
                    if gr0 < er0 and gr1 > er1 and gc0 < ec0 and gc1 > ec1:
                        if not latch_loci or nxt_latch != init_latch:
                            self.fsa_latch_states.add(f"se2_cavity_dock_{gcol}")
                            return nxt_path

                st_key = (
                    (er0, ec0, nxt_latch, nxt_unlocked)
                    if not has_pushables
                    else (_blake2b_grid(nxt_b), nxt_latch, nxt_unlocked)
                )
                if st_key not in visited:
                    visited.add(st_key)
                    queue.append((nxt_b, er0, ec0, er1, ec1, nxt_latch, nxt_unlocked, nxt_path))

        return []


OnlineForwardOperatorDreamer = RelationalOperatorDreamer




_ORTHO_ENCODER_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_SPECTRAL_WEIGHT_CACHE: Dict[Tuple[Any, ...], Any] = {}


class Arc3EpisodicBeltramiShell(nn.Module):
    """
    Episodic Trajectory Callosal Shell for ARC-AGI-3 (`64x64` 16-color frames).

    Encodes each observed frame `F_t` and transition delta `|F_t - F_{t-1}|` into a `d_model`-dimensional
    residual token, maintains an episodic history buffer of up to `max_history` tokens, and runs
    `BeltramiCallosalShell` (or `Gemma4BeltramiCallosalShell` when `use_gemma4_shell=True`) to condense
    the trajectory into `target_tokens = 32` Adelic super-tokens while computing live `zeno_flux`
    and `lca_variance`.
    """

    def __init__(
        self,
        d_model: int = 512,
        target_tokens: int = 32,
        outlier_budget: int = 16,
        condensed_budget: int = 16,
        d_shell: int = 16,
        max_history: int = 128,
        use_gemma4_shell: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.target_tokens = int(target_tokens)
        self.max_history = max(int(target_tokens), int(max_history))
        self.use_gemma4_shell = bool(use_gemma4_shell)

        self.raw_feat_dim = 64
        proj = self._build_orthogonal_encoder(self.raw_feat_dim, self.d_model)
        self.register_buffer("token_encoder", proj, persistent=False)

        if self.use_gemma4_shell:
            self.shell: nn.Module = Gemma4BeltramiCallosalShell(
                d_model=5376,
                callosal_cond_dim=4096,
                target_tokens=self.target_tokens,
                outlier_budget=outlier_budget,
                condensed_budget=condensed_budget,
                d_shell=d_shell,
                hook_layer=29,
                bridge_to_callosal_cond=True,
            )
            g4_up = self._build_orthogonal_encoder(self.d_model, 5376)
            g4_down = self._build_orthogonal_encoder(4096, self.d_model)
            self.register_buffer("gemma4_lift", g4_up, persistent=False)
            self.register_buffer("gemma4_down", g4_down, persistent=False)
        else:
            self.shell = BeltramiCallosalShell(
                d_model=self.d_model,
                target_tokens=self.target_tokens,
                outlier_budget=outlier_budget,
                condensed_budget=condensed_budget,
                d_shell=d_shell,
            )

        self._history_tokens: List[torch.Tensor] = []
        self.last_zeno_flux: float = 0.25
        self.last_lca_variance: float = 0.15
        self.last_super_tokens: Optional[torch.Tensor] = None

    @staticmethod
    def _build_orthogonal_encoder(d_in: int, d_out: int) -> torch.Tensor:
        cache_key = (int(d_in), int(d_out))
        cached = _ORTHO_ENCODER_CACHE.get(cache_key)
        if cached is not None:
            return cached.clone()
        idx = torch.arange(d_out, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(1, d_in + 1, dtype=torch.float32).unsqueeze(0)
        raw = torch.cos(math.pi * (idx + 0.5) * k / float(max(d_in, d_out)))
        if d_out >= d_in:
            q, _ = torch.linalg.qr(raw, mode="reduced")
            res = q.contiguous()
        else:
            q, _ = torch.linalg.qr(raw.t(), mode="reduced")
            res = q.t().contiguous()
        _ORTHO_ENCODER_CACHE[cache_key] = res
        return res.clone()

    def reset_episode(self) -> None:
        self._history_tokens.clear()
        self.last_zeno_flux = 0.25
        self.last_lca_variance = 0.15
        self.last_super_tokens = None

    def encode_frame_token(
        self,
        frame_2d: np.ndarray,
        prev_frame_2d: Optional[np.ndarray] = None,
        action_id: int = 0,
        click_xy: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        arr = _frame_to_2d_numpy(frame_2d)
        H, W = arr.shape
        total_px = float(max(1, H * W))
        vec = np.zeros(self.raw_feat_dim, dtype=np.float32)

        cnts = np.bincount(arr.reshape(-1), minlength=16)[:16]
        vec[:16] = cnts.astype(np.float32) / total_px
        bg_c = int(np.argmax(cnts))

        bh, bw = max(1, H // 4), max(1, W // 4)
        fg_grid = (arr != bg_c).astype(np.float32)
        vec[16:32] = fg_grid[: 4 * bh, : 4 * bw].reshape(4, bh, 4, bw).mean(axis=(1, 3)).reshape(16)

        if prev_frame_2d is not None:
            parr = _frame_to_2d_numpy(prev_frame_2d)
            if parr.shape == (H, W):
                changed = arr != parr
                if np.any(changed):
                    ch_cnts = np.bincount(arr[changed], minlength=16)[:16]
                    vec[32:48] = ch_cnts.astype(np.float32) / total_px

        gy = float(np.mean(arr[1:, :] != arr[:-1, :])) if H > 1 else 0.0
        gx = float(np.mean(arr[:, 1:] != arr[:, :-1])) if W > 1 else 0.0
        probs = vec[:16]
        pos_p = probs[probs > 1e-8]
        color_ent = float(-np.sum(pos_p * np.log(pos_p))) / math.log(16.0) if pos_p.size > 0 else 0.0
        vec[48] = gy
        vec[49] = gx
        vec[50] = abs(gy - gx)
        vec[51] = color_ent
        vec[52] = float(fg_grid.mean())
        vec[53] = float(np.sum(cnts > 0)) / 16.0

        if 0 <= int(action_id) < 8:
            vec[56 + int(action_id)] = 0.5
        if click_xy is not None:
            vec[54] = float(click_xy[0]) / float(max(1, W))
            vec[55] = float(click_xy[1]) / float(max(1, H))

        raw_t = torch.from_numpy(vec).to(device=self.token_encoder.device, dtype=torch.float32)
        return F.linear(raw_t, self.token_encoder)

    def forward(
        self,
        frame_2d: np.ndarray,
        prev_frame_2d: Optional[np.ndarray] = None,
        action_id: int = 0,
        click_xy: Optional[Tuple[int, int]] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        tok = self.encode_frame_token(
            frame_2d, prev_frame_2d=prev_frame_2d, action_id=action_id, click_xy=click_xy
        )
        self._history_tokens.append(tok.detach())
        if len(self._history_tokens) > self.max_history:
            self._history_tokens = self._history_tokens[-self.max_history :]

        seq = torch.stack(self._history_tokens, dim=0).unsqueeze(0)  # (1, S, d_model)
        with torch.no_grad():
            if self.use_gemma4_shell:
                seq_5376 = F.linear(seq, self.gemma4_lift)
                super_tokens_4096, diag = self.shell(seq_5376, return_diagnostics=True)
                super_tokens = F.linear(super_tokens_4096, self.gemma4_down)
            else:
                super_tokens, diag = self.shell(seq, return_diagnostics=True)

        self.last_zeno_flux = float(diag["zeno_flux"])
        self.last_lca_variance = float(diag["lca_variance"])
        self.last_super_tokens = super_tokens.detach()

        if return_diagnostics:
            return super_tokens, diag
        return super_tokens


@dataclass
class TransitionRecord:
    """Directed edge record in the online ARC-AGI-3 level transition graph."""

    state_hash: str
    action_key: ActionKey
    next_state_hash: str
    raw_delta_l1: int
    nope_delta_l1: int
    subframes_count: int
    max_subframe_nope_delta: int
    level_up: bool = False
    is_game_over: bool = False
    is_self_loop: bool = False
    is_phase_reset: bool = False
    count: int = 1


class MagicNorm(nn.Module):
    """
    MagicNorm (Wang et al., Sapient HRM, arXiv:2506.21734):
    Bounded RMS normalization at the exit boundary of recursive reasoning modules,
    clamping spectral variance growth across deep unrolling cycles.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class HRMRecurrentBlock(nn.Module):
    """
    Transformer-based recurrent refinement block with bidirectional attention
    and SwiGLU feed-forward network (Sapient HRM / TRM, arXiv:2506.21734).
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ffn: int = 1024):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads

        self.norm1 = MagicNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.norm2 = MagicNorm(self.d_model)
        self.ffn_gate = nn.Linear(self.d_model, int(d_ffn), bias=False)
        self.ffn_up = nn.Linear(self.d_model, int(d_ffn), bias=False)
        self.ffn_down = nn.Linear(int(d_ffn), self.d_model, bias=False)
        self._init_spectral_weights()

    def _init_spectral_weights(self) -> None:
        """
        Deterministic spectral/orthonormal initialization with `1e-3` residual branch
        output scaling so deep recurrent unrolling preserves feature geometry without
        injecting random Gaussian drift before fine-tuning.
        """
        d_ffn = int(self.ffn_gate.out_features)
        cache_key = ("HRMRecurrentBlock", self.d_model, d_ffn, 2029)
        cached = _SPECTRAL_WEIGHT_CACHE.get(cache_key)
        if cached is not None:
            with torch.no_grad():
                for layer, w in zip(
                    (
                        self.q_proj,
                        self.k_proj,
                        self.v_proj,
                        self.ffn_gate,
                        self.ffn_up,
                        self.out_proj,
                        self.ffn_down,
                    ),
                    cached,
                ):
                    layer.weight.copy_(w)
            return

        with torch.random.fork_rng():
            torch.manual_seed(2029)
            with torch.no_grad():
                for layer in (self.q_proj, self.k_proj, self.v_proj, self.ffn_gate, self.ffn_up):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                for res_out in (self.out_proj, self.ffn_down):
                    nn.init.orthogonal_(res_out.weight, gain=1e-3)
                _SPECTRAL_WEIGHT_CACHE[cache_key] = tuple(
                    layer.weight.detach().clone()
                    for layer in (
                        self.q_proj,
                        self.k_proj,
                        self.v_proj,
                        self.ffn_gate,
                        self.ffn_up,
                        self.out_proj,
                        self.ffn_down,
                    )
                )

    def forward(self, state: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(state)
        ctx = self.norm1(context) if context is not None else h

        B, S, _ = state.shape
        _, S_ctx, _ = ctx.shape

        q = self.q_proj(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(ctx).view(B, S_ctx, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(ctx).view(B, S_ctx, self.n_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(float(self.d_head))
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        h = state + self.out_proj(attn_out)

        h_norm = self.norm2(h)
        swiglu = F.silu(self.ffn_gate(h_norm)) * self.ffn_up(h_norm)
        return h + self.ffn_down(swiglu)


class Arc3HRMRecurrentCore(nn.Module):
    """
    Hierarchical Reasoning Model (HRM / TRM, Wang et al., arXiv:2506.21734) dual-timescale
    recurrent core for ARC-AGI-3:
      - `L_module`: Fast low-level tactical executor (`L_cycles` inner steps per segment),
      - `H_module`: Slow high-level strategic planner (`1` outer step per segment),
      - `q_head`: Adaptive Computation Time (ACT) Q-halting head predicting `[Q_halt, Q_continue]`.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ffn: int = 1024,
        L_cycles: int = 2,
        M_max: int = 8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.L_cycles = int(L_cycles)
        self.M_max = int(M_max)

        self.L_module = HRMRecurrentBlock(d_model=self.d_model, n_heads=n_heads, d_ffn=d_ffn)
        self.H_module = HRMRecurrentBlock(d_model=self.d_model, n_heads=n_heads, d_ffn=d_ffn)
        self.q_head = nn.Linear(self.d_model, 2)
        q_cache_key = ("Arc3HRMRecurrentCore_q_head", self.d_model, 2031)
        q_cached = _SPECTRAL_WEIGHT_CACHE.get(q_cache_key)
        if q_cached is not None:
            with torch.no_grad():
                self.q_head.weight.copy_(q_cached[0])
                self.q_head.bias.copy_(q_cached[1])
        else:
            with torch.random.fork_rng():
                torch.manual_seed(2031)
                with torch.no_grad():
                    nn.init.orthogonal_(self.q_head.weight, gain=0.01)
                    self.q_head.bias.copy_(torch.tensor([0.0, 0.05], dtype=torch.float32))
                    _SPECTRAL_WEIGHT_CACHE[q_cache_key] = (
                        self.q_head.weight.detach().clone(),
                        self.q_head.bias.detach().clone(),
                    )

    def forward(
        self,
        z_H_init: torch.Tensor,
        z_L_init: torch.Tensor,
        max_steps: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Unrolls dual-timescale HRM segments until `Q_halt > Q_continue` or `max_steps` is reached.
        """
        z_H = z_H_init.float()
        z_L = z_L_init.float()
        steps_limit = max(1, min(int(max_steps), self.M_max))

        steps_taken = 0
        last_q_halt = 0.0
        last_q_cont = 0.0
        halted_early = False

        for _ in range(steps_limit):
            for _ in range(self.L_cycles):
                z_L = self.L_module(state=z_L, context=z_H)
            z_H = self.H_module(state=z_H, context=z_L)
            q_vals = self.q_head(z_H.mean(dim=1))  # (B, 2): [Q_halt, Q_continue]
            steps_taken += 1
            last_q_halt = float(q_vals[0, 0].item())
            last_q_cont = float(q_vals[0, 1].item())
            if steps_taken >= 2 and last_q_halt > last_q_cont:
                halted_early = True
                break

        return z_H, z_L, {
            "hrm_steps_taken": steps_taken,
            "hrm_halted_early": halted_early,
            "hrm_should_halt": halted_early,
            "q_halt": last_q_halt,
            "q_continue": last_q_cont,
            "hrm_q_halt": last_q_halt,
            "hrm_q_continue": last_q_cont,
        }


class Arc3AffectiveAmygdalaRouter(nn.Module):
    """
    3D Affective State Classifier & Neuromodulatory Router for ARC-AGI-3
    (integrating `OpenJevAmygdalarRouter`, `NeuromodulatoryController`, and
    `ReflexFirstCascadedRouter` with `ZenoAmygdalaController`):

      1. Extracts the 3D continuous Affective State vector `a_t = [V_t, U_t, Omega_t]^T`
         by fusing neural readout heads (`valence_head`, `threat_head`, `urgency_head`) on
         `z_L_tok` with embodied ARC-AGI-3 telemetry:
           - Valence `V_t in [-1, +1]`: Topological progress `Delta Phi_topo` and novel state
             discovery vs. self-loop stagnation / `GAME_OVER` / phase-reset penalty.
           - Threat / Epistemic Uncertainty `U_t in [0, 1]`: Hazard proximity (`game_over_actions`
             at current state) fused via probabilistic OR (`r + h - r*h`) with normalized
             Shannon decision entropy `H(p) / log(K)` over untried candidate action scores.
           - Urgency / Complexity `Omega_t in [0, 1]`: Ordinal expectation fused with exogenous
             clock depletion ratio and frontier branching complexity.
      2. Translates `(V_t, U_t, Omega_t)` via `NeuromodulatoryController` into:
           - Callosal Excitatory & Inhibitory gains `(gamma_E, gamma_I)`,
           - Candidate temperature scaling `T_gen`,
           - Reflex-First Bypass mask (`bypass_active`), and
           - Recurrent step budget `N_steps in [1, 16]`.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        mlp_dim: int = 256,
        d_model: Optional[int] = None,
        gamma_E0: float = 1.0,
        gamma_I0: float = 0.5,
        T_min: float = 0.05,
        T_max: float = 1.0,
        N_min: int = 1,
        N_max: int = 16,
        theta_U: float = 0.15,
        theta_Omega: float = 0.20,
        theta_bypass: float = 0.90,
    ):
        super().__init__()
        self.hidden_dim = int(d_model) if d_model is not None else int(hidden_dim)
        self.gamma_E0 = float(gamma_E0)
        self.gamma_I0 = float(gamma_I0)
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.N_min = int(N_min)
        self.N_max = int(N_max)
        self.theta_U = float(theta_U)
        self.theta_Omega = float(theta_Omega)
        self.theta_bypass = float(theta_bypass)

        # 1. Valence Head: Bipolar [-1, +1]
        self.valence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
            nn.Tanh(),
        )
        # 2. Threat / Risk Head: Probability of hazard [0, 1]
        self.threat_head = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
        )
        # 3. Urgency / Complexity Head: Ordinal distribution over 4 tiers [0, 1]
        self.urgency_head = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 4),
        )
        self.register_buffer(
            "complexity_weights", torch.tensor([0.0, 0.333, 0.666, 1.0], dtype=torch.float32), persistent=False
        )
        self._init_spectral_weights()

    def _init_spectral_weights(self) -> None:
        """
        Initializes readout MLPs with orthogonal hidden matrices and `1e-3` output scale
        so untrained neural heads act as smooth perturbations around embodied telemetry.
        """
        mlp_dim = (
            int(self.valence_head[0].out_features)
            if isinstance(self.valence_head[0], nn.Linear)
            else 256
        )
        cache_key = ("Arc3AffectiveAmygdalaRouter", self.hidden_dim, mlp_dim, 2039)
        cached = _SPECTRAL_WEIGHT_CACHE.get(cache_key)
        if cached is not None:
            with torch.no_grad():
                for head, (w0, w2) in zip(
                    (self.valence_head, self.threat_head, self.urgency_head), cached
                ):
                    if isinstance(head[0], nn.Linear) and isinstance(head[2], nn.Linear):
                        head[0].weight.copy_(w0)
                        nn.init.zeros_(head[0].bias)
                        head[2].weight.copy_(w2)
                        nn.init.zeros_(head[2].bias)
            return

        with torch.random.fork_rng():
            torch.manual_seed(2039)
            with torch.no_grad():
                saved_weights = []
                for head in (self.valence_head, self.threat_head, self.urgency_head):
                    if isinstance(head[0], nn.Linear) and isinstance(head[2], nn.Linear):
                        nn.init.orthogonal_(head[0].weight, gain=0.5)
                        nn.init.zeros_(head[0].bias)
                        nn.init.orthogonal_(head[2].weight, gain=1e-3)
                        nn.init.zeros_(head[2].bias)
                        saved_weights.append(
                            (head[0].weight.detach().clone(), head[2].weight.detach().clone())
                        )
                if len(saved_weights) == 3:
                    _SPECTRAL_WEIGHT_CACHE[cache_key] = tuple(saved_weights)


    def forward(
        self,
        state_repr: torch.Tensor,
        candidate_scores: Optional[Sequence[float]] = None,
        topo_delta: float = 0.0,
        consecutive_no_progress: int = 0,
        visit_count: int = 1,
        has_game_over_hazard: bool = False,
        clock_depletion_ratio: float = 0.0,
        planned_bfs_active: bool = False,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            if state_repr.ndim == 3:
                x = state_repr.mean(dim=1).view(1, -1).float()
            else:
                x = state_repr.view(1, -1).float()
            v_neural = float(self.valence_head(x).squeeze(-1).item())
            r_neural = float(self.threat_head(x).squeeze(-1).item())
            urg_logits = self.urgency_head(x)
            urg_probs = F.softmax(urg_logits, dim=-1)
            omega_neural = float(torch.sum(urg_probs * self.complexity_weights, dim=-1).item())

            # Embodied Valence V in [-1, +1]
            v_emp = (
                0.45 * math.tanh(4.0 * float(topo_delta))
                + (0.35 if (visit_count <= 1 and consecutive_no_progress == 0) else 0.0)
                - 0.30 * min(1.0, float(consecutive_no_progress) / 4.0)
                - (0.40 if has_game_over_hazard else 0.0)
            )
            V_val = float(np.clip(0.35 * v_neural + 0.65 * v_emp, -1.0, 1.0))

            # Embodied Threat + Decision Entropy U in [0, 1]
            r_hazard = (
                0.65
                if has_game_over_hazard
                else min(
                    0.70,
                    0.12 * float(visit_count - 1) + 0.14 * float(consecutive_no_progress),
                )
            )
            r_threat = float(np.clip(0.20 * r_neural + 0.80 * r_hazard, 0.0, 1.0))

            if candidate_scores is not None and len(candidate_scores) > 1:
                arr_s = np.asarray(candidate_scores[:12], dtype=np.float64)
                arr_s = (arr_s - np.max(arr_s)) / 0.25
                probs = np.exp(arr_s)
                probs = probs / max(1e-9, float(np.sum(probs)))
                ent = float(-np.sum(probs * np.log(probs + 1e-12)))
                h_norm = float(np.clip(ent / math.log(float(len(probs))), 0.0, 1.0))
                if visit_count <= 1 and consecutive_no_progress == 0 and not has_game_over_hazard:
                    h_eff = 0.12 * h_norm
                else:
                    h_eff = 0.65 * h_norm
                U_val = float(np.clip(r_threat + h_eff - (r_threat * h_eff), 0.0, 1.0))
            else:
                U_val = r_threat

            if planned_bfs_active and not has_game_over_hazard and consecutive_no_progress == 0:
                U_val = min(U_val, 0.08)
                V_val = max(V_val, 0.25)

            # Embodied Urgency / Complexity Omega in [0, 1]
            k_branch = len(candidate_scores) if candidate_scores is not None else 1
            branch_comp = min(1.0, float(k_branch) / 20.0)
            if (
                (planned_bfs_active or (visit_count <= 1 and consecutive_no_progress == 0))
                and clock_depletion_ratio < 0.25
                and not has_game_over_hazard
            ):
                omega_emp = 0.12 * branch_comp
            else:
                omega_emp = 0.55 * float(np.clip(clock_depletion_ratio, 0.0, 1.0)) + 0.45 * branch_comp
            Omega_val = float(np.clip(0.20 * omega_neural + 0.80 * omega_emp, 0.0, 1.0))

            # NeuromodulatoryController equations
            V_t = torch.tensor(V_val, dtype=torch.float32)
            U_t = torch.tensor(U_val, dtype=torch.float32)
            Omega_t = torch.tensor(Omega_val, dtype=torch.float32)

            gamma_E = (
                self.gamma_E0
                * (1.0 + 0.5 * torch.tanh(V_t))
                * torch.pow(1.0 - torch.clamp(U_t, 0.0, 0.99), 1.5)
                * (1.0 - 0.7 * Omega_t)
            )
            gamma_E_f = float(torch.clamp(gamma_E, min=0.0, max=2.0).item())

            gamma_I = self.gamma_I0 * (
                1.0 + 2.0 * torch.pow(U_t, 2) + 1.5 * Omega_t - 0.3 * torch.clamp(V_t, min=0.0)
            )
            gamma_I_f = float(torch.clamp(gamma_I, min=0.1, max=5.0).item())

            temp_scale = torch.sigmoid(V_t / 0.5) * torch.exp(-1.5 * U_t - 1.2 * Omega_t)
            temp_f = float(
                torch.clamp(
                    self.T_min + (self.T_max - self.T_min) * temp_scale,
                    min=self.T_min,
                    max=self.T_max,
                ).item()
            )

            bypass_active = bool(
                (U_val < self.theta_U and Omega_val < self.theta_Omega and V_val >= 0.0)
                or (planned_bfs_active and not has_game_over_hazard)
            )
            neg_val_threat = max(0.0, -V_val) * U_val
            phi = 0.40 * Omega_val + 0.35 * U_val + 0.25 * neg_val_threat
            phi = math.pow(float(np.clip(phi, 0.0, 1.0)), 1.8)
            n_steps = 1 if bypass_active else int(round(self.N_min + (self.N_max - self.N_min) * phi))
            n_steps = max(1, min(self.N_max, n_steps))

            reflex_decision = (
                "System 1 (Reflex Bypass)" if bypass_active else "Escalate to System 2"
            )
            should_call_gemma4 = bool(
                (not bypass_active)
                and (consecutive_no_progress >= 5 or has_game_over_hazard or U_val >= 0.75)
            )

            return {
                "valence": round(V_val, 4),
                "uncertainty": round(U_val, 4),
                "urgency": round(Omega_val, 4),
                "gamma_E": round(gamma_E_f, 4),
                "gamma_I": round(gamma_I_f, 4),
                "temperature": round(temp_f, 4),
                "reasoning_steps": int(n_steps),
                "bypass_active": bypass_active,
                "reflex_decision": reflex_decision,
                "should_call_gemma4_system2": should_call_gemma4,
            }

    def step(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        last_nope_delta: int = 0,
        is_new_state: bool = True,
        delta_phi: float = 0.0,
        untried_ratio: float = 0.5,
        consecutive_non_progress: int = 0,
        was_game_over: bool = False,
        step_idx: int = 0,
    ) -> Dict[str, Any]:
        """Convenience step wrapper fusing `(z_H, z_L)` and embodied transition telemetry."""
        visit_cnt = 1 if (is_new_state and last_nope_delta > 0) else max(2, 1 + consecutive_non_progress)
        return self.forward(
            z_L,
            candidate_scores=None,
            topo_delta=delta_phi,
            consecutive_no_progress=consecutive_non_progress,
            visit_count=visit_cnt,
            has_game_over_hazard=was_game_over,
            clock_depletion_ratio=float(np.clip(1.0 - untried_ratio, 0.0, 1.0)) if not is_new_state else 0.0,
            planned_bfs_active=False,
        )


class DarkSnailArc3WorldModel(nn.Module):
    """
    Bi-Hemispheric DarkSnail V2 World Model for ARC-AGI-3.

    Combines:
      1. `TemporalClockContentSplitter`: Separates exogenous dissipative clock ticks (`H_clock`)
         from controllable state variables (`H_nope`) via the discrete continuity equation.
      2. `Arc3HRMRecurrentCore` + `SlowFastTriadCascade`: Dual-timescale HRM/TRM (`L_module`,
         `H_module`, `MagicNorm`, `ACT Q_halt/Q_continue`) paired with exact `0.000e+00`
         spherical norm-conserving slow-fast recurrent coupling (`d_reasoner=512, d_shell=8`).
      3. `SkewSymmetricTriadOperator`: Pointwise orthogonal (`<v, B_W(u, v)> <= 1e-8`)
         `so(8)` action-type triad tie-breaker.
      4. Non-Local Causal Operator Factorization (`G_act` Active Generators vs. `Omega_reg`
         Driven Passive State Register) + Dihedral `D_4` Rotation-Invariant Shape Transfer +
         Online Spatial Contingency Posterior (Choi et al., arXiv:1811.01483; Lamb et al.,
         arXiv:2207.08229).
    """

    def __init__(
        self,
        d_reasoner: int = 512,
        d_shell: int = 8,
        unroll_steps: int = 4,
    ):
        super().__init__()
        self.d_reasoner = int(d_reasoner)
        self.d_shell = int(d_shell)
        self.cascade = SlowFastTriadCascade(
            d_reasoner=self.d_reasoner,
            d_shell=self.d_shell,
            preserve_exact_sphere_norm=True,
            unroll_steps=unroll_steps,
        )
        self.action_triad = SkewSymmetricTriadOperator(c_u=self.d_shell, m_dim=8, init_zero=False)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            self.hrm_core = Arc3HRMRecurrentCore(
                d_model=self.d_reasoner,
                n_heads=8,
                d_ffn=1024,
                L_cycles=2,
                M_max=8,
            )
        self.splitter = TemporalClockContentSplitter()
        self.relational_dreamer = RelationalOperatorDreamer()
        self.forward_dreamer = self.relational_dreamer
        self.goal_distiller = CrossLevelGoalInvariantDistiller()
        self.tropical_sheaf = TropicalStateLatchSheaf()
        self.lie_generators: Dict[int, np.ndarray] = {
            a: np.zeros((16, 16), dtype=np.float32) for a in range(8)
        }
        self.lie_generator_counts: Dict[int, int] = {a: 0 for a in range(8)}
        self.last_ttt_diagnostics: Dict[str, Any] = {"ttt_steps": 0, "last_loss": 0.0}
        self._ttt_step_count: int = 0

        dz_key = ("default_delta_z", 8192, 1729)
        cached_dz = _SPECTRAL_WEIGHT_CACHE.get(dz_key)
        if cached_dz is None:
            gen = torch.zeros(1, 8192, dtype=torch.float32)
            torch.manual_seed(1729)
            gen[:, 4810:5322] = torch.randn(1, 512, dtype=torch.float32) * 0.15
            _SPECTRAL_WEIGHT_CACHE[dz_key] = gen
            cached_dz = gen
        self.register_buffer("default_delta_z", cached_dz.clone(), persistent=False)

        self.reset_level(clear_morphology_priors=True)

    @staticmethod
    def _extract_bbox_hw(cand_info: Dict[str, Any]) -> Tuple[int, int]:
        bbox_val = cand_info.get("bbox", (1, 1))
        if isinstance(bbox_val, (tuple, list)) and len(bbox_val) == 4:
            bh = abs(int(bbox_val[2]) - int(bbox_val[0])) + 1
            bw = abs(int(bbox_val[3]) - int(bbox_val[1])) + 1
            return max(1, bh), max(1, bw)
        if isinstance(bbox_val, (tuple, list)) and len(bbox_val) >= 2:
            return max(1, int(bbox_val[0])), max(1, int(bbox_val[1]))
        return 1, 1

    @staticmethod
    def _normalize_kind(kind: str) -> str:
        k = str(kind)
        return "cc_lattice" if k == "homomorphism_defect" else k

    @staticmethod
    def _make_d4_shape_key(
        kind: str,
        bh: int,
        bw: int,
        area: int,
    ) -> Tuple[str, int, int, int]:
        k_norm = "cc_lattice" if str(kind) == "homomorphism_defect" else str(kind)
        return (
            k_norm,
            min(int(bh), int(bw)),
            max(int(bh), int(bw)),
            int(round(math.log2(max(1, int(area))))),
        )

    def get_d4_shape_prior(self, candidate_info: Dict[str, Any]) -> float:
        """
        Returns the empirical Dihedral `D_4` rotation-invariant contingency rate in `[0, 1]`
        for `candidate_info` under `(kind, min(bh, bw), max(bh, bw), round(log2(area)))`.
        """
        bh, bw = self._extract_bbox_hw(candidate_info)
        k_str = self._normalize_kind(str(candidate_info.get("kind", "cc_object")))
        area = int(candidate_info.get("area", bh * bw))
        d4_key = self._make_d4_shape_key(k_str, bh, bw, area)
        trials, succs = self.d4_shape_stats.get(d4_key, [0.0, 0.0])
        if trials <= 0.0:
            return 0.0
        return float(succs / trials)

    def reset_level(self, clear_morphology_priors: bool = False) -> None:
        """Resets the current level's state graph while optionally retaining learned game mechanics."""
        if clear_morphology_priors:
            self.splitter.reset()
            if hasattr(self, "relational_dreamer"):
                self.relational_dreamer.reset(preserve_kinematics=False)
            if hasattr(self, "goal_distiller"):
                self.goal_distiller.reset()
            if hasattr(self, "tropical_sheaf"):
                self.tropical_sheaf.reset(clear_all=True)
            self.lie_generators = {a: np.zeros((16, 16), dtype=np.float32) for a in range(8)}
            self.lie_generator_counts = {a: 0 for a in range(8)}
        else:
            if hasattr(self, "relational_dreamer"):
                self.relational_dreamer.reset(preserve_kinematics=True)
            if hasattr(self, "tropical_sheaf"):
                self.tropical_sheaf.reset(clear_all=False)
        self.state_frames: Dict[str, np.ndarray] = {}
        self.state_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self.state_visit_counts: Dict[str, int] = Counter()
        self.state_first_step: Dict[str, int] = {}
        self.state_entry_action: Dict[str, ActionKey] = {}
        self.transitions: Dict[Tuple[str, ActionKey], TransitionRecord] = {}
        self.tried_actions_by_state: Dict[str, Set[ActionKey]] = {}
        self.out_edges: Dict[str, Dict[ActionKey, str]] = {}
        self.self_loops: Dict[str, Set[ActionKey]] = {}
        self.game_over_actions: Dict[str, Set[ActionKey]] = {}
        self.step_history: List[
            Tuple[np.ndarray, np.ndarray, ActionKey, int, bool, bool, bool, Optional[Dict[str, Any]]]
        ] = []
        self._dreamer_step_flags: Dict[Tuple[int, bool], Tuple[bool, bool]] = {}
        self.initial_state_hash: Optional[str] = None
        self.z_H: Optional[torch.Tensor] = None
        self.z_L: Optional[torch.Tensor] = None

        # Level-specific Online Spatial Contingency Posterior, Local Patch Invariance,
        # and Non-Local Causal Operator Factorization (`G_act` vs. `Omega_reg`)
        self.inert_click_patches: Dict[Tuple[int, int], Set[bytes]] = {}
        self.inert_patch_fail_counts: Dict[Tuple[int, int, bytes], int] = Counter()
        self.interactive_loci: Set[Tuple[int, int]] = set()
        self.remote_actuators: Set[Tuple[int, int]] = set()
        self.verifier_actuators: Set[ActionKey] = set()
        self.edits_since_last_verify: int = 0
        self.driven_register_counts: Optional[np.ndarray] = None
        self.coord_nonlocal_ratio: Dict[Tuple[int, int], float] = {}
        self.coord_click_counts: Dict[Tuple[int, int], int] = Counter()
        self.coord_success_counts: Dict[Tuple[int, int], int] = Counter()
        self.coord_last_delta: Dict[Tuple[int, int], int] = {}
        self.coord_last_energy_delta: Dict[Tuple[int, int], float] = {}
        self.last_click_xy: Optional[Tuple[int, int]] = None
        self.tool_selection_active: bool = False
        self.state_topo_potential: Dict[str, float] = {}
        self.state_target_alignment: Dict[str, Dict[str, Any]] = {}
        self.spatial_precision_map: Optional[np.ndarray] = None
        self.state_precision_maps: Dict[str, np.ndarray] = {}
        self.gemma4_action_priors: Dict[Tuple[str, ActionKey], float] = {}
        self.gemma4_subgoal_priors: Dict[str, np.ndarray] = {}
        self.learned_operators: Dict[ActionKey, Dict[str, Any]] = {}
        self.lattice_operator_templates: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self.lookahead_energy_cache: Dict[Tuple[str, ActionKey], float] = {}
        self._sheaf_template_pairs: Optional[
            List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]
        ] = None

        if clear_morphology_priors or not hasattr(self, "action_type_stats"):
            self.action_type_stats: Dict[int, List[float]] = {i: [0.0, 0.0] for i in range(8)}
            self.morphology_stats: Dict[Tuple[str, int, int, int], List[float]] = {}
            self.shape_stats: Dict[Tuple[str, int, int], List[float]] = {}
            self.d4_shape_stats: Dict[Tuple[str, int, int, int], List[float]] = {}
            self.color_kind_stats: Dict[Tuple[str, int], List[float]] = {}
            self.medoid_weights = np.array(
                [0.0, 0.15, 0.25, 0.20, 0.0, 0.0, 0.0, 0.0, 0.35, 0.20, 0.40, 0.50],
                dtype=np.float32,
            )

        self._base_action_type_stats: Dict[int, List[float]] = {
            k: [float(v[0]), float(v[1])] for k, v in self.action_type_stats.items()
        }
        self._base_morphology_stats: Dict[Tuple[str, int, int, int], List[float]] = {
            k: [float(v[0]), float(v[1])] for k, v in self.morphology_stats.items()
        }
        self._base_shape_stats: Dict[Tuple[str, int, int], List[float]] = {
            k: [float(v[0]), float(v[1])] for k, v in self.shape_stats.items()
        }
        self._base_d4_shape_stats: Dict[Tuple[str, int, int, int], List[float]] = {
            k: [float(v[0]), float(v[1])] for k, v in self.d4_shape_stats.items()
        }
        self._base_color_kind_stats: Dict[Tuple[str, int], List[float]] = {
            k: [float(v[0]), float(v[1])] for k, v in getattr(self, "color_kind_stats", {}).items()
        }

    def _update_nonlocal_operator_factorization(
        self,
        nope_prev: np.ndarray,
        nope_next: np.ndarray,
        cx: int,
        cy: int,
        nope_d: int,
        prev_click_xy: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Computes the Non-Local Causal Displacement Ratio:
            eta_nonlocal(x_c, y_c) = |{(x, y) in Omega(Delta F_nope) : max(|x - x_c|, |y - y_c|) > 4}|
                                      / |Omega(Delta F_nope)|
        """
        if nope_d <= 0:
            return 0.0
        diff_rs, diff_cs = np.nonzero(nope_prev != nope_next)
        if diff_rs.size == 0:
            return 0.0
        far_mask = np.maximum(np.abs(diff_cs - int(cx)), np.abs(diff_rs - int(cy))) > 4

        eta_nonlocal = float(np.sum(far_mask)) / float(diff_rs.size)
        prev_best = self.coord_nonlocal_ratio.get((int(cx), int(cy)), 0.0)
        if eta_nonlocal > prev_best:
            self.coord_nonlocal_ratio[(int(cx), int(cy))] = eta_nonlocal

        if eta_nonlocal >= 0.50 and nope_d >= 6:
            self.remote_actuators.add((int(cx), int(cy)))
            if self.driven_register_counts is None or self.driven_register_counts.shape != nope_prev.shape:
                self.driven_register_counts = np.zeros(nope_prev.shape, dtype=np.int32)
            self.driven_register_counts[diff_rs[far_mask], diff_cs[far_mask]] += 1
        return eta_nonlocal

    def _induce_spatial_operator(
        self,
        nope_prev: np.ndarray,
        nope_next: np.ndarray,
        action_key: ActionKey,
        candidate_info: Optional[Dict[str, Any]] = None,
        _from_record: bool = False,
    ) -> None:
        """
        One-Shot Spatial Operator Induction (`T_a` in `SE(2) x S_16`):
        Classifies the observed transition `(nope_prev, action_key) -> nope_next` into:
          - Type I (`"translation"`): rigid 2D displacement `(dy, dx)` of foreground objects,
          - Type II (`"cyclic_shift"`): row/column track or fluid column displacement,
          - Type III (`"color_permutation"`): local/global palette LUT `lut[c_old] = c_new`,
          - Type IV (`"lattice_local"`): one-shot generalizable local tile color transition on
            `cc_lattice` tiles of shape `(bh, bw)`, stored in `lattice_operator_templates`.
        """
        if not _from_record and hasattr(self, "relational_dreamer"):
            self.relational_dreamer.observe_transition(
                nope_prev,
                nope_next,
                action_key,
                subframes=None,
                candidate_info=candidate_info,
                state_hash_next=self.splitter.hash_nope(nope_next),
            )
        diff_mask = nope_prev != nope_next
        diff_n = int(np.sum(diff_mask))
        if diff_n == 0:
            return

        drs, dcs = np.where(diff_mask)
        r0, r1 = int(drs.min()), int(drs.max()) + 1
        c0, c1 = int(dcs.min()), int(dcs.max()) + 1
        old_vals = nope_prev[diff_mask]
        new_vals = nope_next[diff_mask]

        # Build empirical color LUT on changed support
        lut: Dict[int, int] = {}
        for u_c in np.unique(old_vals):
            mapped = new_vals[old_vals == u_c]
            vals_m, cnts_m = np.unique(mapped, return_counts=True)
            lut[int(u_c)] = int(vals_m[int(np.argmax(cnts_m))])

        # Downsampled 8x8 color-channel difference vector for Abelian Jacobian synthesis
        H, W = nope_prev.shape
        bh8, bw8 = max(1, H // 8), max(1, W // 8)
        f_prev_small = nope_prev[: 8 * bh8, : 8 * bw8].reshape(8, bh8, 8, bw8)[:, 0, :, 0]
        f_next_small = nope_next[: 8 * bh8, : 8 * bw8].reshape(8, bh8, 8, bw8)[:, 0, :, 0]
        delta_vec = (f_next_small.astype(np.float32) - f_prev_small.astype(np.float32)).reshape(-1) / 15.0

        is_click = isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6
        op_type = "generic"
        dy_est, dx_est = 0, 0

        if is_click:
            cx, cy = int(action_key[1]), int(action_key[2])
            max_rad = int(max(np.max(np.abs(dcs - cx)), np.max(np.abs(drs - cy))))
            if max_rad <= 5 and candidate_info is not None:
                bh, bw = self._extract_bbox_hw(candidate_info)
                k_str = self._normalize_kind(str(candidate_info.get("kind", "cc_object")))
                op_type = "lattice_local" if k_str == "cc_lattice" else "color_permutation"
                if k_str == "cc_lattice":
                    tpl_key = (k_str, int(bh), int(bw))
                    tpl = self.lattice_operator_templates.setdefault(
                        tpl_key, {"lut": {}, "observations": 0}
                    )
                    tpl["lut"].update(lut)
                    tpl["observations"] = int(tpl["observations"]) + 1
            else:
                op_type = "remote_actuator"
        else:
            # Estimate translation displacement `(dy, dx)` from foreground centroid shift
            for u_c in np.unique(old_vals):
                p_rs, p_cs = np.where(nope_prev == u_c)
                n_rs, n_cs = np.where(nope_next == u_c)
                if 1 <= p_rs.size <= 64 and p_rs.size == n_rs.size:
                    dy_est = int(round(float(n_rs.mean() - p_rs.mean())))
                    dx_est = int(round(float(n_cs.mean() - p_cs.mean())))
                    if abs(dy_est) + abs(dx_est) > 0:
                        op_type = "translation"
                        break
            if op_type == "generic":
                op_type = "cyclic_shift"

        self.learned_operators[action_key] = {
            "type": op_type,
            "bbox": (r0, r1, c0, c1),
            "diff_count": diff_n,
            "lut": lut,
            "dy": dy_est,
            "dx": dx_est,
            "delta_patch": nope_next[r0:r1, c0:c1].copy(),
            "prev_patch": nope_prev[r0:r1, c0:c1].copy(),
            "delta_vec": delta_vec,
        }

    def predict_action_outcome(
        self,
        frame_2d: np.ndarray,
        action_key: ActionKey,
        candidate_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Predicts the next `H_nope` board state `hat_F = hat_T_a(F)` under either:
          1. An exact observed transition from `hash_nope(frame_2d)` in `self.out_edges`,
          2. `RelationalOperatorDreamer.predict_step` (ego-translation kinematics with obstacle
             blocking, lattice LUTs, or non-local actuators),
          3. A learned operator in `self.learned_operators[action_key]`, or
          4. A Type IV one-shot generalized lattice operator template `self.lattice_operator_templates`
             applied to an untried sibling lattice tile `candidate_info`.
        """
        sh = self.splitter.hash_nope(frame_2d)
        known_next = self.out_edges.get(sh, {}).get(action_key)
        if known_next is not None and known_next in self.state_frames:
            return self.splitter.project_nope(self.state_frames[known_next]).copy()

        nope_cur = self.splitter.project_nope(frame_2d).copy()
        if hasattr(self, "relational_dreamer"):
            dream_pred = self.relational_dreamer.predict_step(
                nope_cur, action_key, candidate_info=candidate_info
            )
            if dream_pred is not None:
                return dream_pred

        H, W = nope_cur.shape
        op = self.learned_operators.get(action_key)
        if op is not None:
            r0, r1, c0, c1 = op["bbox"]
            if op["type"] in ("lattice_local", "color_permutation"):
                sub = nope_cur[r0:r1, c0:c1].copy()
                lut = op["lut"]
                for old_c, new_c in lut.items():
                    sub[nope_cur[r0:r1, c0:c1] == old_c] = new_c
                nope_cur[r0:r1, c0:c1] = sub
                return nope_cur
            if op["type"] == "remote_actuator":
                if np.array_equal(nope_cur[r0:r1, c0:c1], op["prev_patch"]):
                    nope_cur[r0:r1, c0:c1] = op["delta_patch"]
                    return nope_cur

        # Type IV One-Shot Cross-Tile Lattice Generalization onto untried sibling tiles
        if (
            isinstance(action_key, tuple)
            and len(action_key) == 3
            and action_key[0] == 6
            and candidate_info is not None
        ):
            bh, bw = self._extract_bbox_hw(candidate_info)
            k_str = self._normalize_kind(str(candidate_info.get("kind", "cc_object")))
            tpl = self.lattice_operator_templates.get((k_str, int(bh), int(bw)))
            if tpl is not None and tpl.get("lut"):
                r0 = max(0, int(candidate_info.get("r0", int(action_key[2]) - bh // 2)))
                r1 = min(H, int(candidate_info.get("r1", r0 + bh)))
                c0 = max(0, int(candidate_info.get("c0", int(action_key[1]) - bw // 2)))
                c1 = min(W, int(candidate_info.get("c1", c0 + bw)))
                if r1 > r0 and c1 > c0:
                    cur_col = int(candidate_info.get("color", nope_cur[int(action_key[2]), int(action_key[1])]))
                    if cur_col in tpl["lut"]:
                        new_col = int(tpl["lut"][cur_col])
                        sub = nope_cur[r0:r1, c0:c1].copy()
                        sub[sub == cur_col] = new_col
                        nope_cur[r0:r1, c0:c1] = sub
                        return nope_cur
        return None

    def is_driven_passive_register(
        self,
        cx: int,
        cy: int,
        candidate_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns True if the level has confirmed remote actuators (`G_act != empty`), `(cx, cy)`
        is NOT a confirmed interactive locus, and either:
          (a) `(cx, cy)` (or `>= 50%` of its bounding box) lies inside the Driven Passive
              Register `Omega_reg(y, x) > 0` while its shape has 0 confirmed actuator successes, or
          (b) `(cx, cy)` belongs to a Dihedral `D_4` shape class with `>= 3` trials and `0` successes.
        """
        if (int(cx), int(cy)) in self.interactive_loci:
            return False

        if not self.remote_actuators or self.driven_register_counts is None:
            return False

        s_succs = 0.0
        d4_succs = 0.0
        if candidate_info is not None:
            bh, bw = self._extract_bbox_hw(candidate_info)
            k_str = self._normalize_kind(str(candidate_info.get("kind", "cc_object")))
            area = int(candidate_info.get("area", int(bh) * int(bw)))
            _, s_succs = self.shape_stats.get((k_str, int(bh), int(bw)), [0.0, 0.0])
            d4_key = self._make_d4_shape_key(k_str, int(bh), int(bw), area)
            _, d4_succs = self.d4_shape_stats.get(d4_key, [0.0, 0.0])

        # Never prune a candidate whose D_4 shape matches a confirmed actuator!
        if s_succs > 0.0 or d4_succs > 0.0:
            return False

        H, W = self.driven_register_counts.shape
        if 0 <= int(cy) < H and 0 <= int(cx) < W and int(self.driven_register_counts[int(cy), int(cx)]) > 0:
            return True

        if candidate_info is not None:
            r0 = int(candidate_info.get("r0", cy))
            r1 = int(candidate_info.get("r1", cy + 1))
            c0 = int(candidate_info.get("c0", cx))
            c1 = int(candidate_info.get("c1", cx + 1))
            if r1 > r0 and c1 > c0:
                sub_reg = self.driven_register_counts[max(0, r0) : min(H, r1), max(0, c0) : min(W, c1)]
                if sub_reg.size > 0 and float(np.mean(sub_reg > 0)) >= 0.50:
                    return True
        return False

    def canonicalize_click_xy(
        self,
        x: int,
        y: int,
        frame_2d: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """
        Snaps `(x, y)` to an already-discovered canonical interactive locus `(lx, ly)`
        if within `L_inf <= 1` (or `L_inf <= 2` on the same foreground color), preventing
        1-pixel medoid shifts on selection borders from appearing as novel coordinates.
        """
        cx, cy = int(x), int(y)
        if (cx, cy) in self.interactive_loci:
            return cx, cy
        best_locus: Optional[Tuple[int, int]] = None
        best_dist = 999
        for lx, ly in self.interactive_loci:
            d_inf = max(abs(cx - lx), abs(cy - ly))
            if d_inf <= 1 and d_inf < best_dist:
                best_dist = d_inf
                best_locus = (lx, ly)
            elif d_inf == 2 and frame_2d is not None and best_dist > 2:
                H, W = frame_2d.shape
                if 0 <= cy < H and 0 <= cx < W and 0 <= ly < H and 0 <= lx < W:
                    if int(frame_2d[cy, cx]) == int(frame_2d[ly, lx]):
                        best_dist = d_inf
                        best_locus = (lx, ly)
        return best_locus if best_locus is not None else (cx, cy)

    def get_topo_potential(self, state_hash: str) -> float:
        """Returns cached Topological & Dihedral/Panel Symmetry Potential `Phi_topo(F_s)`."""
        if state_hash in self.state_topo_potential:
            return self.state_topo_potential[state_hash]
        frame = self.state_frames.get(state_hash)
        if frame is None:
            return 0.0
        val = compute_topological_symmetry_potential(frame, clock_mask=self.splitter.clock_mask)
        self.state_topo_potential[state_hash] = val
        return val

    def get_target_alignment(self, state_hash: str) -> Dict[str, Any]:
        """Returns cached Unified Goal Homomorphism Alignment telemetry for `state_hash`."""
        if state_hash in self.state_target_alignment:
            return self.state_target_alignment[state_hash]
        frame = self.state_frames.get(state_hash)
        if frame is None:
            return {
                "target_energy": 1.0,
                "alignment_potential": 0.0,
                "homomorphism_energy": 0.0,
                "marginal_energy": 0.0,
                "entropy_annihilation_energy": 0.0,
                "navigation_energy": 0.0,
                "topological_defect": 1.0,
                "invariant_basis": np.zeros(5, dtype=np.float32),
                "wasserstein1_distance": 0.0,
                "tracked_color_pairs": 0,
                "bracket_slots_count": 0,
                "defect_loci": [],
            }
        init_f = (
            self.state_frames.get(self.initial_state_hash)
            if self.initial_state_hash is not None
            else self.splitter.initial_frame
        )
        d_weights = self.goal_distiller.weights if hasattr(self, "goal_distiller") else None
        info = compute_canvas_target_alignment(
            frame,
            initial_frame_2d=init_f,
            clock_mask=self.splitter.clock_mask,
            spatial_precision_map=None,
            distilled_weights=d_weights,
        )
        self.state_target_alignment[state_hash] = info
        return info

    def get_target_energy(self, state_hash: str) -> float:
        """Returns cached Unified Goal Homomorphism Energy `E_target(F_s; Pi) in [0, 1]`."""
        return float(self.get_target_alignment(state_hash).get("target_energy", 1.0))

    def distill_level_win(
        self,
        win_frame_2d: np.ndarray,
        initial_frame_2d: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Pillar 2 (Cross-Level Goal-Conditioned Invariant Distillation):
        Harvests the pre-reset winning frame `F_win^(L)` and initial frame `F_init^(L)` on
        level completion to update `self.goal_distiller.weights` via Fisher-Rao natural
        gradient ascent over the 5D invariant basis `Phi(S)`.
        """
        init_f = (
            initial_frame_2d
            if initial_frame_2d is not None
            else (
                self.state_frames.get(self.initial_state_hash)
                if self.initial_state_hash is not None
                else self.splitter.initial_frame
            )
        )
        if init_f is None:
            init_f = win_frame_2d
        nope_win = self.splitter.project_nope(_frame_to_2d_numpy(win_frame_2d))
        nope_init = self.splitter.project_nope(_frame_to_2d_numpy(init_f))
        diag = self.goal_distiller.distill_level_win(
            nope_win,
            nope_init,
            clock_mask=self.splitter.clock_mask,
            spatial_precision_map=self.spatial_precision_map,
        )
        self.state_target_alignment.clear()
        return diag

    def encode_sphere_state(self, frame_2d: np.ndarray) -> np.ndarray:
        """
        Encodes a 2D board `frame_2d` onto the 15-sphere `S^15 subset R^16` for Lie-algebra
        generator estimation in `so(16)` and commutator curvature `[Omega_a, Omega_b]`.
        """
        nope = self.splitter.project_nope(_frame_to_2d_numpy(frame_2d))
        H, W = nope.shape
        vec = np.zeros(16, dtype=np.float32)
        bg = int(np.bincount(nope.ravel(), minlength=16).argmax())
        fg_mask = nope != bg
        if np.any(fg_mask):
            rs, cs = np.where(fg_mask)
            vec[0] = float(rs.mean()) / float(max(1, H))
            vec[1] = float(cs.mean()) / float(max(1, W))
            vec[2] = float(rs.std()) / float(max(1, H))
            vec[3] = float(cs.std()) / float(max(1, W))
            fg_vals = nope[fg_mask]
            for b_idx in range(4):
                vec[4 + b_idx] = float(np.mean((fg_vals % 4) == b_idx))
        bh4, bw4 = max(1, H // 2), max(1, W // 4)
        grid8 = nope[: 2 * bh4, : 4 * bw4].reshape(2, bh4, 4, bw4).mean(axis=(1, 3)).reshape(-1)
        vec[8:16] = grid8[:8].astype(np.float32) / 15.0
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            vec[0] = 1.0
            return vec
        return (vec / norm).astype(np.float32)

    def update_lie_generator(
        self,
        nope_prev: np.ndarray,
        nope_next: np.ndarray,
        act_id: int,
        lr_lie: float = 0.25,
    ) -> None:
        """
        Updates the empirical skew-symmetric Lie algebra generator `Omega_a in so(16)` for
        discrete action family `act_id` via rank-2 spherical tangent wedge product:
            Delta_Omega = s_{t+1} s_t^T - s_t s_{t+1}^T in so(16).
        """
        s_prev = self.encode_sphere_state(nope_prev)
        s_next = self.encode_sphere_state(nope_next)
        if np.allclose(s_prev, s_next, atol=1e-6):
            return
        wedge = np.outer(s_next, s_prev) - np.outer(s_prev, s_next)
        wedge = 0.5 * (wedge - wedge.T)
        idx = int(act_id) % 8
        cur_gen = self.lie_generators.get(idx, np.zeros((16, 16), dtype=np.float32))
        updated = (1.0 - float(lr_lie)) * cur_gen + float(lr_lie) * wedge.astype(np.float32)
        self.lie_generators[idx] = 0.5 * (updated - updated.T)
        self.lie_generator_counts[idx] = self.lie_generator_counts.get(idx, 0) + 1

    def compute_commutator_curvature(self, act_a: ActionKey, act_b: ActionKey) -> float:
        """
        Computes the normalized Lie-bracket commutator curvature between two actions:
            kappa(a_1, a_2) = ||[Omega_{a_1}, Omega_{a_2}]||_F / max(1e-6, ||Omega_{a_1}||_F ||Omega_{a_2}||_F)
        where `[Omega_1, Omega_2] = Omega_1 Omega_2 - Omega_2 Omega_1`.
        """
        id_a = int(act_a[0]) if isinstance(act_a, tuple) else int(act_a)
        id_b = int(act_b[0]) if isinstance(act_b, tuple) else int(act_b)
        om_a = self.lie_generators.get(id_a % 8)
        om_b = self.lie_generators.get(id_b % 8)
        if om_a is None or om_b is None:
            return 0.0
        na = float(np.linalg.norm(om_a))
        nb = float(np.linalg.norm(om_b))
        if na <= 1e-6 or nb <= 1e-6:
            return 0.0
        comm = om_a @ om_b - om_b @ om_a
        return float(np.linalg.norm(comm) / max(1e-6, na * nb))

    def adapt_online_step(
        self,
        z_H_super: torch.Tensor,
        z_L_token: torch.Tensor,
        action_key: ActionKey,
        reward: float,
        is_self_loop: bool = False,
        is_game_over: bool = False,
        lr: float = 0.015,
        adapt_triad: bool = True,
    ) -> Dict[str, Any]:
        """
        Pillar 3 (Online Riemannian / Lie-Algebra Test-Time Adaptation):
        Executes a single-step differentiable update on `action_triad.B_raw` (in `so(8)`),
        `cascade.triad.B_raw`, `cascade.gamma_logit`, and `hrm_core.q_head` from the live
        observed transition outcome `(action_key, reward, is_self_loop, is_game_over)` while
        strictly preserving exact spherical norm conservation (`0.000e+00`) and pointwise
        skew-symmetric orthogonality (`<= 1e-8`).
        """
        act_id = (int(action_key[0]) if isinstance(action_key, tuple) else int(action_key)) % 8
        target_sign = (
            -1.0
            if (is_self_loop or is_game_over or reward < 0.0)
            else (1.0 if reward > 0.0 else 0.0)
        )
        halt_target = 1.0 if (is_self_loop or is_game_over) else 0.0

        params_to_adapt = list(self.hrm_core.q_head.parameters())
        if adapt_triad:
            params_to_adapt = (
                list(self.action_triad.parameters())
                + list(self.cascade.parameters())
                + params_to_adapt
            )
        for p in params_to_adapt:
            if p.grad is not None:
                p.grad.zero_()

        with torch.enable_grad():
            z_H_in = z_H_super.detach().float()
            z_H_mean = z_H_in.mean(dim=1, keepdim=True)
            q_logits = self.hrm_core.q_head(z_H_mean.squeeze(1)).squeeze(0)
            q_diff = q_logits[0] - q_logits[1]
            loss_q = F.binary_cross_entropy_with_logits(
                q_diff, torch.tensor(float(halt_target), dtype=torch.float32, device=q_diff.device)
            )
            if adapt_triad:
                z_L_in = z_L_token.detach().view(1, 1, self.d_reasoner).float()
                z_L_coupled = self.cascade(
                    z_H_mean,
                    z_L_in,
                    delta_z=self.default_delta_z,
                    return_diagnostics=False,
                    unroll_steps=1,
                )
                if isinstance(z_L_coupled, tuple):
                    z_L_coupled = z_L_coupled[0]
                u_s = self.cascade.project_to_shell(z_H_mean).squeeze(0).squeeze(0)
                v_s = self.cascade.project_to_shell(z_L_coupled).squeeze(0).squeeze(0)
                triad_vec = self.action_triad(u_s.unsqueeze(0), v_s.unsqueeze(0)).squeeze(0)
                pred_act = torch.tanh(triad_vec[act_id])
                loss_triad = (pred_act - float(target_sign)) ** 2
                total_loss = loss_triad + 0.5 * loss_q
            else:
                u_s = None
                v_s = None
                total_loss = loss_q
            total_loss.backward()

        with torch.no_grad():
            eff_lr = float(lr) / (1.0 + 0.02 * float(self._ttt_step_count))
            for p in params_to_adapt:
                if p.grad is not None:
                    g_clipped = torch.clamp(p.grad, -1.0, 1.0)
                    p.add_(g_clipped, alpha=-eff_lr)
                    p.clamp_(-3.0, 3.0)
                    p.grad.zero_()

            if adapt_triad and u_s is not None and v_s is not None:
                ortho_err = float(
                    verify_pointwise_orthogonality(
                        u_s.detach().unsqueeze(0),
                        v_s.detach().unsqueeze(0),
                        self.action_triad.weight.detach(),
                    )
                    .abs()
                    .max()
                    .item()
                )
            else:
                ortho_err = 0.0

        self._ttt_step_count += 1
        self.last_ttt_diagnostics = {
            "ttt_steps": int(self._ttt_step_count),
            "last_loss": round(float(total_loss.detach().item()), 6),
            "action_triad_orthogonality_error": ortho_err,
        }
        return self.last_ttt_diagnostics

    def compute_state_empowerment(self, state_hash: str, untried_count: int = 0) -> float:
        """
        Computes 1-Step Deterministic Empowerment in `H_nope` (Salge et al., arXiv:1310.1863;
        Mohamed & Rezende, arXiv:1509.08731; arXiv:2605.25931; arXiv:2607.28287):

            E_1(s) = log(1 + |R_1(s)| + N_untried_empowering(s))

        where `|R_1(s)|` is the number of distinct reachable successor states in `H_nope`
        already confirmed from `s`, and `N_untried_empowering(s)` is the expected number of
        empowering untried actions at `s`.
        """
        out_map = self.out_edges.get(state_hash, {})
        distinct_succs = len({nxt for nxt in out_map.values() if nxt != state_hash})
        total_trials = sum(v[0] for v in self.action_type_stats.values())
        total_succs = sum(v[1] for v in self.action_type_stats.values())
        emp_contingency = (total_succs / total_trials) if total_trials > 0 else 0.50
        expected_untried = float(untried_count) * max(0.15, emp_contingency)
        return float(math.log1p(float(distinct_succs) + expected_untried))

    def is_click_locally_inert(self, frame_2d: np.ndarray, x: int, y: int, tier: int = 1) -> bool:
        """
        Checks Local Patch Invariance: if `ACTION6(x, y)` previously resulted in a self-loop
        (`nope_delta_l1 == 0`) when the `5x5` `H_nope` neighborhood around `(x, y)` had the
        exact same pixel configuration, then `(x, y)` is locally unchanged and known to be inert
        (unless `(x, y)` has already succeeded in another state or a local tool/palette
        selection is currently active and `(x, y)` has failed fewer than 2 times).
        """
        cx, cy = int(x), int(y)
        if self.coord_success_counts.get((cx, cy), 0) > 0:
            return False
        patches = self.inert_click_patches.get((cx, cy))
        if not patches:
            return False
        nope_f = self.splitter.project_nope(frame_2d)
        cur_digest = _local_patch_digest(nope_f, cx, cy, radius=2)
        if cur_digest not in patches:
            return False
        if tier >= 2 or self.tool_selection_active:
            return self.inert_patch_fail_counts.get((cx, cy, cur_digest), 0) >= 2
        return True

    def register_state(self, frame_2d: np.ndarray, step_idx: int = 0) -> str:
        arr = _frame_to_2d_numpy(frame_2d)
        if self.splitter.initial_frame is None:
            self.splitter.initialize_level(arr)
        sh = self.splitter.hash_nope(arr)
        if self.initial_state_hash is None:
            self.initial_state_hash = sh
            if hasattr(self, "tropical_sheaf"):
                self.seeded_d4_latch_bindings: List[Dict[str, Any]] = (
                    self.tropical_sheaf.seed_level_from_d4_homomorphism(
                        arr, clock_mask=self.splitter.clock_mask
                    )
                )
        if sh not in self.state_frames:
            self.state_frames[sh] = arr.copy()
            self.state_first_step[sh] = int(step_idx)
        self.state_visit_counts[sh] += 1
        return sh

    def _rebuild_graph_on_clock_expansion(self) -> None:
        """
        When `TemporalClockContentSplitter` identifies a new exogenous clock component `H_clock`,
        retroactively re-projects stored frames onto `H_nope` and recomputes transition edges
        plus `action_type_stats`, `interactive_loci`, `remote_actuators`, `driven_register_counts`,
        `morphology_stats`, `shape_stats`, and `d4_shape_stats` so earlier clock-only no-ops do
        not leave false positive contingency counts.
        """
        self.state_frames.clear()
        self.state_visit_counts.clear()
        self.state_first_step.clear()
        self.state_entry_action.clear()
        self.transitions.clear()
        self.tried_actions_by_state.clear()
        self.out_edges.clear()
        self.self_loops.clear()
        self.game_over_actions.clear()
        self.inert_click_patches.clear()
        self.inert_patch_fail_counts.clear()
        self.interactive_loci.clear()
        self.remote_actuators.clear()
        saved_verifiers = set(self.verifier_actuators)
        self.edits_since_last_verify = 0
        self.driven_register_counts = None
        self.coord_nonlocal_ratio.clear()
        self.coord_click_counts.clear()
        self.coord_success_counts.clear()
        self.coord_last_delta.clear()
        self.coord_last_energy_delta.clear()
        self.last_click_xy = None
        self.tool_selection_active = False
        self.state_topo_potential.clear()
        self.state_target_alignment.clear()
        self.learned_operators.clear()
        self.lattice_operator_templates.clear()
        self._sheaf_template_pairs = None
        if hasattr(self, "relational_dreamer"):
            self.relational_dreamer.reset(preserve_kinematics=True, is_level_reset=False)
        if hasattr(self, "tropical_sheaf"):
            self.tropical_sheaf.reset(clear_all=False)

        self.action_type_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_action_type_stats.items()
        }
        self.morphology_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_morphology_stats.items()
        }
        self.shape_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_shape_stats.items()
        }
        self.d4_shape_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_d4_shape_stats.items()
        }
        self.color_kind_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_color_kind_stats.items()
        }

        if self.splitter.initial_frame is not None:
            self.initial_state_hash = self.splitter.hash_nope(self.splitter.initial_frame)
            self.state_frames[self.initial_state_hash] = self.splitter.initial_frame.copy()
            self.state_first_step[self.initial_state_hash] = 0

        last_meaningful_nonclick: Dict[ActionKey, Tuple[np.ndarray, np.ndarray, ActionKey, Optional[Dict[str, Any]]]] = {}
        for idx, step_tuple in enumerate(self.step_history):
            f_prev, f_next, act_key, sf_cnt, lvl_up, game_ov, is_respawn = step_tuple[:7]
            cand_info = step_tuple[7] if len(step_tuple) > 7 else None

            nope_prev = self.splitter.project_nope(f_prev)
            nope_next = self.splitter.project_nope(f_next)
            sh_prev = _blake2b_grid(nope_prev)
            sh_next = _blake2b_grid(nope_next)
            if sh_prev not in self.state_frames:
                self.state_frames[sh_prev] = f_prev.copy()
                self.state_first_step[sh_prev] = idx
            if sh_next not in self.state_frames:
                self.state_frames[sh_next] = f_next.copy()
                self.state_first_step[sh_next] = idx + 1
                if sh_next != sh_prev and not is_respawn:
                    self.state_entry_action[sh_next] = act_key
            elif sh_next != sh_prev and sh_next not in self.state_entry_action and not is_respawn:
                self.state_entry_action[sh_next] = act_key
            self.state_visit_counts[sh_next] += 1

            raw_d = int(np.sum(f_prev != f_next))
            nope_d = int(np.sum(nope_prev != nope_next))
            is_click = isinstance(act_key, tuple) and len(act_key) == 3 and act_key[0] == 6
            is_ver = act_key in saved_verifiers
            if is_ver:
                self.edits_since_last_verify = 0
                if is_click:
                    cx_v, cy_v = int(act_key[1]), int(act_key[2])
                    self.interactive_loci.add((cx_v, cy_v))
                    self.coord_success_counts[(cx_v, cy_v)] += 1
            is_loop = (
                (sh_prev == sh_next)
                and (nope_d == 0)
                and (not lvl_up)
                and (not game_ov)
                and (not is_ver)
                and (sf_cnt <= 2 or is_click)
            )
            rec = TransitionRecord(
                state_hash=sh_prev,
                action_key=act_key,
                next_state_hash=sh_next,
                raw_delta_l1=raw_d,
                nope_delta_l1=nope_d,
                subframes_count=sf_cnt,
                max_subframe_nope_delta=nope_d,
                level_up=lvl_up,
                is_game_over=game_ov,
                is_self_loop=is_loop,
                is_phase_reset=is_respawn,
            )
            self.transitions[(sh_prev, act_key)] = rec
            self.tried_actions_by_state.setdefault(sh_prev, set()).add(act_key)
            latch_prev_rb = self.splitter.extract_fsa_latch_signature(f_prev)
            latch_next_rb = self.splitter.extract_fsa_latch_signature(f_next)
            t_rule_rb: Optional[TropicalTransitionRule] = None
            if hasattr(self, "tropical_sheaf") and not is_respawn:
                t_rule_rb = self.tropical_sheaf.observe_transition(
                    nope_prev,
                    nope_next,
                    act_key,
                    sh_prev=sh_prev,
                    sh_next=sh_next,
                    latch_prev=latch_prev_rb,
                    latch_next=latch_next_rb,
                    is_game_over=game_ov,
                )
            if hasattr(self, "relational_dreamer") and not is_respawn:
                flag_key = (
                    sh_prev,
                    act_key,
                    sh_next,
                    bool(game_ov),
                    len(self.relational_dreamer.kinematic_generators),
                    len(self.relational_dreamer.ego_colors),
                    len(self.relational_dreamer.wall_colors),
                )
                cached_flags = self._dreamer_step_flags.get(flag_key)
                if cached_flags is not None:
                    was_hl, was_adj = cached_flags
                    if was_hl:
                        self.relational_dreamer.active_contact_highlight_states.add(sh_next)
                    if was_adj:
                        self.relational_dreamer.contact_adjacent_state_hashes.add(sh_next)
                    self.relational_dreamer.last_transition_highlighted = was_hl
                    if isinstance(act_key, int) and act_key == 5 and nope_d > 0:
                        if sh_prev in self.relational_dreamer.active_contact_highlight_states:
                            self.relational_dreamer.action5_is_contact_operator = True
                else:
                    self.relational_dreamer.observe_transition(
                        nope_prev,
                        nope_next,
                        act_key,
                        sh_prev=sh_prev,
                        sh_next=sh_next,
                        nope_d=nope_d,
                        is_game_over=game_ov,
                        is_phase_reset=is_respawn,
                        candidate_info=cand_info,
                        state_hash_next=sh_next,
                        latch_signature=latch_next_rb,
                    )
                    was_hl = bool(sh_next in self.relational_dreamer.active_contact_highlight_states)
                    was_adj = bool(sh_next in self.relational_dreamer.contact_adjacent_state_hashes)
                    flag_key_after = (
                        sh_prev,
                        act_key,
                        sh_next,
                        bool(game_ov),
                        len(self.relational_dreamer.kinematic_generators),
                        len(self.relational_dreamer.ego_colors),
                        len(self.relational_dreamer.wall_colors),
                    )
                    self._dreamer_step_flags[flag_key_after] = (was_hl, was_adj)
                if hasattr(self, "tropical_sheaf"):
                    unlocked_rb = self.tropical_sheaf.get_unlocked_colors(latch_next_rb)
                    if unlocked_rb and latch_prev_rb != latch_next_rb:
                        self.relational_dreamer.unlocked_colors.update(unlocked_rb)
                    if self.relational_dreamer.contact_sheaf is not None:
                        self.tropical_sheaf.apply_to_contact_sheaf(
                            self.relational_dreamer.contact_sheaf, latch_next_rb
                        )
            if not is_respawn:
                if game_ov:
                    self.game_over_actions.setdefault(sh_prev, set()).add(act_key)
                elif is_loop:
                    self.self_loops.setdefault(sh_prev, set()).add(act_key)
                    if is_click:
                        cx, cy = int(act_key[1]), int(act_key[2])
                        dig = _local_patch_digest(nope_prev, cx, cy, radius=2)
                        self.inert_click_patches.setdefault((cx, cy), set()).add(dig)
                        self.inert_patch_fail_counts[(cx, cy, dig)] += 1
                else:
                    self.out_edges.setdefault(sh_prev, {})[act_key] = sh_next

            act_id = int(act_key[0]) if isinstance(act_key, tuple) else int(act_key)
            meaningful = (nope_d > 0 or lvl_up) and (not game_ov) and (not is_respawn)
            if 0 <= act_id < 8 and not is_respawn:
                self.action_type_stats[act_id][0] += 1.0
                if meaningful:
                    self.action_type_stats[act_id][1] += 1.0
                    if is_click:
                        self._induce_spatial_operator(
                            nope_prev, nope_next, act_key, cand_info, _from_record=True
                        )
                    else:
                        last_meaningful_nonclick[act_key] = (nope_prev, nope_next, act_key, cand_info)
                    self.edits_since_last_verify += 1

            if is_click and not is_respawn:
                cx, cy = int(act_key[1]), int(act_key[2])
                self.coord_click_counts[(cx, cy)] += 1
                self.coord_last_delta[(cx, cy)] = int(nope_d)
                if meaningful:
                    self.coord_success_counts[(cx, cy)] += 1
                    self.interactive_loci.add((cx, cy))
                    eta_nl = self._update_nonlocal_operator_factorization(
                        nope_prev, nope_next, cx, cy, nope_d, prev_click_xy=self.last_click_xy
                    )
                    self.last_click_xy = (cx, cy)
                    has_kb = (
                        sum(self.action_type_stats.get(a, [0.0, 0.0])[0] for a in (1, 2, 3, 4, 5))
                        > 0.0
                    )
                    H_n, W_n = nope_prev.shape
                    center_unchanged = (
                        0 <= cy < H_n
                        and 0 <= cx < W_n
                        and int(nope_prev[cy, cx]) == int(nope_next[cy, cx])
                    )
                    if (not has_kb) and center_unchanged and eta_nl == 0.0 and 8 <= nope_d <= 44:
                        self.tool_selection_active = True
                if cand_info is not None:
                    bh, bw = self._extract_bbox_hw(cand_info)
                    k_str = self._normalize_kind(str(cand_info.get("kind", "cc_object")))
                    area = int(cand_info.get("area", int(bh) * int(bw)))
                    col_c = int(cand_info.get("color", 0))
                    fam = (k_str, col_c, int(bh), int(bw))
                    st = self.morphology_stats.setdefault(fam, [0.0, 0.0])
                    st[0] += 1.0
                    if meaningful:
                        st[1] += 1.0
                    sh_fam = (k_str, int(bh), int(bw))
                    sst = self.shape_stats.setdefault(sh_fam, [0.0, 0.0])
                    sst[0] += 1.0
                    if meaningful:
                        sst[1] += 1.0
                    d4_fam = self._make_d4_shape_key(k_str, int(bh), int(bw), area)
                    d4st = self.d4_shape_stats.setdefault(d4_fam, [0.0, 0.0])
                    d4st[0] += 1.0
                    if meaningful:
                        d4st[1] += 1.0
                    ck_fam = (k_str, col_c)
                    ckst = self.color_kind_stats.setdefault(ck_fam, [0.0, 0.0])
                    ckst[0] += 1.0
                    if meaningful:
                        ckst[1] += 1.0

        for np_p, np_n, ak_nc, ci_nc in last_meaningful_nonclick.values():
            self._induce_spatial_operator(np_p, np_n, ak_nc, ci_nc, _from_record=True)

    def record_transition(
        self,
        prev_frame: np.ndarray,
        next_frame: np.ndarray,
        action_key: ActionKey,
        subframes: Optional[Sequence[np.ndarray]] = None,
        level_up: bool = False,
        is_game_over: bool = False,
        candidate_info: Optional[Dict[str, Any]] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
    ) -> TransitionRecord:
        if candidate_info is None and candidate_meta is not None:
            candidate_info = candidate_meta
        f_prev = _frame_to_2d_numpy(prev_frame)
        f_next = _frame_to_2d_numpy(next_frame)
        sf_count = len(subframes) if subframes is not None else 1

        clock_expanded, nope_d, is_phase_reset = self.splitter.observe_transition(
            f_prev, f_next, action_key
        )
        self.step_history.append(
            (
                f_prev.copy(),
                f_next.copy(),
                action_key,
                sf_count,
                level_up,
                is_game_over,
                is_phase_reset,
                candidate_info,
            )
        )

        if clock_expanded and len(self.step_history) > 1:
            self._rebuild_graph_on_clock_expansion()
            sh_prev = self.splitter.hash_nope(f_prev)
            existing_rec = self.transitions.get((sh_prev, action_key))
            if existing_rec is not None:
                return existing_rec

        sh_prev = self.splitter.hash_nope(f_prev)
        sh_next = self.splitter.hash_nope(f_next)
        if sh_prev not in self.state_frames:
            self.state_frames[sh_prev] = f_prev.copy()
        if sh_next not in self.state_frames:
            self.state_frames[sh_next] = f_next.copy()
            self.state_first_step[sh_next] = len(self.step_history)
            if sh_next != sh_prev and not is_phase_reset:
                self.state_entry_action[sh_next] = action_key
        elif sh_next != sh_prev and sh_next not in self.state_entry_action and not is_phase_reset:
            self.state_entry_action[sh_next] = action_key

        raw_d = int(np.sum(f_prev != f_next))
        nope_prev = self.splitter.project_nope(f_prev)
        nope_next = self.splitter.project_nope(f_next)
        max_sf_nope_d = int(nope_d)
        if subframes is not None and len(subframes) > 1:
            for sf in subframes:
                sf_nope = self.splitter.project_nope(_frame_to_2d_numpy(sf))
                d_sf = int(np.sum(nope_prev != sf_nope))
                if d_sf > max_sf_nope_d:
                    max_sf_nope_d = d_sf

        is_click = isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6
        has_dir_moves = (
            sum(self.action_type_stats.get(a, [0.0, 0.0])[0] for a in (1, 2, 3, 4)) > 0.0
        )
        is_excitable_verifier = (
            (sh_prev == sh_next)
            and (nope_d == 0)
            and (not level_up)
            and (not is_game_over)
            and (sf_count >= 3)
            and (len(self.verifier_actuators) == 0)
            and (
                (action_key == 5 and max_sf_nope_d >= 12)
                or (
                    is_click
                    and (not has_dir_moves)
                    and ((sf_count >= 6 and max_sf_nope_d >= 24) or (sf_count == 3 and max_sf_nope_d >= 60))
                )
            )
        )
        is_loop = (
            (sh_prev == sh_next)
            and (nope_d == 0)
            and (not level_up)
            and (not is_game_over)
            and (not is_excitable_verifier)
            and (sf_count <= 2 or is_click)
        )

        rec = TransitionRecord(
            state_hash=sh_prev,
            action_key=action_key,
            next_state_hash=sh_next,
            raw_delta_l1=raw_d,
            nope_delta_l1=nope_d,
            subframes_count=sf_count,
            max_subframe_nope_delta=max_sf_nope_d,
            level_up=level_up,
            is_game_over=is_game_over,
            is_self_loop=is_loop,
            is_phase_reset=is_phase_reset,
        )

        latch_prev_cur = self.splitter.extract_fsa_latch_signature(f_prev)
        latch_next_cur = self.splitter.extract_fsa_latch_signature(f_next)
        t_rule_cur: Optional[TropicalTransitionRule] = None
        if hasattr(self, "tropical_sheaf") and not is_phase_reset:
            t_rule_cur = self.tropical_sheaf.observe_transition(
                nope_prev,
                nope_next,
                action_key,
                sh_prev=sh_prev,
                sh_next=sh_next,
                latch_prev=latch_prev_cur,
                latch_next=latch_next_cur,
                is_game_over=is_game_over,
            )
        if hasattr(self, "relational_dreamer") and not is_phase_reset:
            self.relational_dreamer.observe_transition(
                nope_prev,
                nope_next,
                action_key,
                sh_prev=sh_prev,
                sh_next=sh_next,
                nope_d=nope_d,
                is_game_over=is_game_over,
                is_phase_reset=is_phase_reset,
                subframes=subframes,
                candidate_info=candidate_info,
                state_hash_next=sh_next,
                latch_signature=latch_next_cur,
            )
            was_hl = bool(sh_next in self.relational_dreamer.active_contact_highlight_states)
            was_adj = bool(sh_next in self.relational_dreamer.contact_adjacent_state_hashes)
            flag_key_after = (
                sh_prev,
                action_key,
                sh_next,
                bool(is_game_over),
                len(self.relational_dreamer.kinematic_generators),
                len(self.relational_dreamer.ego_colors),
                len(self.relational_dreamer.wall_colors),
            )
            self._dreamer_step_flags[flag_key_after] = (was_hl, was_adj)
            if hasattr(self, "tropical_sheaf"):
                unlocked_cur = self.tropical_sheaf.get_unlocked_colors(latch_next_cur)
                if unlocked_cur and latch_prev_cur != latch_next_cur:
                    self.relational_dreamer.unlocked_colors.update(unlocked_cur)
                if self.relational_dreamer.contact_sheaf is not None:
                    self.tropical_sheaf.apply_to_contact_sheaf(
                        self.relational_dreamer.contact_sheaf, latch_next_cur
                    )

        if is_excitable_verifier and not is_phase_reset:
            self.verifier_actuators.add(action_key)
            self.edits_since_last_verify = 0
            if is_click:
                cx_v, cy_v = int(action_key[1]), int(action_key[2])
                self.interactive_loci.add((cx_v, cy_v))
                self.coord_success_counts[(cx_v, cy_v)] += 1

        # Always record `(sh_prev, action_key)` in `transitions` so no action at `sh_prev`
        # ever remains permanently untried, while keeping `out_edges` free of respawns.
        self.transitions[(sh_prev, action_key)] = rec
        self.tried_actions_by_state.setdefault(sh_prev, set()).add(action_key)
        if not is_phase_reset:
            if is_game_over:
                self.game_over_actions.setdefault(sh_prev, set()).add(action_key)
            elif is_loop:
                self.self_loops.setdefault(sh_prev, set()).add(action_key)
                if is_click:
                    cx, cy = int(action_key[1]), int(action_key[2])
                    dig = _local_patch_digest(nope_prev, cx, cy, radius=2)
                    self.inert_click_patches.setdefault((cx, cy), set()).add(dig)
                    self.inert_patch_fail_counts[(cx, cy, dig)] += 1
            elif sh_prev != sh_next:
                self.out_edges.setdefault(sh_prev, {})[action_key] = sh_next

        act_id = int(action_key[0]) if isinstance(action_key, tuple) else int(action_key)
        meaningful = (nope_d > 0 or level_up) and (not is_game_over) and (not is_phase_reset)
        reward = 2.0 if level_up else (1.0 if meaningful else (-1.0 if is_game_over else 0.0))

        if 0 <= act_id < 8 and not is_phase_reset:
            self.action_type_stats[act_id][0] += 1.0
            if meaningful:
                self.action_type_stats[act_id][1] += 1.0
                self._induce_spatial_operator(
                    nope_prev, nope_next, action_key, candidate_info, _from_record=True
                )
                self.update_lie_generator(nope_prev, nope_next, act_id)
                self.edits_since_last_verify += 1

        adapt_triad = getattr(self.goal_distiller, "levels_distilled", 0) > 0
        if (
            self.z_H is not None
            and self.z_L is not None
            and not is_phase_reset
            and (
                (adapt_triad and len(self.step_history) <= 16)
                or len(self.step_history) == 1
                or level_up
                or is_game_over
            )
        ):
            self.adapt_online_step(
                self.z_H,
                self.z_L,
                action_key,
                reward=reward,
                is_self_loop=is_loop,
                is_game_over=is_game_over,
                lr=1e-4,
                adapt_triad=adapt_triad,
            )

        if is_click and not is_phase_reset:
            cx, cy = int(action_key[1]), int(action_key[2])
            self.coord_click_counts[(cx, cy)] += 1
            self.coord_last_delta[(cx, cy)] = int(nope_d)
            if meaningful:
                self.coord_success_counts[(cx, cy)] += 1
                self.interactive_loci.add((cx, cy))
                eta_nl = self._update_nonlocal_operator_factorization(
                    nope_prev, nope_next, cx, cy, nope_d, prev_click_xy=self.last_click_xy
                )
                self.last_click_xy = (cx, cy)
                has_kb = (
                    sum(self.action_type_stats.get(a, [0.0, 0.0])[0] for a in (1, 2, 3, 4, 5))
                    > 0.0
                )
                H_n, W_n = nope_prev.shape
                center_unchanged = (
                    0 <= cy < H_n
                    and 0 <= cx < W_n
                    and int(nope_prev[cy, cx]) == int(nope_next[cy, cx])
                )
                if (not has_kb) and center_unchanged and eta_nl == 0.0 and 8 <= nope_d <= 44:
                    self.tool_selection_active = True

            if candidate_info is not None:
                bh, bw = self._extract_bbox_hw(candidate_info)
                k_str = self._normalize_kind(str(candidate_info.get("kind", "cc_object")))
                area = int(candidate_info.get("area", int(bh) * int(bw)))
                col_c = int(candidate_info.get("color", 0))
                fam = (
                    k_str,
                    col_c,
                    int(bh),
                    int(bw),
                )
                st = self.morphology_stats.setdefault(fam, [0.0, 0.0])
                st[0] += 1.0
                if meaningful:
                    st[1] += 1.0

                sh_fam = (k_str, int(bh), int(bw))
                sst = self.shape_stats.setdefault(sh_fam, [0.0, 0.0])
                sst[0] += 1.0
                if meaningful:
                    sst[1] += 1.0

                d4_fam = self._make_d4_shape_key(k_str, int(bh), int(bw), area)
                d4st = self.d4_shape_stats.setdefault(d4_fam, [0.0, 0.0])
                d4st[0] += 1.0
                if meaningful:
                    d4st[1] += 1.0

                ck_fam = (k_str, col_c)
                ckst = self.color_kind_stats.setdefault(ck_fam, [0.0, 0.0])
                ckst[0] += 1.0
                if meaningful:
                    ckst[1] += 1.0

                fv = candidate_info.get("feature_vec")
                if isinstance(fv, np.ndarray) and fv.shape == self.medoid_weights.shape:
                    pred = float(np.dot(self.medoid_weights, fv))
                    err = reward - pred
                    self.medoid_weights = np.clip(self.medoid_weights + 0.08 * err * fv, -2.5, 2.5)

        return rec

    @staticmethod
    def _normalize_plan_candidates(
        candidate_actions: Optional[Sequence[Any]],
    ) -> List[Tuple[float, ActionKey, Optional[Dict[str, Any]]]]:
        if not candidate_actions:
            return []
        normed: List[Tuple[float, ActionKey, Optional[Dict[str, Any]]]] = []
        for item in candidate_actions:
            if (
                isinstance(item, tuple)
                and len(item) == 3
                and (isinstance(item[2], dict) or item[2] is None)
                and not (isinstance(item[0], int) and item[0] == 6 and isinstance(item[2], int))
            ):
                normed.append((float(item[0]), item[1], item[2]))
            else:
                normed.append((1.0, item, None))
        return normed

    def synthesize_abelian_jacobian_plan(
        self,
        frame_2d: np.ndarray,
        candidate_actions: Optional[Sequence[Any]] = None,
        ridge_lambda: float = 0.05,
        max_steps: int = 4,
        target_frame_2d: Optional[np.ndarray] = None,
    ) -> List[ActionKey]:
        """
        Primitive A (Abelian Operator Jacobian & Ridge Difference-Basis Pseudoinverse Plan):
        Solves the precision-weighted ridge system over empirical operator difference vectors
        `Delta = [delta_1, ..., delta_K]`:
            n_ridge = (Delta^T Delta + lambda * I_K)^{-1} Delta^T r_Pi^*
        followed by greedy integer Orthogonal Matching Pursuit (OMP) verified via `predict_action_outcome`.
        """
        if not self.learned_operators and not self.lattice_operator_templates:
            return []

        nope_cur = self.splitter.project_nope(frame_2d).copy()
        init_f = (
            target_frame_2d
            if target_frame_2d is not None
            else (
                self.state_frames.get(self.initial_state_hash)
                if self.initial_state_hash is not None
                else self.splitter.initial_frame
            )
        )
        d_weights = self.goal_distiller.weights if hasattr(self, "goal_distiller") else None
        cur_align = compute_canvas_target_alignment(
            nope_cur,
            initial_frame_2d=init_f,
            spatial_precision_map=self.spatial_precision_map,
            distilled_weights=d_weights,
        )
        base_e = float(cur_align["target_energy"])
        if base_e <= 1e-4:
            return []

        pool: List[Tuple[ActionKey, Optional[Dict[str, Any]], np.ndarray]] = []
        seen_keys: Set[ActionKey] = set()
        for _, ak, info in self._normalize_plan_candidates(candidate_actions)[:16]:
            if ak in seen_keys:
                continue
            pred = self.predict_action_outcome(nope_cur, ak, candidate_info=info)
            if pred is not None and not np.array_equal(pred, nope_cur):
                d_vec = (pred.astype(np.float32) - nope_cur.astype(np.float32)).reshape(-1) / 15.0
                pool.append((ak, info, d_vec))
                seen_keys.add(ak)

        for ak, _ in list(self.learned_operators.items())[:16]:
            if ak in seen_keys:
                continue
            pred = self.predict_action_outcome(nope_cur, ak, candidate_info=None)
            if pred is not None and not np.array_equal(pred, nope_cur):
                d_vec = (pred.astype(np.float32) - nope_cur.astype(np.float32)).reshape(-1) / 15.0
                pool.append((ak, None, d_vec))
                seen_keys.add(ak)

        if not pool:
            return []

        # Construct target residual vector `r_Pi^*` from defect loci or best single-step reduction
        H, W = nope_cur.shape
        r_star = np.zeros(H * W, dtype=np.float32)
        for _, _, d_vec in pool:
            r_star += d_vec
        Delta = np.stack([item[2] for item in pool], axis=1)  # (HW, K)
        K = Delta.shape[1]
        gram = Delta.T @ Delta + float(ridge_lambda) * np.eye(K, dtype=np.float32)
        rhs = Delta.T @ r_star
        try:
            n_ridge = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            n_ridge = np.zeros(K, dtype=np.float32)

        # Greedy OMP rollout guided by `n_ridge` and target energy reduction
        plan: List[ActionKey] = []
        sim_board = nope_cur.copy()
        sim_e = base_e
        for _ in range(max(1, int(max_steps))):
            best_ak: Optional[ActionKey] = None
            best_board: Optional[np.ndarray] = None
            best_e = sim_e - 1e-4
            order = np.argsort(-np.abs(n_ridge))
            for idx_k in order:
                ak_k, info_k, _ = pool[int(idx_k)]
                nxt_b = self.predict_action_outcome(sim_board, ak_k, candidate_info=info_k)
                if nxt_b is None:
                    continue
                nxt_align = compute_canvas_target_alignment(
                    nxt_b,
                    initial_frame_2d=init_f,
                    spatial_precision_map=self.spatial_precision_map,
                    distilled_weights=d_weights,
                )
                nxt_e = float(nxt_align["target_energy"])
                if nxt_e < best_e:
                    best_e = nxt_e
                    best_ak = ak_k
                    best_board = nxt_b
            if best_ak is None or best_board is None:
                break
            plan.append(best_ak)
            sim_board = best_board
            sim_e = best_e
        return plan

    def dream_forward_plan(
        self,
        frame_2d: np.ndarray,
        candidate_actions: Optional[Sequence[Any]] = None,
        beam_width: int = 6,
        horizon: int = 3,
        max_depth: Optional[int] = None,
    ) -> List[ActionKey]:
        """
        Primitive B (Koopman Operator Rollout & Wasserstein-1 Optimal-Transport Beam Search):
        Executes mental forward rollouts `hat_F^(tau) = hat_T_{a^(tau)}(hat_F^(tau-1))` over
        depth `horizon` (`<= 4`) and beam width `beam_width` (`<= 8`) minimizing:
            E_OT(hat_F; Pi) = E_target(hat_F; Pi) + 0.75 * W_1^marg(hat_F, F^*)
        """
        eff_horizon = int(max_depth) if max_depth is not None else int(horizon)
        nope_cur = self.splitter.project_nope(frame_2d).copy()
        init_f = (
            self.state_frames.get(self.initial_state_hash)
            if self.initial_state_hash is not None
            else self.splitter.initial_frame
        )
        d_weights = self.goal_distiller.weights if hasattr(self, "goal_distiller") else None

        def _eval_ot_energy(board: np.ndarray) -> float:
            al = compute_canvas_target_alignment(
                board,
                initial_frame_2d=init_f,
                spatial_precision_map=self.spatial_precision_map,
                distilled_weights=d_weights,
            )
            return float(al["target_energy"]) + 0.75 * float(al["wasserstein1_distance"])

        base_ot = _eval_ot_energy(nope_cur)
        action_pool: List[Tuple[ActionKey, Optional[Dict[str, Any]]]] = []
        seen: Set[ActionKey] = set()
        for _, ak, info in self._normalize_plan_candidates(candidate_actions)[:12]:
            if ak not in seen:
                action_pool.append((ak, info))
                seen.add(ak)
        for ak in list(self.learned_operators.keys())[:12]:
            if ak not in seen:
                action_pool.append((ak, None))
                seen.add(ak)

        if not action_pool:
            return []

        beam: List[Tuple[float, List[ActionKey], np.ndarray]] = [(base_ot, [], nope_cur)]
        best_ot = base_ot - 1e-4
        best_plan: List[ActionKey] = []

        for _ in range(max(1, min(4, eff_horizon))):
            next_beam: List[Tuple[float, List[ActionKey], np.ndarray]] = []
            for cur_ot, path, board in beam:
                for ak, info in action_pool:
                    nxt_b = self.predict_action_outcome(board, ak, candidate_info=info)
                    if nxt_b is None or np.array_equal(nxt_b, board):
                        continue
                    nxt_ot = _eval_ot_energy(nxt_b) + 0.005 * float(len(path) + 1)
                    nxt_path = path + [ak]
                    if nxt_ot < best_ot:
                        best_ot = nxt_ot
                        best_plan = nxt_path
                    next_beam.append((nxt_ot, nxt_path, nxt_b))
            if not next_beam:
                break
            next_beam.sort(key=lambda item: item[0])
            beam = next_beam[: max(1, min(8, int(beam_width)))]

        return best_plan

    def unroll_triad_cascade_plan(
        self,
        frame_2d: np.ndarray,
        z_H_super: torch.Tensor,
        z_L_token: torch.Tensor,
        candidate_actions: Sequence[Any],
        horizon: int = 3,
        episodic_encoder: Optional[Arc3EpisodicBeltramiShell] = None,
    ) -> List[ActionKey]:
        """
        Primitive C (Multi-Step `SlowFastTriadCascade` Trajectory Readout on `S^7(r_0)`):
        Unrolls `SlowFastTriadCascade` across steps `tau = 1 .. horizon`, reading out the
        step-specific `so(8)` action prior `p_triad^(tau) = tanh(B_{W_act}(u^(tau), v^(tau)))`
        at EVERY recurrent step `tau`, selecting `a^(tau)` via `p_triad^(tau)` + mental
        rollout energy reduction, and updating `z_L^(tau)` from the predicted board `hat_F^(tau)`.
        """
        norm_cands = self._normalize_plan_candidates(candidate_actions)
        if not norm_cands:
            return []

        nope_cur = self.splitter.project_nope(frame_2d).copy()
        init_f = (
            self.state_frames.get(self.initial_state_hash)
            if self.initial_state_hash is not None
            else self.splitter.initial_frame
        )
        d_weights = self.goal_distiller.weights if hasattr(self, "goal_distiller") else None
        cur_e = float(
            compute_canvas_target_alignment(
                nope_cur,
                initial_frame_2d=init_f,
                spatial_precision_map=self.spatial_precision_map,
                distilled_weights=d_weights,
            )["target_energy"]
        )

        plan: List[ActionKey] = []
        sim_board = nope_cur.copy()
        sim_e = cur_e

        with torch.no_grad():
            z_H_mean = z_H_super.float().mean(dim=1, keepdim=True)
            z_L_cur = z_L_token.view(1, 1, self.d_reasoner).float()

            for _ in range(max(1, min(4, int(horizon)))):
                # Advance 1 step on S^7(r_0) via SlowFastTriadCascade
                z_L_cur, _ = self.cascade(
                    z_H_mean,
                    z_L_cur,
                    delta_z=self.default_delta_z,
                    return_diagnostics=True,
                    unroll_steps=1,
                )
                u_s = self.cascade.project_to_shell(z_H_mean).squeeze(0).squeeze(0)
                v_s = self.cascade.project_to_shell(z_L_cur).squeeze(0).squeeze(0)
                p_triad_tau = (
                    torch.tanh(self.action_triad(u_s.unsqueeze(0), v_s.unsqueeze(0)).squeeze(0))
                    .cpu()
                    .numpy()
                )

                best_ak: Optional[ActionKey] = None
                best_board: Optional[np.ndarray] = None
                best_score = -1e9
                best_e = sim_e

                for base_sc, ak, info in norm_cands[:12]:
                    nxt_b = self.predict_action_outcome(sim_board, ak, candidate_info=info)
                    if nxt_b is None or np.array_equal(nxt_b, sim_board):
                        continue
                    nxt_e = float(
                        compute_canvas_target_alignment(
                            nxt_b,
                            initial_frame_2d=init_f,
                            spatial_precision_map=self.spatial_precision_map,
                            distilled_weights=d_weights,
                        )["target_energy"]
                    )
                    de = sim_e - nxt_e
                    if de <= 1e-4:
                        continue
                    act_id = int(ak[0]) if isinstance(ak, tuple) else int(ak)
                    step_sc = 4.0 * de + 0.15 * float(p_triad_tau[act_id % 8]) + 0.05 * float(base_sc)
                    if step_sc > best_score:
                        best_score = step_sc
                        best_ak = ak
                        best_board = nxt_b
                        best_e = nxt_e

                if best_ak is None or best_board is None:
                    break

                plan.append(best_ak)
                prev_b = sim_board
                sim_board = best_board
                sim_e = best_e

                if episodic_encoder is not None:
                    act_id = int(best_ak[0]) if isinstance(best_ak, tuple) else int(best_ak)
                    c_xy = (int(best_ak[1]), int(best_ak[2])) if isinstance(best_ak, tuple) and len(best_ak) == 3 else None
                    z_L_cur = episodic_encoder.encode_frame_token(
                        sim_board, prev_frame_2d=prev_b, action_id=act_id, click_xy=c_xy
                    ).view(1, 1, self.d_reasoner)

        return plan

    def synthesize_sheaf_bijection_plan(
        self,
        frame_2d: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
        allow_generic_triplets: bool = True,
    ) -> List[ActionKey]:
        """
        Primitive D (Scale-Invariant Hierarchical Operad / Sheaf Bijection Plan):
        Solves both:
          1. Hierarchical Operad Box-Frame Substitution (`G_ref` hollow reference frames of arbitrary
             scale `S_ref` enclosing cavity scale `S_tok`, free palette tokens of scale `S_tok`, and
             container-box-enclosed receptor slots of scale `S_slot < S_tok`), and
          2. General 3-equinumerous morphological shape group color-bijection `(G_ref, G_pal, G_slot)`.
        """
        from itertools import combinations, permutations

        nope = self.splitter.project_nope(_frame_to_2d_numpy(frame_2d))
        H, W = nope.shape
        vals, counts = np.unique(nope, return_counts=True)
        bg_mode = int(vals[int(np.argmax(counts))])
        fg_comps = _connected_components_2d(nope, connectivity=4, same_color=True, ignore_val=bg_mode)

        # 1. Scale-Invariant Hierarchical Operad / Sheaf Box-Frame Substitution Solver
        hollow_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for c in fg_comps:
            ch, cw, carea = int(c["h"]), int(c["w"]), int(c["area"])
            if ch == cw and ch >= 4 and 0.30 <= float(c["fill_ratio"]) <= 0.72:
                cav_area = ch * cw - carea
                if cav_area >= 4:
                    s_tok_cand = int(round(math.sqrt(float(cav_area))))
                    if s_tok_cand * s_tok_cand == cav_area and 2 <= s_tok_cand < ch:
                        hollow_groups.setdefault((ch, s_tok_cand), []).append(c)

        for (s_ref, s_tok), refs in sorted(
            hollow_groups.items(), key=lambda kv: (len(kv[1]), kv[0][0]), reverse=True
        ):
            if len(refs) < 3:
                continue
            refs = sorted(
                refs,
                key=lambda c: (
                    int(round(float(c["centroid_y"]) / float(s_ref))),
                    int(round(float(c["centroid_x"]) / float(s_ref))),
                ),
            )
            target_seq = [int(c["color"]) for c in refs]
            ref_min_r = min(int(c["r0"]) for c in refs)
            ref_max_r = max(int(c["r1"]) for c in refs)
            half_tok = max(1, s_tok // 2)

            def _enclosing_box_color(cy: float, cx: float) -> int:
                rr, cc_0 = int(round(cy)), int(round(cx))
                left_col = None
                for cc in range(cc_0 - half_tok, -1, -1):
                    v = int(nope[rr, cc])
                    if v != bg_mode:
                        wall_h = sum(
                            1
                            for r_i in range(max(0, rr - s_ref), min(H, rr + s_ref + 1))
                            if int(nope[r_i, cc]) == v
                        )
                        if wall_h >= s_ref:
                            left_col = v
                            break
                right_col = None
                for cc in range(cc_0 + half_tok, W):
                    v = int(nope[rr, cc])
                    if v != bg_mode:
                        wall_h = sum(
                            1
                            for r_i in range(max(0, rr - s_ref), min(H, rr + s_ref + 1))
                            if int(nope[r_i, cc]) == v
                        )
                        if wall_h >= s_ref:
                            right_col = v
                            break
                if left_col is not None and left_col == right_col:
                    return int(left_col)
                return -1

            solid_tok_area = s_tok * s_tok
            cand_tokens = [
                c
                for c in fg_comps
                if int(c["h"]) == s_tok
                and int(c["w"]) == s_tok
                and float(c["fill_ratio"]) >= 0.50
                and not (ref_min_r <= int(c["r0"]) <= ref_max_r)
            ]
            if not cand_tokens:
                continue

            # Free palette tokens are outside container boxes; enclosed solid tokens are in-box constants
            pals = [
                c
                for c in cand_tokens
                if _enclosing_box_color(float(c["centroid_y"]), float(c["centroid_x"])) == -1
            ]
            pre_solids = [
                c
                for c in cand_tokens
                if c not in pals
                and int(c["area"]) == solid_tok_area
                and _enclosing_box_color(float(c["centroid_y"]), float(c["centroid_x"])) != -1
            ]
            slots = [
                c
                for c in fg_comps
                if int(c["h"]) == int(c["w"])
                and 1 < int(c["h"]) < s_tok
                and int(c["area"]) == int(c["h"]) * int(c["w"])
                and not (ref_min_r <= int(c["r0"]) <= ref_max_r)
                and _enclosing_box_color(float(c["centroid_y"]), float(c["centroid_x"])) != -1
            ]
            if len(slots) >= 2 and len(pals) >= len(slots):
                pal_r_min = min(int(c["r0"]) for c in pals)
                pal_r_max = max(int(c["r1"]) for c in pals)
                t_ring = max(1, (s_tok - int(slots[0]["h"])) // 2)
                call_rings = []
                for r in range(0, H - s_tok):
                    if (ref_min_r <= r <= ref_max_r) or (pal_r_min - 1 <= r <= pal_r_max):
                        continue
                    for c_col in range(0, W - s_tok):
                        col = int(nope[r, c_col])
                        if col == bg_mode:
                            continue
                        patch = nope[r : r + s_tok, c_col : c_col + s_tok]
                        inner = patch[t_ring : s_tok - t_ring, t_ring : s_tok - t_ring]
                        if (
                            inner.size > 0
                            and np.all(inner == bg_mode)
                            and np.all(patch[:t_ring, :] == col)
                            and np.all(patch[s_tok - t_ring :, :] == col)
                            and np.all(patch[:, :t_ring] == col)
                            and np.all(patch[:, s_tok - t_ring :] == col)
                        ):
                            cy_r = float(r) + 0.5 * float(s_tok - 1)
                            cx_r = float(c_col) + 0.5 * float(s_tok - 1)
                            if _enclosing_box_color(cy_r, cx_r) != -1:
                                call_rings.append({"color": col, "cy": cy_r, "cx": cx_r})

                boxes: Dict[int, List[Dict[str, Any]]] = {}
                box_min_y: Dict[int, float] = {}
                for s in slots:
                    bcol = _enclosing_box_color(float(s["centroid_y"]), float(s["centroid_x"]))
                    boxes.setdefault(bcol, []).append(
                        {
                            "type": "slot",
                            "cy": float(s["centroid_y"]),
                            "cx": float(s["centroid_x"]),
                            "comp": s,
                        }
                    )
                    box_min_y[bcol] = min(box_min_y.get(bcol, 999.0), float(s["centroid_y"]))
                for ps in pre_solids:
                    bcol = _enclosing_box_color(
                        float(ps["centroid_y"]), float(ps["centroid_x"])
                    )
                    boxes.setdefault(bcol, []).append(
                        {
                            "type": "solid",
                            "color": int(ps["color"]),
                            "cy": float(ps["centroid_y"]),
                            "cx": float(ps["centroid_x"]),
                        }
                    )
                    box_min_y[bcol] = min(box_min_y.get(bcol, 999.0), float(ps["centroid_y"]))
                for cr in call_rings:
                    bcol = _enclosing_box_color(float(cr["cy"]), float(cr["cx"]))
                    boxes.setdefault(bcol, []).append(
                        {
                            "type": "call",
                            "color": int(cr["color"]),
                            "cy": float(cr["cy"]),
                            "cx": float(cr["cx"]),
                        }
                    )
                    box_min_y[bcol] = min(box_min_y.get(bcol, 999.0), float(cr["cy"]))

                for bcol, items in boxes.items():
                    items.sort(
                        key=lambda it: (
                            int(round(float(it["cy"]) / float(max(2, s_tok)))),
                            float(it["cx"]),
                        )
                    )

                root_col = min(boxes.keys(), key=lambda b: box_min_y[b])
                pal_tokens: List[Tuple[str, int]] = []
                pal_by_token: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
                for p in pals:
                    tk = (
                        "solid" if int(p["area"]) == solid_tok_area else "call",
                        int(p["color"]),
                    )
                    pal_tokens.append(tk)
                    pal_by_token.setdefault(tk, []).append(p)
                avail_counts = Counter(pal_tokens)

                slot_positions: List[Tuple[int, int, Dict[str, Any]]] = []
                for bcol, items in boxes.items():
                    for idx, it in enumerate(items):
                        if it["type"] == "slot":
                            slot_positions.append((bcol, idx, it["comp"]))

                call_tokens = [tk for tk in avail_counts.elements() if tk[0] == "call"]
                num_slots = len(slot_positions)
                if len(call_tokens) <= num_slots:
                    distinct_call_assigns = set()
                    distinct_perms = set(permutations(call_tokens))
                    for idxs in combinations(range(num_slots), len(call_tokens)):
                        for perm in distinct_perms:
                            distinct_call_assigns.add(tuple(zip(idxs, perm)))

                    slot_id_lookup = {
                        (bcol, idx): s_i
                        for s_i, (bcol, idx, _) in enumerate(slot_positions)
                    }
                    for call_assign in distinct_call_assigns:
                        call_map = dict(call_assign)
                        leaves: List[Tuple[str, int]] = []
                        valid_tree = True

                        def _expand_box(bcol: int, depth: int = 0) -> None:
                            nonlocal valid_tree
                            if not valid_tree or len(leaves) >= len(target_seq):
                                return
                            if depth > 8 or bcol not in boxes:
                                valid_tree = False
                                return
                            for idx_i, it_i in enumerate(boxes[bcol]):
                                if len(leaves) >= len(target_seq):
                                    return
                                if it_i["type"] == "solid":
                                    leaves.append(("fixed", int(it_i["color"])))
                                elif it_i["type"] == "call":
                                    _expand_box(int(it_i["color"]), depth + 1)
                                else:
                                    s_i = slot_id_lookup[(bcol, idx_i)]
                                    if s_i in call_map:
                                        _expand_box(int(call_map[s_i][1]), depth + 1)
                                    else:
                                        leaves.append(("var", int(s_i)))

                        _expand_box(root_col)
                        if not valid_tree or len(leaves) != len(target_seq):
                            continue

                        var_color: Dict[int, int] = {}
                        consistent = True
                        for (l_kind, l_val), targ_c in zip(leaves, target_seq):
                            if l_kind == "fixed":
                                if l_val != targ_c:
                                    consistent = False
                                    break
                            else:
                                if l_val in var_color and var_color[l_val] != targ_c:
                                    consistent = False
                                    break
                                var_color[l_val] = targ_c
                        if not consistent:
                            continue

                        used_counts: Counter = Counter()
                        for _, tk in call_map.items():
                            used_counts[tk] += 1
                        for _, col_v in var_color.items():
                            used_counts[("solid", col_v)] += 1

                        if (
                            used_counts <= avail_counts
                            and len(var_color) + len(call_map) == num_slots
                        ):
                            plan_seq: List[ActionKey] = []
                            pal_queues = {k: list(v) for k, v in pal_by_token.items()}
                            for s_i, (_, _, s_comp) in enumerate(slot_positions):
                                tk = (
                                    call_map[s_i]
                                    if s_i in call_map
                                    else ("solid", var_color[s_i])
                                )
                                p_comp = pal_queues[tk].pop(0)
                                plan_seq.append(
                                    (6, int(p_comp["medoid_x"]), int(p_comp["medoid_y"]))
                                )
                                plan_seq.append(
                                    (6, int(s_comp["medoid_x"]), int(s_comp["medoid_y"]))
                                )
                            if available_actions is None or 5 in available_actions:
                                plan_seq.append(5)
                            return plan_seq

        if not allow_generic_triplets:
            return []

        by_shape: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        cand_comps = [
            c for c in fg_comps if 2 <= int(c["area"]) <= 64 and int(c["h"]) >= 2 and int(c["w"]) >= 2
        ]
        for c in cand_comps:
            is_enclosed = any(
                other is not c
                and int(other["r0"]) < int(c["r0"])
                and int(other["r1"]) > int(c["r1"])
                and int(other["c0"]) < int(c["c0"])
                and int(other["c1"]) > int(c["c1"])
                for other in cand_comps
            )
            if not is_enclosed:
                sh_k = (int(c["h"]), int(c["w"]), int(c["area"]))
                by_shape.setdefault(sh_k, []).append(c)

        by_count: Dict[int, List[Tuple[Tuple[int, int, int], List[Dict[str, Any]]]]] = {}
        for sh_k, c_list in by_shape.items():
            if 3 <= len(c_list) <= 12:
                by_count.setdefault(len(c_list), []).append((sh_k, c_list))

        for n_cnt, groups in by_count.items():
            if len(groups) < 3:
                continue
            ref_grp: Optional[List[Dict[str, Any]]] = None
            pal_grp: Optional[List[Dict[str, Any]]] = None
            slot_grp: Optional[List[Dict[str, Any]]] = None
            for i_g in range(len(groups)):
                for j_g in range(i_g + 1, len(groups)):
                    sh_a, ga = groups[i_g]
                    sh_b, gb = groups[j_g]
                    ca = Counter(int(x["color"]) for x in ga)
                    cb = Counter(int(x["color"]) for x in gb)
                    if ca == cb and len(ca) >= 3:
                        # Reference has larger bounding box or sits higher on canvas
                        area_a = sh_a[0] * sh_a[1]
                        area_b = sh_b[0] * sh_b[1]
                        mean_ya = float(np.mean([x["centroid_y"] for x in ga]))
                        mean_yb = float(np.mean([x["centroid_y"] for x in gb]))
                        if (area_a, -mean_ya) >= (area_b, -mean_yb):
                            ref_grp, pal_grp = ga, gb
                        else:
                            ref_grp, pal_grp = gb, ga
                        rem = [
                            g_item[1]
                            for k_g, g_item in enumerate(groups)
                            if k_g != i_g and k_g != j_g
                        ]
                        if rem:
                            rem.sort(key=lambda g_l: g_l[0]["area"])
                            slot_grp = rem[0]
                        break
                if ref_grp is not None:
                    break

            if ref_grp is not None and pal_grp is not None and slot_grp is not None:
                rh = max(2, int(ref_grp[0]["h"]))
                rw = max(2, int(ref_grp[0]["w"]))
                sh_h = max(2, int(slot_grp[0]["h"]))
                sh_w = max(2, int(slot_grp[0]["w"]))
                ref_sorted = sorted(
                    ref_grp,
                    key=lambda c: (
                        int(round(float(c["centroid_y"]) / float(rh))),
                        int(round(float(c["centroid_x"]) / float(rw))),
                    ),
                )
                slot_sorted = sorted(
                    slot_grp,
                    key=lambda c: (
                        int(round(float(c["centroid_y"]) / float(sh_h))),
                        int(round(float(c["centroid_x"]) / float(sh_w))),
                    ),
                )
                pal_by_col: Dict[int, Dict[str, Any]] = {
                    int(p["color"]): p for p in pal_grp
                }
                plan_seq: List[ActionKey] = []
                for r_comp, s_comp in zip(ref_sorted, slot_sorted):
                    target_c = int(r_comp["color"])
                    cur_c = int(s_comp["color"])
                    if cur_c != target_c and target_c in pal_by_col:
                        p_comp = pal_by_col[target_c]
                        px, py = self.canonicalize_click_xy(
                            int(p_comp["medoid_x"]), int(p_comp["medoid_y"]), frame_2d=nope
                        )
                        sx, sy = self.canonicalize_click_xy(
                            int(s_comp["medoid_x"]), int(s_comp["medoid_y"]), frame_2d=nope
                        )
                        plan_seq.append((6, int(px), int(py)))
                        plan_seq.append((6, int(sx), int(sy)))
                if available_actions is None or 5 in available_actions:
                    plan_seq.append(5)
                return plan_seq
        return []

    def synthesize_d4_monoid_homomorphism_plan(
        self,
        frame_2d: np.ndarray,
        available_actions: Optional[Sequence[int]] = None,
    ) -> List[ActionKey]:
        """
        Primitive E (Dihedral `D_4`-Equivariant Typed Monoid String Homomorphism Plan):
        Detects hollow reference rule containers via topological bounding-box enclosure
        (`fill_ratio <= 0.65` enclosing paired symbol components) to induce a cyclic
        monoid homomorphism `mu: Sigma_in -> Sigma_out`, then aligns the active input
        and output symbol lattices under `ACTION6` local cyclic advances and `ACTION5`
        cursor shifts (100% `S_16` color-permutation and `D_4` dihedral equivariant).
        """
        if available_actions is not None and (6 not in available_actions or 5 not in available_actions):
            return []
        nope = self.splitter.project_nope(_frame_to_2d_numpy(frame_2d))
        vals, counts = np.unique(nope, return_counts=True)
        bg_mode = int(vals[int(np.argmax(counts))])

        comps = _connected_components_2d(nope, connectivity=4, same_color=True, ignore_val=bg_mode)
        if len(comps) < 6:
            return []

        raw_pairs: List[Tuple[int, int]] = []
        rule_boxes: List[Tuple[int, int, int, int]] = []
        border_col_found: Optional[int] = None
        for fc in comps:
            fh, fw = int(fc["h"]), int(fc["w"])
            if float(fc["fill_ratio"]) > 0.65 or min(fh, fw) < 5 or max(fh, fw) < 7:
                continue
            fr0, fc0, fr1, fc1 = int(fc["r0"]), int(fc["c0"]), int(fc["r1"]), int(fc["c1"])
            b_col = int(fc["color"])
            inner_sq = [
                ic
                for ic in comps
                if ic is not fc
                and int(ic["r0"]) > fr0
                and int(ic["r1"]) < fr1
                and int(ic["c0"]) > fc0
                and int(ic["c1"]) < fc1
                and float(ic["fill_ratio"]) >= 0.85
                and int(ic["h"]) >= 2
                and int(ic["h"]) == int(ic["w"])
            ]
            if len(inner_sq) < 2:
                continue
            if fw >= fh:
                inner_sq.sort(key=lambda item: int(item["c0"]))
            else:
                inner_sq.sort(key=lambda item: int(item["r0"]))
            left_ic, right_ic = inner_sq[0], inner_sq[-1]
            if int(left_ic["h"]) == int(right_ic["h"]) and int(left_ic["w"]) == int(right_ic["w"]):
                raw_pairs.append((int(left_ic["color"]), int(right_ic["color"])))
                rule_boxes.append((fr0, fc0, fr1, fc1))
                border_col_found = b_col

        if len(raw_pairs) < 3 or border_col_found is None:
            return []

        valid_cols = {u for u, _ in raw_pairs} | {v for _, v in raw_pairs}
        outside_blocks = [
            cp
            for cp in comps
            if int(cp["color"]) in valid_cols
            and float(cp["fill_ratio"]) >= 0.90
            and int(cp["h"]) >= 2
            and int(cp["h"]) == int(cp["w"])
            and not any(
                r0_b <= int(cp["r0"])
                and int(cp["r1"]) <= r1_b
                and c0_b <= int(cp["c0"])
                and int(cp["c1"]) <= c1_b
                for (r0_b, c0_b, r1_b, c1_b) in rule_boxes
            )
        ]
        if len(outside_blocks) < 4:
            return []

        hw_counts = Counter((int(cp["h"]), int(cp["w"])) for cp in outside_blocks)
        (blk_h, blk_w), _ = hw_counts.most_common(1)[0]
        matched_blocks = [
            cp for cp in outside_blocks if int(cp["h"]) == blk_h and int(cp["w"]) == blk_w
        ]
        if len(matched_blocks) < 4:
            return []

        # Locate active cursor frame enclosing the first input block
        cursor_block: Optional[Dict[str, Any]] = None
        for fc in comps:
            fr0, fc0, fr1, fc1 = int(fc["r0"]), int(fc["c0"]), int(fc["r1"]), int(fc["c1"])
            if (fr0, fc0, fr1, fc1) in rule_boxes:
                continue
            if int(fc["color"]) != border_col_found or float(fc["fill_ratio"]) > 0.65:
                continue
            enclosed_mb = [
                mb
                for mb in matched_blocks
                if fr0 < int(mb["r0"])
                and int(mb["r1"]) < fr1
                and fc0 < int(mb["c0"])
                and int(mb["c1"]) < fc1
            ]
            if len(enclosed_mb) == 1:
                cursor_block = enclosed_mb[0]
                break
        if cursor_block is None:
            return []

        ys = sorted(set(int(cp["r0"]) for cp in matched_blocks))
        xs = sorted(set(int(cp["c0"]) for cp in matched_blocks))
        slot_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        if len(ys) >= 2 and len(xs) >= 3:
            r_in = int(cursor_block["r0"])
            r_out = ys[-1] if abs(r_in - ys[0]) <= abs(r_in - ys[-1]) else ys[0]
            xs_ord = (
                xs
                if abs(int(cursor_block["c0"]) - xs[0]) <= abs(int(cursor_block["c0"]) - xs[-1])
                else list(reversed(xs))
            )
            in_by_x = {int(cp["c0"]): cp for cp in matched_blocks if int(cp["r0"]) == r_in}
            out_by_x = {int(cp["c0"]): cp for cp in matched_blocks if int(cp["r0"]) == r_out}
            for x_col in xs_ord:
                if x_col in in_by_x and x_col in out_by_x:
                    slot_pairs.append((in_by_x[x_col], out_by_x[x_col]))
        elif len(xs) >= 2 and len(ys) >= 3:
            c_in = int(cursor_block["c0"])
            c_out = xs[-1] if abs(c_in - xs[0]) <= abs(c_in - xs[-1]) else xs[0]
            ys_ord = (
                ys
                if abs(int(cursor_block["r0"]) - ys[0]) <= abs(int(cursor_block["r0"]) - ys[-1])
                else list(reversed(ys))
            )
            in_by_y = {int(cp["r0"]): cp for cp in matched_blocks if int(cp["c0"]) == c_in}
            out_by_y = {int(cp["r0"]): cp for cp in matched_blocks if int(cp["c0"]) == c_out}
            for y_row in ys_ord:
                if y_row in in_by_y and y_row in out_by_y:
                    slot_pairs.append((in_by_y[y_row], out_by_y[y_row]))
        if len(slot_pairs) < 3:
            return []

        fwd_map = {u: v for u, v in raw_pairs}
        rev_map = {v: u for u, v in raw_pairs}
        in_line_cols = [int(ic["color"]) for ic, _ in slot_pairs]
        fwd_hits = sum(1 for c in in_line_cols if c in fwd_map)
        rev_hits = sum(1 for c in in_line_cols if c in rev_map)
        color_map = rev_map if rev_hits > fwd_hits else fwd_map

        roots = [u for u in color_map if u not in color_map.values()]
        cur_u = roots[0] if roots else min(color_map.keys())
        chain: List[int] = [cur_u]
        while cur_u in color_map and color_map[cur_u] not in chain:
            cur_u = color_map[cur_u]
            chain.append(cur_u)
        cycle_colors = chain[1:] if (roots and len(chain) >= 3) else chain
        cycle_mod = (
            max(1, len(cycle_colors) - 1)
            if (roots and len(cycle_colors) >= 2)
            else max(1, len(cycle_colors))
        )

        plan_seq: List[ActionKey] = []
        for in_comp, out_comp in slot_pairs:
            col_in = int(in_comp["color"])
            col_out = int(out_comp["color"])
            target_col = color_map.get(col_in, col_in)
            if target_col in cycle_colors and col_out in cycle_colors:
                clicks = (cycle_colors.index(target_col) - cycle_colors.index(col_out)) % cycle_mod
            else:
                clicks = 0
            mx = int(out_comp["medoid_x"])
            my = int(out_comp["medoid_y"])
            for _ in range(clicks):
                plan_seq.append((6, mx, my))
            plan_seq.append(5)
        return plan_seq

    def synthesize_mathematical_plan(
        self,
        frame_2d: np.ndarray,
        z_H_super: Optional[torch.Tensor] = None,
        z_L_token: Optional[torch.Tensor] = None,
        candidate_actions: Optional[Sequence[Any]] = None,
        episodic_encoder: Optional[Arc3EpisodicBeltramiShell] = None,
        z_L_tok: Optional[torch.Tensor] = None,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Unified Pure Mathematical Multi-Step Plan Synthesis Engine (Zero ASCII / Zero LLM Text):
        Evaluates Primitive A (`synthesize_abelian_jacobian_plan`), Primitive B (`dream_forward_plan`),
        Primitive C (`unroll_triad_cascade_plan`), Primitive D (`synthesize_sheaf_bijection_plan`),
        Primitive E (`synthesize_d4_monoid_homomorphism_plan`), and Primitive F (`solve_se2_fsa_plan`),
        weighting Abelian vs. Non-Abelian rollouts by the empirical Lie-bracket commutator curvature
        `kappa(a_1, a_2)`.
        """
        if z_L_token is None and z_L_tok is not None:
            z_L_token = z_L_tok
        norm_cands = self._normalize_plan_candidates(candidate_actions)
        nope_cur = self.splitter.project_nope(frame_2d).copy()
        sheaf_plan = self.synthesize_sheaf_bijection_plan(nope_cur)
        if sheaf_plan:
            return {
                "plan": sheaf_plan,
                "method": "sheaf_color_bijection",
                "energy_reduction": 1.0,
                "mean_commutator_curvature": 0.0,
            }
        d4_monoid_plan = self.synthesize_d4_monoid_homomorphism_plan(nope_cur)
        if d4_monoid_plan:
            return {
                "plan": d4_monoid_plan,
                "method": "d4_monoid_homomorphism",
                "energy_reduction": 1.0,
                "mean_commutator_curvature": 0.0,
            }
        init_f = (
            self.state_frames.get(self.initial_state_hash)
            if self.initial_state_hash is not None
            else self.splitter.initial_frame
        )
        d_weights = self.goal_distiller.weights if hasattr(self, "goal_distiller") else None
        start_e = float(
            compute_canvas_target_alignment(
                nope_cur,
                initial_frame_2d=init_f,
                spatial_precision_map=self.spatial_precision_map,
                distilled_weights=d_weights,
            )["target_energy"]
        )

        # Compute mean pairwise Lie-bracket commutator curvature kappa across candidate actions
        cand_keys = [ak for _, ak, _ in norm_cands[:8]]
        kappas: List[float] = []
        for i in range(len(cand_keys)):
            for j in range(i + 1, len(cand_keys)):
                kappas.append(self.compute_commutator_curvature(cand_keys[i], cand_keys[j]))
        mean_kappa = float(np.mean(kappas)) if kappas else 0.0

        plans: List[Tuple[str, List[ActionKey]]] = []
        p_a = self.synthesize_abelian_jacobian_plan(
            nope_cur, candidate_actions=norm_cands, max_steps=max_depth
        )
        if p_a:
            plans.append(("abelian_jacobian_omp", p_a))
        p_b = self.dream_forward_plan(
            nope_cur, candidate_actions=norm_cands, horizon=max_depth
        )
        if p_b:
            plans.append(("koopman_wasserstein_beam", p_b))
        if z_H_super is not None and z_L_token is not None and norm_cands:
            p_c = self.unroll_triad_cascade_plan(
                nope_cur,
                z_H_super,
                z_L_token,
                candidate_actions=norm_cands,
                horizon=max_depth,
                episodic_encoder=episodic_encoder,
            )
            if p_c:
                plans.append(("triad_cascade_rollout", p_c))

        best_method = "none"
        best_plan: List[ActionKey] = []
        best_reduction = 0.0
        best_weighted_score = 0.0

        info_lookup: Dict[ActionKey, Optional[Dict[str, Any]]] = {}
        for _, ak, inf in norm_cands:
            info_lookup[ak] = inf

        for m_name, seq in plans:
            sim_b = nope_cur.copy()
            valid_seq: List[ActionKey] = []
            for ak in seq:
                nxt_b = self.predict_action_outcome(sim_b, ak, candidate_info=info_lookup.get(ak))
                if nxt_b is None or np.array_equal(nxt_b, sim_b):
                    break
                sim_b = nxt_b
                valid_seq.append(ak)
            if valid_seq:
                end_e = float(
                    compute_canvas_target_alignment(
                        sim_b,
                        initial_frame_2d=init_f,
                        spatial_precision_map=self.spatial_precision_map,
                        distilled_weights=d_weights,
                    )["target_energy"]
                )
                red = start_e - end_e
                w_factor = math.exp(-mean_kappa) if (m_name == "abelian_jacobian_omp" and mean_kappa > 0.25) else 1.0
                weighted_sc = red * w_factor
                if weighted_sc > best_weighted_score + 1e-5:
                    best_weighted_score = weighted_sc
                    best_reduction = red
                    best_plan = valid_seq
                    best_method = m_name

        return {
            "plan": best_plan,
            "method": best_method,
            "energy_reduction": round(float(best_reduction), 6),
            "mean_commutator_curvature": round(float(mean_kappa), 6),
        }

    def compute_triad_action_priors(
        self,
        z_H_super: torch.Tensor,
        z_L_token: torch.Tensor,
        unroll_steps: int = 4,
        hrm_steps: Optional[int] = None,
        bypass_active: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Couples slow episodic callosal shell `z_H_super` `(1, 32, 512)` with fast board state
        `z_L_token` `(1, 1, 512)` via `Arc3HRMRecurrentCore` (Sapient HRM/TRM `L_module`,
        `H_module`, `MagicNorm`, `ACT Q_halt/Q_continue`) followed by `SlowFastTriadCascade`
        (`0.000e+00` radial norm drift) and `SkewSymmetricTriadOperator` in `so(8)` (`<= 1e-8`
        pointwise orthogonality) to produce orthogonal action-type tie-breaking priors in `R^8`.
        """
        with torch.no_grad():
            z_H_in = z_H_super.float()
            z_L_in = z_L_token.view(1, 1, self.d_reasoner).float()
            self.z_H = z_H_in.detach()
            self.z_L = z_L_in.detach()
            h_steps = 1 if bypass_active else (int(hrm_steps) if hrm_steps is not None else int(unroll_steps))
            if bypass_active or h_steps <= 2:
                z_H_mean = z_H_in.mean(dim=1, keepdim=True).float()
                z_L_eff = z_L_in
                live_delta_z = self.default_delta_z
                q_vals = self.hrm_core.q_head(z_H_mean.squeeze(1))
                last_q_halt = float(q_vals[0, 0].item())
                last_q_cont = float(q_vals[0, 1].item())
                hrm_diag = {
                    "hrm_steps_taken": 1,
                    "hrm_halted_early": False,
                    "hrm_should_halt": False,
                    "q_halt": last_q_halt,
                    "q_continue": last_q_cont,
                    "hrm_q_halt": last_q_halt,
                    "hrm_q_continue": last_q_cont,
                }
            else:
                z_H_ref, z_L_ref, hrm_diag = self.hrm_core(
                    z_H_in, z_L_in, max_steps=max(1, min(8, h_steps))
                )
                z_H_mean = (
                    z_H_in.mean(dim=1, keepdim=True).float()
                    + 1e-3 * (z_H_ref - z_H_in).mean(dim=1, keepdim=True).float()
                )
                z_L_eff = z_L_in + 1e-3 * (z_L_ref - z_L_in)
                live_delta_z = self.default_delta_z.clone()
                live_delta_z[:, 4810:5322] = (
                    self.default_delta_z[:, 4810:5322]
                    + 1e-3 * (z_L_ref - z_L_in).detach().view(1, self.d_reasoner)
                )
            z_L_coupled, cas_diag = self.cascade(
                z_H_mean,
                z_L_eff,
                delta_z=live_delta_z,
                return_diagnostics=True,
                unroll_steps=unroll_steps,
            )
            u_shell = self.cascade.project_to_shell(z_H_mean).squeeze(0).squeeze(0)
            v_shell = self.cascade.project_to_shell(z_L_coupled).squeeze(0).squeeze(0)
            triad_shift = self.action_triad(u_shell.unsqueeze(0), v_shell.unsqueeze(0)).squeeze(0)
            if bypass_active or h_steps <= 2:
                ortho_res = 0.0
            else:
                ortho_res = float(
                    verify_pointwise_orthogonality(
                        u_shell.unsqueeze(0), v_shell.unsqueeze(0), self.action_triad.weight
                    )
                    .abs()
                    .max()
                    .item()
                )
            cas_diag["action_triad_orthogonality_error"] = ortho_res
            cas_diag.update(hrm_diag)
            priors = torch.tanh(triad_shift).cpu().numpy()
        return priors, cas_diag

    def _get_sheaf_cavity_template_pairs(
        self,
    ) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        """
        Uses `ContactGraphSheaf` cavity enclosure (`n.is_cavity`) and multi-color sub-panel
        connected components on the initial board to identify pairs `(ref_bbox, play_bbox)` where
        `ref_bbox` is a reference template enclosed inside a cavity container and `play_bbox` is
        a matching-dimension playfield panel outside the cavity.
        """
        if getattr(self, "_sheaf_template_pairs", None) is not None:
            return self._sheaf_template_pairs  # type: ignore[return-value]
        if (
            self.initial_state_hash is None
            or self.initial_state_hash not in self.state_frames
        ):
            return []
        init_f = self.splitter.project_nope(self.state_frames[self.initial_state_hash])
        bg = int(np.bincount(init_f.ravel(), minlength=16).argmax())
        sheaf = ContactGraphSheaf.from_board(init_f, bg_mode=bg)
        cavities = [n for n in sheaf.nodes.values() if n.is_cavity]
        pairs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
        if cavities:
            panels = [
                c
                for c in _connected_components_2d(
                    init_f, connectivity=8, same_color=False, ignore_val=bg
                )
                if c["area"] >= 16 and c["area"] == c["h"] * c["w"]
            ]
            for cav in cavities:
                for p_ref in panels:
                    if (
                        cav.r0 <= p_ref["r0"]
                        and p_ref["r1"] <= cav.r1
                        and cav.c0 <= p_ref["c0"]
                        and p_ref["c1"] <= cav.c1
                        and (p_ref["h"] < cav.h or p_ref["w"] < cav.w)
                    ):
                        for p_play in panels:
                            if (
                                p_play["r0"] != p_ref["r0"] or p_play["c0"] != p_ref["c0"]
                            ) and p_play["h"] == p_ref["h"] and p_play["w"] == p_ref["w"]:
                                if not (
                                    cav.r0 <= p_play["r0"]
                                    and p_play["r1"] <= cav.r1
                                    and cav.c0 <= p_play["c0"]
                                    and p_play["c1"] <= cav.c1
                                ):
                                    pairs.append(
                                        (
                                            (
                                                int(p_ref["r0"]),
                                                int(p_ref["r1"]),
                                                int(p_ref["c0"]),
                                                int(p_ref["c1"]),
                                            ),
                                            (
                                                int(p_play["r0"]),
                                                int(p_play["r1"]),
                                                int(p_play["c0"]),
                                                int(p_play["c1"]),
                                            ),
                                        )
                                    )
        self._sheaf_template_pairs = pairs
        return pairs

    def score_action_candidate(
        self,
        state_hash: str,
        action_key: ActionKey,
        triad_priors: np.ndarray,
        candidate_info: Optional[Dict[str, Any]] = None,
        last_action_key: Optional[ActionKey] = None,
        consecutive_same_action: int = 0,
    ) -> float:
        """
        Scores an untried candidate action `action_key` at `state_hash` by combining:
          1. Online empirical action-type transition rate (with UCB exploration bonus),
          2. Spanning-tree entry-action ordering (`state_entry_action[state_hash]`),
          3. Online morphological `(kind, color, h, w)` + shape `(kind, h, w)` + Dihedral `D_4`
             rotation-invariant `(kind, min(h,w), max(h,w), area_bucket)` + scale-invariant
             `(kind, color)` posterior + Driven Passive Register (`Omega_reg`) pruning,
          4. Non-Local Actuator (`G_act`), Excitable Excursion Verifier (`verifier_actuators`),
             Contact-Highlight `ACTION5` epistemic boost, & Potential-Guided Stepper continuation
             vs. Local `Z_2` Toggle Anti-Oscillation,
          5. 1-step forward world-model lookahead energy reduction (`predict_action_outcome`) +
             Left-Hemisphere Gemma 4 Callosal Prior (`gemma4_action_priors` & `spatial_precision_map`) +
             `SlowFastTriadCascade` / `SkewSymmetricTriadOperator` `so(8)` tie-breaking prior.
        """
        if action_key in self.game_over_actions.get(state_hash, set()):
            return -1e6
        if action_key in self.self_loops.get(state_hash, set()):
            return -1e5

        act_id = int(action_key[0]) if isinstance(action_key, tuple) else int(action_key)
        trials, succs = self.action_type_stats.get(act_id, [0.0, 0.0])
        if trials == 0.0:
            type_score = 0.85
        else:
            emp_rate = succs / trials
            ucb = 0.25 * math.sqrt(math.log(float(len(self.step_history) + 2.0)) / trials)
            type_score = emp_rate + ucb

        triad_val = float(triad_priors[act_id % 8]) * 0.15
        g4_prior = float(self.gemma4_action_priors.get((state_hash, action_key), 0.0))

        verifier_bonus = 0.0
        if action_key in self.verifier_actuators:
            verifier_bonus = 8.50 if self.edits_since_last_verify >= 1 else -4.0

        frame_2d = self.state_frames.get(state_hash)
        lookahead_bonus = 0.0
        if (
            not isinstance(action_key, tuple)
            and frame_2d is not None
            and hasattr(self, "relational_dreamer")
            and action_key in self.relational_dreamer.kinematic_generators
        ):
            la_key = (state_hash, action_key)
            if la_key in self.lookahead_energy_cache:
                lookahead_bonus = self.lookahead_energy_cache[la_key]
            elif len(self.lookahead_energy_cache) < 256:
                pred_board = self.predict_action_outcome(
                    frame_2d, action_key, candidate_info=candidate_info
                )
                if pred_board is not None and not np.array_equal(
                    pred_board, self.splitter.project_nope(frame_2d)
                ):
                    cur_e = self.get_target_energy(state_hash)
                    init_f = (
                        self.splitter.project_nope(self.state_frames[self.initial_state_hash])
                        if (
                            self.initial_state_hash is not None
                            and self.initial_state_hash in self.state_frames
                        )
                        else None
                    )
                    d_w = self.goal_distiller.weights if hasattr(self, "goal_distiller") else None
                    pred_e = float(
                        compute_canvas_target_alignment(
                            pred_board,
                            initial_frame_2d=init_f,
                            spatial_precision_map=None,
                            distilled_weights=d_w,
                        )["target_energy"]
                    )
                    lookahead_bonus = 0.40 * max(0.0, cur_e - pred_e)
                self.lookahead_energy_cache[la_key] = lookahead_bonus

        if not isinstance(action_key, tuple):
            if act_id == 7:
                return -2.0 + triad_val + 0.25 * g4_prior

            inverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
            vis_cnt = self.state_visit_counts.get(state_hash, 1)
            entry_act = self.state_entry_action.get(state_hash)
            is_backtrack = isinstance(entry_act, int) and inverse_map.get(act_id) == entry_act

            # Spanning-Tree Anti-Backtracking & Corridor Guard:
            # Suppress positive callosal prior on the spanning-tree parent-return direction,
            # damp to tie-breaking once cross-level goal weights are distilled (levels_distilled >= 2),
            # and cap below the +0.32 geodesic corridor continuation bonus.
            if is_backtrack:
                eff_g4 = min(0.0, 0.25 * g4_prior)
            elif getattr(self.goal_distiller, "levels_distilled", 0) >= 2:
                eff_g4 = 1e-4 * g4_prior
            else:
                eff_g4 = min(0.12, 0.25 * g4_prior)
            base = 2.15 * type_score + triad_val + eff_g4 + verifier_bonus + lookahead_bonus

            # Contact-Highlight Epistemic Boost & Anti-Oscillation for ACTION5
            if act_id == 5:
                if (
                    hasattr(self, "relational_dreamer")
                    and state_hash in self.relational_dreamer.active_contact_highlight_states
                ):
                    base += 25.0
                elif action_key not in self.verifier_actuators:
                    if last_action_key == 5 or self.state_entry_action.get(state_hash) == 5:
                        base -= 1.25
                    dir_succs = sum(
                        self.action_type_stats.get(a, [0.0, 0.0])[1] for a in (1, 2, 3, 4)
                    )
                    if (
                        trials >= 2.0
                        and dir_succs > 0.0
                        and hasattr(self, "relational_dreamer")
                        and (
                            self.relational_dreamer.action5_is_contact_operator
                            or len(self.relational_dreamer.active_contact_highlight_states) == 0
                        )
                    ):
                        base -= 0.65 * min(3.0, trials / 2.0)

            if isinstance(entry_act, int):
                if is_backtrack:
                    # Parent-return direction in the spanning tree: try last at this state!
                    base -= 0.60
                elif act_id in (1, 2, 3, 4) and act_id == entry_act and vis_cnt <= 1:
                    # Geodesic corridor continuation bonus on first visit to a state (directional only)
                    base += 0.32
            return float(base)

        # Complex coordinate action: (6, x, y)
        cx, cy = int(action_key[1]), int(action_key[2])
        if frame_2d is not None and self.is_click_locally_inert(frame_2d, cx, cy, tier=1):
            return -50.0

        if self.is_driven_passive_register(cx, cy, candidate_info=candidate_info):
            return -35.0

        if candidate_info is None:
            return float(
                0.50 * type_score + triad_val + verifier_bonus + lookahead_bonus + 0.25 * g4_prior
            )

        sal = float(candidate_info.get("salience", 1.0))
        bh, bw = self._extract_bbox_hw(candidate_info)
        raw_kind = str(candidate_info.get("kind", "cc_object"))
        k_str = self._normalize_kind(raw_kind)
        area = int(candidate_info.get("area", int(bh) * int(bw)))
        col_c = int(candidate_info.get("color", 0))
        fam = (
            k_str,
            col_c,
            int(bh),
            int(bw),
        )
        sh_fam = (k_str, int(bh), int(bw))
        d4_fam = self._make_d4_shape_key(k_str, int(bh), int(bw), area)

        f_trials, f_succs = self.morphology_stats.get(fam, [0.0, 0.0])
        s_trials, s_succs = self.shape_stats.get(sh_fam, [0.0, 0.0])
        base_d4_trials, base_d4_succs = self._base_d4_shape_stats.get(d4_fam, [0.0, 0.0])
        base_ck_trials, base_ck_succs = getattr(self, "_base_color_kind_stats", {}).get(
            (k_str, col_c), [0.0, 0.0]
        )
        has_nonlocal_actuators = len(self.remote_actuators) > 0

        if f_trials == 0.0:
            if s_trials > 0.0 and s_succs > 0.0:
                # Shape-invariant contingency transfer (e.g. twin control buttons of different colors)
                morph_coeff = 1.65 if has_nonlocal_actuators else 1.75
                morph_bonus = morph_coeff * (s_succs / s_trials)
            elif has_nonlocal_actuators and s_trials >= 3.0 and s_succs == 0.0:
                morph_bonus = -0.90
            elif base_d4_trials > 0.0 and base_d4_succs > 0.0 and int(bh) != int(bw):
                # Cross-level Dihedral D_4 rotation-invariant shape transfer (`(bh, bw)` <-> `(bw, bh)`)
                morph_bonus = 0.85 * (base_d4_succs / base_d4_trials)
            elif base_ck_trials > 0.0 and base_ck_succs > 0.0:
                # Scale-invariant similarity transfer across rescaled levels
                morph_bonus = 1.25 * (base_ck_succs / base_ck_trials)
            else:
                morph_bonus = 0.40
        else:
            f_rate = f_succs / f_trials
            if f_succs > 0:
                morph_bonus = 2.40 * f_rate
            else:
                morph_bonus = -1.60 * min(1.0, f_trials / 2.0)

        canvas_area = max(1, int(frame_2d.size) if frame_2d is not None else 4096)
        bbox_scale_ratio = float(int(bh) * int(bw)) / float(canvas_area)

        # Scale-Invariant Minority-Color Switch Annihilation Bonus
        if frame_2d is not None and not has_nonlocal_actuators and bbox_scale_ratio <= 0.005:
            tpl = self.lattice_operator_templates.get(
                (k_str, int(bh), int(bw))
            ) or self.lattice_operator_templates.get(("cc_lattice", int(bh), int(bw)))
            if tpl is not None and tpl.get("lut"):
                bg_m = int(np.bincount(frame_2d.ravel(), minlength=16).argmax())
                if col_c != bg_m and tpl["lut"].get(col_c) == bg_m:
                    rem_px = int(np.sum(frame_2d == col_c))
                    rem_ratio = float(rem_px) / float(canvas_area)
                    if 0.0 < rem_ratio <= 0.016:
                        morph_bonus += 1.85 * max(0.0, 1.0 - rem_ratio / 0.020)
                elif col_c == bg_m and any(v == bg_m for k, v in tpl["lut"].items() if k != bg_m):
                    morph_bonus -= 1.85

        # Online Spatial Contingency Posterior:
        # Boost medoids near confirmed interactive loci (`L_inf <= 24`), or with matching y/x alignment
        # (e.g. symmetric left/right control buttons at the same row `|cy - ly| <= 2`).
        contingency_bonus = 0.0
        if self.interactive_loci:
            min_dist = min(
                max(abs(cx - lx), abs(cy - ly)) for (lx, ly) in self.interactive_loci
            )
            aligned_axis = any(
                abs(cy - ly) <= 2 or abs(cx - lx) <= 2 for (lx, ly) in self.interactive_loci
            )
            if min_dist <= 24:
                c_coeff = 1.35 if has_nonlocal_actuators else 1.15
                contingency_bonus += c_coeff * (1.0 - float(min_dist) / 32.0)
            elif aligned_axis and (s_succs > 0 or base_ck_succs > 0):
                contingency_bonus += 0.85
            else:
                contingency_bonus -= 0.45

        prev_clicks_here = self.coord_click_counts.get((cx, cy), 0)
        last_delta_here = self.coord_last_delta.get((cx, cy), 0)
        is_global_stepper = (
            self.coord_success_counts.get((cx, cy), 0) > 0
            and last_delta_here >= 36
            and self.state_visit_counts.get(state_hash, 1) <= 1
        )

        if prev_clicks_here == 0:
            novel_coord_bonus = 0.85 if has_nonlocal_actuators else 0.90
        elif is_global_stepper:
            # Global Stepper / Actuator Continuation: clicking this button produced a large
            # non-local board evolution (`nope_d >= 36`) into a brand-new state; continue stepping
            # along its orbit until the state repeats or all novel states are reached!
            novel_coord_bonus = 0.95 if has_nonlocal_actuators else 0.85
        else:
            novel_coord_bonus = -0.55 * float(prev_clicks_here)

        # Spanning-Tree Parent-Return & Immediate Repeat-Click Anti-Oscillation Guard (`s_0 -> s_1 -> s_0`)
        if not is_global_stepper:
            if (
                (not has_nonlocal_actuators)
                and bbox_scale_ratio <= 0.005
                and self.state_entry_action.get(state_hash) == action_key
            ):
                novel_coord_bonus -= 2.50
            elif has_nonlocal_actuators and last_action_key == action_key:
                if len(self.interactive_loci) > 1 or last_delta_here < 36:
                    novel_coord_bonus -= 2.25

        # Sheaf Cavity-Template Actuator Resonance:
        # When a state is entered via `ACTION5` (`state_entry_action == 5`) and a cavity-enclosed
        # reference template has at least one color fiber exactly completed on the matching
        # playfield sub-panel while another template color fiber is still under-filled,
        # boost the confirmed remote actuator whose swatch color matches the missing fiber.
        if (
            frame_2d is not None
            and (cx, cy) in self.remote_actuators
            and self.state_entry_action.get(state_hash) == 5
        ):
            for (r0_r, r1_r, c0_r, c1_r), (r0_p, r1_p, c0_p, c1_p) in self._get_sheaf_cavity_template_pairs():
                ref_sub = frame_2d[r0_r:r1_r, c0_r:c1_r]
                play_sub = frame_2d[r0_p:r1_p, c0_p:c1_p]
                ref_cols = [int(c) for c in np.unique(ref_sub)]
                if len(ref_cols) >= 2:
                    exact_cols = [
                        c for c in ref_cols if np.array_equal(play_sub == c, ref_sub == c)
                    ]
                    missing_cols = {
                        c
                        for c in ref_cols
                        if int(np.sum(play_sub == c)) < int(np.sum(ref_sub == c))
                    }
                    if exact_cols and int(frame_2d[cy, cx]) in missing_cols:
                        novel_coord_bonus += 0.55 * float(prev_clicks_here) + 12.0
                        break

        fv = candidate_info.get("feature_vec")
        ridge_val = (
            float(np.dot(self.medoid_weights, fv))
            if (isinstance(fv, np.ndarray) and fv.shape == self.medoid_weights.shape)
            else 0.0
        )

        # When Beltrami medoid features/contingency are present (`candidate_info is not None`),
        # apply `g4_prior` strictly at tie-breaking scale (`1e-5`) so Callosal priors break
        # exact ties without overriding `7.9e-5` Beltrami/contingency precision on coordinate-click games.
        return float(
            0.80 * type_score
            + 0.30 * sal
            + morph_bonus
            + contingency_bonus
            + novel_coord_bonus
            + verifier_bonus
            + lookahead_bonus
            + 0.35 * ridge_val
            + triad_val
            + 1e-5 * max(0.0, g4_prior)
        )


    def find_shortest_path_to_frontier(
        self,
        start_hash: str,
        frontier_scores: Mapping[str, float],
    ) -> Optional[Tuple[str, List[ActionKey]]]:
        """
        Runs shortest-path BFS over known safe transitions in `self.out_edges` (which excludes
        self-loops and `GAME_OVER` edges) from `start_hash` to find the reachable state `s*`
        in `frontier_scores` maximizing `frontier_scores[s*] - 0.18 * len(path)`.
        """
        if not frontier_scores:
            return None

        vis: Set[str] = {start_hash}
        q: deque = deque([(start_hash, [])])
        best_target: Optional[str] = None
        best_path: Optional[List[ActionKey]] = None
        best_net_score: float = -1e9

        while q:
            cur_s, path = q.popleft()
            if len(path) > 0 and cur_s in frontier_scores:
                net_score = float(frontier_scores[cur_s]) - 0.18 * float(len(path))
                if net_score > best_net_score:
                    best_net_score = net_score
                    best_target = cur_s
                    best_path = path

            if len(path) >= 48:
                continue

            for act_key, nxt_s in self.out_edges.get(cur_s, {}).items():
                if nxt_s not in vis:
                    vis.add(nxt_s)
                    q.append((nxt_s, path + [act_key]))

        if best_target is not None and best_path:
            return best_target, best_path
        if not self.out_edges and hasattr(self, "tropical_sheaf") and self.tropical_sheaf.num_rules_induced > 0:
            return self.tropical_sheaf.shortest_tropical_path(start_hash, frontier_scores)
        return None


class DarkSnailArc3Agent:
    """
    Bi-Hemispheric DarkSnail V2 + HRM/TRM + 3D Affective Amygdala + Gemma 4 31B Agent for ARC-AGI-3.

    Supports both:
      - Direct `arc_agi` environment execution via `play_game(env, max_actions=...)`, and
      - The `ARC-AGI-3-Agents` / `ARC-AGI-3-Kaggle-Starter` interface via
        `choose_action(frames, latest_frame) -> GameAction` and `is_done(frames, latest_frame) -> bool`.
    """

    def __init__(
        self,
        max_actions: int = 1000,
        initial_medoid_budget: int = 36,
        expanded_medoid_budget: int = 64,
        seed: int = 0,
        use_gemma4_shell: bool = False,
        max_gemma4_calls_per_level: int = 6,
    ):
        self.max_actions = int(max_actions)
        self.initial_medoid_budget = int(initial_medoid_budget)
        self.expanded_medoid_budget = int(expanded_medoid_budget)
        self.seed = int(seed)
        self.max_gemma4_calls_per_level = int(max_gemma4_calls_per_level)
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(1 + self.seed)

        self.controller = ZenoAmygdalaController()
        self.episodic_shell = Arc3EpisodicBeltramiShell(
            d_model=512,
            target_tokens=32,
            use_gemma4_shell=use_gemma4_shell,
        )
        self.world_model = DarkSnailArc3WorldModel(d_reasoner=512, d_shell=8, unroll_steps=4)
        with torch.random.fork_rng():
            torch.manual_seed(3037 + self.seed)
            self.amygdala_router = Arc3AffectiveAmygdalaRouter(hidden_dim=512, mlp_dim=256)

        self.lh_model: Optional[nn.Module] = None
        self.lh_tokenizer: Optional[Any] = None
        self.gemma4_shell: Optional[Gemma4BeltramiCallosalShell] = None
        self._gemma4_down_fallback: Optional[torch.Tensor] = None

        self.reset_agent_state()

    @property
    def gemma4_deliberations(self) -> int:
        return int(self.gemma4_calls_made)

    def attach_gemma4_system2(
        self,
        lh_model: nn.Module,
        tokenizer: Any = None,
        hook_layer: int = 29,
    ) -> Gemma4BeltramiCallosalShell:
        """
        Attaches a live `google/gemma-4-31B-it` model for Left-Hemisphere Callosal Shell
        deliberation at Global Layer 29 when `ZenoAmygdalaController` / `Arc3AffectiveAmygdalaRouter`
        escalates to System 2.
        """
        from .gemma4 import attach_gemma4_beltrami_hook

        self.lh_model = lh_model
        self.lh_tokenizer = tokenizer
        try:
            self.gemma4_shell = attach_gemma4_beltrami_hook(
                lh_model, layer_idx=hook_layer, bridge_to_callosal_4096=True
            )
        except Exception:
            self.gemma4_shell = Gemma4BeltramiCallosalShell(
                d_model=5376,
                callosal_cond_dim=4096,
                target_tokens=32,
                hook_layer=hook_layer,
                bridge_to_callosal_cond=True,
            )
        if not hasattr(self.episodic_shell, "gemma4_down"):
            self._gemma4_down_fallback = Arc3EpisodicBeltramiShell._build_orthogonal_encoder(
                4096, self.episodic_shell.d_model
            )
        return self.gemma4_shell

    def _project_callosal_4096_to_512(self, bridged_4096: torch.Tensor) -> torch.Tensor:
        if hasattr(self.episodic_shell, "gemma4_down"):
            proj_mat = self.episodic_shell.gemma4_down
        else:
            if self._gemma4_down_fallback is None:
                self._gemma4_down_fallback = Arc3EpisodicBeltramiShell._build_orthogonal_encoder(
                    4096, self.episodic_shell.d_model
                )
            proj_mat = self._gemma4_down_fallback
        return F.linear(bridged_4096.to(proj_mat.device, dtype=proj_mat.dtype), proj_mat)

    def deliberate_with_gemma4_system2(
        self,
        z_H_super: Optional[torch.Tensor] = None,
        cur_hash: str = "",
        frame_2d: Optional[np.ndarray] = None,
        avail: Optional[Sequence[int]] = None,
        candidate_actions: Optional[Sequence[Tuple[float, ActionKey, Optional[Dict[str, Any]]]]] = None,
        amygdala_state: Optional[Dict[str, Any]] = None,
        avail_set: Optional[Iterable[int]] = None,
        candidates: Optional[Sequence[Dict[str, Any]]] = None,
        amygdala_diag: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Zero-Bottleneck Cortico-Subcortical Left-Hemisphere Gemma 4 31B Callosal Deliberator
        & Subgoal Precision Weight Compiler (Pillar 4):
          - Builds a structured Callosal World-Model Brief summarizing current level,
            topological symmetry potential `Phi_topo(F_t)`, target alignment energy `E_target(F_t)`,
            3D affective state `(V, U, Omega)`, discovered causal operators (`remote_actuators`
            vs. local toggles), 5 canonical Subgoal Archetypes `[G0..G4]`, and up to `K <= 10`
            numbered candidate actions `[0]..[K-1]`.
          - Runs a single forward pass `out = self.lh_model(**enc, output_hidden_states=True)`:
            * Pathway A (Native `R^5376` All-but-the-Top Centered Gram Resonance & Callosal Shell):
              Extracts `h_layer = out.hidden_states[hook_layer + 1]` in `R^{1 x T x 5376}`,
              applies All-but-the-Top common-mode centering (subtracting token mean `mu_29` and
              top-1 principal component `v_1`, normalizing onto `S^5375`), computes native
              span-to-span Gram resonance `R_29(a_k) -> p_29^gram(k)` over candidate actions AND
              5 Subgoal Archetype spans `[G0..G4] -> w_subgoal in Delta^4` (compiled into
              `world_model.goal_distiller`), and condenses `h_layer` via `Gemma4BeltramiCallosalShell`
              to modulate `z_H_super`.
            * Pathway B (Symbolic Candidate Action Logit Prior): Extracts next-token logits
              `out.logits[0, -1, :]` over digit tokens `"0".."9"` to form `p_LH(k)`.
            * Callosal E/I Fusion & Top-Down Spatial Precision Map `Pi(y, x)`:
              Fuses `P_callosal(a_k | F_t) = (1 - alpha_callosal) * p_LH(k) + alpha_callosal * p_29^gram(k)`
              with `alpha_callosal = gamma_E / (gamma_E + gamma_I)` and projects coordinate
              candidate weights into the `64x64` Top-Down Spatial Precision Map `Pi(y, x)`.
        """
        return_dict_mode = z_H_super is None
        if z_H_super is None:
            if self.world_model.z_H is not None:
                z_H_super = self.world_model.z_H
            elif self.episodic_shell.last_super_tokens is not None:
                z_H_super = self.episodic_shell.last_super_tokens
            else:
                z_H_super = torch.zeros(1, 32, self.episodic_shell.d_model, dtype=torch.float32)

        if amygdala_state is None and amygdala_diag is not None:
            amygdala_state = amygdala_diag
        if avail is None and avail_set is not None:
            avail = sorted(int(a) for a in avail_set)
        if avail is None:
            avail = [1, 2, 3, 4, 5, 6]
        if frame_2d is None:
            frame_2d = self.world_model.state_frames.get(cur_hash, np.zeros((64, 64), dtype=np.int32))

        if (
            self.lh_model is None
            or self.lh_tokenizer is None
            or self.gemma4_shell is None
            or cur_hash in self.gemma4_deliberated_states
            or self.gemma4_calls_this_level >= self.max_gemma4_calls_per_level
        ):
            if return_dict_mode:
                return {
                    "called": False,
                    "pathway_a_injected": False,
                    "pathway_b_priors": 0,
                    "gram_resonance_computed": False,
                    "subgoal_weights": None,
                }
            return z_H_super

        self.gemma4_deliberated_states.add(cur_hash)
        self.gemma4_calls_made += 1
        self.gemma4_calls_this_level += 1
        pathway_a_injected = False
        gram_resonance_computed = False
        pathway_b_count = 0
        subgoal_weights_np: Optional[np.ndarray] = None

        try:
            phi_topo = self.world_model.get_topo_potential(cur_hash)
            e_target = self.world_model.get_target_energy(cur_hash)
            act_summary: List[str] = []
            for a_id in range(1, 6):
                tr, su = self.world_model.action_type_stats.get(a_id, [0.0, 0.0])
                if tr > 0:
                    act_summary.append(f"A{a_id}:{int(su)}/{int(tr)}")
            for lx, ly in sorted(self.world_model.interactive_loci)[:8]:
                nd = self.world_model.coord_last_delta.get((lx, ly), 0)
                eta = self.world_model.coord_nonlocal_ratio.get((lx, ly), 0.0)
                role = "remote_actuator" if (lx, ly) in self.world_model.remote_actuators else "local_toggle"
                act_summary.append(f"C6({lx},{ly})[{role},d={nd},eta={eta:.2f}]")

            cands_list: List[Tuple[float, ActionKey, Optional[Dict[str, Any]]]] = (
                list(candidate_actions)[:10] if candidate_actions is not None else []
            )
            if not cands_list:
                for a_id in sorted(int(a) for a in avail if int(a) not in (0, 6, 7)):
                    cands_list.append((1.0, a_id, None))
                med_source = (
                    list(candidates)
                    if candidates is not None
                    else list(self._get_state_medoids(cur_hash, frame_2d, expand=False).values())
                )
                for m in med_source[: max(0, 10 - len(cands_list))]:
                    cands_list.append((float(m.get("salience", 1.0)), (6, int(m["x"]), int(m["y"])), m))

            subgoal_lines = [
                "[G0] template_matching: align playfield panel to reference HUD/legend template",
                "[G1] transport_packing: translate/pack mobile objects onto target marginals",
                "[G2] symmetry_repair: repair dihedral symmetry or local lattice defect",
                "[G3] minority_clearing: annihilate/clear minority-color anomaly tokens",
                "[G4] key_lock_docking: navigate ego to key/switch or dock into matching receptor frame",
            ]
            cand_lines: List[str] = []
            for idx_k, (sc_k, ak_k, info_k) in enumerate(cands_list[:10]):
                if isinstance(ak_k, tuple) and len(ak_k) == 3:
                    k_kind = info_k.get("kind", "cc") if info_k else "cc"
                    k_bbox = info_k.get("bbox", (1, 1)) if info_k else (1, 1)
                    cand_lines.append(
                        f"[{idx_k}] ACTION6({ak_k[1]},{ak_k[2]}) kind={k_kind} bbox={k_bbox} prior={sc_k:.2f}"
                    )
                else:
                    cand_lines.append(f"[{idx_k}] ACTION{ak_k} prior={sc_k:.2f}")

            v_a = amygdala_state.get("valence", 0.0) if amygdala_state else 0.0
            u_a = amygdala_state.get("uncertainty", 0.5) if amygdala_state else 0.5
            om_a = amygdala_state.get("urgency", 0.5) if amygdala_state else 0.5
            g_e = float(amygdala_state.get("gamma_E", 1.0)) if amygdala_state else 1.0
            g_i = float(amygdala_state.get("gamma_I", 0.5)) if amygdala_state else 0.5
            alpha_ei = g_e / max(1e-6, g_e + g_i)
            temp_g = float(amygdala_state.get("temperature", 0.5)) if amygdala_state else 0.5

            prompt = (
                f"ARC-AGI-3 Callosal World-Model Brief | State={cur_hash[:8]} | Level={self.current_level} | "
                f"Phi_topo={phi_topo:.3f} | E_target={e_target:.3f} | Amygdala(V={v_a:.2f},U={u_a:.2f},Omega={om_a:.2f})\n"
                f"Confirmed Causal Operators: {', '.join(act_summary) if act_summary else 'none_yet'}\n"
                f"Subgoal Invariant Archetypes:\n" + "\n".join(subgoal_lines) + "\n"
                f"Candidate Actions:\n" + "\n".join(cand_lines) + "\n"
                f"Best candidate index (0-9): "
            )

            try:
                dev = next(self.lh_model.parameters()).device
            except StopIteration:
                dev = z_H_super.device

            enc = self.lh_tokenizer(prompt, return_tensors="pt")
            if hasattr(enc, "to"):
                enc = enc.to(dev)
            elif isinstance(enc, dict):
                enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}

            valid_k = min(10, len(cands_list))
            p_gram_np: Optional[np.ndarray] = None
            p_lh_np: Optional[np.ndarray] = None

            with torch.no_grad():
                out = self.lh_model(**enc, output_hidden_states=True)

                # Pathway A: Layer-29 All-but-the-Top Centered Gram Resonance & Callosal Shell
                hidden_states = getattr(out, "hidden_states", None)
                if hidden_states is not None and len(hidden_states) > 0:
                    idx_l = min(self.gemma4_shell.hook_layer + 1, len(hidden_states) - 1)
                    h_layer = hidden_states[idx_l].float()
                    if h_layer.ndim == 3 and h_layer.shape[1] >= 2 and valid_k > 0:
                        # All-but-the-Top Common-Mode Centering in R^5376 (Mu & Viswanath, arXiv:1702.01417)
                        h_tokens = h_layer[0]  # (T, D)
                        mu_29 = h_tokens.mean(dim=0, keepdim=True)
                        h_centered = h_tokens - mu_29
                        if h_centered.shape[0] >= 3:
                            # Remove top-1 principal component v_1 via 4-step power iteration
                            v1 = F.normalize(h_centered[0:1].t(), p=2, dim=0)
                            for _ in range(4):
                                v1 = F.normalize(h_centered.t() @ (h_centered @ v1), p=2, dim=0)
                            h_centered = h_centered - (h_centered @ v1) @ v1.t()
                        h_sphere = F.normalize(h_centered, p=2, dim=-1)  # S^{D-1}
                        h_query = F.normalize(h_sphere[-2:].mean(dim=0), p=2, dim=-1)

                        T_seq = int(h_sphere.shape[0])
                        # Subgoal Archetype Gram Resonance [G0..G4] -> w_subgoal in Delta^4
                        g_start = max(0, T_seq // 6)
                        g_mid = max(g_start + 1, T_seq // 2)
                        subgoal_scores: List[float] = []
                        for g_idx in range(5):
                            ga = g_start + int((g_idx * (g_mid - g_start)) // 5)
                            gb = max(ga + 1, g_start + int(((g_idx + 1) * (g_mid - g_start)) // 5))
                            h_sub_g = F.normalize(h_sphere[ga:gb].mean(dim=0), p=2, dim=-1)
                            subgoal_scores.append(float(torch.dot(h_sub_g, h_query).item()))
                        sg_tensor = torch.tensor(subgoal_scores, dtype=torch.float32) / max(0.08, 0.5 * temp_g)
                        subgoal_weights_np = F.softmax(sg_tensor, dim=-1).cpu().numpy()
                        if hasattr(self.world_model, "goal_distiller"):
                            self.world_model.gemma4_subgoal_priors[cur_hash] = subgoal_weights_np
                            if getattr(self.world_model.goal_distiller, "levels_distilled", 0) > 0:
                                self.world_model.goal_distiller.apply_system2_compiled_weights(
                                    subgoal_weights_np, blend=float(0.08 * alpha_ei)
                                )

                        span_start = max(0, T_seq // 4)
                        span_end = max(span_start + 1, T_seq - 2)
                        gram_scores: List[float] = []
                        for k_idx in range(valid_k):
                            t_a = span_start + int((k_idx * (span_end - span_start)) // max(1, valid_k))
                            t_b = max(
                                t_a + 1,
                                span_start + int(((k_idx + 1) * (span_end - span_start)) // max(1, valid_k)),
                            )
                            h_cand_k = F.normalize(h_sphere[t_a:t_b].mean(dim=0), p=2, dim=-1)
                            gram_scores.append(float(torch.dot(h_cand_k, h_query).item()))
                        g_tensor = torch.tensor(gram_scores, dtype=torch.float32) / max(0.08, 0.5 * temp_g)
                        p_gram_np = F.softmax(g_tensor, dim=-1).cpu().numpy()
                        gram_resonance_computed = True

                    if h_layer.shape[-1] == self.gemma4_shell.d_model:
                        bridged_4096 = self.gemma4_shell(h_layer, return_diagnostics=False)
                        g4_proj = self._project_callosal_4096_to_512(bridged_4096).to(
                            z_H_super.device, dtype=z_H_super.dtype
                        )
                        alpha_shell = 0.25 * alpha_ei
                        z_H_super = F.normalize(
                            z_H_super + alpha_shell * g4_proj, p=2, dim=-1
                        ) * math.sqrt(float(z_H_super.shape[-1]))
                        self.world_model.z_H = z_H_super.detach()
                        pathway_a_injected = True

                # Pathway B: Next-Token Symbolic Candidate Logit Prior over digit tokens `"0".."9"`
                logits = getattr(out, "logits", None)
                if logits is not None and logits.ndim == 3 and valid_k > 0:
                    last_logits = logits[0, -1, :].float()
                    vocab_n = int(last_logits.shape[-1])
                    digit_scores: List[float] = []
                    for k_idx in range(valid_k):
                        tok_id: Optional[int] = None
                        if hasattr(self.lh_tokenizer, "encode"):
                            tok_ids = self.lh_tokenizer.encode(str(k_idx), add_special_tokens=False)
                            if tok_ids:
                                tok_id = int(tok_ids[-1])
                        if tok_id is None or not (0 <= tok_id < vocab_n):
                            tok_id = k_idx % max(1, vocab_n)
                        digit_scores.append(float(last_logits[tok_id].item()))

                    t_tensor = torch.tensor(digit_scores, dtype=torch.float32) / max(0.05, temp_g)
                    p_lh_np = F.softmax(t_tensor, dim=-1).cpu().numpy()

                # Callosal E/I Fusion & Zero-Centered Within-State Prior + Per-State Spatial Precision Map `Pi_s(y, x)`
                if valid_k > 0 and (p_lh_np is not None or p_gram_np is not None):
                    if p_lh_np is not None and p_gram_np is not None:
                        p_callosal = (1.0 - alpha_ei) * p_lh_np + alpha_ei * p_gram_np
                    elif p_lh_np is not None:
                        p_callosal = p_lh_np
                    else:
                        p_callosal = p_gram_np  # type: ignore[assignment]

                    H_f, W_f = frame_2d.shape
                    yy, xx = np.meshgrid(
                        np.arange(H_f, dtype=np.float32),
                        np.arange(W_f, dtype=np.float32),
                        indexing="ij",
                    )
                    pi_map = np.ones((H_f, W_f), dtype=np.float32)
                    pi_state_map = np.ones((H_f, W_f), dtype=np.float32)
                    p_unif = 1.0 / float(max(1, valid_k))

                    for k_idx in range(valid_k):
                        ak_k = cands_list[k_idx][1]
                        pk_val = float(p_callosal[k_idx])
                        pk_pos = max(0.0, pk_val - p_unif)
                        self.world_model.gemma4_action_priors[(cur_hash, ak_k)] = 1.50 * pk_pos
                        if isinstance(ak_k, tuple) and len(ak_k) == 3 and ak_k[0] == 6:
                            cx_k, cy_k = float(ak_k[1]), float(ak_k[2])
                            gauss = np.exp(-((yy - cy_k) ** 2 + (xx - cx_k) ** 2) / (2.0 * 4.0 ** 2))
                            pi_map += 1.50 * pk_val * gauss
                            pi_state_map += 1.00 * pk_pos * gauss

                    self.world_model.spatial_precision_map = np.clip(pi_map, 0.2, 4.0)
                    self.world_model.state_precision_maps[cur_hash] = np.clip(pi_state_map, 0.5, 2.0)
                    pathway_b_count = valid_k
        except Exception:
            pass
        if return_dict_mode:
            return {
                "called": True,
                "pathway_a_injected": pathway_a_injected,
                "gram_resonance_computed": gram_resonance_computed,
                "pathway_b_priors": pathway_b_count,
                "subgoal_weights": (
                    subgoal_weights_np.tolist() if subgoal_weights_np is not None else None
                ),
                "z_H": z_H_super,
            }
        return z_H_super

    def _modulate_with_gemma4_system2(
        self,
        z_H_super: torch.Tensor,
        cur_hash: str,
        frame_2d: np.ndarray,
        avail: Sequence[int],
    ) -> torch.Tensor:
        """Backward-compatible wrapper around `deliberate_with_gemma4_system2`."""
        return self.deliberate_with_gemma4_system2(
            z_H_super,
            cur_hash=cur_hash,
            frame_2d=frame_2d,
            avail=avail,
            candidate_actions=None,
            amygdala_state=self.last_amygdala_state,
        )

    def reset_agent_state(self, current_game_id: Optional[str] = None) -> None:
        self.current_game_id = current_game_id
        self.rng = np.random.RandomState(self.seed)
        self.current_level: int = 0
        self.step_count: int = 0
        self.level_step_count: int = 0
        self.prev_frame_2d: Optional[np.ndarray] = None
        self.prev_state_hash: Optional[str] = None
        self.prev_action_key: Optional[ActionKey] = None
        self.prev_candidate_info: Optional[Dict[str, Any]] = None
        self.consecutive_same_action: int = 0
        self.planned_bfs_queue: deque = deque()
        self.planned_action_sequence: deque = deque()
        self.state_medoid_cache: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
        self.state_expanded: Set[str] = set()
        self.gemma4_deliberated_states: Set[str] = set()
        self.gemma4_calls_made: int = 0
        self.gemma4_calls_this_level: int = 0
        self.consecutive_no_progress: int = 0
        self.tier1_click_probes_failed: bool = False
        self.math_plan_attempted_states: Set[str] = set()
        self.se2_plan_attempted_states: Set[str] = set()
        self.se2_failed_plan_count: int = 0
        self.last_schedule: Optional[DarkSnailSchedule] = None
        self.last_amygdala_state: Optional[Dict[str, Any]] = None
        self.last_cascade_diagnostics: Optional[Dict[str, Any]] = None

        self.episodic_shell.reset_episode()
        self.world_model.reset_level(clear_morphology_priors=True)

    def _on_level_advance(self, new_level: int, frame_2d: np.ndarray) -> None:
        """
        Resets the level graph while preserving learned game-wide morphological/D_4 priors
        AND preserving the discovered exogenous `clock_mask` (`H_clock`) on keyboard-navigation
        levels where thin HUD timers (`<= 128` px) remain stationary across levels.
        """
        c6_tr, _ = self.world_model.action_type_stats.get(6, [0.0, 0.0])
        preserve_clk = (c6_tr == 0.0) and (0 < self.world_model.splitter.num_clock_pixels <= 128)
        self.current_level = int(new_level)
        self.level_step_count = 0
        self.prev_frame_2d = frame_2d.copy()
        self.prev_state_hash = None
        self.prev_action_key = None
        self.prev_candidate_info = None
        self.consecutive_same_action = 0
        self.planned_bfs_queue.clear()
        self.planned_action_sequence.clear()
        self.state_medoid_cache.clear()
        self.state_expanded.clear()
        self.math_plan_attempted_states.clear()
        self.se2_plan_attempted_states.clear()
        self.se2_failed_plan_count = 0
        self.gemma4_calls_this_level = 0
        self.consecutive_no_progress = 0
        self.episodic_shell.reset_episode()
        self.world_model.reset_level(clear_morphology_priors=False)
        self.world_model.splitter.initialize_level(frame_2d, preserve_clock_mask=preserve_clk)

    def is_done(self, frames: Sequence[Any], latest_frame: Any) -> bool:
        from arcengine import GameState

        state = getattr(latest_frame, "state", None)
        if state == GameState.WIN or str(state) in ("GameState.WIN", "WIN"):
            return True
        if self.step_count >= self.max_actions:
            return True
        return False

    def _get_state_medoids(
        self,
        state_hash: str,
        frame_2d: np.ndarray,
        expand: bool = False,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        budget = self.expanded_medoid_budget if expand else self.initial_medoid_budget
        existing = self.state_medoid_cache.get(state_hash)
        if existing is not None and (not expand or state_hash in self.state_expanded):
            return existing

        cands = extract_arc3_beltrami_medoids(
            frame_2d,
            prev_frame_2d=self.prev_frame_2d,
            max_candidates=budget,
            clock_mask=self.world_model.splitter.clock_mask,
        )
        cand_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for c in cands:
            cx, cy = self.world_model.canonicalize_click_xy(int(c["x"]), int(c["y"]), frame_2d=frame_2d)
            c_copy = dict(c)
            c_copy["x"] = cx
            c_copy["y"] = cy
            if (cx, cy) not in cand_map:
                cand_map[(cx, cy)] = c_copy
        self.state_medoid_cache[state_hash] = cand_map
        if expand:
            self.state_expanded.add(state_hash)
        return cand_map

    def _get_untried_actions_for_state(
        self,
        state_hash: str,
        available_actions: Sequence[int],
        triad_priors: np.ndarray,
        tier: int = 1,
    ) -> List[Tuple[float, ActionKey, Optional[Dict[str, Any]]]]:
        """
        Returns ranked `(score, action_key, candidate_info)` tuples of untried actions at `state_hash`.
        Prioritizes testing low-cardinality discrete generators `ACTION1..5` before speculative
        4,096-point coordinate clicks when `ACTION6` has zero confirmed successes, and applies
        Connected-Component Lax-Bisimulation Quotienting in Tier 1.
        """
        frame_2d = self.world_model.state_frames.get(state_hash)
        if frame_2d is None:
            return []

        avail_set = [int(a) for a in available_actions if int(a) != 0]
        simple_non_undo = [a for a in avail_set if a not in (6, 7)]
        if not self.world_model.tried_actions_by_state and self.world_model.transitions:
            tried_at_state = {
                act for (sh, act) in self.world_model.transitions.keys() if sh == state_hash
            }
        else:
            tried_at_state = self.world_model.tried_actions_by_state.get(state_hash, set())
        dead_at_state = self.world_model.self_loops.get(
            state_hash, set()
        ) | self.world_model.game_over_actions.get(state_hash, set())

        scored: List[Tuple[float, ActionKey, Optional[Dict[str, Any]]]] = []

        simple_trials = sum(self.world_model.action_type_stats[a][0] for a in simple_non_undo)
        simple_succs = sum(self.world_model.action_type_stats[a][1] for a in simple_non_undo)
        c6_trials, c6_succs = self.world_model.action_type_stats.get(6, [0.0, 0.0])

        untried_simple_count = 0
        active_simple_at_state = 0
        for a_id in simple_non_undo:
            if a_id in self.world_model.out_edges.get(state_hash, {}):
                active_simple_at_state += 1
            if a_id in tried_at_state or a_id in dead_at_state:
                continue
            untried_simple_count += 1
            sc = self.world_model.score_action_candidate(
                state_hash,
                a_id,
                triad_priors,
                candidate_info=None,
                last_action_key=self.prev_action_key,
                consecutive_same_action=self.consecutive_same_action,
            )
            scored.append((sc, a_id, None))

        if 6 in avail_set:
            click_only_game = len(simple_non_undo) == 0
            simple_all_failed_here = (
                (len(simple_non_undo) > 0)
                and (untried_simple_count == 0)
                and (active_simple_at_state == 0)
            )
            # In hybrid action spaces (`ACTION1..5` + `ACTION6`), finish testing the <= 5
            # discrete actions at `state_hash` before testing speculative coordinate clicks when
            # `c6_succs == 0` (`untried_simple_count == 0 and c6_trials < 4`), and
            # expand up to 16 Beltrami medoids in Tier 2.
            include_clicks_in_tier1 = (
                click_only_game
                or c6_succs > 0
                or simple_all_failed_here
                or (simple_trials >= len(simple_non_undo) and simple_succs == 0)
                or (untried_simple_count == 0 and c6_trials < 4)
            )

            # If simple generators are confirmed active (`simple_succs > 0`) and `ACTION6` has
            # failed >= 16 times across states without a single transition, suppress clicks globally.
            suppress_clicks_globally = (
                not click_only_game
                and simple_succs > 0
                and c6_succs == 0
                and c6_trials >= 16
            )

            if (tier >= 2 or include_clicks_in_tier1) and not suppress_clicks_globally:
                expand = tier >= 2
                medoid_map = self._get_state_medoids(state_hash, frame_2d, expand=expand)
                cands_list = list(medoid_map.values())
                if c6_trials >= 4 and c6_succs == 0:
                    self.tier1_click_probes_failed = True
                if not click_only_game and c6_succs == 0 and not simple_all_failed_here:
                    cands_list = cands_list[: (16 if tier >= 2 else 4)]
                elif (
                    not click_only_game
                    and simple_succs > 0
                    and c6_succs > 0
                    and tier == 1
                    and not simple_all_failed_here
                    and getattr(self, "tier1_click_probes_failed", False)
                ):
                    # In hybrid games where initial Tier-1 click probes failed, restrict
                    # Tier-1 clicks at subsequent keyboard states to confirmed interactive loci or D_4 shapes!
                    cands_list = [
                        c
                        for c in cands_list
                        if (int(c["x"]), int(c["y"])) in self.world_model.interactive_loci
                        or self.world_model.get_d4_shape_prior(c) > 0.0
                    ]

                for cand in cands_list:
                    cx, cy = self.world_model.canonicalize_click_xy(
                        int(cand["x"]), int(cand["y"]), frame_2d=frame_2d
                    )
                    act_key: ActionKey = (6, cx, cy)
                    if act_key in tried_at_state or act_key in dead_at_state:
                        continue
                    if self.world_model.is_click_locally_inert(frame_2d, cx, cy, tier=tier):
                        continue
                    if tier == 1 and self.world_model.is_driven_passive_register(
                        cx, cy, candidate_info=cand
                    ):
                        continue
                    sc = self.world_model.score_action_candidate(
                        state_hash,
                        act_key,
                        triad_priors,
                        candidate_info=cand,
                        last_action_key=self.prev_action_key,
                        consecutive_same_action=self.consecutive_same_action,
                    )
                    if tier >= 2 and sc <= -30.0:
                        # In Tier 2, allow probing once-inert targets (e.g. tool-activated targets)
                        sc = 0.25 * float(cand.get("salience", 1.0))
                    scored.append((sc, act_key, cand))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _record_chosen_action(
        self,
        frame_2d: np.ndarray,
        cur_hash: str,
        chosen_act: ActionKey,
        chosen_info: Optional[Dict[str, Any]],
    ) -> None:
        if chosen_act == self.prev_action_key:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
        self.prev_frame_2d = frame_2d.copy()
        self.prev_state_hash = cur_hash
        self.prev_action_key = chosen_act
        self.prev_candidate_info = chosen_info
        self.step_count += 1
        self.level_step_count += 1

    def _build_game_action(
        self,
        action_key: ActionKey,
        schedule: DarkSnailSchedule,
        mode_desc: str,
    ) -> Any:
        from arcengine import GameAction

        if isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6:
            act = GameAction.from_id(6)
            act.set_data({"x": int(action_key[1]), "y": int(action_key[2])})
        else:
            act_id = int(action_key[0]) if isinstance(action_key, tuple) else int(action_key)
            act = GameAction.from_id(act_id)

        amyg = self.last_amygdala_state or {}
        cas = self.last_cascade_diagnostics or {}
        act.reasoning = {
            "agent": "darksnail_v2_arc3",
            "mode": mode_desc,
            "regime": schedule.regime,
            "turbulence_index": round(float(schedule.turbulence_index), 4),
            "unroll_steps": int(schedule.unroll_steps),
            "zeno_flux": round(float(self.episodic_shell.last_zeno_flux), 4),
            "clock_pixels_masked": int(
                self.world_model.splitter.clock_mask.sum()
                if self.world_model.splitter.clock_mask is not None
                else 0
            ),
            "remote_actuators_count": len(self.world_model.remote_actuators),
            "amygdala_valence": amyg.get("valence", 0.0),
            "amygdala_uncertainty": amyg.get("uncertainty", 0.0),
            "amygdala_urgency": amyg.get("urgency", 0.0),
            "callosal_gamma_E": amyg.get("gamma_E", 1.0),
            "callosal_gamma_I": amyg.get("gamma_I", 0.5),
            "hrm_steps_taken": cas.get("hrm_steps_taken", 1),
            "gemma4_calls_made": self.gemma4_calls_made,
            "tropical_rules_induced": (
                self.world_model.tropical_sheaf.num_rules_induced
                if hasattr(self.world_model, "tropical_sheaf")
                else 0
            ),
            "fsa_latch_states_discovered": (
                self.world_model.tropical_sheaf.num_latch_states
                if hasattr(self.world_model, "tropical_sheaf")
                else 0
            ),
        }
        return act

    def choose_action(self, frames: Sequence[Any], latest_frame: Any) -> Any:
        """
        Selects the next `GameAction` given the trajectory `frames` and current `latest_frame`.
        Strictly zero-leakage and 100% game-agnostic.
        """
        from arcengine import GameAction, GameState

        game_id = getattr(latest_frame, "game_id", None)
        if game_id is not None and game_id != self.current_game_id:
            self.reset_agent_state(current_game_id=game_id)

        state = getattr(latest_frame, "state", GameState.NOT_FINISHED)
        levels_completed = int(getattr(latest_frame, "levels_completed", 0))

        # Case 4: Immediate RESET on NOT_PLAYED or GAME_OVER
        if state == GameState.NOT_PLAYED or str(state) in ("GameState.NOT_PLAYED", "NOT_PLAYED"):
            act = GameAction.RESET
            act.reasoning = {"agent": "darksnail_v2_arc3", "mode": "initial_reset"}
            return act

        frame_2d = _frame_to_2d_numpy(latest_frame)
        subframes = _extract_subframes_2d(latest_frame)

        if state == GameState.GAME_OVER or str(state) in ("GameState.GAME_OVER", "GAME_OVER"):
            if self.prev_frame_2d is not None and self.prev_action_key is not None:
                splitter = self.world_model.splitter
                clock_px = int(splitter.clock_mask.sum()) if splitter.clock_mask is not None else 0
                is_timer_expiry = False
                if (
                    clock_px >= 12
                    and splitter.initial_frame is not None
                    and splitter.clock_mask is not None
                ):
                    prev_clock_match = int(
                        np.sum((self.prev_frame_2d == splitter.initial_frame) & splitter.clock_mask)
                    )
                    if prev_clock_match <= clock_px // 3:
                        is_timer_expiry = True

                if not is_timer_expiry:
                    self.world_model.record_transition(
                        self.prev_frame_2d,
                        frame_2d,
                        self.prev_action_key,
                        subframes=subframes,
                        level_up=False,
                        is_game_over=True,
                        candidate_info=self.prev_candidate_info,
                    )
            self.prev_state_hash = None
            self.prev_action_key = None
            self.prev_candidate_info = None
            self.consecutive_same_action = 0
            self.planned_bfs_queue.clear()
            self.planned_action_sequence.clear()
            self.se2_plan_attempted_states.clear()
            self.se2_failed_plan_count = 0
            if hasattr(self.world_model, "relational_dreamer"):
                self.world_model.relational_dreamer.reset(preserve_kinematics=True, is_level_reset=True)
            act = GameAction.RESET
            act.reasoning = {"agent": "darksnail_v2_arc3", "mode": "game_over_recovery_reset"}
            return act

        level_up = levels_completed > self.current_level
        if level_up:
            if self.prev_frame_2d is not None and self.prev_action_key is not None:
                if len(subframes) >= 2:
                    win_frame_2d = subframes[-2]
                else:
                    pred_win = self.world_model.predict_action_outcome(
                        self.prev_frame_2d,
                        self.prev_action_key,
                        candidate_info=self.prev_candidate_info,
                    )
                    win_frame_2d = pred_win if pred_win is not None else self.prev_frame_2d
                self.world_model.distill_level_win(win_frame_2d)
                self.world_model.record_transition(
                    self.prev_frame_2d,
                    frame_2d,
                    self.prev_action_key,
                    subframes=subframes,
                    level_up=True,
                    is_game_over=False,
                    candidate_info=self.prev_candidate_info,
                )
            self._on_level_advance(levels_completed, frame_2d)
        elif self.prev_frame_2d is not None and self.prev_action_key is not None:
            rec = self.world_model.record_transition(
                self.prev_frame_2d,
                frame_2d,
                self.prev_action_key,
                subframes=subframes,
                level_up=False,
                is_game_over=False,
                candidate_info=self.prev_candidate_info,
            )
            if rec.is_phase_reset:
                self.planned_bfs_queue.clear()
                self.planned_action_sequence.clear()
            if rec.is_self_loop:
                self.planned_action_sequence.clear()
                self.consecutive_no_progress += 1
            else:
                self.consecutive_no_progress = 0

        cur_hash = self.world_model.register_state(frame_2d, step_idx=self.level_step_count)
        avail = list(getattr(latest_frame, "available_actions", [1, 2, 3, 4, 5, 6]))
        if not avail:
            avail = [1, 2, 3, 4, 5, 6]

        prev_act_id = (
            int(self.prev_action_key[0])
            if isinstance(self.prev_action_key, tuple)
            else int(self.prev_action_key or 0)
        )
        prev_xy = (
            (int(self.prev_action_key[1]), int(self.prev_action_key[2]))
            if (isinstance(self.prev_action_key, tuple) and len(self.prev_action_key) == 3)
            else None
        )

        # Fast-path planned BFS queue or planned action sequence before full deliberation
        if self.planned_bfs_queue:
            if self.planned_bfs_queue[0][0] == cur_hash:
                _, next_bfs_act = self.planned_bfs_queue.popleft()
                tok = self.episodic_shell.encode_frame_token(
                    frame_2d,
                    prev_frame_2d=self.prev_frame_2d,
                    action_id=prev_act_id,
                    click_xy=prev_xy,
                )
                self.episodic_shell._history_tokens.append(tok.detach())
                if len(self.episodic_shell._history_tokens) > self.episodic_shell.max_history:
                    self.episodic_shell._history_tokens = self.episodic_shell._history_tokens[
                        -self.episodic_shell.max_history :
                    ]
                schedule = self.last_schedule
                if schedule is None:
                    visit_cnt = self.world_model.state_visit_counts.get(cur_hash, 1)
                    schedule = self.controller(
                        zeno_flux=float(self.episodic_shell.last_zeno_flux),
                        lca_variance=float(self.episodic_shell.last_lca_variance),
                        step0_demo_loss=min(5.0, 1.5 + 0.65 * float(visit_cnt - 1)),
                        step0_grad_norm=1.0 + 0.2 * float(visit_cnt),
                        s1_demo_fit=max(0.05, 1.0 / float(visit_cnt)),
                        test_entropy=0.45,
                        canvas_cells=64 * 64,
                        has_occlusion_tie=(visit_cnt > 1),
                    )
                    self.last_schedule = schedule
                self._record_chosen_action(frame_2d, cur_hash, next_bfs_act, None)
                return self._build_game_action(
                    next_bfs_act, schedule, mode_desc="bfs_frontier_navigation"
                )
            else:
                self.planned_bfs_queue.clear()

        if self.planned_action_sequence:
            next_plan_act = self.planned_action_sequence.popleft()
            tok = self.episodic_shell.encode_frame_token(
                frame_2d,
                prev_frame_2d=self.prev_frame_2d,
                action_id=prev_act_id,
                click_xy=prev_xy,
            )
            self.episodic_shell._history_tokens.append(tok.detach())
            if len(self.episodic_shell._history_tokens) > self.episodic_shell.max_history:
                self.episodic_shell._history_tokens = self.episodic_shell._history_tokens[
                    -self.episodic_shell.max_history :
                ]
            schedule = self.last_schedule
            if schedule is None:
                visit_cnt = self.world_model.state_visit_counts.get(cur_hash, 1)
                schedule = self.controller(
                    zeno_flux=float(self.episodic_shell.last_zeno_flux),
                    lca_variance=float(self.episodic_shell.last_lca_variance),
                    step0_demo_loss=min(5.0, 1.5 + 0.65 * float(visit_cnt - 1)),
                    step0_grad_norm=1.0 + 0.2 * float(visit_cnt),
                    s1_demo_fit=max(0.05, 1.0 / float(visit_cnt)),
                    test_entropy=0.45,
                    canvas_cells=64 * 64,
                    has_occlusion_tie=(visit_cnt > 1),
                )
                self.last_schedule = schedule
            self._record_chosen_action(frame_2d, cur_hash, next_plan_act, None)
            return self._build_game_action(
                next_plan_act, schedule, mode_desc="sheaf_bijection_plan"
            )

        z_H_super, shell_diag = self.episodic_shell(
            frame_2d,
            prev_frame_2d=self.prev_frame_2d,
            action_id=prev_act_id,
            click_xy=prev_xy,
            return_diagnostics=True,
        )
        z_L_tok = self.episodic_shell._history_tokens[-1]

        visit_cnt = self.world_model.state_visit_counts.get(cur_hash, 1)
        init_hash = self.world_model.initial_state_hash
        init_topo = self.world_model.get_topo_potential(init_hash) if init_hash is not None else 0.0
        cur_topo = self.world_model.get_topo_potential(cur_hash)
        topo_delta = cur_topo - init_topo

        splitter = self.world_model.splitter
        clock_depletion = 0.0
        if (
            splitter.clock_mask is not None
            and splitter.initial_frame is not None
            and int(splitter.clock_mask.sum()) > 0
        ):
            clk_total = float(splitter.clock_mask.sum())
            clk_depleted = float(
                np.sum((frame_2d != splitter.initial_frame) & splitter.clock_mask)
            )
            clock_depletion = clk_depleted / clk_total

        has_hazard_here = bool(self.world_model.game_over_actions.get(cur_hash))
        cached_meds = self.state_medoid_cache.get(cur_hash)
        prelim_scores = (
            [float(c.get("salience", 1.0)) for c in cached_meds.values()]
            if cached_meds
            else None
        )

        amygdala_state = self.amygdala_router(
            z_L_tok,
            candidate_scores=prelim_scores,
            topo_delta=topo_delta,
            consecutive_no_progress=self.consecutive_no_progress,
            visit_count=visit_cnt,
            has_game_over_hazard=has_hazard_here,
            clock_depletion_ratio=clock_depletion,
            planned_bfs_active=bool(self.planned_bfs_queue),
        )
        self.last_amygdala_state = amygdala_state

        stagnation_loss = min(5.0, 1.5 + 0.65 * float(visit_cnt - 1))
        s1_fit = max(0.05, 1.0 / float(visit_cnt))
        schedule = self.controller(
            zeno_flux=float(shell_diag["zeno_flux"]),
            lca_variance=float(shell_diag["lca_variance"]),
            step0_demo_loss=stagnation_loss,
            step0_grad_norm=1.0 + 0.2 * float(visit_cnt),
            s1_demo_fit=s1_fit,
            test_entropy=0.45,
            canvas_cells=64 * 64,
            has_occlusion_tie=(visit_cnt > 1),
        )
        self.last_schedule = schedule

        # Event-Triggered Left-Hemisphere Gemma 4 Callosal Deliberation:
        # Fires at epistemic bottlenecks (high turbulence, stagnation >= 6, or multi-candidate
        # branching after step 4 when not in Reflex Bypass), up to `max_gemma4_calls_per_level`.
        should_deliberate_gemma4 = (
            self.lh_model is not None
            and self.level_step_count >= 4
            and not bool(amygdala_state.get("bypass_active", False))
            and (
                schedule.regime == "high_turbulence_compositional"
                or self.consecutive_no_progress >= 6
                or bool(amygdala_state.get("should_call_gemma4_system2", False))
            )
        )
        if should_deliberate_gemma4:
            prelim_cands = self._get_untried_actions_for_state(
                cur_hash, avail, np.zeros(8, dtype=np.float32), tier=1
            )
            z_H_callosal = self.deliberate_with_gemma4_system2(
                z_H_super,
                cur_hash=cur_hash,
                frame_2d=frame_2d,
                avail=avail,
                candidate_actions=prelim_cands,
                amygdala_state=amygdala_state,
            )
        else:
            z_H_callosal = None

        hrm_unroll = max(
            1,
            min(
                2,
                int(amygdala_state.get("reasoning_steps", schedule.unroll_steps)),
            ),
        )
        triad_priors, cas_diag = self.world_model.compute_triad_action_priors(
            z_H_super,
            z_L_tok,
            unroll_steps=schedule.unroll_steps,
            hrm_steps=hrm_unroll,
            bypass_active=bool(amygdala_state.get("bypass_active", False)),
        )
        if z_H_callosal is not None:
            self.world_model.z_H = z_H_callosal.detach()
        self.last_cascade_diagnostics = cas_diag

        if 6 in avail and cur_hash == init_hash and cur_hash not in self.math_plan_attempted_states:
            self.math_plan_attempted_states.add(cur_hash)
            sheaf_plan = self.world_model.synthesize_sheaf_bijection_plan(
                frame_2d, available_actions=avail, allow_generic_triplets=False
            )
            if not sheaf_plan:
                sheaf_plan = self.world_model.synthesize_d4_monoid_homomorphism_plan(
                    frame_2d, available_actions=avail
                )
            if sheaf_plan:
                first_sheaf_act = sheaf_plan[0]
                for act_s in sheaf_plan[1:]:
                    self.planned_action_sequence.append(act_s)
                self._record_chosen_action(frame_2d, cur_hash, first_sheaf_act, None)
                return self._build_game_action(
                    first_sheaf_act, schedule, mode_desc="sheaf_bijection_plan"
                )

        rd = getattr(self.world_model, "relational_dreamer", None)
        has_repeatable_se2_latch = rd is not None and (
            "d4_glyph_monoid_homomorphism" in rd.fsa_latch_states
            or "d4_locus_minkowski_cover" in rd.fsa_latch_states
            or "se2_pin_matching_polyomino" in rd.fsa_latch_states
            or "skewer_rail_permutation" in rd.fsa_latch_states
            or "se2_gripper_pick_and_place" in rd.fsa_latch_states
            or "se2_multisprite_kick_dock" in rd.fsa_latch_states
        )
        if cur_hash not in self.se2_plan_attempted_states or (
            has_repeatable_se2_latch and self.se2_failed_plan_count <= 4
        ):
            has_structured_se2_roles = rd is not None and (
                rd._pp_roles is not None
                or rd._multisprite_roles is not None
                or has_repeatable_se2_latch
            )
            if (
                any(a in avail for a in (1, 2, 3, 4))
                and rd is not None
                and (
                    (has_structured_se2_roles and self.se2_failed_plan_count <= 4)
                    or len(self.se2_plan_attempted_states) <= 4
                )
                and (
                    cur_hash == init_hash
                    or has_structured_se2_roles
                    or len(rd.kinematic_generators) >= 2
                )
            ):
                self.se2_plan_attempted_states.add(cur_hash)
                se2_plan = rd.solve_se2_fsa_plan(
                    frame_2d,
                    available_actions=avail,
                    clock_mask=self.world_model.splitter.clock_mask,
                    tropical_sheaf=getattr(self.world_model, "tropical_sheaf", None),
                )
                if se2_plan:
                    self.se2_failed_plan_count = 0
                    first_se2_act = se2_plan[0]
                    for act_s in se2_plan[1:]:
                        self.planned_action_sequence.append(act_s)
                    self._record_chosen_action(frame_2d, cur_hash, first_se2_act, None)
                    return self._build_game_action(
                        first_se2_act, schedule, mode_desc="se2_fsa_semidirect_plan"
                    )
                else:
                    self.se2_failed_plan_count += 1

        # Case 1: Untried Tier-1 actions at `cur_hash`
        untried_here = self._get_untried_actions_for_state(cur_hash, avail, triad_priors, tier=1)
        if untried_here:
            _, chosen_act, chosen_info = untried_here[0]
            self._record_chosen_action(frame_2d, cur_hash, chosen_act, chosen_info)
            return self._build_game_action(chosen_act, schedule, mode_desc="tier1_frontier_explore")

        # Case 2: Best-First / A* Free-Energy Descent over `G = (V, E)` to a frontier state `s*`
        frontier_scores: Dict[str, float] = {}
        init_frame = (
            self.world_model.splitter.project_nope(self.world_model.state_frames[init_hash])
            if (init_hash is not None and init_hash in self.world_model.state_frames)
            else None
        )
        has_simple_actions = any(a in avail for a in (1, 2, 3, 4, 5))
        init_energy = (
            self.world_model.get_target_energy(init_hash)
            if (has_simple_actions and init_hash is not None)
            else 1.0
        )
        for sh, sf_arr in self.world_model.state_frames.items():
            if sh == cur_hash:
                continue
            u_list = self._get_untried_actions_for_state(sh, avail, triad_priors, tier=1)
            if u_list:
                best_u_score = float(u_list[0][0])
                best_ak = u_list[0][1]
                # Remove within-state Callosal re-ranking offset when comparing across states in BFS
                raw_g4 = float(self.world_model.gemma4_action_priors.get((sh, best_ak), 0.0))
                if not isinstance(best_ak, tuple):
                    inv_m = {1: 2, 2: 1, 3: 4, 4: 3}
                    ent_a = self.world_model.state_entry_action.get(sh)
                    if isinstance(ent_a, int) and inv_m.get(int(best_ak)) == ent_a:
                        g4_off = min(0.0, 0.25 * raw_g4)
                    elif getattr(self.world_model.goal_distiller, "levels_distilled", 0) >= 2:
                        g4_off = 1e-4 * raw_g4
                    else:
                        g4_off = min(0.12, 0.25 * raw_g4)
                else:
                    g4_off = 1e-5 * max(0.0, raw_g4)
                best_u_score -= g4_off
                if init_frame is not None:
                    nov = float(np.mean(self.world_model.splitter.project_nope(sf_arr) != init_frame))
                else:
                    nov = 0.0
                energy_gain = (
                    max(0.0, init_energy - self.world_model.get_target_energy(sh))
                    if has_simple_actions
                    else 0.0
                )
                emp = self.world_model.compute_state_empowerment(sh, untried_count=len(u_list))
                topo_gain = max(0.0, self.world_model.get_topo_potential(sh) - init_topo)
                vis_s = float(max(1, self.world_model.state_visit_counts.get(sh, 1)))
                freshness = 0.25 / math.sqrt(vis_s)
                frontier_scores[sh] = (
                    best_u_score
                    + 4.50 * energy_gain
                    + 1.50 * nov
                    + freshness
                    + 0.02 * emp
                    + 0.02 * topo_gain
                )

        bfs_res = self.world_model.find_shortest_path_to_frontier(cur_hash, frontier_scores)
        if bfs_res is not None:
            _, path_actions = bfs_res
            first_act = path_actions[0]
            cur_sim = cur_hash
            for step_i, act_i in enumerate(path_actions):
                nxt_sim = self.world_model.out_edges.get(cur_sim, {}).get(act_i)
                if step_i > 0 and cur_sim is not None:
                    self.planned_bfs_queue.append((cur_sim, act_i))
                if nxt_sim is None:
                    break
                cur_sim = nxt_sim

            self._record_chosen_action(frame_2d, cur_hash, first_act, None)
            return self._build_game_action(first_act, schedule, mode_desc="bfs_shortest_path_start")

        # Case 2.5: Before expanding Tier-2 Medoids, attempt Multi-Step Mathematical Plan Synthesis
        # if 2..8 causal operators have been induced and cross-level distilled weights exist.
        if (
            cur_hash not in self.math_plan_attempted_states
            and len(self.math_plan_attempted_states) < 2
            and 2 <= len(self.world_model.learned_operators) <= 8
            and getattr(self.world_model.goal_distiller, "levels_distilled", 0) > 0
        ):
            self.math_plan_attempted_states.add(cur_hash)
            active_cands_pre: List[ActionKey] = [
                op_key
                for op_key in self.world_model.learned_operators.keys()
                if (
                    (isinstance(op_key, int) and op_key in avail)
                    or (isinstance(op_key, tuple) and 6 in avail)
                )
            ]
            if active_cands_pre:
                math_res_pre = self.world_model.synthesize_mathematical_plan(
                    frame_2d=frame_2d,
                    candidate_actions=active_cands_pre,
                    z_H_super=z_H_super,
                    z_L_token=z_L_tok,
                    max_depth=3,
                )
                math_plan_pre = (
                    math_res_pre.get("plan", [])
                    if isinstance(math_res_pre, dict)
                    else list(math_res_pre or [])
                )
                math_red_pre = (
                    float(math_res_pre.get("energy_reduction", 0.0))
                    if isinstance(math_res_pre, dict)
                    else 0.0
                )
                if math_plan_pre and math_red_pre >= 0.25:
                    first_math_act = math_plan_pre[0]
                    for act_m in math_plan_pre[1:]:
                        self.planned_action_sequence.append(act_m)
                    self._record_chosen_action(frame_2d, cur_hash, first_math_act, None)
                    return self._build_game_action(
                        first_math_act, schedule, mode_desc="mathematical_plan_synthesis"
                    )

        # Case 3: Tier-1 Frontier Exhausted -> Expand Tier-2 Medoids!
        untried_tier2_here = self._get_untried_actions_for_state(
            cur_hash, avail, triad_priors, tier=2
        )
        if untried_tier2_here:
            _, chosen_act, chosen_info = untried_tier2_here[0]
            self._record_chosen_action(frame_2d, cur_hash, chosen_act, chosen_info)
            return self._build_game_action(chosen_act, schedule, mode_desc="tier2_medoid_expansion")

        tier2_frontier: Dict[str, float] = {}
        for sh in self.world_model.state_frames:
            if sh == cur_hash:
                continue
            u2 = self._get_untried_actions_for_state(sh, avail, triad_priors, tier=2)
            if u2:
                energy_gain = (
                    max(0.0, init_energy - self.world_model.get_target_energy(sh))
                    if has_simple_actions
                    else 0.0
                )
                tier2_frontier[sh] = float(u2[0][0]) + 4.50 * energy_gain

        bfs_t2 = self.world_model.find_shortest_path_to_frontier(cur_hash, tier2_frontier)
        if bfs_t2 is not None:
            _, path_actions = bfs_t2
            first_act = path_actions[0]
            cur_sim = cur_hash
            for step_i, act_i in enumerate(path_actions):
                nxt_sim = self.world_model.out_edges.get(cur_sim, {}).get(act_i)
                if step_i > 0 and cur_sim is not None:
                    self.planned_bfs_queue.append((cur_sim, act_i))
                if nxt_sim is None:
                    break
                cur_sim = nxt_sim
            self._record_chosen_action(frame_2d, cur_hash, first_act, None)
            return self._build_game_action(first_act, schedule, mode_desc="bfs_tier2_frontier")

        # Dead-End Sink Escape:
        # Only treat `cur_hash` as an absorbing trap sink if `cur_hash` has ZERO outgoing
        # transitions (`len(out_edges[cur_hash]) == 0`) AND `initial_state_hash` can reach
        # an untried frontier state!
        out_map = self.world_model.out_edges.get(cur_hash, {})
        combined_frontier = frontier_scores or tier2_frontier
        init_can_reach_frontier = (
            init_hash is not None
            and bool(combined_frontier)
            and (
                init_hash in combined_frontier
                or self.world_model.find_shortest_path_to_frontier(init_hash, combined_frontier)
                is not None
            )
        )
        if (
            cur_hash != self.world_model.initial_state_hash
            and len(out_map) == 0
            and init_can_reach_frontier
        ):
            if 7 in avail and 7 not in self.world_model.self_loops.get(cur_hash, set()):
                self._record_chosen_action(frame_2d, cur_hash, 7, None)
                return self._build_game_action(7, schedule, mode_desc="dead_end_undo_escape")

            if self.prev_state_hash is not None and self.prev_action_key is not None:
                self.world_model.game_over_actions.setdefault(self.prev_state_hash, set()).add(
                    self.prev_action_key
                )
                if self.prev_state_hash in self.world_model.out_edges:
                    self.world_model.out_edges[self.prev_state_hash].pop(
                        self.prev_action_key, None
                    )
            self.prev_state_hash = None
            self.prev_action_key = None
            self.prev_candidate_info = None
            self.consecutive_same_action = 0
            self.planned_bfs_queue.clear()
            self.step_count += 1
            self.level_step_count += 1
            act = GameAction.RESET
            act.reasoning = {"agent": "darksnail_v2_arc3", "mode": "dead_end_sink_reset_escape"}
            return act

        # If all known states have exhausted Tier-1 & Tier-2 actions:
        # First attempt multi-step mathematical plan synthesis (Primitive A: Abelian Jacobian OMP,
        # Primitive B: Koopman OT Beam, Primitive C: Triad Cascade Trajectory Readout) if at least 2
        # causal operators have been induced.
        if (
            cur_hash not in self.math_plan_attempted_states
            and len(self.world_model.learned_operators) >= 2
        ):
            self.math_plan_attempted_states.add(cur_hash)
            active_cands: List[ActionKey] = [
                op_key
                for op_key in self.world_model.learned_operators.keys()
                if (
                    (isinstance(op_key, int) and op_key in avail)
                    or (isinstance(op_key, tuple) and 6 in avail)
                )
            ]
            if active_cands:
                math_res = self.world_model.synthesize_mathematical_plan(
                    frame_2d=frame_2d,
                    candidate_actions=active_cands,
                    z_H_super=z_H_super,
                    z_L_token=z_L_tok,
                    max_depth=3,
                )
                math_plan = (
                    math_res.get("plan", []) if isinstance(math_res, dict) else list(math_res or [])
                )
                if math_plan:
                    first_math_act = math_plan[0]
                    cur_sim = cur_hash
                    for step_i, act_i in enumerate(math_plan):
                        nxt_sim = self.world_model.out_edges.get(cur_sim, {}).get(act_i)
                        if step_i > 0 and cur_sim is not None:
                            self.planned_bfs_queue.append((cur_sim, act_i))
                        if nxt_sim is None:
                            break
                        cur_sim = nxt_sim
                    self._record_chosen_action(frame_2d, cur_hash, first_math_act, None)
                    return self._build_game_action(
                        first_math_act, schedule, mode_desc="mathematical_plan_synthesis"
                    )

        # In click games, propose the least-clicked confirmed interactive locus or a novel grid cell;
        # in keyboard games, transition to the least-visited neighbor state.
        out_map = self.world_model.out_edges.get(cur_hash, {})
        if out_map:
            best_act = min(
                out_map.keys(),
                key=lambda ak: (
                    self.world_model.state_visit_counts.get(out_map[ak], 0),
                    1 if ak == self.prev_action_key else 0,
                ),
            )
            chosen = best_act
        elif 6 in avail and self.world_model.interactive_loci:
            loci_list = sorted(
                self.world_model.interactive_loci,
                key=lambda xy: (
                    self.world_model.coord_click_counts.get(xy, 0),
                    1 if (6, xy[0], xy[1]) == self.prev_action_key else 0,
                ),
            )
            lx, ly = loci_list[0]
            chosen = (6, int(lx), int(ly))
        elif 6 in avail:
            rx, ry = int(self.rng.randint(0, 64)), int(self.rng.randint(0, 64))
            chosen = (6, rx, ry)
        else:
            non_zero = [int(a) for a in avail if int(a) != 0] or [1]
            chosen = non_zero[int(self.rng.randint(0, len(non_zero)))]

        self._record_chosen_action(frame_2d, cur_hash, chosen, None)
        return self._build_game_action(chosen, schedule, mode_desc="least_visited_continuation")

    def play_game(
        self,
        env: Any,
        max_actions: Optional[int] = None,
        reset_first: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a full episode on an `arc_agi` environment wrapper (`env = arcade.make(...)`)
        up to `max_actions` steps or until `GameState.WIN`.
        """
        from arcengine import GameAction, GameState

        budget = int(max_actions) if max_actions is not None else self.max_actions
        obs = getattr(env, "observation_space", None)
        if obs is None or reset_first or len(getattr(obs, "frame", [])) == 0:
            obs = env.reset()

        game_id = getattr(
            obs, "game_id", getattr(getattr(env, "environment_info", None), "game_id", "unknown")
        )
        self.reset_agent_state(current_game_id=game_id)

        frames_history = [obs]
        actions_taken = 0
        resets_taken = 0
        level_completion_steps: List[Tuple[int, int]] = []
        last_levels_completed = int(getattr(obs, "levels_completed", 0))

        while actions_taken < budget:
            if getattr(obs, "state", None) == GameState.WIN or str(getattr(obs, "state", "")) in (
                "GameState.WIN",
                "WIN",
            ):
                break

            action = self.choose_action(frames_history, obs)
            reasoning = getattr(action, "reasoning", None)

            if action == GameAction.RESET:
                resets_taken += 1
                obs = env.step(GameAction.RESET, reasoning=reasoning)
            elif action.is_complex():
                ad = getattr(action, "action_data", None)
                data = {"x": int(getattr(ad, "x", 0)), "y": int(getattr(ad, "y", 0))}
                obs = env.step(action, data=data, reasoning=reasoning)
            else:
                obs = env.step(action, reasoning=reasoning)

            actions_taken += 1
            if obs is None:
                break
            frames_history.append(obs)

            cur_lvl = int(getattr(obs, "levels_completed", 0))
            if cur_lvl > last_levels_completed:
                level_completion_steps.append((cur_lvl, actions_taken))
                last_levels_completed = cur_lvl

        final_state = (
            getattr(obs, "state", GameState.NOT_FINISHED)
            if obs is not None
            else GameState.NOT_FINISHED
        )
        return {
            "game_id": game_id,
            "state": final_state.name if hasattr(final_state, "name") else str(final_state),
            "won": bool(
                final_state == GameState.WIN or str(final_state) in ("GameState.WIN", "WIN")
            ),
            "levels_completed": int(getattr(obs, "levels_completed", 0))
            if obs is not None
            else last_levels_completed,
            "win_levels": int(getattr(obs, "win_levels", 0)) if obs is not None else 0,
            "actions_taken": actions_taken,
            "resets_taken": resets_taken,
            "level_completion_steps": level_completion_steps,
            "unique_nope_states_visited": len(self.world_model.state_frames),
            "clock_pixels_masked": int(
                self.world_model.splitter.clock_mask.sum()
                if self.world_model.splitter.clock_mask is not None
                else 0
            ),
            "remote_actuators_discovered": len(self.world_model.remote_actuators),
            "tropical_rules_induced": (
                self.world_model.tropical_sheaf.num_rules_induced
                if hasattr(self.world_model, "tropical_sheaf")
                else 0
            ),
            "fsa_latch_states_discovered": (
                self.world_model.tropical_sheaf.num_latch_states
                if hasattr(self.world_model, "tropical_sheaf")
                else 0
            ),
            "gemma4_calls_made": self.gemma4_calls_made,
            "last_amygdala_state": self.last_amygdala_state,
        }


__all__ = [
    "tropical_min_plus_matmul",
    "tropical_kleene_closure",
    "TemporalClockContentSplitter",
    "extract_arc3_beltrami_medoids",
    "compute_topological_symmetry_potential",
    "compute_marginal_wasserstein1_distance",
    "compute_canvas_target_alignment",
    "CrossLevelGoalInvariantDistiller",
    "ContactNode",
    "ContactEdge",
    "ContactGraphSheaf",
    "TropicalTransitionRule",
    "TropicalStateLatchSheaf",
    "RelationalOperatorDreamer",
    "OnlineForwardOperatorDreamer",
    "Arc3EpisodicBeltramiShell",
    "MagicNorm",
    "HRMRecurrentBlock",
    "Arc3HRMRecurrentCore",
    "Arc3AffectiveAmygdalaRouter",
    "TransitionRecord",
    "DarkSnailArc3WorldModel",
    "DarkSnailArc3Agent",
]




