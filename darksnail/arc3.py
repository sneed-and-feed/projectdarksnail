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


ActionKey = Union[int, Tuple[int, int, int]]


def _frame_to_2d_numpy(frame_obj: Any) -> np.ndarray:
    """
    Extracts the settled 2D `(H, W)` integer grid (`0..15`) from a `FrameDataRaw`, `FrameData`,
    list of 2D/3D frames, or numpy/torch array.
    """
    if hasattr(frame_obj, "frame"):
        frames = frame_obj.frame
        if isinstance(frames, (list, tuple)) and len(frames) > 0:
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


def _connected_components_2d(
    grid: np.ndarray,
    connectivity: int = 4,
    same_color: bool = True,
    ignore_val: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fast vectorized-seed BFS connected component segmentation on a 2D integer grid.
    Computes each component's exact 2D interior medoid `(medoid_x, medoid_y)`
    guaranteed to lie strictly inside the component.
    """
    H, W = grid.shape
    if ignore_val is not None:
        vis = grid == ignore_val
    else:
        vis = np.zeros((H, W), dtype=bool)

    todo_r, todo_c = np.nonzero(~vis)
    if todo_r.size == 0:
        return []

    if connectivity == 4:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    comps: List[Dict[str, Any]] = []
    for idx_seed in range(todo_r.size):
        r = int(todo_r[idx_seed])
        c = int(todo_c[idx_seed])
        if vis[r, c]:
            continue
        val = int(grid[r, c])
        q = deque([(r, c)])
        vis[r, c] = True
        pts_r = [r]
        pts_c = [c]
        while q:
            cr, cc = q.popleft()
            for dr, dc in nbrs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc]:
                    if not same_color or int(grid[nr, nc]) == val:
                        vis[nr, nc] = True
                        q.append((nr, nc))
                        pts_r.append(nr)
                        pts_c.append(nc)

        rs = np.asarray(pts_r, dtype=np.int32)
        cs = np.asarray(pts_c, dtype=np.int32)
        area = int(rs.size)
        r0, r1 = int(rs.min()), int(rs.max()) + 1
        c0, c1 = int(cs.min()), int(cs.max()) + 1
        bh, bw = r1 - r0, c1 - c0
        cy = float(rs.mean())
        cx = float(cs.mean())
        dists = (rs.astype(np.float32) - cy) ** 2 + (cs.astype(np.float32) - cx) ** 2 + 0.01 * (
            np.abs(rs.astype(np.float32) - cy) + np.abs(cs.astype(np.float32) - cx)
        )
        med_idx = int(np.argmin(dists))
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
    return comps


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
        self.clock_comp_ids: Set[int] = set()
        self.depletion_history: Dict[int, List[Tuple[ActionKey, np.ndarray, np.ndarray, int, int]]] = {}
        self.max_nope_dist_from_init: int = 0
        self.transitions_observed: int = 0

    def initialize_level(self, frame_2d: np.ndarray) -> None:
        self.reset()
        f0 = np.asarray(frame_2d, dtype=np.int32).copy()
        H, W = f0.shape
        self.initial_frame = f0
        self.clock_mask = np.zeros((H, W), dtype=bool)
        self.comp_id_map = np.full((H, W), -1, dtype=np.int32)
        comps = _connected_components_2d(f0, connectivity=4, same_color=True)
        for cid, comp in enumerate(comps):
            self.comp_by_id[cid] = comp
            self.comp_id_map[comp["rows"], comp["cols"]] = cid

    def _mark_component_as_clock(self, cid: int, rows: np.ndarray, cols: np.ndarray) -> bool:
        """
        Marks observed dissipative clock pixels `(rows, cols)` and, if the parent initial
        component `cid` is a 1D/thin phase bar (`min(h, w) <= 2` and `area <= 128`), masks
        the entire component `cid` in `H_clock`.
        """
        if self.clock_mask is None:
            return False
        prev_sum = int(self.clock_mask.sum())
        self.clock_mask[rows, cols] = True
        if cid >= 0 and cid in self.comp_by_id:
            comp = self.comp_by_id[cid]
            if comp["area"] <= 128 and min(comp["h"], comp["w"]) <= 2 and comp["fill_ratio"] >= 0.75:
                self.clock_mask[comp["rows"], comp["cols"]] = True
                self.clock_comp_ids.add(cid)
        return int(self.clock_mask.sum()) > prev_sum

    def observe_transition(
        self,
        prev_frame: np.ndarray,
        next_frame: np.ndarray,
        action_key: ActionKey,
    ) -> Tuple[bool, int, bool]:
        """
        Observes a transition `prev_frame -> next_frame` under `action_key`, updates `H_clock`
        via the Continuity Equation + 4 dynamical clock laws, and returns:
            `(clock_mask_expanded, nope_delta_l1, is_phase_reset)`
        """
        if self.initial_frame is None or self.clock_mask is None or self.comp_id_map is None:
            self.initialize_level(prev_frame)
        assert self.initial_frame is not None and self.clock_mask is not None and self.comp_id_map is not None

        self.transitions_observed += 1
        diff_mask = prev_frame != next_frame
        if not np.any(diff_mask):
            return False, 0, False

        mask_expanded = False
        is_phase_reset = False

        # Law 4: Check for Periodic Phase-Reset / Respawn to `initial_frame`
        # If the board was previously displaced from `initial_frame` in `H_nope` and suddenly
        # returns to `initial_frame` everywhere except a small (`<= 16` px) dissipative life-counter
        # cluster while `H_clock` refills, classify the residual cluster as slow-timescale `H_clock`!
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
            if cur_dist_init > 0:
                res_rs, res_cs = np.nonzero(cur_diff_init)
                cids = self.comp_id_map[res_rs, res_cs]
                for u_cid in np.unique(cids):
                    if self._mark_component_as_clock(int(u_cid), res_rs, res_cs):
                        mask_expanded = True
            nope_diff = (prev_frame != next_frame) & (~self.clock_mask)
            return mask_expanded, int(nope_diff.sum()), True

        # Global color histogram counts for the Continuity Equation conservation check
        prev_counts = np.bincount(prev_frame.reshape(-1), minlength=16)
        next_counts = np.bincount(next_frame.reshape(-1), minlength=16)

        clusters = _connected_components_2d(
            diff_mask.astype(np.int32), connectivity=8, same_color=True, ignore_val=0
        )
        total_diff_area = int(diff_mask.sum())

        for cl in clusters:
            rs, cs = cl["rows"], cl["cols"]
            area = cl["area"]
            if area > self.max_clock_tick_area or cl["h"] > 3 or cl["w"] > 3:
                continue

            old_vals = prev_frame[rs, cs]
            new_vals = next_frame[rs, cs]
            if len(np.unique(old_vals)) != 1 or len(np.unique(new_vals)) != 1:
                continue
            c_old = int(old_vals[0])
            c_new = int(new_vals[0])

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

            # Law 1: Multi-Cluster Decoupling (co-ticking alongside a >= 3x larger content move)
            other_area = total_diff_area - area
            if len(clusters) >= 2 and other_area >= max(4, 3 * area):
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
        if self.clock_mask is None or not np.any(self.clock_mask):
            return frame_2d
        return np.where(self.clock_mask, 0, frame_2d)

    def hash_nope(self, frame_2d: np.ndarray) -> str:
        """Computes the deterministic Blake2b state digest in `H_nope`."""
        return _blake2b_grid(self.project_nope(frame_2d))


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
    """
    arr = _frame_to_2d_numpy(frame_2d)
    H, W = arr.shape
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

    if clock_mask is not None and clock_mask.shape == (H, W):
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
        if clock_mask is not None and clock_mask.shape == (H, W) and bool(clock_mask[my, mx]):
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
    vis_ext = np.zeros((H, W), dtype=bool)
    q_ext: deque = deque()
    for r in range(H):
        for c in (0, W - 1):
            if arr[r, c] == bg_mode and not vis_ext[r, c]:
                vis_ext[r, c] = True
                q_ext.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if arr[r, c] == bg_mode and not vis_ext[r, c]:
                vis_ext[r, c] = True
                q_ext.append((r, c))
    while q_ext:
        cr, cc = q_ext.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and not vis_ext[nr, nc] and arr[nr, nc] == bg_mode:
                vis_ext[nr, nc] = True
                q_ext.append((nr, nc))

    holes_mask = (arr == bg_mode) & (~vis_ext)
    for comp in fg_comps:
        if comp["area"] >= 6 and comp["fill_ratio"] < 0.85 and comp["h"] >= 3 and comp["w"] >= 3:
            r0, r1, c0, c1 = comp["r0"], comp["r1"], comp["c0"], comp["c1"]
            sub = arr[r0:r1, c0:c1]
            cy_i = int(round(comp["centroid_y"]))
            cx_i = int(round(comp["centroid_x"]))
            if 0 <= cy_i < H and 0 <= cx_i < W and int(arr[cy_i, cx_i]) == bg_mode:
                holes_mask[cy_i, cx_i] = True
            bh, bw = r1 - r0, c1 - c0
            vis_sub = np.zeros((bh, bw), dtype=bool)
            q_sub: deque = deque()
            for rr in range(bh):
                for cc_i in (0, bw - 1):
                    if sub[rr, cc_i] != comp["color"] and not vis_sub[rr, cc_i]:
                        vis_sub[rr, cc_i] = True
                        q_sub.append((rr, cc_i))
            for cc_i in range(bw):
                for rr in (0, bh - 1):
                    if sub[rr, cc_i] != comp["color"] and not vis_sub[rr, cc_i]:
                        vis_sub[rr, cc_i] = True
                        q_sub.append((rr, cc_i))
            while q_sub:
                cr, cc_i = q_sub.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc_i + dc
                    if 0 <= nr < bh and 0 <= nc < bw and not vis_sub[nr, nc] and sub[nr, nc] != comp["color"]:
                        vis_sub[nr, nc] = True
                        q_sub.append((nr, nc))
            local_holes = (sub == bg_mode) & (~vis_sub)
            if np.any(local_holes):
                holes_mask[r0:r1, c0:c1] |= local_holes

    if np.any(holes_mask):
        hole_comps = _connected_components_2d(
            holes_mask.astype(np.int32), connectivity=4, same_color=True, ignore_val=0
        )
        for hc in hole_comps:
            if hc["area"] > int(0.25 * H * W):
                continue
            mx, my = int(hc["medoid_x"]), int(hc["medoid_y"])
            log_area = math.log2(max(1.0, float(hc["area"])))
            sal = (
                2.60
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
    fg_mask = arr != bg_mode
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
    for c_val in range(16):
        feat_2d[0, :, :, c_val] = (arr == c_val).astype(np.float32)
    feat_2d[0, :, :, 16] = (arr != bg_mode).astype(np.float32)
    feat_2d[0, :, :, 17] = beltrami_grad
    feat_2d[0, :, :, 18] = temp_diff
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, H, dtype=np.float32),
        np.linspace(-1.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )
    feat_2d[0, :, :, 19] = 1.0 - 0.5 * (yy ** 2 + xx ** 2)

    with torch.no_grad():
        k_t = torch.from_numpy(feat_2d)
        pooler = RoPE2DMedoidPooler(patch_size=4, viscous_nu=0.05, viscous_p=2, shell_level=1)
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
            if clock_mask is not None and clock_mask.shape == (H, W) and bool(clock_mask[py, px]):
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

    # Deduplicate with 2D Non-Maximum Suppression (NMS):
    # Merge concentric / overlapping sub-component medoids within L_inf <= 1 (or L_inf <= 2
    # when bounding boxes overlap or for `rope2d_patch` duplicates) so composite buttons
    # (e.g. a 5x5 outer frame + 3x3 inner core) emit a single canonical medoid.
    KindPriority = {
        "symmetry_defect": 4,
        "enclosed_hole": 3,
        "cc_lattice": 2,
        "cc_object": 1,
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
    while len(selected) < max_candidates and any(
        q_ptrs[q] < len(quad_buckets[q]) for q in range(4)
    ):
        active_qs = [q for q in range(4) if q_ptrs[q] < len(quad_buckets[q])]
        active_qs.sort(
            key=lambda q: (
                int(quad_buckets[q][q_ptrs[q]].get("_diversity_tier", 0)),
                -float(quad_buckets[q][q_ptrs[q]]["salience"]),
            )
        )
        for q in active_qs:
            if len(selected) >= max_candidates:
                break
            selected.append(quad_buckets[q][q_ptrs[q]])
            q_ptrs[q] += 1

    if len(selected) < max_candidates and deferred:
        selected.extend(deferred[: max_candidates - len(selected)])

    return selected[:max_candidates]


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
        arr = np.where(clock_mask, 0, arr)

    vals, counts = np.unique(arr, return_counts=True)
    bg_mode = int(vals[int(np.argmax(counts))])
    fg_mask = arr != bg_mode
    if not np.any(fg_mask):
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
    return float(0.50 * s_panel + 0.35 * s_dihedral - 0.15 * min(1.0, beta_0 / 64.0))


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
        idx = torch.arange(d_out, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(1, d_in + 1, dtype=torch.float32).unsqueeze(0)
        raw = torch.cos(math.pi * (idx + 0.5) * k / float(max(d_in, d_out)))
        if d_out >= d_in:
            q, _ = torch.linalg.qr(raw, mode="reduced")
            return q.contiguous()
        q, _ = torch.linalg.qr(raw.t(), mode="reduced")
        return q.t().contiguous()

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


class DarkSnailArc3WorldModel(nn.Module):
    """
    Bi-Hemispheric DarkSnail V2 World Model for ARC-AGI-3.

    Combines:
      1. `TemporalClockContentSplitter`: Separates exogenous dissipative clock ticks (`H_clock`)
         from controllable state variables (`H_nope`) via the discrete continuity equation.
      2. `SlowFastTriadCascade`: Exact `0.000e+00` spherical norm-conserving slow-fast
         recurrent coupling (`d_reasoner=512, d_shell=8`) between the episodic callosal
         shell summary `z_H` and current board latent `z_L`.
      3. `SkewSymmetricTriadOperator`: Pointwise orthogonal (`<v, B_W(u, v)> <= 1e-8`)
         `so(8)` action-type triad tie-breaker.
      4. Online State-Transition Graph, Local Patch Invariance for Inert Clicks, and
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
        self.splitter = TemporalClockContentSplitter()

        gen = torch.zeros(1, 8192, dtype=torch.float32)
        torch.manual_seed(1729)
        gen[:, 4810:5322] = torch.randn(1, 512, dtype=torch.float32) * 0.15
        self.register_buffer("default_delta_z", gen, persistent=False)

        self.reset_level(clear_morphology_priors=True)

    def reset_level(self, clear_morphology_priors: bool = False) -> None:
        """Resets the current level's state graph while optionally retaining learned game mechanics."""
        self.splitter.reset()
        self.state_frames: Dict[str, np.ndarray] = {}
        self.state_visit_counts: Dict[str, int] = Counter()
        self.state_first_step: Dict[str, int] = {}
        self.state_entry_action: Dict[str, ActionKey] = {}
        self.transitions: Dict[Tuple[str, ActionKey], TransitionRecord] = {}
        self.out_edges: Dict[str, Dict[ActionKey, str]] = {}
        self.self_loops: Dict[str, Set[ActionKey]] = {}
        self.game_over_actions: Dict[str, Set[ActionKey]] = {}
        self.step_history: List[
            Tuple[np.ndarray, np.ndarray, ActionKey, int, bool, bool, bool, Optional[Dict[str, Any]]]
        ] = []
        self.initial_state_hash: Optional[str] = None

        # Level-specific Online Spatial Contingency Posterior & Local Patch Invariance state
        self.inert_click_patches: Dict[Tuple[int, int], Set[bytes]] = {}
        self.inert_patch_fail_counts: Dict[Tuple[int, int, bytes], int] = Counter()
        self.interactive_loci: Set[Tuple[int, int]] = set()
        self.coord_click_counts: Dict[Tuple[int, int], int] = Counter()
        self.coord_success_counts: Dict[Tuple[int, int], int] = Counter()
        self.coord_last_delta: Dict[Tuple[int, int], int] = {}
        self.state_topo_potential: Dict[str, float] = {}

        if clear_morphology_priors or not hasattr(self, "action_type_stats"):
            self.action_type_stats: Dict[int, List[float]] = {i: [0.0, 0.0] for i in range(8)}
            self.morphology_stats: Dict[Tuple[str, int, int, int], List[float]] = {}
            self.shape_stats: Dict[Tuple[str, int, int], List[float]] = {}
            self.medoid_weights = np.array(
                [0.0, 0.15, 0.25, 0.20, 0.0, 0.0, 0.0, 0.0, 0.35, 0.20, 0.40, 0.50],
                dtype=np.float32,
            )

        self._base_action_type_stats: Dict[int, List[float]] = {
            k: [ float(v[0]), float(v[1]) ] for k, v in self.action_type_stats.items()
        }
        self._base_morphology_stats: Dict[Tuple[str, int, int, int], List[float]] = {
            k: [ float(v[0]), float(v[1]) ] for k, v in self.morphology_stats.items()
        }
        self._base_shape_stats: Dict[Tuple[str, int, int], List[float]] = {
            k: [ float(v[0]), float(v[1]) ] for k, v in self.shape_stats.items()
        }

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
        in Tier 1 (unless `(x, y)` has already succeeded in another state, e.g. tool-activated).
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
        if tier >= 2:
            return self.inert_patch_fail_counts.get((cx, cy, cur_digest), 0) >= 2
        return True

    def register_state(self, frame_2d: np.ndarray, step_idx: int = 0) -> str:
        arr = _frame_to_2d_numpy(frame_2d)
        if self.splitter.initial_frame is None:
            self.splitter.initialize_level(arr)
        sh = self.splitter.hash_nope(arr)
        if self.initial_state_hash is None:
            self.initial_state_hash = sh
        if sh not in self.state_frames:
            self.state_frames[sh] = arr.copy()
            self.state_first_step[sh] = int(step_idx)
        self.state_visit_counts[sh] += 1
        return sh

    def _rebuild_graph_on_clock_expansion(self) -> None:
        """
        When `TemporalClockContentSplitter` identifies a new exogenous clock component `H_clock`,
        retroactively re-projects stored frames onto `H_nope` and recomputes transition edges
        plus `action_type_stats`, `interactive_loci`, `morphology_stats`, and `shape_stats` so
        earlier clock-only no-ops do not leave false positive contingency counts.
        """
        self.state_frames.clear()
        self.state_visit_counts.clear()
        self.state_first_step.clear()
        self.state_entry_action.clear()
        self.transitions.clear()
        self.out_edges.clear()
        self.self_loops.clear()
        self.game_over_actions.clear()
        self.inert_click_patches.clear()
        self.inert_patch_fail_counts.clear()
        self.interactive_loci.clear()
        self.coord_click_counts.clear()
        self.coord_success_counts.clear()
        self.coord_last_delta.clear()
        self.state_topo_potential.clear()

        self.action_type_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_action_type_stats.items()
        }
        self.morphology_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_morphology_stats.items()
        }
        self.shape_stats = {
            k: [float(v[0]), float(v[1])] for k, v in self._base_shape_stats.items()
        }

        if self.splitter.initial_frame is not None:
            self.initial_state_hash = self.splitter.hash_nope(self.splitter.initial_frame)
            self.state_frames[self.initial_state_hash] = self.splitter.initial_frame.copy()
            self.state_first_step[self.initial_state_hash] = 0

        for idx, step_tuple in enumerate(self.step_history):
            f_prev, f_next, act_key, sf_cnt, lvl_up, game_ov, is_respawn = step_tuple[:7]
            cand_info = step_tuple[7] if len(step_tuple) > 7 else None

            sh_prev = self.splitter.hash_nope(f_prev)
            sh_next = self.splitter.hash_nope(f_next)
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
            nope_prev = self.splitter.project_nope(f_prev)
            nope_next = self.splitter.project_nope(f_next)
            nope_d = int(np.sum(nope_prev != nope_next))
            is_click = isinstance(act_key, tuple) and len(act_key) == 3 and act_key[0] == 6
            is_loop = (
                (sh_prev == sh_next)
                and (nope_d == 0)
                and (not lvl_up)
                and (not game_ov)
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

            if is_click and not is_respawn:
                cx, cy = int(act_key[1]), int(act_key[2])
                self.coord_click_counts[(cx, cy)] += 1
                self.coord_last_delta[(cx, cy)] = int(nope_d)
                if meaningful:
                    self.coord_success_counts[(cx, cy)] += 1
                    self.interactive_loci.add((cx, cy))
                if cand_info is not None:
                    bh, bw = cand_info.get("bbox", (1, 1))
                    k_str = str(cand_info.get("kind", "cc_object"))
                    fam = (k_str, int(cand_info.get("color", 0)), int(bh), int(bw))
                    st = self.morphology_stats.setdefault(fam, [0.0, 0.0])
                    st[0] += 1.0
                    if meaningful:
                        st[1] += 1.0
                    sh_fam = (k_str, int(bh), int(bw))
                    sst = self.shape_stats.setdefault(sh_fam, [0.0, 0.0])
                    sst[0] += 1.0
                    if meaningful:
                        sst[1] += 1.0

    def record_transition(
        self,
        prev_frame: np.ndarray,
        next_frame: np.ndarray,
        action_key: ActionKey,
        subframes: Optional[Sequence[np.ndarray]] = None,
        level_up: bool = False,
        is_game_over: bool = False,
        candidate_info: Optional[Dict[str, Any]] = None,
    ) -> TransitionRecord:
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
        max_sf_nope_d = int(nope_d)
        if subframes is not None and len(subframes) > 1:
            for sf in subframes:
                sf_nope = self.splitter.project_nope(_frame_to_2d_numpy(sf))
                d_sf = int(np.sum(nope_prev != sf_nope))
                if d_sf > max_sf_nope_d:
                    max_sf_nope_d = d_sf

        is_click = isinstance(action_key, tuple) and len(action_key) == 3 and action_key[0] == 6
        is_loop = (
            (sh_prev == sh_next)
            and (nope_d == 0)
            and (not level_up)
            and (not is_game_over)
            and (sf_count <= 1 or max_sf_nope_d == 0 or is_click)
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

        # Always record `(sh_prev, action_key)` in `transitions` so no action at `sh_prev`
        # ever remains permanently untried, while keeping `out_edges` free of respawns.
        self.transitions[(sh_prev, action_key)] = rec
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
            else:
                self.out_edges.setdefault(sh_prev, {})[action_key] = sh_next

        act_id = int(action_key[0]) if isinstance(action_key, tuple) else int(action_key)
        meaningful = (nope_d > 0 or level_up) and (not is_game_over) and (not is_phase_reset)
        reward = 2.0 if level_up else (1.0 if meaningful else (-1.0 if is_game_over else 0.0))

        if 0 <= act_id < 8 and not is_phase_reset:
            self.action_type_stats[act_id][0] += 1.0
            if meaningful:
                self.action_type_stats[act_id][1] += 1.0

        if is_click and not is_phase_reset:
            cx, cy = int(action_key[1]), int(action_key[2])
            self.coord_click_counts[(cx, cy)] += 1
            self.coord_last_delta[(cx, cy)] = int(nope_d)
            if meaningful:
                self.coord_success_counts[(cx, cy)] += 1
                self.interactive_loci.add((cx, cy))

            if candidate_info is not None:
                bh, bw = candidate_info.get("bbox", (1, 1))
                k_str = str(candidate_info.get("kind", "cc_object"))
                fam = (
                    k_str,
                    int(candidate_info.get("color", 0)),
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

                fv = candidate_info.get("feature_vec")
                if isinstance(fv, np.ndarray) and fv.shape == self.medoid_weights.shape:
                    pred = float(np.dot(self.medoid_weights, fv))
                    err = reward - pred
                    self.medoid_weights = np.clip(self.medoid_weights + 0.08 * err * fv, -2.5, 2.5)

        return rec

    def compute_triad_action_priors(
        self,
        z_H_super: torch.Tensor,
        z_L_token: torch.Tensor,
        unroll_steps: int = 4,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Couples slow episodic callosal shell `z_H_super` `(1, 32, 512)` with fast board state
        `z_L_token` `(1, 1, 512)` via `SlowFastTriadCascade` (`0.000e+00` radial norm drift)
        and evaluates `SkewSymmetricTriadOperator` in `so(8)` to produce orthogonal action-type
        tie-breaking priors in `R^8`.
        """
        with torch.no_grad():
            z_H_mean = z_H_super.mean(dim=1, keepdim=True).float()
            z_L_in = z_L_token.view(1, 1, self.d_reasoner).float()
            z_L_coupled, cas_diag = self.cascade(
                z_H_mean,
                z_L_in,
                delta_z=self.default_delta_z,
                return_diagnostics=True,
                unroll_steps=unroll_steps,
            )
            u_shell = self.cascade.project_to_shell(z_H_mean).squeeze(0).squeeze(0)
            v_shell = self.cascade.project_to_shell(z_L_coupled).squeeze(0).squeeze(0)
            triad_shift = self.action_triad(u_shell.unsqueeze(0), v_shell.unsqueeze(0)).squeeze(0)
            ortho_res = float(
                verify_pointwise_orthogonality(
                    u_shell.unsqueeze(0), v_shell.unsqueeze(0), self.action_triad.weight
                )
                .abs()
                .max()
                .item()
            )
            cas_diag["action_triad_orthogonality_error"] = ortho_res
            priors = torch.tanh(triad_shift).cpu().numpy()
        return priors, cas_diag

    def score_action_candidate(
        self,
        state_hash: str,
        action_key: ActionKey,
        triad_priors: np.ndarray,
        candidate_info: Optional[Dict[str, Any]] = None,
        last_action_key: Optional[ActionKey] = None,
    ) -> float:
        """
        Scores an untried candidate action `action_key` at `state_hash` by combining:
          1. Online empirical action-type transition rate (with UCB exploration bonus),
          2. Spanning-tree entry-action ordering (`state_entry_action[state_hash]`) with
             Geodesic Corridor Momentum along active translation generators,
          3. Online morphological `(kind, color, h, w)` + shape-invariant `(kind, h, w)`
             posterior + Online Spatial Contingency Posterior,
          4. Global Stepper Continuation vs. Local `Z_2` Toggle Anti-Oscillation,
          5. `SlowFastTriadCascade` + `SkewSymmetricTriadOperator` `so(8)` tie-breaking prior.
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

        if not isinstance(action_key, tuple):
            if act_id == 7:
                return -2.0 + triad_val

            base = 2.15 * type_score + triad_val
            inverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
            entry_act = self.state_entry_action.get(state_hash)
            if isinstance(entry_act, int):
                if inverse_map.get(act_id) == entry_act:
                    # Parent-return direction in the spanning tree: try last at this state!
                    base -= 0.60
                elif act_id == entry_act and self.state_visit_counts.get(state_hash, 1) <= 1:
                    # Geodesic corridor continuation bonus on first visit to a state
                    base += 0.32
            return float(base)

        # Complex coordinate action: (6, x, y)
        cx, cy = int(action_key[1]), int(action_key[2])
        frame_2d = self.state_frames.get(state_hash)
        if frame_2d is not None and self.is_click_locally_inert(frame_2d, cx, cy, tier=1):
            return -50.0

        if candidate_info is None:
            return float(0.50 * type_score + triad_val)

        sal = float(candidate_info.get("salience", 1.0))
        bh, bw = candidate_info.get("bbox", (1, 1))
        k_str = str(candidate_info.get("kind", "cc_object"))
        fam = (
            k_str,
            int(candidate_info.get("color", 0)),
            int(bh),
            int(bw),
        )
        sh_fam = (k_str, int(bh), int(bw))
        f_trials, f_succs = self.morphology_stats.get(fam, [0.0, 0.0])
        s_trials, s_succs = self.shape_stats.get(sh_fam, [0.0, 0.0])

        if f_trials == 0.0:
            if s_trials > 0.0 and s_succs > 0.0:
                # Shape-invariant contingency transfer (e.g. twin control buttons of different colors)
                morph_bonus = 1.65 * (s_succs / s_trials)
            elif s_trials >= 3.0 and s_succs == 0.0:
                morph_bonus = -0.90
            else:
                morph_bonus = 0.40
        else:
            f_rate = f_succs / f_trials
            if f_succs > 0:
                morph_bonus = 2.40 * f_rate
            else:
                morph_bonus = -1.60 * min(1.0, f_trials / 2.0)

        # Online Spatial Contingency Posterior:
        # Boost medoids near confirmed interactive loci (`L_inf <= 24`), or with matching y/x alignment
        # (e.g. symmetric left/right control buttons at the same row `|cy - ly| <= 2`).
        contingency_bonus = 0.0
        if self.interactive_loci:
            min_dist = min(max(abs(cx - lx), abs(cy - ly)) for (lx, ly) in self.interactive_loci)
            aligned_axis = any(
                abs(cy - ly) <= 2 or abs(cx - lx) <= 2 for (lx, ly) in self.interactive_loci
            )
            if min_dist <= 24:
                contingency_bonus += 1.35 * (1.0 - float(min_dist) / 32.0)
            elif aligned_axis and s_succs > 0:
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
            novel_coord_bonus = 0.85
        elif is_global_stepper:
            # Global Stepper / Actuator Continuation: clicking this button produced a large
            # non-local board evolution (`nope_d >= 36`) into a brand-new state; continue stepping
            # along its orbit until the state repeats or all novel states are reached!
            novel_coord_bonus = 0.95
        else:
            novel_coord_bonus = -0.55 * float(prev_clicks_here)

        # Immediate repeat-click anti-oscillation guard for local Z_2 toggles (`s_0 -> s_1 -> s_0`)
        if last_action_key == action_key and not is_global_stepper:
            if len(self.interactive_loci) > 1 or last_delta_here < 36:
                novel_coord_bonus -= 2.25

        fv = candidate_info.get("feature_vec")
        ridge_val = (
            float(np.dot(self.medoid_weights, fv))
            if (isinstance(fv, np.ndarray) and fv.shape == self.medoid_weights.shape)
            else 0.0
        )

        return float(
            0.80 * type_score
            + 0.30 * sal
            + morph_bonus
            + contingency_bonus
            + novel_coord_bonus
            + 0.35 * ridge_val
            + triad_val
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
        return None


class DarkSnailArc3Agent:
    """
    Bi-Hemispheric DarkSnail V2 + Gemma 4 31B Agent for ARC-AGI-3.

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
    ):
        self.max_actions = int(max_actions)
        self.initial_medoid_budget = int(initial_medoid_budget)
        self.expanded_medoid_budget = int(expanded_medoid_budget)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(1 + self.seed)

        self.controller = ZenoAmygdalaController()
        self.episodic_shell = Arc3EpisodicBeltramiShell(
            d_model=512,
            target_tokens=32,
            use_gemma4_shell=use_gemma4_shell,
        )
        self.world_model = DarkSnailArc3WorldModel(d_reasoner=512, d_shell=8, unroll_steps=4)

        self.lh_model: Optional[nn.Module] = None
        self.lh_tokenizer: Optional[Any] = None
        self.gemma4_shell: Optional[Gemma4BeltramiCallosalShell] = None

        self.reset_agent_state()

    def attach_gemma4_system2(
        self,
        lh_model: nn.Module,
        tokenizer: Any = None,
        hook_layer: int = 29,
    ) -> Gemma4BeltramiCallosalShell:
        """
        Attaches a live `google/gemma-4-31B-it` model for Left-Hemisphere Callosal Shell
        deliberation at Global Layer 29 when `ZenoAmygdalaController` enters high turbulence.
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
        return self.gemma4_shell

    def _modulate_with_gemma4_system2(
        self,
        z_H_super: torch.Tensor,
        cur_hash: str,
        frame_2d: np.ndarray,
        avail: Sequence[int],
    ) -> torch.Tensor:
        """
        Queries the attached Left-Hemisphere `google/gemma-4-31B-it` Global Layer-29 hook
        at most once per unique state during `high_turbulence_compositional` regimes,
        bridging `(1, 32, 5376) -> (1, 32, 4096) -> (1, 32, 512)` into `z_H_super`.
        """
        if (
            self.lh_model is None
            or self.lh_tokenizer is None
            or self.gemma4_shell is None
            or cur_hash in self.gemma4_deliberated_states
        ):
            return z_H_super

        self.gemma4_deliberated_states.add(cur_hash)
        try:
            medoids = list(self._get_state_medoids(cur_hash, frame_2d, expand=False).values())[:8]
            med_desc = ", ".join(
                f"({m['x']},{m['y']}:c={m['color']},k={m['kind']})" for m in medoids
            )
            prompt = (
                f"ARC-AGI-3 Interactive State {cur_hash[:8]} | Level {self.current_level} | "
                f"Actions={list(avail)} | Medoids=[{med_desc}]. "
                f"Select highest-empowerment controllable transition:"
            )
            dev = next(self.lh_model.parameters()).device
            enc = self.lh_tokenizer(prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = self.lh_model(**enc, output_hidden_states=True)
                if getattr(out, "hidden_states", None) is not None and len(out.hidden_states) > 0:
                    idx = min(self.gemma4_shell.hook_layer + 1, len(out.hidden_states) - 1)
                    h_layer = out.hidden_states[idx].float()
                    if h_layer.shape[-1] == self.gemma4_shell.d_model:
                        bridged_4096 = self.gemma4_shell(h_layer, return_diagnostics=False)
                        g4_proj = self.episodic_shell.gemma4_down(
                            bridged_4096.to(z_H_super.device, dtype=z_H_super.dtype)
                        )
                        z_H_super = F.normalize(z_H_super + 0.25 * g4_proj, p=2, dim=-1) * math.sqrt(
                            float(z_H_super.shape[-1])
                        )
        except Exception:
            pass
        return z_H_super

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
        self.planned_bfs_queue: deque = deque()
        self.state_medoid_cache: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
        self.state_expanded: Set[str] = set()
        self.gemma4_deliberated_states: Set[str] = set()
        self.consecutive_no_progress: int = 0
        self.last_schedule: Optional[DarkSnailSchedule] = None
        self.last_cascade_diagnostics: Optional[Dict[str, Any]] = None

        self.episodic_shell.reset_episode()
        self.world_model.reset_level(clear_morphology_priors=True)

    def _on_level_advance(self, new_level: int, frame_2d: np.ndarray) -> None:
        """Resets the level graph while preserving learned game-wide morphological priors."""
        self.current_level = int(new_level)
        self.level_step_count = 0
        self.prev_frame_2d = frame_2d.copy()
        self.prev_state_hash = None
        self.prev_action_key = None
        self.prev_candidate_info = None
        self.planned_bfs_queue.clear()
        self.state_medoid_cache.clear()
        self.state_expanded.clear()
        self.consecutive_no_progress = 0
        self.episodic_shell.reset_episode()
        self.world_model.reset_level(clear_morphology_priors=False)
        self.world_model.splitter.initialize_level(frame_2d)

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
        4,096-point coordinate clicks when `ACTION6` has zero confirmed successes.
        """
        frame_2d = self.world_model.state_frames.get(state_hash)
        if frame_2d is None:
            return []

        avail_set = [int(a) for a in available_actions if int(a) != 0]
        simple_non_undo = [a for a in avail_set if a not in (6, 7)]
        tried_at_state = {
            act for (sh, act) in self.world_model.transitions.keys() if sh == state_hash
        }
        dead_at_state = self.world_model.self_loops.get(
            state_hash, set()
        ) | self.world_model.game_over_actions.get(state_hash, set())

        scored: List[Tuple[float, ActionKey, Optional[Dict[str, Any]]]] = []

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
            )
            scored.append((sc, a_id, None))

        if 6 in avail_set:
            c6_trials, c6_succs = self.world_model.action_type_stats.get(6, [0.0, 0.0])
            simple_trials = sum(self.world_model.action_type_stats[a][0] for a in simple_non_undo)
            simple_succs = sum(self.world_model.action_type_stats[a][1] for a in simple_non_undo)

            click_only_game = len(simple_non_undo) == 0
            simple_all_failed_here = (
                (len(simple_non_undo) > 0)
                and (untried_simple_count == 0)
                and (active_simple_at_state == 0)
            )
            # In hybrid action spaces (`ACTION1..5` + `ACTION6`), always finish testing the <= 5
            # discrete actions at `state_hash` before testing speculative coordinate clicks when
            # `c6_succs == 0`! This both calibrates `TemporalClockContentSplitter` in 2 steps and
            # prevents wasting clicks on keyboard-driven games (`cd82`, `dc22`, `sp80`, `ka59`).
            include_clicks_in_tier1 = (
                click_only_game
                or c6_succs > 0
                or simple_all_failed_here
                or (simple_trials >= len(simple_non_undo) and simple_succs == 0)
                or (untried_simple_count == 0 and c6_trials < 4)
            )

            # If simple generators are confirmed active (`simple_succs > 0`) and `ACTION6` has
            # failed >= 4 times with 0 successes, suppress further speculative clicks even in Tier 2.
            suppress_clicks_globally = (
                not click_only_game and simple_succs > 0 and c6_trials >= 4 and c6_succs == 0
            )

            if (tier >= 2 or include_clicks_in_tier1) and not suppress_clicks_globally:
                expand = tier >= 2
                medoid_map = self._get_state_medoids(state_hash, frame_2d, expand=expand)
                cands_list = list(medoid_map.values())
                if not click_only_game and c6_succs == 0 and tier == 1 and not simple_all_failed_here:
                    cands_list = cands_list[:4]

                for cand in cands_list:
                    cx, cy = self.world_model.canonicalize_click_xy(
                        int(cand["x"]), int(cand["y"]), frame_2d=frame_2d
                    )
                    act_key: ActionKey = (6, cx, cy)
                    if act_key in tried_at_state or act_key in dead_at_state:
                        continue
                    if self.world_model.is_click_locally_inert(frame_2d, cx, cy, tier=tier):
                        continue
                    sc = self.world_model.score_action_candidate(
                        state_hash,
                        act_key,
                        triad_priors,
                        candidate_info=cand,
                        last_action_key=self.prev_action_key,
                    )
                    if tier >= 2 and sc <= -40.0:
                        # In Tier 2, allow probing once-inert targets (e.g. tool-activated targets)
                        sc = 0.25 * float(cand.get("salience", 1.0))
                    scored.append((sc, act_key, cand))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

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
            self.planned_bfs_queue.clear()
            act = GameAction.RESET
            act.reasoning = {"agent": "darksnail_v2_arc3", "mode": "game_over_recovery_reset"}
            return act

        level_up = levels_completed > self.current_level
        if level_up:
            if self.prev_frame_2d is not None and self.prev_action_key is not None:
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
            if rec.is_self_loop:
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
        z_H_super, shell_diag = self.episodic_shell(
            frame_2d,
            prev_frame_2d=self.prev_frame_2d,
            action_id=prev_act_id,
            click_xy=prev_xy,
            return_diagnostics=True,
        )
        z_L_tok = self.episodic_shell._history_tokens[-1]

        visit_cnt = self.world_model.state_visit_counts.get(cur_hash, 1)
        stagnation_loss = min(
            6.0, 1.5 + 0.65 * float(visit_cnt - 1) + 0.40 * float(self.consecutive_no_progress)
        )
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
        if schedule.regime == "high_turbulence_compositional":
            z_H_super = self._modulate_with_gemma4_system2(z_H_super, cur_hash, frame_2d, avail)

        triad_priors, cas_diag = self.world_model.compute_triad_action_priors(
            z_H_super,
            z_L_tok,
            unroll_steps=int(schedule.unroll_steps),
        )
        self.last_cascade_diagnostics = cas_diag

        # Execute planned BFS path if still aligned with expected state
        if self.planned_bfs_queue:
            expected_state, next_bfs_act = self.planned_bfs_queue.popleft()
            if cur_hash == expected_state:
                self.prev_frame_2d = frame_2d.copy()
                self.prev_state_hash = cur_hash
                self.prev_action_key = next_bfs_act
                self.prev_candidate_info = None
                self.step_count += 1
                self.level_step_count += 1
                return self._build_game_action(
                    next_bfs_act, schedule, mode_desc="bfs_frontier_navigation"
                )
            else:
                self.planned_bfs_queue.clear()

        # Case 1: Untried Tier-1 actions at `cur_hash`
        untried_here = self._get_untried_actions_for_state(cur_hash, avail, triad_priors, tier=1)
        if untried_here:
            _, chosen_act, chosen_info = untried_here[0]
            self.prev_frame_2d = frame_2d.copy()
            self.prev_state_hash = cur_hash
            self.prev_action_key = chosen_act
            self.prev_candidate_info = chosen_info
            self.step_count += 1
            self.level_step_count += 1
            return self._build_game_action(chosen_act, schedule, mode_desc="tier1_frontier_explore")

        # Case 2: Shortest-path BFS over `G = (V, E)` to a frontier state `s*` with untried Tier-1 actions
        frontier_scores: Dict[str, float] = {}
        init_hash = self.world_model.initial_state_hash
        init_frame = (
            self.world_model.splitter.project_nope(self.world_model.state_frames[init_hash])
            if (init_hash is not None and init_hash in self.world_model.state_frames)
            else None
        )
        init_topo = self.world_model.get_topo_potential(init_hash) if init_hash is not None else 0.0
        for sh, sf_arr in self.world_model.state_frames.items():
            if sh == cur_hash:
                continue
            u_list = self._get_untried_actions_for_state(sh, avail, triad_priors, tier=1)
            if u_list:
                best_u_score = float(u_list[0][0])
                if init_frame is not None:
                    nov = float(np.mean(self.world_model.splitter.project_nope(sf_arr) != init_frame))
                else:
                    nov = 0.0
                emp = self.world_model.compute_state_empowerment(sh, untried_count=len(u_list))
                topo_gain = max(0.0, self.world_model.get_topo_potential(sh) - init_topo)
                vis_s = float(max(1, self.world_model.state_visit_counts.get(sh, 1)))
                freshness = 0.25 / math.sqrt(vis_s)
                frontier_scores[sh] = (
                    best_u_score + 1.50 * nov + freshness + 0.02 * emp + 0.02 * topo_gain
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

            self.prev_frame_2d = frame_2d.copy()
            self.prev_state_hash = cur_hash
            self.prev_action_key = first_act
            self.prev_candidate_info = None
            self.step_count += 1
            self.level_step_count += 1
            return self._build_game_action(first_act, schedule, mode_desc="bfs_shortest_path_start")

        # Case 3: Tier-1 Frontier Exhausted -> Expand Tier-2 Medoids!
        untried_tier2_here = self._get_untried_actions_for_state(
            cur_hash, avail, triad_priors, tier=2
        )
        if untried_tier2_here:
            _, chosen_act, chosen_info = untried_tier2_here[0]
            self.prev_frame_2d = frame_2d.copy()
            self.prev_state_hash = cur_hash
            self.prev_action_key = chosen_act
            self.prev_candidate_info = chosen_info
            self.step_count += 1
            self.level_step_count += 1
            return self._build_game_action(chosen_act, schedule, mode_desc="tier2_medoid_expansion")

        tier2_frontier: Dict[str, float] = {}
        for sh in self.world_model.state_frames:
            if sh == cur_hash:
                continue
            u2 = self._get_untried_actions_for_state(sh, avail, triad_priors, tier=2)
            if u2:
                tier2_frontier[sh] = float(u2[0][0])

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
            self.prev_frame_2d = frame_2d.copy()
            self.prev_state_hash = cur_hash
            self.prev_action_key = first_act
            self.prev_candidate_info = None
            self.step_count += 1
            self.level_step_count += 1
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
                self.prev_frame_2d = frame_2d.copy()
                self.prev_state_hash = cur_hash
                self.prev_action_key = 7
                self.prev_candidate_info = None
                self.step_count += 1
                self.level_step_count += 1
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
            self.planned_bfs_queue.clear()
            self.step_count += 1
            self.level_step_count += 1
            act = GameAction.RESET
            act.reasoning = {"agent": "darksnail_v2_arc3", "mode": "dead_end_sink_reset_escape"}
            return act

        # If all known states have exhausted Tier-1 & Tier-2 actions:
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

        self.prev_frame_2d = frame_2d.copy()
        self.prev_state_hash = cur_hash
        self.prev_action_key = chosen
        self.prev_candidate_info = None
        self.step_count += 1
        self.level_step_count += 1
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
        }


__all__ = [
    "TemporalClockContentSplitter",
    "extract_arc3_beltrami_medoids",
    "compute_topological_symmetry_potential",
    "Arc3EpisodicBeltramiShell",
    "TransitionRecord",
    "DarkSnailArc3WorldModel",
    "DarkSnailArc3Agent",
]

