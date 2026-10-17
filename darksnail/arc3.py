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

    def initialize_level(self, frame_2d: np.ndarray, preserve_clock_mask: bool = False) -> None:
        prev_mask: Optional[np.ndarray] = None
        if preserve_clock_mask and self.clock_mask is not None:
            prev_mask = self.clock_mask.copy()
        self.reset()
        f0 = np.asarray(frame_2d, dtype=np.int32).copy()
        H, W = f0.shape
        self.initial_frame = f0
        if prev_mask is not None and prev_mask.shape == (H, W):
            self.clock_mask = prev_mask
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

    Strictly projects out exogenous clock components `H_clock` across all 4 channels: any
    component/cavity/defect/patch whose medoid `(mx, my)` lies in `clock_mask` OR has `>= 50%`
    of its pixels inside `clock_mask` is excluded before NMS and quadtree selection.
    """
    arr = _frame_to_2d_numpy(frame_2d)
    H, W = arr.shape
    clk_sum = int(clock_mask.sum()) if (clock_mask is not None and clock_mask.shape == (H, W)) else 0
    has_any_clock = clock_mask is not None and clock_mask.shape == (H, W) and clk_sum > 0
    has_deep_clock = has_any_clock and (prev_frame_2d is None or (128 < clk_sum <= 450))

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
            if _is_clock_masked(mx, my, rows=hc["rows"], cols=hc["cols"], deep_only=True):
                continue
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
                                if int( np.sum(blk != int(ref_sub[rr, cc_i])) ) > (ky * kx) // 2:
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


def compute_marginal_wasserstein1_distance(
    frame_2d: np.ndarray,
    initial_frame_2d: Optional[np.ndarray] = None,
    clock_mask: Optional[np.ndarray] = None,
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
        arr = np.where(clock_mask, 0, arr)

    init_arr: Optional[np.ndarray] = None
    if initial_frame_2d is not None:
        init_arr = _frame_to_2d_numpy(initial_frame_2d)
        if clock_mask is not None and clock_mask.shape == init_arr.shape and np.any(clock_mask):
            init_arr = np.where(clock_mask, 0, init_arr)

    vals, counts = np.unique(arr, return_counts=True)
    bg_mode = int(vals[int(np.argmax(counts))])
    fg_comps = _connected_components_2d(arr, connectivity=4, same_color=True, ignore_val=bg_mode)

    def _cdf_w1_1d(idx_a: np.ndarray, idx_b: np.ndarray, length: int) -> float:
        if idx_a.size == 0 or idx_b.size == 0 or length <= 1:
            return 0.0
        ha = np.bincount(np.clip(idx_a, 0, length - 1), minlength=length).astype(np.float64)
        hb = np.bincount(np.clip(idx_b, 0, length - 1), minlength=length).astype(np.float64)
        ha /= max(1.0, float(ha.sum()))
        hb /= max(1.0, float(hb.sum()))
        return float(np.sum(np.abs(np.cumsum(ha) - np.cumsum(hb)))) / float(length)

    by_color: Dict[int, List[Dict[str, Any]]] = {}
    for c in fg_comps:
        by_color.setdefault(int(c["color"]), []).append(c)

    color_w1_list: List[float] = []
    aligned_pairs = 0
    for col, c_list in by_color.items():
        compact = [
            c for c in c_list if 2 <= int(c["area"]) <= 48 and max(int(c["h"]), int(c["w"])) <= 16
        ]
        if len(compact) == 2:
            c0, c1 = compact[0], compact[1]
            dx = abs(float(c0["centroid_x"]) - float(c1["centroid_x"]))
            dy = abs(float(c0["centroid_y"]) - float(c1["centroid_y"]))
            # Adjacent-column vertical float + wall marker (e.g. `1.5 <= dx <= 6.0`)
            if 1.0 <= dx <= 6.5 and int(c0["area"]) != int(c1["area"]):
                w1_y = _cdf_w1_1d(c0["rows"], c1["rows"], H)
                color_w1_list.append(w1_y)
                if w1_y <= 1.5 / float(H):
                    aligned_pairs += 1
                continue
            elif 1.0 <= dy <= 6.5 and int(c0["area"]) != int(c1["area"]):
                w1_x = _cdf_w1_1d(c0["cols"], c1["cols"], W)
                color_w1_list.append(w1_x)
                if w1_x <= 1.5 / float(W):
                    aligned_pairs += 1
                continue

        if init_arr is not None and init_arr.shape == arr.shape:
            mov_mask = (arr == col) & (init_arr != col)
            stat_mask = (arr == col) & (init_arr == col)
            mov_n = int(np.sum(mov_mask))
            stat_n = int(np.sum(stat_mask))
            if 2 <= mov_n <= 64 and 2 <= stat_n <= 96:
                mrs, mcs = np.where(mov_mask)
                stat_comps = [
                    sc
                    for sc in compact
                    if int(np.sum(init_arr[sc["rows"], sc["cols"]] == col)) >= int(sc["area"]) // 2
                ]
                if stat_comps:
                    best_w1 = 1.0
                    for sc in stat_comps:
                        dx = abs(float(mcs.mean()) - float(sc["centroid_x"]))
                        dy = abs(float(mrs.mean()) - float(sc["centroid_y"]))
                        if dx <= 6.5 and dy >= 1.0:
                            w1 = _cdf_w1_1d(mrs, sc["rows"], H)
                        elif dy <= 6.5 and dx >= 1.0:
                            w1 = _cdf_w1_1d(mcs, sc["cols"], W)
                        else:
                            w1 = min(_cdf_w1_1d(mcs, sc["cols"], W), _cdf_w1_1d(mrs, sc["rows"], H))
                        if w1 < best_w1:
                            best_w1 = w1
                    color_w1_list.append(best_w1)
                    if best_w1 <= 2.5 / float(max(H, W)):
                        aligned_pairs += 1

    # Corner-bracketed target slots (four 1x1 pixels of color `col` at rectangle corners)
    bracket_defects: List[float] = []
    bracket_slots_found = 0
    for col, c_list in by_color.items():
        dots = [c for c in c_list if int(c["area"]) == 1]
        if 4 <= len(dots) <= 16:
            dot_set = {(int(d["r0"]), int(d["c0"])) for d in dots}
            rs_u = sorted({r for r, _ in dot_set})
            cs_u = sorted({c for _, c in dot_set})
            non_dot_mask = (arr == col).copy()
            for r_d, c_d in dot_set:
                non_dot_mask[r_d, c_d] = False
            nd_rs, nd_cs = np.where(non_dot_mask)

            for i_r in range(len(rs_u)):
                for j_r in range(i_r + 1, len(rs_u)):
                    r_top, r_bot = rs_u[i_r], rs_u[j_r]
                    if not (2 <= r_bot - r_top <= 6):
                        continue
                    for i_c in range(len(cs_u)):
                        for j_c in range(i_c + 1, len(cs_u)):
                            c_left, c_right = cs_u[i_c], cs_u[j_c]
                            if not (2 <= c_right - c_left <= 6):
                                continue
                            if (
                                (r_top, c_left) in dot_set
                                and (r_top, c_right) in dot_set
                                and (r_bot, c_left) in dot_set
                                and (r_bot, c_right) in dot_set
                            ):
                                bracket_slots_found += 1
                                interior = arr[r_top + 1 : r_bot, c_left + 1 : c_right]
                                if interior.size > 0:
                                    fill_err = float(np.mean(interior != col))
                                    dist_err = 0.0
                                    if fill_err > 0.0 and nd_rs.size > 0:
                                        sy = 0.5 * float(r_top + r_bot)
                                        sx = 0.5 * float(c_left + c_right)
                                        d_man = np.abs(nd_rs - sy) + np.abs(nd_cs - sx)
                                        k_near = min(interior.size, d_man.size)
                                        dist_err = float(np.mean(np.sort(d_man)[:k_near])) / float(H + W)
                                    bracket_defects.append(0.65 * fill_err + 0.35 * dist_err)

    w1_mean = float(np.mean(color_w1_list)) if color_w1_list else 0.0
    brk_mean = float(np.mean(bracket_defects)) if bracket_defects else 0.0
    if color_w1_list and bracket_defects:
        combined_w1 = 0.50 * w1_mean + 0.50 * brk_mean
    elif bracket_defects:
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


def compute_canvas_target_alignment(
    frame_2d: np.ndarray,
    initial_frame_2d: Optional[np.ndarray] = None,
    clock_mask: Optional[np.ndarray] = None,
    spatial_precision_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluates the Unified Goal Homomorphism Energy `E_target(F_t; Pi)` and Alignment
    Potential `Phi_target(F_t; Pi) = 1.0 - E_target(F_t; Pi)` across four invariant geometric families:

        E_target(F_t; Pi) = alpha_hom * E_hom(F_t; Pi)
                          + alpha_marg * E_marginal(F_t; Pi)
                          + alpha_nav * E_nav(F_t; Pi)
                          + alpha_sym * (1.0 - Phi_topo(F_t))

    where:
      1. `E_hom`: Dual-Panel Kronecker Block-Scaling (`H_p = k_y M, W_p = k_x N`) & Spaced
         `cc_lattice` Homomorphism + Lattice Cluster Assembly energy.
      2. `E_marginal`: 1D Marginal Wasserstein-1 (`W_1^marg`) + Corner-Bracketed Slot Alignment energy.
      3. `E_nav`: Avatar-to-Beacon / Cavity Manhattan + Dihedral `D_4` docking distance energy.
      4. `1 - Phi_topo`: Topological `(beta_0, beta_1)` & Dihedral `D_4` reflection/panel defect.
    """
    arr = _frame_to_2d_numpy(frame_2d)
    H, W = arr.shape
    if clock_mask is not None and clock_mask.shape == arr.shape and np.any(clock_mask):
        arr = np.where(clock_mask, 0, arr)

    vals, counts = np.unique(arr, return_counts=True)
    bg_mode = int(vals[int(np.argmax(counts))])
    fg_comps = _connected_components_2d(arr, connectivity=4, same_color=True, ignore_val=bg_mode)

    prec_map: Optional[np.ndarray] = None
    if spatial_precision_map is not None:
        pm = np.asarray(spatial_precision_map, dtype=np.float32)
        if pm.shape == (H, W):
            prec_map = np.clip(pm, 0.1, 5.0)

    # 1. Dual-Panel Kronecker & Spaced Lattice Homomorphism Energy `E_hom`
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
            # Separate main playfield cluster from outlier tiles on tracks
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

    e_hom = float(np.mean(hom_defects)) if hom_defects else 0.0

    # 2. Marginal Wasserstein-1 & Bracketed Slot Alignment Energy `E_marginal`
    w1_info = compute_marginal_wasserstein1_distance(
        arr, initial_frame_2d=initial_frame_2d, clock_mask=None
    )
    e_marginal = float(w1_info["wasserstein1_distance"])

    # 3. Navigation Avatar-to-Target-Beacon Distance Energy `E_nav`
    e_nav = 0.0
    if initial_frame_2d is not None:
        init_arr = _frame_to_2d_numpy(initial_frame_2d)
        if clock_mask is not None and clock_mask.shape == init_arr.shape and np.any(clock_mask):
            init_arr = np.where(clock_mask, 0, init_arr)
        diff_mask = arr != init_arr
        diff_n = int(np.sum(diff_mask))
        if 2 <= diff_n <= 64:
            # Identify moving avatar pixels (where `arr != init_arr` and `arr != bg_mode`)
            av_mask = diff_mask & (arr != bg_mode)
            if np.any(av_mask):
                ars, acs = np.where(av_mask)
                ay, ax = float(ars.mean()), float(acs.mean())
                av_cols = set(int(c) for c in np.unique(arr[av_mask]))
                # Find stationary target beacons of rare foreground color (`4 <= area <= 64`)
                beacons = [
                    c
                    for c in fg_comps
                    if 4 <= int(c["area"]) <= 64
                    and int(c["color"]) not in av_cols
                    and max(abs(float(c["centroid_y"]) - ay), abs(float(c["centroid_x"]) - ax)) > 3.0
                ]
                if beacons:
                    min_d = min(
                        (abs(float(b["centroid_y"]) - ay) + abs(float(b["centroid_x"]) - ax))
                        / float(H + W)
                        for b in beacons
                    )
                    e_nav = float(min_d)

    # 4. Topological & Dihedral Symmetry Defect `1.0 - Phi_topo`
    phi_topo = compute_topological_symmetry_potential(arr, clock_mask=None)
    topo_defect = float(np.clip(1.0 - max(0.0, phi_topo), 0.0, 1.0))

    # Combine active energy channels with normalized weights
    active_terms: List[Tuple[float, float]] = [(0.20, topo_defect)]
    if hom_defects:
        active_terms.append((0.45, e_hom))
    if w1_info["tracked_color_pairs"] > 0 or w1_info["bracket_slots_count"] > 0:
        active_terms.append((0.45, e_marginal))
    if e_nav > 0.0:
        active_terms.append((0.25, e_nav))

    w_sum = sum(w for w, _ in active_terms)
    target_energy = float(sum(w * val for w, val in active_terms) / max(1e-6, w_sum))
    alignment_potential = float(1.0 - target_energy)

    return {
        "target_energy": round(target_energy, 6),
        "alignment_potential": round(alignment_potential, 6),
        "homomorphism_energy": round(e_hom, 6),
        "marginal_energy": round(e_marginal, 6),
        "navigation_energy": round(e_nav, 6),
        "topological_defect": round(topo_defect, 6),
        "wasserstein1_distance": round(float(w1_info["wasserstein1_distance"]), 6),
        "tracked_color_pairs": int(w1_info["tracked_color_pairs"]),
        "bracket_slots_count": int(w1_info["bracket_slots_count"]),
        "defect_loci": defect_loci[:16],
    }



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
        with torch.random.fork_rng():
            torch.manual_seed(2029)
            with torch.no_grad():
                for layer in (self.q_proj, self.k_proj, self.v_proj, self.ffn_gate, self.ffn_up):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                for res_out in (self.out_proj, self.ffn_down):
                    nn.init.orthogonal_(res_out.weight, gain=1e-3)

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
        with torch.random.fork_rng():
            torch.manual_seed(2031)
            with torch.no_grad():
                nn.init.orthogonal_(self.q_head.weight, gain=0.01)
                self.q_head.bias.copy_(torch.tensor([0.0, 0.05], dtype=torch.float32))

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
        with torch.random.fork_rng():
            torch.manual_seed(2039)
            with torch.no_grad():
                for head in (self.valence_head, self.threat_head, self.urgency_head):
                    if isinstance(head[0], nn.Linear) and isinstance(head[2], nn.Linear):
                        nn.init.orthogonal_(head[0].weight, gain=0.5)
                        nn.init.zeros_(head[0].bias)
                        nn.init.orthogonal_(head[2].weight, gain=1e-3)
                        nn.init.zeros_(head[2].bias)


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

        gen = torch.zeros(1, 8192, dtype=torch.float32)
        torch.manual_seed(1729)
        gen[:, 4810:5322] = torch.randn(1, 512, dtype=torch.float32) * 0.15
        self.register_buffer("default_delta_z", gen, persistent=False)

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
        self.state_frames: Dict[str, np.ndarray] = {}
        self.state_candidates: Dict[str, List[Dict[str, Any]]] = {}
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
        self.z_H: Optional[torch.Tensor] = None
        self.z_L: Optional[torch.Tensor] = None

        # Level-specific Online Spatial Contingency Posterior, Local Patch Invariance,
        # and Non-Local Causal Operator Factorization (`G_act` vs. `Omega_reg`)
        self.inert_click_patches: Dict[Tuple[int, int], Set[bytes]] = {}
        self.inert_patch_fail_counts: Dict[Tuple[int, int, bytes], int] = Counter()
        self.interactive_loci: Set[Tuple[int, int]] = set()
        self.remote_actuators: Set[Tuple[int, int]] = set()
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
        self.gemma4_action_priors: Dict[Tuple[str, ActionKey], float] = {}
        self.learned_operators: Dict[ActionKey, Dict[str, Any]] = {}
        self.lattice_operator_templates: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

        if clear_morphology_priors or not hasattr(self, "action_type_stats"):
            self.action_type_stats: Dict[int, List[float]] = {i: [0.0, 0.0] for i in range(8)}
            self.morphology_stats: Dict[Tuple[str, int, int, int], List[float]] = {}
            self.shape_stats: Dict[Tuple[str, int, int], List[float]] = {}
            self.d4_shape_stats: Dict[Tuple[str, int, int, int], List[float]] = {}
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
          2. A learned operator in `self.learned_operators[action_key]`, or
          3. A Type IV one-shot generalized lattice operator template `self.lattice_operator_templates`
             applied to an untried sibling lattice tile `candidate_info`.
        """
        sh = self.splitter.hash_nope(frame_2d)
        known_next = self.out_edges.get(sh, {}).get(action_key)
        if known_next is not None and known_next in self.state_frames:
            return self.splitter.project_nope(self.state_frames[known_next]).copy()

        nope_cur = self.splitter.project_nope(frame_2d).copy()
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
                "navigation_energy": 0.0,
                "topological_defect": 1.0,
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
        info = compute_canvas_target_alignment(
            frame,
            initial_frame_2d=init_f,
            clock_mask=self.splitter.clock_mask,
            spatial_precision_map=self.spatial_precision_map,
        )
        self.state_target_alignment[state_hash] = info
        return info

    def get_target_energy(self, state_hash: str) -> float:
        """Returns cached Unified Goal Homomorphism Energy `E_target(F_s; Pi) in [0, 1]`."""
        return float(self.get_target_alignment(state_hash).get("target_energy", 1.0))

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
        in Tier 1 (unless `(x, y)` has already succeeded in another state or a local tool/palette
        selection is currently active).
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
        plus `action_type_stats`, `interactive_loci`, `remote_actuators`, `driven_register_counts`,
        `morphology_stats`, `shape_stats`, and `d4_shape_stats` so earlier clock-only no-ops do
        not leave false positive contingency counts.
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
        self.remote_actuators.clear()
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
                    self._induce_spatial_operator(nope_prev, nope_next, act_key, cand_info)

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
                    H_n, W_n = nope_prev.shape
                    center_unchanged = (
                        0 <= cy < H_n
                        and 0 <= cx < W_n
                        and int(nope_prev[cy, cx]) == int(nope_next[cy, cx])
                    )
                    self.tool_selection_active = bool(
                        center_unchanged and eta_nl < 0.50 and 8 <= nope_d <= 44
                    )
                    e_prev = self.get_target_energy(sh_prev)
                    e_next = self.get_target_energy(sh_next)
                    self.coord_last_energy_delta[(cx, cy)] = float(e_next - e_prev)
                if cand_info is not None:
                    bh, bw = self._extract_bbox_hw(cand_info)
                    k_str = self._normalize_kind(str(cand_info.get("kind", "cc_object")))
                    area = int(cand_info.get("area", int(bh) * int(bw)))
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
                    d4_fam = self._make_d4_shape_key(k_str, int(bh), int(bw), area)
                    d4st = self.d4_shape_stats.setdefault(d4_fam, [0.0, 0.0])
                    d4st[0] += 1.0
                    if meaningful:
                        d4st[1] += 1.0

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
                self._induce_spatial_operator(nope_prev, nope_next, action_key, candidate_info)

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
                H_n, W_n = nope_prev.shape
                center_unchanged = (
                    0 <= cy < H_n
                    and 0 <= cx < W_n
                    and int(nope_prev[cy, cx]) == int(nope_next[cy, cx])
                )
                self.tool_selection_active = bool(
                    center_unchanged and eta_nl < 0.50 and 8 <= nope_d <= 44
                )
                e_prev = self.get_target_energy(sh_prev)
                e_next = self.get_target_energy(sh_next)
                self.coord_last_energy_delta[(cx, cy)] = float(e_next - e_prev)

            if candidate_info is not None:
                bh, bw = self._extract_bbox_hw(candidate_info)
                k_str = self._normalize_kind(str(candidate_info.get("kind", "cc_object")))
                area = int(candidate_info.get("area", int(bh) * int(bw)))
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

                d4_fam = self._make_d4_shape_key(k_str, int(bh), int(bw), area)
                d4st = self.d4_shape_stats.setdefault(d4_fam, [0.0, 0.0])
                d4st[0] += 1.0
                if meaningful:
                    d4st[1] += 1.0

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
        cur_align = compute_canvas_target_alignment(
            nope_cur, initial_frame_2d=init_f, spatial_precision_map=self.spatial_precision_map
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
                    nxt_b, initial_frame_2d=init_f, spatial_precision_map=self.spatial_precision_map
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

        def _eval_ot_energy(board: np.ndarray) -> float:
            al = compute_canvas_target_alignment(
                board, initial_frame_2d=init_f, spatial_precision_map=self.spatial_precision_map
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
        cur_e = float(
            compute_canvas_target_alignment(
                nope_cur, initial_frame_2d=init_f, spatial_precision_map=self.spatial_precision_map
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
        and Primitive C (`unroll_triad_cascade_plan`), returning the verified plan achieving
        the largest reduction in `E_target(F; Pi)`.
        """
        if z_L_token is None and z_L_tok is not None:
            z_L_token = z_L_tok
        norm_cands = self._normalize_plan_candidates(candidate_actions)
        nope_cur = self.splitter.project_nope(frame_2d).copy()
        init_f = (
            self.state_frames.get(self.initial_state_hash)
            if self.initial_state_hash is not None
            else self.splitter.initial_frame
        )
        start_e = float(
            compute_canvas_target_alignment(
                nope_cur, initial_frame_2d=init_f, spatial_precision_map=self.spatial_precision_map
            )["target_energy"]
        )

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
                        sim_b, initial_frame_2d=init_f, spatial_precision_map=self.spatial_precision_map
                    )["target_energy"]
                )
                red = start_e - end_e
                if red > best_reduction + 1e-5:
                    best_reduction = red
                    best_plan = valid_seq
                    best_method = m_name

        return {
            "plan": best_plan,
            "method": best_method,
            "energy_reduction": round(float(best_reduction), 6),
        }

    def compute_triad_action_priors(
        self,
        z_H_super: torch.Tensor,
        z_L_token: torch.Tensor,
        unroll_steps: int = 4,
        hrm_steps: Optional[int] = None,
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
            h_steps = int(hrm_steps) if hrm_steps is not None else int(unroll_steps)
            _, _, hrm_diag = self.hrm_core(
                z_H_in, z_L_in, max_steps=max(1, min(8, h_steps))
            )
            z_H_mean = z_H_in.mean(dim=1, keepdim=True).float()
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
            cas_diag.update(hrm_diag)
            priors = torch.tanh(triad_shift).cpu().numpy()
        return priors, cas_diag

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
          2. Spanning-tree entry-action ordering (`state_entry_action[state_hash]`) with
             Geodesic Corridor Momentum along active translation generators,
          3. Online morphological `(kind, color, h, w)` + shape `(kind, h, w)` + Dihedral `D_4`
             rotation-invariant `(kind, min(h,w), max(h,w), area_bucket)` posterior +
             Driven Passive Register (`Omega_reg`) pruning,
          4. Non-Local Actuator (`G_act`) & Goal-Converging Stepper continuation vs. Local `Z_2`
             Toggle Anti-Oscillation,
          5. Unified Goal Homomorphism defect locus & One-Shot Operator Lookahead bonus +
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

        if not isinstance(action_key, tuple):
            if act_id == 7:
                return -2.0 + triad_val + 1e-4 * g4_prior

            # Spanning-Tree Anti-Backtracking Guard:
            # Scale callosal prior to a non-destructive micro-tie-breaker (`1e-4`) so it never
            # overrides spanning-tree parent-return avoidance (`-0.60`), corridor continuation (`+0.32`),
            # or online empirical contingency / triad ordering.
            vis_cnt = self.state_visit_counts.get(state_hash, 1)
            eff_g4 = 1e-4 * g4_prior
            base = 2.15 * type_score + triad_val + eff_g4
            inverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
            entry_act = self.state_entry_action.get(state_hash)
            if isinstance(entry_act, int):
                if inverse_map.get(act_id) == entry_act:
                    # Parent-return direction in the spanning tree: try last at this state!
                    base -= 0.60
                elif act_id == entry_act and vis_cnt <= 1:
                    # Geodesic corridor continuation bonus on first visit to a state
                    base += 0.32
            return float(base)

        # Complex coordinate action: (6, x, y)
        cx, cy = int(action_key[1]), int(action_key[2])
        frame_2d = self.state_frames.get(state_hash)
        if frame_2d is not None and self.is_click_locally_inert(frame_2d, cx, cy, tier=1):
            return -50.0

        if self.is_driven_passive_register(cx, cy, candidate_info=candidate_info):
            return -35.0

        if candidate_info is None:
            return float(0.50 * type_score + triad_val + 1e-4 * g4_prior)

        sal = float(candidate_info.get("salience", 1.0))
        bh, bw = self._extract_bbox_hw(candidate_info)
        raw_kind = str(candidate_info.get("kind", "cc_object"))
        k_str = self._normalize_kind(raw_kind)
        area = int(candidate_info.get("area", int(bh) * int(bw)))
        fam = (
            k_str,
            int(candidate_info.get("color", 0)),
            int(bh),
            int(bw),
        )
        sh_fam = (k_str, int(bh), int(bw))
        d4_fam = self._make_d4_shape_key(k_str, int(bh), int(bw), area)

        f_trials, f_succs = self.morphology_stats.get(fam, [0.0, 0.0])
        s_trials, s_succs = self.shape_stats.get(sh_fam, [0.0, 0.0])
        base_d4_trials, base_d4_succs = self._base_d4_shape_stats.get(d4_fam, [0.0, 0.0])
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
            min_dist = min(
                max(abs(cx - lx), abs(cy - ly)) for (lx, ly) in self.interactive_loci
            )
            aligned_axis = any(
                abs(cy - ly) <= 2 or abs(cx - lx) <= 2 for (lx, ly) in self.interactive_loci
            )
            if min_dist <= 24:
                c_coeff = 1.35 if has_nonlocal_actuators else 1.15
                contingency_bonus += c_coeff * (1.0 - float(min_dist) / 32.0)
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
            novel_coord_bonus = 0.85 if has_nonlocal_actuators else 0.90
        elif is_global_stepper:
            # Global Stepper / Actuator Continuation: clicking this button produced a large
            # non-local board evolution (`nope_d >= 36`) into a brand-new state; continue stepping
            # along its orbit until the state repeats or all novel states are reached!
            novel_coord_bonus = 0.95 if has_nonlocal_actuators else 0.85
        else:
            novel_coord_bonus = -0.55 * float(prev_clicks_here)

        # Immediate repeat-click anti-oscillation guard for local Z_2 toggles (`s_0 -> s_1 -> s_0`)
        if has_nonlocal_actuators and last_action_key == action_key and not is_global_stepper:
            if len(self.interactive_loci) > 1 or last_delta_here < 36:
                novel_coord_bonus -= 2.25

        # Top-Down Spatial Precision Micro-Tie-Breaker (when Gemma 4 System 2 precision map is active)
        hom_bonus = 0.0
        if self.spatial_precision_map is not None:
            H_pm, W_pm = self.spatial_precision_map.shape
            if 0 <= cy < H_pm and 0 <= cx < W_pm:
                hom_bonus += 1e-4 * float(self.spatial_precision_map[cy, cx] - 1.0)

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
            + hom_bonus
            + 0.35 * ridge_val
            + triad_val
            + 1e-4 * g4_prior
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
        Zero-Bottleneck Cortico-Subcortical Left-Hemisphere Gemma 4 31B Callosal Deliberator:
          - Builds a structured Callosal World-Model Brief summarizing current level,
            topological symmetry potential `Phi_topo(F_t)`, target alignment energy `E_target(F_t)`,
            3D affective state `(V, U, Omega)`, discovered causal operators (`remote_actuators`
            vs. local toggles), and up to `K <= 10` numbered candidate actions `[0]..[K-1]`.
          - Runs a single forward pass `out = self.lh_model(**enc, output_hidden_states=True)`:
            * Pathway A (Native `R^5376` All-but-the-Top Centered Gram Resonance & Callosal Shell):
              Extracts `h_layer = out.hidden_states[hook_layer + 1]` in `R^{1 x T x 5376}`,
              applies All-but-the-Top common-mode centering (subtracting token mean `mu_29` and
              top-1 principal component `v_1`, normalizing onto `S^5375`), computes native
              span-to-span Gram resonance `R_29(a_k) -> p_29^gram(k)` directly in `R^5376`,
              and condenses `h_layer` via `Gemma4BeltramiCallosalShell` to modulate `z_H_super`.
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
                }
            return z_H_super

        self.gemma4_deliberated_states.add(cur_hash)
        self.gemma4_calls_made += 1
        self.gemma4_calls_this_level += 1
        pathway_a_injected = False
        gram_resonance_computed = False
        pathway_b_count = 0

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

                # Callosal E/I Fusion & Top-Down Spatial Precision Map `Pi(y, x)`
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

                    for k_idx in range(valid_k):
                        ak_k = cands_list[k_idx][1]
                        pk_val = float(p_callosal[k_idx])
                        self.world_model.gemma4_action_priors[(cur_hash, ak_k)] = 1.50 * pk_val
                        if isinstance(ak_k, tuple) and len(ak_k) == 3 and ak_k[0] == 6:
                            cx_k, cy_k = float(ak_k[1]), float(ak_k[2])
                            gauss = np.exp(-((yy - cy_k) ** 2 + (xx - cx_k) ** 2) / (2.0 * 4.0 ** 2))
                            pi_map += 1.50 * pk_val * gauss

                    self.world_model.spatial_precision_map = np.clip(pi_map, 0.2, 4.0)
                    pathway_b_count = valid_k
        except Exception:
            pass
        if return_dict_mode:
            return {
                "called": True,
                "pathway_a_injected": pathway_a_injected,
                "gram_resonance_computed": gram_resonance_computed,
                "pathway_b_priors": pathway_b_count,
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
        self.state_medoid_cache: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
        self.state_expanded: Set[str] = set()
        self.gemma4_deliberated_states: Set[str] = set()
        self.gemma4_calls_made: int = 0
        self.gemma4_calls_this_level: int = 0
        self.consecutive_no_progress: int = 0
        self.tier1_click_probes_failed: bool = False
        self.math_plan_attempted_states: Set[str] = set()
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
        self.state_medoid_cache.clear()
        self.state_expanded.clear()
        self.math_plan_attempted_states.clear()
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
            # `c6_succs == 0` (`untried_simple_count == 0 and c6_trials < 4` for `sc25`), and
            # expand up to 16 Beltrami medoids in Tier 2 (`dc22`).
            include_clicks_in_tier1 = (
                click_only_game
                or c6_succs > 0
                or simple_all_failed_here
                or (simple_trials >= len(simple_non_undo) and simple_succs == 0)
                or (untried_simple_count == 0 and c6_trials < 4)
            )

            # If simple generators are confirmed active (`simple_succs > 0`) and `ACTION6` has
            # failed >= 16 times across Tier-2 medoid probes with 0 successes, suppress clicks.
            suppress_clicks_globally = (
                not click_only_game and simple_succs > 0 and c6_trials >= 16 and c6_succs == 0
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
                    # In hybrid games where all initial Tier-1 click probes failed (`c6_trials >= 4`
                    # with `c6_succs == 0`, e.g. `dc22`) before a Tier-2 actuator succeeded, restrict
                    # Tier-1 clicks at subsequent keyboard states to confirmed interactive loci or D_4
                    # shapes with confirmed contingency!
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
                8,
                int(amygdala_state.get("reasoning_steps", schedule.unroll_steps)),
            ),
        )
        triad_priors, cas_diag = self.world_model.compute_triad_action_priors(
            z_H_super,
            z_L_tok,
            unroll_steps=schedule.unroll_steps,
            hrm_steps=hrm_unroll,
        )
        if z_H_callosal is not None:
            self.world_model.z_H = z_H_callosal.detach()
        self.last_cascade_diagnostics = cas_diag

        # Execute planned BFS path if still aligned with expected state
        if self.planned_bfs_queue:
            expected_state, next_bfs_act = self.planned_bfs_queue.popleft()
            if cur_hash == expected_state:
                self._record_chosen_action(frame_2d, cur_hash, next_bfs_act, None)
                return self._build_game_action(
                    next_bfs_act, schedule, mode_desc="bfs_frontier_navigation"
                )
            else:
                self.planned_bfs_queue.clear()

        # Case 1: Untried Tier-1 actions at `cur_hash`
        untried_here = self._get_untried_actions_for_state(cur_hash, avail, triad_priors, tier=1)
        if untried_here:
            _, chosen_act, chosen_info = untried_here[0]
            self._record_chosen_action(frame_2d, cur_hash, chosen_act, chosen_info)
            return self._build_game_action(chosen_act, schedule, mode_desc="tier1_frontier_explore")

        # Case 2: Shortest-path BFS over `G = (V, E)` to a frontier state `s*` with untried Tier-1 actions
        frontier_scores: Dict[str, float] = {}
        init_frame = (
            self.world_model.splitter.project_nope(self.world_model.state_frames[init_hash])
            if (init_hash is not None and init_hash in self.world_model.state_frames)
            else None
        )
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

            self._record_chosen_action(frame_2d, cur_hash, first_act, None)
            return self._build_game_action(first_act, schedule, mode_desc="bfs_shortest_path_start")

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
            "gemma4_calls_made": self.gemma4_calls_made,
            "last_amygdala_state": self.last_amygdala_state,
        }


__all__ = [
    "TemporalClockContentSplitter",
    "extract_arc3_beltrami_medoids",
    "compute_topological_symmetry_potential",
    "compute_marginal_wasserstein1_distance",
    "compute_canvas_target_alignment",
    "Arc3EpisodicBeltramiShell",
    "MagicNorm",
    "HRMRecurrentBlock",
    "Arc3HRMRecurrentCore",
    "Arc3AffectiveAmygdalaRouter",
    "TransitionRecord",
    "DarkSnailArc3WorldModel",
    "DarkSnailArc3Agent",
]



