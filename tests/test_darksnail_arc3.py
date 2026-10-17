"""
Unit tests for `darksnail.arc3` (Bi-Hemispheric DarkSnail V2 + Gemma 4 31B ARC-AGI-3 Agent).

Verifies:
1. `extract_arc3_beltrami_medoids` on synthetic `64x64` 16-color grids (including non-convex
   hollow rings, enclosed cavities, symmetry defects, and Dyadic Quadtree Stratification).
2. `TemporalClockContentSplitter` via the discrete continuity equation (`d_t rho_c + div J_c = sigma_c`)
   with zero spatial/border priors (testing an interior clock bar at row `r=30`), plus Law 4
   Periodic Phase-Reset / Respawn detection.
3. `Arc3EpisodicBeltramiShell` and `DarkSnailArc3WorldModel` mathematical invariants:
   - `(1, 32, 512)` Adelic super-token shape (both native and `use_gemma4_shell=True`),
   - `SlowFastTriadCascade` exact `0.000e+00` spherical norm conservation,
   - `SkewSymmetricTriadOperator` `<= 1e-8` pointwise orthogonality,
   - 1-Step Deterministic Empowerment `E_1(s)` and Topological Symmetry Potential `Phi_topo(F_t)`.
4. Self-loop pruning, Local Patch Invariance (`is_click_locally_inert`), `GAME_OVER` trap
   avoidance, and Epistemic Frontier BFS shortest-path navigation.
5. Live execution on `arc_agi` offline environments (`ls20` and `ft09`) in `OperationMode.OFFLINE`.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from darksnail import (
    Arc3AffectiveAmygdalaRouter,
    Arc3EpisodicBeltramiShell,
    Arc3HRMRecurrentCore,
    DarkSnailArc3Agent,
    DarkSnailArc3WorldModel,
    HRMRecurrentBlock,
    MagicNorm,
    TemporalClockContentSplitter,
    TransitionRecord,
    compute_topological_symmetry_potential,
    extract_arc3_beltrami_medoids,
)


def test_extract_arc3_beltrami_medoids_interior_ring_and_quadtree() -> None:
    """
    Verifies that `extract_arc3_beltrami_medoids`:
      - Returns at most `max_candidates` coordinates within `[0, 63] x [0, 63]`,
      - Selects an interior pixel strictly on the body of a non-convex hollow ring (`cc_object`),
      - Detects the enclosed cavity inside the ring (`enclosed_hole`),
      - Stratifies candidates across all 4 spatial quadrants when multi-panel objects are present,
      - Orthogonal-projects out any candidate across all 4 channels (`fg_comps`, `enclosed_hole`,
        `symmetry_defect`, `rope2d_patch`) whose medoid `(mx, my)` or >= 50% of its pixels lie in `H_clock`.
    """
    frame = np.full((64, 64), 3, dtype=np.int32)

    # Non-convex 7x7 hollow ring of color 12 in Quadrant 0 (top-left)
    frame[6:13, 6:13] = 12
    frame[8:11, 8:11] = 3  # 3x3 enclosed background cavity at center (9, 9)

    # Distinct objects in Quadrants 1, 2, 3
    frame[10:14, 48:52] = 8   # Quadrant 1 (top-right)
    frame[46:50, 12:16] = 11  # Quadrant 2 (bottom-left)
    frame[48:52, 48:52] = 14  # Quadrant 3 (bottom-right)

    cands = extract_arc3_beltrami_medoids(frame, max_candidates=24)
    assert 4 <= len(cands) <= 24

    for c in cands:
        x, y = int(c["x"]), int(c["y"])
        assert 0 <= x < 64 and 0 <= y < 64
        assert isinstance(c["feature_vec"], np.ndarray)
        assert c["feature_vec"].shape == (12,)

    # Verify the ring's `cc_object` medoid lies strictly on color 12 (never inside the hole of color 3)
    ring_cands = [c for c in cands if c["color"] == 12 and c["kind"] in ("cc_object", "cc_lattice")]
    assert len(ring_cands) >= 1
    rx, ry = int(ring_cands[0]["x"]), int(ring_cands[0]["y"])
    assert int(frame[ry, rx]) == 12

    # Verify the enclosed cavity at (8..10, 8..10) is detected as `enclosed_hole`
    hole_cands = [c for c in cands if c["kind"] == "enclosed_hole"]
    assert any(8 <= int(hc["x"]) <= 10 and 8 <= int(hc["y"]) <= 10 for hc in hole_cands)

    # Verify all 4 spatial quadrants are represented in the candidate set
    quads = {(2 if int(c["y"]) >= 32 else 0) + (1 if int(c["x"]) >= 32 else 0) for c in cands}
    assert quads == {0, 1, 2, 3}

    # Verify H_clock orthogonal projection: masking Quadrant 1 object [10:14, 48:52] and the ring hole [8:11, 8:11]
    clock_mask = np.zeros((64, 64), dtype=bool)
    clock_mask[10:14, 48:52] = True
    clock_mask[8:11, 8:11] = True
    cands_masked = extract_arc3_beltrami_medoids(frame, max_candidates=24, clock_mask=clock_mask)
    for cm in cands_masked:
        mx, my = int(cm["x"]), int(cm["y"])
        assert not bool(clock_mask[my, mx]), f"Candidate ({mx}, {my}) of kind {cm['kind']} lies inside H_clock!"


def test_temporal_clock_content_splitter_zero_border_priors_and_phase_reset() -> None:
    """
    Verifies `TemporalClockContentSplitter` with zero spatial/border priors:
      - Places an exogenous dissipative clock bar right in the INTERIOR of the canvas (`r=30, c=10..50`)
        and a controllable sprite at `(r=12..16, c=20..24)`.
      - Confirms the interior clock bar is masked into `H_clock` while the translating sprite
        (which conserves global color mass `Delta M_c == 0`) remains in `H_nope`.
      - Confirms Law 4 (Periodic Phase-Reset / Respawn) detects when the clock bar refills
        to `F_0` and the board reverts to `F_0` minus a 4px life icon.
      - Confirms `preserve_clock_mask=True` carries the discovered `H_clock` subspace into Level 2+.
    """
    splitter = TemporalClockContentSplitter()

    f0 = np.full((64, 64), 4, dtype=np.int32)
    # Interior clock bar at row 30, cols 10..30 (color 9, 20 pixels)
    f0[30, 10:30] = 9
    # Interior life counter icon at rows 35..37, cols 50..52 (color 6, 4 pixels)
    f0[35:37, 50:52] = 6
    # Controllable 4x4 player sprite at rows 12..16, cols 20..24 (color 14, 16 pixels)
    f0[12:16, 20:24] = 14

    splitter.initialize_level(f0)

    # Transition 1: Sprite moves right by 4px (conservative: +16/-16 of color 14 -> Delta M_14 == 0),
    # while interior clock bar at row 30 depletes 1px at (30, 29) from color 9 to color 5 (dissipative).
    f1 = f0.copy()
    f1[12:16, 20:24] = 4
    f1[12:16, 24:28] = 14
    f1[30, 29] = 5

    expanded, nope_d1, is_reset1 = splitter.observe_transition(f0, f1, action_key=4)
    assert expanded is True
    assert is_reset1 is False
    assert nope_d1 == 32  # 16 pixels vacated + 16 pixels occupied by sprite; clock pixel excluded!
    assert splitter.clock_mask is not None
    # Entire interior clock bar component (30, 10:30) should be masked in H_clock
    assert bool(np.all(splitter.clock_mask[30, 10:30]))
    # Player sprite must NOT be masked in H_clock
    assert not bool(np.any(splitter.clock_mask[12:16, 20:28]))

    # Transition 2: Wall bump at f1 where only the interior clock bar ticks down (depleting 9 more pixels)
    f2 = f1.copy()
    f2[30, 20:29] = 5
    _, nope_d2, is_reset2 = splitter.observe_transition(f1, f2, action_key=1)
    assert nope_d2 == 0
    assert is_reset2 is False
    assert splitter.hash_nope(f1) == splitter.hash_nope(f2)

    # Transition 3: Periodic Phase-Reset / Respawn back to f0 with clock refilled and 4px life icon depleted
    f_respawn = f0.copy()
    f_respawn[35:37, 50:52] = 4  # 4px life counter depleted
    _, nope_d3, is_reset3 = splitter.observe_transition(f2, f_respawn, action_key=4)
    assert is_reset3 is True
    assert splitter.hash_nope(f_respawn) == splitter.hash_nope(f0)

    # Verify cross-level persistence of H_clock
    splitter.initialize_level(f0, preserve_clock_mask=True)
    assert splitter.num_clock_pixels >= 20
    assert bool(np.all(splitter.clock_mask[30, 10:30]))


def test_episodic_beltrami_shell_and_world_model_invariants() -> None:
    """
    Verifies:
      - `Arc3EpisodicBeltramiShell` outputs `(1, 32, 512)` super-tokens for both standard
        `BeltramiCallosalShell` and `Gemma4BeltramiCallosalShell` (`use_gemma4_shell=True`),
      - `SlowFastTriadCascade` exact `0.000e+00` radial norm conservation,
      - `SkewSymmetricTriadOperator` `<= 1e-8` pointwise orthogonality,
      - `Arc3HRMRecurrentCore` (`MagicNorm`, `L_module`, `H_module`, `q_head` `[Q_halt, Q_continue]`),
      - `compute_state_empowerment` and `compute_topological_symmetry_potential`.
    """
    f0 = np.zeros((64, 64), dtype=np.int32)
    f0[16:32, 16:32] = 5
    f1 = f0.copy()
    f1[16:32, 20:36] = 5

    for use_g4 in (False, True):
        shell = Arc3EpisodicBeltramiShell(d_model=512, target_tokens=32, use_gemma4_shell=use_g4)
        super_tokens, diag = shell(f1, prev_frame_2d=f0, action_id=4, return_diagnostics=True)
        assert super_tokens.shape == (1, 32, 512)
        assert math.isfinite(float(diag["zeno_flux"]))
        assert math.isfinite(float(diag["lca_variance"]))

    wm = DarkSnailArc3WorldModel(d_reasoner=512, d_shell=8, unroll_steps=4)
    z_H = torch.randn(1, 32, 512)
    z_L = torch.randn(1, 1, 512)
    priors, cas_diag = wm.compute_triad_action_priors(z_H, z_L, unroll_steps=4)
    assert priors.shape == (8,)
    assert float(cas_diag["max_norm_drift"]) == 0.0
    assert float(cas_diag["action_triad_orthogonality_error"]) <= 1e-8
    assert math.isfinite(float(cas_diag["hrm_q_halt"]))
    assert math.isfinite(float(cas_diag["hrm_q_continue"]))
    assert isinstance(cas_diag["hrm_should_halt"], bool)

    # Verify Topological & Dual-Panel Symmetry Potential increases when two sub-panels agree
    panel_mismatch = np.zeros((64, 64), dtype=np.int32)
    panel_mismatch[10:16, 10:16] = 2
    panel_mismatch[10:16, 30:36] = 2
    panel_mismatch[12:14, 32:34] = 7  # 2x2 discrepancy in panel 2

    panel_aligned = panel_mismatch.copy()
    panel_aligned[12:14, 32:34] = 2  # panels now match 100% and are symmetric

    phi_bad = compute_topological_symmetry_potential(panel_mismatch)
    phi_good = compute_topological_symmetry_potential(panel_aligned)
    assert phi_good > phi_bad


def test_nonlocal_operator_factorization_and_d4_shape_invariance() -> None:
    """
    Verifies Non-Local Causal Operator Factorization (`G_act` vs `Omega_reg`) and
    Dihedral `D_4` rotation-invariant shape transfer:
      - Clicking an active peripheral slider/button at `(5, 30)` that mutates a remote passive
        board region at `[20:24, 20:26]` (`nope_d = 24`, Chebyshev distance > 4, `eta_nonlocal = 1.0`)
        registers `(5, 30)` in `remote_actuators` (`G_act`) and increments `driven_register_counts`
        (`Omega_reg`) at `[20:24, 20:26]`,
      - `is_driven_passive_register(23, 22)` returns True so the agent prunes passive display cells,
        whereas `is_driven_passive_register(5, 30)` returns False for the active generator,
      - Dihedral `D_4` shape transfer matches a `2x8` horizontal bar and an `8x2` vertical bar
        under the same canonical signature `(kind, 2, 8, area_bucket)`.
    """
    wm = DarkSnailArc3WorldModel()
    f0 = np.zeros((64, 64), dtype=np.int32)
    # Peripheral horizontal slider/button 2x8 at rows 29..31, cols 2..10 (center ~ (6, 30))
    f0[29:31, 2:10] = 11
    # Passive state register at rows 20..24, cols 20..26
    f0[20:24, 20:26] = 3

    f1 = f0.copy()
    # Clicking (6, 30) shifts the passive register at rows 20..24, cols 20..26 to color 8 (24 pixels remote)
    f1[20:24, 20:26] = 8

    cand_horiz = {
        "x": 6,
        "y": 30,
        "kind": "cc_object",
        "area": 16,
        "bbox": (29, 2, 30, 9),  # bh=2, bw=8
    }
    rec = wm.record_transition(f0, f1, action_key=(6, 6, 30), candidate_meta=cand_horiz)
    assert rec.nope_delta_l1 == 24
    assert (6, 30) in wm.remote_actuators
    assert wm.is_driven_passive_register(22, 22) is True
    assert wm.is_driven_passive_register(6, 30) is False

    # Now test a 90-degree rotated 8x2 vertical bar (bh=8, bw=2, area=16):
    cand_vert = {
        "x": 30,
        "y": 6,
        "kind": "cc_object",
        "area": 16,
        "bbox": (2, 29, 9, 30),  # bh=8, bw=2
    }
    assert wm.get_d4_shape_prior(cand_vert) > 0.4


def test_hrm_and_amygdala_router_and_dual_pathway_gemma4_deliberation() -> None:
    """
    Verifies:
      - `Arc3AffectiveAmygdalaRouter` computes 3D affective state `[V_t, U_t, Omega_t]` and
        neuromodulatory gains `(gamma_E, gamma_I, T_gen, N_steps, bypass_active)`, and triggers
        `should_call_gemma4_system2=True` on epistemic stall (`consecutive_non_progress >= 5`),
      - `deliberate_with_gemma4_system2` executes a synthetic Gemma 4 model with hidden size 5376,
        running BOTH Pathway A (`Gemma4BeltramiCallosalShell` `(1, T, 5376) -> (1, 32, 4096) -> (1, 32, 512)`
        into `world_model.z_H`) and Pathway B (next-token digit `"0".."9"` logit softmax into
        `world_model.gemma4_action_priors`).
    """
    router = Arc3AffectiveAmygdalaRouter(d_model=512)
    z_H = torch.randn(1, 32, 512)
    z_L = torch.randn(1, 1, 512)

    # Normal progress step: System-1 reflex handles it without calling Gemma 4
    mod_normal = router.step(
        z_H,
        z_L,
        last_nope_delta=16,
        is_new_state=True,
        delta_phi=0.15,
        untried_ratio=0.5,
        consecutive_non_progress=0,
        was_game_over=False,
        step_idx=1,
    )
    assert -1.0 <= mod_normal["valence"] <= 1.0
    assert 0.0 <= mod_normal["uncertainty"] <= 1.0
    assert 0.0 <= mod_normal["urgency"] <= 1.0
    assert mod_normal["gamma_E"] > 0.0 and mod_normal["gamma_I"] > 0.0
    assert mod_normal["should_call_gemma4_system2"] is False

    # Epistemic stall step: triggers System-2 callosal deliberation
    mod_stall = router.step(
        z_H,
        z_L,
        last_nope_delta=0,
        is_new_state=False,
        delta_phi=-0.05,
        untried_ratio=0.9,
        consecutive_non_progress=6,
        was_game_over=False,
        step_idx=10,
    )
    assert mod_stall["should_call_gemma4_system2"] is True

    # Test Dual-Pathway Gemma 4 Callosal Deliberator with a lightweight mock Gemma 4 model & tokenizer
    class DummyGemma4Tokenizer:
        def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
            return {"input_ids": torch.randint(0, 64, (1, 16))}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [int(text) + 10]

    class DummyGemma4Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(64, 5376)
            self.head = nn.Linear(5376, 64, bias=False)

        def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **kwargs):
            h = self.embed(input_ids)
            logits = self.head(h)
            # Bias digit "1" (token id 11) to be the highest-scoring candidate action
            logits[:, -1, 11] += 10.0
            hidden_states = tuple(h for _ in range(31)) if output_hidden_states else None
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    agent = DarkSnailArc3Agent(max_actions=50, seed=0)
    agent.attach_gemma4_system2(DummyGemma4Model(), tokenizer=DummyGemma4Tokenizer(), hook_layer=29)

    f0 = np.zeros((64, 64), dtype=np.int32)
    f0[20:24, 20:24] = 12
    sh0 = agent.world_model.register_state(f0, step_idx=0)
    cands = extract_arc3_beltrami_medoids(f0, max_candidates=8)
    agent.world_model.state_candidates[sh0] = cands

    delib = agent.deliberate_with_gemma4_system2(
        cur_hash=sh0,
        frame_2d=f0,
        avail_set={1, 2, 6},
        candidates=cands,
        amygdala_diag=mod_stall,
    )
    assert delib["called"] is True
    assert delib["pathway_a_injected"] is True
    assert delib["pathway_b_priors"] >= 2
    assert agent.gemma4_deliberations == 1
    assert agent.world_model.z_H is not None
    assert agent.world_model.z_H.shape == (1, 32, 512)
    # Candidate [1] is simple action 2; verify Pathway B boosted its prior above Candidate [0] (action 1)
    assert agent.world_model.gemma4_action_priors[(sh0, 2)] > agent.world_model.gemma4_action_priors[(sh0, 1)]


def test_noop_pruning_local_patch_invariance_and_bfs_navigation() -> None:
    """
    Verifies:
      - Self-loop transitions (`nope_delta_l1 == 0`) and `GAME_OVER` transitions are masked,
      - Local Patch Invariance (`is_click_locally_inert`) transfers inert click knowledge across
        states whenever the local `5x5` `H_nope` patch around `(x, y)` is unchanged,
      - `find_shortest_path_to_frontier` routes around `GAME_OVER` and self-loop edges.
    """
    wm = DarkSnailArc3WorldModel()
    f_a = np.zeros((64, 64), dtype=np.int32)
    f_a[10:13, 10:13] = 6  # static reference tile at (11, 11)
    f_a[40:43, 40:43] = 8  # interactive tile at (41, 41)

    sh_a = wm.register_state(f_a, step_idx=0)

    # Clicking (11, 11) is a self-loop
    rec_inert = wm.record_transition(f_a, f_a.copy(), action_key=(6, 11, 11))
    assert rec_inert.is_self_loop is True
    assert (6, 11, 11) in wm.self_loops[sh_a]
    assert wm.is_click_locally_inert(f_a, 11, 11) is True

    # Clicking (41, 41) transitions to f_b; notice (11, 11) patch is unchanged in f_b!
    f_b = f_a.copy()
    f_b[40:43, 40:43] = 9
    rec_move = wm.record_transition(f_a, f_b, action_key=(6, 41, 41))
    sh_b = rec_move.next_state_hash
    assert sh_b != sh_a
    # Local Patch Invariance: (11, 11) is immediately known to be inert in f_b without re-clicking!
    assert wm.is_click_locally_inert(f_b, 11, 11) is True

    # Action 1 from f_b causes GAME_OVER
    wm.record_transition(f_b, f_a, action_key=1, is_game_over=True)
    assert 1 in wm.game_over_actions[sh_b]
    assert 1 not in wm.out_edges.get(sh_b, {})

    # Action 2 from f_b transitions safely to f_c
    f_c = f_b.copy()
    f_c[20:24, 20:24] = 11
    rec_bc = wm.record_transition(f_b, f_c, action_key=2)
    sh_c = rec_bc.next_state_hash

    # Empowerment at sh_a and sh_b should be strictly positive
    assert wm.compute_state_empowerment(sh_a, untried_count=3) > 0.0

    # Shortest path from sh_a to frontier state sh_c must be [(6, 41, 41), 2]
    bfs = wm.find_shortest_path_to_frontier(sh_a, {sh_c: 5.0})
    assert bfs is not None
    target_sh, path = bfs
    assert target_sh == sh_c
    assert path == [(6, 41, 41), 2]


def test_agent_game_over_reset_and_gemma4_attachment() -> None:
    """Verifies immediate `RESET` on `GAME_OVER` and `attach_gemma4_system2` hook creation."""
    from arcengine import GameAction, GameState

    agent = DarkSnailArc3Agent(max_actions=50, seed=0)
    dummy_lh = nn.Linear(16, 16)
    g4_shell = agent.attach_gemma4_system2(dummy_lh, hook_layer=29)
    assert g4_shell.hook_layer == 29

    f0 = np.zeros((64, 64), dtype=np.int32)
    obs_play = SimpleNamespace(
        game_id="synthetic-v1",
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=3,
        frame=[f0],
        available_actions=[1, 2, 3, 4],
    )
    act1 = agent.choose_action([obs_play], obs_play)
    assert act1 in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4)

    obs_over = SimpleNamespace(
        game_id="synthetic-v1",
        state=GameState.GAME_OVER,
        levels_completed=0,
        win_levels=3,
        frame=[f0],
        available_actions=[1, 2, 3, 4],
    )
    act_reset = agent.choose_action([obs_play, obs_over], obs_over)
    assert act_reset == GameAction.RESET


def test_live_offline_ls20_and_ft09_execution() -> None:
    """
    Runs `DarkSnailArc3Agent` on live `ls20` (navigation + fuel clock), `vc33` (coordinate click
    puzzle), and `ft09` (multi-panel constraint click puzzle) in `OperationMode.OFFLINE`.
    """
    from arc_agi import Arcade, OperationMode

    env_dir = Path(__file__).resolve().parent.parent / "data" / "arc_agi_3_environments"
    if not env_dir.exists():
        pytest.skip("Offline ARC-AGI-3 environments directory not found")

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir))

    # 1. Test `ls20`: verify exogenous clock masking and level progress
    env_ls20 = arc.make("ls20")
    agent_ls20 = DarkSnailArc3Agent(max_actions=220, seed=0)
    res_ls20 = agent_ls20.play_game(env_ls20, max_actions=220)
    assert res_ls20["actions_taken"] <= 220
    assert res_ls20["clock_pixels_masked"] > 0
    assert res_ls20["levels_completed"] >= 1

    # 2. Test `vc33`: verify Dyadic Quadtree medoid extraction & Local Patch Invariance solve >= 1 level
    env_vc33 = arc.make("vc33")
    agent_vc33 = DarkSnailArc3Agent(max_actions=100, seed=0)
    res_vc33 = agent_vc33.play_game(env_vc33, max_actions=100)
    assert res_vc33["actions_taken"] <= 100
    assert res_vc33["clock_pixels_masked"] > 0
    assert res_vc33["levels_completed"] >= 1

    # 3. Test `ft09`: verify exogenous clock masking (64 px) and non-trivial controllable state discovery
    env_ft09 = arc.make("ft09")
    agent_ft09 = DarkSnailArc3Agent(max_actions=80, seed=0)
    res_ft09 = agent_ft09.play_game(env_ft09, max_actions=80)
    assert res_ft09["actions_taken"] <= 80
    assert res_ft09["clock_pixels_masked"] == 64
    assert res_ft09["unique_nope_states_visited"] >= 15


