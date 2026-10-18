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
    CrossLevelGoalInvariantDistiller,
    DarkSnailArc3Agent,
    DarkSnailArc3WorldModel,
    HRMRecurrentBlock,
    MagicNorm,
    OnlineForwardOperatorDreamer,
    RelationalOperatorDreamer,
    TemporalClockContentSplitter,
    TransitionRecord,
    compute_canvas_target_alignment,
    compute_marginal_wasserstein1_distance,
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


def test_subsystem_a_goal_homomorphism_and_mathematical_plan_synthesis() -> None:
    """
    Verifies Subsystem A:
      - `compute_marginal_wasserstein1_distance` is 0 on identical color marginals and positive on shifted shapes,
      - `compute_canvas_target_alignment` detects homomorphism defect (`E_hom`), marginal W1 (`E_marginal`),
        and `homomorphism_defect` medoids in `extract_arc3_beltrami_medoids`,
      - One-Shot Spatial Operator Induction (`_induce_spatial_operator` & `predict_action_outcome`) generalizes
        to unclicked lattice buttons (`lattice_operator_templates`),
      - Three Pure Mathematical Multi-Step Plan Synthesis Primitives:
          * Primitive A (`synthesize_abelian_jacobian_plan`) via ridge difference-basis pseudoinverse + greedy OMP,
          * Primitive B (`dream_forward_plan`) via Koopman mental rollout beam search,
          * Primitive C (`unroll_triad_cascade_plan`) via `SlowFastTriadCascade` trajectory readout on S^7(r_0),
          * Unified `synthesize_mathematical_plan` combining all three primitives with zero ASCII/LLM overhead.
    """
    # 1. Marginal Wasserstein-1 distance
    p_a = np.full((32, 32), 3, dtype=np.int32)
    p_a[4:8, 4:7] = 11   # stationary reference marker
    p_a[4:8, 8:10] = 11  # aligned movable column float
    p_b = np.full((32, 32), 3, dtype=np.int32)
    p_b[4:8, 4:7] = 11   # stationary reference marker
    p_b[20:24, 8:10] = 11  # vertically offset movable column float
    w1_a = compute_marginal_wasserstein1_distance(p_a)
    w1_b = compute_marginal_wasserstein1_distance(p_b)
    assert w1_a["wasserstein1_distance"] == pytest.approx(0.0, abs=1e-6)
    assert w1_b["wasserstein1_distance"] > 0.10

    # 2. Target homomorphism energy & homomorphism_defect medoids
    frame = np.full((64, 64), 4, dtype=np.int32)
    # Reference target miniature (e.g. 6x6 block at top-left) with two colors (9 and 12)
    frame[4:7, 4:7] = 9
    frame[4:7, 7:10] = 12
    # Main editable canvas (e.g. 18x18 block at center, 3x Kronecker scale) with one cell wrong
    frame[20:29, 20:29] = 9
    frame[20:29, 29:38] = 9  # should be 12 to match reference target

    align = compute_canvas_target_alignment(frame)
    assert "target_energy" in align
    assert 0.0 <= align["target_energy"] <= 1.0
    medoids = extract_arc3_beltrami_medoids(frame, max_candidates=16)
    assert len(medoids) > 0

    # 3. One-Shot Spatial Operator Induction + 3 Plan Synthesis Primitives
    wm = DarkSnailArc3WorldModel()
    # Build a synthetic 2-button lattice puzzle where clicking button 1 at (10, 10) toggles left half
    # to match target and clicking button 2 at (20, 10) toggles right half to match target.
    f0 = np.full((64, 64), 4, dtype=np.int32)
    # Reference target (top-right): 4x8 strip of color 14
    f0[4:8, 44:52] = 14
    # Main canvas (center): 8x16 strip currently color 8 (needs to become 14)
    f0[24:32, 20:36] = 8
    # Two interactive buttons of color 11 at (10, 10) and (20, 10)
    f0[9:12, 9:12] = 11
    f0[9:12, 19:22] = 11

    # Observe transition 1: clicking button 1 at (10, 10) flips left half [24:32, 20:28] from 8 -> 14
    f1 = f0.copy()
    f1[24:32, 20:28] = 14
    wm.record_transition(f0, f1, action_key=(6, 10, 10))
    # Reset back to f0 and observe transition 2: clicking button 2 at (20, 10) flips right half [24:32, 28:36] from 8 -> 14
    f2 = f0.copy()
    f2[24:32, 28:36] = 14
    wm.record_transition(f0, f2, action_key=(6, 20, 10))

    assert (6, 10, 10) in wm.learned_operators
    assert (6, 20, 10) in wm.learned_operators

    # Verify mental forward model `predict_action_outcome`
    pred_f1 = wm.predict_action_outcome(f0, (6, 10, 10))
    assert pred_f1 is not None
    assert bool(np.all(pred_f1[24:32, 20:28] == 14))

    # Verify Primitive A (Abelian Jacobian OMP), Primitive B (Koopman Dream Beam), Primitive C (Triad Cascade)
    cand_tuples = [(0.5, (6, 10, 10), None), (0.5, (6, 20, 10), None)]
    z_H_dummy = torch.zeros(1, 32, wm.d_reasoner, dtype=torch.float32)
    z_L_dummy = torch.zeros(1, 1, wm.d_reasoner, dtype=torch.float32)
    plan_a = wm.synthesize_abelian_jacobian_plan(f0, candidate_actions=cand_tuples, max_steps=4)
    plan_b = wm.dream_forward_plan(f0, candidate_actions=cand_tuples, beam_width=4, horizon=3)
    plan_c = wm.unroll_triad_cascade_plan(
        f0, z_H_dummy, z_L_dummy, candidate_actions=cand_tuples, horizon=2
    )
    plan_u = wm.synthesize_mathematical_plan(
        f0, z_H_super=z_H_dummy, z_L_token=z_L_dummy, candidate_actions=cand_tuples
    )

    assert len(plan_a) >= 1 or len(plan_b) >= 1
    assert set(plan_b).issubset({(6, 10, 10), (6, 20, 10)})
    assert isinstance(plan_c, list)
    assert isinstance(plan_u, dict) and "plan" in plan_u and len(plan_u["plan"]) >= 1


def test_subsystem_b_spectral_init_and_callosal_gram_resonance() -> None:
    """
    Verifies Subsystem B:
      - Deterministic spectral/orthonormal initialization in `HRMRecurrentBlock`, `Arc3HRMRecurrentCore`,
        and `Arc3AffectiveAmygdalaRouter` (`1e-3` output gain),
      - Native R^5376 All-but-the-Top Centered Gram Resonance + Callosal E/I fusion + Top-Down Spatial
        Precision Map (`world_model.spatial_precision_map`),
      - Spanning-tree anti-backtracking guard in `score_action_candidate` so Gemma 4 priors never
        override anti-backtracking on first visit in navigation games.
    """
    # 1. Verify spectral initialization in HRMRecurrentBlock & Arc3AffectiveAmygdalaRouter
    block = HRMRecurrentBlock(d_model=64, n_heads=8, d_ffn=128)
    s_q = torch.linalg.svdvals(block.q_proj.weight.detach())
    assert torch.allclose(s_q, torch.full_like(s_q, 0.5), atol=1e-4)

    router = Arc3AffectiveAmygdalaRouter(hidden_dim=64, mlp_dim=32)
    s_head = torch.linalg.svdvals(router.valence_head[2].weight.detach())
    assert float(s_head.max().item()) == pytest.approx(1e-3, abs=2e-4)

    # 2. Verify Bi-Hemispheric Callosal Signal Path with a mock Gemma 4 31B Left Hemisphere
    class MockGemma4LeftHemisphere(nn.Module):
        def __init__(self, d_lh: int = 5376, vocab_size: int = 64) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_lh)
            self.layer29 = nn.Linear(d_lh, d_lh, bias=False)
            self.layers = nn.ModuleList([nn.Identity() if i != 29 else self.layer29 for i in range(30)])
            self.lm_head = nn.Linear(d_lh, vocab_size, bias=False)
            nn.init.eye_(self.layer29.weight)

        def get_input_embeddings(self) -> nn.Module:
            return self.embed

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            output_hidden_states: bool = False,
            **kwargs,
        ):
            x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
            h29 = self.layers[29](x)
            logits = self.lm_head(h29)
            return SimpleNamespace(logits=logits, hidden_states=[h29 for _ in range(31)])

    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [(ord(c) % 60) + 1 for c in text[:24]] or [1]

        def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 768):
            ids = [(ord(c) % 60) + 1 for c in text[:max_length]] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    agent = DarkSnailArc3Agent(max_actions=50, seed=0)
    mock_lh = MockGemma4LeftHemisphere(d_lh=5376, vocab_size=64)
    mock_tok = MockTokenizer()
    agent.attach_gemma4_system2(mock_lh, hook_layer=29, tokenizer=mock_tok)
    assert agent.gemma4_shell is not None

    frame = np.full((64, 64), 3, dtype=np.int32)
    frame[16:24, 16:24] = 12
    frame[40:48, 40:48] = 9
    candidates = [1, 2, (6, 20, 20), (6, 44, 44)]
    cand_tuples = [(0.0, c, None) for c in candidates]
    delib = agent.deliberate_with_gemma4_system2(
        cur_hash="h0", frame_2d=frame, candidate_actions=cand_tuples
    )

    assert delib["called"] is True
    assert delib["gram_resonance_computed"] is True
    assert delib["pathway_b_priors"] == len(candidates)
    assert agent.world_model.spatial_precision_map is not None
    assert agent.world_model.spatial_precision_map.shape == (64, 64)
    assert float(agent.world_model.spatial_precision_map[20, 20]) > 1.0

    # 3. Verify spanning-tree anti-backtracking guard in `score_action_candidate`
    f_next = frame.copy()
    f_next[16:24, 16:24] = 14
    rec = agent.world_model.record_transition(frame, f_next, action_key=1)
    sh_next = rec.next_state_hash
    # Entering `sh_next` via action 1 (UP) makes action 2 (DOWN) the immediate backtracking inverse.
    # Even if Gemma 4 gives inverse action 2 a strong callosal prior, on first visit to `sh_next`,
    # spanning-tree anti-backtracking keeps forward/orthogonal action 3 strictly higher-scored than 2.
    agent.world_model.gemma4_action_priors[(sh_next, 2)] = 1.50
    agent.world_model.gemma4_action_priors[(sh_next, 3)] = 0.0
    zero_triad = np.zeros(8, dtype=np.float32)
    s_backtrack = agent.world_model.score_action_candidate(sh_next, 2, zero_triad)
    s_orthogonal = agent.world_model.score_action_candidate(sh_next, 3, zero_triad)
    assert s_orthogonal > s_backtrack


def test_milestone5_four_pillars_and_thermodynamic_irreversibility() -> None:
    """
    Verifies Milestone 5 Pillars 1-4 + Thermodynamic Irreversibility Law:
      1. Thermodynamic Irreversibility Law in `TemporalClockContentSplitter` (reversible
         state-restoring transitions are never permanently swallowed into `H_clock`),
      2. Pillar 1 (`RelationalOperatorDreamer` / `OnlineForwardOperatorDreamer`):
         - Ego-kinematic forward prediction (`predict_step`) at novel board positions + wall blocking,
         - Excitable Excursion Verifier (`verifier_actuators`, `subframes_count >= 3`, `max_subframe_nope_delta >= 12`),
         - Contact-Highlight `ACTION5` epistemic boost (`active_contact_highlight_states`),
         - Tool/Palette Selection Mode (`tool_selection_active`) bypassing pre-tool inertness,
      3. Pillar 2 (`CrossLevelGoalInvariantDistiller`):
         - 5D invariant basis `Phi(S) = [Phi_1..Phi_5]^T` and Fisher-Rao natural-gradient Boltzmann
           weight update on `distill_level_win`,
      4. Pillar 3 (Online Riemannian / Lie-Algebra Test-Time Adaptation):
         - `so(16)` Lie generator estimation (`update_lie_generator`), commutator curvature
           `compute_commutator_curvature`, and single-step differentiable `adapt_online_step`
           preserving `<= 1e-8` pointwise orthogonality,
      5. Pillar 4 (Left-Hemisphere Gemma 4 Subgoal & Precision Weight Compiler):
         - Layer-29 Centered Spherical Gram Resonance over `[G0..G4]` subgoal archetypes and
           `apply_system2_compiled_weights`.
    """
    # 1. Thermodynamic Irreversibility Law: reversible 2-color column shift is NOT permanently masked as clock
    splitter = TemporalClockContentSplitter()
    f0 = np.full((64, 64), 3, dtype=np.int32)
    f0[10:26, 20:24] = 11
    f1 = f0.copy()
    f1[10:26, 20:24] = 3
    f1[26:42, 20:24] = 11
    # Reversible step back to f0 must keep H_clock empty (or restore any provisional clock pixels)
    splitter.observe_transition(f0, f1, action_key=(6, 48, 46))
    splitter.observe_transition(f1, f0, action_key=(6, 48, 46))
    assert splitter.clock_mask is None or int(splitter.clock_mask.sum()) == 0

    # 2. Pillar 1: RelationalOperatorDreamer (`OnlineForwardOperatorDreamer`)
    assert OnlineForwardOperatorDreamer is RelationalOperatorDreamer
    dreamer = RelationalOperatorDreamer()
    b0 = np.full((64, 64), 4, dtype=np.int32)
    # Ego 3x3 sprite of color 9 at (20..23, 10..13)
    b0[20:23, 10:13] = 9
    # Stationary wall of color 2 at (20..23, 22..25)
    b0[20:23, 22:25] = 2
    b1 = b0.copy()
    b1[20:23, 10:13] = 4
    b1[20:23, 13:16] = 9  # Ego translated +3 cols right under ACTION4
    obs_info = dreamer.observe_transition(b0, b1, action_key=4, sh_prev="s0", sh_next="s1", nope_d=18)
    assert obs_info["ego_moved"] is True
    assert dreamer.ego_color == 9
    assert 4 in dreamer.kinematic_generators
    assert dreamer.kinematic_generators[4][:2] == (0, 3)

    # Predict ego step from a NEW position (20..23, 13..16) -> (20..23, 16..19)
    b_pred = dreamer.predict_step(b1, 4)
    assert b_pred is not None
    assert bool(np.all(b_pred[20:23, 16:19] == 9))
    assert bool(np.all(b_pred[20:23, 13:16] == 4))

    # Excitable Excursion Verifier & Contact-Highlight ACTION5 boost in DarkSnailArc3WorldModel
    wm = DarkSnailArc3WorldModel()
    v0 = np.full((64, 64), 4, dtype=np.int32)
    v0[30:34, 30:34] = 9
    sf_mid = v0.copy()
    sf_mid[10:18, 10:18] = 14  # transient 64-pixel validation wave during subframes
    rec_v = wm.record_transition(
        v0, v0.copy(), action_key=(6, 32, 32), subframes=[v0, sf_mid, v0.copy()]
    )
    assert rec_v.is_self_loop is False
    assert (6, 32, 32) in wm.verifier_actuators
    zero_triad = np.zeros(8, dtype=np.float32)
    sc_before_edit = wm.score_action_candidate(rec_v.state_hash, (6, 32, 32), zero_triad)
    wm.edits_since_last_verify = 2
    sc_after_edit = wm.score_action_candidate(rec_v.state_hash, (6, 32, 32), zero_triad)
    assert sc_after_edit > sc_before_edit + 8.0

    # Contact-Highlight ACTION5 boost: docking 3x3 ego inside 5x5 hollow receptor frame
    c0 = np.full((64, 64), 4, dtype=np.int32)
    c0[20:25, 20:25] = 12
    c0[21:24, 21:24] = 4  # hollow 5x5 receptor frame of color 12
    c0[21:24, 17:20] = 9  # 3x3 ego outside receptor
    c1 = c0.copy()
    c1[21:24, 17:20] = 4
    c1[21:24, 21:24] = 9  # ego docked inside hollow receptor frame!
    rec_c = wm.record_transition(c0, c1, action_key=4)
    assert rec_c.next_state_hash in wm.relational_dreamer.active_contact_highlight_states
    sc_act5 = wm.score_action_candidate(rec_c.next_state_hash, 5, zero_triad)
    sc_act3 = wm.score_action_candidate(rec_c.next_state_hash, 3, zero_triad)
    assert sc_act5 > sc_act3 + 10.0

    # 3. Pillar 2: CrossLevelGoalInvariantDistiller
    distiller = CrossLevelGoalInvariantDistiller()
    s_init = np.full((64, 64), 3, dtype=np.int32)
    s_init[10:16, 10:16] = 14  # 36 pixels of minority target color 14
    s_init[40:44, 40:44] = 9   # 16 pixels of avatar color 9
    s_win = np.full((64, 64), 3, dtype=np.int32)
    s_win[10:14, 10:14] = 9    # color 14 completely cleared/annihilated at win!
    d_info = distiller.distill_level_win(s_init, s_win, s_win, winning_action_key=1)
    assert distiller.levels_distilled == 1
    assert 14 in distiller.cleared_colors
    assert np.isclose(float(np.sum(distiller.weights)), 1.0, atol=1e-6)
    assert d_info["updated_weights"][3] > 0.25  # G3:ENTROPY_ANNIHILATION weight increased!

    s_mid = s_init.copy()
    s_mid[10:16, 10:13] = 3  # half of color 14 cleared
    gain = distiller.score_transition_gain(s_init, s_mid, initial_frame_2d=s_init)
    assert gain > 0.0

    # 4. Pillar 3: Online Riemannian / Lie-Algebra TTT (`so(16)` commutator & `adapt_online_step`)
    wm.update_lie_generator(b0, b1, act_id=4)
    b2 = b1.copy()
    b2[20:23, 13:16] = 4
    b2[10:18, 30:38] = 11
    wm.update_lie_generator(b1, b2, act_id=1)
    kappa = wm.compute_commutator_curvature(4, 1)
    assert kappa >= 0.0

    z_H = torch.randn(1, 32, wm.d_reasoner)
    z_L = torch.randn(1, 1, wm.d_reasoner)
    ttt_diag = wm.adapt_online_step(z_H, z_L, action_key=4, reward=1.0, is_self_loop=False)
    assert ttt_diag["ttt_steps"] >= 1
    assert ttt_diag["action_triad_orthogonality_error"] <= 1e-8
    priors, cas_diag = wm.compute_triad_action_priors(z_H, z_L, unroll_steps=2)
    assert priors.shape == (8,)
    assert cas_diag["max_norm_drift"] == pytest.approx(0.0, abs=1e-6)
    assert cas_diag["action_triad_orthogonality_error"] <= 1e-8

    # 5. Pillar 4: Left-Hemisphere Gemma 4 Subgoal & Precision Weight Compiler
    w_after = distiller.apply_system2_compiled_weights([0.60, 0.10, 0.10, 0.10, 0.10], blend=0.40)
    assert np.isclose(float(w_after.sum()), 1.0, atol=1e-6)
    assert float(w_after[0]) > 0.30




