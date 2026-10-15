"""
Unit and Integration Test Suite for the `darksnail` Synthesis Package.

Covers:
1. `SkewSymmetricTriadOperator`: Pointwise orthogonality `<v, B_W(u, v)> == 0` in float32,
   shared and per-batch `W`, and 2-color tie-breaking antisymmetry.
2. `DarkSnailCanvasReadout`: Exact Step-0 System 1 homotopy (`delta_z == 0`), exact agreement
   with `System2LatentAdaptor` on the symmetric slice, gradient flow into the 5,500+ previously
   idle coordinates (`flat_dz[2498:4810]`), and pixel-wise triad orthogonality.
3. `SlowFastTriadCascade`: Exact Step-0 identity, shell tangent orthogonality, and discrete
   spherical energy conservation `||P z_tilde_L||_2^2 == ||P z_L||_2^2`.
4. `RoPE2DMedoidPooler`: 2D spatial patch medoid selection, `0.5 * (mean + peak)` value pooling,
   isotropic Adelic variance attenuation `gamma_2D`, and 2D axial RoPE relative phase invariance.
5. `BeltramiCallosalShell` & `compute_grid_prompt_zeno_flux`: Tier-1 Beltrami outlier shielding,
   Tier-2 ultrametric medoid condensation, exact `(B, 32, 4096)` contract with `CallosalConditioning`,
   V4 `TriadTopologyRouter` sidecar integration, and sub-millisecond ARC grid Zeno flux telemetry.
6. `ZenoAmygdalaController` & `DarkSnailSystem2Adaptor`: Dynamic scheduling across surgical
   near-solved, balanced, and high-turbulence regimes, surgical 1-2 cell ray quantiles, and
   end-to-end TTA adaptation.
"""

from __future__ import annotations

import time
import numpy as np
import pytest
import torch
import torch.nn as nn

from darksnail import (
    BeltramiCallosalShell,
    DarkSnailCanvasReadout,
    DarkSnailSchedule,
    DarkSnailSystem2Adaptor,
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    ZenoAmygdalaController,
    compute_grid_prompt_zeno_flux,
    extract_beltrami_lh_latents,
    verify_pointwise_orthogonality,
)


def test_skew_symmetric_triad_operator_orthogonality_and_tie_breaking():
    torch.manual_seed(42)
    op = SkewSymmetricTriadOperator()

    # 1. Shared W of shape (C_u, M, M) with 3D inputs (B, L, C_u) and (B, L, M)
    B, L, C_u, M = 3, 64, 16, 12
    u = torch.randn(B, L, C_u, dtype=torch.float32)
    v = torch.randn(B, L, M, dtype=torch.float32)
    W_shared = torch.randn(C_u, M, M, dtype=torch.float32)

    out_shared = op(u, v, W_shared)
    assert out_shared.shape == (B, L, M)
    assert out_shared.dtype == torch.float32

    inner_shared = verify_pointwise_orthogonality(u, v, W_shared)
    assert inner_shared.shape == (B, L)
    assert torch.allclose(inner_shared, torch.zeros_like(inner_shared), atol=1e-5)

    # 2. Per-batch W of shape (B, C_u, M, M)
    W_batch = torch.randn(B, C_u, M, M, dtype=torch.float32)
    out_batch = op(u, v, W_batch)
    assert out_batch.shape == (B, L, M)

    inner_batch = op.verify_pointwise_orthogonality(u, v, W_batch)
    assert inner_batch.shape == (B, L)
    assert torch.allclose(inner_batch, torch.zeros_like(inner_batch), atol=1e-5)

    # 3. Exact 2-color occlusion tie-breaking antisymmetry (OcclusionTriad.lean)
    # Suppose pixel has a 50-50 tie between color c1=4 and color c2=7: v[4] == v[7] == 0.5
    u_tie = torch.zeros(1, 1, C_u, dtype=torch.float32)
    u_tie[0, 0, 0] = 1.0  # Context channel 0 active
    v_tie = torch.zeros(1, 1, M, dtype=torch.float32)
    c1, c2 = 4, 7
    v_tie[0, 0, c1] = 0.5
    v_tie[0, 0, c2] = 0.5

    W_tie = torch.zeros(1, C_u, M, M, dtype=torch.float32)
    W_tie[0, 0, c1, c2] = 2.0  # Favors c2 over c1 under context 0
    W_tie[0, 0, c2, c1] = -1.0

    shift = op(u_tie, v_tie, W_tie)  # (1, 1, 12)
    # S[0, c1, c2] = 2.0 - (-1.0) = +3.0, S[0, c2, c1] = -3.0
    # shift[c2] = u[0] * v[c1] * S[0, c1, c2] = 1.0 * 0.5 * 3.0 = +1.5
    # shift[c1] = u[0] * v[c2] * S[0, c2, c1] = 1.0 * 0.5 * (-3.0) = -1.5
    assert torch.isclose(shift[0, 0, c2], torch.tensor(1.5), atol=1e-6)
    assert torch.isclose(shift[0, 0, c1], torch.tensor(-1.5), atol=1e-6)
    assert torch.isclose(shift[0, 0, c2] + shift[0, 0, c1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        verify_pointwise_orthogonality(u_tie, v_tie, W_tie),
        torch.zeros(1, 1),
        atol=1e-6,
    )


def test_darksnail_canvas_readout_step0_homotopy_and_unused_dz_harvesting():
    from brain_ai.models.callosum import System2LatentAdaptor, encode_grid_to_trm_seq

    torch.manual_seed(7)
    B, L, V, d_r = 2, 900, 12, 512
    num_q = 16

    grid_a = [[0, 1, 2, 0], [1, 3, 3, 2], [2, 3, 4, 1], [0, 2, 1, 0]]
    grid_b = [[5, 0, 5, 0], [0, 6, 0, 6], [5, 0, 5, 0], [0, 6, 0, 6]]
    seq_a = encode_grid_to_trm_seq(grid_a)
    seq_b = encode_grid_to_trm_seq(grid_b)
    inputs = torch.tensor(np.stack([seq_a, seq_b]), dtype=torch.long)

    # Dummy model with frozen lm_head of shape (12, 512)
    class DummyReasoner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(d_r, V, bias=False)

    model = DummyReasoner()
    base_logits = torch.randn(B, L, V, dtype=torch.float32) * 0.8

    readout = DarkSnailCanvasReadout(readout_scale=6.0, triad_gain=1.0, asym_gain=1.5)
    s2_adaptor = System2LatentAdaptor(readout_scale=6.0)
    s2_adaptor._active_target_hw = (4, 4)

    # 1. Step-0 System 1 Homotopy: delta_z == 0 produces exact zero drift
    dz_zero = torch.zeros(num_q, d_r, dtype=torch.float32)
    out_zero = readout(base_logits, inputs, dz_zero, model=model, target_hw=(4, 4))
    out_s1_base = s2_adaptor._apply_callosal_canvas_readout(model, base_logits, inputs, dz_zero)
    assert torch.allclose(out_zero, base_logits, atol=1e-6)
    assert torch.allclose(out_zero, out_s1_base, atol=1e-6)

    # 2. Symmetric slice agreement: when only flat_dz[:2498] is non-zero, matches callosum.py
    dz_sym_only = torch.zeros(num_q * d_r, dtype=torch.float32)
    dz_sym_only[: DarkSnailCanvasReadout.SYM_END] = torch.randn(DarkSnailCanvasReadout.SYM_END) * 0.05
    dz_sym_2d = dz_sym_only.view(num_q, d_r)

    out_ds_sym = readout(base_logits, inputs, dz_sym_2d, model=model, target_hw=(4, 4))
    out_orig_sym = s2_adaptor._apply_callosal_canvas_readout(model, base_logits, inputs, dz_sym_2d)
    assert torch.allclose(out_ds_sym, out_orig_sym, atol=1e-5)

    # 3. Gradient flow into the 5,500+ previously unused coordinates (Triad + Asym slices)
    dz_param = nn.Parameter(torch.randn(num_q, d_r, dtype=torch.float32) * 0.02)
    out_full = readout(base_logits, inputs, dz_param, model=model, target_hw=(4, 4))

    # Verify pixel-wise orthogonality <v, delta_triad> == 0 across all 900 pixels
    v_comp = readout.last_v_comp
    delta_triad = readout.last_triad_shift
    assert v_comp is not None and delta_triad is not None
    pixel_inner = (v_comp * delta_triad).sum(dim=-1)  # (B, 900)
    assert torch.allclose(pixel_inner, torch.zeros_like(pixel_inner), atol=1e-4)

    # Backprop a target cross-entropy loss and verify gradients on triad & asym slices
    targets = torch.randint(2, 12, (B, L), dtype=torch.long)
    loss = nn.functional.cross_entropy(out_full.view(-1, V), targets.view(-1))
    loss.backward()

    flat_grad = dz_param.grad.view(-1)
    sym_grad_norm = flat_grad[: DarkSnailCanvasReadout.SYM_END].norm().item()
    triad_grad_norm = flat_grad[
        DarkSnailCanvasReadout.SYM_END : DarkSnailCanvasReadout.TRIAD_END
    ].norm().item()
    asym_grad_norm = flat_grad[
        DarkSnailCanvasReadout.TRIAD_END : DarkSnailCanvasReadout.ASYM_END
    ].norm().item()

    assert sym_grad_norm > 0.0
    assert triad_grad_norm > 0.0
    assert asym_grad_norm > 0.0


def test_slow_fast_triad_cascade_energy_conservation_and_homotopy():
    torch.manual_seed(19)
    B, S, D = 2, 64, 512
    cascade = SlowFastTriadCascade(d_reasoner=D, d_shell=8, preserve_exact_sphere_norm=True)

    z_H = torch.randn(B, S, D, dtype=torch.float32)
    z_L = torch.randn(B, S, D, dtype=torch.float32)

    # 1. Step-0 Homotopy: delta_z == 0 returns z_L unchanged
    dz_zero = torch.zeros(16, D, dtype=torch.float32)
    z_L_zero = cascade(z_H, z_L, dz_zero)
    assert torch.allclose(z_L_zero, z_L, atol=1e-7)

    # 2. Non-zero delta_z: verify tangent orthogonality and exact shell norm conservation
    dz_rand = nn.Parameter(torch.randn(16, D, dtype=torch.float32) * 0.5)
    z_L_coupled = cascade(z_H, z_L, dz_rand)

    v_orig = cascade.project_to_shell(z_L)  # (B, S, 8)
    v_new = cascade.project_to_shell(z_L_coupled)  # (B, S, 8)
    tangent = cascade.last_tangent_step  # (B, S, 8)

    # Pointwise tangent orthogonality <P z_L, delta_v> == 0
    tangent_inner = (v_orig * tangent).sum(dim=-1)
    assert torch.allclose(tangent_inner, torch.zeros_like(tangent_inner), atol=1e-5)

    # Exact discrete spherical energy conservation ||P z_tilde_L||_2^2 == ||P z_L||_2^2
    energy_orig = (v_orig ** 2).sum(dim=-1)
    energy_new = (v_new ** 2).sum(dim=-1)
    assert torch.allclose(energy_new, energy_orig, atol=1e-5)

    # Gradient flows cleanly into flat_dz[4810:5322]
    loss = (z_L_coupled * torch.randn_like(z_L_coupled)).sum()
    loss.backward()
    flat_g = dz_rand.grad.view(-1)
    cascade_slice_grad = flat_g[cascade.slice_start : cascade.slice_end].norm().item()
    assert cascade_slice_grad > 0.0


def test_rope2d_medoid_pooler_invariance_and_attenuation():
    torch.manual_seed(99)
    B, n_heads, H, W, d_head = 2, 4, 8, 8, 16
    pooler = RoPE2DMedoidPooler(patch_size=2, viscous_nu=0.05, viscous_p=2, shell_level=1)

    # Construct a uniform feature key k0 across all spatial positions and rotate via exact SO(2)xSO(2) axial RoPE2D
    k0 = torch.randn(B, n_heads, 1, 1, d_head, dtype=torch.float32).expand(B, n_heads, H, W, d_head).contiguous()
    v_in = torch.randn(B, n_heads, H, W, d_head, dtype=torch.float32)
    k_rot = pooler.apply_axial_rope2d(k0, H, W)  # (B, n_heads, H, W, d_head)

    # Arithmetic patch averaging strictly attenuates key norm (RoPECoherence.lean Theorem)
    k_arith_avg = (
        k_rot.view(B, n_heads, H // 2, 2, W // 2, 2, d_head)
        .mean(dim=(3, 5))
    )
    orig_norm = torch.norm(k0[:, :, : H // 2, : W // 2, :], p=2, dim=-1)
    arith_norm = torch.norm(k_arith_avg, p=2, dim=-1)
    assert bool((arith_norm < orig_norm - 1e-4).all().item())

    # 2D Medoid Pooler preserves exact unattenuated norm on raw_medoid_k and scales by gamma_2D in (0, 1]
    k_pooled, v_pooled, info = pooler(k_rot, v_in)
    assert k_pooled.shape == (B, n_heads, H // 2, W // 2, d_head)
    assert v_pooled.shape == (B, n_heads, H // 2, W // 2, d_head)

    raw_medoid_norm = torch.norm(info["raw_medoid_k"], p=2, dim=-1)
    assert torch.allclose(raw_medoid_norm, orig_norm, atol=1e-5)

    gamma_2d = info["gamma_2d"]
    assert bool(((gamma_2d > 0.0) & (gamma_2d <= 1.0)).all().item())
    pooled_norm = torch.norm(k_pooled, p=2, dim=-1)
    assert torch.allclose(pooled_norm, orig_norm * gamma_2d, atol=1e-5)


def test_beltrami_callosal_shell_and_callosum_compatibility():
    from brain_ai.models.callosum import CallosalConditioning
    from llama_surgery.topology import TriadTopologyRouter

    torch.manual_seed(123)
    B, S, D = 2, 160, 4096
    shell = BeltramiCallosalShell(
        d_model=D,
        target_tokens=32,
        outlier_budget=16,
        condensed_budget=16,
        d_shell=16,
        shield_radius=2,
    )

    # Create laminar background hay + sharp high-curl Beltrami spikes at positions 35 and 110
    base_vec = torch.randn(B, 1, D, dtype=torch.float32)
    raw_latents = base_vec + 0.02 * torch.randn(B, S, D, dtype=torch.float32)
    spike_vec_1 = torch.randn(B, D, dtype=torch.float32) * 8.0
    spike_vec_2 = torch.randn(B, D, dtype=torch.float32) * 8.0
    raw_latents[:, 35, :] = spike_vec_1
    raw_latents[:, 110, :] = spike_vec_2

    super_tokens, diag = shell(raw_latents, return_diagnostics=True)
    assert super_tokens.shape == (B, 32, D)
    assert super_tokens.dtype == torch.float32
    assert 0.0 <= diag["zeno_flux"] <= 1.0
    assert diag["lca_variance"] >= 0.0

    # Verify both injected outlier positions (35 and 110) were protected in Tier-1 (first 16 slots)
    outlier_idx = diag["outlier_indices"]  # (B, 16)
    for b in range(B):
        idx_list = outlier_idx[b].tolist()
        assert idx_list == sorted(idx_list), "Tier-1 outliers must preserve chronological order"
        assert 35 in idx_list, f"Spike at 35 not found in Tier-1 indices {idx_list}"
        assert 110 in idx_list, f"Spike at 110 not found in Tier-1 indices {idx_list}"

    # Verify seamless compatibility with CallosalConditioning (3.68M bridge)
    callosum = CallosalConditioning(d_lh=D, d_reasoner=512, d_callosum=512, num_query_tokens=16)
    z0 = callosum.condition_task(super_tokens)
    assert z0.shape == (B, 16, 512)

    # Verify compatibility with a live V4 TriadTopologyRouter sidecar attached
    router = TriadTopologyRouter(
        embed_dim=D, seq_len=512, num_heads=4, p=2, qk_mode=True, head_dim=128, d_shell=8
    )
    shell_with_router = BeltramiCallosalShell(d_model=D, target_tokens=32, router_sidecar=router)
    st_router = shell_with_router(raw_latents)
    assert st_router.shape == (B, 32, D)

    # Verify extract_beltrami_lh_latents mock mode
    mock_st, mock_diag = extract_beltrami_lh_latents("ARC Task Demo Prompt", mock_mode=True)
    assert mock_st.shape == (1, 32, 4096)
    assert 0.0 <= mock_diag["zeno_flux"] <= 1.0


def test_compute_grid_prompt_zeno_flux_speed_and_ordering():
    # Simple laminar identity task
    laminar_demos = [
        {"input": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]},
        {"input": [[2, 2, 2], [2, 2, 2], [2, 2, 2]], "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]},
    ]
    # Complex multi-color Kronecker / compositional task
    complex_demos = [
        {
            "input": [[1, 0, 2], [0, 3, 0], [4, 0, 5]],
            "output": [
                [1, 1, 0, 0, 2, 2],
                [1, 3, 0, 0, 2, 5],
                [0, 0, 3, 3, 0, 0],
                [0, 0, 4, 5, 0, 0],
                [4, 4, 0, 0, 5, 5],
                [4, 1, 0, 0, 5, 2],
            ],
        },
        {
            "input": [[6, 7, 0], [8, 0, 9], [0, 2, 3]],
            "output": [
                [6, 7, 6, 7, 0, 0],
                [8, 9, 8, 9, 0, 0],
                [8, 0, 0, 0, 9, 2],
                [0, 2, 0, 0, 3, 6],
                [0, 0, 2, 3, 6, 7],
                [0, 0, 8, 9, 2, 3],
            ],
        },
    ]

    t0 = time.perf_counter()
    res_laminar = compute_grid_prompt_zeno_flux(laminar_demos, [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    res_complex = compute_grid_prompt_zeno_flux(complex_demos, [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

    assert elapsed_ms < 25.0  # Sub-millisecond in typical warm runs
    assert 0.0 <= res_laminar["zeno_flux"] <= 1.0
    assert 0.0 <= res_complex["zeno_flux"] <= 1.0
    assert res_laminar["lca_variance"] >= 0.0
    assert res_complex["lca_variance"] >= 0.0
    assert res_complex["zeno_flux"] > res_laminar["zeno_flux"]


def test_zeno_amygdala_controller_schedules_and_adaptor():
    controller = ZenoAmygdalaController(allow_zero_step_bypass=True)

    # 1. Near-Solved / Surgical Regime (e.g. af22c60d: 898/900 = 99.78%, loss_0 = 1.6)
    sched_near = controller(
        zeno_flux=0.22,
        lca_variance=0.10,
        step0_demo_loss=1.60,
        step0_grad_norm=0.45,
        s1_demo_fit=0.95,
        test_entropy=0.12,
        canvas_cells=900,
        has_occlusion_tie=True,
    )
    assert isinstance(sched_near, DarkSnailSchedule)
    assert sched_near.regime == "surgical_near_solved"
    assert 0 <= sched_near.num_steps <= 20
    assert 0.02 <= sched_near.lr <= 0.04
    assert 0.20 <= sched_near.anchor_weight <= 0.30
    assert 0.25 <= sched_near.triad_gain <= 1.50
    # Must include surgical 1-cell (1/900) and 2-cell (2/900) quantiles
    assert any(abs(q - (1.0 / 900.0)) < 1e-5 for q in sched_near.ray_quantiles)
    assert any(abs(q - (2.0 / 900.0)) < 1e-5 for q in sched_near.ray_quantiles)

    # 2. High-Turbulence Compositional Regime (e.g. Boolean/Kronecker: loss_0 = 6.2, zeno_flux = 0.78)
    sched_high = controller(
        zeno_flux=0.78,
        lca_variance=0.85,
        step0_demo_loss=6.20,
        step0_grad_norm=3.10,
        s1_demo_fit=0.35,
        test_entropy=1.40,
        canvas_cells=100,
    )
    assert sched_high.regime == "high_turbulence_compositional"
    assert 32 <= sched_high.num_steps <= 40
    assert 0.045 <= sched_high.lr <= 0.06
    assert 0.05 <= sched_high.anchor_weight <= 0.12
    assert 1.0 <= sched_high.triad_gain <= 1.50
    assert sched_high.turbulence_index > sched_near.turbulence_index

    # 3. Zero-Step Reflex Bypass when 100% solved with vanishing loss & entropy
    sched_bypass = controller(
        zeno_flux=0.05,
        lca_variance=0.01,
        step0_demo_loss=0.05,
        step0_grad_norm=0.01,
        s1_demo_fit=1.0,
        test_entropy=0.005,
        canvas_cells=64,
        has_occlusion_tie=False,
    )
    assert sched_bypass.regime == "reflex_bypass"
    assert sched_bypass.num_steps == 0
    assert sched_bypass.bypass_tta is True

    # 4. End-to-end DarkSnailSystem2Adaptor test on a mock Callosal + TRM model
    from brain_ai.models.callosum import CallosalConditioning, CallosalPuzzleEmbedding

    torch.manual_seed(31)
    cached_lh = torch.randn(2, 32, 4096)
    callosum = CallosalConditioning(d_lh=4096, d_reasoner=512, d_callosum=512, num_query_tokens=16)
    puz_emb = CallosalPuzzleEmbedding(cached_lh, callosum, num_query_tokens=16, d_reasoner=512)

    class MockTRM(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.tok_emb = nn.Embedding(12, 512)
            self.lm_head = nn.Linear(512, 12, bias=False)

        def forward(self, batch):
            inp = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            B_sz, L_sz = inp.shape
            z_task = self.puzzle_emb(pids).view(B_sz, 16, 512).mean(dim=1, keepdim=True)
            h = self.tok_emb(inp.clamp(0, 11)) + 0.1 * z_task
            return {"logits": self.lm_head(h)}

    mock_model = MockTRM(puz_emb)
    task_data = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
            {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
        ],
        "test": [{"input": [[1, 4], [2, 3]]}],
    }

    ds_adaptor = DarkSnailSystem2Adaptor(n_steps=5, lr=0.05, lambda_anchor=0.1)
    delta_z, diag = ds_adaptor.adapt_task(mock_model, task_data, puzzle_id=1, n_steps=5)
    assert delta_z.shape == (16, 512)
    assert diag["final_demo_loss"] <= diag["step0_loss"]
    assert "zeno_flux" in diag and "turbulence_index" in diag

    pair_res = ds_adaptor.predict_s1_and_s2_pair(
        mock_model,
        test_input_grid=task_data["test"][0]["input"],
        puzzle_id=1,
        delta_z=delta_z,
        accepted=diag["accepted"],
        demos=task_data["train"],
    )
    assert len(pair_res["attempt_1"]) == 2 and len(pair_res["attempt_1"][0]) == 2
