# SPDX-License-Identifier: CC0-1.0
"""
Unit & Integration Verification Suite for `darksnail.hooks`.

Verifies the four composable Mechanistic Interpretability & Norm-Preserving Spherical
Lie-Algebraic Steering Hook pillars and `DarkSnailHookSession`:
1. `DarkSnailHookSession` backbone discovery (`model.model.layers` vs `model.language_model.layers`
   / `model.model.language_model.layers`) and deterministic `handle.remove()` cleanup on both
   normal context exit and raised exceptions.
2. `BeltramiShellProbeHook`: Non-destructive residual telemetry (`output is input`), Tier-1
   Beltrami exterior-curl spike preservation (`max cosine = 1.0000`), deterministic Haar-QR
   isometric bridge (`P_iso @ P_iso.T = I_4096`, `seed=20260404`), and `detokenize_tier1`
   BPE subword welding.
3. `SphericalLieSteeringHook`: Pointwise tangent orthogonality (`|<v, dv_perp>| <= 1e-8`),
   exact radial sphere retraction (`|||h_steered||_2 - ||h||_2| <= 1e-6`), and tuple output
   preservation.
4. `SphericalLieSteeringHook` Lie-algebraic utilities: Skew-symmetrization `A = 0.5 * (M - M^T)`,
   rank-2 wedge generator `z_{t+1} z_t^T - z_t z_{t+1}^T in so(M)`, and commutator curvature
   `kappa(A_1, A_2) = ||[A_1, A_2]||_F / (||A_1||_F ||A_2||_F)`.
5. `AllButTheTopGramHook`: Proposition 5.1 anisotropic cone debiasing (per-prompt token mean
   removal + power-iteration top-PC projection onto `S^{d-1}`), span-to-span Callosal Gram
   resonance `R_l(a_k)`, single-token digit-logit fusion `alpha_callosal`, and One-Sided ReLU
   Callosal Margin `max(0, P_callosal(a_k) - 1/K)`.
6. `PRoPESubspaceHook`: Proportional RoPE (`partial_rotary_factor = 0.25`) `q_nope` / `k_nope`
   (`384`-D out of `512`-D) subspace extraction across `contiguous_prefix` and `hf_split_half`
   layouts, translation-invariant Beltrami salience `S_Beltrami(t)`, and zero mutation of
   `attention_k_eq_v=True` tensors.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from darksnail import (
    AllButTheTopGramHook,
    BeltramiShellProbeHook,
    DarkSnailHookSession,
    PRoPESubspaceHook,
    SphericalLieSteeringHook,
    build_isometric_haar_qr_bridge,
    discover_decoder_layers,
    resolve_default_hook_layer,
)
from darksnail.gemma4 import apply_gemma4_prope, build_gemma4_prope_cache


class _DummyDecoderLayer(nn.Module):
    def __init__(self, d_model: int, return_tuple: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.return_tuple = return_tuple

    def forward(self, x: torch.Tensor):
        if self.return_tuple:
            return (x, {"kv_cache": True})
        return x


class _DummyLlamaBackbone(nn.Module):
    def __init__(self, num_layers: int = 32, d_model: int = 128) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_DummyDecoderLayer(d_model=d_model, return_tuple=True) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.model.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class _DummyGemma4Backbone(nn.Module):
    def __init__(self, num_layers: int = 60, d_model: int = 128, nested_vlm: bool = False) -> None:
        super().__init__()
        layers = nn.ModuleList(
            [_DummyDecoderLayer(d_model=d_model, return_tuple=True) for _ in range(num_layers)]
        )
        if nested_vlm:
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = layers
        else:
            self.language_model = nn.Module()
            self.language_model.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = discover_decoder_layers(self)
        h = x
        for layer in layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class _DummyBpeTokenizer:
    """Mock BPE tokenizer testing subword continuation welding (`snail` + `cada` -> `snailcada`)."""

    def __init__(self) -> None:
        self._id_to_piece = {
            10: " The",
            11: " secret",
            12: " word",
            13: " is",
            14: " snail",
            15: "cada",
            16: ".",
        }

    def decode(self, ids) -> str:
        if isinstance(ids, int):
            return self._id_to_piece.get(ids, "")
        return "".join(self._id_to_piece.get(int(i), "") for i in ids)


def test_hook_session_layer_discovery_and_cleanup_on_exception():
    llama_model = _DummyLlamaBackbone(num_layers=32, d_model=64)
    gemma_model = _DummyGemma4Backbone(num_layers=60, d_model=64, nested_vlm=False)
    gemma_vlm_model = _DummyGemma4Backbone(num_layers=60, d_model=64, nested_vlm=True)

    assert len(discover_decoder_layers(llama_model)) == 32
    assert len(discover_decoder_layers(gemma_model)) == 60
    assert len(discover_decoder_layers(gemma_vlm_model)) == 60

    assert resolve_default_hook_layer(llama_model) == 16
    assert resolve_default_hook_layer(gemma_model) == 29
    assert resolve_default_hook_layer(gemma_vlm_model) == 29
    # Legacy layer_idx=16 on 60-layer Gemma 4 auto-snaps to Global Layer 29
    assert resolve_default_hook_layer(gemma_model, layer_idx=16, snap_gemma4_global=True) == 29

    probe = BeltramiShellProbeHook(
        outlier_budget=4,
        condensed_budget=4,
        target_tokens=8,
        isometric_target_dim=64,
    )
    steerer = SphericalLieSteeringHook(dt=0.15, subspace_dim=16)

    x = torch.randn(1, 24, 64)
    with DarkSnailHookSession(llama_model, hooks=[probe, (20, steerer)]) as session:
        assert probe.resolved_layer_idx == 16
        assert steerer.resolved_layer_idx == 20
        assert len(session.handles) == 2
        assert len(llama_model.model.layers[16]._forward_hooks) == 1
        assert len(llama_model.model.layers[20]._forward_hooks) == 1
        _ = llama_model(x)

    # Normal context exit must strip all registered hooks
    assert len(llama_model.model.layers[16]._forward_hooks) == 0
    assert len(llama_model.model.layers[20]._forward_hooks) == 0

    # Exception inside context block must also strip all registered hooks
    with pytest.raises(RuntimeError, match="simulated forward failure"):
        with DarkSnailHookSession(gemma_vlm_model) as session:
            session.register(probe, layer_idx=None)
            assert probe.resolved_layer_idx == 29
            assert len(gemma_vlm_model.model.language_model.layers[29]._forward_hooks) == 1
            raise RuntimeError("simulated forward failure")

    assert len(gemma_vlm_model.model.language_model.layers[29]._forward_hooks) == 0


def test_beltrami_shell_probe_hook_spike_recovery_and_haar_qr_isometry():
    # 1. Verify deterministic row-orthonormal Haar-QR bridge (5376 -> 4096)
    p_iso = build_isometric_haar_qr_bridge(
        d_in=5376,
        d_out=4096,
        seed=20260404,
        device=torch.device("cpu"),
    )
    assert p_iso.shape == (4096, 5376)
    gram = torch.matmul(p_iso, p_iso.transpose(0, 1))
    ident = torch.eye(4096, dtype=torch.float32)
    max_ortho_err = (gram - ident).abs().amax().item()
    assert max_ortho_err <= 1e-5, f"Expected P_iso @ P_iso.T == I_4096, got {max_ortho_err}"

    # 2. Verify non-destructive forward pass and 1.0000 synthetic needle spike preservation
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260404)
        B, S, D = 1, 64, 128
        # Smooth low-rank background + high-curl orthogonal synthetic needle at tokens 31 and 32
        t_grid = torch.linspace(0.0, 1.0, S, dtype=torch.float32).unsqueeze(-1)
        u_bg = torch.randn(1, 1, D)
        v_bg = torch.randn(1, 1, D)
        hidden = torch.cos(t_grid) * u_bg + torch.sin(t_grid) * v_bg

        needle_vec = torch.randn(D, dtype=torch.float32) * 25.0
        needle_vec_2 = torch.randn(D, dtype=torch.float32) * 24.0
        hidden[0, 31, :] = needle_vec
        hidden[0, 32, :] = needle_vec_2

    prompt_ids = [10] * S
    prompt_ids[31] = 14  # " snail"
    prompt_ids[32] = 15  # "cada"

    layer = _DummyDecoderLayer(d_model=D, return_tuple=True)
    probe = BeltramiShellProbeHook(
        outlier_budget=8,
        condensed_budget=8,
        target_tokens=16,
        isometric_target_dim=D,
        isometric_seed=20260404,
    ).bind_input_ids(torch.tensor([prompt_ids], dtype=torch.long))

    handle = layer.register_forward_hook(probe)
    try:
        out = layer(hidden)
    finally:
        handle.remove()

    # Probe hook MUST return the unmodified output object
    assert out[0] is hidden
    telem = probe.last_telemetry
    assert telem
    assert telem["condensed_shell"].shape == (1, 16, D)
    assert telem["tier1_outliers_native"].shape == (1, 8, D)
    assert telem["tier1_outlier_indices"].shape == (1, 8)
    assert telem["tier2_medoid_indices"].shape == (1, 8)
    assert 0.0 <= telem["zeno_flux_phi_lh"] <= 1.0
    assert telem["lca_variance"] >= 0.0

    # Needle token 31 must be captured in Tier-1 with exact 1.0000 cosine similarity
    max_cos = probe.compute_spike_preservation_cosine(needle_vec)
    assert abs(max_cos - 1.0) <= 1e-6, f"Expected 1.0000 spike recovery, got {max_cos}"
    assert 31 in telem["tier1_outlier_indices"][0].tolist()
    assert 32 in telem["tier1_outlier_indices"][0].tolist()

    # Verify detokenize_tier1 BPE subword welding (" snail" + "cada" -> "snailcada")
    tokenizer = _DummyBpeTokenizer()
    welded = probe.detokenize_tier1(tokenizer, top_k=2)
    assert "snailcada" in welded


def test_spherical_lie_steering_hook_orthogonality_and_norm_conservation():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77)
        B, S, D = 2, 16, 96
        hidden = torch.randn(B, S, D, dtype=torch.float32) * 3.5

    layer = _DummyDecoderLayer(d_model=D, return_tuple=True)
    steerer = SphericalLieSteeringHook(
        subspace_dim=32,
        dt=0.25,
        steps=1,
        seed=20260404,
    )

    handle = layer.register_forward_hook(steerer)
    try:
        out = layer(hidden)
    finally:
        handle.remove()

    assert isinstance(out, tuple) and len(out) == 2
    h_steered = out[0]
    assert h_steered.shape == hidden.shape

    telem = steerer.last_telemetry
    assert telem
    assert telem["max_tangent_orthogonality_error"] <= 1e-8, (
        f"Expected |<v, dv_perp>| <= 1e-8, got {telem['max_tangent_orthogonality_error']}"
    )
    assert telem["max_subspace_norm_drift"] <= 1e-6, (
        f"Expected subspace norm drift <= 1e-6, got {telem['max_subspace_norm_drift']}"
    )
    assert telem["max_full_norm_drift"] <= 1e-6, (
        f"Expected |||h_steered||_2 - ||h||_2| <= 1e-6, got {telem['max_full_norm_drift']}"
    )

    # Direct norm check across all (B, S) positions
    orig_norms = hidden.norm(p=2, dim=-1)
    steered_norms = h_steered.norm(p=2, dim=-1)
    assert (steered_norms - orig_norms).abs().amax().item() <= 1e-6
    assert not torch.allclose(h_steered, hidden, atol=1e-4)


def test_spherical_lie_generators_and_commutator_curvature():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        M = 24
        raw_mat = torch.randn(M, M, dtype=torch.float32)
        a_skew = SphericalLieSteeringHook.build_so_generator(raw_mat)
        assert (a_skew + a_skew.transpose(-1, -2)).abs().amax().item() <= 1e-7

        z_t = torch.randn(M, dtype=torch.float32)
        z_next = torch.randn(M, dtype=torch.float32)
        omega = SphericalLieSteeringHook.build_wedge_generator(z_t, z_next)
        assert omega.shape == (M, M)
        assert (omega + omega.transpose(-1, -2)).abs().amax().item() <= 1e-7

        # Commuting generators must have zero commutator curvature
        kappa_self = SphericalLieSteeringHook.commutator_curvature(omega, omega)
        assert abs(kappa_self) <= 1e-7

        # Independent non-commuting generators in so(M) must have strictly positive curvature
        z_alt = torch.randn(M, dtype=torch.float32)
        omega_2 = SphericalLieSteeringHook.build_wedge_generator(z_next, z_alt)
        kappa_cross = SphericalLieSteeringHook.commutator_curvature(omega, omega_2)
        assert kappa_cross > 1e-2

        # Verify online generator EMA update remains skew-symmetric
        steerer = SphericalLieSteeringHook(subspace_dim=M)
        updated_gen = steerer.update_online_generator(z_t, z_next, generator_id=0, lr=0.5)
        assert (updated_gen + updated_gen.t()).abs().amax().item() <= 1e-7


def test_all_but_the_top_gram_hook_cone_debiasing_and_relu_margin():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        B, S, D = 1, 32, 128
        # Orthonormal basis directions for cone axis, top anisotropic PC, and 3 candidate semantics
        q_basis, _ = torch.linalg.qr(torch.randn(D, 6, dtype=torch.float32), mode="reduced")
        common_cone_axis = q_basis[:, 0].view(1, 1, D) * 6.0
        dominant_pc = q_basis[:, 1].view(1, 1, D) * 3.0
        dir_opt0 = q_basis[:, 2].view(1, 1, D)
        dir_opt1 = q_basis[:, 3].view(1, 1, D)
        dir_opt2 = q_basis[:, 4].view(1, 1, D)

        t_idx = torch.arange(S, dtype=torch.float32).view(1, S, 1)
        pc_loadings = torch.cos(t_idx * (2.0 * 3.14159265 / float(S)))

        noise = torch.randn(B, S, D) * 0.08
        hidden = common_cone_axis + pc_loadings * dominant_pc + noise
        # Candidate spans: opt0=[4:8], opt1=[10:14], opt2=[16:20], goal=[24:28] (aligned with opt1)
        hidden[:, 4:8, :] += 1.20 * dir_opt0
        hidden[:, 10:14, :] += 1.20 * dir_opt1
        hidden[:, 16:20, :] += 1.20 * dir_opt2
        hidden[:, 24:28, :] += 1.20 * dir_opt1
        # Zero-mean balancing spans so dir_opt1 does not shift the global centroid
        hidden[:, 20:24, :] -= 1.20 * dir_opt1

    gram_hook = AllButTheTopGramHook(
        goal_span=(24, 28),
        candidate_spans=[(4, 8), (10, 14), (16, 20)],
        num_pcs=1,
        alpha_callosal=0.65,
        temperature=0.10,
    ).bind_digit_logits(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

    layer = _DummyDecoderLayer(d_model=D, return_tuple=True)
    handle = layer.register_forward_hook(gram_hook)
    try:
        out = layer(hidden)
    finally:
        handle.remove()

    assert out[0] is hidden
    res = gram_hook.last_telemetry
    assert res
    assert res["raw_mean_cosine"] > 0.65
    assert abs(res["debiased_mean_cosine"]) < 0.15
    assert res["unit_sphere_norm_error"] <= 1e-6

    # Option 1 (planted in goal span [0:12]) must achieve highest Callosal Gram resonance
    resonances = res["gram_resonances"]
    p_callosal = res["p_callosal"]
    relu_margin = res["relu_callosal_margin"]
    assert resonances.shape == (3,)
    assert int(resonances.argmax().item()) == 1
    assert res["best_candidate_idx"] == 1
    assert abs(p_callosal.sum().item() - 1.0) <= 1e-6
    assert (relu_margin >= 0.0).all()
    assert relu_margin[1].item() > 0.0
    assert relu_margin[0].item() == 0.0
    assert relu_margin[2].item() == 0.0


def test_prope_subspace_hook_nope_extraction_and_k_eq_v_safety():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        B, H_kv, S, D = 2, 4, 24, 512
        # Construct positionally constant semantic key tensor across sequence positions
        k_semantic = torch.randn(B, H_kv, 1, D, dtype=torch.float32).expand(B, H_kv, S, D).clone()
        # Spike token 11 in the semantic subspace
        k_semantic[:, :, 11, :] = torch.randn(B, H_kv, D, dtype=torch.float32) * 8.0

    for layout in ("contiguous_prefix", "hf_split_half"):
        cos_2d, sin_2d = build_gemma4_prope_cache(
            seq_len=S,
            head_dim=D,
            theta=1_000_000.0,
            partial_rotary_factor=0.25,
            device=k_semantic.device,
            layout=layout,
        )
        k_prope = apply_gemma4_prope(k_semantic, cos_2d, sin_2d, layout=layout)
        k_prope_snapshot = k_prope.clone()

        prope_hook = PRoPESubspaceHook(
            head_dim=D,
            partial_rotary_factor=0.25,
            layout=layout,
        )
        proj = nn.Identity()
        handle = proj.register_forward_hook(prope_hook)
        try:
            out = proj(k_prope)
        finally:
            handle.remove()

        # Zero mutation of attention_k_eq_v=True key/value stream
        assert out is k_prope
        assert torch.equal(k_prope, k_prope_snapshot)

        telem = prope_hook.last_telemetry
        assert telem
        assert telem["attention_k_eq_v_preserved"] is True
        assert telem["value_stream_mutated"] is False
        assert telem["k_rope"].shape == (B, H_kv, S, 128)
        assert telem["k_nope"].shape == (B, H_kv, S, 384)
        assert telem["beltrami_salience"].shape == (B, S)

        # Unrotated 384-D k_nope subspace must be invariant across non-spiked positions [0..9]
        nope_sub = telem["k_nope"]
        pos_drift = (nope_sub[:, :, 1:10, :] - nope_sub[:, :, 0:9, :]).abs().amax().item()
        assert pos_drift == 0.0

        # Spiked token 11 must have top Beltrami salience in the unrotated NoPE subspace
        salience = telem["beltrami_salience"]
        top_tokens = salience.topk(k=2, dim=-1).indices
        for b in range(B):
            assert 11 in top_tokens[b].tolist()
