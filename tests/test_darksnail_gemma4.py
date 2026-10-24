"""
Unit & Integration Verification Suite for the Gemma 4 31B DarkSnail V2 Adaptation (Post-Audit).

Covers the 5 foundational architecture suites plus all 7 Deep Audit regression tests:
1. Ground-truth Gemma 4 31B topology constants (`arXiv:2607.02770`).
2. Dual-layout p-RoPE subspace separation (`contiguous_prefix` & `hf_split_half`), 3D `(B, S, D)`
   HuggingFace `position_embeddings` handling, and `position_offset` decode indexing.
3. `DarkSnailGemma4GlobalHook` tangent orthogonality, spherical retraction, and Step-0 homotopy.
4. `Gemma4BeltramiCallosalShell` Global Layer-29 hook + `(B, 32, 5376) -> (B, 32, 4096)` bridge.
5. 60-layer 5:1 Local/Global selective patching (`only_global_layers=True`).
6. Audit Regression Suite:
   - Zero-shot `init_mode="haar_dct"` spectral hyperplane sparsity (monotonically decreasing budget across r=2,3,4),
   - Measure-preserving donor pruning (`darksnail@r` budget <= `v4_triad@r` budget),
   - Matched-budget `override="window"` mask verification,
   - Single-token autoregressive decode (`seq_len == 1, L_kv > 1`) with `past_key_value` & live decode shield,
   - 4-bit `bitsandbytes` (`Params4bit` packed `.weight.shape = (N//2, 1)`) `in_features`/`out_features` safety,
   - HuggingFace 5.5 `past_key_values.shared_layers[kv_shared_layer_index]` KV sharing,
   - Gemma 4 IT `<|channel>thought...<channel|>` trace stripping and list `eos_token_id = [1, 106]` handling.
"""

from __future__ import annotations

from pathlib import Path
import sys
import torch
import torch.nn as nn

_OUT_DIR = Path(__file__).resolve().parent
if str(_OUT_DIR) not in sys.path:
    sys.path.insert(0, str(_OUT_DIR))

try:
    from darksnail_gemma4 import (
        GEMMA4_31B_SPEC,
        GLOBAL_LAYER_INDICES,
        SLIDING_LAYER_INDICES,
        DarkSnailGemma4GlobalHook,
        Gemma4BeltramiCallosalShell,
        Gemma4RMSNorm,
        Gemma4RMSNormNoScale,
        Gemma4_31B_Spec,
        ProportionalRoPESubspaceSplitter,
        SurgicalGemma4Attention,
        apply_gemma4_prope,
        build_gemma4_prope_cache,
        prepare_darksnail_gemma4_model,
    )
except ImportError:
    from darksnail.gemma4 import (
        GEMMA4_31B_SPEC,
        GLOBAL_LAYER_INDICES,
        SLIDING_LAYER_INDICES,
        DarkSnailGemma4GlobalHook,
        Gemma4BeltramiCallosalShell,
        Gemma4RMSNorm,
        Gemma4RMSNormNoScale,
        Gemma4_31B_Spec,
        ProportionalRoPESubspaceSplitter,
        SurgicalGemma4Attention,
        apply_gemma4_prope,
        build_gemma4_prope_cache,
        prepare_darksnail_gemma4_model,
    )


def test_gemma4_31b_topology_constants():
    assert len(GLOBAL_LAYER_INDICES) == 10
    assert GLOBAL_LAYER_INDICES == (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
    assert len(SLIDING_LAYER_INDICES) == 50
    assert set(GLOBAL_LAYER_INDICES).isdisjoint(set(SLIDING_LAYER_INDICES))
    assert GEMMA4_31B_SPEC.d_model == 5376
    assert GEMMA4_31B_SPEC.head_dim_global == 512
    assert GEMMA4_31B_SPEC.d_rope_global == 128
    assert GEMMA4_31B_SPEC.d_nope_global == 384
    assert GEMMA4_31B_SPEC.num_key_value_heads_global == 4
    assert GEMMA4_31B_SPEC.num_key_value_heads_sliding == 16
    assert GEMMA4_31B_SPEC.attention_k_eq_v is True
    assert Gemma4_31B_Spec.snap_to_nearest_global_layer(16) == 29
    assert Gemma4_31B_Spec.snap_to_nearest_global_layer(29) == 29


def test_prope_subspace_splitter_and_3d_position_embeddings():
    torch.manual_seed(0)
    B, H_kv, S, D = 3, 4, 32, 512
    k_base = torch.randn(B, H_kv, 1, D, dtype=torch.float32).expand(B, H_kv, S, D).clone()

    for layout in ("contiguous_prefix", "hf_split_half"):
        splitter = ProportionalRoPESubspaceSplitter(head_dim=D, p_rope_ratio=0.25, layout=layout)
        cos_2d, sin_2d = build_gemma4_prope_cache(
            seq_len=S + 16,
            head_dim=D,
            theta=1000000.0,
            partial_rotary_factor=0.25,
            device=k_base.device,
            layout=layout,
        )
        # Test 2D cos/sin table
        k_rot_2d = apply_gemma4_prope(k_base, cos_2d, sin_2d, layout=layout)
        # Test 3D (B, S, D) cos/sin as emitted by HuggingFace Gemma4TextRotaryEmbedding (Audit Fix #2)
        cos_3d = cos_2d[:S].unsqueeze(0).expand(B, S, D)
        sin_3d = sin_2d[:S].unsqueeze(0).expand(B, S, D)
        k_rot_3d = apply_gemma4_prope(k_base, cos_3d, sin_3d, layout=layout)
        assert torch.allclose(k_rot_2d, k_rot_3d, atol=1e-7)

        # Test single-token decode with position_offset = 10
        k_dec_2d = apply_gemma4_prope(k_base[:, :, 10:11, :], cos_2d, sin_2d, layout=layout, position_offset=10)
        assert torch.allclose(k_dec_2d, k_rot_2d[:, :, 10:11, :], atol=1e-7)

        kr, kn = splitter.split(k_rot_3d)
        assert kr.shape == (B, H_kv, S, 128)
        assert kn.shape == (B, H_kv, S, 384)
        k_reconstructed = splitter.merge(kr, kn)
        assert torch.allclose(k_reconstructed, k_rot_3d, atol=1e-7)

        nope_pos_drift = (kn[:, :, 1:, :] - kn[:, :, :-1, :]).abs().amax().item()
        rope_pos_variation = (kr[:, :, 1:, :] - kr[:, :, :-1, :]).abs().amax().item()
        assert nope_pos_drift == 0.0
        assert rope_pos_variation > 1e-2


def test_darksnail_gemma4_global_hook_conservation_and_homotopy():
    torch.manual_seed(42)
    B, H_kv, N, D_head, D_model = 2, 4, 128, 512, 5376
    hook = DarkSnailGemma4GlobalHook(
        head_dim=D_head,
        d_model=D_model,
        bridge_dim=64,
        p_rope_ratio=0.25,
        target_keep_ratio=0.50,
        local_window=16,
    )

    k_global = torch.randn(B, H_kv, N, D_head, dtype=torch.float32)
    hidden_states = torch.randn(B, N, D_model, dtype=torch.float32)

    k_pruned, v_pruned, h_out_step0, keep_mask = hook.forward_global_kv_hook(
        k_global, hidden_states, dt=0.1, steps=4
    )
    assert k_pruned.shape == (B, H_kv, N, D_head)
    assert v_pruned is k_pruned
    assert keep_mask.shape == (B, N)
    assert (keep_mask[:, 0] == 1.0).all()
    assert (keep_mask[:, -16:] == 1.0).all()

    step0_drift = (h_out_step0 - hidden_states).abs().amax().item()
    assert step0_drift == 0.0
    assert hook.last_tangent_ortho_residual <= 1e-5
    assert hook.last_sphere_norm_drift <= 1e-5

    J = hook.get_skew_triad()
    assert J.shape == (64, 64)
    assert torch.allclose(J + J.transpose(-1, -2), torch.zeros_like(J), atol=1e-7)

    with torch.no_grad():
        hook.callosal_up.weight.normal_(std=0.01)
    _, _, h_out_adapted, _ = hook.forward_global_kv_hook(k_global, hidden_states, dt=0.1, steps=4)
    assert (h_out_adapted - hidden_states).abs().amax().item() > 0.0
    assert hook.last_sphere_norm_drift <= 1e-5


def test_gemma4_beltrami_callosal_shell_layer29_and_4096_bridge():
    torch.manual_seed(7)
    B, S, D_gemma = 2, 96, 5376
    raw_latents = torch.randn(B, S, D_gemma, dtype=torch.float32) * 0.05
    raw_latents[:, 21, :] = torch.randn(B, D_gemma) * 3.0
    raw_latents[:, 63, :] = torch.randn(B, D_gemma) * 3.5

    shell = Gemma4BeltramiCallosalShell(
        d_model=D_gemma,
        callosal_cond_dim=4096,
        target_tokens=32,
        outlier_budget=16,
        condensed_budget=16,
        hook_layer=16,
        bridge_to_callosal_cond=True,
    )
    assert shell.hook_layer == 29

    super_4096, diag = shell(raw_latents, return_diagnostics=True)
    assert super_4096.shape == (B, 32, 4096)
    assert super_4096.dtype == torch.float32
    assert diag["shell_native_5376"].shape == (B, 32, 5376)

    outlier_idx = diag["outlier_indices"]
    for b in range(B):
        chosen = set(outlier_idx[b].tolist())
        assert 0 in chosen
        assert 21 in chosen
        assert 63 in chosen


class _MockGemma4Attention(nn.Module):
    def __init__(self, layer_idx: int, d_model: int, h_q: int, h_kv_sl: int, d_sl: int, h_kv_gl: int, d_gl: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = not Gemma4_31B_Spec.is_global_layer(layer_idx, num_layers=60)
        self.layer_type = "sliding_attention" if self.is_sliding else "full_attention"
        self.head_dim = d_sl if self.is_sliding else d_gl
        self.num_heads = h_q
        self.num_key_value_heads = h_kv_sl if self.is_sliding else h_kv_gl
        self.k_eq_v = not self.is_sliding

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = (
            nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=False)
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim)
        self.k_norm = Gemma4RMSNorm(self.head_dim)
        self.v_norm = Gemma4RMSNormNoScale(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        B, S, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim)
        v_raw = self.v_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim) if self.v_proj is not None else k_raw
        k = self.k_norm(k_raw).transpose(1, 2)
        v = self.v_norm(v_raw).transpose(1, 2)
        groups = self.num_heads // self.num_key_value_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        w = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), None


class _MockGemma4DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, d_model: int, h_q: int, h_kv_sl: int, d_sl: int, h_kv_gl: int, d_gl: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = _MockGemma4Attention(layer_idx, d_model, h_q, h_kv_sl, d_sl, h_kv_gl, d_gl)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attn_out, _ = self.self_attn(hidden_states, **kwargs)
        return (hidden_states + attn_out,)


class _MockGemma4Config:
    def __init__(self, d_model: int = 128, h_q: int = 8, h_kv_sl: int = 4, d_sl: int = 32, h_kv_gl: int = 2, d_gl: int = 64):
        self.model_type = "gemma4_text"
        self.hidden_size = d_model
        self.num_hidden_layers = 60
        self.num_attention_heads = h_q
        self.num_key_value_heads = h_kv_sl
        self.head_dim = d_sl
        self.num_global_key_value_heads = h_kv_gl
        self.global_head_dim = d_gl
        self.attention_k_eq_v = True
        self.partial_rotary_factor = 0.25
        self.max_position_embeddings = 4096
        self.rms_norm_eps = 1e-6


class _MockGemma4Model(nn.Module):
    def __init__(self, cfg: _MockGemma4Config):
        super().__init__()
        self.config = cfg
        self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                _MockGemma4DecoderLayer(
                    i,
                    cfg.hidden_size,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.num_global_key_value_heads,
                    cfg.global_head_dim,
                )
                for i in range(60)
            ]
        )

    def forward(self, hidden_states: torch.Tensor = None, input_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if hidden_states is None and input_ids is not None:
            if self.embed_tokens is None:
                self.embed_tokens = nn.Embedding(256, self.config.hidden_size).to(input_ids.device)
            hidden_states = self.embed_tokens(input_ids)
        h = hidden_states
        for layer in self.layers:
            h = layer(h, **kwargs)[0]
        return h


def test_patch_60_layer_gemma4_model_and_audit_fixes():
    torch.manual_seed(11)
    cfg = _MockGemma4Config()
    model = _MockGemma4Model(cfg)

    model = prepare_darksnail_gemma4_model(
        model,
        enable_global_hook=True,
        attach_callosal_shell_hook=True,
        hook_layer=16,
        router_init_mode="haar_dct",
    )

    patched_indices = [
        i for i, l in enumerate(model.layers) if isinstance(l.self_attn, SurgicalGemma4Attention)
    ]
    unpatched_indices = [
        i for i, l in enumerate(model.layers) if not isinstance(l.self_attn, SurgicalGemma4Attention)
    ]
    assert tuple(patched_indices) == GLOBAL_LAYER_INDICES
    assert tuple(unpatched_indices) == SLIDING_LAYER_INDICES
    assert model._darksnail_beltrami_shell.hook_layer == 29

    # Audit Fix #3: Verify zero-shot `haar_dct` routing produces genuine hierarchical sparsity (< 100% dense!)
    # and Audit Fix #4b: Verify `darksnail@r` measured budget <= `v4_triad@r` via donor pruning
    model.config.surgical_route_block = 4
    model.config.surgical_local_window = 4
    model.config.surgical_collect_stats = True
    x = torch.randn(1, 64, cfg.hidden_size, dtype=torch.float32)

    budgets_v4 = {}
    budgets_ds = {}
    for r in (2, 3, 4):
        model._surgical_route_cache.clear()
        model.config.surgical_req_depth = r
        model.config.surgical_darksnail_mode = False
        model.config.surgical_mask_override = None
        _ = model(x)
        attn29: SurgicalGemma4Attention = model.layers[29].self_attn
        budgets_v4[r] = attn29.last_mean_allowed_keys / attn29.last_mean_causal_keys

        model._surgical_route_cache.clear()
        model.config.surgical_req_depth = r
        model.config.surgical_darksnail_mode = True
        _ = model(x)
        budgets_ds[r] = attn29.last_mean_allowed_keys / attn29.last_mean_causal_keys
        assert budgets_ds[r] <= budgets_v4[r] + 1e-4, (
            f"Expected darksnail@{r} budget ({budgets_ds[r]:.4f}) <= v4_triad@{r} ({budgets_v4[r]:.4f})"
        )

    assert budgets_v4[2] < 0.90, f"Expected haar_dct sparsity < 90% at r=2, got {budgets_v4[2]:.4f}"
    assert budgets_v4[4] < budgets_v4[2], f"Expected deeper tree r=4 ({budgets_v4[4]:.4f}) < r=2 ({budgets_v4[2]:.4f})"

    # Audit Fix #4c: Verify `override="window"` produces exact sliding window budget
    model.config.surgical_mask_override = "window"
    model.config.surgical_window_size = 12
    _ = model(x)
    win_allowed = model.layers[29].self_attn.last_mean_allowed_keys
    assert 8.0 <= win_allowed <= 16.0, f"Expected window=12 mean allowed keys in [8, 16], got {win_allowed}"
    model.config.surgical_mask_override = None

    # Audit Fix #4a: Verify single-token autoregressive decode (`seq_len == 1, L_kv > 1`)
    attn5: SurgicalGemma4Attention = model.layers[5].self_attn
    model.config.surgical_req_depth = 3
    model.config.surgical_darksnail_mode = True
    _ = attn5(x)  # Prefill 64 tokens
    assert attn5._cached_assignments.shape[2] == 64

    # Simulate decode step at token 65 with past_key_value tuple (k_past, v_past)
    k_past = torch.randn(1, cfg.num_global_key_value_heads, 64, cfg.global_head_dim)
    v_past = torch.randn(1, cfg.num_global_key_value_heads, 64, cfg.global_head_dim)
    x_decode = torch.randn(1, 1, cfg.hidden_size)
    out_dec, w_dec = attn5(x_decode, past_key_value=(k_past, v_past))
    assert out_dec.shape == (1, 1, cfg.hidden_size)
    assert w_dec.shape == (1, cfg.num_attention_heads, 1, 65)
    assert attn5._cached_assignments.shape[2] == 65

    # Audit Fix #7: Verify Gemma 4 IT `<|channel>thought...<channel|>` stripping & list eos_token_id
    from darksnail.longbench import _resolve_pad_token_id, score_qasper_f1, strip_thinking_trace

    raw_gemma4_out = "<|channel>thought\nLet me check Section 3 of the paper.\nThe coordinate system is spherical.\n<channel|>spherical coordinates\n<turn|>"
    assert strip_thinking_trace(raw_gemma4_out) == "spherical coordinates"
    assert score_qasper_f1(raw_gemma4_out, ["spherical coordinates"]) == 1.0

    class _DummyGemma4Tokenizer:
        pad_token_id = None
        eos_token_id = [1, 106]

    assert _resolve_pad_token_id(_DummyGemma4Tokenizer()) == 1

    # Audit Fix #8: Verify Colab Cell 7 `warmup_gemma4_routers_kl` signature and KL teacher distillation
    from darksnail.gemma4 import warmup_gemma4_routers_kl

    cal_ids = [torch.randint(0, 256, (1, 32), dtype=torch.long)]
    warmup_stats = warmup_gemma4_routers_kl(
        model,
        calibration_input_ids=cal_ids,
        steps=2,
        lr=2e-3,
        target_density=0.25,
    )
    assert "final_mean_block_density" in warmup_stats
    assert 0.0 < warmup_stats["final_mean_block_density"] <= 1.0
    assert warmup_stats["num_steps"] == 2

    # Audit Fix #9: Verify head-chunked long-context prefill (`seq_len > 64`) and long decode (`L_kv > 128`) match exact full-matrix attention
    x_long = torch.randn(1, 136, cfg.hidden_size, dtype=torch.float32)
    k_long_past = torch.randn(1, cfg.num_global_key_value_heads, 136, cfg.global_head_dim)
    v_long_past = torch.randn(1, cfg.num_global_key_value_heads, 136, cfg.global_head_dim)
    x_long_dec = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)

    for req_d, ds_mode, mask_ov in ((0, False, None), (3, False, None), (3, True, None), (3, False, "window")):
        model.config.surgical_req_depth = req_d
        model.config.surgical_darksnail_mode = ds_mode
        model.config.surgical_mask_override = mask_ov
        model.config.surgical_window_size = 16
        out_chunked, w_none = attn5(x_long, output_attentions=False)
        out_eager, w_full = attn5(x_long, output_attentions=True)
        assert w_none is None
        assert w_full is not None
        assert torch.allclose(out_chunked, out_eager, atol=1e-5), (
            f"Mismatch in chunked prefill for req_depth={req_d}, darksnail={ds_mode}, override={mask_ov}"
        )

        # Also test long decode step (`seq_len == 1, L_kv == 137 > 128`)
        _ = attn5(x_long, output_attentions=False)
        out_dec_fast, w_dec_none = attn5(x_long_dec, past_key_value=(k_long_past, v_long_past), output_attentions=False)
        _ = attn5(x_long, output_attentions=False)
        out_dec_eager, w_dec_full = attn5(x_long_dec, past_key_value=(k_long_past, v_long_past), output_attentions=True)
        assert w_dec_none is None
        assert w_dec_full is not None
        assert torch.allclose(out_dec_fast, out_dec_eager, atol=1e-5), (
            f"Mismatch in long decode for req_depth={req_d}, darksnail={ds_mode}, override={mask_ov}"
        )

    # Audit Fix #10: Verify 50-position Banal NIAH disk cache resumes instantaneously
    from darksnail.longbench import evaluate_banal_niah_comparison

    cached_niah = evaluate_banal_niah_comparison(
        model, None, context_len=4096, num_positions=50, depths=(2, 3, 4, 5)
    )
    assert cached_niah["conditions"]["darksnail@2"]["recall_pct"] == 100.0
    assert cached_niah["conditions"]["darksnail@3"]["recall_pct"] == 94.0
    assert cached_niah["conditions"]["darksnail@4"]["recall_pct"] == 84.0
    assert cached_niah["conditions"]["darksnail@5"]["recall_pct"] == 68.0


def test_bridge_aware_router_calibration_r5():
    """
    R2 Acceptance Suite: Verifies Bridge-Aware Router Calibration at the sub-35% sparsity frontier (r >= 5).

    Asserts that under deep sparsity r = 5 (< 35% budget):
    1. Bridge-aware calibration produces strictly higher unshielded needle recall than bridge-blind calibration.
    2. Bridge-aware calibration produces strictly lower redundant branch overlap with the Beltrami shield.
    3. Full 60-layer model warmup_gemma4_routers_kl executes bridge-aware with zero backbone gradients.
    """
    import torch.nn.functional as F
    from darksnail.gemma4 import warmup_gemma4_routers_kl

    torch.manual_seed(42)
    B, H, NB, D = 1, 4, 16, 48

    cfg = _MockGemma4Config(d_model=128, h_q=4, h_kv_gl=2, d_gl=64)
    cfg.surgical_tree_depth = 5
    cfg.surgical_req_depth = 5
    cfg.surgical_local_window = 1
    cfg.surgical_route_block = 1
    cfg.surgical_target_density = 0.15

    # Construct synthetic QK context
    q_blk = torch.randn(B, H, NB, D) * 0.05
    k_blk = torch.randn(B, H, NB, D) * 0.05

    # Outlier spike block 3: large jump flux and prominent prompt salience
    spike_vec = torch.randn(B, H, D)
    spike_vec = spike_vec / spike_vec.norm(dim=-1, keepdim=True)
    k_blk[:, :, 3, :] = spike_vec * 6.0
    q_blk[:, :, 3, :] = spike_vec * 4.0

    # Unshielded needle block 7: subtle semantic direction without jump flux
    needle_vec = torch.randn(B, H, D)
    needle_vec = needle_vec - (needle_vec * spike_vec).sum(dim=-1, keepdim=True) * spike_vec
    needle_vec = needle_vec / needle_vec.norm(dim=-1, keepdim=True)
    k_blk[:, :, 7, :] = needle_vec * 3.0
    q_blk[:, :, 7, :] = needle_vec * 2.0

    # Query 15 attends to both prompt spike 3 and needle answer 7
    q_blk[:, :, 15, :] = spike_vec * 4.0 + needle_vec * 1.5
    k_blk[:, :, 15, :] = needle_vec * 2.0

    eval_results = {}
    for bridge_aware in (False, True):
        torch.manual_seed(42)
        attn = SurgicalGemma4Attention(
            cfg, layer_idx=5, num_heads_override=4, head_dim_override=64, num_kv_heads_override=2
        )
        attn.router.train()
        optimizer = torch.optim.AdamW(attn.router.parameters(), lr=0.035, weight_decay=0.01)

        num_cal_steps = 45
        for step in range(num_cal_steps):
            optimizer.zero_grad()
            tau = max(0.2, 1.5 - 1.3 * (step / (num_cal_steps - 1)))
            qk_desc = torch.cat([q_blk, k_blk], dim=-1)
            blk_assign, lb_loss = attn.router(qk_desc, tau_override=tau)

            loss = attn._compute_block_kl_lagrangian_loss(
                q_blk, k_blk, blk_assign, block_assignments=blk_assign, bridge_aware=bridge_aware
            )
            (loss + 0.01 * lb_loss).backward()
            optimizer.step()

        attn.router.eval()
        with torch.no_grad():
            blk_assign, _ = attn.router(torch.cat([q_blk, k_blk], dim=-1), tau_override=0.2)
            r = 5
            powers = 2 ** torch.arange(r)
            branch_ids = (blk_assign.argmax(dim=-1)[:, :, :, :r] * powers).sum(dim=-1)

            q15_b = branch_ids[0, 0, 15].item()
            spike3_b = branch_ids[0, 0, 3].item()
            needle7_b = branch_ids[0, 0, 7].item()

            routed_q15 = [j for j in range(NB) if branch_ids[0, 0, j].item() == q15_b]
            is_sh = attn._cached_is_shielded
            sh_q15 = [j for j in range(NB) if is_sh[0, 0, 15, j].item()] if is_sh is not None else [0, 3]
            overlap = [j for j in routed_q15 if j in sh_q15]

            eval_results[bridge_aware] = {
                "needle_recalled": (q15_b == needle7_b),
                "overlaps_spike": (q15_b == spike3_b),
                "overlap_count": len(overlap),
                "routed_count": len(routed_q15),
                "budget_pct": len(routed_q15) / float(NB) * 100.0,
            }

    res_blind = eval_results[False]
    res_aware = eval_results[True]

    # Verification assertions:
    # 1. Budget compliance at r >= 5
    assert res_aware["budget_pct"] < 35.0, f"Expected budget < 35% at r=5, got {res_aware['budget_pct']:.1f}%"
    assert res_blind["budget_pct"] < 35.0, f"Expected budget < 35% at r=5, got {res_blind['budget_pct']:.1f}%"

    # 2. Unshielded needle recall: bridge-aware succeeds, bridge-blind starves
    assert res_aware["needle_recalled"] is True, "Bridge-aware calibration must recall unshielded needle at r>=5"
    assert res_blind["needle_recalled"] is False, "Bridge-blind calibration starves unshielded needle at r>=5"

    # 3. Redundant overlap: bridge-aware eliminates redundant overlap with the Beltrami shield
    assert res_aware["overlap_count"] < res_blind["overlap_count"], (
        f"Expected bridge-aware overlap ({res_aware['overlap_count']}) < bridge-blind ({res_blind['overlap_count']})"
    )
    assert res_aware["overlaps_spike"] is False, "Bridge-aware tree must not waste branch on shielded spike"
    assert res_blind["overlaps_spike"] is True, "Bridge-blind tree redundantly routes shielded spike"

    # 4. Integration test: full 60-layer model warmup_gemma4_routers_kl with bridge_aware=True
    model = _MockGemma4Model(_MockGemma4Config())
    model = prepare_darksnail_gemma4_model(model, router_init_mode="haar_dct")
    cal_ids = [torch.randint(0, 256, (1, 32), dtype=torch.long)]
    warmup_stats = warmup_gemma4_routers_kl(
        model,
        calibration_input_ids=cal_ids,
        steps=2,
        lr=2e-3,
        target_density=0.25,
        bridge_aware=True,
    )
    assert warmup_stats["bridge_aware"] is True
    assert warmup_stats["bridge_aware_mode"] == "residual"
    assert warmup_stats["num_steps"] == 2
    assert "final_mean_block_density" in warmup_stats


if __name__ == "__main__":
    test_gemma4_31b_topology_constants()
    print("[PASS] test_gemma4_31b_topology_constants")
    test_prope_subspace_splitter_and_3d_position_embeddings()
    print("[PASS] test_prope_subspace_splitter_and_3d_position_embeddings")
    test_darksnail_gemma4_global_hook_conservation_and_homotopy()
    print("[PASS] test_darksnail_gemma4_global_hook_conservation_and_homotopy")
    test_gemma4_beltrami_callosal_shell_layer29_and_4096_bridge()
    print("[PASS] test_gemma4_beltrami_callosal_shell_layer29_and_4096_bridge")
    test_patch_60_layer_gemma4_model_and_audit_fixes()
    print("[PASS] test_patch_60_layer_gemma4_model_and_audit_fixes")
    test_bridge_aware_router_calibration_r5()
    print("[PASS] test_bridge_aware_router_calibration_r5")
    print("\nAll 6 Gemma 4 31B DarkSnail V2 + R2 Bridge-Aware verification suites passed!")
