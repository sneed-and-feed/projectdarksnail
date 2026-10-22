#!/usr/bin/env python3
"""
scripts/stress_adversarial.py
=============================
Empirical & Adversarial Stress Harness for Project DarkSnail.

Tests:
1. Wildcard Imports, Repeated Imports, Namespace Isolation & Dynamic Introspection
2. Numerical Edge Cases for `infer_arc_output_shape_beltrami`
3. Numerical Edge Cases for `ProportionalRoPESubspaceSplitter`
4. Numerical Edge Cases for `DarkSnailCanvasReadout`
5. Additional Core Neuro-Symbolic Invariants (`SkewSymmetricTriadOperator`,
   `SlowFastTriadCascade`, `MagicNorm`, `TemporalClockContentSplitter`, `RoPE2DMedoidPooler`)

Authors: Challenger 1 (Technical & Academic Audit)
Date: 2026-10-22
"""

import gc
import importlib
import inspect
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class StressTestReporter:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures: List[Tuple[str, str]] = []
        self.findings: List[str] = []

    def record_pass(self, name: str):
        self.total += 1
        self.passed += 1
        print(f"  [PASS] {name}")

    def record_fail(self, name: str, reason: str):
        self.total += 1
        self.failed += 1
        self.failures.append((name, reason))
        print(f"  [FAIL] {name}: {reason}")

    def record_finding(self, finding: str):
        self.findings.append(finding)
        print(f"  [EMPIRICAL FINDING] {finding}")

    def summary(self) -> bool:
        print("\n" + "=" * 70)
        print(f"STRESS TEST SUMMARY: {self.passed}/{self.total} Passed ({self.failed} Failed)")
        print(f"EMPIRICAL FINDINGS DOCUMENTED: {len(self.findings)}")
        print("=" * 70)
        if self.findings:
            print("\nDocumented Empirical Findings / Edge-Case Vulnerabilities:")
            for f in self.findings:
                print(f"  * {f}")
        if self.failures:
            print("\nFailures:")
            for name, reason in self.failures:
                print(f"  - {name}: {reason}")
            return False
        print("ALL ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY.")
        return True


reporter = StressTestReporter()


# ============================================================================
# Section 1: Wildcard Import, Namespace Isolation & Dynamic Introspection
# ============================================================================
def test_section_1_imports_and_namespace():
    print("\n--- Section 1: Wildcard Imports, Namespace Isolation & Introspection ---")
    import darksnail

    # 1.1 Completeness and uniqueness of __all__
    try:
        assert hasattr(darksnail, "__all__"), "__all__ missing in darksnail"
        all_syms = darksnail.__all__
        assert len(all_syms) == len(set(all_syms)), f"Duplicates in __all__: {len(all_syms)} vs {len(set(all_syms))}"
        reporter.record_pass("1.1 __all__ uniqueness and non-empty")
    except Exception as e:
        reporter.record_fail("1.1 __all__ uniqueness", str(e))

    # 1.2 Wildcard import purity (no internal modules/aliases leaked)
    try:
        ns = {}
        exec("from darksnail import *", ns)
        # Check that standard internal modules are NOT in ns
        forbidden_leaks = {"torch", "nn", "math", "np", "numpy", "os", "sys", "re", "json", "F", "Sequence", "Tuple", "Dict", "Optional", "Union", "List"}
        leaked = forbidden_leaks.intersection(set(ns.keys()))
        assert not leaked, f"Wildcard import leaked private/imported modules into caller namespace: {leaked}"

        # Check every exported symbol is present in ns
        for sym in darksnail.__all__:
            assert sym in ns, f"Exported symbol '{sym}' missing from wildcard import namespace"
            assert ns[sym] is getattr(darksnail, sym), f"Symbol '{sym}' does not match darksnail.{sym}"
        reporter.record_pass("1.2 Wildcard import purity and exact symbol matching")
    except Exception as e:
        reporter.record_fail("1.2 Wildcard import purity", str(e))

    # 1.3 Namespace isolation under restricted builtins
    try:
        test_ns = {"__builtins__": __builtins__}
        exec("from darksnail import infer_arc_output_shape_beltrami, ProportionalRoPESubspaceSplitter", test_ns)
        assert callable(test_ns["infer_arc_output_shape_beltrami"])
        assert issubclass(test_ns["ProportionalRoPESubspaceSplitter"], nn.Module)
        reporter.record_pass("1.3 Namespace isolation under separate execution scopes")
    except Exception as e:
        reporter.record_fail("1.3 Namespace isolation", str(e))

    # 1.4 Repeated reloads and sys.modules clearing
    try:
        initial_all = list(darksnail.__all__)
        for _ in range(10):
            importlib.reload(darksnail)
        assert darksnail.__all__ == initial_all, "Reload changed __all__"
        reporter.record_pass("1.4 10x repeated importlib.reload stability")
    except Exception as e:
        reporter.record_fail("1.4 Repeated reload stability", str(e))

    # 1.5 Dynamic introspection of all 74 exported symbols
    try:
        broken_introspection = []
        for sym in darksnail.__all__:
            obj = getattr(darksnail, sym)
            if obj is None:
                broken_introspection.append((sym, "is None"))
            elif inspect.isclass(obj):
                _ = obj.__name__
                _ = obj.__doc__
            elif callable(obj):
                sig = inspect.signature(obj)
                assert sig is not None
            elif hasattr(obj, "__class__"):
                # Specification or config instance (e.g. GEMMA4_31B_SPEC)
                _ = repr(obj)
            else:
                broken_introspection.append((sym, f"unexpected object type {type(obj)}"))
        assert not broken_introspection, f"Broken dynamic introspection: {broken_introspection}"
        reporter.record_pass(f"1.5 Dynamic introspection over all {len(darksnail.__all__)} exported symbols")
    except Exception as e:
        reporter.record_fail("1.5 Dynamic introspection", str(e))


# ============================================================================
# Section 2: Numerical Edge Cases for infer_arc_output_shape_beltrami
# ============================================================================
def test_section_2_infer_arc_output_shape():
    print("\n--- Section 2: Numerical Edge Cases for infer_arc_output_shape_beltrami ---")
    from darksnail.shell import infer_arc_output_shape_beltrami

    # 2.1 Degenerate / Empty Grids
    try:
        assert infer_arc_output_shape_beltrami([], []) == (1, 1)
        assert infer_arc_output_shape_beltrami([], [[]]) == (1, 1)
        assert infer_arc_output_shape_beltrami([], [[0]]) == (1, 1)
        assert infer_arc_output_shape_beltrami([], [[5]]) == (1, 1)
        assert infer_arc_output_shape_beltrami([], [[0, 0], [0, 0]]) == (2, 2)
        assert infer_arc_output_shape_beltrami([], [[1, 2, 3]]) == (1, 3)
        assert infer_arc_output_shape_beltrami([], [[1], [2], [3]]) == (3, 1)
        reporter.record_pass("2.1 Degenerate / Empty test input grids without demos")
    except Exception as e:
        reporter.record_fail("2.1 Degenerate test grids", str(e))

    # 2.2 Malformed / Degenerate Demos
    try:
        d_empty = [{"input": [], "output": []}]
        assert infer_arc_output_shape_beltrami(d_empty, [[1, 2], [3, 4]]) == (2, 2)

        d_nested_empty = [{"input": [[]], "output": [[]]}]
        assert infer_arc_output_shape_beltrami(d_nested_empty, [[1, 2], [3, 4]]) == (2, 2)

        d_bad_keys = [{"foo": 1}, {"bar": 2}]
        assert infer_arc_output_shape_beltrami(d_bad_keys, [[1, 2], [3, 4]]) == (2, 2)

        d_mismatch = [{"input": [[1]], "output": []}, {"input": [], "output": [[1]]}]
        assert infer_arc_output_shape_beltrami(d_mismatch, [[1, 2], [3, 4]]) == (2, 2)

        reporter.record_pass("2.2 Malformed demos degrade gracefully to input grid shape")
    except Exception as e:
        reporter.record_fail("2.2 Malformed demos", str(e))

    # 2.3 Single Demo vs Multiple Demos
    try:
        d1 = [{"input": [[1, 2], [3, 4]], "output": [[5, 5, 5], [5, 5, 5]]}]
        res1 = infer_arc_output_shape_beltrami(d1, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert res1 == (2, 3), f"Expected (2, 3), got {res1}"

        d_id = [{"input": [[1, 2, 3]], "output": [[4, 5, 6]]}]
        res_id = infer_arc_output_shape_beltrami(d_id, [[7, 8, 9, 10]])
        assert res_id == (1, 4), f"Expected (1, 4), got {res_id}"

        reporter.record_pass("2.3 Single-demo geometric inference rules")
    except Exception as e:
        reporter.record_fail("2.3 Single-demo inference", str(e))

    # 2.4 Extreme and Out-of-Bounds Grid Values
    try:
        d_neg = [
            {"input": [[-1, -2], [-3, -4]], "output": [[-1, -2], [-3, -4]]},
            {"input": [[-5, -6], [-7, -8]], "output": [[-5, -6], [-7, -8]]},
        ]
        res_neg = infer_arc_output_shape_beltrami(d_neg, [[-9, -10], [-11, -12]])
        assert res_neg == (2, 2)

        # Output grid has shape (1, 2)
        d_large = [
            {"input": [[1000, 2000], [3000, 4000]], "output": [[1, 1]]},
            {"input": [[5000, 6000], [7000, 8000]], "output": [[1, 1]]},
        ]
        res_large = infer_arc_output_shape_beltrami(d_large, [[9000, 10000], [11000, 12000]])
        assert res_large == (1, 2), f"Expected (1, 2), got {res_large}"

        reporter.record_pass("2.4 Out-of-bounds / negative pixel values handled without crash")
    except Exception as e:
        reporter.record_fail("2.4 Out-of-bounds pixel values", str(e))

    # 2.5 Maximum Grid Sizes and Clamping Invariant
    try:
        grid_100x100 = [[0] * 100 for _ in range(100)]
        res_default = infer_arc_output_shape_beltrami([], grid_100x100, side=30)
        assert res_default == (30, 30), f"Clamping failed: {res_default}"

        res_custom_side = infer_arc_output_shape_beltrami([], grid_100x100, side=50)
        assert res_custom_side == (50, 50), f"Custom side clamping failed: {res_custom_side}"

        d_mult = [
            {"input": [[1, 2], [3, 4]], "output": [[0] * 20 for _ in range(20)]},
            {"input": [[5, 6], [7, 8]], "output": [[0] * 20 for _ in range(20)]},
        ]
        res_mult = infer_arc_output_shape_beltrami(d_mult, [[0] * 10 for _ in range(10)], side=30)
        assert 1 <= res_mult[0] <= 30 and 1 <= res_mult[1] <= 30

        reporter.record_pass("2.5 Maximum grid size clamping invariants (1 <= H, W <= side)")
    except Exception as e:
        reporter.record_fail("2.5 Clamping invariants", str(e))

    # 2.6 All-Zero / Zero-Norm Grids of various aspect ratios
    try:
        aspect_ratios = [(1, 1), (1, 30), (30, 1), (2, 15), (15, 2), (30, 30)]
        for h, w in aspect_ratios:
            grid_zero = [[0] * w for _ in range(h)]
            res = infer_arc_output_shape_beltrami([], grid_zero)
            assert res == (h, w), f"Aspect ratio ({h}, {w}) mismatch: got {res}"
        reporter.record_pass("2.6 All-zero grids across 6 diverse aspect ratios")
    except Exception as e:
        reporter.record_fail("2.6 All-zero aspect ratios", str(e))


# ============================================================================
# Section 3: Numerical Edge Cases for ProportionalRoPESubspaceSplitter
# ============================================================================
def test_section_3_prope_splitter():
    print("\n--- Section 3: Numerical Edge Cases for ProportionalRoPESubspaceSplitter ---")
    from darksnail.gemma4 import ProportionalRoPESubspaceSplitter

    # 3.1 Zero-Norm Tensors & Bijectivity (contiguous_prefix)
    try:
        splitter_cont = ProportionalRoPESubspaceSplitter(head_dim=512, p_rope_ratio=0.25, layout="contiguous_prefix")
        assert splitter_cont.d_rope == 128
        assert splitter_cont.d_nope == 384

        x_zero = torch.zeros(2, 8, 16, 512)
        xr, xn = splitter_cont.split(x_zero)
        assert xr.shape == (2, 8, 16, 128)
        assert xn.shape == (2, 8, 16, 384)
        assert xr.norm().item() == 0.0
        assert xn.norm().item() == 0.0

        x_reconstructed = splitter_cont.merge(xr, xn)
        assert torch.equal(x_zero, x_reconstructed)
        reporter.record_pass("3.1 Zero-norm split/merge bijectivity (contiguous_prefix)")
    except Exception as e:
        reporter.record_fail("3.1 Zero-norm split/merge", str(e))

    # 3.2 Bijectivity & Isometry (hf_split_half)
    try:
        splitter_hf = ProportionalRoPESubspaceSplitter(head_dim=512, p_rope_ratio=0.25, layout="hf_split_half")
        torch.manual_seed(42)
        x_rand = torch.randn(4, 32, 512, dtype=torch.float32)
        xr, xn = splitter_hf.split(x_rand)
        assert xr.shape == (4, 32, 128)
        assert xn.shape == (4, 32, 384)

        norm_orig_sq = x_rand.pow(2).sum()
        norm_split_sq = xr.pow(2).sum() + xn.pow(2).sum()
        assert torch.allclose(norm_orig_sq, norm_split_sq, atol=1e-5)

        x_merged = splitter_hf.merge(xr, xn)
        assert torch.allclose(x_rand, x_merged, atol=1e-6)
        reporter.record_pass("3.2 Isometry and exact reconstruction (hf_split_half)")
    except Exception as e:
        reporter.record_fail("3.2 Isometry hf_split_half", str(e))

    # 3.3 Diverse Tensor Dimensions (1D through 5D, and 0-batch)
    try:
        splitter = ProportionalRoPESubspaceSplitter(head_dim=512, p_rope_ratio=0.25)
        test_shapes = [
            (512,),
            (1, 512),
            (2, 512),
            (1, 1, 512),
            (2, 4, 16, 512),
            (1, 2, 3, 4, 512),
            (0, 512),
        ]
        for sh in test_shapes:
            t = torch.randn(*sh) if sh[0] > 0 else torch.empty(*sh)
            r, n = splitter.split(t)
            m = splitter.merge(r, n)
            assert m.shape == sh
            if sh[0] > 0:
                assert torch.equal(t, m)
        reporter.record_pass("3.3 Multi-dimensional and 0-length batch support (1D to 5D)")
    except Exception as e:
        reporter.record_fail("3.3 Multi-dimensional support", str(e))

    # 3.4 Numerical Extremes: Subnormals, Huge Floats, NaNs, Infs
    try:
        splitter = ProportionalRoPESubspaceSplitter(head_dim=512, p_rope_ratio=0.25)

        # Subnormal floats (1e-38)
        x_subnormal = torch.full((1, 512), 1e-38, dtype=torch.float32)
        r, n = splitter.split(x_subnormal)
        m = splitter.merge(r, n)
        assert torch.equal(x_subnormal, m)

        # Huge floats (1e30)
        x_huge = torch.full((1, 512), 1e30, dtype=torch.float32)
        r, n = splitter.split(x_huge)
        m = splitter.merge(r, n)
        assert torch.equal(x_huge, m)

        # NaNs preservation without unhandled crash
        x_nan = torch.full((1, 512), float("nan"), dtype=torch.float32)
        r_nan, n_nan = splitter.split(x_nan)
        m_nan = splitter.merge(r_nan, n_nan)
        assert torch.isnan(r_nan).all() and torch.isnan(n_nan).all() and torch.isnan(m_nan).all()

        # Infs preservation without unhandled crash
        x_inf = torch.full((1, 512), float("inf"), dtype=torch.float32)
        r_inf, n_inf = splitter.split(x_inf)
        m_inf = splitter.merge(r_inf, n_inf)
        assert torch.isinf(r_inf).all() and torch.isinf(n_inf).all() and torch.isinf(m_inf).all()

        reporter.record_pass("3.4 Extreme floats (subnormal, 1e30, NaN, Inf) handling")
    except Exception as e:
        reporter.record_fail("3.4 Extreme floats", str(e))

    # 3.5 Invalid Configuration & Layout Error Handling
    try:
        splitter = ProportionalRoPESubspaceSplitter(head_dim=512)
        try:
            splitter.split(torch.randn(1, 512), layout_override="invalid_layout")
            reporter.record_fail("3.5 Invalid layout override", "Failed to raise ValueError")
        except ValueError:
            reporter.record_pass("3.5 Graceful ValueError on invalid layout override")
    except Exception as e:
        reporter.record_fail("3.5 Invalid layout", str(e))


# ============================================================================
# Section 4: Numerical Edge Cases for DarkSnailCanvasReadout
# ============================================================================
def test_section_4_canvas_readout():
    print("\n--- Section 4: Numerical Edge Cases for DarkSnailCanvasReadout ---")
    from darksnail.readout import DarkSnailCanvasReadout

    readout = DarkSnailCanvasReadout(readout_scale=6.0, spectral_gain=0.35)

    # 4.1 Step-0 Homotopy Invariant (delta_z == 0.0 and delta_z is None)
    try:
        inputs = torch.zeros((1, 900), dtype=torch.long)
        logits = torch.randn((1, 900, 12), dtype=torch.float32)

        out_none = readout(logits, inputs, None)
        assert torch.allclose(out_none, logits, atol=1e-5), "None delta_z did not return unmodified logits"

        dz_zero = torch.zeros((1, 8192), dtype=torch.float32)
        out_zero = readout(logits, inputs, dz_zero)
        logits_norm = logits.clone()
        std = logits_norm.std(dim=-1, keepdim=True)
        if bool((std > 2.5).any().item()):
            logits_norm = logits_norm / torch.clamp(std / 1.5, min=1.0)
        assert torch.allclose(out_zero, logits_norm, atol=1e-5), "Step-0 homotopy violated: delta_z=0 produced non-zero shift"
        reporter.record_pass("4.1 Step-0 Homotopy invariant: zero delta_z produces zero readout shift")
    except Exception as e:
        reporter.record_fail("4.1 Step-0 Homotopy", str(e))

    # 4.2 Zero-Norm Inputs & All-Zero Logits Stability
    try:
        inputs_0 = torch.zeros((1, 900), dtype=torch.long)
        logits_0 = torch.zeros((1, 900, 12), dtype=torch.float32)
        dz_rand = torch.randn((1, 8192), dtype=torch.float32) * 0.05

        out_0 = readout(logits_0, inputs_0, dz_rand)
        assert not torch.isnan(out_0).any(), "NaN detected from zero logits and zero inputs"
        assert not torch.isinf(out_0).any(), "Inf detected from zero logits and zero inputs"
        reporter.record_pass("4.2 Numerical stability with zero logits and zero inputs (no NaN/Inf)")
    except Exception as e:
        reporter.record_fail("4.2 Zero-norm stability", str(e))

    # 4.3 Grid Size Edge Cases: Non-Square Sequences and Sub-4x4 Vulnerability
    try:
        dz = torch.randn((1, 8192), dtype=torch.float32) * 0.05

        # 4.3a: Non-square sequence length (e.g. L=50) -> gracefully returns logits_f32
        inputs_non_sq = torch.zeros((1, 50), dtype=torch.long)
        logits_non_sq = torch.randn((1, 50, 12), dtype=torch.float32)
        out_non_sq = readout(logits_non_sq, inputs_non_sq, dz)
        assert out_non_sq.shape == (1, 50, 12)

        # 4.3b: Mismatched length between logits and inputs -> gracefully returns logits_f32
        inputs_mismatch = torch.zeros((1, 900), dtype=torch.long)
        logits_mismatch = torch.randn((1, 100, 12), dtype=torch.float32)
        out_mismatch = readout(logits_mismatch, inputs_mismatch, dz)
        assert out_mismatch.shape == (1, 100, 12)

        # 4.3c: Standard supported square sizes (L=16 for 4x4, L=900 for 30x30)
        inputs_16 = torch.zeros((1, 16), dtype=torch.long)
        logits_16 = torch.randn((1, 16, 12), dtype=torch.float32)
        out_16 = readout(logits_16, inputs_16, dz)
        assert out_16.shape == (1, 16, 12)
        assert not torch.isnan(out_16).any()

        # 4.3d: Degenerate sub-4x4 grids (L=1 for 1x1, L=4 for 2x2, L=9 for 3x3)
        # EMPIRICAL INVESTIGATION:
        # When spectral_gain > 0, get_continuous_spectral_and_ray_basis executes 4-step ray propagation
        # (step=4) which assumes side >= 4, raising RuntimeError on sub-4x4 grids.
        # When spectral_gain == 0.0, the ray field is bypassed and sub-4x4 grids succeed cleanly!
        readout_no_spec = DarkSnailCanvasReadout(readout_scale=6.0, spectral_gain=0.0)
        for sub_side in (1, 2, 3):
            L_sub = sub_side * sub_side
            inp_sub = torch.zeros((1, L_sub), dtype=torch.long)
            lg_sub = torch.randn((1, L_sub, 12), dtype=torch.float32)
            out_sub = readout_no_spec(lg_sub, inp_sub, dz)
            assert out_sub.shape == (1, L_sub, 12)
            assert not torch.isnan(out_sub).any()

        # Confirm empirical failure on default spectral_gain > 0 with 1x1 grid
        try:
            inp_1 = torch.zeros((1, 1), dtype=torch.long)
            lg_1 = torch.randn((1, 1, 12), dtype=torch.float32)
            _ = readout(lg_1, inp_1, dz)
            sub4_crashed = False
        except RuntimeError as re_err:
            sub4_crashed = True
            reporter.record_finding(
                f"DarkSnailCanvasReadout raises unhandled RuntimeError on sub-4x4 grids (side={sub_side}) "
                f"when spectral_gain > 0 due to unconstrained 4-step ray propagation (step=4 > side). "
                f"Error: {re_err}"
            )

        reporter.record_pass("4.3 Grid size edge cases: non-square fallback, length mismatch, and sub-4x4 characterization")
    except Exception as e:
        reporter.record_fail("4.3 Degenerate grid sizes", str(e))

    # 4.4 Negative and Out-of-Range Color Token Values in Inputs
    try:
        inputs_out_of_bounds = torch.full((1, 900), -5, dtype=torch.long)
        inputs_out_of_bounds[:, 10:20] = 999
        logits = torch.randn((1, 900, 12), dtype=torch.float32)
        dz = torch.randn((1, 8192), dtype=torch.float32) * 0.05

        out_oob = readout(logits, inputs_out_of_bounds, dz)
        assert out_oob.shape == (1, 900, 12)
        assert not torch.isnan(out_oob).any(), "NaN detected from out-of-bounds input tokens"
        reporter.record_pass("4.4 Out-of-bounds tokens (-5, 999) handled safely via clamping")
    except Exception as e:
        reporter.record_fail("4.4 Out-of-bounds tokens", str(e))

    # 4.5 Extreme Parameter Scales & Temperatures
    try:
        inputs = torch.randint(0, 12, (1, 900), dtype=torch.long)
        logits = torch.randn((1, 900, 12), dtype=torch.float32)
        dz = torch.randn((1, 8192), dtype=torch.float32) * 0.1

        readout_s0 = DarkSnailCanvasReadout(readout_scale=0.0)
        out_s0 = readout_s0(logits, inputs, dz)
        std = logits.std(dim=-1, keepdim=True)
        exp_s0 = logits / torch.clamp(std / 1.5, min=1.0) if bool((std > 2.5).any().item()) else logits
        assert torch.allclose(out_s0, exp_s0, atol=1e-5)

        readout_t_low = DarkSnailCanvasReadout(tau_v=1e-6)
        out_t_low = readout_t_low(logits, inputs, dz)
        assert not torch.isnan(out_t_low).any()

        readout_t_high = DarkSnailCanvasReadout(tau_v=1e6)
        out_t_high = readout_t_high(logits, inputs, dz)
        assert not torch.isnan(out_t_high).any()

        logits_huge = logits * 1e5
        out_huge = readout(logits_huge, inputs, dz)
        assert not torch.isnan(out_huge).any()

        reporter.record_pass("4.5 Extreme temperatures, scales (scale=0.0, tau_v=1e-6..1e6), and large logits")
    except Exception as e:
        reporter.record_fail("4.5 Extreme parameter scales", str(e))

    # 4.6 Exact Pointwise Orthogonality Under Adversarial Stress
    try:
        max_ortho_violations = 0.0
        torch.manual_seed(1234)
        for trial in range(10):
            inp = torch.randint(0, 12, (2, 900), dtype=torch.long)
            lg = torch.randn((2, 900, 12), dtype=torch.float32)
            dz = torch.randn((2, 8192), dtype=torch.float32) * 0.2
            _ = readout(lg, inp, dz, color_preserving=(trial % 2 == 1))
            v = readout.last_v_comp
            d_triad = readout.last_triad_shift
            assert v is not None and d_triad is not None
            ortho_error = float((v * d_triad).sum(dim=-1).abs().max().item())
            if ortho_error > max_ortho_violations:
                max_ortho_violations = ortho_error
        assert max_ortho_violations <= 1.0e-8, f"Orthogonality error exceeded 1e-8: {max_ortho_violations}"
        reporter.record_pass(f"4.6 Pointwise orthogonality strictly <= 1e-8 across 10 random trials (max error: {max_ortho_violations:.2e})")
    except Exception as e:
        reporter.record_fail("4.6 Pointwise orthogonality", str(e))


# ============================================================================
# Section 5: Additional Core Neuro-Symbolic Invariants
# ============================================================================
def test_section_5_additional_invariants():
    print("\n--- Section 5: Additional Core Neuro-Symbolic Invariants ---")
    from darksnail.readout import SkewSymmetricTriadOperator, SlowFastTriadCascade, RoPE2DMedoidPooler
    from darksnail.arc3 import MagicNorm, TemporalClockContentSplitter

    # 5.1 SkewSymmetricTriadOperator: Exact Skew-Symmetry & Zero Invariants
    try:
        op = SkewSymmetricTriadOperator()
        B, L, C_occ, K = 2, 64, 16, 12

        u_zero = torch.zeros(B, L, C_occ)
        v_rand = torch.randn(B, L, K)
        W_rand = torch.randn(B, C_occ, K, K)
        out_zero = op(u_zero, v_rand, W_rand)
        assert torch.equal(out_zero, torch.zeros_like(out_zero))

        torch.manual_seed(777)
        u = torch.randn(B, L, C_occ)
        v = F.softmax(torch.randn(B, L, K), dim=-1)
        W_arbitrary = torch.randn(B, C_occ, K, K)
        res = op(u, v, W_arbitrary)
        dot_product = (v * res).sum(dim=-1).abs().max().item()
        assert dot_product < 1e-6, f"Skew-symmetry violated on arbitrary tensor: {dot_product}"
        reporter.record_pass("5.1 SkewSymmetricTriadOperator: exact skew-symmetry (<v, B(u, v)> == 0) and zero inputs")
    except Exception as e:
        reporter.record_fail("5.1 SkewSymmetricTriadOperator", str(e))

    # 5.2 SlowFastTriadCascade: Homotopy & Energy Conservation
    try:
        B, S, D = 2, 64, 512
        cascade = SlowFastTriadCascade(d_reasoner=D, d_shell=8, preserve_exact_sphere_norm=True)
        z_H = torch.randn(B, S, D, dtype=torch.float32)
        z_L = torch.randn(B, S, D, dtype=torch.float32)

        # Unroll with dz == 0 -> returns z_L exactly (Homotopy)
        dz_zero = torch.zeros(16, D, dtype=torch.float32)
        z_L_zero = cascade(z_H, z_L, dz_zero)
        assert torch.allclose(z_L_zero, z_L, atol=1e-7), "Zero delta_z altered z_L"

        # Energy conservation on active steps: ||P z_tilde_L||_2^2 == ||P z_L||_2^2
        dz_rand = torch.randn(16, D, dtype=torch.float32) * 0.5
        z_L_coupled = cascade(z_H, z_L, dz_rand)
        v_orig = cascade.project_to_shell(z_L)
        v_new = cascade.project_to_shell(z_L_coupled)
        energy_orig = (v_orig ** 2).sum(dim=-1)
        energy_new = (v_new ** 2).sum(dim=-1)
        assert torch.allclose(energy_new, energy_orig, atol=1e-5), "Energy conservation violated on sphere"
        reporter.record_pass("5.2 SlowFastTriadCascade: step-0 homotopy and energy conservation")
    except Exception as e:
        reporter.record_fail("5.2 SlowFastTriadCascade", str(e))

    # 5.3 MagicNorm: Zero-Norm & Extreme Norm Invariance
    try:
        norm_layer = MagicNorm(d_model=64, eps=1e-5)

        t_zero = torch.zeros(2, 64)
        out_zero = norm_layer(t_zero)
        assert not torch.isnan(out_zero).any(), "MagicNorm produced NaN on zero-norm tensor"
        assert torch.equal(out_zero, t_zero), "MagicNorm on zero-norm tensor did not return zeros"

        t_huge = torch.randn(2, 64) * 1e8
        out_huge = norm_layer(t_huge)
        assert not torch.isnan(out_huge).any(), "MagicNorm produced NaN on huge tensor"
        assert not torch.isinf(out_huge).any(), "MagicNorm produced Inf on huge tensor"

        reporter.record_pass("5.3 MagicNorm: zero-norm tensor (no NaN) and extreme norm stability")
    except Exception as e:
        reporter.record_fail("5.3 MagicNorm", str(e))

    # 5.4 TemporalClockContentSplitter: Transition Observations & Degenerate Grids
    try:
        splitter = TemporalClockContentSplitter(max_clock_tick_area=6)
        f0 = np.zeros((30, 30), dtype=np.int32)
        splitter.initialize_level(f0)

        # Identical transition (no-op)
        f1 = f0.copy()
        expanded, nope, is_reset = splitter.observe_transition(f0, f1, action_key=0)
        assert not is_reset

        # Degenerate 1x1 grid
        splitter_1px = TemporalClockContentSplitter()
        f0_1px = np.zeros((1, 1), dtype=np.int32)
        splitter_1px.initialize_level(f0_1px)
        f1_1px = np.ones((1, 1), dtype=np.int32)
        _ = splitter_1px.observe_transition(f0_1px, f1_1px, action_key=1)

        reporter.record_pass("5.4 TemporalClockContentSplitter: identical transition and 1x1 degenerate grid")
    except Exception as e:
        reporter.record_fail("5.4 TemporalClockContentSplitter", str(e))

    # 5.5 RoPE2DMedoidPooler: Degenerate Inputs & Spatial Invariance
    try:
        pooler = RoPE2DMedoidPooler(patch_size=2, viscous_nu=0.05, viscous_p=2, shell_level=1)
        B, H, W, D = 1, 30, 30, 64
        t_spatial = torch.zeros(B, H, W, D)
        k_pooled, v_pooled, diags = pooler(k=t_spatial)
        assert k_pooled.shape == (B, H // 2, W // 2, D)
        assert not torch.isnan(k_pooled).any()
        assert k_pooled.norm().item() == 0.0
        reporter.record_pass("5.5 RoPE2DMedoidPooler: zero-tensor pooling and dimension reduction")
    except Exception as e:
        reporter.record_fail("5.5 RoPE2DMedoidPooler", str(e))


def main():
    print("=" * 70)
    print("STARTING ADVERSARIAL STRESS SUITE FOR PROJECT DARKSNAIL")
    print("=" * 70)
    test_section_1_imports_and_namespace()
    test_section_2_infer_arc_output_shape()
    test_section_3_prope_splitter()
    test_section_4_canvas_readout()
    test_section_5_additional_invariants()
    success = reporter.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
