# Implementation Action Plan: Project DarkSnail CDS Extension (Multi-Step Spherical Cascade & Continuous Field Readout)

## 0. Pre-Implementation Optimality Audit (Critical Findings from 4-Subagent Codebase Inspection)

Before formulating the execution steps, four specialized subagents audited `darksnail`, `brain-ai`, and `sparse-ai`. They identified **seven critical architectural nuances and latent bugs** that must be addressed for the extension to be mathematically and empirically optimal:

1. **Dual-Repository Lean 4 Synchronization (`OcclusionTriad.lean`):**
   `darksnail/formalization/OcclusionTriad.lean` is mirrored byte-for-byte in the active Lake project at `sparse-ai/formalization/Formalization/Analysis/OcclusionTriad.lean`. Using the discrete-time Pythagorean identity (`shellEnergy_add_of_orthogonal`), we proved $0 < \|v_L\|_2^2 \implies 0 < \|v_L + \delta v_L\|_2^2$ unconditionally—eliminating any need for external non-zero denominator hypotheses in `darksnail_multistep_spherical_invariance`. Both new theorems have already been pre-verified via `lean_run_code` with `0 sorry` and `0` warnings.
2. **`SlowFastTriadCascade` Was Instantiated But Disconnected (`antenna.py:L408` & `callosum.py:L897-L967`):**
   In `darksnail/antenna.py:L408`, `self.cascade = SlowFastTriadCascade()` is initialized with `lm_weight=None` (falling back to a cosine DCT basis instead of the top-8 right singular vectors $P \in \mathbb{R}^{512 \times 8}$ of $\mathbf W_{\mathrm{lm}}$) and is never called inside `_extract_terminal_logits` (`callosum.py:L1334-L1484`) or `_darksnail_canvas_readout_hook`. Furthermore, `_get_pre_terminal_carry` (`callosum.py:L897-L967`) overwrites `prev_carry` in-place, caching only step $15$.
3. **Backward Jacobian Distortion in ULP Refinement (`readout.py:L120-L134`):**
   In `SkewSymmetricTriadOperator.forward`, the 12-iteration coordinate-wise ULP refinement loop differentiates through `corr = cur_rem / safe_vj`, which multiplies the backward gradient by an asymmetric rank-1 oblique projection $(I - \frac{1}{v_j} e_j v^\top)$ and adds $M$ sequential clone/where autograd nodes per step (compounding $16\times$ at $K=16$). Wrapping the ULP polish as a **straight-through estimator** (`out = out + (out_corr - out).detach()`) preserves exact forward IEEE-754 orthogonality ($\le 1.0 \times 10^{-8}$) while keeping the exact symmetric tangent-space Jacobian $(I - \hat{v}\hat{v}^\top)$ in the backward pass.
4. **`2,414` Unwired Coordinates in `flat_dz[:, 5658:8072]` (`readout.py:L790-L805`):**
   While `[5322:5466]` (`u_bg_fg`, 144 coords) and `[5466:5658]` (`u_occ_rel`, 192 coords) are sliced, **`[5658:8072]` (`2,414` coords) are currently unreferenced in `readout.py`**, receiving zero gradient! We wire `[5658:8072]` (`201 * 12 + 2 = 2,414` coords) to drive the Continuous Spectral / Phase-Space Readout and Extended Multi-Bounce Ray-March / Enclosure channels.
5. **Contract Violation in `_pinv_cache` (`readout.py:L652` & `callosum.py:L1203`):**
   `_locate_lm_weight` (`readout.py:L678`) returns `head_mod.weight.detach().float()`, allocating a temporary `float32` tensor whenever `head_mod.weight` is `bfloat16`, and `_get_lm_pinv` keys `_pinv_cache` by `int(lm_weight.data_ptr())`—violating Invariant #5 and risking use-after-free pointer collisions. All caches (`_basis_cache`, `_pinv_cache`, and spectral caches) are upgraded to content-addressed `hashlib.blake2b` digests.
6. **Dynamic-Shape Demonstration Horizon Mismatch (`callosum.py:L1609-L1610`):**
   When demonstrations have variable output shapes (`len(set(out_shapes)) > 1`), `active_target_hw` is set to `None` during TTA, causing `readout.py:L279` to fall back to the *input* grid dimensions `(hb, wb)`. Supporting per-sample `target_hw: Sequence[Tuple[int, int]]` in `get_spatial_relational_and_occlusion_tensors` aligns `in_box`, `eos_border`, and `pad_ext` on dynamic-shape tasks while preserving `0.000e+00` Step-0 homotopy.
7. **Hidden LR/Anchor Clamp in `callosum.py:L1635-L1637`:**
   `System2LatentAdaptor.adapt_task` clamps `step_lr = max(step_lr, 0.05)` and `anchor_wt = min(anchor_wt, 0.1)` on $30 \times 30$ grids, silently overriding `ZenoAmygdalaController`'s surgical regime ($\eta \in [0.022, 0.036]$, $\lambda_{\mathrm{anchor}} \in [0.22, 0.30]$). We add a `respect_schedule: bool = True` flag when dispatched from `DarkSnailSystem2Adaptor` so surgical schedules are honored verbatim.

---

## 1. Subagent Orchestration & Parallel Execution Topology

Execution runs across **6 specialized subagents** in two coordinated waves (4 parallel implementation subagents in Wave 1 with strictly disjoint file ownership, followed by 2 verification & benchmark subagents in Wave 2):

- **Wave 1 (Disjoint Parallel Implementation):**
  - **Subagent 1 (`Lean 4 Formalization Auditor & Prover`):** `darksnail/formalization/OcclusionTriad.lean` & `sparse-ai/formalization/Formalization/Analysis/OcclusionTriad.lean`.
  - **Subagent 2 (`Readout & Cascade Engineer`):** `darksnail/darksnail/readout.py` & `darksnail/darksnail/__init__.py` (Workstream B + Workstream C.2).
  - **Subagent 3 (`Callosal Shell & Shape Predictor Engineer`):** `darksnail/darksnail/shell.py` & `darksnail/scripts/evaluate_darksnail_arc.py` (Workstream C.1 + evaluation harness integration).
  - **Subagent 4 (`Zeno Controller & Callosum Trajectory Engineer`):** `darksnail/darksnail/antenna.py` & `brain-ai/brain_ai/models/callosum.py` (Workstream D + multi-step carry caching + `infer_arc_output_shape` sync).
- **Wave 2 (Verification Gates & Benchmarks):**
  - **Subagent 5 (`Unit & Invariant Gate Engineer`):** `darksnail/tests/test_darksnail.py` ($N = 3,600$ `float32` cell bench across $K \in \{1, 4, 8, 16\}$, orthogonality $\le 1.0 \times 10^{-8}$, radial drift `== 0.000e+00`, Step-0 homotopy `== 0.000e+00`).
  - **Subagent 6 (`ARC Regression & Colab Pro+ Packaging Engineer`):** `27 / 27` ARC-AGI-1 regression gate, `419`-pair shape recovery benchmark, and `notebooks/darksnail_v2_colab.ipynb` update for live `HF_TOKEN` / `Meta-Llama-3.1-8B-Instruct` sweeps.

---

## 2. Detailed Technical Specification by Workstream

### Workstream A — Lean 4 Formalization
- **Target Files:**
  - `darksnail/formalization/OcclusionTriad.lean`
  - `sparse-ai/formalization/Formalization/Analysis/OcclusionTriad.lean`
- **New Definitions & Machine-Checked Theorems:**
  1. `tangentTriadStep`, `sphericalRetract`, `sphericalTriadStep`, and `multistepSphericalTriad`.
  2. Supporting Lemmas: `shellEnergy_nonneg`, `shellEnergy_eq_zero_iff`, `tangentTriadStep_orthogonal`, `shellEnergy_add_of_orthogonal`, `shellEnergy_scale`, `sphericalTriadStep_preserves_energy`, `sphericalTriadStep_zero`.
  3. **`darksnail_multistep_spherical_invariance` (Theorem 3b):**
     Proves by induction on $T \in \mathbb{N}$ that $\|\tilde v_L^{(T)}\|_2^2 = \|v_L^{(0)}\|_2^2 \text{ and } \forall t < T,\; \langle v_L^{(t)}, \delta v_L^{(t)} \rangle = 0$.
  4. **`darksnail_multistep_step0_homotopy` (Theorem 4b):**
     Proves by induction on $T \in \mathbb{N}$ that at $W_{\mathrm{rec}} = 0$ and $\delta z = (0, 0)$, `multistepSphericalTriad M γ 0 u v₀ T = v₀` and `darksnailCarryTrajectory step_S1 γ 0 0 0 z₀ T = system1CarryTrajectory step_S1 z₀ T`.

### Workstream B — Multi-Step Unrolled `SlowFastTriadCascade` (`darksnail/readout.py`)
- **Guard Buffer Allocation (`flat_dz[:, 8072:8192]`, `120` coords):**
  - `8072:8088` (`16` coords): Per-step temporal modulation weights $\boldsymbol{\alpha} \in \mathbb R^{16}$ ($t \in \{0,\dots,15\}$), entering as $\gamma_t = \gamma_{\mathrm{cas}}(1 + \alpha_t)$.
  - `8088:8104` (`16` coords): Per-step adaptive step-size / Lie-generator mixing scalars $\beta_t \in \mathbb{R}^{16}$.
  - `8104:8168` (`64` coords): Step-specific Lie-algebra skew modulation basis $\Delta W_{\mathrm{step}} \in \mathbb{R}^{8 \times 8}$.
  - `8168:8192` (`24` coords): Continuous spectral phase-space dispersion & horizon damping scalars ($2 \times 12$ color channels).
- **Lie-Group Tangent Stepping & Zero-Drift Retraction:**
  - Project onto top $M = 8$ right singular vectors $P \in \mathbb R^{512 \times 8}$ ($P^\top P = I$) of $\mathbf W_{\mathrm{lm}}$.
  - Apply two-pass Kahan/Parlett tangent re-orthogonalization + straight-through ULP refinement (`out + (out_corr - out).detach()`).
  - Apply spherical retraction with post-retraction radial polish so $\|\tilde v_L^{(T)}\|_2^2 - \|v_L^{(0)}\|_2^2 == 0.000\mathrm{e+}00$ in `float32`.
  - Lift back via $\tilde z_L^{(t)} = z_L^{(t)} + (\tilde v_L^{(t)} - z_L^{(t)} P) P^\top$.

### Workstream C — Continuous Dynamical Field Readout & Outlier-Guided Shape Predictor (`darksnail/shell.py` & `darksnail/readout.py`)
- **Beltrami Outlier Count Wiring (`shell.py`):**
  - Export thresholded salience spike cardinalities (`salience_spike_count`, `raw_spike_count`, `outlier_score`) from `BeltramiCallosalShell.forward` out-of-band without altering the `(B, 32, 4096)` `float32` return tensor.
  - Implement `extract_grid_beltrami_shape_descriptors` and `infer_arc_output_shape_beltrami` (Rules 8a–8d after Rules 1–7) to resolve dynamic object-count and enclosure bounding-box cases.
- **Continuous Spectral / Phase-Space Readout (`ContinuousSpectralFieldReadout` in `readout.py`):**
  - Harvest `flat_dz[:, 5658:8072]` (`2,414` coords = `201 * 12 + 2`) and `8168:8192` (`24` coords) together with `[0:5658]` and `RoPE2DMedoidPooler`, with $\gamma_{2\mathrm{D}}(S) = \exp(-\nu p^{2\ell}(\sigma_y^2(S) + \sigma_x^2(S))) \in [0.05, 1.0]$.
  - Replace all `data_ptr()` and salted `hash()` cache keys with `hashlib.blake2b`.

### Workstream D — Adaptive Unroll Dispatch in `ZenoAmygdalaController` (`darksnail/antenna.py`)
- **Surgical Near-Solved Regime** ($\mathcal L_0 < 2.5$ or S1 Fit $\ge 0.90$):
  - `unroll_steps = 1`, with schedule parameters $N_{\mathrm{TTA}} \in [12, 18],\; \eta \in [0.022, 0.036],\; \lambda_{\mathrm{anchor}} \in [0.22, 0.30]$.
  - Surgical ray quantiles: $[1/N_{\mathrm{cells}}, 2/N_{\mathrm{cells}}, 0.02]$.
- **High-Turbulence / Multi-Hop Regime** ($\mathcal L_0 > 4.5 \text{ or } \Phi_{\mathrm{LH}} \ge 0.55 \land \mathcal T_{\mathrm{zeno}} \ge 0.55$):
  - `unroll_steps` $K \in \{4, 8, 16\}$ based on $(\mathcal T_{\mathrm{zeno}}, \mathcal L_0)$.
  - TTA adaptation budget: $N_{\mathrm{TTA}} \in [33, 40]$.
- **Balanced Deliberation Regime** ($2.5 \le \mathcal L_0 \le 4.5$):
  - `unroll_steps = 4 if (t_zeno >= 0.45 or has_occlusion_tie) else 1`.
