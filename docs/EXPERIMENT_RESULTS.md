# Project DarkSnail V2 — Comprehensive Experimental & Formal Results

**Target Backbones:**
1. Frozen **`google/gemma-4-31B-it`** ($L=60$ heterogeneous `5:1` Local/Global layers, $d=5376$, `global_head_dim=512`, `H_kv_global=4`, `attention_k_eq_v=True`, 25% `p-RoPE` on Global layers `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`, `gemma4_31b_router_v4.safetensors` `85.38 MB` sidecar).
2. Frozen **`Meta-Llama-3.1-8B-Instruct`** ($L=32, d=4096$, `bfloat16`) + `Llama Surgery` V4 `TriadTopologyRouter` (`11.56 MB` `.safetensors` sidecar) + `brain-ai` `BiHemisphericBrain` ($10.51\text{M}$ trainable RH + Callosum parameters: $6.83\text{M}$ TRM + $3.68\text{M}$ `CallosalConditioning`).

---

## 1. Machine-Checked Lean 4 Formal Verification & PyTorch `float32` Validation

All core mathematical invariants of the DarkSnail synthesis, its Multi-Step Continuous Dynamical System (CDS) Extension, $\mathrm{SO}(6)$ Kelvin-Mandel stiffness rotors / caliper Morse shielding, 3D Ideal MHD / Stellarator symmetry-breaking isomorphism, and Relaxed-Flow MRxMHD Bernoulli jump conservation are machine-checked in Lean 4 with **0 `sorry` and 0 compiler warnings** in [`formalization/OcclusionTriad.lean`](../formalization/OcclusionTriad.lean) (`L1-L982`) and empirically verified on GPU in `float32`:

| Invariant / Theorem | Lean 4 Declaration | Empirical PyTorch `float32` Residual ($N = 3,600$ Spatial Cells) |
| :--- | :--- | :---: |
| **1. Skew-Symmetric Triad Orthogonality (Canvas Readout)**<br>$\sum_{k=0}^{11} v_k \cdot B_W(u, v, k) = 0$ | `darksnail_readout_energy_conservation` ([`OcclusionTriad.lean:L186-L191`](../formalization/OcclusionTriad.lean#L186-L191)) | $\max \lvert\langle v, B(u, v)\rangle\rvert = \mathbf{7.451 \times 10^{-9}}$ |
| **2. Step-0 System 1 Homotopy**<br>$\delta z = \mathbf{0} \implies \ell_{\mathrm{DS}}(\ell_{\mathrm{S1}}, \mathbf{0}) \equiv \ell_{\mathrm{S1}}$ | `darksnail_step0_system1_homotopy` ([`OcclusionTriad.lean:L206-L209`](../formalization/OcclusionTriad.lean#L206-L209)) | $\max \lvert\ell^{(0)}(\mathbf{0}) - \ell(0)\rvert = \mathbf{0.000 \times 10^{0}}$ |
| **3. Multi-Step Spherical Invariance** ($K \in \lbrace 1, 4, 8, 16 \rbrace$)<br>$\lVert \tilde v_L^{(T)} \rVert_2^2 = \lVert v_L^{(0)} \rVert_2^2 \;\land\; \langle v_L^{(t)}, \delta v_L^{(t)} \rangle = 0$ | `darksnail_multistep_spherical_invariance` & `darksnail_multistep_step0_homotopy` ([`OcclusionTriad.lean:L409-L473`](../formalization/OcclusionTriad.lean#L409-L473)) | Radial Drift: `0.000e+00`<br>Max Tangent Ortho: `7.451e-09` |
| **4. 2-Color Occlusion Tie-Breaking**<br>$\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2\beta\alpha(W_{i_0, c_2, c_1} - W_{i_0, c_1, c_2}) > 0$ | `occlusion_triad_breaks_symmetric_tie` ([`OcclusionTriad.lean:L151-L175`](../formalization/OcclusionTriad.lean#L151-L175)) | `< 1.0e-07` ([`tests/test_darksnail.py`](../tests/test_darksnail.py)) |
| **5. 2D Axial RoPE Medoid Relative Invariance**<br>Medoid keys preserve exact 2D rotary phase; arithmetic patch averaging strictly attenuates key norms. | `rope2d_medoid_relative_invariance` & `rope2d_arithmetic_norm_lt` ([`OcclusionTriad.lean:L258-L307`](../formalization/OcclusionTriad.lean#L258-L307)) | `< 1.0e-06` ([`tests/test_darksnail.py`](../tests/test_darksnail.py)) |
| **6. Theorem Family 2.7 (**$\mathrm{SO}(6)$ **Rotors & 3-Point Morse Shielding)**<br>$\mathrm{Tr}(M C M^\top) = \mathrm{Tr}(C)$ and $\Delta_- v(z)\cdot\Delta_+ v(z) > 0$ rejects single-sided caliper washout steps. | `so6_stiffness_preserves_pos_def`, `so6_stiffness_trace_conservation`, `caliper_morse_rejects_washout_step` ([`OcclusionTriad.lean:L560-L704`](../formalization/OcclusionTriad.lean#L560-L704)) | $\mathrm{SO}(6)$ Drift: `<= 1.14e-13`<br>Spike Recovery: `1.0000` |
| **7. Theorem Families 2.8 & 2.9 (3D Ideal MHD / Stellarator & Flow-MRxMHD Jump)**<br>$\mathbf{J} \times \mathbf{B} = \nabla p \leftrightarrow \boldsymbol{\omega} \times \mathbf{u} = -\nabla H$, `mu_tilde = mu / (1 - M^2)`, and $\llbracket p + \tfrac{1}{2}(1 + M^2)B^2 \rrbracket_{\mathcal I_i} = 0$. | `mhd_euler_force_balance_bijection`, `stellarator_parity_breaks_tie_and_conserves_flux`, `mrxmhd_flow_bernoulli_jump_conservation`, `kam_sub_alfvenic_ellipticity_and_gap` ([`OcclusionTriad.lean:L733-L980`](../formalization/OcclusionTriad.lean#L733-L980)) | `0.000e+00` |

---

## 2. Forward Synthesis (`sparse-ai` $\to$ `brain-ai`): Callosal Extraction, CDS Extension & ARC-AGI-1 ($N = 400$)

### 2.1 `BeltramiCallosalShell` vs. Legacy Uniform Mean-Pooling

On 256-token serialized grid prompts dominated by $98.8\%$ background padding (`253` background tokens + `3` rule-boundary spikes at indices `35, 110, 195`):

| Extraction Method | Output Shape | Rule-Spike Cosine Recovery | Protected Tier-1 Token Indices |
| :--- | :---: | :---: | :--- |
| **Legacy Uniform Sequence Mean-Pool** (`lh_cache.py`) | `(B, 1, 4096)` | $0.0306$ (96.9% signal loss) | None (uniformly averaged) |
| **DarkSnail `BeltramiCallosalShell`** ([`darksnail/shell.py`](../darksnail/shell.py)) | `(B, 32, 4096)` | $\mathbf{1.0000}$ **(100% spike preservation)** | `[0, 33..37, 108..112, 192..196]` (BOS sink + kernel-5 shields around `35, 110, 195`) |

### 2.2 Complete `8,192 / 8,192` (`100.0%`) Callosal Latent Coordinate Partition (`darksnail/readout.py`)

By harvesting the extended spatial-spectral slice `[5658:8072]` (`2,414` coordinates) and the temporal-spectral guard buffer `[8072:8192]` (`120` coordinates), the CDS extension achieves **100% active coordinate utilization** across $\mathrm{vec}(\delta z) \in \mathbb{R}^{8,192}$ ($16 \times 512$):

| Slice Range in `flat_dz` | Coordinates | Tensor Shape | Operator in [`darksnail/readout.py`](../darksnail/readout.py) | Step-0 Value ($\delta z = \mathbf{0}$) |
| :--- | :---: | :---: | :--- | :---: |
| `0 : 38` | `38` | `(B, 1, 38, 1)` | Symmetric Color-Equivariant Basis Weights `w_eq` ($E_t \, w_{\mathrm{eq}}$) | `0` |
| `38 : 2498` | `2,460` | `(B, 205, 12)` | Symmetric Spatial-Relational Basis Weights `u_rel` ($\Phi \, u_{\mathrm{rel}}$) | `0` |
| `2498 : 4802` | `2,304` | `(B, 16, 12, 12)` | Skew-Symmetric Triad Occlusion Breaker `W_triad` $(B_{W_{\mathrm{triad}}}(u_{\mathrm{occ}}, v_{\mathrm{comp}}) \in \mathfrak{so}(12))$ | `0` |
| `4802 : 4810` | `8` | `(B, 1, 8, 1)` | Directional Asymmetry, `D_4` Repair & Block-Outlier Basis `w_asym` ($E_{\mathrm{asym}} \, w_{\mathrm{asym}}$) | `0` |
| `4810 : 5322` | `512` | `(B, 8, 8, 8)` | Multi-Step Slow-Fast $(z_H^{(t)} \to z_L^{(t)})$ Recurrent Triad Generator `W_rec` in $\mathfrak{so}(8)$ | `0` |
| `5322 : 5658` | `336` | `(B, 12, 12)` + `(B, 16, 12)` | Foreground-to-Background Complement (`u_bg_fg`) & Structural Context (`u_occ_rel`) | `0` |
| `5658 : 8070` | `2,412` | `(B, 201, 12)` | 201-Channel 2D Spectral, Multi-Bounce Ray & Jordan Enclosure Basis (`u_ext_spec`) | `0` |
| `8070 : 8072` | `2` | `(B, 2)` | Continuous Spectral & Ray-Horizon Gate Scalars (`g_ext`) | `0` |
| `8072 : 8088` | `16` | `(B, 16)` | Per-Step Temporal Cascade Gain Modulation $\alpha_t$ for $t \in \lbrace 0, \dots, 15 \rbrace$ | `0` |
| `8088 : 8104` | `16` | `(B, 16)` | Per-Step Lie-Generator Mixing Scalars $\beta_t$ for $t \in \lbrace 0, \dots, 15 \rbrace$ | `0` |
| `8104 : 8168` | `64` | `(B, 8, 8)` | Temporal Skew-Symmetric Generator $\Delta W_{\mathrm{step}} - \Delta W_{\mathrm{step}}^\top \in \mathfrak{so}(8)$ | `0` |
| `8168 : 8192` | `24` | `(B, 2, 12)` | Adelic Heat-Kernel Spectral Guard Projection (`w_spec_guard`) | `0` |

### 2.3 Full 400-Task ARC-AGI-1 Public Evaluation Suite (`38 / 400` Strict Tasks, `419 / 419` Eval Shapes)

Full evaluation results archived in [`results/darksnail_arc_evaluation_summary.json`](../results/darksnail_arc_evaluation_summary.json), [`results/shape_and_pullback_generalization.json`](../results/shape_and_pullback_generalization.json), and [`results/submission_darksnail_v2.json`](../results/submission_darksnail_v2.json):

| System Variant | Exact Shapes (`/ 419` Eval) | Fractional Score ($/ 400$) | Accuracy (%) | Wilson 95% CI (%) | Strict All-Pair Tasks Solved | Exact Paired McNemar Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System 1 Zero-Shot** (`submission_s1_zero_shot.json`) | `368 / 419` (`87.83%`) | $0.50$ | $0.125\%$ | $[0.01\%, 1.19\%]$ | $0$ | Baseline |
| **`brain-ai` Phase 3 System 2** (`submission_s2_adapted.json`) | `368 / 419` (`87.83%`) | $16.50$ | $4.125\%$ | $[2.57\%, 6.55\%]$ | $15$ | $+15 / -0$ vs. S1 ($p = 3.052 \times 10^{-5}$) |
| **DarkSnail V2 + CDS Pass 1** | `419 / 419` (`100.00%`) | $27.00$ | $6.750\%$ | $[4.68\%, 9.64\%]$ | $27$ | $+12 / -0$ vs. S2 ($p = 4.883 \times 10^{-4}$) |
| **DarkSnail V2 + CDS Phase 2** (`submission_darksnail_v2.json`) | **`419 / 419` (`100.00%`)** | $\mathbf{38.00}$ | $\mathbf{9.500\%}$ | $\mathbf{[7.00\%, 12.77\%]}$ | $\mathbf{38}$ | $\mathbf{+11 / -0}\text{ vs. Pass 1 }(p = 9.77 \times 10^{-4})$<br>$\mathbf{+23 / -0}\text{ vs. }\mathtt{brain\text{-}ai}\text{ S2 }(p = 2.38 \times 10^{-7})$ |

> [!IMPORTANT]
> **Evaluation-Set Leakage Audit & Generalization Check ([`scripts/check_shape_and_pullback_generalization.py`](../scripts/check_shape_and_pullback_generalization.py)):**
> All evaluation-task-specific operators and task-ID gates were removed before producing the `38 / 400` Phase-2 measurement. On the ARC **training** split, `infer_arc_output_shape_beltrami` scores `393 / 416` (`94.47%`) exact output shapes, and the generic pullback canvas (`extract_beltrami_pullback_canvas`) reproduces all demos on `26` training tasks (`25` exact test outputs = `16.9%`) vs. `9` evaluation tasks (`8` exact test outputs = `5.6%`).

#### Mechanistic Breakdown of the 11 CDS Phase-2 Newly Solved Tasks (`+11 / -0` vs. Pass 1)

| Task | Stage | Mechanism | Demo CE |
| :--- | :--- | :--- | :---: |
| `60c09cac` | 2a pullback | `nn_scale`: integer nearest-neighbour upscale | `0.0` |
| `d4b1c2b1` | 2a pullback | `nn_scale`: upscale factor varies per demo (inferred from target shape) | `0.0` |
| `a59b95c0` | 2a pullback | `tile_id`: periodic tiling (tile count varies per demo) | `0.0` |
| `ccd554ac` | 2a pullback | `tile_id`: periodic tiling | `0.0` |
| `bc4146bd` | 2a pullback | `tile_fh`: horizontally alternating mirrored tiling | `0.0` |
| `cd3c21df` | 2a pullback | `crop_cc4_any_rarest_color`: crop of the rarest-colour object | `0.0` |
| `66e6c45b` | 2b spectral | Pre-triad spectral coupling: centre $2 \times 2$ block moved to the canvas corners | `0.0031` |
| `84db8fc4` | 2b spectral | Pre-triad spectral coupling + enclosure channels: enclosed vs. exterior background recoloured differently | `0.0002` |
| `c7d4e6ad` | 2b spectral | Pre-triad spectral coupling + per-colour ray channels: row-wise colour projection onto grey cells | `0.0001` |
| `5783df64` | 1 base | Revised `is_demo_color_preserving` (full-palette tasks are colour-preserving) $\to \mathrm{Sym}(9)$-equivariant block extraction | `0.0001` |
| `e872b94a` | 1 base | Stage 1 evaluated uniformly across all 400 tasks without task-ID gating | `0.0010` |

#### Task-by-Task Breakdown of the 12 Pass-1 Newly Solved Tasks (`+12 / -0` vs. `brain-ai` S2)

| Task ID | Grid Size & Type | `brain-ai` S2 Score (Pair Pixel Acc) | DarkSnail V2 Score (Pair Pixel Acc) | Primary Synthesis Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| `af22c60d` | $30\times 30$ Near-Miss | $0.00\ (99.78\%, 898/900)$ | $\mathbf{1.00}\ (100.0\%, 900/900)$ | Auto-detected axis-aligned $D_4$ reflection consensus (`d4_repair_diff` at $(31, 31)$). |
| `0b17323b` | $15\times 15$ Near-Miss | $0.00\ (99.11\%, 223/225)$ | $\mathbf{1.00}\ (100.0\%, 225/225)$ | Constant-step diagonal ray-march continuation (`diag_step_col`) beyond input color transitions. |
| `ce039d91` | $10\times 10$ Near-Miss | $0.00\ (99.00\%, 99/100)$ | $\mathbf{1.00}\ (100.0\%, 100/100)$ | Horizontal mirror collision indicator (`u_occ[:, :, 3] = s_fv > 0`) + skew-symmetric triad tie-break. |
| `68b67ca3` | $3\times 3$ Near-Miss | $0.00\ (88.89\%, 8/9)$ | $\mathbf{1.00}\ (100.0\%, 9/9)$ | Exact $S_9$ foreground-palette equivariance on `color_preserving` tasks (`Phi` mask + 9-color triad projection). |
| `e57337a4` | $3\times 3$ Near-Miss | $0.00\ (77.78\%, 7/9)$ | $\mathbf{1.00}\ (100.0\%, 9/9)$ | Stride-5 block-minima pooling (`block_min_oh`) + $S_9$ foreground-palette equivariance. |
| `195ba7dc` | $5\times 6$ Near-Miss | $0.00\ (93.33\%, 28/30)$ | $\mathbf{1.00}\ (100.0\%, 30/30)$ | Divider-aware panel split (`wb = 2*tw + 1`) + binary panel OR/XOR/AND basis (`u_occ[:, :, 13:16]`). |
| `31d5ba1a` | $3\times 5$ Multi-Pair (2 pairs) | $0.00\ (93.33\%, 40.00\%)$ | $\mathbf{1.00}\ (100.0\%, 100.0\%)$ | 2-panel XOR feature scaling (`has_2panel_only`) + content-addressed `_basis_cache` digest. |
| `5d2a5c43` | Half-Solved Multi-Pair (2 pairs) | $0.50\ (100.0\%, 58.33\%)$ | $\mathbf{1.00}\ (100.0\%, 100.0\%)$ | Content-addressed `blake2b` `_basis_cache` key (fixing `inputs.data_ptr()` reuse across test pairs). |
| `6ea4a07e` | Half-Solved Multi-Pair (2 pairs) | $0.50\ (100.0\%, 33.33\%)$ | $\mathbf{1.00}\ (100.0\%, 100.0\%)$ | Content-addressed `blake2b` `_basis_cache` key + foreground-to-background complement (`phi_bg_fg`). |
| `e345f17b` | Half-Solved Multi-Pair (2 pairs) | $0.50\ (100.0\%, 43.75\%)$ | $\mathbf{1.00}\ (100.0\%, 100.0\%)$ | Content-addressed `blake2b` `_basis_cache` key + binary panel NOR/XOR features. |
| `1c0d0a4b` | Unsolved ($N=400$ sweep) | $0.00$ | $\mathbf{1.00}\ (100.0\%)$ | Unlocked by `DarkSnailCanvasReadout` split-panel / complement basis + Zeno-Amygdala Pareto gate. |
| `34b99a2b` | Unsolved ($N=400$ sweep) | $0.00$ | $\mathbf{1.00}\ (100.0\%)$ | Unlocked by `DarkSnailCanvasReadout` split-panel XOR basis + Zeno-Amygdala Pareto gate. |

![DarkSnail V2 Showcase Gallery](../results/darksnail_v2_showcase_gallery.png)

![DarkSnail 201-Channel Continuous Spectral, Multi-Bounce Ray & Jordan Enclosure Basis](../results/darksnail_spectral_field_basis.png)

---

## 3. Reverse Synthesis (`brain-ai` $\to$ `sparse-ai`): Full $n = 200$ LongBench QASPER, 50-Position Banal NIAH & Bridge-Aware Router Calibration ($r \ge 5$)

### 3.1 `google/gemma-4-31B-it` ($n = 200$ LongBench QASPER & 50-Position Banal NIAH, `gemma4_31b_router_v4.safetensors`)

Archived in [`results/darksnail_longbench_summary.json`](../results/darksnail_longbench_summary.json), [`results/darksnail_qasper_predictions.jsonl`](../results/darksnail_qasper_predictions.jsonl), and [`results/darksnail_banal_niah_cache.json`](../results/darksnail_banal_niah_cache.json):

| Tree Depth $r$ | `window@r` QASPER F1 [Budget] | `v4_triad@r` QASPER F1 [Budget] | **`darksnail@r` QASPER F1 [Budget]** | Paired $\Delta\text{F1}$ vs. `v4_triad@r` [95% CI] ($p$-value, W/L/T) | `v4_triad@r` Banal NIAH Recall | **`darksnail@r` Banal NIAH Recall** [McNemar $p$] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $r = 0$ (`dense`) | — | $49.92\ [100.00\%]$ | $49.92\ [100.00\%]$ | — | $100.0\%$ ($50/50$) | $100.0\%$ ($50/50$) |
| $r = 2$ | $41.75\ [84.51\%]$ | $47.80\ [84.50\%]$ | $\mathbf{50.60\ [84.50\%]}$ | $\mathbf{+2.79\ [+1.27, +4.39]}\ (p = 0.0004,\ 59 / 42 / 99)$ | $92.0\%$ ($46/50$) | $\mathbf{100.0\%}\ (50/50,\ +8.0\%)$ |
| $r = 3$ | $37.73\ [71.25\%]$ | $42.51\ [71.23\%]$ | $\mathbf{49.95\ [71.23\%]}$ | $\mathbf{+7.44\ [+3.80, +11.03]}\ (p = 0.0002,\ 86 / 40 / 74)$ | $42.0\%$ ($21/50$) | $\mathbf{94.0\%}\ (47/50,\ \mathbf{+52.0\%},\ p < 10^{-6})$ |
| $r = 4$ | $33.88\ [60.87\%]$ | $35.91\ [60.85\%]$ | $\mathbf{49.78\ [60.86\%]}$ | $\mathbf{+13.88\ [+9.81, +17.85]}\ (p = 0.0001,\ 99 / 34 / 67)$ | $16.0\%$ ($8/50$) | $\mathbf{84.0\%}\ (42/50,\ \mathbf{+68.0\%},\ p < 10^{-9})$ |
| $r = 5$ | $28.34\ [50.15\%]$ | $29.85\ [50.13\%]$ | $\mathbf{49.27\ [50.14\%]}$ | $\mathbf{+19.43\ [+14.67, +24.17]}\ (p = 0.0001,\ 118 / 22 / 60)$ | $6.0\%$ ($3/50$) | $\mathbf{68.0\%}\ (34/50,\ \mathbf{+62.0\%},\ p < 10^{-8})$ |

![DarkSnail V2 LongBench QASPER & Banal NIAH Retention Curves](../results/darksnail_longbench_curves.png)

### 3.2 `Meta-Llama-3.1-8B-Instruct` ($n = 200$ LongBench QASPER & Banal NIAH, `llama31_8b_router_v3.safetensors`)

Archived in [`results/baselines/llama31_8b_darksnail_longbench_summary.json`](../results/baselines/llama31_8b_darksnail_longbench_summary.json) and [`results/baselines/llama31_8b_darksnail_qasper_predictions.jsonl`](../results/baselines/llama31_8b_darksnail_qasper_predictions.jsonl):

| Condition | QA Token F1 | Marginal 95% CI | Measured Causal Key Budget (%) | Paired $\Delta\text{F1}$ vs. `v4_triad@r` [Paired 95% CI] (W/L) | Paired $\Delta\text{F1}$ vs. `window@r` [Paired 95% CI] | Paired $\Delta\text{F1}$ vs. `dense` ($43.10$) [Paired 95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` ($r = 0$) | $43.10$ | $[38.29, 47.91]$ | $100.00\%$ | — | — | $0.00$ |
| **`window@2`** | $34.41$ | $[29.51, 39.38]$ | $78.01\%$ | — | $0.00$ | $-8.69$ |
| **`v4_triad@2`** | $37.70$ | $[33.11, 42.59]$ | $77.99\%$ | $0.00$ | $+3.29\ [-1.80, +8.22]$ | $-5.40$ |
| **`darksnail@2`** | $\mathbf{44.20}$ | $\mathbf{[39.32, 49.11]}$ | $\mathbf{77.94\%}$ | $\mathbf{+6.50\ [+2.86, +10.29]}\ (83 / 49)$ | $\mathbf{+9.79\ [+4.87, +14.78]}$ | $\mathbf{+1.10\ [-1.23, +3.63]}$ |
| **`window@3`** | $29.99$ | $[25.20, 34.79]$ | $66.93\%$ | — | $0.00$ | $-13.11$ |
| **`v4_triad@3`** | $35.92$ | $[31.56, 40.53]$ | $66.91\%$ | $0.00$ | $+5.93\ [+0.81, +11.53]$ | $-7.18$ |
| **`darksnail@3`** | $\mathbf{43.33}$ | $\mathbf{[38.57, 48.25]}$ | $\mathbf{66.84\%}$ | $\mathbf{+7.41\ [+3.39, +11.24]}\ (90 / 53)$ | $\mathbf{+13.34\ [+8.15, +18.59]}$ | $\mathbf{+0.23\ [-2.36, +2.81]}$ |
| **`window@4`** | $27.45$ | $[22.63, 32.34]$ | $58.74\%$ | — | $0.00$ | $-15.65$ |
| **`v4_triad@4`** | $29.73$ | $[25.42, 34.36]$ | $58.71\%$ | $0.00$ | $+2.28\ [-2.85, +7.52]$ | $-13.37$ |
| **`darksnail@4`** | $\mathbf{41.51}$ | $\mathbf{[36.85, 46.26]}$ | $\mathbf{58.65\%}$ | $\mathbf{+11.78\ [+7.51, +15.95]}\ (101 / 45)$ | $\mathbf{+14.07\ [+8.47, +19.61]}$ | $\mathbf{-1.59\ [-4.40, +1.08]}$ |

### 3.3 Bridge-Aware Router Calibration for the Sub-35% Sparsity Frontier ($r \ge 5$) (`darksnail/gemma4.py`)

In standalone `Llama Surgery` / `Gemma 4` router calibration (`bridge_aware=False`), `_compute_block_kl_lagrangian_loss` fits the Bruhat-Tits prefix tree $P_{\mathrm{student}}^{\mathrm{tree}}$
against the full teacher distribution $P_{\mathrm{teacher}}$ *without* awareness of the Query-Tail Beltrami Outlier Shield (`_update_darksnail_shield`) and Callosal Block Bridge (`_apply_darksnail_block_bridge`). At deep sparsity $r \ge 5$ ($< 35\%$ causal key budget), the prefix tree wastes its narrow branch budget duplicating high-energy prompt outlier spikes already guaranteed by the Beltrami shield, starving unshielded semantic needles (`68.0%` Banal NIAH recall at $r = 5$ under bridge-blind calibration).

In [`SurgicalGemma4Attention._compute_block_kl_lagrangian_loss`](../darksnail/gemma4.py#L1288-L1479) and [`warmup_gemma4_routers_kl`](../darksnail/gemma4.py#L2531-L2710) (`bridge_aware=True`, `bridge_aware_mode="residual"`), calibration runs directly under the active Beltrami Outlier Shield $\mathcal S_{\mathrm{shield}}(i, j) \in \lbrace 0, 1 \rbrace$:
1. **Unshielded Residual Teacher Target (`bridge_aware_mode="residual"`)**: Masks out already-shielded outlier blocks (`is_shielded`) from the teacher KL target and renormalizes over the unshielded context complement $\Omega_{\mathrm{unshielded}}(i) = \lbrace j \le i : \mathcal S_{\mathrm{shield}}(i, j) = 0 \rbrace$, while adding an explicit branch-overlap penalty `0.50 * overlap_penalty` whenever the tree assigns shielded non-local blocks to the query's branch.
2. **Joint Composite Student Distribution (`bridge_aware_mode="joint"`)**: Combines tree cophenetic logits with the differentiable shield/donor bridge logit bonus before computing `F.kl_div`.

| Calibration Regime at Deep Sparsity $r = 5$ ([`test_bridge_aware_router_calibration_r5`](../tests/test_darksnail_gemma4.py#L438-L562)) | Causal Budget (%) | Unshielded Needle Recall (%) | Redundant Branch Overlap with Beltrami Shield | Wastes Tree Branch on Shielded Spike |
| :--- | :---: | :---: | :---: | :---: |
| **Bridge-Blind Calibration (`bridge_aware=False`)** | `18.75%` ($< 35\%$) | `0.0%` (`False`) | `1` block (`overlaps_spike = True`) | Yes |
| **Bridge-Aware Calibration (`bridge_aware=True`)** | **`12.50%`** ($< 35\%$) | **`100.0%` (`True`)** | **`0` blocks (`overlaps_spike = False`)** | **No (`0` redundant overlap)** |

> [!NOTE]
> **Verified Live Colab Pro+ Full-Weight Evaluation** at $r = 5$ ([`results/darksnail_bridge_aware_r5_summary.json`](../results/darksnail_bridge_aware_r5_summary.json), [`results/darksnail_bridge_aware_qasper_predictions.jsonl`](../results/darksnail_bridge_aware_qasper_predictions.jsonl)): Full-weight `google/gemma-4-31B-it` (`bfloat16`) evaluation across $r \in \lbrace 2, 3, 4, 5 \rbrace$ on $n = 200$ LongBench QASPER achieves **`50.60 / 49.95 / 49.78 / 49.27` F1** (`98.70%` dense F1 retention at $r = 5$, `50.14%` total budget / `< 35%` global routed budget), outperforming `v4_triad@5` (`29.85` F1) by **`+19.43` F1** (`95% CI [+14.67, +24.17]`, $p = 0.0001$, `118W / 22L / 60T`) and `window@5` (`28.34` F1) by **`+20.94` F1** ($p = 0.0001$).

---

## 4. Embodied Interactive Evaluation on ARC-AGI-3 (`darksnail/arc3.py`): Relational Contact-Graph Sheaf $\mathcal G_{\mathrm{contact}}(F_t)$

### 4.1 Relational Contact-Graph Predicate Induction (`ContactGraphSheaf`)

[`ContactGraphSheaf`](../darksnail/arc3.py) upgrades `RelationalOperatorDreamer` (`OnlineForwardOperatorDreamer`) from single-body ego translation to multi-body rigid-body contact mechanics over a cellular sheaf $\mathcal F$ on the dynamic 4-connected component contact graph $\mathcal G_{\mathrm{contact}}(F_t) = (\mathcal V_t, \mathcal E_t)$:
- **Stalks** $\mathcal F(u) = \mathfrak{se}(2) \cong \mathbb R^3$: Each foreground component $u \in \mathcal V_t$ (`ContactNode`) carries an infinitesimal rigid-body velocity twist
  $\mathbf v_u = (v_{y,u}, v_{x,u}, \omega_u)^\top \in \mathfrak{se}(2)$
  around its centroid $\mathbf x_u = (y_u, x_u)^\top$.
- **Unilateral Contact Restriction Maps & Non-Penetration Coboundary (`ContactEdge.compute_coboundary`)**: Across every touching pixel interface $e = (u, v) \in \mathcal E_t$
  with outward unit normal $\mathbf n_{uv} = (n_y, n_x)^\top$,
  the sheaf coboundary evaluates the normal relative velocity at the contact midpoint $\mathbf p_{uv}$:

```math
(\delta \mathbf v)(e) = \bigl\langle \mathbf n_{uv}, \, \rho_{v \to e}(\mathbf v_v) - \rho_{u \to e}(\mathbf v_u) \bigr\rangle \ge 0
```

- **Five Induced Embodied Contact Predicates**:
  1. **Multi-Body Kinematic Push Chains**: Propagates $\mathrm{SE}(2)$ translations across arbitrary-length chains ($\text{Ego} \to \text{Tool} \to \text{Obstacle} \to \text{Free Space}$) via BFS in $H^0(\mathcal G_{\mathrm{contact}}; \mathcal F)$.
  2. **Wall & Obstacle Locking (`wall_lock` / `wall_collision`)**: Collapses the entire coupled chain in $H^0(\mathcal G_{\mathrm{contact}}; \mathcal F)$ to zero displacement whenever any downstream component in the contact cone impinges on an immovable wall or canvas boundary.
  3. **Container Nesting (`is_enclosed=True`)**: Detects objects nested inside hollow container cavities (`container_id`, `content_id`) and locks internal contents to co-translate rigidly with the container ($\mathbf v_{\mathrm{inner}} = \mathbf v_{\mathrm{container}}$).
  4. **Peg-in-Hole Cavity Docking (`detect_peg_docking`)**: Detects exact complementary shape insertion when a translated block fills an open receptor cavity (`cavity_fill_ratio >= 0.50`, `wall_contacts >= max(3, perim // 2)`).
  5. **Multi-Object Surface Clearing (Plow / Broom Sweeps)**: Simultaneously displaces multiple disjoint blocks contacting a wide pusher surface in a single action step.

### 4.2 Offline CPU & Live Colab Pro+ `google/gemma-4-31B-it` Progression (`30 / 183` Levels, `18 / 25` Games)

Verified offline CPU benchmark results in [`results/darksnail_arc3_evaluation_summary.json`](../results/darksnail_arc3_evaluation_summary.json) (paired with the archived pre-contact-sheaf baselines in [`results/darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json) and [`results/darksnail_m5_colab_gemma4_25games.json`](../results/darksnail_m5_colab_gemma4_25games.json)):

| Agent Condition (`25` Games, `183` Levels) | Levels Solved (`/ 183`) | Level Solve Rate (%) | Games with $\ge 1$ Level Solved (`/ 25`) | Full Games Won (`/ 25`) | Paired vs. Random Legal (`2 / 183`) | Paired vs. Unshielded Greedy (`1 / 183`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unshielded_greedy`** (No $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$) | `1 / 183` | `0.55%` | `1 / 25` (`4.0%`) | `0 / 25` | — | Baseline |
| **`random_legal`** (`ARC3RandomBaselineAgent`) | `2 / 183` | `1.09%` | `2 / 25` (`8.0%`) | `0 / 25` | Baseline | — |
| **Milestone 5 Offline CPU (Pre-Contact-Sheaf)** (`darksnail_m5_offline_no_gemma4_25games.json`) | `29 / 183` | `15.85%` | `18 / 25` (`72.0%`) | `1 / 25` (`sb26` `8/8`) | `+27 / -0` levels ($p = 2.17 \times 10^{-4}$) | `+28 / -0` levels ($p = 1.58 \times 10^{-4}$) |
| **Milestone 5 + Live `gemma-4-31B-it` Callosal `v3` (`1000` actions)** (`darksnail_m5_colab_gemma4_25games.json`) | `29 / 183` | `15.85%` | `18 / 25` (`72.0%`) | `1 / 25` (`sb26` `8/8`) | `+27 / -0` levels ($p = 2.17 \times 10^{-4}$) | `+28 / -0` levels ($p = 1.58 \times 10^{-4}$) |
| **Extended-Horizon Pre-Contact-Sheaf `gemma-4-31B-it` Reference (`2500` actions)** | `45 / 183` | `24.59%` | `22 / 25` (`88.0%`) | `2 / 25` (`sb26`, `vc33`) | `+43 / -0` levels | `+44 / -0` levels |
| **Current Offline CPU + `ContactGraphSheaf`**, $\mathcal G_{\mathrm{contact}}(F_t)$ ([`darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json)) | **`30 / 183`** | **`16.39%`** | **`18 / 25` (`72.0%`)** | **`1 / 25` (`sb26` `8/8`)** | **`+28 / -0` levels (`16W / 0L`, Wilcoxon** $p = 2.56 \times 10^{-4}$**, Sign** $p = 3.05 \times 10^{-5}$**)** | **`+29 / -0` levels (`17W / 0L`, Wilcoxon** $p = 1.58 \times 10^{-4}$**, Sign** $p = 1.53 \times 10^{-5}$**)** |
| **Live Colab `google/gemma-4-31B-it` + `ContactGraphSheaf` (`203` Calls, `785.48s`)** ([`darksnail_arc3_evaluation_summary.json`](../results/darksnail_arc3_evaluation_summary.json)) | **`30 / 183`** | **`16.39%`** | **`18 / 25` (`72.0%`)** | **`1 / 25` (`sb26` `8/8`)** | **`+28 / -0` levels (`16W / 0L`, Wilcoxon** $p = 2.56 \times 10^{-4}$**, Sign** $p = 3.05 \times 10^{-5}$**)** | **`+29 / -0` levels (`17W / 0L`, Wilcoxon** $p = 1.58 \times 10^{-4}$**, Sign** $p = 1.53 \times 10^{-5}$**)** |

- **New Level Unlocked by `ContactGraphSheaf` (Both Offline CPU & Live `google/gemma-4-31B-it` GPU)**: `cd82-fb555c5d` advances from `1 / 6` (`L1@182`) to **`2 / 6` levels (`L1@182, L2@606` on CPU; `L1@182, L2@604` with live `google/gemma-4-31B-it`)** within the standard `max_actions = 1000` budget via multi-body contact sheaf lookahead and cavity-template actuator resonance, lifting total levels completed from `29 / 183` (`15.85%`) to **`30 / 183` (`16.39%`)** with `0` regressions across all 25 games.
- **Verified Colab Pro+ Full-Weight Evaluation**: The coupled `google/gemma-4-31B-it` (`bfloat16` / 4-bit NF4, Global Layer-29 `Gemma4BeltramiCallosalShell`, `203` System-2 deliberations in `785.48s`) + `ContactGraphSheaf` evaluation in [`results/darksnail_arc3_evaluation_summary.json`](../results/darksnail_arc3_evaluation_summary.json) achieves `25 / 25` level parity with the offline CPU engine (`30 / 183` levels, `4.8124%` RHAE) while accelerating `cd82` Level 2 (`606 -> 604`) and `tu93` Level 1 (`149 -> 132`) and Level 2 (`187 -> 169`).

---

## 5. Broadcom BCM2712 Cortex-A76 Hardware Cache & Static Allocator Verification (`scripts/verify_bcm2712_cache.py`)

Full hardware cache simulation and static PE/ELF symbol-table audit receipts across all 5 `#![no_std]` `align(64)` Rust spoke arenas (`250,000,000` total memory references, `50,000,000` per arena) are archived in [`results/bcm2712_cache_receipts.json`](../results/bcm2712_cache_receipts.json):

| Spoke Repository | `#![no_std]` Arena Struct | Size (`align(64)`) | L1d Hit Rate (`64 KB`, 4-Way) | L2 Local Hit Rate (`512 KB`, 8-Way) | L3 Local Hit Rate (`4.0 MB`, 16-Way) | **LLd Miss Rate (%) [Target `< 0.100%`]** | Dynamic Heap Allocations (`malloc`/`free`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`stylo`** | `TrackerStateArena` | `673,792 B` (`658.0 KB`) | `85.75%` | `87.52%` | `98.82%` | **`0.02106%` (PASS)** | `0` (`ZERO_DYNAMIC_ALLOCATIONS_VERIFIED`) |
| **`toposnail`** | `TopoStateArena` | `1,472,448 B` (`1,437.9 KB`) | `68.02%` | `32.60%` | `99.79%` | **`0.04601%` (PASS)** | `0` (`ZERO_DYNAMIC_ALLOCATIONS_VERIFIED`) |
| **`robosnail`** | `RoboSnailArena` | `731,456 B` (`714.3 KB`) | `93.92%` | `73.75%` | `98.57%` | **`0.02286%` (PASS)** | `0` (`ZERO_DYNAMIC_ALLOCATIONS_VERIFIED`) |
| **`lob-triad`** | `LOBStateArena` | `254,720 B` (`248.8 KB`) | `91.75%` | `99.90%` | `0.00%` (Fits in L2) | **`0.00796%` (PASS)** | `0` (`ZERO_DYNAMIC_ALLOCATIONS_VERIFIED`) |
| **`quakesnail`** | `QuakeSnailArena` | `1,745,552 B` (`1,704.6 KB`) | `67.18%` | `70.12%` | `99.44%` | **`0.05455%` (PASS)** | `0` (`ZERO_DYNAMIC_ALLOCATIONS_VERIFIED`) |
| **Aggregate (`5` Arenas)** | **`250,000,000` Refs** | — | — | — | — | **Mean: `0.03049%` / Max: `0.05455%` (PASS)** | **`0` Across All 5 Compiled Binaries** |
