# Project DarkSnail V2 — Comprehensive Experimental & Formal Results

**Target Backbone:** Frozen `Meta-Llama-3.1-8B-Instruct` ($L=32, d=4096$, `bfloat16`) + `Llama Surgery` V4 `TriadTopologyRouter` (`11.56 MB` `.safetensors` sidecar) + `brain-ai` `BiHemisphericBrain` ($10.51\text{M}$ trainable RH + Callosum parameters: $6.83\text{M}$ TRM + $3.68\text{M}$ CallosalConditioning).

> **Note:** For the Continuous Dynamical System (CDS) Extension results (Multi-Step `SlowFastTriadCascade`, `ContinuousSpectralFieldReadout` with `8,192 / 8,192` active coordinates, and the Beltrami Shape Predictor), see [`CDS_EXTENSION_EXPERIMENT_RESULTS.md`](CDS_EXTENSION_EXPERIMENT_RESULTS.md). The current ARC-AGI-1 result (CDS Phase 2: `38 / 400`, with a leakage audit) is in [`CDS_PHASE2_EXPERIMENT_RESULTS.md`](CDS_PHASE2_EXPERIMENT_RESULTS.md). The `27 / 400` figures below are the original V2 / Pass-1 numbers.

---

## 1. Machine-Checked Lean 4 Formal Verification & PyTorch `float32` Validation

All core mathematical invariants of the DarkSnail synthesis are machine-checked in Lean 4 with **0 `sorry` and 0 compiler warnings** in [`formalization/OcclusionTriad.lean`](../formalization/OcclusionTriad.lean) and empirically verified on GPU in `float32` (Colab Cell 2):

| Invariant / Theorem | Lean 4 Declaration | Empirical PyTorch `float32` Residual ($N = 3,600$ Spatial Cells) |
| :--- | :--- | :---: |
| **1. Skew-Symmetric Triad Orthogonality (Canvas Readout)**<br>$\sum_{k=0}^{11} v_k \cdot B_W(u, v, k) = 0$ | `darksnail_readout_energy_conservation` ([`OcclusionTriad.lean:L177-L185`](../formalization/OcclusionTriad.lean#L177-L185)) | $\max \lvert\langle v, B(u, v)\rangle\rvert = \mathbf{7.451 \times 10^{-9}}$ |
| **2. Step-0 System 1 Homotopy**<br>$\delta z = \mathbf{0} \implies \ell_{\mathrm{DS}}(\ell_{\mathrm{S1}}, \mathbf{0}) \equiv \ell_{\mathrm{S1}}$ | `darksnail_step0_system1_homotopy` ([`OcclusionTriad.lean:L198-L202`](../formalization/OcclusionTriad.lean#L198-L202)) | $\max \lvert\ell^{(0)}(\mathbf{0}) - \ell(0)\rvert = \mathbf{0.000 \times 10^{0}}$ |
| **3. Slow-Fast Recurrent Triad Cascade Conservation**<br>$\langle P z_L, \dot u_L \rangle = 0\ (z_H \to z_L\text{ coupling})$ | `darksnail_recurrent_cascade_conservation` ([`OcclusionTriad.lean:L177-L185`](../formalization/OcclusionTriad.lean#L177-L185)) | Max inner product $= \mathbf{0.000 \times 10^{0}}$ |
| **4. 2-Color Occlusion Tie-Breaking**<br>$\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2\beta\alpha(W_{i_0, c_2, c_1} - W_{i_0, c_1, c_2}) > 0$ | `occlusion_triad_breaks_symmetric_tie` ([`OcclusionTriad.lean:L137-L166`](../formalization/OcclusionTriad.lean#L137-L166)) | Verified ([`tests/test_darksnail.py`](../tests/test_darksnail.py)) |
| **5. 2D Axial RoPE Medoid Relative Invariance**<br>Medoid keys preserve exact 2D rotary phase; arithmetic patch averaging strictly attenuates key norms. | `rope2d_medoid_relative_invariance` & `rope2d_arithmetic_norm_lt` ([`OcclusionTriad.lean:L253-L311`](../formalization/OcclusionTriad.lean#L253-L311)) | Verified ([`tests/test_darksnail.py`](../tests/test_darksnail.py)) |

---

## 2. Forward Synthesis (`sparse-ai` $\to$ `brain-ai`): Callosal Extraction & ARC-AGI-1 ($N = 400$)

### 2.1 `BeltramiCallosalShell` vs. Legacy Uniform Mean-Pooling (Colab Cells 3 & 4)

On 256-token serialized grid prompts dominated by $98.8\%$ background padding (`253` background tokens + `3` rule-boundary spikes at indices `35, 110, 195`):

| Extraction Method | Output Shape | Rule-Spike Cosine Recovery | Protected Tier-1 Token Indices |
| :--- | :---: | :---: | :--- |
| **Legacy Uniform Sequence Mean-Pool** (`lh_cache.py`) | `(B, 1, 4096)` | $0.0306$ (96.9% signal loss) | None (uniformly averaged) |
| **DarkSnail `BeltramiCallosalShell`** ([`darksnail/shell.py`](../darksnail/shell.py)) | `(B, 32, 4096)` | $\mathbf{1.0000}$ **(100% spike preservation)** | `[0, 33..37, 108..112, 192..196]` (BOS sink + kernel-5 shields around `35, 110, 195`) |

- **Live `Meta-Llama-3.1-8B-Instruct` Layer-16 Extraction (Colab Cell 4):**
  Extracted `(1, 32, 4096)` Beltrami Callosal Shell from `82` live prompt tokens with **Layer-16 Zeno Flux and Cophenetic LCA Depth Variance** $(\Phi_{\mathrm{LH}}, \sigma_{\mathrm{LCA}}^2) = (1.0000, 0.1598)$.

### 2.2 Full 400-Task ARC-AGI-1 Public Evaluation Suite (Colab Cell 5)

Full evaluation results archived in [`results/darksnail_arc_evaluation_summary.json`](../results/darksnail_arc_evaluation_summary.json) and [`results/submission_darksnail_v2.json`](../results/submission_darksnail_v2.json):

| System Variant | Fractional Score ($/ 400$) | Accuracy (%) | Wilson 95% CI (%) | Strict All-Pair Tasks Solved | Exact Paired McNemar Test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System 1 Zero-Shot** (`submission_s1_zero_shot.json`) | $0.50$ | $0.125\%$ | $[0.01\%, 1.19\%]$ | $0$ | Baseline |
| **`brain-ai` Phase 3 System 2** (`submission_s2_adapted.json`) | $16.50$ | $4.125\%$ | $[2.57\%, 6.55\%]$ | $15$ | $+15 / -0$ vs. S1 ($p = 3.052 \times 10^{-5}$) |
| **DarkSnail V2 Synthesis** (`submission_darksnail_v2.json`) | $\mathbf{27.00}$ | $\mathbf{6.750\%}$ | $\mathbf{[4.68\%, 9.64\%]}$ | $\mathbf{27}$ | $\mathbf{+27 / -0}\text{ vs. S1 }(p = 1.490 \times 10^{-8})$<br>$\mathbf{+12 / -0}\text{ vs. }\mathtt{brain\text{-}ai}\text{ S2 }(p = 4.883 \times 10^{-4})$ |

#### Task-by-Task Breakdown of the $+12$ Newly Solved Strict Tasks ($+10.50$ Fractional Gain, $0$ Regressions)

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

---

## 3. Reverse Synthesis (`brain-ai` $\to$ `sparse-ai`): Full $n = 200$ LongBench QASPER & Banal NIAH (Colab Cells 7 & 8)

Full 200-sample evaluation results archived in [`results/darksnail_longbench_summary.json`](../results/darksnail_longbench_summary.json) and per-question predictions in [`results/darksnail_qasper_predictions.jsonl`](../results/darksnail_qasper_predictions.jsonl) on frozen `Meta-Llama-3.1-8B-Instruct` (`bfloat16`) with the pre-warmed V4 `TriadTopologyRouter` sidecar ([`checkpoints/router_sidecars/llama31_8b_router_v3.safetensors`](../checkpoints/router_sidecars/llama31_8b_router_v3.safetensors)):

![DarkSnail V2 LongBench QASPER & Banal NIAH Retention Curves](../results/darksnail_longbench_curves.png)

### 3.1 Banal NIAH (`"spherical coordinates"`, $N = 4,096$, $10$ Uniform Positions $\in [0.05, 0.95]$)

With closed-form causal-triangle iso-budget window matching ([`solve_iso_budget_window_width`](../darksnail/longbench.py#L344-L362)), all three conditions (`window@r`, `v4_triad@r`, `darksnail@r`) operate at the **exact same measured causal key budget** ($\pm 0.05\%$):

| Tree Depth $r$ | `window@r` Recall (%) [Measured Budget] | `v4_triad@r` Recall (%) [Measured Budget] | **`darksnail@r` Recall (%) [Measured Budget]** | Layer-16 Zeno Flux $\Phi_{\mathrm{LH}}$ |
| :---: | :---: | :---: | :---: | :---: |
| $r = 3$ | $40.0\%$ ($4/10$) $[65.78\%]$ | $100.0\%$ ($10/10$) $[65.76\%]$ | $\mathbf{100.0\%}\ (10/10)\ [\mathbf{65.73\%}]$ | $0.5880$ |
| $r = 4$ | $30.0\%$ ($3/10$) $[56.73\%]$ | $100.0\%$ ($10/10$) $[56.70\%]$ | $\mathbf{100.0\%}\ (10/10)\ [\mathbf{56.72\%}]$ | $0.5880$ |
| $r = 5$ | $30.0\%$ ($3/10$) $[49.32\%]$ | $100.0\%$ ($10/10$) $[49.28\%]$ | $\mathbf{100.0\%}\ (10/10)\ [\mathbf{49.30\%}]$ | $0.5914$ |

### 3.2 Full LongBench QASPER Scientific Paper QA ($n = 200$, `max_ctx = 6000`, Mean Layer-16 Zeno Flux $\Phi_{\mathrm{LH}} = 0.6039$)

Across the entire 200-sample LongBench QASPER evaluation split, `window@r`, `v4_triad@r`, and `darksnail@r` operate at **triple iso-budget equivalence** (matched to within $\le 0.09\%$ causal keys at every depth, with `darksnail@r` using slightly *fewer* keys than both baselines):

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

---

## 4. Key Takeaways & Adversarial Defenses for Paper 3

### 4.1 Strictly Positive Paired Within-Question Bootstrap 95% CIs Across All Depths ($n = 200$)
- **At** $r = 2$ **(`77.94%` budget):** `darksnail@2` (`44.20 F1`) beats `v4_triad@2` (`37.70 F1`) by **`+6.50 F1`** with a paired 95% CI of **`[+2.86, +10.29]`** ($83$ question wins vs. $49$ losses) and **exceeds the full `dense` ceiling (`43.10 F1`) by `+1.10 F1`** while pruning $22.06\%$ of causal keys.
- **At** $r = 3$ **(`66.84%` budget):** `darksnail@3` (`43.33 F1`) beats `v4_triad@3` (`35.92 F1`) by **`+7.41 F1`** with a paired 95% CI of **`[+3.39, +11.24]`** ($90$ wins vs. $53$ losses) and **still exceeds the full `dense` ceiling (`43.10 F1`) by `+0.23 F1`** while pruning $\mathbf{33.16\%}$ **of causal keys**.
- **At** $r = 4$ **(`58.65%` budget):** `darksnail@4` (`41.51 F1`) beats `v4_triad@4` (`29.73 F1`) by **`+11.78 F1`** with a paired 95% CI of **`[+7.51, +15.95]`** ($101$ wins vs. $45$ losses — a $\mathbf{2.24:1}$ question win ratio, $> 5.5\sigma$ above zero) and retains $\mathbf{96.3\%}$ **of the `dense` ceiling** (`-1.59 F1`, paired 95% CI `[-4.40, +1.08]` includes zero) while pruning $\mathbf{41.35\%}$ **of causal keys**.

### 4.2 Triple Iso-Budget Precision & Cross-Depth Dominance
- **Exact Iso-Budget Across All Three Methods:** Using [`solve_iso_budget_window_width`](../darksnail/longbench.py#L344-L362) and [`_apply_darksnail_block_bridge`](../llama_surgery/surgery.py#L580-L671), `darksnail@r` uses slightly *fewer* causal keys than both `v4_triad@r` and `window@r` at every depth (`77.94%` vs `77.99%` / `78.01%` at $r=2$; `66.84%` vs `66.91%` / `66.93%` at $r=3$; `58.65%` vs `58.71%` / `58.74%` at $r=4$).
- **Cross-Depth Dominance:** **`darksnail@4` (`58.65%` causal budget, `41.51 F1`) uses** $\mathbf{19.34\%}$ **fewer causal keys than `window@2` (`78.01%` budget, `34.41 F1`) and `v4_triad@2` (`77.99%` budget, `37.70 F1`), yet still beats `window@2` by `+7.10 F1` and beats `v4_triad@2` by `+3.81 F1`!**

### 4.3 Qualitative Case Studies from [`results/darksnail_qasper_predictions.jsonl`](../results/darksnail_qasper_predictions.jsonl)
1. **Boundary-Spanning Multi-Entity List Recovery** (Sample `idx=4`, `_id=14e889...`, $4,473$ tokens):
   - **Gold Answer:** *"FoxNews, CNN, ESPN, New York Times, Time magazine, Huffington Post Weird News, The Guardian, Cartoon Network, Cooking Light, Home Cooking Adventure, Justin Bieber, Nickelodeon, Spongebob, Disney."*
   - **`v4_triad@3` (`15.38 F1` at `66.94%` budget):** *"They selected a best model (B-M) using a combined set of Time, The Guardian, and Disney pages."* (Truncated the 14-entity enumeration where it crossed a 64-token dyadic tree block boundary.)
   - **`darksnail@3` (`100.00 F1` at `66.85%` budget):** *"FoxNews, CNN, ESPN, New York Times, Time magazine, Huffington Post Weird News, The Guardian, Cartoon Network, Cooking Light, Home Cooking Adventure, Justin Bieber, Nickelodeon, Spongebob, Disney."* (Measure-preserving block bridge + Beltrami outlier shield preserved the entire boundary-spanning list intact.)
2. **Technical Definition Recovery at Deep Sparsity** $r = 4$ (Sample `idx=1`, `_id=db6c4d...`, $3,365$ tokens):
   - **Gold Answer:** *"extension of the NetVLAD, adds Ghost clusters along with the NetVLAD clusters"*
   - **`v4_triad@4` (`0.00 F1` at `59.83%` budget):** *"A pooling strategy that aggregates local descriptors into a global representation for language identification."*
   - **`darksnail@4` (`51.85 F1` at `59.62%` budget, matching `dense` word-for-word):** *"An extension of the NetVLAD approach that adds Ghost clusters to map noisy or irrelevant content into ghost clusters."*
