# Project DarkSnail V2 — Bi-Hemispheric $\times$ Ultrametric Triad Synthesis

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Zero_Sorry_Verified-brightgreen)](#lean-4-formal-verification-occlusiontriadlean)
[![Pytest](https://img.shields.io/badge/pytest-8%2F8_passed-brightgreen)](#unit-test-suite)
[![ARC-AGI-1](https://img.shields.io/badge/ARC--AGI--1_Eval-27.00_%2F_400_(6.75%25)-blue)](#empirical-evaluation-on-arc-agi-1-n--400)
[![LongBench QASPER](https://img.shields.io/badge/LongBench_QASPER_%40_r%3D3-50.26_F1_(%2B16.27_vs_V4_Triad)-orange)](#reverse-synthesis-longbench-qasper--banal-niah-darksnail-to-llama-surgery)

**Project DarkSnail** unifies the surgical ultrametric attention topology of [`sparse-ai` (`Llama Surgery`)](../sparse-ai/papers/llama_surgery.md) and the heterogeneous transcallosal System 1 / System 2 architecture of [`brain-ai` (`BiHemisphericBrain`)](../brain-ai/docs/preprint/bihemispheric_ai_preprint.tex) onto a single frozen `Meta-Llama-3.1-8B-Instruct` ($L=32, d=4096$, `bfloat16`) backbone.

---

## Architectural Synthesis Overview

| Bottleneck in `brain-ai` / `sparse-ai` | Mathematical / Systems Primitive | DarkSnail V2 Synthesis Module |
| :--- | :--- | :--- |
| **1. Uniform Sequence Dilution (`lh_cache.py`)**: `adaptive_avg_pool1d(raw_latents, 32)` dilutes high-salience rule tokens in background grid padding. | **Beltrami Triad-Flux Outlier Shielding & Adelic Medoids** (`adelic_cache.py`, `RoPECoherence.lean`). | [`darksnail.shell.BeltramiCallosalShell`](darksnail/shell.py): Two-tier `(B, 32, 4096)` extraction combining top-16 Beltrami outlier-shielded states + 16 ultrametric medoid super-tokens with `0.5*(mean+peak)` pooling. |
| **2. Symmetric Readout Tie on 2-Color Occlusions (`callosum.py`)**: Only $2,498 / 8,192$ coordinates of $\delta z \in \mathbb{R}^{16 \times 512}$ are used; symmetric spatial bases tie on 1–2 pixel foreground occlusions. | **Skew-Symmetric Bilinear Triad Operator** (`TriadConservation.lean`, `topology.py`). | [`darksnail.readout.DarkSnailCanvasReadout`](darksnail/readout.py): Harvests $3,160$ previously zero-gradient coordinates of $\delta z$ for $S_9$-equivariant skew-symmetric triad occlusion breaking, directional asymmetry, $D_4$ reflection consensus repair, and split-panel XOR/AND completion. |
| **3. Static Amygdala Cosine Conflict (`amygdala.py`)**: 512-D cosine conflict concentrates in $[0.441, 0.449]$, acting only as a static domain gate. | **Live Zeno Triad Flux & LCA Depth Variance** (`speculative.py`, `ZenoCascade.lean`). | [`darksnail.antenna.ZenoAmygdalaController`](darksnail/antenna.py): Combines multi-prime Zeno triad flux $\Phi_{\mathrm{LH}}$, LCA depth variance $\sigma_{\mathrm{LCA}}^2$, and Step-0 demo gradients to schedule TTA steps $N_{\mathrm{TTA}} \in [0, 40]$, step size $\eta$, and ray quantiles $q_{\mathrm{ray}}$. |
| **4. Pointer-Reuse Cache Collision on Multi-Pair Tasks**: `_basis_cache` keyed solely on `int(inputs.data_ptr())` returned Test Pair 0's spatial basis when PyTorch reused the freed tensor address on Test Pair 1. | Content-addressed cryptographic digest (`hashlib.blake2b`) over grid tensor bytes. | Fixed in both [`darksnail/readout.py`](darksnail/readout.py) and [`brain-ai/brain_ai/models/callosum.py`](../brain-ai/brain_ai/models/callosum.py), upgrading all 3 half-solved multi-pair ARC tasks (`5d2a5c43`, `6ea4a07e`, `e345f17b`) from $0.50 \to 1.00$. |
| **5. Deep-Sparsity ($r \ge 3$) QASPER & Banal NIAH Drop in `Llama Surgery`**: Multi-token decode steps (`seq_len == 1`) and isolated tail blocks lose line-of-sight to peripheral factual citations at $r = 3, 4, 5$. | **Beltrami + Query-Tail Span Shield & Measure-Preserving Block Bridge** (`surgery.py`, `longbench.py`). | [`darksnail.longbench`](darksnail/longbench.py): Shields top-96 Beltrami + Query-Tail spans (`kernel_size=13`) across decode and bridges the question tail block (`NB-1`) at **$\le$ `v4_triad@r` causal key budget**, boosting LongBench QASPER F1 by **$+16.27$ at $r=3$** and **$+15.64$ at $r=4$**. |

---

## Lean 4 Formal Verification (`OcclusionTriad.lean`)

All mathematical invariants governing Project DarkSnail are machine-checked in Lean 4 with **zero `sorry` and zero compiler warnings** in [`formalization/OcclusionTriad.lean`](formalization/OcclusionTriad.lean) (and integrated into [`sparse-ai/formalization/Formalization/Analysis/OcclusionTriad.lean`](../sparse-ai/formalization/Formalization/Analysis/OcclusionTriad.lean)):

| Theorem Name | Formal Statement Summary | Lines |
| :--- | :--- | :--- |
| `skew_triad_swap_neg` & `skew_triad_self_zero` | Antisymmetry $S(W, i, c_2, c_1) = -S(W, i, c_1, c_2)$ and vanishing self-interaction $S(W, i, c, c) = 0$. | L56–L69 |
| `occlusion_triad_breaks_symmetric_tie` | On a 2-color collision $v(c_1) = v(c_2) = \alpha > 0$ where symmetric readout ties ($\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)$), the skew-symmetric triad shift satisfies $\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2\beta\alpha(W(i_0, c_2, c_1) - W(i_0, c_1, c_2)) > 0$. | L124–L175 |
| `darksnail_readout_energy_conservation` | Pointwise orthogonality $\sum_k v(k) \cdot B_W(u, v)_k = 0$ on the canvas logit manifold and $\langle z_L, \dot{z}_L\rangle = 0$ on the Slow-Fast recurrent cascade. | L184–L208 |
| `darksnail_step0_system1_homotopy` | At $\delta z = \mathbf{0}$ ($\Delta_{\mathrm{sym}} = 0, W = 0$), $\ell_{\mathrm{DS}}(\ell_{\mathrm{S1}}, 0, 0, u, v, k) = \ell_{\mathrm{S1}}(k)$ identically (100% System 1 equivalence at Step 0). | L217–L235 |
| `rope2d_medoid_relative_invariance` & `rope2d_arithmetic_norm_lt` | 2D axial RoPE medoid keys preserve exact relative displacement $(\theta_{y^*} - \theta_{y_q}, \theta_{x^*} - \theta_{x_q})$, whereas 2D arithmetic patch averaging strictly attenuates squared key norms. | L267–L313 |

---

## Empirical Evaluation on ARC-AGI-1 ($N = 400$)

Full benchmark results from [`scripts/evaluate_darksnail_arc.py`](scripts/evaluate_darksnail_arc.py) (stored in [`results/darksnail_arc_evaluation_summary.json`](results/darksnail_arc_evaluation_summary.json)):

| System Variant | Fractional Score ($/ 400$) | Accuracy (%) | Wilson 95% CI (%) | Strict All-Pair Tasks Solved | Paired Exact McNemar $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System 1 Zero-Shot** (`submission_s1_zero_shot.json`) | $0.50$ | $0.125\%$ | $[0.01\%, 1.19\%]$ | $0$ | Baseline |
| **`brain-ai` Phase 3 System 2** (`submission_s2_adapted.json`) | $16.50$ | $4.125\%$ | $[2.57\%, 6.55\%]$ | $15$ | $p = 3.05 \times 10^{-5}$ (vs. S1) |
| **Project DarkSnail V2 Synthesis** (`submission_darksnail_v2.json`) | **$27.00$** | **$6.750\%$** | **$[4.68\%, 9.64\%]$** | **$27$** | **$p = 1.49 \times 10^{-8}$ (vs. S1)**<br>**$p = 4.88 \times 10^{-4}$ (vs. `brain-ai` S2, $+12 / -0$)** |

### Newly Solved Near-Miss & Multi-Pair Tasks ($+10.50$ Fractional / $+12$ Strict Tasks)

| Task ID | Category | `brain-ai` S2 Score (Pair Pixel Acc) | DarkSnail V2 Score (Pair Pixel Acc) | Mechanism Responsible |
| :--- | :--- | :---: | :---: | :--- |
| `af22c60d` | Near-Miss ($30\times 30$) | $0.00$ ($99.8\%$, $898/900$) | **$1.00$ ($100.0\%$, $900/900$)** | Auto-detected axis-aligned dihedral reflection consensus (`d4_repair_diff` at axis $(31, 31)$). |
| `0b17323b` | Near-Miss ($15\times 15$) | $0.00$ ($99.1\%$, $223/225$) | **$1.00$ ($100.0\%$, $225/225$)** | Constant-step diagonal ray-march continuation (`diag_step_col`) beyond input color transitions. |
| `ce039d91` | Near-Miss ($10\times 10$) | $0.00$ ($99.0\%$, $99/100$) | **$1.00$ ($100.0\%$, $100/100$)** | Horizontal mirror indicator (`u_occ[:, :, 3] = s_fv > 0`) + skew-symmetric triad tie-breaking. |
| `68b67ca3` | Near-Miss ($3\times 3$) | $0.00$ ($88.9\%$, $8/9$) | **$1.00$ ($100.0\%$, $9/9$)** | Exact $S_9$ foreground-palette equivariance on `color_preserving` tasks (`Phi` foreground mask + $S_9$ triad projection). |
| `e57337a4` | Near-Miss ($3\times 3$) | $0.00$ ($77.8\%$, $7/9$) | **$1.00$ ($100.0\%$, $9/9$)** | Stride-5 block-minima pooling (`block_min_oh`) + $S_9$ foreground-palette equivariance. |
| `195ba7dc` | Near-Miss ($5\times 6$) | $0.00$ ($93.3\%$, $28/30$) | **$1.00$ ($100.0\%$, $30/30$)** | Divider-aware panel split (`wb = 2*tw + 1`) + binary panel OR/XOR/AND basis (`u_occ[:, :, 13:16]`). |
| `31d5ba1a` | Near-Miss ($3\times 5$, 2 pairs) | $0.00$ ($93.3\%, 40.0\%$) | **$1.00$ ($100.0\%, 100.0\%$)** | 2-panel XOR feature scaling (`has_2panel_only`) + content-addressed basis cache key. |
| `5d2a5c43` | Half-Solved (2 pairs) | $0.50$ ($100.0\%, 58.3\%$) | **$1.00$ ($100.0\%, 100.0\%$)** | Content-addressed `_basis_cache` digest + divider-aware split-panel boolean features. |
| `6ea4a07e` | Half-Solved (2 pairs) | $0.50$ ($100.0\%, 33.3\%$) | **$1.00$ ($100.0\%, 100.0\%$)** | Content-addressed `_basis_cache` digest + global foreground-to-background complement (`phi_bg_fg`). |
| `e345f17b` | Half-Solved (2 pairs) | $0.50$ ($100.0\%, 43.8\%$) | **$1.00$ ($100.0\%, 100.0\%$)** | Content-addressed `_basis_cache` digest + binary panel NOR/XOR features. |
| `1c0d0a4b` | Unsolved ($N=400$ sweep) | $0.00$ | **$1.00$ ($100.0\%$)** | Unlocked by `DarkSnailCanvasReadout` split-panel / complement basis + Zeno-Amygdala Pareto gate. |
| `34b99a2b` | Unsolved ($N=400$ sweep) | $0.00$ | **$1.00$ ($100.0\%$)** | Unlocked by `DarkSnailCanvasReadout` split-panel XOR basis + Zeno-Amygdala Pareto gate. |

### Visual Gallery: `brain-ai` S2 Baseline vs. `DarkSnail V2` (100.0% Exact Solves)

![DarkSnail V2 Showcase Gallery](results/darksnail_v2_showcase_gallery.png)

---

## Reverse Synthesis: LongBench QASPER & Banal NIAH (`DarkSnail` $\to$ `Llama Surgery`)

Evaluating `Meta-Llama-3.1-8B-Instruct` (`bfloat16`) patched with `SurgicalLlamaAttention` (`V4 TriadTopologyRouter` sidecar) + Layer-16 `BeltramiCallosalShell` telemetry hook + DarkSnail Query-Tail Outlier Shield & Measure-Preserving Block Bridge ([`results/darksnail_longbench_summary.json`](results/darksnail_longbench_summary.json)):

### LongBench QASPER Scientific Paper QA ($\max\_\mathrm{ctx} = 6,000$, Mean Layer-16 Zeno Flux $\Phi_{\mathrm{LH}} = 0.6055$)

| Tree Depth $r$ | Sliding Window (`window@r`) F1 | Standalone `Llama Surgery` (`v4_triad@r`) F1 [Measured Budget] | **Project DarkSnail V2 (`darksnail@r`) F1 [Measured Budget]** | $\Delta\text{F1}$ vs. `v4_triad@r` | $\Delta\text{F1}$ vs. `window@r` | $\Delta\text{F1}$ vs. `dense` ($45.17$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$r = 0$ (`dense`)** | — | $45.17$ $[100.00\%]$ | $45.17$ $[100.00\%]$ | — | — | $0.00$ |
| **$r = 2$** | $20.80$ $[62.61\%]$ | $44.15$ $[77.65\%]$ | **$52.02$ $[77.65\%]$** | **$+7.87$** | **$+31.22$** | **$+6.85$** |
| **$r = 3$** | $13.88$ $[55.47\%]$ | $33.99$ $[66.49\%]$ | **$50.26$ $[66.44\%]$** | **$+16.27$** | **$+36.38$** | **$+5.09$** |
| **$r = 4$** | $15.18$ $[49.76\%]$ | $31.40$ $[58.19\%]$ | **$47.04$ $[58.18\%]$** | **$+15.64$** | **$+31.86$** | **$+1.87$** |

> **Key Finding (Subspace Denoising + Callosal Bridging):** At every tree depth $r \in \{2, 3, 4\}$, `darksnail@r` uses the **exact same or slightly lower measured causal key budget** as standalone `v4_triad@r` (`77.65%` vs `77.65%`, `66.44%` vs `66.49%`, `58.18%` vs `58.19%`), yet boosts QASPER QA Token F1 by **$+7.87$ to $+16.27$ points** over `v4_triad@r`, **$+31.22$ to $+36.38$ points** over `window@r`, and even **outperforms unmodified `dense` attention ($45.17$ F1)** across all three sparsity levels by pruning distracting background sections while shielding the exact citation clauses needed by the question tail.

![DarkSnail V2 LongBench QASPER & Banal NIAH Curves](results/darksnail_longbench_curves.png)

---

## Repository Structure & Quickstart

```text
darksnail/
├── checkpoints/router_sidecars/
│   └── llama31_8b_router_v3.safetensors  # Pre-warmed V4 Triad router sidecar (11.56 MB)
├── darksnail/
│   ├── __init__.py          # Package exports
│   ├── shell.py             # BeltramiCallosalShell & compute_grid_prompt_zeno_flux
│   ├── readout.py           # SkewSymmetricTriadOperator, DarkSnailCanvasReadout, SlowFastTriadCascade, RoPE2DMedoidPooler
│   ├── antenna.py           # ZenoAmygdalaController & DarkSnailSystem2Adaptor
│   └── longbench.py         # LongBench QASPER & Banal NIAH synthesis evaluators
├── formalization/
│   └── OcclusionTriad.lean  # Machine-checked Lean 4 proofs (0 sorry, 0 warnings)
├── llama_surgery/           # Patched SurgicalLlamaAttention with DarkSnail shield & block bridge
├── ultrametric/             # Block-list sparse attention kernels
├── notebooks/
│   └── darksnail_v2_colab.ipynb # Self-contained Google Colab GPU research notebook (Cells 1-8)
├── results/
│   ├── darksnail_arc_evaluation_summary.json
│   ├── darksnail_longbench_summary.json
│   ├── darksnail_longbench_curves.png
│   ├── darksnail_v2_showcase_gallery.png
│   └── submission_darksnail_v2.json
├── scripts/
│   └── evaluate_darksnail_arc.py
├── tests/
│   └── test_darksnail.py    # 8-suite pytest verification
└── darksnail_synthesis_plan.md
```

### Unit Test Suite
```powershell
$env:PYTHONPATH="c:\Users\x\Documents\antigravity\darksnail;c:\Users\x\Documents\antigravity\brain-ai"
c:\Users\x\Documents\antigravity\sparse-ai\.venv\Scripts\python.exe -m pytest tests/test_darksnail.py -v
```

### Google Colab GPU Execution
Open [`notebooks/darksnail_v2_colab.ipynb`](notebooks/darksnail_v2_colab.ipynb) in Google Colab (T4 / L4 / A100 runtime) to run GPU-accelerated full-benchmark sweeps across both **ARC-AGI-1** ($N=400$) and **LongBench QASPER / Banal NIAH** on live `Meta-Llama-3.1-8B-Instruct`.
