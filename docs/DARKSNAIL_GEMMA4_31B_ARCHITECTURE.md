# Project DarkSnail V2 -> Google Gemma 4 31B & Embodied ARC-AGI-3 Architecture Specification

**Target Backbone:** `google/gemma-4-31B` / `google/gemma-4-31B-it` ([arXiv:2607.02770](https://arxiv.org/abs/2607.02770), [HuggingFace `config.json`](https://huggingface.co/google/gemma-4-31B/blob/main/config.json))  
**Source Architecture:** `sneed-and-feed/projectdarksnail` (`CC0-1.0` Public Domain)  
**Core Deliverables:** [`darksnail/gemma4.py`](../darksnail/gemma4.py), [`darksnail/arc3.py`](../darksnail/arc3.py), [`darksnail/shell.py`](../darksnail/shell.py), [`darksnail/readout.py`](../darksnail/readout.py), [`darksnail/antenna.py`](../darksnail/antenna.py), [`darksnail/__init__.py`](../darksnail/__init__.py), [`papers/darksnail_gemma4_arc3_preprint.md`](../papers/darksnail_gemma4_arc3_preprint.md), [`notebooks/darksnail_v2_colab.ipynb`](../notebooks/darksnail_v2_colab.ipynb), [`notebooks/darksnail_arc3_colab.ipynb`](../notebooks/darksnail_arc3_colab.ipynb), [`tests/test_darksnail_gemma4.py`](../tests/test_darksnail_gemma4.py), [`tests/test_darksnail_arc3.py`](../tests/test_darksnail_arc3.py)

---

## 1. Backbone Structural Mapping: `Meta-Llama-3.1-8B-Instruct` vs. `google/gemma-4-31B` (`darksnail/gemma4.py`)

Ground-truth architectural parameters are codified in `Gemma4_31B_Spec` (`GEMMA4_31B_SPEC`), `GLOBAL_LAYER_INDICES = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)`, and `SLIDING_LAYER_INDICES` in [`darksnail/gemma4.py`](../darksnail/gemma4.py):

| Architectural Axis | `Meta-Llama-3.1-8B-Instruct` (Original) | `google/gemma-4-31B-it` (`arXiv:2607.02770`) | DarkSnail V2 Gemma 4 31B Adaptation |
| :--- | :--- | :--- | :--- |
| **Decoder Topology** | `32` homogeneous full-attention layers | `60` heterogeneous layers (`5:1` Local/Global interleave, `vocab_size = 262,144`, `max_position_embeddings = 262,144`) | **Selective Global Surgery (`only_global_layers=True`)**: Leave all `50` Local Sliding-Window (`W=1024`) layers untouched at native FlashAttention/SDPA speed; patch strictly the `10` Global Full-Attention layers at indices `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`. |
| **Residual Stream (`d_model`)** | `4,096` (`intermediate_size = 14,336`) | `5,376` (`intermediate_size = 21,504`, `rms_norm_eps = 1e-6`, `final_logit_softcapping = 30.0`) | **`Gemma4BeltramiCallosalShell` (`5376 -> 4096` Isometric Bridge)** + **`DarkSnailGemma4GlobalHook` (`5376 <-> 64`)**: Native extraction on `5376`-D plus deterministic row-orthonormal Haar-pair projection `_build_isometric_callosal_bridge(5376, 4096)` to `(B, 32, 4096)` `float32` so frozen `brain-ai` `CallosalConditioning` (`3.68M` params) requires zero re-training. |
| **Callosal Extraction Layer** | Layer `16` (`16 / 32 = 50%` depth) | **Layer `29`** (`5th` Global layer, `30 / 60 = 50%` depth, $r^\ast \approx 0.722$ cumulative receptive-field depth) | **`attach_gemma4_beltrami_hook(model, layer_idx=29)`**: Layer `16` in Gemma 4 is a `1024`-token local sliding layer (`16 % 6 == 4`); hooking Global Layer `29` restores full `256K`-context task visibility. Any legacy call with `layer_idx=16` auto-snaps to `29` via `Gemma4_31B_Spec.snap_to_nearest_global_layer`. |
| **Attention Heads & GQA** | `H_q = 32`, `H_kv = 8` (`G = 4`), `head_dim = 128` | **Local (50):** `H_q = 32`, `H_kv = 16` (`G = 2`), `head_dim_sliding = 256`<br>**Global (10):** `H_q = 32`, `H_kv = 4` (`G = 8`), `head_dim_global = 512` | **`SurgicalGemma4Attention`**: Per-layer heterogeneous geometry awareness (`head_dim = 512`, `num_kv_heads = 4`, `num_kv_groups = 8`, `scale = 1.0`, zero-copy submodule transfer `skip_proj_init=True`, and 4-bit `bitsandbytes` `in_features`/`out_features` inspection). |
| **QK/V Norms & `attention_k_eq_v`** | No QK/V norms (`scale = 1/sqrt(128)`), separate `k_proj` and `v_proj` | `q_norm = Gemma4RMSNorm(512)`, `k_norm = Gemma4RMSNorm(512)`, `v_norm = Gemma4RMSNormNoScale(512)`, `scale = 1.0`.<br>**Global layers:** `attention_k_eq_v = True` (`v_proj = None`, `v = v_norm(k_raw)` before `k_norm` and `p-RoPE`). | **Unified `K = V` Pruning & Pre-RoPE Value Path**: Single `[B, N]` keep mask computed on `k_nope` prunes both `K` and `V` (and propagates to `shared_kv_states` / `past_key_values.shared_layers` when `store_full_length_kv=True`) while preserving `v = v_norm(k_raw)` prior to `k_norm` and `p-RoPE`. |
| **Rotary Encoding (`p-RoPE`)** | Full `128`-D RoPE (`theta = 500,000`) | **Local:** Full `256`-D RoPE (`rope_theta_sliding = 10,000.0`)<br>**Global:** **Proportional RoPE (`p-RoPE`)**, `partial_rotary_factor = 0.25`, `rope_theta_global = 1,000,000.0` (`d_rope = 128` rotated channels, `d_nope = 384` unrotated channels). | **`ProportionalRoPESubspaceSplitter(head_dim=512, p_rope_ratio=0.25)`**: Supports both HuggingFace's `rotate_half` split-half layout (`[0:64] U [256:320]` rotated vs. `[64:256] U [320:512]` unrotated) and contiguous prefix layout (`[0:128]` vs. `[128:512]`). Routes and computes Beltrami flux strictly on `k_nope` (`384`-D). |
| **Layer Norm Pattern** | 2-norm Pre-LN | **4-norm Sandwich Norm + `layer_scalar`** (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`) | Preserves all 4 Sandwich RMSNorms and `layer_scalar` buffer; `callosal_up` is zero-initialized (`zero_init_up_for_homotopy=True`) for exact Step-0 System 1 homotopy (`Theorem 2.4`). |

---

## 2. Subtle Gemma 4 31B Pitfalls Resolved & Reverse Synthesis Benchmarks

### 2.1 The HuggingFace `rotate_half` p-RoPE Channel Split (`[0:64] U [256:320]`) & 3D/Position-Indexed `apply_gemma4_prope`
In HuggingFace [`modeling_rope_utils._compute_proportional_rope_parameters`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py), `partial_rotary_factor = 0.25` on `global_head_dim = 512` computes `rope_angles = 64` non-zero frequencies and pads `nope_angles = 192` zeros into `inv_freq` (length `256`), then duplicates `emb = cat([freqs, freqs], dim=-1)` (length `512`). Because `apply_rotary_pos_emb` uses `rotate_half(x)` (which pairs `x[..., :256]` with `x[..., 256:]`):
* Under HuggingFace's native `Gemma4RotaryEmbedding` (`layout="hf_split_half"`), the `128` rotated channels are **`[0:64]` and `[256:320]`**, while the `384` unrotated channels are **`[64:256]` and `[320:512]`**.
* Under contiguous prefix slicing (`layout="contiguous_prefix"`, used in [`scripts/build_gemma4_surgical_hooks.py`](../scripts/build_gemma4_surgical_hooks.py)), the `128` rotated channels are **`[0:128]`** and the `384` unrotated channels are **`[128:512]`**.
* `ProportionalRoPESubspaceSplitter` supports both layouts losslessly (`||merge(*split(k)) - k||_inf == 0.0`), guaranteeing `0.000e+00` positional phase drift on `k_nope` across the full `256K` context window.
* `apply_gemma4_prope` handles 2D `(max_seq_len, head_dim)`, 3D `(B, S, head_dim)`, and 4D `(B, 1, S, head_dim)` `cos/sin` tables indexed by `position_ids` or `position_offset` during single-token autoregressive decoding (`seq_len == 1, past_len > 0`).

### 2.2 `attention_k_eq_v = True` Pre-Norm / Pre-RoPE Value Branch
In [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py), Global layers set `v_proj = None`, but $V$ is **not** post-RoPE $K$:
1. Raw key projection: $K_{\mathrm{raw}} = W_K h \in \mathbb{R}^{B \times S \times 4 \times 512}$.
2. Value branch: $V = \mathrm{Gemma4RMSNormNoScale}(K_{\mathrm{raw}})$ (unrotated, scale-free RMSNorm).
3. Key branch: $K = \mathrm{p\text{-}RoPE}\bigl(\mathrm{Gemma4RMSNorm}(K_{\mathrm{raw}})\bigr)$.

`SurgicalGemma4Attention` preserves this exact split while `DarkSnailGemma4GlobalHook.forward_global_kv_hook` supports both zero-copy `v_pruned = k_pruned` (when `v_global is None`) and synchronized mask multiplication `v_pruned = v_global * keep_mask` (when `v_global` is passed explicitly).

### 2.3 Overcoming `k_norm` L2-Norm Flattening via `k_nope` Beltrami Flux & Query-Tail Shielding
Because Gemma 4 applies per-head `k_norm = Gemma4RMSNorm(512)` before p-RoPE, the full-vector L2 norm $\lVert k_t \rVert_2$ is nearly constant across tokens. `DarkSnailGemma4GlobalHook.compute_nope_beltrami_salience` and `SurgicalGemma4Attention` combine:
1. **Subspace Spectral Energy** $\lVert k_{\mathrm{nope}, t} \rVert_2$ (measuring how much of the fixed `RMSNorm` energy budget is concentrated in the 75% semantic `384`-D subspace vs. the 25% `128`-D rotary carrier),
2. **Beltrami Adjacent Jump Flux & Cross-Shell Curl** $\lVert k_{\mathrm{nope}, t} - k_{\mathrm{nope}, t-1} \rVert_2 \cdot (0.25 + 0.75 \sqrt{1 - \langle \hat u_{\mathrm{coarse}, t}, \hat u_{\mathrm{fine}, t} \rangle^2})$ shielded by a 1D max-pooling neighborhood filter (`shield_radius = 2`), and
3. **Query-Tail Outlier Shield & Measure-Preserving Donor Pruning**: Shields top-96 Beltrami + Query-Tail contiguous spans (`kernel_size = 13` during prefill, `kernel_size = 9` and `decode_k = 64` during single-token decode) and evicts an equal number of lowest-salience donor tokens (`rem = add_bh + max(8, round(0.55 * add_bh))`) so the causal key budget of `darksnail@r` never exceeds `v4_triad@r`.

### 2.4 Block-KL + Dual Lagrangian Sidecar Calibration & LongBench QASPER / Banal NIAH Results
`warmup_gemma4_routers_kl` calibrates the 10-global-layer router sidecar ([`checkpoints/router_sidecars/gemma4_31b_router_v4.safetensors`](../checkpoints/router_sidecars/gemma4_31b_router_v4.safetensors), `85.38 MB`) over `num_steps = 40` (`lr = 2e-3`, `target_density = 0.25`, temperature annealing $\tau \in [1.50 \to 0.35]$) using self-supervised Block-KL divergence + Dual Lagrangian density regularization on `384`-D `(q_nope, k_nope)` descriptors (falling back to deterministic orthonormal Haar-DCT spectral hyperplanes `_init_router_haar_dct_spectral_hash` when no sidecar is present).

#### Full $n = 200$ LongBench QASPER Scientific QA on `google/gemma-4-31B-it` (`bfloat16`, `max_ctx = 6,000`)

| Tree Depth $r$ | `window@r` F1 [Budget] | `v4_triad@r` F1 [Budget] | **`darksnail@r` F1 [Budget]** | Paired $\Delta\text{F1}$ vs. `v4_triad@r` [95% CI] ($p$-value, `W/L/T`) | Paired $\Delta\text{F1}$ vs. `window@r` [95% CI] ($p$-value) | Paired $\Delta\text{F1}$ vs. `dense` (`49.92`) [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $r = 0$ (`dense`) | — | `49.92` `[100.00%]` | `49.92` `[100.00%]` | — | — | `0.00` |
| $r = 2$ | `42.55` `[84.55%]` | `44.70` `[84.54%]` | **`49.89` `[84.54%]`** | **`+5.18` `[+2.40, +8.14]`** ($p = 0.0002$, `70/37/93`) | **`+7.33` `[+3.37, +11.33]`** ($p = 0.0005$) | `-0.04` `[-1.92, +1.80]` |
| $r = 3$ | `37.75` `[72.13%]` | `42.04` `[72.11%]` | **`48.62` `[72.11%]`** | **`+6.58` `[+3.24, +9.96]`** ($p = 0.0002$, `84/41/75`) | **`+10.87` `[+6.30, +15.42]`** ($p = 0.0001$) | `-1.30` `[-2.79, +0.09]` |
| $r = 4$ | `34.77` `[62.25%]` | `35.81` `[62.23%]` | **`48.61` `[62.23%]`** | **`+12.80` `[+8.98, +16.70]`** ($p = 0.0001$, `103/34/63`) | **`+13.84` `[+8.52, +19.12]`** ($p = 0.0001$) | `-1.31` `[-3.73, +1.07]` |

#### 50-Position Banal NIAH (`"spherical coordinates"`, $N = 4,096$) on `google/gemma-4-31B-it` (`bfloat16`)

| Tree Depth $r$ | Measured Causal Budget | `window@r` Recall [Wilson 95% CI] | `v4_triad@r` Recall [Wilson 95% CI] | **`darksnail@r` Recall [Wilson 95% CI]** | Gain vs. `v4_triad@r` | Exact McNemar $p$-value (`Wins / Losses`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $r = 2$ | `82.5%` (`1,689.9` keys) | `62.0%` `[48.15%, 74.14%]` (`31/50`) | `92.0%` `[81.16%, 96.85%]` (`46/50`) | **`100.0%` `[92.87%, 100.0%]` (`50/50`)** | **`+8.0%`** | $p = 0.1250$ (`+4 / -0`) |
| $r = 3$ | `70.2%` (`1,438.0` keys) | `46.0%` `[32.98%, 59.60%]` (`23/50`) | `42.0%` `[29.38%, 55.77%]` (`21/50`) | **`94.0%` `[83.78%, 97.94%]` (`47/50`)** | **`+52.0%`** | $p < 10^{-6}$ (`+26 / -0`) |
| $r = 4$ | `59.9%` (`1,227.1` keys) | `32.0%` `[20.76%, 45.81%]` (`16/50`) | `16.0%` `[8.34%, 28.51%]` (`8/50`) | **`84.0%` `[71.49%, 91.66%]` (`42/50`)** | **`+68.0%`** | $p < 10^{-9}$ (`+34 / -0`) |
| $r = 5$ | `52.3%` (`1,071.4` keys) | `24.0%` `[14.29%, 37.41%]` (`12/50`) | `6.0%` `[2.06%, 16.22%]` (`3/50`) | **`68.0%` `[54.19%, 79.24%]` (`34/50`)** | **`+62.0%`** | $p < 10^{-8}$ (`+31 / -0`) |

---

## 3. Core Right-Hemisphere Primitives (`darksnail/shell.py`, `darksnail/readout.py`, `darksnail/antenna.py`)

| Module & Exported Symbols | Key Mathematical Signatures & Invariants |
| :--- | :--- |
| **[`darksnail/shell.py`](../darksnail/shell.py)**: `BeltramiCallosalShell`, `PullbackCanvasResult`, `compute_grid_prompt_zeno_flux`, `extract_beltrami_lh_latents`, `extract_beltrami_pullback_canvas`, `extract_grid_beltrami_shape_descriptors`, `infer_arc_output_shape_beltrami` | Condenses variable-length residual streams `(B, S, D)` into `target_tokens = 32` super-tokens (`outlier_budget = 16` Tier-1 Beltrami exterior-curl outliers + `condensed_budget = 16` Tier-2 adelic medoid tokens attenuated by $\gamma_c = \exp(-\nu \cdot \mathrm{Var}(P_{S_c}))$ with `viscous_nu = 0.05`). Achieves `1.0000` rule-spike cosine preservation and `419 / 419` (`100.0%` eval) / `393 / 416` (`94.47%` train) ARC-AGI-1 shape prediction. |
| **[`darksnail/readout.py`](../darksnail/readout.py)**: `SkewSymmetricTriadOperator`, `verify_pointwise_orthogonality`, `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, `RoPE2DMedoidPooler`, `ContinuousSpectralFieldReadout`, `get_continuous_spectral_and_ray_basis` | Harvests `8,192 / 8,192` (`100.0%`) coordinates of `delta_z` $\in \mathbb{R}^{16 \times 512}$ (`0:2498` symmetric basis, `2498:4802` $\mathfrak{so}(12)$ 3-tensor triad occlusion breaker, `4802:4810` directional Z-order basis, `4810:5322` slow-fast cascade, `5322:5658` background-foreground complement, `5658:8072` 201-channel continuous spectral/ray/Jordan enclosure field `Phi_spec`, `8072:8192` multi-step guard generators). `SlowFastTriadCascade(d_reasoner=512, d_shell=8, preserve_exact_sphere_norm=True)` unrolls $K \in \lbrace 1, 4, 8, 16 \rbrace$ steps with `0.000e+00` radial drift and $\langle v, B_W(u, v) \rangle \le 10^{-8}$. |
| **[`darksnail/antenna.py`](../darksnail/antenna.py)**: `ZenoAmygdalaController`, `DarkSnailSchedule`, `DarkSnailSystem2Adaptor` | Fuses Left-Hemisphere `zeno_flux` and `lca_variance`, Step-0 demonstration loss/fit, and Right-Hemisphere predictive entropy to schedule `num_steps` $\in [0, 40]$, `lr` $\in [0.02, 0.06]$, `anchor_weight` $\in [0.05, 0.30]$, `triad_gain` $\in [0.25, 1.50]$, `unroll_steps` $\in \lbrace 1, 4, 8, 16 \rbrace$, and `ray_quantiles`. Solves `38.00 / 400` (`9.50%`) ARC-AGI-1 evaluation tasks ($p = 2.38 \times 10^{-7}$ vs. `brain-ai` S2). |

---

## 4. Embodied Interactive Architecture on ARC-AGI-3 (`darksnail/arc3.py` — Milestone 5 + Argon Refactoring + Callosal v3)

[`darksnail/arc3.py`](../darksnail/arc3.py) extends Project DarkSnail to embodied $64 \times 64$ 16-color (`0..15`) interactive POMDP environments (`arc_agi` SDK, `25` public games, `183` levels, `max_actions = 1000`, hybrid action space `ACTION1..5`, `ACTION6(x, y)` over $4,096$ coordinates, and `ACTION7` undo) with **zero game-ID branching** and **zero inspection of environment source files**.

### 4.1 `TemporalClockContentSplitter` ($\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$): Continuity Equation & Thermodynamic Irreversibility Guard
`TemporalClockContentSplitter(max_clock_tick_area=6)` constructs an online dynamical isomorphism of `ProportionalRoPESubspaceSplitter` (`k_rope` vs. `k_nope`) grounded in the discrete continuity equation with zero spatial/border coordinate priors:

```math
\partial_t \rho_c(x, y) + \nabla \cdot \mathbf J_c(x, y) = \sigma_c(x, y)
```

* **Conservative Content Translations** ($\mathcal H_{\mathrm{nope}}, \sigma_c = 0$):
  Preserve global color mass ($\Delta M_c = 0$) or translate non-background colors across clusters within the same step.
* **Exogenous Dissipative Clocks** ($\mathcal H_{\mathrm{clock}}, \sigma_c \neq 0$):
  Satisfy strict global color-mass non-conservation ($\Delta M_{c_{\mathrm{old}}} \le -\lvert D \rvert, \; \Delta M_{c_{\mathrm{new}}} \ge \lvert D \rvert$, cluster area $\le 6$, bounding box $\le 3 \times 3$, distance $> 5$ from any `ACTION6` click) under four laws:
  1. *Law 1 (Multi-Cluster Decoupling)*: Co-ticks alongside a spatially disjoint content transition of area $\ge \max(4, 3 \lvert D \rvert)$.
  2. *Law 2 (Cross-Action Invariance)*: Distinct actions $a_1 \neq a_2$ produce identical-area dissipative depletion at Manhattan distance $\le 4$ inside the same initial connected component `dom_cid`.
  3. *Law 3 (Constant-Velocity 1-Way Phase Front Propagation)*: Consecutive dissipative ticks inside `dom_cid` advance by $1 \le d_1(D_{t-1}, D_t) \le 2$. When confirmed on a thin bar (`area <= 128`, `min(h, w) <= 2`, `fill_ratio >= 0.75`), `_mark_component_as_clock` masks the entire bar in `clock_mask`.
  4. *Law 4 (Periodic Phase-Reset / Respawn Conservation)*: Detects timer expiry / fuel refill back to `initial_frame` (`is_phase_reset = True`) and masks any small ($\le 16$ px) life-counter decrement into `clock_mask`.
* **Thermodynamic Irreversibility Guard (`reversible_mask`)**: Because exogenous clocks evolve monotonically between phase resets, any interior pixel (`2 <= r < H - 2`, `2 <= c < W - 2`) that reverses its color transition (`last_changed_from_color[r, c] == next_frame[r, c]`) on a non-reset step has positive action contingency and is permanently marked in `self.reversible_mask[r, c] = True`, immediately clearing `self.clock_mask` and `self.clock_comp_ids` for that component (e.g., protecting reversible sliding columns in `tu93`).
* **Retroactive Graph Quotienting & Cross-Level Persistence**: Whenever `clock_mask` expands, `DarkSnailArc3WorldModel._rebuild_graph_on_clock_expansion()` retroactively re-projects the entire `step_history` onto `project_nope(frame_2d)` (`hash_nope`), merging aliased states and converting clock-only ticks into `self_loops`. On level advance (`_on_level_advance`), stationary HUD clock masks (`0 < num_clock_pixels <= 128`) are preserved (`preserve_clock_mask=True`) for instant Step-0 clock immunity on subsequent levels.

### 4.2 `extract_arc3_beltrami_medoids`: Hoisted Caching, 4-Channel Extraction & Diversity Tiering
`extract_arc3_beltrami_medoids(frame_2d, prev_frame_2d=None, max_candidates=36, clock_mask=None)` discretizes the $4,096$-pixel `ACTION6(x, y)` space into at most `max_candidates` (`36` in Tier 1, `64` in Tier 2) interior medoids:
* **Hoisted Module-Level Singletons**: Reuses `_DEFAULT_ROPE2D_MEDOID_POOLER = RoPE2DMedoidPooler(patch_size=4, viscous_nu=0.05, viscous_p=2, shell_level=1)` and memoizes 2D connected-component segmentation in `_CC2D_CACHE` (keyed by `(_blake2b_grid(grid), connectivity, same_color, ignore_val)`, up to `512` entries).
* **Four Geometric & Spectral Channels**:
  1. `cc_object` / `cc_lattice`: 4-connected foreground components with exact interior medoid $(x^\ast, y^\ast) \in \mathcal C$, morphological self-information widget bonus, and minority-color bracket marker proximity.
  2. `enclosed_hole`: 4-connected background cavities not reaching the canvas exterior plus local component interior holes, with adaptive multi-cavity salience (`base_hole_sal = 1.85` when $\ge 16$ holes else `2.60`).
  3. `symmetry_defect` & `homomorphism_defect` (Channel 3 / 3b): Mirror/rotational discrepancy components (`mismatch_ratio <= 0.22`) on the foreground bounding box plus Dual-Panel Kronecker Block-Scaling (`1 <= ky, kx <= 4`) cell mismatch centers.
  4. `rope2d_patch`: 20-channel feature field (`16` one-hot colors + foreground mask + Beltrami boundary gradient + temporal innovation + radial prior) pooled via `_DEFAULT_ROPE2D_MEDOID_POOLER`.
* **2D NMS & Dyadic Quadtree Diversity Stratification**: Deduplicates candidates via `KindPriority` NMS, then stratifies across the 4 canvas quadrants using `_diversity_tier = occ_rank // 2` (prioritizing the first 2 instances of each distinct `(kind, bbox)` shape per quadrant before taking 3rd..$N$-th repetitions of identical lattice tiles) while deferring `homomorphism_defect` and excess $1 \times 1$ speckles (`speckle_counts[sk] >= 4`) to `deferred`.

### 4.3 `RelationalOperatorDreamer` (`OnlineForwardOperatorDreamer`) & Excitable Excursion Verifiers
* **Ego-Kinematic Generator Discovery**: Under `ACTION1..4`, `observe_transition` identifies the controllable ego sprite (`ego_colors`, `ego_signatures`), estimates displacement `(dy, dx, is_sliding)` in `kinematic_generators`, and populates the pairwise relational interaction kernel `interaction_kernel` (`wall_colors`, `pushable_colors`, `hazard_colors`, `hazard_signatures`). `predict_step` translates the ego sprite from its **current** board coordinates with obstacle and pushable-block physics.
* **Contact-Highlight & Topological Cavity Docking (`ACTION5` Boost)**: Detects when the ego sprite docks inside a hollow receptor cavity or triggers an adjacent stationary color highlight, recording `sh_next` in `active_contact_highlight_states` (`action5_is_contact_operator = True`). In `score_action_candidate`, `ACTION5` receives a `+25.0` epistemic lock-in boost at `active_contact_highlight_states` (solving `g50t` Level 1).
* **Tool/Palette Selection Mode (`tool_selection_active`)**: Automatically sets `tool_selection_active = True` when a click in a click-only game mutates a surrounding selection frame (`8 <= nope_d <= 44`, `eta_nl == 0.0`) while leaving the clicked center pixel unchanged (`center_unchanged`), allowing previously inert canvas targets to be re-probed in `is_click_locally_inert`.
* **Excitable Excursion Verifiers (`verifier_actuators`)**: When an action produces a zero net state change (`sh_prev == sh_next` and `nope_d == 0`) but triggers a multi-subframe validation wave (`sf_count >= 3` with `max_sf_nope_d >= 12` on `ACTION5`, or `sf_count >= 6, max_sf_nope_d >= 24` / `sf_count == 3, max_sf_nope_d >= 60` on `ACTION6`), `record_transition` registers `action_key` in `verifier_actuators` rather than `self_loops`. `score_action_candidate` awards `+8.50` when `edits_since_last_verify >= 1` and penalizes `-4.0` when `edits_since_last_verify == 0` (solving `tn36` Level 1 in `13` steps).

### 4.4 Scale-Invariant `compute_canvas_target_alignment` (`_TARGET_ALIGNMENT_CACHE`) & `CrossLevelGoalInvariantDistiller`
`compute_canvas_target_alignment` (memoized in `_TARGET_ALIGNMENT_CACHE` up to `512` entries) evaluates the 5D dimensionless invariant basis $\boldsymbol{\Phi}(S) = [\Phi_1, \Phi_2, \Phi_3, \Phi_4, \Phi_5]^\top \in [0, 1]^5$:
1. **Homomorphism** $\Phi_1 = \mathcal E_{\mathrm{hom}}$: Dual-panel Kronecker block-scaling, spaced `cc_lattice` homomorphism, and factored per-slot sub-panel template matching.
2. **Marginal Transport** $\Phi_2 = \mathcal E_{\mathrm{marginal}}$: Closed-form $\mathcal O(HW)$ 1D Marginal Wasserstein-1 distance
   $W_1^{\mathrm{marg}}$ (`compute_marginal_wasserstein1_distance` with scale-invariant `adj_band = max(4.5, min(H, W) / 9.8)`) and corner-bracketed slot defect (`max_bracket_span = max(3, min(H, W) // 10)`).
3. **Topological Symmetry** $\Phi_3 = 1 - \Phi_{\mathrm{topo}}$:
   Topological Betti $(\beta_0, \beta_1)$
   and dihedral $D_4$ / multi-panel symmetry defect (`compute_topological_symmetry_potential`).
4. **MDL Annihilation** $\Phi_4 = \mathcal H_{\mathrm{MDL}}$: SMiRL / Solomonoff minority-color mass annihilation (`minority_cap = max(8, (H * W) // 64)` and `cleared_colors`).
5. **Geodesic Navigation** $\Phi_5 = \mathcal E_{\mathrm{nav}}$: Avatar-to-beacon Manhattan navigation + same-color solid-ego to hollow-receptor-frame cavity docking.

On every level completion (`level_up = levels_completed > self.current_level`), `choose_action` harvests the **pre-reset winning frame** from `subframes[-2]` (when `len(subframes) >= 2`, falling back to `predict_action_outcome`) and invokes `CrossLevelGoalInvariantDistiller.distill_level_win` (`beta = 1.35`, `alpha_floor = 0.05`) to update $\mathbf w^{(L+1)} \in \Delta^4$ via Fisher-Rao natural-gradient Boltzmann mirror descent:

```math
w_k^{(L+1)} \propto w_k^{(L)} \exp\left(-\beta \cdot \mathrm{clip}\left(\frac{\Phi_k(S_{\mathrm{win}}^{(L)}) - \Phi_k(S_0^{(L)})}{\max(\lvert \Phi_k(S_0^{(L)}) \rvert, 0.15) + 0.03}, -3, 3\right)\right)
```

### 4.5 Four Pure Mathematical Multi-Step Synthesis Engines & Promoted Case 2.5
`DarkSnailArc3WorldModel.synthesize_mathematical_plan` combines four non-LLM mathematical planners:
* **Primitive A (`synthesize_abelian_jacobian_plan`)**: Solves the ridge-regularized difference-basis normal equations
  $n_{\mathrm{ridge}} = (\mathbf{\Delta}^\top \mathbf{\Delta} + \lambda \mathbf I_K)^{-1} \mathbf{\Delta}^\top \mathbf r^\star$ (`ridge_lambda = 0.05`) followed by greedy integer Orthogonal Matching Pursuit (OMP).
* **Primitive B (`dream_forward_plan`)**: Koopman operator mental beam search (`beam_width = 6`, `horizon = 3`) minimizing
  $\mathcal E_{\mathrm{OT}}(\hat F; \Pi) = \mathcal E_{\mathrm{target}}(\hat F; \Pi) + 0.75 \, W_1^{\mathrm{marg}}(\hat F, F^\star)$.
* **Primitive C (`unroll_triad_cascade_plan`)**: Co-evolving multi-step `SlowFastTriadCascade` trajectory readout on $\mathbb S^7(r_0)$, coupling step-specific $\mathfrak{so}(8)$ triad priors
  $\mathbf p_{\mathrm{triad}}^{(\tau)} = \tanh(B_{W_{\mathrm{act}}}(\mathbf u^{(\tau)}, \mathbf v^{(\tau)}))$ with mental rollout energy reduction.
* **Primitive D (`synthesize_sheaf_bijection_plan`)**: Scale-invariant hierarchical operad / cellular sheaf cavity substitution CSP. Discovers hollow square reference frames $\mathcal G_{\mathrm{ref}}$ of arbitrary outer side
  $S_{\mathrm{ref}} \ge 4$ and foreground area
  $A_{\mathrm{ref}}$ whose interior cavity side length is given by exact area conservation:

```math
S_{\mathrm{tok}} = \sqrt{S_{\mathrm{ref}}^2 - A_{\mathrm{ref}}}, \qquad 2 \le S_{\mathrm{tok}} < S_{\mathrm{ref}}
```

  Using scale-invariant Jordan wall ray-casting (`_enclosing_box_color` with `wall_h >= s_ref`), it separates free palette tokens ($S_{\mathrm{tok}} \times S_{\mathrm{tok}}$ outside container boxes)
  from container-enclosed solid constants, call rings, and receptor slots ($S_{\mathrm{slot}} < S_{\mathrm{tok}}$), solving the recursive sheaf tree expansion `_expand_box` to win all `8 / 8` levels of `sb26` in `124` actions (`100.0%` RHAE).
* **Online** $\mathfrak{so}(16)$ **Lie Generators, Commutator Curvature** $\kappa(a_1, a_2)$ **& Promoted Case 2.5**:
  `update_lie_generator` projects 15-sphere board transitions (`encode_sphere_state`) onto skew-symmetric generators $\Omega_a \in \mathfrak{so}(16)$
  via rank-2 tangent wedge products $\mathbf s_{t+1}\mathbf s_t^\top - \mathbf s_t\mathbf s_{t+1}^\top$, and `compute_commutator_curvature` evaluates:

```math
\kappa(a_1, a_2) = \frac{\lVert [\Omega_{a_1}, \Omega_{a_2}] \rVert_F}{\max(10^{-6}, \lVert \Omega_{a_1} \rVert_F \lVert \Omega_{a_2} \rVert_F)}
```

  weighting Abelian OMP by $\exp(-\bar{\kappa})$ when $\bar{\kappa} > 0.25$. In `choose_action`, `synthesize_mathematical_plan` is promoted to **Case 2.5** (ahead of Tier-2 medoid expansion whenever `2 <= len(learned_operators) <= 8` and `levels_distilled > 0`, executing plans with `energy_reduction >= 0.25`), while 1-step forward lookahead (`lookahead_energy_cache`) directly informs `score_action_candidate`.

### 4.6 Unshackled `Arc3HRMRecurrentCore`, `Arc3AffectiveAmygdalaRouter` & Online $\mathfrak{so}(8)$ Triad TTT
* **Deterministic Spectral Initialization (`_init_spectral_weights`)**: `HRMRecurrentBlock(d_model=512, n_heads=8, d_ffn=1024)` (with `MagicNorm`), `Arc3HRMRecurrentCore(L_cycles=2, M_max=8)`, and `Arc3AffectiveAmygdalaRouter(hidden_dim=512, mlp_dim=256)` initialize all weights inside isolated `torch.random.fork_rng()` contexts using orthogonal projections (`gain = 0.5`) and `gain = 1e-3` on residual exit branches (`out_proj`, `ffn_down`, and Amygdala MLP output layers).
* **Online Differentiable** $\mathfrak{so}(8)$ **Triad & ACT Halting TTT (`adapt_online_step`)**: Updates `hrm_core.q_head`, `action_triad.B_raw`, `cascade.triad.B_raw`, and `cascade.gamma_logit` online (`lr = 1e-4`, decayed by $1 / (1 + 0.02 \, t_{\mathrm{TTT}})$, `adapt_triad = (levels_distilled > 0)`) from observed transition outcomes while maintaining `0.000e+00` spherical norm drift and $\le 10^{-8}$ pointwise triad orthogonality.

### 4.7 `deliberate_with_gemma4_system2`: Layer-29 Gram Resonance, 5-Archetype Compilation & Callosal v3 Coupling
When `ZenoAmygdalaController` / `Arc3AffectiveAmygdalaRouter` escalates to System 2 (`should_deliberate_gemma4`), `DarkSnailArc3Agent.deliberate_with_gemma4_system2` executes **a single forward pass** `out = self.lh_model(**enc, output_hidden_states=True)` (zero autoregressive token generation) over the structured Callosal World-Model Brief:
1. **Pathway A (Layer-29 All-but-the-Top Centered Gram Resonance & 5-Archetype Invariant Compilation)**:
   Extracts `h_layer = out.hidden_states[hook_layer + 1]` $\in \mathbb R^{1 \times T \times 5376}$ at Global Layer `29`, subtracts the per-prompt token mean $\mu_{29}$,
   projects out the top principal component $\mathbf v_1$ via 4-step power iteration (`All-but-the-Top`, `arXiv:1702.01417`), and normalizes onto $\mathbb S^{5375}$:
   * Computes span-to-span Gram resonance $p_{29}^{\mathrm{gram}}(k)$ between the query tail
     $h_{\mathrm{query}}^{(29)}$ and candidate action spans $[0]\dots[K-1]$.
   * Computes Gram resonance against the 5 Subgoal Invariant Archetype spans (`[G0:SHEAF_HOMOMORPHISM]`, `[G1:OPTIMAL_TRANSPORT]`, `[G2:DIHEDRAL_SYMMETRY]`, `[G3:ENTROPY_ANNIHILATION]`, `[G4:GEODESIC_DOCKING]`) to compile $\mathbf w_{\mathrm{subgoal}} \in \Delta^4$, blending into `goal_distiller.apply_system2_compiled_weights(subgoal_weights_np, blend=0.08 * alpha_ei)` when `levels_distilled > 0`.
   * Condenses `h_layer` via `Gemma4BeltramiCallosalShell` (`5376 -> 4096 -> 512`) to modulate `z_H_super`.
2. **Pathway B & E/I Callosal Fusion**:
   Extracts single-token digit logits `"0".."9"` from `out.logits[0, -1, :]` to form $p_{\mathrm{LH}}(k)$
   and fuses both pathways via the Amygdala's Excitatory/Inhibitory ratio $\alpha_{\mathrm{EI}} = \gamma_E / (\gamma_E + \gamma_I)$:

```math
P_{\mathrm{callosal}}(a_k \mid F_t) = (1 - \alpha_{\mathrm{EI}}) \, p_{\mathrm{LH}}(k) + \alpha_{\mathrm{EI}} \, p_{29}^{\mathrm{gram}}(k)
```

3. **One-Sided ReLU Callosal Margin (`pk_pos`)**:
   To prevent two-sided zero-centering from assigning negative priors to evaluated candidates `#2..#10` (which would demote them below un-evaluated candidates `#11..#36` that carry `g4_prior = 0.0`), `deliberate_with_gemma4_system2` applies a one-sided ReLU margin above uniform $1/K$:

```math
\tilde P_{\mathrm{callosal}}^+(a_k \mid F_t) = \max\left(0, \, P_{\mathrm{callosal}}(a_k \mid F_t) - \frac{1}{K}\right), \qquad g_4(s_t, a_k) = 1.50 \, \tilde P_{\mathrm{callosal}}^+(a_k \mid F_t) \ge 0
```

4. **Beltrami Medoid Precision-Floor Tie-Breaking, Decoupled Discrete Priors & State-Scoped BFS Isolation**:
   * **Medoid Coordinate Clicks (`ACTION6(x, y)` with `candidate_info is not None`)**: Enters `score_action_candidate` strictly at tie-breaking scale `1e-5 * max(0.0, g4_prior)` so Callosal priors break exact ties without overriding $7.9 \times 10^{-5}$ Beltrami/contingency precision on `ft09`, `sc25`, and `cd82`.
   * **Discrete Actions (`ACTION1..5`)**: Enters with `0.25` decoupled weight capped at `0.12` (`eff_g4 = min(0.12, 0.25 * g4_prior)` so it never overrides the `+0.32` geodesic corridor continuation bonus), damped to `1e-4 * g4_prior` once `levels_distilled >= 2`, and suppressed via `min(0.0, 0.25 * g4_prior)` when `is_backtrack` is the spanning-tree parent-return direction.
   * **State-Scoped Precision & Cross-State BFS Isolation**: Stores per-state spatial precision maps in `state_precision_maps[cur_hash]` (`np.clip(pi_state_map, 0.5, 2.0)`) without polluting global `state_target_alignment` caches (`spatial_precision_map=None` in `get_target_alignment`), and subtracts `g4_off` from `best_u_score` before computing cross-state BFS `frontier_scores[sh]`.

---

## 5. Embodied ARC-AGI-3 Evaluation (25 Games, 183 Levels) & Callosal Ablation

Full results from [`scripts/evaluate_darksnail_arc3.py`](../scripts/evaluate_darksnail_arc3.py), archived in [`results/darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json), [`results/darksnail_m5_colab_gemma4_25games.json`](../results/darksnail_m5_colab_gemma4_25games.json), and [`results/darksnail_m5_no_gemma4_vs_gemma4_comparison.csv`](../results/darksnail_m5_no_gemma4_vs_gemma4_comparison.csv):

### 5.1 Overall 25-Game (`183`-Level) Benchmark Progression (`max_actions = 1000`, `seed = 0`)

| Agent Policy / Architectural Pass | Levels Solved (`/ 183`) | Level Solve Rate (%) | Games with $\ge 1$ Level Solved (`/ 25`) | Full Games Won (`/ 25`) | Official RHAE Scorecard (%) | Exact Sign Test $p$-value (vs. Random) | Wilcoxon Signed-Rank $p$-value ($N=25$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `unshielded_greedy` (Ablation without $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$) | `1 / 183` | `0.55%` | `1 / 25` (`4.0%`) | `0 / 25` | `0.0002%` | — | — |
| `random_legal` (Uniform Legal Action Baseline) | `2 / 183` | `1.09%` | `2 / 25` (`8.0%`) | `0 / 25` | `0.0013%` | Baseline | Baseline |
| `darksnail_v2_arc3` — Pass 1 (Initial $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ + Medoid Explorer) | `12 / 183` | `6.56%` | `10 / 25` (`40.0%`) | `0 / 25` | `0.2140%` | $p = 0.0156$ | $p = 0.0269$ |
| `darksnail_v2_arc3` — Pass 2 (+ Non-Local Causal $\mathcal G_{\mathrm{act}} \text{ vs. } \Omega_{\mathrm{reg}} + D_4$ Transfer) | `16 / 183` | `8.74%` | `14 / 25` (`56.0%`) | `0 / 25` | `0.4810%` | $p = 0.0020$ | $p = 0.0112$ |
| `darksnail_v2_arc3` — Pass 3 (+ Geodesic Corridor Momentum + Hybrid Budget) | `18 / 183` | `9.84%` | `15 / 25` (`60.0%`) | `0 / 25` | `0.5512%` | $p = 2.44 \times 10^{-4}$ | $p = 0.0021$ |
| `darksnail_v2_arc3` — Pass 4 / Milestone 4 (+ Subsystem A Plan Synthesis + Spectral Init) | `19 / 183` | `10.38%` | `16 / 25` (`64.0%`) | `0 / 25` | `0.6274%` | $p = 3.05 \times 10^{-5}$ | $p = 4.38 \times 10^{-4}$ |
| **`darksnail_v2_arc3` — Pass 5 / Milestone 5 + Argon (`No-Gemma-4` Offline Baseline)** | **`29 / 183`** | **`15.85%`** | **`18 / 25` (`72.0%`)** | **`1 / 25` (`sb26` `8/8`)** | **`4.8124%`** | $\mathbf{p = 3.05 \times 10^{-5}}$ (`16W/0L` levels)<br>$\mathbf{p = 7.63 \times 10^{-6}}$ (`18W/0L` RHAE) | $\mathbf{p = 2.17 \times 10^{-4}}$ (levels)<br>$\mathbf{p = 1.96 \times 10^{-4}}$ (RHAE) |
| **`darksnail_v2_arc3` — Live Colab `google/gemma-4-31B-it` (`bfloat16`, Callosal `v3`, `197` Calls, `1126.4s`)** | **`29 / 183`** | **`15.85%`** | **`18 / 25` (`72.0%`)** | **`1 / 25` (`sb26` `8/8`)** | **`4.8123%`** | $\mathbf{p = 3.05 \times 10^{-5}}$ (`16W/0L` levels)<br>$\mathbf{p = 7.63 \times 10^{-6}}$ (`18W/0L` RHAE) | $\mathbf{p = 2.17 \times 10^{-4}}$ (levels)<br>$\mathbf{p = 1.96 \times 10^{-4}}$ (RHAE) |

### 5.2 Three-Run Live Colab GPU (`google/gemma-4-31B-it`) Callosal Coupling Ablation (`v1 -> v2 -> v3`)

| Evaluation Condition (`25` Games, `183` Levels, `max_actions = 1000`) | Levels Solved (`/ 183`) | Games Solved (`/ 25`) | Full Games Won | Official RHAE (%) | Live Gemma 4 Calls | Key Game Trajectory Diagnostics (`g50t`, `sc25`, `ft09`, `cd82`, `vc33`, `tu93`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`No-Gemma-4` Offline Baseline** (`use_gemma4_shell = False`) | **`29 / 183`** (`15.85%`) | **`18 / 25`** (`72.0%`) | `1 / 25` (`sb26`) | **`4.8124%`** | `0` (`635.5s`) | `g50t`: `L1@885`, `sc25`: `L1@141, L2@543`, `ft09`: `L1@619`, `cd82`: `L1@182`, `vc33`: `L1@3, L2@171`, `tu93`: `L1@149, L2@187, L3@504` |
| **Live Colab `gemma-4-31B-it` — Run 1 (`v1`)**: Uncentered $P_{\mathrm{callosal}} \in [0, 1]$ in cross-state BFS + global $\Pi(y, x)$ | `26 / 183` (`14.21%`) | `16 / 25` (`64.0%`) | `1 / 25` (`sb26`) | `4.7798%` | `180` (`1216.4s`) | Uncentered `+0.375` offset inflated deliberated states in BFS: `g50t` `0/7`, `sc25` `0/6`, `cd82` `L1@802`, `vc33` `L2@329`; `tu93` accelerated to `L1@132, L2@169, L3@746` |
| **Live Colab `gemma-4-31B-it` — Run 2 (`v2`)**: Two-sided zero-centered $P_{\mathrm{callosal}} - 1/K$ + state-scoped `Pi_s(y, x)` + BFS offset removal | `26 / 183` (`14.21%`) | `16 / 25` (`64.0%`) | `1 / 25` (`sb26`) | `4.7787%` | `177` (`1184.2s`) | **Restored `g50t` (`L1@885`) and `vc33` (`L2@171`)**, improved `cd82` (`L1@652`); but negative priors (`-0.147`) on ranks `#2..#10` demoted top-10 lattice medoids below `#11..#36` on `ft09` (`0/6`) and `sc25` (`0/6`) |
| **Live Colab `gemma-4-31B-it` — Run 3 (`v3`, Final)**: One-Sided ReLU Margin $\max(0, P_{\mathrm{callosal}} - 1/K)$ + Beltrami Medoid Precision-Floor Tie-Breaking (`1e-5` bonus) | **`29 / 183`** (`15.85%`) | **`18 / 25`** (`72.0%`) | `1 / 25` (`sb26`) | **`4.8123%`** | `197` (`1126.4s`) | **`25 / 25` level parity (`0` losses)**: restored `sc25` (`L1@141, L2@543`), `ft09` (`L1@619`), `cd82` (`L1@182`), `r11l` (`L1@180`), `g50t` (`L1@885`), `vc33` (`L1@3, L2@171`), while **accelerating `tu93` L1 (`149 -> 132`) and L2 (`187 -> 169`)** |

---

## 6. Drop-In Integration Contracts

### 6.1 Patch & Hook `google/gemma-4-31B-it` for LongBench QASPER / Banal NIAH / ARC-AGI-1
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from darksnail.gemma4 import prepare_darksnail_gemma4_model
from darksnail.longbench import set_long_context_condition

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Patches only the 10 Global layers [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
# and attaches the Layer-29 (5376 -> 4096) Gemma4BeltramiCallosalShell hook
model = prepare_darksnail_gemma4_model(
    model,
    sidecar_path="checkpoints/router_sidecars/gemma4_31b_router_v4.safetensors",
    enable_global_hook=True,
    attach_callosal_shell_hook=True,
    hook_layer=29,
)

# Configure tree depth r in {2, 3, 4, 5} with DarkSnail Query-Tail Outlier Shield
set_long_context_condition(model, req_depth=3, darksnail_mode=True, shield_k=96, decode_k=64)
```

### 6.2 Embodied Interactive `DarkSnailArc3Agent` with Live `google/gemma-4-31B-it` Callosal Deliberation
```python
from arc_agi import Arcade, OperationMode
from darksnail.arc3 import DarkSnailArc3Agent

arcade = Arcade(operation_mode=OperationMode.OFFLINE)
env = arcade.make("sb26-7fbdac44")

agent = DarkSnailArc3Agent(
    max_actions=1000,
    initial_medoid_budget=36,
    expanded_medoid_budget=64,
    seed=0,
    use_gemma4_shell=True,
    max_gemma4_calls_per_level=6,
)
# Optional: attach live patched google/gemma-4-31B-it for Global Layer-29 System-2 deliberation
agent.attach_gemma4_system2(model, tokenizer=tokenizer, hook_layer=29)

result = agent.play_game(env, max_actions=1000)
```

### 6.3 Standalone Surgical Hook Usage (`scripts/build_gemma4_surgical_hooks.py` Contract)
```python
from darksnail.gemma4 import DarkSnailGemma4GlobalHook

hook = DarkSnailGemma4GlobalHook(head_dim=512, d_model=5376, bridge_dim=64, p_rope_ratio=0.25)
k_pruned, v_pruned, hidden_out, keep_mask = hook.forward_global_kv_hook(
    k_global=k_global,           # [B, 4, N, 512]
    hidden_states=hidden_states, # [B, N, 5376]
    dt=0.1,
    steps=4,
)
```

### 6.4 Unified Mechanistic Interpretability & Spherical Lie-Steering Hooks (`darksnail/hooks.py`)
```python
from darksnail.hooks import (
    AllButTheTopGramHook,
    BeltramiShellProbeHook,
    DarkSnailHookSession,
    PRoPESubspaceHook,
    SphericalLieSteeringHook,
)

probe = BeltramiShellProbeHook(outlier_budget=16, condensed_budget=16, isometric_target_dim=4096)
steerer = SphericalLieSteeringHook(subspace_dim=64, dt=0.15)
gram = AllButTheTopGramHook(
    goal_span=(0, 128),
    candidate_spans=[(128, 136), (136, 144), (144, 152)],
    alpha_callosal=0.65,
)
prope = PRoPESubspaceHook(head_dim=512, partial_rotary_factor=0.25, layout="hf_split_half")

with DarkSnailHookSession(model, hooks=[probe, steerer, gram, (29, prope)]):
    _ = model(input_ids)
```

