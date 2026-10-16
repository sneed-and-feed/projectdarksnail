# PROJECT DARKSNAIL V2: Bi-Hemispheric $\times$ Ultrametric Triad Synthesis Blueprint

> [!IMPORTANT]
> **Executive Verdict on the Draft Spec:** The 5-point synthesis between [`llama_surgery.md`](file:///c:/Users/x/Documents/antigravity/sparse-ai/papers/llama_surgery.md) and [`bihemispheric_ai_preprint.tex`](file:///c:/Users/x/Documents/antigravity/brain-ai/docs/preprint/bihemispheric_ai_preprint.tex) is not merely conceptually aligned—a line-by-line inspection of both codebases reveals **exact tensor-shape and spare-parameter matches** that allow us to go significantly further than the initial spec with **zero newly trained backbone weights**.

```mermaid
flowchart LR
    subgraph LH["Left Hemisphere: Frozen Llama-3.1-8B + V4 Triad Sidecar (11.58 MB)"]
        Prompt["Serialized ARC Prompt (T_L tokens)"] --> L0_15["Layers 0..15: SurgicalLlamaAttention (qk_mode, W=128)"]
        L0_15 --> L16["Layer 16: TriadTopologyRouter + BeltramiAdelicCache"]
        L16 -->|Tier-1 Outlier Shield (16) + Tier-2 Medoids (16)| Z_LH["32 Adelic Super-Tokens (1, 32, 4096)"]
        L16 -->|last_triad_flux + LCA Variance| Zeno["Live Zeno Flux \Phi_t"]
    end

    subgraph CC["Corpus Callosum: Dale + Homeostatic Bridge"]
        Z_LH --> Dale["CallosalConditioning (DaleDifferentialCrossAttention)"]
        Dale --> Z0["Callosal Prior Z_0 (16 x 512)"]
        Z0 --> Adapt["System2LatentAdaptor: Z* = Z_0 + \delta z (8,192 float32 params)"]
    end

    subgraph Amygdala["Zeno-Amygdala Controller (darksnail.antenna)"]
        Zeno --> Gate["3-Signal Arbitrator: \Phi_t + \mathcal{L}_0 + H(S1)"]
        Gate -->|Schedule N_TTA, \eta, \lambda_anchor, Ray Quantile| Adapt
    end

    subgraph RH["Right Hemisphere: Frozen TRM + Skew-Symmetric Triad Readout"]
        Adapt -->|Slots 0..38 + K_rel*12| SymReadout["Symmetric Basis: E_t w_eq + \Phi u_rel"]
        Adapt -->|Unused \delta z Slice (5,500+ free coords)| TriadReadout["Skew-Symmetric Triad Breaker: B_W(u, v) in so(12)"]
        Adapt -->|Terminal Carry Step 16| RecurrentTriad["Slow-Fast Triad Coupling: z_L + B_W(z_H^{(15)}, z_L^{(15)})"]
        SymReadout & TriadReadout & RecurrentTriad --> Canvas["W_lm^\dagger Readout -> Exact Output Grid"]
    end
```

---

## 1. Code-Level Audit: 5 Upgrades to the Draft Synthesis Plan

Our deep audit of [`sparse-ai`](file:///c:/Users/x/Documents/antigravity/sparse-ai) and [`brain-ai`](file:///c:/Users/x/Documents/antigravity/brain-ai) uncovered five code-level mechanics where the actual implementations allow a much tighter mathematical and systems synthesis than the initial draft:

### Upgrade 1 (`darksnail.shell`): Exact 32-Slot Replacement of `adaptive_avg_pool1d` in `lh_cache.py`
- **What the Draft Spec Assumed:** Based on Eq. (2) of [`bihemispheric_ai_preprint.tex`](file:///c:/Users/x/Documents/antigravity/brain-ai/docs/preprint/bihemispheric_ai_preprint.tex#L204-L209), the draft assumed `brain-ai` collapses Layer 16 states to a single mean vector $\frac{1}{T_L}\sum_t h_{L,t}^{(16)}$ and directly populates $Z_0 \in \mathbb{R}^{16 \times 512}$.
- **What the Code Actually Does:** In [`extract_lh_latents`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/lh_cache.py#L144-L145), `brain-ai` actually pools variable-length Layer 16 states into **`target_tokens = 32` fixed temporal bins** via `F.adaptive_avg_pool1d(raw_latents.transpose(1, 2).float(), target_tokens=32)` of shape `(1, 32, 4096)`. Then [`CallosalConditioning`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L469-L553) projects `(1, 32, 4096) -> (1, 32, 512)` through [`DaleLinear`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L22-L81) and compresses 32 slots into $Z_0 \in \mathbb{R}^{16 \times 512}$ using 16 learned `task_queries` in [`DaleDifferentialCrossAttention`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L147-L267).
- **Why Uniform 32-Bin Pooling Fails on Serialized Grids:** `F.adaptive_avg_pool1d(..., 32)` averages contiguous slices of the text prompt, mixing high-salience rule transitions (`[Input Grid]`, `[Output Grid]`, rare foreground integers `1..9`) with long runs of `"0 0 0 0"` background tokens.
- **The Upgrade:** Keep the pretrained `CallosalConditioning` weights (`3.68M` params) 100% frozen by matching its exact `(1, 32, 4096)` input contract at Layer 16 using [`BeltramiAdelicCache`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/adelic_cache.py#L118-L299):
  1. **16 Tier-1 Beltrami Outlier-Shielded Tokens ($K_{\mathrm{out}} = 16$):** Extracted via [`BeltramiAdelicCache.compute_beltrami_triad_flux`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/adelic_cache.py#L142-L163), which multiplies the cross-shell exterior curl $\|\hat{u}_{\mathrm{coarse}} \wedge \hat{u}_{\mathrm{fine}}\|_2 = \sqrt{1 - \langle \hat{u}_{\mathrm{coarse}}, \hat{u}_{\mathrm{fine}}\rangle^2}$ by the fine-shell temporal innovation $\|u_{\mathrm{fine}}(t) - u_{\mathrm{fine}}(t-1)\|_2$ with a kernel-5 `max_pool1d` neighborhood shield. This locks onto foreground color anomalies and grid delimiters.
  2. **16 Tier-2 Ultrametric Medoid Super-Tokens ($C = 16$):** Condenses the remaining background/structural tokens into 16 tree clusters using [`_pool_tensor_blocks`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/surgery.py#L246-L280) (`0.5 * (mean_pool + peak_pool)`), preserving RoPE phase coherence via [`rope_medoid_relative_invariance`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/RoPECoherence.lean#L86-L97).

---

### Upgrade 2 (`darksnail.readout`): Harvesting the 5,500+ Unused Coordinates of $\delta z$ for the 3-Tensor Skew-Symmetric Triad Breaker
- **What the Draft Spec Assumed:** Wrote a 2-tensor formula $B(u, v) = ((A - A^\top)u) \odot (W_{\mathrm{child}} v)$ and proposed adding new coupling parameters.
- **What the Code & Lean 4 Proof Actually Provide:**
  1. In [`TriadConservation.lean`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/TriadConservation.lean#L43-L61) and [`TriadTopologyRouter`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/topology.py#L201-L227), the V4 operator is a **3-tensor skew-symmetrized bilinear operator** in $\mathfrak{so}(M)$:
     $$B_W(u, v)_k = \sum_{i,j} u_i \, v_j \, (W_{i,j,k} - W_{i,k,j})$$
     By theorem [`triad_pointwise_orthogonal`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/TriadConservation.lean#L54-L61), this satisfies **exact pointwise orthogonality** $\langle v, B_W(u, v)\rangle = 0$ for *every* parent state $u$ and child state $v$ individually—without needing an equal-and-opposite parent back-reaction to conserve energy!
  2. In [`_apply_callosal_canvas_readout`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1242-L1289), the full-rank branch flattens `delta_z` into `flat_dz` of length $16 \times 512 = 8{,}192$, but **only consumes the first $38 + 12 K_{\mathrm{rel}} \approx 2{,}600$ scalars** (`w_eq = flat_dz[:38]` and `u_rel = flat_dz[38 : 38 + K_rel*12]`). **Over 5,500 coordinates of $\delta z$ currently have zero direct readout path (`grad = 0` in the readout)!**
- **The Upgrade (Zero New Trainable Weights, Zero Step-0 Drift):**
  Map a slice of those idle coordinates in `flat_dz` directly to the skew-symmetric triad generator $W \in \mathbb{R}^{C_{\mathrm{coarse}} \times 12 \times 12}$ (with skew-symmetrization $S_{i,j,k} = W_{i,j,k} - W_{i,k,j}$ across the 12 color logits):
  - Let $u_{b,\ell} \in \mathbb{R}^{C_{\mathrm{coarse}}}$ be the multi-layer structural/panel occupancy summary at pixel $\ell$ (extracted from the competing candidate foreground channels of `E_t`: input foreground, $D_4$ reflections, translations, and split-panel subgrid overlays in [`_get_spatial_relational_tensors`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1129-L1178)).
  - Let $v_{b,\ell} \in \mathbb{R}^{12}$ be the local symmetric color logit vector (or foreground color indicator).
  - Inject the skew-symmetric triad shift in `float32`:
    $$\Delta \ell_{\mathrm{triad}}(b, \ell, k) = \sum_{i=1}^{C_{\mathrm{coarse}}} \sum_{j=0}^{11} u_{b,\ell,i} \, v_{b,\ell,j} \, (W_{i,j,k} - W_{i,k,j})$$
  - **Why this solves the 1–2 pixel occlusion near-misses** (`af22c60d` at $898/900 = 99.78\%$, `0b17323b` at $223/225 = 99.11\%$, `ce039d91` at $99/100 = 99.00\%$, and the half-solved multi-pair tasks `5d2a5c43`, `6ea4a07e`, `e345f17b`):
    1. **Strict Asymmetry ($A \succ B \implies B \prec A$):** If color $j$ and color $k$ collide at an occluded pixel ($v_j > 0, v_k > 0$) under structural context $u_i$, then $S_{i,j,k} = -S_{i,k,j}$ adds $+\alpha$ to color $k$ and $-\alpha$ to color $j$, breaking knife-edge symmetry ties that additive channels in `E_t` cannot distinguish.
    2. **Exact Energy Conservation:** By [`triad_pointwise_orthogonal`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/TriadConservation.lean#L54-L61), $\langle v_{b,\ell}, \Delta \ell_{\mathrm{triad}}(b, \ell) \rangle = 0$ at every pixel, preventing logit norm explosion on background regions (`0` padding) in $30 \times 30$ grids.
    3. **Exact Step-0 Homotopy:** Because $W$ is sliced directly from $\delta z$ (which is initialized at $\mathbf{0}$), $W = 0$ at Step 0 (`inviscid_homotopy_ground_state`), preserving 100% mathematical equivalence with System 1 at $t = 0$.

---

### Upgrade 3 (`darksnail.cascade`): Slow-Fast ($z_H \to z_L$) Terminal-Step Triad Coupling in `callosum.py`
- **What the Code Currently Does:** In [`System2LatentAdaptor.adapt_task`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1649-L1650), [`_get_pre_terminal_carry`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1318-L1372) caches the 15-step unrolled recurrent states $(z_H^{(15)}, z_L^{(15)})$ under `torch.no_grad()` and runs only the 16th terminal step with `Z_0 + delta_z`. However, as diagnosed in [`RESEARCH_FINDINGS_P3_RQ3.md`](file:///c:/Users/x/Documents/antigravity/brain-ai/docs/RESEARCH_FINDINGS_P3_RQ3.md#L76), prefix slicing (`lm_head(z_H)[:, 16:]`) attenuates how `delta_z` steers the 900 canvas tokens inside the recurrent block.
- **The Upgrade:** Inject a low-rank skew-symmetric triad coupling between the Slow state $z_H^{(15)}$ (parent shell) and Fast state $z_L^{(15)}$ (child shell) on the 900 canvas positions before the terminal recurrent step, parameterized by another idle slice of `flat_dz`:
  $$\tilde{z}_L^{(15)} = z_L^{(15)} + B_{W_{\mathrm{rec}}}\!\left(P z_H^{(15)}, \; P z_L^{(15)}\right) P^\top$$
  where $P \in \mathbb{R}^{512 \times d_{\mathrm{shell}}}$ ($d_{\mathrm{shell}} = 8$) projects onto the top-8 singular vectors of $\mathbf{W}_{\mathrm{lm}}$. By [`triad_pointwise_orthogonal`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/TriadConservation.lean#L54-L61), this rotates $z_L^{(15)}$ strictly on the constant-energy sphere $\|P z_L^{(15)}\|_2 = \text{const}$, complementing the Rajan–Abbott spectral radius theorem ($\mathbb{E}[\lambda_{\mathrm{outlier}}] = 0$).

---

### Upgrade 4 (`darksnail.antenna`): Tri-Signal Zeno-Amygdala Controller for Dynamic TTA & Ray Quantile Scheduling
- **What the Code Currently Does:**
  - [`OpenJevAmygdalarRouter`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/amygdala.py#L22-L120) computes cosine conflict $\mathcal{C} \in [0.441, 0.449]$, which is nearly constant across ARC tasks.
  - Meanwhile, [`System2LatentAdaptor.adapt_task`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1622-L1766) runs a static `num_steps = 30`, static `lr = 0.05`, static `anchor_weight = 0.1`, and in [`predict_s1_and_s2_pair`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1926-L1943) uses a fixed velocity ray quantile `q_idx = int(0.15 * numel)` whenever `attempt_1` degrades or ties.
  - Crucially, [`RESEARCH_FINDINGS_P3_RQ3.md`](file:///c:/Users/x/Documents/antigravity/brain-ai/docs/RESEARCH_FINDINGS_P3_RQ3.md#L51) reveals that on `attempt_1`, System 2 improves $186/368$ shape-matched grids, ties $15$, and **over-adapts on $167/368$ grids** (especially large $30 \times 30$ canvases where 30 steps at $\eta = 0.05$ overfits demonstration background regions).
- **The Upgrade:** Combine [`ZenoSpeculativeGenerator._compute_zeno_next_depth`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/speculative.py#L80-L111) with [`ReflexFirstCascadedRouter`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/amygdala.py#L193-L299) using three zero-leakage signals available at Step 0:
  1. **LH Zeno Triad Flux ($\Phi_{\mathrm{LH}}$) & Cophenetic Depth Spread ($\sigma_{\mathrm{LCA}}^2$):** Extracted from [`TriadTopologyRouter.last_triad_flux`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/topology.py#L231) and [`compute_expected_block_distance`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/topology.py#L71-L115) across the serialized demonstration prompt. High $\Phi_{\mathrm{LH}}$ signals multi-scale compositional rule complexity (e.g., split-panel Boolean or Kronecker fractal rules).
  2. **Step-0 Demonstration Loss ($\mathcal{L}_{\mathrm{demo}}^{(0)}$) & Gradient Norm ($\|\nabla_{\delta z} \mathcal{L}_{\mathrm{demo}}^{(0)}\|_2$):** Already computed at Step 0 of [`adapt_task`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/callosum.py#L1685-L1687). Notice from Table 2.5 in [`RESEARCH_FINDINGS_P3_RQ3.md`](file:///c:/Users/x/Documents/antigravity/brain-ai/docs/RESEARCH_FINDINGS_P3_RQ3.md#L137-L156) that tasks where System 1 is already near-optimal (`ca8f78db` at $94.67\%$, `929ab4e9` at $94.10\%$, `903d1b4a` at $90.62\%$) have low Step-0 demo loss ($\mathcal{L}_0 \in [1.38, 2.98]$), whereas discrete Boolean/Kronecker tasks (`0c9aba6e`, `506d28a5`, `be03b35f`) have high Step-0 demo loss ($\mathcal{L}_0 \in [5.9, 7.2]$).
  3. **RH Test-Grid Predictive Entropy ($H_{\mathrm{test}}^{(0)}$) & Canvas Area ($H_{\mathrm{out}} \times W_{\mathrm{out}}$):** Exact analogue of the Zeno entropy gate in [`speculative.py`](file:///c:/Users/x/Documents/antigravity/sparse-ai/src/llama_surgery/speculative.py#L94-L101).
- **Dynamic Zeno-Amygdala Schedule:**
  - **Low-Turbulence / Near-Solved Regime ($\mathcal{L}_0 < 2.5$ or S1 Demo Fit $\ge 90\%$):** High anchor tether ($\lambda_{\mathrm{anchor}} = 0.25$), conservative step budget ($N_{\mathrm{TTA}} = 15$), and surgically targeted ray extrapolation quantile ($q = 0.02$, flipping only the top $1\text{–}2\%$ most uncertain occluded pixels—directly targeting the 1–2 pixel near-miss grids without disturbing the $98\%$ correct canvas!).
  - **High-Turbulence / Compositional Rule Regime ($\Phi_{\mathrm{LH}}$ high, $\mathcal{L}_0 > 4.5$):** Full deliberation ($N_{\mathrm{TTA}} = 35$, $\lambda_{\mathrm{anchor}} = 0.08$, triad coupling weight $\gamma_{\mathrm{triad}} = 1.0$, ray quantile $q = 0.15$).

---

### Upgrade 5 (`darksnail.rope2d`): Formal Transfer of `RoPECoherence.lean` to 2D Axial Spatial Pooling
- **What `RoPECoherence.lean` Proves in 1D:**
  - Arithmetic key averaging destroys the $\mathrm{SO}(2)$ rotary norm and phase ([`key_arithmetic_norm_lt`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/RoPECoherence.lean#L105-L120), [`key_arithmetic_phase_destroyed`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/RoPECoherence.lean#L140-L154)).
  - Selecting the cluster **Medoid Key** $k_{t^*}$ preserves exact relative rotary invariance $\langle R(\theta_q) q, R(\theta_{t^*}) k \rangle = \langle q, R(\theta_{t^*} - \theta_q) k \rangle$ ([`rope_medoid_relative_invariance`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/RoPECoherence.lean#L86-L97)).
  - Scaling condensed keys by $\gamma = \exp(-\nu p^{2\ell} \mathrm{Var}(P_S))$ bounds logit distortion linearly in $\gamma - 1$ ([`condensed_key_logit_scaling`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/RoPECoherence.lean#L172-L181)).
- **The 2D Extension for [`RoPE2D`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/hrm_scaled.py#L29-L68) and Multi-Scale Canvas Pooling:**
  In 2D axial RoPE, the rotary operator factors over $\mathrm{SO}(2)_y \times \mathrm{SO}(2)_x$:
  $$R_{2\mathrm{D}}(y, x) = R(\theta_y) \oplus R(\theta_x)$$
  When pooling $k \times k$ spatial patches ($2\times 2, 3\times 3, 4\times 4$ Kronecker blocks in `E_t` or inside `AxialAttention2D`), arithmetic averaging across a 2D patch induces independent phase cancellation along both $y$ and $x$ axes:
  $$\|\bar{k}_{\mathrm{patch}}\|_2^2 = \frac{1}{|S|^2} \sum_{(y,x),(y',x') \in S} \left(\cos(\theta_y - \theta_{y'}) \|k_y\|^2 + \cos(\theta_x - \theta_{x'}) \|k_x\|^2\right) < \|k\|_2^2$$
  Replacing uniform 2D block pooling with **2D Medoid-Key + Isotropic Adelic Variance Attenuation** $\gamma_{2\mathrm{D}} = \exp\!\left(-\nu p^{2\ell}(\mathrm{Var}_y(S) + \mathrm{Var}_x(S))\right)$ preserves exact 2D translational and dihedral phase coherence across Kronecker scales.

---

## 2. Proposed `darksnail` Architecture & File Layout

We can implement Project DarkSnail cleanly as a modular bridge package (either at the workspace root `c:\Users\x\Documents\antigravity\darksnail\` or integrated directly into `brain-ai/brain_ai/darksnail/`) that imports `llama_surgery` and `brain_ai`:

| Module | Target Path | Core Responsibility |
| :--- | :--- | :--- |
| **`darksnail.shell`** | `brain-ai/brain_ai/darksnail/shell.py` | Wraps frozen `Llama-3.1-8B-Instruct` + [`llama31_8b_router_sidecar.safetensors`](file:///c:/Users/x/Documents/antigravity/sparse-ai/checkpoints/router_sidecars) at Layer 16; replaces `F.adaptive_avg_pool1d(32)` in [`lh_cache.py`](file:///c:/Users/x/Documents/antigravity/brain-ai/brain_ai/models/lh_cache.py#L144) with 16 Beltrami Outlier-Shielded tokens + 16 Medoid Super-Tokens and exports live Zeno flux $\Phi_t$. |
| **`darksnail.readout`** | `brain-ai/brain_ai/darksnail/readout.py` | Implements the `float32` 3-tensor skew-symmetric triad occlusion breaker $B_W(u, v)_k = \sum_{i,j} u_i v_j (W_{i,j,k} - W_{i,k,j})$ and Slow-Fast ($z_H \to z_L$) recurrent triad coupling using the idle coordinates of `delta_z` (`flat_dz[offset:]`), preserving $\delta z = \mathbf{0} \implies \Delta \ell_{\mathrm{triad}} = \mathbf{0}$. |
| **`darksnail.antenna`** | `brain-ai/brain_ai/darksnail/antenna.py` | Implements `ZenoAmygdalaController`: fuses Layer-16 Zeno triad flux $\Phi_{\mathrm{LH}}$, cophenetic LCA variance, Step-0 demo loss $\mathcal{L}_0$, and test-grid predictive entropy $H_{\mathrm{test}}^{(0)}$ to dynamically schedule $(N_{\mathrm{TTA}}, \eta, \lambda_{\mathrm{anchor}}, \gamma_{\mathrm{triad}}, q_{\mathrm{ray}})$. |
| **`darksnail.formal`** | `sparse-ai/formalization/Formalization/Analysis/OcclusionTriad.lean` | Lean 4 formalization proving: (1) skew-symmetric color-pair tie-breaking antisymmetry, (2) pixel-wise logit energy conservation $\langle v, B_W(u, v)\rangle = 0$, (3) Step-0 System 1 homotopy invariance at $\delta z = 0$, and (4) 2D axial RoPE medoid relative invariance (`rope2d_medoid_relative_invariance`). |
| **`darksnail.eval`** | `brain-ai/scripts/evaluate_darksnail_arc.py` | Fast diagnostic & full-benchmark evaluator targeting the near-miss ARC grids (`af22c60d`, `0b17323b`, `ce039d91`, `5d2a5c43`, `6ea4a07e`, `e345f17b`) and the full 400-task ARC-AGI-1 suite. |

---

## 3. Execution Roadmap

1. **Step 1 — Formal Verification (`OcclusionTriad.lean` via `lean-lsp` MCP):**
   Machine-check the DarkSnail synthesis theorems in Lean 4 (extending [`TriadConservation.lean`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/TriadConservation.lean) and [`RoPECoherence.lean`](file:///c:/Users/x/Documents/antigravity/sparse-ai/formalization/Formalization/Analysis/RoPECoherence.lean)):
   - `occlusion_triad_tie_breaker`: proves that for any symmetric tie $v_{c_1} = v_{c_2} > 0$, a non-zero skew-symmetric triad weight $W_{i,c_1,c_2} > W_{i,c_2,c_1}$ strictly separates the post-readout logits $\ell(c_1) - \ell(c_2) = 2 u_i v_{c_1} (W_{i,c_1,c_2} - W_{i,c_2,c_1}) \neq 0$ while keeping total logit energy invariant.
   - `darksnail_step0_system1_homotopy`: proves exact zero-perturbation equivalence at $\delta z = 0$.
   - `rope2d_medoid_relative_invariance`: extends `rope_medoid_relative_invariance` to $\mathrm{SO}(2)_y \times \mathrm{SO}(2)_x$.
2. **Step 2 — Implement `darksnail.readout`, `darksnail.antenna`, and `darksnail.shell`:**
   Build the PyTorch modules with full unit tests verifying zero-drift at Step 0 (`torch.allclose(s1_logits_darksnail, s1_logits_baseline, atol=1e-6)`), exact pointwise orthogonality $\langle v, B_W(u, v)\rangle = 0$ in `float32`, and shape compatibility with the existing `CallosalConditioning` checkpoint.
3. **Step 3 — Empirical Verification on Near-Miss & Solved ARC Grids:**
   Run an ablation on the solved + near-miss ARC tasks (`af22c60d`, `0b17323b`, `ce039d91`, `5d2a5c43`, `6ea4a07e`, `e345f17b`, `bbb1b8b6`, `929ab4e9`, `ca8f78db`) to verify that the skew-symmetric triad occlusion breaker + Zeno-Amygdala adaptive ray/anchor schedule converts the 1–2 pixel occlusion near-misses into exact matches with zero regressions on the 19 already-solved test grids.
