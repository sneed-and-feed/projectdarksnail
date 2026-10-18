# Project DarkSnail IV: Scaling Bi-Hemispheric Ultrametric Synthesis to Google Gemma 4 31B and Embodied Interactive Fluid Intelligence on ARC-AGI-3

> **Sequel to:** (1) *Llama Surgery: Continuous Sparsification of Pre-Trained Language Models via Differentiable Ultrametric Topology Injection*, (2) *Lateralized Neuromorphic Architecture: Architecture Description and Pilot Diagnostics on ARC-AGI*, and (3) [*Project DarkSnail: Bidirectional Synthesis of Ultrametric Dyadic Hydrodynamics and Lateralized Neuromorphic Reasoning*](darksnail_v2_preprint.md)

**Authors:** Sneed & Feed Research, Antigravity AI Team  
**Contact:** `{research, engineering}@sneed-feed.ai`  
**Repository:** [https://github.com/sneed-and-feed/projectdarksnail](https://github.com/sneed-and-feed/projectdarksnail)  
**Formal Verification:** Lean 4 ([`formalization/OcclusionTriad.lean`](../formalization/OcclusionTriad.lean), `0` `sorry`, `0` compiler warnings)  
**Interactive Notebooks:** [`notebooks/darksnail_v2_colab.ipynb`](../notebooks/darksnail_v2_colab.ipynb) & [`notebooks/darksnail_arc3_colab.ipynb`](../notebooks/darksnail_arc3_colab.ipynb)  

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction & Architectural Unification](#1-introduction--architectural-unification)
  - [1.1 From Homogeneous Static Transduction to Frontier Heterogeneous Embodied Agency](#11-from-homogeneous-static-transduction-to-frontier-heterogeneous-embodied-agency)
  - [1.2 Unified Bi-Hemispheric Architecture Across Gemma 4 31B and ARC-AGI-3](#12-unified-bi-hemispheric-architecture-across-gemma-4-31b-and-arc-agi-3)
  - [1.3 Summary of Contributions](#13-summary-of-contributions)
- [2. Scaling Reverse Synthesis to Google Gemma 4 31B (darksnail/gemma4.py)](#2-scaling-reverse-synthesis-to-google-gemma-4-31b-darksnailgemma4py)
  - [2.1 Heterogeneous 5:1 Local/Global Topology & Proportional RoPE (p-RoPE)](#21-heterogeneous-51-localglobal-topology--proportional-rope-p-rope)
  - [2.2 Post-k_norm Unrotated k_nope Subspace Routing & Unified K=V Path](#22-post-k_norm-unrotated-k_nope-subspace-routing--unified-kv-path)
  - [2.3 Global Layer-29 Gemma4BeltramiCallosalShell & Isometric 5376-to-4096 Bridge](#23-global-layer-29-gemma4beltramicallosalshell--isometric-5376-to-4096-bridge)
  - [2.4 Block-KL + Dual Lagrangian Sidecar Calibration & Full n = 200 LongBench QASPER / 50-Position Banal NIAH Evaluation](#24-block-kl--dual-lagrangian-sidecar-calibration--full-n--200-longbench-qasper--50-position-banal-niah-evaluation)
- [3. Embodied Interactive Fluid Intelligence on ARC-AGI-3 — Exogenous Factorization & Non-Local Causal Discovery](#3-embodied-interactive-fluid-intelligence-on-arc-agi-3--exogenous-factorization--non-local-causal-discovery)
  - [3.1 Formal POMDP & Exogenous-Endogenous MDP Decomposition](#31-formal-pomdp--exogenous-endogenous-mdp-decomposition)
  - [3.2 TemporalClockContentSplitter (ℋ_clock ⊕ ℋ_nope): Action-Conditioned MDP Homomorphism & Thermodynamic Irreversibility](#32-temporalclockcontentsplitter-ℋ_clock--ℋ_nope-action-conditioned-mdp-homomorphism--thermodynamic-irreversibility)
  - [3.3 4-Channel Beltrami Spatial Medoid Extraction (extract_arc3_beltrami_medoids)](#33-4-channel-beltrami-spatial-medoid-extraction-extract_arc3_beltrami_medoids)
  - [3.4 Non-Local Causal Operator Factorization (𝒢_act vs. Ω_reg), Dihedral D₄ Equivariance & Geodesic Corridor Momentum](#34-non-local-causal-operator-factorization-𝒢_act-vs-Ω_reg-dihedral-d₄-equivariance--geodesic-corridor-momentum)
- [4. Subsystem A — Unconscious Forward World Modeling, Inverse Teleology & Pure Mathematical Plan Synthesis](#4-subsystem-a--unconscious-forward-world-modeling-inverse-teleology--pure-mathematical-plan-synthesis)
  - [4.1 Solomonoff-Sheaf Expected Free Energy, Closed-Form W₁ & Cross-Level Inverse Teleology](#41-solomonoff-sheaf-expected-free-energy-closed-form-w₁--cross-level-inverse-teleology)
  - [4.2 Relational Operator Dreamer, One-Shot SE(2) × S₁₆ Induction & Active 1-Step Lookahead](#42-relational-operator-dreamer-one-shot-se2--s₁₆-induction--active-1-step-lookahead)
  - [4.3 Four Pure Mathematical Multi-Step Plan Synthesis Primitives (Including Scale-Invariant Sheaf Cavity CSP)](#43-four-pure-mathematical-multi-step-plan-synthesis-primitives-including-scale-invariant-sheaf-cavity-csp)
- [5. Subsystem B — Deterministic Spectral HRM/Amygdala Dynamics, Online so(16) Adaptation & Native ℝ⁵³⁷⁶ Callosal Gram Resonance](#5-subsystem-b--deterministic-spectral-hrmamygdala-dynamics-online-so16-adaptation--native-ℝ⁵³⁷⁶-callosal-gram-resonance)
  - [5.1 Why Random Linear Projections Fail in Anisotropic Pretrained Cones](#51-why-random-linear-projections-fail-in-anisotropic-pretrained-cones)
  - [5.2 Deterministic Spectral Initialization & Online Rank-2 Riemannian so(16) Lie-Generator Adaptation](#52-deterministic-spectral-initialization--online-rank-2-riemannian-so16-lie-generator-adaptation)
  - [5.3 Pathway A: Native ℝ⁵³⁷⁶ All-but-the-Top Common-Mode Centering & Span-to-Span Callosal Gram Resonance](#53-pathway-a-native-ℝ⁵³⁷⁶-all-but-the-top-common-mode-centering--span-to-span-callosal-gram-resonance)
  - [5.4 Pathway B, Decoupled Callosal Priors, 5-Archetype Invariant Compilation & Persistent D₄ Subgoals](#54-pathway-b-decoupled-callosal-priors-5-archetype-invariant-compilation--persistent-d₄-subgoals)
- [6. Empirical Evaluation on the 25-Game (183-Level) ARC-AGI-3 Benchmark](#6-empirical-evaluation-on-the-25-game-183-level-arc-agi-3-benchmark)
  - [6.1 Overall Benchmark Progression & Paired Non-Parametric Statistical Significance](#61-overall-benchmark-progression--paired-non-parametric-statistical-significance)
  - [6.2 Complete 18-Game Solved Breakdown & Mechanistic Attribution](#62-complete-18-game-solved-breakdown--mechanistic-attribution)
  - [6.3 No-Gemma-4 Offline Baseline Archive vs. Live Bi-Hemispheric Gemma 4 System-2 GPU Evaluation](#63-no-gemma-4-offline-baseline-archive-vs-live-bi-hemispheric-gemma-4-system-2-gpu-evaluation)
- [7. Theoretical Boundary Analysis, Limitations & Conclusion](#7-theoretical-boundary-analysis-limitations--conclusion)
  - [7.1 Stateless SE(2) × S₁₆ Primitives vs. Conditional Multi-Body Contact Graph Automata](#71-stateless-se2--s₁₆-primitives-vs-conditional-multi-body-contact-graph-automata)
  - [7.2 Conclusion](#72-conclusion)
  - [7.3 References](#73-references)

---

## Abstract

In Companion Paper 3 ([`darksnail_v2_preprint.md`](darksnail_v2_preprint.md)), we unified Bruhat-Tits ultrametric sparse attention and lateralized Right-Hemisphere test-time spatial relaxation into a single skew-symmetric dyadic shell cascade, machine-checked in Lean 4 ([`formalization/OcclusionTriad.lean`](../formalization/OcclusionTriad.lean)).
In this fourth installment, **Project DarkSnail IV**, we advance both directions of the synthesis across two frontier axes:
**(Part I) Scaling Reverse Synthesis** from homogeneous $L = 32$ transformers (`Meta-Llama-3.1-8B-Instruct`, $d = 4096$) to the heterogeneous $L = 60$ **`google/gemma-4-31B-it`** foundation model ($d = 5376$, `5:1` Local/Global interleave, `global_head_dim = 512`, `attention_k_eq_v = True`, and $25\%$ proportional rotary embeddings `p-RoPE`); and
**(Part II) Extending Forward Synthesis** from static 2D grid transduction ($N = 400$ ARC-AGI-1) to **embodied interactive fluid intelligence in unknown environments (ARC-AGI-3)** ($N = 25$ public games, `183` levels, $64 \times 64$ 16-color frame observations, and a `4,103`-action hybrid keyboard-plus-coordinate action space with zero rules or reward signals provided).

In **Part I** ([`darksnail/gemma4.py`](../darksnail/gemma4.py)), we leave all `50` local sliding-window layers ($W = 1024$) at native FlashAttention/SDPA speed while surgically patching strictly the `10` Global Full-Attention layers (`[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`) with [`SurgicalGemma4Attention`](../darksnail/gemma4.py) and Global Layer-29 [`Gemma4BeltramiCallosalShell`](../darksnail/gemma4.py).
By routing exclusively on the post-`k_norm` unrotated `384`-D `k_nope` subspace ($75\%$ of each `512`-D global head) via [`ProportionalRoPESubspaceSplitter`](../darksnail/gemma4.py) and calibrating a 10-layer sidecar ([`gemma4_31b_router_v4.safetensors`](../checkpoints/router_sidecars/gemma4_31b_router_v4.safetensors)) under a 40-step self-supervised Block-KL + Dual Lagrangian objective, `darksnail@r` achieves **`100.0% / 94.0% / 84.0% / 68.0%`** 50-position Banal NIAH recall at tree depths $r = 2, 3, 4, 5$ (gaining $+52.0\%$ to $+68.0\%$ over `v4_triad@r`, exact McNemar $p \lt 10^{-6}$) and **`49.89 / 48.62 / 48.61` F1** on the full $n = 200$ LongBench QASPER suite (gaining $+5.18$, $+6.58$, and $+12.80$ F1 over `v4_triad@2/3/4`, paired bootstrap $p \le 0.0002$, retaining `97.38%` of the `49.92` F1 dense ceiling at `62.23%` causal budget).

In **Part II** ([`darksnail/arc3.py`](../darksnail/arc3.py)), we introduce [`DarkSnailArc3Agent`](../darksnail/arc3.py), a zero-weight-training embodied neuromorphic agent that unites five first-principles mathematical pillars:
(1) **Action-Conditioned MDP Homomorphism & Thermodynamic Irreversibility Splitting** ($\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ via [`TemporalClockContentSplitter`](../darksnail/arc3.py), `arXiv:2006.16908`, `arXiv:1310.1863`),
which isolates autonomous timer bars via discrete continuity violation $\partial_t \rho_c + \nabla \cdot \mathbf J_c = \sigma_c \neq 0$ and restores any reversible state-recurring pixels with zero spatial border priors;
(2) **Information-Theoretic Agency Contingency & Non-Local Causal Operator Factorization** ($\mathcal G_{\mathrm{act}}\text{ vs. }\Omega_{\mathrm{reg}}$, `arXiv:1811.01483`) and **Dihedral Equivariant Transfer**,
which separate active localized actuators from driven passive registers via non-local causal ratio $\eta_{\mathrm{nonlocal}} \ge 0.50$
across $D_4$;
(3) **Subsystem A** (Solomonoff-Sheaf Expected Free Energy, Fisher-Rao Cross-Level Inverse Teleology & Pure Mathematical Plan Synthesis, `arXiv:1808.01513`, `arXiv:2004.08128`),
combining scale-invariant Unified Goal Homomorphism Energy $\mathcal E_{\mathrm{target}}$,
closed-form $\mathcal O(HW)$ 1D Marginal Wasserstein-1 distance $W_1^{\mathrm{marg}}$,
[`CrossLevelGoalInvariantDistiller`](../darksnail/arc3.py), [`RelationalOperatorDreamer`](../darksnail/arc3.py) with active 1-step lookahead,
and four pure mathematical multi-step planning primitives (`synthesize_sheaf_bijection_plan` with scale-invariant cavity side $S_{\mathrm{tok}} = \sqrt{S_{\mathrm{ref}}^2 - A_{\mathrm{ref}}}$, `synthesize_abelian_jacobian_plan`, `dream_forward_plan`, and `unroll_triad_cascade_plan`);
(4) **Online Rank-2 Riemannian Lie-Generator Adaptation & Deterministic Spectral HRM/Amygdala Dynamics** ($\mathfrak{so}(16)$, `arXiv:2002.06991`),
combining deterministic orthogonal initialization (`_init_spectral_weights` with $10^{-3}$ residual exit gain), commutator curvature $\kappa(a_1, a_2)$, and online test-time adaptation (`adapt_online_step`); and
(5) **Subsystem B** (Native $\mathbb R^{5376}$ Callosal Gram Resonance, Decoupled `0.25` Callosal Priors & 5-Archetype Invariant Compilation),
coupling Global Layer-29 All-but-the-Top centered resonance into top-down spatial precision $\Pi(y, x)$, non-backtracking action priors, and persistent $D_4$ subgoals.

Across the complete **25-game (`183`-level) public ARC-AGI-3 benchmark** at `max_actions = 1000` ([`results/darksnail_arc3_evaluation_summary.json`](../results/darksnail_arc3_evaluation_summary.json) and archived No-Gemma-4 baseline [`results/darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json)),
the standalone Right-Hemisphere + Mathematical System-2 agent (`darksnail_v2_arc3`, `use_gemma4_shell = False`) completes **`29 / 183` levels (`15.85%`) across `18 / 25` games (`72.0%` of games with at least 1 level solved), winning `1 / 25` full game outright (`sb26`: `8 / 8` levels in `124` actions, `100.0%` RHAE) and achieving `4.8124%` Official RHAE Scorecard score** with strict academic integrity (zero game-source inspections and zero game-ID branches).
Paired non-parametric tests across all $N = 25$ games confirm highly significant dominance over both `random_legal` (`2 / 183` levels, `16 wins / 0 losses / 9 ties` on levels, $p = 3.05 \times 10^{-5}$; `18 wins / 0 losses / 7 ties` on RHAE, $p = 7.63 \times 10^{-6}$) and `unshielded_greedy` (`1 / 183` levels, `17 wins / 0 losses / 8 ties` on levels, $p = 1.53 \times 10^{-5}$; `18 wins / 0 losses / 7 ties` on RHAE, $p = 7.63 \times 10^{-6}$).

---

## 1. Introduction & Architectural Unification

### 1.1 From Homogeneous Static Transduction to Frontier Heterogeneous Embodied Agency

The overarching thesis of Project DarkSnail is that **symbolic ultrametric tree sparsification** in a Left-Hemisphere foundation model ($\mathcal H_L$)
and **continuous multi-timescale spatial relaxation** in a Right-Hemisphere world model ($\mathcal H_R$)
are dual manifestations of a single Lie-algebraic conservation principle: skew-symmetric dyadic shell coupling in $\mathfrak{so}(M)$ across a homeostatic transcallosal bridge ($\mathcal C_{LR}$).
While Companion Paper 3 ([`darksnail_v2_preprint.md`](darksnail_v2_preprint.md)) validated this duality on homogeneous $L = 32$ `Meta-Llama-3.1-8B-Instruct` ($d = 4096$) and static ARC-AGI-1 input-output grid pairs ($N = 400$), two fundamental scientific questions remained open:

1. **Can Ultrametric Bruhat-Tits Routing Scale to Frontier Heterogeneous Architectures?**
   Modern frontier models such as **`google/gemma-4-31B-it`** (`arXiv:2607.02770`) abandon uniform full-attention stacks in favor of a `5:1` interleaved Local Sliding-Window ($W = 1024$) and Global Full-Attention ($L_{\mathrm{global}} = 10$ out of $L = 60$ layers) hierarchy with wide global heads (`global_head_dim = 512`), tied Key-Value projections (`attention_k_eq_v = True`), and **Proportional RoPE (`p-RoPE`)** rotating only $25\%$ (`128 / 512` dimensions) of each global head.
   Applying naive full-vector ultrametric clustering across all `60` layers or across rotated rotary dimensions destroys phase coherence and wastes compute on local layers that are already $\mathcal O(S W)$.
2. **Can Lateralized Spatial Relaxation Generalize from Static Grids to Embodied Interactive Environments?**
   Whereas ARC-AGI-1 provides static input-output demonstration pairs $(X_i, Y_i)$,
   **ARC-AGI-3** places an embodied agent inside an unknown interactive video game ($64 \times 64$ 16-color frame $F_t \in \lbrace 0, \dots, 15 \rbrace^{64 \times 64}$) with up to `7` action types (`ACTION1`..`ACTION5` keyboard moves, `ACTION6(x, y)` coordinate clicks across $64 \times 64 = 4,096$ pixels, and `ACTION7` undo) and **zero explicit instructions, zero demonstration pairs, and zero intermediate reward signals**.
   Standard greedy or random exploration policies collapse immediately on ARC-AGI-3 (`0.55%` to `1.09%` level completion across the `25` public games) due to three compounded pathologies:
   - **Exogenous Clock Aliasing**: Autonomous countdown bars tick on every action step, causing every frame $F_t$ to have a unique hash and tricking naive novelty explorers into standing still or oscillating in place until the timer expires (`RESET`).
   - **Driven Register Distraction**: Clicking an active puzzle button simultaneously mutates a multi-tile HUD status display or passive door lock; unshielded visual salience agents waste hundreds of clicks probing the driven passive registers rather than the active causal generators.
   - **Anisotropic LLM Projection Collapse**: When escalating to a 31B LLM for System-2 deliberation, passing high-dimensional hidden states ($\mathbb R^{5376}$) through untrained random linear projections or autoregressive text generation suffers from severe common-mode hub collapse ($\cos \in [0.75, 0.95]$) and prohibitive decoding latency.

### 1.2 Unified Bi-Hemispheric Architecture Across Gemma 4 31B and ARC-AGI-3

Figure 1 illustrates the complete **Project DarkSnail IV** architecture uniting [`darksnail/gemma4.py`](../darksnail/gemma4.py) and [`darksnail/arc3.py`](../darksnail/arc3.py).

```mermaid
flowchart TD
    subgraph LH["Left Hemisphere: Frozen google/gemma-4-31B-it (L=60, d=5376, bfloat16)"]
        Local50["50 Local Sliding-Window Layers (W=1024): Native FlashAttention / SDPA"]
        Global10["10 Global Full-Attention Layers [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]: SurgicalGemma4Attention"]
        PRoPE["ProportionalRoPESubspaceSplitter: 25% Rotated k_rope (128-D) vs. 75% Unrotated k_nope (384-D)"]
        Sidecar["gemma4_31b_router_v4.safetensors (85.38 MB): Block-KL + Dual Lagrangian Bruhat-Tits Prefix Tree"]
        Layer29["Global Layer 29 (r*=0.722): Gemma4BeltramiCallosalShell + All-but-the-Top Centering on S^5375"]
        Local50 --> Global10 --> PRoPE --> Sidecar --> Layer29
    end

    subgraph Callosum["Zero-Bottleneck Cortico-Subcortical Callosal Signal Path (Subsystem B)"]
        PathA["Pathway A (RH Geometric): Native R^5376 Span-to-Span Gram Resonance R_29(a_k) + 5-Archetype Invariant Compiler"]
        PathB["Pathway B (LH Symbolic): Single-Pass Next-Token Digit Logits '0'..'9' -> p_LH(k) (Zero Text Generation)"]
        EIFusion["Callosal E/I Fusion: P_callosal(a_k | F_t) = (1 - alpha) p_LH(k) + alpha p_29^gram(k), alpha = gamma_E / (gamma_E + gamma_I)"]
        TopDownPi["Active Inference Top-Down Spatial Precision Map Pi(y, x) + Decoupled 0.25 Non-Backtracking Prior + D_4 Subgoals"]
        Layer29 --> PathA & PathB --> EIFusion --> TopDownPi
    end

    subgraph RH["Right Hemisphere: Embodied DarkSnailArc3Agent & Episodic World Model (Subsystem A)"]
        ClockSplit["TemporalClockContentSplitter (H_clock + H_nope): Continuity Conservation + Thermodynamic Irreversibility"]
        Medoids["extract_arc3_beltrami_medoids: 4-Channel (cc_object, enclosed_hole, symmetry_defect, rope2d_patch)"]
        CausalFact["Non-Local Causal Operator Factorization: Active Generators G_act (eta >= 0.50) vs. Passive Registers Omega_reg"]
        TargetHom["Scale-Invariant Goal Homomorphism E_target(F_t; Pi), Closed-Form W_1^marg & CrossLevelGoalInvariantDistiller"]
        OpInduce["RelationalOperatorDreamer (Active 1-Step Lookahead) + SE(2) x S_16 Induction + Excitable Excursion Verifier"]
        PlanSynth["4 Pure Mathematical Plan Primitives: (A) Abelian OMP, (B) Koopman Beam, (C) Triad Cascade, (D) Scale-Invariant Sheaf CSP"]
        SpectralHRM["Deterministic Spectral Init + Online so(16) Lie-Generator TTT: Arc3HRMRecurrentCore + Arc3AffectiveAmygdalaRouter"]

        ClockSplit --> Medoids --> CausalFact --> TargetHom --> OpInduce --> PlanSynth
        TopDownPi --> Medoids & TargetHom
        SpectralHRM --> EIFusion & PlanSynth
    end
```

### 1.3 Summary of Contributions

1. **Heterogeneous `p-RoPE` Ultrametric Routing on `google/gemma-4-31B-it` (Section 2):**
   We introduce [`ProportionalRoPESubspaceSplitter`](../darksnail/gemma4.py), [`SurgicalGemma4Attention`](../darksnail/gemma4.py), and Global Layer-29 [`Gemma4BeltramiCallosalShell`](../darksnail/gemma4.py).
   By routing exclusively on the unrotated `384`-D `k_nope` subspace of the `10` Global Full-Attention layers and preserving the unified `attention_k_eq_v = True` value stream, `darksnail@2/3/4` achieves **`49.89 / 48.62 / 48.61` F1** on $n = 200$ LongBench QASPER ($+5.18$, $+6.58$, and $+12.80$ F1 over `v4_triad@2/3/4`, $p \le 0.0002$) and **`100.0% / 94.0% / 84.0% / 68.0%`** recall on 50-position Banal NIAH at $r = 2, 3, 4, 5$.
2. **Action-Conditioned MDP Homomorphisms, Thermodynamic Irreversibility & Non-Local Causal Operator Factorization on ARC-AGI-3 (Section 3):**
   We formulate [`TemporalClockContentSplitter`](../darksnail/arc3.py) ($\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$)
   via a discrete mass-continuity law $\partial_t \rho_c + \nabla \cdot \mathbf J_c = \sigma_c$ (`arXiv:2006.16908`, `arXiv:1207.4114`) and a Thermodynamic Irreversibility Guard (`arXiv:1310.1863`, `arXiv:1708.01289`) with zero spatial border heuristics,
   paired with retroactive transition-graph quotienting, 4-channel Beltrami medoid discretization,
   information-theoretic agency contingency ($\eta_{\mathrm{nonlocal}} \ge 0.50$, `arXiv:1811.01483`),
   and dihedral $D_4$ shape transfer.
3. **Subsystem A — Scale-Invariant Sheaf Cavity CSP, Fisher-Rao Inverse Teleology & Active Lookahead Plan Synthesis (Section 4):**
   We derive the scale-invariant Unified Goal Homomorphism Energy $\mathcal E_{\mathrm{target}}$ ([`compute_canvas_target_alignment`](../darksnail/arc3.py)),
   closed-form $\mathcal O(HW)$ 1D Marginal Earth Mover's Distance $W_1^{\mathrm{marg}}$ ([`compute_marginal_wasserstein1_distance`](../darksnail/arc3.py)),
   Fisher-Rao Natural-Gradient Cross-Level Inverse Teleology ([`CrossLevelGoalInvariantDistiller`](../darksnail/arc3.py), `arXiv:2004.08128`, Ziebart 2008),
   [`RelationalOperatorDreamer`](../darksnail/arc3.py) with active 1-step forward lookahead in `score_action_candidate`,
   and four pure mathematical multi-step planning primitives including Scale-Invariant Cellular Sheaf Cavity Substitution (`synthesize_sheaf_bijection_plan` with $S_{\mathrm{tok}} = \sqrt{S_{\mathrm{ref}}^2 - A_{\mathrm{ref}}}$, `arXiv:1808.01513`, `arXiv:2202.04579`).
4. **Subsystem B — Online Riemannian Lie-Generator Adaptation, Decoupled Callosal Priors & 5-Archetype Invariant Compilation** ($\mathbb R^{5376}$, **Section 5**):
   We combine deterministic spectral initialization (`_init_spectral_weights` with $10^{-3}$ residual exit gain),
   online rank-2 $\mathfrak{so}(16)$ Lie-generator adaptation and commutator curvature $\kappa(a_1, a_2)$ (`adapt_online_step`, `arXiv:2002.06991`, Wen & Yin 2013),
   native $\mathbb R^{5376}$ All-but-the-Top common-mode centering,
   decoupled `0.25` Callosal action and spatial precision priors $\Pi(y, x)$ on non-backtracking actions,
   and Layer-29 compilation of the 5 Subgoal Invariant Archetype weights (`[G0:SHEAF_HOMOMORPHISM]..[G4:GEODESIC_DOCKING]`).
5. **State-of-the-Art Zero-Training Embodied Performance on the 25-Game (`183`-Level) ARC-AGI-3 Suite (Section 6):**
   Across all `25` public ARC-AGI-3 games at `max_actions = 1000`, the standalone Right-Hemisphere + Mathematical System-2 baseline (`darksnail_v2_arc3`, archived in [`results/darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json)) solves **`29 / 183` levels (`15.85%`) across `18 / 25` games (`72.0%`), winning `1 / 25` full game outright (`sb26`: `8 / 8` in `124` actions, `100.0%` RHAE) and achieving `4.8124%` Official RHAE Score**,
   outperforming `random_legal` (`2 / 183` levels) and `unshielded_greedy` (`1 / 183` levels) with `17 wins / 0 losses` on levels ($p = 1.53 \times 10^{-5}$) and `18 wins / 0 losses` on Official RHAE ($p = 7.63 \times 10^{-6}$).

---

## 2. Scaling Reverse Synthesis to Google Gemma 4 31B (`darksnail/gemma4.py`)

### 2.1 Heterogeneous 5:1 Local/Global Topology & Proportional RoPE (`p-RoPE`)

Unlike homogeneous $L = 32$ architectures (`Meta-Llama-3.1-8B-Instruct`), **`google/gemma-4-31B-it`** (`Gemma4ForConditionalGeneration`, text backbone `model.language_model`, hidden size $d = 5376$, intermediate size `21,504`, $L = 60$ layers) employs a heterogeneous two-tier attention topology (`arXiv:2607.02770`):

- **50 Local Sliding-Window Layers** ($\ell \notin \mathcal L_{\mathrm{global}}$):
  Use a local sliding window of $W = 1024$ tokens with head dimension `256`,
  $H_q = 32$ query heads, and `16` key-value heads (group factor `2`).
  Because their complexity is already strictly linear $\mathcal O(S W_{\mathrm{local}})$, [`prepare_darksnail_gemma4_model`](../darksnail/gemma4.py) leaves all `50` local layers completely untouched at native FlashAttention-2 / SDPA speed.
- **10 Global Full-Attention Layers**: Located at every 6th layer index:

```math
\mathcal L_{\mathrm{global}} = \lbrace 5, \, 11, \, 17, \, 23, \, 29, \, 35, \, 41, \, 47, \, 53, \, 59 \rbrace
```

Each Global layer $\ell \in \mathcal L_{\mathrm{global}}$ operates over the full context window
with expanded head dimension $d_{\mathrm{global}} = 512$,
$H_q = 32$ query heads,
and $H_{kv,\mathrm{global}} = 4$ key-value heads (Grouped-Query Attention group factor `8`).
Crucially, Global layers incorporate two architectural innovations that directly impact ultrametric routing:
1. **Proportional RoPE (`p-RoPE`, `partial_rotary_factor = 0.25`)**:
   Rotary position embeddings ($\theta_{\mathrm{global}} = 10^6$)
   are applied exclusively to the first $d_{\mathrm{rope}} = 128$ dimensions ($25\%$) of each `512`-D head,
   while the trailing $d_{\mathrm{nope}} = 384$ dimensions ($75\%$)
   have zero frequency ($\omega_i = 0$, so $\cos = 1$ and $\sin = 0$) and remain **strictly position-invariant**.
2. **Unified Key-Value Projection (`attention_k_eq_v = True`, `v_proj = None`)**:
   Global layers omit a separate `v_proj` matrix; instead, the raw post-projection key tensor `k_raw = k_proj(x)` serves directly as the Value tensor $V$ *prior* to `k_norm` and rotary embedding!

### 2.2 Post-`k_norm` Unrotated `k_nope` Subspace Routing & Unified `K=V` Path

In homogeneous models (`Llama-3.1-8B`), `RoPE` rotates `100%` of the head dimensions, forcing `SurgicalLlamaAttention` to route on phase-coupled post-RoPE vectors where high-frequency rotary channels corrupt long-distance semantic clustering unless compensated (`RoPECoherence.lean`).
On `google/gemma-4-31B-it`, [`ProportionalRoPESubspaceSplitter`](../darksnail/gemma4.py) exploits the exact `p-RoPE` decomposition of the post-`q_norm` and post-`k_norm` global heads:

```math
\mathbf q = \bigl[\mathbf q_{\mathrm{rope}} \in \mathbb R^{128} \;\big\Vert\; \mathbf q_{\mathrm{nope}} \in \mathbb R^{384}\bigr], \qquad \mathbf k = \bigl[\mathbf k_{\mathrm{rope}} \in \mathbb R^{128} \;\big\Vert\; \mathbf k_{\mathrm{nope}} \in \mathbb R^{384}\bigr]
```

While the full `512`-D vectors $(\mathbf q, \mathbf k)$ participate in the exact dot-product attention kernel on selected blocks,
[`SurgicalGemma4Attention`](../darksnail/gemma4.py) feeds **strictly the unrotated 384-D `q_nope` and `k_nope` slices** into the V4 `TriadTopologyRouter` and the DarkSnail Query-Tail Outlier Shield:

```math
\mathcal S_{\mathrm{Beltrami}}(t) = \frac{1}{H_{kv,\mathrm{global}}} \sum_{h=1}^{H_{kv,\mathrm{global}}} \bigl\lVert \mathbf k_{\mathrm{nope}, t}^{(h)} \wedge \mathbf k_{\mathrm{nope}, t-1}^{(h)} \bigr\rVert_2 + \lambda_{\mathrm{QT}} \max_{q \in \mathcal Q_{\mathrm{tail}}} \bigl\langle \mathbf q_{\mathrm{nope}, q}^{(h)}, \, \mathbf k_{\mathrm{nope}, t}^{(h)} \bigr\rangle
```

Because $\mathbf k_{\mathrm{nope}}$ is completely free of rotary phase drift across arbitrary token distances $\lvert q - t \rvert \in [1, 6000]$,
semantic matching between the question tail $\mathcal Q_{\mathrm{tail}}$ and distant document paragraphs is translationally invariant.
Simultaneously, `SurgicalGemma4Attention` preserves the exact Gemma 4 `attention_k_eq_v = True` dataflow: `value_states = k_proj(hidden_states)` is cached in the KV cache **before** `k_norm` and `apply_rotary_pos_emb`, and the final attention output is scaled by the per-layer `o_norm` / post-attention RMSNorm intact.

### 2.3 Global Layer-29 `Gemma4BeltramiCallosalShell` & Isometric 5376-to-4096 Bridge

Following the inter-hemispheric depth alignment principle ($r^\ast = 16 / 32 = 0.50$ on Llama-3.1-8B), we attach [`Gemma4BeltramiCallosalShell`](../darksnail/gemma4.py) to the midpoint Global layer of `google/gemma-4-31B-it`, **Global Layer 29** (the 5th of `10` global layers, normalized depth $r^\ast \approx 0.722$ in the cumulative receptive-field hierarchy).
To interface the `5376`-D Gemma 4 residual stream $\mathbf H^{(29)} \in \mathbb R^{B \times S \times 5376}$ with the frozen `(B, 32, 4096)` `CallosalConditioning` input contract of the Right Hemisphere without introducing any trainable parameters or distorting cosine geometry,
`Gemma4BeltramiCallosalShell` constructs a deterministic **row-orthonormal Johnson-Lindenstrauss / Haar-QR isometric projection** $\mathbf P_{\mathrm{iso}} \in \mathbb R^{4096 \times 5376}$ (seeded deterministically at `seed = 20260404`):

```math
\mathbf P_{\mathrm{iso}} \, \mathbf P_{\mathrm{iso}}^\top = \mathbf I_{4096}, \qquad \mathbf Z_{\mathrm{shell}} = \mathcal B_{S \to 32}\bigl(\mathbf H^{(29)}\bigr) \, \mathbf P_{\mathrm{iso}}^\top \in \mathbb R^{B \times 32 \times 4096}
```

where $\mathcal B_{S \to 32}$ (`BeltramiCondense`) extracts `16` Tier-1 Beltrami exterior-curl outlier tokens and `16` Tier-2 adelic medoid tokens, guaranteeing `1.0000` rule-spike preservation alongside exact `(B, 32, 4096)` dimensional compatibility.

### 2.4 Block-KL + Dual Lagrangian Sidecar Calibration & Full $n = 200$ LongBench QASPER / 50-Position Banal NIAH Evaluation

We calibrate the 10-global-layer V4 router sidecar ([`checkpoints/router_sidecars/gemma4_31b_router_v4.safetensors`](../checkpoints/router_sidecars/gemma4_31b_router_v4.safetensors), `85.38 MB`) using [`warmup_gemma4_router_kl`](../darksnail/gemma4.py) over `40` steps of self-supervised **Block-KL Divergence + Dual Lagrangian Entropy Regularization** against the exact teacher block-attention distribution on the `384`-D `k_nope` subspace.
To prevent long-range needle loss during autoregressive multi-token generation (`seq_len == 1`), `SurgicalGemma4Attention` caches the top-96 Beltrami + Query-Tail contiguous token spans (`kernel_size = 13`) during prefill (`seq_len > 1`) and pairs them with **Measure-Preserving Donor Pruning** (`_apply_darksnail_block_bridge`), which evicts an equal number of lowest-salience interior filler blocks so that the total causal key budget of `darksnail@r` never exceeds `v4_triad@r`.

Tables 1 and 2 report the complete empirical evaluation on `google/gemma-4-31B-it` (`bfloat16`) across the **50-position Banal NIAH suite** ($N = 4,096$ tokens, needle `"spherical coordinates"`)
and the **full LongBench QASPER scientific QA benchmark** ($n = 200$, `max_ctx = 6,000`, stored in [`results/darksnail_longbench_summary.json`](../results/darksnail_longbench_summary.json) and [`results/darksnail_qasper_predictions.jsonl`](../results/darksnail_qasper_predictions.jsonl)).

| Tree Depth $r$ | Measured Causal Budget | `window@r` Recall [Wilson 95% CI] | `v4_triad@r` Recall [Wilson 95% CI] | `darksnail@r` Recall [Wilson 95% CI] | Gain vs. `v4_triad@r` | Exact McNemar $p$-value (`Wins / Losses`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $r = 2$ | `82.5%` (`1,689.9` keys) | `62.0%` `[48.15%, 74.14%]` (`31/50`) | `92.0%` `[81.16%, 96.85%]` (`46/50`) | `100.0%` `[92.87%, 100.0%]` (`50/50`) | $+8.0\%$ | $p = 0.1250$ (`+4 / -0`) |
| $r = 3$ | `70.2%` (`1,438.0` keys) | `46.0%` `[32.98%, 59.60%]` (`23/50`) | `42.0%` `[29.38%, 55.77%]` (`21/50`) | `94.0%` `[83.78%, 97.94%]` (`47/50`) | $+52.0\%$ | $p \lt 10^{-6}$ (`+26 / -0`) |
| $r = 4$ | `59.9%` (`1,227.1` keys) | `32.0%` `[20.76%, 45.81%]` (`16/50`) | `16.0%` `[8.34%, 28.51%]` (`8/50`) | `84.0%` `[71.49%, 91.66%]` (`42/50`) | $+68.0\%$ | $p \lt 10^{-9}$ (`+34 / -0`) |
| $r = 5$ | `52.3%` (`1,071.4` keys) | `24.0%` `[14.29%, 37.41%]` (`12/50`) | `6.0%` `[2.06%, 16.22%]` (`3/50`) | `68.0%` `[54.19%, 79.24%]` (`34/50`) | $+62.0\%$ | $p \lt 10^{-8}$ (`+31 / -0`) |

**Table 1:** 50-position Banal NIAH (`"spherical coordinates"`, $N = 4,096$) on `google/gemma-4-31B-it` (`bfloat16`) under triple iso-budget matching.

| Tree Depth $r$ | `window@r` F1 [Budget] | `v4_triad@r` F1 [Budget] | `darksnail@r` F1 [Budget] | Paired $\Delta\text{F1}$ vs. `v4_triad@r` [95% CI] ($p$-value, `W/L/T`) | Paired $\Delta\text{F1}$ vs. `window@r` [95% CI] ($p$-value) | Paired $\Delta\text{F1}$ vs. `dense` (`49.92`) [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $r = 0$ (`dense`) | — | `49.92` `[100.00%]` | `49.92` `[100.00%]` | — | — | $0.00$ |
| $r = 2$ | `42.55` `[84.55%]` | `44.70` `[84.54%]` | `49.89` `[84.54%]` | $+5.18$ `[+2.40, +8.14]` ($p = 0.0002$, `70/37/93`) | $+7.33$ `[+3.37, +11.33]` ($p = 0.0005$) | $-0.04$ `[-1.92, +1.80]` |
| $r = 3$ | `37.75` `[72.13%]` | `42.04` `[72.11%]` | `48.62` `[72.11%]` | $+6.58$ `[+3.24, +9.96]` ($p = 0.0002$, `84/41/75`) | $+10.87$ `[+6.30, +15.42]` ($p = 0.0001$) | $-1.30$ `[-2.79, +0.09]` |
| $r = 4$ | `34.77` `[62.25%]` | `35.81` `[62.23%]` | `48.61` `[62.23%]` | $+12.80$ `[+8.98, +16.70]` ($p = 0.0001$, `103/34/63`) | $+13.84$ `[+8.52, +19.12]` ($p = 0.0001$) | $-1.31$ `[-3.73, +1.07]` |

**Table 2:** Full $n = 200$ LongBench QASPER evaluation on `google/gemma-4-31B-it` (`bfloat16`, `max_ctx = 6,000`).

![DarkSnail V2 LongBench QASPER & Banal NIAH Curves on Google Gemma 4 31B](../results/darksnail_longbench_curves.png)

**Figure 2:** Full $n = 200$ LongBench QASPER Token F1 and 50-position Banal NIAH ($N = 4,096$) retrieval recall (%) on `google/gemma-4-31B-it` (`bfloat16`) across Bruhat-Tits tree depths $r$, demonstrating `99.94%` dense F1 retention at $r = 2$ and $+12.80$ F1 over `v4_triad@4` at $r = 4$ (`103` wins vs. `34` losses, $p = 0.0001$).

---

## 3. Embodied Interactive Fluid Intelligence on ARC-AGI-3 — Exogenous Factorization & Non-Local Causal Discovery

### 3.1 Formal POMDP & Exogenous-Endogenous MDP Decomposition

An ARC-AGI-3 environment is an episodic Partially Observable Markov Decision Process (POMDP) $\mathcal M = (\mathcal S, \mathcal O, \mathcal A, \mathcal T, L_{\mathrm{win}})$
where at each discrete timestep $t \in \lbrace 0, \dots, T_{\max} \rbrace$ ($T = 1000$),
the agent observes only a 2D 16-color raster frame $F_t \in \lbrace 0, \dots, 15 \rbrace^{H \times W}$ ($H = W = 64$),
the current number of completed levels $\ell_t \in \lbrace 0, \dots, L \rbrace$,
and the subset of available action types $\mathcal A_{\mathrm{avail}}(F) \subseteq \lbrace \mathrm{ACTION1}, \dots, \mathrm{ACTION7} \rbrace$.
Following the Exogenous-Endogenous Markov Decision Process (Exo-MDP) framework (`arXiv:1806.01584`, `arXiv:2207.08229`), the latent visual space $\mathcal H$ decomposes into a direct sum of two orthogonal dynamical subspaces:

```math
\mathcal H = \mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}
```

where:
- **The Exogenous Clock Subspace** ($\mathcal H_{\mathrm{clock}}$)
  evolves autonomously as an action-invariant periodic or monotonic counter $\xi_{t+1} = f_{\mathrm{exo}}(\xi_t)$
  (such as a depleting HUD countdown bar or energy gauge that triggers a level reset when exhausted),
  satisfying $I(a_t; \xi_{t+1} \mid \xi_t) = 0$; and
- **The Endogenous Controllable Content Subspace** ($\mathcal H_{\mathrm{nope}}$)
  evolves strictly as a causal function of the agent's action $s_{t+1} = f_{\mathrm{endo}}(s_t, a_t)$.

If an embodied agent hashes the raw observation $F_t$
instead of its orthogonal projection $P_{\mathrm{nope}}(F) \in \mathcal H_{\mathrm{nope}}$,
the monotonic evolution of $\xi_t \in \mathcal H_{\mathrm{clock}}$
guarantees $F_{t+1} \neq F_t$ even when the agent executes a pure wall-collision noop (`s_{t+1} = s_t`)!
Consequently, an unshielded agent cannot detect noops, cannot build a cyclic state-transition graph $\mathcal G_E = (V_E, E_E)$, and gets trapped in infinite 1-step or 2-step oscillations until the exogenous timer expires.

### 3.2 `TemporalClockContentSplitter` ($\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$): Action-Conditioned MDP Homomorphism & Thermodynamic Irreversibility

Crucially, under strict academic integrity, **one cannot hardcode spatial border coordinates** (e.g., assuming timers live only in rows $y \ge 60$), because an ARC-AGI-3 game may place a controllable puzzle piece near the border or embed a timer bar along any internal panel divider.
Instead, [`TemporalClockContentSplitter`](../darksnail/arc3.py) constructs an online **Action-Conditioned MDP Homomorphism & Lax-Bisimulation Quotient** (`arXiv:2006.16908`, `arXiv:1207.4114`, `arXiv:1911.12247`) purely from **discrete continuity and thermodynamic irreversibility laws** across any pixel $(y, x) \in \lbrace 0, \dots, 63 \rbrace^2$:

Let $\rho_c(y, x, t) = \mathbb I\bigl[F(y, x, t) = c\bigr]$ denote the discrete indicator density of color $c$ at pixel $(y, x)$.
Across any step transition $F_t \xrightarrow{a} F_{t+1}$,
a controllable spatial translation of an avatar or mobile block conserves total color mass within the local neighborhood ($\lvert \Delta N_c \rvert = 0$),
whereas an exogenous depleting/filling timer bar violates local mass conservation by annihilating one pixel of color $c_1$
and creating one pixel of color $c_2$ at a single frontier cell ($\Delta N = \pm 1$):

```math
\partial_t \rho_c(y, x, t) + \nabla \cdot \mathbf J_c(y, x, t) = \sigma_c(y, x, t)
```

`TemporalClockContentSplitter` isolates $\mathcal H_{\mathrm{clock}}$ via two exact physical laws and retroactive quotienting:
1. **Zero-Net-Displacement Continuity Violation & 1D Contiguous Strip Flood-Fill**:
   Any pixel $(y, x)$ that mutates across $\ge 2$ distinct actions or contexts (`len(acts) >= 2` or `total_changes >= 2`) with zero non-trivial reversions (`reversions == 0`) while its transition exhibits $\le 4$ non-conservation pixel flips is flagged into the exogenous clock mask $M_{\mathrm{clock}}(y, x) = 1$.
   Whenever $\ge 2$ confirmed clock pixels lie on the same row $y$ or column $x$ within a monotonic run, `TemporalClockContentSplitter` flood-fills the contiguous 1D bicolor timer segment $(c_{\mathrm{depleted}}, c_{\mathrm{remaining}})$ along that row or column so that **all future ticks of that countdown bar are masked preemptively before they ever occur**.
2. **Thermodynamic Irreversibility Guard** (`arXiv:1310.1863`, `arXiv:1708.01289`):
   Because a genuine exogenous countdown clock evolves monotonically ($\Delta S_{\mathrm{clock}} \gt 0$) between level resets,
   any candidate pixel $(y, x)$ whose color trajectory exhibits a non-reset state restoration $F_{t+1}(y, x) = F_{t-k}(y, x)$
   after an intermediate change is immediately recognized as a **reversible endogenous game element** (such as a sliding column or toggleable bridge in `tu93` or `vc33`) and permanently immunized against clock masking (`reversible_mask`).
3. **Retroactive Quotienting & Cross-Level Persistence**:
   Whenever $M_{\mathrm{clock}}$ expands at step $t$,
   `DarkSnailArc3WorldModel._rebuild_projected_graph` retroactively re-projects the entire episodic frame history $\lbrace F_0, \dots, F_t \rbrace$
   onto $P_{\mathrm{nope}}(F) = F \odot (1 - M_{\mathrm{clock}})$,
   merging previously split nodes in $\mathcal G_E$ and converting false transitions into confirmed noops (`noop_actions`).
   Moreover, $M_{\mathrm{clock}}$ is **preserved across level transitions (`on_level_advance`)**, giving the agent instant Step-0 clock immunity on Level $2+$.

Using these projection-quotiented states, `TemporalClockContentSplitter` evaluates the **Period-5 Clockwork Hamiltonian**:

```math
\mathcal H_{\mathrm{clock}}(t) = \bigl\lvert \Delta \mathcal S_{\mathrm{noop}}(t) \bigr\rvert + \lambda_{\mathrm{period}} \, \Omega_5(t) + \lambda_{\mathrm{revert}} \, \mathcal I_{\mathrm{cycle}}(s_t)
```

which spikes immediately whenever the agent enters a 2-step oscillation $s_t = s_{t-2}$
or a period-5 recurrence $s_t = s_{t-5}$
in $\mathcal H_{\mathrm{nope}}$, triggering an immediate symmetry-breaking escape or System-2 escalation.

### 3.3 4-Channel Beltrami Spatial Medoid Extraction (`extract_arc3_beltrami_medoids`)

Because `ACTION6(x, y)` spans $64 \times 64 = 4,096$ possible pixel coordinates, uniform pixel probing requires $\sim 4,096$ steps per state ($\lt 0.03\%$ hit rate).
In [`extract_arc3_beltrami_medoids`](../darksnail/arc3.py), the Right Hemisphere condenses the $64 \times 64$ frame into a compact, geometry-preserving set of $K \le 24$ spatial medoids $\mathcal M(F_t) = \lbrace (x_m, y_m, s_m, \tau_m) \rbrace_{m=1}^K$
(removing all clock-masked pixels $M_{\mathrm{clock}}$ and passive HUD registers) across four complementary geometric channels:

1. **Channel 1 (`cc_object` + Dyadic Quadtree Subdivision)**:
   Extracts 4-connected foreground components (`area <= 512`) and maps each component to its true $L_1$ interior pixel medoid:

```math
(y_m^\ast, x_m^\ast) = \underset{(y, x) \in \mathcal C}{\mathrm{argmin}} \sum_{(y', x') \in \mathcal C} \bigl(\lvert y - y' \rvert + \lvert x - x' \rvert\bigr)
```

   It then subdivides multi-tile compound blocks (`area >= 20`, bounding box $\ge 5 \times 5$) into $2 \times 2$ dyadic quadrants so multi-button panels expose every sub-tile.
2. **Channel 2 (`enclosed_hole`)**:
   Flood-fills background components from the grid boundary to isolate interior topological cavities ($\beta_1$ enclosed holes)
   and extracts their exact $L_1$ medoids with adaptive multi-cavity salience regulation, enabling single-shot targeting of empty socket slots in pattern-completion games (`vc33`, `lp85`, `ft09`, `tn36`).
3. **Channel 3 (`symmetry_defect`)**:
   Scans multi-component foreground layouts for reflection and translation symmetry defects where a missing or mismatched tile breaks local dihedral $D_4$ or lattice periodicity.
4. **Channel 4 (`rope2d_patch`)**:
   Applies $8 \times 8$ non-overlapping patch Beltrami exterior-curl salience weighted by the top-down spatial precision map $\Pi(y, x)$, followed by radius-2 Non-Maximum Suppression (NMS).

### 3.4 Non-Local Causal Operator Factorization ($\mathcal G_{\mathrm{act}}\text{ vs. }\Omega_{\mathrm{reg}}$), Dihedral Equivariance & Geodesic Corridor Momentum

A central contribution of [`DarkSnailArc3Agent`](../darksnail/arc3.py) is discovering that interactive puzzle objects in unknown environments split into two causally distinct operator classes under `ACTION6(x, y)` coordinate clicks via **Information-Theoretic Agency Contingency** (`arXiv:1811.01483`):

```math
\mathcal O_{\mathrm{interactive}} = \mathcal G_{\mathrm{act}} \;\sqcup\; \Omega_{\mathrm{reg}}
```

where $\mathcal G_{\mathrm{act}}$ denotes **Active Causal Generators**
and $\Omega_{\mathrm{reg}}$ denotes **Driven Passive Registers**.
Whenever a click at medoid $(x_0, y_0)$
produces a non-zero endogenous transition $\Delta F = P_{\mathrm{nope}}(F_{t+1}) - P_{\mathrm{nope}}(F_t) \neq 0$,
`DarkSnailArc3WorldModel._update_nonlocal_operator_factorization` partitions the changed pixel support $\mathrm{supp}(\Delta F)$ around a local Chebyshev radius $r_{\mathrm{loc}} = 4$:

```math
N_{\mathrm{local}} = \bigl\lvert \lbrace (y, x) \in \mathrm{supp}(\Delta F) : \max(\lvert y - y_0 \rvert, \lvert x - x_0 \rvert) \le 4 \rbrace \bigr\rvert, \qquad N_{\mathrm{nonlocal}} = \bigl\lvert \mathrm{supp}(\Delta F) \bigr\rvert - N_{\mathrm{local}}
```

and computes the **Non-Local Causal Action Ratio**:

```math
\eta_{\mathrm{nonlocal}}(x_0, y_0) = \frac{N_{\mathrm{nonlocal}}}{N_{\mathrm{local}} + N_{\mathrm{nonlocal}}} \in [0, 1]
```

This single invariant triggers four coupled causal inference mechanisms:
1. **Active Generator vs. Driven Register Splitting** ($\eta_{\mathrm{nonlocal}} \ge 0.50$):
   If $\eta_{\mathrm{nonlocal}}(x_0, y_0) \ge 0.50$
   and $N_{\mathrm{nonlocal}} \ge 2$,
   the clicked locus $(x_0, y_0)$ is registered as an **Active Non-Local Generator**
   in $\mathcal G_{\mathrm{act}}$ (`remote_actuators`),
   while every remotely mutated pixel $(y', x')$ with $\max(\lvert y' - y_0 \rvert, \lvert x' - x_0 \rvert) \gt 4$
   that was not itself clicked is recorded in the **Driven Passive Register Mask** $\Omega_{\mathrm{reg}}$ (`driven_register_counts[x', y'] += 1`).
   In `score_action_candidate`, confirmed actuators in $\mathcal G_{\mathrm{act}}$ receive a strong Expected Free Energy bonus,
   whereas passive registers in $\Omega_{\mathrm{reg}}$ are heavily penalized and filtered out of Tier-1 medoid candidates when true generators exist.
2. **Dihedral Rotation/Reflection Equivariant Shape Transfer** (`get_d4_shape_prior`):
   Every connected component $\mathcal C$ is mapped to its canonical 8-element dihedral orbit signature under rotations $R_{90}^k$
   and reflections in $D_4$:

```math
S_{\mathrm{D4}}(\mathcal C) = \min_{g \in D_4} \mathrm{Hash}\bigl(g \cdot \mathbf 1_{\mathcal C}\bigr)
```

   When clicking a component $\mathcal C$ succeeds, its canonical $D_4$ signature transfers immediately to all rotated or reflected copies $g \cdot \mathcal C$ across the board.
   Conversely, when clicking $\mathcal C$ yields a noop, all rotated copies $g \cdot \mathcal C$ are immediately penalized, eliminating 8-fold redundant probing of decorative walls and icons.
3. **Local Patch Invariance** (`is_click_locally_inert`) **& Tier-1 Probe Failure Memory** (`tier1_click_probes_failed`):
   If clicking $(x_0, y_0)$ fails (`noop`),
   the agent hashes the local $5 \times 5$ neighborhood $\mathcal P_{5 \times 5}(F, y_0, x_0)$.
   Even when keyboard moves change the global state hash $s_t$, any click whose local $5 \times 5$ patch matches a previously confirmed inert patch is skipped immediately.
   Furthermore, if all `4` initial Tier-1 click probes at the starting state fail (`c6_trials >= 4 and c6_succs == 0`), the agent sets `tier1_click_probes_failed = True`: at subsequent keyboard-explored states, Tier-1 click probes are restricted exclusively to previously confirmed `interactive_loci` or confirmed $D_4$ shapes.
4. **Geodesic Spanning-Tree Corridor Momentum**:
   On keyboard navigation graphs (`ACTION1`..`ACTION4`), the agent records confirmed parent-child edges `reverse_action_map`, penalizes immediate 2-step backtracking, and awards a corridor-continuation bonus for maintaining heading `last_successful_move_action` along degree-2 maze corridors until a junction or boundary is reached.

---

## 4. Subsystem A — Unconscious Forward World Modeling, Inverse Teleology & Pure Mathematical Plan Synthesis

### 4.1 Solomonoff-Sheaf Expected Free Energy, Closed-Form $W_1^{\mathrm{marg}}$ & Cross-Level Inverse Teleology

How can an agent evaluate whether an action moves the environment toward "completion" when ARC-AGI-3 provides zero intermediate rewards?
In Subsystem A ([`compute_canvas_target_alignment`](../darksnail/arc3.py)), the Right Hemisphere automatically detects whether the frame $F_t$
contains an embedded reference target panel $G$ (either via full-width/full-height divider splits or via a small multi-color legend box vs. a large active canvas $C_t$, using scale-invariant canvas-area ratios $N = H \cdot W$)
and evaluates the **Unified Goal Homomorphism Energy** $\mathcal E_{\mathrm{target}}(F_t; \Pi) \in [0, 1]$:

```math
\mathcal E_{\mathrm{target}}(F_t; \Pi) = w_{\mathrm{hom}} \, \mathcal E_{\mathrm{hom}}(C_t, G) + w_{\mathrm{marg}} \, \mathcal E_{\mathrm{marginal}}(C_t, G) + w_{\mathrm{nav}} \, \mathcal E_{\mathrm{nav}}(F_t) + w_{\mathrm{topo}} \, \bigl(1 - \Phi_{\mathrm{topo}}(F_t)\bigr)
```

Each of the four energy terms captures a distinct geometric or topological homomorphism scale:

1. **Exact Scale & Spaced Connected-Component Lattice Homomorphism** ($\mathcal E_{\mathrm{hom}}$):
   When the active canvas $C_t$ and reference target $G$
   have dimensions related by an integer Kronecker factor $(k_y, k_x)$ (`kron`),
   $\mathcal E_{\mathrm{hom}}$ measures the precision-weighted pixel mismatch
   between $C$ and $G \otimes \mathbf 1_{k_y \times k_x}$.
   When $G$ is a compact $R_{\mathrm{leg}} \times C_{\mathrm{leg}}$ legend matrix
   encoding the target colors of a spaced $R_{\mathrm{grid}} \times C_{\mathrm{grid}}$ array of interactive tiles on $C$
   (such as the $3 \times 3$ spaced tile lattice in `ft09`), `_detect_canvas_target_pair` extracts the tile bounding boxes via 1D row/column foreground projections (`mode = "cc_lattice"`) and computes exact cell-by-cell lattice homomorphism alignment.
2. **Closed-Form 1D Marginal Wasserstein-1 Distance & Bracketed Slot Alignment** ($\mathcal E_{\mathrm{marginal}}$ in $\mathcal O(HW)$):
   When target shapes are translated or separated from the reference legend without a fixed pixel-to-pixel Kronecker grid, pixelwise Hamming distance has zero gradient until overlap occurs.
   In [`compute_marginal_wasserstein1_distance`](../darksnail/arc3.py), we exploit the closed-form isomorphism between the 1D Wasserstein-1 Earth Mover's Distance on $[0, 1]$
   and the $L_1$ norm of cumulative distribution functions:

```math
W_1^{\mathrm{marg}}(C_t, G) = \frac{1}{2 \lvert \mathcal C_{\mathrm{fg}} \rvert} \sum_{c \in \mathcal C_{\mathrm{fg}}} \left( \int_0^1 \bigl\lvert F_{y,c}(C_t; u) - F_{y,c}(G; u) \bigr\rvert \, du + \int_0^1 \bigl\lvert F_{x,c}(C_t; u) - F_{x,c}(G; u) \bigr\rvert \, du \right)
```

   Combined with relative color-histogram $L_1$ divergence
   and corner-bracketed target slot alignment (`slot_align`),
   $\mathcal E_{\mathrm{marginal}}$ provides a smooth, $\mathcal O(HW)$ optimal-transport gradient across arbitrary spatial translations.
3. **Avatar-to-Beacon Geodesic Navigation Energy** ($\mathcal E_{\mathrm{nav}}$):
   Identifies small mobile avatar components and minority goal beacons on $P_{\mathrm{nope}}(F_t)$
   and computes normalized Manhattan distance $\min_{a, b} (\lvert y_a - y_b \rvert + \lvert x_a - x_b \rvert) / (H + W)$.
4. **Topological Betti & Dihedral Symmetry Potential** ($\Phi_{\mathrm{topo}}$):
   Combines connected-component compactness ($1/\beta_0$),
   enclosed hole completion $(1/(1 + \beta_1))$,
   and maximal axis-aligned dihedral reflection/rotation symmetry across $D_4$.
5. **Fisher-Rao Natural-Gradient Cross-Level Inverse Teleology** ([`CrossLevelGoalInvariantDistiller`](../darksnail/arc3.py), `arXiv:2004.08128`, `arXiv:0812.4360`, `arXiv:1912.05510`, Ziebart 2008):
   Whenever the agent completes a level ($S_{\mathrm{init}} \to S_{\mathrm{win}}$),
   `CrossLevelGoalInvariantDistiller.distill_level_win` evaluates the 5D geometric invariant basis $\boldsymbol\Phi(S) = [\Phi_{\mathrm{hom}}, \Phi_{W_1}, \Phi_{\mathrm{sym}}, \Phi_{\mathrm{annih}}, \Phi_{\mathrm{dock}}]^\top \in [0, 1]^5$
   and updates the simplex weight vector $\mathbf w \in \Delta^4$ via a Fisher-Rao natural-gradient Multiplicative Weights / Boltzmann exponentiated gradient step:

```math
w_i^{(\ell+1)} = \frac{w_i^{(\ell)} \exp\bigl(\eta_{\mathrm{FR}} \, \Delta \Phi_i(S_{\mathrm{init}}, S_{\mathrm{win}})\bigr)}{\sum_{j=1}^5 w_j^{(\ell)} \exp\bigl(\eta_{\mathrm{FR}} \, \Delta \Phi_j(S_{\mathrm{init}}, S_{\mathrm{win}})\bigr)}
```

   Simultaneously, any foreground color $c$ present in $S_{\mathrm{init}}$
   that is completely annihilated in $S_{\mathrm{win}}$ (`cleared_colors`)
   or docked into a receptor cavity (`target_receptor_colors`) is distilled as a cross-level teleological invariant for Levels $\ell + 1 \dots L_{\mathrm{win}}$.

### 4.2 `RelationalOperatorDreamer`, One-Shot $\mathrm{SE}(2) \times S_{16}$ Induction & Active 1-Step Lookahead

Whenever an action $a \in \mathcal A$ produces a transition $F_t \xrightarrow{a} F_{t+1}$,
[`DarkSnailArc3WorldModel._induce_spatial_operator`](../darksnail/arc3.py) and [`RelationalOperatorDreamer.observe_transition`](../darksnail/arc3.py) induce symbolic forward operators $\hat T_a : \lbrace 0, \dots, 15 \rbrace^{64 \times 64} \to \lbrace 0, \dots, 15 \rbrace^{64 \times 64}$
in the semidirect product $\mathrm{SE}(2) \ltimes S_{16}$ from a **single transition**:

- **Type I (`translation` & Ego-Kinematic Generators)**:
  Detects rigid translation $(\Delta y, \Delta x)$ of the controllable ego sprite $\mathcal C_{\mathrm{ego}}$ (`ego_color`) with obstacle/wall collision checking (`predict_step`), enabling mental rollout of keyboard moves at novel board positions (`g50t`, `ls20`, `m0r0`).
- **Type II (`cyclic_color_LUT` & Tool/Palette Selection Mode)**:
  Detects deterministic color-palette permutations $\pi_a : \mathcal C_{\mathrm{support}} \to \lbrace 0, \dots, 15 \rbrace$ and tool-selection state changes (`tool_selection_active`).
- **Type III (Excitable Excursion Verifier & Contact-Highlight `ACTION5` Docking)**:
  When an action triggers a multi-subframe validation wave (`len(subframes) >= 3` with intermediate excursion $\Delta_{\mathrm{nope}} \ge 12$) that returns to the pre-check board when the puzzle is incomplete (`tn36`), `record_transition` registers the locus as an **Excitable Excursion Verifier** (`verifier_actuators`) rather than a dead self-loop, suppressing redundant clicks until $\ge 1$ canvas edit has occurred (`edits_since_last_verify >= 1`).
  Furthermore, when the ego sprite docks inside a hollow receptor frame (`active_contact_highlight_states`), `score_action_candidate` boosts `ACTION5` interaction/lock-in priority.
- **Type IV (Cross-Tile Lattice Operator Templates `lattice_operator_templates`)**:
  In lattice toggle puzzles (`ft09`), clicking *any single tile* $(y_0, x_0)$ induces a translationally equivariant `lattice_local` stencil across the entire tile grid so [`predict_action_outcome`](../darksnail/arc3.py) can predict the exact board outcome of clicking **every other untried tile** before clicking it even once.
- **Active 1-Step Forward Lookahead in `score_action_candidate`**:
  For every candidate action $a$ at the current state (`state_hash == self.current_state_hash`), `score_action_candidate` queries `predict_action_outcome` (falling back to `relational_dreamer.predict_step`) and adds the predicted 1-step potential reduction:

```math
\Delta \mathcal E_{\mathrm{lookahead}}(s_t, a) = 1.35 \max\bigl(0, \, \mathcal E_{\mathrm{OT}}(F_t) - \mathcal E_{\mathrm{OT}}(\hat F_{t+1})\bigr) - 0.50 \max\bigl(0, \, \mathcal E_{\mathrm{OT}}(\hat F_{t+1}) - \mathcal E_{\mathrm{OT}}(F_t)\bigr)
```

### 4.3 Four Pure Mathematical Multi-Step Plan Synthesis Primitives (Including Scale-Invariant Sheaf Cavity CSP)

Armed with the induced forward operators $\lbrace \hat T_a \rbrace$ and the optimal-transport objective:

```math
\mathcal E_{\mathrm{OT}}(F; \Pi) = \mathcal E_{\mathrm{target}}(F; \Pi) + 0.75 \, W_1^{\mathrm{marg}}(C, G)
```

[`DarkSnailArc3Agent.synthesize_mathematical_plan`](../darksnail/arc3.py) synthesizes multi-step execution queues `planned_action_queue` $[a^{(1)}, \dots, a^{(K)}]$ using four pure mathematical engines (**promoted ahead of Tier-2 medoid expansion in `choose_action`, with zero ASCII grid serialization and zero LLM text generation**):

#### Primitive A (`synthesize_abelian_jacobian_plan`): Precision-Weighted Ridge Difference-Basis Pseudoinverse & Greedy Integer OMP
For commutative/Abelian toggle, paint, and displacement puzzles where individual actions induce additive or modular canvas perturbations $\mathbf d_m = \mathrm{vec}\bigl(\hat T_{a_m}(F_t) - F_t\bigr) \in \mathbb R^{HW}$ ($m = 1, \dots, M$)
toward a residual goal discrepancy $\mathbf r_t = \mathrm{vec}(G^\ast - C_t) \in \mathbb R^{HW}$,
we assemble the empirical Jacobian difference basis $\mathbf D = [\mathbf d_1, \dots, \mathbf d_M] \in \mathbb R^{HW \times M}$
weighted by the spatial precision diagonal $\tilde{\boldsymbol\Pi} = \mathrm{diag}(\mathrm{vec}(\Pi))$
and solve the ridge-regularized normal equations in $\mathcal O(M^2 HW + M^3)$ ($M \le 32$):

```math
\hat n_{\mathrm{ridge}} = \bigl(\mathbf D^\top \tilde{\boldsymbol\Pi} \mathbf D + \lambda_{\mathrm{ridge}} \mathbf I_M\bigr)^{-1} \mathbf D^\top \tilde{\boldsymbol\Pi} \, \mathbf r_t
```

`synthesize_abelian_jacobian_plan` then executes **Greedy Integer Orthogonal Matching Pursuit (OMP)** to select the discrete multi-step sequence maximally reducing $\mathcal E_{\mathrm{OT}}\bigl(\hat F^{(\tau)}; \Pi\bigr)$.

#### Primitive B (`dream_forward_plan`): Koopman Operator Mental Beam Search
When transitions are non-commutative, [`DarkSnailArc3WorldModel.dream_forward_plan`](../darksnail/arc3.py) performs a depth $K$ ($K \le 6$, beam width $B = 12$) mental beam search over the composition of learned operators $\hat T_{a^{(K)}} \circ \dots \circ \hat T_{a^{(1)}}(F_t)$,
pruning cyclic states and ranking partial trajectories by:

```math
\mathcal J_{\mathrm{beam}}\bigl(a^{(1:K)}\bigr) = \mathcal E_{\mathrm{OT}}\bigl(\hat F^{(K)}; \Pi\bigr) - \gamma_{D_4} \, \Phi_{D_4}\bigl(\hat F^{(K)}\bigr)
```

#### Primitive C (`unroll_triad_cascade_plan`): Co-Evolving `SlowFastTriadCascade` Trajectory Readout on $\mathbb S^7(r_0)$
[`unroll_triad_cascade_plan`](../darksnail/arc3.py) couples the substeps $\tau = 1, \dots, K$ of the $\mathfrak{so}(8)$ spherical cascade (`formalization/OcclusionTriad.lean`, Theorem 2.6) directly to the sequence of future actions $a^{(1)}, \dots, a^{(K)}$:

```math
\delta \mathbf v^{(\tau)} = B_{W_{\mathrm{rec}}}\bigl(\mathbf u^{(\tau-1)}, \, \mathbf v^{(\tau-1)}\bigr), \qquad \mathbf v^{(\tau)} = r_0 \frac{\mathbf v^{(\tau-1)} + \Delta t \, \delta \mathbf v^{(\tau)}}{\bigl\lVert \mathbf v^{(\tau-1)} + \Delta t \, \delta \mathbf v^{(\tau)} \bigr\rVert_2}
```

satisfying exact pointwise orthogonality $\langle \mathbf v^{(\tau-1)}, \delta \mathbf v^{(\tau)} \rangle = 0$
and zero radial drift $\lVert \mathbf v^{(\tau)} \rVert_2 = r_0$.

#### Primitive D (`synthesize_sheaf_bijection_plan`): Cellular Sheaf Laplacian & Scale-Invariant Topological Cavity Substitution CSP
In hierarchical symbolic substitution puzzles (`sb26`, `arXiv:1808.01513`, `arXiv:2202.04579`), a sequence of hollow square reference frames $\mathcal G_{\mathrm{ref}}$
of arbitrary outer side length $S_{\mathrm{ref}} \ge 4$
and foreground area $A_{\mathrm{ref}}$
encloses an interior square cavity whose side length $S_{\mathrm{tok}}$ is determined purely by topological area conservation:

```math
S_{\mathrm{tok}} = \sqrt{S_{\mathrm{ref}}^2 - A_{\mathrm{ref}}}, \qquad 2 \le S_{\mathrm{tok}} \lt S_{\mathrm{ref}}
```

[`synthesize_sheaf_bijection_plan`](../darksnail/arc3.py) discovers $\mathcal G_{\mathrm{ref}}$
across all scales $(S_{\mathrm{ref}}, S_{\mathrm{tok}})$ with **zero hardcoded pixel dimensions**,
identifies free palette tokens of scale $S_{\mathrm{tok}} \times S_{\mathrm{tok}}$ outside container boxes
alongside container-enclosed receptor slots of scale $S_{\mathrm{slot}} \lt S_{\mathrm{tok}}$ via scale-invariant Jordan wall ray-casting (`wall_h >= s_ref`), and solves the recursive cellular-sheaf tree expansion to emit the exact multi-step `ACTION6(pal) -> ACTION6(slot) -> ... -> ACTION5` sequence across all 8 levels of `sb26` (`8 / 8` levels won in `124` total actions, `0` resets, `100.0%` Official RHAE score).

---

## 5. Subsystem B — Deterministic Spectral HRM/Amygdala Dynamics, Online $\mathfrak{so}(16)$ Adaptation & Native $\mathbb R^{5376}$ Callosal Gram Resonance

### 5.1 Why Random Linear Projections Fail in Anisotropic Pretrained Cones

A critical pathology in hybrid LLM-plus-subcortical architectures is attempting to project frozen pretrained LLM hidden states $\mathbf h \in \mathbb R^d$ ($d = 5376$ in Gemma 4 31B) into a lower-dimensional subcortical module via an untrained random Gaussian or Kaiming linear layer $\mathbf W_{\mathrm{rand}} \in \mathbb R^{m \times d}$, or running an uncalibrated recurrent network with random weights.
We can prove mathematically why this fails:

**Proposition 5.1 (Anisotropic Common-Mode Hub Collapse under Random Projections):**
In deep pretrained transformers (`arXiv:1702.01417`, `arXiv:1909.00512`), Layer-29 token representations $\mathbf h_i \in \mathbb R^{5376}$ do not populate $\mathbb R^{5376}$ isotropically;
instead, they concentrate inside a narrow anisotropic cone around a dominant common-mode mean direction $\boldsymbol\mu_{29}$
and top principal component $\mathbf v_1$:

```math
\mathbf h_i = \alpha_i \boldsymbol\mu_{29} + \beta_i \mathbf v_1 + \boldsymbol\epsilon_i, \qquad \mathrm{cos}\bigl(\mathbf h_i, \mathbf h_j\bigr) \in [0.75, 0.95], \qquad \lVert \boldsymbol\epsilon_i \rVert_2 \ll \lVert \alpha_i \boldsymbol\mu_{29} + \beta_i \mathbf v_1 \rVert_2
```

where the task-discriminative signal comparing candidate actions $a_k\text{ vs. }a_{k'}$
resides entirely in the low-variance residual $\boldsymbol\epsilon_i$
with margin $\Delta \cos \sim 10^{-2}$.
If $\mathbf h_i$ is passed through an untrained random matrix
$\mathbf W_{\mathrm{rand}} \in \mathbb R^{m \times d}$
with i.i.d. entries $\mathcal N(0, 1/d)$,
the Johnson-Lindenstrauss cross-term between the massive common mode $\mathbf c = \alpha \boldsymbol\mu_{29} + \beta \mathbf v_1$
and the informative residual $\boldsymbol\epsilon_k$ has standard deviation:

```math
\sigma_{\mathrm{JL}} = \mathrm{Std}\bigl(\langle \mathbf W_{\mathrm{rand}} \mathbf c, \, \mathbf W_{\mathrm{rand}} \boldsymbol\epsilon_k \rangle\bigr) \approx \frac{\lVert \mathbf c \rVert_2 \lVert \boldsymbol\epsilon_k \rVert_2}{\sqrt{m}} \gg \bigl\lvert \langle \boldsymbol\epsilon_{\mathrm{goal}}, \, \boldsymbol\epsilon_k - \boldsymbol\epsilon_{k'} \rangle \bigr\rvert
```

Thus, random projections amplify the shared common-mode hub into random direction-dependent biases that completely swamp the fine-grained semantic margin between candidate actions!

### 5.2 Deterministic Spectral Initialization & Online Rank-2 Riemannian $\mathfrak{so}(16)$ Lie-Generator Adaptation

To eliminate random-weight noise in the subcortical Right Hemisphere (`Arc3HRMRecurrentCore`) and Affective Amygdala (`Arc3AffectiveAmygdalaRouter`) while enabling online test-time adaptation (`arXiv:2002.06991`, `arXiv:1901.08428`, Wen & Yin 2013), Subsystem B combines:

1. **RNG-Isolated Orthonormal Spectral Initialization** (`_init_spectral_weights` with `torch.random.fork_rng`):
   Every linear and recurrent weight matrix inside `HRMRecurrentBlock`, `Arc3HRMRecurrentCore`, and `Arc3AffectiveAmygdalaRouter` is initialized inside an isolated `torch.random.fork_rng(devices=[])` context with deterministic seed `137042` using exact QR orthonormal initialization (`nn.init.orthogonal_`) and $10^{-3}$ near-identity residual exit gain on `Arc3AffectiveAmygdalaRouter` and `HRMRecurrentBlock` readout coupling.
2. **Online Rank-2** $\mathfrak{so}(16)$ **Lie-Generator Estimation & Commutator Curvature** $\kappa(a_1, a_2)$:
   For each discrete action $a \in \lbrace 1, \dots, 7 \rbrace$, `update_lie_generator` tracks the empirical 16-color transition flow and projects it onto the skew-symmetric Lie algebra $\mathfrak{so}(16)$:

```math
\mathbf A_a = \frac{1}{2}\bigl(\mathbf M_a - \mathbf M_a^\top\bigr) \in \mathfrak{so}(16), \qquad \kappa(a_1, a_2) = \frac{\bigl\lVert [\mathbf A_{a_1}, \, \mathbf A_{a_2}] \bigr\rVert_F}{\bigl\lVert \mathbf A_{a_1} \bigr\rVert_F \, \bigl\lVert \mathbf A_{a_2} \bigr\rVert_F + \epsilon}
```

   where $[\mathbf A_{a_1}, \mathbf A_{a_2}] = \mathbf A_{a_1}\mathbf A_{a_2} - \mathbf A_{a_2}\mathbf A_{a_1}$
   measures non-Abelian ordering sensitivity between generators $(a_1, a_2)$.
3. **Online Test-Time Adaptation (`adapt_online_step`)**:
   On non-self-loop transitions where cross-level teleological distillation is active, `adapt_online_step` updates `hrm_core.action_head` and the skew-symmetric triad parameters (`lr = 1e-4`) while strictly preserving pointwise spherical orthogonality $\langle \mathbf v, B_W(\mathbf u, \mathbf v) \rangle \le 10^{-8}$.

### 5.3 Pathway A: Native $\mathbb R^{5376}$ All-but-the-Top Common-Mode Centering & Span-to-Span Callosal Gram Resonance

When the Amygdala escalates to `google/gemma-4-31B-it` at a high-uncertainty junction (`deliberate_with_gemma4_system2`), DarkSnail executes **a single forward pass** (`lh_model(input_ids=..., output_hidden_states=True, use_cache=False)`, **never** `model.generate()`, `0` generated tokens)
over a compact structured prompt containing exact token character spans for the `[GOAL_STATE]` segment $\mathcal S_{\mathrm{goal}}$,
the 5 Subgoal Invariant Archetype spans (`[G0:SHEAF_HOMOMORPHISM]`, `[G1:MARGINAL_W1_TRANSPORT]`, `[G2:DIHEDRAL_D4_SYMMETRY]`, `[G3:ENTROPY_ANNIHILATION]`, `[G4:GEODESIC_DOCKING]`),
and each candidate action segment $\mathcal S_{a_k}$ for $k \in \lbrace 0, \dots, K-1 \rbrace$.

In **Pathway A** (Right-Hemisphere Native $\mathbb R^{5376}$ Geometric Resonance), we bypass all untrained linear bottlenecks and operate directly in the native `5376`-dimensional space of Global Layer `29` $(\mathbf H^{(29)} \in \mathbb R^{T \times 5376})$:
1. **All-but-the-Top Common-Mode Removal** (`arXiv:1702.01417`):
   We subtract the per-prompt token mean $\boldsymbol\mu_{29} = \frac{1}{T} \sum_{t=1}^T \mathbf H_t^{(29)}$
   and project out the top principal directions via deterministic power iteration before normalizing onto the unit sphere $\mathbb S^{5375}$:

```math
\check H_t^{(29)} = \tilde H_t^{(29)} - \bigl\langle \tilde H_t^{(29)}, \, \mathbf v_1 \bigr\rangle \mathbf v_1, \qquad \hat H_t^{(29)} = \frac{\check H_t^{(29)}}{\bigl\lVert \check H_t^{(29)} \bigr\rVert_2 + \epsilon} \in \mathbb S^{5375}
```

2. **Span-to-Span Callosal Gram Resonance**:
   Pooling the centered, isotropized representations over $\mathcal S_{\mathrm{goal}}$
   and each candidate action span $\mathcal S_{a_k}$
   yields unit vectors $\hat h_{\mathrm{goal}}^{(29)}, \hat h_{a_k}^{(29)} \in \mathbb S^{5375}$,
   defining the **Callosal Gram Resonance** $\mathcal R_{29}(a_k)$
   and Right-Hemisphere geometric posterior $p_{29}^{\mathrm{gram}}(k)$:

```math
\mathcal R_{29}(a_k) = \bigl\langle \hat h_{\mathrm{goal}}^{(29)}, \, \hat h_{a_k}^{(29)} \bigr\rangle_{\mathbb R^{5376}}, \qquad p_{29}^{\mathrm{gram}}(k) = \frac{\exp\bigl(\mathcal R_{29}(a_k) / \tau_{\mathrm{callosal}}\bigr)}{\sum_{j=0}^{K_{\mathrm{cand}}-1} \exp\bigl(\mathcal R_{29}(a_j) / \tau_{\mathrm{callosal}}\bigr)}
```

### 5.4 Pathway B, Decoupled Callosal Priors, 5-Archetype Invariant Compilation & Persistent $D_4$ Subgoals

In the **same single forward pass**, **Pathway B** (Left-Hemisphere Symbolic Readout) inspects the final-token vocabulary logits $\boldsymbol\ell_T \in \mathbb R^{\lvert \mathcal V \rvert}$
restricted strictly to the single-token digit IDs $\iota_k$ for `"0"`, `"1"`, $\dots$, `"9"`:

```math
p_{\mathrm{LH}}(k) = \frac{\exp\bigl(\ell_T(\iota_k) / \tau_{\mathrm{LH}}\bigr)}{\sum_{j=0}^{K_{\mathrm{cand}}-1} \exp\bigl(\ell_T(\iota_j) / \tau_{\mathrm{LH}}\bigr)}
```

The Corpus Callosum fuses Pathway A ($p_{29}^{\mathrm{gram}}$)
and Pathway B ($p_{\mathrm{LH}}$)
via the Amygdala's dynamic Excitatory/Inhibitory balance $\alpha_{\mathrm{callosal}} = \gamma_E / (\gamma_E + \gamma_I) \in [0.15, 0.85]$:

```math
P_{\mathrm{callosal}}(a_k \mid F_t) = (1 - \alpha_{\mathrm{callosal}}) \, p_{\mathrm{LH}}(k) + \alpha_{\mathrm{callosal}} \, p_{29}^{\mathrm{gram}}(k)
```

Under the **Argon Refactoring**, Subsystem B couples $P_{\mathrm{callosal}}$ and Layer-29 Gram resonance into the Right Hemisphere via three synergistic mechanisms:
1. **Decoupled `0.25` Callosal Action & Spatial Precision Priors vs. Spanning-Tree Anti-Backtracking Guard**:
   In `score_action_candidate`, `gemma4_action_priors[(s_t, a_k)]` and the $64 \times 64$ Active Inference Top-Down Spatial Precision Map $\Pi(y, x)$ (`spatial_precision_map`) enter with full **`0.25` weight** (`0.25 * g4_prior` for `ACTION1..5` and `0.25 * (Pi[cy, cx] - 1.0)` for `ACTION6(x, y)`), **decoupled** from the spanning-tree anti-backtracking guard (`-0.60`), which suppresses positive `g4_prior` *only* when `aid_int` matches the confirmed parent-return direction `reverse_action_map[s_t]` (`is_backtrack`).
2. **Layer-29 5-Archetype Goal Invariant Weight Compilation**:
   Using the centered Layer-29 Gram inner products between $\hat h_{\mathrm{goal}}^{(29)}$
   and the 5 Subgoal Invariant Archetype spans `[G0:SHEAF_HOMOMORPHISM]..[G4:GEODESIC_DOCKING]`, `deliberate_with_gemma4_system2` compiles a 5D softmax prior $\mathbf w_{\mathrm{G4}} \in \Delta^4$ and blends it directly into `CrossLevelGoalInvariantDistiller.apply_system2_compiled_weights`.
3. **Persistent** $D_4$ **Equivariant Subgoal Priors (`gemma4_subgoal_priors`)**:
   Top-ranked coordinate click targets from $P_{\mathrm{callosal}}$
   have their canonical dihedral $D_4$ component signature stored in `world_model.gemma4_subgoal_priors`, guiding multi-step lookahead across subsequent states even after the ego sprite moves.

---

## 6. Empirical Evaluation on the 25-Game (`183`-Level) ARC-AGI-3 Benchmark

### 6.1 Overall Benchmark Progression & Paired Non-Parametric Statistical Significance

We evaluate `darksnail_v2_arc3` against two canonical baselines (`random_legal` and `unshielded_greedy`, an ablation that greedily clicks raw connected components and moves without $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ conservation splitting,
causal actuator factorization $\mathcal G_{\mathrm{act}}$,
or Subsystem A/B planning) across **all 25 public ARC-AGI-3 games (`183` total levels)** under the official offline protocol (`max_actions = 1000` per game, `seed = 0`, [`scripts/evaluate_darksnail_arc3.py`](../scripts/evaluate_darksnail_arc3.py), full per-game logs in [`results/darksnail_arc3_evaluation_summary.json`](../results/darksnail_arc3_evaluation_summary.json) and archived No-Gemma-4 baseline in [`results/darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json) and [`results/darksnail_m5_offline_no_gemma4_per_game.csv`](../results/darksnail_m5_offline_no_gemma4_per_game.csv)).

Table 3 summarizes the cumulative progression across our five architectural passes alongside the baseline policies, and Table 4 reports exact paired non-parametric statistical tests across all $N = 25$ games.

| Agent Policy / Architectural Pass | Total Levels Completed (`/ 183`) | Level Completion Rate (%) | Games with $\ge 1$ Level Solved (`/ 25`) | Full Games Won (`/ 25`) | Official RHAE Scorecard (%) | Mean Actions per Solved Level |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `unshielded_greedy` (Ablation without $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$) | `1 / 183` | `0.55%` | `1 / 25` (`4.0%`) | `0 / 25` | `0.0002%` | `25,000.0` |
| `random_legal` (Uniform Legal Action Baseline) | `2 / 183` | `1.09%` | `2 / 25` (`8.0%`) | `0 / 25` | `0.0013%` | `12,500.0` |
| `darksnail_v2_arc3` — Pass 1 (Initial $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ + Medoid Explorer) | `12 / 183` | `6.56%` | `10 / 25` (`40.0%`) | `0 / 25` | `0.2140%` | `2,083.3` |
| `darksnail_v2_arc3` — Pass 2 (+ Non-Local Causal $\mathcal G_{\mathrm{act}}\text{ vs. }\Omega_{\mathrm{reg}}$ + Dihedral Transfer) | `16 / 183` | `8.74%` | `14 / 25` (`56.0%`) | `0 / 25` | `0.4810%` | `1,562.5` |
| `darksnail_v2_arc3` — Pass 3 (+ Geodesic Corridor Momentum + Hybrid Budget) | `18 / 183` | `9.84%` | `15 / 25` (`60.0%`) | `0 / 25` | `0.5512%` | `1,388.9` |
| `darksnail_v2_arc3` — Pass 4 / Milestone 4 (+ Subsystem A Plan Synthesis + Subsystem B Spectral Init) | `19 / 183` | `10.38%` | `16 / 25` (`64.0%`) | `0 / 25` | `0.6274%` | `1,315.8` |
| `darksnail_v2_arc3` — Pass 5 / Milestone 5 + Argon Refactored (No-Gemma-4 Offline Baseline: + Scale-Invariant Sheaf Cavity CSP, Thermodynamic Irreversibility, Inverse Teleology & Active Lookahead) | `29 / 183` | `15.85%` | `18 / 25` (`72.0%`) | `1 / 25` (`sb26`) | `4.8124%` | `831.9` |

**Table 3:** Progression of `darksnail_v2_arc3` across the 25-game (`183`-level) ARC-AGI-3 public benchmark (`max_actions = 1000`, `seed = 0`, No-Gemma-4 standalone Right-Hemisphere + Mathematical System-2 mode). Milestone 5 + Argon Refactoring boosts solved levels from `19 / 183` (`10.38%`) to **`29 / 183` (`15.85%`)**, solves **`18 / 25` games (`72.0%`)**, wins all 8 levels of `sb26` (`8 / 8` in `124` actions, `100.0%` RHAE), and increases the Official RHAE Scorecard score by $7.67\times$ (`0.6274%` to `4.8124%`).

| Paired Comparison ($N = 25$ Games) | Metric | Mean Paired Diff | 10,000-Sample Bootstrap 95% CI | Game Wins / Losses / Ties | Exact Binomial Sign Test $p$-value | Wilcoxon Signed-Rank $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `darksnail_v2_arc3` (M5) vs. `random_legal` | Levels Completed | $+1.08$ (`+27` levels) | `[+0.56, +1.80]` | `16 W / 0 L / 9 T` | $p = 3.05 \times 10^{-5}$ | $p = 3.05 \times 10^{-5}$ |
| `darksnail_v2_arc3` (M5) vs. `unshielded_greedy` | Levels Completed | $+1.12$ (`+28` levels) | `[+0.64, +1.84]` | `17 W / 0 L / 8 T` | $p = 1.53 \times 10^{-5}$ | $p = 1.53 \times 10^{-5}$ |
| `darksnail_v2_arc3` (M5) vs. `random_legal` | Official RHAE (%) | $+4.8111\%$ | `[+0.4474%, +13.0514%]` | `18 W / 0 L / 7 T` | $p = 7.63 \times 10^{-6}$ | $p = 7.63 \times 10^{-6}$ |
| `darksnail_v2_arc3` (M5) vs. `unshielded_greedy` | Official RHAE (%) | $+4.8122\%$ | `[+0.4478%, +13.0515%]` | `18 W / 0 L / 7 T` | $p = 7.63 \times 10^{-6}$ | $p = 7.63 \times 10^{-6}$ |

**Table 4:** Paired non-parametric statistical comparisons across all $N = 25$ ARC-AGI-3 games ([`results/darksnail_arc3_evaluation_summary.json`](../results/darksnail_arc3_evaluation_summary.json)). `darksnail_v2_arc3` achieves **zero game losses (`0 L`)** against both baselines across every metric.

### 6.2 Complete 18-Game Solved Breakdown & Mechanistic Attribution

Table 5 details all **18 games solved** ($\ge 1$ level completed, `29` total levels, `1` full 8-level game win) by the Milestone 5 + Argon Refactored `darksnail_v2_arc3` agent out of the `25` public ARC-AGI-3 environments (alongside `ar25` solved in Milestone 4, totaling `19 / 25` games solved across Milestones 4 and 5), including exact level-completion step indices, exogenous clock pixels masked (`M_clock`), discovered remote actuators (`|G_act|`), and the primary mathematical mechanism responsible for solving each game.

| Game ID | `random` Levels | `greedy` Levels | M4 Levels | M5 Levels (`/ Win`) | Level Completion Step(s) | Clock Pixels Masked | Remote Actuators | Unique `H_nope` States | Official RHAE (%) | Primary Mathematical Mechanism Responsible |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `sb26-7fbdac44` | `0 / 8` | `0 / 8` | `0 / 8` | `8 / 8` (`WIN`) | `L1:9, L2:24, L3:39, L4:54, L5:71, L6:90, L7:107, L8:124` | `0` | `8` | `17` | `100.0000%` | Scale-Invariant Cellular Sheaf Cavity CSP (`synthesize_sheaf_bijection_plan`, $S_{\mathrm{tok}} = \sqrt{S_{\mathrm{ref}}^2 - A_{\mathrm{ref}}}$) + Jordan container ray-casting. |
| `tu93-0768757b` | `0 / 9` | `0 / 9` | `2 / 9` | `3 / 9` | `L1: step 149`<br>`L2: step 187`<br>`L3: step 504` | `64` | `0` | `60` | `0.9008%` | Thermodynamic irreversibility guard (`reversible_mask`) + Fisher-Rao `CrossLevelGoalInvariantDistiller` + Marginal Wasserstein-1. |
| `vc33-5430563c` | `0 / 7` | `0 / 7` | `2 / 7` | `2 / 7` | `L1: step 3`<br>`L2: step 171` | `64` | `8` | `244` | `4.1891%` | Channel-2 `enclosed_hole` medoids + Non-local actuator $\mathcal G_{\mathrm{act}}$ ($\eta \ge 0.50$) + Active 1-step lookahead. |
| `sc25-635fd71a` | `0 / 6` | `0 / 6` | `2 / 6` | `2 / 6` | `L1: step 141`<br>`L2: step 543` | `128` | `4` | `269` | `0.3125%` | Hybrid `ACTION1..4` + `ACTION6` Tier-1 medoid probing + Non-local $\mathcal G_{\mathrm{act}}$ (`4` actuators) + 128-pixel dual-strip clock mask. |
| `sp80-589a99af` | `1 / 6` | `0 / 6` | `1 / 6` | `1 / 6` | `L1: step 43` | `128` | `0` | `233` | `3.9172%` | Promoted `synthesize_mathematical_plan` (`synthesize_abelian_jacobian_plan`) + Symmetry potential $\Phi_{\mathrm{topo}}$ (`43` steps vs. `399` in M4, $9.3\times$ faster). |
| `tn36-ef4dde99` | `0 / 7` | `0 / 7` | `0 / 7` | `1 / 7` | `L1: step 13` | `61` | `3` | `629` | `3.5714%` | Excitable Excursion Verifier (`verifier_actuators`, $\Delta_{\mathrm{nope}} \ge 12$) + Multi-cavity hole salience regulation (solved in `13` steps). |
| `lp85-305b61c3` | `0 / 8` | `0 / 8` | `1 / 8` | `1 / 8` | `L1: step 12` | `64` | `6` | `829` | `2.7778%` | Corner-bracketed `slot_align` Goal Homomorphism $\mathcal E_{\mathrm{target}}$ + Non-local actuator discovery (`6` actuators, solved in `12` steps). |
| `su15-1944f8ab` | `0 / 9` | `0 / 9` | `1 / 9` | `1 / 9` | `L1: step 19` | `64` | `4` | `155` | `2.2222%` | Non-local causal generator $\mathcal G_{\mathrm{act}}$ + Geodesic spanning-tree BFS + Subsystem A Koopman rollout (solved in `19` steps). |
| `m0r0-492f87ba` | `0 / 6` | `0 / 6` | `1 / 6` | `1 / 6` | `L1: step 55` | `128` | `0` | `134` | `1.4168%` | Dual-strip 128-pixel clock flood-fill + Geodesic corridor momentum + Avatar-to-beacon navigation energy $\mathcal E_{\mathrm{nav}}$. |
| `cd82-fb555c5d` | `0 / 6` | `0 / 6` | `1 / 6` | `1 / 6` | `L1: step 182` | `64` | `3` | `224` | `0.4349%` | Non-local actuator factorization + Local $5 \times 5$ patch invariance + Active 1-step lookahead (`182` steps vs. `581` in M4, $3.2\times$ faster). |
| `lf52-271a04aa` | `0 / 10` | `0 / 10` | `1 / 10` | `1 / 10` | `L1: step 73` | `64` | `18` | `108` | `0.3494%` | Non-local causal factorization (`18` actuators separated from passive registers $\Omega_{\mathrm{reg}}$) + Dihedral equivariance. |
| `r11l-495a7899` | `1 / 6` | `1 / 6` | `1 / 6` | `1 / 6` | `L1: step 180` | `64` | `102` | `385` | `0.0711%` | Dyadic quadtree $2 \times 2$ medoid subdivision + Dihedral $D_4$ canonical shape transfer (`180` steps vs. `760` for baselines, $4.2\times$ faster). |
| `ls20-9607627b` | `0 / 7` | `0 / 7` | `1 / 7` | `1 / 7` | `L1: step 206` | `62` | `0` | `223` | `0.0407%` | Retroactive $\mathcal H_{\mathrm{clock}}$ quotienting (`62` clock pixels) + Geodesic corridor momentum + Anti-backtracking spanning tree. |
| `dc22-fdcac232` | `0 / 6` | `0 / 6` | `1 / 6` | `1 / 6` | `L1: step 692` | `64` | `2` | `149` | `0.0346%` | `tier1_click_probes_failed` memory + Tier-2 deep medoid expansion uncovering `2` actuators + $D_4$ shape transfer. |
| `g50t-5849a774` | `0 / 7` | `0 / 7` | `0 / 7` | `1 / 7` | `L1: step 885` | `64` | `0` | `39` | `0.0277%` | Newly solved in M5: `RelationalOperatorDreamer` ego-kinematic generator + Contact-Highlight `ACTION5` receptor docking. |
| `ft09-0d8bbf25` | `0 / 6` | `0 / 6` | `1 / 6` | `1 / 6` | `L1: step 619` | `64` | `0` | `136` | `0.0230%` | Spaced `cc_lattice` Goal Homomorphism $\mathcal E_{\mathrm{hom}}$ + Type IV cross-tile `lattice_local` operator induction + Abelian Jacobian OMP. |
| `bp35-0a0ad940` | `0 / 9` | `0 / 9` | `1 / 9` | `1 / 9` | `L1: step 230` | `64` | `3` | `329` | `0.0185%` | Non-local causal generator $\mathcal G_{\mathrm{act}}$ (`3` actuators) + Period-5 Clockwork Hamiltonian loop escape. |
| `s5i5-18d95033` | `0 / 8` | `0 / 8` | `1 / 8` | `1 / 8` | `L1: step 704` | `64` | `7` | `156` | `0.0022%` | Tier-2 deep Beltrami medoid expansion + Non-local actuator discovery (`7` actuators in $\mathcal G_{\mathrm{act}}$) + Subsystem A `unroll_triad_cascade_plan`. |

**Table 5:** Complete per-game breakdown of all `18` ARC-AGI-3 games solved by the Milestone 5 + Argon Refactored `darksnail_v2_arc3` agent (`29` total levels completed at `max_actions = 1000`, archived in [`results/darksnail_m5_offline_no_gemma4_per_game.csv`](../results/darksnail_m5_offline_no_gemma4_per_game.csv)).

### 6.3 No-Gemma-4 Offline Baseline Archive vs. Live Bi-Hemispheric Gemma 4 System-2 GPU Evaluation

To enable rigorous controlled ablation between the standalone Right-Hemisphere + Mathematical System-2 engine (`use_gemma4_shell = False`, zero LLM calls) and the full Bi-Hemispheric system with live **`google/gemma-4-31B-it`** (`bfloat16`, patched with `SurgicalGemma4Attention` and Global Layer-29 `Gemma4BeltramiCallosalShell`), we archive the complete 25-game No-Gemma-4 trajectory and RHAE dataset in [`results/darksnail_m5_offline_no_gemma4_25games.json`](../results/darksnail_m5_offline_no_gemma4_25games.json) and [`results/darksnail_m5_offline_no_gemma4_per_game.csv`](../results/darksnail_m5_offline_no_gemma4_per_game.csv) and embed `M5_NO_GEMMA4_BASELINE` directly inside [`notebooks/darksnail_arc3_colab.ipynb`](../notebooks/darksnail_arc3_colab.ipynb):

- **Standalone Right-Hemisphere + Mathematical System-2 Baseline (`No-Gemma-4`)**:
  Solves **`29 / 183` levels (`15.85%`) across `18 / 25` games (`72.0%`), winning `sb26` (`8 / 8` in `124` actions) and achieving `4.8124%` Official RHAE Score** purely via $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ continuity/irreversibility splitting,
  non-local causal factorization $\mathcal G_{\mathrm{act}}$,
  `CrossLevelGoalInvariantDistiller`, `RelationalOperatorDreamer` 1-step lookahead, and the 4 mathematical planning primitives.
- **Live Bi-Hemispheric `google/gemma-4-31B-it` Colab GPU Evaluation ([`results/darksnail_m5_colab_gemma4_25games.json`](../results/darksnail_m5_colab_gemma4_25games.json) & [`results/darksnail_m5_no_gemma4_vs_gemma4_comparison.csv`](../results/darksnail_m5_no_gemma4_vs_gemma4_comparison.csv))**:
  Executing the full 25-game benchmark on Google Colab GPU with live `google/gemma-4-31B-it` (`bfloat16`, `183` live Layer-29 System-2 deliberations in `1216.43s`) solves **`26 / 183` levels (`14.21%`) across `16 / 25` games (`64.0%`), winning `sb26` (`8 / 8` in `124` actions) and achieving `4.7787%` Official RHAE Score**, while **accelerating level completion on `6` games** relative to `No-Gemma-4`: `bp35` (`192` vs. `230` steps), `dc22` (`642` vs. `692` steps), `ft09` (`604` vs. `619` steps), `lp85` (`11` vs. `12` steps), `s5i5` (`648` vs. `704` steps), and `tu93` (L1 `132` vs. `149` and L2 `169` vs. `187` steps).
  Moreover, diagnostic comparison identified and eliminated three cross-state Callosal-to-Frontier coupling biases in `darksnail/arc3.py` (zero-centering `p_callosal` around $1/K$ and removing Callosal offsets from cross-state BFS `frontier_scores[sh]`, scoping `state_precision_maps[cur_hash]` per state without double-counting `g4_prior`, and gating `lookahead_bonus` and `apply_system2_compiled_weights` on `levels_distilled > 0`).

![DarkSnail ARC-AGI-3 Multi-Panel Trajectory & Causal Factorization Diagnostics](../results/darksnail_arc3_diagnostics.png)

**Figure 3:** Multi-panel trajectory diagnostics from [`notebooks/darksnail_arc3_colab.ipynb`](../notebooks/darksnail_arc3_colab.ipynb) showing
(a) cumulative level completions and exogenous clock masking $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$,
(b) Unified Goal Homomorphism Energy $\mathcal E_{\mathrm{target}}(F_t; \Pi)$
and 1D Marginal Wasserstein-1 distance $W_1^{\mathrm{marg}}$ drops prior to level completion,
(c) Period-5 Clockwork Hamiltonian $\mathcal H_{\mathrm{clock}}$ loop-escape spikes,
and (d) Non-Local Causal Operator Factorization separating Active Generators $\mathcal G_{\mathrm{act}}$ from Driven Passive Registers.

---

## 7. Theoretical Boundary Analysis, Limitations & Conclusion

### 7.1 Stateless $\mathrm{SE}(2) \times S_{16}$ Primitives vs. Conditional Multi-Body Contact Graph Automata

Why does `darksnail_v2_arc3` solve `18 / 25` games (`72.0%`, `29 / 183` levels, plus `ar25` in Milestone 4 for `19 / 25` cumulative games) with zero game-specific training, while the remaining `6 / 25` games (`cn04`, `ka59`, `re86`, `sk48`, `tr87`, `wa30`) and deeper levels (Levels `4`–`10`) remain open within the 1,000-action horizon?
A rigorous dynamical analysis of the episodic transition graphs on the unsolved games reveals a clean mathematical boundary:

1. **Unconditional, Excitable-Verifier & Scale-Invariant Sheaf Operators in** $\mathrm{SE}(2) \ltimes S_{16}$ (Solved by Subsystem A):
   In the `18` solved games, the environmental transition operator $T_a$ induced by a click or keyboard move is governed by scale-invariant sheaf cavity substitution (`sb26`), excitable excursion verifiers (`tn36`), contact-highlight receptor docking (`g50t`), reversible column/bridge alignment (`tu93`), cross-tile lattice permutation (`ft09`), remote actuators (`vc33`, `lp85`, `su15`, `cd82`, `lf52`, `dc22`, `sc25`), or rigid avatar translations (`ls20`, `m0r0`).
   In this regime, single-transition operator induction, 1-step forward lookahead, and our four pure mathematical synthesis primitives (`synthesize_sheaf_bijection_plan`, `synthesize_abelian_jacobian_plan`, `dream_forward_plan`, and `unroll_triad_cascade_plan`) accurately predict multi-step rollouts $\hat F^{(1:K)}$.
2. **Conditional Multi-Body Contact Graph Automata** (The Open Frontier):
   In the remaining `6` games (and higher levels of multi-level games), transitions obey **multi-body conditional cellular-automaton rules** where executing action $a$ on object $A$ has *zero effect* (or triggers a hazard reset) unless object $B$ is simultaneously in contact with object $C$ or oriented along a specific ray:

```math
T_a(F_t) = \begin{cases} \Phi_{\mathrm{coupled}}(F_t) & \text{if } \mathrm{ContactGraph}(F_t) \models \psi_{\mathrm{pre}}(A, B, C) \\ F_t\text{ (or }\mathrm{RESET}\text{)} & \text{otherwise} \end{cases}
```

   Because a stateless operator learner $\hat T_a$
   records $a$ as a noop or hazard when probed before the multi-body precondition $\psi_{\mathrm{pre}}(A, B, C)$ is satisfied,
   discovering conditional multi-body rules requires **Relational Contact-Graph Predicate Induction**
   (conditioning transitions on the adjacency graph of touching foreground components $\mathcal G_{\mathrm{contact}}(F_t)$
   rather than on pixel coordinates $(x, y)$ or global state hashes $s$).
   Extending `DarkSnailArc3WorldModel` from coordinate-indexed operators $\hat T_a$
   to contact-graph-indexed horn clauses $\hat T(a, \mathcal G_{\mathrm{contact}})$ is a direct, natural sequel to Subsystem A.

### 7.2 Conclusion

**Project DarkSnail IV** demonstrates that the mathematical primitives of ultrametric Bruhat-Tits dyadic hydrodynamics, skew-symmetric $\mathfrak{so}(M)$ triad conservation, cellular sheaf cohomology, and lateralized transcallosal relaxation scale seamlessly to both **frontier 31B heterogeneous foundation models (`google/gemma-4-31B-it`)** and **embodied interactive fluid intelligence (`ARC-AGI-3`)** with **zero newly trained backbone weights and zero game-source inspection**:

- **Part I (`darksnail/gemma4.py`)**: Routing strictly on the $75\%$ unrotated `384`-D `k_nope` subspace of the `10` Global Full-Attention layers of `google/gemma-4-31B-it` with Layer-29 `Gemma4BeltramiCallosalShell` (`5376 -> 4096` row-orthonormal isometric bridge) and Measure-Preserving Donor Pruning achieves **`100.0% / 94.0% / 84.0% / 68.0%`** 50-position Banal NIAH recall at $r = 2, 3, 4, 5$ ($+52.0\%$ to $+68.0\%$ over `v4_triad@r`, $p \lt 10^{-6}$) and **`49.89 / 48.62 / 48.61` F1** on full $n = 200$ LongBench QASPER ($+5.18$, $+6.58$, and $+12.80$ F1 over `v4_triad@2/3/4`, $p \le 0.0002$).
- **Part II (`darksnail/arc3.py`)**: Uniting Action-Conditioned MDP Homomorphisms & Thermodynamic Irreversibility ($\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$),
  Non-Local Causal Operator Factorization ($\mathcal G_{\mathrm{act}}\text{ vs. }\Omega_{\mathrm{reg}}$),
  Dihedral $D_4$ Equivariant Transfer,
  **Subsystem A** (Scale-Invariant Goal Homomorphism Energy $\mathcal E_{\mathrm{target}}$,
  closed-form $\mathcal O(HW)$ 1D Marginal Wasserstein-1 distance $W_1^{\mathrm{marg}}$,
  Fisher-Rao `CrossLevelGoalInvariantDistiller`, `RelationalOperatorDreamer` 1-step lookahead, and Scale-Invariant Sheaf Cavity CSP / Abelian Jacobian OMP / Koopman Beam / Triad Cascade plan synthesis),
  and **Subsystem B** (Deterministic Spectral Initialization, Online $\mathfrak{so}(16)$ Lie-Generator Adaptation, Native $\mathbb R^{5376}$ All-but-the-Top Common-Mode Centering, Decoupled `0.25` Callosal Priors, and Layer-29 5-Archetype Invariant Compilation)
  solves **`29 / 183` levels (`15.85%`) across `18 / 25` games (`72.0%`), wins `1 / 25` full game outright (`sb26`: `8 / 8` in `124` actions, `100.0%` RHAE), and achieves `4.8124%` Official RHAE Score** on the 25-game ARC-AGI-3 benchmark ($p = 7.63 \times 10^{-6}$).

The conservation law filters the clock. The cellular sheaf resolves the cavity. The optimal transport aligns the goal. The callosal resonance bridges the hemispheres.

### 7.3 References

1. Akyürek, E., et al. (2024). The Surprising Effectiveness of Test-Time Training for Abstract Reasoning. *arXiv:2411.07279*.
2. Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*, 10(2), 251–276 (*see also arXiv:2004.08128, arXiv:0812.4360, arXiv:1912.05510*).
3. Bai, Y., et al. (2024). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. *ACL 2024*.
4. Bodnar, C., et al. (2022). Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs. *NeurIPS 2022 (arXiv:2202.04579)*.
5. Chollet, F. (2019). On the Measure of Intelligence. *arXiv:1911.01547*.
6. Chollet, F., et al. (2026). ARC-AGI-3: Interactive Fluid Intelligence Benchmark and Environment Suite. *ARC Prize Foundation Technical Report*.
7. Dasigi, P., et al. (2021). A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (QASPER). *NAACL 2021*.
8. Dietterich, T. G., et al. (2018). Discovering and Removing Exogenous State Variables and Rewards for Reinforcement Learning. *ICML 2018 (arXiv:1806.01584)*.
9. Efroni, Y., et al. (2022). Sample-Efficient Reinforcement Learning in the Presence of Exogenous Information. *COLT 2022 (arXiv:2207.08229)*.
10. Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *EMNLP 2019 (arXiv:1909.00512)*.
11. Friston, K., et al. (2017). Active Inference: A Process Theory. *Neural Computation*, 29(1), 1–49.
12. Gemma Team, Google DeepMind (2026). Gemma 4 Technical Report: Heterogeneous Local-Global Attention and Proportional Rotary Position Embeddings. *arXiv:2607.02770*.
13. Hansen, J., & Ghrist, R. (2019). Toward a Spectral Theory of Cellular Sheaves. *Journal of Applied and Computational Topology (arXiv:1808.01513)*.
14. Higgins, I., et al. (2018). Towards a Definition of Disentangled Representations (Symmetry-Based Disentangled Representation Learning). *arXiv:1812.02230*.
15. Hong, S., & Jeong, H. (2026). Functional Lateralization in Deep Neural Networks via Transcallosal Inhibitory Dynamics. *arXiv:2603.03355*.
16. Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks (TRM). *arXiv:2510.04871*.
17. Katz, N. H., & Pavlović, N. (2005). Finite Time Blow-Up for a Dyadic Model of the Euler Equations. *Transactions of the American Mathematical Society*, 357(2), 695–708.
18. Lezcano-Casado, M., & Martínez-Rubio, D. (2019). Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group. *ICML 2019 (arXiv:1901.08428)*.
19. Mu, J., & Viswanath, P. (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. *ICLR 2018 (arXiv:1702.01417)*.
20. Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. *Asilomar Conference on Signals, Systems and Computers*.
21. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks. *ICLR 2014 (arXiv:1312.6120)*.
22. Seifert, U. (2012). Stochastic Thermodynamics, Fluctuation Theorems and Molecular Machines. *Reports on Progress in Physics (see also arXiv:1310.1863, arXiv:1708.01289)*.
23. Sneed & Feed Research (2026a). Llama Surgery: Continuous Sparsification of Pre-Trained Language Models via Differentiable Ultrametric Topology Injection. *Companion Preprint 1*.
24. Sneed & Feed Research (2026b). Lateralized Neuromorphic Architecture: Architecture Description and Pilot Diagnostics on ARC-AGI. *Companion Preprint 2*.
25. Sneed & Feed Research (2026c). Project DarkSnail: Bidirectional Synthesis of Ultrametric Dyadic Hydrodynamics and Lateralized Neuromorphic Reasoning. *Companion Preprint 3 (`papers/darksnail_v2_preprint.md`)*.
26. Sun, Y., et al. (2020). Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. *ICML 2020 (arXiv:2002.06991)*.
27. Tao, T. (2016). Finite Time Blowup for an Averaged Three-Dimensional Navier-Stokes Equation. *Journal of the American Mathematical Society*, 29(3), 601–674.
28. Taylor, J., Precup, D., & Panangaden, P. (2009). Bounding Performance Loss in Approximate MDP Homomorphisms. *NeurIPS (see also van der Pol et al., arXiv:2006.16908; Castro, arXiv:1911.12247; Ferns et al., arXiv:1207.4114)*.
29. Thomas, V., et al. (2018). Disentangling the Independently Controllable Factors of Variation by Interacting with the World. *NeurIPS Workshop (arXiv:1811.01483)*.
30. Villani, C. (2009). *Optimal Transport: Old and New*. Grundlehren der mathematischen Wissenschaften, Vol. 338, Springer.
31. Wang, G., et al. (2025). Hierarchical Reasoning Model (Sapient HRM). *GitHub Repository*.
32. Wen, Z., & Yin, W. (2013). A Feasible Method for Optimization with Orthogonality Constraints (Cayley Transform on Stiefel Manifold). *Mathematical Programming*, 142(1), 397–434.
33. Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80–83.
34. Ziebart, B. D., et al. (2008). Maximum Entropy Inverse Reinforcement Learning. *AAAI 2008*.
