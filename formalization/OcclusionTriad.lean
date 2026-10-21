import Formalization.Analysis.TriadConservation
import Formalization.Analysis.RoPECoherence
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

/-!
# Project DarkSnail: Skew-Symmetric Color-Occlusion Triad & 2D Axial RoPE Coherence

This module formalizes the core mathematical synthesis of **Project DarkSnail**
(System 2 Ultrametric & Continuous-Time Triad Reasoning on ARC-AGI):

1. **Skew-Symmetric Color-Occlusion Antisymmetry (`skew_triad_swap_neg`)**:
   For the skew-symmetrized color-occlusion interaction tensor
   $$S(W, i, c_1, c_2) = W(i, c_1, c_2) - W(i, c_2, c_1),$$
   swapping competing foreground colors $c_1 \leftrightarrow c_2$ strictly negates the interaction
   ($S(W, i, c_2, c_1) = -S(W, i, c_1, c_2)$) and self-interaction vanishes identically
   ($S(W, i, c, c) = 0$).

2. **Exact Tie-Breaking of Symmetric Readout Collisions (`occlusion_triad_breaks_symmetric_tie`)**:
   Whereas any additive or symmetric readout $\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)$
   cannot distinguish which of two colliding foreground colors $c_1, c_2$ occludes the other at a
   shared pixel, the DarkSnail bilinear triad readout shift
   $$\ell_{\mathrm{post}}(k) = \ell_{\mathrm{sym}}(k) + B_W(u, v)_k$$
   produces an exact antisymmetric margin
   $$\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2)
     = 2 \beta \alpha \bigl(W(i_0, c_2, c_1) - W(i_0, c_1, c_2)\bigr),$$
   strictly breaking the 2-color occlusion tie whenever $W(i_0, c_2, c_1) > W(i_0, c_1, c_2)$.

3. **Zero-Energy-Explosion Orthogonality (`darksnail_readout_energy_conservation`)**:
   By `triad_pointwise_orthogonal`, the DarkSnail triad logit shift is strictly orthogonal to the
   active color probability state $v$ ($\langle v, B_W(u, v) \rangle = 0$), and the Slow-Fast
   ($z_H \to z_L$) recurrent triad pump $\dot{z}_L = B_{W_{\mathrm{rec}}}(z_H, z_L)$ has zero
   radial energy growth ($\langle z_L, \dot{z}_L \rangle = 0$).

4. **Step-0 System 1 Homotopy Equivalence (`darksnail_step0_system1_homotopy`)**:
   At the Step-0 initialization ($\Delta_{\mathrm{sym}} = 0$, $W = 0$), the full DarkSnail readout
   reduces identically to the pretrained System 1 baseline logit $\ell_{\mathrm{S1}}(k)$.

5. **2D Axial RoPE Medoid Invariance & Patch-Averaging Attenuation
   (`rope2d_medoid_relative_invariance`, `rope2d_arithmetic_norm_lt`)**:
   In 2D grid coordinates $(y, x)$, selecting a 2D medoid key preserves exact relative 2D
   rotational displacement $(y^* - y_q, x^* - x_q)$ across both vertical and horizontal axes,
   whereas arithmetic patch averaging across distinct 2D grid phases strictly attenuates the
   squared 2D key norm.

6. **Multi-Step Unrolled Slow-Fast Spherical Invariance & Step-0 Homotopy
   (`darksnail_multistep_spherical_invariance`, `darksnail_multistep_step0_homotopy`)**:
   Across any $T$-step unrolled Slow-Fast spherical triad cascade, the terminal retracted state
   preserves the exact initial squared Euclidean norm and reduces identically to the unmodified
   System 1 trajectory at $W_{\mathrm{rec}} = 0$.

7. **SO(6) Kelvin-Mandel Stiffness Rotor Conservation & Caliper Morse Shielding
   (`kelvinMandelDipRotor_orthonormal`, `elasticEnergy6_rotateStiffness6`,
   `so6_stiffness_preserves_pos_def`, `so6_stiffness_trace_conservation`,
   `morseImpulse_iff_strict_extremum`, `caliper_morse_rejects_washout_step`,
   `caliper_morse_shields_thin_stringer`)**:
   Connecting Project DarkSnail to 3D anisotropic subsurface velocity modeling (`g3vm`),
   6D Kelvin-Mandel orthogonal stiffness rotors preserve elastic strain energy, strict
   positive-definiteness (coercivity), and tensor trace across tilted strata, while the 3-point
   discrete Morse critical-point impulse filter strictly rejects single-sided caliper washout steps
   and monotone compaction ramps while recovering thin-bed stringer spikes at 100% amplitude.

8. **3D Ideal MHD / Stellarator Symmetry-Breaking Isomorphism
   (`cross3_eq_triadOperator`, `mhd_euler_force_balance_bijection`,
   `mhd_flux_surface_confinement_iff`, `stellarator_parity_breaks_tie_and_conserves_flux`,
   `stellarator_zero_torsion_homotopy_iff`)**:
   Establishes the exact isomorphism between the $C = 3$ Levi-Civita instance of `triadOperator`
   and the 3D cross product, proving the involutive force-balance bijection between 3D ideal MHD
   ($\mathbf{J} \times \mathbf{B} = \nabla p$) and steady Euler/Beltrami hydrodynamics
   ($\boldsymbol{\omega} \times \mathbf{u} = -\nabla H$), tangent magnetic flux-surface confinement
   ($\mathbf{B} \cdot \nabla\psi = 0 \iff \langle v, B_W(u, v) \rangle = 0$), non-stellarator-symmetric
   parity tie-breaking across resonant Fourier modes, and exact zero-torsion homotopy
   ($\tau_0 = 0 \iff \delta z = 0$).
-/

namespace Ultrametric.Analysis.OcclusionTriad

open scoped BigOperators
open Real Finset

/-- Explicit-dimension wrapper around `Formalization.Analysis.triadOperator`:
$$B_W(u, v)_k = \sum_{i, j \in \mathrm{Fin}\,C} u_i v_j (W_{i,j,k} - W_{i,k,j}).$$ -/
def triadOperator (C : ℕ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  Formalization.Analysis.triadOperator W u v k

/-- Skew-symmetrized color-occlusion interaction tensor
$$S(W, i, c_1, c_2) = W(i, c_1, c_2) - W(i, c_2, c_1)$$
representing the directed occlusion dominance of color $c_2$ over $c_1$ under structural mode $i$. -/
def skewTriad {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c₁ c₂ : Fin C) : ℝ :=
  W i c₁ c₂ - W i c₂ c₁

/-! ### Theorem 1: Skew-Symmetric Color-Occlusion Antisymmetry -/

/-- **Theorem 1 (`skew_triad_swap_neg`)**:
For the skew-symmetrized color-occlusion interaction $S(W, i, c_1, c_2) = W(i, c_1, c_2) - W(i, c_2, c_1)$,
swapping $c_1 \leftrightarrow c_2$ negates the interaction:
$$S(W, i, c_2, c_1) = -S(W, i, c_1, c_2),$$
and self-interaction vanishes identically:
$$S(W, i, c_1, c_1) = 0.$$ -/
theorem skew_triad_swap_neg {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c₁ c₂ : Fin C) :
    skewTriad W i c₂ c₁ = -skewTriad W i c₁ c₂ ∧ skewTriad W i c₁ c₁ = 0 := by
  unfold skewTriad; constructor <;> ring

/-- Self-occlusion interaction vanishes for any color $c$. -/
theorem skew_triad_self_zero {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c : Fin C) :
    skewTriad W i c c = 0 :=
  (skew_triad_swap_neg W i c c).2

/-! ### Theorem 2: Exact Tie-Breaking of Symmetric 2-Color Collisions -/

/-- DarkSnail post-readout logit combining a symmetric/additive baseline logit $\ell_{\mathrm{sym}}$
with the skew-symmetric bilinear triad shift $B_W(u, v)$. -/
def postLogit (C : ℕ) (ℓ_sym : Fin C → ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_sym k + triadOperator C W u v k

/-- Helper lemma: evaluation of `triadOperator` under a single active structural mode $u(i_0) = \beta$
and a 2-color collision state $v(c_1) = v(c_2) = \alpha$ (with $c_1 \neq c_2$). -/
lemma triadOperator_two_color_eval {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ k : Fin C) (α β : ℝ)
    (hc : c₁ ≠ c₂)
    (hu_i0 : u i₀ = β)
    (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α)
    (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    triadOperator C W u v k =
      β * α * (W i₀ c₁ k - W i₀ k c₁) + β * α * (W i₀ c₂ k - W i₀ k c₂) := by
  unfold triadOperator Formalization.Analysis.triadOperator
  rw [Finset.sum_eq_single i₀]
  · rw [← Finset.add_sum_erase _ _ (Finset.mem_univ c₁),
        ← Finset.add_sum_erase _ _ (Finset.mem_erase.mpr ⟨hc.symm, Finset.mem_univ c₂⟩),
        Finset.sum_eq_zero, add_zero, hu_i0, hv_c1, hv_c2]
    intro j hj
    simp only [Finset.mem_erase] at hj
    simp [hv_other j hj.2.1 hj.1]
  · exact fun i _ hi => Finset.sum_eq_zero fun j _ => by simp [hu_other i hi]
  · exact fun h => (h (Finset.mem_univ i₀)).elim

/-- **Theorem 2 (`occlusion_triad_breaks_symmetric_tie`)**:
Let $\ell_{\mathrm{sym}} : \mathrm{Fin}\,C \to \mathbb{R}$ be any symmetric/additive readout logit vector
that ties on two competing foreground colors $c_1, c_2$ ($\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)$).
In a 2-color collision state $v(c_1) = v(c_2) = \alpha > 0$ (and $v(c) = 0$ for $c \notin \{c_1, c_2\}$)
under a single structural mode $u(i_0) = \beta > 0$ (and $u(i) = 0$ for $i \neq i_0$), the DarkSnail
post-readout logit $\ell_{\mathrm{post}}(k) = \ell_{\mathrm{sym}}(k) + \mathrm{triadOperator}\,C\,W\,u\,v\,k$
satisfies:
$$\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2 \cdot \beta \cdot \alpha \cdot (W(i_0, c_2, c_1) - W(i_0, c_1, c_2)),$$
and whenever $W(i_0, c_2, c_1) > W(i_0, c_1, c_2)$, we strictly have
$$\ell_{\mathrm{post}}(c_1) > \ell_{\mathrm{post}}(c_2).$$ -/
theorem occlusion_triad_breaks_symmetric_tie {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ : Fin C) (α β : ℝ)
    (hα : 0 < α) (hβ : 0 < β)
    (h_tie : ℓ_sym c₁ = ℓ_sym c₂)
    (hu_i0 : u i₀ = β)
    (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α)
    (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    (postLogit C ℓ_sym W u v c₁ - postLogit C ℓ_sym W u v c₂ =
      2 * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂)) ∧
    (W i₀ c₂ c₁ > W i₀ c₁ c₂ →
      postLogit C ℓ_sym W u v c₁ > postLogit C ℓ_sym W u v c₂) := by
  have h_diff :
      postLogit C ℓ_sym W u v c₁ - postLogit C ℓ_sym W u v c₂ =
        2 * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂) := by
    by_cases hc : c₁ = c₂
    · subst hc; ring
    · unfold postLogit
      rw [triadOperator_two_color_eval W u v i₀ c₁ c₂ c₁ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other,
          triadOperator_two_color_eval W u v i₀ c₁ c₂ c₂ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other,
          h_tie]
      ring
  exact ⟨h_diff, fun hW => by nlinarith [mul_pos hβ hα]⟩

/-! ### Theorem 3: DarkSnail Readout & Recurrent Triad Energy Conservation -/

/-- **Theorem 3 (`darksnail_readout_energy_conservation`)**:
For any structural vector $u$, color state $v$, and triad tensor $W$, the inner product of $v$
with the DarkSnail triad logit shift vanishes identically:
$$\sum_{k \in \mathrm{Fin}\,C} v(k) \cdot (\mathrm{triadOperator}\,C\,W\,u\,v\,k) = 0,$$
and for the Slow-Fast ($z_H \to z_L$) recurrent terminal step
$\dot{z}_L = \mathrm{triadOperator}\,M\,W_{\mathrm{rec}}\,z_H\,z_L$, the radial energy growth rate
$\langle z_L, \dot{z}_L \rangle$ vanishes identically. -/
theorem darksnail_readout_energy_conservation {C M : ℕ}
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ)
    (W_rec : Fin M → Fin M → Fin M → ℝ) (z_H z_L : Fin M → ℝ) :
    (∑ k : Fin C, v k * triadOperator C W u v k) = 0 ∧
    Formalization.Analysis.shellInner z_L (triadOperator M W_rec z_H z_L) = 0 :=
  ⟨Formalization.Analysis.triad_pointwise_orthogonal W u v,
   Formalization.Analysis.triad_pointwise_orthogonal W_rec z_H z_L⟩

/-! ### Theorem 4: Step-0 System 1 Homotopy Equivalence -/

/-- Full DarkSnail readout combining pretrained System 1 logits $\ell_{\mathrm{S1}}$,
symmetric residual shift $\Delta_{\mathrm{sym}}$, and skew-symmetric triad readout shift. -/
def darksnailReadout (C : ℕ) (ℓ_S1 Δ_sym : Fin C → ℝ)
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_S1 k + Δ_sym k + triadOperator C W u v k

/-- **Theorem 4 (`darksnail_step0_system1_homotopy`)**:
At the Step-0 initialization $\Delta_{\mathrm{sym}} = 0$ and $W = 0$ (corresponding to $\delta z = 0$),
the DarkSnail readout equals the pretrained System 1 logit identically for all colors $k$:
$$\ell_{\mathrm{DS}}(\ell_{\mathrm{S1}}, 0, 0, u, v, k) = \ell_{\mathrm{S1}}(k).$$ -/
theorem darksnail_step0_system1_homotopy {C : ℕ}
    (ℓ_S1 u v : Fin C → ℝ) (k : Fin C) :
    darksnailReadout C ℓ_S1 0 0 u v k = ℓ_S1 k := by
  simp [darksnailReadout, triadOperator, Formalization.Analysis.triadOperator]

/-! ### Theorem 5: 2D Axial RoPE Medoid Invariance & Patch-Averaging Attenuation -/

/-- 2D rotation on $\mathbb{R} \times \mathbb{R}$ by angle $\theta$. -/
noncomputable def rot2 (θ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (Real.cos θ * v.1 - Real.sin θ * v.2, Real.sin θ * v.1 + Real.cos θ * v.2)

/-- Standard inner product on $\mathbb{R} \times \mathbb{R}$. -/
def inner2 (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/-- Squared Euclidean norm on $\mathbb{R} \times \mathbb{R}$. -/
def normSq2 (v : ℝ × ℝ) : ℝ :=
  inner2 v v

/-- 2D axial RoPE embedding on $(\mathbb{R} \times \mathbb{R}) \times (\mathbb{R} \times \mathbb{R})$
applying vertical rotation $\theta_y$ to the $y$-slice and horizontal rotation $\theta_x$ to the $x$-slice. -/
noncomputable def rope2d (θ_y θ_x : ℝ) (q_y q_x : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (rot2 θ_y q_y, rot2 θ_x q_x)

/-- 2D axial inner product summing the vertical and horizontal slice inner products. -/
def inner2d (u v : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  inner2 u.1 v.1 + inner2 u.2 v.2

/-- Squared Euclidean norm on the 2D axial space $(\mathbb{R} \times \mathbb{R}) \times (\mathbb{R} \times \mathbb{R})$. -/
def normSq2d (v : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  normSq2 v.1 + normSq2 v.2

/-- Arithmetic midpoint of two 2D vectors. -/
noncomputable def avg2 (u v : ℝ × ℝ) : ℝ × ℝ :=
  ((u.1 + v.1) / 2, (u.2 + v.2) / 2)

/-- 2D arithmetic patch average of a key $(k_y, k_x)$ across two 2D grid phases $(\theta_{y,1}, \theta_{x,1})$
and $(\theta_{y,2}, \theta_{x,2})$. -/
noncomputable def rope2dPatchAvg (θ_y1 θ_x1 θ_y2 θ_x2 : ℝ) (k_y k_x : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (avg2 (rot2 θ_y1 k_y) (rot2 θ_y2 k_y), avg2 (rot2 θ_x1 k_x) (rot2 θ_x2 k_x))

/-- Single-axis rotational shift invariance for `rot2` and `inner2`. -/
lemma inner2_rot2_shift (α β : ℝ) (u v : ℝ × ℝ) :
    inner2 (rot2 α u) (rot2 β v) = inner2 u (rot2 (β - α) v) := by
  unfold inner2 rot2; rw [Real.cos_sub, Real.sin_sub]; ring

/-- **Theorem 5a (`rope2d_medoid_relative_invariance`)**:
For any query at 2D angular coordinate $(y_q, x_q)$ and 2D medoid key at coordinate $(y^*, x^*)$,
the 2D axial rotary inner product is invariant under global grid translation and depends purely on
the relative 2D displacement $(y^* - y_q, x^* - x_q)$:
$$\langle \mathrm{rope2d}(y_q, x_q, q_y, q_x), \mathrm{rope2d}(y^*, x^*, k_y, k_x) \rangle
  = \langle q_y, \mathcal{R}_{y^* - y_q} k_y \rangle + \langle q_x, \mathcal{R}_{x^* - x_q} k_x \rangle.$$ -/
theorem rope2d_medoid_relative_invariance
    (y_q x_q y_star x_star : ℝ) (q_y q_x k_y k_x : ℝ × ℝ) :
    inner2d (rope2d y_q x_q q_y q_x) (rope2d y_star x_star k_y k_x) =
      inner2 q_y (rot2 (y_star - y_q) k_y) + inner2 q_x (rot2 (x_star - x_q) k_x) := by
  unfold inner2d rope2d; rw [inner2_rot2_shift, inner2_rot2_shift]

/-- Exact squared norm identity for the 2-point arithmetic average of rotated 2D vectors:
$$\left\|\frac{\mathcal{R}_{\theta_1} k + \mathcal{R}_{\theta_2} k}{2}\right\|^2
  = \|k\|^2 \cdot \frac{1 + \cos(\theta_1 - \theta_2)}{2}.$$ -/
lemma normSq2_avg2_rot2 (θ₁ θ₂ : ℝ) (k : ℝ × ℝ) :
    normSq2 (avg2 (rot2 θ₁ k) (rot2 θ₂ k)) =
      normSq2 k * (1 + Real.cos (θ₁ - θ₂)) / 2 := by
  unfold normSq2 inner2 avg2 rot2
  rw [Real.cos_sub]
  linear_combination
    ((k.1 ^ 2 + k.2 ^ 2) / 4) * Real.cos_sq_add_sin_sq θ₁ +
    ((k.1 ^ 2 + k.2 ^ 2) / 4) * Real.cos_sq_add_sin_sq θ₂

/-- Non-negativity of `normSq2`. -/
lemma normSq2_nonneg (k : ℝ × ℝ) : 0 ≤ normSq2 k :=
  add_nonneg (mul_self_nonneg k.1) (mul_self_nonneg k.2)

/-- Non-expansion of single-axis arithmetic average squared norm. -/
lemma normSq2_avg2_rot2_le (θ₁ θ₂ : ℝ) (k : ℝ × ℝ) :
    normSq2 (avg2 (rot2 θ₁ k) (rot2 θ₂ k)) ≤ normSq2 k := by
  rw [normSq2_avg2_rot2]
  nlinarith [Real.cos_le_one (θ₁ - θ₂), normSq2_nonneg k]

/-- Strict attenuation of single-axis arithmetic average squared norm when $\cos(\theta_1 - \theta_2) < 1$
and $\|k\|^2 > 0$. -/
lemma normSq2_avg2_rot2_lt (θ₁ θ₂ : ℝ) (k : ℝ × ℝ)
    (hcos : Real.cos (θ₁ - θ₂) < 1) (hk : 0 < normSq2 k) :
    normSq2 (avg2 (rot2 θ₁ k) (rot2 θ₂ k)) < normSq2 k := by
  rw [normSq2_avg2_rot2]; nlinarith

/-- **Theorem 5b (`rope2d_arithmetic_norm_lt`)**:
Whenever either the vertical phase difference satisfies $\cos(\theta_{y,1} - \theta_{y,2}) < 1$ with
$\|k_y\|^2 > 0$, or the horizontal phase difference satisfies $\cos(\theta_{x,1} - \theta_{x,2}) < 1$
with $\|k_x\|^2 > 0$, 2D arithmetic patch averaging strictly attenuates the squared 2D key norm:
$$\|\mathrm{rope2dPatchAvg}(\theta_{y,1}, \theta_{x,1}, \theta_{y,2}, \theta_{x,2}, k_y, k_x)\|^2
  < \|(k_y, k_x)\|^2.$$ -/
theorem rope2d_arithmetic_norm_lt
    (θ_y1 θ_x1 θ_y2 θ_x2 : ℝ) (k_y k_x : ℝ × ℝ)
    (h_att : (Real.cos (θ_y1 - θ_y2) < 1 ∧ 0 < normSq2 k_y) ∨
             (Real.cos (θ_x1 - θ_x2) < 1 ∧ 0 < normSq2 k_x)) :
    normSq2d (rope2dPatchAvg θ_y1 θ_x1 θ_y2 θ_x2 k_y k_x) < normSq2d (k_y, k_x) := by
  unfold normSq2d rope2dPatchAvg
  rcases h_att with ⟨hcy, hky⟩ | ⟨hcx, hkx⟩
  · linarith [normSq2_avg2_rot2_lt θ_y1 θ_y2 k_y hcy hky, normSq2_avg2_rot2_le θ_x1 θ_x2 k_x]
  · linarith [normSq2_avg2_rot2_le θ_y1 θ_y2 k_y, normSq2_avg2_rot2_lt θ_x1 θ_x2 k_x hcx hkx]

/-! ### Theorem 6: Multi-Step Unrolled Slow-Fast Spherical Invariance & Step-0 Homotopy -/

/-- Scaled tangent triad step $\delta v_k = \gamma \cdot B_{W_{\mathrm{rec}}}(u, v)_k$. -/
def tangentTriadStep (M : ℕ) (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) (k : Fin M) : ℝ :=
  γ * triadOperator M W_rec u v k

/-- Spherical norm retraction $\tilde{v}_k = \frac{\|v\|_2}{\|v_{\mathrm{upd}}\|_2} v_{\mathrm{upd}, k}$. -/
noncomputable def sphericalRetract {M : ℕ} (v v_upd : Fin M → ℝ) (k : Fin M) : ℝ :=
  (Real.sqrt (Formalization.Analysis.shellEnergy v) /
   Real.sqrt (Formalization.Analysis.shellEnergy v_upd)) * v_upd k

/-- Single-step tangent triad update followed by spherical norm retraction:
$$\tilde{v}_L = \frac{\|v_L\|_2}{\|v_L + \delta v_L\|_2} (v_L + \delta v_L), \qquad
  \delta v_L = \gamma B_{W_{\mathrm{rec}}}(u, v_L).$$ -/
noncomputable def sphericalTriadStep (M : ℕ) (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) : Fin M → ℝ :=
  let δv := tangentTriadStep M γ W_rec u v
  sphericalRetract v (fun k => v k + δv k)

/-- $T$-step unrolled Slow-Fast spherical triad trajectory $\tilde{v}_L^{(t)}$. -/
noncomputable def multistepSphericalTriad (M : ℕ) (γ : ℕ → ℝ)
    (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ) : ℕ → (Fin M → ℝ)
  | 0 => v₀
  | t + 1 =>
      let v_t := multistepSphericalTriad M γ W_rec u v₀ t
      sphericalTriadStep M (γ t) (W_rec t) (u t) v_t

/-- Non-negativity of `shellEnergy`. -/
lemma shellEnergy_nonneg {M : ℕ} (x : Fin M → ℝ) :
    0 ≤ Formalization.Analysis.shellEnergy x := by
  unfold Formalization.Analysis.shellEnergy Formalization.Analysis.shellInner
  exact Finset.sum_nonneg fun k _ => mul_self_nonneg (x k)

/-- `shellEnergy x = 0` if and only if $x$ vanishes pointwise. -/
lemma shellEnergy_eq_zero_iff {M : ℕ} (x : Fin M → ℝ) :
    Formalization.Analysis.shellEnergy x = 0 ↔ ∀ k : Fin M, x k = 0 := by
  unfold Formalization.Analysis.shellEnergy Formalization.Analysis.shellInner
  simp [Finset.sum_eq_zero_iff_of_nonneg (fun k _ => mul_self_nonneg (x k))]

/-- Pointwise orthogonality $\langle v, \delta v \rangle = 0$ for the scaled tangent triad step. -/
lemma tangentTriadStep_orthogonal {M : ℕ} (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) :
    Formalization.Analysis.shellInner v (tangentTriadStep M γ W_rec u v) = 0 := by
  calc Formalization.Analysis.shellInner v (tangentTriadStep M γ W_rec u v)
    _ = γ * Formalization.Analysis.shellInner v (Formalization.Analysis.triadOperator W_rec u v) := by
        unfold Formalization.Analysis.shellInner tangentTriadStep triadOperator
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => by ring
    _ = 0 := by rw [Formalization.Analysis.triad_pointwise_orthogonal, mul_zero]

/-- Discrete-time Pythagorean identity under orthogonal tangent updates $\langle v, \delta v \rangle = 0$. -/
lemma shellEnergy_add_of_orthogonal {M : ℕ} (v δv : Fin M → ℝ)
    (h_orth : Formalization.Analysis.shellInner v δv = 0) :
    Formalization.Analysis.shellEnergy (fun k => v k + δv k) =
      Formalization.Analysis.shellEnergy v + Formalization.Analysis.shellEnergy δv := by
  have h_expand :
      Formalization.Analysis.shellEnergy (fun k => v k + δv k) =
        Formalization.Analysis.shellEnergy v +
        2 * Formalization.Analysis.shellInner v δv +
        Formalization.Analysis.shellEnergy δv := by
    unfold Formalization.Analysis.shellEnergy Formalization.Analysis.shellInner
    simp only [← Finset.sum_add_distrib, Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  linarith

/-- Quadratic homogeneity of `shellEnergy` under scalar multiplication. -/
lemma shellEnergy_scale {M : ℕ} (c : ℝ) (x : Fin M → ℝ) :
    Formalization.Analysis.shellEnergy (fun k => c * x k) =
      c ^ 2 * Formalization.Analysis.shellEnergy x := by
  unfold Formalization.Analysis.shellEnergy Formalization.Analysis.shellInner
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- Single-step spherical triad update preserves exact shell energy $\|\tilde{v}\|_2^2 = \|v\|_2^2$. -/
lemma sphericalTriadStep_preserves_energy {M : ℕ} (γ : ℝ)
    (W_rec : Fin M → Fin M → Fin M → ℝ) (u v : Fin M → ℝ) :
    Formalization.Analysis.shellEnergy (sphericalTriadStep M γ W_rec u v) =
      Formalization.Analysis.shellEnergy v := by
  unfold sphericalTriadStep sphericalRetract
  dsimp only
  rw [shellEnergy_scale]
  by_cases hv0 : Formalization.Analysis.shellEnergy v = 0
  · simp [hv0]
  · have hv_nonneg := shellEnergy_nonneg v
    have h_pyth := shellEnergy_add_of_orthogonal v _ (tangentTriadStep_orthogonal γ W_rec u v)
    have h_upd_pos :
        0 < Formalization.Analysis.shellEnergy (fun k => v k + tangentTriadStep M γ W_rec u v k) := by
      linarith [lt_of_le_of_ne hv_nonneg (Ne.symm hv0),
                shellEnergy_nonneg (tangentTriadStep M γ W_rec u v)]
    rw [div_pow, Real.sq_sqrt hv_nonneg, Real.sq_sqrt (le_of_lt h_upd_pos),
        div_mul_cancel₀ _ (ne_of_gt h_upd_pos)]

/-- **Theorem 6a (`darksnail_multistep_spherical_invariance`)**:
Across any $T$-step unrolled Slow-Fast spherical triad cascade $(v_L^{(t)})_{t=0}^T$,
the terminal retracted state preserves the exact initial squared Euclidean norm
$\|\tilde{v}_L^{(T)}\|_2^2 = \|v_L^{(0)}\|_2^2$, and at every intermediate step $t$,
the tangent triad perturbation $\delta v_L^{(t)}$ is strictly orthogonal to $v_L^{(t)}$:
$\langle v_L^{(t)}, \delta v_L^{(t)} \rangle = 0$. -/
theorem darksnail_multistep_spherical_invariance {M : ℕ}
    (γ : ℕ → ℝ) (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ) (T : ℕ) :
    Formalization.Analysis.shellEnergy (multistepSphericalTriad M γ W_rec u v₀ T) =
      Formalization.Analysis.shellEnergy v₀ ∧
    (∀ t : ℕ,
      let v_t := multistepSphericalTriad M γ W_rec u v₀ t
      let δv_t := tangentTriadStep M (γ t) (W_rec t) (u t) v_t
      Formalization.Analysis.shellInner v_t δv_t = 0) := by
  refine ⟨?_, fun t => tangentTriadStep_orthogonal (γ t) (W_rec t) (u t) _⟩
  induction T with
  | zero => rfl
  | succ t ih => simp [multistepSphericalTriad, sphericalTriadStep_preserves_energy, ih]

/-- At $W_{\mathrm{rec}} = 0$, the single-step spherical triad update reduces to the identity. -/
lemma sphericalTriadStep_zero {M : ℕ} (γ : ℝ) (u v : Fin M → ℝ) :
    sphericalTriadStep M γ 0 u v = v := by
  funext k
  have h_tan : (fun i => v i + tangentTriadStep M γ 0 u v i) = v := by
    funext i; simp [tangentTriadStep, triadOperator, Formalization.Analysis.triadOperator]
  unfold sphericalTriadStep sphericalRetract
  dsimp only
  rw [h_tan, congr_fun h_tan k]
  by_cases hv0 : Formalization.Analysis.shellEnergy v = 0
  · simp [hv0, (shellEnergy_eq_zero_iff v).mp hv0 k]
  · have hv_pos := lt_of_le_of_ne (shellEnergy_nonneg v) (Ne.symm hv0)
    rw [div_self (ne_of_gt (Real.sqrt_pos.mpr hv_pos)), one_mul]

/-- Unmodified System 1 recurrent carry trajectory $(z_H^{(t)}, z_L^{(t)})_{t=0}^T$. -/
def system1CarryTrajectory {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) : ℕ → (Fin M → ℝ) × (Fin M → ℝ)
  | 0 => z₀
  | t + 1 => step_S1 t (system1CarryTrajectory step_S1 z₀ t)

/-- DarkSnail $T$-step unrolled Slow-Fast recurrent carry trajectory coupling
$z_H^{(t)} \to z_L^{(t)}$ via `sphericalTriadStep` and latent perturbation $(\delta z_H^{(t)}, \delta z_L^{(t)})$. -/
noncomputable def darksnailCarryTrajectory {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (γ : ℕ → ℝ) (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (δz_H δz_L : ℕ → Fin M → ℝ)
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) : ℕ → (Fin M → ℝ) × (Fin M → ℝ)
  | 0 => z₀
  | t + 1 =>
      let z_t := darksnailCarryTrajectory step_S1 γ W_rec δz_H δz_L z₀ t
      let v_tilde := sphericalTriadStep M (γ t) (W_rec t) z_t.1 z_t.2
      step_S1 t (fun k => z_t.1 k + δz_H t k, fun k => v_tilde k + δz_L t k)

/-- **Theorem 6b (`darksnail_multistep_step0_homotopy`)**:
At the Step-0 initialization $W_{\mathrm{rec}} = 0$ and $\delta z = (0, 0)$, the $T$-step unrolled
spherical shell trajectory remains stationary at $v_0$, and the full $T$-step unrolled DarkSnail
recurrent carry trajectory reduces identically to the unmodified System 1 recurrent carry
trajectory $(z_H^{(t)}, z_L^{(t)})_{t=0}^T$ for all $T \in \mathbb{N}$. -/
theorem darksnail_multistep_step0_homotopy {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (γ : ℕ → ℝ) (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ)
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) (T : ℕ) :
    multistepSphericalTriad M γ 0 u v₀ T = v₀ ∧
    darksnailCarryTrajectory step_S1 γ 0 0 0 z₀ T =
      system1CarryTrajectory step_S1 z₀ T := by
  constructor <;> induction T with
  | zero => rfl
  | succ t ih =>
    simp [multistepSphericalTriad, darksnailCarryTrajectory, system1CarryTrajectory,
          ih, sphericalTriadStep_zero]

/-! ### Theorem 7 (Family 2.7): SO(6) Kelvin-Mandel Stiffness Rotor Conservation & Caliper Morse Shielding -/

/-- 6D Kelvin-Mandel quadratic elastic strain energy form
$$E_C(\varepsilon) = \varepsilon^\top C \varepsilon = \sum_{i, j \in \mathrm{Fin}\,6} \varepsilon_i C_{i,j} \varepsilon_j$$
(`g3vm/anisotropy.py`). -/
def elasticEnergy6 (C : Fin 6 → Fin 6 → ℝ) (ε : Fin 6 → ℝ) : ℝ :=
  ∑ i : Fin 6, ∑ j : Fin 6, ε i * C i j * ε j

/-- $6 \times 6$ matrix-vector product $(M x)_i = \sum_{j \in \mathrm{Fin}\,6} M_{i,j} x_j$. -/
def matVec6 (M : Fin 6 → Fin 6 → ℝ) (x : Fin 6 → ℝ) (i : Fin 6) : ℝ :=
  ∑ j : Fin 6, M i j * x j

/-- $6 \times 6$ matrix transpose $(M^\top)_{i,j} = M_{j,i}$. -/
def matTranspose6 (M : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ :=
  M j i

/-- Rotated $6 \times 6$ Kelvin-Mandel stiffness tensor under a 6D rotor $M$:
$$(M C M^\top)_{i,j} = \sum_{k, l \in \mathrm{Fin}\,6} M_{i,k} C_{k,l} M_{j,l}.$$ -/
def rotateStiffness6 (M C : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ :=
  ∑ k : Fin 6, ∑ l : Fin 6, M i k * C k l * M j l

/-- Trace of a $6 \times 6$ Kelvin-Mandel stiffness tensor $\mathrm{Tr}(C) = \sum_{i \in \mathrm{Fin}\,6} C_{i,i}$. -/
def trace6 (C : Fin 6 → Fin 6 → ℝ) : ℝ :=
  ∑ i : Fin 6, C i i

/-- 6D norm-preserving orthogonality predicate on Kelvin-Mandel strain space:
$$\|M^\top x\|_2^2 = \|x\|_2^2 \quad \forall x \in \mathbb{R}^6.$$ -/
def IsOrthogonal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ x : Fin 6 → ℝ,
    Formalization.Analysis.shellEnergy (matVec6 (matTranspose6 M) x) =
      Formalization.Analysis.shellEnergy x

/-- Column-orthonormality predicate $M^\top M = I_6$ ($\sum_{i \in \mathrm{Fin}\,6} M_{i,k} M_{i,l} = \delta_{k,l}$)
for $6 \times 6$ Kelvin-Mandel stiffness rotors. -/
def IsColOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ k l : Fin 6, (∑ i : Fin 6, M i k * M i l) = if k = l then 1 else 0

/-- Row-orthonormality predicate $M M^\top = I_6$ ($\sum_{k \in \mathrm{Fin}\,6} M_{i,k} M_{j,k} = \delta_{i,j}$)
for $6 \times 6$ Kelvin-Mandel stiffness rotors. -/
def IsRowOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ i j : Fin 6, (∑ k : Fin 6, M i k * M j k) = if i = j then 1 else 0

/-- Any row-orthonormal $6 \times 6$ matrix $M$ ($M M^\top = I_6$) preserves the 6D Euclidean
strain norm (`IsOrthogonal6 M`). -/
lemma isOrthogonal6_of_rowOrthonormal6 (M : Fin 6 → Fin 6 → ℝ)
    (hM : IsRowOrthonormal6 M) :
    IsOrthogonal6 M := by
  intro x
  unfold Formalization.Analysis.shellEnergy Formalization.Analysis.shellInner matVec6 matTranspose6
  calc (∑ k : Fin 6, (∑ i : Fin 6, M i k * x i) * (∑ j : Fin 6, M j k * x j))
    _ = ∑ i : Fin 6, ∑ j : Fin 6, x i * x j * (∑ k : Fin 6, M i k * M j k) := by
        simp only [Fin.sum_univ_six]; ring
    _ = ∑ k : Fin 6, x k * x k := by simp [hM _ _]

/-- Explicit planar bedding-dip Kelvin-Mandel stiffness rotor $M(R_y(\theta)) : \mathrm{Fin}\,6 \to \mathrm{Fin}\,6 \to \mathbb{R}$
for a dip tilt by angle $\theta$ in the $(x_1, x_3)$ plane (`build_kelvin_mandel_so6_rotor` in `g3vm/anisotropy.py`). -/
noncomputable def kelvinMandelDipRotor (θ : ℝ) (i j : Fin 6) : ℝ :=
  let c := Real.cos θ
  let s := Real.sin θ
  let s2 := Real.sqrt 2
  match i.val, j.val with
  | 0, 0 => c * c
  | 0, 2 => s * s
  | 0, 4 => s2 * c * s
  | 1, 1 => 1
  | 2, 0 => s * s
  | 2, 2 => c * c
  | 2, 4 => -s2 * c * s
  | 3, 3 => c
  | 3, 5 => -s
  | 4, 0 => -s2 * c * s
  | 4, 2 => s2 * c * s
  | 4, 4 => c * c - s * s
  | 5, 3 => s
  | 5, 5 => c
  | _, _ => 0

/-- At zero bedding dip ($\theta = 0$), the planar Kelvin-Mandel dip rotor reduces to the $6 \times 6$ identity matrix. -/
lemma kelvinMandelDipRotor_zero (i j : Fin 6) :
    kelvinMandelDipRotor 0 i j = if i = j then 1 else 0 := by
  fin_cases i <;> fin_cases j <;> simp [kelvinMandelDipRotor]

/-- For every bedding dip angle $\theta \in \mathbb{R}$, the planar Kelvin-Mandel dip rotor
`kelvinMandelDipRotor θ` is row-orthonormal ($M M^\top = I_6$), column-orthonormal
($M^\top M = I_6$), and orthogonal (`IsOrthogonal6`). -/
lemma kelvinMandelDipRotor_orthonormal (θ : ℝ) :
    IsRowOrthonormal6 (kelvinMandelDipRotor θ) ∧
    IsColOrthonormal6 (kelvinMandelDipRotor θ) ∧
    IsOrthogonal6 (kelvinMandelDipRotor θ) := by
  have h_row : ∀ φ : ℝ, IsRowOrthonormal6 (kelvinMandelDipRotor φ) := by
    intro φ a b
    have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
    have h2 := Real.cos_sq_add_sin_sq φ
    have h4 : Real.cos φ ^ 4 + Real.sin φ ^ 4 = 1 - 2 * Real.cos φ ^ 2 * Real.sin φ ^ 2 := by
      linear_combination (Real.cos φ ^ 2 + Real.sin φ ^ 2 + 1) * h2
    rw [Fin.sum_univ_six]
    fin_cases a <;>
      simp only [kelvinMandelDipRotor, Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
        Nat.one_mod, Nat.reduceMod, Nat.mod_succ, zero_mul, add_zero, zero_add] <;>
      fin_cases b <;>
      simp only [mul_zero, add_zero, Fin.mk.injEq, Nat.reduceEqDiff, ↓reduceIte] <;>
      ring_nf <;> (try simp only [hs2]) <;> try linarith [h2, h4]
  have h_tr : ∀ a b : Fin 6, kelvinMandelDipRotor θ a b = kelvinMandelDipRotor (-θ) b a := by
    intro a b; fin_cases a <;> fin_cases b <;> simp [kelvinMandelDipRotor]
  exact ⟨h_row θ, fun k l => by simpa only [h_tr] using h_row (-θ) k l,
    isOrthogonal6_of_rowOrthonormal6 _ (h_row θ)⟩

/-- **Theorem 7a (`elasticEnergy6_rotateStiffness6`)**:
For any $6 \times 6$ matrices $M, C$ and strain vector $\varepsilon : \mathrm{Fin}\,6 \to \mathbb{R}$,
the elastic strain energy of the rotated stiffness tensor `rotateStiffness6 M C` equals the elastic
strain energy of $C$ evaluated at the counter-rotated strain `matVec6 (matTranspose6 M) ε`:
$$\varepsilon^\top (M C M^\top) \varepsilon = (M^\top \varepsilon)^\top C (M^\top \varepsilon).$$ -/
theorem elasticEnergy6_rotateStiffness6 (M C : Fin 6 → Fin 6 → ℝ) (ε : Fin 6 → ℝ) :
    elasticEnergy6 (rotateStiffness6 M C) ε =
      elasticEnergy6 C (matVec6 (matTranspose6 M) ε) := by
  simp only [elasticEnergy6, rotateStiffness6, matVec6, matTranspose6, Fin.sum_univ_six]; ring

/-- **Theorem 7b (`so6_stiffness_preserves_pos_def`)**:
If $M$ satisfies `IsOrthogonal6 M` and $C$ is coercive (strictly positive-definite) with modulus
$\mu_{\min} > 0$, then the rotated TTI stiffness tensor `rotateStiffness6 M C` preserves the
exact coercivity lower bound:
$$\forall \varepsilon \in \mathbb{R}^6, \quad \mu_{\min} \|\varepsilon\|_2^2 \le \varepsilon^\top (M C M^\top) \varepsilon,$$
and for any non-zero strain ($\exists i, \varepsilon_i \neq 0$), we strictly have
$$0 < \varepsilon^\top (M C M^\top) \varepsilon.$$ -/
theorem so6_stiffness_preserves_pos_def (M C : Fin 6 → Fin 6 → ℝ) (μ_min : ℝ)
    (hM : IsOrthogonal6 M)
    (hμ : 0 < μ_min)
    (h_coercive : ∀ ξ : Fin 6 → ℝ,
      μ_min * Formalization.Analysis.shellEnergy ξ ≤ elasticEnergy6 C ξ) :
    (∀ ε : Fin 6 → ℝ,
      μ_min * Formalization.Analysis.shellEnergy ε ≤
        elasticEnergy6 (rotateStiffness6 M C) ε) ∧
    (∀ ε : Fin 6 → ℝ, (∃ i : Fin 6, ε i ≠ 0) →
      0 < elasticEnergy6 (rotateStiffness6 M C) ε) := by
  have h_bound : ∀ ε : Fin 6 → ℝ,
      μ_min * Formalization.Analysis.shellEnergy ε ≤
        elasticEnergy6 (rotateStiffness6 M C) ε := fun ε => by
    rw [elasticEnergy6_rotateStiffness6, ← hM ε]
    exact h_coercive _
  refine ⟨h_bound, fun ε ⟨i, hi⟩ => lt_of_lt_of_le (mul_pos hμ ?_) (h_bound ε)⟩
  exact lt_of_le_of_ne (shellEnergy_nonneg ε) fun h => hi ((shellEnergy_eq_zero_iff ε).mp h.symm i)

/-- **Theorem 7c (`so6_stiffness_trace_conservation`)**:
For any column-orthonormal $6 \times 6$ Kelvin-Mandel rotor $M$ ($\sum_i M_{i,k} M_{i,l} = \delta_{k,l}$)
and stiffness tensor $C$, rotating $C$ into tilted strata preserves its exact trace:
$$\mathrm{Tr}(M C M^\top) = \mathrm{Tr}(C).$$ -/
theorem so6_stiffness_trace_conservation (M C : Fin 6 → Fin 6 → ℝ)
    (hM : IsColOrthonormal6 M) :
    trace6 (rotateStiffness6 M C) = trace6 C := by
  unfold trace6 rotateStiffness6
  calc (∑ i : Fin 6, ∑ k : Fin 6, ∑ l : Fin 6, M i k * C k l * M i l)
    _ = ∑ k : Fin 6, ∑ l : Fin 6, C k l * (∑ i : Fin 6, M i k * M i l) := by
        simp only [Fin.sum_univ_six]; ring
    _ = ∑ i : Fin 6, C i i := by simp [hM _ _]

/-- 1D backward sonic difference $\Delta_- v(z) = v(z) - v(z - 1)$ along borehole depth $z \in \mathbb{Z}$
(`BeltramiWellLogShell` in `g3vm/engine.py`). -/
def deltaMinus (v : ℤ → ℝ) (z : ℤ) : ℝ :=
  v z - v (z - 1)

/-- 1D forward sonic difference $\Delta_+ v(z) = v(z) - v(z + 1)$ along borehole depth $z \in \mathbb{Z}$. -/
def deltaPlus (v : ℤ → ℝ) (z : ℤ) : ℝ :=
  v z - v (z + 1)

/-- 3-point discrete Morse critical-point impulse product
$$I_{\mathrm{Morse}}(v, z) = \Delta_- v(z) \cdot \Delta_+ v(z) = (v(z) - v(z - 1))(v(z) - v(z + 1)).$$ -/
def morseImpulseProd (v : ℤ → ℝ) (z : ℤ) : ℝ :=
  deltaMinus v z * deltaPlus v z

/-- Predicate characterizing a 3-point discrete Morse critical impulse ($I_{\mathrm{Morse}}(v, z) > 0$). -/
def isMorseImpulse (v : ℤ → ℝ) (z : ℤ) : Prop :=
  0 < morseImpulseProd v z

/-- 3-point discrete Laplacian spike amplitude estimator
$$\Delta_{3\mathrm{pt}} v(z) = \frac{\Delta_- v(z) + \Delta_+ v(z)}{2} = v(z) - \frac{v(z - 1) + v(z + 1)}{2}.$$ -/
noncomputable def laplacian3pt (v : ℤ → ℝ) (z : ℤ) : ℝ :=
  (deltaMinus v z + deltaPlus v z) / 2

/-- **Theorem 7d (`morseImpulse_iff_strict_extremum`)**:
The 3-point discrete Morse condition `isMorseImpulse v z` holds if and only if depth $z$ is either
a strict local sonic peak ($v(z - 1) < v(z)$ and $v(z + 1) < v(z)$) or a strict local sonic trough
($v(z) < v(z - 1)$ and $v(z) < v(z + 1)$). -/
theorem morseImpulse_iff_strict_extremum (v : ℤ → ℝ) (z : ℤ) :
    isMorseImpulse v z ↔
      (v (z - 1) < v z ∧ v (z + 1) < v z) ∨ (v z < v (z - 1) ∧ v z < v (z + 1)) := by
  unfold isMorseImpulse morseImpulseProd deltaMinus deltaPlus
  rw [mul_pos_iff]
  simp only [sub_pos, sub_neg]

/-- **Theorem 7e (`caliper_morse_rejects_washout_step`)**:
1. On any single-sided borehole caliper washout step where either flank is flat
   ($v(z) = v(z - 1)$ or $v(z) = v(z + 1)$), the Morse impulse product vanishes identically
   (`morseImpulseProd v z = 0`) and `¬ isMorseImpulse v z`.
2. On any monotone washout ramp or compaction trend
   ($v(z - 1) \le v(z) \le v(z + 1)$ or $v(z + 1) \le v(z) \le v(z - 1)$),
   the Morse impulse product is non-positive (`morseImpulseProd v z ≤ 0`) and `¬ isMorseImpulse v z`. -/
theorem caliper_morse_rejects_washout_step (v : ℤ → ℝ) (z : ℤ) :
    ((v z = v (z - 1) ∨ v z = v (z + 1)) →
      morseImpulseProd v z = 0 ∧ ¬ isMorseImpulse v z) ∧
    (((v (z - 1) ≤ v z ∧ v z ≤ v (z + 1)) ∨ (v (z + 1) ≤ v z ∧ v z ≤ v (z - 1))) →
      morseImpulseProd v z ≤ 0 ∧ ¬ isMorseImpulse v z) := by
  refine ⟨fun h_flat => ?_, fun h_mono => ?_⟩
  · have h0 : morseImpulseProd v z = 0 := by
      unfold morseImpulseProd deltaMinus deltaPlus
      rcases h_flat with h | h <;> { rw [h]; ring }
    exact ⟨h0, by unfold isMorseImpulse; linarith⟩
  · have h_le : morseImpulseProd v z ≤ 0 := by
      unfold morseImpulseProd deltaMinus deltaPlus
      rcases h_mono with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> nlinarith
    exact ⟨h_le, by unfold isMorseImpulse; linarith⟩

/-- **Theorem 7f (`caliper_morse_shields_thin_stringer`)**:
For any thin-bed carbonate, cryogenic water-ice, or high-Ti ilmenite stringer spike embedded in
background stratum velocity $v_{\mathrm{bg}}$ with $v(z - 1) = v_{\mathrm{bg}}$,
$v(z + 1) = v_{\mathrm{bg}}$, and $v(z) = v_{\mathrm{bg}} + \delta_{\mathrm{spike}}$
($\delta_{\mathrm{spike}} \neq 0$), the 3-point discrete Morse operator satisfies
`morseImpulseProd v z = δ_spike ^ 2`, strictly triggers `isMorseImpulse v z`, and recovers 100% of
the spike amplitude via `laplacian3pt v z = δ_spike`. -/
theorem caliper_morse_shields_thin_stringer (v : ℤ → ℝ) (z : ℤ) (v_bg δ_spike : ℝ)
    (h_left : v (z - 1) = v_bg)
    (h_right : v (z + 1) = v_bg)
    (h_center : v z = v_bg + δ_spike)
    (h_nz : δ_spike ≠ 0) :
    morseImpulseProd v z = δ_spike ^ 2 ∧
    isMorseImpulse v z ∧
    laplacian3pt v z = δ_spike := by
  have h_prod : morseImpulseProd v z = δ_spike ^ 2 := by
    unfold morseImpulseProd deltaMinus deltaPlus; rw [h_left, h_right, h_center]; ring
  refine ⟨h_prod, by unfold isMorseImpulse; rw [h_prod]; positivity, ?_⟩
  unfold laplacian3pt deltaMinus deltaPlus; rw [h_left, h_right, h_center]; ring

/-! ### Theorem 8 (Family 2.8): 3D Ideal MHD / Stellarator Symmetry-Breaking Isomorphism -/

/-- 3D vector cross product $(u \times v)_k$ on $\mathrm{Fin}\,3 \to \mathbb{R}$:
$$u \times v = (u_1 v_2 - u_2 v_1,\, u_2 v_0 - u_0 v_2,\, u_0 v_1 - u_1 v_0).$$ -/
def cross3 (u v : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  match k.val with
  | 0 => u 1 * v 2 - u 2 * v 1
  | 1 => u 2 * v 0 - u 0 * v 2
  | _ => u 0 * v 1 - u 1 * v 0

/-- 3D cyclic Levi-Civita generator tensor $W_{\mathrm{LC}} : \mathrm{Fin}\,3 \to \mathrm{Fin}\,3 \to \mathrm{Fin}\,3 \to \mathbb{R}$
taking value $1$ on even cyclic triples $(0, 1, 2), (1, 2, 0), (2, 0, 1)$ and $0$ otherwise, so that
its skew-symmetrization $W_{\mathrm{LC}}(i, j, k) - W_{\mathrm{LC}}(i, k, j) = \varepsilon_{ijk}$
is the exact 3D Levi-Civita antisymmetric symbol. -/
def leviCivitaHalfTensor (i j k : Fin 3) : ℝ :=
  match i.val, j.val, k.val with
  | 0, 1, 2 => 1
  | 1, 2, 0 => 1
  | 2, 0, 1 => 1
  | _, _, _ => 0

/-- **Theorem 8a (`cross3_eq_triadOperator`)**:
For all 3D vector fields $u, v : \mathrm{Fin}\,3 \to \mathbb{R}$ and component $k \in \mathrm{Fin}\,3$,
the $C = 3$ DarkSnail `triadOperator` with `leviCivitaHalfTensor` is identically the 3D cross product:
$$B_{W_{\mathrm{LC}}}(u, v)_k = (u \times v)_k.$$
In particular, both the 3D MHD Lorentz force $\mathbf{J} \times \mathbf{B}$ and the 3D hydrodynamic
Lamb vector $\boldsymbol{\omega} \times \mathbf{u}$ are exact $C = 3$ instances of `triadOperator`. -/
theorem cross3_eq_triadOperator (u v : Fin 3 → ℝ) (k : Fin 3) :
    triadOperator 3 leviCivitaHalfTensor u v k = cross3 u v k := by
  unfold triadOperator Formalization.Analysis.triadOperator leviCivitaHalfTensor cross3
  fin_cases k <;> simp [Fin.sum_univ_three] <;> ring

/-- 3D ideal magnetohydrodynamic (MHD) static force-balance predicate
$$\mathbf{J} \times \mathbf{B} = \nabla p \iff \forall k \in \mathrm{Fin}\,3,\; B_{W_{\mathrm{LC}}}(\mathbf{J}, \mathbf{B})_k = (\nabla p)_k.$$ -/
def IsIdealMHDBalance (J B grad_p : Fin 3 → ℝ) : Prop :=
  ∀ k : Fin 3, triadOperator 3 leviCivitaHalfTensor J B k = grad_p k

/-- Steady 3D Euler / Beltrami hydrodynamic force-balance predicate
$$\boldsymbol{\omega} \times \mathbf{u} = -\nabla H \iff \forall k \in \mathrm{Fin}\,3,\; B_{W_{\mathrm{LC}}}(\boldsymbol{\omega}, \mathbf{u})_k = -(\nabla H)_k,$$
where $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ is vorticity and $H = p + \frac{1}{2}|\mathbf{u}|^2$ is the Bernoulli head. -/
def IsEulerLambBalance (ω u grad_H : Fin 3 → ℝ) : Prop :=
  ∀ k : Fin 3, triadOperator 3 leviCivitaHalfTensor ω u k = -grad_H k

/-- Canonical involutive isomorphism mapping a 3D ideal MHD state $(\mathbf{J}, \mathbf{B}, \nabla p)$
to its dual steady Euler hydrodynamic state $(\boldsymbol{\omega}, \mathbf{u}, \nabla H) = (\mathbf{J}, \mathbf{B}, -\nabla p)$. -/
def mhdToEulerMap (state : (Fin 3 → ℝ) × (Fin 3 → ℝ) × (Fin 3 → ℝ)) :
    (Fin 3 → ℝ) × (Fin 3 → ℝ) × (Fin 3 → ℝ) :=
  (state.1, state.2.1, -state.2.2)

/-- `mhdToEulerMap` is a strict involution on 3D force-balance triples. -/
lemma mhdToEulerMap_involutive : Function.Involutive mhdToEulerMap := by
  rintro ⟨J, B, grad_p⟩; simp [mhdToEulerMap]

/-- **Theorem 8b (`mhd_euler_force_balance_bijection`)**:
1. The MHD-to-Euler state map `mhdToEulerMap` $(\mathbf{J}, \mathbf{B}, \nabla p) \mapsto (\mathbf{J}, \mathbf{B}, -\nabla p)$
   is a bijection and an exact involution (`mhdToEulerMap (mhdToEulerMap s) = s`).
2. A triple $(\mathbf{J}, \mathbf{B}, \nabla p)$ satisfies 3D ideal MHD force balance
   $\mathbf{J} \times \mathbf{B} = \nabla p$ if and only if its image under `mhdToEulerMap`
   satisfies steady Euler / Beltrami Lamb-vector balance $\boldsymbol{\omega} \times \mathbf{u} = -\nabla H$,
   which in turn holds if and only if `cross3 J B k = grad_p k` for all $k \in \mathrm{Fin}\,3$. -/
theorem mhd_euler_force_balance_bijection :
    Function.Bijective mhdToEulerMap ∧
    (∀ s : (Fin 3 → ℝ) × (Fin 3 → ℝ) × (Fin 3 → ℝ),
      mhdToEulerMap (mhdToEulerMap s) = s) ∧
    (∀ J B grad_p : Fin 3 → ℝ,
      IsIdealMHDBalance J B grad_p ↔
        IsEulerLambBalance (mhdToEulerMap (J, B, grad_p)).1
          (mhdToEulerMap (J, B, grad_p)).2.1
          (mhdToEulerMap (J, B, grad_p)).2.2) ∧
    (∀ J B grad_p : Fin 3 → ℝ,
      IsIdealMHDBalance J B grad_p ↔ (∀ k : Fin 3, cross3 J B k = grad_p k)) := by
  refine ⟨mhdToEulerMap_involutive.bijective, mhdToEulerMap_involutive,
    fun J B grad_p => by simp [IsIdealMHDBalance, IsEulerLambBalance, mhdToEulerMap],
    fun J B grad_p => by simp [IsIdealMHDBalance, cross3_eq_triadOperator]⟩

/-- Magnetic flux-surface tangency (confinement) predicate:
$$\mathbf{B} \cdot \nabla\psi = \langle \mathbf{B}, \nabla\psi \rangle = 0,$$
expressing that the magnetic field $\mathbf{B}$ lies everywhere tangent to constant-flux surfaces $\psi = \text{const}$. -/
def IsFluxSurfaceConfined {C : ℕ} (B grad_ψ : Fin C → ℝ) : Prop :=
  Formalization.Analysis.shellInner B grad_ψ = 0

/-- **Theorem 8c (`mhd_flux_surface_confinement_iff`)**:
1. Under the generalized MHD / stellarator force-balance relation $\nabla\psi = B_W(u, v)$
   (identifying active current $u \leftrightarrow \mathbf{J}$ and magnetic field $v \leftrightarrow \mathbf{B}$),
   tangent flux-surface confinement $\mathbf{B} \cdot \nabla\psi = 0$ is equivalent to triad orthogonality
   $\langle v, B_W(u, v) \rangle = 0$.
2. Both `IsFluxSurfaceConfined v (triadOperator C W u v)` in arbitrary mode dimension $C$ and
   3D magnetic Lorentz orthogonality $\mathbf{B} \cdot (\mathbf{J} \times \mathbf{B}) = 0$ hold
   identically for all interaction tensors $W$ and fields $u, v, \mathbf{J}, \mathbf{B}$. -/
theorem mhd_flux_surface_confinement_iff {C : ℕ}
    (W : Fin C → Fin C → Fin C → ℝ) (u v grad_ψ : Fin C → ℝ)
    (J B : Fin 3 → ℝ) :
    (grad_ψ = triadOperator C W u v →
      (IsFluxSurfaceConfined v grad_ψ ↔
        Formalization.Analysis.shellInner v (triadOperator C W u v) = 0)) ∧
    IsFluxSurfaceConfined v (triadOperator C W u v) ∧
    Formalization.Analysis.shellInner B (cross3 J B) = 0 := by
  refine ⟨fun h => by rw [h]; rfl,
    Formalization.Analysis.triad_pointwise_orthogonal W u v, ?_⟩
  rw [show cross3 J B = triadOperator 3 leviCivitaHalfTensor J B from
    funext fun k => (cross3_eq_triadOperator J B k).symm]
  exact Formalization.Analysis.triad_pointwise_orthogonal leviCivitaHalfTensor J B

/-- Non-stellarator-symmetric (parity-odd $\sin(m\theta - n\zeta)$) triad perturbation shift
$$\delta z(\tau_0)_k = \tau_0 \cdot B_W(u, v)_k$$
scaled by the magnetic torsion / non-axisymmetric parity amplitude $\tau_0 \in \mathbb{R}$. -/
def stellaratorParityShift (C : ℕ) (τ₀ : ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  τ₀ * triadOperator C W u v k

/-- Full stellarator Fourier mode readout combining a stellarator-symmetric (parity-even
$\cos(m\theta - n\zeta)$) baseline $\ell_{\mathrm{sym}}$ with the skew-symmetric parity perturbation
`stellaratorParityShift`:
$$\ell_{\mathrm{stell}}(k) = \ell_{\mathrm{sym}}(k) + \tau_0 B_W(u, v)_k.$$ -/
def stellaratorReadout (C : ℕ) (ℓ_sym : Fin C → ℝ) (τ₀ : ℝ)
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_sym k + stellaratorParityShift C τ₀ W u v k

/-- **Theorem 8d (`stellarator_parity_breaks_tie_and_conserves_flux`)**:
Let $\ell_{\mathrm{sym}} : \mathrm{Fin}\,C \to \mathbb{R}$ be any stellarator-symmetric (parity-even)
baseline readout that ties on two resonant Fourier modes $c_1, c_2$
($\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)$). In a 2-mode resonant collision state
$v(c_1) = v(c_2) = \alpha > 0$ (and $v(c) = 0$ for $c \notin \{c_1, c_2\}$) driven by active current
mode $u(i_0) = \beta > 0$ (and $u(i) = 0$ for $i \neq i_0$) with non-stellarator-symmetric torsion
parameter $\tau_0 > 0$:
1. **Resonant Mode Tie-Breaking**:
   $$\ell_{\mathrm{stell}}(c_1) - \ell_{\mathrm{stell}}(c_2) = 2 \tau_0 \beta \alpha \bigl(W(i_0, c_2, c_1) - W(i_0, c_1, c_2)\bigr),$$
   and whenever $W(i_0, c_2, c_1) > W(i_0, c_1, c_2)$, we strictly have
   $\ell_{\mathrm{stell}}(c_1) > \ell_{\mathrm{stell}}(c_2)$.
2. **Tangent Flux-Surface Confinement**:
   The non-stellarator-symmetric perturbation $\delta z(\tau_0)$ remains strictly tangent to the
   magnetic flux surface (`IsFluxSurfaceConfined v (stellaratorParityShift C τ₀ W u v)`, i.e.,
   $\langle v, \delta z(\tau_0) \rangle = 0$). -/
theorem stellarator_parity_breaks_tie_and_conserves_flux {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (τ₀ : ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ : Fin C) (α β : ℝ)
    (hτ : 0 < τ₀) (hα : 0 < α) (hβ : 0 < β)
    (h_tie : ℓ_sym c₁ = ℓ_sym c₂)
    (hu_i0 : u i₀ = β)
    (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α)
    (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    (stellaratorReadout C ℓ_sym τ₀ W u v c₁ -
      stellaratorReadout C ℓ_sym τ₀ W u v c₂ =
        2 * τ₀ * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂)) ∧
    (W i₀ c₂ c₁ > W i₀ c₁ c₂ →
      stellaratorReadout C ℓ_sym τ₀ W u v c₁ >
        stellaratorReadout C ℓ_sym τ₀ W u v c₂) ∧
    IsFluxSurfaceConfined v (stellaratorParityShift C τ₀ W u v) := by
  have h_diff :
      stellaratorReadout C ℓ_sym τ₀ W u v c₁ -
        stellaratorReadout C ℓ_sym τ₀ W u v c₂ =
          2 * τ₀ * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂) := by
    by_cases hc : c₁ = c₂
    · subst hc; ring
    · simp only [stellaratorReadout, stellaratorParityShift, h_tie,
        triadOperator_two_color_eval W u v i₀ c₁ c₂ _ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other]
      ring
  exact ⟨h_diff, fun hW => by nlinarith [mul_pos (mul_pos hτ hβ) hα],
    tangentTriadStep_orthogonal τ₀ W u v⟩

/-- **Theorem 8e (`stellarator_zero_torsion_homotopy_iff`)**:
For any non-degenerate triad interaction (`∃ k : Fin C, triadOperator C W u v k ≠ 0`),
the non-stellarator-symmetric parity shift $\delta z(\tau_0) = \mathrm{stellaratorParityShift}\,C\,\tau_0\,W\,u\,v$
vanishes if and only if the magnetic torsion parameter vanishes ($\tau_0 = 0 \iff \delta z(\tau_0) = 0$),
and at $\tau_0 = 0$ the full stellarator readout reduces identically to the stellarator-symmetric
baseline $\ell_{\mathrm{sym}}$. -/
theorem stellarator_zero_torsion_homotopy_iff {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (τ₀ : ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ)
    (h_nondeg : ∃ k : Fin C, triadOperator C W u v k ≠ 0) :
    (τ₀ = 0 ↔ stellaratorParityShift C τ₀ W u v = 0) ∧
    (τ₀ = 0 → ∀ k : Fin C, stellaratorReadout C ℓ_sym τ₀ W u v k = ℓ_sym k) := by
  refine ⟨⟨fun hτ => funext fun k => by simp [stellaratorParityShift, hτ], fun h0 => ?_⟩,
    fun hτ k => by simp [stellaratorReadout, stellaratorParityShift, hτ]⟩
  rcases h_nondeg with ⟨k, hk⟩
  exact (mul_eq_zero.mp (congr_fun h0 k)).resolve_right hk

end Ultrametric.Analysis.OcclusionTriad
