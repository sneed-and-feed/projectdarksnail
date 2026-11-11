# DarkSnail Atlas: Unified Theorem-to-Silicon Verification Matrix

## 1. Executive Summary

The **Project DarkSnail** ecosystem unites relational foundation world modeling, a universal proof-carrying SIMD kernel (`triad-core`), sub-35% causal attention sparsity, embodied visuomotor manipulation, Level-3 tick microstructure tomography, optical relative guidance and navigation, anisotropic topology optimization, non-symmetric magnetohydrodynamic equilibrium, ocean acoustic tomography, crust-to-core 3D seismic raytracing and early warning, time-domain astrophysics, and 1D chromosomal copy-number detection into a single machine-verified 12-repository architecture.

Every mathematical claim across the hub, universal kernel, and 10 domain spokes is grounded in machine-checked formal proofs in Lean 4 (`formalization/OcclusionTriad.lean`, `formalization/BlockBridge.lean`, and `formalization/TriadCore.lean`), verified on Broadcom BCM2712 Cortex-A76 hardware cache geometry (Raspberry Pi 5), audited for zero dynamic heap allocations in `#![no_std]` Rust static arenas, and benchmarked against physical and empirical ground truth.

```text
                        [Project DarkSnail Hub]
            (Relational Sheaf, Beltrami Shell & Block-Bridge)
                                   │
                        [triad-core Universal ABI]
           (Lean 4 + #![no_std] Quad-SIMD Arena: 894,912 B)
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    ▼                              ▼                              ▼
[RoboSnail]                  [Stylo 6-DOF GNC]               [LOB-Triad]
Embodied Robotics            EO/IR Autonomous Intercept      Microstructure L3
731 KB Arena / 60 Hz         673 KB / 0 Weights / 0 Swaps    254 KB / 135k ticks/s
    │                              │                              │
    ├──────────────────────────────┼──────────────────────────────┤
    ▼                              ▼                              ▼
[TopoSnail & BMHD]           [G3VM v0.7.0 & QuakeSnail]      [OATSE v0.2.0]
3D MHD & TTI Topology        Crust-to-Core 3D & Live EEW     Ocean Acoustic & SAS
1.47 MB / b2 = 0             1.75 MB / 96.8% RMSE Drop       1.60 MB / Zero Ringing
    │                              │                              │
    ├──────────────────────────────┴──────────────────────────────┤
    ▼                                                             ▼
[AstroSnail v0.2.0]                                         [CellSnail v0.3.0]
Time-Domain Sky & Cosmic Web                                1D Chromosomal CNV Engine
1.01 MB Arena / Kepler & 2MASS                              1.30 MB Arena / 0.04073% LLd
    │                                                             │
    └──────────────────────────────┬──────────────────────────────┘
                                   ▼
                 [Theorem-to-Silicon Verification Layer]
            Lean 4 (0 sorry) // BCM2712 Cache (< 0.06% LLd) // 0 Alloc
```

---

## 2. Mathematical Foundations: The Core Theorem Hierarchy

The mathematical core of DarkSnail is formalized across the hub (`formalization/OcclusionTriad.lean`, `formalization/BlockBridge.lean`), universal kernel (`triad-core/formalization/TriadCore.lean`), and domain spokes (`formalization/OcclusionTriad.lean`). Every theorem compiles with 0 `sorry` tokens and 0 custom axioms under Lean 4 and Mathlib 4.

### Theorem 2.1: Skew-Symmetric Triad Antisymmetry

For any structural mode $i$ and competing states $c_1, c_2 \in \mathrm{Fin}\,C$, the skew-symmetrized interaction tensor:

```math
S(W, i, c_1, c_2) = W(i, c_1, c_2) - W(i, c_2, c_1)
```

satisfies strict antisymmetry under index swap:

```math
S(W, i, c_2, c_1) = -S(W, i, c_1, c_2)
```

and self-interaction vanishes identically:

```math
S(W, i, c_1, c_1) = 0
```

- **Formal Symbols**: `skew_triad_swap_neg`, `skew_triad_self_zero`
- **Ecosystem Realization**: `darksnail/readout.py`, `triad-core/rust/triad_core_rs/src/lib.rs`, `robosnail/rust/robosnail_rs/src/lib.rs`, `lob-triad/rust/lob_triad_rs/src/lib.rs`

### Theorem 2.2: Exact Tie-Breaking of Symmetric Readout Collisions

Let baseline logit vector $\ell_{\mathrm{sym}}$ tie on two competing candidates with equal baseline scores:

```math
\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)
```

In a two-mode collision state with active mode weight $\beta > 0$ and collision amplitude $\alpha > 0$, the bilinear triad shift generates an exact antisymmetric margin:

```math
\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2 \beta \alpha \bigl(W(i_0, c_2, c_1) - W(i_0, c_1, c_2)\bigr)
```

whenever the mode priority condition holds:

```math
W(i_0, c_2, c_1) > W(i_0, c_1, c_2) \implies \ell_{\mathrm{post}}(c_1) > \ell_{\mathrm{post}}(c_2)
```

resolving ambiguous collisions deterministically with zero drift.

- **Formal Symbol**: `occlusion_triad_breaks_symmetric_tie`
- **Ecosystem Realization**: `darksnail/arc3.py`, `stylo/darksnail/tracker.py`, `quakesnail/core/association.py`, `astrosnail/astrometry.py`

### Theorem 2.3: Pointwise Tangent Orthogonality and Zero Radial Energy Growth

For any structural mode $u$, probability state $v$, and interaction tensor $W$, the inner product of $v$ with the bilinear triad shift vanishes identically:

```math
\sum_{k \in \mathrm{Fin}\,C} v_k \cdot B_W(u, v)(k) = 0
```

For continuous-time recurrent state $\dot z_L = B_W(z_H, z_L)$, the radial energy growth rate satisfies:

```math
\langle z_L, \dot z_L \rangle = 0
```

- **Formal Symbol**: `darksnail_readout_energy_conservation`
- **Ecosystem Realization**: `darksnail/shell.py`, `triad-core/triad_core/operators.py`, `stylo/darksnail/tracker.py`, `robosnail/rust/robosnail_rs/src/lib.rs`

### Theorem 2.4: Step-0 System 1 Homotopy Equivalence

At initial parameterization $W = 0$ and $\Delta_{\mathrm{sym}} = 0$, the full DarkSnail readout reduces identically to pretrained baseline logits:

```math
\ell_{\mathrm{DS}}(\ell_{\mathrm{S1}}, 0, 0, u, v, k) = \ell_{\mathrm{S1}}(k)
```

- **Formal Symbol**: `darksnail_step0_system1_homotopy`
- **Ecosystem Realization**: `darksnail/gemma4.py`, `stylo/darksnail/gnc_6dof.py`

### Theorem 2.5: 2D Axial RoPE Medoid Invariance and Patch Attenuation

In 2D axial coordinates $(y, x)$, selecting a 2D medoid key preserves exact relative rotary displacement:

```math
\langle \mathcal R_{y_q, x_q} q, \mathcal R_{y^\ast, x^\ast} k \rangle = \langle q_y, \mathcal R_{y^\ast - y_q} k_y \rangle + \langle q_x, \mathcal R_{x^\ast - x_q} k_x \rangle
```

while arithmetic patch averaging across distinct 2D grid phases strictly attenuates the squared key norm:

```math
\left\|\mathrm{rope2dPatchAvg}(\theta_{y,1}, \theta_{x,1}, \theta_{y,2}, \theta_{x,2}, k_y, k_x)\right\|^2 < \|(k_y, k_x)\|^2
```

- **Formal Symbols**: `rope2d_medoid_relative_invariance`, `rope2d_arithmetic_norm_lt`
- **Ecosystem Realization**: `darksnail/gemma4.py`, `stylo/rust/darksnail_tracker_rs/src/lib.rs`, `astrosnail/astrometry.py`

### Theorem 2.6: Multi-Step Unrolled Slow-Fast Spherical Invariance

Across any $T$-step unrolled Slow-Fast spherical triad cascade, retracted state energy is preserved:

```math
\|\tilde v_L^{(T)}\|^2 = \|v_L^{(0)}\|^2
```

and at $W_{\mathrm{rec}} = 0$, the unrolled cascade reduces identically to the System 1 carry trajectory:

```math
z^{(T)}(W_{\mathrm{rec}} = 0) \equiv z_{\mathrm{S1}}^{(T)}
```

- **Formal Symbols**: `darksnail_multistep_spherical_invariance`, `darksnail_multistep_step0_homotopy`
- **Ecosystem Realization**: `darksnail/shell.py`, `triad-core/rust/triad_core_rs/src/lib.rs`, `robosnail/rust/robosnail_rs/src/lib.rs`

### Theorem 2.7: SO(6) Kelvin-Mandel Stiffness Conservation and Discrete Morse Shielding

For any orthogonal 6D Kelvin-Mandel rotor $M \in \mathrm{SO}(6)$ and stiffness tensor $C$:
1. Elastic strain energy is preserved under rotated strain:

```math
\varepsilon^\top (M C M^\top) \varepsilon = (M^\top \varepsilon)^\top C (M^\top \varepsilon)
```

2. Strict coercivity modulus $\mu_{\min} > 0$ and tensor trace $\mathrm{Tr}(M C M^\top) = \mathrm{Tr}(C)$ are strictly conserved.
3. The 3-point discrete Morse impulse gate:

```math
I_{\mathrm{Morse}}(v, z) = (v(z) - v(z - 1)) \cdot (v(z) - v(z + 1))
```

strictly rejects monotone ramps and flat washout steps ($I_{\mathrm{Morse}} \le 0$), while recovering isolated spikes and 1D focal amplicons at 100% amplitude.

- **Formal Symbols**: `kelvinMandelDipRotor_orthonormal`, `elasticEnergy6_rotateStiffness6`, `so6_stiffness_preserves_pos_def`, `so6_stiffness_trace_conservation`, `morseImpulse_iff_strict_extremum`, `caliper_morse_rejects_washout_step`, `caliper_morse_shields_thin_stringer`
- **Ecosystem Realization**: `triad-core`, `toposnail`, `g3vm`, `oatse`, `quakesnail`, `lob-triad`, `astrosnail`, `cellsnail`

### Theorem 2.8: 3D Ideal MHD / Stellarator Isomorphism and Relaxed-Flow MRxMHD

1. The $C = 3$ Levi-Civita triad operator is identically the 3D vector cross product:

```math
B_{W_{\mathrm{LC}}}(u, v) = u \times v
```

2. Ideal MHD static force balance ($\mathbf{J} \times \mathbf{B} = \nabla p$) is in exact bijective, involutive correspondence with steady Euler hydrodynamics ($\boldsymbol{\omega} \times \mathbf{u} = -\nabla H$).
3. Magnetic flux-surface tangency ($\mathbf{B} \cdot \nabla\psi = 0$) is equivalent to triad orthogonality.
4. Non-stellarator-symmetric parity perturbation $\delta z(\tau_0) = \tau_0 B_W(u, v)$ breaks resonant mode ties while maintaining exact flux-surface confinement.
5. In Relaxed-Flow MRxMHD with sub-Alfvénic Mach flow ($M^2 < 1$), collinear Beltrami fields have vanishing interior Lorentz force, and the interface Bernoulli jump:

```math
p_2 - p_1 = \frac{(1 + M_1^2) B_1^2 - (1 + M_2^2) B_2^2}{2}
```

reduces identically to static Taylor-Hudson relaxation at zero flow ($M_1 = M_2 = 0$).

- **Formal Symbols**: `cross3_eq_triadOperator`, `mhd_euler_force_balance_bijection`, `mhd_flux_surface_confinement_iff`, `stellarator_parity_breaks_tie_and_conserves_flux`, `stellarator_zero_torsion_homotopy_iff`, `sub_alfvenic_beltrami_homotopy_zero_flow`, `beltrami_cross_helicity_collinear_lorentz_vanishes`, `mrxmhd_flow_bernoulli_jump_conservation`, `kam_sub_alfvenic_ellipticity_and_gap`
- **Ecosystem Realization**: `bmhd/bmhd/engine.py`, `toposnail/toposnail/stellarator_coupling.py`

### Theorems 3.1–3.4: Block-Bridge Sparse Attention (`formalization/BlockBridge.lean`)

At high compression ratios ($r \ge 5$), causal query blocks routed solely by smooth low-frequency block centroids suffer needle blindspots under uniform high-frequency distractors. Bridge-Aware calibration injects local high-frequency salience spikes alongside medoid block keys while bounding total active key-value blocks strictly beneath `< 35%` of sequence length (`block_sparse_budget_lt_35pct`).

---

## 3. Microarchitecture and Static `#![no_std]` Rust Arenas

Every real-time Rust crate in the ecosystem executes inside a pre-allocated static arena adhering to `#![no_std]` zero-heap disciplines. All arenas are aligned to 64-byte cache line boundaries.

| Repository | Static Arena Struct | Arena Size (Bytes) | SRAM Budget | Memory Alignment | Dynamic Heap Allocs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `projectdarksnail` | `SparseRouteArena` | 202,368 B | 1.50 MB | 64-byte | 0 alloc |
| `triad-core` | `TriadUniversalArena` | 894,912 B | 1.20 MB | 64-byte | 0 alloc |
| `stylo` | `TrackerStateArena` | 673,792 B | 1.50 MB | 64-byte | 0 alloc |
| `robosnail` | `RoboSnailArena` | 731,456 B | 2.00 MB | 64-byte | 0 alloc |
| `lob-triad` | `LOBStateArena` | 254,720 B | 1.50 MB | 64-byte | 0 alloc |
| `toposnail` | `TopoStateArena` | 1,472,448 B | 1.50 MB | 64-byte | 0 alloc |
| `quakesnail` | `QuakeSnailArena` | 1,745,552 B | 2.00 MB | 64-byte | 0 alloc |
| `astrosnail` | `AstroSnailArena` | 1,012,352 B | 1.50 MB | 64-byte | 0 alloc |
| `cellsnail` | `CellSnailArena` | 1,303,360 B | 2.00 MB | 64-byte | 0 alloc |

*(Streaming volumetric spokes `g3vm`, `bmhd`, and `oatse` evaluate 3D/4D fields via matrix-free `chunk_z = 16` Z-slab streaming within `< 1.65 MB` to `23.8 MB` active heap, strictly inside the `< 35 MB` edge SWaP-C envelope).*

---

## 4. Broadcom BCM2712 Cortex-A76 Hardware Cache Telemetry

Hardware cache verification was conducted using the Broadcom BCM2712 Cortex-A76 cache geometry verifier (`scripts/verify_bcm2712_cache.py` and `triad-core/scripts/verify_bcm2712_arena.py`).

### Cache Hierarchy Specifications
- **L1 Data Cache (L1d)**: 64 KB, 4-way set associative, 64-byte line size (256 sets)
- **L2 Cache**: 512 KB private per core, 8-way set associative, 64-byte line size (1024 sets)
- **L3 Cache (LLd)**: 2.0–4.0 MB shared, 16-way set associative, 64-byte line size
- **Simulation Scale**: 50,000,000 memory references per arena
- **Hard Requirement**: LLd miss rate `< 0.100%` across all static arenas

### Cache Verification Results

| Repository | Arena Struct | Size (KB) | Total References | L1d Hit Rate | L2 Hit Rate | L3 Hit Rate | LLd Miss Rate | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `triad-core` | `TriadUniversalArena` | 873.94 KB | 50,000,000 | 58.05% | 88.02% | 99.35% | 0.03280% | PASS (`< 0.050%`) |
| `stylo` | `TrackerStateArena` | 658.00 KB | 50,000,000 | 85.75% | 87.52% | 98.82% | 0.02106% | PASS (`< 0.100%`) |
| `toposnail` | `TopoStateArena` | 1437.94 KB | 50,000,000 | 68.02% | 32.60% | 99.79% | 0.04601% | PASS (`< 0.100%`) |
| `robosnail` | `RoboSnailArena` | 714.31 KB | 50,000,000 | 93.92% | 73.75% | 98.57% | 0.02286% | PASS (`< 0.100%`) |
| `lob-triad` | `LOBStateArena` | 248.75 KB | 50,000,000 | 91.75% | 99.90% | 0.00% | 0.00796% | PASS (`< 0.100%`) |
| `quakesnail` | `QuakeSnailArena` | 1704.64 KB | 50,000,000 | 67.18% | 70.12% | 99.44% | 0.05455% | PASS (`< 0.100%`) |
| `cellsnail` | `CellSnailArena` | 1272.81 KB | 50,000,000 | 45.44% | 70.37% | 99.75% | 0.04073% | PASS (`< 0.050%`) |

### Cache Verification Analysis
- **Zero Capacity Evictions in L3**: Because each static arena fits within on-chip L3 SRAM, steady-state capacity evictions to external LPDDR4X DRAM are strictly zero.
- **Pure Compulsory Misses**: Measured LLd misses correspond strictly to cold cache-line compulsory fills at initialization.

---

## 5. Static Allocator Audit (Zero Dynamic Heap Allocations)

Static binary import-table analysis (`PE`, `ELF`, `Mach-O`) confirms zero dynamic heap allocator imports across all compiled release `#![no_std]` libraries:

| Repository | Compiled Binary Target | Dynamic Heap Allocators Imported | Stepping Loop Allocation Calls | Audit Result |
| :--- | :--- | :--- | :--- | :--- |
| `projectdarksnail` | `darksnail_sparse_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `route_and_attend_block_sparse` | ZERO ALLOC VERIFIED |
| `triad-core` | `triad_core_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 across all 6 C-ABI primitives | ZERO ALLOC VERIFIED |
| `stylo` | `darksnail_tracker_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `tracker_step_frame` | ZERO ALLOC VERIFIED |
| `toposnail` | `toposnail_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `toposnail_step_iter` | ZERO ALLOC VERIFIED |
| `robosnail` | `robosnail_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `robosnail_process_cable_glare` | ZERO ALLOC VERIFIED |
| `lob-triad` | `lob_triad_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `lob_arena_process_tick` | ZERO ALLOC VERIFIED |
| `quakesnail` | `quakesnail_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `quakesnail_process_station_waveform` | ZERO ALLOC VERIFIED |
| `astrosnail` | `astrosnail_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `astrosnail_step_frame` | ZERO ALLOC VERIFIED |
| `cellsnail` | `cellsnail_rs` | 0 (`malloc`, `calloc`, `realloc` absent) | 0 in `morse_gate_genomic_f32` | ZERO ALLOC VERIFIED |

All stepping loops operate exclusively on pointers to pre-allocated static arenas or stack registers, guaranteeing deterministic Worst-Case Execution Time (WCET) free from heap fragmentation, lock contention, or garbage collection jitter.

---

## 6. Cross-Ecosystem Empirical Benchmarks Matrix

| Repository & Domain | Baseline Reference | DarkSnail Empirical Result | Performance Delta / Key Invariant | Benchmark Artifact |
| :--- | :--- | :--- | :--- | :--- |
| `projectdarksnail` ARC-AGI-1 | Direct Zero-Shot `gemma-4-31B-it` (`71 / 400` = `17.75%`) | `DarkSnailV2Solver` (`144.83 / 400` = `36.21%` official, `141 / 400` strict) | `+103.98%` relative gain (`0` task-ID branches; `393 / 416` = `94.47%` train split) | `darksnail_v2_evaluation_summary.json` |
| `projectdarksnail` Gemma 4 Sparse | `v4_triad@5`: `29.85 F1` (`0.0%` unshielded needle) | `darksnail@5`: `49.27 F1` (`98.70%` of dense, `100.0%` needle) | `+19.43 F1` at `r = 5` (`p = 0.0001`, `118W/22L/60T`, `< 35%` KV budget) | `darksnail_bridge_aware_r5_summary.json` |
| `projectdarksnail` ARC-AGI-3 | Random Legal (`2 / 183` levels) & Unshielded Greedy (`1 / 183`) | `63 / 183` levels (`34.43%`, offline CPU & live `gemma-4-31B-it` GPU) | `25 / 25` games entered (`100%`), `4 / 25` full wins, `19.9237%` RHAE | `darksnail_arc3_evaluation_summary.json` |
| `triad-core` 6-Vertical Kernel | Disparate domain-specific libraries | 6/6 verticals passing in single `894,912 B` static arena | `0` track swaps, `0.000e+00` Cayley drift, `0.0328%` BCM2712 LLd miss | `triad_core_audit_summary.json` |
| `stylo` EO/IR & 6-DOF GNC | Kalman/IoU (`77` swaps) & neural RPO policies | `9,406 FPS` Rust SIMD (`0` swaps); `0.04956 m` terminal miss | `0` neural weights, `0.0155 m/s` soft-docking contact speed | `stylo_6dof_gnc_summary.json` |
| `robosnail` Manipulation | Specular floor glare failure & actuator wear drift | `1.0000` (`100%`) 2mm cable spike recovery; `0.000e+00` rotor drift | Exact `SO(16)` online Cayley TTT & unilateral contact-chain sheaf | `robosnail_evaluation_summary.json` |
| `lob-triad` L3 Order Book | Spoof-polluted `OFI_raw = -20,000` & `10k` ticks/s target | `135,985` ticks/s (`254 KB` arena); `OFI_purified = +19,800` | `100%` (`502/502`) glitch rejection, `100%` (`204/204`) sweep detection | `lob_triad_evaluation_summary.json` |
| `toposnail` 3D Bionic CAD | Isotropic SIMP checkerboard & trapped `beta_2` voids | `0` trapped cavities (`b_2 = 0`), `86.6%` compliance reduction | Blind `.STL`/`.OBJ` mesh-to-bionic streaming (`< 35 MB` RAM) | `stellarator_bionic_coupling_summary.json` |
| `quakesnail` Live 3D EEW | Legacy GlobalQuake (`41.34 km` error, `countFelt` heuristics) | `0.0012 km` 3D error (`94.6%`–`96.8%` Mode B RMSE reduction) | `0` false glitch triggers on `10,000` spikes, `0` "people felt" counters | `quakesnail_evaluation_summary.json` |
| `g3vm` Utah FORGE & Lunar | Straight ray (`0.1506 km` FORGE, `0.5059 km` Lunar ISRU) | Blind 3D LET (`0.0243 km` FORGE, `0.0215 km` Lunar ISRU) | `83.86%` & `95.75%` RMSE reduction (`0.0` smearing, `< 96 KB` heap) | `g3vm_v070_let_benchmark_summary.json` |
| `bmhd` 3D Stellarator MHD | Axisymmetric/diffusive smoothers across stepped interfaces | Exact Flow-MRxMHD Bernoulli jump (`0.00e+0` interior Lorentz error) | `15/15` Mode B wins across `W7-X`, `HSX`, `NCSX`, `ITER`, & `MAST` | `bmhd_stylo_bridge_summary.json` |
| `oatse` Ocean Acoustics | Isotropic kriging washout across thermoclines & seabed targets | `100%` proud/buried seabed spike recovery (`< 1.60 MB` active heap) | Quarantines `M_2` internal tides while isolating moving UUVs | `oatse_evaluation_summary.json` |
| `astrosnail` Sky Survey | Sigma-clip (`10k/10k` false glints) & `PatchAvgKalmanIoU` | `5/5` (`100%`) blind Kepler `SAP_QUALITY` cosmic-ray match; `27/27` wins | `1,088.3 FPS` mean throughput; `0.000e+00` 2MASS Galactic Center phase drift | `real_astronomical_smoke_test.json` |
| `cellsnail` 1D Genomic CNV | Sliding-window smoothers & `HMMcopy` (`F_edge = 0.369`) | `0` spurious ERCC pairs (`GSE118767`); `F_edge = 0.971` focal amplicons | Boundary-pinned 1D chromosomal CNV detection (`GSE148673`, `GSE57872`) | `cellsnail_evaluation_summary.json` |

---

## 7. Deliverables Index

### Machine-Checked Formal Proofs (Lean 4, `0 sorry`, `0` custom axioms)
- `darksnail/formalization/OcclusionTriad.lean` & `darksnail/formalization/BlockBridge.lean`
- `triad-core/formalization/TriadCore.lean`
- `stylo/formalization/OcclusionTriad.lean`
- `robosnail/formalization/OcclusionTriad.lean`
- `lob-triad/formalization/OcclusionTriad.lean`
- `toposnail/formalization/OcclusionTriad.lean`
- `quakesnail/formalization/OcclusionTriad.lean`
- `g3vm/formalization/OcclusionTriad.lean`
- `bmhd/formalization/OcclusionTriad.lean`
- `oatse/formalization/OcclusionTriad.lean`
- `astrosnail/formalization/OcclusionTriad.lean`
- `cellsnail/formalization/OcclusionTriad.lean`

### Portable Zero-Install WebGL2 Studios
Self-contained single-file browser workstations (`0` external CDN dependencies):
- `triad-core/studio/triad_core_studio.html`: Universal 6-vertical functor switchboard and live theorem telemetry
- `stylo/studio/darksnail_tracker_studio.html`: 240 Hz MIL-STD-3009 4-quad EO/IR tracker and 6-DOF intercept HUD
- `robosnail/studio/robosnail_portable_studio.html`: 60 Hz 3D kinematic contact-chain sheaf and 2mm cable glare filter
- `lob-triad/studio/lob_triad_portable_studio.html`: Level-3 limit order book reconstruction and purified OFI studio
- `toposnail/studio/toposnail_portable_studio.html`: Interactive 3D CAD-to-bionic topology optimization workstation
- `quakesnail/studio/quakesnail_portable_studio.html`: 3D Earth globe, Wadati-Benioff cutaway, and live P/S ShakeMap HUD
- `g3vm/studio/g3vm_portable_studio.html`: Global 3D spherical Earth crust-to-core bent ray paths (`P`, `S`, `PmP`, `PcP`, `PKIKP`)
- `bmhd/studio/bmhd_portable_studio.html`: 3D non-symmetric stellarator flux surfaces, Poincaré maps, and MRxMHD solver
- `oatse/studio/oatse_portable_studio.html`: 3D/4D SOFAR channel ray-fan, thermocline, and SAS seabed impedance studio
- `astrosnail/studio/astrosnail_portable_studio.html`: Time-domain light-curve triage, crowded-field astrometry, and 3D cosmic-web studio
- `cellsnail/studio/cellsnail_portable_studio.html`: Synchronized chromosomal CNV and single-cell genomic workstation

### Colab GPU Notebooks
Self-contained `.ipynb` notebooks in `darksnail/notebooks/` for reproducible remote GPU execution:
- `darksnail/notebooks/darksnail_v2_colab.ipynb`: ARC-AGI-1 & LongBench evaluation with `google/gemma-4-31B-it`
- `darksnail/notebooks/darksnail_bridge_aware_r5_colab.ipynb`: Bridge-Aware sparse router calibration at $r \ge 5$ (`< 35%` KV budget)
- `darksnail/notebooks/darksnail_arc3_colab.ipynb`: ARC-AGI-3 interactive multi-step world-model evaluation

---

## 8. Citation

```bibtex
@software{projectdarksnail_2026,
  author       = {Cohen, Noah},
  title        = {Project DarkSnail: Bi-Hemispheric x Ultrametric Triad Synthesis},
  year         = {2026},
  version      = {0.2.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23497692},
  url          = {https://doi.org/10.5281/zenodo.23497692}
}
```
