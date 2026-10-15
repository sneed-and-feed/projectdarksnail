# Ultrametric AI v3: The "Galaxy Brain" Roadmap

This roadmap formalizes the third generation of the Dynamic Ultrametric Attention architecture. Having established 11.5x to 28x speedups via block-sparse Triton kernels and Multi-Prime Adèlic routing in V1 and V2, V3 targets zero-overhead self-speculation, $O(\log N)$ context compression, deterministic MoE routing, continuous sparsification of dense LLMs, and multi-dimensional spatial hierarchies for vision models.

## The Core Philosophy: A Compiler for Learned Structure

While V1 and V2 focused heavily on the specific geometry of ultrametric trees, the true breakthrough of this architecture is broader: **differentiable topology discovery compiled directly into hardware execution**. 

The $p$-adic tree is merely one instance of a routing topology. By relaxing discrete routing choices into a continuous Gumbel-Sigmoid manifold, polarizing them into hard binary masks, and mapping those masks directly into $O(N \log N)$ block-sparse Triton coordinate lists, we have built a framework where a neural network dynamically writes its own optimal hardware execution graph at runtime. Whether the underlying structure is an adèlic tree, a learned graph neighborhood, or a spatial Quad-Tree, the compiler remains identical: the model learns the shape of the data, and the hardware dynamically conforms to that shape.

---

## 1. Llama Surgery (Continual Sparsification)
**[x] COMPLETED:** Surgically injected block-sparse routing into Llama-3.1-8B via PyTorch dynamic hooks. Resolved gradient collapse via Straight-Through Estimators, anchored generation syntax with Attention Sinks (Token 0), and achieved massive acceleration using a custom block-sparse Triton kernel heavily pipelined for Ampere/Hopper (A100/H100) Tensor Memory Accelerators.

Standard from-scratch gating fails due to "routing absorption" (Aquino-Michaels, 2026), where Q/K/V projections co-adapt to nullify the routing bias. To safely transition a dense model from $O(N^2)$ dense to $O(N \log N)$ sparse attention, we deploy a **Continuous Logit Homotopy** (inspired by Savarese et al., 2020).

### Mathematical Formulation
For each attention layer $\ell$, we inject a `MultiPrimeTopologyRouter`. The modified attention equation is:

```math
\mathrm{Attn}_\ell(Q, K, V) = \mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + M_{\mathrm{causal}} + \sum_{p \in \mathcal{P}} g_{\ell,h}^{(p)} \cdot \alpha^{(p)} \cdot T_{ij}^{(p)} \right) V
```

Where $T_{ij}^{(p)}$ is the tree distance matrix, and the Gumbel-Sigmoid gate is $g \in [0, 1]$.

**Surgical Initialization:** To achieve a mathematically guaranteed zero-shock insertion, the router's logit projection bias $z$ is initialized to $\mu_{\mathrm{init}} \ll 0$ (e.g., -5.0). Thus, $\mathbb{E}[g] \approx 0$, making the injected tree bias negligibly small and preserving the pre-trained manifold identically at step 0.

### Training Curriculum
We optimize the joint loss:

```math
\mathcal{L}(t) = \mathcal{L}_{\mathrm{CE}}(\Theta_{\mathrm{frozen}}, \Phi) + \lambda(t) \cdot \frac{1}{L \cdot H \cdot |\mathcal{P}|} \sum_{\ell, h, p} (1 - g_{\ell,h}^{(p)})
```

- **Auxiliary Loss Ramp-up $\lambda(t)$:** Slowly monotonically increases the penalty for density.
- **Temperature Annealing $\tau(t)$:** Anneals from $\tau_{\max} = 1.0$ down to $\tau_{\min} = 0.1$, forcing the continuous gates to polarize into hard binary decisions $g \in \{0, 1\}$.

Once polarized, the routing depths $d_h = \mathbb{1}[z > 0]$ are extracted and mapped directly into the Triton block-sparse kernel.

### Downstream Evaluations (Next Steps)
While raw perplexity validates that the pre-trained manifold is preserved, the critical next frontier is formal benchmarking on precise long-range retrieval and reasoning tasks:
- **Needle In A Haystack (NIAH):** Theoretically, the $p$-adic tree perfectly isolates the "needle". Because the needle and the query share highly similar semantic embeddings, the router will deterministically map them to the same branch on the Bruhat-Tits tree. The needle never has to compete with the dense noise of the haystack for attention mass.
- **ZeroSCROLLS & Multi-hop QA:** Validating multi-document reasoning and summarization where information is sparsely distributed across a 100k+ context.
- **Long-Context Code (RepoBench):** Validating cross-file syntactic routing where distant function definitions must map to local invocations.

---

## 2. Zero-Overhead Self-Speculative Decoding
**Goal:** Utilize the dynamic structural topology for speculative decoding using a single model.

By dynamically toggling the `req_depth` parameter in the Triton kernel, the *same model* acts as both the drafter and the verifier, requiring zero extra parameters and zero additional VRAM.
- **Drafter:** Executes with `req_depth=4` (maximum sparsity, skipping $\sim 95\%$ of blocks, 8x faster).
- **Verifier:** Executes with `req_depth=0` (dense, full KV cache processing for verification).

### Algorithmic Speedup Bounds
Generate $K$ draft tokens autoregressively with $Q(\cdot)$, then verify in parallel with $P(\cdot)$. 
For rejection sampling, a token is accepted if $u \le \min\left(1, \frac{P(x)}{Q(x)}\right)$. 
Given marginal acceptance probability $\alpha$, expected tokens per step is $\mathbb{E}[N] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$.

Let $C_{\mathrm{sparse}} \approx \frac{1}{8} C_{\mathrm{dense}}$. The expected theoretical wall-clock speedup $S$ is:

```math
S = \frac{\mathbb{E}[N]}{1 + K \left(\frac{C_{\mathrm{sparse}}}{C_{\mathrm{dense}}}\right)}
```

Assuming $K=4$ and $\alpha=0.75$, $S = \frac{2.73}{1.5} \approx 1.82\times$ speedup. 
If $\alpha \to 0.9$ (due to local sliding windows matching exact Markovian grammar), $S \to 2.72\times$.

---

## 3. Ultrametric Mixture of Experts (U-MoE)
**Goal:** Use the exact same ultrametric routing vector $r_m$ to dictate sparse computation.

Standard MoE uses a flat, unstable `Softmax(Wx)` router. U-MoE replaces this with **Topological Token Assignments**. The Gumbel-Softmax gates in the attention router are already load-balanced to populate the tree, making the MoE routing "free" (zero overhead) and topologically deterministic.

### Adèlic Expert Assignment
If a tree has depth $D$, we define $E = p^k$ experts ($k \le D$). 
A token $m$ with routing vector $r_m = (c_{m,0}, c_{m,1}, \dots, c_{m,D-1})$ routes to expert $e(m)$:

```math
e(m) = \sum_{j=0}^{k-1} c_{m,j} p^{k-1-j}
```

Experts sharing a deep Lowest Common Ancestor (LCA) in the $p$-adic tree process semantically similar tokens. This eliminates parameter symmetries and routing collapse.

### Block-Sparse Dispatch Kernel
A custom Triton kernel handles the deterministic bucketing:
1. Execute prefix sum on the $k$-bit prefix of $r_m$.
2. Parallel Triton kernels load `expert_offsets` and run tiled GEMMs strictly on assigned blocks.
3. Scatter output back to original sequences via inverse permutation.

---

## 4. Infinite Context via Adèlic KV-Fusion ($O(\log N)$ Memory)
**Goal:** Break the $O(N)$ physical memory constraint of standard KV caches.

When a block exits the local sliding window $W$, we algorithmically fuse it into its parent node in the ultrametric tree.

### Fractal Storage Bounds
For arity $p$, when $p$ adjacent blocks exit the sliding window at depth $d$, they are compressed:

```math
K^{(d-1)}_{j} = f_\theta \left( K^{(d)}_{p \cdot j}, \dots, K^{(d)}_{p \cdot j + (p-1)} \right)
```

This recursively ascends the tree hierarchy. The total active footprint is bounded by $W$ at each depth. 
Maximum depth is $\lceil \log_p (N / B) \rceil$, yielding $O(\log N)$ KV memory growth. 
At 1M tokens ($B=128, W=4, p=2$), standard memory uses $\approx 7800$ blocks, while Adèlic KV-Fusion requires just $52$ blocks (a 150x reduction).

### The Fusion Operator $f_\theta$
- **Averaging:** Fast $O(1)$ mean pooling.
- **Attention-Weighted Merge (ToMe-style):** Weight tokens based on aggregate attention scores before eviction, preserving "attention sinks" losslessly.

The Triton kernel is modified to directly map the query's `req_depth` to the physical depth level of the hierarchical `tree_cache`, completely eliminating dynamic branching.

---

## 5. 2D/3D Multi-Prime Topologies (Vision Transformers)
**Goal:** Apply Adèlic routing to continuous spatial domains (images/video) to eliminate the quadratic bottleneck.

Standard Quad-Trees suffer from "spatial blind spots." By applying the Adèlic product formula spatially, we overlap multiple co-prime trees (e.g., $2\times 2$, $3\times 3$, $5\times 5$). Their union mathematically guarantees the elimination of all blind spots.

### Spatially-Adaptive Routing
We transition to a per-block spatially-adaptive depth gate $g(m) \in [0, D]$.
- **Homogeneous Backgrounds (Ultra-Sparse):** Learn max depth $D$. They only attend to structurally identical local patches.
- **High-Frequency Foregrounds (Dense):** Learn shallow depth $0$ (the root). They attend globally to track moving objects or cross-image semantics.

### Triton Kernel Block-Sparse Logic
For a base $q$, the coordinate space is quantized into $q^D$ grid cells. 
The LCA computation in Triton avoids complex Morton bit-interleaving by utilizing simple scale-factor quantization:
```python
# A block of size `q_base**(max_depth - req_depth)` must contain both coordinates.
scale_factor = q_base ** (max_depth - req_depth)
match_x = (m_x // scale_factor) == (n_x // scale_factor)
match_y = (m_y // scale_factor) == (n_y // scale_factor)

# If the topology constraint fails, skip the expensive HBM load entirely
if not (match_x and match_y):
    continue
```
In Video (3D, Oct-Tree $p=8$), temporal dimension $t$ is included in the quantization, allowing static background pixels to skip both spatial and temporal attention across frames, effectively resolving the sequence length explosion of video diffusion models.
