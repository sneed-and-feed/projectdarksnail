# Project DarkSnail V2 -> Google Gemma 4 31B Adaptation Specification

**Target Backbone:** `google/gemma-4-31B` / `google/gemma-4-31B-it` ([arXiv:2607.02770](https://arxiv.org/abs/2607.02770), [HuggingFace `config.json`](https://huggingface.co/google/gemma-4-31B/blob/main/config.json))  
**Source Architecture:** `sneed-and-feed/projectdarksnail` (`CC0-1.0` Public Domain)  
**Deliverables:** `darksnail_gemma4.py`, `gemini-code-1792152508835.py`, `test_darksnail_gemma4.py`

## 1. Backbone Structural Mapping: `Meta-Llama-3.1-8B-Instruct` vs. `google/gemma-4-31B`

| Architectural Axis | `Meta-Llama-3.1-8B-Instruct` (Original) | `google/gemma-4-31B` (`arXiv:2607.02770`) | DarkSnail V2 Gemma 4 31B Adaptation |
| :--- | :--- | :--- | :--- |
| **Decoder Topology** | `32` homogeneous full-attention layers | `60` heterogeneous layers (`5:1` Local/Global interleave) | **Selective Global Surgery (`only_global_layers=True`)**: Leave all `50` Local Sliding-Window (`W=1024`) layers untouched; patch strictly the `10` Global Full-Attention layers at indices `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`. |
| **Residual Stream (`d_model`)** | `4,096` (`intermediate_size = 14,336`) | `5,376` (`intermediate_size = 21,504`, `embed_scale = sqrt(5376)`) | **`Gemma4BeltramiCallosalShell` (`5376 -> 4096` Isometric Bridge)** + **`DarkSnailGemma4GlobalHook` (`5376 <-> 64`)**: Native extraction on `5376`-D plus deterministic row-orthonormal projection to `(B, 32, 4096)` `float32` so frozen `brain-ai` `CallosalConditioning` (`3.68M` params) requires zero re-training. |
| **Callosal Extraction Layer** | Layer `16` (`16 / 32 = 50%` depth) | **Layer `29`** (`5th` Global layer, `30 / 60 = 50%` depth) | **`attach_gemma4_beltrami_hook(model, layer_idx=29)`**: Layer `16` in Gemma 4 is a `1024`-token local sliding layer (`16 % 6 == 4`); hooking Global Layer `29` restores full `256K`-context task visibility. Any legacy call with `layer_idx=16` auto-snaps to `29`. |
| **Attention Heads & GQA** | `H_q = 32`, `H_kv = 8` (`G = 4`), `head_dim = 128` | **Local (50):** `H_q = 32`, `H_kv = 16` (`G = 2`), `head_dim = 256`<br>**Global (10):** `H_q = 32`, `H_kv = 4` (`G = 8`), `global_head_dim = 512` | **`SurgicalGemma4Attention`**: Per-layer heterogeneous geometry awareness (`head_dim = 512`, `num_kv_heads = 4`, `num_kv_groups = 8`, `scale = 1.0`). |
| **QK/V Norms & `attention_k_eq_v`** | No QK/V norms (`scale = 1/sqrt(128)`), separate `k_proj` and `v_proj` | `q_norm = RMSNorm(512)`, `k_norm = RMSNorm(512)`, `v_norm = RMSNormNoScale(512)`, `scale = 1.0`.<br>**Global layers:** `attention_k_eq_v = True` (`v_proj = None`, `v = v_norm(k_raw)` before `k_norm` and `p-RoPE`). | **Unified `K = V` Pruning & Pre-RoPE Value Path**: Single `[B, N]` keep mask computed on `k_nope` prunes both `K` and `V` (and propagates to `shared_kv_states` when `store_full_length_kv=True`) while preserving `v = v_norm(k_raw)` prior to `k_norm` and `p-RoPE`. |
| **Rotary Encoding (`p-RoPE`)** | Full `128`-D RoPE (`theta = 500,000`) | **Local:** Full `256`-D RoPE (`theta = 10,000`)<br>**Global:** **Proportional RoPE (`p-RoPE`)**, `partial_rotary_factor = 0.25`, `theta = 1,000,000.0` (`128` rotated channels, `384` unrotated channels). | **`ProportionalRoPESubspaceSplitter`**: Supports both HuggingFace's `rotate_half` split-half layout (`[0:64] U [256:320]` rotated vs. `[64:256] U [320:512]` unrotated) and contiguous prefix layout (`[0:128]` vs. `[128:512]`). Routes and computes Beltrami flux strictly on `k_nope` (`384`-D). |
| **Layer Norm Pattern** | 2-norm Pre-LN | **4-norm Sandwich Norm + `layer_scalar`** (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`) | Preserves all 4 Sandwich RMSNorms and `layer_scalar` buffer; `callosal_up` is zero-initialized for exact Step-0 System 1 homotopy (`Theorem 2.4`). |

## 2. Subtle Gemma 4 31B Pitfalls Resolved in This Adaptation

### 2.1 The HuggingFace `rotate_half` p-RoPE Channel Split (`[0:64] U [256:320]`)
In HuggingFace [`modeling_rope_utils._compute_proportional_rope_parameters`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py), `partial_rotary_factor = 0.25` on `global_head_dim = 512` computes `rope_angles = 64` non-zero frequencies and pads `nope_angles = 192` zeros into `inv_freq` (length `256`), then duplicates `emb = cat([freqs, freqs], dim=-1)` (length `512`). Because `apply_rotary_pos_emb` uses `rotate_half(x)` (which pairs `x[..., :256]` with `x[..., 256:]`):
* Under HuggingFace's native `Gemma4RotaryEmbedding` (`prope_layout="hf_split_half"`), the `128` rotated channels are **`[0:64]` and `[256:320]`**, while the `384` unrotated channels are **`[64:256]` and `[320:512]`**.
* Under contiguous prefix slicing (`prope_layout="contiguous_prefix"`, used in `gemini-code-1792152508835.py`), the `128` rotated channels are **`[0:128]`** and the `384` unrotated channels are **`[128:512]`**.
* `ProportionalRoPESubspaceSplitter` supports both layouts losslessly (`||merge(*split(k)) - k||_inf == 0.0`), guaranteeing `0.000e+00` positional phase drift on `k_nope` across the full `256K` context window.

### 2.2 `attention_k_eq_v = True` Pre-Norm / Pre-RoPE Value Branch
In [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py), Global layers set `v_proj = None`, but $V$ is **not** post-RoPE $K$:
1. Raw key projection: $K_{\mathrm{raw}} = W_K h \in \mathbb{R}^{B \times S \times 4 \times 512}$.
2. Value branch: $V = \mathrm{RMSNormNoScale}(K_{\mathrm{raw}})$ (unrotated, scale-free RMSNorm).
3. Key branch: $K = \mathrm{p\text{-}RoPE}\bigl(\mathrm{RMSNorm}(K_{\mathrm{raw}})\bigr)$.

`SurgicalGemma4Attention` preserves this exact split while `DarkSnailGemma4GlobalHook.forward_global_kv_hook` supports both zero-copy `v_pruned = k_pruned` (when `v_global is None`) and synchronized mask multiplication `v_pruned = v_global * keep_mask` (when `v_global` is passed explicitly).

### 2.3 Overcoming `k_norm` L2-Norm Flattening via `k_nope` Beltrami Flux
Because Gemma 4 applies per-head `k_norm = RMSNorm(512)` before p-RoPE, the full-vector L2 norm $\lVert k_t \rVert_2$ is nearly constant across tokens. `DarkSnailGemma4GlobalHook` combines:
1. **Subspace Spectral Energy** $\lVert k_{\mathrm{nope}, t} \rVert_2$ (measuring how much of the fixed `RMSNorm` energy budget is concentrated in the 75% semantic subspace vs. the 25% rotary carrier), and
2. **Beltrami Adjacent Jump Flux & Cross-Shell Curl** $\lVert k_{\mathrm{nope}, t} - k_{\mathrm{nope}, t-1} \rVert_2 \cdot \sqrt{1 - \langle \hat u_{\mathrm{coarse}, t}, \hat u_{\mathrm{fine}, t} \rangle^2}$ shielded by a 1D max-pooling neighborhood filter, which remains sharp under per-head `RMSNorm`.

## 3. Drop-In Integration into `projectdarksnail`

1. **Copy Module:** Place `darksnail_gemma4.py` into `projectdarksnail/darksnail/gemma4.py` (and re-export from `darksnail/__init__.py`).
2. **Patch & Hook `google/gemma-4-31B` for LongBench / NIAH / ARC-AGI:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from darksnail_gemma4 import prepare_darksnail_gemma4_model
   from darksnail.longbench import set_long_context_condition

   model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-4-31B-it",
       torch_dtype=torch.bfloat16,
       device_map="auto",
   )
   # Patches only the 10 Global layers [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
   # and attaches the Layer-29 (5376 -> 4096) BeltramiCallosalShell hook
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
3. **Standalone Surgical Hook Usage (`gemini-code-1792152508835.py` Contract):**
   ```python
   from darksnail_gemma4 import DarkSnailGemma4GlobalHook

   hook = DarkSnailGemma4GlobalHook(head_dim=512, d_model=5376, bridge_dim=64, p_rope_ratio=0.25)
   k_pruned, v_pruned, hidden_out, keep_mask = hook.forward_global_kv_hook(
       k_global=k_global,          # [B, 4, N, 512]
       hidden_states=hidden_states, # [B, N, 5376]
       dt=0.1,
       steps=4,
   )
   ```
