import torch.nn as nn
from .surgery import SurgicalLlamaAttention


def _discover_decoder_layers(model):
    """
    Discovers the decoder layer ModuleList across standard and multimodal-wrapped
    HuggingFace causal LM architectures (Llama, Qwen2/3, Qwen3.5/3.6, Gemma2/3/4, Mistral).
    """
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        if hasattr(m, "layers"):
            return m.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        f"Could not locate decoder `.layers` on model of type {type(model).__name__}"
    )


def _safe_cfg_attr(cfg, attr: str, default=None, layer_idx: int = None):
    """Safely read config attributes on both homogeneous and heterogeneous (Gemma 4) configs."""
    if layer_idx is not None:
        per_layer = getattr(cfg, "per_layer_config", None)
        if per_layer is not None and 0 <= layer_idx < len(per_layer):
            try:
                val = getattr(per_layer[layer_idx], attr, None)
                if val is not None:
                    return val
            except Exception:
                pass
    try:
        val = getattr(cfg, attr, default)
        return default if val is None else val
    except Exception:
        d = getattr(cfg, "__dict__", {})
        if attr in d and d[attr] is not None:
            return d[attr]
        return default


def patch_llama_model(
    model,
    tree_depth: int = 4,
    arity: int = 2,
    tau_init: float = 1.0,
    preserve_sinks: bool = True,
    req_depth: int = None,
    max_dist: int = None,
    init_mode: str = "collapse",
    qk_mode: bool = False,
    prime_arities = None,
    swin_shift: bool = False,
    route_stride: int = 1,
    distill_kl: bool = False,
    target_density: float = 0.25,
    route_block: int = 128,
    only_global_layers: bool = False,
    triad_mode: bool = False,
    d_shell: int = 8,
    **kwargs
):
    """
    Surgically injects Dynamic Topology Routers into a pre-trained causal LM
    (Llama, Qwen2/3, Qwen3.5/3.6, Gemma2/3/4, Mistral), replacing self_attn layers
    with SurgicalLlamaAttention and transferring pre-trained weights, scaling, and Q/K/V norms.
    """
    text_config = getattr(model.config, "text_config", model.config)
    configs = [model.config] if text_config is model.config else [model.config, text_config]

    for cfg in configs:
        setattr(cfg, "surgical_p", arity)
        setattr(cfg, "surgical_tree_mode", kwargs.get("tree_mode", False))
        setattr(cfg, "surgical_tree_depth", tree_depth)
        setattr(cfg, "surgical_tau_init", tau_init)
        setattr(cfg, "surgical_tau", kwargs.get("tau", tau_init))
        setattr(cfg, "preserve_sinks", preserve_sinks)
        setattr(cfg, "surgical_init_mode", init_mode)
        setattr(cfg, "surgical_qk_mode", qk_mode)
        setattr(cfg, "surgical_prime_arities", prime_arities)
        setattr(cfg, "surgical_swin_shift", swin_shift)
        setattr(cfg, "surgical_route_stride", route_stride)
        setattr(cfg, "surgical_distill_kl", distill_kl)
        setattr(cfg, "surgical_target_density", target_density)
        setattr(cfg, "surgical_route_block", route_block)
        setattr(cfg, "surgical_triad_mode", triad_mode)
        setattr(cfg, "surgical_d_shell", d_shell)
        if "local_window" in kwargs:
            setattr(cfg, "surgical_local_window", kwargs["local_window"])
        if "attention_backend" in kwargs:
            setattr(cfg, "surgical_attention_backend", kwargs["attention_backend"])
        if req_depth is not None:
            setattr(cfg, "surgical_req_depth", req_depth)
        if max_dist is not None:
            setattr(cfg, "surgical_max_dist", max_dist)

    model._surgical_route_cache = {}
    layers = _discover_decoder_layers(model)
    group_counters = {}

    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue
        old_attn = layer.self_attn
        is_kv_shared = bool(getattr(old_attn, "is_kv_shared_layer", False))
        if not hasattr(old_attn, "q_proj") or not hasattr(old_attn, "o_proj"):
            # Skip non-standard / linear-attention layers in hybrid architectures (e.g. Qwen3_5GatedDeltaNet)
            continue
        if not hasattr(old_attn, "k_proj") and not is_kv_shared:
            continue
        if only_global_layers:
            is_sliding = getattr(old_attn, "is_sliding", False) or (
                getattr(layer, "attention_type", None) == "sliding_attention"
            ) or (
                getattr(old_attn, "layer_type", None) == "sliding_attention"
            )
            if is_sliding:
                continue

        # Infer per-layer head_dim, num_heads, and num_kv_heads from old_attn / weights / per_layer_config
        q_out, _ = old_attn.q_proj.weight.shape
        o_in = old_attn.o_proj.weight.shape[1]

        head_dim = getattr(old_attn, "head_dim", None)
        if head_dim is None:
            nh_cfg = _safe_cfg_attr(text_config, "num_attention_heads", 1, layer_idx=i)
            head_dim = _safe_cfg_attr(text_config, "head_dim", o_in // max(1, nh_cfg), layer_idx=i)

        # Note: o_proj input dimension is always num_heads * head_dim (even when q_proj is 2x for gated attention)
        num_heads = getattr(old_attn, "num_heads", None)
        if num_heads is None:
            num_heads = o_in // max(1, head_dim)

        num_kv_heads = getattr(old_attn, "num_key_value_heads", None)
        if num_kv_heads is None:
            if hasattr(old_attn, "k_proj") and old_attn.k_proj is not None:
                k_out, _ = old_attn.k_proj.weight.shape
                num_kv_heads = k_out // max(1, head_dim)
            else:
                kv_groups = getattr(old_attn, "num_key_value_groups", None)
                if kv_groups is not None and kv_groups > 0:
                    num_kv_heads = max(1, num_heads // kv_groups)
                else:
                    num_kv_heads = _safe_cfg_attr(text_config, "num_key_value_heads", num_heads, layer_idx=i)

        group_key = (num_heads, head_dim)
        patched_idx = group_counters.get(group_key, 0)
        group_counters[group_key] = patched_idx + 1

        new_attn = SurgicalLlamaAttention(
            text_config,
            layer_idx=i,
            head_dim_override=head_dim,
            num_heads_override=num_heads,
            num_kv_heads_override=num_kv_heads,
        )
        new_attn.root_config = model.config
        new_attn.patched_layer_idx = patched_idx
        new_attn._route_cache_ref = model._surgical_route_cache

        # Preserve Gemma 4 KV-sharing and layer-type metadata
        new_attn.is_kv_shared_layer = is_kv_shared
        new_attn.store_full_length_kv = bool(getattr(old_attn, "store_full_length_kv", False))
        new_attn.layer_type = getattr(old_attn, "layer_type", None)
        new_attn.is_sliding = bool(getattr(old_attn, "is_sliding", False))
        new_attn.sliding_window = getattr(old_attn, "sliding_window", None)
        new_attn.is_causal = bool(getattr(old_attn, "is_causal", True))

        # Reuse exact projection modules/weights (supports gated 2x q_proj, k_eq_v, and KV-shared layers)
        if q_out != new_attn.q_proj.weight.shape[0]:
            new_attn.q_proj = nn.Linear(
                old_attn.q_proj.in_features,
                old_attn.q_proj.out_features,
                bias=old_attn.q_proj.bias is not None,
            )
        new_attn.q_proj.weight = old_attn.q_proj.weight
        if hasattr(old_attn, "k_proj") and old_attn.k_proj is not None:
            new_attn.k_proj.weight = old_attn.k_proj.weight
        else:
            new_attn.k_proj = None
        if hasattr(old_attn, "v_proj") and old_attn.v_proj is not None:
            new_attn.v_proj.weight = old_attn.v_proj.weight
        else:
            new_attn.v_proj = None
        new_attn.o_proj.weight = old_attn.o_proj.weight

        # Copy projection biases if present
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            old_proj = getattr(old_attn, proj_name, None)
            new_proj = getattr(new_attn, proj_name, None)
            if old_proj is not None and new_proj is not None and hasattr(old_proj, "bias") and old_proj.bias is not None:
                new_proj.bias = old_proj.bias

        # Preserve q_norm / k_norm / v_norm and custom scaling if present (e.g., Qwen3/3.5, Gemma3/4)
        new_attn.q_norm = getattr(old_attn, "q_norm", None)
        new_attn.k_norm = getattr(old_attn, "k_norm", None)
        new_attn.v_norm = getattr(old_attn, "v_norm", None)
        new_attn.scale = getattr(old_attn, "scaling", new_attn.scale)

        # Move to the same device and dtype, but keep router parameters in float32
        # so +/-5.0 homotopy biases never suffer from bfloat16 ULP stagnation (ULP(5.0) = 0.03125)!
        new_attn.to(old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
        new_attn.router.float()

        # Replace the module
        layer.self_attn = new_attn

    return model


def inject_surgery(model, **kwargs):
    """
    Iterates over decoder layers, replaces self_attn with SurgicalLlamaAttention,
    and copies over the pre-trained q_proj, k_proj, v_proj, o_proj weights.
    """
    return patch_llama_model(model, **kwargs)


def save_router_sidecar(model, path: str, extra_metadata: dict = None) -> dict:
    """
    Saves all DynamicTopologyRouter parameters and Dual Lagrangian density_lambda buffers
    from a surgically-patched model into a compact `.safetensors` sidecar file (~15-40 MB),
    embedding the surgical topology configuration into the safetensors header metadata.
    When `route_stride > 1`, automatically syncs trained anchor router weights to shared
    non-anchor layers so sidecars never export dead zero weights on odd layers.
    """
    import json
    import os
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    layers = _discover_decoder_layers(model)
    tensors = {}
    patched_layer_indices = []
    anchor_routers = {}

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "router"):
            continue
        patched_layer_indices.append(i)
        group_key = (attn.num_heads, attn.head_dim)
        stride = max(1, int(getattr(attn, "route_stride", 1)))
        p_idx = int(getattr(attn, "patched_layer_idx", i))
        w_norm = attn.router.route_heads.weight.detach().float().norm().item()
        if (p_idx % stride == 0) or (w_norm > 1e-6):
            anchor_routers[group_key] = attn.router.state_dict()
        elif group_key in anchor_routers:
            # Sync non-anchor layer with trained anchor router from the same (H, D) group
            attn.router.load_state_dict(anchor_routers[group_key])

        for k, v in attn.router.state_dict().items():
            tensors[f"layers.{i}.self_attn.router.{k}"] = v.detach().contiguous().cpu()
        if hasattr(attn, "density_lambda") and isinstance(attn.density_lambda, torch.Tensor):
            tensors[f"layers.{i}.self_attn.density_lambda"] = attn.density_lambda.detach().contiguous().cpu()

    if not tensors:
        raise RuntimeError("No SurgicalLlamaAttention routers found on model to save.")

    cfg = model.config
    meta = {
        "format": "sparse-ai-router-sidecar-v4",
        "model_type": str(getattr(cfg, "model_type", type(model).__name__)),
        "name_or_path": str(getattr(cfg, "_name_or_path", "")),
        "patched_layers": json.dumps(patched_layer_indices),
        "surgical_tree_depth": str(getattr(cfg, "surgical_tree_depth", 5)),
        "surgical_p": str(getattr(cfg, "surgical_p", 2)),
        "surgical_qk_mode": str(bool(getattr(cfg, "surgical_qk_mode", False))),
        "surgical_prime_arities": json.dumps(getattr(cfg, "surgical_prime_arities", None)),
        "surgical_swin_shift": str(bool(getattr(cfg, "surgical_swin_shift", False))),
        "surgical_route_stride": str(getattr(cfg, "surgical_route_stride", 1)),
        "surgical_route_block": str(getattr(cfg, "surgical_route_block", 128)),
        "surgical_triad_mode": str(bool(getattr(cfg, "surgical_triad_mode", False))),
        "surgical_d_shell": str(getattr(cfg, "surgical_d_shell", 8)),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            meta[str(k)] = str(v) if isinstance(v, str) else json.dumps(v)

    try:
        from safetensors.torch import save_file
        save_file(tensors, path, metadata=meta)
    except ImportError:
        torch.save({"tensors": tensors, "metadata": meta}, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[RouterSidecar] Saved {len(tensors)} tensors across {len(patched_layer_indices)} layers to {path} ({size_mb:.2f} MB)")
    return meta


def load_router_sidecar(model, path: str, strict: bool = True) -> dict:
    """
    Loads a `.safetensors` router sidecar into a surgically-patched model.
    Returns the metadata dictionary stored in the sidecar header.
    """
    import torch

    meta = {}
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        tensors = load_file(path, device="cpu")
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        tensors = ckpt.get("tensors", ckpt)
        meta = ckpt.get("metadata", {})

    layers = _discover_decoder_layers(model)
    loaded_layers = 0
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "router"):
            continue
        prefix = f"layers.{i}.self_attn.router."
        router_sd = {
            k[len(prefix):]: v.to(device=next(attn.router.parameters()).device, dtype=next(attn.router.parameters()).dtype)
            for k, v in tensors.items()
            if k.startswith(prefix)
        }
        if router_sd:
            attn.router.load_state_dict(router_sd, strict=strict)
            loaded_layers += 1
        lam_key = f"layers.{i}.self_attn.density_lambda"
        if lam_key in tensors and hasattr(attn, "density_lambda"):
            attn.density_lambda.copy_(tensors[lam_key].to(attn.density_lambda.device, dtype=attn.density_lambda.dtype))

    print(f"[RouterSidecar] Loaded pre-warmed router weights into {loaded_layers} layers from {path}")
    return meta

