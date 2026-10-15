"""Long-Context Retrieval & QA Synthesis (LongBench QASPER + Banal NIAH).

Demonstrates the reverse direction of Project DarkSnail: how the Bi-Hemispheric
Callosal Shell (`BeltramiCallosalShell`), Zeno-Amygdala turbulence telemetry,
and Query-Tail Beltrami Outlier Shielding upgrade `Llama Surgery` (`V4 TriadTopologyRouter`)
on `Meta-Llama-3.1-8B-Instruct` at deep ultrametric tree sparsity (r = 2, 3, 4, 5)
without increasing the measured causal key budget.
"""

from __future__ import annotations

import json
import math
import os
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .shell import BeltramiCallosalShell


# ============================================================================
# Banal NIAH & LongBench QASPER Templates (matching sparse-ai benchmarks)
# ============================================================================

HAYSTACK_BASE = (
    "The geometry of language is not a linear sequence of events, but a complex, "
    "multifaceted web of relationships. When we speak or write, we traverse a "
    "topological space where concepts are linked not just by their adjacency in time, "
    "but by their semantic and syntactic depth. In standard dense attention, every token "
    "is forced to compare itself against every prior token, incurring quadratic memory "
    "and computational scaling. Dynamic tree routing decomposes this monolithic field into "
    "hierarchical branches of a Bruhat-Tits tree, enabling tokens to communicate with logarithmic "
    "efficiency. Recent advancements in p-adic analysis confirm that discrete prefix sharing "
    "is an exact combinatorial invariant for hardware block scheduling. "
)

NEEDLE_CONFIGS = {
    "outlier": {
        "needle_key": "The secret operative passcode is 'KRAKEN-7729'.",
        "query_text": "What is the secret operative passcode? The secret operative passcode is '",
        "answer_target": "KRAKEN-7729",
    },
    "banal": {
        "needle_key": "The primary coordinate system used for the manifold calculation is spherical coordinates.",
        "query_text": (
            "What is the primary coordinate system used for the manifold calculation? "
            "The primary coordinate system used for the manifold calculation is "
        ),
        "answer_target": "spherical coordinates",
    },
}

QASPER_PROMPT = (
    "You are given a scientific article and a question. Answer the question as concisely "
    "as you can, using a single phrase or sentence if possible. If the question cannot be "
    'answered based on the information in the article, write "unanswerable". If the '
    'question is a yes/no question, answer "yes", "no", or "unanswerable". Do not '
    "provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the "
    "above article as concisely as you can, using a single phrase or sentence if possible. "
    "If the question cannot be answered based on the information in the article, write "
    '"unanswerable". If the question is a yes/no question, answer "yes", "no", or '
    '"unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:'
)


# ============================================================================
# Metric & Dataset Helpers
# ============================================================================

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(w for w in s.split() if w not in {"a", "an", "the"})
    return " ".join(s.split())


def qa_f1(pred: str, ref: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(ref).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2.0 * prec * rec / (prec + rec)


def score_qasper_f1(pred: str, answers: Sequence[str]) -> float:
    pred_first_line = pred.lstrip("\n").split("\n")[0]
    if not answers:
        return 0.0
    return max(qa_f1(pred_first_line, a) for a in answers)


def bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> List[float]:
    if not xs:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def build_niah_context(
    tokenizer,
    target_len: int = 4096,
    needle_depth_ratio: float = 0.5,
    needle_type: str = "banal",
) -> Tuple[torch.Tensor, str, int]:
    """Construct an exact-length context containing a needle at `needle_depth_ratio`."""
    cfg = NEEDLE_CONFIGS.get(needle_type, NEEDLE_CONFIGS["banal"])
    needle_text = f" {cfg['needle_key']} "
    query_text = f"\n\n{cfg['query_text']}"
    answer_target = cfg["answer_target"]

    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)

    filler_tokens_needed = target_len - len(needle_ids) - len(query_ids)
    if filler_tokens_needed < 64:
        raise ValueError(f"target_len {target_len} is too short for needle + query.")

    base_ids = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)
    repeats = (filler_tokens_needed // len(base_ids)) + 2
    full_filler = (base_ids * repeats)[:filler_tokens_needed]

    insert_idx = int(filler_tokens_needed * needle_depth_ratio)
    insert_idx = max(10, min(filler_tokens_needed - 10, insert_idx))

    context_ids = full_filler[:insert_idx] + needle_ids + full_filler[insert_idx:] + query_ids
    input_ids = torch.tensor([context_ids], dtype=torch.long)
    return input_ids, answer_target, insert_idx


def load_qasper_dataset() -> List[Dict[str, Any]]:
    """Load LongBench QASPER test split (200 samples), with Hub zip fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "qasper", split="test", trust_remote_code=True)
        return [dict(x) for x in ds]
    except Exception:
        import zipfile
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("qasper.jsonl"))
            with z.open(name) as f:
                return [json.loads(line) for line in f]


def build_qasper_prompt_ids(tokenizer, sample: Dict[str, Any], max_ctx: int = 6000) -> Tuple[torch.Tensor, bool]:
    """LongBench convention: middle-truncate prompts exceeding `max_ctx` tokens."""
    prompt = QASPER_PROMPT.format(context=sample["context"], input=sample["input"])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    truncated = len(ids) > max_ctx
    if truncated:
        half = max_ctx // 2
        prompt = (
            tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True)
        )
    if getattr(tokenizer, "chat_template", None):
        ids_out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(ids_out, torch.Tensor):
            ids_out = ids_out["input_ids"]
    else:
        ids_out = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids_out, truncated


# ============================================================================
# Model Patching, Layer-16 Beltrami Hook, and Condition Management
# ============================================================================

def _resolve_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return getattr(model, "layers", [])


def attn_modules(model: nn.Module) -> List[nn.Module]:
    return [
        layer.self_attn
        for layer in _resolve_layers(model)
        if hasattr(layer, "self_attn") and layer.self_attn is not None
    ]


def attach_layer16_beltrami_hook(model: nn.Module, layer_idx: int = 16) -> BeltramiCallosalShell:
    """Attach a non-destructive Layer-16 BeltramiCallosalShell telemetry hook to `model`.

    On every prefill forward pass (`seq_len > 1`), extracts the `(B, 32, D)` callosal
    shell and caches live `zeno_flux` and `lca_variance` in `model._darksnail_layer16_telemetry`.
    """
    layers = _resolve_layers(model)
    if not layers:
        raise ValueError("Could not resolve transformer layers on model.")
    target_idx = min(layer_idx, len(layers) - 1)
    d_model = int(getattr(model.config, "hidden_size", 4096))

    if hasattr(model, "_darksnail_layer16_hook_handle") and model._darksnail_layer16_hook_handle is not None:
        model._darksnail_layer16_hook_handle.remove()

    shell = BeltramiCallosalShell(
        d_model=d_model,
        target_tokens=32,
        outlier_budget=16,
        condensed_budget=16,
        d_shell=8,
        shield_radius=2,
    ).to(next(model.parameters()).device)
    shell.eval()

    model._darksnail_beltrami_shell = shell
    model._darksnail_layer16_telemetry = {"zeno_flux": 0.0, "lca_variance": 0.0}

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor) and h.dim() == 3 and h.shape[1] > 1:
            with torch.no_grad():
                h_win = h[:, -min(256, h.shape[1]) :, :].detach()
                _, telemetry = shell(h_win, return_diagnostics=True)
                z_val = telemetry.get("zeno_flux", 0.0)
                l_val = telemetry.get("lca_variance", 0.0)
                model._darksnail_layer16_telemetry = {
                    "zeno_flux": float(z_val.mean().item() if isinstance(z_val, torch.Tensor) else z_val),
                    "lca_variance": float(l_val.mean().item() if isinstance(l_val, torch.Tensor) else l_val),
                }

    model._darksnail_layer16_hook_handle = layers[target_idx].register_forward_hook(_hook)
    return shell


def prepare_darksnail_llama_model(
    model: nn.Module,
    sidecar_path: Optional[str | Path] = None,
    attention_backend: str = "block_list",
    layer16_hook: bool = True,
) -> nn.Module:
    """Patch `model` with V4 Triad `SurgicalLlamaAttention`, load router sidecar, and attach Layer-16 hook."""
    from llama_surgery import inject_surgery, load_router_sidecar

    already_patched = any(hasattr(m, "_update_darksnail_shield") for m in attn_modules(model))
    if not already_patched:
        model = inject_surgery(
            model,
            tree_depth=5,
            arity=2,
            preserve_sinks=True,
            init_mode="collapse",
            tree_mode=False,
            qk_mode=True,
            prime_arities=[2, 3],
            swin_shift=True,
            route_stride=2,
            distill_kl=True,
            target_density=0.25,
            triad_mode=True,
            d_shell=8,
        )

    if sidecar_path is None:
        default_sidecar = (
            Path(__file__).resolve().parent.parent
            / "checkpoints"
            / "router_sidecars"
            / "llama31_8b_router_v3.safetensors"
        )
        if default_sidecar.exists():
            sidecar_path = default_sidecar

    if sidecar_path is not None and Path(sidecar_path).exists():
        load_router_sidecar(model, str(sidecar_path), strict=False)

    tcfg = getattr(model.config, "text_config", model.config)
    for cfg in ([model.config] if tcfg is model.config else [model.config, tcfg]):
        cfg.use_cache = True
        cfg.surgical_attention_backend = attention_backend

    if layer16_hook:
        attach_layer16_beltrami_hook(model, layer_idx=16)

    model.eval()
    return model


def set_long_context_condition(
    model: nn.Module,
    req_depth: int,
    *,
    darksnail_mode: bool = False,
    override: Optional[str] = None,
    window: Optional[int] = None,
    collect_stats: bool = True,
    attention_backend: Optional[str] = None,
    shield_k: int = 96,
    bridge_blocks: int = 4,
    decode_k: int = 64,
) -> None:
    """Configure attention routing condition on a surgically-patched model."""
    tcfg = getattr(model.config, "text_config", model.config)
    for cfg in ([model.config] if tcfg is model.config else [model.config, tcfg]):
        cfg.surgical_req_depth = int(req_depth)
        cfg.surgical_darksnail_mode = bool(darksnail_mode)
        cfg.surgical_mask_override = override
        cfg.surgical_window_size = window
        cfg.surgical_collect_stats = bool(collect_stats)
        cfg.surgical_darksnail_shield_k = int(shield_k)
        cfg.surgical_darksnail_bridge_blocks = int(bridge_blocks)
        cfg.surgical_darksnail_decode_k = int(decode_k)
        if attention_backend is not None:
            cfg.surgical_attention_backend = attention_backend

    if hasattr(model, "_surgical_route_cache") and isinstance(model._surgical_route_cache, dict):
        model._surgical_route_cache.clear()
    for m in attn_modules(model):
        m._darksnail_shield_tokens = None
        m._darksnail_token_salience = None


def collect_measured_budget(model: nn.Module) -> Tuple[Optional[float], Optional[float]]:
    """Return `(mean_allowed_keys, mean_causal_keys)` across layers from the last prefill."""
    allowed, causal = [], []
    for m in attn_modules(model):
        if getattr(m, "last_mean_allowed_keys", None) is not None:
            allowed.append(float(m.last_mean_allowed_keys))
            causal.append(float(m.last_mean_causal_keys))
            m.last_mean_allowed_keys = None
    if not allowed:
        return None, None
    return sum(allowed) / len(allowed), sum(causal) / len(causal)


# ============================================================================
# Banal NIAH & LongBench QASPER Head-to-Head Evaluators
# ============================================================================

@torch.no_grad()
def evaluate_banal_niah_comparison(
    model: nn.Module,
    tokenizer,
    *,
    context_len: int = 4096,
    num_positions: int = 10,
    depths: Sequence[int] = (3, 4, 5),
    needle_type: str = "banal",
    max_new_tokens: int = 12,
    include_window: bool = True,
) -> Dict[str, Any]:
    """Compare `window@r`, `v4_triad@r`, and `darksnail@r` on Banal NIAH (`spherical coordinates`)."""
    device = next(model.parameters()).device
    if num_positions <= 1:
        depth_ratios = [0.5]
    else:
        depth_ratios = [0.05 + 0.90 * (i / (num_positions - 1)) for i in range(num_positions)]

    summary: Dict[str, Any] = {
        "context_len": context_len,
        "num_positions": num_positions,
        "needle_type": needle_type,
        "depths": list(depths),
        "conditions": {},
    }

    for r in depths:
        cond_names = (["window"] if include_window else []) + ["v4_triad", "darksnail"]
        stats = {
            c: {"successes": 0, "total": num_positions, "budgets_pct": [], "allowed_keys": [], "zeno_flux": []}
            for c in cond_names
        }

        for ratio in depth_ratios:
            input_ids, target_str, _ = build_niah_context(
                tokenizer, target_len=context_len, needle_depth_ratio=ratio, needle_type=needle_type
            )
            input_ids = input_ids.to(device)
            prompt_len = input_ids.shape[1]
            clean_target = target_str.replace("'", "").strip().lower()

            # 1. Run standalone V4 Triad (routed@r)
            set_long_context_condition(
                model, req_depth=r, darksnail_mode=False, collect_stats=True, attention_backend="eager"
            )
            out_v4 = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            gen_v4 = tokenizer.decode(out_v4[0, prompt_len:], skip_special_tokens=True).strip().lower()
            pass_v4 = (clean_target in gen_v4) or (" " in clean_target and clean_target.split()[0] in gen_v4)
            allowed_v4, causal_v4 = collect_measured_budget(model)
            stats["v4_triad"]["successes"] += int(pass_v4)
            if allowed_v4 and causal_v4:
                stats["v4_triad"]["budgets_pct"].append(100.0 * allowed_v4 / causal_v4)
                stats["v4_triad"]["allowed_keys"].append(allowed_v4)

            # 2. Run DarkSnail V2 (darksnail@r) at matched/lower budget
            set_long_context_condition(
                model,
                req_depth=r,
                darksnail_mode=True,
                collect_stats=True,
                attention_backend="eager",
                shield_k=96,
                decode_k=64,
            )
            out_ds = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            gen_ds = tokenizer.decode(out_ds[0, prompt_len:], skip_special_tokens=True).strip().lower()
            pass_ds = (clean_target in gen_ds) or (" " in clean_target and clean_target.split()[0] in gen_ds)
            allowed_ds, causal_ds = collect_measured_budget(model)
            stats["darksnail"]["successes"] += int(pass_ds)
            if allowed_ds and causal_ds:
                stats["darksnail"]["budgets_pct"].append(100.0 * allowed_ds / causal_ds)
                stats["darksnail"]["allowed_keys"].append(allowed_ds)
            if hasattr(model, "_darksnail_layer16_telemetry"):
                stats["darksnail"]["zeno_flux"].append(model._darksnail_layer16_telemetry.get("zeno_flux", 0.0))

            # 3. Matched-budget sliding window baseline (window@r)
            if include_window:
                w_keys = max(1, int(round(allowed_v4 if allowed_v4 else context_len * (2.0 ** (-r)))))
                set_long_context_condition(
                    model,
                    req_depth=r,
                    darksnail_mode=False,
                    override="window",
                    window=w_keys,
                    collect_stats=True,
                    attention_backend="eager",
                )
                out_win = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                gen_win = tokenizer.decode(out_win[0, prompt_len:], skip_special_tokens=True).strip().lower()
                pass_win = (clean_target in gen_win) or (" " in clean_target and clean_target.split()[0] in gen_win)
                allowed_win, causal_win = collect_measured_budget(model)
                stats["window"]["successes"] += int(pass_win)
                if allowed_win and causal_win:
                    stats["window"]["budgets_pct"].append(100.0 * allowed_win / causal_win)
                    stats["window"]["allowed_keys"].append(allowed_win)

        for c in cond_names:
            st = stats[c]
            recall = 100.0 * st["successes"] / max(1, st["total"])
            b_pct = sum(st["budgets_pct"]) / len(st["budgets_pct"]) if st["budgets_pct"] else None
            a_keys = sum(st["allowed_keys"]) / len(st["allowed_keys"]) if st["allowed_keys"] else None
            z_flux = sum(st["zeno_flux"]) / len(st["zeno_flux"]) if st["zeno_flux"] else None
            summary["conditions"][f"{c}@{r}"] = {
                "r": r,
                "condition": c,
                "recall_pct": round(recall, 2),
                "successes": st["successes"],
                "total": st["total"],
                "measured_budget_pct": round(b_pct, 2) if b_pct is not None else None,
                "measured_allowed_keys": round(a_keys, 1) if a_keys is not None else None,
                "mean_zeno_flux": round(z_flux, 4) if z_flux is not None else None,
            }

    set_long_context_condition(model, req_depth=0, darksnail_mode=False)
    return summary


@torch.no_grad()
def evaluate_qasper_comparison(
    model: nn.Module,
    tokenizer,
    *,
    num_samples: int = 20,
    max_ctx: int = 6000,
    max_new_tokens: int = 64,
    depths: Sequence[int] = (2, 3, 4),
    include_window: bool = True,
    include_dense: bool = True,
    attention_backend: str = "block_list",
) -> Dict[str, Any]:
    """Compare `dense`, `window@r`, `v4_triad@r` (`routed@r`), and `darksnail@r` on LongBench QASPER."""
    device = next(model.parameters()).device
    data = load_qasper_dataset()[:num_samples]

    conds: List[str] = []
    if include_dense:
        conds.append("dense")
    for r in depths:
        if include_window:
            conds.append(f"window@{r}")
        conds.append(f"v4_triad@{r}")
        conds.append(f"darksnail@{r}")

    per_f1: Dict[str, List[float]] = {c: [] for c in conds}
    fracs: Dict[str, List[float]] = {c: [] for c in conds}
    zeno_logs: List[float] = []
    n_trunc = 0

    for sample in data:
        ids, trunc = build_qasper_prompt_ids(tokenizer, sample, max_ctx=max_ctx)
        n_trunc += int(trunc)
        ids = ids.to(device)
        prompt_len = ids.shape[1]

        def _gen() -> str:
            if hasattr(model, "_surgical_route_cache") and isinstance(model._surgical_route_cache, dict):
                model._surgical_route_cache.clear()
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

        if include_dense:
            set_long_context_condition(
                model, req_depth=0, darksnail_mode=False, collect_stats=True, attention_backend=attention_backend
            )
            pred_d = _gen()
            per_f1["dense"].append(score_qasper_f1(pred_d, sample["answers"]))
            collect_measured_budget(model)
            fracs["dense"].append(1.0)

        for r in depths:
            # 1. Standalone V4 Triad (routed@r)
            set_long_context_condition(
                model, req_depth=r, darksnail_mode=False, collect_stats=True, attention_backend=attention_backend
            )
            pred_v4 = _gen()
            per_f1[f"v4_triad@{r}"].append(score_qasper_f1(pred_v4, sample["answers"]))
            allowed_v4, causal_v4 = collect_measured_budget(model)
            if allowed_v4 and causal_v4:
                fracs[f"v4_triad@{r}"].append(allowed_v4 / causal_v4)

            # 2. DarkSnail V2 (darksnail@r)
            set_long_context_condition(
                model,
                req_depth=r,
                darksnail_mode=True,
                collect_stats=True,
                attention_backend=attention_backend,
                shield_k=96,
                bridge_blocks=4,
                decode_k=64,
            )
            pred_ds = _gen()
            per_f1[f"darksnail@{r}"].append(score_qasper_f1(pred_ds, sample["answers"]))
            allowed_ds, causal_ds = collect_measured_budget(model)
            if allowed_ds and causal_ds:
                fracs[f"darksnail@{r}"].append(allowed_ds / causal_ds)
            if hasattr(model, "_darksnail_layer16_telemetry"):
                zeno_logs.append(model._darksnail_layer16_telemetry.get("zeno_flux", 0.0))

            # 3. Matched-budget window@r (window override always runs in eager path)
            if include_window:
                w_keys = max(1, int(round(allowed_v4 if allowed_v4 else prompt_len * (2.0 ** (-r)))))
                set_long_context_condition(
                    model,
                    req_depth=r,
                    darksnail_mode=False,
                    override="window",
                    window=w_keys,
                    collect_stats=True,
                    attention_backend="eager",
                )
                pred_w = _gen()
                per_f1[f"window@{r}"].append(score_qasper_f1(pred_w, sample["answers"]))
                allowed_w, causal_w = collect_measured_budget(model)
                if allowed_w and causal_w:
                    fracs[f"window@{r}"].append(allowed_w / causal_w)

    set_long_context_condition(model, req_depth=0, darksnail_mode=False, attention_backend=attention_backend)

    results: Dict[str, Any] = {}
    for c in conds:
        vals = per_f1[c]
        fr = fracs[c]
        results[c] = {
            "f1": round(100.0 * sum(vals) / max(1, len(vals)), 2),
            "f1_ci95": [round(100.0 * x, 2) for x in bootstrap_ci(vals)],
            "n": len(vals),
            "measured_budget_pct": round(100.0 * sum(fr) / len(fr), 2) if fr else None,
        }

    for r in depths:
        ds_c = f"darksnail@{r}"
        v4_c = f"v4_triad@{r}"
        win_c = f"window@{r}"
        if ds_c in per_f1 and v4_c in per_f1:
            diffs_v4 = [a - b for a, b in zip(per_f1[ds_c], per_f1[v4_c])]
            results[ds_c]["minus_v4_triad_f1"] = round(100.0 * sum(diffs_v4) / max(1, len(diffs_v4)), 2)
        if ds_c in per_f1 and win_c in per_f1:
            diffs_w = [a - b for a, b in zip(per_f1[ds_c], per_f1[win_c])]
            results[ds_c]["minus_window_f1"] = round(100.0 * sum(diffs_w) / max(1, len(diffs_w)), 2)

    return {
        "n_samples": len(data),
        "n_truncated": n_trunc,
        "depths": list(depths),
        "mean_layer16_zeno_flux": round(sum(zeno_logs) / len(zeno_logs), 4) if zeno_logs else None,
        "results": results,
    }


def load_v4_baseline_references(repo_root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the archived 200-sample QASPER and 50-position Banal NIAH V4 Triad baseline JSONs."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    qasper_path = root / "results" / "baselines" / "qasper_router_summary_block_list_v4_triad.json"
    niah_path = root / "results" / "baselines" / "niah_sweep_banal_v4_triad.json"
    out: Dict[str, Any] = {}
    if qasper_path.exists():
        with open(qasper_path, "r", encoding="utf-8") as f:
            out["qasper_v4_200"] = json.load(f)
    if niah_path.exists():
        with open(niah_path, "r", encoding="utf-8") as f:
            out["banal_niah_v4_50"] = json.load(f)
    return out
