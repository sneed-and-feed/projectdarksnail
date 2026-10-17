"""
Project DarkSnail V2 — Full ARC-AGI-1 Evaluation & Ablation Suite
=================================================================

Evaluates the Bi-Hemispheric x Ultrametric Triad Synthesis (`darksnail.shell`,
`darksnail.readout`, `darksnail.antenna`) across the 400-task ARC-AGI-1 Public
Evaluation Suite, comparing against:
  1. System 1 Zero-Shot Baseline (`submission_s1_zero_shot.json`)
  2. Bi-Hemispheric System 2 Baseline (`submission_s2_adapted.json`)
  3. Project DarkSnail V2 (`DarkSnailCanvasReadout` + `ZenoAmygdalaController`
     + `compute_grid_prompt_zeno_flux`)
"""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
if (WORKSPACE_DIR / "brain-ai").exists():
    sys.path.insert(0, str(WORKSPACE_DIR / "brain-ai"))
if (WORKSPACE_DIR / "sparse-ai" / "src").exists():
    sys.path.insert(0, str(WORKSPACE_DIR / "sparse-ai" / "src"))

from darksnail.antenna import ZenoAmygdalaController
from darksnail.readout import DarkSnailCanvasReadout
from darksnail.shell import (
    compute_grid_prompt_zeno_flux,
    extract_beltrami_pullback_canvas,
    extract_grid_beltrami_shape_descriptors,
    infer_arc_output_shape_beltrami,
)


# ---------------------------------------------------------------------------
# Standalone Grid Encoding & Shape Inference Fallbacks (Colab-ready)
# ---------------------------------------------------------------------------

def encode_grid_to_trm_seq(
    grid: Sequence[Sequence[int]], pad_r: int = 0, pad_c: int = 0, side: int = 30
) -> np.ndarray:
    """Encodes a 2D ARC grid to TRM canonical sequence (PAD=0, EOS=1, colors 2..11)."""
    if not grid or len(grid) == 0:
        return np.zeros(side * side, dtype=np.uint8)
    g = np.asarray(grid, dtype=np.int32)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    if g.size == 0 or g.shape[0] == 0 or g.shape[1] == 0:
        return np.zeros(side * side, dtype=np.uint8)
    if g.shape[0] > side or g.shape[1] > side:
        g = g[:side, :side]
    nrow, ncol = g.shape
    pad_r = max(0, min(pad_r, side - nrow))
    pad_c = max(0, min(pad_c, side - ncol))
    g_tok = np.clip(g, 0, 9) + 2
    pad_after_r = max(0, side - pad_r - nrow)
    pad_after_c = max(0, side - pad_c - ncol)
    out = np.pad(g_tok, ((pad_r, pad_after_r), (pad_c, pad_after_c)), constant_values=0)
    if pad_r + nrow < side:
        out[pad_r + nrow, pad_c : pad_c + ncol] = 1
    if pad_c + ncol < side:
        out[pad_r : pad_r + nrow, pad_c + ncol] = 1
    return out.astype(np.uint8).flatten()


def infer_arc_output_shape(
    demos: Sequence[Mapping[str, Any]],
    test_input_grid: Sequence[Sequence[int]],
    side: int = 30,
) -> Tuple[int, int]:
    """Infers target output grid dimensions (H_out, W_out) strictly from demos and test input via Beltrami outlier & topological enclosure rules."""
    return infer_arc_output_shape_beltrami(demos, test_input_grid, side=side)


def is_demo_color_preserving(demos: Optional[Sequence[Mapping[str, Any]]]) -> bool:
    """Determines whether every demo output only uses foreground colors from its demo input across multiple palettes."""
    if not demos or len(demos) < 2:
        return False
    in_palettes = []
    diff_out_colors = []
    for d in demos:
        inp_arr = np.asarray(d["input"], dtype=int)
        out_arr = np.asarray(d["output"], dtype=int)
        inp_c = set(np.unique(inp_arr).tolist()) - {0}
        out_c = set(np.unique(out_arr).tolist()) - {0}
        if not out_c or not out_c.issubset(inp_c):
            return False
        in_palettes.append(frozenset(inp_c))
        if inp_arr.shape == out_arr.shape:
            diff_m = inp_arr != out_arr
            if diff_m.any():
                dc = set(np.unique(out_arr[diff_m]).tolist()) - {0}
                diff_out_colors.append(frozenset(dc))
    # If every demo only writes a single fixed fill color across all changed pixels, it is a fixed-color fill rule
    if (
        len(diff_out_colors) == len(demos)
        and len(set(diff_out_colors)) == 1
        and len(next(iter(diff_out_colors))) == 1
    ):
        return False
    if len(set(in_palettes)) > 1:
        return True
    return len(in_palettes[0]) >= 6


def wilson_ci(k: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * total)


def weighted_ce(demo_logits: torch.Tensor, demo_targets: torch.Tensor) -> torch.Tensor:
    B, L, V = demo_logits.shape
    per_tok = F.cross_entropy(
        demo_logits.reshape(B * L, V),
        demo_targets.reshape(B * L),
        reduction="none",
    ).view(B, L)
    weights = torch.where(demo_targets >= 1, 1.0, 0.15)
    return (per_tok * weights).sum() / weights.sum().clamp(min=1.0)


def grid_to_base_logits(
    grid: List[List[int]], margin: float = 0.75, device: str = "cpu"
) -> torch.Tensor:
    seq = encode_grid_to_trm_seq(grid)
    t = torch.tensor(np.asarray(seq), dtype=torch.long, device=device).unsqueeze(0)
    oh = F.one_hot(t.clamp(0, 11), num_classes=12).float()
    return oh * margin


def score_task_submission(
    preds: List[Dict[str, List[List[int]]]],
    test_pairs: List[Dict[str, Any]],
) -> Tuple[float, bool, List[float]]:
    n_pairs = len(test_pairs)
    if n_pairs == 0 or len(preds) < n_pairs:
        return 0.0, False, []
    pair_solved = 0
    pair_accs: List[float] = []
    for idx, pair in enumerate(test_pairs):
        gt = np.array(pair["output"], dtype=int)
        a1 = np.array(preds[idx]["attempt_1"], dtype=int)
        a2 = np.array(preds[idx]["attempt_2"], dtype=int)
        m1 = a1.shape == gt.shape and bool((a1 == gt).all())
        m2 = a2.shape == gt.shape and bool((a2 == gt).all())
        if m1 or m2:
            pair_solved += 1
        acc1 = float((a1 == gt).mean()) if a1.shape == gt.shape else 0.0
        acc2 = float((a2 == gt).mean()) if a2.shape == gt.shape else 0.0
        pair_accs.append(max(acc1, acc2))
    frac = pair_solved / float(n_pairs)
    return frac, (pair_solved == n_pairs), pair_accs


def run_darksnail_tta_on_task(
    tid: str,
    task: Dict[str, Any],
    s1_task_preds: List[Dict[str, List[List[int]]]],
    s2_base_task_preds: List[Dict[str, List[List[int]]]],
    t_diag: Dict[str, Any],
    readout: DarkSnailCanvasReadout,
    controller: ZenoAmygdalaController,
    device: str = "cpu",
) -> Dict[str, Any]:
    demos = task["train"]
    demo_target_hws = [
        (len(d["output"]), len(d["output"][0]))
        for d in demos
        if d.get("output") and d["output"][0]
    ]
    out_shapes = demo_target_hws
    active_target_hw = out_shapes[0] if len(set(out_shapes)) == 1 else None
    color_pres = is_demo_color_preserving(demos)

    in_seqs = np.asarray([encode_grid_to_trm_seq(d["input"]) for d in demos])
    out_seqs = np.asarray([encode_grid_to_trm_seq(d["output"]) for d in demos])
    demo_in = torch.tensor(in_seqs, dtype=torch.long, device=device)
    demo_out = torch.tensor(out_seqs, dtype=torch.long, device=device)

    base_demo = F.one_hot(demo_in.clamp(0, 11), num_classes=12).float() * 0.75

    flux_info = compute_grid_prompt_zeno_flux(demos, task["test"])
    sched = controller(
        zeno_flux=flux_info["zeno_flux"],
        lca_variance=flux_info["lca_variance"],
        step0_demo_loss=t_diag.get("step0_loss", 4.0),
        step0_grad_norm=t_diag.get("grad_norm_step0", 10.0),
        s1_demo_fit=0.5,
        test_entropy=0.4,
        canvas_cells=(active_target_hw[0] * active_target_hw[1]) if active_target_hw else 100,
    )

    supports_demo_hws = "demo_target_hws" in inspect.signature(readout.forward).parameters
    extra_demo_kwargs = {"demo_target_hws": demo_target_hws} if supports_demo_hws else {}

    best_loss = 1e9
    best_dz = torch.zeros(16, 512, dtype=torch.float32, device=device)
    actual_steps = 0
    winning_stage = "none"
    pb_rule_name = "same_size_identity"

    # --- Stage 1: Base Symmetric + Skew-Symmetric Triad + Asymmetry Readout (spectral_gain = 0) ---
    # Runs on every task; acceptance is gated solely by demo CE (no test labels, no task IDs).
    if True:
        dz = nn.Parameter(torch.zeros(16, 512, dtype=torch.float32, device=device))
        opt = torch.optim.Adam([dz], lr=0.05)
        for step in range(30):
            actual_steps = step + 1
            opt.zero_grad()
            logits = readout(
                logits=base_demo,
                inputs=demo_in,
                delta_z=dz,
                target_hw=active_target_hw,
                color_preserving=color_pres,
                triad_gain_override=sched.triad_gain,
                asym_gain_override=sched.asym_gain,
                spectral_gain_override=0.0,
                **extra_demo_kwargs,
            )
            ce = weighted_ce(logits, demo_out)
            reg = 0.5 * 0.1 * (dz.pow(2).mean())
            loss = ce + reg
            loss.backward()
            opt.step()
            ce_val = float(ce.item())
            if ce_val < best_loss:
                best_loss = ce_val
                best_dz = dz.data.clone()
            if step == 5 and best_loss > 0.45:
                break
        if best_loss <= 0.0032:
            winning_stage = "stage1_base"

    # --- Stage 2: Spatial Pullback Homography (2a) & Pre-Triad Spectral-Ray-Enclosure Coupling (2b) ---
    pb_info = extract_beltrami_pullback_canvas(demos, test_input=task["test"][0]["input"])
    pb_rule_name = str(pb_info.get("rule_name", "same_size_identity"))
    use_pullback_stage = (not pb_info["is_same_size"]) and bool(pb_info.get("demo_exact_match", False))

    if winning_stage == "none" and use_pullback_stage:
        pb_demo_in = pb_info["demo_inputs_pullback"].to(device)
        pb_demo_sec = pb_info["demo_secondary_pullback"].to(device)
        pb_base_demo = F.one_hot(pb_demo_in.clamp(0, 11), num_classes=12).float() * 0.75
        dz_pb = nn.Parameter(torch.zeros(16, 512, dtype=torch.float32, device=device))
        opt_pb = torch.optim.Adam([dz_pb], lr=0.05)
        pb_best_loss = 1e9
        pb_best_dz = dz_pb.data.clone()
        for step in range(15):
            actual_steps += 1
            opt_pb.zero_grad()
            logits = readout(
                logits=pb_base_demo,
                inputs=pb_demo_in,
                delta_z=dz_pb,
                target_hw=demo_target_hws,
                color_preserving=True,
                triad_gain_override=sched.triad_gain,
                asym_gain_override=sched.asym_gain,
                spectral_gain_override=0.0,
                inputs_pullback=pb_demo_in,
                secondary_pullback=pb_demo_sec,
                **extra_demo_kwargs,
            )
            ce = weighted_ce(logits, demo_out)
            reg = 0.5 * 0.1 * (dz_pb.pow(2).mean())
            loss = ce + reg
            loss.backward()
            opt_pb.step()
            ce_val = float(ce.item())
            if ce_val < pb_best_loss:
                pb_best_loss = ce_val
                pb_best_dz = dz_pb.data.clone()
        if pb_best_loss <= 0.0032:
            best_loss = pb_best_loss
            best_dz = pb_best_dz
            winning_stage = "stage2a_pullback"

    if winning_stage == "none":
        dz_sp = nn.Parameter(torch.zeros(16, 512, dtype=torch.float32, device=device))
        opt_sp = torch.optim.Adam([dz_sp], lr=0.05)
        sp_best_loss = 1e9
        sp_best_dz = dz_sp.data.clone()
        for step in range(35):
            actual_steps += 1
            opt_sp.zero_grad()
            logits = readout(
                logits=base_demo,
                inputs=demo_in,
                delta_z=dz_sp,
                target_hw=demo_target_hws,
                color_preserving=color_pres,
                triad_gain_override=sched.triad_gain,
                asym_gain_override=sched.asym_gain,
                spectral_gain_override=0.35,
                **extra_demo_kwargs,
            )
            ce = weighted_ce(logits, demo_out)
            reg = 0.5 * 0.1 * (dz_sp.pow(2).mean())
            loss = ce + reg
            loss.backward()
            opt_sp.step()
            ce_val = float(ce.item())
            if ce_val < sp_best_loss:
                sp_best_loss = ce_val
                sp_best_dz = dz_sp.data.clone()
            if step == 5 and sp_best_loss > 0.45:
                break
        if sp_best_loss < best_loss:
            best_loss = sp_best_loss
            best_dz = sp_best_dz
        if sp_best_loss <= 0.0032:
            winning_stage = "stage2b_spectral"

    test_preds: List[Dict[str, List[List[int]]]] = []
    if best_loss <= 0.0032 and winning_stage != "none":
        with torch.no_grad():
            for pidx, pair in enumerate(task["test"]):
                test_in = pair["input"]
                thw = infer_arc_output_shape(demos, test_in)
                if winning_stage == "stage2a_pullback":
                    pb_pair = extract_beltrami_pullback_canvas(demos, test_input=test_in, target_shape=thw)
                    pb_t_in = pb_pair["test_input_pullback"].to(device)
                    pb_t_sec = pb_pair["test_secondary_pullback"].to(device)
                    base_test = F.one_hot(pb_t_in.clamp(0, 11), num_classes=12).float() * 0.75
                    l_out = readout(
                        logits=base_test,
                        inputs=pb_t_in,
                        delta_z=best_dz,
                        target_hw=thw,
                        color_preserving=True,
                        triad_gain_override=sched.triad_gain,
                        asym_gain_override=sched.asym_gain,
                        spectral_gain_override=0.0,
                        inputs_pullback=pb_t_in,
                        secondary_pullback=pb_t_sec,
                    )
                elif winning_stage == "stage2b_spectral":
                    t_in = torch.tensor(
                        np.asarray([encode_grid_to_trm_seq(test_in)]),
                        dtype=torch.long,
                        device=device,
                    )
                    base_test = F.one_hot(t_in.clamp(0, 11), num_classes=12).float() * 0.75
                    l_out = readout(
                        logits=base_test,
                        inputs=t_in,
                        delta_z=best_dz,
                        target_hw=thw,
                        color_preserving=color_pres,
                        triad_gain_override=sched.triad_gain,
                        asym_gain_override=sched.asym_gain,
                        spectral_gain_override=0.35,
                    )
                else:
                    t_in = torch.tensor(
                        np.asarray([encode_grid_to_trm_seq(test_in)]),
                        dtype=torch.long,
                        device=device,
                    )
                    s1_grid = s1_task_preds[pidx]["attempt_1"]
                    base_test = grid_to_base_logits(s1_grid, margin=0.75, device=device)
                    l_out = readout(
                        logits=base_test,
                        inputs=t_in,
                        delta_z=best_dz,
                        target_hw=thw,
                        color_preserving=color_pres,
                        triad_gain_override=sched.triad_gain,
                        asym_gain_override=sched.asym_gain,
                        spectral_gain_override=0.0,
                    )
                c_pred = (
                    torch.argmax(l_out[0, :, 2:], dim=-1).cpu().numpy().reshape(30, 30)
                )
                pred_grid = c_pred[: thw[0], : thw[1]].astype(int).tolist()
                backup_grid = s2_base_task_preds[pidx]["attempt_1"]
                if backup_grid == pred_grid:
                    backup_grid = s2_base_task_preds[pidx]["attempt_2"]
                test_preds.append({"attempt_1": pred_grid, "attempt_2": backup_grid})
        gate_accepted = True
    else:
        test_preds = s2_base_task_preds
        gate_accepted = False

    return {
        "demo_ce": best_loss,
        "color_preserving": color_pres,
        "zeno_flux": flux_info["zeno_flux"],
        "lca_variance": flux_info["lca_variance"],
        "turbulence_index": sched.turbulence_index,
        "scheduled_steps": sched.num_steps,
        "actual_steps": actual_steps,
        "gate_accepted": gate_accepted,
        "winning_stage": winning_stage,
        "pullback_rule": pb_rule_name,
        "test_preds": test_preds,
    }


def resolve_arc_eval_dir() -> Path:
    candidates = [
        WORKSPACE_DIR / "brain-ai" / "data" / "arc_extracted" / "ARC-AGI-master" / "data" / "evaluation",
        ROOT_DIR / "data" / "ARC-AGI" / "data" / "evaluation",
        Path("/tmp/ARC-AGI/data/evaluation"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    target_clone = Path("/tmp/ARC-AGI")
    print("Cloning fchollet/ARC-AGI evaluation dataset into /tmp/ARC-AGI...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/fchollet/ARC-AGI.git", str(target_clone)],
        check=True,
    )
    return target_clone / "data" / "evaluation"


def resolve_baseline_dir() -> Path:
    bundled = ROOT_DIR / "results" / "baselines"
    if (bundled / "submission_s1_zero_shot.json").exists():
        return bundled
    return WORKSPACE_DIR / "brain-ai" / "results" / "arc_callosum_p3"


def main() -> None:
    t0 = time.time()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_dir = resolve_arc_eval_dir()
    base_dir = resolve_baseline_dir()

    s1_path = base_dir / "submission_s1_zero_shot.json"
    s2_base_path = base_dir / "submission_s2_adapted.json"
    p3_comp_path = base_dir / "p3_rq3_comprehensive_comparison.json"

    with open(s1_path, "r", encoding="utf-8") as f:
        sub_s1 = json.load(f)
    with open(s2_base_path, "r", encoding="utf-8") as f:
        sub_s2_base = json.load(f)
    with open(p3_comp_path, "r", encoding="utf-8") as f:
        p3_comp = json.load(f)

    per_task_diag = p3_comp.get("tta_diagnostics", {}).get("per_task_diagnostics", {})

    tasks: Dict[str, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(eval_dir)):
        if fname.endswith(".json"):
            tid = fname[:-5]
            with open(eval_dir / fname, "r", encoding="utf-8") as f:
                tasks[tid] = json.load(f)

    print(f"Loaded {len(tasks)} ARC-AGI-1 evaluation tasks (device={device}).", flush=True)

    readout = DarkSnailCanvasReadout().to(device)
    controller = ZenoAmygdalaController().to(device)

    sub_darksnail: Dict[str, List[Dict[str, List[List[int]]]]] = {}
    task_telemetry: Dict[str, Dict[str, Any]] = {}

    s1_frac_total = 0.0
    s1_strict_total = 0
    s2_base_frac_total = 0.0
    s2_base_strict_total = 0
    ds_frac_total = 0.0
    ds_strict_total = 0

    near_miss_9 = [
        "68b67ca3",
        "31d5ba1a",
        "ce039d91",
        "8ba14f53",
        "9110e3c5",
        "e57337a4",
        "195ba7dc",
        "0b17323b",
        "af22c60d",
    ]
    half_solved_3 = ["5d2a5c43", "6ea4a07e", "e345f17b"]
    total_test_pairs = 0
    exact_shapes = 0

    shard_spec = os.environ.get("DARKSNAIL_SHARD", "")
    merge_n = int(os.environ.get("DARKSNAIL_MERGE_SHARDS", "0"))
    shard_dir = ROOT_DIR / "results" / "shards"
    if shard_spec:
        shard_k, shard_n = (int(x) for x in shard_spec.split("/"))
        torch.set_num_threads(2)
        task_items = [(i, kv) for i, kv in enumerate(tasks.items()) if i % shard_n == shard_k]
    elif merge_n > 0:
        task_items = []
        for k in range(merge_n):
            with open(shard_dir / f"shard_{k}.json", "r", encoding="utf-8") as f:
                sh = json.load(f)
            task_telemetry.update(sh["task_telemetry"])
            sub_darksnail.update(sh["sub_darksnail"])
            total_test_pairs += sh["total_test_pairs"]
            exact_shapes += sh["exact_shapes"]
        task_telemetry = {tid: task_telemetry[tid] for tid in tasks}
        sub_darksnail = {tid: sub_darksnail[tid] for tid in tasks}
    else:
        task_items = list(enumerate(tasks.items()))

    for idx, (tid, task) in task_items:
        s1_preds = sub_s1[tid]
        s2_base_preds = sub_s2_base[tid]

        s1_frac, s1_strict, s1_accs = score_task_submission(s1_preds, task["test"])
        s2_b_frac, s2_b_strict, s2_b_accs = score_task_submission(
            s2_base_preds, task["test"]
        )

        ds_res = run_darksnail_tta_on_task(
            tid,
            task,
            s1_preds,
            s2_base_preds,
            per_task_diag.get(tid, {}),
            readout,
            controller,
            device=device,
        )
        readout._basis_cache.clear()
        readout._spectral_basis_cache.clear()
        readout._ext_op_cache.clear()

        ds_preds = ds_res["test_preds"]
        sub_darksnail[tid] = ds_preds
        ds_frac, ds_strict, ds_accs = score_task_submission(ds_preds, task["test"])
        for pair in task["test"]:
            total_test_pairs += 1
            gt_hw = (len(pair["output"]), len(pair["output"][0]))
            if tuple(infer_arc_output_shape(task["train"], pair["input"])) == gt_hw:
                exact_shapes += 1

        s1_frac_total += s1_frac
        s1_strict_total += int(s1_strict)
        s2_base_frac_total += s2_b_frac
        s2_base_strict_total += int(s2_b_strict)
        ds_frac_total += ds_frac
        ds_strict_total += int(ds_strict)

        task_telemetry[tid] = {
            "s1_frac": s1_frac,
            "s2_base_frac": s2_b_frac,
            "darksnail_frac": ds_frac,
            "s1_strict": s1_strict,
            "s2_base_strict": s2_b_strict,
            "darksnail_strict": ds_strict,
            "s2_base_pair_accs": s2_b_accs,
            "darksnail_pair_accs": ds_accs,
            "demo_ce": ds_res["demo_ce"],
            "gate_accepted": ds_res["gate_accepted"],
            "winning_stage": ds_res.get("winning_stage", "none"),
            "pullback_rule": ds_res.get("pullback_rule", "same_size_identity"),
            "zeno_flux": ds_res["zeno_flux"],
            "lca_variance": ds_res["lca_variance"],
            "turbulence_index": ds_res["turbulence_index"],
            "scheduled_steps": ds_res["scheduled_steps"],
            "actual_steps": ds_res["actual_steps"],
        }

        if tid in near_miss_9 or tid in half_solved_3:
            print(
                f"[{idx+1:3d}/400] Task {tid}: "
                f"S1={s1_frac:.2f} | S2_Base={s2_b_frac:.2f} ({[round(a*100,1) for a in s2_b_accs]}) "
                f"-> DarkSnail={ds_frac:.2f} ({[round(a*100,1) for a in ds_accs]}) "
                f"[demo_ce={ds_res['demo_ce']:.4f}, gate={ds_res['gate_accepted']}, stage={ds_res.get('winning_stage')}]",
                flush=True,
            )
        elif (idx + 1) % 50 == 0:
            print(
                f"[{idx+1:3d}/400] Progress: S1={s1_frac_total:.2f}, "
                f"S2_Base={s2_base_frac_total:.2f}, DarkSnail={ds_frac_total:.2f} "
                f"[strict={ds_strict_total}] ({time.time() - t0:.1f}s)",
                flush=True,
            )

    if shard_spec:
        shard_dir.mkdir(parents=True, exist_ok=True)
        with open(shard_dir / f"shard_{shard_k}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_telemetry": task_telemetry,
                    "sub_darksnail": sub_darksnail,
                    "total_test_pairs": total_test_pairs,
                    "exact_shapes": exact_shapes,
                },
                f,
            )
        print(f"Shard {shard_k}/{shard_n} done: {len(task_telemetry)} tasks ({time.time() - t0:.1f}s)", flush=True)
        return

    s1_frac_total = sum(t["s1_frac"] for t in task_telemetry.values())
    s1_strict_total = sum(int(t["s1_strict"]) for t in task_telemetry.values())
    s2_base_frac_total = sum(t["s2_base_frac"] for t in task_telemetry.values())
    s2_base_strict_total = sum(int(t["s2_base_strict"]) for t in task_telemetry.values())
    ds_frac_total = sum(t["darksnail_frac"] for t in task_telemetry.values())
    ds_strict_total = sum(int(t["darksnail_strict"]) for t in task_telemetry.values())

    N = len(tasks)
    b_ds_vs_s1 = sum(
        1
        for t in task_telemetry.values()
        if t["darksnail_strict"] and not t["s1_strict"]
    )
    c_ds_vs_s1 = sum(
        1
        for t in task_telemetry.values()
        if t["s1_strict"] and not t["darksnail_strict"]
    )
    b_ds_vs_s2 = sum(
        1
        for t in task_telemetry.values()
        if t["darksnail_strict"] and not t["s2_base_strict"]
    )
    c_ds_vs_s2 = sum(
        1
        for t in task_telemetry.values()
        if t["s2_base_strict"] and not t["darksnail_strict"]
    )

    p_ds_vs_s1 = exact_mcnemar_p(b_ds_vs_s1, c_ds_vs_s1)
    p_ds_vs_s2 = exact_mcnemar_p(b_ds_vs_s2, c_ds_vs_s2)

    ci_s1 = wilson_ci(s1_frac_total, N)
    ci_s2 = wilson_ci(s2_base_frac_total, N)
    ci_ds = wilson_ci(ds_frac_total, N)

    near_miss_report = {
        tid: {
            "s2_base_frac": task_telemetry[tid]["s2_base_frac"],
            "darksnail_frac": task_telemetry[tid]["darksnail_frac"],
            "s2_base_pair_accs": task_telemetry[tid]["s2_base_pair_accs"],
            "darksnail_pair_accs": task_telemetry[tid]["darksnail_pair_accs"],
            "demo_ce": task_telemetry[tid]["demo_ce"],
            "zeno_flux": task_telemetry[tid]["zeno_flux"],
        }
        for tid in near_miss_9
    }
    half_solved_report = {
        tid: {
            "s2_base_frac": task_telemetry[tid]["s2_base_frac"],
            "darksnail_frac": task_telemetry[tid]["darksnail_frac"],
            "s2_base_pair_accs": task_telemetry[tid]["s2_base_pair_accs"],
            "darksnail_pair_accs": task_telemetry[tid]["darksnail_pair_accs"],
            "demo_ce": task_telemetry[tid]["demo_ce"],
            "zeno_flux": task_telemetry[tid]["zeno_flux"],
        }
        for tid in half_solved_3
    }

    solved_tasks_list = sorted(
        [tid for tid, t in task_telemetry.items() if t["darksnail_frac"] > 0.0]
    )
    strict_solved_tasks_list = sorted(
        [tid for tid, t in task_telemetry.items() if t["darksnail_strict"]]
    )
    pass1_path = ROOT_DIR / "results" / "darksnail_arc_evaluation_summary_pass1.json"
    with open(pass1_path, "r", encoding="utf-8") as f:
        pass1_27 = set(json.load(f)["solved_tasks"])
    b_ds_vs_pass1 = sum(1 for tid in strict_solved_tasks_list if tid not in pass1_27)
    c_ds_vs_pass1 = sum(1 for tid in pass1_27 if tid not in strict_solved_tasks_list)
    p_ds_vs_pass1 = exact_mcnemar_p(b_ds_vs_pass1, c_ds_vs_pass1)

    phase2_new_tasks = {
        tid: {
            "winning_stage": task_telemetry[tid]["winning_stage"],
            "pullback_rule": task_telemetry[tid]["pullback_rule"],
            "demo_ce": round(float(task_telemetry[tid]["demo_ce"]), 6),
            "darksnail_frac": task_telemetry[tid]["darksnail_frac"],
            "darksnail_strict": task_telemetry[tid]["darksnail_strict"],
            "darksnail_pair_accs": task_telemetry[tid]["darksnail_pair_accs"],
            "s2_base_frac": task_telemetry[tid]["s2_base_frac"],
        }
        for tid in sorted(task_telemetry.keys())
        if task_telemetry[tid]["darksnail_frac"] > 0.0 and tid not in pass1_27
    }

    summary = {
        "n_tasks": N,
        "elapsed_sec": round(time.time() - t0, 2),
        "metrics": {
            "system1_zero_shot": {
                "fractional_score": s1_frac_total,
                "fractional_pct": round(100.0 * s1_frac_total / N, 3),
                "strict_tasks_solved": s1_strict_total,
                "wilson_95_ci_pct": [
                    round(100.0 * ci_s1[0], 2),
                    round(100.0 * ci_s1[1], 2),
                ],
            },
            "brain_ai_s2_baseline": {
                "fractional_score": s2_base_frac_total,
                "fractional_pct": round(100.0 * s2_base_frac_total / N, 3),
                "strict_tasks_solved": s2_base_strict_total,
                "wilson_95_ci_pct": [
                    round(100.0 * ci_s2[0], 2),
                    round(100.0 * ci_s2[1], 2),
                ],
            },
            "darksnail_v2_synthesis": {
                "fractional_score": ds_frac_total,
                "fractional_pct": round(100.0 * ds_frac_total / N, 3),
                "strict_tasks_solved": ds_strict_total,
                "wilson_95_ci_pct": [
                    round(100.0 * ci_ds[0], 2),
                    round(100.0 * ci_ds[1], 2),
                ],
            },
        },
        "paired_mcnemar_tests": {
            "darksnail_vs_system1": {
                "gained_strict": b_ds_vs_s1,
                "lost_strict": c_ds_vs_s1,
                "exact_two_sided_p": p_ds_vs_s1,
            },
            "darksnail_vs_brain_ai_s2": {
                "gained_strict": b_ds_vs_s2,
                "lost_strict": c_ds_vs_s2,
                "exact_two_sided_p": p_ds_vs_s2,
            },
            "darksnail_phase2_vs_pass1_27": {
                "gained_strict": b_ds_vs_pass1,
                "lost_strict": c_ds_vs_pass1,
                "exact_two_sided_p": p_ds_vs_pass1,
            },
        },
        "near_miss_9_breakdown": near_miss_report,
        "multi_pair_half_solved_3_breakdown": half_solved_report,
        "phase2_newly_solved_breakdown": phase2_new_tasks,
        "solved_tasks": solved_tasks_list,
        "strict_solved_tasks": strict_solved_tasks_list,
        "exact_output_shapes_beltrami": exact_shapes,
        "total_test_pairs": total_test_pairs,
        "beltrami_shape_accuracy_pct": round(100.0 * exact_shapes / max(1, total_test_pairs), 2),
    }

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(
        results_dir / "darksnail_arc_evaluation_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "submission_darksnail_v2.json", "w", encoding="utf-8") as f:
        json.dump(sub_darksnail, f)

    print("\n" + "=" * 72)
    print("PROJECT DARKSNAIL V2 — ARC-AGI-1 PUBLIC EVALUATION SUMMARY (N = 400)")
    print("=" * 72)
    print(
        f"System 1 Zero-Shot      : {s1_frac_total:5.2f} / 400 ({100.0*s1_frac_total/N:5.2f}%) "
        f"[strict: {s1_strict_total}]"
    )
    print(
        f"brain-ai S2 Baseline    : {s2_base_frac_total:5.2f} / 400 ({100.0*s2_base_frac_total/N:5.2f}%) "
        f"[strict: {s2_base_strict_total}]"
    )
    print(
        f"DarkSnail V2 Synthesis  : {ds_frac_total:5.2f} / 400 ({100.0*ds_frac_total/N:5.2f}%) "
        f"[strict: {ds_strict_total}]"
    )
    print(
        f"McNemar vs System 1     : +{b_ds_vs_s1} / -{c_ds_vs_s1} strict tasks (p = {p_ds_vs_s1:.3e})"
    )
    print(
        f"McNemar vs brain-ai S2  : +{b_ds_vs_s2} / -{c_ds_vs_s2} strict tasks (p = {p_ds_vs_s2:.3e})"
    )
    print(
        f"McNemar vs Pass-1 (27)  : +{b_ds_vs_pass1} / -{c_ds_vs_pass1} strict tasks (p = {p_ds_vs_pass1:.3e})"
    )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
