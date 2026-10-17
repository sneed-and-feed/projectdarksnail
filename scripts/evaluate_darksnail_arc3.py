"""
scripts/evaluate_darksnail_arc3.py: Offline 25-Game ARC-AGI-3 Benchmark Evaluation Suite.

Evaluates three strictly zero-leakage agent conditions across all 25 public ARC-AGI-3
environments in `OperationMode.OFFLINE`:
  1. `random_legal`: Uniform random selection over `available_actions` (with uniform
     random `(x, y)` in `[0, 63]^2` for `ACTION6`, and `RESET` on `GAME_OVER`).
  2. `unshielded_greedy`: Ablated greedy baseline without `TemporalClockContentSplitter`,
     without `extract_arc3_beltrami_medoids`, and without `Arc3EpisodicBeltramiShell` /
     `SlowFastTriadCascade` (hashes raw frames including exogenous clocks and uses
     unshielded greedy action cycling + uniform coordinate sampling).
  3. `darksnail_v2_arc3`: Full `DarkSnailArc3Agent` integrating `TemporalClockContentSplitter`,
     `extract_arc3_beltrami_medoids` (with Dyadic Quadtree Stratification),
     `Arc3EpisodicBeltramiShell`, `DarkSnailArc3WorldModel` (`SlowFastTriadCascade` +
     `SkewSymmetricTriadOperator` + 1-Step Empowerment + Topological Symmetry Potential),
     and Epistemic Frontier BFS Graph Search.

Outputs per-game and aggregate metrics plus paired statistical tests to
`results/darksnail_arc3_evaluation_summary.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from darksnail.arc3 import DarkSnailArc3Agent, _frame_to_2d_numpy


def run_random_legal_episode(env: Any, max_actions: int = 400, seed: int = 0) -> Dict[str, Any]:
    """Condition 1: Uniform random legal action baseline (`random_legal`)."""
    rng = np.random.RandomState(seed)
    obs = getattr(env, "observation_space", None)
    if obs is None or len(getattr(obs, "frame", [])) == 0:
        obs = env.reset()

    game_id = getattr(
        obs, "game_id", getattr(getattr(env, "environment_info", None), "game_id", "unknown")
    )
    actions_taken = 0
    resets_taken = 0
    level_completion_steps: List[Tuple[int, int]] = []
    last_levels_completed = int(getattr(obs, "levels_completed", 0))
    raw_hashes: Set[str] = set()

    while actions_taken < max_actions:
        state = getattr(obs, "state", GameState.NOT_FINISHED)
        if state == GameState.WIN or str(state) in ("GameState.WIN", "WIN"):
            break

        f2d = _frame_to_2d_numpy(obs)
        raw_hashes.add(hashlib.blake2b(np.ascontiguousarray(f2d, dtype=np.int8).tobytes(), digest_size=16).hexdigest())

        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or str(state) in (
            "GameState.NOT_PLAYED",
            "NOT_PLAYED",
            "GameState.GAME_OVER",
            "GAME_OVER",
        ):
            resets_taken += 1
            obs = env.step(GameAction.RESET)
        else:
            avail = [int(a) for a in getattr(obs, "available_actions", [1, 2, 3, 4, 5, 6]) if int(a) != 0]
            if not avail:
                avail = [1, 2, 3, 4, 5, 6]
            act_id = int(avail[int(rng.randint(0, len(avail)))])
            act = GameAction.from_id(act_id)
            if act.is_complex():
                rx, ry = int(rng.randint(0, 64)), int(rng.randint(0, 64))
                act.set_data({"x": rx, "y": ry})
                obs = env.step(act, data={"x": rx, "y": ry})
            else:
                obs = env.step(act)

        actions_taken += 1
        if obs is None:
            break
        cur_lvl = int(getattr(obs, "levels_completed", 0))
        if cur_lvl > last_levels_completed:
            level_completion_steps.append((cur_lvl, actions_taken))
            last_levels_completed = cur_lvl

    final_state = getattr(obs, "state", GameState.NOT_FINISHED) if obs is not None else GameState.NOT_FINISHED
    return {
        "game_id": game_id,
        "state": final_state.name if hasattr(final_state, "name") else str(final_state),
        "won": bool(final_state == GameState.WIN or str(final_state) in ("GameState.WIN", "WIN")),
        "levels_completed": int(getattr(obs, "levels_completed", 0)) if obs is not None else last_levels_completed,
        "win_levels": int(getattr(obs, "win_levels", 0)) if obs is not None else 0,
        "actions_taken": actions_taken,
        "resets_taken": resets_taken,
        "level_completion_steps": level_completion_steps,
        "unique_nope_states_visited": len(raw_hashes),
        "clock_pixels_masked": 0,
    }


def run_unshielded_greedy_episode(env: Any, max_actions: int = 400, seed: int = 0) -> Dict[str, Any]:
    """
    Condition 2: Unshielded Greedy Baseline (`unshielded_greedy`).
    Ablates `TemporalClockContentSplitter` (hashes raw frames including exogenous clocks, so
    clock ticks alias wall-bumps into apparently new states), ablates `extract_arc3_beltrami_medoids`
    (samples uniform grid coordinates on `ACTION6`), and ablates `SlowFastTriadCascade` / BFS.
    """
    rng = np.random.RandomState(seed)
    obs = getattr(env, "observation_space", None)
    if obs is None or len(getattr(obs, "frame", [])) == 0:
        obs = env.reset()

    game_id = getattr(
        obs, "game_id", getattr(getattr(env, "environment_info", None), "game_id", "unknown")
    )
    actions_taken = 0
    resets_taken = 0
    level_completion_steps: List[Tuple[int, int]] = []
    last_levels_completed = int(getattr(obs, "levels_completed", 0))

    tried_raw: Dict[str, Set[int]] = {}
    prev_raw_hash: str | None = None
    prev_act_id: int | None = None
    prev_f2d: np.ndarray | None = None
    noop_raw: Dict[str, Set[int]] = {}

    while actions_taken < max_actions:
        state = getattr(obs, "state", GameState.NOT_FINISHED)
        if state == GameState.WIN or str(state) in ("GameState.WIN", "WIN"):
            break

        f2d = _frame_to_2d_numpy(obs)
        cur_raw_hash = hashlib.blake2b(
            np.ascontiguousarray(f2d, dtype=np.int8).tobytes(), digest_size=16
        ).hexdigest()

        if prev_raw_hash is not None and prev_act_id is not None and prev_f2d is not None:
            if int(np.sum(f2d != prev_f2d)) == 0:
                noop_raw.setdefault(prev_raw_hash, set()).add(prev_act_id)

        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or str(state) in (
            "GameState.NOT_PLAYED",
            "NOT_PLAYED",
            "GameState.GAME_OVER",
            "GAME_OVER",
        ):
            resets_taken += 1
            prev_raw_hash = None
            prev_act_id = None
            prev_f2d = None
            obs = env.step(GameAction.RESET)
        else:
            avail = [
                int(a)
                for a in getattr(obs, "available_actions", [1, 2, 3, 4, 5, 6])
                if int(a) not in (0, 7)
            ]
            if not avail:
                avail = [1, 2, 3, 4, 5, 6]
            dead_here = noop_raw.get(cur_raw_hash, set())
            tried_here = tried_raw.setdefault(cur_raw_hash, set())
            candidates = [a for a in avail if a not in dead_here and a not in tried_here]
            if not candidates:
                candidates = [a for a in avail if a not in dead_here] or avail

            act_id = int(candidates[0])
            tried_here.add(act_id)
            act = GameAction.from_id(act_id)
            prev_raw_hash = cur_raw_hash
            prev_act_id = act_id
            prev_f2d = f2d.copy()

            if act.is_complex():
                # Uniform grid coordinate sampling (no Beltrami medoid pooler)
                rx, ry = int(rng.randint(0, 64)), int(rng.randint(0, 64))
                act.set_data({"x": rx, "y": ry})
                obs = env.step(act, data={"x": rx, "y": ry})
            else:
                obs = env.step(act)

        actions_taken += 1
        if obs is None:
            break
        cur_lvl = int(getattr(obs, "levels_completed", 0))
        if cur_lvl > last_levels_completed:
            level_completion_steps.append((cur_lvl, actions_taken))
            last_levels_completed = cur_lvl
            tried_raw.clear()
            noop_raw.clear()

    final_state = getattr(obs, "state", GameState.NOT_FINISHED) if obs is not None else GameState.NOT_FINISHED
    return {
        "game_id": game_id,
        "state": final_state.name if hasattr(final_state, "name") else str(final_state),
        "won": bool(final_state == GameState.WIN or str(final_state) in ("GameState.WIN", "WIN")),
        "levels_completed": int(getattr(obs, "levels_completed", 0)) if obs is not None else last_levels_completed,
        "win_levels": int(getattr(obs, "win_levels", 0)) if obs is not None else 0,
        "actions_taken": actions_taken,
        "resets_taken": resets_taken,
        "level_completion_steps": level_completion_steps,
        "unique_nope_states_visited": len(tried_raw),
        "clock_pixels_masked": 0,
    }


def compute_paired_stats(a_vals: Sequence[float], b_vals: Sequence[float], seed: int = 1729) -> Dict[str, Any]:
    """
    Computes paired non-parametric statistical comparison (`a_vals - b_vals`):
      - Mean difference and 10,000-sample paired bootstrap 95% CI,
      - Exact two-sided binomial sign test p-value,
      - Wilcoxon signed-rank test p-value (via scipy.stats if available).
    """
    arr_a = np.asarray(a_vals, dtype=np.float64)
    arr_b = np.asarray(b_vals, dtype=np.float64)
    diffs = arr_a - arr_b
    n = int(diffs.size)
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    n_eff = pos + neg

    # Exact two-sided binomial sign test p-value
    if n_eff == 0:
        sign_p = 1.0
    else:
        k = min(pos, neg)
        tail = sum(math.comb(n_eff, i) for i in range(k + 1)) / float(2 ** n_eff)
        sign_p = min(1.0, 2.0 * tail)

    wilcoxon_p = sign_p
    try:
        from scipy.stats import wilcoxon

        if n_eff > 0:
            res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            wilcoxon_p = float(res.pvalue)
    except Exception:
        pass

    rng = np.random.RandomState(seed)
    boot_Idx = rng.randint(0, n, size=(10000, n))
    boot_means = diffs[boot_Idx].mean(axis=1)
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    return {
        "n_pairs": n,
        "mean_diff": float(diffs.mean()),
        "sum_diff": float(diffs.sum()),
        "wins": pos,
        "losses": neg,
        "ties": ties,
        "sign_test_p_value": float(sign_p),
        "wilcoxon_p_value": float(wilcoxon_p),
        "bootstrap_95_ci": [ci_low, ci_high],
    }


def evaluate_condition(
    condition_name: str,
    env_dir: Path,
    game_ids: List[str],
    max_actions: int,
    seed: int,
) -> Dict[str, Any]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir))
    scorecard_id = arc.create_scorecard()
    t0 = time.perf_counter()

    per_game: List[Dict[str, Any]] = []
    for idx, gid in enumerate(game_ids):
        gt0 = time.perf_counter()
        env = arc.make(gid, scorecard_id=scorecard_id)
        if condition_name == "random_legal":
            res = run_random_legal_episode(env, max_actions=max_actions, seed=seed + idx)
        elif condition_name == "unshielded_greedy":
            res = run_unshielded_greedy_episode(env, max_actions=max_actions, seed=seed + idx)
        elif condition_name == "darksnail_v2_arc3":
            agent = DarkSnailArc3Agent(max_actions=max_actions, seed=seed)
            res = agent.play_game(env, max_actions=max_actions)
        else:
            raise ValueError(f"Unknown condition: {condition_name}")
        res["wall_time_sec"] = round(time.perf_counter() - gt0, 3)
        per_game.append(res)
        print(
            f"[{condition_name}] ({idx+1:02d}/{len(game_ids):02d}) {res['game_id']}: "
            f"levels={res['levels_completed']}/{res['win_levels']} won={res['won']} "
            f"actions={res['actions_taken']} clock_px={res['clock_pixels_masked']} "
            f"({res['wall_time_sec']:.2f}s)",
            flush=True,
        )

    scorecard = arc.close_scorecard(scorecard_id)
    sc_dict = scorecard.model_dump() if scorecard is not None else {}
    env_scores_by_id = {
        item["id"]: float(item.get("score", 0.0)) for item in sc_dict.get("environments", [])
    }
    for g_res in per_game:
        g_res["official_rhae_score"] = env_scores_by_id.get(g_res["game_id"], 0.0)

    total_levels_completed = sum(int(g["levels_completed"]) for g in per_game)
    total_levels_available = sum(int(g["win_levels"]) for g in per_game)
    games_with_level_solved = sum(1 for g in per_game if int(g["levels_completed"]) >= 1)
    full_games_won = sum(1 for g in per_game if bool(g["won"]))
    total_actions = sum(int(g["actions_taken"]) for g in per_game)

    return {
        "condition": condition_name,
        "max_actions_per_game": max_actions,
        "num_games": len(game_ids),
        "total_levels_available": total_levels_available,
        "total_levels_completed": total_levels_completed,
        "level_completion_rate_pct": round(
            100.0 * float(total_levels_completed) / float(max(1, total_levels_available)), 2
        ),
        "games_with_at_least_1_level_completed": games_with_level_solved,
        "games_with_at_least_1_level_pct": round(
            100.0 * float(games_with_level_solved) / float(max(1, len(game_ids))), 2
        ),
        "full_games_won": full_games_won,
        "official_rhae_scorecard_score_pct": round(float(sc_dict.get("score", 0.0)), 4),
        "total_actions_taken": total_actions,
        "mean_actions_per_completed_level": round(
            float(total_actions) / float(total_levels_completed), 2
        )
        if total_levels_completed > 0
        else None,
        "wall_time_sec": round(time.perf_counter() - t0, 2),
        "per_game": per_game,
    }


PUBLIC_25_GAME_IDS: List[str] = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DarkSnail V2 on 25 ARC-AGI-3 offline games.")
    parser.add_argument(
        "--env-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "arc_agi_3_environments"),
        help="Path to offline ARC-AGI-3 environments directory.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=1000,
        help="Action budget per game.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "results" / "darksnail_arc3_evaluation_summary.json"),
        help="Output JSON summary path.",
    )
    args = parser.parse_args()

    env_dir = Path(args.env_dir)
    probe_arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir))
    game_ids = sorted({e.game_id.split("-")[0] for e in probe_arc.available_environments})
    print(f"Discovered {len(game_ids)} offline ARC-AGI-3 games: {game_ids}", flush=True)

    conditions = ["random_legal", "unshielded_greedy", "darksnail_v2_arc3"]
    results_by_cond: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        print(f"\n=== Evaluating Condition: {cond} (max_actions={args.max_actions}) ===", flush=True)
        results_by_cond[cond] = evaluate_condition(
            cond, env_dir=env_dir, game_ids=game_ids, max_actions=args.max_actions, seed=args.seed
        )

    ds_levels = [g["levels_completed"] for g in results_by_cond["darksnail_v2_arc3"]["per_game"]]
    rnd_levels = [g["levels_completed"] for g in results_by_cond["random_legal"]["per_game"]]
    grd_levels = [g["levels_completed"] for g in results_by_cond["unshielded_greedy"]["per_game"]]

    ds_rhae = [g["official_rhae_score"] for g in results_by_cond["darksnail_v2_arc3"]["per_game"]]
    rnd_rhae = [g["official_rhae_score"] for g in results_by_cond["random_legal"]["per_game"]]
    grd_rhae = [g["official_rhae_score"] for g in results_by_cond["unshielded_greedy"]["per_game"]]

    summary = {
        "benchmark": "ARC-AGI-3 Public Offline Suite (25 Games, 183 Levels)",
        "max_actions_per_game": args.max_actions,
        "seed": args.seed,
        "conditions": results_by_cond,
        "paired_statistics": {
            "darksnail_vs_random_levels_completed": compute_paired_stats(ds_levels, rnd_levels),
            "darksnail_vs_unshielded_greedy_levels_completed": compute_paired_stats(
                ds_levels, grd_levels
            ),
            "darksnail_vs_random_rhae_score": compute_paired_stats(ds_rhae, rnd_rhae),
            "darksnail_vs_unshielded_greedy_rhae_score": compute_paired_stats(ds_rhae, grd_rhae),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved complete evaluation summary to {out_path}", flush=True)

    print("\n| Condition | Levels Completed | Level % | Games >= 1 Level | Full Games Won | Official RHAE Score (%) |")
    print("|---|---:|---:|---:|---:|---:|")
    for cond in conditions:
        c = results_by_cond[cond]
        print(
            f"| `{cond}` | {c['total_levels_completed']} / {c['total_levels_available']} "
            f"| {c['level_completion_rate_pct']:.2f}% "
            f"| {c['games_with_at_least_1_level_completed']} / {c['num_games']} "
            f"| {c['full_games_won']} / {c['num_games']} "
            f"| {c['official_rhae_scorecard_score_pct']:.2f}% |"
        )


if __name__ == "__main__":
    main()
