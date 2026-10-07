"""Read-only post-training evaluator for RAWD v3 Run6 seed42.

The evaluator never resumes training and never mutates the training artifact.
It requires a completed 130-episode run before producing evaluation outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RUN6 = ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full"
DEFAULT_RUN5 = ROOT / "artifacts/training_v3/run5_context_aware_dqn"

RUN5_FALLBACK = {
    "episodes": 130, "steps": 1223, "active_candidate": 907,
    "target_available": 685, "actionable": 506, "selected": 14,
    "selection_rate": 14 / 506, "successful": 0, "semantic": 0,
    "detector": 0, "confirmation": 0, "successful_replay": 0,
    "mean_rank": 21.16, "median_rank": 20.0,
    "top1": 14 / 506, "top3": 7.51 / 100, "top5": 12.25 / 100,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (index - low))


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("reward", 0.0) or 0.0) for row in rows]
    return {
        "n": len(values), "mean": mean(values), "median": median(values),
        "min": min(values) if values else 0.0, "max": max(values) if values else 0.0,
        "p25": pct(values, .25), "p75": pct(values, .75),
    }


def bool_value(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def canonical_episode_map(metrics: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    site_order: list[str] = []
    for row in metrics:
        site = str(row.get("site_id", ""))
        if site and site not in site_order:
            site_order.append(site)
    return {(site, episode): site_index * 2 + episode + 1
            for site_index, site in enumerate(site_order)
            for episode in (0, 1)}


def attach_canonical(rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    mapping = canonical_episode_map(metrics)
    for row in rows:
        row["canonical_episode"] = mapping.get((str(row.get("site_id", "")), int(row.get("episode", 0))), 0)


def completion_gate(run6: Path, summary: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    completed = int(summary.get("episodes", 0) or 0)
    completed = max(completed, int(training.get("completed_episodes", 0) or 0))
    return {"canonical_episodes": completed, "required": 130, "pass": completed == 130}


def checkpoint_validation(run6: Path, summary: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(str(summary.get("checkpoint", "")))
    recovery = checkpoint.parent / "rawd_v3_run6_confirmation_aware_dqn_seed42_recovery.pt"
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint), "exists": checkpoint.exists(),
        "sha256": sha256(checkpoint) if checkpoint.exists() else None,
        "recovery_snapshot": str(recovery), "recovery_exists": recovery.exists(),
        "final_checkpoint_load": False, "recovery_load": False,
        "online_network": False, "target_network": False, "optimizer": False,
        "replay": False, "rng": False, "episode_counter": False,
    }
    if not checkpoint.exists():
        return result
    try:
        import torch
        final = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result["final_checkpoint_load"] = True
        result["online_network"] = "network_state_dict" in final
        result["optimizer"] = "optimizer_state_dict" in final
        if recovery.exists():
            snap = torch.load(recovery, map_location="cpu", weights_only=False)
            result["recovery_load"] = True
            result["target_network"] = "target_state_dict" in snap
            result["optimizer"] = result["optimizer"] and "optimizer_state_dict" in snap
            result["replay"] = all(key in snap for key in ("replay_items", "replay_general_items", "replay_confirmation_items", "replay_successful_items"))
            result["rng"] = all(key in snap for key in ("python_random_state", "numpy_random_state", "torch_random_state"))
            result["episode_counter"] = int(snap.get("episode_count", -1)) == 130
    except Exception as exc:  # diagnostic output, never training
        result["load_error"] = repr(exc)
    result["sha256_matches_summary"] = result["sha256"] == summary.get("checkpoint_sha256")
    result["pass"] = bool(result["final_checkpoint_load"] and result["recovery_load"] and result["online_network"] and result["target_network"] and result["optimizer"] and result["replay"] and result["rng"] and result["episode_counter"])
    return result


def baseline(run5: Path) -> dict[str, Any]:
    path = run5 / "dqn_seed42_summary.json"
    if not path.exists():
        return {**RUN5_FALLBACK, "source": "documented_fallback"}
    raw = load_json(path)
    q = raw.get("reproduction_q_rank", {})
    values = {
        "episodes": raw.get("episodes", 130), "steps": raw.get("steps", 1223),
        "active_candidate": raw.get("active_candidate_transitions", 907),
        "target_available": raw.get("reproduction_targets_available", 685),
        "actionable": raw.get("target_actionable_transitions", 506),
        "selected": raw.get("reproduction_actions_selected", 14),
        "selection_rate": raw.get("selection_rate", 14 / 506),
        "successful": raw.get("successful_reproduction_transitions", 0),
        "semantic": raw.get("NOT_EQUAL_recurrence", 0), "detector": raw.get("Detector_refire", 0),
        "confirmation": raw.get("DQN_REPRODUCTION_confirmations", 0),
        "successful_replay": raw.get("successful_total_sample_draws", 0),
        "mean_rank": q.get("mean", 21.16), "median_rank": q.get("median", 20.0),
        "top1": q.get("top1", 14 / 506), "top3": q.get("top3", 7.51 / 100), "top5": q.get("top5", 12.25 / 100),
        "source": str(path),
    }
    return values


def q_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["reproduction_target_q_rank"]) for row in rows if row.get("target_actionable") and row.get("reproduction_target_q_rank") is not None]
    gaps = []
    for row in rows:
        if row.get("target_actionable") and row.get("reproduction_target_q_rank") is not None:
            # The runner records the rank, but not the two raw Q values.
            # Keep the gap explicitly unavailable instead of inventing it.
            if row.get("reproduction_q_gap") is not None:
                gaps.append(float(row["reproduction_q_gap"]))
    return {"n": len(ranks), "mean_rank": mean(ranks), "median_rank": median(ranks),
            "top1": sum(rank <= 1 for rank in ranks) / max(1, len(ranks)),
            "top3": sum(rank <= 3 for rank in ranks) / max(1, len(ranks)),
            "top5": sum(rank <= 5 for rank in ranks) / max(1, len(ranks)),
            "mean_q_gap": mean(gaps) if gaps else None, "q_gap_available": bool(gaps)}


def funnel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "active_candidate": "active_candidate", "target_available": "reproduction_target_available",
        "mapped": "target_mapped", "actionable": "target_actionable",
        "selected": "reproduction_action_selected", "effect": "candidate_relevant_effect",
        "semantic": "not_equal_recurrence", "detector": "detector_refire",
        "same_chain": "same_logical_key", "confirmation": "dqn_reproduction_confirmation",
    }
    out = {name: sum(bool(row.get(key)) for row in rows) for name, key in keys.items()}
    out["selection_rate"] = out["selected"] / max(1, out["actionable"])
    out["successful_reproduction"] = sum(bool_value(row, "is_successful_reproduction") for row in rows)
    return out


def learning_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, lo, hi in (("early", 1, 43), ("middle", 44, 86), ("late", 87, 130)):
        window = [row for row in rows if lo <= int(row.get("canonical_episode", 0)) <= hi]
        f = funnel(window)
        q = q_stats(window)
        result[name] = {"episodes": [lo, hi], **f, **q}
    result["trend"] = {
        "late_selection_gt_early": result["late"]["selection_rate"] > result["early"]["selection_rate"],
        "late_rank_lt_early": result["late"]["mean_rank"] < result["early"]["mean_rank"] if result["late"]["n"] and result["early"]["n"] else False,
        "late_confirmation_gt_early": result["late"]["confirmation"] > result["early"]["confirmation"],
    }
    return result


def classification(run6: dict[str, Any], run5: dict[str, Any], curve: dict[str, Any], replay: dict[str, Any], reward: dict[str, Any], attribution: dict[str, Any]) -> str:
    health = run6.get("health", {})
    if any(int(health.get(key, 0) or 0) for key in ("nan_q", "inf_q", "nan_loss", "inf_loss", "nan_inf_grad")):
        return "TRAINING_INSTABILITY"
    if attribution.get("status") == "TELEMETRY_MISSING":
        return "TELEMETRY_CONTRACT_MISMATCH"
    if attribution["fp"] or attribution["fn"] or not reward["hierarchy_satisfied"]:
        return "REWARD_SIGNAL_FAILURE"
    if replay["successful_transitions"] and replay["successful_exposure"] is not None and replay["successful_exposure"] < .999:
        return "REPLAY_EXPOSURE_FAILURE"
    improvement = run6["funnel"]["selection_rate"] > run5["selection_rate"] and run6["funnel"]["successful_reproduction"] > 0
    trend = all(curve["trend"].values())
    rank_improved = run6["q"]["n"] and run6["q"]["mean_rank"] < run5["mean_rank"]
    if improvement and rank_improved and trend and run6["funnel"]["confirmation"] > 0:
        return "CLEAR_LEARNING_IMPROVEMENT"
    if run6["funnel"]["successful_reproduction"] or run6["funnel"]["confirmation"]:
        return "PARTIAL_LEARNING_IMPROVEMENT"
    return "NO_MEASURABLE_IMPROVEMENT"


def evaluate(run6_dir: Path, run5_dir: Path) -> dict[str, Any]:
    summary = load_json(run6_dir / "dqn_seed42_summary.json")
    training = load_json(run6_dir / "run6_seed42_training_summary.json") if (run6_dir / "run6_seed42_training_summary.json").exists() else {}
    gate = completion_gate(run6_dir, summary, training)
    if not gate["pass"]:
        return {"status": "RUN6_NOT_COMPLETE", "completion_gate": gate}
    metrics = load_jsonl(run6_dir / "dqn_seed42_metrics.jsonl")
    rows = load_jsonl(run6_dir / "dqn_seed42_transitions.jsonl")
    attach_canonical(rows, metrics)
    f = funnel(rows)
    q = q_stats(rows)
    run5 = baseline(run5_dir)
    attribution_upgrade = [row for row in rows if float(row.get("reward_components", {}).get("confirmation_upgrade", 0.0) or 0.0) > 0]
    lifecycle_fields = ("same_chain_recurrence", "confirmation_event_present", "confirmation_event_key", "confirmation_source", "confirmation_event_step")
    lifecycle_available = bool(attribution_upgrade) and all(all(field in row for field in lifecycle_fields) for row in attribution_upgrade)
    if not lifecycle_available:
        attribution = {"status": "TELEMETRY_MISSING", "actual_confirmations": "MISSING", "rewarded_upgrades": len(attribution_upgrade), "tp": "MISSING", "fp": "MISSING", "fn": "MISSING", "tn": "MISSING", "precision": "MISSING", "recall": "MISSING"}
    else:
        actual = [row for row in rows if bool(row.get("same_logical_key")) and bool(row.get("dqn_reproduction_confirmation"))]
        fp = sum(row not in actual for row in attribution_upgrade)
        fn = sum(row not in attribution_upgrade for row in actual)
        attribution = {"status": "COMPLETE", "actual_confirmations": len(actual), "rewarded_upgrades": len(attribution_upgrade), "tp": len(actual) - fn, "fp": fp, "fn": fn, "tn": len(rows) - fp - fn - (len(actual) - fn), "precision": (len(actual) - fn) / max(1, len(attribution_upgrade)), "recall": (len(actual) - fn) / max(1, len(actual))}
    productive = [row for row in rows if bool(row.get("is_successful_reproduction"))]
    failed = [row for row in rows if bool(row.get("is_failed_reproduction"))]
    unrelated = [row for row in rows if bool(row.get("is_confirmation_context")) and not bool(row.get("is_reproduction_attempt"))]
    reward = {"productive": distribution(productive), "failed": distribution(failed), "unrelated": distribution(unrelated)}
    reward["hierarchy_satisfied"] = reward["productive"]["mean"] > reward["failed"]["mean"] and reward["productive"]["mean"] > reward["unrelated"]["mean"]
    batches = summary.get("batch_telemetry", {}).get("rows", [])
    success_batches = [row for row in batches if row.get("successful_reproduction_samples", 0)]
    replay = {"successful_transitions": sum(bool(row.get("is_successful_reproduction")) for row in rows), "successful_admissions": sum(bool(row.get("is_successful_reproduction")) for row in rows), "failed_admissions": sum(bool(row.get("is_failed_reproduction")) for row in rows), "partial_admissions": sum(bool(row.get("is_partial_reproduction")) for row in rows), "batch_count": len(batches), "successful_batches": len(success_batches), "successful_exposure": (len(success_batches) / len(batches) if success_batches else (None if not any(row.get("is_successful_reproduction") for row in rows) else 0.0)), "unique_successful": summary.get("successful_unique_transitions", 0), "max_draws": summary.get("successful_sample_concentration", {}).get("max_draws_per_transition", 0), "median_draws": summary.get("successful_sample_concentration", {}).get("median_draws_per_transition", 0)}
    action_counts = Counter(str(row.get("selected_action", {}).get("action_type", "other")) for row in rows)
    ar_actions = Counter(str(row.get("selected_action", {}).get("action_type", "other")) for row in rows if row.get("is_confirmation_context"))
    run6 = {"episodes": summary.get("episodes"), "steps": summary.get("steps"), "ppo_steps": summary.get("ppo_steps"), "optimizer_updates": summary.get("optimizer_updates"), "target_updates": summary.get("target_update_count"), "health": summary.get("health", {}), "funnel": f, "q": q, "action_counts": dict(action_counts), "action_replay_action_counts": dict(ar_actions), "infra_failures": sum(1 for row in metrics if row.get("status") == "TRAIN_INFRA_FAILURE"), "checkpoint": checkpoint_validation(run6_dir, summary)}
    curve = learning_curve(rows)
    comparison = {"DQN transitions": [run5["steps"], run6["steps"], run6["steps"] - run5["steps"]], "Active candidate": [run5["active_candidate"], f["active_candidate"], f["active_candidate"] - run5["active_candidate"]], "Target available": [run5["target_available"], f["target_available"], f["target_available"] - run5["target_available"]], "Actionable": [run5["actionable"], f["actionable"], f["actionable"] - run5["actionable"]], "Reproduction selected": [run5["selected"], f["selected"], f["selected"] - run5["selected"]], "Selection rate": [run5["selection_rate"], f["selection_rate"], f["selection_rate"] - run5["selection_rate"]], "Successful reproduction": [run5["successful"], f["successful_reproduction"], f["successful_reproduction"] - run5["successful"]], "Semantic recurrence": [run5["semantic"], f["semantic"], f["semantic"] - run5["semantic"]], "Detector recurrence": [run5["detector"], f["detector"], f["detector"] - run5["detector"]], "Confirmation": [run5["confirmation"], f["confirmation"], f["confirmation"] - run5["confirmation"]], "Successful replay samples": [run5["successful_replay"], summary.get("successful_total_sample_draws", 0), summary.get("successful_total_sample_draws", 0) - run5["successful_replay"]], "Mean Q-rank": [run5["mean_rank"], q["mean_rank"], q["mean_rank"] - run5["mean_rank"]], "Median Q-rank": [run5["median_rank"], q["median_rank"], q["median_rank"] - run5["median_rank"]], "Top-1": [run5["top1"], q["top1"], q["top1"] - run5["top1"]], "Top-3": [run5["top3"], q["top3"], q["top3"] - run5["top3"]], "Top-5": [run5["top5"], q["top5"], q["top5"] - run5["top5"]]}
    result = {"status": "RUN6_COMPLETE", "completion_gate": gate, "run5_baseline": run5, "run6": run6, "candidate_funnel": f, "q_analysis": q, "learning_curve": curve, "reward": reward, "attribution": attribution, "replay": replay, "comparison": comparison, "improvement_factors": {"selection_rate": f["selection_rate"] / run5["selection_rate"] if run5["selection_rate"] else None, "mean_q_rank_delta": q["mean_rank"] - run5["mean_rank"], "successful_reproduction_absolute_gain": f["successful_reproduction"] - run5["successful"], "confirmation_absolute_gain": f["confirmation"] - run5["confirmation"]}, "classification": classification(run6, run5, curve, replay, reward, attribution), "safety": {"training_process_modified": False, "reward_changed": training.get("reward_changed", False), "gt_runtime_access": training.get("gt_runtime_access", False), "ppo_frozen": training.get("ppo_frozen", False), "candidate_a_unchanged": training.get("candidate_a_hashes_unchanged", False), "candidate_b_unchanged": training.get("candidate_b_hashes_unchanged", False)}}
    return result


def self_test() -> None:
    assert completion_gate(Path("."), {"episodes": 5}, {})["pass"] is False
    assert completion_gate(Path("."), {"episodes": 130}, {})["pass"] is True
    assert learning_curve([{"canonical_episode": 1, "target_actionable": 1, "reproduction_target_q_rank": 3, "reproduction_action_selected": 1}])["early"]["selection_rate"] == 1.0
    assert baseline(Path("C:/does/not/exist"))["steps"] == 1223
    print("RUN6_POST_EVALUATOR_SELF_TEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run6-dir", type=Path, default=DEFAULT_RUN6)
    parser.add_argument("--run5-dir", type=Path, default=DEFAULT_RUN5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = evaluate(args.run6_dir, args.run5_dir)
    if result.get("status") == "RUN6_NOT_COMPLETE":
        print("RUN6_NOT_COMPLETE")
        print(json.dumps(result, indent=2))
        return
    out = args.run6_dir / "evaluation"
    write_json(out / "run6_seed42_evaluation_summary.json", result)
    write_json(out / "run5_vs_run6_seed42_comparison.json", result["comparison"])
    write_json(out / "run6_seed42_learning_curve.json", result["learning_curve"])
    write_json(out / "run6_seed42_reward_replay_summary.json", {"reward": result["reward"], "replay": result["replay"], "attribution": result["attribution"]})
    print("RUN6 COMPLETE")
    print(f"Run5 reproduction selection: {result['run5_baseline']['selection_rate']:.2%}")
    print(f"Run6 reproduction selection: {result['candidate_funnel']['selection_rate']:.2%}")
    print(f"Run5 successful reproduction: {result['run5_baseline']['successful']}")
    print(f"Run6 successful reproduction: {result['candidate_funnel']['successful_reproduction']}")
    print(f"Run5 confirmation: {result['run5_baseline']['confirmation']}")
    print(f"Run6 confirmation: {result['candidate_funnel']['confirmation']}")
    print(f"Run5 mean Q-rank: {result['run5_baseline']['mean_rank']:.2f}")
    print(f"Run6 mean Q-rank: {result['q_analysis']['mean_rank']:.2f}")
    print(f"Early → Late selection: {result['learning_curve']['early']['selection_rate']:.2%} → {result['learning_curve']['late']['selection_rate']:.2%}")
    print(f"Early → Late rank: {result['learning_curve']['early']['mean_rank']:.2f} → {result['learning_curve']['late']['mean_rank']:.2f}")
    print(f"CLASSIFICATION={result['classification']}")


if __name__ == "__main__":
    main()
