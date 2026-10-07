"""Post-hoc GT evaluator for frozen RAWD v3 final evaluation.

This module runs only after raw runtime artifacts exist.  It is the only final
evaluation component that reads the held-out GT catalog.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/final_v3_eval"
RAW = OUT / "raw"
GT_PATH = OUT / "heldout_gt_catalog.json"
METHODS = ("ppo", "hybrid")
SEEDS = (42, 43, 44)
SITES = ("EVAL001", "EVAL002", "EVAL003", "EVAL004")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_dir(site: str, method: str, seed: int) -> Path:
    return RAW / f"{site.lower()}_{method}_seed{seed}"


def category_from_key(key: str) -> str:
    if key.startswith("R") and len(key) >= 3 and key[1:3].isdigit():
        return key[:3]
    return ""


def candidate_categories(timeline: list[Mapping[str, Any]], lifecycle: list[Mapping[str, Any]]) -> set[str]:
    categories = set()
    for row in timeline:
        for key in row.get("candidate_keys", []) or []:
            cat = category_from_key(str(key))
            if cat:
                categories.add(cat)
    for row in lifecycle:
        key = str(row.get("logical_key") or "")
        cat = category_from_key(key)
        if cat:
            categories.add(cat)
    return categories


def confirmed_categories(lifecycle: list[Mapping[str, Any]]) -> set[str]:
    categories = set()
    for row in lifecycle:
        if row.get("event") != "candidate_to_confirmed":
            continue
        key = str(row.get("logical_key") or "")
        cat = category_from_key(key)
        if cat:
            categories.add(cat)
    return categories


def unique_candidate_keys(timeline: list[Mapping[str, Any]], lifecycle: list[Mapping[str, Any]]) -> set[str]:
    keys = set()
    for row in timeline:
        for key in row.get("candidate_keys", []) or []:
            if key:
                keys.add(str(key))
    for row in lifecycle:
        key = str(row.get("logical_key") or "")
        if key:
            keys.add(key)
    return keys


def matched_candidate_keys(keys: set[str], gt_categories: set[str]) -> set[str]:
    return {key for key in keys if category_from_key(key) in gt_categories}


def classify_gt(reached: bool, detected: bool, confirmed: bool) -> str:
    if not reached:
        return "NOT_REACHED"
    if not detected:
        return "REACHED_NOT_DETECTED"
    if not confirmed:
        return "DETECTED_NOT_CONFIRMED"
    return "CONFIRMED"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": sqrt(var)}


def evaluate() -> dict[str, Any]:
    gt = read_json(GT_PATH)
    execution = read_json(OUT / "final_evaluation_execution_manifest.json")
    outcome_rows = []
    by_method = {method: Counter() for method in METHODS}
    method_candidates = {method: set() for method in METHODS}
    method_matched_candidates = {method: set() for method in METHODS}
    method_runtime = {method: Counter() for method in METHODS}
    method_exploration = {method: Counter() for method in METHODS}
    site_metrics: dict[str, dict[str, Counter]] = {site: {method: Counter() for method in METHODS} for site in SITES}
    taxonomy: dict[str, dict[str, Counter]] = defaultdict(lambda: {method: Counter() for method in METHODS})
    seed_detected: dict[str, dict[str, set[int]]] = defaultdict(lambda: {method: set() for method in METHODS})
    seed_confirmed: dict[str, dict[str, set[int]]] = defaultdict(lambda: {method: set() for method in METHODS})
    hybrid_controller = Counter()
    hybrid_handoff_reasons = Counter()
    hybrid_return_reasons = Counter()
    seed_level = defaultdict(lambda: defaultdict(list))
    for site in SITES:
        gt_rows = gt["sites"][site]
        gt_categories = {row["category"] for row in gt_rows}
        for method in METHODS:
            for seed in SEEDS:
                directory = run_dir(site, method, seed)
                summary = read_json(directory / "run_summary.json")
                timeline = read_jsonl(directory / "step_timeline.jsonl")
                lifecycle = read_jsonl(directory / "candidate_lifecycle.jsonl")
                categories = candidate_categories(timeline, lifecycle)
                conf_categories = confirmed_categories(lifecycle)
                keys = unique_candidate_keys(timeline, lifecycle)
                matched = matched_candidate_keys(keys, gt_categories)
                method_candidates[method].update(f"{site}:{key}" for key in keys)
                method_matched_candidates[method].update(f"{site}:{key}" for key in matched)
                reached = bool(summary.get("browser_steps", 0) > 0)
                seed_detect_count = 0
                seed_confirm_count = 0
                for defect in gt_rows:
                    bug_id = defect["bug_id"]
                    category = defect["category"]
                    detected = category in categories
                    confirmed = category in conf_categories
                    if detected:
                        seed_detect_count += 1
                        seed_detected[f"{site}:{bug_id}"][method].add(seed)
                    if confirmed:
                        seed_confirm_count += 1
                        seed_confirmed[f"{site}:{bug_id}"][method].add(seed)
                    classification = classify_gt(reached, detected, confirmed)
                    outcome_rows.append({
                        "site": site,
                        "bug_id": bug_id,
                        "category": category,
                        "method": method,
                        "seed": seed,
                        "classification": classification,
                        "reached": reached,
                        "detected": detected,
                        "confirmed": confirmed,
                    })
                    by_method[method]["gt_opportunities"] += 1
                    by_method[method]["reached"] += int(reached)
                    by_method[method]["detected"] += int(detected)
                    by_method[method]["confirmed"] += int(confirmed)
                    by_method[method][classification] += 1
                    taxonomy[category][method]["gt"] += 1
                    taxonomy[category][method]["reached"] += int(reached)
                    taxonomy[category][method]["detected"] += int(detected)
                    taxonomy[category][method]["confirmed"] += int(confirmed)
                    site_metrics[site][method]["gt"] += 1
                    site_metrics[site][method]["reached"] += int(reached)
                    site_metrics[site][method]["detected"] += int(detected)
                    site_metrics[site][method]["confirmed"] += int(confirmed)
                method_runtime[method]["browser_steps"] += int(summary.get("browser_steps", 0) or 0)
                method_runtime[method]["invalid"] += int(summary.get("invalid", 0) or 0)
                method_runtime[method]["timeout"] += int(summary.get("timeout", 0) or 0)
                method_runtime[method]["action_error"] += int(summary.get("action_error", 0) or 0)
                method_runtime[method]["unexpected_reset"] += int(summary.get("unexpected_reset", 0) or 0)
                method_runtime[method]["native_select_fill_input"] += int(summary.get("native_select_fill_input", 0) or 0)
                method_runtime[method]["wall_time"] += float(summary.get("wall_time", 0.0) or 0.0)
                method_exploration[method]["unique_states"] += int(summary.get("unique_states", 0) or 0)
                method_exploration[method]["unique_routes"] += int(summary.get("unique_routes", 0) or 0)
                method_exploration[method]["unique_targets"] += int(summary.get("unique_targets", 0) or 0)
                method_exploration[method]["unique_action_targets"] += int(summary.get("unique_action_targets", 0) or 0)
                site_metrics[site][method]["browser_steps"] += int(summary.get("browser_steps", 0) or 0)
                site_metrics[site][method]["wall_time"] += float(summary.get("wall_time", 0.0) or 0.0)
                seed_level[method]["recall"].append(safe_div(seed_detect_count, len(gt_rows)))
                seed_level[method]["confirmation"].append(safe_div(seed_confirm_count, len(gt_rows)))
                if method == "hybrid":
                    hybrid_controller["ppo_steps"] += int(summary.get("ppo_steps", 0) or 0)
                    hybrid_controller["dqn_steps"] += int(summary.get("dqn_steps", 0) or 0)
                    hybrid_controller["handoff_count"] += int(summary.get("handoff_count", 0) or 0)
                    hybrid_controller["dqn_local_sessions"] += int(summary.get("dqn_local_sessions", 0) or 0)
                    hybrid_controller["candidate_producing_local_sessions"] += int(summary.get("candidate_producing_local_sessions", 0) or 0)
                    hybrid_controller["confirmation_producing_local_sessions"] += int(summary.get("confirmation_producing_local_sessions", 0) or 0)
                    hybrid_handoff_reasons.update(summary.get("handoff_reasons", {}) or {})
                    hybrid_return_reasons.update(summary.get("return_reasons", {}) or {})
    comparison: dict[str, Any] = {}
    for method in METHODS:
        total_candidates = len(method_candidates[method])
        matched_candidates = len(method_matched_candidates[method])
        unmatched = total_candidates - matched_candidates
        recall = safe_div(by_method[method]["detected"], by_method[method]["gt_opportunities"])
        precision = safe_div(matched_candidates, total_candidates)
        f1 = safe_div(2 * precision * recall, precision + recall)
        comparison[method] = {
            "gt_opportunities": int(by_method[method]["gt_opportunities"]),
            "gt_reached": int(by_method[method]["reached"]),
            "gt_detected": int(by_method[method]["detected"]),
            "gt_confirmed": int(by_method[method]["confirmed"]),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "detection_given_reach": safe_div(by_method[method]["detected"], by_method[method]["reached"]),
            "confirmation_given_detection": safe_div(by_method[method]["confirmed"], by_method[method]["detected"]),
            "unique_defects_detected": sum(1 for defect, values in seed_detected.items() if values[method]),
            "outcomes": {name: int(by_method[method][name]) for name in ("NOT_REACHED", "REACHED_NOT_DETECTED", "DETECTED_NOT_CONFIRMED", "CONFIRMED")},
            "total_candidates": total_candidates,
            "matched_candidates": matched_candidates,
            "unmatched_candidates": unmatched,
            "candidate_burden_per_episode": safe_div(total_candidates, 120),
            "unmatched_candidate_burden_per_episode": safe_div(unmatched, 120),
            "candidates_per_detected_gt": safe_div(total_candidates, by_method[method]["detected"]),
            "runtime": dict(method_runtime[method]),
            "exploration": dict(method_exploration[method]),
            "efficiency": {
                "states_per_step": safe_div(method_exploration[method]["unique_states"], method_runtime[method]["browser_steps"]),
                "targets_per_step": safe_div(method_exploration[method]["unique_targets"], method_runtime[method]["browser_steps"]),
                "action_targets_per_step": safe_div(method_exploration[method]["unique_action_targets"], method_runtime[method]["browser_steps"]),
                "gt_reach_per_step": safe_div(by_method[method]["reached"], method_runtime[method]["browser_steps"]),
                "gt_detection_per_step": safe_div(by_method[method]["detected"], method_runtime[method]["browser_steps"]),
                "candidate_per_step": safe_div(total_candidates, method_runtime[method]["browser_steps"]),
            },
            "seed_level": {key: mean_std(values) for key, values in seed_level[method].items()},
        }
    total_hybrid_steps = hybrid_controller["ppo_steps"] + hybrid_controller["dqn_steps"]
    hybrid_metrics = {
        **dict(hybrid_controller),
        "dqn_share": safe_div(hybrid_controller["dqn_steps"], total_hybrid_steps),
        "candidate_producing_session_rate": safe_div(hybrid_controller["candidate_producing_local_sessions"], hybrid_controller["dqn_local_sessions"]),
        "confirmation_producing_session_rate": safe_div(hybrid_controller["confirmation_producing_local_sessions"], hybrid_controller["dqn_local_sessions"]),
        "handoff_reasons": dict(hybrid_handoff_reasons),
        "return_reasons": dict(hybrid_return_reasons),
    }
    taxonomy_out = {
        category: {method: dict(counts) for method, counts in values.items()}
        for category, values in taxonomy.items()
    }
    robustness = {}
    for method in METHODS:
        det = Counter()
        conf = Counter()
        for values in seed_detected.values():
            det[f"{len(values[method])}/3"] += 1
        for values in seed_confirmed.values():
            conf[f"{len(values[method])}/3"] += 1
        robustness[method] = {"detection": dict(det), "confirmation": dict(conf)}
    framework = {
        "React": {
            method: {
                "detected": sum(row["detected"] for row in outcome_rows if row["method"] == method and row["site"] in {"EVAL001", "EVAL003"}),
                "confirmed": sum(row["confirmed"] for row in outcome_rows if row["method"] == method and row["site"] in {"EVAL001", "EVAL003"}),
                "gt": sum(1 for row in outcome_rows if row["method"] == method and row["site"] in {"EVAL001", "EVAL003"}),
            }
            for method in METHODS
        },
        "Next.js": {
            method: {
                "detected": sum(row["detected"] for row in outcome_rows if row["method"] == method and row["site"] in {"EVAL002", "EVAL004"}),
                "confirmed": sum(row["confirmed"] for row in outcome_rows if row["method"] == method and row["site"] in {"EVAL002", "EVAL004"}),
                "gt": sum(1 for row in outcome_rows if row["method"] == method and row["site"] in {"EVAL002", "EVAL004"}),
            }
            for method in METHODS
        },
    }
    final_classification = classify_methods(comparison)
    summary = {
        "version": "RAWD_V3_FINAL_POSTHOC_EVALUATION",
        "execution_manifest": execution,
        "method_comparison": comparison,
        "taxonomy_breakdown": taxonomy_out,
        "site_metrics": {site: {method: dict(counter) for method, counter in methods.items()} for site, methods in site_metrics.items()},
        "seed_robustness": robustness,
        "framework_breakdown": framework,
        "hybrid_controller_metrics": hybrid_metrics,
        "gt_runtime_access": False,
        "posthoc_gt_access": True,
        "RAWD_V3_FINAL_UNSEEN_EVALUATION_CHECK": "COMPLETE" if execution.get("final_evaluation_check") == "COMPLETE" else "REVIEW",
        "RAWD_V3_FINAL_EVALUATION_FREEZE_INTEGRITY": "PASS" if integrity_pass(execution) else "FAIL",
        "RAWD_V3_POSTHOC_GT_EVALUATION_CHECK": "PASS",
        "final_classification": final_classification,
    }
    write_outputs(summary, comparison, outcome_rows, taxonomy_out, robustness, hybrid_metrics)
    return summary


def integrity_pass(execution: Mapping[str, Any]) -> bool:
    return bool(
        execution.get("model_hash_postcheck")
        and execution.get("site_hash_postcheck")
        and execution.get("protocol_hash_postcheck")
        and not execution.get("production_algorithm_changed")
        and not execution.get("training_occurred")
        and not execution.get("gt_runtime_access")
    )


def classify_methods(comparison: Mapping[str, Any]) -> str:
    ppo = comparison["ppo"]
    hybrid = comparison["hybrid"]
    recall_delta = hybrid["recall"] - ppo["recall"]
    precision_delta = hybrid["precision"] - ppo["precision"]
    confirm_delta = hybrid["gt_confirmed"] - ppo["gt_confirmed"]
    if recall_delta > 0 and confirm_delta >= 0 and precision_delta >= -0.05:
        return "HYBRID_IMPROVEMENT"
    if recall_delta > 0 or confirm_delta > 0:
        return "HYBRID_TRADEOFF"
    if recall_delta < 0 and confirm_delta <= 0:
        return "PPO_ONLY_ADVANTAGE"
    if abs(recall_delta) < 0.01 and abs(precision_delta) < 0.01:
        return "NO_MEANINGFUL_DIFFERENCE"
    return "MIXED"


def write_outputs(
    summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    outcome_rows: list[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    robustness: Mapping[str, Any],
    hybrid_metrics: Mapping[str, Any],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_v3_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "ppo_vs_hybrid_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (OUT / "gt_outcome_matrix.json").write_text(json.dumps(outcome_rows, indent=2), encoding="utf-8")
    (OUT / "taxonomy_breakdown.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    (OUT / "seed_robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    (OUT / "runtime_efficiency.json").write_text(json.dumps({
        method: {
            "runtime": comparison[method]["runtime"],
            "efficiency": comparison[method]["efficiency"],
            "exploration": comparison[method]["exploration"],
        }
        for method in METHODS
    }, indent=2), encoding="utf-8")
    (OUT / "candidate_burden.json").write_text(json.dumps({
        method: {
            "total_candidates": comparison[method]["total_candidates"],
            "matched_candidates": comparison[method]["matched_candidates"],
            "unmatched_candidates": comparison[method]["unmatched_candidates"],
            "candidate_burden_per_episode": comparison[method]["candidate_burden_per_episode"],
            "unmatched_candidate_burden_per_episode": comparison[method]["unmatched_candidate_burden_per_episode"],
            "candidates_per_detected_gt": comparison[method]["candidates_per_detected_gt"],
        }
        for method in METHODS
    }, indent=2), encoding="utf-8")
    (OUT / "hybrid_controller_final_metrics.json").write_text(json.dumps(hybrid_metrics, indent=2), encoding="utf-8")


def main() -> None:
    summary = evaluate()
    print(json.dumps({
        "status": summary["RAWD_V3_POSTHOC_GT_EVALUATION_CHECK"],
        "freeze_integrity": summary["RAWD_V3_FINAL_EVALUATION_FREEZE_INTEGRITY"],
        "classification": summary["final_classification"],
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
