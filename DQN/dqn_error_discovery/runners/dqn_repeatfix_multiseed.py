"""Aggregate fixed-configuration DQN repeatfix multi-seed diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ACTION_TYPES = (
    "noop", "click_element", "scroll_down", "scroll_up", "inspect_dom",
    "inspect_layout", "inspect_network", "inspect_console", "inspect_cart",
    "fill_input", "press_enter", "change_viewport_mobile",
    "change_viewport_desktop", "finish_episode",
)


def _ratio(value: Any, total: Any) -> float:
    try:
        return float(value or 0.0) / float(total or 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _entropy(counts: Mapping[str, Any]) -> float:
    total = sum(float(v or 0.0) for v in counts.values())
    if total <= 0:
        return 0.0
    probabilities = [float(v or 0.0) / total for v in counts.values() if float(v or 0.0) > 0]
    return float(-sum(p * np.log(p) for p in probabilities))


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "std": None}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    actions = report.get("action_distribution", {})
    total = int(metrics.get("total_steps", 0) or 0)
    dominant_type, dominant_count = (max(actions.items(), key=lambda item: item[1]) if actions else (None, 0))
    q_summary = report.get("q_value_diagnostic_summary", {})
    q_by_type = q_summary.get("max_q_by_action_type", {}) if isinstance(q_summary, Mapping) else {}
    type_values: dict[str, list[float]] = {}
    for row in report.get("step_summaries", []) or []:
        diagnostic = row.get("q_diagnostic", {}) if isinstance(row, Mapping) else {}
        for key, value in (diagnostic.get("type_value_by_action_type", {}) or {}).items():
            type_values.setdefault(str(key), []).append(float(value))
    type_means = {key: float(np.mean(values)) for key, values in type_values.items() if values}
    top_values = sorted(type_means.items(), key=lambda item: item[1], reverse=True)
    return {
        "total_steps": total,
        "action_distribution": dict(actions),
        "click_ratio": _ratio(actions.get("click_element", 0), total),
        "viewport_ratio": _ratio(sum(actions.get(k, 0) for k in ("change_viewport_mobile", "change_viewport_desktop")), total),
        "scroll_ratio": _ratio(sum(actions.get(k, 0) for k in ("scroll_down", "scroll_up")), total),
        "dominant_action_type": dominant_type,
        "dominant_action_ratio": _ratio(dominant_count, total),
        "unique_action_types": metrics.get("action_type_count", 0),
        "action_type_entropy": metrics.get("action_type_entropy", 0.0),
        "unique_local_action_targets": metrics.get("unique_local_action_targets", 0),
        "functional_local_states": metrics.get("unique_local_states", 0),
        "raw_local_states": None,
        "repeat_action_ratio": metrics.get("repeat_local_action_ratio", 0.0),
        "recent_window_repeat_ratio": metrics.get("recent_window_repeat_ratio", 0.0),
        "cycle_repeat_ratio": metrics.get("cycle_repeat_ratio", 0.0),
        "unproductive_repeat_ratio": metrics.get("unproductive_repeat_ratio", 0.0),
        "useful_reproduction_repeat_count": metrics.get("useful_reproduction_repeat_count", 0),
        "candidate_count": metrics.get("candidate_anomaly_count", 0),
        "unique_candidate_signatures": metrics.get("unique_candidate_signatures", 0),
        "confirmed_count": metrics.get("confirmed_anomaly_count", 0),
        "candidate_to_confirmed_count": metrics.get("candidate_to_confirmed_count", 0),
        "action_error_count": metrics.get("action_error_count", 0),
        "mean_reward": _ratio(report.get("reward_component_totals", {}).get("final_reward", 0.0), total),
        "dominant_action_q": q_by_type.get(dominant_type) if dominant_type else None,
        "type_value_top1": top_values[0] if top_values else None,
        "type_value_top2": top_values[1] if len(top_values) > 1 else None,
        "type_value_top_gap": (top_values[0][1] - top_values[1][1]) if len(top_values) > 1 else None,
        "softmax_type_entropy_temperature_1": _entropy(q_by_type),
        "repeat_breakdown": dict(report.get("repeat_breakdown", {})),
        "action_type_repeat_breakdown": dict(report.get("action_type_repeat_breakdown", {})),
    }


def _training_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    actions = manifest.get("action_distribution", {})
    total = sum(int(v or 0) for v in actions.values())
    episodes = manifest.get("episode_summaries", [])
    blocks = {}
    for start, end in ((1, 10), (11, 20), (21, 30)):
        items = [x for x in episodes if start <= int(x.get("episode", 0) or 0) <= end]
        dist: dict[str, int] = {}
        greedy: dict[str, int] = {}
        random: dict[str, int] = {}
        for item in items:
            for target, source in ((dist, "action_distribution"), (greedy, "greedy_action_distribution"), (random, "random_action_distribution")):
                for key, value in (item.get(source, {}) or {}).items():
                    target[key] = target.get(key, 0) + int(value or 0)
        block_total = sum(dist.values())
        greedy_total = sum(greedy.values())
        random_total = sum(random.values())
        repeat = {}
        for item in items:
            for key, value in (item.get("repeat_event_counts", {}) or {}).items():
                repeat[key] = repeat.get(key, 0) + int(value or 0)
        blocks[f"EP{start}-{end}"] = {
            "action_distribution": dist,
            "greedy_action_distribution": greedy,
            "random_action_distribution": random,
            "dominant_action_type": max(dist, key=dist.get) if dist else None,
            "dominant_action_ratio": _ratio(max(dist.values()) if dist else 0, block_total),
            "action_type_entropy": _entropy(dist),
            "random_click_ratio": _ratio(random.get("click_element", 0), random_total),
            "greedy_click_ratio": _ratio(greedy.get("click_element", 0), greedy_total),
            "overall_click_ratio": _ratio(dist.get("click_element", 0), block_total),
            "repeat_event_counts": repeat,
            "mean_reward": float(np.mean([float(x.get("episode_reward", 0.0) or 0.0) for x in items])) if items else 0.0,
        }
    return {
        "total_transitions": total,
        "random_action_count": manifest.get("random_action_count", 0),
        "greedy_action_count": manifest.get("greedy_action_count", 0),
        "action_distribution": dict(actions),
        "random_action_distribution": dict(manifest.get("random_action_distribution", {})),
        "greedy_action_distribution": dict(manifest.get("greedy_action_distribution", {})),
        "random_click_ratio": _ratio(manifest.get("random_action_distribution", {}).get("click_element", 0), manifest.get("random_action_count", 0)),
        "greedy_click_ratio": _ratio(manifest.get("greedy_action_distribution", {}).get("click_element", 0), manifest.get("greedy_action_count", 0)),
        "overall_click_ratio": _ratio(actions.get("click_element", 0), total),
        "action_type_entropy": _entropy(actions),
        "mean_reward": manifest.get("mean_episode_reward", 0.0),
        "mean_loss": manifest.get("mean_loss", 0.0),
        "blocks": blocks,
    }


def aggregate(base: Path, seeds: list[int]) -> dict[str, Any]:
    rows = {}
    training = {}
    for seed in seeds:
        manifest = _load(base / "artifacts" / "models" / f"site001_dqn_factorized_repeatfix_seed{seed}_30ep.manifest.json")
        report = _load(base / "artifacts" / "sanity" / f"dqn_site001_repeatfix_seed{seed}.json")
        attribution_path = base / "artifacts" / "sanity" / f"dqn_site001_repeatfix_seed{seed}_attribution.json"
        attribution = _load(attribution_path) if attribution_path.exists() else {}
        rows[str(seed)] = _eval_summary(report)
        confirmation = attribution.get("confirmation_progress", {}) if isinstance(attribution, Mapping) else {}
        repeat_diag = attribution.get("repeat_diagnostics", {}) if isinstance(attribution, Mapping) else {}
        rows[str(seed)]["confirmation_opportunities"] = {
            "candidate_needing_reproduction_count": len(confirmation.get("pending_candidates", []) or []),
            "matching_reproduction_opportunity_count": sum(
                1 for item in confirmation.get("pending_candidates", []) or [] if item.get("followup_actions")
            ),
            "actual_matching_reproduction_action_count": int(confirmation.get("useful_reproduction_repeat_count", 0) or 0),
            "confirmation_progress_count": int(confirmation.get("useful_reproduction_repeat_count", 0) or 0),
            "confirmed_count": int(confirmation.get("confirmed_count", 0) or 0),
            "repeat_blind_spot_ratio": repeat_diag.get("repeat_blind_spot_ratio"),
            "cycle_examples": repeat_diag.get("cycle_examples", {}),
        }
        training[str(seed)] = _training_summary(manifest)
    numeric_fields = ["click_ratio", "dominant_action_ratio", "unique_action_types", "action_type_entropy", "unique_local_action_targets", "repeat_action_ratio", "recent_window_repeat_ratio", "cycle_repeat_ratio", "unproductive_repeat_ratio", "candidate_count", "confirmed_count", "action_error_count", "mean_reward"]
    aggregate_rows = {}
    for field in numeric_fields:
        values = [float(rows[str(seed)].get(field, 0.0) or 0.0) for seed in seeds]
        aggregate_rows[field] = _summary(values)
    dominant_types = [rows[str(seed)]["dominant_action_type"] for seed in seeds]
    observation_capability = {
        "obs_dim": 635,
        "pending_confirmation_present": False,
        "expected_reproduction_action_or_target_present": False,
        "recent_action_history_present": False,
        "repeat_or_cycle_state_present": False,
        "history_features_present": ["previous_action_type", "step_index", "no_change_steps"],
        "basis": "models/observation_encoder.py history features; pending/repeat trackers remain reward-side state",
    }
    return {
        "schema_version": "dqn-repeatfix-multiseed-v1",
        "configuration": {
            "site_id": "site001", "seeds": seeds, "episodes": 30, "max_steps": 20,
            "max_candidates": 32, "architecture": "factorized_v1",
            "exploration_sampling_mode": "action_type_balanced",
            "repeat_semantics": "confirmation_aware_immediate_recent_cycle",
        },
        "seed_summaries": rows,
        "training_summaries": training,
        "aggregate_numeric": aggregate_rows,
        "action_distribution_by_seed": {str(seed): rows[str(seed)]["action_distribution"] for seed in seeds},
        "collapse_classification": {
            "dominant_types": dominant_types,
            "type_stable": len(set(dominant_types)) == 1,
            "type_unstable": len(set(dominant_types)) > 1 and all(rows[str(seed)]["dominant_action_ratio"] >= 0.5 for seed in seeds),
        },
        "confirmation_opportunities": {
            "available_in_standard_sanity_artifact": False,
            "by_seed": {str(seed): rows[str(seed)]["confirmation_opportunities"] for seed in seeds},
            "confirmation_progress_count": int(sum(rows[str(seed)]["confirmation_opportunities"]["confirmation_progress_count"] for seed in seeds)),
            "confirmed_count": int(sum(rows[str(seed)]["confirmation_opportunities"]["confirmed_count"] for seed in seeds)),
        },
        "observation_capability": observation_capability,
        "root_cause_summary": [
            "multi_seed_evidence_is_required_before_seed42_specific_reward_changes",
            "reward_and_repeat_trackers_depend_on state not fully represented in 635-d observation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_repeatfix_multiseed.json")
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    result = aggregate(Path(args.base), [int(value) for value in args.seeds.split(",") if value.strip()])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
