"""Run PPO stochastic explorer sanity across multiple seeds and aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.runners.run_policy_sanity_check import run_sanity_check


AGGREGATE_METRICS = [
    "total_steps",
    "unique_states",
    "state_novelty_ratio",
    "unique_urls",
    "unique_targets",
    "target_novelty_ratio",
    "unique_action_targets",
    "action_target_novelty_ratio",
    "unique_action_types",
    "repeat_action_target_ratio",
    "repeat_state_ratio",
    "flow_progress_count",
    "candidate_anomaly_count",
    "unique_candidate_signatures",
    "confirmed_anomaly_count",
    "unique_confirmed_signatures",
    "no_effect_count",
    "invalid_action_count",
    "action_error_count",
    "timeout_count",
    "action_type_count",
    "action_type_entropy",
    "policy_entropy_mean",
]


def aggregate_seed_summaries(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in AGGREGATE_METRICS:
        values = [
            float(summary.get("metrics", {}).get(metric))
            for summary in summaries
            if isinstance(summary.get("metrics", {}).get(metric), (int, float))
        ]
        aggregate[metric] = _mean_min_max(values)
    action_distribution: dict[str, dict[str, float]] = {}
    action_types = sorted({key for summary in summaries for key in summary.get("action_distribution", {})})
    for action_type in action_types:
        values = [float(summary.get("action_distribution", {}).get(action_type, 0.0)) for summary in summaries]
        action_distribution[action_type] = _mean_min_max(values)
    action_error_distribution: dict[str, dict[str, float]] = {}
    error_types = sorted({key for summary in summaries for key in summary.get("action_errors_by_type", {})})
    for action_type in error_types:
        values = [float(summary.get("action_errors_by_type", {}).get(action_type, 0.0)) for summary in summaries]
        action_error_distribution[action_type] = _mean_min_max(values)
    return {
        "metrics": aggregate,
        "action_distribution": action_distribution,
        "action_errors_by_type": action_error_distribution,
        "action_error_categories": _aggregate_counter_field(summaries, "action_error_categories"),
        "diagnostic": explorer_role_check(summaries),
    }


def explorer_role_check(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    issues = []
    if not summaries:
        return {"PPO_EXPLORER_ROLE_CHECK": "REVIEW", "issues": ["no_seed_results"]}
    repeat_values = [
        float(summary.get("metrics", {}).get("repeat_action_target_ratio", 1.0) or 0.0)
        for summary in summaries
    ]
    action_target_values = [
        float(summary.get("metrics", {}).get("unique_action_targets", 0.0) or 0.0)
        for summary in summaries
    ]
    action_type_values = [
        float(summary.get("metrics", {}).get("unique_action_types", 0.0) or 0.0)
        for summary in summaries
    ]
    action_errors = [
        float(summary.get("metrics", {}).get("action_error_count", 0.0) or 0.0)
        for summary in summaries
    ]
    if float(np.mean(repeat_values)) >= 0.7:
        issues.append("high_mean_repeat_action_target_ratio")
    if float(np.mean(action_target_values)) <= 5.0:
        issues.append("low_mean_action_target_diversity")
    if float(np.mean(action_type_values)) <= 3.0:
        issues.append("low_mean_action_type_diversity")
    if float(np.mean(action_errors)) > 5.0:
        issues.append("high_mean_action_error_count")
    return {"PPO_EXPLORER_ROLE_CHECK": "REVIEW" if issues else "PASS", "issues": issues}


def build_comparison(greedy_summary: Mapping[str, Any] | None, stochastic_summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    stochastic_aggregate = aggregate_seed_summaries(stochastic_summaries)
    return {
        "greedy_baseline": _compact_summary(greedy_summary) if greedy_summary else {},
        "stochastic_by_seed": [_compact_summary(summary) for summary in stochastic_summaries],
        "stochastic_aggregate": stochastic_aggregate,
        "PPO_GREEDY_DIAGNOSTIC": _greedy_diagnostic(greedy_summary),
        "PPO_EXPLORER_ROLE_CHECK": stochastic_aggregate["diagnostic"]["PPO_EXPLORER_ROLE_CHECK"],
    }


def run_multiseed(args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        output_name = f"ppo_{args.site_id}_stochastic_seed{seed}.json"
        child = argparse.Namespace(
            mode="ppo",
            config=args.config,
            base_url=args.base_url,
            site_id=args.site_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            max_candidates=args.max_candidates,
            seed=int(seed),
            headless=args.headless,
            model_path=args.model_path,
            output_dir=str(output_dir),
            output_name=output_name,
            greedy=False,
            selection_mode="stochastic",
        )
        summary = run_sanity_check(child)
        summary["selection_mode"] = "stochastic"
        summaries.append(_strip_step_summaries(summary))
    aggregate = aggregate_seed_summaries(summaries)
    result = {
        "model_path": args.model_path,
        "site_id": summaries[0].get("site") if summaries else args.site_id,
        "url": summaries[0].get("url") if summaries else args.base_url,
        "episodes_per_seed": args.episodes,
        "max_steps": args.max_steps,
        "seeds": [int(seed) for seed in args.seeds],
        "selection_mode": "stochastic",
        "reward_version": "ppo_reward_config_current",
        "seed_results": summaries,
        "aggregate": aggregate,
    }
    aggregate_path = Path(args.aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    greedy_summary = _load_json(args.greedy_baseline_json) if args.greedy_baseline_json else None
    comparison = build_comparison(greedy_summary, summaries)
    comparison.update(
        {
            "model_path": args.model_path,
            "site_id": result["site_id"],
            "url": result["url"],
            "episodes_per_seed": args.episodes,
            "max_steps": args.max_steps,
            "seeds": result["seeds"],
        }
    )
    comparison_path = Path(args.comparison_output)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(_jsonable(comparison), ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics = build_action_error_diagnostics(summaries, aggregate)
    diagnostics_path = Path(args.error_diagnostics_output)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(_jsonable(diagnostics), ensure_ascii=False, indent=2), encoding="utf-8")
    result["aggregate_path"] = str(aggregate_path)
    result["comparison_path"] = str(comparison_path)
    result["error_diagnostics_path"] = str(diagnostics_path)
    return result


def build_action_error_diagnostics(summaries: list[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    errors_by_action_type = _aggregate_counter_field(summaries, "action_errors_by_type")
    errors_by_category = _aggregate_counter_field(summaries, "action_error_categories")
    failed_targets = _aggregate_failed_targets(summaries)
    synthetic_errors = 0
    negative_bid_errors = 0
    raw_flag_only_errors = 0
    detail_count = 0
    by_seed = []
    for summary in summaries:
        details = summary.get("action_error_details", [])
        if not isinstance(details, list):
            details = []
        detail_count += len(details)
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            bid = str(detail.get("bid") or "")
            source = str(detail.get("source") or "")
            if bid.startswith("-"):
                negative_bid_errors += 1
            if bid.startswith("-") or source in {"bbox_fallback", "fullscreen_clickable_fallback", "emergency_bbox_fallback"}:
                synthetic_errors += 1
            if detail.get("browsergym_raw_last_action_error") and not detail.get("execution_action_error"):
                raw_flag_only_errors += 1
        by_seed.append(
            {
                "seed": summary.get("seed"),
                "total_steps": summary.get("metrics", {}).get("total_steps"),
                "action_error_count": summary.get("metrics", {}).get("action_error_count"),
                "errors_by_action_type": summary.get("action_errors_by_type", {}),
                "errors_by_category": summary.get("action_error_categories", {}),
                "error_rate_by_action_type": summary.get("error_rate_by_action_type", {}),
                "failed_targets": summary.get("failed_target_ids", []),
                "action_error_details": details,
            }
        )
    root_cause = []
    if raw_flag_only_errors:
        root_cause.append("browsergym_raw_last_action_error_was_previously_counted_as_execution_error")
    if negative_bid_errors:
        root_cause.append("negative_bid_errors_observed")
    if not detail_count:
        root_cause.append("no_explicit_execution_errors_after_filtering")
    return {
        "seed_summaries": by_seed,
        "errors_by_action_type": errors_by_action_type,
        "errors_by_category": errors_by_category,
        "error_rate_by_action_type": _aggregate_error_rates(summaries),
        "failed_targets": failed_targets,
        "synthetic_bid_summary": {
            "synthetic_or_fallback_error_count": synthetic_errors,
            "negative_bid_error_count": negative_bid_errors,
        },
        "stale_candidate_summary": {
            "candidate_stale_count": int(errors_by_category.get("candidate-stale", 0) or 0),
            "candidate_index_mismatch_count": int(errors_by_category.get("candidate-index-mismatch", 0) or 0),
        },
        "ppo_vs_dqn_mask_comparison": {
            "shared_action_space_build_action_mask": True,
            "safe_action_mask_extra_dqn_step": "DQN can suppress explicitly repeated local actions, but execution-validity comes from ActionSpace.build_action_mask().",
        },
        "root_cause_summary": root_cause,
        "aggregate": aggregate,
    }


def _compact_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    return {
        "seed": summary.get("seed"),
        "selection_mode": summary.get("selection_mode"),
        "metrics": summary.get("metrics", {}),
        "action_distribution": summary.get("action_distribution", {}),
        "action_errors_by_type": summary.get("action_errors_by_type", {}),
        "action_error_categories": summary.get("action_error_categories", {}),
        "error_rate_by_action_type": summary.get("error_rate_by_action_type", {}),
        "candidate_signatures": summary.get("candidate_signatures", []),
        "confirmed_signatures": summary.get("confirmed_signatures", []),
        "top_repeated_actions": summary.get("top_repeated_actions", []),
        "diagnostic": summary.get("diagnostic", {}),
        "summary_path": summary.get("summary_path", ""),
    }


def _greedy_diagnostic(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return "N/A"
    metrics = summary.get("metrics", {})
    actions = summary.get("action_distribution", {})
    total = float(metrics.get("total_steps", 0.0) or 0.0)
    dominant = max((float(value or 0.0) for value in actions.values()), default=0.0)
    if total and dominant / total >= 0.7:
        return "REVIEW"
    if float(metrics.get("repeat_action_target_ratio", 0.0) or 0.0) >= 0.7:
        return "REVIEW"
    return "PASS"


def _strip_step_summaries(summary: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(summary)
    item.pop("step_summaries", None)
    return item


def _mean_min_max(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def _aggregate_counter_field(summaries: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for summary in summaries:
        field_values = summary.get(field, {})
        if not isinstance(field_values, Mapping):
            continue
        for key, value in field_values.items():
            values[str(key)] = values.get(str(key), 0) + int(value or 0)
    return values


def _aggregate_error_rates(summaries: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    action_types = sorted({key for summary in summaries for key in summary.get("action_distribution", {})})
    for action_type in action_types:
        selected = sum(int(summary.get("action_distribution", {}).get(action_type, 0) or 0) for summary in summaries)
        errors = sum(int(summary.get("action_errors_by_type", {}).get(action_type, 0) or 0) for summary in summaries)
        rates[action_type] = {"selected": float(selected), "errors": float(errors), "error_rate": float(errors / selected) if selected else 0.0}
    return rates


def _aggregate_failed_targets(summaries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for summary in summaries:
        for row in summary.get("failed_target_ids", []) or []:
            if not isinstance(row, Mapping):
                continue
            target_id = str(row.get("target_id") or "")
            if target_id:
                counts[target_id] = counts.get(target_id, 0) + int(row.get("count", 0) or 0)
    return [
        {"target_id": target_id, "count": count}
        for target_id, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    ]


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model-path", default="artifacts/models/site001_ppo_rewardv2_sanity.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="artifacts/sanity")
    parser.add_argument("--aggregate-output", default="artifacts/sanity/ppo_site001_stochastic_multiseed.json")
    parser.add_argument("--comparison-output", default="artifacts/sanity/ppo_site001_greedy_vs_stochastic.json")
    parser.add_argument("--error-diagnostics-output", default="artifacts/sanity/ppo_stochastic_action_error_diagnostics.json")
    parser.add_argument("--greedy-baseline-json", default="artifacts/sanity/ppo_site001_rewardv2_sanity.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_multiseed(args)
    if args.quiet:
        print(result["aggregate_path"])
        print(result["comparison_path"])
        print(result["error_diagnostics_path"])
    else:
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
