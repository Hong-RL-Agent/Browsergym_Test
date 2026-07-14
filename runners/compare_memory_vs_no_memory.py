from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.policy_safe_metrics import write_json


SUMMARY_FIELDS = [
    "no_memory_model_path",
    "memory_model_path",
    "seed",
    "episodes",
    "max_steps",
    "fallback_mode",
    "disable_fallback",
    "compared_site_ids",
    "no_memory_avg_reward",
    "memory_avg_reward",
    "no_memory_repeated_action_rate",
    "memory_repeated_action_rate",
    "repeated_action_rate_delta",
    "no_memory_action_diversity_score",
    "memory_action_diversity_score",
    "action_diversity_delta",
    "no_memory_unique_clicked_target_count",
    "memory_unique_clicked_target_count",
    "unique_clicked_target_delta",
    "no_memory_unique_clicked_element_key_count",
    "memory_unique_clicked_element_key_count",
    "unique_clicked_element_key_delta",
    "no_memory_repeated_action_signature_count",
    "memory_repeated_action_signature_count",
    "repeated_action_signature_delta",
    "no_memory_repeated_action_type_count",
    "memory_repeated_action_type_count",
    "repeated_action_type_delta",
    "no_memory_repeated_element_key_click_count",
    "memory_repeated_element_key_click_count",
    "repeated_element_key_click_delta",
    "no_memory_action_fallback_exploration_redirect_count",
    "memory_action_fallback_exploration_redirect_count",
    "no_memory_verification_action_after_high_value_click_count",
    "memory_verification_action_after_high_value_click_count",
    "no_memory_low_value_generic_no_response_suppressed_count",
    "memory_low_value_generic_no_response_suppressed_count",
    "no_memory_paper_unique_clicked_element_key_count",
    "memory_paper_unique_clicked_element_key_count",
    "no_memory_paper_repeated_action_signature_count",
    "memory_paper_repeated_action_signature_count",
    "no_memory_paper_action_fallback_count",
    "memory_paper_action_fallback_count",
    "no_memory_paper_action_diversity_score",
    "memory_paper_action_diversity_score",
    "no_memory_paper_verification_after_high_value_count",
    "memory_paper_verification_after_high_value_count",
    "no_memory_paper_unique_detected_candidates",
    "memory_paper_unique_detected_candidates",
    "no_memory_paper_known_bug_reward_total",
    "memory_paper_known_bug_reward_total",
    "no_memory_paper_catalog_guided_action_count",
    "memory_paper_catalog_guided_action_count",
    "no_memory_fallback_applied_count",
    "memory_fallback_applied_count",
    "no_memory_fallback_applied_rate",
    "memory_fallback_applied_rate",
    "fallback_applied_rate_delta",
    "no_memory_policy_executed_action_mismatch_count",
    "memory_policy_executed_action_mismatch_count",
    "policy_executed_action_mismatch_delta",
    "no_memory_fallback_penalty_total",
    "memory_fallback_penalty_total",
    "no_memory_fallback_reward_capped_count",
    "memory_fallback_reward_capped_count",
    "no_memory_functional_action_count",
    "memory_functional_action_count",
    "functional_action_delta",
    "no_memory_state_coverage_count",
    "memory_state_coverage_count",
    "state_coverage_delta",
    "no_memory_unique_detected_anomaly_count",
    "memory_unique_detected_anomaly_count",
    "unique_detected_anomaly_delta",
    "memory_better_on_repetition",
    "memory_better_on_diversity",
    "memory_better_on_state_coverage",
    "comparison_valid",
    "comparison_invalid_reason",
    "strict_preflight",
    "allow_partial_sites",
    "invalid_site_ids",
    "invalid_site_reasons",
    "excluded_site_ids",
    "failed_site_count",
    "failed_site_ids",
    "timeout_excluded_from_comparison_count",
    "partial_site_excluded_count",
    "zero_interaction_site_ids",
    "connection_refused_site_ids",
    "all_enabled_sites_interacted",
    "comparison_failed_fast",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory-config", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--no-memory-model", required=True)
    parser.add_argument("--memory-model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/comparison/v4_memory_ablation")
    parser.add_argument("--no-memory-summary", default="")
    parser.add_argument("--memory-summary", default="")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--strict-preflight", type=_parse_bool, default=True)
    parser.add_argument("--allow-partial-sites", type=_parse_bool, default=False)
    parser.add_argument("--fallback-mode", choices=["train", "eval", "strict"], default="eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    no_memory_summary_path = Path(args.no_memory_summary) if args.no_memory_summary else output_dir / "no_memory_evaluation_summary.json"
    memory_summary_path = Path(args.memory_summary) if args.memory_summary else output_dir / "memory_evaluation_summary.json"

    preflight_invalid = []
    if args.strict_preflight and not args.skip_evaluation:
        preflight_invalid = _preflight_configs([args.no_memory_config, args.memory_config])
        if preflight_invalid and not args.allow_partial_sites:
            comparison = _failed_fast_summary(
                no_memory_model_path=args.no_memory_model,
                memory_model_path=args.memory_model,
                seed=args.seed,
                episodes=args.episodes,
                max_steps=args.max_steps,
                fallback_mode=args.fallback_mode,
                strict_preflight=args.strict_preflight,
                allow_partial_sites=args.allow_partial_sites,
                invalid_sites=preflight_invalid,
                reason="preflight validation failed before evaluation",
            )
            summary_path = output_dir / "comparison_summary.json"
            table_path = output_dir / "comparison_table.csv"
            write_json(summary_path, comparison)
            write_comparison_table(table_path, comparison)
            print(f"comparison_summary_path: {summary_path}")
            print(f"comparison_table_path: {table_path}")
            return 2

    if not args.skip_evaluation and (not args.no_memory_summary or not args.memory_summary):
        _run_evaluation(
            config=args.no_memory_config,
            model=args.no_memory_model,
            output=no_memory_summary_path,
            use_memory=False,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            fallback_mode=args.fallback_mode,
        )
        _run_evaluation(
            config=args.memory_config,
            model=args.memory_model,
            output=memory_summary_path,
            use_memory=True,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            fallback_mode=args.fallback_mode,
        )

    no_memory = _read_json(no_memory_summary_path)
    memory = _read_json(memory_summary_path)
    comparison = build_comparison_summary(
        no_memory,
        memory,
        no_memory_model_path=args.no_memory_model,
        memory_model_path=args.memory_model,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        fallback_mode=args.fallback_mode,
        strict_preflight=args.strict_preflight,
        allow_partial_sites=args.allow_partial_sites,
        preflight_invalid_sites=preflight_invalid,
    )
    summary_path = output_dir / "comparison_summary.json"
    table_path = output_dir / "comparison_table.csv"
    write_json(summary_path, comparison)
    write_comparison_table(table_path, comparison)
    print(f"comparison_summary_path: {summary_path}")
    print(f"comparison_table_path: {table_path}")
    return 0 if comparison.get("comparison_valid") or args.allow_partial_sites else 2


def build_comparison_summary(
    no_memory: Mapping[str, Any],
    memory: Mapping[str, Any],
    *,
    no_memory_model_path: str,
    memory_model_path: str,
    seed: int,
    episodes: int,
    max_steps: int,
    fallback_mode: str = "eval",
    strict_preflight: bool = True,
    allow_partial_sites: bool = False,
    preflight_invalid_sites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preflight_invalid_sites = preflight_invalid_sites or []
    validation = validate_comparison_inputs(
        no_memory,
        memory,
        strict_preflight=strict_preflight,
        allow_partial_sites=allow_partial_sites,
        preflight_invalid_sites=preflight_invalid_sites,
    )
    compared_site_ids = list(validation["compared_site_ids"])
    no_repetition = _site_float_metric(no_memory, compared_site_ids, "repeated_action_rate")
    mem_repetition = _site_float_metric(memory, compared_site_ids, "repeated_action_rate")
    no_diversity = _site_float_metric(no_memory, compared_site_ids, "action_diversity_score")
    mem_diversity = _site_float_metric(memory, compared_site_ids, "action_diversity_score")
    no_state = _site_int_metric(no_memory, compared_site_ids, "state_coverage_count")
    mem_state = _site_int_metric(memory, compared_site_ids, "state_coverage_count")
    return {
        "no_memory_model_path": no_memory_model_path,
        "memory_model_path": memory_model_path,
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "fallback_mode": str(fallback_mode or "eval"),
        "disable_fallback": str(fallback_mode or "eval") == "strict",
        "compared_site_ids": compared_site_ids,
        "no_memory_avg_reward": _site_float_metric(no_memory, compared_site_ids, "avg_reward", "average_reward"),
        "memory_avg_reward": _site_float_metric(memory, compared_site_ids, "avg_reward", "average_reward"),
        "no_memory_repeated_action_rate": no_repetition,
        "memory_repeated_action_rate": mem_repetition,
        "repeated_action_rate_delta": round(mem_repetition - no_repetition, 6),
        "no_memory_action_diversity_score": no_diversity,
        "memory_action_diversity_score": mem_diversity,
        "action_diversity_delta": round(mem_diversity - no_diversity, 6),
        "no_memory_unique_clicked_target_count": _site_int_metric(no_memory, compared_site_ids, "unique_clicked_target_count"),
        "memory_unique_clicked_target_count": _site_int_metric(memory, compared_site_ids, "unique_clicked_target_count"),
        "unique_clicked_target_delta": _site_int_metric(memory, compared_site_ids, "unique_clicked_target_count")
        - _site_int_metric(no_memory, compared_site_ids, "unique_clicked_target_count"),
        "no_memory_unique_clicked_element_key_count": _site_int_metric(no_memory, compared_site_ids, "unique_clicked_element_key_count"),
        "memory_unique_clicked_element_key_count": _site_int_metric(memory, compared_site_ids, "unique_clicked_element_key_count"),
        "unique_clicked_element_key_delta": _site_int_metric(memory, compared_site_ids, "unique_clicked_element_key_count")
        - _site_int_metric(no_memory, compared_site_ids, "unique_clicked_element_key_count"),
        "no_memory_repeated_action_signature_count": _site_int_metric(no_memory, compared_site_ids, "repeated_action_signature_count"),
        "memory_repeated_action_signature_count": _site_int_metric(memory, compared_site_ids, "repeated_action_signature_count"),
        "repeated_action_signature_delta": _site_int_metric(memory, compared_site_ids, "repeated_action_signature_count")
        - _site_int_metric(no_memory, compared_site_ids, "repeated_action_signature_count"),
        "no_memory_repeated_action_type_count": _site_int_metric(no_memory, compared_site_ids, "repeated_action_type_count"),
        "memory_repeated_action_type_count": _site_int_metric(memory, compared_site_ids, "repeated_action_type_count"),
        "repeated_action_type_delta": _site_int_metric(memory, compared_site_ids, "repeated_action_type_count")
        - _site_int_metric(no_memory, compared_site_ids, "repeated_action_type_count"),
        "no_memory_repeated_element_key_click_count": _site_int_metric(no_memory, compared_site_ids, "repeated_element_key_click_count"),
        "memory_repeated_element_key_click_count": _site_int_metric(memory, compared_site_ids, "repeated_element_key_click_count"),
        "repeated_element_key_click_delta": _site_int_metric(memory, compared_site_ids, "repeated_element_key_click_count")
        - _site_int_metric(no_memory, compared_site_ids, "repeated_element_key_click_count"),
        "no_memory_action_fallback_exploration_redirect_count": _site_int_metric(no_memory, compared_site_ids, "action_fallback_exploration_redirect_count"),
        "memory_action_fallback_exploration_redirect_count": _site_int_metric(memory, compared_site_ids, "action_fallback_exploration_redirect_count"),
        "no_memory_verification_action_after_high_value_click_count": _site_int_metric(no_memory, compared_site_ids, "verification_action_after_high_value_click_count"),
        "memory_verification_action_after_high_value_click_count": _site_int_metric(memory, compared_site_ids, "verification_action_after_high_value_click_count"),
        "no_memory_low_value_generic_no_response_suppressed_count": _site_int_metric(no_memory, compared_site_ids, "low_value_generic_no_response_suppressed_count"),
        "memory_low_value_generic_no_response_suppressed_count": _site_int_metric(memory, compared_site_ids, "low_value_generic_no_response_suppressed_count"),
        "no_memory_paper_unique_clicked_element_key_count": _site_int_metric(no_memory, compared_site_ids, "unique_clicked_element_key_count"),
        "memory_paper_unique_clicked_element_key_count": _site_int_metric(memory, compared_site_ids, "unique_clicked_element_key_count"),
        "no_memory_paper_repeated_action_signature_count": _site_int_metric(no_memory, compared_site_ids, "repeated_action_signature_count"),
        "memory_paper_repeated_action_signature_count": _site_int_metric(memory, compared_site_ids, "repeated_action_signature_count"),
        "no_memory_paper_action_fallback_count": _site_int_metric(no_memory, compared_site_ids, "action_fallback_count"),
        "memory_paper_action_fallback_count": _site_int_metric(memory, compared_site_ids, "action_fallback_count"),
        "no_memory_paper_action_diversity_score": _site_float_metric(no_memory, compared_site_ids, "action_diversity_score"),
        "memory_paper_action_diversity_score": _site_float_metric(memory, compared_site_ids, "action_diversity_score"),
        "no_memory_paper_verification_after_high_value_count": _site_int_metric(
            no_memory, compared_site_ids, "verification_action_after_high_value_click_count"
        ),
        "memory_paper_verification_after_high_value_count": _site_int_metric(
            memory, compared_site_ids, "verification_action_after_high_value_click_count"
        ),
        "no_memory_paper_unique_detected_candidates": _site_int_metric(no_memory, compared_site_ids, "unique_detected_anomaly_count"),
        "memory_paper_unique_detected_candidates": _site_int_metric(memory, compared_site_ids, "unique_detected_anomaly_count"),
        "no_memory_paper_known_bug_reward_total": _site_float_metric(no_memory, compared_site_ids, "known_bug_reward_total"),
        "memory_paper_known_bug_reward_total": _site_float_metric(memory, compared_site_ids, "known_bug_reward_total"),
        "no_memory_paper_catalog_guided_action_count": _site_int_metric(no_memory, compared_site_ids, "catalog_guided_action_count"),
        "memory_paper_catalog_guided_action_count": _site_int_metric(memory, compared_site_ids, "catalog_guided_action_count"),
        "no_memory_fallback_applied_count": _site_int_metric(no_memory, compared_site_ids, "fallback_applied_count"),
        "memory_fallback_applied_count": _site_int_metric(memory, compared_site_ids, "fallback_applied_count"),
        "no_memory_fallback_applied_rate": _site_float_metric(no_memory, compared_site_ids, "fallback_applied_rate"),
        "memory_fallback_applied_rate": _site_float_metric(memory, compared_site_ids, "fallback_applied_rate"),
        "fallback_applied_rate_delta": round(
            _site_float_metric(memory, compared_site_ids, "fallback_applied_rate")
            - _site_float_metric(no_memory, compared_site_ids, "fallback_applied_rate"),
            6,
        ),
        "no_memory_policy_executed_action_mismatch_count": _site_int_metric(no_memory, compared_site_ids, "policy_executed_action_mismatch_count"),
        "memory_policy_executed_action_mismatch_count": _site_int_metric(memory, compared_site_ids, "policy_executed_action_mismatch_count"),
        "policy_executed_action_mismatch_delta": _site_int_metric(memory, compared_site_ids, "policy_executed_action_mismatch_count")
        - _site_int_metric(no_memory, compared_site_ids, "policy_executed_action_mismatch_count"),
        "no_memory_fallback_penalty_total": _site_float_metric(no_memory, compared_site_ids, "fallback_penalty_total"),
        "memory_fallback_penalty_total": _site_float_metric(memory, compared_site_ids, "fallback_penalty_total"),
        "no_memory_fallback_reward_capped_count": _site_int_metric(no_memory, compared_site_ids, "fallback_reward_capped_count"),
        "memory_fallback_reward_capped_count": _site_int_metric(memory, compared_site_ids, "fallback_reward_capped_count"),
        "no_memory_functional_action_count": _site_int_metric(no_memory, compared_site_ids, "functional_action_count"),
        "memory_functional_action_count": _site_int_metric(memory, compared_site_ids, "functional_action_count"),
        "functional_action_delta": _site_int_metric(memory, compared_site_ids, "functional_action_count")
        - _site_int_metric(no_memory, compared_site_ids, "functional_action_count"),
        "no_memory_state_coverage_count": no_state,
        "memory_state_coverage_count": mem_state,
        "state_coverage_delta": mem_state - no_state,
        "no_memory_unique_detected_anomaly_count": _site_int_metric(no_memory, compared_site_ids, "unique_detected_anomaly_count"),
        "memory_unique_detected_anomaly_count": _site_int_metric(memory, compared_site_ids, "unique_detected_anomaly_count"),
        "unique_detected_anomaly_delta": _site_int_metric(memory, compared_site_ids, "unique_detected_anomaly_count")
        - _site_int_metric(no_memory, compared_site_ids, "unique_detected_anomaly_count"),
        "memory_better_on_repetition": mem_repetition < no_repetition if compared_site_ids else False,
        "memory_better_on_diversity": mem_diversity > no_diversity if compared_site_ids else False,
        "memory_better_on_state_coverage": mem_state > no_state if compared_site_ids else False,
        "policy_uses_bug_labels": False,
        "known_bug_catalog_used": False,
        "comparison_focus": "exploration_diversity_and_repetition",
        **validation,
    }


def write_comparison_table(path: str | Path, comparison: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in SUMMARY_FIELDS:
            writer.writerow({"metric": key, "value": comparison.get(key)})


def validate_comparison_inputs(
    no_memory: Mapping[str, Any],
    memory: Mapping[str, Any],
    *,
    strict_preflight: bool = True,
    allow_partial_sites: bool = False,
    preflight_invalid_sites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preflight_invalid_sites = preflight_invalid_sites or []
    no_invalid = _invalid_sites(no_memory, run_label="no_memory")
    mem_invalid = _invalid_sites(memory, run_label="memory")
    invalid_by_site: dict[str, list[str]] = {}
    for item in [*preflight_invalid_sites, *no_invalid, *mem_invalid]:
        site_id = str(item.get("site_id") or "")
        if not site_id:
            continue
        invalid_by_site.setdefault(site_id, [])
        reason = str(item.get("reason") or item.get("error_message") or "invalid site")
        run_label = str(item.get("run") or "")
        prefix = f"{run_label}: " if run_label else ""
        if prefix + reason not in invalid_by_site[site_id]:
            invalid_by_site[site_id].append(prefix + reason)

    no_site_ids = set(_site_ids(no_memory))
    mem_site_ids = set(_site_ids(memory))
    enabled_intersection = sorted(no_site_ids & mem_site_ids)
    invalid_site_ids = sorted(invalid_by_site)
    compared_site_ids = [site_id for site_id in enabled_intersection if site_id not in invalid_by_site]
    if strict_preflight and invalid_site_ids and not allow_partial_sites:
        compared_site_ids = []
    failed_site_ids = sorted(
        {
            str(item.get("site_id") or "")
            for item in [*no_invalid, *mem_invalid]
            if str(item.get("kind") or "") == "failed"
        }
    )
    timeout_excluded_site_ids = sorted(
        {
            str(item.get("site_id") or "")
            for item in [*no_invalid, *mem_invalid]
            if str(item.get("kind") or "") == "timeout"
        }
    )
    partial_site_ids = sorted(
        {
            str(item.get("site_id") or "")
            for item in [*no_invalid, *mem_invalid]
            if str(item.get("kind") or "") == "partial"
        }
    )
    connection_refused_site_ids = sorted(
        {
            str(item.get("site_id") or "")
            for item in [*preflight_invalid_sites, *no_invalid, *mem_invalid]
            if "ERR_CONNECTION_REFUSED" in str(item.get("reason") or item.get("error_message") or "")
        }
    )
    zero_interaction_site_ids = sorted(
        {
            str(item.get("site_id") or "")
            for item in [*no_invalid, *mem_invalid]
            if str(item.get("kind") or "") == "zero_interaction"
        }
    )
    comparison_failed_fast = bool(strict_preflight and invalid_site_ids and not allow_partial_sites)
    comparison_valid = bool(compared_site_ids) and not comparison_failed_fast and (
        allow_partial_sites or not invalid_site_ids
    )
    invalid_reason = ""
    if comparison_failed_fast:
        invalid_reason = "invalid enabled sites found; rerun after all enabled sites are reachable and interactive"
    elif not compared_site_ids:
        invalid_reason = "no valid overlapping sites to compare"
    return {
        "comparison_valid": comparison_valid,
        "comparison_invalid_reason": invalid_reason,
        "strict_preflight": bool(strict_preflight),
        "allow_partial_sites": bool(allow_partial_sites),
        "invalid_site_ids": invalid_site_ids,
        "invalid_site_reasons": {site_id: reasons for site_id, reasons in sorted(invalid_by_site.items())},
        "excluded_site_ids": invalid_site_ids,
        "compared_site_ids": compared_site_ids,
        "failed_site_count": len(failed_site_ids),
        "failed_site_ids": failed_site_ids,
        "timeout_excluded_from_comparison_count": len(timeout_excluded_site_ids),
        "partial_site_excluded_count": len(partial_site_ids),
        "zero_interaction_site_ids": zero_interaction_site_ids,
        "connection_refused_site_ids": connection_refused_site_ids,
        "all_enabled_sites_interacted": not invalid_site_ids,
        "comparison_failed_fast": comparison_failed_fast,
    }


def _invalid_sites(summary: Mapping[str, Any], *, run_label: str) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    sites = _site_results(summary)
    for site_id in _site_ids(summary):
        site = sites.get(site_id, {})
        reasons: list[tuple[str, str]] = []
        status = str(site.get("status") or "")
        error_message = str(site.get("error_message") or "")
        excluded_reason = str(site.get("excluded_from_comparison_reason") or "")
        if site.get("valid_for_comparison") is False:
            kind = "timeout" if excluded_reason == "timeout" or "timeout" in error_message.lower() else "partial" if bool(site.get("partial")) else "failed"
            reasons.append((kind, f"valid_for_comparison == false: {excluded_reason or 'invalid'}"))
        if status == "failed":
            reasons.append(("failed", "status == failed"))
        if bool(site.get("partial")):
            reasons.append(("partial", "partial == true"))
        if int(site.get("timeout_count") or 0) > 0 or "timeout" in error_message.lower() or excluded_reason == "timeout":
            reasons.append(("timeout", "timeout site excluded from comparison"))
        if "ERR_CONNECTION_REFUSED" in error_message:
            reasons.append(("connection_refused", "error_message contains ERR_CONNECTION_REFUSED"))
        if int(site.get("completed_episodes") or 0) == 0:
            reasons.append(("failed", "completed_episodes == 0"))
        if site.get("valid_completed_episodes") is not None and int(site.get("valid_completed_episodes") or 0) == 0:
            reasons.append(("failed", "valid_completed_episodes == 0"))
        if not str(site.get("observed_url_sample") or ""):
            reasons.append(("zero_interaction", "observed_url_sample is empty"))
        if _candidate_count(site) <= 0:
            reasons.append(("zero_interaction", "candidate_count == 0"))
        if int(site.get("openended_interactive_candidate_count") or 0) == 0:
            reasons.append(("zero_interaction", "openended_interactive_candidate_count == 0"))
        if int(site.get("policy_action_activity_count") or 0) == 0:
            reasons.append(("zero_interaction", "policy_action_activity_count == 0"))
        action_counts = site.get("action_counts")
        if not isinstance(action_counts, Mapping) or not action_counts:
            reasons.append(("zero_interaction", "action_counts == {}"))
        for kind, reason in reasons:
            invalid.append({"site_id": site_id, "run": run_label, "kind": kind, "reason": reason, "error_message": error_message})
    return invalid


def _candidate_count(site: Mapping[str, Any]) -> int:
    for key in (
        "candidate_count",
        "openended_interactive_candidate_count",
        "functional_priority_candidate_count",
        "category_candidate_count",
        "filter_candidate_count",
        "tab_candidate_count",
        "high_value_functional_candidate_count",
    ):
        try:
            value = int(site.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _site_results(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sites = summary.get("sites")
    if isinstance(sites, Mapping):
        return {str(key): value for key, value in sites.items() if isinstance(value, Mapping)}
    per_site = summary.get("per_site_metrics")
    if isinstance(per_site, list):
        return {
            str(item.get("site_id")): item
            for item in per_site
            if isinstance(item, Mapping) and item.get("site_id")
        }
    site_ids = _site_ids(summary)
    if len(site_ids) == 1:
        return {site_ids[0]: summary}
    return {}


def _preflight_configs(config_paths: list[str]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for config_path in config_paths:
        config = _read_json(config_path)
        sites = config.get("sites", [])
        if not isinstance(sites, list):
            continue
        for site in sites:
            if not isinstance(site, Mapping) or site.get("enabled") is False:
                continue
            site_id = str(site.get("site_id") or "")
            base_url = str(site.get("base_url") or "")
            key = (site_id, base_url)
            if key in seen:
                continue
            seen.add(key)
            if not base_url:
                invalid.append({"site_id": site_id, "run": "preflight", "kind": "connection", "reason": "base_url is empty"})
                continue
            try:
                with urllib.request.urlopen(base_url, timeout=5) as response:
                    if int(getattr(response, "status", 200) or 200) >= 500:
                        invalid.append(
                            {
                                "site_id": site_id,
                                "run": "preflight",
                                "kind": "connection",
                                "reason": f"preflight HTTP status {getattr(response, 'status', '')}",
                            }
                        )
            except urllib.error.URLError as exc:
                invalid.append(
                    {
                        "site_id": site_id,
                        "run": "preflight",
                        "kind": "connection_refused" if "Connection refused" in str(exc) else "connection",
                        "reason": f"preflight connection failed: {exc}",
                        "error_message": str(exc),
                    }
                )
    return invalid


def _failed_fast_summary(
    *,
    no_memory_model_path: str,
    memory_model_path: str,
    seed: int,
    episodes: int,
    max_steps: int,
    fallback_mode: str,
    strict_preflight: bool,
    allow_partial_sites: bool,
    invalid_sites: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    invalid_by_site: dict[str, list[str]] = {}
    for item in invalid_sites:
        site_id = str(item.get("site_id") or "")
        if site_id:
            invalid_by_site.setdefault(site_id, []).append(str(item.get("reason") or reason))
    invalid_site_ids = sorted(invalid_by_site)
    return {
        "no_memory_model_path": no_memory_model_path,
        "memory_model_path": memory_model_path,
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "fallback_mode": str(fallback_mode or "eval"),
        "disable_fallback": str(fallback_mode or "eval") == "strict",
        "comparison_valid": False,
        "comparison_invalid_reason": reason,
        "strict_preflight": bool(strict_preflight),
        "allow_partial_sites": bool(allow_partial_sites),
        "invalid_site_ids": invalid_site_ids,
        "invalid_site_reasons": invalid_by_site,
        "excluded_site_ids": invalid_site_ids,
        "compared_site_ids": [],
        "failed_site_count": 0,
        "failed_site_ids": [],
        "timeout_excluded_from_comparison_count": len(invalid_site_ids),
        "partial_site_excluded_count": 0,
        "zero_interaction_site_ids": [],
        "connection_refused_site_ids": invalid_site_ids,
        "all_enabled_sites_interacted": False,
        "comparison_failed_fast": True,
        "policy_uses_bug_labels": False,
        "known_bug_catalog_used": False,
        "comparison_focus": "exploration_diversity_and_repetition",
    }


def _run_evaluation(
    *,
    config: str,
    model: str,
    output: Path,
    use_memory: bool,
    episodes: int,
    max_steps: int,
    seed: int,
    fallback_mode: str = "eval",
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "runners" / "evaluate_multisite_browsergym_agent.py"),
        "--config",
        config,
        "--model-path",
        model,
        "--output",
        str(output),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--use-memory-encoder",
        "true" if use_memory else "false",
        "--fallback-mode",
        str(fallback_mode or "eval"),
    ]
    if str(fallback_mode or "eval") == "strict":
        cmd.extend(["--disable-fallback", "true"])
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _float_metric(summary: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if summary.get(key) is not None:
            try:
                return float(summary.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _int_metric(summary: Mapping[str, Any], key: str) -> int:
    try:
        return int(summary.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _site_float_metric(summary: Mapping[str, Any], site_ids: list[str], *keys: str) -> float:
    sites = _site_results(summary)
    values: list[float] = []
    for site_id in site_ids:
        site = sites.get(site_id, {})
        for key in keys:
            if site.get(key) is not None:
                try:
                    values.append(float(site.get(key) or 0.0))
                except (TypeError, ValueError):
                    values.append(0.0)
                break
    if values:
        return round(sum(values) / len(values), 6)
    return _float_metric(summary, *keys) if not site_ids else 0.0


def _site_int_metric(summary: Mapping[str, Any], site_ids: list[str], key: str) -> int:
    sites = _site_results(summary)
    if site_ids:
        total = 0
        for site_id in site_ids:
            try:
                total += int(sites.get(site_id, {}).get(key) or 0)
            except (TypeError, ValueError):
                pass
        return total
    return _int_metric(summary, key)


def _compared_site_ids(no_memory: Mapping[str, Any], memory: Mapping[str, Any]) -> list[str]:
    no_sites = set(_site_ids(no_memory))
    memory_sites = set(_site_ids(memory))
    return sorted(no_sites & memory_sites)


def _site_ids(summary: Mapping[str, Any]) -> list[str]:
    if isinstance(summary.get("enabled_site_ids"), list):
        return [str(item) for item in summary["enabled_site_ids"]]
    if isinstance(summary.get("sites"), Mapping):
        return [str(key) for key in summary["sites"].keys()]
    if isinstance(summary.get("per_site_metrics"), list):
        return [
            str(item.get("site_id"))
            for item in summary["per_site_metrics"]
            if isinstance(item, Mapping) and item.get("site_id")
        ]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
