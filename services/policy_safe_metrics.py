"""Metric helpers for v3 policy-safe BrowserGym PPO reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Mapping, Sequence

import json


VERSION = "v3_policy_safe"
EVALUATION_DIR = Path("artifacts/evaluation/v3_policy_safe")
TRAINING_LOG_DIR = Path("artifacts/training_logs/v3_policy_safe")
COMPARISON_NOTE_PATH = EVALUATION_DIR / "comparison_note.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_precision_recall_f1(
    *,
    true_positive_count: int,
    false_positive_count: int,
    false_negative_count: int,
) -> dict[str, float]:
    tp = max(0, int(true_positive_count or 0))
    fp = max(0, int(false_positive_count or 0))
    fn = max(0, int(false_negative_count or 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def build_metric_counts(
    *,
    known_bug_total: int,
    known_bug_matched_count: int,
    unique_detected_anomaly_count: int,
) -> dict[str, int]:
    tp = max(0, int(known_bug_matched_count or 0))
    unique_detected = max(0, int(unique_detected_anomaly_count or 0))
    known_total = max(0, int(known_bug_total or 0))
    fp = max(0, unique_detected - tp)
    fn = max(0, known_total - tp)
    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "known_bug_total": known_total,
        "known_bug_matched_count": tp,
        "unique_detected_anomaly_count": unique_detected,
    }


def build_per_site_metric(site_id: str, site_result: Mapping[str, Any]) -> dict[str, Any]:
    known_total = _known_bug_total(site_result)
    matched_count = int(site_result.get("known_bug_match_count") or len(site_result.get("matched_bug_ids", []) or []))
    unique_detected = int(
        site_result.get("unique_detected_anomaly_count")
        or site_result.get("unique_detected_candidates")
        or site_result.get("deduped_detected_candidates")
        or site_result.get("detected_bug_count")
        or 0
    )
    raw_detected = int(site_result.get("total_detected_candidates") or site_result.get("raw_detected_candidates") or unique_detected)
    functional_action_count = int(site_result.get("functional_action_count") or 0)
    completed_episodes = _completed_episodes(site_result)
    no_functional_episode_count = int(site_result.get("no_functional_action_episode_count") or 0)
    if functional_action_count <= 0 and completed_episodes > 0:
        no_functional_episode_count = max(no_functional_episode_count, completed_episodes)
    functional_action_signal_reward_total = float(site_result.get("functional_action_signal_reward_total") or 0.0)
    if functional_action_count <= 0:
        functional_action_signal_reward_total = 0.0
    semantic_action_type_counts = dict(site_result.get("semantic_action_type_counts") or {})
    clicked_semantic_action_types = dict(site_result.get("clicked_semantic_action_types") or {})
    attempted_semantic_action_types = dict(site_result.get("attempted_semantic_action_types") or clicked_semantic_action_types)
    memory_enabled = bool(site_result.get("memory_encoder_enabled", False))
    requested_episodes = int(site_result.get("requested_episodes") or site_result.get("episodes") or site_result.get("episode_count") or 0)
    valid_completed_episodes = int(site_result.get("valid_completed_episodes") or 0)
    if site_result.get("valid_completed_episodes") is None and not bool(site_result.get("partial", False)):
        valid_completed_episodes = completed_episodes
    if site_result.get("valid_for_comparison") is None:
        valid_for_comparison = (
            str(site_result.get("status") or "ok") != "failed"
            and not bool(site_result.get("partial", False))
            and completed_episodes > 0
        )
    else:
        valid_for_comparison = bool(site_result.get("valid_for_comparison", False))
    repeated_action_rate = float(site_result.get("repeated_action_rate") or _rate_from_counts(site_result.get("action_counts", {})))
    repeated_click_target_rate = float(
        site_result.get("repeated_click_target_rate")
        or _safe_rate(site_result.get("repeated_click_target_count", 0), site_result.get("click_element_count", 0))
    )
    counts = build_metric_counts(
        known_bug_total=known_total,
        known_bug_matched_count=matched_count,
        unique_detected_anomaly_count=unique_detected,
    )
    metrics = compute_precision_recall_f1(
        true_positive_count=counts["true_positive_count"],
        false_positive_count=counts["false_positive_count"],
        false_negative_count=counts["false_negative_count"],
    )
    return {
        "site_id": site_id,
        "status": str(site_result.get("status") or "ok"),
        "error_type": str(site_result.get("error_type") or ""),
        "error_message": str(site_result.get("error_message") or ""),
        "requested_episodes": requested_episodes,
        "completed_episodes": completed_episodes,
        "valid_completed_episodes": valid_completed_episodes,
        "valid_for_comparison": valid_for_comparison,
        "excluded_from_comparison_reason": str(site_result.get("excluded_from_comparison_reason") or ""),
        "partial": bool(site_result.get("partial", False)),
        "base_url": site_result.get("requested_base_url") or site_result.get("base_url") or "",
        "episode_count": completed_episodes,
        "avg_reward": float(site_result.get("average_reward") or site_result.get("avg_reward") or 0.0),
        "detected_anomaly_count": raw_detected,
        "raw_detected_anomaly_count": raw_detected,
        "unique_detected_anomaly_count": unique_detected,
        "matched_bug_ids": list(site_result.get("matched_bug_ids", []) or []),
        "missed_bug_ids": list(site_result.get("missed_bug_ids", []) or []),
        **counts,
        **metrics,
        "invalid_action_count": int(site_result.get("invalid_action_count") or 0),
        "no_effect_action_count": int(site_result.get("no_effect_action_count") or 0),
        "avg_step_elapsed_ms": int(site_result.get("avg_step_elapsed_ms") or 0),
        "max_step_elapsed_ms": int(site_result.get("max_step_elapsed_ms") or 0),
        "reset_elapsed_ms": int(site_result.get("reset_elapsed_ms") or 0),
        "reset_timeout_ms": int(site_result.get("reset_timeout_ms") or 0),
        "reset_retry_count": int(site_result.get("reset_retry_count") or 0),
        "reset_retry_used": int(site_result.get("reset_retry_used") or 0),
        "reset_failure_count": int(site_result.get("reset_failure_count") or 0),
        "reset_failure_reason": str(site_result.get("reset_failure_reason") or ""),
        "site_elapsed_ms": int(site_result.get("site_elapsed_ms") or 0),
        "episode_elapsed_ms": int(site_result.get("episode_elapsed_ms") or 0),
        "early_stop_count": int(site_result.get("early_stop_count") or 0),
        "early_stop_reasons": dict(site_result.get("early_stop_reasons") or {}),
        "timeout_count": int(site_result.get("timeout_count") or 0),
        "use_memory_encoder": bool(site_result.get("use_memory_encoder", site_result.get("memory_encoder_enabled", False))),
        "memory_encoder_enabled": memory_enabled,
        "memory_encoder_type": str(site_result.get("memory_encoder_type") or "") if memory_enabled else "",
        "memory_hidden_size": int(site_result.get("memory_hidden_size") or 0) if memory_enabled else 0,
        "memory_state_reset_count": int(
            site_result.get("memory_state_reset_count", site_result.get("episode_memory_reset_count", 0)) or 0
        ),
        "episode_memory_reset_count": int(site_result.get("episode_memory_reset_count") or 0),
        "memory_state_update_count": int(site_result.get("memory_state_update_count") or 0) if memory_enabled else 0,
        "memory_state_norm_mean": float(site_result.get("memory_state_norm_mean") or 0.0) if memory_enabled else 0.0,
        "repeated_action_rate": repeated_action_rate,
        "repeated_click_target_rate": repeated_click_target_rate,
        "repeated_action_rate_before_memory": site_result.get("repeated_action_rate_before_memory"),
        "repeated_action_rate_after_memory": site_result.get("repeated_action_rate_after_memory") if memory_enabled else None,
        "diversity_reward_total": float(site_result.get("diversity_reward_total") or 0.0),
        "repeated_action_penalty_total": float(site_result.get("repeated_action_penalty_total") or 0.0),
        "first_click_reward_count": int(site_result.get("first_click_reward_count") or 0),
        "new_action_type_reward_count": int(site_result.get("new_action_type_reward_count") or 0),
        "new_target_reward_count": int(site_result.get("new_target_reward_count") or 0),
        "inspect_reward_count": int(site_result.get("inspect_reward_count") or 0),
        "open_detail_panel_repeat_penalty_count": int(site_result.get("open_detail_panel_repeat_penalty_count") or 0),
        "no_effect_action_penalty_count": int(site_result.get("no_effect_action_penalty_count") or 0),
        "action_diversity_score": float(site_result.get("action_diversity_score") or 0.0),
        "unique_action_type_count": int(site_result.get("unique_action_type_count") or 0),
        "unique_clicked_target_count": int(site_result.get("unique_clicked_target_count") or 0),
        "unique_element_key_count": int(site_result.get("unique_element_key_count") or 0),
        "unique_clicked_element_key_count": int(site_result.get("unique_clicked_element_key_count") or 0),
        "repeated_action_type_count": int(site_result.get("repeated_action_type_count") or 0),
        "repeated_action_signature_count": int(site_result.get("repeated_action_signature_count") or 0),
        "repeated_action_count": int(
            site_result.get("repeated_action_count")
            if site_result.get("repeated_action_count") is not None
            else site_result.get("repeated_action_signature_count")
            or 0
        ),
        "repeated_element_key_click_count": int(site_result.get("repeated_element_key_click_count") or 0),
        "element_key_collision_count": int(site_result.get("element_key_collision_count") or 0),
        "element_key_missing_count": int(site_result.get("element_key_missing_count") or 0),
        "first_click_step": site_result.get("first_click_step"),
        "open_detail_panel_count": int(site_result.get("open_detail_panel_count") or 0),
        "click_retry_button_count": int(site_result.get("click_retry_button_count") or 0),
        "functional_action_count": functional_action_count,
        "functional_priority_candidate_count": int(site_result.get("functional_priority_candidate_count") or 0),
        "functional_priority_click_count": int(site_result.get("functional_priority_click_count") or 0),
        "filter_candidate_count": int(site_result.get("filter_candidate_count") or 0),
        "filter_click_count": int(site_result.get("filter_click_count") or 0),
        "category_candidate_count": int(site_result.get("category_candidate_count") or 0),
        "category_click_count": int(site_result.get("category_click_count") or 0),
        "tab_candidate_count": int(site_result.get("tab_candidate_count") or 0),
        "tab_click_count": int(site_result.get("tab_click_count") or 0),
        "enroll_candidate_count": int(site_result.get("enroll_candidate_count") or 0),
        "enroll_click_count": int(site_result.get("enroll_click_count") or 0),
        "search_input_click_count": int(site_result.get("search_input_click_count") or 0),
        "search_input_fill_count": int(site_result.get("search_input_fill_count") or 0),
        "low_value_generic_candidate_count": int(site_result.get("low_value_generic_candidate_count") or 0),
        "low_value_generic_click_count": int(site_result.get("low_value_generic_click_count") or 0),
        "low_value_generic_no_response_suppressed_count": int(site_result.get("low_value_generic_no_response_suppressed_count") or 0),
        "policy_selected_low_value_generic_count": int(site_result.get("policy_selected_low_value_generic_count") or 0),
        "filtered_no_response_suppressed_count": int(site_result.get("filtered_no_response_suppressed_count") or 0),
        "no_response_suppressed_by_text_delta_count": int(site_result.get("no_response_suppressed_by_text_delta_count") or 0),
        "no_response_suppressed_by_candidate_delta_count": int(site_result.get("no_response_suppressed_by_candidate_delta_count") or 0),
        "no_response_suppressed_by_filter_semantics_count": int(site_result.get("no_response_suppressed_by_filter_semantics_count") or 0),
        "search_click_no_effect_suppressed_count": int(site_result.get("search_click_no_effect_suppressed_count") or 0),
        "no_progress_delayed_by_unclicked_functional_candidate_count": int(
            site_result.get("no_progress_delayed_by_unclicked_functional_candidate_count") or 0
        ),
        "visited_target_suppression_count": int(site_result.get("visited_target_suppression_count") or 0),
        "visited_target_skipped_count": int(site_result.get("visited_target_skipped_count") or 0),
        "visited_element_key_suppression_count": int(site_result.get("visited_element_key_suppression_count") or 0),
        "unclicked_high_value_candidate_selected_count": int(site_result.get("unclicked_high_value_candidate_selected_count") or 0),
        "unclicked_functional_candidate_selected_count": int(site_result.get("unclicked_functional_candidate_selected_count") or 0),
        "unclicked_element_key_selected_count": int(site_result.get("unclicked_element_key_selected_count") or 0),
        "memory_guided_candidate_switch_count": int(site_result.get("memory_guided_candidate_switch_count") or 0),
        "repeated_action_redirect_count": int(site_result.get("repeated_action_redirect_count") or 0),
        "repeated_target_redirect_count": int(site_result.get("repeated_target_redirect_count") or 0),
        "finish_delayed_by_unclicked_candidate_count": int(site_result.get("finish_delayed_by_unclicked_candidate_count") or 0),
        "repeated_meta_action_suppression_count": int(site_result.get("repeated_meta_action_suppression_count") or 0),
        "inspect_console_repeat_suppression_count": int(site_result.get("inspect_console_repeat_suppression_count") or 0),
        "inspect_network_repeat_suppression_count": int(site_result.get("inspect_network_repeat_suppression_count") or 0),
        "meta_action_redirect_to_functional_count": int(site_result.get("meta_action_redirect_to_functional_count") or 0),
        "semantic_action_type_counts": semantic_action_type_counts,
        "clicked_semantic_action_types": clicked_semantic_action_types,
        "attempted_semantic_action_types": attempted_semantic_action_types,
        "semantic_no_effect_click_count": int(site_result.get("semantic_no_effect_click_count") or 0),
        "high_value_no_effect_click_count": int(site_result.get("high_value_no_effect_click_count") or 0),
        "semantic_button_no_response_count": int(site_result.get("semantic_button_no_response_count") or 0),
        "functional_no_effect_anomaly_count": int(site_result.get("functional_no_effect_anomaly_count") or 0),
        "workout_add_no_effect_count": int(site_result.get("workout_add_no_effect_count") or 0),
        "cart_no_effect_count": int(site_result.get("cart_no_effect_count") or 0),
        "add_no_effect_count": int(site_result.get("add_no_effect_count") or 0),
        "enroll_no_effect_count": int(site_result.get("enroll_no_effect_count") or 0),
        "high_value_enroll_no_response_count": int(site_result.get("high_value_enroll_no_response_count") or 0),
        "high_value_button_no_response_count": int(site_result.get("high_value_button_no_response_count") or 0),
        "missed_semantic_action_reason": dict(site_result.get("missed_semantic_action_reason") or {}),
        "no_effect_click_to_new_candidate_fallback_count": int(site_result.get("no_effect_click_to_new_candidate_fallback_count") or 0),
        "action_fallback_count": int(site_result.get("action_fallback_count") or 0),
        "action_fallback_invalid_action_count": int(site_result.get("action_fallback_invalid_action_count") or 0),
        "action_fallback_visited_element_key_count": int(site_result.get("action_fallback_visited_element_key_count") or 0),
        "action_fallback_low_value_candidate_count": int(site_result.get("action_fallback_low_value_candidate_count") or 0),
        "action_fallback_exploration_redirect_count": int(site_result.get("action_fallback_exploration_redirect_count") or 0),
        "policy_selected_valid_action_count": int(site_result.get("policy_selected_valid_action_count") or 0),
        "policy_selected_invalid_action_count": int(site_result.get("policy_selected_invalid_action_count") or 0),
        "policy_selected_visited_element_count": int(site_result.get("policy_selected_visited_element_count") or 0),
        "policy_selected_low_priority_action_count": int(site_result.get("policy_selected_low_priority_action_count") or 0),
        "fallback_applied_count": int(site_result.get("fallback_applied_count") or 0),
        "fallback_applied_rate": float(site_result.get("fallback_applied_rate") or 0.0),
        "fallback_penalty_total": float(site_result.get("fallback_penalty_total") or 0.0),
        "fallback_reward_capped_count": int(site_result.get("fallback_reward_capped_count") or 0),
        "policy_executed_action_match_count": int(site_result.get("policy_executed_action_match_count") or 0),
        "policy_executed_action_mismatch_count": int(site_result.get("policy_executed_action_mismatch_count") or 0),
        "unclicked_functional_priority_candidate_count": int(site_result.get("unclicked_functional_priority_candidate_count") or 0),
        "high_value_functional_candidate_count": int(site_result.get("high_value_functional_candidate_count") or 0),
        "first_functional_action_step": site_result.get("first_functional_action_step"),
        "unique_functional_action_type_count": int(site_result.get("unique_functional_action_type_count") or 0),
        "unique_functional_target_count": int(site_result.get("unique_functional_target_count") or 0),
        "debug_meta_action_count": int(site_result.get("debug_meta_action_count") or 0),
        "debug_meta_repeat_count": int(site_result.get("debug_meta_repeat_count") or 0),
        "no_functional_action_episode_count": no_functional_episode_count,
        "functional_action_signal_delta_count": int(site_result.get("functional_action_signal_delta_count") or 0),
        "functional_action_network_delta_count": int(site_result.get("functional_action_network_delta_count") or 0),
        "verification_action_after_high_value_click_count": int(site_result.get("verification_action_after_high_value_click_count") or 0),
        "verification_action_redirect_count": int(site_result.get("verification_action_redirect_count") or 0),
        "high_value_click_pending_verification_count": int(site_result.get("high_value_click_pending_verification_count") or 0),
        "high_value_click_verified_count": int(site_result.get("high_value_click_verified_count") or 0),
        "high_value_click_without_verification_count": int(site_result.get("high_value_click_without_verification_count") or 0),
        "meta_action_repeated_priority_lowered_count": int(site_result.get("meta_action_repeated_priority_lowered_count") or 0),
        "policy_selected_meta_action_suppressed_count": int(site_result.get("policy_selected_meta_action_suppressed_count") or 0),
        "reward_functional_action_total": float(site_result.get("reward_functional_action_total") or 0.0),
        "penalty_debug_meta_total": float(site_result.get("penalty_debug_meta_total") or 0.0),
        "fallback_reason_counts": dict(site_result.get("fallback_reason_counts") or {}),
        "known_bug_reward_total": 0.0,
        "signal_reward_total": float(site_result.get("signal_reward_total") or 0.0),
        "exploration_reward_total": float(site_result.get("exploration_reward_total") or 0.0),
        "ui_dom_signal_reward_total": float(site_result.get("ui_dom_signal_reward_total") or 0.0),
        "console_runtime_signal_reward_total": float(site_result.get("console_runtime_signal_reward_total") or 0.0),
        "network_api_signal_reward_total": float(site_result.get("network_api_signal_reward_total") or 0.0),
        "cross_layer_signal_reward_total": float(site_result.get("cross_layer_signal_reward_total") or 0.0),
        "security_signal_reward_total": float(site_result.get("security_signal_reward_total") or 0.0),
        "repeated_penalty_total": float(site_result.get("repeated_penalty_total") or 0.0),
        "multi_signal_anomaly_count": int(site_result.get("multi_signal_anomaly_count") or 0),
        "console_error_count": int(site_result.get("console_error_count") or 0),
        "runtime_exception_count": int(site_result.get("runtime_exception_count") or 0),
        "network_error_count": int(site_result.get("network_error_count") or 0),
        "api_4xx_count": int(site_result.get("api_4xx_count") or 0),
        "api_5xx_count": int(site_result.get("api_5xx_count") or 0),
        "api_timeout_count": int(site_result.get("api_timeout_count") or 0),
        "api_ui_mismatch_count": int(site_result.get("api_ui_mismatch_count") or 0),
        "security_signal_count": int(site_result.get("security_signal_count") or 0),
        "signal_delta_reward_total": float(site_result.get("signal_delta_reward_total") or 0.0),
        "inspect_action_reward_total": float(site_result.get("inspect_action_reward_total") or 0.0),
        "functional_action_signal_reward_total": functional_action_signal_reward_total,
        "target_signal_types": list(site_result.get("target_signal_types") or []),
        "observed_signal_types": list(site_result.get("observed_signal_types") or []),
        "missing_signal_types": list(site_result.get("missing_signal_types") or []),
        "signal_type_coverage": float(site_result.get("signal_type_coverage") or 0.0),
        "signal_discovery_score": float(site_result.get("signal_discovery_score") or 0.0),
        "ui_dom_signal_count": int(site_result.get("ui_dom_signal_count") or 0),
        "console_runtime_signal_count": int(site_result.get("console_runtime_signal_count") or 0),
        "network_api_signal_count": int(site_result.get("network_api_signal_count") or 0),
        "cross_layer_signal_count": int(site_result.get("cross_layer_signal_count") or 0),
        "state_coverage_count": int(site_result.get("state_coverage_count") or 0),
        "new_url_count": int(site_result.get("new_url_count") or 0),
        "new_dom_state_count": int(site_result.get("new_dom_state_count") or 0),
        "signal_collector_enabled": bool(site_result.get("signal_collector_enabled", True)),
        "playwright_console_listener_enabled": bool(site_result.get("playwright_console_listener_enabled", False)),
        "playwright_pageerror_listener_enabled": bool(site_result.get("playwright_pageerror_listener_enabled", False)),
        "playwright_network_listener_enabled": bool(site_result.get("playwright_network_listener_enabled", False)),
        "playwright_request_listener_enabled": bool(site_result.get("playwright_request_listener_enabled", False)),
        "playwright_response_listener_enabled": bool(site_result.get("playwright_response_listener_enabled", False)),
        "playwright_requestfailed_listener_enabled": bool(site_result.get("playwright_requestfailed_listener_enabled", False)),
        "playwright_listener_warning": str(site_result.get("playwright_listener_warning") or ""),
    }


def aggregate_site_metrics(per_site_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item.get("true_positive_count") or 0) for item in per_site_metrics)
    fp = sum(int(item.get("false_positive_count") or 0) for item in per_site_metrics)
    fn = sum(int(item.get("false_negative_count") or 0) for item in per_site_metrics)
    known_total = sum(int(item.get("known_bug_total") or 0) for item in per_site_metrics)
    known_matched = sum(int(item.get("known_bug_matched_count") or 0) for item in per_site_metrics)
    detected = sum(int(item.get("detected_anomaly_count") or 0) for item in per_site_metrics)
    raw_detected = sum(int(item.get("raw_detected_anomaly_count") or item.get("detected_anomaly_count") or 0) for item in per_site_metrics)
    unique_detected = sum(int(item.get("unique_detected_anomaly_count") or 0) for item in per_site_metrics)
    early_stop_reasons: Counter[str] = Counter()
    for item in per_site_metrics:
        reasons = item.get("early_stop_reasons", {})
        if isinstance(reasons, Mapping):
            early_stop_reasons.update({str(key): int(value or 0) for key, value in reasons.items()})
    fallback_reason_counts: Counter[str] = Counter()
    semantic_action_type_counts: Counter[str] = Counter()
    clicked_semantic_action_types: Counter[str] = Counter()
    attempted_semantic_action_types: Counter[str] = Counter()
    observed_signal_types: set[str] = set()
    target_signal_types: set[str] = set()
    for item in per_site_metrics:
        reasons = item.get("fallback_reason_counts", {})
        if isinstance(reasons, Mapping):
            fallback_reason_counts.update({str(key): int(value or 0) for key, value in reasons.items()})
        semantic_counts = item.get("semantic_action_type_counts", {})
        if isinstance(semantic_counts, Mapping):
            semantic_action_type_counts.update({str(key): int(value or 0) for key, value in semantic_counts.items()})
        clicked_semantics = item.get("clicked_semantic_action_types", {})
        if isinstance(clicked_semantics, Mapping):
            clicked_semantic_action_types.update({str(key): int(value or 0) for key, value in clicked_semantics.items()})
        attempted_semantics = item.get("attempted_semantic_action_types", {})
        if isinstance(attempted_semantics, Mapping):
            attempted_semantic_action_types.update({str(key): int(value or 0) for key, value in attempted_semantics.items()})
        observed_signal_types.update(str(value) for value in item.get("observed_signal_types", []) or [] if value)
        target_signal_types.update(str(value) for value in item.get("target_signal_types", []) or [] if value)
    signal_metrics = compute_signal_discovery_metrics(per_site_metrics, target_signal_types=sorted(target_signal_types))
    metrics = compute_precision_recall_f1(
        true_positive_count=tp,
        false_positive_count=fp,
        false_negative_count=fn,
    )
    memory_enabled = any(bool(item.get("memory_encoder_enabled", False)) for item in per_site_metrics)
    return {
        **metrics,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "known_bug_total": known_total,
        "known_bug_matched_count": known_matched,
        "detected_anomaly_count": detected,
        "requested_episodes": sum(int(item.get("requested_episodes") or item.get("episode_count") or 0) for item in per_site_metrics),
        "valid_completed_episodes": sum(int(item.get("valid_completed_episodes") or 0) for item in per_site_metrics),
        "timeout_excluded_from_comparison_count": sum(
            1 for item in per_site_metrics if str(item.get("excluded_from_comparison_reason") or "") == "timeout"
        ),
        "partial_site_excluded_count": sum(
            1 for item in per_site_metrics if bool(item.get("partial")) and not bool(item.get("valid_for_comparison"))
        ),
        "raw_detected_anomaly_count": raw_detected,
        "unique_detected_anomaly_count": unique_detected,
        "avg_step_elapsed_ms": _avg_int(item.get("avg_step_elapsed_ms") for item in per_site_metrics),
        "max_step_elapsed_ms": max((int(item.get("max_step_elapsed_ms") or 0) for item in per_site_metrics), default=0),
        "reset_elapsed_ms": _avg_int(item.get("reset_elapsed_ms") for item in per_site_metrics),
        "site_elapsed_ms": _avg_int(item.get("site_elapsed_ms") for item in per_site_metrics),
        "episode_elapsed_ms": _avg_int(item.get("episode_elapsed_ms") for item in per_site_metrics),
        "early_stop_count": sum(int(item.get("early_stop_count") or 0) for item in per_site_metrics),
        "early_stop_reasons": dict(early_stop_reasons),
        "timeout_count": sum(int(item.get("timeout_count") or 0) for item in per_site_metrics),
        "use_memory_encoder": any(bool(item.get("use_memory_encoder", item.get("memory_encoder_enabled", False))) for item in per_site_metrics),
        "memory_encoder_enabled": memory_enabled,
        "memory_encoder_type": next((str(item.get("memory_encoder_type") or "") for item in per_site_metrics if item.get("memory_encoder_type")), ""),
        "memory_hidden_size": max((int(item.get("memory_hidden_size") or 0) for item in per_site_metrics), default=0),
        "memory_state_reset_count": sum(int(item.get("memory_state_reset_count", item.get("episode_memory_reset_count", 0)) or 0) for item in per_site_metrics),
        "episode_memory_reset_count": sum(int(item.get("episode_memory_reset_count") or 0) for item in per_site_metrics),
        "memory_state_update_count": sum(int(item.get("memory_state_update_count") or 0) for item in per_site_metrics) if memory_enabled else 0,
        "memory_state_norm_mean": _avg_float(item.get("memory_state_norm_mean") for item in per_site_metrics) if memory_enabled else 0.0,
        "repeated_action_rate": _avg_float(item.get("repeated_action_rate") for item in per_site_metrics),
        "repeated_click_target_rate": _avg_float(item.get("repeated_click_target_rate") for item in per_site_metrics),
        "repeated_action_rate_before_memory": None,
        "repeated_action_rate_after_memory": _avg_float(item.get("repeated_action_rate_after_memory") for item in per_site_metrics) if memory_enabled else None,
        "diversity_reward_total": sum(float(item.get("diversity_reward_total") or 0.0) for item in per_site_metrics),
        "repeated_action_penalty_total": sum(float(item.get("repeated_action_penalty_total") or 0.0) for item in per_site_metrics),
        "first_click_reward_count": sum(int(item.get("first_click_reward_count") or 0) for item in per_site_metrics),
        "new_action_type_reward_count": sum(int(item.get("new_action_type_reward_count") or 0) for item in per_site_metrics),
        "new_target_reward_count": sum(int(item.get("new_target_reward_count") or 0) for item in per_site_metrics),
        "inspect_reward_count": sum(int(item.get("inspect_reward_count") or 0) for item in per_site_metrics),
        "open_detail_panel_repeat_penalty_count": sum(int(item.get("open_detail_panel_repeat_penalty_count") or 0) for item in per_site_metrics),
        "no_effect_action_penalty_count": sum(int(item.get("no_effect_action_penalty_count") or 0) for item in per_site_metrics),
        "action_diversity_score": _avg_float(item.get("action_diversity_score") for item in per_site_metrics),
        "unique_action_type_count": sum(int(item.get("unique_action_type_count") or 0) for item in per_site_metrics),
        "unique_clicked_target_count": sum(int(item.get("unique_clicked_target_count") or 0) for item in per_site_metrics),
        "unique_element_key_count": sum(int(item.get("unique_element_key_count") or 0) for item in per_site_metrics),
        "unique_clicked_element_key_count": sum(int(item.get("unique_clicked_element_key_count") or 0) for item in per_site_metrics),
        "repeated_action_type_count": sum(int(item.get("repeated_action_type_count") or 0) for item in per_site_metrics),
        "repeated_action_signature_count": sum(int(item.get("repeated_action_signature_count") or 0) for item in per_site_metrics),
        "repeated_action_count": sum(int(item.get("repeated_action_count") or item.get("repeated_action_signature_count") or 0) for item in per_site_metrics),
        "repeated_element_key_click_count": sum(int(item.get("repeated_element_key_click_count") or 0) for item in per_site_metrics),
        "element_key_collision_count": sum(int(item.get("element_key_collision_count") or 0) for item in per_site_metrics),
        "element_key_missing_count": sum(int(item.get("element_key_missing_count") or 0) for item in per_site_metrics),
        "first_click_step": min(
            [int(item.get("first_click_step")) for item in per_site_metrics if item.get("first_click_step") is not None],
            default=None,
        ),
        "open_detail_panel_count": sum(int(item.get("open_detail_panel_count") or 0) for item in per_site_metrics),
        "click_retry_button_count": sum(int(item.get("click_retry_button_count") or 0) for item in per_site_metrics),
        "functional_action_count": sum(int(item.get("functional_action_count") or 0) for item in per_site_metrics),
        "functional_priority_candidate_count": sum(int(item.get("functional_priority_candidate_count") or 0) for item in per_site_metrics),
        "functional_priority_click_count": sum(int(item.get("functional_priority_click_count") or 0) for item in per_site_metrics),
        "filter_candidate_count": sum(int(item.get("filter_candidate_count") or 0) for item in per_site_metrics),
        "filter_click_count": sum(int(item.get("filter_click_count") or 0) for item in per_site_metrics),
        "category_candidate_count": sum(int(item.get("category_candidate_count") or 0) for item in per_site_metrics),
        "category_click_count": sum(int(item.get("category_click_count") or 0) for item in per_site_metrics),
        "tab_candidate_count": sum(int(item.get("tab_candidate_count") or 0) for item in per_site_metrics),
        "tab_click_count": sum(int(item.get("tab_click_count") or 0) for item in per_site_metrics),
        "enroll_candidate_count": sum(int(item.get("enroll_candidate_count") or 0) for item in per_site_metrics),
        "enroll_click_count": sum(int(item.get("enroll_click_count") or 0) for item in per_site_metrics),
        "search_input_click_count": sum(int(item.get("search_input_click_count") or 0) for item in per_site_metrics),
        "search_input_fill_count": sum(int(item.get("search_input_fill_count") or 0) for item in per_site_metrics),
        "low_value_generic_candidate_count": sum(int(item.get("low_value_generic_candidate_count") or 0) for item in per_site_metrics),
        "low_value_generic_click_count": sum(int(item.get("low_value_generic_click_count") or 0) for item in per_site_metrics),
        "low_value_generic_no_response_suppressed_count": sum(
            int(item.get("low_value_generic_no_response_suppressed_count") or 0) for item in per_site_metrics
        ),
        "policy_selected_low_value_generic_count": sum(int(item.get("policy_selected_low_value_generic_count") or 0) for item in per_site_metrics),
        "filtered_no_response_suppressed_count": sum(int(item.get("filtered_no_response_suppressed_count") or 0) for item in per_site_metrics),
        "no_response_suppressed_by_text_delta_count": sum(int(item.get("no_response_suppressed_by_text_delta_count") or 0) for item in per_site_metrics),
        "no_response_suppressed_by_candidate_delta_count": sum(int(item.get("no_response_suppressed_by_candidate_delta_count") or 0) for item in per_site_metrics),
        "no_response_suppressed_by_filter_semantics_count": sum(int(item.get("no_response_suppressed_by_filter_semantics_count") or 0) for item in per_site_metrics),
        "search_click_no_effect_suppressed_count": sum(int(item.get("search_click_no_effect_suppressed_count") or 0) for item in per_site_metrics),
        "no_progress_delayed_by_unclicked_functional_candidate_count": sum(
            int(item.get("no_progress_delayed_by_unclicked_functional_candidate_count") or 0) for item in per_site_metrics
        ),
        "visited_target_suppression_count": sum(int(item.get("visited_target_suppression_count") or 0) for item in per_site_metrics),
        "visited_target_skipped_count": sum(int(item.get("visited_target_skipped_count") or 0) for item in per_site_metrics),
        "visited_element_key_suppression_count": sum(int(item.get("visited_element_key_suppression_count") or 0) for item in per_site_metrics),
        "unclicked_high_value_candidate_selected_count": sum(int(item.get("unclicked_high_value_candidate_selected_count") or 0) for item in per_site_metrics),
        "unclicked_functional_candidate_selected_count": sum(int(item.get("unclicked_functional_candidate_selected_count") or 0) for item in per_site_metrics),
        "unclicked_element_key_selected_count": sum(int(item.get("unclicked_element_key_selected_count") or 0) for item in per_site_metrics),
        "memory_guided_candidate_switch_count": sum(int(item.get("memory_guided_candidate_switch_count") or 0) for item in per_site_metrics),
        "repeated_action_redirect_count": sum(int(item.get("repeated_action_redirect_count") or 0) for item in per_site_metrics),
        "repeated_target_redirect_count": sum(int(item.get("repeated_target_redirect_count") or 0) for item in per_site_metrics),
        "finish_delayed_by_unclicked_candidate_count": sum(int(item.get("finish_delayed_by_unclicked_candidate_count") or 0) for item in per_site_metrics),
        "repeated_meta_action_suppression_count": sum(int(item.get("repeated_meta_action_suppression_count") or 0) for item in per_site_metrics),
        "inspect_console_repeat_suppression_count": sum(int(item.get("inspect_console_repeat_suppression_count") or 0) for item in per_site_metrics),
        "inspect_network_repeat_suppression_count": sum(int(item.get("inspect_network_repeat_suppression_count") or 0) for item in per_site_metrics),
        "meta_action_redirect_to_functional_count": sum(int(item.get("meta_action_redirect_to_functional_count") or 0) for item in per_site_metrics),
        "semantic_action_type_counts": dict(semantic_action_type_counts),
        "clicked_semantic_action_types": dict(clicked_semantic_action_types),
        "attempted_semantic_action_types": dict(attempted_semantic_action_types),
        "semantic_no_effect_click_count": sum(int(item.get("semantic_no_effect_click_count") or 0) for item in per_site_metrics),
        "high_value_no_effect_click_count": sum(int(item.get("high_value_no_effect_click_count") or 0) for item in per_site_metrics),
        "semantic_button_no_response_count": sum(int(item.get("semantic_button_no_response_count") or 0) for item in per_site_metrics),
        "functional_no_effect_anomaly_count": sum(int(item.get("functional_no_effect_anomaly_count") or 0) for item in per_site_metrics),
        "workout_add_no_effect_count": sum(int(item.get("workout_add_no_effect_count") or 0) for item in per_site_metrics),
        "cart_no_effect_count": sum(int(item.get("cart_no_effect_count") or 0) for item in per_site_metrics),
        "add_no_effect_count": sum(int(item.get("add_no_effect_count") or 0) for item in per_site_metrics),
        "enroll_no_effect_count": sum(int(item.get("enroll_no_effect_count") or 0) for item in per_site_metrics),
        "high_value_enroll_no_response_count": sum(int(item.get("high_value_enroll_no_response_count") or 0) for item in per_site_metrics),
        "high_value_button_no_response_count": sum(int(item.get("high_value_button_no_response_count") or 0) for item in per_site_metrics),
        "no_effect_click_to_new_candidate_fallback_count": sum(int(item.get("no_effect_click_to_new_candidate_fallback_count") or 0) for item in per_site_metrics),
        "action_fallback_count": sum(int(item.get("action_fallback_count") or 0) for item in per_site_metrics),
        "action_fallback_invalid_action_count": sum(int(item.get("action_fallback_invalid_action_count") or 0) for item in per_site_metrics),
        "action_fallback_visited_element_key_count": sum(int(item.get("action_fallback_visited_element_key_count") or 0) for item in per_site_metrics),
        "action_fallback_low_value_candidate_count": sum(int(item.get("action_fallback_low_value_candidate_count") or 0) for item in per_site_metrics),
        "action_fallback_exploration_redirect_count": sum(int(item.get("action_fallback_exploration_redirect_count") or 0) for item in per_site_metrics),
        "policy_selected_valid_action_count": sum(int(item.get("policy_selected_valid_action_count") or 0) for item in per_site_metrics),
        "policy_selected_invalid_action_count": sum(int(item.get("policy_selected_invalid_action_count") or 0) for item in per_site_metrics),
        "policy_selected_visited_element_count": sum(int(item.get("policy_selected_visited_element_count") or 0) for item in per_site_metrics),
        "policy_selected_low_priority_action_count": sum(int(item.get("policy_selected_low_priority_action_count") or 0) for item in per_site_metrics),
        "fallback_applied_count": sum(int(item.get("fallback_applied_count") or 0) for item in per_site_metrics),
        "fallback_applied_rate": _safe_rate(
            sum(int(item.get("fallback_applied_count") or 0) for item in per_site_metrics),
            sum(
                int(item.get("policy_selected_valid_action_count") or 0)
                + int(item.get("policy_selected_invalid_action_count") or 0)
                for item in per_site_metrics
            ),
        ),
        "fallback_penalty_total": sum(float(item.get("fallback_penalty_total") or 0.0) for item in per_site_metrics),
        "fallback_reward_capped_count": sum(int(item.get("fallback_reward_capped_count") or 0) for item in per_site_metrics),
        "policy_executed_action_match_count": sum(int(item.get("policy_executed_action_match_count") or 0) for item in per_site_metrics),
        "policy_executed_action_mismatch_count": sum(int(item.get("policy_executed_action_mismatch_count") or 0) for item in per_site_metrics),
        "unclicked_functional_priority_candidate_count": sum(int(item.get("unclicked_functional_priority_candidate_count") or 0) for item in per_site_metrics),
        "high_value_functional_candidate_count": sum(int(item.get("high_value_functional_candidate_count") or 0) for item in per_site_metrics),
        "first_functional_action_step": min(
            [int(item.get("first_functional_action_step")) for item in per_site_metrics if item.get("first_functional_action_step") is not None],
            default=None,
        ),
        "unique_functional_action_type_count": sum(int(item.get("unique_functional_action_type_count") or 0) for item in per_site_metrics),
        "unique_functional_target_count": sum(int(item.get("unique_functional_target_count") or 0) for item in per_site_metrics),
        "debug_meta_action_count": sum(int(item.get("debug_meta_action_count") or 0) for item in per_site_metrics),
        "debug_meta_repeat_count": sum(int(item.get("debug_meta_repeat_count") or 0) for item in per_site_metrics),
        "no_functional_action_episode_count": sum(int(item.get("no_functional_action_episode_count") or 0) for item in per_site_metrics),
        "functional_action_signal_delta_count": sum(int(item.get("functional_action_signal_delta_count") or 0) for item in per_site_metrics),
        "functional_action_network_delta_count": sum(int(item.get("functional_action_network_delta_count") or 0) for item in per_site_metrics),
        "verification_action_after_high_value_click_count": sum(
            int(item.get("verification_action_after_high_value_click_count") or 0) for item in per_site_metrics
        ),
        "verification_action_redirect_count": sum(int(item.get("verification_action_redirect_count") or 0) for item in per_site_metrics),
        "high_value_click_pending_verification_count": sum(
            int(item.get("high_value_click_pending_verification_count") or 0) for item in per_site_metrics
        ),
        "high_value_click_verified_count": sum(int(item.get("high_value_click_verified_count") or 0) for item in per_site_metrics),
        "high_value_click_without_verification_count": sum(int(item.get("high_value_click_without_verification_count") or 0) for item in per_site_metrics),
        "meta_action_repeated_priority_lowered_count": sum(
            int(item.get("meta_action_repeated_priority_lowered_count") or 0) for item in per_site_metrics
        ),
        "policy_selected_meta_action_suppressed_count": sum(
            int(item.get("policy_selected_meta_action_suppressed_count") or 0) for item in per_site_metrics
        ),
        "reward_functional_action_total": sum(float(item.get("reward_functional_action_total") or 0.0) for item in per_site_metrics),
        "penalty_debug_meta_total": sum(float(item.get("penalty_debug_meta_total") or 0.0) for item in per_site_metrics),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "known_bug_reward_total": 0.0,
        "signal_reward_total": sum(float(item.get("signal_reward_total") or 0.0) for item in per_site_metrics),
        "exploration_reward_total": sum(float(item.get("exploration_reward_total") or 0.0) for item in per_site_metrics),
        "ui_dom_signal_reward_total": sum(float(item.get("ui_dom_signal_reward_total") or 0.0) for item in per_site_metrics),
        "console_runtime_signal_reward_total": sum(float(item.get("console_runtime_signal_reward_total") or 0.0) for item in per_site_metrics),
        "network_api_signal_reward_total": sum(float(item.get("network_api_signal_reward_total") or 0.0) for item in per_site_metrics),
        "cross_layer_signal_reward_total": sum(float(item.get("cross_layer_signal_reward_total") or 0.0) for item in per_site_metrics),
        "security_signal_reward_total": sum(float(item.get("security_signal_reward_total") or 0.0) for item in per_site_metrics),
        "repeated_penalty_total": sum(float(item.get("repeated_penalty_total") or 0.0) for item in per_site_metrics),
        "multi_signal_anomaly_count": sum(int(item.get("multi_signal_anomaly_count") or 0) for item in per_site_metrics),
        "console_error_count": sum(int(item.get("console_error_count") or 0) for item in per_site_metrics),
        "runtime_exception_count": sum(int(item.get("runtime_exception_count") or 0) for item in per_site_metrics),
        "network_error_count": sum(int(item.get("network_error_count") or 0) for item in per_site_metrics),
        "api_4xx_count": sum(int(item.get("api_4xx_count") or 0) for item in per_site_metrics),
        "api_5xx_count": sum(int(item.get("api_5xx_count") or 0) for item in per_site_metrics),
        "api_timeout_count": sum(int(item.get("api_timeout_count") or 0) for item in per_site_metrics),
        "api_ui_mismatch_count": sum(int(item.get("api_ui_mismatch_count") or 0) for item in per_site_metrics),
        "security_signal_count": sum(int(item.get("security_signal_count") or 0) for item in per_site_metrics),
        "signal_delta_reward_total": sum(float(item.get("signal_delta_reward_total") or 0.0) for item in per_site_metrics),
        "inspect_action_reward_total": sum(float(item.get("inspect_action_reward_total") or 0.0) for item in per_site_metrics),
        "functional_action_signal_reward_total": sum(float(item.get("functional_action_signal_reward_total") or 0.0) for item in per_site_metrics),
        "playwright_console_listener_enabled": any(bool(item.get("playwright_console_listener_enabled")) for item in per_site_metrics),
        "playwright_pageerror_listener_enabled": any(bool(item.get("playwright_pageerror_listener_enabled")) for item in per_site_metrics),
        "playwright_network_listener_enabled": any(bool(item.get("playwright_network_listener_enabled")) for item in per_site_metrics),
        "playwright_request_listener_enabled": any(bool(item.get("playwright_request_listener_enabled")) for item in per_site_metrics),
        "playwright_response_listener_enabled": any(bool(item.get("playwright_response_listener_enabled")) for item in per_site_metrics),
        "playwright_requestfailed_listener_enabled": any(bool(item.get("playwright_requestfailed_listener_enabled")) for item in per_site_metrics),
        "playwright_listener_warnings": [
            str(item.get("playwright_listener_warning") or "")
            for item in per_site_metrics
            if item.get("playwright_listener_warning")
        ],
        "target_signal_types": sorted(target_signal_types),
        "observed_signal_types": sorted(observed_signal_types or set(signal_metrics.get("observed_signal_types", []))),
        "missing_signal_types": sorted(target_signal_types - observed_signal_types) if target_signal_types else [],
        **signal_metrics,
    }


def compute_signal_discovery_metrics(
    per_site_metrics: Sequence[Mapping[str, Any]],
    *,
    target_signal_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    targets = {str(item) for item in target_signal_types or [] if item}
    observed: set[str] = set()
    ui_dom_count = 0
    console_runtime_count = 0
    network_api_count = 0
    cross_layer_count = 0
    security_count = 0
    state_coverage_count = 0
    new_url_count = 0
    new_dom_state_count = 0
    for item in per_site_metrics:
        ui_dom = int(item.get("ui_dom_signal_count") or item.get("multi_signal_anomaly_count") or 0)
        console_runtime = int(item.get("console_runtime_signal_count") or item.get("console_error_count") or 0) + int(
            item.get("runtime_exception_count") or 0
        )
        network_api = int(item.get("network_api_signal_count") or item.get("network_error_count") or 0) + int(item.get("api_4xx_count") or 0) + int(
            item.get("api_5xx_count") or 0
        ) + int(item.get("api_timeout_count") or 0)
        cross_layer = int(item.get("cross_layer_signal_count") or item.get("api_ui_mismatch_count") or 0)
        security = int(item.get("security_signal_count") or 0)
        ui_dom_count += ui_dom
        console_runtime_count += console_runtime
        network_api_count += network_api
        cross_layer_count += cross_layer
        security_count += security
        if ui_dom:
            observed.update({"ui-dom", "interaction", "layout"})
        if console_runtime:
            if int(item.get("console_error_count") or 0):
                observed.add("console-error")
            if int(item.get("runtime_exception_count") or 0):
                observed.add("runtime-exception")
        if network_api:
            if int(item.get("network_error_count") or 0):
                observed.add("network-error")
            if int(item.get("api_4xx_count") or 0):
                observed.add("api-4xx")
            if int(item.get("api_5xx_count") or 0):
                observed.add("api-5xx")
            if int(item.get("api_timeout_count") or 0):
                observed.add("api-timeout")
        if cross_layer:
            observed.add("api-ui-mismatch")
        if security:
            observed.update({"auth-permission-anomaly", "sensitive-data-exposure", "token-exposure"})
        for value in item.get("observed_signal_types", []) or []:
            if value:
                observed.add(str(value))
        state_coverage_count += int(item.get("state_coverage_count") or 0)
        new_url_count += int(item.get("new_url_count") or 0)
        new_dom_state_count += int(item.get("new_dom_state_count") or 0)
    coverage = (len(observed & targets) / len(targets)) if targets else (len(observed) / 5.0 if observed else 0.0)
    signal_discovery_score = min(
        1.0,
        (0.2 if ui_dom_count else 0.0)
        + (0.2 if console_runtime_count else 0.0)
        + (0.2 if network_api_count else 0.0)
        + (0.2 if cross_layer_count else 0.0)
        + (0.2 if security_count else 0.0),
    )
    return {
        "signal_discovery_score": round(signal_discovery_score, 4),
        "unique_signal_type_count": len(observed),
        "signal_type_coverage": round(coverage, 4),
        "ui_dom_signal_count": ui_dom_count,
        "console_runtime_signal_count": console_runtime_count,
        "network_api_signal_count": network_api_count,
        "cross_layer_signal_count": cross_layer_count,
        "security_signal_count": security_count,
        "state_coverage_count": state_coverage_count,
        "new_url_count": new_url_count,
        "new_dom_state_count": new_dom_state_count,
        "observed_signal_types": sorted(observed),
    }


def split_site_metrics(per_site_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_site_metrics = aggregate_site_metrics(per_site_metrics)
    valid_sites = [item for item in per_site_metrics if str(item.get("status") or "") != "failed"]
    failed_sites = [item for item in per_site_metrics if str(item.get("status") or "") == "failed"]
    valid_site_metrics = aggregate_site_metrics(valid_sites)
    valid_site_metrics["valid_site_count"] = len(valid_sites)
    valid_site_metrics["valid_site_ids"] = [str(item.get("site_id") or "") for item in valid_sites]

    excluded_known_bug_ids: list[str] = []
    for item in failed_sites:
        for bug_id in item.get("missed_bug_ids", []) or []:
            if bug_id:
                excluded_known_bug_ids.append(str(bug_id))
    failed_site_metrics = {
        "failed_site_count": len(failed_sites),
        "failed_site_ids": [str(item.get("site_id") or "") for item in failed_sites],
        "excluded_known_bug_count": len(sorted(set(excluded_known_bug_ids))),
        "excluded_known_bug_ids": sorted(set(excluded_known_bug_ids)),
        "excluded_from_valid_metrics": True,
    }
    return {
        "all_site_metrics": all_site_metrics,
        "valid_site_metrics": valid_site_metrics,
        "failed_site_metrics": failed_site_metrics,
    }


def strip_known_bug_metrics_for_signal_discovery(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    known_keys = (
        "precision",
        "recall",
        "f1_score",
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "known_bug_total",
        "known_bug_matched_count",
    )
    for section_name in ("all_site_metrics", "valid_site_metrics"):
        section = result.get(section_name)
        if not isinstance(section, Mapping):
            continue
        stripped = dict(section)
        for key in known_keys:
            stripped[key] = None
        result[section_name] = stripped
    return result


def comparison_note() -> dict[str, Any]:
    return {
        "previous_version": "v2_browsergym_raw_obs",
        "previous_result_status": "excluded_from_final_metrics",
        "exclusion_reason": "possible label leakage through action mask using catalog_bug_id_matches",
        "current_version": VERSION,
        "current_result_status": "valid_after_policy_safe_audit",
        "policy_safety_checks": [
            "observation_vector_ignores_bug_label_metadata",
            "action_mask_ignores_bug_label_metadata",
            "action_mask_stats_ignore_bug_label_metadata",
            "known_bug_matching_is_post_action_only",
            "reward_eval_paths_policy_safe",
        ],
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_comparison_note(path: str | Path = COMPARISON_NOTE_PATH) -> None:
    write_json(path, comparison_note())


def _known_bug_total(site_result: Mapping[str, Any]) -> int:
    if site_result.get("known_bug_total") is not None:
        return int(site_result.get("known_bug_total") or 0)
    matched = len(site_result.get("matched_bug_ids", []) or [])
    missed = len(site_result.get("missed_bug_ids", []) or [])
    return matched + missed


def _completed_episodes(site_result: Mapping[str, Any]) -> int:
    if site_result.get("completed_episodes") is not None:
        return int(site_result.get("completed_episodes") or 0)
    return int(site_result.get("episodes") or site_result.get("episode_count") or 0)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    try:
        denom = int(denominator or 0)
        if denom <= 0:
            return 0.0
        return round(int(numerator or 0) / denom, 4)
    except (TypeError, ValueError):
        return 0.0


def _rate_from_counts(counts: Any) -> float:
    if not isinstance(counts, Mapping):
        return 0.0
    total = 0
    repeated = 0
    for value in counts.values():
        count = int(value or 0)
        total += count
        if count > 1:
            repeated += count - 1
    return round(repeated / total, 4) if total else 0.0


def _avg_int(values: Any) -> int:
    normalized = [int(value or 0) for value in values if value is not None]
    return int(sum(normalized) / len(normalized)) if normalized else 0


def _avg_float(values: Any) -> float:
    normalized = [float(value or 0.0) for value in values if value is not None]
    return float(sum(normalized) / len(normalized)) if normalized else 0.0
