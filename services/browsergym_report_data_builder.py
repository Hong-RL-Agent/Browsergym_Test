"""Build PDF report data from BrowserGym PPO autonomous exploration artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def build_browsergym_report_data(
    site_id: str,
    base_url: str,
    detected_bugs_path: str | Path,
    training_summary_path: str | Path,
    transition_log_path: str | Path,
    bug_catalog_path: str | Path,
    model_path: str | Path | None = None,
    api_results_path: str | Path | None = None,
) -> dict:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    detected_bugs = _read_json_list(detected_bugs_path)
    confirmed_detected = [item for item in detected_bugs if item.get("matched_bug_id")]
    catalog_related = [item for item in detected_bugs if not item.get("matched_bug_id") and _is_catalog_related(item.get("evidence", {}))]
    exploratory = [item for item in detected_bugs if not item.get("matched_bug_id") and not _is_catalog_related(item.get("evidence", {}))]
    training_summary = _read_json_dict(training_summary_path)
    transitions = _read_jsonl(transition_log_path)
    known_bugs = _load_bug_catalog(site_id, bug_catalog_path)
    transition_summary = _summarize_transitions(transitions)
    api_results = _read_json_dict(api_results_path) if api_results_path else {}
    api_anomalies = _api_anomalies_for_site(api_results, site_id)

    matched_bug_ids = sorted(
        {
            str(item.get("matched_bug_id"))
            for item in detected_bugs
            if item.get("matched_bug_id")
        }
    )
    known_bug_ids = [_bug_id(item) for item in known_bugs if _bug_id(item)]
    missed_bug_ids = [bug_id for bug_id in known_bug_ids if bug_id not in set(matched_bug_ids)]
    detected_by_bug = _best_detected_by_bug(detected_bugs)
    actions_by_bug = _actions_by_bug(transitions)

    summary = {
        "episodes": training_summary.get("episodes") or transition_summary["episode_count"],
        "average_reward": training_summary.get("average_reward", training_summary.get("average_episode_reward", transition_summary["average_reward"])),
        "detected_bug_count": len(matched_bug_ids),
        "total_detected_candidates": training_summary.get("total_detected_candidates", sum(len(row.get("anomalies", []) or []) for row in transitions)),
        "unique_detected_candidates": training_summary.get("unique_detected_candidates", len(detected_bugs)),
        "known_bug_match_count": len(matched_bug_ids),
        "matched_bug_ids": matched_bug_ids,
        "missed_bug_ids": missed_bug_ids,
        "precision": training_summary.get("precision", _safe_div(len(matched_bug_ids), max(1, len(detected_bugs)))),
        "known_bug_precision": training_summary.get(
            "known_bug_precision",
            _safe_div(len(matched_bug_ids), max(1, len(confirmed_detected) + len(catalog_related))),
        ),
        "catalog_precision": training_summary.get(
            "catalog_precision",
            _safe_div(len(confirmed_detected), max(1, len(confirmed_detected) + len(catalog_related))),
        ),
        "exploratory_anomaly_count": training_summary.get("exploratory_anomaly_count", len(exploratory)),
        "catalog_related_anomaly_count": len(catalog_related),
        "recall": training_summary.get("recall", _safe_div(len(matched_bug_ids), max(1, len(known_bug_ids)))),
        "false_positive_count": training_summary.get("false_positive_count", 0),
        "purchase_click_count": training_summary.get("purchase_click_count", transition_summary["purchase_click_count"]),
        "inspect_cart_count": training_summary.get("inspect_cart_count", transition_summary["action_counts"].get("inspect_cart", 0)),
        "cart_count_detected_count": training_summary.get("cart_count_detected_count", transition_summary["cart_count_detected_count"]),
        "button_no_response_candidates": training_summary.get("button_no_response_candidates", transition_summary["button_no_response_candidates"]),
        "action_counts": training_summary.get("action_counts", transition_summary["action_counts"]),
        "raw_anomaly_candidate_count": _summary_value(training_summary, transition_summary, "raw_anomaly_candidate_count", 0),
        "filtered_false_positive_count": _summary_value(training_summary, transition_summary, "filtered_false_positive_count", 0),
        "duplicate_anomaly_count": _summary_value(training_summary, transition_summary, "duplicate_anomaly_count", 0),
        "anomaly_filter_reason_counts": _summary_value(training_summary, transition_summary, "anomaly_filter_reason_counts", {}),
        "finding_promotion_reason": _summary_value(training_summary, transition_summary, "finding_promotion_reason", ""),
        "finding_rejection_reason": _summary_value(training_summary, transition_summary, "finding_rejection_reason", ""),
        "action_diversity_score": _summary_value(training_summary, transition_summary, "action_diversity_score", 0.0),
        "unique_action_type_count": _summary_value(training_summary, transition_summary, "unique_action_type_count", 0),
        "repeated_action_type_count": _summary_value(training_summary, transition_summary, "repeated_action_type_count", 0),
        "repeated_strategy_count": _summary_value(training_summary, transition_summary, "repeated_strategy_count", 0),
        "consecutive_same_strategy_count": _summary_value(training_summary, transition_summary, "consecutive_same_strategy_count", 0),
        "generated_opportunity_count": _summary_value(training_summary, transition_summary, "generated_opportunity_count", 0),
        "required_opportunity_count": _summary_value(training_summary, transition_summary, "required_opportunity_count", 0),
        "executed_opportunity_count": _summary_value(training_summary, transition_summary, "executed_opportunity_count", 0),
        "verified_opportunity_count": _summary_value(training_summary, transition_summary, "verified_opportunity_count", 0),
        "failed_opportunity_count": _summary_value(training_summary, transition_summary, "failed_opportunity_count", 0),
        "pending_opportunity_count": _summary_value(training_summary, transition_summary, "pending_opportunity_count", _summary_value(training_summary, transition_summary, "remaining_opportunity_count", 0)),
        "remaining_required_opportunity_count": _summary_value(training_summary, transition_summary, "remaining_required_opportunity_count", 0),
        "pending_required_opportunity_count": _summary_value(training_summary, transition_summary, "pending_required_opportunity_count", _summary_value(training_summary, transition_summary, "remaining_required_opportunity_count", 0)),
        "action_opportunity_coverage_rate": _summary_value(training_summary, transition_summary, "action_opportunity_coverage_rate", 0.0),
        "action_coverage_rate": _summary_value(training_summary, transition_summary, "action_coverage_rate", _summary_value(training_summary, transition_summary, "action_opportunity_coverage_rate", 0.0)),
        "required_opportunity_completion_rate": _summary_value(training_summary, transition_summary, "required_opportunity_completion_rate", 0.0),
        "unverified_anomaly_count": _summary_value(training_summary, transition_summary, "unverified_anomaly_count", 0),
        "verified_finding_count": _summary_value(training_summary, transition_summary, "verified_finding_count", 0),
        "anomaly_verification_required": _summary_value(training_summary, transition_summary, "anomaly_verification_required", False),
        "anomaly_verification_completed": _summary_value(training_summary, transition_summary, "anomaly_verification_completed", False),
        "finish_allowed": _summary_value(training_summary, transition_summary, "finish_allowed", False),
        "finish_blocked_reason": _summary_value(training_summary, transition_summary, "finish_blocked_reason", ""),
        "opportunity_diagnostics": _summary_value(training_summary, transition_summary, "opportunity_diagnostics", []),
        "dom_exhaustive_action_mode": _summary_value(training_summary, transition_summary, "dom_exhaustive_action_mode", True),
        "completed_reason": _summary_value(training_summary, transition_summary, "completed_reason", ""),
        "valid_scan_run": _summary_value(training_summary, transition_summary, "valid_scan_run", False),
        "action_budget_status": _summary_value(training_summary, transition_summary, "action_budget_status", ""),
        "page_type": _summary_value(training_summary, transition_summary, "page_type", ""),
        "raw_event_log_count": _summary_value(training_summary, transition_summary, "raw_event_log_count", 0),
        "executed_action_count": _summary_value(training_summary, transition_summary, "executed_action_count", transition_summary["total_steps"]),
        "successful_action_count": _summary_value(training_summary, transition_summary, "successful_action_count", 0),
        "failed_action_count": _summary_value(training_summary, transition_summary, "failed_action_count", 0),
        "runner_failed_action_count": _summary_value(training_summary, transition_summary, "runner_failed_action_count", 0),
        "timeline_failed_action_count": _summary_value(training_summary, transition_summary, "timeline_failed_action_count", 0),
        "summary_failed_action_count": _summary_value(training_summary, transition_summary, "summary_failed_action_count", 0),
        "exploration_failed_action_count": _summary_value(training_summary, transition_summary, "exploration_failed_action_count", 0),
        "failed_action_count_mismatch": _summary_value(training_summary, transition_summary, "failed_action_count_mismatch", False),
        "failed_action_count_mismatch_reason": _summary_value(training_summary, transition_summary, "failed_action_count_mismatch_reason", ""),
        "valid_report_metrics": _summary_value(training_summary, transition_summary, "valid_report_metrics", True),
        "report_metric_warning": _summary_value(training_summary, transition_summary, "report_metric_warning", ""),
        "action_log_count": _summary_value(training_summary, transition_summary, "action_log_count", 0),
        "state_log_count": _summary_value(training_summary, transition_summary, "state_log_count", 0),
        "network_log_count": _summary_value(training_summary, transition_summary, "network_log_count", 0),
        "issue_log_count": _summary_value(training_summary, transition_summary, "issue_log_count", 0),
        "network_failure_count": _summary_value(training_summary, transition_summary, "network_failure_count", 0),
        "console_error_count": _summary_value(training_summary, transition_summary, "console_error_count", 0),
        "runtime_error_count": _summary_value(training_summary, transition_summary, "runtime_error_count", 0),
        "warning_issue_count": _summary_value(training_summary, transition_summary, "warning_issue_count", 0),
        "critical_issue_count": _summary_value(training_summary, transition_summary, "critical_issue_count", 0),
        "total_problem_signal_count": _summary_value(training_summary, transition_summary, "total_problem_signal_count", 0),
        "missing_failure_reason_count": _summary_value(training_summary, transition_summary, "missing_failure_reason_count", 0),
        "network_failure_detected": _summary_value(training_summary, transition_summary, "network_failure_detected", False),
        "network_verification_required": _summary_value(training_summary, transition_summary, "network_verification_required", False),
        "network_verification_performed": _summary_value(training_summary, transition_summary, "network_verification_performed", False),
        "network_verification_success": _summary_value(training_summary, transition_summary, "network_verification_success", False),
        "console_error_detected": _summary_value(training_summary, transition_summary, "console_error_detected", False),
        "console_verification_required": _summary_value(training_summary, transition_summary, "console_verification_required", False),
        "console_verification_performed": _summary_value(training_summary, transition_summary, "console_verification_performed", False),
        "console_verification_success": _summary_value(training_summary, transition_summary, "console_verification_success", False),
        "has_login_form": _summary_value(training_summary, transition_summary, "has_login_form", False),
        "has_username_or_email_input": _summary_value(training_summary, transition_summary, "has_username_or_email_input", False),
        "has_password_input": _summary_value(training_summary, transition_summary, "has_password_input", False),
        "has_login_submit": _summary_value(training_summary, transition_summary, "has_login_submit", False),
        "username_or_email_filled": _summary_value(training_summary, transition_summary, "username_or_email_filled", False),
        "password_filled": _summary_value(training_summary, transition_summary, "password_filled", False),
        "login_submit_clicked": _summary_value(training_summary, transition_summary, "login_submit_clicked", False),
        "login_result_checked": _summary_value(training_summary, transition_summary, "login_result_checked", False),
        "login_flow_attempted": _summary_value(training_summary, transition_summary, "login_flow_attempted", False),
        "login_flow_completed": _summary_value(training_summary, transition_summary, "login_flow_completed", False),
        "login_flow_status": _summary_value(training_summary, transition_summary, "login_flow_status", ""),
        "repeated_inspect_dom_count": _summary_value(training_summary, transition_summary, "repeated_inspect_dom_count", 0),
        "repeated_inspect_dom_penalty_applied": _summary_value(training_summary, transition_summary, "repeated_inspect_dom_penalty_applied", 0),
        "repeated_inspect_dom_blocked": _summary_value(training_summary, transition_summary, "repeated_inspect_dom_blocked", 0),
    }

    bugs = [
        _build_bug_row(
            known_bug,
            detected_by_bug.get(_bug_id(known_bug)),
            actions_by_bug.get(_bug_id(known_bug), []),
            summary,
        )
        for known_bug in known_bugs
    ]

    report_data = {
        "report_id": f"browsergym-{site_id}-{stamp}",
        "title": "웹사이트 오류 발견 보고서 / Web Application Error Report",
        "mode": "BrowserGym Autonomous PPO",
        "site_id": site_id,
        "base_url": base_url,
        "author": "J.A.W.S BrowserGym PPO Agent",
        "created_at": created_at,
        "page_name": f"{site_id} Main Page",
        "model": {
            "name": "PPO Actor-Critic",
            "path": str(model_path or training_summary.get("model_path") or ""),
            "environment": "BrowserGym openended",
            "policy_type": "MLP Actor-Critic",
        },
        "summary": summary,
        "environment": {
            "browser": "Chromium",
            "engine": "BrowserGym + Playwright",
            "rl_algorithm": "PPO",
            "execution_mode": "Autonomous exploration",
            "scenario_used": False,
            "base_url": base_url,
        },
        "transition_summary": transition_summary,
        "bugs": bugs,
        "confirmed_bugs": confirmed_detected,
        "catalog_related_anomalies": catalog_related,
        "exploratory_anomalies": exploratory,
        "api_anomalies": api_anomalies,
        "api_summary": _api_summary_for_site(api_results, site_id, api_anomalies),
        "analysis": _analysis_text(site_id, summary),
    }
    return report_data


def _build_bug_row(
    known_bug: Mapping[str, Any],
    detected: Optional[Mapping[str, Any]],
    rl_actions: List[str],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    bug_id = _bug_id(known_bug)
    evidence = dict(detected.get("evidence", {})) if detected else {}
    bug_type = str(known_bug.get("type") or known_bug.get("bug_type") or "")
    row = {
        "bug_id": bug_id,
        "type": bug_type,
        "severity": known_bug.get("severity", "Medium"),
        "detected": bool(detected),
        "status_label": "탐지 성공" if detected else "미탐지 / 추가 학습 필요",
        "detected_by": "BrowserGym PPO Agent" if detected else "-",
        "selector": known_bug.get("selector") or known_bug.get("selector_text") or "",
        "screen_location": known_bug.get("screen_location", ""),
        "related_files": _as_list(known_bug.get("related_files")),
        "symptom": known_bug.get("symptom") or known_bug.get("description") or "",
        "intended_cause": known_bug.get("intended_cause", ""),
        "expected_agent_behavior": known_bug.get("expected_agent_behavior", ""),
        "anomaly_type": detected.get("type") if detected else "",
        "confidence": detected.get("confidence") if detected else None,
        "evidence": evidence,
        "evidence_json": json.dumps(evidence, ensure_ascii=False, indent=2),
        "rl_actions": rl_actions,
        "impact": _as_list(known_bug.get("impact")) or _default_impact(bug_type),
        "recommended_fix": _as_list(known_bug.get("recommended_fix")) or _default_recommended_fix(bug_type),
        "matched_bug_id": detected.get("matched_bug_id") if detected else "",
        "match_reason": detected.get("match_reason") if detected else "",
        "scenario_used": False,
    }
    if bug_id == "site001-bug01":
        row["browsergym_details"] = {
            "inspect_cart_count": summary.get("inspect_cart_count", 0),
            "purchase_click_count": summary.get("purchase_click_count", 0),
            "cart_count_before": evidence.get("cart_count_before"),
            "cart_count_after": evidence.get("cart_count_after"),
            "clicked_text": evidence.get("clicked_text") or evidence.get("clicked_name"),
            "button_no_response_candidates": summary.get("button_no_response_candidates", 0),
        }
    elif bug_id == "site001-bug02":
        row["browsergym_details"] = {
            "duplicated_titles": evidence.get("duplicated_titles", {}),
            "book_context": evidence.get("book_context"),
        }
    elif bug_id == "site001-bug03":
        action_counts = summary.get("action_counts", {}) if isinstance(summary.get("action_counts"), Mapping) else {}
        row["browsergym_details"] = {
            "change_viewport_mobile_count": action_counts.get("change_viewport_mobile", 0),
            "inspect_layout_count": action_counts.get("inspect_layout", 0),
            "viewport_type": evidence.get("viewport_type"),
            "layout_overlap_count": evidence.get("layout_overlap_count"),
        }
    elif bug_id == "site003-bug01":
        row["browsergym_details"] = {
            "clicked_text": evidence.get("clicked_text") or evidence.get("clicked_name"),
            "before_url": evidence.get("before_url"),
            "after_url": evidence.get("after_url"),
            "dom_changed": evidence.get("dom_changed"),
            "modal_opened": evidence.get("modal_opened"),
            "form_opened": evidence.get("form_opened"),
        }
    elif bug_id == "site003-bug02":
        row["browsergym_details"] = {
            "section_text": evidence.get("section_text"),
            "empty_state_text": evidence.get("empty_state_text"),
            "chart_visible": evidence.get("chart_visible"),
            "ui_contradiction": evidence.get("ui_contradiction"),
        }
    elif bug_id == "site003-bug03":
        row["browsergym_details"] = {
            "selector": evidence.get("selector"),
            "child_bbox": evidence.get("child_bbox"),
            "parent_bbox": evidence.get("parent_bbox"),
            "overflow_right": evidence.get("overflow_right"),
            "layout_overflow_count": evidence.get("layout_overflow_count"),
            "bbox_source": evidence.get("bbox_source"),
            "section_keyword_matches": evidence.get("section_keyword_matches"),
            "target_keyword_matches": evidence.get("target_keyword_matches"),
            "expected_elements": evidence.get("expected_elements"),
            "evidence_quality": "fallback evidence; strengthen with DOM bbox if possible"
            if evidence.get("bbox_source") == "viewport_fallback" or not evidence.get("child_bbox")
            else "bbox evidence",
        }
    else:
        row["browsergym_details"] = {}
    return row


def _summarize_transitions(transitions: List[Mapping[str, Any]]) -> Dict[str, Any]:
    action_counts: Counter[str] = Counter()
    reward_totals: defaultdict[str, float] = defaultdict(float)
    episode_rewards: defaultdict[str, float] = defaultdict(float)
    episodes = set()
    purchase_click_count = 0
    cart_count_detected_count = 0
    button_no_response_candidates = 0
    raw_anomaly_candidate_count = 0
    duplicate_anomaly_count = 0
    anomaly_seen_keys = set()
    latest_metrics: Dict[str, Any] = {}

    for row in transitions:
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            episodes.add(episode_id)
            episode_rewards[episode_id] += float(row.get("reward", 0.0) or 0.0)
        action = row.get("action", {}) if isinstance(row.get("action"), Mapping) else {}
        action_type = str(row.get("action_type") or action.get("action_type") or "")
        if action_type:
            action_counts[action_type] += 1
        if bool(action.get("is_purchase_action")):
            purchase_click_count += 1
        state = row.get("state_summary", {}) if isinstance(row.get("state_summary"), Mapping) else {}
        if state.get("cart_count_detected"):
            cart_count_detected_count += 1
        for key, value in (row.get("reward_breakdown", {}) or {}).items():
            try:
                reward_totals[str(key)] += float(value or 0.0)
            except (TypeError, ValueError):
                pass
        for key in (
            "generated_opportunity_count",
            "required_opportunity_count",
            "executed_opportunity_count",
            "verified_opportunity_count",
            "failed_opportunity_count",
            "remaining_required_opportunity_count",
            "action_opportunity_coverage_rate",
            "required_opportunity_completion_rate",
            "unverified_anomaly_count",
            "verified_finding_count",
            "anomaly_verification_required",
            "anomaly_verification_completed",
            "finish_allowed",
            "finish_blocked_reason",
            "opportunity_diagnostics",
            "completed_reason",
            "valid_scan_run",
            "action_budget_status",
            "page_type",
            "raw_event_log_count",
            "executed_action_count",
            "successful_action_count",
            "failed_action_count",
            "runner_failed_action_count",
            "timeline_failed_action_count",
            "summary_failed_action_count",
            "exploration_failed_action_count",
            "failed_action_count_mismatch",
            "failed_action_count_mismatch_reason",
            "valid_report_metrics",
            "report_metric_warning",
            "action_log_count",
            "state_log_count",
            "network_log_count",
            "issue_log_count",
            "network_failure_count",
            "console_error_count",
            "runtime_error_count",
            "warning_issue_count",
            "critical_issue_count",
            "total_problem_signal_count",
            "missing_failure_reason_count",
            "network_failure_detected",
            "network_verification_required",
            "network_verification_performed",
            "network_verification_success",
            "console_error_detected",
            "console_verification_required",
            "console_verification_performed",
            "console_verification_success",
            "has_login_form",
            "has_username_or_email_input",
            "has_password_input",
            "has_login_submit",
            "username_or_email_filled",
            "password_filled",
            "login_submit_clicked",
            "login_result_checked",
            "login_flow_attempted",
            "login_flow_completed",
            "login_flow_status",
            "raw_anomaly_candidate_count",
            "filtered_false_positive_count",
            "duplicate_anomaly_count",
            "anomaly_filter_reason_counts",
            "finding_promotion_reason",
            "finding_rejection_reason",
            "action_diversity_score",
            "unique_action_type_count",
            "repeated_action_type_count",
            "repeated_strategy_count",
            "consecutive_same_strategy_count",
            "repeated_inspect_dom_count",
            "repeated_inspect_dom_penalty_applied",
            "repeated_inspect_dom_blocked",
        ):
            if key in row:
                latest_metrics[key] = row.get(key)
        anomalies = row.get("anomalies", []) if isinstance(row.get("anomalies"), list) else row.get("detected_anomalies", []) if isinstance(row.get("detected_anomalies"), list) else []
        raw_anomaly_candidate_count += len(anomalies)
        for anomaly in anomalies:
            if not isinstance(anomaly, Mapping):
                continue
            key = str(anomaly.get("type") or "") + "|" + str((anomaly.get("evidence") or {}).get("clicked_text") if isinstance(anomaly.get("evidence"), Mapping) else "")
            if key in anomaly_seen_keys:
                duplicate_anomaly_count += 1
            anomaly_seen_keys.add(key)
        button_no_response_candidates += sum(1 for item in anomalies if item.get("type") == "button-no-response")

    average_reward = _safe_div(sum(episode_rewards.values()), max(1, len(episodes)))
    return {
        "action_counts": dict(action_counts),
        "total_steps": len(transitions),
        "episode_count": len(episodes),
        "average_reward": average_reward,
        "most_common_actions": action_counts.most_common(6),
        "reward_breakdown_summary": {key: round(value, 4) for key, value in sorted(reward_totals.items())},
        "purchase_click_count": purchase_click_count,
        "cart_count_detected_count": cart_count_detected_count,
        "button_no_response_candidates": button_no_response_candidates,
        "raw_anomaly_candidate_count": raw_anomaly_candidate_count,
        "duplicate_anomaly_count": duplicate_anomaly_count,
        "filtered_false_positive_count": max(0, raw_anomaly_candidate_count - len(anomaly_seen_keys)),
        "anomaly_filter_reason_counts": {"duplicate_anomaly": duplicate_anomaly_count} if duplicate_anomaly_count else {},
        **latest_metrics,
    }


def _summary_value(training_summary: Mapping[str, Any], transition_summary: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in training_summary:
        return training_summary.get(key)
    if key in transition_summary:
        return transition_summary.get(key)
    return default


def _actions_by_bug(transitions: List[Mapping[str, Any]]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for row in transitions:
        action = row.get("action", {}) if isinstance(row.get("action"), Mapping) else {}
        action_type = str(row.get("action_type") or action.get("action_type") or "")
        anomalies = row.get("anomalies", []) if isinstance(row.get("anomalies"), list) else []
        for anomaly in anomalies:
            bug_id = str(anomaly.get("matched_bug_id") or "")
            if bug_id and action_type and action_type not in result[bug_id]:
                result[bug_id].append(action_type)
    return dict(result)


def _best_detected_by_bug(detected_bugs: List[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    best: Dict[str, Mapping[str, Any]] = {}
    for item in detected_bugs:
        bug_id = str(item.get("matched_bug_id") or "")
        if not bug_id:
            continue
        previous = best.get(bug_id)
        if previous is None or float(item.get("confidence", 0.0) or 0.0) > float(previous.get("confidence", 0.0) or 0.0):
            best[bug_id] = item
    return best


def _is_catalog_related(evidence: Any) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    return bool(
        evidence.get("catalog_bug_id_matches")
        or evidence.get("catalog_keyword_matches")
        or evidence.get("section_keyword_matches")
        or evidence.get("target_keyword_matches")
        or evidence.get("selector_hint")
        or evidence.get("data_bug_id")
    )


def _load_bug_catalog(site_id: str, path: str | Path) -> List[Dict[str, Any]]:
    raw = _read_json(path)
    data = raw if isinstance(raw, dict) else {}
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    bugs = data.get("bugs") or data.get("items") or data if isinstance(data, list) else data.get("bugs", [])
    if isinstance(bugs, list) and bugs:
        return [dict(item) for item in bugs if isinstance(item, Mapping)]
    if site_id == "site001":
        return [
            {"id": "site001-bug01", "type": "button-no-response", "severity": "High", "selector": "[data-bug-id=\"site001-bug01\"]", "symptom": "구매하기 버튼 클릭 후 장바구니 카운트가 증가하지 않음"},
            {"id": "site001-bug02", "type": "duplicated-rendering", "severity": "Medium", "selector": ".book-card", "symptom": "도서 카드가 중복 렌더링됨"},
            {"id": "site001-bug03", "type": "layout-overlap", "severity": "Medium", "selector": "mobile layout", "symptom": "모바일 viewport에서 레이아웃 겹침 발생"},
        ]
    return []


def _analysis_text(site_id: str, summary: Mapping[str, Any]) -> str:
    return (
        "본 보고서는 BrowserGym 기반 PPO Actor-Critic 에이전트가 시나리오 없이 웹사이트를 자율 탐색하여 생성한 오류 탐지 결과이다. "
        "에이전트는 DOM, accessibility tree, element visibility, cart state, layout overlap, action history를 observation으로 사용하고, "
        "click_element, inspect_cart, inspect_layout, change_viewport_mobile 등의 action을 선택했다. "
        f"최종 평가에서 {site_id}의 known bug {summary.get('known_bug_match_count', 0)}개를 매칭했으며 "
        f"recall {summary.get('recall', 0)}을 달성했다."
    )


def _read_json_dict(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_list(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _api_anomalies_for_site(api_results: Mapping[str, Any], site_id: str) -> List[Dict[str, Any]]:
    anomalies = api_results.get("anomalies", []) if isinstance(api_results, Mapping) else []
    if not isinstance(anomalies, list):
        return []
    return [
        dict(item)
        for item in anomalies
        if isinstance(item, Mapping) and str(item.get("site_id") or "") == site_id
    ]


def _api_summary_for_site(
    api_results: Mapping[str, Any],
    site_id: str,
    api_anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sites = api_results.get("sites", {}) if isinstance(api_results, Mapping) else {}
    site_result = sites.get(site_id, {}) if isinstance(sites, Mapping) else {}
    summary = site_result.get("summary", {}) if isinstance(site_result, Mapping) else {}
    if isinstance(summary, Mapping) and summary:
        result = dict(summary)
    else:
        counts = Counter(str(item.get("anomaly_type") or item.get("type") or "") for item in api_anomalies)
        result = {
            "api_5xx_count": counts.get("api-5xx-error", 0),
            "api_4xx_unexpected_count": counts.get("api-4xx-unexpected", 0),
            "schema_mismatch_count": counts.get("api-schema-mismatch", 0),
            "timeout_count": counts.get("api-timeout", 0),
        }
    result["api_anomaly_count"] = len(api_anomalies)
    return result


def _bug_id(bug: Mapping[str, Any]) -> str:
    return str(bug.get("id") or bug.get("bug_id") or bug.get("name") or "")


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _default_impact(bug_type: str) -> List[str]:
    return {
        "button-no-response": ["핵심 사용자 액션 실패", "구매 전환 흐름 차단"],
        "duplicated-rendering": ["콘텐츠 신뢰도 저하", "사용자 탐색 혼란"],
        "layout-overlap": ["모바일 사용성 저하", "클릭 및 판독 가능성 저하"],
    }.get(bug_type, ["사용자 경험 및 품질 저하"])


def _default_recommended_fix(bug_type: str) -> List[str]:
    return {
        "button-no-response": ["클릭 핸들러와 상태 업데이트 연결 확인", "cart_count 변경 회귀 테스트 추가"],
        "duplicated-rendering": ["렌더링 데이터 중복 제거", "목록 key 및 filter 로직 검증"],
        "layout-overlap": ["모바일 CSS breakpoint 조정", "레이아웃 시각 회귀 테스트 추가"],
    }.get(bug_type, ["원인 코드 수정 후 회귀 테스트 추가"])
