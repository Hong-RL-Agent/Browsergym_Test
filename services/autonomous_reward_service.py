"""Autonomous reward shaping for BrowserGym PPO.

The reward function intentionally avoids using pre-action catalog metadata or
hardcoded bug identifiers. Known-bug reward is applied only after anomaly
matching has produced post-action match metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from services.multisignal_collector import collect_multi_signal_counts
from services.web_error_finding_service import finding_signature


FUNCTIONAL_ACTION_TYPES = {
    "click_element",
    "fill_input",
    "press_enter",
    "submit_form",
    "click_button",
    "navigate_link",
    "select_option",
    "trigger_api_action",
}

DEBUG_META_ACTION_TYPES = {
    "open_detail_panel",
    "click_retry_button",
    "scroll_down",
    "scroll_up",
    "inspect_dom",
    "inspect_layout",
    "inspect_network",
    "inspect_console",
    "inspect_cart",
    "click_trigger_button",
    "click_recovery_button",
}


def calculate_autonomous_reward(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    anomalies: List[Dict[str, Any]],
    known_bugs: Optional[List[Dict[str, Any]]] = None,
    history: Optional[Mapping[str, Any]] = None,
    site_profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    history = history or {}
    site_profile = site_profile or {}
    reward_mode = str(site_profile.get("reward_mode") or "signal_based")
    # Training reward is blind by design: ground-truth/known-bug matches are metrics only.
    use_known_bug_reward = False
    action_type = str(action.get("action_type") or "")
    clicked_candidate = _clicked_candidate(before_observation, action)
    before_signature = _state_signature(before_observation)
    after_signature = _state_signature(after_observation)

    reward_new_anomaly = 0.0
    reward_known_bug_match = 0.0
    reward_type_diversity = 0.0
    reward_first_click_element = 0.0
    reward_new_action_type = 0.0
    reward_new_target = 0.0
    reward_first_inspect_dom = 0.0
    reward_first_inspect_network = 0.0
    reward_state_change = 0.0
    penalty_repeat_action = 0.0
    penalty_same_target_click = 0.0
    penalty_same_action_type_repeat = 0.0
    penalty_same_action_signature_repeat = 0.0
    penalty_open_detail_panel_repeat = 0.0
    penalty_no_effect_open_detail = 0.0
    penalty_invalid_action = 0.0
    penalty_no_effect = 0.0
    penalty_step_cost = -0.01
    penalty_timeout = 0.0
    reward_first_functional_action = 0.0
    reward_new_functional_action_type = 0.0
    reward_new_functional_target = 0.0
    reward_functional_action_caused_state_change = 0.0
    reward_functional_action_caused_signal_delta = 0.0
    reward_functional_action_caused_network_request = 0.0
    reward_verified_finding = 0.0
    reward_finding_evidence = 0.0
    reward_finding_coverage = 0.0
    reward_finding_reproduction = 0.0
    penalty_no_functional_action_episode = 0.0
    penalty_debug_meta_repeat = 0.0
    penalty_repeated_meta_action = 0.0
    penalty_retry_button_repeat = 0.0
    reward_email_input_filled = 0.0
    reward_password_input_filled = 0.0
    reward_submit_clicked = 0.0
    reward_submit_result_checked = 0.0
    penalty_repeated_same_input_fill = 0.0
    penalty_email_repeated_password_pending = 0.0
    penalty_submit_missing = 0.0
    penalty_login_flow_incomplete_early_stop = 0.0
    penalty_targetless_action_success = 0.0
    penalty_inspect_dom_failure_completed = 0.0
    signal_counts = collect_multi_signal_counts(before_observation, after_observation, action, anomalies)

    valid_anomalies: List[Dict[str, Any]] = []
    seen_finding_signatures = set(str(item) for item in (history.get("seen_anomaly_keys", set()) or set()))
    current_finding_signatures: set[str] = set()
    new_anomaly_confidences: List[float] = []
    new_verified_finding_rewards: List[float] = []
    new_verified_evidence_rewards: List[float] = []
    reproduction_rewards: List[float] = []
    for anomaly in anomalies or []:
        if not isinstance(anomaly, Mapping):
            continue
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        anomaly_type = str(anomaly.get("type") or "")
        if anomaly.get("human_review_status") == "likely_false_positive":
            continue
        if anomaly_type == "low-visibility-interaction":
            continue
        if anomaly_type == "action-error" and not _anomaly_error_message(anomaly):
            continue

        anomaly_dict = dict(anomaly)
        signature = str(anomaly_dict.get("signature") or finding_signature(anomaly_dict))
        anomaly_dict["signature"] = signature
        valid_anomalies.append(anomaly_dict)
        current_finding_signatures.add(signature)
        if signature not in seen_finding_signatures:
            if confidence >= 0.5:
                new_anomaly_confidences.append(confidence)
            verified_reward = _verified_finding_reward(anomaly_dict)
            if verified_reward > 0.0:
                new_verified_finding_rewards.append(verified_reward)
                new_verified_evidence_rewards.append(_evidence_richness_reward(anomaly_dict))
        elif anomaly_dict.get("reproducible") or anomaly_dict.get("reproduced"):
            reproduction_rewards.append(_reproduction_reward(anomaly_dict))
    if new_anomaly_confidences:
        reward_new_anomaly = 0.0
    if new_verified_finding_rewards:
        reward_verified_finding = max(new_verified_finding_rewards)
        reward_finding_evidence = min(1.0, max(new_verified_evidence_rewards or [0.0]))
    if reproduction_rewards:
        reward_finding_reproduction = min(2.0, max(reproduction_rewards))

    matched_ids: set[str] = set()
    reward_known_bug_match = 0.0

    current_types = {str(item.get("type") or "") for item in valid_anomalies if item.get("type")}
    seen_types = {
        str(item).split(":", 1)[0]
        for item in history.get("seen_anomaly_keys", set()) or set()
        if str(item)
    }
    new_types = {item for item in current_types if item and item not in seen_types}
    reward_type_diversity = 0.0
    reward_finding_coverage = min(0.3, 0.1 * len({sig.split("|", 1)[0] for sig in current_finding_signatures if sig not in seen_finding_signatures}))

    state_changed = _url(before_observation) != _url(after_observation) or before_signature != after_signature
    if _url(before_observation) != _url(after_observation):
        reward_state_change += 0.2
    if before_signature != after_signature:
        reward_state_change += 0.1
    if _is_new_interactive_click(before_observation, action, history):
        reward_state_change += 0.05

    action_counts = history.get("action_type_counts", {})
    current_action_type_count = int(action_counts.get(action_type, 0) or 0) if isinstance(action_counts, Mapping) else 0
    if action_type == "click_element" and current_action_type_count == 0:
        reward_first_click_element = 0.5
    if action_type and current_action_type_count == 0:
        reward_new_action_type = 0.2
    target_signature = _target_signature(before_observation, action, clicked_candidate)
    seen_targets = history.get("target_signatures", set()) or set()
    if target_signature and target_signature not in seen_targets:
        reward_new_target = 0.3
    if action_type == "inspect_dom" and current_action_type_count == 0:
        reward_first_inspect_dom = 0.2
    if action_type == "inspect_network" and current_action_type_count == 0:
        reward_first_inspect_network = 0.2

    login_flow = history.get("login_flow", {}) if isinstance(history, Mapping) else {}
    if not isinstance(login_flow, Mapping):
        login_flow = {}
    login_form_present = bool(login_flow.get("has_login_form")) or _has_login_form_candidates(before_observation)
    candidate_key = _candidate_key(clicked_candidate) if clicked_candidate else ""
    if action_type == "fill_input":
        if clicked_candidate is None:
            if bool(action.get("action_success")) or bool(action.get("filled")):
                penalty_targetless_action_success -= 0.5
        else:
            filled_keys = history.get("filled_input_element_keys", set()) or set()
            if candidate_key and candidate_key in filled_keys:
                penalty_repeated_same_input_fill -= 0.2
            if _is_password_candidate(clicked_candidate) and not bool(login_flow.get("password_input_filled")):
                reward_password_input_filled = 0.1
            elif _is_email_text_candidate(clicked_candidate) and not bool(login_flow.get("email_input_filled")):
                reward_email_input_filled = 0.1
            if (
                _is_email_text_candidate(clicked_candidate)
                and bool(login_flow.get("email_input_filled"))
                and bool(login_flow.get("has_password_input"))
                and not bool(login_flow.get("password_input_filled"))
            ):
                penalty_email_repeated_password_pending -= 0.2
    if action_type in {"click_element", "click_submit"} and clicked_candidate is None:
        if bool(action.get("action_success")) or bool(action.get("clicked")):
            penalty_targetless_action_success -= 0.5
    if action_type == "click_submit":
        if clicked_candidate is not None and _is_submit_candidate(clicked_candidate) and not bool(login_flow.get("submit_clicked")):
            reward_submit_clicked = 0.2
        elif clicked_candidate is None and (bool(action.get("action_success")) or bool(action.get("clicked"))):
            penalty_targetless_action_success -= 0.5
    submit_checked_before = bool(login_flow.get("submit_result_checked"))
    submit_clicked_before = bool(login_flow.get("submit_clicked"))
    if submit_clicked_before and not submit_checked_before:
        if action_type in {"inspect_network", "inspect_console", "inspect_dom", "check_login_state"} or state_changed:
            reward_submit_result_checked = 0.2
    if action_type == "finish_episode" and login_form_present and not bool(login_flow.get("login_flow_completed")):
        penalty_login_flow_incomplete_early_stop -= 1.0
    if (
        login_form_present
        and bool(login_flow.get("has_submit_button"))
        and not bool(login_flow.get("submit_clicked"))
        and action_type in {"finish_episode", "noop"}
    ):
        penalty_submit_missing -= 0.2
    if action_type == "inspect_dom" and bool(action.get("failed")) and not str(action.get("failure_reason") or ""):
        penalty_inspect_dom_failure_completed -= 0.5

    is_functional_action = _is_functional_action(action_type)
    is_debug_meta_action = _is_debug_meta_action(action_type)
    functional_action_count = int(history.get("functional_action_count", 0) or 0)
    functional_counts = history.get("functional_action_type_counts", {})
    if not isinstance(functional_counts, Mapping):
        functional_counts = {}
    seen_functional_targets = history.get("functional_target_signatures", set()) or set()
    signal_delta_count = _signal_delta_count(signal_counts)
    network_delta_count = _network_delta_count(signal_counts)
    if is_functional_action:
        if functional_action_count == 0:
            reward_first_functional_action = 0.6
        if int(functional_counts.get(action_type, 0) or 0) == 0:
            reward_new_functional_action_type = 0.25
        if target_signature and target_signature not in seen_functional_targets:
            reward_new_functional_target = 0.35
        if state_changed:
            reward_functional_action_caused_state_change = 0.35
        if signal_delta_count > 0:
            reward_functional_action_caused_signal_delta = min(1.0, 0.25 * signal_delta_count)
        if network_delta_count > 0:
            reward_functional_action_caused_network_request = min(0.6, 0.2 * network_delta_count)
    elif is_debug_meta_action:
        if current_action_type_count >= 1:
            penalty_debug_meta_repeat -= 0.4
            if action_type in {"inspect_console", "inspect_network", "inspect_dom"}:
                penalty_repeated_meta_action -= 0.5
        if action_type == "click_retry_button" and current_action_type_count >= 1:
            penalty_retry_button_repeat -= 0.6
        if functional_action_count == 0 and int(history.get("step_index", 0) or 0) >= 2:
            penalty_no_functional_action_episode -= 0.4
    elif functional_action_count == 0 and int(history.get("step_index", 0) or 0) >= 2:
        penalty_no_functional_action_episode -= 0.4

    action_key = _action_key(action)
    if action_key and action_key == history.get("last_action_key"):
        penalty_repeat_action -= 0.1
    if action_type == "noop" and history.get("last_action_type") == "noop":
        penalty_repeat_action -= 0.05
    if isinstance(action_counts, Mapping) and current_action_type_count >= 2:
        penalty_repeat_action -= 0.1
        penalty_same_action_type_repeat -= 0.5
    if action_type == "click_element" and history.get("last_action_type") == "click_element":
        consecutive = int(history.get("consecutive_action_count", 0) or 0)
        if consecutive >= 4:
            penalty_repeat_action -= 0.2

    signature = _action_signature(before_observation, action, clicked_candidate)
    signature_counts = history.get("action_signature_counts", {})
    if signature and isinstance(signature_counts, Mapping):
        legacy_signature = _legacy_action_signature(before_observation, action, clicked_candidate)
        if int(signature_counts.get(signature, 0) or 0) >= 1 or int(signature_counts.get(legacy_signature, 0) or 0) >= 1:
            penalty_same_action_signature_repeat -= 0.7
    if action_type == "open_detail_panel" and current_action_type_count >= 2:
        penalty_open_detail_panel_repeat -= 0.5
    elif action_type == "open_detail_panel" and current_action_type_count >= 1:
        penalty_open_detail_panel_repeat -= 0.3

    if action_type == "click_element" and clicked_candidate:
        click_counts = history.get("click_target_counts", history.get("purchase_click_counts", {}))
        click_key = _candidate_key(clicked_candidate)
        if isinstance(click_counts, Mapping) and int(click_counts.get(click_key, 0) or 0) >= 1:
            penalty_same_target_click -= 0.2
        if _safe_visibility(clicked_candidate) <= 0.0:
            penalty_invalid_action -= 0.2

    if action.get("invalid") or action.get("failed"):
        penalty_invalid_action -= 0.2
    if signal_counts["api_timeout_count"]:
        penalty_timeout -= 0.2
    if int(after_observation.get("history", {}).get("no_change_steps", 0) or 0) >= 2:
        penalty_no_effect -= 0.3
    made_new_signal = state_changed or bool(new_types) or bool(matched_ids) or _signal_delta_count(signal_counts) > 0
    if not made_new_signal:
        penalty_no_effect -= 0.3
        if action_type == "open_detail_panel":
            penalty_no_effect_open_detail -= 0.5

    repeated_anomaly_count = _repeated_anomaly_count(valid_anomalies, history)
    if repeated_anomaly_count:
        penalty_no_effect -= min(0.25, 0.05 * repeated_anomaly_count)

    signal_rewards = _signal_rewards(signal_counts, valid_anomalies)
    raw_signal_reward_total = sum(signal_rewards.values())
    # Raw signal/category rewards are metrics only. Rewarding each signal directly
    # can teach PPO to farm noisy console/network/UI symptoms rather than find
    # distinct verified defects.
    signal_reward_total = sum(signal_rewards.values())
    signal_reward_total = 0.0
    inspect_action_reward_total = reward_first_inspect_dom + reward_first_inspect_network
    functional_action_signal_reward_total = signal_reward_total if is_functional_action and signal_delta_count > 0 else 0.0
    reward_functional_action_total = (
        reward_first_functional_action
        + reward_new_functional_action_type
        + reward_new_functional_target
        + reward_functional_action_caused_state_change
        + reward_functional_action_caused_signal_delta
        + reward_functional_action_caused_network_request
    )
    penalty_debug_meta_total = (
        penalty_debug_meta_repeat
        + penalty_repeated_meta_action
        + penalty_retry_button_repeat
        + penalty_open_detail_panel_repeat
        + penalty_no_functional_action_episode
    )
    login_form_coverage_reward_total = (
        reward_email_input_filled
        + reward_password_input_filled
        + reward_submit_clicked
        + reward_submit_result_checked
    )
    login_flow_penalty_total = (
        penalty_repeated_same_input_fill
        + penalty_email_repeated_password_pending
        + penalty_submit_missing
        + penalty_login_flow_incomplete_early_stop
        + penalty_targetless_action_success
        + penalty_inspect_dom_failure_completed
    )
    exploration_reward_total = (
        reward_state_change
        + reward_first_click_element
        + reward_new_action_type
        + reward_new_target
        + reward_first_inspect_dom
        + reward_first_inspect_network
    )
    ui_dom_signal_reward_total = sum(
        signal_rewards[key]
        for key in (
            "reward_button_no_response_detected",
            "reward_form_no_feedback_detected",
            "reward_layout_overlap_detected",
            "reward_layout_overflow_detected",
            "reward_duplicated_rendering_detected",
            "reward_empty_state_rendering_detected",
        )
    )
    console_runtime_signal_reward_total = sum(
        signal_rewards[key]
        for key in (
            "reward_console_error_detected",
            "reward_console_warning_detected",
            "reward_runtime_exception_detected",
            "reward_page_error_detected",
            "reward_unhandled_rejection_detected",
        )
    )
    network_api_signal_reward_total = sum(
        signal_rewards[key]
        for key in (
            "reward_network_request_failed",
            "reward_api_4xx_detected",
            "reward_api_5xx_detected",
            "reward_api_timeout_detected",
            "reward_cors_error_detected",
            "reward_static_asset_failure_detected",
        )
    )
    cross_layer_signal_reward_total = sum(
        signal_rewards[key]
        for key in (
            "reward_api_ui_mismatch_detected",
            "reward_api_success_ui_failure_detected",
            "reward_api_failure_ui_success_detected",
            "reward_stale_data_rendering_detected",
        )
    )
    security_signal_reward_total = sum(
        signal_rewards[key]
        for key in (
            "reward_auth_permission_anomaly_detected",
            "reward_sensitive_data_exposure_signal_detected",
            "reward_token_exposure_signal_detected",
        )
    )
    repeated_penalty_total = (
        penalty_repeat_action
        + penalty_same_target_click
        + penalty_same_action_type_repeat
        + penalty_same_action_signature_repeat
        + penalty_open_detail_panel_repeat
        + penalty_no_effect_open_detail
        + penalty_no_effect
        + penalty_debug_meta_repeat
        + penalty_repeated_meta_action
        + penalty_retry_button_repeat
        + penalty_no_functional_action_episode
    )

    reward_total = (
        reward_new_anomaly
        + reward_known_bug_match
        + reward_type_diversity
        + reward_verified_finding
        + reward_finding_evidence
        + reward_finding_coverage
        + reward_finding_reproduction
        + reward_first_click_element
        + reward_new_action_type
        + reward_new_target
        + reward_first_inspect_dom
        + reward_first_inspect_network
        + reward_functional_action_total
        + login_form_coverage_reward_total
        + reward_state_change
        + penalty_repeat_action
        + penalty_same_target_click
        + penalty_same_action_type_repeat
        + penalty_same_action_signature_repeat
        + penalty_open_detail_panel_repeat
        + penalty_no_effect_open_detail
        + penalty_invalid_action
        + penalty_no_effect
        + penalty_timeout
        + penalty_debug_meta_repeat
        + penalty_repeated_meta_action
        + penalty_retry_button_repeat
        + penalty_no_functional_action_episode
        + login_flow_penalty_total
        + penalty_step_cost
    )
    breakdown = {
        "reward_mode": reward_mode,
        "use_known_bug_reward": use_known_bug_reward,
        "use_known_bug_for_training": bool(site_profile.get("use_known_bug_for_training", False)),
        "training_uses_site_specific_bug_catalog": bool(site_profile.get("training_uses_site_specific_bug_catalog", False)),
        "policy_uses_bug_labels": False,
        "action_type": action_type,
        "target_signature": target_signature,
        "reward_new_anomaly": reward_new_anomaly,
        "reward_known_bug_match": reward_known_bug_match,
        "reward_type_diversity": reward_type_diversity,
        "reward_first_click_element": reward_first_click_element,
        "reward_new_action_type": reward_new_action_type,
        "reward_new_target": reward_new_target,
        "reward_first_inspect_dom": reward_first_inspect_dom,
        "reward_first_inspect_network": reward_first_inspect_network,
        "reward_state_change": reward_state_change,
        "reward_verified_finding": reward_verified_finding,
        "reward_finding_evidence": reward_finding_evidence,
        "reward_finding_coverage": reward_finding_coverage,
        "reward_finding_reproduction": reward_finding_reproduction,
        "penalty_repeat_action": penalty_repeat_action,
        "penalty_same_target_click": penalty_same_target_click,
        "penalty_same_action_type_repeat": penalty_same_action_type_repeat,
        "penalty_same_action_signature_repeat": penalty_same_action_signature_repeat,
        "penalty_open_detail_panel_repeat": penalty_open_detail_panel_repeat,
        "penalty_no_effect_open_detail": penalty_no_effect_open_detail,
        "penalty_invalid_action": penalty_invalid_action,
        "penalty_no_effect": penalty_no_effect,
        "penalty_no_functional_action_episode": penalty_no_functional_action_episode,
        "penalty_debug_meta_repeat": penalty_debug_meta_repeat,
        "penalty_repeated_meta_action": penalty_repeated_meta_action,
        "penalty_retry_button_repeat": penalty_retry_button_repeat,
        "reward_email_input_filled": reward_email_input_filled,
        "reward_password_input_filled": reward_password_input_filled,
        "reward_submit_clicked": reward_submit_clicked,
        "reward_submit_result_checked": reward_submit_result_checked,
        "login_form_coverage_reward_total": login_form_coverage_reward_total,
        "penalty_repeated_same_input_fill": penalty_repeated_same_input_fill,
        "penalty_email_repeated_password_pending": penalty_email_repeated_password_pending,
        "penalty_submit_missing": penalty_submit_missing,
        "penalty_login_flow_incomplete_early_stop": penalty_login_flow_incomplete_early_stop,
        "penalty_targetless_action_success": penalty_targetless_action_success,
        "penalty_inspect_dom_failure_completed": penalty_inspect_dom_failure_completed,
        "login_flow_penalty_total": login_flow_penalty_total,
        "penalty_timeout": penalty_timeout,
        "penalty_step_cost": penalty_step_cost,
        "reward_first_functional_action": reward_first_functional_action,
        "reward_new_functional_action_type": reward_new_functional_action_type,
        "reward_new_functional_target": reward_new_functional_target,
        "reward_functional_action_caused_state_change": reward_functional_action_caused_state_change,
        "reward_functional_action_caused_signal_delta": reward_functional_action_caused_signal_delta,
        "reward_functional_action_caused_network_request": reward_functional_action_caused_network_request,
        "reward_functional_action_total": reward_functional_action_total,
        "penalty_debug_meta_total": penalty_debug_meta_total,
        "action_category": "functional" if is_functional_action else "debug_meta" if is_debug_meta_action else "global",
        "is_functional_action": is_functional_action,
        "is_debug_meta_action": is_debug_meta_action,
        "functional_action_count": 1.0 if is_functional_action else 0.0,
        "debug_meta_action_count": 1.0 if is_debug_meta_action else 0.0,
        "functional_action_signal_delta_count": float(signal_delta_count if is_functional_action else 0),
        "functional_action_network_delta_count": float(network_delta_count if is_functional_action else 0),
        "no_functional_action_episode_count": 1.0 if penalty_no_functional_action_episode < 0.0 else 0.0,
        "debug_meta_repeat_count": 1.0 if penalty_debug_meta_repeat < 0.0 or penalty_repeated_meta_action < 0.0 or penalty_retry_button_repeat < 0.0 or penalty_open_detail_panel_repeat < 0.0 else 0.0,
        "repeated_meta_action_penalty_count": 1.0 if penalty_repeated_meta_action < 0.0 else 0.0,
        "click_retry_button_repeat_penalty_count": 1.0 if penalty_retry_button_repeat < 0.0 else 0.0,
        "reward_total": reward_total,
        "known_bug_reward_total": reward_known_bug_match,
        "signal_reward_total": signal_reward_total,
        "raw_signal_reward_metric_total": raw_signal_reward_total,
        "direct_signal_reward_used_for_policy": False,
        "verified_finding_reward_total": reward_verified_finding,
        "finding_evidence_reward_total": reward_finding_evidence,
        "finding_coverage_reward_total": reward_finding_coverage,
        "finding_reproduction_reward_total": reward_finding_reproduction,
        "signal_delta_reward_total": signal_reward_total,
        "inspect_action_reward_total": inspect_action_reward_total,
        "functional_action_signal_reward_total": functional_action_signal_reward_total,
        "exploration_reward_total": exploration_reward_total,
        "ui_dom_signal_reward_total": ui_dom_signal_reward_total,
        "console_runtime_signal_reward_total": console_runtime_signal_reward_total,
        "network_api_signal_reward_total": network_api_signal_reward_total,
        "cross_layer_signal_reward_total": cross_layer_signal_reward_total,
        "security_signal_reward_total": security_signal_reward_total,
        "repeated_penalty_total": repeated_penalty_total,
        "multi_signal_anomaly_count": sum(1 for value in signal_counts.values() if _positive_numeric(value)),
        "diversity_reward": (
            reward_first_click_element
            + reward_new_action_type
            + reward_new_target
            + reward_first_inspect_dom
            + reward_first_inspect_network
        ),
        "repeated_action_penalty": (
            penalty_repeat_action
            + penalty_same_target_click
            + penalty_same_action_type_repeat
            + penalty_same_action_signature_repeat
            + penalty_open_detail_panel_repeat
            + penalty_no_effect_open_detail
            + login_flow_penalty_total
        ),
        "first_click_reward_count": 1.0 if reward_first_click_element > 0.0 else 0.0,
        "new_action_type_reward_count": 1.0 if reward_new_action_type > 0.0 else 0.0,
        "new_target_reward_count": 1.0 if reward_new_target > 0.0 else 0.0,
        "inspect_reward_count": (
            (1.0 if reward_first_inspect_dom > 0.0 else 0.0)
            + (1.0 if reward_first_inspect_network > 0.0 else 0.0)
        ),
        "open_detail_panel_repeat_penalty_count": 1.0 if penalty_open_detail_panel_repeat < 0.0 else 0.0,
        "no_effect_action_penalty_count": 1.0 if penalty_no_effect < 0.0 or penalty_no_effect_open_detail < 0.0 else 0.0,
        # Backward-compatible aliases for existing CSV/report code.
        "known_bug_reward": reward_known_bug_match,
        "anomaly_reward": reward_new_anomaly,
        "exploration_reward": reward_state_change + reward_new_action_type + reward_new_target,
        "error_signal_reward": reward_type_diversity,
        "repetition_penalty": penalty_repeat_action + penalty_same_target_click + penalty_same_action_type_repeat + penalty_same_action_signature_repeat,
        "penalty": (
            penalty_invalid_action
            + penalty_no_effect
            + penalty_no_effect_open_detail
            + penalty_timeout
            + penalty_debug_meta_repeat
            + penalty_repeated_meta_action
            + penalty_retry_button_repeat
            + penalty_no_functional_action_episode
            + penalty_step_cost
        ),
        "final_reward": reward_total,
    }
    breakdown.update(signal_rewards)
    breakdown.update(signal_counts)
    return reward_total, breakdown


def apply_fallback_reward_policy(
    reward: float,
    reward_breakdown: Mapping[str, Any] | None,
    *,
    fallback_applied: bool,
    fallback_penalty: float = 0.25,
    fallback_reward_cap_enabled: bool = True,
    fallback_reward_cap_value: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Prevent fallback-selected actions from donating uncapped positive reward to PPO."""
    breakdown = dict(reward_breakdown or {})
    if not fallback_applied:
        breakdown.setdefault("fallback_applied", False)
        breakdown.setdefault("fallback_penalty", 0.0)
        breakdown.setdefault("fallback_penalty_total", 0.0)
        breakdown.setdefault("fallback_reward_capped_count", 0.0)
        return float(reward), breakdown

    original_reward = float(reward or 0.0)
    capped_reward = original_reward
    capped_count = 0.0
    if bool(fallback_reward_cap_enabled) and original_reward > float(fallback_reward_cap_value):
        capped_reward = float(fallback_reward_cap_value)
        capped_count = 1.0
    penalty = abs(float(fallback_penalty or 0.0))
    final_reward = capped_reward - penalty
    breakdown.update(
        {
            "fallback_applied": True,
            "fallback_original_reward": original_reward,
            "fallback_reward_after_cap": capped_reward,
            "fallback_reward_cap_enabled": bool(fallback_reward_cap_enabled),
            "fallback_reward_cap_value": float(fallback_reward_cap_value),
            "fallback_reward_capped_count": capped_count,
            "fallback_penalty": -penalty,
            "fallback_penalty_total": -penalty,
            "fallback_adjusted_reward": final_reward,
        }
    )
    return final_reward, breakdown


def fallback_curriculum_penalty(
    base_penalty: float,
    *,
    curriculum_enabled: bool,
    episode_index: int = 0,
    warmup_episodes: int = 1,
) -> float:
    base = abs(float(base_penalty or 0.0))
    if not curriculum_enabled:
        return base
    warmup = max(1, int(warmup_episodes or 1))
    episode = max(0, int(episode_index or 0))
    return base * (1.0 + min(1.0, episode / warmup))


def _signal_rewards(signal_counts: Mapping[str, int], anomalies: List[Mapping[str, Any]]) -> Dict[str, float]:
    anomaly_types = {str(anomaly.get("type") or "") for anomaly in anomalies}
    return {
        "reward_button_no_response_detected": 0.4 if "button-no-response" in anomaly_types else 0.0,
        "reward_form_no_feedback_detected": 0.4 if "form-no-feedback" in anomaly_types else 0.0,
        "reward_layout_overlap_detected": 0.25 if "layout-overlap" in anomaly_types or signal_counts.get("layout_anomaly_count", 0) else 0.0,
        "reward_layout_overflow_detected": 0.25 if "layout-overflow" in anomaly_types else 0.0,
        "reward_duplicated_rendering_detected": 0.25 if "duplicated-rendering" in anomaly_types else 0.0,
        "reward_empty_state_rendering_detected": 0.25 if "empty-state-rendering" in anomaly_types else 0.0,
        "reward_console_error_detected": 0.5 if signal_counts.get("delta_console_error_count", 0) else 0.0,
        "reward_console_warning_detected": 0.1 if signal_counts.get("delta_console_warning_count", 0) else 0.0,
        "reward_runtime_exception_detected": 0.6 if signal_counts.get("delta_runtime_exception_count", 0) else 0.0,
        "reward_page_error_detected": 0.6 if signal_counts.get("delta_page_error_count", 0) else 0.0,
        "reward_unhandled_rejection_detected": 0.6 if signal_counts.get("delta_unhandled_rejection_count", 0) else 0.0,
        "reward_network_request_failed": 0.4 if signal_counts.get("delta_network_request_failed_count", 0) else 0.0,
        "reward_api_4xx_detected": 0.4 if signal_counts.get("delta_api_4xx_count", 0) else 0.0,
        "reward_api_5xx_detected": 0.6 if signal_counts.get("delta_api_5xx_count", 0) else 0.0,
        "reward_api_timeout_detected": 0.5 if signal_counts.get("delta_api_timeout_count", 0) else 0.0,
        "reward_cors_error_detected": 0.4 if signal_counts.get("cors_error_count", 0) else 0.0,
        "reward_static_asset_failure_detected": 0.3 if signal_counts.get("delta_static_asset_failure_count", 0) else 0.0,
        "reward_api_ui_mismatch_detected": 0.7 if signal_counts.get("api_ui_mismatch_count", 0) else 0.0,
        "reward_api_success_ui_failure_detected": 0.7 if signal_counts.get("api_success_ui_failure_count", 0) else 0.0,
        "reward_api_failure_ui_success_detected": 0.7 if signal_counts.get("api_failure_ui_success_count", 0) else 0.0,
        "reward_stale_data_rendering_detected": 0.4 if signal_counts.get("stale_data_rendering_count", 0) else 0.0,
        "reward_auth_permission_anomaly_detected": 0.5 if signal_counts.get("delta_auth_permission_anomaly_count", 0) else 0.0,
        "reward_sensitive_data_exposure_signal_detected": 0.8 if signal_counts.get("delta_sensitive_data_exposure_signal_count", 0) else 0.0,
        "reward_token_exposure_signal_detected": 0.8 if signal_counts.get("delta_token_exposure_signal_count", 0) else 0.0,
    }


def _verified_finding_reward(anomaly: Mapping[str, Any]) -> float:
    if str(anomaly.get("classification") or "") != "verified_browser_signal":
        return 0.0
    if anomaly.get("verified") is False:
        return 0.0
    confidence = max(0.0, min(1.0, float(anomaly.get("confidence", 0.0) or 0.0)))
    severity = str(anomaly.get("severity") or "").lower()
    severity_score = {
        "critical": 10.0,
        "high": 8.0,
        "medium": 5.0,
        "low": 2.0,
    }.get(severity, 0.5)
    return severity_score * confidence


def _evidence_richness_reward(anomaly: Mapping[str, Any]) -> float:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return 0.0
    evidence_keys = {
        "status",
        "url",
        "method",
        "message",
        "count_delta",
        "request_count_delta",
        "matched_backend_error_patterns",
        "action_signature",
        "trace_id",
        "db_engine",
        "db_invariant",
        "missing_required_fields",
        "type_mismatch_fields",
        "source_policies",
    }
    present = sum(1 for key in evidence_keys if evidence.get(key) not in (None, "", [], {}))
    source_policies = anomaly.get("source_policies")
    source_count = len(source_policies) if isinstance(source_policies, list) else 0
    cross_layer_bonus = 0.0
    if source_count >= 4:
        cross_layer_bonus = 0.8
    elif source_count >= 3:
        cross_layer_bonus = 0.5
    elif source_count >= 2:
        cross_layer_bonus = 0.3
    trace_bonus = min(0.5, max(0.0, float(anomaly.get("correlation_confidence", 0.0) or 0.0)) * 0.5)
    return min(1.0, 0.1 * present + cross_layer_bonus + trace_bonus)


def _reproduction_reward(anomaly: Mapping[str, Any]) -> float:
    if str(anomaly.get("classification") or "") != "verified_browser_signal":
        return 0.0
    source_policies = anomaly.get("source_policies")
    independent_evidence = isinstance(source_policies, list) and len(set(str(item) for item in source_policies)) >= 2
    return 2.0 if independent_evidence else 1.0


def _is_functional_action(action_type: str) -> bool:
    return action_type in FUNCTIONAL_ACTION_TYPES


def _is_debug_meta_action(action_type: str) -> bool:
    return action_type in DEBUG_META_ACTION_TYPES


def _signal_delta_count(signal_counts: Mapping[str, Any]) -> int:
    return sum(
        max(0, int(signal_counts.get(key, 0) or 0))
        for key in (
            "delta_console_error_count",
            "delta_console_warning_count",
            "delta_page_error_count",
            "delta_runtime_exception_count",
            "delta_unhandled_rejection_count",
            "delta_network_request_failed_count",
            "delta_api_4xx_count",
            "delta_api_5xx_count",
            "delta_api_timeout_count",
            "delta_static_asset_failure_count",
            "delta_auth_permission_anomaly_count",
            "delta_sensitive_data_exposure_signal_count",
            "delta_token_exposure_signal_count",
        )
    )


def _network_delta_count(signal_counts: Mapping[str, Any]) -> int:
    return sum(
        max(0, int(signal_counts.get(key, 0) or 0))
        for key in (
            "delta_network_request_failed_count",
            "delta_api_4xx_count",
            "delta_api_5xx_count",
            "delta_api_timeout_count",
            "delta_static_asset_failure_count",
        )
    )


def _positive_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return False


def _post_action_matched_ids(anomalies: List[Mapping[str, Any]], known_matches: List[Mapping[str, Any]]) -> set[str]:
    ids = {
        str(match.get("matched_bug_id"))
        for match in known_matches
        if isinstance(match, Mapping) and match.get("matched_bug_id")
    }
    if ids:
        return ids
    return {
        str(anomaly.get("matched_bug_id"))
        for anomaly in anomalies
        if isinstance(anomaly, Mapping) and anomaly.get("matched_bug_id")
    }


def _is_new_interactive_click(
    before_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    history: Mapping[str, Any],
) -> bool:
    candidate = _clicked_candidate(before_observation, action)
    if not candidate:
        return False
    bid = str(candidate.get("bid", ""))
    clicked = history.get("clicked_bids", set())
    return bool(bid and bid not in clicked)


def _clicked_candidate(before_observation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if action.get("action_type") not in {"click_element", "click_submit", "fill_input", "press_enter"}:
        return None
    candidates = before_observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _has_login_form_candidates(observation: Mapping[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if not isinstance(candidates, list):
        return False
    return bool(
        any(isinstance(candidate, Mapping) and _is_email_text_candidate(candidate) for candidate in candidates)
        and any(isinstance(candidate, Mapping) and _is_password_candidate(candidate) for candidate in candidates)
        and any(isinstance(candidate, Mapping) and _is_submit_candidate(candidate) for candidate in candidates)
    )


def _is_password_candidate(candidate: Mapping[str, Any]) -> bool:
    if not bool(candidate.get("fillable")):
        return False
    if bool(candidate.get("is_password")):
        return True
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    return input_type == "password"


def _is_email_text_candidate(candidate: Mapping[str, Any]) -> bool:
    if not bool(candidate.get("fillable")) or _is_password_candidate(candidate):
        return False
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ("name", "text", "placeholder", "aria_label", "title", "id")
    ).lower()
    return bool(
        input_type in {"", "text", "email", "search"}
        or role in {"textbox", "searchbox"}
        or tag == "textarea"
        or any(token in text for token in ("email", "username", "user", "login"))
    )


def _is_submit_candidate(candidate: Mapping[str, Any]) -> bool:
    if bool(candidate.get("is_submit")):
        return True
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    semantic = str(candidate.get("semantic_action_type") or "").lower()
    text = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "aria_label", "title")).lower()
    return bool(input_type == "submit" or semantic in {"submit", "login"} or any(token in text for token in ("submit", "login", "sign in", "로그인")))


def _candidate_key(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    return str(candidate.get("element_key") or candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")


def _target_signature(
    before_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> str:
    action_type = str(action.get("action_type") or "")
    if action_type in {"inspect_dom", "inspect_network", "inspect_console", "inspect_layout", "change_viewport_mobile", "finish_episode"}:
        return action_type
    target = _candidate_key(candidate) if candidate else ""
    if not target:
        target = str(action.get("clicked_bid") or action.get("clicked_text") or action.get("target_text") or "")
    if not target:
        target = str(action.get("candidate_index", 0) or 0)
    return f"{action_type}:{target}"


def _action_signature(
    before_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> str:
    action_type = str(action.get("action_type") or "")
    if action.get("action_signature"):
        return str(action.get("action_signature"))
    element_key = str(action.get("action_element_key") or _candidate_key(candidate) or "")
    if element_key:
        return f"{action_type}::{element_key}"
    candidate_index = int(action.get("candidate_index", 0) or 0)
    return f"{action_type}::candidate-{candidate_index}"


def _legacy_action_signature(
    before_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> str:
    action_type = str(action.get("action_type") or "")
    candidate_index = int(action.get("candidate_index", 0) or 0)
    return f"{action_type}:{candidate_index}:{_target_signature(before_observation, action, candidate)}"


def _repeated_anomaly_count(anomalies: List[Dict[str, Any]], history: Mapping[str, Any]) -> int:
    seen = history.get("seen_anomaly_keys", set())
    if not seen:
        return 0
    count = 0
    for anomaly in anomalies:
        key = f"{anomaly.get('type')}:{_anomaly_target(anomaly)}"
        if key in seen:
            count += 1
    return count


def _anomaly_target(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return ""
    target = evidence.get("target")
    if isinstance(target, Mapping):
        return str(target.get("bid") or "")
    return str(evidence.get("selector") or evidence.get("clicked_bid") or "")


def _url(observation: Mapping[str, Any]) -> str:
    return str(observation.get("page_state", {}).get("url", "") or "")


def _state_signature(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    page_state = observation.get("page_state", {})
    return (
        page_state.get("url"),
        page_state.get("title"),
        page_state.get("page_text_length"),
        page_state.get("dom_node_count"),
        len(observation.get("candidate_elements", []) or []),
        page_state.get("cart_count"),
    )


def _action_key(action: Mapping[str, Any]) -> str:
    return f"{action.get('action_type')}:{action.get('candidate_index', 0)}"


def _safe_visibility(candidate: Mapping[str, Any] | None) -> float:
    candidate = candidate or {}
    if "visibility" in candidate and candidate.get("visibility") is not None:
        try:
            value = float(candidate.get("visibility") or 0.0)
        except (TypeError, ValueError):
            value = 1.0
    elif "visible" in candidate:
        value = 1.0 if bool(candidate.get("visible")) else 0.0
    else:
        value = 1.0
    return max(0.0, min(1.0, value))


def _anomaly_error_message(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return ""
    value = evidence.get("error")
    return str(value).strip() if value is not None else ""
