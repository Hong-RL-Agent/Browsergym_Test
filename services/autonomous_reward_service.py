"""Autonomous reward shaping for scenario-free BrowserGym PPO."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


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
    anomaly_reward = 0.0
    exploration_reward = 0.0
    error_signal_reward = 0.0
    purchase_action_reward = 0.0
    cart_inspection_reward = 0.0
    known_bug_reward = 0.0
    catalog_reward = 0.0
    openended_reward = 0.0
    repetition_penalty = 0.0
    penalty = 0.0

    action_type = str(action.get("action_type") or "")
    site_id = str(action.get("site_id") or before_observation.get("page_state", {}).get("site_id") or "")
    if not site_id:
        url = str(before_observation.get("page_state", {}).get("url") or "")
        site_id = "site003" if ":9222" in url else "site001" if ":9220" in url else ""
    is_openended_profile = str((site_profile or {}).get("mode") or "") == "openended_anomaly_discovery"
    reward_profile = (
        "site001_profile"
        if site_id == "site001"
        else "openended_profile"
        if is_openended_profile
        else "catalog_profile"
        if site_profile
        else "generic_profile"
    )
    clicked_candidate = _clicked_candidate(before_observation, action)
    is_purchase_click = action_type == "click_element" and bool(
        clicked_candidate and clicked_candidate.get("is_purchase_action")
    )
    is_catalog_click = action_type == "click_element" and bool(
        clicked_candidate and clicked_candidate.get("catalog_bug_id_matches")
    )
    before_state = before_observation.get("page_state", {}) if isinstance(before_observation.get("page_state"), Mapping) else {}
    after_state = after_observation.get("page_state", {}) if isinstance(after_observation.get("page_state"), Mapping) else {}

    for anomaly in anomalies:
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        if anomaly.get("matched_bug_id"):
            continue
        if anomaly.get("human_review_status") == "likely_false_positive":
            penalty -= 0.05
            continue
        anomaly_type = anomaly.get("type")
        if anomaly_type == "low-visibility-interaction":
            penalty -= 0.05
            continue
        if anomaly_type == "action-error" and not _anomaly_error_message(anomaly):
            continue
        if confidence >= 0.8:
            anomaly_reward += 0.25
        elif confidence >= 0.5:
            anomaly_reward += 0.1

        if anomaly_type in {"layout-overlap", "action-error", "broken-navigation"}:
            error_signal_reward += 0.1
        if is_openended_profile:
            if anomaly_type in {
                "button-no-response",
                "form-no-feedback",
                "async-hang",
                "timeout-no-feedback",
                "api-forbidden",
                "cart-quantity-mismatch",
                "sparse-data-rendering",
                "broken-navigation",
                "network-error",
                "console-error",
                "api-ui-mismatch",
            }:
                openended_reward += 0.5
            elif anomaly_type in {"layout-overlap", "layout-overflow"}:
                openended_reward += 0.4
            elif anomaly_type == "duplicated-rendering":
                openended_reward += 0.3
            if confidence >= 0.6:
                openended_reward += 0.3

        if anomaly_type in {"button-no-response", "empty-state-rendering"}:
            catalog_reward += 0.6
        if anomaly_type == "layout-overflow":
            catalog_reward += 0.7
            if anomaly.get("evidence", {}).get("overflow_right") or anomaly.get("evidence", {}).get("overflow_bottom"):
                catalog_reward += 0.2
        if anomaly.get("evidence", {}).get("catalog_bug_id_matches"):
            catalog_reward += 0.2

    if is_catalog_click:
        catalog_counts = history.get("catalog_click_counts", {})
        catalog_key = _candidate_key(clicked_candidate)
        click_count = int(catalog_counts.get(catalog_key, 0) or 0) if isinstance(catalog_counts, Mapping) else 0
        matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
        clicked_bug_ids = {str(item) for item in clicked_candidate.get("catalog_bug_id_matches", []) or []}
        if click_count == 0 and not clicked_bug_ids.intersection(matched_bug_ids):
            catalog_reward += 0.3
        no_route_change = _url(before_observation) == _url(after_observation)
        no_modal_or_form = not _has_modal_or_dialog(after_observation) and not _has_form_signal(after_observation)
        if no_route_change and no_modal_or_form and _state_signature(before_observation) == _state_signature(after_observation):
            catalog_reward += 0.4

    if _has_catalog_candidate(before_observation) and not history.get("catalog_candidate_seen"):
        catalog_reward += 0.1
    if site_id == "site9800" and site_profile:
        profile_bug_types = {
            str(item.get("type") or "")
            for item in (site_profile or {}).get("bugs", []) or []
            if isinstance(item, Mapping)
        }
        if clicked_candidate and clicked_candidate.get("catalog_bug_id_matches"):
            catalog_reward += 0.1
        if action_type == "fill_input":
            catalog_reward += 0.2
        if action_type == "inspect_network" and {"api-forbidden", "api-ui-mismatch"}.intersection(profile_bug_types):
            catalog_reward += 0.3
        if action_type == "inspect_cart" and "cart-quantity-mismatch" in profile_bug_types:
            catalog_reward += 0.3
        if action_type == "inspect_layout":
            site9800_counts = history.get("action_type_counts", {})
            if isinstance(site9800_counts, Mapping) and int(site9800_counts.get("inspect_layout", 0) or 0) >= 3:
                penalty -= 0.2
        action_hints = {str(item) for item in (clicked_candidate or {}).get("action_hints", []) or []}
        normalized_action = {
            "click_element": "click",
            "fill_input": "fill_input",
            "press_enter": "press_enter",
            "inspect_network": "inspect_network",
            "inspect_console": "inspect_console",
            "inspect_cart": "inspect_cart",
            "inspect_dom": "inspect_dom",
        }.get(action_type, action_type)
        if normalized_action in action_hints:
            catalog_reward += 0.3
    if is_openended_profile:
        candidates = before_observation.get("candidate_elements", []) or []
        if isinstance(candidates, list):
            if any(isinstance(candidate, Mapping) and candidate.get("is_interactive") for candidate in candidates):
                openended_reward += 0.05
            if any(isinstance(candidate, Mapping) and candidate.get("openended_keyword_matches") for candidate in candidates):
                openended_reward += 0.1
            if any(isinstance(candidate, Mapping) and candidate.get("is_form_field") for candidate in candidates):
                openended_reward += 0.1
        if action_type == "click_element" and clicked_candidate:
            action_counts = history.get("action_type_counts", {})
            click_count = int(action_counts.get("click_element", 0) or 0) if isinstance(action_counts, Mapping) else 0
            if click_count == 0:
                openended_reward += 0.3
            if clicked_candidate.get("openended_keyword_matches") or clicked_candidate.get("is_interactive"):
                openended_reward += 0.1
            if _url(before_observation) != _url(after_observation) or _state_signature(before_observation) != _state_signature(after_observation):
                openended_reward += 0.4
    if action_type == "inspect_layout" and _has_layout_catalog_candidate(before_observation):
        catalog_reward += 0.2
    if action_type == "inspect_dom" and bool(before_state.get("has_empty_state_text") or after_state.get("has_empty_state_text")):
        catalog_reward += 0.2

    if is_purchase_click and site_id in {"", "site001"}:
        purchase_key = _candidate_key(clicked_candidate)
        purchase_counts = history.get("purchase_click_counts", {})
        if isinstance(purchase_counts, Mapping) and int(purchase_counts.get(purchase_key, 0) or 0) == 0:
            purchase_action_reward += 0.4
        if _cart_count_detected(after_observation):
            purchase_action_reward += 0.2
        before_count = _cart_count(before_observation)
        after_count = _cart_count(after_observation)
        if before_count is not None and after_count is not None and before_count == after_count:
            purchase_action_reward += 1.0
        if history.get("inspected_cart_before_purchase"):
            purchase_action_reward += 0.1

    if action_type == "inspect_cart" and site_id in {"", "site001"} and _cart_count_detected(after_observation):
        cart_inspection_reward += 0.2

    if _url(before_observation) != _url(after_observation):
        exploration_reward += 0.2
    if _state_signature(before_observation) != _state_signature(after_observation):
        exploration_reward += 0.1
    if _is_new_interactive_click(before_observation, action, history):
        exploration_reward += 0.05
    if action_type == "change_viewport_mobile" and not history.get("mobile_viewport_seen"):
        exploration_reward += 0.4
    if action_type == "inspect_layout" and _viewport_type(before_observation) == "mobile":
        action_counts = history.get("action_type_counts", {})
        inspect_count = int(action_counts.get("inspect_layout", 0) or 0) if isinstance(action_counts, Mapping) else 0
        matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
        if inspect_count < 2 and "site001-bug03" not in matched_bug_ids:
            exploration_reward += 0.4
    if _viewport_type(after_observation) == "mobile" and _layout_overlap_count(after_observation) >= 2:
        matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
        exploration_reward += 0.4 if "site001-bug03" in matched_bug_ids else 1.0

    for match in known_bugs or []:
        if not match.get("matched_bug_id"):
            continue
        matched_bug_id = str(match.get("matched_bug_id"))
        if matched_bug_id == "site001-bug01":
            known_bug_reward += 1.3
        elif matched_bug_id == "site001-bug02":
            known_bug_reward += 1.2
        elif matched_bug_id == "site001-bug03":
            known_bug_reward += 1.3
        elif str(match.get("type") or match.get("bug_type") or match.get("matched_type") or "") == "layout-overflow":
            known_bug_reward += 1.2
        else:
            known_bug_reward += 1.0
        if site_id == "site9800":
            known_bug_reward += 0.5

    action_key = _action_key(action)
    action_counts = history.get("action_type_counts", {})
    if action_key and action_key == history.get("last_action_key"):
        repetition_penalty -= 0.1
    if action_type == "noop" and history.get("last_action_type") == "noop":
        repetition_penalty -= 0.05
    if isinstance(action_counts, Mapping) and int(action_counts.get(action_type, 0) or 0) >= 2:
        repetition_penalty -= 0.1
    if action_type == "change_viewport_mobile" and history.get("mobile_viewport_seen"):
        repetition_penalty -= 0.1
    if action_type == "inspect_layout" and isinstance(action_counts, Mapping):
        if int(action_counts.get("inspect_layout", 0) or 0) >= 2:
            repetition_penalty -= 0.05
        if "site001-bug03" in set(history.get("matched_bug_ids", set()) or set()):
            repetition_penalty -= 0.05
    if action_type == "inspect_cart" and isinstance(action_counts, Mapping):
        if int(action_counts.get("inspect_cart", 0) or 0) >= 2:
            repetition_penalty -= 0.1
    if site_id != "site001" and action_type == "inspect_cart":
        penalty -= 0.2
        if history.get("last_action_type") == "inspect_cart":
            penalty -= 0.4
        if isinstance(action_counts, Mapping) and int(action_counts.get("inspect_cart", 0) or 0) >= 2:
            penalty -= 0.5
    if is_openended_profile and isinstance(action_counts, Mapping):
        if action_type == "inspect_dom" and int(action_counts.get("inspect_dom", 0) or 0) >= 2:
            penalty -= 0.2
        if history.get("last_action_type") == action_type and int(action_counts.get(action_type, 0) or 0) >= 2:
            penalty -= 0.2
        if action_type == "inspect_dom" and int(action_counts.get("inspect_dom", 0) or 0) >= 5:
            penalty -= 0.3
        if action_type == "click_element" and clicked_candidate:
            click_counts = history.get("catalog_click_counts", {})
            click_key = _candidate_key(clicked_candidate)
            if isinstance(click_counts, Mapping) and int(click_counts.get(click_key, 0) or 0) >= 1:
                penalty -= 0.3
    if action_type == "inspect_layout" and isinstance(action_counts, Mapping):
        if history.get("last_action_type") == "inspect_layout" and int(action_counts.get("inspect_layout", 0) or 0) >= 4:
            penalty -= 0.1
    if action_type == "finish_episode" and isinstance(action_counts, Mapping) and _has_catalog_candidate(before_observation):
        if int(action_counts.get("click_element", 0) or 0) == 0:
            penalty -= 0.5
    if is_purchase_click:
        purchase_counts = history.get("purchase_click_counts", {})
        purchase_key = _candidate_key(clicked_candidate)
        if isinstance(purchase_counts, Mapping) and int(purchase_counts.get(purchase_key, 0) or 0) >= 2:
            repetition_penalty -= 0.1
    if is_catalog_click:
        catalog_counts = history.get("catalog_click_counts", {})
        catalog_key = _candidate_key(clicked_candidate)
        catalog_click_count = int(catalog_counts.get(catalog_key, 0) or 0) if isinstance(catalog_counts, Mapping) else 0
        if catalog_click_count >= 2:
            repetition_penalty -= 0.2
        if catalog_click_count >= 3:
            repetition_penalty -= 0.5
        matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
        clicked_bug_ids = {str(item) for item in clicked_candidate.get("catalog_bug_id_matches", []) or []}
        if clicked_bug_ids.intersection(matched_bug_ids):
            repetition_penalty -= 0.4
        if isinstance(action_counts, Mapping) and int(action_counts.get("click_element", 0) or 0) >= 20:
            repetition_penalty -= 0.2
        if isinstance(action_counts, Mapping) and int(action_counts.get("click_element", 0) or 0) >= 30:
            repetition_penalty -= 0.3
    if action_type == "click_element" and history.get("last_action_type") == "click_element":
        consecutive = int(history.get("consecutive_action_count", 0) or 0)
        if consecutive >= 4:
            repetition_penalty -= 0.2
    if action_type == "click_element" and clicked_candidate and float(clicked_candidate.get("visibility", 0.0) or 0.0) <= 0.0:
        penalty -= 0.2
    if action.get("invalid") or action.get("failed"):
        penalty -= 0.2
    if int(after_observation.get("history", {}).get("no_change_steps", 0) or 0) >= 2:
        penalty -= 0.05
    unmatched_count = sum(1 for anomaly in anomalies if not anomaly.get("matched_bug_id"))
    if unmatched_count >= 10:
        penalty -= 0.1
    repeated_anomaly_count = _repeated_anomaly_count(anomalies, history)
    if repeated_anomaly_count:
        penalty -= min(0.25, 0.05 * repeated_anomaly_count)

    penalty += repetition_penalty
    final_reward = _clamp(
        anomaly_reward
        + exploration_reward
        + error_signal_reward
        + known_bug_reward
        + catalog_reward
        + openended_reward
        + purchase_action_reward
        + cart_inspection_reward
        + penalty,
        -1.0,
        1.5,
    )
    return final_reward, {
        "reward_profile": reward_profile,
        "purchase_action_reward": purchase_action_reward,
        "cart_inspection_reward": cart_inspection_reward,
        "catalog_reward": catalog_reward,
        "openended_reward": openended_reward,
        "known_bug_reward": known_bug_reward,
        "anomaly_reward": anomaly_reward,
        "exploration_reward": exploration_reward,
        "error_signal_reward": error_signal_reward,
        "repetition_penalty": repetition_penalty,
        "penalty": penalty,
        "final_reward": final_reward,
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
    if action.get("action_type") != "click_element":
        return None
    candidates = before_observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _candidate_key(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    return str(candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")


def _cart_count_detected(observation: Mapping[str, Any]) -> bool:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("cart_count_detected")):
            return True
    return False


def _cart_count(observation: Mapping[str, Any]) -> int | None:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("cart_count_detected")):
            value = section.get("cart_count")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def _has_catalog_candidate(observation: Mapping[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, Mapping) and bool(candidate.get("catalog_bug_id_matches")) for candidate in candidates
    )


def _has_layout_catalog_candidate(observation: Mapping[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, Mapping)
        and (
            bool(candidate.get("is_chart_related"))
            or bool(candidate.get("catalog_bug_id_matches"))
            or bool(candidate.get("catalog_keyword_matches"))
        )
        for candidate in candidates
    )


def _has_modal_or_dialog(observation: Mapping[str, Any]) -> bool:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("has_modal_or_dialog")):
            return True
    return False


def _has_form_signal(observation: Mapping[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        tag = str(candidate.get("tag", "")).lower()
        role = str(candidate.get("role", "")).lower()
        text = " ".join(str(candidate.get(key, "") or "") for key in ("text", "name"))
        if tag in {"form", "input", "textarea", "select"} or role in {"textbox", "combobox"}:
            return True
        if any(token in text for token in ("새 운동", "운동 기록", "운동명")):
            return True
    return False


def _repeated_anomaly_count(anomalies: List[Dict[str, Any]], history: Mapping[str, Any]) -> int:
    seen = history.get("seen_anomaly_keys", set())
    if not seen:
        return 0
    count = 0
    for anomaly in anomalies:
        key = f"{anomaly.get('type')}:{anomaly.get('matched_bug_id') or _anomaly_target(anomaly)}"
        if key in seen:
            count += 1
    return count


def _anomaly_target(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return ""
    target = evidence.get("target")
    if isinstance(target, Mapping):
        return str(target.get("bid") or target.get("data_bug_id") or "")
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


def _viewport_type(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {})
    value = str(page_state.get("viewport_type") or observation.get("runtime_signals", {}).get("viewport_type") or "")
    if value:
        return value
    width = int(page_state.get("viewport_width", 0) or 0)
    return "mobile" if width and width <= 768 else "desktop"


def _layout_overlap_count(observation: Mapping[str, Any]) -> int:
    try:
        return int(observation.get("layout_signals", {}).get("layout_overlap_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _anomaly_error_message(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return ""
    value = evidence.get("error")
    return str(value).strip() if value is not None else ""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
