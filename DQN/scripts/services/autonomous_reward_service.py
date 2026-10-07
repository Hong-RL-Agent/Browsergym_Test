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
) -> Tuple[float, Dict[str, float]]:
    history = history or {}
    anomaly_reward = 0.0
    error_signal_reward = 0.0
    known_bug_reward = 0.0
    penalty = 0.0
    matched_ids = {match.get("matched_bug_id") for match in known_bugs or [] if match.get("matched_bug_id")}
    unmatched_count = int(history.get("unmatched_anomaly_count", 0) or 0)

    for anomaly in anomalies:
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        matched_bug_id = anomaly.get("matched_bug_id")
        if not matched_bug_id:
            for match in known_bugs or []:
                if match.get("type") == anomaly.get("type"):
                    matched_bug_id = match.get("matched_bug_id")
                    break
        if matched_bug_id:
            continue
        if anomaly.get("type") == "low-visibility-interaction":
            penalty -= 0.05
            continue
        if confidence >= 0.8:
            anomaly_reward += 0.25
        elif confidence >= 0.5:
            anomaly_reward += 0.1

        anomaly_type = anomaly.get("type")
        if anomaly_type == "layout-overlap" and matched_bug_id:
            error_signal_reward += 0.2
        elif anomaly_type == "action-error":
            error_signal_reward += 0.1
        elif anomaly_type == "broken-navigation":
            error_signal_reward += 0.2

    exploration_reward = 0.0
    if _url(before_observation) != _url(after_observation):
        exploration_reward += 0.2
    if _state_signature(before_observation) != _state_signature(after_observation):
        exploration_reward += 0.1
    if _is_new_interactive_click(before_observation, action, history):
        exploration_reward += 0.05

    for match in known_bugs or []:
        if not match.get("matched_bug_id"):
            continue
        confidence = float(match.get("confidence", 0.0) or 0.0)
        known_bug_reward += 1.3 if confidence >= 0.8 else 1.2

    action_key = _action_key(action)
    if action_key and action_key == history.get("last_action_key"):
        penalty -= 0.1
    if action.get("action_type") == "noop" and history.get("last_action_type") == "noop":
        penalty -= 0.05
    if action.get("invalid") or action.get("failed"):
        penalty -= 0.2
    if int(after_observation.get("history", {}).get("no_change_steps", 0) or 0) >= 2:
        penalty -= 0.05
    if unmatched_count >= 10 and not matched_ids:
        penalty -= 0.1

    final_reward = _clamp(
        anomaly_reward + exploration_reward + error_signal_reward + known_bug_reward + penalty,
        -1.0,
        1.5,
    )
    return final_reward, {
        "anomaly_reward": anomaly_reward,
        "exploration_reward": exploration_reward,
        "error_signal_reward": error_signal_reward,
        "known_bug_reward": known_bug_reward,
        "penalty": penalty,
        "final_reward": final_reward,
    }


def _is_new_interactive_click(
    before_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    history: Mapping[str, Any],
) -> bool:
    if action.get("action_type") != "click_element":
        return False
    candidates = before_observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    if not isinstance(candidates, list) or not (0 <= index < len(candidates)):
        return False
    bid = str(candidates[index].get("bid", ""))
    clicked = history.get("clicked_bids", set())
    return bool(bid and bid not in clicked)


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
    )


def _action_key(action: Mapping[str, Any]) -> str:
    return f"{action.get('action_type')}:{action.get('candidate_index', 0)}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
