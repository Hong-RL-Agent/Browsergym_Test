"""Opt-in reward shaping for safe, local security exploration."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


SECURITY_ANOMALY_TYPES = {
    "api-forbidden", "api-ui-mismatch", "network-error", "console-error",
    "form-no-feedback", "broken-navigation", "action-error", "timeout-no-feedback",
}


def apply_security_specialty_reward(
    general_reward: float,
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    anomalies: List[Dict[str, Any]],
    known_matches: List[Dict[str, Any]],
    history: Mapping[str, Any],
) -> Tuple[float, Dict[str, float | str | bool]]:
    new_state = float(_state_signature(before_observation) != _state_signature(after_observation))
    security_surface = float(_is_new_security_surface(before_observation, action, history))
    evidence_complete = float(any(_has_complete_evidence(anomaly) for anomaly in anomalies))
    confirmed_ids = {
        str(match.get("matched_bug_id")) for match in known_matches if match.get("matched_bug_id")
    }
    previously_matched = {str(value) for value in history.get("matched_bug_ids", set()) or set()}
    confirmed_reproducible = float(bool(confirmed_ids - previously_matched))
    severity_weight = max((_severity_weight(anomaly) for anomaly in anomalies), default=0.0)
    repeated_action = float(_action_key(action) == str(history.get("last_action_key") or ""))
    no_change_streak = float(int(after_observation.get("history", {}).get("no_change_steps", 0) or 0) >= 2)
    execution_failed = float(bool(action.get("failed") or action.get("invalid")))
    false_positive = float(any(
        anomaly.get("human_review_status") == "likely_false_positive" for anomaly in anomalies
    ))

    raw_security_reward = (
        2.0 * new_state
        + 3.0 * security_surface
        + 6.0 * evidence_complete
        + 8.0 * confirmed_reproducible
        + 2.0 * severity_weight
        - 1.0 * repeated_action
        - 2.0 * no_change_streak
        - 4.0 * execution_failed
        - 8.0 * false_positive
    )
    normalized_security_reward = _clamp(raw_security_reward / 8.0, -1.0, 1.5)
    final_reward = _clamp(0.35 * float(general_reward) + 0.65 * normalized_security_reward, -1.0, 1.5)
    return final_reward, {
        "reward_profile": "security-v1",
        "general_reward": float(general_reward),
        "new_state": new_state,
        "new_security_surface": security_surface,
        "evidence_complete": evidence_complete,
        "confirmed_reproducible_finding": confirmed_reproducible,
        "severity_weight": severity_weight,
        "repeated_action": -repeated_action,
        "no_state_change_streak": -no_change_streak,
        "action_execution_failed": -execution_failed,
        "false_positive": -false_positive,
        "raw_security_reward": raw_security_reward,
        "normalized_security_reward": normalized_security_reward,
        "final_reward": final_reward,
        "reward_breakdown_valid": True,
    }


def _is_new_security_surface(
    observation: Mapping[str, Any], action: Mapping[str, Any], history: Mapping[str, Any]
) -> bool:
    action_type = str(action.get("action_type") or "")
    security_actions = {"inspect_network", "inspect_console", "inspect_dom", "fill_input", "press_enter"}
    if action_type not in security_actions:
        return False
    counts = history.get("action_type_counts", {})
    if isinstance(counts, Mapping) and int(counts.get(action_type, 0) or 0) > 0:
        return False
    if action_type in {"inspect_network", "inspect_console", "inspect_dom"}:
        return True
    candidates = observation.get("candidate_elements", []) or []
    return any(
        isinstance(candidate, Mapping)
        and (candidate.get("is_form_field") or _contains_security_keyword(candidate))
        for candidate in candidates
    )


def _contains_security_keyword(candidate: Mapping[str, Any]) -> bool:
    text = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "role", "type")).lower()
    return any(token in text for token in ("auth", "login", "password", "token", "admin", "permission"))


def _has_complete_evidence(anomaly: Mapping[str, Any]) -> bool:
    if str(anomaly.get("type") or "") not in SECURITY_ANOMALY_TYPES:
        return False
    evidence = anomaly.get("evidence", {})
    return isinstance(evidence, Mapping) and any(value not in (None, "", [], {}) for value in evidence.values())


def _severity_weight(anomaly: Mapping[str, Any]) -> float:
    severity = str(anomaly.get("severity") or "").lower()
    if severity in {"critical", "5"}:
        return 1.0
    if severity in {"high", "4"}:
        return 0.75
    if severity in {"medium", "3"}:
        return 0.5
    if str(anomaly.get("type") or "") in SECURITY_ANOMALY_TYPES:
        return 0.25
    return 0.0


def _state_signature(observation: Mapping[str, Any]) -> tuple[str, str]:
    state = observation.get("page_state", {}) or {}
    return (
        str(state.get("url") or ""),
        str(state.get("state_signature") or state.get("dom_signature") or state.get("title") or ""),
    )


def _action_key(action: Mapping[str, Any]) -> str:
    return f"{action.get('action_type')}:{int(action.get('candidate_index', 0) or 0)}"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
