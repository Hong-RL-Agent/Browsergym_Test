"""Coverage-first reward shaping for BrowserGym PPO exploration."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from models.observation_v2 import functional_state_fingerprint


PPO_REWARD_CONFIG: Dict[str, float] = {
    "new_state": 0.50,
    "new_url": 0.50,
    "new_target": 0.30,
    "new_action_target": 0.20,
    "new_action_type": 0.10,
    "new_evidence": 0.10,
    "flow_progress": 0.40,
    "candidate_anomaly": 0.30,
    "confirmed_anomaly_first": 1.00,
    "confirmed_anomaly_upgrade": 0.70,
    "repeat_action_target": -0.20,
    "pending_reproduction_repeat": -0.05,
    "repeat_state": -0.15,
    "repeat_action_type": 0.0,
    "no_effect": -0.20,
    "invalid_action": -0.50,
    "timeout": -0.50,
    "early_finish": -0.20,
    "positive_cap": 2.00,
    "negative_floor": -1.00,
}

PPO_REWARD_SCHEMA_VERSION = "rawd_ppo_functional_coverage_reward_v2"
_REWARD_VALUE_KEYS = (
    "new_state", "new_target", "new_action_target", "new_action_type",
    "flow_progress", "new_evidence", "candidate_anomaly", "confirmed_anomaly",
    "repeat_penalty", "invalid_penalty", "timeout_penalty", "no_effect_penalty", "finish_penalty",
)


def calculate_autonomous_reward(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    anomalies: List[Dict[str, Any]],
    known_bugs: Optional[List[Dict[str, Any]]] = None,
    history: Optional[Mapping[str, Any]] = None,
    site_profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    del known_bugs, site_profile
    mutable_history = history if isinstance(history, dict) else {}
    action_type = str(action.get("action_type") or "")
    clicked_candidate = _action_candidate(before_observation, action)
    state_key, _ = functional_state_fingerprint(after_observation)
    before_state_key, _ = functional_state_fingerprint(before_observation)
    # The pre-action state is already observed.  Seed it before novelty
    # evaluation so an action with no transition cannot earn new-state credit.
    mutable_history.setdefault("episode_seen_states", set()).add(before_state_key)
    url_key = _url_path(after_observation)
    target_key = _target_identity(clicked_candidate) if clicked_candidate else ""
    action_target_key = _action_target_key(action_type, target_key, action)
    outcome_identity = _outcome_key(after_observation, action)
    reward_event_id = _reward_event_id(state_key, action_target_key, outcome_identity)
    seen_reward_events = mutable_history.setdefault("episode_reward_event_ids", set())
    duplicate_reward_event = reward_event_id in seen_reward_events
    seen_reward_events.add(reward_event_id)

    components: Dict[str, float] = {
        "new_state": 0.0,
        "new_functional_state": 0.0,
        "new_url": 0.0,
        "new_target": 0.0,
        "new_action_target": 0.0,
        "new_action_type": 0.0,
        "action_type_novelty": 0.0,
        "flow_progress": 0.0,
        "new_flow": 0.0,
        "new_evidence": 0.0,
        "target_novelty": 0.0,
        "candidate_anomaly": 0.0,
        "confirmed_anomaly": 0.0,
        "repeat_penalty": 0.0,
        "invalid_penalty": 0.0,
        "timeout_penalty": 0.0,
        "no_effect_penalty": 0.0,
        "no_effect_deferred": 0.0,
        "unproductive_repeat": 0.0,
        "reward_event_duplicate": 1.0 if duplicate_reward_event else 0.0,
        "finish_penalty": 0.0,
    }

    repeated_state = False
    # Responsive layout changes caused by global actions (viewport/scroll/
    # inspect) are not functional exploration transitions.  Keep the state
    # fingerprint for telemetry/history, but do not let those actions farm
    # functional-state reward.
    functional_action = _is_functional_action(action_type)
    if functional_action and _mark_first_seen(mutable_history, "episode_seen_states", state_key):
        components["new_state"] = PPO_REWARD_CONFIG["new_state"]
        components["new_functional_state"] = components["new_state"]
    else:
        repeated_state = True

    if functional_action and _mark_first_seen(mutable_history, "episode_seen_urls", url_key):
        components["new_url"] = PPO_REWARD_CONFIG["new_url"]

    if target_key and _mark_first_seen(mutable_history, "episode_seen_targets", target_key):
        components["new_target"] = PPO_REWARD_CONFIG["new_target"]
        components["target_novelty"] = components["new_target"]

    if action_target_key and _mark_first_seen(mutable_history, "episode_seen_action_targets", action_target_key):
        components["new_action_target"] = PPO_REWARD_CONFIG["new_action_target"]

    if action_type and _mark_first_seen(mutable_history, "episode_seen_action_types", action_type):
        components["new_action_type"] = PPO_REWARD_CONFIG["new_action_type"]
        components["action_type_novelty"] = components["new_action_type"]

    flow_key = _flow_progress_key(before_observation, after_observation, action, clicked_candidate) if functional_action else ""
    if flow_key and _mark_first_seen(mutable_history, "episode_seen_flows", flow_key):
        components["flow_progress"] = PPO_REWARD_CONFIG["flow_progress"]
        components["new_flow"] = components["flow_progress"]

    if _has_new_evidence(before_observation, after_observation):
        components["new_evidence"] = PPO_REWARD_CONFIG["new_evidence"]

    anomaly_components = _score_anomaly_novelty(anomalies, mutable_history)
    components["candidate_anomaly"] = anomaly_components["candidate_anomaly"]
    components["confirmed_anomaly"] = anomaly_components["confirmed_anomaly"]

    pending_reproduction_repeat = _has_pending_reproduction_for_action(mutable_history, anomalies, action_target_key)
    same_outcome = _same_outcome(mutable_history, before_observation, after_observation, action)
    unproductive_repeat = repeated_state and _is_repeated_action_target(mutable_history, action_target_key) and same_outcome and not _has_new_evidence(before_observation, after_observation)
    if unproductive_repeat and not pending_reproduction_repeat:
        components["repeat_penalty"] += PPO_REWARD_CONFIG["repeat_state"] + PPO_REWARD_CONFIG["repeat_action_target"]
        components["unproductive_repeat"] = 1.0

    if action.get("invalid") or action.get("failed") or action.get("mask_violation"):
        components["invalid_penalty"] = PPO_REWARD_CONFIG["invalid_action"]

    if _action_timeout(action, after_observation) and not _has_rule(anomalies, "R15"):
        components["timeout_penalty"] = PPO_REWARD_CONFIG["timeout"]
    elif _action_timeout(action, after_observation) and _has_rule(anomalies, "R15"):
        components["timeout_penalty"] = PPO_REWARD_CONFIG["timeout"]

    effect_pending = _effect_pending(after_observation)
    if effect_pending:
        components["no_effect_deferred"] = 1.0
    elif _no_effect(before_observation, after_observation, action) and not _has_rule(anomalies, "R01"):
        components["no_effect_penalty"] = PPO_REWARD_CONFIG["no_effect"]

    if action_type == "finish_episode" and _early_finish(mutable_history):
        components["finish_penalty"] = PPO_REWARD_CONFIG["early_finish"]

    positive_total = sum(components[key] for key in _REWARD_VALUE_KEYS if components[key] > 0)
    negative_total = sum(components[key] for key in _REWARD_VALUE_KEYS if components[key] < 0)
    positive_capped = min(PPO_REWARD_CONFIG["positive_cap"], positive_total)
    negative_capped = max(PPO_REWARD_CONFIG["negative_floor"], negative_total)
    total = positive_capped + negative_capped

    components["positive_total"] = positive_total
    components["positive_capped"] = positive_capped
    components["negative_total"] = negative_total
    components["negative_capped"] = negative_capped
    components["total"] = total
    components["final_reward"] = total
    components["gt_reward_used"] = 0.0
    mutable_history["reward_schema_version"] = PPO_REWARD_SCHEMA_VERSION
    mutable_history["last_reward_after_observation"] = dict(after_observation)
    mutable_history["last_reward_action"] = dict(action)
    mutable_history["last_reward_event_id"] = reward_event_id
    mutable_history["last_reward_event"] = {"event_id": reward_event_id, "state_fingerprint": state_key, "flow_id": flow_key, "target_identity": target_key, "evidence_identity": outcome_identity}
    _remember_reward_history(mutable_history, action_target_key, state_key)
    return total, components


def _score_anomaly_novelty(anomalies: List[Mapping[str, Any]], history: Dict[str, Any]) -> Dict[str, float]:
    components = {"candidate_anomaly": 0.0, "confirmed_anomaly": 0.0}
    seen_candidates = history.setdefault("seen_candidate_signatures", set())
    seen_confirmed = history.setdefault("seen_confirmed_signatures", set())
    pending = history.setdefault("pending_reproduction_signatures", set())
    pending_action_targets = history.setdefault("pending_reproduction_action_targets", set())
    for anomaly in anomalies or []:
        signature = str(anomaly.get("signature") or "")
        if not signature:
            continue
        status = str(anomaly.get("status") or "candidate")
        policy = str(anomaly.get("confirmation_policy") or "")
        action_target = _anomaly_action_target_key(anomaly)
        if status == "confirmed":
            if signature in seen_confirmed:
                continue
            seen_confirmed.add(signature)
            if signature in seen_candidates:
                components["confirmed_anomaly"] += PPO_REWARD_CONFIG["confirmed_anomaly_upgrade"]
            else:
                components["confirmed_anomaly"] += PPO_REWARD_CONFIG["confirmed_anomaly_first"]
            pending.discard(signature)
            if action_target:
                pending_action_targets.discard(action_target)
            continue
        if signature not in seen_candidates:
            seen_candidates.add(signature)
            components["candidate_anomaly"] += PPO_REWARD_CONFIG["candidate_anomaly"]
        if policy == "reproduce":
            pending.add(signature)
            if action_target:
                pending_action_targets.add(action_target)
    return components


def _mark_first_seen(history: Dict[str, Any], key_name: str, value: str) -> bool:
    if not value:
        return False
    seen = history.setdefault(key_name, set())
    if value in seen:
        return False
    seen.add(value)
    return True


def _remember_reward_history(history: Dict[str, Any], action_target_key: str, state_key: str) -> None:
    if action_target_key:
        history["last_reward_action_target_key"] = action_target_key
    if state_key:
        history["last_reward_state_key"] = state_key
    history["last_reward_outcome_key"] = _outcome_key(history.get("last_reward_after_observation", {}), history.get("last_reward_action", {}))


def _same_outcome(history: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    current = _outcome_key(after, action)
    previous = str(history.get("last_reward_outcome_key") or "")
    return bool(previous and current == previous)


def _outcome_key(observation: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    page = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    payload = {
        "execution_success": bool(action.get("execution_success", not action.get("failed"))) if isinstance(action, Mapping) else False,
        "network": bool(isinstance(runtime, Mapping) and (runtime.get("network_request") or runtime.get("request_started") or runtime.get("request_completed"))),
        "ui_settled": bool(isinstance(runtime, Mapping) and runtime.get("ui_settled")),
        "url": _url_path(observation),
        "dom_count": page.get("dom_node_count") if isinstance(page, Mapping) else None,
    }
    return str(payload)


def _reward_event_id(state_key: str, action_target_key: str, outcome_key: str) -> str:
    payload = json.dumps({"state": state_key, "action": action_target_key, "outcome": outcome_key}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_repeated_action_target(history: Mapping[str, Any], action_target_key: str) -> bool:
    return bool(action_target_key and history.get("last_reward_action_target_key") == action_target_key)


def _has_pending_reproduction_for_action(
    history: Mapping[str, Any],
    anomalies: List[Mapping[str, Any]],
    action_target_key: str,
) -> bool:
    if not action_target_key:
        return False
    pending_targets = history.get("pending_reproduction_action_targets", set())
    if action_target_key in pending_targets:
        return True
    return any(
        str(item.get("confirmation_policy") or "") == "reproduce"
        and str(item.get("status") or "") == "candidate"
        and _anomaly_action_target_key(item) == action_target_key
        for item in anomalies or []
    )


def _has_rule(anomalies: List[Mapping[str, Any]], rule_id: str) -> bool:
    return any(str(item.get("rule_id") or item.get("evidence", {}).get("rule_id") or "") == rule_id for item in anomalies or [])


def _action_timeout(action: Mapping[str, Any], after_observation: Mapping[str, Any]) -> bool:
    if bool(action.get("timeout") or action.get("timed_out") or action.get("timeout_occurred")):
        return True
    runtime = after_observation.get("runtime_signals", {})
    page_state = after_observation.get("page_state", {})
    return bool(
        isinstance(runtime, Mapping)
        and (runtime.get("timeout") or runtime.get("timed_out") or runtime.get("timeout_occurred"))
        or isinstance(page_state, Mapping)
        and (page_state.get("timeout") or page_state.get("timed_out") or page_state.get("timeout_occurred"))
    )


def _no_effect(before_observation: Mapping[str, Any], after_observation: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    if action.get("invalid") or action.get("failed"):
        return False
    before_state, _ = functional_state_fingerprint(before_observation)
    after_state, _ = functional_state_fingerprint(after_observation)
    return before_state == after_state


def _effect_pending(observation: Mapping[str, Any]) -> bool:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    network = observation.get("network", {}) if isinstance(observation, Mapping) else {}
    temporal = observation.get("temporal", {}) if isinstance(observation, Mapping) else {}
    for source in (runtime, network, temporal):
        if isinstance(source, Mapping) and (source.get("pending_request") or source.get("pending_request_count", 0) or source.get("request_pending") or source.get("ui_settled") is False):
            return True
    return False


def _has_new_evidence(before_observation: Mapping[str, Any], after_observation: Mapping[str, Any]) -> bool:
    for key in ("network_events", "console_events", "runtime_events", "semantic_pairs", "evidence"):
        before = before_observation.get(key, []) if isinstance(before_observation, Mapping) else []
        after = after_observation.get(key, []) if isinstance(after_observation, Mapping) else []
        if isinstance(after, list) and len(after) > (len(before) if isinstance(before, list) else 0):
            return True
    before_runtime = before_observation.get("runtime_signals", {}) if isinstance(before_observation, Mapping) else {}
    after_runtime = after_observation.get("runtime_signals", {}) if isinstance(after_observation, Mapping) else {}
    return isinstance(before_runtime, Mapping) and isinstance(after_runtime, Mapping) and any(
        bool(after_runtime.get(key)) and not bool(before_runtime.get(key))
        for key in ("network_request", "console_error", "runtime_error", "response_parsed")
    )


def _early_finish(history: Mapping[str, Any]) -> bool:
    action_types = history.get("episode_seen_action_types", set())
    targets = history.get("episode_seen_targets", set())
    return len(action_types or set()) <= 1 and len(targets or set()) == 0


def _flow_progress_key(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    clicked_candidate: Mapping[str, Any] | None,
) -> str:
    for source in (action, after_observation.get("runtime_signals", {}), after_observation.get("page_state", {})):
        if not isinstance(source, Mapping):
            continue
        for key in ("flow_progress", "flow_transition", "semantic_transition"):
            value = source.get(key)
            if value:
                return str(value if not isinstance(value, bool) else f"{key}:{_url_path(after_observation)}")
    before_path = _url_path(before_observation)
    after_path = _url_path(after_observation)
    if before_path != after_path and clicked_candidate:
        semantic = _candidate_semantic(clicked_candidate)
        if semantic:
            return f"{semantic}:{before_path}->{after_path}"
    return ""


def _candidate_semantic(candidate: Mapping[str, Any]) -> str:
    text = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "aria_label", "role", "tag")).lower()
    if candidate.get("is_search_related") or "search" in text:
        return "search"
    if candidate.get("is_cart_related") or candidate.get("is_purchase_action") or "cart" in text or "buy" in text:
        return "cart"
    if candidate.get("is_login_related") or "login" in text or "sign in" in text:
        return "login"
    if candidate.get("is_submit_related") or "submit" in text or "save" in text:
        return "submit"
    if "detail" in text or "product" in text:
        return "detail"
    return ""


def _action_candidate(before_observation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if action.get("action_type") not in {"click_element", "fill_input", "select_option", "press_enter"}:
        return None
    candidates = before_observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _is_functional_action(action_type: str) -> bool:
    """Return whether an action can establish functional application progress."""
    return action_type in {"click_element", "fill_input", "select_option", "press_enter"}


def _target_identity(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    for key in ("bid", "entity_id", "product_id", "item_id", "post_id", "comment_id", "id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    href = _url_path_from_value(candidate.get("href"))
    role = str(candidate.get("role") or "")
    tag = str(candidate.get("tag") or "")
    name = _normalize_text(str(candidate.get("name") or candidate.get("aria_label") or ""))
    if href and (role or tag or name):
        return f"fingerprint:{role}:{tag}:{href}:{name}"
    return ""


def _action_target_key(action_type: str, target_key: str, action: Mapping[str, Any]) -> str:
    # Only target-bearing actions earn action-target novelty.  Global actions
    # such as viewport/scroll/inspect must not manufacture a target key from
    # candidate slot 0.
    if target_key:
        return f"{action_type}:{target_key}"
    return ""


def _anomaly_action_target_key(anomaly: Mapping[str, Any]) -> str:
    action_type = str(anomaly.get("action_type") or "")
    target = str(anomaly.get("target_id") or "")
    if action_type and target:
        return f"{action_type}:bid:{target}" if not target.startswith(("bid:", "entity_id:", "fingerprint:")) else f"{action_type}:{target}"
    return ""


def _state_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {})
    if not isinstance(page_state, Mapping):
        page_state = {}
    candidates = observation.get("candidate_elements", []) or []
    candidate_summary = []
    if isinstance(candidates, list):
        for candidate in candidates[:12]:
            if isinstance(candidate, Mapping):
                candidate_summary.append(_target_identity(candidate) or str(candidate.get("role") or candidate.get("tag") or ""))
    return "|".join(
        [
            _url_path(observation),
            str(page_state.get("title") or ""),
            str(page_state.get("dom_node_count") or ""),
            str(page_state.get("page_text_length") or ""),
            ",".join(candidate_summary),
        ]
    )


def _url_path(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {})
    url = page_state.get("url") if isinstance(page_state, Mapping) else ""
    return _url_path_from_value(url)


def _url_path_from_value(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    parsed = urlparse(text)
    return parsed.path or text.split("?", 1)[0] or "/"


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())
