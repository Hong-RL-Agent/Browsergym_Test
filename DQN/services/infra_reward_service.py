"""Reward shaping for infra anomaly learning inside the BrowserGym PPO loop."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


INFRA_ACTIONS = {
    "inspect_server_health",
    "inspect_port_status",
    "inspect_latency",
    "inspect_server_logs",
    "inspect_runtime_metrics",
}

REWARD_BY_TYPE = {
    "server-port-not-open": 2.0,
    "server-connection-refused": 2.0,
    "server-timeout": 1.5,
    "server-5xx-error": 1.5,
    "server-healthcheck-failed": 1.2,
    "server-log-exception": 1.2,
    "server-latency-high": 0.8,
    "server-process-down": 2.0,
    "server-memory-high": 1.0,
    "server-cpu-high": 1.0,
}


def calculate_infra_reward(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    infra_anomalies: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    action_type = str(action.get("action_type") or "")
    seen = history.get("seen_anomaly_keys", set()) if isinstance(history, Mapping) else set()
    reward = 0.0
    repeat_penalty = 0.0
    breakdown: dict[str, Any] = {
        "reward_infra_anomaly_found": 0.0,
        "reward_server_timeout": 0.0,
        "reward_server_5xx": 0.0,
        "reward_port_closed": 0.0,
        "penalty_false_positive_infra": 0.0,
        "infra_repeat_penalty": 0.0,
        "infra_noop_penalty": 0.0,
    }
    for anomaly in infra_anomalies:
        anomaly_type = str(anomaly.get("type") or "")
        value = float(REWARD_BY_TYPE.get(anomaly_type, 0.5))
        key = f"{anomaly_type}:{_target_key(anomaly)}"
        if key in seen or f"{anomaly_type}:" in seen:
            repeat_penalty -= 0.3
            continue
        reward += value
        breakdown["reward_infra_anomaly_found"] += value
        if anomaly_type == "server-timeout":
            breakdown["reward_server_timeout"] += value
        elif anomaly_type == "server-5xx-error":
            breakdown["reward_server_5xx"] += value
        elif anomaly_type in {"server-port-not-open", "server-connection-refused", "server-process-down"}:
            breakdown["reward_port_closed"] += value

    if action_type == "noop":
        breakdown["infra_noop_penalty"] = -0.2
    elif action_type in INFRA_ACTIONS and history.get("last_action_type") == action_type:
        breakdown["infra_noop_penalty"] = -0.2

    if not infra_anomalies and action_type in INFRA_ACTIONS and _looks_healthy(after_observation):
        breakdown["penalty_false_positive_infra"] = -1.0

    reward += repeat_penalty
    reward += float(breakdown["infra_noop_penalty"])
    reward += float(breakdown["penalty_false_positive_infra"])
    breakdown["infra_repeat_penalty"] = repeat_penalty
    breakdown["infra_reward_total"] = reward
    return reward, breakdown


def _target_key(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if isinstance(evidence, Mapping):
        return str(evidence.get("port") or evidence.get("url") or "")
    return ""


def _looks_healthy(observation: Mapping[str, Any]) -> bool:
    infra = observation.get("infra_signals") if isinstance(observation, Mapping) else {}
    if not isinstance(infra, Mapping):
        return False
    status = _int(infra.get("response_status"))
    return bool(infra.get("port_open")) and bool(infra.get("health_check_ok")) and (status == 0 or status < 400)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
