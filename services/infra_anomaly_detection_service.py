"""Rule-based infra anomaly detection for BrowserGym PPO observations."""

from __future__ import annotations

from typing import Any, Mapping


def detect_infra_anomalies(
    observation: Mapping[str, Any],
    step_info: Mapping[str, Any] | None = None,
    latency_threshold_ms: int = 2000,
) -> list[dict[str, Any]]:
    infra = _infra(observation)
    if not infra:
        return []
    base_url = str(_page(observation).get("url") or infra.get("base_url") or "")
    port = infra.get("port", "")
    anomalies: list[dict[str, Any]] = []
    local_infra_available = _local_infra_available(infra)
    if local_infra_available:
        if not bool(infra.get("port_open")):
            anomalies.append(_anomaly("server-port-not-open", "High", 0.95, infra, base_url, port))
        if bool(infra.get("connection_refused")):
            anomalies.append(_anomaly("server-connection-refused", "High", 0.9, infra, base_url, port))
        if bool(infra.get("timeout_occurred")):
            anomalies.append(_anomaly("server-timeout", "High", 0.9, infra, base_url, port))
        if bool(infra.get("port_open")) and not bool(infra.get("health_check_ok")):
            anomalies.append(_anomaly("server-healthcheck-failed", "High", 0.85, infra, base_url, port))
        status = _int(infra.get("response_status"))
        if status >= 500:
            anomalies.append(_anomaly("server-5xx-error", "High", 0.9, infra, base_url, port))
        latency_ms = _float(infra.get("response_latency_ms"))
        if latency_ms >= float(latency_threshold_ms):
            anomalies.append(_anomaly("server-latency-high", "Medium", 0.75, infra, base_url, port))
        if infra.get("process_alive") is False and not bool(infra.get("port_open")):
            anomalies.append(_anomaly("server-process-down", "High", 0.85, infra, base_url, port))
    if _int(infra.get("server_log_exception_count")) > 0:
        anomalies.append(_anomaly("server-log-exception", "High", 0.8, infra, base_url, port))
    if _int(infra.get("server_log_error_count")) > 0:
        anomalies.append(_anomaly("server-log-error", "High", 0.82, infra, base_url, port))
    if _int(infra.get("server_log_timeout_count")) > 0:
        anomalies.append(_anomaly("server-log-timeout", "Medium", 0.75, infra, base_url, port))
    if _int(infra.get("server_log_db_error_count")) > 0:
        anomalies.append(_anomaly("server-log-database-error", "High", 0.86, infra, base_url, port))
    if _int(infra.get("server_log_auth_error_count")) > 0:
        anomalies.append(_anomaly("server-log-auth-error", "Medium", 0.72, infra, base_url, port))
    if bool(infra.get("db_hook_enabled")) and _int(infra.get("db_connection_ok_count")) == 0:
        anomalies.append(_anomaly("database-connection-error", "High", 0.85, infra, base_url, port))
    if _int(infra.get("db_connection_error_count")) > 0:
        anomalies.append(_anomaly("database-hook-error", "High", 0.8, infra, base_url, port))
    if _int(infra.get("db_integrity_error_count")) > 0:
        anomalies.append(_anomaly("database-integrity-failure", "High", 0.9, infra, base_url, port))
    if _float(infra.get("memory_usage_mb")) >= 1024.0:
        anomalies.append(_anomaly("server-memory-high", "Medium", 0.65, infra, base_url, port))
    if _float(infra.get("cpu_usage_percent")) >= 85.0:
        anomalies.append(_anomaly("server-cpu-high", "Medium", 0.65, infra, base_url, port))
    return _dedupe(anomalies)


def _anomaly(kind: str, severity: str, confidence: float, infra: Mapping[str, Any], base_url: str, port: Any) -> dict[str, Any]:
    evidence = {
        "url": base_url,
        "port": port,
        "health_endpoint": infra.get("health_endpoint", ""),
        "health_status_code": infra.get("health_status_code", ""),
        "response_status": infra.get("response_status", ""),
        "response_latency_ms": infra.get("response_latency_ms", ""),
        "timeout_ms": infra.get("timeout_ms", ""),
        "server_error_message": infra.get("server_error_message", ""),
        "server_log_exception_count": infra.get("server_log_exception_count", ""),
        "infra": dict(infra),
    }
    anomaly = {"type": kind, "severity": severity, "confidence": confidence, "evidence": evidence}
    if kind.startswith("server-log") or kind.startswith("database-"):
        anomaly["classification"] = "verified_browser_signal"
        anomaly["signature"] = f"{kind}|{base_url}|{port}|{evidence.get('server_error_message') or evidence.get('infra', {}).get('db_error_message') or ''}"[:400]
    return anomaly


def _dedupe(anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for anomaly in anomalies:
        key = str(anomaly.get("type") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(anomaly)
    return unique


def _infra(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("infra_signals") if isinstance(observation, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _page(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("page_state") if isinstance(observation, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _local_infra_available(infra: Mapping[str, Any]) -> bool:
    message = str(infra.get("server_error_message") or "").lower()
    if "infra observation skipped for non-local host" in message:
        return False
    if infra.get("infra_observation_skipped") or infra.get("non_local_host"):
        return False
    return True


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
