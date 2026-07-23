"""General service-error risk policy with confidence and observability separated."""
from __future__ import annotations

from typing import Any, Mapping

POLICY_VERSION = "risk-v3-general-service"

SECURITY_TYPES = {
    "xss", "authorization-bypass", "data-exposure", "api-forbidden",
    "api-auth-bypass", "csrf", "ssrf", "sql-injection",
}
NOT_OBSERVABLE_TYPES = {
    "wal-disk-stall", "mvcc-version-chain-overload", "nic-ring-buffer-overrun",
    "db-index-fragmentation", "kernel-error", "hardware-error",
}
CORE_TYPES = {
    "button-no-response", "form-no-feedback", "network-error", "api-ui-mismatch",
    "api-timeout", "api-5xx-error", "server-unavailable", "database-error",
}
DATA_TYPES = {
    "data-loss", "duplicate-transaction", "duplicate-charge", "partial-commit",
    "cart-quantity-mismatch", "api-ui-mismatch", "data-inconsistency",
    "inventory-overcommit", "stale-data", "api-schema-mismatch",
}
VISUAL_TYPES = {
    "layout-overlap", "layout-overflow", "duplicated-rendering",
    "low-visibility-interaction", "visual-regression",
}


def score_anomaly(
    anomaly: Mapping[str, Any],
    reproduction_attempts: int = 1,
    reproduction_successes: int = 1,
) -> dict[str, Any]:
    anomaly_type = str(anomaly.get("type") or "unknown").lower()
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    raw_confidence = _clamp(_float(anomaly.get("confidence"), 0.5))
    attempts = max(1, int(reproduction_attempts or 1))
    successes = max(0, min(attempts, int(reproduction_successes or 0)))
    reproduction_rate = successes / attempts
    evidence_completeness = _evidence_completeness(evidence)
    confidence = _clamp(0.40 * reproduction_rate + 0.40 * evidence_completeness + 0.20 * raw_confidence)

    common = {
        "confidence": round(confidence, 4),
        "reproduction_attempts": attempts,
        "reproduction_successes": successes,
        "evidence_completeness": round(evidence_completeness, 4),
        "policy_version": POLICY_VERSION,
    }

    if anomaly_type in SECURITY_TYPES:
        return {
            **common,
            "score": None,
            "level": "NOT_ASSESSED",
            "assessment_status": "SECURITY_EXCLUDED",
            "component_scores": None,
            "reason_codes": ["SECURITY_POLICY_REQUIRED"],
        }

    if _is_not_observable(anomaly_type, evidence):
        return {
            **common,
            "score": None,
            "level": "NOT_ASSESSED",
            "assessment_status": "NOT_OBSERVABLE",
            "component_scores": None,
            "reason_codes": ["INTERNAL_EVIDENCE_REQUIRED"],
        }

    components = {
        "core_function_impact": _core_function_points(anomaly_type, evidence),
        "data_impact": _data_points(anomaly_type, evidence),
        "impact_scope": _scope_points(evidence),
        "recovery_difficulty": _recovery_points(anomaly_type, evidence),
        "reproduction_frequency": _frequency_points(reproduction_rate, evidence),
    }
    score = min(100, max(0, sum(components.values())))
    status = _assessment_status(reproduction_rate, evidence_completeness)
    reasons = _reason_codes(anomaly_type, components, reproduction_rate, evidence_completeness)
    return {
        **common,
        "score": score,
        "level": _risk_level(score),
        "assessment_status": status,
        "component_scores": components,
        "reason_codes": reasons,
    }


def _core_function_points(anomaly_type: str, evidence: Mapping[str, Any]) -> int:
    if _truthy(evidence, "service_unavailable", "core_transaction_blocked", "submit_failed", "save_failed"):
        return 35
    if anomaly_type in CORE_TYPES:
        return 30
    if anomaly_type in VISUAL_TYPES:
        return 22 if _truthy(evidence, "blocks_interaction", "critical_element_hidden") else 8
    if _truthy(evidence, "workflow_blocked", "action_failed"):
        return 25
    return 12


def _data_points(anomaly_type: str, evidence: Mapping[str, Any]) -> int:
    if anomaly_type in {"data-loss", "duplicate-transaction", "duplicate-charge", "partial-commit"}:
        return 25
    if anomaly_type in DATA_TYPES or _truthy(evidence, "data_mismatch", "wrong_amount", "wrong_quantity"):
        return 20
    if _truthy(evidence, "stale_data", "ordering_mismatch"):
        return 12
    return 0


def _scope_points(evidence: Mapping[str, Any]) -> int:
    scope = str(evidence.get("impact_scope") or evidence.get("scope") or "").lower()
    if scope in {"service", "global", "all-users"}:
        return 15
    if scope in {"feature", "workflow", "multiple-pages"}:
        return 11
    if scope in {"page", "environment"}:
        return 7
    breadth = _clamp(_float(evidence.get("environment_breadth"), 0.0))
    if breadth >= 0.8:
        return 12
    if breadth >= 0.4:
        return 8
    return 4


def _recovery_points(anomaly_type: str, evidence: Mapping[str, Any]) -> int:
    recovery = str(evidence.get("recovery") or "").lower()
    if recovery in {"impossible", "failed", "none"} or _truthy(evidence, "recovery_failed", "no_workaround"):
        return 15
    if recovery in {"alternate-path", "manual"} or _truthy(evidence, "workaround_available"):
        return 8
    if recovery in {"retry", "refresh", "automatic"}:
        return 3
    if anomaly_type in CORE_TYPES:
        return 12
    if anomaly_type in VISUAL_TYPES:
        return 5
    return 7


def _frequency_points(reproduction_rate: float, evidence: Mapping[str, Any]) -> int:
    breadth = _clamp(_float(evidence.get("environment_breadth"), 0.0))
    return round(10 * _clamp(0.75 * reproduction_rate + 0.25 * breadth))


def _evidence_completeness(evidence: Mapping[str, Any]) -> float:
    groups = [
        ("before_url", "after_url", "route_changed"),
        ("selector", "clicked_text", "action_type"),
        ("dom_changed", "page_text_delta", "candidate_delta"),
        ("network_error", "status", "response_status", "api_error_count"),
        ("console_error", "error", "server_log_exception_count"),
    ]
    observed = sum(any(evidence.get(key) not in (None, "", [], {}) for key in group) for group in groups)
    return observed / len(groups)


def _assessment_status(reproduction_rate: float, evidence_completeness: float) -> str:
    if reproduction_rate >= 0.67 and evidence_completeness >= 0.60:
        return "CONFIRMED"
    if reproduction_rate > 0 or evidence_completeness >= 0.40:
        return "SUSPECTED"
    return "NEEDS_REVIEW"


def _is_not_observable(anomaly_type: str, evidence: Mapping[str, Any]) -> bool:
    if anomaly_type not in NOT_OBSERVABLE_TYPES:
        return False
    return not _truthy(evidence, "server_log_available", "db_metrics_available", "infra_metrics_available")


def _reason_codes(
    anomaly_type: str,
    components: Mapping[str, int],
    reproduction_rate: float,
    evidence_completeness: float,
) -> list[str]:
    reasons = []
    if components["core_function_impact"] >= 25:
        reasons.append("CORE_FLOW_IMPACT")
    if components["data_impact"] >= 20:
        reasons.append("DATA_INTEGRITY_IMPACT")
    if components["recovery_difficulty"] >= 12:
        reasons.append("DIFFICULT_TO_RECOVER")
    if reproduction_rate >= 0.8:
        reasons.append("HIGH_REPRODUCTION_RATE")
    if evidence_completeness >= 0.6:
        reasons.append("STRONG_EVIDENCE")
    if anomaly_type in VISUAL_TYPES:
        reasons.append("FRONTEND_IMPACT")
    return reasons or ["LIMITED_IMPACT_OR_EVIDENCE"]


def _risk_level(score: int) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "INFORMATIONAL"


def _truthy(evidence: Mapping[str, Any], *keys: str) -> bool:
    return any(bool(evidence.get(key)) for key in keys)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
