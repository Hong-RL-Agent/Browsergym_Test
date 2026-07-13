"""Versioned Impact x Likelihood x Confidence risk scoring."""
from __future__ import annotations
from typing import Any, Mapping

POLICY_VERSION = "risk-v2"
SECURITY_TYPES = {"xss", "authorization-bypass", "data-exposure", "api-forbidden"}
CORE_TYPES = {"button-no-response", "form-no-feedback", "network-error", "api-ui-mismatch"}

def score_anomaly(anomaly: Mapping[str, Any], reproduction_attempts: int = 1, reproduction_successes: int = 1) -> dict[str, Any]:
    anomaly_type = str(anomaly.get("type") or "unknown").lower()
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    confidence_raw = _clamp(float(anomaly.get("confidence", 0.5) or 0.5))
    reproduction_rate = _clamp(reproduction_successes / max(1, reproduction_attempts))
    business, user = ((1.0, 0.95) if anomaly_type in CORE_TYPES else (0.55, 0.60))
    security = 1.0 if anomaly_type in SECURITY_TYPES else 0.20
    scope = 0.75 if evidence.get("route_changed") is False else 0.55
    recoverability = 0.85 if anomaly_type in {"network-error", "button-no-response", "form-no-feedback"} else 0.50
    impact = 0.30 * business + 0.25 * user + 0.20 * security + 0.15 * scope + 0.10 * recoverability
    path_frequency = _clamp(float(evidence.get("path_frequency", 0.5) or 0.5))
    environment_breadth = _clamp(float(evidence.get("environment_breadth", 0.5) or 0.5))
    likelihood = 0.35 * reproduction_rate + 0.25 * path_frequency + 0.20 * environment_breadth + 0.20 * (0.9 if anomaly_type in CORE_TYPES else 0.6)
    evidence_fields = sum(1 for value in evidence.values() if value not in (None, "", [], {}))
    evidence_completeness = min(1.0, evidence_fields / 6.0)
    confidence = 0.35 * reproduction_rate + 0.25 * evidence_completeness + 0.40 * confidence_raw
    score = round(100 * impact * likelihood * confidence)
    if anomaly_type in SECURITY_TYPES: score = max(score, 65)
    level = "CRITICAL" if score >= 85 else "HIGH" if score >= 65 else "MEDIUM" if score >= 40 else "LOW" if score >= 20 else "INFORMATIONAL"
    reasons = []
    if anomaly_type in CORE_TYPES: reasons.append("CORE_FLOW_IMPACT")
    if anomaly_type in SECURITY_TYPES: reasons.append("SECURITY_OR_DATA_RISK")
    if reproduction_rate >= 0.8: reasons.append("HIGH_REPRODUCTION_RATE")
    if evidence_completeness >= 0.66: reasons.append("STRONG_EVIDENCE")
    return {"score": score, "level": level, "impact": round(impact, 4), "likelihood": round(likelihood, 4), "confidence": round(confidence, 4), "reproduction_attempts": reproduction_attempts, "reproduction_successes": reproduction_successes, "policy_version": POLICY_VERSION, "reason_codes": reasons or ["LIMITED_EVIDENCE"]}

def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
