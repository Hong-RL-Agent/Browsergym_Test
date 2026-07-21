"""Evidence oracles for extended BrowserGym policies.

These helpers intentionally return finding candidates from post-action evidence.
They do not inspect bug ids, catalogs, site groups, target signal types, or
ground-truth labels.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping


def detect_extended_policy_findings(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    findings.extend(_db_consistency_oracle(after_observation, action_info))
    findings.extend(_trace_correlation_oracle(after_observation, action_info))
    findings.extend(_api_contract_oracle(before_observation, after_observation, action_info))
    findings.extend(_login_scenario_oracle(before_observation, after_observation, action_info))
    return _dedupe(findings)


def make_finding(
    *,
    category: str,
    finding_type: str,
    severity: str,
    confidence: float,
    evidence: Mapping[str, Any] | Iterable[Any],
    source_policies: Iterable[str],
    affected_resource: str = "",
    correlation_confidence: float = 0.0,
    reproducible: bool = False,
    verified: bool = True,
) -> Dict[str, Any]:
    evidence_map = _evidence_mapping(evidence)
    finding = {
        "detected": True,
        "category": category,
        "finding_type": finding_type,
        "type": finding_type,
        "severity": severity.lower(),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "evidence": evidence_map,
        "affected_resource": affected_resource or _affected_resource(evidence_map),
        "source_policies": sorted({str(item) for item in source_policies if str(item)}),
        "correlation_confidence": max(0.0, min(1.0, float(correlation_confidence or 0.0))),
        "reproducible": bool(reproducible),
        "verified": bool(verified),
        "classification": "verified_browser_signal" if verified else "evidence_candidate",
        "is_duplicate": False,
    }
    finding["signature"] = finding_signature(finding)
    return finding


def finding_signature(finding: Mapping[str, Any]) -> str:
    evidence = _mapping(finding.get("evidence"))
    raw = "|".join(
        str(part).lower()[:160]
        for part in (
            finding.get("finding_type") or finding.get("type") or "",
            evidence.get("endpoint") or evidence.get("url") or evidence.get("page_url") or "",
            evidence.get("action_signature") or "",
            finding.get("affected_resource") or "",
            evidence.get("exception_type") or evidence.get("db_invariant") or evidence.get("auth_relation") or evidence.get("root_cause") or "",
        )
    )
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"finding:{raw}:{digest}"


def _db_consistency_oracle(after_observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Dict[str, Any]]:
    infra = _infra(after_observation)
    merged = _merge(infra, action_info)
    findings: List[Dict[str, Any]] = []
    if _bool(merged.get("db_invariant_violation")) or _int(merged.get("db_integrity_error_count")) > 0:
        findings.append(
            make_finding(
                category="backend_db",
                finding_type="db_invariant_violation",
                severity="high",
                confidence=0.9,
                evidence={
                    "db_engine": merged.get("db_engine"),
                    "db_invariant": merged.get("db_invariant") or "integrity_check",
                    "db_error_message": merged.get("db_error_message"),
                    "action_signature": _action_signature(action_info),
                },
                affected_resource=str(merged.get("db_engine") or "database"),
                source_policies=["relational_db_hook_policy"],
            )
        )
    if _bool(merged.get("partial_commit_detected")) or _bool(merged.get("transaction_partial_commit")):
        findings.append(
            make_finding(
                category="backend_db",
                finding_type="transaction_partial_commit",
                severity="high",
                confidence=0.86,
                evidence={"transaction_status": merged.get("transaction_status"), "action_signature": _action_signature(action_info)},
                affected_resource="transaction",
                source_policies=["relational_db_hook_policy"],
            )
        )
    if _bool(merged.get("rollback_failed")) or _bool(merged.get("rollback_detected")):
        findings.append(
            make_finding(
                category="backend_db",
                finding_type="rollback_failed",
                severity="medium",
                confidence=0.78,
                evidence={"transaction_status": merged.get("transaction_status"), "action_signature": _action_signature(action_info)},
                affected_resource="transaction",
                source_policies=["relational_db_hook_policy"],
            )
        )
    if _bool(merged.get("deadlock_detected")) or _bool(merged.get("lock_timeout_detected")):
        findings.append(
            make_finding(
                category="backend_db",
                finding_type="deadlock_or_lock_timeout",
                severity="medium",
                confidence=0.8,
                evidence={"db_error_message": merged.get("db_error_message"), "action_signature": _action_signature(action_info)},
                affected_resource="database-lock",
                source_policies=["relational_db_hook_policy"],
            )
        )
    if _float(merged.get("stock_delta")) < 0 and _bool(merged.get("negative_stock_or_balance")):
        findings.append(
            make_finding(
                category="backend_db",
                finding_type="negative_stock_or_balance",
                severity="high",
                confidence=0.88,
                evidence={"stock_delta": merged.get("stock_delta"), "balance_delta": merged.get("balance_delta"), "action_signature": _action_signature(action_info)},
                affected_resource="inventory-or-balance",
                source_policies=["relational_db_hook_policy"],
            )
        )
    return findings


def _trace_correlation_oracle(after_observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Dict[str, Any]]:
    runtime = _runtime(after_observation)
    infra = _infra(after_observation)
    merged = _merge(runtime, infra, action_info)
    if not _bool(merged.get("trace_available") or merged.get("trace_collector_available")):
        return []
    confidence = _float(merged.get("correlation_confidence"))
    trace_id = str(merged.get("trace_id") or merged.get("last_trace_id") or merged.get("request_id") or "")
    findings: List[Dict[str, Any]] = []
    if trace_id and (_bool(merged.get("server_exception_in_same_trace")) or _int(merged.get("trace_error_count")) > 0):
        findings.append(
            make_finding(
                category="backend_trace",
                finding_type="server_exception_in_same_trace",
                severity="high",
                confidence=max(0.75, confidence),
                evidence={
                    "trace_id": trace_id,
                    "endpoint": merged.get("related_endpoint") or merged.get("api_probe_url"),
                    "method": merged.get("related_http_method") or merged.get("api_probe_method"),
                    "exception_type": merged.get("exception_type") or "server_exception",
                    "action_signature": _action_signature(action_info),
                },
                affected_resource=str(merged.get("related_endpoint") or merged.get("api_probe_url") or trace_id),
                source_policies=["trace_correlation_policy", "api_probe_policy"],
                correlation_confidence=confidence,
            )
        )
    if trace_id and _bool(merged.get("db_anomaly_in_same_trace") or merged.get("db_query_in_same_trace")):
        findings.append(
            make_finding(
                category="backend_trace",
                finding_type="db_anomaly_in_same_trace",
                severity="high",
                confidence=max(0.78, confidence),
                evidence={"trace_id": trace_id, "db_query_in_same_trace": True, "action_signature": _action_signature(action_info)},
                affected_resource=trace_id,
                source_policies=["trace_correlation_policy", "relational_db_hook_policy"],
                correlation_confidence=confidence,
            )
        )
    return findings


def _api_contract_oracle(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    runtime = _runtime(after_observation)
    merged = _merge(runtime, action_info)
    if not _bool(merged.get("api_contract_available") or merged.get("schema_available")):
        return []
    source_policies = ["api_contract_policy"]
    findings: List[Dict[str, Any]] = []
    missing = _list(merged.get("missing_required_fields"))
    mismatch = _list(merged.get("type_mismatch_fields"))
    unexpected = _list(merged.get("unexpected_fields"))
    contract_violation_count = _int(merged.get("contract_violation_count")) + len(missing) + len(mismatch)
    verified = _bool(merged.get("ui_contract_mismatch") or merged.get("api_ui_match_checked"))
    classification_confidence = 0.78 if verified else 0.45
    if contract_violation_count > 0 or missing or mismatch:
        findings.append(
            make_finding(
                category="api_contract",
                finding_type="api_schema_violation",
                severity="medium" if verified else "low",
                confidence=classification_confidence,
                evidence={
                    "endpoint": merged.get("api_probe_url") or merged.get("endpoint"),
                    "method": merged.get("api_probe_method") or merged.get("method"),
                    "missing_required_fields": missing,
                    "type_mismatch_fields": mismatch,
                    "unexpected_fields": unexpected,
                    "contract_violation_count": contract_violation_count,
                    "action_signature": _action_signature(action_info),
                },
                affected_resource=str(merged.get("api_probe_url") or merged.get("endpoint") or ""),
                source_policies=source_policies,
                verified=verified,
            )
        )
    if _bool(merged.get("status_code_contract_violation")):
        findings.append(
            make_finding(
                category="api_contract",
                finding_type="status_code_contract_violation",
                severity="medium",
                confidence=0.75,
                evidence={"status": merged.get("api_probe_status"), "endpoint": merged.get("api_probe_url"), "action_signature": _action_signature(action_info)},
                affected_resource=str(merged.get("api_probe_url") or ""),
                source_policies=source_policies,
            )
        )
    if _bool(merged.get("ui_contract_mismatch")):
        findings.append(
            make_finding(
                category="api_contract",
                finding_type="api_contract_ui_mismatch",
                severity="medium",
                confidence=0.82,
                evidence={"endpoint": merged.get("api_probe_url"), "action_signature": _action_signature(action_info)},
                affected_resource=str(merged.get("api_probe_url") or ""),
                source_policies=source_policies + ["api_probe_policy"],
            )
        )
    return findings


def _login_scenario_oracle(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    runtime = _runtime(after_observation)
    merged = _merge(runtime, action_info)
    source_policies = ["login_scenario_policy"]
    findings: List[Dict[str, Any]] = []
    if _bool(merged.get("login_redirect_mismatch")):
        findings.append(
            make_finding(
                category="auth_security",
                finding_type="login_redirect_mismatch",
                severity="medium",
                confidence=0.76,
                evidence={"post_login_redirect_url": merged.get("post_login_url"), "expected_redirect_url": merged.get("expected_redirect_url"), "action_signature": _action_signature(action_info)},
                affected_resource=str(merged.get("post_login_url") or ""),
                source_policies=source_policies,
            )
        )
    if _bool(merged.get("logout_failed")) or _bool(merged.get("protected_page_access_after_logout")):
        findings.append(
            make_finding(
                category="auth_security",
                finding_type="protected_page_access_after_logout",
                severity="high",
                confidence=0.86,
                evidence={"page_url": merged.get("post_logout_url") or merged.get("post_login_url"), "auth_relation": "logout_access_block", "action_signature": _action_signature(action_info)},
                affected_resource=str(merged.get("post_logout_url") or merged.get("post_login_url") or ""),
                source_policies=source_policies,
            )
        )
    if _bool(merged.get("role_based_access_violation")) or _bool(merged.get("unauthorized_authenticated_access")):
        findings.append(
            make_finding(
                category="auth_security",
                finding_type="role_based_access_violation",
                severity="high",
                confidence=0.88,
                evidence={"current_role": merged.get("current_role") or merged.get("user_role"), "resource": merged.get("protected_resource"), "auth_relation": "role_resource", "action_signature": _action_signature(action_info)},
                affected_resource=str(merged.get("protected_resource") or ""),
                source_policies=source_policies,
            )
        )
    if _bool(merged.get("sensitive_data_visible_after_switch")):
        findings.append(
            make_finding(
                category="auth_security",
                finding_type="sensitive_data_visible_after_role_switch",
                severity="high",
                confidence=0.9,
                evidence={"current_role": merged.get("current_role") or merged.get("user_role"), "action_signature": _action_signature(action_info)},
                affected_resource=str(merged.get("protected_resource") or "sensitive-data"),
                source_policies=source_policies,
            )
        )
    return findings


def _dedupe(findings: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []
    for finding in findings:
        signature = str(finding.get("signature") or finding_signature(finding))
        if signature in seen:
            duplicate = dict(finding)
            duplicate["is_duplicate"] = True
            continue
        seen.add(signature)
        result.append(finding)
    return result


def _runtime(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("runtime_signals") if isinstance(observation, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _infra(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("infra_signals") if isinstance(observation, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _merge(*items: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        if isinstance(item, Mapping):
            merged.update(item)
    return merged


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _evidence_mapping(evidence: Mapping[str, Any] | Iterable[Any]) -> Dict[str, Any]:
    if isinstance(evidence, Mapping):
        return dict(evidence)
    return {"items": [str(item) for item in evidence]}


def _affected_resource(evidence: Mapping[str, Any]) -> str:
    return str(evidence.get("endpoint") or evidence.get("url") or evidence.get("page_url") or evidence.get("resource") or "")


def _action_signature(action_info: Mapping[str, Any]) -> str:
    action = action_info.get("action")
    if isinstance(action, Mapping):
        return str(action.get("action_signature") or action.get("action_type") or "")
    return str(action_info.get("action_signature") or action_info.get("action_type") or "")


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
