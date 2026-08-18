"""Policy-safe web error finding heuristics.

The functions in this module use only post-action browser-visible evidence:
DOM/page text, console/runtime signals, network entries, and action outcome
metadata. They intentionally do not read bug ids, site groups, target signal
types, or evaluation reference metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping


BACKEND_ERROR_PATTERNS = (
    "sql syntax",
    "sqlstate",
    "database error",
    "db error",
    "constraint violation",
    "duplicate key",
    "foreign key",
    "deadlock",
    "transaction rollback",
    "jdbc",
    "hibernate",
    "nullpointerexception",
    "illegalstateexception",
    "internal server error",
    "stack trace",
    "traceback",
)

TOKEN_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"\b(access|refresh|id)[_-]?token\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
)

SENSITIVE_PATTERNS = (
    re.compile(r"\bpassword\s*[:=]\s*['\"][^'\"]{3,}", re.IGNORECASE),
    re.compile(r"\b(secret|client_secret)\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
)


def detect_web_error_findings(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    findings.extend(_network_findings(before_observation, after_observation, action_info))
    findings.extend(_api_probe_findings(before_observation, after_observation, action_info))
    findings.extend(_runtime_findings(before_observation, after_observation, action_info))
    findings.extend(_backend_text_findings(after_observation, action_info))
    findings.extend(_security_text_findings(after_observation, action_info))
    return _dedupe_findings(findings)


def finding_signature(finding: Mapping[str, Any]) -> str:
    evidence = _mapping(finding.get("evidence"))
    return "|".join(
        str(part).lower()[:160]
        for part in (
            finding.get("type") or "",
            evidence.get("url") or evidence.get("endpoint") or "",
            evidence.get("status") or "",
            evidence.get("message") or evidence.get("error") or "",
            evidence.get("action_signature") or evidence.get("action_type") or "",
        )
    )


def _network_findings(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    runtime = _runtime(after_observation)
    before_runtime = _runtime(before_observation)
    entries = _network_entries(runtime) + _network_entries(action_info)
    auth_expected_success = _bool(action_info.get("auth_expected_success") or runtime.get("auth_expected_success"))
    request_without_auth = _bool(action_info.get("request_without_auth") or runtime.get("request_without_auth"))
    before_count = _int(before_runtime.get("network_request_count"))
    after_count = max(_int(runtime.get("network_request_count")), len(entries))
    findings: List[Dict[str, Any]] = []
    if after_count > before_count:
        findings.append(
            _finding(
                "network-coverage",
                "low",
                0.5,
                {
                    "request_count_delta": after_count - before_count,
                    "action_type": _action_type(action_info),
                    "action_signature": _action_signature(action_info),
                },
            )
        )
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        status = _int(entry.get("status") or entry.get("status_code"))
        url = str(entry.get("url") or entry.get("endpoint") or "")
        method = str(entry.get("method") or "").upper()
        message = str(entry.get("error") or entry.get("failure") or entry.get("status_text") or "")
        if status >= 500:
            findings.append(
                _finding(
                    "api-5xx",
                    "high",
                    0.9,
                    {"status": status, "url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
                )
            )
        elif status in {401, 403} and auth_expected_success:
            findings.append(
                _finding(
                    "auth-permission-anomaly",
                    "medium",
                    0.75,
                    {"status": status, "url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
                )
            )
        elif 200 <= status < 300 and request_without_auth:
            findings.append(
                _finding(
                    "auth-unauthorized-access",
                    "high",
                    0.85,
                    {
                        "status": status,
                        "url": url,
                        "method": method,
                        "message": "unauthenticated API request returned success",
                        "action_signature": _action_signature(action_info),
                    },
                )
            )
        elif status >= 400 and status not in {401, 403}:
            findings.append(
                _finding(
                    "api-4xx",
                    "medium",
                    0.65,
                    {"status": status, "url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
                )
            )
        if _looks_like_timeout(message) or bool(entry.get("timed_out")):
            findings.append(
                _finding(
                    "api-timeout",
                    "medium",
                    0.8,
                    {"url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
                )
            )
        if bool(entry.get("failed")) or message:
            findings.append(
                _finding(
                    "network-error",
                    "medium",
                    0.7,
                    {"url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
                )
            )
    if _int(runtime.get("network_request_failed_count")) > _int(before_runtime.get("network_request_failed_count")):
        findings.append(
            _finding(
                "network-error",
                "medium",
                0.75,
                {"message": "network request failed count increased", "action_signature": _action_signature(action_info)},
            )
        )
    return findings


def _api_probe_findings(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    runtime = _runtime(after_observation)
    merged = dict(runtime)
    merged.update({key: value for key, value in action_info.items() if isinstance(key, str)})
    findings: List[Dict[str, Any]] = []
    status = _int(merged.get("api_probe_status"))
    url = str(merged.get("api_probe_url") or "")
    method = str(merged.get("api_probe_method") or "").upper()
    message = str(merged.get("api_probe_message") or merged.get("api_probe_warning") or merged.get("api_probe_body_summary") or "")
    if status >= 500:
        findings.append(
            _finding(
                "api-5xx",
                "high",
                0.9,
                {"status": status, "url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
            )
        )
    if status == 0 and _bool(merged.get("api_probe_replayed")) and message:
        findings.append(
            _finding(
                "network-error",
                "medium",
                0.7,
                {"url": url, "method": method, "message": message, "action_signature": _action_signature(action_info)},
            )
        )
    if _bool(merged.get("api_ui_match_checked")):
        expected_ui_change = _bool(merged.get("expect_ui_change_after_api"))
        ui_changed = _bool(merged.get("ui_changed_after_api"))
        if expected_ui_change and 200 <= status < 300 and not ui_changed:
            before_state = _mapping(before_observation.get("page_state")) if isinstance(before_observation, Mapping) else {}
            after_state = _mapping(after_observation.get("page_state")) if isinstance(after_observation, Mapping) else {}
            findings.append(
                _finding(
                    "api-ui-mismatch",
                    "medium",
                    0.8,
                    {
                        "status": status,
                        "url": url,
                        "method": method,
                        "message": "successful API probe did not produce observable UI/state change",
                        "before_url": before_state.get("url"),
                        "after_url": after_state.get("url"),
                        "action_signature": _action_signature(action_info),
                    },
                )
            )
    return findings


def _runtime_findings(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    before = _runtime(before_observation)
    after = _runtime(after_observation)
    findings: List[Dict[str, Any]] = []
    for count_key, list_key, finding_type, severity, confidence in (
        ("console_error_count", "console_errors", "console-error", "medium", 0.75),
        ("page_error_count", "page_errors", "runtime-exception", "high", 0.85),
        ("runtime_exception_count", "runtime_exceptions", "runtime-exception", "high", 0.85),
        ("unhandled_rejection_count", "unhandled_rejections", "runtime-exception", "high", 0.8),
    ):
        delta = _int(after.get(count_key)) - _int(before.get(count_key))
        items = _list(after.get(list_key))
        if delta > 0 or items:
            findings.append(
                _finding(
                    finding_type,
                    severity,
                    confidence,
                    {
                        "message": _first_text(items) or f"{count_key} increased",
                        "count_delta": max(1, delta),
                        "action_signature": _action_signature(action_info),
                    },
                )
            )
    return findings


def _backend_text_findings(after_observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # These are single-pass substring matches against arbitrary page text, not a
    # structured signal (status code, thrown exception, etc.), so they classify as
    # exploratory rather than verified_browser_signal -- a lucky/incidental text
    # match must not pay out the same large one-shot reward as a corroborated finding.
    text = _combined_text(after_observation, action_info)
    findings: List[Dict[str, Any]] = []
    matched = [pattern for pattern in BACKEND_ERROR_PATTERNS if pattern in text.lower()]
    if matched:
        finding_type = "database-error" if any("sql" in item or "database" in item or "jdbc" in item or "hibernate" in item for item in matched) else "backend-error"
        findings.append(
            _finding(
                finding_type,
                "high",
                0.85,
                {
                    "message": matched[0],
                    "matched_backend_error_patterns": matched[:5],
                    "action_signature": _action_signature(action_info),
                },
                classification="exploratory_anomaly",
            )
        )
    return findings


def _security_text_findings(after_observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # Same reasoning as _backend_text_findings: plain regex matches on visible text
    # are candidates, not verified signals, until corroborated.
    text = _combined_text(after_observation, action_info)
    findings: List[Dict[str, Any]] = []
    if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
        findings.append(
            _finding(
                "security-token-leak",
                "high",
                0.9,
                {"message": "token-like secret exposed in browser-visible text", "action_signature": _action_signature(action_info)},
                classification="exploratory_anomaly",
            )
        )
    if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
        findings.append(
            _finding(
                "sensitive-data-exposure",
                "high",
                0.85,
                {"message": "sensitive value exposed in browser-visible text", "action_signature": _action_signature(action_info)},
                classification="exploratory_anomaly",
            )
        )
    lowered = text.lower()
    if "cors" in lowered and ("blocked" in lowered or "policy" in lowered):
        findings.append(
            _finding(
                "cors-error",
                "medium",
                0.75,
                {"message": "CORS policy error visible to browser", "action_signature": _action_signature(action_info)},
                classification="exploratory_anomaly",
            )
        )
    return findings


def _finding(
    finding_type: str,
    severity: str,
    confidence: float,
    evidence: Mapping[str, Any],
    *,
    classification: str = "verified_browser_signal",
) -> Dict[str, Any]:
    finding = {
        "type": finding_type,
        "severity": severity,
        "confidence": float(confidence),
        "classification": classification,
        "evidence": dict(evidence),
    }
    finding["signature"] = finding_signature(finding)
    return finding


def _dedupe_findings(findings: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []
    for finding in findings:
        signature = str(finding.get("signature") or finding_signature(finding))
        if signature in seen:
            continue
        seen.add(signature)
        result.append(finding)
    return result


def _runtime(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("runtime_signals") if isinstance(observation, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _network_entries(source: Mapping[str, Any]) -> List[Any]:
    if not isinstance(source, Mapping):
        return []
    value = source.get("network_entries") or source.get("requests") or source.get("responses")
    return value if isinstance(value, list) else []


def _combined_text(after_observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> str:
    runtime = _runtime(after_observation)
    page_state = after_observation.get("page_state") if isinstance(after_observation, Mapping) else {}
    page_state = page_state if isinstance(page_state, Mapping) else {}
    parts: List[str] = [
        str(page_state.get("page_text_sample") or ""),
        str(runtime.get("page_text") or ""),
        str(runtime.get("console_errors") or ""),
        str(runtime.get("network_errors") or ""),
        str(action_info.get("error") or action_info.get("action_error") or ""),
    ]
    for entry in _network_entries(runtime) + _network_entries(action_info):
        if isinstance(entry, Mapping):
            parts.append(str(entry.get("body") or entry.get("text") or entry.get("message") or entry.get("error") or ""))
    return "\n".join(parts)


def _action_type(action_info: Mapping[str, Any]) -> str:
    action = action_info.get("action")
    if isinstance(action, Mapping):
        return str(action.get("action_type") or "")
    return str(action_info.get("action_type") or "")


def _action_signature(action_info: Mapping[str, Any]) -> str:
    action = action_info.get("action")
    if isinstance(action, Mapping):
        return str(action.get("action_signature") or action.get("action_type") or "")
    return str(action_info.get("action_signature") or action_info.get("action_type") or "")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _first_text(items: List[Any]) -> str:
    for item in items:
        if isinstance(item, Mapping):
            for key in ("message", "text", "error"):
                if item.get(key):
                    return str(item.get(key))
        elif item:
            return str(item)
    return ""


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _looks_like_timeout(message: str) -> bool:
    lowered = str(message or "").lower()
    return "timeout" in lowered or "timed out" in lowered or "err_timed_out" in lowered
