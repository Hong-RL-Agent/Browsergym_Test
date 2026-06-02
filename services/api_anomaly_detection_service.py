"""Response inspection heuristics for direct API fuzzing."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping


SUMMARY_KEYS = {
    "api-5xx-error": "api_5xx_count",
    "api-4xx-unexpected": "api_4xx_unexpected_count",
    "api-schema-mismatch": "schema_mismatch_count",
    "api-timeout": "timeout_count",
    "api-auth-bypass": "auth_bypass_count",
    "api-latency-regression": "latency_regression_count",
}


def detect_api_anomalies(
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    response: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    expected_status = [int(item) for item in observation.get("expected_status", [200]) or [200]]
    status = response.get("status")
    mutation = str(action.get("mutation") or "")
    if response.get("timeout"):
        anomalies.append(_anomaly("api-timeout", "High", observation, action, response))
        return anomalies
    if status is None:
        anomalies.append(_anomaly("api-response-body-invalid", "Medium", observation, action, response))
        return anomalies
    status = int(status)
    if status >= 500:
        anomalies.append(_anomaly("api-5xx-error", "High", observation, action, response))
    elif status == 403:
        if mutation in {"send_valid_request", "check_response_schema", "check_latency"} and not observation.get("auth_required"):
            anomalies.append(_anomaly("api-forbidden", "Medium", observation, action, response))
    elif 400 <= status <= 499 and status not in expected_status:
        if mutation in {"send_valid_request", "repeat_request", "parallel_request", "check_response_schema", "check_latency"}:
            anomalies.append(_anomaly("api-4xx-unexpected", "Medium", observation, action, response))
    if mutation in {"missing_auth", "invalid_auth", "expired_auth"} and observation.get("auth_required") and 200 <= status < 300:
        anomalies.append(_anomaly("api-auth-bypass", "Critical", observation, action, response))
    if mutation in {
        "missing_required_param",
        "null_param",
        "invalid_type_param",
        "invalid_enum_param",
        "boundary_number_param",
    } and 200 <= status < 300:
        anomalies.append(_anomaly("api-validation-missing", "Medium", observation, action, response))
    if _body_invalid(response):
        anomalies.append(_anomaly("api-response-body-invalid", "Medium", observation, action, response))
    schema_errors = _schema_errors(observation.get("response_schema", {}), response.get("json_body"))
    if schema_errors and 200 <= status < 300:
        anomaly = _anomaly("api-schema-mismatch", "Medium", observation, action, response)
        anomaly["evidence"]["schema_errors"] = schema_errors
        anomalies.append(anomaly)
    latency_threshold_ms = _latency_threshold_ms(observation)
    latency_ms = float(response.get("latency_ms", 0.0) or 0.0)
    if latency_threshold_ms and latency_ms > latency_threshold_ms:
        anomaly = _anomaly("api-latency-regression", "Medium", observation, action, response)
        anomaly["evidence"]["latency_threshold_ms"] = latency_threshold_ms
        anomalies.append(anomaly)
    return anomalies


def summarize_api_anomalies(anomalies: List[Mapping[str, Any]]) -> Dict[str, int]:
    summary = {
        "api_5xx_count": 0,
        "api_4xx_unexpected_count": 0,
        "schema_mismatch_count": 0,
        "timeout_count": 0,
        "auth_bypass_count": 0,
        "latency_regression_count": 0,
        "api_anomaly_count": len(anomalies),
    }
    for anomaly in anomalies:
        key = SUMMARY_KEYS.get(str(anomaly.get("anomaly_type") or ""))
        if key:
            summary[key] += 1
    return summary


def anomaly_key(anomaly: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(anomaly.get("site_id") or ""),
        str(anomaly.get("api_id") or ""),
        str(anomaly.get("mutation") or ""),
        str(anomaly.get("anomaly_type") or ""),
        int(anomaly.get("status") or 0),
    )


def _anomaly(
    anomaly_type: str,
    severity: str,
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    response: Mapping[str, Any],
) -> Dict[str, Any]:
    request = response.get("request", {}) if isinstance(response.get("request"), Mapping) else {}
    return {
        "source": "api",
        "site_id": str(observation.get("site_id") or ""),
        "api_id": str(observation.get("api_id") or ""),
        "method": str(observation.get("method") or request.get("method") or ""),
        "url": str(response.get("url") or action.get("url") or request.get("url") or ""),
        "mutation": str(action.get("mutation") or ""),
        "status": response.get("status"),
        "latency_ms": int(float(response.get("latency_ms", 0.0) or 0.0)),
        "anomaly_type": anomaly_type,
        "type": anomaly_type,
        "severity": severity,
        "confidence": _confidence(anomaly_type),
        "evidence": {
            "request": _safe_request(request),
            "response_status": response.get("status"),
            "response_body_preview": str(response.get("body_preview") or "")[:1000],
            "expected_status": observation.get("expected_status", [200]),
            "exception": response.get("exception", ""),
        },
    }


def _safe_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    headers = dict(request.get("headers", {})) if isinstance(request.get("headers"), Mapping) else {}
    for key in list(headers):
        if key.lower() in {"authorization", "x-api-key", "x-auth-token"}:
            headers[key] = "[redacted]"
    return {
        "method": request.get("method"),
        "url": request.get("url"),
        "query": request.get("query", {}),
        "headers": headers,
        "body": request.get("body", {}),
    }


def _confidence(anomaly_type: str) -> float:
    return {
        "api-5xx-error": 0.95,
        "api-timeout": 0.9,
        "api-auth-bypass": 0.85,
        "api-schema-mismatch": 0.75,
        "api-validation-missing": 0.65,
    }.get(anomaly_type, 0.7)


def _body_invalid(response: Mapping[str, Any]) -> bool:
    content_type = str(response.get("content_type") or "").lower()
    body = str(response.get("body_preview") or "")
    if not body:
        return False
    if "application/json" in content_type and response.get("json_body") is None:
        try:
            json.loads(body)
        except json.JSONDecodeError:
            return True
    return False


def _schema_errors(schema: Any, body: Any) -> List[str]:
    if not isinstance(schema, Mapping) or not schema:
        return []
    if body is None:
        return ["response body is not JSON"]
    if isinstance(body, list):
        if not body:
            return []
        body = body[0]
    if not isinstance(body, Mapping):
        return ["response body is not an object"]
    errors = []
    for key, expected in schema.items():
        if key not in body:
            errors.append(f"missing field: {key}")
            continue
        if not _matches_type(body.get(key), expected):
            errors.append(f"type mismatch: {key}")
    return errors


def _matches_type(value: Any, expected: Any) -> bool:
    expected_type = str(expected.get("type") if isinstance(expected, Mapping) else expected).lower()
    if expected_type in {"string", "str"}:
        return isinstance(value, str)
    if expected_type in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type in {"integer", "int"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type in {"boolean", "bool"}:
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, Mapping)
    return True


def _latency_threshold_ms(observation: Mapping[str, Any]) -> int | None:
    value = observation.get("latency_threshold_ms")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
