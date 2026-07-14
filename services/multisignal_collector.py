"""Policy-safe multi-layer signal collection for BrowserGym observations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SIGNAL_COUNT_FIELDS = (
    "console_error_count",
    "console_warning_count",
    "page_error_count",
    "runtime_exception_count",
    "unhandled_rejection_count",
    "network_request_count",
    "network_request_failed_count",
    "api_2xx_count",
    "api_3xx_count",
    "api_4xx_count",
    "api_5xx_count",
    "api_timeout_count",
    "cors_error_count",
    "static_asset_failure_count",
    "api_ui_mismatch_count",
    "api_success_ui_failure_count",
    "api_failure_ui_success_count",
    "stale_data_rendering_count",
    "auth_permission_anomaly_count",
    "sensitive_data_exposure_signal_count",
    "token_exposure_signal_count",
    "security_signal_count",
    "ui_anomaly_count",
    "dom_anomaly_count",
    "layout_anomaly_count",
    "interaction_anomaly_count",
)

DELTA_SIGNAL_FIELDS = (
    "console_error_count",
    "console_warning_count",
    "page_error_count",
    "runtime_exception_count",
    "unhandled_rejection_count",
    "network_request_failed_count",
    "api_4xx_count",
    "api_5xx_count",
    "api_timeout_count",
    "static_asset_failure_count",
    "auth_permission_anomaly_count",
    "sensitive_data_exposure_signal_count",
    "token_exposure_signal_count",
)

LISTENER_STATUS_FIELDS = (
    "signal_collector_enabled",
    "playwright_console_listener_enabled",
    "playwright_pageerror_listener_enabled",
    "playwright_network_listener_enabled",
    "playwright_request_listener_enabled",
    "playwright_response_listener_enabled",
    "playwright_requestfailed_listener_enabled",
    "playwright_listener_warning",
)


def collect_multi_signal_counts(
    before_observation: Mapping[str, Any] | None = None,
    after_observation: Mapping[str, Any] | None = None,
    action: Mapping[str, Any] | None = None,
    anomalies: Sequence[Mapping[str, Any]] | None = None,
    step_info: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    before_observation = before_observation or {}
    after_observation = after_observation or {}
    step_info = step_info or {}
    before_counts = _collect_single_observation_counts(before_observation)
    after_counts = _collect_single_observation_counts(after_observation, anomalies=anomalies, step_info=step_info)
    counts = dict(after_counts)
    after_runtime = _mapping(after_observation.get("runtime_signals"))
    for field in LISTENER_STATUS_FIELDS:
        value = after_runtime.get(field, step_info.get(field))
        if field.endswith("_warning"):
            counts[field] = str(value or "")
        else:
            counts[field] = bool(value)
    for field in DELTA_SIGNAL_FIELDS:
        counts[f"before_{field}"] = int(before_counts.get(field, 0) or 0)
        counts[f"after_{field}"] = int(after_counts.get(field, 0) or 0)
        explicit_delta = after_runtime.get(f"delta_{field}")
        if explicit_delta is not None:
            counts[f"delta_{field}"] = max(0, _int(explicit_delta))
        else:
            counts[f"delta_{field}"] = max(0, counts[f"after_{field}"] - counts[f"before_{field}"])
    counts["network_error_count"] = counts.get("network_request_failed_count", 0)
    counts["before_network_error_count"] = counts.get("before_network_request_failed_count", 0)
    counts["after_network_error_count"] = counts.get("after_network_request_failed_count", 0)
    counts["delta_network_error_count"] = counts.get("delta_network_request_failed_count", 0)
    return counts


def _collect_single_observation_counts(
    observation: Mapping[str, Any],
    *,
    anomalies: Sequence[Mapping[str, Any]] | None = None,
    step_info: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    step_info = step_info or {}
    counts = {field: 0 for field in SIGNAL_COUNT_FIELDS}

    runtime = _merge_mappings(
        _mapping(observation.get("runtime_signals")),
        step_info,
    )
    infra = _merge_mappings(
        _mapping(observation.get("infra_signals")),
        step_info,
    )
    layout = _merge_mappings(_mapping(observation.get("layout_signals")))

    counts["console_error_count"] = _count_value(runtime, "console_error_count", "console_errors")
    counts["console_warning_count"] = _count_value(runtime, "console_warning_count", "console_warnings")
    counts["page_error_count"] = _count_value(runtime, "page_error_count", "page_errors")
    counts["runtime_exception_count"] = _count_value(runtime, "runtime_exception_count", "runtime_exceptions")
    counts["unhandled_rejection_count"] = _count_value(runtime, "unhandled_rejection_count", "unhandled_rejections")

    network_entries = _list_value(runtime.get("network_entries")) + _list_value(step_info.get("network_entries"))
    counts["network_request_count"] = max(_count_value(runtime, "network_request_count", "request_count"), len(network_entries))
    counts["network_request_failed_count"] = _count_value(
        runtime,
        "network_request_failed_count",
        "network_error_count",
        "network_errors",
        "api_error_count",
    )
    status_codes = _status_codes(runtime, step_info, network_entries)
    for status in status_codes:
        if 200 <= status < 300:
            counts["api_2xx_count"] += 1
        elif 300 <= status < 400:
            counts["api_3xx_count"] += 1
        elif 400 <= status < 500:
            counts["api_4xx_count"] += 1
        elif status >= 500:
            counts["api_5xx_count"] += 1
    counts["api_4xx_count"] += _count_value(runtime, "api_4xx_count", "server_4xx_count")
    counts["api_5xx_count"] += _count_value(runtime, "api_5xx_count", "server_5xx_count")
    counts["api_timeout_count"] = _count_value(runtime, "api_timeout_count", "timeout_count") + int(bool(infra.get("timeout_occurred")))
    counts["cors_error_count"] = _count_value(runtime, "cors_error_count", "cors_errors")
    counts["static_asset_failure_count"] = _count_value(runtime, "static_asset_failure_count", "asset_failure_count")

    for anomaly in anomalies or []:
        anomaly_type = str(anomaly.get("type") or "")
        evidence = _mapping(anomaly.get("evidence"))
        if anomaly_type in {"button-no-response", "form-no-feedback"}:
            counts["interaction_anomaly_count"] += 1
        if anomaly_type in {"layout-overlap", "layout-overflow"}:
            counts["layout_anomaly_count"] += 1
        if anomaly_type in {"duplicated-rendering", "empty-state-rendering", "sparse-data-rendering", "stale-data-rendering"}:
            counts["dom_anomaly_count"] += 1
        if anomaly_type in {"api-ui-mismatch", "network-error"}:
            counts["api_ui_mismatch_count"] += 1
        if anomaly_type == "api-success-ui-failure":
            counts["api_success_ui_failure_count"] += 1
        if anomaly_type == "api-failure-ui-success":
            counts["api_failure_ui_success_count"] += 1
        if anomaly_type in {"stale-data-rendering", "sparse-data-rendering"}:
            counts["stale_data_rendering_count"] += 1
        if anomaly_type in {"auth-permission-anomaly", "api-forbidden"} or _int(evidence.get("network_status")) in {401, 403}:
            counts["auth_permission_anomaly_count"] += 1
        if anomaly_type == "sensitive-data-exposure":
            counts["sensitive_data_exposure_signal_count"] += 1
        if anomaly_type == "token-exposure":
            counts["token_exposure_signal_count"] += 1
        if _int(evidence.get("api_403_count")):
            counts["api_4xx_count"] += _int(evidence.get("api_403_count"))

    counts["layout_anomaly_count"] += _count_value(layout, "layout_overlap_count", "layout_overflow_count")
    counts["ui_anomaly_count"] = (
        counts["layout_anomaly_count"]
        + counts["interaction_anomaly_count"]
        + counts["dom_anomaly_count"]
    )
    security_text = " ".join(
        str(runtime.get(key) or "")
        for key in ("page_text", "console_errors", "network_errors")
    ).lower()
    if "token" in security_text or "bearer " in security_text or "authorization" in security_text:
        counts["token_exposure_signal_count"] += 1
    if "password" in security_text or "secret" in security_text:
        counts["sensitive_data_exposure_signal_count"] += 1
    counts["security_signal_count"] = (
        counts["auth_permission_anomaly_count"]
        + counts["sensitive_data_exposure_signal_count"]
        + counts["token_exposure_signal_count"]
    )
    return counts


def _merge_mappings(*items: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if isinstance(item, Mapping):
            merged.update(item)
    return merged


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _count_value(source: Mapping[str, Any], *keys: str) -> int:
    total = 0
    for key in keys:
        value = source.get(key)
        if isinstance(value, list):
            total += len(value)
        elif isinstance(value, Mapping):
            total += len(value)
        else:
            total += _int(value)
    return total


def _status_codes(runtime: Mapping[str, Any], step_info: Mapping[str, Any], network_entries: Sequence[Any]) -> list[int]:
    codes: list[int] = []
    for source in (runtime, step_info):
        for key in ("network_status_codes", "status_codes"):
            value = source.get(key)
            if isinstance(value, list):
                codes.extend(_int(item) for item in value if _int(item))
    for entry in network_entries:
        if isinstance(entry, Mapping):
            status = _int(entry.get("status") or entry.get("status_code"))
            if status:
                codes.append(status)
    return codes


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
