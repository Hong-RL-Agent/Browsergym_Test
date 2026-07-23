"""API catalog loading and normalization for direct backend fuzzing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlparse


SUPPORTED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


@dataclass(frozen=True)
class ApiEndpoint:
    api_id: str
    method: str
    path: str
    query_schema: Dict[str, Any] = field(default_factory=dict)
    path_schema: Dict[str, Any] = field(default_factory=dict)
    body_schema: Dict[str, Any] = field(default_factory=dict)
    header_schema: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    expected_status: List[int] = field(default_factory=lambda: [200])
    auth_required: bool = False
    required_params: List[str] = field(default_factory=list)
    enum_values: Dict[str, List[Any]] = field(default_factory=dict)
    latency_threshold_ms: int | None = None
    test_safe: bool = False


@dataclass(frozen=True)
class ApiSite:
    site_id: str
    base_url: str
    health_check: str = "/"
    apis: List[ApiEndpoint] = field(default_factory=list)
    auth: Dict[str, Any] = field(default_factory=dict)
    allow_mutating_requests: bool = False


def load_api_catalog(path: str | Path, start_port: int | None = None, end_port: int | None = None) -> List[ApiSite]:
    data = _read_json_dict(Path(path))
    raw_sites = data.get("sites", [])
    if not isinstance(raw_sites, list):
        raise ValueError("API catalog requires a sites list.")
    sites = [_normalize_site(item) for item in raw_sites if isinstance(item, Mapping)]
    if start_port is not None or end_port is not None:
        sites = [
            site for site in sites
            if _port_in_range(_port_from_url(site.base_url), start_port, end_port)
        ]
    return sites


def iter_endpoints(sites: Iterable[ApiSite]) -> Iterable[tuple[ApiSite, ApiEndpoint]]:
    for site in sites:
        for endpoint in site.apis:
            yield site, endpoint


def site_to_dict(site: ApiSite) -> Dict[str, Any]:
    return {
        "site_id": site.site_id,
        "base_url": site.base_url,
        "health_check": site.health_check,
        "auth": site.auth,
        "apis": [endpoint_to_dict(endpoint) for endpoint in site.apis],
    }


def endpoint_to_dict(endpoint: ApiEndpoint) -> Dict[str, Any]:
    return {
        "api_id": endpoint.api_id,
        "method": endpoint.method,
        "path": endpoint.path,
        "query_schema": endpoint.query_schema,
        "path_schema": endpoint.path_schema,
        "body_schema": endpoint.body_schema,
        "header_schema": endpoint.header_schema,
        "response_schema": endpoint.response_schema,
        "expected_status": endpoint.expected_status,
        "auth_required": endpoint.auth_required,
        "required_params": endpoint.required_params,
        "enum_values": endpoint.enum_values,
        "latency_threshold_ms": endpoint.latency_threshold_ms,
        "test_safe": endpoint.test_safe,
    }


def _normalize_site(raw: Mapping[str, Any]) -> ApiSite:
    site_id = str(raw.get("site_id") or "").strip()
    base_url = str(raw.get("base_url") or "").rstrip("/")
    if not site_id or not base_url:
        raise ValueError(f"Invalid API site entry: {raw}")
    raw_apis = raw.get("apis", [])
    if not isinstance(raw_apis, list):
        raw_apis = []
    return ApiSite(
        site_id=site_id,
        base_url=base_url,
        health_check=str(raw.get("health_check") or "/"),
        apis=[_normalize_endpoint(item) for item in raw_apis if isinstance(item, Mapping)],
        auth=dict(raw.get("auth", {})) if isinstance(raw.get("auth"), Mapping) else {},
        allow_mutating_requests=bool(raw.get("allow_mutating_requests", False)),
    )


def _normalize_endpoint(raw: Mapping[str, Any]) -> ApiEndpoint:
    method = str(raw.get("method") or "GET").upper()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported API method: {method}")
    expected_status = raw.get("expected_status", [200])
    if not isinstance(expected_status, list):
        expected_status = [expected_status]
    required_params = raw.get("required_params", [])
    if not isinstance(required_params, list):
        required_params = []
    return ApiEndpoint(
        api_id=str(raw.get("api_id") or f"{method}-{raw.get('path', '/')}".strip("/")).strip(),
        method=method,
        path=str(raw.get("path") or "/"),
        query_schema=_dict(raw.get("query_schema")),
        path_schema=_dict(raw.get("path_schema")),
        body_schema=_dict(raw.get("body_schema")),
        header_schema=_dict(raw.get("header_schema")),
        response_schema=_dict(raw.get("response_schema")),
        expected_status=[int(status) for status in expected_status],
        auth_required=bool(raw.get("auth_required")),
        required_params=[str(item) for item in required_params],
        enum_values={
            str(key): value
            for key, value in _dict(raw.get("enum_values")).items()
            if isinstance(value, list)
        },
        latency_threshold_ms=_optional_int(raw.get("latency_threshold_ms")),
        test_safe=bool(raw.get("test_safe", False)),
    )


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json_dict(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _port_from_url(base_url: str) -> int | None:
    parsed = urlparse(base_url)
    return parsed.port


def _port_in_range(port: int | None, start_port: int | None, end_port: int | None) -> bool:
    if port is None:
        return False
    if start_port is not None and port < start_port:
        return False
    if end_port is not None and port > end_port:
        return False
    return True
