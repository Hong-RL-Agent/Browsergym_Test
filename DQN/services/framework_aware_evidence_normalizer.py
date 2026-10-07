"""Conservative, framework-aware normalization of raw network evidence.

This module is detector-facing only.  It deliberately keeps the raw event in
each normalized record and does not depend on scenario ground truth or agent
observations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from urllib.parse import parse_qs, urlparse
from typing import Any


SEMANTIC_CLASSES = (
    "USER_FUNCTIONAL_REQUEST",
    "FRAMEWORK_NAVIGATION",
    "PREFETCH",
    "STATIC_ASSET",
    "BACKGROUND_ACTIVITY",
    "DEVELOPMENT_INFRASTRUCTURE",
    "UNKNOWN",
)
FUNCTIONAL_RELEVANCE = ("RELEVANT", "IRRELEVANT", "POSSIBLY_RELEVANT", "UNKNOWN")
FRAMEWORK_CONTEXTS = ("GENERIC", "REACT", "NEXT", "UNKNOWN")

_STATIC_TYPES = {"script", "stylesheet", "style", "image", "media", "font", "manifest", "texttrack"}
_STATIC_SUFFIXES = (".js", ".mjs", ".cjs", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".avif", ".woff", ".woff2", ".ttf", ".otf", ".map")
_DEV_TOKENS = ("webpack-hmr", "hot-update", "/__vite", "vite-hmr", "sockjs", "livereload", "hmr")
_BACKGROUND_TOKENS = ("analytics", "telemetry", "beacon", "metrics", "sentry", "rum", "heartbeat")
_FUNCTIONAL_PATH_TOKENS = ("/api/", "/graphql", "/search", "/cart", "/order", "/profile", "/account", "/checkout", "/login", "/logout")


def normalize_network_event(event: Mapping[str, Any], *, action_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return one normalized event without mutating *event*."""
    raw = dict(event)
    context = action_context or {}
    url = str(raw.get("url") or raw.get("request_url") or raw.get("path") or "")
    method = str(raw.get("method") or "GET").upper()
    resource_type = str(raw.get("resource_type") or raw.get("type") or "").lower()
    headers = _headers(raw)
    headers.update(_headers(raw.get("headers")))
    response_headers = _headers(raw.get("response_headers"))
    path = urlparse(url).path.lower()
    query = parse_qs(urlparse(url).query)
    framework_context, framework_reasons = detect_framework_context(raw)
    action_id = str(context.get("action_id") or raw.get("trigger_action_id") or "")
    action_correlated = bool(action_id or raw.get("action_correlated"))
    semantic_class, reasons, confidence = _classify(
        raw, url=url, path=path, method=method, resource_type=resource_type,
        headers=headers, response_headers=response_headers, query=query,
        framework_context=framework_context, action_correlated=action_correlated,
    )
    reasons = framework_reasons + reasons
    relevance = {
        "USER_FUNCTIONAL_REQUEST": "RELEVANT",
        "FRAMEWORK_NAVIGATION": "IRRELEVANT",
        "PREFETCH": "IRRELEVANT",
        "STATIC_ASSET": "IRRELEVANT",
        "BACKGROUND_ACTIVITY": "IRRELEVANT",
        "DEVELOPMENT_INFRASTRUCTURE": "IRRELEVANT",
        "UNKNOWN": "POSSIBLY_RELEVANT" if action_correlated else "UNKNOWN",
    }[semantic_class]
    status = _int_or_none(raw.get("status") or raw.get("status_code") or raw.get("response_status"))
    transport_failed = bool(raw.get("network_failure") or raw.get("transport_failed") or raw.get("event") == "requestfailed")
    http_failed = status is not None and status >= 400
    return {
        "request_id": str(raw.get("request_id") or raw.get("trace_id") or ""),
        "url": url,
        "method": method,
        "resource_type": resource_type,
        "status": status,
        "transport_failed": transport_failed,
        "http_failed": http_failed,
        "started_at": raw.get("started_at"),
        "ended_at": raw.get("ended_at"),
        "relative_ms": raw.get("relative_ms"),
        "semantic_class": semantic_class,
        "functional_relevance": relevance,
        "classification_confidence": confidence,
        "classification_reasons": reasons or ["INSUFFICIENT_CLASSIFICATION_EVIDENCE"],
        "action_correlated": action_correlated,
        "trigger_action_id": action_id,
        "framework_context": framework_context,
        "raw_network_event": raw,
    }


def normalize_network_events(events: Iterable[Mapping[str, Any]], *, action_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return [normalize_network_event(event, action_context=action_context) for event in events if isinstance(event, Mapping)]


def detect_framework_context(event: Mapping[str, Any]) -> tuple[str, list[str]]:
    url = str(event.get("url") or event.get("request_url") or "").lower()
    headers = _headers(event)
    headers.update(_headers(event.get("headers")))
    headers.update(_headers(event.get("response_headers")))
    tokens = " ".join([url, " ".join(f"{k}:{v}" for k, v in headers.items())]).lower()
    if any(token in tokens for token in ("/_next/", "next-router", "next-url", "next-router-state-tree", "text/x-component")):
        return "NEXT", ["NEXT_RUNTIME_OR_PROTOCOL_MARKER"]
    if any(token in tokens for token in ("react", "react-refresh", "react-refresh-runtime")):
        return "REACT", ["REACT_RUNTIME_MARKER"]
    if any(token in tokens for token in ("webpack", "vite", "sockjs", "hot-update")):
        return "GENERIC", ["DEVELOPMENT_RUNTIME_MARKER"]
    return "GENERIC", []


def _classify(raw: Mapping[str, Any], *, url: str, path: str, method: str, resource_type: str,
              headers: Mapping[str, str], response_headers: Mapping[str, str], query: Mapping[str, list[str]],
              framework_context: str, action_correlated: bool) -> tuple[str, list[str], str]:
    tokens = " ".join([url.lower(), " ".join(f"{k}:{v}" for k, v in headers.items()), " ".join(f"{k}:{v}" for k, v in response_headers.items())]).lower()
    if resource_type in {"websocket", "eventsource"} and any(token in tokens for token in _DEV_TOKENS):
        return "DEVELOPMENT_INFRASTRUCTURE", ["DEV_LONG_LIVED_CHANNEL"], "HIGH"
    if any(token in tokens for token in _DEV_TOKENS):
        return "DEVELOPMENT_INFRASTRUCTURE", ["DEV_RUNTIME_MARKER"], "HIGH"
    if resource_type in _STATIC_TYPES or path.endswith(_STATIC_SUFFIXES):
        return "STATIC_ASSET", ["RESOURCE_TYPE_OR_STATIC_SUFFIX"], "HIGH"
    prefetch = any(headers.get(key, "") in {"prefetch", "1", "true"} for key in ("purpose", "sec-purpose", "x-middleware-prefetch", "next-router-prefetch"))
    prefetch = prefetch or "prefetch" in query or "prefetch" in path
    if framework_context == "NEXT" and prefetch:
        return "PREFETCH", ["NEXT_PREFETCH_METADATA"], "HIGH"
    if framework_context == "NEXT" and ("rsc" in tokens or "text/x-component" in tokens or "next-router-state-tree" in tokens):
        return "FRAMEWORK_NAVIGATION", ["NEXT_NAVIGATION_PROTOCOL"], "HIGH"
    if any(token in tokens for token in _BACKGROUND_TOKENS):
        return "BACKGROUND_ACTIVITY", ["BACKGROUND_ENDPOINT_MARKER"], "MEDIUM"
    if method in {"POST", "PUT", "PATCH", "DELETE"} and action_correlated:
        return "USER_FUNCTIONAL_REQUEST", ["MUTATING_METHOD_ACTION_CORRELATED"], "HIGH"
    if action_correlated and (any(token in path for token in _FUNCTIONAL_PATH_TOKENS) or resource_type in {"xhr", "fetch"}):
        return "USER_FUNCTIONAL_REQUEST", ["ACTION_CORRELATED_FUNCTIONAL_ENDPOINT"], "MEDIUM"
    return "UNKNOWN", [], "LOW"


def _headers(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key).lower(): str(item).lower() for key, item in value.items()}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
