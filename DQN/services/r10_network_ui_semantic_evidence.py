"""Network/UI semantic evidence extraction for R10-style diagnostics.

This module is GT-blind: it only uses runtime browser observations and network
events.  It does not classify defects; it produces structured semantic pairs
for downstream detectors and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


MAX_JSON_ARRAY_ITEMS = 8
MAX_FIELD_COUNT = 120


def semantic_pair_confidence(pair: Mapping[str, Any]) -> float | None:
    """Return canonical confidence, accepting the legacy field as fallback."""
    value = pair.get("confidence") if "confidence" in pair else pair.get("pair_confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_runtime_network_telemetry(
    observation: Mapping[str, Any], info: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize BrowserGym temporal network events without copying bodies."""
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    temporal = runtime.get("temporal_network", {}) if isinstance(runtime, Mapping) else {}
    temporal = temporal if isinstance(temporal, Mapping) else {}
    request_events = [row for row in temporal.get("request_events", []) or [] if isinstance(row, Mapping)]
    completed = [row for row in temporal.get("completed_requests", []) or [] if isinstance(row, Mapping)]
    failed = [row for row in temporal.get("failed_requests", []) or [] if isinstance(row, Mapping)]
    events = request_events + completed + failed
    parsed = sum(int("response_body_json" in row and row.get("response_body_json") is not None) for row in completed)
    http_errors = sum(int(isinstance(row.get("status"), (int, float)) and int(row.get("status")) >= 400) for row in completed)
    action_context = temporal.get("action_context", {}) if isinstance(temporal.get("action_context"), Mapping) else {}
    action_start = action_context.get("action_start_timestamp")
    action_end = action_context.get("action_end_timestamp")
    associations = []
    response_by_id = {str(row.get("request_id") or ""): row for row in completed if row.get("request_id")}
    failed_by_id = {str(row.get("request_id") or ""): row for row in failed if row.get("request_id")}
    # Merge request-start and completion streams by stable request_id. A
    # request may start in one action and complete in a later action.
    request_by_id: dict[str, dict[str, Any]] = {}
    anonymous_events: list[Mapping[str, Any]] = []
    for event in request_events + completed + failed:
        request_id = str(event.get("request_id") or "")
        if not request_id:
            anonymous_events.append(event)
            continue
        row = request_by_id.setdefault(request_id, {})
        for key, value in event.items():
            if value is not None and (key not in row or row[key] in (None, "")):
                row[key] = value
    # Start/response/failure are lifecycle events for one request, not
    # duplicate associations. Duplicates are measured on the canonical
    # association rows below; by construction each request_id occurs once.
    duplicate_request_ids = 0
    association_events = list(request_by_id.values()) + anonymous_events
    for event in association_events:
        request_id = str(event.get("request_id") or "")
        response = response_by_id.get(request_id) or failed_by_id.get(request_id) or {}
        request_start = event.get("request_start_timestamp")
        strict = False
        outside_window = False
        if isinstance(action_start, (int, float)) and isinstance(action_end, (int, float)) and isinstance(request_start, (int, float)):
            strict = float(action_start) <= float(request_start) <= float(action_end)
            outside_window = not strict
        if strict:
            association_type = "STRICT_ACTION_CAUSED"
        elif outside_window:
            association_type = "BACKGROUND_UNASSOCIATED"
        elif request_start is None:
            association_type = "TEMPORAL_ONLY"
        else:
            association_type = "UNKNOWN"
        associations.append({
            "action_id": action_context.get("action_id"),
            "action_type": action_context.get("action_type"),
            "target_identity": action_context.get("target_identity"),
            "request_id": request_id,
            "method": event.get("method"),
            "url": event.get("url"),
            "request_start_timestamp": request_start,
            "response_timestamp": response.get("response_timestamp") or event.get("response_timestamp"),
            "association_type": association_type,
            "association_confidence": 1.0 if strict else 0.0,
        })
    strict_count = sum(int(row["association_type"] == "STRICT_ACTION_CAUSED") for row in associations)
    return {
        "raw_request_count": len(request_by_id) + len(anonymous_events),
        "raw_response_count": len(completed),
        "normalized_request_count": len(request_by_id) + len(anonymous_events),
        "normalized_response_count": len(completed),
        "failed_request_count": len(failed),
        "http_error_response_count": http_errors,
        "parsed_response_count": parsed,
        "events": events,
        "request_events": request_events,
        "associations": associations,
        "strict_associated_request_count": strict_count,
        "duplicate_request_id_count": duplicate_request_ids,
        "request_id_linkage_mismatch_count": sum(
            int(str(row.get("request_id") or "") not in response_by_id and str(row.get("request_id") or "") not in failed_by_id)
            for row in association_events if row.get("request_id")
        ),
        "schema": "runtime_signals.temporal_network.completed_requests+failed_requests",
    }
KEY_ALIASES = {
    "amount": "price",
    "cost": "price",
    "fee": "price",
    "planname": "plan",
    "plan_name": "plan",
    "tier": "plan",
    "usageamount": "usage",
    "usage_amount": "usage",
}
IDENTITY_KEYS = (
    "id",
    "product_id",
    "item_id",
    "plan_id",
    "subscription_id",
    "order_id",
    "user_id",
    "name",
    "plan",
    "type",
)
VALUE_KEYS = {
    "plan",
    "name",
    "type",
    "category",
    "status",
    "price",
    "amount",
    "cost",
    "fee",
    "usage",
    "quantity",
    "count",
    "total",
    "subtotal",
    "balance",
    "limit",
    "current",
    "used",
}


@dataclass(frozen=True)
class SemanticValue:
    raw: Any
    normalized: Any
    value_type: str
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "value_type": self.value_type,
            "unit": self.unit,
        }


def build_network_ui_semantic_evidence(
    *,
    action_context: Mapping[str, Any],
    network_entries: Iterable[Mapping[str, Any]],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact semantic evidence without consulting GT."""
    action_id = str(action_context.get("action_id") or "")
    action_ts = action_context.get("action_timestamp")
    requests = [_request_summary(entry, action_id=action_id, action_timestamp=action_ts) for entry in network_entries if isinstance(entry, Mapping)]
    network_fields = extract_network_fields(network_entries)
    ui_fields = extract_ui_fields(observation)
    pairs = pair_network_ui_fields(network_fields, ui_fields)
    return {
        "action_id": action_id,
        "requests": requests[:40],
        "responses": [row for row in requests if row.get("status") is not None][:40],
        "network_fields": network_fields[:MAX_FIELD_COUNT],
        "ui_fields": ui_fields[:MAX_FIELD_COUNT],
        "pairs": pairs[:MAX_FIELD_COUNT],
        "equal_pairs": sum(1 for row in pairs if row.get("relation") == "EQUAL"),
        "not_equal_pairs": sum(1 for row in pairs if row.get("relation") == "NOT_EQUAL"),
        "uncertain_pairs": sum(1 for row in pairs if row.get("relation") == "PAIR_UNCERTAIN"),
        "gt_runtime_access": False,
    }


def extract_network_fields(network_entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for entry in network_entries:
        if not isinstance(entry, Mapping):
            continue
        body = _response_payload(entry)
        if body is None:
            continue
        for path, value in _flatten_semantic(body):
            key = normalize_key(path.split(".")[-1])
            if not key:
                continue
            semantic = normalize_value(value)
            fields.append(
                {
                    "source": "network_response",
                    "request_id": str(entry.get("request_id") or ""),
                    "url": str(entry.get("url") or ""),
                    "normalized_url": normalize_url(str(entry.get("url") or "")),
                    "method": str(entry.get("method") or "").upper(),
                    "status": _int_or_none(entry.get("status")),
                    "key": key,
                    "path": path,
                    "value": value,
                    "semantic_value": semantic.to_dict(),
                    "entity_hint": _entity_hint(path, body),
                }
            )
            if len(fields) >= MAX_FIELD_COUNT:
                return fields
    return fields


def extract_ui_fields(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    fields: list[dict[str, Any]] = []
    for row in page_state.get("ui_semantic_fields", []) or []:
        if not isinstance(row, Mapping):
            continue
        fields.extend(_ui_field_from_key_value(row.get("key"), row.get("value"), row, source="dom_semantic"))
    for container_key in ("ui_entities", "rendered_entities", "cart_items", "result_entities", "entities", "items"):
        for item in _as_list(page_state.get(container_key)):
            if isinstance(item, Mapping):
                for key, value in item.items():
                    fields.extend(_ui_field_from_key_value(key, value, item, source=f"page_state.{container_key}"))
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates[:64]:
            if not isinstance(candidate, Mapping) or not bool(candidate.get("visible", True)):
                continue
            for key in ("value", "text", "name", "aria_label", "title"):
                value = candidate.get(key)
                if value not in (None, ""):
                    label = candidate.get("label") or candidate.get("name") or candidate.get("aria_label") or key
                    fields.extend(_ui_field_from_key_value(label, value, candidate, source="candidate"))
    for key, value in _parse_text_key_values(str(page_state.get("page_text") or page_state.get("page_text_sample") or "")):
        fields.extend(_ui_field_from_key_value(key, value, {"display_text": f"{key}: {value}"}, source="visible_text"))
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for field in fields:
        deduped.setdefault((field["key"], str(field["semantic_value"]["normalized"]), field["source"]), field)
    return list(deduped.values())[:MAX_FIELD_COUNT]


def pair_network_ui_fields(network_fields: list[Mapping[str, Any]], ui_fields: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for field in ui_fields:
        by_key.setdefault(str(field.get("key") or ""), []).append(field)
    for nfield in network_fields:
        key = str(nfield.get("key") or "")
        candidates = by_key.get(key, [])
        if not candidates:
            continue
        for ufield in candidates[:4]:
            nval = nfield.get("semantic_value", {})
            uval = ufield.get("semantic_value", {})
            relation = _relation(nval, uval)
            confidence = 1.0 if relation in {"EQUAL", "NOT_EQUAL"} else 0.7
            pairs.append(
                {
                    "network_key": key,
                    "network_path": nfield.get("path"),
                    "network_value": nfield.get("value"),
                    "network_semantic_value": nval,
                    "ui_key": ufield.get("key"),
                    "ui_value": ufield.get("value"),
                    "ui_semantic_value": uval,
                    "match_basis": "normalized_key",
                    "relation": relation,
                    "confidence": confidence,
                    "pair_confidence": confidence,
                    "pair_method": "EXACT_KEY",
                    "request_id": nfield.get("request_id"),
                    "url": nfield.get("url"),
                    "method": nfield.get("method"),
                    "status": nfield.get("status"),
                }
            )
    return pairs


def normalize_key(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_가-힣]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        return ""
    return KEY_ALIASES.get(text, text)


def normalize_value(value: Any) -> SemanticValue:
    if isinstance(value, bool):
        return SemanticValue(value, value, "boolean")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return SemanticValue(value, float(value), "number")
    text = " ".join(str(value or "").strip().split())
    percent = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*%", text)
    if percent:
        return SemanticValue(value, float(percent.group(1)), "number", "%")
    embedded_percent = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*%", text)
    if embedded_percent:
        return SemanticValue(value, float(embedded_percent.group(1).replace(",", "")), "number", "%")
    currency = re.search(r"[$￦₩€£]\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if currency:
        return SemanticValue(value, float(currency.group(1).replace(",", "")), "number")
    points = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:pts?|points?)\b", text, flags=re.IGNORECASE)
    if points:
        return SemanticValue(value, float(points.group(1).replace(",", "")), "number")
    numeric = re.sub(r"[,₩$€£원\s]", "", text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric or ""):
        return SemanticValue(value, float(numeric), "number")
    return SemanticValue(value, text.lower(), "string")


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or "/"


def _request_summary(entry: Mapping[str, Any], *, action_id: str, action_timestamp: Any) -> dict[str, Any]:
    req_ms = entry.get("relative_ms")
    return {
        "request_id": str(entry.get("request_id") or ""),
        "action_id": action_id,
        "action_timestamp": action_timestamp,
        "timestamp_request_start": entry.get("request_relative_ms") or req_ms,
        "timestamp_response": req_ms,
        "delta_ms": _delta_ms(req_ms, entry.get("request_relative_ms")),
        "method": str(entry.get("method") or "").upper(),
        "url": str(entry.get("url") or ""),
        "normalized_url": normalize_url(str(entry.get("url") or "")),
        "resource_type": str(entry.get("resource_type") or ""),
        "request_body_summary": entry.get("request_body_summary") or "",
        "status": _int_or_none(entry.get("status")),
        "content_type": _content_type(entry),
        "response_body_available": bool(entry.get("response_body_available")),
        "response_body_summary": entry.get("response_body_summary"),
        "request_failed": bool(entry.get("network_failure")),
        "failure_reason": entry.get("failure_text") or "",
    }


def _response_payload(entry: Mapping[str, Any]) -> Any:
    if "response_body_json" in entry:
        return entry.get("response_body_json")
    if "response_body_text" in entry:
        text = str(entry.get("response_body_text") or "")
        try:
            return json.loads(text)
        except Exception:
            return dict(_parse_text_key_values(text))
    if "response_body_summary" in entry:
        text = str(entry.get("response_body_summary") or "")
        try:
            return json.loads(text)
        except Exception:
            return dict(_parse_text_key_values(text))
    return None


def _flatten_semantic(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (Mapping, list)):
                rows.extend(_flatten_semantic(item, path))
            elif item not in (None, ""):
                rows.append((path, item))
    elif isinstance(value, list):
        for idx, item in enumerate(value[:MAX_JSON_ARRAY_ITEMS]):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            rows.extend(_flatten_semantic(item, path))
    return rows


def _ui_field_from_key_value(key: Any, value: Any, source_row: Mapping[str, Any], *, source: str) -> list[dict[str, Any]]:
    normalized_key = normalize_key(key)
    if not normalized_key or value in (None, ""):
        return []
    semantic = normalize_value(value)
    return [
        {
            "source": source,
            "key": normalized_key,
            "value": value,
            "display_text": source_row.get("display_text") or source_row.get("text") or "",
            "element_role": source_row.get("element_role") or source_row.get("role") or "",
            "element_id": source_row.get("element_id") or source_row.get("id") or source_row.get("bid") or "",
            "dom_path": source_row.get("dom_path") or "",
            "visible": bool(source_row.get("visible", True)),
            "semantic_value": semantic.to_dict(),
        }
    ]


def _parse_text_key_values(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in re.split(r"[\n\r]+", text or ""):
        line = " ".join(line.strip().split())
        if not line or len(line) > 160:
            continue
        match = re.match(r"^([A-Za-z가-힣][A-Za-z0-9가-힣 _/-]{1,40})\s*[:：=]\s*(.{1,80})$", line)
        if match:
            rows.append((match.group(1), match.group(2)))
    return rows[:MAX_FIELD_COUNT]


def _relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if left.get("value_type") != right.get("value_type"):
        return "PAIR_UNCERTAIN"
    if left.get("unit") != right.get("unit") and left.get("unit") and right.get("unit"):
        return "PAIR_UNCERTAIN"
    return "EQUAL" if left.get("normalized") == right.get("normalized") else "NOT_EQUAL"


def _entity_hint(path: str, body: Any) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        return {}
    hint = {}
    for key in IDENTITY_KEYS:
        if key in body and body[key] not in (None, ""):
            hint[key] = body[key]
    return hint


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        return [value]
    return []


def _content_type(entry: Mapping[str, Any]) -> str:
    headers = entry.get("response_headers") if isinstance(entry.get("response_headers"), Mapping) else {}
    return str(headers.get("content-type") or headers.get("Content-Type") or entry.get("content_type") or "").split(";", 1)[0].lower()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _delta_ms(right: Any, left: Any) -> int | None:
    try:
        return int(float(right) - float(left))
    except (TypeError, ValueError):
        return None
