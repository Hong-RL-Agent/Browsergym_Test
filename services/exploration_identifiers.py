"""Stable identifiers for comparing exploration runs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


VOLATILE_QUERY_KEYS = {"_", "cache", "nonce", "timestamp", "ts"}


def normalized_url(value: str) -> str:
    parts = urlsplit(str(value or ""))
    query = urlencode(
        sorted((key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True) if key.lower() not in VOLATILE_QUERY_KEYS)
    )
    path = re.sub(r"/+$", "", parts.path or "/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def state_id(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    candidate_signatures = sorted(_candidate_signature(item) for item in candidates if isinstance(item, Mapping))
    payload = {
        "url": normalized_url(str(page.get("url") or "")),
        "title": str(page.get("title") or "").strip().lower(),
        "viewport": str(page.get("viewport_type") or ""),
        "modal": bool(page.get("has_modal") or page.get("has_dialog")),
        "form": bool(page.get("has_form")),
        "candidates": candidate_signatures,
    }
    return _digest(payload, "state")


def action_id(before_state_id: str, action: Mapping[str, Any], candidate: Mapping[str, Any] | None = None) -> str:
    payload = {
        "state": before_state_id,
        "type": str(action.get("action_type") or ""),
        "candidate": _candidate_signature(candidate or {}),
    }
    return _digest(payload, "action")


def transition_id(before_state_id: str, stable_action_id: str, after_state_id: str) -> str:
    return _digest({"before": before_state_id, "action": stable_action_id, "after": after_state_id}, "transition")


def _candidate_signature(candidate: Mapping[str, Any]) -> str:
    return "|".join(
        str(candidate.get(key) or "").strip().lower()[:160]
        for key in ("role", "name", "text", "bid", "selector", "type")
    )


def _digest(payload: Mapping[str, Any], prefix: str) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"

