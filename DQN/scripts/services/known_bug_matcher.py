"""Known-bug matching helpers for autonomous BrowserGym PPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


TYPE_ALIASES = {
    "layout-overlap": {"layout-overlap", "css-layout"},
    "duplicated-rendering": {"duplicated-rendering", "component-rendering"},
    "button-no-response": {"button-no-response"},
}
SITE001_FALLBACKS = [
    {
        "id": "site001-bug01",
        "type": "button-no-response",
        "selector_text": "구매하기",
        "description": "Purchase button does not respond.",
    },
    {
        "id": "site001-bug02",
        "type": "component-rendering",
        "description": "Recommended book card is rendered more than once.",
    },
    {
        "id": "site001-bug03",
        "type": "css-layout",
        "description": "Layout overlap in product/book UI.",
    },
]


def load_known_bugs(site_id: str) -> List[Dict[str, Any]]:
    path = Path("datasets") / site_id / "bug_catalog.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []
        bugs = _extract_bugs(data)
        if bugs:
            return bugs
    if site_id == "site001":
        return [dict(item) for item in SITE001_FALLBACKS]
    return []


def match_anomalies_to_known_bugs(
    anomalies: List[Dict[str, Any]],
    known_bugs: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for anomaly in anomalies:
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        if confidence < threshold:
            continue
        best: Dict[str, Any] | None = None
        best_score = 0.0
        for bug in known_bugs:
            score, reason = _score_match(anomaly, bug)
            if score > best_score:
                bug_id = _bug_id(bug)
                best = {
                    **anomaly,
                    "matched_bug_id": bug_id,
                    "match_reason": reason,
                    "match_score": score,
                }
                best_score = score
        if best and best_score >= threshold:
            matches.append(best)
    return matches


def _extract_bugs(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("bugs", "items", "bug_catalog", "known_bugs"):
            bugs = data.get(key)
            if isinstance(bugs, list):
                return [item for item in bugs if isinstance(item, dict)]
        if data.get("id") or data.get("bug_id"):
            return [data]
    return []


def _score_match(anomaly: Mapping[str, Any], bug: Mapping[str, Any]) -> tuple[float, str]:
    anomaly_type = str(anomaly.get("type", ""))
    bug_type = str(bug.get("type") or bug.get("bug_type") or "")
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    bug_id = _bug_id(bug)

    if _has_data_bug_id(evidence, bug):
        return 0.95, "type-and-data-bug-id"

    type_match = _types_match(anomaly_type, bug_type)
    if not type_match:
        return _site001_fallback_score(anomaly_type, evidence, bug_id)

    score = 0.65
    reason = "type"
    if _text_matches(evidence, bug):
        score = 0.8
        reason = "type-and-selector-text"

    if anomaly_type == "button-no-response" and "구매하기" in _evidence_text(evidence) and bug_id == "site001-bug01":
        return 0.9, "site001-purchase-button"
    if anomaly_type == "duplicated-rendering" and bug_id == "site001-bug02":
        return 0.85, "site001-duplicated-rendering"
    if anomaly_type == "layout-overlap" and bug_id == "site001-bug03":
        return 0.85, "site001-layout-overlap"
    return score, reason


def _site001_fallback_score(
    anomaly_type: str,
    evidence: Mapping[str, Any],
    bug_id: str,
) -> tuple[float, str]:
    if anomaly_type == "button-no-response" and bug_id == "site001-bug01":
        if "구매하기" in _evidence_text(evidence):
            return 0.85, "site001-button-no-response"
        return 0.6, "site001-button-no-response"
    if anomaly_type == "duplicated-rendering" and bug_id == "site001-bug02":
        return 0.85, "site001-duplicated-rendering"
    if anomaly_type == "layout-overlap" and bug_id == "site001-bug03":
        return 0.85, "site001-layout-overlap"
    return 0.0, ""


def _types_match(anomaly_type: str, bug_type: str) -> bool:
    if not bug_type:
        return False
    aliases = TYPE_ALIASES.get(anomaly_type, {anomaly_type})
    return bug_type in aliases or anomaly_type == bug_type


def _has_data_bug_id(evidence: Mapping[str, Any], bug: Mapping[str, Any]) -> bool:
    expected = bug.get("data-bug-id") or bug.get("data_bug_id")
    selector = bug.get("selector")
    if not expected and isinstance(selector, Mapping):
        expected = selector.get("data-bug-id") or selector.get("data_bug_id")
    expected = str(expected or "")
    if not expected:
        return False
    values = {str(value) for value in evidence.values() if value is not None}
    return expected in values


def _text_matches(evidence: Mapping[str, Any], bug: Mapping[str, Any]) -> bool:
    selector_text = str(bug.get("selector_text") or bug.get("text") or bug.get("name") or "").lower()
    if not selector_text:
        selector = bug.get("selector")
        if isinstance(selector, Mapping):
            selector_text = str(selector.get("text") or selector.get("name") or "").lower()
    return bool(selector_text and selector_text in _evidence_text(evidence).lower())


def _evidence_text(evidence: Mapping[str, Any]) -> str:
    return " ".join(str(value) for value in evidence.values() if value is not None)


def _bug_id(bug: Mapping[str, Any]) -> str:
    return str(bug.get("id") or bug.get("bug_id") or bug.get("name") or "")
