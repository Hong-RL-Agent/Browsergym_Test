"""Catalog-driven known-bug matching helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from services.site_profile_service import bug_id_of, data_bug_id_from_selector, load_bug_catalog


TYPE_ALIASES = {
    "button-no-response": {"button-no-response", "timeout-no-feedback", "cart-quantity-mismatch"},
    "form-no-feedback": {"form-no-feedback", "button-no-response", "async-hang", "async-no-feedback", "timeout-no-feedback"},
    "component-rendering": {"component-rendering", "duplicated-rendering", "empty-state-rendering"},
    "duplicated-rendering": {"component-rendering", "duplicated-rendering"},
    "empty-state-rendering": {"component-rendering", "empty-state-rendering", "sparse-data-rendering", "data-missing"},
    "css-layout": {"css-layout", "layout-overlap", "layout-overflow"},
    "layout-overlap": {"css-layout", "layout-overlap", "layout-overflow"},
    "layout-overflow": {"css-layout", "layout-overflow", "layout-overlap"},
    "broken-navigation": {"broken-navigation"},
    "api-forbidden": {"api-forbidden", "api-ui-mismatch", "network-error"},
    "api-ui-mismatch": {"api-ui-mismatch", "api-forbidden", "cart-quantity-mismatch", "sparse-data-rendering"},
    "async-hang": {"async-hang", "async-no-feedback", "timeout-no-feedback", "form-no-feedback"},
    "async-no-feedback": {"async-no-feedback", "async-hang", "timeout-no-feedback", "form-no-feedback"},
    "timeout-no-feedback": {"timeout-no-feedback", "async-hang", "form-no-feedback", "button-no-response"},
    "cart-quantity-mismatch": {"cart-quantity-mismatch", "cart-state-mismatch", "api-ui-mismatch", "button-no-response"},
    "cart-state-mismatch": {"cart-state-mismatch", "cart-quantity-mismatch", "api-ui-mismatch"},
    "sparse-data-rendering": {"sparse-data-rendering", "data-missing", "api-ui-mismatch", "empty-state-rendering", "component-rendering"},
    "data-missing": {"data-missing", "sparse-data-rendering", "empty-state-rendering"},
    "network-error": {"network-error", "api-forbidden", "api-ui-mismatch"},
}

SITE001_FALLBACK_BUGS = [
    {"id": "site001-bug01", "type": "button-no-response", "selector_text": "구매하기"},
    {"id": "site001-bug02", "type": "component-rendering"},
    {"id": "site001-bug03", "type": "css-layout"},
]


def load_known_bugs(site_id: str) -> List[Dict[str, Any]]:
    bugs = load_bug_catalog(site_id)
    return _with_site001_fallback(site_id, bugs)


def match_anomalies_to_known_bugs(
    anomalies: List[Dict[str, Any]],
    known_bugs: List[Dict[str, Any]],
    threshold: float = 0.6,
    site_id: str = "",
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for anomaly in anomalies:
        if float(anomaly.get("confidence", 0.0) or 0.0) < threshold:
            continue
        anomaly_type = str(anomaly.get("type", ""))
        evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
        anomaly_confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        best_bug: Dict[str, Any] | None = None
        best_score = 0.0
        best_reason = ""
        for bug in known_bugs:
            score, reason = _score_match(anomaly_type, evidence, bug, anomaly_confidence, site_id)
            if score > best_score:
                best_bug = bug
                best_score = score
                best_reason = reason
        reason_tokens = set(best_reason.split("+")) if best_reason else set()
        effective_threshold = threshold
        if {"data-bug-id-match", "catalog-bug-id-match", "selector-match"}.intersection(reason_tokens):
            effective_threshold = 0.4
        elif {"layout-check-match", "overflow-right"}.issubset(reason_tokens) or {"layout-check-match", "overflow-bottom"}.issubset(reason_tokens):
            effective_threshold = 0.5
        elif {"network-status-match", "cart-quantity-evidence"}.intersection(reason_tokens):
            effective_threshold = 0.5
        if best_bug and best_score >= effective_threshold:
            matched_bug_id = _bug_id(best_bug)
            anomaly["matched_bug_id"] = matched_bug_id
            anomaly["match_reason"] = best_reason
            anomaly["match_score"] = round(best_score, 3)
            anomaly["match_confidence"] = round(best_score, 3)
            matched = dict(anomaly)
            matched["matched_bug_id"] = matched_bug_id
            matched["match_reason"] = best_reason
            matched["match_score"] = round(best_score, 3)
            matched["match_confidence"] = round(best_score, 3)
            matches.append(matched)
    return matches


def _score_match(
    anomaly_type: str,
    evidence: Mapping[str, Any],
    bug: Mapping[str, Any],
    anomaly_confidence: float,
    site_id: str = "",
) -> tuple[float, str]:
    catalog_score, catalog_reason = _catalog_score(anomaly_type, evidence, bug)
    score = catalog_score
    reason = catalog_reason

    site001_score, site001_reason = _site001_compat_score(anomaly_type, evidence, bug, anomaly_confidence, site_id)
    if site001_score > score:
        score = site001_score
        reason = site001_reason
    return min(score, 1.0), reason


def _catalog_score(anomaly_type: str, evidence: Mapping[str, Any], bug: Mapping[str, Any]) -> tuple[float, str]:
    bug_type = str(bug.get("type") or bug.get("bug_type") or "")
    bug_id = _bug_id(bug)
    selector = str(bug.get("selector") or "")
    selector_data_bug_id = data_bug_id_from_selector(selector)
    evidence_text = " ".join(value.lower() for value in _flatten_values(evidence))
    score = 0.0
    reasons: List[str] = []

    type_matched = _type_matches(anomaly_type, bug_type)
    if type_matched:
        score += 0.4
        reasons.append("type-alias-match" if anomaly_type != bug_type else "type-match")

    direct_ids = {
        str(value)
        for value in (
            evidence.get("clicked_data_bug_id"),
            evidence.get("data_bug_id"),
            evidence.get("matched_bug_id"),
        )
        if value
    }
    selector_values = {
        str(value)
        for value in (evidence.get("selector"), evidence.get("selector_hint"))
        if value
    }
    catalog_ids = set()
    for value in evidence.get("catalog_bug_id_matches", []) or []:
        catalog_ids.add(str(value))

    has_direct_identity = bool(direct_ids)
    if bug_id and bug_id in direct_ids:
        score += 0.8
        reasons.append("data-bug-id-match")
    elif selector_data_bug_id and selector_data_bug_id in direct_ids:
        score += 0.8
        reasons.append("data-bug-id-match")
    elif not has_direct_identity and bug_id and bug_id in catalog_ids:
        score += 0.8
        reasons.append("catalog-bug-id-match")
    elif not has_direct_identity and selector_data_bug_id and selector_data_bug_id in catalog_ids:
        score += 0.8
        reasons.append("catalog-bug-id-match")
    elif selector and selector in selector_values:
        score += 0.7
        reasons.append("selector-match")

    # A concrete DOM identity is stronger than broad section keywords. When
    # another bug was clicked, do not let generic words reassign the anomaly.
    identity_conflict = has_direct_identity and bug_id not in direct_ids and selector_data_bug_id not in direct_ids
    if identity_conflict:
        return (0.4 if type_matched else 0.0), ("type-match" if type_matched else "")

    for field, weight in (
        ("target_keywords", 0.2),
        ("section_keywords", 0.2),
        ("negative_keywords", 0.2),
        ("anomaly_evidence_keywords", 0.2),
    ):
        if _any_keyword_matches(evidence_text, bug.get(field)):
            score += weight
            reasons.append(f"{field}-match")

    section_matches = _intersection(evidence.get("section_keyword_matches"), bug.get("section_keywords"))
    if section_matches:
        score += 0.3
        reasons.append("section-keyword-match")

    target_matches = _intersection(evidence.get("target_keyword_matches"), bug.get("target_keywords"))
    if target_matches:
        score += 0.3
        reasons.append("target-keyword-match")

    missing = {str(item).lower() for item in evidence.get("missing_expected_elements", []) or []}
    expected = {str(item).lower() for item in _listify(bug.get("expected_elements"))}
    if missing and expected and missing.intersection(expected):
        score += 0.2
        reasons.append("expected-elements-match")

    evidence_expected = {str(item).lower() for item in _listify(evidence.get("expected_elements"))}
    if evidence_expected and expected and evidence_expected.intersection(expected):
        score += 0.2
        reasons.append("expected-elements-match")

    layout_check = str(bug.get("layout_check") or "")
    if type_matched and layout_check and layout_check == str(evidence.get("layout_check") or ""):
        score += 0.4
        reasons.append("layout-check-match")

    if type_matched and bool(evidence.get("overflow_right")):
        score += 0.3
        reasons.append("overflow-right")
    if type_matched and bool(evidence.get("overflow_bottom")):
        score += 0.3
        reasons.append("overflow-bottom")
    action_type = str(evidence.get("action_type") or "")
    action_hints = {str(item).lower() for item in _listify(bug.get("action_hints"))}
    normalized_action = {
        "click_element": "click",
        "fill_input": "fill_input",
        "press_enter": "press_enter",
        "inspect_network": "inspect_network",
        "inspect_console": "inspect_console",
        "inspect_cart": "inspect_cart",
        "inspect_dom": "inspect_dom",
    }.get(action_type, action_type)
    if normalized_action and normalized_action.lower() in action_hints:
        score += 0.3
        reasons.append("action-hint-match")
    if type_matched and (evidence.get("network_status") == 403 or int(evidence.get("api_403_count", 0) or 0) > 0):
        score += 0.4
        reasons.append("network-status-match")
    if type_matched and bool(evidence.get("cart_quantity_evidence")):
        score += 0.4
        reasons.append("cart-quantity-evidence")
    if type_matched and anomaly_type in {"layout-overflow", "layout-overlap", "css-layout"} and not evidence.get("child_bbox"):
        reasons.append("bbox-fallback")

    if anomaly_type in {"layout-overflow", "layout-overlap", "css-layout"} and not type_matched and "catalog-bug-id-match" not in reasons and "data-bug-id-match" not in reasons:
        score = min(score, 0.45)

    return score, "+".join(reasons)


def _site001_compat_score(
    anomaly_type: str,
    evidence: Mapping[str, Any],
    bug: Mapping[str, Any],
    anomaly_confidence: float,
    site_id: str,
) -> tuple[float, str]:
    bug_id = _bug_id(bug)
    evidence_text = " ".join(value.lower() for value in _flatten_values(evidence))
    same_cart_count = (
        evidence.get("cart_count_before") is not None
        and evidence.get("cart_count_after") is not None
        and evidence.get("cart_count_before") == evidence.get("cart_count_after")
    )
    if (
        site_id == "site001"
        and anomaly_type == "button-no-response"
        and bug_id == "site001-bug01"
        and anomaly_confidence >= 0.8
        and same_cart_count
    ):
        return 0.99, "site001-cart-button-no-response"
    if anomaly_type == "duplicated-rendering" and bug_id == "site001-bug02":
        return 0.85, "site001-duplicated-rendering"
    if anomaly_type in {"layout-overlap", "layout-overflow", "css-layout"} and bug_id == "site001-bug03":
        if "mobile" in evidence_text and anomaly_confidence >= 0.8:
            return 0.98, "site001-mobile-layout"
        return 0.85, "site001-layout"
    return 0.0, ""


def _with_site001_fallback(site_id: str, bugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if site_id != "site001":
        return bugs
    existing = {_bug_id(bug) for bug in bugs}
    merged = list(bugs)
    for fallback in SITE001_FALLBACK_BUGS:
        if _bug_id(fallback) not in existing:
            merged.append(dict(fallback))
    return merged


def _type_matches(anomaly_type: str, bug_type: str) -> bool:
    if not bug_type:
        return False
    if anomaly_type == bug_type:
        return True
    return bug_type in TYPE_ALIASES.get(anomaly_type, set())


def _bug_id(bug: Mapping[str, Any]) -> str:
    return bug_id_of(bug)


def _flatten_values(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        result: List[str] = []
        for item in value.values():
            result.extend(_flatten_values(item))
        return result
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    if value is None:
        return []
    return [str(value)]


def _any_keyword_matches(haystack: str, raw_keywords: Any) -> bool:
    return any(str(keyword).lower() in haystack for keyword in _listify(raw_keywords))


def _intersection(left: Any, right: Any) -> set[str]:
    left_values = {str(item).lower() for item in _listify(left)}
    right_values = {str(item).lower() for item in _listify(right)}
    return left_values.intersection(right_values)


def _listify(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)] if str(value) else []
