"""Catalog-driven site profile helpers for BrowserGym training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


KEYWORD_FIELDS = (
    "target_keywords",
    "section_keywords",
    "negative_keywords",
    "expected_elements",
    "anomaly_evidence_keywords",
    "action_hints",
)

OPENENDED_PROFILES: Dict[str, Dict[str, Any]] = {
    "openended_commerce": {
        "mode": "openended_anomaly_discovery",
        "interaction_keywords": [
            "sign in",
            "login",
            "로그인",
            "cart",
            "basket",
            "장바구니",
            "checkout",
            "payment",
            "결제",
            "buy",
            "purchase",
            "add to cart",
            "search",
            "검색",
            "filter",
            "sort",
            "submit",
            "save",
            "next",
            "continue",
            "view",
            "details",
            "add",
        ],
        "form_keywords": [
            "email",
            "password",
            "username",
            "customer",
            "address",
            "phone",
            "coupon",
            "search",
        ],
        "priority_actions": [
            "click_element",
            "inspect_layout",
            "inspect_dom",
            "change_viewport_mobile",
        ],
        "avoid_repeating_actions": True,
    },
    "openended_generic": {
        "mode": "openended_anomaly_discovery",
        "interaction_keywords": [
            "login",
            "sign in",
            "search",
            "submit",
            "save",
            "continue",
            "next",
            "add",
            "details",
            "filter",
            "sort",
        ],
        "form_keywords": ["email", "password", "username", "search", "name"],
        "priority_actions": ["click_element", "inspect_layout", "inspect_dom", "change_viewport_mobile"],
        "avoid_repeating_actions": True,
    },
}


def load_bug_catalog(site_id: str, catalog_path: str | Path | None = None) -> List[Dict[str, Any]]:
    path = Path(catalog_path) if catalog_path else _find_catalog(site_id)
    if path is None or not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        bugs = raw.get("bugs") or raw.get("items") or raw.get("bug_catalog") or raw.get("known_bugs") or []
        if isinstance(bugs, list):
            return [dict(item) for item in bugs if isinstance(item, Mapping)]
    return []


def load_training_site_config(
    site_id: str,
    config_path: str | Path = "configs/training_sites.json",
) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sites = raw.get("sites", []) if isinstance(raw, Mapping) else []
    if not isinstance(sites, list):
        return {}
    for item in sites:
        if isinstance(item, Mapping) and str(item.get("site_id") or "") == site_id:
            return dict(item)
    return {}


def build_site_profile(
    site_id: str,
    bugs: Optional[Iterable[Mapping[str, Any]]] = None,
    catalog_path: str | Path | None = None,
    exploration_profile: str | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    bug_items = [dict(item) for item in (bugs if bugs is not None else load_bug_catalog(site_id, catalog_path))]
    openended_profile = _resolve_openended_profile(exploration_profile)
    profile: Dict[str, Any] = {
        "site_id": site_id,
        "bugs": bug_items,
        "selectors": [],
        "data_bug_ids": [],
        "bug_types": [],
        "mode": openended_profile.get("mode", "known_bug_catalog" if bug_items else "generic"),
        "exploration_profile": openended_profile.get("name", ""),
        "interaction_keywords": [],
        "form_keywords": [],
        "priority_actions": [],
        "avoid_repeating_actions": bool(openended_profile.get("avoid_repeating_actions", False)),
    }
    for field in KEYWORD_FIELDS:
        profile[field] = []
    for field in ("interaction_keywords", "form_keywords", "priority_actions"):
        for value in _listify(openended_profile.get(field)):
            _append_unique(profile[field], value)
    for bug in bug_items:
        selector = str(bug.get("selector") or "")
        bug_id = bug_id_of(bug)
        bug_type = str(bug.get("type") or bug.get("bug_type") or "")
        if selector:
            _append_unique(profile["selectors"], selector)
            data_bug_id = data_bug_id_from_selector(selector)
            if data_bug_id:
                _append_unique(profile["data_bug_ids"], data_bug_id)
        if bug_id:
            _append_unique(profile["data_bug_ids"], bug_id)
        if bug_type:
            _append_unique(profile["bug_types"], bug_type)
        for field in KEYWORD_FIELDS:
            for value in _listify(bug.get(field)):
                _append_unique(profile[field], value)
    return profile


def _resolve_openended_profile(profile: str | Mapping[str, Any] | None) -> Dict[str, Any]:
    if isinstance(profile, Mapping):
        result = dict(profile)
        result.setdefault("name", str(result.get("profile") or "custom_openended"))
        return result
    profile_name = str(profile or "").strip()
    if not profile_name:
        return {}
    result = dict(OPENENDED_PROFILES.get(profile_name, OPENENDED_PROFILES["openended_generic"]))
    result["name"] = profile_name
    return result


def validate_site_identity(site_id: str, observation: Mapping[str, Any]) -> Dict[str, Any]:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    data_bug_ids = page_state.get("data_bug_ids_found", []) if isinstance(page_state, Mapping) else []
    if not isinstance(data_bug_ids, list):
        data_bug_ids = []
    observed = [str(value) for value in data_bug_ids if str(value)]
    expected_prefix = f"{site_id}-bug" if site_id else ""
    mismatched = [value for value in observed if expected_prefix and not value.startswith(expected_prefix)]
    matched = [value for value in observed if expected_prefix and value.startswith(expected_prefix)]
    identity_match = True
    warnings: List[str] = []
    if observed and mismatched:
        identity_match = False
        warnings.append(
            "WARNING: observed data-bug-id values do not match requested site_id "
            f"{site_id}. Expected prefix: {expected_prefix}. Observed: {', '.join(observed[:12])}. "
            "This usually means base_url is mapped to the wrong website."
        )
    return {
        "requested_site_id": site_id,
        "expected_bug_id_prefix": expected_prefix,
        "data_bug_ids_found": observed,
        "matching_data_bug_ids": matched,
        "mismatched_data_bug_ids": mismatched,
        "site_identity_match": identity_match,
        "identity_warnings": warnings,
    }


def match_candidate_to_profile(candidate: Mapping[str, Any], site_profile: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not site_profile:
        return {"keyword_matches": [], "selector_match": False, "bug_id_matches": []}
    text = _candidate_haystack(candidate)
    keyword_pool = []
    for field in ("target_keywords", "section_keywords", "negative_keywords", "expected_elements"):
        keyword_pool.extend(_listify(site_profile.get(field)))
    keyword_matches = sorted({kw for kw in keyword_pool if _keyword_in_text(kw, text)})
    data_bug_id = str(candidate.get("data_bug_id") or "")
    selector_hint = str(candidate.get("selector_hint") or "")
    bug_id_matches: List[str] = []
    selector_match = False
    for bug in site_profile.get("bugs", []) or []:
        if not isinstance(bug, Mapping):
            continue
        bug_id = bug_id_of(bug)
        selector = str(bug.get("selector") or "")
        selector_data_bug_id = data_bug_id_from_selector(selector)
        if bug_id and data_bug_id and data_bug_id in {bug_id, selector_data_bug_id}:
            _append_unique(bug_id_matches, bug_id)
            selector_match = True
        elif selector and selector_hint and selector == selector_hint:
            _append_unique(bug_id_matches, bug_id)
            selector_match = True
        else:
            bug_keywords = []
            for field in ("target_keywords", "section_keywords", "negative_keywords", "expected_elements"):
                bug_keywords.extend(_listify(bug.get(field)))
            if bug_id and any(_keyword_in_text(keyword, text) for keyword in bug_keywords):
                _append_unique(bug_id_matches, bug_id)
    return {
        "keyword_matches": keyword_matches,
        "selector_match": selector_match,
        "bug_id_matches": bug_id_matches,
    }


def profile_keyword_matches(text: str, site_profile: Optional[Mapping[str, Any]], field: str) -> List[str]:
    if not site_profile:
        return []
    return sorted({kw for kw in _listify(site_profile.get(field)) if _keyword_in_text(kw, text)})


def bug_id_of(bug: Mapping[str, Any]) -> str:
    return str(bug.get("bug_id") or bug.get("id") or bug.get("name") or "")


def data_bug_id_from_selector(selector: str) -> str:
    match = re.search(r'data-bug-id\s*=\s*["\']([^"\']+)["\']', str(selector or ""))
    return match.group(1) if match else ""


def _find_catalog(site_id: str) -> Path | None:
    for path in (
        Path("datasets") / site_id / "bug_catalog.json",
        Path("data") / site_id / "bug_catalog.json",
        Path("sites") / site_id / "bug_catalog.json",
    ):
        if path.exists():
            return path
    return None


def _candidate_haystack(candidate: Mapping[str, Any]) -> str:
    keys = (
        "text",
        "name",
        "label",
        "aria_label",
        "title",
        "id",
        "class_name",
        "data_bug_id",
        "selector_hint",
        "tag",
        "role",
    )
    return " ".join(str(candidate.get(key, "") or "") for key in keys).lower()


def _keyword_in_text(keyword: Any, text: str) -> bool:
    keyword_text = str(keyword or "").strip().lower()
    return bool(keyword_text and keyword_text in text)


def _listify(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)] if str(value) else []


def _append_unique(items: List[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)
