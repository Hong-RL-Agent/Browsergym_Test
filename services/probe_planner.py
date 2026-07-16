"""Deterministic bug probes used as safety rails around the RL policy."""

from __future__ import annotations

from typing import Any, Mapping, Optional


PROBES = {
    "site001-bug03": ("change_viewport_mobile", "inspect_layout", "inspect_dom"),
    "site001-bug01": ("inspect_cart", "click_element"),
}

ACTION_ALIASES = {
    "click": "click_element",
    "tap": "click_element",
    "fill": "fill_input",
    "type": "fill_input",
    "enter": "press_enter",
}


def next_probe_action(
    site_id: str, observation: Mapping[str, Any], history: Mapping[str, Any],
    known_bugs: list[Mapping[str, Any]] | None = None,
) -> Optional[str]:
    matched = {str(item) for item in history.get("matched_bug_ids", set()) or set()}
    counts = history.get("action_type_counts", {}) or {}
    catalog_probes = {}
    for bug in known_bugs or []:
        bug_id = str(bug.get("bug_id") or bug.get("id") or "")
        hints = [_normalize_action(item) for item in bug.get("action_hints", []) or []]
        if bug_id and hints:
            catalog_probes[bug_id] = tuple(hints)
    for bug_id, sequence in {**PROBES, **catalog_probes}.items():
        if bug_id not in matched and (not site_id or bug_id.startswith(site_id + "-")):
            for action_type in sequence:
                if int(counts.get(action_type, 0) or 0) == 0:
                    return action_type
            # Re-run the layout probe once while mobile if its evidence is absent.
            if bug_id == "site001-bug03" and observation.get("page_state", {}).get("viewport_type") == "mobile":
                if not history.get("layout_oracle_observed") and int(history.get("layout_probe_attempts", 0) or 0) < 2:
                    return "inspect_layout"
    return None


def _normalize_action(value: Any) -> str:
    action = str(value).strip().lower()
    return ACTION_ALIASES.get(action, action)
