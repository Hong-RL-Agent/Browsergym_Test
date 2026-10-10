"""Single source of truth for v3 mask eligibility and executor support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from models.action_types import CANONICAL_ACTION_TYPES


@dataclass(frozen=True)
class ActionContract:
    action_type: str
    requires_target: bool
    executor: str


V3_ACTION_TYPES = CANONICAL_ACTION_TYPES

ACTION_CONTRACTS = {
    action_type: ActionContract(action_type, action_type in {"click_element", "fill_input", "select_option", "press_enter"}, action_type)
    for action_type in V3_ACTION_TYPES
}


def _visible_enabled(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("visible", True)) and bool(candidate.get("enabled", True)) and not bool(
        candidate.get("hidden") or candidate.get("disabled") or candidate.get("aria_hidden")
    )


def _has_identity_and_bbox(candidate: Mapping[str, Any]) -> bool:
    bbox = candidate.get("bbox")
    return bool(str(candidate.get("bid") or "").strip()) and isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and float(bbox[2] or 0) > 0 and float(bbox[3] or 0) > 0


def is_native_select(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("canonical_dom_resolved")) and str(candidate.get("canonical_dom_tag") or "").lower() == "select"


def is_fill_input_target(candidate: Mapping[str, Any]) -> bool:
    if not _visible_enabled(candidate) or not _has_identity_and_bbox(candidate):
        return False
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
    if input_type in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"}:
        return False
    if is_native_select(candidate) or role == "combobox":
        return False
    return bool(candidate.get("fillable") or tag in {"input", "textarea"} or role in {"textbox", "searchbox"} or candidate.get("contenteditable"))


def is_select_option_target(candidate: Mapping[str, Any]) -> bool:
    return _visible_enabled(candidate) and _has_identity_and_bbox(candidate) and is_native_select(candidate)


def is_click_target(candidate: Mapping[str, Any]) -> bool:
    if not _visible_enabled(candidate) or not _has_identity_and_bbox(candidate):
        return False
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return bool(candidate.get("clickable") or candidate.get("set_of_marks") or role in {"button", "link", "checkbox", "radio", "menuitem", "tab", "option"} or tag in {"button", "a", "summary", "option", "label"})


def is_target_eligible(action_type: str, candidate: Mapping[str, Any]) -> bool:
    if action_type == "click_element":
        return is_click_target(candidate)
    if action_type == "fill_input":
        return is_fill_input_target(candidate)
    if action_type == "select_option":
        return is_select_option_target(candidate)
    if action_type == "press_enter":
        return is_fill_input_target(candidate) or is_select_option_target(candidate)
    return False
