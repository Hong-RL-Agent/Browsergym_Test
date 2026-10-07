"""Discrete J.A.W.S action space utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


class ActionSpace:
    action_types = [
        "noop",
        "click_element",
        "scroll_down",
        "scroll_up",
        "inspect_dom",
        "inspect_layout",
        "inspect_network",
        "inspect_console",
        "inspect_cart",
        "inspect_server_health",
        "inspect_port_status",
        "inspect_latency",
        "inspect_server_logs",
        "inspect_runtime_metrics",
        "fill_input",
        "press_enter",
        "change_viewport_mobile",
        "change_viewport_desktop",
        "finish_episode",
    ]

    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = max_candidates
        self._type_to_id = {name: idx for idx, name in enumerate(self.action_types)}

    def encode(self, action_type: str, candidate_index: int = 0) -> int:
        if action_type not in self._type_to_id:
            raise ValueError(f"Unknown action_type: {action_type}")
        if not 0 <= candidate_index < self.max_candidates:
            raise ValueError(f"candidate_index must be in [0, {self.max_candidates})")
        return self._type_to_id[action_type] * self.max_candidates + candidate_index

    def decode(self, action_id: int) -> Dict[str, Any]:
        action_id = int(action_id)
        if not 0 <= action_id < self.get_action_dim():
            raise ValueError(f"action_id must be in [0, {self.get_action_dim()})")
        action_type_id = action_id // self.max_candidates
        candidate_index = action_id % self.max_candidates
        return {
            "action_type_id": action_type_id,
            "action_type": self.action_types[action_type_id],
            "candidate_index": candidate_index,
        }

    def is_element_action(self, action_type: str) -> bool:
        return action_type in {"click_element", "fill_input", "press_enter"}

    def is_infra_action(self, action_type: str) -> bool:
        return action_type in {
            "inspect_server_health",
            "inspect_port_status",
            "inspect_latency",
            "inspect_server_logs",
            "inspect_runtime_metrics",
        }

    def get_action_dim(self) -> int:
        return len(self.action_types) * self.max_candidates

    def build_action_mask(self, raw_observation: Mapping[str, Any]) -> np.ndarray:
        mask = np.zeros(self.get_action_dim(), dtype=np.float32)
        candidates = raw_observation.get("candidate_elements", []) if raw_observation else []
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        page_state = raw_observation.get("page_state", {}) if isinstance(raw_observation, Mapping) else {}
        runtime_signals = raw_observation.get("runtime_signals", {}) if isinstance(raw_observation, Mapping) else {}
        infra_signals = raw_observation.get("infra_signals", {}) if isinstance(raw_observation, Mapping) else {}
        infra_enabled = _is_infra_port(page_state, infra_signals)

        for action_type in self.action_types:
            if self.is_infra_action(action_type) and not infra_enabled:
                continue
            if action_type == "finish_episode":
                history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
                if int(history.get("step_index", 0) or 0) < 5:
                    continue
            if self.is_element_action(action_type):
                for idx in range(min(candidate_count, self.max_candidates)):
                    candidate = candidates[idx] if isinstance(candidates, list) else {}
                    if isinstance(candidate, Mapping) and _is_executable_candidate_action(action_type, candidate):
                        mask[self.encode(action_type, idx)] = 1.0
            else:
                mask[self.encode(action_type, 0)] = 1.0

        if not mask.any():
            mask[self.encode("noop", 0)] = 1.0
        return mask

    def build_action_mask_stats(
        self,
        raw_observation: Mapping[str, Any],
        action_mask: np.ndarray,
        selected_action_id: int | None = None,
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        valid_type_count = 0
        for action_type in self.action_types:
            count = 0
            for index in range(self.max_candidates):
                action_id = self.encode(action_type, index)
                if action_id < len(action_mask) and float(action_mask[action_id]) > 0.0:
                    count += 1
            by_type[action_type] = count
            if count:
                valid_type_count += 1
        selected = self.decode(int(selected_action_id)) if selected_action_id is not None else {}
        return {
            "valid_action_count": int(np.count_nonzero(action_mask)),
            "valid_action_type_count": valid_type_count,
            "valid_actions_by_type": by_type,
            "masked_action_count": int(len(action_mask) - np.count_nonzero(action_mask)),
            "selected_action_index": int(selected_action_id) if selected_action_id is not None else None,
            "selected_action_type": selected.get("action_type", ""),
            "selected_candidate_index": selected.get("candidate_index", None),
        }


def _is_infra_port(page_state: Mapping[str, Any], infra_signals: Any) -> bool:
    port = None
    if isinstance(infra_signals, Mapping):
        port = infra_signals.get("port")
    if port in (None, ""):
        url = str(page_state.get("url") or "")
        for token in url.split(":")[2:3]:
            digits = "".join(char for char in token if char.isdigit())
            if digits:
                port = digits
                break
    try:
        value = int(port)
    except (TypeError, ValueError):
        return False
    return 9000 <= value <= 9100


def _is_executable_candidate_action(action_type: str, candidate: Mapping[str, Any]) -> bool:
    if action_type == "click_element":
        return _is_clickable_candidate(candidate)
    if action_type in {"fill_input", "press_enter"}:
        return _is_input_candidate(candidate)
    return False


def _is_clickable_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_visible_enabled(candidate):
        return False
    if not _has_executable_target_identity(candidate):
        return False
    if not _has_bbox(candidate.get("bbox")):
        return False
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    source = str(candidate.get("source") or "")
    semantic_clickable = bool(
        candidate.get("clickable")
        or candidate.get("set_of_marks")
        or role in {"button", "link", "checkbox", "radio", "menuitem", "tab"}
        or tag in {"button", "a", "summary", "option", "label"}
    )
    if not semantic_clickable:
        return False
    if source in {"bbox_fallback", "emergency_bbox_fallback", "fullscreen_clickable_fallback"}:
        return bool(candidate.get("clickable") or candidate.get("set_of_marks"))
    return True


def _is_input_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_visible_enabled(candidate):
        return False
    if not _has_executable_target_identity(candidate):
        return False
    if not _has_bbox(candidate.get("bbox")):
        return False
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
    if input_type in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"}:
        return False
    return bool(candidate.get("fillable") or tag in {"input", "textarea"} or role in {"textbox", "searchbox", "combobox"})


def _is_visible_enabled(candidate: Mapping[str, Any]) -> bool:
    if not bool(candidate.get("visible", True)):
        return False
    if not bool(candidate.get("enabled", True)):
        return False
    if bool(candidate.get("hidden") or candidate.get("disabled") or candidate.get("aria_hidden")):
        return False
    return _safe_visibility(candidate) > 0.0


def _has_executable_target_identity(candidate: Mapping[str, Any]) -> bool:
    return bool(str(candidate.get("bid") or "").strip())


def _has_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        return float(value[2] or 0.0) > 0.0 and float(value[3] or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _safe_visibility(candidate: Mapping[str, Any]) -> float:
    if "visibility" in candidate and candidate.get("visibility") is not None:
        try:
            value = float(candidate.get("visibility") or 0.0)
        except (TypeError, ValueError):
            value = 1.0
    elif "visible" in candidate:
        value = 1.0 if bool(candidate.get("visible")) else 0.0
    else:
        value = 1.0
    if value != value:
        return 1.0
    return max(0.0, min(1.0, value))
