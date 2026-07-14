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
        "inspect_network_status",
        "inspect_api_response",
        "inspect_console_errors",
        "inspect_resource_loading",
        "inspect_alert_card",
        "inspect_metric_card",
        "inspect_timeline",
        "click_trigger_button",
        "click_retry_button",
        "click_recovery_button",
        "open_detail_panel",
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
        return action_type in {
            "click_element",
            "fill_input",
            "press_enter",
            "click_trigger_button",
            "click_retry_button",
            "click_recovery_button",
            "open_detail_panel",
        }

    def is_click_action(self, action_type: str) -> bool:
        return action_type in {
            "click_element",
            "click_trigger_button",
            "click_retry_button",
            "click_recovery_button",
            "open_detail_panel",
        }

    def is_infra_action(self, action_type: str) -> bool:
        return action_type in {
            "inspect_server_health",
            "inspect_port_status",
            "inspect_latency",
            "inspect_server_logs",
            "inspect_runtime_metrics",
            "inspect_network_status",
            "inspect_api_response",
            "inspect_console_errors",
            "inspect_resource_loading",
            "inspect_alert_card",
            "inspect_metric_card",
            "inspect_timeline",
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
        site_id = str(page_state.get("site_id") or runtime_signals.get("site_id") or "")
        if not site_id:
            url = str(page_state.get("url") or "")
            site_id = "site003" if ":9221" in url else "site001" if ":9220" in url else ""
        infra_enabled = _is_infra_port(page_state, infra_signals)

        for action_type in self.action_types:
            if site_id and site_id not in {"site001", "site9800"} and action_type == "inspect_cart":
                continue
            if self.is_infra_action(action_type) and not infra_enabled:
                continue
            if action_type == "finish_episode":
                history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
                if int(history.get("step_index", 0) or 0) < 5:
                    continue
            if self.is_element_action(action_type):
                for idx in range(min(candidate_count, self.max_candidates)):
                    candidate = candidates[idx] if isinstance(candidates, list) else {}
                    if not isinstance(candidate, Mapping):
                        continue
                    if action_type in {"fill_input", "press_enter"} and not _is_input_candidate(candidate):
                        continue
                    if self.is_click_action(action_type) and not _is_clickable_candidate(candidate):
                        continue
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
        selected_action_id: int,
    ) -> Dict[str, Any]:
        candidates = raw_observation.get("candidate_elements", []) if raw_observation else []
        if not isinstance(candidates, list):
            candidates = []
        selected = self.decode(int(selected_action_id))
        selected_index = int(selected.get("candidate_index", 0) or 0)
        selected_candidate = candidates[selected_index] if 0 <= selected_index < len(candidates) else {}
        if not isinstance(selected_candidate, Mapping):
            selected_candidate = {}

        valid_click_count = 0
        valid_fill_count = 0
        valid_global_action_count = 0
        for action_type in self.action_types:
            if self.is_click_action(action_type):
                valid_click_count += self._valid_count_for_type(action_mask, action_type)
            elif action_type == "fill_input":
                valid_fill_count += self._valid_count_for_type(action_mask, action_type)
            elif not self.is_element_action(action_type):
                action_id = self.encode(action_type, 0)
                if action_id < len(action_mask) and float(action_mask[action_id]) > 0.0:
                    valid_global_action_count += 1

        return {
            "candidate_count": min(len(candidates), self.max_candidates),
            "valid_action_count": int(np.count_nonzero(action_mask)),
            "valid_click_count": valid_click_count,
            "valid_fill_count": valid_fill_count,
            "valid_global_action_count": valid_global_action_count,
            "selected_action_type": str(selected.get("action_type") or ""),
            "selected_candidate_index": selected_index,
            "selected_candidate_role": str(selected_candidate.get("role") or ""),
            "selected_candidate_tag": str(selected_candidate.get("tag") or ""),
            "selected_candidate_visible": bool(selected_candidate.get("visible")),
            "selected_candidate_enabled": bool(selected_candidate.get("enabled", True)),
            "selected_candidate_clickable": bool(selected_candidate.get("clickable")),
        }

    def _valid_count_for_type(self, action_mask: np.ndarray, action_type: str) -> int:
        start = self.encode(action_type, 0)
        end = start + self.max_candidates
        return int(np.count_nonzero(action_mask[start:end]))


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


def _is_clickable_candidate(candidate: Mapping[str, Any]) -> bool:
    return _is_visible_enabled(candidate) and bool(candidate.get("clickable"))


def _is_input_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_visible_enabled(candidate):
        return False
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
    if input_type in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"}:
        return False
    if tag in {"input", "textarea"}:
        return True
    return role in {"textbox", "searchbox"}


def _is_visible_enabled(candidate: Mapping[str, Any]) -> bool:
    visibility = _safe_visibility(candidate)
    if not bool(candidate.get("visible", visibility > 0.0)):
        return False
    if not bool(candidate.get("enabled", True)):
        return False
    if visibility <= 0.0:
        return False
    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            width = float(bbox[2] or 0.0)
            height = float(bbox[3] or 0.0)
        except (TypeError, ValueError):
            return False
        if width <= 0.0 or height <= 0.0:
            return False
    return True


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
