"""Discrete J.A.W.S action space utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


class ActionSpace:
    action_aliases = {
        "click": "click_element",
        "tap": "click_element",
        "fill": "fill_input",
        "type": "fill_input",
        "enter": "press_enter",
    }
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
        action_type = self.action_aliases.get(str(action_type).strip().lower(), action_type)
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
                    if isinstance(candidate, Mapping) and candidate.get("catalog_bug_id_matches"):
                        mask[self.encode(action_type, idx)] = 1.0
                    elif isinstance(candidate, Mapping) and bool(candidate.get("visible", True)):
                        mask[self.encode(action_type, idx)] = 1.0
                    elif site_id == "site001":
                        mask[self.encode(action_type, idx)] = 1.0
            else:
                mask[self.encode(action_type, 0)] = 1.0

        if not mask.any():
            mask[self.encode("noop", 0)] = 1.0
        return mask


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
