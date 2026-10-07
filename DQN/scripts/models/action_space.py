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
        return action_type == "click_element"

    def get_action_dim(self) -> int:
        return len(self.action_types) * self.max_candidates

    def build_action_mask(self, raw_observation: Mapping[str, Any]) -> np.ndarray:
        mask = np.zeros(self.get_action_dim(), dtype=np.float32)
        candidates = raw_observation.get("candidate_elements", []) if raw_observation else []
        candidate_count = len(candidates) if isinstance(candidates, list) else 0

        for action_type in self.action_types:
            if self.is_element_action(action_type):
                for idx in range(min(candidate_count, self.max_candidates)):
                    mask[self.encode(action_type, idx)] = 1.0
            else:
                mask[self.encode(action_type, 0)] = 1.0

        return mask
