"""Versioned 20-action space for RAWD Hybrid v3."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .action_contract import ACTION_CONTRACTS, V3_ACTION_TYPES, is_target_eligible


class V3ActionSpace:
    action_types = list(V3_ACTION_TYPES)

    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = int(max_candidates)
        self._type_to_id = {name: index for index, name in enumerate(self.action_types)}

    def encode(self, action_type: str, candidate_index: int = 0) -> int:
        if action_type not in self._type_to_id:
            raise ValueError(f"Unknown action_type: {action_type}")
        if not 0 <= candidate_index < self.max_candidates:
            raise ValueError("candidate_index out of range")
        return self._type_to_id[action_type] * self.max_candidates + candidate_index

    def decode(self, action_id: int) -> dict[str, Any]:
        action_id = int(action_id)
        if not 0 <= action_id < self.get_action_dim():
            raise ValueError(f"action_id must be in [0, {self.get_action_dim()})")
        return {
            "action_type_id": action_id // self.max_candidates,
            "action_type": self.action_types[action_id // self.max_candidates],
            "candidate_index": action_id % self.max_candidates,
        }

    def get_action_dim(self) -> int:
        return len(self.action_types) * self.max_candidates

    def is_element_action(self, action_type: str) -> bool:
        contract = ACTION_CONTRACTS.get(action_type)
        return bool(contract and contract.requires_target)

    def is_infra_action(self, action_type: str) -> bool:
        return action_type in {"inspect_server_health", "inspect_port_status", "inspect_latency", "inspect_server_logs", "inspect_runtime_metrics"}

    def build_action_mask(self, raw_observation: Mapping[str, Any]) -> np.ndarray:
        mask = np.zeros(self.get_action_dim(), dtype=np.float32)
        candidates = raw_observation.get("candidate_elements", []) if raw_observation else []
        candidates = candidates if isinstance(candidates, list) else []
        page_state = raw_observation.get("page_state", {}) if isinstance(raw_observation, Mapping) else {}
        infra = raw_observation.get("infra_signals", {}) if isinstance(raw_observation, Mapping) else {}
        port = infra.get("port") if isinstance(infra, Mapping) else None
        infra_enabled = bool(port) or "localhost" in str(page_state.get("url") or "")
        for action_type in self.action_types:
            if self.is_infra_action(action_type) and not infra_enabled:
                continue
            if action_type == "finish_episode" and int((raw_observation.get("history", {}) or {}).get("step_index", 0) or 0) < 5:
                continue
            if self.is_element_action(action_type):
                for index, candidate in enumerate(candidates[: self.max_candidates]):
                    if isinstance(candidate, Mapping) and is_target_eligible(action_type, candidate):
                        mask[self.encode(action_type, index)] = 1.0
            else:
                mask[self.encode(action_type, 0)] = 1.0
        if not mask.any():
            mask[self.encode("noop", 0)] = 1.0
        return mask
