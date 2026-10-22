"""Explicit legacy/canonical DQN contracts and strict checkpoint guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from models.action_types import CANONICAL_ACTION_SPACE_VERSION, CANONICAL_ACTION_TYPES, action_space_mapping_hash


LEGACY_ACTION_TYPES = (
    "noop", "click_element", "scroll_down", "scroll_up", "inspect_dom",
    "inspect_layout", "inspect_network", "inspect_console", "inspect_cart",
    "inspect_server_health", "inspect_port_status", "inspect_latency",
    "inspect_server_logs", "inspect_runtime_metrics", "fill_input",
    "press_enter", "change_viewport_mobile", "change_viewport_desktop",
    "finish_episode",
)


@dataclass(frozen=True)
class DQNContract:
    contract_version: str
    model_generation: str
    observation_dim: int
    action_types: tuple[str, ...]
    candidate_slots: int
    history_dim: int
    architecture_version: str
    observation_schema_version: str
    evaluation_only: bool
    production_compatible: bool

    @property
    def action_dim(self) -> int:
        return len(self.action_types) * self.candidate_slots

    @property
    def type_head_dim(self) -> int:
        return len(self.action_types)

    @property
    def candidate_head_shape(self) -> tuple[int, int]:
        return (len(self.action_types), self.candidate_slots)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_types"] = list(self.action_types)
        result["action_dim"] = self.action_dim
        result["type_head_dim"] = self.type_head_dim
        result["candidate_head_shape"] = list(self.candidate_head_shape)
        result["action_schema_hash"] = action_schema_hash(self.action_types, self.candidate_slots)
        result["contract_hash"] = contract_hash(result)
        return result


LEGACY_DQN_CONTRACT = DQNContract(
    contract_version="rawd_legacy_dqn_v1",
    model_generation="legacy_dqn_v1",
    observation_dim=976,
    action_types=LEGACY_ACTION_TYPES,
    candidate_slots=32,
    history_dim=269,
    architecture_version="factorized_v1_confobs_histv2",
    observation_schema_version="browsergym-ui-dqn-v1",
    evaluation_only=True,
    production_compatible=False,
)

CANONICAL_DQN_CONTRACT = DQNContract(
    contract_version="rawd_canonical_dqn_runtime_v1",
    model_generation="canonical_dqn_v1",
    observation_dim=979,
    action_types=tuple(CANONICAL_ACTION_TYPES),
    candidate_slots=32,
    history_dim=272,
    architecture_version="factorized_v1_confobs_histv2",
    observation_schema_version="rawd_observation_legacy_v1",
    evaluation_only=False,
    production_compatible=True,
)


def action_schema_hash(action_types: tuple[str, ...] | list[str], candidate_slots: int) -> str:
    payload = {"action_types": list(action_types), "candidate_slots": int(candidate_slots)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def contract_hash(contract: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in contract.items() if key not in {"contract_hash"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest().upper()


class ContractActionSpace:
    def __init__(self, contract: DQNContract) -> None:
        self.action_types = list(contract.action_types)
        self.max_candidates = contract.candidate_slots

    def encode(self, action_type: str, candidate_index: int = 0) -> int:
        if action_type not in self.action_types:
            raise ValueError(f"unknown action type: {action_type}")
        if not 0 <= int(candidate_index) < self.max_candidates:
            raise ValueError("candidate index out of range")
        return self.action_types.index(action_type) * self.max_candidates + int(candidate_index)

    def decode(self, action_id: int) -> dict[str, Any]:
        action_id = int(action_id)
        if not 0 <= action_id < len(self.action_types) * self.max_candidates:
            raise ValueError("action id out of range")
        return {"action_type_id": action_id // self.max_candidates, "action_type": self.action_types[action_id // self.max_candidates], "candidate_index": action_id % self.max_candidates}

    def get_action_dim(self) -> int:
        return len(self.action_types) * self.max_candidates


def validate_checkpoint_contract(payload: Mapping[str, Any], expected: DQNContract) -> None:
    manifest = payload.get("manifest") if isinstance(payload.get("manifest"), Mapping) else payload
    actual = {
        "observation_dim": manifest.get("dqn_obs_dim", manifest.get("observation_dim")),
        "action_dim": manifest.get("action_dim"),
        "type_head_dim": manifest.get("type_head_dim", manifest.get("num_action_types")),
        "candidate_head_shape": manifest.get("candidate_head_shape"),
        "history_dim": manifest.get("history_context_dim"),
        "architecture_version": manifest.get("dqn_architecture", manifest.get("architecture_version")),
        "action_schema_hash": manifest.get("action_schema_hash"),
        "observation_schema_hash": manifest.get("observation_schema_hash"),
    }
    expected_values = {
        "observation_dim": expected.observation_dim,
        "action_dim": expected.action_dim,
        "type_head_dim": expected.type_head_dim,
        "candidate_head_shape": list(expected.candidate_head_shape),
        "history_dim": expected.history_dim,
        "architecture_version": expected.architecture_version,
    }
    mismatch = {key: (actual[key], value) for key, value in expected_values.items() if actual[key] != value}
    expected_action_hash = action_schema_hash(expected.action_types, expected.candidate_slots)
    if actual["action_schema_hash"] is not None and actual["action_schema_hash"] != expected_action_hash:
        mismatch["action_schema_hash"] = (actual["action_schema_hash"], expected_action_hash)
    elif expected.production_compatible and actual["action_schema_hash"] != expected_action_hash:
        mismatch["action_schema_hash"] = (actual["action_schema_hash"], expected_action_hash)
    if mismatch:
        raise RuntimeError(f"P0_DQN_ACTION_SPACE_OR_OBSERVATION_CONTRACT_MISMATCH: {mismatch}")


def load_legacy_dqn_checkpoint(path: str) -> tuple[FactorizedQNetwork, ContractActionSpace, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    validate_checkpoint_contract(payload, LEGACY_DQN_CONTRACT)
    model = FactorizedQNetwork(LEGACY_DQN_CONTRACT.observation_dim, LEGACY_DQN_CONTRACT.type_head_dim, LEGACY_DQN_CONTRACT.candidate_slots)
    model.load_state_dict(payload.get("model", payload))
    model.eval()
    return model, ContractActionSpace(LEGACY_DQN_CONTRACT), LEGACY_DQN_CONTRACT.as_dict()
