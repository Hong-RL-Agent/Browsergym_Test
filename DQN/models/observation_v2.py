"""Deterministic Observation-v2 and explicit coverage memory.

This module is intentionally model-agnostic.  It does not change legacy 635/976
checkpoint inputs and does not add a recurrent network.  It provides a stable,
auditable state/action/outcome representation for a future policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from models.action_types import CANONICAL_ACTION_TYPES


OBSERVATION_V2_VERSION = "rawd_observation_v2_contract_v1"
_ACTION_TYPES = tuple(CANONICAL_ACTION_TYPES)


def _stable_unit(value: Any) -> float:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _number(value: Any, denominator: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(np.clip(number / denominator if denominator else 0.0, 0.0, 1.0))


def _candidate_identity(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    for key in ("bid", "entity_id", "id", "product_id", "item_id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return ""


def _state_components(observation: Mapping[str, Any]) -> dict[str, Any]:
    page = observation.get("page_state") if isinstance(observation.get("page_state"), Mapping) else {}
    candidates = observation.get("candidate_elements") if isinstance(observation.get("candidate_elements"), list) else []
    candidates = [item for item in candidates if isinstance(item, Mapping)]
    interactive = [
        {
            "tag": str(item.get("tag") or "").lower(),
            "role": str(item.get("role") or "").lower(),
            "bid": _candidate_identity(item),
            "visible": bool(item.get("visible")),
            "enabled": bool(item.get("enabled")),
        }
        for item in candidates
    ]
    return {
        "url_path": str(page.get("url") or "").split("?", 1)[0],
        "page_identity": str(page.get("page_identity") or page.get("title") or ""),
        "visible_interactive_count": len([x for x in interactive if x["visible"]]),
        "interactive_structure": interactive,
        "form_state": observation.get("form_state", {}),
        "selected_option_state": observation.get("selected_option_state", {}),
        "modal_navigation_state": observation.get("modal_state", observation.get("navigation_state", {})),
        "major_semantic_content": observation.get("semantic_content", {}),
    }


def functional_state_fingerprint(observation: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    components = _state_components(observation)
    payload = json.dumps(components, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), components


@dataclass
class CoverageMemory:
    """Episode/session deterministic coverage tracker."""

    visited_states: set[str] = field(default_factory=set)
    used_action_types: set[str] = field(default_factory=set)
    visited_targets: set[str] = field(default_factory=set)
    visited_flows: set[str] = field(default_factory=set)
    state_action_pairs: set[tuple[str, str]] = field(default_factory=set)
    state_action_outcomes: set[tuple[str, str, str]] = field(default_factory=set)

    def reset_episode(self) -> None:
        self.visited_states.clear()
        self.used_action_types.clear()
        self.visited_targets.clear()
        self.visited_flows.clear()
        self.state_action_pairs.clear()
        self.state_action_outcomes.clear()

    def observe(self, state_fingerprint: str, action: Mapping[str, Any], outcome: Mapping[str, Any], flow: str = "") -> dict[str, Any]:
        action_type = str(action.get("action_type") or "")
        target = str(action.get("target_identity") or "")
        outcome_key = json.dumps(dict(outcome), sort_keys=True, default=str, separators=(",", ":"))
        state_action = (state_fingerprint, action_type + ":" + target)
        state_action_outcome = (state_fingerprint, state_action[1], outcome_key)
        result = {
            "new_state": state_fingerprint not in self.visited_states,
            "new_action_type": action_type not in self.used_action_types if action_type else False,
            "new_target": target not in self.visited_targets if target else False,
            "new_flow": flow not in self.visited_flows if flow else False,
            "new_state_action": state_action not in self.state_action_pairs,
            "new_state_action_outcome": state_action_outcome not in self.state_action_outcomes,
        }
        self.visited_states.add(state_fingerprint)
        if action_type:
            self.used_action_types.add(action_type)
        if target:
            self.visited_targets.add(target)
        if flow:
            self.visited_flows.add(flow)
        self.state_action_pairs.add(state_action)
        self.state_action_outcomes.add(state_action_outcome)
        return result

    def summary(self) -> dict[str, int]:
        return {
            "visited_states": len(self.visited_states),
            "used_action_types": len(self.used_action_types),
            "visited_targets": len(self.visited_targets),
            "visited_flows": len(self.visited_flows),
            "state_action_pairs": len(self.state_action_pairs),
            "state_action_outcomes": len(self.state_action_outcomes),
        }


def _feature_schema() -> list[dict[str, Any]]:
    groups: list[tuple[str, int, list[str]]] = [
        ("current_state", 8, ["url_hash", "page_hash", "visible_interactive_count", "interactive_structure_hash", "form_state_hash", "selected_option_hash", "modal_navigation_hash", "semantic_content_hash"]),
        ("previous_action", 20 + 32 + 4, ["action_type_one_hot[20]", "target_slot_one_hot[32]", "option_hash", "execution_success", "action_age"]),
        ("previous_outcome", 10, ["url_changed", "dom_changed", "ui_changed", "functional_state_changed", "network_request", "network_failure", "console_error", "target_disappeared", "execution_success", "response_parsed"]),
        ("transition", 8, ["interactive_added", "interactive_removed", "visible_count_delta", "target_state_changed", "semantic_change", "layout_changed", "modal_changed", "path_changed"]),
        ("runtime_network", 7, ["request_started", "request_completed", "request_failed", "http_error", "parsed_response", "pending_request_count", "strict_action_requests"]),
        ("temporal", 6, ["since_action", "until_request", "until_completion", "until_ui_change", "pending_flag", "ui_settled"]),
        ("coverage", 28, ["used_action_type_one_hot[20]", "visited_state_count", "visited_target_count", "visited_flow_count", "state_action_count", "state_action_outcome_count", "new_state", "new_state_action"]),
        ("candidate_semantics", 12, ["tag_hash", "role_hash", "name_hash", "fillable", "selectable", "clickable", "option_count", "selected_option_hash", "disabled", "visible", "enabled", "bbox_valid"]),
    ]
    index = 0
    schema = []
    for group, dimension, names in groups:
        schema.append({"group": group, "start": index, "end": index + dimension - 1, "dimension": dimension, "features": names, "dtype": "float32", "normalization": "[0,1]", "default": 0.0})
        index += dimension
    return schema


FEATURE_SCHEMA = _feature_schema()
OBSERVATION_V2_DIM = sum(int(item["dimension"]) for item in FEATURE_SCHEMA)


def observation_v2_schema() -> dict[str, Any]:
    payload = {"version": OBSERVATION_V2_VERSION, "dimension": OBSERVATION_V2_DIM, "groups": FEATURE_SCHEMA, "action_types": list(_ACTION_TYPES)}
    schema_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()
    return {"schema_version": OBSERVATION_V2_VERSION, "schema_dim": OBSERVATION_V2_DIM, "schema_hash": schema_hash, "groups": FEATURE_SCHEMA, "action_types": list(_ACTION_TYPES)}


class ObservationV2Encoder:
    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = int(max_candidates)
        self.schema = observation_v2_schema()

    def encode(self, observation: Mapping[str, Any], *, previous_action: Mapping[str, Any] | None = None, previous_outcome: Mapping[str, Any] | None = None, transition: Mapping[str, Any] | None = None, temporal: Mapping[str, Any] | None = None, coverage: CoverageMemory | None = None, coverage_delta: Mapping[str, Any] | None = None, candidate: Mapping[str, Any] | None = None) -> np.ndarray:
        previous_action = previous_action or {}
        previous_outcome = previous_outcome or {}
        transition = transition or {}
        temporal = temporal or {}
        coverage_delta = coverage_delta or {}
        coverage = coverage or CoverageMemory()
        state_fp, components = functional_state_fingerprint(observation)
        vector = np.zeros(OBSERVATION_V2_DIM, dtype=np.float32)
        group = {item["group"]: item for item in FEATURE_SCHEMA}
        offset = group["current_state"]["start"]
        state_values = [
            _stable_unit(components["url_path"]), _stable_unit(components["page_identity"]), _number(components["visible_interactive_count"], 128),
            _stable_unit(components["interactive_structure"]), _stable_unit(components["form_state"]), _stable_unit(components["selected_option_state"]),
            _stable_unit(components["modal_navigation_state"]), _stable_unit(components["major_semantic_content"]),
        ]
        vector[offset:offset + 8] = state_values
        offset = group["previous_action"]["start"]
        action_type = str(previous_action.get("action_type") or "")
        if action_type in _ACTION_TYPES:
            vector[offset + _ACTION_TYPES.index(action_type)] = 1.0
        slot = previous_action.get("candidate_slot")
        if isinstance(slot, int) and 0 <= slot < self.max_candidates:
            vector[offset + 20 + slot] = 1.0
        vector[offset + 52] = _stable_unit(previous_action.get("option_identity", ""))
        vector[offset + 53] = _bool(previous_action.get("execution_success"))
        vector[offset + 54] = _number(previous_action.get("action_age"), 300)
        offset = group["previous_outcome"]["start"]
        outcome_names = ["url_changed", "dom_changed", "ui_changed", "functional_state_changed", "network_request", "network_failure", "console_error", "target_disappeared", "execution_success", "response_parsed"]
        vector[offset:offset + 10] = [_bool(previous_outcome.get(name)) for name in outcome_names]
        offset = group["transition"]["start"]
        transition_names = ["interactive_added", "interactive_removed", "visible_count_delta", "target_state_changed", "semantic_change", "layout_changed", "modal_changed", "path_changed"]
        vector[offset:offset + 8] = [_number(transition.get(name), 128) if name == "visible_count_delta" else _bool(transition.get(name)) for name in transition_names]
        offset = group["runtime_network"]["start"]
        network_names = ["request_started", "request_completed", "request_failed", "http_error", "parsed_response", "pending_request_count", "strict_action_requests"]
        vector[offset:offset + 7] = [_number(previous_outcome.get(name), 32) if name in {"pending_request_count", "strict_action_requests"} else _bool(previous_outcome.get(name)) for name in network_names]
        offset = group["temporal"]["start"]
        temporal_names = ["since_action", "until_request", "until_completion", "until_ui_change", "pending_flag", "ui_settled"]
        vector[offset:offset + 6] = [_number(temporal.get(name), 300) if name not in {"pending_flag", "ui_settled"} else _bool(temporal.get(name)) for name in temporal_names]
        offset = group["coverage"]["start"]
        for index, name in enumerate(_ACTION_TYPES):
            vector[offset + index] = _bool(name in coverage.used_action_types)
        summary = coverage.summary()
        vector[offset + 20:offset + 28] = [
            _number(summary["visited_states"], 128), _number(summary["visited_targets"], 128), _number(summary["visited_flows"], 64),
            _number(summary["state_action_pairs"], 512), _number(summary["state_action_outcomes"], 512), _bool(coverage_delta.get("new_state")), _bool(coverage_delta.get("new_state_action")), 0.0,
        ]
        offset = group["candidate_semantics"]["start"]
        candidate = candidate or {}
        vector[offset:offset + 12] = [
            _stable_unit(str(candidate.get("tag") or "").lower()), _stable_unit(str(candidate.get("role") or "").lower()), _stable_unit(str(candidate.get("aria_label") or candidate.get("name") or "").lower()),
            _bool(candidate.get("fillable")), _bool(candidate.get("selectable")), _bool(candidate.get("clickable")), _number(candidate.get("option_count"), 32), _stable_unit(candidate.get("selected_option", "")),
            _bool(candidate.get("disabled")), _bool(candidate.get("visible")), _bool(candidate.get("enabled")), _bool(candidate.get("bbox")),
        ]
        return vector
