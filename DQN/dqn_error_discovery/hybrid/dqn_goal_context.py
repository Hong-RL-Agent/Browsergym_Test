"""Canonical, GT-free goal context and local-mask contract for future DQN training."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from models.action_types import CANONICAL_ACTION_TYPES
from models.observation_v2 import observation_v2_schema

STRATEGIES = ("ACTION_REPLAY", "SEQUENCE_REPLAY", "STATE_REVISIT", "OBSERVATION_RECHECK", "TEMPORAL_RECHECK")
FAMILIES = ("interaction", "navigation", "console", "layout", "accessibility", "semantic", "network")
ORACLES = ("ACTION_EFFECT_ORACLE", "NETWORK_ORACLE", "CONSOLE_RUNTIME_ORACLE", "LAYOUT_ORACLE", "SEMANTIC_CONSISTENCY_ORACLE", "STATE_TRANSITION_ORACLE", "TEMPORAL_ORACLE")
OUTCOMES = ("FAILED_REPRODUCTION", "PARTIAL_REPRODUCTION", "SUCCESSFUL_REPRODUCTION")


def goal_schema() -> dict[str, Any]:
    groups = [
        ("active_candidate", 1, ["active_candidate"]),
        ("strategy", 5, [f"strategy[{x}]" for x in STRATEGIES]),
        ("candidate_family", 7, [f"family[{x}]" for x in FAMILIES]),
        ("desired_action", 20, [f"action[{x}]" for x in CANONICAL_ACTION_TYPES]),
        ("target_state", 6, ["target_mapped", "target_actionable", "target_slot_norm", "target_visible", "target_enabled", "target_bbox_valid"]),
        ("option", 3, ["option_present", "option_index_norm", "option_selected"]),
        ("evidence", 7, [f"oracle[{x}]" for x in ORACLES]),
        ("progress", 4, ["confirmation_progress", "same_chain_progress", "remaining_local_budget", "prior_execution_success"]),
        ("prior_outcome", 3, [f"outcome[{x}]" for x in OUTCOMES]),
        ("target_identity_compact", 3, ["target_kind_hash", "region_hash", "logical_entity_hash"]),
    ]
    start = 0
    result = []
    for name, dim, features in groups:
        result.append({"group": name, "start": start, "end": start + dim - 1, "dimension": dim, "features": features, "dtype": "float32", "normalization": "[0,1]", "default": 0.0})
        start += dim
    return {"version": "rawd_dqn_goal_context_v1", "dimension": start, "groups": result, "gt_fields": [], "runtime_source": "detector_candidate_context"}


def local_history_schema() -> dict[str, Any]:
    groups = [("last_action_type", 20), ("last_target_slot", 32), ("last_outcome", 10)]
    start = 0
    rows = []
    for name, dim in groups:
        rows.append({"group": name, "start": start, "end": start + dim - 1, "dimension": dim, "dtype": "float32", "normalization": "[0,1]"})
        start += dim
    return {"version": "rawd_dqn_local_history_v1", "dimension": start, "groups": rows}


def canonical_dqn_observation_schema() -> dict[str, Any]:
    base = observation_v2_schema()
    goal = goal_schema()
    history = local_history_schema()
    payload = {"version": "rawd_dqn_observation_v2_goal_context_v1", "base": base, "goal": goal, "history": history, "dimension": base["schema_dim"] + goal["dimension"] + history["dimension"]}
    payload["schema_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return payload


def encode_goal_context(context: Mapping[str, Any]) -> np.ndarray:
    schema = goal_schema()
    vector = np.zeros(schema["dimension"], dtype=np.float32)
    offset = 0
    vector[offset] = float(bool(context.get("active_candidate"))); offset += 1
    strategy = str(context.get("strategy") or "")
    if strategy in STRATEGIES: vector[offset + STRATEGIES.index(strategy)] = 1.0
    offset += len(STRATEGIES)
    family = str(context.get("candidate_family") or "")
    if family in FAMILIES: vector[offset + FAMILIES.index(family)] = 1.0
    offset += len(FAMILIES)
    action = str(context.get("desired_action_type") or "")
    if action in CANONICAL_ACTION_TYPES: vector[offset + CANONICAL_ACTION_TYPES.index(action)] = 1.0
    offset += len(CANONICAL_ACTION_TYPES)
    for key in ("target_mapped", "target_actionable", "target_slot_norm", "target_visible", "target_enabled", "target_bbox_valid"):
        vector[offset] = float(np.clip(float(context.get(key, 0.0) or 0.0), 0.0, 1.0)); offset += 1
    option = context.get("option_identity") if isinstance(context.get("option_identity"), Mapping) else {}
    vector[offset:offset + 3] = [float(bool(option)), float(np.clip(float(option.get("index", 0) or 0) / 31.0, 0.0, 1.0)), float(bool(option.get("selected")))]; offset += 3
    oracle = str(context.get("oracle_family") or "")
    if oracle in ORACLES: vector[offset + ORACLES.index(oracle)] = 1.0
    offset += len(ORACLES)
    for key in ("confirmation_progress", "same_chain_progress", "remaining_local_budget", "prior_execution_success"):
        vector[offset] = float(np.clip(float(context.get(key, 0.0) or 0.0), 0.0, 1.0)); offset += 1
    outcome = str(context.get("prior_outcome") or "")
    if outcome in OUTCOMES: vector[offset + OUTCOMES.index(outcome)] = 1.0
    return vector


def local_action_mask(action_space: Any, candidates: list[Mapping[str, Any]], context: Mapping[str, Any]) -> np.ndarray:
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    strategy = str(context.get("strategy") or "")
    target = str(context.get("target_identity") or "")
    target_slot = next((i for i, item in enumerate(candidates[:action_space.max_candidates]) if str(item.get("bid") or "") in {target, target.removeprefix("bid:")}), None)
    if strategy == "ACTION_REPLAY" and target_slot is not None and context.get("target_actionable"):
        action_type = str(context.get("desired_action_type") or "")
        if action_type in action_space.action_types:
            mask[action_space.encode(action_type, target_slot)] = 1.0
    else:
        for action_type in action_space.action_types:
            slot_count = len(candidates) if action_type in {"click_element", "fill_input", "select_option", "press_enter"} else 1
            for slot in range(min(slot_count, action_space.max_candidates)):
                mask[action_space.encode(action_type, slot)] = 1.0
    return mask
