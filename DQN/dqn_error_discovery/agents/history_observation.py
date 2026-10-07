"""DQN-only recent action history observation augmentation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from dqn_error_discovery.agents.confirmation_observation import _target_identity
from dqn_error_discovery.agents.repeat_semantics import cycle_repeat_length


HISTORY_OBSERVATION_VERSION = "v1"
HISTORY_OBSERVATION_V2_VERSION = "v2"
DQN_HISTORY_WINDOW = 4
HISTORY_CONTEXT_DIM = 238
HISTORY_CONTEXT_V2_DIM = 269
HISTORY_STREAK_CAP = 4
TARGET_REQUIRED_ACTION_TYPES = {"click_element", "fill_input", "press_enter"}
PROSPECTIVE_REPEAT_RISK_FLAGS = ("immediate", "recent", "cycle")


def candidate_target_identity(candidate: Mapping[str, Any] | None) -> str:
    return _target_identity(candidate)


def action_identity(
    action_type: str,
    candidate: Mapping[str, Any] | None = None,
    *,
    requires_target: bool | None = None,
) -> str:
    requires_target = bool(requires_target) if requires_target is not None else action_type in TARGET_REQUIRED_ACTION_TYPES
    target = candidate_target_identity(candidate) if requires_target else ""
    return f"{action_type}:{target}" if target else str(action_type)


def append_history_transition(
    history: dict[str, Any],
    *,
    action_type: str,
    candidate: Mapping[str, Any] | None,
    meaningful_action_effect: bool,
    productive_transition: bool,
    confirmation_progress: bool,
    unproductive_repeat: bool,
) -> None:
    records = history.setdefault("dqn_history_records", [])
    requires_target = action_type in TARGET_REQUIRED_ACTION_TYPES
    records.append(
        {
            "action_type": str(action_type),
            "requires_target": int(requires_target),
            "target_identity": candidate_target_identity(candidate) if requires_target else None,
            "meaningful_action_effect": int(bool(meaningful_action_effect)),
            "productive_transition": int(bool(productive_transition)),
            "confirmation_progress": int(bool(confirmation_progress)),
            "unproductive_repeat": int(bool(unproductive_repeat)),
            "action_identity": action_identity(action_type, candidate, requires_target=requires_target),
            "valid": 1,
        }
    )
    del records[:-DQN_HISTORY_WINDOW]


def _current_slots(target_identity: str | None, candidates: Sequence[Mapping[str, Any]], max_candidates: int) -> list[int]:
    if not target_identity:
        return []
    return [
        index for index, candidate in enumerate(candidates[:max_candidates])
        if candidate_target_identity(candidate) == target_identity
    ]


def build_dqn_history_context(
    history: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
    action_types: Sequence[str],
    max_candidates: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    history = history or {}
    observation = observation or {}
    candidates = observation.get("candidate_elements", [])
    candidates = candidates if isinstance(candidates, list) else []
    records = [row for row in list(history.get("dqn_history_records", []) or []) if isinstance(row, Mapping)]
    recent = list(reversed(records[-DQN_HISTORY_WINDOW:]))
    vector = np.zeros(HISTORY_CONTEXT_DIM, dtype=np.float32)
    metadata: dict[str, Any] = {
        "recent_action_identities": [],
        "historical_target_identities": [],
        "target_current_slots": [],
        "history_valid_mask": [],
        "outcome_flags": [],
        "same_action_identity_streak": 0,
        "distinct_action_identity_count": 0,
        "active_cycle_length": 0,
    }
    action_type_count = len(action_types)
    for slot in range(DQN_HISTORY_WINDOW):
        row = recent[slot] if slot < len(recent) else None
        valid = bool(row and row.get("valid", 1))
        metadata["history_valid_mask"].append(int(valid))
        if not valid:
            metadata["recent_action_identities"].append("")
            metadata["historical_target_identities"].append(None)
            metadata["target_current_slots"].append([])
            metadata["outcome_flags"].append([0, 0, 0, 0])
            continue
        vector[212 + slot] = 1.0
        action_type = str(row.get("action_type") or "")
        if action_type in action_types:
            vector[slot * action_type_count + action_types.index(action_type)] = 1.0
        offset = 76 + slot
        requires_target = bool(row.get("requires_target", action_type in {"click_element", "fill_input", "press_enter"}))
        vector[offset] = float(requires_target)
        target_identity = row.get("target_identity")
        slots = _current_slots(str(target_identity) if target_identity else None, candidates, max_candidates)
        for candidate_slot in slots:
            vector[80 + slot * max_candidates + candidate_slot] = 1.0
        vector[208 + slot] = float(bool(slots)) if target_identity else 0.0
        flags = [
            int(bool(row.get("meaningful_action_effect"))),
            int(bool(row.get("productive_transition"))),
            int(bool(row.get("confirmation_progress"))),
            int(bool(row.get("unproductive_repeat"))),
        ]
        for flag_index, flag in enumerate(flags):
            vector[216 + slot * 4 + flag_index] = float(flag)
        metadata["recent_action_identities"].append(str(row.get("action_identity") or action_type))
        metadata["historical_target_identities"].append(target_identity)
        metadata["target_current_slots"].append(slots)
        metadata["outcome_flags"].append(flags)

    identities = [str(row.get("action_identity") or row.get("action_type") or "") for row in recent if row.get("valid", 1)]
    streak = 0
    if identities:
        for identity in identities:
            if identity == identities[0]:
                streak += 1
            else:
                break
    distinct_count = len(set(identities))
    chronological = list(reversed(identities))
    cycle_length = cycle_repeat_length(chronological, max_cycle=3)
    vector[232] = min(streak, HISTORY_STREAK_CAP) / HISTORY_STREAK_CAP
    vector[233] = min(distinct_count, DQN_HISTORY_WINDOW) / DQN_HISTORY_WINDOW
    vector[234 + min(cycle_length, 3)] = 1.0
    metadata.update({
        "same_action_identity_streak": streak,
        "distinct_action_identity_count": distinct_count,
        "active_cycle_length": cycle_length,
    })
    return vector, metadata


def _reward_action_target_key(action_type: str) -> str:
    return f"{action_type}:0"


def _record_repeat_key(record: Mapping[str, Any]) -> str:
    action_type = str(record.get("action_type") or "")
    if not action_type:
        return ""
    if bool(record.get("requires_target", action_type in TARGET_REQUIRED_ACTION_TYPES)):
        return action_identity(action_type, None, requires_target=True)
    return _reward_action_target_key(action_type)


def prospective_repeat_flags(
    history: Mapping[str, Any] | None,
    action_type: str,
    action_types: Sequence[str],
    *,
    window: int = DQN_HISTORY_WINDOW,
    max_cycle: int = 3,
) -> dict[str, bool]:
    """Return repeat risk for a hypothetical current action without mutating history."""
    if action_type in TARGET_REQUIRED_ACTION_TYPES or action_type not in action_types:
        return {"immediate": False, "recent": False, "cycle": False}
    full_history = list((history or {}).get("dqn_action_target_history", []) or [])
    if full_history:
        action_history = [str(item) for item in full_history if item]
    else:
        records = [row for row in list((history or {}).get("dqn_history_records", []) or []) if isinstance(row, Mapping)]
        action_history = [_record_repeat_key(row) for row in records if row.get("valid", 1)]
        action_history = [item for item in action_history if item]
    candidate_key = _reward_action_target_key(action_type)
    current_sequence = action_history + [candidate_key]
    recent_window = action_history[max(0, len(action_history) - window):]
    return {
        "immediate": bool(action_history and action_history[-1] == candidate_key),
        "recent": bool(candidate_key in recent_window),
        "cycle": bool(cycle_repeat_length(current_sequence, max_cycle=max_cycle) > 0),
    }


def build_dqn_history_context_v2(
    history: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
    action_types: Sequence[str],
    max_candidates: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build predictive history context v2.

    The policy vector contains raw recent actions/target remaps plus prospective
    repeat risk by action type. Post-hoc outcome flags remain in metadata only.
    """
    history = history or {}
    observation = observation or {}
    candidates = observation.get("candidate_elements", [])
    candidates = candidates if isinstance(candidates, list) else []
    records = [row for row in list(history.get("dqn_history_records", []) or []) if isinstance(row, Mapping)]
    recent = list(reversed(records[-DQN_HISTORY_WINDOW:]))
    vector = np.zeros(HISTORY_CONTEXT_V2_DIM, dtype=np.float32)
    metadata: dict[str, Any] = {
        "version": HISTORY_OBSERVATION_V2_VERSION,
        "recent_action_identities": [],
        "historical_target_identities": [],
        "target_current_slots": [],
        "history_valid_mask": [],
        "raw_transition_outcomes": [],
        "outcome_flags": [],
        "prospective_repeat_risk_ordering": "action_type-major: immediate,recent,cycle",
        "prospective_repeat_risk_by_action_type": {},
        "hypothetical_repeat_flags": {},
        "same_action_identity_streak": 0,
        "distinct_action_identity_count": 0,
        "active_cycle_length": 0,
    }
    action_type_count = len(action_types)
    for slot in range(DQN_HISTORY_WINDOW):
        row = recent[slot] if slot < len(recent) else None
        valid = bool(row and row.get("valid", 1))
        metadata["history_valid_mask"].append(int(valid))
        if not valid:
            metadata["recent_action_identities"].append("")
            metadata["historical_target_identities"].append(None)
            metadata["target_current_slots"].append([])
            metadata["outcome_flags"].append([0, 0, 0, 0])
            metadata["raw_transition_outcomes"].append({})
            continue
        action_type = str(row.get("action_type") or "")
        if action_type in action_types:
            vector[slot * action_type_count + action_types.index(action_type)] = 1.0
        offset = 76 + slot
        requires_target = bool(row.get("requires_target", action_type in TARGET_REQUIRED_ACTION_TYPES))
        vector[offset] = float(requires_target)
        target_identity = row.get("target_identity")
        slots = _current_slots(str(target_identity) if target_identity else None, candidates, max_candidates)
        for candidate_slot in slots:
            vector[80 + slot * max_candidates + candidate_slot] = 1.0
        vector[208 + slot] = float(bool(slots)) if target_identity else 0.0
        flags = [
            int(bool(row.get("meaningful_action_effect"))),
            int(bool(row.get("productive_transition"))),
            int(bool(row.get("confirmation_progress"))),
            int(bool(row.get("unproductive_repeat"))),
        ]
        metadata["recent_action_identities"].append(str(row.get("action_identity") or action_type))
        metadata["historical_target_identities"].append(target_identity)
        metadata["target_current_slots"].append(slots)
        metadata["outcome_flags"].append(flags)
        metadata["raw_transition_outcomes"].append({
            "meaningful_action_effect": flags[0],
            "productive_transition": flags[1],
            "confirmation_progress": flags[2],
            "unproductive_repeat": flags[3],
        })

    for action_type_index, action_type in enumerate(action_types):
        flags = prospective_repeat_flags(history, action_type, action_types)
        base = 212 + action_type_index * len(PROSPECTIVE_REPEAT_RISK_FLAGS)
        for flag_index, name in enumerate(PROSPECTIVE_REPEAT_RISK_FLAGS):
            vector[base + flag_index] = float(flags[name])
        metadata["prospective_repeat_risk_by_action_type"][str(action_type)] = dict(flags)
        metadata["hypothetical_repeat_flags"][str(action_type)] = dict(flags)

    identities = [str(row.get("action_identity") or row.get("action_type") or "") for row in recent if row.get("valid", 1)]
    streak = 0
    if identities:
        for identity in identities:
            if identity == identities[0]:
                streak += 1
            else:
                break
    distinct_count = len(set(identities))
    chronological = [_record_repeat_key(row) for row in records[-DQN_HISTORY_WINDOW:] if isinstance(row, Mapping) and row.get("valid", 1)]
    cycle_length = cycle_repeat_length([item for item in chronological if item], max_cycle=3)
    metadata.update({
        "same_action_identity_streak": streak,
        "distinct_action_identity_count": distinct_count,
        "active_cycle_length": cycle_length,
    })
    return vector, metadata


def augment_dqn_history_observation(base: np.ndarray, confirmation: np.ndarray, history_context: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    confirmation = np.asarray(confirmation, dtype=np.float32).reshape(-1)
    history_context = np.asarray(history_context, dtype=np.float32).reshape(-1)
    if history_context.size != HISTORY_CONTEXT_DIM:
        raise ValueError(f"history context must have {HISTORY_CONTEXT_DIM} values, got {history_context.size}")
    return np.concatenate([base, confirmation, history_context]).astype(np.float32)


def augment_dqn_history_observation_v2(base: np.ndarray, confirmation: np.ndarray, history_context: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    confirmation = np.asarray(confirmation, dtype=np.float32).reshape(-1)
    history_context = np.asarray(history_context, dtype=np.float32).reshape(-1)
    if history_context.size != HISTORY_CONTEXT_V2_DIM:
        raise ValueError(f"history context v2 must have {HISTORY_CONTEXT_V2_DIM} values, got {history_context.size}")
    return np.concatenate([base, confirmation, history_context]).astype(np.float32)
