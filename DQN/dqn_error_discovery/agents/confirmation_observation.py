"""DQN-only confirmation context augmentation.

The base BrowserGym observation remains unchanged.  The legacy helper converts
the reward-side pending reproduction tracker into a fixed 72-value policy
input.  RAWD v3 uses the same overall DQN observation size but replaces the
padded 72-value slice with a native 73-value active-candidate context.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dqn_error_discovery.agents.confirmation_reproduction import ACTION_REPLAY, ACTION_REPRODUCTION, OBSERVATION_RECHECK, OBSERVATION_REPRODUCTION, derive_reproduction_context


CONFIRMATION_OBSERVATION_VERSION = "v1"
CONFIRMATION_CONTEXT_DIM = 72
ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION = "v3_active_candidate_confirmation_context"
ACTIVE_CANDIDATE_CONFIRMATION_CONTEXT_DIM = 73
PENDING_COUNT_CAP = 8
ATTEMPT_COUNT_CAP = 4
RULE_IDS = tuple(f"R{index:02d}" for index in range(1, 16))
ACTIVE_CATEGORY_SLICE = slice(0, 15)
ACTIVE_ACTION_SLICE = slice(15, 35)
ACTIVE_TARGET_SLICE = slice(35, 67)
ACTIVE_FLAG_SLICE = slice(67, 73)
ACTIVE_FLAG_NAMES = (
    "active_candidate_exists",
    "pending_confirmation",
    "independent_reproduction_required",
    "origin_action_available",
    "origin_target_mapped",
    "active_target_actionable",
)
ACTIVE_CONFIRMATION_SCHEMA = {
    "version": ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION,
    "dimension": ACTIVE_CANDIDATE_CONFIRMATION_CONTEXT_DIM,
    "taxonomy": {"start": ACTIVE_CATEGORY_SLICE.start, "end": ACTIVE_CATEGORY_SLICE.stop - 1, "labels": list(RULE_IDS)},
    "action_type": {"start": ACTIVE_ACTION_SLICE.start, "end": ACTIVE_ACTION_SLICE.stop - 1},
    "target": {"start": ACTIVE_TARGET_SLICE.start, "end": ACTIVE_TARGET_SLICE.stop - 1, "max_candidates": 32},
    "flags": {
        "start": ACTIVE_FLAG_SLICE.start,
        "end": ACTIVE_FLAG_SLICE.stop - 1,
        "labels": list(ACTIVE_FLAG_NAMES),
    },
}


def _target_identity(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    for key in ("bid", "entity_id", "product_id", "cart_item_id", "item_id", "post_id", "comment_id", "id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    href = str(candidate.get("href") or "")
    role = str(candidate.get("role") or "")
    tag = str(candidate.get("tag") or "")
    name = str(candidate.get("name") or candidate.get("aria_label") or "").strip().lower()
    if href and (role or tag or name):
        return f"fingerprint:{role}:{tag}:{href}:{name}"
    return ""


def _split_action_target(value: str) -> tuple[str, str]:
    text = str(value or "")
    if ":" not in text:
        return text, ""
    action_type, target = text.split(":", 1)
    return action_type, target


def _candidate_matches(candidate: Mapping[str, Any], expected_target: str) -> bool:
    if not expected_target:
        return False
    identity = _target_identity(candidate)
    if identity == expected_target:
        return True
    # Tracker action targets use action_type:bid:<value>; compare the stable
    # target suffix so a candidate index or DOM ordering change is irrelevant.
    return identity and identity.split(":", 1)[-1] == expected_target.split(":", 1)[-1]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_target_identity_from_record(record: Mapping[str, Any]) -> str:
    derived = derive_reproduction_context(record)
    if derived.strategy == OBSERVATION_RECHECK:
        return ""
    if derived.reproduction_target:
        return derived.reproduction_target
    evidence = _as_mapping(record.get("evidence"))
    value = record.get("reproduction_target_identity") or evidence.get("reproduction_target_identity")
    if value is not None and str(value).strip():
        text = str(value).strip()
        return text if ":" in text else f"bid:{text}"
    for key in ("expected_target_identity", "target_identity", "target_id", "target_bid", "clicked_bid", "bid"):
        value = record.get(key)
        if value is None or not str(value).strip():
            value = evidence.get(key)
        if value is not None and str(value).strip():
            text = str(value).strip()
            return text if ":" in text else f"bid:{text}"
    action_target = str(record.get("action_target") or "")
    _, expected = _split_action_target(action_target)
    if expected:
        return expected if ":" in expected else f"bid:{expected}"
    return ""


def _candidate_rule_id(record: Mapping[str, Any]) -> str:
    evidence = _as_mapping(record.get("evidence"))
    return str(record.get("rule_id") or evidence.get("rule_id") or "").strip()


def _candidate_action_type(record: Mapping[str, Any]) -> str:
    derived = derive_reproduction_context(record)
    if derived.strategy == OBSERVATION_RECHECK:
        return ""
    if derived.reproduction_action_type:
        return derived.reproduction_action_type
    evidence = _as_mapping(record.get("evidence"))
    action_target = str(record.get("action_target") or "")
    action_type, _ = _split_action_target(action_target)
    return str(
        record.get("expected_action_type")
        or record.get("reproduction_action_type")
        or record.get("action_type")
        or evidence.get("action_type")
        or action_type
        or ""
    ).strip()


def _pending_candidate_from_history(history: Mapping[str, Any]) -> Mapping[str, Any] | None:
    records = _records(history)
    if not records:
        return None
    pending = [row for row in records if bool(row.get("pending_confirmation", True))]
    action_pending = [row for row in pending if row.get("confirmation_mode") == ACTION_REPRODUCTION or row.get("action_target")]
    return _active_pending(action_pending or pending or records, [])


def _matching_target_slots(target_identity: str, candidates: list[Mapping[str, Any]], max_candidates: int) -> list[int]:
    if not target_identity:
        return []
    return [
        index
        for index, candidate in enumerate(candidates[:max_candidates])
        if _candidate_matches(candidate, target_identity)
    ]


def _action_mask_valid(action_space: Any, mask: Any, action_type: str, slot: int) -> bool:
    if mask is None or not action_type:
        return False
    try:
        action_index = int(action_space.encode(action_type, slot))
        return bool(np.asarray(mask).reshape(-1)[action_index] > 0)
    except Exception:
        return False


def build_active_candidate_confirmation_context(
    history: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
    action_space: Any,
    *,
    active_candidate: Mapping[str, Any] | None = None,
    mask: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build RAWD v3's native 73-value active-candidate context.

    The encoder is taxonomy-generic and uses only runtime candidate metadata.
    Raw endpoint strings and semantic field names remain available to the
    detector/tracker but are intentionally not embedded into the DQN vector.
    """
    history = history or {}
    observation = observation or {}
    candidates = observation.get("candidate_elements", [])
    candidates = candidates if isinstance(candidates, list) else []
    candidates = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    max_candidates = int(getattr(action_space, "max_candidates", 32) or 32)
    vector = np.zeros(ACTIVE_CANDIDATE_CONFIRMATION_CONTEXT_DIM, dtype=np.float32)
    active = active_candidate if isinstance(active_candidate, Mapping) else _pending_candidate_from_history(history)
    metadata: dict[str, Any] = {
        "schema_version": ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION,
        "schema": ACTIVE_CONFIRMATION_SCHEMA,
        "active_candidate_exists": False,
        "active_pending_key": "",
        "active_rule_id": "",
        "expected_action_type": "",
        "expected_target_identity": None,
        "matching_candidate_slots": [],
        "origin_target_mapped": 0,
        "active_target_actionable": 0,
        "pending_confirmation": 0,
        "independent_reproduction_required": 0,
        "origin_action_available": 0,
        "reproduction_strategy": "",
        "requires_action_reproduction": 0,
        "nonzero_indices": [],
        "decoded_category": "",
        "decoded_origin_action": "",
        "decoded_target_index": None,
        "decoded_flags": {name: 0 for name in ACTIVE_FLAG_NAMES},
    }
    if not active:
        return vector, metadata

    rule_id = _candidate_rule_id(active)
    derived = derive_reproduction_context(active)
    active_key = str(active.get("logical_key") or active.get("signature") or active.get("id") or "")
    action_type = _candidate_action_type(active)
    target_identity = _candidate_target_identity_from_record(active)
    matching_slots = _matching_target_slots(target_identity, candidates, max_candidates)
    pending = bool(
        active.get("pending_confirmation")
        or str(active.get("lifecycle_status") or "").upper() in {"DETECTED", "PENDING_CONFIRMATION", "DETECTED_NOT_CONFIRMED"}
        or str(active.get("status") or "").lower() in {"candidate", "detected", "pending"}
    )
    reproduction_required = bool(
        pending
        and (
            str(active.get("confirmation_policy") or "").lower() == "reproduce"
            or str(active.get("confirmation_mode") or "").upper() == ACTION_REPRODUCTION
            or bool(target_identity)
            or bool(action_type)
        )
    )
    if derived.strategy == OBSERVATION_RECHECK:
        reproduction_required = bool(pending)

    vector[ACTIVE_FLAG_SLICE.start + 0] = 1.0
    vector[ACTIVE_FLAG_SLICE.start + 1] = float(pending)
    vector[ACTIVE_FLAG_SLICE.start + 2] = float(reproduction_required)
    vector[ACTIVE_FLAG_SLICE.start + 3] = float(bool(action_type))
    vector[ACTIVE_FLAG_SLICE.start + 4] = float(bool(matching_slots))
    if rule_id in RULE_IDS:
        vector[ACTIVE_CATEGORY_SLICE.start + RULE_IDS.index(rule_id)] = 1.0
    if action_type in getattr(action_space, "action_types", []):
        vector[ACTIVE_ACTION_SLICE.start + list(action_space.action_types).index(action_type)] = 1.0
    for slot in matching_slots:
        vector[ACTIVE_TARGET_SLICE.start + slot] = 1.0
    actionable = any(_action_mask_valid(action_space, mask, action_type, slot) for slot in matching_slots) if matching_slots else False
    vector[ACTIVE_FLAG_SLICE.start + 5] = float(actionable)

    nonzero_indices = [int(index) for index in np.flatnonzero(vector)]
    flag_values = vector[ACTIVE_FLAG_SLICE].astype(float).tolist()
    metadata.update(
        {
            "active_candidate_exists": True,
            "active_pending_key": active_key,
            "active_rule_id": rule_id,
            "expected_action_type": action_type,
            "expected_target_identity": target_identity or None,
            "matching_candidate_slots": matching_slots,
            "origin_target_mapped": int(bool(matching_slots)),
            "active_target_actionable": int(actionable),
            "pending_confirmation": int(pending),
            "independent_reproduction_required": int(reproduction_required),
            "origin_action_available": int(bool(action_type)),
            "reproduction_strategy": derived.strategy,
            "requires_action_reproduction": int(derived.requires_action_reproduction),
            "nonzero_indices": nonzero_indices,
            "nonzero_values": [float(vector[index]) for index in nonzero_indices],
            "decoded_category": rule_id if rule_id in RULE_IDS else "",
            "decoded_origin_action": action_type if action_type in getattr(action_space, "action_types", []) else "",
            "decoded_target_index": matching_slots[0] if matching_slots else None,
            "decoded_flags": {name: int(flag_values[index]) for index, name in enumerate(ACTIVE_FLAG_NAMES)},
        }
    )
    return vector, metadata


def _records(history: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = history.get("pending_reproduction_records", {})
    if isinstance(records, Mapping) and records:
        result = []
        for value in records.values():
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            row.setdefault("confirmation_mode", ACTION_REPRODUCTION if row.get("action_target") else OBSERVATION_REPRODUCTION)
            result.append(row)
        return result
    targets = history.get("pending_reproduction_action_targets", set()) or set()
    return [
        {
            "action_target": str(value),
            "confirmation_mode": ACTION_REPRODUCTION,
            "rule_id": "",
            "progress": 0.0,
            "attempt_count": 0,
        }
        for value in targets
    ]


def _active_pending(records: list[dict[str, Any]], candidates: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    def ranking(record: Mapping[str, Any]) -> tuple[int, float, int, str]:
        target = str(record.get("action_target") or "")
        resolvable = int(any(_candidate_matches(candidate, _split_action_target(target)[1]) for candidate in candidates))
        progress = float(record.get("progress", 0.0) or 0.0)
        attempt = int(record.get("created_step", 0) or 0)
        key = str(record.get("logical_key") or record.get("signature") or target)
        return (-resolvable, -progress, attempt, key)

    action_records = [record for record in records if record.get("confirmation_mode") == ACTION_REPRODUCTION]
    # Action instructions are policy-relevant; observation-only pending items
    # remain visible when no executable action instruction is pending.
    return min(action_records or records, key=ranking) if records else None


def build_dqn_confirmation_context(
    history: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
    action_space: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the fixed 72-value pre-action confirmation context."""
    history = history or {}
    observation = observation or {}
    candidates = observation.get("candidate_elements", [])
    candidates = candidates if isinstance(candidates, list) else []
    candidates = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    records = _records(history)
    active = _active_pending(records, candidates)

    context = np.zeros(CONFIRMATION_CONTEXT_DIM, dtype=np.float32)
    metadata: dict[str, Any] = {
        "pending_confirmation_exists": bool(records),
        "confirmation_mode": "",
        "active_pending_key": "",
        "active_rule_id": "",
        "expected_action_type": "",
        "expected_target_identity": None,
        "expected_action_requires_target": 0,
        "expected_target_visible": 0,
        "matching_candidate_slots": [],
        "confirmation_progress": 0.0,
        "reproduction_attempt_count": 0.0,
        "pending_confirmation_count": len(records),
    }
    if not active:
        context[71] = min(len(records), PENDING_COUNT_CAP) / PENDING_COUNT_CAP
        return context, metadata

    mode = str(active.get("confirmation_mode") or (ACTION_REPRODUCTION if active.get("action_target") else OBSERVATION_REPRODUCTION))
    action_target = str(active.get("action_target") or "")
    expected_action_type, expected_target = _split_action_target(action_target) if mode == ACTION_REPRODUCTION else ("", "")
    requires_target = int(mode == ACTION_REPRODUCTION and bool(expected_target))
    matching_slots = [index for index, candidate in enumerate(candidates[: action_space.max_candidates]) if _candidate_matches(candidate, expected_target)] if requires_target else []
    rule_id = str(active.get("rule_id") or "")
    progress = min(max(float(active.get("progress", 0.0) or 0.0), 0.0), 1.0)
    attempts = min(max(int(active.get("attempt_count", 0) or 0), 0), ATTEMPT_COUNT_CAP) / ATTEMPT_COUNT_CAP
    pending_count = min(len(records), PENDING_COUNT_CAP) / PENDING_COUNT_CAP

    context[0] = 1.0
    if rule_id in RULE_IDS:
        context[1 + RULE_IDS.index(rule_id)] = 1.0
    if expected_action_type in action_space.action_types:
        context[16 + action_space.action_types.index(expected_action_type)] = 1.0
    context[35] = float(requires_target)
    for slot in matching_slots:
        context[36 + slot] = 1.0
    context[68] = float(bool(matching_slots)) if requires_target else 0.0
    context[69] = progress
    context[70] = attempts
    context[71] = pending_count
    metadata.update(
        {
            "active_pending_key": str(active.get("logical_key") or active.get("signature") or action_target),
            "confirmation_mode": mode,
            "active_rule_id": rule_id,
            "expected_action_type": expected_action_type,
            "expected_target_identity": expected_target or None,
            "expected_action_requires_target": requires_target,
            "expected_target_visible": int(bool(matching_slots)) if requires_target else 0,
            "matching_candidate_slots": matching_slots,
            "confirmation_progress": progress,
            "reproduction_attempt_count": attempts,
            "pending_confirmation_count": len(records),
        }
    )
    return context, metadata


def augment_dqn_observation(base_vector: np.ndarray, context: np.ndarray) -> np.ndarray:
    base = np.asarray(base_vector, dtype=np.float32).reshape(-1)
    context = np.asarray(context, dtype=np.float32).reshape(-1)
    if context.size != CONFIRMATION_CONTEXT_DIM:
        raise ValueError(f"confirmation context must have {CONFIRMATION_CONTEXT_DIM} values, got {context.size}")
    return np.concatenate([base, context]).astype(np.float32)
