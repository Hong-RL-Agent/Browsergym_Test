"""Canonical flattened action ordering shared by PPO, DQN and runners."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

CANONICAL_ACTION_SPACE_VERSION = "rawd_v3_canonical_v1"
CANONICAL_ACTION_TYPES = (
    "noop", "click_element", "scroll_down", "scroll_up", "inspect_dom",
    "inspect_layout", "inspect_network", "inspect_console", "inspect_cart",
    "inspect_server_health", "inspect_port_status", "inspect_latency",
    "inspect_server_logs", "inspect_runtime_metrics", "fill_input",
    "select_option", "press_enter", "change_viewport_mobile",
    "change_viewport_desktop", "finish_episode",
)


def action_space_metadata(max_candidates: int = 32, action_types: Sequence[str] = CANONICAL_ACTION_TYPES) -> dict[str, Any]:
    ordered = list(action_types)
    ranges = {
        name: {
            "type_index": index,
            "first_flattened_index": index * int(max_candidates),
            "last_flattened_index": (index + 1) * int(max_candidates) - 1,
            "candidate_slots": [0, int(max_candidates) - 1],
        }
        for index, name in enumerate(ordered)
    }
    return {
        "action_space_version": CANONICAL_ACTION_SPACE_VERSION,
        "action_types": ordered,
        "action_dim": len(ordered) * int(max_candidates),
        "max_candidates": int(max_candidates),
        "ranges": ranges,
    }


def action_space_mapping_hash(max_candidates: int = 32, action_types: Sequence[str] = CANONICAL_ACTION_TYPES) -> str:
    payload = json.dumps(action_space_metadata(max_candidates, action_types), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_action_space_metadata(
    metadata: Mapping[str, Any],
    *,
    max_candidates: int = 32,
    action_types: Sequence[str] = CANONICAL_ACTION_TYPES,
) -> None:
    expected = action_space_metadata(max_candidates, action_types)
    actual_types = list(metadata.get("action_types") or [])
    actual_dim = int(metadata.get("action_dim", -1) or -1)
    if actual_types != expected["action_types"] or actual_dim != expected["action_dim"]:
        raise RuntimeError("PPO_ACTION_SPACE_MAPPING_MISMATCH")
    actual_hash = str(metadata.get("mapping_hash") or "")
    if actual_hash and actual_hash != action_space_mapping_hash(max_candidates, action_types):
        raise RuntimeError("PPO_ACTION_SPACE_MAPPING_MISMATCH")
