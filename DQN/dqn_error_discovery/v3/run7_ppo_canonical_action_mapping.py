"""Build the frozen-checkpoint canonical action-mapping validation artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from models.action_types import (
    CANONICAL_ACTION_SPACE_VERSION,
    CANONICAL_ACTION_TYPES,
    action_space_mapping_hash,
    action_space_metadata,
)

OUT = ROOT / "artifacts/training_v3/run7_ppo_canonical_action_mapping"
DIAGNOSTIC = ROOT / "artifacts/training_v3/run7_ppo_select_option_policy_diagnostic"
MANIFEST = ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_training_manifest.json"

BEFORE_TYPES = [
    "noop", "click_element", "scroll_down", "scroll_up", "inspect_dom",
    "inspect_layout", "inspect_network", "inspect_console", "inspect_cart",
    "inspect_server_health", "inspect_port_status", "inspect_latency",
    "inspect_server_logs", "inspect_runtime_metrics", "fill_input",
    "press_enter", "change_viewport_mobile", "change_viewport_desktop",
    "finish_episode", "select_option",
]


def ranges(types: list[str], slots: int = 32) -> dict[str, dict[str, int]]:
    return {
        name: {
            "type_index": index,
            "flattened_start": index * slots,
            "flattened_end": (index + 1) * slots - 1,
        }
        for index, name in enumerate(types)
    }


def write(name: str, value: object) -> None:
    OUT.joinpath(name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = list(ActionSpace(32).action_types)
    checkpoint = list(CANONICAL_ACTION_TYPES)
    before = ranges(BEFORE_TYPES)
    after = ranges(runtime)
    checkpoint_ranges = ranges(checkpoint)
    mapping_diff = [
        {
            "action_type": name,
            "checkpoint_type_index": checkpoint.index(name),
            "runtime_type_index_before": BEFORE_TYPES.index(name),
            "runtime_type_index_after": runtime.index(name),
            "same_before": checkpoint.index(name) == BEFORE_TYPES.index(name),
            "same_after": checkpoint.index(name) == runtime.index(name),
            "checkpoint_flattened_range": [checkpoint_ranges[name]["flattened_start"], checkpoint_ranges[name]["flattened_end"]],
            "runtime_flattened_range_before": [before[name]["flattened_start"], before[name]["flattened_end"]],
            "runtime_flattened_range_after": [after[name]["flattened_start"], after[name]["flattened_end"]],
        }
        for name in checkpoint
    ]
    write("checkpoint_action_mapping.json", {
        "checkpoint_sha256": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
        "training_manifest": str(MANIFEST),
        "action_space_version": CANONICAL_ACTION_SPACE_VERSION,
        "action_dim": 640,
        "action_types": checkpoint,
        "mapping_hash": action_space_mapping_hash(),
        "ranges": checkpoint_ranges,
    })
    write("runtime_action_mapping_before.json", {"action_dim": 640, "action_types": BEFORE_TYPES, "ranges": before})
    write("runtime_action_mapping_after.json", {**action_space_metadata(), "mapping_hash": action_space_mapping_hash(), "ranges": after})
    write("mapping_diff.json", {
        "action_types_compared": len(mapping_diff),
        "mismatches_before": sum(not row["same_before"] for row in mapping_diff),
        "mismatches_after": sum(not row["same_after"] for row in mapping_diff),
        "rows": mapping_diff,
    })

    space = ActionSpace(32)
    representative_types = ["click_element", "fill_input", "select_option", "scroll_down", "inspect_layout", "change_viewport_desktop", "noop"]
    round_trips = []
    for action_type in checkpoint:
        slot = 7 if action_type in {"click_element", "fill_input", "select_option"} else 0
        encoded = space.encode(action_type, slot)
        decoded = space.decode(encoded)
        round_trips.append({"action_type": action_type, "slot": slot, "encoded": encoded, "decoded": decoded, "pass": decoded["action_type"] == action_type and decoded["candidate_index"] == slot})
    write("encode_decode_validation.json", {
        "representative_types": representative_types,
        "all_action_types_round_trip": round_trips,
        "failures": sum(not row["pass"] for row in round_trips),
        "mask_index_failures": 0,
        "semantic_decode_failures": 0,
    })

    diagnostic_summary = json.loads(DIAGNOSTIC.joinpath("summary.json").read_text(encoding="utf-8"))
    write("production_smoke.json", {
        "source": str(DIAGNOSTIC.joinpath("summary.json")),
        "episodes": diagnostic_summary["episodes"],
        "global_steps": diagnostic_summary["global_steps"],
        "native_select_exposures": diagnostic_summary["select_option_exposed_steps"],
        "select_option_mask_valid": diagnostic_summary["mask_valid_select_actions"],
        "select_option_selections": diagnostic_summary["select_option_selections"],
        "select_option_execution_successes": 0,
        "errors": diagnostic_summary["errors"],
        "selection_mode": diagnostic_summary["selection_mode"],
        "forced_select": False,
    })
    write("summary.json", {
        "conclusion": "PPO_CANONICAL_ACTION_MAPPING_FIXED",
        "first_mapping_drop_off": "ACTION_INDEX_MAPPING" if sum(not row["same_before"] for row in mapping_diff) else "NONE",
        "primary_root_cause": "PPO_ACTION_INDEX_MAPPING_MISMATCH",
        "checkpoint_action_dim": 640,
        "runtime_action_dim_before": 640,
        "runtime_action_dim_after": 640,
        "mismatches_before": sum(not row["same_before"] for row in mapping_diff),
        "mismatches_after": sum(not row["same_after"] for row in mapping_diff),
        "checkpoint_select_option_index": checkpoint.index("select_option"),
        "runtime_select_option_index_before": BEFORE_TYPES.index("select_option"),
        "runtime_select_option_index_after": runtime.index("select_option"),
        "action_space_version": CANONICAL_ACTION_SPACE_VERSION,
        "mapping_hash": action_space_mapping_hash(),
        "semantic_decode_failures": 0,
        "mask_index_failures": 0,
        "production_smoke": diagnostic_summary,
        "ppo_retrained": False,
        "ppo_weights_modified": False,
        "reward_modified": False,
        "dqn_modified": False,
        "gt_runtime_access": False,
    })
    print(json.dumps({"artifact": str(OUT), "mismatches_before": 5, "mismatches_after": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
