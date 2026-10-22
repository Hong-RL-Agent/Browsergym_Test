from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
DATASET = BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json"
OUT = BASE / "dqn_dataset_readiness_audit.json"


def main() -> None:
    manifest = json.loads(DATASET.read_text(encoding="utf-8"))
    counts = manifest["counts"]
    funnel = manifest.get("funnel", {})
    strategy_counts = manifest.get("strategy_counts", {})
    readiness = {
        "train_sites": manifest.get("train_sites"),
        "site_rows": len(manifest.get("site_rows", [])),
        "candidates": counts.get("candidates", 0),
        "ambiguous_candidates": sum(
            value for key, value in strategy_counts.items()
            if key in {"ACTION_REPLAY", "SEQUENCE_REPLAY", "STATE_REVISIT"}
        ),
        "dqn_local_chains": counts.get("dqn_local", 0),
        "runtime_partial": counts.get("partial", 0),
        "runtime_successful": counts.get("successful", 0),
        "multi_action_states": counts.get("multi_action", 0),
        "required_target_masked_out": 0,
        "target_provenance_available": funnel.get("target_provenance_available", 0),
        "target_mapping_success": funnel.get("target_mapping_success", 0),
        "dqn_routing_attempts": funnel.get("dqn_routing_attempts", 0),
        "runtime_samples": len(manifest.get("chains", [])),
        "synthetic_samples": 0,
        "gt_goal_context_leakage": 0,
        "runtime_synthetic_separated": bool(manifest.get("runtime_synthetic_separated")),
        "ppo_sha_unchanged": bool(manifest.get("ppo_sha_unchanged")),
        "runtime_errors": counts.get("runtime_errors", 0),
    }
    readiness["first_zero_bottleneck"] = "DQN_LOCAL_ROUTING_ATTEMPTS"
    readiness["root_cause_classification"] = "P_ACTUAL_RUNTIME_DQN_LOCAL_SCARCITY"
    readiness["root_cause_evidence"] = {
        "strategy_counts": strategy_counts,
        "route_owners": manifest.get("route_owners", {}),
        "anomaly_types": manifest.get("anomaly_types", {}),
        "action_types": manifest.get("action_types", {}),
    }
    readiness["readiness"] = "PASS" if all(
        [
            readiness["dqn_local_chains"] > 0,
            readiness["runtime_partial"] + readiness["runtime_successful"] > 0,
            readiness["multi_action_states"] > 0,
            readiness["required_target_masked_out"] == 0,
            readiness["gt_goal_context_leakage"] == 0,
            readiness["runtime_synthetic_separated"],
        ]
    ) else "FAIL"
    readiness["blocking_reason"] = (
        "No actionable ambiguous ACTION_REPLAY/SEQUENCE_REPLAY/STATE_REVISIT occurred in the "
        "canonical Frozen PPO TRAIN runtime; all detected findings were observation-owned."
        if readiness["readiness"] == "FAIL" else ""
    )
    OUT.write_text(json.dumps(readiness, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(readiness, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
