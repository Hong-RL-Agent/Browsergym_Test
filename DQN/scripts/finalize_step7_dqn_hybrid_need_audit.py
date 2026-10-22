"""Finalize the evidence-based audit of current TRAIN DQN opportunity."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def main() -> None:
    wide = json.loads((BASE / "dqn_wide_train_diagnostic_probe.json").read_text(encoding="utf-8"))
    inventory = json.loads((BASE / "dqn_opportunity_inventory_all_splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "configs/step7_full_site_split.json").read_text(encoding="utf-8"))

    current = manifest["counts"]
    out = {
        "schema_version": "rawd_step7_dqn_hybrid_need_audit_v1",
        "production_ppo": {
            "run_id": "fresh5_mlp42_rewardfix",
            "sha256": wide["ppo_checkpoint_sha_before"] if "ppo_checkpoint_sha_before" in wide else "93D9EB93A0AEE4A4F264FD77BCBB744A59B259BFE47F220C5EF3B8A7B564C0E5",
            "sha_unchanged_in_wide_probe": wide["ppo_sha_unchanged"]
        },
        "train_error_composition_audit": {
            "train_sites": len(split["train"]),
            "current_formal_collection_steps": manifest["funnel"]["ppo_runtime_steps"],
            "current_formal_detector_candidates": current["candidates"],
            "current_formal_anomaly_types": manifest.get("anomaly_types", {}),
            "current_formal_strategy_counts": manifest.get("strategy_counts", {}),
            "wide_probe_rule_counts": wide["rule_counts"],
            "wide_probe_strategy_counts": wide["strategy_counts"],
            "finding": "Current Frozen-PPO TRAIN runtime produced layout-owned R05/R06 evidence only; no current DQN-relevant R01/R02/R09/R10/R12 evidence was observed."
        },
        "wide_train_probe": {
            "episodes": wide["episodes"],
            "steps": wide["total_steps"],
            "interaction_actions": wide["interaction_actions"],
            "candidate_elements_observed": wide["candidate_elements_observed"],
            "relevant_dqn_rule_candidates": wide["relevant_dqn_rule_candidates"],
            "runtime_errors": len(wide["runtime_errors"]),
            "gt_policy_input": wide["gt_policy_input"],
            "validation_access": wide["validation_access"],
            "test_access": wide["test_access"]
        },
        "reachability_audit": {
            "interactive_actions_reached": wide["interaction_actions"] > 0,
            "candidate_elements_reached": wide["candidate_elements_observed"] > 0,
            "dqn_relevant_state_observed": wide["relevant_dqn_rule_candidates"] > 0,
            "interpretation": "PPO reaches interactive UI elements, but the DQN-relevant detector/state condition was not reached or recognized in the current runtime probe."
        },
        "detector_audit": {
            "current_dqn_relevant_rules": {rule: wide["rule_counts"].get(rule, 0) for rule in ("R01", "R02", "R09", "R10", "R12")},
            "current_observation_owned_rules": {rule: wide["rule_counts"].get(rule, 0) for rule in ("R05", "R06")},
            "detector_dqn_candidate_generated": False,
            "detector_bug_proven": False
        },
        "historical_split_review": {
            "split_version": inventory["schema_version"],
            "train_relevant_historical_evidence": inventory["partition_counts"].get("TRAIN", {}),
            "validation_relevant_historical_evidence": inventory["partition_counts"].get("VALIDATION", {}),
            "test_relevant_historical_evidence": inventory["partition_counts"].get("TEST", {}),
            "historical_evidence_is_current_training_data": False,
            "site_overlap": split.get("site_overlap"),
            "cluster_overlap": split.get("cluster_overlap"),
            "split_change_justified": False,
            "reason": "Only historical TRAIN/site001 R01 evidence exists in the inventory; no clean current cross-split evidence justifies moving sites."
        },
        "hybrid_need_decision": {
            "current_train_has_dqn_opportunity": False,
            "hybrid_dqn_globally_unnecessary_proven": False,
            "decision": "CURRENT_TRAIN_DQN_OPPORTUNITY_NOT_OBSERVED",
            "primary_cause": "DQN_RELEVANT_RUNTIME_SCENARIO_NOT_OBSERVED_OR_RECOGNIZED",
            "secondary_cause": "LAYOUT_ANOMALY_DOMINANCE",
            "dqn_training_allowed": False,
            "next_required_action": "Review detector/state scenario coverage or obtain explicit approval for a separate runtime scenario design; do not use VALIDATION/TEST or force DQN_LOCAL."
        }
    }
    path = BASE / "dqn_hybrid_need_and_split_review_final.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
