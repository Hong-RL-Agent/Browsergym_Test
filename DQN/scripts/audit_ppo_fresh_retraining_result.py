from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
RUNS = [
    ("history_mlp", 42, "fresh5_mlp42_rewardfix"),
    ("history_mlp", 43, "fresh5_mlp43_rewardfix"),
    ("history_mlp", 44, "fresh5_mlp44_rewardfix"),
    ("gru", 42, "fresh5_gru42_rewardfix"),
    ("gru", 43, "fresh5_gru43_rewardfix"),
    ("gru", 44, "fresh5_gru44_rewardfix"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    # Correct the typo defensively without changing the run contract.
    runs = [(a, s, ("fresh_gru44_rewardfix" if p.lower() == "fresh_gru44_rewardfix" else p)) for a, s, p in RUNS]
    validation = json.loads((OUT / "ppo_validation_results_canonical.json").read_text(encoding="utf-8"))
    by_run = {f"{x['architecture']}:{x['seed']}": x for x in validation["per_run"]}
    manifest = []
    for arch, seed, prefix in runs:
        meta_path = OUT / f"{prefix}_metadata.json"
        ckpt = OUT / f"{prefix}_full.pt"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        progress = (OUT / f"{prefix}_progress.jsonl").read_text(encoding="utf-8").splitlines()
        update_complete = [json.loads(x) for x in progress if '"event": "UPDATE_COMPLETE"' in x]
        training_lines = (OUT / f"{prefix}_training.jsonl").read_text(encoding="utf-8").splitlines()
        reward_ctx = [json.loads(x) for x in training_lines if '"event_type": "PPO_REWARD_CONTEXT"' in x]
        validation_run = by_run[f"{arch}:{seed}"]
        manifest.append({
            "architecture": arch,
            "seed": seed,
            "run_id": prefix,
            "checkpoint": str(ckpt),
            "checkpoint_file_sha256": sha(ckpt),
            "metadata_checkpoint_sha256": meta.get("checkpoint_file_sha256"),
            "completed_updates": len(update_complete),
            "final_update": update_complete[-1].get("update") if update_complete else None,
            "episodes": meta.get("episodes"),
            "steps": meta.get("steps"),
            "optimizer_steps": meta.get("optimizer_steps"),
            "observation_dim": meta.get("observation_schema", {}).get("schema_dim"),
            "observation_hash": meta.get("observation_schema", {}).get("schema_hash"),
            "action_dim": meta.get("action_dim"),
            "action_hash": meta.get("action_schema_hash"),
            "reward_version": meta.get("reward_version"),
            "training_contract": meta.get("training_contract_version"),
            "fresh_initialization": meta.get("legacy_initialization") is False,
            "gt_runtime_access": meta.get("gt_runtime_access"),
            "runtime_errors": meta.get("runtime_errors"),
            "reward_context_events": len(reward_ctx),
            "reward_context_gt_events": sum(float(x.get("gt_reward_used", 0)) != 0 for x in reward_ctx),
            "validation_integrity_gate": validation_run.get("integrity_gate"),
            "validation_functional_gate": validation_run.get("functional_exploration_gate"),
            "validation_metrics": {k: validation_run.get(k) for k in (
                "interaction_opportunity_steps", "interaction_selected", "click_opportunity_steps",
                "click_selected", "fill_opportunity_steps", "fill_selected", "select_opportunity_steps",
                "select_selected", "functional_state_transitions", "viewport_only_steps", "runtime_errors",
            )},
        })
    manifest_doc = {
        "schema_version": "rawd_ppo_fresh_retraining_manifest_v2",
        "runs": manifest,
        "exactly_six_runs": len(manifest) == 6,
        "all_full_training": all(x["completed_updates"] == 24 and x["final_update"] == 24 for x in manifest),
        "all_fresh_initialization": all(x["fresh_initialization"] for x in manifest),
        "validation_artifact": str(OUT / "ppo_validation_results_canonical.json"),
        "validation_pass": validation.get("validation_pass"),
    }
    (OUT / "ppo_fresh_retraining_manifest_v2.json").write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")
    candidate_count = sum(
            x["validation_functional_gate"].get("status") == "PASS" and x["validation_integrity_gate"] == "PASS"
            for x in manifest
        )
    blocker = {
        "status": "READY_FOR_PRODUCTION_SELECTION" if candidate_count else "RAWD_PPO_DQN_DATA_STAGE_PARTIAL",
        "last_completed_stage": "PPO_FRESH_TRAINING_AND_VALIDATION",
        "first_blocking_failure": None if candidate_count else "NO_FUNCTIONAL_EXPLORATION_GATE_PASS_CANDIDATE",
        "production_candidate_count": candidate_count,
        "reason": ("Functional and integrity gates passed for one or more fresh checkpoints."
                   if candidate_count else
                   "No fresh checkpoint passed both integrity and functional exploration gates."),
        "dqn_collection_started": False,
        "dqn_training_started": False,
        "evidence": str(OUT / "ppo_validation_results_canonical.json"),
        "manifest": str(OUT / "ppo_fresh_retraining_manifest_v2.json"),
        "safe_next_action": ("Select exactly one validation-eligible checkpoint, freeze it, clean-reload it, and run the production functional smoke before DQN collection."
                             if candidate_count else
                             "Diagnose/fix PPO functional exploration or reward/validation contract, then run a new fresh PPO training cycle; do not freeze a failed candidate or collect DQN data."),
    }
    (OUT / "ppo_retraining_selection_blocker.json").write_text(json.dumps(blocker, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(OUT / "ppo_fresh_retraining_manifest_v2.json"), "blocker": blocker}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
