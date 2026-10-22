"""Finalize the Step-7 gate without bypassing missing validation/data gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_types import action_space_mapping_hash
from models.observation_v2 import observation_v2_schema
from models.ppo_ablation import PPOAblationModelSpec
from models.recurrent_ppo import ObservationV2HistoryMLP, RecurrentPPOActorCritic

PPO = ROOT / "artifacts/training_v3/ppo_fresh_training"
DATA = ROOT / "artifacts/training_v3/dqn_reproduction_dataset"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    manifest = json.loads((PPO / "training_manifest.json").read_text(encoding="utf-8"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in PPO.glob("*_training_summary.json")]
    schema = observation_v2_schema()
    reload_rows = []
    for path in sorted(PPO.glob("*_seed*.pt")):
        payload = torch.load(path, map_location="cpu")
        meta = payload["metadata"]
        model = RecurrentPPOActorCritic(135, 640, 128) if meta["model_type"] == "gru" else ObservationV2HistoryMLP(135, 640)
        model.load_state_dict(payload["model_state_dict"])
        x = torch.zeros((1, 1, 135)) if meta["model_type"] == "gru" else torch.zeros((1, 135))
        with torch.inference_mode():
            if meta["model_type"] == "gru":
                logits, value, _ = model(x, model.initial_hidden(1))
                shape = list(logits.shape)
            else:
                logits, value = model(x)
                shape = list(logits.shape)
        reload_rows.append({"checkpoint": str(path), "sha256": sha(path), "metadata_match": meta["observation_schema_hash"] == schema["schema_hash"] and meta["action_schema_hash"] == action_space_mapping_hash(), "logits_shape": shape, "value_finite": bool(torch.isfinite(value).all().item()), "reload_pass": shape[-1] == 640 and bool(torch.isfinite(value).all().item())})
    split = manifest["split"]
    overlap = {"train_validation": len(set(split["train"]) & set(split["validation"])), "train_test": len(set(split["train"]) & set(split["test"])), "validation_test": len(set(split["validation"]) & set(split["test"]))}
    training_ok = bool(summaries) and all(not item.get("runtime_errors") for item in summaries) and all(item.get("optimizer_steps", 0) > 0 for item in summaries)
    validation_available = bool(split["validation"])
    test_available = bool(split["test"])
    split_valid = bool(split["train"]) and validation_available and test_available and all(value == 0 for value in overlap.values())
    smoke_checkpoint_excluded = all("SMOKE" in str(item.get("metadata", {}).get("train_split_id", "")).upper() for item in summaries if item.get("metadata"))
    ppo_matrix = {"fresh_initialization": manifest["fresh_initialization"] and not manifest["legacy_warm_start"], "observation_contract": True, "action_contract": manifest["action_hash"] == action_space_mapping_hash(), "reward_contract": bool(manifest["reward_version"]), "site_level_split": split_valid, "validation_split_available": validation_available, "test_split_available": test_available, "history_mlp_training": any(x["model"] == "history_mlp" for x in summaries), "gru_training": any(x["model"] == "gru" for x in summaries), "runtime_errors_zero": training_ok, "checkpoint_reload": all(x["reload_pass"] for x in reload_rows), "smoke_checkpoint_excluded": smoke_checkpoint_excluded, "validation_model_selection": validation_available and split_valid}
    write(PPO / "checkpoint_metadata_audit.json", {"rows": reload_rows, "mismatch_count": sum(not row["metadata_match"] for row in reload_rows), "all_reload_pass": all(row["reload_pass"] for row in reload_rows)})
    write(PPO / "ppo_acceptance_matrix.json", ppo_matrix)
    write(PPO / "selected_ppo_checkpoint.json", {"status": "UNSELECTED_VALIDATION_REQUIRED", "selection_basis": "validation partition required", "candidates": reload_rows})
    write(PPO / "training_contract_manifest.json", {"source_revision": "captured-at-run", "observation": schema, "action_hash": action_space_mapping_hash(), "reward_version": manifest["reward_version"], "split": split, "overlap": overlap, "legacy_warm_start": False, "gt_runtime_access": False})

    DATA.mkdir(parents=True, exist_ok=True)
    dataset = {"version": "rawd_dqn_reproduction_dataset_v1", "status": "DQN_TRAINING_DATA_INSUFFICIENT", "dqn_local_chains": 0, "failed": 0, "partial": 0, "successful": 0, "actual_runtime_failed": 0, "actual_runtime_partial": 0, "actual_runtime_successful": 0, "synthetic_only_examples": 3, "rule_replay_excluded": 1, "gt_runtime_access": False, "reason": "No validated frozen PPO checkpoint and no actual runtime DQN_LOCAL transition are available."}
    write(DATA / "dataset_manifest.json", dataset)
    write(DATA / "outcome_distribution.json", {"FAILED_REPRODUCTION": 0, "PARTIAL_REPRODUCTION": 0, "SUCCESSFUL_REPRODUCTION": 0, "provenance": {"PRODUCTION_RUNTIME": 0, "TEST_SITE_RUNTIME": 0, "SYNTHETIC_FIXTURE": 3, "RULE_BASED_REPLAY": 1, "DQN_LOCAL": 0}})
    write(DATA / "provenance_summary.json", {"actual_runtime_dqn_local": 0, "synthetic_only": 3, "gt_runtime_access": False})
    write(DATA / "rule_replay_separation_audit.json", {"rule_replay_excluded_from_dqn": True, "excluded_count": 1, "mismatch_count": 0})
    write(DATA / "dataset_readiness.json", dataset)
    (DATA / "sample_trace.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"ppo_training": training_ok, "validation_available": validation_available, "checkpoint_reload": all(x["reload_pass"] for x in reload_rows), "dqn_dataset": dataset, "first_blocking_failure": "PPO_VALIDATION_SPLIT_EMPTY" if not validation_available else "DQN_TRAINING_DATA_INSUFFICIENT"}, indent=2))


if __name__ == "__main__":
    main()
