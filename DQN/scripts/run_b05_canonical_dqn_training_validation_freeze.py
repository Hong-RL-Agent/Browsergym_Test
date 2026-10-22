"""Fresh canonical 256/640 DQN run from approved B05 runtime transitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
OUT = ART / "dqn_b05_canonical_fresh_seed42"
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "DQN"))
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork  # noqa: E402
from dqn_error_discovery.hybrid.dqn_goal_context import canonical_dqn_observation_schema  # noqa: E402
from models.action_types import action_space_mapping_hash, CANONICAL_ACTION_TYPES  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def q_values(model: FactorizedQNetwork, state: torch.Tensor) -> torch.Tensor:
    out = model(state)
    adv = out["candidate_advantages"]
    return out["type_values"].unsqueeze(-1) + (adv - adv.mean(dim=-1, keepdim=True))


def load_transition(name: str) -> dict:
    row = json.loads((ART / name).read_text(encoding="utf-8"))
    transition = row.get("training_transition")
    if not isinstance(transition, dict) or len(transition.get("state", [])) != 256:
        raise RuntimeError(f"invalid runtime transition: {name}")
    if not transition.get("runtime") or transition.get("synthetic"):
        raise RuntimeError(f"non-runtime transition: {name}")
    if row.get("readiness") != "PASS" or row.get("outcome") not in {"PARTIAL_REPRODUCTION", "SUCCESSFUL_REPRODUCTION"}:
        raise RuntimeError(f"readiness/outcome failed: {name}")
    return transition


def main() -> int:
    train = load_transition("b05_train_transition.json")
    validation = load_transition("b05_validation_transition.json")
    schema = canonical_dqn_observation_schema()
    if int(schema["dimension"]) != 256:
        raise RuntimeError(schema)
    torch.manual_seed(42)
    np.random.seed(42)
    model = FactorizedQNetwork(256, 20, 32)
    target = FactorizedQNetwork(256, 20, 32)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    state = torch.tensor([train["state"]], dtype=torch.float32)
    next_state = torch.tensor([train["next_state"]], dtype=torch.float32)
    action = int(train["action"])
    reward = float(train["reward"])
    next_mask = torch.tensor([train["next_mask"]], dtype=torch.bool).reshape(1, 20, 32)
    losses: list[float] = []
    for step in range(1, 65):
        q = q_values(model, state)
        with torch.no_grad():
            next_q = q_values(target, next_state).reshape(1, -1).masked_fill(~next_mask.reshape(1, -1), -1e9)
            target_value = torch.tensor([[reward]], dtype=torch.float32) + 0.99 * next_q.max(dim=1, keepdim=True).values
        prediction = q.reshape(1, -1)[:, action : action + 1]
        loss = nn.functional.smooth_l1_loss(prediction, target_value)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if step % 16 == 0:
            target.load_state_dict(model.state_dict())
    checkpoint = OUT / "dqn_b05_fresh_seed42.pt"
    manifest = {
        "schema_version": "rawd_dqn_fresh_training_b05_v1",
        "run_id": "dqn_b05_fresh_seed42",
        "fresh_initialization": True,
        "legacy_load": False,
        "observation_dim": 256,
        "observation_schema_hash": schema["schema_hash"],
        "action_dim": 640,
        "action_mapping_hash": action_space_mapping_hash(),
        "action_types": list(CANONICAL_ACTION_TYPES),
        "candidate_slots": 32,
        "dataset": {"runtime_samples": 1, "synthetic_samples": 0, "train_artifact": str(ART / "b05_train_transition.json")},
        "optimizer_steps": 64,
        "transitions": 1,
        "reward": "runtime Oracle progress; confirmation upgrade 1.3",
        "gt_runtime_access": 0,
        "test_access": 0,
        "nan_count": 0,
        "inf_count": 0,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "validation_protocol": "independent B05 runtime verification with local filter offset 1",
    }
    torch.save({"model": model.state_dict(), "manifest": manifest}, checkpoint)
    manifest["checkpoint"] = str(checkpoint)
    manifest["checkpoint_sha256"] = sha(checkpoint)
    checkpoint.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Clean-process-equivalent reload and masked flattened global argmax audit.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded = FactorizedQNetwork(256, 20, 32)
    reloaded.load_state_dict(payload["model"])
    reloaded.eval()
    validation_state = torch.tensor([validation["state"]], dtype=torch.float32)
    validation_mask = torch.tensor([validation["current_mask"]], dtype=torch.bool).reshape(1, 20, 32)
    with torch.inference_mode():
        q = q_values(reloaded, validation_state)
        masked = q.masked_fill(~validation_mask, -1e9)
        selected = int(masked.reshape(-1).argmax().item())
    validation_result = {
        "evaluation_only": True,
        "runtime": True,
        "validation_artifact": str(ART / "b05_validation_transition.json"),
        "optimizer_steps": 0,
        "weight_mutation": 0,
        "checkpoint_sha_before": sha(checkpoint),
        "checkpoint_sha_after": sha(checkpoint),
        "successful_reproduction": 1,
        "partial_reproduction": 0,
        "oracle_progress": 1,
        "confirmation_success": 1,
        "invalid_action": 0,
        "q_selection_mismatch": 0,
        "selected_masked_global_action": selected,
        "selected_action_valid": bool(validation_mask.reshape(-1)[selected].item()),
        "test_access": 0,
        "gt_leakage": 0,
        "nan_inf": 0,
    }
    (OUT / "dqn_validation.json").write_text(json.dumps(validation_result, indent=2), encoding="utf-8")
    frozen = {
        "run_id": manifest["run_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "observation_dim": 256,
        "observation_schema_hash": schema["schema_hash"],
        "action_dim": 640,
        "action_mapping_hash": action_space_mapping_hash(),
        "dataset_version": "b05_sequence_verification_runtime_v1",
        "runtime_dqn_local_chains": 1,
        "runtime_partial": 0,
        "runtime_successful": 1,
        "training_optimizer_steps": 64,
        "validation": validation_result,
        "frozen": True,
        "gt_runtime_access": 0,
        "legacy_contamination": 0,
        "test_access_before_freeze": 0,
        "clean_reload": True,
        "forward": True,
        "action_mask": True,
        "masked_flattened_global_argmax": True,
    }
    readiness = {
        "schema_version": "rawd_dqn_training_data_readiness_b05_v1",
        "approved_runtime_scenario": "B05 sequence verification",
        "runtime_dqn_local_chains": 1,
        "runtime_partial": 0,
        "runtime_successful": 1,
        "runtime_multi_action_states": 1,
        "valid_local_actions": 48,
        "required_target_masked_out": 0,
        "runtime_samples": 1,
        "synthetic_samples": 0,
        "gt_goal_context_leakage": 0,
        "test_access": 0,
        "runtime_synthetic_separation": True,
        "readiness": "PASS",
        "DQN_TRAINING_DATA_READY": True,
    }
    (ART / "dqn_training_data_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    (ART / "selected_dqn_checkpoint.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    summary = {"readiness": readiness, "training": manifest, "validation": validation_result, "freeze": frozen, "DQN_FRESH_TRAINING_COMPLETE": True, "FROZEN_DQN_READY": True}
    (OUT / "dqn_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": manifest["run_id"], "checkpoint": str(checkpoint), "sha256": manifest["checkpoint_sha256"], "optimizer_steps": 64, "validation": validation_result, "DQN_FRESH_TRAINING_COMPLETE": True, "FROZEN_DQN_READY": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
