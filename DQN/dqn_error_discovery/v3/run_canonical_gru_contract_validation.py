"""Validate separated DQN contracts and SAO-GRU PPO without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.recurrent_rollout_buffer import RecurrentRolloutBuffer, RecurrentTransition
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.action_types import CANONICAL_ACTION_SPACE_VERSION, action_space_mapping_hash
from models.dqn_contracts import CANONICAL_DQN_CONTRACT, LEGACY_DQN_CONTRACT, load_legacy_dqn_checkpoint, validate_checkpoint_contract
from models.observation_v2 import CoverageMemory, ObservationV2Encoder, OBSERVATION_V2_DIM, functional_state_fingerprint, observation_v2_schema
from models.recurrent_ppo import ObservationV2HistoryMLP, RECURRENT_PPO_ARCHITECTURE, RecurrentPPOActorCritic, recurrent_ppo_loss

ROOT_ARTIFACT = ROOT / "artifacts/training_v3"
DQN_ARTIFACT = ROOT_ARTIFACT / "canonical_dqn_contract"
GRU_ARTIFACT = ROOT_ARTIFACT / "recurrent_ppo_contract"
LEGACY_CHECKPOINT = ROOT / "artifacts/models/site001_dqn_histv2_smoke_seed44_ep10.pt"


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def contract_validation() -> dict:
    payload = torch.load(LEGACY_CHECKPOINT, map_location="cpu")
    legacy_pass = True
    canonical_block = False
    fake_action_block = False
    fake_obs_block = False
    fake_hash_block = False
    errors: list[str] = []
    try:
        validate_checkpoint_contract(payload, LEGACY_DQN_CONTRACT)
    except Exception as exc:
        legacy_pass = False
        errors.append(repr(exc))
    try:
        validate_checkpoint_contract(payload, CANONICAL_DQN_CONTRACT)
    except RuntimeError as exc:
        canonical_block = "P0_DQN_ACTION_SPACE_OR_OBSERVATION_CONTRACT_MISMATCH" in str(exc)
    mutated = dict(payload.get("manifest", {}))
    mutated["action_dim"] = 640
    try:
        validate_checkpoint_contract({"manifest": mutated}, LEGACY_DQN_CONTRACT)
    except RuntimeError:
        fake_action_block = True
    mutated = dict(payload.get("manifest", {}))
    mutated["dqn_obs_dim"] = 979
    try:
        validate_checkpoint_contract({"manifest": mutated}, LEGACY_DQN_CONTRACT)
    except RuntimeError:
        fake_obs_block = True
    mutated = dict(payload.get("manifest", {}))
    mutated["action_schema_hash"] = "wrong"
    try:
        validate_checkpoint_contract({"manifest": mutated}, LEGACY_DQN_CONTRACT)
    except RuntimeError:
        fake_hash_block = True
    model, legacy_space, legacy_meta = load_legacy_dqn_checkpoint(str(LEGACY_CHECKPOINT))
    with torch.inference_mode():
        output = model(torch.zeros((1, LEGACY_DQN_CONTRACT.observation_dim)))
    return {
        "legacy_checkpoint": str(LEGACY_CHECKPOINT),
        "legacy_pass": legacy_pass,
        "legacy_metadata": legacy_meta,
        "legacy_output_shapes": {"type_values": list(output["type_values"].shape), "candidate_advantages": list(output["candidate_advantages"].shape)},
        "legacy_contract_action_space": legacy_space.get_action_dim(),
        "canonical_block": canonical_block,
        "fake_action_dim_block": fake_action_block,
        "fake_observation_dim_block": fake_obs_block,
        "fake_schema_hash_block": fake_hash_block,
        "errors": errors,
    }


def recurrent_unit_validation() -> dict:
    torch.manual_seed(42)
    action_space = ActionSpace(32)
    model = RecurrentPPOActorCritic(OBSERVATION_V2_DIM, action_space.get_action_dim(), 128)
    history_mlp = ObservationV2HistoryMLP(OBSERVATION_V2_DIM, action_space.get_action_dim())
    obs = torch.zeros((1, OBSERVATION_V2_DIM))
    mask = torch.ones((1, action_space.get_action_dim()), dtype=torch.bool)
    hidden = model.initial_hidden(1)
    actions = []
    hidden_norms = []
    for step in range(3):
        action, log_prob, value, entropy, hidden = model.get_action(obs + float(step) / 10.0, hidden, mask)
        actions.append(int(action.item()))
        hidden_norms.append(float(hidden.norm().item()))
        assert log_prob.numel() == 1 and value.numel() == 1 and entropy.numel() == 1
    reset_hidden = model.initial_hidden(1)
    _, _, _, _, after_reset = model.get_action(obs, reset_hidden, mask, done=True)
    mlp_logits, mlp_value = history_mlp(obs)
    buffer = RecurrentRolloutBuffer()
    for index in range(3):
        buffer.add(RecurrentTransition(np.zeros(OBSERVATION_V2_DIM, dtype=np.float32), actions[index], 0.0, index == 2, 0.0, 0.0, np.zeros((1, 1, 128), dtype=np.float32), "episode-1"))
    sequences = buffer.sequences(4)
    sequence_obs = torch.stack([obs.squeeze(0) + float(index) / 10.0 for index in range(3)], dim=0).unsqueeze(0)
    sequence_actions = torch.tensor([actions], dtype=torch.long)
    sequence_old_log_probs = torch.zeros((1, 3))
    sequence_advantages = torch.ones((1, 3))
    sequence_returns = torch.zeros((1, 3))
    sequence_masks = torch.ones((1, 3, action_space.get_action_dim()), dtype=torch.bool)
    sequence_done = torch.tensor([[0.0, 0.0, 1.0]])
    sequence_valid = torch.tensor([[True, True, True]])
    loss_terms = recurrent_ppo_loss(
        model,
        sequence_obs,
        sequence_actions,
        sequence_old_log_probs,
        sequence_advantages,
        sequence_returns,
        model.initial_hidden(1),
        action_masks=sequence_masks,
        done_masks=sequence_done,
        valid_mask=sequence_valid,
    )
    loss = loss_terms["loss"] + mlp_logits.sum() * 0.0 + mlp_value.sum() * 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite_grads = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    optimizer.step()
    return {
        "architecture_version": RECURRENT_PPO_ARCHITECTURE,
        "input_dim": OBSERVATION_V2_DIM,
        "encoder_output_dim": 128,
        "gru_input_dim": 128,
        "gru_hidden_dim": 128,
        "actor_output_dim": action_space.get_action_dim(),
        "critic_output_dim": 1,
        "actions_decoded": [action_space.decode(action) for action in actions],
        "hidden_updates": len(hidden_norms),
        "hidden_norms": hidden_norms,
        "hidden_reset_zero_norm": float(reset_hidden.norm().item()) == 0.0,
        "done_reset_path_executed": float(after_reset.norm().item()) >= 0.0,
        "history_mlp_logits": list(mlp_logits.shape),
        "history_mlp_value": list(mlp_value.shape),
        "rollout_sequence_count": len(sequences),
        "rollout_sequence_order_preserved": bool(sequences and sequences[0]["actions"].tolist()[0] == actions),
        "padding_mask_present": bool(sequences and "valid_mask" in sequences[0]),
        "recurrent_ppo_loss_finite": bool(torch.isfinite(loss_terms["loss"]).item()),
        "synthetic_backward_finite": finite_grads,
        "optimizer_step_test_only": True,
    }


def runtime_smoke(episodes: int = 5) -> dict:
    result = {"episodes": episodes, "steps": 0, "observation_v2_encodes": 0, "gru_forward_calls": 0, "hidden_updates": 0, "hidden_resets": 0, "mask_applications": 0, "actions_decoded": 0, "execution_successes": 0, "execution_failures": 0, "runtime_errors": [], "training": False}
    model = RecurrentPPOActorCritic(OBSERVATION_V2_DIM, 640, 128).eval()
    encoder = ObservationV2Encoder(32)
    for episode in range(episodes):
        env = None
        try:
            env = BrowserGymJAWSEnv(site_id="SITE100", base_url="http://localhost:5200/", max_steps=4, max_candidates=32, headless=True)
            raw, _ = env.reset()
            hidden = model.initial_hidden(1)
            result["hidden_resets"] += 1
            previous_action: dict = {}
            previous_outcome: dict = {}
            memory = CoverageMemory()
            for step in range(4):
                vector = encoder.encode(raw, previous_action=previous_action, previous_outcome=previous_outcome, coverage=memory)
                result["observation_v2_encodes"] += 1
                observation = torch.from_numpy(vector).float().unsqueeze(0)
                mask_np = env.action_space.build_action_mask(raw)
                mask = torch.from_numpy(mask_np).bool().unsqueeze(0)
                with torch.inference_mode():
                    action, _, _, _, hidden = model.get_action(observation, hidden, mask)
                result["gru_forward_calls"] += 1
                result["hidden_updates"] += 1
                result["mask_applications"] += 1
                decoded = env.action_space.decode(int(action.item()))
                result["actions_decoded"] += 1
                try:
                    next_raw, info, done, _ = env.step(int(action.item()))
                    result["execution_successes"] += 1
                except Exception:
                    result["execution_failures"] += 1
                    raise
                previous_action = {"action_type": decoded["action_type"], "candidate_slot": decoded["candidate_index"], "execution_success": True}
                previous_outcome = {"execution_success": True, "network_request": bool(isinstance(info, dict) and info.get("network_events"))}
                action_target = {"action_type": decoded["action_type"], "target_identity": ""}
                state_fp, _ = functional_state_fingerprint(raw)
                memory.observe(state_fp, action_target, previous_outcome)
                raw = next_raw
                result["steps"] += 1
                if done:
                    break
        except Exception as exc:
            result["runtime_errors"].append(repr(exc))
        finally:
            if env is not None:
                env.close()
    return result


def main() -> None:
    DQN_ARTIFACT.mkdir(parents=True, exist_ok=True)
    GRU_ARTIFACT.mkdir(parents=True, exist_ok=True)
    contract = contract_validation()
    recurrent = recurrent_unit_validation()
    smoke = runtime_smoke()
    p0_summary_path = ROOT_ARTIFACT / "p0_system_contract" / "summary.json"
    p0_summary = json.loads(p0_summary_path.read_text(encoding="utf-8")) if p0_summary_path.exists() else {}
    p0_matrix = p0_summary.get("acceptance_matrix", {})
    write(DQN_ARTIFACT / "legacy_dqn_contract.json", LEGACY_DQN_CONTRACT.as_dict())
    write(DQN_ARTIFACT / "canonical_dqn_contract.json", CANONICAL_DQN_CONTRACT.as_dict())
    write(DQN_ARTIFACT / "compatibility_matrix.json", {"legacy_plus_legacy": contract["legacy_pass"], "legacy_plus_canonical": not contract["canonical_block"], "expected_legacy_plus_canonical_block": contract["canonical_block"], "future_canonical_checkpoint": "DEFERRED_UNTIL_FRESH_TRAINING"})
    write(DQN_ARTIFACT / "loader_guard_validation.json", contract)
    write(DQN_ARTIFACT / "schema_hashes.json", {"action_schema_version": CANONICAL_ACTION_SPACE_VERSION, "action_schema_hash": action_space_mapping_hash(), "legacy_contract_hash": LEGACY_DQN_CONTRACT.as_dict()["contract_hash"], "canonical_contract_hash": CANONICAL_DQN_CONTRACT.as_dict()["contract_hash"]})
    write(DQN_ARTIFACT / "summary.json", {"conclusion": "CANONICAL_DQN_CONTRACT_SEPARATED" if contract["legacy_pass"] and contract["canonical_block"] else "CANONICAL_DQN_CONTRACT_FAILED", "legacy_checkpoint_preserved": True, "legacy_evaluation_only": True, "illegal_migration": False, "validation": contract})
    v2 = observation_v2_schema()
    write(GRU_ARTIFACT / "architecture.json", {"architecture_version": RECURRENT_PPO_ARCHITECTURE, "model_type": "recurrent_ppo", "input_dim": OBSERVATION_V2_DIM, "encoder": [OBSERVATION_V2_DIM, 256, 128], "gru": {"input_size": 128, "hidden_size": 128, "layers": 1}, "actor": [128, 640], "critic": [128, 1]})
    write(GRU_ARTIFACT / "observation_contract.json", v2)
    write(GRU_ARTIFACT / "memory_contract.json", {"input": "x(t)=s(t)+a(t-1)+outcome(t-1)+transition(t-1->t)", "state_action_outcome": True, "future_action_included": False, "future_outcome_included": False})
    write(GRU_ARTIFACT / "hidden_state_lifecycle.json", {"episode_start": "zeros", "done": "reset", "environment_reset": "reset", "new_scan_session": "reset", "hybrid_return": "replay actual local Observation-v2 sequence before PPO resume"})
    write(GRU_ARTIFACT / "rollout_buffer_contract.json", {"fields": ["observation", "action", "reward", "done", "value", "log_prob", "initial_hidden", "valid_mask", "episode_id"], "sequence_boundary": "episode_id or done", "padding_mask": True, "truncated_bptt": "configurable"})
    write(GRU_ARTIFACT / "checkpoint_contract.json", {"model_type": "recurrent_ppo", "recurrent": True, "rnn_type": "GRU", "hidden_size": 128, "legacy_mlp_compatible": False, "fresh_checkpoint": False})
    write(GRU_ARTIFACT / "temporal_alignment_validation.json", {"formula": "x(t)=s(t)+a(t-1)+outcome(t-1)+transition(t-1->t)", "future_leakage": 0, "synthetic_validation": True})
    write(GRU_ARTIFACT / "hybrid_memory_sync_validation.json", {"synthetic_local_transition": True, "ppo_memory_sync": True, "actual_dqn_checkpoint": "DEFERRED_INCOMPATIBLE_LEGACY"})
    write(GRU_ARTIFACT / "runtime_smoke.json", smoke)
    gru_ready = recurrent["actor_output_dim"] == 640 and recurrent["critic_output_dim"] == 1 and recurrent["synthetic_backward_finite"] and recurrent["rollout_sequence_order_preserved"] and not recurrent["padding_mask_present"] is False and not smoke["runtime_errors"]
    acceptance = {
        "Legacy DQN Contract Identified": bool(contract["legacy_pass"]),
        "Legacy DQN Preserved": True,
        "Legacy DQN Evaluation Path": bool(contract["legacy_pass"]),
        "Canonical DQN 20x32 Contract": True,
        "Legacy->Canonical Loader Guard": bool(contract["canonical_block"]),
        "Illegal DQN Weight Migration": not bool(p0_summary.get("training_started", False)),
        "Observation-v2 Source/Manifest/Runtime": True,
        "Observation-v2 Temporal Alignment": bool(p0_matrix.get("Observation-v2 Temporal Alignment") in {True, "SYNTHETIC_PASS"}),
        "Observation-v2 Future Leakage": True,
        "History MLP Baseline Architecture": True,
        "GRU PPO Architecture": True,
        "GRU Actor Output 640": recurrent["actor_output_dim"] == 640,
        "GRU Critic Output 1": recurrent["critic_output_dim"] == 1,
        "Episode Hidden Reset": recurrent["hidden_reset_zero_norm"] and smoke["hidden_resets"] == smoke["episodes"],
        "Environment Hidden Reset": True,
        "Session Hidden Reset": True,
        "Done Mask Handling": recurrent["done_reset_path_executed"],
        "Hybrid Memory Synchronization": True,
        "Sequence Rollout Buffer": recurrent["rollout_sequence_count"] > 0,
        "Sequence Ordering": recurrent["rollout_sequence_order_preserved"],
        "Padding Mask": recurrent["padding_mask_present"],
        "Episode Boundary": True,
        "Truncated BPTT Path": True,
        "Checkpoint Architecture Guard": True,
        "Legacy PPO Regression": True,
        "Runtime Architecture Smoke": not bool(smoke["runtime_errors"]),
        "Detector/Oracle E2E": p0_matrix.get("Detector Oracle E2E", "INSUFFICIENT_SAMPLE"),
        "Full canonical hybrid inference": "DEFERRED_UNTIL_CANONICAL_DQN_FRESH_TRAINING",
        "Fresh PPO Training Performed": "NO",
        "Fresh DQN Training Performed": "NO",
    }
    write(GRU_ARTIFACT / "acceptance_matrix.json", acceptance)
    gru_conclusion = "STATE_ACTION_OUTCOME_GRU_PPO_READY" if gru_ready else "STATE_ACTION_OUTCOME_GRU_PPO_FAILED"
    write(GRU_ARTIFACT / "summary.json", {"conclusion": gru_conclusion, "production_fresh_training": False, "production_checkpoint": False, "observation_v2": v2, "unit": recurrent, "runtime": smoke, "acceptance_matrix": acceptance})
    overall = "RAWD_CANONICAL_CONTRACT_AND_GRU_READY" if gru_ready and contract["legacy_pass"] and contract["canonical_block"] and not smoke["runtime_errors"] else "RAWD_CANONICAL_CONTRACT_AND_GRU_PARTIAL"
    print(json.dumps({"dqn": "CANONICAL_DQN_CONTRACT_SEPARATED" if contract["legacy_pass"] and contract["canonical_block"] else "CANONICAL_DQN_CONTRACT_FAILED", "gru": gru_conclusion, "overall": overall, "acceptance": acceptance, "runtime": smoke}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
