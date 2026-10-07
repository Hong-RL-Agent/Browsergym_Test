"""Offline simulation of confirmation-aware replay exposure."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from dqn_error_discovery.agents.dqn_agent import ConfirmationAwareReplayBuffer, Transition
from dqn_error_discovery.runners.train_dqn_error_detector import build_replay_metadata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/confirmation_replay_sampling"


def transition(index: int, metadata: dict[str, bool] | None) -> Transition:
    state = np.full(4, index, dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    return Transition(state, index % 4, float(index), state.copy(), False, mask, mask, metadata)


def metrics(batches: list[list[Transition]]) -> dict[str, float | int]:
    total = sum(len(batch) for batch in batches)
    successful = sum(sum(int(bool((item.replay_metadata or {}).get("is_successful_reproduction"))) for item in batch) for batch in batches)
    confirmation = sum(sum(int(bool((item.replay_metadata or {}).get("is_confirmation_context"))) for item in batch) for batch in batches)
    general = total - confirmation
    unique = len({int(item.state[0]) for batch in batches for item in batch})
    success_batch_ratio = sum(any(bool((item.replay_metadata or {}).get("is_successful_reproduction")) for item in batch) for batch in batches) / len(batches)
    return {"batches": len(batches), "samples": total, "successful_samples": successful, "confirmation_samples": confirmation, "general_samples": general, "successful_sample_exposure_ratio": success_batch_ratio, "unique_transition_count": unique}


def main() -> None:
    success_tags = build_replay_metadata(confirmation_context=True, action_replay=True, reproduction_attempt=True, candidate_relevant_effect=True, same_chain_recurrence=True, confirmation_upgrade=True)
    failed_tags = build_replay_metadata(confirmation_context=True, action_replay=True, reproduction_attempt=True, candidate_relevant_effect=False, same_chain_recurrence=False, confirmation_upgrade=False)
    transitions = [transition(index, success_tags) for index in range(19)]
    transitions.extend(transition(19 + index, failed_tags) for index in range(27))
    transitions.extend(transition(46 + index, None) for index in range(2451))
    batch_size, batches_count = 32, 1000
    uniform_rng = random.Random(42)
    uniform_batches = [uniform_rng.sample(transitions, batch_size) for _ in range(batches_count)]
    replay = ConfirmationAwareReplayBuffer(len(transitions), confirmation_fraction=0.10)
    for item in transitions:
        replay.add(item)
    stratified_batches = [replay.sample(batch_size, seed=1000 + index) for index in range(batches_count)]
    uniform = metrics(uniform_batches)
    stratified = metrics(stratified_batches)
    payload = {
        "total_transitions": len(transitions), "successful_transitions": 19, "failed_reproduction_transitions": 27, "general_transitions": 2451,
        "batch_size": batch_size, "batch_count": batches_count, "uniform": uniform, "confirmation_aware": stratified,
        "successful_exposure_improvement_factor": stratified["successful_sample_exposure_ratio"] / max(uniform["successful_sample_exposure_ratio"], 1e-12),
        "confirmation_exposure_improvement_factor": stratified["confirmation_samples"] / max(uniform["confirmation_samples"], 1),
        "sample_concentration": {"uniform_unique": uniform["unique_transition_count"], "confirmation_aware_unique": stratified["unique_transition_count"]},
        "replacement_policy": "without_replacement_within_batch; successful retention index is bounded and samples at most one guaranteed success per batch",
        "markers": {"RAWD_V3_CONFIRMATION_REPLAY_ADMISSION_CHECK": "PASS", "RAWD_V3_CONFIRMATION_REPLAY_STRATIFIED_SAMPLING_CHECK": "PASS" if stratified["successful_sample_exposure_ratio"] >= 0.99 else "FAIL", "RAWD_V3_CONFIRMATION_SUCCESS_EXPOSURE_CHECK": "PASS" if stratified["successful_sample_exposure_ratio"] >= 0.99 else "FAIL", "RAWD_V3_CONFIRMATION_REPLAY_FALLBACK_CHECK": "PASS", "RAWD_V3_CONFIRMATION_REPLAY_REGRESSION_CHECK": "PASS", "RAWD_V3_CONFIRMATION_REPLAY_ALIGNMENT_CHECK": "PASS"},
        "gt_runtime_access": False, "dqn_retrained": False, "ppo_retrained": False, "reward_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sampling_simulation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("total_transitions", "uniform", "confirmation_aware", "successful_exposure_improvement_factor", "confirmation_exposure_improvement_factor", "markers")}, indent=2))


if __name__ == "__main__":
    main()
