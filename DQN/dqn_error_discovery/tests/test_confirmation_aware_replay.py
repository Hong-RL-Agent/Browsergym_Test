import numpy as np

from dqn_error_discovery.agents.dqn_agent import ConfirmationAwareReplayBuffer, Transition
from dqn_error_discovery.runners.train_dqn_error_detector import build_replay_metadata


def _transition(index: int, metadata: dict[str, bool] | None = None) -> Transition:
    state = np.full(3, index, dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    return Transition(state, index % 4, float(index), state.copy(), False, mask, mask, metadata)


def test_successful_confirmation_admission_and_metadata():
    tags = build_replay_metadata(
        confirmation_context=True, action_replay=True, reproduction_attempt=True,
        candidate_relevant_effect=True, same_chain_recurrence=True, confirmation_upgrade=True,
    )
    assert tags["is_confirmation_context"] is True
    assert tags["is_successful_reproduction"] is True
    replay = ConfirmationAwareReplayBuffer(20)
    replay.add(_transition(1, tags))
    assert replay.retention_stats()["confirmation_context"] == 1
    assert replay.retention_stats()["successful_reproduction"] == 1


def test_normal_transition_is_not_confirmation_transition():
    tags = build_replay_metadata(
        confirmation_context=False, action_replay=False, reproduction_attempt=False,
        candidate_relevant_effect=False, same_chain_recurrence=False, confirmation_upgrade=False,
    )
    replay = ConfirmationAwareReplayBuffer(20)
    replay.add(_transition(1, tags))
    assert replay.retention_stats()["general"] == 1
    assert replay.retention_stats()["confirmation_context"] == 0


def test_failed_reproduction_is_retained_but_not_successful():
    tags = build_replay_metadata(
        confirmation_context=True, action_replay=True, reproduction_attempt=True,
        candidate_relevant_effect=True, same_chain_recurrence=False, confirmation_upgrade=False,
    )
    assert tags["is_failed_reproduction"] is True
    assert tags["is_successful_reproduction"] is False
    replay = ConfirmationAwareReplayBuffer(20)
    replay.add(_transition(1, tags))
    assert replay.retention_stats()["confirmation_context"] == 1
    assert replay.retention_stats()["successful_reproduction"] == 0


def test_stratified_sampling_guarantees_success_when_available():
    replay = ConfirmationAwareReplayBuffer(100, confirmation_fraction=0.10)
    success = build_replay_metadata(confirmation_context=True, action_replay=True, reproduction_attempt=True, candidate_relevant_effect=True, same_chain_recurrence=True, confirmation_upgrade=True)
    failed = build_replay_metadata(confirmation_context=True, action_replay=True, reproduction_attempt=True, candidate_relevant_effect=False, same_chain_recurrence=False, confirmation_upgrade=False)
    for index in range(19):
        replay.add(_transition(index, success if index == 0 else failed))
    for index in range(19, 119):
        replay.add(_transition(index, None))
    batch = replay.sample(32, seed=42)
    assert any((item.replay_metadata or {}).get("is_successful_reproduction") for item in batch)
    assert sum(bool((item.replay_metadata or {}).get("is_confirmation_context")) for item in batch) >= 1
    assert sum(not bool((item.replay_metadata or {}).get("is_confirmation_context")) for item in batch) >= 1


def test_empty_confirmation_falls_back_to_general_uniform_sampling():
    replay = ConfirmationAwareReplayBuffer(20)
    for index in range(10):
        replay.add(_transition(index))
    batch = replay.sample(4, seed=7)
    assert len(batch) == 4
    assert all(not bool((item.replay_metadata or {}).get("is_confirmation_context")) for item in batch)
    assert replay.sample_telemetry["confirmation_samples"] == 0


def test_sampling_seed_is_deterministic():
    first = ConfirmationAwareReplayBuffer(20)
    second = ConfirmationAwareReplayBuffer(20)
    for index in range(10):
        item = _transition(index)
        first.add(item)
        second.add(_transition(index))
    assert [int(item.state[0]) for item in first.sample(5, seed=123)] == [int(item.state[0]) for item in second.sample(5, seed=123)]
