import numpy as np
import pytest
import torch

from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, ReplayBuffer, Transition
from dqn_error_discovery.agents.history_observation import (
    HISTORY_CONTEXT_DIM,
    action_identity,
    append_history_transition,
    build_dqn_history_context,
)


ACTION_TYPES = ["noop", "click_element", "scroll_down", "scroll_up"]


def candidate(bid):
    return {"bid": str(bid), "role": "button", "visible": True, "bbox": [1, 2, 3, 4]}


def context(history=None, candidates=None):
    return build_dqn_history_context(history or {}, {"candidate_elements": candidates or []}, ACTION_TYPES, 32)


def test_empty_history_has_fixed_shape_and_zero_valid_mask():
    vector, metadata = context()
    assert vector.shape == (238,)
    assert np.all(vector[:76] == 0)
    assert metadata["history_valid_mask"] == [0, 0, 0, 0]
    assert vector[234:238].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_history_is_newest_first_and_action_onehot_is_correct():
    history = {"dqn_history_records": [
        {"action_type": "scroll_down", "action_identity": "scroll_down", "valid": 1},
        {"action_type": "click_element", "action_identity": "click_element:bid:1", "valid": 1},
    ]}
    vector, metadata = context(history, [candidate(1)])
    assert metadata["recent_action_identities"][:2] == ["click_element:bid:1", "scroll_down"]
    assert vector[1] == 1.0
    assert vector[76 + 0] == 1.0
    assert vector[76 + 1] == 0.0
    assert metadata["history_valid_mask"] == [1, 1, 0, 0]


def test_target_identity_remaps_when_candidate_slot_changes():
    history = {"dqn_history_records": [{
        "action_type": "click_element", "requires_target": 1,
        "target_identity": "bid:1", "action_identity": "click_element:bid:1", "valid": 1,
    }]}
    vector, metadata = context(history, [candidate(9), candidate(1)])
    assert metadata["target_current_slots"] == [[1], [], [], []]
    assert vector[80 + 1] == 1.0
    assert vector[80 + 0] == 0.0


def test_bbox_change_does_not_break_stable_identity_and_absent_target_is_hidden():
    history = {"dqn_history_records": [{
        "action_type": "click_element", "requires_target": 1,
        "target_identity": "bid:1", "action_identity": "click_element:bid:1", "valid": 1,
    }]}
    changed = dict(candidate(1), bbox=[100, 200, 300, 400])
    vector, metadata = context(history, [changed])
    assert metadata["target_current_slots"] == [[0], [], [], []]
    assert vector[208] == 1.0
    vector, metadata = context(history, [candidate(2)])
    assert metadata["target_current_slots"] == [[], [], [], []]
    assert vector[208] == 0.0
    assert not vector[80:112].any()


def test_candidate_independent_action_has_no_target_requirement():
    history = {"dqn_history_records": [{
        "action_type": "scroll_down", "requires_target": 0,
        "target_identity": None, "action_identity": "scroll_down", "valid": 1,
    }]}
    vector, _ = context(history, [candidate(1)])
    assert vector[76] == 0.0
    assert not vector[80:112].any()
    assert vector[208] == 0.0


def test_outcome_flags_are_encoded():
    history = {"dqn_history_records": [{
        "action_type": "scroll_down", "action_identity": "scroll_down", "valid": 1,
        "meaningful_action_effect": 1, "productive_transition": 0,
        "confirmation_progress": 1, "unproductive_repeat": 1,
    }]}
    vector, metadata = context(history)
    assert vector[216:220].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert metadata["outcome_flags"][0] == [1, 0, 1, 1]


def test_streak_distinct_and_cycle_features():
    history = {"dqn_history_records": [
        {"action_type": "scroll_down", "action_identity": "scroll_down", "valid": 1},
        {"action_type": "scroll_up", "action_identity": "scroll_up", "valid": 1},
        {"action_type": "scroll_down", "action_identity": "scroll_down", "valid": 1},
        {"action_type": "scroll_up", "action_identity": "scroll_up", "valid": 1},
    ]}
    vector, metadata = context(history)
    assert metadata["distinct_action_identity_count"] == 2
    assert metadata["active_cycle_length"] == 2
    assert vector[233] == 0.5
    assert vector[234 + 2] == 1.0


def test_append_history_keeps_only_four_completed_transitions():
    history = {}
    for _ in range(5):
        append_history_transition(
            history, action_type="scroll_down", candidate=None,
            meaningful_action_effect=True, productive_transition=False,
            confirmation_progress=False, unproductive_repeat=False,
        )
    assert len(history["dqn_history_records"]) == 4


def test_history_reset_is_empty_between_episodes():
    history = {}
    append_history_transition(history, action_type="scroll_down", candidate=None,
                              meaningful_action_effect=True, productive_transition=False,
                              confirmation_progress=False, unproductive_repeat=False)
    fresh = {}
    vector, metadata = context(fresh)
    assert not vector[:234].any()
    assert metadata["history_valid_mask"] == [0, 0, 0, 0]


def test_current_action_is_absent_until_next_state():
    history = {}
    before, _ = context(history)
    assert not before[:234].any()
    append_history_transition(history, action_type="scroll_down", candidate=None,
                              meaningful_action_effect=True, productive_transition=False,
                              confirmation_progress=False, unproductive_repeat=False)
    after, _ = context(history)
    assert after[0] == 0.0
    assert after[2] == 1.0


def test_action_identity_uses_stable_target_only_when_required():
    assert action_identity("scroll_down", candidate(1)) == "scroll_down"
    assert action_identity("click_element", candidate(1)) == "click_element:bid:1"


def test_history_context_and_network_shapes():
    base = np.zeros(635, dtype=np.float32)
    confirmation = np.zeros(72, dtype=np.float32)
    history, _ = context()
    observation = np.concatenate([base, confirmation, history])
    assert observation.shape == (945,)
    network = FactorizedQNetwork(945, 19, 32)
    output = network(torch.from_numpy(observation).float().unsqueeze(0))
    assert output["type_values"].shape == (1, 19)
    assert output["candidate_advantages"].shape == (1, 19, 32)


def test_replay_stores_945_current_and_next_vectors():
    buffer = ReplayBuffer(2)
    state = np.zeros(945, dtype=np.float32)
    buffer.add(Transition(state, 0, 0.0, state.copy(), False, np.ones(608), np.ones(608)))
    assert buffer.items[0].state.shape == (945,)
    assert buffer.items[0].next_state.shape == (945,)


def test_ground_truth_fields_do_not_change_history():
    clean = {"dqn_history_records": [{"action_type": "scroll_down", "action_identity": "scroll_down", "valid": 1}]}
    dirty = {"dqn_history_records": [{"action_type": "scroll_down", "action_identity": "scroll_down", "valid": 1, "data_bug_id": "gt"}]}
    clean_vector, _ = context(clean)
    dirty_vector, _ = context(dirty)
    np.testing.assert_array_equal(clean_vector, dirty_vector)
