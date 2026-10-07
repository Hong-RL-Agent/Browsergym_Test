import copy

import numpy as np
import pytest
import torch

from dqn_error_discovery.agents.confirmation_observation import CONFIRMATION_CONTEXT_DIM
from dqn_error_discovery.agents.dqn_agent import (
    FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    FactorizedQNetwork,
    ReplayBuffer,
    Transition,
)
from dqn_error_discovery.agents.history_observation import (
    HISTORY_CONTEXT_V2_DIM,
    PROSPECTIVE_REPEAT_RISK_FLAGS,
    action_identity,
    append_history_transition,
    augment_dqn_history_observation_v2,
    build_dqn_history_context_v2,
    prospective_repeat_flags,
)
from dqn_error_discovery.runners.run_policy_sanity_check import _validate_checkpoint_architecture
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder


ACTION_TYPES = list(ActionSpace.action_types)


def candidate(bid, *, bbox=None):
    return {"bid": str(bid), "role": "button", "visible": True, "bbox": bbox or [1, 2, 3, 4]}


def record(action_type, *, target=None, **flags):
    requires_target = action_type in {"click_element", "fill_input", "press_enter"}
    return {
        "action_type": action_type,
        "requires_target": int(requires_target),
        "target_identity": f"bid:{target}" if target is not None else None,
        "action_identity": f"{action_type}:bid:{target}" if target is not None else action_type,
        "valid": 1,
        **flags,
    }


def build(history=None, candidates=None):
    return build_dqn_history_context_v2(history or {}, {"candidate_elements": candidates or []}, ACTION_TYPES, 32)


def risk_index(action_type, flag):
    return 212 + ACTION_TYPES.index(action_type) * len(PROSPECTIVE_REPEAT_RISK_FLAGS) + PROSPECTIVE_REPEAT_RISK_FLAGS.index(flag)


def test_v2_history_dim_is_269():
    vector, metadata = build()

    assert vector.shape == (HISTORY_CONTEXT_V2_DIM,)
    assert HISTORY_CONTEXT_V2_DIM == 269
    assert metadata["version"] == "v2"


def test_final_dqn_obs_is_976_and_ppo_obs_is_635():
    base = np.zeros(635, dtype=np.float32)
    confirmation = np.zeros(CONFIRMATION_CONTEXT_DIM, dtype=np.float32)
    history, _ = build()

    observation = augment_dqn_history_observation_v2(base, confirmation, history)

    assert observation.shape == (976,)
    assert ObservationEncoder(32).get_obs_dim() == 635


def test_policy_vector_excludes_outcome_flags_history_mask_streak_distinct_and_cycle_onehot():
    base_history = {"dqn_history_records": [record("scroll_down")]}
    outcome_history = {
        "dqn_history_records": [
            record(
                "scroll_down",
                meaningful_action_effect=1,
                productive_transition=1,
                confirmation_progress=1,
                unproductive_repeat=1,
            )
        ]
    }

    base, base_meta = build(base_history)
    outcome, outcome_meta = build(outcome_history)

    np.testing.assert_array_equal(base, outcome)
    assert outcome_meta["outcome_flags"][0] == [1, 1, 1, 1]
    assert base.shape == (269,)
    assert base[212:].shape == (57,)


def test_history_valid_mask_is_metadata_only_not_policy_slice():
    vector, metadata = build({"dqn_history_records": [record("noop"), record("scroll_down")]})

    assert metadata["history_valid_mask"] == [1, 1, 0, 0]
    assert vector[212] == 0.0  # noop immediate risk, not slot0 validity
    assert vector[212:216].tolist() != [1.0, 1.0, 0.0, 0.0]


def test_streak_distinct_and_cycle_are_metadata_only():
    history = {
        "dqn_history_records": [
            record("scroll_up"),
            record("scroll_down"),
            record("scroll_up"),
            record("scroll_down"),
        ],
        "dqn_action_target_history": ["scroll_up:0", "scroll_down:0", "scroll_up:0", "scroll_down:0"],
    }

    vector, metadata = build(history)

    assert metadata["distinct_action_identity_count"] == 2
    assert metadata["active_cycle_length"] == 2
    assert vector[232] != 0.5  # old v1 streak slot is now risk matrix space
    assert vector[234:238].tolist() != [0.0, 0.0, 1.0, 0.0]


def test_prospective_risk_ordering_is_action_type_major():
    history = {"dqn_action_target_history": ["scroll_up:0"]}
    vector, metadata = build({**history, "dqn_history_records": [record("scroll_up")]})
    scroll_up = ACTION_TYPES.index("scroll_up")

    assert metadata["prospective_repeat_risk_ordering"] == "action_type-major: immediate,recent,cycle"
    assert vector[212 + scroll_up * 3 + 0] == 1.0
    assert vector[212 + scroll_up * 3 + 1] == 1.0
    assert vector[212 + scroll_up * 3 + 2] == 1.0


def test_zero_history_has_zero_prospective_risk_matrix():
    vector, metadata = build()

    assert not vector[212:].any()
    assert all(not any(flags.values()) for flags in metadata["prospective_repeat_risk_by_action_type"].values())


def test_absent_target_has_hidden_target_visible_flag():
    history = {"dqn_history_records": [record("click_element", target=1)]}
    vector, metadata = build(history, [candidate(2)])

    assert metadata["target_current_slots"] == [[], [], [], []]
    assert vector[208] == 0.0
    assert not vector[80:112].any()


def test_padding_is_distinguishable_from_noop_without_valid_mask():
    empty, _ = build()
    noop, _ = build({"dqn_history_records": [record("noop")]})

    assert not empty[:76].any()
    assert noop[0] == 1.0
    assert noop[:76].sum() == 1.0


def test_recent_action_ordering_is_newest_first():
    vector, metadata = build({"dqn_history_records": [record("scroll_down"), record("scroll_up")]})

    assert metadata["recent_action_identities"][:2] == ["scroll_up", "scroll_down"]
    assert vector[ACTION_TYPES.index("scroll_up")] == 1.0
    assert vector[19 + ACTION_TYPES.index("scroll_down")] == 1.0


def test_stable_target_remap_ignores_bbox_and_slot_order():
    history = {"dqn_history_records": [record("click_element", target=1)]}
    vector, metadata = build(history, [candidate(9), candidate(1, bbox=[99, 88, 77, 66])])

    assert metadata["target_current_slots"] == [[1], [], [], []]
    assert vector[80 + 1] == 1.0
    assert vector[80 + 0] == 0.0


def test_target_dependent_type_prospective_risk_is_zero():
    history = {"dqn_action_target_history": ["click_element:bid:1"]}

    flags = prospective_repeat_flags(history, "click_element", ACTION_TYPES)

    assert flags == {"immediate": False, "recent": False, "cycle": False}


def test_scroll_immediate_risk():
    history = {"dqn_action_target_history": ["scroll_up:0"]}

    assert prospective_repeat_flags(history, "scroll_up", ACTION_TYPES)["immediate"] is True
    assert prospective_repeat_flags(history, "scroll_down", ACTION_TYPES)["immediate"] is False


def test_scroll_recent_risk_is_encoded():
    history = {"dqn_action_target_history": ["scroll_up:0", "noop:0", "scroll_down:0"]}
    vector, metadata = build({**history, "dqn_history_records": [record("scroll_up"), record("noop"), record("scroll_down")]})

    assert vector[risk_index("scroll_up", "recent")] == 1.0
    assert metadata["prospective_repeat_risk_by_action_type"]["scroll_up"]["recent"] is True


def test_cycle2_prospective_risk_for_scroll_up():
    history = {"dqn_action_target_history": ["scroll_up:0", "scroll_down:0", "scroll_up:0", "scroll_down:0"]}

    assert prospective_repeat_flags(history, "scroll_up", ACTION_TYPES)["cycle"] is True
    assert prospective_repeat_flags(history, "noop", ACTION_TYPES)["cycle"] is False


def test_cycle3_prospective_risk():
    history = {
        "dqn_action_target_history": [
            "scroll_up:0",
            "scroll_down:0",
            "noop:0",
            "scroll_up:0",
            "scroll_down:0",
        ]
    }

    assert prospective_repeat_flags(history, "noop", ACTION_TYPES)["cycle"] is True


def test_prospective_helper_is_side_effect_free():
    history = {"dqn_action_target_history": ["scroll_up:0"]}
    before = copy.deepcopy(history)

    prospective_repeat_flags(history, "scroll_up", ACTION_TYPES)

    assert history == before


def test_state_t_contains_no_action_t_outcome():
    history = {}
    before, _ = build(history)
    append_history_transition(
        history,
        action_type="scroll_down",
        candidate=None,
        meaningful_action_effect=False,
        productive_transition=False,
        confirmation_progress=False,
        unproductive_repeat=True,
    )
    after, _ = build(history)

    assert before[risk_index("scroll_down", "immediate")] == 0.0
    assert after[risk_index("scroll_down", "immediate")] == 1.0


def test_replay_state_and_next_are_976():
    buffer = ReplayBuffer(1)
    state = np.zeros(976, dtype=np.float32)

    buffer.add(Transition(state, 0, 0.0, state.copy(), False, np.ones(608), np.ones(608)))

    assert buffer.items[0].state.shape == (976,)
    assert buffer.items[0].next_state.shape == (976,)


def test_new_network_accepts_976():
    network = FactorizedQNetwork(976, 19, 32)
    outputs = network(torch.zeros((2, 976)))

    assert outputs["type_values"].shape == (2, 19)
    assert outputs["candidate_advantages"].shape == (2, 19, 32)


def test_old_945_checkpoint_is_rejected_for_histv2():
    payload = {
        "manifest": {
            "dqn_architecture": "factorized_v1_confobs_hist",
            "dqn_obs_dim": 945,
            "confirmation_context_dim": 72,
            "history_context_dim": 238,
        }
    }

    with pytest.raises(ValueError, match="refusing to load checkpoint"):
        _validate_checkpoint_architecture(payload, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE)


def test_histv2_checkpoint_requires_976_manifest():
    payload = {
        "manifest": {
            "dqn_architecture": FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
            "dqn_obs_dim": 945,
            "confirmation_context_dim": 72,
            "history_context_dim": 238,
            "history_observation_version": "v1",
        }
    }

    with pytest.raises(ValueError, match="dqn_obs_dim=976"):
        _validate_checkpoint_architecture(payload, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE)


def test_ground_truth_independent():
    clean = {"dqn_history_records": [record("scroll_down")]}
    dirty = {"dqn_history_records": [record("scroll_down", data_bug_id="gt", ground_truth=True)]}

    clean_vector, _ = build(clean)
    dirty_vector, _ = build(dirty)

    np.testing.assert_array_equal(clean_vector, dirty_vector)


def test_action_identity_still_uses_stable_target_only_when_required():
    assert action_identity("scroll_down", candidate(1)) == "scroll_down"
    assert action_identity("click_element", candidate(1)) == "click_element:bid:1"
