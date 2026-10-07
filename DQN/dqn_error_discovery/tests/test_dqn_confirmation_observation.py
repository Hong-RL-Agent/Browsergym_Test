import numpy as np
import pytest
import torch

from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from dqn_error_discovery.agents.confirmation_observation import (
    CONFIRMATION_CONTEXT_DIM,
    augment_dqn_observation,
    build_dqn_confirmation_context,
)
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from dqn_error_discovery.agents.dqn_agent import ReplayBuffer, Transition
from dqn_error_discovery.runners.run_policy_sanity_check import _validate_checkpoint_architecture


def candidate(bid, *, bbox=None):
    item = {"bid": bid, "role": "button", "tag": "button", "name": f"item-{bid}"}
    if bbox is not None:
        item["bbox"] = bbox
    return item


def pending(target="20", action="click_element", rule="R01", **extra):
    record = {
        "signature": "sig-1",
        "logical_key": "R01|product|20",
        "rule_id": rule,
        "action_target": action if not target else f"{action}:bid:{target}",
        "progress": 0.5,
        "attempt_count": 7,
        "created_step": 1,
    }
    record.update(extra)
    return {"pending_reproduction_records": {"sig-1": record}}


def context(history=None, candidates=None):
    return build_dqn_confirmation_context(
        history or {},
        {"candidate_elements": candidates or []},
        ActionSpace(32),
    )


def test_empty_context_is_fixed_zero_vector():
    values, meta = context()
    assert values.shape == (72,)
    assert np.all(values == 0)
    assert meta["pending_confirmation_exists"] is False


def test_rule_action_and_target_are_encoded():
    values, meta = context(pending(), [candidate(20)])
    assert values[0] == 1
    assert values[1] == 1  # R01
    assert values[16 + ActionSpace.action_types.index("click_element")] == 1
    assert values[35] == 1
    assert values[36] == 1
    assert values[68] == 1
    assert meta["matching_candidate_slots"] == [0]


def test_target_matching_ignores_slot_and_bbox_changes():
    first, _ = context(pending(), [candidate(20, bbox=[1, 2, 3, 4])])
    second, meta = context(pending(), [candidate(99), candidate(20, bbox=[9, 8, 7, 6])])
    assert first[36] == 1
    assert second[37] == 1
    assert meta["matching_candidate_slots"] == [1]


def test_invisible_target_has_no_fake_candidate_slot():
    values, meta = context(pending("20"), [candidate(21)])
    assert values[0] == 1
    assert values[36:68].sum() == 0
    assert values[68] == 0
    assert meta["expected_target_visible"] == 0


def test_candidate_independent_pending_action_has_no_target_requirement():
    values, meta = context(pending("", action="scroll_down"))
    assert values[35] == 0
    assert values[36:68].sum() == 0
    assert values[68] == 0
    assert values[16 + ActionSpace.action_types.index("scroll_down")] == 1
    assert meta["expected_action_requires_target"] == 0


def test_progress_attempt_and_pending_count_are_bounded():
    history = pending(progress=2.0, attempt_count=100)
    history["pending_reproduction_records"].update({f"sig-{i}": dict(next(iter(history["pending_reproduction_records"].values())), signature=f"sig-{i}") for i in range(2, 12)})
    values, meta = context(history, [candidate(20)])
    assert values[69] == 1.0
    assert values[70] == 1.0
    assert values[71] == 1.0
    assert meta["pending_confirmation_count"] == 11


def test_active_pending_selection_prefers_resolvable_then_progress_and_is_deterministic():
    history = {
        "pending_reproduction_records": {
            "a": {"signature": "a", "logical_key": "z", "rule_id": "R02", "action_target": "click_element:bid:999", "progress": 1, "created_step": 0},
            "b": {"signature": "b", "logical_key": "a", "rule_id": "R01", "action_target": "click_element:bid:20", "progress": 0, "created_step": 4},
        }
    }
    _, meta = context(history, [candidate(20)])
    assert meta["active_pending_key"] == "a"
    history["pending_reproduction_records"]["a"]["action_target"] = "click_element:bid:20"
    _, meta = context(history, [candidate(20)])
    assert meta["active_pending_key"] == "z"


def test_base_and_dqn_dimensions_and_factorized_heads():
    assert ObservationEncoder(32).get_obs_dim() == 635
    base = np.zeros(635, dtype=np.float32)
    values, _ = context(pending(), [candidate(20)])
    assert augment_dqn_observation(base, values).shape == (707,)
    network = FactorizedQNetwork(707, 19, 32)
    outputs = network(torch.zeros((1, 707), dtype=torch.float32))
    type_values = outputs["type_values"]
    advantages = outputs["candidate_advantages"]
    assert tuple(type_values.shape) == (1, 19)
    assert tuple(advantages.shape) == (1, 19, 32)


def test_old_factorized_checkpoint_is_rejected_for_confirmation_architecture():
    with pytest.raises(ValueError):
        _validate_checkpoint_architecture(
            {"manifest": {"dqn_architecture": "factorized_v1", "dqn_obs_dim": 635}},
            "factorized_v1_confobs",
        )


def test_ground_truth_metadata_does_not_change_context():
    clean, _ = context(pending(), [candidate(20)])
    dirty_history = pending()
    dirty_history["data_bug_id"] = "forbidden"
    dirty, _ = context(dirty_history, [dict(candidate(20), data_bug_id="forbidden")])
    np.testing.assert_array_equal(clean, dirty)


def test_action_dim_and_context_feature_offsets_are_stable():
    space = ActionSpace(32)
    assert space.get_action_dim() == 608
    assert 1 + 15 + 19 + 1 + 32 + 1 + 1 + 1 + 1 == CONFIRMATION_CONTEXT_DIM


def test_pending_count_without_resolvable_active_target_still_surfaces_pending():
    values, meta = context({"pending_reproduction_action_targets": {"click_element:bid:999"}}, [candidate(20)])
    assert values[0] == 1
    assert values[71] > 0
    assert meta["expected_target_visible"] == 0


def test_current_context_does_not_read_post_action_fields():
    before = pending()
    after = dict(before)
    after["confirmation_progress_after"] = 1.0
    first, _ = context(before, [candidate(20)])
    second, _ = context(after, [candidate(20)])
    np.testing.assert_array_equal(first, second)


def test_next_context_can_reflect_tracker_update():
    history = pending()
    before, _ = context(history, [candidate(20)])
    history["pending_reproduction_records"]["sig-1"]["attempt_count"] = 1
    after, _ = context(history, [candidate(20)])
    assert before[70] == 1.0
    assert after[70] == 0.25


def test_replay_stores_707_current_and_next_vectors():
    buffer = ReplayBuffer(4)
    vector = np.zeros(707, dtype=np.float32)
    mask = np.ones(608, dtype=np.float32)
    buffer.add(Transition(vector, 0, 0.0, vector.copy(), False, mask, mask))
    item = buffer.sample(1)[0]
    assert item.state.shape == (707,)
    assert item.next_state.shape == (707,)


def test_context_values_are_finite_and_bounded():
    values, _ = context(pending(progress=-1, attempt_count=-2), [candidate(20)])
    assert np.isfinite(values).all()
    assert values.min() >= 0
    assert values.max() <= 1


def test_target_identity_is_not_candidate_index_based():
    values, _ = context(pending("20"), [candidate(99), candidate(20)])
    assert values[36] == 0
    assert values[37] == 1


def test_ppo_base_dimension_is_unchanged_by_dqn_augmentation():
    encoder = ObservationEncoder(32)
    raw = {"page_state": {}, "candidate_elements": [], "runtime_signals": {}, "layout_signals": {}, "infra_signals": {}, "history": {}}
    assert encoder.encode_observation(raw).shape == (635,)


def test_pending_rule_is_zero_for_unknown_rule_id():
    values, _ = context(pending(rule="R99"), [candidate(20)])
    assert values[1:16].sum() == 0


def test_expected_action_onehot_is_zero_when_no_pending():
    values, _ = context()
    assert values[16:35].sum() == 0


def test_multiple_matching_slots_are_all_marked():
    values, meta = context(pending("20"), [candidate(20), candidate(20)])
    assert values[36] == 1 and values[37] == 1
    assert meta["matching_candidate_slots"] == [0, 1]


def test_progress_is_clamped_to_unit_interval():
    values, _ = context(pending(progress=-10), [candidate(20)])
    assert values[69] == 0


def test_attempt_count_is_normalized_by_declared_cap():
    values, _ = context(pending(attempt_count=2), [candidate(20)])
    assert values[70] == 0.5


def test_context_dtype_is_float32():
    values, _ = context(pending(), [candidate(20)])
    assert values.dtype == np.float32
