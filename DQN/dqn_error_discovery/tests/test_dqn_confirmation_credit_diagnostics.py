import copy

import numpy as np
import torch

from dqn_error_discovery.agents.confirmation_credit_diagnostics import (
    aggregate_context_rows,
    classify_forced_probe,
    evidence_reproduced,
    feature_group_ablation,
    flip_direction,
    input_weight_norms,
    resolve_expected_action,
    reward_credit_summary,
)
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from models.action_space import ActionSpace


def meta(**overrides):
    value = {
        "pending_confirmation_exists": True,
        "active_rule_id": "R05",
        "active_pending_key": "logical-A",
        "expected_action_type": "click_element",
        "expected_target_identity": "bid:7",
        "expected_action_requires_target": 1,
        "expected_target_visible": 1,
        "matching_candidate_slots": [2],
        "confirmation_progress": 0.0,
    }
    value.update(overrides)
    return value


def test_forced_probe_uses_only_tracker_expected_action_and_target():
    space = ActionSpace(4)
    mask = np.zeros(space.get_action_dim(), dtype=np.float32)
    mask[space.encode("click_element", 2)] = 1
    result = resolve_expected_action(meta(), space, mask)
    assert result["expected_action_type"] == "click_element"
    assert result["expected_target_slot"] == 2
    assert result["expected_action_id"] == space.encode("click_element", 2)


def test_ground_truth_is_not_used_by_resolver():
    space = ActionSpace(4)
    mask = np.ones(space.get_action_dim(), dtype=np.float32)
    value = meta(catalog_bug_id_matches=["forbidden"], ground_truth=True)
    assert resolve_expected_action(value, space, mask)["expected_target_identity"] == "bid:7"


def test_expected_action_validity_and_target_resolution():
    space = ActionSpace(4)
    mask = np.zeros(space.get_action_dim(), dtype=np.float32)
    mask[space.encode("click_element", 1)] = 1
    result = resolve_expected_action(meta(matching_candidate_slots=[2]), space, mask)
    assert result["expected_action_defined"]
    assert not result["expected_action_mask_valid"]
    assert not result["forced_action_executable"]


def test_candidate_independent_action_resolves_slot_zero():
    space = ActionSpace(4)
    mask = np.zeros(space.get_action_dim(), dtype=np.float32)
    mask[space.encode("scroll_down", 0)] = 1
    result = resolve_expected_action(meta(expected_action_type="scroll_down", expected_action_requires_target=0, expected_target_visible=0, matching_candidate_slots=[]), space, mask)
    assert result["expected_target_slot"] == 0
    assert result["forced_action_executable"]


def test_target_required_non_element_action_is_mask_valid_but_not_executable():
    space = ActionSpace(4)
    mask = np.ones(space.get_action_dim(), dtype=np.float32)
    result = resolve_expected_action(meta(expected_action_type="noop", matching_candidate_slots=[0]), space, mask)
    assert result["expected_action_mask_valid"]
    assert not result["target_semantics_valid"]
    assert not result["forced_action_executable"]


def test_probe_classification_and_progress_negative_reward():
    assert classify_forced_probe(executable=True, evidence=False, progress=False, confirmed=False) == "EXECUTED_NO_EVIDENCE"
    assert classify_forced_probe(executable=True, evidence=True, progress=False, confirmed=False) == "EVIDENCE_REPRODUCED_NO_PROGRESS"
    summary = reward_credit_summary([{"total_reward": -0.3, "confirmation_progress_after": True, "confirmed_after": False}])
    assert summary["progress_with_negative_reward_count"] == 1
    assert summary["negative_count"] == 1


def test_evidence_reproduction_uses_runtime_evidence_only():
    before = [{"rule_id": "R05", "target_id": "", "evidence": {"target_bid": "7"}}]
    after = [{"rule_id": "R05", "target_id": "", "evidence": {"target_bid": "7"}}]
    assert evidence_reproduced(before, after, meta())
    assert not evidence_reproduced(before, [], meta())


def test_flip_direction_categories():
    assert flip_direction("scroll_down", "click_element", "click_element") == "FLIP_TOWARD_EXPECTED"
    assert flip_direction("click_element", "scroll_down", "click_element") == "FLIP_AWAY_FROM_EXPECTED"
    assert flip_direction("scroll_down", "scroll_up", "click_element") == "FLIP_OTHER_TO_OTHER"


def test_expected_rank_real_vs_zero_aggregation():
    result = aggregate_context_rows([
        {"expected_type_context_delta": 0.2, "expected_type_rank_real": 2, "expected_type_rank_zero": 4, "flip_direction": "FLIP_TOWARD_EXPECTED"},
        {"expected_type_context_delta": -0.1, "expected_type_rank_real": 5, "expected_type_rank_zero": 3, "flip_direction": "FLIP_AWAY_FROM_EXPECTED"},
    ])
    assert result["positive_delta_count"] == 1
    assert result["negative_delta_count"] == 1
    assert result["rank_delta_zero_minus_real_mean"] == 0.0
    assert result["flip_direction_counts"]["FLIP_TOWARD_EXPECTED"] == 1


def test_feature_group_ablation_and_expected_action_onehot():
    space = ActionSpace(4)
    network = FactorizedQNetwork(707, len(space.action_types), 4)
    state = np.ones(707, dtype=np.float32)
    mask = np.ones(space.get_action_dim(), dtype=np.float32)
    result = feature_group_ablation(network, state, mask, space, meta())
    assert set(result) == {"real", "pending_rule", "expected_action_type_onehot", "expected_target_mask_visible", "progress_attempt_pending_count"}
    assert "expected_type_value_delta_vs_real" in result["expected_action_type_onehot"]


def test_target_mask_ablation_reports_candidate_advantage():
    space = ActionSpace(4)
    network = FactorizedQNetwork(707, len(space.action_types), 4)
    result = feature_group_ablation(network, np.ones(707, dtype=np.float32), np.ones(space.get_action_dim(), dtype=np.float32), space, meta())
    assert "expected_target_advantage_delta_vs_real" in result["expected_target_mask_visible"]


def test_input_weight_norms_split_base_and_context():
    network = FactorizedQNetwork(707, 19, 4)
    norms = input_weight_norms(network)
    assert norms["base_l2"] > 0
    assert norms["confirmation_l2"] > 0
    assert "confirmation_to_base_mean_abs_ratio" in norms


def test_probe_helpers_do_not_mutate_state_or_network():
    space = ActionSpace(4)
    network = FactorizedQNetwork(707, len(space.action_types), 4)
    state = np.ones(707, dtype=np.float32)
    before_state = state.copy()
    before_weights = {key: value.detach().clone() for key, value in network.state_dict().items()}
    feature_group_ablation(network, state, np.ones(space.get_action_dim(), dtype=np.float32), space, meta())
    assert np.array_equal(state, before_state)
    assert all(torch.equal(value, network.state_dict()[key]) for key, value in before_weights.items())


def test_reward_summary_counts_zero_positive_negative_and_confirmed_nonpositive():
    result = reward_credit_summary([
        {"total_reward": 1.0, "confirmation_progress_after": False, "confirmed_after": False},
        {"total_reward": 0.0, "confirmation_progress_after": False, "confirmed_after": True},
        {"total_reward": -1.0, "confirmation_progress_after": True, "confirmed_after": False},
    ])
    assert result["positive_count"] == 1
    assert result["zero_count"] == 1
    assert result["negative_count"] == 1
    assert result["confirmed_with_nonpositive_reward_count"] == 1


def test_no_replay_or_training_objects_are_required_for_probe_helpers():
    # The helper API accepts only metadata, arrays, a network, and an action space.
    assert "replay" not in resolve_expected_action.__code__.co_varnames
    assert "optimizer" not in feature_group_ablation.__code__.co_varnames


def test_unresolved_target_is_not_marked_executable():
    space = ActionSpace(4)
    mask = np.ones(space.get_action_dim(), dtype=np.float32)
    result = resolve_expected_action(meta(expected_target_visible=0), space, mask)
    assert result["expected_action_defined"]
    assert not result["forced_action_executable"]


def test_unknown_expected_type_is_not_resolvable():
    space = ActionSpace(4)
    result = resolve_expected_action(meta(expected_action_type="unknown"), space, np.ones(space.get_action_dim(), dtype=np.float32))
    assert not result["expected_action_defined"]
    assert not result["forced_action_executable"]


def test_context_aggregation_counts_help_hurt_and_zero_separately():
    result = aggregate_context_rows([
        {"expected_type_context_delta": 1.0, "expected_type_rank_real": 1, "expected_type_rank_zero": 2, "flip_direction": "NO_FLIP"},
        {"expected_type_context_delta": -1.0, "expected_type_rank_real": 3, "expected_type_rank_zero": 2, "flip_direction": "NO_FLIP"},
        {"expected_type_context_delta": 0.0, "expected_type_rank_real": 2, "expected_type_rank_zero": 2, "flip_direction": "NO_FLIP"},
    ])
    assert result["positive_delta_count"] == 1
    assert result["negative_delta_count"] == 1
    assert result["zero_delta_count"] == 1


def test_forced_probe_funnel_rates_are_safe_at_zero_denominator():
    assert classify_forced_probe(executable=False, evidence=False, progress=False, confirmed=False) == "NOT_EXECUTABLE"
    assert reward_credit_summary([])["mean_reward"] == 0.0
