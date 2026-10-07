from __future__ import annotations

import numpy as np
import torch

from dqn_error_discovery.agents.confirmation_learning_diagnostics import (
    ConfirmationFunnelAggregator,
    classify_failure,
    classify_reproduction_selection,
    expected_target_rank_diagnostic,
    factorized_confirmation_step_diagnostic,
    is_reproduction_opportunity,
    real_vs_zero_context_diagnostic,
)
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, select_factorized_greedy_action
from models.action_space import ActionSpace


def _metadata(**overrides):
    value = {
        "pending_confirmation_exists": True,
        "active_rule_id": "R01",
        "expected_action_type": "click_element",
        "expected_action_requires_target": 1,
        "expected_target_visible": 1,
        "matching_candidate_slots": [2],
    }
    value.update(overrides)
    return value


def _mask(action_space: ActionSpace) -> np.ndarray:
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for action_type in ("noop", "click_element", "scroll_down", "inspect_dom"):
        slots = range(4) if action_type == "click_element" else range(1)
        for slot in slots:
            mask[action_space.encode(action_type, slot)] = 1.0
    return mask


def test_reproduction_opportunity_detection():
    assert is_reproduction_opportunity(_metadata())
    assert not is_reproduction_opportunity(_metadata(expected_target_visible=0))
    assert is_reproduction_opportunity(
        _metadata(expected_action_type="scroll_down", expected_action_requires_target=0, expected_target_visible=0)
    )


def test_expected_type_rank():
    action_space = ActionSpace(max_candidates=32)
    network = _rank_network(action_space, expected_type="click_element")
    state = np.zeros(707, dtype=np.float32)
    diagnostic = factorized_confirmation_step_diagnostic(
        network, state, _mask(action_space), action_space, action_space.encode("scroll_down", 0), _metadata()
    )
    assert diagnostic["expected_type_rank"] == 2
    assert diagnostic["top1_type"] == "scroll_down"


def test_expected_target_rank():
    action_space = ActionSpace(max_candidates=32)
    advantages = np.zeros((len(action_space.action_types), action_space.max_candidates), dtype=np.float32)
    type_index = action_space.action_types.index("click_element")
    advantages[type_index, 1] = 0.8
    advantages[type_index, 2] = 0.4
    diagnostic = expected_target_rank_diagnostic(
        advantages,
        _mask(action_space),
        action_space,
        _metadata(matching_candidate_slots=[2]),
        selected_action={"candidate_index": 1},
    )
    assert diagnostic["expected_target_rank"] == 2
    assert diagnostic["target_match"] is False


def test_type_match_classification():
    result = classify_reproduction_selection(_metadata(), {"action_type": "click_element", "candidate_index": 3})
    assert result["type_match"] is True
    assert result["full_match"] is False


def test_full_match_classification():
    result = classify_reproduction_selection(_metadata(), {"action_type": "click_element", "candidate_index": 2})
    assert result["type_match"] is True
    assert result["full_match"] is True


def test_candidate_independent_full_match():
    result = classify_reproduction_selection(
        _metadata(expected_action_type="scroll_down", expected_action_requires_target=0, matching_candidate_slots=[]),
        {"action_type": "scroll_down", "candidate_index": 0},
    )
    assert result["full_match"] is True


def test_funnel_aggregation():
    funnel = ConfirmationFunnelAggregator()
    funnel.record(_metadata(), {"action_type": "click_element", "candidate_index": 2}, confirmation_progress=True)
    summary = funnel.summary()
    assert summary["reproduction_opportunity_count"] == 1
    assert summary["expected_action_type_selected_count"] == 1
    assert summary["expected_action_and_target_selected_count"] == 1
    assert summary["confirmation_progress_count"] == 1


def test_failure_attribution():
    wrong_type = classify_reproduction_selection(_metadata(), {"action_type": "scroll_down", "candidate_index": 0})
    wrong_target = classify_reproduction_selection(_metadata(), {"action_type": "click_element", "candidate_index": 1})
    full = classify_reproduction_selection(_metadata(), {"action_type": "click_element", "candidate_index": 2})
    assert classify_failure(wrong_type, False, False) == "NO_TYPE_SELECTION"
    assert classify_failure(wrong_target, False, False) == "TYPE_SELECTED_WRONG_TARGET"
    assert classify_failure(full, False, False) == "FULL_MATCH_NO_PROGRESS"
    assert classify_failure(full, True, False) == "PROGRESS_NO_CONFIRMATION"
    assert classify_failure(full, True, True) == "CONFIRMED"


def test_real_vs_zero_context_forward():
    action_space = ActionSpace(max_candidates=32)
    network = _context_sensitive_network(action_space)
    state = np.zeros(707, dtype=np.float32)
    state[635] = 1.0
    diagnostic = real_vs_zero_context_diagnostic(network, state, _mask(action_space), action_space, _metadata())
    assert diagnostic["real_expected_type_value"] > diagnostic["zero_expected_type_value"]


def test_expected_type_context_delta():
    action_space = ActionSpace(max_candidates=32)
    network = _context_sensitive_network(action_space)
    state = np.zeros(707, dtype=np.float32)
    state[635] = 1.0
    diagnostic = real_vs_zero_context_diagnostic(network, state, _mask(action_space), action_space, _metadata())
    assert diagnostic["expected_type_context_delta"] > 0.0


def test_selection_flip_calculation():
    action_space = ActionSpace(max_candidates=32)
    network = _context_sensitive_network(action_space)
    state = np.zeros(707, dtype=np.float32)
    state[635] = 1.0
    diagnostic = real_vs_zero_context_diagnostic(network, state, _mask(action_space), action_space, _metadata())
    assert diagnostic["selection_flip"] is True
    assert diagnostic["real_selected_type"] == "click_element"
    assert diagnostic["zero_selected_type"] == "scroll_down"


def test_type_value_vector_difference():
    action_space = ActionSpace(max_candidates=32)
    network = _context_sensitive_network(action_space)
    state = np.zeros(707, dtype=np.float32)
    state[635] = 1.0
    diagnostic = real_vs_zero_context_diagnostic(network, state, _mask(action_space), action_space, _metadata())
    assert diagnostic["type_value_l1"] > 0.0
    assert diagnostic["type_value_l2"] > 0.0
    assert diagnostic["type_value_max_abs"] > 0.0


def test_diagnostic_does_not_change_training_or_action():
    action_space = ActionSpace(max_candidates=32)
    network = _context_sensitive_network(action_space)
    state = np.zeros(707, dtype=np.float32)
    state[635] = 1.0
    mask = _mask(action_space)
    before = select_factorized_greedy_action(network, state, mask, action_space)
    _ = real_vs_zero_context_diagnostic(network, state, mask, action_space, _metadata())
    after = select_factorized_greedy_action(network, state, mask, action_space)
    assert after == before
    assert np.array_equal(mask, _mask(action_space))
    assert state[635] == 1.0


def test_ground_truth_independence():
    metadata = _metadata(data_bug_id="BUG-1", matched_bug_id="BUG-2", catalog_bug_id_matches=["BUG-3"])
    result = classify_reproduction_selection(metadata, {"action_type": "click_element", "candidate_index": 2})
    assert result["full_match"] is True
    assert "BUG" not in str(result)


def test_expected_target_context_delta():
    action_space = ActionSpace(max_candidates=32)
    network = _context_sensitive_network(action_space, target_slot=2)
    state = np.zeros(707, dtype=np.float32)
    state[635] = 1.0
    diagnostic = real_vs_zero_context_diagnostic(network, state, _mask(action_space), action_space, _metadata())
    assert diagnostic["expected_target_context_delta"] is not None
    assert diagnostic["expected_target_context_delta"] > 0.0


def _rank_network(action_space: ActionSpace, expected_type: str) -> FactorizedQNetwork:
    network = FactorizedQNetwork(707, len(action_space.action_types), action_space.max_candidates)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.type_head.bias[action_space.action_types.index("scroll_down")] = 2.0
        network.type_head.bias[action_space.action_types.index(expected_type)] = 1.0
    return network


def _context_sensitive_network(action_space: ActionSpace, target_slot: int | None = None) -> FactorizedQNetwork:
    network = FactorizedQNetwork(707, len(action_space.action_types), action_space.max_candidates)
    click_index = action_space.action_types.index("click_element")
    scroll_index = action_space.action_types.index("scroll_down")
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.encoder[0].weight[0, 635] = 1.0
        network.encoder[2].weight[0, 0] = 1.0
        network.type_head.weight[click_index, 0] = 3.0
        network.type_head.bias[scroll_index] = 1.0
        if target_slot is not None:
            network.candidate_head.weight[click_index * action_space.max_candidates + target_slot, 0] = 2.0
    return network
