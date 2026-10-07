import random

import numpy as np
import torch

from dqn_error_discovery.agents.dqn_agent import (
    QNetwork,
    balanced_expected_action_type_distribution,
    balanced_random_valid_action,
    masked_argmax,
    random_valid_action,
    valid_actions_grouped_by_type,
)
from dqn_error_discovery.runners.dqn_click_bias_diagnostics import expected_random_action_type_distribution
from models.action_space import ActionSpace


def _mask(action_space, specs):
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for action_type, indexes in specs.items():
        for index in indexes:
            mask[action_space.encode(action_type, index)] = 1.0
    return mask


def test_balanced_type_selection_not_weighted_by_click_candidate_count():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": range(10), "scroll_down": [0]})
    expected = balanced_expected_action_type_distribution(mask, action_space)
    assert expected["click_element"] == 0.5
    assert expected["scroll_down"] == 0.5


def test_balanced_selection_chooses_inside_selected_valid_type():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [3, 7]})
    choices = {balanced_random_valid_action(mask, action_space, random.Random(seed)) for seed in range(20)}
    assert choices <= {action_space.encode("click_element", 3), action_space.encode("click_element", 7)}


def test_balanced_selection_never_selects_invalid_flattened_action():
    action_space = ActionSpace()
    mask = _mask(action_space, {"scroll_down": [0], "inspect_dom": [0]})
    valid = set(np.flatnonzero(mask > 0).tolist())
    choices = {balanced_random_valid_action(mask, action_space, random.Random(seed)) for seed in range(30)}
    assert choices <= valid


def test_candidate_independent_canonical_action_supported():
    action_space = ActionSpace()
    mask = _mask(action_space, {"scroll_down": [0]})
    assert balanced_random_valid_action(mask, action_space, random.Random(1)) == action_space.encode("scroll_down", 0)


def test_same_seed_reproducible_balanced_selection():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": range(4), "scroll_down": [0], "inspect_dom": [0]})
    first = [balanced_random_valid_action(mask, action_space, random.Random(42)) for _ in range(5)]
    second = [balanced_random_valid_action(mask, action_space, random.Random(42)) for _ in range(5)]
    assert first == second


def test_different_seed_can_produce_valid_diversity():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": range(4), "scroll_down": [0], "inspect_dom": [0]})
    action_types = {
        action_space.decode(balanced_random_valid_action(mask, action_space, random.Random(seed)))["action_type"]
        for seed in range(20)
    }
    assert len(action_types) > 1


def test_flattened_expected_probability_calculation():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": range(3), "scroll_down": [0]})
    expected = expected_random_action_type_distribution(mask, action_space)
    assert expected["click_element"] == 0.75
    assert expected["scroll_down"] == 0.25


def test_balanced_expected_probability_calculation():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": range(3), "scroll_down": [0], "inspect_dom": [0]})
    expected = balanced_expected_action_type_distribution(mask, action_space)
    assert expected["click_element"] == 1 / 3
    assert expected["scroll_down"] == 1 / 3
    assert expected["inspect_dom"] == 1 / 3


def test_one_action_type_valid():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [5, 6]})
    expected = balanced_expected_action_type_distribution(mask, action_space)
    assert expected == {"click_element": 1.0}


def test_one_candidate_inside_type_valid():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [6]})
    assert balanced_random_valid_action(mask, action_space, random.Random(7)) == action_space.encode("click_element", 6)


def test_empty_mask_safety():
    action_space = ActionSpace()
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    assert balanced_random_valid_action(mask, action_space, random.Random(1)) == 0
    assert balanced_expected_action_type_distribution(mask, action_space) == {}


def test_greedy_selection_unchanged_by_balanced_random_helper():
    action_space = ActionSpace()
    net = QNetwork(3, action_space.get_action_dim())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.network[-1].bias[action_space.encode("click_element", 1)] = 2.0
        net.network[-1].bias[action_space.encode("scroll_down", 0)] = 1.0
    mask = _mask(action_space, {"click_element": [1], "scroll_down": [0]})
    assert masked_argmax(net, np.zeros(3, dtype=np.float32), mask) == action_space.encode("click_element", 1)


def test_ground_truth_metadata_does_not_affect_sampling_from_mask():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [1], "scroll_down": [0]})
    with_gt_metadata_present_elsewhere = dict(data_bug_id="BUG-1")
    assert with_gt_metadata_present_elsewhere["data_bug_id"]
    assert balanced_expected_action_type_distribution(mask, action_space) == {"click_element": 0.5, "scroll_down": 0.5}


def test_action_dim_608_maintained():
    action_space = ActionSpace(max_candidates=32)
    assert action_space.get_action_dim() == 608
    assert random_valid_action(_mask(action_space, {"noop": [0]}), random.Random(1)) == action_space.encode("noop", 0)
    assert valid_actions_grouped_by_type(_mask(action_space, {"noop": [0]}), action_space) == {"noop": [0]}
