import math

import numpy as np
import pytest
import torch

from dqn_error_discovery.agents.dqn_agent import (
    FLAT_MAX_SELECTION,
    QNetwork,
    TYPE_LOGMEANEXP_SELECTION,
    balanced_random_valid_action,
    double_dqn_next_actions,
    masked_argmax,
    normalized_logmeanexp,
    q_value_diagnostics,
    select_greedy_action,
    type_aware_argmax_batch,
    type_aware_argmax_from_q_values,
    valid_actions_grouped_by_type,
    validate_selection_mode_consistency,
)
from models.action_space import ActionSpace


def _mask(action_space, specs):
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for action_type, indexes in specs.items():
        for index in indexes:
            mask[action_space.encode(action_type, index)] = 1.0
    return mask


def _q(action_space, values):
    q_values = torch.full((action_space.get_action_dim(),), -5.0)
    for action_type, index, value in values:
        q_values[action_space.encode(action_type, index)] = float(value)
    return q_values


class StaticQNetwork(torch.nn.Module):
    def __init__(self, q_values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("q_values", q_values.float())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.q_values.unsqueeze(0).repeat(observations.shape[0], 1)


def test_action_type_grouping():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [2, 7], "inspect_dom": [0]})
    assert valid_actions_grouped_by_type(mask, action_space) == {
        "click_element": [action_space.encode("click_element", 2), action_space.encode("click_element", 7)],
        "inspect_dom": [action_space.encode("inspect_dom", 0)],
    }


def test_normalized_logmeanexp_exact_calculation():
    values = torch.tensor([0.1, 0.3, -0.2])
    expected = torch.logsumexp(values, dim=0) - math.log(3)
    assert float(normalized_logmeanexp(values)) == pytest.approx(float(expected))


def test_candidate_duplication_invariance():
    single = normalized_logmeanexp(torch.tensor([0.2]))
    duplicated = normalized_logmeanexp(torch.tensor([0.2] * 16))
    assert float(single) == pytest.approx(float(duplicated))


def test_single_candidate_score_equals_q():
    assert float(normalized_logmeanexp(torch.tensor([0.42]))) == pytest.approx(0.42)


def test_invalid_action_exclusion():
    action_space = ActionSpace()
    mask = _mask(action_space, {"inspect_dom": [0]})
    q_values = _q(action_space, [("click_element", 5, 100.0), ("inspect_dom", 0, 0.2)])
    assert type_aware_argmax_from_q_values(q_values, torch.from_numpy(mask), action_space) == action_space.encode("inspect_dom", 0)


def test_empty_type_exclusion():
    action_space = ActionSpace()
    mask = _mask(action_space, {"scroll_down": [0]})
    q_values = _q(action_space, [("click_element", 0, 9.0), ("scroll_down", 0, 0.1)])
    assert type_aware_argmax_from_q_values(q_values, torch.from_numpy(mask), action_space) == action_space.encode("scroll_down", 0)


def test_best_type_selection_uses_normalized_logmeanexp():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [0, 1, 2, 3], "inspect_network": [0]})
    q_values = _q(
        action_space,
        [
            ("click_element", 0, -0.2),
            ("click_element", 1, -0.2),
            ("click_element", 2, -0.2),
            ("click_element", 3, 0.5),
            ("inspect_network", 0, 0.3),
        ],
    )
    click_score = normalized_logmeanexp(torch.tensor([-0.2, -0.2, -0.2, 0.5]))
    assert float(click_score) < 0.3
    assert type_aware_argmax_from_q_values(q_values, torch.from_numpy(mask), action_space) == action_space.encode("inspect_network", 0)


def test_best_candidate_within_selected_type():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [0, 2], "inspect_dom": [0]})
    q_values = _q(action_space, [("click_element", 0, 0.4), ("click_element", 2, 0.8), ("inspect_dom", 0, 0.1)])
    assert type_aware_argmax_from_q_values(q_values, torch.from_numpy(mask), action_space) == action_space.encode("click_element", 2)


def test_flat_mode_unchanged():
    action_space = ActionSpace()
    q_values = _q(action_space, [("click_element", 1, 0.9), ("inspect_network", 0, 0.5)])
    network = StaticQNetwork(q_values)
    state = np.zeros(3, dtype=np.float32)
    mask = _mask(action_space, {"click_element": [1], "inspect_network": [0]})
    assert select_greedy_action(network, state, mask, action_space, FLAT_MAX_SELECTION) == masked_argmax(network, state, mask)


def test_type_aware_behavior_greedy():
    action_space = ActionSpace()
    q_values = _q(
        action_space,
        [("click_element", 0, -1.0), ("click_element", 1, 0.9), ("inspect_console", 0, 0.4)],
    )
    network = StaticQNetwork(q_values)
    mask = _mask(action_space, {"click_element": [0, 1], "inspect_console": [0]})
    assert select_greedy_action(network, np.zeros(3, dtype=np.float32), mask, action_space, TYPE_LOGMEANEXP_SELECTION) == action_space.encode("inspect_console", 0)


def test_type_aware_double_dqn_online_next_selection():
    action_space = ActionSpace()
    q_values = _q(
        action_space,
        [("click_element", 0, 0.9), ("click_element", 1, -1.0), ("click_element", 2, -1.0), ("inspect_dom", 0, 0.5)],
    )
    mask = _mask(action_space, {"click_element": [0, 1, 2], "inspect_dom": [0]})
    selected = type_aware_argmax_batch(q_values.unsqueeze(0), torch.from_numpy(mask).bool().unsqueeze(0), action_space)
    assert int(selected.item()) == action_space.encode("inspect_dom", 0)


def test_target_network_evaluates_selected_flattened_action():
    action_space = ActionSpace()
    online_q = _q(action_space, [("inspect_network", 0, 0.8), ("scroll_down", 0, 0.1)])
    target_q = _q(action_space, [("inspect_network", 0, 1.7), ("scroll_down", 0, 9.0)])
    online = StaticQNetwork(online_q)
    next_states = torch.zeros((1, 3))
    next_masks = torch.from_numpy(_mask(action_space, {"inspect_network": [0], "scroll_down": [0]})).bool().unsqueeze(0)
    next_actions = double_dqn_next_actions(online, next_states, next_masks, action_space, TYPE_LOGMEANEXP_SELECTION)
    target_value = target_q.gather(0, next_actions.squeeze(1))
    assert int(next_actions.item()) == action_space.encode("inspect_network", 0)
    assert float(target_value.item()) == pytest.approx(1.7)


def test_invalid_high_q_next_action_ignored():
    action_space = ActionSpace()
    q_values = _q(action_space, [("click_element", 3, 100.0), ("fill_input", 0, 0.2)])
    online = StaticQNetwork(q_values)
    mask = _mask(action_space, {"fill_input": [0]})
    next_actions = double_dqn_next_actions(
        online,
        torch.zeros((1, 3)),
        torch.from_numpy(mask).bool().unsqueeze(0),
        action_space,
        TYPE_LOGMEANEXP_SELECTION,
    )
    assert int(next_actions.item()) == action_space.encode("fill_input", 0)


def test_behavior_and_target_selection_mode_consistency():
    validate_selection_mode_consistency(TYPE_LOGMEANEXP_SELECTION, TYPE_LOGMEANEXP_SELECTION)
    with pytest.raises(ValueError):
        validate_selection_mode_consistency(TYPE_LOGMEANEXP_SELECTION, FLAT_MAX_SELECTION)


def test_balanced_epsilon_random_unchanged():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [2, 3], "scroll_down": [0]})
    action = balanced_random_valid_action(mask, action_space)
    assert action in set(np.flatnonzero(mask > 0).tolist())


def test_checkpoint_state_dict_compatibility(tmp_path):
    action_space = ActionSpace(max_candidates=32)
    network = QNetwork(635, action_space.get_action_dim())
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": network.state_dict()}, path)
    loaded = QNetwork(635, action_space.get_action_dim())
    loaded.load_state_dict(torch.load(path, map_location="cpu")["model"])
    assert loaded.network[-1].out_features == 608


def test_action_dim_608():
    assert ActionSpace(max_candidates=32).get_action_dim() == 608


def test_ground_truth_independence():
    action_space = ActionSpace()
    mask = _mask(action_space, {"inspect_dom": [0]})
    q_values = _q(action_space, [("inspect_dom", 0, 0.2)])
    metadata = {"data_bug_id": "BUG-1", "known_bug": True}
    assert metadata["known_bug"]
    assert type_aware_argmax_from_q_values(q_values, torch.from_numpy(mask), action_space) == action_space.encode("inspect_dom", 0)


def test_q_diagnostics_include_type_scores():
    action_space = ActionSpace()
    q_values = _q(action_space, [("click_element", 0, 0.2), ("click_element", 1, 0.2), ("inspect_dom", 0, 0.1)])
    network = StaticQNetwork(q_values)
    mask = _mask(action_space, {"click_element": [0, 1], "inspect_dom": [0]})
    action_id = action_space.encode("click_element", 0)
    diagnostics = q_value_diagnostics(network, np.zeros(3, dtype=np.float32), mask, action_space, action_id)
    assert diagnostics["click_type_score"] == pytest.approx(0.2)
    assert diagnostics["selected_type_score"] == pytest.approx(0.2)
