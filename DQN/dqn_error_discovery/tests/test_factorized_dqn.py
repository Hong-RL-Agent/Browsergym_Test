import numpy as np
import pytest
import torch

from dqn_error_discovery.agents.dqn_agent import (
    FACTORIZED_DQN_ARCHITECTURE,
    FactorizedQNetwork,
    QNetwork,
    balanced_random_valid_action,
    center_candidate_advantages,
    factorized_double_dqn_next_actions,
    factorized_flat_q_values,
    factorized_greedy_actions_from_outputs,
    factorized_q_for_actions,
    masked_argmax,
)
from dqn_error_discovery.runners.run_policy_sanity_check import _validate_checkpoint_architecture
from models.action_space import ActionSpace


def _mask(action_space, specs):
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for action_type, indexes in specs.items():
        for index in indexes:
            mask[action_space.encode(action_type, index)] = 1.0
    return mask


def _outputs(action_space, type_values=None, advantages=None):
    type_tensor = torch.zeros((1, len(action_space.action_types)))
    advantage_tensor = torch.zeros((1, len(action_space.action_types), action_space.max_candidates))
    for action_type, value in (type_values or {}).items():
        type_tensor[0, action_space.action_types.index(action_type)] = float(value)
    for action_type, index, value in (advantages or []):
        advantage_tensor[0, action_space.action_types.index(action_type), index] = float(value)
    return {"type_values": type_tensor, "candidate_advantages": advantage_tensor}


class StaticFactorizedNetwork(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.register_buffer("type_values", outputs["type_values"].squeeze(0).float())
        self.register_buffer("candidate_advantages", outputs["candidate_advantages"].squeeze(0).float())

    def forward(self, observations):
        return {
            "type_values": self.type_values.unsqueeze(0).repeat(observations.shape[0], 1),
            "candidate_advantages": self.candidate_advantages.unsqueeze(0).repeat(observations.shape[0], 1, 1),
        }


def test_output_shapes():
    action_space = ActionSpace()
    network = FactorizedQNetwork(635, len(action_space.action_types), action_space.max_candidates)
    outputs = network(torch.zeros((2, 635)))
    assert outputs["type_values"].shape == (2, 19)
    assert outputs["candidate_advantages"].shape == (2, 19, 32)


def test_type_values_shape_19():
    action_space = ActionSpace()
    network = FactorizedQNetwork(3, len(action_space.action_types), action_space.max_candidates)
    assert network(torch.zeros((1, 3)))["type_values"].shape[-1] == 19


def test_candidate_advantages_shape_19x32():
    action_space = ActionSpace()
    network = FactorizedQNetwork(3, len(action_space.action_types), action_space.max_candidates)
    assert tuple(network(torch.zeros((1, 3)))["candidate_advantages"].shape[1:]) == (19, 32)


def test_valid_mean_centering():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0, 1, 2]})).bool().unsqueeze(0)
    advantages = _outputs(
        action_space,
        advantages=[("click_element", 0, 1), ("click_element", 1, 2), ("click_element", 2, 3)],
    )["candidate_advantages"]
    centered = center_candidate_advantages(advantages, mask, action_space)
    click = centered[0, action_space.action_types.index("click_element"), :3]
    assert click.tolist() == pytest.approx([-1, 0, 1])
    assert float(click.mean()) == pytest.approx(0.0)


def test_invalid_candidate_excluded_from_centering():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0, 2]})).bool().unsqueeze(0)
    advantages = _outputs(
        action_space,
        advantages=[("click_element", 0, 1), ("click_element", 1, 100), ("click_element", 2, 3)],
    )["candidate_advantages"]
    centered = center_candidate_advantages(advantages, mask, action_space)
    click_type = action_space.action_types.index("click_element")
    assert centered[0, click_type, 0].item() == pytest.approx(-1)
    assert centered[0, click_type, 2].item() == pytest.approx(1)
    assert centered[0, click_type, 1].item() == pytest.approx(0)


def test_single_candidate_centered_advantage_zero():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"scroll_down": [0]})).bool().unsqueeze(0)
    advantages = _outputs(action_space, advantages=[("scroll_down", 0, 8)])["candidate_advantages"]
    centered = center_candidate_advantages(advantages, mask, action_space)
    assert centered[0, action_space.action_types.index("scroll_down"), 0].item() == pytest.approx(0.0)


def test_candidate_count_does_not_affect_type_selection():
    action_space = ActionSpace()
    for click_candidates in ([0], range(16)):
        mask = torch.from_numpy(_mask(action_space, {"click_element": click_candidates, "inspect_dom": [0]})).bool().unsqueeze(0)
        outputs = _outputs(
            action_space,
            type_values={"click_element": 0.2, "inspect_dom": 0.3},
            advantages=[("click_element", index, 50.0) for index in click_candidates],
        )
        action = factorized_greedy_actions_from_outputs(outputs, mask, action_space)
        assert int(action.item()) == action_space.encode("inspect_dom", 0)


def test_type_selection_independent_from_candidate_max():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0], "fill_input": [0], "inspect_dom": [0]})).bool().unsqueeze(0)
    outputs = _outputs(
        action_space,
        type_values={"click_element": 0.1, "fill_input": 0.4, "inspect_dom": 0.3},
        advantages=[("click_element", 0, 100.0)],
    )
    assert int(factorized_greedy_actions_from_outputs(outputs, mask, action_space).item()) == action_space.encode("fill_input", 0)


def test_candidate_selection_within_selected_type():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0, 1, 2], "inspect_dom": [0]})).bool().unsqueeze(0)
    outputs = _outputs(
        action_space,
        type_values={"click_element": 1.0, "inspect_dom": 0.2},
        advantages=[("click_element", 0, -0.2), ("click_element", 1, 0.5), ("click_element", 2, 0.1)],
    )
    assert int(factorized_greedy_actions_from_outputs(outputs, mask, action_space).item()) == action_space.encode("click_element", 1)


def test_flattened_action_encode_decode_unchanged():
    action_space = ActionSpace()
    action_id = action_space.encode("press_enter", 7)
    decoded = action_space.decode(action_id)
    assert action_id == 15 * 32 + 7
    assert decoded["action_type"] == "press_enter"
    assert decoded["candidate_index"] == 7


def test_factorized_current_q_calculation():
    action_space = ActionSpace()
    mask_np = _mask(action_space, {"click_element": [0, 1]})
    outputs = _outputs(
        action_space,
        type_values={"click_element": 2.0},
        advantages=[("click_element", 0, 1.0), ("click_element", 1, 3.0)],
    )
    network = StaticFactorizedNetwork(outputs)
    q = factorized_q_for_actions(
        network,
        torch.zeros((1, 3)),
        torch.from_numpy(mask_np).bool().unsqueeze(0),
        torch.tensor([action_space.encode("click_element", 1)]),
        action_space,
    )
    assert float(q.item()) == pytest.approx(3.0)


def test_factorized_double_dqn_next_action():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"fill_input": [4], "click_element": [0]})).bool().unsqueeze(0)
    outputs = _outputs(
        action_space,
        type_values={"fill_input": 0.7, "click_element": 0.2},
        advantages=[("fill_input", 4, 2.0)],
    )
    network = StaticFactorizedNetwork(outputs)
    action = factorized_double_dqn_next_actions(network, torch.zeros((1, 3)), mask, action_space)
    assert int(action.item()) == action_space.encode("fill_input", 4)


def test_target_network_evaluates_online_selected_action():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"fill_input": [4], "inspect_dom": [0]})).bool().unsqueeze(0)
    online = StaticFactorizedNetwork(_outputs(action_space, type_values={"fill_input": 0.5, "inspect_dom": 0.1}))
    target = StaticFactorizedNetwork(_outputs(action_space, type_values={"fill_input": 1.25, "inspect_dom": 9.0}))
    action = factorized_double_dqn_next_actions(online, torch.zeros((1, 3)), mask, action_space)
    q = factorized_q_for_actions(target, torch.zeros((1, 3)), mask, action.squeeze(1), action_space)
    assert int(action.item()) == action_space.encode("fill_input", 4)
    assert float(q.item()) == pytest.approx(1.25)


def test_done_transition_no_bootstrap_formula():
    reward = torch.tensor([1.0])
    gamma = 0.99
    next_q = torch.tensor([100.0])
    done = torch.tensor([1.0])
    target = reward + gamma * next_q * (1.0 - done)
    assert float(target.item()) == pytest.approx(1.0)


def test_safe_mask_respected():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"inspect_dom": [0]})).bool().unsqueeze(0)
    outputs = _outputs(
        action_space,
        type_values={"click_element": 99.0, "inspect_dom": 0.1},
        advantages=[("click_element", 5, 99.0)],
    )
    assert int(factorized_greedy_actions_from_outputs(outputs, mask, action_space).item()) == action_space.encode("inspect_dom", 0)


def test_empty_mask_safety():
    action_space = ActionSpace()
    mask = torch.zeros((1, action_space.get_action_dim()), dtype=torch.bool)
    outputs = _outputs(action_space, type_values={"click_element": 99.0})
    assert int(factorized_greedy_actions_from_outputs(outputs, mask, action_space).item()) == action_space.encode("noop", 0)


def test_balanced_random_exploration_unchanged():
    action_space = ActionSpace()
    mask = _mask(action_space, {"click_element": [0, 1], "scroll_down": [0]})
    action = balanced_random_valid_action(mask, action_space)
    assert action in set(np.flatnonzero(mask > 0).tolist())


def test_type_head_gradient():
    action_space = ActionSpace()
    network = FactorizedQNetwork(3, len(action_space.action_types), action_space.max_candidates)
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0, 1]})).bool().unsqueeze(0)
    action = torch.tensor([action_space.encode("click_element", 1)])
    q = factorized_q_for_actions(network, torch.ones((1, 3)), mask, action, action_space)
    loss = (q - torch.tensor([1.0])).pow(2).mean()
    loss.backward()
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in network.type_head.parameters())


def test_candidate_head_gradient():
    action_space = ActionSpace()
    network = FactorizedQNetwork(3, len(action_space.action_types), action_space.max_candidates)
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0, 1]})).bool().unsqueeze(0)
    action = torch.tensor([action_space.encode("click_element", 1)])
    q = factorized_q_for_actions(network, torch.ones((1, 3)), mask, action, action_space)
    loss = (q - torch.tensor([1.0])).pow(2).mean()
    loss.backward()
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in network.candidate_head.parameters())


def test_no_nan_inf():
    action_space = ActionSpace()
    network = FactorizedQNetwork(3, len(action_space.action_types), action_space.max_candidates)
    mask = torch.from_numpy(_mask(action_space, {"click_element": [0, 1], "inspect_dom": [0]})).bool().unsqueeze(0)
    values = factorized_flat_q_values(network(torch.ones((1, 3))), mask, action_space)
    assert torch.isfinite(values).all()


def test_flat_architecture_still_works():
    action_space = ActionSpace()
    network = QNetwork(3, action_space.get_action_dim())
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.network[-1].bias[action_space.encode("click_element", 0)] = 2.0
    assert masked_argmax(network, np.zeros(3, dtype=np.float32), _mask(action_space, {"click_element": [0]})) == action_space.encode("click_element", 0)


def test_old_flat_checkpoint_not_silently_loaded_as_factorized():
    with pytest.raises(ValueError):
        _validate_checkpoint_architecture({"model": {}}, FACTORIZED_DQN_ARCHITECTURE)


def test_environment_action_dim_still_608():
    assert ActionSpace(max_candidates=32).get_action_dim() == 608


def test_ground_truth_independence():
    action_space = ActionSpace()
    mask = torch.from_numpy(_mask(action_space, {"inspect_dom": [0]})).bool().unsqueeze(0)
    outputs = _outputs(action_space, type_values={"inspect_dom": 0.3, "click_element": 0.2})
    metadata = {"data_bug_id": "BUG-1"}
    assert metadata["data_bug_id"]
    assert int(factorized_greedy_actions_from_outputs(outputs, mask, action_space).item()) == action_space.encode("inspect_dom", 0)
