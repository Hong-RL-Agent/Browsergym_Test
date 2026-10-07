import numpy as np
import pytest
import torch

from dqn_error_discovery.agents.dqn_agent import QNetwork, Transition
from dqn_error_discovery.runners.dqn_qbias_diagnostics import (
    aggregate_reward_components,
    double_dqn_target_samples,
    q_stats_by_action_type,
)
from dqn_error_discovery.runners.train_dqn_error_detector import DQN_REWARD_CONFIG, calculate_dqn_reward
from models.action_space import ActionSpace


def _candidate(bid="target", role="button", tag="button", **extra):
    item = {
        "bid": bid,
        "name": "Target",
        "text": "Target",
        "role": role,
        "tag": tag,
        "clickable": True,
        "visible": True,
        "enabled": True,
        "visibility": 1.0,
        "bbox": [10, 10, 100, 30],
    }
    item.update(extra)
    return item


def _obs(url="http://app.test/", *, text_len=100, candidates=None, **page_extra):
    page = {
        "url": url,
        "title": "Page",
        "dom_node_count": 10,
        "page_text_length": text_len,
        "quantity": 1,
    }
    page.update(page_extra)
    return {
        "page_state": page,
        "candidate_elements": candidates if candidates is not None else [_candidate()],
        "runtime_signals": {},
    }


def _action(action_type="scroll_down", candidate_index=0, **extra):
    item = {"action_type": action_type, "candidate_index": candidate_index}
    item.update(extra)
    return item


def _anomaly(signature="sig-a", **extra):
    item = {
        "type": "button-no-response",
        "rule_id": "R01",
        "status": "candidate",
        "signature": signature,
        "confirmation_policy": "reproduce",
        "action_type": "click_element",
        "target_id": "target",
    }
    item.update(extra)
    return item


def test_same_scroll_down_new_local_action_target_once():
    history = {}
    before = _obs()
    after = _obs(text_len=120)
    _, first = calculate_dqn_reward(before, after, _action("scroll_down"), [], history)
    _, second = calculate_dqn_reward(before, after, _action("scroll_down"), [], history)
    assert first["new_local_action_target"] == DQN_REWARD_CONFIG["new_local_action_target"]
    assert second["new_local_action_target"] == 0.0


def test_repeat_scroll_applies_repeat_action_penalty():
    history = {}
    before = _obs()
    after = _obs(text_len=120)
    calculate_dqn_reward(before, after, _action("scroll_down"), [], history)
    _, components = calculate_dqn_reward(before, after, _action("scroll_down"), [], history)
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]


def test_reward_tracker_same_episode_persists():
    history = {}
    calculate_dqn_reward(_obs(), _obs(text_len=120), _action("scroll_down"), [], history)
    assert "local_seen_states" in history
    assert "local_seen_action_targets" in history


def test_episode_boundary_tracker_reset():
    before = _obs()
    after = _obs(text_len=120)
    _, first = calculate_dqn_reward(before, after, _action("scroll_down"), [], {})
    _, reset_first = calculate_dqn_reward(before, after, _action("scroll_down"), [], {})
    assert first["new_local_state"] == reset_first["new_local_state"]
    assert reset_first["repeat_action_penalty"] == 0.0


def test_same_local_state_repeated_no_new_local_state():
    history = {}
    same = _obs()
    calculate_dqn_reward(same, same, _action("noop"), [], history)
    _, components = calculate_dqn_reward(same, same, _action("noop"), [], history)
    assert components["new_local_state"] == 0.0
    assert components["repeat_state_penalty"] == DQN_REWARD_CONFIG["repeat_state"]


def test_candidate_signature_reward_once():
    history = {}
    first_reward, first = calculate_dqn_reward(_obs(), _obs(), _action("click_element"), [_anomaly("same")], history)
    second_reward, second = calculate_dqn_reward(_obs(), _obs(), _action("click_element"), [_anomaly("same")], history)
    assert first["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert second["candidate_anomaly"] == 0.0
    assert second["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert first_reward != second_reward


def test_next_state_masked_argmax_invalid_action_not_selected():
    action_space = ActionSpace()
    net = QNetwork(3, action_space.get_action_dim())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.network[-1].bias[action_space.encode("click_element", 5)] = 100.0
        net.network[-1].bias[action_space.encode("scroll_down", 0)] = 1.0
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[action_space.encode("scroll_down", 0)] = 1.0
    transition = Transition(np.zeros(3, dtype=np.float32), 0, 0.0, np.zeros(3, dtype=np.float32), False, mask)
    sample = double_dqn_target_samples(net, net, [transition], action_space, 0.99)[0]
    assert sample["online_next_argmax_action"]["action_type"] == "scroll_down"
    assert sample["online_next_argmax_valid"] is True


def test_double_dqn_target_uses_valid_next_action():
    action_space = ActionSpace()
    net = QNetwork(3, action_space.get_action_dim())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.network[-1].bias[action_space.encode("inspect_dom", 0)] = 2.0
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[action_space.encode("inspect_dom", 0)] = 1.0
    transition = Transition(np.zeros(3, dtype=np.float32), 0, 1.0, np.zeros(3, dtype=np.float32), False, mask)
    sample = double_dqn_target_samples(net, net, [transition], action_space, 0.5)[0]
    assert sample["td_target"] == 2.0


def test_done_transition_no_bootstrap():
    action_space = ActionSpace()
    net = QNetwork(3, action_space.get_action_dim())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.network[-1].bias[action_space.encode("inspect_dom", 0)] = 10.0
    mask = np.ones(action_space.get_action_dim(), dtype=np.float32)
    transition = Transition(np.zeros(3, dtype=np.float32), 0, 1.25, np.zeros(3, dtype=np.float32), True, mask)
    sample = double_dqn_target_samples(net, net, [transition], action_space, 0.99)[0]
    assert sample["td_target"] == 1.25


def test_action_type_reward_aggregation():
    records = [
        {"action_type": "scroll_down", "reward_components": {"new_local_state": 0.2, "final_reward": 0.4}},
        {"action_type": "scroll_down", "reward_components": {"repeat_action_penalty": -0.3, "final_reward": -0.3}},
    ]
    summary = aggregate_reward_components(records)
    assert summary["scroll_down"]["count"] == 2.0
    assert summary["scroll_down"]["new_local_state"] == 0.2
    assert summary["scroll_down"]["mean_final_reward"] == pytest.approx(0.05)


def test_action_type_q_aggregation():
    action_space = ActionSpace()
    q_values = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    q_values[action_space.encode("click_element", 0)] = 2.0
    q_values[action_space.encode("click_element", 1)] = 4.0
    q_values[action_space.encode("scroll_down", 0)] = 1.0
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[action_space.encode("click_element", 0)] = 1.0
    mask[action_space.encode("click_element", 1)] = 1.0
    mask[action_space.encode("scroll_down", 0)] = 1.0
    stats = q_stats_by_action_type(q_values, mask, action_space)
    assert stats["click_element"]["valid_count"] == 2.0
    assert stats["click_element"]["max_q"] == 4.0
    assert stats["click_element"]["mean_q"] == 3.0


def test_scroll_boundary_no_effect_penalty():
    same = _obs()
    _, components = calculate_dqn_reward(same, same, _action("scroll_down"), [], {})
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_ground_truth_metadata_does_not_affect_qbias_reward_diagnostic():
    plain_reward, plain_components = calculate_dqn_reward(_obs(), _obs(text_len=120), _action("click_element"), [], {})
    before = _obs(candidates=[_candidate(data_bug_id="BUG", catalog_bug_id_matches=["BUG"])])
    after = _obs(text_len=120, candidates=[_candidate(data_bug_id="BUG", catalog_bug_id_matches=["BUG"])])
    gt_reward, gt_components = calculate_dqn_reward(before, after, _action("click_element"), [], {})
    assert gt_reward == plain_reward
    assert gt_components["final_reward"] == plain_components["final_reward"]
