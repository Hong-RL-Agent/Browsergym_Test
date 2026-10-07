import argparse

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.runners.ppo_explorer_multiseed import aggregate_seed_summaries
from dqn_error_discovery.runners.ppo_policy_diagnostics import masked_policy_distribution
from dqn_error_discovery.runners.run_policy_sanity_check import SanityMetricsAggregator, _selection_mode
from models.action_space import ActionSpace


def _agent_and_mask():
    torch.manual_seed(11)
    action_space = ActionSpace(max_candidates=32)
    agent = PPOAgent(635, action_space.get_action_dim())
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[action_space.encode("click_element", 0)] = 1.0
    mask[action_space.encode("inspect_dom", 0)] = 1.0
    mask[action_space.encode("scroll_down", 0)] = 1.0
    return agent, action_space, mask


def test_stochastic_selection_samples_masked_distribution():
    agent, _, mask = _agent_and_mask()
    selected = agent.select_action(np.zeros(635, dtype=np.float32), mask)

    assert mask[selected["action_id"]] == 1.0
    assert np.isfinite(selected["log_prob"])


def test_greedy_selection_is_valid_argmax():
    agent, action_space, mask = _agent_and_mask()
    distribution = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), mask)
    selected = agent.select_greedy_action(np.zeros(635, dtype=np.float32), mask)

    assert selected["action_id"] == int(np.argmax(distribution["probabilities"]))
    assert mask[selected["action_id"]] == 1.0
    assert action_space.decode(selected["action_id"])["candidate_index"] == 0


def test_invalid_action_not_selected_by_stochastic():
    agent, action_space, mask = _agent_and_mask()
    invalid_id = action_space.encode("click_element", 7)

    assert mask[invalid_id] == 0.0
    assert all(agent.select_action(np.zeros(635, dtype=np.float32), mask)["action_id"] != invalid_id for _ in range(50))


def test_same_logits_different_seed_stochastic_samples_are_valid():
    agent, _, mask = _agent_and_mask()
    sampled = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        sampled.append(agent.select_action(np.zeros(635, dtype=np.float32), mask)["action_id"])

    assert sampled
    assert all(mask[action_id] == 1.0 for action_id in sampled)


def test_greedy_result_is_deterministic():
    agent, _, mask = _agent_and_mask()
    first = agent.select_greedy_action(np.zeros(635, dtype=np.float32), mask)["action_id"]

    assert all(agent.select_greedy_action(np.zeros(635, dtype=np.float32), mask)["action_id"] == first for _ in range(10))


def test_selection_mode_recorded_from_new_argument():
    args = argparse.Namespace(mode="ppo", greedy=False, selection_mode="stochastic")

    assert _selection_mode(args) == "stochastic"


def test_legacy_greedy_flag_still_maps_to_greedy():
    args = argparse.Namespace(mode="ppo", greedy=True, selection_mode="")

    assert _selection_mode(args) == "greedy"


def test_multiseed_aggregate_mean_min_max():
    summaries = [
        {"metrics": {"unique_action_targets": 10, "repeat_action_target_ratio": 0.5}, "action_distribution": {"click_element": 20}},
        {"metrics": {"unique_action_targets": 20, "repeat_action_target_ratio": 0.25}, "action_distribution": {"click_element": 10}},
    ]
    aggregate = aggregate_seed_summaries(summaries)

    assert aggregate["metrics"]["unique_action_targets"] == {"mean": 15.0, "min": 10.0, "max": 20.0}
    assert aggregate["metrics"]["repeat_action_target_ratio"] == {"mean": 0.375, "min": 0.25, "max": 0.5}
    assert aggregate["action_distribution"]["click_element"] == {"mean": 15.0, "min": 10.0, "max": 20.0}


def test_empty_aggregate_has_no_divide_by_zero():
    aggregate = aggregate_seed_summaries([])

    assert aggregate["metrics"]["total_steps"] == {"mean": 0.0, "min": 0.0, "max": 0.0}
    assert aggregate["diagnostic"]["PPO_EXPLORER_ROLE_CHECK"] == "REVIEW"


def test_ground_truth_metadata_does_not_affect_explorer_metrics():
    plain = SanityMetricsAggregator("ppo")
    with_gt = SanityMetricsAggregator("ppo")
    before_plain = {"page_state": {"url": "/"}, "candidate_elements": [{"bid": "a", "role": "button"}]}
    before_gt = {"page_state": {"url": "/"}, "candidate_elements": [{"bid": "a", "role": "button", "data_bug_id": "BUG"}]}
    after = {"page_state": {"url": "/"}, "candidate_elements": [{"bid": "a", "role": "button"}]}
    action = {"action_type": "click_element", "candidate_index": 0}
    plain.record_step(step=1, before_observation=before_plain, after_observation=after, action=action, reward=0.0)
    with_gt.record_step(step=1, before_observation=before_gt, after_observation=after, action=action, reward=0.0)

    assert plain.summary()["metrics"]["unique_action_targets"] == with_gt.summary()["metrics"]["unique_action_targets"]


def test_stochastic_does_not_modify_logits_or_reward_policy():
    agent, _, mask = _agent_and_mask()
    obs = np.zeros(635, dtype=np.float32)
    before = masked_policy_distribution(agent, obs, mask)
    agent.select_action(obs, mask)
    after = masked_policy_distribution(agent, obs, mask)

    assert np.allclose(before["logits"], after["logits"])
    assert np.allclose(before["probabilities"], after["probabilities"])
