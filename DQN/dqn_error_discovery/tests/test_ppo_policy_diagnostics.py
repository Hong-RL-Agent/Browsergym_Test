import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer
from dqn_error_discovery.runners.ppo_policy_diagnostics import (
    _sanitize_diagnostic,
    action_type_probability_mass,
    masked_policy_distribution,
    summarize_advantages,
    top_k_valid_actions,
)
from models.action_space import ActionSpace


def _obs(candidates=None):
    return {
        "page_state": {"url": "http://localhost:9220/", "site_id": "site001"},
        "runtime_signals": {},
        "infra_signals": {},
        "history": {"step_index": 10},
        "candidate_elements": candidates or [_candidate("a"), _candidate("b")],
    }


def _candidate(bid="a", **extra):
    item = {
        "bid": bid,
        "role": "button",
        "tag": "button",
        "name": "Button",
        "text": "Button",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "visibility": 1.0,
        "bbox": [0, 0, 20, 20],
        "source": "axtree",
    }
    item.update(extra)
    return item


def _agent_and_mask():
    torch.manual_seed(7)
    action_space = ActionSpace(max_candidates=32)
    agent = PPOAgent(635, action_space.get_action_dim())
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[action_space.encode("click_element", 0)] = 1.0
    mask[action_space.encode("inspect_dom", 0)] = 1.0
    return agent, action_space, mask


def test_masked_distribution_invalid_action_probability_zero():
    agent, action_space, mask = _agent_and_mask()
    distribution = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), mask)

    assert distribution["probabilities"][action_space.encode("click_element", 1)] == 0.0


def test_masked_rollout_log_prob_is_finite():
    agent, _, mask = _agent_and_mask()
    selected = agent.select_action(np.zeros(635, dtype=np.float32), mask)

    assert np.isfinite(selected["log_prob"])


def test_update_log_prob_receives_same_stored_mask(monkeypatch):
    agent, action_space, mask = _agent_and_mask()
    buffer = RolloutBuffer()
    action_id = action_space.encode("inspect_dom", 0)
    buffer.add(np.zeros(635, dtype=np.float32), action_id, -0.5, 1.0, True, 0.0, mask, {})
    buffer.compute_returns_and_advantages(0.0, agent.gamma, agent.gae_lambda)
    captured = {}
    original = agent.policy.evaluate_actions

    def wrapped(obs, actions, action_masks):
        captured["mask"] = action_masks.detach().cpu().numpy()
        return original(obs, actions, action_masks)

    monkeypatch.setattr(agent.policy, "evaluate_actions", wrapped)
    agent.update(buffer)

    assert captured["mask"][0, action_id] == 1.0
    assert captured["mask"][0, action_space.encode("click_element", 1)] == 0.0


def test_rollout_buffer_stores_action_mask_alignment():
    _, action_space, mask = _agent_and_mask()
    buffer = RolloutBuffer()
    action_id = action_space.encode("click_element", 0)
    buffer.add(np.zeros(635, dtype=np.float32), action_id, -0.5, 1.0, True, 0.0, mask, {})

    assert np.array_equal(buffer.action_masks[0], mask)


def test_candidate_independent_action_canonical_slot():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs())

    assert mask[action_space.encode("scroll_down", 0)] == 1.0
    assert sum(mask[action_space.encode("scroll_down", index)] for index in range(1, 32)) == 0.0


def test_action_type_probability_mass_sums_to_one():
    agent, action_space, mask = _agent_and_mask()
    distribution = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), mask)
    masses = action_type_probability_mass(distribution["probabilities"], mask, action_space)

    assert abs(sum(masses.values()) - 1.0) < 1e-6


def test_top_k_returns_valid_actions_only():
    agent, action_space, mask = _agent_and_mask()
    distribution = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), mask)
    rows = top_k_valid_actions(distribution["probabilities"], distribution["logits"], mask, action_space, _obs(), top_k=5)

    assert rows
    assert all(mask[row["action_id"]] == 1.0 for row in rows)


def test_greedy_argmax_valid_action():
    agent, _, mask = _agent_and_mask()
    selected = agent.select_greedy_action(np.zeros(635, dtype=np.float32), mask)

    assert mask[selected["action_id"]] == 1.0


def test_stochastic_sample_valid_action():
    agent, _, mask = _agent_and_mask()

    assert all(mask[agent.select_action(np.zeros(635, dtype=np.float32), mask)["action_id"]] == 1.0 for _ in range(20))


def test_entropy_uses_masked_distribution():
    agent, _, mask = _agent_and_mask()
    full_mask = np.ones_like(mask)
    masked_entropy = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), mask)["entropy"]
    full_entropy = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), full_mask)["entropy"]

    assert masked_entropy < full_entropy


def test_action_type_probability_aggregation():
    agent, action_space, mask = _agent_and_mask()
    distribution = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), mask)
    masses = action_type_probability_mass(distribution["probabilities"], mask, action_space)

    assert set(k for k, v in masses.items() if v > 0) == {"click_element", "inspect_dom"}


def test_advantage_normalization_finite():
    summary = summarize_advantages(np.array([1.0, -1.0, 0.5], dtype=np.float32))

    assert summary["finite"] is True


def test_done_boundary_prevents_advantage_leakage():
    buffer = RolloutBuffer()
    mask = np.ones(608, dtype=np.float32)
    buffer.add(np.zeros(635, dtype=np.float32), 0, 0.0, 1.0, True, 0.0, mask, {})
    buffer.add(np.zeros(635, dtype=np.float32), 0, 0.0, 10.0, False, 0.0, mask, {})
    buffer.compute_returns_and_advantages(0.0)

    assert buffer.advantages[0] == 1.0


def test_ground_truth_metadata_does_not_affect_logits_or_mask():
    agent, action_space, _ = _agent_and_mask()
    plain = _obs([_candidate("a")])
    with_gt = _obs([_candidate("a", data_bug_id="BUG", catalog_bug_id_matches=["BUG"])])
    plain_mask = action_space.build_action_mask(plain)
    gt_mask = action_space.build_action_mask(with_gt)
    plain_dist = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), plain_mask)
    gt_dist = masked_policy_distribution(agent, np.zeros(635, dtype=np.float32), gt_mask)

    assert np.array_equal(plain_mask, gt_mask)
    assert np.allclose(plain_dist["probabilities"], gt_dist["probabilities"])


def test_query_or_volatile_metadata_not_kept_in_diagnostic():
    sanitized = _sanitize_diagnostic(
        {
            "query": "secret search",
            "candidate": {"password": "secret", "bid": "11"},
            "action_type": "click_element",
        }
    )

    assert "query" not in sanitized
    assert "password" not in sanitized["candidate"]
    assert sanitized["candidate"]["bid"] == "11"
