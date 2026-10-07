import numpy as np

from dqn_error_discovery.runners.dqn_click_bias_diagnostics import (
    candidate_reward_signature_dedup,
    controlled_max_over_many_simulation,
    equivalent_click_group_summary,
    expected_random_action_type_distribution,
    logical_click_group_key,
    q_statistics_by_action_type,
    repeat_diagnostics,
    training_distribution_summary,
    valid_count_by_action_type,
)
from models.action_space import ActionSpace


def _candidate(**extra):
    item = {
        "bid": "1",
        "role": "button",
        "tag": "button",
        "name": "Open",
        "href": "",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "bbox": [0, 0, 10, 10],
    }
    item.update(extra)
    return item


def test_candidate_count_increases_expected_max_diagnostic():
    result = controlled_max_over_many_simulation(mean=0.0, std=1.0, candidate_counts=(1, 4, 16), trials=2000, seed=7)
    assert result["16"] > result["4"] > result["1"]


def test_action_type_valid_count():
    action_space = ActionSpace()
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[action_space.encode("click_element", 0)] = 1
    mask[action_space.encode("click_element", 1)] = 1
    mask[action_space.encode("scroll_down", 0)] = 1
    counts = valid_count_by_action_type(mask, action_space)
    assert counts["click_element"] == 2
    assert counts["scroll_down"] == 1


def test_action_type_q_statistics():
    action_space = ActionSpace()
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    q_values = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for idx, value in enumerate([1.0, 3.0, 5.0]):
        action_id = action_space.encode("click_element", idx)
        mask[action_id] = 1
        q_values[action_id] = value
    scroll_id = action_space.encode("scroll_down", 0)
    mask[scroll_id] = 1
    q_values[scroll_id] = 2.0
    stats = q_statistics_by_action_type(q_values, mask, action_space)
    assert stats["click_element"]["q_mean"] == 3.0
    assert stats["click_element"]["q_max"] == 5.0
    assert stats["click_element"]["q_median"] == 3.0
    assert stats["scroll_down"]["q_std"] == 0.0


def test_max_selection_uplift():
    action_space = ActionSpace()
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    q_values = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for idx, value in enumerate([0.0, 0.0, 3.0]):
        action_id = action_space.encode("click_element", idx)
        mask[action_id] = 1
        q_values[action_id] = value
    stats = q_statistics_by_action_type(q_values, mask, action_space)
    assert stats["click_element"]["max_selection_uplift"] == 2.0


def test_uniform_valid_action_expected_probability():
    action_space = ActionSpace()
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for idx in range(3):
        mask[action_space.encode("click_element", idx)] = 1
    mask[action_space.encode("scroll_down", 0)] = 1
    expected = expected_random_action_type_distribution(mask, action_space)
    assert expected["click_element"] == 0.75
    assert expected["scroll_down"] == 0.25


def test_random_action_distribution_aggregation():
    manifest = {
        "action_distribution": {"click_element": 5},
        "random_action_distribution": {"click_element": 2, "scroll_down": 2},
        "greedy_action_distribution": {},
        "episode_summaries": [],
    }
    summary = training_distribution_summary(manifest)
    assert summary["random_click_ratio"] == 0.5


def test_greedy_action_distribution_aggregation():
    manifest = {
        "action_distribution": {"click_element": 5},
        "random_action_distribution": {},
        "greedy_action_distribution": {"click_element": 3, "inspect_dom": 1},
        "episode_summaries": [],
    }
    summary = training_distribution_summary(manifest)
    assert summary["greedy_click_ratio"] == 0.75


def test_same_href_candidates_equivalent_group():
    rows = [
        {"logical_group": logical_click_group_key(_candidate(bid="1", href="/products/1")), "q": 1.0, "bid": "1"},
        {"logical_group": logical_click_group_key(_candidate(bid="2", href="/products/1")), "q": 2.0, "bid": "2"},
    ]
    summary = equivalent_click_group_summary(rows)
    assert summary["equivalent_click_groups"] == 1
    assert summary["duplication_factor"] == 2.0


def test_different_entity_id_different_logical_group():
    assert logical_click_group_key(_candidate(entity_id="a")) != logical_click_group_key(_candidate(entity_id="b"))


def test_immediate_repeat_detection():
    result = repeat_diagnostics(["A", "A", "B"], window=4)
    assert result["immediate_repeat_count"] == 1


def test_recent_window_repeat_detection():
    result = repeat_diagnostics(["A", "B", "A"], window=4)
    assert result["recent_window_repeat_count"] == 1
    assert result["repeat_blind_spot_count"] == 1


def test_cycle_repeat_detection():
    result = repeat_diagnostics(["A", "B", "A", "B"], window=4)
    assert result["cycle_repeat_count"] >= 1


def test_candidate_reward_signature_repeat_dedup():
    result = candidate_reward_signature_dedup(["sig-a", "sig-a", "sig-b"])
    assert result["unique_signature_count"] == 2
    assert result["repeated_signature_count"] == 1


def test_ground_truth_metadata_does_not_affect_logical_group():
    plain = logical_click_group_key(_candidate(href="/products/1"))
    with_gt = logical_click_group_key(_candidate(href="/products/1", data_bug_id="BUG", catalog_bug_id_matches=["BUG"]))
    assert with_gt == plain
