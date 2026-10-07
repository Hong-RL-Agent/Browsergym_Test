import numpy as np

from dqn_error_discovery.runners.dqn_candidate_reward_attribution import (
    candidate_reward_concentration,
    classify_candidate_attribution,
    click_effect_labels,
    logical_candidate_group_key,
    reward_combination_key,
    summarize_candidate_attribution,
    summarize_candidate_fragmentation,
    summarize_click_rewards,
    summarize_no_effect,
    summarize_repeat_positive,
    summarize_reward_combinations,
    summarize_rule_rewards,
    training_block_candidate_reward,
)
from dqn_error_discovery.runners.train_dqn_error_detector import DQN_REWARD_CONFIG
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward, initialize_dqn_candidate_reward_baseline


def _anomaly(signature="sig-a", rule_id="R01", **extra):
    item = {
        "rule_id": rule_id,
        "status": "candidate",
        "signature": signature,
        "target_id": "11",
        "action_type": "click_element",
        "url_path": "/products",
        "issue_type": "no_response",
        "confirmation_policy": "reproduce",
        "evidence": {"rule_id": rule_id, "clicked_bid": "11", "clicked_name": "Buy"},
    }
    item.update(extra)
    return item


def _record(action_type="click_element", reward=0.0, components=None, anomalies=None, **extra):
    item = {
        "episode": 1,
        "step": 1,
        "action_type": action_type,
        "reward_total": reward,
        "reward_components": components or {},
        "candidate_anomalies": anomalies or [],
        "immediate_repeat": False,
        "functional_state_changed": False,
        "raw_state_changed": False,
        "viewport_context_changed": False,
        "action_error": False,
    }
    item.update(extra)
    return item


def test_candidate_reward_combination_candidate_repeat_no_effect():
    key = reward_combination_key(
        {
            "candidate_anomaly": 0.7,
            "repeat_action_penalty": -0.3,
            "no_effect_penalty": -0.3,
        }
    )
    assert key == "candidate+repeat+no_effect"


def test_candidate_reward_combination_none_is_stable():
    assert reward_combination_key({}) == "none"


def test_click_reward_summary_positive_zero_negative_and_candidate_mean():
    records = [
        _record(reward=0.1, components={"candidate_anomaly": 0.7}),
        _record(reward=0.0, components={}),
        _record(reward=-0.3, components={"repeat_action_penalty": -0.3}),
        _record(action_type="scroll_down", reward=1.0, components={"candidate_anomaly": 0.7}),
    ]
    summary = summarize_click_rewards(records)
    assert summary["click_count"] == 3
    assert summary["positive"]["count"] == 1
    assert summary["zero"]["count"] == 1
    assert summary["negative"]["count"] == 1
    assert np.isclose(summary["candidate_reward_mean_per_click"], 0.7 / 3.0)


def test_first_seen_candidate_attribution_action_caused_r01():
    attribution = classify_candidate_attribution(
        anomaly=_anomaly(rule_id="R01"),
        action={"action_type": "click_element"},
        before_signatures=set(),
        functional_changed=False,
        raw_changed=False,
        viewport_changed=False,
    )
    assert attribution == "ACTION_CAUSED"


def test_preexisting_signature_overrides_action_caused():
    attribution = classify_candidate_attribution(
        anomaly=_anomaly(signature="same", rule_id="R01"),
        action={"action_type": "click_element"},
        before_signatures={"same"},
        functional_changed=False,
        raw_changed=False,
        viewport_changed=False,
    )
    assert attribution == "PREEXISTING"


def test_layout_candidate_with_viewport_change_is_revealed():
    attribution = classify_candidate_attribution(
        anomaly=_anomaly(rule_id="R05"),
        action={"action_type": "scroll_down"},
        before_signatures=set(),
        functional_changed=False,
        raw_changed=True,
        viewport_changed=True,
    )
    assert attribution == "ACTION_REVEALED"


def test_logical_group_ignores_bbox_and_viewport_metadata():
    left = logical_candidate_group_key(_anomaly(evidence={"rule_id": "R05", "target_bid": "20", "viewport_bbox": [0, 0, 100, 100]}))
    right = logical_candidate_group_key(_anomaly(evidence={"rule_id": "R05", "target_bid": "20", "viewport_bbox": [0, 100, 100, 100]}))
    assert left == right


def test_ground_truth_metadata_does_not_affect_logical_group():
    plain = logical_candidate_group_key(_anomaly())
    with_gt = logical_candidate_group_key(
        _anomaly(data_bug_id="BUG", catalog_bug_id_matches=["BUG"], evidence={"rule_id": "R01", "clicked_bid": "11", "data_bug_id": "BUG"})
    )
    assert with_gt == plain


def test_signature_fragmentation_detects_multiple_raw_for_one_logical_group():
    group = logical_candidate_group_key(_anomaly(signature="a"))
    records = [
        _record(anomalies=[{"rule_id": "R01", "signature": "a", "logical_group": group}]),
        _record(anomalies=[{"rule_id": "R01", "signature": "b", "logical_group": group}]),
    ]
    summary = summarize_candidate_fragmentation(records)
    assert summary["raw_unique_candidate_signatures"] == 2
    assert summary["logical_unique_candidate_groups"] == 1
    assert summary["candidate_signature_fragmentation_ratio"] == 2.0


def test_rule_reward_summary_counts_rewarded_first_seen_only():
    records = [
        _record(
            components={"candidate_anomaly": DQN_REWARD_CONFIG["candidate_anomaly"]},
            anomalies=[{"rule_id": "R01", "signature": "a", "logical_group": "g", "first_seen": True}],
        ),
        _record(
            components={"candidate_anomaly": 0.0},
            anomalies=[{"rule_id": "R01", "signature": "a", "logical_group": "g", "first_seen": False}],
        ),
    ]
    summary = summarize_rule_rewards(records)
    assert summary["R01"]["observation_count"] == 2
    assert summary["R01"]["rewarded_signature_count"] == 1


def test_candidate_attribution_click_ratios():
    records = [
        _record(anomalies=[{"rule_id": "R01", "signature": "a", "attribution": "ACTION_CAUSED"}]),
        _record(anomalies=[{"rule_id": "R05", "signature": "b", "attribution": "PREEXISTING"}]),
    ]
    summary = summarize_candidate_attribution(records)
    assert summary["click"]["ACTION_CAUSED"] == 1
    assert summary["click"]["PREEXISTING"] == 1
    assert summary["click_ratios"]["PREEXISTING"] == 0.5


def test_candidate_reward_concentration():
    records = [
        _record(reward=0.1, components={"candidate_anomaly": 0.7, "repeat_action_penalty": -0.3, "no_effect_penalty": -0.3}),
        _record(reward=0.2, components={"new_local_action_target": 0.2}),
    ]
    summary = candidate_reward_concentration(records)
    assert summary["click_candidate_positive_mass"] == 0.7
    assert summary["click_candidate_positive_mass_ratio"] > 1.0


def test_no_effect_analysis_finds_functional_unchanged_without_no_effect():
    records = [_record(reward=0.1, components={"candidate_anomaly": 0.7})]
    summary = summarize_no_effect(records)
    assert summary["functional_unchanged_without_no_effect_count"] == 1


def test_repeat_positive_analysis_counts_repeat_positive_click():
    records = [
        _record(
            reward=0.1,
            components={"candidate_anomaly": 0.7, "repeat_action_penalty": -0.3, "no_effect_penalty": -0.3},
            immediate_repeat=True,
        )
    ]
    summary = summarize_repeat_positive(records)
    assert summary["repeat_positive_click_count"] == 1
    assert summary["repeat_no_effect_positive_click_count"] == 1


def test_click_effect_labels_separates_no_effect_and_anomaly_causal():
    labels = click_effect_labels(
        components={"no_effect_penalty": -0.3},
        functional_changed=False,
        raw_changed=False,
        viewport_changed=False,
        attribution_counts={"ACTION_CAUSED": 1},
    )
    assert "ANOMALY_CAUSAL" in labels
    assert "NO_EFFECT" in labels


def test_training_block_candidate_reward_aggregates_click_components():
    manifest = {
        "episode_summaries": [
            {
                "episode": 1,
                "action_distribution": {"click_element": 2},
                "action_type_reward_component_totals": {"click_element": {"candidate_anomaly": 0.7, "final_reward": 0.1}},
                "action_type_reward_component_event_counts": {"click_element": {"candidate_anomaly": 1, "repeat_action_penalty": 2}},
            },
            {
                "episode": 11,
                "action_distribution": {"click_element": 1},
                "action_type_reward_component_totals": {"click_element": {"candidate_anomaly": 0.0, "final_reward": -0.3}},
                "action_type_reward_component_event_counts": {"click_element": {"no_effect_penalty": 1}},
            },
        ]
    }
    summary = training_block_candidate_reward(manifest)
    assert summary["EP1-10"]["click_count"] == 2
    assert summary["EP1-10"]["candidate_reward_click_event_count"] == 1
    assert summary["EP11-20"]["no_effect_click_count"] == 1


def test_reward_combination_summary_aggregates_records():
    records = [
        _record(components={"candidate_anomaly": 0.7}),
        _record(components={"repeat_action_penalty": -0.3}),
    ]
    summary = summarize_reward_combinations(records)
    assert summary["candidate_only"] == 1
    assert summary["repeat_only"] == 1


def _obs(candidates=None, **page_extra):
    page_state = {
        "url": "http://app.test/products",
        "title": "Products",
        "page_text_length": 100,
        "dom_node_count": 10,
    }
    page_state.update(page_extra)
    return {"page_state": page_state, "candidate_elements": candidates or [{"bid": "11", "role": "button", "name": "Buy"}]}


def test_initial_preexisting_candidate_gets_no_candidate_reward():
    history = {}
    anomaly = _anomaly(signature="r05-a", rule_id="R05", target_id="18", evidence={"rule_id": "R05", "target_bid": "18"})
    initialize_dqn_candidate_reward_baseline(history, [anomaly])
    _, components = calculate_dqn_reward(
        _obs(),
        _obs(page_text_length=120),
        {"action_type": "scroll_down", "candidate_index": 0},
        [anomaly],
        history,
    )
    assert components["candidate_anomaly"] == 0.0


def test_first_r01_same_click_target_is_causally_eligible():
    history = {}
    anomaly = _anomaly(signature="r01-11", rule_id="R01", target_id="11", evidence={"rule_id": "R01", "clicked_bid": "11"})
    _, components = calculate_dqn_reward(
        _obs(candidates=[{"bid": "11", "role": "button", "name": "Buy"}]),
        _obs(candidates=[{"bid": "11", "role": "button", "name": "Buy"}]),
        {"action_type": "click_element", "candidate_index": 0},
        [anomaly],
        history,
    )
    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]


def test_repeated_r01_same_logical_target_gets_no_candidate_reward():
    history = {}
    first = _anomaly(signature="r01-a", rule_id="R01", target_id="11", evidence={"rule_id": "R01", "clicked_bid": "11"})
    second = _anomaly(signature="r01-b", rule_id="R01", target_id="11", evidence={"rule_id": "R01", "clicked_bid": "11"})
    calculate_dqn_reward(_obs(), _obs(), {"action_type": "click_element", "candidate_index": 0}, [first], history)
    _, components = calculate_dqn_reward(_obs(), _obs(), {"action_type": "click_element", "candidate_index": 0}, [second], history)
    assert components["candidate_anomaly"] == 0.0


def test_scroll_revealed_r05_new_logical_candidate_is_eligible():
    history = {}
    anomaly = _anomaly(signature="r05-a", rule_id="R05", target_id="18", evidence={"rule_id": "R05", "target_bid": "18"})
    _, components = calculate_dqn_reward(
        _obs(page_text_length=100),
        _obs(page_text_length=300),
        {"action_type": "scroll_down", "candidate_index": 0},
        [anomaly],
        history,
    )
    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]


def test_r05_raw_signature_fragmentation_is_deduped_by_logical_key():
    history = {}
    first = _anomaly(signature="r05-a", rule_id="R05", target_id="18", evidence={"rule_id": "R05", "target_bid": "18", "viewport_bbox": [0, 0, 100, 100]})
    second = _anomaly(signature="r05-b", rule_id="R05", target_id="18", evidence={"rule_id": "R05", "target_bid": "18", "viewport_bbox": [0, 100, 100, 100]})
    calculate_dqn_reward(_obs(page_text_length=100), _obs(page_text_length=300), {"action_type": "scroll_down", "candidate_index": 0}, [first], history)
    _, components = calculate_dqn_reward(_obs(page_text_length=300), _obs(page_text_length=500), {"action_type": "scroll_down", "candidate_index": 0}, [second], history)
    assert components["candidate_anomaly"] == 0.0


def test_unknown_new_candidate_gets_no_candidate_reward():
    history = {}
    anomaly = _anomaly(signature="r05-unknown", rule_id="R05", target_id="18", evidence={"rule_id": "R05", "target_bid": "18"})
    _, components = calculate_dqn_reward(
        _obs(),
        _obs(),
        {"action_type": "click_element", "candidate_index": 0},
        [anomaly],
        history,
    )
    assert components["candidate_anomaly"] == 0.0
