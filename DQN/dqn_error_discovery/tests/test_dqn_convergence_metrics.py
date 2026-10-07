from dqn_error_discovery.runners.dqn_convergence_metrics import (
    aggregate_training_block,
    block_training_summary,
    classify_convergence,
    convergence_summary,
    loss_trajectory,
)


def _episode(ep, *, click=0, scroll=0, random_click=0, greedy_click=0):
    return {
        "episode": ep,
        "steps": click + scroll,
        "action_distribution": {"click_element": click, "scroll_down": scroll},
        "random_action_count": random_click + scroll,
        "greedy_action_count": greedy_click,
        "random_action_distribution": {"click_element": random_click, "scroll_down": scroll},
        "greedy_action_distribution": {"click_element": greedy_click},
        "action_type_reward_summary": {
            "click_element": {
                "count": click,
                "reward_mean": 0.25,
                "positive_transition_count": click,
                "zero_transition_count": 0,
                "negative_transition_count": 0,
            }
        },
        "action_type_reward_component_totals": {"click_element": {"candidate_anomaly": 0.7}},
        "action_type_reward_component_event_counts": {"click_element": {"candidate_anomaly": 1}},
    }


def test_training_block_aggregation():
    block = aggregate_training_block([_episode(1, click=2, scroll=3, random_click=1, greedy_click=1)], 1, 10)
    assert block["total_actions"] == 5
    assert block["overall_click_ratio"] == 0.4
    assert block["random_click_ratio"] == 0.25
    assert block["click_reward"]["count"] == 2
    assert block["click_reward"]["mean"] == 0.25
    assert block["click_reward"]["component_event_counts"]["candidate_anomaly"] == 1


def test_ep10_20_30_block_summary_keys():
    summary = block_training_summary({"episode_summaries": [_episode(10), _episode(20), _episode(30)]})
    assert set(summary) == {"EP1-10", "EP11-20", "EP21-30"}


def test_q_uplift_and_checkpoint_ordering_in_summary():
    report = {
        "q_statistics_by_action_type": {"click_element": {"q_max_mean": 0.2, "q_mean_mean": -0.1, "q_std_mean": 0.1, "max_selection_uplift_mean": 0.3}},
        "random_vs_greedy_distribution": {"training_click_ratio": 0.4},
        "sampled_replay_distribution": {"counts": {"click_element": 4, "scroll_down": 6}},
        "immediate_repeat": {"immediate_repeat_ratio": 0.1, "recent_window_repeat_ratio": 0.2, "cycle_repeat_ratio": 0.0},
    }
    sanity = {"metrics": {"total_steps": 10, "repeat_local_action_ratio": 0.1}, "action_distribution": {"click_element": 5}}
    summary = convergence_summary(manifest={"epsilon_trajectory": [], "episode_summaries": []}, checkpoint_reports={"EP10": report}, sanity_reports={"EP10": sanity})
    assert summary["click_uplift_trajectory"]["EP10"] == 0.3
    assert summary["greedy_eval_trajectory"]["EP10"]["click_ratio"] == 0.5


def test_repeat_metric_aggregation_gap():
    report = {
        "q_statistics_by_action_type": {},
        "random_vs_greedy_distribution": {},
        "sampled_replay_distribution": {},
        "immediate_repeat": {"immediate_repeat_ratio": 0.3, "recent_window_repeat_ratio": 0.8, "cycle_repeat_ratio": 0.6},
    }
    summary = convergence_summary(manifest={"episode_summaries": []}, checkpoint_reports={"EP30": report}, sanity_reports={})
    assert summary["repeat_trajectory"]["EP30"]["recent_minus_immediate"] == 0.5


def test_td_error_and_loss_aggregation():
    manifest = {
        "loss_history": [
            {"episode": 1, "loss": 0.5, "td_error_abs_mean": 0.7},
            {"episode": 11, "loss": 0.25, "td_error_abs_mean": 0.4},
        ]
    }
    summary = loss_trajectory(manifest)
    assert summary["EP1-10"]["loss"]["mean"] == 0.5
    assert summary["EP11-20"]["td_error_abs_mean"]["mean"] == 0.4


def test_empty_metric_safety():
    summary = convergence_summary(manifest={}, checkpoint_reports={}, sanity_reports={})
    assert summary["q_trajectory"] == {}
    assert loss_trajectory({})["EP1-10"]["loss"]["mean"] is None


def test_ground_truth_independence():
    report = {
        "q_statistics_by_action_type": {"click_element": {"q_mean_mean": -0.1, "max_selection_uplift_mean": 0.3}},
        "random_vs_greedy_distribution": {"training_click_ratio": 0.1, "data_bug_id": "BUG"},
        "sampled_replay_distribution": {},
        "immediate_repeat": {},
        "data_bug_id": "BUG",
    }
    summary = convergence_summary(manifest={"known_bug": "BUG"}, checkpoint_reports={"EP30": report}, sanity_reports={"EP30": {"metrics": {"total_steps": 1}, "action_distribution": {"click_element": 1}}})
    assert "structural_max_over_many_bias" in summary["root_cause_summary"]


def test_classify_convergence_structural_bias():
    causes = classify_convergence(
        {"EP30": {"click_q_mean": -0.05}},
        {"EP30": {"click_ratio": 1.0, "repeat_action_ratio": 0.9}},
        {"EP30": 0.25},
    )
    assert "structural_max_over_many_bias" in causes
    assert "persistent_click_max_selection_uplift" in causes
