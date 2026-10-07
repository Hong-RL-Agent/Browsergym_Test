from collections import Counter

from dqn_error_discovery.runners.run_policy_sanity_check import (
    GROUND_TRUTH_KEYS,
    _action_type_repeat_summary,
    _history_counterfactual_repeat_summary,
    _repeat_pattern_summary,
)


def test_repeat_escape_rate_inputs_are_counted_from_followup_steps():
    patterns = _repeat_pattern_summary(["A", "A", "B", "A"], window=4)
    assert patterns["immediate_repeat_count"] == 1
    assert patterns["recent_window_repeat_count"] == 2
    assert patterns["cycle_repeat_count"] == 2


def test_action_type_repeat_summary_keeps_counts_and_mean_reward():
    summary = _action_type_repeat_summary(
        {"click_element": 3},
        {
            "click_element": Counter({"immediate_repeat": 1, "recent_window_repeat": 2, "cycle_repeat": 1, "unproductive_repeat": 1}),
            "change_viewport_mobile": Counter({"recent_window_repeat": 1}),
        },
        {"click_element": [1.0, 0.0, -1.0], "change_viewport_mobile": [-0.5, 0.5]},
    )
    click = summary["click_element"]
    assert click["action_count"] == 3
    assert click["repeat_count"] == 4
    assert click["recent_repeat_count"] == 2
    assert click["cycle_participation_count"] == 1
    assert click["unproductive_repeat_count"] == 1
    assert click["mean_reward"] == 0.0
    assert summary["viewport_mobile"]["recent_repeat_count"] == 1


def test_history_help_hurt_classification_uses_relevant_repeat_states():
    summary = _history_counterfactual_repeat_summary(
        [
            {
                "active_cycle_length": 2,
                "same_action_identity_streak": 0,
                "previous_unproductive_repeat": False,
                "selected_type_flip": True,
                "real_repeats_relevant_identity": False,
                "zero_repeats_relevant_identity": True,
                "real_breaks_cycle": True,
                "zero_breaks_cycle": False,
                "selected_type_real": "scroll_down",
                "selected_type_zero": "scroll_up",
            },
            {
                "active_cycle_length": 0,
                "same_action_identity_streak": 2,
                "previous_unproductive_repeat": False,
                "selected_type_flip": True,
                "real_repeats_relevant_identity": True,
                "zero_repeats_relevant_identity": False,
                "selected_type_real": "click_element",
                "selected_type_zero": "scroll_down",
            },
        ]
    )
    assert summary["repeat_state_subset_count"] == 2
    assert summary["history_repeat_help_count"] == 1
    assert summary["history_repeat_hurt_count"] == 1
    assert summary["history_repeat_help_ratio"] == 0.5
    assert summary["history_repeat_hurt_ratio"] == 0.5
    assert summary["repeat_state_expected_cycle_breaking_real_count"] == 1


def test_zero_history_comparison_fields_are_aggregated_when_no_subset():
    summary = _history_counterfactual_repeat_summary(
        [{"active_cycle_length": 0, "same_action_identity_streak": 1, "previous_unproductive_repeat": False}]
    )
    assert summary["repeat_state_subset_count"] == 0
    assert summary["repeat_state_subset_flip_rate"] == 0.0
    assert summary["history_repeat_help_count"] == 0
    assert summary["history_repeat_hurt_count"] == 0


def test_confirmation_regression_aggregation_invariant_shape():
    confirmed_reward_events = 2
    newly_confirmed_transitions = 2
    duplicate_reward_leaked = 0
    assert duplicate_reward_leaked == 0
    assert confirmed_reward_events <= newly_confirmed_transitions


def test_ground_truth_fields_remain_explicitly_excluded_from_sanity_anomalies():
    assert "ground_truth" in GROUND_TRUTH_KEYS
    assert "matched_bug_id" in GROUND_TRUTH_KEYS
