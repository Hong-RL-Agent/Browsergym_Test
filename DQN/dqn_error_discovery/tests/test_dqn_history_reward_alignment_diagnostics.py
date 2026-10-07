import numpy as np

from dqn_error_discovery.agents.history_reward_alignment_diagnostics import (
    HISTORY_OFFSET,
    ablated_state,
    aggregate_ablation,
    conditional_reward_table,
    confirmation_confounding,
    escape_metric_interpretation,
    predictability_classification,
    rank_cycle_bias_features,
    reward_timing,
    summarize_predictability,
)
from dqn_error_discovery.runners.run_history_reward_alignment_diagnostic import _hypotheses, _root_cause


def test_temporal_ordering_predictability_classifies_visible_cycle_state():
    history = {"active_cycle_length": 1, "same_action_identity_streak": 2, "previous_unproductive_repeat": False}
    repeat = {"cycle_repeat": True}

    assert predictability_classification(history, repeat) == "PREDICTABLE_FROM_STATE"


def test_predictability_counts_all_classes():
    rows = [
        {"predictability": "PREDICTABLE_FROM_STATE"},
        {"predictability": "PARTIALLY_PREDICTABLE"},
        {"predictability": "NOT_PREDICTABLE_UNTIL_AFTER_ACTION"},
        {"predictability": "PREDICTABLE_FROM_STATE"},
    ]

    assert summarize_predictability(rows) == {
        "PREDICTABLE_FROM_STATE": 2,
        "PARTIALLY_PREDICTABLE": 1,
        "NOT_PREDICTABLE_UNTIL_AFTER_ACTION": 1,
    }


def test_reward_timing_separates_repeat_and_confirmation_components():
    timing = reward_timing(
        {
            "new_local_state": 0.2,
            "new_local_action_target": 0.4,
            "flow_progress": 0.3,
            "candidate_anomaly": 0.7,
            "confirmed_anomaly": 1.0,
            "confirmation_upgrade": 1.3,
            "repeat_action_penalty": -0.3,
            "repeat_state_penalty": -0.15,
            "no_effect_penalty": -0.4,
            "final_reward": 3.0,
        }
    )

    assert timing["novelty_reward"] == 0.6000000000000001
    assert timing["confirmation_reward"] == 1.0
    assert timing["upgrade_reward"] == 1.3
    assert timing["repeat_penalty"] == -0.44999999999999996


def test_feature_group_ablation_zeroes_only_history_span():
    state = np.ones(945, dtype=np.float32)
    ablated = ablated_state(state, spans=[(0, 76)])

    assert float(ablated[0]) == 1.0
    assert float(ablated[HISTORY_OFFSET]) == 0.0
    assert float(ablated[HISTORY_OFFSET + 75]) == 0.0
    assert float(ablated[HISTORY_OFFSET + 76]) == 1.0


def test_outcome_flag_ablation_can_zero_single_indexes():
    state = np.ones(945, dtype=np.float32)
    ablated = ablated_state(state, indexes=[216, 220, 224, 228])

    assert float(ablated[HISTORY_OFFSET + 216]) == 0.0
    assert float(ablated[HISTORY_OFFSET + 217]) == 1.0
    assert float(ablated[HISTORY_OFFSET + 228]) == 0.0


def test_aggregate_ablation_and_ranking_use_margin_magnitude():
    rows = [
        {"group": "cycle_onehot", "cycle_q_margin_delta": -2.0, "selected_action_flip": True},
        {"group": "outcome_flags", "cycle_q_margin_delta": 0.5, "selected_action_flip": False},
    ]

    summary = aggregate_ablation(rows)
    ranking = rank_cycle_bias_features(summary)

    assert summary["cycle_onehot"]["mean_abs_cycle_q_margin_delta"] == 2.0
    assert ranking[0]["group"] == "cycle_onehot"


def test_conditional_reward_aggregation_splits_continue_and_break():
    rows = [
        {
            "reward": -0.5,
            "confirmation_transition_events": [],
            "history_state_diagnostic": {
                "previous_unproductive_repeat": True,
                "active_cycle_length": 1,
                "selected_action_identity": "scroll_up",
                "cycle_continuation_identity": "scroll_up",
            },
        },
        {
            "reward": 1.0,
            "confirmation_transition_events": [{"newly_confirmed": True}],
            "history_state_diagnostic": {
                "previous_unproductive_repeat": True,
                "active_cycle_length": 1,
                "selected_action_identity": "noop",
                "cycle_continuation_identity": "scroll_up",
            },
        },
    ]

    table = conditional_reward_table(rows)

    assert table["previous_unproductive=1|continue_cycle"]["count"] == 1
    assert table["previous_unproductive=1|break_cycle"]["confirmation_rate"] == 1.0


def test_confirmation_confounding_compares_cycle_scroll_to_break():
    rows = [
        {
            "action_type": "scroll_up",
            "reward": -0.5,
            "confirmation_transition_events": [],
            "history_state_diagnostic": {
                "active_cycle_length": 1,
                "selected_action_identity": "scroll_up",
                "cycle_continuation_identity": "scroll_up",
            },
        },
        {
            "action_type": "noop",
            "reward": 2.0,
            "confirmation_transition_events": [{"newly_confirmed": True}],
            "history_state_diagnostic": {
                "active_cycle_length": 1,
                "selected_action_identity": "noop",
                "cycle_continuation_identity": "scroll_up",
            },
        },
    ]

    confounding = confirmation_confounding(rows)

    assert confounding["cycle_scroll"]["confirmation_rate"] == 0.0
    assert confounding["cycle_break"]["confirmation_rate"] == 1.0


def test_hypotheses_and_root_cause_include_bootstrap_and_feature_inversion():
    hypotheses = _hypotheses(
        q_rows=[{"q_margin": {}}],
        immediate={"cycle_continue": {"mean": -0.5}, "cycle_break": {"mean": 0.1}},
        confounding={"cycle_scroll": {"confirmation_rate": 0.0}, "cycle_break": {"confirmation_rate": 0.5}},
        td_cycle={"bootstrap_scroll_rate": 1.0, "mean_bootstrap": {"mean": 2.0}, "mean_td_target": {"mean": 2.5}},
        td_break={"mean_bootstrap": {"mean": 1.0}, "mean_td_target": {"mean": 1.5}},
        feature_summary={"cycle_onehot": {"mean_abs_cycle_q_margin_delta": 1.0}},
        outcome_summary={},
    )

    assert hypotheses["H4_bootstrap_self_reinforcement"]["verdict"] == "PASS"
    assert hypotheses["H5_feature_association_inversion"]["verdict"] == "PASS"
    assert _root_cause(hypotheses)["primary"] == "FEATURE_ASSOCIATION_INVERSION"


def test_escape_metric_interpretation_declares_not_a_metric_bug():
    result = escape_metric_interpretation(
        {
            "repeat_local_action_ratio": 0.9,
            "repeat_escape_rate": 1.0,
            "cycle_repeat_ratio": 0.75,
            "cycle_escape_rate": 0.714,
        }
    )

    assert result["repeat_and_escape_can_coexist"] is True
    assert result["metric_bug"] is False


def test_ground_truth_independence_invariant_shape():
    hypotheses = _hypotheses(
        q_rows=[],
        immediate={"cycle_continue": {"mean": -1.0}, "cycle_break": {"mean": 0.0}},
        confounding={"cycle_scroll": {"confirmation_rate": 0.0}, "cycle_break": {"confirmation_rate": 0.0}},
        td_cycle={"bootstrap_scroll_rate": 0.0, "mean_bootstrap": {"mean": 0.0}, "mean_td_target": {"mean": -1.0}},
        td_break={"mean_bootstrap": {"mean": 0.0}, "mean_td_target": {"mean": 0.0}},
        feature_summary={},
        outcome_summary={},
    )

    assert "ground_truth" not in str(hypotheses).lower()
