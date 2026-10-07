from dqn_error_discovery.runners.dqn_history_final_multiseed_validation import (
    NO_HISTORY_707_REFERENCE,
    classify_attractor,
    classify_overall,
    final_hard_invariants,
    mean_std,
)


def test_aggregate_mean_std_uses_population_std():
    stats = mean_std([0.0, 0.3, 0.6])
    assert stats["mean"] == 0.3
    assert round(stats["std"], 6) == 0.244949


def test_history_no_history_reference_comparison_values_are_available():
    assert NO_HISTORY_707_REFERENCE[43]["unproductive"] == 0.825
    history_mean = mean_std([0.0, 0.2, 0.4])["mean"]
    reference_mean = mean_std([NO_HISTORY_707_REFERENCE[seed]["unproductive"] for seed in (42, 43, 44)])["mean"]
    assert history_mean < reference_mean


def test_help_hurt_aggregate_values_can_be_averaged():
    stats = mean_std([0.0, 0.5, 1.0])
    assert stats["mean"] == 0.5
    assert stats["max"] == 1.0


def test_attractor_classification_detects_cycle_and_healthy_diverse():
    assert classify_attractor({"cycle_repeat": 0.8, "dominant_action": "scroll_down", "dominant_ratio": 0.3}) == "CYCLE_ATTRACTOR"
    assert classify_attractor({"cycle_repeat": 0.0, "dominant_action": "noop", "dominant_ratio": 0.6}) == "EARLY_FINISH_COLLAPSE"
    assert classify_attractor({"cycle_repeat": 0.0, "dominant_action": "click_element", "dominant_ratio": 0.2, "unique_action_types": 6, "action_entropy": 1.6}) == "HEALTHY_DIVERSE"


def test_freeze_eligibility_requires_all_hard_invariants():
    rows = [
        {
            "seed": 42,
            "training": {"nan_inf": False, "exploding_q": False},
            "evaluation": {
                "action_error": 0,
                "action_opportunities": 1,
                "action_executable": 1,
                "duplicate_leaked": 0,
                "confirmation_reward_events": 2,
                "newly_confirmed": 2,
            },
        }
    ]
    assert final_hard_invariants(rows)["pass"]
    rows[0]["evaluation"]["duplicate_leaked"] = 1
    assert not final_hard_invariants(rows)["pass"]


def test_overall_classification_stable_healthy_is_ground_truth_independent():
    rows = [
        {"seed": 42, "evaluation": {"cycle_repeat": 0.0, "dominant_ratio": 0.2, "ground_truth": True}},
        {"seed": 43, "evaluation": {"cycle_repeat": 0.1, "dominant_ratio": 0.3}},
        {"seed": 44, "evaluation": {"cycle_repeat": 0.2, "dominant_ratio": 0.4}},
    ]
    classes = {"42": "HEALTHY_DIVERSE", "43": "HEALTHY_DIVERSE", "44": "OTHER"}
    assert classify_overall(rows, classes) == "STABLE_HEALTHY"
