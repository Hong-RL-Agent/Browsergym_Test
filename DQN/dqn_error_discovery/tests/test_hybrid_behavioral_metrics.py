from __future__ import annotations

from dqn_error_discovery.runners.run_hybrid_behavioral_validation import (
    aggregate_validation,
    attribution_matrix,
    classify_seed,
    coverage_retention,
    normalize_confirmation_records,
    pending_confirmation_churn,
    ppo_candidate_dqn_opportunity_count,
    same_candidate_rehandoff,
    summarize_hybrid_artifact,
    summarize_ppo_only_artifact,
)


def _hybrid_summary() -> dict:
    return {
        "configuration": {"seed": 42, "episodes": 1},
        "episode_summary": {"total_steps": 6, "ppo_steps": 3, "dqn_steps": 3},
        "handoff_events": [
            {"step": 2, "local_session_id": 1, "reason": "NEW_CANDIDATE", "entry_candidate_keys": ["A"], "entry_pending_confirmation_keys": []},
            {"step": 5, "local_session_id": 2, "reason": "PENDING_CONFIRMATION", "entry_candidate_keys": ["A"], "entry_pending_confirmation_keys": ["A"]},
        ],
        "return_events": [
            {"step": 3, "local_session_id": 1, "reason": "NOVELTY_EXHAUSTED", "local_steps": 2},
            {"step": 6, "local_session_id": 2, "reason": "CONFIRMED", "local_steps": 1},
        ],
        "timeline": [
            {"step": 1, "active_agent": "PPO_GLOBAL", "action_type": "click_element", "reward_components": {"new_state": 0.5}, "new_candidate_keys": ["A"]},
            {"step": 2, "active_agent": "PPO_GLOBAL", "action_type": "click_element", "reward_components": {}, "new_candidate_keys": []},
            {"step": 3, "active_agent": "DQN_LOCAL", "local_session_id": 1, "action_type": "inspect_layout", "reward": 0.7, "reward_components": {"candidate_anomaly": 0.7}, "repeat_diagnostic": {}, "new_candidate_keys": ["B"]},
            {"step": 4, "active_agent": "PPO_GLOBAL", "action_type": "scroll_down", "reward_components": {}, "new_candidate_keys": []},
            {"step": 5, "active_agent": "DQN_LOCAL", "local_session_id": 2, "action_type": "click_element", "reward": 1.3, "reward_components": {"confirmation_upgrade": 1.3}, "repeat_diagnostic": {"recent_repeat": True}, "new_candidate_keys": []},
            {"step": 6, "active_agent": "DQN_LOCAL", "local_session_id": 2, "action_type": "noop", "reward": 0.0, "reward_components": {}, "repeat_diagnostic": {"cycle_repeat": True, "unproductive_repeat": True}, "new_candidate_keys": []},
        ],
        "candidate_attribution": {
            "unique_candidates": 2,
            "ppo_discovered_candidates": 1,
            "dqn_discovered_candidates": 1,
            "candidate_first_agent": {"A": "PPO", "B": "DQN"},
        },
        "confirmation_attribution": {
            "unique_confirmed": 1,
            "confirmation_records": [{"logical_key": "A", "discovered_by": "PPO", "confirmed_by": "DQN", "local_session_id": 2}],
            "duplicate_reward_leak": 0,
        },
        "ppo_global_coverage": {"unique_states": 3, "unique_urls": 1, "unique_targets": 2, "unique_action_targets": 3, "unique_action_types": 2, "meaningful_state_transitions": 1},
        "hard_invariants": {
            "all_pass": True,
            "action_error_zero": True,
            "duplicate_reward_leak_zero": True,
            "ground_truth_used": False,
            "zero_step_switch_loop": False,
            "local_budget_respected": True,
            "separate_buffers": True,
            "separate_reward_functions": True,
        },
    }


def test_productive_session_classification() -> None:
    row = summarize_hybrid_artifact(_hybrid_summary())
    assert row["productive_sessions"] == 2
    assert row["unproductive_sessions"] == 0
    assert row["productive_local_session_rate"] == 1.0


def test_same_candidate_rehandoff() -> None:
    handoffs = _hybrid_summary()["handoff_events"]
    count, distances = same_candidate_rehandoff(handoffs, _hybrid_summary()["timeline"])
    assert count == 1
    assert distances == [1]


def test_pending_churn_excludes_confirmed_candidate() -> None:
    handoffs = _hybrid_summary()["handoff_events"]
    assert pending_confirmation_churn(handoffs, _hybrid_summary()["confirmation_attribution"]) == 0


def test_pending_churn_counts_unresolved_reentry() -> None:
    summary = _hybrid_summary()
    summary["confirmation_attribution"]["confirmation_records"] = []
    assert pending_confirmation_churn(summary["handoff_events"], summary["confirmation_attribution"]) == 0
    summary["handoff_events"].append({"step": 7, "local_session_id": 3, "reason": "PENDING_CONFIRMATION", "entry_candidate_keys": ["A"], "entry_pending_confirmation_keys": ["A"]})
    assert pending_confirmation_churn(summary["handoff_events"], summary["confirmation_attribution"]) == 1


def test_attribution_matrix_and_opportunity_rate() -> None:
    summary = _hybrid_summary()
    matrix = attribution_matrix(summary["confirmation_attribution"]["confirmation_records"])
    assert matrix["PPO->DQN"] == 1
    assert ppo_candidate_dqn_opportunity_count(summary["candidate_attribution"], summary["handoff_events"]) == 1


def test_confirmation_record_normalization_matches_rule_target() -> None:
    records = [{"logical_key": "R06|/|layout-overflow|target:11", "discovered_by": "UNKNOWN", "confirmed_by": "DQN"}]
    first_agent = {"R06|layout-overflow|layout-overflow|11|desktop": "PPO"}
    normalized = normalize_confirmation_records(records, first_agent)
    assert normalized[0]["discovered_by"] == "PPO"


def test_ppo_only_summary_and_coverage_retention() -> None:
    ppo = summarize_ppo_only_artifact({"metrics": {"total_steps": 6, "unique_states": 6, "unique_urls": 1, "unique_targets": 4, "unique_action_targets": 6, "unique_candidate_signatures": 1, "unique_confirmed_signatures": 0, "action_error_count": 0}, "reward_component_totals": {"final_reward": 3.0}})
    hybrid = summarize_hybrid_artifact(_hybrid_summary())
    retention = coverage_retention(hybrid, ppo)
    assert retention["state_coverage_retention"] == 0.5
    assert retention["target_coverage_retention"] == 0.5


def test_session_yield_and_dqn_repeat() -> None:
    row = summarize_hybrid_artifact(_hybrid_summary())
    assert row["efficiency"]["confirmation_yield_per_local_session"] == 0.5
    assert row["dqn_repeat"]["recent"] == 1 / 3
    assert row["dqn_repeat"]["cycle"] == 1 / 3
    assert row["dqn_repeat"]["unproductive"] == 1 / 3


def test_ground_truth_independence_affects_classification() -> None:
    row = summarize_hybrid_artifact(_hybrid_summary())
    assert classify_seed(row, {"state_coverage_retention": 1.0}) == "HEALTHY_HYBRID"
    row["hard_invariants"]["ground_truth_used"] = True
    assert classify_seed(row, {"state_coverage_retention": 1.0}) == "OTHER"


def test_aggregate_validation_passes_healthy_seed() -> None:
    summary = _hybrid_summary()
    aggregate = aggregate_validation({42: summary}, {42: {"metrics": {"total_steps": 6, "unique_states": 3, "unique_urls": 1, "unique_targets": 2, "unique_action_targets": 3, "unique_candidate_signatures": 1, "unique_confirmed_signatures": 0}}})
    assert aggregate["seed_classification"][42] == "HEALTHY_HYBRID"
    assert aggregate["HYBRID_BEHAVIOR_CHECK"] == "PASS"
