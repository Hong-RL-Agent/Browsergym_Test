from __future__ import annotations

from dqn_error_discovery.hybrid.final_behavioral_metrics import (
    build_final_behavioral_summary,
    classify_final_seed,
    final_churn_breakdown,
    final_freeze_manifest,
)


def _summary() -> dict:
    return {
        "configuration": {"seed": 42, "episodes": 2, "max_steps": 10},
        "episode_summary": {"total_steps": 20, "ppo_steps": 10, "dqn_steps": 10, "handoff_count": 3, "return_count": 3},
        "handoff_events": [
            {"step": 2, "local_session_id": 1, "reason": "PENDING_CONFIRMATION", "trigger_candidate_key": "A", "entry_context_fingerprint": "X", "entry_confirmation_progress": 0, "entry_target_resolvable": True, "entry_reproduction_requirement_fingerprint": "r", "eligibility_reason": "FIRST_HANDOFF", "eligibility_reasons": []},
            {"step": 6, "local_session_id": 2, "reason": "PENDING_CONFIRMATION", "trigger_candidate_key": "A", "entry_context_fingerprint": "Y", "entry_confirmation_progress": 0, "entry_target_resolvable": True, "entry_reproduction_requirement_fingerprint": "r", "eligibility_reason": "CONTEXT_CHANGED", "eligibility_reasons": ["CONTEXT_CHANGED"]},
            {"step": 12, "local_session_id": 3, "reason": "PENDING_CONFIRMATION", "trigger_candidate_key": "A", "entry_context_fingerprint": "X", "entry_confirmation_progress": 0, "entry_target_resolvable": True, "entry_reproduction_requirement_fingerprint": "r", "eligibility_reason": "FIRST_HANDOFF", "eligibility_reasons": []},
        ],
        "return_events": [
            {"step": 3, "local_session_id": 1, "reason": "NOVELTY_EXHAUSTED", "local_steps": 1},
            {"step": 7, "local_session_id": 2, "reason": "CONFIRMED", "local_steps": 1, "resolved_candidate_key": "A"},
            {"step": 13, "local_session_id": 3, "reason": "NOVELTY_EXHAUSTED", "local_steps": 1},
        ],
        "timeline": [
            {"step": 1, "active_agent": "PPO_GLOBAL", "action_type": "click_element"},
            {"step": 2, "active_agent": "DQN_LOCAL", "local_session_id": 1, "action_type": "inspect_layout", "reward": 0.0, "reward_components": {}, "repeat_diagnostic": {}},
            {"step": 5, "active_agent": "PPO_GLOBAL", "action_type": "click_element"},
            {"step": 6, "active_agent": "DQN_LOCAL", "local_session_id": 2, "action_type": "click_element", "reward": 1.0, "reward_components": {"confirmed_anomaly": 1.0}, "repeat_diagnostic": {}},
            {"step": 11, "active_agent": "PPO_GLOBAL", "action_type": "click_element"},
            {"step": 12, "active_agent": "DQN_LOCAL", "local_session_id": 3, "action_type": "inspect_layout", "reward": 0.5, "reward_components": {"new_local_state": 0.5}, "repeat_diagnostic": {}},
        ],
        "candidate_attribution": {"unique_candidates": 1, "ppo_discovered_candidates": 1, "dqn_discovered_candidates": 0, "candidate_first_agent": {"A": "PPO"}},
        "confirmation_attribution": {"unique_confirmed": 1, "confirmation_records": [{"logical_key": "A", "discovered_by": "PPO", "confirmed_by": "DQN", "local_session_id": 2}], "duplicate_reward_leak": 0},
        "ppo_global_coverage": {"unique_states": 5, "unique_urls": 1, "unique_targets": 4, "unique_action_targets": 5, "unique_action_types": 2, "meaningful_state_transitions": 3},
        "pending_handoff_guard": {"stale_session_rehandoff_prevented": 4, "stale_pending_evaluations": 8},
        "hard_invariants": {"all_pass": True, "ground_truth_used": False, "zero_step_switch_loop": False, "action_error_zero": True, "duplicate_reward_leak_zero": True, "local_budget_respected": True, "separate_buffers": True, "separate_reward_functions": True},
    }


def _ppo() -> dict:
    return {"metrics": {"total_steps": 20, "unique_states": 6, "unique_urls": 1, "unique_targets": 4, "unique_action_targets": 6, "unique_candidate_signatures": 1, "unique_confirmed_signatures": 0}, "reward_component_totals": {"final_reward": 0.0}}


def test_cross_episode_recurrence_excluded_from_pathological_churn() -> None:
    churn = final_churn_breakdown(_summary())
    assert churn["within_episode_same_candidate_rehandoff"] == 1
    assert churn["cross_episode_same_logical_key_recurrence"] == 1
    assert churn["meaningful_rehandoff"] == 1
    assert churn["pathological_churn_sessions"] == 0


def test_stale_same_episode_rehandoff_is_pathological() -> None:
    summary = _summary()
    summary["handoff_events"][1]["entry_context_fingerprint"] = "X"
    summary["handoff_events"][1]["eligibility_reason"] = ""
    summary["handoff_events"][1]["eligibility_reasons"] = []
    churn = final_churn_breakdown(summary)
    assert churn["stale_rehandoff"] == 1
    assert churn["cooldown_only_rehandoff"] == 1
    assert churn["pathological_churn_sessions"] == 1


def test_exact_vs_related_candidate_distinction() -> None:
    summary = _summary()
    summary["handoff_events"].append({"step": 8, "local_session_id": 4, "reason": "PENDING_CONFIRMATION", "trigger_candidate_key": "R05|/|layout-overlap|target:28", "entry_context_fingerprint": "Z", "entry_confirmation_progress": 0, "entry_target_resolvable": True, "entry_reproduction_requirement_fingerprint": "r", "eligibility_reason": "FIRST_HANDOFF", "eligibility_reasons": []})
    summary["handoff_events"].append({"step": 9, "local_session_id": 5, "reason": "PENDING_CONFIRMATION", "trigger_candidate_key": "R05|layout-overlap|layout-overlap|28|desktop", "entry_context_fingerprint": "Z", "entry_confirmation_progress": 0, "entry_target_resolvable": True, "entry_reproduction_requirement_fingerprint": "r", "eligibility_reason": "FIRST_HANDOFF", "eligibility_reasons": []})
    churn = final_churn_breakdown(summary)
    assert churn["related_candidate_rehandoff"] >= 1
    assert churn["exact_canonical_candidate_rehandoff"] == 1


def test_final_seed_classification_uses_pathological_churn_not_raw_churn() -> None:
    summary = _summary()
    churn = final_churn_breakdown(summary)
    hybrid = build_final_behavioral_summary({42: summary}, {42: _ppo()})["seed_summaries"]["42"]["hybrid"]
    assert classify_final_seed(hybrid, {"state_coverage_retention": 0.8, "target_coverage_retention": 1.0}, churn) == "HEALTHY_HYBRID"


def test_final_freeze_eligibility_and_manifest() -> None:
    result = build_final_behavioral_summary({42: _summary()}, {42: _ppo()})
    assert result["overall_classification"] == "HYBRID_V1_STABLE"
    assert result["HYBRID_BEHAVIOR_CHECK"] == "FINAL PASS"
    manifest = final_freeze_manifest(result)
    assert manifest["pending_rehandoff"] == "meaningful-change-only"
    assert manifest["cooldown_only_rehandoff"] == "disabled"


def test_ground_truth_blocks_final_pass() -> None:
    summary = _summary()
    summary["hard_invariants"]["ground_truth_used"] = True
    result = build_final_behavioral_summary({42: summary}, {42: _ppo()})
    assert result["HYBRID_CONTROLLER_CHECK"] == "REVIEW"
