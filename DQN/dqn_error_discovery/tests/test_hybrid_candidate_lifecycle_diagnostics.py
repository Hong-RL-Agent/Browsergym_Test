from __future__ import annotations

from dqn_error_discovery.hybrid.candidate_lifecycle_diagnostics import (
    analyze_candidate_lifecycle,
    confirmation_match_class,
    confirmed_return_definition,
    key_match,
    novelty_exhausted_definition,
    return_priority_order,
)


def _summary() -> dict:
    return {
        "configuration": {"episodes": 1, "max_steps": 60},
        "handoff_events": [
            {
                "step": 1,
                "local_session_id": 1,
                "reason": "PENDING_CONFIRMATION",
                "entry_candidate_keys": ["R05|layout-overlap|layout-overlap|18|10"],
                "entry_pending_confirmation_keys": [],
                "trigger_candidate_key": "click_element:bid:10",
                "entry_context_fingerprint": "/|functional_target|||||||||||ACTION_REPRODUCTION||click_element|bid:10|1",
                "entry_target_resolvable": True,
                "entry_confirmation_progress": 0,
            },
            {
                "step": 7,
                "local_session_id": 2,
                "reason": "PENDING_CONFIRMATION",
                "entry_candidate_keys": ["R05|layout-overlap|layout-overlap|18|10"],
                "entry_pending_confirmation_keys": [],
                "trigger_candidate_key": "click_element:bid:10",
                "entry_context_fingerprint": "/|functional_target|||||||||||ACTION_REPRODUCTION||click_element|bid:10|1",
                "entry_target_resolvable": True,
                "entry_confirmation_progress": 0,
            },
            {
                "step": 12,
                "local_session_id": 3,
                "reason": "PENDING_CONFIRMATION",
                "entry_candidate_keys": ["R06|layout-overflow|layout-overflow|11|desktop"],
                "entry_pending_confirmation_keys": [],
                "trigger_candidate_key": "R06|/|layout-overflow|target:11",
                "entry_context_fingerprint": "/|functional_target|||||||||||OBSERVATION_REPRODUCTION|R06|||0",
                "entry_target_resolvable": True,
                "entry_confirmation_progress": 0,
            },
        ],
        "return_events": [
            {"step": 3, "local_session_id": 1, "reason": "NOVELTY_EXHAUSTED", "local_steps": 2, "session_productive": False, "session_confirmation_progress": False},
            {"step": 9, "local_session_id": 2, "reason": "NOVELTY_EXHAUSTED", "local_steps": 2, "session_productive": True, "session_confirmation_progress": True},
            {"step": 14, "local_session_id": 3, "reason": "NOVELTY_EXHAUSTED", "local_steps": 2, "session_productive": True, "session_confirmation_progress": True},
        ],
        "timeline": [
            {"episode": 1, "step": 2, "active_agent": "DQN_LOCAL", "local_session_id": 1, "action_type": "inspect_layout", "reward": -0.3, "reward_components": {}, "pending_count": 1, "newly_confirmed": []},
            {"episode": 1, "step": 3, "active_agent": "DQN_LOCAL", "local_session_id": 1, "action_type": "inspect_console", "reward": -0.3, "reward_components": {}, "pending_count": 1, "newly_confirmed": []},
            {"episode": 1, "step": 8, "active_agent": "DQN_LOCAL", "local_session_id": 2, "action_type": "inspect_layout", "reward": 1.3, "reward_components": {"confirmation_upgrade": 1.3}, "pending_count": 0, "newly_confirmed": ["R05|/|layout-overlap|target:10"]},
            {"episode": 1, "step": 9, "active_agent": "DQN_LOCAL", "local_session_id": 2, "action_type": "inspect_console", "reward": -0.3, "reward_components": {}, "pending_count": 0, "newly_confirmed": ["R05|/|layout-overlap|target:10"]},
            {"episode": 1, "step": 13, "active_agent": "DQN_LOCAL", "local_session_id": 3, "action_type": "inspect_layout", "reward": 1.3, "reward_components": {"confirmation_upgrade": 1.3}, "pending_count": 0, "newly_confirmed": ["R06|/|layout-overflow|target:11"]},
        ],
        "confirmation_attribution": {
            "confirmation_records": [
                {"logical_key": "R05|/|layout-overlap|target:10", "confirmed_by": "DQN", "local_session_id": 2},
                {"logical_key": "R06|/|layout-overflow|target:11", "confirmed_by": "DQN", "local_session_id": 3},
            ]
        },
        "pending_handoff_guard": {
            "pending_handoff_suppressed_count": 3,
            "suppression_events": [
                {"step": 4, "logical_key": "click_element:bid:10", "reasons": ["SAME_CANDIDATE_COOLDOWN"]},
                {"step": 4, "logical_key": "click_element:bid:10", "reasons": ["FUNCTIONAL_BYPASS_BLOCKED"], "bypass_trigger": "FUNCTIONAL_REGION"},
            ],
            "rehandoff_due_to_progress": 0,
            "rehandoff_due_to_context_change": 0,
            "rehandoff_due_to_target_resolvable": 0,
        },
    }


def test_return_priority_audit() -> None:
    assert return_priority_order()[:2] == ["EPISODE_DONE", "CONFIRMED"]
    assert "NOVELTY_EXHAUSTED" in return_priority_order()[4]


def test_novelty_and_confirmed_return_definitions() -> None:
    assert "novelty_miss_streak" in novelty_exhausted_definition()
    assert "entry_candidate_keys" in confirmed_return_definition()


def test_key_matching_exact_and_related() -> None:
    assert key_match("A", "A").exact
    assert key_match("R06|/|layout-overflow|target:11", "R06|layout-overflow|layout-overflow|11|desktop").related
    assert not key_match("R05|target:10", "R06|target:10").related


def test_confirmation_key_matching_classification() -> None:
    assert confirmation_match_class("A", ["A"]) == "TRIGGER_CANDIDATE_CONFIRMED"
    assert confirmation_match_class("R06|/|layout-overflow|target:11", ["R06|layout-overflow|layout-overflow|11|desktop"]) == "TRIGGER_CANDIDATE_CONFIRMED_RELATED"
    assert confirmation_match_class("A", ["B"]) == "OTHER_CANDIDATE_CONFIRMED"


def test_pending_confirmed_metric_scope_and_confirmed_return_mismatch() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["pending_confirmed_metric_definition"].startswith("session-level")
    assert diagnostic["pending_confirmed_sessions"] == 2
    assert diagnostic["confirmed_return_count"] == 0
    assert diagnostic["trigger_confirmed_but_return_not_confirmed_count"] == 1


def test_trigger_vs_other_candidate_confirmation_counts() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["trigger_candidate_confirmed_sessions_related"] == 1
    assert diagnostic["other_candidate_confirmed_sessions"] == 1


def test_progress_context_resolvability_snapshot_timing() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    override = diagnostic["override_audit"]
    assert override["actual_progress_changes"] == 0
    assert override["actual_context_changes"] == 0
    assert override["actual_resolvability_changes"] == 0


def test_cooldown_only_and_stale_pending_classification() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["cooldown_only_rehandoff"]["count"] == 1
    assert diagnostic["stale_pending"]["stale_pending_rehandoffs"] == 1


def test_candidate_timeline_and_rotation_detection() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["top_rehandoff_candidate_timelines"][0]["logical_key"] == "click_element:bid:10"
    assert diagnostic["pending_candidate_rotation_count"] == 0


def test_confirmed_candidate_removed_from_pending_audit() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["confirmed_candidate_rehandoff_count"] == 0
    assert diagnostic["tracker_pending_removal_normal"] is True


def test_suppression_denominator_semantics() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["suppression_denominator"]["suppression_count"] == 2
    assert diagnostic["suppression_denominator"]["unique_candidate_step_suppressions"] == 1
    assert diagnostic["metric_inflation"]["suppression_is_candidate_evaluation_event"] is True


def test_churn_and_rehandoff_metric_semantics_present() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert "session-level" in diagnostic["churn_metric_definition"]
    assert "session-pair" in diagnostic["same_candidate_rehandoff_metric_definition"]


def test_root_cause_verdicts_include_scope_mismatch_and_stale_pending() -> None:
    diagnostic = analyze_candidate_lifecycle(_summary())
    assert diagnostic["root_cause_verdicts"]["B_METRIC_SCOPE_MISMATCH"] == "PASS"
    assert diagnostic["root_cause_verdicts"]["C_STALE_PENDING_REHANDOFF"] == "PASS"


def test_ground_truth_independence() -> None:
    names = set(analyze_candidate_lifecycle.__code__.co_names)
    assert not names.intersection({"ground_truth", "known_bug", "injected", "expected_defect"})
