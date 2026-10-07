from __future__ import annotations

from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.run_hybrid_behavioral_validation import summarize_hybrid_artifact


def obs(functional: bool = False) -> dict:
    candidates = [{"bid": "28", "role": "searchbox", "tag": "input", "is_form_field": True, "text": "search"}] if functional else []
    return {"page_state": {"url": "http://site.test/", "title": "Site"}, "candidate_elements": candidates}


def pending_meta(
    key: str,
    *,
    rule_id: str = "R05",
    mode: str = "OBSERVATION_REPRODUCTION",
    progress: float = 0.0,
) -> dict:
    return {
        "pending_confirmation_exists": True,
        "confirmation_mode": mode,
        "active_pending_key": key,
        "active_rule_id": rule_id,
        "expected_action_type": "click_element" if mode == "ACTION_REPRODUCTION" else "",
        "expected_target_identity": "bid:28" if mode == "ACTION_REPRODUCTION" else "",
        "expected_action_requires_target": mode == "ACTION_REPRODUCTION",
        "expected_target_visible": True,
        "confirmation_progress": progress,
    }


def history_with_pending(key: str, *, confirmed: bool = False, seen: set[str] | None = None) -> dict:
    pending = {} if confirmed else {key: {"logical_key": key, "rule_id": key.split("|", 1)[0]}}
    confirmed_keys = {key} if confirmed else set()
    return {
        "pending_reproduction_records": pending,
        "rewarded_confirmed_logical_keys": confirmed_keys,
        "seen_confirmed_signatures": set(),
        "seen_candidate_signatures": set(seen or ()),
        "seen_reward_candidate_keys": set(seen or ()),
    }


def pending_handoff(controller: HybridController, key: str, *, rule_id: str = "R05") -> None:
    raw_key = f"{rule_id}|layout-overlap|layout-overlap|28|desktop"
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    reason = controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(),
        anomalies=[{"logical_key": raw_key, "status": "candidate"}],
        reward_components={},
        confirmation_metadata=pending_meta(key, rule_id=rule_id),
        history=history_with_pending(key, seen={raw_key}),
    )
    assert reason == "PENDING_CONFIRMATION"


def test_tracker_candidate_canonical_key_available_and_stored() -> None:
    key = "R05|/|layout-overlap|target:28"
    controller = HybridController()
    pending_handoff(controller, key)
    assert controller.local_session is not None
    assert controller.local_session.trigger_candidate_key == key
    assert controller.handoff_events[-1]["trigger_candidate_key"] == key
    assert controller.handoff_events[-1]["entry_anomaly_keys"] != controller.handoff_events[-1]["entry_tracker_candidate_keys"]


def test_r01_r05_r06_canonical_confirmation_events_return_confirmed() -> None:
    cases = [
        ("R01|/|button-no-response|target:28", "R01", "ACTION_REPRODUCTION"),
        ("R05|/|layout-overlap|target:28", "R05", "OBSERVATION_REPRODUCTION"),
        ("R06|/|layout-overflow|target:28", "R06", "OBSERVATION_REPRODUCTION"),
    ]
    for key, rule_id, mode in cases:
        controller = HybridController(max_local_steps=1, novelty_patience=1)
        controller.record_agent_step(AgentMode.PPO_GLOBAL)
        raw_key = f"{rule_id}|raw-key|28|desktop"
        assert controller.maybe_handoff_after_ppo_step(
            step=1,
            observation=obs(),
            anomalies=[{"logical_key": raw_key, "status": "candidate"}],
            reward_components={},
            confirmation_metadata=pending_meta(key, rule_id=rule_id, mode=mode),
            history=history_with_pending(key, seen={raw_key}),
        ) == "PENDING_CONFIRMATION"
        returned = controller.maybe_return_after_dqn_step(
            step=2,
            observation=obs(),
            anomalies=[],
            reward_components={},
            repeat_diagnostic={},
            done=False,
            confirmation_events=[{"logical_key": key, "rule_id": rule_id, "current_status": "confirmed"}],
            tracker_history=history_with_pending(key, confirmed=True),
            confirmation_metadata=pending_meta(key, rule_id=rule_id, mode=mode),
        )
        assert returned == "CONFIRMED"
        assert controller.return_events[-1]["resolved_candidate_key"] == key


def test_tracker_status_confirmed_recognized_without_same_step_event() -> None:
    key = "R05|/|layout-overlap|target:28"
    controller = HybridController(max_local_steps=1)
    pending_handoff(controller, key)
    assert controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        tracker_history=history_with_pending(key, confirmed=True),
        confirmation_metadata=pending_meta(key),
    ) == "CONFIRMED"


def test_other_candidate_confirmed_does_not_resolve_trigger_candidate() -> None:
    trigger = "R05|/|layout-overlap|target:28"
    other = "R06|/|layout-overflow|target:11"
    controller = HybridController(max_local_steps=1)
    pending_handoff(controller, trigger)
    assert controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": other, "current_status": "confirmed"}],
        tracker_history=history_with_pending(trigger),
        confirmation_metadata=pending_meta(trigger),
    ) == "LOCAL_BUDGET"


def test_unrelated_confirmed_entry_anomaly_does_not_end_active_candidate_session() -> None:
    trigger = "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points"
    other = "R06|/|layout-overflow|target:98"
    controller = HybridController(max_local_steps=2, novelty_patience=3)
    pending_handoff(controller, trigger, rule_id="R10")

    returned = controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(),
        anomalies=[{"logical_key": other, "status": "confirmed", "rule_id": "R06"}],
        reward_components={"confirmed_anomaly": 2.0},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": other, "rule_id": "R06", "current_status": "confirmed"}],
        tracker_history=history_with_pending(trigger),
        confirmation_metadata=pending_meta(trigger, rule_id="R10"),
    )

    assert returned == ""
    assert controller.active_agent == AgentMode.DQN_LOCAL
    assert controller.local_session is not None
    assert controller.local_session.trigger_candidate_key == trigger
    assert controller.unrelated_confirmation_events[-1]["unrelated_confirmed_keys"] == [other]
    assert controller.unrelated_confirmation_events[-1]["caused_return"] is False


def test_unrelated_same_category_confirmed_key_does_not_end_active_session() -> None:
    trigger = "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points"
    other = "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/user/state|tier"
    controller = HybridController(max_local_steps=2, novelty_patience=3)
    pending_handoff(controller, trigger, rule_id="R10")

    returned = controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(),
        anomalies=[{"logical_key": other, "status": "confirmed", "rule_id": "R10"}],
        reward_components={"confirmed_anomaly": 2.0},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": other, "rule_id": "R10", "current_status": "confirmed"}],
        tracker_history=history_with_pending(trigger),
        confirmation_metadata=pending_meta(trigger, rule_id="R10"),
    )

    assert returned == ""
    assert controller.active_agent == AgentMode.DQN_LOCAL
    assert controller.unrelated_confirmation_events[-1]["unrelated_confirmed_keys"] == [other]


def test_confirmed_priority_beats_novelty_exhausted_and_local_budget() -> None:
    key = "R06|/|layout-overflow|target:28"
    controller = HybridController(max_local_steps=1, novelty_patience=1)
    pending_handoff(controller, key, rule_id="R06")
    returned = controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": key, "current_status": "confirmed"}],
        tracker_history=history_with_pending(key, confirmed=True),
        confirmation_metadata=pending_meta(key, rule_id="R06"),
    )
    assert returned == "CONFIRMED"
    assert controller.return_events[-1]["reason"] == "CONFIRMED"


def test_generic_functional_session_behavior_unchanged() -> None:
    controller = HybridController(max_local_steps=1)
    assert controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(functional=True),
        anomalies=[],
        reward_components={},
        confirmation_metadata={},
        history={},
    ) == "FUNCTIONAL_REGION"
    assert controller.local_session is not None
    assert controller.local_session.trigger_candidate_key == ""
    assert controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(functional=True),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": "R05|/|layout-overlap|target:28", "current_status": "confirmed"}],
        tracker_history=history_with_pending("R05|/|layout-overlap|target:28", confirmed=True),
    ) == "LOCAL_BUDGET"


def test_pending_any_and_trigger_confirmed_metrics_are_split() -> None:
    summary = {
        "configuration": {"seed": 42, "episodes": 1},
        "episode_summary": {"total_steps": 2, "ppo_steps": 1, "dqn_steps": 1},
        "handoff_events": [
            {"step": 1, "local_session_id": 1, "reason": "PENDING_CONFIRMATION", "trigger_candidate_key": "A", "entry_candidate_keys": [], "entry_pending_confirmation_keys": ["A"]}
        ],
        "return_events": [{"step": 2, "local_session_id": 1, "reason": "LOCAL_BUDGET"}],
        "timeline": [{"step": 2, "active_agent": "DQN_LOCAL", "local_session_id": 1, "reward_components": {}, "repeat_diagnostic": {}, "new_candidate_keys": []}],
        "candidate_attribution": {"candidate_first_agent": {}},
        "confirmation_attribution": {"unique_confirmed": 1, "confirmation_records": [{"logical_key": "B", "confirmed_by": "DQN", "local_session_id": 1}], "duplicate_reward_leak": 0},
        "ppo_global_coverage": {},
        "hard_invariants": {"all_pass": True, "ground_truth_used": False, "zero_step_switch_loop": False, "local_budget_respected": True, "separate_buffers": True, "separate_reward_functions": True},
    }
    row = summarize_hybrid_artifact(summary)
    assert row["pending_any_candidate_confirmed_sessions"] == 1
    assert row["pending_trigger_candidate_confirmed_sessions"] == 0
    assert row["pending_handoff_confirmed_count"] == 0


def test_confirmed_trigger_candidate_removed_from_pending_handoff_eligibility() -> None:
    key = "R05|/|layout-overlap|target:28"
    controller = HybridController()
    pending_handoff(controller, key)
    assert controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": key, "current_status": "confirmed"}],
        tracker_history=history_with_pending(key, confirmed=True),
        confirmation_metadata=pending_meta(key),
    ) == "CONFIRMED"
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    assert controller.maybe_handoff_after_ppo_step(
        step=3,
        observation=obs(),
        anomalies=[],
        reward_components={},
        confirmation_metadata=pending_meta(key),
        history=history_with_pending(key),
    ) == ""


def test_raw_anomaly_keys_still_available() -> None:
    key = "R05|/|layout-overlap|target:28"
    controller = HybridController()
    pending_handoff(controller, key)
    event = controller.handoff_events[-1]
    assert "entry_candidate_keys" in event
    assert "entry_anomaly_keys" in event
    assert "entry_tracker_candidate_keys" in event


def test_reward_semantics_and_ground_truth_independent() -> None:
    names = set(HybridController.maybe_return_after_dqn_step.__code__.co_names)
    assert "DQN_REWARD_CONFIG" not in names
    assert not names.intersection({"ground_truth", "known_bug", "injected", "expected_defect"})
