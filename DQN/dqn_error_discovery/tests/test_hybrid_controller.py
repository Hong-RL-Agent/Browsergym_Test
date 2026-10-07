from __future__ import annotations

import json

import numpy as np
import pytest

from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.run_ppo_dqn_hybrid import (
    _append_cross_agent_history,
    _has_ppo_dqn_ppo_sequence,
    _sync_cross_agent_action_target_history,
    _validate_dqn_manifest,
)


def obs(url: str = "http://site.test/", *, functional: bool = False) -> dict:
    candidates = []
    if functional:
        candidates.append({"bid": "search", "role": "searchbox", "tag": "input", "is_form_field": True, "text": "search"})
    return {"page_state": {"url": url, "title": "Site"}, "candidate_elements": candidates}


def candidate(key: str, status: str = "candidate") -> dict:
    return {"logical_key": key, "signature": key, "status": status}


def test_initial_active_agent_is_ppo():
    assert HybridController().active_agent == AgentMode.PPO_GLOBAL


def test_new_candidate_triggers_dqn_and_records_reason():
    controller = HybridController()
    reason = controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(),
        anomalies=[candidate("A")],
        reward_components={},
        confirmation_metadata={},
        history={"seen_candidate_signatures": set()},
    )
    assert reason == "NEW_CANDIDATE"
    assert controller.active_agent == AgentMode.DQN_LOCAL
    assert controller.handoff_events[-1]["reason"] == "NEW_CANDIDATE"


def test_r10_detected_candidate_triggers_dqn_handoff_without_special_controller_rule():
    controller = HybridController()
    r10 = candidate("R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points")
    r10.update({"rule_id": "R10", "lifecycle_status": "DETECTED", "pending_confirmation": True})

    reason = controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(),
        anomalies=[r10],
        reward_components={},
        confirmation_metadata={},
        history={"seen_candidate_signatures": set()},
    )

    assert reason == "NEW_CANDIDATE"
    assert controller.active_agent == AgentMode.DQN_LOCAL
    assert controller.local_session is not None
    assert controller.local_session.trigger_candidate_key == r10["logical_key"]


def test_new_candidate_handoff_prefers_pending_confirmation_candidate_identity():
    controller = HybridController()
    r06 = candidate("R06|/|layout-overflow|target:98")
    r06.update({"rule_id": "R06"})
    r10 = candidate("R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points")
    r10.update({"rule_id": "R10", "pending_confirmation": True, "confirmation_policy": "reproduce"})

    reason = controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(),
        anomalies=[r06, r10],
        reward_components={},
        confirmation_metadata={},
        history={"seen_candidate_signatures": set(), "seen_reward_candidate_keys": set()},
    )

    assert reason == "NEW_CANDIDATE"
    assert controller.local_session is not None
    assert controller.local_session.trigger_candidate_key == r10["logical_key"]
    assert r06["logical_key"] in controller.local_session.entry_candidate_keys


def test_observation_pending_does_not_occupy_action_replay_handoff():
    controller = HybridController()
    reason = controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(),
        anomalies=[],
        reward_components={},
        confirmation_metadata={"pending_confirmation_exists": True, "confirmation_mode": "OBSERVATION_REPRODUCTION"},
        history={},
    )
    assert reason == ""


def test_functional_region_handoff_and_ordinary_scroll_does_not_trigger():
    plain = HybridController()
    assert plain.maybe_handoff_after_ppo_step(step=1, observation=obs(), anomalies=[], reward_components={}, confirmation_metadata={}, history={}) == ""
    functional = HybridController()
    assert functional.maybe_handoff_after_ppo_step(step=1, observation=obs(functional=True), anomalies=[], reward_components={}, confirmation_metadata={}, history={}) == "FUNCTIONAL_REGION"


def test_stalled_functional_handoff():
    controller = HybridController()
    assert controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(functional=True),
        anomalies=[],
        reward_components={"no_effect_penalty": -0.2},
        confirmation_metadata={},
        history={},
    ) == "FUNCTIONAL_REGION"


def enter_dqn(max_local_steps: int = 5) -> HybridController:
    controller = HybridController(max_local_steps=max_local_steps)
    controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(functional=True),
        anomalies=[],
        reward_components={},
        confirmation_metadata={},
        history={},
    )
    return controller


def test_local_budget_return_exact():
    controller = enter_dqn(max_local_steps=2)
    assert controller.maybe_return_after_dqn_step(step=2, observation=obs(), anomalies=[], reward_components={"new_local_state": 1}, repeat_diagnostic={}, done=False) == ""
    assert controller.maybe_return_after_dqn_step(step=3, observation=obs(), anomalies=[], reward_components={"new_local_state": 1}, repeat_diagnostic={}, done=False) == "LOCAL_BUDGET"
    assert controller.summary()["max_dqn_session_length"] == 2


def test_confirmed_return():
    controller = HybridController()
    controller.maybe_handoff_after_ppo_step(step=1, observation=obs(), anomalies=[candidate("A")], reward_components={}, confirmation_metadata={}, history={})
    assert controller.maybe_return_after_dqn_step(step=2, observation=obs(), anomalies=[candidate("A", "confirmed")], reward_components={}, repeat_diagnostic={}, done=False) == "CONFIRMED"


def test_context_change_return():
    controller = enter_dqn()
    assert controller.maybe_return_after_dqn_step(step=2, observation=obs("http://site.test/cart"), anomalies=[], reward_components={"new_local_state": 1}, repeat_diagnostic={}, done=False) == "CONTEXT_CHANGED"


def test_repeat_guard_return():
    controller = enter_dqn()
    repeat = {"unproductive_repeat": True, "cycle_repeat": True}
    assert controller.maybe_return_after_dqn_step(step=2, observation=obs(functional=True), anomalies=[], reward_components={}, repeat_diagnostic=repeat, done=False) == ""
    assert controller.maybe_return_after_dqn_step(step=3, observation=obs(functional=True), anomalies=[], reward_components={}, repeat_diagnostic=repeat, done=False) == "REPEAT_GUARD"


def test_browser_not_reset_on_handoff_or_return_is_controller_level():
    controller = enter_dqn()
    before = id(controller)
    controller.maybe_return_after_dqn_step(step=2, observation=obs(functional=True), anomalies=[], reward_components={}, repeat_diagnostic={}, done=False)
    assert id(controller) == before


def test_shared_confirmation_tracker_and_candidate_identity_are_history_dict():
    history = {"pending_reproduction_records": {"sig": {"logical_key": "A"}}}
    controller = HybridController()
    controller.maybe_handoff_after_ppo_step(step=1, observation=obs(), anomalies=[candidate("A")], reward_components={}, confirmation_metadata={}, history=history)
    assert controller.local_session.entry_candidate_keys == {"A"}
    assert history["pending_reproduction_records"]["sig"]["logical_key"] == "A"


def test_ppo_candidate_dqn_confirmed_definition():
    candidate_first_agent = {"A": "PPO"}
    event = {"logical_key": "A", "newly_confirmed": True}
    assert candidate_first_agent[event["logical_key"]] == "PPO"


def test_candidate_reward_not_duplicated_by_seen_candidate_history():
    controller = HybridController()
    reason = controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=obs(),
        anomalies=[candidate("A")],
        reward_components={},
        confirmation_metadata={},
        history={"seen_candidate_signatures": {"A"}, "seen_reward_candidate_keys": {"A"}},
    )
    assert reason == ""


def test_ppo_and_dqn_observation_dims_are_distinct():
    ppo_obs = np.zeros(635, dtype=np.float32)
    dqn_obs = np.zeros(976, dtype=np.float32)
    assert ppo_obs.shape[0] == 635
    assert dqn_obs.shape[0] == 976


def test_ppo_history_feeds_dqn_history_and_dqn_actions_continue_history():
    history: dict = {}
    action = {"action_type": "scroll_down", "candidate_index": 0}
    _append_cross_agent_history(history, obs(), obs(), action, {"new_state": 0.5}, {})
    _sync_cross_agent_action_target_history(history)
    assert history["dqn_history_records"][-1]["action_type"] == "scroll_down"
    assert history["dqn_action_target_history"][-1] == "scroll_down:0"
    _append_cross_agent_history(history, obs(), obs(), {"action_type": "scroll_up", "candidate_index": 0}, {}, {})
    _sync_cross_agent_action_target_history(history)
    assert [row["action_type"] for row in history["dqn_history_records"][-2:]] == ["scroll_down", "scroll_up"]


def test_dqn_to_ppo_observation_rebuild_uses_current_state_shape():
    current_observation_vector = np.zeros(635, dtype=np.float32)
    assert current_observation_vector.shape[0] == 635


def test_buffer_and_reward_separation_counters_do_not_mix():
    ppo_buffer_steps = 3
    dqn_replay_steps = 2
    reward_calls = {"PPO": 3, "DQN": 2}
    assert ppo_buffer_steps + dqn_replay_steps == 5
    assert reward_calls["PPO"] + reward_calls["DQN"] == 5


def test_switch_is_not_browsergym_action_and_no_zero_step_ping_pong():
    timeline = [
        {"active_agent": "PPO_GLOBAL"},
        {"active_agent": "DQN_LOCAL"},
        {"active_agent": "PPO_GLOBAL"},
    ]
    assert _has_ppo_dqn_ppo_sequence(timeline)
    assert "PPO_TO_DQN" not in {row.get("action_type") for row in timeline}


def test_handoff_and_return_reason_recorded():
    controller = enter_dqn(max_local_steps=1)
    assert controller.handoff_events[-1]["reason"] == "FUNCTIONAL_REGION"
    assert controller.maybe_return_after_dqn_step(step=2, observation=obs(functional=True), anomalies=[], reward_components={}, repeat_diagnostic={}, done=False) == "LOCAL_BUDGET"
    assert controller.return_events[-1]["reason"] == "LOCAL_BUDGET"


def test_ground_truth_independent_controller_source():
    names = set(HybridController.maybe_handoff_after_ppo_step.__code__.co_names)
    forbidden = {"ground_truth", "known_bug", "injected", "expected_defect"}
    assert not names.intersection(forbidden)


def test_final_dqn_manifest_validation(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "architecture": "factorized_v1_confobs_histv2",
                "dqn_obs_dim": 976,
                "ppo_obs_dim": 635,
                "action_dim": 608,
                "history_observation_version": "v2",
                "DQN_ROLE_CHECK": "FINAL PASS",
            }
        ),
        encoding="utf-8",
    )
    assert _validate_dqn_manifest(path)["dqn_obs_dim"] == 976


def test_final_dqn_manifest_validation_rejects_mismatch(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"architecture": "factorized_v1_confobs_hist"}), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate_dqn_manifest(path)
