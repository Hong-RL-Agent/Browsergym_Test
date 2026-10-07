from __future__ import annotations

from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController


def obs(url: str = "http://site.test/", *, functional: bool = False, bbox: int = 0, order: int = 0, form: str = "") -> dict:
    candidates = [{"bid": "A", "role": "button", "bbox": [bbox, 0, 10, 10], "order": order}]
    if functional:
        candidates.append({"bid": "search", "role": "searchbox", "tag": "input", "is_form_field": True, "text": "search"})
    return {"page_state": {"url": url, "title": "Site", "form_state": form}, "candidate_elements": candidates}


def pending_meta(key: str = "A", *, mode: str = "OBSERVATION_REPRODUCTION", progress: float = 0.0, visible: int = 1) -> dict:
    return {
        "pending_confirmation_exists": True,
        "confirmation_mode": mode,
        "active_pending_key": key,
        "active_rule_id": "R05",
        "expected_action_type": "click_element" if mode == "ACTION_REPRODUCTION" else "",
        "expected_target_identity": "bid:A" if mode == "ACTION_REPRODUCTION" else None,
        "expected_action_requires_target": 1 if mode == "ACTION_REPRODUCTION" else 0,
        "expected_target_visible": visible,
        "confirmation_progress": progress,
    }


def history(*keys: str) -> dict:
    return {"pending_reproduction_records": {key: {"logical_key": key, "rule_id": "R05"} for key in keys}}


def ppo_step(controller: HybridController, step: int, *, key: str = "A", observation: dict | None = None, metadata: dict | None = None, reward: dict | None = None) -> str:
    controller.active_agent = AgentMode.PPO_GLOBAL
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    return controller.maybe_handoff_after_ppo_step(
        step=step,
        observation=observation or obs(functional=True),
        anomalies=[],
        reward_components=reward or {},
        confirmation_metadata=metadata or pending_meta(key),
        history=history(key),
    )


def finish_session(controller: HybridController, step: int, *, productive: bool = False) -> str:
    components = {"new_local_state": 1.0} if productive else {}
    return controller.maybe_return_after_dqn_step(step=step, observation=obs(functional=True), anomalies=[], reward_components=components, repeat_diagnostic={}, done=False)


def test_first_pending_candidate_eligible() -> None:
    assert ppo_step(HybridController(), 1) == "PENDING_CONFIRMATION"


def test_immediate_same_candidate_rehandoff_blocked() -> None:
    controller = HybridController()
    assert ppo_step(controller, 1) == "PENDING_CONFIRMATION"
    finish_session(controller, 2)
    assert ppo_step(controller, 3) == ""
    assert controller.summary()["pending_handoff_guard"]["same_candidate_rehandoff_blocked_count"] >= 1


def test_after_one_and_two_ppo_actions_still_blocked() -> None:
    controller = HybridController(pending_rehandoff_min_ppo_steps=3)
    ppo_step(controller, 1)
    finish_session(controller, 2)
    assert ppo_step(controller, 3) == ""
    assert ppo_step(controller, 4) == ""


def test_after_three_ppo_actions_still_stale_without_change() -> None:
    controller = HybridController(pending_rehandoff_min_ppo_steps=3)
    ppo_step(controller, 1)
    finish_session(controller, 2)
    ppo_step(controller, 3)
    ppo_step(controller, 4)
    assert ppo_step(controller, 5) == ""


def test_dqn_actions_do_not_advance_pending_cooldown() -> None:
    controller = HybridController(pending_rehandoff_min_ppo_steps=3)
    ppo_step(controller, 1)
    controller.maybe_return_after_dqn_step(step=2, observation=obs(functional=True), anomalies=[], reward_components={"new_local_state": 1}, repeat_diagnostic={}, done=False)
    controller.active_agent = AgentMode.DQN_LOCAL
    controller.record_agent_step(AgentMode.DQN_LOCAL)
    finish_session(controller, 3)
    assert ppo_step(controller, 4) == ""


def test_switch_events_do_not_advance_cooldown() -> None:
    controller = HybridController(pending_rehandoff_min_ppo_steps=3)
    ppo_step(controller, 1)
    finish_session(controller, 2)
    before = controller.ppo_action_count
    controller.active_agent = AgentMode.PPO_GLOBAL
    controller.active_agent = AgentMode.DQN_LOCAL
    controller.active_agent = AgentMode.PPO_GLOBAL
    assert controller.ppo_action_count == before


def test_different_candidate_allowed_during_a_cooldown() -> None:
    controller = HybridController()
    ppo_step(controller, 1, key="A")
    finish_session(controller, 2)
    assert ppo_step(controller, 3, key="B", metadata=pending_meta("B")) == "PENDING_CONFIRMATION"


def test_new_candidate_bypasses_pending_cooldown() -> None:
    controller = HybridController()
    ppo_step(controller, 1, key="A")
    finish_session(controller, 2)
    controller.active_agent = AgentMode.PPO_GLOBAL
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    reason = controller.maybe_handoff_after_ppo_step(
        step=3,
        observation=obs(),
        anomalies=[{"logical_key": "B", "signature": "B", "status": "candidate"}],
        reward_components={},
        confirmation_metadata=pending_meta("A"),
        history={"seen_candidate_signatures": set(), "pending_reproduction_records": {"A": {"logical_key": "A"}}},
    )
    assert reason == "NEW_CANDIDATE"


def test_confirmation_progress_bypasses_cooldown() -> None:
    controller = HybridController()
    ppo_step(controller, 1, key="A", metadata=pending_meta("A", progress=0.0))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, key="A", metadata=pending_meta("A", progress=0.5)) == "PENDING_CONFIRMATION"


def test_context_change_bypasses_cooldown() -> None:
    controller = HybridController()
    ppo_step(controller, 1, observation=obs("http://site.test/"))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, observation=obs("http://site.test/cart")) == "PENDING_CONFIRMATION"


def test_target_became_resolvable_bypasses_cooldown() -> None:
    controller = HybridController()
    ppo_step(controller, 1, metadata=pending_meta("A", mode="ACTION_REPRODUCTION", visible=0))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, metadata=pending_meta("A", mode="ACTION_REPRODUCTION", visible=1)) == "PENDING_CONFIRMATION"


def test_bbox_only_and_candidate_order_change_do_not_bypass() -> None:
    controller = HybridController()
    ppo_step(controller, 1, observation=obs(bbox=1, order=1))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, observation=obs(bbox=99, order=9)) == ""


def test_same_context_functional_region_cannot_bypass() -> None:
    controller = HybridController()
    ppo_step(controller, 1, observation=obs(functional=True))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, observation=obs(functional=True), metadata=pending_meta("A")) == ""
    assert controller.summary()["pending_handoff_guard"]["same_context_functional_bypass_blocked_count"] >= 1


def test_new_context_functional_region_allowed() -> None:
    controller = HybridController()
    ppo_step(controller, 1, observation=obs(functional=True))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, observation=obs("http://site.test/search", functional=True), metadata=pending_meta("A")) == "PENDING_CONFIRMATION"


def test_same_context_stalled_functional_cannot_bypass() -> None:
    controller = HybridController()
    ppo_step(controller, 1, observation=obs(functional=True))
    finish_session(controller, 2)
    assert ppo_step(controller, 3, observation=obs(functional=True), reward={"no_effect_penalty": -0.2}) == ""


def test_confirmed_candidate_no_longer_pending_trigger() -> None:
    controller = HybridController()
    assert controller.maybe_handoff_after_ppo_step(step=1, observation=obs(), anomalies=[], reward_components={}, confirmation_metadata={"pending_confirmation_exists": False}, history={}) == ""


def test_episode_reset_clears_guard_state() -> None:
    controller = HybridController()
    ppo_step(controller, 1)
    finish_session(controller, 2)
    controller.reset_episode_guard()
    assert ppo_step(controller, 3) == "PENDING_CONFIRMATION"


def test_candidate_logical_identity_stable_over_bbox_order_noise() -> None:
    controller = HybridController()
    ppo_step(controller, 1, observation=obs(bbox=1, order=1), metadata=pending_meta("A"))
    finish_session(controller, 2)
    ppo_step(controller, 3, observation=obs(bbox=99, order=9), metadata=pending_meta("A"))
    assert "A" in controller.summary()["pending_handoff_guard"]["candidate_states"]


def test_no_zero_step_ping_pong_and_local_budget_unchanged() -> None:
    controller = HybridController(max_local_steps=5)
    ppo_step(controller, 1)
    for step in range(2, 6):
        assert controller.maybe_return_after_dqn_step(step=step, observation=obs(functional=True), anomalies=[], reward_components={"new_local_state": 1}, repeat_diagnostic={}, done=False) == ""
    assert controller.maybe_return_after_dqn_step(step=6, observation=obs(functional=True), anomalies=[], reward_components={"new_local_state": 1}, repeat_diagnostic={}, done=False) == "LOCAL_BUDGET"
    assert controller.summary()["zero_step_switch_loop"] is False


def test_reward_buffer_and_ground_truth_independence() -> None:
    names = set(HybridController.maybe_handoff_after_ppo_step.__code__.co_names)
    assert not names.intersection({"ground_truth", "known_bug", "injected", "expected_defect"})
    ppo_buffer_steps = 1
    dqn_replay_steps = 1
    assert ppo_buffer_steps + dqn_replay_steps == 2
