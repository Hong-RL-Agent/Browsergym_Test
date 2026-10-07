from __future__ import annotations

from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController


def obs(url: str = "http://site.test/", *, functional: bool = False, bbox: int = 0, order: int = 0, form: str = "") -> dict:
    candidates = [{"bid": "A", "role": "button", "bbox": [bbox, 0, 1, 1], "order": order}]
    if functional:
        candidates.append({"bid": "search", "role": "searchbox", "tag": "input", "is_form_field": True, "text": "search"})
    return {"page_state": {"url": url, "title": "Site", "form_state": form}, "candidate_elements": candidates}


def meta(
    key: str = "A",
    *,
    mode: str = "OBSERVATION_REPRODUCTION",
    progress: float = 0.0,
    visible: bool = True,
    action_type: str = "",
    target: str = "",
    rule: str = "R05",
) -> dict:
    return {
        "pending_confirmation_exists": True,
        "confirmation_mode": mode,
        "active_pending_key": key,
        "active_rule_id": rule,
        "expected_action_type": action_type or ("click_element" if mode == "ACTION_REPRODUCTION" else ""),
        "expected_target_identity": target or ("bid:A" if mode == "ACTION_REPRODUCTION" else ""),
        "expected_action_requires_target": mode == "ACTION_REPRODUCTION",
        "expected_target_visible": visible,
        "confirmation_progress": progress,
    }


def hist(key: str = "A", *, progress: float = 0.0, mode: str = "OBSERVATION_REPRODUCTION", action_type: str = "", target: str = "", rule: str = "R05") -> dict:
    return {
        "pending_reproduction_records": {
            key: {
                "logical_key": key,
                "rule_id": rule,
                "confirmation_mode": mode,
                "expected_action_type": action_type or ("click_element" if mode == "ACTION_REPRODUCTION" else ""),
                "expected_target_identity": target or ("bid:A" if mode == "ACTION_REPRODUCTION" else None),
                "context_requirements": ["url_path"],
                "action_target": f"{action_type or 'click_element'}:{target or 'bid:A'}" if mode == "ACTION_REPRODUCTION" else "",
                "progress": progress,
            }
        }
    }


def ppo(controller: HybridController, step: int, *, key: str = "A", observation: dict | None = None, metadata: dict | None = None, history: dict | None = None, anomalies: list[dict] | None = None, reward: dict | None = None) -> str:
    controller.active_agent = AgentMode.PPO_GLOBAL
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    return controller.maybe_handoff_after_ppo_step(
        step=step,
        observation=observation or obs(functional=True),
        anomalies=anomalies or [],
        reward_components=reward or {},
        confirmation_metadata=metadata if metadata is not None else meta(key),
        history=history if history is not None else hist(key),
    )


def finish(controller: HybridController, step: int, *, key: str = "A", progress: float = 0.0, metadata: dict | None = None, history: dict | None = None, observation: dict | None = None) -> str:
    return controller.maybe_return_after_dqn_step(
        step=step,
        observation=observation or obs(functional=True),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        tracker_history=history or hist(key, progress=progress),
        confirmation_metadata=metadata or meta(key, progress=progress),
    )


def test_first_pending_eligible_and_unresolved_becomes_stale() -> None:
    controller = HybridController()
    assert ppo(controller, 1) == "PENDING_CONFIRMATION"
    finish(controller, 2)
    assert ppo(controller, 3) == ""
    guard = controller.summary()["pending_handoff_guard"]
    assert guard["stale_pending_handoff_blocked"] >= 1


def test_ppo_steps_alone_do_not_make_stale_candidate_eligible() -> None:
    controller = HybridController(pending_rehandoff_min_ppo_steps=3)
    assert ppo(controller, 1) == "PENDING_CONFIRMATION"
    finish(controller, 2)
    for step in range(3, 13):
        assert ppo(controller, step) == ""
    assert controller.summary()["handoff_count"] == 1


def test_progress_increase_eligible_once_and_second_increase_eligible_again() -> None:
    controller = HybridController()
    assert ppo(controller, 1, history=hist(progress=0.0), metadata=meta(progress=0.0)) == "PENDING_CONFIRMATION"
    finish(controller, 2, progress=0.0)
    assert ppo(controller, 3, history=hist(progress=0.5), metadata=meta(progress=0.5)) == "PENDING_CONFIRMATION"
    finish(controller, 4, progress=0.5)
    assert ppo(controller, 5, history=hist(progress=0.5), metadata=meta(progress=0.5)) == ""
    assert ppo(controller, 6, history=hist(progress=1.0), metadata=meta(progress=1.0)) == "PENDING_CONFIRMATION"
    assert controller.summary()["pending_handoff_guard"]["rehandoff_due_to_progress"] == 2


def test_material_context_change_eligible_once_bbox_and_order_noise_ignored() -> None:
    controller = HybridController()
    assert ppo(controller, 1, observation=obs(functional=True, bbox=1, order=1)) == "PENDING_CONFIRMATION"
    finish(controller, 2)
    assert ppo(controller, 3, observation=obs(functional=True, bbox=99, order=9)) == ""
    assert ppo(controller, 4, observation=obs("http://site.test/cart", functional=True)) == "PENDING_CONFIRMATION"
    finish(controller, 5, metadata=meta(), history=hist(), observation=obs("http://site.test/cart", functional=True))
    assert ppo(controller, 6, observation=obs("http://site.test/cart", functional=True)) == ""
    assert controller.summary()["pending_handoff_guard"]["rehandoff_due_to_context_change"] == 1


def test_target_false_to_true_eligible_but_persistent_true_not_reused() -> None:
    controller = HybridController()
    key = "A"
    assert ppo(controller, 1, metadata=meta(key, mode="ACTION_REPRODUCTION", visible=True), history=hist(key, mode="ACTION_REPRODUCTION")) == "PENDING_CONFIRMATION"
    state = controller.pending_handoff_state[key]
    state.last_target_resolvable = False
    finish(controller, 2, metadata=meta(key, mode="ACTION_REPRODUCTION", visible=False), history=hist(key, mode="ACTION_REPRODUCTION"))
    state.last_target_resolvable = False
    assert ppo(controller, 3, metadata=meta(key, mode="ACTION_REPRODUCTION", visible=True), history=hist(key, mode="ACTION_REPRODUCTION")) == "PENDING_CONFIRMATION"
    finish(controller, 4, metadata=meta(key, mode="ACTION_REPRODUCTION", visible=True), history=hist(key, mode="ACTION_REPRODUCTION"))
    assert ppo(controller, 5, metadata=meta(key, mode="ACTION_REPRODUCTION", visible=True), history=hist(key, mode="ACTION_REPRODUCTION")) == ""


def test_reproduction_requirement_change_eligible_same_requirement_stale() -> None:
    controller = HybridController()
    assert ppo(controller, 1, metadata=meta("A", mode="ACTION_REPRODUCTION", target="bid:A"), history=hist("A", mode="ACTION_REPRODUCTION", target="bid:A")) == "PENDING_CONFIRMATION"
    finish(controller, 2, metadata=meta("A", mode="ACTION_REPRODUCTION", target="bid:A"), history=hist("A", mode="ACTION_REPRODUCTION", target="bid:A"))
    assert ppo(controller, 3, metadata=meta("A", mode="ACTION_REPRODUCTION", target="bid:A"), history=hist("A", mode="ACTION_REPRODUCTION", target="bid:A")) == ""
    assert ppo(controller, 4, metadata=meta("A", mode="ACTION_REPRODUCTION", target="bid:B"), history=hist("A", mode="ACTION_REPRODUCTION", target="bid:B")) == "PENDING_CONFIRMATION"
    assert controller.summary()["pending_handoff_guard"]["rehandoff_due_to_reproduction_requirement"] == 1


def test_different_candidate_and_new_candidate_unaffected() -> None:
    controller = HybridController()
    assert ppo(controller, 1, key="A", history=hist("A"), metadata=meta("A")) == "PENDING_CONFIRMATION"
    finish(controller, 2, key="A")
    assert ppo(controller, 3, key="B", history=hist("B"), metadata=meta("B")) == "PENDING_CONFIRMATION"
    finish(controller, 4, key="B")
    reason = ppo(
        controller,
        5,
        anomalies=[{"logical_key": "C", "signature": "C", "status": "candidate"}],
        history={"seen_candidate_signatures": set(), "pending_reproduction_records": {"A": {"logical_key": "A"}}},
        metadata=meta("A"),
    )
    assert reason == "NEW_CANDIDATE"


def test_multiple_stale_candidates_do_not_rotate_into_dqn() -> None:
    controller = HybridController()
    assert ppo(controller, 1, key="A", history=hist("A"), metadata=meta("A")) == "PENDING_CONFIRMATION"
    finish(controller, 2, key="A")
    assert ppo(controller, 3, key="B", history=hist("B"), metadata=meta("B")) == "PENDING_CONFIRMATION"
    finish(controller, 4, key="B")
    pending = {"pending_reproduction_records": {"A": hist("A")["pending_reproduction_records"]["A"], "B": hist("B")["pending_reproduction_records"]["B"]}}
    assert ppo(controller, 5, key="A", history=pending, metadata=meta("A")) == ""
    assert ppo(controller, 6, key="B", history=pending, metadata=meta("B")) == ""


def test_stale_pending_does_not_bypass_same_context_functional_or_stalled() -> None:
    controller = HybridController()
    assert ppo(controller, 1, observation=obs(functional=True)) == "PENDING_CONFIRMATION"
    finish(controller, 2)
    assert ppo(controller, 3, observation=obs(functional=True), reward={}) == ""
    assert ppo(controller, 4, observation=obs(functional=True), reward={"no_effect_penalty": -0.2}) == ""
    guard = controller.summary()["pending_handoff_guard"]
    assert guard["same_context_functional_bypass_blocked_count"] >= 1
    assert guard["same_context_stalled_bypass_blocked_count"] >= 1


def test_new_functional_context_allowed() -> None:
    controller = HybridController()
    assert ppo(controller, 1, observation=obs(functional=True)) == "PENDING_CONFIRMATION"
    finish(controller, 2)
    assert ppo(controller, 3, observation=obs("http://site.test/search", functional=True), metadata={}, history={}) == "FUNCTIONAL_REGION"


def test_confirmed_candidate_excluded_and_episode_reset_clears_state() -> None:
    controller = HybridController()
    assert ppo(controller, 1) == "PENDING_CONFIRMATION"
    assert controller.maybe_return_after_dqn_step(
        step=2,
        observation=obs(functional=True),
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
        confirmation_events=[{"logical_key": "A", "current_status": "confirmed"}],
        tracker_history={"rewarded_confirmed_logical_keys": {"A"}, "pending_reproduction_records": {}},
        confirmation_metadata=meta("A"),
    ) == "CONFIRMED"
    assert ppo(controller, 3, observation=obs(functional=False)) == ""
    controller.reset_episode_guard()
    assert ppo(controller, 4) == "PENDING_CONFIRMATION"


def test_canonical_key_ground_truth_reward_return_buffer_independence() -> None:
    controller = HybridController()
    assert ppo(controller, 1, key="R05|/|layout-overlap|target:28", history=hist("R05|/|layout-overlap|target:28"), metadata=meta("R05|/|layout-overlap|target:28")) == "PENDING_CONFIRMATION"
    names = set(HybridController.maybe_handoff_after_ppo_step.__code__.co_names)
    assert not names.intersection({"ground_truth", "known_bug", "injected", "expected_defect", "DQN_REWARD_CONFIG", "rollout_buffer", "replay_buffer"})
