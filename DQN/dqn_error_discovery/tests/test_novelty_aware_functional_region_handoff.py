from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController


def _observation(*, form_state="idle", target_text="Search"):
    return {
        "page_state": {"url": "http://localhost:9225/", "form_state": form_state},
        "candidate_elements": [{"role": "textbox", "tag": "input", "text": target_text, "is_form_field": True}],
    }


def _handoff(controller, observation):
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    return controller.maybe_handoff_after_ppo_step(
        step=1,
        observation=observation,
        anomalies=[],
        reward_components={},
        confirmation_metadata={},
        history={},
    )


def _return(controller, observation):
    return controller.maybe_return_after_dqn_step(
        step=2,
        observation=observation,
        anomalies=[],
        reward_components={},
        repeat_diagnostic={},
        done=False,
    )


def test_first_time_region_is_eligible():
    controller = HybridController(max_local_steps=1, functional_region_novelty_gate_enabled=True)
    assert _handoff(controller, _observation()) == "FUNCTIONAL_REGION"


def test_unchanged_region_is_suppressed_after_unproductive_session():
    controller = HybridController(max_local_steps=1, functional_region_novelty_gate_enabled=True)
    obs = _observation()
    assert _handoff(controller, obs) == "FUNCTIONAL_REGION"
    assert _return(controller, obs) == "LOCAL_BUDGET"
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    assert controller.maybe_handoff_after_ppo_step(
        step=3, observation=obs, anomalies=[], reward_components={}, confirmation_metadata={}, history={}
    ) == ""
    assert controller.functional_region_opportunities[-1]["handoff_performed"] is False
    assert "NO_REGION_NOVELTY" in controller.functional_region_opportunities[-1]["suppression_reasons"]


def test_semantic_state_change_reopens_region():
    controller = HybridController(max_local_steps=1, functional_region_novelty_gate_enabled=True)
    obs = _observation()
    assert _handoff(controller, obs) == "FUNCTIONAL_REGION"
    assert _return(controller, obs) == "LOCAL_BUDGET"
    changed = _observation(form_state="submitted")
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    assert controller.maybe_handoff_after_ppo_step(
        step=3, observation=changed, anomalies=[], reward_components={}, confirmation_metadata={}, history={}
    ) == "FUNCTIONAL_REGION"


def test_candidate_handoff_bypasses_functional_novelty_gate():
    controller = HybridController(max_local_steps=1, functional_region_novelty_gate_enabled=True)
    obs = _observation()
    assert _handoff(controller, obs) == "FUNCTIONAL_REGION"
    assert _return(controller, obs) == "LOCAL_BUDGET"
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    anomaly = {"status": "candidate", "logical_key": "R01|candidate"}
    assert controller.maybe_handoff_after_ppo_step(
        step=3, observation=obs, anomalies=[anomaly], reward_components={}, confirmation_metadata={}, history={"seen_candidate_signatures": set()}
    ) == "NEW_CANDIDATE"


def test_episode_reset_clears_region_history():
    controller = HybridController(functional_region_novelty_gate_enabled=True)
    assert _handoff(controller, _observation()) == "FUNCTIONAL_REGION"
    assert controller.functional_region_history
    controller.reset_episode_guard()
    assert controller.functional_region_history == {}


def test_v1_default_keeps_legacy_rehandoff_behavior():
    controller = HybridController(max_local_steps=1)
    obs = _observation()
    assert _handoff(controller, obs) == "FUNCTIONAL_REGION"
    assert _return(controller, obs) == "LOCAL_BUDGET"
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    assert controller.maybe_handoff_after_ppo_step(
        step=3, observation=obs, anomalies=[], reward_components={}, confirmation_metadata={}, history={}
    ) == "FUNCTIONAL_REGION"


def test_static_metadata_does_not_create_relevant_evidence_without_relevance():
    controller = HybridController(max_local_steps=1, functional_region_novelty_gate_enabled=True)
    obs = _observation()
    assert _handoff(controller, obs) == "FUNCTIONAL_REGION"
    assert _return(controller, obs) == "LOCAL_BUDGET"
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    assert controller.maybe_handoff_after_ppo_step(
        step=3, observation=obs, anomalies=[], reward_components={}, confirmation_metadata={"functional_relevance": "IRRELEVANT"}, history={}
    ) == ""


def test_candidate_logical_key_is_not_region_key():
    controller = HybridController(functional_region_novelty_gate_enabled=True)
    assert _handoff(controller, _observation()) == "FUNCTIONAL_REGION"
    event = controller.handoff_events[-1]
    assert event["trigger_candidate_key"] == ""
    assert event["functional_region_novelty"]["region_key"] != event["trigger_candidate_key"]
