from dqn_error_discovery.framework_dev.handoff_quality_diagnostic import _session_records, _summary


def _runtime():
    return {
        "timeline": [
            {"episode": 1, "step": 1, "active_agent": "PPO_GLOBAL", "action_target": "click_element:bid:1", "state_after": "s0", "url_after": "http://test/"},
            {"episode": 1, "step": 2, "active_agent": "DQN_LOCAL", "local_session_id": 1, "action_target": "scroll_down:0", "state_before": "s0", "state_after": "s1", "url_after": "http://test/", "reward_components": {"new_local_state": 0.2, "new_local_action_target": 0.2}, "repeat_diagnostic": {"no_effect": False}},
            {"episode": 1, "step": 3, "active_agent": "DQN_LOCAL", "local_session_id": 1, "action_target": "scroll_down:0", "state_before": "s1", "state_after": "s1", "url_after": "http://test/", "reward_components": {}, "repeat_diagnostic": {"no_effect": True}},
            {"episode": 1, "step": 4, "active_agent": "PPO_GLOBAL", "action_target": "click_element:bid:2", "state_after": "s2", "url_after": "http://test/"},
        ],
        "handoff_events": [{"step": 1, "local_session_id": 1, "reason": "FUNCTIONAL_REGION", "entry_url": "http://test/", "entry_functional_region": "functional_target", "entry_target_resolvable": False, "entry_candidate_keys": [], "entry_pending_confirmation_keys": [], "trigger_candidate_key": "", "trigger_candidate_rule_id": "", "ppo_action_count": 1}],
        "return_events": [{"step": 3, "local_session_id": 1, "reason": "NOVELTY_EXHAUSTED", "local_steps": 2, "session_productive": True}],
        "episode_summary": {"total_steps": 4, "ppo_steps": 2, "dqn_steps": 2},
    }


def test_extracts_local_session_and_reason():
    records = _session_records(_runtime(), "REACT", 42)
    assert len(records) == 1
    assert records[0]["handoff_reason"] == "FUNCTIONAL_REGION"
    assert records[0]["local_steps"] == 2


def test_region_repeat_and_low_actionability_are_diagnostic_only():
    runtime = _runtime()
    records = _session_records(runtime, "REACT", 42)
    assert records[0]["region_visit_index"] == 1
    assert records[0]["actionability"] == "LOW_ACTIONABILITY"
    assert records[0]["dqn_worthy_region"] is False


def test_weak_productivity_is_separate_from_existing_productive_metric():
    records = _session_records(_runtime(), "REACT", 42)
    assert records[0]["weak_productive"] is True
    assert records[0]["strong_productive"] is False
    assert records[0]["productive_existing_metric"] is True


def test_early_futility_and_no_effect_are_recorded():
    records = _session_records(_runtime(), "REACT", 42)
    assert records[0]["early_futility"] is False
    assert records[0]["no_effect_actions"] == 1
    assert records[0]["no_effect_rate"] == 0.5


def test_summary_computes_functional_region_rates():
    runtime = _runtime()
    report = _summary(_session_records(runtime, "REACT", 42), runtime, "REACT", 42)
    assert report["functional_region_sessions"] == 1
    assert report["functional_region_strong_productive_rate"] == 0.0
    assert report["zero_gain_rate"] == 0.0


def test_diagnostic_is_gt_independent():
    runtime = _runtime()
    assert "ground_truth" not in runtime
    records = _session_records(runtime, "REACT", 42)
    assert records[0]["framework"] == "REACT"


def test_reserved_sites_are_not_an_input():
    assert "SITE002" not in _runtime()
    assert "SITE003" not in _runtime()
