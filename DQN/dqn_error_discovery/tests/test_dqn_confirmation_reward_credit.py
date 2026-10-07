from dqn_error_discovery.runners.train_dqn_error_detector import DQN_REWARD_CONFIG, calculate_dqn_reward


def _candidate(bid="A"):
    return {"bid": bid, "name": bid, "role": "button", "tag": "button", "clickable": True, "visible": True}


def _obs(candidates=None):
    return {
        "page_state": {"url": "http://app.test/", "page_text_length": 10, "dom_node_count": 1},
        "candidate_elements": candidates or [_candidate()],
        "runtime_signals": {},
    }


def _action(action_type="click_element", candidate_index=0):
    return {"action_type": action_type, "candidate_index": candidate_index}


def _anomaly(rule="R01", status="candidate", signature="sig-A"):
    return {
        "type": "button-no-response" if rule == "R01" else "layout-overlap",
        "rule_id": rule,
        "status": status,
        "signature": signature,
        "confirmation_policy": "reproduce",
        "action_type": "click_element" if rule == "R01" else "scroll_down",
        "target_id": "A",
        "issue_type": "layout-overlap" if rule != "R01" else "button-no-response",
        "evidence": {"rule_id": rule, "target_bid": "A"},
    }


def _reward(history, anomaly, action=None, before=None, after=None):
    before = before or _obs()
    after = after or _obs()
    return calculate_dqn_reward(before, after, action or _action(), [anomaly], history)


def test_candidate_to_confirmed_applies_existing_upgrade_value():
    history = {}
    _reward(history, _anomaly(), before=_obs(), after=_obs())
    _, components = _reward(history, _anomaly(), before=_obs(), after=_obs())
    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]


def test_r01_reproduction_confirmation_credit_and_no_effect_can_coexist():
    history = {}
    same = _obs()
    _reward(history, _anomaly(), before=same, after=same)
    _, components = _reward(history, _anomaly(), before=same, after=same)
    assert components["confirmation_upgrade"] > 0
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_r01_repeated_confirmation_has_no_second_credit():
    history = {}
    _reward(history, _anomaly())
    _reward(history, _anomaly())
    _, components = _reward(history, _anomaly())
    assert components["confirmed_anomaly"] == 0
    assert components["confirmation_upgrade"] == 0
    assert history["last_confirmation_reward_duplicate_blocks"] == 1


def test_observation_reproduction_credits_r05_once():
    history = {}
    anomaly = _anomaly("R05", signature="r05-A")
    _reward(history, anomaly)
    _, second = _reward(history, anomaly)
    _, third = _reward(history, anomaly)
    assert second["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert third["confirmation_upgrade"] == 0
    assert history["newly_confirmed_transition_count"] == 1


def test_observation_reproduction_credits_r06_once():
    history = {}
    anomaly = _anomaly("R06", signature="r06-A")
    _reward(history, anomaly)
    _, second = _reward(history, anomaly)
    assert second["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert history["confirmed_reward_event_count"] == 1


def test_two_logical_candidates_each_receive_one_credit():
    history = {}
    for key in ("A", "B"):
        anomaly = _anomaly("R05", signature=f"r05-{key}")
        anomaly["target_id"] = key
        anomaly["evidence"]["target_bid"] = key
        _reward(history, anomaly)
        _, components = _reward(history, anomaly)
        assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert history["newly_confirmed_transition_count"] == 2


def test_progress_without_confirmed_does_not_create_shaping_reward():
    history = {"last_confirmation_progress": True, "last_confirmation_confirmed": False}
    _, components = _reward(history, _anomaly("R05", signature="progress-only"))
    assert components["confirmed_anomaly"] == 0
    assert components["confirmation_upgrade"] == 0


def test_direct_confirmed_uses_existing_confirmed_value_once():
    history = {}
    anomaly = _anomaly("R13", status="confirmed", signature="r13-A")
    anomaly["confirmation_policy"] = "strong-evidence"
    _, first = _reward(history, anomaly)
    _, second = _reward(history, anomaly)
    assert first["confirmed_anomaly"] == DQN_REWARD_CONFIG["confirmed_anomaly"]
    assert second["confirmed_anomaly"] == 0
    assert history["last_confirmation_reward_duplicate_blocks"] == 1


def test_ground_truth_fields_do_not_change_credit():
    history_a, history_b = {}, {}
    anomaly_a = _anomaly("R05", signature="same")
    anomaly_b = dict(anomaly_a, matched_bug_id="GT-1", data_bug_id="GT-1", ground_truth={"bug": "GT-1"})
    _reward(history_a, anomaly_a)
    _reward(history_b, anomaly_b)
    _, a = _reward(history_a, anomaly_a)
    _, b = _reward(history_b, anomaly_b)
    assert a["confirmation_upgrade"] == b["confirmation_upgrade"]


def test_reward_transition_metadata_is_recorded():
    history = {}
    anomaly = _anomaly("R05", signature="metadata")
    _reward(history, anomaly)
    _reward(history, anomaly)
    transition = history["last_confirmation_transition"]
    assert transition["previous_status"] == "candidate"
    assert transition["current_status"] == "confirmed"
    assert transition["newly_confirmed"] is True
    assert transition["mode"] == "OBSERVATION_REPRODUCTION"
