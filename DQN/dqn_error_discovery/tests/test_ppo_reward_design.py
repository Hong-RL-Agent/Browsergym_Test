from services.autonomous_reward_service import PPO_REWARD_CONFIG, calculate_autonomous_reward


def _candidate(bid: str = "buy", text: str = "Buy", **extra) -> dict:
    item = {
        "bid": bid,
        "name": text,
        "text": text,
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
    }
    item.update(extra)
    return item


def _obs(url: str = "http://app.test/page", *, text_len: int = 10, candidates: list[dict] | None = None, **page_extra) -> dict:
    page_state = {
        "url": url,
        "title": "Page",
        "page_text_length": text_len,
        "dom_node_count": 1,
    }
    page_state.update(page_extra)
    return {"page_state": page_state, "candidate_elements": candidates or [_candidate()], "runtime_signals": {}}


def _action(action_type: str = "click_element", candidate_index: int = 0, **extra) -> dict:
    item = {"action_type": action_type, "candidate_index": candidate_index}
    item.update(extra)
    return item


def _anomaly(rule_id: str = "R13", status: str = "candidate", signature: str = "sig-a", **extra) -> dict:
    item = {
        "type": "api-server-error" if rule_id == "R13" else "button-no-response",
        "rule_id": rule_id,
        "status": status,
        "confidence": 0.99,
        "signature": signature,
        "confirmation_policy": "strong-evidence" if rule_id not in {"R01", "R02", "R04", "R05", "R06"} else "reproduce",
        "action_type": "click_element",
        "target_id": "buy",
        "evidence": {"rule_id": rule_id},
    }
    item.update(extra)
    return item


def _reward(before=None, after=None, action=None, anomalies=None, history=None):
    return calculate_autonomous_reward(
        before or _obs(),
        after or _obs(text_len=20),
        action or _action(),
        anomalies or [],
        known_bugs=[],
        history=history if history is not None else {},
        site_profile={"bugs": [{"type": "should-not-affect-reward"}]},
    )


def test_new_state_reward():
    _, components = _reward()

    assert components["new_state"] == PPO_REWARD_CONFIG["new_state"]


def test_repeated_state_no_new_state_and_penalty():
    history = {}
    _reward(history=history)
    _, components = _reward(history=history)

    assert components["new_state"] == 0.0
    assert components["repeat_penalty"] <= PPO_REWARD_CONFIG["repeat_state"]


def test_new_target_reward():
    _, components = _reward()

    assert components["new_target"] == PPO_REWARD_CONFIG["new_target"]


def test_new_action_target_reward():
    _, components = _reward()

    assert components["new_action_target"] == PPO_REWARD_CONFIG["new_action_target"]


def test_candidate_anomaly_first_seen_reward():
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")])

    assert components["candidate_anomaly"] == PPO_REWARD_CONFIG["candidate_anomaly"]


def test_same_candidate_repeated_has_no_extra_reward():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)

    assert components["candidate_anomaly"] == 0.0


def test_confirmed_first_discovery_reward_is_one():
    _, components = _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")])

    assert components["confirmed_anomaly"] == PPO_REWARD_CONFIG["confirmed_anomaly_first"]


def test_candidate_to_confirmed_upgrade_reward():
    history = {}
    _, first = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, second = _reward(anomalies=[_anomaly("R01", "confirmed", "r01-a")], history=history)

    assert first["candidate_anomaly"] == PPO_REWARD_CONFIG["candidate_anomaly"]
    assert second["confirmed_anomaly"] == PPO_REWARD_CONFIG["confirmed_anomaly_upgrade"]


def test_confidence_alone_does_not_get_confirmed_reward():
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a", confidence=0.99)])

    assert components["candidate_anomaly"] == PPO_REWARD_CONFIG["candidate_anomaly"]
    assert components["confirmed_anomaly"] == 0.0


def test_same_confirmed_anomaly_repeated_has_no_extra_reward():
    history = {}
    _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")], history=history)

    assert components["confirmed_anomaly"] == 0.0


def test_normal_repeat_action_target_penalty():
    history = {}
    _reward(history=history)
    _, components = _reward(history=history)

    assert components["repeat_penalty"] <= PPO_REWARD_CONFIG["repeat_action_target"]


def test_r01_pending_reproduction_repeat_penalty_is_softened():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)

    assert PPO_REWARD_CONFIG["repeat_action_target"] < components["repeat_penalty"] <= PPO_REWARD_CONFIG["pending_reproduction_repeat"]


def test_r01_confirmed_then_repeat_gets_normal_penalty():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _reward(anomalies=[_anomaly("R01", "confirmed", "r01-a")], history=history)
    _, components = _reward(history=history)

    assert components["repeat_penalty"] <= PPO_REWARD_CONFIG["repeat_action_target"]


def test_invalid_action_penalty():
    _, components = _reward(action=_action(invalid=True))

    assert components["invalid_penalty"] == PPO_REWARD_CONFIG["invalid_action"]


def test_timeout_without_anomaly_penalty():
    _, components = _reward(action=_action(timeout=True))

    assert components["timeout_penalty"] == PPO_REWARD_CONFIG["timeout"]


def test_confirmed_r15_timeout_reward_and_cost():
    reward, components = _reward(action=_action(timeout=True), anomalies=[_anomaly("R15", "confirmed", "r15-a")])

    assert components["confirmed_anomaly"] == PPO_REWARD_CONFIG["confirmed_anomaly_first"]
    assert components["timeout_penalty"] == PPO_REWARD_CONFIG["timeout"]
    assert reward > 0.0


def test_no_effect_normal_action_penalty():
    same = _obs()
    _, components = _reward(before=same, after=same)

    assert components["no_effect_penalty"] == PPO_REWARD_CONFIG["no_effect"]


def test_r01_no_effect_candidate_avoids_no_effect_penalty():
    same = _obs()
    _, components = _reward(before=same, after=same, anomalies=[_anomaly("R01", "candidate", "r01-a")])

    assert components["no_effect_penalty"] == 0.0
    assert components["candidate_anomaly"] == PPO_REWARD_CONFIG["candidate_anomaly"]


def test_flow_progress_reward():
    after = _obs("http://app.test/detail", flow_progress="detail")
    _, components = _reward(after=after)

    assert components["flow_progress"] == PPO_REWARD_CONFIG["flow_progress"]


def test_same_flow_repeated_gets_no_flow_reward():
    history = {}
    after = _obs("http://app.test/detail", flow_progress="detail")
    _reward(after=after, history=history)
    _, components = _reward(after=after, history=history)

    assert components["flow_progress"] == 0.0


def test_reward_positive_cap():
    anomalies = [
        _anomaly("R13", "confirmed", "r13-a"),
        _anomaly("R14", "confirmed", "r14-a", type="network-failure"),
    ]
    reward, components = _reward(after=_obs("http://app.test/detail", flow_progress="detail"), anomalies=anomalies)

    assert components["positive_capped"] <= PPO_REWARD_CONFIG["positive_cap"]
    assert reward <= PPO_REWARD_CONFIG["positive_cap"]


def test_ground_truth_metadata_does_not_change_reward():
    before = _obs(candidates=[_candidate(data_bug_id="SECRET", catalog_bug_id_matches=["BUG"])])
    after = _obs(candidates=[_candidate(data_bug_id="SECRET", catalog_bug_id_matches=["BUG"])], text_len=20)
    plain_reward, plain_components = _reward(before=_obs(), after=_obs(text_len=20))
    gt_reward, gt_components = _reward(before=before, after=after)

    assert gt_reward == plain_reward
    assert gt_components["candidate_anomaly"] == plain_components["candidate_anomaly"]
