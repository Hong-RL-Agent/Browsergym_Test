import numpy as np

from dqn_error_discovery.agents.dqn_agent import ReplayBuffer, Transition
from dqn_error_discovery.runners.train_dqn_error_detector import (
    DQN_REWARD_CONFIG,
    _is_productive_r10_confirmation_reproduction,
    calculate_dqn_reward,
    initialize_dqn_candidate_reward_baseline,
)


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


def _obs(url: str = "http://app.test/product/1", *, text_len: int = 10, candidates: list[dict] | None = None, **page_extra) -> dict:
    page_state = {
        "url": url,
        "title": "Product",
        "page_text_length": text_len,
        "dom_node_count": 1,
        "quantity": 1,
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


def _r10_action_replay(status: str = "candidate") -> dict:
    return {
        "type": "api-ui-mismatch",
        "rule_id": "R10",
        "status": status,
        "confidence": 1.0,
        "signature": "r10-replay-a",
        "logical_key": "R10|api-ui-mismatch|/api/test|points",
        "confirmation_policy": "reproduce",
        "action_type": "click_element",
        "target_id": "buy",
        "reproduction_strategy": "ACTION_REPLAY",
        "reproduction_action_type": "click_element",
        "reproduction_target_identity": "bid:buy",
        "requires_action_reproduction": True,
        "evidence": {
            "rule_id": "R10",
            "source": "network_ui_semantic",
            "relation": "NOT_EQUAL",
            "pair_confidence": 1.0,
            "ui_settled": True,
        },
    }


def _reward(before=None, after=None, action=None, anomalies=None, history=None):
    return calculate_dqn_reward(
        before or _obs(),
        after or _obs(text_len=20),
        action or _action(),
        anomalies or [],
        history if history is not None else {},
    )


def test_new_local_state_reward():
    _, components = _reward()

    assert components["new_local_state"] == DQN_REWARD_CONFIG["new_local_state"]


def test_confirmed_r10_action_replay_is_positive_and_suppresses_contradictory_penalties():
    history = {}
    candidate = _r10_action_replay()
    initialize_dqn_candidate_reward_baseline(history, [candidate])
    history["seen_candidates"] = {"r10-replay-a"}
    confirmed = _r10_action_replay(status="confirmed")
    reward, components = _reward(
        before=_obs(text_len=10),
        after=_obs(text_len=10),
        action=_action(candidate_index=0),
        anomalies=[confirmed],
        history=history,
    )
    assert reward > 0
    assert _is_productive_r10_confirmation_reproduction(
        anomalies=[confirmed],
        action=_action(candidate_index=0),
        history={"last_confirmation_confirmed": True},
        reward_components={"confirmation_upgrade": DQN_REWARD_CONFIG["confirmation_upgrade"]},
        no_effect_diagnostic={"meaningful_action_effect": False},
    ) is True
    assert reward > 0


def test_unproductive_repeat_still_keeps_repeat_and_no_effect_penalties():
    history = {"dqn_action_target_history": ["click_element:bid:buy"]}
    reward, components = _reward(
        before=_obs(text_len=10),
        after=_obs(text_len=10),
        action=_action(candidate_index=0),
        anomalies=[],
        history=history,
    )
    assert reward < 0
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_repeated_local_state_penalty():
    history = {}
    _reward(history=history)
    _, components = _reward(history=history)

    assert components["new_local_state"] == 0.0
    assert components["repeat_state_penalty"] == DQN_REWARD_CONFIG["repeat_state"]


def test_new_local_action_target_reward():
    _, components = _reward()

    assert components["new_local_action_target"] == DQN_REWARD_CONFIG["new_local_action_target"]


def test_candidate_first_seen_reward():
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")])

    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]


def test_candidate_repeated_no_extra_reward():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)

    assert components["candidate_anomaly"] == 0.0


def test_confirmed_first_seen_reward():
    _, components = _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")])

    assert components["confirmed_anomaly"] == DQN_REWARD_CONFIG["confirmed_anomaly"]


def test_candidate_to_confirmed_upgrade():
    history = {}
    _, first = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, second = _reward(anomalies=[_anomaly("R01", "confirmed", "r01-a")], history=history)

    assert first["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert second["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]


def test_confidence_candidate_does_not_get_confirmed_reward():
    _, components = _reward(anomalies=[_anomaly("R01", "candidate", "r01-a", confidence=0.99)])

    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert components["confirmed_anomaly"] == 0.0


def test_same_confirmed_anomaly_repeated_gets_repeat_penalty():
    history = {}
    _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")], history=history)

    assert components["confirmed_anomaly"] == 0.0
    assert components["repeat_anomaly_penalty"] == DQN_REWARD_CONFIG["repeat_anomaly"]


def test_r01_pending_reproduction_wrong_action_does_not_get_useful_repeat_action():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, components = _reward(
        action=_action("scroll_down"),
        anomalies=[_anomaly("R01", "candidate", "r01-a")],
        history=history,
    )

    assert components["confirmation_upgrade"] == 0.0
    assert components["repeat_action_penalty"] == 0.0
    assert components["repeat_state_penalty"] == DQN_REWARD_CONFIG["repeat_state"]


def test_r01_becomes_confirmed_upgrade_reward():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R01", "confirmed", "r01-a")], history=history)

    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]


def test_r01_confirmed_then_repeat_gets_repeat_action_and_anomaly_penalty():
    history = {}
    _reward(anomalies=[_anomaly("R01", "candidate", "r01-a")], history=history)
    _reward(anomalies=[_anomaly("R01", "confirmed", "r01-a")], history=history)
    _, components = _reward(anomalies=[_anomaly("R01", "confirmed", "r01-a")], history=history)

    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]
    assert components["repeat_anomaly_penalty"] == DQN_REWARD_CONFIG["repeat_anomaly"]


def test_general_no_effect_penalty():
    same = _obs()
    _, components = _reward(before=same, after=same)

    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_r01_candidate_no_response_keeps_no_effect_penalty():
    same = _obs()
    _, components = _reward(before=same, after=same, anomalies=[_anomaly("R01", "candidate", "r01-a")])

    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_invalid_action_penalty():
    _, components = _reward(action=_action(invalid=True))

    assert components["invalid_penalty"] == DQN_REWARD_CONFIG["invalid_action"]


def test_action_error_penalty():
    _, components = _reward(action=_action(failed=True))

    assert components["action_error_penalty"] == DQN_REWARD_CONFIG["action_error"]


def test_timeout_without_anomaly_penalty():
    _, components = _reward(action=_action(timeout=True))

    assert components["timeout_penalty"] == DQN_REWARD_CONFIG["timeout"]


def test_new_confirmed_r15_timeout_remains_positive():
    reward, components = _reward(action=_action(timeout=True), anomalies=[_anomaly("R15", "confirmed", "r15-a")])

    assert components["confirmed_anomaly"] == DQN_REWARD_CONFIG["confirmed_anomaly"]
    assert components["timeout_penalty"] == DQN_REWARD_CONFIG["timeout"]
    assert reward > 0.0


def test_repeated_confirmed_r15_penalized():
    history = {}
    _reward(action=_action(timeout=True), anomalies=[_anomaly("R15", "confirmed", "r15-a")], history=history)
    reward, components = _reward(action=_action(timeout=True), anomalies=[_anomaly("R15", "confirmed", "r15-a")], history=history)

    assert components["confirmed_anomaly"] == 0.0
    assert components["repeat_anomaly_penalty"] == DQN_REWARD_CONFIG["repeat_anomaly"]
    assert components["timeout_penalty"] == DQN_REWARD_CONFIG["timeout"]
    assert reward < 0.0


def test_flow_progress_reward():
    _, components = _reward(after=_obs("http://app.test/cart", flow_progress="cart"))

    assert components["flow_progress"] == DQN_REWARD_CONFIG["flow_progress"]


def test_new_url_only_has_no_global_url_reward():
    neutral = [_candidate("plain", "Plain")]
    _, components = _reward(
        before=_obs(candidates=neutral),
        after=_obs("http://app.test/another", text_len=20, candidates=neutral),
    )

    assert "new_url" not in components
    assert components["flow_progress"] == 0.0


def test_reward_cap():
    anomalies = [_anomaly("R13", "confirmed", "a"), _anomaly("R14", "confirmed", "b")]
    reward, components = _reward(after=_obs("http://app.test/cart", flow_progress="cart"), anomalies=anomalies)

    assert components["positive_capped"] <= DQN_REWARD_CONFIG["positive_cap"]
    assert reward <= DQN_REWARD_CONFIG["positive_cap"]


def test_ground_truth_metadata_does_not_change_reward():
    plain_reward, plain_components = _reward()
    before = _obs(candidates=[_candidate(data_bug_id="BUG", catalog_bug_id_matches=["BUG"])])
    after = _obs(candidates=[_candidate(data_bug_id="BUG", catalog_bug_id_matches=["BUG"])], text_len=20)
    gt_reward, gt_components = _reward(before=before, after=after)

    assert gt_reward == plain_reward
    assert gt_components["candidate_anomaly"] == plain_components["candidate_anomaly"]


def test_missing_signature_anomaly_does_not_crash_or_reward():
    reward, components = _reward(anomalies=[{"rule_id": "R13", "status": "confirmed", "confidence": 1.0}])

    assert components["confirmed_anomaly"] == 0.0
    assert reward >= 0.0


def test_replay_transition_stores_final_reward():
    reward, _ = _reward(anomalies=[_anomaly("R13", "confirmed", "r13-a")])
    replay = ReplayBuffer(4)
    state = np.zeros(3, dtype=np.float32)
    next_state = np.ones(3, dtype=np.float32)
    mask = np.ones(2, dtype=np.float32)

    replay.add(Transition(state, 1, reward, next_state, False, mask))

    assert len(replay) == 1
    assert replay.items[0].reward == reward
