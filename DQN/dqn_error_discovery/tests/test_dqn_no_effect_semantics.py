from dqn_error_discovery.runners.train_dqn_error_detector import (
    DQN_REWARD_CONFIG,
    calculate_dqn_reward,
    dqn_no_effect_diagnostic,
)


def _candidate(bid="target", **extra):
    item = {
        "bid": bid,
        "name": "Target",
        "text": "Target",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "enabled": True,
        "bbox": [10, 10, 100, 30],
    }
    item.update(extra)
    return item


def _obs(url="http://app.test/page", *, candidates=None, runtime=None, **page_extra):
    page = {
        "url": url,
        "title": "Page",
        "dom_node_count": 10,
        "page_text_length": 100,
        "quantity": 1,
        "viewport_type": "desktop",
        "viewport_width": 1280,
        "viewport_height": 720,
        "scroll_y": 0,
    }
    page.update(page_extra)
    return {
        "page_state": page,
        "candidate_elements": candidates if candidates is not None else [_candidate()],
        "runtime_signals": runtime or {},
    }


def _action(action_type="click_element", candidate_index=0, **extra):
    item = {"action_type": action_type, "candidate_index": candidate_index}
    item.update(extra)
    return item


def _anomaly(rule_id="R01", status="candidate", signature="r01-a", target_id="target", **extra):
    item = {
        "type": "button-no-response",
        "rule_id": rule_id,
        "status": status,
        "signature": signature,
        "confirmation_policy": "reproduce" if rule_id in {"R01", "R02", "R04", "R05", "R06"} else "strong-evidence",
        "action_type": "click_element",
        "target_id": target_id,
        "evidence": {"rule_id": rule_id, "clicked_bid": target_id},
    }
    item.update(extra)
    return item


def test_click_no_response_new_r01_candidate_and_no_effect_can_coexist():
    same = _obs()
    _, components = calculate_dqn_reward(same, same, _action(), [_anomaly()], {})
    diag = dqn_no_effect_diagnostic(same, same, _action())
    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]
    assert diag["no_effect"] is True


def test_same_r01_repeated_has_no_candidate_reward_no_effect_and_repeat():
    history = {}
    same = _obs()
    calculate_dqn_reward(same, same, _action(), [_anomaly("R01", "candidate", "r01-a")], history)
    _, components = calculate_dqn_reward(same, same, _action(), [_anomaly("R01", "candidate", "r01-a")], history)
    assert components["candidate_anomaly"] == 0.0
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["pending_reproduction_repeat"]


def test_r01_reproduction_confirmed_keeps_no_effect_and_confirmed_reward():
    history = {}
    same = _obs()
    calculate_dqn_reward(same, same, _action(), [_anomaly("R01", "candidate", "r01-a")], history)
    _, components = calculate_dqn_reward(same, same, _action(), [_anomaly("R01", "confirmed", "r01-a")], history)
    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_unproductive_r01_repetition_has_no_confirmed_reward():
    history = {}
    same = _obs()
    calculate_dqn_reward(same, same, _action(), [_anomaly("R01", "candidate", "r01-a")], history)
    _, components = calculate_dqn_reward(same, same, _action(), [_anomaly("R01", "candidate", "r01-a")], history)
    assert components["confirmed_anomaly"] == 0.0
    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_candidate_anomaly_does_not_exempt_no_effect():
    same = _obs()
    _, components = calculate_dqn_reward(same, same, _action(), [_anomaly("R05", target_id="panel")], {})
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_click_url_change_is_meaningful_effect():
    diag = dqn_no_effect_diagnostic(_obs(), _obs("http://app.test/next"), _action())
    assert diag["meaningful_action_effect"] is True
    assert diag["no_effect"] is False


def test_click_opens_modal_is_meaningful_effect():
    before = _obs(modal_state="closed")
    after = _obs(modal_state="open", candidates=[_candidate(), _candidate("dialog", role="dialog")])
    diag = dqn_no_effect_diagnostic(before, after, _action())
    assert diag["no_effect"] is False


def test_click_functional_state_change_is_meaningful_effect():
    diag = dqn_no_effect_diagnostic(_obs(quantity=1), _obs(quantity=2), _action())
    assert diag["no_effect"] is False


def test_only_page_text_length_change_is_no_effect_for_click():
    diag = dqn_no_effect_diagnostic(_obs(page_text_length=100), _obs(page_text_length=200), _action())
    assert diag["no_effect"] is True
    assert "RAW_OBSERVATION_CHANGE_IGNORED" in diag["no_effect_reasons"]


def test_only_candidate_ordering_change_is_no_effect_for_click():
    before = _obs(candidates=[_candidate("a"), _candidate("b")])
    after = _obs(candidates=[_candidate("b"), _candidate("a")])
    assert dqn_no_effect_diagnostic(before, after, _action())["no_effect"] is True


def test_only_bbox_change_is_no_effect_for_click():
    before = _obs(candidates=[_candidate("a", bbox=[0, 0, 10, 10])])
    after = _obs(candidates=[_candidate("a", bbox=[0, 100, 10, 10])])
    assert dqn_no_effect_diagnostic(before, after, _action())["no_effect"] is True


def test_scroll_actual_move_is_not_no_effect():
    before = _obs(scroll_y=0, page_text_length=100)
    after = _obs(scroll_y=400, page_text_length=200)
    assert dqn_no_effect_diagnostic(before, after, _action("scroll_down"))["no_effect"] is False


def test_scroll_boundary_no_movement_is_no_effect():
    same = _obs(scroll_y=400)
    assert dqn_no_effect_diagnostic(same, same, _action("scroll_down"))["no_effect"] is True


def test_viewport_mode_change_is_not_no_effect():
    before = _obs(viewport_type="desktop", viewport_width=1280)
    after = _obs(viewport_type="mobile", viewport_width=390)
    assert dqn_no_effect_diagnostic(before, after, _action("change_viewport_mobile"))["no_effect"] is False


def test_viewport_already_same_is_no_effect():
    same = _obs(viewport_type="mobile", viewport_width=390)
    assert dqn_no_effect_diagnostic(same, same, _action("change_viewport_mobile"))["no_effect"] is True


def test_execution_error_is_separate_from_no_effect():
    same = _obs()
    diag = dqn_no_effect_diagnostic(same, same, _action(failed=True))
    _, components = calculate_dqn_reward(same, same, _action(failed=True), [], {})
    assert diag["execution_error"] is True
    assert diag["no_effect"] is False
    assert components["action_error_penalty"] == DQN_REWARD_CONFIG["action_error"]
    assert components["no_effect_penalty"] == 0.0


def test_r05_present_unrelated_click_no_effect():
    same = _obs()
    anomaly = _anomaly("R05", signature="r05-a", target_id="panel", evidence={"rule_id": "R05", "target_bid": "panel"})
    _, components = calculate_dqn_reward(same, same, _action(), [anomaly], {})
    assert components["candidate_anomaly"] == 0.0
    assert components["no_effect_penalty"] == DQN_REWARD_CONFIG["no_effect"]


def test_ground_truth_has_no_effect_on_no_effect_semantics():
    plain = _obs()
    with_gt = _obs(candidates=[_candidate(data_bug_id="BUG", catalog_bug_id_matches=["BUG"])])
    assert dqn_no_effect_diagnostic(plain, plain, _action())["no_effect"] == dqn_no_effect_diagnostic(with_gt, with_gt, _action())["no_effect"]
