from dqn_error_discovery.runners.train_dqn_error_detector import (
    DQN_REWARD_CONFIG,
    calculate_dqn_reward,
    dqn_repeat_diagnostic,
)


def _candidate(bid="A", **extra):
    item = {
        "bid": bid,
        "name": bid,
        "text": bid,
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "enabled": True,
        "bbox": [0, 0, 10, 10],
    }
    item.update(extra)
    return item


def _obs(url="http://app.test/page", *, candidates=None, **page_extra):
    page = {
        "url": url,
        "title": "Page",
        "quantity": 1,
        "viewport_type": "desktop",
        "viewport_width": 1280,
        "viewport_height": 720,
        "scroll_y": 0,
        "page_text_length": 100,
    }
    page.update(page_extra)
    return {
        "page_state": page,
        "candidate_elements": candidates if candidates is not None else [_candidate("A"), _candidate("B"), _candidate("C")],
        "runtime_signals": {},
    }


def _action(action_type="click_element", candidate_index=0):
    return {"action_type": action_type, "candidate_index": candidate_index}


def _anomaly(rule_id="R01", status="candidate", signature="r01-a", target_id="A"):
    return {
        "rule_id": rule_id,
        "status": status,
        "signature": signature,
        "type": "button-no-response",
        "confirmation_policy": "reproduce",
        "action_type": "click_element",
        "target_id": target_id,
        "evidence": {"rule_id": rule_id, "clicked_bid": target_id},
    }


def _reward(history, action, *, before=None, after=None, anomalies=None):
    before = before or _obs()
    after = after or before
    return calculate_dqn_reward(before, after, action, anomalies or [], history)


def _diag(history, action_target, *, components=None, before=None, after=None, action=None, anomalies=None):
    before = before or _obs()
    after = after or before
    action = action or _action()
    return dqn_repeat_diagnostic(
        history=history,
        before_observation=before,
        after_observation=after,
        action=action,
        anomalies=anomalies or [],
        reward_components=components or {},
        action_target_key=action_target,
    )


def test_immediate_repeat_a_a():
    history = {"dqn_action_target_history": ["A"]}
    diag = _diag(history, "A")
    assert diag["immediate_repeat"] is True
    assert diag["unproductive_repeat"] is True


def test_recent_repeat_a_b_a():
    history = {"dqn_action_target_history": ["A", "B"]}
    diag = _diag(history, "A")
    assert diag["recent_window_repeat"] is True
    assert diag["immediate_repeat"] is False


def test_cycle_length_2_a_b_a_b():
    history = {"dqn_action_target_history": ["A", "B", "A"]}
    diag = _diag(history, "B")
    assert diag["cycle_repeat"] is True
    assert diag["cycle_length"] == 2


def test_cycle_length_3_a_b_c_a_b_c():
    history = {"dqn_action_target_history": ["A", "B", "C", "A", "B"]}
    diag = _diag(history, "C")
    assert diag["cycle_repeat"] is True
    assert diag["cycle_length"] == 3


def test_no_repeat_a_b_c_d():
    history = {"dqn_action_target_history": ["A", "B", "C"]}
    diag = _diag(history, "D")
    assert diag["immediate_repeat"] is False
    assert diag["recent_window_repeat"] is False
    assert diag["cycle_repeat"] is False


def test_click_different_targets_not_same_action_target_repeat():
    history = {}
    same = _obs()
    _reward(history, _action("click_element", 0), before=same, after=same)
    _, components = _reward(history, _action("click_element", 1), before=same, after=same)
    assert components["repeat_action_penalty"] == 0.0


def test_click_a_b_a_recent_repeat():
    history = {}
    same = _obs()
    _reward(history, _action("click_element", 0), before=same, after=same)
    _reward(history, _action("click_element", 1), before=same, after=same)
    _, components = _reward(history, _action("click_element", 0), before=same, after=same)
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]
    assert history["last_repeat_diagnostic"]["recent_window_repeat"] is True


def test_functional_progress_exempts_repeated_action_from_unproductive_penalty():
    history = {"dqn_action_target_history": ["click_element:bid:A"]}
    before = _obs(quantity=1)
    after = _obs(quantity=2)
    _, components = _reward(history, _action("click_element", 0), before=before, after=after)
    assert components["repeat_action_penalty"] == 0.0
    assert history["last_repeat_diagnostic"]["productive_transition"] is True


def test_new_eligible_causal_candidate_is_productive():
    history = {"dqn_action_target_history": ["click_element:bid:A"]}
    same = _obs()
    _, components = _reward(history, _action("click_element", 0), before=same, after=same, anomalies=[_anomaly()])
    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert components["repeat_action_penalty"] == 0.0
    assert history["last_repeat_diagnostic"]["productive_transition"] is True


def test_candidate_to_confirmed_is_useful_reproduction():
    history = {}
    same = _obs()
    _reward(history, _action("click_element", 0), before=same, after=same, anomalies=[_anomaly()])
    _, components = _reward(history, _action("click_element", 0), before=same, after=same, anomalies=[_anomaly(status="confirmed")])
    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["pending_reproduction_repeat"]
    assert history["last_repeat_diagnostic"]["useful_reproduction_repeat"] is True


def test_pending_reproduction_exact_target_with_confirmation_progress_is_useful():
    history = {"dqn_action_target_history": ["click_element:bid:A"], "pending_reproduction_action_targets": {"click_element:bid:A"}}
    same = _obs()
    _, components = _reward(history, _action("click_element", 0), before=same, after=same, anomalies=[_anomaly(status="confirmed")])
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["pending_reproduction_repeat"]


def test_pending_candidate_wrong_target_has_no_useful_exemption():
    history = {"dqn_action_target_history": ["click_element:bid:B"], "pending_reproduction_action_targets": {"click_element:bid:A"}}
    same = _obs()
    _, components = _reward(history, _action("click_element", 1), before=same, after=same, anomalies=[_anomaly(target_id="A")])
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]
    assert history["last_repeat_diagnostic"]["useful_reproduction_repeat"] is False


def test_same_reproduction_repeated_with_no_progress_is_unproductive():
    history = {}
    same = _obs()
    _reward(history, _action("click_element", 0), before=same, after=same, anomalies=[_anomaly()])
    _, components = _reward(history, _action("click_element", 0), before=same, after=same, anomalies=[_anomaly()])
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["pending_reproduction_repeat"]
    assert history["last_repeat_diagnostic"]["useful_reproduction_repeat"] is True


def test_viewport_cycle_without_new_progress_is_unproductive():
    history = {"dqn_action_target_history": ["change_viewport_mobile:0", "change_viewport_desktop:0", "change_viewport_mobile:0"]}
    before = _obs(viewport_type="mobile", viewport_width=390)
    after = _obs(viewport_type="desktop", viewport_width=1280)
    _, components = _reward(history, _action("change_viewport_desktop"), before=before, after=after)
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]
    assert history["last_repeat_diagnostic"]["cycle_repeat"] is True


def test_scroll_down_up_cycle_without_new_progress_is_unproductive():
    history = {"dqn_action_target_history": ["scroll_down:0", "scroll_up:0", "scroll_down:0"]}
    before = _obs(scroll_y=400)
    after = _obs(scroll_y=0)
    _, components = _reward(history, _action("scroll_up"), before=before, after=after)
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]


def test_scroll_reveals_new_action_revealed_candidate_is_productive():
    history = {"dqn_action_target_history": ["scroll_down:0"]}
    before = _obs(scroll_y=0, page_text_length=100)
    after = _obs(scroll_y=300, page_text_length=300)
    anomaly = _anomaly("R05", signature="r05-a", target_id="panel")
    _, components = _reward(history, _action("scroll_down"), before=before, after=after, anomalies=[anomaly])
    assert components["candidate_anomaly"] == DQN_REWARD_CONFIG["candidate_anomaly"]
    assert components["repeat_action_penalty"] == 0.0


def test_immediate_and_cycle_apply_repeat_penalty_once():
    history = {"dqn_action_target_history": ["noop:0", "noop:0"]}
    _, components = _reward(history, _action("noop"), before=_obs(), after=_obs())
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]


def test_recent_and_cycle_apply_repeat_penalty_once():
    history = {}
    same = _obs()
    _reward(history, _action("click_element", 0), before=same, after=same)
    _reward(history, _action("click_element", 1), before=same, after=same)
    _reward(history, _action("click_element", 0), before=same, after=same)
    _, components = _reward(history, _action("click_element", 1), before=same, after=same)
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]


def test_raw_observation_only_change_does_not_exempt_cycle():
    history = {"dqn_action_target_history": ["click_element:bid:A", "click_element:bid:B", "click_element:bid:A"]}
    before = _obs(page_text_length=100)
    after = _obs(page_text_length=200)
    _, components = _reward(history, _action("click_element", 1), before=before, after=after)
    assert components["repeat_action_penalty"] == DQN_REWARD_CONFIG["repeat_action_target"]


def test_ground_truth_metadata_no_effect_on_repeat():
    base_history = {"dqn_action_target_history": ["click_element:bid:A"]}
    gt_history = {"dqn_action_target_history": ["click_element:bid:A"]}
    plain = _obs()
    with_gt = _obs(candidates=[_candidate("A", data_bug_id="BUG"), _candidate("B")])
    _, base = _reward(base_history, _action("click_element", 0), before=plain, after=plain)
    _, gt = _reward(gt_history, _action("click_element", 0), before=with_gt, after=with_gt)
    assert base["repeat_action_penalty"] == gt["repeat_action_penalty"]
