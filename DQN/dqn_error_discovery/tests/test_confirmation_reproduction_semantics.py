import numpy as np
import pytest

from dqn_error_discovery.agents.confirmation_observation import build_dqn_confirmation_context
from dqn_error_discovery.agents.confirmation_reproduction import (
    ACTION_REPLAY,
    ACTION_REPRODUCTION,
    OBSERVATION_RECHECK,
    OBSERVATION_REPRODUCTION,
    ExpectedReproduction,
    action_semantics_matrix,
    derive_reproduction_context,
    expected_reproduction_for_anomaly,
    is_reproduction_instruction_executable,
    same_reproduction_context,
)
from dqn_error_discovery.runners.train_dqn_error_detector import DQN_REWARD_CONFIG, calculate_dqn_reward
from models.action_space import ActionSpace


SPACE = ActionSpace(4)


def _candidate(bid="A", role="button", tag="button"):
    return {"bid": bid, "role": role, "tag": tag, "clickable": True, "fillable": role == "textbox", "visible": True, "enabled": True, "bbox": [0, 0, 20, 20]}


def _obs(candidates=None, url="http://app.test/page"):
    return {"page_state": {"url": url}, "candidate_elements": candidates or [_candidate()], "runtime_signals": {}}


def _anomaly(rule_id, *, action_type="click_element", target_id="A", viewport=None, issue_type=None):
    evidence = {"rule_id": rule_id, "target_bid": target_id, "issue_type": issue_type or {"R01": "button-no-response", "R02": "broken-navigation", "R05": "layout-overlap", "R06": "layout-overflow"}.get(rule_id, "issue")}
    if viewport is not None:
        evidence["viewport"] = viewport
    return {"rule_id": rule_id, "type": evidence["issue_type"], "issue_type": evidence["issue_type"], "status": "candidate", "confirmation_policy": "reproduce", "action_type": action_type, "target_id": target_id, "evidence": evidence, "url_path": "/page", "signature": f"{rule_id}-{target_id}-{viewport}"}


def _reward(anomaly, history, action=None):
    return calculate_dqn_reward(_obs(), _obs(), action or {"action_type": "click_element", "candidate_index": 0}, [anomaly], history)


def test_action_semantics_matrix_matches_executor_vocabulary():
    rows = {row["action_type"]: row for row in action_semantics_matrix(SPACE)}
    assert len(rows) == 19
    assert rows["click_element"]["requires_candidate_target"]
    assert rows["fill_input"]["requires_candidate_target"]
    assert rows["press_enter"]["requires_candidate_target"]
    for action_type in ("scroll_down", "scroll_up", "noop", "change_viewport_mobile", "change_viewport_desktop", "inspect_dom", "inspect_network"):
        assert not rows[action_type]["requires_candidate_target"]
        assert not rows[action_type]["consumes_candidate_slot"]


@pytest.mark.parametrize("action_type", ["click_element", "fill_input", "press_enter"])
def test_target_action_types_require_a_candidate(action_type):
    instruction = ExpectedReproduction(action_type, "bid:A", ACTION_REPRODUCTION, rule_id="R01")
    result = is_reproduction_instruction_executable(instruction, action_space=SPACE, observation=_obs([_candidate("A", "textbox" if action_type != "click_element" else "button")]), mask=np.ones(SPACE.get_action_dim(), dtype=np.float32))
    assert result["semantically_executable"]


@pytest.mark.parametrize("action_type", ["scroll_down", "scroll_up", "noop", "change_viewport_mobile", "change_viewport_desktop"])
def test_candidate_independent_action_types_do_not_require_target(action_type):
    rows = {row["action_type"]: row for row in action_semantics_matrix(SPACE)}
    assert rows[action_type]["executable_without_target"]
    assert not rows[action_type]["usable_for_confirmation_reproduction"]


def test_candidate_independent_instruction_with_target_is_rejected():
    with pytest.raises(ValueError):
        ExpectedReproduction("scroll_down", "bid:A", ACTION_REPRODUCTION, rule_id="R06")


def test_r01_is_action_reproduction_and_executes_same_target():
    instruction = expected_reproduction_for_anomaly(_anomaly("R01"))
    assert instruction.mode == ACTION_REPRODUCTION
    assert instruction.action_type == "click_element"
    assert instruction.target_identity == "bid:A"
    assert instruction.requires_target


def test_r02_preserves_triggering_action_and_target():
    instruction = expected_reproduction_for_anomaly(_anomaly("R02", action_type="click_element", target_id="A"))
    assert instruction.mode == ACTION_REPRODUCTION
    assert instruction.action_type == "click_element"
    assert instruction.target_identity == "bid:A"


@pytest.mark.parametrize("rule_id", ["R05", "R06"])
def test_layout_rules_are_observation_reproduction_without_fake_action_target(rule_id):
    instruction = expected_reproduction_for_anomaly(_anomaly(rule_id, action_type="scroll_down"))
    assert instruction.mode == OBSERVATION_REPRODUCTION
    assert instruction.action_type is None
    assert instruction.target_identity is None
    assert not instruction.requires_target


def test_r05_does_not_generate_noop_or_scroll_target_instruction():
    instruction = expected_reproduction_for_anomaly(_anomaly("R05", action_type="noop"))
    assert instruction.action_type is None
    assert instruction.target_identity is None


def test_r06_does_not_generate_scroll_target_instruction():
    instruction = expected_reproduction_for_anomaly(_anomaly("R06", action_type="scroll_down"))
    assert instruction.action_type is None
    assert instruction.target_identity is None


def test_observation_reproduction_context_ignores_bbox_noise():
    left = _anomaly("R05", viewport=[0, 0, 1280, 720])
    right = _anomaly("R05", viewport=[35, 17, 1280, 720])
    assert same_reproduction_context(left, right)


def test_observation_reproduction_context_rejects_entity_or_viewport_change():
    left = _anomaly("R06", target_id="A", viewport=[0, 0, 375, 812])
    assert not same_reproduction_context(left, _anomaly("R06", target_id="B", viewport=[0, 0, 375, 812]))
    assert not same_reproduction_context(left, _anomaly("R06", target_id="A", viewport=[0, 0, 1280, 720]))


def test_r05_observation_reproduction_progresses_on_same_logical_evidence():
    history = {}
    anomaly = _anomaly("R05", viewport=[0, 0, 1280, 720])
    _reward(anomaly, history, action={"action_type": "scroll_down", "candidate_index": 0})
    _, components = _reward(dict(anomaly), history, action={"action_type": "noop", "candidate_index": 0})
    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert history["last_confirmation_progress"]
    assert history["last_confirmation_confirmed"]


def test_observation_context_mismatch_does_not_progress():
    history = {}
    _reward(_anomaly("R05", target_id="A", viewport=[0, 0, 1280, 720]), history)
    _, components = _reward(_anomaly("R05", target_id="A", viewport=[0, 0, 375, 812]), history)
    assert components["confirmation_upgrade"] == 0


def test_r01_same_click_target_progresses():
    history = {}
    anomaly = _anomaly("R01")
    _reward(anomaly, history)
    _, components = _reward(dict(anomaly), history)
    assert components["confirmation_upgrade"] == DQN_REWARD_CONFIG["confirmation_upgrade"]
    assert history["last_confirmation_progress"]
    assert history["last_confirmation_confirmed"]


def test_r01_wrong_target_does_not_progress():
    history = {}
    _reward(_anomaly("R01", target_id="A"), history)
    _, components = _reward(_anomaly("R01", target_id="B"), history)
    assert components["confirmation_upgrade"] == 0


def test_observation_pending_context_has_no_action_or_target_features():
    history = {"pending_reproduction_records": {"r05": {"signature": "r05", "logical_key": "R05|/page|layout-overlap|target:A", "rule_id": "R05", "confirmation_mode": OBSERVATION_REPRODUCTION, "action_target": "", "progress": 0.0, "attempt_count": 0}}}
    context, metadata = build_dqn_confirmation_context(history, _obs(), SPACE)
    assert metadata["pending_confirmation_exists"]
    assert metadata["confirmation_mode"] == OBSERVATION_REPRODUCTION
    assert metadata["expected_action_type"] == ""
    assert metadata["expected_target_identity"] is None
    assert metadata["expected_action_requires_target"] == 0
    assert np.all(context[16:69] == 0.0)
    assert context[0] == 1.0
    assert context[1 + 4] == 1.0


def test_action_reproduction_instructions_are_executable():
    for action_type in ("click_element", "fill_input", "press_enter"):
        instruction = ExpectedReproduction(action_type, "bid:A", ACTION_REPRODUCTION, rule_id="R01")
        candidate = _candidate("A", "button" if action_type == "click_element" else "textbox")
        mask = np.zeros(SPACE.get_action_dim(), dtype=np.float32)
        mask[SPACE.encode(action_type, 0)] = 1.0
        assert is_reproduction_instruction_executable(instruction, action_space=SPACE, observation=_obs([candidate]), mask=mask)["semantically_executable"]


def test_ground_truth_metadata_does_not_change_instruction():
    base = _anomaly("R05", target_id="A")
    changed = dict(base, ground_truth=True, data_bug_id="BUG-1", catalog_bug_id_matches=["BUG-1"])
    assert expected_reproduction_for_anomaly(base) == expected_reproduction_for_anomaly(changed)


def test_unsupported_action_target_has_no_fake_fallback():
    assert expected_reproduction_for_anomaly(_anomaly("R01", action_type="scroll_down")) is None


def test_observation_mode_does_not_create_action_bonus_or_action_instruction():
    instruction = expected_reproduction_for_anomaly(_anomaly("R06"))
    assert instruction.mode == OBSERVATION_REPRODUCTION
    assert instruction.action_type is None
    assert instruction.target_identity is None


def test_confirmation_threshold_is_not_encoded_in_semantics_helper():
    assert ExpectedReproduction("click_element", "bid:A", ACTION_REPRODUCTION, rule_id="R01").requires_target


def test_r06_separates_evidence_target_from_trigger_and_has_no_reproduction_target():
    row = _anomaly("R06", action_type="click_element", target_id="94")
    row.update(evidence_target_identity="bid:94", trigger_action_type="click_element", trigger_target_identity="bid:73")
    context = derive_reproduction_context(row)

    assert context.evidence_target == "bid:94"
    assert context.trigger_target == "bid:73"
    assert context.strategy == OBSERVATION_RECHECK
    assert context.reproduction_target is None
    assert not context.requires_action_reproduction


def test_r10_replays_runtime_trigger_target_not_evidence_target():
    row = _anomaly("R10", action_type="click_element", target_id="94")
    row.update(evidence_target_identity="bid:94", trigger_action_type="click_element", trigger_target_identity="bid:73")
    context = derive_reproduction_context(row)

    assert context.evidence_target == "bid:94"
    assert context.trigger_target == "bid:73"
    assert context.strategy == ACTION_REPLAY
    assert context.reproduction_action_type == "click_element"
    assert context.reproduction_target == "bid:73"
    assert context.requires_action_reproduction


def test_equal_evidence_and_trigger_targets_remain_separate_contract_fields():
    row = _anomaly("R01", target_id="73")
    row.update(evidence_target_identity="bid:73", trigger_action_type="click_element", trigger_target_identity="bid:73")
    context = derive_reproduction_context(row)
    assert context.evidence_target == context.reproduction_target == "bid:73"
    assert "evidence_target_identity" in context.as_dict()
    assert "reproduction_target_identity" in context.as_dict()


def test_observation_only_candidate_derives_observation_recheck():
    row = _anomaly("R06", action_type="", target_id="94")
    row.update(evidence_target_identity="bid:94", trigger_action_type=None, trigger_target_identity=None)
    context = derive_reproduction_context(row)
    assert context.strategy == OBSERVATION_RECHECK
    assert not context.requires_action_reproduction


def test_action_caused_candidate_derives_action_replay():
    row = _anomaly("R01", action_type="click_element", target_id="94")
    row.update(evidence_target_identity="bid:94", trigger_action_type="click_element", trigger_target_identity="bid:73")
    context = derive_reproduction_context(row)
    assert context.strategy == ACTION_REPLAY
    assert context.reproduction_target == "bid:73"
