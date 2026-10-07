from pathlib import Path

import numpy as np

from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.confirmation_forced_reproduction_diagnostic import ACTION_SOURCE, diagnostic_forced_action, mask_rejection_trace

SPACE = V3ActionSpace(32)


def candidate(**updates):
    row = {"rule_id": "R01", "signature": "runtime-candidate", "status": "candidate", "pending_confirmation": True, "confirmation_policy": "reproduce", "action_type": "click_element", "target_id": "runtime-target"}
    row.update(updates)
    return row


def observation():
    return {"candidate_elements": [{"bid": "runtime-target", "role": "button", "tag": "button", "visible": True, "enabled": True}]}


def valid_mask():
    mask = np.zeros(640, dtype=np.float32)
    mask[SPACE.encode("click_element", 0)] = 1.0
    return mask


def test_no_active_candidate_has_no_forced_action():
    assert diagnostic_forced_action(None, observation(), {}, valid_mask())[0] is None


def test_not_pending_has_no_forced_action():
    assert diagnostic_forced_action(candidate(status="ignored", pending_confirmation=False, confirmation_policy=""), observation(), {}, valid_mask())[0] is None


def test_missing_origin_action_has_no_forced_action():
    assert diagnostic_forced_action(candidate(action_type=""), observation(), {}, valid_mask())[0] is None


def test_mapping_failure_has_no_forced_action():
    assert diagnostic_forced_action(candidate(target_id="absent"), observation(), {}, valid_mask())[0] is None


def test_invalid_mask_has_no_forced_action():
    assert diagnostic_forced_action(candidate(), observation(), {}, np.zeros(640, dtype=np.float32))[0] is None


def test_valid_context_generates_runtime_origin_action():
    action, details = diagnostic_forced_action(candidate(), observation(), {}, valid_mask())
    assert action is not None
    assert action["action_type"] == "click_element"
    assert action["candidate_index"] == 0
    assert action["action_id"] == SPACE.encode("click_element", 0)
    assert action["action_source"] == ACTION_SOURCE
    assert action["failed"] is False
    assert details["mask_valid"] is True


def test_runner_source_has_no_hardcoded_target_endpoint_or_gt_runtime_access():
    source = Path(__file__).parents[1] / "v3" / "confirmation_forced_reproduction_diagnostic.py"
    text = source.read_text(encoding="utf-8").lower()
    assert all(token not in text for token in ("buy buy", "/api/product/buy", '"points"', "bug_catalog"))
    assert '"gt_runtime_access": false' in text


def mask_candidate(**updates):
    row = {"bid": "x", "visible": True, "enabled": True, "bbox": [0, 0, 20, 10], "clickable": True, "role": "button", "tag": "button"}
    row.update(updates)
    return row


def test_mask_reason_not_visible_matches_production_mask():
    row = mask_candidate(visible=False)
    assert mask_rejection_trace("click_element", row)["reason"] == "MASK_REJECT_NOT_VISIBLE"
    assert SPACE.build_action_mask({"candidate_elements": [row]})[SPACE.encode("click_element", 0)] == 0


def test_mask_reason_disabled_matches_production_mask():
    row = mask_candidate(enabled=False)
    assert mask_rejection_trace("click_element", row)["reason"] == "MASK_REJECT_DISABLED"
    assert SPACE.build_action_mask({"candidate_elements": [row]})[SPACE.encode("click_element", 0)] == 0


def test_mask_reason_not_click_compatible_matches_production_mask():
    row = mask_candidate(clickable=False, role="", tag="div")
    assert mask_rejection_trace("click_element", row)["reason"] == "MASK_REJECT_ACTION_TYPE_MISMATCH"
    assert SPACE.build_action_mask({"candidate_elements": [row]})[SPACE.encode("click_element", 0)] == 0


def test_mask_reason_zero_bbox_matches_production_mask():
    row = mask_candidate(bbox=[0, 0, 0, 10])
    assert mask_rejection_trace("click_element", row)["reason"] == "MASK_REJECT_ZERO_BBOX"
    assert SPACE.build_action_mask({"candidate_elements": [row]})[SPACE.encode("click_element", 0)] == 0


def test_mask_reason_action_type_incompatible_matches_production_mask():
    row = mask_candidate()
    assert mask_rejection_trace("fill_input", row)["reason"] == "MASK_REJECT_ACTION_TYPE_MISMATCH"
    assert SPACE.build_action_mask({"candidate_elements": [row]})[SPACE.encode("fill_input", 0)] == 0


def test_valid_click_target_is_actionable_and_matches_production_mask():
    row = mask_candidate()
    assert mask_rejection_trace("click_element", row)["actionable"] is True
    assert SPACE.build_action_mask({"candidate_elements": [row]})[SPACE.encode("click_element", 0)] == 1
