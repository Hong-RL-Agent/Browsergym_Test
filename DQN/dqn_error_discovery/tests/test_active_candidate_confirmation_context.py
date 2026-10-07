import math

import numpy as np

from dqn_error_discovery.agents.confirmation_observation import (
    ACTIVE_ACTION_SLICE,
    ACTIVE_CANDIDATE_CONFIRMATION_CONTEXT_DIM,
    ACTIVE_CATEGORY_SLICE,
    ACTIVE_FLAG_SLICE,
    ACTIVE_TARGET_SLICE,
    RULE_IDS,
    build_active_candidate_confirmation_context,
)
from dqn_error_discovery.v3.action_space import V3ActionSpace


SPACE = V3ActionSpace(32)


def candidate(bid: str, text: str = "Buy Buy") -> dict:
    return {
        "bid": bid,
        "role": "button",
        "tag": "button",
        "text": text,
        "visible": True,
        "enabled": True,
        "clickable": True,
        "bbox": [0, 0, 100, 40],
        "set_of_marks": True,
    }


def active(rule_id: str = "R10", action_type: str = "click_element", target_id: str = "buy") -> dict:
    return {
        "rule_id": rule_id,
        "signature": f"{rule_id}|example|target:{target_id}",
        "status": "candidate",
        "lifecycle_status": "DETECTED",
        "pending_confirmation": True,
        "confirmation_policy": "reproduce",
        "action_type": action_type,
        "target_id": target_id,
        "evidence": {"rule_id": rule_id, "action_type": action_type, "target_bid": target_id},
    }


def build(row=None, candidates=None):
    obs = {"candidate_elements": candidates if candidates is not None else [candidate("buy")]}
    mask = SPACE.build_action_mask(obs)
    return build_active_candidate_confirmation_context({}, obs, SPACE, active_candidate=row, mask=mask)


def test_active_r10_candidate_encodes_category_action_target_and_flags():
    vector, meta = build(active())

    assert vector.shape == (ACTIVE_CANDIDATE_CONFIRMATION_CONTEXT_DIM,)
    assert vector[ACTIVE_CATEGORY_SLICE.start + RULE_IDS.index("R10")] == 1.0
    assert vector[ACTIVE_ACTION_SLICE.start + SPACE.action_types.index("click_element")] == 1.0
    assert vector[ACTIVE_TARGET_SLICE.start + 0] == 1.0
    assert meta["decoded_category"] == "R10"
    assert meta["decoded_origin_action"] == "click_element"
    assert meta["decoded_target_index"] == 0
    assert meta["decoded_flags"]["active_candidate_exists"] == 1
    assert meta["decoded_flags"]["pending_confirmation"] == 1
    assert meta["decoded_flags"]["independent_reproduction_required"] == 1
    assert meta["decoded_flags"]["origin_action_available"] == 1
    assert meta["decoded_flags"]["origin_target_mapped"] == 1
    assert meta["decoded_flags"]["active_target_actionable"] == 1


def test_no_active_candidate_is_neutral_zero_context():
    vector, meta = build(None)

    assert not np.count_nonzero(vector)
    assert meta["active_candidate_exists"] is False


def test_target_reorder_remaps_by_identity_not_stored_index():
    step1, meta1 = build(active(), [candidate("buy"), candidate("other", "Other")])
    step2, meta2 = build(active(), [candidate("other", "Other"), candidate("x"), candidate("buy")])

    assert step1[ACTIVE_TARGET_SLICE.start + 0] == 1.0
    assert step2[ACTIVE_TARGET_SLICE.start + 2] == 1.0
    assert meta1["decoded_target_index"] == 0
    assert meta2["decoded_target_index"] == 2


def test_missing_target_clears_target_vector_and_mapping_flag():
    vector, meta = build(active(), [candidate("other", "Other")])

    assert not np.count_nonzero(vector[ACTIVE_TARGET_SLICE])
    assert meta["decoded_target_index"] is None
    assert meta["decoded_flags"]["origin_target_mapped"] == 0
    assert meta["decoded_flags"]["active_target_actionable"] == 0


def test_non_r10_taxonomy_uses_same_generic_encoder():
    vector, meta = build(active("R06", "scroll_up", "panel"), [candidate("panel", "Panel")])

    assert vector[ACTIVE_CATEGORY_SLICE.start + RULE_IDS.index("R06")] == 1.0
    assert meta["decoded_category"] == "R06"
    assert meta["reproduction_strategy"] == "OBSERVATION_RECHECK"
    assert meta["decoded_origin_action"] == ""
    assert meta["expected_target_identity"] is None
    assert meta["decoded_flags"]["origin_action_available"] == 0
    assert meta["decoded_flags"]["origin_target_mapped"] == 0
    assert meta["decoded_flags"]["active_target_actionable"] == 0
    assert meta["decoded_flags"]["independent_reproduction_required"] == 1


def test_non_click_originating_action_is_encoded():
    vector, meta = build(active("R01", "press_enter", "search"), [candidate("search", "Search")])

    assert vector[ACTIVE_ACTION_SLICE.start + SPACE.action_types.index("press_enter")] == 1.0
    assert meta["decoded_origin_action"] == "press_enter"


def test_context_is_finite_and_fixed_size():
    vector, _ = build(active())

    assert vector.shape == (73,)
    assert math.isfinite(float(vector.sum()))
    assert not np.isnan(vector).any()
    assert not np.isinf(vector).any()
