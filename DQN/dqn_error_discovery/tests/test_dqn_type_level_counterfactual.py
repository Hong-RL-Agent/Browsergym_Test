import numpy as np

from dqn_error_discovery.runners.dqn_type_level_counterfactual import (
    FLAT_MAX,
    TYPE_MEAN_MAX,
    TYPE_MEDIAN_MAX,
    TYPE_TOPK_MEAN_MAX,
    offline_counterfactual_from_records,
    score_action_types,
    select_counterfactual_action,
)
from models.action_space import ActionSpace


def _space():
    return ActionSpace(max_candidates=32)


def _mask_for(space, pairs):
    mask = np.zeros(space.get_action_dim(), dtype=np.float32)
    for action_type, candidate_index in pairs:
        mask[space.encode(action_type, candidate_index)] = 1.0
    return mask


def test_flat_max_selects_highest_flattened_valid_action():
    space = _space()
    mask = _mask_for(space, [("click_element", 0), ("scroll_down", 0)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("click_element", 0)] = 10.0
    q[space.encode("scroll_down", 0)] = 1.0

    selected = select_counterfactual_action(q, mask, space, mode=FLAT_MAX)

    assert selected["action_type"] == "click_element"
    assert selected["action_id"] == space.encode("click_element", 0)


def test_type_mean_reduces_single_click_outlier_bias():
    space = _space()
    mask = _mask_for(space, [("click_element", 0), ("click_element", 1), ("scroll_down", 0)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("click_element", 0)] = 10.0
    q[space.encode("click_element", 1)] = -10.0
    q[space.encode("scroll_down", 0)] = 1.0

    flat = select_counterfactual_action(q, mask, space, mode=FLAT_MAX)
    type_mean = select_counterfactual_action(q, mask, space, mode=TYPE_MEAN_MAX)

    assert flat["action_type"] == "click_element"
    assert type_mean["action_type"] == "scroll_down"


def test_type_median_scores_by_type_median():
    space = _space()
    mask = _mask_for(space, [("click_element", 0), ("click_element", 1), ("click_element", 2), ("inspect_dom", 0)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("click_element", 0)] = 5.0
    q[space.encode("click_element", 1)] = -2.0
    q[space.encode("click_element", 2)] = -2.0
    q[space.encode("inspect_dom", 0)] = 0.0

    selected = select_counterfactual_action(q, mask, space, mode=TYPE_MEDIAN_MAX)

    assert selected["action_type"] == "inspect_dom"


def test_type_topk_mean_uses_top_k_within_type_for_score():
    space = _space()
    mask = _mask_for(space, [("click_element", 0), ("click_element", 1), ("click_element", 2), ("fill_input", 0)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("click_element", 0)] = 3.0
    q[space.encode("click_element", 1)] = 2.0
    q[space.encode("click_element", 2)] = -10.0
    q[space.encode("fill_input", 0)] = 2.1

    selected = select_counterfactual_action(q, mask, space, mode=TYPE_TOPK_MEAN_MAX, top_k=2)

    assert selected["action_type"] == "click_element"


def test_invalid_actions_are_never_selected():
    space = _space()
    mask = _mask_for(space, [("scroll_down", 0)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("click_element", 0)] = 100.0
    q[space.encode("scroll_down", 0)] = 1.0

    selected = select_counterfactual_action(q, mask, space, mode=FLAT_MAX)

    assert selected["action_type"] == "scroll_down"


def test_empty_mask_falls_back_to_noop():
    space = _space()
    mask = np.zeros(space.get_action_dim(), dtype=np.float32)
    q = np.zeros(space.get_action_dim(), dtype=np.float32)

    selected = select_counterfactual_action(q, mask, space, mode=TYPE_MEAN_MAX)

    assert selected["action_type"] == "noop"
    assert selected["action_id"] == space.encode("noop", 0)


def test_score_action_types_reports_uplift():
    space = _space()
    mask = _mask_for(space, [("click_element", 0), ("click_element", 1)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("click_element", 0)] = 3.0
    q[space.encode("click_element", 1)] = 1.0

    scores = score_action_types(q, mask, space, mode=TYPE_MEAN_MAX)

    assert scores["click_element"]["q_mean"] == 2.0
    assert scores["click_element"]["max_selection_uplift"] == 1.0


def test_offline_counterfactual_aggregates_click_ratios():
    records = [
        {"counterfactual_selections": {FLAT_MAX: {"action_type": "click_element"}, TYPE_MEAN_MAX: {"action_type": "scroll_down"}}},
        {"counterfactual_selections": {FLAT_MAX: {"action_type": "click_element"}, TYPE_MEAN_MAX: {"action_type": "fill_input"}}},
    ]

    summary = offline_counterfactual_from_records(records)

    assert summary["click_ratio_by_mode"][FLAT_MAX] == 1.0
    assert summary["click_ratio_by_mode"][TYPE_MEAN_MAX] == 0.0
    assert summary["disagreement_with_flat"]["TYPE_MEAN_MAX_vs_flat"] == 1.0


def test_action_dim_remains_608():
    assert _space().get_action_dim() == 608


def test_ground_truth_metadata_not_used_by_selector():
    space = _space()
    mask = _mask_for(space, [("click_element", 0), ("scroll_down", 0)])
    q = np.zeros(space.get_action_dim(), dtype=np.float32)
    q[space.encode("scroll_down", 0)] = 1.0
    q[space.encode("click_element", 0)] = 0.5
    metadata = {"data_bug_id": "BUG-1", "known_bug": "ignored"}

    selected = select_counterfactual_action(q, mask, space, mode=TYPE_MEAN_MAX)

    assert metadata["data_bug_id"] == "BUG-1"
    assert selected["action_type"] == "scroll_down"
