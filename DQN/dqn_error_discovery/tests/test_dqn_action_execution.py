import random

import numpy as np
import torch

from adapters.browsergym_action_adapter import BrowserGymActionAdapter
from dqn_error_discovery.agents.dqn_agent import masked_argmax, random_valid_action
from envs.browsergym_jaws_env import normalize_action_error
from models.action_space import ActionSpace


def _candidate(bid="1", role="button", tag="button", **extra):
    item = {
        "bid": bid,
        "text": "Target",
        "name": "Target",
        "role": role,
        "tag": tag,
        "visible": True,
        "enabled": True,
        "clickable": True,
        "visibility": 1.0,
        "bbox": [10, 10, 80, 24],
        "source": "axtree",
    }
    item.update(extra)
    return item


def _obs(candidates):
    return {
        "page_state": {"url": "http://localhost:9220/", "site_id": "site001"},
        "runtime_signals": {},
        "infra_signals": {},
        "history": {"step_index": 10},
        "candidate_elements": candidates,
    }


class _FixedNet(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, observations):
        return self.values.unsqueeze(0).repeat(observations.shape[0], 1)


def test_encode_decode_roundtrip_all_actions():
    action_space = ActionSpace(max_candidates=32)
    assert action_space.get_action_dim() == 608
    for action_id in range(action_space.get_action_dim()):
        decoded = action_space.decode(action_id)
        assert action_space.encode(decoded["action_type"], decoded["candidate_index"]) == action_id
    assert action_space.decode(0) == {"action_type_id": 0, "action_type": "noop", "candidate_index": 0}
    assert action_space.decode(31)["candidate_index"] == 31
    assert action_space.decode(32) == {"action_type_id": 1, "action_type": "click_element", "candidate_index": 0}
    assert action_space.decode(63)["action_type"] == "click_element"
    assert action_space.decode(607) == {"action_type_id": 18, "action_type": "finish_episode", "candidate_index": 31}


def test_click_candidate_mask_alignment():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs([_candidate("a"), _candidate("b")]))
    action_id = action_space.encode("click_element", 1)
    assert mask[action_id] == 1.0
    assert action_space.decode(action_id)["candidate_index"] == 1


def test_non_clickable_candidate_click_mask_false():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs([_candidate(clickable=False, role="generic", tag="div")]))
    assert mask[action_space.encode("click_element", 0)] == 0.0


def test_textbox_candidate_fill_input_valid():
    action_space = ActionSpace()
    candidate = _candidate("input-1", role="textbox", tag="input", clickable=False, fillable=True, input_type="text")
    mask = action_space.build_action_mask(_obs([candidate]))
    assert mask[action_space.encode("fill_input", 0)] == 1.0


def test_button_candidate_fill_input_invalid():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs([_candidate("button-1")]))
    assert mask[action_space.encode("fill_input", 0)] == 0.0


def test_scroll_down_candidate_independent_canonical_slot_only():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs([_candidate("a")]))
    assert mask[action_space.encode("scroll_down", 0)] == 1.0
    assert sum(mask[action_space.encode("scroll_down", i)] for i in range(1, action_space.max_candidates)) == 0.0


def test_inspect_dom_candidate_independent_canonical_slot_only():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs([_candidate("a")]))
    assert mask[action_space.encode("inspect_dom", 0)] == 1.0
    assert sum(mask[action_space.encode("inspect_dom", i)] for i in range(1, action_space.max_candidates)) == 0.0


def test_masked_greedy_selection_ignores_invalid_high_q():
    action_space = ActionSpace()
    values = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    invalid_id = action_space.encode("click_element", 5)
    valid_id = action_space.encode("inspect_dom", 0)
    values[invalid_id] = 100.0
    values[valid_id] = 1.0
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    mask[valid_id] = 1.0
    assert masked_argmax(_FixedNet(values), np.zeros(635, dtype=np.float32), mask) == valid_id


def test_epsilon_random_uses_valid_action_set_only():
    action_space = ActionSpace()
    valid_ids = {action_space.encode("inspect_dom", 0), action_space.encode("scroll_down", 0)}
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    for action_id in valid_ids:
        mask[action_id] = 1.0
    rng = random.Random(7)
    assert {random_valid_action(mask, rng) for _ in range(20)} <= valid_ids


def test_empty_action_mask_falls_back_to_noop_not_click():
    action_space = ActionSpace()
    mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
    selected = masked_argmax(_FixedNet(np.ones(action_space.get_action_dim())), np.zeros(635, dtype=np.float32), mask)
    assert selected == action_space.encode("noop", 0)


def test_negative_bid_resolvable_candidate_is_valid():
    action_space = ActionSpace()
    candidate = _candidate("-1000000007", source="axtree", clickable=True, set_of_marks=True)
    mask = action_space.build_action_mask(_obs([candidate]))
    assert mask[action_space.encode("click_element", 0)] == 1.0


def test_negative_bid_non_resolvable_placeholder_is_not_clickable():
    action_space = ActionSpace()
    candidate = _candidate("-1000000007", source="bbox_fallback", clickable=False, role="generic", tag="div")
    mask = action_space.build_action_mask(_obs([candidate]))
    assert mask[action_space.encode("click_element", 0)] == 0.0


def test_candidate_index_mapping_reaches_same_identity_in_adapter():
    action_space = ActionSpace()
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    action_id = action_space.encode("click_element", 2)
    decoded = action_space.decode(action_id)
    browser_action = BrowserGymActionAdapter().adapt(decoded, _obs(candidates))
    assert decoded["candidate_index"] == 2
    assert browser_action.action == "click('c')"


def test_candidate_order_consistency_mask_decode_adapter():
    action_space = ActionSpace()
    candidates = [_candidate("first"), _candidate("second")]
    mask = action_space.build_action_mask(_obs(candidates))
    action_id = action_space.encode("click_element", 1)
    assert mask[action_id] == 1.0
    assert BrowserGymActionAdapter().adapt(action_space.decode(action_id), _obs(candidates)).action == "click('second')"


def test_stale_nonexistent_candidate_error_category():
    assert normalize_action_error("waiting for locator('[bid=\"missing\"]') failed: timeout") in {
        "target-not-found",
        "playwright-timeout",
    }


def test_action_error_normalization_target_not_clickable():
    assert normalize_action_error("element intercepts pointer events and is not clickable") == "target-not-clickable"


def test_ground_truth_metadata_does_not_affect_action_mask():
    action_space = ActionSpace()
    plain = _obs([_candidate("a")])
    with_gt = _obs([_candidate("a", data_bug_id="BUG-1", catalog_bug_id_matches=["BUG-1"])])
    assert np.array_equal(action_space.build_action_mask(plain), action_space.build_action_mask(with_gt))
