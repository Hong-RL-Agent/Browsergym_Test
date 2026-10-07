import numpy as np

from adapters.browsergym_action_adapter import BrowserGymActionAdapter
from dqn_error_discovery.runners.run_policy_sanity_check import SanityMetricsAggregator
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
        "set_of_marks": False,
        "visibility": 1.0,
        "bbox": [10, 10, 80, 24],
        "source": "axtree",
    }
    item.update(extra)
    return item


def _obs(candidates=None, **runtime):
    return {
        "page_state": {"url": "http://localhost:9220/", "title": "Page", "dom_node_count": 1, "page_text_length": 10},
        "runtime_signals": dict(runtime),
        "infra_signals": {},
        "history": {"step_index": 10},
        "candidate_elements": candidates if candidates is not None else [_candidate()],
    }


def _record(action_type="scroll_down", *, info=None, action=None, before=None, after=None):
    agg = SanityMetricsAggregator("ppo")
    agg.record_step(
        step=1,
        before_observation=before or _obs(),
        after_observation=after or _obs(),
        action=action or {"action_type": action_type, "candidate_index": 0},
        reward=0.0,
        reward_components={},
        anomalies=[],
        info=info or {},
    )
    return agg.summary()


def test_ppo_click_mask_valid_candidate_execution_resolvable():
    action_space = ActionSpace()
    obs = _obs([_candidate("clickable")])
    mask = action_space.build_action_mask(obs)
    action_id = action_space.encode("click_element", 0)
    assert mask[action_id] == 1.0
    assert BrowserGymActionAdapter().adapt(action_space.decode(action_id), obs).action == "click('clickable')"


def test_ppo_non_clickable_candidate_click_mask_false():
    action_space = ActionSpace()
    obs = _obs([_candidate(clickable=False, role="generic", tag="div")])
    assert action_space.build_action_mask(obs)[action_space.encode("click_element", 0)] == 0.0


def test_ppo_non_resolvable_placeholder_click_false():
    action_space = ActionSpace()
    obs = _obs([_candidate("", source="bbox_fallback", clickable=True)])
    assert action_space.build_action_mask(obs)[action_space.encode("click_element", 0)] == 0.0


def test_ppo_valid_textbox_fill_input_true():
    action_space = ActionSpace()
    obs = _obs([_candidate("input-1", role="textbox", tag="input", clickable=False, fillable=True, input_type="text")])
    assert action_space.build_action_mask(obs)[action_space.encode("fill_input", 0)] == 1.0


def test_ppo_statictext_fill_input_false():
    action_space = ActionSpace()
    obs = _obs([_candidate("txt", role="statictext", tag="span", clickable=False)])
    assert action_space.build_action_mask(obs)[action_space.encode("fill_input", 0)] == 0.0


def test_ppo_valid_input_press_enter_true():
    action_space = ActionSpace()
    obs = _obs([_candidate("input-1", role="searchbox", tag="input", clickable=False, fillable=True)])
    assert action_space.build_action_mask(obs)[action_space.encode("press_enter", 0)] == 1.0


def test_ppo_statictext_press_enter_false():
    action_space = ActionSpace()
    obs = _obs([_candidate("txt", role="statictext", tag="span", clickable=False)])
    assert action_space.build_action_mask(obs)[action_space.encode("press_enter", 0)] == 0.0


def test_ppo_candidate_independent_action_canonical_slot_only():
    action_space = ActionSpace()
    mask = action_space.build_action_mask(_obs([_candidate("a")]))
    for action_type in ("noop", "scroll_down", "scroll_up", "inspect_dom", "change_viewport_mobile", "finish_episode"):
        assert mask[action_space.encode(action_type, 0)] == 1.0
        assert sum(mask[action_space.encode(action_type, idx)] for idx in range(1, action_space.max_candidates)) == 0.0


def test_scroll_boundary_no_effect_is_not_action_error():
    summary = _record("scroll_down")
    assert summary["metrics"]["no_effect_count"] == 1
    assert summary["metrics"]["action_error_count"] == 0


def test_same_viewport_action_is_not_action_error():
    summary = _record("change_viewport_mobile", info={"viewport_changed": True})
    assert summary["metrics"]["action_error_count"] == 0


def test_empty_inspect_result_is_not_action_error():
    summary = _record("inspect_console", info={"console_inspected": True, "console_error_count": 0})
    assert summary["metrics"]["action_error_count"] == 0


def test_candidate_mapping_mask_decode_adapter_same_identity():
    action_space = ActionSpace()
    candidates = [_candidate("first"), _candidate("second")]
    obs = _obs(candidates)
    action_id = action_space.encode("click_element", 1)
    assert action_space.build_action_mask(obs)[action_id] == 1.0
    assert action_space.decode(action_id)["candidate_index"] == 1
    assert BrowserGymActionAdapter().adapt(action_space.decode(action_id), obs).action == "click('second')"


def test_stale_candidate_normalized_category():
    assert normalize_action_error("Element is not attached to the DOM, locator resolved to detached node") == "target-detached"


def test_internal_missing_bid_normalized_target_not_resolvable():
    assert normalize_action_error("active page or bid not found") == "target-not-resolvable"


def test_internal_fill_error_is_counted_as_execution_error():
    summary = _record(
        "fill_input",
        before=_obs([_candidate("input-1", role="textbox", tag="input", fillable=True)]),
        action={"action_type": "fill_input", "candidate_index": 0, "failed": True},
        info={"last_action_error": True, "execution_action_error": True, "fill_error": "element is not editable"},
    )
    assert summary["metrics"]["action_error_count"] == 1
    assert summary["action_error_categories"] == {"input-target-not-editable": 1}


def test_negative_resolvable_bid_allowed():
    action_space = ActionSpace()
    obs = _obs([_candidate("-1000000007", clickable=True, set_of_marks=True, source="axtree")])
    action_id = action_space.encode("click_element", 0)
    assert action_space.build_action_mask(obs)[action_id] == 1.0
    assert BrowserGymActionAdapter().adapt(action_space.decode(action_id), obs).action == "click('-1000000007')"


def test_negative_non_resolvable_placeholder_excluded():
    action_space = ActionSpace()
    obs = _obs([_candidate("-1000000007", clickable=False, role="generic", tag="div", source="bbox_fallback")])
    assert action_space.build_action_mask(obs)[action_space.encode("click_element", 0)] == 0.0


def test_ground_truth_metadata_does_not_change_ppo_mask():
    action_space = ActionSpace()
    plain = _obs([_candidate("a")])
    with_gt = _obs([_candidate("a", data_bug_id="BUG-1", catalog_bug_id_matches=["BUG-1"])])
    assert np.array_equal(action_space.build_action_mask(plain), action_space.build_action_mask(with_gt))


def test_raw_browsergym_last_action_error_flag_alone_is_not_execution_error():
    summary = _record(
        "click_element",
        before=_obs([_candidate("a")]),
        after=_obs([_candidate("a")], browsergym_raw_last_action_error=True, last_action_error=False),
        action={"action_type": "click_element", "candidate_index": 0, "failed": False},
        info={"browsergym_raw_last_action_error": True, "last_action_error": False, "execution_action_error": False},
    )
    assert summary["metrics"]["action_error_count"] == 0
