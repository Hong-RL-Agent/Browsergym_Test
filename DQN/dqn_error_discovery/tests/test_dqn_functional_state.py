import numpy as np

from dqn_error_discovery.runners.train_dqn_error_detector import (
    DQN_REWARD_CONFIG,
    calculate_dqn_reward,
    _build_functional_local_state_signature,
    _build_viewport_context_signature,
)


def _candidate(bid="a", *, bbox=None, **extra):
    item = {
        "bid": bid,
        "name": f"Target {bid}",
        "text": f"Target {bid}",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "enabled": True,
        "visibility": 1.0,
        "bbox": bbox or [10, 10, 100, 30],
    }
    item.update(extra)
    return item


def _obs(url="http://app.test/product/1", *, text_len=100, candidates=None, **page_extra):
    page = {
        "url": url,
        "title": "Product",
        "dom_node_count": 10,
        "page_text_length": text_len,
        "quantity": 1,
        "cart_count": 0,
        "viewport_width": 1280,
        "viewport_height": 720,
        "viewport_type": "desktop",
    }
    page.update(page_extra)
    return {
        "page_state": page,
        "candidate_elements": candidates if candidates is not None else [_candidate("a"), _candidate("b")],
        "runtime_signals": {},
    }


def _action(action_type="scroll_down", candidate_index=0):
    return {"action_type": action_type, "candidate_index": candidate_index}


def test_scroll_visible_candidate_change_not_functional_new_state():
    before = _obs(candidates=[_candidate("a"), _candidate("b")], text_len=100)
    after = _obs(candidates=[_candidate("b"), _candidate("c")], text_len=180)
    assert _build_functional_local_state_signature(before) == _build_functional_local_state_signature(after)

    history = {}
    calculate_dqn_reward(before, before, _action("noop"), [], history)
    _, components = calculate_dqn_reward(before, after, _action("scroll_down"), [], history)

    assert components["new_local_state"] == 0.0
    assert components["repeat_state_penalty"] == DQN_REWARD_CONFIG["repeat_state"]


def test_candidate_ordering_only_not_functional_change():
    before = _obs(candidates=[_candidate("a"), _candidate("b"), _candidate("c")])
    after = _obs(candidates=[_candidate("c"), _candidate("a"), _candidate("b")])
    assert _build_functional_local_state_signature(before) == _build_functional_local_state_signature(after)


def test_bbox_only_not_functional_change():
    before = _obs(candidates=[_candidate("a", bbox=[10, 10, 100, 30])])
    after = _obs(candidates=[_candidate("a", bbox=[10, 300, 100, 30])])
    assert _build_functional_local_state_signature(before) == _build_functional_local_state_signature(after)


def test_quantity_change_is_functional_change():
    assert _build_functional_local_state_signature(_obs(quantity=1)) != _build_functional_local_state_signature(_obs(quantity=2))


def test_modal_opens_is_functional_change():
    assert _build_functional_local_state_signature(_obs(modal_state="closed")) != _build_functional_local_state_signature(_obs(modal_state="open"))


def test_validation_appears_is_functional_change():
    assert _build_functional_local_state_signature(_obs(validation_state="none")) != _build_functional_local_state_signature(_obs(validation_state="invalid"))


def test_current_page_change_is_functional_change():
    assert _build_functional_local_state_signature(_obs(current_page=1)) != _build_functional_local_state_signature(_obs(current_page=2))


def test_status_change_is_functional_change():
    assert _build_functional_local_state_signature(_obs(status="pending")) != _build_functional_local_state_signature(_obs(status="completed"))


def test_url_path_change_is_functional_change():
    assert _build_functional_local_state_signature(_obs("/product/1")) != _build_functional_local_state_signature(_obs("/product/2"))


def test_same_functional_state_reward_once():
    before = _obs()
    after = _obs()
    history = {}
    _, first = calculate_dqn_reward(before, after, _action("noop"), [], history)
    _, second = calculate_dqn_reward(before, after, _action("noop"), [], history)
    assert first["new_local_state"] == DQN_REWARD_CONFIG["new_local_state"]
    assert second["new_local_state"] == 0.0


def test_scroll_exposes_new_target_no_local_state_but_new_action_target_possible():
    before = _obs(candidates=[_candidate("a")])
    after = _obs(candidates=[_candidate("a"), _candidate("new")], text_len=180)
    history = {}
    calculate_dqn_reward(before, before, _action("noop"), [], history)
    _, scroll_components = calculate_dqn_reward(before, after, _action("scroll_down"), [], history)
    _, click_components = calculate_dqn_reward(after, after, _action("click_element", 1), [], history)
    assert scroll_components["new_local_state"] == 0.0
    assert click_components["new_local_action_target"] == DQN_REWARD_CONFIG["new_local_action_target"]


def test_viewport_context_can_change_while_functional_signature_is_same():
    before = _obs(candidates=[_candidate("a")], text_len=100)
    after = _obs(candidates=[_candidate("b")], text_len=200)
    assert _build_viewport_context_signature(before) != _build_viewport_context_signature(after)
    assert _build_functional_local_state_signature(before) == _build_functional_local_state_signature(after)


def test_ground_truth_metadata_does_not_affect_functional_signature():
    plain = _obs(candidates=[_candidate("a")])
    with_gt = _obs(candidates=[_candidate("a", data_bug_id="BUG", catalog_bug_id_matches=["BUG"])])
    assert _build_functional_local_state_signature(plain) == _build_functional_local_state_signature(with_gt)
    assert np.isclose(calculate_dqn_reward(plain, plain, _action("noop"), [], {})[0], calculate_dqn_reward(with_gt, with_gt, _action("noop"), [], {})[0])


def test_dom_node_count_only_not_functional_change():
    assert _build_functional_local_state_signature(_obs(dom_node_count=10)) == _build_functional_local_state_signature(_obs(dom_node_count=20))
