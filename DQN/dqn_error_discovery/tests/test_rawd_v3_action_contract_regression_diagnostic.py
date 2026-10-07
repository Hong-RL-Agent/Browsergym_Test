from dqn_error_discovery.v3.action_contract_regression_diagnostic import _join


def test_visible_combobox_joins_to_native_select_by_geometry():
    dom = {"nodes": [{"tagName": "SELECT", "id": "fixture-select", "bbox": [1, 2, 10, 20], "accessibleNameHint": "fixture-select"}]}
    candidates = [{"slot": 0, "bid": "9", "tag": "", "role": "combobox", "bbox": [1, 2, 10, 20], "name": "Choose"}]
    joins = _join(dom, candidates)
    assert joins[0]["dom_matches"][0]["tagName"] == "SELECT"
    assert joins[0]["dom_matches"][0]["close_geometry"] is True


def test_custom_combobox_does_not_join_to_native_select():
    dom = {"nodes": [{"tagName": "DIV", "id": "custom", "bbox": [1, 2, 10, 20], "accessibleNameHint": "Custom"}]}
    candidates = [{"slot": 0, "bid": "14", "tag": "", "role": "combobox", "bbox": [1, 2, 10, 20], "name": "Custom"}]
    joins = _join(dom, candidates)
    assert joins[0]["dom_matches"][0]["tagName"] == "DIV"


def test_diagnostic_join_does_not_change_contract_mask():
    from dqn_error_discovery.v3.action_space import V3ActionSpace

    space = V3ActionSpace(32)
    candidate = {"tag": "", "role": "combobox", "visible": True, "enabled": True, "bbox": [1, 2, 10, 20], "bid": "9"}
    observation = {"candidate_elements": [candidate]}
    mask = space.build_action_mask(observation)
    assert not mask[space.encode("select_option", 0)]
