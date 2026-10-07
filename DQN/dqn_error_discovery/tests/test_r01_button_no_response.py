from copy import deepcopy

from services.anomaly_detection_service import detect_anomalies


def _observation() -> dict:
    candidate = {
        "bid": "buy-1",
        "name": "Add to cart",
        "text": "Add to cart",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "is_purchase_action": True,
    }
    return {
        "page_state": {
            "url": "http://example.test/product",
            "title": "Product",
            "page_text_length": 100,
            "page_text_sample": "Product Add to cart",
            "dom_node_count": 10,
            "cart_count_detected": True,
            "cart_count": 0,
            "cart_text": "Cart (0)",
            "has_modal_or_dialog": False,
            "has_form": False,
        },
        "runtime_signals": {},
        "candidate_elements": [candidate],
    }


def _click_action(**extra: object) -> dict:
    action = {"action_type": "click_element", "candidate_index": 0}
    result = {"action": action}
    result.update(extra)
    return result


def _button_no_response_types(before: dict, after: dict, action_info: dict | None = None) -> list[str]:
    return [item["type"] for item in detect_anomalies(before, after, action_info or _click_action())]


def test_r01_all_no_change_allows_button_no_response():
    before = _observation()
    after = deepcopy(before)

    assert "button-no-response" in _button_no_response_types(before, after)


def test_r01_network_change_blocks_button_no_response():
    before = _observation()
    after = deepcopy(before)

    types = _button_no_response_types(before, after, _click_action(network_changed=True))

    assert "button-no-response" not in types


def test_r01_feedback_blocks_button_no_response():
    before = _observation()
    after = deepcopy(before)
    after["page_state"]["page_text_sample"] = "Product Add to cart success"

    assert "button-no-response" not in _button_no_response_types(before, after)


def test_r01_modal_or_form_blocks_button_no_response():
    before = _observation()
    modal_after = deepcopy(before)
    modal_after["page_state"]["has_modal_or_dialog"] = True
    form_after = deepcopy(before)
    form_after["page_state"]["has_form"] = True

    assert "button-no-response" not in _button_no_response_types(before, modal_after)
    assert "button-no-response" not in _button_no_response_types(before, form_after)


def test_r01_cart_state_change_blocks_button_no_response():
    before = _observation()
    after = deepcopy(before)
    after["page_state"]["cart_count"] = 1
    after["page_state"]["cart_text"] = "Cart (1)"

    assert "button-no-response" not in _button_no_response_types(before, after)


def test_r01_last_action_error_blocks_button_no_response():
    before = _observation()
    after = deepcopy(before)

    types = _button_no_response_types(before, after, _click_action(last_action_error=True))

    assert "button-no-response" not in types
