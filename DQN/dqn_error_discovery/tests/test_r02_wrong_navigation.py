from copy import deepcopy

from services.anomaly_detection_service import detect_anomalies


def _observation(url: str, *, candidate: dict | None = None, entity_id: str = "") -> dict:
    page_state = {
        "url": url,
        "title": "Page",
        "page_text_length": 100,
        "page_text_sample": "Page",
        "dom_node_count": 10,
        "cart_count": 0,
        "has_modal_or_dialog": False,
        "has_form": False,
    }
    if entity_id:
        page_state["entity_id"] = entity_id
    return {
        "page_state": page_state,
        "runtime_signals": {},
        "candidate_elements": [candidate or _product_candidate("17")],
    }


def _product_candidate(product_id: str) -> dict:
    return {
        "bid": f"product-{product_id}",
        "name": f"Product {product_id}",
        "text": f"Product {product_id}",
        "role": "link",
        "tag": "a",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "href": f"/products/{product_id}?ref=home",
        "product_id": product_id,
    }


def _click_action(**extra: object) -> dict:
    action = {"action_type": "click_element", "candidate_index": 0}
    result = {"action": action}
    result.update(extra)
    return result


def _types(before: dict, after: dict, action_info: dict | None = None) -> list[str]:
    return [item["type"] for item in detect_anomalies(before, after, action_info or _click_action())]


def test_r02_normal_entity_navigation_does_not_emit_broken_navigation():
    before = _observation("http://shop.test/products")
    after = _observation("http://shop.test/products/17", entity_id="17")

    assert "broken-navigation" not in _types(before, after)


def test_r02_entity_mismatch_emits_broken_navigation():
    before = _observation("http://shop.test/products")
    after = _observation("http://shop.test/products/21", entity_id="21")

    assert "broken-navigation" in _types(before, after)


def test_r02_normal_login_redirect_chain_does_not_emit_mismatch():
    before = _observation("http://shop.test/login")
    after = _observation("http://shop.test/dashboard")
    after["runtime_signals"]["login_success"] = True
    after["runtime_signals"]["post_login_url"] = "http://shop.test/dashboard"

    types = _types(
        before,
        after,
        _click_action(
            login_success=True,
            expected_url="/dashboard",
            redirect_chain=["/login", "/auth/callback", "/dashboard"],
        ),
    )

    assert "login-redirect-mismatch" not in types


def test_r02_login_success_wrong_route_emits_mismatch():
    before = _observation("http://shop.test/login")
    after = _observation("http://shop.test/products/999")
    after["runtime_signals"]["login_success"] = True
    after["runtime_signals"]["post_login_url"] = "http://shop.test/products/999"

    types = _types(before, after, _click_action(login_success=True, expected_url="/dashboard"))

    assert "login-redirect-mismatch" in types


def test_r02_action_error_does_not_emit_wrong_navigation():
    before = _observation("http://shop.test/products")
    after = _observation("http://shop.test/products/21", entity_id="21")

    types = _types(before, after, _click_action(last_action_error=True))

    assert "broken-navigation" not in types
    assert "login-redirect-mismatch" not in types


def test_r02_no_response_stays_out_of_wrong_navigation():
    before = _observation("http://shop.test/products")
    after = deepcopy(before)

    assert "broken-navigation" not in _types(before, after)


def test_r02_api_failure_does_not_emit_wrong_navigation():
    before = _observation("http://shop.test/login")
    after = deepcopy(before)

    types = _types(before, after, _click_action(login_success=True, expected_url="/dashboard", response_status=500))

    assert "broken-navigation" not in types
    assert "login-redirect-mismatch" not in types
