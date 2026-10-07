from services.anomaly_detection_service import detect_anomalies


def _button() -> dict:
    return {
        "bid": "inspect",
        "name": "Inspect",
        "text": "Inspect",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 100, 100, 40],
    }


def _item(cart_item_id: str, price, quantity) -> dict:
    return {"cart_item_id": cart_item_id, "unit_price": price, "quantity": quantity}


def _observation(
    *,
    cart_items: list[dict] | None = None,
    subtotal=None,
    total=None,
    discount_amount=None,
    tax=None,
    shipping=None,
    loading: bool = False,
    api_entities: list[dict] | None = None,
) -> dict:
    page_state = {
        "url": "http://app.test/cart",
        "title": "Cart",
        "page_text": "Loading" if loading else "Cart",
        "page_text_sample": "Loading" if loading else "Cart",
        "page_text_length": 4,
        "dom_node_count": len(cart_items or []) + 1,
        "viewport_width": 800,
        "viewport_height": 600,
        "cart_items": cart_items or [],
        "cart_count": len(cart_items or []),
        "loading": loading,
    }
    if subtotal is not None:
        page_state["subtotal"] = subtotal
    if total is not None:
        page_state["total"] = total
    if discount_amount is not None:
        page_state["discount_amount"] = discount_amount
    if tax is not None:
        page_state["tax"] = tax
    if shipping is not None:
        page_state["shipping"] = shipping
    if api_entities is not None:
        page_state["api_entities"] = api_entities
    return {
        "page_state": page_state,
        "runtime_signals": {"loading": loading},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _r11(before: dict, after: dict, **info) -> list[dict]:
    anomalies = detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": "inspect_cart", "candidate_index": 0},
            **info,
        },
    )
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R11"]


def test_r11_normal_subtotal_does_not_emit():
    after = _observation(cart_items=[_item("a", 1000, 2)], subtotal=2000)

    assert not _r11(_observation(), after)


def test_r11_wrong_subtotal_emits_cart_total_mismatch():
    after = _observation(cart_items=[_item("a", 1000, 2)], subtotal=3000)

    anomaly = _r11(_observation(), after)[0]
    assert anomaly["type"] == "cart-total-mismatch"
    assert anomaly["evidence"]["expected_value"] == 2000


def test_r11_multiple_items_normal_sum_does_not_emit():
    after = _observation(cart_items=[_item("a", 1000, 2), _item("b", 3000, 1)], subtotal=5000)

    assert not _r11(_observation(), after)


def test_r11_multiple_items_wrong_sum_emits():
    after = _observation(cart_items=[_item("a", 1000, 2), _item("b", 3000, 1)], subtotal=6000)

    assert _r11(_observation(), after)


def test_r11_price_formatting_does_not_emit():
    after = _observation(cart_items=[_item("a", "1,000원", "2")], subtotal="2,000원")

    assert not _r11(_observation(), after)


def test_r11_discount_breakdown_normal_total_does_not_emit():
    after = _observation(subtotal=10000, discount_amount=1000, shipping=3000, tax=0, total=12000)

    assert not _r11(_observation(), after)


def test_r11_discount_breakdown_wrong_total_emits():
    after = _observation(subtotal=10000, discount_amount=1000, shipping=3000, tax=0, total=15000)

    anomaly = _r11(_observation(), after)[0]
    assert anomaly["evidence"]["calculation_type"] == "total_breakdown"


def test_r11_missing_operands_does_not_emit():
    after = _observation(total=10000)

    assert not _r11(_observation(), after)


def test_r11_api_total_vs_ui_total_is_not_calculation_error():
    before = _observation()
    after = _observation(total=6000, api_entities=[{"product_id": "1", "total": 5000}])

    assert not _r11(before, after, response_status=200, api_entities=[{"product_id": "1", "total": 5000}])


def test_r11_invalid_quantity_but_math_correct_does_not_emit():
    after = _observation(cart_items=[_item("a", 1000, 0)], subtotal=0)

    assert not _r11(_observation(), after)


def test_r11_duplicate_dom_item_is_not_double_counted():
    after = _observation(cart_items=[_item("a", 1000, 2), _item("a", 1000, 2)], subtotal=2000)

    assert not _r11(_observation(), after)


def test_r11_api_500_does_not_emit():
    after = _observation(cart_items=[_item("a", 1000, 2)], subtotal=3000)

    assert not _r11(_observation(), after, response_status=500)


def test_r11_network_failure_does_not_emit():
    after = _observation(cart_items=[_item("a", 1000, 2)], subtotal=3000)

    assert not _r11(_observation(), after, network_error="connection reset")


def test_r11_ui_loading_does_not_emit():
    after = _observation(cart_items=[_item("a", 1000, 2)], subtotal=3000, loading=True)

    assert not _r11(_observation(), after)


def test_r11_float_tolerance_does_not_emit():
    after = _observation(cart_items=[_item("a", 5.00000005, 2)], subtotal=10.0)

    assert not _r11(_observation(), after)
