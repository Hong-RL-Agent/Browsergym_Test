from copy import deepcopy

from services.anomaly_detection_service import detect_anomalies


def _observation(
    *,
    text: str = "Products list item one item two price 10",
    url: str = "http://shop.test/products",
    candidate_count: int = 6,
    runtime: dict | None = None,
) -> dict:
    candidates = [
        {
            "bid": f"c-{index}",
            "name": f"Item {index}",
            "text": f"Item {index}",
            "role": "button",
            "tag": "button",
            "clickable": True,
            "visible": True,
            "visibility": 1.0,
        }
        for index in range(candidate_count)
    ]
    return {
        "page_state": {
            "url": url,
            "title": "Products",
            "page_text": text,
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": max(candidate_count, 1),
            "cart_count": 0,
            "has_modal_or_dialog": False,
            "has_form": False,
        },
        "runtime_signals": dict(runtime or {}),
        "candidate_elements": candidates,
    }


def _action(**extra: object) -> dict:
    result = {"action": {"action_type": "click_element", "candidate_index": 0}}
    result.update(extra)
    return result


def _types(before: dict, after: dict, action_info: dict | None = None) -> list[str]:
    return [item["type"] for item in detect_anomalies(before, after, action_info or _action())]


def test_r03_uncaught_type_error_emits_runtime_exception():
    before = _observation()
    after = deepcopy(before)

    assert "runtime-exception" in _types(before, after, _action(console_errors=["Uncaught TypeError: x is undefined"]))


def test_r03_pageerror_reference_error_emits_runtime_exception():
    before = _observation()
    after = _observation(runtime={"page_errors": ["ReferenceError: missingVar is not defined"]})

    assert "runtime-exception" in _types(before, after)


def test_r03_console_warning_only_does_not_emit_runtime_exception():
    before = _observation()
    after = deepcopy(before)

    types = _types(before, after, _action(console_messages=[{"type": "warning", "text": "React dev warning"}]))

    assert "runtime-exception" not in types


def test_r03_action_error_only_does_not_emit_runtime_exception():
    before = _observation()
    after = deepcopy(before)

    types = _types(before, after, _action(last_action_error=True, action_error="locator failed"))

    assert "runtime-exception" not in types


def test_r03_api_500_only_does_not_emit_runtime_exception():
    before = _observation()
    after = deepcopy(before)

    types = _types(before, after, _action(response_status=500))

    assert "runtime-exception" not in types
    assert "empty-state-rendering" not in types


def test_r03_api_500_plus_frontend_exception_emits_runtime_exception():
    before = _observation()
    after = deepcopy(before)

    types = _types(before, after, _action(response_status=500, page_errors=["Uncaught TypeError: Cannot read properties of undefined"]))

    assert "runtime-exception" in types


def test_r03_normal_empty_state_does_not_emit_empty_state_rendering():
    before = _observation(text="Search products results", candidate_count=4)
    after = _observation(text="검색 결과가 없습니다", candidate_count=1)

    assert "empty-state-rendering" not in _types(before, after)


def test_r03_content_collapse_emits_empty_state_rendering():
    before = _observation(text=" ".join(["Product card title price description"] * 30), candidate_count=12)
    after = _observation(text="Main", candidate_count=1)

    assert "empty-state-rendering" in _types(before, after)


def test_r03_visible_undefined_value_emits_rendering_anomaly():
    before = _observation()
    after = _observation(text="상품 상세 가격: undefined", candidate_count=4)

    assert "empty-state-rendering" in _types(before, after)


def test_r03_visible_nan_value_emits_rendering_anomaly():
    before = _observation()
    after = _observation(text="Cart 총액: NaN", candidate_count=4)

    assert "empty-state-rendering" in _types(before, after)
