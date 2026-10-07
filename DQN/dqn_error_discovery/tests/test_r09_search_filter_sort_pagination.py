from services.anomaly_detection_service import detect_anomalies


def _result(entity_id: str, **attrs) -> dict:
    return {"entity_id": entity_id, **attrs}


def _button(text: str = "Apply") -> dict:
    return {
        "bid": "action",
        "name": text,
        "text": text,
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 100, 100, 40],
    }


def _observation(
    results: list[dict],
    *,
    page: int = 1,
    total_pages: int = 3,
    selected: dict | None = None,
    url: str = "http://app.test/results",
) -> dict:
    selected = selected or {}
    text = " ".join(str(value) for result in results for value in result.values())
    return {
        "page_state": {
            "url": url,
            "title": "Results",
            "page_text": text,
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": len(results) + 1,
            "viewport_width": 800,
            "viewport_height": 600,
            "current_page": page,
            "total_pages": total_pages,
            "result_entities": results,
            "cart_count": 0,
            **selected,
        },
        "runtime_signals": {},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _r09(before: dict, after: dict, **info) -> list[dict]:
    anomalies = detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": "click_element", "candidate_index": 0},
            **info,
        },
    )
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R09"]


def test_r09_filter_all_results_match_does_not_emit():
    before = _observation([_result("1", status="active"), _result("2", status="inactive")])
    after = _observation(
        [_result("1", status="active"), _result("3", status="active")],
        selected={"filter_key": "status", "filter_value": "active"},
    )

    assert not _r09(before, after, filter_key="status", filter_value="active", success=True)


def test_r09_filter_violating_entity_emits():
    before = _observation([_result("1", status="active"), _result("2", status="inactive")])
    after = _observation(
        [_result("1", status="active"), _result("2", status="inactive")],
        selected={"filter_key": "status", "filter_value": "active"},
    )

    anomaly = _r09(before, after, filter_key="status", filter_value="active", success=True)[0]
    assert anomaly["type"] == "filter-result-mismatch"
    assert anomaly["evidence"]["violating_entity_ids"] == ["2"]


def test_r09_filter_same_results_but_all_match_does_not_emit():
    before = _observation([_result("1", status="active"), _result("2", status="active")])
    after = _observation(
        [_result("1", status="active"), _result("2", status="active")],
        selected={"filter_key": "status", "filter_value": "active"},
    )

    assert not _r09(before, after, filter_key="status", filter_value="active", success=True)


def test_r09_filter_complete_no_response_does_not_emit():
    before = _observation([_result("1", status="active"), _result("2", status="inactive")])
    after = _observation([_result("1", status="active"), _result("2", status="inactive")])

    assert not _r09(before, after, filter_key="status", filter_value="active")


def test_r09_filter_api_500_does_not_emit():
    before = _observation([_result("1", status="active")])
    after = _observation(
        [_result("1", status="inactive")],
        selected={"filter_key": "status", "filter_value": "active"},
    )

    assert not _r09(before, after, filter_key="status", filter_value="active", response_status=500)


def test_r09_filter_action_error_does_not_emit():
    before = _observation([_result("1", status="active")])
    after = _observation(
        [_result("1", status="inactive")],
        selected={"filter_key": "status", "filter_value": "active"},
    )

    assert not _r09(before, after, filter_key="status", filter_value="active", success=True, last_action_error=True)


def test_r09_sort_ascending_ok_does_not_emit():
    before = _observation([_result("1", price=3), _result("2", price=1)])
    after = _observation([_result("2", price=1), _result("1", price=3)], selected={"sort_key": "price", "sort_direction": "asc"})

    assert not _r09(before, after, sort_key="price", sort_direction="asc", success=True)


def test_r09_sort_ascending_wrong_emits():
    before = _observation([_result("1", price=1), _result("2", price=2)])
    after = _observation([_result("1", price=1), _result("3", price=3), _result("2", price=2)], selected={"sort_key": "price", "sort_direction": "asc"})

    anomaly = _r09(before, after, sort_key="price", sort_direction="asc", success=True)[0]
    assert anomaly["type"] == "sort-order-mismatch"


def test_r09_sort_descending_ok_does_not_emit():
    before = _observation([_result("1", price=1), _result("2", price=3)])
    after = _observation([_result("2", price=3), _result("1", price=1)], selected={"sort_key": "price", "sort_direction": "desc"})

    assert not _r09(before, after, sort_key="price", sort_direction="desc", success=True)


def test_r09_pagination_page_one_to_two_ok_does_not_emit():
    before = _observation([_result("1"), _result("2")], page=1, total_pages=3)
    after = _observation([_result("3"), _result("4")], page=2, total_pages=3)

    assert not _r09(before, after, pagination_action="next", success=True)


def test_r09_pagination_next_possible_but_page_and_results_stay_emits():
    before = _observation([_result("1"), _result("2")], page=1, total_pages=3)
    after = _observation([_result("1"), _result("2")], page=1, total_pages=3)

    anomaly = _r09(before, after, pagination_action="next", success=True)
    assert anomaly[0]["type"] == "pagination-mismatch"


def test_r09_pagination_last_page_stays_does_not_emit():
    before = _observation([_result("5"), _result("6")], page=3, total_pages=3)
    after = _observation([_result("5"), _result("6")], page=3, total_pages=3)

    assert not _r09(before, after, pagination_action="next", success=True)


def test_r09_search_structured_query_all_match_does_not_emit():
    before = _observation([_result("1", title="Apple"), _result("2", title="Banana")])
    after = _observation([_result("1", title="Apple pie")])

    assert not _r09(before, after, search_key="title", search_value="apple", success=True)


def test_r09_search_structured_query_violation_emits():
    before = _observation([_result("1", title="Apple"), _result("2", title="Banana")])
    after = _observation([_result("1", title="Apple pie"), _result("2", title="Banana")])

    anomaly = _r09(before, after, search_key="title", search_value="apple", success=True)[0]
    assert anomaly["type"] == "search-result-mismatch"
    assert anomaly["evidence"]["violating_entity_ids"] == ["2"]


def test_r09_search_query_semantics_unknown_does_not_emit():
    before = _observation([_result("1", title="Apple")])
    after = _observation([_result("2", title="Banana")])

    assert not _r09(before, after, query="apple", success=True)
