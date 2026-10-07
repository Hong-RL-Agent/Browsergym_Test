from services.anomaly_detection_service import detect_anomalies


def _button() -> dict:
    return {
        "bid": "submit",
        "name": "Submit",
        "text": "Submit",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 100, 100, 40],
    }


def _observation(text: str = "Submit", *, runtime_entries: list[dict] | None = None, page_state_extra: dict | None = None) -> dict:
    page_state = {
        "url": "http://app.test/page",
        "title": "Page",
        "page_text": text,
        "page_text_sample": text,
        "page_text_length": len(text),
        "dom_node_count": 1,
        "viewport_width": 800,
        "viewport_height": 600,
        "cart_count": 0,
    }
    page_state.update(page_state_extra or {})
    return {
        "page_state": page_state,
        "runtime_signals": {"network_entries": runtime_entries or []},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _entry(
    method: str = "GET",
    url: str = "http://app.test/api/users",
    *,
    status: int | None = None,
    failure: str = "",
    resource_type: str = "fetch",
    response_received: bool = False,
    request_id: str = "req-1",
    **extra,
) -> dict:
    item = {
        "method": method,
        "url": url,
        "resource_type": resource_type,
        "response_received": response_received,
        "request_id": request_id,
        "network_error": failure,
        **extra,
    }
    if status is not None:
        item["status"] = status
    return item


def _anomalies(entry: dict | None = None, *, before=None, after=None, last_action_error: bool = False, **info) -> list[dict]:
    before = before or _observation()
    after = after or _observation()
    entries = [entry] if entry else []
    return detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": "click_element", "candidate_index": 0},
            "network_entries": entries,
            "last_action_error": last_action_error,
            **info,
        },
    )


def _rule(anomalies: list[dict], rule_id: str) -> list[dict]:
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == rule_id]


def _r15(anomalies: list[dict]) -> list[dict]:
    return _rule(anomalies, "R15")


def _r14(anomalies: list[dict]) -> list[dict]:
    return _rule(anomalies, "R14")


def _r13(anomalies: list[dict]) -> list[dict]:
    return _rule(anomalies, "R13")


def test_r15_request_timeout_emits():
    anomaly = _r15(_anomalies(_entry(timeout=True)))[0]

    assert anomaly["evidence"]["issue_type"] == "request-timeout"


def test_r15_err_timed_out_emits_not_r14():
    anomalies = _anomalies(_entry(failure="net::ERR_TIMED_OUT"))

    assert _r15(anomalies)
    assert not _r14(anomalies)


def test_r15_timeout_error_waiting_for_response_emits_request_timeout():
    anomalies = _anomalies(_entry(failure="TimeoutError: waiting for response timed out", response_wait_started=True))

    assert _r15(anomalies)[0]["evidence"]["issue_type"] == "request-timeout"


def test_r15_http_504_prefers_r13():
    anomalies = _anomalies(_entry(status=504, failure="timed out", response_received=True))

    assert not _r15(anomalies)
    assert _r13(anomalies)


def test_r15_http_500_prefers_r13():
    anomalies = _anomalies(_entry(status=500, response_received=True))

    assert not _r15(anomalies)
    assert _r13(anomalies)


def test_r15_connection_refused_is_r14_not_r15():
    anomalies = _anomalies(_entry(failure="net::ERR_CONNECTION_REFUSED"))

    assert not _r15(anomalies)
    assert _r14(anomalies)


def test_r15_dns_failure_is_r14_not_r15():
    anomalies = _anomalies(_entry(failure="net::ERR_NAME_NOT_RESOLVED"))

    assert not _r15(anomalies)
    assert _r14(anomalies)


def test_r15_locator_timeout_without_app_operation_is_action_error_area():
    anomalies = _anomalies(
        None,
        last_action_error=True,
        action_error="TimeoutError: locator(\"button\").click waiting for selector not found",
        timeout=True,
    )

    assert not _r15(anomalies)


def test_r15_navigation_timeout_emits():
    anomaly = _r15(
        _anomalies(
            _entry(
                url="http://app.test/products/1",
                resource_type="document",
                timeout=True,
                navigation_started=True,
                navigation_complete=False,
            )
        )
    )[0]

    assert anomaly["evidence"]["issue_type"] == "navigation-timeout"


def test_r15_async_timeout_emits_async_hang():
    anomalies = _anomalies(None, async_started=True, loading=True, pending=True, timeout=True)
    anomaly = _r15(anomalies)[0]

    assert anomaly["type"] == "async-hang"
    assert anomaly["evidence"]["issue_type"] == "async-timeout"


def test_r15_loading_without_timeout_does_not_emit():
    assert not _r15(_anomalies(None, loading=True, elapsed_seconds=2, timeout=False))


def test_r15_slow_but_completed_without_budget_does_not_emit():
    assert not _r15(_anomalies(_entry(duration_ms=5000, completed=True, response_received=True, status=200)))


def test_r15_performance_budget_exceeded_emits():
    anomaly = _r15(_anomalies(None, operation_started=True, max_latency_ms=1000, actual_latency_ms=2500))[0]

    assert anomaly["evidence"]["issue_type"] == "performance-budget-exceeded"


def test_r15_threshold_not_exceeded_does_not_emit():
    assert not _r15(_anomalies(None, operation_started=True, configured_timeout_ms=5000, elapsed_ms=2000, completed=False))


def test_r15_threshold_exceeded_emits():
    anomaly = _r15(_anomalies(None, operation_started=True, configured_timeout_ms=3000, elapsed_ms=5000, completed=False))[0]

    assert anomaly["evidence"]["configured_timeout_ms"] == 3000
    assert anomaly["evidence"]["elapsed_ms"] == 5000


def test_r15_cancellation_does_not_emit():
    assert not _r15(_anomalies(_entry(failure="net::ERR_ABORTED", timeout=False)))


def test_r15_static_image_timeout_does_not_emit():
    assert not _r15(
        _anomalies(_entry(url="http://app.test/logo.png", resource_type="image", failure="net::ERR_TIMED_OUT", timeout=True))
    )


def test_r15_xhr_timeout_emits():
    assert _r15(_anomalies(_entry(resource_type="xhr", timeout=True)))


def test_r15_network_timeout_is_not_r01_no_response():
    anomalies = _anomalies(_entry(method="POST", url="http://app.test/api/orders", timeout=True))
    types = [item["type"] for item in anomalies]

    assert _r15(anomalies)
    assert "button-no-response" not in types


def test_r15_duplicate_timeout_source_emits_one_anomaly():
    entry = _entry(timeout=True, request_id="same")
    after = _observation(runtime_entries=[entry])
    anomalies = _anomalies(entry, after=after)

    assert len(_r15(anomalies)) == 1


def test_r15_response_status_timeout_text_prefers_r13():
    anomalies = _anomalies(_entry(status=504, response_received=True, failure="timed out"))

    assert not _r15(anomalies)
    assert _r13(anomalies)


def test_r15_completed_request_with_stale_timeout_text_does_not_emit():
    anomalies = _anomalies(_entry(status=200, response_received=True, completed=True, failure="previous request timed out"))

    assert not _r15(anomalies)
