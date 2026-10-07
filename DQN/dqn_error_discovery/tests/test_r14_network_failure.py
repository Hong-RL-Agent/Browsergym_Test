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


def _observation(text: str = "Submit", *, runtime_entries: list[dict] | None = None) -> dict:
    return {
        "page_state": {
            "url": "http://app.test/page",
            "title": "Page",
            "page_text": text,
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": 1,
            "viewport_width": 800,
            "viewport_height": 600,
            "cart_count": 0,
        },
        "runtime_signals": {"network_entries": runtime_entries or []},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _entry(
    method: str = "GET",
    url: str = "http://app.test/api/users",
    *,
    status: int | None = None,
    failure: str = "net::ERR_CONNECTION_REFUSED",
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


def _r14(anomalies: list[dict]) -> list[dict]:
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R14"]


def _r13(anomalies: list[dict]) -> list[dict]:
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R13"]


def test_r14_connection_refused_emits():
    anomaly = _r14(_anomalies(_entry(failure="net::ERR_CONNECTION_REFUSED")))[0]

    assert anomaly["type"] == "network-failure"
    assert anomaly["evidence"]["issue_type"] == "connection-refused"


def test_r14_connection_reset_emits():
    assert _r14(_anomalies(_entry(failure="net::ERR_CONNECTION_RESET")))[0]["evidence"]["issue_type"] == "connection-reset"


def test_r14_dns_failure_emits():
    assert _r14(_anomalies(_entry(failure="net::ERR_NAME_NOT_RESOLVED")))[0]["evidence"]["issue_type"] == "dns-failure"


def test_r14_fetch_failed_emits():
    assert _r14(_anomalies(_entry(failure="Failed to fetch", resource_type="xhr")))


def test_r14_http_500_does_not_emit_r14():
    anomalies = _anomalies(_entry(status=500, failure="failed", response_received=True))

    assert not _r14(anomalies)
    assert _r13(anomalies)


def test_r14_http_503_does_not_emit_r14():
    anomalies = _anomalies(_entry(status=503, failure="", response_received=True))

    assert not _r14(anomalies)
    assert _r13(anomalies)


def test_r14_http_404_response_does_not_emit():
    assert not _r14(_anomalies(_entry(status=404, failure="", response_received=True)))


def test_r14_client_timeout_does_not_emit():
    assert not _r14(_anomalies(_entry(failure="Timeout 30000ms exceeded", timeout=True)))


def test_r14_err_timed_out_does_not_emit():
    assert not _r14(_anomalies(_entry(failure="net::ERR_TIMED_OUT")))


def test_r14_http_504_response_does_not_emit_r14():
    anomalies = _anomalies(_entry(status=504, failure="", response_received=True))

    assert not _r14(anomalies)
    assert _r13(anomalies)


def test_r14_navigation_cancellation_does_not_emit():
    assert not _r14(_anomalies(_entry(failure="net::ERR_ABORTED", navigation_changed=True)))


def test_r14_cancelled_only_does_not_emit():
    assert not _r14(_anomalies(_entry(failure="request cancelled")))


def test_r14_static_image_failure_does_not_emit():
    assert not _r14(_anomalies(_entry(url="http://app.test/logo.png", failure="net::ERR_CONNECTION_RESET", resource_type="image")))


def test_r14_font_failure_does_not_emit():
    assert not _r14(_anomalies(_entry(url="http://app.test/font.woff2", failure="net::ERR_CONNECTION_RESET", resource_type="font")))


def test_r14_xhr_connection_reset_emits():
    assert _r14(_anomalies(_entry(failure="net::ERR_CONNECTION_RESET", resource_type="xhr")))


def test_r14_document_connection_refused_emits():
    assert _r14(_anomalies(_entry(url="http://app.test/products/1", failure="net::ERR_CONNECTION_REFUSED", resource_type="document")))


def test_r14_action_error_only_does_not_emit():
    assert not _r14(_anomalies(None, last_action_error=True, action_error="selector stale"))


def test_r14_network_failure_is_not_r01_no_response():
    before = _observation("Submit")
    after = _observation("Submit")
    anomalies = _anomalies(_entry(method="POST", url="http://app.test/api/orders", failure="net::ERR_CONNECTION_REFUSED"), before=before, after=after)
    types = [item["type"] for item in anomalies]

    assert _r14(anomalies)
    assert "button-no-response" not in types


def test_r14_duplicate_source_emits_one_anomaly():
    entry = _entry(failure="net::ERR_CONNECTION_RESET", request_id="same")
    after = _observation(runtime_entries=[entry])
    anomalies = _anomalies(entry, after=after)

    assert len(_r14(anomalies)) == 1


def test_r14_status_plus_network_error_prefers_r13():
    anomalies = _anomalies(_entry(status=500, failure="net::ERR_CONNECTION_RESET", response_received=True))

    assert not _r14(anomalies)
    assert _r13(anomalies)
