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


def _observation(*, text: str = "Page", runtime_entries: list[dict] | None = None, ui_entities: list[dict] | None = None) -> dict:
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
            "ui_entities": ui_entities or [],
            "cart_count": 0,
        },
        "runtime_signals": {"network_entries": runtime_entries or []},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _entry(
    method: str,
    url: str,
    status: int | None = None,
    *,
    resource_type: str = "fetch",
    content_type: str = "application/json",
    response_received: bool = True,
    request_id: str = "req-1",
    **extra,
) -> dict:
    item = {
        "method": method,
        "url": url,
        "resource_type": resource_type,
        "content_type": content_type,
        "response_received": response_received,
        "request_id": request_id,
        **extra,
    }
    if status is not None:
        item["status"] = status
    return item


def _anomalies(entry: dict | None = None, *, before=None, after=None, action_type: str = "click_element", **info) -> list[dict]:
    before = before or _observation()
    after = after or _observation()
    entries = []
    if entry:
        entries.append(entry)
    anomalies = detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": action_type, "candidate_index": 0},
            "network_entries": entries,
            **info,
        },
    )
    return anomalies


def _r13(anomalies: list[dict]) -> list[dict]:
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R13"]


def test_r13_http_500_api_emits():
    anomalies = _anomalies(_entry("POST", "http://app.test/api/orders", 500))

    assert _r13(anomalies)[0]["type"] == "api-server-error"


def test_r13_http_503_api_emits():
    anomalies = _anomalies(_entry("GET", "http://app.test/api/users", 503))

    assert _r13(anomalies)


def test_r13_http_504_response_emits_not_timeout():
    anomalies = _anomalies(_entry("GET", "http://app.test/api/orders", 504, response_received=True))

    assert _r13(anomalies)
    assert _r13(anomalies)[0]["evidence"]["timeout"] is False


def test_r13_client_side_timeout_without_response_does_not_emit():
    anomalies = _anomalies(_entry("GET", "http://app.test/api/orders", None, response_received=False, timeout=True))

    assert not _r13(anomalies)


def test_r13_connection_refused_without_response_does_not_emit():
    anomalies = _anomalies(
        _entry("GET", "http://app.test/api/orders", None, response_received=False, network_error="ERR_CONNECTION_REFUSED")
    )

    assert not _r13(anomalies)


def test_r13_http_403_does_not_emit_r13_or_forbidden_without_context():
    anomalies = _anomalies(_entry("GET", "http://app.test/api/admin", 403))

    assert not _r13(anomalies)
    assert "api-forbidden" not in [item["type"] for item in anomalies]


def test_r13_http_401_does_not_emit():
    assert not _r13(_anomalies(_entry("GET", "http://app.test/api/me", 401)))


def test_r13_http_422_validation_rejection_does_not_emit():
    assert not _r13(_anomalies(_entry("POST", "http://app.test/api/forms", 422)))


def test_r13_http_404_generic_api_does_not_emit():
    assert not _r13(_anomalies(_entry("GET", "http://app.test/api/missing", 404)))


def test_r13_http_200_does_not_emit():
    assert not _r13(_anomalies(_entry("GET", "http://app.test/api/users", 200)))


def test_r13_http_200_body_error_without_server_metadata_does_not_emit():
    assert not _r13(_anomalies(_entry("GET", "http://app.test/api/users", 200, response_body={"error": "bad"})))


def test_r13_api_500_plus_frontend_typeerror_keeps_independent_r13():
    after = _observation()
    after["runtime_signals"]["page_errors"] = ["Uncaught TypeError: failed render"]
    anomalies = _anomalies(_entry("POST", "http://app.test/api/orders", 500), after=after)
    types = [item["type"] for item in anomalies]

    assert _r13(anomalies)
    assert "runtime-exception" in types


def test_r13_static_image_500_does_not_emit():
    anomalies = _anomalies(_entry("GET", "http://app.test/logo.png", 500, resource_type="image", content_type="image/png"))

    assert not _r13(anomalies)


def test_r13_font_500_does_not_emit():
    anomalies = _anomalies(_entry("GET", "http://app.test/font.woff2", 500, resource_type="font", content_type="font/woff2"))

    assert not _r13(anomalies)


def test_r13_xhr_500_emits():
    anomalies = _anomalies(_entry("POST", "http://app.test/orders", 500, resource_type="xhr"))

    assert _r13(anomalies)


def test_r13_duplicate_network_entries_emit_one_anomaly():
    entry = _entry("POST", "http://app.test/api/orders?token=secret", 500, request_id="same")
    after = _observation(runtime_entries=[entry])
    anomalies = _anomalies(entry, after=after)

    assert len(_r13(anomalies)) == 1
    assert _r13(anomalies)[0]["evidence"]["request_path"] == "/api/orders"


def test_r13_request_500_is_not_r01_no_response():
    before = _observation(text="Submit")
    after = _observation(text="Submit")
    anomalies = _anomalies(_entry("POST", "http://app.test/api/orders", 500), before=before, after=after)
    types = [item["type"] for item in anomalies]

    assert _r13(anomalies)
    assert "button-no-response" not in types


def test_r13_api_500_does_not_create_r10_truth_mismatch():
    after = _observation(ui_entities=[{"product_id": "1", "price": 9000}])
    anomalies = _anomalies(
        _entry("GET", "http://app.test/api/products/1", 500),
        after=after,
        api_entities=[{"product_id": "1", "price": 10000}],
    )

    assert _r13(anomalies)
    assert not [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R10"]
