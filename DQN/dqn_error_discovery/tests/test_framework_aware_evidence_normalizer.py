from services.framework_aware_evidence_normalizer import normalize_network_event


def _ctx(action_id="a1"):
    return {"action_id": action_id, "action_type": "click_element", "target_id": "bid:7"}


def test_static_assets_are_irrelevant_and_raw_is_preserved():
    event = {"request_id": "r1", "url": "https://app.test/assets/app.js", "resource_type": "script", "status": 200}
    result = normalize_network_event(event, action_context=_ctx())
    assert result["semantic_class"] == "STATIC_ASSET"
    assert result["functional_relevance"] == "IRRELEVANT"
    assert result["raw_network_event"] == event


def test_next_prefetch_is_not_a_functional_request():
    event = {
        "request_id": "r2",
        "url": "https://app.test/_next/data/build/page.json",
        "resource_type": "fetch",
        "headers": {"purpose": "prefetch"},
        "status": 200,
    }
    result = normalize_network_event(event, action_context=_ctx())
    assert result["framework_context"] == "NEXT"
    assert result["semantic_class"] == "PREFETCH"
    assert result["functional_relevance"] == "IRRELEVANT"


def test_next_rsc_navigation_is_framework_navigation():
    event = {
        "request_id": "r3",
        "url": "https://app.test/dashboard",
        "resource_type": "fetch",
        "headers": {"rsc": "1", "next-router-state-tree": "tree"},
        "status": 200,
    }
    result = normalize_network_event(event, action_context=_ctx())
    assert result["semantic_class"] == "FRAMEWORK_NAVIGATION"


def test_action_correlated_mutation_is_functional():
    event = {"request_id": "r4", "url": "https://app.test/api/cart", "method": "POST", "resource_type": "fetch", "status": 201}
    result = normalize_network_event(event, action_context=_ctx())
    assert result["semantic_class"] == "USER_FUNCTIONAL_REQUEST"
    assert result["functional_relevance"] == "RELEVANT"
    assert result["trigger_action_id"] == "a1"


def test_transport_and_http_failures_are_preserved():
    http = normalize_network_event({"url": "https://app.test/api/orders", "method": "POST", "status": 500}, action_context=_ctx())
    transport = normalize_network_event({"url": "https://app.test/api/orders", "event": "requestfailed", "network_failure": True}, action_context=_ctx())
    assert http["http_failed"] is True and http["transport_failed"] is False
    assert transport["transport_failed"] is True and transport["http_failed"] is False


def test_ambiguous_event_is_not_aggressively_suppressed():
    result = normalize_network_event({"url": "https://app.test/resource", "resource_type": "fetch", "status": 200})
    assert result["semantic_class"] == "UNKNOWN"
    assert result["functional_relevance"] == "UNKNOWN"


def test_dev_websocket_is_development_infrastructure():
    result = normalize_network_event({"url": "ws://app.test/webpack-hmr", "resource_type": "websocket"})
    assert result["semantic_class"] == "DEVELOPMENT_INFRASTRUCTURE"
    assert result["functional_relevance"] == "IRRELEVANT"
