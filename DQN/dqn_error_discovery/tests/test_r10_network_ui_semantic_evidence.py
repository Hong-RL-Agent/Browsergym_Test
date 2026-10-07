from __future__ import annotations

from services.r10_network_ui_semantic_evidence import build_network_ui_semantic_evidence, normalize_value


def test_normalize_currency_and_number_equivalent() -> None:
    assert normalize_value("39,000원").normalized == 39000.0
    assert normalize_value(39000).normalized == 39000.0


def test_builds_not_equal_network_ui_pair_from_json_and_ui_fields() -> None:
    evidence = build_network_ui_semantic_evidence(
        action_context={"action_id": "ep1_step3_action", "action_timestamp": 1000},
        network_entries=[
            {
                "request_id": "r1",
                "relative_ms": 120,
                "request_relative_ms": 80,
                "method": "POST",
                "url": "http://app.test/api/subscription/update",
                "resource_type": "fetch",
                "status": 200,
                "response_headers": {"content-type": "application/json"},
                "response_body_available": True,
                "response_body_json": {"subscription": {"plan": "PRO", "price": 39000, "usage": 80}},
            }
        ],
        observation={
            "page_state": {
                "ui_semantic_fields": [
                    {"key": "plan", "value": "Basic", "display_text": "Plan: Basic", "element_role": "text", "visible": True},
                    {"key": "price", "value": "19,000원", "display_text": "Price: 19,000원", "element_role": "text", "visible": True},
                    {"key": "usage", "value": "80", "display_text": "Usage: 80", "element_role": "text", "visible": True},
                ]
            }
        },
    )

    assert evidence["requests"][0]["action_id"] == "ep1_step3_action"
    assert evidence["requests"][0]["normalized_url"] == "/api/subscription/update"
    assert len(evidence["network_fields"]) >= 3
    assert len(evidence["ui_fields"]) >= 3
    relations = {(row["network_key"], row["relation"]) for row in evidence["pairs"]}
    assert ("plan", "NOT_EQUAL") in relations
    assert ("price", "NOT_EQUAL") in relations
    assert ("usage", "EQUAL") in relations
    assert evidence["not_equal_pairs"] == 2
    assert evidence["equal_pairs"] == 1


def test_text_response_key_value_pairs_are_supported() -> None:
    evidence = build_network_ui_semantic_evidence(
        action_context={"action_id": "a"},
        network_entries=[
            {
                "request_id": "r2",
                "method": "GET",
                "url": "http://app.test/api/usage",
                "status": 200,
                "response_body_text": "usage: 75\nplan: PRO",
            }
        ],
        observation={"page_state": {"page_text": "usage: 70\nplan: PRO"}},
    )

    relations = {(row["network_key"], row["relation"]) for row in evidence["pairs"]}
    assert ("usage", "NOT_EQUAL") in relations
    assert ("plan", "EQUAL") in relations
