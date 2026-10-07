from services.anomaly_detection_service import AnomalyConfirmationTracker, detect_anomalies


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


def _observation(
    ui_entities: list[dict],
    *,
    api_entities: list[dict] | None = None,
    loading: bool = False,
) -> dict:
    text = " ".join(str(value) for entity in ui_entities for value in entity.values())
    return {
        "page_state": {
            "url": "http://app.test/products/1",
            "title": "Detail",
            "page_text": "Loading" if loading else text,
            "page_text_sample": "Loading" if loading else text,
            "page_text_length": len(text),
            "dom_node_count": len(ui_entities) + 1,
            "viewport_width": 800,
            "viewport_height": 600,
            "ui_entities": ui_entities,
            "api_entities": api_entities or [],
            "cart_count": 0,
            "loading": loading,
        },
        "runtime_signals": {"loading": loading},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _r10(before: dict, after: dict, api_entities: list[dict], **info) -> list[dict]:
    anomalies = detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": "inspect_network", "candidate_index": 0},
            "response_status": 200,
            "api_entities": api_entities,
            **info,
        },
    )
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R10"]


def test_r10_same_entity_same_price_does_not_emit():
    api = [{"product_id": "1", "price": 10000}]
    after = _observation([{"product_id": "1", "price": 10000}])

    assert not _r10(_observation([]), after, api)


def test_r10_same_entity_price_mismatch_emits():
    api = [{"product_id": "1", "price": 10000}]
    after = _observation([{"product_id": "1", "price": 12000}])

    anomaly = _r10(_observation([]), after, api)[0]
    assert anomaly["type"] == "product-detail-mismatch"
    assert anomaly["evidence"]["field"] == "price"


def test_r10_price_formatting_difference_does_not_emit():
    api = [{"product_id": "1", "price": 12000}]
    after = _observation([{"product_id": "1", "price": "12,000원"}])

    assert not _r10(_observation([]), after, api)


def test_r10_quantity_mismatch_emits_cart_quantity_mismatch():
    api = [{"product_id": "1", "quantity": 2}]
    after = _observation([{"product_id": "1", "quantity": 1}])

    anomaly = _r10(_observation([]), after, api)[0]
    assert anomaly["type"] == "cart-quantity-mismatch"


def test_r10_different_entity_does_not_emit():
    api = [{"product_id": "1", "price": 10000}]
    after = _observation([{"product_id": "2", "price": 12000}])

    assert not _r10(_observation([]), after, api)


def test_r10_api_500_does_not_emit():
    api = [{"product_id": "1", "price": 10000}]
    after = _observation([{"product_id": "1", "price": 12000}])

    assert not _r10(_observation([]), after, api, response_status=500)


def test_r10_network_failure_does_not_emit():
    api = [{"product_id": "1", "price": 10000}]
    after = _observation([{"product_id": "1", "price": 12000}])

    assert not _r10(_observation([]), after, api, network_error="connection reset")


def test_r10_loading_ui_does_not_emit():
    api = [{"product_id": "1", "price": 10000}]
    after = _observation([{"product_id": "1", "price": 9000}], loading=True)

    assert not _r10(_observation([]), after, api)


def test_r10_stale_ui_after_settled_emits_stale_data_rendering():
    before = _observation(
        [{"product_id": "1", "status": "pending"}],
        api_entities=[{"product_id": "1", "status": "pending"}],
    )
    api = [{"product_id": "1", "status": "completed"}]
    after = _observation([{"product_id": "1", "status": "pending"}])

    anomaly = _r10(before, after, api)[0]
    assert anomaly["type"] == "stale-data-rendering"


def test_r10_same_text_different_field_semantics_does_not_emit():
    api = [{"product_id": "1", "quantity": 2}]
    after = _observation([{"product_id": "1", "total": 2}])

    assert not _r10(_observation([]), after, api)


def test_r10_numeric_representation_does_not_emit():
    api = [{"product_id": "1", "quantity": 2}]
    after = _observation([{"product_id": "1", "quantity": "2"}])

    assert not _r10(_observation([]), after, api)


def test_r10_multiple_field_mismatch_groups_evidence():
    api = [{"product_id": "1", "price": 100, "quantity": 2}]
    after = _observation([{"product_id": "1", "price": 200, "quantity": 3}])

    anomaly = _r10(_observation([]), after, api)[0]
    fields = {item["field"] for item in anomaly["evidence"]["mismatched_fields"]}
    assert fields == {"price", "quantity"}


def test_r10_r09_filter_case_api_and_ui_agree_does_not_emit():
    api = [{"user_id": "42", "status": "inactive"}]
    after = _observation([{"user_id": "42", "status": "inactive"}])

    assert not _r10(_observation([]), after, api, filter_key="status", filter_value="active")


def test_r10_does_not_calculate_total_without_api_total_field():
    api = [{"product_id": "1", "unit_price": 100, "quantity": 2}]
    after = _observation([{"product_id": "1", "total": 300}])

    assert not _r10(_observation([]), after, api)


def test_r10_direct_api_total_mismatch_emits():
    api = [{"product_id": "1", "total": 200}]
    after = _observation([{"product_id": "1", "total": 300}])

    anomaly = _r10(_observation([]), after, api)[0]
    assert anomaly["evidence"]["field"] == "total"


def _semantic_observation(pairs: list[dict], *, settled_pairs: list[dict] | None = None, loading: bool = False) -> dict:
    obs = _observation([], loading=loading)
    semantic = {
        "action_id": "act-r10",
        "pairs": pairs,
        "gt_runtime_access": False,
    }
    if settled_pairs is not None:
        semantic["settled_pairs"] = settled_pairs
    obs["page_state"]["network_ui_semantic"] = semantic
    obs["runtime_signals"]["network_ui_semantic"] = semantic
    return obs


def _semantic_pair(
    *,
    relation: str = "NOT_EQUAL",
    method: str = "EXACT_KEY",
    confidence: float = 1.0,
    network_value=39000,
    ui_value=19000,
    key: str = "price",
) -> dict:
    return {
        "network_key": key,
        "network_path": key,
        "network_value": network_value,
        "network_semantic_value": {"raw": network_value, "normalized": float(network_value), "value_type": "number", "unit": ""},
        "ui_key": key,
        "ui_value": ui_value,
        "ui_semantic_value": {"raw": ui_value, "normalized": float(ui_value), "value_type": "number", "unit": ""},
        "match_basis": "normalized_key",
        "relation": relation,
        "pair_confidence": confidence,
        "pair_method": method,
        "request_id": "req-1",
        "url": "http://app.test/api/subscription/update",
        "status": 200,
    }


def test_r10_semantic_high_confidence_not_equal_emits():
    after = _semantic_observation([_semantic_pair()])

    anomaly = _r10(_observation([]), after, [], response_status=200)[0]

    assert anomaly["type"] == "api-ui-mismatch"
    assert anomaly["rule_id"] == "R10"
    assert anomaly["evidence"]["source"] == "network_ui_semantic"
    assert anomaly["evidence"]["field"] == "price"
    assert anomaly["evidence"]["request_path"] == "/api/subscription/update"
    assert anomaly["status"] == "candidate"
    assert anomaly["lifecycle_status"] == "DETECTED"
    assert anomaly["pending_confirmation"] is True


def test_r10_semantic_equal_does_not_emit():
    after = _semantic_observation([_semantic_pair(relation="EQUAL", network_value=39000, ui_value=39000)])

    assert not _r10(_observation([]), after, [], response_status=200)


def test_r10_semantic_uncertain_does_not_emit():
    after = _semantic_observation([_semantic_pair(relation="PAIR_UNCERTAIN", method="EXACT_KEY")])

    assert not _r10(_observation([]), after, [], response_status=200)


def test_r10_semantic_missing_ui_does_not_emit():
    pair = _semantic_pair()
    pair["ui_value"] = ""

    after = _semantic_observation([pair])

    assert not _r10(_observation([]), after, [], response_status=200)


def test_r10_semantic_settled_equal_suppresses_transient_mismatch():
    transient = _semantic_pair(relation="NOT_EQUAL", network_value=39000, ui_value=19000)
    settled = _semantic_pair(relation="EQUAL", network_value=39000, ui_value=39000)
    after = _semantic_observation([transient], settled_pairs=[settled])

    assert not _r10(_observation([]), after, [], response_status=200)


def test_r10_semantic_empty_settled_pairs_falls_back_to_fresh_pairs():
    after = _semantic_observation([_semantic_pair()], settled_pairs=[])

    anomaly = _r10(_observation([]), after, [], response_status=200)[0]

    assert anomaly["type"] == "api-ui-mismatch"
    assert anomaly["evidence"]["request_path"] == "/api/subscription/update"


def test_r10_semantic_normalized_url_keeps_stable_logical_path():
    pair = _semantic_pair()
    pair.pop("url", None)
    pair["normalized_url"] = "http://app.test/api/product/buy?nonce=123"
    after = _semantic_observation([pair])

    anomaly = _r10(_observation([]), after, [], response_status=200)[0]

    assert anomaly["evidence"]["request_path"] == "/api/product/buy"
    assert "/api/product/buy" in anomaly["signature"]


def test_r10_dqn_post_action_recurrence_confirms_with_fresh_pairs():
    tracker = AnomalyConfirmationTracker()
    initial_pair = _semantic_pair(key="points", network_value=4003.2, ui_value=1.55)
    initial_pair["url"] = "http://app.test/api/product/buy"
    initial = _r10(
        _observation([]),
        _semantic_observation([initial_pair]),
        [],
        response_status=200,
        step_id=1,
        action={"action_type": "click_element", "action_id": "ppo-magic-buy", "candidate_index": 0},
        agent_mode="PPO_GLOBAL",
        confirmation_tracker=tracker,
    )[0]

    recurrent_pair = _semantic_pair(key="points", network_value=5003.2, ui_value=1.55)
    recurrent_pair.pop("url", None)
    recurrent_pair["normalized_url"] = "http://app.test/api/product/buy?request_id=abc"
    recurrent = _r10(
        _semantic_observation([]),
        _semantic_observation([recurrent_pair], settled_pairs=[]),
        [],
        response_status=200,
        step_id=2,
        action={"action_type": "click_element", "action_id": "dqn-buy-buy", "candidate_index": 0},
        agent_mode="DQN_LOCAL",
        confirmation_tracker=tracker,
    )[0]

    assert initial["signature"] == recurrent["signature"]
    assert recurrent["status"] == "confirmed"
    assert recurrent["confirmation_source"] == "DQN_REPRODUCTION"


def test_r10_duplicate_recurrence_event_does_not_double_confirm():
    tracker = AnomalyConfirmationTracker()
    pair = _semantic_pair(key="points", network_value=4003.2, ui_value=1.55)
    pair["url"] = "http://app.test/api/product/buy"
    _r10(
        _observation([]),
        _semantic_observation([pair]),
        [],
        response_status=200,
        step_id=1,
        action={"action_type": "click_element", "action_id": "ppo-magic-buy", "candidate_index": 0},
        agent_mode="PPO_GLOBAL",
        confirmation_tracker=tracker,
    )

    first = _r10(
        _semantic_observation([]),
        _semantic_observation([pair], settled_pairs=[]),
        [],
        response_status=200,
        step_id=2,
        action={"action_type": "click_element", "action_id": "dqn-buy-buy", "candidate_index": 0},
        agent_mode="DQN_LOCAL",
        confirmation_tracker=tracker,
    )[0]
    duplicate = _r10(
        _semantic_observation([]),
        _semantic_observation([pair], settled_pairs=[]),
        [],
        response_status=200,
        step_id=2,
        action={"action_type": "click_element", "action_id": "dqn-buy-buy", "candidate_index": 0},
        agent_mode="DQN_LOCAL",
        confirmation_tracker=tracker,
    )[0]

    assert first["status"] == "confirmed"
    assert duplicate["confirmation"]["seen_count"] == first["confirmation"]["seen_count"]


def test_r10_semantic_duplicate_signature_is_stable():
    after = _semantic_observation([_semantic_pair(network_value=39000, ui_value=19000), _semantic_pair(network_value=39000, ui_value=12000)])

    anomalies = _r10(_observation([]), after, [], response_status=200)

    assert len(anomalies) == 1
    assert anomalies[0]["signature"].endswith("|api_ui_semantic|/api/subscription/update|price")
