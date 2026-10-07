from services.anomaly_detection_service import (
    AnomalyConfirmationTracker,
    _standardize_anomalies,
    detect_anomalies,
)


def _button(bid: str = "buy", text: str = "Buy", **extra) -> dict:
    item = {
        "bid": bid,
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
    item.update(extra)
    return item


def _observation(text: str = "Buy", *, candidates: list[dict] | None = None, runtime_entries: list[dict] | None = None) -> dict:
    return {
        "page_state": {
            "url": "http://app.test/page?ts=1",
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
        "candidate_elements": candidates or [_button()],
        "layout_signals": {},
    }


def _entry(
    method: str = "POST",
    url: str = "http://app.test/api/orders",
    *,
    status: int | None = 500,
    resource_type: str = "fetch",
    response_received: bool = True,
    request_id: str = "req-1",
    failure: str = "",
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


def _detect(before=None, after=None, *, tracker=None, step=1, candidate_index=0, entries=None, **info):
    return detect_anomalies(
        before or _observation(),
        after or _observation(),
        {
            "action": {"action_type": "click_element", "candidate_index": candidate_index},
            "step_id": step,
            "network_entries": entries or [],
            "confirmation_tracker": tracker,
            **info,
        },
    )


def _rule(items: list[dict], rule_id: str) -> list[dict]:
    return [item for item in items if item.get("rule_id") == rule_id]


def _r01(items: list[dict]) -> list[dict]:
    return _rule(items, "R01")


def _r13(items: list[dict]) -> list[dict]:
    return _rule(items, "R13")


def _r14(items: list[dict]) -> list[dict]:
    return _rule(items, "R14")


def _r15(items: list[dict]) -> list[dict]:
    return _rule(items, "R15")


def test_r01_first_observation_is_candidate_with_signature():
    anomaly = _r01(_detect())[0]

    assert anomaly["status"] == "candidate"
    assert anomaly["signature"]
    assert anomaly["rule_id"] == "R01"


def test_r01_second_independent_step_confirms():
    tracker = AnomalyConfirmationTracker()

    first = _r01(_detect(tracker=tracker, step=1))[0]
    second = _r01(_detect(tracker=tracker, step=2))[0]

    assert first["status"] == "candidate"
    assert second["status"] == "confirmed"
    assert second["confirmation"]["seen_count"] >= 2


def test_same_step_duplicate_counts_once():
    tracker = AnomalyConfirmationTracker()

    first = _r01(_detect(tracker=tracker, step=5))[0]
    second = _r01(_detect(tracker=tracker, step=5))[0]

    assert first["signature"] == second["signature"]
    assert second["confirmation"]["seen_count"] == 1
    assert second["status"] == "candidate"


def test_r06_geometry_only_reproduction_does_not_confirm():
    tracker = AnomalyConfirmationTracker()
    first = {
        "rule_id": "R06", "type": "layout-overflow", "status": "candidate",
        "confirmation_policy": "reproduce", "signature": "R06-geometry",
        "evidence": {"rule_id": "R06", "functional_impact": "NONE"},
    }
    second = dict(first, evidence={"rule_id": "R06", "functional_impact": "NONE"})
    tracker.update(first, 1)
    tracker.update(second, 2)
    assert second["status"] == "candidate"
    assert second["confirmation"]["confirmed"] is False


def test_r06_functional_clipping_reproduction_can_confirm():
    tracker = AnomalyConfirmationTracker()
    first = {
        "rule_id": "R06", "type": "layout-overflow", "status": "candidate",
        "confirmation_policy": "reproduce", "signature": "R06-functional",
        "evidence": {"rule_id": "R06", "functional_impact": "HIGH"},
    }
    second = dict(first, evidence={"rule_id": "R06", "functional_impact": "HIGH"})
    tracker.update(first, 1)
    tracker.update(second, 2)
    assert second["status"] == "confirmed"
    assert second["confirmation"]["confirmed"] is True


def test_r06_unknown_impact_does_not_confirm():
    tracker = AnomalyConfirmationTracker()
    anomaly = {
        "rule_id": "R06", "type": "layout-overflow", "status": "candidate",
        "confirmation_policy": "reproduce", "signature": "R06-unknown",
        "evidence": {"rule_id": "R06", "functional_impact": "UNKNOWN"},
    }
    tracker.update(anomaly, 1)
    tracker.update(anomaly, 2)
    assert anomaly["status"] == "candidate"


def test_r01_different_button_target_has_different_signature():
    left = _r01(_detect(before=_observation(candidates=[_button("buy-a", "Buy A")]), after=_observation(candidates=[_button("buy-a", "Buy A")])))[0]
    right = _r01(_detect(before=_observation(candidates=[_button("buy-b", "Buy B")]), after=_observation(candidates=[_button("buy-b", "Buy B")])))[0]

    assert left["signature"] != right["signature"]


def test_r13_http_500_strong_evidence_confirmed():
    anomaly = _r13(_detect(entries=[_entry(status=500)]))[0]

    assert anomaly["status"] == "confirmed"
    assert anomaly["confirmation_policy"] == "strong-evidence"


def test_server_error_text_without_status_is_not_strong_confirmed():
    anomalies = _detect(last_action_error=True, action_error="server error")

    assert not _r13(anomalies)
    assert all(item.get("status") != "confirmed" for item in anomalies if item.get("rule_id") == "R13")


def test_r14_connection_refused_strong_evidence_confirmed():
    anomaly = _r14(_detect(entries=[_entry(status=None, response_received=False, failure="net::ERR_CONNECTION_REFUSED")]))[0]

    assert anomaly["status"] == "confirmed"


def test_r15_timeout_started_incomplete_strong_evidence_confirmed():
    anomaly = _r15(_detect(entries=[_entry(status=None, response_received=False, timeout=True, request_started=True)]))[0]

    assert anomaly["status"] == "confirmed"


def test_r12_followup_policy_confirms_without_reproduction():
    anomaly = {
        "type": "persistence-mismatch",
        "confidence": 0.86,
        "evidence": {
            "rule_id": "R12",
            "issue_type": "update-persistence-mismatch",
            "operation": "update",
            "entity_id": "user-1",
            "field": "name",
            "written_value": "Lee",
            "observed_persisted_value": "Kim",
            "followup_observation_present": True,
            "refetch_detected": True,
        },
    }

    standardized = _standardize_anomalies([anomaly], _observation(), _observation(), {"action": {"action_type": "click_element"}}, "click_element")

    assert standardized[0]["confirmation_policy"] == "followup"
    assert standardized[0]["status"] == "confirmed"


def test_high_confidence_first_seen_r01_still_candidate():
    anomaly = _r01(_detect())[0]

    anomaly["confidence"] = 0.95
    standardized = _standardize_anomalies([anomaly], _observation(), _observation(), {"action": {"action_type": "click_element"}}, "click_element")

    assert standardized[0]["confidence"] == 0.95
    assert standardized[0]["status"] == "candidate"


def test_r13_query_parameters_do_not_change_signature():
    left = _r13(_detect(entries=[_entry(url="http://app.test/api/orders?ts=1", status=500)]))[0]
    right = _r13(_detect(entries=[_entry(url="http://app.test/api/orders?ts=2", status=500)]))[0]

    assert left["signature"] == right["signature"]


def test_signature_and_evidence_do_not_expose_secret_query():
    anomaly = _r13(_detect(entries=[_entry(url="http://app.test/api/user?token=SECRET", status=500)]))[0]

    assert "SECRET" not in anomaly["signature"]
    assert "SECRET" not in str(anomaly["evidence"])


def test_r10_field_mismatch_signatures_are_field_specific():
    base = {
        "type": "cart-quantity-mismatch",
        "confidence": 0.86,
        "entity_id": "product-1",
        "entity_type": "product",
        "evidence": {"rule_id": "R10", "entity_id": "product-1", "entity_type": "product"},
    }
    price = {**base, "evidence": {**base["evidence"], "field": "price", "api_value": 100, "ui_value": 200}}
    quantity = {**base, "evidence": {**base["evidence"], "field": "quantity", "api_value": 1, "ui_value": 2}}

    standardized = _standardize_anomalies([price, quantity], _observation(), _observation(), {"action": {"action_type": "click_element"}}, "click_element")

    assert standardized[0]["signature"] != standardized[1]["signature"]


def _r10_semantic_anomaly(path: str = "/api/product/buy", field: str = "points", *, relation: str = "NOT_EQUAL") -> dict:
    return {
        "type": "api-ui-mismatch",
        "confidence": 0.88,
        "entity_id": path,
        "entity_type": "api_ui_semantic",
        "evidence": {
            "rule_id": "R10",
            "issue_type": "api-ui-mismatch",
            "source": "network_ui_semantic",
            "entity_id": path,
            "entity_type": "api_ui_semantic",
            "field": field,
            "network_key": field,
            "ui_key": field,
            "relation": relation,
            "pair_method": "EXACT_KEY",
            "pair_confidence": 1.0,
            "request_path": path,
            "mismatched_fields": [{"field": field, "api_value": 4003.2, "ui_value": 1.55}],
        },
    }


def _standardize_r10(anomaly: dict, tracker: AnomalyConfirmationTracker, *, step: int, action_id: str, agent_mode: str = "PPO_GLOBAL") -> dict:
    return _standardize_anomalies(
        [anomaly],
        _observation(),
        _observation(),
        {
            "step_id": step,
            "action_id": action_id,
            "agent_mode": agent_mode,
            "action": {"action_type": "click_element", "candidate_index": 0, "action_id": action_id},
            "confirmation_tracker": tracker,
        },
        "click_element",
    )[0]


def test_r10_initial_detection_is_candidate_not_confirmed():
    tracker = AnomalyConfirmationTracker()
    anomaly = _standardize_r10(_r10_semantic_anomaly(), tracker, step=1, action_id="a1")

    assert anomaly["rule_id"] == "R10"
    assert anomaly["status"] == "candidate"
    assert anomaly["lifecycle_status"] == "DETECTED"
    assert anomaly["pending_confirmation"] is True
    assert anomaly["confirmation_policy"] == "reproduce"
    assert anomaly["confirmation"]["confirmed"] is False


def test_r10_same_step_duplicate_does_not_confirm():
    tracker = AnomalyConfirmationTracker()
    first = _standardize_r10(_r10_semantic_anomaly(), tracker, step=1, action_id="a1")
    second = _standardize_r10(_r10_semantic_anomaly(), tracker, step=1, action_id="a1")

    assert first["status"] == "candidate"
    assert second["status"] == "candidate"
    assert second["confirmation"]["seen_count"] == 1
    assert second["confirmation"].get("confirmed") is False


def test_r10_later_dqn_reproduction_confirms_same_logical_key():
    tracker = AnomalyConfirmationTracker()
    first = _standardize_r10(_r10_semantic_anomaly(), tracker, step=1, action_id="ppo-a1", agent_mode="PPO_GLOBAL")
    second = _standardize_r10(_r10_semantic_anomaly(), tracker, step=2, action_id="dqn-a2", agent_mode="DQN_LOCAL")

    assert first["signature"] == second["signature"]
    assert second["status"] == "confirmed"
    assert second["lifecycle_status"] == "CONFIRMED"
    assert second["pending_confirmation"] is False
    assert second["confirmation_source"] == "DQN_REPRODUCTION"


def test_r10_later_unrelated_key_does_not_confirm_original():
    tracker = AnomalyConfirmationTracker()
    first = _standardize_r10(_r10_semantic_anomaly(path="/api/product/buy", field="points"), tracker, step=1, action_id="a1")
    second = _standardize_r10(_r10_semantic_anomaly(path="/api/product/refund", field="points"), tracker, step=2, action_id="a2", agent_mode="DQN_LOCAL")

    assert first["signature"] != second["signature"]
    assert second["status"] == "candidate"
    assert second["confirmation"]["confirmed"] is False


def test_r10_uncertain_recurrence_does_not_confirm():
    tracker = AnomalyConfirmationTracker()
    first = _standardize_r10(_r10_semantic_anomaly(), tracker, step=1, action_id="a1")
    uncertain = _r10_semantic_anomaly(relation="PAIR_UNCERTAIN")
    uncertain["evidence"]["mismatched_fields"] = []
    second = _standardize_r10(uncertain, tracker, step=2, action_id="a2", agent_mode="DQN_LOCAL")

    assert first["status"] == "candidate"
    assert second["status"] == "candidate"
    assert second["signature"] == first["signature"]
    assert second["confirmation"]["confirmed"] is False


def test_r13_status_is_part_of_signature():
    left = _r13(_detect(entries=[_entry(status=500)]))[0]
    right = _r13(_detect(entries=[_entry(status=503)]))[0]

    assert left["signature"] != right["signature"]


def test_ground_truth_metadata_does_not_affect_signature_or_status():
    plain = _r01(_detect(before=_observation(candidates=[_button("buy-a", "Buy A")]), after=_observation(candidates=[_button("buy-a", "Buy A")])))[0]
    with_gt = _r01(
        _detect(
            before=_observation(candidates=[_button("buy-a", "Buy A", data_bug_id="BUG-SECRET")]),
            after=_observation(candidates=[_button("buy-a", "Buy A", data_bug_id="BUG-SECRET")]),
        )
    )[0]

    assert plain["signature"] == with_gt["signature"]
    assert with_gt["status"] == "candidate"
