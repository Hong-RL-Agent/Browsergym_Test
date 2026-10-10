from services.anomaly_detection_service import _r10_high_confidence_not_equal_pair, _standardize_anomalies


def test_detector_reads_canonical_confidence() -> None:
    pair = {
        "network_key": "status",
        "ui_key": "status",
        "network_value": "server",
        "ui_value": "client",
        "relation": "NOT_EQUAL",
        "pair_method": "EXACT_KEY",
        "confidence": 0.92,
    }
    assert _r10_high_confidence_not_equal_pair(pair) is True


def test_missing_confidence_is_not_eligible() -> None:
    pair = {
        "network_key": "status",
        "ui_key": "status",
        "network_value": "server",
        "ui_value": "client",
        "relation": "NOT_EQUAL",
        "pair_method": "EXACT_KEY",
    }
    assert _r10_high_confidence_not_equal_pair(pair) is False


def test_detector_to_candidate_preserves_pair_and_detector_confidence() -> None:
    anomaly = {
        "type": "api-ui-mismatch",
        "confidence": 0.88,
        "evidence": {
            "rule_id": "R10",
            "issue_type": "api-ui-mismatch",
            "relation": "NOT_EQUAL",
            "pair_method": "EXACT_KEY",
            "confidence": 0.92,
            "network_key": "status",
            "ui_key": "status",
            "network_value": "server",
            "ui_value": "client",
        },
    }
    rows = _standardize_anomalies(
        [anomaly],
        {"candidate_elements": [{"bid": "button:save"}]},
        {"candidate_elements": [{"bid": "button:save"}]},
        {"action": {"action_id": "a1", "action_type": "click_element", "candidate_index": 0}},
        "click_element",
    )
    assert rows[0]["pair_confidence"] == 0.92
    assert rows[0]["semantic_pair_confidence"] == 0.92
    assert rows[0]["confidence"] == 0.88
    assert rows[0]["evidence"]["confidence"] == 0.92
