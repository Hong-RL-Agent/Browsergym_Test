from __future__ import annotations

from dqn_error_discovery.v3.evidence_candidate_conversion_diagnostic import (
    _candidate_gate_trace,
    _raw_evidence,
    _rule_trace,
    _semantic_evidence,
)


def test_raw_and_semantic_evidence_are_diagnostic_only():
    before = {"page_state": {"url": "http://test/"}, "candidate_elements": [{"bid": 1}]}
    after = {"page_state": {"url": "http://test/"}, "candidate_elements": [{"bid": 1}]}
    raw = _raw_evidence(before, after, {}, {"action_type": "click_element", "candidate_index": 0})
    semantic = _semantic_evidence([], {}, raw)
    assert raw["no_effect"] is True
    assert semantic["types"] == []


def test_emitted_rule_and_candidate_gate_are_separate():
    anomaly = {
        "rule_id": "R01",
        "issue_type": "button-no-response",
        "type": "button-no-response",
        "confidence": 0.5,
        "status": "candidate",
        "action_type": "click_element",
        "target_id": "bid:1",
        "signature": "R01:test",
        "evidence": {"rule_id": "R01", "target": {"bid": 1}},
    }
    before = {"page_state": {"url": "http://test/"}, "candidate_elements": [{"bid": 1, "role": "button"}]}
    after = {"page_state": {"url": "http://test/", "form_state": "changed"}, "candidate_elements": [{"bid": 1, "role": "button"}]}
    action = {"action_type": "click_element", "candidate_index": 0}
    rules = _rule_trace([anomaly], before, after, action, {})
    gates = _candidate_gate_trace([anomaly], before, after, action, {}, {})
    assert rules["rules_passed"] == ["R01"]
    assert gates["anomalies"][0]["candidate_eligible"] is True


def test_no_gt_or_reserved_sites_are_embedded_in_trace_helpers():
    assert "SITE002" not in _raw_evidence({}, {}, {}, {}).get("site", "")
    assert "SITE003" not in _raw_evidence({}, {}, {}, {}).get("site", "")
