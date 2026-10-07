from dqn_error_discovery.framework_dev.r06_semantics_diagnostic import _trace_row


def test_decision_reason_is_diagnostic_only():
    row = _trace_row({
        "evidence_status": "AVAILABLE",
        "functional_impact": "MEDIUM",
        "functional_impact_reasons": ["not_reachable_by_normal_scroll", "meaningful_text_clipped"],
        "clipping_ratio": 0.8,
        "reachable_via_scroll_y": False,
        "hit_test_valid": False,
        "hit_test_blocked": False,
    })
    assert row["impact_decision_reason"] == "MEDIUM_BY_NO_SCROLL_REACHABILITY_AND_READABILITY"
    assert row["functional_impact"] == "MEDIUM"


def test_unknown_reason_distinguishes_missing_evidence():
    row = _trace_row({"evidence_status": "UNAVAILABLE", "target_resolved": False, "functional_impact": "UNKNOWN"})
    assert row["impact_decision_reason"] == "EVIDENCE_MISSING"
