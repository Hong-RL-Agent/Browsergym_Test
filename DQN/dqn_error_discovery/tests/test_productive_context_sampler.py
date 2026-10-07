from dqn_error_discovery.v3.productive_context_sampler import classify_context, semantic_fingerprint


def _obs(url="http://test/", bid="a"):
    return {"page_state": {"url": url, "page_text_sample": "page"}, "candidate_elements": [{"bid": bid, "role": "button", "tag": "button"}]}


def test_candidate_context_is_tier_one():
    result = classify_context(_obs(), _obs(), [{"status": "candidate", "signature": "r01"}], {})
    assert result["admit"] is True
    assert result["context_tier"] == "TIER1_CANDIDATE_CONTEXT"


def test_evidence_context_is_tier_two():
    result = classify_context(_obs(), _obs("http://test/changed"), [{"rule_id": "R13", "evidence": {"network": "failed"}}], {})
    assert result["context_tier"] == "TIER2_DETECTOR_OPPORTUNITY"
    assert "network_failure" in result["evidence_types"]


def test_repeated_zero_gain_context_is_skipped():
    result = classify_context(_obs(), _obs(), [], {}, previous_zero_gain=True)
    assert result["admit"] is False
    assert result["skip_reason"] == "REPEATED_ZERO_GAIN_NO_CHANGE"


def test_semantic_fingerprint_is_stable():
    assert semantic_fingerprint(_obs()) == semantic_fingerprint(_obs())
