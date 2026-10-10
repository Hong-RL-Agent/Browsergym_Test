from services.r10_network_ui_semantic_evidence import normalize_runtime_network_telemetry


def _observation(request_events, completed=(), failed=(), **context):
    return {"runtime_signals": {"temporal_network": {
        "request_events": list(request_events),
        "completed_requests": list(completed),
        "failed_requests": list(failed),
        "action_context": context,
    }}}


def test_association_is_one_row_per_request_id_and_uses_request_start() -> None:
    result = normalize_runtime_network_telemetry(_observation(
        [{"request_id": "r1", "method": "GET", "url": "/api", "request_start_timestamp": 110}],
        [{"request_id": "r1", "status": 200, "response_timestamp": 130}],
        action_id="a1", action_type="click_element", target_identity="button:save",
        action_start_timestamp=100, action_end_timestamp=120,
    ))
    assert result["normalized_request_count"] == 1
    assert result["strict_associated_request_count"] == 1
    assert result["request_id_linkage_mismatch_count"] == 0
    assert result["associations"][0]["association_type"] == "STRICT_ACTION_CAUSED"
    assert result["associations"][0]["response_timestamp"] == 130


def test_prior_action_request_completion_is_background_and_not_strict() -> None:
    result = normalize_runtime_network_telemetry(_observation(
        [],
        [{"request_id": "prior", "status": 200, "response_timestamp": 150,
          "request_start_timestamp": 90}],
        action_id="a2", action_type="click_element", target_identity="button:next",
        action_start_timestamp=100, action_end_timestamp=140,
    ))
    assert result["strict_associated_request_count"] == 0
    assert result["associations"][0]["association_type"] == "BACKGROUND_UNASSOCIATED"


def test_missing_request_start_is_temporal_only_and_not_strict() -> None:
    result = normalize_runtime_network_telemetry(_observation(
        [{"request_id": "r-missing", "method": "GET", "url": "/api"}],
        [{"request_id": "r-missing", "status": 200, "response_timestamp": 120}],
        action_id="a1", action_type="fill_input", target_identity="input:q",
        action_start_timestamp=100, action_end_timestamp=130,
    ))
    assert result["strict_associated_request_count"] == 0
    assert result["associations"][0]["association_type"] == "TEMPORAL_ONLY"
