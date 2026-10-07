from dqn_error_discovery.v3.r10_dqn_confirmation_diagnostic import (
    build_funnel,
    classify_confirmation_source,
    summarize_detector_artifact,
)


def _r10_anomaly(status: str = "confirmed") -> dict:
    return {
        "rule_id": "R10",
        "status": status,
        "confirmation_policy": "strong-evidence",
        "signature": "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points",
        "evidence": {
            "request_path": "/api/product/buy",
            "field": "points",
            "pair_method": "EXACT_KEY",
            "pair_confidence": 1.0,
        },
    }


def test_same_step_lifecycle_confirmation_is_identified() -> None:
    assert classify_confirmation_source(_r10_anomaly()) == "SAME_STEP_LIFECYCLE"


def test_dqn_reproduction_confirmation_is_identified_when_trace_proves_recurrence() -> None:
    assert (
        classify_confirmation_source(
            _r10_anomaly(),
            dqn_session_id="dqn-session-1",
            same_mismatch_reobserved=True,
        )
        == "DQN_REPRODUCTION"
    )


def test_funnel_does_not_invent_dqn_session_for_same_step_confirmed_r10() -> None:
    funnel = build_funnel("site006", {"action_target": "Magic Buy"}, _r10_anomaly())

    assert funnel["r10_candidate_admitted"] is True
    assert funnel["confirmation_source"] == "SAME_STEP_LIFECYCLE"
    assert funnel["handoff_started"] is False
    assert funnel["dqn_local_session_started"] is False
    assert funnel["local_action_candidates_generated"] == 0


def test_summary_reports_context_and_reproduction_bottlenecks_without_synthetic_candidates() -> None:
    summary = summarize_detector_artifact(
        {
            "cases": [
                {
                    "site": "site006",
                    "actions": [
                        {
                            "action_target": "Magic Buy",
                            "r10_anomalies": [_r10_anomaly()],
                        }
                    ],
                }
            ]
        }
    )

    assert summary["actual_r10_candidate_emitting_cases"] == 1
    assert summary["insufficient_real_r10_candidates"] is True
    assert summary["r10_candidates_admitted"] == 1
    assert summary["controller_handoffs"] == 0
    assert summary["dqn_induced_confirmations"] == 0
    assert summary["site006_confirmation_provenance"] == "SAME_STEP_LIFECYCLE"
    assert summary["primary_bottleneck"] == "CONFIRMATION_ALREADY_AUTOMATIC"
    assert summary["markers"]["RAWD_V3_R10_CONFIRMATION_PROVENANCE_CHECK"] == "PASS"
