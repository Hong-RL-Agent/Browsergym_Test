from dqn_error_discovery.v3.r10_confirmation_lifecycle_check import evaluate_lifecycle


def _summary(status: str = "candidate") -> dict:
    return {
        "cases": [
            {
                "site": "site006",
                "url": "http://localhost:9115/",
                "r10_detector_fired": 1,
                "r10_candidates": [
                    {
                        "rule_id": "R10",
                        "signature": "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points",
                        "logical_key": "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points",
                        "status": status,
                        "lifecycle_status": "CONFIRMED" if status == "confirmed" else "DETECTED",
                        "pending_confirmation": status != "confirmed",
                    }
                ],
            },
            {"site": "site007", "url": "http://localhost:9116/", "r10_detector_fired": 0, "r10_candidates": []},
            {"site": "site008", "url": "http://localhost:9117/", "r10_detector_fired": 0, "r10_candidates": []},
        ]
    }


def test_r10_detected_candidate_handoffs_to_dqn_and_remains_unconfirmed_without_reproduction() -> None:
    result = evaluate_lifecycle(_summary())

    assert result["r10_candidates_admitted"] == 1
    assert result["initial_detected"] == 1
    assert result["initial_same_step_confirmed"] == 0
    assert result["pending_confirmation"] == 1
    assert result["controller_handoffs"] == 1
    assert result["dqn_local_sessions"] == 1
    assert result["detected_not_confirmed"] == 1
    assert result["site006_handoff_result"] == "NEW_CANDIDATE"
    assert result["site006_dqn_session_result"] is True
    assert result["markers"]["RAWD_V3_R10_CONFIRMATION_LIFECYCLE_CHECK"] == "PASS"


def test_same_step_confirmed_r10_fails_lifecycle_separation_check() -> None:
    result = evaluate_lifecycle(_summary(status="confirmed"))

    assert result["initial_same_step_confirmed"] == 1
    assert result["markers"]["RAWD_V3_R10_SAME_STEP_CONFIRMATION_GUARD_CHECK"] == "FAIL"
    assert result["markers"]["RAWD_V3_R10_CONFIRMATION_LIFECYCLE_CHECK"] == "FAIL"
