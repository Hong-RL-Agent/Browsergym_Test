from dqn_error_discovery.hybrid.hybrid_controller import HybridController
from dqn_error_discovery.v3.r10_local_reproduction_diagnostic import _context_snapshot, _summarize


def _r10() -> dict:
    return {
        "rule_id": "R10",
        "status": "candidate",
        "signature": "R10|api-ui-mismatch|api-ui-mismatch|api_ui_semantic|/api/product/buy|points",
        "originating_action_id": "site006:trigger:magic-buy",
        "evidence": {
            "request_path": "/api/product/buy",
            "request_method": "POST",
            "field": "points",
            "network_value": 4003.2,
            "ui_value": "Earn 1.55 points on purchase",
            "pair_method": "EXACT_KEY",
            "pair_confidence": 1.0,
        },
    }


def test_context_snapshot_marks_r10_fields_available_without_changing_observation() -> None:
    snapshot = _context_snapshot(_r10())

    assert snapshot["category"]["value"] == "R10"
    assert snapshot["endpoint"]["value"] == "/api/product/buy"
    assert snapshot["semantic_key"]["value"] == "points"
    assert snapshot["network_value"]["availability"] == "AVAILABLE"
    assert snapshot["originating_target"]["availability"] == "MISSING"


def test_summary_classifies_missing_trigger_candidate_as_generation_bottleneck() -> None:
    result = _summarize("candidate_b", 42, _r10(), "NEW_CANDIDATE", HybridController(), [])

    assert result["primary_bottleneck"] == "TRIGGER_ACTION_NOT_GENERATED"
    assert result["markers"]["RAWD_V3_R10_TRIGGER_ACTION_GENERATION_CHECK"] == "BOTTLENECK"
    assert result["markers"]["RAWD_V3_R10_LOCAL_REPRODUCTION_DIAGNOSTIC_CHECK"] == "PASS"


def test_summary_classifies_valid_relevant_unselected_action_as_low_q() -> None:
    result = _summarize(
        "candidate_b",
        42,
        _r10(),
        "NEW_CANDIDATE",
        HybridController(),
        [
            {
                "local_action_candidate_count": 2,
                "exact_trigger_candidates": 1,
                "equivalent_trigger_candidates": 0,
                "reproduction_relevant_candidates": 1,
                "relevant_mask_valid": 1,
                "relevant_masked_out": 0,
                "relevant_q_ranks": [{"q_rank": 2, "action_id": 1, "q_value": 0.1}],
                "selected_action": {"reproduction_relevance": "IRRELEVANT"},
                "effect": {"generic_effect": False, "candidate_relevant_effect": False, "r10_reproduction_effect": False},
                "reward_components": {},
            }
        ],
    )

    assert result["primary_bottleneck"] == "TRIGGER_ACTION_LOW_Q"
    assert result["relevant_action_average_q_rank"] == 2
    assert result["markers"]["RAWD_V3_R10_TRIGGER_ACTION_Q_SELECTION_CHECK"] == "BOTTLENECK"


def test_summary_classifies_recreated_evidence_without_detector_refire() -> None:
    result = _summarize(
        "candidate_b",
        42,
        _r10(),
        "NEW_CANDIDATE",
        HybridController(),
        [
            {
                "local_action_candidate_count": 2,
                "exact_trigger_candidates": 0,
                "equivalent_trigger_candidates": 1,
                "reproduction_relevant_candidates": 1,
                "relevant_mask_valid": 1,
                "relevant_masked_out": 0,
                "relevant_q_ranks": [{"q_rank": 1, "action_id": 32, "q_value": 2.0}],
                "selected_action": {"reproduction_relevance": "EQUIVALENT_TRIGGER_REPRODUCTION"},
                "effect": {
                    "generic_effect": True,
                    "candidate_relevant_effect": True,
                    "r10_reproduction_effect": True,
                    "original_endpoint_hit": 1,
                    "same_key_pairs": 1,
                    "semantic_pairs": 1,
                    "same_key_not_equal_pairs": 1,
                },
                "r10_detector_refired": 0,
                "same_logical_key_recurrence": 0,
                "reward_components": {"repeat_action_penalty": -0.3},
            }
        ],
    )

    assert result["primary_bottleneck"] == "DETECTOR_RECURRENCE_FAILURE"
    assert result["repeat_penalty_affecting_trigger"] is True
    assert result["markers"]["RAWD_V3_R10_EVIDENCE_RECURRENCE_CHECK"] == "PASS"
    assert result["markers"]["RAWD_V3_R10_DETECTOR_RECURRENCE_CHECK"] == "FAIL"


def test_summary_marks_recurrence_wiring_pass_when_dqn_reproduction_confirms() -> None:
    result = _summarize(
        "candidate_b",
        42,
        _r10(),
        "NEW_CANDIDATE",
        HybridController(),
        [
            {
                "local_step": 1,
                "local_action_candidate_count": 4,
                "exact_trigger_candidates": 0,
                "equivalent_trigger_candidates": 1,
                "reproduction_relevant_candidates": 1,
                "relevant_mask_valid": 1,
                "relevant_masked_out": 0,
                "relevant_q_ranks": [{"q_rank": 1, "action_id": 32, "q_value": 2.0}],
                "selected_action": {"reproduction_relevance": "EQUIVALENT_TRIGGER_REPRODUCTION"},
                "effect": {
                    "generic_effect": True,
                    "candidate_relevant_effect": True,
                    "r10_reproduction_effect": True,
                    "original_endpoint_hit": 1,
                    "same_key_pairs": 1,
                    "semantic_pairs": 1,
                    "same_key_not_equal_pairs": 1,
                },
                "r10_detector_refired": 1,
                "same_logical_key_recurrence": 1,
                "confirmed": True,
                "dqn_reproduction_confirmed": True,
                "confirmation_sources": ["DQN_REPRODUCTION"],
                "reward_components": {},
            }
        ],
    )

    assert result["primary_bottleneck"] == "R10_REPRODUCTION_SUCCESS"
    assert result["site006_final_status"] == "CONFIRMED"
    assert result["site006_confirmation_provenance"] == "DQN_REPRODUCTION"
    assert result["first_refire_step"] == 1
    assert result["first_confirmation_step"] == 1
    assert result["markers"]["RAWD_V3_R10_RECURRENCE_WIRING_CHECK"] == "PASS"


def test_summary_marks_context_encoding_bottleneck_when_confirmation_vector_lacks_active_r10() -> None:
    result = _summarize(
        "candidate_b",
        42,
        _r10(),
        "NEW_CANDIDATE",
        HybridController(),
        [
            {
                "local_step": 1,
                "observation_shape": [984],
                "local_action_candidate_count": 32,
                "exact_trigger_candidates": 0,
                "equivalent_trigger_candidates": 1,
                "reproduction_relevant_candidates": 1,
                "relevant_mask_valid": 1,
                "relevant_masked_out": 0,
                "relevant_q_ranks": [{"q_rank": 4, "action_id": 32, "q_value": 35.0}],
                "selected_action": {"action_type": "scroll_up", "reproduction_relevance": "IRRELEVANT"},
                "effect": {"generic_effect": False, "candidate_relevant_effect": False, "r10_reproduction_effect": False},
                "feature_encoding": {
                    "active_category_encoding": "CONTROLLER_ONLY",
                    "originating_target_encoding": "NOT_ENCODED",
                    "pending_confirmation_encoding": "CONTROLLER_ONLY",
                },
                "buy_buy_click_q": {
                    "action_type": "click_element",
                    "target": "Buy Buy",
                    "candidate_index": 0,
                    "mask_valid": True,
                    "q_rank": 4,
                    "q_value": 35.0,
                },
                "scroll_up_q": {"action_type": "scroll_up", "q_rank": 1, "q_value": 36.0},
                "q_gap_scroll_minus_buy": 1.0,
                "action_type_q_summary": {
                    "click_element": {"mean": 35.0, "max": 35.0},
                    "scroll_up": {"mean": 36.0, "max": 36.0},
                },
                "reward_components": {},
            }
        ],
    )

    assert result["q_selection_primary_root_cause"] == "ACTIVE_CANDIDATE_CONTEXT_NOT_ENCODED"
    assert result["markers"]["RAWD_V3_R10_ACTIVE_CONTEXT_ENCODING_CHECK"] == "BOTTLENECK"
    assert result["markers"]["RAWD_V3_DQN_ACTION_INDEX_MAPPING_CHECK"] == "PASS"
