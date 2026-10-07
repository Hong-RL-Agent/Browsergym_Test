from __future__ import annotations

from dqn_error_discovery.framework_dev.scenario_evaluator import (
    _classify,
    compare_evaluations,
    load_scenario_catalog,
    validate_catalog,
)


def _scenario(scenario_id="R-N01", scenario_type="NORMAL", expected="NO_DEFECT"):
    return {"scenario": {"scenario_id": scenario_id, "normal_or_defect": scenario_type, "expected_detector_result": expected}}


def test_catalog_validation_rejects_duplicate_ids():
    catalog = {"scenarios": [{"scenario_id": "x", "scenario_type": "NORMAL", "trigger_action": "click", "expected_final_behavior": "ok"}, {"scenario_id": "x", "scenario_type": "DEFECT", "trigger_action": "click", "expected_final_behavior": "bad"}]}
    assert not validate_catalog(catalog)["valid"]


def test_normal_scenario_candidate_is_fp():
    assert _classify(_scenario(), "R06", False) == "NORMAL_SCENARIO_FP"


def test_defect_matching_and_wrong_rule_are_distinct():
    assert _classify(_scenario("R-D02", "DEFECT", "R05"), "R05", False) == "DEFECT_SCENARIO_TRUE_POSITIVE"
    assert _classify(_scenario("R-D02", "DEFECT", "R05"), "R06", False) == "DEFECT_SCENARIO_WRONG_RULE"


def test_trajectory_exposure_classification_is_explicit():
    assert _classify(_scenario("R-N01", "NORMAL"), "R06", True) == "TRAJECTORY_EXPOSURE_NORMAL_FP"
    assert _classify(_scenario("R-D02", "DEFECT", "R05"), "R05", True) == "TRAJECTORY_EXPOSURE_TRUE_DEFECT"


def test_unmapped_is_not_a_false_negative():
    assert _classify(None, "R06", False) == "UNMAPPED"


def test_compare_preserves_candidate_identity():
    before = {"candidates": [{"candidate_key": "a", "classification": "NORMAL_SCENARIO_FP", "mapping_confidence": "HIGH", "confirmed": False}]}
    after = {"candidates": [{"candidate_key": "a", "classification": "NORMAL_SCENARIO_FP", "mapping_confidence": "HIGH", "confirmed": False}, {"candidate_key": "b", "classification": "UNMAPPED", "mapping_confidence": "UNMAPPED", "confirmed": False}]}
    result = compare_evaluations(before, after)
    assert result["new_after_temporal"] == ["b"]
    assert result["removed_by_temporal"] == []


def test_catalog_files_are_evaluation_only():
    catalog = load_scenario_catalog("development_sites/react_async/SCENARIOS.json")
    assert all(item["evaluation_only"] for item in catalog["scenarios"])
