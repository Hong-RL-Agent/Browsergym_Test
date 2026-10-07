from __future__ import annotations

from services.layout_semantics_diagnostic_service import analyze_layout_semantics


def _evaluation(rows):
    return {"candidates": rows}


def _row(rule="R06", classification="NORMAL_SCENARIO_FP", source=None):
    return {
        "framework": "React",
        "scenario_id": "R-N01",
        "candidate_key": f"{rule}|candidate",
        "rule_id": rule,
        "route": "/",
        "target": "15",
        "source": source or {"viewport": [0, 0, 375, 812], "target_bbox": [40, 780, 100, 40], "visibility": 0.5, "overflow_dimension": "vertical", "click_blocked": False},
        "evidence_window": {"episode": 1, "step": 2, "action_type": "scroll_down"},
        "classification": classification,
        "confirmed": False,
        "before_or_new_after_temporal": "PERSISTED_FROM_V1",
    }


def test_parent_child_relation_is_preserved_when_present():
    result = analyze_layout_semantics(_evaluation([_row("R05", source={"viewport": [0, 0, 375, 812], "same_element": False, "parent_child_like": True, "intersection_area": 10, "overlap_ratio": 0.5})]), {"timeline": []}, "React")
    assert result["candidates"][0]["relationship"] == "PARENT_CHILD"


def test_r06_viewport_overflow_and_clickable_overflow_are_recorded():
    result = analyze_layout_semantics(_evaluation([_row()]), {"timeline": []}, "React")
    assert result["summary"]["viewport_overflow_fp_count"] == 1
    assert result["summary"]["visible_clickable_overflow_fp_count"] == 1


def test_hit_test_is_explicitly_unknown_without_browser_probe():
    result = analyze_layout_semantics(_evaluation([_row()]), {"timeline": []}, "React")
    assert result["candidates"][0]["hit_test"]["hit_test_blocked_by_other"] is None


def test_confirmed_geometry_only_fp_is_counted():
    row = _row()
    row["confirmed"] = True
    result = analyze_layout_semantics(_evaluation([row]), {"timeline": []}, "React")
    assert result["summary"]["confirmed_geometry_only_fp_count"] == 1


def test_actual_defect_probe_is_separate_from_runtime():
    result = analyze_layout_semantics(_evaluation([]), {"timeline": []}, "Next")
    assert result["actual_defect_probe"]["scenario_id"] == "N-D03"
    assert result["runtime_isolation"]["ground_truth_used_in_runtime"] is False


def test_no_r05_candidate_is_reported_without_inventing_one():
    result = analyze_layout_semantics(_evaluation([_row()]), {"timeline": []}, "React")
    assert result["summary"]["R05_fp_count"] == 0


def test_site_catalog_is_not_an_input_to_runtime():
    result = analyze_layout_semantics(_evaluation([_row()]), {"timeline": []}, "React")
    assert result["runtime_isolation"]["diagnostic_only"] is True
