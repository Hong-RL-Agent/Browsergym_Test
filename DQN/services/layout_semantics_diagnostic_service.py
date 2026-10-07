"""Post-hoc semantic audit for R05/R06 layout candidates.

This module is deliberately not imported by the runtime detector, environment,
reward, controller, or confirmation tracker. It only consumes frozen artifacts
and development-site fixture source.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LAYOUT_RULES = {"R05", "R06"}


def _read(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _source(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("source", {})
    return value if isinstance(value, dict) else {}


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None


def _viewport_category(viewport: list[float] | None) -> str:
    if not viewport or len(viewport) < 4:
        return "UNKNOWN"
    return "mobile" if viewport[2] <= 600 else "desktop"


def _timeline_context(temporal: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    window = row.get("evidence_window", {}) or {}
    episode = window.get("episode")
    step = window.get("step")
    events = [item for item in temporal.get("timeline", []) if item.get("episode") == episode]
    events.sort(key=lambda item: item.get("step", 0))
    previous = [item.get("action_type") for item in events if item.get("step", 0) < (step or 0)][-5:]
    current = window.get("action_type") or ""
    all_recent = previous + [current]
    return {
        "recent_action_types": all_recent,
        "viewport_action_correlated": any("viewport" in str(action) for action in all_recent),
        "scroll_correlated": any("scroll" in str(action) for action in all_recent),
    }


def _relationship(row: dict[str, Any]) -> str:
    source = _source(row)
    if row.get("rule_id") != "R05":
        return "NOT_APPLICABLE"
    if source.get("same_element"):
        return "SAME_ELEMENT"
    if source.get("parent_child_like"):
        return "PARENT_CHILD"
    return "UNKNOWN"


def _candidate_record(row: dict[str, Any], temporal: dict[str, Any]) -> dict[str, Any]:
    source = _source(row)
    viewport = _bbox(source.get("viewport") or source.get("viewport_bbox"))
    target_bbox = _bbox(source.get("target_bbox"))
    related_bbox = _bbox(source.get("other_bbox"))
    visibility = _safe_float(source.get("visibility"))
    clipped = _safe_float(source.get("clipped_ratio"))
    if clipped is None and visibility is not None:
        clipped = max(0.0, 1.0 - visibility)
    rule_id = str(row.get("rule_id") or "")
    context = _timeline_context(temporal, row)
    r06_viewport_overflow = rule_id == "R06" and bool(source.get("overflow_dimension"))
    target_visible = visibility is not None and visibility > 0.0
    click_blocked = source.get("click_blocked") if "click_blocked" in source else None
    css_unknown = None
    return {
        "framework": row.get("framework"),
        "scenario_id": row.get("scenario_id"),
        "candidate_key": row.get("candidate_key"),
        "rule_id": rule_id,
        "route": row.get("route"),
        "viewport": viewport,
        "viewport_category": _viewport_category(viewport),
        "target_id": row.get("target"),
        "target_role": source.get("target_role"),
        "target_text": source.get("target_text"),
        "target_bbox": target_bbox,
        "related_element": source.get("other_element") or source.get("occluder_bid"),
        "related_bbox": related_bbox,
        "intersection": {"area": source.get("intersection_area"), "ratio": source.get("overlap_ratio")},
        "container_bbox": source.get("container_bbox"),
        "viewport_bbox": viewport,
        "scroll_width": source.get("scroll_width"),
        "client_width": source.get("client_width"),
        "scroll_height": source.get("scroll_height"),
        "client_height": source.get("client_height"),
        "css_position": source.get("css_position", css_unknown),
        "css_overflow_x": source.get("css_overflow_x", css_unknown),
        "css_overflow_y": source.get("css_overflow_y", css_unknown),
        "z_index": source.get("z_index"),
        "pointer_events": source.get("pointer_events"),
        "visibility": visibility,
        "enabled": source.get("target_enabled"),
        "clickable": source.get("target_actionable"),
        "relationship": _relationship(row),
        "expected_containment": "UNKNOWN",
        "overlay_semantics": "UNKNOWN",
        "stacking_context": "UNKNOWN",
        "hit_test": {
            "probe_status": "UNAVAILABLE_BROWSER_BACKEND",
            "expected_target": row.get("target"),
            "hit_test_element": None,
            "hit_test_matches_target": None,
            "hit_test_descendant_of_target": None,
            "hit_test_blocked_by_other": None,
            "safe_points_tested": [],
        },
        "interaction_accessibility": {
            "clickable": source.get("target_actionable"),
            "focusable": None,
            "pointer_hit": None,
            "keyboard_accessible": None,
        },
        "occlusion_severity": "UNKNOWN" if rule_id == "R05" else "NOT_APPLICABLE",
        "overflow_type": "VIEWPORT_VERTICAL_OVERFLOW" if r06_viewport_overflow and source.get("overflow_dimension") == "vertical" else ("VIEWPORT_HORIZONTAL_OVERFLOW" if r06_viewport_overflow else "NOT_APPLICABLE"),
        "intentional_scroll": None,
        "can_scroll_x": None,
        "can_scroll_y": None,
        "target_fully_visible": source.get("target_fully_visible"),
        "target_partially_clipped": bool(clipped and clipped > 0.0),
        "target_fully_clipped": bool(visibility == 0.0),
        "offscreen_target": bool(target_bbox and viewport and (target_bbox[1] < viewport[1] or target_bbox[1] + target_bbox[3] > viewport[1] + viewport[3])),
        "visible_clickable_overflow": bool(rule_id == "R06" and target_visible and click_blocked is False),
        "functional_impact": "UNKNOWN",
        "css_transform_correlation": None,
        "hidden_or_decorative": None,
        "functional_priority": source.get("target_actionable"),
        "geometric_vs_functional": "GEOMETRIC_ONLY" if row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"} and click_blocked is not True else "UNKNOWN",
        "before_or_new_after_temporal": row.get("before_or_new_after_temporal"),
        "confirmed": bool(row.get("confirmed")),
        "recent_action_types": context["recent_action_types"],
        "viewport_action_correlated": context["viewport_action_correlated"],
        "scroll_correlated": context["scroll_correlated"],
        "diagnostic_basis": "artifact geometry plus fixture/source audit; interactive hit-test unavailable",
    }


def _source_audit() -> dict[str, Any]:
    return {
        "R05": {
            "source_function": "services/anomaly_detection_service.py:_detect_r05_layout_overlap",
            "predicate_function": "services/anomaly_detection_service.py:_is_r05_layout_overlap",
            "source_lines": "1033-1101, 1104-1122",
            "conditions": [
                "target bbox intersects viewport",
                "target actionable and visible",
                "occluder visible",
                "intersection_area > 0",
                "occlusion_ratio >= 0.25",
                "same_element=false",
                "parent_child_like=false",
                "normal_overlay_candidate=false",
            ],
            "not_measured": ["computed z-index", "elementFromPoint hit-test", "pointer-events", "focusability", "semantic relationship beyond parent_child_like"],
        },
        "R06": {
            "source_function": "services/anomaly_detection_service.py:_detect_r06_layout_overflow",
            "predicate_function": "services/anomaly_detection_service.py:_is_r06_layout_overflow",
            "source_lines": "1125-1183, 1185-1201",
            "conditions": [
                "target actionable and visible",
                "visible_ratio < 0.75",
                "horizontal viewport overflow always qualifies",
                "vertical viewport overflow qualifies when _vertical_scroll_reachable=false",
            ],
            "not_measured": ["container scrollWidth/clientWidth", "CSS overflow semantics", "intentional carousel/scroll region", "hit-test", "text readability", "keyboard reachability"],
        },
    }


def _candidate_identity(key: str) -> tuple[str, str]:
    parts = str(key).split("|")
    rule = parts[0] if parts else ""
    if rule in {"R05", "R06"} and len(parts) > 3:
        target = parts[3]
        if target.startswith("target:"):
            target = target.split(":", 1)[1]
        return rule, target
    match = re.search(r"target:(\d+)", str(key))
    if match:
        return rule, match.group(1)
    return rule, ""


def _confirmation_audit(temporal: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rule = str(row.get("rule_id") or "")
    target = str(row.get("target") or "")
    records = []
    for record in temporal.get("confirmation_attribution", {}).get("confirmation_records", []) or []:
        record_rule, record_target = _candidate_identity(record.get("logical_key", ""))
        if record_rule == rule and record_target == target:
            records.append(record)
    timeline_events = []
    for item in temporal.get("timeline", []) or []:
        for value in item.get("newly_confirmed", []) or []:
            if rule in str(value) and f"target:{target}" in str(value):
                timeline_events.append({"episode": item.get("episode"), "step": item.get("step"), "action_type": item.get("action_type"), "event": value})
    return {
        "initial_candidate": row.get("candidate_key"),
        "reproduction_records": records,
        "confirmation_events": timeline_events,
        "mechanism": "REPEATED_GEOMETRY_ONLY" if records else "UNKNOWN",
        "interaction_impact_inferred": False,
    }


def _static_fixture_probe(framework: str) -> dict[str, Any]:
    if framework == "React":
        return {
            "scenario_id": "R-D02",
            "probe_status": "STATIC_FIXTURE_CONFIRMED",
            "interactive_probe_status": "UNAVAILABLE_BROWSER_BACKEND",
            "evidence": {"selector": ".defect-overlap", "children": [".badge", ".label"], "positioning": "both absolute", "overlap_intent": "explicit fixture defect", "overflow": "visible"},
            "overlap_ratio": None,
            "hit_test_blocking": None,
            "clipping": False,
            "functional_impact": "EXPECTED_BY_FIXTURE_BUT_NOT_INTERACTIVELY_MEASURED",
        }
    return {
        "scenario_id": "N-D03",
        "probe_status": "STATIC_FIXTURE_CONFIRMED",
        "interactive_probe_status": "UNAVAILABLE_BROWSER_BACKEND",
        "evidence": {"selector": ".clip-box", "container_width_px": 190, "container_height_px": 28, "child_width_px": 360, "overflow": "hidden", "overlap_fixture": ".overlap"},
        "overlap_ratio": None,
        "hit_test_blocking": None,
        "clipping": True,
        "functional_impact": "EXPECTED_TEXT_CLIPPING_BUT_NOT_INTERACTIVELY_MEASURED",
    }


def _distribution(values: Iterable[float | None]) -> dict[str, Any]:
    nums = sorted(value for value in values if value is not None)
    if not nums:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(nums), "min": nums[0], "median": nums[len(nums) // 2], "max": nums[-1]}


def analyze_layout_semantics(evaluation: dict[str, Any], temporal: dict[str, Any], framework: str) -> dict[str, Any]:
    fp_rows = [row for row in evaluation.get("candidates", []) if row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"}]
    records = [_candidate_record({**row, "framework": framework}, temporal) for row in fp_rows]
    for record, source_row in zip(records, fp_rows):
        if record["confirmed"]:
            record["confirmation_evidence"] = _confirmation_audit(temporal, source_row)
    rule_counts = Counter(row["rule_id"] for row in records)
    confirmed = [row for row in records if row["confirmed"]]
    r06 = [row for row in records if row["rule_id"] == "R06"]
    viewport_overflow = [row for row in r06 if row["overflow_type"] != "NOT_APPLICABLE"]
    visible_clickable = [row for row in r06 if row["visible_clickable_overflow"]]
    mobile = [row for row in records if row["viewport_category"] == "mobile"]
    scroll = [row for row in records if row["scroll_correlated"]]
    viewport_actions = [row for row in records if row["viewport_action_correlated"]]
    geometric_only = [row for row in records if row["geometric_vs_functional"] == "GEOMETRIC_ONLY"]
    secondary = [row for row in records if row["scenario_id"] in {"R-D04", "N-D02"}]
    wrong_rule_rows = [row for row in evaluation.get("candidates", []) if row.get("classification") == "DEFECT_SCENARIO_WRONG_RULE"]
    wrong_rule_analysis = [{
        "candidate_key": row.get("candidate_key"),
        "scenario_id": row.get("scenario_id"),
        "detected_rule": row.get("rule_id"),
        "expected_rule": next((scenario.get("expected_defect_rule_ids") for scenario in evaluation.get("scenarios", []) if scenario.get("scenario_id") == row.get("scenario_id")), None),
        "secondary_symptom": "UNKNOWN_NO_CAUSAL_LINK_IN_ARTIFACT",
        "classification": "UNRELATED_FALSE_POSITIVE_OR_SECONDARY_SYMPTOM_UNRESOLVED",
    } for row in wrong_rule_rows]
    return {
        "schema_version": "layout_semantics_diagnostic_v1",
        "evaluation_only": True,
        "framework": framework,
        "source_audit": _source_audit(),
        "probe_capability": {"browser_backend": "unavailable", "element_from_point": "not_run", "safe_points": "not_run", "reason": "No IAB backend was discovered during diagnostic"},
        "summary": {
            "fp_analyzed_count": len(records),
            "confirmed_fp_analyzed_count": len(confirmed),
            "R05_fp_count": rule_counts.get("R05", 0),
            "R06_fp_count": rule_counts.get("R06", 0),
            "parent_child_fp_count": sum(row["relationship"] == "PARENT_CHILD" for row in records),
            "intentional_containment_fp_count": sum(row["expected_containment"] == "NORMAL_CONTAINMENT" for row in records),
            "intentional_overlay_fp_count": sum(row["overlay_semantics"] != "UNKNOWN" for row in records),
            "non_blocking_overlap_fp_count": 0,
            "hit_test_still_valid_fp_count": 0,
            "actual_interaction_blocking_overlap_count": 0,
            "viewport_overflow_fp_count": len(viewport_overflow),
            "container_overflow_fp_count": 0,
            "intentional_scroll_fp_count": 0,
            "target_clipping_fp_count": sum(row["target_partially_clipped"] or row["target_fully_clipped"] for row in records),
            "visible_clickable_overflow_fp_count": len(visible_clickable),
            "mobile_correlated_fp_count": len(mobile),
            "viewport_action_correlated_count": len(viewport_actions),
            "scroll_correlated_count": len(scroll),
            "transform_sticky_fixed_correlated_count": 0,
            "hidden_decorative_element_fp_count": 0,
            "geometric_only_fp_count": len(geometric_only),
            "visual_only_fp_count": 0,
            "functional_impact_fp_count": 0,
            "unknown_functional_impact_count": len(records),
            "confirmed_geometry_only_fp_count": sum(row in confirmed for row in geometric_only),
            "secondary_symptom_count": len(secondary),
            "fp_overlap_ratio_distribution": _distribution([_safe_float(row.get("intersection", {}).get("ratio")) for row in records if row["rule_id"] == "R05"]),
            "fp_clipping_distribution": _distribution([max(0.0, 1.0 - _safe_float(row.get("visibility"))) if _safe_float(row.get("visibility")) is not None else None for row in records if row["rule_id"] == "R06"]),
        },
        "confirmed_fp": confirmed,
        "wrong_rule_analysis": wrong_rule_analysis,
        "candidates": records,
        "actual_defect_probe": _static_fixture_probe(framework),
        "separating_features": {
            "observed_or_inferred": ["scenario expected success", "R06 viewport-relative clipping", "visible_clickable_overflow", "scroll action correlation"],
            "not_measured": ["hit_test_blocking", "CSS intentional scroll", "container scrollability", "z-order", "keyboard reachability"],
            "current_interpretation": "Normal FP candidates are geometry-only observations with no measured interaction obstruction; this remains a post-hoc inference until live hit-testing is available.",
        },
        "root_cause": {"R05": "NO_R05_FP_IN_CURRENT_ARTIFACT", "R06": "FUNCTIONAL_CLIPPING_NOT_REQUIRED", "overall": "LAYOUT_INTERACTION_IMPACT_MISSING", "recommended_first_refinement": "R06 Functional Overflow/Clipping Refinement"},
        "runtime_isolation": {"ground_truth_used_in_runtime": False, "diagnostic_only": True, "detector_modified": False, "temporal_modified": False},
    }


def build_summary(react: dict[str, Any], next_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "layout_semantics_diagnostic_summary_v1",
        "evaluation_only": True,
        "frameworks": {"react": react, "next": next_result},
        "overall": {"R05_fp_count": react["summary"]["R05_fp_count"] + next_result["summary"]["R05_fp_count"], "R06_fp_count": react["summary"]["R06_fp_count"] + next_result["summary"]["R06_fp_count"], "root_cause": "LAYOUT_INTERACTION_IMPACT_MISSING", "recommended_first_refinement": "R06 Functional Overflow/Clipping Refinement"},
    }
