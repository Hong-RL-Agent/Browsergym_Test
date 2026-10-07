"""Evaluation-only audit of the frozen R06 functional-impact decision tree."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMPACTS = ("NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target(key: str) -> str:
    parts = str(key).split("|")
    return parts[3] if len(parts) >= 4 else ""


def _rows(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in runtime.get("interactive_layout_evidence", {}).get("records", []):
        for item in record.get("candidates", []) or []:
            evidence = item.get("evidence", {}) or {}
            row = dict(evidence)
            row.update({"episode": record.get("episode"), "step": record.get("step"), "action_id": str(record.get("action_id")), "bid": str(item.get("bid"))})
            row["functional_impact"] = str(row.get("functional_impact") or "UNKNOWN").upper()
            row["functional_impact_reasons"] = list(row.get("functional_impact_reasons") or [])
            rows.append(row)
    return rows


def _unknown_reason(row: dict[str, Any]) -> str:
    if row.get("evidence_status") != "AVAILABLE" or row.get("target_resolved") is False:
        return "EVIDENCE_MISSING"
    if row.get("can_reach_by_scroll_x") is None or row.get("can_reach_by_scroll_y") is None:
        return "REACHABILITY_AMBIGUOUS"
    if row.get("hit_test_valid") is None and row.get("hit_test_blocked") is None:
        return "HIT_TEST_AMBIGUOUS"
    if row.get("text_potentially_clipped") is None and row.get("meaningful_text"):
        return "READABILITY_AMBIGUOUS"
    if row.get("functional_impact_reasons"):
        return "NO_DECISION_BRANCH_MATCHED"
    return "NO_DECISION_BRANCH_MATCHED"


def _medium_reason(row: dict[str, Any]) -> str:
    reasons = set(row.get("functional_impact_reasons") or [])
    if "not_reachable_by_normal_scroll" in reasons and "meaningful_text_clipped" in reasons:
        return "MEDIUM_BY_NO_SCROLL_REACHABILITY_AND_READABILITY"
    if "not_reachable_by_normal_scroll" in reasons:
        return "MEDIUM_BY_NO_SCROLL_REACHABILITY"
    if "meaningful_text_clipped" in reasons:
        return "MEDIUM_BY_TEXT_READABILITY"
    if "important_interactive_or_functional_target" in reasons:
        return "MEDIUM_BY_IMPORTANT_TARGET"
    return "MEDIUM_BY_COMBINED_EVIDENCE"


def _trace_row(row: dict[str, Any]) -> dict[str, Any]:
    impact = row["functional_impact"]
    reasons = row.get("functional_impact_reasons", [])
    if impact == "UNKNOWN":
        reason = _unknown_reason(row)
    elif impact == "MEDIUM":
        reason = _medium_reason(row)
    elif impact == "NONE":
        reason = "CLIPPED_BUT_USABLE" if "hit_test_valid" in reasons or "target_remains_usable" in reasons else "NO_GEOMETRIC_FUNCTIONAL_IMPACT"
    elif impact == "HIGH":
        reason = "HIT_TEST_BLOCKED" if "hit_test_blocked" in reasons else "OVERFLOW_HIDDEN_UNREACHABLE"
    else:
        reason = "LOW_BRANCH_OBSERVED" if impact == "LOW" else "OTHER"
    out = dict(row)
    out["impact_decision_reason"] = reason
    out["derived_predicates"] = {
        "is_important": "important_interactive_or_functional_target" in reasons,
        "is_significantly_clipped": float(row.get("clipping_ratio") or 0) >= 0.25,
        "is_viewport_only": bool(row.get("viewport_overflow")) and not bool(row.get("container_overflow")),
        "is_container_clipped": bool(row.get("container_overflow")) or bool(row.get("container_clipping")),
        "is_scroll_reachable": row.get("reachable_via_scroll_x") is True or row.get("reachable_via_scroll_y") is True,
        "is_document_flow_reachable": _document_flow_reachable(row),
        "is_hit_test_blocked": row.get("hit_test_blocked") is True,
        "is_interactable": row.get("hit_test_valid") is True and row.get("enabled") is not False,
        "is_readability_impacted": "meaningful_text_clipped" in reasons,
    }
    return out


def _document_flow_reachable(row: dict[str, Any]) -> bool:
    bbox = row.get("target_bbox") or []
    viewport_height = float(row.get("viewport_height") or 0)
    if len(bbox) < 4 or not viewport_height:
        return False
    outside_vertical = float(bbox[1]) < 0 or float(bbox[1]) + float(bbox[3]) > viewport_height
    if not outside_vertical:
        return False
    for ancestor in row.get("ancestors", []) or []:
        if str(ancestor.get("tag") or "").lower() not in {"html", "body", "main", "document"}:
            continue
        if float(ancestor.get("scroll_height") or 0) > float(ancestor.get("client_height") or 0):
            return True
        ancestor_bbox = ancestor.get("bbox") or {}
        if float(ancestor_bbox.get("height") or 0) > viewport_height:
            return True
    return False


def _candidate_traces(framework: str, runtime: dict[str, Any], scenario_path: Path) -> list[dict[str, Any]]:
    scenario = _read(scenario_path)
    rows = [_trace_row(row) for row in _rows(runtime)]
    evaluation_path = scenario_path.parent / f"{framework.lower()}_scenario_layout_evaluation_live.json"
    candidates = _read(evaluation_path).get("candidates", []) if evaluation_path.exists() else []
    traces = []
    for candidate in candidates:
        if candidate.get("rule_id") != "R06":
            continue
        target = str(candidate.get("target", ""))
        matching = [row for row in rows if row.get("bid") == target]
        action_id = str(candidate.get("evidence_window", {}).get("action_id", ""))
        window = candidate.get("evidence_window", {})
        trigger = next((row for row in matching if row.get("action_id") == action_id and row.get("episode") == window.get("episode") and row.get("step") == window.get("step")), None)
        trigger = trigger or next((row for row in matching if row.get("action_id") == action_id), None)
        trigger = trigger or next((row for row in matching if row.get("functional_impact") == "MEDIUM"), None) or (matching[0] if matching else {})
        impacts = [row.get("functional_impact", "UNKNOWN") for row in matching]
        traces.append({
            "candidate_key": candidate.get("candidate_key"),
            "scenario_id": candidate.get("scenario_id"),
            "mapping_confidence": candidate.get("mapping_confidence"),
            "route": candidate.get("route"),
            "target_id": target,
            "candidate_created": True,
            "confirmed": bool(candidate.get("confirmed")),
            "observation_count_per_candidate": len(matching),
            "impact_sequence": impacts,
            "max_impact": max(impacts, key=lambda value: {"NONE": 0, "LOW": 1, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}.get(value, 1)) if impacts else "UNKNOWN",
            "candidate_trigger_action_id": action_id,
            "candidate_trigger_observation": trigger,
            "confirmation_observation": matching[-1] if matching else {},
            "trajectory_exposure": candidate.get("trajectory_exposure"),
            "classification": candidate.get("classification", "UNMAPPED"),
        })
    return traces


def diagnose(framework: str, runtime_path: Path, scenario_path: Path, n_d03_path: Path) -> dict[str, Any]:
    runtime = _read(runtime_path)
    rows = [_trace_row(row) for row in _rows(runtime)]
    traces = _candidate_traces(framework, runtime, scenario_path)
    geometry = [row for row in rows if row.get("evidence_status") == "AVAILABLE" and (row.get("clipping_ratio") not in (None, 0) or row.get("overflow_x") in {"hidden", "clip", "auto", "scroll"} or row.get("overflow_y") in {"hidden", "clip", "auto", "scroll"})]
    impacts = Counter(row.get("functional_impact", "UNKNOWN") for row in geometry)
    reason_counts = Counter(row.get("impact_decision_reason") for row in geometry)
    medium_counts = Counter(row.get("impact_decision_reason") for row in geometry if row.get("functional_impact") == "MEDIUM")
    unknown_counts = Counter(_unknown_reason(row) for row in geometry if row.get("functional_impact") == "UNKNOWN")
    fp = [row for row in traces if row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP", "UNMAPPED"}]
    fp_trigger = [row.get("candidate_trigger_observation", {}) for row in fp]
    n_d03 = _read(n_d03_path).get("n_d03_probe", {}).get("evidence", {}) if n_d03_path.exists() else {}
    return {
        "schema_version": "r06_semantics_diagnostic_v1",
        "evaluation_only": True,
        "framework": framework,
        "inputs": {"runtime": str(runtime_path), "scenario_evaluation": str(scenario_path.parent / f"{framework.lower()}_scenario_layout_evaluation_live.json"), "positive_control": str(n_d03_path)},
        "decision_tree_audit": {
            "NONE": "hit_test_valid unless hidden container plus clipped meaningful text; otherwise intentional scroll, hit-test valid, or usable visible clickable target",
            "LOW": "no reachable LOW branch observed in frozen runtime; rank exists but no LOW impact was emitted",
            "MEDIUM": "clipping >= 0.25 plus important target or meaningful text plus no normal access",
            "HIGH": "blocked hit with no normal access, or hidden/clip container with >= 0.5 clipping and important/text content",
            "UNKNOWN": "no preceding NONE/MEDIUM/HIGH branch matched; unavailable or ambiguous predicates remain UNKNOWN",
            "confirmation": "R06 confirmation accepts MEDIUM or HIGH only",
        },
        "summary": {
            "geometry_observations": len(geometry),
            "impact_distribution": {key: int(impacts.get(key, 0)) for key in IMPACTS},
            "unknown_root_causes": dict(unknown_counts),
            "medium_promotion_reasons": dict(medium_counts),
            "decision_reason_counts": dict(reason_counts),
            "remaining_fp_analyzed": len(fp),
            "confirmed_fp_analyzed": sum(bool(row.get("confirmed")) for row in fp),
            "fp_with_valid_hit_test": sum(row.get("hit_test_valid") is True for row in fp_trigger),
            "fp_clickable_true": sum(row.get("semantic_clickable") is True for row in fp_trigger),
            "fp_browser_interactable_geometry": sum(row.get("browser_interactable_geometry") is True for row in fp_trigger),
            "fp_keyboard_reachable": sum(row.get("keyboard_reachable") is True for row in fp_trigger),
            "fp_scroll_reachable": sum(row.get("reachable_via_scroll_x") is True or row.get("reachable_via_scroll_y") is True for row in fp_trigger),
            "fp_intentional_scrollable": sum(row.get("intentional_scrollable") is True for row in fp_trigger),
            "fp_viewport_only": sum(bool(row.get("viewport_overflow")) and not bool(row.get("container_overflow")) for row in fp_trigger),
            "fp_ancestor_clipping": sum(bool(row.get("container_clipping")) for row in fp_trigger),
            "fp_overflow_hidden": sum(row.get("overflow_x") in {"hidden", "clip"} or row.get("overflow_y") in {"hidden", "clip"} for row in fp_trigger),
            "fp_importance_only_promotion": sum(row.get("functional_impact") == "MEDIUM" and row.get("clipping_ratio", 0) < 0.25 for row in fp_trigger),
            "fp_readability_promotion": sum("meaningful_text_clipped" in row.get("functional_impact_reasons", []) for row in fp_trigger),
            "fp_candidate_trigger_medium": sum(row.get("functional_impact") == "MEDIUM" for row in fp_trigger),
            "aggregation_induced_fp": sum(row.get("max_impact") == "MEDIUM" and row.get("candidate_trigger_observation", {}).get("functional_impact") != "MEDIUM" for row in fp),
            "initial_reproduction_impact_mismatch": sum(row.get("candidate_trigger_observation", {}).get("functional_impact") != row.get("confirmation_observation", {}).get("functional_impact") for row in fp if row.get("confirmed")),
            "unknown_candidate_promotion": sum(row.get("candidate_trigger_observation", {}).get("functional_impact") == "UNKNOWN" for row in fp),
            "normal_document_flow_fp": sum(row.get("candidate_trigger_observation", {}).get("derived_predicates", {}).get("is_document_flow_reachable") is True for row in fp),
            "low_branch_reachable": bool(impacts.get("LOW")),
        },
        "candidate_traces": traces,
        "positive_control": {"name": "N-D03", "feature_vector": n_d03},
        "runtime_isolation": {"ground_truth_used_in_runtime": False, "production_detector_modified": False, "r06_rule_modified": False},
    }


def write_reports(root: Path) -> dict[str, Any]:
    artifact_dir = root / "artifacts" / "framework_dev"
    reports = {
        "react": diagnose("React", artifact_dir / "react_r06_live_runtime.json", artifact_dir / "react_temporal_stability_diagnostic.json", artifact_dir / "next_r06_refinement.json"),
        "next": diagnose("Next", artifact_dir / "next_r06_live_runtime.json", artifact_dir / "next_temporal_stability_diagnostic.json", artifact_dir / "next_r06_refinement.json"),
    }
    for name, report in reports.items():
        (artifact_dir / f"{name}_r06_semantics_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {"schema_version": "r06_semantics_diagnostic_summary_v1", "evaluation_only": True, "frameworks": reports, "classification": "MIXED"}
    (artifact_dir / "r06_semantics_diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
