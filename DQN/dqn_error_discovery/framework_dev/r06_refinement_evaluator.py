"""Evaluation-only audit for the R06 functional-overflow refinement.

This module is deliberately outside the runtime detector import graph.  It
replays existing DEV_REACT/DEV_NEXT semantic-diagnostic candidates and runs a
small deterministic N-D03 evidence probe; it never supplies scenario truth to
the detector during normal execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.anomaly_detection_service import detect_anomalies


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _n_d03_probe() -> dict[str, Any]:
    observation = {
        "page_state": {
            "url": "http://localhost:9226/",
            "viewport_width": 390,
            "viewport_height": 844,
            "document_height": 844,
        },
        "runtime_signals": {},
        "candidate_elements": [{
            "bid": "dev-next-nd03-content",
            "role": "generic",
            "tag": "span",
            "text": "Clipped text extends beyond the intended layout and cannot be read completely",
            "name": "Clipped text extends beyond the intended layout and cannot be read completely",
            "visible": True,
            "visibility": 1.0,
            "enabled": True,
            "clickable": False,
            "bbox": [0, 0, 360, 28],
            "container_bbox": [0, 0, 190, 28],
            "overflow_x": "hidden",
            "overflow_y": "hidden",
            "scroll_width": 360,
            "client_width": 190,
            "scroll_height": 28,
            "client_height": 28,
            "can_reach_by_scroll_x": False,
            "can_reach_by_scroll_y": False,
            "hit_test_blocked": True,
        }],
    }
    anomalies = detect_anomalies(
        observation,
        observation,
        {"action": {"action_type": "inspect_layout", "candidate_index": 0}},
    )
    r06 = next((item for item in anomalies if item.get("rule_id") == "R06"), None)
    return {
        "success": r06 is not None,
        "detected": r06 is not None,
        "evidence": (r06 or {}).get("evidence", {}),
        "source": "evaluation-only deterministic N-D03 geometry probe",
        "ground_truth_used_in_runtime": False,
    }


def build_r06_refinement_report(root: Path, framework: str) -> dict[str, Any]:
    prefix = framework.lower()
    semantic = _read(root / "artifacts" / "framework_dev" / f"{prefix}_layout_semantics_diagnostic.json")
    evaluation = _read(root / "artifacts" / "framework_dev" / f"{prefix}_scenario_layout_evaluation.json")
    rows = []
    for item in semantic.get("candidates", []):
        impact = str(item.get("functional_impact") or "UNKNOWN").upper()
        retained = impact in {"MEDIUM", "HIGH"}
        rows.append({
            "framework": framework,
            "scenario_id": item.get("scenario_id"),
            "candidate_key": item.get("candidate_key"),
            "rule_id": item.get("rule_id"),
            "viewport": item.get("viewport"),
            "target_id": item.get("target_id"),
            "before_r06": True,
            "after_r06": retained,
            "functional_impact": impact,
            "suppression_reason": None if retained else "geometry_only_or_insufficient_functional_evidence",
            "confirmed_before": bool(item.get("confirmed")),
            "confirmed_after": bool(item.get("confirmed")) and retained,
            "classification": "FUNCTIONAL_IMPACT_CANDIDATE" if retained else "GEOMETRY_ONLY_OBSERVATION",
            "evidence_source": "layout_semantics_diagnostic",
        })
    before = evaluation.get("summary", {})
    fp_rows = [row for row in rows if row.get("scenario_id") and row.get("rule_id") == "R06"]
    after_candidates = sum(row["after_r06"] for row in fp_rows)
    before_fp = int(before.get("layout_fp_candidates", 0) or 0)
    before_confirmed = int(before.get("confirmed_layout_fp", 0) or 0)
    after_fp = sum(row["after_r06"] and row.get("scenario_id") for row in fp_rows)
    after_confirmed = sum(row["confirmed_after"] and row.get("scenario_id") for row in fp_rows)
    probe = _n_d03_probe() if prefix == "next" else {"success": False, "detected": None, "reason": "N-D03 is DEV_NEXT-only"}
    return {
        "schema_version": "r06_refinement_evaluation_v1",
        "evaluation_only": True,
        "framework": framework,
        "mode": "posthoc_artifact_replay_plus_micro_probe",
        "matched_runtime_rerun": False,
        "inputs": {
            "semantic_diagnostic": f"artifacts/framework_dev/{prefix}_layout_semantics_diagnostic.json",
            "scenario_evaluation": f"artifacts/framework_dev/{prefix}_scenario_layout_evaluation.json",
        },
        "candidates": rows,
        "summary": {
            "r06_candidates_before": before_fp,
            "r06_candidates_after_replay": int(after_fp),
            "r06_fp_reduction_replay": round((before_fp - after_fp) / before_fp, 6) if before_fp else None,
            "confirmed_r06_fp_before": before_confirmed,
            "confirmed_r06_fp_after_replay": int(after_confirmed),
            "geometry_only_observations": sum(not row["after_r06"] for row in fp_rows),
            "functional_impact_candidates": after_candidates,
            "unknown_functional_impact": sum(row["functional_impact"] == "UNKNOWN" for row in fp_rows),
            "r05_changed": False,
        },
        "n_d03_probe": probe,
        "runtime_isolation": {
            "ground_truth_used_in_runtime": False,
            "catalog_used_in_runtime": False,
            "production_detector_modified": False,
            "temporal_modified": False,
        },
        "note": "Replay metrics are diagnostic until a matched frozen DEV runtime rerun supplies functional browser evidence.",
    }


def write_reports(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or root / "artifacts" / "framework_dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {framework: build_r06_refinement_report(root, framework) for framework in ("React", "Next")}
    for framework, report in reports.items():
        (output_dir / f"{framework.lower()}_r06_refinement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "schema_version": "r06_refinement_summary_v1",
        "evaluation_only": True,
        "frameworks": reports,
        "classification": "R06_REFINEMENT_DIAGNOSTIC_PENDING_MATCHED_RERUN",
        "runtime_isolation": {"ground_truth_used_in_runtime": False, "site_scope": ["DEV_REACT", "DEV_NEXT"]},
    }
    (output_dir / "r06_refinement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
