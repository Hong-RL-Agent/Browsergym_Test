"""Post-hoc metrics for live interactive layout evidence artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_runtime_artifact(path: Path, framework: str, before_fp: int, before_confirmed: int) -> dict[str, Any]:
    runtime = _read(path)
    records = runtime.get("interactive_layout_evidence", {}).get("records", [])
    evidence_rows = []
    for record in records:
        for item in record.get("candidates", []) or []:
            evidence = item.get("evidence", {}) or {}
            if not isinstance(evidence, dict):
                continue
            evidence_rows.append({
                "episode": record.get("episode"),
                "step": record.get("step"),
                "action_id": record.get("action_id"),
                "action_type": record.get("action_type"),
                "stability_state": record.get("stability_state"),
                "bid": item.get("bid"),
                "evidence_status": evidence.get("evidence_status", "UNAVAILABLE"),
                "functional_impact": str(evidence.get("functional_impact") or "UNKNOWN").upper(),
                "hit_test_valid": evidence.get("hit_test_valid"),
                "hit_test_blocked": evidence.get("hit_test_blocked"),
                "overflow_x": evidence.get("overflow_x"),
                "overflow_y": evidence.get("overflow_y"),
                "target_bbox": evidence.get("target_bbox"),
                "visible_bbox": evidence.get("visible_bbox"),
                "visible_ratio": evidence.get("visible_ratio"),
                "scroll_width": evidence.get("scroll_width"),
                "client_width": evidence.get("client_width"),
                "scroll_height": evidence.get("scroll_height"),
                "client_height": evidence.get("client_height"),
                "intentional_scrollable": evidence.get("intentional_scrollable"),
                "reachable_via_scroll_x": evidence.get("reachable_via_scroll_x"),
                "reachable_via_scroll_y": evidence.get("reachable_via_scroll_y"),
                "keyboard_reachable": evidence.get("keyboard_reachable"),
                "clipping_ratio": evidence.get("clipping_ratio"),
                "mutation_guard": evidence.get("mutation_guard", {}),
            })
    geometry = [row for row in evidence_rows if row.get("evidence_status") == "AVAILABLE" and (
        row.get("clipping_ratio", 0) not in (None, 0) or row.get("intentional_scrollable") is True or
        row.get("overflow_x") in {"hidden", "clip", "auto", "scroll"} or row.get("overflow_y") in {"hidden", "clip", "auto", "scroll"}
    )]
    complete_fields = {
        "target_bbox", "visible_bbox", "visible_ratio", "clipping_ratio", "overflow_x", "overflow_y",
        "scroll_width", "client_width", "scroll_height", "client_height", "reachable_via_scroll_x",
        "reachable_via_scroll_y", "hit_test_valid", "hit_test_blocked", "keyboard_reachable",
    }
    complete = [row for row in geometry if row.get("evidence_status") == "AVAILABLE" and all(row.get(key) is not None for key in complete_fields)]
    impacts = Counter(row.get("functional_impact", "UNKNOWN") for row in geometry)
    candidate_keys = runtime.get("candidate_attribution", {}).get("candidate_first_agent", {})
    r06_keys = sorted(key for key in candidate_keys if str(key).startswith("R06|"))
    confirmed_keys = {
        str(row.get("logical_key")) for row in runtime.get("confirmation_attribution", {}).get("confirmation_records", [])
        if row.get("rule_id") == "R06"
    }
    guard_rows = [row for row in evidence_rows if row.get("mutation_guard")]
    guard_pass = all(all(bool(value) for value in row.get("mutation_guard", {}).values()) for row in guard_rows) if guard_rows else None
    return {
        "schema_version": "interactive_layout_evidence_v1",
        "evaluation_only": True,
        "framework": framework,
        "runtime_artifact": str(path),
        "summary": {
            "geometry_observations": len(geometry),
            "evidence_complete": len(complete),
            "geometry_evidence_observations": len(geometry),
            "functional_impact_distribution": {key: int(impacts.get(key, 0)) for key in ("NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN")},
            "evidence_completeness_rate": round(len(complete) / len(geometry), 6) if geometry else None,
            "impact_distribution": {key: int(impacts.get(key, 0)) for key in ("NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN")},
            "hit_test_available_count": sum(row.get("hit_test_valid") is not None or row.get("hit_test_blocked") is not None for row in geometry),
            "hit_test_valid_count": sum(row.get("hit_test_valid") is True for row in geometry),
            "hit_test_blocked_count": sum(row.get("hit_test_blocked") is True for row in geometry),
            "overflow_css_available_count": sum(row.get("overflow_x") is not None or row.get("overflow_y") is not None for row in geometry),
            "scroll_reachability_available_count": sum(row.get("reachable_via_scroll_x") is not None or row.get("reachable_via_scroll_y") is not None for row in geometry),
            "keyboard_evidence_available_count": sum("keyboard_reachable" in row for row in geometry),
            "r06_candidates_after": len(r06_keys),
            "confirmed_r06_after": len(confirmed_keys),
            "r06_fp_before": before_fp,
            "r06_confirmed_fp_before": before_confirmed,
            "r06_fp_reduction": round((before_fp - len(r06_keys)) / before_fp, 6) if before_fp else None,
            "confirmed_r06_fp_reduction": round((before_confirmed - len(confirmed_keys)) / before_confirmed, 6) if before_confirmed else None,
            "mutation_guard_pass": guard_pass,
            "collector_calls": len(records),
            "collector_success": sum(any((item.get("evidence") or {}).get("evidence_status") == "AVAILABLE" for item in record.get("candidates", [])) for record in records),
            "collector_unavailable": sum(not any((item.get("evidence") or {}).get("evidence_status") == "AVAILABLE" for item in record.get("candidates", [])) for record in records),
        },
        "evidence_rows": evidence_rows,
        "r06_candidate_keys": r06_keys,
        "confirmed_r06_keys": sorted(confirmed_keys),
        "runtime_isolation": {"ground_truth_used_in_runtime": False, "catalog_used_in_runtime": False},
    }
