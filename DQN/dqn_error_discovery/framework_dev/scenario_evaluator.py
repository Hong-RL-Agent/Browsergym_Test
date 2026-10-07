"""Scenario-level, post-hoc evaluation for development-site artifacts.

The evaluator consumes frozen-run artifacts and scenario catalogs.  It never
gets imported by the environment, detector, reward, or controller paths.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LAYOUT_RULES = {"R05", "R06"}
CONFIDENCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNMAPPED": 0}


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _expected_rules(value: Any) -> list[str]:
    if value in (None, "", "NO_DEFECT"):
        return []
    return sorted({part.strip() for part in str(value).split("/") if part.strip().startswith("R")})


def _route_path(url: str | None) -> str:
    if not url:
        return ""
    return re.sub(r"^https?://[^/]+", "", str(url)) or "/"


def _target_from_key(key: str) -> str:
    parts = str(key).split("|")
    if len(parts) >= 5:
        # Layout discovery keys use position 3; click candidates use 4.
        return parts[3] if _rule_from_key(key) in LAYOUT_RULES else parts[4]
    match = re.search(r"target:(\d+)", str(key))
    return match.group(1) if match else ""


def _rule_from_key(key: str) -> str:
    return str(key).split("|", 1)[0]


def _target_from_event(event: dict[str, Any]) -> str:
    target = event.get("target")
    if target not in (None, ""):
        return str(target)
    return _target_from_key(event.get("logical_key", event.get("signature", "")))


def load_scenario_catalog(path: str | Path) -> dict[str, Any]:
    raw = _read_json(path)
    scenarios = []
    for source in raw.get("scenarios", []):
        trigger = str(source.get("trigger_action", ""))
        marker = re.search(r"data-testid=([\w-]+)", trigger)
        expected_rules = _expected_rules(source.get("expected_detector_result"))
        scenarios.append({
            "scenario_id": source.get("scenario_id"),
            "framework": source.get("framework", raw.get("framework", "")),
            "scenario_type": source.get("scenario_type", source.get("normal_or_defect")),
            "route": source.get("route", ""),
            "trigger_action": trigger,
            "target_signature": source.get("target_signature") or (marker.group(1) if marker else ""),
            "expected_intermediate_behavior": source.get("expected_intermediate_behavior", ""),
            "expected_final_behavior": source.get("expected_final_behavior", ""),
            "expected_defect_rule_ids": expected_rules,
            "expected_layout_behavior": source.get(
                "expected_layout_behavior",
                "layout" if any(rule in LAYOUT_RULES for rule in expected_rules) else "none",
            ),
            "evaluation_only": True,
        })
    return {
        "site_id": raw.get("site_id", ""),
        "framework": raw.get("framework", ""),
        "scenarios": scenarios,
    }


def validate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    scenarios = catalog.get("scenarios", [])
    ids = [item.get("scenario_id") for item in scenarios]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate scenario IDs")
    for item in scenarios:
        if item.get("scenario_type") not in {"NORMAL", "DEFECT"}:
            errors.append(f"invalid scenario_type: {item.get('scenario_id')}")
        if not item.get("scenario_id"):
            errors.append("missing scenario_id")
        if not item.get("trigger_action"):
            errors.append(f"missing trigger_action: {item.get('scenario_id')}")
        if not item.get("expected_final_behavior"):
            errors.append(f"missing expected_final_behavior: {item.get('scenario_id')}")
    return {"valid": not errors, "errors": errors, "scenario_count": len(scenarios)}


def _scenario_evidence_map(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for scenario in evidence.get("scenario_results", []):
        for event in scenario.get("candidate_events", []):
            key = event.get("logical_key", "")
            if key:
                result[key].append({"scenario": scenario, "event": event})
    return result


def _scenario_evidence_target_map(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for scenario in evidence.get("scenario_results", []):
        for event in scenario.get("candidate_events", []):
            target = _target_from_event(event)
            if target:
                result[target].append({"scenario": scenario, "event": event})
    return result


def _confirmation_identities(temporal: dict[str, Any]) -> set[tuple[str, str]]:
    records = temporal.get("confirmation_attribution", {}).get("confirmation_records", [])
    identities = set()
    for record in records:
        key = record.get("logical_key", "")
        rule = record.get("rule_id") or _rule_from_key(key)
        target = _target_from_key(key)
        if rule in LAYOUT_RULES and target:
            identities.add((rule, target))
    return identities


def _unique_layout_candidates(diagnostic: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for window in diagnostic.get("windows", []):
        status_by_key = window.get("layout_candidate_classification", {})
        source_by_key = {item.get("signature"): item for item in window.get("detector_layout_candidates", [])}
        for key in window.get("persistent_layout_candidate_keys", []):
            source = source_by_key.get(key, {})
            row = candidates.setdefault(key, {
                "candidate_key": key,
                "rule_id": _rule_from_key(key),
                "target": _target_from_key(key),
                "route": _route_path(window.get("route_before")),
                "persistent_or_transient": status_by_key.get(key, "PERSISTENT_LAYOUT_CANDIDATE"),
                "first_seen_ms": window.get("stable_decision_ms"),
                "stable_decision_ms": window.get("stable_decision_ms"),
                "detector_invocation_ms": window.get("stable_decision_ms"),
                "evidence_window": {
                    "episode": window.get("episode"),
                    "step": window.get("step"),
                    "action_id": window.get("action_id"),
                    "action_type": window.get("action_type"),
                    "target_id": window.get("target_id"),
                    "stability_decision_timeline": window.get("stability_decision_timeline", []),
                },
                "source": source,
            })
        # A candidate can be observed before stable; preserve earliest timing.
        for sample in window.get("per_sample_layout_candidates", []) or []:
            for item in sample.get("layout_candidates", []) or []:
                key = item.get("signature")
                if key in candidates and sample.get("relative_ms") is not None:
                    candidates[key]["first_seen_ms"] = min(candidates[key]["first_seen_ms"], sample["relative_ms"])
    return candidates


def _candidate_match(key: str, route: str, evidence_map: dict[str, list[dict[str, Any]]], target_map: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], str, str]:
    exact = evidence_map.get(key, [])
    if len(exact) == 1:
        return exact, "HIGH", "explicit candidate logical key"
    if len(exact) > 1:
        return [], "UNMAPPED", "ambiguous explicit candidate logical key"
    target = _target_from_key(key)
    fallback = [item for item in target_map.get(target, []) if _route_path(item["scenario"].get("route", "")) in {"", route}]
    if len(fallback) == 1:
        return fallback, "MEDIUM", "unique target signature fallback"
    if len(fallback) > 1:
        return [], "UNMAPPED", "ambiguous target signature"
    return [], "UNMAPPED", "no scenario evidence match"


def _classify(match: dict[str, Any] | None, rule_id: str, exposure: bool, expected_rules: Iterable[str] | None = None) -> str:
    if not match:
        return "UNMAPPED"
    scenario = match["scenario"]
    expected = set(expected_rules or _expected_rules(scenario.get("expected_detector_result")) or scenario.get("rule_ids", []))
    if scenario.get("normal_or_defect") == "NORMAL":
        return "TRAJECTORY_EXPOSURE_NORMAL_FP" if exposure else "NORMAL_SCENARIO_FP"
    if rule_id in expected:
        return "TRAJECTORY_EXPOSURE_TRUE_DEFECT" if exposure else "DEFECT_SCENARIO_TRUE_POSITIVE"
    return "DEFECT_SCENARIO_WRONG_RULE"


def _status_for_scenario(scenario: dict[str, Any], evidence: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    raw = next((item for item in evidence.get("scenario_results", []) if item.get("scenario_id") == scenario.get("scenario_id")), None)
    if not raw or not raw.get("reached"):
        return "NOT_REACHED"
    if not candidates:
        return "REACHED_NO_CANDIDATE"
    if any(item.get("confirmed") for item in candidates):
        return "CONFIRMED"
    return "REACHED_CANDIDATE"


def _raw_scenario_status(scenario_id: str, evidence: dict[str, Any], has_candidate: bool, confirmed: bool) -> str:
    raw = next((item for item in evidence.get("scenario_results", []) if item.get("scenario_id") == scenario_id), None)
    if not raw or not raw.get("reached"):
        return "NOT_REACHED"
    if confirmed:
        return "CONFIRMED"
    return "REACHED_CANDIDATE" if has_candidate else "REACHED_NO_CANDIDATE"


def _run_summary(candidates: Iterable[dict[str, Any]], scenarios: list[dict[str, Any]], evidence: dict[str, Any], *, primary_only: bool = False) -> dict[str, Any]:
    rows = list(candidates)
    if primary_only:
        rows = [row for row in rows if CONFIDENCE_RANK.get(row.get("mapping_confidence", "UNMAPPED"), 0) >= CONFIDENCE_RANK["MEDIUM"]]
    by_scenario = defaultdict(list)
    for row in rows:
        for match in row.get("scenario_matches", []):
            by_scenario[match["scenario"].get("scenario_id")].append(row)
    normal = [item for item in scenarios if item.get("scenario_type") == "NORMAL"]
    defects = [item for item in scenarios if item.get("scenario_type") == "DEFECT"]
    raw_by_id = {item.get("scenario_id"): item for item in evidence.get("scenario_results", [])}
    normal_reached = [item for item in normal if raw_by_id.get(item.get("scenario_id"), {}).get("reached")]
    layout_defects = [item for item in defects if set(item.get("expected_defect_rule_ids", [])) & LAYOUT_RULES]
    defect_reached = [item for item in layout_defects if raw_by_id.get(item.get("scenario_id"), {}).get("reached")]
    normal_with = [item for item in normal_reached if any(row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"} for row in by_scenario.get(item.get("scenario_id"), []))]
    normal_confirmed = [item for item in normal_with if any(row.get("confirmed") for row in by_scenario.get(item.get("scenario_id"), []))]
    detected = [item for item in defect_reached if any(row.get("classification") in {"DEFECT_SCENARIO_TRUE_POSITIVE", "TRAJECTORY_EXPOSURE_TRUE_DEFECT"} for row in by_scenario.get(item.get("scenario_id"), []))]
    confirmed = [item for item in detected if any(row.get("confirmed") for row in by_scenario.get(item.get("scenario_id"), []))]
    def ratio(num: int, den: int) -> float:
        return round(num / den, 6) if den else None
    return {
        "layout_candidates": len(rows),
        "mapped": sum(row.get("mapping_confidence") in {"HIGH", "MEDIUM"} for row in rows),
        "mapping_rate": ratio(sum(row.get("mapping_confidence") in {"HIGH", "MEDIUM"} for row in rows), len(rows)),
        "normal_layout_scenarios_reached": len(normal_reached),
        "normal_layout_scenarios_total": len(normal),
        "normal_layout_scenarios_with_candidate": len(normal_with),
        "normal_layout_scenarios_with_confirmed": len(normal_confirmed),
        "layout_fp_candidates": sum(row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"} for row in rows),
        "confirmed_layout_fp": sum(row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"} and row.get("confirmed") for row in rows),
        "layout_defect_scenarios_reached": len(defect_reached),
        "layout_defect_scenarios_total": len(layout_defects),
        "layout_tp_detected": len(detected),
        "layout_tp_confirmed": len(confirmed),
        "layout_detection_given_reach": ratio(len(detected), len(defect_reached)),
        "layout_fp_rate_given_reach": ratio(len(normal_with), len(normal_reached)),
        "layout_confirmed_fp_rate_given_reach": ratio(len(normal_confirmed), len(normal_reached)),
        "trajectory_exposure_tp": sum(row.get("classification") == "TRAJECTORY_EXPOSURE_TRUE_DEFECT" for row in rows),
        "trajectory_exposure_fp": sum(row.get("classification") == "TRAJECTORY_EXPOSURE_NORMAL_FP" for row in rows),
        "unmapped": sum(row.get("mapping_confidence") == "UNMAPPED" for row in rows),
        "low_confidence": sum(row.get("mapping_confidence") == "LOW" for row in rows),
    }


def evaluate_layout_artifacts(*, catalog_path: str | Path, diagnostic_path: str | Path, temporal_path: str | Path, baseline_path: str | Path, scenario_evidence_path: str | Path) -> dict[str, Any]:
    catalog = load_scenario_catalog(catalog_path)
    validation = validate_catalog(catalog)
    if not validation["valid"]:
        raise ValueError("invalid scenario catalog: " + "; ".join(validation["errors"]))
    diagnostic = _read_json(diagnostic_path)
    temporal = _read_json(temporal_path)
    baseline = _read_json(baseline_path)
    evidence = _read_json(scenario_evidence_path)
    evidence_map = _scenario_evidence_map(evidence)
    target_map = _scenario_evidence_target_map(evidence)
    baseline_keys = set(baseline.get("candidate_attribution", {}).get("candidate_first_agent", {}).keys())
    baseline_layout_keys = {key for key in baseline_keys if _rule_from_key(key) in LAYOUT_RULES}
    candidates = _unique_layout_candidates(diagnostic)
    confirmation_ids = _confirmation_identities(temporal)
    rows = []
    for key, row in sorted(candidates.items()):
        matches, confidence, mapping_reason = _candidate_match(key, row["route"], evidence_map, target_map)
        exposure = key not in baseline_layout_keys
        scenario_matches = []
        for item in matches:
            scenario = item["scenario"]
            expected_rules = next((entry.get("expected_defect_rule_ids", []) for entry in catalog["scenarios"] if entry.get("scenario_id") == scenario.get("scenario_id")), [])
            scenario_matches.append({
                "scenario": scenario,
                "event": item["event"],
                "classification": _classify(item, row["rule_id"], exposure, expected_rules),
            })
        primary_match = scenario_matches[0] if len(scenario_matches) == 1 else None
        row.update({
            "scenario_id": primary_match["scenario"].get("scenario_id") if primary_match else None,
            "scenario_type": primary_match["scenario"].get("normal_or_defect") if primary_match else None,
            "mapping_confidence": confidence,
            "mapping_reason": mapping_reason,
            "scenario_matches": scenario_matches,
            "before_or_new_after_temporal": "PERSISTED_FROM_V1" if key in baseline_layout_keys else "NEW_AFTER_TEMPORAL",
            "trajectory_exposure": exposure,
            "confirmed": (row["rule_id"], row["target"]) in confirmation_ids,
            "classification": primary_match["classification"] if primary_match else "UNMAPPED",
            "scenario_reachability": _raw_scenario_status(primary_match["scenario"].get("scenario_id"), evidence, True, (row["rule_id"], row["target"]) in confirmation_ids) if primary_match else "UNMAPPED",
        })
        rows.append(row)
    scenario_rows = []
    for scenario in catalog["scenarios"]:
        linked = [row for row in rows if any(match["scenario"].get("scenario_id") == scenario.get("scenario_id") for match in row.get("scenario_matches", []))]
        scenario_rows.append({
            **scenario,
            "reachability": _status_for_scenario(scenario, evidence, linked),
            "layout_candidate_keys": [row["candidate_key"] for row in linked],
            "confirmed": any(row.get("confirmed") for row in linked),
        })
    baseline_rows = []
    for key in sorted(baseline_layout_keys):
        matches, confidence, mapping_reason = _candidate_match(key, "", evidence_map, target_map)
        scenario_matches = []
        for item in matches:
            expected_rules = next((entry.get("expected_defect_rule_ids", []) for entry in catalog["scenarios"] if entry.get("scenario_id") == item["scenario"].get("scenario_id")), [])
            scenario_matches.append({
                "scenario": item["scenario"],
                "event": item["event"],
                "classification": _classify(item, _rule_from_key(key), False, expected_rules),
            })
        primary_match = scenario_matches[0] if len(scenario_matches) == 1 else None
        baseline_rows.append({
            "candidate_key": key,
            "rule_id": _rule_from_key(key),
            "target": _target_from_key(key),
            "route": "",
            "scenario_id": primary_match["scenario"].get("scenario_id") if primary_match else None,
            "scenario_type": primary_match["scenario"].get("normal_or_defect") if primary_match else None,
            "mapping_confidence": confidence,
            "mapping_reason": mapping_reason,
            "scenario_matches": scenario_matches,
            "before_or_new_after_temporal": "BASELINE_V1",
            "trajectory_exposure": False,
            "persistent_or_transient": "UNKNOWN_BASELINE",
            "confirmed": (_rule_from_key(key), _target_from_key(key)) in _confirmation_identities(baseline),
            "classification": primary_match["classification"] if primary_match else "UNMAPPED",
        })
    summary = _run_summary(rows, catalog["scenarios"], evidence)
    primary_summary = _run_summary(rows, catalog["scenarios"], evidence, primary_only=True)
    baseline_summary = _run_summary(baseline_rows, catalog["scenarios"], evidence)
    baseline_primary_summary = _run_summary(baseline_rows, catalog["scenarios"], evidence, primary_only=True)
    return {
        "schema_version": "scenario_layout_evaluation_v1",
        "evaluation_only": True,
        "site_id": catalog["site_id"],
        "framework": catalog["framework"],
        "catalog": {"path": str(catalog_path), "validation": validation, "scenario_count": len(catalog["scenarios"])},
        "inputs": {"diagnostic": str(diagnostic_path), "temporal": str(temporal_path), "baseline": str(baseline_path), "scenario_evidence": str(scenario_evidence_path)},
        "candidate_matching": {"explicit_scenario_marker": "scenario_evidence candidate logical key", "fallbacks": ["route", "trigger action", "target signature", "logical key", "action correlation"], "primary_confidence": ["HIGH", "MEDIUM"]},
        "summary": summary,
        "primary_summary": primary_summary,
        "baseline": {"summary": baseline_summary, "primary_summary": baseline_primary_summary, "candidates": baseline_rows},
        "candidates": rows,
        "scenarios": scenario_rows,
        "representative": {
            "normal_fp": next((row for row in rows if row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"}), None),
            "true_positive": next((row for row in rows if row.get("classification") in {"DEFECT_SCENARIO_TRUE_POSITIVE", "TRAJECTORY_EXPOSURE_TRUE_DEFECT"} and row.get("scenario_reachability") != "NOT_REACHED"), None),
        },
        "runtime_isolation": {"ground_truth_used_in_runtime": False, "catalog_used_in_runtime": False, "production_detector_modified": False},
    }


def reclassify_root_cause(result: dict[str, Any]) -> str:
    rows = result.get("candidates", [])
    mapped = [row for row in rows if row.get("mapping_confidence") in {"HIGH", "MEDIUM"}]
    if not mapped or len(rows) - len(mapped) >= len(mapped):
        return "EVALUATOR_MAPPING_INSUFFICIENT"
    fp = sum(row.get("classification") in {"NORMAL_SCENARIO_FP", "TRAJECTORY_EXPOSURE_NORMAL_FP"} for row in mapped)
    # A matching defect rule on a NOT_REACHED scenario is retained in the
    # candidate table, but cannot establish a reachable true positive.
    tp = sum(row.get("classification") in {"DEFECT_SCENARIO_TRUE_POSITIVE", "TRAJECTORY_EXPOSURE_TRUE_DEFECT"} and row.get("scenario_reachability") != "NOT_REACHED" for row in mapped)
    if fp and tp:
        return "MIXED_TRUE_DEFECT_AND_FP"
    if fp:
        return "PERSISTENT_LAYOUT_FALSE_POSITIVE"
    if tp:
        return "TRAJECTORY_EXPOSURE_TRUE_DEFECT"
    return "EVALUATOR_MAPPING_INSUFFICIENT"


def compare_evaluations(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def counts(data: dict[str, Any]) -> dict[str, int]:
        rows = data.get("candidates", [])
        return {
            "layout_candidates": len(rows),
            "mapped": sum(row.get("mapping_confidence") in {"HIGH", "MEDIUM"} for row in rows),
            "normal_fp": sum(row.get("classification") == "NORMAL_SCENARIO_FP" for row in rows),
            "confirmed_normal_fp": sum(row.get("classification") == "NORMAL_SCENARIO_FP" and row.get("confirmed") for row in rows),
            "true_defect": sum(row.get("classification") == "DEFECT_SCENARIO_TRUE_POSITIVE" for row in rows),
            "confirmed_true_defect": sum(row.get("classification") == "DEFECT_SCENARIO_TRUE_POSITIVE" and row.get("confirmed") for row in rows),
        }
    before_keys = {row["candidate_key"] for row in before.get("candidates", [])}
    after_keys = {row["candidate_key"] for row in after.get("candidates", [])}
    return {"before": counts(before), "after": counts(after), "new_after_temporal": sorted(after_keys - before_keys), "removed_by_temporal": sorted(before_keys - after_keys)}
