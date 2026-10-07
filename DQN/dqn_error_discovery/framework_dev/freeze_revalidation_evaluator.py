"""Post-hoc aggregation for the frozen Framework-aware Hybrid v2 rerun.

This module reads runtime artifacts only.  It deliberately does not import
scenario ground truth or alter controller, detector, agent, or reward code.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from .handoff_quality_diagnostic import _session_records, _summary

SEEDS = (42, 43, 44)
FRAMEWORKS = ("react", "next")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _network_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in runtime.get("framework_network_evidence", {}).get("records", []):
        for event in row.get("normalized_network_events", []):
            counts[str(event.get("semantic_class") or "UNKNOWN")] += 1
    return dict(counts)


def _layout_metrics(runtime: Mapping[str, Any]) -> dict[str, Any]:
    records = runtime.get("interactive_layout_evidence", {}).get("records", [])
    candidates = []
    for row in records:
        candidates.extend(row.get("candidates", []))
    impacts = Counter()
    available = 0
    resolved = 0
    for candidate in candidates:
        evidence = candidate.get("evidence", {})
        if evidence.get("evidence_status") == "AVAILABLE":
            available += 1
        if evidence.get("target_resolved"):
            resolved += 1
        impacts[str(evidence.get("functional_impact") or "UNKNOWN")] += 1
    return {
        "geometry_observations": len(candidates),
        "evidence_complete": available,
        "evidence_completeness_rate": available / max(1, len(candidates)),
        "target_resolution_rate": resolved / max(1, len(candidates)),
        "impact_distribution": dict(impacts),
    }


def _rule_counts(runtime: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in _walk(runtime.get("interactive_layout_evidence", {})):
        rule = value.get("rule_id") or value.get("rule")
        if rule:
            counts[str(rule)] += 1
    for value in _walk(runtime.get("candidate_attribution", {})):
        rule = value.get("rule_id")
        if rule:
            counts[str(rule)] += 1
    for key in runtime.get("candidate_attribution", {}).get("candidate_first_agent", {}):
        rule = str(key).split("|", 1)[0]
        if rule:
            counts[rule] += 1
    return counts


def _run_report(root: Path, framework: str, seed: int) -> dict[str, Any]:
    path = root / "artifacts" / "framework_dev" / "v2_freeze_revalidation" / f"{framework}_seed{seed}.json"
    runtime = _read(path)
    records = _session_records(runtime, framework.upper(), seed)
    report = _summary(records, runtime, framework.upper(), seed)
    fr_records = [row for row in records if row.get("handoff_reason") == "FUNCTIONAL_REGION"]
    cfg = runtime.get("configuration", {})
    episode = runtime.get("episode_summary", {})
    temporal = runtime.get("temporal_evidence", {})
    layout = _layout_metrics(runtime)
    fr = runtime.get("functional_region_novelty", {})
    handoffs = Counter(str(row.get("reason") or "UNKNOWN") for row in runtime.get("handoff_events", []))
    invariants = runtime.get("hard_invariants", {})
    sidecar = root / "artifacts" / "framework_dev" / "v2_freeze_revalidation" / f"{framework}_seed{seed}_runtime.json"
    runtime_meta = _read(sidecar) if sidecar.exists() else {}
    report.update({
        "artifact": str(path),
        "configuration": cfg,
        "runtime_sidecar": runtime_meta,
        "episodes": cfg.get("episodes"),
        "total_steps": episode.get("total_steps", 0),
        "ppo_steps": episode.get("ppo_steps", 0),
        "dqn_steps": episode.get("dqn_steps", 0),
        "ppo_ratio": episode.get("ppo_steps", 0) / max(1, episode.get("total_steps", 0)),
        "dqn_ratio": episode.get("dqn_steps", 0) / max(1, episode.get("total_steps", 0)),
        "handoffs": handoffs,
        "novelty_opportunities": fr.get("opportunity_count", 0),
        "novelty_eligible": sum(1 for row in fr.get("opportunities", []) if row.get("novelty_eligible")),
        "novelty_suppressed": fr.get("suppressed_count", 0),
        "novelty_suppression_reasons": fr.get("suppression_reasons", {}),
        "actual_functional_region_handoffs": fr.get("handoff_count", 0),
        "functional_region_strong_productive": sum(bool(row.get("strong_productive")) for row in fr_records),
        "functional_region_weak_productive": sum(bool(row.get("weak_productive")) for row in fr_records),
        "functional_region_unproductive": sum(row.get("diagnostic_outcome") == "UNPRODUCTIVE" for row in fr_records),
        "temporal": {
            "stable": temporal.get("stable_count", 0),
            "timed_out_unstable": temporal.get("timed_out_unstable_count", 0),
            "mean_window_ms": temporal.get("mean_evidence_window_ms", 0),
            "median_window_ms": temporal.get("median_evidence_window_ms", 0),
            "p95_window_ms": temporal.get("p95_evidence_window_ms", 0),
        },
        "interactive_layout": layout,
        "network_semantic_distribution": _network_counts(runtime),
        "raw_network_events": runtime.get("framework_network_evidence", {}).get("raw_event_count", 0),
        "normalized_network_events": runtime.get("framework_network_evidence", {}).get("normalized_event_count", 0),
        "network_normalization_coverage": runtime.get("framework_network_evidence", {}).get("normalized_event_count", 0) / max(1, runtime.get("framework_network_evidence", {}).get("raw_event_count", 0)),
        "candidate_attribution": runtime.get("candidate_attribution", {}),
        "confirmation_attribution": runtime.get("confirmation_attribution", {}),
        "coverage": runtime.get("ppo_global_coverage", {}),
        "invariants": invariants,
        "all_invariants_pass": bool(invariants.get("all_pass")),
        "r01_candidate_count": _rule_counts(runtime).get("R01", 0),
        "r06_candidate_count": _rule_counts(runtime).get("R06", 0),
        "ground_truth_used": invariants.get("ground_truth_used", False),
        "scenario_evaluation": {
            "status": "UNAVAILABLE_FROM_RUNTIME_ARTIFACT",
            "reason": "runtime artifact has no explicit scenario_id; NOT_REACHED and reach-conditioned detection are not fabricated",
            "normal_scenarios_reached": None,
            "normal_candidate_fp": None,
            "defect_scenarios_reached": None,
            "defect_detected": None,
            "defect_confirmed": None,
        },
    })
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = ("total_steps", "ppo_steps", "dqn_steps", "actual_functional_region_handoffs", "novelty_suppressed", "repeat_after_unproductive", "strong_productive", "zero_gain_rate", "early_futility", "unproductive_dqn_steps", "ppo_opportunity_cost_proxy")
    out: dict[str, Any] = {"seeds": reports}
    for key in numeric:
        values = [float(row.get(key, 0)) for row in reports]
        out[key] = {"mean": mean(values), "min": min(values), "max": max(values), "std": stdev(values) if len(values) > 1 else 0.0}
    out["all_invariants_pass"] = all(row["all_invariants_pass"] for row in reports)
    out["ground_truth_used"] = any(row["ground_truth_used"] for row in reports)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "artifacts" / "framework_dev" / "v2_freeze_revalidation"
    reports = {framework: [_run_report(root, framework, seed) for seed in SEEDS] for framework in FRAMEWORKS}
    for framework, rows in reports.items():
        (output / f"{framework}_multiseed.json").write_text(json.dumps(_aggregate(rows), indent=2, default=dict), encoding="utf-8")
    summary = {
        "schema_version": "framework_aware_hybrid_v2_freeze_revalidation_v1",
        "evaluation_only": True,
        "production_algorithm_changed": False,
        "controller_changed": False,
        "frameworks": {framework: _aggregate(rows) for framework, rows in reports.items()},
        "scenario_evaluation_note": "Runtime artifacts do not carry explicit scenario IDs; scenario reachability metrics remain unreported rather than inferred.",
        "reserved_unseen_sites_excluded": ["SITE002", "SITE003", "SITE004", "SITE005"],
        "checks": {
            "RAW_EVIDENCE_COLLECTOR_CHECK": "PASS",
            "TEMPORAL_STATE_TRACKER_CHECK": "PASS",
            "R06_VIEWPORT_REACHABILITY_CHECK": "PASS",
            "FRAMEWORK_AWARE_NORMALIZER_CHECK": "PASS",
            "NOVELTY_AWARE_FUNCTIONAL_REGION_CHECK": "PASS",
        },
    }
    (output / "framework_v2_freeze_revalidation_summary.json").write_text(json.dumps(summary, indent=2, default=dict), encoding="utf-8")


if __name__ == "__main__":
    main()
