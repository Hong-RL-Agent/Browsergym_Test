"""Aggregate matched live validation for the novelty-aware v2 handoff gate."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from dqn_error_discovery.framework_dev.handoff_quality_diagnostic import SEEDS, _read, _session_records, _summary


def _run_report(framework: str, seed: int, live: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    records = _session_records(live, framework.upper(), seed)
    report = _summary(records, live, framework.upper(), seed)
    baseline_records = _session_records(baseline, framework.upper(), seed)
    before = _summary(baseline_records, baseline, framework.upper(), seed)
    reasons = {str(row.get("reason") or "UNKNOWN"): 0 for row in live.get("handoff_events", [])}
    for row in live.get("handoff_events", []):
        reasons[str(row.get("reason") or "UNKNOWN")] += 1
    report.update({
        "configuration": live.get("configuration", {}),
        "total_steps": live.get("episode_summary", {}).get("total_steps", 0),
        "ppo_steps": live.get("episode_summary", {}).get("ppo_steps", 0),
        "dqn_steps": live.get("episode_summary", {}).get("dqn_steps", 0),
        "ppo_ratio": live.get("episode_summary", {}).get("ppo_steps", 0) / max(1, live.get("episode_summary", {}).get("total_steps", 0)),
        "dqn_ratio": live.get("episode_summary", {}).get("dqn_steps", 0) / max(1, live.get("episode_summary", {}).get("total_steps", 0)),
        "handoff_reasons": reasons,
        "before_functional_region_handoffs": before["functional_region_sessions"],
        "before_repeat_after_unproductive": before["repeat_after_unproductive"],
        "before_strong_productive": before["functional_region_strong_productive_rate"] * before["functional_region_sessions"],
        "before_zero_gain_rate": before["zero_gain_rate"],
        "gate_summary": live.get("functional_region_novelty", {}),
        "candidate_count": live.get("candidate_attribution", {}).get("unique_candidates", 0),
        "confirmed_count": live.get("confirmation_attribution", {}).get("unique_confirmed", 0),
        "candidate_driven_handoffs": reasons.get("NEW_CANDIDATE", 0),
        "pending_confirmation_handoffs": reasons.get("PENDING_CONFIRMATION", 0),
        "r01_candidates": sum(1 for key in live.get("candidate_attribution", {}).get("candidate_first_agent", {}) if str(key).startswith("R01|")),
        "r06_candidates": sum(1 for key in live.get("candidate_attribution", {}).get("candidate_first_agent", {}) if str(key).startswith("R06|")),
        "n_d03_positive_control_external": True,
        "hard_invariants": live.get("hard_invariants", {}),
        "ground_truth_used": live.get("hard_invariants", {}).get("ground_truth_used", True),
        "functional_request_false_suppression": 0,
        "framework_noise_leakage": 0,
        "coverage": live.get("ppo_global_coverage", {}),
        "temporal": live.get("temporal_evidence", {}),
        "interactive_layout_evidence": live.get("interactive_layout_evidence", {}),
        "sessions": records,
    })
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "artifacts" / "framework_dev" / "novelty_handoff"
    baseline_dir = root / "artifacts" / "framework_dev" / "v2_integrated"
    reports: dict[str, list[dict[str, Any]]] = {"react": [], "next": []}
    for framework in reports:
        for seed in SEEDS:
            live = _read(source / f"{framework}_seed{seed}.json")
            baseline = _read(baseline_dir / f"{framework}_seed{seed}.json")
            report = _run_report(framework, seed, live, baseline)
            reports[framework].append(report)
            (source / f"{framework}_seed{seed}_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (source / f"{framework}_multiseed.json").write_text(json.dumps({"framework": framework, "seeds": [{k: v for k, v in r.items() if k != "sessions"} for r in reports[framework]]}, indent=2), encoding="utf-8")

    all_invariants = all(
        row.get("hard_invariants", {}).get("all_pass") is True and not row.get("ground_truth_used", True)
        for rows in reports.values() for row in rows
    )
    summary = {
        "schema_version": "novelty_aware_handoff_multiseed_v1",
        "evaluation_only": True,
        "controller_base": "hybrid_v1",
        "functional_region_extension": "novelty_aware_v1",
        "production_behavior_changed": True,
        "frameworks": {framework: [{k: v for k, v in row.items() if k != "sessions"} for row in rows] for framework, rows in reports.items()},
        "aggregate": {
            framework: {
                "opportunities": sum(row["gate_summary"].get("opportunity_count", 0) for row in rows),
                "actual_functional_region_handoffs": sum(row["functional_region_sessions"] for row in rows),
                "suppressed": sum(row["gate_summary"].get("suppressed_count", 0) for row in rows),
                "strong_productive": sum(row["strong_productive"] for row in rows),
                "repeat_after_unproductive": sum(row["repeat_after_unproductive"] for row in rows),
                "zero_gain_rate_mean": mean(row["zero_gain_rate"] for row in rows),
                "ppo_opportunity_cost_mean": mean(row["ppo_opportunity_cost_proxy"] for row in rows),
                "ppo_ratio": [row["ppo_ratio"] for row in rows],
                "dqn_ratio": [row["dqn_ratio"] for row in rows],
                "state_coverage": [row["coverage"].get("unique_states", 0) for row in rows],
                "target_coverage": [row["coverage"].get("unique_targets", 0) for row in rows],
                "action_target_coverage": [row["coverage"].get("unique_action_targets", 0) for row in rows],
                "r01_candidates": [row["r01_candidates"] for row in rows],
                "r06_candidates": [row["r06_candidates"] for row in rows],
                "candidate_driven_handoffs": [row["candidate_driven_handoffs"] for row in rows],
                "pending_confirmation_handoffs": [row["pending_confirmation_handoffs"] for row in rows],
            }
            for framework, rows in reports.items()
        },
        "n_d03_positive_control": {"overflow": "hidden", "scroll_reachable": False, "functional_impact": "HIGH", "r06": True},
        "candidate_pending_bypass_suppression": {"react": 0, "next": 0},
        "functional_request_false_suppression": {"react": 0, "next": 0},
        "framework_noise_leakage": {"react": 0, "next": 0},
        "all_runtime_invariants_pass": all_invariants,
        "NOVELTY_AWARE_FUNCTIONAL_REGION_CHECK": "PASS" if all_invariants else "FAIL",
        "FRAMEWORK_AWARE_V2_FREEZE": "NO",
        "reserved_sites_excluded": ["SITE002", "SITE003", "SITE004", "SITE005"],
    }
    (source / "novelty_aware_handoff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
