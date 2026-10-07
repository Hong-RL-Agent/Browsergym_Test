"""Counterfactual replay for the v2 novelty-aware functional-region gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dqn_error_discovery.framework_dev.handoff_quality_diagnostic import SEEDS, _session_records, _read


def _replay(framework: str, seed: int, runtime: dict[str, Any]) -> dict[str, Any]:
    records = _session_records(runtime, framework.upper(), seed)
    functional = [row for row in records if row["handoff_reason"] == "FUNCTIONAL_REGION"]
    kept = [row for row in functional if not row["repeat_region"] or row["meaningful_steps"]]
    suppressed = [row for row in functional if row not in kept]
    candidate = [row for row in records if row["handoff_reason"] == "NEW_CANDIDATE"]
    pending = [row for row in records if row["handoff_reason"] == "PENDING_CONFIRMATION"]
    return {
        "framework": framework,
        "seed": seed,
        "baseline_functional_region_handoffs": len(functional),
        "would_keep": len(kept),
        "would_suppress": len(suppressed),
        "suppressed_first_time": sum(not row["repeat_region"] for row in suppressed),
        "suppressed_repeat": sum(row["repeat_region"] for row in suppressed),
        "suppressed_repeat_after_unproductive": sum(row["repeat_after_unproductive"] for row in suppressed),
        "strong_productive_functional_sessions_suppressed": sum(row["strong_productive"] for row in suppressed),
        "candidate_driven_sessions_suppressed": 0,
        "pending_confirmation_sessions_suppressed": 0,
        "candidate_driven_sessions": len(candidate),
        "pending_confirmation_sessions": len(pending),
        "replay_model": "episode-local region proxy from handoff-quality artifacts; live rerun remains authoritative",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "artifacts" / "framework_dev" / "v2_integrated"
    output = root / "artifacts" / "framework_dev" / "novelty_handoff"
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, list[dict[str, Any]]] = {"react": [], "next": []}
    for framework in reports:
        for seed in SEEDS:
            report = _replay(framework, seed, _read(source / f"{framework}_seed{seed}.json"))
            reports[framework].append(report)
    summary = {
        "schema_version": "novelty_aware_handoff_replay_v1",
        "evaluation_only": True,
        "production_behavior_changed": False,
        "controller_base": "hybrid_v1",
        "functional_region_extension": "novelty_aware_v1",
        "frameworks": reports,
        "strong_productive_baseline": {"react": 4, "next": 6},
        "candidate_pending_bypass_suppression": {"react": 0, "next": 0},
        "reserved_sites_excluded": ["SITE002", "SITE003", "SITE004", "SITE005"],
    }
    (output / "novelty_aware_handoff_replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
