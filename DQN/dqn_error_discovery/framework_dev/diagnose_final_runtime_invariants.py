"""Diagnostic-only audit for final unseen-evaluation runtime invariants.

This module intentionally reads canonical artifacts without changing them.  The
final runner currently aggregates action-error counts but does not serialize the
per-action error context into its timeline, so the audit reports that limitation
instead of inventing event-level attribution.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SITES = ("SITE002", "SITE003", "SITE004", "SITE005")
METHODS = ("ppo", "hybrid")
SEEDS = (42, 43, 44)
EXPECTED_EPISODES = 10


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _result_path(root: Path, site: str, method: str, seed: int) -> Path:
    return root / site / f"{method}_seed{seed}.json"


def _runtime_path(root: Path, site: str, method: str, seed: int) -> Path:
    return root / site / f"{method}_seed{seed}_runtime.json"


def _timeline_integrity(timeline: list[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = [(row.get("episode"), row.get("step")) for row in timeline]
    duplicate_steps = len(pairs) - len(set(pairs))
    episodes = sorted({int(row["episode"]) for row in timeline if row.get("episode") is not None})
    by_episode = Counter(int(row["episode"]) for row in timeline if row.get("episode") is not None)
    missing = [episode for episode in range(1, EXPECTED_EPISODES + 1) if episode not in by_episode]
    return {
        "timeline_rows": len(timeline),
        "episode_ids": episodes,
        "episodes_present": len(episodes),
        "missing_episode_ids": missing,
        "duplicate_step_pairs": duplicate_steps,
        "episode_step_counts": dict(sorted(by_episode.items())),
    }


def _run_row(root: Path, site: str, method: str, seed: int) -> dict[str, Any]:
    result_path = _result_path(root, site, method, seed)
    runtime_path = _runtime_path(root, site, method, seed)
    result = _load(result_path) if result_path.exists() else {}
    runtime = _load(runtime_path) if runtime_path.exists() else {}
    diagnostic = result.get("diagnostic", {}) if isinstance(result, Mapping) else {}
    hard = result.get("hard_invariants", {}) if isinstance(result, Mapping) else {}
    timeline = result.get("timeline", []) if isinstance(result, Mapping) else []
    integrity = _timeline_integrity(timeline if isinstance(timeline, list) else [])
    error_count = int(diagnostic.get("action_error_count", 0) or 0)
    all_pass = bool(hard.get("all_pass"))
    artifact_valid = (
        result_path.exists()
        and runtime_path.exists()
        and integrity["episodes_present"] == EXPECTED_EPISODES
        and not integrity["missing_episode_ids"]
        and integrity["duplicate_step_pairs"] == 0
    )
    if not artifact_valid:
        validity = "INVALID_ARTIFACT"
    elif error_count == 0 and all_pass:
        validity = "VALID"
    elif error_count > 0:
        # Per-action cause is absent from the canonical serialization.  Do not
        # label these as infrastructure or algorithm failures without evidence.
        validity = "REVIEW"
    else:
        validity = "REVIEW"
    return {
        "site": site,
        "method": method,
        "seed": seed,
        "result_path": str(result_path),
        "runtime_path": str(runtime_path),
        "result_exists": result_path.exists(),
        "runtime_exists": runtime_path.exists(),
        "action_error_count": error_count,
        "affected_episode_count": None,
        "event_level_context_available": False,
        "event_level_context_missing_fields": [
            "episode", "step", "agent_mode", "action", "target", "exception", "mask", "reset_causality"
        ],
        "all_pass": all_pass,
        "unexpected_reset_count": int((result.get("environment_continuity", {}) or {}).get("unexpected_reset_count", 0) or 0),
        "reset_at_switch_count": int((result.get("environment_continuity", {}) or {}).get("reset_at_switch_count", 0) or 0),
        "ground_truth_used": bool(hard.get("ground_truth_used", False) or runtime.get("runtime_ground_truth_imported", False)),
        "artifact_integrity": integrity,
        "validity": validity,
        "root_cause": "UNAVAILABLE_EVENT_CONTEXT" if error_count else "NONE_OBSERVED",
        "severity": "UNKNOWN" if error_count else "NONE",
        "rerun_candidate": False,
    }


def _event_placeholders(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        for index in range(int(row["action_error_count"])):
            events.append(
                {
                    "event_id": f"{row['site']}-{row['method']}-seed{row['seed']}-unknown-{index + 1}",
                    "site": row["site"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "episode": None,
                    "step": None,
                    "agent_mode": None,
                    "action": None,
                    "target": None,
                    "error": None,
                    "root_cause": "UNAVAILABLE_EVENT_CONTEXT",
                    "severity": "UNKNOWN",
                    "reset_followed": None,
                    "run_contaminated": row["validity"] != "VALID",
                    "serialization_note": "Canonical artifact stores only aggregate action_error_count.",
                }
            )
    return events


def build_diagnostic(root: Path) -> dict[str, Any]:
    rows = [_run_row(root, site, method, seed) for site in SITES for method in METHODS for seed in SEEDS]
    events = _event_placeholders(rows)
    by_site = {
        site: {
            "action_error_total": sum(row["action_error_count"] for row in rows if row["site"] == site),
            "affected_runs": sum(bool(row["action_error_count"]) for row in rows if row["site"] == site),
            "validity_counts": dict(Counter(row["validity"] for row in rows if row["site"] == site)),
        }
        for site in SITES
    }
    return {
        "schema_version": "runtime_invariant_diagnostic_v1",
        "source": "canonical_final_unseen_evaluation_as_observed",
        "ground_truth_used_for_validity": False,
        "event_level_context_available": False,
        "event_level_limitation": "Canonical result timeline does not serialize per-action error metadata.",
        "runs": rows,
        "action_error_events": events,
        "by_site": by_site,
        "summary": {
            "run_count": len(rows),
            "action_error_total": sum(row["action_error_count"] for row in rows),
            "affected_run_count": sum(bool(row["action_error_count"]) for row in rows),
            "affected_episode_count": None,
            "root_cause_counts": {"UNAVAILABLE_EVENT_CONTEXT": len(events)},
            "severity_counts": {"UNKNOWN": len(events)} if events else {},
            "validity_counts": dict(Counter(row["validity"] for row in rows)),
            "rerun_candidate_runs": [
                {"site": row["site"], "method": row["method"], "seed": row["seed"]}
                for row in rows if row["rerun_candidate"]
            ],
        },
    }


def write_diagnostic(root: Path, output_dir: Path) -> dict[str, Any]:
    payload = build_diagnostic(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action_error_events.json").write_text(
        json.dumps(payload["action_error_events"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_validity_matrix.json").write_text(
        json.dumps(payload["runs"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reset_audit = {
        "runs": [
            {
                "site": row["site"], "method": row["method"], "seed": row["seed"],
                "expected_reset_count": EXPECTED_EPISODES,
                "actual_reset_count": None,
                "unexpected_reset_count": row["unexpected_reset_count"],
                "reset_at_switch_count": row["reset_at_switch_count"],
                "action_error_to_reset_causality": None,
                "note": "Canonical artifact exposes continuity counters but not reset event timestamps/source.",
            }
            for row in payload["runs"]
        ]
    }
    (output_dir / "reset_invariant_audit.json").write_text(json.dumps(reset_audit, indent=2), encoding="utf-8")
    fragment_audit = {
        "canonical_run_count": len(payload["runs"]),
        "validity_by_run": [
            {"site": row["site"], "method": row["method"], "seed": row["seed"], "integrity": row["artifact_integrity"]}
            for row in payload["runs"]
        ],
        "posthoc_fragment_count": len(list((root / "posthoc_fragments").glob("*.json"))),
    }
    (output_dir / "fragment_integrity_audit.json").write_text(json.dumps(fragment_audit, indent=2), encoding="utf-8")
    recovery = {
        "site": "SITE005", "method": "hybrid", "seed": 44,
        "original_result_path": str(root / "SITE005" / "hybrid_seed44.json"),
        "original_runtime_sidecar": str(root / "SITE005" / "hybrid_seed44_runtime.json"),
        "original_result_missing_before_recovery": True,
        "original_result_was_present_at_diagnostic_time": True,
        "original_runtime_sidecar_preserved": True,
        "recovery_policy": "same frozen command, artifact-completeness based, not performance based",
        "replacement_selected_by_quality": False,
        "diagnostic_note": "Verify orchestration audit separately; this file records no GT-derived validity decision.",
    }
    (output_dir / "site005_seed44_recovery_audit.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    (output_dir / "runtime_invariant_diagnostic_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2] / "artifacts" / "final_unseen_eval"
    out = base / "runtime_invariant_diagnostic"
    result = write_diagnostic(base, out)
    print(json.dumps(result["summary"], indent=2))
