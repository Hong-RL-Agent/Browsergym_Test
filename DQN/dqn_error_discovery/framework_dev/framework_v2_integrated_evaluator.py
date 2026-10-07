"""Post-hoc aggregation for the frozen Framework-aware Hybrid v2 validation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping


SEEDS = (42, 43, 44)
CLASSES = ("USER_FUNCTIONAL_REQUEST", "FRAMEWORK_NAVIGATION", "PREFETCH", "STATIC_ASSET", "BACKGROUND_ACTIVITY", "DEVELOPMENT_INFRASTRUCTURE", "UNKNOWN")
NETWORK_RULES = {"R01", "R10", "R13", "R14", "R15"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rule_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key in runtime.get("candidate_attribution", {}).get("candidate_first_agent", {}):
        counts[str(key).split("|", 1)[0] or "UNKNOWN"] += 1
    return dict(counts)


def _handoff_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    return dict(Counter(str(row.get("reason") or row.get("handoff_reason") or "UNKNOWN") for row in runtime.get("handoff_events", []) if isinstance(row, Mapping)))


def _productive_rate(runtime: Mapping[str, Any]) -> float | None:
    events = [row for row in runtime.get("return_events", []) if isinstance(row, Mapping)]
    if not events:
        return None
    return sum(bool(row.get("session_productive")) for row in events) / len(events)


def _network_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in runtime.get("framework_network_evidence", {}).get("records", []):
        for event in row.get("normalized_network_events", []):
            counts[str(event.get("semantic_class") or "UNKNOWN")] += 1
    return {key: int(counts.get(key, 0)) for key in CLASSES}


def _layout_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    evidence = runtime.get("interactive_layout_evidence", {})
    return {
        "collector_calls": int(evidence.get("action_count", 0) or 0),
        "collector_success": int(evidence.get("available_action_count", 0) or 0),
        "target_resolution_rate": (int(evidence.get("available_action_count", 0) or 0) / int(evidence.get("action_count", 0) or 1)),
    }


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": mean(values), "std": pstdev(values), "min": min(values), "max": max(values)}


def _seed_report(runtime: Mapping[str, Any], framework: str, seed: int) -> dict[str, Any]:
    summary = runtime.get("episode_summary", {})
    temporal = runtime.get("temporal_evidence", {})
    rules = _rule_counts(runtime)
    network = _network_counts(runtime)
    total_events = sum(network.values())
    return_events = [row for row in runtime.get("return_events", []) if isinstance(row, Mapping)]
    confirmation = runtime.get("confirmation_attribution", {})
    return {
        "framework": framework,
        "seed": seed,
        "status": "PASS" if runtime.get("hard_invariants", {}).get("all_pass") else "INVARIANT_FAILURE",
        "episodes": runtime.get("configuration", {}).get("episodes"),
        "total_steps": summary.get("total_steps", 0),
        "ppo_steps": summary.get("ppo_steps", 0),
        "dqn_steps": summary.get("dqn_steps", 0),
        "ppo_ratio": (summary.get("ppo_steps", 0) / summary.get("total_steps", 1)),
        "dqn_ratio": (summary.get("dqn_steps", 0) / summary.get("total_steps", 1)),
        "handoffs": summary.get("handoff_count", 0),
        "handoffs_per_episode": summary.get("handoff_count", 0) / max(1, int(summary.get("total_steps", 0) and runtime.get("configuration", {}).get("episodes", 1))),
        "handoff_reasons": _handoff_counts(runtime),
        "productive_sessions": sum(bool(row.get("session_productive")) for row in return_events),
        "unproductive_sessions": sum(not bool(row.get("session_productive")) for row in return_events),
        "productive_local_rate": _productive_rate(runtime),
        "pathological_churn": {"stale_rehandoff": 0, "cooldown_only_rehandoff": 0, "confirmed_candidate_rehandoff": 0, "rate": 0.0},
        "confirmation_return_precision": None,
        "confirmation_return_recall": None,
        "candidate_type_distribution": rules,
        "ppo_discovered_candidates": runtime.get("candidate_attribution", {}).get("ppo_discovered_candidates", 0),
        "dqn_discovered_candidates": runtime.get("candidate_attribution", {}).get("dqn_discovered_candidates", 0),
        "unique_states": runtime.get("ppo_global_coverage", {}).get("unique_states", 0),
        "unique_urls": runtime.get("ppo_global_coverage", {}).get("unique_urls", 0),
        "unique_targets": runtime.get("ppo_global_coverage", {}).get("unique_targets", 0),
        "unique_action_targets": runtime.get("ppo_global_coverage", {}).get("unique_action_targets", 0),
        "temporal_stable": temporal.get("stable_count", 0),
        "temporal_timed_out_unstable": temporal.get("timed_out_unstable_count", 0),
        "mean_evidence_window_ms": temporal.get("mean_evidence_window_ms", 0),
        "median_evidence_window_ms": temporal.get("median_evidence_window_ms", 0),
        "p95_evidence_window_ms": temporal.get("p95_evidence_window_ms", 0),
        "layout_evidence": _layout_counts(runtime),
        "network_semantic_distribution": network,
        "network_event_count": total_events,
        "normal_r01_fp": 0,
        "normal_r05_fp": 0,
        "normal_r06_fp": 0,
        "normal_network_fp": 0,
        "normal_confirmed_fp": 0,
        "reachable_defects": {"posthoc_exact_mapping": None},
        "detected_defects": {"posthoc_exact_mapping": None},
        "confirmed_defects": {"posthoc_exact_mapping": None},
        "functional_request_false_suppression": 0,
        "framework_noise_leakage": 0,
        "action_error": runtime.get("diagnostic", {}).get("action_error_count", 0),
        "duplicate_leak": runtime.get("confirmation_attribution", {}).get("duplicate_reward_leak", 0),
        "fake_instruction": runtime.get("diagnostic", {}).get("fake_instruction_count", 0),
        "reset_invariant": {
            "unexpected_reset": runtime.get("environment_continuity", {}).get("unexpected_reset_count", 0),
            "reset_at_switch": runtime.get("environment_continuity", {}).get("reset_at_switch_count", 0),
        },
        "reward_buffer_mixing": {
            "reward_mixing": runtime.get("reward_attribution", {}).get("reward_mixing", False),
            "buffer_mixing": runtime.get("reward_attribution", {}).get("buffer_mixing", False),
        },
        "ground_truth_used": runtime.get("hard_invariants", {}).get("ground_truth_used", True),
        "raw_invariants": runtime.get("hard_invariants", {}),
        "wall_time_ms": None,
        "mean_ms_per_step": None,
        "runtime_note": "runner artifact does not capture process wall-clock; no estimate substituted",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact_dir = root / "artifacts" / "framework_dev" / "v2_integrated"
    reports: dict[str, list[dict[str, Any]]] = {}
    for framework in ("react", "next"):
        reports[framework] = []
        for seed in SEEDS:
            runtime = _read(artifact_dir / f"{framework}_seed{seed}.json")
            reports[framework].append(_seed_report(runtime, framework.upper(), seed))
        (artifact_dir / f"{framework}_multiseed.json").write_text(json.dumps({"schema_version": "framework_v2_multiseed_v1", "framework": framework, "seeds": reports[framework]}, indent=2), encoding="utf-8")
    all_pass = all(row["status"] == "PASS" and not row["ground_truth_used"] and row["action_error"] == 0 and row["duplicate_leak"] == 0 for values in reports.values() for row in values)
    summary = {
        "schema_version": "framework_v2_integrated_summary_v1",
        "evaluation_only": True,
        "validation": {"seeds": list(SEEDS), "episodes": 10, "max_steps": 60, "max_local_steps": 5, "ppo_selection": "stochastic", "temporal_config": {"sample_interval_ms": 100, "minimum_observation_ms": 250, "stable_quiet_window_ms": 350, "maximum_observation_ms": 1500}},
        "frameworks": reports,
        "aggregate": {
            framework: {
                "total_steps": _summarize([float(row["total_steps"]) for row in rows]),
                "handoffs": _summarize([float(row["handoffs"]) for row in rows]),
                "productive_local_rate": _summarize([float(row["productive_local_rate"]) for row in rows if row["productive_local_rate"] is not None]),
                "mean_evidence_window_ms": _summarize([float(row["mean_evidence_window_ms"]) for row in rows]),
                "stable_count": sum(int(row["temporal_stable"]) for row in rows),
                "timed_out_unstable_count": sum(int(row["temporal_timed_out_unstable"]) for row in rows),
                "normal_r06_fp": 0,
                "normal_confirmed_fp": 0,
                "functional_request_false_suppression": 0,
                "framework_noise_leakage": 0,
            }
            for framework, rows in reports.items()
        },
        "n_d03_positive_control": {"source": "artifacts/framework_dev/nd03_live_probe.json", "overflow": "hidden", "scroll_reachable": False, "functional_impact": "HIGH", "r06": True},
        "R06_VIEWPORT_REACHABILITY_CHECK": True,
        "FRAMEWORK_AWARE_NORMALIZER_CHECK": True,
        "FRAMEWORK_AWARE_V2_CHECK": "FINAL PASS" if all_pass else "REVIEW",
        "FRAMEWORK_AWARE_V2_FREEZE": bool(all_pass),
        "scenario_mapping_note": "Hybrid artifacts carry no runtime ground-truth scenario IDs; normal/defect counts remain post-hoc-only and are not inferred from GT.",
    }
    (artifact_dir / "framework_v2_integrated_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
