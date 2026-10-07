"""Run offline Hybrid candidate lifecycle diagnostics from existing artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.hybrid.candidate_lifecycle_diagnostics import analyze_candidate_lifecycle


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    seed_summaries: dict[str, Any] = {}
    for seed in seeds:
        source = artifact_dir / args.hybrid_template.format(seed=seed)
        source_summary = json.loads(source.read_text(encoding="utf-8"))
        diagnostic = analyze_candidate_lifecycle(source_summary)
        diagnostic["source_artifact"] = str(source)
        diagnostic["diagnostic_conditions"] = {
            "source": "existing frozen Hybrid runtime artifact",
            "training": "none",
            "ppo_update": False,
            "dqn_update": False,
            "seed": seed,
            "episodes": source_summary.get("configuration", {}).get("episodes"),
            "max_steps": source_summary.get("configuration", {}).get("max_steps"),
            "max_local_steps": source_summary.get("configuration", {}).get("max_local_steps"),
            "pending_rehandoff_min_ppo_steps": source_summary.get("configuration", {}).get("pending_rehandoff_min_ppo_steps"),
        }
        out = output_dir / args.seed_output_template.format(seed=seed)
        out.write_text(json.dumps(_jsonable(diagnostic), ensure_ascii=False, indent=2), encoding="utf-8")
        seed_summaries[str(seed)] = diagnostic
    multi = aggregate(seed_summaries)
    multi["seeds"] = seeds
    multi["source_template"] = args.hybrid_template
    multi["HYBRID_BEHAVIOR_CHECK"] = "REVIEW"
    multi["Controller Freeze"] = "NO"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(multi), ensure_ascii=False, indent=2), encoding="utf-8")
    return multi


def aggregate(seed_summaries: dict[str, Any]) -> dict[str, Any]:
    total = Counter()
    verdict_counts: dict[str, Counter] = {}
    for seed, row in seed_summaries.items():
        for key in (
            "pending_sessions",
            "pending_confirmed_sessions",
            "trigger_candidate_confirmed_sessions_exact",
            "trigger_candidate_confirmed_sessions_related",
            "other_candidate_confirmed_sessions",
            "confirmed_return_count",
            "novelty_return_count",
            "trigger_confirmed_but_return_not_confirmed_count",
            "pending_candidate_rotation_count",
            "confirmed_candidate_rehandoff_count",
            "exact_confirmed_candidate_rehandoff_count",
            "related_confirmed_candidate_rehandoff_count",
            "cross_episode_contamination_count",
        ):
            total[key] += int(row.get(key, 0) or 0)
        cooldown = row.get("cooldown_only_rehandoff", {})
        total["cooldown_only_rehandoff_count"] += int(cooldown.get("count", 0) or 0)
        total["cooldown_only_productive_count"] += int(cooldown.get("productive_count", 0) or 0)
        total["cooldown_only_confirmed_count"] += int(cooldown.get("confirmed_count", 0) or 0)
        stale = row.get("stale_pending", {})
        total["stale_pending_candidates"] += int(stale.get("stale_pending_candidates", 0) or 0)
        total["stale_pending_rehandoffs"] += int(stale.get("stale_pending_rehandoffs", 0) or 0)
        override = row.get("override_audit", {})
        for key in (
            "actual_progress_changes",
            "progress_overrides",
            "missed_progress_override",
            "actual_context_changes",
            "context_overrides",
            "missed_context_override",
            "actual_resolvability_changes",
            "resolvability_overrides",
            "missed_resolvability_override",
        ):
            total[key] += int(override.get(key, 0) or 0)
        for key, value in row.get("root_cause_verdicts", {}).items():
            verdict_counts.setdefault(key, Counter())[str(value)] += 1
    root_causes = Counter()
    for row in seed_summaries.values():
        root_causes.update(row.get("root_cause", []) or [])
    return {
        "seed_summaries": {
            seed: {
                "pending_sessions": row.get("pending_sessions"),
                "pending_confirmed_sessions": row.get("pending_confirmed_sessions"),
                "trigger_candidate_confirmed_sessions_related": row.get("trigger_candidate_confirmed_sessions_related"),
                "other_candidate_confirmed_sessions": row.get("other_candidate_confirmed_sessions"),
                "confirmed_return_count": row.get("confirmed_return_count"),
                "novelty_return_count": row.get("novelty_return_count"),
                "cooldown_only_rehandoff": row.get("cooldown_only_rehandoff"),
                "stale_pending": row.get("stale_pending"),
                "pending_candidate_rotation_count": row.get("pending_candidate_rotation_count"),
                "exact_confirmed_candidate_rehandoff_count": row.get("exact_confirmed_candidate_rehandoff_count"),
                "related_confirmed_candidate_rehandoff_count": row.get("related_confirmed_candidate_rehandoff_count"),
                "root_cause": row.get("root_cause"),
            }
            for seed, row in seed_summaries.items()
        },
        "totals": dict(total),
        "cooldown_only_confirmation_rate": _ratio(total["cooldown_only_confirmed_count"], total["cooldown_only_rehandoff_count"]),
        "cooldown_only_productive_rate": _ratio(total["cooldown_only_productive_count"], total["cooldown_only_rehandoff_count"]),
        "root_cause_verdict_counts": {key: dict(value) for key, value in verdict_counts.items()},
        "final_root_cause": [key for key, _ in root_causes.most_common()],
    }


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
        return float(numerator) / denom if denom else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/sanity")
    parser.add_argument("--hybrid-template", default="hybrid_v1_guard_behavior_seed{seed}.json")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default="artifacts/sanity")
    parser.add_argument("--seed-output-template", default="hybrid_candidate_lifecycle_seed{seed}.json")
    parser.add_argument("--output", default="artifacts/sanity/hybrid_candidate_lifecycle_multiseed.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
