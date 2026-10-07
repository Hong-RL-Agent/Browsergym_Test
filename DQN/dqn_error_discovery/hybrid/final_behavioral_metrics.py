"""Final Hybrid v1 behavioral validation metrics.

This module is intentionally measurement-only. It reads runtime artifacts and
classifies residual handoff churn without changing controller behavior.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Mapping

from dqn_error_discovery.hybrid.candidate_lifecycle_diagnostics import analyze_candidate_lifecycle, key_match
from dqn_error_discovery.runners.run_hybrid_behavioral_validation import (
    aggregate_validation,
    coverage_retention,
    summarize_hybrid_artifact,
    summarize_ppo_only_artifact,
)


MEANINGFUL_ELIGIBILITY = {
    "PROGRESS_CHANGED",
    "CONTEXT_CHANGED",
    "TARGET_BECAME_RESOLVABLE",
    "REPRODUCTION_REQUIREMENT_CHANGED",
}


def build_final_behavioral_summary(
    hybrid_by_seed: Mapping[int, Mapping[str, Any]],
    ppo_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    controller_version: str = "hybrid_controller_v1_stable_pending",
) -> dict[str, Any]:
    base = aggregate_validation(hybrid_by_seed, ppo_by_seed)
    final_rows: dict[str, Any] = {}
    classifications: dict[str, str] = {}
    for seed, summary in hybrid_by_seed.items():
        lifecycle = analyze_candidate_lifecycle(summary)
        hybrid_row = summarize_hybrid_artifact(summary)
        ppo_row = summarize_ppo_only_artifact(ppo_by_seed.get(seed, {}))
        coverage = coverage_retention(hybrid_row, ppo_row)
        churn = final_churn_breakdown(summary, lifecycle)
        eligibility = eligibility_signal_table(hybrid_row)
        return_efficiency = return_reason_efficiency(hybrid_row)
        classification = classify_final_seed(hybrid_row, coverage, churn)
        final_rows[str(seed)] = {
            "hybrid": hybrid_row,
            "ppo_only": ppo_row,
            "coverage_retention": coverage,
            "lifecycle": _compact_lifecycle(lifecycle),
            "final_churn": churn,
            "eligibility_signals": eligibility,
            "return_efficiency": return_efficiency,
            "classification": classification,
        }
        classifications[str(seed)] = classification
    overall = classify_final_overall(final_rows)
    result = {
        "controller_version": controller_version,
        "validation_only": True,
        "base_behavioral_aggregate": base,
        "seeds": sorted(int(seed) for seed in hybrid_by_seed),
        "seed_summaries": final_rows,
        "aggregate": aggregate_final_rows(final_rows),
        "seed_classification": classifications,
        "overall_classification": overall,
        "HYBRID_BEHAVIOR_CHECK": "FINAL PASS" if overall == "HYBRID_V1_STABLE" else "REVIEW",
        "HYBRID_CONTROLLER_CHECK": "FINAL PASS" if overall == "HYBRID_V1_STABLE" else "REVIEW",
        "HYBRID_CONTROLLER_FREEZE": "YES" if overall == "HYBRID_V1_STABLE" else "NO",
    }
    return result


def final_churn_breakdown(summary: Mapping[str, Any], lifecycle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    lifecycle = lifecycle or analyze_candidate_lifecycle(summary)
    max_steps = int(summary.get("configuration", {}).get("max_steps", 0) or 0)
    session_audits = [row for row in lifecycle.get("session_audits", []) or [] if isinstance(row, Mapping)]
    handoffs = [row for row in summary.get("handoff_events", []) or [] if isinstance(row, Mapping)]
    sessions = [row for row in session_audits if row.get("trigger_candidate_key")]

    seen_exact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    within_episode_same = 0
    cross_episode_same = 0
    exact_rehandoff = 0
    related_rehandoff = 0
    meaningful_rehandoff = 0
    stale_rehandoff_sessions: set[int] = set()
    cooldown_only_sessions: set[int] = set()
    stale_rotation_sessions: set[int] = set()
    meaningful_rotation_sessions: set[int] = set()
    candidate_rotations: list[dict[str, Any]] = []

    pending_sequence_by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sessions:
        key = str(row.get("trigger_candidate_key") or "")
        episode = _episode(row, max_steps)
        sid = int(row.get("local_session_id", 0) or 0)
        meaningful = _meaningful_eligibility(row)
        previous_same = seen_exact.get(key, [])
        if previous_same:
            same_ep_previous = [prior for prior in previous_same if _episode(prior, max_steps) == episode]
            other_ep_previous = [prior for prior in previous_same if _episode(prior, max_steps) != episode]
            if same_ep_previous:
                within_episode_same += 1
                exact_rehandoff += 1
                prior = same_ep_previous[-1]
                if meaningful:
                    meaningful_rehandoff += 1
                elif _same_stale_material_state(row, prior):
                    stale_rehandoff_sessions.add(sid)
                    cooldown_only_sessions.add(sid)
            elif other_ep_previous:
                cross_episode_same += 1
        for prior_key, prior_rows in seen_exact.items():
            if prior_key == key:
                continue
            if any(_episode(prior, max_steps) == episode and key_match(prior_key, key).related for prior in prior_rows):
                related_rehandoff += 1
                break
        pending_sequence_by_episode[episode].append(row)
        seen_exact[key].append(row)

    for episode, rows in pending_sequence_by_episode.items():
        previous_index: dict[str, int] = {}
        for index, row in enumerate(rows):
            key = str(row.get("trigger_candidate_key") or "")
            if key in previous_index and previous_index[key] != index - 1:
                sid = int(row.get("local_session_id", 0) or 0)
                event = {"episode": episode, "logical_key": key, "session": sid}
                if _meaningful_eligibility(row):
                    meaningful_rotation_sessions.add(sid)
                    event["classification"] = "MEANINGFUL_ROTATION"
                else:
                    stale_rotation_sessions.add(sid)
                    event["classification"] = "STALE_ROTATION"
                candidate_rotations.append(event)
            previous_index[key] = index

    exact_confirmed = _within_episode_exact_confirmed_rehandoff(summary, max_steps)
    pathological_sessions = set(stale_rehandoff_sessions) | set(cooldown_only_sessions) | set(stale_rotation_sessions)
    if exact_confirmed:
        pathological_sessions.update(range(-exact_confirmed, 0))
    all_local_sessions = len(handoffs)
    guard = summary.get("pending_handoff_guard", {}) if isinstance(summary.get("pending_handoff_guard"), Mapping) else {}
    return {
        "raw_pending_churn_rate": _safe_get_base_pending_rate(summary),
        "within_episode_same_candidate_rehandoff": within_episode_same,
        "cross_episode_same_logical_key_recurrence": cross_episode_same,
        "exact_canonical_candidate_rehandoff": exact_rehandoff,
        "related_candidate_rehandoff": related_rehandoff,
        "stale_rehandoff": len(stale_rehandoff_sessions),
        "cooldown_only_rehandoff": len(cooldown_only_sessions),
        "stale_candidate_rotation": len(stale_rotation_sessions),
        "meaningful_rotation": len(meaningful_rotation_sessions),
        "meaningful_rehandoff": meaningful_rehandoff,
        "new_candidate_handoff": sum(1 for row in handoffs if row.get("reason") == "NEW_CANDIDATE"),
        "cross_episode_key_recurrence": cross_episode_same,
        "pathological_churn_sessions": len(pathological_sessions),
        "pathological_churn_rate": _ratio(len(pathological_sessions), all_local_sessions),
        "stale_sessions_prevented": int(guard.get("stale_session_rehandoff_prevented", 0) or 0),
        "false_stale_suppression": int(guard.get("false_stale_suppression_count", 0) or 0),
        "stale_pending_unique_candidates": int(guard.get("stale_pending_unique_candidates", 0) or 0),
        "stale_candidate_evaluations": int(guard.get("stale_pending_evaluations", 0) or 0),
        "stale_rehandoff_actual": len(stale_rehandoff_sessions),
        "exact_confirmed_candidate_rehandoff": exact_confirmed,
        "related_confirmed_candidate_rehandoff": int(lifecycle.get("related_confirmed_candidate_rehandoff_count", 0) or 0),
        "rotation_examples": candidate_rotations[:10],
    }


def eligibility_signal_table(hybrid_row: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    rows = hybrid_row.get("local_sessions", []) if isinstance(hybrid_row.get("local_sessions"), list) else []
    by_session = {int(row.get("local_session_id", 0) or 0): row for row in rows}
    table = {key: {"eligible_events": 0, "actual_handoffs": 0, "productive_sessions": 0, "confirmed_sessions": 0} for key in ("FIRST_HANDOFF", *sorted(MEANINGFUL_ELIGIBILITY))}
    for row in rows:
        sid = int(row.get("local_session_id", 0) or 0)
        session = by_session.get(sid, row)
        reasons = set(str(item) for item in row.get("eligibility_reasons", []) or [])
        if row.get("entry_reason") == "NEW_CANDIDATE" or not reasons:
            reasons.add(str(row.get("eligibility_reason") or "FIRST_HANDOFF"))
        for reason in reasons:
            if reason not in table:
                continue
            table[reason]["eligible_events"] += 1
            table[reason]["actual_handoffs"] += 1
            table[reason]["productive_sessions"] += int(bool(session.get("productive")))
            table[reason]["confirmed_sessions"] += int(bool(session.get("trigger_candidate_confirmed") or session.get("any_candidate_confirmed")))
    return table


def return_reason_efficiency(hybrid_row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in hybrid_row.get("local_sessions", []) or []:
        if isinstance(row, Mapping):
            grouped[str(row.get("return_reason") or "")].append(row)
    result: dict[str, dict[str, Any]] = {}
    for reason, rows in grouped.items():
        result[reason] = {
            "sessions": len(rows),
            "productive_rate": _ratio(sum(1 for row in rows if row.get("productive")), len(rows)),
            "confirmed": sum(1 for row in rows if row.get("trigger_candidate_confirmed") or row.get("any_candidate_confirmed")),
            "mean_steps": mean(int(row.get("session_length", 0) or 0) for row in rows) if rows else 0.0,
        }
    return result


def classify_final_seed(hybrid_row: Mapping[str, Any], coverage: Mapping[str, Any], churn: Mapping[str, Any]) -> str:
    hard = hybrid_row.get("hard_invariants", {}) if isinstance(hybrid_row.get("hard_invariants"), Mapping) else {}
    if not hard.get("all_pass") or hard.get("ground_truth_used") or hard.get("zero_step_switch_loop"):
        return "OTHER"
    if int(churn.get("pathological_churn_sessions", 0) or 0) > 0:
        return "PATHOLOGICAL_CHURN"
    if float(coverage.get("state_coverage_retention", 1.0) or 0.0) < 0.5 and float(coverage.get("target_coverage_retention", 1.0) or 0.0) < 0.5:
        return "COVERAGE_REGRESSION"
    confirmed = int(hybrid_row.get("confirmation_attribution", {}).get("ppo_candidate_dqn_confirmed_count", 0) or 0)
    productive = float(hybrid_row.get("productive_local_session_rate", 0.0) or 0.0)
    if confirmed <= 0 and productive < 0.35:
        return "DQN_LOW_VALUE"
    if productive < 0.4 and float(hybrid_row.get("unproductive_handoff_rate", 0.0) or 0.0) > 0.6:
        return "HANDOFF_OVERUSE"
    return "HEALTHY_HYBRID"


def classify_final_overall(rows: Mapping[str, Mapping[str, Any]]) -> str:
    classes = Counter(str(row.get("classification") or "") for row in rows.values())
    if classes.get("PATHOLOGICAL_CHURN", 0):
        return "RESIDUAL_PATHOLOGICAL_CHURN"
    if classes.get("COVERAGE_REGRESSION", 0):
        return "COVERAGE_TRADEOFF_TOO_HIGH"
    if classes.get("DQN_LOW_VALUE", 0):
        return "DQN_CONTRIBUTION_TOO_LOW"
    if classes.get("OTHER", 0):
        return "REVIEW"
    return "HYBRID_V1_STABLE"


def aggregate_final_rows(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        output: list[float] = []
        for row in rows.values():
            current: Any = row
            for key in path:
                current = current.get(key, {}) if isinstance(current, Mapping) else {}
            if isinstance(current, (int, float)):
                output.append(float(current))
        return output

    return {
        "orchestration": {
            "ppo_step_ratio_mean": _mean(values(("hybrid", "ppo_step_ratio"))),
            "dqn_step_ratio_mean": _mean(values(("hybrid", "dqn_step_ratio"))),
            "handoffs_per_episode_mean": _mean(values(("hybrid", "handoffs_per_episode"))),
            "productive_rate_mean": _mean(values(("hybrid", "productive_local_session_rate"))),
            "unproductive_rate_mean": _mean(values(("hybrid", "unproductive_handoff_rate"))),
            "raw_pending_churn_mean": _mean(values(("final_churn", "raw_pending_churn_rate"))),
            "pathological_churn_rate_mean": _mean(values(("final_churn", "pathological_churn_rate"))),
        },
        "churn": {
            "within_episode_same_candidate_rehandoff": int(sum(values(("final_churn", "within_episode_same_candidate_rehandoff")))),
            "cross_episode_same_logical_key_recurrence": int(sum(values(("final_churn", "cross_episode_same_logical_key_recurrence")))),
            "stale_rehandoff": int(sum(values(("final_churn", "stale_rehandoff")))),
            "cooldown_only_rehandoff": int(sum(values(("final_churn", "cooldown_only_rehandoff")))),
            "stale_candidate_rotation": int(sum(values(("final_churn", "stale_candidate_rotation")))),
            "meaningful_rehandoff": int(sum(values(("final_churn", "meaningful_rehandoff")))),
            "pathological_churn_sessions": int(sum(values(("final_churn", "pathological_churn_sessions")))),
            "stale_sessions_prevented": int(sum(values(("final_churn", "stale_sessions_prevented")))),
            "false_stale_suppression": int(sum(values(("final_churn", "false_stale_suppression")))),
        },
        "contribution": {
            "ppo_candidates": int(sum(values(("hybrid", "candidate_attribution", "ppo_discovered_candidates")))),
            "dqn_candidates": int(sum(values(("hybrid", "candidate_attribution", "dqn_discovered_candidates")))),
            "ppo_to_dqn_opportunities": int(sum(values(("hybrid", "confirmation_attribution", "ppo_candidate_dqn_opportunity_count")))),
            "ppo_to_dqn_confirmed": int(sum(values(("hybrid", "confirmation_attribution", "ppo_candidate_dqn_confirmed_count")))),
            "ppo_to_dqn_confirmation_rate": _ratio(
                sum(values(("hybrid", "confirmation_attribution", "ppo_candidate_dqn_confirmed_count"))),
                sum(values(("hybrid", "confirmation_attribution", "ppo_candidate_dqn_opportunity_count"))),
            ),
        },
        "coverage": {
            "state_retention_mean": _mean(values(("coverage_retention", "state_coverage_retention"))),
            "target_retention_mean": _mean(values(("coverage_retention", "target_coverage_retention"))),
            "action_target_retention_mean": _mean(values(("coverage_retention", "action_target_coverage_retention"))),
        },
    }


def final_freeze_manifest(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "PPO Global Explorer + DQN Local Specialist",
        "controller_version": summary.get("controller_version", "hybrid_controller_v1_stable_pending"),
        "ppo_obs_dim": 635,
        "dqn_obs_dim": 976,
        "action_dim": 608,
        "dqn_max_local_steps": 5,
        "handoff_priority": ["NEW_CANDIDATE", "PENDING_CONFIRMATION", "FUNCTIONAL_REGION", "STALLED_FUNCTIONAL"],
        "return_priority": ["EPISODE_DONE", "CONFIRMED", "CONTEXT_CHANGED", "REPEAT_GUARD", "NOVELTY_EXHAUSTED", "LOCAL_BUDGET"],
        "pending_rehandoff": "meaningful-change-only",
        "cooldown_only_rehandoff": "disabled",
        "confirmation_identity": "canonical_tracker_logical_key",
        "validation_site": "site001",
        "validation_seeds": summary.get("seeds", [42, 43, 44]),
        "HYBRID_BEHAVIOR_CHECK": summary.get("HYBRID_BEHAVIOR_CHECK"),
        "HYBRID_CONTROLLER_CHECK": summary.get("HYBRID_CONTROLLER_CHECK"),
        "HYBRID_CONTROLLER_FREEZE": summary.get("HYBRID_CONTROLLER_FREEZE"),
    }


def _compact_lifecycle(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pending_sessions": lifecycle.get("pending_sessions", 0),
        "pending_confirmed_sessions": lifecycle.get("pending_confirmed_sessions", 0),
        "trigger_candidate_confirmed_sessions_exact": lifecycle.get("trigger_candidate_confirmed_sessions_exact", 0),
        "confirmed_return_count": lifecycle.get("confirmed_return_count", 0),
        "trigger_confirmed_but_return_not_confirmed_count": lifecycle.get("trigger_confirmed_but_return_not_confirmed_count", 0),
        "cooldown_only_rehandoff": lifecycle.get("cooldown_only_rehandoff", {}),
        "stale_pending": lifecycle.get("stale_pending", {}),
        "exact_confirmed_candidate_rehandoff_count": lifecycle.get("exact_confirmed_candidate_rehandoff_count", 0),
        "related_confirmed_candidate_rehandoff_count": lifecycle.get("related_confirmed_candidate_rehandoff_count", 0),
    }


def _meaningful_eligibility(row: Mapping[str, Any]) -> bool:
    reason = str(row.get("eligibility_reason") or "")
    reasons = {str(item) for item in row.get("eligibility_reasons", []) or []}
    return reason in MEANINGFUL_ELIGIBILITY or bool(reasons.intersection(MEANINGFUL_ELIGIBILITY))


def _same_stale_material_state(row: Mapping[str, Any], prior: Mapping[str, Any]) -> bool:
    return (
        row.get("entry_context_fingerprint") == prior.get("entry_context_fingerprint")
        and row.get("trigger_candidate_progress_at_entry") == prior.get("trigger_candidate_progress_at_entry")
        and row.get("trigger_candidate_resolvable_at_entry") == prior.get("trigger_candidate_resolvable_at_entry")
        and row.get("entry_reproduction_requirement_fingerprint") == prior.get("entry_reproduction_requirement_fingerprint")
    )


def _within_episode_exact_confirmed_rehandoff(summary: Mapping[str, Any], max_steps: int) -> int:
    handoffs = [row for row in summary.get("handoff_events", []) or [] if isinstance(row, Mapping)]
    records = [
        row
        for row in (summary.get("confirmation_attribution", {}) or {}).get("confirmation_records", []) or []
        if isinstance(row, Mapping)
    ]
    handoff_by_session = {int(row.get("local_session_id", 0) or 0): row for row in handoffs}
    count = 0
    for record in records:
        key = str(record.get("logical_key") or "")
        confirmed_session = int(record.get("local_session_id", 0) or 0)
        confirmed_handoff = handoff_by_session.get(confirmed_session, {})
        confirmed_episode = _episode(confirmed_handoff, max_steps)
        for handoff in handoffs:
            sid = int(handoff.get("local_session_id", 0) or 0)
            if sid <= confirmed_session:
                continue
            if _episode(handoff, max_steps) != confirmed_episode:
                continue
            if str(handoff.get("trigger_candidate_key") or "") == key:
                count += 1
                break
    return count


def _episode(row: Mapping[str, Any], max_steps: int) -> int:
    if max_steps <= 0:
        return 0
    return (int(row.get("handoff_step", row.get("step", 0)) or 0) - 1) // max_steps


def _safe_get_base_pending_rate(summary: Mapping[str, Any]) -> float:
    row = summarize_hybrid_artifact(summary)
    return float(row.get("pending_confirmation_churn_rate", 0.0) or 0.0)


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
        return float(numerator) / denom if denom else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
