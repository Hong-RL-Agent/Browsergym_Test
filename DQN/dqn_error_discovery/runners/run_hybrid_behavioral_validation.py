"""Aggregate RAWD Hybrid v1 behavioral validation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SEEDS = (42, 43, 44)
HANDOFF_REASONS = ("NEW_CANDIDATE", "PENDING_CONFIRMATION", "FUNCTIONAL_REGION", "STALLED_FUNCTIONAL")
RETURN_REASONS = ("CONFIRMED", "CONTEXT_CHANGED", "REPEAT_GUARD", "NOVELTY_EXHAUSTED", "LOCAL_BUDGET", "EPISODE_DONE")


def summarize_hybrid_artifact(summary: Mapping[str, Any]) -> dict[str, Any]:
    config = summary.get("configuration", {}) if isinstance(summary.get("configuration"), Mapping) else {}
    episode_summary = summary.get("episode_summary", {}) if isinstance(summary.get("episode_summary"), Mapping) else {}
    timeline = [row for row in summary.get("timeline", []) or [] if isinstance(row, Mapping)]
    handoffs = [row for row in summary.get("handoff_events", []) or [] if isinstance(row, Mapping)]
    returns = [row for row in summary.get("return_events", []) or [] if isinstance(row, Mapping)]
    guard = summary.get("pending_handoff_guard", {}) if isinstance(summary.get("pending_handoff_guard"), Mapping) else {}
    candidate_attr = summary.get("candidate_attribution", {}) if isinstance(summary.get("candidate_attribution"), Mapping) else {}
    confirmation_attr = summary.get("confirmation_attribution", {}) if isinstance(summary.get("confirmation_attribution"), Mapping) else {}
    coverage = summary.get("ppo_global_coverage", {}) if isinstance(summary.get("ppo_global_coverage"), Mapping) else {}
    hard = summary.get("hard_invariants", {}) if isinstance(summary.get("hard_invariants"), Mapping) else {}
    diagnostic = summary.get("diagnostic", {}) if isinstance(summary.get("diagnostic"), Mapping) else {}

    total_steps = int(episode_summary.get("total_steps", len(timeline)) or 0)
    ppo_steps = int(episode_summary.get("ppo_steps", sum(1 for row in timeline if row.get("active_agent") == "PPO_GLOBAL")) or 0)
    dqn_steps = int(episode_summary.get("dqn_steps", sum(1 for row in timeline if row.get("active_agent") == "DQN_LOCAL")) or 0)
    episodes = int(config.get("episodes", 0) or 0)
    local_sessions = _session_summaries(timeline, handoffs, returns, confirmation_attr)
    productive = [row for row in local_sessions if row["productive"]]
    unproductive = [row for row in local_sessions if not row["productive"]]
    pending_sessions = [row for row in local_sessions if row["entry_reason"] == "PENDING_CONFIRMATION"]
    pending_any_confirmed_sessions = [row for row in pending_sessions if row["any_candidate_confirmed"]]
    pending_trigger_confirmed_sessions = [row for row in pending_sessions if row["trigger_candidate_confirmed"]]
    progress_sessions = [row for row in local_sessions if row["confirmation_progress"]]

    handoff_reason_counts = Counter(str(row.get("reason") or "") for row in handoffs)
    return_reason_counts = Counter(str(row.get("reason") or "") for row in returns)
    same_rehandoff, same_distances = same_candidate_rehandoff(handoffs, timeline)
    pending_churn = pending_confirmation_churn(handoffs, confirmation_attr)
    confirmation_records = normalize_confirmation_records(
        confirmation_attr.get("confirmation_records", []) or [],
        candidate_attr.get("candidate_first_agent", {}) if isinstance(candidate_attr.get("candidate_first_agent"), Mapping) else {},
    )
    matrix = attribution_matrix(confirmation_records)
    ppo_to_dqn_confirmed = int(matrix.get("PPO->DQN", candidate_attr.get("ppo_candidate_dqn_confirmed_count", 0)) or 0)
    opportunity = max(ppo_candidate_dqn_opportunity_count(candidate_attr, handoffs), ppo_to_dqn_confirmed)
    dqn_confirmed = sum(count for key, count in matrix.items() if key.endswith("->DQN"))

    dqn_repeat = _dqn_repeat_summary(timeline)
    ppo_behavior = _ppo_behavior_summary(timeline)
    coverage_summary = {
        "unique_states": int(coverage.get("unique_states", 0) or 0),
        "unique_urls": int(coverage.get("unique_urls", 0) or 0),
        "unique_targets": int(coverage.get("unique_targets", 0) or 0),
        "unique_action_targets": int(coverage.get("unique_action_targets", 0) or 0),
        "unique_action_types": int(coverage.get("unique_action_types", 0) or 0),
        "meaningful_state_transitions": int(coverage.get("meaningful_state_transitions", 0) or 0),
    }
    confirmed = int(confirmation_attr.get("unique_confirmed", confirmation_attr.get("newly_confirmed", 0)) or 0)
    local_session_count = len(local_sessions)
    return {
        "seed": int(config.get("seed", 0) or 0),
        "episodes": episodes,
        "total_steps": total_steps,
        "ppo_steps": ppo_steps,
        "dqn_steps": dqn_steps,
        "ppo_step_ratio": _ratio(ppo_steps, total_steps),
        "dqn_step_ratio": _ratio(dqn_steps, total_steps),
        "handoff_count": len(handoffs),
        "return_count": len(returns),
        "handoffs_per_episode": _ratio(len(handoffs), episodes),
        "handoff_rate": _ratio(len(handoffs), total_steps),
        "local_session_count": local_session_count,
        "mean_local_session": mean([row["session_length"] for row in local_sessions]) if local_sessions else 0.0,
        "max_local_session": max([row["session_length"] for row in local_sessions], default=0),
        "handoff_reasons": {reason: int(handoff_reason_counts.get(reason, 0)) for reason in HANDOFF_REASONS},
        "handoff_reason_ratios": {reason: _ratio(handoff_reason_counts.get(reason, 0), len(handoffs)) for reason in HANDOFF_REASONS},
        "return_reasons": {reason: int(return_reason_counts.get(reason, 0)) for reason in RETURN_REASONS},
        "return_reason_ratios": {reason: _ratio(return_reason_counts.get(reason, 0), len(returns)) for reason in RETURN_REASONS},
        "local_sessions": local_sessions,
        "productive_sessions": len(productive),
        "unproductive_sessions": len(unproductive),
        "productive_local_session_rate": _ratio(len(productive), local_session_count),
        "unproductive_handoff_rate": _ratio(len(unproductive), local_session_count),
        "pending_handoff_count": len(pending_sessions),
        "pending_handoff_productive_count": sum(1 for row in pending_sessions if row["productive"]),
        "pending_any_candidate_confirmed_sessions": len(pending_any_confirmed_sessions),
        "pending_trigger_candidate_confirmed_sessions": len(pending_trigger_confirmed_sessions),
        "pending_handoff_confirmed_count": len(pending_trigger_confirmed_sessions),
        "pending_handoff_unproductive_count": sum(1 for row in pending_sessions if not row["productive"]),
        "pending_handoff_success_rate": _ratio(sum(1 for row in pending_sessions if row["productive"]), len(pending_sessions)),
        "same_candidate_rehandoff_count": same_rehandoff,
        "mean_ppo_steps_between_same_candidate_handoffs": mean(same_distances) if same_distances else None,
        "pending_confirmation_churn_count": pending_churn,
        "pending_confirmation_churn_rate": _ratio(pending_churn, len(pending_sessions)),
        "pending_handoff_guard": {
            "pending_rehandoff_min_ppo_steps": int(guard.get("pending_rehandoff_min_ppo_steps", 0) or 0),
            "pending_handoff_candidate_count": int(guard.get("pending_handoff_candidate_count", 0) or 0),
            "pending_handoff_suppressed_count": int(guard.get("pending_handoff_suppressed_count", 0) or 0),
            "pending_rehandoff_eligible_count": int(guard.get("pending_rehandoff_eligible_count", 0) or 0),
            "pending_rehandoff_blocked_count": int(guard.get("pending_rehandoff_blocked_count", 0) or 0),
            "same_candidate_rehandoff_blocked_count": int(guard.get("same_candidate_rehandoff_blocked_count", 0) or 0),
            "same_context_functional_bypass_blocked_count": int(guard.get("same_context_functional_bypass_blocked_count", 0) or 0),
            "same_context_stalled_bypass_blocked_count": int(guard.get("same_context_stalled_bypass_blocked_count", 0) or 0),
            "rehandoff_due_to_progress": int(guard.get("rehandoff_due_to_progress", 0) or 0),
            "rehandoff_due_to_context_change": int(guard.get("rehandoff_due_to_context_change", 0) or 0),
            "rehandoff_due_to_target_resolvable": int(guard.get("rehandoff_due_to_target_resolvable", 0) or 0),
            "rehandoff_due_to_reproduction_requirement": int(guard.get("rehandoff_due_to_reproduction_requirement", 0) or 0),
            "stale_pending_evaluations": int(guard.get("stale_pending_evaluations", 0) or 0),
            "stale_pending_handoff_blocked": int(guard.get("stale_pending_handoff_blocked", 0) or 0),
            "stale_session_rehandoff_prevented": int(guard.get("stale_session_rehandoff_prevented", 0) or 0),
            "eligibility_reasons": guard.get("eligibility_reasons", {}) if isinstance(guard.get("eligibility_reasons"), Mapping) else {},
            "suppression_reasons": guard.get("suppression_reasons", {}) if isinstance(guard.get("suppression_reasons"), Mapping) else {},
        },
        "candidate_attribution": {
            "unique_candidates": int(candidate_attr.get("unique_candidates", 0) or 0),
            "ppo_discovered_candidates": int(candidate_attr.get("ppo_discovered_candidates", 0) or 0),
            "dqn_discovered_candidates": int(candidate_attr.get("dqn_discovered_candidates", 0) or 0),
            "dqn_new_candidate_count": int(candidate_attr.get("dqn_discovered_candidates", 0) or 0),
        },
        "confirmation_attribution": {
            "unique_confirmed": confirmed,
            "attribution_matrix": matrix,
            "normalized_confirmation_records": confirmation_records,
            "ppo_candidate_dqn_opportunity_count": opportunity,
            "ppo_candidate_dqn_confirmed_count": ppo_to_dqn_confirmed,
            "ppo_to_dqn_confirmation_rate": _ratio(ppo_to_dqn_confirmed, opportunity),
            "dqn_confirmed": dqn_confirmed,
            "duplicate_reward_leak": int(confirmation_attr.get("duplicate_reward_leak", 0) or 0),
        },
        "coverage": coverage_summary,
        "dqn_progress_sessions": len(progress_sessions),
        "dqn_new_local_state_sessions": sum(1 for row in local_sessions if row["new_local_state"]),
        "dqn_new_target_sessions": sum(1 for row in local_sessions if row["new_target"]),
        "dqn_repeat": dqn_repeat,
        "ppo_behavior": ppo_behavior,
        "transition_matrix": agent_transition_matrix(timeline),
        "session_pattern": session_pattern(timeline),
        "efficiency": {
            "steps_per_confirmed": _ratio(total_steps, confirmed) if confirmed else None,
            "dqn_steps_per_dqn_confirmed": _ratio(dqn_steps, dqn_confirmed) if dqn_confirmed else None,
            "handoffs_per_confirmed": _ratio(len(handoffs), confirmed) if confirmed else None,
            "confirmation_yield_per_local_session": _ratio(dqn_confirmed, local_session_count),
        },
        "hard_invariants": {
            "all_pass": bool(hard.get("all_pass")),
            "action_error_zero": bool(hard.get("action_error_zero")),
            "action_error_count": int(diagnostic.get("action_error_count", 0) or 0),
            "fake_instruction_zero": bool(hard.get("fake_instruction_zero", True)),
            "fake_instruction_count": int(diagnostic.get("fake_instruction_count", 0) or 0),
            "duplicate_reward_leak_zero": bool(hard.get("duplicate_reward_leak_zero")),
            "ground_truth_used": bool(hard.get("ground_truth_used")),
            "zero_step_switch_loop": bool(hard.get("zero_step_switch_loop")),
            "local_budget_respected": bool(hard.get("local_budget_respected")),
            "separate_buffers": bool(hard.get("separate_buffers")),
            "separate_reward_functions": bool(hard.get("separate_reward_functions")),
        },
    }


def summarize_ppo_only_artifact(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), Mapping) else {}
    reward_totals = summary.get("reward_component_totals", {}) if isinstance(summary.get("reward_component_totals"), Mapping) else {}
    return {
        "total_steps": int(metrics.get("total_steps", 0) or 0),
        "unique_states": int(metrics.get("unique_states", 0) or 0),
        "unique_urls": int(metrics.get("unique_urls", 0) or 0),
        "unique_targets": int(metrics.get("unique_targets", 0) or 0),
        "unique_action_targets": int(metrics.get("unique_action_targets", 0) or 0),
        "candidates": int(metrics.get("unique_candidate_signatures", metrics.get("candidate_anomaly_count", 0)) or 0),
        "confirmed": int(metrics.get("unique_confirmed_signatures", metrics.get("confirmed_anomaly_count", 0)) or 0),
        "mean_reward": _ratio(float(reward_totals.get("final_reward", reward_totals.get("total", 0.0)) or 0.0), int(metrics.get("total_steps", 0) or 0)),
        "action_error": int(metrics.get("action_error_count", 0) or 0),
    }


def aggregate_validation(hybrid_by_seed: Mapping[int, Mapping[str, Any]], ppo_by_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted(hybrid_by_seed)
    hybrid_rows = {seed: summarize_hybrid_artifact(hybrid_by_seed[seed]) for seed in seeds}
    ppo_rows = {seed: summarize_ppo_only_artifact(ppo_by_seed[seed]) for seed in seeds if seed in ppo_by_seed}
    coverage_comparison = {seed: coverage_retention(hybrid_rows[seed], ppo_rows.get(seed, {})) for seed in seeds}
    classifications = {seed: classify_seed(hybrid_rows[seed], coverage_comparison.get(seed, {})) for seed in seeds}
    guard_classifications = {seed: classify_guard_seed(hybrid_rows[seed]) for seed in seeds}
    overall = classify_overall(hybrid_rows, coverage_comparison, classifications)
    guard_overall = classify_guard_overall(guard_classifications)
    return {
        "seeds": seeds,
        "hybrid": hybrid_rows,
        "ppo_only": ppo_rows,
        "coverage_comparison": coverage_comparison,
        "aggregate_orchestration": aggregate_numeric(
            hybrid_rows,
            ("ppo_step_ratio", "dqn_step_ratio", "handoffs_per_episode", "mean_local_session", "productive_local_session_rate", "unproductive_handoff_rate", "pending_confirmation_churn_rate"),
        ),
        "aggregate_contribution": aggregate_contribution(hybrid_rows),
        "aggregate_coverage": aggregate_coverage(hybrid_rows, ppo_rows),
        "aggregate_efficiency": aggregate_efficiency(hybrid_rows),
        "seed_classification": classifications,
        "pending_guard_seed_classification": guard_classifications,
        "overall_classification": overall,
        "pending_guard_overall_classification": guard_overall,
        "HYBRID_PENDING_GUARD_CHECK": "PASS" if guard_overall == "PENDING_CHURN_FIXED" else "REVIEW",
        "HYBRID_BEHAVIOR_CHECK": "PASS" if overall == "HYBRID_BEHAVIOR_PASS" else "REVIEW",
    }


def _session_summaries(
    timeline: list[Mapping[str, Any]],
    handoffs: list[Mapping[str, Any]],
    returns: list[Mapping[str, Any]],
    confirmation_attr: Mapping[str, Any],
) -> list[dict[str, Any]]:
    handoff_by_id = {int(row.get("local_session_id", 0) or 0): row for row in handoffs}
    return_by_id = {int(row.get("local_session_id", 0) or 0): row for row in returns}
    records = [row for row in confirmation_attr.get("confirmation_records", []) or [] if isinstance(row, Mapping)]
    confirmations_by_session = Counter(int(row.get("local_session_id", 0) or 0) for row in records if row.get("confirmed_by") == "DQN")
    confirmation_keys_by_session: dict[int, set[str]] = defaultdict(set)
    for record in records:
        sid = int(record.get("local_session_id", 0) or 0)
        key = str(record.get("logical_key") or "")
        if sid and key:
            confirmation_keys_by_session[sid].add(key)
    rows_by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in timeline:
        if row.get("active_agent") == "DQN_LOCAL":
            sid = int(row.get("local_session_id", 0) or 0)
            if sid:
                rows_by_session[sid].append(row)
    session_ids = sorted(set(rows_by_session) | set(handoff_by_id) | set(return_by_id))
    sessions: list[dict[str, Any]] = []
    for sid in session_ids:
        rows = rows_by_session.get(sid, [])
        handoff = handoff_by_id.get(sid, {})
        returned = return_by_id.get(sid, {})
        confirmed_count = int(confirmations_by_session.get(sid, 0))
        trigger_key = str(handoff.get("trigger_candidate_key") or "")
        confirmed_keys = confirmation_keys_by_session.get(sid, set())
        trigger_confirmed = bool(trigger_key and trigger_key in confirmed_keys)
        new_candidate_count = sum(len(row.get("new_candidate_keys", []) or []) for row in rows)
        component_totals = Counter()
        repeat_counts = Counter()
        for row in rows:
            for key, value in (row.get("reward_components", {}) or {}).items():
                component_totals[key] += float(value or 0.0)
            repeat = row.get("repeat_diagnostic", {}) or {}
            for key in ("immediate_repeat", "recent_repeat", "cycle_repeat", "unproductive_repeat"):
                repeat_counts[key] += int(bool(repeat.get(key)))
        confirmation_progress = component_totals["confirmation_upgrade"] > 0.0 or component_totals["confirmed_anomaly"] > 0.0 or confirmed_count > 0
        new_local_state = component_totals["new_local_state"] > 0.0
        new_target = component_totals["new_local_action_target"] > 0.0 or component_totals["new_target"] > 0.0 or component_totals["new_action_target"] > 0.0
        flow_progress = component_totals["flow_progress"] > 0.0
        productive = bool(new_candidate_count or confirmation_progress or new_local_state or new_target or flow_progress)
        sessions.append(
            {
                "local_session_id": sid,
                "entry_reason": str(handoff.get("reason") or ""),
                "session_length": len(rows),
                "return_reason": str(return_by_id.get(sid, {}).get("reason") or ""),
                "trigger_candidate_key": trigger_key,
                "resolved_candidate_key": str(returned.get("resolved_candidate_key") or ""),
                "trigger_candidate_confirmed": bool(trigger_confirmed or (trigger_key and returned.get("resolved_candidate_key") == trigger_key)),
                "any_candidate_confirmed": confirmed_count > 0,
                "entry_candidate_count": len(handoff.get("entry_candidate_keys", []) or []),
                "entry_pending_count": len(handoff.get("entry_pending_confirmation_keys", []) or []),
                "confirmed_count": confirmed_count,
                "new_candidate_count": new_candidate_count,
                "confirmation_progress": bool(confirmation_progress),
                "new_local_state": bool(new_local_state),
                "new_target": bool(new_target),
                "flow_progress": bool(flow_progress),
                "productive": productive,
                "repeat_counts": dict(repeat_counts),
                "reward_total": sum(float(row.get("reward", 0.0) or 0.0) for row in rows),
            }
        )
    return sessions


def same_candidate_rehandoff(handoffs: list[Mapping[str, Any]], timeline: list[Mapping[str, Any]]) -> tuple[int, list[int]]:
    count = 0
    distances: list[int] = []
    previous: list[tuple[int, set[str]]] = []
    for handoff in handoffs:
        step = int(handoff.get("step", 0) or 0)
        keys = set(handoff.get("entry_pending_confirmation_keys", []) or []) | set(handoff.get("entry_candidate_keys", []) or [])
        if not keys:
            continue
        for prev_step, prev_keys in previous:
            if keys.intersection(prev_keys):
                count += 1
                distances.append(sum(1 for row in timeline if prev_step < int(row.get("step", 0) or 0) < step and row.get("active_agent") == "PPO_GLOBAL"))
                break
        previous.append((step, keys))
    return count, distances


def pending_confirmation_churn(handoffs: list[Mapping[str, Any]], confirmation_attr: Mapping[str, Any]) -> int:
    confirmed_keys = {str(row.get("logical_key") or "") for row in confirmation_attr.get("confirmation_records", []) or []}
    active_pending: dict[str, int] = {}
    churn = 0
    for handoff in handoffs:
        if handoff.get("reason") != "PENDING_CONFIRMATION":
            continue
        keys = set(handoff.get("entry_pending_confirmation_keys", []) or []) | set(handoff.get("entry_candidate_keys", []) or [])
        for key in keys:
            if key in confirmed_keys:
                continue
            if key in active_pending:
                churn += 1
                break
            active_pending[key] = int(handoff.get("local_session_id", 0) or 0)
    return churn


def ppo_candidate_dqn_opportunity_count(candidate_attr: Mapping[str, Any], handoffs: list[Mapping[str, Any]]) -> int:
    first_agent = candidate_attr.get("candidate_first_agent", {}) if isinstance(candidate_attr.get("candidate_first_agent"), Mapping) else {}
    ppo_keys = {str(key) for key, agent in first_agent.items() if agent == "PPO"}
    dqn_entry_keys: set[str] = set()
    for row in handoffs:
        dqn_entry_keys.update(str(key) for key in row.get("entry_candidate_keys", []) or [])
        dqn_entry_keys.update(str(key) for key in row.get("entry_pending_confirmation_keys", []) or [])
    return len(ppo_keys.intersection(dqn_entry_keys))


def attribution_matrix(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in records:
        counter[f"{row.get('discovered_by', 'UNKNOWN')}->{row.get('confirmed_by', 'UNKNOWN')}"] += 1
    for key in ("PPO->PPO", "PPO->DQN", "DQN->DQN", "DQN->PPO"):
        counter.setdefault(key, 0)
    return dict(counter)


def normalize_confirmation_records(records: Iterable[Mapping[str, Any]], candidate_first_agent: Mapping[str, str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        if row.get("discovered_by") in {"PPO", "DQN"}:
            normalized.append(row)
            continue
        inferred = _infer_discovered_by(str(row.get("logical_key") or ""), candidate_first_agent)
        row["discovered_by"] = inferred or row.get("discovered_by") or "UNKNOWN"
        normalized.append(row)
    return normalized


def _infer_discovered_by(confirmed_key: str, candidate_first_agent: Mapping[str, str]) -> str:
    if confirmed_key in candidate_first_agent:
        return str(candidate_first_agent[confirmed_key])
    rule = confirmed_key.split("|", 1)[0] if confirmed_key else ""
    tokens = {token for token in confirmed_key.replace("target:", "|").replace("bid:", "|").split("|") if token}
    matches: list[str] = []
    for candidate_key, agent in candidate_first_agent.items():
        if rule and not str(candidate_key).startswith(rule + "|"):
            continue
        candidate_tokens = {token for token in str(candidate_key).replace("target:", "|").replace("bid:", "|").split("|") if token}
        if tokens.intersection(candidate_tokens):
            matches.append(str(agent))
    if "PPO" in matches:
        return "PPO"
    if "DQN" in matches:
        return "DQN"
    return ""


def _dqn_repeat_summary(timeline: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in timeline if row.get("active_agent") == "DQN_LOCAL"]
    counts = Counter()
    for row in rows:
        repeat = row.get("repeat_diagnostic", {}) or {}
        for key in ("immediate_repeat", "recent_repeat", "cycle_repeat", "unproductive_repeat"):
            counts[key] += int(bool(repeat.get(key)))
    total = len(rows)
    return {
        "steps": total,
        "repeat": _ratio(counts["immediate_repeat"] + counts["recent_repeat"] + counts["cycle_repeat"], total),
        "immediate": _ratio(counts["immediate_repeat"], total),
        "recent": _ratio(counts["recent_repeat"], total),
        "cycle": _ratio(counts["cycle_repeat"], total),
        "unproductive": _ratio(counts["unproductive_repeat"], total),
        "counts": dict(counts),
    }


def _ppo_behavior_summary(timeline: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in timeline if row.get("active_agent") == "PPO_GLOBAL"]
    actions = [str(row.get("action_type") or "") for row in rows]
    repeats = sum(1 for index in range(1, len(actions)) if actions[index] and actions[index] == actions[index - 1])
    return {"action_distribution": dict(Counter(actions)), "unique_action_types": len(set(item for item in actions if item)), "repeated_action_ratio": _ratio(repeats, len(actions))}


def agent_transition_matrix(timeline: list[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter()
    agents = [str(row.get("active_agent") or "") for row in timeline]
    for left, right in zip(agents, agents[1:]):
        counter[f"{left}->{right}"] += 1
    for key in ("PPO_GLOBAL->PPO_GLOBAL", "PPO_GLOBAL->DQN_LOCAL", "DQN_LOCAL->DQN_LOCAL", "DQN_LOCAL->PPO_GLOBAL"):
        counter.setdefault(key, 0)
    return dict(counter)


def session_pattern(timeline: list[Mapping[str, Any]]) -> list[str]:
    pattern: list[str] = []
    last = ""
    length = 0
    for row in timeline:
        agent = "PPO" if row.get("active_agent") == "PPO_GLOBAL" else "DQN"
        if agent != last and last:
            pattern.append(f"{last}{length}")
            length = 0
        last = agent
        length += 1
    if last:
        pattern.append(f"{last}{length}")
    return pattern


def coverage_retention(hybrid: Mapping[str, Any], ppo: Mapping[str, Any]) -> dict[str, Any]:
    cov = hybrid.get("coverage", {}) if isinstance(hybrid.get("coverage"), Mapping) else {}
    return {
        "state_coverage_retention": _ratio(cov.get("unique_states", 0), ppo.get("unique_states", 0)),
        "target_coverage_retention": _ratio(cov.get("unique_targets", 0), ppo.get("unique_targets", 0)),
        "action_target_coverage_retention": _ratio(cov.get("unique_action_targets", 0), ppo.get("unique_action_targets", 0)),
        "delta_states": int(cov.get("unique_states", 0) or 0) - int(ppo.get("unique_states", 0) or 0),
        "delta_targets": int(cov.get("unique_targets", 0) or 0) - int(ppo.get("unique_targets", 0) or 0),
        "delta_action_targets": int(cov.get("unique_action_targets", 0) or 0) - int(ppo.get("unique_action_targets", 0) or 0),
        "delta_candidates": int(hybrid.get("candidate_attribution", {}).get("unique_candidates", 0) or 0) - int(ppo.get("candidates", 0) or 0),
        "delta_confirmed": int(hybrid.get("confirmation_attribution", {}).get("unique_confirmed", 0) or 0) - int(ppo.get("confirmed", 0) or 0),
    }


def aggregate_numeric(rows: Mapping[int, Mapping[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {key: {str(seed): float(row.get(key, 0.0) or 0.0) for seed, row in rows.items()} | {"mean": mean(float(row.get(key, 0.0) or 0.0) for row in rows.values()) if rows else 0.0} for key in keys}


def aggregate_contribution(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed, row in rows.items():
        cand = row["candidate_attribution"]
        conf = row["confirmation_attribution"]
        result[str(seed)] = {
            "ppo_candidates": cand["ppo_discovered_candidates"],
            "dqn_candidates": cand["dqn_discovered_candidates"],
            "ppo_to_dqn_confirmed": conf["attribution_matrix"].get("PPO->DQN", 0),
            "dqn_to_dqn_confirmed": conf["attribution_matrix"].get("DQN->DQN", 0),
            "dqn_progress_sessions": row["dqn_progress_sessions"],
        }
    result["total"] = {
        key: sum(seed_row[key] for seed_row in result.values() if isinstance(seed_row, Mapping))
        for key in ("ppo_candidates", "dqn_candidates", "ppo_to_dqn_confirmed", "dqn_to_dqn_confirmed", "dqn_progress_sessions")
    }
    return result


def aggregate_coverage(hybrid_rows: Mapping[int, Mapping[str, Any]], ppo_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("unique_states", "unique_targets", "unique_action_targets")
    result: dict[str, Any] = {}
    for key in keys:
        ppo_values = [float(row.get(key, 0) or 0) for row in ppo_rows.values()]
        hybrid_values = [float(row.get("coverage", {}).get(key, 0) or 0) for row in hybrid_rows.values()]
        result[key] = {"ppo_only_mean": mean(ppo_values) if ppo_values else 0.0, "hybrid_mean": mean(hybrid_values) if hybrid_values else 0.0}
    result["candidates"] = {
        "ppo_only_mean": mean(float(row.get("candidates", 0) or 0) for row in ppo_rows.values()) if ppo_rows else 0.0,
        "hybrid_mean": mean(float(row.get("candidate_attribution", {}).get("unique_candidates", 0) or 0) for row in hybrid_rows.values()) if hybrid_rows else 0.0,
    }
    result["confirmed"] = {
        "ppo_only_mean": mean(float(row.get("confirmed", 0) or 0) for row in ppo_rows.values()) if ppo_rows else 0.0,
        "hybrid_mean": mean(float(row.get("confirmation_attribution", {}).get("unique_confirmed", 0) or 0) for row in hybrid_rows.values()) if hybrid_rows else 0.0,
    }
    return result


def aggregate_efficiency(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {str(seed): row.get("efficiency", {}) for seed, row in rows.items()}


def classify_seed(row: Mapping[str, Any], coverage: Mapping[str, Any]) -> str:
    hard = row.get("hard_invariants", {})
    if not hard.get("all_pass") or hard.get("ground_truth_used") or hard.get("zero_step_switch_loop"):
        return "OTHER"
    if float(coverage.get("state_coverage_retention", 1.0) or 0.0) < 0.5 and int(row.get("dqn_steps", 0) or 0) > int(row.get("ppo_steps", 0) or 0):
        return "COVERAGE_REGRESSION"
    if float(row.get("pending_confirmation_churn_rate", 0.0) or 0.0) >= 0.5 and int(row.get("pending_handoff_count", 0) or 0) >= 3:
        return "PENDING_CHURN"
    if float(row.get("unproductive_handoff_rate", 0.0) or 0.0) >= 0.7 and int(row.get("local_session_count", 0) or 0) >= 3:
        return "HANDOFF_OVERUSE"
    if int(row.get("dqn_steps", 0) or 0) > 0 and int(row.get("productive_sessions", 0) or 0) == 0:
        return "DQN_LOW_CONTRIBUTION"
    return "HEALTHY_HYBRID"


def classify_overall(rows: Mapping[int, Mapping[str, Any]], coverage: Mapping[int, Mapping[str, Any]], classifications: Mapping[int, str]) -> str:
    if all(value == "HEALTHY_HYBRID" for value in classifications.values()):
        return "HYBRID_BEHAVIOR_PASS"
    counts = Counter(classifications.values())
    if counts.get("PENDING_CHURN", 0):
        return "PENDING_CONFIRMATION_CHURN"
    if counts.get("HANDOFF_OVERUSE", 0):
        return "HANDOFF_OVERUSE"
    if counts.get("COVERAGE_REGRESSION", 0):
        return "COVERAGE_REGRESSION"
    if counts.get("DQN_LOW_CONTRIBUTION", 0):
        return "DQN_LOW_CONTRIBUTION"
    return "OTHER"


def classify_guard_seed(row: Mapping[str, Any]) -> str:
    guard = row.get("pending_handoff_guard", {}) if isinstance(row.get("pending_handoff_guard"), Mapping) else {}
    if not guard or int(guard.get("pending_handoff_suppressed_count", 0) or 0) == 0:
        return "NO_RUNTIME_SUPPRESSION"
    if int(guard.get("same_context_functional_bypass_blocked_count", 0) or 0) == 0 and int(guard.get("same_context_stalled_bypass_blocked_count", 0) or 0) == 0 and float(row.get("pending_confirmation_churn_rate", 0.0) or 0.0) >= 0.5:
        return "CHURN_REMAINS"
    if float(row.get("pending_confirmation_churn_rate", 0.0) or 0.0) >= 0.5:
        return "CHURN_REMAINS"
    if int(row.get("confirmation_attribution", {}).get("ppo_candidate_dqn_confirmed_count", 0) or 0) == 0:
        return "OVER_SUPPRESSION"
    return "PENDING_CHURN_FIXED"


def classify_guard_overall(classifications: Mapping[int, str]) -> str:
    counts = Counter(classifications.values())
    if counts.get("CHURN_REMAINS", 0):
        return "CHURN_REMAINS"
    if counts.get("OVER_SUPPRESSION", 0):
        return "OVER_SUPPRESSION"
    if counts.get("NO_RUNTIME_SUPPRESSION", 0):
        return "REVIEW"
    return "PENDING_CHURN_FIXED"


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
        return float(numerator) / denom if denom else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/sanity")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--hybrid-template", default="hybrid_v1_behavior_seed{seed}.json")
    parser.add_argument("--ppo-template", default="ppo_only_behavior_seed{seed}.json")
    parser.add_argument("--output", default="artifacts/sanity/hybrid_v1_behavior_multiseed.json")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    hybrid = {seed: _load_json(artifact_dir / args.hybrid_template.format(seed=seed)) for seed in seeds}
    ppo = {seed: _load_json(artifact_dir / args.ppo_template.format(seed=seed)) for seed in seeds}
    summary = aggregate_validation(hybrid, ppo)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
