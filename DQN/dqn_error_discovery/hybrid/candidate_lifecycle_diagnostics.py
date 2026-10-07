"""Offline lifecycle diagnostics for Hybrid pending-confirmation handoffs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from dqn_error_discovery.hybrid.hybrid_controller import HybridController, ReturnReason


@dataclass(frozen=True)
class KeyMatch:
    exact: bool
    related: bool


def return_priority_order() -> list[str]:
    return [item.value for item in HybridController.return_priority]


def novelty_exhausted_definition() -> str:
    return "session.novelty_miss_streak >= novelty_patience; local novelty is any positive new_local_state, new_local_action_target, candidate_anomaly, confirmation_upgrade, or confirmed_anomaly"


def confirmed_return_definition() -> str:
    return "ReturnReason.CONFIRMED resolves a trigger_candidate_key using canonical confirmation event/tracker keys, with entry_candidate_keys anomaly exact-key matching only as generic fallback"


def analyze_candidate_lifecycle(summary: Mapping[str, Any]) -> dict[str, Any]:
    handoffs = [dict(row) for row in summary.get("handoff_events", []) or [] if isinstance(row, Mapping)]
    returns = [dict(row) for row in summary.get("return_events", []) or [] if isinstance(row, Mapping)]
    timeline = [dict(row) for row in summary.get("timeline", []) or [] if isinstance(row, Mapping)]
    confirmation = summary.get("confirmation_attribution", {}) if isinstance(summary.get("confirmation_attribution"), Mapping) else {}
    records = [dict(row) for row in confirmation.get("confirmation_records", []) or [] if isinstance(row, Mapping)]
    guard = summary.get("pending_handoff_guard", {}) if isinstance(summary.get("pending_handoff_guard"), Mapping) else {}

    rows_by_session = _rows_by_session(timeline)
    return_by_session = {int(row.get("local_session_id", 0) or 0): row for row in returns}
    records_by_session = _records_by_session(records)
    pending_sessions = [row for row in handoffs if row.get("reason") == "PENDING_CONFIRMATION"]
    pending_session_ids = [int(row.get("local_session_id", 0) or 0) for row in pending_sessions]
    session_audits = [
        _session_audit(handoff, return_by_session.get(int(handoff.get("local_session_id", 0) or 0), {}), rows_by_session.get(int(handoff.get("local_session_id", 0) or 0), []), records_by_session.get(int(handoff.get("local_session_id", 0) or 0), []))
        for handoff in pending_sessions
    ]

    pending_confirmed = [row for row in session_audits if row["any_confirmation_event"]]
    trigger_confirmed = [row for row in session_audits if row["trigger_candidate_confirmed_related"]]
    other_confirmed = [row for row in session_audits if row["other_candidate_confirmed"]]
    trigger_non_confirmed_return = [row for row in trigger_confirmed if row["return_reason"] != ReturnReason.CONFIRMED.value]
    max_steps = int(summary.get("configuration", {}).get("max_steps", 0) or 0)
    cooldown = cooldown_only_rehandoffs(session_audits, max_steps=max_steps)
    stale = stale_pending_sessions(session_audits, max_steps=max_steps)
    rotation = pending_candidate_rotation(session_audits)
    confirmed_rehandoffs = confirmed_candidate_rehandoffs(handoffs, records)
    top_rehandoff = top_candidate_timelines(session_audits, limit=5)
    confirmed_timelines = confirmed_candidate_timelines(session_audits, records, limit=3)
    override_audit = override_signal_audit(session_audits, guard)
    suppression = suppression_denominator(guard)
    return_reasons = Counter(str(row.get("reason") or "") for row in returns)
    result = {
        "return_priority_order": return_priority_order(),
        "return_priority_matches_design": return_priority_order()
        == ["EPISODE_DONE", "CONFIRMED", "CONTEXT_CHANGED", "REPEAT_GUARD", "NOVELTY_EXHAUSTED", "LOCAL_BUDGET"],
        "novelty_exhausted_definition": novelty_exhausted_definition(),
        "pending_confirmed_metric_definition": "session-level: any confirmation record attributed to the DQN local_session_id, not necessarily the trigger candidate",
        "confirmed_return_definition": confirmed_return_definition(),
        "metric_scope_table": {
            "pending_confirmed_metric": "any candidate confirmed during the DQN session",
            "CONFIRMED_return_condition": "trigger candidate canonical key confirmed by event/tracker state; generic sessions retain entry anomaly fallback",
            "confirmation_reward": "logical confirmation event key from DQN reward/ConfirmationTracker",
            "tracker_progress": "confirmation_metadata active pending progress/attempt fields",
            "candidate_handoff_guard": "trigger_candidate_key from active pending key plus context/resolvability snapshots",
        },
        "logical_key_pipeline": {
            "detector_anomaly": "logical_key/signature, often R05|layout-overlap|...|target ids",
            "confirmation_tracker": "pending_reproduction_records logical_key/action_target",
            "handoff_eligibility": "confirmation_metadata.active_pending_key or pending record logical_key",
            "trigger_candidate_key": "stored on LocalSession at PENDING_CONFIRMATION handoff",
            "newly_confirmed_event": "reward_history last_confirmation_transition_events logical_key",
            "return_confirmed": "confirmed anomaly key in entry_candidate_keys, exact",
            "pending_guard_state": "trigger_candidate_key",
        },
        "pending_sessions": len(pending_sessions),
        "pending_confirmed_sessions": len(pending_confirmed),
        "trigger_candidate_confirmed_sessions_exact": sum(1 for row in session_audits if row["trigger_candidate_confirmed_exact"]),
        "trigger_candidate_confirmed_sessions_related": len(trigger_confirmed),
        "other_candidate_confirmed_sessions": len(other_confirmed),
        "confirmed_return_count": int(return_reasons.get(ReturnReason.CONFIRMED.value, 0)),
        "novelty_return_count": int(return_reasons.get(ReturnReason.NOVELTY_EXHAUSTED.value, 0)),
        "trigger_confirmed_but_return_not_confirmed_count": len(trigger_non_confirmed_return),
        "trigger_confirmed_but_return_not_confirmed": trigger_non_confirmed_return,
        "override_audit": override_audit,
        "cooldown_only_rehandoff": cooldown,
        "stale_pending": stale,
        "pending_candidate_rotation_count": rotation["count"],
        "pending_candidate_rotation_examples": rotation["examples"],
        "top_rehandoff_candidate_timelines": top_rehandoff,
        "confirmed_candidate_timelines": confirmed_timelines,
        "confirmed_candidate_rehandoff_count": confirmed_rehandoffs["count"],
        "exact_confirmed_candidate_rehandoff_count": confirmed_rehandoffs["exact_count"],
        "related_confirmed_candidate_rehandoff_count": confirmed_rehandoffs["related_count"],
        "confirmed_candidate_rehandoff_examples": confirmed_rehandoffs["examples"],
        "tracker_pending_removal_normal": confirmed_rehandoffs["count"] == 0,
        "cross_episode_contamination_count": cross_episode_contamination(timeline, session_audits),
        "suppression_denominator": suppression,
        "churn_metric_definition": "session-level diagnostic: a pending-confirmation session re-enters with a logical key already seen unresolved in an earlier pending handoff",
        "same_candidate_rehandoff_metric_definition": "session-pair diagnostic: each handoff whose candidate/pending key overlaps any previous handoff is counted once",
        "metric_inflation": {
            "suppression_is_candidate_evaluation_event": True,
            "suppression_can_count_multiple_reasons_per_ppo_step": True,
            "same_candidate_rehandoff_is_not_unique_candidate_count": True,
        },
        "handoff_increase_attribution": handoff_increase_attribution(summary, session_audits),
        "session_audits": session_audits,
        "root_cause_verdicts": verdicts(session_audits, guard, return_reasons, suppression, rotation, confirmed_rehandoffs),
    }
    result["root_cause"] = root_cause(result)
    return result


def _session_audit(
    handoff: Mapping[str, Any],
    return_event: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    trigger_key = str(handoff.get("trigger_candidate_key") or "")
    confirmation_keys = [str(row.get("logical_key") or "") for row in records if str(row.get("logical_key") or "")]
    exact = [key for key in confirmation_keys if key == trigger_key]
    related = [key for key in confirmation_keys if key_match(trigger_key, key).related]
    other = [key for key in confirmation_keys if not key_match(trigger_key, key).related]
    component_positive = _component_positive(rows)
    return {
        "local_session_id": int(handoff.get("local_session_id", 0) or 0),
        "handoff_step": int(handoff.get("step", 0) or 0),
        "trigger_candidate_key": trigger_key,
        "trigger_candidate_rule_id": _rule_id(trigger_key) or _rule_from_context(str(handoff.get("entry_context_fingerprint") or "")),
        "trigger_candidate_status_at_entry": "pending",
        "trigger_candidate_confirmation_mode": _mode_from_context(str(handoff.get("entry_context_fingerprint") or "")),
        "trigger_candidate_progress_at_entry": int(handoff.get("entry_confirmation_progress", 0) or 0),
        "trigger_candidate_resolvable_at_entry": bool(handoff.get("entry_target_resolvable")),
        "entry_candidate_keys": list(handoff.get("entry_candidate_keys", []) or []),
        "entry_pending_confirmation_keys": list(handoff.get("entry_pending_confirmation_keys", []) or []),
        "entry_context_fingerprint": str(handoff.get("entry_context_fingerprint") or ""),
        "entry_reproduction_requirement_fingerprint": str(handoff.get("entry_reproduction_requirement_fingerprint") or ""),
        "eligibility_reason": str(handoff.get("eligibility_reason") or ""),
        "eligibility_reasons": list(handoff.get("eligibility_reasons", []) or []),
        "return_reason": str(return_event.get("reason") or ""),
        "return_step": int(return_event.get("step", 0) or 0),
        "local_steps": int(return_event.get("local_steps", len(rows)) or len(rows)),
        "return_session_productive": bool(return_event.get("session_productive")),
        "return_session_confirmation_progress": bool(return_event.get("session_confirmation_progress")),
        "any_confirmation_event": bool(confirmation_keys),
        "confirmation_event_keys": confirmation_keys,
        "trigger_candidate_confirmed_exact": bool(exact),
        "trigger_candidate_confirmed_related": bool(related),
        "trigger_candidate_confirmed_keys": related,
        "other_candidate_confirmed": bool(other),
        "other_candidate_confirmed_keys": other,
        "new_candidate_count": sum(len(row.get("new_candidate_keys", []) or []) for row in rows),
        "candidate_count_at_exit": int(rows[-1].get("candidate_count", 0) or 0) if rows else 0,
        "pending_count_at_exit": int(rows[-1].get("pending_count", 0) or 0) if rows else 0,
        "context_fingerprint_changed_at_return": bool(
            return_event.get("context_fingerprint")
            and str(return_event.get("context_fingerprint")) != str(handoff.get("entry_context_fingerprint") or "")
        ),
        "component_progress": component_positive,
        "step_trace": _step_trace(trigger_key, rows, confirmation_keys),
    }


def key_match(left: str, right: str) -> KeyMatch:
    if not left or not right:
        return KeyMatch(False, False)
    exact = left == right
    if exact:
        return KeyMatch(True, True)
    left_rule = _rule_id(left)
    right_rule = _rule_id(right)
    left_targets = _target_tokens(left)
    right_targets = _target_tokens(right)
    related = bool(left_rule and right_rule and left_rule == right_rule and left_targets.intersection(right_targets))
    if not related and not left_rule and not right_rule and left_targets and right_targets:
        related = bool(left_targets.intersection(right_targets))
    return KeyMatch(False, related)


def cooldown_only_rehandoffs(session_audits: list[Mapping[str, Any]], *, max_steps: int = 0) -> dict[str, Any]:
    previous: dict[str, Mapping[str, Any]] = {}
    count = productive = confirmed = 0
    examples: list[dict[str, Any]] = []
    for row in session_audits:
        key = str(row.get("trigger_candidate_key") or "")
        if not key:
            continue
        prior = previous.get(key)
        if prior:
            meaningful_eligibility = _meaningful_eligibility(row)
            same_episode = _same_episode(row, prior, max_steps=max_steps)
            same_context = row.get("entry_context_fingerprint") == prior.get("entry_context_fingerprint")
            same_progress = row.get("trigger_candidate_progress_at_entry") == prior.get("trigger_candidate_progress_at_entry")
            same_resolvable = row.get("trigger_candidate_resolvable_at_entry") == prior.get("trigger_candidate_resolvable_at_entry")
            if same_episode and same_context and same_progress and same_resolvable and not meaningful_eligibility:
                count += 1
                productive += int(bool(row.get("return_session_productive") or row.get("component_progress")))
                confirmed += int(bool(row.get("any_confirmation_event")))
                if len(examples) < 10:
                    examples.append({"logical_key": key, "previous_session": prior.get("local_session_id"), "session": row.get("local_session_id"), "return_reason": row.get("return_reason"), "confirmed": row.get("any_confirmation_event")})
        previous[key] = row
    return {
        "count": count,
        "productive_count": productive,
        "confirmed_count": confirmed,
        "productive_rate": _ratio(productive, count),
        "confirmation_rate": _ratio(confirmed, count),
        "examples": examples,
    }


def stale_pending_sessions(session_audits: list[Mapping[str, Any]], *, max_steps: int = 0) -> dict[str, Any]:
    previous: dict[str, Mapping[str, Any]] = {}
    stale_keys: set[str] = set()
    stale_sessions: list[dict[str, Any]] = []
    for row in session_audits:
        key = str(row.get("trigger_candidate_key") or "")
        prior = previous.get(key)
        if key and prior and not _meaningful_eligibility(row) and _same_episode(row, prior, max_steps=max_steps) and not prior.get("return_session_productive") and row.get("entry_context_fingerprint") == prior.get("entry_context_fingerprint") and row.get("trigger_candidate_progress_at_entry") == prior.get("trigger_candidate_progress_at_entry") and row.get("trigger_candidate_resolvable_at_entry") == prior.get("trigger_candidate_resolvable_at_entry"):
            stale_keys.add(key)
            stale_sessions.append({"logical_key": key, "session": row.get("local_session_id"), "previous_session": prior.get("local_session_id"), "return_reason": row.get("return_reason")})
        if key:
            previous[key] = row
    return {"stale_pending_candidates": len(stale_keys), "stale_pending_rehandoffs": len(stale_sessions), "stale_pending_sessions": stale_sessions}


def _meaningful_eligibility(row: Mapping[str, Any]) -> bool:
    allowed = {"PROGRESS_CHANGED", "CONTEXT_CHANGED", "TARGET_BECAME_RESOLVABLE", "REPRODUCTION_REQUIREMENT_CHANGED"}
    reason = str(row.get("eligibility_reason") or "")
    if reason in allowed:
        return True
    return bool(allowed.intersection(str(item) for item in row.get("eligibility_reasons", []) or []))


def pending_candidate_rotation(session_audits: list[Mapping[str, Any]]) -> dict[str, Any]:
    sequence = [str(row.get("trigger_candidate_key") or "") for row in session_audits if row.get("trigger_candidate_key")]
    count = 0
    examples: list[list[str]] = []
    for index, key in enumerate(sequence):
        if key in sequence[:index] and sequence[index - 1] != key:
            count += 1
            if len(examples) < 5:
                examples.append(sequence[max(0, index - 4) : index + 1])
    return {"count": count, "examples": examples}


def _same_episode(left: Mapping[str, Any], right: Mapping[str, Any], *, max_steps: int = 0) -> bool:
    if max_steps <= 0:
        return True
    left_episode = (int(left.get("handoff_step", 0) or 0) - 1) // max_steps
    right_episode = (int(right.get("handoff_step", 0) or 0) - 1) // max_steps
    return left_episode == right_episode


def confirmed_candidate_rehandoffs(handoffs: list[Mapping[str, Any]], records: list[Mapping[str, Any]]) -> dict[str, Any]:
    confirmed_steps = [(str(record.get("logical_key") or ""), int(record.get("local_session_id", 0) or 0)) for record in records]
    exact_count = 0
    related_count = 0
    examples: list[dict[str, Any]] = []
    for confirmed_key, confirmed_sid in confirmed_steps:
        for handoff in handoffs:
            sid = int(handoff.get("local_session_id", 0) or 0)
            if sid <= confirmed_sid:
                continue
            keys = [str(handoff.get("trigger_candidate_key") or "")] + [str(key) for key in handoff.get("entry_candidate_keys", []) or []] + [str(key) for key in handoff.get("entry_pending_confirmation_keys", []) or []]
            exact = confirmed_key in keys
            related = any(key_match(key, confirmed_key).related for key in keys)
            if exact or related:
                exact_count += int(exact)
                related_count += int(related and not exact)
                if len(examples) < 10:
                    examples.append({"confirmed_key": confirmed_key, "confirmed_session": confirmed_sid, "rehandoff_session": sid, "reason": handoff.get("reason"), "exact": exact, "related": related})
                break
    return {"count": exact_count + related_count, "exact_count": exact_count, "related_count": related_count, "examples": examples}


def override_signal_audit(session_audits: list[Mapping[str, Any]], guard: Mapping[str, Any]) -> dict[str, Any]:
    progress_changes = context_changes = resolvability_changes = 0
    previous: dict[str, Mapping[str, Any]] = {}
    for row in session_audits:
        key = str(row.get("trigger_candidate_key") or "")
        prior = previous.get(key)
        if key and prior:
            progress_changes += int(row.get("trigger_candidate_progress_at_entry") != prior.get("trigger_candidate_progress_at_entry"))
            context_changes += int(row.get("entry_context_fingerprint") != prior.get("entry_context_fingerprint"))
            resolvability_changes += int(row.get("trigger_candidate_resolvable_at_entry") != prior.get("trigger_candidate_resolvable_at_entry"))
        if key:
            previous[key] = row
    progress_override = int(guard.get("rehandoff_due_to_progress", 0) or 0)
    context_override = int(guard.get("rehandoff_due_to_context_change", 0) or 0)
    resolvability_override = int(guard.get("rehandoff_due_to_target_resolvable", 0) or 0)
    return {
        "actual_progress_changes": progress_changes,
        "progress_overrides": progress_override,
        "missed_progress_override": max(0, progress_changes - progress_override),
        "actual_context_changes": context_changes,
        "context_overrides": context_override,
        "missed_context_override": max(0, context_changes - context_override),
        "actual_resolvability_changes": resolvability_changes,
        "resolvability_overrides": resolvability_override,
        "missed_resolvability_override": max(0, resolvability_changes - resolvability_override),
    }


def suppression_denominator(guard: Mapping[str, Any]) -> dict[str, Any]:
    events = [row for row in guard.get("suppression_events", []) or [] if isinstance(row, Mapping)]
    unique_step_key = {(row.get("step"), row.get("logical_key")) for row in events}
    return {
        "suppression_count": len(events),
        "unique_candidate_step_suppressions": len(unique_step_key),
        "unique_suppressed_candidates": len({row.get("logical_key") for row in events}),
        "definition": "candidate-evaluation diagnostic events; bypass checks can add additional suppression events for the same candidate/PPO step",
    }


def handoff_increase_attribution(summary: Mapping[str, Any], session_audits: list[Mapping[str, Any]]) -> dict[str, Any]:
    handoffs = [row for row in summary.get("handoff_events", []) or [] if isinstance(row, Mapping)]
    reasons = Counter(str(row.get("reason") or "") for row in handoffs)
    cooldown = cooldown_only_rehandoffs(session_audits)
    rotation = pending_candidate_rotation(session_audits)
    return {
        "new_candidate_handoffs": int(reasons.get("NEW_CANDIDATE", 0)),
        "pending_handoffs": int(reasons.get("PENDING_CONFIRMATION", 0)),
        "cooldown_only_rehandoffs": cooldown["count"],
        "pending_candidate_rotation_count": rotation["count"],
        "longer_or_shorter_episode_effect": int(summary.get("episode_summary", {}).get("total_steps", 0) or 0),
    }


def top_candidate_timelines(session_audits: list[Mapping[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in session_audits:
        key = str(row.get("trigger_candidate_key") or "")
        if key:
            grouped[key].append(row)
    top = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:limit]
    return [
        {
            "logical_key": key,
            "handoff_count": len(rows),
            "timeline": [
                {
                    "session": row.get("local_session_id"),
                    "handoff_step": row.get("handoff_step"),
                    "return_reason": row.get("return_reason"),
                    "confirmed": row.get("any_confirmation_event"),
                    "trigger_confirmed": row.get("trigger_candidate_confirmed_related"),
                    "productive": row.get("return_session_productive"),
                }
                for row in rows
            ],
        }
        for key, rows in top
    ]


def confirmed_candidate_timelines(session_audits: list[Mapping[str, Any]], records: list[Mapping[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records[:limit]:
        sid = int(record.get("local_session_id", 0) or 0)
        session = next((row for row in session_audits if int(row.get("local_session_id", 0) or 0) == sid), {})
        result.append(
            {
                "logical_key": record.get("logical_key"),
                "discovered_by": record.get("discovered_by"),
                "confirmed_by": record.get("confirmed_by"),
                "confirmation_session": sid,
                "trigger_candidate_key": session.get("trigger_candidate_key"),
                "return_reason": session.get("return_reason"),
                "trigger_candidate_confirmed_related": session.get("trigger_candidate_confirmed_related"),
            }
        )
    return result


def cross_episode_contamination(timeline: list[Mapping[str, Any]], session_audits: list[Mapping[str, Any]]) -> int:
    episode_by_global_step = {
        (int(row.get("episode", 0) or 0) - 1) * 60 + int(row.get("step", 0) or 0): int(row.get("episode", 0) or 0)
        for row in timeline
    }
    seen_by_episode: dict[int, set[str]] = defaultdict(set)
    contamination = 0
    for row in session_audits:
        episode = episode_by_global_step.get(int(row.get("handoff_step", 0) or 0), 0)
        key = str(row.get("trigger_candidate_key") or "")
        if episode and key:
            other_seen = set().union(*(keys for ep, keys in seen_by_episode.items() if ep != episode)) if seen_by_episode else set()
            contamination += int(key in other_seen and not seen_by_episode[episode])
            seen_by_episode[episode].add(key)
    return contamination


def verdicts(session_audits: list[Mapping[str, Any]], guard: Mapping[str, Any], return_reasons: Counter[str], suppression: Mapping[str, Any], rotation: Mapping[str, Any], confirmed_rehandoffs: Mapping[str, Any]) -> dict[str, str]:
    pending_confirmed = sum(1 for row in session_audits if row.get("any_confirmation_event"))
    trigger_confirmed = sum(1 for row in session_audits if row.get("trigger_candidate_confirmed_related"))
    confirmed_returns = int(return_reasons.get(ReturnReason.CONFIRMED.value, 0))
    override = override_signal_audit(session_audits, guard)
    stale = stale_pending_sessions(session_audits)
    return {
        "A_CONFIRMED_RETURN_PROPAGATION_BUG": "PARTIAL" if trigger_confirmed and confirmed_returns == 0 else "REJECT",
        "B_METRIC_SCOPE_MISMATCH": "PASS" if pending_confirmed > trigger_confirmed else "REJECT",
        "C_STALE_PENDING_REHANDOFF": "PASS" if stale["stale_pending_rehandoffs"] > 0 else "REJECT",
        "D_CANDIDATE_ROTATION_CHURN": "PASS" if int(rotation.get("count", 0) or 0) > 0 else "REJECT",
        "E_OVERRIDE_SIGNAL_NOT_WIRED": "PASS" if any(override[key] > 0 for key in ("missed_progress_override", "missed_context_override", "missed_resolvability_override")) else "REJECT",
        "F_OVERRIDE_SIGNAL_ABSENT": "PASS" if override["actual_progress_changes"] == 0 and override["actual_context_changes"] == 0 and override["actual_resolvability_changes"] == 0 else "PARTIAL",
        "G_METRIC_INFLATION": "PASS" if int(suppression.get("suppression_count", 0) or 0) > int(suppression.get("unique_candidate_step_suppressions", 0) or 0) else "PARTIAL",
        "confirmed_candidate_rehandoff": "PASS" if int(confirmed_rehandoffs.get("count", 0) or 0) > 0 else "REJECT",
    }


def root_cause(result: Mapping[str, Any]) -> list[str]:
    verdict = result.get("root_cause_verdicts", {})
    causes = [name for name, value in verdict.items() if value == "PASS"]
    return causes or [name for name, value in verdict.items() if value == "PARTIAL"]


def _rows_by_session(timeline: list[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in timeline:
        sid = int(row.get("local_session_id", 0) or 0)
        if row.get("active_agent") == "DQN_LOCAL" and sid:
            grouped[sid].append(row)
    return grouped


def _records_by_session(records: list[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row.get("local_session_id", 0) or 0)].append(row)
    return grouped


def _component_positive(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        components = row.get("reward_components", {}) if isinstance(row.get("reward_components"), Mapping) else {}
        if any(float(components.get(key, 0.0) or 0.0) > 0.0 for key in ("new_local_state", "new_local_action_target", "candidate_anomaly", "confirmation_upgrade", "confirmed_anomaly")):
            return True
    return False


def _step_trace(trigger_key: str, rows: list[Mapping[str, Any]], confirmation_keys: list[str]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        newly = [key for key in row.get("newly_confirmed", []) or [] if key in confirmation_keys]
        trace.append(
            {
                "local_step": index,
                "trigger_candidate_key": trigger_key,
                "action_type": row.get("action_type"),
                "reward": row.get("reward"),
                "pending_count_after": row.get("pending_count"),
                "confirmation_event_keys": newly,
                "confirmation_match_class": confirmation_match_class(trigger_key, newly),
            }
        )
    return trace


def confirmation_match_class(trigger_key: str, confirmation_keys: Iterable[str]) -> str:
    keys = list(confirmation_keys)
    if any(key == trigger_key for key in keys):
        return "TRIGGER_CANDIDATE_CONFIRMED"
    if any(key_match(trigger_key, key).related for key in keys):
        return "TRIGGER_CANDIDATE_CONFIRMED_RELATED"
    if keys:
        return "OTHER_CANDIDATE_CONFIRMED"
    return "NO_CONFIRMATION"


def _rule_id(key: str) -> str:
    first = str(key or "").split("|", 1)[0]
    return first if len(first) == 3 and first.startswith("R") else ""


def _rule_from_context(context: str) -> str:
    for token in str(context or "").split("|"):
        if len(token) == 3 and token.startswith("R"):
            return token
    return ""


def _mode_from_context(context: str) -> str:
    if "ACTION_REPRODUCTION" in context:
        return "ACTION_REPRODUCTION"
    if "OBSERVATION_REPRODUCTION" in context:
        return "OBSERVATION_REPRODUCTION"
    return ""


def _target_tokens(key: str) -> set[str]:
    tokens = set()
    parts = str(key or "").replace("target:", "|").replace("bid:", "|").replace("click_element:", "|").split("|")
    for part in parts:
        text = part.strip()
        if text and (text.isdigit() or text.startswith("bid:")):
            tokens.add(text.removeprefix("bid:"))
    return tokens


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
        return float(numerator) / denom if denom else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
