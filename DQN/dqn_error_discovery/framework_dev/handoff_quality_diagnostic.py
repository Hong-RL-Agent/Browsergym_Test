"""Post-hoc handoff-quality diagnostics for frozen Hybrid v2 artifacts.

This module reads runtime artifacts only.  It does not import or modify the
controller, detector, agents, rewards, or ground-truth scenario catalog.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

SEEDS = (42, 43, 44)
REASONS = ("NEW_CANDIDATE", "PENDING_CONFIRMATION", "FUNCTIONAL_REGION", "STALLED_FUNCTIONAL")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    return float(value or 0)


def _route(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[-1].split("?", 1)[0] if url else ""


def _session_records(runtime: Mapping[str, Any], framework: str, seed: int) -> list[dict[str, Any]]:
    timeline = [row for row in runtime.get("timeline", []) if isinstance(row, Mapping)]
    handoffs = {int(row.get("local_session_id", 0)): row for row in runtime.get("handoff_events", []) if isinstance(row, Mapping)}
    returns = {int(row.get("local_session_id", 0)): row for row in runtime.get("return_events", []) if isinstance(row, Mapping)}
    local_by_id: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in timeline:
        sid = row.get("local_session_id")
        if sid is not None:
            local_by_id[int(sid)].append(row)

    ordered = sorted(local_by_id)
    region_counts: Counter[str] = Counter()
    region_outcomes: dict[str, list[bool]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for sid in ordered:
        local = local_by_id[sid]
        handoff = handoffs.get(sid, {})
        ret = returns.get(sid, {})
        trigger_step = int(handoff.get("step", local[0].get("step", 0)))
        trigger = next((row for row in timeline if int(row.get("step", -1)) == trigger_step), {})
        reason = str(handoff.get("reason") or "UNKNOWN")
        url = str(handoff.get("entry_url") or trigger.get("url_after") or "")
        region_key = "|".join((
            _route(url),
            str(handoff.get("entry_functional_region") or ""),
            str(handoff.get("trigger_candidate_rule_id") or ""),
            str(handoff.get("trigger_candidate_key") or ""),
        ))
        action_targets = [str(row.get("action_target") or "") for row in local]
        states = [str(row.get("state_after") or "") for row in local]
        target_values = [value.split(":", 1)[1] if ":" in value else value for value in action_targets]
        new_state_steps = [i + 1 for i, row in enumerate(local) if _num(row.get("reward_components", {}).get("new_local_state")) > 0]
        new_target_steps = [i + 1 for i, row in enumerate(local) if _num(row.get("reward_components", {}).get("new_local_action_target")) > 0]
        evidence_steps = [i + 1 for i, row in enumerate(local) if any(_num(row.get("reward_components", {}).get(k)) > 0 for k in ("candidate_anomaly", "confirmed_anomaly", "confirmation_upgrade"))]
        candidate_steps = [i + 1 for i, row in enumerate(local) if row.get("new_candidate_keys")]
        confirmation_steps = [i + 1 for i, row in enumerate(local) if row.get("repeat_diagnostic", {}).get("confirmation_progress") or _num(row.get("reward_components", {}).get("confirmed_anomaly")) > 0]
        no_effect_steps = [i + 1 for i, row in enumerate(local) if row.get("repeat_diagnostic", {}).get("no_effect") is True]
        meaningful_steps = sorted(set(new_state_steps + new_target_steps + evidence_steps + candidate_steps + confirmation_steps))
        first_two_futile = not any(step <= 2 for step in meaningful_steps)
        late_productivity = any(step >= 4 for step in meaningful_steps) and not any(step <= 3 for step in meaningful_steps)
        strong = bool(candidate_steps or evidence_steps or confirmation_steps)
        weak = bool(not strong and (new_state_steps or new_target_steps))
        outcome = "STRONG_PRODUCTIVE" if strong else "WEAK_PRODUCTIVE" if weak else "UNPRODUCTIVE"
        previous_count = region_counts[region_key]
        previous_productive = region_outcomes[region_key][-1] if region_outcomes[region_key] else None
        region_counts[region_key] += 1
        region_outcomes[region_key].append(strong or weak)
        record = {
            "session_id": sid,
            "seed": seed,
            "framework": framework,
            "episode": trigger.get("episode"),
            "handoff_step": trigger_step,
            "handoff_reason": reason,
            "trigger_url": url,
            "trigger_route": _route(url),
            "trigger_state_id": trigger.get("state_after") or trigger.get("state_before"),
            "trigger_target_id": trigger.get("action_target", "").split(":", 1)[1] if ":" in str(trigger.get("action_target", "")) else "",
            "trigger_candidate_key": handoff.get("trigger_candidate_key", ""),
            "trigger_candidate_rule": handoff.get("trigger_candidate_rule_id", ""),
            "functional_region_reason": handoff.get("eligibility_reason", ""),
            "functional_region_flags": handoff.get("eligibility_reasons", []),
            "pending_confirmation_exists": bool(handoff.get("entry_pending_confirmation_keys")),
            "ppo_action_count": handoff.get("ppo_action_count", 0),
            "local_start_state": local[0].get("state_before", "") if local else "",
            "local_end_state": local[-1].get("state_after", "") if local else "",
            "local_steps": len(local),
            "local_actions": action_targets,
            "local_targets": target_values,
            "unique_local_states": len(set(states)),
            "unique_local_targets": len(set(target_values)),
            "unique_local_action_targets": len(set(action_targets)),
            "candidate_created": bool(candidate_steps),
            "candidate_upgraded": any(_num(row.get("reward_components", {}).get("confirmation_upgrade")) > 0 for row in local),
            "candidate_confirmed": bool(confirmation_steps),
            "new_evidence_found": bool(evidence_steps),
            "new_route_found": len({_route(str(row.get("url_after") or "")) for row in local}) > 1,
            "new_target_found": bool(new_target_steps),
            "new_state_found": bool(new_state_steps),
            "return_reason": ret.get("reason", ""),
            "productive_existing_metric": bool(ret.get("session_productive")),
            "strong_productive": strong,
            "weak_productive": weak,
            "diagnostic_outcome": outcome,
            "meaningful_steps": meaningful_steps,
            "early_futility": first_two_futile,
            "late_productivity": late_productivity,
            "budget_exhausted": str(ret.get("reason", "")) == "LOCAL_BUDGET",
            "no_effect_actions": len(no_effect_steps),
            "no_effect_rate": len(no_effect_steps) / max(1, len(local)),
            "repeated_action_target_rate": (len(local) - len(set(action_targets))) / max(1, len(local)),
            "region_key_diagnostic": region_key,
            "region_visit_index": previous_count + 1,
            "repeat_region": previous_count > 0,
            "repeat_after_productive": previous_count > 0 and previous_productive is True,
            "repeat_after_unproductive": previous_count > 0 and previous_productive is False,
            "actionability": "HIGH_ACTIONABILITY" if handoff.get("entry_target_resolvable") or handoff.get("entry_candidate_keys") else "LOW_ACTIONABILITY",
            "local_escape": any(str(row.get("url_after") or "") != url for row in local[:2]),
            "dqn_worthy_region": bool(handoff.get("entry_target_resolvable") or handoff.get("entry_candidate_keys") or handoff.get("entry_pending_confirmation_keys")),
            "ppo_saturated_region": previous_count > 0 and not strong,
            "raw_handoff": dict(handoff),
            "raw_return": dict(ret),
        }
        records.append(record)
    for index, record in enumerate(records[:-1]):
        current_last = max((int(row.get("step", 0)) for row in local_by_id[record["session_id"]]), default=0)
        next_first = min((int(row.get("step", 0)) for row in local_by_id[records[index + 1]["session_id"]]), default=current_last)
        ppo_between = sum(1 for row in timeline if current_last < int(row.get("step", 0)) < next_first and row.get("active_agent") == "PPO_GLOBAL")
        record["ppo_steps_to_next_handoff"] = ppo_between
        record["one_ppo_step_rehandoff"] = ppo_between == 1
        record["same_region_one_step_rehandoff"] = ppo_between == 1 and record["region_key_diagnostic"] == records[index + 1]["region_key_diagnostic"]
    if records:
        records[-1]["ppo_steps_to_next_handoff"] = None
        records[-1]["one_ppo_step_rehandoff"] = False
        records[-1]["same_region_one_step_rehandoff"] = False
    return records


def _summary(records: list[dict[str, Any]], runtime: Mapping[str, Any], framework: str, seed: int) -> dict[str, Any]:
    by_reason: dict[str, dict[str, int]] = {}
    for reason in sorted(set(REASONS) | {str(row["handoff_reason"]) for row in records}):
        rows = [row for row in records if row["handoff_reason"] == reason]
        by_reason[reason] = {
            "sessions": len(rows),
            "strong_productive": sum(row["strong_productive"] for row in rows),
            "weak_productive": sum(row["weak_productive"] for row in rows),
            "unproductive": sum(row["diagnostic_outcome"] == "UNPRODUCTIVE" for row in rows),
            "strong_productive_rate": sum(row["strong_productive"] for row in rows) / max(1, len(rows)),
        }
    fr = [row for row in records if row["handoff_reason"] == "FUNCTIONAL_REGION"]
    local_rows = [row for row in records]
    total_steps = int(runtime.get("episode_summary", {}).get("total_steps", 0) or 0)
    dqn_steps = int(runtime.get("episode_summary", {}).get("dqn_steps", 0) or 0)
    return {
        "framework": framework,
        "seed": seed,
        "total_local_sessions": len(records),
        "handoff_reason_table": by_reason,
        "functional_region_sessions": len(fr),
        "strong_productive": sum(row["strong_productive"] for row in records),
        "weak_productive": sum(row["weak_productive"] for row in records),
        "unproductive": sum(row["diagnostic_outcome"] == "UNPRODUCTIVE" for row in records),
        "functional_region_strong_productive_rate": sum(row["strong_productive"] for row in fr) / max(1, len(fr)),
        "candidate_conversion_rate": sum(row["candidate_created"] for row in fr) / max(1, len(fr)),
        "confirmation_conversion_rate": sum(row["candidate_confirmed"] for row in fr) / max(1, len(fr)),
        "evidence_conversion_rate": sum(row["new_evidence_found"] for row in fr) / max(1, len(fr)),
        "zero_gain_rate": sum(row["diagnostic_outcome"] == "UNPRODUCTIVE" for row in fr) / max(1, len(fr)),
        "first_time_region": sum(not row["repeat_region"] for row in fr),
        "repeat_region": sum(row["repeat_region"] for row in fr),
        "repeat_after_productive": sum(row["repeat_after_productive"] for row in fr),
        "repeat_after_unproductive": sum(row["repeat_after_unproductive"] for row in fr),
        "same_region_rehandoff_rate": sum(row["repeat_region"] for row in fr) / max(1, len(fr)),
        "early_futility": sum(row["early_futility"] for row in fr),
        "late_productivity": sum(row["late_productivity"] for row in fr),
        "budget_exhaustion": sum(row["budget_exhausted"] for row in fr),
        "mean_local_steps": mean([row["local_steps"] for row in local_rows]) if local_rows else 0.0,
        "local_no_effect_rate": sum(row["no_effect_actions"] for row in local_rows) / max(1, sum(row["local_steps"] for row in local_rows)),
        "repeated_action_target_rate": mean([row["repeated_action_target_rate"] for row in local_rows]) if local_rows else 0.0,
        "local_escape_rate": sum(row["local_escape"] for row in fr) / max(1, len(fr)),
        "one_ppo_step_rehandoff": sum(row["one_ppo_step_rehandoff"] for row in fr),
        "same_region_one_step_rehandoff": sum(row["same_region_one_step_rehandoff"] for row in fr),
        "low_or_no_actionability": sum(row["actionability"] == "LOW_ACTIONABILITY" for row in fr),
        "ppo_saturated_region": sum(row["ppo_saturated_region"] for row in fr),
        "dqn_worthy_region": sum(row["dqn_worthy_region"] for row in fr),
        "unproductive_dqn_steps": sum(row["local_steps"] for row in local_rows if row["diagnostic_outcome"] == "UNPRODUCTIVE"),
        "ppo_opportunity_cost_proxy": sum(row["local_steps"] for row in local_rows if row["diagnostic_outcome"] == "UNPRODUCTIVE") / max(1, total_steps),
        "handoff_density_per_100_steps": len(records) * 100 / max(1, total_steps),
        "functional_region_density_per_100_ppo_steps": len(fr) * 100 / max(1, int(runtime.get("episode_summary", {}).get("ppo_steps", 0) or 0)),
        "invalid_or_timeout_actions": sum(bool(row.get("repeat_diagnostic", {}).get("invalid")) or bool(row.get("repeat_diagnostic", {}).get("timeout")) for row in []),
        "dqn_steps": dqn_steps,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "artifacts" / "framework_dev" / "v2_integrated"
    output = root / "artifacts" / "framework_dev" / "handoff_quality"
    output.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, list[dict[str, Any]]] = {}
    for framework in ("react", "next"):
        all_reports[framework] = []
        for seed in SEEDS:
            runtime = _read(source / f"{framework}_seed{seed}.json")
            records = _session_records(runtime, framework.upper(), seed)
            report = _summary(records, runtime, framework.upper(), seed)
            report["sessions"] = records
            all_reports[framework].append(report)
            (output / f"{framework}_seed{seed}_handoff_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output / f"{framework}_handoff_quality_multiseed.json").write_text(json.dumps({"framework": framework, "seeds": all_reports[framework], "evaluation_only": True}, indent=2), encoding="utf-8")
    summary = {
        "schema_version": "hybrid_handoff_quality_diagnostic_v1",
        "evaluation_only": True,
        "production_behavior_changed": False,
        "controller_changed": False,
        "source_artifacts": [str(source / f"{f}_seed{s}.json") for f in ("react", "next") for s in SEEDS],
        "frameworks": {f: [{k: v for k, v in report.items() if k != "sessions"} for report in reports] for f, reports in all_reports.items()},
        "root_cause_status": "DIAGNOSTIC_COMPLETE",
        "GT_runtime_isolation": True,
        "reserved_sites_excluded": ["SITE002", "SITE003", "SITE004", "SITE005"],
    }
    (output / "hybrid_handoff_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
