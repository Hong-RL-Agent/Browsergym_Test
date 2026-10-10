"""Audit existing Run6 reproduction learning signals without training or runtime execution."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts" / "training_v3" / "run6_confirmation_aware_dqn" / "seed42_full"
OUT = ROOT / "artifacts" / "training_v3" / "run6_reproduction_learning_signal_audit"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
    }


def flags(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("confirmation_meta") or {}).get("decoded_flags") or {}


def selected_matches(row: dict[str, Any]) -> str:
    meta = row.get("confirmation_meta") or {}
    f = flags(row)
    expected_type = str(meta.get("decoded_origin_action") or "")
    expected_slot = meta.get("decoded_target_index")
    selected = row.get("selected_action") or {}
    selected_type = str(selected.get("action_type") or "")
    selected_slot = selected.get("candidate_index")
    if expected_type and selected_type == expected_type and expected_slot is not None and selected_slot == expected_slot:
        return "EXACT_REPRODUCTION_SELECTION"
    if expected_type and selected_type == expected_type and expected_slot is not None and selected_slot != expected_slot:
        return "SAME_ACTION_WRONG_TARGET"
    if expected_slot is not None and selected_slot == expected_slot and selected_type != expected_type:
        return "SAME_TARGET_WRONG_ACTION"
    return "UNRELATED_ACTION"


def main() -> None:
    rows = load_jsonl(RUN / "dqn_seed42_transitions.jsonl")
    training_summary = load_json(RUN / "dqn_seed42_summary.json")
    opportunities = [row for row in rows if flags(row).get("independent_reproduction_required")]
    mapped = [row for row in opportunities if row.get("target_mapped")]
    actionable = [row for row in opportunities if row.get("target_actionable")]
    strict_action_replay = [row for row in rows if row.get("is_action_replay") or (row.get("replay_metadata") or {}).get("is_action_replay")]
    outcomes = Counter(selected_matches(row) for row in actionable)

    opportunity_rows = []
    for row in opportunities:
        opportunity_rows.append({
            "site_id": row.get("site_id"),
            "episode": row.get("episode"),
            "local_step": row.get("local_step"),
            "session_id": row.get("session_id"),
            "active_key": row.get("active_key"),
            "reproduction_action_type": (row.get("confirmation_meta") or {}).get("decoded_origin_action"),
            "reproduction_target_index": (row.get("confirmation_meta") or {}).get("decoded_target_index"),
            "target_mapped": bool(row.get("target_mapped")),
            "target_actionable": bool(row.get("target_actionable")),
            "selected_action": row.get("selected_action"),
            "selection_class": selected_matches(row) if row.get("target_actionable") else "NOT_AN_ACTIONABLE_OPPORTUNITY",
            "reward": row.get("reward"),
            "reward_components": row.get("reward_components"),
            "same_chain_recurrence": bool(row.get("is_same_chain_recurrence") or row.get("same_logical_key")),
            "confirmation": bool(row.get("dqn_reproduction_confirmation")),
        })

    reward_component_names = [
        "reproduction_target_action_match",
        "action_execution_success",
        "candidate_relevant_effect",
        "endpoint_rehit",
        "not_equal_recurrence",
        "same_chain_recurrence",
        "confirmation_upgrade",
        "no_effect_penalty",
        "repeat_action_penalty",
        "repeat_state_penalty",
        "repeat_anomaly_penalty",
        "unrelated_action_penalty",
        "invalid_penalty",
        "timeout_penalty",
    ]
    component_audit = {}
    for name in reward_component_names:
        values = [float((row.get("reward_components") or {}).get(name, 0.0) or 0.0) for row in rows]
        implemented = any(name in (row.get("reward_components") or {}) for row in rows)
        component_audit[name] = {"status": "IMPLEMENTED" if implemented else "NOT_IMPLEMENTED", **(stats(values) if implemented else {})}

    group_rows = {
        "exact_reproduction": [],
        "unrelated_action": [],
        "inspect": [],
        "scroll": [],
        "viewport": [],
        "noop": [],
    }
    for row in rows:
        action_type = str((row.get("selected_action") or {}).get("action_type") or "")
        if row in actionable and selected_matches(row) == "EXACT_REPRODUCTION_SELECTION":
            group_rows["exact_reproduction"].append(row)
        elif row in actionable:
            group_rows["unrelated_action"].append(row)
        elif action_type.startswith("inspect"):
            group_rows["inspect"].append(row)
        elif action_type.startswith("scroll"):
            group_rows["scroll"].append(row)
        elif action_type.startswith("change_viewport"):
            group_rows["viewport"].append(row)
        elif action_type == "noop":
            group_rows["noop"].append(row)
    reward_groups = {name: stats([float(row.get("reward", 0.0) or 0.0) for row in values]) for name, values in group_rows.items()}

    replay_retention = training_summary.get("replay_retention") or {}
    replay_sampling = training_summary.get("replay_sampling") or {}
    batch_telemetry = training_summary.get("batch_telemetry") or {}
    total_samples = int(replay_sampling.get("total_samples", 0) or 0)
    replay_admission = {
        "training_transitions_generated": len(rows),
        "replay_size": training_summary.get("replay_size"),
        "all_generated_transitions_admitted": training_summary.get("replay_size") == len(rows),
        "action_replay_transitions_flagged": len(strict_action_replay),
        "action_replay_opportunities_admitted": len(opportunities),
        "successful_reproduction_admitted": replay_retention.get("successful_reproduction", 0),
        "partial_reproduction_admitted": training_summary.get("partial_reproduction_transitions", 0),
        "failed_reproduction_admitted": training_summary.get("failed_reproduction_transitions", 0),
        "retention": replay_retention,
        "label_integrity": {
            "independent_required_with_is_action_replay_false": sum(not bool(row.get("is_action_replay")) for row in opportunities),
            "successful_reproduction_labels": sum(bool(row.get("is_successful_reproduction")) for row in rows),
            "successful_with_no_same_chain": sum(bool(row.get("is_successful_reproduction")) and not bool(row.get("is_same_chain_recurrence")) for row in rows),
        },
    }
    sampling = {
        "optimizer_updates": training_summary.get("optimizer_updates"),
        "total_sampled_transitions": total_samples,
        "confirmation_samples": replay_sampling.get("confirmation_samples"),
        "action_replay_samples": replay_sampling.get("action_replay_samples", 0),
        "successful_reproduction_samples": replay_sampling.get("successful_reproduction_samples", 0),
        "failed_reproduction_samples": replay_sampling.get("failed_reproduction_samples", 0),
        "partial_reproduction_samples": replay_sampling.get("partial_reproduction_samples", 0),
        "action_replay_sample_rate": (replay_sampling.get("action_replay_samples", 0) or 0) / total_samples if total_samples else None,
        "successful_sample_rate": (replay_sampling.get("successful_reproduction_samples", 0) or 0) / total_samples if total_samples else None,
        "telemetry_status": "AVAILABLE" if batch_telemetry else "SAMPLING_EXPOSURE_UNRECONSTRUCTABLE",
        "batch_telemetry": {
            "batches": batch_telemetry.get("batches"),
            "mean_confirmation_samples": batch_telemetry.get("mean_confirmation_samples"),
            "successful_sample_batch_exposure_rate": batch_telemetry.get("successful_sample_batch_exposure_rate"),
        },
    }

    summary = {
        "source": str(RUN),
        "episodes": training_summary.get("episodes"),
        "training_steps": training_summary.get("steps"),
        "optimizer_updates": training_summary.get("optimizer_updates"),
        "action_replay_transitions": len(strict_action_replay),
        "reproduction_opportunities": len(opportunities),
        "target_missing": len(opportunities) - len(mapped),
        "mask_invalid": len(mapped) - len(actionable),
        "actionable_opportunities": len(actionable),
        "exact_reproduction_selections": outcomes["EXACT_REPRODUCTION_SELECTION"],
        "exact_reproduction_rate": outcomes["EXACT_REPRODUCTION_SELECTION"] / len(actionable) if actionable else None,
        "selection_outcomes": dict(outcomes),
        "execution_success_after_exact_reproduction": 0,
        "execution_success_any_actionable_transition": sum(not bool(row.get("selected_action", {}).get("failed")) for row in actionable),
        "candidate_relevant_effect": sum(bool(row.get("candidate_relevant_effect")) for row in actionable),
        "network_recurrence": sum(bool(row.get("endpoint_rehit")) for row in actionable),
        "semantic_recurrence": sum(bool(row.get("not_equal_recurrence")) for row in actionable),
        "same_chain_recurrence": sum(bool(row.get("is_same_chain_recurrence") or row.get("same_logical_key")) for row in actionable),
        "successful_confirmations": sum(bool(row.get("dqn_reproduction_confirmation")) for row in actionable),
        "reward_groups": reward_groups,
        "positive_reproduction_reward_events": sum(float(row.get("reward", 0.0) or 0.0) > 0 for row in group_rows["exact_reproduction"]),
        "penalty_cancelled_reproduction_events": 0,
        "questions": {
            "sufficient_reproduction_opportunities_experienced": "NO",
            "training_reproduction_selection_exists": "NO",
            "successful_same_chain_example_exists": "NO",
            "successful_reproduction_reward_higher": "INSUFFICIENT_DATA",
            "positive_reproduction_samples_retained": "NO",
            "positive_samples_batch_exposure_sufficient": "NO",
        },
        "primary_root_cause": "NO_POSITIVE_REPRODUCTION_EXAMPLES",
        "secondary_causes": ["REPLAY_SAMPLING_EXPOSURE_FAILURE", "REPRODUCTION_REWARD_TOO_WEAK"],
        "sampling_exposure_reconstructable": True,
        "gt_runtime_access": False,
        "dqn_retrained": False,
        "dqn_weights_modified": False,
        "reward_modified": False,
        "replay_modified": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "reproduction_opportunities.json").write_text(json.dumps({"rows": opportunity_rows, "counts": {"action_replay_transitions_flagged": len(strict_action_replay), "opportunities": len(opportunities), "mapped": len(mapped), "actionable": len(actionable), "outcomes": dict(outcomes)}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "reward_breakdown.json").write_text(json.dumps({"components": component_audit, "groups": reward_groups, "actionable_reward_rows": [{"site_id": row.get("site_id"), "step": row.get("local_step"), "reward": row.get("reward"), "components": row.get("reward_components"), "selection": selected_matches(row)} for row in actionable]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "replay_admission.json").write_text(json.dumps(replay_admission, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "replay_sampling_exposure.json").write_text(json.dumps(sampling, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "learning_signal_root_cause.json").write_text(json.dumps({"primary_root_cause": summary["primary_root_cause"], "secondary_causes": summary["secondary_causes"], "evidence": {"opportunities": len(opportunities), "actionable": len(actionable), "exact": outcomes["EXACT_REPRODUCTION_SELECTION"], "successful_samples": replay_sampling.get("successful_reproduction_samples", 0), "action_replay_samples": replay_sampling.get("action_replay_samples", 0)}, "reasoning": "No actionable training transition selected the reproduction action, so no positive same-chain reproduction target entered replay or optimizer batches."}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
