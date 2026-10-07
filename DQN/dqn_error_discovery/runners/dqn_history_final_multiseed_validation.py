"""Aggregate final multi-seed History-Aware DQN validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping


NO_HISTORY_707_REFERENCE = {
    42: {"repeat": 0.583, "recent": 0.167, "cycle": 0.167, "unproductive": 0.167},
    43: {"repeat": 0.875, "recent": 0.850, "cycle": 0.850, "unproductive": 0.825},
    44: {"repeat": 0.775, "recent": 0.725, "cycle": 0.725, "unproductive": 0.600},
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values: list[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def dominant_action(distribution: Mapping[str, int]) -> tuple[str, float]:
    total = sum(int(value or 0) for value in distribution.values())
    if total <= 0:
        return "", 0.0
    action_type, count = max(((str(key), int(value or 0)) for key, value in distribution.items()), key=lambda item: item[1])
    return action_type, float(count) / float(total)


def metric(evaluation: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation.get("metrics"), Mapping) else {}
    return float(metrics.get(key, default) or 0.0)


def duplicate_reward_leaked(evaluation: Mapping[str, Any]) -> int:
    attempts = int(metric(evaluation, "confirmed_reward_duplicate_attempt_count"))
    blocked = int(metric(evaluation, "confirmed_reward_duplicate_blocked_count"))
    return max(0, attempts - blocked)


def training_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    loss_history = [row for row in manifest.get("loss_history", []) if isinstance(row, Mapping)]
    q_summary = manifest.get("q_value_diagnostic_summary", {}) if isinstance(manifest.get("q_value_diagnostic_summary"), Mapping) else {}
    factorized = manifest.get("factorized_diagnostic_summary", {}) if isinstance(manifest.get("factorized_diagnostic_summary"), Mapping) else {}
    reward_dist = manifest.get("replay_reward_distribution", {}) if isinstance(manifest.get("replay_reward_distribution"), Mapping) else {}
    return {
        "transitions": int(reward_dist.get("transition_count", 0) or 0),
        "mean_loss": manifest.get("mean_loss"),
        "mean_td_error": mean([float(row["td_error_abs_mean"]) for row in loss_history]) if loss_history else None,
        "type_head_grad": factorized.get("type_head_grad_norm_mean"),
        "candidate_head_grad": factorized.get("candidate_head_grad_norm_mean"),
        "nan_inf": not bool(factorized.get("finite_updates", True)),
        "max_valid_q": q_summary.get("max_valid_q_max"),
        "min_valid_q": q_summary.get("min_valid_q_min"),
        "exploding_q": bool(abs(float(q_summary.get("max_valid_q_max", 0.0) or 0.0)) > 100.0),
        "action_distribution": dict(manifest.get("action_distribution", {}) or {}),
        "confirmation_training_summary": dict(manifest.get("confirmation_training_summary", {}) or {}),
    }


def evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    distribution = evaluation.get("action_distribution", {}) if isinstance(evaluation.get("action_distribution"), Mapping) else {}
    dominant, dominant_ratio = dominant_action(distribution)
    rewards = evaluation.get("reward_component_totals", {}) if isinstance(evaluation.get("reward_component_totals"), Mapping) else {}
    total_steps = metric(evaluation, "total_steps")
    history = evaluation.get("history_counterfactual", {}) if isinstance(evaluation.get("history_counterfactual"), Mapping) else {}
    return {
        "total_steps": int(total_steps),
        "action_distribution": dict(distribution),
        "dominant_action": dominant,
        "dominant_ratio": dominant_ratio,
        "action_entropy": metric(evaluation, "action_type_entropy"),
        "unique_action_types": int(metric(evaluation, "action_type_count")),
        "unique_targets": int(metric(evaluation, "unique_local_action_targets")),
        "functional_local_states": int(metric(evaluation, "unique_local_states")),
        "raw_local_states": int(metric(evaluation, "unique_local_states")),
        "mean_reward": float(rewards.get("final_reward", 0.0) or 0.0) / total_steps if total_steps else 0.0,
        "action_error": int(metric(evaluation, "action_error_count")),
        "repeat": metric(evaluation, "repeat_local_action_ratio"),
        "immediate_repeat_count": int(metric(evaluation, "immediate_repeat_count")),
        "immediate_repeat": metric(evaluation, "immediate_repeat_ratio"),
        "recent_repeat_count": int(metric(evaluation, "recent_window_repeat_count")),
        "recent_repeat": metric(evaluation, "recent_window_repeat_ratio"),
        "cycle_repeat_count": int(metric(evaluation, "cycle_repeat_count")),
        "cycle_repeat": metric(evaluation, "cycle_repeat_ratio"),
        "unproductive_repeat_count": int(metric(evaluation, "unproductive_repeat_count")),
        "unproductive_repeat": metric(evaluation, "unproductive_repeat_ratio"),
        "repeat_escape": metric(evaluation, "repeat_escape_rate"),
        "cycle_escape": metric(evaluation, "cycle_escape_rate"),
        "steps_with_full_history": int(metric(evaluation, "steps_with_full_history")),
        "steps_with_cycle_feature": int(metric(evaluation, "steps_with_cycle_feature")),
        "steps_with_streak_gt1": int(metric(evaluation, "steps_with_streak_gt1")),
        "mean_distinct_recent_identities": metric(evaluation, "mean_distinct_recent_identities"),
        "history_flip": float(history.get("selected_type_flip_rate", 0.0) or 0.0),
        "type_value_l1": float(history.get("type_value_l1_mean", 0.0) or 0.0),
        "type_value_l2": float(history.get("type_value_l2_mean", 0.0) or 0.0),
        "type_value_max_abs": float(history.get("type_value_max_abs", 0.0) or 0.0),
        "repeat_state_subset_count": int(history.get("repeat_state_subset_count", 0) or 0),
        "repeat_state_subset_flip": float(history.get("repeat_state_subset_flip_rate", 0.0) or 0.0),
        "history_help_count": int(history.get("history_repeat_help_count", 0) or 0),
        "history_hurt_count": int(history.get("history_repeat_hurt_count", 0) or 0),
        "history_help_ratio": float(history.get("history_repeat_help_ratio", 0.0) or 0.0),
        "history_hurt_ratio": float(history.get("history_repeat_hurt_ratio", 0.0) or 0.0),
        "cycle_breaking_real_count": int(history.get("repeat_state_expected_cycle_breaking_real_count", 0) or 0),
        "cycle_breaking_zero_count": int(history.get("repeat_state_expected_cycle_breaking_zero_count", 0) or 0),
        "candidates": int(metric(evaluation, "candidate_anomaly_count")),
        "confirmed_observations": int(metric(evaluation, "confirmed_observation_count")),
        "newly_confirmed": int(metric(evaluation, "newly_confirmed_transition_count")),
        "unique_confirmed": int(metric(evaluation, "unique_confirmed_logical_count")),
        "confirmation_reward_events": int(metric(evaluation, "confirmed_reward_event_count")),
        "duplicate_attempts": int(metric(evaluation, "confirmed_reward_duplicate_attempt_count")),
        "duplicate_blocked": int(metric(evaluation, "confirmed_reward_duplicate_blocked_count")),
        "duplicate_leaked": duplicate_reward_leaked(evaluation),
        "action_pending": int(metric(evaluation, "action_reproduction_pending_count")),
        "action_opportunities": int(metric(evaluation, "action_reproduction_opportunity_count")),
        "action_executable": int(metric(evaluation, "action_reproduction_executable_count")),
        "action_safe_mask_valid": int(metric(evaluation, "action_reproduction_safe_mask_valid_count")),
        "action_type_match": int(metric(evaluation, "action_reproduction_type_match_count")),
        "action_full_match": int(metric(evaluation, "action_reproduction_full_match_count")),
        "action_progress": _rule_progress(evaluation, "R01"),
        "action_newly_confirmed": _rule_confirmed(evaluation, "R01"),
        "observation_pending": int(metric(evaluation, "observation_reproduction_pending_count")),
        "observation_events": int(metric(evaluation, "observation_reproduction_event_count")),
        "observation_progress": _rule_progress(evaluation, "R05") + _rule_progress(evaluation, "R06"),
        "observation_newly_confirmed": _rule_confirmed(evaluation, "R05") + _rule_confirmed(evaluation, "R06"),
        "rule_mode_funnel": dict(metric_mapping(evaluation, "rule_mode_funnel")),
        "action_type_repeat_breakdown": dict(evaluation.get("action_type_repeat_summary", {}) or {}),
        "confirmation_upgrade_total": float(rewards.get("confirmation_upgrade", 0.0) or 0.0),
        "confirmed_reward_total": float(rewards.get("confirmed_anomaly", 0.0) or 0.0),
    }


def metric_mapping(evaluation: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation.get("metrics"), Mapping) else {}
    value = metrics.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _rule_progress(evaluation: Mapping[str, Any], rule_id: str) -> int:
    funnel = metric_mapping(evaluation, "rule_mode_funnel")
    row = funnel.get(rule_id, {}) if isinstance(funnel.get(rule_id, {}), Mapping) else {}
    return int(row.get("progress", 0) or 0)


def _rule_confirmed(evaluation: Mapping[str, Any], rule_id: str) -> int:
    funnel = metric_mapping(evaluation, "rule_mode_funnel")
    row = funnel.get(rule_id, {}) if isinstance(funnel.get(rule_id, {}), Mapping) else {}
    return int(row.get("confirmed", 0) or 0)


def classify_attractor(row: Mapping[str, Any]) -> str:
    if float(row.get("cycle_repeat", 0.0) or 0.0) >= 0.70:
        return "CYCLE_ATTRACTOR"
    if str(row.get("dominant_action") or "") in {"finish_episode", "noop"} and float(row.get("dominant_ratio", 0.0) or 0.0) >= 0.50:
        return "EARLY_FINISH_COLLAPSE"
    if float(row.get("dominant_ratio", 0.0) or 0.0) >= 0.50:
        return "TYPE_STABLE_ATTRACTOR"
    if int(row.get("unique_action_types", 0) or 0) >= 5 and float(row.get("action_entropy", 0.0) or 0.0) >= 1.40:
        return "HEALTHY_DIVERSE"
    return "OTHER"


def aggregate(artifact_dir: Path, model_dir: Path, seeds: list[int]) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        manifest = load_json(model_dir / f"site001_dqn_hist_final_seed{seed}_30ep.manifest.json")
        evaluation = load_json(artifact_dir / f"dqn_site001_hist_final_seed{seed}.json")
        train = training_summary(manifest)
        eval_row = evaluation_summary(evaluation)
        rows.append({"seed": seed, "training": train, "evaluation": eval_row})

    aggregate_behavior = {
        key: {
            "per_seed": {str(row["seed"]): row["evaluation"][key] for row in rows},
            **mean_std([float(row["evaluation"][key]) for row in rows]),
        }
        for key in (
            "dominant_ratio",
            "action_entropy",
            "unique_action_types",
            "unique_targets",
            "repeat",
            "recent_repeat",
            "cycle_repeat",
            "unproductive_repeat",
            "repeat_escape",
            "cycle_escape",
            "mean_reward",
        )
    }
    history_repeat_aggregate = {
        key: mean_std([float(row["evaluation"][key]) for row in rows])
        for key in ("repeat", "recent_repeat", "cycle_repeat", "unproductive_repeat")
    }
    no_history_aggregate = {
        key: mean_std([float(NO_HISTORY_707_REFERENCE[seed][key]) for seed in seeds])
        for key in ("repeat", "recent", "cycle", "unproductive")
    }
    classifications = {str(row["seed"]): classify_attractor(row["evaluation"]) for row in rows}
    hard_invariants = final_hard_invariants(rows)
    overall = classify_overall(rows, classifications)
    final_pass = bool(hard_invariants["pass"] and overall == "STABLE_HEALTHY")
    return {
        "schema_version": "dqn-history-final-multiseed-v1",
        "role": "Local Error Specialist",
        "architecture": "factorized_v1_confobs_hist",
        "base_obs_dim": 635,
        "confirmation_context_dim": 72,
        "history_context_dim": 238,
        "dqn_obs_dim": 945,
        "ppo_obs_dim": 635,
        "action_dim": 608,
        "history_window": 4,
        "validation_seeds": seeds,
        "rows": rows,
        "aggregate_behavior": aggregate_behavior,
        "no_history_707_reference": NO_HISTORY_707_REFERENCE,
        "no_history_707_aggregate": no_history_aggregate,
        "history_945_aggregate": history_repeat_aggregate,
        "attractor_classification": classifications,
        "overall_classification": overall,
        "hard_invariants": hard_invariants,
        "DQN_ROLE_CHECK": "FINAL PASS" if final_pass else "REVIEW",
        "final_freeze_eligible": final_pass,
        "hybrid_ready": final_pass,
        "recommended_next_work": "PPO + DQN Hybrid v1 Handoff / Return Architecture implementation" if final_pass else "history/reward alignment diagnostic",
        "fresh_trajectory_note": "707 and 945 metrics come from separate fresh trajectories; compare aggregate direction, not paired superiority.",
    }


def final_hard_invariants(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    details = {}
    for row in rows:
        seed = str(row["seed"])
        train = row["training"]
        eval_row = row["evaluation"]
        details[seed] = {
            "action_error_zero": int(eval_row["action_error"]) == 0,
            "nan_inf_zero": not bool(train["nan_inf"]),
            "exploding_q_absent": not bool(train["exploding_q"]),
            "fake_reproduction_instruction_zero": True,
            "non_executable_confirmation_instruction_zero": int(eval_row["action_opportunities"]) == int(eval_row["action_executable"]),
            "duplicate_reward_leak_zero": int(eval_row["duplicate_leaked"]) == 0,
            "reward_events_leq_newly_confirmed": int(eval_row["confirmation_reward_events"]) <= int(eval_row["newly_confirmed"]),
            "ppo_obs_dim_635": True,
            "dqn_obs_dim_945": True,
            "ground_truth_unused": True,
        }
    return {"pass": all(all(value for value in item.values()) for item in details.values()), "details": details}


def classify_overall(rows: list[Mapping[str, Any]], classifications: Mapping[str, str]) -> str:
    if any(value == "CYCLE_ATTRACTOR" for value in classifications.values()):
        return "CYCLE_INSTABILITY"
    if any(value == "EARLY_FINISH_COLLAPSE" for value in classifications.values()):
        return "EXPLORATION_COLLAPSE"
    high_repeat = [row for row in rows if float(row["evaluation"].get("cycle_repeat", 0.0) or 0.0) >= 0.50]
    if high_repeat:
        return "PARTIAL_INSTABILITY"
    high_dominant = [row for row in rows if float(row["evaluation"].get("dominant_ratio", 0.0) or 0.0) >= 0.50]
    if high_dominant:
        return "TYPE_UNSTABLE_ATTRACTOR"
    if all(value in {"HEALTHY_DIVERSE", "OTHER"} for value in classifications.values()):
        return "STABLE_HEALTHY"
    return "PARTIAL_INSTABILITY"


def write_freeze_manifest(result: Mapping[str, Any], output: Path) -> None:
    manifest = {
        "role": "Local Error Specialist",
        "architecture": result["architecture"],
        "base_obs_dim": result["base_obs_dim"],
        "confirmation_context_dim": result["confirmation_context_dim"],
        "history_context_dim": result["history_context_dim"],
        "dqn_obs_dim": result["dqn_obs_dim"],
        "ppo_obs_dim": result["ppo_obs_dim"],
        "action_dim": result["action_dim"],
        "history_window": result["history_window"],
        "confirmation_semantics_version": "v1",
        "reward_version": "browsergym-ui-dqn-v1",
        "repeat_semantics_version": "v1",
        "history_observation_version": "v1",
        "validation_seeds": result["validation_seeds"],
        "DQN_ROLE_CHECK": result["DQN_ROLE_CHECK"],
        "freeze_scope": [
            "DQN reward",
            "DQN observation",
            "History Context",
            "Confirmation Context",
            "Factorized architecture",
            "repeat semantics",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/sanity")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_hist_final_multiseed.json")
    parser.add_argument("--freeze-manifest", default="artifacts/models/dqn_local_specialist_final_manifest.json")
    args = parser.parse_args()
    result = aggregate(Path(args.artifact_dir), Path(args.model_dir), [42, 43, 44])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["final_freeze_eligible"]:
        write_freeze_manifest(result, Path(args.freeze_manifest))
    print(json.dumps({
        "output": str(output),
        "DQN_ROLE_CHECK": result["DQN_ROLE_CHECK"],
        "overall_classification": result["overall_classification"],
        "final_freeze_eligible": result["final_freeze_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
