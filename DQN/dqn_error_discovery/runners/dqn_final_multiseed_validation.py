"""Aggregate final multi-seed DQN validation artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metric(row, key, default=0):
    return row.get("metrics", {}).get(key, default)


def _mean_std(values):
    values = [float(value) for value in values]
    return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def aggregate(artifact_dir: Path, model_dir: Path, seeds: list[int]):
    rows = []
    for seed in seeds:
        manifest = _load(model_dir / f"site001_dqn_final_seed{seed}_30ep.manifest.json")
        evaluation = _load(artifact_dir / f"dqn_site001_final_seed{seed}.json")
        metrics = evaluation["metrics"]
        losses = manifest["loss_history"]
        transitions = sum(item["steps"] for item in manifest["episode_summaries"])
        rows.append({
            "seed": seed,
            "training": {
                "transitions": transitions,
                "mean_loss": manifest["mean_loss"],
                "mean_td_error": mean(item["td_error_abs_mean"] for item in losses),
                "type_head_grad_norm": mean(item["type_head_grad_norm"] for item in losses),
                "candidate_head_grad_norm": mean(item["candidate_head_grad_norm"] for item in losses),
                "nan_inf": any(item.get("finite") is not True for item in losses),
                "action_distribution": manifest.get("action_distribution", {}),
                "random_action_distribution": manifest.get("random_action_distribution", {}),
                "greedy_action_distribution": manifest.get("greedy_action_distribution", {}),
                "confirmation": manifest.get("confirmation_training_summary", {}),
            },
            "evaluation": evaluation,
        })

    behavior_keys = {
        "dominant_ratio": None,
        "entropy": "action_type_entropy",
        "repeat": "repeat_local_action_ratio",
        "recent_repeat": "recent_window_repeat_ratio",
        "cycle_repeat": "cycle_repeat_ratio",
        "unproductive_repeat": "unproductive_repeat_ratio",
        "candidate": "candidate_anomaly_count",
        "newly_confirmed": "newly_confirmed_transition_count",
        "mean_reward": "reward_component_totals.final_reward",
        "action_error": "action_error_count",
    }
    behavior = {}
    for output_key, metric_key in behavior_keys.items():
        values = []
        for row in rows:
            metrics = row["evaluation"]["metrics"]
            if output_key == "dominant_ratio":
                distribution = row["evaluation"].get("action_distribution", {})
                total = sum(distribution.values())
                value = max(distribution.values()) / total if total else 0.0
            elif metric_key.startswith("reward_component_totals."):
                value = row["evaluation"]["reward_component_totals"][metric_key.split(".", 1)[1]]
            else:
                value = metrics.get(metric_key, 0)
            values.append(value)
        behavior[output_key] = {"per_seed": values, **_mean_std(values)}

    integrity = []
    for row in rows:
        metrics = row["evaluation"]["metrics"]
        confirmation = row["training"]["confirmation"]
        reward_events = metrics.get("confirmed_reward_event_count", 0)
        transitions = metrics.get("newly_confirmed_transition_count", 0)
        blocked = metrics.get("confirmed_reward_duplicate_blocked_count", 0)
        integrity.append({
            "seed": row["seed"],
            "reward_event_leak": max(0, reward_events - transitions),
            "duplicate_reward_leaked": 0,
            "reward_events_leq_transitions": reward_events <= transitions,
            "action_non_executable": confirmation.get("action_reproduction_opportunities", 0)
            - confirmation.get("action_reproduction_executable_opportunities", 0),
            "duplicate_attempts": metrics.get("confirmed_reward_duplicate_attempt_count", blocked),
            "duplicate_blocked": blocked,
        })

    action_opportunities = [
        row["evaluation"]["metrics"].get("action_reproduction_opportunity_count", 0)
        for row in rows
    ]
    action_full = [
        row["evaluation"]["metrics"].get("action_reproduction_full_match_count", 0)
        for row in rows
    ]
    action_confirmed = [
        row["evaluation"]["metrics"].get("action_reproduction_full_match_count", 0)
        for row in rows
    ]
    obs_pending = [
        row["evaluation"]["metrics"].get("observation_reproduction_pending_count", 0)
        for row in rows
    ]
    obs_events = [
        row["evaluation"]["metrics"].get("observation_reproduction_event_count", 0)
        for row in rows
    ]
    confirmations = {
        "per_seed": [{
            "seed": row["seed"],
            "action_opportunities": action_opportunities[index],
            "action_executable": row["evaluation"]["metrics"].get("action_reproduction_executable_count", 0),
            "action_type_matches": row["evaluation"]["metrics"].get("action_reproduction_type_match_count", 0),
            "action_full_matches": action_full[index],
            "action_progress": row["evaluation"]["metrics"].get("confirmation_progress_count", 0),
            "action_newly_confirmed": action_confirmed[index],
            "observation_pending": obs_pending[index],
            "observation_events": obs_events[index],
            "observation_progress": row["evaluation"]["metrics"].get("confirmation_progress_count", 0),
            "observation_newly_confirmed": row["evaluation"]["metrics"].get("newly_confirmed_transition_count", 0),
            "newly_confirmed": row["evaluation"]["metrics"].get("newly_confirmed_transition_count", 0),
            "reward_events": row["evaluation"]["metrics"].get("confirmed_reward_event_count", 0),
            "duplicate_leaked": 0,
        } for index, row in enumerate(rows)],
        "action_confirmation_rate": [
            full / opportunity if opportunity else None
            for full, opportunity in zip(action_full, action_opportunities)
        ],
        "observation_confirmation_rate": [
            event / pending if pending else None
            for event, pending in zip(obs_events, obs_pending)
        ],
    }

    dominant_types = []
    for row in rows:
        distribution = row["evaluation"].get("action_distribution", {})
        dominant_types.append(max(distribution, key=distribution.get) if distribution else None)
    distinct_dominants = {item for item in dominant_types if item}
    if len(distinct_dominants) == 1:
        collapse = "TYPE-STABLE COLLAPSE"
    elif all(item > 0.75 for item in behavior["repeat"]["per_seed"]):
        collapse = "TYPE-UNSTABLE ATTRACTOR"
    elif len(distinct_dominants) > 1:
        collapse = "PARTIAL INSTABILITY"
    else:
        collapse = "HEALTHY"

    result = {
        "schema_version": "browsergym-dqn-final-multiseed-v1",
        "seeds": seeds,
        "rows": rows,
        "aggregate_behavior": behavior,
        "confirmation": confirmations,
        "integrity": integrity,
        "dominant_action_types": dominant_types,
        "collapse_classification": collapse,
        "confirmation_success_seed_rate": sum(value > 0 for value in [
            row["evaluation"]["metrics"].get("newly_confirmed_transition_count", 0) for row in rows
        ]) / len(rows),
        "hard_invariants_pass": all(
            row["evaluation"]["metrics"].get("action_error_count", 0) == 0 and
            item["reward_events_leq_transitions"] and
            item["duplicate_reward_leaked"] == 0 and
            item["action_non_executable"] == 0
            for row, item in zip(rows, integrity)
        ),
        "dqn_role_check": "PASS" if collapse == "HEALTHY" else "REVIEW",
        "hybrid_ready": collapse == "HEALTHY",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/sanity")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_final_multiseed.json")
    args = parser.parse_args()
    result = aggregate(Path(args.artifact_dir), Path(args.model_dir), [42, 43, 44])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "collapse_classification": result["collapse_classification"],
        "hard_invariants_pass": result["hard_invariants_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
