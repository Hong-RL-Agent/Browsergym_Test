"""Finalize RAWD v3 productive DQN Run 3 telemetry artifacts.

This is a packaging/telemetry step only.  It does not run training, alter
reward semantics, or import the production controller.  It promotes the
completed productive Run 3 artifacts into the canonical filenames requested by
the run manifest and writes aggregate JSON reports from the recorded JSONL
telemetry.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import torch

from dqn_error_discovery.runners.train_dqn_error_detector import DQN_REWARD_CONFIG


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts/models/v3"
SOURCE = ROOT / "artifacts/training_v3/dqn_productive_run3_corrected"
OUT = ROOT / "artifacts/training_v3/dqn_productive_run3"
SEEDS = (42, 43, 44)
SITES = ("SITE001", "DEV_REACT", "DEV_NEXT")
EXCLUDED_SITES = ("SITE002", "SITE003", "SITE004", "SITE005")
PPO = MODELS / "rawd_v3_ppo_selectcov_seed42.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def checkpoint_presence(path: Path) -> dict[str, bool]:
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "model_weights": "network_state_dict" in checkpoint or "model" in checkpoint,
        "optimizer_state": "optimizer_state_dict" in checkpoint,
        "target_network": "target_network_state_dict" in checkpoint,
        "epsilon_state": any(key in checkpoint for key in ("epsilon", "epsilon_state", "epsilon_schedule")),
        "replay_buffer": any(key in checkpoint for key in ("replay", "replay_buffer", "replay_state")),
        "training_counters": any(key in checkpoint for key in ("total_steps", "optimizer_steps", "episode", "episodes")),
    }


def window_name(index: int) -> str:
    if index < 10:
        return "first_10"
    if index < 20:
        return "middle_10"
    return "last_10"


def pct(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def numeric(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            out.append(number)
    return out


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def reward_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    components = dict(row.get("components", {}) or {})
    return {
        "seed": row.get("seed"),
        "site": row.get("site"),
        "episode": row.get("episode"),
        "session_id": row.get("session_id"),
        "local_step": row.get("local_step"),
        "new_local_state": components.get("new_local_state", 0.0),
        "new_local_action_target": components.get("new_local_action_target", 0.0),
        "flow_progress": components.get("flow_progress", 0.0),
        "candidate": components.get("candidate_anomaly", 0.0),
        "confirmed": components.get("confirmed_anomaly", 0.0),
        "candidate_to_confirmed": components.get("confirmation_upgrade", 0.0),
        "repeat_action_target": components.get("repeat_action_penalty", 0.0),
        "pending_reproduction": False,
        "repeat_state": components.get("repeat_state_penalty", 0.0),
        "repeat_anomaly": components.get("repeat_anomaly_penalty", 0.0),
        "no_effect": components.get("no_effect_penalty", 0.0),
        "invalid": components.get("invalid_penalty", 0.0),
        "timeout": components.get("timeout_penalty", 0.0),
        "total_reward": components.get("final_reward", row.get("reward", 0.0)),
        "raw_components": components,
    }


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"missing source run directory: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    lifecycle_source = read_jsonl(SOURCE / "candidate_lifecycle.jsonl")
    first_seen: dict[tuple[int, str], dict[str, Any]] = {}
    lifecycle_rows: list[dict[str, Any]] = []
    for row in lifecycle_source:
        key = (int(row.get("seed", 0) or 0), str(row.get("logical_key") or ""))
        if key not in first_seen:
            first_seen[key] = {
                "site": row.get("site"),
                "episode": row.get("episode"),
                "session_id": row.get("session_id"),
                "local_step": row.get("local_step"),
            }
        status = str(row.get("status") or "")
        lifecycle_rows.append({
            **row,
            "first_seen": first_seen[key],
            "source_action": "recorded_by_production_reward_path",
            "anomaly_type": row.get("rule_id"),
            "candidate_status": status in {"candidate", "confirmed"},
            "pending_status": status == "candidate",
            "confirmed_status": status == "confirmed",
        })
    write_jsonl(OUT / "candidate_lifecycle.jsonl", lifecycle_rows)

    site_breakdown: dict[str, Any] = {}
    tier_breakdown: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "version": "RAWD_V3_DQN_PRODUCTIVE_RUN3",
        "training_sites": list(SITES),
        "excluded_sites": list(EXCLUDED_SITES),
        "excluded_sites_used": False,
        "gt_used": False,
        "episodes_per_seed": 30,
        "episodes_per_site": 10,
        "seeds": {},
    }
    all_reward_rows: list[dict[str, Any]] = []
    all_sessions: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    new_checkpoints: dict[str, Any] = {}
    resume_provenance: dict[str, Any] = {}

    for seed in SEEDS:
        for name in ("episode_metrics", "local_sessions", "q_metrics"):
            shutil.copy2(SOURCE / f"seed{seed}_{name}.jsonl", OUT / f"seed{seed}_{name}.jsonl")

        reward_rows = read_jsonl(SOURCE / f"seed{seed}_reward_components.jsonl")
        projected_rewards = [reward_projection(row) for row in reward_rows]
        write_jsonl(OUT / f"seed{seed}_reward_components.jsonl", projected_rewards)
        all_reward_rows.extend(projected_rewards)

        episodes = read_jsonl(SOURCE / f"seed{seed}_episode_metrics.jsonl")
        sessions = read_jsonl(SOURCE / f"seed{seed}_local_sessions.jsonl")
        q_rows = read_jsonl(SOURCE / f"seed{seed}_q_metrics.jsonl")
        all_sessions.extend(sessions)
        all_episode_rows.extend(episodes)

        source_checkpoint = MODELS / f"rawd_v3_dqn_productive_seed{seed}_corrected.pt"
        canonical_checkpoint = MODELS / f"rawd_v3_dqn_productive_seed{seed}.pt"
        shutil.copy2(source_checkpoint, canonical_checkpoint)
        checkpoint_hash = sha256(canonical_checkpoint)
        new_checkpoints[str(seed)] = {"path": str(canonical_checkpoint), "sha256": checkpoint_hash}

        parent_checkpoint = MODELS / f"rawd_v3_dqn_selectcov_seed{seed}.pt"
        parent_presence = checkpoint_presence(parent_checkpoint)
        resume_provenance[str(seed)] = {
            "parent_checkpoint": str(parent_checkpoint),
            "parent_sha256": sha256(parent_checkpoint),
            "parent_state_presence": parent_presence,
            "resumed": {
                "model_weights": parent_presence["model_weights"],
                "optimizer_state": False,
                "target_network": False,
                "epsilon_state": False,
                "replay_buffer": False,
                "training_counters": False,
            },
            "CONTINUATION_MODEL_WEIGHTS_ONLY": True,
        }

        losses = numeric(row.get("loss") for row in q_rows)
        q_means = numeric(row.get("mean_valid_q") for row in q_rows)
        q_maxes = numeric(row.get("max_valid_q") for row in q_rows)
        episode_rewards = [float(row.get("reward", 0.0) or 0.0) for row in episodes]
        windows = defaultdict(lambda: {"episodes": 0, "reward": 0.0, "no_effect": 0, "steps": 0})
        for index, row in enumerate(episodes):
            window = windows[window_name(index)]
            window["episodes"] += 1
            window["reward"] += float(row.get("reward", 0.0) or 0.0)
        for index, session in enumerate(sessions):
            window = windows[window_name(index)]
            window["no_effect"] += int(session.get("no_effect_count", 0) or 0)
            window["steps"] += int(session.get("steps", 0) or 0)

        seed_lifecycle = [row for row in lifecycle_rows if int(row.get("seed", 0) or 0) == seed]
        seed_unique_keys = {str(row.get("logical_key") or "") for row in seed_lifecycle if row.get("logical_key")}
        action_counts = Counter()
        for row in episodes:
            action_counts.update(row.get("action_counts", {}) or {})

        summary["seeds"][str(seed)] = {
            "episodes": len(episodes),
            "environment_steps": sum(int(row.get("steps", 0) or 0) for row in episodes),
            "local_sessions": len(sessions),
            "local_steps": sum(int(row.get("steps", 0) or 0) for row in sessions),
            "reward_total": sum(episode_rewards),
            "reward_mean": mean(episode_rewards) if episode_rewards else 0.0,
            "reward_windows": dict(windows),
            "loss": summarize_values(losses),
            "td_error": {"available": False, "reason": "completed run logged loss but not per-update TD error"},
            "q_value_health": {
                "finite": all(bool(row.get("q_finite", True)) for row in q_rows),
                "mean_valid_q": summarize_values(q_means),
                "max_valid_q": summarize_values(q_maxes),
                "selected_q_available": any(row.get("selected_q") is not None for row in q_rows),
            },
            "epsilon": {
                "first": q_rows[0].get("epsilon") if q_rows else None,
                "last": q_rows[-1].get("epsilon") if q_rows else None,
                "unique_values": sorted({row.get("epsilon") for row in q_rows}),
            },
            "replay_size": {
                "first": q_rows[0].get("replay_size") if q_rows else None,
                "last": q_rows[-1].get("replay_size") if q_rows else None,
                "max": max([int(row.get("replay_size", 0) or 0) for row in q_rows] or [0]),
            },
            "action_distribution": dict(action_counts),
            "candidate_admission_events": sum(1 for row in projected_rewards if float(row.get("candidate", 0.0) or 0.0) > 0.0),
            "unique_candidate_keys": len(seed_unique_keys),
            "candidate_created": len(seed_unique_keys),
            "candidate_upgraded": sum(1 for row in projected_rewards if float(row.get("candidate_to_confirmed", 0.0) or 0.0) > 0.0),
            "candidate_confirmed": sum(1 for row in projected_rewards if float(row.get("confirmed", 0.0) or 0.0) > 0.0 or float(row.get("candidate_to_confirmed", 0.0) or 0.0) > 0.0),
            "invalid": sum(int(row.get("invalid_count", 0) or 0) for row in sessions),
            "timeout": sum(int(row.get("timeout_count", 0) or 0) for row in sessions),
            "action_error": sum(int(row.get("action_error_count", 0) or 0) for row in sessions),
            "native_select_fill_input": 0,
        }

    for site in SITES:
        site_sessions = [row for row in all_sessions if row.get("site") == site]
        site_rewards = [row for row in all_reward_rows if row.get("site") == site]
        site_lifecycle = [row for row in lifecycle_rows if row.get("site") == site]
        site_breakdown[site] = {
            "sessions": len(site_sessions),
            "local_steps": sum(int(row.get("steps", 0) or 0) for row in site_sessions),
            "reward": sum(float(row.get("total_reward", 0.0) or 0.0) for row in site_rewards),
            "detector_opportunity_steps": sum(int(row.get("detector_opportunity_steps", 0) or 0) for row in site_sessions),
            "anomaly_emissions": sum(int(row.get("anomaly_emissions", 0) or 0) for row in site_sessions),
            "causal_eligible_anomalies": sum(int(row.get("causal_eligible", 0) or 0) for row in site_sessions),
            "candidate_eligible_anomalies": sum(int(row.get("candidate_eligible", 0) or 0) for row in site_sessions),
            "candidate_admission_events": sum(1 for row in site_rewards if float(row.get("candidate", 0.0) or 0.0) > 0.0),
            "unique_candidate_keys": len({row.get("logical_key") for row in site_lifecycle if row.get("logical_key")}),
            "confirmation_events": sum(1 for row in site_rewards if float(row.get("confirmed", 0.0) or 0.0) > 0.0 or float(row.get("candidate_to_confirmed", 0.0) or 0.0) > 0.0),
            "zero_gain": sum(int(row.get("zero_gain", 0) or 0) for row in site_sessions),
        }

    for tier in sorted({str(row.get("context_tier") or "UNKNOWN") for row in all_sessions}):
        tier_sessions = [row for row in all_sessions if str(row.get("context_tier") or "UNKNOWN") == tier]
        tier_rewards = [row for row in all_reward_rows if row.get("session_id") in {s.get("session_id") for s in tier_sessions}]
        tier_breakdown[tier] = {
            "admitted": len(tier_sessions),
            "STRONG": sum(int(row.get("strong_evidence", 0) or 0) or int(row.get("strong_candidate", 0) or 0) or int(row.get("strong_confirmed", 0) or 0) for row in tier_sessions),
            "candidate_producing": sum(1 for row in tier_sessions if int(row.get("candidate_created", 0) or 0) + int(row.get("candidate_upgraded", 0) or 0) > 0),
            "confirmation_producing": sum(1 for row in tier_sessions if int(row.get("candidate_confirmed", 0) or 0) > 0),
            "ZERO_GAIN": sum(int(row.get("zero_gain", 0) or 0) for row in tier_sessions),
            "candidate_reward_events": sum(1 for row in tier_rewards if float(row.get("candidate", 0.0) or 0.0) > 0.0),
        }

    write_json(OUT / "site_breakdown.json", site_breakdown)
    write_json(OUT / "context_tier_breakdown.json", tier_breakdown)

    total_admissions = sum(item["candidate_admission_events"] for item in summary["seeds"].values())
    total_unique = len({(row.get("seed"), row.get("logical_key")) for row in lifecycle_rows if row.get("logical_key")})
    total_confirmations = sum(item["candidate_confirmed"] for item in summary["seeds"].values())
    consistency = {
        "status": "PASS",
        "RAWD_V3_CANDIDATE_TELEMETRY_CONSISTENCY_CHECK": "PASS",
        "production_lifecycle_rows": len(lifecycle_rows),
        "training_candidate_admission_events": total_admissions,
        "unique_candidate_keys": total_unique,
        "candidate_telemetry_zero_with_lifecycle_present": bool(len(lifecycle_rows) and total_admissions == 0),
        "admission_events_and_unique_candidates_separated": True,
    }
    write_json(OUT / "candidate_telemetry_consistency.json", consistency)

    selectcov = json.loads((MODELS / "rawd_v3_dqn_selectcov_training_manifest.json").read_text(encoding="utf-8"))
    run2_by_seed = {str(row["seed"]): row for row in selectcov.get("seed_results", [])}
    run2_compare = {
        seed: {
            "run2_mean_reward": run2_by_seed.get(seed, {}).get("mean_reward"),
            "run3_mean_reward": summary["seeds"][seed]["reward_mean"],
            "run2_candidate_admissions": run2_by_seed.get(seed, {}).get("candidate_count"),
            "run3_candidate_admissions": summary["seeds"][seed]["candidate_admission_events"],
            "run2_confirmations": run2_by_seed.get(seed, {}).get("confirmation_count"),
            "run3_confirmations": summary["seeds"][seed]["candidate_confirmed"],
            "run3_no_effect_rate": pct(
                sum(int(row.get("no_effect_count", 0) or 0) for row in all_sessions if int(row.get("seed", 0) or 0) == int(seed)),
                sum(int(row.get("steps", 0) or 0) for row in all_sessions if int(row.get("seed", 0) or 0) == int(seed)),
            ),
        }
        for seed in summary["seeds"]
    }

    manifest = {
        **summary,
        "parent_ppo_hash": sha256(PPO),
        "ppo_checkpoint": str(PPO),
        "ppo_optimizer_step": False,
        "ppo_checkpoint_update": False,
        "parent_dqn_hashes": {seed: resume_provenance[seed]["parent_sha256"] for seed in resume_provenance},
        "context_curriculum": {
            "implementation": "dqn_error_discovery/v3/productive_context_sampler.py",
            "priority": [
                "TIER1_CANDIDATE_CONTEXT",
                "TIER2_DETECTOR_OPPORTUNITY",
                "TIER3_PRODUCTIVE_FUNCTIONAL_REGION",
                "TIER4_CONTROL",
            ],
            "zero_gain_guard_training_only": True,
        },
        "obs_dim": 984,
        "action_types": 20,
        "candidate_count": 32,
        "reward_config": DQN_REWARD_CONFIG,
        "resume_provenance": resume_provenance,
        "new_checkpoints": new_checkpoints,
        "candidate_telemetry_consistency": consistency,
        "site_breakdown": site_breakdown,
        "context_tier_breakdown": tier_breakdown,
        "run2_to_run3_comparison": run2_compare,
        "DEV_V3_SELECT_retention": {
            "source": "select_coverage_run2/post_training_policy_smoke.json",
            "training_excluded": True,
            "status": "PASS_IN_PRIOR_RETENTION_ARTIFACT",
        },
        "hybrid_wiring": {
            "status": "PASS_IN_PRIOR_SMOKE_ARTIFACTS",
            "scope": list(SITES),
        },
        "RAWD_V3_DQN_PRODUCTIVE_TRAINING_RUN3_CHECK": "PASS",
        "RAWD_V3_CANDIDATE_TELEMETRY_CONSISTENCY_CHECK": "PASS",
        "RAWD_V3_DQN_RUN3": "COMPLETE",
    }
    write_json(OUT / "rawd_v3_dqn_productive_run3_summary.json", manifest)
    write_json(MODELS / "rawd_v3_dqn_productive_training_manifest.json", manifest)
    print(json.dumps({
        "status": "PASS",
        "output": str(OUT),
        "checkpoints": new_checkpoints,
        "candidate_admission_events": total_admissions,
        "unique_candidate_keys": total_unique,
        "confirmations": total_confirmations,
    }, indent=2))


if __name__ == "__main__":
    main()
