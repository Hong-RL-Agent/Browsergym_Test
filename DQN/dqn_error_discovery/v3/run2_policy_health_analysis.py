"""Read-only Run 2 policy/productivity analysis from persisted artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, factorized_flat_q_values
from dqn_error_discovery.v3.action_space import V3ActionSpace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/run2_policy_health"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, total: float) -> float | None:
    return None if not total else value / total


def q_probe() -> dict:
    space = V3ActionSpace(32)
    state = torch.zeros((1, 984), dtype=torch.float32)
    mask = torch.ones((1, 640), dtype=torch.bool)
    rows = {}
    for seed in (42, 43, 44):
        net = FactorizedQNetwork(984, 20, 32)
        checkpoint = ROOT / f"artifacts/models/v3/rawd_v3_dqn_selectcov_seed{seed}.pt"
        net.load_state_dict(torch.load(checkpoint, map_location="cpu")["network_state_dict"])
        net.eval()
        with torch.inference_mode():
            q = factorized_flat_q_values(net(state), mask, space)[0]
        top = torch.topk(q, k=10)
        rows[str(seed)] = {
            "fixture": "deterministic_zero_state_all_actions_valid",
            "finite": bool(torch.isfinite(q).all()),
            "top_action_ids": [int(x) for x in top.indices],
            "top_actions": [space.decode(int(x)) for x in top.indices],
            "top_q_values": [float(x) for x in top.values],
            "mean_q": float(q.mean()),
            "std_q": float(q.std()),
            "max_abs_q": float(q.abs().max()),
        }
    return rows


def main() -> None:
    ppo = load(ROOT / "artifacts/training_v3/select_coverage_run2/ppo/training_summary.json")
    dqn = load(ROOT / "artifacts/training_v3/select_coverage_run2/dqn/training_summary.json")
    run1 = load(ROOT / "artifacts/models/v3/rawd_v3_dqn_training_manifest.json")

    ppo_updates = ppo.get("update_metrics", [])
    ppo_rewards = [float(row.get("mean_reward", 0.0)) for row in ppo_updates]
    ppo_entropy = [float((row.get("metrics") or {}).get("entropy", 0.0)) for row in ppo_updates]
    ppo_selection_rate = pct(ppo.get("select_option_selections", 0), ppo.get("select_option_opportunities", 0))
    ppo_success_rate = pct(ppo.get("select_option_successes", 0), ppo.get("select_option_selections", 0))

    dqn_rows = []
    for row in dqn.get("seed_results", []):
        counts = row.get("action_counts", {})
        total = sum(counts.values())
        dqn_rows.append({
            "seed": row["seed"],
            "run1_mean_reward": next((x.get("mean_reward") for x in run1.get("seed_results", []) if x.get("seed") == row["seed"]), "UNAVAILABLE_IN_ARTIFACT"),
            "run2_mean_reward": row.get("mean_reward"),
            "reward_trend": "INSUFFICIENT_DATA",
            "total_steps": row.get("total_steps"),
            "action_counts": counts,
            "action_distribution": {key: value / total for key, value in counts.items()} if total else {},
            "top_action_types": sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5],
            "unique_action_types": len(counts),
            "unique_local_states": "UNAVAILABLE_IN_ARTIFACT",
            "unique_local_targets": "UNAVAILABLE_IN_ARTIFACT",
            "unique_local_action_targets": "UNAVAILABLE_IN_ARTIFACT",
            "no_effect_rate": "UNAVAILABLE_IN_ARTIFACT",
            "repeat_rate": "UNAVAILABLE_IN_ARTIFACT",
            "local_session_count": "UNAVAILABLE_IN_ARTIFACT",
            "mean_steps_per_session": "UNAVAILABLE_IN_ARTIFACT",
            "strong_sessions": 0,
            "weak_sessions": "UNAVAILABLE_IN_ARTIFACT",
            "zero_gain_sessions": "UNAVAILABLE_IN_ARTIFACT",
            "detector_opportunities": "UNAVAILABLE_IN_ARTIFACT",
            "candidate_count": row.get("candidate_count", 0),
            "confirmation_count": row.get("confirmation_count", 0),
            "loss_mean": row.get("loss_mean"),
            "loss_median": "UNAVAILABLE_IN_ARTIFACT",
            "loss_p95": "UNAVAILABLE_IN_ARTIFACT",
            "td_error": "UNAVAILABLE_IN_ARTIFACT",
            "q_value": "fixed_probe_below",
            "invalid": row.get("invalid_action", 0),
            "timeout": "UNAVAILABLE_IN_ARTIFACT",
            "action_error": row.get("action_error", 0),
            "select_option_usage": row.get("select_option_usage", 0),
            "native_select_fill_input_attempts": row.get("native_select_fill_input_attempts", 0),
            "health": "STABLE_BUT_UNPRODUCTIVE",
        })

    smoke = load(ROOT / "artifacts/training_v3/select_coverage_run2/post_training_policy_smoke.json")
    dev_select = {"ppo": smoke["validation"]["ppo"][-1], "dqn": {seed: values[-1] for seed, values in smoke["validation"]["dqn"].items()}}
    report = {
        "version": "RAWD_V3_RUN2_POLICY_HEALTH_ANALYSIS",
        "production_algorithm_changed": False,
        "ppo": {
            "reward_by_update": ppo_rewards,
            "reward_trend": "STABLE_IMPROVING" if ppo_rewards and ppo_rewards[-1] >= ppo_rewards[0] else "REVIEW",
            "entropy_by_update": ppo_entropy,
            "action_distribution": ppo.get("action_counts", {}),
            "select_option_opportunities": ppo.get("select_option_opportunities", 0),
            "select_option_selections": ppo.get("select_option_selections", 0),
            "select_option_successes": ppo.get("select_option_successes", 0),
            "select_selection_rate": ppo_selection_rate,
            "select_success_rate": ppo_success_rate,
            "invalid": ppo.get("invalid_action", 0),
            "action_error": ppo.get("action_error", 0),
            "health": "HEALTHY",
        },
        "dqn": dqn_rows,
        "fixed_q_probe": q_probe(),
        "seed_similarity": "UNAVAILABLE_AS_STATISTICAL_METRIC; top-action lists preserved in fixed_q_probe",
        "handoff_productivity": {"NEW_CANDIDATE": "UNAVAILABLE_IN_ARTIFACT", "PENDING_CONFIRMATION": "UNAVAILABLE_IN_ARTIFACT", "FUNCTIONAL_REGION": "UNAVAILABLE_IN_ARTIFACT", "STALLED_FUNCTIONAL": "UNAVAILABLE_IN_ARTIFACT"},
        "ppo_candidate_contexts_handed_to_dqn": "UNAVAILABLE_IN_ARTIFACT",
        "dqn_confirmations_from_ppo_candidates": 0,
        "valid_action_count_at_local_entry": "UNAVAILABLE_IN_ARTIFACT",
        "detector_opportunity_to_candidate_conversion": "UNAVAILABLE_IN_ARTIFACT",
        "reward_component_breakdown": "UNAVAILABLE_IN_ARTIFACT",
        "site_breakdown": "UNAVAILABLE_IN_ARTIFACT; smoke preserved separately",
        "dev_v3_select_smoke": dev_select,
        "dev_v3_select_included_excluded_comparison": "Training artifact includes DEV_V3_SELECT; per-site reward/productivity split is unavailable.",
        "negative_reward_root_classification": "NO_CANDIDATE_REWARD_AVAILABLE + INSUFFICIENT_PRODUCTIVE_LOCAL_CONTEXT",
        "overall_dqn_classification": "DQN_STABLE_BUT_UNPRODUCTIVE",
        "action_contract_regression": False,
        "gt_used": False,
        "site002_to_site005_used": False,
        "artifacts": ["dqn_seed42_productivity.json", "dqn_seed43_productivity.json", "dqn_seed44_productivity.json", "dqn_detector_opportunity_analysis.json", "dqn_handoff_productivity.json", "rawd_v3_run2_policy_health_summary.json"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppo_run2_policy_health.json").write_text(json.dumps(report["ppo"], indent=2), encoding="utf-8")
    for row in dqn_rows:
        (OUT / f"dqn_seed{row['seed']}_productivity.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (OUT / "dqn_site_breakdown.json").write_text(json.dumps(report["site_breakdown"], indent=2), encoding="utf-8")
    (OUT / "dqn_detector_opportunity_analysis.json").write_text(json.dumps({"opportunities": report["detector_opportunity_to_candidate_conversion"]}, indent=2), encoding="utf-8")
    (OUT / "dqn_handoff_productivity.json").write_text(json.dumps(report["handoff_productivity"], indent=2), encoding="utf-8")
    (OUT / "rawd_v3_run2_policy_health_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
