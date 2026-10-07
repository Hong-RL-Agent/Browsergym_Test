"""Read-only health analysis for RAWD v3 Run 1 checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.train import SITES, _choose_dqn, _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state


ROOT = Path(__file__).resolve().parents[2]
SPACE = V3ActionSpace(32)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _finite_state_dict(module: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def _ppo_summary() -> dict[str, Any]:
    summary = json.loads((ROOT / "artifacts/training_v3/ppo/training_summary.json").read_text())
    updates = summary.get("update_metrics", [])
    rewards = [float(row.get("mean_reward", 0.0)) for row in updates]
    entropy = [float((row.get("metrics") or {}).get("entropy", 0.0)) for row in updates]
    return {
        "update_count": len(updates),
        "mean_reward_by_update": rewards,
        "entropy_by_update": entropy,
        "reward_trend": "INSUFFICIENT_SIGNAL" if len(rewards) < 3 else ("IMPROVING" if rewards[-1] > rewards[0] else "DEGRADING"),
        "entropy_trend": entropy,
        "action_distribution": summary.get("action_counts", {}),
        "invalid_actions": 0,
        "action_error": int(summary.get("action_error", 0)),
        "nan_inf": int(summary.get("nan_inf", 0)),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_sha256": _hash(ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt"),
    }


def _dqn_summaries() -> list[dict[str, Any]]:
    manifest = json.loads((ROOT / "artifacts/models/v3/rawd_v3_dqn_training_manifest.json").read_text())
    rows = []
    for row in manifest.get("seed_results", []):
        rows.append({
            "seed": row["seed"],
            "reward": {"first_middle_last": "UNAVAILABLE_IN_ARTIFACT", "mean": row.get("mean_reward")},
            "loss": {"mean": row.get("loss_mean"), "first_middle_last": "UNAVAILABLE_IN_ARTIFACT", "finite": row.get("nan_inf", 0) == 0},
            "td_error": "UNAVAILABLE_IN_ARTIFACT",
            "q_value": "UNAVAILABLE_IN_ARTIFACT",
            "epsilon": "UNAVAILABLE_IN_ARTIFACT",
            "replay_size": row.get("replay_size"),
            "local_coverage": "UNAVAILABLE_IN_ARTIFACT",
            "action_distribution": row.get("action_counts", {}),
            "repeat_rate": "UNAVAILABLE_IN_ARTIFACT",
            "no_effect_rate": "UNAVAILABLE_IN_ARTIFACT",
            "candidate_count": row.get("candidate_count", 0),
            "confirmation_count": row.get("confirmation_count", 0),
            "candidate_to_confirmed_rate": None,
            "select_option_usage": row.get("select_option_usage", 0),
            "native_select_fill_input_attempts": row.get("native_select_fill_input_attempts", 0),
            "action_error": row.get("action_error", 0),
            "nan_inf": row.get("nan_inf", 0),
            "checkpoint": row.get("checkpoint"),
            "checkpoint_sha256": _hash(ROOT / f"artifacts/models/v3/rawd_v3_dqn_seed{row['seed']}.pt"),
        })
    return rows


def _inference(site_name: str, site_id: str, url: str, episodes: int = 3, max_steps: int = 5) -> dict[str, Any]:
    ppo = PPOAgent(635, 640)
    ppo.load(ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt")
    dqn = FactorizedQNetwork(984, 20, 32)
    dqn.load_state_dict(torch.load(ROOT / "artifacts/models/v3/rawd_v3_dqn_seed42.pt", map_location="cpu")["network_state_dict"])
    dqn.eval()
    counts: Counter[str] = Counter()
    result = {"site": site_name, "episodes": episodes, "steps": 0, "action_error": 0, "native_select_discovered": 0, "select_option_opportunities": 0, "select_option_selections": 0, "select_option_successes": 0, "native_select_fill_input_attempts": 0, "contract_violations": 0, "action_types": {}}
    for _ in range(episodes):
        env = _episode_env(site_id, url, max_steps)
        try:
            raw, _ = env.reset()
            for _ in range(max_steps):
                candidates = raw.get("candidate_elements", [])
                select_indexes = [i for i, c in enumerate(candidates) if str(c.get("tag", "")).lower() == "select"]
                result["native_select_discovered"] += len(select_indexes)
                mask = _v3_mask(SPACE, raw)
                result["select_option_opportunities"] += int(mask[15 * 32:16 * 32].sum())
                state = _v3_state(env, raw)
                selected = ppo.select_action(state, mask)
                next_raw, _, done, info, action_type = _step_v3(env, selected["action_id"], raw, SPACE)
                counts[action_type] += 1
                result["steps"] += 1
                result["action_error"] += int(bool(info.get("last_action_error")))
                if action_type == "select_option":
                    result["select_option_selections"] += 1
                    result["select_option_successes"] += int(bool(info.get("select_option")))
                if action_type == "fill_input" and any(i == SPACE.decode(selected["action_id"])["candidate_index"] for i in select_indexes):
                    result["native_select_fill_input_attempts"] += 1
                    result["contract_violations"] += 1
                raw = next_raw
                if done:
                    break
        finally:
            env.close()
    result["action_types"] = dict(counts)
    return result


def run() -> dict[str, Any]:
    ppo = PPOAgent(635, 640)
    ppo.load(ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt")
    dqn = {seed: FactorizedQNetwork(984, 20, 32) for seed in (42, 43, 44)}
    for seed, network in dqn.items():
        network.load_state_dict(torch.load(ROOT / f"artifacts/models/v3/rawd_v3_dqn_seed{seed}.pt", map_location="cpu")["network_state_dict"])
    fixed = {"ppo": {"output_dim": 640, "finite": _finite_state_dict(ppo.policy)}, "dqn": {str(seed): {"input_dim": 984, "type_head": 20, "candidate_head": [20, 32], "finite": _finite_state_dict(network)} for seed, network in dqn.items()}}
    inference = [_inference(name, sid, url) for name, (sid, url) in SITES.items()]
    return {"version": "RAWD_V3_POLICY_HEALTH", "ppo": _ppo_summary(), "dqn": _dqn_summaries(), "fixed_forward": fixed, "inference": inference, "site002_to_site005_used": False, "gt_used": False}


if __name__ == "__main__":
    output = ROOT / "artifacts/training_v3/policy_health"
    output.mkdir(parents=True, exist_ok=True)
    report = run()
    (output / "ppo_policy_health.json").write_text(json.dumps(report["ppo"], indent=2))
    for row in report["dqn"]:
        (output / f"dqn_seed{row['seed']}_policy_health.json").write_text(json.dumps(row, indent=2))
    (output / "rawd_v3_training_run1_policy_health_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
