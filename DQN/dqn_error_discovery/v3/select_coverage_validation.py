"""Read-only validation for RAWD v3 select coverage Run 2 checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.candidate_dom_enrichment import enrich_candidates_with_dom
from dqn_error_discovery.v3.train import (
    COVERAGE_SITES,
    SITES,
    _choose_dqn,
    _dqn_state,
    _enrich_observation,
    _episode_env,
    _step_v3,
    _v3_mask,
    _v3_state,
)
from envs.browsergym_jaws_env import _active_page

ROOT = Path(__file__).resolve().parents[2]
SPACE = V3ActionSpace(32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_policy(site_name: str, site_id: str, url: str, ppo: PPOAgent, dqn: FactorizedQNetwork | None, episodes: int = 3, max_steps: int = 5) -> dict:
    result = {
        "site": site_name,
        "episodes": episodes,
        "steps": 0,
        "action_error": 0,
        "native_select_discovered": 0,
        "select_option_opportunities": 0,
        "select_option_selections": 0,
        "select_option_successes": 0,
        "native_select_fill_input_attempts": 0,
        "contract_violations": 0,
        "action_types": {},
        "handoffs": 0,
        "returns": 0,
    }
    counts: Counter[str] = Counter()
    for _ in range(episodes):
        env = _episode_env(site_id, url, max_steps)
        history = {}
        try:
            raw, _ = env.reset()
            raw = _enrich_observation(env, raw)
            for _ in range(max_steps):
                candidates = raw.get("candidate_elements", [])
                native_indexes = [index for index, candidate in enumerate(candidates) if candidate.get("canonical_native_control_kind") == "native_select"]
                result["native_select_discovered"] += len(native_indexes)
                mask = _v3_mask(SPACE, raw)
                result["select_option_opportunities"] += int(mask[15 * 32:16 * 32].sum())
                state = _dqn_state(env, raw, history) if dqn is not None else _v3_state(env, raw)
                if dqn is None:
                    selected = ppo.select_action(state, mask)
                    action_id = selected["action_id"]
                else:
                    action_id = _choose_dqn(dqn, state, mask, SPACE, epsilon=0.1)
                candidate_index = SPACE.decode(action_id).get("candidate_index")
                action_type = SPACE.decode(action_id).get("action_type")
                if action_type == "fill_input" and candidate_index in native_indexes:
                    result["native_select_fill_input_attempts"] += 1
                    result["contract_violations"] += 1
                next_raw, _, done, info, decoded_type = _step_v3(env, action_id, raw, SPACE)
                next_raw = _enrich_observation(env, next_raw)
                counts[decoded_type] += 1
                result["steps"] += 1
                result["action_error"] += int(bool(info.get("last_action_error")))
                if decoded_type == "select_option":
                    result["select_option_selections"] += 1
                    result["select_option_successes"] += int(bool(info.get("select_option")))
                result["handoffs"] += int(bool(info.get("handoff")))
                result["returns"] += int(bool(info.get("return_to_ppo")))
                raw = next_raw
                if done:
                    break
        finally:
            env.close()
    result["action_types"] = dict(counts)
    return result


def main() -> None:
    ppo = PPOAgent(635, 640)
    ppo.load(ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt")
    dqn = {}
    for seed in (42, 43, 44):
        network = FactorizedQNetwork(984, 20, 32)
        network.load_state_dict(torch.load(ROOT / f"artifacts/models/v3/rawd_v3_dqn_selectcov_seed{seed}.pt", map_location="cpu")["network_state_dict"])
        network.eval()
        dqn[seed] = network

    fixed = {
        "ppo": {"obs_dim": 635, "output_dim": 640, "finite": all(bool(torch.isfinite(value).all()) for value in ppo.policy.state_dict().values()), "checkpoint_sha256": sha256(ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt")},
        "dqn": {
            str(seed): {
                "obs_dim": 984,
                "type_head": 20,
                "candidate_head": [20, 32],
                "finite": all(bool(torch.isfinite(value).all()) for value in network.state_dict().values()),
                "checkpoint_sha256": sha256(ROOT / f"artifacts/models/v3/rawd_v3_dqn_selectcov_seed{seed}.pt"),
            }
            for seed, network in dqn.items()
        },
    }
    validation = {"ppo": [], "dqn": {str(seed): [] for seed in dqn}}
    for site_name, (site_id, url) in SITES.items():
        validation["ppo"].append(run_policy(site_name, site_id, url, ppo, None))
        for seed, network in dqn.items():
            validation["dqn"][str(seed)].append(run_policy(site_name, site_id, url, ppo, network))
    validation["ppo"].append(run_policy("DEV_V3_SELECT", *COVERAGE_SITES["DEV_V3_SELECT"], ppo, None))
    for seed, network in dqn.items():
        validation["dqn"][str(seed)].append(run_policy("DEV_V3_SELECT", *COVERAGE_SITES["DEV_V3_SELECT"], ppo, network))
    report = {
        "version": "RAWD_V3_SELECT_COVERAGE_VALIDATION",
        "fixed_forward": fixed,
        "validation": validation,
        "dimensions": {"ppo_obs": 635, "ppo_action": 640, "dqn_obs": 984, "dqn_type_head": 20, "dqn_candidate_head": [20, 32]},
        "gt_used": False,
        "excluded_sites_used": False,
    }
    output = ROOT / "artifacts/training_v3/select_coverage_run2/post_training_policy_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
