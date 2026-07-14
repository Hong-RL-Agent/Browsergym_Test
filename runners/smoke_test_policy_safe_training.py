from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Mapping

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.ppo_agent import PPOAgent
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.browsergym_training_service import _enrich_action, _update_history
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs
from services.site_profile_service import load_training_site_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--debug-action-mask", type=_parse_bool, default=False)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--model-path", default="artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    encoder = ObservationEncoder()
    action_space = ActionSpace()
    agent = PPOAgent(encoder.get_obs_dim(), action_space.get_action_dim())
    model_loaded = False
    if args.model_path and os.path.exists(args.model_path):
        agent.load(args.model_path)
        model_loaded = True

    site_config = load_training_site_config(args.site_id)
    env = BrowserGymJAWSEnv(
        site_id=args.site_id,
        base_url=args.base_url,
        max_steps=args.max_steps,
        headless=args.headless,
        requires_login=bool(site_config.get("requires_login")),
        login_config=site_config.get("login") if isinstance(site_config.get("login"), Mapping) else None,
    )
    history: Dict[str, Any] = {
        "clicked_bids": set(),
        "last_action_key": None,
        "last_action_type": None,
        "action_type_counts": {},
        "seen_anomaly_keys": set(),
    }
    known_bugs = None
    steps_completed = 0
    reward_total = 0.0
    done = False

    try:
        observation, _ = env.reset()
        _assert_candidate_and_mask(action_space, observation)
        for step in range(1, args.max_steps + 1):
            obs_vector = encoder.encode_observation(observation)
            action_mask = action_space.build_action_mask(observation)
            selected = agent.select_action(obs_vector, action_mask)
            action_id = int(selected["action_id"])
            stats = action_space.build_action_mask_stats(observation, action_mask, action_id)
            _assert_mask_stats(stats)
            if args.debug_action_mask:
                _log_action_mask_stats(args.site_id, step, stats)

            action = action_space.decode(action_id)
            action["action_id"] = action_id
            action["site_id"] = args.site_id
            _enrich_action(action, observation)
            next_observation, _, done, step_info = env.step(action_id)
            action["failed"] = bool(step_info.get("last_action_error"))
            anomalies = detect_anomalies(observation, next_observation, {"action": action, **step_info})
            if known_bugs is None:
                known_bugs = load_known_bugs(args.site_id)
            matches = match_anomalies_to_known_bugs(anomalies, known_bugs, site_id=args.site_id)
            action["matched_bug_ids"] = [
                str(match.get("matched_bug_id")) for match in matches if match.get("matched_bug_id")
            ]
            reward, reward_breakdown = calculate_autonomous_reward(
                observation,
                next_observation,
                action,
                anomalies,
                matches,
                history,
            )
            if not math.isfinite(float(reward_breakdown.get("reward_total", reward))):
                raise RuntimeError("reward_total is not finite")
            reward_total += float(reward)
            _update_history(history, observation, action, anomalies)
            observation = next_observation
            steps_completed = step
            if done:
                break
    finally:
        env.close()

    result = {
        "status": "ok",
        "site_id": args.site_id,
        "base_url": args.base_url,
        "model_loaded": model_loaded,
        "steps_completed": steps_completed,
        "done": done,
        "reward_total": reward_total,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _assert_candidate_and_mask(action_space: ActionSpace, observation: Mapping[str, Any]) -> None:
    candidates = observation.get("candidate_elements", [])
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    if candidate_count <= 0:
        raise RuntimeError("env.reset() succeeded but candidate_count is 0")
    mask = action_space.build_action_mask(observation)
    stats = action_space.build_action_mask_stats(observation, mask, int(np.flatnonzero(mask)[0]))
    _assert_mask_stats(stats)


def _assert_mask_stats(stats: Mapping[str, Any]) -> None:
    if int(stats.get("candidate_count", 0) or 0) <= 0:
        raise RuntimeError("candidate_count is 0")
    if int(stats.get("valid_action_count", 0) or 0) <= 0:
        raise RuntimeError("valid_action_count is 0")
    if int(stats.get("valid_click_count", 0) or 0) <= 0 and int(stats.get("valid_global_action_count", 0) or 0) <= 0:
        raise RuntimeError("both valid_click_count and valid_global_action_count are 0")


def _log_action_mask_stats(site_id: str, step: int, stats: Mapping[str, Any]) -> None:
    print(
        "[mask] "
        f"site={site_id} step={step} "
        f"candidates={stats.get('candidate_count', 0)} "
        f"valid={stats.get('valid_action_count', 0)} "
        f"click={stats.get('valid_click_count', 0)} "
        f"fill={stats.get('valid_fill_count', 0)} "
        f"global={stats.get('valid_global_action_count', 0)}"
    )
    print(
        "[action] "
        f"type={stats.get('selected_action_type', '')} "
        f"candidate={stats.get('selected_candidate_index', 0)} "
        f"role={stats.get('selected_candidate_role', '')} "
        f"tag={stats.get('selected_candidate_tag', '')} "
        f"visible={stats.get('selected_candidate_visible', False)} "
        f"enabled={stats.get('selected_candidate_enabled', False)} "
        f"clickable={stats.get('selected_candidate_clickable', False)}"
    )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
