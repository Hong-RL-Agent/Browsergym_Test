from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

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
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    encoder = ObservationEncoder()
    action_space = ActionSpace()
    agent = PPOAgent(encoder.get_obs_dim(), action_space.get_action_dim())
    agent.load(args.model_path)
    known_bugs = load_known_bugs(args.site_id)

    episode_rewards: List[float] = []
    total_detected_candidates = 0
    possible_known_bug_count = len(known_bugs)
    unique_candidates: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    matched_bug_ids: set[str] = set()

    for episode_index in range(1, args.episodes + 1):
        env = BrowserGymJAWSEnv(
            site_id=args.site_id,
            base_url=args.base_url,
            max_steps=args.max_steps,
            headless=args.headless,
        )
        reward_total = 0.0
        history: Dict[str, Any] = {
            "clicked_bids": set(),
            "last_action_key": None,
            "last_action_type": None,
            "episode_seen_anomalies": set(),
            "unmatched_anomaly_count": 0,
        }
        try:
            observation, _ = env.reset()
            for _ in range(args.max_steps):
                obs_vector = encoder.encode_observation(observation)
                action_mask = action_space.build_action_mask(observation)
                selected = agent.select_greedy_action(obs_vector, action_mask)
                action = action_space.decode(selected["action_id"])
                action["action_id"] = selected["action_id"]
                next_observation, _, done, step_info = env.step(selected["action_id"])
                action["failed"] = bool(step_info.get("action_error") or step_info.get("fallback_error"))
                detector_info = {
                    "action": action,
                    "episode_seen_anomalies": history["episode_seen_anomalies"],
                    **step_info,
                }
                anomalies = detect_anomalies(observation, next_observation, detector_info)
                matches = match_anomalies_to_known_bugs(anomalies, known_bugs)
                anomalies = _merge_known_matches(anomalies, matches)
                history["unmatched_anomaly_count"] += sum(1 for item in anomalies if not item.get("matched_bug_id"))
                reward, _ = calculate_autonomous_reward(observation, next_observation, action, anomalies, matches, history)
                reward_total += reward
                total_detected_candidates += len(anomalies)
                for anomaly in anomalies:
                    if anomaly.get("matched_bug_id"):
                        matched_bug_ids.add(str(anomaly["matched_bug_id"]))
                    key = _candidate_key(episode_index, anomaly)
                    existing = unique_candidates.get(key)
                    if not existing or float(anomaly.get("confidence", 0.0) or 0.0) > float(
                        existing.get("confidence", 0.0) or 0.0
                    ):
                        unique_candidates[key] = anomaly
                    if anomaly.get("type") in {"layout-overlap", "duplicated-rendering"}:
                        history["episode_seen_anomalies"].add(anomaly.get("type"))
                _update_history(history, observation, action)
                observation = next_observation
                if done:
                    break
        except Exception as exc:
            print(f"[eval] episode {episode_index} stopped after exception: {exc}")
        finally:
            env.close()
        episode_rewards.append(reward_total)

    unique_detected_candidates = len(unique_candidates)
    matched_unique_known_bug_count = len(matched_bug_ids)
    known_bug_match_count = sum(1 for anomaly in unique_candidates.values() if anomaly.get("matched_bug_id"))
    false_positive_count = sum(1 for anomaly in unique_candidates.values() if not anomaly.get("matched_bug_id"))
    unmatched_anomaly_count = total_detected_candidates - known_bug_match_count
    precision = known_bug_match_count / unique_detected_candidates if unique_detected_candidates else 0.0
    recall_denominator = max(1, possible_known_bug_count)
    recall = min(1.0, matched_unique_known_bug_count / recall_denominator)
    result = {
        "episodes": args.episodes,
        "average_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "detected_bug_count": unique_detected_candidates,
        "total_detected_candidates": total_detected_candidates,
        "unique_detected_candidates": unique_detected_candidates,
        "known_bug_match_count": known_bug_match_count,
        "matched_bug_ids": sorted(matched_bug_ids),
        "unmatched_anomaly_count": unmatched_anomaly_count,
        "false_positive_count": false_positive_count,
        "precision": precision,
        "recall": recall,
    }
    print(json.dumps(result, indent=2))
    return 0


def _update_history(history: Dict[str, Any], observation: Dict[str, Any], action: Dict[str, Any]) -> None:
    if action.get("action_type") == "click_element":
        candidates = observation.get("candidate_elements", []) or []
        index = int(action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates):
            bid = candidates[index].get("bid")
            if bid:
                history.setdefault("clicked_bids", set()).add(str(bid))
    history["last_action_key"] = f"{action.get('action_type')}:{action.get('candidate_index', 0)}"
    history["last_action_type"] = action.get("action_type")


def _candidate_key(episode_index: int, anomaly: Dict[str, Any]) -> tuple[Any, ...]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), dict) else {}
    return (
        episode_index,
        anomaly.get("type"),
        anomaly.get("matched_bug_id"),
        evidence.get("target_bid") or evidence.get("bid") or evidence.get("repeated_text"),
    )


def _merge_known_matches(
    anomalies: List[Dict[str, Any]],
    known_matches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    match_by_type = {match.get("type"): match for match in known_matches}
    merged: List[Dict[str, Any]] = []
    for anomaly in anomalies:
        item = dict(anomaly)
        match = match_by_type.get(item.get("type"))
        if match:
            item["matched_bug_id"] = match.get("matched_bug_id")
            item["match_reason"] = match.get("match_reason")
            item["match_score"] = match.get("match_score")
        merged.append(item)
    return merged


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
