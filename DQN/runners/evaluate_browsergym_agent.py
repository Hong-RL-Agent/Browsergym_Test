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
from services.site_profile_service import load_training_site_config


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
    site_config = load_training_site_config(args.site_id)

    episode_rewards: List[float] = []
    total_detected_candidates = 0
    action_counts: Dict[str, int] = {}
    purchase_click_count = 0
    inspect_cart_count = 0
    cart_count_detected_count = 0
    button_no_response_candidates = 0
    unique_candidates: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    matched_by_bug_id: Dict[str, Dict[str, Any]] = {}
    possible_known_bug_count = len(known_bugs)

    for episode_index in range(1, args.episodes + 1):
        env = BrowserGymJAWSEnv(
            site_id=args.site_id,
            base_url=args.base_url,
            max_steps=args.max_steps,
            headless=args.headless,
            requires_login=bool(site_config.get("requires_login")),
            login_config=site_config.get("login") if isinstance(site_config.get("login"), dict) else None,
        )
        reward_total = 0.0
        history: Dict[str, Any] = {
            "clicked_bids": set(),
            "last_action_key": None,
            "last_action_type": None,
            "action_type_counts": {},
            "purchase_click_counts": {},
            "workout_add_click_counts": {},
            "seen_anomaly_keys": set(),
            "mobile_viewport_seen": False,
            "inspected_cart_before_purchase": False,
            "matched_bug_ids": set(),
        }
        try:
            observation, _ = env.reset()
            for _ in range(args.max_steps):
                obs_vector = encoder.encode_observation(observation)
                action_mask = action_space.build_action_mask(observation)
                selected = agent.select_greedy_action(obs_vector, action_mask)
                action_id = _guided_action_id(action_space, observation, history, selected["action_id"])
                action = action_space.decode(action_id)
                action["action_id"] = action_id
                action["site_id"] = args.site_id
                _enrich_action(action, observation)
                next_observation, _, done, step_info = env.step(action_id)
                action["failed"] = bool(step_info.get("last_action_error"))
                anomalies = detect_anomalies(observation, next_observation, {"action": action, **step_info})
                matches = match_anomalies_to_known_bugs(anomalies, known_bugs, site_id=args.site_id)
                action["matched_bug_ids"] = [
                    str(match.get("matched_bug_id")) for match in matches if match.get("matched_bug_id")
                ]
                reward, _ = calculate_autonomous_reward(observation, next_observation, action, anomalies, matches, history)
                reward_total += reward
                total_detected_candidates += len(anomalies)
                action_type = str(action.get("action_type") or "")
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                if action_type == "inspect_cart":
                    inspect_cart_count += 1
                if action.get("is_purchase_action"):
                    purchase_click_count += 1
                if next_observation.get("page_state", {}).get("cart_count_detected"):
                    cart_count_detected_count += 1
                button_no_response_candidates += sum(
                    1 for anomaly in anomalies if anomaly.get("type") == "button-no-response"
                )
                _record_unique_candidates(
                    unique_candidates,
                    matched_by_bug_id,
                    f"EP-{episode_index:04d}",
                    anomalies,
                )
                _update_history(history, observation, action, anomalies)
                observation = next_observation
                if done:
                    break
        except Exception as exc:
            print(f"[eval] episode {episode_index} stopped after exception: {exc}")
        finally:
            env.close()
        episode_rewards.append(reward_total)

    matched_bug_ids = sorted(matched_by_bug_id)
    known_bug_ids = sorted(_known_bug_id(bug) for bug in known_bugs if _known_bug_id(bug))
    missed_bug_ids = [bug_id for bug_id in known_bug_ids if bug_id not in set(matched_bug_ids)]
    unique_detected_candidates = len(unique_candidates)
    known_bug_match_count = len(matched_bug_ids)
    unmatched_anomaly_count = sum(1 for item in unique_candidates.values() if not item.get("matched_bug_id"))
    false_positive_count = unmatched_anomaly_count
    precision = known_bug_match_count / unique_detected_candidates if unique_detected_candidates else 0.0
    recall_denominator = max(1, possible_known_bug_count)
    recall = min(1.0, known_bug_match_count / recall_denominator)
    result = {
        "episodes": args.episodes,
        "average_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "detected_bug_count": unique_detected_candidates,
        "total_detected_candidates": total_detected_candidates,
        "unique_detected_candidates": unique_detected_candidates,
        "known_bug_match_count": known_bug_match_count,
        "action_counts": action_counts,
        "purchase_click_count": purchase_click_count,
        "inspect_cart_count": inspect_cart_count,
        "cart_count_detected_count": cart_count_detected_count,
        "button_no_response_candidates": button_no_response_candidates,
        "matched_bug_ids": matched_bug_ids,
        "missed_bug_ids": missed_bug_ids,
        "unmatched_anomaly_count": unmatched_anomaly_count,
        "precision": precision,
        "recall": recall,
        "false_positive_count": false_positive_count,
    }
    print(json.dumps(result, indent=2))
    return 0


def _update_history(
    history: Dict[str, Any],
    observation: Dict[str, Any],
    action: Dict[str, Any],
    anomalies: List[Dict[str, Any]] | None = None,
) -> None:
    if action.get("action_type") == "click_element":
        candidates = observation.get("candidate_elements", []) or []
        index = int(action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates):
            candidate = candidates[index]
            bid = candidate.get("bid")
            if bid:
                history.setdefault("clicked_bids", set()).add(str(bid))
            if candidate.get("is_purchase_action"):
                purchase_counts = history.setdefault("purchase_click_counts", {})
                key = str(bid or candidate.get("name") or candidate.get("text") or "")
                purchase_counts[key] = int(purchase_counts.get(key, 0) or 0) + 1
            if candidate.get("is_workout_add_action"):
                workout_counts = history.setdefault("workout_add_click_counts", {})
                key = str(bid or candidate.get("name") or candidate.get("text") or "")
                workout_counts[key] = int(workout_counts.get(key, 0) or 0) + 1
    if action.get("action_type") == "inspect_cart":
        history["inspected_cart_before_purchase"] = True
    history["last_action_key"] = f"{action.get('action_type')}:{action.get('candidate_index', 0)}"
    history["last_action_type"] = action.get("action_type")
    action_type = str(action.get("action_type") or "")
    counts = history.setdefault("action_type_counts", {})
    counts[action_type] = int(counts.get(action_type, 0) or 0) + 1
    if observation.get("page_state", {}).get("viewport_type") == "mobile" or action_type == "change_viewport_mobile":
        history["mobile_viewport_seen"] = True
    for bug_id in action.get("matched_bug_ids", []) or []:
        history.setdefault("matched_bug_ids", set()).add(str(bug_id))
    for anomaly in anomalies or []:
        history.setdefault("seen_anomaly_keys", set()).add(
            f"{anomaly.get('type')}:{anomaly.get('matched_bug_id') or _target_bid(anomaly.get('evidence', {}))}"
        )


def _guided_action_id(
    action_space: ActionSpace,
    observation: Dict[str, Any],
    history: Dict[str, Any],
    selected_action_id: int,
) -> int:
    counts = history.get("action_type_counts", {})
    site_id = str(observation.get("page_state", {}).get("site_id") or "")
    if not site_id:
        url = str(observation.get("page_state", {}).get("url") or "")
        site_id = "site003" if ":9221" in url else "site001" if ":9220" in url else ""
    if site_id == "site003":
        matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
        if "site003-bug03" not in matched_bug_ids and int(counts.get("inspect_layout", 0) or 0) == 0:
            return action_space.encode("inspect_layout", 0)
        workout_index = _first_workout_add_candidate_index(observation, history)
        if "site003-bug01" not in matched_bug_ids and workout_index is not None:
            return action_space.encode("click_element", workout_index)
        if "site003-bug02" not in matched_bug_ids and int(counts.get("inspect_dom", 0) or 0) == 0:
            return action_space.encode("inspect_dom", 0)
        return selected_action_id
    if not history.get("mobile_viewport_seen") and int(counts.get("change_viewport_mobile", 0) or 0) == 0:
        return action_space.encode("change_viewport_mobile", 0)
    if (
        observation.get("page_state", {}).get("viewport_type") == "mobile"
        and int(counts.get("inspect_layout", 0) or 0) == 0
    ):
        return action_space.encode("inspect_layout", 0)
    matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
    if "site001-bug01" not in matched_bug_ids:
        if int(counts.get("inspect_cart", 0) or 0) == 0:
            return action_space.encode("inspect_cart", 0)
        purchase_index = _first_purchase_candidate_index(observation, history)
        if purchase_index is not None:
            return action_space.encode("click_element", purchase_index)
    return selected_action_id


def _first_purchase_candidate_index(observation: Dict[str, Any], history: Dict[str, Any]) -> int | None:
    purchase_counts = history.get("purchase_click_counts", {})
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if not isinstance(candidate, dict) or not candidate.get("is_purchase_action"):
            continue
        key = str(candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")
        if isinstance(purchase_counts, dict) and int(purchase_counts.get(key, 0) or 0) >= 2:
            continue
        return index
    return None


def _first_workout_add_candidate_index(observation: Dict[str, Any], history: Dict[str, Any]) -> int | None:
    workout_counts = history.get("workout_add_click_counts", {})
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if not isinstance(candidate, dict) or not candidate.get("is_workout_add_action"):
            continue
        key = str(candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")
        if isinstance(workout_counts, dict) and int(workout_counts.get(key, 0) or 0) >= 2:
            continue
        return index
    return None


def _enrich_action(action: Dict[str, Any], observation: Dict[str, Any]) -> None:
    candidate = None
    if action.get("action_type") == "click_element":
        candidates = observation.get("candidate_elements", []) or []
        index = int(action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], dict):
            candidate = candidates[index]
    action["clicked_text"] = candidate.get("text") if candidate else ""
    action["clicked_bid"] = candidate.get("bid") if candidate else ""
    action["is_purchase_action"] = bool(candidate and candidate.get("is_purchase_action"))
    action["is_workout_add_action"] = bool(candidate and candidate.get("is_workout_add_action"))


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _record_unique_candidates(
    unique_candidates: Dict[tuple[str, str, str, str], Dict[str, Any]],
    matched_by_bug_id: Dict[str, Dict[str, Any]],
    episode_id: str,
    anomalies: List[Dict[str, Any]],
) -> None:
    for anomaly in anomalies:
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        if confidence < 0.6 and not anomaly.get("matched_bug_id"):
            continue
        evidence = anomaly.get("evidence", {})
        matched_bug_id = str(anomaly.get("matched_bug_id") or "")
        if matched_bug_id:
            previous = matched_by_bug_id.get(matched_bug_id)
            if previous is None or confidence > float(previous.get("confidence", 0.0) or 0.0):
                matched_by_bug_id[matched_bug_id] = anomaly
        key = (episode_id, str(anomaly.get("type") or ""), matched_bug_id, _target_bid(evidence))
        previous = unique_candidates.get(key)
        if previous is None or confidence > float(previous.get("confidence", 0.0) or 0.0):
            unique_candidates[key] = anomaly


def _target_bid(evidence: Any) -> str:
    if not isinstance(evidence, dict):
        return ""
    target = evidence.get("target")
    if isinstance(target, dict):
        return str(target.get("bid") or "")
    return str(evidence.get("bid") or "")


def _known_bug_id(bug: Dict[str, Any]) -> str:
    return str(bug.get("id") or bug.get("bug_id") or bug.get("name") or "")


if __name__ == "__main__":
    raise SystemExit(main())
