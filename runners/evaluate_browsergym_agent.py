from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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
from services.policy_safe_metrics import (
    VERSION,
    build_metric_counts,
    compute_precision_recall_f1,
    utc_now_iso,
    write_comparison_note,
)
from services.site_profile_service import load_training_site_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-path", default="artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--eval-mode", choices=("unguided_eval", "guided_oracle_eval"), default="unguided_eval")
    parser.add_argument("--output", default="artifacts/evaluation/v3_policy_safe/evaluation_summary_single_site.json")
    parser.add_argument("--site-timeout-seconds", type=int, default=60)
    parser.add_argument("--episode-timeout-seconds", type=int, default=0)
    parser.add_argument("--reset-timeout-ms", type=int, default=30000)
    parser.add_argument("--reset-retry-count", type=int, default=0)
    parser.add_argument("--no-progress-patience", type=int, default=0)
    parser.add_argument("--use-memory-encoder", type=_parse_bool, default=False)
    parser.add_argument("--memory-encoder-type", default="gru")
    parser.add_argument("--memory-hidden-size", type=int, default=128)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    encoder = ObservationEncoder()
    action_space = ActionSpace()
    agent = PPOAgent(
        encoder.get_obs_dim(),
        action_space.get_action_dim(),
        use_memory_encoder=args.use_memory_encoder,
        memory_encoder_type=args.memory_encoder_type,
        memory_hidden_size=args.memory_hidden_size,
    )
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
    site_started_at = time.monotonic()
    reset_elapsed_values: List[int] = []
    reset_retry_used = 0
    reset_failure_count = 0
    reset_failure_reason = ""
    step_elapsed_values: List[int] = []
    episode_elapsed_values: List[int] = []
    early_stop_reasons: Dict[str, int] = {}
    timeout_count = 0
    completed_episodes = 0
    episode_memory_reset_count = 0
    memory_state_norm_values: List[float] = []

    for episode_index in range(1, args.episodes + 1):
        _raise_if_timeout(site_started_at, args.site_timeout_seconds, "site", args.site_id, episode_index)
        episode_started_at = time.monotonic()
        early_stop_reason = ""
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
            observation = {}
            for reset_attempt in range(0, max(0, args.reset_retry_count) + 1):
                try:
                    reset_started_at = time.monotonic()
                    observation, _ = env.reset()
                    reset_elapsed_ms = int((time.monotonic() - reset_started_at) * 1000)
                    reset_elapsed_values.append(reset_elapsed_ms)
                    if reset_elapsed_ms > args.reset_timeout_ms:
                        raise TimeoutError(f"env reset exceeded {args.reset_timeout_ms}ms for {args.site_id}")
                    reset_retry_used = max(reset_retry_used, reset_attempt)
                    break
                except Exception as reset_exc:
                    reset_failure_count += 1
                    reset_failure_reason = str(reset_exc)
                    if reset_attempt >= max(0, args.reset_retry_count):
                        raise
                    reset_retry_used = max(reset_retry_used, reset_attempt + 1)
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = BrowserGymJAWSEnv(
                        site_id=args.site_id,
                        base_url=args.base_url,
                        max_steps=args.max_steps,
                        headless=args.headless,
                        requires_login=bool(site_config.get("requires_login")),
                        login_config=site_config.get("login") if isinstance(site_config.get("login"), dict) else None,
                    )
            memory_state = agent.reset_memory_state()
            episode_memory_reset_count += 1
            for _ in range(args.max_steps):
                _raise_if_timeout(site_started_at, args.site_timeout_seconds, "site", args.site_id, episode_index)
                _raise_if_timeout(episode_started_at, args.episode_timeout_seconds, "episode", args.site_id, episode_index)
                before_signature = _state_signature(observation)
                before_unique_count = len(unique_candidates)
                before_matched_count = len(matched_by_bug_id)
                obs_vector = encoder.encode_observation(observation)
                action_mask = action_space.build_action_mask(observation)
                selected = agent.select_greedy_action(obs_vector, action_mask, memory_state=memory_state)
                memory_state = selected.get("memory_state", memory_state)
                memory_state_norm_values.append(float(selected.get("memory_state_norm", 0.0) or 0.0))
                action_id = int(selected["action_id"])
                action = action_space.decode(action_id)
                action["action_id"] = action_id
                action["site_id"] = args.site_id
                _enrich_action(action, observation)
                step_started_at = time.monotonic()
                next_observation, _, done, step_info = env.step(action_id)
                step_elapsed_values.append(int((time.monotonic() - step_started_at) * 1000))
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
                progress = _episode_progress_made(
                    before_unique_count=before_unique_count,
                    after_unique_count=len(unique_candidates),
                    before_matched_count=before_matched_count,
                    after_matched_count=len(matched_by_bug_id),
                    before_signature=before_signature,
                    after_signature=_state_signature(next_observation),
                )
                if progress:
                    history["no_progress_steps"] = 0
                else:
                    history["no_progress_steps"] = int(history.get("no_progress_steps", 0) or 0) + 1
                _update_history(history, observation, action, anomalies)
                observation = next_observation
                if args.no_progress_patience > 0 and int(history.get("no_progress_steps", 0) or 0) >= args.no_progress_patience:
                    early_stop_reason = "no_progress"
                    early_stop_reasons[early_stop_reason] = int(early_stop_reasons.get(early_stop_reason, 0) or 0) + 1
                    done = True
                if done:
                    break
            if not early_stop_reason and not done:
                early_stop_reason = "max_steps"
                early_stop_reasons[early_stop_reason] = int(early_stop_reasons.get(early_stop_reason, 0) or 0) + 1
            completed_episodes += 1
            episode_elapsed_values.append(int((time.monotonic() - episode_started_at) * 1000))
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                timeout_count += 1
                early_stop_reasons["timeout"] = int(early_stop_reasons.get("timeout", 0) or 0) + 1
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
    counts = build_metric_counts(
        known_bug_total=possible_known_bug_count,
        known_bug_matched_count=known_bug_match_count,
        unique_detected_anomaly_count=unique_detected_candidates,
    )
    metrics = compute_precision_recall_f1(
        true_positive_count=counts["true_positive_count"],
        false_positive_count=counts["false_positive_count"],
        false_negative_count=counts["false_negative_count"],
    )
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1_score"]
    if args.eval_mode != "unguided_eval":
        precision = None
        recall = None
        f1 = None
    result = {
        "version": VERSION,
        "eval_mode": args.eval_mode,
        "metric_source": "unguided_eval" if args.eval_mode == "unguided_eval" else "oracle/debug only; precision/recall/f1 disabled",
        "action_selection": "ppo_policy_only",
        "guided_action_used": False,
        "oracle_debug_only": args.eval_mode == "guided_oracle_eval",
        "model_path": args.model_path,
        "evaluated_at": utc_now_iso(),
        "site_count": 1,
        "episode_count": args.episodes,
        "memory_encoder_enabled": bool(args.use_memory_encoder),
        "memory_encoder_type": args.memory_encoder_type if args.use_memory_encoder else "",
        "memory_hidden_size": args.memory_hidden_size if args.use_memory_encoder else 0,
        "episode_memory_reset_count": episode_memory_reset_count,
        "memory_state_norm_mean": float(np.mean(memory_state_norm_values)) if memory_state_norm_values else 0.0,
        "episodes": args.episodes,
        "completed_episodes": completed_episodes,
        "partial": completed_episodes < args.episodes or bool(early_stop_reasons),
        "avg_step_elapsed_ms": _avg_int(step_elapsed_values),
        "max_step_elapsed_ms": max(step_elapsed_values) if step_elapsed_values else 0,
        "reset_elapsed_ms": _avg_int(reset_elapsed_values),
        "reset_timeout_ms": args.reset_timeout_ms,
        "reset_retry_count": args.reset_retry_count,
        "reset_retry_used": reset_retry_used,
        "reset_failure_count": reset_failure_count,
        "reset_failure_reason": reset_failure_reason,
        "site_elapsed_ms": int((time.monotonic() - site_started_at) * 1000),
        "episode_elapsed_ms": _avg_int(episode_elapsed_values),
        "early_stop_count": sum(int(value or 0) for value in early_stop_reasons.values()),
        "early_stop_reasons": early_stop_reasons,
        "timeout_count": timeout_count,
        "average_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "detected_bug_count": unique_detected_candidates,
        "detected_anomaly_count": total_detected_candidates,
        "total_detected_candidates": total_detected_candidates,
        "unique_detected_anomaly_count": unique_detected_candidates,
        "unique_detected_candidates": unique_detected_candidates,
        "known_bug_match_count": known_bug_match_count,
        "known_bug_total": counts["known_bug_total"],
        "known_bug_matched_count": counts["known_bug_matched_count"],
        "true_positive_count": counts["true_positive_count"],
        "false_positive_count": counts["false_positive_count"],
        "false_negative_count": counts["false_negative_count"],
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
        "f1": f1,
        "f1_score": f1,
    }
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        write_comparison_note(os.path.join(os.path.dirname(output_path), "comparison_note.json"))
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


def _raise_if_timeout(started_at: float, timeout_seconds: int, scope: str, site_id: str, episode_index: int) -> None:
    if timeout_seconds <= 0:
        return
    if time.monotonic() - started_at > timeout_seconds:
        raise TimeoutError(f"{scope} timeout exceeded {timeout_seconds}s for {site_id} episode={episode_index}")


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
        key = _canonical_anomaly_key(anomaly)
        previous = unique_candidates.get(key)
        if previous is None or confidence > float(previous.get("confidence", 0.0) or 0.0):
            unique_candidates[key] = anomaly


def _canonical_anomaly_key(anomaly: Dict[str, Any]) -> tuple[str, str, str, str]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), dict) else {}
    matched_bug_id = str(anomaly.get("matched_bug_id") or "")
    anomaly_type = str(anomaly.get("type") or "")
    if matched_bug_id:
        return ("matched", anomaly_type, matched_bug_id, "")
    target = str(
        evidence.get("selector_hint")
        or evidence.get("selector")
        or evidence.get("clicked_text")
        or evidence.get("candidate_text")
        or _target_bid(evidence)
    )
    bbox = evidence.get("bbox") or evidence.get("child_bbox") or []
    rounded_bbox = ",".join(str(int(round(float(value or 0) / 10.0) * 10)) for value in bbox[:4]) if isinstance(bbox, list) else ""
    return (str(anomaly.get("site_id") or ""), anomaly_type, target[:80].lower(), rounded_bbox)


def _state_signature(observation: Dict[str, Any]) -> tuple[str, str, int, int, int]:
    page_state = observation.get("page_state", {}) if isinstance(observation, dict) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, dict) else []
    url = str(page_state.get("url") or "") if isinstance(page_state, dict) else ""
    title = str(page_state.get("title") or "") if isinstance(page_state, dict) else ""
    text = str(page_state.get("text") or page_state.get("visible_text") or "") if isinstance(page_state, dict) else ""
    dom_node_count = int(page_state.get("dom_node_count") or 0) if isinstance(page_state, dict) else 0
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    return (url, title, len(text), dom_node_count, candidate_count)


def _episode_progress_made(
    *,
    before_unique_count: int,
    after_unique_count: int,
    before_matched_count: int,
    after_matched_count: int,
    before_signature: tuple[str, str, int, int, int],
    after_signature: tuple[str, str, int, int, int],
) -> bool:
    return (
        int(after_unique_count or 0) > int(before_unique_count or 0)
        or int(after_matched_count or 0) > int(before_matched_count or 0)
        or before_signature != after_signature
    )


def _avg_int(values: List[int]) -> int:
    return int(sum(values) / len(values)) if values else 0


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
