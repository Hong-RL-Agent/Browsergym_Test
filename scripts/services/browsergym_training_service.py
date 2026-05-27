"""Single-environment BrowserGym PPO training service."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs


class BrowserGymTrainingService:
    def __init__(
        self,
        site_id: str,
        base_url: str,
        episodes: int = 10,
        max_steps: int = 20,
        max_candidates: int = 32,
        headless: bool = True,
        output_dir: Optional[str] = None,
        model_output_path: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.site_id = site_id
        self.base_url = base_url
        self.episodes = episodes
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.headless = headless
        self.output_dir = Path(output_dir or f"artifacts/browsergym/{site_id}")
        self.model_output_path = Path(model_output_path or f"artifacts/models/{site_id}_browsergym_ppo.pt")
        self.seed = seed

        self.encoder = ObservationEncoder(max_candidates=max_candidates)
        self.action_space = ActionSpace(max_candidates=max_candidates)
        self.agent = PPOAgent(self.encoder.get_obs_dim(), self.action_space.get_action_dim())
        self.known_bugs = load_known_bugs(site_id)
        self.transition_log_path = self.output_dir / "rl_transition_log.jsonl"
        self.summary_path = self.output_dir / "training_summary.json"
        self.detected_bugs_path = self.output_dir / "detected_bugs.json"
        self.detected_bugs: List[Dict[str, Any]] = []
        self._detected_bug_keys: set[tuple[Any, ...]] = set()

    def train(self) -> Dict[str, Any]:
        _set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.transition_log_path.write_text("", encoding="utf-8")

        episode_rewards: List[float] = []
        total_steps = 0
        last_update: Dict[str, float] = {}

        for episode_index in range(1, self.episodes + 1):
            episode_id = f"EP-{episode_index:04d}"
            print(f"[train] starting {episode_id}")
            env = BrowserGymJAWSEnv(
                site_id=self.site_id,
                base_url=self.base_url,
                max_steps=self.max_steps,
                max_candidates=self.max_candidates,
                headless=self.headless,
            )
            buffer = RolloutBuffer()
            episode_reward = 0.0
            episode_anomaly_count = 0
            history: Dict[str, Any] = {
                "clicked_bids": set(),
                "last_action_key": None,
                "last_action_type": None,
                "episode_seen_anomalies": set(),
                "unmatched_anomaly_count": 0,
            }

            try:
                observation, _ = env.reset()
                done = False
                for step in range(1, self.max_steps + 1):
                    obs_vector = self.encoder.encode_observation(observation)
                    action_mask = self.action_space.build_action_mask(observation)
                    selected = self.agent.select_action(obs_vector, action_mask)
                    action_id = selected["action_id"]
                    action = self.action_space.decode(action_id)
                    action["action_id"] = action_id

                    next_observation, _, done, step_info = env.step(action_id)
                    action["failed"] = bool(step_info.get("action_error") or step_info.get("fallback_error"))
                    detector_info = {
                        "action": action,
                        "episode_seen_anomalies": history["episode_seen_anomalies"],
                        **step_info,
                    }
                    anomalies = detect_anomalies(observation, next_observation, detector_info)
                    known_matches = match_anomalies_to_known_bugs(anomalies, self.known_bugs)
                    anomalies = _merge_known_matches(anomalies, known_matches)
                    history["unmatched_anomaly_count"] += sum(1 for item in anomalies if not item.get("matched_bug_id"))
                    reward, reward_breakdown = calculate_autonomous_reward(
                        observation,
                        next_observation,
                        action,
                        anomalies,
                        known_matches,
                        history,
                    )

                    buffer.add(
                        obs_vector,
                        action_id,
                        selected["log_prob"],
                        reward,
                        done,
                        selected["value"],
                        action_mask,
                        step_info,
                    )
                    episode_reward += reward
                    total_steps += 1
                    episode_anomaly_count += len(anomalies)

                    self._record_detected_bugs(episode_id, step, anomalies, known_matches)
                    for anomaly in anomalies:
                        if anomaly.get("type") in {"layout-overlap", "duplicated-rendering"}:
                            history["episode_seen_anomalies"].add(anomaly.get("type"))
                    self._append_transition(
                        {
                            "site_id": self.site_id,
                            "episode_id": episode_id,
                            "step": step,
                            "state_summary": {
                                "url": observation.get("page_state", {}).get("url", ""),
                                "candidate_count": len(observation.get("candidate_elements", []) or []),
                            },
                            "action": action,
                            "reward": reward,
                            "reward_breakdown": reward_breakdown,
                            "anomalies": anomalies,
                            "done": done,
                        }
                    )

                    _update_history(history, observation, action)
                    observation = next_observation
                    if done:
                        break
            except Exception as exc:
                self._append_transition(
                    {
                        "site_id": self.site_id,
                        "episode_id": episode_id,
                        "step": len(buffer) + 1,
                        "state_summary": {"url": "", "candidate_count": 0},
                        "action": {"action_id": None, "action_type": "exception", "candidate_index": 0},
                        "reward": 0.0,
                        "reward_breakdown": {"final_reward": 0.0},
                        "anomalies": [{"type": "episode-exception", "confidence": 1.0, "evidence": {"error": str(exc)}}],
                        "done": True,
                    }
                )
                print(f"[train] {episode_id} stopped after exception: {exc}")
            finally:
                env.close()

            if len(buffer) > 0:
                last_value = 0.0 if done else self.agent.estimate_value(self.encoder.encode_observation(observation))
                buffer.compute_returns_and_advantages(last_value, self.agent.gamma, self.agent.gae_lambda)
                last_update = self.agent.update(buffer)

            episode_rewards.append(episode_reward)
            print(
                f"[train] finished {episode_id}: reward={episode_reward:.3f}, "
                f"anomalies={episode_anomaly_count}, loss={last_update}"
            )

        self.agent.save(self.model_output_path)
        summary = {
            "site_id": self.site_id,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "total_steps": total_steps,
            "average_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "detected_bug_count": len(self.detected_bugs),
            "known_bug_match_count": sum(1 for bug in self.detected_bugs if bug.get("matched_bug_id")),
            "matched_bug_ids": sorted(
                {
                    str(bug.get("matched_bug_id"))
                    for bug in self.detected_bugs
                    if bug.get("matched_bug_id")
                }
            ),
            "model_path": str(self.model_output_path),
            "transition_log_path": str(self.transition_log_path),
            "last_update": last_update,
        }
        self.summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        self.detected_bugs_path.write_text(json.dumps(_jsonable(self.detected_bugs), indent=2), encoding="utf-8")
        print(f"[train] summary: {self.summary_path}")
        print(f"[train] model: {self.model_output_path}")
        return summary

    def _append_transition(self, row: Dict[str, Any]) -> None:
        with self.transition_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def _record_detected_bugs(
        self,
        episode_id: str,
        step: int,
        anomalies: List[Dict[str, Any]],
        known_matches: List[Dict[str, Any]],
    ) -> None:
        match_by_type = {match.get("type"): match for match in known_matches}
        for anomaly in anomalies:
            match = match_by_type.get(anomaly.get("type"), {})
            matched_bug_id = anomaly.get("matched_bug_id") or match.get("matched_bug_id")
            evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), dict) else {}
            target_bid = evidence.get("target_bid") or evidence.get("bid") or evidence.get("target")
            key = (episode_id, anomaly.get("type"), matched_bug_id, target_bid)
            row = {
                "episode_id": episode_id,
                "step": step,
                "type": anomaly.get("type"),
                "confidence": anomaly.get("confidence", 0.0),
                "matched_bug_id": matched_bug_id,
                "evidence": evidence,
            }
            if key in self._detected_bug_keys:
                self._replace_if_better(row, matched_bug_id)
                continue
            self._detected_bug_keys.add(key)
            if matched_bug_id and self._replace_if_better(row, matched_bug_id):
                continue
            self.detected_bugs.append(row)

    def _replace_if_better(self, row: Dict[str, Any], matched_bug_id: Any) -> bool:
        if not matched_bug_id:
            return False
        for index, existing in enumerate(self.detected_bugs):
            if existing.get("matched_bug_id") == matched_bug_id:
                if float(row.get("confidence", 0.0) or 0.0) > float(existing.get("confidence", 0.0) or 0.0):
                    self.detected_bugs[index] = row
                return True
        return False


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
