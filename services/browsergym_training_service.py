"""Single-environment BrowserGym PPO training service."""

from __future__ import annotations

import json
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
from services.site_profile_service import build_site_profile, load_training_site_config, validate_site_identity


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
        load_model_path: Optional[str] = None,
        seed: int = 42,
        entropy_coef: float = 0.02,
        use_memory_encoder: bool = False,
        memory_encoder_type: str = "gru",
        memory_hidden_size: int = 128,
    ) -> None:
        self.site_id = site_id
        self.base_url = base_url
        self.episodes = episodes
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.headless = headless
        self.output_dir = Path(output_dir or f"artifacts/browsergym/v3_policy_safe/{site_id}")
        self.model_output_path = Path(model_output_path or f"artifacts/models/{site_id}_browsergym_ppo_v3_policy_safe.pt")
        self.load_model_path = Path(load_model_path) if load_model_path else None
        self.seed = seed
        self.entropy_coef = max(0.02, float(entropy_coef))
        self.use_memory_encoder = bool(use_memory_encoder)
        self.memory_encoder_type = str(memory_encoder_type or "gru")
        self.memory_hidden_size = int(memory_hidden_size)

        self.encoder = ObservationEncoder(max_candidates=max_candidates)
        self.action_space = ActionSpace(max_candidates=max_candidates)
        self.agent = PPOAgent(
            self.encoder.get_obs_dim(),
            self.action_space.get_action_dim(),
            entropy_coef=self.entropy_coef,
            use_memory_encoder=self.use_memory_encoder,
            memory_encoder_type=self.memory_encoder_type,
            memory_hidden_size=self.memory_hidden_size,
        )
        if self.load_model_path and self.load_model_path.exists():
            self.agent.load(self.load_model_path)
        self.known_bugs = load_known_bugs(site_id)
        self.site_config = load_training_site_config(site_id)
        self.site_profile = build_site_profile(
            site_id,
            self.known_bugs,
            exploration_profile=self.site_config.get("exploration_profile"),
        )
        self.site_profile.update(
            {
                "reward_mode": "signal_based",
                "use_known_bug_reward": False,
                "use_known_bug_for_training": False,
                "use_known_bug_for_evaluation": False,
                "training_uses_site_specific_bug_catalog": False,
                "policy_uses_bug_labels": False,
            }
        )
        self.transition_log_path = self.output_dir / "rl_transition_log.jsonl"
        self.summary_path = self.output_dir / "training_summary.json"
        self.detected_bugs_path = self.output_dir / "detected_bugs.json"
        self.detected_bugs: List[Dict[str, Any]] = []
        self._detected_bug_index: Dict[tuple[str, str, str, str], int] = {}
        self._matched_bug_index: Dict[str, int] = {}

    def train(self) -> Dict[str, Any]:
        _set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.transition_log_path.write_text("", encoding="utf-8")

        episode_rewards: List[float] = []
        total_steps = 0
        action_counts: Dict[str, int] = {}
        purchase_click_count = 0
        inspect_cart_count = 0
        cart_count_detected_count = 0
        button_no_response_candidates = 0
        last_update: Dict[str, float] = {}
        reward_signal_totals: Counter[str] = Counter()

        for episode_index in range(1, self.episodes + 1):
            episode_id = f"EP-{episode_index:04d}"
            print(f"[train] starting {episode_id}")
            env = BrowserGymJAWSEnv(
                site_id=self.site_id,
                base_url=self.base_url,
                max_steps=self.max_steps,
                max_candidates=self.max_candidates,
                headless=self.headless,
                site_profile=self.site_profile,
                requires_login=bool(self.site_config.get("requires_login")),
                login_config=self.site_config.get("login") if isinstance(self.site_config.get("login"), Mapping) else None,
            )
            buffer = RolloutBuffer()
            episode_reward = 0.0
            episode_anomaly_count = 0
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
            memory_state = self.agent.reset_memory_state()

            try:
                observation, _ = env.reset()
                identity = validate_site_identity(self.site_id, observation)
                for warning in identity.get("identity_warnings", []) or []:
                    print(warning)
                done = False
                for step in range(1, self.max_steps + 1):
                    _attach_action_history_to_observation(observation, history)
                    obs_vector = self.encoder.encode_observation(observation)
                    action_mask = self.action_space.build_action_mask(observation)
                    selected = self.agent.select_action(obs_vector, action_mask, memory_state=memory_state)
                    selected_memory_state = selected.get("memory_state", memory_state)
                    action_id = int(selected["action_id"])
                    action = self.action_space.decode(action_id)
                    action["action_id"] = action_id
                    action["site_id"] = self.site_id
                    _enrich_action(action, observation)

                    next_observation, _, done, step_info = env.step(action_id)
                    action["failed"] = bool(step_info.get("last_action_error"))
                    anomalies = detect_anomalies(
                        observation,
                        next_observation,
                        {"action": action, "site_profile": self.site_profile, **step_info},
                        site_profile=self.site_profile,
                    )
                    known_matches = (
                        match_anomalies_to_known_bugs(anomalies, self.known_bugs, site_id=self.site_id)
                        if bool(self.site_profile.get("use_known_bug_for_training", False))
                        else []
                    )
                    action["matched_bug_ids"] = [
                        str(match.get("matched_bug_id")) for match in known_matches if match.get("matched_bug_id")
                    ]
                    reward, reward_breakdown = calculate_autonomous_reward(
                        observation,
                        next_observation,
                        action,
                        anomalies,
                        known_matches,
                        history,
                        self.site_profile,
                    )
                    for key in (
                        "known_bug_reward_total",
                        "signal_reward_total",
                        "exploration_reward_total",
                        "ui_dom_signal_reward_total",
                        "console_runtime_signal_reward_total",
                        "network_api_signal_reward_total",
                        "cross_layer_signal_reward_total",
                        "security_signal_reward_total",
                        "repeated_penalty_total",
                        "signal_delta_reward_total",
                        "inspect_action_reward_total",
                        "functional_action_signal_reward_total",
                        "reward_functional_action_total",
                        "penalty_debug_meta_total",
                        "multi_signal_anomaly_count",
                        "console_error_count",
                        "runtime_exception_count",
                        "network_request_failed_count",
                        "api_4xx_count",
                        "api_5xx_count",
                        "api_timeout_count",
                        "api_ui_mismatch_count",
                        "security_signal_count",
                        "playwright_console_listener_enabled",
                        "playwright_pageerror_listener_enabled",
                        "playwright_network_listener_enabled",
                        "playwright_request_listener_enabled",
                        "playwright_response_listener_enabled",
                        "playwright_requestfailed_listener_enabled",
                        "functional_action_count",
                        "debug_meta_action_count",
                        "debug_meta_repeat_count",
                        "no_functional_action_episode_count",
                        "functional_action_signal_delta_count",
                        "functional_action_network_delta_count",
                    ):
                        reward_signal_totals[key] += float(reward_breakdown.get(key, 0.0) or 0.0)

                    buffer.add(
                        obs_vector,
                        action_id,
                        selected["log_prob"],
                        reward,
                        done,
                        selected["value"],
                        action_mask,
                        selected_memory_state,
                        step_info,
                    )
                    memory_state = selected_memory_state
                    episode_reward += reward
                    total_steps += 1
                    episode_anomaly_count += len(anomalies)
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

                    if bool(self.site_profile.get("use_known_bug_for_evaluation", False)):
                        self._record_detected_bugs(episode_id, step, anomalies, known_matches)
                    self._append_transition(
                        {
                            "site_id": self.site_id,
                            "episode_id": episode_id,
                            "step": step,
                            "state_summary": {
                                "url": observation.get("page_state", {}).get("url", ""),
                                "viewport_type": observation.get("page_state", {}).get("viewport_type", ""),
                                "cart_count": observation.get("page_state", {}).get("cart_count"),
                                "cart_count_detected": observation.get("page_state", {}).get("cart_count_detected"),
                                "candidate_count": len(observation.get("candidate_elements", []) or []),
                            },
                            "action": action,
                            "action_type": action_type,
                            "reward": reward,
                            "reward_breakdown": reward_breakdown,
                            "post_action_matches": {
                                "matched_bug_ids": action.get("matched_bug_ids", []),
                                "anomaly_catalog_bug_id_matches": [
                                    anomaly.get("evidence", {}).get("catalog_bug_id_matches", [])
                                    for anomaly in anomalies
                                ],
                            },
                            "anomalies": anomalies,
                            "done": done,
                        }
                    )

                    _update_history(history, observation, action, anomalies, after_observation=next_observation)
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
                last_value = 0.0 if done else self.agent.estimate_value(
                    self.encoder.encode_observation(observation),
                    memory_state=memory_state,
                )
                buffer.compute_returns_and_advantages(last_value, self.agent.gamma, self.agent.gae_lambda)
                last_update = self.agent.update(buffer)

            episode_rewards.append(episode_reward)
            print(
                f"[train] finished {episode_id}: reward={episode_reward:.3f}, "
                f"anomalies={episode_anomaly_count}, loss={last_update}"
            )

        self.agent.save(self.model_output_path)
        matched_bug_ids = sorted(
            {str(bug.get("matched_bug_id")) for bug in self.detected_bugs if bug.get("matched_bug_id")}
        )
        known_bug_ids = sorted(str(bug.get("bug_id") or bug.get("id") or bug.get("name") or "") for bug in self.known_bugs)
        known_bug_ids = [bug_id for bug_id in known_bug_ids if bug_id]
        missed_bug_ids = [bug_id for bug_id in known_bug_ids if bug_id not in set(matched_bug_ids)]
        unique_detected_candidates = len(self.detected_bugs)
        known_bug_match_count = len(matched_bug_ids)
        summary = {
            "site_id": self.site_id,
            "reward_mode": "signal_based",
            "use_known_bug_reward": False,
            "use_known_bug_for_training": False,
            "use_known_bug_for_evaluation": False,
            "training_uses_site_specific_bug_catalog": False,
            "policy_uses_bug_labels": False,
            "memory_encoder_enabled": self.use_memory_encoder,
            "memory_encoder_type": self.memory_encoder_type if self.use_memory_encoder else "",
            "memory_hidden_size": self.memory_hidden_size if self.use_memory_encoder else 0,
            "episode_memory_reset_count": self.episodes,
            "target_signal_types": list(self.site_profile.get("target_signal_types", []) or []),
            "observed_signal_types": _observed_signal_types_from_totals(reward_signal_totals),
            "missing_signal_types": sorted(
                set(str(item) for item in self.site_profile.get("target_signal_types", []) or [])
                - set(_observed_signal_types_from_totals(reward_signal_totals))
            ),
            "signal_type_coverage": _signal_type_coverage(
                self.site_profile.get("target_signal_types", []) or [],
                _observed_signal_types_from_totals(reward_signal_totals),
            ),
            "signal_collector_enabled": True,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "total_steps": total_steps,
            "average_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "detected_bug_count": unique_detected_candidates,
            "unique_detected_candidates": unique_detected_candidates,
            "known_bug_match_count": known_bug_match_count,
            "action_counts": action_counts,
            "purchase_click_count": purchase_click_count,
            "inspect_cart_count": inspect_cart_count,
            "cart_count_detected_count": cart_count_detected_count,
            "button_no_response_candidates": button_no_response_candidates,
            "known_bug_reward_total": float(reward_signal_totals.get("known_bug_reward_total", 0.0)),
            "signal_reward_total": float(reward_signal_totals.get("signal_reward_total", 0.0)),
            "signal_delta_reward_total": float(reward_signal_totals.get("signal_delta_reward_total", 0.0)),
            "inspect_action_reward_total": float(reward_signal_totals.get("inspect_action_reward_total", 0.0)),
            "functional_action_signal_reward_total": float(reward_signal_totals.get("functional_action_signal_reward_total", 0.0)),
            "functional_action_count": int(reward_signal_totals.get("functional_action_count", 0.0)),
            "first_functional_action_step": None,
            "unique_functional_action_type_count": int(
                sum(1 for key in ("click_element", "fill_input", "press_enter") if int(action_counts.get(key, 0) or 0) > 0)
            ),
            "unique_functional_target_count": int(reward_signal_totals.get("functional_action_count", 0.0)),
            "debug_meta_action_count": int(reward_signal_totals.get("debug_meta_action_count", 0.0)),
            "open_detail_panel_count": int(action_counts.get("open_detail_panel", 0) or 0),
            "click_retry_button_count": int(action_counts.get("click_retry_button", 0) or 0),
            "debug_meta_repeat_count": int(reward_signal_totals.get("debug_meta_repeat_count", 0.0)),
            "no_functional_action_episode_count": int(reward_signal_totals.get("no_functional_action_episode_count", 0.0)),
            "functional_action_signal_delta_count": int(reward_signal_totals.get("functional_action_signal_delta_count", 0.0)),
            "functional_action_network_delta_count": int(reward_signal_totals.get("functional_action_network_delta_count", 0.0)),
            "reward_functional_action_total": float(reward_signal_totals.get("reward_functional_action_total", 0.0)),
            "penalty_debug_meta_total": float(reward_signal_totals.get("penalty_debug_meta_total", 0.0)),
            "exploration_reward_total": float(reward_signal_totals.get("exploration_reward_total", 0.0)),
            "ui_dom_signal_reward_total": float(reward_signal_totals.get("ui_dom_signal_reward_total", 0.0)),
            "console_runtime_signal_reward_total": float(reward_signal_totals.get("console_runtime_signal_reward_total", 0.0)),
            "network_api_signal_reward_total": float(reward_signal_totals.get("network_api_signal_reward_total", 0.0)),
            "cross_layer_signal_reward_total": float(reward_signal_totals.get("cross_layer_signal_reward_total", 0.0)),
            "security_signal_reward_total": float(reward_signal_totals.get("security_signal_reward_total", 0.0)),
            "repeated_penalty_total": float(reward_signal_totals.get("repeated_penalty_total", 0.0)),
            "multi_signal_anomaly_count": int(reward_signal_totals.get("multi_signal_anomaly_count", 0.0)),
            "console_error_count": int(reward_signal_totals.get("console_error_count", 0.0)),
            "runtime_exception_count": int(reward_signal_totals.get("runtime_exception_count", 0.0)),
            "network_error_count": int(reward_signal_totals.get("network_request_failed_count", 0.0)),
            "api_4xx_count": int(reward_signal_totals.get("api_4xx_count", 0.0)),
            "api_5xx_count": int(reward_signal_totals.get("api_5xx_count", 0.0)),
            "api_timeout_count": int(reward_signal_totals.get("api_timeout_count", 0.0)),
            "api_ui_mismatch_count": int(reward_signal_totals.get("api_ui_mismatch_count", 0.0)),
            "security_signal_count": int(reward_signal_totals.get("security_signal_count", 0.0)),
            "playwright_console_listener_enabled": bool(reward_signal_totals.get("playwright_console_listener_enabled", 0.0)),
            "playwright_pageerror_listener_enabled": bool(reward_signal_totals.get("playwright_pageerror_listener_enabled", 0.0)),
            "playwright_network_listener_enabled": bool(reward_signal_totals.get("playwright_network_listener_enabled", 0.0)),
            "playwright_request_listener_enabled": bool(reward_signal_totals.get("playwright_request_listener_enabled", 0.0)),
            "playwright_response_listener_enabled": bool(reward_signal_totals.get("playwright_response_listener_enabled", 0.0)),
            "playwright_requestfailed_listener_enabled": bool(reward_signal_totals.get("playwright_requestfailed_listener_enabled", 0.0)),
            "matched_bug_ids": matched_bug_ids,
            "missed_bug_ids": missed_bug_ids,
            "precision": known_bug_match_count / unique_detected_candidates if unique_detected_candidates else 0.0,
            "recall": min(1.0, known_bug_match_count / max(1, len(known_bug_ids))),
            "model_path": str(self.model_output_path),
            "transition_log_path": str(self.transition_log_path),
            "last_update": last_update,
            "entropy_coef": self.agent.entropy_coef,
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
            confidence = float(anomaly.get("confidence", 0.0) or 0.0)
            if confidence < 0.6 and not anomaly.get("matched_bug_id"):
                continue
            match = match_by_type.get(anomaly.get("type"), {})
            matched_bug_id = anomaly.get("matched_bug_id") or match.get("matched_bug_id")
            record = {
                "episode_id": episode_id,
                "step": step,
                "type": anomaly.get("type"),
                "confidence": confidence,
                "matched_bug_id": matched_bug_id,
                "match_reason": anomaly.get("match_reason") or match.get("match_reason"),
                "evidence": anomaly.get("evidence", {}),
            }
            key = _canonical_detected_key(record)
            existing_index = self._detected_bug_index.get(key)
            if existing_index is not None:
                if confidence > float(self.detected_bugs[existing_index].get("confidence", 0.0) or 0.0):
                    self.detected_bugs[existing_index] = record
                continue
            if matched_bug_id:
                matched_key = str(matched_bug_id)
                matched_index = self._matched_bug_index.get(matched_key)
                if matched_index is not None:
                    if confidence > float(self.detected_bugs[matched_index].get("confidence", 0.0) or 0.0):
                        self.detected_bugs[matched_index] = record
                    continue
                self._matched_bug_index[matched_key] = len(self.detected_bugs)
            self._detected_bug_index[key] = len(self.detected_bugs)
            self.detected_bugs.append(record)


def _update_history(
    history: Dict[str, Any],
    observation: Dict[str, Any],
    action: Dict[str, Any],
    anomalies: Optional[List[Dict[str, Any]]] = None,
    after_observation: Optional[Mapping[str, Any]] = None,
) -> None:
    _update_login_flow_state(history, observation=observation, action=action, after_observation=after_observation)
    if action.get("action_type") in {"click_element", "click_submit", "fill_input", "press_enter"}:
        candidates = observation.get("candidate_elements", []) or []
        index = int(action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates):
            candidate = candidates[index]
            bid = candidate.get("bid")
            key = _candidate_key(candidate)
            if key:
                target_counts = history.setdefault("click_target_counts", {})
                if action.get("action_type") in {"click_element", "click_submit"} and isinstance(target_counts, dict):
                    target_counts[key] = int(target_counts.get(key, 0) or 0) + 1
                    history.setdefault("visited_targets", set()).add(key)
                    history.setdefault("visited_element_keys", set()).add(key)
                    element_counts = history.setdefault("element_key_click_counts", {})
                    if isinstance(element_counts, dict):
                        element_counts[key] = int(element_counts.get(key, 0) or 0) + 1
            if bid:
                if action.get("action_type") == "fill_input":
                    history.setdefault("filled_bids", set()).add(str(bid))
                    history.setdefault("filled_input_element_keys", set()).add(key)
                elif action.get("action_type") == "press_enter":
                    history.setdefault("pressed_enter_bids", set()).add(str(bid))
                else:
                    history.setdefault("clicked_bids", set()).add(str(bid))
            if candidate.get("is_purchase_action"):
                purchase_counts = history.setdefault("purchase_click_counts", {})
                purchase_counts[key] = int(purchase_counts.get(key, 0) or 0) + 1
            if candidate.get("is_workout_add_action"):
                workout_counts = history.setdefault("workout_add_click_counts", {})
                workout_counts[key] = int(workout_counts.get(key, 0) or 0) + 1
                history["workout_add_clicked"] = True
            if candidate.get("functional_priority_candidate") or candidate.get("functional_priority"):
                priority_counts = history.setdefault("functional_priority_click_counts", {})
                priority_counts[key] = int(priority_counts.get(key, 0) or 0) + 1
                semantic_type = str(candidate.get("semantic_action_type") or "")
                if semantic_type:
                    semantic_counts = history.setdefault("semantic_action_type_counts", {})
                    semantic_counts[semantic_type] = int(semantic_counts.get(semantic_type, 0) or 0) + 1
                    history.setdefault("clicked_semantic_action_types", set()).add(semantic_type)
            if action.get("action_type") == "click_element" and candidate.get("is_high_value_functional_candidate"):
                history["verification_pending_after_high_value_click"] = True
                history["high_value_click_pending_verification_count"] = int(
                    history.get("high_value_click_pending_verification_count", 0) or 0
                ) + 1
                history["high_value_click_without_verification_count"] = int(
                    history.get("high_value_click_without_verification_count", 0) or 0
                ) + 1
    if _has_workout_add_candidate(observation):
        history["workout_add_candidate_seen"] = True
    if action.get("action_type") == "inspect_cart":
        history["inspected_cart_before_purchase"] = True
    if action.get("action_type") in {"inspect_dom", "inspect_network", "inspect_console"} and history.get(
        "verification_pending_after_high_value_click"
    ):
        history["verification_action_after_high_value_click_count"] = int(
            history.get("verification_action_after_high_value_click_count", 0) or 0
        ) + 1
        history["high_value_click_verified_count"] = int(
            history.get("high_value_click_verified_count", 0) or 0
        ) + 1
        history["high_value_click_pending_verification_count"] = max(
            0,
            int(history.get("high_value_click_pending_verification_count", 0) or 0) - 1,
        )
        history["high_value_click_without_verification_count"] = max(
            0,
            int(history.get("high_value_click_without_verification_count", 0) or 0) - 1,
        )
        history["verification_pending_after_high_value_click"] = False
    history["last_action_key"] = f"{action.get('action_type')}:{action.get('candidate_index', 0)}"
    action_type = str(action.get("action_type") or "")
    if _is_functional_action_type(action_type):
        history["functional_action_count"] = int(history.get("functional_action_count", 0) or 0) + 1
        functional_counts = history.setdefault("functional_action_type_counts", {})
        if isinstance(functional_counts, dict):
            functional_counts[action_type] = int(functional_counts.get(action_type, 0) or 0) + 1
        target_signature = _policy_safe_target_signature(observation, action)
        if target_signature:
            history.setdefault("functional_target_signatures", set()).add(target_signature)
    if _is_debug_meta_action_type(action_type):
        history["debug_meta_action_count"] = int(history.get("debug_meta_action_count", 0) or 0) + 1
    previous_action_type = history.get("last_action_type")
    consecutive = history.setdefault("consecutive_action_type_counts", {})
    if previous_action_type == action_type:
        consecutive[action_type] = int(consecutive.get(action_type, 0) or 0) + 1
    else:
        consecutive[action_type] = 1
    history["last_action_type"] = action.get("action_type")
    counts = history.setdefault("action_type_counts", {})
    counts[action_type] = int(counts.get(action_type, 0) or 0) + 1
    target_signature = _policy_safe_target_signature(observation, action)
    if target_signature:
        history.setdefault("target_signatures", set()).add(target_signature)
        history.setdefault("visited_targets", set()).add(target_signature)
    action_signature = f"{action_type}:{int(action.get('candidate_index', 0) or 0)}:{target_signature}"
    action_signature = str(action.get("action_signature") or action_signature)
    signature_counts = history.setdefault("action_signature_counts", {})
    signature_counts[action_signature] = int(signature_counts.get(action_signature, 0) or 0) + 1
    if observation.get("page_state", {}).get("viewport_type") == "mobile" or action_type == "change_viewport_mobile":
        history["mobile_viewport_seen"] = True
    for bug_id in action.get("matched_bug_ids", []) or []:
        history.setdefault("matched_bug_ids", set()).add(str(bug_id))
    for anomaly in anomalies or []:
        seen = history.setdefault("seen_anomaly_keys", set())
        if anomaly.get("signature"):
            seen.add(str(anomaly.get("signature")))
        seen.add(f"{anomaly.get('type')}:{anomaly.get('matched_bug_id') or _target_bid(anomaly.get('evidence', {}))}")


def _guided_action_id(
    action_space: ActionSpace,
    observation: Dict[str, Any],
    history: Dict[str, Any],
    selected_action_id: int,
) -> int:
    counts = history.get("action_type_counts", {})
    infra_action = _guided_infra_action_id(action_space, observation, history)
    if infra_action is not None:
        return infra_action
    if _has_openended_interactive_candidate(observation):
        consecutive_counts = history.get("consecutive_action_type_counts", {})
        inspect_dom_count = int(counts.get("inspect_dom", 0) or 0) if isinstance(counts, dict) else 0
        click_count = int(counts.get("click_element", 0) or 0) if isinstance(counts, dict) else 0
        if click_count == 0:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
        if (
            inspect_dom_count >= 5
            or (
                isinstance(consecutive_counts, dict)
                and int(consecutive_counts.get("inspect_dom", 0) or 0) >= 2
            )
        ):
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
            return action_space.encode("inspect_layout", 0)
        if int(counts.get("inspect_layout", 0) or 0) == 0:
            return action_space.encode("inspect_layout", 0)
        return selected_action_id
    if not history.get("mobile_viewport_seen") and int(counts.get("change_viewport_mobile", 0) or 0) == 0:
        return action_space.encode("change_viewport_mobile", 0)
    if (
        observation.get("page_state", {}).get("viewport_type") == "mobile"
        and int(counts.get("inspect_layout", 0) or 0) == 0
    ):
        return action_space.encode("inspect_layout", 0)
    purchase_index = _first_purchase_candidate_index(observation, history)
    if purchase_index is not None:
        return action_space.encode("click_element", purchase_index)
    return selected_action_id


def _guided_infra_action_id(
    action_space: ActionSpace,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> Optional[int]:
    infra = observation.get("infra_signals", {}) if isinstance(observation, Mapping) else {}
    if not isinstance(infra, Mapping):
        return None
    try:
        port = int(infra.get("port") or 0)
    except (TypeError, ValueError):
        return None
    if not 9000 <= port <= 9100:
        return None
    counts = history.get("action_type_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    for action_type in (
        "inspect_port_status",
        "inspect_server_health",
        "inspect_latency",
        "inspect_server_logs",
        "inspect_runtime_metrics",
    ):
        if int(counts.get(action_type, 0) or 0) == 0:
            return action_space.encode(action_type, 0)
    return None


def _first_visible_input_index(observation: Mapping[str, Any], history: Mapping[str, Any]) -> Optional[int]:
    filled = set(history.get("filled_bids", set()) or set())
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if not isinstance(candidate, Mapping) or not candidate.get("is_form_field"):
            continue
        if _safe_visibility(candidate) <= 0.0:
            continue
        bid = str(candidate.get("bid") or "")
        if bid and bid in filled:
            continue
        return index
    return None


def _first_purchase_candidate_index(observation: Dict[str, Any], history: Dict[str, Any]) -> Optional[int]:
    purchase_counts = history.get("purchase_click_counts", {})
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if not isinstance(candidate, dict) or not candidate.get("is_purchase_action"):
            continue
        key = _candidate_key(candidate)
        if isinstance(purchase_counts, dict) and int(purchase_counts.get(key, 0) or 0) >= 2:
            continue
        return index
    return None


def _first_workout_add_candidate_index(observation: Dict[str, Any], history: Dict[str, Any]) -> Optional[int]:
    workout_counts = history.get("workout_add_click_counts", {})
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if not isinstance(candidate, dict) or not candidate.get("is_workout_add_action"):
            continue
        if not candidate.get("clickable") or _safe_visibility(candidate) <= 0.0:
            continue
        key = str(candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")
        if isinstance(workout_counts, dict) and int(workout_counts.get(key, 0) or 0) >= 2:
            continue
        return index
    return None


def _first_openended_candidate_index(observation: Dict[str, Any], history: Dict[str, Any]) -> Optional[int]:
    click_counts = history.get("element_key_click_counts", history.get("click_target_counts", {}))
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    ranked: List[tuple[float, int]] = []
    for index, candidate in enumerate(candidates[:32]):
        if not isinstance(candidate, dict):
            continue
        if candidate.get("is_form_field"):
            continue
        if not candidate.get("is_interactive") and not candidate.get("clickable"):
            continue
        if _safe_visibility(candidate) <= 0.0:
            continue
        key = _candidate_key(candidate)
        if isinstance(click_counts, dict) and int(click_counts.get(key, 0) or 0) >= 2:
            continue
        score = float(candidate.get("openended_action_priority", 0.0) or 0.0)
        score += 3.0 if candidate.get("functional_priority_candidate") or candidate.get("functional_priority") else 0.0
        score += 2.0 if str(candidate.get("semantic_action_type") or "") == "workout_add" else 0.0
        score += 1.0 if candidate.get("is_high_value_functional_candidate") else 0.0
        score += 1.0 if candidate.get("openended_keyword_matches") else 0.0
        score += 0.5 if candidate.get("clickable") else 0.0
        score += 0.8 if str(candidate.get("role") or "").lower() in {"button", "link", "menuitem"} else 0.0
        score += 0.6 if str(candidate.get("tag") or "").lower() in {"button", "a"} else 0.0
        score += 0.5 if any(candidate.get(flag) for flag in ("is_login_related", "is_cart_related", "is_checkout_related", "is_search_related", "is_filter_related", "is_submit_related")) else 0.0
        score -= 0.2 if candidate.get("is_form_field") else 0.0
        ranked.append((score, index))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def _has_openended_interactive_candidate(observation: Dict[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, dict)
        and bool(candidate.get("is_interactive") or candidate.get("clickable"))
        and _safe_visibility(candidate) > 0.0
        for candidate in candidates
    )


def _has_workout_add_candidate(observation: Dict[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, dict) and bool(candidate.get("is_workout_add_action")) for candidate in candidates
    )


def _safe_visibility(candidate: Mapping[str, Any]) -> float:
    if "visibility" in candidate and candidate.get("visibility") is not None:
        try:
            value = float(candidate.get("visibility") or 0.0)
        except (TypeError, ValueError):
            value = 1.0
    elif "visible" in candidate:
        value = 1.0 if bool(candidate.get("visible")) else 0.0
    else:
        value = 1.0
    if value != value:
        return 1.0
    return max(0.0, min(1.0, value))


def _attach_action_history_to_observation(observation: Mapping[str, Any], history: Mapping[str, Any]) -> None:
    if not isinstance(observation, dict):
        return
    _update_login_flow_state(history, observation=observation, action=None, after_observation=None)
    obs_history = observation.setdefault("history", {})
    if not isinstance(obs_history, dict):
        obs_history = {}
        observation["history"] = obs_history
    counts = history.get("action_type_counts", {})
    obs_history["action_type_counts"] = dict(counts) if isinstance(counts, Mapping) else {}
    obs_history["login_flow"] = dict(history.get("login_flow", {}) or {})


def _update_login_flow_state(
    history: Dict[str, Any],
    *,
    observation: Mapping[str, Any],
    action: Mapping[str, Any] | None,
    after_observation: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    candidates = candidates if isinstance(candidates, list) else []
    has_email = any(isinstance(candidate, Mapping) and _is_email_text_candidate(candidate) for candidate in candidates)
    has_password = any(isinstance(candidate, Mapping) and bool(candidate.get("is_password")) for candidate in candidates)
    has_submit = any(isinstance(candidate, Mapping) and bool(candidate.get("is_submit")) for candidate in candidates)
    state = history.setdefault("login_flow", {})
    if not isinstance(state, dict):
        state = {}
        history["login_flow"] = state
    state["has_email_or_text_input"] = bool(state.get("has_email_or_text_input") or has_email)
    state["has_password_input"] = bool(state.get("has_password_input") or has_password)
    state["has_submit_button"] = bool(state.get("has_submit_button") or has_submit)
    state["has_login_form"] = bool(
        state.get("has_login_form")
        or (state["has_email_or_text_input"] and state["has_password_input"] and state["has_submit_button"])
    )
    if action:
        action_type = str(action.get("action_type") or "")
        index = int(action.get("candidate_index", 0) or 0)
        candidate = candidates[index] if 0 <= index < len(candidates) and isinstance(candidates[index], Mapping) else {}
        if action_type == "fill_input" and candidate:
            if candidate.get("is_password"):
                state["password_input_filled"] = True
                history.setdefault("filled_password_element_keys", set()).add(_candidate_key(candidate))
            elif _is_email_text_candidate(candidate):
                state["email_input_filled"] = True
                history.setdefault("filled_email_element_keys", set()).add(_candidate_key(candidate))
            state["login_flow_attempted"] = True
        if action_type == "click_submit" and candidate and candidate.get("is_submit"):
            state["submit_clicked"] = True
            state["login_flow_attempted"] = True
            state["login_flow_status"] = "submitted"
        if state.get("submit_clicked") and action_type in {"inspect_network", "inspect_console", "inspect_dom", "check_login_state"}:
            state["submit_result_checked"] = True
        if after_observation and state.get("submit_clicked"):
            before_url = str((observation.get("page_state", {}) or {}).get("url") or "")
            after_url = str((after_observation.get("page_state", {}) or {}).get("url") or "")
            if before_url and after_url and before_url != after_url:
                state["submit_result_checked"] = True
            if _dom_state_hash(observation) != _dom_state_hash(after_observation):
                state["submit_result_checked"] = True
    if state.get("submit_clicked") and state.get("submit_result_checked"):
        state["login_flow_completed"] = True
        state["login_flow_status"] = "verified"
    elif state.get("submit_clicked"):
        state["login_flow_status"] = "submitted"
    elif state.get("login_flow_attempted"):
        state["login_flow_status"] = "in_progress"
    elif state.get("has_login_form"):
        state["login_flow_status"] = "not_started"
    else:
        state["login_flow_status"] = "not_applicable"
    state["required_actions_remaining"] = _login_required_actions_remaining(state)
    return state


def _login_required_actions_remaining(state: Mapping[str, Any]) -> list[str]:
    if not bool(state.get("has_login_form")):
        return []
    remaining: list[str] = []
    if bool(state.get("has_email_or_text_input")) and not bool(state.get("email_input_filled")):
        remaining.append("fill_email_or_text_input")
    if bool(state.get("has_password_input")) and not bool(state.get("password_input_filled")):
        remaining.append("fill_password_input")
    if bool(state.get("has_submit_button")) and not bool(state.get("submit_clicked")):
        remaining.append("click_submit")
    if bool(state.get("submit_clicked")) and not bool(state.get("submit_result_checked")):
        remaining.append("verify_submit_result")
    return remaining


def _is_email_text_candidate(candidate: Mapping[str, Any]) -> bool:
    if not bool(candidate.get("fillable")):
        return False
    if bool(candidate.get("is_password")):
        return False
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ("name", "text", "placeholder", "aria_label", "title", "id")
    ).lower()
    return bool(
        input_type in {"", "text", "email", "search"}
        or role in {"textbox", "searchbox"}
        or tag == "textarea"
        or any(token in text for token in ("email", "username", "user", "login"))
    )


def _enrich_action(action: Dict[str, Any], observation: Dict[str, Any]) -> None:
    candidate = None
    if action.get("action_type") in {"click_element", "click_submit", "fill_input", "press_enter"}:
        candidates = observation.get("candidate_elements", []) or []
        index = int(action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], dict):
            candidate = candidates[index]
    if action.get("action_type") == "fill_input":
        action["input_text"] = _input_text_for_candidate(candidate, int(action.get("candidate_index", 0) or 0))
    action["clicked_text"] = candidate.get("text") if candidate else ""
    action["clicked_bid"] = candidate.get("bid") if candidate else ""
    action["action_bid"] = candidate.get("bid") if candidate else ""
    action["action_text"] = candidate.get("text") or candidate.get("name") if candidate else ""
    action["action_element_key"] = _candidate_key(candidate)
    action["is_purchase_action"] = bool(candidate and candidate.get("is_purchase_action"))
    action["is_workout_add_action"] = bool(candidate and candidate.get("is_workout_add_action"))
    action["functional_priority_candidate"] = bool(candidate and (candidate.get("functional_priority_candidate") or candidate.get("functional_priority")))
    action["semantic_action_type"] = str(candidate.get("semantic_action_type") or "") if candidate else ""
    action["action_semantic_type"] = action["semantic_action_type"]
    action["target_selector"] = str(candidate.get("selector") or candidate.get("locator") or candidate.get("selector_hint") or "") if candidate else ""
    action["is_high_value_functional_candidate"] = bool(candidate and candidate.get("is_high_value_functional_candidate"))
    action["is_low_value_generic_candidate"] = bool(candidate and candidate.get("is_low_value_generic_candidate"))
    action["dom_state_hash"] = _dom_state_hash(observation)
    action["action_signature"] = _action_signature_from_parts(
        action_type=str(action.get("action_type") or ""),
        element_key=str(action.get("action_element_key") or ""),
        candidate_index=int(action.get("candidate_index", 0) or 0),
    )


def _policy_safe_target_signature(observation: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    action_type = str(action.get("action_type") or "")
    if action_type in {"inspect_dom", "inspect_network", "inspect_console", "inspect_layout", "change_viewport_mobile", "finish_episode"}:
        return action_type
    candidates = observation.get("candidate_elements", []) or []
    candidate = None
    index = int(action.get("candidate_index", 0) or 0)
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        candidate = candidates[index]
    target = ""
    if candidate:
        target = _candidate_key(candidate)
    if not target:
        target = str(action.get("clicked_bid") or action.get("clicked_text") or action.get("target_text") or index)
    return f"{action_type}:{target}"
    action["is_interactive"] = bool(candidate and candidate.get("is_interactive"))
    action["openended_keyword_matches"] = list(candidate.get("openended_keyword_matches", [])) if candidate else []


def _candidate_key(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    return str(candidate.get("element_key") or candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")


def _action_signature_from_parts(*, action_type: str, element_key: str, candidate_index: int) -> str:
    if element_key:
        return f"{action_type}::{element_key}"
    return f"{action_type}::candidate-{candidate_index}"


def _dom_state_hash(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    payload = {
        "url": str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        "title": str(page_state.get("title") or "") if isinstance(page_state, Mapping) else "",
        "text_length": int(page_state.get("page_text_length") or len(str(page_state.get("page_text") or ""))) if isinstance(page_state, Mapping) else 0,
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _is_functional_action_type(action_type: str) -> bool:
    return action_type in {
        "click_element",
        "fill_input",
        "press_enter",
        "submit_form",
        "click_button",
        "navigate_link",
        "select_option",
        "trigger_api_action",
    }


def _is_debug_meta_action_type(action_type: str) -> bool:
    return action_type in {
        "open_detail_panel",
        "click_retry_button",
        "scroll_down",
        "scroll_up",
        "inspect_dom",
        "inspect_layout",
        "inspect_network",
        "inspect_console",
        "inspect_cart",
        "click_trigger_button",
        "click_recovery_button",
    }


def _input_text_for_candidate(candidate: Optional[Mapping[str, Any]], candidate_index: int = 0) -> str:
    if not candidate:
        return "test"
    haystack = " ".join(
        str(candidate.get(key) or "")
        for key in (
            "text",
            "name",
            "aria_label",
            "title",
            "id",
            "class_name",
        )
    ).lower()
    if "sparse" in haystack:
        return "sparse"
    if "async" in haystack:
        return "ASYNC_TEST"
    if "hang" in haystack:
        return "HANG_TEST"
    if "forbidden" in haystack or "403" in haystack:
        return "forbidden"
    if "search" in haystack:
        return "sparse"
    if "password" in haystack:
        return "commerce123"
    if "email" in haystack or "customer" in haystack or "username" in haystack:
        return "customer@smart.com"
    if candidate_index <= 2:
        return "admin@smart.com"
    if candidate_index <= 3:
        return "admin123"
    return "test"


def _canonical_detected_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    evidence = record.get("evidence", {}) if isinstance(record.get("evidence"), Mapping) else {}
    matched_bug_id = str(record.get("matched_bug_id") or "")
    anomaly_type = str(record.get("type") or "")
    if matched_bug_id:
        return ("matched", anomaly_type, matched_bug_id, "")
    catalog_ids = evidence.get("catalog_bug_id_matches", []) if isinstance(evidence.get("catalog_bug_id_matches"), list) else []
    primary_catalog = str(catalog_ids[0]) if catalog_ids else ""
    if anomaly_type == "layout-overflow":
        section = ",".join(str(item) for item in evidence.get("section_keyword_matches", []) or [])
        return ("layout", primary_catalog, section, "")
    if anomaly_type == "empty-state-rendering":
        section = ",".join(str(item) for item in evidence.get("section_keyword_matches", []) or [])
        return ("empty", primary_catalog, section, "")
    if anomaly_type == "button-no-response":
        target = str(evidence.get("clicked_data_bug_id") or evidence.get("clicked_text") or evidence.get("clicked_name") or _target_bid(evidence))
        return ("button", primary_catalog, _normalize_text(target), "")
    selector = str(evidence.get("selector_hint") or evidence.get("selector") or evidence.get("data_bug_id") or "")
    text = _normalize_text(str(evidence.get("candidate_text") or evidence.get("clicked_text") or _target_bid(evidence)))[:80]
    return (anomaly_type, primary_catalog, selector or text, "")


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _observed_signal_types_from_totals(totals: Mapping[str, Any]) -> List[str]:
    observed = []
    if int(totals.get("multi_signal_anomaly_count", 0) or 0):
        observed.extend(["ui-dom", "interaction", "layout"])
    if int(totals.get("console_error_count", 0) or 0):
        observed.append("console-error")
    if int(totals.get("runtime_exception_count", 0) or 0):
        observed.append("runtime-exception")
    if int(totals.get("network_request_failed_count", 0) or totals.get("network_error_count", 0) or 0):
        observed.append("network-error")
    if int(totals.get("api_4xx_count", 0) or 0):
        observed.append("api-4xx")
    if int(totals.get("api_5xx_count", 0) or 0):
        observed.append("api-5xx")
    if int(totals.get("api_timeout_count", 0) or 0):
        observed.append("api-timeout")
    if int(totals.get("api_ui_mismatch_count", 0) or 0):
        observed.append("api-ui-mismatch")
    if int(totals.get("security_signal_count", 0) or 0):
        observed.extend(["auth-permission-anomaly", "sensitive-data-exposure", "token-exposure"])
    return sorted(set(observed))


def _signal_type_coverage(targets: Any, observed: Any) -> float:
    target_set = {str(item) for item in targets or [] if item}
    observed_set = {str(item) for item in observed or [] if item}
    if not target_set:
        return round(len(observed_set) / 5.0, 4) if observed_set else 0.0
    return round(len(target_set & observed_set) / len(target_set), 4)


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


def _target_bid(evidence: Any) -> str:
    if not isinstance(evidence, dict):
        return ""
    target = evidence.get("target")
    if isinstance(target, dict):
        return str(target.get("bid") or "")
    return str(evidence.get("bid") or "")
