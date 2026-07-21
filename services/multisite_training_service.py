"""Shared PPO training across multiple BrowserGym sites."""

from __future__ import annotations

import json
import random
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import (
    apply_fallback_reward_policy,
    calculate_autonomous_reward,
    fallback_curriculum_penalty,
)
from services.browsergym_training_service import (
    _enrich_action,
    _target_bid,
    _update_login_flow_state,
    _update_history,
)
from services.episode_csv_logger import EpisodeCsvLogger, infer_run_id
from services.infra_anomaly_detection_service import detect_infra_anomalies
from services.infra_reward_service import calculate_infra_reward
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs
from services.policy_safe_metrics import VERSION
from services.site_profile_service import build_site_profile, validate_site_identity


class MultiSiteTrainingService:
    def __init__(
        self,
        config_path: str | Path,
        total_updates: int = 20,
        episodes_per_site: int = 1,
        max_steps: int = 25,
        max_candidates: int = 32,
        seed: int = 42,
        headless: bool = True,
        load_model_path: str | Path | None = None,
        save_model_path: str | Path | None = None,
        entropy_coef: float = 0.02,
        strict_site_validation: bool = False,
        enable_csv_logging: bool = True,
        csv_log_dir: str | Path = "artifacts/training_logs/v3_policy_safe",
        log_observation_detail: bool = True,
        log_action_space: bool = True,
        log_raw_json: bool = False,
        run_id: str | None = None,
        enable_guided_actions: bool = False,
        debug_action_mask: bool = False,
        use_memory_encoder: bool = False,
        memory_encoder_type: str = "gru",
        memory_hidden_size: int = 128,
        fresh_start: bool | None = None,
        resume: bool | None = None,
        fallback_penalty: float | None = None,
        fallback_reward_cap_enabled: bool | None = None,
        fallback_reward_cap_value: float | None = None,
        fallback_curriculum_enabled: bool | None = None,
        fail_on_episode_exception: bool | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = _read_json_dict(self.config_path)
        self.blind_url_training = _is_blind_url_training_config(self.config)
        _reject_known_bug_reward_for_training(self.config)
        self.policy_safety_audit = _policy_safety_audit(self.config, self.blind_url_training)
        self.configured_sites = _configured_sites(self.config.get("sites", []))
        self.sites = _validate_sites(self.config.get("sites", []))
        self.configured_site_ids = [str(site.get("site_id") or "") for site in self.configured_sites]
        self.enabled_site_ids = [str(site.get("site_id") or "") for site in self.sites]
        self.disabled_site_ids = [
            str(site.get("site_id") or "")
            for site in self.configured_sites
            if site.get("enabled") is False
        ]
        self.output_dir = Path(str(self.config.get("output_dir") or "artifacts/multisite/v3_policy_safe"))
        configured_model_path = self.config.get("shared_model_path") or "artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt"
        self.shared_model_path = Path(save_model_path or configured_model_path)
        self.load_model_path = Path(load_model_path) if load_model_path else None
        self.resume = bool(self.config.get("resume", False) if resume is None else resume)
        self.fresh_start = bool(self.config.get("fresh_start", not self.resume) if fresh_start is None else fresh_start)
        if self.resume:
            self.fresh_start = False
        if self.fresh_start:
            self.resume = False
        self.checkpoint_loaded = False
        self.checkpoint_path = ""
        self.total_updates = int(total_updates)
        self.episodes_per_site = int(episodes_per_site)
        self.max_steps = int(max_steps)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)
        self.headless = bool(headless)
        self.entropy_coef = max(0.02, float(entropy_coef))
        self.strict_site_validation = bool(strict_site_validation)
        self.enable_csv_logging = bool(enable_csv_logging)
        self.csv_log_dir = Path(csv_log_dir)
        self.log_observation_detail = bool(log_observation_detail)
        self.log_action_space = bool(log_action_space)
        self.log_raw_json = bool(log_raw_json)
        self.run_id = run_id or str(self.config.get("run_id") or self.config.get("config_name") or "") or infer_run_id(
            self.config,
            self.config_path,
            fallback_prefix="training",
        )
        self.batch_id = self.config_path.stem
        self.enable_guided_actions = bool(enable_guided_actions or self.config.get("enable_guided_actions", False))
        self.debug_action_mask = bool(debug_action_mask or self.config.get("debug_action_mask", False))
        self.use_memory_encoder = bool(use_memory_encoder or self.config.get("use_memory_encoder", False))
        self.memory_encoder_type = str(self.config.get("memory_encoder_type") or memory_encoder_type or "gru")
        self.memory_hidden_size = int(self.config.get("memory_hidden_size", memory_hidden_size or 128))
        self.fallback_penalty = float(
            self.config.get("fallback_penalty", 0.25) if fallback_penalty is None else fallback_penalty
        )
        self.fallback_reward_cap_enabled = bool(
            self.config.get("fallback_reward_cap_enabled", True)
            if fallback_reward_cap_enabled is None
            else fallback_reward_cap_enabled
        )
        self.fallback_reward_cap_value = float(
            self.config.get("fallback_reward_cap_value", 0.0)
            if fallback_reward_cap_value is None
            else fallback_reward_cap_value
        )
        self.fallback_curriculum_enabled = bool(
            self.config.get("fallback_curriculum_enabled", True)
            if fallback_curriculum_enabled is None
            else fallback_curriculum_enabled
        )
        self.fail_on_episode_exception = bool(
            self.config.get("fail_on_episode_exception", False)
            if fail_on_episode_exception is None
            else fail_on_episode_exception
        )

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
        initial_model_path = (
            self.load_model_path
            if self.resume and self.load_model_path
            else self.shared_model_path
            if self.resume and self.shared_model_path.exists()
            else None
        )
        if initial_model_path and initial_model_path.exists():
            self.agent.load(initial_model_path)
            self.checkpoint_loaded = True
            self.checkpoint_path = str(initial_model_path)

        self.site_states: Dict[str, Dict[str, Any]] = {}
        for site in self.sites:
            site_id = site["site_id"]
            site_dir = self.output_dir / site_id
            known_bugs = [] if self.blind_url_training else _load_known_bugs_for_site(site)
            site_profile = build_site_profile(
                site_id,
                known_bugs,
                exploration_profile=site.get("exploration_profile"),
            )
            site_profile.update(_reward_mode_config(self.config, site))
            site_profile.update(_observability_hook_config(site))
            if self.blind_url_training:
                site_profile.pop("site_group", None)
                site_profile.pop("target_signal_types", None)
                site_profile.pop("expected_signal_types", None)
                site_profile["bugs"] = []
                site_profile["selectors"] = []
                site_profile["data_bug_ids"] = []
                site_profile["bug_types"] = []
            self.site_states[site_id] = {
                "site": site,
                "known_bugs": known_bugs,
                "site_profile": site_profile,
                "output_dir": site_dir,
                "transition_log_path": site_dir / "rl_transition_log.jsonl",
                "summary_path": site_dir / "training_summary.json",
                "detected_bugs_path": site_dir / "detected_bugs.json",
                "detected_bugs": [],
                "detected_index": {},
                "matched_index": {},
                "episode_rewards": [],
                "scaled_episode_rewards": [],
                "action_counts": Counter(),
                "raw_reward_total": 0.0,
                "scaled_reward_total": 0.0,
                "total_steps": 0,
                "total_anomalies": 0,
                "invalid_action_count": 0,
                "no_effect_action_count": 0,
                "diversity_reward_total": 0.0,
                "repeated_action_penalty_total": 0.0,
                "first_click_reward_count": 0,
                "new_action_type_reward_count": 0,
                "new_target_reward_count": 0,
                "inspect_reward_count": 0,
                "open_detail_panel_repeat_penalty_count": 0,
                "no_effect_action_penalty_count": 0,
                "functional_action_count": 0,
                "functional_priority_candidate_count": 0,
                "functional_priority_click_count": 0,
                "semantic_action_type_counts": Counter(),
                "clicked_semantic_action_types": Counter(),
                "no_effect_click_to_new_candidate_fallback_count": 0,
                "unclicked_functional_priority_candidate_count": 0,
                "high_value_functional_candidate_count": 0,
                "observed_functional_priority_targets": set(),
                "clicked_functional_priority_targets": set(),
                "first_functional_action_step": None,
                "unique_functional_action_types": set(),
                "unique_functional_targets": set(),
                "debug_meta_action_count": 0,
                "debug_meta_repeat_count": 0,
                "no_functional_action_episode_count": 0,
                "functional_action_signal_delta_count": 0,
                "functional_action_network_delta_count": 0,
                "reward_functional_action_total": 0.0,
                "penalty_debug_meta_total": 0.0,
                "click_retry_button_repeat_penalty_count": 0,
                "known_bug_reward_total": 0.0,
                "signal_reward_total": 0.0,
                "exploration_reward_total": 0.0,
                "ui_dom_signal_reward_total": 0.0,
                "console_runtime_signal_reward_total": 0.0,
                "network_api_signal_reward_total": 0.0,
                "cross_layer_signal_reward_total": 0.0,
                "security_signal_reward_total": 0.0,
                "repeated_penalty_total": 0.0,
                "multi_signal_anomaly_count": 0,
                "console_error_count": 0,
                "runtime_exception_count": 0,
                "network_error_count": 0,
                "api_4xx_count": 0,
                "api_5xx_count": 0,
                "api_timeout_count": 0,
                "api_ui_mismatch_count": 0,
                "security_signal_count": 0,
                "unique_clicked_targets": set(),
                "unique_clicked_element_keys": set(),
                "observed_element_keys": set(),
                "element_key_missing_count": 0,
                "element_key_collision_count": 0,
                "first_click_step": None,
                "episode_memory_reset_count": 0,
                "memory_state_update_count": 0,
                "memory_state_norm_values": [],
                "mask_stat_count": 0,
                "valid_action_total": 0,
                "valid_click_total": 0,
                "valid_fill_total": 0,
                "errors": [],
                "episode_errors": [],
                "preflight": {},
            }

    def train(self) -> Dict[str, Any]:
        _set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shared_model_path.parent.mkdir(parents=True, exist_ok=True)
        for state in self.site_states.values():
            state["output_dir"].mkdir(parents=True, exist_ok=True)
            state["transition_log_path"].write_text("", encoding="utf-8")
        csv_logger = (
            EpisodeCsvLogger(
                self.csv_log_dir,
                self.run_id,
                batch_id=self.batch_id,
                phase="training",
                log_observation_detail=self.log_observation_detail,
                log_action_space=self.log_action_space,
                log_raw_json=self.log_raw_json,
            )
            if self.enable_csv_logging
            else None
        )

        try:
            self._run_preflight_checks()
            update_summaries: List[Dict[str, Any]] = []
            global_episode = 0
            last_update_metrics: Dict[str, float] = {}

            for update_idx in range(1, self.total_updates + 1):
                shared_buffer = RolloutBuffer()
                per_site_update: Dict[str, Any] = {}
                print(f"[multisite-train] update {update_idx}/{self.total_updates}")

                for site in self.sites:
                    site_id = site["site_id"]
                    state = self.site_states[site_id]
                    site_rewards: List[float] = []
                    site_scaled_rewards: List[float] = []
                    site_anomalies = 0
                    for local_episode in range(1, self.episodes_per_site + 1):
                        global_episode += 1
                        episode_id = f"U{update_idx:04d}-{site_id}-EP{local_episode:03d}"
                        episode = self._collect_episode(
                            site,
                            state,
                            episode_id,
                            csv_logger=csv_logger,
                            update_id=update_idx,
                            local_episode_id=local_episode,
                        )
                        if len(episode["buffer"]) > 0:
                            shared_buffer.extend(episode["buffer"])
                        site_rewards.append(float(episode["raw_reward"]))
                        site_scaled_rewards.append(float(episode["scaled_reward"]))
                        site_anomalies += int(episode["anomaly_count"])

                    per_site_update[site_id] = {
                        "episodes": self.episodes_per_site,
                        "average_reward": float(np.mean(site_rewards)) if site_rewards else 0.0,
                        "raw_average_reward": float(np.mean(site_rewards)) if site_rewards else 0.0,
                        "scaled_average_reward": float(np.mean(site_scaled_rewards)) if site_scaled_rewards else 0.0,
                        "reward_scale": _reward_scale(site),
                        "anomaly_count": site_anomalies,
                    }
                    print(
                        f"  [site {site_id}] reward={per_site_update[site_id]['average_reward']:.3f}, "
                        f"anomalies={site_anomalies}"
                    )

                if len(shared_buffer) > 0:
                    last_update_metrics = self.agent.update(shared_buffer)
                else:
                    last_update_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
                self.agent.save(self.shared_model_path)
                self._write_site_outputs(last_update_metrics)

                update_summary = {
                    "update": update_idx,
                    "buffer_steps": len(shared_buffer),
                    "sites": per_site_update,
                    "loss": last_update_metrics,
                    "shared_model_path": str(self.shared_model_path),
                }
                update_summaries.append(update_summary)
                print(f"  [ppo] loss={last_update_metrics}")
                print(f"  [model] {self.shared_model_path}")

            summary = self._build_multisite_summary(update_summaries, last_update_metrics)
            summary_path = self.output_dir / "multisite_training_summary.json"
            summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_policy_safe_training_summary(summary)
            print(f"[multisite-train] summary: {summary_path}")
            return summary
        finally:
            if csv_logger is not None:
                csv_logger.close()

    def _collect_episode(
        self,
        site: Mapping[str, Any],
        state: Dict[str, Any],
        episode_id: str,
        csv_logger: EpisodeCsvLogger | None = None,
        update_id: int | None = None,
        local_episode_id: int | None = None,
    ) -> Dict[str, Any]:
        site_id = str(site["site_id"])
        reward_scale = _reward_scale(site)
        env = BrowserGymJAWSEnv(
            site_id=site_id,
            base_url=str(site["base_url"]),
            max_steps=self.max_steps,
            max_candidates=self.max_candidates,
            headless=self.headless,
            site_profile=state["site_profile"],
            requires_login=bool(site.get("requires_login")),
            login_config=site.get("login") if isinstance(site.get("login"), Mapping) else None,
        )
        buffer = RolloutBuffer()
        episode_reward = 0.0
        scaled_episode_reward = 0.0
        episode_anomaly_count = 0
        done = False
        observation: Dict[str, Any] = {}
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
        state["episode_memory_reset_count"] = int(state.get("episode_memory_reset_count", 0) or 0) + 1

        try:
            observation, _ = env.reset()
            for step in range(1, self.max_steps + 1):
                _attach_action_history_to_observation(observation, history)
                obs_vector = self.encoder.encode_observation(observation)
                action_mask = self.action_space.build_action_mask(observation)
                selected = self.agent.select_action(obs_vector, action_mask, memory_state=memory_state)
                selected_memory_state = selected.get("memory_state", memory_state)
                state.setdefault("memory_state_norm_values", []).append(
                    float(selected.get("memory_state_norm", 0.0) or 0.0)
                )
                state["memory_state_update_count"] = int(state.get("memory_state_update_count", 0) or 0) + 1
                policy_action_id = int(selected["action_id"])
                policy_action = self.action_space.decode(policy_action_id)
                policy_action["action_id"] = policy_action_id
                policy_action["site_id"] = site_id
                _enrich_action(policy_action, observation)
                action_id = policy_action_id
                fallback_reason = ""
                if self.enable_guided_actions:
                    action_id = _policy_safe_guided_action_id(
                        self.action_space,
                        observation,
                        history,
                        action_id,
                        action_mask,
                    )
                    if action_id != policy_action_id:
                        fallback_reason = "guided_policy_safe_override"
                fallback_applied = action_id != policy_action_id
                if fallback_applied:
                    state["fallback_applied_count"] = int(state.get("fallback_applied_count", 0) or 0) + 1
                    state["policy_executed_action_mismatch_count"] = int(
                        state.get("policy_executed_action_mismatch_count", 0) or 0
                    ) + 1
                else:
                    state["policy_executed_action_match_count"] = int(
                        state.get("policy_executed_action_match_count", 0) or 0
                    ) + 1
                if action_id != selected["action_id"]:
                    selected = self.agent.score_action(
                        obs_vector,
                        action_mask,
                        action_id,
                        memory_state=selected_memory_state,
                    )
                    selected_memory_state = selected.get("memory_state", selected_memory_state)

                action = self.action_space.decode(action_id)
                action["action_id"] = action_id
                action["site_id"] = site_id
                _enrich_action(action, observation)
                _attach_policy_execution_fields(
                    action,
                    policy_action_id=policy_action_id,
                    policy_action=policy_action,
                    executed_action_id=action_id,
                    fallback_applied=fallback_applied,
                    fallback_reason=fallback_reason,
                )
                if self.debug_action_mask:
                    stats = self.action_space.build_action_mask_stats(observation, action_mask, action_id)
                    _log_action_mask_stats(site_id, step, stats)
                else:
                    stats = self.action_space.build_action_mask_stats(observation, action_mask, action_id)
                state["mask_stat_count"] += 1
                state["valid_action_total"] += int(stats.get("valid_action_count", 0) or 0)
                state["valid_click_total"] += int(stats.get("valid_click_count", 0) or 0)
                state["valid_fill_total"] += int(stats.get("valid_fill_count", 0) or 0)
                _accumulate_candidate_semantics(state, observation)
                if csv_logger is not None:
                    csv_logger.log_observation(site_id, episode_id, step, step, observation)
                    csv_logger.log_action_space(
                        site_id,
                        episode_id,
                        step,
                        step,
                        observation,
                        self.action_space,
                        action_mask,
                        action_id,
                    )

                next_observation, _, done, step_info = env.step(action_id)
                action["failed"] = bool(step_info.get("last_action_error"))
                anomalies = detect_anomalies(
                    observation,
                    next_observation,
                    {"action": action, "site_profile": state["site_profile"], **step_info},
                    site_profile=state["site_profile"],
                )
                infra_anomalies = detect_infra_anomalies(
                    next_observation,
                    {"action": action, "site_profile": state["site_profile"], **step_info},
                )
                anomalies.extend(infra_anomalies)
                known_matches: List[Dict[str, Any]] = []
                action["matched_bug_ids"] = []
                reward, reward_breakdown = calculate_autonomous_reward(
                    observation,
                    next_observation,
                    action,
                    anomalies,
                    known_matches,
                    history,
                    state["site_profile"],
                )
                infra_reward, infra_breakdown = calculate_infra_reward(
                    observation,
                    next_observation,
                    action,
                    infra_anomalies,
                    history,
                )
                reward += infra_reward
                reward_breakdown.update(infra_breakdown)
                effective_fallback_penalty = fallback_curriculum_penalty(
                    self.fallback_penalty,
                    curriculum_enabled=self.fallback_curriculum_enabled,
                    episode_index=len(state.get("episode_rewards", []) or []),
                    warmup_episodes=max(1, self.episodes_per_site),
                )
                reward, reward_breakdown = apply_fallback_reward_policy(
                    reward,
                    reward_breakdown,
                    fallback_applied=fallback_applied,
                    fallback_penalty=effective_fallback_penalty,
                    fallback_reward_cap_enabled=self.fallback_reward_cap_enabled,
                    fallback_reward_cap_value=self.fallback_reward_cap_value,
                )
                if fallback_applied:
                    state["fallback_penalty_total"] = float(state.get("fallback_penalty_total", 0.0) or 0.0) + float(
                        reward_breakdown.get("fallback_penalty_total", 0.0) or 0.0
                    )
                    state["fallback_reward_capped_count"] = int(state.get("fallback_reward_capped_count", 0) or 0) + int(
                        float(reward_breakdown.get("fallback_reward_capped_count", 0.0) or 0.0)
                    )
                    if reward > 0:
                        state["fallback_action_success_count"] = int(state.get("fallback_action_success_count", 0) or 0) + 1
                elif reward > 0:
                    state["policy_action_success_count"] = int(state.get("policy_action_success_count", 0) or 0) + 1
                if float(reward_breakdown.get("penalty_invalid_action", 0.0) or 0.0) < 0.0:
                    state["invalid_action_count"] += 1
                if float(reward_breakdown.get("penalty_no_effect", 0.0) or 0.0) < 0.0:
                    state["no_effect_action_count"] += 1
                _accumulate_reward_breakdown(state, reward_breakdown)
                raw_reward = reward
                scaled_reward = _clamp_reward(raw_reward * reward_scale)
                reward_breakdown["raw_final_reward"] = raw_reward
                reward_breakdown["scaled_final_reward"] = scaled_reward
                reward_breakdown["reward_scale"] = reward_scale
                if csv_logger is not None:
                    csv_logger.log_step(
                        site_id=site_id,
                        base_url=str(site.get("base_url") or ""),
                        episode_id=episode_id,
                        step_id=step,
                        tick_id=step,
                        before_observation=observation,
                        after_observation=next_observation,
                        action=action,
                        action_mask=action_mask,
                        step_info=step_info,
                        anomalies=anomalies,
                        known_matches=known_matches,
                        reward=scaled_reward,
                        reward_breakdown=reward_breakdown,
                        done=done,
                    )
                    csv_logger.log_observation(site_id, episode_id, step, f"{step}.after", next_observation)

                buffer.add(
                    obs_vector,
                    action_id,
                    selected["log_prob"],
                    scaled_reward,
                    done,
                    selected["value"],
                    action_mask,
                    selected_memory_state,
                    step_info,
                )
                memory_state = selected_memory_state
                episode_reward += raw_reward
                scaled_episode_reward += scaled_reward
                state["raw_reward_total"] += raw_reward
                state["scaled_reward_total"] += scaled_reward
                episode_anomaly_count += len(anomalies)
                state["total_steps"] += 1
                state["total_anomalies"] += len(anomalies)
                state["action_counts"][str(action.get("action_type") or "")] += 1
                if action.get("action_type") == "click_element":
                    target = str(action.get("clicked_bid") or action.get("clicked_text") or "")
                    element_key = str(action.get("action_element_key") or "")
                    if target:
                        state["unique_clicked_targets"].add(target)
                    if element_key:
                        state.setdefault("unique_clicked_element_keys", set()).add(element_key)
                    if state.get("first_click_step") is None:
                        state["first_click_step"] = step
                    if action.get("functional_priority_candidate"):
                        state["functional_priority_click_count"] = int(state.get("functional_priority_click_count", 0) or 0) + 1
                        if element_key or target:
                            state.setdefault("clicked_functional_priority_targets", set()).add(element_key or target)
                    semantic_type = str(action.get("semantic_action_type") or "")
                    if semantic_type:
                        state.setdefault("clicked_semantic_action_types", Counter())[semantic_type] += 1
                if bool(state["site_profile"].get("use_known_bug_for_evaluation", False)):
                    self._record_detected_bug(state, episode_id, step, anomalies, known_matches)
                self._append_transition(
                    state["transition_log_path"],
                    {
                        "site_id": site_id,
                        "episode_id": episode_id,
                        "step": step,
                        "state_summary": {
                            "url": observation.get("page_state", {}).get("url", ""),
                            "viewport_type": observation.get("page_state", {}).get("viewport_type", ""),
                            "candidate_count": len(observation.get("candidate_elements", []) or []),
                        },
                        "action": action,
                        "action_type": action.get("action_type"),
                        "reward": scaled_reward,
                        "raw_reward": raw_reward,
                        "scaled_reward": scaled_reward,
                        "reward_scale": reward_scale,
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
                    },
                )
                _record_seen_anomaly_signatures(history, anomalies)
                _update_history(history, observation, action, anomalies, after_observation=next_observation)
                for verification_key in (
                    "verification_action_after_high_value_click_count",
                    "high_value_click_pending_verification_count",
                    "high_value_click_verified_count",
                    "high_value_click_without_verification_count",
                ):
                    state[verification_key] = int(history.get(verification_key, 0) or 0)
                observation = next_observation
                if done:
                    break
        except Exception as exc:
            error_record = self._record_episode_exception(
                state,
                site_id=site_id,
                update_id=update_id,
                episode_id=episode_id,
                local_episode_id=local_episode_id,
                step=len(buffer) + 1,
                exc=exc,
            )
            self._append_transition(
                state["transition_log_path"],
                {
                    "site_id": site_id,
                    "episode_id": episode_id,
                    "step": len(buffer) + 1,
                    "state_summary": {"url": "", "candidate_count": 0},
                    "action": {"action_id": None, "action_type": "exception", "candidate_index": 0, "site_id": site_id},
                    "reward": 0.0,
                    "reward_breakdown": {"final_reward": 0.0},
                    "anomalies": [
                        {
                            "type": "episode-exception",
                            "confidence": 1.0,
                            "evidence": {
                                "error": str(exc),
                                "exception_type": type(exc).__name__,
                                "traceback": error_record.get("traceback", ""),
                            },
                        }
                    ],
                    "done": True,
                },
            )
            done = True
            print(f"  [site {site_id}] {episode_id} exception: {exc}")
            print(error_record.get("traceback", ""))
            if self.fail_on_episode_exception:
                raise
        finally:
            env.close()

        if len(buffer) > 0:
            last_value = 0.0 if done else self.agent.estimate_value(
                self.encoder.encode_observation(observation),
                memory_state=memory_state,
            )
            buffer.compute_returns_and_advantages(last_value, self.agent.gamma, self.agent.gae_lambda)
        state["episode_rewards"].append(episode_reward)
        state["scaled_episode_rewards"].append(scaled_episode_reward)
        return {
            "buffer": buffer,
            "raw_reward": episode_reward,
            "scaled_reward": scaled_episode_reward,
            "anomaly_count": episode_anomaly_count,
        }

    def _record_detected_bug(
        self,
        state: Dict[str, Any],
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
            existing_index = state["detected_index"].get(key)
            if existing_index is not None:
                if confidence > float(state["detected_bugs"][existing_index].get("confidence", 0.0) or 0.0):
                    state["detected_bugs"][existing_index] = record
                continue
            if matched_bug_id:
                matched_key = str(matched_bug_id)
                matched_index = state["matched_index"].get(matched_key)
                if matched_index is not None:
                    if confidence > float(state["detected_bugs"][matched_index].get("confidence", 0.0) or 0.0):
                        state["detected_bugs"][matched_index] = record
                    continue
                state["matched_index"][matched_key] = len(state["detected_bugs"])
            state["detected_index"][key] = len(state["detected_bugs"])
            state["detected_bugs"].append(record)

    def _write_site_outputs(self, last_update_metrics: Mapping[str, Any]) -> None:
        for site_id, state in self.site_states.items():
            summary = self._build_site_summary(site_id, state, last_update_metrics)
            state["summary_path"].write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
            state["detected_bugs_path"].write_text(
                json.dumps(_jsonable(state["detected_bugs"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _build_site_summary(
        self,
        site_id: str,
        state: Mapping[str, Any],
        last_update_metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        known_bugs = state["known_bugs"]
        detected_bugs = state["detected_bugs"]
        known_bug_ids = sorted(_known_bug_id(bug) for bug in known_bugs if _known_bug_id(bug))
        matched_bug_ids = sorted({str(item.get("matched_bug_id")) for item in detected_bugs if item.get("matched_bug_id")})
        missed_bug_ids = [bug_id for bug_id in known_bug_ids if bug_id not in set(matched_bug_ids)]
        unique_detected = len(detected_bugs)
        known_match_count = len(matched_bug_ids)
        has_catalog = bool(state["site"].get("has_bug_catalog")) and bool(known_bug_ids)
        confirmed_count = sum(1 for item in detected_bugs if item.get("matched_bug_id"))
        catalog_related_count = sum(
            1 for item in detected_bugs if not item.get("matched_bug_id") and _is_catalog_related(item.get("evidence", {}))
        )
        deduped_known_bug_candidates = confirmed_count + catalog_related_count
        return {
            "site_id": site_id,
            "base_url": state["site"].get("base_url"),
            "shared_model_path": str(self.shared_model_path),
            **_summary_reward_mode_fields(state["site_profile"]),
            "known_bug_catalog_available": has_catalog,
            "episodes": len(state["episode_rewards"]),
            "max_steps": self.max_steps,
            "total_steps": state["total_steps"],
            "average_episode_reward": float(np.mean(state["episode_rewards"])) if state["episode_rewards"] else 0.0,
            "raw_average_reward": float(np.mean(state["episode_rewards"])) if state["episode_rewards"] else 0.0,
            "scaled_average_reward": float(np.mean(state["scaled_episode_rewards"])) if state["scaled_episode_rewards"] else 0.0,
            "reward_scale": _reward_scale(state["site"]),
            "detected_bug_count": unique_detected,
            "deduped_detected_candidates": unique_detected,
            "unique_detected_candidates": unique_detected,
            "total_detected_candidates": state["total_anomalies"],
            "known_bug_match_count": known_match_count if has_catalog else None,
            "matched_bug_ids": matched_bug_ids if has_catalog else [],
            "missed_bug_ids": missed_bug_ids if has_catalog else [],
            "precision": (known_match_count / unique_detected if unique_detected else 0.0) if has_catalog else None,
            "known_bug_precision": (known_match_count / deduped_known_bug_candidates if deduped_known_bug_candidates else 0.0)
            if has_catalog
            else None,
            "catalog_precision": (confirmed_count / deduped_known_bug_candidates if deduped_known_bug_candidates else 0.0)
            if has_catalog
            else None,
            "known_bug_candidate_count": deduped_known_bug_candidates if has_catalog else None,
            "exploratory_anomaly_count": max(0, unique_detected - deduped_known_bug_candidates) if has_catalog else None,
            "recall": (min(1.0, known_match_count / max(1, len(known_bug_ids)))) if has_catalog else None,
            "openended_anomaly_discovery": None
            if has_catalog
            else {
                "unique_detected_candidates": unique_detected,
                "total_detected_candidates": state["total_anomalies"],
                "anomaly_types": dict(Counter(str(item.get("type") or "") for item in detected_bugs)),
            },
            "action_counts": dict(state["action_counts"]),
            "click_element_count": int(state["action_counts"].get("click_element", 0)),
            "action_diversity_score": _action_diversity_score(state["action_counts"]),
            "unique_action_type_count": len([key for key, value in state["action_counts"].items() if int(value or 0) > 0]),
            "unique_clicked_target_count": len(state.get("unique_clicked_targets", set()) or set()),
            "unique_element_key_count": len(state.get("observed_element_keys", set()) or set()),
            "unique_clicked_element_key_count": len(state.get("unique_clicked_element_keys", set()) or set()),
            "element_key_missing_count": int(state.get("element_key_missing_count", 0) or 0),
            "element_key_collision_count": int(state.get("element_key_collision_count", 0) or 0),
            "first_click_step": state.get("first_click_step"),
            "open_detail_panel_count": int(state["action_counts"].get("open_detail_panel", 0)),
            "fallback_applied_count": int(state.get("fallback_applied_count", 0) or 0),
            "fallback_applied_rate": _safe_rate(state.get("fallback_applied_count", 0), state.get("total_steps", 0)),
            "fallback_penalty_total": float(state.get("fallback_penalty_total", 0.0) or 0.0),
            "fallback_reward_capped_count": int(state.get("fallback_reward_capped_count", 0) or 0),
            "policy_action_success_count": int(state.get("policy_action_success_count", 0) or 0),
            "fallback_action_success_count": int(state.get("fallback_action_success_count", 0) or 0),
            "policy_executed_action_match_count": int(state.get("policy_executed_action_match_count", 0) or 0),
            "policy_executed_action_mismatch_count": int(state.get("policy_executed_action_mismatch_count", 0) or 0),
            "fallback_penalty": self.fallback_penalty,
            "fallback_reward_cap_enabled": self.fallback_reward_cap_enabled,
            "fallback_reward_cap_value": self.fallback_reward_cap_value,
            "fallback_curriculum_enabled": self.fallback_curriculum_enabled,
            "diversity_reward_total": float(state.get("diversity_reward_total", 0.0) or 0.0),
            "repeated_action_penalty_total": float(state.get("repeated_action_penalty_total", 0.0) or 0.0),
            "login_form_coverage_reward_total": float(state.get("login_form_coverage_reward_total", 0.0) or 0.0),
            "login_flow_penalty_total": float(state.get("login_flow_penalty_total", 0.0) or 0.0),
            "reward_email_input_filled": float(state.get("reward_email_input_filled", 0.0) or 0.0),
            "reward_password_input_filled": float(state.get("reward_password_input_filled", 0.0) or 0.0),
            "reward_submit_clicked": float(state.get("reward_submit_clicked", 0.0) or 0.0),
            "reward_submit_result_checked": float(state.get("reward_submit_result_checked", 0.0) or 0.0),
            "penalty_login_flow_incomplete_early_stop": float(state.get("penalty_login_flow_incomplete_early_stop", 0.0) or 0.0),
            "penalty_targetless_action_success": float(state.get("penalty_targetless_action_success", 0.0) or 0.0),
            **_state_signal_summary(state),
            "first_click_reward_count": int(state.get("first_click_reward_count", 0) or 0),
            "new_action_type_reward_count": int(state.get("new_action_type_reward_count", 0) or 0),
            "new_target_reward_count": int(state.get("new_target_reward_count", 0) or 0),
            "inspect_reward_count": int(state.get("inspect_reward_count", 0) or 0),
            "open_detail_panel_repeat_penalty_count": int(state.get("open_detail_panel_repeat_penalty_count", 0) or 0),
            "no_effect_action_penalty_count": int(state.get("no_effect_action_penalty_count", 0) or 0),
            "inspect_cart_count": int(state["action_counts"].get("inspect_cart", 0)),
            "last_update": dict(last_update_metrics),
            "transition_log_path": str(state["transition_log_path"]),
            "preflight": dict(state.get("preflight", {})),
            "episode_exception_count": len(state.get("episode_errors", []) or state.get("errors", []) or []),
            "episode_errors": list(state.get("episode_errors", []) or state.get("errors", []) or []),
            "valid_training_run": not bool(state.get("episode_errors") or state.get("errors")),
        }

    def _record_episode_exception(
        self,
        state: Dict[str, Any],
        *,
        site_id: str,
        update_id: int | None,
        episode_id: str,
        local_episode_id: int | None,
        step: int,
        exc: Exception,
    ) -> Dict[str, Any]:
        record = {
            "site_id": site_id,
            "update_id": update_id,
            "episode_id": episode_id,
            "local_episode_id": local_episode_id,
            "step": step,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        state.setdefault("errors", []).append(record)
        state.setdefault("episode_errors", []).append(record)
        return record

    def _run_preflight_checks(self) -> None:
        print("[multisite-preflight]")
        for site in self.sites:
            site_id = str(site["site_id"])
            state = self.site_states[site_id]
            result: Dict[str, Any] = {
                "site_id": site_id,
                "base_url": site.get("base_url"),
                "has_bug_catalog": bool(site.get("has_bug_catalog")),
                "evaluation_mode": "known_bug_catalog"
                if bool(site.get("use_known_bug_for_evaluation", self.config.get("use_known_bug_for_evaluation", False)))
                and bool(site.get("has_bug_catalog"))
                else "signal_discovery",
                "connection": "failed",
                "candidate_count": 0,
                "site_identity_match": True if bool(site.get("has_bug_catalog")) else "unknown",
                "identity_warnings": [],
            }
            env = BrowserGymJAWSEnv(
                site_id=site_id,
                base_url=str(site["base_url"]),
                max_steps=1,
                max_candidates=self.max_candidates,
                headless=self.headless,
                site_profile=state["site_profile"],
                requires_login=bool(site.get("requires_login")),
                login_config=site.get("login") if isinstance(site.get("login"), Mapping) else None,
            )
            try:
                observation, reset_info = env.reset()
                identity = validate_site_identity(site_id, observation)
                if not bool(site.get("has_bug_catalog")) and not identity.get("data_bug_ids_found"):
                    identity["site_identity_match"] = "unknown"
                    identity["identity_warnings"] = []
                result.update(
                    {
                        "connection": "ok",
                        "observed_url": observation.get("page_state", {}).get("url", ""),
                        "observed_title": observation.get("page_state", {}).get("title", ""),
                        "candidate_count": len(observation.get("candidate_elements", []) or []),
                        "login_required": bool(reset_info.get("login_required")),
                        "login_attempted": bool(reset_info.get("login_attempted")),
                        "login_success": bool(reset_info.get("login_success")),
                        "post_login_url": str(reset_info.get("post_login_url") or ""),
                        **identity,
                    }
                )
                if bool(site.get("has_bug_catalog")) and bool(reset_info.get("login_success")) and not identity.get("data_bug_ids_found"):
                    result.setdefault("identity_warnings", []).append(
                        "WARNING: bug_catalog loaded and login succeeded, but no data-bug-id was found. Check site9800 UI data-bug-id placement."
                    )
            except Exception as exc:
                result["error"] = str(exc)
                result["identity_warnings"] = [f"WARNING: {site_id} preflight failed: {exc}"]
            finally:
                env.close()
            state["preflight"] = result
            print(f"{site_id}:")
            print(f"  base_url: {result.get('base_url')}")
            print(f"  connection: {result.get('connection')}")
            print(f"  observed_url: {result.get('observed_url', '')}")
            print(f"  site_identity_match: {result.get('site_identity_match')}")
            print(f"  candidate_count: {result.get('candidate_count')}")
            print(f"  login_required: {result.get('login_required')}")
            print(f"  login_attempted: {result.get('login_attempted')}")
            print(f"  login_success: {result.get('login_success')}")
            print(f"  post_login_url: {result.get('post_login_url')}")
            print(f"  has_bug_catalog: {result.get('has_bug_catalog')}")
            print(f"  evaluation_mode: {result.get('evaluation_mode')}")
            for warning in result.get("identity_warnings", []) or []:
                print(f"  {warning}")
            if self.strict_site_validation and (
                result.get("connection") != "ok"
                or (bool(site.get("has_bug_catalog")) and result.get("site_identity_match") is False)
            ):
                raise RuntimeError(f"Preflight validation failed for {site_id}: {result}")

    def _build_multisite_summary(
        self,
        update_summaries: List[Dict[str, Any]],
        last_update_metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        site_summaries = {
            site_id: self._build_site_summary(site_id, state, last_update_metrics)
            for site_id, state in self.site_states.items()
        }
        rewards = [summary["average_episode_reward"] for summary in site_summaries.values()]
        return {
            "run_id": self.run_id,
            "config_name": str(self.config.get("config_name") or self.config.get("run_id") or self.config_path.stem),
            "config_path": str(self.config_path),
            "shared_model_path": str(self.shared_model_path),
            "output_dir": str(self.output_dir),
            "total_updates": self.total_updates,
            "episodes_per_site": self.episodes_per_site,
            "max_steps": self.max_steps,
            "eval_episodes": int(self.config.get("eval_episodes", self.config.get("episodes", 0)) or 0),
            "memory_encoder_enabled": self.use_memory_encoder,
            "use_memory_encoder": self.use_memory_encoder,
            "memory_encoder_type": self.memory_encoder_type if self.use_memory_encoder else "",
            "memory_hidden_size": self.memory_hidden_size if self.use_memory_encoder else 0,
            "episode_memory_reset_count": sum(
                int(state.get("episode_memory_reset_count", 0) or 0) for state in self.site_states.values()
            ),
            "memory_state_reset_count": sum(
                int(state.get("episode_memory_reset_count", 0) or 0) for state in self.site_states.values()
            ),
            "memory_state_update_count": (
                sum(
                    int(state.get("memory_state_update_count", 0) or 0) for state in self.site_states.values()
                )
                if self.use_memory_encoder
                else 0
            ),
            "memory_state_norm_mean": _mean_memory_state_norm(self.site_states.values()) if self.use_memory_encoder else 0.0,
            "repeated_action_rate": _repeated_action_rate(self.site_states.values()),
            "repeated_click_target_rate": _repeated_click_target_rate(self.site_states.values()),
            "repeated_action_rate_before_memory": None,
            "repeated_action_rate_after_memory": _repeated_action_rate(self.site_states.values()) if self.use_memory_encoder else None,
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_path": self.checkpoint_path,
            "fresh_start": self.fresh_start,
            "resume": self.resume,
            "model_path": str(self.shared_model_path),
            "checkpoint_isolated_by_run_id": _checkpoint_isolated_by_run_id(self.run_id, self.shared_model_path, self.output_dir),
            "allow_known_bug_reward_for_debug_only": False,
            "use_ground_truth_for_evaluation_only": bool(self.config.get("use_ground_truth_for_evaluation_only", False)),
            "use_known_bug_reward": False,
            "use_known_bug_for_training": False,
            "known_bug_reward_total": 0.0,
            "policy_uses_bug_labels": False,
            **self.policy_safety_audit,
            **self._site_config_summary(),
            "episode_exception_count": _episode_exception_count(self.site_states.values()),
            "episode_errors": _episode_errors(self.site_states.values()),
            "valid_training_run": _episode_exception_count(self.site_states.values()) == 0,
            "fail_on_episode_exception": self.fail_on_episode_exception,
            "average_reward_across_sites": float(np.mean(rewards)) if rewards else 0.0,
            "sites": site_summaries,
            "updates": update_summaries,
            "last_update": dict(last_update_metrics),
        }

    def _write_policy_safe_training_summary(self, summary: Mapping[str, Any]) -> None:
        try:
            path = self.csv_log_dir / "training_summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            site_summaries = summary.get("sites", {}) if isinstance(summary.get("sites"), Mapping) else {}
            total_steps = sum(int(state.get("total_steps", 0) or 0) for state in self.site_states.values())
            total_episodes = self.total_updates * self.episodes_per_site * len(self.sites)
            all_rewards = [
                float(reward)
                for state in self.site_states.values()
                for reward in state.get("episode_rewards", []) or []
            ]
            mask_count = sum(int(state.get("mask_stat_count", 0) or 0) for state in self.site_states.values())
            known_bug_match_count = sum(
                int(site.get("known_bug_match_count") or 0)
                for site in site_summaries.values()
                if isinstance(site, Mapping)
            )
            unique_anomaly_count = sum(
                int(site.get("unique_detected_candidates") or 0)
                for site in site_summaries.values()
                if isinstance(site, Mapping)
            )
            payload = {
                "version": VERSION,
                "run_id": self.run_id,
                "config_name": str(self.config.get("config_name") or self.config.get("run_id") or self.config_path.stem),
                "config_path": str(self.config_path),
                "model_path": str(self.shared_model_path),
                "output_dir": str(self.output_dir),
                "total_updates": self.total_updates,
                "episodes_per_site": self.episodes_per_site,
                "max_steps": self.max_steps,
                "eval_episodes": int(self.config.get("eval_episodes", self.config.get("episodes", 0)) or 0),
                "memory_encoder_enabled": self.use_memory_encoder,
                "use_memory_encoder": self.use_memory_encoder,
                "memory_encoder_type": self.memory_encoder_type if self.use_memory_encoder else "",
                "memory_hidden_size": self.memory_hidden_size if self.use_memory_encoder else 0,
                "episode_memory_reset_count": sum(
                    int(state.get("episode_memory_reset_count", 0) or 0) for state in self.site_states.values()
                ),
                "memory_state_reset_count": sum(
                    int(state.get("episode_memory_reset_count", 0) or 0) for state in self.site_states.values()
                ),
                "memory_state_update_count": (
                    sum(int(state.get("memory_state_update_count", 0) or 0) for state in self.site_states.values())
                    if self.use_memory_encoder
                    else 0
                ),
                "memory_state_norm_mean": _mean_memory_state_norm(self.site_states.values()) if self.use_memory_encoder else 0.0,
                "repeated_action_rate": _repeated_action_rate(self.site_states.values()),
                "repeated_click_target_rate": _repeated_click_target_rate(self.site_states.values()),
                "repeated_action_rate_before_memory": None,
                "repeated_action_rate_after_memory": _repeated_action_rate(self.site_states.values()) if self.use_memory_encoder else None,
                "checkpoint_loaded": self.checkpoint_loaded,
                "checkpoint_path": self.checkpoint_path,
                "fresh_start": self.fresh_start,
                "resume": self.resume,
                "checkpoint_isolated_by_run_id": _checkpoint_isolated_by_run_id(self.run_id, self.shared_model_path, self.output_dir),
                "site_count": len(self.sites),
                "site_ids": [str(site.get("site_id") or "") for site in self.sites],
                **self.policy_safety_audit,
                **self._site_config_summary(),
                "enable_guided_actions": self.enable_guided_actions,
                "debug_action_mask": self.debug_action_mask,
                "reward_mode": str(self.config.get("reward_mode") or "signal_based"),
                "allow_known_bug_reward_for_debug_only": False,
                "use_ground_truth_for_evaluation_only": bool(self.config.get("use_ground_truth_for_evaluation_only", False)),
                "use_known_bug_reward": False,
                "use_known_bug_for_training": False,
                "use_known_bug_for_evaluation": bool(self.config.get("use_known_bug_for_evaluation", False)),
                "training_uses_site_specific_bug_catalog": False
                if self.blind_url_training
                else bool(self.config.get("training_uses_site_specific_bug_catalog", False)),
                "policy_uses_bug_labels": False,
                "episode_exception_count": _episode_exception_count(self.site_states.values()),
                "episode_errors": _episode_errors(self.site_states.values()),
                "valid_training_run": _episode_exception_count(self.site_states.values()) == 0,
                "fail_on_episode_exception": self.fail_on_episode_exception,
                "target_signal_types": []
                if self.blind_url_training
                else list(self.config.get("target_signal_types", []) or []),
                "observed_signal_types": _observed_signal_types_from_state_totals(self.site_states.values()),
                "missing_signal_types": sorted(
                    set(str(item) for item in ([] if self.blind_url_training else self.config.get("target_signal_types", []) or []))
                    - set(_observed_signal_types_from_state_totals(self.site_states.values()))
                ),
                "signal_type_coverage": _signal_type_coverage(
                    [] if self.blind_url_training else self.config.get("target_signal_types", []) or [],
                    _observed_signal_types_from_state_totals(self.site_states.values()),
                ),
                "per_site_signal_coverage": {
                    site_id: {
                        "target_signal_types": _state_signal_summary(state).get("target_signal_types", []),
                        "observed_signal_types": _state_signal_summary(state).get("observed_signal_types", []),
                        "missing_signal_types": _state_signal_summary(state).get("missing_signal_types", []),
                        "signal_type_coverage": _state_signal_summary(state).get("signal_type_coverage", 0.0),
                    }
                    for site_id, state in self.site_states.items()
                },
                "signal_collector_enabled": True,
                "playwright_console_listener_enabled": any(
                    bool(state.get("playwright_console_listener_enabled")) for state in self.site_states.values()
                ),
                "playwright_pageerror_listener_enabled": any(
                    bool(state.get("playwright_pageerror_listener_enabled")) for state in self.site_states.values()
                ),
                "playwright_network_listener_enabled": any(
                    bool(state.get("playwright_network_listener_enabled")) for state in self.site_states.values()
                ),
                "playwright_request_listener_enabled": any(
                    bool(state.get("playwright_request_listener_enabled")) for state in self.site_states.values()
                ),
                "playwright_response_listener_enabled": any(
                    bool(state.get("playwright_response_listener_enabled")) for state in self.site_states.values()
                ),
                "playwright_requestfailed_listener_enabled": any(
                    bool(state.get("playwright_requestfailed_listener_enabled")) for state in self.site_states.values()
                ),
                "site_errors": {
                    site_id: list(state.get("errors", []) or [])
                    for site_id, state in self.site_states.items()
                    if state.get("errors")
                },
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "avg_episode_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "avg_update_reward": float(
                    np.mean([
                        float(site.get("average_reward", 0.0) or 0.0)
                        for update in summary.get("updates", []) or []
                        for site in (update.get("sites", {}) or {}).values()
                        if isinstance(site, Mapping)
                    ])
                )
                if summary.get("updates")
                else 0.0,
                "known_bug_match_count": known_bug_match_count,
                "unique_anomaly_count": unique_anomaly_count,
                "invalid_action_count": sum(int(state.get("invalid_action_count", 0) or 0) for state in self.site_states.values()),
                "no_effect_action_count": sum(int(state.get("no_effect_action_count", 0) or 0) for state in self.site_states.values()),
                "diversity_reward_total": sum(float(state.get("diversity_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "repeated_action_penalty_total": sum(float(state.get("repeated_action_penalty_total", 0.0) or 0.0) for state in self.site_states.values()),
                "first_click_reward_count": sum(int(state.get("first_click_reward_count", 0) or 0) for state in self.site_states.values()),
                "new_action_type_reward_count": sum(int(state.get("new_action_type_reward_count", 0) or 0) for state in self.site_states.values()),
                "new_target_reward_count": sum(int(state.get("new_target_reward_count", 0) or 0) for state in self.site_states.values()),
                "inspect_reward_count": sum(int(state.get("inspect_reward_count", 0) or 0) for state in self.site_states.values()),
                "open_detail_panel_repeat_penalty_count": sum(int(state.get("open_detail_panel_repeat_penalty_count", 0) or 0) for state in self.site_states.values()),
                "no_effect_action_penalty_count": sum(int(state.get("no_effect_action_penalty_count", 0) or 0) for state in self.site_states.values()),
                "known_bug_reward_total": 0.0,
                "signal_reward_total": sum(float(state.get("signal_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "exploration_reward_total": sum(float(state.get("exploration_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "ui_dom_signal_reward_total": sum(float(state.get("ui_dom_signal_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "console_runtime_signal_reward_total": sum(float(state.get("console_runtime_signal_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "network_api_signal_reward_total": sum(float(state.get("network_api_signal_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "cross_layer_signal_reward_total": sum(float(state.get("cross_layer_signal_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "security_signal_reward_total": sum(float(state.get("security_signal_reward_total", 0.0) or 0.0) for state in self.site_states.values()),
                "repeated_penalty_total": sum(float(state.get("repeated_penalty_total", 0.0) or 0.0) for state in self.site_states.values()),
                "multi_signal_anomaly_count": sum(int(state.get("multi_signal_anomaly_count", 0) or 0) for state in self.site_states.values()),
                "console_error_count": sum(int(state.get("console_error_count", 0) or 0) for state in self.site_states.values()),
                "runtime_exception_count": sum(int(state.get("runtime_exception_count", 0) or 0) for state in self.site_states.values()),
                "network_error_count": sum(int(state.get("network_error_count", 0) or 0) for state in self.site_states.values()),
                "api_4xx_count": sum(int(state.get("api_4xx_count", 0) or 0) for state in self.site_states.values()),
                "api_5xx_count": sum(int(state.get("api_5xx_count", 0) or 0) for state in self.site_states.values()),
                "api_timeout_count": sum(int(state.get("api_timeout_count", 0) or 0) for state in self.site_states.values()),
                "api_ui_mismatch_count": sum(int(state.get("api_ui_mismatch_count", 0) or 0) for state in self.site_states.values()),
                "security_signal_count": sum(int(state.get("security_signal_count", 0) or 0) for state in self.site_states.values()),
                "action_diversity_score": float(np.mean([
                    _action_diversity_score(state.get("action_counts", {}))
                    for state in self.site_states.values()
                ])) if self.site_states else 0.0,
                "unique_action_type_count": sum(
                    len([key for key, value in state.get("action_counts", {}).items() if int(value or 0) > 0])
                    for state in self.site_states.values()
                ),
                "unique_clicked_target_count": sum(len(state.get("unique_clicked_targets", set()) or set()) for state in self.site_states.values()),
                "unique_element_key_count": sum(len(state.get("observed_element_keys", set()) or set()) for state in self.site_states.values()),
                "unique_clicked_element_key_count": sum(len(state.get("unique_clicked_element_keys", set()) or set()) for state in self.site_states.values()),
                "element_key_missing_count": sum(int(state.get("element_key_missing_count", 0) or 0) for state in self.site_states.values()),
                "element_key_collision_count": sum(int(state.get("element_key_collision_count", 0) or 0) for state in self.site_states.values()),
                "first_click_step": min(
                    [int(state["first_click_step"]) for state in self.site_states.values() if state.get("first_click_step") is not None],
                    default=None,
                ),
                "open_detail_panel_count": sum(int(state.get("action_counts", {}).get("open_detail_panel", 0) or 0) for state in self.site_states.values()),
                "click_retry_button_count": sum(int(state.get("action_counts", {}).get("click_retry_button", 0) or 0) for state in self.site_states.values()),
                "functional_action_count": sum(int(state.get("functional_action_count", 0) or 0) for state in self.site_states.values()),
                "functional_priority_candidate_count": sum(int(state.get("functional_priority_candidate_count", 0) or 0) for state in self.site_states.values()),
                "functional_priority_click_count": sum(int(state.get("functional_priority_click_count", 0) or 0) for state in self.site_states.values()),
                "semantic_action_type_counts": _sum_counters(
                    state.get("semantic_action_type_counts", {}) for state in self.site_states.values()
                ),
                "clicked_semantic_action_types": _sum_counters(
                    state.get("clicked_semantic_action_types", {}) for state in self.site_states.values()
                ),
                "no_effect_click_to_new_candidate_fallback_count": sum(
                    int(state.get("no_effect_click_to_new_candidate_fallback_count", 0) or 0)
                    for state in self.site_states.values()
                ),
                "fallback_applied_count": sum(int(state.get("fallback_applied_count", 0) or 0) for state in self.site_states.values()),
                "fallback_applied_rate": _safe_rate(
                    sum(int(state.get("fallback_applied_count", 0) or 0) for state in self.site_states.values()),
                    total_steps,
                ),
                "fallback_penalty_total": sum(float(state.get("fallback_penalty_total", 0.0) or 0.0) for state in self.site_states.values()),
                "fallback_reward_capped_count": sum(int(state.get("fallback_reward_capped_count", 0) or 0) for state in self.site_states.values()),
                "policy_action_success_count": sum(int(state.get("policy_action_success_count", 0) or 0) for state in self.site_states.values()),
                "fallback_action_success_count": sum(int(state.get("fallback_action_success_count", 0) or 0) for state in self.site_states.values()),
                "policy_executed_action_match_count": sum(
                    int(state.get("policy_executed_action_match_count", 0) or 0) for state in self.site_states.values()
                ),
                "policy_executed_action_mismatch_count": sum(
                    int(state.get("policy_executed_action_mismatch_count", 0) or 0) for state in self.site_states.values()
                ),
                "fallback_penalty": self.fallback_penalty,
                "fallback_reward_cap_enabled": self.fallback_reward_cap_enabled,
                "fallback_reward_cap_value": self.fallback_reward_cap_value,
                "fallback_curriculum_enabled": self.fallback_curriculum_enabled,
                "unclicked_functional_priority_candidate_count": sum(
                    len((state.get("observed_functional_priority_targets", set()) or set()) - (state.get("clicked_functional_priority_targets", set()) or set()))
                    for state in self.site_states.values()
                ),
                "high_value_functional_candidate_count": sum(int(state.get("high_value_functional_candidate_count", 0) or 0) for state in self.site_states.values()),
                "first_functional_action_step": min(
                    [int(state["first_functional_action_step"]) for state in self.site_states.values() if state.get("first_functional_action_step") is not None],
                    default=None,
                ),
                "unique_functional_action_type_count": len(
                    set().union(*(state.get("unique_functional_action_types", set()) or set() for state in self.site_states.values()))
                )
                if self.site_states
                else 0,
                "unique_functional_target_count": sum(len(state.get("unique_functional_targets", set()) or set()) for state in self.site_states.values()),
                "debug_meta_action_count": sum(int(state.get("debug_meta_action_count", 0) or 0) for state in self.site_states.values()),
                "debug_meta_repeat_count": sum(int(state.get("debug_meta_repeat_count", 0) or 0) for state in self.site_states.values()),
                "no_functional_action_episode_count": sum(int(state.get("no_functional_action_episode_count", 0) or 0) for state in self.site_states.values()),
                "functional_action_signal_delta_count": sum(int(state.get("functional_action_signal_delta_count", 0) or 0) for state in self.site_states.values()),
                "functional_action_network_delta_count": sum(int(state.get("functional_action_network_delta_count", 0) or 0) for state in self.site_states.values()),
                "reward_functional_action_total": sum(float(state.get("reward_functional_action_total", 0.0) or 0.0) for state in self.site_states.values()),
                "penalty_debug_meta_total": sum(float(state.get("penalty_debug_meta_total", 0.0) or 0.0) for state in self.site_states.values()),
                "avg_valid_action_count": (
                    sum(int(state.get("valid_action_total", 0) or 0) for state in self.site_states.values()) / mask_count
                    if mask_count
                    else 0.0
                ),
                "avg_valid_click_count": (
                    sum(int(state.get("valid_click_total", 0) or 0) for state in self.site_states.values()) / mask_count
                    if mask_count
                    else 0.0
                ),
                "avg_valid_fill_count": (
                    sum(int(state.get("valid_fill_total", 0) or 0) for state in self.site_states.values()) / mask_count
                    if mask_count
                    else 0.0
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[multisite-train] failed to write v3 policy-safe training summary: {exc}")

    @staticmethod
    def _append_transition(path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def _site_config_summary(self) -> Dict[str, Any]:
        if self.blind_url_training:
            return {
                "configured_site_ids": self.configured_site_ids,
                "enabled_site_ids": self.enabled_site_ids,
                "disabled_site_ids": self.disabled_site_ids,
                "skipped_site_ids": self.disabled_site_ids,
                "site_groups": {},
                "target_signal_types_by_site": {},
            }
        site_groups = {
            str(site.get("site_id") or ""): str(site.get("site_group") or "")
            for site in self.configured_sites
            if site.get("site_id")
        }
        target_by_site = {
            str(site.get("site_id") or ""): list(site.get("target_signal_types") or [])
            for site in self.configured_sites
            if site.get("site_id")
        }
        return {
            "configured_site_ids": self.configured_site_ids,
            "enabled_site_ids": self.enabled_site_ids,
            "disabled_site_ids": self.disabled_site_ids,
            "skipped_site_ids": self.disabled_site_ids,
            "site_groups": site_groups,
            "target_signal_types_by_site": target_by_site,
        }


def _load_known_bugs_for_site(site: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not bool(site.get("has_bug_catalog")):
        return []
    path = site.get("bug_catalog_path")
    if path:
        raw = _read_json(Path(str(path)))
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
        if isinstance(raw, Mapping):
            bugs = raw.get("bugs") or raw.get("items") or raw.get("bug_catalog") or raw.get("known_bugs") or []
            if isinstance(bugs, list):
                return [dict(item) for item in bugs if isinstance(item, Mapping)]
    return load_known_bugs(str(site.get("site_id") or ""))


def _policy_safe_guided_action_id(
    action_space: ActionSpace,
    observation: Dict[str, Any],
    history: Dict[str, Any],
    selected_action_id: int,
    action_mask: np.ndarray,
) -> int:
    counts = history.get("action_type_counts", {})
    step_index = int(observation.get("history", {}).get("step_index", 0) or 0)
    if not isinstance(counts, Mapping):
        counts = {}

    selected = action_space.decode(selected_action_id)
    last_action_key = str(history.get("last_action_key") or "")
    selected_key = f"{selected.get('action_type')}:{selected.get('candidate_index', 0)}"
    repeated_selected = selected_key == last_action_key

    for period, action_type in (
        (5, "inspect_dom"),
        (7, "inspect_layout"),
        (11, "inspect_console"),
        (13, "inspect_network"),
    ):
        if step_index > 0 and step_index % period == 0 and int(counts.get(action_type, 0) or 0) < max(1, step_index // period):
            action_id = _masked_action_id(action_space, action_mask, action_type, 0)
            if action_id is not None and (repeated_selected or action_id != selected_action_id):
                return action_id

    input_index = _first_untried_input_index(observation, history, action_space.max_candidates)
    if input_index is not None:
        action_id = _masked_action_id(action_space, action_mask, "fill_input", input_index)
        if action_id is not None:
            return action_id

    if repeated_selected or int(counts.get("click_element", 0) or 0) == 0:
        click_index = _first_untried_clickable_index(observation, history, action_space.max_candidates)
        if click_index is not None:
            action_id = _masked_action_id(action_space, action_mask, "click_element", click_index)
            if action_id is not None:
                return action_id

    return selected_action_id


def _first_untried_input_index(
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
    max_candidates: int,
) -> Optional[int]:
    candidates = observation.get("candidate_elements", []) or []
    filled = {str(item) for item in history.get("filled_bids", set()) or set()}
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        if not bool(candidate.get("is_form_field")):
            continue
        if not _is_visible_enabled_candidate(candidate):
            continue
        bid = str(candidate.get("bid") or "")
        if bid and bid in filled:
            continue
        return index
    return None


def _first_untried_clickable_index(
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
    max_candidates: int,
) -> Optional[int]:
    candidates = observation.get("candidate_elements", []) or []
    clicked = {str(item) for item in history.get("clicked_bids", set()) or set()}
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        if not bool(candidate.get("clickable") or candidate.get("is_interactive")):
            continue
        if not _is_visible_enabled_candidate(candidate):
            continue
        bid = str(candidate.get("bid") or "")
        if bid and bid in clicked:
            continue
        return index
    return None


def _is_visible_enabled_candidate(candidate: Mapping[str, Any]) -> bool:
    visible = bool(candidate.get("visible", _safe_visibility(candidate) > 0.0)) or _safe_visibility(candidate) > 0.0
    return visible and bool(candidate.get("enabled", True))


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


def _masked_action_id(
    action_space: ActionSpace,
    action_mask: np.ndarray,
    action_type: str,
    candidate_index: int,
) -> Optional[int]:
    try:
        action_id = action_space.encode(action_type, candidate_index)
    except ValueError:
        return None
    if action_id >= len(action_mask) or float(action_mask[action_id]) <= 0.0:
        return None
    return action_id


def _record_seen_anomaly_signatures(history: Dict[str, Any], anomalies: List[Mapping[str, Any]]) -> None:
    seen = history.setdefault("seen_anomaly_keys", set())
    if not isinstance(seen, set):
        seen = set(seen or [])
        history["seen_anomaly_keys"] = seen
    for anomaly in anomalies or []:
        if not isinstance(anomaly, Mapping):
            continue
        signature = anomaly.get("signature")
        if signature:
            seen.add(str(signature))
        evidence = anomaly.get("evidence", {})
        target = ""
        if isinstance(evidence, Mapping):
            target = str(evidence.get("clicked_bid") or evidence.get("selector") or "")
            nested = evidence.get("target")
            if not target and isinstance(nested, Mapping):
                target = str(nested.get("element_key") or nested.get("bid") or nested.get("text") or "")
        seen.add(f"{anomaly.get('type')}:{target}")


def _attach_policy_execution_fields(
    action: Dict[str, Any],
    *,
    policy_action_id: int,
    policy_action: Mapping[str, Any],
    executed_action_id: int,
    fallback_applied: bool,
    fallback_reason: str,
) -> None:
    action.update(
        {
            "policy_action_index": int(policy_action_id),
            "policy_action_type": str(policy_action.get("action_type") or ""),
            "policy_action_signature": str(policy_action.get("action_signature") or ""),
            "policy_action_element_key": str(policy_action.get("action_element_key") or ""),
            "executed_action_index": int(executed_action_id),
            "executed_action_type": str(action.get("action_type") or ""),
            "executed_action_signature": str(action.get("action_signature") or ""),
            "executed_action_element_key": str(action.get("action_element_key") or ""),
            "fallback_applied": bool(fallback_applied),
            "fallback_reason": str(fallback_reason or ""),
        }
    )


def _safe_rate(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator or 0)
        if den <= 0:
            return 0.0
        return float(numerator or 0) / den
    except (TypeError, ValueError):
        return 0.0


def _log_action_mask_stats(site_id: str, step: int, stats: Mapping[str, Any]) -> None:
    print(
        "[mask] "
        f"site={site_id} "
        f"step={step} "
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


def _reward_scale(site: Mapping[str, Any]) -> float:
    value = site.get("reward_scale")
    if value is None:
        site_id = str(site.get("site_id") or "")
        return 0.5 if site_id == "site001" else 1.2 if site_id == "site003" else 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _reward_mode_config(config: Mapping[str, Any], site: Mapping[str, Any]) -> Dict[str, Any]:
    blind_url_training = _is_blind_url_training_config(config)
    return {
        "reward_mode": str(site.get("reward_mode") or config.get("reward_mode") or "signal_based"),
        "allow_known_bug_reward_for_debug_only": False,
        "use_ground_truth_for_evaluation_only": bool(
            site.get(
                "use_ground_truth_for_evaluation_only",
                config.get("use_ground_truth_for_evaluation_only", config.get("use_known_bug_for_evaluation", False)),
            )
        )
        and not blind_url_training,
        "use_known_bug_reward": False,
        "use_known_bug_for_training": False,
        "use_known_bug_for_evaluation": False
        if blind_url_training
        else bool(site.get("use_known_bug_for_evaluation", config.get("use_known_bug_for_evaluation", False))),
        "training_uses_site_specific_bug_catalog": False
        if blind_url_training
        else bool(site.get("training_uses_site_specific_bug_catalog", config.get("training_uses_site_specific_bug_catalog", False))),
        "policy_uses_bug_labels": False,
        "target_signal_types": []
        if blind_url_training
        else list(site.get("target_signal_types") or config.get("target_signal_types") or []),
        "signal_collector_enabled": True,
    }


def _observability_hook_config(site: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "server_log_paths": _string_list(site.get("server_log_paths") or site.get("serverLogPaths")),
        "database_paths": _string_list(
            site.get("database_paths") or site.get("db_paths") or site.get("databasePaths") or site.get("dbPaths")
        ),
        "db_engine": str(site.get("db_engine") or site.get("dbEngine") or ""),
        "relational_db_hook_available": bool(site.get("relational_db_hook_available") or site.get("relationalDbHookAvailable")),
        "postgres_hook_available": bool(site.get("postgres_hook_available") or site.get("postgresHookAvailable")),
        "mysql_hook_available": bool(site.get("mysql_hook_available") or site.get("mysqlHookAvailable")),
        "db_invariant_available": bool(site.get("db_invariant_available") or site.get("dbInvariantAvailable")),
        "trace_collector_available": bool(site.get("trace_collector_available") or site.get("traceCollectorAvailable")),
        "api_contract_available": bool(site.get("api_contract_available") or site.get("apiContractAvailable")),
        "schema_available": bool(site.get("schema_available") or site.get("schemaAvailable")),
        "openapi_spec_available": bool(site.get("openapi_spec_available") or site.get("openapiSpecAvailable")),
        "openapi_spec_path": str(site.get("openapi_spec_path") or site.get("openapiSpecPath") or ""),
        "api_contract": site.get("api_contract") if isinstance(site.get("api_contract"), Mapping) else {},
    }


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _summary_reward_mode_fields(profile: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "reward_mode": str(profile.get("reward_mode") or "signal_based"),
        "allow_known_bug_reward_for_debug_only": bool(profile.get("allow_known_bug_reward_for_debug_only", False)),
        "use_ground_truth_for_evaluation_only": bool(profile.get("use_ground_truth_for_evaluation_only", False)),
        "use_known_bug_reward": False,
        "use_known_bug_for_training": False,
        "use_known_bug_for_evaluation": bool(profile.get("use_known_bug_for_evaluation", False)),
        "training_uses_site_specific_bug_catalog": bool(profile.get("training_uses_site_specific_bug_catalog", False)),
        "policy_uses_bug_labels": False,
    }


def _state_signal_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "known_bug_reward_total": 0.0,
        "signal_reward_total": float(state.get("signal_reward_total", 0.0) or 0.0),
        "exploration_reward_total": float(state.get("exploration_reward_total", 0.0) or 0.0),
        "ui_dom_signal_reward_total": float(state.get("ui_dom_signal_reward_total", 0.0) or 0.0),
        "console_runtime_signal_reward_total": float(state.get("console_runtime_signal_reward_total", 0.0) or 0.0),
        "network_api_signal_reward_total": float(state.get("network_api_signal_reward_total", 0.0) or 0.0),
        "cross_layer_signal_reward_total": float(state.get("cross_layer_signal_reward_total", 0.0) or 0.0),
        "security_signal_reward_total": float(state.get("security_signal_reward_total", 0.0) or 0.0),
        "repeated_penalty_total": float(state.get("repeated_penalty_total", 0.0) or 0.0),
        "multi_signal_anomaly_count": int(state.get("multi_signal_anomaly_count", 0) or 0),
        "console_error_count": int(state.get("console_error_count", 0) or 0),
        "runtime_exception_count": int(state.get("runtime_exception_count", 0) or 0),
        "network_error_count": int(state.get("network_error_count", 0) or 0),
        "api_4xx_count": int(state.get("api_4xx_count", 0) or 0),
        "api_5xx_count": int(state.get("api_5xx_count", 0) or 0),
        "api_timeout_count": int(state.get("api_timeout_count", 0) or 0),
        "api_ui_mismatch_count": int(state.get("api_ui_mismatch_count", 0) or 0),
        "security_signal_count": int(state.get("security_signal_count", 0) or 0),
        "signal_delta_reward_total": float(state.get("signal_delta_reward_total", 0.0) or 0.0),
        "inspect_action_reward_total": float(state.get("inspect_action_reward_total", 0.0) or 0.0),
        "functional_action_signal_reward_total": float(state.get("functional_action_signal_reward_total", 0.0) or 0.0),
        "functional_action_count": int(state.get("functional_action_count", 0) or 0),
        "functional_priority_candidate_count": int(state.get("functional_priority_candidate_count", 0) or 0),
        "functional_priority_click_count": int(state.get("functional_priority_click_count", 0) or 0),
        "semantic_action_type_counts": dict(state.get("semantic_action_type_counts", {}) or {}),
        "clicked_semantic_action_types": dict(state.get("clicked_semantic_action_types", {}) or {}),
        "no_effect_click_to_new_candidate_fallback_count": int(state.get("no_effect_click_to_new_candidate_fallback_count", 0) or 0),
        "unclicked_functional_priority_candidate_count": len(
            (state.get("observed_functional_priority_targets", set()) or set())
            - (state.get("clicked_functional_priority_targets", set()) or set())
        ),
        "high_value_functional_candidate_count": int(state.get("high_value_functional_candidate_count", 0) or 0),
        "first_functional_action_step": state.get("first_functional_action_step"),
        "unique_functional_action_type_count": len(state.get("unique_functional_action_types", set()) or set()),
        "unique_functional_target_count": len(state.get("unique_functional_targets", set()) or set()),
        "debug_meta_action_count": int(state.get("debug_meta_action_count", 0) or 0),
        "open_detail_panel_count": int(state.get("action_counts", {}).get("open_detail_panel", 0) or 0),
        "click_retry_button_count": int(state.get("action_counts", {}).get("click_retry_button", 0) or 0),
        "debug_meta_repeat_count": int(state.get("debug_meta_repeat_count", 0) or 0),
        "no_functional_action_episode_count": int(state.get("no_functional_action_episode_count", 0) or 0),
        "functional_action_signal_delta_count": int(state.get("functional_action_signal_delta_count", 0) or 0),
        "functional_action_network_delta_count": int(state.get("functional_action_network_delta_count", 0) or 0),
        "verification_action_after_high_value_click_count": int(
            state.get("verification_action_after_high_value_click_count", 0) or 0
        ),
        "high_value_click_pending_verification_count": int(state.get("high_value_click_pending_verification_count", 0) or 0),
        "high_value_click_verified_count": int(state.get("high_value_click_verified_count", 0) or 0),
        "high_value_click_without_verification_count": int(state.get("high_value_click_without_verification_count", 0) or 0),
        "reward_functional_action_total": float(state.get("reward_functional_action_total", 0.0) or 0.0),
        "penalty_debug_meta_total": float(state.get("penalty_debug_meta_total", 0.0) or 0.0),
        "target_signal_types": list(state.get("site_profile", {}).get("target_signal_types", []) or []),
        "observed_signal_types": _observed_signal_types_from_state(state),
        "missing_signal_types": sorted(
            set(str(item) for item in state.get("site_profile", {}).get("target_signal_types", []) or [])
            - set(_observed_signal_types_from_state(state))
        ),
        "signal_type_coverage": _signal_type_coverage(
            state.get("site_profile", {}).get("target_signal_types", []) or [],
            _observed_signal_types_from_state(state),
        ),
    }


def _accumulate_reward_breakdown(state: Dict[str, Any], reward_breakdown: Mapping[str, Any]) -> None:
    state["diversity_reward_total"] = float(state.get("diversity_reward_total", 0.0) or 0.0) + float(
        reward_breakdown.get("diversity_reward", 0.0) or 0.0
    )
    state["repeated_action_penalty_total"] = float(state.get("repeated_action_penalty_total", 0.0) or 0.0) + float(
        reward_breakdown.get("repeated_action_penalty", 0.0) or 0.0
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
        "login_form_coverage_reward_total",
        "login_flow_penalty_total",
        "reward_email_input_filled",
        "reward_password_input_filled",
        "reward_submit_clicked",
        "reward_submit_result_checked",
        "penalty_repeated_same_input_fill",
        "penalty_email_repeated_password_pending",
        "penalty_submit_missing",
        "penalty_login_flow_incomplete_early_stop",
        "penalty_targetless_action_success",
        "penalty_inspect_dom_failure_completed",
    ):
        state[key] = float(state.get(key, 0.0) or 0.0) + float(reward_breakdown.get(key, 0.0) or 0.0)
    for key in (
        "first_click_reward_count",
        "new_action_type_reward_count",
        "new_target_reward_count",
        "inspect_reward_count",
        "open_detail_panel_repeat_penalty_count",
        "no_effect_action_penalty_count",
        "multi_signal_anomaly_count",
        "console_error_count",
        "runtime_exception_count",
        "api_4xx_count",
        "api_5xx_count",
        "api_timeout_count",
        "api_ui_mismatch_count",
        "security_signal_count",
        "delta_console_error_count",
        "delta_runtime_exception_count",
        "delta_network_request_failed_count",
        "delta_api_4xx_count",
        "delta_api_5xx_count",
        "delta_api_timeout_count",
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
        "click_retry_button_repeat_penalty_count",
    ):
        state[key] = int(state.get(key, 0) or 0) + int(float(reward_breakdown.get(key, 0.0) or 0.0))
    state["network_error_count"] = int(state.get("network_error_count", 0) or 0) + int(
        float(reward_breakdown.get("network_request_failed_count", 0.0) or 0.0)
    )
    action_type = str(reward_breakdown.get("action_type") or "")
    if bool(reward_breakdown.get("is_functional_action")) and action_type:
        state.setdefault("unique_functional_action_types", set()).add(action_type)
        target_signature = str(reward_breakdown.get("target_signature") or "")
        if target_signature:
            state.setdefault("unique_functional_targets", set()).add(target_signature)
        if state.get("first_functional_action_step") is None:
            state["first_functional_action_step"] = int(state.get("total_steps", 0) or 0) + 1


def _accumulate_candidate_semantics(state: Dict[str, Any], observation: Mapping[str, Any]) -> None:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if not isinstance(candidates, list):
        return
    current_observation_element_keys: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        semantic_type = str(candidate.get("semantic_action_type") or "")
        element_key = str(candidate.get("element_key") or "")
        if element_key:
            observed = state.setdefault("observed_element_keys", set())
            if element_key in current_observation_element_keys:
                state["element_key_collision_count"] = int(state.get("element_key_collision_count", 0) or 0) + 1
            current_observation_element_keys.add(element_key)
            observed.add(element_key)
        else:
            state["element_key_missing_count"] = int(state.get("element_key_missing_count", 0) or 0) + 1
        is_priority = bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority"))
        if semantic_type:
            state.setdefault("semantic_action_type_counts", Counter())[semantic_type] += 1
        if is_priority:
            state["functional_priority_candidate_count"] = int(state.get("functional_priority_candidate_count", 0) or 0) + 1
            state.setdefault("observed_functional_priority_targets", set()).add(_policy_safe_candidate_key(candidate))
        if candidate.get("is_high_value_functional_candidate"):
            state["high_value_functional_candidate_count"] = int(state.get("high_value_functional_candidate_count", 0) or 0) + 1


def _policy_safe_candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("element_key") or candidate.get("bid") or candidate.get("name") or candidate.get("text") or candidate.get("selector_hint") or "")


def _sum_counters(values: Iterable[Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if isinstance(value, Mapping):
            counter.update({str(key): int(count or 0) for key, count in value.items()})
    return dict(counter)


def _action_diversity_score(action_counts: Mapping[str, Any]) -> float:
    total = sum(int(value or 0) for value in action_counts.values())
    if total <= 0:
        return 0.0
    active = sum(1 for value in action_counts.values() if int(value or 0) > 0)
    return min(1.0, active / 6.0)


def _observed_signal_types_from_state_totals(states: Any) -> List[str]:
    combined: Dict[str, Any] = {}
    for state in states:
        if not isinstance(state, Mapping):
            continue
        for key in (
            "multi_signal_anomaly_count",
            "console_error_count",
            "runtime_exception_count",
            "network_error_count",
            "api_4xx_count",
            "api_5xx_count",
            "api_timeout_count",
            "api_ui_mismatch_count",
            "security_signal_count",
        ):
            combined[key] = int(combined.get(key, 0) or 0) + int(state.get(key, 0) or 0)
    return _observed_signal_types_from_state(combined)


def _observed_signal_types_from_state(state: Mapping[str, Any]) -> List[str]:
    observed = []
    if int(state.get("multi_signal_anomaly_count", 0) or 0):
        observed.append("ui-dom")
        observed.append("interaction")
        observed.append("layout")
    if int(state.get("console_error_count", 0) or 0):
        observed.append("console-error")
    if int(state.get("runtime_exception_count", 0) or 0):
        observed.append("runtime-exception")
    if (
        int(state.get("network_error_count", 0) or 0)
    ):
        observed.append("network-error")
    if int(state.get("api_4xx_count", 0) or 0):
        observed.append("api-4xx")
    if int(state.get("api_5xx_count", 0) or 0):
        observed.append("api-5xx")
    if int(state.get("api_timeout_count", 0) or 0):
        observed.append("api-timeout")
    if int(state.get("api_ui_mismatch_count", 0) or 0):
        observed.append("api-ui-mismatch")
    if int(state.get("security_signal_count", 0) or 0):
        observed.extend(["auth-permission-anomaly", "sensitive-data-exposure", "token-exposure"])
    return sorted(set(observed))


def _signal_type_coverage(targets: Any, observed: Any) -> float:
    target_set = {str(item) for item in targets or [] if item}
    observed_set = {str(item) for item in observed or [] if item}
    if not target_set:
        return round(len(observed_set) / 5.0, 4) if observed_set else 0.0
    return round(len(target_set & observed_set) / len(target_set), 4)


def _mean_memory_state_norm(states: Any) -> float:
    values: List[float] = []
    for state in states:
        if not isinstance(state, Mapping):
            continue
        values.extend(float(value) for value in state.get("memory_state_norm_values", []) or [])
    return float(np.mean(values)) if values else 0.0


def _repeated_action_rate(states: Any) -> float:
    repeated = 0
    total = 0
    for state in states:
        if not isinstance(state, Mapping):
            continue
        counts = state.get("action_counts", {})
        if not isinstance(counts, Mapping):
            continue
        for count in counts.values():
            value = int(count or 0)
            total += value
            if value > 1:
                repeated += value - 1
    return round(repeated / total, 4) if total else 0.0


def _repeated_click_target_rate(states: Any) -> float:
    repeated = 0
    total = 0
    for state in states:
        if not isinstance(state, Mapping):
            continue
        total += int(state.get("action_counts", {}).get("click_element", 0) or 0) if isinstance(state.get("action_counts"), Mapping) else 0
        repeated += int(state.get("repeated_click_target_count", 0) or 0)
    return round(repeated / total, 4) if total else 0.0


def _checkpoint_isolated_by_run_id(run_id: str, model_path: Path, output_dir: Path) -> bool:
    normalized = str(run_id or "")
    if not normalized:
        return False
    return normalized in str(model_path).replace("\\", "/") and normalized in str(output_dir).replace("\\", "/")


def _is_blind_url_training_config(config: Mapping[str, Any]) -> bool:
    if bool(config.get("blind_url_training")):
        return True
    run_id = str(config.get("run_id") or config.get("config_name") or "")
    return "blind_url" in run_id or "blind_urls" in run_id


def _reject_known_bug_reward_for_training(config: Mapping[str, Any]) -> None:
    forbidden_keys = ("use_known_bug_reward", "allow_known_bug_reward_for_debug_only", "use_known_bug_for_training")
    for key in forbidden_keys:
        if bool(config.get(key, False)):
            raise ValueError(f"{key}=true is not allowed in PPO training config; ground truth is evaluation-only.")
    sites = config.get("sites", [])
    if not isinstance(sites, list):
        return
    for site in sites:
        if not isinstance(site, Mapping):
            continue
        for key in forbidden_keys:
            if bool(site.get(key, False)):
                site_id = str(site.get("site_id") or "<unknown>")
                raise ValueError(
                    f"{key}=true is not allowed for training site {site_id}; ground truth is evaluation-only."
                )


def _policy_safety_audit(config: Mapping[str, Any], blind_url_training: bool | None = None) -> Dict[str, Any]:
    blind = _is_blind_url_training_config(config) if blind_url_training is None else bool(blind_url_training)
    sites = config.get("sites", [])
    site_items = [item for item in sites if isinstance(item, Mapping)] if isinstance(sites, list) else []
    return {
        "blind_url_training": blind,
        "metadata_used_for_training": False,
        "site_group_used_in_training": False,
        "target_signal_types_used_in_training": False,
        "known_bug_catalog_used_in_training": False,
        "evaluation_reference_loaded_in_training": False,
        "reward_uses_only_observed_signals": True,
        "blind_config_has_site_group": any("site_group" in item for item in site_items),
        "blind_config_has_target_signal_types": bool(config.get("target_signal_types"))
        or any("target_signal_types" in item or "expected_signal_types" in item for item in site_items),
        "blind_config_has_known_bug_catalog": any(
            item.get("has_bug_catalog") or item.get("bug_catalog_path") or item.get("known_bug_catalog")
            for item in site_items
        ),
    }


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.5, float(value)))


def _validate_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("Multi-site config requires a non-empty sites list.")
    sites = []
    for item in raw_sites:
        if not isinstance(item, Mapping):
            continue
        if item.get("enabled") is False:
            continue
        site_id = str(item.get("site_id") or "")
        base_url = str(item.get("base_url") or "")
        if not site_id or not base_url:
            raise ValueError("Each site entry requires site_id and base_url.")
        sites.append(dict(item))
    if not sites:
        raise ValueError("No valid site entries found in multi-site config.")
    return sites


def _configured_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_sites, list):
        return []
    return [dict(item) for item in raw_sites if isinstance(item, Mapping)]


def _read_json_dict(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object config: {path}")
    return data


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _known_bug_id(bug: Mapping[str, Any]) -> str:
    return str(bug.get("id") or bug.get("bug_id") or bug.get("name") or "")


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


def _is_catalog_related(evidence: Any) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    return bool(
        evidence.get("catalog_bug_id_matches")
        or evidence.get("catalog_keyword_matches")
        or evidence.get("section_keyword_matches")
        or evidence.get("target_keyword_matches")
        or evidence.get("selector_hint")
        or evidence.get("data_bug_id")
    )


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _episode_errors(states: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for state in states:
        for item in state.get("episode_errors", state.get("errors", [])) or []:
            if isinstance(item, Mapping):
                errors.append(dict(item))
    return errors


def _episode_exception_count(states: Iterable[Mapping[str, Any]]) -> int:
    return len(_episode_errors(states))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return dict(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
