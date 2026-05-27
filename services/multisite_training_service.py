"""Shared PPO training across multiple BrowserGym sites."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
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
from services.browsergym_training_service import (
    _catalog_guided_action_id,
    _enrich_action,
    _first_openended_candidate_index,
    _guided_action_id,
    _has_openended_interactive_candidate,
    _target_bid,
    _update_history,
)
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs
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
    ) -> None:
        self.config_path = Path(config_path)
        self.config = _read_json_dict(self.config_path)
        self.sites = _validate_sites(self.config.get("sites", []))
        self.output_dir = Path(str(self.config.get("output_dir") or "artifacts/multisite"))
        configured_model_path = self.config.get("shared_model_path") or "artifacts/models/jaws_browsergym_shared_ppo.pt"
        self.shared_model_path = Path(save_model_path or configured_model_path)
        self.load_model_path = Path(load_model_path) if load_model_path else None
        self.total_updates = int(total_updates)
        self.episodes_per_site = int(episodes_per_site)
        self.max_steps = int(max_steps)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)
        self.headless = bool(headless)
        self.entropy_coef = max(0.02, float(entropy_coef))
        self.strict_site_validation = bool(strict_site_validation)

        self.encoder = ObservationEncoder(max_candidates=max_candidates)
        self.action_space = ActionSpace(max_candidates=max_candidates)
        self.agent = PPOAgent(
            self.encoder.get_obs_dim(),
            self.action_space.get_action_dim(),
            entropy_coef=self.entropy_coef,
        )
        initial_model_path = self.load_model_path or (self.shared_model_path if self.shared_model_path.exists() else None)
        if initial_model_path and initial_model_path.exists():
            self.agent.load(initial_model_path)

        self.site_states: Dict[str, Dict[str, Any]] = {}
        for site in self.sites:
            site_id = site["site_id"]
            site_dir = self.output_dir / site_id
            known_bugs = _load_known_bugs_for_site(site)
            site_profile = build_site_profile(
                site_id,
                known_bugs,
                exploration_profile=site.get("exploration_profile"),
            )
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
                "preflight": {},
            }

    def train(self) -> Dict[str, Any]:
        _set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shared_model_path.parent.mkdir(parents=True, exist_ok=True)
        for state in self.site_states.values():
            state["output_dir"].mkdir(parents=True, exist_ok=True)
            state["transition_log_path"].write_text("", encoding="utf-8")
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
                    episode = self._collect_episode(site, state, episode_id)
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
        print(f"[multisite-train] summary: {summary_path}")
        return summary

    def _collect_episode(self, site: Mapping[str, Any], state: Dict[str, Any], episode_id: str) -> Dict[str, Any]:
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

        try:
            observation, _ = env.reset()
            for step in range(1, self.max_steps + 1):
                obs_vector = self.encoder.encode_observation(observation)
                action_mask = self.action_space.build_action_mask(observation)
                selected = self.agent.select_action(obs_vector, action_mask)
                action_id = _guided_multisite_action_id(
                    self.action_space,
                    observation,
                    history,
                    selected["action_id"],
                    training=True,
                )
                if action_id != selected["action_id"]:
                    selected = self.agent.score_action(obs_vector, action_mask, action_id)

                action = self.action_space.decode(action_id)
                action["action_id"] = action_id
                action["site_id"] = site_id
                _enrich_action(action, observation)

                next_observation, _, done, step_info = env.step(action_id)
                action["failed"] = bool(step_info.get("last_action_error"))
                anomalies = detect_anomalies(
                    observation,
                    next_observation,
                    {"action": action, "site_profile": state["site_profile"], **step_info},
                    site_profile=state["site_profile"],
                )
                known_matches = match_anomalies_to_known_bugs(anomalies, state["known_bugs"], site_id=site_id)
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
                    state["site_profile"],
                )
                raw_reward = reward
                scaled_reward = _clamp_reward(raw_reward * reward_scale)
                reward_breakdown["raw_final_reward"] = raw_reward
                reward_breakdown["scaled_final_reward"] = scaled_reward
                reward_breakdown["reward_scale"] = reward_scale

                buffer.add(
                    obs_vector,
                    action_id,
                    selected["log_prob"],
                    scaled_reward,
                    done,
                    selected["value"],
                    action_mask,
                    step_info,
                )
                episode_reward += raw_reward
                scaled_episode_reward += scaled_reward
                state["raw_reward_total"] += raw_reward
                state["scaled_reward_total"] += scaled_reward
                episode_anomaly_count += len(anomalies)
                state["total_steps"] += 1
                state["total_anomalies"] += len(anomalies)
                state["action_counts"][str(action.get("action_type") or "")] += 1
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
                        "catalog_matches": {
                            "candidate_catalog_matches": action.get("catalog_bug_id_matches", []),
                            "candidate_keyword_matches": action.get("catalog_keyword_matches", []),
                            "anomaly_catalog_bug_id_matches": [
                                anomaly.get("evidence", {}).get("catalog_bug_id_matches", [])
                                for anomaly in anomalies
                            ],
                        },
                        "anomalies": anomalies,
                        "done": done,
                    },
                )
                _update_history(history, observation, action, anomalies)
                observation = next_observation
                if done:
                    break
        except Exception as exc:
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
                    "anomalies": [{"type": "episode-exception", "confidence": 1.0, "evidence": {"error": str(exc)}}],
                    "done": True,
                },
            )
            done = True
            print(f"  [site {site_id}] {episode_id} exception: {exc}")
        finally:
            env.close()

        if len(buffer) > 0:
            last_value = 0.0 if done else self.agent.estimate_value(self.encoder.encode_observation(observation))
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
            "inspect_cart_count": int(state["action_counts"].get("inspect_cart", 0)),
            "last_update": dict(last_update_metrics),
            "transition_log_path": str(state["transition_log_path"]),
            "preflight": dict(state.get("preflight", {})),
        }

    def _run_preflight_checks(self) -> None:
        print("[multisite-preflight]")
        for site in self.sites:
            site_id = str(site["site_id"])
            state = self.site_states[site_id]
            result: Dict[str, Any] = {
                "site_id": site_id,
                "base_url": site.get("base_url"),
                "has_bug_catalog": bool(site.get("has_bug_catalog")),
                "evaluation_mode": "known_bug_catalog" if bool(site.get("has_bug_catalog")) else "openended_anomaly_discovery",
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
            "config_path": str(self.config_path),
            "shared_model_path": str(self.shared_model_path),
            "output_dir": str(self.output_dir),
            "total_updates": self.total_updates,
            "episodes_per_site": self.episodes_per_site,
            "max_steps": self.max_steps,
            "average_reward_across_sites": float(np.mean(rewards)) if rewards else 0.0,
            "sites": site_summaries,
            "updates": update_summaries,
            "last_update": dict(last_update_metrics),
        }

    @staticmethod
    def _append_transition(path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


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


def _guided_multisite_action_id(
    action_space: ActionSpace,
    observation: Dict[str, Any],
    history: Dict[str, Any],
    selected_action_id: int,
    training: bool = True,
) -> int:
    site_id = _site_id_from_observation(observation)
    if site_id == "site001":
        return _guided_action_id(action_space, observation, history, selected_action_id)

    selected = action_space.decode(selected_action_id)
    if site_id not in {"site001", "site9800"} and selected.get("action_type") == "inspect_cart":
        selected_action_id = action_space.encode("noop", 0)

    counts = history.get("action_type_counts", {})
    step_index = int(observation.get("history", {}).get("step_index", 0) or 0)
    if _has_openended_interactive_candidate(observation) and not _has_catalog_candidate(observation):
        consecutive_counts = history.get("consecutive_action_type_counts", {})
        click_count = int(counts.get("click_element", 0) or 0) if isinstance(counts, Mapping) else 0
        inspect_dom_count = int(counts.get("inspect_dom", 0) or 0) if isinstance(counts, Mapping) else 0
        inspect_layout_count = int(counts.get("inspect_layout", 0) or 0) if isinstance(counts, Mapping) else 0
        if click_count == 0 and step_index <= 5:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
        if (
            inspect_dom_count >= 5
            or (
                isinstance(consecutive_counts, Mapping)
                and int(consecutive_counts.get("inspect_dom", 0) or 0) >= 2
            )
        ):
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
            return action_space.encode("inspect_layout", 0)
        if training and random.random() < 0.25:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
        if inspect_layout_count >= 8:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
            if inspect_dom_count < 5:
                return action_space.encode("inspect_dom", 0)
    if _has_catalog_candidate(observation):
        catalog_action = _catalog_guided_action_id(action_space, observation, history)
        if catalog_action is not None:
            return catalog_action
    if isinstance(counts, Mapping) and int(counts.get("click_element", 0) or 0) >= 20:
        if _has_chart_related_candidate(observation):
            return action_space.encode("inspect_layout", 0)
        if _page_has_catalog_empty_state(observation):
            return action_space.encode("inspect_dom", 0)
    catalog_click_index = _first_clickable_catalog_index(observation) or _first_candidate_index(observation, "catalog_bug_id_matches")
    if catalog_click_index is not None and not history.get("catalog_target_clicked"):
        if training and step_index <= 5 and random.random() < 0.3:
            return action_space.encode("click_element", catalog_click_index)
        if not training and int(counts.get("click_element", 0) or 0) == 0:
            return action_space.encode("click_element", catalog_click_index)

    if _page_has_catalog_empty_state(observation):
        if training and random.random() < 0.2:
            return action_space.encode("inspect_dom", 0)
        if int(counts.get("inspect_dom", 0) or 0) == 0:
            return action_space.encode("inspect_dom", 0)

    if _has_chart_related_candidate(observation):
        if training and random.random() < 0.2:
            return action_space.encode("inspect_layout", 0)
        if int(counts.get("inspect_layout", 0) or 0) == 0:
            return action_space.encode("inspect_layout", 0)

    return selected_action_id


def _first_candidate_index(observation: Mapping[str, Any], flag: str) -> Optional[int]:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if isinstance(candidate, Mapping) and bool(candidate.get(flag)):
            return index
    return None


def _first_clickable_catalog_index(observation: Mapping[str, Any]) -> Optional[int]:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for index, candidate in enumerate(candidates[:32]):
        if (
            isinstance(candidate, Mapping)
            and bool(candidate.get("catalog_bug_id_matches"))
            and bool(candidate.get("clickable"))
            and float(candidate.get("visibility", 0.0) or 0.0) > 0.0
        ):
            return index
    return None


def _has_catalog_candidate(observation: Mapping[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, Mapping) and bool(candidate.get("catalog_bug_id_matches")) for candidate in candidates
    )


def _site_id_from_observation(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime_signals = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    site_id = str(
        (page_state.get("site_id") if isinstance(page_state, Mapping) else "")
        or (runtime_signals.get("site_id") if isinstance(runtime_signals, Mapping) else "")
        or ""
    )
    if site_id:
        return site_id
    url = str(page_state.get("url") or "") if isinstance(page_state, Mapping) else ""
    return "site003" if ":9222" in url else "site001" if ":9220" in url else ""


def _page_has_catalog_empty_state(observation: Mapping[str, Any]) -> bool:
    page_state = observation.get("page_state", {})
    if isinstance(page_state, Mapping) and page_state.get("has_empty_state_text"):
        return True
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, Mapping) and bool(candidate.get("is_empty_state_related")) for candidate in candidates
    )


def _has_chart_related_candidate(observation: Mapping[str, Any]) -> bool:
    candidates = observation.get("candidate_elements", []) or []
    return isinstance(candidates, list) and any(
        isinstance(candidate, Mapping)
        and (
            candidate.get("is_chart_related")
            or candidate.get("is_weekly_stats_related")
            or candidate.get("is_empty_state_related")
        )
        for candidate in candidates
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


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.5, float(value)))


def _validate_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("Multi-site config requires a non-empty sites list.")
    sites = []
    for item in raw_sites:
        if not isinstance(item, Mapping):
            continue
        site_id = str(item.get("site_id") or "")
        base_url = str(item.get("base_url") or "")
        if not site_id or not base_url:
            raise ValueError("Each site entry requires site_id and base_url.")
        sites.append(dict(item))
    if not sites:
        raise ValueError("No valid site entries found in multi-site config.")
    return sites


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
