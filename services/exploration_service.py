"""Read-only PPO exploration service used by the HTTP job API."""
from __future__ import annotations
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
import numpy as np
import torch
from agents.ppo_agent import PPOAgent
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from runners.evaluate_browsergym_agent import _enrich_action
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.coverage_service import compare_coverage, graph_from_transitions, load_graph
from services.exploration_identifiers import action_id, normalized_url, state_id, transition_id
from services.general_policy import select_policy_action, update_general_history
from services.risk_scoring_service import score_anomaly
from services.site_profile_service import build_site_profile

EventCallback = Callable[[dict[str, Any]], None]

class ExplorationService:
    def __init__(self, *, session_id: str, target_url: str, model_path: str, output_dir: str, mode: str = "coverage", episodes: int = 3, max_steps: int = 25, headless: bool = True, allowed_hosts: list[str] | None = None, allow_destructive_actions: bool = False, reference_graph_path: str | None = None, on_event: EventCallback | None = None) -> None:
        self.session_id, self.target_url, self.mode = session_id, target_url, mode
        self.model_path, self.output_dir = Path(model_path), Path(output_dir)
        self.episodes, self.max_steps, self.headless = max(1, episodes), max(1, max_steps), headless
        self.allowed_hosts = allowed_hosts or ["localhost", "127.0.0.1", "::1"]
        self.allow_destructive_actions = allow_destructive_actions
        self.reference_graph_path, self.on_event = reference_graph_path, on_event or (lambda event: None)
        self._validate_target()

    def run(self) -> dict[str, Any]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        transition_path = self.output_dir / "transitions.jsonl"
        screenshot_dir = self.output_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        transition_path.write_text("", encoding="utf-8")
        encoder, action_space = ObservationEncoder(), ActionSpace()
        agent = PPOAgent(encoder.get_obs_dim(), action_space.get_action_dim())
        agent.load(self.model_path)
        site_id = f"job-{self.session_id[:12]}"
        site_profile = build_site_profile(site_id, [])
        rows: list[dict[str, Any]] = []
        anomaly_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        episode_rewards, action_counts = [], Counter()
        started = time.monotonic()
        total_budget = self.episodes * self.max_steps
        completed_steps = 0

        for episode in range(1, self.episodes + 1):
            env = BrowserGymJAWSEnv(site_id=site_id, base_url=self.target_url, max_steps=self.max_steps, headless=self.headless, site_profile=site_profile)
            history: dict[str, Any] = {"step_index": 0, "action_type_counts": {}, "visited_candidates": set(), "fault_inputs": set()}
            reward_total = 0.0
            try:
                observation, _ = env.reset()
                for step in range(1, self.max_steps + 1):
                    before_id = state_id(observation)
                    mask = action_space.build_action_mask(observation)
                    selected = agent.select_greedy_action(encoder.encode_observation(observation), mask)
                    selected_id = select_policy_action(action_space, observation, history, selected["action_id"], self.mode, self.allow_destructive_actions)
                    action = action_space.decode(selected_id)
                    action.update({"action_id": selected_id, "site_id": site_id, "policy": "general-v1" if selected_id != selected["action_id"] else "ppo"})
                    _enrich_action(action, observation)
                    candidates = observation.get("candidate_elements", []) or []
                    index = int(action.get("candidate_index", 0) or 0)
                    candidate = candidates[index] if isinstance(candidates, list) and 0 <= index < len(candidates) else {}
                    stable_action_id = action_id(before_id, action, candidate)
                    next_observation, _, done, info = env.step(selected_id)
                    after_id = state_id(next_observation)
                    stable_transition_id = transition_id(before_id, stable_action_id, after_id)
                    anomalies = detect_anomalies(observation, next_observation, {"action": action, "site_profile": site_profile, **info}, site_profile=site_profile)
                    reward, _ = calculate_autonomous_reward(observation, next_observation, action, anomalies, [], history, site_profile)
                    reward_total += reward
                    row = {"schema_version": "exploration-log-v1", "session_id": self.session_id, "episode": episode, "step": step, "state_id_before": before_id, "action_id": stable_action_id, "transition_id": stable_transition_id, "state_id_after": after_id, "url_before": normalized_url(str(observation.get("page_state", {}).get("url") or "")), "url_after": normalized_url(str(next_observation.get("page_state", {}).get("url") or "")), "action_type": action.get("action_type"), "candidate": _compact_candidate(candidate), "policy": action.get("policy"), "new_state": before_id != after_id, "success": not bool(info.get("last_action_error")), "anomalies": anomalies, "reward": reward, "timestamp_ms": round(time.time() * 1000)}
                    screenshot_file = f"episode-{episode:02d}-step-{step:03d}.png"
                    if env.capture_screenshot(str(screenshot_dir / screenshot_file)):
                        row["screenshot_file"] = screenshot_file
                    rows.append(row)
                    with transition_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    for anomaly in anomalies: anomaly_occurrences[_anomaly_key(anomaly)].append({"episode": episode, "step": step, **anomaly})
                    action_counts[str(action.get("action_type") or "unknown")] += 1
                    update_general_history(history, observation, action)
                    observation = next_observation
                    completed_steps += 1
                    self.on_event({"type": "step", "progress": min(99, round(completed_steps * 100 / total_budget)), "episode": episode, "step": step, "action": action.get("action_type"), "action_id": stable_action_id, "policy": action.get("policy"), "url": row["url_after"], "success": row["success"], "new_state": row["new_state"], "candidate_count": len(candidates), "anomaly_count": len(anomalies), "anomalies": anomalies, "screenshot_file": row.get("screenshot_file")})
                    if done: break
            finally:
                env.close()
            episode_rewards.append(reward_total)

        findings = []
        for key, occurrences in anomaly_occurrences.items():
            best = max(occurrences, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
            reproduced_episodes = len({int(item["episode"]) for item in occurrences})
            finding = {"fingerprint": key, "type": best.get("type"), "confidence": best.get("confidence"), "evidence": best.get("evidence", {}), "occurrences": len(occurrences), "risk": score_anomaly(best, self.episodes, reproduced_episodes)}
            findings.append(finding)
        findings.sort(key=lambda item: item["risk"]["score"], reverse=True)
        graph = graph_from_transitions(rows)
        reference = load_graph(self.reference_graph_path)
        coverage = compare_coverage(graph, reference)
        elapsed = max(0.001, time.monotonic() - started)
        result = {"session_id": self.session_id, "status": "completed", "mode": self.mode, "target_url": self.target_url, "episodes": self.episodes, "steps": len(rows), "duration_seconds": round(elapsed, 3), "average_reward": round(float(np.mean(episode_rewards)) if episode_rewards else 0.0, 4), "action_counts": dict(action_counts), "coverage": coverage, "graph": graph, "findings": findings, "metrics": {"error_yield": round(len(findings) / max(1, len(rows)), 4), "exploration_efficiency": round((coverage["visited_states"] + 2 * len(findings)) / max(1, len(rows)), 4), "steps_per_second": round(len(rows) / elapsed, 3)}, "artifacts": {"transitions": str(transition_path), "result": str(self.output_dir / "result.json")}}
        (self.output_dir / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.on_event({"type": "complete", "progress": 100, "finding_count": len(findings)})
        return result

    def _validate_target(self) -> None:
        parsed = urlsplit(self.target_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname: raise ValueError("target_url must be an absolute HTTP(S) URL")
        if parsed.hostname.lower() not in {host.lower() for host in self.allowed_hosts}: raise ValueError(f"target host is not allowed: {parsed.hostname}")
        if not self.model_path.exists(): raise FileNotFoundError(f"model not found: {self.model_path}")

def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {key: candidate.get(key) for key in ("role", "name", "text", "bid", "selector", "type") if candidate.get(key) not in (None, "")}

def _anomaly_key(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    return "|".join(str(value or "").lower()[:160] for value in (anomaly.get("type"), evidence.get("before_url"), evidence.get("clicked_text"), evidence.get("selector"), evidence.get("error")))
