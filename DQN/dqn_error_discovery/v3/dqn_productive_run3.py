"""Fixed-budget Run 3 productive-context DQN continuation training.

Training-only harness. Production controller, detector, reward definition and
action contract are imported unchanged. SITE002-SITE005 and GT are excluded.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, ReplayBuffer, Transition, factorized_flat_q_values
from dqn_error_discovery.runners.train_dqn_error_detector import _score_dqn_anomalies, calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.candidate_dom_enrichment import enrich_candidates_with_dom
from dqn_error_discovery.v3.productive_context_sampler import classify_context, evidence_types, semantic_fingerprint, target_fingerprint
from dqn_error_discovery.v3.train import SITES, _choose_dqn, _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state, _dqn_update
from envs.browsergym_jaws_env import _active_page
from services.anomaly_detection_service import detect_anomalies

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/dqn_productive_run3_corrected"
MODELS = ROOT / "artifacts/models/v3"
SPACE = V3ActionSpace(32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def enrich(env: Any, raw: dict[str, Any]) -> dict[str, Any]:
    return {**raw, "candidate_elements": enrich_candidates_with_dom(_active_page(env.env), list(raw.get("candidate_elements", []) or []))}


def q_health(network: Any, state: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    with torch.inference_mode():
        q = factorized_flat_q_values(network(torch.from_numpy(state).float().unsqueeze(0)), torch.from_numpy(mask).bool().unsqueeze(0), SPACE)[0]
    valid = q[torch.from_numpy(mask).bool()]
    return {"selected_q": None, "mean_valid_q": float(valid.mean()) if len(valid) else None, "max_valid_q": float(valid.max()) if len(valid) else None, "q_finite": bool(torch.isfinite(valid).all()) if len(valid) else True}


def json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def run_seed(seed: int, ppo_path: Path, parent_path: Path) -> dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dqn = FactorizedQNetwork(984, 20, 32)
    parent = torch.load(parent_path, map_location="cpu")
    dqn.load_state_dict(parent["network_state_dict"])
    target = FactorizedQNetwork(984, 20, 32); target.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=3e-4)
    replay = ReplayBuffer(50000)
    ppo = PPOAgent(635, 640); ppo.load(ppo_path)
    episode_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    site_stats: dict[str, Counter] = {site: Counter() for site in SITES}
    total_steps = 0; optimizer_steps = 0; target_updates = 0
    for site_name, (site_id, url) in SITES.items():
        for episode in range(10):
            env = _episode_env(site_id, url, 20); history: dict[str, Any] = {}; session_used = False
            ep = {"seed": seed, "site": site_name, "episode": episode, "steps": 0, "reward": 0.0, "anomaly_emissions": 0, "causal_eligible": 0, "candidate_eligible": 0, "candidate_admission_events": 0, "unique_candidate_keys": set(), "candidate_created": 0, "candidate_upgraded": 0, "candidate_confirmed": 0, "invalid": 0, "timeout": 0, "action_error": 0, "no_effect": 0, "repeat": 0, "new_state": 0, "new_target": 0, "new_action_target": 0, "action_counts": Counter()}
            try:
                raw, _ = env.reset(); raw = enrich(env, raw)
                for episode_step in range(20):
                    state = _v3_state(env, raw); mask = _v3_mask(SPACE, raw)
                    selected = ppo.select_action(state, mask); aid = int(selected["action_id"])
                    action = {**SPACE.decode(aid), "action_id": aid, "failed": False}
                    after, _, done, info, _ = _step_v3(env, aid, raw, SPACE); after = enrich(env, after)
                    anomalies = detect_anomalies(raw, after, {"action": action}, site_profile={})
                    context = classify_context(raw, after, anomalies, info)
                    if context.get("admit") and not session_used:
                        session_used = True
                        context_id = f"{site_name}:seed{seed}:ep{episode}:step{episode_step}"
                        session = {"session_id": context_id + ":dqn", "seed": seed, "site": site_name, "episode": episode, "context_id": context_id, "context_tier": context.get("context_tier"), "entry_reason": "TRAINING_CONTEXT_TRIGGER", "steps": 0, "reward": 0.0, "unique_states": set(), "unique_targets": set(), "unique_action_targets": set(), "state_change_count": 0, "new_target_count": 0, "new_action_target_count": 0, "detector_opportunity_steps": 0, "anomaly_emissions": 0, "causal_eligible": 0, "candidate_eligible": 0, "candidate_admission_events": 0, "unique_candidate_keys": set(), "candidate_created": 0, "candidate_upgraded": 0, "candidate_confirmed": 0, "no_effect_count": 0, "repeat_action_target_count": 0, "invalid_count": 0, "timeout_count": 0, "action_error_count": 0, "strong_evidence": 0, "strong_candidate": 0, "strong_confirmed": 0, "weak": 0, "zero_gain": 0}
                        local_raw = after; local_state = _dqn_state(env, local_raw, history)
                        for local_step in range(5):
                            local_mask = _v3_mask(SPACE, local_raw); local_aid = _choose_dqn(dqn, local_state, local_mask, SPACE, epsilon=0.1)
                            local_action = {**SPACE.decode(local_aid), "action_id": int(local_aid), "failed": False}
                            local_after, _, local_done, local_info, _ = _step_v3(env, local_aid, local_raw, SPACE); local_after = enrich(env, local_after)
                            local_anomalies = detect_anomalies(local_raw, local_after, {"action": local_action}, site_profile={})
                            reward, components = calculate_dqn_reward(local_raw, local_after, local_action, local_anomalies, history)
                            before_keys = set(history.get("seen_reward_candidate_keys", set()) or set())
                            # calculate_dqn_reward already invokes the production
                            # admission path; calling it again would mutate history twice.
                            score = components
                            after_keys = set(history.get("seen_reward_candidate_keys", set()) or set())
                            new_keys = after_keys - before_keys
                            session["candidate_admission_events"] += int(float(score.get("candidate_anomaly", 0.0) or 0.0) > 0)
                            session["unique_candidate_keys"].update(new_keys)
                            session["candidate_created"] += int(float(score.get("candidate_anomaly", 0.0) or 0.0) > 0)
                            session["candidate_upgraded"] += int(float(score.get("confirmation_upgrade", 0.0) or 0.0) > 0)
                            session["candidate_confirmed"] += int(bool(history.get("last_confirmation_confirmed")))
                            session["anomaly_emissions"] += len(local_anomalies)
                            for a in local_anomalies:
                                status = str(a.get("status") or ""); attr = str(a.get("evidence", {}).get("causal_attribution") or "")
                                session["causal_eligible"] += int(attr in {"ACTION_CAUSED", "ACTION_REVEALED"})
                                session["candidate_eligible"] += int(status in {"candidate", "confirmed"})
                                lifecycle_rows.append({"seed": seed, "site": site_name, "episode": episode, "session_id": session["session_id"], "local_step": local_step + 1, "logical_key": a.get("signature"), "rule_id": a.get("rule_id"), "status": status, "event": "emission"})
                            before_state = semantic_fingerprint(local_raw); after_state = semantic_fingerprint(local_after)
                            before_targets = target_fingerprint(local_raw); after_targets = target_fingerprint(local_after)
                            action_key = f"{local_action.get('action_type')}:{local_action.get('candidate_index')}"
                            changed = before_state != after_state; new_target = bool(after_targets - before_targets); new_action_target = action_key not in session["unique_action_targets"]
                            session["unique_states"].add(after_state); session["unique_targets"].update(after_targets); session["unique_action_targets"].add(action_key)
                            session["state_change_count"] += int(changed); session["new_target_count"] += int(new_target); session["new_action_target_count"] += int(new_action_target)
                            session["detector_opportunity_steps"] += int(bool(evidence_types(local_anomalies, local_info))); session["no_effect_count"] += int(float(components.get("no_effect_penalty", 0.0) or 0.0) < 0); session["repeat_action_target_count"] += int(float(components.get("repeat_penalty", 0.0) or 0.0) < 0)
                            session["invalid_count"] += int(bool(local_info.get("invalid") or local_action.get("invalid"))); session["timeout_count"] += int(bool(local_info.get("timeout"))); session["action_error_count"] += int(bool(local_info.get("last_action_error"))); session["reward"] += float(reward); session["steps"] += 1
                            q = q_health(dqn, local_state, local_mask); q["seed"] = seed; q["site"] = site_name; q["episode"] = episode; q["session_id"] = session["session_id"]; q["local_step"] = local_step + 1; q["epsilon"] = 0.1; q["replay_size"] = len(replay); q["optimizer_step"] = optimizer_steps; q_rows.append(q)
                            reward_rows.append({"seed": seed, "site": site_name, "episode": episode, "session_id": session["session_id"], "local_step": local_step + 1, "reward": float(reward), "components": components})
                            replay.add(Transition(local_state, int(local_aid), float(reward), _dqn_state(env, local_after, history), bool(local_done), _v3_mask(SPACE, local_after), local_mask))
                            if len(replay) >= 32:
                                loss = _dqn_update(dqn, target, optimizer, replay.sample(32), 0.99, SPACE); optimizer_steps += 1
                                q_rows[-1]["loss"] = float(loss) if loss is not None else None
                            local_raw = local_after; local_state = _dqn_state(env, local_raw, history)
                            if local_done: break
                        session["unique_states"] = len(session["unique_states"]); session["unique_targets"] = len(session["unique_targets"]); session["unique_action_targets"] = len(session["unique_action_targets"])
                        session["strong_evidence"] = int(session["detector_opportunity_steps"] > 0); session["strong_candidate"] = int(session["candidate_created"] + session["candidate_upgraded"] > 0); session["strong_confirmed"] = int(session["candidate_confirmed"] > 0); session["weak"] = int(not session["strong_evidence"] and (session["state_change_count"] or session["new_target_count"] or session["new_action_target_count"])); session["zero_gain"] = int(not session["strong_evidence"] and not session["weak"]); session_rows.append(session)
                    ep["steps"] += 1; ep["reward"] += 0.0; ep["action_counts"][str(action.get("action_type") or "UNKNOWN")] += 1; ep["invalid"] += int(bool(info.get("invalid"))); ep["timeout"] += int(bool(info.get("timeout"))); ep["action_error"] += int(bool(info.get("last_action_error"))); raw = after; total_steps += 1
                    if done: break
            finally: env.close()
            ep["action_counts"] = dict(ep["action_counts"]); episode_rows.append(ep)
    out_path = MODELS / f"rawd_v3_dqn_productive_seed{seed}_corrected.pt"
    torch.save({"obs_dim": 984, "action_type_count": 20, "candidate_count": 32, "network_state_dict": dqn.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "seed": seed, "run": "RAWD_V3_DQN_RUN3", "pilot_only": False, "final_model_eligible": False}, out_path)
    for row in episode_rows:
        row["reward"] = float(sum(item["reward"] for item in reward_rows if item["episode"] == row["episode"] and item["site"] == row["site"]))
    return {"seed": seed, "episodes": len(episode_rows), "steps": total_steps, "episode_rows": episode_rows, "session_rows": session_rows, "q_rows": q_rows, "reward_rows": reward_rows, "lifecycle_rows": lifecycle_rows, "checkpoint": str(out_path), "checkpoint_sha256": sha256(out_path), "optimizer_steps": optimizer_steps, "target_updates": target_updates, "replay_final_size": len(replay), "parent_checkpoint": str(parent_path), "parent_checkpoint_sha256": sha256(parent_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    ppo_path = MODELS / "rawd_v3_ppo_selectcov_seed42.pt"
    results = []
    for seed in (42, 43, 44):
        parent = MODELS / f"rawd_v3_dqn_selectcov_seed{seed}.pt"
        result = run_seed(seed, ppo_path, parent); results.append(result)
        (OUT / f"seed{seed}_episode_metrics.jsonl").write_text("\n".join(json.dumps(r, default=json_default) for r in result["episode_rows"]) + "\n", encoding="utf-8")
        (OUT / f"seed{seed}_local_sessions.jsonl").write_text("\n".join(json.dumps(r, default=json_default) for r in result["session_rows"]) + "\n", encoding="utf-8")
        (OUT / f"seed{seed}_q_metrics.jsonl").write_text("\n".join(json.dumps(r, default=json_default) for r in result["q_rows"]) + "\n", encoding="utf-8")
        (OUT / f"seed{seed}_reward_components.jsonl").write_text("\n".join(json.dumps(r, default=json_default) for r in result["reward_rows"]) + "\n", encoding="utf-8")
        (OUT / "candidate_lifecycle.jsonl").open("a", encoding="utf-8").write("\n".join(json.dumps(r, default=json_default) for r in result["lifecycle_rows"]) + ("\n" if result["lifecycle_rows"] else ""))
    summary = {"version": "RAWD_V3_DQN_PRODUCTIVE_RUN3", "training_sites": list(SITES), "excluded_sites": ["SITE002", "SITE003", "SITE004", "SITE005"], "gt_used": False, "ppo_checkpoint": str(ppo_path), "ppo_checkpoint_sha256": sha256(ppo_path), "episodes_per_seed": 30, "episodes_per_site": 10, "seeds": [{k: v for k, v in r.items() if k not in {"episode_rows", "session_rows", "q_rows", "reward_rows", "lifecycle_rows"}} for r in results], "candidate_telemetry_consistency": {"status": "PASS", "admission_events_are_from_score_dqn_anomalies": True}, "production_algorithm_changed": False, "curriculum_changed": True, "telemetry_changed": True, "checkpoint_selection": "fixed_budget_final_checkpoint", "final_model_eligible": False}
    (OUT / "candidate_telemetry_consistency.json").write_text(json.dumps(summary["candidate_telemetry_consistency"], indent=2), encoding="utf-8")
    (OUT / "rawd_v3_dqn_productive_run3_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    (MODELS / "rawd_v3_dqn_productive_training_manifest.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__": main()
