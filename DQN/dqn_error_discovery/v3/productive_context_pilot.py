"""Fixed 15-episode, training-only productive-context pilot."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, ReplayBuffer, Transition, factorized_flat_q_values
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.candidate_dom_enrichment import enrich_candidates_with_dom
from dqn_error_discovery.v3.productive_context_sampler import classify_context, semantic_fingerprint, target_fingerprint
from dqn_error_discovery.v3.train import SITES, _choose_dqn, _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state
from envs.browsergym_jaws_env import _active_page
from services.anomaly_detection_service import detect_anomalies
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/productive_context_pilot"
SPACE = V3ActionSpace(32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def enrich(env, raw):
    return {**raw, "candidate_elements": enrich_candidates_with_dom(_active_page(env.env), list(raw.get("candidate_elements", []) or []))}


def q_metrics(network, state, mask, action_id, reward, next_state, next_mask, gamma=0.99):
    with torch.inference_mode():
        current = factorized_flat_q_values(network(torch.from_numpy(state).float().unsqueeze(0)), torch.from_numpy(mask).bool().unsqueeze(0), SPACE)[0]
        nxt = factorized_flat_q_values(network(torch.from_numpy(next_state).float().unsqueeze(0)), torch.from_numpy(next_mask).bool().unsqueeze(0), SPACE)[0]
        next_value = float(nxt[torch.from_numpy(next_mask).bool()].max()) if np.any(next_mask) else 0.0
        selected = float(current[action_id])
        target = float(reward + gamma * next_value)
    return {"q_selected": selected, "q_mean_valid": float(current[torch.from_numpy(mask).bool()].mean()), "q_max_valid": float(current[torch.from_numpy(mask).bool()].max()), "td_error": abs(selected - target)}


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    ppo_path = ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt"
    dqn_parent = ROOT / "artifacts/models/v3/rawd_v3_dqn_selectcov_seed42.pt"
    ppo = PPOAgent(635, 640)
    ppo.load(ppo_path)
    dqn = FactorizedQNetwork(984, 20, 32)
    dqn.load_state_dict(torch.load(dqn_parent, map_location="cpu")["network_state_dict"])
    target = FactorizedQNetwork(984, 20, 32)
    target.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=3e-4)
    replay = ReplayBuffer(50000)
    contexts, sessions, reward_rows, q_rows = [], [], [], []
    tier_counts = Counter()
    site_stats = {name: Counter() for name in SITES}
    total_steps = 0
    skipped = 0
    for site_name, (site_id, url) in SITES.items():
        for episode in range(5):
            env = _episode_env(site_id, url, 20)
            history = {}
            session_used = False
            try:
                raw, _ = env.reset()
                raw = enrich(env, raw)
                for episode_step in range(20):
                    state = _v3_state(env, raw)
                    mask = _v3_mask(SPACE, raw)
                    selected = ppo.select_action(state, mask)
                    action_id = int(selected["action_id"])
                    action = {**SPACE.decode(action_id), "action_id": action_id, "failed": False}
                    next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, SPACE)
                    next_raw = enrich(env, next_raw)
                    anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                    context = classify_context(raw, next_raw, anomalies, info)
                    if context.get("admit") and not session_used:
                        context_id = f"{site_name}:ep{episode}:step{episode_step}"
                        record = {"context_id": context_id, "site": site_name, "episode": episode, "global_step": total_steps, "route": next_raw.get("page_state", {}).get("route", ""), "state_fingerprint": context.get("after_state"), "source_agent": "PPO", "handoff_reason": "TRAINING_CONTEXT_TRIGGER", "context_tier": context.get("context_tier"), "candidate_present": context.get("candidate_present", False), "pending_confirmation": False, "detector_opportunity_present": bool(context.get("evidence_types")), "detector_evidence_types": context.get("evidence_types", []), "functional_region_key": "UNAVAILABLE_IN_ARTIFACT", "new_state": context.get("state_changed", False), "new_target": context.get("new_target", False), "new_action_target": False, "new_evidence": bool(context.get("evidence_types")), "valid_action_count": int(mask.sum()), "valid_target_count": int(np.count_nonzero(mask[1 * 32:])), "previous_local_sessions_in_region": 0, "previous_productive_result": ""}
                        contexts.append(record)
                        tier_counts[record["context_tier"]] += 1
                        site_stats[site_name]["contexts"] += 1
                        session_used = True
                        session = run_local_session(env, dqn, target, optimizer, replay, raw=next_raw, context=record, episode=episode, start_step=episode_step, total_steps=total_steps, history=history)
                        sessions.append(session["summary"])
                        reward_rows.extend(session["reward_rows"])
                        q_rows.extend(session["q_rows"])
                        for key, value in session["summary"].items():
                            if key in {"strong", "weak", "zero_gain", "state_change_count", "new_target_count", "new_action_target_count", "detector_opportunity_steps", "candidate_created", "candidate_upgraded", "candidate_confirmed", "no_effect_count", "repeat_action_target_count", "invalid_count", "timeout_count", "action_error_count"}:
                                site_stats[site_name][key] += int(value or 0)
                    elif not context.get("admit") and context.get("skip_reason"):
                        skipped += 1
                    raw = next_raw
                    total_steps += 1
                    if done:
                        break
            finally:
                env.close()
    checkpoint = ROOT / "artifacts/models/v3/rawd_v3_dqn_productive_context_pilot_seed42.pt"
    torch.save({"obs_dim": 984, "action_type_count": 20, "candidate_count": 32, "network_state_dict": dqn.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "seed": 42, "pilot_only": True, "eligible_for_final_model": False}, checkpoint)
    funnel = {"productive_contexts": len(contexts), "local_sessions": len(sessions), "state_change_sessions": sum(1 for row in sessions if row.get("state_change_count", 0)), "detector_opportunity_steps": sum(row.get("detector_opportunity_steps", 0) for row in sessions), "candidate_created": sum(row.get("candidate_created", 0) for row in sessions), "candidate_confirmed": sum(row.get("candidate_confirmed", 0) for row in sessions)}
    summary = {"version": "RAWD_V3_PRODUCTIVE_CONTEXT_PILOT", "pilot_sites": list(SITES), "episodes_per_site": 5, "total_steps": total_steps, "tier_counts": dict(tier_counts), "admitted_contexts": len(contexts), "skipped_repeated_zero_gain_contexts": skipped, "site_stats": {key: dict(value) for key, value in site_stats.items()}, "funnel": funnel, "mean_local_steps": float(np.mean([row.get("steps", 0) for row in sessions])) if sessions else 0.0, "strong": sum(row.get("strong", 0) for row in sessions), "weak": sum(row.get("weak", 0) for row in sessions), "zero_gain": sum(row.get("zero_gain", 0) for row in sessions), "invalid": sum(row.get("invalid_count", 0) for row in sessions), "timeout": sum(row.get("timeout_count", 0) for row in sessions), "action_error": sum(row.get("action_error_count", 0) for row in sessions), "mean_reward": float(np.mean([row.get("reward_total", 0.0) for row in sessions])) if sessions else 0.0, "q_telemetry_count": len(q_rows), "reward_component_rows": len(reward_rows), "parent_dqn_checkpoint": str(dqn_parent), "parent_dqn_sha256": sha256(dqn_parent), "pilot_checkpoint": str(checkpoint), "pilot_checkpoint_sha256": sha256(checkpoint), "pilot_only": True, "eligible_for_final_model": False, "gt_used": False, "site002_to_site005_used": False, "action_contract_regression": False}
    (OUT / "context_availability.json").write_text(json.dumps({"tier_counts": dict(tier_counts), "admitted_contexts": contexts, "skipped_repeated_zero_gain_contexts": skipped}, indent=2), encoding="utf-8")
    (OUT / "local_sessions.jsonl").write_text("\n".join(json.dumps(row) for row in sessions) + "\n", encoding="utf-8")
    (OUT / "reward_components.jsonl").write_text("\n".join(json.dumps(row) for row in reward_rows) + "\n", encoding="utf-8")
    (OUT / "q_training_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in q_rows) + "\n", encoding="utf-8")
    (OUT / "candidate_funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    (OUT / "site_breakdown.json").write_text(json.dumps(summary["site_stats"], indent=2), encoding="utf-8")
    (OUT / "rawd_v3_productive_context_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_local_session(env, network, target, optimizer, replay, raw, context, episode, start_step, total_steps, history):
    reward_rows, q_rows = [], []
    state = _dqn_state(env, raw, history)
    session_id = f"{context['context_id']}:dqn"
    summary = {"session_id": session_id, "context_id": context["context_id"], "context_tier": context["context_tier"], "entry_reason": context["handoff_reason"], "steps": 0, "unique_states": set(), "unique_targets": set(), "unique_action_targets": set(), "state_change_count": 0, "new_target_count": 0, "new_action_target_count": 0, "detector_opportunity_steps": 0, "candidate_created": 0, "candidate_upgraded": 0, "candidate_confirmed": 0, "no_effect_count": 0, "repeat_action_target_count": 0, "invalid_count": 0, "timeout_count": 0, "action_error_count": 0, "reward_total": 0.0, "strong": 0, "weak": 0, "zero_gain": 0}
    for local_step in range(5):
        mask = _v3_mask(SPACE, raw)
        action_id = _choose_dqn(network, state, mask, SPACE, epsilon=0.1)
        action = {**SPACE.decode(action_id), "action_id": action_id, "failed": False}
        next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, SPACE)
        next_raw = enrich(env, next_raw)
        anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
        reward, components = calculate_dqn_reward(raw, next_raw, action, anomalies, history)
        next_state = _dqn_state(env, next_raw, history)
        next_mask = _v3_mask(SPACE, next_raw)
        replay.add(Transition(state, action_id, float(reward), next_state, bool(done), next_mask, mask))
        loss = None
        if len(replay) >= 32:
            from dqn_error_discovery.v3.train import _dqn_update
            loss = _dqn_update(network, target, optimizer, replay.sample(32), 0.99, SPACE)
        q = q_metrics(network, state, mask, action_id, reward, next_state, next_mask)
        q_rows.append({"session_id": session_id, "local_step": local_step + 1, "epsilon": 0.1, "replay_size": len(replay), "loss": loss, **q})
        before_state = semantic_fingerprint(raw)
        after_state = semantic_fingerprint(next_raw)
        before_targets = target_fingerprint(raw)
        after_targets = target_fingerprint(next_raw)
        action_target = f"{action_type}:{SPACE.decode(action_id).get('candidate_index')}"
        changed = before_state != after_state
        new_target = bool(after_targets - before_targets)
        new_action_target = action_target not in summary["unique_action_targets"]
        summary["unique_states"].add(after_state)
        summary["unique_targets"].update(after_targets)
        summary["unique_action_targets"].add(action_target)
        summary["state_change_count"] += int(changed)
        summary["new_target_count"] += int(new_target)
        summary["new_action_target_count"] += int(new_action_target)
        from dqn_error_discovery.v3.productive_context_sampler import evidence_types
        evidence = evidence_types(anomalies, info)
        summary["detector_opportunity_steps"] += int(bool(evidence))
        candidate = any(item.get("candidate") or item.get("is_candidate") for item in anomalies if isinstance(item, dict))
        confirmed = any(item.get("confirmed") or str(item.get("status")) == "confirmed" for item in anomalies if isinstance(item, dict))
        summary["candidate_created"] += int(candidate)
        summary["candidate_confirmed"] += int(confirmed)
        summary["no_effect_count"] += int(float(components.get("no_effect_penalty", 0.0)) < 0)
        summary["repeat_action_target_count"] += int(float(components.get("repeat_penalty", 0.0)) < 0)
        summary["invalid_count"] += int(bool(action.get("invalid") or info.get("invalid")))
        summary["action_error_count"] += int(bool(info.get("last_action_error")))
        summary["reward_total"] += float(reward)
        reward_rows.append({"session_id": session_id, "local_step": local_step + 1, "reward": reward, "components": components})
        raw, state = next_raw, next_state
        summary["steps"] += 1
        if done:
            break
    summary["strong"] = int(summary["candidate_created"] or summary["candidate_confirmed"] or summary["detector_opportunity_steps"])
    summary["weak"] = int(not summary["strong"] and (summary["state_change_count"] or summary["new_target_count"] or summary["new_action_target_count"]))
    summary["zero_gain"] = int(not summary["strong"] and not summary["weak"])
    summary["unique_states"] = len(summary["unique_states"])
    summary["unique_targets"] = len(summary["unique_targets"])
    summary["unique_action_targets"] = len(summary["unique_action_targets"])
    return {"summary": summary, "reward_rows": reward_rows, "q_rows": q_rows}


if __name__ == "__main__":
    main()
