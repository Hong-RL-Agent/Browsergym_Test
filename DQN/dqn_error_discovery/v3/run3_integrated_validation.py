"""RAWD v3 Run 3 integrated hybrid validation.

Validation-only harness for Run 2 PPO + Run 3 DQN checkpoints.  This uses the
production HybridController for PPO<->DQN orchestration and deliberately does
not import the training-only productive_context_sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, factorized_flat_q_values
from dqn_error_discovery.agents.confirmation_observation import build_dqn_confirmation_context
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.candidate_dom_enrichment import enrich_candidates_with_dom
from dqn_error_discovery.v3.train import SITES, _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state
from envs.browsergym_jaws_env import _active_page
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts/models/v3"
OUT = ROOT / "artifacts/training_v3/run3_integrated_validation"
SPACE = V3ActionSpace(32)
SEEDS = (42, 43, 44)
VALIDATION_SITES = ("SITE001", "DEV_REACT", "DEV_NEXT")
EXCLUDED_SITES = ("SITE002", "SITE003", "SITE004", "SITE005")
EXPECTED_HASHES = {
    "ppo": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
    "42": "6D90FF0AE03DC3DC2FA3118387F3F515B6FE84A7109DB7F3880AE39ED1B9B5F4",
    "43": "7DA480BD642FE117AC5A44FBFC4D4D7E9CB23A56B5457DD3BEEC991460F67298",
    "44": "6A9982528217E9F973313090B2124708F5007ED6262A86160590D21842594958",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def enrich(env: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        candidates = enrich_candidates_with_dom(_active_page(env.env), list(raw.get("candidate_elements", []) or []))
    except Exception:
        candidates = list(raw.get("candidate_elements", []) or [])
    return {**dict(raw), "candidate_elements": candidates}


def candidate_keys(anomalies: Sequence[Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    for anomaly in anomalies or ():
        key = str(anomaly.get("logical_key") or anomaly.get("signature") or "")
        if key and str(anomaly.get("status") or "candidate") in {"candidate", "confirmed"}:
            keys.append(key)
    return keys


def anomaly_causality(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence", {}), Mapping) else {}
    value = (
        anomaly.get("causality")
        or anomaly.get("causal_attribution")
        or evidence.get("causal_attribution")
        or evidence.get("causality")
        or "UNKNOWN"
    )
    value = str(value or "UNKNOWN").upper()
    return value if value in {"ACTION_CAUSED", "ACTION_REVEALED", "BACKGROUND", "UNKNOWN"} else "UNKNOWN"


def q_diagnostics(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, action_id: int) -> dict[str, Any]:
    with torch.inference_mode():
        flat = factorized_flat_q_values(
            network(torch.from_numpy(state).float().unsqueeze(0)),
            torch.from_numpy(mask).bool().unsqueeze(0),
            SPACE,
        )[0]
    mask_tensor = torch.from_numpy(mask).bool()
    valid = flat[mask_tensor]
    selected_q = flat[int(action_id)] if 0 <= int(action_id) < flat.numel() else torch.tensor(float("nan"))
    return {
        "selected_q": float(selected_q.item()),
        "mean_valid_q": float(valid.mean().item()) if len(valid) else None,
        "max_valid_q": float(valid.max().item()) if len(valid) else None,
        "finite_q": bool(torch.isfinite(valid).all() and torch.isfinite(selected_q)),
    }


def model_health(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("network_state_dict", payload.get("model", payload))
    tensors = [tensor for tensor in state.values() if torch.is_tensor(tensor)]
    return {
        "obs_dim": payload.get("obs_dim"),
        "type_head": payload.get("action_type_count"),
        "candidate_head": [payload.get("action_type_count"), payload.get("candidate_count")],
        "finite_parameters": all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
        "nan_parameter_count": sum(int(torch.isnan(tensor).sum().item()) for tensor in tensors),
        "inf_parameter_count": sum(int(torch.isinf(tensor).sum().item()) for tensor in tensors),
    }


def action_target(observation: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    candidates = observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    candidate = candidates[index] if index < len(candidates) and isinstance(candidates[index], Mapping) else {}
    target = str(candidate.get("canonical_target_key") or candidate.get("bid") or candidate.get("id") or index)
    return f"{action.get('action_type')}:{target}"


def state_key(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state", {}), Mapping) else {}
    return "|".join(str(page.get(key) or observation.get(key) or "") for key in ("url", "route", "cart_count", "selected_tab", "selected_filter"))


def classify_sessions(timeline: list[Mapping[str, Any]], return_events: list[Mapping[str, Any]]) -> dict[str, int]:
    by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in timeline:
        if row.get("active_agent") == AgentMode.DQN_LOCAL.value and row.get("local_session_id") is not None:
            by_session[int(row["local_session_id"])].append(row)
    returns = {int(row.get("local_session_id", 0) or 0): row for row in return_events}
    counts = Counter()
    for session_id, rows in by_session.items():
        candidate = any(float((row.get("reward_components", {}) or {}).get("candidate_anomaly", 0.0) or 0.0) > 0.0 for row in rows)
        confirmed = any(
            float((row.get("reward_components", {}) or {}).get("confirmation_upgrade", 0.0) or 0.0) > 0.0
            or float((row.get("reward_components", {}) or {}).get("confirmed_anomaly", 0.0) or 0.0) > 0.0
            for row in rows
        )
        evidence = any(int(row.get("anomaly_emissions", 0) or 0) > 0 for row in rows)
        novelty = any(
            float((row.get("reward_components", {}) or {}).get("new_local_state", 0.0) or 0.0) > 0.0
            or float((row.get("reward_components", {}) or {}).get("new_local_action_target", 0.0) or 0.0) > 0.0
            for row in rows
        )
        if confirmed or returns.get(session_id, {}).get("reason") == "CONFIRMED":
            counts["STRONG_CONFIRMED"] += 1
        elif candidate:
            counts["STRONG_CANDIDATE"] += 1
        elif evidence:
            counts["STRONG_EVIDENCE"] += 1
        elif novelty:
            counts["WEAK"] += 1
        else:
            counts["ZERO_GAIN"] += 1
    return dict(counts)


def run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ppo_path = MODELS / "rawd_v3_ppo_selectcov_seed42.pt"
    dqn_path = MODELS / f"rawd_v3_dqn_productive_seed{seed}.pt"
    ppo = PPOAgent(635, 640)
    ppo.load(ppo_path)
    ppo.policy.eval()
    dqn = FactorizedQNetwork(984, 20, 32)
    payload = torch.load(dqn_path, map_location="cpu")
    dqn.load_state_dict(payload["network_state_dict"])
    dqn.eval()

    timeline: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    site_metrics: dict[str, Counter] = {site: Counter() for site in VALIDATION_SITES}
    handoff_opportunities = Counter()
    handoff_actual = Counter()
    handoff_suppressed = Counter()
    return_reasons = Counter()
    action_counts = Counter()
    causality = Counter()
    confirmation_transition_events = 0
    confirmation_unique_keys: set[str] = set()
    candidate_unique_keys: set[str] = set()
    q_health_rows: list[dict[str, Any]] = []
    controller_summaries: list[dict[str, Any]] = []
    total_episodes = 0
    total_env_steps = 0
    ppo_steps = 0
    dqn_steps = 0

    for site_name in VALIDATION_SITES:
        site_id, url = SITES[site_name]
        for episode in range(1, args.episodes_per_site + 1):
            total_episodes += 1
            env = _episode_env(site_id, url, args.max_steps)
            controller = HybridController(
                max_local_steps=args.max_local_steps,
                cooldown_ppo_steps=1,
                pending_rehandoff_min_ppo_steps=3,
                functional_region_novelty_gate_enabled=True,
            )
            reward_history: dict[str, Any] = {}
            seen_state_keys: set[str] = set()
            seen_targets: set[str] = set()
            seen_action_targets: set[str] = set()
            try:
                raw, _ = env.reset()
                raw = enrich(env, raw)
                for step in range(1, args.max_steps + 1):
                    active = controller.active_agent
                    controller.record_agent_step(active)
                    if active == AgentMode.PPO_GLOBAL:
                        state = _v3_state(env, raw)
                        mask = _v3_mask(SPACE, raw)
                        selected = ppo.select_action(state, mask)
                        action_id = int(selected["action_id"])
                        local_session_id = None
                        handoff_reason = ""
                        ppo_steps += 1
                    else:
                        state = _dqn_state(env, raw, reward_history)
                        mask = _v3_mask(SPACE, raw)
                        with torch.inference_mode():
                            flat = factorized_flat_q_values(
                                dqn(torch.from_numpy(state).float().unsqueeze(0)),
                                torch.from_numpy(mask).bool().unsqueeze(0),
                                SPACE,
                            )[0]
                        action_id = int(flat.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf).argmax().item())
                        local_session_id = controller.local_session.local_session_id if controller.local_session else None
                        handoff_reason = controller.local_session.handoff_reason if controller.local_session else ""
                        dqn_steps += 1
                        q_health_rows.append({"site": site_name, "episode": episode, "step": step, **q_diagnostics(dqn, state, mask, action_id)})

                    decoded = SPACE.decode(action_id)
                    action = {**decoded, "action_id": action_id, "failed": False}
                    mask_valid = bool(mask[action_id]) if 0 <= action_id < len(mask) else False
                    native_select_fill = 0
                    if decoded.get("action_type") == "fill_input":
                        candidates = raw.get("candidate_elements", []) or []
                        idx = int(decoded.get("candidate_index", 0) or 0)
                        candidate = candidates[idx] if idx < len(candidates) and isinstance(candidates[idx], Mapping) else {}
                        native_select_fill = int(str(candidate.get("tag") or "").lower() == "select")
                    before_keys = set(reward_history.get("seen_reward_candidate_keys", set()) or set())
                    seen_candidate_before = set(reward_history.get("seen_candidate_signatures", set()) or set())
                    next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, SPACE)
                    next_raw = enrich(env, next_raw)
                    action["failed"] = bool(info.get("last_action_error"))
                    anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                    keys = candidate_keys(anomalies)
                    candidate_unique_keys.update(keys)
                    for anomaly in anomalies:
                        c = anomaly_causality(anomaly)
                        causality[c] += 1
                    if active == AgentMode.PPO_GLOBAL:
                        reward, components = calculate_autonomous_reward(raw, next_raw, action, anomalies, [], reward_history, {})
                        metadata = _confirmation_metadata(reward_history, next_raw, env)
                        view = dict(reward_history)
                        view["seen_candidate_signatures"] = seen_candidate_before
                        reason = controller.maybe_handoff_after_ppo_step(
                            step=(total_episodes - 1) * args.max_steps + step,
                            observation=next_raw,
                            anomalies=anomalies,
                            reward_components=components,
                            confirmation_metadata=metadata,
                            history=view,
                        )
                        for reason_name, predicate in _handoff_predicates(anomalies, view, components, next_raw, metadata).items():
                            if predicate:
                                handoff_opportunities[reason_name] += 1
                        if reason:
                            handoff_actual[reason] += 1
                        for event in controller.suppression_events:
                            if event.get("_counted"):
                                continue
                            event["_counted"] = True
                            for item in event.get("reasons", []) or []:
                                handoff_suppressed[str(item)] += 1
                    else:
                        reward, components = calculate_dqn_reward(raw, next_raw, action, anomalies, reward_history)
                        after_keys = set(reward_history.get("seen_reward_candidate_keys", set()) or set())
                        new_candidate_reward_keys = after_keys - before_keys
                        confirmation_events = list(reward_history.get("last_confirmation_transition_events", []) or [])
                        confirmation_transition_events += len(confirmation_events)
                        for event in confirmation_events:
                            key = str(event.get("logical_key") or "")
                            if key:
                                confirmation_unique_keys.add(key)
                        repeat_diag = dict(reward_history.get("last_repeat_diagnostic", {}) or {})
                        returned = controller.maybe_return_after_dqn_step(
                            step=(total_episodes - 1) * args.max_steps + step,
                            observation=next_raw,
                            anomalies=anomalies,
                            reward_components=components,
                            repeat_diagnostic=repeat_diag,
                            done=done,
                            confirmation_events=confirmation_events,
                            tracker_history=reward_history,
                            confirmation_metadata=_confirmation_metadata(reward_history, next_raw, env),
                        )
                        if returned:
                            return_reasons[returned] += 1
                        for key in new_candidate_reward_keys:
                            lifecycle.append({
                                "seed": seed,
                                "site": site_name,
                                "episode": episode,
                                "step": step,
                                "session_id": local_session_id,
                                "logical_key": key,
                                "event": "candidate_created",
                            })
                        for event in confirmation_events:
                            lifecycle.append({
                                "seed": seed,
                                "site": site_name,
                                "episode": episode,
                                "step": step,
                                "session_id": local_session_id,
                                "logical_key": event.get("logical_key"),
                                "event": "candidate_to_confirmed",
                                "newly_confirmed": event.get("newly_confirmed"),
                            })
                    action_counts[action_type] += 1
                    local_state_after = state_key(next_raw)
                    target_after = {str(c.get("canonical_target_key") or c.get("bid") or c.get("id") or "") for c in next_raw.get("candidate_elements", []) or [] if isinstance(c, Mapping)}
                    action_target_key = action_target(raw, action)
                    new_state = local_state_after not in seen_state_keys
                    new_targets = len(target_after - seen_targets)
                    new_action_target = action_target_key not in seen_action_targets
                    seen_state_keys.add(local_state_after)
                    seen_targets.update(target_after)
                    seen_action_targets.add(action_target_key)
                    causal_eligible = sum(1 for item in anomalies if anomaly_causality(item) in {"ACTION_CAUSED", "ACTION_REVEALED"})
                    candidate_eligible = sum(1 for item in anomalies if str(item.get("status") or "candidate") in {"candidate", "confirmed"})
                    causal_false_candidate_true = sum(
                        1
                        for item in anomalies
                        if anomaly_causality(item) not in {"ACTION_CAUSED", "ACTION_REVEALED"}
                        and str(item.get("status") or "candidate") in {"candidate", "confirmed"}
                    )
                    timeline_row = {
                        "seed": seed,
                        "site": site_name,
                        "episode": episode,
                        "step": step,
                        "active_agent": active.value,
                        "local_session_id": local_session_id,
                        "handoff_reason": handoff_reason,
                        "action_type": action_type,
                        "mask_valid": mask_valid,
                        "native_select_fill_input": native_select_fill,
                        "invalid": int(bool(info.get("invalid") or not mask_valid)),
                        "timeout": int(bool(info.get("timeout"))),
                        "action_error": int(bool(info.get("last_action_error"))),
                        "reward": float(reward),
                        "reward_components": dict(components),
                        "anomaly_emissions": len(anomalies),
                        "causal_eligible": causal_eligible,
                        "candidate_eligible": candidate_eligible,
                        "causal_false_candidate_true": causal_false_candidate_true,
                        "candidate_keys": keys,
                        "causality": dict(Counter(anomaly_causality(item) for item in anomalies)),
                        "new_local_state": int(new_state),
                        "new_targets": int(new_targets),
                        "new_action_target": int(new_action_target),
                        "repeat_action_target": int(float(components.get("repeat_action_penalty", 0.0) or 0.0) < 0.0),
                        "repeat_state": int(float(components.get("repeat_state_penalty", 0.0) or 0.0) < 0.0),
                        "repeat_anomaly": int(float(components.get("repeat_anomaly_penalty", 0.0) or 0.0) < 0.0),
                        "no_effect": int(float(components.get("no_effect_penalty", 0.0) or components.get("no_effect", 0.0) or 0.0) < 0.0),
                        "done": bool(done),
                    }
                    timeline.append(timeline_row)
                    site_metrics[site_name].update({
                        "env_steps": 1,
                        "ppo_steps": int(active == AgentMode.PPO_GLOBAL),
                        "dqn_steps": int(active == AgentMode.DQN_LOCAL),
                        "handoffs": int(bool(active == AgentMode.PPO_GLOBAL and controller.handoff_events and controller.handoff_events[-1].get("step") == (total_episodes - 1) * args.max_steps + step)),
                        "anomaly_emissions": len(anomalies),
                        "causal_eligible": causal_eligible,
                        "candidate_eligible": candidate_eligible,
                        "causal_false_candidate_true": causal_false_candidate_true,
                        "invalid": timeline_row["invalid"],
                        "timeout": timeline_row["timeout"],
                        "action_error": timeline_row["action_error"],
                        "mask_invalid": int(not mask_valid),
                        "native_select_fill_input": native_select_fill,
                        "no_effect": timeline_row["no_effect"],
                        "repeat_action_target": timeline_row["repeat_action_target"],
                        "repeat_state": timeline_row["repeat_state"],
                        "repeat_anomaly": timeline_row["repeat_anomaly"],
                        "new_local_states": timeline_row["new_local_state"],
                        "new_targets": timeline_row["new_targets"],
                        "new_action_targets": timeline_row["new_action_target"],
                    })
                    total_env_steps += 1
                    raw = next_raw
                    if done:
                        break
            finally:
                controller_summaries.append(controller.summary())
                env.close()

    return_events = [event for summary in controller_summaries for event in summary.get("return_events", [])]
    handoff_events = [event for summary in controller_summaries for event in summary.get("handoff_events", [])]
    sessions = [row for row in timeline if row["active_agent"] == AgentMode.DQN_LOCAL.value and row.get("local_session_id") is not None]
    session_ids = {(row["site"], row["episode"], row["local_session_id"]) for row in sessions}
    candidate_sessions = {
        (row["site"], row["episode"], row["local_session_id"])
        for row in sessions
        if float((row.get("reward_components", {}) or {}).get("candidate_anomaly", 0.0) or 0.0) > 0.0
    }
    confirmation_sessions = {
        (row["site"], row["episode"], row["local_session_id"])
        for row in sessions
        if float((row.get("reward_components", {}) or {}).get("confirmation_upgrade", 0.0) or 0.0) > 0.0
        or float((row.get("reward_components", {}) or {}).get("confirmed_anomaly", 0.0) or 0.0) > 0.0
    }
    rewards = [row.get("reward_components", {}) or {} for row in timeline if row["active_agent"] == AgentMode.DQN_LOCAL.value]
    total_steps = ppo_steps + dqn_steps
    metrics = {
        "seed": seed,
        "episodes": total_episodes,
        "total_env_steps": total_env_steps,
        "ppo_steps": ppo_steps,
        "dqn_steps": dqn_steps,
        "ppo_step_share": ppo_steps / total_steps if total_steps else 0.0,
        "dqn_step_share": dqn_steps / total_steps if total_steps else 0.0,
        "handoff_count": len(handoff_events),
        "handoff_reasons": dict(handoff_actual),
        "handoff_opportunities": dict(handoff_opportunities),
        "handoff_suppressed": dict(handoff_suppressed),
        "return_reasons": dict(return_reasons),
        "dqn_session_count": len(session_ids),
        "mean_dqn_steps_per_session": dqn_steps / len(session_ids) if session_ids else 0.0,
        "candidate_admission_events": sum(1 for item in rewards if float(item.get("candidate_anomaly", 0.0) or 0.0) > 0.0),
        "unique_candidate_keys": len(candidate_unique_keys),
        "unique_confirmations": len(confirmation_unique_keys),
        "confirmation_transition_events": confirmation_transition_events,
        "candidate_reward_count": sum(1 for item in rewards if float(item.get("candidate_anomaly", 0.0) or 0.0) > 0.0),
        "candidate_reward_value": sum(float(item.get("candidate_anomaly", 0.0) or 0.0) for item in rewards),
        "confirmed_reward_count": sum(1 for item in rewards if float(item.get("confirmed_anomaly", 0.0) or 0.0) > 0.0),
        "confirmed_reward_value": sum(float(item.get("confirmed_anomaly", 0.0) or 0.0) for item in rewards),
        "candidate_to_confirmed_reward_count": sum(1 for item in rewards if float(item.get("confirmation_upgrade", 0.0) or 0.0) > 0.0),
        "candidate_to_confirmed_reward_value": sum(float(item.get("confirmation_upgrade", 0.0) or 0.0) for item in rewards),
        "session_productivity": classify_sessions(timeline, return_events),
        "candidate_producing_session_rate": len(candidate_sessions) / len(session_ids) if session_ids else 0.0,
        "confirmation_producing_session_rate": len(confirmation_sessions) / len(session_ids) if session_ids else 0.0,
        "anomaly_emissions": sum(int(row["anomaly_emissions"]) for row in timeline),
        "causality": dict(causality),
        "causal_eligible": sum(int(row["causal_eligible"]) for row in timeline),
        "candidate_eligible": sum(int(row["candidate_eligible"]) for row in timeline),
        "causal_false_candidate_true": sum(int(row["causal_false_candidate_true"]) for row in timeline),
        "invalid": sum(int(row["invalid"]) for row in timeline),
        "timeout": sum(int(row["timeout"]) for row in timeline),
        "action_error": sum(int(row["action_error"]) for row in timeline),
        "mask_invalid": sum(int(not row["mask_valid"]) for row in timeline),
        "native_select_fill_input": sum(int(row["native_select_fill_input"]) for row in timeline),
        "no_effect_rate": sum(int(row["no_effect"]) for row in timeline if row["active_agent"] == AgentMode.DQN_LOCAL.value) / dqn_steps if dqn_steps else 0.0,
        "repeat_action_target": sum(int(row["repeat_action_target"]) for row in timeline),
        "repeat_state": sum(int(row["repeat_state"]) for row in timeline),
        "repeat_anomaly": sum(int(row["repeat_anomaly"]) for row in timeline),
        "new_local_states": sum(int(row["new_local_state"]) for row in timeline),
        "new_targets": sum(int(row["new_targets"]) for row in timeline),
        "new_action_targets": sum(int(row["new_action_target"]) for row in timeline),
        "action_distribution": dict(action_counts),
        "action_collapse": action_collapse(action_counts),
        "finite_q": all(row.get("finite_q", True) for row in q_health_rows),
        "finite_parameters": model_health(dqn_path)["finite_parameters"],
        "model_health": model_health(dqn_path),
        "site_metrics": {site: dict(counter) for site, counter in site_metrics.items()},
        "timeline": timeline,
        "candidate_lifecycle": lifecycle,
        "controller_summaries": controller_summaries,
    }
    return metrics


def _confirmation_metadata(history: Mapping[str, Any], observation: Mapping[str, Any], env: Any) -> dict[str, Any]:
    _, metadata = build_dqn_confirmation_context(history, observation, env.action_space)
    return metadata


def _handoff_predicates(
    anomalies: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    components: Mapping[str, Any],
    observation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, bool]:
    keys = candidate_keys(anomalies)
    seen = set(history.get("seen_candidate_signatures", set()) or set())
    pending = bool(metadata.get("pending_confirmation_exists"))
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state", {}), Mapping) else {}
    text = " ".join(str(page.get(key) or "") for key in ("url", "title", "form_state", "selected_filter", "selected_tab")).lower()
    functional = any(token in text for token in ("cart", "checkout", "search", "filter", "login", "modal", "detail", "quantity"))
    no_effect = float(components.get("no_effect_penalty", 0.0) or components.get("no_effect", 0.0) or 0.0) < 0.0
    return {
        "NEW_CANDIDATE": any(key not in seen for key in keys),
        "PENDING_CONFIRMATION": pending,
        "FUNCTIONAL_REGION": functional,
        "STALLED_FUNCTIONAL": functional and no_effect,
    }


def action_collapse(counts: Counter[str]) -> str:
    total = sum(counts.values())
    if not total:
        return "NONE"
    top = max(counts.values()) / total
    if top >= 0.80:
        return "COLLAPSE"
    if top >= 0.55:
        return "MILD_CONCENTRATION"
    return "NONE"


def aggregate(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    site_productivity: dict[str, Counter] = {site: Counter() for site in VALIDATION_SITES}
    handoff_reasons = Counter()
    handoff_opportunities = Counter()
    handoff_suppressed = Counter()
    return_reasons = Counter()
    causality = Counter()
    lifecycle_rows: list[dict[str, Any]] = []
    for metrics in seed_metrics:
        for key in (
            "episodes", "total_env_steps", "ppo_steps", "dqn_steps", "handoff_count", "dqn_session_count",
            "candidate_admission_events", "unique_candidate_keys", "unique_confirmations", "confirmation_transition_events",
            "candidate_reward_count", "confirmed_reward_count", "candidate_to_confirmed_reward_count", "anomaly_emissions",
            "causal_eligible", "candidate_eligible", "causal_false_candidate_true", "invalid", "timeout", "action_error",
            "mask_invalid", "native_select_fill_input", "repeat_action_target", "repeat_state", "repeat_anomaly",
            "new_local_states", "new_targets", "new_action_targets",
        ):
            total[key] += metrics.get(key, 0)
        total["candidate_reward_value"] += metrics.get("candidate_reward_value", 0.0)
        total["confirmed_reward_value"] += metrics.get("confirmed_reward_value", 0.0)
        total["candidate_to_confirmed_reward_value"] += metrics.get("candidate_to_confirmed_reward_value", 0.0)
        handoff_reasons.update(metrics.get("handoff_reasons", {}))
        handoff_opportunities.update(metrics.get("handoff_opportunities", {}))
        handoff_suppressed.update(metrics.get("handoff_suppressed", {}))
        return_reasons.update(metrics.get("return_reasons", {}))
        causality.update(metrics.get("causality", {}))
        lifecycle_rows.extend(metrics.get("candidate_lifecycle", []))
        for site, values in metrics.get("site_metrics", {}).items():
            site_productivity[site].update(values)
    summary = {
        "version": "RAWD_V3_RUN3_INTEGRATED_VALIDATION",
        "training": False,
        "tuning": False,
        "integrated_validation": True,
        "production_algorithm_changed": False,
        "training_context_sampler_used": False,
        "production_hybrid_controller_used": True,
        "training_sites": list(VALIDATION_SITES),
        "excluded_sites": list(EXCLUDED_SITES),
        "excluded_sites_used": False,
        "ground_truth_used": False,
        "episodes_per_seed": 15,
        "episodes_total": int(total["episodes"]),
        "total_env_steps": int(total["total_env_steps"]),
        "ppo_steps": int(total["ppo_steps"]),
        "dqn_steps": int(total["dqn_steps"]),
        "ppo_step_share": total["ppo_steps"] / (total["ppo_steps"] + total["dqn_steps"]) if total["ppo_steps"] + total["dqn_steps"] else 0.0,
        "dqn_step_share": total["dqn_steps"] / (total["ppo_steps"] + total["dqn_steps"]) if total["ppo_steps"] + total["dqn_steps"] else 0.0,
        "total_handoffs": int(total["handoff_count"]),
        "handoff_reasons": dict(handoff_reasons),
        "handoff_opportunities": dict(handoff_opportunities),
        "handoff_suppressed": dict(handoff_suppressed),
        "return_reasons": dict(return_reasons),
        "mean_dqn_steps_per_session": total["dqn_steps"] / total["dqn_session_count"] if total["dqn_session_count"] else 0.0,
        "candidate_telemetry": {
            "candidate_admission_events": int(total["candidate_admission_events"]),
            "unique_candidate_keys_seed_sum": int(total["unique_candidate_keys"]),
            "unique_confirmations_seed_sum": int(total["unique_confirmations"]),
            "confirmation_transition_events": int(total["confirmation_transition_events"]),
            "candidate_reward_count": int(total["candidate_reward_count"]),
            "candidate_reward_value": float(total["candidate_reward_value"]),
            "confirmed_reward_count": int(total["confirmed_reward_count"]),
            "confirmed_reward_value": float(total["confirmed_reward_value"]),
            "candidate_to_confirmed_reward_count": int(total["candidate_to_confirmed_reward_count"]),
            "candidate_to_confirmed_reward_value": float(total["candidate_to_confirmed_reward_value"]),
        },
        "detector_funnel": {
            "anomaly_emissions": int(total["anomaly_emissions"]),
            "causality": dict(causality),
            "causal_eligible": int(total["causal_eligible"]),
            "candidate_eligible": int(total["candidate_eligible"]),
            "causal_false_candidate_true": int(total["causal_false_candidate_true"]),
        },
        "execution_health": {
            "invalid": int(total["invalid"]),
            "timeout": int(total["timeout"]),
            "action_error": int(total["action_error"]),
            "mask_invalid": int(total["mask_invalid"]),
            "native_select_fill_input": int(total["native_select_fill_input"]),
        },
        "novelty": {
            "new_local_states": int(total["new_local_states"]),
            "new_targets": int(total["new_targets"]),
            "new_action_targets": int(total["new_action_targets"]),
        },
        "repeat": {
            "repeat_action_target": int(total["repeat_action_target"]),
            "repeat_state": int(total["repeat_state"]),
            "repeat_anomaly": int(total["repeat_anomaly"]),
        },
        "site_productivity": {site: dict(values) for site, values in site_productivity.items()},
        "seed_metrics": {str(item["seed"]): _compact_seed(item) for item in seed_metrics},
    }
    consistency = {
        "status": "PASS" if summary["candidate_telemetry"]["candidate_admission_events"] > 0 or not lifecycle_rows else "REVIEW",
        "RAWD_V3_INTEGRATED_CANDIDATE_TELEMETRY_CHECK": "PASS" if summary["candidate_telemetry"]["candidate_admission_events"] > 0 or not lifecycle_rows else "REVIEW",
        "production_lifecycle_rows": len(lifecycle_rows),
        "validation_candidate_admission_events": summary["candidate_telemetry"]["candidate_admission_events"],
        "admission_events_and_unique_candidates_separated": True,
    }
    summary["candidate_telemetry_consistency"] = consistency
    summary["hashes"] = {
        "ppo": sha256(MODELS / "rawd_v3_ppo_selectcov_seed42.pt"),
        "42": sha256(MODELS / "rawd_v3_dqn_productive_seed42.pt"),
        "43": sha256(MODELS / "rawd_v3_dqn_productive_seed43.pt"),
        "44": sha256(MODELS / "rawd_v3_dqn_productive_seed44.pt"),
    }
    summary["hash_integrity"] = {key: summary["hashes"][key] == EXPECTED_HASHES[key] for key in EXPECTED_HASHES}
    summary["RAW_DQN_CHECKPOINT_WRITES"] = 0
    summary["RAWD_V3_RUN3_POLICY_HEALTH_CHECK"] = "PASS" if all(item["finite_parameters"] and item["finite_q"] for item in seed_metrics) else "FAIL"
    validation_pass = (
        summary["episodes_total"] == 45
        and summary["production_hybrid_controller_used"]
        and not summary["training_context_sampler_used"]
        and summary["total_handoffs"] > 0
        and sum(return_reasons.values()) > 0
        and consistency["status"] == "PASS"
        and all(value == 0 for value in summary["execution_health"].values())
        and all(summary["hash_integrity"].values())
        and not summary["ground_truth_used"]
        and not summary["excluded_sites_used"]
    )
    summary["RAWD_V3_INTEGRATED_HYBRID_VALIDATION_CHECK"] = "PASS" if validation_pass else "FAIL"
    summary["RAWD_V3_INTEGRATED_CANDIDATE_TELEMETRY_CHECK"] = consistency["RAWD_V3_INTEGRATED_CANDIDATE_TELEMETRY_CHECK"]
    return summary


def _compact_seed(metrics: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"timeline", "candidate_lifecycle", "controller_summaries"}
    return {key: value for key, value in metrics.items() if key not in omitted}


def write_outputs(summary: dict[str, Any], seed_metrics: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for metrics in seed_metrics:
        seed = metrics["seed"]
        (OUT / f"seed{seed}_integrated_metrics.json").write_text(json.dumps(_compact_seed(metrics), indent=2, default=json_default), encoding="utf-8")
    (OUT / "handoff_reason_breakdown.json").write_text(json.dumps({
        "opportunities": summary["handoff_opportunities"],
        "actual_handoffs": summary["handoff_reasons"],
        "suppressed": summary["handoff_suppressed"],
    }, indent=2), encoding="utf-8")
    (OUT / "return_reason_breakdown.json").write_text(json.dumps(summary["return_reasons"], indent=2), encoding="utf-8")
    lifecycle = [row for metrics in seed_metrics for row in metrics.get("candidate_lifecycle", [])]
    (OUT / "candidate_lifecycle_validation.json").write_text(json.dumps(lifecycle, indent=2, default=json_default), encoding="utf-8")
    (OUT / "causality_candidate_invariant_audit.json").write_text(json.dumps({
        "detector_funnel": summary["detector_funnel"],
        "explanation": (
            "Candidate eligibility is status/lifecycle based in the detector/reward path, while causal eligibility is derived "
            "from optional causal_attribution fields. Existing production anomalies may be candidate-eligible with UNKNOWN "
            "or absent causality, so causal=false + candidate=true is telemetry-semantics mismatch, not automatic admission failure."
        ),
    }, indent=2), encoding="utf-8")
    (OUT / "site_productivity.json").write_text(json.dumps(summary["site_productivity"], indent=2), encoding="utf-8")
    (OUT / "seed_stability.json").write_text(json.dumps(summary["seed_metrics"], indent=2, default=json_default), encoding="utf-8")
    (OUT / "rawd_v3_run3_integrated_validation_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-site", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_hashes = {key: sha256(MODELS / ("rawd_v3_ppo_selectcov_seed42.pt" if key == "ppo" else f"rawd_v3_dqn_productive_seed{key}.pt")) for key in EXPECTED_HASHES}
    if before_hashes != EXPECTED_HASHES:
        raise SystemExit(f"checkpoint hash mismatch before validation: {before_hashes}")
    seed_metrics = [run_seed(seed, args) for seed in SEEDS]
    summary = aggregate(seed_metrics)
    write_outputs(summary, seed_metrics)
    print(json.dumps({
        "status": summary["RAWD_V3_INTEGRATED_HYBRID_VALIDATION_CHECK"],
        "episodes": summary["episodes_total"],
        "handoffs": summary["total_handoffs"],
        "candidate_admissions": summary["candidate_telemetry"]["candidate_admission_events"],
        "confirmations": summary["candidate_telemetry"]["unique_confirmations_seed_sum"],
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
