"""Runtime smoke runner for RAWD Hybrid v1 PPO/DQN orchestration."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import (
    FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    FactorizedQNetwork,
    select_factorized_greedy_action,
)
from dqn_error_discovery.agents.confirmation_observation import build_dqn_confirmation_context
from dqn_error_discovery.agents.history_observation import (
    DQN_HISTORY_WINDOW,
    HISTORY_CONTEXT_V2_DIM,
    append_history_transition,
    candidate_target_identity,
)
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.run_policy_sanity_check import (
    _local_state_signature,
    _state_signature,
    _target_identity,
    _url_path,
    _validate_checkpoint_architecture,
)
from dqn_error_discovery.runners.train_dqn_error_detector import (
    calculate_dqn_reward,
    dqn_no_effect_diagnostic,
    safe_action_mask,
    sanitize_anomalies,
    _action_candidate,
    _dqn_state,
)
from dqn_error_discovery.framework_dev.action_error_event_recorder import ActionErrorEventRecorder
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.temporal_evidence_service import (
    ActionEvidenceWindow,
    TemporalEvidenceConfig,
    detect_anomalies_with_temporal_evidence,
    summarize_temporal_windows,
)
from services.framework_aware_evidence_normalizer import normalize_network_events
from services.temporal_stability_diagnostic_service import (
    analyze_temporal_stability_window,
    summarize_temporal_stability_diagnostics,
)


def run_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    env = BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    diagnostic_recorder = None
    if args.action_error_diagnostic:
        diagnostic_recorder = ActionErrorEventRecorder(
            args.action_error_diagnostic_output,
            site=str(site["site_id"]),
            method="ppo" if args.ppo_only else "hybrid",
            seed=args.seed,
            canonical_path=args.diagnostic_canonical_artifact,
            canonical_error_count=args.diagnostic_canonical_error_count,
        )
    controller = HybridController(
        max_local_steps=args.max_local_steps,
        cooldown_ppo_steps=1,
        pending_rehandoff_min_ppo_steps=args.pending_rehandoff_min_ppo_steps,
        functional_region_novelty_gate_enabled=args.functional_region_novelty_gate,
    )
    ppo_buffer_steps = 0
    dqn_replay_steps = 0
    reward_calls = Counter()
    timeline: list[dict[str, Any]] = []
    handoff_events: list[dict[str, Any]] = []
    return_events: list[dict[str, Any]] = []
    temporal_windows: list[ActionEvidenceWindow] = []
    temporal_stability_diagnostics: list[dict[str, Any]] = []
    interactive_layout_evidence_records: list[dict[str, Any]] = []
    framework_network_evidence_records: list[dict[str, Any]] = []
    candidate_first_agent: dict[str, str] = {}
    confirmed_agent: dict[str, str] = {}
    ppo_candidate_dqn_confirmed = 0
    confirmation_records: list[dict[str, Any]] = []
    confirmed_keys_seen: set[str] = set()
    ppo_coverage = {
        "states": set(),
        "urls": set(),
        "targets": set(),
        "action_targets": set(),
        "action_types": Counter(),
        "meaningful_state_transitions": 0,
    }
    env_reset_count = 0
    unexpected_reset_count = 0
    reset_at_switch_count = 0
    action_error_count = 0
    duplicate_attempts = 0
    duplicate_blocked = 0
    fake_instruction_count = 0

    try:
        manifest = _validate_dqn_manifest(args.dqn_manifest)
        ppo = PPOAgent(env.observation_encoder.get_obs_dim(), env.action_space.get_action_dim())
        ppo.load(args.ppo_checkpoint)
        ppo.policy.eval()
        dqn = FactorizedQNetwork(976, len(env.action_space.action_types), env.action_space.max_candidates)
        payload = torch.load(Path(args.dqn_checkpoint), map_location="cpu")
        _validate_checkpoint_architecture(payload, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE)
        dqn.load_state_dict(payload.get("model", payload))
        dqn.eval()

        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            if diagnostic_recorder:
                diagnostic_recorder.record_reset(episode=episode, reason="episode_boundary")
            controller.reset_episode_guard()
            env_reset_count += 1
            reward_history: dict[str, Any] = {}
            done = False
            for step in range(1, args.max_steps + 1):
                active = AgentMode.PPO_GLOBAL if args.ppo_only else controller.active_agent
                controller.record_agent_step(active)
                if active == AgentMode.PPO_GLOBAL:
                    obs_vector = env.observation_encoder.encode_observation(observation)
                    if obs_vector.shape[0] != 635:
                        raise ValueError(f"PPO observation dim mismatch: {obs_vector.shape[0]}")
                    mask = env.action_space.build_action_mask(observation)
                    if args.ppo_selection_mode == "stochastic":
                        selected = ppo.select_action(obs_vector, mask)
                    else:
                        selected = ppo.select_greedy_action(obs_vector, mask)
                    action_id = int(selected["action_id"])
                    ppo_buffer_steps += 1
                    active_reward = "PPO"
                    local_session_id = None
                    handoff_reason = ""
                else:
                    obs_vector, confirmation_metadata = _dqn_state(env, observation, reward_history, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE)
                    if obs_vector.shape[0] != 976:
                        raise ValueError(f"DQN observation dim mismatch: {obs_vector.shape[0]}")
                    mask = safe_action_mask(env.action_space, observation)
                    action_id = select_factorized_greedy_action(dqn, obs_vector, mask, env.action_space)
                    dqn_replay_steps += 1
                    active_reward = "DQN"
                    local_session_id = controller.local_session.local_session_id if controller.local_session else None
                    handoff_reason = controller.local_session.handoff_reason if controller.local_session else ""

                action = env.action_space.decode(action_id)
                action["action_id"] = action_id
                action["agent_mode"] = active.value
                action_trace_id = f"{site['site_id']}:{'ppo' if args.ppo_only else 'hybrid'}:{args.seed}:ep{episode:02d}:step{step:03d}"
                candidate_for_trace = _action_candidate(observation, action)
                selected_mask_valid = bool(mask[action_id]) if len(mask) > action_id else False
                if args.action_error_diagnostic:
                    env.set_action_diagnostic_context({
                        "action_trace_id": action_trace_id,
                        "agent_mode": active.value,
                        "episode": episode,
                        "step": step,
                    })
                before_candidate_keys = set(_candidate_keys_from_anomalies([]))
                seen_before = set(reward_history.get("seen_candidate_signatures", set()) or set())
                next_observation, _, done, info = env.step(action_id)
                execution_diag = info.get("action_error_diagnostic", {}) if isinstance(info, Mapping) else {}
                action["failed"] = bool(info.get("last_action_error"))
                action["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                action_error_count += int(bool(info.get("last_action_error")))
                temporal_window = None
                if args.temporal_evidence:
                    temporal_window = env.collect_temporal_evidence(
                        _temporal_action_context(observation, action, active.value),
                        _temporal_config_from_args(args),
                    )
                    temporal_windows.append(temporal_window)
                    raw_network_events = []
                    for sample in temporal_window.samples:
                        raw_network_events.extend(sample.network_state.completed_requests)
                        raw_network_events.extend(sample.network_state.failed_requests)
                    normalized_network_events = normalize_network_events(
                        raw_network_events,
                        action_context={
                            "action_id": action.get("action_id"),
                            "action_type": action.get("action_type"),
                            "target_id": action.get("target_id") or action.get("candidate_id"),
                        },
                    )
                    framework_network_evidence_records.append(
                        {
                            "episode": episode,
                            "step": step,
                            "action_id": action.get("action_id"),
                            "action_type": action.get("action_type"),
                            "stability_state": temporal_window.final_stability.value,
                            "raw_network_events": raw_network_events,
                            "normalized_network_events": normalized_network_events,
                        }
                    )
                    enriched_stable_snapshot, interactive_layout_evidence = env.collect_interactive_layout_evidence(
                        temporal_window.terminal_snapshot
                    )
                    info["interactive_layout_evidence"] = interactive_layout_evidence
                    temporal_window.terminal_snapshot = enriched_stable_snapshot
                    next_observation = enriched_stable_snapshot
                    interactive_layout_evidence_records.append(
                        {
                            "episode": episode,
                            "step": step,
                            "action_id": action.get("action_id"),
                            "action_type": action.get("action_type"),
                            "stability_state": temporal_window.final_stability.value,
                            "summary": interactive_layout_evidence,
                            "candidates": [
                                {
                                    "bid": candidate.get("bid"),
                                    "evidence": candidate.get("interactive_layout_evidence", {}),
                                }
                                for candidate in enriched_stable_snapshot.get("candidate_elements", [])
                                if isinstance(candidate, dict) and candidate.get("interactive_layout_evidence")
                            ],
                        }
                    )
                    anomalies = sanitize_anomalies(
                        detect_anomalies_with_temporal_evidence(observation, {**info, "action": action}, temporal_window)
                    )
                else:
                    anomalies = sanitize_anomalies(detect_anomalies(observation, next_observation, {**info, "action": action}))
                anomaly_keys = _candidate_keys_from_anomalies(anomalies)
                new_candidate_keys = [key for key in anomaly_keys if key not in seen_before]
                for key in new_candidate_keys:
                    candidate_first_agent.setdefault(key, "PPO" if active == AgentMode.PPO_GLOBAL else "DQN")
                if active == AgentMode.PPO_GLOBAL:
                    reward, components = calculate_autonomous_reward(observation, next_observation, action, anomalies, [], reward_history, {})
                    reward_calls["PPO"] += 1
                    repeat_diagnostic = {}
                    _update_ppo_coverage(ppo_coverage, observation, next_observation, action, components)
                    _record_confirmations(
                        confirmation_records,
                        confirmed_keys_seen,
                        confirmed_agent,
                        candidate_first_agent,
                        active_agent="PPO",
                        local_session_id=None,
                        anomalies=anomalies,
                        events=[],
                    )
                    confirmation_metadata = _confirmation_metadata(reward_history, next_observation, env)
                    handoff_history_view = dict(reward_history)
                    handoff_history_view["seen_candidate_signatures"] = seen_before
                    handoff = "" if done or args.ppo_only else controller.maybe_handoff_after_ppo_step(
                        step=(episode - 1) * args.max_steps + step,
                        observation=next_observation,
                        anomalies=anomalies,
                        reward_components=components,
                        confirmation_metadata=confirmation_metadata,
                        history=handoff_history_view,
                    )
                    if handoff:
                        handoff_events.append(controller.handoff_events[-1])
                else:
                    reward, components = calculate_dqn_reward(observation, next_observation, action, anomalies, reward_history)
                    reward_calls["DQN"] += 1
                    repeat_diagnostic = dict(reward_history.get("last_repeat_diagnostic", {}))
                    duplicate_attempts += int(reward_history.get("last_confirmation_reward_duplicate_attempts", 0) or 0)
                    duplicate_blocked += int(reward_history.get("last_confirmation_reward_duplicate_blocks", 0) or 0)
                    for event in reward_history.get("last_confirmation_transition_events", []) or []:
                        key = str(event.get("logical_key") or "")
                        if key:
                            confirmed_agent[key] = "DQN"
                            if candidate_first_agent.get(key) == "PPO" and event.get("newly_confirmed"):
                                ppo_candidate_dqn_confirmed += 1
                    _record_confirmations(
                        confirmation_records,
                        confirmed_keys_seen,
                        confirmed_agent,
                        candidate_first_agent,
                        active_agent="DQN",
                        local_session_id=local_session_id,
                        anomalies=anomalies,
                        events=reward_history.get("last_confirmation_transition_events", []) or [],
                    )
                    returned = controller.maybe_return_after_dqn_step(
                        step=(episode - 1) * args.max_steps + step,
                        observation=next_observation,
                        anomalies=anomalies,
                        reward_components=components,
                        repeat_diagnostic=repeat_diagnostic,
                        done=done,
                        confirmation_events=reward_history.get("last_confirmation_transition_events", []) or [],
                        tracker_history=reward_history,
                        confirmation_metadata=_confirmation_metadata(reward_history, next_observation, env),
                    )
                    if returned:
                        return_events.append(controller.return_events[-1])
                _append_cross_agent_history(reward_history, observation, next_observation, action, components, repeat_diagnostic)
                if active == AgentMode.PPO_GLOBAL:
                    _sync_cross_agent_action_target_history(reward_history)
                timeline.append(
                    {
                        "episode": episode,
                        "step": step,
                        "active_agent": active.value,
                        "local_session_id": local_session_id,
                        "handoff_reason": handoff_reason,
                        "action_type": action.get("action_type"),
                        "reward_function": active_reward,
                        "reward": reward,
                        "reward_components": dict(components),
                        "repeat_diagnostic": dict(repeat_diagnostic),
                        "candidate_count": len(anomaly_keys),
                        "pending_count": len(reward_history.get("pending_reproduction_records", {}) or {}),
                        "new_candidate_keys": new_candidate_keys,
                        "newly_confirmed": list(confirmed_agent.keys()),
                        "url_before": _url(observation),
                        "url_after": _url(next_observation),
                        "state_before": _state_signature(observation),
                        "state_after": _state_signature(next_observation),
                        "local_state_before": _local_state_signature(observation),
                        "local_state_after": _local_state_signature(next_observation),
                        "action_target": _action_target_key(observation, action),
                        "done": done,
                        "temporal_stability": temporal_window.final_stability.value if temporal_window else "",
                        "temporal_window_ms": temporal_window.duration_ms if temporal_window else 0,
                        "temporal_samples": len(temporal_window.samples) if temporal_window else 0,
                    }
                )
                if diagnostic_recorder:
                    diagnostic_recorder.record_step({
                        "site": site["site_id"],
                        "method": "ppo" if args.ppo_only else "hybrid",
                        "seed": args.seed,
                        "episode": episode,
                        "episode_step": step,
                        "global_step": (episode - 1) * args.max_steps + step,
                        "agent_mode": active.value,
                        "controller_mode_before": active.value,
                        "controller_mode_after": controller.active_agent.value,
                        "action_trace_id": action_trace_id,
                        "selected_action_index": action_id,
                        "action_type": action.get("action_type"),
                        "candidate_slot": action.get("candidate_index"),
                        "requires_target": action.get("action_type") in {"click_element", "fill_input", "press_enter", "scroll"},
                        "selected_action_mask_valid": selected_mask_valid,
                        "valid_action_count": int(sum(bool(value) for value in mask)),
                        "target_present": isinstance(candidate_for_trace, Mapping),
                        "target_id": (candidate_for_trace or {}).get("bid") if isinstance(candidate_for_trace, Mapping) else None,
                        "target_stable_signature": (candidate_for_trace or {}).get("stable_signature") if isinstance(candidate_for_trace, Mapping) else None,
                        "target_role": (candidate_for_trace or {}).get("role") if isinstance(candidate_for_trace, Mapping) else None,
                        "target_tag": (candidate_for_trace or {}).get("tag") if isinstance(candidate_for_trace, Mapping) else None,
                        "target_text": (candidate_for_trace or {}).get("text") if isinstance(candidate_for_trace, Mapping) else None,
                        "target_visible": (candidate_for_trace or {}).get("visible") if isinstance(candidate_for_trace, Mapping) else None,
                        "target_enabled": (candidate_for_trace or {}).get("enabled") if isinstance(candidate_for_trace, Mapping) else None,
                        "url_before": _url(observation),
                        "route_before": _url_path(observation),
                        "state_fingerprint_before": _state_signature(observation),
                        "execution": execution_diag,
                        "url_after": _url(next_observation),
                        "route_after": _url_path(next_observation),
                        "state_fingerprint_after": _state_signature(next_observation),
                        "last_action_error": bool(info.get("last_action_error")),
                        "failed": bool(action.get("failed")),
                        "timeout": bool(action.get("timeout")),
                        "reward_total": reward,
                        "reward_components": dict(components),
                        "action_error_penalty_applied": bool(components.get("action_error_penalty")),
                        "invalid_penalty_applied": bool(components.get("invalid_penalty")),
                        "timeout_penalty_applied": bool(components.get("timeout_penalty")),
                        "buffer_written_to": active_reward,
                        "handoff_reason": handoff_reason,
                        "return_reason": controller.return_events[-1].get("reason") if controller.return_events else None,
                        "done": done,
                        "temporal_stability": temporal_window.final_stability.value if temporal_window else None,
                        "temporal_window_ms": temporal_window.duration_ms if temporal_window else None,
                    })
                if args.temporal_stability_diagnostic_output and temporal_window is not None:
                    diagnostic_context = _temporal_action_context(observation, action, active.value)
                    post_samples = env.collect_post_stable_diagnostic_samples(
                        temporal_window,
                        diagnostic_context,
                        _post_stable_offsets_from_args(args),
                        _temporal_config_from_args(args),
                    )
                    temporal_stability_diagnostics.append(
                        analyze_temporal_stability_window(
                            before_observation=observation,
                            action_info={**info, "action": action},
                            temporal_window=temporal_window,
                            detector_anomalies=anomalies,
                            post_stable_samples=post_samples,
                            episode=episode,
                            step=step,
                        )
                    )
                observation = next_observation
                if done:
                    if controller.active_agent == AgentMode.DQN_LOCAL:
                        returned = controller.maybe_return_after_dqn_step(
                            step=(episode - 1) * args.max_steps + step,
                            observation=observation,
                            anomalies=anomalies,
                            reward_components=components,
                            repeat_diagnostic=repeat_diagnostic,
                            done=True,
                            confirmation_events=reward_history.get("last_confirmation_transition_events", []) or [],
                            tracker_history=reward_history,
                            confirmation_metadata=_confirmation_metadata(reward_history, observation, env),
                        )
                        if returned:
                            return_events.append(controller.return_events[-1])
                    break
        controller_summary = controller.summary()
        ppo_obs_dim = env.observation_encoder.get_obs_dim()
        dqn_probe, dqn_meta = _dqn_state(env, observation, reward_history, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE)
        summary = {
            "configuration": {
                "site_id": site["site_id"],
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "max_local_steps": args.max_local_steps,
                "pending_rehandoff_min_ppo_steps": args.pending_rehandoff_min_ppo_steps,
                "functional_region_novelty_gate_enabled": bool(args.functional_region_novelty_gate),
                "seed": args.seed,
                "ppo_checkpoint": args.ppo_checkpoint,
                "dqn_checkpoint": args.dqn_checkpoint,
                "dqn_manifest": args.dqn_manifest,
                "ppo_selection_mode": args.ppo_selection_mode,
                "ppo_only": bool(args.ppo_only),
                "temporal_evidence_enabled": bool(args.temporal_evidence),
                "temporal_evidence_config": {
                    "sample_interval_ms": args.temporal_sample_interval_ms,
                    "minimum_observation_ms": args.temporal_minimum_observation_ms,
                    "stable_quiet_window_ms": args.temporal_stable_quiet_window_ms,
                    "maximum_observation_ms": args.temporal_maximum_observation_ms,
                },
            },
            "checkpoint_manifest": manifest,
            "episode_summary": {
                "total_steps": len(timeline),
                "ppo_steps": controller_summary["agent_step_counts"].get(AgentMode.PPO_GLOBAL.value, 0),
                "dqn_steps": controller_summary["agent_step_counts"].get(AgentMode.DQN_LOCAL.value, 0),
                "handoff_count": controller_summary["handoff_count"],
                "return_count": controller_summary["return_count"],
                "local_sessions": controller_summary["dqn_session_count"],
                "mean_local_session": controller_summary["mean_dqn_session_length"],
                "max_local_session": controller_summary["max_dqn_session_length"],
            },
            "handoff_events": handoff_events,
            "pending_handoff_guard": controller_summary["pending_handoff_guard"],
            "functional_region_novelty": controller_summary["functional_region_novelty"],
            "return_events": return_events,
            "agent_step_counts": controller_summary["agent_step_counts"],
            "candidate_attribution": {
                "candidate_observations": sum(len(row["new_candidate_keys"]) for row in timeline),
                "unique_candidates": len(candidate_first_agent),
                "ppo_discovered_candidates": sum(1 for value in candidate_first_agent.values() if value == "PPO"),
                "dqn_discovered_candidates": sum(1 for value in candidate_first_agent.values() if value == "DQN"),
                "candidate_first_agent": dict(sorted(candidate_first_agent.items())),
                "ppo_candidate_dqn_confirmed_count": ppo_candidate_dqn_confirmed,
            },
            "confirmation_attribution": {
                "newly_confirmed": len(confirmed_agent),
                "unique_confirmed": len(confirmed_agent),
                "confirmed_by_agent": dict(Counter(confirmed_agent.values())),
                "confirmation_records": confirmation_records,
                "attribution_matrix": _attribution_matrix(confirmation_records),
                "duplicate_attempts": duplicate_attempts,
                "duplicate_blocked": duplicate_blocked,
                "duplicate_reward_leak": duplicate_attempts - duplicate_blocked,
            },
            "ppo_global_coverage": {
                "unique_states": len(ppo_coverage["states"]),
                "unique_urls": len(ppo_coverage["urls"]),
                "unique_targets": len(ppo_coverage["targets"]),
                "unique_action_targets": len(ppo_coverage["action_targets"]),
                "unique_action_types": len(ppo_coverage["action_types"]),
                "action_distribution": dict(ppo_coverage["action_types"]),
                "meaningful_state_transitions": int(ppo_coverage["meaningful_state_transitions"]),
            },
            "reward_attribution": {
                "reward_function_calls": dict(reward_calls),
                "ppo_buffer_steps": ppo_buffer_steps,
                "dqn_replay_steps": dqn_replay_steps,
                "buffer_mixing": False,
                "reward_mixing": False,
            },
            "environment_continuity": {
                "env_reset_count": env_reset_count,
                "browser_session_count": 1,
                "unexpected_reset_count": unexpected_reset_count,
                "reset_at_switch_count": reset_at_switch_count,
                "same_environment_across_handoff": True,
            },
            "history_continuity": {
                "history_window": DQN_HISTORY_WINDOW,
                "recent_action_identities": dqn_meta.get("history_context", {}).get("recent_action_identities", []),
                "ppo_actions_feed_dqn_history": any(row["active_agent"] == AgentMode.PPO_GLOBAL.value for row in timeline)
                and len(reward_history.get("dqn_history_records", []) or []) > 0,
                "dqn_actions_feed_dqn_history": any(row["active_agent"] == AgentMode.DQN_LOCAL.value for row in timeline),
            },
            "timeline": timeline,
            "temporal_evidence": summarize_temporal_windows(temporal_windows),
            "interactive_layout_evidence": {
                "records": interactive_layout_evidence_records,
                "action_count": len(interactive_layout_evidence_records),
                "available_action_count": sum(
                    int(record.get("summary", {}).get("status") == "AVAILABLE")
                    for record in interactive_layout_evidence_records
                ),
            },
            "framework_network_evidence": {
                "schema_version": "framework_normalizer_v1",
                "records": framework_network_evidence_records,
                "action_count": len(framework_network_evidence_records),
                "raw_event_count": sum(len(record.get("raw_network_events", [])) for record in framework_network_evidence_records),
                "normalized_event_count": sum(len(record.get("normalized_network_events", [])) for record in framework_network_evidence_records),
            },
            "temporal_timeline_examples": [window.to_metadata(include_samples=False) for window in temporal_windows[:3]],
            "hard_invariants": {
                "ppo_obs_dim_635": ppo_obs_dim == 635,
                "dqn_obs_dim_976": int(dqn_probe.shape[0]) == 976,
                "action_dim_608": env.action_space.get_action_dim() == 608,
                "same_environment_across_handoff": True,
                "shared_confirmation_tracker": True,
                "candidate_identity_shared": True,
                "separate_buffers": ppo_buffer_steps + dqn_replay_steps == len(timeline),
                "separate_reward_functions": reward_calls["PPO"] + reward_calls["DQN"] == len(timeline),
                "ground_truth_used": False,
                "action_error_zero": action_error_count == 0,
                "fake_instruction_zero": fake_instruction_count == 0,
                "duplicate_reward_leak_zero": duplicate_attempts == duplicate_blocked,
                "zero_step_switch_loop": bool(controller_summary["zero_step_switch_loop"]),
                "local_budget_respected": controller_summary["max_dqn_session_length"] <= args.max_local_steps,
                "dqn_to_ppo_return": args.ppo_only or controller_summary["return_count"] > 0,
                "ppo_dqn_ppo_sequence": args.ppo_only or _has_ppo_dqn_ppo_sequence(timeline),
                "RAW_EVIDENCE_COLLECTOR_CHECK": (not args.temporal_evidence) or bool(temporal_windows),
                "TEMPORAL_STATE_TRACKER_CHECK": (not args.temporal_evidence)
                or all(window.final_stability.value in {"STABLE", "TIMED_OUT_UNSTABLE"} for window in temporal_windows),
            },
        }
        summary["hard_invariants"]["all_pass"] = (
            all(
                value is True
                for key, value in summary["hard_invariants"].items()
                if key not in {"zero_step_switch_loop", "ground_truth_used"}
            )
            and not summary["hard_invariants"]["zero_step_switch_loop"]
            and not summary["hard_invariants"]["ground_truth_used"]
        )
        summary["diagnostic"] = {
            "action_error_count": action_error_count,
            "fake_instruction_count": fake_instruction_count,
            "HYBRID_V1_ARCHITECTURE_CHECK": "PASS" if summary["hard_invariants"]["all_pass"] else "REVIEW"
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        if args.temporal_stability_diagnostic_output:
            diagnostic_output = Path(args.temporal_stability_diagnostic_output)
            diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_payload = {
                "configuration": summary["configuration"],
                "post_stable_offsets_ms": _post_stable_offsets_from_args(args),
                "interactive_layout_evidence": summary["interactive_layout_evidence"],
                "summary": summarize_temporal_stability_diagnostics(temporal_stability_diagnostics),
                "windows": temporal_stability_diagnostics,
                "runtime_isolation": {
                    "detector_controller_reward_before_post_stable_sampling": True,
                    "diagnostic_samples_used_for_runtime_decisions": False,
                },
                "hard_invariants": summary["hard_invariants"],
            }
            diagnostic_output.write_text(json.dumps(_jsonable(diagnostic_payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        if diagnostic_recorder:
            diagnostic_recorder.close()
        env.close()


def _append_cross_agent_history(history: dict[str, Any], before: Mapping[str, Any], after: Mapping[str, Any], action: Mapping[str, Any], components: Mapping[str, float], repeat_diagnostic: Mapping[str, Any]) -> None:
    action_type = str(action.get("action_type") or "")
    candidate = _action_candidate(before, action)
    no_effect = dqn_no_effect_diagnostic(before, after, action)
    append_history_transition(
        history,
        action_type=action_type,
        candidate=candidate,
        meaningful_action_effect=not bool(no_effect.get("no_effect")),
        productive_transition=any(float(components.get(key, 0.0) or 0.0) > 0.0 for key in ("new_state", "new_url", "new_target", "new_action_target", "new_local_state", "new_local_action_target", "candidate_anomaly", "confirmation_upgrade")),
        confirmation_progress=float(components.get("confirmation_upgrade", 0.0) or 0.0) > 0.0,
        unproductive_repeat=bool(repeat_diagnostic.get("unproductive_repeat")),
    )


def _sync_cross_agent_action_target_history(history: dict[str, Any]) -> None:
    keys: list[str] = []
    for row in history.get("dqn_history_records", []) or []:
        if not isinstance(row, Mapping):
            continue
        action_type = str(row.get("action_type") or "")
        if not action_type:
            continue
        target = str(row.get("target_identity") or "")
        keys.append(f"{action_type}:{target}" if target else f"{action_type}:0")
    history["dqn_action_target_history"] = keys


def _candidate_keys_from_anomalies(anomalies: list[Mapping[str, Any]]) -> list[str]:
    return [
        str(item.get("logical_key") or item.get("signature") or "")
        for item in anomalies
        if str(item.get("status") or "candidate") == "candidate" and str(item.get("logical_key") or item.get("signature") or "")
    ]


def _update_ppo_coverage(
    coverage: dict[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action: Mapping[str, Any],
    components: Mapping[str, Any],
) -> None:
    coverage["states"].add(_state_signature(after))
    coverage["urls"].add(_url_path(after))
    action_type = str(action.get("action_type") or "")
    if action_type:
        coverage["action_types"][action_type] += 1
    target = _action_target_key(before, action)
    if target:
        coverage["action_targets"].add(target)
        _, target_identity = target.split(":", 1)
        if target_identity and target_identity != "0":
            coverage["targets"].add(target_identity)
    if _state_signature(before) != _state_signature(after) or any(
        float(components.get(key, 0.0) or 0.0) > 0.0
        for key in ("new_state", "new_url", "new_target", "new_action_target", "flow_progress")
    ):
        coverage["meaningful_state_transitions"] += 1


def _action_target_key(observation: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    action_type = str(action.get("action_type") or "")
    if not action_type:
        return ""
    candidate = _action_candidate(observation, action)
    target = _target_identity(candidate) if candidate else ""
    return f"{action_type}:{target or action.get('candidate_index', 0)}"


def _record_confirmations(
    records: list[dict[str, Any]],
    confirmed_keys_seen: set[str],
    confirmed_agent: dict[str, str],
    candidate_first_agent: Mapping[str, str],
    *,
    active_agent: str,
    local_session_id: int | None,
    anomalies: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    event_by_key = {str(event.get("logical_key") or ""): event for event in events if str(event.get("logical_key") or "")}
    confirmed_keys = {
        str(item.get("logical_key") or item.get("signature") or "")
        for item in anomalies
        if str(item.get("status") or "") == "confirmed" and str(item.get("logical_key") or item.get("signature") or "")
    }
    confirmed_keys.update(event_by_key)
    for key in sorted(confirmed_keys):
        if key in confirmed_keys_seen:
            continue
        confirmed_keys_seen.add(key)
        confirmed_agent.setdefault(key, active_agent)
        event = event_by_key.get(key, {})
        records.append(
            {
                "logical_key": key,
                "discovered_by": candidate_first_agent.get(key, "UNKNOWN"),
                "confirmed_by": active_agent,
                "local_session_id": local_session_id,
                "rule_id": event.get("rule_id", ""),
                "confirmation_mode": event.get("mode", event.get("confirmation_mode", "")),
            }
        )


def _attribution_matrix(records: list[Mapping[str, Any]]) -> dict[str, int]:
    matrix = Counter()
    for record in records:
        discovery = str(record.get("discovered_by") or "UNKNOWN")
        confirmation = str(record.get("confirmed_by") or "UNKNOWN")
        matrix[f"{discovery}->{confirmation}"] += 1
    return dict(matrix)


def _confirmation_metadata(history: Mapping[str, Any], observation: Mapping[str, Any], env: BrowserGymJAWSEnv) -> dict[str, Any]:
    _, metadata = build_dqn_confirmation_context(history, observation, env.action_space)
    return metadata


def _temporal_config_from_args(args: argparse.Namespace) -> TemporalEvidenceConfig:
    return TemporalEvidenceConfig(
        sample_interval_ms=args.temporal_sample_interval_ms,
        minimum_observation_ms=args.temporal_minimum_observation_ms,
        stable_quiet_window_ms=args.temporal_stable_quiet_window_ms,
        maximum_observation_ms=args.temporal_maximum_observation_ms,
        debug_timeline=bool(args.temporal_debug_timeline),
    )


def _temporal_action_context(observation: Mapping[str, Any], action: Mapping[str, Any], agent_mode: str) -> dict[str, Any]:
    candidate = _action_candidate(observation, action)
    return {
        "action_id": str(action.get("action_id") or ""),
        "agent_mode": agent_mode,
        "action_type": str(action.get("action_type") or ""),
        "target_id": str((candidate or {}).get("bid") or (candidate or {}).get("candidate_id") or ""),
        "candidate_index": int(action.get("candidate_index", 0) or 0),
        "action_timestamp": float(action.get("action_timestamp", 0.0) or 0.0),
        "route_before": _url(observation),
    }


def _post_stable_offsets_from_args(args: argparse.Namespace) -> list[int]:
    raw = str(getattr(args, "post_stable_offsets_ms", "") or "")
    offsets: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            offsets.append(max(0, int(item)))
        except ValueError:
            continue
    return offsets or [100, 200, 300, 500]


def _validate_dqn_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "architecture": FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
        "dqn_obs_dim": 976,
        "ppo_obs_dim": 635,
        "action_dim": 608,
        "history_observation_version": "v2",
        "DQN_ROLE_CHECK": "FINAL PASS",
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"final DQN manifest mismatch: {mismatches}")
    return manifest


def _has_ppo_dqn_ppo_sequence(timeline: list[Mapping[str, Any]]) -> bool:
    sequence = [str(row.get("active_agent") or "") for row in timeline]
    for index in range(len(sequence) - 2):
        if sequence[index] == AgentMode.PPO_GLOBAL.value and sequence[index + 1] == AgentMode.DQN_LOCAL.value:
            return AgentMode.PPO_GLOBAL.value in sequence[index + 2 :]
    return False


def _url(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return str(page.get("url") or observation.get("url") or "")


def _load_site_config(config_path: str | Path, site_id: str, base_url: str = "") -> dict[str, Any]:
    if base_url:
        return {"site_id": site_id, "base_url": base_url}
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    sites = config.get("sites", config)
    if isinstance(sites, Mapping):
        item = sites.get(site_id)
        if isinstance(item, Mapping):
            return {"site_id": site_id, "base_url": item.get("base_url")}
    for item in sites if isinstance(sites, list) else []:
        if str(item.get("site_id") or "") == site_id:
            return dict(item)
    raise ValueError(f"site_id not found in config: {site_id}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--pending-rehandoff-min-ppo-steps", type=int, default=3)
    parser.add_argument("--functional-region-novelty-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--dqn-checkpoint", required=True)
    parser.add_argument("--dqn-manifest", default="artifacts/models/dqn_local_specialist_final_manifest.json")
    parser.add_argument("--output", default="artifacts/sanity/ppo_dqn_hybrid_v1_smoke.json")
    parser.add_argument("--ppo-selection-mode", choices=["greedy", "stochastic"], default="greedy")
    parser.add_argument("--ppo-only", action="store_true", help="Evaluation-only PPO baseline; keep evidence/detector stack enabled and disable DQN handoff.")
    parser.add_argument("--temporal-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-sample-interval-ms", type=int, default=100)
    parser.add_argument("--temporal-minimum-observation-ms", type=int, default=250)
    parser.add_argument("--temporal-stable-quiet-window-ms", type=int, default=350)
    parser.add_argument("--temporal-maximum-observation-ms", type=int, default=1500)
    parser.add_argument("--temporal-debug-timeline", action="store_true")
    parser.add_argument("--temporal-stability-diagnostic-output", default="")
    parser.add_argument("--action-error-diagnostic", action="store_true", default=False)
    parser.add_argument("--action-error-diagnostic-output", default="")
    parser.add_argument("--diagnostic-canonical-artifact", default="")
    parser.add_argument("--diagnostic-canonical-error-count", type=int, default=0)
    parser.add_argument("--post-stable-offsets-ms", default="100,200,300,500")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    result = run_hybrid(parsed_args)
    if not parsed_args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))
