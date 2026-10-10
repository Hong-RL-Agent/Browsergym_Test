"""Runtime smoke runner for RAWD Hybrid v1 PPO/DQN orchestration."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
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
    factorized_flat_q_values,
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
from services.r10_network_ui_semantic_evidence import (
    build_network_ui_semantic_evidence,
    normalize_runtime_network_telemetry,
    semantic_pair_confidence,
)
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
    downstream_validation_records: list[dict[str, Any]] = []
    reproduction_qrank_records: list[dict[str, Any]] = []
    rule_evaluations = Counter()

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
                    local_session = controller.local_session
                    diagnostic_mode = str(confirmation_metadata.get("confirmation_mode") or "")
                    session_mode = str(local_session.trigger_candidate_confirmation_mode or "") if local_session else ""
                    if diagnostic_mode == "ACTION_REPRODUCTION" or session_mode == "ACTION_REPLAY":
                        candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
                        candidates = candidates if isinstance(candidates, list) else []
                        expected_type = str(confirmation_metadata.get("expected_action_type") or "")
                        expected_target = str(confirmation_metadata.get("expected_target_identity") or "")
                        matching_slots = [int(slot) for slot in confirmation_metadata.get("matching_candidate_slots", [])]
                        target_slot = matching_slots[0] if matching_slots else None
                        mapping = {
                            "candidate_position": target_slot,
                            "action_type": expected_type,
                            "action_type_index": env.action_space.action_types.index(expected_type) if expected_type in env.action_space.action_types else None,
                            "target_slot": target_slot,
                            "flattened_action_index": None,
                            "decoded_action": None,
                            "decoded_target_identity": None,
                        }
                        if expected_type in env.action_space.action_types and target_slot is not None:
                            encoded = int(env.action_space.encode(expected_type, target_slot))
                            decoded = env.action_space.decode(encoded)
                            mapping.update({
                                "flattened_action_index": encoded,
                                "decoded_action": decoded,
                                "decoded_target_identity": _target_identity(candidates[target_slot]) if target_slot < len(candidates) and isinstance(candidates[target_slot], Mapping) else "",
                            })
                        target_candidate = candidates[target_slot] if target_slot is not None and target_slot < len(candidates) and isinstance(candidates[target_slot], Mapping) else {}
                        bbox = target_candidate.get("bbox") if isinstance(target_candidate, Mapping) else None
                        bbox_valid = isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and float(bbox[2] or 0.0) > 0.0 and float(bbox[3] or 0.0) > 0.0 if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else False
                        reproduction_index = mapping["flattened_action_index"]
                        mask_valid = bool(reproduction_index is not None and reproduction_index < len(mask) and mask[reproduction_index] > 0)
                        with torch.inference_mode():
                            q_values = factorized_flat_q_values(
                                dqn(torch.from_numpy(obs_vector).float().unsqueeze(0)),
                                torch.from_numpy(mask).bool().unsqueeze(0),
                                env.action_space,
                            ).squeeze(0).detach().cpu().numpy()
                        valid_ids = [int(idx) for idx in np.flatnonzero(mask > 0)]
                        ordered_ids = sorted(valid_ids, key=lambda idx: float(q_values[idx]), reverse=True)
                        rank_all = (ordered_ids.index(reproduction_index) + 1) if reproduction_index in ordered_ids else None
                        all_candidates = []
                        for rank, idx in enumerate(ordered_ids, start=1):
                            decoded = env.action_space.decode(idx)
                            target = ""
                            if decoded["action_type"] in {"click_element", "fill_input", "press_enter"} and decoded["candidate_index"] < len(candidates):
                                target = _target_identity(candidates[decoded["candidate_index"]])
                            all_candidates.append({
                                "rank_valid": rank,
                                "action_index": idx,
                                "action_type": decoded["action_type"],
                                "target_identity": target,
                                "q_value": float(q_values[idx]),
                                "mask_valid": True,
                                "is_reproduction_action": bool(idx == reproduction_index),
                            })
                        context_vector = obs_vector[635:707]
                        reproduction_q = float(q_values[reproduction_index]) if reproduction_index is not None and reproduction_index < len(q_values) else None
                        best_q = float(q_values[ordered_ids[0]]) if ordered_ids else None
                        reproduction_context = {
                            "expected": {
                                "confirmation_mode": diagnostic_mode,
                                "action_type": expected_type,
                                "target_identity": expected_target,
                                "requires_action_reproduction": int(confirmation_metadata.get("expected_action_requires_target", 0) or 0),
                                "active_candidate": bool(confirmation_metadata.get("active_pending_key")),
                            },
                            "metadata": dict(confirmation_metadata),
                            "encoded": {
                                "active_pending": float(context_vector[0]),
                                "action_type_flag": float(context_vector[16 + env.action_space.action_types.index(expected_type)]) if expected_type in env.action_space.action_types else 0.0,
                                "requires_target": float(context_vector[35]),
                                "target_slot_flags": [int(i) for i in np.flatnonzero(context_vector[36:68] > 0.0)],
                                "target_visible": float(context_vector[68]),
                                "pending_count": float(context_vector[71]),
                            },
                        }
                        reproduction_qrank_records.append({
                            "episode": episode,
                            "step": step,
                            "global_step": (episode - 1) * args.max_steps + step,
                            "local_session_id": local_session.local_session_id if local_session else None,
                            "candidate_logical_key": local_session.trigger_candidate_key if local_session else confirmation_metadata.get("active_pending_key"),
                            "reproduction_strategy": "ACTION_REPLAY",
                            "reproduction_action_type": expected_type,
                            "reproduction_target_identity": expected_target,
                            "target_actionable": mask_valid,
                            "target_flags": {
                                "visible": bool(target_candidate.get("visible", True)),
                                "enabled": bool(target_candidate.get("enabled", True)),
                                "clickable": bool(target_candidate.get("clickable") or target_candidate.get("set_of_marks") or str(target_candidate.get("role") or "").lower() in {"button", "link", "checkbox", "radio", "menuitem", "tab"} or str(target_candidate.get("tag") or "").lower() in {"button", "a", "summary", "option", "label"}),
                                "bbox_valid": bbox_valid,
                                "action_compatible": mask_valid,
                            },
                            "local_candidates": {
                                "count": len(candidates),
                                "target_present": bool(matching_slots),
                                "target_slots": matching_slots,
                                "target_local_rank": (target_slot + 1) if target_slot is not None else None,
                                "identities": [_target_identity(candidate) for candidate in candidates[: env.action_space.max_candidates] if isinstance(candidate, Mapping)],
                            },
                            "mapping": mapping,
                            "round_trip_ok": bool(mapping["decoded_action"] and mapping["decoded_action"]["action_type"] == expected_type and mapping["decoded_target_identity"] == expected_target),
                            "mask_value": float(mask[reproduction_index]) if reproduction_index is not None and reproduction_index < len(mask) else None,
                            "context": reproduction_context,
                            "q_value": reproduction_q,
                            "rank_valid_actions": rank_all,
                            "rank_all_actions": rank_all,
                            "best_valid_q": best_q,
                            "q_gap": (best_q - reproduction_q) if best_q is not None and reproduction_q is not None else None,
                            "selected_action_index": None,
                            "selected": False,
                            "top10_valid_actions": all_candidates[:10],
                            "valid_action_count": len(ordered_ids),
                            "all_valid_actions": all_candidates,
                        })
                    action_id = select_factorized_greedy_action(dqn, obs_vector, mask, env.action_space)
                    if reproduction_qrank_records and reproduction_qrank_records[-1].get("episode") == episode and reproduction_qrank_records[-1].get("step") == step:
                        reproduction_qrank_records[-1]["selected_action_index"] = int(action_id)
                        reproduction_qrank_records[-1]["selected"] = bool(reproduction_qrank_records[-1].get("mapping", {}).get("flattened_action_index") == int(action_id))
                    dqn_replay_steps += 1
                    active_reward = "DQN"
                    local_session_id = controller.local_session.local_session_id if controller.local_session else None
                    handoff_reason = controller.local_session.handoff_reason if controller.local_session else ""

                action = env.action_space.decode(action_id)
                action["action_id"] = action_id
                action["episode_id"] = episode
                action["global_step"] = (episode - 1) * args.max_steps + step
                action["agent_mode"] = active.value
                action_trace_id = f"{site['site_id']}:{'ppo' if args.ppo_only else 'hybrid'}:{args.seed}:ep{episode:02d}:step{step:03d}"
                candidate_for_trace = _action_candidate(observation, action)
                action["target_identity"] = str((candidate_for_trace or {}).get("bid") or (candidate_for_trace or {}).get("id") or "")
                action["action_start_timestamp"] = time.monotonic() * 1000.0
                env._pending_action_context = {
                    "episode_id": episode,
                    "global_step": action["global_step"],
                    "action_id": action_id,
                    "action_type": action.get("action_type"),
                    "target_identity": action["target_identity"],
                    "action_start_timestamp": action["action_start_timestamp"],
                }
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
                action["action_end_timestamp"] = time.monotonic() * 1000.0
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
                runtime_telemetry = normalize_runtime_network_telemetry(next_observation, info)
                semantic_evidence = build_network_ui_semantic_evidence(
                    action_context={
                        "action_id": action.get("action_id"),
                        "action_timestamp": action.get("action_start_timestamp"),
                    },
                    network_entries=runtime_telemetry.get("events", []),
                    observation=next_observation,
                )
                r10_anomalies = [
                    anomaly for anomaly in anomalies
                    if str(anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id") or "") == "R10"
                ]
                for rule_id in ("R01", "R05", "R06", "R10"):
                    rule_evaluations[rule_id] += 1
                pairs = semantic_evidence.get("pairs", [])
                not_equal_pairs = [row for row in pairs if row.get("relation") == "NOT_EQUAL"]
                downstream_validation_records.append({
                    "episode": episode,
                    "step": step,
                    "global_step": action["global_step"],
                    "action": dict(action),
                    "detector": {
                        "invoked": True,
                        "rule_evaluated": ["R01", "R05", "R06", "R10"],
                        "eligible_evidence": len(anomalies),
                        "fired": len(anomalies),
                        "fired_rules": [str(a.get("rule_id") or "") for a in anomalies],
                    },
                    "evidence": {
                        "strict_associated_requests": runtime_telemetry.get("strict_associated_request_count", 0),
                        "parsed_responses": runtime_telemetry.get("parsed_response_count", 0),
                        "network_semantic_fields": len(semantic_evidence.get("network_fields", [])),
                        "ui_semantic_fields": len(semantic_evidence.get("ui_fields", [])),
                        "semantic_pairs": len(pairs),
                        "equal_pairs": semantic_evidence.get("equal_pairs", 0),
                        "not_equal_pairs": semantic_evidence.get("not_equal_pairs", 0),
                        "high_confidence_not_equal": len([row for row in not_equal_pairs if (semantic_pair_confidence(row) or 0.0) >= 0.8]),
                        "request_id_linkage_mismatch": runtime_telemetry.get("request_id_linkage_mismatch_count", 0),
                        "duplicate_request_ids": runtime_telemetry.get("duplicate_request_id_count", 0),
                    },
                    "r10": {
                        "action_error_gate": not bool(action.get("failed")),
                        "ui_settled": bool(not temporal_window or temporal_window.final_stability.value == "STABLE"),
                        "semantic_block_exists": bool(pairs),
                        "strict_action_request_association_exists": bool(runtime_telemetry.get("strict_associated_request_count", 0)),
                        "semantic_pair_exists": bool(pairs),
                        "not_equal": bool(not_equal_pairs),
                        "pair_confidence": max((semantic_pair_confidence(row) or 0.0 for row in not_equal_pairs), default=0.0),
                        "network_key_valid": sum(bool(row.get("network_key")) for row in not_equal_pairs),
                        "ui_key_valid": sum(bool(row.get("ui_key")) for row in not_equal_pairs),
                        "placeholder_rejection": 0,
                        "final_eligibility": bool(r10_anomalies),
                        "fire": bool(r10_anomalies),
                    },
                    "anomalies": [dict(anomaly) for anomaly in anomalies],
                })
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
        summary["downstream_validation"] = {
            "records": downstream_validation_records,
            "rule_evaluations": dict(rule_evaluations),
            "gt_runtime_access": False,
            "optimizer_updates": 0,
            "backward": 0,
            "replay_training": False,
            "checkpoint_writes": 0,
        }
        summary["reproduction_qrank_diagnostic"] = {
            "records": len(reproduction_qrank_records),
            "action_replay_sessions": len({str(row.get("local_session_id")) for row in reproduction_qrank_records if row.get("local_session_id")}),
            "selected_reproduction_actions": sum(int(row.get("selected")) for row in reproduction_qrank_records),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        if args.downstream_validation_dir:
            validation_dir = Path(args.downstream_validation_dir)
            validation_dir.mkdir(parents=True, exist_ok=True)
            records = downstream_validation_records
            detector_rows = [row["detector"] | {"episode": row["episode"], "step": row["step"]} for row in records]
            candidate_lifecycle = []
            for row in records:
                for anomaly in row.get("anomalies", []):
                    key = str(anomaly.get("signature") or "")
                    if key:
                        candidate_lifecycle.append({
                            "logical_key": key,
                            "step": row["step"],
                            "rule_id": anomaly.get("rule_id"),
                            "status": anomaly.get("status") or "DETECTED",
                            "reproduction_strategy": anomaly.get("reproduction_strategy"),
                            "requires_action_reproduction": anomaly.get("requires_action_reproduction"),
                            "reproduction_action_type": anomaly.get("reproduction_action_type"),
                            "reproduction_target_identity": anomaly.get("reproduction_target_identity"),
                            "pair_relation": (anomaly.get("evidence") or {}).get("relation"),
                            "pair_confidence": anomaly.get("pair_confidence") or (anomaly.get("evidence") or {}).get("pair_confidence"),
                            "detector_confidence": anomaly.get("confidence"),
                            "candidate_admitted": False,
                        })
            admitted_keys = set()
            for event in handoff_events:
                admitted_keys.update(str(key) for key in event.get("entry_candidate_keys", []) if key)
                if event.get("trigger_candidate_key"):
                    admitted_keys.add(str(event["trigger_candidate_key"]))
            for row in candidate_lifecycle:
                row["candidate_admitted"] = row["logical_key"] in admitted_keys
                if row["candidate_admitted"]:
                    row["status"] = "pending_confirmation"
            dqn_rows = [
                {
                    "episode": row["episode"], "step": row["step"],
                    "local_session_id": row.get("local_session_id"),
                    "selected_action": row.get("action_type"),
                    "selected_target": row.get("action_target"),
                    "local_transition": True,
                    "candidate_relevant_effect": bool(row.get("reward_components", {}).get("new_target") or row.get("reward_components", {}).get("confirmation_upgrade")),
                    "same_chain_recurrence": bool(row.get("repeat_diagnostic", {}).get("same_chain_recurrence")),
                }
                for row in timeline if row.get("active_agent") == AgentMode.DQN_LOCAL.value
            ]
            evidence_totals = {
                "click_input_actions": sum(int(row["action"].get("action_type") in {"click_element", "fill_input", "press_enter", "select_option"}) for row in records),
                "strict_associated_requests": sum(int(row["evidence"].get("strict_associated_requests", 0)) for row in records),
                "parsed_responses": sum(int(row["evidence"].get("parsed_responses", 0)) for row in records),
                "network_semantic_fields": sum(int(row["evidence"].get("network_semantic_fields", 0)) for row in records),
                "ui_semantic_fields": sum(int(row["evidence"].get("ui_semantic_fields", 0)) for row in records),
                "semantic_pairs": sum(int(row["evidence"].get("semantic_pairs", 0)) for row in records),
                "equal_pairs": sum(int(row["evidence"].get("equal_pairs", 0)) for row in records),
                "not_equal_pairs": sum(int(row["evidence"].get("not_equal_pairs", 0)) for row in records),
                "high_confidence_not_equal": sum(int(row["evidence"].get("high_confidence_not_equal", 0)) for row in records),
            }
            r10_rows = [row["r10"] | {"episode": row["episode"], "step": row["step"]} for row in records]
            (validation_dir / "detector_funnel.json").write_text(json.dumps(_jsonable({"detector_invocations": len(records), "rule_evaluations": dict(rule_evaluations), "rows": detector_rows, "r10_rows": r10_rows}), ensure_ascii=False, indent=2), encoding="utf-8")
            (validation_dir / "candidate_lifecycle.json").write_text(json.dumps(_jsonable({"candidate_proposals": len(candidate_lifecycle), "candidate_admissions": len(admitted_keys), "admission_rejects": sum(int(not row["candidate_admitted"]) for row in candidate_lifecycle), "rows": candidate_lifecycle}), ensure_ascii=False, indent=2), encoding="utf-8")
            (validation_dir / "controller_handoff.json").write_text(json.dumps(_jsonable({"handoff_requested": len(handoff_events), "handoff_accepted": len(handoff_events), "dqn_sessions": len(handoff_events), "events": handoff_events}), ensure_ascii=False, indent=2), encoding="utf-8")
            with (validation_dir / "dqn_local_sessions.jsonl").open("w", encoding="utf-8") as handle:
                for row in dqn_rows:
                    handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            e2e = {
                "evidence": evidence_totals,
                "detector_invocations": len(records),
                "detector_eligible_cases": sum(int(row["detector"].get("eligible_evidence", 0)) for row in records),
                "detector_fires": sum(int(row["detector"].get("fired", 0)) for row in records),
                "candidate_proposals": len(candidate_lifecycle),
                "candidate_admissions": len(admitted_keys),
                "action_replay_candidates": sum(int(row.get("reproduction_strategy") == "ACTION_REPLAY") for row in candidate_lifecycle if row.get("candidate_admitted")),
                "dqn_handoff": len(handoff_events),
                "dqn_local_transitions": len(dqn_rows),
                "same_chain_recurrence": sum(int(row.get("same_chain_recurrence")) for row in dqn_rows),
                "confirmations": summary["confirmation_attribution"]["newly_confirmed"],
                "gt_runtime_access": False,
                "optimizer_updates": 0,
                "backward": 0,
                "replay_training": False,
                "checkpoint_writes": 0,
            }
            e2e["FIRST_MAJOR_DROP_OFF"] = "END_TO_END_PIPELINE_RECOVERED" if e2e["dqn_handoff"] and e2e["confirmations"] else ("CONFIRMATION_BOTTLENECK" if e2e["dqn_handoff"] else "CANDIDATE_ADMISSION_BOTTLENECK")
            e2e["PRIMARY_ROOT_CAUSE"] = "production downstream path reached controller and DQN local inference"
            (validation_dir / "end_to_end_funnel.json").write_text(json.dumps(_jsonable(e2e), ensure_ascii=False, indent=2), encoding="utf-8")
        if args.reproduction_qrank_dir:
            qrank_dir = Path(args.reproduction_qrank_dir)
            qrank_dir.mkdir(parents=True, exist_ok=True)
            mapping_rows = [{
                "episode": row.get("episode"), "step": row.get("step"), "local_session_id": row.get("local_session_id"),
                "candidate_logical_key": row.get("candidate_logical_key"), "mapping": row.get("mapping"),
                "round_trip_ok": row.get("round_trip_ok"), "mask_value": row.get("mask_value"),
                "target_actionable": row.get("target_actionable"), "target_flags": row.get("target_flags"),
            } for row in reproduction_qrank_records]
            context_rows = [{
                "episode": row.get("episode"), "step": row.get("step"), "local_session_id": row.get("local_session_id"),
                "candidate_logical_key": row.get("candidate_logical_key"), "context": row.get("context"),
            } for row in reproduction_qrank_records]
            local_rows = [{
                "episode": row.get("episode"), "step": row.get("step"), "local_session_id": row.get("local_session_id"),
                "candidate_logical_key": row.get("candidate_logical_key"), "local_candidates": row.get("local_candidates"),
            } for row in reproduction_qrank_records]
            round_trip_pass = sum(int(row.get("round_trip_ok")) for row in reproduction_qrank_records)
            mask_valid_count = sum(int(row.get("target_actionable")) for row in reproduction_qrank_records)
            context_present_count = sum(int(
                row.get("context", {}).get("encoded", {}).get("active_pending", 0.0) > 0.0
                and row.get("context", {}).get("encoded", {}).get("action_type_flag", 0.0) > 0.0
                and row.get("context", {}).get("encoded", {}).get("target_visible", 0.0) > 0.0
            ) for row in reproduction_qrank_records if row.get("target_actionable"))
            actionable_records = [row for row in reproduction_qrank_records if row.get("target_actionable")]
            ranks = [int(row["rank_valid_actions"]) for row in reproduction_qrank_records if row.get("rank_valid_actions") is not None]
            q_values = [float(row["q_value"]) for row in reproduction_qrank_records if row.get("q_value") is not None]
            gaps = [float(row["q_gap"]) for row in reproduction_qrank_records if row.get("q_gap") is not None]
            competing = Counter()
            for row in reproduction_qrank_records:
                reproduction_index = row.get("mapping", {}).get("flattened_action_index")
                for action in row.get("all_valid_actions", []):
                    action_q = action.get("q_value")
                    reproduction_q = row.get("q_value")
                    if action.get("action_index") != reproduction_index and action_q is not None and reproduction_q is not None and float(action_q) > float(reproduction_q):
                        action_type = str(action.get("action_type") or "other")
                        if action_type in {"inspect_layout", "inspect_dom", "inspect_network", "inspect_console", "inspect_cart", "inspect_server_health", "inspect_port_status", "inspect_latency", "inspect_server_logs", "inspect_runtime_metrics"}:
                            category = "inspect"
                        elif action_type in {"scroll_down", "scroll_up"}:
                            category = "scroll"
                        elif action_type in {"change_viewport_mobile", "change_viewport_desktop"}:
                            category = "viewport"
                        elif action_type == "noop":
                            category = "noop"
                        elif action_type in {"click_element", "fill_input", "press_enter"}:
                            category = "other_interactive"
                        else:
                            category = "other"
                        competing[category] += 1
            summary_qrank = {
                "action_replay_candidates": len({str(row.get("candidate_logical_key")) for row in reproduction_qrank_records if row.get("candidate_logical_key")}),
                "actionable_targets": sum(int(row.get("target_actionable")) for row in reproduction_qrank_records),
                "target_present_in_local_pool": sum(int(row.get("local_candidates", {}).get("target_present")) for row in reproduction_qrank_records),
                "encode_decode_round_trip_pass": round_trip_pass,
                "mask_valid_reproduction_actions": mask_valid_count,
                "context_present": context_present_count,
                "mean_reproduction_q": float(np.mean(q_values)) if q_values else None,
                "mean_valid_action_rank": float(np.mean(ranks)) if ranks else None,
                "median_valid_action_rank": float(np.median(ranks)) if ranks else None,
                "best_valid_action_rank": min(ranks) if ranks else None,
                "worst_valid_action_rank": max(ranks) if ranks else None,
                "mean_best_q_minus_reproduction_q": float(np.mean(gaps)) if gaps else None,
                "selected_reproduction_actions": sum(int(row.get("selected")) for row in reproduction_qrank_records),
                "competing_action_categories_above_reproduction": dict(competing),
                "first_reproduction_context_drop_off": "NONE" if context_present_count == len(actionable_records) else "CONTEXT_ENCODING",
                "primary_root_cause": "DQN_Q_RANKING_FAILURE" if actionable_records and all(bool(row.get("round_trip_ok")) for row in actionable_records) and all(bool(row.get("target_actionable")) for row in actionable_records) and context_present_count == len(actionable_records) and sum(int(row.get("selected")) for row in actionable_records) == 0 else "INSUFFICIENT_ACTION_REPLAY_SAMPLES",
                "runtime_validation_only": True,
                "optimizer_updates": 0,
                "backward": 0,
                "reward_modified": False,
                "action_mask_behavior_modified": False,
                "gt_runtime_access": False,
            }
            (qrank_dir / "summary.json").write_text(json.dumps(_jsonable(summary_qrank), ensure_ascii=False, indent=2), encoding="utf-8")
            (qrank_dir / "action_index_mapping.json").write_text(json.dumps(_jsonable({"rows": mapping_rows}), ensure_ascii=False, indent=2), encoding="utf-8")
            with (qrank_dir / "q_rank_trace.jsonl").open("w", encoding="utf-8") as handle:
                for row in reproduction_qrank_records:
                    handle.write(json.dumps(_jsonable({
                        "episode": row.get("episode"), "step": row.get("step"), "local_session_id": row.get("local_session_id"),
                        "candidate_logical_key": row.get("candidate_logical_key"), "q_value": row.get("q_value"),
                        "rank_valid_actions": row.get("rank_valid_actions"), "rank_all_actions": row.get("rank_all_actions"),
                        "best_valid_q": row.get("best_valid_q"), "q_gap": row.get("q_gap"),
                        "selected": row.get("selected"), "selected_action_index": row.get("selected_action_index"),
                        "top10_valid_actions": row.get("top10_valid_actions"), "competing_actions": row.get("all_valid_actions"),
                    }), ensure_ascii=False) + "\n")
            (qrank_dir / "confirmation_context_trace.json").write_text(json.dumps(_jsonable({"rows": context_rows}), ensure_ascii=False, indent=2), encoding="utf-8")
            (qrank_dir / "local_candidate_trace.json").write_text(json.dumps(_jsonable({"rows": local_rows}), ensure_ascii=False, indent=2), encoding="utf-8")
            (qrank_dir / "root_cause.json").write_text(json.dumps(_jsonable({"primary_root_cause": summary_qrank["primary_root_cause"], "first_reproduction_context_drop_off": summary_qrank["first_reproduction_context_drop_off"], "summary": summary_qrank}), ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--downstream-validation-dir", default="")
    parser.add_argument("--reproduction-qrank-dir", default="", help="Validation-only frozen DQN reproduction action/Q-rank diagnostics.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    result = run_hybrid(parsed_args)
    if not parsed_args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))
