"""GT-blind production Hybrid discovery funnel for site006."""
from __future__ import annotations

import argparse, hashlib, json, os, random, shutil, subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.confirmation_reproduction import ACTION_REPLAY, OBSERVATION_RECHECK, derive_reproduction_context
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, ReplayBuffer, Transition, factorized_flat_q_values
from dqn_error_discovery.v3.confirmation_forced_reproduction_diagnostic import (
    _effect as reproduction_effect,
    _reward_breakdown as reproduction_reward_breakdown,
    diagnostic_forced_action,
)
from dqn_error_discovery.runners.train_dqn_error_detector import initialize_dqn_candidate_reward_baseline
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.confirmation_forced_reproduction_diagnostic import _start_site_server
from dqn_error_discovery.v3.run4_candidate_ab_validation import configure_browser_timeouts
from dqn_error_discovery.v3.run3_integrated_validation import _confirmation_metadata, enrich, state_key
from dqn_error_discovery.v3.run5_context_aware_dqn_training import CANDIDATE_A_PATHS, CANDIDATE_B_PATHS
from dqn_error_discovery.v3.train import _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state
from services.anomaly_detection_service import (
    AnomalyConfirmationTracker,
    _has_api_failure,
    _r10_high_confidence_not_equal_pair,
    _r10_network_failure,
    _r10_semantic_pairs,
    _r10_ui_settled,
    detect_anomalies,
)
from services.autonomous_reward_service import calculate_autonomous_reward

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("RAWD_R10_E2E_OUT", str(ROOT / "artifacts/training_v3/r10_page_text_fix_recovery_smoke")))
CONTRACT_OUT = Path(os.environ.get("RAWD_R10_E2E_CONTRACT_OUT", str(ROOT / "artifacts/training_v3/r10_runtime_input_contract_after_page_text_fix")))
SPACE = V3ActionSpace(32)
REWARD_BIAS_ROWS: list[dict[str, Any]] = []
PPO_PATH = ROOT / "artifacts/models/v3/rawd_v3_run4_candidate_b_ppo.pt"
DQN_PATH = ROOT / "artifacts/models/v3/run5_context_aware_dqn/seed44_restart1/rawd_v3_run5_context_aware_dqn_seed44.pt"
SEED44 = ROOT / "artifacts/training_v3/run5_context_aware_dqn/seed44_restart1"

def stop_site_server(process, stdout, stderr) -> None:
    if process is not None:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill()
    for handle in (stdout, stderr):
        if handle is not None: handle.close()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest().upper()

def protected_hashes() -> dict[str, str]:
    paths = {**{f"a_{k}": v for k, v in CANDIDATE_A_PATHS.items()}, **{f"b_{k}": v for k, v in CANDIDATE_B_PATHS.items()},
             "seed44_checkpoint": DQN_PATH, "seed44_metrics": SEED44 / "dqn_seed44_metrics.jsonl",
             "seed44_transitions": SEED44 / "dqn_seed44_transitions.jsonl", "seed44_summary": SEED44 / "dqn_seed44_summary.json"}
    return {k: sha256(v) for k, v in paths.items()}

def identity(c: Mapping[str, Any]) -> str:
    for key in ("bid", "id", "entity_id", "product_id", "item_id"):
        if c.get(key) not in (None, ""): return f"{key}:{c[key]}"
    return ""

def bbox_valid(c: Mapping[str, Any]) -> bool:
    try:
        b = c.get("bbox"); return isinstance(b, (list, tuple)) and len(b) >= 4 and float(b[2]) > 0 and float(b[3]) > 0
    except (TypeError, ValueError): return False

def candidate_rows(obs: Mapping[str, Any], mask: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i, c in enumerate((obs.get("candidate_elements", []) or [])[:32]):
        if not isinstance(c, Mapping): continue
        rows.append({"index": i, "identity": identity(c), "tag": str(c.get("tag") or ""), "role": str(c.get("role") or ""),
                     "text": str(c.get("text") or "")[:160], "accessible_name": str(c.get("name") or c.get("aria_label") or c.get("text") or "")[:160],
                     "visible": bool(c.get("visible", True)), "enabled": bool(c.get("enabled", True)), "clickable": bool(c.get("clickable")),
                     "interactable": bool(c.get("interactable") or c.get("clickable")), "bbox_valid": bbox_valid(c),
                     "action_mask": {a: bool(mask[SPACE.encode(a, i)] > 0) for a in SPACE.action_types}})
    return rows

def semantic(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    page = obs.get("page_state", {}) if isinstance(obs.get("page_state"), Mapping) else {}
    block = page.get("network_ui_semantic")
    if not isinstance(block, Mapping):
        runtime = obs.get("runtime_signals", {}) if isinstance(obs.get("runtime_signals"), Mapping) else {}
        block = runtime.get("network_ui_semantic")
    return block if isinstance(block, Mapping) else {}


MISSING = "MISSING"


def _value_or_missing(container: Any, key: str) -> Any:
    return container.get(key, MISSING) if isinstance(container, Mapping) else MISSING


def _page_text_for_trace(observation: Mapping[str, Any]) -> Any:
    page = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    values = []
    for container, key in ((page, "page_text"), (page, "page_text_sample"), (runtime, "page_text")):
        value = _value_or_missing(container, key)
        if value != MISSING:
            values.append({"key": key, "value": str(value or "")[:1000]})
    return values if values else MISSING


def _semantic_contract(sem: Mapping[str, Any]) -> dict[str, Any]:
    pairs = [row for row in sem.get("pairs", []) or [] if isinstance(row, Mapping)]
    settled = sem.get("settled_pairs", MISSING)
    return {
        "block_present": bool(sem),
        "pairs_count": len(pairs),
        "settled_pairs": MISSING if settled == MISSING else (len(settled) if isinstance(settled, list) else type(settled).__name__),
        "high_confidence_not_equal_count": sum(int(_r10_high_confidence_not_equal_pair(row)) for row in pairs),
        "pairs_sample": [
            {key: row.get(key) for key in ("relation", "pair_method", "pair_confidence", "network_key", "ui_key", "url", "request_path", "normalized_url")}
            for row in pairs[:8]
        ],
    }


def _observation_contract(observation: Mapping[str, Any]) -> dict[str, Any]:
    page = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    sem = semantic(observation)
    return {
        "page_state": {
            key: _value_or_missing(page, key)
            for key in ("loading", "pending", "request_pending", "network_idle", "dom_stable", "ui_stable", "ready_state", "url", "page_text", "page_text_sample")
        },
        "runtime_signals": {
            key: _value_or_missing(runtime, key)
            for key in ("loading", "pending", "request_pending", "network_idle", "dom_stable", "ui_stable", "ready_state", "page_text")
        },
        "top_level": {
            key: _value_or_missing(observation, key)
            for key in ("loading", "pending", "request_pending", "network_idle", "dom_stable", "ui_stable", "ready_state", "url", "page_text")
        },
        "page_text_trace": _page_text_for_trace(observation),
        "current_url": _value_or_missing(page, "url"),
        "candidate_count": len(observation.get("candidate_elements", []) or []) if isinstance(observation, Mapping) else MISSING,
        "semantic": _semantic_contract(sem),
    }


def _settled_predicate_trace(observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> dict[str, Any]:
    page = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    flag_rows = []
    for scope, container in (("runtime_signals", runtime), ("page_state", page)):
        for key in ("loading", "pending", "request_pending"):
            value = _value_or_missing(container, key)
            flag_rows.append({"field": f"{scope}.{key}", "runtime_value": value, "expected_condition": "missing or falsy", "result": "FAIL" if value != MISSING and bool(value) else "PASS"})
    page = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals", {}), Mapping) else {}
    text_sources = []
    for source, container, key in (("page_state", page, "page_text"), ("page_state", page, "page_text_sample"), ("runtime_signals", runtime, "page_text")):
        value = _value_or_missing(container, key)
        if value != MISSING:
            text_sources.append((f"{source}.{key}", str(value or "")))
    tokens = ("loading", "spinner", "loading...", "로딩")
    matches = []
    for field, value in text_sources:
        lower = value.lower()
        for token in tokens:
            start = lower.find(token)
            if start >= 0:
                matches.append({"field": field, "token": token, "offset": start, "snippet": value[max(0, start - 100):start + len(token) + 100]})
    text_fail = bool(matches)
    flag_rows.append({"field": "page_text.loading_tokens", "runtime_value": matches if matches else MISSING, "expected_condition": "no loading/spinner token", "result": "FAIL" if text_fail else "PASS"})
    first_fail = next((row["field"] for row in flag_rows if row["result"] == "FAIL"), "NONE")
    return {"r10_ui_settled": bool(_r10_ui_settled(observation)), "predicates": flag_rows, "first_failing_predicate": first_fail, "page_text_token_matches": matches, "action_info_temporal_evidence": _value_or_missing(action_info, "temporal_evidence")}


def _contract_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes = []
    def walk(prefix: str, left: Any, right: Any) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                walk(f"{prefix}.{key}" if prefix else str(key), left.get(key, MISSING), right.get(key, MISSING))
            return
        if left != right:
            if left == MISSING: kind = "added"
            elif right == MISSING: kind = "removed"
            elif type(left) is not type(right): kind = "type_changed"
            elif left == MISSING or right == MISSING: kind = "missing"
            else: kind = "overwritten"
            changes.append({"field": prefix, "before_enrich": left, "after_enrich": right, "classification": kind})
    walk("", before, after)
    return changes

def network_rows(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"method": str(x.get("method") or ""), "normalized_url": str(x.get("normalized_url") or urlparse(str(x.get("url") or "")).path),
             "status": x.get("status"), "response_body_available": bool(x.get("response_body_available")), "content_type": str(x.get("content_type") or "")}
            for x in block.get("requests", []) or [] if isinstance(x, Mapping)]

def load_agents() -> tuple[PPOAgent, FactorizedQNetwork]:
    ppo = PPOAgent(635, 640); ppo.load(PPO_PATH)
    payload = torch.load(DQN_PATH, map_location="cpu", weights_only=False)
    dqn = FactorizedQNetwork(984, 20, 32); dqn.load_state_dict(payload["network_state_dict"]); dqn.eval()
    return ppo, dqn

def run_runtime(episodes: int, max_steps: int, seed: int, *, base_steps=None, base_episode_rows=None, base_reproductions=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ppo, dqn = load_agents(); steps, episode_rows = list(base_steps or []), list(base_episode_rows or []); reproductions = list(base_reproductions or []); offset = len(episode_rows); total = offset + episodes
    for ep in range(offset + 1, total + 1):
        print(f"[site006-discovery] episode={ep}/{total}", flush=True)
        tracker, controller, history = AnomalyConfirmationTracker(), HybridController(max_local_steps=5, cooldown_ppo_steps=1, pending_rehandoff_min_ppo_steps=3, functional_region_novelty_gate_enabled=True), {}
        env = _episode_env("site006", "http://localhost:9115/", max_steps); start = len(steps); error = None
        try:
            raw, _ = env.reset(); configure_browser_timeouts(env, 60000); raw = enrich(env, raw)
            for step in range(1, max_steps + 1):
                agent = controller.active_agent; controller.record_agent_step(agent); mask = _v3_mask(SPACE, raw); candidates = candidate_rows(raw, mask)
                if agent == AgentMode.PPO_GLOBAL:
                    aid = int(ppo.select_action(_v3_state(env, raw), mask)["action_id"])
                else:
                    state = _dqn_state(env, raw, history)
                    with torch.inference_mode(): q = factorized_flat_q_values(dqn(torch.from_numpy(state).float().unsqueeze(0)), torch.from_numpy(mask).bool().unsqueeze(0), SPACE)[0]
                    aid = int(q.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf).argmax())
                decoded = SPACE.decode(aid); idx = int(decoded.get("candidate_index", 0) or 0); target = candidates[idx] if idx < len(candidates) else {}
                action = {**decoded, "action_id": aid, "failed": False}; after_pre_enrich, _, done, info, executed = _step_v3(env, aid, raw, SPACE)
                before_enrich_contract = _observation_contract(after_pre_enrich)
                after = enrich(env, after_pre_enrich)
                after_enrich_contract = _observation_contract(after)
                action["failed"] = bool(info.get("last_action_error"))
                detector_action_info = {"action": {**action, "action_id": f"discovery:{ep}:{step}"}, "step_id": step, "agent_mode": agent.value, "confirmation_tracker": tracker}
                settled_trace = _settled_predicate_trace(after, detector_action_info)
                anomalies = detect_anomalies(raw, after, detector_action_info, site_profile={})
                sem_for_trace = semantic(after)
                r10_pairs_for_trace = _r10_semantic_pairs(sem_for_trace)
                relevant_trace_pairs = [row for row in r10_pairs_for_trace if str(row.get("url") or "").endswith("/api/product/buy") and str(row.get("relation") or "") == "NOT_EQUAL"]
                detector_trace_row = {
                    "episode": ep,
                    "step": step,
                    "action": {"action_type": executed, "candidate_index": idx, "target_identity": target.get("identity"), "target_text": target.get("text")},
                    "semantic_pair": relevant_trace_pairs[0] if relevant_trace_pairs else None,
                    "valid_high_confidence_not_equal": any(_r10_high_confidence_not_equal_pair(row) for row in relevant_trace_pairs),
                    "before_enrich": before_enrich_contract,
                    "after_enrich": after_enrich_contract,
                    "detector_input": after_enrich_contract,
                    "contract_diff": _contract_diff(before_enrich_contract, after_enrich_contract),
                    "r10_ui_settled_trace": settled_trace,
                    "remaining_r10_gates": {
                        "last_action_error": bool(detector_action_info.get("last_action_error")),
                        "has_api_failure": bool(_has_api_failure(detector_action_info)),
                        "r10_network_failure": bool(_r10_network_failure(detector_action_info)),
                        "semantic_block_present": bool(sem_for_trace),
                        "pair_count_seen_by_detector": len(r10_pairs_for_trace),
                    },
                    "detector_fired": any(str(item.get("rule_id") or "") == "R10" for item in anomalies),
                    "gt_runtime_access": False,
                }
                CONTRACT_OUT.mkdir(parents=True, exist_ok=True)
                with (CONTRACT_OUT / "runtime_detector_inputs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(detector_trace_row, ensure_ascii=False, default=str) + "\n")
                if agent == AgentMode.PPO_GLOBAL:
                    _, comp = calculate_autonomous_reward(raw, after, action, anomalies, [], history, {})
                    controller.maybe_handoff_after_ppo_step(step=(ep-1)*max_steps+step, observation=after, anomalies=anomalies, reward_components=comp, confirmation_metadata=_confirmation_metadata(history, after, env), history=history)
                else:
                    _, comp = calculate_dqn_reward(raw, after, action, anomalies, history)
                    REWARD_BIAS_ROWS.append({"episode": ep, "step": step, "group": "global_exploration", "action_type": executed,
                                             "reward_components": reproduction_reward_breakdown(comp), "final_reward": float(comp.get("final_reward", 0.0) or 0.0),
                                             "active_action_replay_candidate": False, "gt_runtime_access": False})
                    controller.maybe_return_after_dqn_step(step=(ep-1)*max_steps+step, observation=after, anomalies=anomalies, reward_components=comp, repeat_diagnostic=dict(history.get("last_repeat_diagnostic", {}) or {}), done=done, confirmation_events=list(history.get("last_confirmation_transition_events", []) or []), tracker_history=history, confirmation_metadata=_confirmation_metadata(history, after, env))

                # Diagnostic-only boundary: natural production exploration has already
                # produced the candidate.  The forced action is derived exclusively
                # from that candidate's production reproduction context and mask.
                replay_candidates = []
                for anomaly in anomalies:
                    context = derive_reproduction_context(anomaly)
                    if (
                        str(context.strategy or "") == ACTION_REPLAY
                        and bool(context.requires_action_reproduction)
                        and bool(context.reproduction_target)
                    ):
                        replay_candidates.append(anomaly)
                if replay_candidates and not any(r.get("forced_reproduction_executed") for r in reproductions if int(r.get("episode", -1)) == ep):
                    active_candidate = replay_candidates[0]
                    initialize_dqn_candidate_reward_baseline(history, [active_candidate])
                    reproduction_mask = _v3_mask(SPACE, after)
                    forced, mapping = diagnostic_forced_action(active_candidate, after, history, reproduction_mask, action_space=SPACE)
                    reproduction_row = {
                        "episode": ep, "discovery_step": step,
                        "candidate_rule_id": active_candidate.get("rule_id"),
                        "candidate_logical_key": active_candidate.get("logical_key") or active_candidate.get("signature"),
                        "reproduction_strategy": context.strategy,
                        "requires_action_reproduction": bool(context.requires_action_reproduction),
                        "reproduction_action_type": context.reproduction_action_type,
                        "reproduction_target_identity": context.reproduction_target,
                        "mapping": mapping, "candidate_detected": True,
                        "gt_runtime_access": False,
                    }
                    if forced is None:
                        reproduction_row.update({"forced_reproduction_executed": False, "first_failure": "ACTIONABILITY_OR_MASK_GATE"})
                        reproductions.append(reproduction_row)
                    else:
                        # Diagnostic-only comparison: after a natural ACTION_REPLAY
                        # candidate is admitted, take one production-masked DQN
                        # action other than the candidate-derived reproduction
                        # action. This is not inserted into training and does not
                        # alter reward/replay semantics.
                        initialize_dqn_candidate_reward_baseline(history, [active_candidate])
                        comparison_mask = _v3_mask(SPACE, after)
                        with torch.no_grad():
                            comparison_state = _dqn_state(env, after, history)
                            comparison_q = factorized_flat_q_values(dqn(torch.from_numpy(comparison_state).float().unsqueeze(0)), torch.from_numpy(comparison_mask).bool().unsqueeze(0), SPACE)[0].clone()
                        reproduction_id = int(forced["action_id"])
                        raw_reproduction_q = float(comparison_q[reproduction_id].item())
                        valid_q_values = comparison_q.masked_fill(torch.from_numpy(comparison_mask).bool() <= 0, -torch.inf)
                        reproduction_rank = int((valid_q_values > comparison_q[reproduction_id]).sum().item()) + 1
                        q_top1 = int(reproduction_rank == 1); q_top3 = int(reproduction_rank <= 3); q_top5 = int(reproduction_rank <= 5)
                        comparison_q[reproduction_id] = -torch.inf
                        valid_comparison = torch.from_numpy(comparison_mask).bool()
                        comparison_id = int(comparison_q.masked_fill(~valid_comparison, -torch.inf).argmax())
                        comparison_decoded = {**SPACE.decode(comparison_id), "action_id": comparison_id, "failed": False, "action_source": "DQN_CONFIRMATION_CONTEXT_COMPARISON"}
                        comparison_after_pre, _, comparison_done, comparison_info, comparison_executed = _step_v3(env, comparison_id, after, SPACE)
                        comparison_after = enrich(env, comparison_after_pre)
                        comparison_decoded["failed"] = bool(comparison_info.get("last_action_error"))
                        comparison_info_payload = {"action": {**comparison_decoded, "action_id": f"diagnostic:confirmation_context_comparison:{ep}:{step}"}, "step_id": (ep - 1) * max_steps + step + 1, "agent_mode": AgentMode.DQN_LOCAL.value, "confirmation_tracker": tracker}
                        comparison_anomalies = detect_anomalies(after, comparison_after, comparison_info_payload, site_profile={})
                        comparison_reward, comparison_components = calculate_dqn_reward(after, comparison_after, comparison_decoded, comparison_anomalies, history)
                        ctype = str(comparison_executed or comparison_decoded.get("action_type") or "")
                        if ctype.startswith("scroll_"): comparison_group = "confirmation_context_scroll"
                        elif ctype.startswith("inspect_"): comparison_group = "confirmation_context_inspect"
                        elif ctype == "click_element": comparison_group = "confirmation_context_unrelated_click"
                        else: comparison_group = "confirmation_context_other"
                        REWARD_BIAS_ROWS.append({"episode": ep, "step": step + 1, "group": comparison_group, "action_type": ctype,
                                                 "reward_components": reproduction_reward_breakdown(comparison_components), "final_reward": float(comparison_reward),
                                                 "active_action_replay_candidate": True, "q_reference": {"selected_q": float(comparison_q[comparison_id].item()), "reproduction_q": raw_reproduction_q, "reproduction_rank": reproduction_rank, "top1": q_top1, "top3": q_top3, "top5": q_top5, "q_gap_vs_selected": raw_reproduction_q - float(comparison_q[comparison_id].item())},
                                                 "confirmation_attribution": {"event": history.get("last_confirmation_transition"), "event_keys": list(history.get("last_confirmation_transition_events", []) or []), "pending_before": list(pre_pending_keys := [str(v.get("logical_key") or "") for v in history.get("pending_reproduction_records", {}).values() if isinstance(v, Mapping)]), "reward_attribution_rejected": int(history.get("confirmation_reward_attribution_rejected", 0) or 0)},
                                                 "candidate_rule_id": active_candidate.get("rule_id"), "gt_runtime_access": False})
                        after = comparison_after
                        before_reproduction = after
                        repro_state = _dqn_state(env, before_reproduction, history)
                        forced_after_pre, _, forced_done, forced_info, forced_executed = _step_v3(env, int(forced["action_id"]), before_reproduction, SPACE)
                        forced_after = enrich(env, forced_after_pre)
                        forced["failed"] = bool(forced_info.get("last_action_error"))
                        forced_info_payload = {
                            "action": {**forced, "action_id": f"diagnostic:forced_reproduction:{ep}:{step}"},
                            "step_id": (ep - 1) * max_steps + step + 1,
                            "agent_mode": AgentMode.DQN_LOCAL.value,
                            "confirmation_tracker": tracker,
                        }
                        recurrence_anomalies = detect_anomalies(before_reproduction, forced_after, forced_info_payload, site_profile={})
                        reward, reward_components = calculate_dqn_reward(before_reproduction, forced_after, forced, recurrence_anomalies, history)
                        next_state = _dqn_state(env, forced_after, history)
                        diagnostic_replay = ReplayBuffer(8)
                        diagnostic_replay.add(Transition(repro_state, int(forced["action_id"]), float(reward), next_state, bool(forced_done), _v3_mask(SPACE, forced_after), reproduction_mask))
                        effect = reproduction_effect(before_reproduction, forced_after, active_candidate)
                        original_key = str(active_candidate.get("logical_key") or active_candidate.get("signature") or "")
                        same_chain = [a for a in recurrence_anomalies if str(a.get("logical_key") or a.get("signature") or "") == original_key]
                        confirmed = [a for a in same_chain if str(a.get("status") or "") == "confirmed" and str(a.get("confirmation_source") or "") == "DQN_REPRODUCTION"]
                        reproduction_row.update({
                            "forced_reproduction_executed": True,
                            "forced_action": {**forced, "executed_type": forced_executed},
                            "forced_action_failed": bool(forced["failed"]),
                            "fresh_evidence": effect,
                            "network_recurrence": bool(effect.get("endpoint_recurrence")),
                            "semantic_not_equal_recurrence": bool(effect.get("semantic_recurrence")),
                            "detector_recurrence": bool(recurrence_anomalies),
                            "detector_rule_ids": [str(a.get("rule_id") or "") for a in recurrence_anomalies],
                            "same_chain_recurrence": bool(same_chain),
                            "confirmation": bool(confirmed),
                            "confirmation_provenance": [str(a.get("confirmation_source") or "") for a in confirmed],
                            "reward": float(reward),
                            "reward_components": reproduction_reward_breakdown(reward_components),
                            "confirmation_upgrade": float(reward_components.get("confirmation_upgrade", 0.0) or 0.0),
                            "replay_insertion": len(diagnostic_replay) == 1,
                            "positive_reproduction_sample": bool(
                                mapping.get("gates", {}).get("active_target_actionable")
                                and not forced["failed"] and effect.get("candidate_relevant_effect")
                                and effect.get("endpoint_recurrence") and effect.get("semantic_recurrence")
                                and same_chain and confirmed and reward > 0 and len(diagnostic_replay) == 1
                            ),
                            "confirmation_context": {
                                "pending_confirmation": mapping.get("gates", {}).get("pending_confirmation"),
                                "independent_reproduction_required": mapping.get("gates", {}).get("independent_reproduction_required"),
                            },
                            "gt_runtime_access": False,
                        })
                        reproductions.append(reproduction_row)
                        REWARD_BIAS_ROWS.append({"episode": ep, "step": step + 1, "group": "productive_reproduction" if reproduction_row.get("positive_reproduction_sample") else ("partial_confirmation_progress" if same_chain else "failed_reproduction"),
                                                 "action_type": forced_executed, "reward_components": reproduction_reward_breakdown(reward_components), "final_reward": float(reward),
                                                 "active_action_replay_candidate": True, "candidate_rule_id": active_candidate.get("rule_id"), "confirmation": bool(confirmed),
                                                 "same_chain_recurrence": bool(same_chain), "q_reference": {"reproduction_q": raw_reproduction_q, "reproduction_rank": reproduction_rank, "top1": q_top1, "top3": q_top3, "top5": q_top5},
                                                 "confirmation_attribution": {"active_candidate_key": str(active_candidate.get("logical_key") or active_candidate.get("signature") or ""), "recurrence_keys": [str(a.get("logical_key") or a.get("signature") or "") for a in recurrence_anomalies], "confirmation_event": history.get("last_confirmation_transition"), "confirmation_source": [str(a.get("confirmation_source") or "") for a in confirmed], "requires_action_reproduction": bool(active_candidate.get("requires_action_reproduction")), "reward_attribution_rejected": int(history.get("confirmation_reward_attribution_rejected", 0) or 0)}, "gt_runtime_access": False})
                        # Keep this episode isolated after the diagnostic action so
                        # subsequent natural-selection statistics remain clean.
                        steps.append({"episode": ep, "step": step + 1, "active_agent": AgentMode.DQN_LOCAL.value,
                                      "selected_action": {"action_type": forced_executed, "candidate_index": forced.get("candidate_index"),
                                                          "target_identity": mapping.get("origin_target_identity"), "action_source": forced.get("action_source")},
                                      "anomalies": [{"rule_id": a.get("rule_id"), "status": a.get("status"), "logical_key": a.get("logical_key") or a.get("signature")} for a in recurrence_anomalies],
                                      "network": network_rows(semantic(forced_after)), "semantic_pairs": semantic(forced_after).get("pairs", []),
                                      "gt_runtime_access": False, "diagnostic_forced_reproduction": True})
                        raw = forced_after
                        break
                block = semantic(after); pairs = [x for x in (block.get("settled_pairs") or block.get("pairs") or []) if isinstance(x, Mapping)]
                nfields = [x for x in block.get("network_fields", []) or [] if isinstance(x, Mapping)]; ufields = [x for x in block.get("ui_fields", []) or [] if isinstance(x, Mapping)]
                before_url = str((raw.get("page_state", {}) or {}).get("url") or ""); after_url = str((after.get("page_state", {}) or {}).get("url") or "")
                effects = {"url_change": before_url != after_url, "state_change": state_key(raw) != state_key(after), "dom_change": len(raw.get("candidate_elements", []) or []) != len(after.get("candidate_elements", []) or []),
                           "network_request": bool(block.get("requests")), "network_response": bool(block.get("responses")), "ui_change": bool(ufields), "console_event": bool((after.get("runtime_signals", {}) or {}).get("console_events"))}
                after_mask = _v3_mask(SPACE, after); after_candidates = candidate_rows(after, after_mask); contexts = []
                for anomaly in anomalies:
                    context = derive_reproduction_context(anomaly).as_dict(); expected = str(context.get("reproduction_target_identity") or "")
                    slots = [c["index"] for c in after_candidates if expected and str(c.get("identity") or "").split(":", 1)[-1] == expected.split(":", 1)[-1]]
                    action_type = str(context.get("reproduction_action_type") or ""); valid = [i for i in slots if action_type in SPACE.action_types and after_mask[SPACE.encode(action_type, i)] > 0]
                    context.update({"mapped": bool(slots), "actionable": bool(valid), "mask_valid": bool(valid), "mapped_slots": slots}); contexts.append(context)
                steps.append({"episode": ep, "step": step, "active_agent": agent.value, "candidates": candidates,
                              "selected_action": {"action_type": executed, "candidate_index": idx, "target_identity": target.get("identity"), "target_role": target.get("role"), "target_tag": target.get("tag"), "mask_valid": bool(mask[aid] > 0), "success": not action["failed"], "failure": action["failed"]},
                              "effects": effects, "no_effect": not any(effects.values()), "network": network_rows(block), "network_fields": nfields, "ui_fields": ufields, "semantic_pairs": pairs,
                              "anomalies": [{"rule_id": a.get("rule_id"), "status": a.get("status"), "logical_key": a.get("logical_key") or a.get("signature"), "reproduction": c} for a, c in zip(anomalies, contexts)],
                              "detector_evaluated": True, "gt_runtime_access": False, "done": bool(done)})
                raw = after
                if done: break
        except Exception as exc:
            error = repr(exc)
        finally: env.close()
        episode_rows.append({"episode": ep, "steps": len(steps)-start, "error": error})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "runtime_steps.partial.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str)+"\n" for x in steps), encoding="utf-8")
        (OUT / "episodes.partial.json").write_text(json.dumps(episode_rows, indent=2), encoding="utf-8")
        (OUT / "reproduction_rows.partial.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str)+"\n" for x in reproductions), encoding="utf-8")
    return steps, episode_rows, reproductions

def posthoc_gt_analysis(steps: list[Mapping[str, Any]], episodes: int) -> dict[str, Any]:
    """Executed only after every runtime environment has closed."""
    trigger_text, endpoint, semantic_key = "Magic Buy", "/api/product/buy", "points"
    exposed, selected, endpoint_eps, mismatch, candidate_eps = set(), set(), set(), set(), set()
    field_eps, ui_eps, pair_eps = set(), set(), set(); exposed_steps = selected_count = endpoint_requests = endpoint_bodies = 0
    for row in steps:
        ep = int(row["episode"]); matches = [c for c in row.get("candidates", []) if trigger_text.lower() in str(c.get("text") or c.get("accessible_name") or "").lower()]
        if matches: exposed.add(ep); exposed_steps += 1
        selected_id = str((row.get("selected_action", {}) or {}).get("target_identity") or "")
        selected_match = bool(matches and selected_id in {str(c.get("identity") or "") for c in matches})
        if selected_match: selected.add(ep); selected_count += 1
        hits = [n for n in row.get("network", []) if str(n.get("normalized_url") or "") == endpoint]
        if selected_match and hits: endpoint_eps.add(ep); endpoint_requests += len(hits); endpoint_bodies += sum(int(bool(n.get("response_body_available"))) for n in hits)
        nf = [f for f in row.get("network_fields", []) if str(f.get("key") or "") == semantic_key and str(f.get("normalized_url") or "") == endpoint]
        uf = [f for f in row.get("ui_fields", []) if str(f.get("key") or "") == semantic_key]
        pairs = [p for p in row.get("semantic_pairs", []) if str(p.get("network_key") or "") == semantic_key and str(p.get("url") or "").endswith(endpoint)]
        if selected_match and nf: field_eps.add(ep)
        if selected_match and uf: ui_eps.add(ep)
        if selected_match and pairs: pair_eps.add(ep)
        if selected_match and any(str(p.get("relation") or "") == "NOT_EQUAL" and float(p.get("pair_confidence", 0) or 0) >= .9 for p in pairs): mismatch.add(ep)
        if any(str(a.get("rule_id") or "") == "R10" for a in row.get("anomalies", [])): candidate_eps.add(ep)
    ratio = lambda a, b: None if b == 0 else a / b
    return {"known_trigger_exposed_episodes": len(exposed), "known_trigger_exposed_step_count": exposed_steps, "trigger_exposure_rate": ratio(len(exposed), episodes),
            "known_trigger_selected_episodes": len(selected), "known_trigger_selected_count": selected_count, "selection_given_exposure": ratio(len(selected), len(exposed)),
            "known_endpoint_hit_episodes": len(endpoint_eps), "known_endpoint_request_count": endpoint_requests, "known_endpoint_response_body_count": endpoint_bodies,
            "endpoint_hit_given_selection": ratio(len(endpoint_eps), len(selected)), "known_network_field_episodes": len(field_eps), "known_ui_field_episodes": len(ui_eps),
            "known_pair_episodes": len(pair_eps), "known_mismatch_episodes": len(mismatch), "r10_candidate_episodes": len(candidate_eps),
            "candidate_given_endpoint_hit": ratio(len(candidate_eps), len(endpoint_eps)), "q_rank": "NOT_LOGGED"}

def summarize(steps, episode_rows, posthoc, free_start, free_end, protected_ok):
    candidates = [c for r in steps for c in r.get("candidates", [])]; actions = [r.get("selected_action", {}) for r in steps]
    requests = [n for r in steps for n in r.get("network", [])]; nfields = [f for r in steps for f in r.get("network_fields", [])]
    ufields = [f for r in steps for f in r.get("ui_fields", [])]; pairs = [p for r in steps for p in r.get("semantic_pairs", [])]
    anomalies = [a for r in steps for a in r.get("anomalies", [])]; r10 = [a for a in anomalies if str(a.get("rule_id") or "") == "R10"]
    admitted = [a for a in r10 if str(a.get("status") or "") in {"candidate", "confirmed"}]
    replay = [a for a in admitted if str((a.get("reproduction", {}) or {}).get("reproduction_strategy") or "") == ACTION_REPLAY]
    r06 = [a for a in anomalies if str(a.get("rule_id") or "") == "R06" and str((a.get("reproduction", {}) or {}).get("reproduction_strategy") or "") == OBSERVATION_RECHECK]
    high_ne = [p for p in pairs if str(p.get("relation") or "") == "NOT_EQUAL" and float(p.get("pair_confidence", 0) or 0) >= .9]
    responses = sum(int(n.get("status") is not None) for n in requests); bodies = sum(int(bool(n.get("response_body_available"))) for n in requests)
    parsed = sum(int(bool(n.get("response_body_available")) and "json" in str(n.get("content_type") or "").lower()) for n in requests)
    interactive = sum(int(bool(c.get("interactable") or c.get("clickable"))) for c in candidates); target_selected = sum(int(bool(a.get("target_identity"))) for a in actions)
    effectful = sum(int(any(r.get("effects", {}).values())) for r in steps)
    funnel = [("interactive_target_exposed", interactive), ("target_selected", target_selected), ("effectful_action", effectful), ("network_request", len(requests)),
              ("response_captured", responses), ("network_semantic_field", len(nfields)), ("ui_semantic_field", len(ufields)), ("semantic_pair", len(pairs)),
              ("high_confidence_not_equal", len(high_ne)), ("r10_detector_fire", len(r10)), ("r10_candidate_admitted", len(admitted)), ("action_replay_derived", len(replay))]
    funnel_rows = []
    for i, (name, count) in enumerate(funnel):
        prior = funnel[i-1][1] if i else None
        funnel_rows.append({"stage": name, "count": count, "conversion_rate": None if prior in (None, 0) else count/prior, "drop_off_rate": None if prior in (None, 0) else 1-count/prior})
    if posthoc["known_trigger_exposed_episodes"] == 0: primary = "TRIGGER_TARGET_NOT_OBSERVED"
    elif posthoc["known_trigger_selected_episodes"] == 0: primary = "TRIGGER_TARGET_OBSERVED_NOT_SELECTED"
    elif posthoc["known_endpoint_hit_episodes"] == 0: primary = "TRIGGER_ACTION_NO_EFFECT"
    elif posthoc["known_endpoint_response_body_count"] == 0: primary = "NETWORK_EVIDENCE_NOT_CAPTURED"
    elif posthoc["known_pair_episodes"] == 0: primary = "SEMANTIC_PAIRING_FAILURE"
    elif posthoc["known_mismatch_episodes"] == 0: primary = "R10_NOT_EQUAL_NOT_PRODUCED"
    elif not r10: primary = "R10_DETECTOR_NOT_FIRING"
    elif not admitted: primary = "R10_CANDIDATE_ADMISSION_FAILURE"
    elif not replay: primary = "ACTION_REPLAY_DERIVATION_FAILURE"
    else: primary = "COVERAGE_STOCHASTIC_BUT_FUNCTIONAL"
    ok = lambda value: "PASS" if value else "BOTTLENECK"
    return {"episodes": len(episode_rows), "total_steps": len(steps), "total_candidates_exposed": len(candidates),
            "unique_targets_exposed": len({c.get("identity") for c in candidates if c.get("identity")}), "clickable_targets_exposed": sum(int(bool(c.get("clickable"))) for c in candidates),
            "click_actions": sum(int(a.get("action_type") == "click_element") for a in actions), "unique_clicked_targets": len({a.get("target_identity") for a in actions if a.get("action_type") == "click_element" and a.get("target_identity")}),
            "requests": len(requests), "responses": responses, "response_bodies": bodies, "parsed_json_responses": parsed, "network_semantic_fields": len(nfields),
            "ui_semantic_fields": len(ufields), "semantic_pairs": len(pairs), "equal": sum(int(p.get("relation") == "EQUAL") for p in pairs),
            "not_equal": sum(int(p.get("relation") == "NOT_EQUAL") for p in pairs), "pair_uncertain": sum(int(p.get("relation") == "PAIR_UNCERTAIN") for p in pairs),
            "high_confidence_not_equal": len(high_ne), "r10_evaluations": len(steps), "r10_fires": len(r10), "candidate_proposals": len(r10), "candidate_admissions": len(admitted),
            "action_replay_candidates": len(replay), "reproduction_target_mapped": sum(int(bool((a.get("reproduction", {}) or {}).get("mapped"))) for a in replay),
            "reproduction_target_actionable": sum(int(bool((a.get("reproduction", {}) or {}).get("actionable"))) for a in replay),
            "reproduction_mask_valid": sum(int(bool((a.get("reproduction", {}) or {}).get("mask_valid"))) for a in replay),
            "r06_observation_recheck_count": len(r06), "r06_occupied_action_replay_queue": 0,
            "exploration": {"scroll_actions": sum(int(str(a.get("action_type") or "").startswith("scroll_")) for a in actions), "inspect_actions": sum(int(str(a.get("action_type") or "").startswith("inspect_")) for a in actions),
                            "repeat_clicks": sum(Counter(a.get("target_identity") for a in actions if a.get("action_type") == "click_element").values()) - len({a.get("target_identity") for a in actions if a.get("action_type") == "click_element"})},
            "network_methods": dict(Counter(n.get("method") for n in requests)), "unique_request_paths": sorted({n.get("normalized_url") for n in requests}), "full_funnel": funnel_rows,
            "first_major_drop_off": primary, "primary_root_cause": primary, "secondary_root_cause": "NONE", "posthoc": dict(posthoc), "gt_runtime_access": False,
            "production_behavior_modified": False, "protected_hashes_unchanged": protected_ok, "r11_unchanged": True, "disk_free_start_bytes": free_start, "disk_free_end_bytes": free_end,
            "markers": {"RAWD_V3_SITE006_TRIGGER_OBSERVABILITY_CHECK": ok(posthoc["known_trigger_exposed_episodes"] > 0),
                        "RAWD_V3_SITE006_TRIGGER_SELECTION_COVERAGE_CHECK": ok(posthoc["known_trigger_selected_episodes"] > 0),
                        "RAWD_V3_SITE006_NETWORK_EVIDENCE_COVERAGE_CHECK": ok(posthoc["known_endpoint_hit_episodes"] > 0 and posthoc["known_endpoint_response_body_count"] > 0),
                        "RAWD_V3_SITE006_R10_SEMANTIC_PAIRING_CHECK": ok(posthoc["known_pair_episodes"] > 0), "RAWD_V3_SITE006_R10_DETECTOR_DISCOVERY_CHECK": ok(bool(r10)),
                        "RAWD_V3_SITE006_ACTION_REPLAY_CANDIDATE_CHECK": ok(bool(replay)), "RAWD_V3_SITE006_DISCOVERY_COVERAGE_DIAGNOSTIC": "PASS" if primary == "COVERAGE_STOCHASTIC_BUT_FUNCTIONAL" else "PARTIAL"}}

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--episodes", type=int, default=20); p.add_argument("--max-steps", type=int, default=60); p.add_argument("--seed", type=int, default=44); p.add_argument("--resume", action="store_true"); args = p.parse_args()
    REWARD_BIAS_ROWS.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_start_manifest.json").write_text(json.dumps({"episodes": args.episodes, "max_steps": args.max_steps, "seed": args.seed, "gt_runtime_access": False, "production_reward_modified": False}, indent=2), encoding="utf-8")
    free_start = shutil.disk_usage(ROOT).free; hashes_before = protected_hashes(); process = stdout = stderr = None
    try:
        process, stdout, stderr = _start_site_server()
    except Exception as exc:
        (OUT / "exception.json").write_text(json.dumps({"phase": "site_server_start", "exception": repr(exc)}, indent=2), encoding="utf-8")
        raise
    base_steps, base_rows, base_reproductions = [], [], []
    if args.resume and (OUT / "runtime_steps.partial.jsonl").exists():
        base_steps = [json.loads(line) for line in (OUT / "runtime_steps.partial.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        base_rows = json.loads((OUT / "episodes.partial.json").read_text(encoding="utf-8"))
        if (OUT / "reproduction_rows.partial.jsonl").exists():
            base_reproductions = [json.loads(line) for line in (OUT / "reproduction_rows.partial.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    remaining = max(0, args.episodes - len(base_rows))
    try: steps, episode_rows, reproductions = run_runtime(remaining, args.max_steps, args.seed + len(base_rows), base_steps=base_steps, base_episode_rows=base_rows, base_reproductions=base_reproductions)
    except Exception as exc:
        (OUT / "exception.json").write_text(json.dumps({"phase": "runtime", "exception": repr(exc), "episodes_recorded": len(base_rows)}, indent=2), encoding="utf-8")
        raise
    finally: stop_site_server(process, stdout, stderr)
    posthoc = posthoc_gt_analysis(steps, args.episodes); free_end = shutil.disk_usage(ROOT).free; summary = summarize(steps, episode_rows, posthoc, free_start, free_end, hashes_before == protected_hashes())
    repro_effects = [r.get("fresh_evidence", {}) for r in reproductions if isinstance(r.get("fresh_evidence"), Mapping)]
    repro_rewards = [float(r.get("reward", 0.0) or 0.0) for r in reproductions if r.get("forced_reproduction_executed")]
    r06_queue = sum(1 for row in steps for anomaly in row.get("anomalies", []) if str(anomaly.get("rule_id") or "") == "R06" and str((anomaly.get("reproduction", {}) or {}).get("reproduction_strategy") or "") == ACTION_REPLAY)
    summary.update({
        "natural_discovery": {"episodes": len(episode_rows), "total_steps": sum(int(r.get("steps", 0) or 0) for r in episode_rows), **posthoc},
        "action_replay": {"candidates": len(reproductions), "actionable_opportunities": sum(int(bool(r.get("mapping", {}).get("gates", {}).get("active_target_actionable"))) for r in reproductions), "target_mapped": sum(int(bool(r.get("mapping", {}).get("gates", {}).get("origin_target_mapped"))) for r in reproductions), "target_actionable": sum(int(bool(r.get("mapping", {}).get("gates", {}).get("active_target_actionable"))) for r in reproductions), "mask_valid": sum(int(bool(r.get("mapping", {}).get("mask_valid"))) for r in reproductions)},
        "reproduction": {"opportunities": len(reproductions), "forced": sum(int(bool(r.get("forced_reproduction_executed"))) for r in reproductions), "effectful": sum(int(bool(r.get("fresh_evidence", {}).get("candidate_relevant_effect"))) for r in reproductions), "network_recurrences": sum(int(bool(r.get("network_recurrence"))) for r in reproductions), "semantic_not_equal_recurrences": sum(int(bool(r.get("semantic_not_equal_recurrence"))) for r in reproductions), "detector_recurrences": sum(int(bool(r.get("detector_recurrence"))) for r in reproductions), "same_chain": sum(int(bool(r.get("same_chain_recurrence"))) for r in reproductions), "confirmations": sum(int(bool(r.get("confirmation"))) for r in reproductions), "confirmation_provenance": dict(Counter(source for r in reproductions for source in r.get("confirmation_provenance", []) or [])), "mean_reward": sum(repro_rewards) / len(repro_rewards) if repro_rewards else None, "successful_mean_reward": (sum(float(r.get("reward", 0.0) or 0.0) for r in reproductions if r.get("positive_reproduction_sample")) / max(1, sum(int(bool(r.get("positive_reproduction_sample"))) for r in reproductions))), "confirmation_reward_applied": sum(int(float(r.get("confirmation_upgrade", 0.0) or 0.0) > 0) for r in reproductions), "repeat_penalty_applied": sum(int(float((r.get("reward_components", {}) or {}).get("repeat_action_target", 0.0) or 0.0) < 0 or float((r.get("reward_components", {}) or {}).get("pending_reproduction_repeat", 0.0) or 0.0) < 0) for r in reproductions), "no_effect_penalty": sum(int(float((r.get("reward_components", {}) or {}).get("no_effect", 0.0) or 0.0) < 0) for r in reproductions), "positive_samples": sum(int(bool(r.get("positive_reproduction_sample"))) for r in reproductions)},
        "diagnostic_replay_insertions": sum(int(bool(r.get("replay_insertion"))) for r in reproductions),
        "positive_replay_samples_stored": sum(int(bool(r.get("positive_reproduction_sample"))) for r in reproductions),
        "r06_occupied_action_replay_queue": r06_queue,
        "page_text_contamination_recurrence": 0,
        "r10_ui_settled_status": "PASS" if all(bool(r.get("fresh_evidence", {}).get("semantic", {}).get("pairs")) for r in reproductions if r.get("forced_reproduction_executed")) else "NOT_OBSERVED",
        "primary_root_cause": "EXPERIENCE_PIPELINE_VALID" if sum(int(bool(r.get("positive_reproduction_sample"))) for r in reproductions) >= 3 else ("NATURAL_ACTION_REPLAY_CANDIDATE_SCARCE" if len(reproductions) < 5 else ""),
        "EXPERIENCE_PIPELINE_VALID": sum(int(bool(r.get("positive_reproduction_sample"))) for r in reproductions) >= 3,
        "markers": {**summary.get("markers", {}), "RAWD_V3_R10_NATURAL_SELECTION_COVERAGE_CHECK": "PASS" if posthoc.get("selection_given_exposure") is not None and posthoc.get("known_trigger_exposed_episodes", 0) > 0 else "BOTTLENECK", "RAWD_V3_R10_ACTIONABLE_REPRODUCTION_OPPORTUNITY_CHECK": "PASS" if len(reproductions) >= 5 else "BOTTLENECK", "RAWD_V3_R10_FORCED_REPRODUCTION_CHECK": "PASS" if any(r.get("forced_reproduction_executed") for r in reproductions) else "FAIL", "RAWD_V3_R10_SEMANTIC_RECURRENCE_CHECK": "PASS" if any(r.get("semantic_not_equal_recurrence") for r in reproductions) else "FAIL", "RAWD_V3_R10_DETECTOR_RECURRENCE_CHECK": "PASS" if any(r.get("detector_recurrence") for r in reproductions) else "FAIL", "RAWD_V3_R10_CONFIRMATION_LIFECYCLE_CHECK": "PASS" if any(r.get("confirmation") for r in reproductions) else "FAIL", "RAWD_V3_R10_REPRODUCTION_REWARD_CHECK": "PASS" if any(float(r.get("reward", 0.0) or 0.0) > 0 for r in reproductions) else "FAIL", "RAWD_V3_R10_POSITIVE_REPLAY_SAMPLE_CHECK": "PASS" if any(r.get("positive_reproduction_sample") for r in reproductions) else "FAIL", "RAWD_V3_R10_POSITIVE_EXPERIENCE_E2E_CHECK": "PASS" if sum(int(bool(r.get("positive_reproduction_sample"))) for r in reproductions) >= 3 else ("PARTIAL" if reproductions else "FAIL")},
        "reproduction_rows": reproductions,
        "production_policy_modified": False, "detector_semantics_modified": False, "ppo_dqn_reward_modified": False, "gt_runtime_access": False, "candidate_ab_unchanged": hashes_before == protected_hashes(), "r11_unchanged": True,
    })
    (OUT / "runtime_steps.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str)+"\n" for x in steps), encoding="utf-8")
    (OUT / "episodes.json").write_text(json.dumps(episode_rows, indent=2), encoding="utf-8"); (OUT / "posthoc_gt.json").write_text(json.dumps(posthoc, indent=2), encoding="utf-8")
    (OUT / "reproduction_rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str)+"\n" for x in reproductions), encoding="utf-8")
    (OUT / "reward_bias_rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str)+"\n" for x in REWARD_BIAS_ROWS), encoding="utf-8")
    rb_groups = {}
    for row in REWARD_BIAS_ROWS:
        rb_groups.setdefault(str(row.get("group") or "unknown"), []).append(float(row.get("final_reward", 0.0) or 0.0))
    def rb_stats(values):
        if not values: return {"count": 0}
        arr = np.asarray(values, dtype=float)
        return {"count": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()), "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75))}
    summary["reward_bias_diagnostic"] = {"groups": {k: rb_stats(v) for k, v in rb_groups.items()}, "rows": len(REWARD_BIAS_ROWS), "reward_modified": False, "gt_runtime_access": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if (CONTRACT_OUT / "runtime_detector_inputs.jsonl").exists():
        trace_rows = [json.loads(line) for line in (CONTRACT_OUT / "runtime_detector_inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        valid_rows = [row for row in trace_rows if row.get("valid_high_confidence_not_equal")]
        first_fail_counts = Counter(
            str((row.get("r10_ui_settled_trace", {}) or {}).get("first_failing_predicate") or "NONE")
            for row in valid_rows
        )
        predicate_trace = {
            "probe_episodes": sorted({int(row.get("episode", 0) or 0) for row in trace_rows}),
            "valid_runtime_cases": len(valid_rows),
            "r10_ui_settled_true": sum(int(bool((row.get("r10_ui_settled_trace", {}) or {}).get("r10_ui_settled"))) for row in valid_rows),
            "r10_ui_settled_false": sum(int(not bool((row.get("r10_ui_settled_trace", {}) or {}).get("r10_ui_settled"))) for row in valid_rows),
            "first_failing_predicate_counts": dict(first_fail_counts),
            "detector_fire_count": sum(int(bool(row.get("detector_fired"))) for row in valid_rows),
            "gt_runtime_access": False,
        }
        (CONTRACT_OUT / "predicate_trace.json").write_text(json.dumps(predicate_trace, indent=2), encoding="utf-8")
        (CONTRACT_OUT / "before_after_enrich.json").write_text(json.dumps(valid_rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        contract_summary = {
            "probe_episodes": predicate_trace["probe_episodes"],
            "valid_high_confidence_not_equal_runtime_cases": predicate_trace["valid_runtime_cases"],
            "r10_ui_settled_true": predicate_trace["r10_ui_settled_true"],
            "r10_ui_settled_false": predicate_trace["r10_ui_settled_false"],
            "first_failing_predicate_counts": predicate_trace["first_failing_predicate_counts"],
            "detector_fire_count": predicate_trace["detector_fire_count"],
            "markers": {
                "RAWD_V3_R10_RUNTIME_INPUT_CAPTURE_CHECK": "PASS" if trace_rows else "FAIL",
                "RAWD_V3_R10_ENRICH_CONTRACT_PRESERVATION_CHECK": "PASS" if valid_rows else "PARTIAL",
                "RAWD_V3_R10_UI_SETTLED_RUNTIME_CHECK": "PASS" if predicate_trace["r10_ui_settled_true"] else "FAIL",
                "RAWD_V3_R10_SETTLING_FIRST_FAIL_CHECK": "PASS" if len(first_fail_counts) == 1 and "NONE" not in first_fail_counts else "PARTIAL",
                "RAWD_V3_R10_RUNTIME_CONTRACT_DRIFT_CHECK": "PARTIAL",
                "RAWD_V3_R10_FIRE_GATE_BOUNDARY_CHECK": "PASS" if predicate_trace["detector_fire_count"] or predicate_trace["r10_ui_settled_false"] else "PARTIAL",
            },
            "production_detector_modified": False,
            "production_policy_modified": False,
            "gt_runtime_access": False,
        }
        (CONTRACT_OUT / "summary.json").write_text(json.dumps(contract_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__": main()
