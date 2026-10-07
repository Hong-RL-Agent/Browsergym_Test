"""GT-blind active-candidate forced-reproduction diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dqn_error_discovery.agents.confirmation_observation import build_active_candidate_confirmation_context
from dqn_error_discovery.agents.confirmation_reproduction import TARGET_ACTION_TYPES, derive_reproduction_context
from dqn_error_discovery.agents.dqn_agent import ReplayBuffer, Transition
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward, initialize_dqn_candidate_reward_baseline
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.run3_integrated_validation import _confirmation_metadata, enrich, json_default, state_key
from dqn_error_discovery.v3.run4_candidate_ab_validation import configure_browser_timeouts
from dqn_error_discovery.v3.run5_context_aware_dqn_training import active_candidate_from_anomalies, active_endpoint_and_key, anomaly_key, network_requests, semantic_pairs
from dqn_error_discovery.v3.train import _dqn_state, _episode_env, _step_v3, _v3_mask
from services.anomaly_detection_service import AnomalyConfirmationTracker, detect_anomalies

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/training_v3/confirmation_forced_reproduction_diagnostic"
SPACE = V3ActionSpace(32)
ACTION_SOURCE = "DIAGNOSTIC_FORCED_REPRODUCTION"
SITE_ROOT = Path(r"C:\workspace\RL_Errorsite\backend-errorsite\site006")


def candidate_metadata(candidate: Mapping[str, Any] | None, index: int | None = None) -> dict[str, Any]:
    row = candidate if isinstance(candidate, Mapping) else {}
    bbox = row.get("bbox")
    bbox_valid = False
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            bbox_valid = float(bbox[2] or 0) > 0 and float(bbox[3] or 0) > 0
        except (TypeError, ValueError):
            bbox_valid = False
    return {
        "candidate_index": index,
        "identity": str(row.get("canonical_target_key") or row.get("bid") or row.get("id") or ""),
        "bid": row.get("bid"), "tag": row.get("tag"), "role": row.get("role"),
        "text": row.get("text"), "accessible_name": row.get("name") or row.get("aria_label"),
        "selector": row.get("selector") or row.get("locator") or row.get("dom_path"),
        "bbox": bbox, "bbox_valid": bbox_valid,
        "visible": bool(row.get("visible", True)), "visibility": row.get("visibility"),
        "enabled": bool(row.get("enabled", True)), "clickable": bool(row.get("clickable")),
        "interactable": row.get("interactable", row.get("is_interactive")),
        "input_capable": bool(row.get("fillable") or str(row.get("tag") or "").lower() in {"input", "textarea"} or str(row.get("role") or "").lower() in {"textbox", "searchbox"}),
        "select_capable": bool(row.get("canonical_dom_resolved") and str(row.get("canonical_dom_tag") or "").lower() == "select"),
        "dom_attached": bool(row.get("canonical_dom_resolved")),
        "viewport_presence": row.get("in_viewport", row.get("viewport_visible")),
        "occluded": row.get("occluded", row.get("is_occluded")),
        "hidden": bool(row.get("hidden")), "disabled": bool(row.get("disabled")), "aria_hidden": bool(row.get("aria_hidden")),
        "canonical_dom_tag": row.get("canonical_dom_tag"), "canonical_dom_role": row.get("canonical_dom_role"),
        "canonical_dom_name": row.get("canonical_dom_name"), "canonical_resolution_method": row.get("canonical_resolution_method"),
    }


def mask_rejection_trace(action_type: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Explain v3/action_contract.py predicates without changing them."""
    checks: list[dict[str, Any]] = []
    def check(predicate: str, passed: bool, reason: str) -> bool:
        checks.append({"predicate": predicate, "passed": bool(passed), "reason": reason if not passed else ""})
        return bool(passed)
    if not check("candidate.visible == true", bool(candidate.get("visible", True)), "MASK_REJECT_NOT_VISIBLE"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    if not check("candidate.enabled == true", bool(candidate.get("enabled", True)), "MASK_REJECT_DISABLED"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    if not check("not candidate.hidden", not bool(candidate.get("hidden")), "MASK_REJECT_HIDDEN"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    if not check("not candidate.disabled", not bool(candidate.get("disabled")), "MASK_REJECT_DISABLED_FLAG"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    if not check("not candidate.aria_hidden", not bool(candidate.get("aria_hidden")), "MASK_REJECT_ARIA_HIDDEN"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    if not check("candidate.bid is non-empty", bool(str(candidate.get("bid") or "").strip()), "MASK_REJECT_MISSING_IDENTITY"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    bbox = candidate.get("bbox")
    if not check("bbox has four values", isinstance(bbox, (list, tuple)) and len(bbox) >= 4, "MASK_REJECT_INVALID_BBOX"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    try:
        positive_bbox = float(bbox[2] or 0) > 0 and float(bbox[3] or 0) > 0
    except (TypeError, ValueError):
        positive_bbox = False
    if not check("bbox.width > 0 and bbox.height > 0", positive_bbox, "MASK_REJECT_ZERO_BBOX"):
        return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    role, tag = str(candidate.get("role") or "").lower(), str(candidate.get("tag") or "").lower()
    if action_type == "click_element":
        compatible = bool(candidate.get("clickable") or candidate.get("set_of_marks") or role in {"button", "link", "checkbox", "radio", "menuitem", "tab", "option"} or tag in {"button", "a", "summary", "option", "label"})
        if not check("clickable or click-compatible role/tag", compatible, "MASK_REJECT_ACTION_TYPE_MISMATCH"):
            return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    elif action_type in {"fill_input", "press_enter"}:
        input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
        compatible = bool(candidate.get("fillable") or tag in {"input", "textarea"} or role in {"textbox", "searchbox"} or candidate.get("contenteditable")) and input_type not in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"}
        if not check("input-capable target", compatible, "MASK_REJECT_ACTION_TYPE_MISMATCH"):
            return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    elif action_type == "select_option":
        compatible = bool(candidate.get("canonical_dom_resolved") and str(candidate.get("canonical_dom_tag") or "").lower() == "select")
        if not check("canonical DOM target is select", compatible, "MASK_REJECT_ACTION_TYPE_MISMATCH"):
            return {"actionable": False, "first_failing_predicate": checks[-1]["predicate"], "reason": checks[-1]["reason"], "checks": checks}
    else:
        return {"actionable": False, "first_failing_predicate": "action type requires supported target contract", "reason": "MASK_REJECT_ACTION_TYPE_MISMATCH", "checks": checks}
    return {"actionable": True, "first_failing_predicate": None, "reason": "", "checks": checks}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def _site_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:9115/", timeout=2) as response:
            return int(response.status) == 200
    except Exception:
        return False


def _start_site_server() -> tuple[subprocess.Popen[str] | None, Any, Any]:
    if _site_reachable():
        return None, None, None
    ART.mkdir(parents=True, exist_ok=True)
    stdout = (ART / "site006_server.stdout.log").open("w", encoding="utf-8")
    stderr = (ART / "site006_server.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(["node", "server.js"], cwd=str(SITE_ROOT), stdout=stdout, stderr=stderr, text=True)
    deadline = time.time() + 30
    while time.time() < deadline and process.poll() is None:
        if _site_reachable():
            return process, stdout, stderr
        time.sleep(0.5)
    process.terminate()
    stdout.close()
    stderr.close()
    raise RuntimeError("site006 server failed to start")


def diagnostic_forced_action(active_candidate: Mapping[str, Any] | None, observation: Mapping[str, Any], history: Mapping[str, Any], mask: np.ndarray, *, action_space: V3ActionSpace = SPACE) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build an action solely from production active-candidate context."""
    _, meta = build_active_candidate_confirmation_context(history, observation, action_space, active_candidate=active_candidate, mask=mask)
    flags = meta.get("decoded_flags", {}) if isinstance(meta.get("decoded_flags"), Mapping) else {}
    names = ("active_candidate_exists", "pending_confirmation", "independent_reproduction_required", "origin_action_available", "origin_target_mapped", "active_target_actionable")
    gates = {name: bool(flags.get(name, 0)) for name in names}
    details = {
        "gates": gates,
        "origin_action_type": meta.get("decoded_origin_action"),
        "origin_target_identity": meta.get("expected_target_identity"),
        "matching_candidate_slots": list(meta.get("matching_candidate_slots", []) or []),
        "remapped_index": meta.get("decoded_target_index"),
        "mask_valid": False,
        "actionable": gates["active_target_actionable"],
    }
    if not all(gates.values()):
        return None, details
    action_type, slot = str(meta.get("decoded_origin_action") or ""), meta.get("decoded_target_index")
    try:
        action_id = int(action_space.encode(action_type, int(slot)))
    except (TypeError, ValueError, KeyError):
        return None, details
    details["mask_valid"] = bool(0 <= action_id < len(mask) and mask[action_id] > 0)
    if not details["mask_valid"]:
        return None, details
    return {**action_space.decode(action_id), "action_id": action_id, "action_source": ACTION_SOURCE, "failed": False}, details


def _discovery_action_ids(observation: Mapping[str, Any], mask: np.ndarray) -> list[int]:
    """Target/endpoint/taxonomy-blind target-consuming exploration."""
    count = min(len(observation.get("candidate_elements", []) or []), SPACE.max_candidates)
    return [SPACE.encode(kind, slot) for kind in SPACE.action_types if kind in TARGET_ACTION_TYPES for slot in range(count) if mask[SPACE.encode(kind, slot)] > 0]


def _semantic_snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    semantic = page.get("network_ui_semantic")
    if not isinstance(semantic, Mapping):
        runtime = observation.get("runtime_signals", {})
        semantic = runtime.get("network_ui_semantic") if isinstance(runtime, Mapping) else {}
    semantic = semantic if isinstance(semantic, Mapping) else {}
    pairs = semantic_pairs(observation)
    return {
        "requests": network_requests(observation),
        "network_fields": list(semantic.get("network_fields", []) or []),
        "ui_fields": list(semantic.get("ui_fields", []) or []),
        "pairs": pairs,
        "relations": dict(Counter(str(row.get("relation") or "PAIR_UNCERTAIN") for row in pairs)),
    }


def _effect(before: Mapping[str, Any], after: Mapping[str, Any], active: Mapping[str, Any]) -> dict[str, Any]:
    endpoint, field = active_endpoint_and_key(active)
    fresh = _semantic_snapshot(after)
    requests, pairs = fresh["requests"], fresh["pairs"]
    endpoint_rows = [row for row in requests if endpoint and endpoint in str(row.get("normalized_url") or row.get("url") or "")]
    relevant_pairs = [row for row in pairs if field and str(row.get("network_key") or row.get("ui_key") or row.get("field") or "") == field and (not endpoint or endpoint in str(row.get("request_path") or row.get("normalized_url") or row.get("url") or ""))]
    evidence = active.get("evidence", {}) if isinstance(active.get("evidence"), Mapping) else {}
    relation = str(evidence.get("relation") or "")
    semantic_recurrence = bool(relevant_pairs and (not relation or any(str(row.get("relation") or "") == relation for row in relevant_pairs)))
    before_page = before.get("page_state", {}) if isinstance(before.get("page_state"), Mapping) else {}
    after_page = after.get("page_state", {}) if isinstance(after.get("page_state"), Mapping) else {}
    state_changed = state_key(before) != state_key(after)
    ui_changed = str(before_page.get("page_text") or "") != str(after_page.get("page_text") or "")
    return {
        "request_generated": bool(requests),
        "normalized_request_paths": [str(row.get("normalized_url") or row.get("url") or "") for row in requests],
        "response_received": any(row.get("response_received", True) is not False for row in requests),
        "response_body_available": any(bool(row.get("response_body") or row.get("body") or row.get("response_json")) for row in requests),
        "semantic_fields_extracted": bool(fresh["network_fields"] or fresh["ui_fields"]),
        "endpoint_recurrence": bool(endpoint_rows), "semantic_recurrence": semantic_recurrence,
        "candidate_relevant_effect": bool(endpoint_rows or relevant_pairs or state_changed or ui_changed),
        "state_changed": state_changed, "ui_changed": ui_changed, "semantic": fresh,
    }


def _reward_breakdown(components: Mapping[str, Any]) -> dict[str, float]:
    aliases = {
        "new_local_state": "new_local_state", "new_local_action_target": "new_local_action_target", "flow_progress": "flow_progress",
        "candidate_anomaly": "candidate_anomaly", "confirmed_anomaly": "confirmed_anomaly", "confirmation_upgrade": "confirmation_upgrade",
        "repeat_action_target": "repeat_action_penalty", "pending_reproduction_repeat": "repeat_action_penalty",
        "repeat_state": "repeat_state_penalty", "repeat_anomaly": "repeat_anomaly_penalty", "no_effect": "no_effect_penalty", "final_reward": "final_reward",
    }
    return {name: float(components.get(source, 0.0) or 0.0) for name, source in aliases.items()}


def run_episode(episode: int, browser_timeout_ms: int, max_discovery_actions: int) -> dict[str, Any]:
    tracker = AnomalyConfirmationTracker()
    controller = HybridController(max_local_steps=1, cooldown_ppo_steps=0)
    history: dict[str, Any] = {}
    env = _episode_env("site006", "http://localhost:9115/", max_discovery_actions + 4)
    trace: dict[str, Any] = {"episode": episode, "gt_runtime_access": False, "discovery": [], "candidate_sequence": []}
    try:
        raw, _ = env.reset()
        configure_browser_timeouts(env, browser_timeout_ms)
        raw = enrich(env, raw)
        active: Mapping[str, Any] | None = None
        detection_observation = raw
        detection_anomalies: list[Mapping[str, Any]] = []
        observed_candidate: Mapping[str, Any] | None = None
        observed_observation = raw
        observed_anomalies: list[Mapping[str, Any]] = []
        observed_origin_metadata: dict[str, Any] = {}
        discovery_ids = _discovery_action_ids(raw, _v3_mask(SPACE, raw))[:max_discovery_actions]
        for attempt, action_id in enumerate(discovery_ids, start=1):
            if attempt > 1:
                reset_raw, _ = env.reset()
                raw = enrich(env, reset_raw)
                tracker, history = AnomalyConfirmationTracker(), {}
            decoded = SPACE.decode(action_id)
            action = {**decoded, "action_id": int(action_id), "action_source": "DIAGNOSTIC_GENERIC_DISCOVERY", "failed": False}
            after, _, done, info, _ = _step_v3(env, action_id, raw, SPACE)
            after = enrich(env, after)
            action["failed"] = bool(info.get("last_action_error"))
            anomalies = detect_anomalies(raw, after, {
                "action": {**action, "action_id": f"diagnostic:discovery:{episode}:{attempt}"},
                "step_id": 1, "agent_mode": "PPO_GLOBAL", "confirmation_tracker": tracker,
            }, site_profile={})
            for anomaly in anomalies:
                context = derive_reproduction_context(anomaly)
                trace["candidate_sequence"].append({
                    "attempt": attempt,
                    "rule_id": anomaly.get("rule_id"),
                    "logical_key": anomaly_key(anomaly),
                    "evidence_target_identity": context.evidence_target,
                    "trigger_action_type": context.trigger_action_type,
                    "trigger_target_identity": context.trigger_target,
                    "reproduction_strategy": context.strategy,
                    "reproduction_action_type": context.reproduction_action_type,
                    "reproduction_target_identity": context.reproduction_target,
                    "requires_action_reproduction": context.requires_action_reproduction,
                })
            candidate = active_candidate_from_anomalies(anomalies)
            discovery_row = {"attempt": attempt, "action_type": decoded.get("action_type"), "candidate_index": decoded.get("candidate_index"), "anomaly_count": len(anomalies), "candidate_detected": bool(candidate), "done": done}
            if candidate is not None:
                initialize_dqn_candidate_reward_baseline(history, anomalies)
                meta = build_active_candidate_confirmation_context(history, after, SPACE, active_candidate=candidate, mask=_v3_mask(SPACE, after))[1]
                origin_candidates = raw.get("candidate_elements", []) or []
                origin_index = int(decoded.get("candidate_index", 0) or 0)
                origin_element = origin_candidates[origin_index] if origin_index < len(origin_candidates) and isinstance(origin_candidates[origin_index], Mapping) else {}
                slots = list(meta.get("matching_candidate_slots", []) or [])
                current_candidates = after.get("candidate_elements", []) or []
                current_index = int(slots[0]) if slots else None
                current_element = current_candidates[current_index] if current_index is not None and current_index < len(current_candidates) and isinstance(current_candidates[current_index], Mapping) else {}
                reason_trace = mask_rejection_trace(str(meta.get("decoded_origin_action") or ""), current_element)
                if observed_candidate is None:
                    observed_candidate, observed_observation, observed_anomalies = candidate, after, anomalies
                    observed_origin_metadata = candidate_metadata(origin_element, origin_index)
                discovery_row.update({"candidate_rule_id": candidate.get("rule_id"), "candidate_logical_key": anomaly_key(candidate), "candidate_action_type": candidate.get("action_type"), "candidate_target_id": candidate.get("target_id"), "context_flags": meta.get("decoded_flags"), "expected_target_identity": meta.get("expected_target_identity"), "matching_candidate_slots": meta.get("matching_candidate_slots")})
                discovery_row.update({"origin_metadata": candidate_metadata(origin_element, origin_index), "current_metadata": candidate_metadata(current_element, current_index), "mask_rejection_trace": reason_trace})
                required = ("active_candidate_exists", "pending_confirmation", "independent_reproduction_required", "origin_action_available", "origin_target_mapped", "active_target_actionable")
                if all(bool(meta.get("decoded_flags", {}).get(name, 0)) for name in required):
                    active, detection_observation, detection_anomalies = candidate, after, anomalies
                    trace["discovery"].append(discovery_row)
                    break
            trace["discovery"].append(discovery_row)
            raw = after

        if active is None and observed_candidate is not None:
            active, detection_observation, detection_anomalies = observed_candidate, observed_observation, observed_anomalies
        trace["candidate_detected"] = bool(active)
        if active is None:
            trace.update({"status": "ACTIVE_CANDIDATE_UNAVAILABLE", "forced_reproduction_executed": False})
            return trace
        mask = _v3_mask(SPACE, detection_observation)
        forced, mapping = diagnostic_forced_action(active, detection_observation, history, mask)
        current_candidates = detection_observation.get("candidate_elements", []) or []
        current_index = mapping.get("remapped_index")
        current_element = current_candidates[int(current_index)] if current_index is not None and int(current_index) < len(current_candidates) and isinstance(current_candidates[int(current_index)], Mapping) else {}
        rejection = mask_rejection_trace(str(mapping.get("origin_action_type") or ""), current_element)
        trace.update({"active_candidate": True, "original_logical_key": anomaly_key(active), "candidate_rule_id": str(active.get("rule_id") or ""), "mapping": mapping,
                      "origin_metadata": observed_origin_metadata, "current_metadata": candidate_metadata(current_element, int(current_index) if current_index is not None else None),
                      "mask_rejection_trace": rejection,
                      "action_type_target_compatibility": bool(rejection.get("actionable")) if rejection.get("reason") == "MASK_REJECT_ACTION_TYPE_MISMATCH" else bool(rejection.get("reason") != "MASK_REJECT_ACTION_TYPE_MISMATCH")})
        controller.record_agent_step(AgentMode.PPO_GLOBAL)
        trace["handoff_reason"] = controller.maybe_handoff_after_ppo_step(
            step=1, observation=detection_observation, anomalies=detection_anomalies, reward_components={},
            confirmation_metadata=_confirmation_metadata(history, detection_observation, env), history=history,
        )
        if forced is None:
            trace.update({"status": "FORCED_ACTION_GATED", "forced_reproduction_executed": False})
            return trace

        before = detection_observation
        state = _dqn_state(env, before, history, active_candidate=active)
        after, _, done, info, executed_type = _step_v3(env, int(forced["action_id"]), before, SPACE)
        after = enrich(env, after)
        forced["failed"] = bool(info.get("last_action_error"))
        local_anomalies = detect_anomalies(before, after, {
            "action": {**forced, "action_id": f"diagnostic:forced:{episode}"},
            "step_id": 2, "agent_mode": "DQN_LOCAL", "confirmation_tracker": tracker,
        }, site_profile={})
        reward, components = calculate_dqn_reward(before, after, forced, local_anomalies, history)
        next_state, next_mask = _dqn_state(env, after, history, active_candidate=active), _v3_mask(SPACE, after)
        replay = ReplayBuffer(8)
        replay.add(Transition(state, int(forced["action_id"]), float(reward), next_state, bool(done), next_mask, mask))
        effect = _effect(before, after, active)
        same_chain = [row for row in local_anomalies if anomaly_key(row) == anomaly_key(active)]
        confirmed = [row for row in same_chain if str(row.get("status") or "") == "confirmed"]
        provenance = [str(row.get("confirmation_source") or "") for row in confirmed if row.get("confirmation_source")]
        positive = all((trace["candidate_detected"], mapping["gates"]["pending_confirmation"], mapping["gates"]["origin_target_mapped"], mapping["gates"]["active_target_actionable"], not forced["failed"], effect["candidate_relevant_effect"], effect["endpoint_recurrence"], effect["semantic_recurrence"], bool(same_chain), bool(confirmed), reward > 0, len(replay) == 1))
        trace.update({
            "status": "POSITIVE_REPRODUCTION_SAMPLE" if positive else "REPRODUCTION_COMPLETED",
            "forced_reproduction_executed": True, "action": {**forced, "executed_type": executed_type},
            "fresh_evidence": effect, "detector_evaluated": True, "detector_fired": bool(local_anomalies),
            "detector_taxonomies": [str(row.get("rule_id") or "") for row in local_anomalies],
            "recurrence_logical_keys": [anomaly_key(row) for row in local_anomalies],
            "same_logical_chain_recurrence": bool(same_chain), "confirmation_upgrade": bool(confirmed),
            "confirmation_provenance": provenance, "reward": float(reward), "reward_components": _reward_breakdown(components),
            "replay_insertion": len(replay) == 1,
            "replay_transition": {"state_shape": list(state.shape), "action": int(forced["action_id"]), "reward": float(reward), "next_state_shape": list(next_state.shape), "done": bool(done), "confirmation_context": mapping, "logical_key": anomaly_key(active), "provenance": provenance},
            "positive_reproduction_sample": positive,
        })
        return trace
    finally:
        env.close()


def negative_control() -> dict[str, Any]:
    active = {"rule_id": "R01", "signature": "runtime-negative-control", "status": "candidate", "pending_confirmation": True, "confirmation_policy": "reproduce", "action_type": "click_element", "target_id": "runtime-unmapped-target"}
    action, details = diagnostic_forced_action(active, {"candidate_elements": []}, {}, np.zeros(640, dtype=np.float32))
    return {"source": "unit-level synthetic context", "uses_gt": False, "forced_reproduction_count": int(action is not None), "details": details}


def summarize(rows: list[Mapping[str, Any]], negative: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    count = lambda key: sum(int(bool(row.get(key))) for row in rows)
    gate = lambda key: sum(int(bool((row.get("mapping", {}).get("gates", {}) if isinstance(row.get("mapping"), Mapping) else {}).get(key))) for row in rows)
    effects = [row.get("fresh_evidence", {}) for row in rows if isinstance(row.get("fresh_evidence"), Mapping)]
    rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows if row.get("forced_reproduction_executed")]
    positive = [row for row in rows if row.get("positive_reproduction_sample")]
    forced, endpoint, semantic = count("forced_reproduction_executed"), sum(int(bool(x.get("endpoint_recurrence"))) for x in effects), sum(int(bool(x.get("semantic_recurrence"))) for x in effects)
    same_chain, confirmations, replay = count("same_logical_chain_recurrence"), count("confirmation_upgrade"), count("replay_insertion")
    valid = len(rows) >= 5 and all(value >= 3 for value in (forced, endpoint, semantic, same_chain, confirmations, len(positive)))
    if valid: primary = "EXPERIENCE_PIPELINE_VALID"
    elif not rows: primary = "DIAGNOSTIC_RUNNER_FAILURE"
    elif count("candidate_detected") == 0: primary = "ACTIVE_CANDIDATE_UNAVAILABLE"
    elif gate("origin_target_mapped") == 0: primary = "TARGET_MAPPING_FAILURE"
    elif gate("active_target_actionable") == 0: primary = "TARGET_NOT_ACTIONABLE"
    elif forced == 0: primary = "ACTION_REPRODUCTION_FAILURE"
    elif not any(bool(x.get("candidate_relevant_effect")) for x in effects): primary = "EFFECT_RECURRENCE_FAILURE"
    elif endpoint == 0 or semantic == 0: primary = "EVIDENCE_RECURRENCE_FAILURE"
    elif same_chain == 0: primary = "DETECTOR_RECURRENCE_FAILURE"
    elif confirmations == 0: primary = "CONFIRMATION_LIFECYCLE_FAILURE"
    elif not any(value > 0 for value in rewards): primary = "REWARD_ATTRIBUTION_FAILURE"
    else: primary = "REPLAY_INSERTION_FAILURE"
    downstream = lambda value: "NOT_EXECUTED" if forced == 0 else ("PASS" if value >= (1 if smoke else 3) else "FAIL")
    effect_count = sum(int(bool(x.get("candidate_relevant_effect"))) for x in effects)
    return {
        "version": "RAWD_V3_CONFIRMATION_FORCED_REPRODUCTION_DIAGNOSTIC", "runner_implemented": True,
        "previous_result_correction": "DIAGNOSTIC_RUNNER_NOT_IMPLEMENTED",
        "runner_path": str(Path(__file__).resolve()), "hardcoded_target_present": False, "hardcoded_endpoint_present": False, "gt_runtime_access": False,
        "smoke": smoke, "probe_episodes": len(rows), "candidate_detected": count("candidate_detected"), "active_candidate": count("active_candidate"),
        "origin_action_available": gate("origin_action_available"), "target_mapped": gate("origin_target_mapped"), "target_actionable": gate("active_target_actionable"),
        "forced_reproduction_executed": forced, "candidate_relevant_effect": effect_count, "endpoint_recurrence": endpoint, "semantic_recurrence": semantic,
        "detector_recurrence": count("detector_fired"), "same_logical_chain_recurrence": same_chain, "confirmation": confirmations,
        "confirmation_provenance": dict(Counter(source for row in rows for source in row.get("confirmation_provenance", []) or [])),
        "positive_reproduction_samples": len(positive), "mean_final_reward": sum(rewards) / len(rewards) if rewards else None,
        "successful_reward_component_breakdown": [row.get("reward_components", {}) for row in positive],
        "diagnostic_replay_insertions": replay, "positive_replay_samples_stored": len(positive), "negative_control": dict(negative),
        "primary_bottleneck": primary, "EXPERIENCE_PIPELINE_VALID": valid,
        "markers": {
            "RAWD_V3_CONFIRMATION_DIAGNOSTIC_RUNNER_CHECK": "PASS",
            "RAWD_V3_CONFIRMATION_FORCED_REPRODUCTION_CHECK": "PASS" if forced >= (1 if smoke else 3) else "BLOCKED",
            "RAWD_V3_REPRODUCTION_EFFECT_RECURRENCE_CHECK": downstream(effect_count),
            "RAWD_V3_REPRODUCTION_DETECTOR_RECURRENCE_CHECK": downstream(same_chain),
            "RAWD_V3_REPRODUCTION_CONFIRMATION_CHECK": downstream(confirmations),
            "RAWD_V3_REPRODUCTION_REWARD_ATTRIBUTION_CHECK": "NOT_EXECUTED" if forced == 0 else ("PASS" if any(value > 0 for value in rewards) else "FAIL"),
            "RAWD_V3_REPRODUCTION_REPLAY_INSERTION_CHECK": downstream(replay),
            "RAWD_V3_POSITIVE_REPRODUCTION_EXPERIENCE_PIPELINE_CHECK": "PASS" if valid else ("PARTIAL" if positive else "FAIL"),
        }, "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--browser-timeout-ms", type=int, default=15000)
    parser.add_argument("--max-discovery-actions", type=int, default=32)
    args = parser.parse_args()
    count = 1 if args.smoke else max(5, args.episodes)
    started = time.time()
    process, stdout, stderr = _start_site_server()
    try:
        rows = [run_episode(i + 1, args.browser_timeout_ms, args.max_discovery_actions) for i in range(count)]
        summary = summarize(rows, negative_control(), args.smoke)
        summary.update({"wall_time_seconds": time.time() - started, "disk_free_bytes": shutil.disk_usage(ROOT).free})
        suffix = "smoke" if args.smoke else "summary"
        ART.mkdir(parents=True, exist_ok=True)
        (ART / f"confirmation_forced_reproduction_{suffix}_episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=json_default) + "\n" for row in rows), encoding="utf-8")
        write_json(ART / f"confirmation_forced_reproduction_{suffix}.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in (stdout, stderr):
            if handle is not None:
                handle.close()


if __name__ == "__main__":
    main()
