"""Diagnostic-only decomposition of evidence through candidate admission.

This module deliberately does not train, alter detector rules, or use GT.  It
replays the fixed productive-context pilot shape and records the distinction
between broad context evidence, emitted anomalies, and the existing DQN
candidate-reward admission path.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, factorized_flat_q_values
from dqn_error_discovery.runners.train_dqn_error_detector import (
    _candidate_reward_attribution,
    _score_dqn_anomalies,
    calculate_dqn_reward,
)
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.candidate_dom_enrichment import enrich_candidates_with_dom
from dqn_error_discovery.v3.productive_context_sampler import (
    classify_context,
    evidence_types,
    semantic_fingerprint,
    target_fingerprint,
)
from dqn_error_discovery.v3.train import SITES, _choose_dqn, _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state
from envs.browsergym_jaws_env import _active_page
from services.anomaly_detection_service import detect_anomalies

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/evidence_candidate_conversion"
SPACE = V3ActionSpace(32)
RULES = [f"R{i:02d}" for i in range(1, 16)]


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _enrich(env: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **raw,
        "candidate_elements": enrich_candidates_with_dom(
            _active_page(env.env), list(raw.get("candidate_elements", []) or [])
        ),
    }


def _page_state(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("page_state", {})
    return value if isinstance(value, Mapping) else {}


def _raw_evidence(before: Mapping[str, Any], after: Mapping[str, Any], info: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
    before_page = _page_state(before)
    after_page = _page_state(after)
    before_url = str(before_page.get("url") or before.get("url") or "")
    after_url = str(after_page.get("url") or after.get("url") or "")
    before_text = str(before_page.get("page_text_sample") or "")
    after_text = str(after_page.get("page_text_sample") or "")
    before_candidates = before.get("candidate_elements", []) or []
    after_candidates = after.get("candidate_elements", []) or []
    last_error = bool(info.get("last_action_error"))
    return {
        "url_changed": before_url != after_url,
        "dom_changed": len(before_candidates) != len(after_candidates),
        "visual_changed": bool(info.get("visual_changed", False)),
        "layout_changed": bool(info.get("layout_changed", False)),
        "network_request_count": len(info.get("network_entries", []) or []),
        "network_failure_count": int(info.get("network_failure_count", 0) or 0),
        "http_4xx_count": int(info.get("api_4xx_count", 0) or 0),
        "http_5xx_count": int(info.get("api_5xx_count", 0) or 0),
        "console_error_count": int(info.get("console_error_count", 0) or 0),
        "target_present": bool(action.get("candidate_index") is not None),
        "target_visible": bool(info.get("target_visible", True)),
        "target_enabled": bool(info.get("target_enabled", True)),
        "action_effect": before_url != after_url or before_text != after_text or len(before_candidates) != len(after_candidates),
        "no_effect": before_url == after_url and before_text == after_text and len(before_candidates) == len(after_candidates),
        "temporal_status": info.get("temporal_status", "UNAVAILABLE_IN_ARTIFACT"),
        "action_type": str(action.get("action_type") or ""),
        "last_action_error": last_error,
    }


def _semantic_evidence(anomalies: list[Mapping[str, Any]], info: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    types = evidence_types(anomalies, info)
    return {
        "types": types,
        "semantic_no_effect": "no_effect" in types,
        "functional_no_effect": "no_effect" in types and str(raw.get("action_type")) not in {"inspect_dom", "inspect_layout", "inspect_network", "inspect_console"},
        "network_failure": "network_failure" in types,
        "layout": "layout" in types,
        "runtime_error": "console_error" in types or "action_error" in types,
        "causality": "UNKNOWN",
    }


def _rule_trace(anomalies: list[Mapping[str, Any]], before: Mapping[str, Any], after: Mapping[str, Any], action: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    emitted: list[dict[str, Any]] = []
    for anomaly in anomalies:
        evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
        rule = str(anomaly.get("rule_id") or evidence.get("rule_id") or "UNKNOWN")
        attribution = _candidate_reward_attribution(anomaly, before, after, action)
        emitted.append({
            "rule_id": rule,
            "issue_type": anomaly.get("issue_type") or anomaly.get("type"),
            "confidence": anomaly.get("confidence"),
            "status": anomaly.get("status"),
            "attribution": attribution,
            "evidence_keys": sorted(str(key) for key in evidence),
        })
    evaluated = sorted({item["rule_id"] for item in emitted if item["rule_id"] in RULES})
    return {
        "rules_evaluated": evaluated,
        "rules_passed": evaluated,
        "predicate_trace_available": False,
        "predicate_failure_reason": "UNAVAILABLE_FOR_NON_EMITTED_RULES" if not emitted else "NOT_EXPOSED_BY_DETECTOR_API",
        "anomalies_emitted": emitted,
        "detector_api_observed_emission_only": True,
    }


def _candidate_gate_trace(anomalies: list[Mapping[str, Any]], before: Mapping[str, Any], after: Mapping[str, Any], action: Mapping[str, Any], history: dict[str, Any], reward_components: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for anomaly in anomalies:
        attribution = _candidate_reward_attribution(anomaly, before, after, action)
        status = str(anomaly.get("status") or "")
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        rows.append({
            "rule_id": anomaly.get("rule_id"),
            "status": status,
            "confidence": confidence,
            "causal_attribution": attribution,
            "causal_eligible": attribution in {"ACTION_CAUSED", "ACTION_REVEALED"},
            "confidence_eligible": confidence > 0.0,
            "dedup_eligible": True,
            "candidate_eligible": status in {"candidate", "confirmed"} and attribution in {"ACTION_CAUSED", "ACTION_REVEALED"},
            "logical_key_available": bool(anomaly.get("signature")),
        })
    return {
        "anomalies": rows,
        "candidate_reward_component": float(reward_components.get("candidate_anomaly", 0.0) or 0.0),
        "confirmation_reward_component": float(reward_components.get("confirmed_anomaly", 0.0) or 0.0),
        "candidate_admission_api": "_score_dqn_anomalies",
        "history_candidate_keys": len(history.get("seen_reward_candidate_keys", set()) or set()),
    }


def _q_metrics(network: Any, state: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    with torch.inference_mode():
        values = factorized_flat_q_values(network(torch.from_numpy(state).float().unsqueeze(0)), torch.from_numpy(mask).bool().unsqueeze(0), SPACE)[0]
    valid = values[torch.from_numpy(mask).bool()]
    return {"q_mean_valid": float(valid.mean()) if len(valid) else None, "q_max_valid": float(valid.max()) if len(valid) else None}


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    ppo_path = ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt"
    dqn_path = ROOT / "artifacts/models/v3/rawd_v3_dqn_productive_context_pilot_seed42.pt"
    ppo = PPOAgent(635, 640)
    ppo.load(ppo_path)
    dqn = FactorizedQNetwork(984, 20, 32)
    dqn.load_state_dict(torch.load(dqn_path, map_location="cpu")["network_state_dict"])
    traces: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    site_funnel: dict[str, Counter] = {site: Counter() for site in SITES}
    action_matrix: dict[str, Counter] = defaultdict(Counter)
    rule_summary: Counter = Counter()
    anomaly_summary: Counter = Counter()
    gate_summary: Counter = Counter()
    total_steps = 0
    for site_name, (site_id, url) in SITES.items():
        for episode in range(5):
            env = _episode_env(site_id, url, 20)
            session_used = False
            try:
                raw, _ = env.reset()
                raw = _enrich(env, raw)
                for episode_step in range(20):
                    state = _v3_state(env, raw)
                    mask = _v3_mask(SPACE, raw)
                    selected = ppo.select_action(state, mask)
                    action_id = int(selected["action_id"])
                    action = {**SPACE.decode(action_id), "action_id": action_id, "failed": False}
                    next_raw, _, done, info, _ = _step_v3(env, action_id, raw, SPACE)
                    next_raw = _enrich(env, next_raw)
                    anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                    context = classify_context(raw, next_raw, anomalies, info)
                    if context.get("admit") and not session_used:
                        context_id = f"{site_name}:ep{episode}:step{episode_step}"
                        context_row = {"context_id": context_id, "site": site_name, "episode": episode, "step": episode_step, "context_tier": context.get("context_tier"), "evidence_types": context.get("evidence_types", [])}
                        contexts.append(context_row)
                        session_used = True
                        local_raw = next_raw
                        history: dict[str, Any] = {}
                        local_state = _dqn_state(env, local_raw, history)
                        session = {"session_id": f"{context_id}:dqn", "site": site_name, "context_id": context_id, "steps": 0, "detector_opportunity_steps": 0, "candidate_created": 0, "candidate_confirmed": 0}
                        for local_step in range(5):
                            local_mask = _v3_mask(SPACE, local_raw)
                            local_action_id = _choose_dqn(dqn, local_state, local_mask, SPACE, epsilon=0.1)
                            local_action = {**SPACE.decode(local_action_id), "action_id": int(local_action_id), "failed": False}
                            after, _, local_done, local_info, _ = _step_v3(env, local_action_id, local_raw, SPACE)
                            after = _enrich(env, after)
                            local_anomalies = detect_anomalies(local_raw, after, {"action": local_action}, site_profile={})
                            reward, components = calculate_dqn_reward(local_raw, after, local_action, local_anomalies, history)
                            _score_dqn_anomalies(local_anomalies, history, local_raw, after, local_action)
                            raw_ev = _raw_evidence(local_raw, after, local_info, local_action)
                            semantic = _semantic_evidence(local_anomalies, local_info, raw_ev)
                            rules = _rule_trace(local_anomalies, local_raw, after, local_action, local_info)
                            gates = _candidate_gate_trace(local_anomalies, local_raw, after, local_action, history, components)
                            trace_id = f"{site_name}:ep{episode}:session{len(sessions)}:local{local_step + 1}"
                            row = {"conversion_trace_id": trace_id, "site": site_name, "episode": episode, "episode_step": episode_step, "local_step": local_step + 1, "action_type": local_action.get("action_type"), "action_id": int(local_action_id), "raw_evidence": raw_ev, "semantic_evidence": semantic, "detector_opportunity": bool(evidence_types(local_anomalies, local_info)), "detector_opportunity_level": "A_OR_B_CONTEXT_SIGNAL", "rule_trace": rules, "anomaly_emitted": bool(local_anomalies), "anomaly_count": len(local_anomalies), "candidate_gate": gates, "reward": float(reward), "reward_components": components, "state_before": semantic_fingerprint(local_raw), "state_after": semantic_fingerprint(after)}
                            traces.append(row)
                            action_matrix[str(local_action.get("action_type") or "UNKNOWN")]["count"] += 1
                            action_matrix[str(local_action.get("action_type") or "UNKNOWN")]["detector_opportunity"] += int(row["detector_opportunity"])
                            for item in rules["anomalies_emitted"]:
                                rule_summary[f"{item['rule_id']}:emitted"] += 1
                                anomaly_summary[str(item.get("issue_type") or "unknown")] += 1
                            for gate in gates["anomalies"]:
                                gate_summary["causal_eligible"] += int(gate["causal_eligible"])
                                gate_summary["candidate_eligible"] += int(gate["candidate_eligible"])
                            session["steps"] += 1
                            session["detector_opportunity_steps"] += int(row["detector_opportunity"])
                            session["candidate_created"] += int(float(components.get("candidate_anomaly", 0.0) or 0.0) > 0)
                            session["candidate_confirmed"] += int(float(components.get("confirmed_anomaly", 0.0) or 0.0) > 0)
                            local_raw, local_state = after, _dqn_state(env, after, history)
                            if local_done:
                                break
                        sessions.append(session)
                    raw = next_raw
                    total_steps += 1
                    if done:
                        break
            finally:
                env.close()
    for context in contexts:
        site_funnel[context["site"]]["contexts"] += 1
    for row in traces:
        site_funnel[row["site"]]["local_steps"] += 1
        site_funnel[row["site"]]["raw_evidence_steps"] += int(bool(row["raw_evidence"].get("no_effect") or row["raw_evidence"].get("action_effect")))
        site_funnel[row["site"]]["semantic_evidence_steps"] += int(bool(row["semantic_evidence"].get("types")))
        site_funnel[row["site"]]["detector_opportunity_steps"] += int(row["detector_opportunity"])
        site_funnel[row["site"]]["anomaly_emitted_steps"] += int(row["anomaly_emitted"])
        site_funnel[row["site"]]["candidate_eligible_anomalies"] += sum(int(g["candidate_eligible"]) for g in row["candidate_gate"]["anomalies"])
    _write(OUT / "evidence_type_distribution.json", {"semantic_types": dict(Counter(t for row in traces for t in row["semantic_evidence"]["types"])), "raw_no_effect": sum(int(row["raw_evidence"]["no_effect"]) for row in traces)})
    _write(OUT / "action_evidence_matrix.json", {key: dict(value) for key, value in action_matrix.items()})
    (OUT / "detector_rule_trace.jsonl").write_text("\n".join(json.dumps(row, default=_json_default) for row in traces) + "\n", encoding="utf-8")
    _write(OUT / "detector_rule_failure_summary.json", {"rule_summary": dict(rule_summary), "predicate_trace_available": False, "note": "The existing detector API exposes emitted standardized anomalies, not per-rule false predicate reasons."})
    _write(OUT / "anomaly_emission_summary.json", {"total_emitted": sum(int(row["anomaly_count"]) for row in traces), "types": dict(anomaly_summary), "steps_with_emission": sum(int(row["anomaly_emitted"]) for row in traces)})
    _write(OUT / "candidate_admission_summary.json", {"gate_summary": dict(gate_summary), "candidate_created": sum(row["candidate_created"] for row in sessions), "candidate_confirmed": sum(row["candidate_confirmed"] for row in sessions), "admission_api": "_score_dqn_anomalies"})
    _write(OUT / "site_conversion_funnel.json", {key: dict(value) for key, value in site_funnel.items()})
    summary = {"version": "RAWD_V3_EVIDENCE_CANDIDATE_CONVERSION_DIAGNOSTIC", "diagnostic_only": True, "sites": list(SITES), "episodes_per_site": 5, "total_ppo_steps": total_steps, "local_sessions": len(sessions), "local_steps": len(traces), "productive_contexts": len(contexts), "raw_evidence_steps": sum(int(bool(row["raw_evidence"].get("no_effect") or row["raw_evidence"].get("action_effect"))) for row in traces), "semantic_evidence_steps": sum(int(bool(row["semantic_evidence"].get("types"))) for row in traces), "detector_opportunity_steps": sum(int(row["detector_opportunity"]) for row in traces), "rule_passed_emissions": sum(len(row["rule_trace"]["rules_passed"]) for row in traces), "anomaly_emitted": sum(int(row["anomaly_count"]) for row in traces), "causal_eligible": sum(sum(int(g["causal_eligible"]) for g in row["candidate_gate"]["anomalies"]) for row in traces), "candidate_eligible": sum(sum(int(g["candidate_eligible"]) for g in row["candidate_gate"]["anomalies"]) for row in traces), "candidate_created": sum(row["candidate_created"] for row in sessions), "candidate_confirmed": sum(row["candidate_confirmed"] for row in sessions), "site_funnel": {key: dict(value) for key, value in site_funnel.items()}, "production_algorithm_changed": False, "gt_used": False, "site002_to_site005_used": False, "predicate_trace_available": False, "classification_basis": "Existing detector API exposes anomaly emissions and candidate reward admission; per-rule false predicate reasons are unavailable without adding detector instrumentation."}
    _write(OUT / "rawd_v3_evidence_candidate_conversion_summary.json", summary)
    (OUT / "local_sessions.jsonl").write_text("\n".join(json.dumps(row, default=_json_default) for row in sessions) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
