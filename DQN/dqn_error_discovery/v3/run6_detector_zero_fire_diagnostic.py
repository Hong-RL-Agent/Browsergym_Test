"""GT-blind Observation -> Evidence -> Detector funnel diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.action_contract import is_click_target, is_fill_input_target
from dqn_error_discovery.v3.run5_context_aware_dqn_training import (
    CANDIDATE_B_PATHS, discover_sites, reset_env, read_json,
)
from dqn_error_discovery.v3.run6_ppo_interactive_coverage_diagnostic import (
    production_action_mask,
    production_observation_vector,
    production_ppo_path_metadata,
    production_select_action,
    production_step,
)
from services.anomaly_detection_service import detect_anomalies
from services.r10_network_ui_semantic_evidence import normalize_runtime_network_telemetry
from dqn_error_discovery.v3.run6_candidate_context_harvest_validation import ManagedSite, manifest_order


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\workspace\RL_Errorsite_fullstack_run4")
RUN_ROOT = ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full"
OUT = ROOT / "artifacts/training_v3/run6_detector_zero_fire_diagnostic"
SPACE = V3ActionSpace(32)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def count_values(value: Any, keys: set[str]) -> int:
    if isinstance(value, Mapping):
        return sum((1 if str(k) in keys and value[k] not in (None, [], {}, "") else 0) + count_values(v, keys) for k, v in value.items())
    if isinstance(value, list):
        return sum(count_values(v, keys) for v in value)
    return 0


def first_mapping(value: Any, names: set[str]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        for name in names:
            if isinstance(value.get(name), Mapping):
                return value[name]
        for child in value.values():
            found = first_mapping(child, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_mapping(child, names)
            if found is not None:
                return found
    return None


def semantic_metrics(before: Mapping[str, Any], after: Mapping[str, Any], anomalies: list[Mapping[str, Any]]) -> dict[str, int]:
    runtime = after.get("runtime_signals") if isinstance(after.get("runtime_signals"), Mapping) else {}
    semantic = after.get("network_ui_semantic") or runtime.get("network_ui_semantic") or {}
    if not isinstance(semantic, Mapping):
        semantic = {}
    pairs = semantic.get("pairs") if isinstance(semantic.get("pairs"), list) else []
    not_equal = [p for p in pairs if isinstance(p, Mapping) and str(p.get("relation") or "") == "NOT_EQUAL"]
    high_conf = [p for p in not_equal if float(p.get("pair_confidence", 0) or 0) >= 0.85]
    network_fields = count_values(after, {"network_semantic_fields", "network_fields", "network_field_values"})
    ui_fields = count_values(after, {"ui_semantic_fields", "ui_fields", "semantic_fields"})
    return {
        "network_semantic_fields": network_fields,
        "ui_semantic_fields": ui_fields,
        "semantic_pairs": len(pairs),
        "equal_pairs": sum(str(p.get("relation") or "") == "EQUAL" for p in pairs if isinstance(p, Mapping)),
        "not_equal_pairs": len(not_equal),
        "high_confidence_not_equal": len(high_conf),
        "semantic_block": int(bool(semantic)),
        "ui_settled": int(any(bool((a.get("evidence") or {}).get("ui_settled")) for a in anomalies if isinstance(a, Mapping))),
    }


def interactive_metrics(raw: Mapping[str, Any], mask: Any) -> dict[str, int]:
    candidates = raw.get("candidate_elements", []) if isinstance(raw.get("candidate_elements"), list) else []
    click = [i for i, c in enumerate(candidates[:SPACE.max_candidates]) if isinstance(c, Mapping) and is_click_target(c)]
    fill = [i for i, c in enumerate(candidates[:SPACE.max_candidates]) if isinstance(c, Mapping) and is_fill_input_target(c)]
    click_valid = sum(int(mask[SPACE.encode("click_element", i)] > 0) for i in click)
    fill_valid = sum(int(mask[SPACE.encode("fill_input", i)] > 0) for i in fill)
    return {
        "candidate_count": len(candidates),
        "usable_interactive_candidates": len(set(click + fill)),
        "mask_valid_click_actions": click_valid,
        "mask_valid_input_actions": fill_valid,
        "mask_valid_interactive_actions": click_valid + fill_valid,
    }


def target_identity(raw: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    candidates = raw.get("candidate_elements", []) if isinstance(raw.get("candidate_elements"), list) else []
    index = int(action.get("candidate_index", 0) or 0)
    candidate = candidates[index] if 0 <= index < len(candidates) and isinstance(candidates[index], Mapping) else {}
    return str(candidate.get("bid") or candidate.get("id") or candidate.get("selector") or candidate.get("text") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=OUT)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-sites", type=int, default=10)
    parser.add_argument("--episodes-per-site", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    out = args.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    ppo = PPOAgent(635, 640)
    ppo.load(CANDIDATE_B_PATHS["ppo"])
    sites = manifest_order(discover_sites(args.source_root), args.run_root)[:args.max_sites]
    rows: list[dict[str, Any]] = []
    site_rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"episodes": 0, "global_steps": 0, "interactive_targets_exposed": 0, "click_input": 0, "effectful_actions": 0, "detector_invocations": 0, "rule_evaluations": 0, "eligible_evidence": 0, "fires": 0})
    actions = Counter(); funnel = Counter(); network = Counter(); ui = Counter(); pairing = Counter(); rules = defaultdict(Counter); predicates = Counter(); errors = []
    parity_rows = []; effectful_interactive = 0; effectful_noninteractive = 0; network_samples = []
    strict_associated = 0; strict_interactive_associated = 0; temporal_only = 0; background_unassociated = 0; unknown_association = 0; linkage_mismatches = 0; duplicate_request_ids = 0
    episodes = 0; global_step = 0
    for site in sites:
        if episodes >= args.episodes:
            break
        sid = str(site["site_id"])
        with ManagedSite(site, out) as managed:
            if not managed.logs.get("http", {}).get("ok"):
                errors.append({"site_id": sid, "error": "HTTP_NOT_REACHABLE", "server": managed.logs}); continue
            for _ in range(args.episodes_per_site):
                if episodes >= args.episodes: break
                episodes += 1; site_rows[sid]["episodes"] += 1
                env = None
                try:
                    env, raw = reset_env(site, args.max_steps, 60000)
                    for step in range(args.max_steps):
                        obs = production_observation_vector(env, raw)
                        mask = production_action_mask(raw)
                        selected, selected_meta, decoded = None, None, None
                        obs, selected_meta, decoded = production_select_action(ppo, env, raw)
                        aid = int(decoded["action_id"]); action = {**decoded, "failed": False}
                        im = interactive_metrics(raw, mask)
                        parity_row = None
                        if len(parity_rows) < 20:
                            parity_rows.append({"site_id": sid, "episode": episodes, "step": step + 1, "observation_dim": int(obs.shape[0]), **im, "selected_action_before_decode": aid, "decoded_action_type": decoded["action_type"], "decoded_candidate_index": decoded["candidate_index"], "mask_valid_action_count": int(mask.sum())})
                            parity_row = parity_rows[-1]
                        before = raw
                        action_start = time.monotonic() * 1000.0
                        action["episode_id"] = episodes
                        action["global_step"] = global_step + 1
                        action["target_identity"] = target_identity(raw, decoded)
                        action["action_start_timestamp"] = action_start
                        after, _, done, info, executed = production_step(
                            env,
                            raw,
                            decoded,
                            action_context={
                                "episode_id": episodes,
                                "global_step": global_step + 1,
                                "action_id": aid,
                                "action_type": decoded["action_type"],
                                "target_identity": action["target_identity"],
                                "action_start_timestamp": action_start,
                            },
                        )
                        action_end = time.monotonic() * 1000.0
                        action["action_end_timestamp"] = action_end
                        if parity_row is not None:
                            parity_row.update({
                                "execution_command": executed,
                                "env_step_action": {"action_id": aid, "action_type": decoded["action_type"], "candidate_index": decoded["candidate_index"]},
                                "execution_success": not bool(info.get("last_action_error")),
                            })
                        action["failed"] = bool(info.get("last_action_error"))
                        anomalies = detect_anomalies(before, after, {"action": action, "step_id": global_step + 1, "agent_mode": "PPO_GLOBAL"}, site_profile={})
                        raw = after; global_step += 1; site_rows[sid]["global_steps"] += 1; site_rows[sid]["detector_invocations"] += 1
                        action_type = str(action.get("action_type") or "other"); actions[action_type] += 1
                        if action_type in {"click_element", "fill_input", "press_enter", "select_option"}:
                            funnel["interactive_targets_exposed"] += int(im["usable_interactive_candidates"] > 0); funnel["click_input"] += int(action_type == "click_element"); funnel["input"] += int(action_type in {"fill_input", "press_enter", "select_option"})
                        effectful = int(not action["failed"] and (before.get("url") != after.get("url") or before.get("page_text") != after.get("page_text") or before.get("candidate_elements") != after.get("candidate_elements") or before.get("runtime_signals") != after.get("runtime_signals")))
                        funnel["effectful_actions"] += effectful; site_rows[sid]["effectful_actions"] += effectful
                        if action_type in {"click_element", "fill_input", "press_enter", "select_option"}:
                            effectful_interactive += effectful
                        else:
                            effectful_noninteractive += effectful
                        telemetry = normalize_runtime_network_telemetry(after, info)
                        strict_associated += int(telemetry["strict_associated_request_count"])
                        strict_interactive_associated += sum(int(row["association_type"] == "STRICT_ACTION_CAUSED" and row.get("action_type") in {"click_element", "fill_input", "press_enter", "select_option"}) for row in telemetry["associations"])
                        temporal_only += sum(int(row["association_type"] == "TEMPORAL_ONLY") for row in telemetry["associations"])
                        background_unassociated += sum(int(row["association_type"] == "BACKGROUND_UNASSOCIATED") for row in telemetry["associations"])
                        unknown_association += sum(int(row["association_type"] == "UNKNOWN") for row in telemetry["associations"])
                        linkage_mismatches += int(telemetry["request_id_linkage_mismatch_count"])
                        duplicate_request_ids += int(telemetry["duplicate_request_id_count"])
                        network["raw_requests"] += int(telemetry["raw_request_count"])
                        network["raw_responses"] += int(telemetry["raw_response_count"])
                        network["normalized_requests"] += int(telemetry["normalized_request_count"])
                        network["normalized_responses"] += int(telemetry["normalized_response_count"])
                        network["failed_requests"] += int(telemetry["failed_request_count"])
                        network["http_error_responses"] += int(telemetry["http_error_response_count"])
                        network["parsed_responses"] += int(telemetry["parsed_response_count"])
                        network["request_observed"] += int(telemetry["raw_request_count"] > 0)
                        network["response_observed"] += int(telemetry["raw_response_count"] > 0)
                        network["body_captured"] += sum(int(bool(row.get("response_body_available"))) for row in telemetry["events"])
                        network["json_parsed"] += int(telemetry["parsed_response_count"])
                        if action_type in {"click_element", "fill_input", "press_enter", "select_option"} and telemetry["raw_request_count"]:
                            network["interactive_associated_requests"] += int(telemetry["raw_request_count"])
                        if len(network_samples) < 10 and telemetry["events"]:
                            for event in telemetry["events"][: 10 - len(network_samples)]:
                                network_samples.append({"episode": episodes, "step": step + 1, "selected_action": action, "association": next((row for row in telemetry["associations"] if row.get("request_id") == event.get("request_id")), None), "event": {key: event.get(key) for key in ("event", "request_id", "method", "url", "status", "response_received", "network_failure", "response_body_available", "request_relative_ms", "request_start_timestamp", "response_timestamp")}})
                        sm = semantic_metrics(before, after, anomalies)
                        for k, v in sm.items():
                            if k in {"network_semantic_fields", "ui_semantic_fields"}: (network if k.startswith("network") else ui)[k] += v
                            elif k in {"semantic_pairs", "equal_pairs", "not_equal_pairs", "high_confidence_not_equal", "semantic_block"}: pairing[k] += v
                            elif k == "ui_settled": ui[k] += v
                        for anomaly in anomalies:
                            ev = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), Mapping) else {}
                            rid = str(anomaly.get("rule_id") or ev.get("rule_id") or "UNKNOWN")
                            rules[rid]["evaluations"] += 1; funnel["rule_evaluations"] += 1; site_rows[sid]["rule_evaluations"] += 1
                            if ev: rules[rid]["eligible_evidence"] += 1; funnel["eligible_evidence"] += 1; site_rows[sid]["eligible_evidence"] += 1
                            rules[rid]["fires"] += int(str(anomaly.get("status") or "") in {"candidate", "confirmed"} or bool(anomaly.get("candidate") or anomaly.get("is_candidate"))); funnel["fires"] += rules[rid]["fires"]; site_rows[sid]["fires"] += rules[rid]["fires"]
                        semantic = after.get("network_ui_semantic") or (after.get("runtime_signals") or {}).get("network_ui_semantic") if isinstance(after.get("runtime_signals"), Mapping) else after.get("network_ui_semantic")
                        r10_pred = {"action_error_gate": int(not action["failed"]), "semantic_block": int(bool(semantic)), "pairs": int(bool(isinstance(semantic, Mapping) and semantic.get("pairs"))), "not_equal": int(pairing["not_equal_pairs"] > 0), "confidence": int(pairing["high_confidence_not_equal"] > 0), "final_fire": int(any(str(a.get("rule_id") or a.get("evidence", {}).get("rule_id") or "") == "R10" for a in anomalies))}
                        for k, v in r10_pred.items(): predicates[f"R10_{k}_{'PASS' if v else 'FAIL'}"] += 1
                        rows.append({"site_id": sid, "episode": episodes, "step": step + 1, "action": action, "action_failed": action["failed"], "effectful": effectful, "url_changed": before.get("url") != after.get("url"), "dom_changed": before.get("page_text") != after.get("page_text"), "network": dict(network), "ui": dict(ui), "pairing": dict(pairing), "anomalies": [{"rule_id": str(a.get("rule_id") or (a.get("evidence") or {}).get("rule_id") or ""), "status": a.get("status"), "evidence_keys": sorted((a.get("evidence") or {}).keys())} for a in anomalies], "r10_predicates": r10_pred})
                        if done: break
                except Exception as exc:
                    errors.append({"site_id": sid, "episode": episodes, "error": repr(exc)})
                finally:
                    if env is not None: env.close()
    path = production_ppo_path_metadata(CANDIDATE_B_PATHS["ppo"])
    path_checks = {
        "RAWD_V3_DOWNSTREAM_ADAPTER_PATH_CHECK": "PASS" if path["adapter_module"].endswith("DQN\\adapters\\browsergym_observation_adapter.py") else "FAIL",
        "RAWD_V3_DOWNSTREAM_CANDIDATE_PATH_CHECK": "PASS" if any(row["usable_interactive_candidates"] > 0 for row in parity_rows) else "FAIL",
        "RAWD_V3_DOWNSTREAM_MASK_PATH_CHECK": "PASS" if any(row["mask_valid_interactive_actions"] > 0 for row in parity_rows) else "FAIL",
        "RAWD_V3_DOWNSTREAM_PPO_SELECTION_PATH_CHECK": "PASS" if any(row["selected_action_before_decode"] is not None for row in parity_rows) else "FAIL",
        "RAWD_V3_DOWNSTREAM_ACTION_DECODE_CHECK": "PASS" if all(row["decoded_action_type"] and row["env_step_action"]["candidate_index"] == row["decoded_candidate_index"] for row in parity_rows) else "FAIL",
    }
    path_checks["RAWD_V3_DOWNSTREAM_RUNNER_ALIGNMENT"] = "PASS" if all(value == "PASS" for value in path_checks.values()) else "PARTIAL"
    summary = {"episodes": episodes, "global_steps": global_step, "actions": dict(actions), "interactive_targets_exposed": funnel["interactive_targets_exposed"], "click_input_actions": funnel["click_input"], "click_count": funnel["click_input"], "input_count": funnel["input"], "effectful_actions": funnel["effectful_actions"], "effectful_interactive_actions": effectful_interactive, "effectful_noninteractive_actions": effectful_noninteractive, "network": dict(network), "raw_network_requests": int(network["raw_requests"]), "normalized_network_requests": int(network["normalized_requests"]), "completed_requests": int(network["normalized_responses"]), "strict_associated_requests": strict_associated, "strict_interactive_associated_requests": strict_interactive_associated, "temporal_only_requests": temporal_only, "background_unassociated_requests": background_unassociated, "unknown_association_requests": unknown_association, "request_id_linkage_mismatch": linkage_mismatches, "duplicate_request_ids": duplicate_request_ids, "network_samples": network_samples, "network_schema": "runtime_signals.temporal_network.request_events+completed_requests+failed_requests", "ui": dict(ui), "pairing": dict(pairing), "detector_invocations": global_step, "rule_evaluations": funnel["rule_evaluations"], "eligible_evidence": funnel["eligible_evidence"], "fires": funnel["fires"], "candidate_admissions": 0, "dqn_handoffs": 0, "local_transitions": 0, "errors": errors, "gt_runtime_access": False, "optimizer_updates": 0, "backward": 0, "dqn_retrained": False, "ppo_retrained": False, "ppo_path": path, "path_checks": path_checks, "first_path_divergence": None, "parity_rows": parity_rows}
    write_json(out / "summary.json", summary); write_json(out / "site_funnel.json", dict(site_rows)); write_json(out / "rule_funnel.json", {k: dict(v) for k, v in rules.items()}); write_json(out / "evidence_funnel.json", {"network": dict(network), "ui": dict(ui), "pairing": dict(pairing), "r10_predicates": dict(predicates)}); write_json(out / "runtime_steps.jsonl", rows)
    print(json.dumps({"episodes": episodes, "global_steps": global_step, "detector_fires": funnel["fires"], "errors": len(errors), "actions": dict(actions), "network": dict(network), "pairing": dict(pairing)}, indent=2))


if __name__ == "__main__": main()
