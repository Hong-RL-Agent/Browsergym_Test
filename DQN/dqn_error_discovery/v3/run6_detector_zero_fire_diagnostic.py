"""GT-blind Observation -> Evidence -> Detector funnel diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.run5_context_aware_dqn_training import (
    CANDIDATE_B_PATHS, _step_v3, _v3_mask, _v3_state, discover_sites, reset_env, read_json,
)
from services.anomaly_detection_service import detect_anomalies
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=OUT)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-sites", type=int, default=10)
    parser.add_argument("--episodes-per-site", type=int, default=3)
    args = parser.parse_args()
    out = args.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    ppo = PPOAgent(635, 640)
    ppo.load(CANDIDATE_B_PATHS["ppo"])
    sites = manifest_order(discover_sites(args.source_root), args.run_root)[:args.max_sites]
    rows: list[dict[str, Any]] = []
    site_rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"episodes": 0, "global_steps": 0, "interactive_targets_exposed": 0, "click_input": 0, "effectful_actions": 0, "detector_invocations": 0, "rule_evaluations": 0, "eligible_evidence": 0, "fires": 0})
    actions = Counter(); funnel = Counter(); network = Counter(); ui = Counter(); pairing = Counter(); rules = defaultdict(Counter); predicates = Counter(); errors = []
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
                        selected = ppo.select_action(_v3_state(env, raw), _v3_mask(SPACE, raw))
                        aid = int(selected["action_id"]); action = {**SPACE.decode(aid), "action_id": aid, "failed": False}
                        before = raw
                        after, _, done, info, _ = _step_v3(env, aid, raw, SPACE)
                        action["failed"] = bool(info.get("last_action_error"))
                        anomalies = detect_anomalies(before, after, {"action": action, "step_id": global_step + 1, "agent_mode": "PPO_GLOBAL"}, site_profile={})
                        raw = after; global_step += 1; site_rows[sid]["global_steps"] += 1; site_rows[sid]["detector_invocations"] += 1
                        action_type = str(action.get("action_type") or "other"); actions[action_type] += 1
                        if action_type in {"click_element", "input_text", "type_text", "select_option"}: funnel["interactive_targets_exposed"] += int(bool(before.get("candidate_elements"))); funnel["click_input"] += 1
                        effectful = int(not action["failed"] and (before.get("url") != after.get("url") or before.get("page_text") != after.get("page_text") or before.get("candidate_elements") != after.get("candidate_elements") or before.get("runtime_signals") != after.get("runtime_signals")))
                        funnel["effectful_actions"] += effectful; site_rows[sid]["effectful_actions"] += effectful
                        if isinstance(after.get("network_entries"), list): network["requests_or_entries"] += len(after["network_entries"])
                        network["request_observed"] += int(bool(after.get("network_entries") or after.get("network_activity") or after.get("network_request_count")))
                        network["response_observed"] += int(bool(after.get("network_responses") or after.get("api_responses") or after.get("last_api_response")))
                        network["body_captured"] += int(bool(after.get("response_body") or after.get("network_response_body") or after.get("network_bodies")))
                        network["json_parsed"] += int(count_values(after, {"json", "json_body", "parsed_json"}) > 0)
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
    summary = {"episodes": episodes, "global_steps": global_step, "actions": dict(actions), "interactive_targets_exposed": funnel["interactive_targets_exposed"], "click_input_actions": funnel["click_input"], "effectful_actions": funnel["effectful_actions"], "network": dict(network), "ui": dict(ui), "pairing": dict(pairing), "detector_invocations": global_step, "rule_evaluations": funnel["rule_evaluations"], "eligible_evidence": funnel["eligible_evidence"], "fires": funnel["fires"], "candidate_admissions": 0, "errors": errors, "gt_runtime_access": False, "optimizer_updates": 0, "backward": 0, "dqn_retrained": False, "ppo_retrained": False}
    write_json(out / "summary.json", summary); write_json(out / "site_funnel.json", dict(site_rows)); write_json(out / "rule_funnel.json", {k: dict(v) for k, v in rules.items()}); write_json(out / "evidence_funnel.json", {"network": dict(network), "ui": dict(ui), "pairing": dict(pairing), "r10_predicates": dict(predicates)}); write_json(out / "runtime_steps.jsonl", rows)
    print(json.dumps({"episodes": episodes, "global_steps": global_step, "detector_fires": funnel["fires"], "errors": len(errors), "actions": dict(actions), "network": dict(network), "pairing": dict(pairing)}, indent=2))


if __name__ == "__main__": main()
