"""Run an isolated forced-reproduction credit diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from dqn_error_discovery.agents.confirmation_credit_diagnostics import (
    aggregate_context_rows,
    classify_forced_probe,
    evidence_reproduced,
    evidence_snapshot,
    feature_group_ablation,
    flip_direction,
    input_weight_norms,
    resolve_expected_action,
    reward_credit_summary,
)
from dqn_error_discovery.agents.confirmation_observation import augment_dqn_observation, build_dqn_confirmation_context
from dqn_error_discovery.agents.dqn_agent import FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FactorizedQNetwork, factorized_q_value_diagnostics, select_factorized_greedy_action
from dqn_error_discovery.runners.run_policy_sanity_check import _load_site_config, _sanitize_anomalies
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward, initialize_dqn_candidate_reward_baseline, safe_action_mask
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


def _state(env, raw, history):
    base = env.observation_encoder.encode_observation(raw)
    context, metadata = build_dqn_confirmation_context(history, raw, env.action_space)
    return augment_dqn_observation(base, context), metadata


def _positive(components, key):
    return float(components.get(key, 0.0) or 0.0) > 0.0


def _load_network(env, model_path: str) -> FactorizedQNetwork:
    network = FactorizedQNetwork(env.observation_encoder.get_obs_dim() + 72, len(env.action_space.action_types), env.action_space.max_candidates)
    payload = torch.load(Path(model_path), map_location="cpu")
    network.load_state_dict(payload.get("model", payload))
    network.eval()
    return network


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    env = BrowserGymJAWSEnv(site_id=str(site["site_id"]), base_url=str(site["base_url"]), max_steps=args.max_steps, max_candidates=args.max_candidates, headless=args.headless)
    network = _load_network(env, args.model_path)
    rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    greedy_full_matches: list[dict[str, Any]] = []
    normal_action_count = 0
    try:
        for episode in range(1, args.episodes + 1):
            raw, _ = env.reset(); history: dict[str, Any] = {}
            initial = _sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}}))
            initialize_dqn_candidate_reward_baseline(history, initial)
            done = False
            for step in range(1, args.max_steps + 1):
                state, metadata = _state(env, raw, history)
                mask = safe_action_mask(env.action_space, raw)
                expected = resolve_expected_action(metadata, env.action_space, mask)
                opportunity = bool(metadata.get("pending_confirmation_exists")) and bool(metadata.get("expected_action_type")) and (not bool(metadata.get("expected_action_requires_target")) or bool(metadata.get("expected_target_visible")))
                greedy_id = select_factorized_greedy_action(network, state, mask, env.action_space)
                greedy = env.action_space.decode(greedy_id)
                if opportunity:
                    action_id = expected["expected_action_id"] if expected["forced_action_executable"] else greedy_id
                    forced = bool(expected["forced_action_executable"])
                else:
                    action_id = greedy_id; forced = False; normal_action_count += 1
                selected = env.action_space.decode(int(action_id)); before_evidence = evidence_snapshot(_sanitize_anomalies(detect_anomalies(raw, raw, {"action": selected})))
                before_progress = float(metadata.get("confirmation_progress", 0.0) or 0.0)
                before_attempts = float(metadata.get("reproduction_attempt_count", 0.0) or 0.0)
                before_status = [row["status"] for row in before_evidence]
                next_raw, _, done, info = env.step(int(action_id))
                selected["failed"] = bool(info.get("last_action_error")); selected["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = _sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": selected}))
                reward, components = calculate_dqn_reward(raw, next_raw, selected, anomalies, history)
                repeat = dict(history.get("last_repeat_diagnostic", {}))
                next_state, next_metadata = _state(env, next_raw, history)
                after_evidence = evidence_snapshot(anomalies)
                after_progress = float(next_metadata.get("confirmation_progress", 0.0) or 0.0)
                after_status = [row["status"] for row in after_evidence]
                confirmed = _positive(components, "confirmed_anomaly") or any(row["status"] == "confirmed" for row in after_evidence)
                evidence = evidence_reproduced(before_evidence, after_evidence, metadata)
                if opportunity:
                    row = {
                        "episode": episode, "step": step, "rule_id": metadata.get("active_rule_id"), "pending_logical_candidate_key": metadata.get("active_pending_key"),
                        "expected_action_type": metadata.get("expected_action_type"), "expected_target_identity": metadata.get("expected_target_identity"),
                        "actual_action_type": selected.get("action_type"), "actual_target_identity": metadata.get("expected_target_identity") if forced else "",
                        "forced": forced, "expected_action_defined": expected["expected_action_defined"], "expected_target_required": expected["expected_target_required"], "expected_target_visible": expected["expected_target_visible"],
                        "expected_target_slot": expected["expected_target_slot"], "expected_action_mask_valid": expected["expected_action_mask_valid"], "target_semantics_valid": expected["target_semantics_valid"], "forced_action_executable": expected["forced_action_executable"],
                        "type_match": selected.get("action_type") == metadata.get("expected_action_type"), "target_match": (not expected["expected_target_required"] or selected.get("candidate_index") in (metadata.get("matching_candidate_slots") or [])),
                        "full_match": forced, "confirmation_progress_before": before_progress, "confirmation_progress_after": after_progress > before_progress or bool(repeat.get("confirmation_progress")),
                        "candidate_status_before": before_status, "candidate_status_after": after_status, "confirmation_evidence_before": before_evidence, "confirmation_evidence_after": after_evidence,
                        "reproduction_attempt_count_before": before_attempts, "reproduction_attempt_count_after": float(next_metadata.get("reproduction_attempt_count", 0.0) or 0.0),
                        "total_reward": float(reward), "reward_components": dict(components), "no_effect": bool(float(components.get("no_effect_penalty", 0.0) or 0.0) < 0),
                        "repeat": bool(repeat.get("immediate_repeat")), "recent_repeat": bool(repeat.get("recent_window_repeat")), "cycle_repeat": bool(repeat.get("cycle_repeat")), "useful_reproduction": bool(repeat.get("useful_reproduction_repeat")), "unproductive_repeat": bool(repeat.get("unproductive_repeat")),
                        "candidate_reward": float(components.get("candidate_anomaly", 0.0) or 0.0), "confirmed_reward": float(components.get("confirmed_anomaly", 0.0) or 0.0), "candidate_to_confirmed_reward": float(components.get("confirmation_upgrade", 0.0) or 0.0), "repeat_penalty": float(components.get("repeat_action_penalty", 0.0) or 0.0), "no_effect_penalty": float(components.get("no_effect_penalty", 0.0) or 0.0),
                        "evidence_reproduced": evidence, "confirmed_after": confirmed, "classification": classify_forced_probe(executable=bool(expected["forced_action_executable"]), evidence=evidence, progress=bool(after_progress > before_progress or repeat.get("confirmation_progress")), confirmed=confirmed),
                    }
                    rows.append(row)
                    if not forced and row["full_match"]: greedy_full_matches.append(row)
                    ablations.append(feature_group_ablation(network, state, mask, env.action_space, metadata))
                    with torch.inference_mode():
                        # Reuse the existing real/zero computation indirectly through the public Q outputs.
                        real = network(torch.from_numpy(state).float().unsqueeze(0))
                        zero_state = state.copy(); zero_state[635:] = 0.0
                        zero = network(torch.from_numpy(zero_state).float().unsqueeze(0))
                        real_types = real["type_values"].squeeze(0).numpy(); zero_types = zero["type_values"].squeeze(0).numpy()
                        expected_index = env.action_space.action_types.index(str(metadata.get("expected_action_type")))
                        real_id = int(select_factorized_greedy_action(network, state, mask, env.action_space)); zero_id = int(select_factorized_greedy_action(network, zero_state, mask, env.action_space))
                        valid_type_values = [(i, float(real_types[i])) for i in range(len(real_types)) if any(mask[env.action_space.encode(t, s)] > 0 for s in range(env.action_space.max_candidates) for t in [env.action_space.action_types[i]])]
                        valid_zero_values = [(i, float(zero_types[i])) for i in range(len(zero_types)) if any(mask[env.action_space.encode(t, s)] > 0 for s in range(env.action_space.max_candidates) for t in [env.action_space.action_types[i]])]
                        real_rank = 1 + [i for i, _ in sorted(valid_type_values, key=lambda x: x[1], reverse=True)].index(expected_index) if expected_index in [i for i, _ in valid_type_values] else None
                        zero_rank = 1 + [i for i, _ in sorted(valid_zero_values, key=lambda x: x[1], reverse=True)].index(expected_index) if expected_index in [i for i, _ in valid_zero_values] else None
                    real_type = env.action_space.decode(real_id)["action_type"]; zero_type = env.action_space.decode(zero_id)["action_type"]
                    context_rows.append({"rule_id": metadata.get("active_rule_id"), "expected_type_context_delta": float(real_types[expected_index] - zero_types[expected_index]), "expected_type_rank_real": real_rank, "expected_type_rank_zero": zero_rank, "flip_direction": flip_direction(zero_type, real_type, str(metadata.get("expected_action_type")))})
                raw = next_raw
                if done: break
    finally:
        env.close()
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_rule[str(row.get("rule_id") or "UNKNOWN")].append(row)
    report = {
        "model_path": args.model_path, "site_id": str(site["site_id"]), "episodes": args.episodes, "max_steps": args.max_steps, "seed": args.seed,
        "ground_truth_used": False, "production_policy_changed": False, "replay_written": False, "optimizer_called": False, "checkpoint_written": False,
        "existing_training_full_matches": {"count": 3, "detail_available": False, "source": "ep30 manifest confirmation_training_summary; transition rows were not persisted"},
        "forced_reproduction_probe": rows,
        "forced_probe_funnel": {"opportunities": len(rows), "resolvable": sum(bool(row["expected_action_defined"]) for row in rows), "safe_mask_valid": sum(bool(row["expected_action_mask_valid"]) for row in rows), "forced_actions_executed": sum(bool(row["forced"]) for row in rows), "evidence_reproduced": sum(bool(row["forced"]) and bool(row["evidence_reproduced"]) for row in rows), "confirmation_progress": sum(bool(row["forced"]) and bool(row["confirmation_progress_after"]) for row in rows), "confirmed": sum(bool(row["forced"]) and bool(row["confirmed_after"]) for row in rows)},
        "forced_probe_by_rule": {rule: {"opportunities": len(values), "executable": sum(bool(x["forced"]) for x in values), "evidence_reproduced": sum(bool(x["forced"]) and bool(x["evidence_reproduced"]) for x in values), "progress": sum(bool(x["forced"]) and bool(x["confirmation_progress_after"]) for x in values), "confirmed": sum(bool(x["forced"]) and bool(x["confirmed_after"]) for x in values)} for rule, values in sorted(by_rule.items())},
        "reward_credit_analysis": reward_credit_summary(rows),
        "context_delta_by_rule": {rule: aggregate_context_rows([row for row, probe in zip(context_rows, rows) if probe.get("rule_id") == rule]) for rule in sorted(by_rule)},
        "context_delta_overall": aggregate_context_rows(context_rows),
        "feature_group_ablation": ablations,
        "input_weight_norms": input_weight_norms(network),
        "learning_experience_coverage": {"training_opportunities": 434, "expected_type_experience_rate": 52 / 434, "full_reproduction_experience_rate": 3 / 434, "confirmation_progress_experience_rate": 0.0},
        "root_cause_summary": {"forced_probe_classifications": dict(__import__("collections").Counter(row["classification"] for row in rows)), "normal_actions_outside_opportunities": normal_action_count},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json"); parser.add_argument("--site-id", default="site001"); parser.add_argument("--model-path", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=2); parser.add_argument("--max-steps", type=int, default=20); parser.add_argument("--max-candidates", type=int, default=32); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--base-url", default=None); parser.add_argument("--headless", action="store_true", default=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
