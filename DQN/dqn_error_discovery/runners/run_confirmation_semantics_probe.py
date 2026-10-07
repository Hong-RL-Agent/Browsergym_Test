"""Runtime probe for confirmation instruction semantics, without training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.confirmation_observation import augment_dqn_observation, build_dqn_confirmation_context
from dqn_error_discovery.agents.confirmation_reproduction import ACTION_REPRODUCTION, OBSERVATION_REPRODUCTION
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, select_factorized_greedy_action
from dqn_error_discovery.runners.run_policy_sanity_check import _load_site_config, _sanitize_anomalies
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward, initialize_dqn_candidate_reward_baseline, safe_action_mask
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


def _state(env, raw, history):
    base = env.observation_encoder.encode_observation(raw)
    context, metadata = build_dqn_confirmation_context(history, raw, env.action_space)
    return augment_dqn_observation(base, context), metadata


def _load_network(env, path):
    network = FactorizedQNetwork(env.observation_encoder.get_obs_dim() + 72, len(env.action_space.action_types), env.action_space.max_candidates)
    payload = torch.load(Path(path), map_location="cpu")
    network.load_state_dict(payload.get("model", payload))
    network.eval()
    return network


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    env = BrowserGymJAWSEnv(site_id=str(site["site_id"]), base_url=str(site["base_url"]), max_steps=args.max_steps, max_candidates=args.max_candidates, headless=args.headless)
    network = _load_network(env, args.model_path)
    counts = Counter()
    rules: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(1, args.episodes + 1):
            raw, _ = env.reset()
            history: dict[str, Any] = {}
            initial = _sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}}))
            initialize_dqn_candidate_reward_baseline(history, initial)
            done = False
            for step in range(1, args.max_steps + 1):
                state, metadata = _state(env, raw, history)
                mask = safe_action_mask(env.action_space, raw)
                mode = str(metadata.get("confirmation_mode") or "")
                pending = bool(metadata.get("pending_confirmation_exists"))
                action_opportunity = bool(mode == ACTION_REPRODUCTION and metadata.get("expected_action_type") and (not metadata.get("expected_action_requires_target") or metadata.get("expected_target_visible")))
                observation_pending = bool(mode == OBSERVATION_REPRODUCTION and pending)
                mask_valid = False
                if pending:
                    counts["pending_candidates"] += 1
                if mode == ACTION_REPRODUCTION:
                    counts["action_reproduction_pending"] += 1
                if mode == OBSERVATION_REPRODUCTION:
                    counts["observation_reproduction_pending"] += 1
                if action_opportunity:
                    counts["action_reproduction_opportunities"] += 1
                    expected_type = str(metadata.get("expected_action_type"))
                    target_slots = [int(slot) for slot in metadata.get("matching_candidate_slots", []) or []]
                    valid_slots = [slot for slot in target_slots if float(mask[env.action_space.encode(expected_type, slot)]) > 0.0]
                    mask_valid = bool(valid_slots) if metadata.get("expected_action_requires_target") else bool(mask[env.action_space.encode(expected_type, 0)] > 0.0)
                    if mask_valid:
                        counts["safe_mask_valid_opportunities"] += 1
                    if mask_valid and expected_type in {"click_element", "fill_input", "press_enter"}:
                        counts["semantically_executable_opportunities"] += 1
                    else:
                        counts["non_executable_action_instructions"] += 1
                action_id = select_factorized_greedy_action(network, state, mask, env.action_space)
                action = env.action_space.decode(action_id)
                next_raw, _, done, info = env.step(action_id)
                action["failed"] = bool(info.get("last_action_error"))
                action["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = _sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": action}))
                reward, components = calculate_dqn_reward(raw, next_raw, action, anomalies, history)
                progress = bool(history.get("last_confirmation_progress"))
                if action_opportunity:
                    type_match = action.get("action_type") == metadata.get("expected_action_type")
                    target_match = not bool(metadata.get("expected_action_requires_target")) or int(action.get("candidate_index", -1)) in [int(x) for x in metadata.get("matching_candidate_slots", []) or []]
                    if type_match:
                        counts["actual_policy_action_type_matches"] += 1
                    if type_match and target_match:
                        counts["actual_policy_reproduction_matches"] += 1
                if observation_pending:
                    rules[str(metadata.get("active_rule_id") or "UNKNOWN")]["observation_pending"] += 1
                    if progress:
                        counts["observation_reproduction_events"] += 1
                        rules[str(metadata.get("active_rule_id") or "UNKNOWN")]["observation_reproduction_events"] += 1
                if progress:
                    counts["confirmation_progress"] += 1
                confirmed = bool(history.get("last_confirmation_confirmed"))
                if confirmed:
                    counts["confirmed"] += 1
                if pending:
                    rules[str(metadata.get("active_rule_id") or "UNKNOWN")]["pending"] += 1
                rows.append({"episode": episode, "step": step, "rule_id": metadata.get("active_rule_id"), "confirmation_mode": mode, "expected_action_type": metadata.get("expected_action_type"), "expected_target_identity": metadata.get("expected_target_identity"), "selected_action_type": action.get("action_type"), "action_opportunity": action_opportunity, "observation_pending": observation_pending, "safe_mask_valid": bool(action_opportunity and mask_valid), "semantically_executable": bool(action_opportunity and mask_valid and metadata.get("expected_action_type") in {"click_element", "fill_input", "press_enter"}), "progress": progress, "confirmed": confirmed, "reward_components": dict(components), "reward": float(reward)})
                raw = next_raw
                if done:
                    break
    finally:
        env.close()
    report = {
        "model_path": args.model_path,
        "site_id": str(site["site_id"]),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "ground_truth_used": False,
        "training_performed": False,
        "reward_config_changed": False,
        "counts": {
            **{key: int(counts.get(key, 0)) for key in (
                "pending_candidates", "action_reproduction_pending", "observation_reproduction_pending",
                "action_reproduction_opportunities", "semantically_executable_opportunities",
                "safe_mask_valid_opportunities", "actual_policy_action_type_matches",
                "actual_policy_reproduction_matches", "observation_reproduction_events",
                "confirmation_progress", "confirmed", "non_executable_action_instructions",
            )},
        },
        "by_rule": {rule: dict(values) for rule, values in sorted(rules.items())},
        "action_semantics_matrix": __import__("dqn_error_discovery.agents.confirmation_reproduction", fromlist=["action_semantics_matrix"]).action_semantics_matrix(env.action_space),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
