"""Inference-only lifecycle telemetry validation for the frozen Run6 seed42 net."""
from __future__ import annotations

import argparse
import json
from itertools import cycle, islice
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from dqn_error_discovery.runners.train_dqn_error_detector import build_replay_metadata, calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.productive_context_sampler import classify_context
from dqn_error_discovery.v3.run5_context_aware_dqn_training import (
    CANDIDATE_B_PATHS, SiteServer, _dqn_state, _step_v3, _v3_mask, _v3_state,
    active_candidate_from_anomalies, active_endpoint_and_key, active_context_metadata,
    choose_dqn, discover_sites, q_values, read_json, reset_env, transition_telemetry,
)
from agents.ppo_agent import PPOAgent
from services.anomaly_detection_service import AnomalyConfirmationTracker, detect_anomalies


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full"
DEFAULT_MODEL = ROOT / "artifacts/models/v3/run6_confirmation_aware_dqn/seed42_full/rawd_v3_confirmation_aware_dqn_seed42.pt"
SPACE = V3ActionSpace(32)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_ready_sites(source_root: Path, artifact_dir: Path, limit: int) -> list[dict[str, Any]]:
    registry = discover_sites(source_root)
    cached = artifact_dir / "runtime_preflight.json"
    preflight = read_json(cached).get("sites", []) if cached.exists() else []
    ready_by_id = {str(row.get("site_id")): row for row in preflight if row.get("status") == "TRAIN_READY"}
    sites = [ready_by_id.get(str(site.get("site_id"))) for site in registry]
    sites = [site for site in sites if site]
    return sites[:limit] if limit else sites


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(r"C:\workspace\RL_Errorsite_fullstack_run4"))
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--site-limit", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--repeat-sites", action="store_true", help="cycle the selected ready sites for diagnostic episode coverage")
    args = parser.parse_args()
    out = args.artifact_dir / "lifecycle_validation"
    out.mkdir(parents=True, exist_ok=True)
    network = FactorizedQNetwork(984, 20, 32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()
    ppo = PPOAgent(635, 640)
    ppo.load(CANDIDATE_B_PATHS["ppo"])
    sites = load_ready_sites(args.source_root, args.artifact_dir, args.site_limit)
    if args.repeat_sites and sites and args.episodes > 0:
        sites = list(islice(cycle(sites), args.episodes))
    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    episodes = 0
    global_step = 0
    infrastructure_errors = []
    for site in sites:
        if episodes >= args.episodes:
            break
        with SiteServer(site):
            for local_episode in range(2):
                if episodes >= args.episodes:
                    break
                episodes += 1
                env = None
                try:
                    env, raw = reset_env(site, args.max_steps, 60000)
                    history: dict[str, Any] = {}
                    sessions = 0
                    for ppo_step in range(args.max_steps):
                        selected = ppo.select_action(_v3_state(env, raw), _v3_mask(SPACE, raw))
                        aid = int(selected["action_id"])
                        ppo_action = {**SPACE.decode(aid), "action_id": aid, "failed": False}
                        after, _, done, info, _ = _step_v3(env, aid, raw, SPACE)
                        ppo_action["failed"] = bool(info.get("last_action_error"))
                        anomalies = detect_anomalies(raw, after, {"action": ppo_action, "step_id": global_step + 1, "agent_mode": "PPO_GLOBAL"}, site_profile={})
                        context = classify_context(raw, after, anomalies, info)
                        active = active_candidate_from_anomalies(anomalies)
                        raw = after
                        if not context.get("admit") or sessions >= 2:
                            if done:
                                break
                            continue
                        sessions += 1
                        tracker = AnomalyConfirmationTracker()
                        local_raw = raw
                        for local_step in range(args.max_local_steps):
                            mask = _v3_mask(SPACE, local_raw)
                            meta = active_context_metadata(history, local_raw, active, mask)
                            state = _dqn_state(env, local_raw, history, active_candidate=active)
                            q = q_values(network, state, mask)
                            local_id = choose_dqn(network, state, mask, 0.0)
                            action = {**SPACE.decode(local_id), "action_id": int(local_id), "failed": False}
                            local_after, _, local_done, local_info, _ = _step_v3(env, local_id, local_raw, SPACE)
                            action["failed"] = bool(local_info.get("last_action_error"))
                            local_anomalies = detect_anomalies(local_raw, local_after, {"action": action, "step_id": global_step + 1, "agent_mode": "DQN_LOCAL", "confirmation_tracker": tracker}, site_profile={})
                            transition_id = f"{episodes}:{site['site_id']}:session{sessions}:{local_step + 1}:{global_step + 1}"
                            history["transition_id"] = transition_id
                            history["current_transition_step"] = global_step + 1
                            reward, components = calculate_dqn_reward(local_raw, local_after, action, local_anomalies, history)
                            tel = transition_telemetry(q, mask, int(local_id), local_raw, local_after, active, meta, local_anomalies, components, history)
                            metadata = build_replay_metadata(
                                confirmation_context=bool(meta.get("pending_confirmation") or history.get("last_confirmation_progress") or components.get("confirmation_upgrade", 0.0)),
                                action_replay=bool(meta.get("pending_confirmation") and meta.get("decoded_origin_action") == action.get("action_type")),
                                reproduction_attempt=bool(meta.get("pending_confirmation") or history.get("last_confirmation_progress")),
                                candidate_relevant_effect=bool(tel.get("candidate_relevant_effect")),
                                same_chain_recurrence=bool(tel.get("same_chain_recurrence")),
                                confirmation_upgrade=bool(components.get("confirmation_upgrade", 0.0)),
                            )
                            transition_rows.append({"transition_id": transition_id, "site_id": site["site_id"], "episode": episodes, "local_step": local_step + 1, "selected_action": action, "reward_input": {"pre_pending_records": history.get("last_pre_pending_records", "MISSING"), "event": history.get("last_confirmation_transition", "MISSING")}, "reward_output": {"eligible": history.get("last_confirmation_upgrade_eligible", "MISSING"), "components": components, "final_reward": reward}, "replay_metadata": metadata, **tel})
                            global_step += 1
                            local_raw = local_after
                            if local_done:
                                break
                        if done:
                            break
                    rows.append({"site_id": site["site_id"], "episode": episodes, "sessions": sessions})
                except Exception as exc:
                    infrastructure_errors.append({"site_id": site["site_id"], "episode": episodes, "error": repr(exc)})
                finally:
                    if env is not None:
                        env.close()
    ar = [row for row in transition_rows if row.get("requires_action_reproduction") is True]
    obs = [row for row in transition_rows if row.get("requires_action_reproduction") is False and row.get("confirmation_event_present")]
    def matrix(group: list[dict[str, Any]]) -> dict[str, int]:
        tp = sum(bool(row.get("confirmation_upgrade_component", 0.0)) and bool(row.get("same_chain_recurrence")) for row in group)
        fp = sum(bool(row.get("confirmation_upgrade_component", 0.0)) and not bool(row.get("same_chain_recurrence")) for row in group)
        fn = sum(not bool(row.get("confirmation_upgrade_component", 0.0)) and bool(row.get("same_chain_recurrence")) for row in group)
        tn = len(group) - tp - fp - fn
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    result = {"episodes": episodes, "transitions": len(transition_rows), "action_replay_opportunities": sum(bool(row.get("target_actionable")) and row.get("requires_action_reproduction") is True for row in transition_rows), "action_replay_confirmations": sum(bool(row.get("requires_action_reproduction") is True and row.get("confirmation_upgrade_component", 0.0)) for row in transition_rows), "failed_reproductions": sum(bool(row.get("is_failed_reproduction")) for row in transition_rows), "unrelated_actions": sum(bool(row.get("is_confirmation_context")) and not bool(row.get("is_reproduction_attempt")) for row in transition_rows), "observation_confirmations": len(obs), "action_replay_confusion": matrix(ar), "observation_confusion": matrix(obs), "missing_contract_fields": dict(Counter(field for row in transition_rows for field in ("active_candidate_logical_key", "active_candidate_status_before", "active_candidate_status_after", "requires_action_reproduction", "reproduction_strategy", "reproduction_action_type", "reproduction_target_identity", "confirmation_event_key", "confirmation_source", "confirmation_event_step") if row.get(field) == "MISSING")), "infrastructure_errors": infrastructure_errors, "optimizer_updates": 0, "backward": 0, "checkpoint_modified": False, "gt_runtime_access": False, "policy_result_unchanged": True, "transition_rows": transition_rows}
    write_json(out / "runtime_validation.json", result)
    print(json.dumps({key: result[key] for key in ("episodes", "transitions", "action_replay_opportunities", "action_replay_confirmations", "failed_reproductions", "observation_confirmations", "infrastructure_errors")}, indent=2))


if __name__ == "__main__":
    main()
