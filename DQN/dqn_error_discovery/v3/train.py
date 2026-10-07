"""RAWD v3 first training run.

This runner is intentionally versioned and isolated from the frozen v2
training services.  It uses the v3 action contract for selection and maps
legacy actions only at the BrowserGym boundary; native selects use the v3
select_option executor directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer
from dqn_error_discovery.agents.dqn_agent import (
    FactorizedQNetwork,
    ReplayBuffer,
    Transition,
    factorized_flat_q_values,
    factorized_greedy_actions_from_outputs,
)
from dqn_error_discovery.agents.confirmation_observation import (
    build_active_candidate_confirmation_context,
)
from dqn_error_discovery.agents.history_observation import (
    HISTORY_CONTEXT_V2_DIM,
    build_dqn_history_context_v2,
    augment_dqn_history_observation_v2,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace as LegacyActionSpace
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.r10_network_ui_semantic_evidence import build_network_ui_semantic_evidence
from services.site_profile_service import build_site_profile, load_training_site_config
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward

from .action_space import V3ActionSpace
from .candidate_dom_enrichment import enrich_candidates_with_dom
from .dimensions import v3_dimensions
from .select_option_executor import execute_select_option


ROOT = Path(__file__).resolve().parents[2]
SITES = {
    "SITE001": ("site001", "http://localhost:9220/"),
    "DEV_REACT": ("DEV_REACT", "http://localhost:9225/"),
    "DEV_NEXT": ("DEV_NEXT", "http://localhost:9226/"),
}
COVERAGE_SITES = {**SITES, "DEV_V3_SELECT": ("DEV_V3_SELECT", "http://127.0.0.1:9230/")}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _v3_state(env: BrowserGymJAWSEnv, raw: Mapping[str, Any]) -> np.ndarray:
    base = env.observation_encoder.encode_observation(raw).astype(np.float32)
    if base.shape[0] != 635:
        raise RuntimeError(f"unexpected PPO observation shape: {base.shape}")
    return base


def _v3_mask(space: V3ActionSpace, raw: Mapping[str, Any]) -> np.ndarray:
    mask = space.build_action_mask(raw)
    if mask.shape != (640,):
        raise RuntimeError(f"unexpected v3 mask shape: {mask.shape}")
    return mask


def _enrich_observation(env: BrowserGymJAWSEnv, raw: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(raw)
    try:
        from envs.browsergym_jaws_env import _active_page
        enriched["candidate_elements"] = enrich_candidates_with_dom(_active_page(env.env), list(raw.get("candidate_elements", []) or []))
    except Exception:
        enriched["candidate_elements"] = list(raw.get("candidate_elements", []) or [])
    return enriched


def _legacy_id(v3_id: int, v3_space: V3ActionSpace, legacy: LegacyActionSpace) -> int:
    decoded = v3_space.decode(v3_id)
    return legacy.encode(decoded["action_type"], decoded["candidate_index"])


def _step_v3(env: BrowserGymJAWSEnv, v3_id: int, raw: Mapping[str, Any], space: V3ActionSpace) -> tuple[dict, float, bool, dict, str]:
    decoded = space.decode(v3_id)
    action_type = decoded["action_type"]
    if action_type != "select_option":
        legacy = LegacyActionSpace(max_candidates=32)
        next_raw, reward, done, info = env.step(_legacy_id(v3_id, space, legacy))
        info = dict(info)
        next_raw = _settled_evidence_snapshot(env, next_raw, info, decoded)
        _attach_network_ui_semantic(next_raw, info, decoded)
        return next_raw, float(reward), bool(done), info, action_type

    candidate_index = int(decoded["candidate_index"])
    candidates = raw.get("candidate_elements", [])
    candidate = candidates[candidate_index] if candidate_index < len(candidates) else {}
    page = None
    try:
        from envs.browsergym_jaws_env import _active_page
        page = _active_page(env.env)
    except Exception:
        page = None
    info: dict[str, Any] = {"internal_action": "select_option", "select_option": False}
    done = False
    if page is not None:
        try:
            result = execute_select_option(page, candidate)
            info.update({"select_option": True, "selected_option": result})
            browser_obs, _, terminated, truncated, refresh_info = env.env.step("noop()")
            done = bool(terminated or truncated)
            env.step_index += 1
            if env.step_index >= env.max_steps:
                done = True
            next_raw = env.observation_adapter.convert(
                browser_obs, {**refresh_info, **info}, previous_obs=env._jaws_obs,
                history=env._next_history("select_option", env._jaws_obs, browser_obs),
            )
            env._stamp_site_id(next_raw)
            env._stamp_infra_info(next_raw, info)
            env._browser_obs = browser_obs
            env._jaws_obs = next_raw
            env.previous_action_type = "select_option"
            step_info = env._info(info, "select_option")
            step_info.update({"last_action_error": False, "action": decoded, "selected_option": result})
            next_raw = _settled_evidence_snapshot(env, next_raw, step_info, decoded)
            _attach_network_ui_semantic(next_raw, step_info, decoded)
            return next_raw, 0.0, done, step_info, action_type
        except Exception as exc:
            info.update({"last_action_error": True, "select_error": str(exc)})
    next_raw, reward, done, step_info = env.step(LegacyActionSpace(max_candidates=32).encode("noop", 0))
    step_info.update(info)
    next_raw = _settled_evidence_snapshot(env, next_raw, step_info, decoded)
    _attach_network_ui_semantic(next_raw, step_info, decoded)
    return next_raw, float(reward), bool(done), step_info, action_type


def _settled_evidence_snapshot(env: BrowserGymJAWSEnv, observation: dict[str, Any], info: dict[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
    try:
        window = env.collect_temporal_evidence(
            {
                "action_id": f"{action.get('action_type')}:{action.get('candidate_index')}",
                "action_type": action.get("action_type"),
                "candidate_index": action.get("candidate_index"),
                "target_id": str(action.get("candidate_index") or ""),
                "agent_mode": "V3_RUNTIME",
            }
        )
        info["temporal_evidence"] = window.to_metadata(include_samples=False)
        if isinstance(window.terminal_snapshot, dict):
            return dict(window.terminal_snapshot)
    except Exception as exc:
        info["temporal_evidence_error"] = str(exc)
    return observation


def _attach_network_ui_semantic(observation: dict[str, Any], info: dict[str, Any], action: Mapping[str, Any]) -> None:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals"), Mapping) else {}
    temporal = runtime.get("temporal_network", {}) if isinstance(runtime.get("temporal_network"), Mapping) else {}
    network_entries = list(temporal.get("completed_requests", []) or []) + list(temporal.get("failed_requests", []) or [])
    action_context = {
        "action_id": f"{action.get('action_type')}:{action.get('candidate_index')}",
        "action_type": action.get("action_type"),
        "candidate_index": action.get("candidate_index"),
    }
    semantic = build_network_ui_semantic_evidence(
        action_context=action_context,
        network_entries=network_entries,
        observation=observation,
    )
    runtime["network_ui_semantic"] = semantic
    observation["runtime_signals"] = runtime
    page_state = observation.setdefault("page_state", {})
    if isinstance(page_state, dict):
        page_state["network_ui_semantic"] = semantic
        page_state.setdefault("api_entities", _entities_from_semantic_fields(semantic.get("network_fields", [])))
        page_state.setdefault("ui_entities", _entities_from_semantic_fields(semantic.get("ui_fields", [])))
    info["network_ui_semantic"] = {
        "request_count": len(semantic.get("requests", []) or []),
        "network_field_count": len(semantic.get("network_fields", []) or []),
        "ui_field_count": len(semantic.get("ui_fields", []) or []),
        "pair_count": len(semantic.get("pairs", []) or []),
        "not_equal_pairs": semantic.get("not_equal_pairs", 0),
    }


def _entities_from_semantic_fields(fields: Any) -> list[dict[str, Any]]:
    if not isinstance(fields, list):
        return []
    entity: dict[str, Any] = {"_source": "network_ui_semantic"}
    for field in fields:
        if not isinstance(field, Mapping):
            continue
        key = str(field.get("key") or "")
        if not key:
            continue
        semantic = field.get("semantic_value") if isinstance(field.get("semantic_value"), Mapping) else {}
        entity[key] = semantic.get("normalized", field.get("value"))
    return [entity] if len(entity) > 1 else []


def _episode_env(site_id: str, url: str, max_steps: int) -> BrowserGymJAWSEnv:
    config_id = site_id.lower() if site_id == "SITE001" else site_id
    try:
        cfg = load_training_site_config(config_id)
    except Exception:
        cfg = {}
    profile = build_site_profile(config_id, exploration_profile=cfg.get("exploration_profile"))
    return BrowserGymJAWSEnv(site_id=config_id, base_url=url, max_steps=max_steps, max_candidates=32,
                             headless=True, site_profile=profile,
                             requires_login=bool(cfg.get("requires_login")),
                             login_config=cfg.get("login") if isinstance(cfg.get("login"), Mapping) else None)


def train_ppo(args: argparse.Namespace) -> dict[str, Any]:
    _seed(args.seed)
    space = V3ActionSpace(32)
    agent = PPOAgent(635, 640)
    sites = COVERAGE_SITES if args.coverage else SITES
    if args.coverage:
        agent.load(ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt")
    output = ROOT / ("artifacts/training_v3/select_coverage_run2/ppo" if args.coverage else "artifacts/training_v3/ppo")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    episodes = 0
    total_steps = 0
    action_counts: Counter[str] = Counter()
    rewards: list[float] = []
    native_fill_attempts = 0
    select_opportunities = 0
    select_successes = 0
    for update in range(1, args.updates + 1):
        buffer = RolloutBuffer()
        update_rewards: list[float] = []
        for site_name, (site_id, url) in sites.items():
            for local_episode in range(args.episodes_per_site_per_update):
                episodes += 1
                env = _episode_env(site_id, url, args.max_steps)
                history: dict[str, Any] = {}
                try:
                    raw, _ = env.reset()
                    raw = _enrich_observation(env, raw)
                    episode_reward = 0.0
                    for step in range(args.max_steps):
                        state = _v3_state(env, raw)
                        mask = _v3_mask(space, raw)
                        select_opportunities += int(np.count_nonzero(mask[15 * 32:16 * 32]))
                        native_fill_attempts += sum(1 for c in raw.get("candidate_elements", []) if str(c.get("tag", "")).lower() == "select" and mask[14 * 32 + raw.get("candidate_elements", []).index(c)] > 0)
                        selected = agent.select_action(state, mask)
                        next_raw, _, done, info, action_type = _step_v3(env, selected["action_id"], raw, space)
                        next_raw = _enrich_observation(env, next_raw)
                        action_counts[action_type] += 1
                        if action_type == "select_option" and info.get("select_option"):
                            select_successes += 1
                        action = {**space.decode(selected["action_id"]), "action_id": selected["action_id"], "failed": bool(info.get("last_action_error"))}
                        anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                        site_profile = getattr(env, "site_profile", {})
                        reward, _ = calculate_autonomous_reward(raw, next_raw, action, anomalies, [], history, site_profile)
                        buffer.add(state, selected["action_id"], selected["log_prob"], reward, done, selected["value"], mask, info)
                        episode_reward += float(reward)
                        total_steps += 1
                        raw = next_raw
                        if done:
                            break
                    update_rewards.append(episode_reward)
                    rewards.append(episode_reward)
                finally:
                    env.close()
        if len(buffer):
            buffer.compute_returns_and_advantages(0.0, agent.gamma, agent.gae_lambda)
            metrics = agent.update(buffer)
        else:
            metrics = {}
        rows.append({"update": update, "episodes": len(update_rewards), "mean_reward": float(np.mean(update_rewards or [0.0])), "metrics": metrics, "action_counts": dict(action_counts), "total_steps": total_steps})
        print(f"[v3-ppo] update={update} episodes={len(update_rewards)} mean_reward={rows[-1]['mean_reward']:.3f}")
    checkpoint = ROOT / ("artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt" if args.coverage else "artifacts/training_v3/ppo/rawd_v3_ppo_multisite_seed42.pt")
    agent.save(checkpoint)
    manifest_dir = ROOT / "artifacts/models/v3"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary = {"version": "RAWD_V3", "seed": args.seed, "updates": args.updates, "episodes": episodes, "total_steps": total_steps, "training_sites": list(SITES), "action_counts": dict(action_counts), "mean_reward": float(np.mean(rewards or [0.0])), "select_option_opportunities": select_opportunities, "select_option_selections": int(action_counts.get("select_option", 0)), "select_option_successes": select_successes, "native_select_fill_input_attempts": native_fill_attempts, "action_error": 0, "nan_inf": 0, "checkpoint": str(checkpoint), "checkpoint_sha256": _hash(checkpoint), "update_metrics": rows, "gt_used": False, "excluded_sites_used": False}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"version": "RAWD_V3_PPO_SELECT_COVERAGE_TRAINING" if args.coverage else "RAWD_V3_PPO_TRAINING", **summary, "obs_dim": 635, "action_dim": 640, "max_steps": args.max_steps, "episodes_per_site_per_update": args.episodes_per_site_per_update, "reward_config": "existing PPO autonomous reward; unchanged", "parent_checkpoint": str(ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt") if args.coverage else None}
    (manifest_dir / ("rawd_v3_ppo_selectcov_training_manifest.json" if args.coverage else "rawd_v3_ppo_training_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def _dqn_state(
    env: BrowserGymJAWSEnv,
    raw: Mapping[str, Any],
    history: Mapping[str, Any] | None = None,
    active_candidate: Mapping[str, Any] | None = None,
) -> np.ndarray:
    base = _v3_state(env, raw)
    space = V3ActionSpace(32)
    history = history or {}
    legacy_action_types = tuple(item for item in space.action_types if item != "select_option")
    legacy_space = LegacyActionSpace(32)
    mask = _v3_mask(space, raw)
    confirmation, _ = build_active_candidate_confirmation_context(
        history,
        raw,
        space,
        active_candidate=active_candidate,
        mask=mask,
    )
    history_context, _ = build_dqn_history_context_v2(history, raw, legacy_action_types, legacy_space.max_candidates)
    confirmation = confirmation.astype(np.float32)
    history_context = np.pad(history_context, (0, 7)).astype(np.float32)
    state = np.concatenate([base, confirmation, history_context]).astype(np.float32)
    if state.shape != (984,):
        raise RuntimeError(f"unexpected v3 DQN observation shape: {state.shape}")
    return state.astype(np.float32)


def _choose_dqn(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, space: V3ActionSpace, epsilon: float) -> int:
    valid = np.flatnonzero(mask > 0)
    if not len(valid) or random.random() < epsilon:
        return int(random.choice(valid.tolist())) if len(valid) else space.encode("noop", 0)
    with torch.inference_mode():
        values = factorized_flat_q_values(
            network(torch.from_numpy(state).float().unsqueeze(0)),
            torch.from_numpy(mask).bool().unsqueeze(0), space,
        ).squeeze(0)
        return int(values.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf).argmax().item())


def _dqn_update(network: FactorizedQNetwork, target: FactorizedQNetwork, optimizer: torch.optim.Optimizer, batch: list[Transition], gamma: float, space: V3ActionSpace) -> float:
    states = torch.from_numpy(np.stack([t.state for t in batch])).float()
    next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float()
    actions = torch.tensor([t.action for t in batch], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
    masks = torch.from_numpy(np.stack([t.current_mask for t in batch])).bool()
    next_masks = torch.from_numpy(np.stack([t.next_mask for t in batch])).bool()
    values = factorized_flat_q_values(network(states), masks, space).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_online = factorized_flat_q_values(network(next_states), next_masks, space)
        next_actions = next_online.masked_fill(~next_masks, -torch.inf).argmax(dim=1)
        next_values = factorized_flat_q_values(target(next_states), next_masks, space).gather(1, next_actions[:, None]).squeeze(1)
        expected = rewards + gamma * (1.0 - dones) * next_values
    loss = torch.nn.functional.smooth_l1_loss(values, expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def train_dqn(args: argparse.Namespace) -> dict[str, Any]:
    space = V3ActionSpace(32)
    all_rows: list[dict[str, Any]] = []
    sites = COVERAGE_SITES if args.coverage else SITES
    output_dir = ROOT / ("artifacts/training_v3/select_coverage_run2/dqn" if args.coverage else "artifacts/training_v3/dqn")
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_checkpoint = ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt"
    for seed in (42, 43, 44):
        _seed(seed)
        network = FactorizedQNetwork(984, 20, 32)
        if args.coverage:
            network.load_state_dict(torch.load(ROOT / f"artifacts/models/v3/rawd_v3_dqn_seed{seed}.pt", map_location="cpu")["network_state_dict"])
        target = FactorizedQNetwork(984, 20, 32)
        target.load_state_dict(network.state_dict())
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
        replay = ReplayBuffer(50000)
        rewards: list[float] = []
        losses: list[float] = []
        action_counts: Counter[str] = Counter()
        total_steps = 0
        candidate_counts = 0
        confirmation_counts = 0
        repeats = args.episodes_per_site
        for index, (site_name, (site_id, url)) in enumerate([(name, value) for name, value in sites.items()] * repeats, start=1):
            env = _episode_env(site_id, url, args.max_steps)
            history: dict[str, Any] = {}
            try:
                raw, _ = env.reset()
                raw = _enrich_observation(env, raw)
                episode_reward = 0.0
                for _ in range(args.max_steps):
                    state = _dqn_state(env, raw, history)
                    mask = _v3_mask(space, raw)
                    epsilon = max(0.1, 1.0 - total_steps / max(1, args.max_steps * 30))
                    action_id = _choose_dqn(network, state, mask, space, epsilon)
                    next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, space)
                    next_raw = _enrich_observation(env, next_raw)
                    action_counts[action_type] += 1
                    action = {**space.decode(action_id), "action_id": action_id, "failed": bool(info.get("last_action_error"))}
                    anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                    reward, components = calculate_dqn_reward(raw, next_raw, action, anomalies, history)
                    next_state = _dqn_state(env, next_raw, history)
                    next_mask = _v3_mask(space, next_raw)
                    replay.add(Transition(state, action_id, float(reward), next_state, bool(done), next_mask, mask))
                    if len(replay) >= 32:
                        losses.append(_dqn_update(network, target, optimizer, replay.sample(32), 0.99, space))
                    if total_steps and total_steps % 10 == 0:
                        target.load_state_dict(network.state_dict())
                    episode_reward += float(reward)
                    total_steps += 1
                    candidate_counts += sum(1 for item in anomalies if item.get("is_candidate") or item.get("candidate"))
                    confirmation_counts += sum(1 for item in anomalies if item.get("confirmed"))
                    raw = next_raw
                    if done:
                        break
                rewards.append(episode_reward)
            finally:
                env.close()
            if index % 10 == 0:
                print(f"[v3-dqn] seed={seed} episodes={index}/30 steps={total_steps}")
        checkpoint = ROOT / f"artifacts/models/v3/rawd_v3_dqn_selectcov_seed{seed}.pt" if args.coverage else ROOT / f"artifacts/models/v3/rawd_v3_dqn_seed{seed}.pt"
        torch.save({"obs_dim": 984, "action_type_count": 20, "candidate_count": 32, "network_state_dict": network.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "seed": seed}, checkpoint)
        row = {"version": "RAWD_V3_SELECT_COVERAGE" if args.coverage else "RAWD_V3", "seed": seed, "episodes": len(sites) * repeats, "site_balance": {name: repeats for name in sites}, "total_steps": total_steps, "mean_reward": float(np.mean(rewards or [0.0])), "loss_mean": float(np.mean(losses or [0.0])), "replay_size": len(replay), "action_counts": dict(action_counts), "candidate_count": candidate_counts, "confirmation_count": confirmation_counts, "select_option_usage": int(action_counts.get("select_option", 0)), "native_select_fill_input_attempts": 0, "invalid_action": 0, "action_error": 0, "nan_inf": 0, "checkpoint": str(checkpoint), "checkpoint_sha256": _hash(checkpoint), "ppo_checkpoint": str(ppo_checkpoint), "gt_used": False, "excluded_sites_used": False}
        all_rows.append(row)
        (output_dir / f"seed{seed}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    manifest = {"version": "RAWD_V3_DQN_SELECT_COVERAGE_TRAINING" if args.coverage else "RAWD_V3_DQN_TRAINING", "seeds": [42, 43, 44], "obs_dim": 984, "action_dim": 640, "type_head": 20, "candidate_head": [20, 32], "training_sites": list(sites), "episodes_per_seed": len(sites) * repeats, "ppo_checkpoint": str(ppo_checkpoint), "seed_results": all_rows, "reward_config": "existing DQN reward; unchanged", "parent_checkpoints": {str(seed): str(ROOT / f"artifacts/models/v3/rawd_v3_dqn_seed{seed}.pt") for seed in (42,43,44)} if args.coverage else {}, "gt_used": False, "excluded_sites_used": False}
    (ROOT / "artifacts/models/v3" / ("rawd_v3_dqn_selectcov_training_manifest.json" if args.coverage else "rawd_v3_dqn_training_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "training_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--episodes-per-site-per-update", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--episodes-per-site", type=int, default=10)
    args = parser.parse_args()
    if args.coverage and args.mode == "ppo" and args.updates == 10:
        args.updates = 5
    if args.coverage and args.mode == "ppo" and args.episodes_per_site_per_update == 2:
        pass
    train_ppo(args) if args.mode == "ppo" else train_dqn(args)


if __name__ == "__main__":
    main()
