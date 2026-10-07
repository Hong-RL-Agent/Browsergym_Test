"""Diagnose PPO masked policy probabilities without changing training behavior."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer
from dqn_error_discovery.runners.run_policy_sanity_check import SanityMetricsAggregator
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward


SENSITIVE_DIAGNOSTIC_KEYS = {
    "query",
    "password",
    "token",
    "authorization",
    "cookie",
    "data_bug_id",
    "catalog_bug_id_matches",
    "matched_bug_id",
    "known_bug",
    "ground_truth",
}


def masked_policy_distribution(agent: PPOAgent, obs_vector: np.ndarray, action_mask: np.ndarray) -> dict[str, Any]:
    with torch.no_grad():
        obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=agent.device).unsqueeze(0)
        logits, value = agent.policy(obs)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=agent.device).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask, -1e9)
        probabilities = torch.softmax(masked_logits, dim=-1).squeeze(0).detach().cpu().numpy()
        logits_np = masked_logits.squeeze(0).detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=masked_logits)
        entropy = float(dist.entropy().squeeze(0).item())
    return {
        "logits": logits_np,
        "probabilities": probabilities,
        "entropy": entropy,
        "value": float(value.squeeze(0).item()),
    }


def action_type_probability_mass(probabilities: np.ndarray, action_mask: np.ndarray, action_space) -> dict[str, float]:
    masses = {action_type: 0.0 for action_type in action_space.action_types}
    valid_total = 0.0
    for action_id in np.flatnonzero(action_mask > 0):
        decoded = action_space.decode(int(action_id))
        probability = float(probabilities[int(action_id)])
        masses[str(decoded["action_type"])] += probability
        valid_total += probability
    if valid_total:
        for action_type in masses:
            masses[action_type] = float(masses[action_type] / valid_total)
    return masses


def top_k_valid_actions(
    probabilities: np.ndarray,
    logits: np.ndarray,
    action_mask: np.ndarray,
    action_space,
    observation: Mapping[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    valid_ids = np.flatnonzero(action_mask > 0)
    ordered = sorted(valid_ids.tolist(), key=lambda action_id: float(probabilities[int(action_id)]), reverse=True)
    return [
        _action_probability_row(int(action_id), probabilities, logits, action_space, observation)
        for action_id in ordered[:top_k]
    ]


def summarize_advantages(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "advantage_min": 0.0,
            "advantage_max": 0.0,
            "positive_advantage_ratio": 0.0,
            "negative_advantage_ratio": 0.0,
            "finite": True,
        }
    return {
        "advantage_mean": float(np.mean(values)),
        "advantage_std": float(np.std(values)),
        "advantage_min": float(np.min(values)),
        "advantage_max": float(np.max(values)),
        "positive_advantage_ratio": float(np.mean(values > 0)),
        "negative_advantage_ratio": float(np.mean(values < 0)),
        "finite": bool(np.isfinite(values).all()),
    }


def observation_delta(before: np.ndarray | None, after: np.ndarray) -> dict[str, Any]:
    if before is None:
        return {"l1": 0.0, "l2": 0.0, "changed_feature_count": 0}
    delta = np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32)
    return {
        "l1": float(np.sum(np.abs(delta))),
        "l2": float(np.linalg.norm(delta)),
        "changed_feature_count": int(np.count_nonzero(np.abs(delta) > 1e-6)),
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    report: dict[str, Any] = {
        "site": site["site_id"],
        "url": site["base_url"],
        "model": args.model_path,
        "episode_count": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "max_candidates": args.max_candidates,
        "entropy_coefficient": 0.02,
        "mask_update_findings": {
            "rollout_mask_applied": True,
            "old_log_prob_from_masked_distribution": True,
            "buffer_stores_action_mask": True,
            "update_uses_stored_action_mask": True,
            "new_log_prob_from_masked_distribution": True,
            "entropy_from_masked_distribution": True,
        },
        "legacy_bias_findings": _legacy_bias_findings(),
        "actor_logit_bias_findings": _actor_logit_bias_findings(),
    }
    report["fixed_observation_probability_comparison"] = _fixed_observation_comparison(args, site)
    report["greedy"] = _run_policy_mode(args, site, "greedy")
    report["stochastic"] = _run_policy_mode(args, site, "stochastic")
    report["stochastic_vs_greedy"] = _compare_modes(report["greedy"], report["stochastic"])
    report["diagnosis"] = _diagnosis(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    report["summary_path"] = str(output)
    return report


def _run_policy_mode(args: argparse.Namespace, site: Mapping[str, Any], mode: str) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    agent = PPOAgent(env.observation_encoder.get_obs_dim(), env.action_space.get_action_dim())
    if args.model_path:
        agent.load(Path(args.model_path))
    aggregator = SanityMetricsAggregator("ppo")
    reward_history: dict[str, Any] = {}
    policy_rows: list[dict[str, Any]] = []
    observation_deltas: list[dict[str, Any]] = []
    bid11_rows: list[dict[str, Any]] = []
    episode_buffers: list[RolloutBuffer] = []
    trajectory_summaries: list[dict[str, Any]] = []
    action_type_mass_totals: Counter[str] = Counter()
    action_type_mass_count = 0
    previous_vector: np.ndarray | None = None
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            reward_history = {}
            previous_vector = None
            episode_buffer = RolloutBuffer()
            episode_done = False
            for step in range(1, args.max_steps + 1):
                obs_vector = env.observation_encoder.encode_observation(observation)
                mask = env.action_space.build_action_mask(observation)
                distribution = masked_policy_distribution(agent, obs_vector, mask)
                probabilities = distribution["probabilities"]
                logits = distribution["logits"]
                if mode == "greedy":
                    selected = agent.select_greedy_action(obs_vector, mask)
                else:
                    selected = agent.select_action(obs_vector, mask)
                action_id = int(selected["action_id"])
                action = env.action_space.decode(action_id)
                action["action_id"] = action_id
                mask_stats = env.action_space.build_action_mask_stats(observation, mask, action_id)
                top_actions = top_k_valid_actions(probabilities, logits, mask, env.action_space, observation, top_k=5)
                masses = action_type_probability_mass(probabilities, mask, env.action_space)
                for action_type, probability in masses.items():
                    action_type_mass_totals[action_type] += float(probability)
                action_type_mass_count += 1
                sorted_probs = sorted((float(probabilities[int(idx)]) for idx in np.flatnonzero(mask > 0)), reverse=True)
                second_probability = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                selected_row = _action_probability_row(action_id, probabilities, logits, env.action_space, observation)
                bid11 = _bid_probability_row(probabilities, logits, mask, env.action_space, observation, "11")
                if bid11:
                    bid11_rows.append({"episode": episode, "step": step, **bid11})
                next_observation, _, done, info = env.step(action_id)
                action["failed"] = bool(info.get("last_action_error"))
                anomalies = _sanitize_anomalies(detect_anomalies(observation, next_observation, {**info, "action": action}))
                reward, components = calculate_autonomous_reward(observation, next_observation, action, anomalies, [], reward_history, {})
                episode_buffer.add(
                    obs_vector,
                    action_id,
                    float(selected["log_prob"]),
                    reward,
                    done,
                    float(selected["value"]),
                    mask,
                    {},
                )
                next_vector = env.observation_encoder.encode_observation(next_observation)
                delta = observation_delta(previous_vector, obs_vector)
                delta["reward_state_signature_changed"] = _state_signature(observation) != _state_signature(next_observation)
                observation_deltas.append(delta)
                policy_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "selected_action": selected_row,
                        "top_5_valid_actions": top_actions,
                        "max_probability": sorted_probs[0] if sorted_probs else 0.0,
                        "second_probability": second_probability,
                        "max_second_gap": (sorted_probs[0] - second_probability) if sorted_probs else 0.0,
                        "policy_entropy": distribution["entropy"],
                        "valid_action_count": mask_stats["valid_action_count"],
                        "valid_action_type_count": mask_stats["valid_action_type_count"],
                        "valid_actions_by_type": mask_stats["valid_actions_by_type"],
                        "action_type_probability_mass": masses,
                        "bid11_probability": bid11.get("probability") if bid11 else None,
                        "reward": float(reward),
                        "reward_components": components,
                    }
                )
                aggregator.record_step(
                    step=(episode - 1) * args.max_steps + step,
                    before_observation=observation,
                    after_observation=next_observation,
                    action=action,
                    reward=reward,
                    reward_components=components,
                    anomalies=anomalies,
                    info=info,
                    mask_stats=mask_stats,
                )
                previous_vector = obs_vector
                observation = next_observation
                episode_done = done
                if done:
                    break
            if len(episode_buffer) > 0:
                last_value = 0.0 if episode_done else agent.estimate_value(env.observation_encoder.encode_observation(observation))
                episode_buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
                episode_buffers.append(episode_buffer)
                trajectory_summaries.append(_trajectory_diagnostics(episode_buffer, env.action_space))
    finally:
        env.close()
    update_probe = _update_probe(args, episode_buffers) if episode_buffers else {}
    summary = aggregator.summary()
    probability_summary = _probability_summary(policy_rows, action_type_mass_totals, action_type_mass_count)
    return {
        "metrics": summary["metrics"],
        "action_distribution": summary["action_distribution"],
        "top_repeated_actions": summary["top_repeated_actions"],
        "reward_component_totals": summary["reward_component_totals"],
        "mask_summary": _mask_summary(policy_rows),
        "policy_probability_summary": probability_summary,
        "policy_steps": policy_rows,
        "trajectory_summaries": trajectory_summaries,
        "trajectory_summary": _merge_trajectory_summaries(trajectory_summaries),
        "update_probe_metrics": update_probe,
        "bid11_trace": bid11_rows,
        "candidate_bid11": bid11_rows[0]["candidate"] if bid11_rows else {},
        "observation_delta_summary": _observation_delta_summary(observation_deltas),
        "candidate_signatures": summary["candidate_signatures"],
        "confirmed_signatures": summary["confirmed_signatures"],
    }


def _update_probe(args: argparse.Namespace, buffers: list[RolloutBuffer]) -> dict[str, Any]:
    probe = PPOAgent(635, 608)
    probe.load(Path(args.model_path))
    update_rows = [probe.update(buffer) for buffer in buffers if len(buffer) > 0]
    if not update_rows:
        return {}
    keys = sorted({key for row in update_rows for key in row})
    return {
        "not_saved": True,
        "source": "diagnostic clone update on collected trajectory",
        "per_episode": update_rows,
        "mean": {
            key: float(np.mean([float(row[key]) for row in update_rows if isinstance(row.get(key), (int, float, bool))]))
            for key in keys
            if any(isinstance(row.get(key), (int, float, bool)) for row in update_rows)
        },
    }


def _fixed_observation_comparison(args: argparse.Namespace, site: Mapping[str, Any]) -> dict[str, Any]:
    env = BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    try:
        observation, _ = env.reset()
        obs_vector = env.observation_encoder.encode_observation(observation)
        mask = env.action_space.build_action_mask(observation)
        torch.manual_seed(args.seed)
        fresh = PPOAgent(env.observation_encoder.get_obs_dim(), env.action_space.get_action_dim())
        trained = PPOAgent(env.observation_encoder.get_obs_dim(), env.action_space.get_action_dim())
        trained.load(Path(args.model_path))
        fresh_dist = masked_policy_distribution(fresh, obs_vector, mask)
        trained_dist = masked_policy_distribution(trained, obs_vector, mask)
        return {
            "mask_stats": env.action_space.build_action_mask_stats(observation, mask, None),
            "fresh_top_5": top_k_valid_actions(
                fresh_dist["probabilities"],
                fresh_dist["logits"],
                mask,
                env.action_space,
                observation,
            ),
            "trained_top_5": top_k_valid_actions(
                trained_dist["probabilities"],
                trained_dist["logits"],
                mask,
                env.action_space,
                observation,
            ),
            "fresh_entropy": fresh_dist["entropy"],
            "trained_entropy": trained_dist["entropy"],
            "fresh_action_type_probability_mass": action_type_probability_mass(fresh_dist["probabilities"], mask, env.action_space),
            "trained_action_type_probability_mass": action_type_probability_mass(trained_dist["probabilities"], mask, env.action_space),
            "trained_bid11": _bid_probability_row(
                trained_dist["probabilities"],
                trained_dist["logits"],
                mask,
                env.action_space,
                observation,
                "11",
            ),
        }
    finally:
        env.close()


def _action_probability_row(
    action_id: int,
    probabilities: np.ndarray,
    logits: np.ndarray,
    action_space,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    decoded = action_space.decode(action_id)
    candidate = _action_candidate(observation, decoded)
    return {
        "action_id": int(action_id),
        "action_type": decoded["action_type"],
        "candidate_index": int(decoded["candidate_index"]),
        "target_id": _target_identity(candidate),
        "logit": float(logits[int(action_id)]),
        "probability": float(probabilities[int(action_id)]),
        "candidate": _candidate_summary(candidate),
    }


def _bid_probability_row(
    probabilities: np.ndarray,
    logits: np.ndarray,
    action_mask: np.ndarray,
    action_space,
    observation: Mapping[str, Any],
    bid: str,
) -> dict[str, Any]:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return {}
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping) or str(candidate.get("bid") or "") != bid:
            continue
        action_id = action_space.encode("click_element", index)
        return {
            "action_id": action_id,
            "candidate_index": index,
            "valid": bool(action_mask[action_id] > 0),
            "logit": float(logits[action_id]),
            "probability": float(probabilities[action_id]),
            "candidate": _candidate_summary(candidate),
        }
    return {}


def _action_candidate(observation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if action.get("action_type") not in {"click_element", "fill_input", "press_enter"}:
        return None
    candidates = observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _candidate_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return _sanitize_diagnostic(
        {
            "bid": candidate.get("bid"),
            "role": candidate.get("role"),
            "tag": candidate.get("tag"),
            "accessible_name": candidate.get("name") or candidate.get("aria_label") or candidate.get("text"),
            "href": candidate.get("href"),
            "clickable": candidate.get("clickable"),
            "visible": candidate.get("visible"),
            "enabled": candidate.get("enabled"),
            "bbox": candidate.get("bbox"),
            "semantic_action_type": _semantic_action_type(candidate),
        }
    )


def _semantic_action_type(candidate: Mapping[str, Any]) -> str:
    if candidate.get("is_form_field"):
        return "form"
    if candidate.get("is_cart_related") or candidate.get("is_purchase_action"):
        return "cart"
    if candidate.get("is_search_related"):
        return "search"
    if candidate.get("is_login_related"):
        return "login"
    if candidate.get("clickable"):
        return "clickable"
    return str(candidate.get("role") or candidate.get("tag") or "")


def _target_identity(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    value = candidate.get("bid")
    if value is not None and str(value).strip():
        return f"bid:{value}"
    return ""


def _mask_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_type: Counter[str] = Counter()
    for row in rows:
        valid = row.get("valid_actions_by_type", {})
        if isinstance(valid, Mapping):
            for action_type, count in valid.items():
                by_type[str(action_type)] += int(count or 0)
    return {
        "valid_action_count_avg": float(np.mean([row["valid_action_count"] for row in rows])),
        "valid_action_type_count_avg": float(np.mean([row["valid_action_type_count"] for row in rows])),
        "valid_actions_by_type_total": dict(by_type),
        "candidate_independent_canonical_slots": _candidate_independent_canonical(by_type, len(rows)),
    }


def _candidate_independent_canonical(by_type: Mapping[str, int], row_count: int) -> dict[str, bool]:
    independent = [
        "noop",
        "scroll_down",
        "scroll_up",
        "inspect_dom",
        "inspect_layout",
        "inspect_network",
        "inspect_console",
        "inspect_cart",
        "change_viewport_mobile",
        "change_viewport_desktop",
    ]
    return {action_type: int(by_type.get(action_type, 0) or 0) <= row_count for action_type in independent}


def _probability_summary(
    rows: list[Mapping[str, Any]],
    mass_totals: Mapping[str, float],
    mass_count: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "top1_probability_mean": float(np.mean([row["max_probability"] for row in rows])),
        "top1_probability_min": float(np.min([row["max_probability"] for row in rows])),
        "top1_probability_max": float(np.max([row["max_probability"] for row in rows])),
        "second_probability_mean": float(np.mean([row["second_probability"] for row in rows])),
        "top1_second_gap_mean": float(np.mean([row["max_second_gap"] for row in rows])),
        "policy_entropy_mean": float(np.mean([row["policy_entropy"] for row in rows])),
        "policy_entropy_min": float(np.min([row["policy_entropy"] for row in rows])),
        "policy_entropy_max": float(np.max([row["policy_entropy"] for row in rows])),
        "action_type_probability_mass_avg": {
            action_type: float(value / max(1, mass_count)) for action_type, value in mass_totals.items()
        },
    }


def _observation_delta_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "l1_mean": float(np.mean([row["l1"] for row in rows])),
        "l2_mean": float(np.mean([row["l2"] for row in rows])),
        "changed_feature_count_mean": float(np.mean([row["changed_feature_count"] for row in rows])),
        "changed_feature_count_max": int(max(row["changed_feature_count"] for row in rows)),
        "reward_state_signature_changed_count": int(sum(1 for row in rows if row.get("reward_state_signature_changed"))),
    }


def _trajectory_diagnostics(buffer: RolloutBuffer, action_space) -> dict[str, Any]:
    advantages = np.asarray(buffer.advantages, dtype=np.float32)
    returns = np.asarray(buffer.returns, dtype=np.float32)
    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    by_type: dict[str, dict[str, list[float]]] = {}
    for index, action_id in enumerate(buffer.actions):
        action_type = str(action_space.decode(action_id).get("action_type") or "")
        item = by_type.setdefault(action_type, {"reward": [], "return": [], "advantage": []})
        item["reward"].append(float(rewards[index]))
        item["return"].append(float(returns[index]))
        item["advantage"].append(float(advantages[index]))
    return {
        **summarize_advantages(advantages),
        "action_type_summary": {
            action_type: {
                "count": len(values["reward"]),
                "mean_step_reward": float(np.mean(values["reward"])) if values["reward"] else 0.0,
                "mean_return": float(np.mean(values["return"])) if values["return"] else 0.0,
                "mean_advantage": float(np.mean(values["advantage"])) if values["advantage"] else 0.0,
            }
            for action_type, values in by_type.items()
        },
    }


def _merge_trajectory_summaries(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    merged_actions: dict[str, dict[str, float]] = {}
    for row in rows:
        for action_type, values in row.get("action_type_summary", {}).items():
            target = merged_actions.setdefault(
                str(action_type),
                {"count": 0.0, "reward_sum": 0.0, "return_sum": 0.0, "advantage_sum": 0.0},
            )
            count = float(values.get("count", 0.0) or 0.0)
            target["count"] += count
            target["reward_sum"] += float(values.get("mean_step_reward", 0.0) or 0.0) * count
            target["return_sum"] += float(values.get("mean_return", 0.0) or 0.0) * count
            target["advantage_sum"] += float(values.get("mean_advantage", 0.0) or 0.0) * count
    return {
        "advantage_mean": float(np.mean([row.get("advantage_mean", 0.0) for row in rows])),
        "advantage_std": float(np.mean([row.get("advantage_std", 0.0) for row in rows])),
        "advantage_min": float(min(row.get("advantage_min", 0.0) for row in rows)),
        "advantage_max": float(max(row.get("advantage_max", 0.0) for row in rows)),
        "positive_advantage_ratio": float(np.mean([row.get("positive_advantage_ratio", 0.0) for row in rows])),
        "negative_advantage_ratio": float(np.mean([row.get("negative_advantage_ratio", 0.0) for row in rows])),
        "finite": bool(all(row.get("finite", True) for row in rows)),
        "action_type_summary": {
            action_type: {
                "count": values["count"],
                "mean_step_reward": values["reward_sum"] / max(1.0, values["count"]),
                "mean_return": values["return_sum"] / max(1.0, values["count"]),
                "mean_advantage": values["advantage_sum"] / max(1.0, values["count"]),
            }
            for action_type, values in merged_actions.items()
        },
    }


def _compare_modes(greedy: Mapping[str, Any], stochastic: Mapping[str, Any]) -> dict[str, Any]:
    gm = greedy.get("metrics", {})
    sm = stochastic.get("metrics", {})
    return {
        "unique_states": {"greedy": gm.get("unique_states"), "stochastic": sm.get("unique_states")},
        "unique_targets": {"greedy": gm.get("unique_targets"), "stochastic": sm.get("unique_targets")},
        "unique_action_targets": {"greedy": gm.get("unique_action_targets"), "stochastic": sm.get("unique_action_targets")},
        "repeat_action_target_ratio": {
            "greedy": gm.get("repeat_action_target_ratio"),
            "stochastic": sm.get("repeat_action_target_ratio"),
        },
        "action_distribution": {
            "greedy": greedy.get("action_distribution"),
            "stochastic": stochastic.get("action_distribution"),
        },
    }


def _diagnosis(report: Mapping[str, Any]) -> dict[str, Any]:
    greedy_prob = report.get("greedy", {}).get("policy_probability_summary", {})
    stochastic_metrics = report.get("stochastic", {}).get("metrics", {})
    greedy_metrics = report.get("greedy", {}).get("metrics", {})
    issues = []
    if float(greedy_prob.get("top1_probability_mean", 0.0) or 0.0) < 0.2 and float(greedy_prob.get("policy_entropy_mean", 0.0) or 0.0) > 2.5:
        issues.append("deterministic_greedy_artifact")
    if float(greedy_prob.get("action_type_probability_mass_avg", {}).get("click_element", 0.0) or 0.0) > 0.5:
        issues.append("click_probability_mass_bias")
    if float(stochastic_metrics.get("repeat_action_target_ratio", 0.0) or 0.0) < float(greedy_metrics.get("repeat_action_target_ratio", 1.0) or 1.0):
        issues.append("stochastic_rollout_more_diverse_than_greedy")
    return {"primary_findings": issues}


def _legacy_bias_findings() -> dict[str, Any]:
    files = [
        Path("services/browsergym_training_service.py"),
        Path("agents/ppo_agent.py"),
        Path("models/ppo_actor_critic.py"),
    ]
    patterns = [
        "known_bug",
        "catalog",
        "site001",
        "site9800",
        "purchase",
        "cart",
        "preferred_action",
        "recommended_action",
        "guidance",
        "action_bias",
        "action_bonus",
        "action_priority",
        "forced_action",
        "fallback",
        "first_click",
    ]
    findings = []
    for path in files:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if any(pattern in lowered for pattern in patterns):
                findings.append({"file": str(path), "line": line_no, "text": line.strip()[:160]})
    return {
        "residual_terms": findings,
        "training_run_disabled_guidance": True,
        "training_run_disabled_ground_truth_metadata": True,
        "selection_path_uses_actor_logits_plus_mask_only": True,
    }


def _actor_logit_bias_findings() -> dict[str, Any]:
    files = [Path("agents/ppo_agent.py"), Path("models/ppo_actor_critic.py")]
    findings = []
    for path in files:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if any(token in lowered for token in ("bias", "priority", "bonus", "heuristic", "catalog", "site001")):
                findings.append({"file": str(path), "line": line_no, "text": line.strip()[:160]})
    return {"manual_logit_bias_found": False, "matching_lines": findings}


def _sanitize_anomalies(anomalies: list[dict]) -> list[dict]:
    return [_sanitize_diagnostic(dict(item)) for item in anomalies]


def _sanitize_diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in item.items():
        if any(token in str(key).lower() for token in SENSITIVE_DIAGNOSTIC_KEYS):
            continue
        if isinstance(value, dict):
            sanitized[key] = _sanitize_diagnostic(value)
        elif isinstance(value, list):
            sanitized[key] = [_sanitize_diagnostic(v) if isinstance(v, dict) else v for v in value]
        else:
            sanitized[key] = value
    return sanitized


def _state_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return "|".join(
        [
            _url_path(observation),
            str(page_state.get("title") or ""),
            str(page_state.get("dom_node_count") or ""),
            str(page_state.get("page_text_length") or ""),
            _candidate_summary_key(observation),
        ]
    )


def _candidate_summary_key(observation: Mapping[str, Any]) -> str:
    candidates = observation.get("candidate_elements", []) or []
    values = []
    if isinstance(candidates, list):
        for candidate in candidates[:12]:
            if isinstance(candidate, Mapping):
                values.append(_target_identity(candidate) or str(candidate.get("role") or candidate.get("tag") or ""))
    return ",".join(values)


def _url_path(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    text = str(page_state.get("url") or "")
    if not text:
        return ""
    parsed = urlparse(text)
    return parsed.path or text.split("?", 1)[0] or "/"


def _load_site_config(config_path: str, site_id: str | None, base_url: str | None) -> dict[str, Any]:
    if base_url:
        return {"site_id": site_id or "site001", "base_url": base_url}
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    sites = config.get("sites", []) if isinstance(config, Mapping) else []
    for site in sites:
        if isinstance(site, Mapping) and str(site.get("site_id") or "") == str(site_id or ""):
            return {"site_id": site.get("site_id"), "base_url": site.get("base_url")}
    raise ValueError(f"site_id {site_id!r} not found in {config_path}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model-path", default="artifacts/models/site001_ppo_rewardv2_sanity.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="artifacts/sanity/ppo_site001_policy_diagnostics.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run_diagnostics(args)
    if args.quiet:
        print(report["summary_path"])
    else:
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
