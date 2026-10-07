"""Diagnose DQN click candidate multiplicity and repeat-action bias.

This runner is diagnostic-only. It does not change rewards, action selection,
network structure, replay sampling, or policy behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.dqn_agent import (
    FLAT_MAX_SELECTION,
    QNetwork,
    TYPE_LOGMEANEXP_SELECTION,
    balanced_expected_action_type_distribution,
    select_greedy_action,
)
from dqn_error_discovery.runners.train_dqn_error_detector import (
    calculate_dqn_reward,
    initialize_dqn_candidate_reward_baseline,
    safe_action_mask,
    sanitize_anomalies,
    _load_site_config,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


CLICK = "click_element"


def valid_count_by_action_type(mask: np.ndarray, action_space) -> dict[str, int]:
    counts = {action_type: 0 for action_type in action_space.action_types}
    for action_id in np.flatnonzero(mask > 0):
        decoded = action_space.decode(int(action_id))
        counts[str(decoded["action_type"])] += 1
    return counts


def q_statistics_by_action_type(q_values: np.ndarray, mask: np.ndarray, action_space) -> dict[str, dict[str, float]]:
    rows: dict[str, list[float]] = defaultdict(list)
    for action_id in np.flatnonzero(mask > 0):
        decoded = action_space.decode(int(action_id))
        rows[str(decoded["action_type"])].append(float(q_values[int(action_id)]))
    return {
        action_type: {
            "valid_count": float(len(values)),
            "q_max": float(np.max(values)) if values else 0.0,
            "q_mean": float(np.mean(values)) if values else 0.0,
            "q_std": float(np.std(values)) if values else 0.0,
            "q_median": float(np.median(values)) if values else 0.0,
            "max_selection_uplift": float(np.max(values) - np.mean(values)) if values else 0.0,
        }
        for action_type, values in rows.items()
    }


def expected_random_action_type_distribution(mask: np.ndarray, action_space) -> dict[str, float]:
    counts = valid_count_by_action_type(mask, action_space)
    total = sum(counts.values())
    return {action_type: (count / total if total else 0.0) for action_type, count in counts.items()}


def controlled_max_over_many_simulation(
    *,
    mean: float,
    std: float,
    candidate_counts: Sequence[int] = (1, 4, 8, 16),
    trials: int = 5000,
    seed: int = 123,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    result: dict[str, float] = {}
    for count in candidate_counts:
        if count <= 0:
            result[str(count)] = 0.0
            continue
        samples = rng.normal(loc=mean, scale=max(0.0, std), size=(trials, count))
        result[str(count)] = float(np.mean(np.max(samples, axis=1)))
    return result


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def click_candidate_rows(q_values: np.ndarray, mask: np.ndarray, action_space, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    for action_id in np.flatnonzero(mask > 0):
        decoded = action_space.decode(int(action_id))
        if decoded.get("action_type") != CLICK:
            continue
        candidate_index = int(decoded.get("candidate_index", 0))
        candidate = candidates[candidate_index] if isinstance(candidates, list) and candidate_index < len(candidates) else {}
        if not isinstance(candidate, Mapping):
            candidate = {}
        rows.append(
            {
                "action_id": int(action_id),
                "candidate_index": candidate_index,
                "bid": _safe_str(candidate.get("bid")),
                "role": _safe_str(candidate.get("role")),
                "tag": _safe_str(candidate.get("tag")),
                "name": _safe_str(candidate.get("name") or candidate.get("accessible_name") or candidate.get("text")),
                "entity_id": _safe_str(_first_present(candidate, ("entity_id", "product_id", "item_id", "cart_item_id"))),
                "href": _safe_str(candidate.get("href")),
                "semantic_action_type": semantic_action_type(candidate),
                "logical_group": logical_click_group_key(candidate),
                "q": float(q_values[int(action_id)]),
            }
        )
    rows.sort(key=lambda item: float(item["q"]), reverse=True)
    return rows


def semantic_action_type(candidate: Mapping[str, Any]) -> str:
    text = " ".join(
        _safe_str(candidate.get(key))
        for key in ("semantic_action_type", "name", "text", "aria_label", "role", "tag")
    ).lower()
    if candidate.get("is_cart_related") or candidate.get("is_purchase_action") or "cart" in text or "buy" in text:
        return "cart"
    if candidate.get("is_search_related") or "search" in text:
        return "search"
    if "login" in text or "sign in" in text:
        return "login"
    if "submit" in text or "save" in text:
        return "submit"
    if "product" in text or "detail" in text:
        return "detail"
    return "generic"


def logical_click_group_key(candidate: Mapping[str, Any]) -> str:
    for key in ("entity_id", "product_id", "item_id", "cart_item_id", "post_id", "order_id", "user_id"):
        value = candidate.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    href = _url_path_from_value(candidate.get("href"))
    if href:
        return f"href:{href}"
    name = " ".join(_safe_str(candidate.get("name") or candidate.get("text") or candidate.get("aria_label")).lower().split())
    role = _safe_str(candidate.get("role")).lower()
    tag = _safe_str(candidate.get("tag")).lower()
    parent = _safe_str(_first_present(candidate, ("parent_bid", "parent_id", "container_id", "card_id")))
    if parent:
        return f"parent:{parent}:{semantic_action_type(candidate)}"
    if name or role or tag:
        return f"semantic:{semantic_action_type(candidate)}:{role}:{tag}:{name[:80]}"
    bid = _safe_str(candidate.get("bid"))
    return f"bid:{bid}" if bid else "unknown"


def equivalent_click_group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("logical_group") or "unknown")].append(row)
    group_rows = []
    for group, items in groups.items():
        q_values = [float(item.get("q", 0.0) or 0.0) for item in items]
        group_rows.append(
            {
                "group": group,
                "candidate_count": len(items),
                "q_mean": float(np.mean(q_values)) if q_values else 0.0,
                "q_max": float(np.max(q_values)) if q_values else 0.0,
                "bids": sorted({_safe_str(item.get("bid")) for item in items if _safe_str(item.get("bid"))})[:12],
            }
        )
    group_rows.sort(key=lambda item: (int(item["candidate_count"]), float(item["q_max"])), reverse=True)
    raw_count = len(rows)
    logical_count = len(groups)
    return {
        "raw_valid_click_actions": raw_count,
        "equivalent_click_groups": logical_count,
        "duplication_factor": float(raw_count / logical_count) if logical_count else 0.0,
        "groups": group_rows,
    }


def repeat_diagnostics(action_targets: Sequence[str], *, window: int = 4, max_cycle: int = 3) -> dict[str, Any]:
    immediate = 0
    recent = 0
    cycle = 0
    blind_spot = 0
    cycle_examples: Counter[str] = Counter()
    for index, target in enumerate(action_targets):
        is_immediate = index > 0 and target == action_targets[index - 1]
        is_recent = index > 0 and target in action_targets[max(0, index - window):index]
        is_cycle = False
        for length in range(1, max_cycle + 1):
            if index + 1 >= 2 * length:
                left = action_targets[index + 1 - 2 * length:index + 1 - length]
                right = action_targets[index + 1 - length:index + 1]
                if list(left) == list(right):
                    is_cycle = True
                    cycle_examples[" ".join(right)] += 1
                    break
        immediate += int(is_immediate)
        recent += int(is_recent)
        cycle += int(is_cycle)
        blind_spot += int(is_recent and not is_immediate)
    denominator = len(action_targets)
    return {
        "total": denominator,
        "immediate_repeat_count": immediate,
        "recent_window_repeat_count": recent,
        "cycle_repeat_count": cycle,
        "repeat_blind_spot_count": blind_spot,
        "immediate_repeat_ratio": immediate / denominator if denominator else 0.0,
        "recent_window_repeat_ratio": recent / denominator if denominator else 0.0,
        "cycle_repeat_ratio": cycle / denominator if denominator else 0.0,
        "repeat_blind_spot_ratio": blind_spot / denominator if denominator else 0.0,
        "cycle_examples": dict(cycle_examples.most_common(10)),
    }


def candidate_reward_signature_dedup(signatures: Sequence[str]) -> dict[str, Any]:
    counts = Counter(str(signature) for signature in signatures if signature)
    return {
        "unique_signature_count": len(counts),
        "repeated_signature_count": sum(max(0, count - 1) for count in counts.values()),
        "top_repeated_signatures": dict(counts.most_common(10)),
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    env = BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    action_space = env.action_space
    network = QNetwork(env.observation_encoder.get_obs_dim(), action_space.get_action_dim())
    checkpoint = torch.load(Path(args.model_path), map_location="cpu")
    network.load_state_dict(checkpoint.get("model", checkpoint))
    network.eval()

    step_records: list[dict[str, Any]] = []
    click_candidate_distributions: list[dict[str, Any]] = []
    action_targets: list[str] = []
    all_click_candidate_rows: list[dict[str, Any]] = []
    selected_clicks: list[dict[str, Any]] = []
    click_candidate_count_series: list[float] = []
    click_max_q_series: list[float] = []
    expected_random_rows: list[dict[str, float]] = []
    balanced_expected_rows: list[dict[str, float]] = []
    history: dict[str, Any] = {}

    try:
        for episode in range(1, args.episodes + 1):
            raw, _ = env.reset()
            state = env.observation_encoder.encode_observation(raw)
            history = {}
            initialize_dqn_candidate_reward_baseline(
                history,
                sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}})),
            )
            for step in range(1, args.max_steps + 1):
                mask = safe_action_mask(action_space, raw)
                with torch.no_grad():
                    q_values = network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                q_stats = q_statistics_by_action_type(q_values, mask, action_space)
                valid_counts = valid_count_by_action_type(mask, action_space)
                expected_random = expected_random_action_type_distribution(mask, action_space)
                expected_random_rows.append(expected_random)
                balanced_expected = balanced_expected_action_type_distribution(mask, action_space)
                balanced_expected_rows.append(balanced_expected)
                click_rows = click_candidate_rows(q_values, mask, action_space, raw)
                click_summary = equivalent_click_group_summary(click_rows)
                if click_rows:
                    click_candidate_count_series.append(float(len(click_rows)))
                    click_max_q_series.append(float(click_rows[0]["q"]))
                action_id = select_greedy_action(
                    network,
                    state,
                    mask,
                    action_space,
                    mode=args.greedy_selection_mode,
                )
                decoded = action_space.decode(action_id)
                action_type = str(decoded.get("action_type") or "")
                candidate_index = int(decoded.get("candidate_index", 0))
                candidate = _candidate_at(raw, candidate_index) if action_type == CLICK else {}
                action_target = f"{action_type}:{_target_identity(candidate) or candidate_index}"
                action_targets.append(action_target)
                next_raw, _, done, info = env.step(action_id)
                decoded["failed"] = bool(info.get("last_action_error"))
                decoded["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": decoded}))
                reward, components = calculate_dqn_reward(raw, next_raw, decoded, anomalies, history)
                selected_row = next((row for row in click_rows if int(row["action_id"]) == int(action_id)), None)
                if selected_row:
                    selected_clicks.append({**selected_row, "episode": episode, "step": step, "reward": reward})
                all_click_candidate_rows.extend({**row, "episode": episode, "step": step} for row in click_rows)
                click_candidate_distributions.append(
                    {
                        "episode": episode,
                        "step": step,
                        "click_candidates": click_rows[:12],
                        "equivalent_groups": click_summary,
                    }
                )
                step_records.append(
                    {
                        "episode": episode,
                        "step": step,
                        "action_id": int(action_id),
                        "action_type": action_type,
                        "candidate_index": candidate_index,
                        "action_target": action_target,
                        "valid_count_by_action_type": valid_counts,
                        "q_statistics_by_action_type": q_stats,
                        "expected_random_action_type_distribution": expected_random,
                        "balanced_expected_action_type_distribution": balanced_expected,
                        "selected_q": float(q_values[int(action_id)]),
                        "reward": float(reward),
                        "reward_components": components,
                        "candidate_anomalies": [
                            {"rule_id": item.get("rule_id"), "signature": item.get("signature"), "status": item.get("status")}
                            for item in anomalies
                            if item.get("status") == "candidate"
                        ],
                    }
                )
                raw = next_raw
                state = env.observation_encoder.encode_observation(raw)
                if done:
                    break
    finally:
        env.close()

    manifest = _load_manifest(args.model_path)
    training = training_distribution_summary(manifest)
    replay_distribution = dict(training.get("action_distribution", {}) or {})
    report = {
        "model_path": args.model_path,
        "greedy_selection_mode": args.greedy_selection_mode,
        "type_score_method": "normalized_logmeanexp" if args.greedy_selection_mode == TYPE_LOGMEANEXP_SELECTION else None,
        "site_id": site["site_id"],
        "url": site["base_url"],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "valid_count_by_action_type": summarize_valid_counts(step_records),
        "q_statistics_by_action_type": summarize_q_stats(step_records),
        "click_candidate_q_distribution": click_candidate_distribution_summary(all_click_candidate_rows),
        "candidate_count_vs_maxq": {
            "samples": len(click_candidate_count_series),
            "pearson": pearson_correlation(click_candidate_count_series, click_max_q_series),
            "click_candidate_count_mean": _mean(click_candidate_count_series),
            "click_max_q_mean": _mean(click_max_q_series),
        },
        "controlled_max_over_many_simulation": controlled_max_over_many_simulation(
            mean=_mean([float(row.get("q", 0.0) or 0.0) for row in all_click_candidate_rows]),
            std=_std([float(row.get("q", 0.0) or 0.0) for row in all_click_candidate_rows]),
        ),
        "max_selection_uplift": summarize_uplift(step_records),
        "greedy_click_target_top_ranking": {
            "selected_clicks": selected_clicks[:80],
            "top_selected_click_targets": dict(Counter(str(row.get("bid") or row.get("logical_group")) for row in selected_clicks).most_common(10)),
            "top1_click_candidate_by_step": [
                {
                    "episode": row["episode"],
                    "step": row["step"],
                    "bid": row["click_candidates"][0].get("bid") if row["click_candidates"] else "",
                    "logical_group": row["click_candidates"][0].get("logical_group") if row["click_candidates"] else "",
                    "q": row["click_candidates"][0].get("q") if row["click_candidates"] else None,
                }
                for row in click_candidate_distributions
            ],
        },
        "random_vs_greedy_distribution": training,
        "expected_random_action_type_distribution": summarize_expected_random(expected_random_rows),
        "balanced_expected_action_type_distribution": summarize_expected_random(balanced_expected_rows),
        "before_after_expected_random_distribution": {
            "flattened_uniform": summarize_expected_random(expected_random_rows),
            "action_type_balanced": summarize_expected_random(balanced_expected_rows),
        },
        "replay_distribution": replay_distribution,
        "sampled_replay_distribution": sampled_distribution_from_counts(replay_distribution, args.batch_size, args.sample_batches),
        "click_reward_components": click_reward_breakdown(step_records, manifest),
        "click_candidate_anomaly_summary": click_candidate_anomaly_summary(step_records),
        "raw_vs_logical_click_candidates": summarize_equivalent_groups(click_candidate_distributions),
        "equivalent_click_groups": top_equivalent_groups(all_click_candidate_rows),
        "immediate_repeat": repeat_diagnostics(action_targets, window=args.repeat_window),
        "recent_window_repeat": repeat_diagnostics(action_targets, window=args.repeat_window),
        "cycle_repeat": repeat_diagnostics(action_targets, window=args.repeat_window),
        "root_cause_summary": root_cause_summary(step_records, training, all_click_candidate_rows, replay_distribution),
        "step_records": step_records[:120],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def summarize_valid_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for action_type, count in (record.get("valid_count_by_action_type", {}) or {}).items():
            rows[str(action_type)].append(float(count or 0.0))
    return {
        action_type: {"mean": _mean(values), "min": float(np.min(values)), "max": float(np.max(values))}
        for action_type, values in rows.items()
        if values
    }


def summarize_q_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for action_type, stats in (record.get("q_statistics_by_action_type", {}) or {}).items():
            for key in ("valid_count", "q_max", "q_mean", "q_std", "q_median", "max_selection_uplift"):
                rows[str(action_type)][key].append(float(stats.get(key, 0.0) or 0.0))
    return {
        action_type: {f"{key}_mean": _mean(values) for key, values in stats.items()}
        for action_type, stats in rows.items()
    }


def summarize_uplift(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for action_type, stats in (record.get("q_statistics_by_action_type", {}) or {}).items():
            rows[str(action_type)].append(float(stats.get("max_selection_uplift", 0.0) or 0.0))
    return {action_type: {"mean": _mean(values), "max": float(np.max(values)) if values else 0.0} for action_type, values in rows.items()}


def click_candidate_distribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q_values = [float(row.get("q", 0.0) or 0.0) for row in rows]
    return {
        "raw_candidate_observations": len(rows),
        "q_mean": _mean(q_values),
        "q_std": _std(q_values),
        "q_max": float(np.max(q_values)) if q_values else 0.0,
        "q_median": float(np.median(q_values)) if q_values else 0.0,
        "top_bids_by_q": [
            {"bid": row.get("bid"), "logical_group": row.get("logical_group"), "q": row.get("q")}
            for row in sorted(rows, key=lambda item: float(item.get("q", 0.0) or 0.0), reverse=True)[:20]
        ],
    }


def summarize_expected_random(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for action_type, probability in row.items():
            by_type[str(action_type)].append(float(probability or 0.0))
    return {action_type: _mean(values) for action_type, values in by_type.items()}


def training_distribution_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    action_distribution = dict(manifest.get("action_distribution", {}) or {})
    random_distribution = dict(manifest.get("random_action_distribution", {}) or {})
    greedy_distribution = dict(manifest.get("greedy_action_distribution", {}) or {})
    episode_summaries = manifest.get("episode_summaries", []) if isinstance(manifest.get("episode_summaries"), list) else []
    total = sum(int(value or 0) for value in action_distribution.values())
    random_total = sum(int(value or 0) for value in random_distribution.values())
    greedy_total = sum(int(value or 0) for value in greedy_distribution.values())
    return {
        "action_distribution": action_distribution,
        "action_ratios": _ratios(action_distribution),
        "episode_click_ratios": [
            {
                "episode": item.get("episode"),
                "click_ratio": _ratio((item.get("action_distribution", {}) or {}).get(CLICK, 0), item.get("steps", 0)),
            }
            for item in episode_summaries
            if isinstance(item, Mapping)
        ],
        "random_action_distribution": random_distribution,
        "greedy_action_distribution": greedy_distribution,
        "random_click_ratio": _ratio(random_distribution.get(CLICK, 0), random_total),
        "greedy_click_ratio": _ratio(greedy_distribution.get(CLICK, 0), greedy_total),
        "training_click_ratio": _ratio(action_distribution.get(CLICK, 0), total),
    }


def sampled_distribution_from_counts(counts: Mapping[str, Any], batch_size: int, sample_batches: int) -> dict[str, Any]:
    expanded: list[str] = []
    for action_type, count in counts.items():
        expanded.extend([str(action_type)] * int(count or 0))
    if not expanded:
        return {"counts": {}, "ratios": {}}
    rng = random.Random(123)
    sampled: Counter[str] = Counter()
    size = min(max(1, batch_size), len(expanded))
    for _ in range(max(1, sample_batches)):
        sampled.update(rng.sample(expanded, size))
    return {"counts": dict(sampled), "ratios": _ratios(sampled)}


def click_reward_breakdown(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    click_records = [record for record in records if record.get("action_type") == CLICK]
    totals: Counter[str] = Counter()
    for record in click_records:
        for key, value in (record.get("reward_components", {}) or {}).items():
            totals[str(key)] += float(value or 0.0)
    rewards = [float(record.get("reward", 0.0) or 0.0) for record in click_records]
    training_summary = (manifest.get("action_type_reward_summary", {}) or {}).get(CLICK, {}) if isinstance(manifest, Mapping) else {}
    return {
        "eval_count": len(click_records),
        "eval_component_totals": dict(totals),
        "eval_reward_mean": _mean(rewards),
        "eval_positive_ratio": _ratio(sum(1 for value in rewards if value > 0), len(rewards)),
        "eval_zero_ratio": _ratio(sum(1 for value in rewards if value == 0), len(rewards)),
        "eval_negative_ratio": _ratio(sum(1 for value in rewards if value < 0), len(rewards)),
        "training_action_type_reward_summary": training_summary,
    }


def click_candidate_anomaly_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rule_counts: Counter[str] = Counter()
    signatures: list[str] = []
    positive_click_with_candidate = 0
    click_count = 0
    for record in records:
        if record.get("action_type") != CLICK:
            continue
        click_count += 1
        anomalies = record.get("candidate_anomalies", []) or []
        if anomalies and float(record.get("reward", 0.0) or 0.0) > 0:
            positive_click_with_candidate += 1
        for anomaly in anomalies:
            rule_counts[str(anomaly.get("rule_id") or "")] += 1
            signatures.append(str(anomaly.get("signature") or ""))
    return {
        "click_count": click_count,
        "candidate_rule_counts": dict(rule_counts),
        "signature_dedup": candidate_reward_signature_dedup(signatures),
        "positive_click_with_candidate_anomaly_ratio": _ratio(positive_click_with_candidate, click_count),
    }


def summarize_equivalent_groups(step_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    raw_counts = []
    group_counts = []
    factors = []
    for row in step_rows:
        summary = row.get("equivalent_groups", {}) or {}
        raw = float(summary.get("raw_valid_click_actions", 0.0) or 0.0)
        groups = float(summary.get("equivalent_click_groups", 0.0) or 0.0)
        raw_counts.append(raw)
        group_counts.append(groups)
        factors.append(float(summary.get("duplication_factor", 0.0) or 0.0))
    return {
        "raw_click_candidate_count_mean": _mean(raw_counts),
        "raw_click_candidate_count_min": float(np.min(raw_counts)) if raw_counts else 0.0,
        "raw_click_candidate_count_max": float(np.max(raw_counts)) if raw_counts else 0.0,
        "logical_click_target_count_mean": _mean(group_counts),
        "duplication_factor_mean": _mean(factors),
        "duplication_factor_max": float(np.max(factors)) if factors else 0.0,
    }


def top_equivalent_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("logical_group") or "unknown")].append(row)
    result = []
    for group, items in groups.items():
        q_values = [float(item.get("q", 0.0) or 0.0) for item in items]
        result.append(
            {
                "group": group,
                "observations": len(items),
                "unique_bids": sorted({_safe_str(item.get("bid")) for item in items if _safe_str(item.get("bid"))})[:20],
                "q_mean": _mean(q_values),
                "q_max": float(np.max(q_values)) if q_values else 0.0,
            }
        )
    result.sort(key=lambda item: (int(item["observations"]), float(item["q_max"])), reverse=True)
    return result[:20]


def root_cause_summary(
    records: Sequence[Mapping[str, Any]],
    training: Mapping[str, Any],
    click_rows: Sequence[Mapping[str, Any]],
    replay_distribution: Mapping[str, Any],
) -> list[str]:
    causes = []
    q_summary = summarize_q_stats(records)
    click_q = q_summary.get(CLICK, {})
    if float(click_q.get("valid_count_mean", 0.0) or 0.0) > 4 and float(click_q.get("max_selection_uplift_mean", 0.0) or 0.0) > 0.1:
        causes.append("candidate_multiplicity_max_over_many_bias")
    if float(training.get("random_click_ratio", 0.0) or 0.0) >= 0.4:
        causes.append("flattened_random_exploration_click_bias")
    if float(training.get("training_click_ratio", 0.0) or 0.0) >= 0.5:
        causes.append("replay_click_overrepresentation")
    groups = top_equivalent_groups(click_rows)
    if any(len(item.get("unique_bids", [])) > 1 for item in groups):
        causes.append("logical_duplicate_click_candidates")
    repeat = repeat_diagnostics([str(record.get("action_target") or "") for record in records])
    if repeat["repeat_blind_spot_ratio"] > 0.1:
        causes.append("repeat_criterion_blind_spot")
    click_replay = _ratio(replay_distribution.get(CLICK, 0), sum(int(value or 0) for value in replay_distribution.values()))
    if click_replay >= 0.5:
        causes.append("buffer_distribution_dominated_by_click")
    return causes


def _load_manifest(model_path: str) -> dict[str, Any]:
    manifest_path = Path(model_path).with_suffix(".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    manifest = checkpoint.get("manifest", {}) if isinstance(checkpoint, Mapping) else {}
    return dict(manifest) if isinstance(manifest, Mapping) else {}


def _candidate_at(observation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return {}


def _target_identity(candidate: Mapping[str, Any]) -> str:
    bid = _safe_str(candidate.get("bid"))
    return f"bid:{bid}" if bid else logical_click_group_key(candidate)


def _first_present(candidate: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = candidate.get(key)
        if value not in (None, ""):
            return value
    return ""


def _url_path_from_value(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    parsed = urlparse(text)
    return parsed.path or text.split("?", 1)[0] or "/"


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator or 0.0)
        return float(numerator or 0.0) / den if den else 0.0
    except (TypeError, ValueError):
        return 0.0


def _ratios(counts: Mapping[str, Any]) -> dict[str, float]:
    total = sum(int(value or 0) for value in counts.values())
    return {str(key): _ratio(value, total) for key, value in counts.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model-path", default="artifacts/models/site001_dqn_functionalstate_sanity.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-batches", type=int, default=8)
    parser.add_argument("--repeat-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_click_bias_diagnostics.json")
    parser.add_argument(
        "--greedy-selection-mode",
        choices=[FLAT_MAX_SELECTION, TYPE_LOGMEANEXP_SELECTION],
        default=FLAT_MAX_SELECTION,
    )
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_diagnostics(args)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
