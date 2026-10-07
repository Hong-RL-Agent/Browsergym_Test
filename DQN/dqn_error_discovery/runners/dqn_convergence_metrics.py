"""Aggregate DQN 10/20/30 episode convergence diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def aggregate_training_block(episode_summaries: Sequence[Mapping[str, Any]], start: int, end: int) -> dict[str, Any]:
    items = [
        item for item in episode_summaries
        if start <= int(item.get("episode", 0) or 0) <= end
    ]
    action_distribution: Counter[str] = Counter()
    random_distribution: Counter[str] = Counter()
    greedy_distribution: Counter[str] = Counter()
    random_count = 0
    greedy_count = 0
    total_actions = 0
    click_reward_sum = 0.0
    click_reward_count = 0
    click_positive_count = 0
    click_zero_count = 0
    click_negative_count = 0
    click_component_totals: Counter[str] = Counter()
    click_component_events: Counter[str] = Counter()
    for item in items:
        action_distribution.update(_int_counts(item.get("action_distribution", {})))
        random_distribution.update(_int_counts(item.get("random_action_distribution", {})))
        greedy_distribution.update(_int_counts(item.get("greedy_action_distribution", {})))
        random_count += int(item.get("random_action_count", 0) or 0)
        greedy_count += int(item.get("greedy_action_count", 0) or 0)
        total_actions += int(item.get("steps", 0) or 0)
        click_summary = _mapping(item.get("action_type_reward_summary", {})).get("click_element", {})
        if isinstance(click_summary, Mapping):
            count = int(float(click_summary.get("count", 0.0) or 0.0))
            mean = float(click_summary.get("reward_mean", 0.0) or 0.0)
            click_reward_count += count
            click_reward_sum += mean * count
            click_positive_count += int(float(click_summary.get("positive_transition_count", 0.0) or 0.0))
            click_zero_count += int(float(click_summary.get("zero_transition_count", 0.0) or 0.0))
            click_negative_count += int(float(click_summary.get("negative_transition_count", 0.0) or 0.0))
        click_component_totals.update(_float_counts(_mapping(item.get("action_type_reward_component_totals", {})).get("click_element", {})))
        click_component_events.update(_int_counts(_mapping(item.get("action_type_reward_component_event_counts", {})).get("click_element", {})))
    return {
        "episode_range": [start, end],
        "total_actions": total_actions,
        "random_action_count": random_count,
        "greedy_action_count": greedy_count,
        "random_action_distribution": dict(random_distribution),
        "greedy_action_distribution": dict(greedy_distribution),
        "overall_action_distribution": dict(action_distribution),
        "random_click_ratio": _ratio(random_distribution.get("click_element", 0), random_count),
        "greedy_click_ratio": _ratio(greedy_distribution.get("click_element", 0), greedy_count),
        "overall_click_ratio": _ratio(action_distribution.get("click_element", 0), total_actions),
        "action_type_entropy": distribution_entropy(action_distribution),
        "click_reward": {
            "count": click_reward_count,
            "mean": _ratio(click_reward_sum, click_reward_count),
            "positive_ratio": _ratio(click_positive_count, click_reward_count),
            "zero_ratio": _ratio(click_zero_count, click_reward_count),
            "negative_ratio": _ratio(click_negative_count, click_reward_count),
            "component_totals": dict(click_component_totals),
            "component_event_counts": dict(click_component_events),
        },
    }


def block_training_summary(manifest: Mapping[str, Any], blocks: Sequence[tuple[int, int]] = ((1, 10), (11, 20), (21, 30))) -> dict[str, Any]:
    episodes = manifest.get("episode_summaries", [])
    if not isinstance(episodes, list):
        episodes = []
    return {f"EP{start}-{end}": aggregate_training_block(episodes, start, end) for start, end in blocks}


def loss_trajectory(manifest: Mapping[str, Any], blocks: Sequence[tuple[int, int]] = ((1, 10), (11, 20), (21, 30))) -> dict[str, Any]:
    history = manifest.get("loss_history", [])
    if not isinstance(history, list):
        history = []
    result = {}
    for start, end in blocks:
        rows = [row for row in history if start <= int(float(row.get("episode", 0) or 0)) <= end]
        losses = [float(row.get("loss", 0.0) or 0.0) for row in rows]
        td_errors = [float(row.get("td_error_abs_mean", 0.0) or 0.0) for row in rows]
        result[f"EP{start}-{end}"] = {
            "loss": numeric_summary(losses),
            "td_error_abs_mean": numeric_summary(td_errors),
            "nan_or_inf": any(not np.isfinite(value) for value in losses + td_errors),
        }
    return result


def replay_click_ratio(manifest: Mapping[str, Any]) -> float:
    distribution = manifest.get("action_distribution", {})
    if not isinstance(distribution, Mapping):
        return 0.0
    total = sum(int(value or 0) for value in distribution.values())
    return _ratio(distribution.get("click_element", 0), total)


def sampled_click_ratio(sampled_distribution: Mapping[str, Any]) -> float:
    counts = sampled_distribution.get("counts", sampled_distribution)
    if not isinstance(counts, Mapping):
        return 0.0
    total = sum(int(value or 0) for value in counts.values())
    return _ratio(counts.get("click_element", 0), total)


def convergence_summary(
    *,
    manifest: Mapping[str, Any],
    checkpoint_reports: Mapping[str, Mapping[str, Any]],
    sanity_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    q_trajectory = {}
    click_uplift = {}
    replay = {}
    greedy_eval = {}
    repeat = {}
    top_click = {}
    for label, report in checkpoint_reports.items():
        q_stats = report.get("q_statistics_by_action_type", {})
        click = q_stats.get("click_element", {}) if isinstance(q_stats, Mapping) else {}
        q_trajectory[label] = {
            "click_q_max": click.get("q_max_mean"),
            "click_q_mean": click.get("q_mean_mean"),
            "click_q_std": click.get("q_std_mean"),
            "click_q_median": click.get("q_median_mean"),
            "scroll_down_q": _action_q(report, "scroll_down"),
            "fill_input_q": _action_q(report, "fill_input"),
            "inspect_dom_q": _action_q(report, "inspect_dom"),
            "inspect_console_q": _action_q(report, "inspect_console"),
        }
        click_uplift[label] = click.get("max_selection_uplift_mean")
        training = report.get("random_vs_greedy_distribution", {})
        replay[label] = {
            "replay_click_ratio": training.get("training_click_ratio") if isinstance(training, Mapping) else None,
            "sampled_replay_click_ratio": sampled_click_ratio(report.get("sampled_replay_distribution", {})),
        }
        immediate = report.get("immediate_repeat", {})
        repeat[label] = {
            "immediate_repeat_ratio": immediate.get("immediate_repeat_ratio") if isinstance(immediate, Mapping) else None,
            "recent_window_repeat_ratio": immediate.get("recent_window_repeat_ratio") if isinstance(immediate, Mapping) else None,
            "cycle_repeat_ratio": immediate.get("cycle_repeat_ratio") if isinstance(immediate, Mapping) else None,
            "recent_minus_immediate": (
                float(immediate.get("recent_window_repeat_ratio", 0.0) or 0.0)
                - float(immediate.get("immediate_repeat_ratio", 0.0) or 0.0)
            ) if isinstance(immediate, Mapping) else None,
        }
        top_click[label] = (report.get("greedy_click_target_top_ranking", {}) or {}).get("top_selected_click_targets", {})
    for label, sanity in sanity_reports.items():
        metrics = sanity.get("metrics", {})
        actions = sanity.get("action_distribution", {})
        total = int(metrics.get("total_steps", 0) or 0) if isinstance(metrics, Mapping) else 0
        greedy_eval[label] = {
            "action_distribution": actions,
            "click_ratio": _ratio((actions or {}).get("click_element", 0), total),
            "repeat_action_ratio": metrics.get("repeat_local_action_ratio") if isinstance(metrics, Mapping) else None,
            "unique_local_action_targets": metrics.get("unique_local_action_targets") if isinstance(metrics, Mapping) else None,
            "candidate": metrics.get("candidate_anomaly_count") if isinstance(metrics, Mapping) else None,
            "confirmed": metrics.get("confirmed_anomaly_count") if isinstance(metrics, Mapping) else None,
            "action_error": metrics.get("action_error_count") if isinstance(metrics, Mapping) else None,
        }
    return {
        "epsilon_trajectory": manifest.get("epsilon_trajectory", []),
        "training_action_distribution_by_block": block_training_summary(manifest),
        "loss_trajectory": loss_trajectory(manifest),
        "replay_distribution_by_checkpoint": replay,
        "q_trajectory": q_trajectory,
        "click_uplift_trajectory": click_uplift,
        "greedy_eval_trajectory": greedy_eval,
        "repeat_trajectory": repeat,
        "top_click_target_trajectory": top_click,
        "click_reward_trajectory": {
            label: block.get("click_reward", {})
            for label, block in block_training_summary(manifest).items()
        },
        "root_cause_summary": classify_convergence(q_trajectory, greedy_eval, click_uplift),
    }


def classify_convergence(q_trajectory: Mapping[str, Any], greedy_eval: Mapping[str, Any], uplift: Mapping[str, Any]) -> list[str]:
    causes = []
    ep30_q = q_trajectory.get("EP30", {}) if isinstance(q_trajectory, Mapping) else {}
    ep30_eval = greedy_eval.get("EP30", {}) if isinstance(greedy_eval, Mapping) else {}
    if float(ep30_eval.get("click_ratio", 0.0) or 0.0) >= 0.9 and float(ep30_q.get("click_q_mean", 0.0) or 0.0) < 0.0:
        causes.append("structural_max_over_many_bias")
    if float(uplift.get("EP30", 0.0) or 0.0) > 0.15:
        causes.append("persistent_click_max_selection_uplift")
    if float(ep30_eval.get("repeat_action_ratio", 0.0) or 0.0) >= 0.8:
        causes.append("repeat_behavior_remains_high")
    return causes


def numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def distribution_entropy(counts: Mapping[str, Any]) -> float:
    total = sum(int(value or 0) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        probability = float(value or 0) / total
        if probability > 0.0:
            entropy -= probability * float(np.log(probability))
    return entropy


def write_convergence_summary(
    output: str | Path,
    *,
    manifest_path: str | Path,
    checkpoint_report_paths: Mapping[str, str | Path],
    sanity_report_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    checkpoint_reports = {
        label: json.loads(Path(path).read_text(encoding="utf-8"))
        for label, path in checkpoint_report_paths.items()
    }
    sanity_reports = {
        label: json.loads(Path(path).read_text(encoding="utf-8"))
        for label, path in sanity_report_paths.items()
    }
    summary = convergence_summary(manifest=manifest, checkpoint_reports=checkpoint_reports, sanity_reports=sanity_reports)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _action_q(report: Mapping[str, Any], action_type: str) -> Any:
    q_stats = report.get("q_statistics_by_action_type", {})
    if not isinstance(q_stats, Mapping):
        return None
    values = q_stats.get(action_type, {})
    return values.get("q_max_mean") if isinstance(values, Mapping) else None


def _int_counts(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        counter[str(key)] = int(count or 0)
    return counter


def _float_counts(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        counter[str(key)] = float(count or 0.0)
    return counter


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator or 0.0)
        return float(numerator or 0.0) / den if den else 0.0
    except (TypeError, ValueError):
        return 0.0
