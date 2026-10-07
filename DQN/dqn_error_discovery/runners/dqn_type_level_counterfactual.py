"""Counterfactual DQN action selection diagnostics.

This module compares the production flattened greedy argmax with diagnostic
type-level selectors. It does not change training, rewards, replay, masks, or
the production DQN action selection path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.dqn_agent import QNetwork, q_value_diagnostics
from dqn_error_discovery.runners.dqn_click_bias_diagnostics import (
    CLICK,
    click_candidate_rows,
    controlled_max_over_many_simulation,
    equivalent_click_group_summary,
    q_statistics_by_action_type,
    repeat_diagnostics,
    valid_count_by_action_type,
)
from dqn_error_discovery.runners.run_policy_sanity_check import SanityMetricsAggregator
from dqn_error_discovery.runners.train_dqn_error_detector import (
    calculate_dqn_reward,
    initialize_dqn_candidate_reward_baseline,
    safe_action_mask,
    sanitize_anomalies,
    _load_site_config,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


FLAT_MAX = "FLAT_MAX"
TYPE_MEAN_MAX = "TYPE_MEAN_MAX"
TYPE_MEDIAN_MAX = "TYPE_MEDIAN_MAX"
TYPE_TOPK_MEAN_MAX = "TYPE_TOPK_MEAN_MAX"
SELECTION_MODES = (FLAT_MAX, TYPE_MEAN_MAX, TYPE_MEDIAN_MAX, TYPE_TOPK_MEAN_MAX)


def grouped_valid_q(q_values: np.ndarray, mask: np.ndarray, action_space) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action_id in np.flatnonzero(mask > 0):
        decoded = action_space.decode(int(action_id))
        groups[str(decoded["action_type"])].append(
            {
                "action_id": int(action_id),
                "candidate_index": int(decoded["candidate_index"]),
                "q": float(q_values[int(action_id)]),
            }
        )
    for rows in groups.values():
        rows.sort(key=lambda row: (float(row["q"]), -int(row["action_id"])), reverse=True)
    return dict(groups)


def score_action_types(
    q_values: np.ndarray,
    mask: np.ndarray,
    action_space,
    *,
    mode: str,
    top_k: int = 3,
) -> dict[str, dict[str, float]]:
    groups = grouped_valid_q(q_values, mask, action_space)
    scores: dict[str, dict[str, float]] = {}
    for action_type, rows in groups.items():
        values = [float(row["q"]) for row in rows]
        if mode == FLAT_MAX:
            score = float(np.max(values))
        elif mode == TYPE_MEAN_MAX:
            score = float(np.mean(values))
        elif mode == TYPE_MEDIAN_MAX:
            score = float(np.median(values))
        elif mode == TYPE_TOPK_MEAN_MAX:
            selected = sorted(values, reverse=True)[: max(1, min(top_k, len(values)))]
            score = float(np.mean(selected))
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        scores[action_type] = {
            "score": score,
            "valid_count": float(len(values)),
            "q_max": float(np.max(values)),
            "q_mean": float(np.mean(values)),
            "q_std": float(np.std(values)),
            "q_median": float(np.median(values)),
            "max_selection_uplift": float(np.max(values) - np.mean(values)),
        }
    return scores


def select_counterfactual_action(
    q_values: np.ndarray,
    mask: np.ndarray,
    action_space,
    *,
    mode: str,
    top_k: int = 3,
) -> dict[str, Any]:
    scores = score_action_types(q_values, mask, action_space, mode=mode, top_k=top_k)
    if not scores:
        action_id = int(action_space.encode("noop", 0)) if hasattr(action_space, "encode") else 0
        return {"mode": mode, "action_id": action_id, "action_type": "noop", "candidate_index": 0, "type_scores": {}}
    selected_type = max(scores, key=lambda action_type: (scores[action_type]["score"], action_type))
    groups = grouped_valid_q(q_values, mask, action_space)
    selected_row = groups[selected_type][0]
    decoded = action_space.decode(int(selected_row["action_id"]))
    return {
        "mode": mode,
        "action_id": int(selected_row["action_id"]),
        "action_type": str(decoded["action_type"]),
        "candidate_index": int(decoded["candidate_index"]),
        "q": float(selected_row["q"]),
        "type_score": float(scores[selected_type]["score"]),
        "type_scores": scores,
    }


def offline_counterfactual_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, Counter[str]] = {mode: Counter() for mode in SELECTION_MODES}
    disagreements: Counter[str] = Counter()
    for record in records:
        selections = record.get("counterfactual_selections", {})
        flat = selections.get(FLAT_MAX, {}) if isinstance(selections, Mapping) else {}
        flat_type = str(flat.get("action_type") or "")
        for mode in SELECTION_MODES:
            item = selections.get(mode, {}) if isinstance(selections, Mapping) else {}
            action_type = str(item.get("action_type") or "")
            if action_type:
                by_mode[mode][action_type] += 1
            if mode != FLAT_MAX and action_type and action_type != flat_type:
                disagreements[f"{mode}_vs_flat"] += 1
    total = len(records)
    return {
        "total_observations": total,
        "action_distribution_by_mode": {mode: dict(counter) for mode, counter in by_mode.items()},
        "click_ratio_by_mode": {
            mode: _ratio(counter.get(CLICK, 0), total)
            for mode, counter in by_mode.items()
        },
        "disagreement_with_flat": {
            key: _ratio(value, total)
            for key, value in disagreements.items()
        },
    }


def run_counterfactual_rollout(args: argparse.Namespace, mode: str) -> dict[str, Any]:
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
    network = _load_network(args.model_path, env.observation_encoder.get_obs_dim(), action_space.get_action_dim())
    aggregator = SanityMetricsAggregator("dqn")
    action_targets: list[str] = []
    records: list[dict[str, Any]] = []
    click_rows_all: list[dict[str, Any]] = []
    history: dict[str, Any] = {}

    try:
        global_step = 0
        for episode in range(1, args.episodes + 1):
            raw, _ = env.reset()
            state = env.observation_encoder.encode_observation(raw)
            history = {}
            initialize_dqn_candidate_reward_baseline(
                history,
                sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}})),
            )
            for step in range(1, args.max_steps + 1):
                global_step += 1
                mask = safe_action_mask(action_space, raw)
                q_values = _q_values(network, state)
                selections = {
                    item: select_counterfactual_action(q_values, mask, action_space, mode=item, top_k=args.top_k)
                    for item in SELECTION_MODES
                }
                selected = selections[mode]
                action_id = int(selected["action_id"])
                decoded = action_space.decode(action_id)
                action_type = str(decoded["action_type"])
                candidate_index = int(decoded["candidate_index"])
                candidate = _candidate_at(raw, candidate_index) if action_type in {CLICK, "fill_input", "press_enter"} else {}
                action_target = f"{action_type}:{_target_identity(candidate) or candidate_index}"
                action_targets.append(action_target)
                click_rows = click_candidate_rows(q_values, mask, action_space, raw)
                click_rows_all.extend({**row, "episode": episode, "step": step} for row in click_rows)
                q_diag = q_value_diagnostics(network, state, mask, action_space, action_id, top_k=5)
                mask_stats = action_space.build_action_mask_stats(raw, mask, action_id)

                next_raw, _, done, info = env.step(action_id)
                decoded["failed"] = bool(info.get("last_action_error"))
                decoded["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": decoded}))
                reward, reward_components = calculate_dqn_reward(raw, next_raw, decoded, anomalies, history)
                aggregator.record_step(
                    step=global_step,
                    before_observation=raw,
                    after_observation=next_raw,
                    action=decoded,
                    reward=reward,
                    reward_components=reward_components,
                    anomalies=anomalies,
                    info=info,
                    mask_stats=mask_stats,
                    q_diagnostic=q_diag,
                )
                records.append(
                    {
                        "episode": episode,
                        "step": step,
                        "mode": mode,
                        "action_id": action_id,
                        "action_type": action_type,
                        "candidate_index": candidate_index,
                        "action_target": action_target,
                        "selected_q": float(q_values[action_id]),
                        "selected_type_score": float(selected.get("type_score", q_values[action_id]) or 0.0),
                        "counterfactual_selections": {
                            key: _selection_summary(value)
                            for key, value in selections.items()
                        },
                        "valid_count_by_action_type": valid_count_by_action_type(mask, action_space),
                        "q_statistics_by_action_type": q_statistics_by_action_type(q_values, mask, action_space),
                        "click_equivalent_groups": equivalent_click_group_summary(click_rows),
                        "reward": float(reward),
                    }
                )
                raw = next_raw
                state = env.observation_encoder.encode_observation(raw)
                if done:
                    break
    finally:
        env.close()

    summary = aggregator.summary()
    metrics = summary.get("metrics", {})
    action_distribution = summary.get("action_distribution", {})
    return {
        "selection_mode": mode,
        "metrics": metrics,
        "action_distribution": action_distribution,
        "click_ratio": _ratio(action_distribution.get(CLICK, 0), metrics.get("total_steps", 0)),
        "repeat_diagnostics": repeat_diagnostics(action_targets, window=args.repeat_window),
        "offline_counterfactual_on_visited_states": offline_counterfactual_from_records(records),
        "q_statistics_by_action_type": summarize_q_stats_from_records(records),
        "click_candidate_summary": summarize_click_rows(click_rows_all),
        "records": records[: args.max_records],
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    modes = [item.strip() for item in str(args.selection_modes).split(",") if item.strip()]
    for mode in modes:
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unsupported selection mode: {mode}")
    runs = {mode: run_counterfactual_rollout(args, mode) for mode in modes}
    report = {
        "model_path": args.model_path,
        "site_id": args.site_id,
        "url": _load_site_config(args.config, args.site_id, args.base_url)["base_url"],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "counterfactual_only": True,
        "production_greedy_changed": False,
        "selection_modes": modes,
        "runs": runs,
        "comparison": compare_runs(runs),
        "root_cause_summary": root_cause_summary(runs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def compare_runs(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for mode, run in runs.items():
        metrics = run.get("metrics", {})
        repeat = run.get("repeat_diagnostics", {})
        rows[mode] = {
            "total_steps": metrics.get("total_steps"),
            "action_distribution": run.get("action_distribution", {}),
            "click_ratio": run.get("click_ratio"),
            "unique_local_action_targets": metrics.get("unique_local_action_targets"),
            "repeat_action_ratio": metrics.get("repeat_local_action_ratio"),
            "recent_window_repeat_ratio": repeat.get("recent_window_repeat_ratio"),
            "cycle_repeat_ratio": repeat.get("cycle_repeat_ratio"),
            "candidate_anomaly_count": metrics.get("candidate_anomaly_count"),
            "confirmed_anomaly_count": metrics.get("confirmed_anomaly_count"),
            "action_error_count": metrics.get("action_error_count"),
        }
    flat_click = float(rows.get(FLAT_MAX, {}).get("click_ratio", 0.0) or 0.0)
    return {
        "by_mode": rows,
        "click_ratio_reduction_vs_flat": {
            mode: flat_click - float(row.get("click_ratio", 0.0) or 0.0)
            for mode, row in rows.items()
            if mode != FLAT_MAX
        },
    }


def summarize_q_stats_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for action_type, stats in (record.get("q_statistics_by_action_type", {}) or {}).items():
            for key in ("valid_count", "q_max", "q_mean", "q_std", "q_median", "max_selection_uplift"):
                rows[str(action_type)][key].append(float(stats.get(key, 0.0) or 0.0))
    return {
        action_type: {f"{key}_mean": _mean(values) for key, values in stats.items()}
        for action_type, stats in rows.items()
    }


def summarize_click_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q_values = [float(row.get("q", 0.0) or 0.0) for row in rows]
    return {
        "observations": len(rows),
        "q_mean": _mean(q_values),
        "q_std": _std(q_values),
        "q_max": float(np.max(q_values)) if q_values else 0.0,
        "controlled_max_over_many_simulation": controlled_max_over_many_simulation(mean=_mean(q_values), std=_std(q_values)),
    }


def root_cause_summary(runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    causes: list[str] = []
    flat = runs.get(FLAT_MAX, {})
    flat_click = float(flat.get("click_ratio", 0.0) or 0.0)
    for mode in (TYPE_MEAN_MAX, TYPE_MEDIAN_MAX, TYPE_TOPK_MEAN_MAX):
        run = runs.get(mode, {})
        if flat_click >= 0.9 and float(run.get("click_ratio", 0.0) or 0.0) <= 0.5:
            causes.append(f"{mode.lower()}_reduces_flat_click_bias")
    flat_q = (flat.get("q_statistics_by_action_type", {}) or {}).get(CLICK, {})
    if float(flat_q.get("max_selection_uplift_mean", 0.0) or 0.0) > 0.15:
        causes.append("flat_max_over_many_click_uplift")
    return causes


def _load_network(model_path: str, observation_dim: int, action_dim: int) -> QNetwork:
    network = QNetwork(observation_dim, action_dim)
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    network.load_state_dict(checkpoint.get("model", checkpoint))
    network.eval()
    return network


def _q_values(network: QNetwork, state: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0).detach().cpu().numpy()


def _selection_summary(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_id": selection.get("action_id"),
        "action_type": selection.get("action_type"),
        "candidate_index": selection.get("candidate_index"),
        "q": selection.get("q"),
        "type_score": selection.get("type_score"),
    }


def _candidate_at(observation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return {}


def _target_identity(candidate: Mapping[str, Any]) -> str:
    bid = str(candidate.get("bid") or "").strip()
    return f"bid:{bid}" if bid else ""


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
    parser.add_argument("--model-path", default="artifacts/models/site001_dqn_balancedexplore_ep30.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-window", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--selection-modes", default="FLAT_MAX,TYPE_MEAN_MAX,TYPE_MEDIAN_MAX,TYPE_TOPK_MEAN_MAX")
    parser.add_argument("--max-records", type=int, default=80)
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_type_level_counterfactual.json")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_diagnostics(args)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
