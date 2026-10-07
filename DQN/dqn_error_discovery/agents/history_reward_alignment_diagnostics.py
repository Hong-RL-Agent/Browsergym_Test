"""Offline diagnostics for history/reward alignment in the history-aware DQN."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dqn_error_discovery.agents.dqn_agent import (
    FactorizedQNetwork,
    factorized_flat_q_values,
    factorized_greedy_actions_from_outputs,
)


HISTORY_OFFSET = 635 + 72
HISTORY_DIM = 238
SCROLL_ACTIONS = {"scroll_up", "scroll_down"}

HISTORY_FEATURE_GROUPS: dict[str, list[tuple[int, int]]] = {
    "recent_action_type": [(0, 76)],
    "target_history": [(76, 80), (80, 208), (208, 212)],
    "history_valid_mask": [(212, 216)],
    "outcome_flags": [(216, 232)],
    "streak": [(232, 233)],
    "distinct_identity_count": [(233, 234)],
    "cycle_onehot": [(234, 238)],
}

OUTCOME_FLAG_GROUPS: dict[str, list[int]] = {
    "meaningful_action_effect": [216 + slot * 4 for slot in range(4)],
    "productive_transition": [216 + slot * 4 + 1 for slot in range(4)],
    "confirmation_progress": [216 + slot * 4 + 2 for slot in range(4)],
    "unproductive_repeat": [216 + slot * 4 + 3 for slot in range(4)],
}


def mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(items),
        "mean": float(np.mean(items)),
        "median": float(median(items)),
        "min": float(min(items)),
        "max": float(max(items)),
    }


def reward_timing(components: Mapping[str, Any] | None) -> dict[str, float]:
    components = components or {}
    novelty = float(components.get("new_local_state", 0.0) or 0.0) + float(
        components.get("new_local_action_target", 0.0) or 0.0
    )
    return {
        "novelty_reward": novelty,
        "flow_reward": float(components.get("flow_progress", 0.0) or 0.0),
        "candidate_reward": float(components.get("candidate_anomaly", 0.0) or 0.0),
        "confirmation_reward": float(components.get("confirmed_anomaly", 0.0) or 0.0),
        "upgrade_reward": float(components.get("confirmation_upgrade", 0.0) or 0.0),
        "repeat_penalty": float(components.get("repeat_action_penalty", 0.0) or 0.0)
        + float(components.get("repeat_state_penalty", 0.0) or 0.0)
        + float(components.get("repeat_anomaly_penalty", 0.0) or 0.0),
        "no_effect_penalty": float(components.get("no_effect_penalty", 0.0) or 0.0),
        "invalid_timeout": float(components.get("invalid_penalty", 0.0) or 0.0)
        + float(components.get("timeout_penalty", 0.0) or 0.0)
        + float(components.get("action_error_penalty", 0.0) or 0.0),
        "final_reward": float(components.get("final_reward", 0.0) or 0.0),
    }


def predictability_classification(history_diag: Mapping[str, Any], repeat_diag: Mapping[str, Any]) -> str:
    if bool(history_diag.get("previous_unproductive_repeat")):
        return "PREDICTABLE_FROM_STATE"
    if int(history_diag.get("active_cycle_length", 0) or 0) > 0 or int(history_diag.get("same_action_identity_streak", 0) or 0) > 1:
        return "PREDICTABLE_FROM_STATE"
    if bool(repeat_diag.get("recent_window_repeat")):
        return "PARTIALLY_PREDICTABLE"
    return "NOT_PREDICTABLE_UNTIL_AFTER_ACTION"


def summarize_predictability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("predictability") or "") for row in rows)
    return {
        "PREDICTABLE_FROM_STATE": counts.get("PREDICTABLE_FROM_STATE", 0),
        "PARTIALLY_PREDICTABLE": counts.get("PARTIALLY_PREDICTABLE", 0),
        "NOT_PREDICTABLE_UNTIL_AFTER_ACTION": counts.get("NOT_PREDICTABLE_UNTIL_AFTER_ACTION", 0),
    }


def q_values_for_actions(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_ids: Sequence[int],
    action_space,
) -> dict[int, float]:
    with torch.inference_mode():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)
        flat = factorized_flat_q_values(network(state_tensor), mask_tensor, action_space).squeeze(0)
    return {int(action_id): float(flat[int(action_id)].item()) for action_id in action_ids}


def greedy_action_id(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, action_space) -> int:
    with torch.inference_mode():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)
        action = factorized_greedy_actions_from_outputs(network(state_tensor), mask_tensor, action_space)
    return int(action.item())


def zero_history_state(state: np.ndarray) -> np.ndarray:
    result = np.array(state, dtype=np.float32, copy=True)
    result[HISTORY_OFFSET : HISTORY_OFFSET + HISTORY_DIM] = 0.0
    return result


def ablated_state(state: np.ndarray, spans: Sequence[tuple[int, int]] = (), indexes: Sequence[int] = ()) -> np.ndarray:
    result = np.array(state, dtype=np.float32, copy=True)
    for start, end in spans:
        result[HISTORY_OFFSET + start : HISTORY_OFFSET + end] = 0.0
    for index in indexes:
        result[HISTORY_OFFSET + int(index)] = 0.0
    return result


def action_type_values(
    network: FactorizedQNetwork,
    state: np.ndarray,
    action_space,
    action_types: Sequence[str],
) -> dict[str, float]:
    with torch.inference_mode():
        outputs = network(torch.from_numpy(state).float().unsqueeze(0))
        values = outputs["type_values"].squeeze(0).detach().cpu().numpy()
    return {
        action_type: float(values[action_space.action_types.index(action_type)])
        for action_type in action_types
        if action_type in action_space.action_types
    }


def compare_cycle_and_break_q(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    cycle_action_id: int,
    break_action_id: int,
    action_space,
) -> dict[str, Any]:
    zero_state = zero_history_state(state)
    real_q = q_values_for_actions(network, state, mask, [cycle_action_id, break_action_id], action_space)
    zero_q = q_values_for_actions(network, zero_state, mask, [cycle_action_id, break_action_id], action_space)
    real_margin = real_q[cycle_action_id] - real_q[break_action_id]
    zero_margin = zero_q[cycle_action_id] - zero_q[break_action_id]
    return {
        "Q_real_cycle": real_q[cycle_action_id],
        "Q_real_break": real_q[break_action_id],
        "Q_zero_cycle": zero_q[cycle_action_id],
        "Q_zero_break": zero_q[break_action_id],
        "cycle_q_margin_real": real_margin,
        "cycle_q_margin_zero": zero_margin,
        "history_cycle_bias": real_margin - zero_margin,
    }


def feature_ablation_rows(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    cycle_action_id: int,
    break_action_id: int,
    action_space,
    *,
    outcome_flags: bool = False,
) -> list[dict[str, Any]]:
    baseline = compare_cycle_and_break_q(network, state, mask, cycle_action_id, break_action_id, action_space)
    baseline_selected = greedy_action_id(network, state, mask, action_space)
    cycle_type = str(action_space.decode(cycle_action_id).get("action_type") or "")
    break_type = str(action_space.decode(break_action_id).get("action_type") or "")
    groups: dict[str, Any]
    if outcome_flags:
        groups = {key: {"indexes": indexes} for key, indexes in OUTCOME_FLAG_GROUPS.items()}
    else:
        groups = {key: {"spans": spans} for key, spans in HISTORY_FEATURE_GROUPS.items()}
    rows: list[dict[str, Any]] = []
    for name, spec in groups.items():
        state_ab = ablated_state(state, spec.get("spans", ()), spec.get("indexes", ()))
        selected = greedy_action_id(network, state_ab, mask, action_space)
        values = q_values_for_actions(network, state_ab, mask, [cycle_action_id, break_action_id], action_space)
        margin = values[cycle_action_id] - values[break_action_id]
        type_values = action_type_values(network, state_ab, action_space, ["scroll_up", "scroll_down", break_type])
        base_type_values = action_type_values(network, state, action_space, ["scroll_up", "scroll_down", break_type])
        rows.append(
            {
                "group": name,
                "selected_action": str(action_space.decode(selected).get("action_type") or ""),
                "selected_action_flip": bool(selected != baseline_selected),
                "scroll_up_value_delta": type_values.get("scroll_up", 0.0) - base_type_values.get("scroll_up", 0.0),
                "scroll_down_value_delta": type_values.get("scroll_down", 0.0) - base_type_values.get("scroll_down", 0.0),
                "break_action_value_delta": type_values.get(break_type, 0.0) - base_type_values.get(break_type, 0.0),
                "cycle_q_margin_delta": margin - baseline["cycle_q_margin_real"],
                "cycle_q_margin_after_ablation": margin,
                "cycle_action_type": cycle_type,
                "break_action_type": break_type,
            }
        )
    return rows


def aggregate_ablation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("group") or ""), []).append(row)
    summary = {}
    for group, items in by_group.items():
        deltas = [float(item.get("cycle_q_margin_delta", 0.0) or 0.0) for item in items]
        summary[group] = {
            "count": len(items),
            "selected_action_flip_rate": mean_or_zero([1.0 if item.get("selected_action_flip") else 0.0 for item in items]),
            "mean_cycle_q_margin_delta": mean_or_zero(deltas),
            "mean_abs_cycle_q_margin_delta": mean_or_zero([abs(value) for value in deltas]),
            "mean_scroll_up_value_delta": mean_or_zero([float(item.get("scroll_up_value_delta", 0.0) or 0.0) for item in items]),
            "mean_scroll_down_value_delta": mean_or_zero([float(item.get("scroll_down_value_delta", 0.0) or 0.0) for item in items]),
            "mean_break_action_value_delta": mean_or_zero([float(item.get("break_action_value_delta", 0.0) or 0.0) for item in items]),
        }
    return summary


def rank_cycle_bias_features(summary: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "group": group,
            "mean_abs_cycle_q_margin_delta": float(values.get("mean_abs_cycle_q_margin_delta", 0.0) or 0.0),
            "mean_cycle_q_margin_delta": float(values.get("mean_cycle_q_margin_delta", 0.0) or 0.0),
        }
        for group, values in summary.items()
    ]
    return sorted(rows, key=lambda row: row["mean_abs_cycle_q_margin_delta"], reverse=True)


def conditional_reward_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = {
        "previous_unproductive=1|continue_cycle": [],
        "previous_unproductive=1|break_cycle": [],
        "cycle_active|continue_cycle": [],
        "cycle_active|break_cycle": [],
    }
    for row in rows:
        history = row.get("history_state_diagnostic", {}) if isinstance(row.get("history_state_diagnostic"), Mapping) else {}
        reward = float(row.get("reward", 0.0) or 0.0)
        confirmed = bool(row.get("confirmation_transition_events"))
        selected = str(history.get("selected_action_identity") or row.get("action_identity") or "")
        cycle_identity = str(history.get("cycle_continuation_identity") or "")
        continues = bool(cycle_identity and selected == cycle_identity)
        if bool(history.get("previous_unproductive_repeat")):
            key = "previous_unproductive=1|continue_cycle" if continues else "previous_unproductive=1|break_cycle"
            buckets[key].append((reward, confirmed))
        if int(history.get("active_cycle_length", 0) or 0) > 0:
            key = "cycle_active|continue_cycle" if continues else "cycle_active|break_cycle"
            buckets[key].append((reward, confirmed))
    return {
        key: {
            "count": len(values),
            "mean_reward": mean_or_zero([reward for reward, _ in values]),
            "confirmation_rate": mean_or_zero([1.0 if confirmed else 0.0 for _, confirmed in values]),
        }
        for key, values in buckets.items()
    }


def immediate_reward_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = {"cycle_continue": [], "cycle_break": []}
    for row in rows:
        history = row.get("history_state_diagnostic", {}) if isinstance(row.get("history_state_diagnostic"), Mapping) else {}
        if int(history.get("active_cycle_length", 0) or 0) <= 0:
            continue
        selected = str(history.get("selected_action_identity") or row.get("action_identity") or "")
        cycle_identity = str(history.get("cycle_continuation_identity") or "")
        key = "cycle_continue" if selected == cycle_identity else "cycle_break"
        buckets[key].append(float(row.get("reward", 0.0) or 0.0))
    return {
        key: {
            **numeric_summary(values),
            "positive": sum(1 for value in values if value > 0),
            "zero": sum(1 for value in values if value == 0),
            "negative": sum(1 for value in values if value < 0),
        }
        for key, values in buckets.items()
    }


def confirmation_confounding(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cycle_scroll = []
    cycle_break = []
    for row in rows:
        history = row.get("history_state_diagnostic", {}) if isinstance(row.get("history_state_diagnostic"), Mapping) else {}
        if int(history.get("active_cycle_length", 0) or 0) <= 0:
            continue
        action_type = str(row.get("action_type") or "")
        confirmed = bool(row.get("confirmation_transition_events"))
        reward = float(row.get("reward", 0.0) or 0.0)
        selected = str(history.get("selected_action_identity") or row.get("action_identity") or "")
        cycle_identity = str(history.get("cycle_continuation_identity") or "")
        if action_type in SCROLL_ACTIONS and selected == cycle_identity:
            cycle_scroll.append((reward, confirmed))
        else:
            cycle_break.append((reward, confirmed))
    return {
        "cycle_scroll": {
            "count": len(cycle_scroll),
            "confirmation_rate": mean_or_zero([1.0 if confirmed else 0.0 for _, confirmed in cycle_scroll]),
            "mean_reward": mean_or_zero([reward for reward, _ in cycle_scroll]),
        },
        "cycle_break": {
            "count": len(cycle_break),
            "confirmation_rate": mean_or_zero([1.0 if confirmed else 0.0 for _, confirmed in cycle_break]),
            "mean_reward": mean_or_zero([reward for reward, _ in cycle_break]),
        },
    }


def escape_metric_interpretation(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repeat_and_escape_can_coexist": True,
        "metric_bug": False,
        "reason": (
            "repeat ratios count repeated transitions themselves; escape rates count only the following step "
            "after an unproductive or cycle state and can be high when the policy alternates away for one step "
            "but later re-enters the same scroll pattern."
        ),
        "repeat": float(metrics.get("repeat_local_action_ratio", 0.0) or 0.0),
        "repeat_escape": float(metrics.get("repeat_escape_rate", 0.0) or 0.0),
        "cycle_repeat": float(metrics.get("cycle_repeat_ratio", 0.0) or 0.0),
        "cycle_escape": float(metrics.get("cycle_escape_rate", 0.0) or 0.0),
    }
