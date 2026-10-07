"""Diagnostics for DQN confirmation-context learning.

These helpers only inspect already-computed policy inputs and actions. They do
not change action selection, reward, replay, masking, or tracker state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
import torch

from dqn_error_discovery.agents.confirmation_observation import CONFIRMATION_CONTEXT_DIM
from dqn_error_discovery.agents.confirmation_reproduction import ACTION_REPRODUCTION, OBSERVATION_REPRODUCTION
from dqn_error_discovery.agents.dqn_agent import (
    FactorizedQNetwork,
    center_candidate_advantages,
    factorized_greedy_actions_from_outputs,
)


FAILURE_NO_TYPE_SELECTION = "NO_TYPE_SELECTION"
FAILURE_TYPE_SELECTED_WRONG_TARGET = "TYPE_SELECTED_WRONG_TARGET"
FAILURE_FULL_MATCH_NO_PROGRESS = "FULL_MATCH_NO_PROGRESS"
FAILURE_PROGRESS_NO_CONFIRMATION = "PROGRESS_NO_CONFIRMATION"
FAILURE_CONFIRMED = "CONFIRMED"
FAILURE_KEYS = (
    FAILURE_NO_TYPE_SELECTION,
    FAILURE_TYPE_SELECTED_WRONG_TARGET,
    FAILURE_FULL_MATCH_NO_PROGRESS,
    FAILURE_PROGRESS_NO_CONFIRMATION,
    FAILURE_CONFIRMED,
)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def is_reproduction_opportunity(metadata: Mapping[str, Any]) -> bool:
    expected_type = str(metadata.get("expected_action_type") or "")
    requires_target = bool(metadata.get("expected_action_requires_target"))
    target_visible = bool(metadata.get("expected_target_visible"))
    return bool(metadata.get("pending_confirmation_exists")) and str(metadata.get("confirmation_mode") or ACTION_REPRODUCTION) == ACTION_REPRODUCTION and bool(expected_type) and (
        not requires_target or target_visible
    )


def is_observation_reproduction_pending(metadata: Mapping[str, Any]) -> bool:
    return bool(metadata.get("pending_confirmation_exists")) and str(metadata.get("confirmation_mode") or "") == OBSERVATION_REPRODUCTION


def classify_reproduction_selection(
    metadata: Mapping[str, Any],
    selected_action: Mapping[str, Any],
) -> dict[str, Any]:
    expected_type = str(metadata.get("expected_action_type") or "")
    selected_type = str(selected_action.get("action_type") or "")
    requires_target = bool(metadata.get("expected_action_requires_target"))
    matching_slots = [int(slot) for slot in metadata.get("matching_candidate_slots", []) or []]
    selected_slot = int(selected_action.get("candidate_index", -1) or -1)
    type_match = bool(expected_type and selected_type == expected_type)
    target_match = bool(not requires_target or selected_slot in matching_slots)
    full_match = bool(type_match and target_match)
    return {
        "opportunity": is_reproduction_opportunity(metadata),
        "expected_action_type": expected_type,
        "selected_action_type": selected_type,
        "requires_target": requires_target,
        "matching_candidate_slots": matching_slots,
        "selected_target_slot": selected_slot,
        "type_match": type_match,
        "target_match": bool(target_match) if type_match else False,
        "full_match": full_match,
    }


def factorized_confirmation_step_diagnostic(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space: Any,
    selected_action_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    expected_type = str(metadata.get("expected_action_type") or "")
    selected = action_space.decode(int(selected_action_id))
    with torch.inference_mode():
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(np.asarray(mask) > 0).bool().unsqueeze(0)
        outputs = network(state_tensor)
        type_values = outputs["type_values"].squeeze(0).detach().cpu().numpy()
        centered = center_candidate_advantages(outputs["candidate_advantages"], mask_tensor, action_space).squeeze(0)
        centered_np = centered.detach().cpu().numpy()

    valid_types = _valid_type_indexes(mask, action_space)
    type_rows = sorted(
        ((index, float(type_values[index])) for index in valid_types),
        key=lambda item: item[1],
        reverse=True,
    )
    expected_type_index = _type_index(action_space, expected_type)
    selected_type_index = int(selected.get("action_type_id", -1))
    expected_rank = _rank_from_sorted(type_rows, expected_type_index)
    top1 = type_rows[0] if type_rows else (None, None)
    top2 = type_rows[1] if len(type_rows) > 1 else (None, None)

    result: dict[str, Any] = {
        "expected_action_type": expected_type,
        "expected_type_value": _value_at(type_values, expected_type_index),
        "expected_type_rank": expected_rank,
        "selected_action_type": str(selected.get("action_type") or ""),
        "selected_type_value": _value_at(type_values, selected_type_index),
        "top1_type": action_space.action_types[top1[0]] if top1[0] is not None else "",
        "top2_type": action_space.action_types[top2[0]] if top2[0] is not None else "",
        "top1_type_value": top1[1],
        "top2_type_value": top2[1],
        "top1_top2_gap": float(top1[1] - top2[1]) if top1[1] is not None and top2[1] is not None else None,
    }

    if expected_type_index is not None and str(selected.get("action_type") or "") == expected_type:
        result.update(expected_target_rank_diagnostic(centered_np, mask, action_space, metadata, expected_type_index, selected))
    return result


def expected_target_rank_diagnostic(
    centered_advantages: np.ndarray,
    mask: np.ndarray,
    action_space: Any,
    metadata: Mapping[str, Any],
    expected_type_index: int | None = None,
    selected_action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_type = str(metadata.get("expected_action_type") or "")
    type_index = expected_type_index if expected_type_index is not None else _type_index(action_space, expected_type)
    if type_index is None:
        return {}
    matching_slots = [int(slot) for slot in metadata.get("matching_candidate_slots", []) or []]
    valid_slots = [
        slot
        for slot in range(action_space.max_candidates)
        if int(action_space.encode(expected_type, slot)) < len(mask)
        and float(mask[int(action_space.encode(expected_type, slot))]) > 0.0
    ]
    if not matching_slots or not valid_slots:
        return {}
    ranked = sorted(((slot, float(centered_advantages[type_index, slot])) for slot in valid_slots), key=lambda item: item[1], reverse=True)
    expected_slot = max((slot for slot in matching_slots if slot in valid_slots), key=lambda slot: float(centered_advantages[type_index, slot]), default=matching_slots[0])
    selected_slot = int((selected_action or {}).get("candidate_index", -1) or -1)
    return {
        "expected_target_slot": int(expected_slot),
        "expected_target_advantage": float(centered_advantages[type_index, expected_slot]),
        "expected_target_rank": _rank_from_sorted(ranked, expected_slot),
        "selected_target_slot": selected_slot,
        "selected_target_advantage": float(centered_advantages[type_index, selected_slot]) if selected_slot in valid_slots else None,
        "target_match": bool(selected_slot in matching_slots),
    }


def real_vs_zero_context_diagnostic(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space: Any,
    metadata: Mapping[str, Any],
    *,
    base_obs_dim: int = 635,
) -> dict[str, Any]:
    real_state = np.asarray(state, dtype=np.float32).reshape(-1)
    zero_state = real_state.copy()
    if zero_state.size >= base_obs_dim + CONFIRMATION_CONTEXT_DIM:
        zero_state[base_obs_dim : base_obs_dim + CONFIRMATION_CONTEXT_DIM] = 0.0
    expected_type = str(metadata.get("expected_action_type") or "")
    expected_type_index = _type_index(action_space, expected_type)
    with torch.inference_mode():
        mask_tensor = torch.from_numpy(np.asarray(mask) > 0).bool().unsqueeze(0)
        real_outputs = network(torch.from_numpy(real_state).float().unsqueeze(0))
        zero_outputs = network(torch.from_numpy(zero_state).float().unsqueeze(0))
        real_type_values = real_outputs["type_values"].squeeze(0).detach().cpu().numpy()
        zero_type_values = zero_outputs["type_values"].squeeze(0).detach().cpu().numpy()
        real_action = int(factorized_greedy_actions_from_outputs(real_outputs, mask_tensor, action_space).item())
        zero_action = int(factorized_greedy_actions_from_outputs(zero_outputs, mask_tensor, action_space).item())
        real_centered = center_candidate_advantages(real_outputs["candidate_advantages"], mask_tensor, action_space).squeeze(0).detach().cpu().numpy()
        zero_centered = center_candidate_advantages(zero_outputs["candidate_advantages"], mask_tensor, action_space).squeeze(0).detach().cpu().numpy()
    diff = real_type_values - zero_type_values
    real_selected = action_space.decode(real_action)
    zero_selected = action_space.decode(zero_action)
    expected_target_delta = None
    target_diag_real = {}
    target_diag_zero = {}
    if expected_type_index is not None and bool(metadata.get("expected_action_requires_target")):
        target_diag_real = expected_target_rank_diagnostic(real_centered, mask, action_space, metadata, expected_type_index, real_selected)
        target_diag_zero = expected_target_rank_diagnostic(zero_centered, mask, action_space, metadata, expected_type_index, zero_selected)
        if target_diag_real.get("expected_target_advantage") is not None and target_diag_zero.get("expected_target_advantage") is not None:
            expected_target_delta = float(target_diag_real["expected_target_advantage"] - target_diag_zero["expected_target_advantage"])
    return {
        "real_expected_type_value": _value_at(real_type_values, expected_type_index),
        "zero_expected_type_value": _value_at(zero_type_values, expected_type_index),
        "expected_type_context_delta": (
            float(real_type_values[expected_type_index] - zero_type_values[expected_type_index])
            if expected_type_index is not None
            else None
        ),
        "real_selected_type": str(real_selected.get("action_type") or ""),
        "zero_selected_type": str(zero_selected.get("action_type") or ""),
        "selection_flip": bool(real_selected.get("action_type") != zero_selected.get("action_type")),
        "type_value_l1": float(np.mean(np.abs(diff))),
        "type_value_l2": float(np.sqrt(np.mean(diff * diff))),
        "type_value_max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "expected_target_context_delta": expected_target_delta,
        "real_expected_target_rank": target_diag_real.get("expected_target_rank"),
        "zero_expected_target_rank": target_diag_zero.get("expected_target_rank"),
    }


class ConfirmationFunnelAggregator:
    def __init__(self) -> None:
        self.pending_state_step_count = 0
        self.reproduction_opportunity_count = 0
        self.expected_action_type_selected_count = 0
        self.expected_action_and_target_selected_count = 0
        self.confirmation_progress_count = 0
        self.candidate_to_confirmed_count = 0
        self.confirmed_count = 0
        self.failure_counts: Counter[str] = Counter({key: 0 for key in FAILURE_KEYS})
        self.rule_funnel: dict[str, Counter[str]] = defaultdict(Counter)
        self.type_ranks: list[float] = []
        self.target_ranks: list[float] = []
        self.context_rows: list[dict[str, Any]] = []

    def record(
        self,
        metadata: Mapping[str, Any],
        selected_action: Mapping[str, Any],
        *,
        confirmation_progress: bool = False,
        candidate_to_confirmed: bool = False,
        confirmed: bool = False,
        type_diagnostic: Mapping[str, Any] | None = None,
        context_diagnostic: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if bool(metadata.get("pending_confirmation_exists")):
            self.pending_state_step_count += 1
        classification = classify_reproduction_selection(metadata, selected_action)
        opportunity = bool(classification["opportunity"])
        if not opportunity:
            return classification
        rule_id = str(metadata.get("active_rule_id") or "UNKNOWN")
        rule_counter = self.rule_funnel[rule_id]
        self.reproduction_opportunity_count += 1
        rule_counter["opportunities"] += 1
        if classification["type_match"]:
            self.expected_action_type_selected_count += 1
            rule_counter["type_match"] += 1
        if classification["full_match"]:
            self.expected_action_and_target_selected_count += 1
            rule_counter["full_match"] += 1
        if confirmation_progress:
            self.confirmation_progress_count += 1
            rule_counter["progress"] += 1
        if candidate_to_confirmed:
            self.candidate_to_confirmed_count += 1
        if confirmed:
            self.confirmed_count += 1
            rule_counter["confirmed"] += 1
        if type_diagnostic:
            if type_diagnostic.get("expected_type_rank") is not None:
                self.type_ranks.append(float(type_diagnostic["expected_type_rank"]))
            if type_diagnostic.get("expected_target_rank") is not None:
                self.target_ranks.append(float(type_diagnostic["expected_target_rank"]))
        if context_diagnostic:
            self.context_rows.append(dict(context_diagnostic))
        self.failure_counts[classify_failure(classification, confirmation_progress, confirmed)] += 1
        return classification

    def summary(self) -> dict[str, Any]:
        context = _context_summary(self.context_rows)
        return {
            "pending_state_step_count": self.pending_state_step_count,
            "reproduction_opportunity_count": self.reproduction_opportunity_count,
            "expected_action_type_selected_count": self.expected_action_type_selected_count,
            "expected_action_and_target_selected_count": self.expected_action_and_target_selected_count,
            "action_type_match_rate": safe_ratio(self.expected_action_type_selected_count, self.reproduction_opportunity_count),
            "full_reproduction_match_rate": safe_ratio(self.expected_action_and_target_selected_count, self.reproduction_opportunity_count),
            "target_selection_given_type_rate": safe_ratio(self.expected_action_and_target_selected_count, self.expected_action_type_selected_count),
            "progress_given_full_match_rate": safe_ratio(self.confirmation_progress_count, self.expected_action_and_target_selected_count),
            "confirmed_given_progress_rate": safe_ratio(self.confirmed_count, self.confirmation_progress_count),
            "confirmation_progress_count": self.confirmation_progress_count,
            "candidate_to_confirmed_count": self.candidate_to_confirmed_count,
            "confirmed_count": self.confirmed_count,
            "expected_type_rank_mean": float(np.mean(self.type_ranks)) if self.type_ranks else None,
            "expected_target_rank_mean": float(np.mean(self.target_ranks)) if self.target_ranks else None,
            "failure_attribution": dict(self.failure_counts),
            "rule_funnel": {rule: dict(counter) for rule, counter in sorted(self.rule_funnel.items())},
            **context,
        }


def classify_failure(classification: Mapping[str, Any], confirmation_progress: bool, confirmed: bool) -> str:
    if confirmed:
        return FAILURE_CONFIRMED
    if confirmation_progress:
        return FAILURE_PROGRESS_NO_CONFIRMATION
    if bool(classification.get("full_match")):
        return FAILURE_FULL_MATCH_NO_PROGRESS
    if bool(classification.get("type_match")):
        return FAILURE_TYPE_SELECTED_WRONG_TARGET
    return FAILURE_NO_TYPE_SELECTION


def _context_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    deltas = values("expected_type_context_delta")
    target_deltas = values("expected_target_context_delta")
    return {
        "real_expected_type_value_mean": float(np.mean(values("real_expected_type_value"))) if values("real_expected_type_value") else None,
        "zero_expected_type_value_mean": float(np.mean(values("zero_expected_type_value"))) if values("zero_expected_type_value") else None,
        "expected_type_context_delta_mean": float(np.mean(deltas)) if deltas else None,
        "expected_type_context_delta_median": float(np.median(deltas)) if deltas else None,
        "expected_type_context_delta_min": float(np.min(deltas)) if deltas else None,
        "expected_type_context_delta_max": float(np.max(deltas)) if deltas else None,
        "context_selection_flip_rate": safe_ratio(sum(1 for row in rows if bool(row.get("selection_flip"))), len(rows)),
        "type_value_l1_mean": float(np.mean(values("type_value_l1"))) if values("type_value_l1") else None,
        "type_value_l2_mean": float(np.mean(values("type_value_l2"))) if values("type_value_l2") else None,
        "type_value_max_abs": float(np.max(values("type_value_max_abs"))) if values("type_value_max_abs") else None,
        "expected_target_context_delta_mean": float(np.mean(target_deltas)) if target_deltas else None,
    }


def _valid_type_indexes(mask: np.ndarray, action_space: Any) -> list[int]:
    indexes = []
    for type_index, action_type in enumerate(action_space.action_types):
        if any(float(mask[action_space.encode(action_type, slot)]) > 0.0 for slot in range(action_space.max_candidates)):
            indexes.append(type_index)
    return indexes


def _type_index(action_space: Any, action_type: str) -> int | None:
    if action_type not in action_space.action_types:
        return None
    return int(action_space.action_types.index(action_type))


def _rank_from_sorted(sorted_rows: list[tuple[int, float]], wanted_index: int | None) -> int | None:
    if wanted_index is None:
        return None
    for rank, (index, _) in enumerate(sorted_rows, start=1):
        if int(index) == int(wanted_index):
            return rank
    return None


def _value_at(values: np.ndarray, index: int | None) -> float | None:
    if index is None or index < 0 or index >= len(values):
        return None
    return float(values[index])
