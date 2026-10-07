"""Pure helpers for confirmation credit and forced-probe diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import torch

from dqn_error_discovery.agents.confirmation_observation import CONFIRMATION_CONTEXT_DIM
from dqn_error_discovery.agents.dqn_agent import (
    FactorizedQNetwork,
    center_candidate_advantages,
    factorized_greedy_actions_from_outputs,
)


def resolve_expected_action(metadata: Mapping[str, Any], action_space: Any, mask: np.ndarray) -> dict[str, Any]:
    """Resolve only the action described by tracker metadata."""
    expected_type = str(metadata.get("expected_action_type") or "")
    requires_target = bool(metadata.get("expected_action_requires_target"))
    target_semantics_valid = bool(not requires_target or action_space.is_element_action(expected_type))
    matching = [int(slot) for slot in metadata.get("matching_candidate_slots", []) or []]
    valid_slots = [
        slot for slot in range(action_space.max_candidates)
        if float(mask[action_space.encode(expected_type, slot)]) > 0.0
    ] if expected_type in action_space.action_types else []
    target_slot = next((slot for slot in matching if slot in valid_slots), None) if requires_target else (
        0 if expected_type in action_space.action_types and float(mask[action_space.encode(expected_type, 0)]) > 0.0 else None
    )
    action_type_defined = bool(expected_type in action_space.action_types)
    action_defined = bool(
        action_type_defined
        and (not requires_target or (bool(metadata.get("expected_target_visible")) and target_slot is not None))
    )
    action_id = int(action_space.encode(expected_type, int(target_slot or 0))) if action_defined else None
    mask_valid = bool(action_id is not None and float(mask[action_id]) > 0.0)
    return {
        "expected_action_defined": action_type_defined,
        "expected_action_type": expected_type,
        "expected_target_required": requires_target,
        "target_semantics_valid": target_semantics_valid,
        "expected_target_visible": bool(metadata.get("expected_target_visible")),
        "expected_target_identity": str(metadata.get("expected_target_identity") or ""),
        "expected_target_slot": target_slot,
        "expected_action_id": action_id,
        "expected_action_mask_valid": mask_valid,
        "forced_action_executable": bool(action_defined and target_semantics_valid and mask_valid),
        "valid_target_slots": valid_slots,
    }


def evidence_snapshot(anomalies: list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": str(item.get("rule_id") or item.get("evidence", {}).get("rule_id") or ""),
            "issue_type": str(item.get("issue_type") or item.get("evidence", {}).get("issue_type") or item.get("type") or ""),
            "status": str(item.get("status") or ""),
            "signature": str(item.get("signature") or ""),
            "target_id": str(item.get("target_id") or item.get("evidence", {}).get("target_id") or ""),
            "action_type": str(item.get("action_type") or ""),
            "confirmation_policy": str(item.get("confirmation_policy") or ""),
            "evidence": dict(item.get("evidence") or {}) if isinstance(item.get("evidence"), Mapping) else {},
        }
        for item in anomalies or []
        if isinstance(item, Mapping)
    ]


def evidence_reproduced(before: list[Mapping[str, Any]], after: list[Mapping[str, Any]], metadata: Mapping[str, Any]) -> bool:
    """Detect repeated evidence without consulting Ground Truth."""
    rule = str(metadata.get("active_rule_id") or "")
    expected = str(metadata.get("expected_target_identity") or "")
    before_rows = [row for row in before if str(row.get("rule_id") or "") == rule]
    after_rows = [row for row in after if str(row.get("rule_id") or "") == rule]
    if not before_rows or not after_rows:
        return False
    if expected:
        def target_match(row: Mapping[str, Any]) -> bool:
            target = str(row.get("target_id") or "")
            evidence = row.get("evidence", {})
            text = " ".join([target, str(evidence.get("target_bid") or ""), str(evidence.get("target_id") or "")])
            return expected.split(":", 1)[-1] in text
        return any(target_match(row) for row in after_rows)
    return True


def classify_forced_probe(*, executable: bool, evidence: bool, progress: bool, confirmed: bool) -> str:
    if confirmed:
        return "CONFIRMED"
    if progress:
        return "PROGRESS_NO_CONFIRMATION"
    if evidence:
        return "EVIDENCE_REPRODUCED_NO_PROGRESS"
    if executable:
        return "EXECUTED_NO_EVIDENCE"
    return "NOT_EXECUTABLE"


def reward_credit_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(row.get("total_reward", 0.0) or 0.0) for row in rows]
    progress_negative = [row for row in rows if bool(row.get("confirmation_progress_after")) and float(row.get("total_reward", 0.0) or 0.0) < 0]
    confirmed_nonpositive = [row for row in rows if bool(row.get("confirmed_after")) and float(row.get("total_reward", 0.0) or 0.0) <= 0]
    return {
        "count": len(rows),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "positive_count": sum(value > 0 for value in rewards),
        "zero_count": sum(value == 0 for value in rewards),
        "negative_count": sum(value < 0 for value in rewards),
        "progress_with_negative_reward_count": len(progress_negative),
        "confirmed_with_nonpositive_reward_count": len(confirmed_nonpositive),
        "reward_values": rewards,
    }


def flip_direction(zero_type: str, real_type: str, expected_type: str) -> str:
    if zero_type == expected_type and real_type != expected_type:
        return "FLIP_AWAY_FROM_EXPECTED"
    if zero_type != expected_type and real_type == expected_type:
        return "FLIP_TOWARD_EXPECTED"
    if zero_type != real_type:
        return "FLIP_OTHER_TO_OTHER"
    return "NO_FLIP"


def feature_group_ablation(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, action_space: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    groups = {
        "pending_rule": list(range(0, 16)),
        "expected_action_type_onehot": list(range(16, 35)),
        "expected_target_mask_visible": list(range(35, 69)),
        "progress_attempt_pending_count": list(range(69, 72)),
    }
    base = np.asarray(state, dtype=np.float32).reshape(-1)
    mask_tensor = torch.from_numpy(np.asarray(mask) > 0).bool().unsqueeze(0)
    expected_type = str(metadata.get("expected_action_type") or "")
    type_index = action_space.action_types.index(expected_type) if expected_type in action_space.action_types else None
    expected_slot = next(iter([int(x) for x in metadata.get("matching_candidate_slots", []) or []]), None)

    def measure(vector: np.ndarray) -> dict[str, Any]:
        with torch.inference_mode():
            outputs = network(torch.from_numpy(vector).float().unsqueeze(0))
            types = outputs["type_values"].squeeze(0).cpu().numpy()
            selected = int(factorized_greedy_actions_from_outputs(outputs, mask_tensor, action_space).item())
            centered = center_candidate_advantages(outputs["candidate_advantages"], mask_tensor, action_space).squeeze(0).cpu().numpy()
        target_advantage = None
        if type_index is not None and expected_slot is not None:
            target_advantage = float(centered[type_index, expected_slot])
        return {
            "expected_type_value": float(types[type_index]) if type_index is not None else None,
            "selected_type": str(action_space.decode(selected)["action_type"]),
            "expected_target_advantage": target_advantage,
        }

    result = {"real": measure(base)}
    for name, offsets in groups.items():
        ablated = base.copy()
        for offset in offsets:
            index = 635 + offset
            if index < ablated.size:
                ablated[index] = 0.0
        result[name] = measure(ablated)
        result[name]["expected_type_value_delta_vs_real"] = (
            result[name]["expected_type_value"] - result["real"]["expected_type_value"]
            if result[name]["expected_type_value"] is not None and result["real"]["expected_type_value"] is not None else None
        )
        result[name]["expected_target_advantage_delta_vs_real"] = (
            result[name]["expected_target_advantage"] - result["real"]["expected_target_advantage"]
            if result[name]["expected_target_advantage"] is not None and result["real"]["expected_target_advantage"] is not None else None
        )
    return result


def input_weight_norms(network: FactorizedQNetwork, base_obs_dim: int = 635) -> dict[str, float]:
    weight = network.encoder[0].weight.detach().cpu().numpy()
    context = weight[:, base_obs_dim : base_obs_dim + CONFIRMATION_CONTEXT_DIM]
    base = weight[:, :base_obs_dim]
    return {
        "base_mean_abs": float(np.mean(np.abs(base))),
        "base_l2": float(np.linalg.norm(base)),
        "confirmation_mean_abs": float(np.mean(np.abs(context))),
        "confirmation_l2": float(np.linalg.norm(context)),
        "confirmation_to_base_mean_abs_ratio": float(np.mean(np.abs(context)) / np.mean(np.abs(base))) if np.mean(np.abs(base)) else 0.0,
    }


def aggregate_context_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["expected_type_context_delta"]) for row in rows if row.get("expected_type_context_delta") is not None]
    ranks_real = [float(row["expected_type_rank_real"]) for row in rows if row.get("expected_type_rank_real") is not None]
    ranks_zero = [float(row["expected_type_rank_zero"]) for row in rows if row.get("expected_type_rank_zero") is not None]
    flips = Counter(str(row.get("flip_direction") or "") for row in rows)
    return {
        "count": len(rows),
        "mean_expected_type_context_delta": float(np.mean(deltas)) if deltas else 0.0,
        "median_expected_type_context_delta": float(np.median(deltas)) if deltas else 0.0,
        "positive_delta_count": sum(value > 0 for value in deltas),
        "negative_delta_count": sum(value < 0 for value in deltas),
        "zero_delta_count": sum(value == 0 for value in deltas),
        "context_expected_type_help_ratio": float(sum(value > 0 for value in deltas) / len(deltas)) if deltas else 0.0,
        "context_expected_type_hurt_ratio": float(sum(value < 0 for value in deltas) / len(deltas)) if deltas else 0.0,
        "expected_type_rank_real_mean": float(np.mean(ranks_real)) if ranks_real else None,
        "expected_type_rank_zero_mean": float(np.mean(ranks_zero)) if ranks_zero else None,
        "rank_delta_zero_minus_real_mean": float(np.mean([z - r for z, r in zip(ranks_zero, ranks_real)])) if ranks_real and len(ranks_real) == len(ranks_zero) else None,
        "flip_direction_counts": dict(flips),
    }
