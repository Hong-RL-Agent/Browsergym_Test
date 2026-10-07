"""Derived v3 observation/action dimensions; no copied magic totals."""

from __future__ import annotations

from .action_space import V3ActionSpace


PPO_OBS_DIM = 635
CONFIRMATION_FIXED_PREFIX = 16
CONFIRMATION_TAIL_DIM = 4
HISTORY_WINDOW = 4
HISTORY_TARGET_SLOT_DIM = 32
HISTORY_RISK_FLAGS = 3


def confirmation_obs_dim(action_type_count: int) -> int:
    return CONFIRMATION_FIXED_PREFIX + action_type_count + 1 + HISTORY_TARGET_SLOT_DIM + CONFIRMATION_TAIL_DIM


def history_v2_dim(action_type_count: int) -> int:
    action_one_hot = HISTORY_WINDOW * action_type_count
    target_required = HISTORY_WINDOW
    target_slots = HISTORY_WINDOW * HISTORY_TARGET_SLOT_DIM
    current_target = HISTORY_WINDOW
    prospective_risk = action_type_count * HISTORY_RISK_FLAGS
    return action_one_hot + target_required + target_slots + current_target + prospective_risk


def v3_dimensions(max_candidates: int = 32) -> dict[str, int]:
    action_space = V3ActionSpace(max_candidates)
    confirmation = confirmation_obs_dim(len(action_space.action_types))
    history = history_v2_dim(len(action_space.action_types))
    return {
        "ppo_obs_dim": PPO_OBS_DIM,
        "confirmation_obs_dim": confirmation,
        "history_v2_dim": history,
        "dqn_obs_dim": PPO_OBS_DIM + confirmation + history,
        "action_type_count": len(action_space.action_types),
        "candidate_count": max_candidates,
        "action_dim": action_space.get_action_dim(),
    }
