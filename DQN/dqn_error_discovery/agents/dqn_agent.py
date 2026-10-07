"""A small masked Double-DQN implementation for non-security UI error discovery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn

FLAT_MAX_SELECTION = "flat_max"
TYPE_LOGMEANEXP_SELECTION = "type_logmeanexp"
FACTORIZED_SELECTION = "factorized_type_then_candidate"
TYPE_SCORE_NORMALIZED_LOGMEANEXP = "normalized_logmeanexp"
FLAT_DQN_ARCHITECTURE = "flat"
FACTORIZED_DQN_ARCHITECTURE = "factorized_v1"
FACTORIZED_DQN_CONFOBS_ARCHITECTURE = "factorized_v1_confobs"
FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE = "factorized_v1_confobs_hist"
FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE = "factorized_v1_confobs_histv2"
FACTORIZED_DQN_ARCHITECTURE_V1 = FACTORIZED_DQN_ARCHITECTURE


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class FactorizedQNetwork(nn.Module):
    """DQN with separate action-type value and within-type candidate advantage heads."""

    def __init__(self, observation_dim: int, num_action_types: int, max_candidates: int) -> None:
        super().__init__()
        self.num_action_types = int(num_action_types)
        self.max_candidates = int(max_candidates)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.type_head = nn.Linear(256, self.num_action_types)
        self.candidate_head = nn.Linear(256, self.num_action_types * self.max_candidates)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(observations)
        return {
            "type_values": self.type_head(encoded),
            "candidate_advantages": self.candidate_head(encoded).view(
                -1,
                self.num_action_types,
                self.max_candidates,
            ),
        }


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    current_mask: np.ndarray | None = None
    replay_metadata: dict[str, bool] | None = None


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.items: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        return random.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


class ConfirmationAwareReplayBuffer(ReplayBuffer):
    """Uniform replay with a small, lifecycle-tagged confirmation stratum.

    The master ``items`` deque remains the compatibility surface.  Confirmation
    transitions are also indexed logically, without changing transition tensor
    shapes or using taxonomy/ground-truth labels.  Sampling reserves a modest
    confirmation-context portion and guarantees one successful transition when
    one exists; all other samples remain uniform over their respective groups.
    """

    def __init__(self, capacity: int = 50_000, *, confirmation_fraction: float = 0.10, confirmation_capacity: int | None = None) -> None:
        super().__init__(capacity)
        if not 0.0 < confirmation_fraction < 1.0:
            raise ValueError("confirmation_fraction must be between 0 and 1")
        self.confirmation_fraction = float(confirmation_fraction)
        self.confirmation_capacity = int(confirmation_capacity or max(1, round(capacity * confirmation_fraction)))
        self.general_items: deque[Transition] = deque(maxlen=capacity)
        self.confirmation_items: deque[Transition] = deque(maxlen=self.confirmation_capacity)
        self.successful_items: deque[Transition] = deque(maxlen=self.confirmation_capacity)
        self.sample_telemetry: dict[str, int] = {
            "batches": 0, "total_samples": 0, "general_samples": 0,
            "confirmation_samples": 0, "successful_reproduction_samples": 0,
            "failed_reproduction_samples": 0, "partial_reproduction_samples": 0,
        }

    @staticmethod
    def _is_confirmation(transition: Transition) -> bool:
        return bool((transition.replay_metadata or {}).get("is_confirmation_context"))

    @staticmethod
    def _is_successful(transition: Transition) -> bool:
        return bool((transition.replay_metadata or {}).get("is_successful_reproduction"))

    def add(self, transition: Transition) -> None:
        super().add(transition)
        if self._is_confirmation(transition):
            self.confirmation_items.append(transition)
            if self._is_successful(transition):
                self.successful_items.append(transition)
        else:
            self.general_items.append(transition)

    def sample(self, size: int, *, seed: int | None = None) -> list[Transition]:
        if size < 0 or size > len(self.items):
            raise ValueError(f"cannot sample {size} transitions from replay of size {len(self.items)}")
        if size == 0:
            return []
        rng = random.Random(seed) if seed is not None else random
        confirmation_pool = list(self.confirmation_items)
        confirmation_ids = {id(item) for item in confirmation_pool}
        confirmation_pool.extend(item for item in self.successful_items if id(item) not in confirmation_ids)
        general_pool = list(self.general_items)
        quota = min(len(confirmation_pool), max(1, int(round(size * self.confirmation_fraction))) if confirmation_pool else 0)
        selected_confirmation: list[Transition] = []
        if quota:
            successful = [item for item in confirmation_pool if self._is_successful(item)]
            if successful:
                selected_confirmation.append(rng.choice(successful))
            selected_ids = {id(item) for item in selected_confirmation}
            remaining = [item for item in confirmation_pool if id(item) not in selected_ids]
            selected_confirmation.extend(rng.sample(remaining, min(quota - len(selected_confirmation), len(remaining))))
        selected = list(selected_confirmation)
        remaining_count = size - len(selected)
        selected_ids = {id(item) for item in selected}
        available = [item for item in self.items if id(item) not in selected_ids]
        if remaining_count:
            selected.extend(rng.sample(available, remaining_count))
        rng.shuffle(selected)
        self.sample_telemetry["batches"] += 1
        self.sample_telemetry["total_samples"] += len(selected)
        self.sample_telemetry["confirmation_samples"] += sum(int(self._is_confirmation(item)) for item in selected)
        self.sample_telemetry["general_samples"] += sum(int(not self._is_confirmation(item)) for item in selected)
        self.sample_telemetry["successful_reproduction_samples"] += sum(int(self._is_successful(item)) for item in selected)
        self.sample_telemetry["failed_reproduction_samples"] += sum(int(bool((item.replay_metadata or {}).get("is_failed_reproduction"))) for item in selected)
        self.sample_telemetry["partial_reproduction_samples"] += sum(int(bool((item.replay_metadata or {}).get("is_partial_reproduction"))) for item in selected)
        return selected

    def retention_stats(self) -> dict[str, int | float]:
        successful = len(self.successful_items)
        return {
            "total": len(self.items), "general": len(self.general_items),
            "confirmation_context": len(self.confirmation_items),
            "successful_reproduction": successful,
            "confirmation_capacity": self.confirmation_capacity,
        }


def masked_argmax(network: QNetwork, state: np.ndarray, mask: np.ndarray) -> int:
    with torch.inference_mode():
        values = network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0)
    values = values.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf)
    return int(values.argmax().item())


def validate_greedy_selection_mode(mode: str) -> str:
    if mode not in {FLAT_MAX_SELECTION, TYPE_LOGMEANEXP_SELECTION, FACTORIZED_SELECTION}:
        raise ValueError(f"unsupported DQN greedy selection mode: {mode}")
    return mode


def validate_selection_mode_consistency(behavior_mode: str, target_mode: str) -> None:
    validate_greedy_selection_mode(behavior_mode)
    validate_greedy_selection_mode(target_mode)
    if behavior_mode != target_mode:
        raise ValueError(
            "DQN behavior greedy selection and TD target next-action selection "
            f"must match, got {behavior_mode!r} and {target_mode!r}"
        )


def normalized_logmeanexp(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(-torch.inf, dtype=values.dtype, device=values.device)
    return torch.logsumexp(values, dim=0) - math.log(values.numel())


def type_aware_argmax_from_q_values(q_values: torch.Tensor, mask: torch.Tensor, action_space) -> int:
    """Select type by normalized log-mean-exp, then max candidate inside that type."""
    if q_values.ndim != 1:
        raise ValueError("q_values must be a 1D tensor")
    valid_mask = mask > 0
    if not bool(torch.any(valid_mask)):
        return int(action_space.encode("noop", 0))

    best_type: str | None = None
    best_type_score: torch.Tensor | None = None
    best_action = int(action_space.encode("noop", 0))
    for action_type in action_space.action_types:
        ids = [
            action_space.encode(action_type, candidate_index)
            for candidate_index in range(action_space.max_candidates)
            if bool(valid_mask[action_space.encode(action_type, candidate_index)].item())
        ]
        if not ids:
            continue
        action_ids = torch.tensor(ids, dtype=torch.long, device=q_values.device)
        values = q_values.index_select(0, action_ids)
        score = normalized_logmeanexp(values)
        if best_type_score is None or bool(score > best_type_score):
            best_type = str(action_type)
            best_type_score = score
            best_action = int(action_ids[int(torch.argmax(values).item())].item())
    return best_action if best_type is not None else int(action_space.encode("noop", 0))


def type_aware_argmax_batch(q_values: torch.Tensor, masks: torch.Tensor, action_space) -> torch.Tensor:
    """Batch version used for Double-DQN online next-action selection."""
    actions = [
        type_aware_argmax_from_q_values(row, mask, action_space)
        for row, mask in zip(q_values, masks, strict=True)
    ]
    return torch.tensor(actions, dtype=torch.long, device=q_values.device).unsqueeze(1)


def select_greedy_action(
    network: QNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space=None,
    mode: str = FLAT_MAX_SELECTION,
) -> int:
    mode = validate_greedy_selection_mode(mode)
    if mode == FLAT_MAX_SELECTION:
        return masked_argmax(network, state, mask)
    if action_space is None:
        raise ValueError("action_space is required for type-aware greedy selection")
    with torch.inference_mode():
        values = network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0)
    return type_aware_argmax_from_q_values(values, torch.from_numpy(mask), action_space)


def double_dqn_next_actions(
    online_network: QNetwork,
    next_states: torch.Tensor,
    next_masks: torch.Tensor,
    action_space,
    mode: str = FLAT_MAX_SELECTION,
) -> torch.Tensor:
    mode = validate_greedy_selection_mode(mode)
    with torch.no_grad():
        online_q = online_network(next_states)
        if mode == FLAT_MAX_SELECTION:
            masked_q = online_q.masked_fill(~next_masks, -torch.inf)
            return masked_q.argmax(dim=1, keepdim=True)
        return type_aware_argmax_batch(online_q, next_masks, action_space)


def action_mask_to_type_candidate_mask(mask: torch.Tensor, action_space) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    batch_size = mask.shape[0]
    return mask.reshape(batch_size, len(action_space.action_types), action_space.max_candidates).bool()


def center_candidate_advantages(advantages: torch.Tensor, action_mask: torch.Tensor, action_space) -> torch.Tensor:
    candidate_mask = action_mask_to_type_candidate_mask(action_mask, action_space).to(advantages.device)
    valid = candidate_mask.to(dtype=advantages.dtype)
    counts = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
    valid_sum = (advantages * valid).sum(dim=2, keepdim=True)
    valid_mean = valid_sum / counts
    centered = advantages - valid_mean
    return torch.where(candidate_mask, centered, torch.zeros_like(centered))


def factorized_action_values(outputs: dict[str, torch.Tensor], action_mask: torch.Tensor, action_space) -> torch.Tensor:
    type_values = outputs["type_values"]
    advantages = outputs["candidate_advantages"]
    centered = center_candidate_advantages(advantages, action_mask, action_space)
    return type_values.unsqueeze(2) + centered


def factorized_flat_q_values(outputs: dict[str, torch.Tensor], action_mask: torch.Tensor, action_space) -> torch.Tensor:
    values = factorized_action_values(outputs, action_mask, action_space)
    return values.reshape(values.shape[0], len(action_space.action_types) * action_space.max_candidates)


def factorized_q_for_actions(
    network: FactorizedQNetwork,
    states: torch.Tensor,
    action_masks: torch.Tensor,
    actions: torch.Tensor,
    action_space,
) -> torch.Tensor:
    flat_values = factorized_flat_q_values(network(states), action_masks, action_space)
    action_ids = actions.reshape(-1, 1).long().to(flat_values.device)
    return flat_values.gather(1, action_ids).squeeze(1)


def factorized_greedy_actions_from_outputs(
    outputs: dict[str, torch.Tensor],
    action_mask: torch.Tensor,
    action_space,
) -> torch.Tensor:
    candidate_mask = action_mask_to_type_candidate_mask(action_mask, action_space).to(outputs["type_values"].device)
    type_valid = candidate_mask.any(dim=2)
    type_values = outputs["type_values"].masked_fill(~type_valid, -torch.inf)
    best_types = type_values.argmax(dim=1)
    centered = center_candidate_advantages(outputs["candidate_advantages"], action_mask, action_space)
    actions: list[int] = []
    for row_index, type_index in enumerate(best_types.tolist()):
        valid_candidates = candidate_mask[row_index, type_index]
        if not bool(valid_candidates.any()):
            actions.append(int(action_space.encode("noop", 0)))
            continue
        candidate_scores = centered[row_index, type_index].masked_fill(~valid_candidates, -torch.inf)
        candidate_index = int(candidate_scores.argmax().item())
        action_type = action_space.action_types[int(type_index)]
        actions.append(int(action_space.encode(action_type, candidate_index)))
    return torch.tensor(actions, dtype=torch.long, device=outputs["type_values"].device).unsqueeze(1)


def select_factorized_greedy_action(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space,
) -> int:
    with torch.inference_mode():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)
        action = factorized_greedy_actions_from_outputs(network(state_tensor), mask_tensor, action_space)
    return int(action.item())


def factorized_double_dqn_next_actions(
    online_network: FactorizedQNetwork,
    next_states: torch.Tensor,
    next_masks: torch.Tensor,
    action_space,
) -> torch.Tensor:
    with torch.no_grad():
        return factorized_greedy_actions_from_outputs(online_network(next_states), next_masks, action_space)


def factorized_q_value_diagnostics(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space,
    selected_action_id: int,
) -> dict:
    with torch.inference_mode():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)
        outputs = network(state_tensor)
        flat_values = factorized_flat_q_values(outputs, mask_tensor, action_space).squeeze(0).detach().cpu().numpy()
        type_values = outputs["type_values"].squeeze(0).detach().cpu().numpy()
        raw_advantages = outputs["candidate_advantages"].squeeze(0)
        centered = center_candidate_advantages(outputs["candidate_advantages"], mask_tensor, action_space).squeeze(0)
    valid_ids = np.flatnonzero(mask > 0)
    type_value_by_action_type = {
        action_type: float(type_values[index])
        for index, action_type in enumerate(action_space.action_types)
    }
    if len(valid_ids) == 0:
        return {
            "selected_action_q": None,
            "type_value_by_action_type": type_value_by_action_type,
            "candidate_advantage_stats": {},
        }
    valid_values = flat_values[valid_ids]
    max_by_type = {}
    candidate_stats = {}
    for action_type, action_ids in valid_actions_grouped_by_type(mask, action_space).items():
        type_index = action_space.action_types.index(action_type)
        candidate_indexes = [action_space.decode(action_id)["candidate_index"] for action_id in action_ids]
        raw = raw_advantages[type_index, candidate_indexes].detach().cpu().numpy()
        centered_values = centered[type_index, candidate_indexes].detach().cpu().numpy()
        q_values = flat_values[np.asarray(action_ids, dtype=np.int64)]
        max_by_type[action_type] = float(np.max(q_values))
        candidate_stats[action_type] = {
            "valid_count": int(len(action_ids)),
            "raw_advantage_mean": float(np.mean(raw)),
            "raw_advantage_std": float(np.std(raw)),
            "centered_advantage_mean": float(np.mean(centered_values)),
            "centered_advantage_std": float(np.std(centered_values)),
        }
    decoded = action_space.decode(int(selected_action_id))
    return {
        "selected_action_q": float(flat_values[int(selected_action_id)]),
        "selected_type_value": type_value_by_action_type.get(str(decoded["action_type"])),
        "selected_candidate_centered_advantage": float(
            centered[int(decoded["action_type_id"]), int(decoded["candidate_index"])].item()
        ),
        "max_valid_q": float(np.max(valid_values)),
        "mean_valid_q": float(np.mean(valid_values)),
        "min_valid_q": float(np.min(valid_values)),
        "max_q_by_action_type": max_by_type,
        "type_value_by_action_type": type_value_by_action_type,
        "candidate_advantage_stats": candidate_stats,
    }


def q_value_diagnostics(
    network: QNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space,
    selected_action_id: int,
    top_k: int = 5,
) -> dict:
    with torch.inference_mode():
        raw_values = network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    valid_ids = np.flatnonzero(mask > 0)
    if len(valid_ids) == 0:
        return {
            "selected_action_q": None,
            "max_valid_q": None,
            "mean_valid_q": None,
            "min_valid_q": None,
            "top_k_valid_actions": [],
            "max_q_by_action_type": {},
            "type_scores_by_action_type": {},
            "selected_type_score": None,
            "selected_candidate_q": None,
            "selected_type_valid_count": 0,
            "click_type_score": None,
            "click_candidate_max_q": None,
        }
    valid_values = raw_values[valid_ids]
    order = np.argsort(valid_values)[::-1][:top_k]
    top_actions = []
    for position in order:
        action_id = int(valid_ids[position])
        decoded = action_space.decode(action_id)
        top_actions.append(
            {
                "action_id": action_id,
                "action_type": decoded["action_type"],
                "candidate_index": int(decoded["candidate_index"]),
                "q": float(raw_values[action_id]),
            }
        )
    max_by_type = {}
    type_scores = {}
    valid_count_by_type = {}
    for action_id in valid_ids:
        decoded = action_space.decode(int(action_id))
        action_type = str(decoded["action_type"])
        value = float(raw_values[int(action_id)])
        if action_type not in max_by_type or value > max_by_type[action_type]:
            max_by_type[action_type] = value
    for action_type, action_ids in valid_actions_grouped_by_type(mask, action_space).items():
        values = raw_values[np.asarray(action_ids, dtype=np.int64)]
        if len(values) == 0:
            continue
        max_value = float(np.max(values))
        type_scores[action_type] = float(max_value + np.log(np.mean(np.exp(values - max_value))))
        valid_count_by_type[action_type] = int(len(values))
    selected_decoded = action_space.decode(int(selected_action_id)) if 0 <= int(selected_action_id) < len(raw_values) else {}
    selected_type = str(selected_decoded.get("action_type") or "")
    click_actions = valid_actions_grouped_by_type(mask, action_space).get("click_element", [])
    return {
        "selected_action_q": float(raw_values[int(selected_action_id)]) if 0 <= int(selected_action_id) < len(raw_values) else None,
        "max_valid_q": float(np.max(valid_values)),
        "mean_valid_q": float(np.mean(valid_values)),
        "min_valid_q": float(np.min(valid_values)),
        "top_k_valid_actions": top_actions,
        "max_q_by_action_type": max_by_type,
        "type_scores_by_action_type": type_scores,
        "selected_type_score": type_scores.get(selected_type),
        "selected_candidate_q": float(raw_values[int(selected_action_id)]) if 0 <= int(selected_action_id) < len(raw_values) else None,
        "selected_type_valid_count": valid_count_by_type.get(selected_type, 0),
        "click_type_score": type_scores.get("click_element"),
        "click_candidate_max_q": float(np.max(raw_values[np.asarray(click_actions, dtype=np.int64)])) if click_actions else None,
    }


def random_valid_action(mask: np.ndarray, rng: random.Random | None = None) -> int:
    choices = np.flatnonzero(mask > 0)
    if len(choices) == 0:
        return 0
    chooser = rng or random
    return int(chooser.choice(choices.tolist()))


def balanced_random_valid_action(mask: np.ndarray, action_space, rng: random.Random | None = None) -> int:
    """Sample a valid action by choosing action type first, then slot.

    This is only for epsilon exploration. It keeps invalid actions masked out
    while preventing action types with many candidate slots from dominating
    random exploration.
    """
    groups = valid_actions_grouped_by_type(mask, action_space)
    if not groups:
        return 0
    chooser = rng or random
    action_type = chooser.choice(sorted(groups))
    return int(chooser.choice(groups[action_type]))


def valid_actions_grouped_by_type(mask: np.ndarray, action_space) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for action_id in np.flatnonzero(mask > 0):
        decoded = action_space.decode(int(action_id))
        groups.setdefault(str(decoded["action_type"]), []).append(int(action_id))
    return groups


def balanced_expected_action_type_distribution(mask: np.ndarray, action_space) -> dict[str, float]:
    groups = valid_actions_grouped_by_type(mask, action_space)
    if not groups:
        return {}
    probability = 1.0 / len(groups)
    return {action_type: probability for action_type in groups}
