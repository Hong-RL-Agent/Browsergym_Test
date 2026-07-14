"""Rollout storage for single-environment PPO."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: np.ndarray,
        memory_state: np.ndarray | None = None,
        info: Dict[str, Any] | None = None,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.action_masks.append(np.asarray(action_mask, dtype=np.float32))
        if memory_state is None:
            self.memory_states.append(np.zeros(0, dtype=np.float32))
        else:
            self.memory_states.append(np.asarray(memory_state, dtype=np.float32))
        self.infos.append(dict(info or {}))

    def clear(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.action_masks: List[np.ndarray] = []
        self.memory_states: List[np.ndarray] = []
        self.infos: List[Dict[str, Any]] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def extend(self, other: "RolloutBuffer") -> None:
        self.observations.extend(other.observations)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.values.extend(other.values)
        self.action_masks.extend(other.action_masks)
        self.memory_states.extend(other.memory_states)
        self.infos.extend(other.infos)
        self.returns.extend(other.returns)
        self.advantages.extend(other.advantages)

    def __len__(self) -> int:
        return len(self.actions)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)

        for step in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - float(self.dones[step])
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
            next_value = self.values[step]

        returns = advantages + np.asarray(self.values, dtype=np.float32)
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()

    def get_batches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        size = len(self)
        if size == 0:
            return
        indices = np.arange(size)
        np.random.shuffle(indices)

        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        old_log_probs = np.asarray(self.log_probs, dtype=np.float32)
        returns = np.asarray(self.returns, dtype=np.float32)
        advantages = np.asarray(self.advantages, dtype=np.float32)
        action_masks = np.asarray(self.action_masks, dtype=np.float32)
        memory_states = np.asarray(self.memory_states, dtype=np.float32)

        for start in range(0, size, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield {
                "observations": torch.from_numpy(observations[batch_indices]),
                "actions": torch.from_numpy(actions[batch_indices]),
                "old_log_probs": torch.from_numpy(old_log_probs[batch_indices]),
                "returns": torch.from_numpy(returns[batch_indices]),
                "advantages": torch.from_numpy(advantages[batch_indices]),
                "action_masks": torch.from_numpy(action_masks[batch_indices]),
                "memory_states": torch.from_numpy(memory_states[batch_indices]),
            }
