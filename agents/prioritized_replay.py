"""Small proportional prioritized replay buffer for BrowserGym transitions."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any
import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    next_action_mask: np.ndarray
    discount: float


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 50_000, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.items: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transition: Transition) -> None:
        priority = float(self.priorities[: len(self.items)].max()) if self.items else 1.0
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.position] = transition
        self.priorities[self.position] = max(priority, 1e-5)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        count = len(self.items)
        scaled = self.priorities[:count] ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(count, batch_size, replace=count < batch_size, p=probabilities)
        weights = (count * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return [self.items[int(index)] for index in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = max(float(priority), 1e-5)
