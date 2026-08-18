"""A small masked Double-DQN implementation for non-security UI error discovery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


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


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.items: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        return random.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


def masked_argmax(network: QNetwork, state: np.ndarray, mask: np.ndarray) -> int:
    with torch.inference_mode():
        values = network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0)
    values = values.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf)
    return int(values.argmax().item())
