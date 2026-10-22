"""Sequence-aware rollout storage for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class RecurrentTransition:
    observation: np.ndarray
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    hidden_before: np.ndarray
    episode_id: str


class RecurrentRolloutBuffer:
    def __init__(self) -> None:
        self.items: list[RecurrentTransition] = []

    def add(self, transition: RecurrentTransition) -> None:
        self.items.append(transition)

    def clear(self) -> None:
        self.items.clear()

    def sequences(self, sequence_length: int) -> list[dict[str, torch.Tensor | list[str]]]:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        batches: list[dict[str, torch.Tensor | list[str]]] = []
        start = 0
        while start < len(self.items):
            episode_id = self.items[start].episode_id
            rows: list[RecurrentTransition] = []
            index = start
            while index < len(self.items) and len(rows) < sequence_length and self.items[index].episode_id == episode_id:
                rows.append(self.items[index])
                index += 1
                if rows[-1].done:
                    break
            if rows:
                batches.append({
                    "observations": torch.from_numpy(np.stack([row.observation for row in rows])).float().unsqueeze(0),
                    "actions": torch.tensor([[row.action for row in rows]], dtype=torch.long),
                    "rewards": torch.tensor([[row.reward for row in rows]], dtype=torch.float32),
                    "dones": torch.tensor([[float(row.done) for row in rows]], dtype=torch.float32),
                    "values": torch.tensor([[row.value for row in rows]], dtype=torch.float32),
                    "log_probs": torch.tensor([[row.log_prob for row in rows]], dtype=torch.float32),
                    "initial_hidden": torch.from_numpy(rows[0].hidden_before).float(),
                    "valid_mask": torch.ones((1, len(rows)), dtype=torch.bool),
                    "episode_ids": [episode_id],
                })
            start = max(index, start + 1)
        return batches
