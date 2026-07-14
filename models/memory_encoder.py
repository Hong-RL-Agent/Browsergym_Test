"""Episode-local recurrent memory encoder for policy-safe PPO."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class MemoryEncoder(nn.Module):
    """Compact recurrent state over policy-safe observation vectors.

    The encoder receives only the already sanitized PPO observation vector. It
    does not accept site metadata, known-bug labels, target signal types, or any
    evaluation reference fields.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, encoder_type: Literal["gru", "lstm"] = "gru") -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.encoder_type = str(encoder_type or "gru").lower()
        if self.encoder_type != "gru":
            raise ValueError("Only gru memory_encoder_type is currently supported.")
        self.rnn = nn.GRUCell(self.input_size, self.hidden_size)

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(int(batch_size), self.hidden_size, dtype=torch.float32, device=device)

    def forward(self, step_features: torch.Tensor, hidden_state: torch.Tensor | None = None) -> torch.Tensor:
        if step_features.dim() == 1:
            step_features = step_features.unsqueeze(0)
        step_features = step_features.float()
        if hidden_state is None:
            hidden_state = self.initial_state(step_features.shape[0], step_features.device)
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        return self.rnn(step_features, hidden_state.float())
