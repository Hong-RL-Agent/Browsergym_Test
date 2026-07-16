"""Dueling categorical Q-network used by the masked Rainbow DQN agent."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.std_init = std_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = _scaled_noise(self.in_features, self.weight_mu.device)
        eps_out = _scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(value, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, atoms: int = 51) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.value_hidden = NoisyLinear(128, 64)
        self.value_out = NoisyLinear(64, atoms)
        self.adv_hidden = NoisyLinear(128, 64)
        self.adv_out = NoisyLinear(64, action_dim * atoms)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs.float())
        value = self.value_out(F.relu(self.value_hidden(features))).view(-1, 1, self.atoms)
        advantage = self.adv_out(F.relu(self.adv_hidden(features))).view(-1, self.action_dim, self.atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1).clamp(min=1e-6)

    def q_values(self, obs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return (self(obs) * support.view(1, 1, -1)).sum(dim=-1)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def _scaled_noise(size: int, device: torch.device) -> torch.Tensor:
    value = torch.randn(size, device=device)
    return value.sign() * value.abs().sqrt()
