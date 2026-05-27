"""PPO actor-critic network for J.A.W.S BrowserGym training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs.float())
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits, value = self.forward(obs)
        masked_logits = _apply_action_mask(logits, action_mask)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0), entropy.squeeze(0)

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs_batch)
        masked_logits = _apply_action_mask(logits, action_masks)
        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(action_batch.long())
        entropy = dist.entropy()
        return log_probs, values, entropy


def _apply_action_mask(logits: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if action_mask is None:
        return logits
    mask = action_mask.to(device=logits.device, dtype=torch.bool)
    if mask.dim() == 1 and logits.dim() == 2:
        mask = mask.unsqueeze(0)
    return logits.masked_fill(~mask, -1e9)
