"""State-Action-Outcome recurrent PPO actor-critic.

This is a new architecture.  The legacy PPOActorCritic remains unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical


RECURRENT_PPO_ARCHITECTURE = "recurrent_ppo_gru_sao_v1"


class RecurrentPPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=self.hidden_size, batch_first=True)
        self.actor = nn.Linear(self.hidden_size, self.action_dim)
        self.critic = nn.Linear(self.hidden_size, 1)

    def initial_hidden(self, batch_size: int = 1, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros((1, int(batch_size), self.hidden_size), device=device)

    def architecture_metadata(self, observation_schema_version: str, observation_schema_hash: str, action_schema_version: str, action_schema_hash: str, sequence_length: int | None = None) -> dict[str, object]:
        return {
            "architecture_version": RECURRENT_PPO_ARCHITECTURE,
            "model_type": "recurrent_ppo",
            "observation_schema_version": observation_schema_version,
            "observation_schema_hash": observation_schema_hash,
            "observation_dim": self.obs_dim,
            "action_schema_version": action_schema_version,
            "action_schema_hash": action_schema_hash,
            "action_dim": self.action_dim,
            "recurrent": True,
            "rnn_type": "GRU",
            "hidden_size": self.hidden_size,
            "sequence_length": sequence_length,
        }

    def forward(self, observations: torch.Tensor, hidden: torch.Tensor | None = None, done_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        encoded = self.encoder(observations.float())
        if hidden is None:
            hidden = self.initial_hidden(encoded.shape[0], device=encoded.device)
        if done_mask is not None:
            mask = done_mask.to(encoded.device, dtype=encoded.dtype)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            outputs = []
            current = hidden
            for index in range(encoded.shape[1]):
                current = current * (1.0 - mask[:, index].reshape(1, -1, 1))
                step, current = self.gru(encoded[:, index:index + 1], current)
                outputs.append(step)
            recurrent = torch.cat(outputs, dim=1)
            hidden = current
        else:
            recurrent, hidden = self.gru(encoded, hidden)
        logits = self.actor(recurrent)
        values = self.critic(recurrent).squeeze(-1)
        return logits, values, hidden

    def get_action(self, observation: torch.Tensor, hidden: torch.Tensor, action_mask: Optional[torch.Tensor] = None, done: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        done_mask = torch.tensor([[float(done)]], device=observation.device)
        logits, values, next_hidden = self.forward(observation, hidden, done_mask=done_mask)
        step_logits = logits[:, -1]
        step_value = values[:, -1]
        if action_mask is not None:
            mask = action_mask.to(step_logits.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            step_logits = step_logits.masked_fill(~mask, -1e9)
        distribution = Categorical(logits=step_logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), step_value, distribution.entropy(), next_hidden

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor | None = None, action_masks: torch.Tensor | None = None, done_masks: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, next_hidden = self.forward(observations, hidden=hidden, done_mask=done_masks)
        if action_masks is not None:
            mask = action_masks.to(logits.device, dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions.long()), values, distribution.entropy(), next_hidden


def recurrent_ppo_loss(
    model: RecurrentPPOActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    initial_hidden: torch.Tensor,
    *,
    action_masks: torch.Tensor | None = None,
    done_masks: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, torch.Tensor]:
    log_probs, values, entropy, _ = model.evaluate_actions(observations, actions, initial_hidden, action_masks, done_masks)
    valid = torch.ones_like(log_probs, dtype=torch.bool) if valid_mask is None else valid_mask.to(log_probs.device, dtype=torch.bool)
    if valid.dim() == 1:
        valid = valid.unsqueeze(0)
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).masked_select(valid).mean()
    value_loss = ((values - returns).pow(2)).masked_select(valid).mean()
    entropy_mean = entropy.masked_select(valid).mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    return {"loss": total, "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy_mean, "log_probs": log_probs, "values": values}


class ObservationV2HistoryMLP(nn.Module):
    """Feed-forward Observation-v2 comparison baseline without recurrence."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations.float())
        return self.actor(features), self.critic(features).squeeze(-1)
