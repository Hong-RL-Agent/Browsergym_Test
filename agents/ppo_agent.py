"""Single-worker PPO agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

from models.ppo_actor_critic import PPOActorCritic


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        learning_rate: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.policy = PPOActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, obs_vector: np.ndarray, action_mask: np.ndarray) -> Dict[str, Any]:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
            action, log_prob, value, entropy = self.policy.get_action(obs, mask)
        return {
            "action_id": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
        }

    def select_greedy_action(self, obs_vector: np.ndarray, action_mask: np.ndarray) -> Dict[str, Any]:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.policy(obs)
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
            action = torch.argmax(logits, dim=-1)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return {
            "action_id": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
        }

    def score_action(self, obs_vector: np.ndarray, action_mask: np.ndarray, action_id: int) -> Dict[str, Any]:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.as_tensor([int(action_id)], dtype=torch.long, device=self.device)
            mask = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            log_prob, value, entropy = self.policy.evaluate_actions(obs, action, mask)
        return {
            "action_id": int(action_id),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
        }

    def estimate_value(self, obs_vector: np.ndarray) -> float:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value = self.policy(obs)
        return float(value.squeeze(0).item())

    def update(self, rollout_buffer) -> Dict[str, float]:
        if len(rollout_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        self.policy.train()
        metrics = []
        for _ in range(self.update_epochs):
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                returns = batch["returns"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                action_masks = batch["action_masks"].to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                log_probs, values, entropy = self.policy.evaluate_actions(obs, actions, action_masks)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_mean = entropy.mean()
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics.append(
                    {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_mean.item()),
                        "total_loss": float(total_loss.item()),
                    }
                )

        if not metrics:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        return {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_obs_dim = int(checkpoint.get("obs_dim", self.obs_dim))
        checkpoint_action_dim = int(checkpoint.get("action_dim", self.action_dim))
        if checkpoint_obs_dim != self.obs_dim or checkpoint_action_dim != self.action_dim:
            self._load_compatible_policy_state(
                checkpoint["policy_state_dict"],
                checkpoint_obs_dim,
                checkpoint_action_dim,
            )
            return
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def _load_compatible_policy_state(
        self,
        state_dict: Dict[str, Any],
        checkpoint_obs_dim: int,
        checkpoint_action_dim: int,
    ) -> None:
        current = self.policy.state_dict()
        merged = dict(current)
        for name, value in state_dict.items():
            if name not in merged:
                continue
            if tuple(value.shape) == tuple(merged[name].shape):
                merged[name] = value
                continue
            if name == "shared.0.weight" and len(value.shape) == 2:
                rows = min(value.shape[0], merged[name].shape[0])
                cols = min(value.shape[1], merged[name].shape[1])
                merged[name][:rows, :cols] = value[:rows, :cols]
            elif name == "actor.weight" and len(value.shape) == 2:
                rows = min(value.shape[0], merged[name].shape[0])
                merged[name][:rows, :] = value[:rows, :]
            elif name == "actor.bias" and len(value.shape) == 1:
                rows = min(value.shape[0], merged[name].shape[0])
                merged[name][:rows] = value[:rows]
        self.policy.load_state_dict(merged)
        print(
            f"[ppo-agent] loaded compatible checkpoint with obs_dim {checkpoint_obs_dim}, "
            f"action_dim {checkpoint_action_dim}; current obs_dim {self.obs_dim}, action_dim {self.action_dim}."
        )

    def get_policy_state_dict(self) -> Dict[str, Any]:
        return self.policy.state_dict()

    def load_policy_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dict)
