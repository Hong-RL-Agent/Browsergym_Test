"""Single-worker PPO agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

from models.memory_encoder import MemoryEncoder
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
        use_memory_encoder: bool = False,
        memory_encoder_type: str = "gru",
        memory_hidden_size: int = 128,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_memory_encoder = bool(use_memory_encoder)
        self.memory_encoder_type = str(memory_encoder_type or "gru").lower()
        self.memory_hidden_size = int(memory_hidden_size) if self.use_memory_encoder else 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.memory_encoder = (
            MemoryEncoder(obs_dim, self.memory_hidden_size, self.memory_encoder_type).to(self.device)
            if self.use_memory_encoder
            else None
        )
        self.policy = PPOActorCritic(obs_dim, action_dim, memory_dim=self.memory_hidden_size).to(self.device)
        parameters = list(self.policy.parameters())
        if self.memory_encoder is not None:
            parameters.extend(self.memory_encoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def reset_memory_state(self) -> np.ndarray:
        if not self.use_memory_encoder or self.memory_encoder is None:
            return np.zeros(0, dtype=np.float32)
        return self.memory_encoder.initial_state(1, self.device).squeeze(0).detach().cpu().numpy()

    def update_memory_state(self, obs_vector: np.ndarray, memory_state: np.ndarray | None = None) -> np.ndarray:
        if not self.use_memory_encoder or self.memory_encoder is None:
            return np.zeros(0, dtype=np.float32)
        self.memory_encoder.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            hidden = self._memory_tensor(memory_state)
            next_hidden = self.memory_encoder(obs, hidden).squeeze(0)
        return next_hidden.detach().cpu().numpy().astype(np.float32)

    def select_action(
        self,
        obs_vector: np.ndarray,
        action_mask: np.ndarray,
        memory_state: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            memory = self._next_memory_tensor(obs, memory_state)
            policy_obs = self._policy_input(obs, memory)
            mask = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
            action, log_prob, value, entropy = self.policy.get_action(policy_obs, mask)
        return {
            "action_id": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
            "memory_state": memory.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "memory_state_norm": float(torch.linalg.vector_norm(memory).item()),
        }

    def select_greedy_action(
        self,
        obs_vector: np.ndarray,
        action_mask: np.ndarray,
        memory_state: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            memory = self._next_memory_tensor(obs, memory_state)
            policy_obs = self._policy_input(obs, memory)
            logits, value = self.policy(policy_obs)
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
            "memory_state": memory.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "memory_state_norm": float(torch.linalg.vector_norm(memory).item()),
        }

    def score_action(
        self,
        obs_vector: np.ndarray,
        action_mask: np.ndarray,
        action_id: int,
        memory_state: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            memory = self._coerce_memory_tensor(memory_state)
            policy_obs = self._policy_input(obs, memory)
            action = torch.as_tensor([int(action_id)], dtype=torch.long, device=self.device)
            mask = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            log_prob, value, entropy = self.policy.evaluate_actions(policy_obs, action, mask)
        return {
            "action_id": int(action_id),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
            "memory_state": memory.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "memory_state_norm": float(torch.linalg.vector_norm(memory).item()) if self.use_memory_encoder else 0.0,
        }

    def estimate_value(self, obs_vector: np.ndarray, memory_state: np.ndarray | None = None) -> float:
        self.policy.eval()
        with torch.no_grad():
            obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=self.device)
            memory = self._coerce_memory_tensor(memory_state)
            policy_obs = self._policy_input(obs, memory)
            _, value = self.policy(policy_obs)
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
                memory_states = batch.get("memory_states")
                if memory_states is not None:
                    memory_states = memory_states.to(self.device)
                policy_obs = self._batch_policy_input(obs, memory_states)

                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                log_probs, values, entropy = self.policy.evaluate_actions(policy_obs, actions, action_masks)
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
                "use_memory_encoder": self.use_memory_encoder,
                "memory_encoder_type": self.memory_encoder_type,
                "memory_hidden_size": self.memory_hidden_size,
                "policy_state_dict": self.policy.state_dict(),
                "memory_encoder_state_dict": self.memory_encoder.state_dict() if self.memory_encoder is not None else None,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_obs_dim = int(checkpoint.get("obs_dim", self.obs_dim))
        checkpoint_action_dim = int(checkpoint.get("action_dim", self.action_dim))
        checkpoint_memory_hidden_size = int(checkpoint.get("memory_hidden_size", 0) or 0)
        if checkpoint_obs_dim != self.obs_dim or checkpoint_action_dim != self.action_dim:
            self._load_compatible_policy_state(
                checkpoint["policy_state_dict"],
                checkpoint_obs_dim,
                checkpoint_action_dim,
            )
            return
        if checkpoint_memory_hidden_size != self.memory_hidden_size:
            self._load_compatible_policy_state(
                checkpoint["policy_state_dict"],
                checkpoint_obs_dim,
                checkpoint_action_dim,
            )
            return
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if self.memory_encoder is not None and checkpoint.get("memory_encoder_state_dict"):
            self.memory_encoder.load_state_dict(checkpoint["memory_encoder_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                pass

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

    def _memory_tensor(self, memory_state: np.ndarray | None) -> torch.Tensor:
        if not self.use_memory_encoder:
            return torch.zeros(1, 0, dtype=torch.float32, device=self.device)
        if memory_state is None or np.asarray(memory_state).size == 0:
            return torch.zeros(1, self.memory_hidden_size, dtype=torch.float32, device=self.device)
        tensor = torch.as_tensor(memory_state, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _coerce_memory_tensor(self, memory_state: np.ndarray | None) -> torch.Tensor:
        return self._memory_tensor(memory_state)

    def _next_memory_tensor(self, obs: torch.Tensor, memory_state: np.ndarray | None) -> torch.Tensor:
        if not self.use_memory_encoder or self.memory_encoder is None:
            return torch.zeros(1, 0, dtype=torch.float32, device=self.device)
        hidden = self._memory_tensor(memory_state)
        return self.memory_encoder(obs, hidden)

    def _policy_input(self, obs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if not self.use_memory_encoder:
            return obs.float()
        if memory.dim() == 1:
            memory = memory.unsqueeze(0)
        return torch.cat([obs.float(), memory.float()], dim=-1)

    def _batch_policy_input(self, obs: torch.Tensor, memory_states: torch.Tensor | None) -> torch.Tensor:
        if not self.use_memory_encoder:
            return obs.float()
        if memory_states is None or memory_states.numel() == 0:
            memory_states = torch.zeros(obs.shape[0], self.memory_hidden_size, dtype=torch.float32, device=self.device)
        return torch.cat([obs.float(), memory_states.float()], dim=-1)
