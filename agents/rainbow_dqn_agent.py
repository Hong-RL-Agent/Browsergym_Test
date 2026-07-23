"""Action-masked Rainbow DQN: Double, Dueling, C51, PER, n-step and noisy nets."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import torch

from agents.prioritized_replay import PrioritizedReplayBuffer, Transition
from models.rainbow_dqn import RainbowDQN


class RainbowDQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, *, gamma: float = 0.99, n_step: int = 3,
                 learning_rate: float = 1e-4, batch_size: int = 16, replay_capacity: int = 50_000,
                 warmup_steps: int = 16, target_update_interval: int = 250, atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 20.0, device: str = "cpu") -> None:
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.gamma, self.n_step, self.batch_size = gamma, n_step, batch_size
        self.warmup_steps, self.target_update_interval = warmup_steps, target_update_interval
        self.atoms, self.v_min, self.v_max = atoms, v_min, v_max
        self.device = torch.device(device)
        self.support = torch.linspace(v_min, v_max, atoms, device=self.device)
        self.delta_z = (v_max - v_min) / (atoms - 1)
        self.online = RainbowDQN(obs_dim, action_dim, atoms).to(self.device)
        self.target = RainbowDQN(obs_dim, action_dim, atoms).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = PrioritizedReplayBuffer(replay_capacity)
        self.n_step_queue: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque()
        self.training_steps = 0

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, training: bool = True) -> dict[str, Any]:
        self.online.train(training)
        if training:
            self.online.reset_noise()
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online.q_values(tensor, self.support).squeeze(0)
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask, -torch.inf)
            action = int(q_values.argmax().item())
            q_value = float(q_values[action].item())
        return {"action_id": action, "q_value": q_value}

    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray,
                done: bool, next_action_mask: np.ndarray) -> None:
        self.n_step_queue.append((obs.copy(), action, reward, next_obs.copy(), done, next_action_mask.copy()))
        if len(self.n_step_queue) >= self.n_step or done:
            self._flush_one()
        if done:
            while self.n_step_queue:
                self._flush_one()

    def _flush_one(self) -> None:
        reward, discount = 0.0, 1.0
        next_obs, done, next_mask = self.n_step_queue[0][3:]
        for _, _, step_reward, step_next_obs, step_done, step_mask in list(self.n_step_queue)[: self.n_step]:
            reward += discount * step_reward
            next_obs, done, next_mask = step_next_obs, step_done, step_mask
            if step_done:
                break
            discount *= self.gamma
        obs, action = self.n_step_queue[0][:2]
        self.replay.add(Transition(obs, action, reward, next_obs, done, next_mask, 0.0 if done else discount))
        self.n_step_queue.popleft()

    def update(self) -> dict[str, float]:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return {"loss": 0.0, "replay_size": float(len(self.replay)), "trained": 0.0}
        beta = min(1.0, 0.4 + 0.6 * self.training_steps / 100_000)
        batch, indices, weights = self.replay.sample(self.batch_size, beta)
        obs = torch.as_tensor(np.stack([x.observation for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([x.next_observation for x in batch]), dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor([x.discount for x in batch], dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([x.next_action_mask for x in batch]), dtype=torch.bool, device=self.device)
        importance = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        distribution = self.online(obs)[range(len(batch)), actions]
        with torch.no_grad():
            next_q = self.online.q_values(next_obs, self.support).masked_fill(~masks, -torch.inf)
            next_actions = next_q.argmax(dim=1)
            next_dist = self.target(next_obs)[range(len(batch)), next_actions]
            target_z = rewards.unsqueeze(1) + discounts.unsqueeze(1) * self.support.unsqueeze(0)
            target_z = target_z.clamp(self.v_min, self.v_max)
            b = (target_z - self.v_min) / self.delta_z
            lower, upper = b.floor().long(), b.ceil().long()
            projected = torch.zeros_like(next_dist)
            offset = (torch.arange(len(batch), device=self.device) * self.atoms).unsqueeze(1)
            projected.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * (upper.float() - b + (lower == upper))).view(-1))
            projected.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1))
        per_item = -(projected * distribution.log()).sum(dim=1)
        loss = (importance * per_item).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.replay.update_priorities(indices, per_item.detach().cpu().numpy())
        self.training_steps += 1
        self.online.reset_noise()
        if self.training_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.item()), "replay_size": float(len(self.replay)), "beta": beta, "trained": 1.0}

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"obs_dim": self.obs_dim, "action_dim": self.action_dim,
                    "online_state_dict": self.online.state_dict(), "target_state_dict": self.target.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(), "training_steps": self.training_steps}, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint.get("target_state_dict", checkpoint["online_state_dict"]))
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = int(checkpoint.get("training_steps", 0))
