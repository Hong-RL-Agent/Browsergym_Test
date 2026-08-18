"""Train/evaluate a DQN for ordinary BrowserGym UI and runtime error discovery.

This runner intentionally excludes infrastructure, authorization, and security
inspection actions. It records anomaly categories and evidence references only;
it does not attempt to exploit, bypass, or reproduce security weaknesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from dqn_error_discovery.agents.dqn_agent import QNetwork, ReplayBuffer, Transition, masked_argmax
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from services.anomaly_detection_service import detect_anomalies


EXCLUDED_ACTIONS = {
    "inspect_server_health", "inspect_port_status", "inspect_latency",
    "inspect_server_logs", "inspect_runtime_metrics",
}


def safe_action_mask(action_space: ActionSpace, observation: dict) -> np.ndarray:
    mask = action_space.build_action_mask(observation)
    for action_type in EXCLUDED_ACTIONS:
        for index in range(action_space.max_candidates):
            mask[action_space.encode(action_type, index)] = 0.0
    if not mask.any():
        mask[action_space.encode("noop", 0)] = 1.0
    return mask


def anomaly_reward(anomalies: list[dict]) -> float:
    """Reward useful, non-security UI findings without rewarding action errors."""
    if not anomalies:
        return 0.0
    return float(max(0.0, min(1.0, max(float(item.get("confidence", 0.0)) for item in anomalies))))


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = BrowserGymJAWSEnv(
        site_id=args.site_id, base_url=args.base_url, max_steps=args.max_steps,
        max_candidates=args.max_candidates, headless=args.headless,
    )
    action_space = env.action_space
    observation_dim = env.observation_encoder.get_obs_dim()
    action_dim = action_space.get_action_dim()
    online = QNetwork(observation_dim, action_dim)
    target = QNetwork(observation_dim, action_dim); target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.replay_capacity)
    rewards: list[float] = []
    losses: list[float] = []
    findings: dict[str, int] = {}

    try:
        for episode in range(args.episodes):
            raw, _ = env.reset()
            state = env.observation_encoder.encode_observation(raw)
            mask = safe_action_mask(action_space, raw)
            episode_reward = 0.0
            done = False
            while not done:
                epsilon = max(args.epsilon_end, args.epsilon_start - episode / max(1, args.episodes))
                if random.random() < epsilon:
                    choices = np.flatnonzero(mask > 0)
                    action = int(random.choice(choices.tolist()))
                else:
                    action = masked_argmax(online, state, mask)
                next_raw, _, done, info = env.step(action)
                decoded = action_space.decode(action)
                anomalies = detect_anomalies(raw, next_raw, {**info, "action": decoded})
                for anomaly in anomalies:
                    kind = str(anomaly.get("type", "unknown"))
                    findings[kind] = findings.get(kind, 0) + 1
                reward = anomaly_reward(anomalies)
                next_state = env.observation_encoder.encode_observation(next_raw)
                next_mask = safe_action_mask(action_space, next_raw)
                replay.add(Transition(state, action, reward, next_state, done, next_mask))
                episode_reward += reward
                raw, state, mask = next_raw, next_state, next_mask
                if len(replay) >= args.batch_size:
                    batch = replay.sample(args.batch_size)
                    states = torch.from_numpy(np.stack([x.state for x in batch])).float()
                    actions = torch.tensor([x.action for x in batch]).long().unsqueeze(1)
                    batch_rewards = torch.tensor([x.reward for x in batch]).float()
                    next_states = torch.from_numpy(np.stack([x.next_state for x in batch])).float()
                    dones = torch.tensor([x.done for x in batch]).float()
                    next_masks = torch.from_numpy(np.stack([x.next_mask for x in batch])).bool()
                    q_values = online(states).gather(1, actions).squeeze(1)
                    with torch.no_grad():
                        next_online = online(next_states).masked_fill(~next_masks, -torch.inf)
                        next_actions = next_online.argmax(dim=1, keepdim=True)
                        next_q = target(next_states).gather(1, next_actions).squeeze(1)
                        expected = batch_rewards + args.gamma * next_q * (1.0 - dones)
                    loss = nn.functional.smooth_l1_loss(q_values, expected)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 10.0); optimizer.step()
                    losses.append(float(loss.detach()))
            rewards.append(episode_reward)
            if (episode + 1) % args.target_update == 0:
                target.load_state_dict(online.state_dict())
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "browsergym-ui-dqn-v1", "algorithm": "masked-double-dqn",
        "episodes": args.episodes, "seed": args.seed,
        "observation_dim": observation_dim, "action_dim": action_dim,
        "excluded_action_types": sorted(EXCLUDED_ACTIONS),
        "mean_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "finding_counts": findings,
        "base_url_sha256": hashlib.sha256(args.base_url.encode()).hexdigest(),
    }
    torch.save({"model": online.state_dict(), "manifest": manifest}, output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/browsergym_ui_dqn.pt"))
    parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=32); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=50000); parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0); parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
