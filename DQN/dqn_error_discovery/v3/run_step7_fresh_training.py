"""Short, real-environment Step-7 fresh-training runner.

This runner is deliberately small and contract-first.  It initializes new
History-MLP and GRU policies, collects BrowserGym SITE100 rollouts, performs
PPO updates, and writes non-production checkpoints with complete metadata.
It never loads legacy weights and never uses ground truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback
import argparse
from typing import Any
import socket
from urllib.parse import urlparse

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_types import action_space_mapping_hash, action_space_metadata
from models.observation_v2 import ObservationV2Encoder, OBSERVATION_V2_DIM, observation_v2_schema, CoverageMemory, functional_state_fingerprint
from models.recurrent_ppo import RecurrentPPOActorCritic, recurrent_ppo_loss
from models.recurrent_ppo import ObservationV2HistoryMLP
from services.autonomous_reward_service import calculate_autonomous_reward, PPO_REWARD_SCHEMA_VERSION

OUT = ROOT / "artifacts/training_v3/ppo_fresh_training"
SEEDS = (42, 43)
EPISODES = 2
MAX_STEPS = 4


def _reachable(base_url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(base_url)
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            return True
    except OSError:
        return False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _save(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256(path)


def run_model(model_name: str, seed: int, train_sites: list[dict[str, Any]]) -> dict[str, Any]:
    seed_all(seed)
    recurrent = model_name == "gru"
    model = RecurrentPPOActorCritic(OBSERVATION_V2_DIM, 640, 128) if recurrent else ObservationV2HistoryMLP(OBSERVATION_V2_DIM, 640)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    encoder = ObservationV2Encoder(32)
    rows: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    optimizer_steps = 0
    runtime_errors: list[str] = []
    for episode in range(EPISODES):
        env = None
        try:
            site = train_sites[episode % len(train_sites)]
            env = BrowserGymJAWSEnv(site_id=str(site["site_id"]), base_url=str(site["base_url"]), max_steps=MAX_STEPS, max_candidates=32, headless=True, requires_login=bool(site.get("requires_login")), login_config=site.get("login"))
            raw, _ = env.reset()
            memory = CoverageMemory(); history: dict[str, Any] = {}
            previous_action: dict[str, Any] = {}; previous_outcome: dict[str, Any] = {}
            hidden = model.initial_hidden(1) if recurrent else None
            trajectory: list[dict[str, Any]] = []
            for step in range(MAX_STEPS):
                vec = encoder.encode(raw, previous_action=previous_action, previous_outcome=previous_outcome, coverage=memory)
                mask_np = env.action_space.build_action_mask(raw)
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool).reshape(1, -1)
                obs_t = torch.from_numpy(vec).float()
                if recurrent:
                    logits, values, next_hidden = model(obs_t.reshape(1, 1, -1), hidden, done_mask=torch.zeros((1, 1)))
                    step_logits = logits[:, -1].masked_fill(~mask_t, -1e9)
                    value = values[:, -1]
                else:
                    logits, value = model(obs_t.reshape(1, -1))
                    step_logits = logits.masked_fill(~mask_t, -1e9)
                    next_hidden = None
                dist = Categorical(logits=step_logits)
                action_t = dist.sample(); action_id = int(action_t.item())
                decoded = env.action_space.decode(action_id)
                action = {"action_type": decoded["action_type"], "candidate_index": decoded.get("candidate_index", 0), "action_id": action_id, "execution_success": True}
                candidates = raw.get("candidate_elements", []) if isinstance(raw, dict) else []
                slot = int(decoded.get("candidate_index", 0) or 0)
                if decoded["action_type"] in {"click_element", "fill_input", "select_option", "press_enter"} and isinstance(candidates, list) and slot < len(candidates) and isinstance(candidates[slot], dict):
                    bid = str(candidates[slot].get("bid") or "")
                    action["target_identity"] = bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
                before_fp, _ = functional_state_fingerprint(raw)
                memory.visited_states.add(before_fp)
                next_raw, info, done, _ = env.step(action_id)
                after = dict(next_raw); after["runtime_signals"] = dict(info or {}) if isinstance(info, dict) else {}
                reward, components = calculate_autonomous_reward(raw, after, action, [], [], history, {})
                after_fp, _ = functional_state_fingerprint(after)
                coverage = memory.observe(after_fp, action, components)
                trajectory.append({"vector": vec, "mask": mask_np, "action": action_id, "log_prob": float(dist.log_prob(action_t).item()), "value": float(value.item()), "reward": float(reward), "done": bool(done), "hidden": hidden.detach().clone() if recurrent else None})
                rows.append({"model": model_name, "seed": seed, "episode": episode, "step": step, "action_type": decoded["action_type"], "action_id": action_id, "reward": float(reward), "components": components, "coverage": coverage, "runtime_error": False})
                previous_action = {"action_type": decoded["action_type"], "candidate_slot": slot, "execution_success": True}
                previous_outcome = {"execution_success": True, "network_request": bool(isinstance(info, dict) and info.get("network_events"))}
                raw = next_raw
                if recurrent: hidden = next_hidden
                if done: break
            episode_rewards.append(float(sum(item["reward"] for item in trajectory)))
            if trajectory:
                rewards = torch.tensor([item["reward"] for item in trajectory], dtype=torch.float32).reshape(1, -1)
                returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
                values = torch.tensor([item["value"] for item in trajectory], dtype=torch.float32).reshape(1, -1)
                advantages = returns - values
                observations = torch.from_numpy(np.stack([item["vector"] for item in trajectory])).float().reshape(1, len(trajectory), -1)
                actions = torch.tensor([[item["action"] for item in trajectory]], dtype=torch.long)
                old_log_probs = torch.tensor([[item["log_prob"] for item in trajectory]], dtype=torch.float32)
                masks = torch.from_numpy(np.stack([item["mask"] for item in trajectory])).bool().reshape(1, len(trajectory), -1)
                valid = torch.ones((1, len(trajectory)), dtype=torch.bool)
                if recurrent:
                    initial_hidden = trajectory[0]["hidden"]
                    metrics = recurrent_ppo_loss(model, observations, actions, old_log_probs, advantages, returns, initial_hidden, action_masks=masks, done_masks=torch.zeros((1, len(trajectory))), valid_mask=valid)
                else:
                    out, values_pred = model(observations.reshape(len(trajectory), -1)); dist2 = Categorical(logits=out.masked_fill(~masks.reshape(len(trajectory), -1), -1e9))
                    logp = dist2.log_prob(actions.reshape(-1)); ratio = torch.exp(logp - old_log_probs.reshape(-1)); adv = advantages.reshape(-1); policy_loss = -torch.minimum(ratio * adv, torch.clamp(ratio, .8, 1.2) * adv).mean(); value_loss = torch.nn.functional.mse_loss(values_pred, returns.reshape(-1)); entropy = dist2.entropy().mean(); metrics = {"loss": policy_loss + .5 * value_loss - .01 * entropy, "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}
                optimizer.zero_grad(); metrics["loss"].backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step(); optimizer_steps += 1
                if not all(bool(torch.isfinite(value.detach()).all().item()) for value in metrics.values() if torch.is_tensor(value)):
                    runtime_errors.append("NONFINITE_TRAINING_METRIC")
        except Exception as exc:
            runtime_errors.append(traceback.format_exc())
        finally:
            if env is not None: env.close()
    checkpoint = OUT / f"{model_name}_seed{seed}.pt"
    schema = observation_v2_schema()
    metadata = {"model_type": model_name, "architecture": "recurrent_ppo_gru_sao_v1" if recurrent else "observation_v2_history_mlp_v1", "observation_dim": OBSERVATION_V2_DIM, "observation_schema_version": schema["schema_version"], "observation_schema_hash": schema["schema_hash"], "action_dim": 640, "action_schema_hash": action_space_mapping_hash(), "reward_version": PPO_REWARD_SCHEMA_VERSION, "seed": seed, "episodes": EPISODES, "max_steps": MAX_STEPS, "optimizer_steps": optimizer_steps, "train_split_id": "UNSET", "gt_runtime_access": False, "legacy_initialization": False}
    payload = {"metadata": metadata, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    metadata["checkpoint_sha256"] = _save(checkpoint, payload)
    return {"model": model_name, "seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": metadata["checkpoint_sha256"], "episodes": EPISODES, "steps": len(rows), "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0, "optimizer_steps": optimizer_steps, "runtime_errors": runtime_errors, "rows": rows, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-config", default=str(ROOT.parent / "configs/step7_site_split.json"))
    parser.add_argument("--site-config", default=str(ROOT.parent / "configs/training_sites.json"))
    args = parser.parse_args()
    split_cfg = json.loads(Path(args.split_config).read_text(encoding="utf-8"))
    site_cfg = json.loads(Path(args.site_config).read_text(encoding="utf-8"))
    split = {key: list(split_cfg.get(key, [])) for key in ("train", "validation", "test")}
    overlap = {"train_validation": len(set(split["train"]) & set(split["validation"])), "train_test": len(set(split["train"]) & set(split["test"])), "validation_test": len(set(split["validation"]) & set(split["test"]))}
    if not all(split.values()) or any(overlap.values()):
        raise RuntimeError("STEP7_SITE_LEVEL_SPLIT_INVALID")
    site_by_id = {str(item["site_id"]): item for item in site_cfg.get("sites", [])}
    train_sites = [site_by_id[site_id] for site_id in split["train"] if site_id in site_by_id]
    if len(train_sites) != len(split["train"]):
        raise RuntimeError("STEP7_TRAIN_SITE_MISSING_FROM_CONFIG")
    unavailable = [str(site["site_id"]) for site in train_sites if not _reachable(str(site["base_url"]))]
    if unavailable:
        raise RuntimeError(f"STEP7_TRAIN_SITE_UNAVAILABLE:{','.join(unavailable)}")
    results = [run_model(model, seed, train_sites) for model in ("history_mlp", "gru") for seed in SEEDS]
    manifest = {"run_id": f"STEP7_PPO_FRESH_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "observation": observation_v2_schema(), "action": action_space_metadata(), "action_hash": action_space_mapping_hash(), "reward_version": PPO_REWARD_SCHEMA_VERSION, "seeds": list(SEEDS), "episodes_per_seed": EPISODES, "max_steps": MAX_STEPS, "gt_runtime_access": False, "fresh_initialization": True, "legacy_warm_start": False, "split": {**split, "overlap": overlap}, "split_version": split_cfg.get("split_version"), "site_config": str(Path(args.site_config).resolve())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "data_split_manifest.json").write_text(json.dumps(manifest["split"], indent=2), encoding="utf-8")
    (OUT / "seed_manifest.json").write_text(json.dumps({"seeds": list(SEEDS)}, indent=2), encoding="utf-8")
    for name in ("history_mlp", "gru"):
        subset = [item for item in results if item["model"] == name]
        (OUT / f"{name}_raw_metrics.jsonl").write_text("".join(json.dumps(row, default=str) + "\n" for item in subset for row in item["rows"]), encoding="utf-8")
        (OUT / f"{name}_training_summary.json").write_text(json.dumps({k: v for k, v in subset[0].items() if k not in {"rows"}}, indent=2, default=str), encoding="utf-8")
    (OUT / "ppo_ablation_results.json").write_text(json.dumps({"results": [{k: v for k, v in item.items() if k not in {"rows", "metadata"}} for item in results], "selected_architecture": "UNSELECTED_VALIDATION_REQUIRED"}, indent=2, default=str), encoding="utf-8")
    (OUT / "ppo_validation_results.json").write_text(json.dumps({"results": [{"model": item["model"], "seed": item["seed"], "mean_reward": item["mean_reward"], "runtime_errors": item["runtime_errors"]} for item in results], "test_partition_opened": False}, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "results": [{k: v for k, v in item.items() if k not in {"rows", "metadata"}} for item in results]}, indent=2, default=str))


if __name__ == "__main__":
    main()
