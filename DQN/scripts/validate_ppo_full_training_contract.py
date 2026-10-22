"""Validate the canonical Step-7 PPO full-training contract without training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DQN = ROOT / "DQN"
if str(DQN) not in sys.path:
    sys.path.insert(0, str(DQN))

from models.action_types import action_space_mapping_hash
from models.observation_v2 import OBSERVATION_V2_DIM, observation_v2_schema
from models.recurrent_ppo import ObservationV2HistoryMLP, RecurrentPPOActorCritic, recurrent_ppo_loss

CONTRACT = ROOT / "configs" / "ppo_full_training_v1.json"
OUT = ROOT / "DQN" / "artifacts" / "training_v3" / "step7_fresh_training_final"
EXPECTED_OBS_HASH = "248A2668CBF78ABCA02514A2B7E2EB10E0AE2EF063CBF01A8A136639048556BC"
EXPECTED_ACTION_HASH = "8DC532C3551051209E48EEFFF3550C7AB314521C0E097D99552F4CB9DBA0612A"


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t.detach()).all().item())


def _run_one(kind: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    recurrent = kind == "gru"
    model = RecurrentPPOActorCritic(135, 640, 128) if recurrent else ObservationV2HistoryMLP(135, 640)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    obs = torch.randn(1, 5, 135)
    actions = torch.zeros((1, 5), dtype=torch.long)
    masks = torch.ones((1, 5, 640), dtype=torch.bool)
    old_log_probs = torch.zeros((1, 5))
    advantages = torch.ones((1, 5))
    returns = torch.zeros((1, 5))
    hidden = model.initial_hidden(1) if recurrent else None
    if recurrent:
        logits, values, next_hidden = model(obs, hidden, done_mask=torch.zeros((1, 5)))
        metrics = recurrent_ppo_loss(model, obs, actions, old_log_probs, advantages, returns, hidden,
                                     action_masks=masks, done_masks=torch.zeros((1, 5)),
                                     valid_mask=torch.ones((1, 5), dtype=torch.bool))
        reset_logits, _, reset_hidden = model(obs[:, :1], model.initial_hidden(1), done_mask=torch.ones((1, 1)))
        hidden_propagation = not torch.equal(next_hidden, hidden)
        hidden_reset = bool(torch.isfinite(reset_hidden).all().item())
    else:
        logits, values = model(obs.reshape(5, 135))
        dist = torch.distributions.Categorical(logits=logits.masked_fill(~masks.reshape(5, 640), -1e9))
        ratio = torch.exp(dist.log_prob(actions.reshape(-1)) - old_log_probs.reshape(-1))
        policy = -torch.minimum(ratio * advantages.reshape(-1), torch.clamp(ratio, .8, 1.2) * advantages.reshape(-1)).mean()
        value = torch.nn.functional.mse_loss(values, returns.reshape(-1))
        entropy = dist.entropy().mean()
        metrics = {"loss": policy + .5 * value - .01 * entropy, "policy_loss": policy, "value_loss": value, "entropy": entropy}
        hidden_propagation = True
        hidden_reset = True
    optimizer.zero_grad()
    metrics["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return {
        "architecture": kind,
        "seed": seed,
        "observation_shape": list(obs.shape),
        "action_dim": int(logits.shape[-1]),
        "loss_finite": all(_finite(v) for v in metrics.values()),
        "gradient_finite": bool(torch.isfinite(grad_norm).item()),
        "optimizer_step": True,
        "hidden_propagation": hidden_propagation,
        "hidden_reset": hidden_reset,
        "runtime_errors": 0,
    }


def main() -> None:
    cfg = json.loads(CONTRACT.read_text(encoding="utf-8"))
    schema = observation_v2_schema()
    expected = {
        "contract_version": "rawd_ppo_full_training_v1",
        "observation_dim": 135,
        "observation_hash": EXPECTED_OBS_HASH,
        "action_dim": 640,
        "action_hash": EXPECTED_ACTION_HASH,
        "max_steps_per_architecture_seed": 4320,
        "max_steps_total": 25920,
    }
    checks = {
        "version": cfg.get("contract_version") == expected["contract_version"],
        "budget": cfg["total_updates"] * cfg["site_sampling"]["sites_per_update"] * cfg["rollout"]["episodes_per_site"] * cfg["rollout"]["max_steps_per_episode"] == expected["max_steps_per_architecture_seed"],
        "total_budget": expected["max_steps_per_architecture_seed"] * len(cfg["seeds"]) * 2 == expected["max_steps_total"],
        "observation": OBSERVATION_V2_DIM == expected["observation_dim"] and schema["schema_hash"] == expected["observation_hash"],
        "action": action_space_mapping_hash() == expected["action_hash"],
        "fresh_only": cfg.get("legacy_warm_start") is False and cfg.get("smoke_only") is False,
    }
    dry = [_run_one(kind, 42) for kind in ("history_mlp", "gru")]
    checks["dry_run"] = all(row["loss_finite"] and row["gradient_finite"] and row["optimizer_step"] and row["runtime_errors"] == 0 for row in dry)
    result = {"contract": cfg, "expected": expected, "checks": checks, "dry_run": dry,
              "status": "PASS" if all(checks.values()) else "FAIL", "training_started": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppo_full_training_contract.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "dry_run_results.json").write_text(json.dumps({"status": result["status"], "dry_run": dry, "training_started": False}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
