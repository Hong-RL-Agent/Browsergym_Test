"""Diagnostic-only smoke for the Step 7 canonical 256/640 DQN contract.

This does not train, load, or write a production checkpoint.  It validates
the canonical observation/action shapes, mask handling, flattened global
selection, and finite Q output in a clean Python process.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from dqn_error_discovery.agents.dqn_agent import (  # noqa: E402
    FactorizedQNetwork,
    factorized_flat_q_values,
)
from dqn_error_discovery.hybrid.dqn_goal_context import canonical_dqn_observation_schema  # noqa: E402
from models.action_types import CANONICAL_ACTION_TYPES, action_space_mapping_hash  # noqa: E402


class CanonicalActionSpace:
    action_types = list(CANONICAL_ACTION_TYPES)
    max_candidates = 32

    def encode(self, action_type: str, candidate_index: int = 0) -> int:
        return self.action_types.index(action_type) * self.max_candidates + int(candidate_index)

    def get_action_dim(self) -> int:
        return len(self.action_types) * self.max_candidates


def main() -> None:
    schema = canonical_dqn_observation_schema()
    action_space = CanonicalActionSpace()
    obs_dim = int(schema["dimension"])
    action_dim = action_space.get_action_dim()
    model = FactorizedQNetwork(obs_dim, len(action_space.action_types), action_space.max_candidates)
    model.eval()
    state = np.zeros(obs_dim, dtype=np.float32)
    mask = np.zeros(action_dim, dtype=np.float32)
    valid_ids = [action_space.encode("click_element", 2), action_space.encode("fill_input", 4), action_space.encode("scroll_down", 0)]
    mask[valid_ids] = 1.0
    with torch.inference_mode():
        outputs = model(torch.from_numpy(state).unsqueeze(0))
        flat_q = factorized_flat_q_values(outputs, torch.from_numpy(mask).bool().unsqueeze(0), action_space).squeeze(0)
        mask_tensor = torch.from_numpy(mask).bool()
        masked_flat_q = flat_q.masked_fill(~mask_tensor, -torch.inf)
        selected = int(torch.argmax(masked_flat_q).item())
    payload = {
        "schema_version": "rawd_step7_dqn_contract_smoke_v1",
        "smoke_only": True,
        "production": False,
        "training": False,
        "observation_dimension": obs_dim,
        "observation_schema_hash": schema["schema_hash"],
        "base_observation_dimension": schema["base"]["schema_dim"],
        "goal_context_dimension": schema["goal"]["dimension"],
        "local_history_dimension": schema["history"]["dimension"],
        "action_dimension": action_dim,
        "action_mapping_hash": action_space_mapping_hash(),
        "model_architecture": "factorized_v1_confobs_histv2_diagnostic_256",
        "q_output_shape": list(flat_q.shape),
        "valid_action_count": int(mask.sum()),
        "selected_action": selected,
        "selected_action_is_valid": bool(mask[selected] > 0),
        "all_q_finite": bool(torch.isfinite(flat_q).all().item()),
        "global_masked_argmax": True,
        "optimizer_steps": 0,
        "legacy_checkpoint_loaded": False,
        "gt_leakage": 0,
        "pass": bool(obs_dim == 256 and action_dim == 640 and list(flat_q.shape) == [640] and mask[selected] > 0 and torch.isfinite(flat_q).all() and torch.isfinite(masked_flat_q[mask_tensor]).all()),
    }
    out = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/dqn_contract_smoke.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
