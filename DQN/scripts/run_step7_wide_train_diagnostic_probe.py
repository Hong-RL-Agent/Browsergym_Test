"""Wide TRAIN-only diagnostic probe for natural DQN-local opportunity.

This intentionally does not train DQN, mutate PPO, access GT, or promote any
row to the canonical training dataset. It broadens only the diagnostic
observation budget so reachability and detector scarcity can be separated.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from models.observation_v2 import CoverageMemory, ObservationV2Encoder  # noqa: E402
from models.recurrent_ppo import ObservationV2HistoryMLP, RecurrentPPOActorCritic  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402


BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
OUT = BASE / "dqn_wide_train_diagnostic_probe.json"


def main() -> None:
    selected = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "configs/step7_full_site_split.json").read_text(encoding="utf-8"))
    site_cfg = json.loads((ROOT / "configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8-sig"))
    sites = {str(row["site_id"]): row for row in site_cfg["sites"]}
    train_ids = [str(x) for x in split["train"]]
    assert len(train_ids) == 36

    checkpoint = Path(selected["checkpoint"])
    sha_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recurrent = selected.get("architecture") == "gru"
    model = RecurrentPPOActorCritic(135, 640, 128) if recurrent else ObservationV2HistoryMLP(135, 640)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    encoder = ObservationV2Encoder(32)

    action_counts = Counter()
    rule_counts = Counter()
    strategy_counts = Counter()
    site_rows = []
    errors = []
    total_steps = 0
    total_interactions = 0
    total_candidates = 0
    total_relevant_candidates = 0

    for site_index, site_id in enumerate(train_ids):
        site = sites[site_id]
        row = {"site_id": site_id, "steps": 0, "interaction_actions": 0, "candidates": 0, "relevant_candidates": 0, "rules": {}, "strategies": {}}
        env = BrowserGymJAWSEnv(site_id=site_id, base_url=site["base_url"], max_steps=30, max_candidates=32, headless=True)
        raw, _ = env.reset()
        memory = CoverageMemory()
        hidden = model.initial_hidden(1) if recurrent else None
        previous_action, previous_outcome = {}, {}
        try:
            for step in range(30):
                vector = encoder.encode(raw, previous_action=previous_action, previous_outcome=previous_outcome, coverage=memory)
                mask = env.action_space.build_action_mask(raw).astype(bool)
                with torch.inference_mode():
                    if recurrent:
                        logits, _, next_hidden = model(torch.from_numpy(vector).float().reshape(1, 1, -1), hidden, done_mask=torch.zeros(1, 1))
                        logits = logits[:, -1]
                    else:
                        logits, _ = model(torch.from_numpy(vector).float().reshape(1, -1))
                        next_hidden = None
                    action_id = int(Categorical(logits=logits.masked_fill(~torch.as_tensor(mask).reshape(1, -1), -1e9)).sample().item())
                decoded = env.action_space.decode(action_id)
                action_type = str(decoded.get("action_type") or "")
                slot = int(decoded.get("candidate_index", 0))
                action_counts[action_type] += 1
                if action_type in {"click_element", "fill_input", "select_option", "press_enter"}:
                    total_interactions += 1
                    row["interaction_actions"] += 1
                before = raw
                candidates = before.get("candidate_elements", []) if isinstance(before, dict) else []
                row["candidates"] += len(candidates) if isinstance(candidates, list) else 0
                total_candidates += len(candidates) if isinstance(candidates, list) else 0
                target = ""
                if 0 <= slot < len(candidates) and isinstance(candidates[slot], dict):
                    bid = str(candidates[slot].get("bid") or "")
                    target = bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
                next_raw, info, done, _ = env.step(action_id)
                after = dict(next_raw)
                after["runtime_signals"] = dict(info or {}) if isinstance(info, dict) else {}
                action = {"action_type": action_type, "candidate_index": slot, "action_id": action_id, "execution_success": True, "target_identity": target}
                anomalies = detect_anomalies(before, after, {"action": action, "action_type": action_type, "trigger_action_type": action_type, "trigger_target_identity": target, "site_id": site_id, "last_action_error": False}, site_profile={})
                for anomaly in anomalies:
                    rule = str(anomaly.get("rule_id") or "")
                    strategy = str(anomaly.get("reproduction_strategy") or "")
                    rule_counts[rule] += 1
                    strategy_counts[strategy or "UNSPECIFIED"] += 1
                    row["rules"][rule] = row["rules"].get(rule, 0) + 1
                    row["strategies"][strategy or "UNSPECIFIED"] = row["strategies"].get(strategy or "UNSPECIFIED", 0) + 1
                    if rule in {"R01", "R02", "R09", "R10", "R12"}:
                        total_relevant_candidates += 1
                        row["relevant_candidates"] += 1
                row["steps"] += 1
                total_steps += 1
                raw = next_raw
                if recurrent:
                    hidden = next_hidden
                previous_action = {"action_type": action_type, "candidate_slot": slot, "execution_success": True}
                previous_outcome = {"execution_success": True}
                if done:
                    break
        except Exception as exc:
            errors.append({"site_id": site_id, "error": repr(exc)})
        finally:
            env.close()
        site_rows.append(row)

    sha_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    out = {
        "schema_version": "rawd_step7_wide_train_diagnostic_probe_v1",
        "diagnostic_only": True,
        "canonical_dataset_promotion": False,
        "train_sites": train_ids,
        "episodes": len(train_ids),
        "max_steps_per_episode": 30,
        "ppo_run_id": selected["run_id"],
        "ppo_checkpoint_sha_before": sha_before,
        "ppo_checkpoint_sha_after": sha_after,
        "ppo_sha_unchanged": sha_before == sha_after,
        "optimizer_steps": 0,
        "gt_policy_input": False,
        "validation_access": False,
        "test_access": False,
        "total_steps": total_steps,
        "interaction_actions": total_interactions,
        "candidate_elements_observed": total_candidates,
        "relevant_dqn_rule_candidates": total_relevant_candidates,
        "action_counts": dict(action_counts),
        "rule_counts": dict(rule_counts),
        "strategy_counts": dict(strategy_counts),
        "runtime_errors": errors,
        "site_rows": site_rows,
        "dqn_local_chains": 0,
        "multi_action_states": 0,
        "runtime_partial": 0,
        "runtime_successful": 0,
        "classification": "NO_CURRENT_DQN_RELEVANT_RUNTIME_EVIDENCE_OBSERVED" if total_relevant_candidates == 0 else "DQN_RELEVANT_CANDIDATES_OBSERVED_REQUIRES_FUNNEL_AUDIT",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: out[key] for key in ("episodes", "total_steps", "interaction_actions", "candidate_elements_observed", "relevant_dqn_rule_candidates", "rule_counts", "strategy_counts", "runtime_errors", "dqn_local_chains", "classification", "ppo_sha_unchanged")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
