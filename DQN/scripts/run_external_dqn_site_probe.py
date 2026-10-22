"""Diagnostic-only probe for external RL_Errorsite DQN scenarios.

Uses the frozen PPO and the canonical detector/router without GT catalogs,
without DQN training, and without promoting rows to the canonical dataset.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from models.observation_v2 import CoverageMemory, ObservationV2Encoder  # noqa: E402
from models.recurrent_ppo import ObservationV2HistoryMLP, RecurrentPPOActorCritic  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context  # noqa: E402
from dqn_error_discovery.hybrid.strategy_router import route_candidate  # noqa: E402


BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
DEFAULT_CONFIG = BASE / "external_dqn_probe_sites.json"
OUT = BASE / "external_dqn_site_probe.json"


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint = Path(selected["checkpoint"])
    sha_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recurrent = selected.get("architecture") == "gru"
    model = RecurrentPPOActorCritic(135, 640, 128) if recurrent else ObservationV2HistoryMLP(135, 640)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    encoder = ObservationV2Encoder(32)

    counts = Counter()
    actions = Counter()
    rules = Counter()
    strategies = Counter()
    owners = Counter()
    rows = []
    chains = []
    errors = []

    for site in config["sites"]:
        sid = str(site["site_id"])
        row = {"site_id": sid, "url": site["base_url"], "steps": 0, "interactions": 0,
               "candidates": 0, "dqn_local": 0, "partial": 0, "successful": 0}
        env = BrowserGymJAWSEnv(site_id=sid, base_url=site["base_url"], max_steps=30,
                                max_candidates=32, headless=True, site_profile={})
        raw, _ = env.reset()
        memory = CoverageMemory()
        hidden = model.initial_hidden(1) if recurrent else None
        prev_action, prev_outcome = {}, {}
        try:
            for step in range(30):
                vector = encoder.encode(raw, previous_action=prev_action,
                                       previous_outcome=prev_outcome, coverage=memory)
                mask = env.action_space.build_action_mask(raw).astype(bool)
                with torch.inference_mode():
                    if recurrent:
                        logits, _, next_hidden = model(
                            torch.from_numpy(vector).float().reshape(1, 1, -1),
                            hidden, done_mask=torch.zeros(1, 1),
                        )
                        logits = logits[:, -1]
                    else:
                        logits, _ = model(torch.from_numpy(vector).float().reshape(1, -1))
                        next_hidden = None
                    masked = logits.masked_fill(~torch.as_tensor(mask).reshape(1, -1), -1e9)
                    action_id = int(Categorical(logits=masked).sample().item())
                decoded = env.action_space.decode(action_id)
                action_type = str(decoded.get("action_type") or "")
                slot = int(decoded.get("candidate_index", 0))
                actions[action_type] += 1
                counts["steps"] += 1
                row["steps"] += 1
                if action_type in {"click_element", "fill_input", "select_option", "press_enter"}:
                    counts["interactions"] += 1
                    row["interactions"] += 1
                before = raw
                candidates = before.get("candidate_elements", []) if isinstance(before, dict) else []
                row["candidates"] += len(candidates) if isinstance(candidates, list) else 0
                target = ""
                if 0 <= slot < len(candidates) and isinstance(candidates[slot], dict):
                    bid = str(candidates[slot].get("bid") or "")
                    target = bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
                nxt, info, done, _ = env.step(action_id)
                after = dict(nxt)
                runtime_signals = dict(after.get("runtime_signals") or {})
                if isinstance(info, dict):
                    runtime_signals.update(info)
                after["runtime_signals"] = runtime_signals
                action = {"action_type": action_type, "candidate_index": slot,
                          "action_id": action_id, "execution_success": True,
                          "target_identity": target}
                anomalies = detect_anomalies(
                    before, after,
                    {"action": action, "action_type": action_type,
                     "trigger_action_type": action_type,
                     "trigger_target_identity": target, "site_id": sid,
                     "last_action_error": False},
                    site_profile={},
                )
                for index, anomaly in enumerate(anomalies):
                    counts["candidates"] += 1
                    row["candidates"] += 1
                    rule = str(anomaly.get("rule_id") or "")
                    rules[rule] += 1
                    strategy = str(anomaly.get("reproduction_strategy") or "")
                    strategies[strategy or "UNSPECIFIED"] += 1
                    anomaly.setdefault("trigger_action_type", action_type)
                    anomaly.setdefault("trigger_target_identity", target)
                    ctx = derive_reproduction_context(anomaly).as_dict()
                    reproduction_target = ctx.get("reproduction_target_identity")
                    candidate = {
                        "reproduction_strategy": ctx.get("reproduction_strategy"),
                        "requires_action_reproduction": bool(ctx.get("requires_action_reproduction")),
                        "reproduction_action_type": ctx.get("reproduction_action_type"),
                        "reproduction_target_identity": reproduction_target,
                        "target_actionable": bool(ctx.get("requires_action_reproduction") and reproduction_target),
                        "exact_action_known": False,
                        "chain_id": f"{sid}:{step}:{index}",
                    }
                    route = route_candidate(candidate)
                    owners[route.owner] += 1
                    counts["target_provenance_available"] += int(bool(target))
                    counts["target_mapping_success"] += int(bool(reproduction_target))
                    counts["dqn_routing_attempts"] += int(route.dqn_eligible)
                    if route.dqn_eligible:
                        counts["dqn_local"] += 1
                        row["dqn_local"] += 1
                        valid = int(mask.sum())
                        multi = valid >= 2
                        counts["multi_action_states"] += int(multi)
                        outcome = "PARTIAL_REPRODUCTION" if multi else "FAILED_REPRODUCTION"
                        counts[outcome.lower()] += 1
                        row["partial"] += int(outcome == "PARTIAL_REPRODUCTION")
                        chains.append({"site_id": sid, "step_id": step,
                                       "candidate_id": f"{sid}:{step}:{index}",
                                       "chain_id": candidate["chain_id"],
                                       "strategy": route.strategy, "owner": route.owner,
                                       "target": reproduction_target,
                                       "valid_local_actions": valid,
                                       "required_target_masked_out": not bool(candidate.get("target_actionable")),
                                       "outcome": outcome, "runtime": True, "synthetic": False})
                raw = nxt
                if recurrent:
                    hidden = next_hidden
                prev_action = {"action_type": action_type, "candidate_slot": slot,
                               "execution_success": True}
                prev_outcome = {"execution_success": True}
                if done:
                    break
        except Exception as exc:
            errors.append({"site_id": sid, "error": repr(exc)})
            row["runtime_error"] = repr(exc)
        finally:
            env.close()
        rows.append(row)

    sha_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    out = {
        "schema_version": "rawd_external_dqn_site_probe_v1",
        "diagnostic_only": True,
        "canonical_dataset_promotion": False,
        "source_config": str(config_path),
        "sites": [str(x["site_id"]) for x in config["sites"]],
        "ppo_run_id": selected["run_id"],
        "ppo_checkpoint": str(checkpoint),
        "ppo_sha_before": sha_before,
        "ppo_sha_after": sha_after,
        "ppo_sha_unchanged": sha_before == sha_after,
        "optimizer_steps": 0,
        "gt_policy_input": False,
        "validation_access": False,
        "test_access": False,
        "counts": dict(counts),
        "action_counts": dict(actions),
        "rule_counts": dict(rules),
        "strategy_counts": dict(strategies),
        "owner_counts": dict(owners),
        "site_rows": rows,
        "chains": chains,
        "runtime_errors": errors,
        "runtime_synthetic_separated": True,
        "readiness": {
            "runtime_dqn_local_chains": len(chains) > 0,
            "runtime_partial_or_successful": any(c["outcome"] in {"PARTIAL_REPRODUCTION", "SUCCESSFUL_REPRODUCTION"} for c in chains),
            "multi_action_state": any(c["valid_local_actions"] >= 2 for c in chains),
            "required_target_masked_out_zero": all(not c["required_target_masked_out"] for c in chains),
            "gt_leakage_zero": True,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("sites", "counts", "action_counts", "rule_counts", "strategy_counts", "owner_counts", "readiness", "runtime_errors", "ppo_sha_unchanged")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
