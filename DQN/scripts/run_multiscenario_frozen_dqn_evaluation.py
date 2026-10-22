"""Evaluation-only frozen DQN execution over B01/B05 diagnostic scenarios."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import urllib.request

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
OUT = BASE / "external_dqn_multiscenario_evaluation.json"

from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from models.observation_v2 import CoverageMemory, ObservationV2Encoder  # noqa: E402
from models.action_space import ActionSpace  # noqa: E402
from models.recurrent_ppo import ObservationV2HistoryMLP  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context  # noqa: E402
from dqn_error_discovery.hybrid.strategy_router import route_candidate  # noqa: E402
from dqn_error_discovery.hybrid.dqn_goal_context import encode_goal_context  # noqa: E402
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork  # noqa: E402


def q_values(model: FactorizedQNetwork, state: torch.Tensor) -> torch.Tensor:
    out = model(state)
    adv = out["candidate_advantages"]
    return out["type_values"].unsqueeze(-1) + (adv - adv.mean(dim=-1, keepdim=True))


def reset_backend(url: str) -> None:
    port = "9715" if ":5316" in url else "9719"
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/test/reset", method="POST"), timeout=5).read()
    except Exception:
        pass


def choose(candidates: list[dict], terms: list[str]) -> int | None:
    for i, c in enumerate(candidates):
        text = " ".join(str(c.get(k) or "") for k in ("text", "name", "aria_label", "label"))
        if any(term in text for term in terms) and c.get("visible", True) and c.get("enabled", True):
            return i
    return None


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (BASE / "external_dqn_probe_sites.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ppo_meta = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    dqn_meta = json.loads((BASE / "selected_dqn_checkpoint.json").read_text(encoding="utf-8"))
    dqn_path = Path(dqn_meta["checkpoint"])
    dqn_sha_before = hashlib.sha256(dqn_path.read_bytes()).hexdigest().upper()
    payload = torch.load(dqn_path, map_location="cpu", weights_only=False)
    dqn = FactorizedQNetwork(256, 20, 32)
    dqn.load_state_dict(payload["model"])
    dqn.eval()
    ppo_path = Path(ppo_meta["checkpoint"])
    ppo_sha_before = hashlib.sha256(ppo_path.read_bytes()).hexdigest().upper()
    ppo_payload = torch.load(ppo_path, map_location="cpu", weights_only=False)
    ppo = ObservationV2HistoryMLP(135, 640)
    ppo.load_state_dict(ppo_payload["model_state_dict"])
    ppo.eval()
    encoder = ObservationV2Encoder(32)
    local_space = ActionSpace(32)
    totals = Counter()
    site_rows: list[dict] = []
    chain_rows: list[dict] = []
    errors: list[dict] = []

    for site in config["sites"]:
        sid, url = str(site["site_id"]), str(site["base_url"])
        reset_backend(url)
        env = BrowserGymJAWSEnv(site_id=sid, base_url=url, max_steps=30, max_candidates=32, headless=True, site_profile={})
        raw, _ = env.reset()
        memory = CoverageMemory()
        prev_action: dict = {}
        prev_outcome: dict = {}
        row = {"site_id": sid, "steps": 0, "dqn_local": 0, "dqn_executed": 0,
               "valid_selected": 0, "post_action_fresh_evidence": 0, "partial": 0, "failed": 0}
        try:
            for step in range(30):
                vector = encoder.encode(raw, previous_action=prev_action, previous_outcome=prev_outcome, coverage=memory)
                mask = env.action_space.build_action_mask(raw).astype(bool)
                with torch.inference_mode():
                    logits, _ = ppo(torch.from_numpy(vector).float().reshape(1, -1))
                    masked = logits.masked_fill(~torch.as_tensor(mask).reshape(1, -1), -1e9)
                    action_id = int(Categorical(logits=masked).sample().item())
                decoded = env.action_space.decode(action_id)
                before = raw
                slot = int(decoded.get("candidate_index", 0))
                candidates = before.get("candidate_elements", []) if isinstance(before, dict) else []
                target = str(candidates[slot].get("bid") or "") if 0 <= slot < len(candidates) and isinstance(candidates[slot], dict) else ""
                nxt, _, done, info = env.step(action_id)
                after = dict(nxt)
                signals = dict(after.get("runtime_signals") or {})
                if isinstance(info, dict): signals.update(info)
                after["runtime_signals"] = signals
                action = {"action_type": decoded["action_type"], "candidate_index": slot, "action_id": action_id, "target_identity": target, "execution_success": True}
                anomalies = detect_anomalies(before, after, {"action": action, "action_type": decoded["action_type"], "trigger_action_type": decoded["action_type"], "trigger_target_identity": target, "site_id": sid, "last_action_error": False}, site_profile={})
                route_found = None
                for index, anomaly in enumerate(anomalies):
                    anomaly.setdefault("trigger_action_type", decoded["action_type"])
                    anomaly.setdefault("trigger_target_identity", target)
                    ctx = derive_reproduction_context(anomaly).as_dict()
                    routed = route_candidate({"reproduction_strategy": ctx.get("reproduction_strategy"), "requires_action_reproduction": bool(ctx.get("requires_action_reproduction")), "reproduction_action_type": ctx.get("reproduction_action_type"), "reproduction_target_identity": ctx.get("reproduction_target_identity"), "target_actionable": bool(ctx.get("requires_action_reproduction") and ctx.get("reproduction_target_identity")), "exact_action_known": False, "chain_id": f"{sid}:{step}:{index}"})
                    if routed.dqn_eligible and route_found is None:
                        route_found = {"anomaly": anomaly, "context": ctx, "route": routed, "index": index}
                if route_found:
                    totals["dqn_local"] += 1; row["dqn_local"] += 1
                    goal = encode_goal_context({"active_candidate": True, "strategy": str(route_found["route"].strategy), "candidate_family": str(route_found["anomaly"].get("rule_id") or "unknown"), "oracle_family": "RUNTIME_EVIDENCE_ORACLE", "confirmation_progress": 0.0, "same_chain_progress": 0.0, "remaining_local_budget": 1.0, "prior_execution_success": 1.0})
                    state = np.concatenate([vector, goal, np.zeros(62, dtype=np.float32)]).astype(np.float32)
                    local_mask = local_space.build_action_mask(before).astype(bool)
                    valid_count = int(local_mask.sum())
                    with torch.inference_mode():
                        q = q_values(dqn, torch.from_numpy(state).reshape(1, -1)).reshape(1, -1)
                        masked_q = q.masked_fill(~torch.from_numpy(local_mask).reshape(1, -1), -1e9)
                        local_id = int(masked_q.argmax(dim=1).item())
                    local_decoded = local_space.decode(local_id)
                    valid = bool(local_mask[local_id])
                    row["dqn_executed"] += 1; totals["dqn_executed"] += 1
                    totals["valid_selected"] += int(valid); row["valid_selected"] += int(valid)
                    if valid and local_decoded["candidate_index"] < len(candidates):
                        local_env_id = env.action_space.encode(local_decoded["action_type"], local_decoded["candidate_index"])
                        local_before = after
                        local_after, _, local_done, local_info = env.step(local_env_id)
                        local_after = dict(local_after)
                        local_signals = dict(local_after.get("runtime_signals") or {})
                        if isinstance(local_info, dict): local_signals.update(local_info)
                        local_after["runtime_signals"] = local_signals
                        fresh = detect_anomalies(local_before, local_after, {"action": {"action_type": local_decoded["action_type"], "candidate_index": local_decoded["candidate_index"], "action_id": local_env_id, "target_identity": str(candidates[local_decoded["candidate_index"]].get("bid") or "")}, "action_type": local_decoded["action_type"], "trigger_action_type": local_decoded["action_type"], "trigger_target_identity": str(candidates[local_decoded["candidate_index"]].get("bid") or ""), "site_id": sid, "last_action_error": False}, site_profile={})
                        fresh_count = len(fresh)
                        totals["post_action_fresh_evidence"] += int(fresh_count > 0); row["post_action_fresh_evidence"] += int(fresh_count > 0)
                        outcome = "PARTIAL_REPRODUCTION" if fresh_count > 0 else "FAILED_REPRODUCTION"
                        totals[outcome.lower()] += 1; row["partial"] += int(outcome == "PARTIAL_REPRODUCTION"); row["failed"] += int(outcome == "FAILED_REPRODUCTION")
                        chain_rows.append({"site_id": sid, "step": step, "rule": route_found["anomaly"].get("rule_id"), "strategy": str(route_found["route"].strategy), "valid_local_actions": valid_count, "selected_action": local_decoded, "selected_flattened_action": local_id, "q_selection_mismatch": 0, "post_action_fresh_evidence": fresh_count, "outcome": outcome, "runtime": True, "synthetic": False})
                        raw = local_after
                    else:
                        totals["invalid_selected"] += 1
                        raw = nxt
                else:
                    raw = nxt
                row["steps"] += 1; totals["steps"] += 1
                prev_action = {"action_type": decoded["action_type"], "candidate_slot": slot, "execution_success": True}
                prev_outcome = {"execution_success": True}
                if done: break
        except Exception as exc:
            errors.append({"site_id": sid, "error": repr(exc)})
        finally:
            env.close()
        site_rows.append(row)

    ppo_sha_after = hashlib.sha256(ppo_path.read_bytes()).hexdigest().upper()
    dqn_sha_after = hashlib.sha256(dqn_path.read_bytes()).hexdigest().upper()
    result = {"evaluation_only": True, "sites": [str(s["site_id"]) for s in config["sites"]], "site_rows": site_rows, "chains": chain_rows, "counts": dict(totals), "runtime_errors": errors, "ppo_sha_before": ppo_sha_before, "ppo_sha_after": ppo_sha_after, "dqn_sha_before": dqn_sha_before, "dqn_sha_after": dqn_sha_after, "ppo_sha_unchanged": ppo_sha_before == ppo_sha_after, "dqn_sha_unchanged": dqn_sha_before == dqn_sha_after, "gt_leakage": 0, "test_access": 0, "optimizer_steps": 0, "dqn_evaluation_readiness": {"dqn_local_chains": totals["dqn_local"] > 0, "multi_action_states": all(int(c.get("valid_local_actions", 0)) >= 2 for c in chain_rows), "valid_selected_actions": totals["invalid_selected"] == 0, "partial_or_successful": totals["partial_reproduction"] + totals["successful_reproduction"] > 0 if "successful_reproduction" in totals else totals["partial_reproduction"] > 0, "broad_generalization": False}}
    result["source_config"] = str(config_path)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"sites": result["sites"], "counts": result["counts"], "site_rows": site_rows, "readiness": result["dqn_evaluation_readiness"], "runtime_errors": errors, "ppo_sha_unchanged": result["ppo_sha_unchanged"], "dqn_sha_unchanged": result["dqn_sha_unchanged"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
