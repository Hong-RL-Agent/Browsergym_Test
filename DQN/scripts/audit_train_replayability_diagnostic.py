from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from models.observation_v2 import CoverageMemory, ObservationV2Encoder  # noqa: E402
from models.recurrent_ppo import ObservationV2HistoryMLP, RecurrentPPOActorCritic  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402


BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def main() -> None:
    selected = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    sites_cfg = json.loads((ROOT / "configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8-sig"))["sites"]
    sites = {str(x["site_id"]): x for x in sites_cfg}
    # TRAIN-only diagnostic replay. Site001 is included because its historical
    # runtime log contains R01; no GT field is passed to the model or detector.
    site_ids = ["site001"] * 10 + ["site004", "site005", "site006", "site009"]
    recurrent = selected["architecture"] == "gru"
    payload = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    model = RecurrentPPOActorCritic(135, 640, 128) if recurrent else ObservationV2HistoryMLP(135, 640)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    enc = ObservationV2Encoder(32)
    traces = []
    rule_counts = Counter()
    action_counts = Counter()
    errors = []
    for episode_index, sid in enumerate(site_ids):
        torch.manual_seed(200000 + episode_index)
        site = sites[sid]
        env = BrowserGymJAWSEnv(site_id=sid, base_url=site["base_url"], max_steps=15, max_candidates=32, headless=True)
        raw, _ = env.reset()
        memory = CoverageMemory()
        hidden = model.initial_hidden(1) if recurrent else None
        prev_action, prev_outcome = {}, {}
        try:
            for step in range(15):
                vec = enc.encode(raw, previous_action=prev_action, previous_outcome=prev_outcome, coverage=memory)
                mask = env.action_space.build_action_mask(raw).astype(bool)
                with torch.inference_mode():
                    if recurrent:
                        logits, _, next_hidden = model(torch.from_numpy(vec).float().reshape(1, 1, -1), hidden, done_mask=torch.zeros(1, 1))
                        logits = logits[:, -1]
                    else:
                        logits, _ = model(torch.from_numpy(vec).float().reshape(1, -1))
                        next_hidden = None
                    aid = int(Categorical(logits=logits.masked_fill(~torch.as_tensor(mask).reshape(1, -1), -1e9)).sample().item())
                decoded = env.action_space.decode(aid)
                action_type = str(decoded.get("action_type") or "")
                action_counts[action_type] += 1
                slot = int(decoded.get("candidate_index", 0))
                before = raw
                candidates = before.get("candidate_elements", []) if isinstance(before, dict) else []
                target = ""
                if 0 <= slot < len(candidates) and isinstance(candidates[slot], dict):
                    bid = str(candidates[slot].get("bid") or "")
                    target = bid if bid.startswith("bid:") else (f"bid:{bid}" if bid else "")
                nxt, info, done, _ = env.step(aid)
                after = dict(nxt)
                after["runtime_signals"] = dict(info or {}) if isinstance(info, dict) else {}
                action = {"action_type": action_type, "candidate_index": slot, "action_id": aid, "execution_success": True, "target_identity": target}
                anomalies = detect_anomalies(before, after, {"action": action, "action_type": action_type, "trigger_action_type": action_type, "trigger_target_identity": target, "site_id": sid, "last_action_error": False}, site_profile={})
                rules = [str(a.get("rule_id") or "") for a in anomalies]
                for rule in rules:
                    rule_counts[rule] += 1
                traces.append({"site_id": sid, "episode_index": episode_index, "step": step, "action_type": action_type, "target": target, "rule_ids": rules, "anomaly_types": [str(a.get("type") or "") for a in anomalies]})
                raw = nxt
                if recurrent:
                    hidden = next_hidden
                prev_action = {"action_type": action_type, "candidate_slot": slot, "execution_success": True}
                prev_outcome = {"execution_success": True}
                if done:
                    break
        except Exception as exc:
            errors.append({"site_id": sid, "episode_index": episode_index, "error": repr(exc)})
        finally:
            env.close()
    out = {
        "schema_version": "rawd_step7_train_replayability_diagnostic_v1",
        "sites": site_ids,
        "episodes": len(site_ids),
        "steps": len(traces),
        "action_counts": dict(action_counts),
        "rule_counts": dict(rule_counts),
        "dqn_relevant_rules": {k: rule_counts.get(k, 0) for k in ("R01", "R02", "R09", "R10", "R12")},
        "traces": traces,
        "runtime_errors": errors,
        "gt_policy_input": False,
        "validation_access": False,
        "test_access": False,
        "ppo_optimizer_steps": 0,
        "ppo_checkpoint_sha": hashlib.sha256(Path(selected["checkpoint"]).read_bytes()).hexdigest().upper(),
        "classification": "CURRENT_DQN_RELEVANT_RUNTIME_EVIDENCE_OBSERVED" if any(rule_counts.get(k, 0) for k in ("R01", "R02", "R09", "R10", "R12")) else "NO_CURRENT_DQN_RELEVANT_RUNTIME_EVIDENCE_OBSERVED",
    }
    path = BASE / "dqn_train_replayability_diagnostic.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("sites", "episodes", "steps", "action_counts", "dqn_relevant_rules", "runtime_errors", "classification", "ppo_checkpoint_sha")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
