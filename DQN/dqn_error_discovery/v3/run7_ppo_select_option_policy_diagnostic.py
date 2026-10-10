"""Frozen PPO select_option policy/index diagnostic; no training or forced action."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.runners.train_dqn_error_detector import _runtime_compatibility_observation
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.action_types import CANONICAL_ACTION_TYPES

OUT = ROOT / "artifacts/training_v3/run7_ppo_select_option_policy_diagnostic"
CHECKPOINT = ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt"
V3_ACTION_TYPES = list(CANONICAL_ACTION_TYPES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def category(action_type: str) -> str:
    if action_type == "select_option":
        return "select_option"
    if action_type == "click_element":
        return "click"
    if action_type == "fill_input":
        return "fill_input"
    if action_type in {"scroll_down", "scroll_up"}:
        return "scroll"
    if action_type in {"change_viewport_mobile", "change_viewport_desktop"}:
        return "viewport"
    if action_type in {"noop"}:
        return "noop"
    if action_type in {"inspect_dom", "inspect_layout", "inspect_network", "inspect_console", "inspect_cart", "inspect_server_health", "inspect_port_status", "inspect_latency", "inspect_server_logs", "inspect_runtime_metrics"}:
        return "inspect"
    return "other_interactive"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    space = ActionSpace(32)
    ppo = PPOAgent(635, 640)
    ppo.load(CHECKPOINT)
    ppo.policy.eval()
    trace: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"available": 0, "mask_valid": 0, "selected": 0})
    competitors: dict[str, list[float]] = defaultdict(list)
    select_rows: list[dict[str, Any]] = []
    checkpoint_select_probabilities: list[float] = []
    checkpoint_select_ranks: list[int] = []
    errors: list[dict[str, Any]] = []
    episodes = 10
    global_steps = 0
    try:
        for episode in range(1, episodes + 1):
            env = BrowserGymJAWSEnv(site_id="SITE100", base_url="http://localhost:5200/", max_steps=4, max_candidates=32, headless=True)
            try:
                raw, _ = env.reset()
                for step in range(1, 5):
                    raw = _runtime_compatibility_observation(env, raw)
                    candidates = raw.get("candidate_elements", []) if isinstance(raw.get("candidate_elements"), list) else []
                    mask = space.build_action_mask(raw)
                    checkpoint_mask = np.zeros_like(mask)
                    for checkpoint_type in V3_ACTION_TYPES:
                        for slot in range(32):
                            current_index = space.encode(checkpoint_type, slot)
                            checkpoint_index = V3_ACTION_TYPES.index(checkpoint_type) * 32 + slot
                            checkpoint_mask[checkpoint_index] = mask[current_index]
                    obs = env.observation_encoder.encode_observation(raw).astype(np.float32)
                    with torch.no_grad():
                        logits, _ = ppo.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
                    # Explicit masked distribution: invalid actions receive zero probability.
                    safe_logits = logits.squeeze(0).masked_fill(~mask_tensor, -1e9)
                    masked_logits = safe_logits.detach().cpu().numpy()
                    probs = torch.softmax(safe_logits, dim=-1).detach().cpu().numpy()
                    checkpoint_safe_logits = logits.squeeze(0).masked_fill(~torch.as_tensor(checkpoint_mask, dtype=torch.bool), -1e9)
                    checkpoint_probs = torch.softmax(checkpoint_safe_logits, dim=-1).detach().cpu().numpy()
                    selected = ppo.select_action(obs, mask)
                    selected_id = int(selected["action_id"])
                    decoded = space.decode(selected_id)
                    valid_ids = [int(index) for index in np.flatnonzero(mask > 0)]
                    order = sorted(valid_ids, key=lambda index: float(probs[index]), reverse=True)
                    ranks = {action_id: rank for rank, action_id in enumerate(order, 1)}
                    select_slots = [index for index, candidate in enumerate(candidates[:32]) if isinstance(candidate, dict) and str(candidate.get("tag") or "").lower() == "select"]
                    select_ids = [space.encode("select_option", slot) for slot in select_slots if mask[space.encode("select_option", slot)] > 0]
                    checkpoint_valid_ids = [int(index) for index in np.flatnonzero(checkpoint_mask > 0)]
                    checkpoint_order = sorted(checkpoint_valid_ids, key=lambda index: float(checkpoint_probs[index]), reverse=True)
                    checkpoint_ranks = {action_id: rank for rank, action_id in enumerate(checkpoint_order, 1)}
                    for action_type in space.action_types:
                        available = int(bool(select_slots)) if action_type == "select_option" else 1
                        type_ids = [space.encode(action_type, slot) for slot in range(32)] if space.is_element_action(action_type) else [space.encode(action_type, 0)]
                        valid_type = int(any(mask[action_id] > 0 for action_id in type_ids))
                        coverage[action_type]["available"] += available
                        coverage[action_type]["mask_valid"] += valid_type
                        coverage[action_type]["selected"] += int(decoded["action_type"] == action_type)
                    for action_id in valid_ids:
                        if action_id in select_ids:
                            continue
                        competitors[category(space.decode(action_id)["action_type"])].append(float(probs[action_id]))
                    for action_id in select_ids:
                        decoded_select = space.decode(action_id)
                        row = {
                            "episode": episode,
                            "step": step,
                            "candidate_slot": decoded_select["candidate_index"],
                            "target_bid": candidates[decoded_select["candidate_index"]].get("bid"),
                            "target_tag": candidates[decoded_select["candidate_index"]].get("tag"),
                            "option_identity": None,
                            "option_selection_model": "executor_first_enabled_option_different_from_current",
                            "runtime_action_index": int(action_id),
                            "runtime_action_type_index": space.action_types.index("select_option"),
                            "checkpoint_action_type_index": V3_ACTION_TYPES.index("select_option"),
                            "raw_logit": float(logits.squeeze(0)[action_id].item()),
                            "masked_logit": float(masked_logits[action_id]),
                            "probability": float(probs[action_id]),
                            "rank_valid_actions": ranks.get(action_id),
                            "selected": bool(selected_id == action_id),
                            "mask_value": float(mask[action_id]),
                            "checkpoint_action_index": int(V3_ACTION_TYPES.index("select_option") * 32 + decoded_select["candidate_index"]),
                            "checkpoint_raw_logit": float(logits.squeeze(0)[V3_ACTION_TYPES.index("select_option") * 32 + decoded_select["candidate_index"]].item()),
                            "checkpoint_probability": float(checkpoint_probs[V3_ACTION_TYPES.index("select_option") * 32 + decoded_select["candidate_index"]]),
                            "checkpoint_rank_valid_actions": checkpoint_ranks.get(V3_ACTION_TYPES.index("select_option") * 32 + decoded_select["candidate_index"]),
                            "action_index_mapping_compatible": space.action_types.index("select_option") == V3_ACTION_TYPES.index("select_option"),
                        }
                        select_rows.append(row)
                        checkpoint_select_probabilities.append(row["checkpoint_probability"])
                        if row["checkpoint_rank_valid_actions"] is not None:
                            checkpoint_select_ranks.append(row["checkpoint_rank_valid_actions"])
                        for key in ("raw_logit", "masked_logit", "probability", "rank_valid_actions", "selected", "mask_value"):
                            pass
                    trace.append({
                        "episode": episode,
                        "step": step,
                        "select_option_exposed": bool(select_ids),
                        "select_option_valid_count": len(select_ids),
                        "selected_action": decoded,
                        "selected_action_id": selected_id,
                        "selected_probability": float(probs[selected_id]),
                        "selected_rank": ranks.get(selected_id),
                        "valid_action_count": len(valid_ids),
                    })
                    env.set_action_diagnostic_context({"episode_id": f"SITE100:episode:{episode}", "action_id": f"SITE100:episode:{episode}:step:{step}", "action_step": step})
                    raw, _, done, _ = env.step(selected_id)
                    global_steps += 1
                    if done:
                        break
            except Exception as exc:
                errors.append({"episode": episode, "error": repr(exc)})
            finally:
                env.close()
    except Exception as exc:
        errors.append({"fatal": repr(exc)})

    available_steps = sum(int(row["select_option_exposed"]) for row in trace)
    probabilities = [row["probability"] for row in select_rows]
    ranks = [row["rank_valid_actions"] for row in select_rows if row["rank_valid_actions"] is not None]
    summary = {
        "episodes": episodes,
        "global_steps": global_steps,
        "select_option_exposed_steps": available_steps,
        "mask_valid_select_actions": len(select_rows),
        "select_option_selections": sum(int(row["selected"]) for row in select_rows),
        "mean_probability": float(np.mean(probabilities)) if probabilities else None,
        "median_probability": float(np.median(probabilities)) if probabilities else None,
        "max_probability": float(np.max(probabilities)) if probabilities else None,
        "mean_valid_action_rank": float(np.mean(ranks)) if ranks else None,
        "best_rank": min(ranks) if ranks else None,
        "worst_rank": max(ranks) if ranks else None,
        "checkpoint_semantic_mean_probability": float(np.mean(checkpoint_select_probabilities)) if checkpoint_select_probabilities else None,
        "checkpoint_semantic_median_probability": float(np.median(checkpoint_select_probabilities)) if checkpoint_select_probabilities else None,
        "checkpoint_semantic_max_probability": float(np.max(checkpoint_select_probabilities)) if checkpoint_select_probabilities else None,
        "checkpoint_semantic_mean_valid_action_rank": float(np.mean(checkpoint_select_ranks)) if checkpoint_select_ranks else None,
        "checkpoint_semantic_best_rank": min(checkpoint_select_ranks) if checkpoint_select_ranks else None,
        "checkpoint_semantic_worst_rank": max(checkpoint_select_ranks) if checkpoint_select_ranks else None,
        "competitors_above_select_option": {key: sum(1 for value in values if probabilities and value > max(probabilities)) for key, values in competitors.items()},
        "competitor_mean_probability": {key: float(np.mean(values)) for key, values in competitors.items() if values},
        "competitor_max_probability": {key: float(np.max(values)) for key, values in competitors.items() if values},
        "selection_mode": "stochastic PPOAgent.select_action",
        "temperature": 1.0,
        "entropy_setting": "PPOAgent policy categorical entropy; no override",
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_obs_dim": 635,
        "checkpoint_action_dim": 640,
        "current_action_dim": space.get_action_dim(),
        "checkpoint_action_types": V3_ACTION_TYPES,
        "current_action_types": space.action_types,
        "checkpoint_select_option_type_index": V3_ACTION_TYPES.index("select_option"),
        "current_select_option_type_index": space.action_types.index("select_option"),
        "checkpoint_predates_select_option": False,
        "action_index_mapping_mismatch": V3_ACTION_TYPES != space.action_types,
        "gt_runtime_access": False,
        "ppo_retrained": False,
        "ppo_weights_modified": False,
        "errors": errors,
    }
    OUT.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT.joinpath("checkpoint_action_space.json").write_text(json.dumps({"checkpoint_sha256": summary["checkpoint_sha256"], "obs_dim": 635, "action_dim": 640, "action_types": V3_ACTION_TYPES, "select_option_index": 15, "training_manifest": str(ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_training_manifest.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT.joinpath("runtime_action_mapping.json").write_text(json.dumps({"current_action_dim": space.get_action_dim(), "current_action_types": space.action_types, "select_option_index": space.action_types.index("select_option"), "index_mismatch": summary["action_index_mapping_mismatch"], "round_trip": {"action_type": "select_option", "target_slot": 0, "encoded_index": space.encode("select_option", 0), "decoded": space.decode(space.encode("select_option", 0))}}, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT.joinpath("select_option_policy_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in select_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    OUT.joinpath("action_type_coverage.json").write_text(json.dumps({"coverage": dict(coverage), "selected_action_types": dict(Counter(row["selected_action"]["action_type"] for row in trace))}, ensure_ascii=False, indent=2), encoding="utf-8")
    root = "PPO_ACTION_INDEX_MAPPING_MISMATCH" if summary["action_index_mapping_mismatch"] else ("PPO_SELECT_OPTION_POLICY_PROBABILITY_COLLAPSE" if summary["max_probability"] is not None and summary["max_probability"] < 0.01 else "INSUFFICIENT_RUNTIME_SAMPLE")
    OUT.joinpath("root_cause.json").write_text(json.dumps({"primary_root_cause": root, "checkpoint_has_select_option": True, "checkpoint_predates_select_option": False, "action_index_mapping_mismatch": summary["action_index_mapping_mismatch"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
