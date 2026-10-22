from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_types import action_space_mapping_hash
from models.observation_v2 import CoverageMemory, ObservationV2Encoder
from models.recurrent_ppo import RecurrentPPOActorCritic


BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def digest(value: object) -> str:
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest().upper()
    if torch.is_tensor(value):
        return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest().upper()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest().upper()


def trace(site: dict, model: RecurrentPPOActorCritic, steps: int = 5) -> list[dict]:
    env = BrowserGymJAWSEnv(site_id=site["site_id"], base_url=site["base_url"], max_steps=steps, max_candidates=32, headless=True)
    enc = ObservationV2Encoder(32)
    memory = CoverageMemory()
    hidden = model.initial_hidden(1)
    prev_a: dict = {}
    prev_o: dict = {}
    raw, _ = env.reset()
    rows = []
    try:
        for step in range(steps):
            vec = enc.encode(raw, previous_action=prev_a, previous_outcome=prev_o, coverage=memory)
            mask = env.action_space.build_action_mask(raw).astype(bool)
            obs = torch.from_numpy(vec).float()
            with torch.inference_mode():
                logits, _, next_hidden = model(obs.reshape(1, 1, -1), hidden, done_mask=torch.zeros((1, 1)))
            flat_logits = logits[:, -1]
            masked = flat_logits.masked_fill(~torch.as_tensor(mask).reshape(1, -1), -1e9)
            aid = int(torch.argmax(masked, dim=-1).item())
            decoded = env.action_space.decode(aid)
            rows.append({
                "step": step,
                "obs_hash": digest(vec),
                "obs_shape": list(vec.shape),
                "mask_hash": digest(mask.astype(np.uint8)),
                "mask_valid_count": int(mask.sum()),
                "logits_hash": digest(flat_logits),
                "masked_logits_hash": digest(masked),
                "hidden_before_hash": digest(hidden),
                "hidden_after_hash": digest(next_hidden),
                "hidden_before_norm": float(hidden.norm().item()),
                "hidden_after_norm": float(next_hidden.norm().item()),
                "selected_action": aid,
                "decoded": decoded,
                "top_masked": [
                    {"index": int(i), "value": float(masked[0, i].item()), "decoded": env.action_space.decode(int(i))}
                    for i in torch.topk(masked[0], k=min(5, masked.shape[-1])).indices.tolist()
                ],
            })
            nxt, _, done, _ = env.step(aid)
            prev_a = {"action_type": decoded["action_type"], "candidate_slot": int(decoded.get("candidate_index", 0)), "execution_success": True}
            prev_o = {"execution_success": True}
            raw = nxt
            hidden = next_hidden
            if done:
                break
    finally:
        env.close()
    return rows


def main() -> None:
    selected = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "configs/step7_full_site_split.json").read_text(encoding="utf-8"))
    cfg_text = (ROOT / "configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8-sig")
    cfg = json.loads(cfg_text)
    sites = {str(x["site_id"]): x for x in cfg["sites"]}
    checkpoint = Path(selected["checkpoint"])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_validation = RecurrentPPOActorCritic(135, 640, 128)
    model_validation.load_state_dict(payload["model_state_dict"])
    model_validation.eval()
    model_collector = RecurrentPPOActorCritic(135, 640, 128)
    model_collector.load_state_dict(payload["model_state_dict"])
    model_collector.eval()
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "production_sha_match": sha == selected["checkpoint_file_sha256"],
        "architecture": selected["architecture"],
        "seed": selected["seed"],
        "run_id": selected["run_id"],
        "mapping_hash": action_space_mapping_hash(),
        "sites": {},
        "model_mode": {"validation": model_validation.training, "collector": model_collector.training},
    }
    for sid in [str(split["validation"][0]), str(split["validation"][1]), str(split["validation"][2])]:
        validation_trace = trace(sites[sid], model_validation)
        collector_trace = trace(sites[sid], model_collector)
        pairs = []
        for left, right in zip(validation_trace, collector_trace):
            first = None
            for key in ("obs_hash", "mask_hash", "logits_hash", "masked_logits_hash", "hidden_before_hash", "hidden_after_hash", "selected_action"):
                if left.get(key) != right.get(key):
                    first = key
                    break
            pairs.append({"validation": left, "collector": right, "first_diff": first})
        result["sites"][sid] = {"validation_steps": len(validation_trace), "collector_steps": len(collector_trace), "pairs": pairs}
    result["all_same_input_inference"] = all(
        pair["first_diff"] is None
        for site in result["sites"].values()
        for pair in site["pairs"]
    )
    result["first_divergence"] = next(
        ({"site": sid, "step": pair["validation"]["step"], "component": pair["first_diff"], "validation": pair["validation"].get(pair["first_diff"]), "collector": pair["collector"].get(pair["first_diff"])}
         for sid, site in result["sites"].items() for pair in site["pairs"] if pair["first_diff"] is not None),
        None,
    )
    (BASE / "ppo_validation_vs_collector_diff_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"checkpoint_sha_match": result["production_sha_match"], "mapping_hash": result["mapping_hash"], "all_same_input_inference": result["all_same_input_inference"], "first_divergence": result["first_divergence"], "artifact": str(BASE / 'ppo_validation_vs_collector_diff_audit.json')}, indent=2))


if __name__ == "__main__":
    main()
