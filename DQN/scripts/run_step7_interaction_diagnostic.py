from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.observation_v2 import CoverageMemory, ObservationV2Encoder
from models.recurrent_ppo import RecurrentPPOActorCritic, ObservationV2HistoryMLP
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward

BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def sha(value: object) -> str:
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest().upper()
    if torch.is_tensor(value):
        return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest().upper()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest().upper()


def viewport(raw: dict) -> dict:
    candidates = [raw.get("viewport"), raw.get("viewport_size"), raw.get("screen")]
    for item in candidates:
        if isinstance(item, dict):
            width = item.get("width") or item.get("innerWidth")
            height = item.get("height") or item.get("innerHeight")
            if width is not None or height is not None:
                return {"width": width, "height": height}
    for wk, hk in (("viewport_width", "viewport_height"), ("screen_width", "screen_height"), ("inner_width", "inner_height")):
        if wk in raw or hk in raw:
            return {"width": raw.get(wk), "height": raw.get(hk)}
    return {"width": None, "height": None}


def candidate_stats(raw: dict) -> dict:
    candidates = raw.get("candidate_elements", []) if isinstance(raw, dict) else []
    candidates = candidates if isinstance(candidates, list) else []
    click = fill = select = 0
    for c in candidates:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role") or "").lower()
        tag = str(c.get("tag") or c.get("tag_name") or "").lower()
        visible = c.get("visible", c.get("visibility", 1)) not in (False, 0, "hidden")
        enabled = c.get("enabled", True) not in (False, 0)
        clickable = c.get("clickable", role in {"button", "link", "menuitem", "tab", "checkbox", "radio"})
        bbox = c.get("bbox") or []
        bbox_ok = not isinstance(bbox, list) or len(bbox) < 4 or (float(bbox[2]) > 0 and float(bbox[3]) > 0)
        click += int(bool(visible and enabled and clickable and bbox_ok))
        fill += int(bool(role in {"textbox", "searchbox", "combobox"} or tag in {"input", "textarea"}))
        select += int(bool(role == "combobox" and tag == "select"))
    return {"raw_candidates": len(candidates), "raw_clickable": click, "raw_fill": fill, "raw_select": select}


def run_site(site: dict, model, recurrent: bool, steps: int = 15) -> list[dict]:
    env = BrowserGymJAWSEnv(site_id=site["site_id"], base_url=site["base_url"], max_steps=steps, max_candidates=32, headless=True)
    enc = ObservationV2Encoder(32)
    memory = CoverageMemory()
    hidden = model.initial_hidden(1) if recurrent else None
    prev_a: dict = {}
    prev_o: dict = {}
    raw, _ = env.reset()
    out = []
    try:
        for step in range(steps):
            vec = enc.encode(raw, previous_action=prev_a, previous_outcome=prev_o, coverage=memory)
            mask = env.action_space.build_action_mask(raw).astype(bool)
            obs = torch.from_numpy(vec).float()
            with torch.inference_mode():
                if recurrent:
                    logits, _, next_hidden = model(obs.reshape(1, 1, -1), hidden, done_mask=torch.zeros((1, 1)))
                    logits = logits[:, -1]
                else:
                    logits, _ = model(obs)
                    next_hidden = None
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
            mask_t = torch.as_tensor(mask).reshape(1, -1)
            masked = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(masked, dim=-1)[0]
            aid = int(Categorical(logits=masked).sample().item())
            decoded = env.action_space.decode(aid)
            type_indices = {}
            for idx in range(len(mask)):
                d = env.action_space.decode(idx)
                type_indices.setdefault(d["action_type"], []).append(idx)
            type_rows = {}
            for action_type, indices in type_indices.items():
                valid = [i for i in indices if bool(mask[i])]
                best = max(valid, key=lambda i: float(masked[0, i])) if valid else None
                rank = None
                if best is not None:
                    rank = 1 + sum(float(masked[0, j]) > float(masked[0, best]) for j in range(len(mask)) if bool(mask[j]))
                type_rows[action_type] = {"valid_count": len(valid), "best_index": best, "best_logit": float(logits[0, best]) if best is not None else None, "best_probability": float(probs[best]) if best is not None else 0.0, "best_rank": int(rank) if rank is not None else None, "probability_mass": float(probs[valid].sum()) if valid else 0.0}
            before = raw
            before_vp = viewport(before)
            stats = candidate_stats(before)
            nxt, info, done, _ = env.step(aid)
            after = dict(nxt)
            after["runtime_signals"] = dict(info or {}) if isinstance(info, dict) else {}
            after_vp = viewport(after)
            action = {"action_type": decoded["action_type"], "candidate_index": int(decoded.get("candidate_index", 0)), "action_id": aid, "execution_success": True}
            try:
                reward, components = calculate_autonomous_reward(before, after, action, [], [], {}, {})
            except Exception as exc:
                reward, components = None, {"reward_calc_error": repr(exc)}
            anomalies = detect_anomalies(before, after, {"action": action, "action_type": decoded["action_type"], "site_id": site["site_id"], "last_action_error": False}, site_profile={})
            out.append({
                "site_id": site["site_id"], "step": step, "url": before.get("url"), "viewport_before": before_vp, "viewport_after": after_vp,
                "candidate_stats": stats, "mask_hash": sha(mask.astype(np.uint8)), "valid_flattened_count": int(mask.sum()), "type_rows": type_rows,
                "selected_action": aid, "selected_action_type": decoded["action_type"], "selected_slot": int(decoded.get("candidate_index", 0)),
                "logits_hash": sha(logits), "masked_logits_hash": sha(masked), "probabilities_hash": sha(probs),
                "hidden_before_norm": float(hidden.norm().item()) if recurrent else None, "hidden_after_norm": float(next_hidden.norm().item()) if recurrent else None,
                "reward": float(reward) if reward is not None else None, "reward_components": components, "anomaly_types": [str(a.get("type") or "") for a in anomalies],
                "done": bool(done),
            })
            prev_a = {"action_type": decoded["action_type"], "candidate_slot": int(decoded.get("candidate_index", 0)), "execution_success": True}
            prev_o = {"execution_success": True}
            raw = nxt
            if recurrent:
                hidden = next_hidden
            if done:
                break
    finally:
        env.close()
    return out


def main() -> None:
    selected = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8-sig"))
    sites = {str(x["site_id"]): x for x in cfg["sites"]}
    split = json.loads((ROOT / "configs/step7_full_site_split.json").read_text(encoding="utf-8"))
    ck = Path(selected["checkpoint"])
    before_sha = hashlib.sha256(ck.read_bytes()).hexdigest().upper()
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    recurrent = selected["architecture"] == "gru"
    model = RecurrentPPOActorCritic(135, 640, 128) if recurrent else ObservationV2HistoryMLP(135, 640)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    torch.manual_seed(100000 + int(selected.get("seed", 43)))
    chosen = [str(x) for x in split["validation"][:3]] + [str(x) for x in split["train"][:3]]
    traces = {sid: run_site(sites[sid], model, recurrent) for sid in chosen}
    after_sha = hashlib.sha256(ck.read_bytes()).hexdigest().upper()
    all_rows = [r for rows in traces.values() for r in rows]
    selected_counts = Counter(r["selected_action_type"] for r in all_rows)
    mask_valid = {name: sum(int(r["type_rows"].get(name, {}).get("valid_count", 0) > 0) for r in all_rows) for name in ("click_element", "fill_input", "select_option", "press_enter")}
    selected_counts = dict(selected_counts)
    interaction_available = sum(int(any(r["type_rows"].get(name, {}).get("valid_count", 0) > 0 for name in mask_valid)) for r in all_rows)
    interaction_viewport = sum(int(any(r["type_rows"].get(name, {}).get("valid_count", 0) > 0 for name in mask_valid) and r["selected_action_type"].startswith("change_viewport")) for r in all_rows)
    mobile_repeat = sum(int(r["selected_action_type"] == "change_viewport_mobile" and r["viewport_before"].get("width") == 375 and r["viewport_after"] == r["viewport_before"]) for r in all_rows)
    summary = {"sites": chosen, "steps": len(all_rows), "selected_action_counts": selected_counts, "mask_valid_step_counts": mask_valid, "steps_with_any_interaction_mask_valid": interaction_available, "interaction_available_but_viewport_selected": interaction_viewport, "already_mobile_same_viewport_selected": mobile_repeat, "ppo_sha_before": before_sha, "ppo_sha_after": after_sha, "ppo_sha_unchanged": before_sha == after_sha, "optimizer_steps": 0, "gt_runtime_access": 0, "traces": traces}
    (BASE / "ppo_interaction_opportunity_trace.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    (BASE / "ppo_interaction_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("sites", "steps", "selected_action_counts", "mask_valid_step_counts", "steps_with_any_interaction_mask_valid", "interaction_available_but_viewport_selected", "already_mobile_same_viewport_selected", "ppo_sha_unchanged")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
