"""GT-blind PPO global interactive-target exposure and selection diagnostic."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.v3.action_contract import is_click_target, is_fill_input_target, is_target_eligible
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.run5_context_aware_dqn_training import CANDIDATE_B_PATHS, _step_v3, _v3_mask, _v3_state, discover_sites, reset_env, read_json
from dqn_error_discovery.v3.run6_candidate_context_harvest_validation import ManagedSite, manifest_order


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\workspace\RL_Errorsite_fullstack_run4")
RUN_ROOT = ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full"
OUT = ROOT / "artifacts/training_v3/run6_ppo_interactive_coverage_diagnostic"
SPACE = V3ActionSpace(32)
INTERACTIVE_TYPES = {"click_element", "fill_input", "select_option", "press_enter"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def target_identity(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("bid") or candidate.get("id") or candidate.get("selector") or candidate.get("text") or "MISSING")


def distribution(ppo: PPOAgent, obs_vector: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=ppo.device)
        policy_obs = obs.unsqueeze(0) if obs.dim() == 1 else obs
        logits, _ = ppo.policy(policy_obs)
        masked = logits.squeeze(0).masked_fill(~torch.as_tensor(mask, dtype=torch.bool, device=ppo.device), -1e9)
        probs = torch.softmax(masked, dim=-1).cpu().numpy()
        entropy = float(-(torch.softmax(masked, dim=-1) * torch.log_softmax(masked, dim=-1)).sum().item())
    return probs, entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=OUT)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episodes-per-site", type=int, default=3)
    parser.add_argument("--max-sites", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    out = args.artifact_dir; out.mkdir(parents=True, exist_ok=True)
    ppo = PPOAgent(635, 640); ppo.load(CANDIDATE_B_PATHS["ppo"]); ppo.policy.eval()
    sites = manifest_order(discover_sites(args.source_root), args.run_root)[:args.max_sites]
    step_rows: list[dict[str, Any]] = []; rank_rows: list[dict[str, Any]] = []
    actions = Counter(); repeat_type = 0; repeat_exact = 0; previous_action: str | None = None; previous_id: int | None = None
    exposure = Counter(); mask_counts = Counter(); slots = Counter(); rejection = Counter(); competition = Counter(); entropies: list[float] = []
    episodes = 0; global_steps = 0; errors: list[dict[str, Any]] = []
    for site in sites:
        if episodes >= args.episodes: break
        with ManagedSite(site, out) as managed:
            if not managed.logs.get("http", {}).get("ok"):
                errors.append({"site_id": site["site_id"], "error": "HTTP_NOT_REACHABLE"}); continue
            for _ in range(args.episodes_per_site):
                if episodes >= args.episodes: break
                episodes += 1; env = None
                try:
                    env, raw = reset_env(site, args.max_steps, 60000)
                    for step in range(args.max_steps):
                        obs = _v3_state(env, raw); mask = SPACE.build_action_mask(raw)
                        candidates = raw.get("candidate_elements", []) if isinstance(raw.get("candidate_elements"), list) else []
                        click_rows = []; input_rows = []; candidate_rows = []
                        for idx, candidate in enumerate(candidates[:SPACE.max_candidates]):
                            if not isinstance(candidate, Mapping): continue
                            click = bool(is_click_target(candidate)); inp = bool(is_fill_input_target(candidate))
                            row = {"candidate_index": idx, "target_identity": target_identity(candidate), "role": candidate.get("role", "MISSING"), "tag": candidate.get("tag", "MISSING"), "visible": candidate.get("visible", "MISSING"), "enabled": candidate.get("enabled", "MISSING"), "clickable": candidate.get("clickable", "MISSING"), "interactable": bool(click or inp), "bbox_valid": bool(is_target_eligible("click_element", candidate) or is_target_eligible("fill_input", candidate)), "input_compatible": inp, "click_compatible": click}
                            candidate_rows.append(row)
                            if click: click_rows.append(idx)
                            if inp: input_rows.append(idx)
                            slots[str((idx // 8) * 8) + "-" + str((idx // 8) * 8 + 7)] += int(click or inp)
                            for typ, ok in (("click_element", click), ("fill_input", inp)):
                                if not ok: continue
                                aid = SPACE.encode(typ, idx)
                                if mask[aid] > 0: mask_counts["mask_valid_interactive_actions"] += 1
                                else: rejection["mask_false"] += 1
                        interactive_count = len(click_rows) + len(input_rows)
                        exposure["total_candidates"] += len(candidate_rows); exposure["interactive_candidates"] += interactive_count; exposure["steps"] += 1
                        exposure["steps_with_interactive"] += int(interactive_count > 0)
                        exposure["steps_without_interactive"] += int(interactive_count == 0)
                        if len(candidates) >= SPACE.max_candidates and any(is_target_eligible("click_element", c) or is_target_eligible("fill_input", c) for c in candidates[SPACE.max_candidates:] if isinstance(c, Mapping)): exposure["truncation_evidence"] += 1
                        probs, entropy = distribution(ppo, obs, mask); entropies.append(entropy)
                        selected = ppo.select_action(obs, _v3_mask(SPACE, raw)); selected_id = int(selected["action_id"]); decoded = SPACE.decode(selected_id); selected_prob = float(probs[selected_id])
                        actions[decoded["action_type"]] += 1
                        is_repeat_type = previous_action == decoded["action_type"]; is_repeat_exact = previous_id == selected_id
                        repeat_type += int(is_repeat_type); repeat_exact += int(is_repeat_exact)
                        previous_action = decoded["action_type"]; previous_id = selected_id
                        valid_ids = np.flatnonzero(mask > 0); order = sorted(valid_ids.tolist(), key=lambda i: float(probs[i]), reverse=True); ranks = {aid: idx + 1 for idx, aid in enumerate(order)}
                        interactive_ids = [SPACE.encode("click_element", i) for i in click_rows if mask[SPACE.encode("click_element", i)] > 0] + [SPACE.encode("fill_input", i) for i in input_rows if mask[SPACE.encode("fill_input", i)] > 0]
                        best_interactive = max(interactive_ids, key=lambda i: float(probs[i])) if interactive_ids else None
                        best_click = max((float(probs[SPACE.encode("click_element", i)]) for i in click_rows if mask[SPACE.encode("click_element", i)] > 0), default=0.0)
                        best_scroll = max(float(probs[SPACE.encode("scroll_up", 0)]), float(probs[SPACE.encode("scroll_down", 0)]))
                        best_inspect = max((float(probs[SPACE.encode(t, 0)]) for t in ("inspect_dom", "inspect_network", "inspect_console") if mask[SPACE.encode(t, 0)] > 0), default=0.0)
                        competition["best_click_probability_sum"] += best_click; competition["best_scroll_probability_sum"] += best_scroll; competition["best_inspect_probability_sum"] += best_inspect
                        if interactive_count > 0: competition["interactive_available_steps"] += 1; competition["click_or_input_selected_available"] += int(decoded["action_type"] in INTERACTIVE_TYPES)
                        rank_row = {"site_id": site["site_id"], "episode": episodes, "step": step + 1, "interactive_ids": interactive_ids, "best_interactive_action": SPACE.decode(best_interactive) if best_interactive is not None else None, "best_interactive_probability": float(probs[best_interactive]) if best_interactive is not None else 0.0, "best_interactive_rank": ranks.get(best_interactive), "selected_action": decoded, "selected_probability": selected_prob, "selected_rank": ranks.get(selected_id), "best_click_probability": best_click, "best_scroll_probability": best_scroll, "best_inspect_probability": best_inspect, "entropy": entropy}
                        rank_rows.append(rank_row)
                        step_rows.append({"site_id": site["site_id"], "episode": episodes, "step": step + 1, "candidate_count": len(candidate_rows), "click_interactive_count": len(click_rows), "input_interactive_count": len(input_rows), "interactive_count": interactive_count, "mask_valid_interactive_actions": len(interactive_ids), "candidate_pool": candidate_rows, "mask_valid_action_count": int(mask.sum()), "selected_action": decoded, "policy_entropy": entropy, "repeat_action_type": is_repeat_type, "repeat_exact_action": is_repeat_exact})
                        after, _, done, info, _ = _step_v3(env, selected_id, raw, SPACE); raw = after; global_steps += 1
                        if done: break
                except Exception as exc: errors.append({"site_id": site["site_id"], "episode": episodes, "error": repr(exc)})
                finally:
                    if env is not None: env.close()
    valid_interactive_steps = exposure["steps_with_interactive"]; mask_steps = sum(1 for r in rank_rows if r["best_interactive_rank"] is not None); selected_interactive = sum(1 for r in rank_rows if r["selected_action"]["action_type"] in INTERACTIVE_TYPES and r["best_interactive_rank"] is not None)
    best_ranks = [r["best_interactive_rank"] for r in rank_rows if r["best_interactive_rank"] is not None]
    summary = {"episodes": episodes, "global_steps": global_steps, "total_candidates": exposure["total_candidates"], "interactive_candidates": exposure["interactive_candidates"], "steps_with_interactive": valid_interactive_steps, "steps_without_interactive": exposure["steps_without_interactive"], "mean_interactive_candidates_per_step": exposure["interactive_candidates"] / max(1, exposure["steps"]), "mask_valid_interactive_actions": mask_counts["mask_valid_interactive_actions"], "interactive_mask_rate": mask_steps / max(1, valid_interactive_steps), "interactive_selection_given_exposure": selected_interactive / max(1, valid_interactive_steps), "interactive_selection_given_mask": selected_interactive / max(1, mask_steps), "best_interactive_rank_mean": float(np.mean(best_ranks)) if best_ranks else None, "best_interactive_rank_median": float(np.median(best_ranks)) if best_ranks else None, "best_interactive_top1": sum(r <= 1 for r in best_ranks) / max(1, len(best_ranks)), "best_interactive_top3": sum(r <= 3 for r in best_ranks) / max(1, len(best_ranks)), "best_interactive_top5": sum(r <= 5 for r in best_ranks) / max(1, len(best_ranks)), "best_interactive_top10": sum(r <= 10 for r in best_ranks) / max(1, len(best_ranks)), "action_distribution": dict(actions), "action_type_repeat_rate": repeat_type / max(1, global_steps), "exact_action_repeat_rate": repeat_exact / max(1, global_steps), "mean_best_click_probability": competition["best_click_probability_sum"] / max(1, global_steps), "mean_best_scroll_probability": competition["best_scroll_probability_sum"] / max(1, global_steps), "mean_best_inspect_probability": competition["best_inspect_probability_sum"] / max(1, global_steps), "mean_policy_entropy": float(np.mean(entropies)) if entropies else None, "interactive_available_selected_share": competition["click_or_input_selected_available"] / max(1, competition["interactive_available_steps"]), "slot_distribution": dict(slots), "mask_rejection": dict(rejection), "truncation_evidence": exposure["truncation_evidence"], "errors": errors, "gt_runtime_access": False, "ppo_modified": False, "dqn_retrained": False}
    write_json(out / "summary.json", summary); write_json(out / "action_distribution.json", {"counts": dict(actions), "summary": summary}); write_json(out / "mask_funnel.json", {"mask_valid_interactive_actions": mask_counts["mask_valid_interactive_actions"], "mask_rejection": dict(rejection), "interactive_mask_rate": summary["interactive_mask_rate"]})
    write_json(out / "posthoc_gt_trigger_funnel.json", {"runtime_gt_access": False, "status": "NOT_RECONSTRUCTED", "reason": "Diagnostic remained GT-blind; no runtime candidate target was selected."})
    with (out / "step_candidates.jsonl").open("w", encoding="utf-8") as h:
        for row in step_rows: h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (out / "policy_ranks.jsonl").open("w", encoding="utf-8") as h:
        for row in rank_rows: h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "global_steps", "total_candidates", "interactive_candidates", "steps_with_interactive", "mask_valid_interactive_actions", "interactive_selection_given_exposure", "interactive_selection_given_mask", "best_interactive_rank_mean", "action_distribution", "mean_policy_entropy", "errors")}, indent=2))


if __name__ == "__main__": main()
