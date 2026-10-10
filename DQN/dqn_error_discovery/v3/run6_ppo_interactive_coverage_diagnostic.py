"""GT-blind PPO global interactive-target exposure and selection diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from adapters import browsergym_observation_adapter as production_observation_adapter
from adapters.browsergym_observation_adapter import (
    DOM_INTERACTIVE_TAGS,
    INTERACTIVE_ROLES,
    _axtree_nodes,
    _ax_property_bool,
    _ax_value,
    _dom_candidates,
)
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


# Shared PPO execution boundary.  The downstream funnel imports these helpers
# instead of maintaining a second diagnostic implementation of the PPO path.
def production_observation_vector(env: Any, raw: Mapping[str, Any]) -> np.ndarray:
    return _v3_state(env, raw)


def production_action_mask(raw: Mapping[str, Any]) -> np.ndarray:
    return SPACE.build_action_mask(raw)


def production_select_action(
    ppo: PPOAgent, env: Any, raw: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    observation = production_observation_vector(env, raw)
    mask = production_action_mask(raw)
    selected = ppo.select_action(observation, mask)
    decoded = SPACE.decode(int(selected["action_id"]))
    return observation, selected, {**decoded, "action_id": int(selected["action_id"])}


def production_step(
    env: Any,
    raw: Mapping[str, Any],
    selected_action: Mapping[str, Any],
    action_context: Mapping[str, Any] | None = None,
) -> tuple[dict, float, bool, dict, str]:
    context = dict(action_context or {})
    context.setdefault("action_id", int(selected_action["action_id"]))
    context.setdefault("action_type", selected_action.get("action_type"))
    context.setdefault("action_start_timestamp", time.monotonic() * 1000.0)
    env._pending_action_context = context
    return _step_v3(env, int(selected_action["action_id"]), raw, SPACE)


def production_ppo_path_metadata(checkpoint_path: Path) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "adapter_module": production_observation_adapter.__file__,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "obs_dim": 635,
        "action_dim": 640,
        "mask_builder": "V3ActionSpace.build_action_mask",
        "selection": "PPOAgent.select_action",
        "decode": "V3ActionSpace.decode",
        "execution": "run5_context_aware_dqn_training._step_v3",
        "stochastic": True,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def target_identity(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("bid") or candidate.get("id") or candidate.get("selector") or candidate.get("text") or "MISSING")


def raw_interactive_rows(browser_observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Count raw DOM/AXTree interactive candidates before JAWS truncation.

    This is diagnostic-only.  The production adapter and action helpers remain
    unchanged; final candidate eligibility is still measured from the adapted
    production candidate pool below.
    """
    rows: dict[str, dict[str, Any]] = {}

    def add(source: str, identity: Any, role: Any, tag: Any, click: bool, inp: bool) -> None:
        key = str(identity or f"{source}:{len(rows)}")
        row = rows.setdefault(key, {
            "raw_identity": key,
            "sources": [],
            "role": str(role or "MISSING"),
            "tag": str(tag or "MISSING"),
            "click_interactive": False,
            "input_interactive": False,
        })
        if source not in row["sources"]:
            row["sources"].append(source)
        row["click_interactive"] = bool(row["click_interactive"] or click)
        row["input_interactive"] = bool(row["input_interactive"] or inp)

    for node in _axtree_nodes(browser_observation.get("axtree_object")):
        if not isinstance(node, Mapping) or bool(node.get("ignored")):
            continue
        role = str(_ax_value(node.get("role") or node.get("role_name") or node.get("tag")) or "").lower()
        tag = str(_ax_value(node.get("tag") or node.get("tagName")) or "").lower()
        focusable = _ax_property_bool(node, "focusable")
        explicit_clickable = bool(node.get("clickable")) and role not in {"rootwebarea", "document", "generic", "group", "region"}
        clickable = bool(role in INTERACTIVE_ROLES or tag in DOM_INTERACTIVE_TAGS or focusable or explicit_clickable)
        inputable = bool(role in {"textbox", "searchbox", "combobox"} or tag in {"input", "textarea", "select"})
        if clickable or inputable:
            add("axtree", node.get("backendDOMNodeId") or node.get("nodeId") or node.get("id"), role, tag, clickable, inputable)

    for candidate in _dom_candidates(browser_observation.get("dom_object"), ""):
        if not isinstance(candidate, Mapping):
            continue
        role = str(candidate.get("role") or "").lower()
        tag = str(candidate.get("tag") or "").lower()
        click = bool(tag in DOM_INTERACTIVE_TAGS or role in INTERACTIVE_ROLES or candidate.get("clickable"))
        inp = bool(tag in {"input", "textarea", "select"} or role in {"textbox", "searchbox", "combobox"} or candidate.get("fillable"))
        if click or inp:
            add("dom", candidate.get("bid") or candidate.get("id"), role, tag, click, inp)

    extra = browser_observation.get("extra_element_properties")
    if isinstance(extra, Mapping):
        for identity, prop in extra.items():
            if not isinstance(prop, Mapping):
                continue
            role = str(prop.get("role") or "").lower()
            tag = str(prop.get("tag") or prop.get("tagName") or "").lower()
            click = bool(prop.get("clickable") or prop.get("set_of_marks") or role in INTERACTIVE_ROLES or tag in DOM_INTERACTIVE_TAGS)
            inp = bool(prop.get("fillable") or tag in {"input", "textarea", "select"} or role in {"textbox", "searchbox", "combobox"})
            if click or inp:
                add("extra_element_properties", identity, role, tag, click, inp)
    return list(rows.values())


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = args.artifact_dir; out.mkdir(parents=True, exist_ok=True)
    ppo = PPOAgent(635, 640); ppo.load(CANDIDATE_B_PATHS["ppo"]); ppo.policy.eval()
    sites = manifest_order(discover_sites(args.source_root), args.run_root)[:args.max_sites]
    step_rows: list[dict[str, Any]] = []; rank_rows: list[dict[str, Any]] = []
    actions = Counter(); repeat_type = 0; repeat_exact = 0; previous_action: str | None = None; previous_id: int | None = None
    exposure = Counter(); mask_counts = Counter(); slots = Counter(); rejection = Counter(); competition = Counter(); entropies: list[float] = []
    raw_rows_all: list[dict[str, Any]] = []
    encoding_checks: list[dict[str, Any]] = []
    mapping_checks: list[dict[str, Any]] = []
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
                        obs = production_observation_vector(env, raw); mask = production_action_mask(raw)
                        browser_observation = getattr(env, "_browser_obs", {})
                        raw_rows = raw_interactive_rows(browser_observation if isinstance(browser_observation, Mapping) else {})
                        raw_click_rows = [row for row in raw_rows if row["click_interactive"]]
                        raw_input_rows = [row for row in raw_rows if row["input_interactive"]]
                        raw_rows_all.append({
                            "site_id": site["site_id"],
                            "episode": episodes,
                            "step": step + 1,
                            "raw_clickable_count": len(raw_click_rows),
                            "raw_inputable_count": len(raw_input_rows),
                            "raw_interactive_count": len(raw_rows),
                            "raw_interactive_elements": raw_rows,
                        })
                        exposure["raw_clickable"] += len(raw_click_rows)
                        exposure["raw_inputable"] += len(raw_input_rows)
                        exposure["raw_interactive"] += len(raw_rows)
                        exposure["raw_interactive_steps"] += int(bool(raw_rows))
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
                        interactive_indices = sorted(set(click_rows + input_rows))
                        interactive_count = len(interactive_indices)
                        exposure["total_candidates"] += len(candidate_rows); exposure["interactive_candidates"] += interactive_count; exposure["steps"] += 1
                        exposure["steps_with_interactive"] += int(interactive_count > 0)
                        exposure["steps_without_interactive"] += int(interactive_count == 0)
                        if len(candidates) >= SPACE.max_candidates and any(is_target_eligible("click_element", c) or is_target_eligible("fill_input", c) for c in candidates[SPACE.max_candidates:] if isinstance(c, Mapping)): exposure["truncation_evidence"] += 1
                        probs, entropy = distribution(ppo, obs, mask); entropies.append(entropy)
                        _selected_obs, selected, decoded = production_select_action(ppo, env, raw); selected_id = int(selected["action_id"]); selected_prob = float(probs[selected_id])
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
                        for action_id in interactive_ids:
                            decoded_action = SPACE.decode(action_id)
                            candidate_index = int(decoded_action["candidate_index"])
                            expected_identity = candidate_rows[candidate_index]["target_identity"] if candidate_index < len(candidate_rows) else "MISSING"
                            mapping_checks.append({
                                "action_id": int(action_id),
                                "decoded_action_type": decoded_action["action_type"],
                                "decoded_candidate_index": candidate_index,
                                "expected_target_identity": expected_identity,
                                "mapping_valid": expected_identity != "MISSING",
                            })
                        if interactive_indices:
                            encoding_checks.append({
                                "site_id": site["site_id"],
                                "episode": episodes,
                                "step": step + 1,
                                "candidate_indices": interactive_indices,
                                "encoding_valid": all(idx < SPACE.max_candidates for idx in interactive_indices),
                                "source_fields": ["visible", "enabled", "role", "tag", "bbox", "click_compatible", "input_compatible"],
                            })
                        rank_row = {"site_id": site["site_id"], "episode": episodes, "step": step + 1, "interactive_ids": interactive_ids, "best_interactive_action": SPACE.decode(best_interactive) if best_interactive is not None else None, "best_interactive_probability": float(probs[best_interactive]) if best_interactive is not None else 0.0, "best_interactive_rank": ranks.get(best_interactive), "selected_action": decoded, "selected_probability": selected_prob, "selected_rank": ranks.get(selected_id), "best_click_probability": best_click, "best_scroll_probability": best_scroll, "best_inspect_probability": best_inspect, "entropy": entropy}
                        rank_rows.append(rank_row)
                        step_rows.append({"site_id": site["site_id"], "episode": episodes, "step": step + 1, "candidate_count": len(candidate_rows), "click_interactive_count": len(click_rows), "input_interactive_count": len(input_rows), "interactive_count": interactive_count, "mask_valid_interactive_actions": len(interactive_ids), "candidate_pool": candidate_rows, "mask_valid_action_count": int(mask.sum()), "selected_action": decoded, "policy_entropy": entropy, "repeat_action_type": is_repeat_type, "repeat_exact_action": is_repeat_exact})
                        after, _, done, info, _ = production_step(env, raw, decoded); raw = after; global_steps += 1
                        if done: break
                except Exception as exc: errors.append({"site_id": site["site_id"], "episode": episodes, "error": repr(exc)})
                finally:
                    if env is not None: env.close()
    valid_interactive_steps = exposure["steps_with_interactive"]; mask_steps = sum(1 for r in rank_rows if r["best_interactive_rank"] is not None); selected_interactive = sum(1 for r in rank_rows if r["selected_action"]["action_type"] in INTERACTIVE_TYPES and r["best_interactive_rank"] is not None)
    best_ranks = [r["best_interactive_rank"] for r in rank_rows if r["best_interactive_rank"] is not None]
    mapping_static_ok = all(SPACE.decode(SPACE.encode(action_type, index))["action_type"] == action_type and SPACE.decode(SPACE.encode(action_type, index))["candidate_index"] == index for action_type in SPACE.action_types for index in range(SPACE.max_candidates))
    mapping_runtime_mismatches = sum(1 for row in mapping_checks if not row["mapping_valid"])
    summary = {"episodes": episodes, "global_steps": global_steps, "total_candidates": exposure["total_candidates"], "raw_clickable_elements": exposure["raw_clickable"], "raw_inputable_elements": exposure["raw_inputable"], "raw_interactive_elements": exposure["raw_interactive"], "steps_with_raw_interactive": exposure["raw_interactive_steps"], "final_interactive_candidates": exposure["interactive_candidates"], "interactive_candidates": exposure["interactive_candidates"], "steps_with_interactive": valid_interactive_steps, "steps_without_interactive": exposure["steps_without_interactive"], "interactive_exposure_rate": valid_interactive_steps / max(1, exposure["steps"]), "mean_interactive_candidates_per_step": exposure["interactive_candidates"] / max(1, exposure["steps"]), "mask_valid_interactive_actions": mask_counts["mask_valid_interactive_actions"], "interactive_mask_rate": mask_steps / max(1, valid_interactive_steps), "interactive_selection_given_exposure": selected_interactive / max(1, valid_interactive_steps), "interactive_policy_selection_rate": selected_interactive / max(1, mask_steps), "interactive_selection_given_mask": selected_interactive / max(1, mask_steps), "best_interactive_rank_mean": float(np.mean(best_ranks)) if best_ranks else None, "best_interactive_rank_median": float(np.median(best_ranks)) if best_ranks else None, "best_interactive_top1": sum(r <= 1 for r in best_ranks) / max(1, len(best_ranks)), "best_interactive_top3": sum(r <= 3 for r in best_ranks) / max(1, len(best_ranks)), "best_interactive_top5": sum(r <= 5 for r in best_ranks) / max(1, len(best_ranks)), "best_interactive_top10": sum(r <= 10 for r in best_ranks) / max(1, len(best_ranks)), "action_distribution": dict(actions), "action_type_repeat_rate": repeat_type / max(1, global_steps), "exact_action_repeat_rate": repeat_exact / max(1, global_steps), "mean_best_click_probability": competition["best_click_probability_sum"] / max(1, global_steps), "mean_best_scroll_probability": competition["best_scroll_probability_sum"] / max(1, global_steps), "mean_best_inspect_probability": competition["best_inspect_probability_sum"] / max(1, global_steps), "mean_policy_entropy": float(np.mean(entropies)) if entropies else None, "interactive_available_selected_share": competition["click_or_input_selected_available"] / max(1, competition["interactive_available_steps"]), "slot_distribution": dict(slots), "mask_rejection": dict(rejection), "truncation_evidence": exposure["truncation_evidence"], "errors": errors, "gt_runtime_access": False, "ppo_modified": False, "dqn_retrained": False, "ppo_path": production_ppo_path_metadata(CANDIDATE_B_PATHS["ppo"])}
    write_json(out / "summary.json", summary); write_json(out / "action_distribution.json", {"counts": dict(actions), "summary": summary}); write_json(out / "policy_action_distribution.json", {"counts": dict(actions), "interactive_available_action_counts": {}, "summary": summary}); write_json(out / "mask_funnel.json", {"mask_valid_interactive_actions": mask_counts["mask_valid_interactive_actions"], "mask_rejection": dict(rejection), "interactive_mask_rate": summary["interactive_mask_rate"]}); write_json(out / "candidate_exposure_funnel.json", {"summary": summary, "steps": raw_rows_all}); write_json(out / "observation_encoding_check.json", {"checked": len(encoding_checks), "mismatches": sum(1 for row in encoding_checks if not row["encoding_valid"]), "rows": encoding_checks}); write_json(out / "action_mapping_check.json", {"static_mapping_valid": mapping_static_ok, "runtime_checked": len(mapping_checks), "runtime_mismatches": mapping_runtime_mismatches, "rows": mapping_checks})
    write_json(out / "posthoc_gt_trigger_funnel.json", {"runtime_gt_access": False, "status": "NOT_RECONSTRUCTED", "reason": "Diagnostic remained GT-blind; no runtime candidate target was selected."})
    with (out / "step_candidates.jsonl").open("w", encoding="utf-8") as h:
        for row in step_rows: h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (out / "policy_ranks.jsonl").open("w", encoding="utf-8") as h:
        for row in rank_rows: h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "global_steps", "total_candidates", "interactive_candidates", "steps_with_interactive", "mask_valid_interactive_actions", "interactive_selection_given_exposure", "interactive_selection_given_mask", "best_interactive_rank_mean", "action_distribution", "mean_policy_entropy", "errors")}, indent=2))


if __name__ == "__main__": main()
