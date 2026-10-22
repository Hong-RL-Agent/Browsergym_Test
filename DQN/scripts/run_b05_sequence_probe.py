"""Diagnostic-only B05 normal precondition workflow probe."""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context  # noqa: E402
from dqn_error_discovery.hybrid.strategy_router import route_candidate  # noqa: E402
from envs.browsergym_jaws_env import _active_page  # noqa: E402

OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/external_b05_sequence_probe.json"
LOCAL_EXECUTED = False


def reset_backend() -> None:
    try:
        req = urllib.request.Request("http://127.0.0.1:9719/api/test/reset", method="POST")
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def _merge_runtime_signals(observation, info):
    merged = dict(observation.get("runtime_signals") or {}) if isinstance(observation, dict) else {}
    if isinstance(info, dict):
        merged.update(info)
    if isinstance(observation, dict):
        observation["runtime_signals"] = merged
    return observation


def _compare_oracle(observation):
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, dict) else {}
    external = runtime.get("external_runtime_state", {}) if isinstance(runtime, dict) else {}
    state = external.get("compare_lab_state", {}) if isinstance(external, dict) else {}
    if not isinstance(state, dict):
        return {"state": "insufficient_evidence", "progress": False}
    filters = state.get("filters") if isinstance(state.get("filters"), dict) else {}
    last_search = state.get("lastSearch") if isinstance(state.get("lastSearch"), dict) else {}
    compare = state.get("compare") if isinstance(state.get("compare"), dict) else {}
    summary = compare.get("summary") if isinstance(compare.get("summary"), dict) else {}
    count = int(summary.get("count") or 0)
    active = {k: v for k, v in filters.items() if v not in (None, "", [], {})}
    remembered = {k: v for k, v in last_search.items() if v not in (None, "", [], {})}
    mismatch = bool(count > 0 and remembered and any(active.get(k) != v for k, v in remembered.items()))
    return {"state": "satisfied" if mismatch else "unsatisfied", "progress": False, "comparison_count": count, "active_filters": active, "remembered_search": remembered}


def record_step(env, raw, idx, site_id, step_id, label, counts, rows):
    candidates = raw.get("candidate_elements", [])
    c = candidates[idx]
    target = str(c.get("bid") or "")
    before = raw
    next_raw, _reward, done, info = env.step(env.action_space.encode("click_element", idx))
    after = dict(next_raw)
    runtime_signals = dict(after.get("runtime_signals") or {})
    if isinstance(info, dict):
        runtime_signals.update(info)
    after["runtime_signals"] = runtime_signals
    action = {"action": {"action_type": "click_element", "candidate_index": idx, "action_id": env.action_space.encode("click_element", idx), "execution_success": True, "target_identity": target}, "action_type": "click_element", "trigger_action_type": "click_element", "trigger_target_identity": target, "site_id": site_id, "last_action_error": bool(isinstance(info, dict) and info.get("action_error"))}
    anomalies = detect_anomalies(before, after, action, site_profile={})
    routes = []
    for anomaly in anomalies:
        ctx = derive_reproduction_context(anomaly).as_dict()
        candidate = {"reproduction_strategy": ctx.get("reproduction_strategy"), "requires_action_reproduction": bool(ctx.get("requires_action_reproduction")), "reproduction_action_type": ctx.get("reproduction_action_type"), "reproduction_target_identity": ctx.get("reproduction_target_identity"), "target_actionable": bool(ctx.get("requires_action_reproduction") and ctx.get("reproduction_target_identity")), "exact_action_known": False, "chain_id": f"{site_id}:{step_id}"}
        route = route_candidate(candidate)
        local_mask = env.action_space.build_action_mask(after)
        mask_stats = env.action_space.build_action_mask_stats(after, local_mask)
        valid_local_actions = int(mask_stats.get("valid_action_count", 0))
        if route.dqn_eligible:
            counts["canonical_dqn_local"] += 1
            counts["multi_action_local_states"] += int(valid_local_actions >= 2)
        routes.append({"rule": anomaly.get("rule_id"), "strategy": route.strategy, "owner": route.owner, "dqn_eligible": route.dqn_eligible, "target": ctx.get("reproduction_target_identity"), "valid_local_actions": valid_local_actions, "action_mask_stats": mask_stats})
        counts[str(anomaly.get("rule_id") or "unknown")] += 1
        counts["dqn_local"] += int(route.dqn_eligible)
        global LOCAL_EXECUTED
        if route.dqn_eligible and not LOCAL_EXECUTED:
            local_mask = env.action_space.build_action_mask(after)
            valid_ids = [i for i, value in enumerate(local_mask) if float(value) > 0.0]
            non_noop = [i for i in valid_ids if env.action_space.decode(i).get("action_type") not in {"noop", "finish_episode"}]
            selected_id = non_noop[0] if non_noop else (valid_ids[0] if valid_ids else None)
            local_before = after
            oracle_before = _compare_oracle(local_before)
            local_record = {"valid_local_actions": len(valid_ids), "selected_action_id": selected_id, "oracle_before": oracle_before, "selection_method": "diagnostic_first_valid_non_noop"}
            if selected_id is None:
                local_record.update({"execution_success": False, "outcome": "FAILED_REPRODUCTION", "oracle_after": {"state": "insufficient_evidence", "progress": False}})
                counts["runtime_failed"] += 1
            else:
                decoded_local = env.action_space.decode(selected_id)
                local_next, _local_reward, local_done, local_info = env.step(selected_id)
                local_after = _merge_runtime_signals(dict(local_next), local_info)
                oracle_after = _compare_oracle(local_after)
                execution_success = not bool(isinstance(local_info, dict) and local_info.get("last_action_error"))
                before_ext = (local_before.get("runtime_signals", {}) or {}).get("external_runtime_state")
                after_ext = (local_after.get("runtime_signals", {}) or {}).get("external_runtime_state")
                before_page = local_before.get("page_state", {}) or {}
                after_page = local_after.get("page_state", {}) or {}
                meaningful_state_changed = bool(before_ext != after_ext or before_page.get("url") != after_page.get("url") or abs(int(after_page.get("page_text_length", 0) or 0) - int(before_page.get("page_text_length", 0) or 0)) > 40)
                state_changed = local_before.get("runtime_signals") != local_after.get("runtime_signals") or local_before.get("page_state") != local_after.get("page_state")
                if execution_success and oracle_before.get("state") != "satisfied" and oracle_after.get("state") == "satisfied":
                    outcome = "SUCCESSFUL_REPRODUCTION"
                    counts["runtime_successful"] += 1
                elif execution_success and meaningful_state_changed and oracle_after.get("state") != oracle_before.get("state"):
                    outcome = "PARTIAL_REPRODUCTION"
                    counts["runtime_partial"] += 1
                else:
                    outcome = "FAILED_REPRODUCTION"
                    counts["runtime_failed"] += 1
                local_record.update({"selected_action": decoded_local, "execution_success": execution_success, "oracle_after": oracle_after, "state_changed": state_changed, "meaningful_state_changed": meaningful_state_changed, "outcome": outcome, "done": local_done, "info": local_info})
            rows.append({"step": step_id, "label": f"dqn_local_diagnostic_{anomaly.get('rule_id')}", "local_execution": local_record})
            LOCAL_EXECUTED = True
    rows.append({"step": step_id, "label": label, "candidate": {"text": c.get("text"), "name": c.get("name"), "bid": target, "role": c.get("role")}, "anomalies": anomalies, "routes": routes, "info": info, "done": done})
    return next_raw, done


def main() -> int:
    reset_backend()
    env = BrowserGymJAWSEnv(site_id="B05_상품검색비교", base_url="http://127.0.0.1:5320", max_steps=60, max_candidates=300, headless=True, site_profile={})
    counts = Counter()
    rows = []
    try:
        raw, _ = env.reset()
        raw, done = record_step(env, raw, 19, "B05_상품검색비교", 0, "category_laptop", counts, rows)
        for label, desired_bid in [("brand_filter_1", "138"), ("brand_filter_2", "166"), ("spec_filter", "514")]:
            matches = [i for i, c in enumerate(raw.get("candidate_elements", [])) if str(c.get("bid")) == desired_bid and c.get("visible", True)]
            if matches:
                raw, done = record_step(env, raw, matches[0], "B05_상품검색비교", len(rows), label, counts, rows)
            else:
                rows.append({"step": len(rows), "label": label, "found": False, "desired_bid": desired_bid})
        for j in range(3):
            raw, _reward, done, info = env.step(env.action_space.encode("scroll_down", 0))
            rows.append({"step": len(rows), "label": f"scroll_{j}", "action": "scroll_down", "candidate_count": len(raw.get("candidate_elements", []))})
        page = _active_page(env.env)
        input_bids = page.evaluate("""() => Array.from(document.querySelectorAll('label.compare-check input[type=checkbox]')).filter((x) => x.getAttribute('browsergym_visibility_ratio') === '1').map((x) => Number(x.getAttribute('bid')) - 1)""")
        for n, bid in enumerate(input_bids[:2], 1):
            matches = [i for i, c in enumerate(raw.get("candidate_elements", [])) if str(c.get("bid")) == str(bid) and c.get("visible", True)]
            if not matches:
                rows.append({"step": len(rows), "label": f"product_checkbox_{n}", "found": False, "desired_bid": bid})
                continue
            raw, done = record_step(env, raw, matches[0], "B05_상품검색비교", len(rows), f"product_checkbox_{n}", counts, rows)
        for label, terms in [("open_compare", ["상품 비교"]), ("return_search", ["검색 조건으로 돌아가기"])]:
            candidates = raw.get("candidate_elements", [])
            matches = []
            for i, c in enumerate(candidates):
                text = " ".join(str(c.get(k) or "") for k in ("text", "name", "aria_label", "label"))
                if any(term in text for term in terms) and c.get("visible", True) and c.get("enabled", True):
                    matches.append(i)
            if matches:
                raw, done = record_step(env, raw, matches[0], "B05_상품검색비교", len(rows), label, counts, rows)
            else:
                rows.append({"step": len(rows), "label": label, "found": False, "candidate_sample": [{"text": c.get("text"), "name": c.get("name"), "bid": c.get("bid")} for c in candidates if c.get("visible", True)][-30:]})
        page = _active_page(env.env)
        final_state = page.evaluate("() => window.__compareLabState || null")
    finally:
        env.close()
    out = {"diagnostic_only": True, "gt_policy_input": False, "canonical_dataset_promotion": False, "site": "B05_상품검색비교", "steps": rows, "counts": dict(counts), "final_state": final_state, "dqn_local": int(counts["dqn_local"]), "readiness": "PASS" if counts["dqn_local"] > 0 else "FAIL"}
    out["runtime_failed"] = int(counts.get("runtime_failed", 0))
    out["runtime_partial"] = int(counts.get("runtime_partial", 0))
    out["runtime_successful"] = int(counts.get("runtime_successful", 0))
    out["readiness"] = "PASS" if out["dqn_local"] > 0 and (out["runtime_partial"] + out["runtime_successful"]) > 0 and int(counts.get("multi_action_local_states", 0)) > 0 else "FAIL"
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "dqn_local": out["dqn_local"], "readiness": out["readiness"], "final_state": final_state}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
