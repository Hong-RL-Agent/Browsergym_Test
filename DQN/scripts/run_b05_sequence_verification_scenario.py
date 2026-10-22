"""Diagnostic B05 sequence verification with a fresh-evidence Oracle.

The workflow setup is normal runtime interaction.  The local action is chosen
from the observed valid action mask; no bug target or expected action is
injected.  The scenario Oracle requires a fresh same-chain R12 observation
after the local action, so a pre-existing candidate cannot count as progress.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from models.action_space import ActionSpace  # noqa: E402
from models.observation_v2 import CoverageMemory, ObservationV2Encoder  # noqa: E402
from dqn_error_discovery.hybrid.dqn_goal_context import encode_goal_context  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context  # noqa: E402
from dqn_error_discovery.hybrid.strategy_router import route_candidate  # noqa: E402

OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/b05_sequence_verification_scenario.json"
SITE_ID = "B05_상품검색비교"
BASE_URL = "http://127.0.0.1:5320"


def reset_backend() -> None:
    try:
        request = urllib.request.Request("http://127.0.0.1:9719/api/test/reset", method="POST")
        urllib.request.urlopen(request, timeout=5).read()
    except Exception:
        pass


def choose_bid(observation: dict, bid: str) -> int | None:
    for index, candidate in enumerate(observation.get("candidate_elements", [])):
        if str(candidate.get("bid")) == bid and candidate.get("visible", True) and candidate.get("enabled", True):
            return index
    return None


def choose_text(observation: dict, terms: tuple[str, ...]) -> int | None:
    for index, candidate in enumerate(observation.get("candidate_elements", [])):
        text = " ".join(str(candidate.get(key) or "") for key in ("text", "name", "aria_label", "label"))
        if any(term in text for term in terms) and candidate.get("visible", True) and candidate.get("enabled", True):
            return index
    return None


def merge_signals(observation: dict, info: object) -> dict:
    result = dict(observation)
    signals = dict(result.get("runtime_signals") or {})
    if isinstance(info, dict):
        signals.update(info)
    result["runtime_signals"] = signals
    return result


def r12_rows(before: dict, after: dict, action: dict) -> list[dict]:
    anomalies = detect_anomalies(before, after, action, site_profile={})
    return [item for item in anomalies if str(item.get("rule_id")) == "R12"]


def execute_click(env: BrowserGymJAWSEnv, observation: dict, index: int, label: str, rows: list[dict]) -> dict:
    candidate = observation["candidate_elements"][index]
    target = str(candidate.get("bid") or "")
    action_id = env.action_space.encode("click_element", index)
    before = observation
    next_observation, _reward, done, info = env.step(action_id)
    after = merge_signals(dict(next_observation), info)
    action = {
        "action": {"action_type": "click_element", "candidate_index": index, "action_id": action_id},
        "action_type": "click_element",
        "trigger_action_type": "click_element",
        "trigger_target_identity": target,
        "target_identity": target,
        "site_id": SITE_ID,
        "last_action_error": bool(isinstance(info, dict) and info.get("action_error")),
    }
    anomalies = detect_anomalies(before, after, action, site_profile={})
    routes = []
    for anomaly in anomalies:
        context = derive_reproduction_context(anomaly).as_dict()
        route_input = {
            "reproduction_strategy": context.get("reproduction_strategy"),
            "requires_action_reproduction": bool(context.get("requires_action_reproduction")),
            "reproduction_action_type": context.get("reproduction_action_type"),
            "reproduction_target_identity": context.get("reproduction_target_identity"),
            "target_actionable": bool(context.get("requires_action_reproduction") and context.get("reproduction_target_identity")),
            "exact_action_known": False,
            "chain_id": f"{SITE_ID}:b05-sequence-verification",
        }
        route = route_candidate(route_input)
        routes.append({"rule": anomaly.get("rule_id"), "strategy": route.strategy, "owner": route.owner, "dqn_eligible": route.dqn_eligible, "evidence_id": anomaly.get("evidence_id") or (anomaly.get("evidence") or {}).get("event_id")})
    rows.append({"label": label, "action": action, "anomalies": anomalies, "routes": routes, "done": done})
    return after


def main() -> int:
    reset_backend()
    env = BrowserGymJAWSEnv(site_id=SITE_ID, base_url=BASE_URL, max_steps=80, max_candidates=300, headless=True, site_profile={})
    rows: list[dict] = []
    counts = Counter()
    result: dict = {"diagnostic_only": True, "gt_policy_input": False, "canonical_dataset_promotion": False, "site_id": SITE_ID}
    try:
        observation, _ = env.reset()
        for label, bid in (("category_laptop", "35"), ("brand_filter_1", "138"), ("brand_filter_2", "166"), ("spec_filter", "514")):
            index = choose_bid(observation, bid)
            if index is None:
                rows.append({"label": label, "found": False, "bid": bid})
                continue
            observation = execute_click(env, observation, index, label, rows)
        for index in range(3):
            observation, _reward, done, info = env.step(env.action_space.encode("scroll_down", 0))
            observation = merge_signals(dict(observation), info)
            rows.append({"label": f"scroll_{index}", "action_type": "scroll_down", "done": done})
        # Select the first two visible product checkboxes from runtime DOM/candidates.
        checkbox_indices = [
            index for index, candidate in enumerate(observation.get("candidate_elements", []))
            if candidate.get("visible", True) and candidate.get("enabled", True)
            and str(candidate.get("tag") or "").lower() == "input"
            and str(candidate.get("input_type") or candidate.get("type") or "").lower() == "checkbox"
        ][:2]
        for number, index in enumerate(checkbox_indices, 1):
            observation = execute_click(env, observation, index, f"product_checkbox_{number}", rows)
        for label, terms in (("open_compare", ("상품 비교", "비교")), ("return_search", ("검색 조건으로", "돌아가기"))):
            index = choose_text(observation, terms)
            if index is not None:
                observation = execute_click(env, observation, index, label, rows)
        # Require a runtime R12 route and at least two valid local actions.
        r12_route = None
        for row in rows:
            for route in row.get("routes", []):
                if route.get("rule") == "R12" and route.get("owner") == "DQN_LOCAL":
                    r12_route = route
        canonical_space = ActionSpace(32)
        mask = canonical_space.build_action_mask(observation)
        valid_ids = [index for index, value in enumerate(mask) if float(value) > 0.0]
        counts["valid_local_actions"] = len(valid_ids)
        counts["dqn_local_routes"] = int(r12_route is not None)
        if r12_route is None or len(valid_ids) < 2:
            result.update({"rows": rows, "counts": dict(counts), "readiness": "FAIL", "failure": "missing_dqn_local_or_multi_action"})
            return 1
        # Diagnostic baseline: choose the first valid target-bearing local
        # action. R12's runtime detector contract requires the post-action
        # action type to be a real interaction (not inspect_dom).
        click_ids = [index for index in valid_ids if canonical_space.decode(index)["action_type"] == "click_element"]
        # Prefer an observed checkbox/label filter control because it is a
        # normal local state-changing action, while remaining independent of
        # the defect catalog or a pre-written target answer.
        local_filter_clicks = [
            index for index in click_ids
            if (
                str(observation.get("candidate_elements", [])[canonical_space.decode(index)["candidate_index"]].get("role") or "").lower() in {"checkbox", "radio"}
                or (
                    str(observation.get("candidate_elements", [])[canonical_space.decode(index)["candidate_index"]].get("role") or "").lower() == "input"
                    and canonical_space.decode(index)["candidate_index"] >= 3
                )
            or str(observation.get("candidate_elements", [])[canonical_space.decode(index)["candidate_index"]].get("tag") or "").lower() == "label"
            )
        ]
        offset = int(os.environ.get("B05_LOCAL_FILTER_OFFSET", "0") or 0)
        selected_id = local_filter_clicks[offset] if local_filter_clicks and offset < len(local_filter_clicks) else (click_ids[0] if click_ids else next(index for index in valid_ids if canonical_space.decode(index)["action_type"] not in {"noop", "finish_episode"}))
        decoded = canonical_space.decode(selected_id)
        before_local = observation
        env_action_id = env.action_space.encode(decoded["action_type"], decoded["candidate_index"])
        local_next, _reward, done, info = env.step(env_action_id)
        after_local = merge_signals(dict(local_next), info)
        local_action = {"action_type": decoded["action_type"], "candidate_index": decoded["candidate_index"], "action_id": selected_id, "site_id": SITE_ID}
        fresh_r12 = r12_rows(before_local, after_local, local_action)
        same_chain_fresh = bool(fresh_r12 and r12_route.get("evidence_id") != ((fresh_r12[0].get("evidence") or {}).get("event_id") or fresh_r12[0].get("evidence_id")))
        oracle_before = {"state": "insufficient_evidence", "progress": False, "reason": "candidate_exists_but_local_confirmation_not_started"}
        oracle_after = {"state": "satisfied" if same_chain_fresh else "insufficient_evidence", "progress": same_chain_fresh, "fresh_r12_count": len(fresh_r12), "same_chain_fresh": same_chain_fresh}
        outcome = "SUCCESSFUL_REPRODUCTION" if same_chain_fresh else "FAILED_REPRODUCTION"
        counts["runtime_successful"] = int(outcome == "SUCCESSFUL_REPRODUCTION")
        counts["runtime_failed"] = int(outcome == "FAILED_REPRODUCTION")
        encoder = ObservationV2Encoder(32)
        before_vec = encoder.encode(before_local, coverage=CoverageMemory())
        after_vec = encoder.encode(after_local, previous_action=decoded, previous_outcome={"execution_success": True}, coverage=CoverageMemory())
        goal = encode_goal_context({"active_candidate": True, "strategy": "SEQUENCE_REPLAY", "candidate_family": "semantic", "oracle_family": "STATE_TRANSITION_ORACLE", "confirmation_progress": 0.0, "same_chain_progress": 0.0, "remaining_local_budget": 1.0, "prior_execution_success": 0.0})
        import numpy as np
        state_vec = np.concatenate([before_vec, goal, np.zeros(62, dtype=np.float32)]).astype(np.float32)
        next_vec = np.concatenate([after_vec, goal, np.zeros(62, dtype=np.float32)]).astype(np.float32)
        result.update({
            "rows": rows,
            "counts": dict(counts),
            "dqn_local": 1,
            "multi_action_state": len(valid_ids) >= 2,
            "required_target_masked_out": 0,
            "local_action": {"selected": decoded, "selection_method": "diagnostic_first_valid_observed_filter_or_target_action", "execution_success": not bool(isinstance(info, dict) and info.get("action_error")), "done": done},
            "local_candidate_sample": [
                {"index": i, "bid": c.get("bid"), "role": c.get("role"), "tag": c.get("tag"), "text": c.get("text"), "name": c.get("name"), "clickable": c.get("clickable"), "visible": c.get("visible"), "enabled": c.get("enabled")}
                for i, c in enumerate(before_local.get("candidate_elements", [])[:80])
            ],
            "oracle_before": oracle_before,
            "oracle_after": oracle_after,
            "fresh_r12_after_local_action": fresh_r12,
            "training_transition": {"state": state_vec.tolist(), "next_state": next_vec.tolist(), "action": selected_id, "reward": 1.3 if same_chain_fresh else 0.0, "done": False, "current_mask": mask.tolist(), "next_mask": canonical_space.build_action_mask(after_local).tolist(), "runtime": True, "synthetic": False},
            "outcome": outcome,
            "runtime_provenance": True,
            "readiness": "PASS" if outcome in {"PARTIAL_REPRODUCTION", "SUCCESSFUL_REPRODUCTION"} else "FAIL",
            "readiness_reasons": [] if outcome in {"PARTIAL_REPRODUCTION", "SUCCESSFUL_REPRODUCTION"} else ["NO_ORACLE_PROGRESS_OR_RUNTIME_PARTIAL_SUCCESS"],
        })
    finally:
        env.close()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("site_id", "counts", "dqn_local", "multi_action_state", "required_target_masked_out", "local_action", "oracle_before", "oracle_after", "outcome", "readiness")}, indent=2, ensure_ascii=False))
    return 0 if result.get("readiness") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
