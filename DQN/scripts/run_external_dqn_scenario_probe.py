"""Diagnostic scenario setup for RL_Errorsite_DQN_B.

The scripted steps are only environment preconditions.  They are not passed
to the policy, are not used as DQN labels, and are never promoted to training
data.  Detector and router behavior is measured after each real action.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context  # noqa: E402
from dqn_error_discovery.hybrid.strategy_router import route_candidate  # noqa: E402

OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/external_dqn_scenario_probe.json"


def reset_backend(url: str) -> None:
    try:
        req = urllib.request.Request(url + "/api/test/reset", method="POST")
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def choose(candidates: list[dict], terms: list[str]) -> int | None:
    for i, c in enumerate(candidates):
        text = " ".join(str(c.get(k) or "") for k in ("text", "name", "aria_label", "label"))
        if any(term in text for term in terms) and c.get("visible", True) and c.get("enabled", True):
            return i
    return None


def run_scenario(site_id: str, url: str, steps: list[dict]) -> dict:
    reset_backend(url.replace("5316", "9715").replace("5320", "9719"))
    env = BrowserGymJAWSEnv(site_id=site_id, base_url=url, max_steps=40,
                            max_candidates=32, headless=True, site_profile={})
    raw, _ = env.reset()
    rows = []
    counts = Counter()
    try:
        for step_no, spec in enumerate(steps):
            candidates = raw.get("candidate_elements", []) if isinstance(raw, dict) else []
            idx = choose(candidates, spec["terms"])
            if idx is None:
                rows.append({"step": step_no, "requested": spec, "found": False,
                             "candidate_sample": [str(c.get("text") or c.get("name") or "") for c in candidates[:32]]})
                continue
            aid = env.action_space.encode("click_element", idx)
            before = raw
            nxt, _reward, done, info = env.step(aid)
            after = dict(nxt)
            runtime_signals = dict(after.get("runtime_signals") or {})
            if isinstance(info, dict):
                runtime_signals.update(info)
            after["runtime_signals"] = runtime_signals
            c = candidates[idx]
            target = str(c.get("bid") or "")
            action = {"action_type": "click_element", "candidate_index": idx,
                      "action_id": aid, "execution_success": True,
                      "target_identity": target}
            anomalies = detect_anomalies(
                before, after,
                {"action": action, "action_type": "click_element",
                 "trigger_action_type": "click_element",
                 "trigger_target_identity": target, "site_id": site_id,
                 "last_action_error": False},
                site_profile={},
            )
            routed = []
            for anomaly in anomalies:
                ctx = derive_reproduction_context(anomaly).as_dict()
                candidate = {
                    "reproduction_strategy": ctx.get("reproduction_strategy"),
                    "requires_action_reproduction": bool(ctx.get("requires_action_reproduction")),
                    "reproduction_action_type": ctx.get("reproduction_action_type"),
                    "reproduction_target_identity": ctx.get("reproduction_target_identity"),
                    "target_actionable": bool(ctx.get("requires_action_reproduction") and ctx.get("reproduction_target_identity")),
                    "exact_action_known": False,
                    "chain_id": f"{site_id}:{step_no}",
                }
                route = route_candidate(candidate)
                routed.append({"rule": anomaly.get("rule_id"), "strategy": route.strategy,
                               "owner": route.owner, "dqn_eligible": route.dqn_eligible,
                               "target": ctx.get("reproduction_target_identity")})
                counts[str(anomaly.get("rule_id") or "unknown")] += 1
                counts["dqn_local"] += int(route.dqn_eligible)
            rows.append({"step": step_no, "requested": spec, "found": True,
                         "selected_text": c.get("text"), "target": target,
                         "anomalies": anomalies, "routes": routed,
                         "execution_error": info.get("action_error") if isinstance(info, dict) else None})
            raw = nxt
            if done:
                break
    finally:
        env.close()
    return {"site_id": site_id, "url": url, "steps": rows, "counts": dict(counts),
            "dqn_local": int(counts["dqn_local"])}


def main() -> int:
    results = [
        run_scenario("B01_SNS", "http://127.0.0.1:5316", [
            {"name": "saved", "terms": ["저장한 게시물"]},
            {"name": "logo", "terms": ["M Mingle", "Mingle"]},
            {"name": "more", "terms": ["게시물 더보기", "더보기"]},
        ]),
        run_scenario("B05_상품검색비교", "http://127.0.0.1:5320", [
            {"name": "laptop", "terms": ["노트북"]},
            {"name": "compare", "terms": ["상품 비교", "비교"]},
            {"name": "return", "terms": ["검색 조건으로 돌아가기"]},
        ]),
    ]
    summary = {
        "diagnostic_only": True,
        "gt_policy_input": False,
        "canonical_dataset_promotion": False,
        "results": results,
        "total_dqn_local": sum(int(x["dqn_local"]) for x in results),
        "readiness": "PASS" if any(int(x["dqn_local"]) > 0 for x in results) else "FAIL",
        "readiness_reason": "scenario_setup_observed_dqn_local" if any(int(x["dqn_local"]) > 0 for x in results) else "no_dqn_local_after_diagnostic_preconditions",
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Windows consoles may use cp949; keep the diagnostic result printable
    # without changing the UTF-8 artifact written above.
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
