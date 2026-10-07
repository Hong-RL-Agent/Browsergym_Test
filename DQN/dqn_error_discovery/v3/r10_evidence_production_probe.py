"""Scoped R10 network/UI evidence production probe.

Runtime collection is GT-blind.  GT endpoints are loaded only after browser
execution to annotate post-hoc trigger matches in the output artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.train import _episode_env, _step_v3, _v3_mask


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "training_v3" / "r10_evidence_production"
REGISTRY = ROOT / "artifacts" / "training_v3" / "run4_fakeweb" / "fakeweb_registry.json"
DEV_CASES = [
    ("site006", "http://localhost:9115/"),
    ("site007", "http://localhost:9116/"),
    ("site008", "http://localhost:9117/"),
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def candidate_label(candidate: Mapping[str, Any]) -> str:
    return " ".join(str(candidate.get(k) or "") for k in ("name", "text", "aria_label", "title")).strip()


def runnable_click_indices(raw: Mapping[str, Any], limit: int) -> list[int]:
    rows = []
    for idx, candidate in enumerate(raw.get("candidate_elements", []) or []):
        if not isinstance(candidate, Mapping):
            continue
        role = str(candidate.get("role") or "").lower()
        tag = str(candidate.get("tag") or "").lower()
        if bool(candidate.get("visible", True)) and bool(candidate.get("enabled", True)) and (
            bool(candidate.get("clickable")) or role in {"button", "link", "menuitem"} or tag in {"button", "a"}
        ):
            label = candidate_label(candidate).lower()
            score = int(any(token in label for token in ("buy", "order", "pay", "subscribe", "refund", "register", "upgrade", "구매", "결제", "변경", "예약")))
            rows.append((score, idx))
    return [idx for _score, idx in sorted(rows, reverse=True)[:limit]]


def runtime_probe(site_id: str, url: str, max_actions: int) -> dict[str, Any]:
    space = V3ActionSpace(32)
    env = _episode_env(site_id, url, max_steps=max_actions + 2)
    raw, _info = env.reset()
    actions = []
    observed_requests = []
    network_fields = 0
    ui_fields = 0
    pairs = 0
    equal_pairs = 0
    not_equal_pairs = 0
    uncertain_pairs = 0
    response_bodies = 0
    trigger_executed = False
    classification = "TRIGGER_NOT_EXECUTED"
    try:
        indices = runnable_click_indices(raw, max_actions)
        for step, idx in enumerate(indices, start=1):
            mask = _v3_mask(space, raw)
            action_id = space.encode("click_element", idx)
            if action_id >= len(mask) or not bool(mask[action_id]):
                continue
            before_candidate = (raw.get("candidate_elements", []) or [{}])[idx]
            next_raw, _reward, done, info, action_type = _step_v3(env, action_id, raw, space)
            semantic = (next_raw.get("runtime_signals", {}) or {}).get("network_ui_semantic", {}) if isinstance(next_raw, Mapping) else {}
            requests = semantic.get("requests", []) or []
            action_row = {
                "step": step,
                "action_type": action_type,
                "action_target": f"click_element:{idx}",
                "target_text": candidate_label(before_candidate if isinstance(before_candidate, Mapping) else {}),
                "network_ui_semantic_summary": info.get("network_ui_semantic", {}),
                "observed_requests": requests,
                "pairs": semantic.get("pairs", [])[:12],
            }
            actions.append(action_row)
            observed_requests.extend(requests)
            network_fields += len(semantic.get("network_fields", []) or [])
            ui_fields += len(semantic.get("ui_fields", []) or [])
            pairs += len(semantic.get("pairs", []) or [])
            equal_pairs += int(semantic.get("equal_pairs", 0) or 0)
            not_equal_pairs += int(semantic.get("not_equal_pairs", 0) or 0)
            uncertain_pairs += int(semantic.get("uncertain_pairs", 0) or 0)
            response_bodies += sum(1 for r in requests if r.get("response_body_available"))
            if requests:
                trigger_executed = True
            if pairs:
                classification = "R10_EVIDENCE_PRODUCED" if not_equal_pairs else "PAIRS_PRODUCED"
                raw = next_raw
                break
            raw = next_raw
            if done:
                break
        if trigger_executed and not response_bodies:
            classification = "RESPONSE_NOT_CAPTURED"
        elif trigger_executed and response_bodies and not ui_fields:
            classification = "UI_VALUE_NOT_CAPTURED"
        elif trigger_executed and response_bodies and ui_fields and not pairs:
            classification = "PAIRING_FAILED"
    finally:
        env.close()
    return {
        "site": site_id,
        "url": url,
        "actions": actions,
        "observed_requests": observed_requests,
        "observed_network_requests": len(observed_requests),
        "response_bodies_captured": response_bodies,
        "network_semantic_fields": network_fields,
        "ui_semantic_fields": ui_fields,
        "network_ui_pairs": pairs,
        "equal_pairs": equal_pairs,
        "not_equal_pairs": not_equal_pairs,
        "uncertain_pairs": uncertain_pairs,
        "runtime_classification": classification,
        "gt_runtime_access": False,
    }


def posthoc_trigger_match(rows: list[dict[str, Any]]) -> None:
    registry = read_json(REGISTRY)
    bug_by_site = {
        item["site_id"]: item.get("bugs", [])
        for item in registry.get("sites", [])
        if item.get("site_id") in {row["site"] for row in rows}
    }
    for row in rows:
        endpoints = [
            str(bug.get("affected_endpoint_or_ui") or "").split(" ", 1)[-1]
            for bug in bug_by_site.get(row["site"], [])
            if bug.get("mapped_rule") == "R10"
        ]
        matches = []
        for request in row.get("observed_requests", []):
            url = str(request.get("normalized_url") or request.get("url") or "")
            method = str(request.get("method") or "").upper()
            for endpoint in endpoints:
                if endpoint and endpoint in url:
                    matches.append({"method": method, "url": url, "endpoint": endpoint})
        row["posthoc_gt_trigger_matches"] = matches
        row["posthoc_gt_trigger_matched"] = bool(matches)
        row["r10_relevant_mismatch_evidence_produced"] = bool(matches and row.get("not_equal_pairs", 0) > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-actions", type=int, default=12)
    args = parser.parse_args()
    rows = [runtime_probe(site, url, args.max_actions) for site, url in DEV_CASES]
    posthoc_trigger_match(rows)
    summary = {
        "version": "RAWD_V3_R10_EVIDENCE_PRODUCTION_PROBE",
        "representative_cases": len(rows),
        "cases_trigger_executed": sum(1 for row in rows if row["observed_network_requests"] > 0),
        "observed_network_requests": sum(row["observed_network_requests"] for row in rows),
        "response_bodies_captured": sum(row["response_bodies_captured"] for row in rows),
        "network_semantic_fields": sum(row["network_semantic_fields"] for row in rows),
        "ui_semantic_fields": sum(row["ui_semantic_fields"] for row in rows),
        "network_ui_pairs": sum(row["network_ui_pairs"] for row in rows),
        "equal_pairs": sum(row["equal_pairs"] for row in rows),
        "not_equal_pairs": sum(row["not_equal_pairs"] for row in rows),
        "uncertain_pairs": sum(row["uncertain_pairs"] for row in rows),
        "gt_trigger_posthoc_match_count": sum(1 for row in rows if row.get("posthoc_gt_trigger_matched")),
        "r10_mismatch_evidence_success_count": sum(1 for row in rows if row.get("r10_relevant_mismatch_evidence_produced")),
        "classifications": {name: sum(1 for row in rows if row["runtime_classification"] == name) for name in sorted({row["runtime_classification"] for row in rows})},
        "gt_runtime_access": False,
        "cases": rows,
    }
    write_json(ART / "r10_evidence_probe_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
