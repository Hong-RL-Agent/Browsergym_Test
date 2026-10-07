"""Post-hoc evaluator for the frozen unseen-site experiment.

This module is intentionally separate from the runtime runner.  It is the only
place in the final-evaluation pipeline that reads BUGS.md or uses bug selectors.
Runtime artifacts are treated as opaque evidence and are never fed back into
the agent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from envs.browsergym_jaws_env import BrowserGymJAWSEnv, _active_page


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "artifacts" / "final_unseen_eval"
SITES = {
    "SITE002": ("http://localhost:9221/", Path(r"C:\workspace\RL_Errorsite\RL_Errorsite\site002\BUGS.md")),
    "SITE003": ("http://localhost:9222/", Path(r"C:\workspace\RL_Errorsite\RL_Errorsite\site003\BUGS.md")),
    "SITE004": ("http://localhost:9223/", Path(r"C:\workspace\RL_Errorsite\RL_Errorsite\site004\BUGS.md")),
    "SITE005": ("http://localhost:9224/", Path(r"C:\workspace\RL_Errorsite\backend-errorsite\site005\BUGS.md")),
}
SEEDS = (42, 43, 44)
METHODS = ("ppo", "hybrid")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bugs(path: Path, site: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    ids = list(re.finditer(r"(?:\*\*bugId\*\*|##\s*)(?:\s*(?:[:|])\s*)?([A-Za-z0-9_-]+-bug\d+)", text, re.I))
    bugs: list[dict[str, Any]] = []
    for index, match in enumerate(ids):
        start = match.start()
        end = ids[index + 1].start() if index + 1 < len(ids) else len(text)
        block = text[start:end]
        bug_id = match.group(1)
        type_match = re.search(r"(?:\*\*type\*\*|- type)\s*[:|]?\s*([^|\r\n]+)", block, re.I)
        selector_match = re.search(r"data-bug-id[^`|]*`([^`]+)`", block, re.I)
        endpoint_match = re.search(r"(?:API endpoint|endpoint)\s*[:|]?\s*`?([^`\r\n|]+)", block, re.I)
        bugs.append({
            "bug_id": bug_id,
            "site_id": site,
            "type": (type_match.group(1).strip() if type_match else "unknown"),
            "selector": selector_match.group(1).strip() if selector_match else f'[data-bug-id="{bug_id}"]',
            "endpoint": endpoint_match.group(1).strip() if endpoint_match else "",
            "source": str(path),
        })
    return bugs


def _probe_selector_bids() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for site, (url, _) in SITES.items():
        env = BrowserGymJAWSEnv(site_id=site.lower(), base_url=url, max_steps=1, headless=True)
        env.reset()
        page = _active_page(env.env)
        try:
                rows = page.locator("[data-bug-id]").evaluate_all(
                    """els => els.map(e => ({
                        bug: e.getAttribute('data-bug-id'),
                        bid: e.getAttribute('bid'),
                        parentBid: e.parentElement?.closest?.('[bid]')?.getAttribute('bid') || ''
                    }))"""
                )
                for row in rows:
                    key = f"{site}-{row.get('bug', '').split('-bug')[-1]}"
                    result.setdefault(key, {"bids": [], "ancestor_bids": []})
                    if row.get("bid"):
                        result[key]["bids"].append(str(row["bid"]))
                    if row.get("parentBid"):
                        result[key]["ancestor_bids"].append(str(row["parentBid"]))
        finally:
            env.close()
    return result


def _candidate_keys(data: dict[str, Any]) -> set[str]:
    keys = set(data.get("candidate_attribution", {}).get("candidate_first_agent", {}))
    for row in data.get("timeline", []):
        keys.update(str(key) for key in row.get("new_candidate_keys", []) if key)
    return keys


def _confirmed_keys(data: dict[str, Any]) -> set[str]:
    return {
        str(row.get("logical_key"))
        for row in data.get("confirmation_attribution", {}).get("confirmation_records", [])
        if row.get("logical_key")
    }


def _compact(value: Any) -> Any:
    """Keep scalar summary information without retaining giant raw timelines."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return {"count": len(value)}
    if isinstance(value, dict):
        return {str(key): _compact(item) for key, item in value.items() if not isinstance(item, (list, dict)) or len(item) < 50}
    return str(value)


def _rule(key: str) -> str:
    return str(key).split("|", 1)[0]


def _target_ids(key: str) -> set[str]:
    found = set(re.findall(r"target:(-?\d+)", key))
    parts = key.split("|")
    if parts and parts[0] in {"R01", "R02", "R03", "R04"} and len(parts) >= 5:
        if re.fullmatch(r"-?\d+", parts[4]):
            found.add(parts[4])
    if parts and parts[0] in {"R05", "R06", "R07", "R08", "R09"} and len(parts) >= 4:
        if re.fullmatch(r"-?\d+", parts[3]):
            found.add(parts[3])
    return found


def _expected_rules(bug_type: str, endpoint: str) -> set[str]:
    kind = bug_type.lower()
    if "button" in kind or "no-response" in kind:
        return {"R01"}
    if "css" in kind or "layout" in kind or "dropdown" in kind:
        return {"R05", "R06"}
    if endpoint:
        return {"R13", "R14", "R15"}
    return set()


def _is_reached(data: dict[str, Any], bids: set[str]) -> bool:
    if not bids:
        return False
    for row in data.get("timeline", []):
        action_target = str(row.get("action_target") or "")
        if any(re.search(rf"(?:^|:)bid:{re.escape(bid)}(?:$|\b)", action_target) or action_target.endswith(f":{bid}") for bid in bids):
            return True
        state = f"{row.get('state_before', '')} {row.get('state_after', '')}"
        if any(f"bid:{bid}" in state or re.search(rf"\b{re.escape(bid)}\b", state) for bid in bids):
            # State evidence alone is weak; retain it only as reach evidence.
            return True
    return False


def _match_bug(bug: dict[str, Any], data: dict[str, Any], bid_info: dict[str, Any]) -> dict[str, Any]:
    bug_key = f"{bug['site_id']}-{bug['bug_id'].split('-bug')[-1]}"
    bids = set(bid_info.get("bids", [])) | set(bid_info.get("ancestor_bids", []))
    expected = _expected_rules(bug["type"], bug["endpoint"])
    candidates = _candidate_keys(data)
    confirmed = _confirmed_keys(data)
    matching = []
    for key in candidates:
        if expected and _rule(key) not in expected:
            continue
        if bids and _target_ids(key) & bids:
            matching.append(key)
    matching_confirmed = [key for key in confirmed if (not expected or _rule(key) in expected) and (not bids or _target_ids(key) & bids)]
    reached = _is_reached(data, bids)
    if matching:
        confidence = "HIGH" if _target_ids(matching[0]) & bids and expected else "MEDIUM"
        outcome = "CONFIRMED" if matching_confirmed else "DETECTED"
    elif reached:
        confidence = "UNMAPPED"
        outcome = "REACHED_NOT_DETECTED"
    else:
        confidence = "UNMAPPED"
        outcome = "NOT_REACHED"
    return {
        "bug_id": bug["bug_id"], "site_id": bug["site_id"], "type": bug["type"],
        "endpoint": bug["endpoint"], "selector": bug["selector"], "target_bids": sorted(bids),
        "expected_rules": sorted(expected), "outcome": outcome, "mapping_confidence": confidence,
        "candidate_keys": sorted(matching), "confirmed_keys": sorted(matching_confirmed),
        "reached": reached,
    }


def _run_summary(site: str, method: str, seed: int, data: dict[str, Any]) -> dict[str, Any]:
    candidates = _candidate_keys(data)
    confirmed = _confirmed_keys(data)
    rules = Counter(_rule(key) for key in candidates if _rule(key))
    temporal = data.get("temporal_evidence", {})
    layout = data.get("interactive_layout_evidence", {})
    network = data.get("framework_network_evidence", {})
    network_classes = Counter()
    network_relevance = Counter()
    for record in network.get("records", []):
        for event in record.get("normalized_network_events", []):
            network_classes[str(event.get("semantic_class") or "UNKNOWN")] += 1
            network_relevance[str(event.get("functional_relevance") or "UNKNOWN")] += 1
    novelty = data.get("functional_region_novelty", {})
    episodes = data.get("episode_summary", {})
    coverage = data.get("ppo_global_coverage", {})
    return {
        "site_id": site, "method": method, "seed": seed,
        "episodes": int(data.get("configuration", {}).get("episodes", 10)),
        "total_steps": episodes.get("total_steps", 0),
        "ppo_steps": episodes.get("ppo_steps", data.get("agent_step_counts", {}).get("PPO_GLOBAL", 0)),
        "dqn_steps": episodes.get("dqn_steps", data.get("agent_step_counts", {}).get("DQN_LOCAL", 0)),
        "handoffs": episodes.get("handoff_count", 0),
        "local_sessions": episodes.get("local_sessions", 0),
        "unique_candidates": len(candidates), "confirmed_candidates": len(confirmed),
        "candidate_rule_counts": dict(rules),
        "unique_states": len(coverage.get("states", [])) if isinstance(coverage.get("states"), list) else coverage.get("unique_states", 0),
        "unique_urls": len(coverage.get("urls", [])) if isinstance(coverage.get("urls"), list) else coverage.get("unique_urls", 0),
        "unique_targets": len(coverage.get("targets", [])) if isinstance(coverage.get("targets"), list) else coverage.get("unique_targets", 0),
        "unique_action_targets": len(coverage.get("action_targets", [])) if isinstance(coverage.get("action_targets"), list) else coverage.get("unique_action_targets", 0),
        "temporal": _compact(temporal),
        "layout": _compact(layout),
        "network": _compact(network),
        "network_semantic_counts": dict(network_classes),
        "network_relevance_counts": dict(network_relevance),
        "functional_region_novelty": _compact(novelty),
        "hard_invariants": _compact(data.get("hard_invariants", {})),
        "diagnostic": _compact(data.get("diagnostic", {})),
        "handoff_reason_counts": dict(Counter(str(row.get("reason") or "") for row in data.get("handoff_events", []))),
        "runtime_ground_truth_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/final_unseen_eval")
    parser.add_argument("--single-run", nargs=3, metavar=("SITE", "METHOD", "SEED"))
    parser.add_argument("--fragment-dir", default="posthoc_fragments")
    parser.add_argument("--selector-bid-map", default="")
    args = parser.parse_args()
    global EVAL_ROOT
    EVAL_ROOT = ROOT / args.output
    if args.single_run:
        site, method, seed_text = args.single_run
        seed = int(seed_text)
        site_bugs = _parse_bugs(SITES[site][1], site)
        bid_map = _read_json(Path(args.selector_bid_map)) if args.selector_bid_map else _probe_selector_bids()
        data = _read_json(EVAL_ROOT / site / f"{method}_seed{seed}.json")
        payload = {
            "run_summary": _run_summary(site, method, seed, data),
            "outcomes": [
                {"method": method, "seed": seed, **_match_bug(bug, data, bid_map.get(f"{site}-{bug['bug_id'].split('-bug')[-1]}", {}))}
                for bug in site_bugs
            ],
        }
        fragment = EVAL_ROOT / args.fragment_dir / f"{site.lower()}_{method}_seed{seed}.json"
        fragment.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    bugs = {site: _parse_bugs(path, site) for site, (_, path) in SITES.items()}
    bid_map = _probe_selector_bids()
    run_summaries: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for site in SITES:
        for method in METHODS:
            for seed in SEEDS:
                path = EVAL_ROOT / site / f"{method}_seed{seed}.json"
                data = _read_json(path)
                run_summaries.append(_run_summary(site, method, seed, data))
                for bug in bugs[site]:
                    key = f"{site}-{bug['bug_id'].split('-bug')[-1]}"
                    outcomes.append({"method": method, "seed": seed, **_match_bug(bug, data, bid_map.get(key, {}))})
                del data
                gc.collect()

    by_run = {(row["site_id"], row["method"], row["seed"]): row for row in run_summaries}
    overall: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in outcomes if row["method"] == method]
        total = len(selected)
        reached = sum(row["outcome"] != "NOT_REACHED" for row in selected)
        detected = sum(row["outcome"] in {"DETECTED", "CONFIRMED"} and row["mapping_confidence"] in {"HIGH", "MEDIUM"} for row in selected)
        confirmed = sum(row["outcome"] == "CONFIRMED" and row["mapping_confidence"] in {"HIGH", "MEDIUM"} for row in selected)
        overall[method] = {"gt_total": total, "gt_reached": reached, "gt_detected": detected, "gt_confirmed": confirmed,
                           "overall_recall": detected / total if total else 0.0,
                           "detection_given_reach": detected / reached if reached else 0.0,
                           "confirmation_given_detection": confirmed / detected if detected else 0.0,
                           "outcome_counts": dict(Counter(row["outcome"] for row in selected))}

    comparison = {"methods": METHODS, "overall": overall, "site_method_seed": run_summaries,
                  "gt_outcomes": outcomes, "selector_bid_map": bid_map,
                  "ground_truth_runtime_isolation": {"runtime_ground_truth_imported": False, "posthoc_only": True},
                  "classification_policy": "HIGH+MEDIUM primary; LOW and UNMAPPED excluded from primary TP"}
    (EVAL_ROOT / "gt_posthoc_evaluation.json").write_text(json.dumps({"bugs": bugs, **comparison}, indent=2, ensure_ascii=False), encoding="utf-8")
    (EVAL_ROOT / "overall_summary.json").write_text(json.dumps({"overall": overall, "run_summaries": run_summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    (EVAL_ROOT / "ppo_vs_hybrid_comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
