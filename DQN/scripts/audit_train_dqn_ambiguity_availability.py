from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
OUT = BASE / "dqn_train_ambiguity_availability_audit.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    split = load_json(ROOT / "configs/step7_full_site_split.json")
    sites = load_json(ROOT / "configs/rl_errorsite_full_training_sites.json")["sites"]
    site_by_id = {str(x["site_id"]): x for x in sites}
    train_ids = [str(x) for x in split["train"]]
    manifest = load_json(BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json")

    # This is an offline source-name audit only. It is never supplied to policy,
    # detector, reward, router, or DQN training.
    source_patterns = {
        "interaction_or_navigation": r"button|click|navigation|route|search|filter|sort|form|input|field|checkout|cart|purchase|login|redirect",
        "state_or_persistence": r"state|persistence|rollback|cache|session|schema|migration|consistency|integrity|duplicate|stale|source",
        "network_or_temporal": r"network|api|async|timeout|retry|latency|request|response|lock|replication|WAL|queue",
        "layout_or_rendering": r"layout|render|overlap|overflow|scroll|visual|font|color|accessibility",
    }
    source_rows = []
    source_counts = Counter()
    for sid in train_ids:
        raw_path = str(site_by_id[sid].get("source_path", ""))
        name = Path(raw_path).name
        categories = [k for k, pat in source_patterns.items() if re.search(pat, name, re.I)]
        for category in categories:
            source_counts[category] += 1
        source_rows.append({"site_id": sid, "source_path": raw_path, "basename": name, "matched_categories": categories})

    # Historical transition logs are explicitly marked historical and are not
    # treated as current Frozen-PPO runtime training samples.
    historical_rule_counts = Counter()
    historical_action_counts = Counter()
    historical_sites_with_rule = {rule: set() for rule in ("R01", "R02", "R09", "R10", "R12")}
    historical_files = 0
    for sid in train_ids:
        log_path = ROOT / f"DQN/artifacts/browsergym/{sid}/rl_transition_log.jsonl"
        if not log_path.exists():
            continue
        historical_files += 1
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            action_type = str(row.get("action_type") or row.get("action", {}).get("action_type") or "")
            historical_action_counts[action_type] += 1
            for anomaly in row.get("anomalies", []) or []:
                rule = str(anomaly.get("rule_id") or "")
                if rule:
                    historical_rule_counts[rule] += 1
                    if rule in historical_sites_with_rule:
                        historical_sites_with_rule[rule].add(sid)

    current_counts = manifest.get("counts", {})
    current_strategies = manifest.get("strategy_counts", {})
    current_anomalies = manifest.get("anomaly_types", {})
    result = {
        "schema_version": "rawd_step7_dqn_train_ambiguity_availability_audit_v1",
        "scope": {
            "train_sites": train_ids,
            "train_site_count": len(train_ids),
            "validation_used": False,
            "test_used": False,
            "gt_policy_input": False,
            "purpose": "offline availability audit only",
        },
        "current_frozen_ppo_collection": {
            "source_manifest": str(BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json"),
            "runtime_steps": manifest.get("funnel", {}).get("ppo_runtime_steps"),
            "interaction_actions": sum(int(manifest.get("action_types", {}).get(k, 0)) for k in ("click_element", "fill_input", "select_option", "press_enter")),
            "detector_candidates": current_counts.get("candidates"),
            "anomaly_types": current_anomalies,
            "strategy_counts": current_strategies,
            "dqn_local": current_counts.get("dqn_local"),
            "multi_action_states": current_counts.get("multi_action"),
            "partial": current_counts.get("partial"),
            "successful": current_counts.get("successful"),
            "classification": "NO_CURRENT_DQN_LOCAL_OPPORTUNITY_OBSERVED",
        },
        "offline_source_composition": {
            "method": "source_path_basename_keyword_audit",
            "counts": dict(source_counts),
            "rows": source_rows,
            "interpretation": "availability signal only; not ground-truth policy input and not runtime success evidence",
        },
        "historical_runtime_evidence": {
            "method": "pre-existing rl_transition_log.jsonl scan",
            "files_found": historical_files,
            "rule_counts": dict(historical_rule_counts),
            "action_counts": dict(historical_action_counts),
            "sites_by_rule": {k: sorted(v) for k, v in historical_sites_with_rule.items()},
            "provenance": "historical_not_current_frozen_ppo_collection",
        },
        "conclusion": {
            "natural_source_opportunities_indicated": any(source_counts.values()),
            "natural_current_runtime_dqn_opportunity_observed": False,
            "current_dqn_training_ready": False,
            "primary_gap": "current detector/runtime path emits layout-only OBSERVATION_RECHECK candidates; no ambiguous replay/state candidates reach DQN_LOCAL",
            "not_proven": [
                "that the source defects are reachable within the current 15-step rollout",
                "that historical R01/R09/R10/R12 events are reproducible under the new frozen PPO",
                "that target mapping or action mask is defective, because no current ACTION_REPLAY/SEQUENCE_REPLAY/STATE_REVISIT candidate exists",
            ],
            "safe_next_action": "Perform a diagnostic-only replayability audit on TRAIN sites with historical interaction/state evidence, preserving current split and excluding GT from policy input; only after a naturally observed DQN_LOCAL chain may DQN training begin.",
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "source_counts": dict(source_counts),
        "historical_rule_counts": dict(historical_rule_counts),
        "current": result["current_frozen_ppo_collection"],
        "out": str(OUT),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
