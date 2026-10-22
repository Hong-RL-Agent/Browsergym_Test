from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
RULES = ("R01", "R02", "R09", "R10", "R12")


def main() -> None:
    split = json.loads((ROOT / "configs/step7_full_site_split.json").read_text(encoding="utf-8-sig"))
    partition_by = {sid: part for part, ids in (("TRAIN", split["train"]), ("VALIDATION", split["validation"]), ("TEST", split["test"])) for sid in ids}
    part_counts = {part: Counter() for part in ("TRAIN", "VALIDATION", "TEST")}
    by_site = {}
    files_found = 0
    for sid, part in partition_by.items():
        path = ROOT / f"DQN/artifacts/browsergym/{sid}/rl_transition_log.jsonl"
        counts = Counter()
        actions = Counter()
        if path.exists():
            files_found += 1
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
                actions[str(row.get("action_type") or action.get("action_type") or "")] += 1
                for anomaly in row.get("anomalies", []) or []:
                    rule = str(anomaly.get("rule_id") or "")
                    if rule in RULES:
                        counts[rule] += 1
        part_counts[part].update(counts)
        by_site[sid] = {"partition": part, "log_exists": path.exists(), "rule_counts": dict(counts), "interaction_actions": sum(actions[k] for k in ("click_element", "fill_input", "select_option", "press_enter"))}
    current = json.loads((BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "rawd_step7_dqn_opportunity_inventory_v1",
        "historical_log_files_found": files_found,
        "partition_counts": {k: dict(v) for k, v in part_counts.items()},
        "sites_with_relevant_evidence": {part: [sid for sid, row in by_site.items() if row["partition"] == part and row["rule_counts"]] for part in part_counts},
        "by_site": by_site,
        "current_frozen_ppo_collection": {"runtime_steps": current.get("funnel", {}).get("ppo_runtime_steps"), "strategy_counts": current.get("strategy_counts"), "dqn_local": current.get("counts", {}).get("dqn_local"), "multi_action": current.get("counts", {}).get("multi_action"), "partial": current.get("counts", {}).get("partial"), "successful": current.get("counts", {}).get("successful")},
        "decision": {"historical_evidence_is_training_data": False, "test_access_allowed_before_freeze": False, "current_train_runtime_ready": False, "dqn_start_requires": ["current DQN_LOCAL chain", "valid local actions >= 2", "valid target provenance", "runtime PARTIAL or SUCCESSFUL reproduction"], "recommended_path": "approved runtime design or explicitly approved split revision; do not silently move VALIDATION/TEST sites into TRAIN"},
    }
    out = BASE / "dqn_opportunity_inventory_all_splits.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"partition_counts": result["partition_counts"], "sites": result["sites_with_relevant_evidence"], "current": result["current_frozen_ppo_collection"], "out": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
