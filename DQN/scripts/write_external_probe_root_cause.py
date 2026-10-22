from __future__ import annotations
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "DQN/artifacts/training_v3/step7_fresh_training_final/external_dqn_scenario_root_cause.json"
b05 = json.loads((out.parent / "external_dqn_scenario_probe_final.json").read_text(encoding="utf-8"))
b01 = json.loads((out.parent / "external_dqn_scenario_probe.json").read_text(encoding="utf-8"))
result = {
    "diagnostic_only": True,
    "production_dataset_promotion": False,
    "production_ppo_mutated": False,
    "sites": {
        "B01_SNS": {
            "normal_precondition_reached": True,
            "workflow": ["saved_page", "logo_click"],
            "detector_rules": sorted({k for row in b01.get("results", [{}])[0].get("steps", []) for k in row.get("anomalies", []) for k in [str(k.get("rule_id"))]}),
            "dqn_local": 0,
            "finding": "logo click has no href/expected navigation provenance in the observation adapter; current detector did not emit R02/R01",
        },
        "B05_상품검색비교": {
            "normal_precondition_reached": bool(b05.get("precondition_reached")),
            "detector_counts": b05.get("detector_counts"),
            "canonical_dqn_local": b05.get("canonical_dqn_local_routes"),
            "misrouted_dqn": b05.get("misrouted_dqn_routes"),
            "finding": "compare workflow reached and filters reset after return; the repaired runtime-state adapter emitted a fresh R12 sequence-state candidate, and R07 remained observation-owned",
        },
    },
    "readiness": "FAIL",
    "blocking_conditions": [
        "DQN_LOCAL diagnostic executed but Oracle progress was not observed",
        "runtime DQN_LOCAL route is diagnostic-only and not yet a promotable training chain",
        "runtime PARTIAL/SUCCESSFUL reproduction = 0",
    ],
    "next_safe_action": "the measured DQN_LOCAL action executed but Oracle progress was not observed; keep DQN readiness FAIL and obtain an approved runtime scenario/action sequence that yields genuine PARTIAL or SUCCESSFUL reproduction before training-data promotion",
}
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=True))
