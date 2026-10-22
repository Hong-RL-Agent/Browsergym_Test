from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/external_b05_sequence_probe.json"
dst = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/external_dqn_scenario_probe_final.json"
d = json.loads(src.read_text(encoding="utf-8"))
allowed_dqn_rules = {"R01", "R02", "R09", "R10", "R12"}
all_routes = [r for row in d.get("steps", []) for r in row.get("routes", [])]
canonical_routes = [r for r in all_routes if str(r.get("rule")) in allowed_dqn_rules and r.get("dqn_eligible")]
misrouted = [r for r in all_routes if str(r.get("rule")) not in allowed_dqn_rules and r.get("dqn_eligible")]
multi_action_routes = [r for r in canonical_routes if int(r.get("valid_local_actions", 0) or 0) >= 2]
final_state = d.get("final_state") or {}
filters = final_state.get("filters") or {}
last_search = final_state.get("lastSearch") or {}
precondition_reached = bool(last_search.get("brands") and not filters.get("brands") and final_state.get("compare", {}).get("summary", {}).get("count") == 2)
out = {
    "diagnostic_only": True,
    "source_artifact": str(src),
    "site": "B05_상품검색비교",
    "workflow": ["category_laptop", "brand_filter_1", "brand_filter_2", "product_checkbox_1", "product_checkbox_2", "open_compare", "return_search"],
    "precondition_reached": precondition_reached,
    "last_search": last_search,
    "current_filters_after_return": filters,
    "comparison_count": final_state.get("compare", {}).get("summary", {}).get("count"),
    "detector_counts": {k: v for k, v in d.get("counts", {}).items() if k != "dqn_local"},
    "raw_dqn_local_routes": int(d.get("dqn_local", 0)),
    "canonical_dqn_local_routes": len(canonical_routes),
    "misrouted_dqn_routes": len(misrouted),
    "misrouted_rules": sorted({str(r.get("rule")) for r in misrouted}),
    "valid_local_actions_ge_2": bool(multi_action_routes),
    "multi_action_state_count": len(multi_action_routes),
    "runtime_partial_or_successful": int(d.get("counts", {}).get("runtime_partial", 0) or 0) + int(d.get("counts", {}).get("runtime_successful", 0) or 0) > 0,
    "gt_leakage": 0,
    "readiness": "FAIL",
    "readiness_reasons": [
        "NO_CANONICAL_DQN_LOCAL_ROUTE" if not canonical_routes else "",
        "NO_MULTI_ACTION_LOCAL_STATE" if not multi_action_routes else "",
        "NO_LOCAL_ACTION_EXECUTION" if not any("local_execution" in row for row in d.get("steps", [])) else "",
        "NO_RUNTIME_PARTIAL_OR_SUCCESSFUL_REPRODUCTION" if not (int(d.get("counts", {}).get("runtime_partial", 0) or 0) + int(d.get("counts", {}).get("runtime_successful", 0) or 0)) else "",
    ],
    "next_diagnostic": "repair_or_extend_detector_evidence_adapter_for_B05_sequence_state_then_rerun_without_promoting_this_probe_to_training_data",
}
out["readiness_reasons"] = [x for x in out["readiness_reasons"] if x]
dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
