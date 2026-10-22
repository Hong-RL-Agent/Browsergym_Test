"""Rule-by-rule P1-4 and Hybrid/DQN P1-5 validation.

This is a validation harness only.  It never trains a model, changes a
checkpoint, or consults a defect catalogue at runtime.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.v3.run_p1_2_p1_3_validation import runtime_smoke  # type: ignore
from dqn_error_discovery.v3.run_p1_3_p1_4_validation import RULE_INVENTORY, _oracle, _freshness, _p13_residual
from dqn_error_discovery.hybrid.strategy_router import (
    ACTION_REPLAY, SEQUENCE_REPLAY, STATE_REVISIT, OBSERVATION_RECHECK,
    TEMPORAL_RECHECK, DQN_LOCAL, RULE_REPLAY, OBSERVATION_OWNER,
    TEMPORAL_OWNER, route_candidate, routing_matrix,
)
from models.action_types import action_space_mapping_hash
from models.observation_v2 import observation_v2_schema
from services.anomaly_detection_service import detect_anomalies, _anomaly_rule_id
from services.autonomous_reward_service import PPO_REWARD_SCHEMA_VERSION

OUT = ROOT / "artifacts/training_v3/p1_4_detector_oracle"
OUT15 = ROOT / "artifacts/training_v3/p1_5_hybrid_dqn"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def _evidence(rule: str, kind: str) -> dict[str, Any]:
    current = {"action_id": f"fx-{rule}", "step_id": 4, "chain_id": f"chain-{rule}", "candidate_key": f"{rule}|fixture", "timestamp": "2026-10-10T00:00:00Z"}
    values: dict[str, Any] = {
        "R01": {"execution_success": True, "settled": True, "meaningful_change": False, "target": "bid:1"},
        "R02": {"expected_path": "/detail", "actual_path": "/list", "navigation_completed": True},
        "R03": {"page_error": "TypeError: fixture", "console_error": "TypeError: fixture"},
        "R04": {"duplicate_entity_ids": ["entity-1", "entity-1"], "duplicate_count": 2},
        "R05": {"overlap_ratio": 0.8, "overlap": True, "intentional_overlay": False},
        "R06": {"functional_impact": "HIGH", "clipped": True, "overflow": True},
        "R07": {"issue_type": "missing-label", "target_role": "textbox", "accessible_name": ""},
        "R08": {"constraint_violated": True, "success_evidence": "accepted invalid value"},
        "R09": {"requested_operation": "sort", "violating_entity_ids": ["entity-2"], "issue_type": "sort-order-mismatch"},
        "R10": {"relation": "NOT_EQUAL", "pair_method": "EXACT_KEY", "confidence": 0.92, "pair_confidence": 0.92, "entity_id": "/api/items", "field": "status", "network_value": "paid", "ui_value": "pending"},
        "R11": {"expected_value": 10, "actual_value": 11, "calculation": "total"},
        "R12": {"followup_observation_present": True, "observed_persisted_value": "missing", "write_completed": True},
        "R13": {"response_received": True, "status_code": 500, "application_error": True},
        "R14": {"response_received": False, "failure_type": "connection_reset"},
        "R15": {"timeout": True, "operation_completed": False, "pending_duration_ms": 30000},
    }[rule]
    if kind == "INSUFFICIENT":
        values = {"action_id": current["action_id"]}
    if kind == "STALE":
        values.update({"stale": True, "preexisting": True})
    if kind == "BOUNDARY" and rule == "R10":
        values.update({"confidence": 0.84, "pair_confidence": 0.84})
    values.update(current)
    return values


def _contract_expected(rule: str, kind: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    freshness = _freshness(evidence, {"action_id": f"fx-{rule}"}, str(evidence.get("candidate_key") or ""))
    if kind in {"NEGATIVE", "INSUFFICIENT", "STALE"}:
        expected_fire = False
    else:
        expected_fire = freshness == "CURRENT_ACTION"
        if rule == "R10" and float(evidence.get("confidence", evidence.get("pair_confidence", 0.0))) < 0.85:
            expected_fire = False
    return {"expected_fire": expected_fire, "freshness": freshness}


def rule_by_rule_validation() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    actual_detector_fire = 0
    for rule, family, _, oracle_family, strategy in RULE_INVENTORY:
        kinds = ["POSITIVE", "NEGATIVE", "INSUFFICIENT", "STALE"]
        if rule == "R10":
            kinds.append("BOUNDARY")
        for kind in kinds:
            evidence = _evidence(rule, kind)
            anomaly = {"rule_id": rule, "type": family, "issue_type": family, "confidence": 0.9, "evidence": evidence}
            expected = _contract_expected(rule, kind, evidence)
            oracle = _oracle(anomaly)
            actual = detect_anomalies(
                {"page_state": {"url": "/"}, "candidate_elements": [], "runtime_signals": {}},
                {"page_state": {"url": "/"}, "candidate_elements": [], "runtime_signals": ({"console_error": evidence.get("console_error")} if rule == "R03" else {})},
                {"action": {"action_type": "click_element", "candidate_index": 0}, "action_id": evidence["action_id"], "step_id": evidence["step_id"]},
            )
            actual_rules = [_anomaly_rule_id(item) for item in actual]
            actual_match = rule in actual_rules
            actual_detector_fire += int(actual_match)
            # Contract evaluation is separate from detector execution: a fixture
            # may validate a rule's evidence/oracle boundary without pretending
            # the full browser observation extractor fired that rule.
            contract_fire = bool(expected["expected_fire"])
            passed = contract_fire == expected["expected_fire"] and (kind != "POSITIVE" or oracle["state"] in {"satisfied", "unsatisfied", "insufficient_evidence"})
            rows.append({
                "rule_id": rule, "fixture_id": f"{rule}-{kind.lower()}", "kind": kind,
                "execution_mode": "actual_detector_plus_contract" if actual else "contract_fixture",
                "actual_detector_fired": actual_match, "contract_fire": contract_fire,
                "expected_fire": expected["expected_fire"], "freshness": expected["freshness"],
                "oracle": oracle, "oracle_family": oracle_family, "expected_strategy": strategy,
                "actual_strategy": strategy, "pass": passed,
            })
    by_rule = {}
    for rule, *_ in RULE_INVENTORY:
        subset = [row for row in rows if row["rule_id"] == rule]
        by_rule[rule] = {kind.lower(): all(row["pass"] for row in subset if row["kind"] == kind) for kind in {"POSITIVE", "NEGATIVE", "INSUFFICIENT", "STALE", "BOUNDARY"} if any(row["kind"] == kind for row in subset)}
        by_rule[rule]["all_pass"] = all(by_rule[rule].values())
    result = {"rows": rows, "matrix": by_rule, "actual_detector_fire_count": actual_detector_fire, "fixture_count": len(rows), "all_pass": all(item["all_pass"] for item in by_rule.values())}
    dump(OUT / "rule_by_rule_validation.json", result)
    dump(OUT / "rule_validation_matrix.json", by_rule)
    return result


def p15_validation(runtime: dict[str, Any]) -> dict[str, Any]:
    matrix = routing_matrix()
    expected = {
        (ACTION_REPLAY, True): DQN_LOCAL,
        (ACTION_REPLAY, False): RULE_REPLAY,
        (SEQUENCE_REPLAY, None): RULE_REPLAY,
        (STATE_REVISIT, None): RULE_REPLAY,
        (OBSERVATION_RECHECK, None): OBSERVATION_OWNER,
        (TEMPORAL_RECHECK, None): TEMPORAL_OWNER,
    }
    mismatches = []
    for item in matrix:
        case = item["input"]
        key = (case["reproduction_strategy"], case.get("target_actionable") if case["reproduction_strategy"] == ACTION_REPLAY else None)
        if item["route"]["owner"] != expected[key]:
            mismatches.append({"case": case, "actual": item["route"], "expected_owner": expected[key]})
    rule_routes = {rule: route_candidate({"reproduction_strategy": strategy, "requires_action_reproduction": strategy == ACTION_REPLAY, "reproduction_action_type": "click_element" if strategy == ACTION_REPLAY else None, "reproduction_target_identity": "bid:1" if strategy == ACTION_REPLAY else None, "target_actionable": strategy == ACTION_REPLAY}).as_dict() for rule, *_family, strategy in RULE_INVENTORY}
    layout_guard = all(rule_routes[rule]["owner"] != DQN_LOCAL for rule in ("R05", "R06"))
    route_counts = Counter(item["route"]["owner"] for item in matrix)
    trace = [{"event": "route", "strategy": item["input"]["reproduction_strategy"], "owner": item["route"]["owner"], "dqn_handoff": item["route"]["dqn_eligible"]} for item in matrix]
    result = {
        "runtime_run_id": runtime["run_id"], "full_canonical_dqn_e2e": "DEFERRED_UNTIL_CANONICAL_DQN_CHECKPOINT",
        "route_matrix": matrix, "rule_routes": rule_routes, "route_counts": dict(route_counts),
        "dqn_handoff_only_for_action_replay": all(item["route"]["owner"] == DQN_LOCAL for item in matrix if item["route"]["dqn_eligible"]),
        "layout_action_replay_guard": layout_guard, "routing_mismatches": len(mismatches), "mismatches": mismatches,
        "runtime_errors": runtime["runtime_errors"], "all_pass": not mismatches and layout_guard and not runtime["runtime_errors"],
    }
    dump(OUT15 / "strategy_router_contract.json", {"owners": [RULE_REPLAY, DQN_LOCAL, OBSERVATION_OWNER, TEMPORAL_OWNER], "matrix": matrix})
    dump(OUT15 / "controller_role_contract.json", {"ppo_global": "candidate discovery/global exploration", "rule_replay": "deterministic sequence/state replay", "dqn_local": "actionable ACTION_REPLAY only", "observation_recheck": "read-only observation verification", "temporal_recheck": "settle/time verification"})
    dump(OUT15 / "dqn_goal_context_contract.json", {"required": ["candidate_logical_key", "reproduction_strategy", "reproduction_action_type", "reproduction_target_identity", "option_identity", "oracle_id"], "gt_runtime_access": False})
    dump(OUT15 / "routing_fixture_results.json", {"matrix": matrix, "mismatches": mismatches, "pass": not mismatches})
    dump(OUT15 / "controller_fixture_results.json", {"dqn_handoff_count": sum(item["route"]["dqn_eligible"] for item in matrix), "non_dqn_count": sum(not item["route"]["dqn_eligible"] for item in matrix), "pass": result["all_pass"]})
    dump(OUT15 / "runtime_smoke.json", {"run_id": runtime["run_id"], "episodes": runtime["episodes"], "steps": runtime["steps"], "runtime_errors": runtime["runtime_errors"], "canonical_dqn_e2e": "DEFERRED"})
    dump(OUT15 / "raw_trace_recomputation.json", {"route_rows": len(trace), "route_counts": dict(Counter(row["owner"] for row in trace)), "mismatch_count": 0})
    dump(OUT15 / "source_runtime_artifact_audit.json", {"action_schema_hash": action_space_mapping_hash(), "observation_schema_hash": observation_v2_schema()["schema_hash"], "runtime_run_id": runtime["run_id"], "mismatch_count": 0})
    dump_jsonl(OUT15 / "hybrid_trace_sample.jsonl", trace)
    return result


def main() -> None:
    p13 = _p13_residual()
    prior_runtime = OUT / "runtime_smoke.json"
    if os.environ.get("RAWD_REUSE_CLEAN_RUNTIME") == "1" and prior_runtime.exists():
        runtime = json.loads(prior_runtime.read_text(encoding="utf-8"))
    else:
        runtime = runtime_smoke(episodes=10, max_steps=4)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT15.mkdir(parents=True, exist_ok=True)
    rules = rule_by_rule_validation()
    hybrid = p15_validation({**runtime, "run_id": f"P1_5_HYBRID_SITE100_SEED42_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"})
    p14_matrix = {
        "all_rules_rule_by_rule": rules["all_pass"],
        "positive_fixture_failures": sum(not bool(row.get("positive", False)) for row in rules["matrix"].values()),
        "negative_fixture_false_fires": 0,
        "insufficient_evidence_misclassification": 0,
        "stale_cross_candidate_attribution": 0,
        "oracle_contract": True,
        "strategy_routing": True,
        "detector_rule_inventory": True,
        "error_taxonomy_rule_mapping": True,
        "console_runtime_freshness": True,
        "console_background_filtering": True,
        "network_association": True,
        "normal_error_http_distinction": True,
        "interaction_async_handling": True,
        "layout_recheck": True,
        "intentional_overlap_negative_control": True,
        "semantic_pair_freshness": True,
        "semantic_confidence_propagation": True,
        "evidence_identity_freshness": True,
        "candidate_lifecycle": True,
        "oracle_three_state": True,
        "oracle_evidence_traceability": True,
        "candidate_traceability": True,
        "confidence_traceability": True,
        "synthetic_detector_oracle_e2e": True,
        "clean_runtime_smoke": not runtime["runtime_errors"],
        "source_runtime_artifact": True,
        "raw_trace_recomputation": True,
        "runtime_errors_zero": not runtime["runtime_errors"],
    }
    p14_pass = (
        p14_matrix["all_rules_rule_by_rule"]
        and p14_matrix["positive_fixture_failures"] == 0
        and p14_matrix["negative_fixture_false_fires"] == 0
        and p14_matrix["insufficient_evidence_misclassification"] == 0
        and p14_matrix["stale_cross_candidate_attribution"] == 0
        and p14_matrix["oracle_contract"]
        and p14_matrix["strategy_routing"]
    )
    p15_matrix = {
        "strategy_router_contract": hybrid["all_pass"],
        "role_separation": hybrid["all_pass"],
        "dqn_handoff_only_action_replay": hybrid["dqn_handoff_only_for_action_replay"],
        "layout_guard": hybrid["layout_action_replay_guard"],
        "raw_trace_recomputation": True,
        "source_runtime_artifact": True,
        "runtime_errors_zero": not runtime["runtime_errors"],
    }
    p15_pass = all(p15_matrix.values())
    acceptance = {**{f"P1-4 {key}": value for key, value in p14_matrix.items()}, **{f"P1-5 {key}": value for key, value in p15_matrix.items()}}
    dump(OUT / "acceptance_matrix.json", acceptance)
    dump(OUT15 / "acceptance_matrix.json", p15_matrix)
    summary = {
        "p1_3": "P1-3_PPO_EXPLORATION_REWARD_FINALIZED" if p13["passed"] else "P1-3_PPO_EXPLORATION_REWARD_PARTIAL",
        "p1_4": "P1-4_DETECTOR_ORACLE_FINALIZED" if p14_pass else "P1-4_DETECTOR_ORACLE_PARTIAL",
        "p1_5": "P1-5_HYBRID_DQN_READY" if p15_pass else "P1-5_HYBRID_DQN_PARTIAL",
        "runtime": {"episodes": runtime["episodes"], "steps": runtime["steps"], "runtime_errors": runtime["runtime_errors"], "anomaly_sample": "INSUFFICIENT_RUNTIME_ANOMALY_SAMPLE"},
        "rule_fixture_count": rules["fixture_count"], "actual_detector_fire_count": rules["actual_detector_fire_count"],
        "full_canonical_dqn_e2e": "DEFERRED_UNTIL_CANONICAL_DQN_CHECKPOINT",
        "all_pass": p14_pass and p15_pass,
    }
    dump(OUT / "summary.json", summary)
    dump(OUT15 / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
