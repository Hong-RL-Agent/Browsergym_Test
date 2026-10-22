"""Final diagnostic separating runtime scenario scarcity from routing support.

The fixture portion is structural only and is never promoted to runtime data.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from dqn_error_discovery.hybrid.strategy_router import route_candidate, routing_matrix  # noqa: E402
import services.anomaly_detection_service as anomaly_detection_service  # noqa: E402
from services.anomaly_detection_service import detect_anomalies  # noqa: E402

BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def main() -> None:
    wide = json.loads((BASE / "dqn_wide_train_diagnostic_probe.json").read_text(encoding="utf-8"))
    current = json.loads((BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json").read_text(encoding="utf-8"))
    source = inspect.getsource(anomaly_detection_service)
    relevant_rules = ("R01", "R02", "R09", "R10", "R12")
    detector_source_presence = {rule: rule in source for rule in relevant_rules}

    fixtures = [
        {
            "name": "ambiguous_action_replay",
            "candidate": {
                "reproduction_strategy": "ACTION_REPLAY",
                "requires_action_reproduction": True,
                "reproduction_action_type": "click_element",
                "reproduction_target_identity": "bid:1",
                "target_actionable": True,
                "exact_action_known": False,
            },
        },
        {"name": "ambiguous_sequence_replay", "candidate": {"reproduction_strategy": "SEQUENCE_REPLAY", "sequence_action_required": True, "sequence_recoverable": True}},
        {"name": "ambiguous_state_revisit", "candidate": {"reproduction_strategy": "STATE_REVISIT", "restoration_decision_required": True, "restoration_recoverable": True}},
        {"name": "exact_action_replay", "candidate": {"reproduction_strategy": "ACTION_REPLAY", "requires_action_reproduction": True, "reproduction_action_type": "click_element", "reproduction_target_identity": "bid:1", "target_actionable": True, "exact_action_known": True}},
    ]
    fixture_routes = []
    for item in fixtures:
        route = route_candidate(item["candidate"])
        fixture_routes.append({"name": item["name"], "route": route.as_dict()})

    out = {
        "schema_version": "rawd_step7_final_dqn_scenario_diagnostic_v1",
        "diagnostic_only": True,
        "runtime_fixture_separation": True,
        "wide_runtime_probe": {
            "sites": wide["episodes"],
            "steps": wide["total_steps"],
            "interaction_actions": wide["interaction_actions"],
            "relevant_dqn_rules": {rule: wide["rule_counts"].get(rule, 0) for rule in relevant_rules},
            "layout_rules": {rule: wide["rule_counts"].get(rule, 0) for rule in ("R05", "R06")},
            "dqn_local": wide["dqn_local_chains"],
            "runtime_partial": wide["runtime_partial"],
            "runtime_successful": wide["runtime_successful"],
            "runtime_errors": len(wide["runtime_errors"]),
        },
        "formal_collection": {
            "steps": current["funnel"]["ppo_runtime_steps"],
            "candidates": current["counts"]["candidates"],
            "strategy_counts": current["strategy_counts"],
            "dqn_local": current["counts"]["dqn_local"],
        },
        "detector_structural_support": {
            "source_function_found": bool(source),
            "relevant_rule_tokens_present": detector_source_presence,
            "all_relevant_rule_tokens_present": all(detector_source_presence.values()),
        },
        "router_structural_support": {
            "routing_matrix_cases": len(routing_matrix()),
            "fixture_routes": fixture_routes,
            "ambiguous_fixtures_route_to_dqn": all(
                row["route"]["owner"] == "DQN_LOCAL" for row in fixture_routes if row["name"].startswith("ambiguous")
            ),
            "exact_fixture_routes_to_rule": fixture_routes[-1]["route"]["owner"] == "RULE_BASED_REPLAY",
        },
        "conclusion": {
            "detector_or_router_supports_dqn_path_structurally": all(detector_source_presence.values()) and all(row["route"]["owner"] == "DQN_LOCAL" for row in fixture_routes if row["name"].startswith("ambiguous")),
            "current_train_natural_dqn_case_observed": wide["relevant_dqn_rule_candidates"] > 0,
            "detector_bug_proven": False,
            "router_bug_proven": False,
            "final_root_cause": "CURRENT_TRAIN_RUNTIME_SCENARIO_NOT_OBSERVED_OR_RECOGNIZED",
            "hybrid_dqn_globally_unnecessary_proven": False,
            "dqn_training_allowed": False,
            "next_action": "Do not train DQN from fixtures. Obtain a separately approved runtime scenario/source coverage change, then recollect TRAIN-only runtime data."
        },
    }
    out_path = BASE / "dqn_final_scenario_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
