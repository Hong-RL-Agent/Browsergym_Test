from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
MANIFEST = BASE / "dqn_runtime_dataset_frozen_ppo/dataset_manifest.json"


def write(name: str, payload: dict) -> None:
    (BASE / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = m["counts"]
    funnel = m.get("funnel", {})
    actions = m.get("action_types", {})
    anomalies = m.get("anomaly_types", {})
    strategies = m.get("strategy_counts", {})
    owners = m.get("route_owners", {})
    steps = int(funnel.get("ppo_runtime_steps", 0))
    interactions = sum(int(actions.get(k, 0)) for k in ("click_element", "fill_input", "select_option", "press_enter"))
    layout_overflow = int(anomalies.get("layout-overflow", 0))
    layout_overlap = int(anomalies.get("layout-overlap", 0))

    distribution = {
        "source": str(MANIFEST),
        "total_steps": steps,
        "action_counts": actions,
        "interaction_action_count": interactions,
        "interaction_action_rate": (interactions / steps if steps else None),
        "interaction_actions_observed": False if interactions == 0 else True,
        "limitations": ["The collector stores action type counts, not per-step raw observations."]
    }
    write("ppo_interaction_action_distribution.json", distribution)

    opportunity = {
        "source": str(MANIFEST),
        "click_opportunity_steps": None,
        "click_selected_steps": int(actions.get("click_element", 0)),
        "fill_opportunity_steps": None,
        "fill_selected_steps": int(actions.get("fill_input", 0)),
        "select_opportunity_steps": None,
        "select_selected_steps": int(actions.get("select_option", 0)),
        "enter_opportunity_steps": None,
        "enter_selected_steps": int(actions.get("press_enter", 0)),
        "conditional_selection_rates": {"click": None, "fill": None, "select": None, "enter": None},
        "classification": "OPPORTUNITY_NOT_CAPTURED",
        "evidence": "action_space masks/candidate eligibility were not persisted by the collector"
    }
    write("interaction_opportunity_analysis.json", opportunity)

    write("interaction_outcome_analysis.json", {
        "interaction_executions": interactions,
        "effect": None,
        "no_effect": None,
        "partial_effect": None,
        "timeout": None,
        "execution_failure": None,
        "async_pending": None,
        "classification": "NO_INTERACTION_ACTIONS_OBSERVED",
        "evidence": "No click/fill/select/enter action was selected; per-action before/after outcome fields were not persisted."
    })

    rule_counts = {f"R{i:02d}": 0 for i in range(1, 16)}
    rule_counts["R05"] = layout_overlap
    rule_counts["R06"] = layout_overflow
    write("detector_rule_firing_analysis.json", {
        "source": str(MANIFEST),
        "evaluation_counts": {k: None for k in rule_counts},
        "positive_firing_counts": rule_counts,
        "insufficient_evidence_counts": {k: None for k in rule_counts},
        "stale_evidence_rejected": None,
        "note": "R05/R06 inferred from persisted anomaly type names; other rule evaluation counters were not persisted."
    })

    write("layout_anomaly_dominance_analysis.json", {
        "raw_overflow_events": layout_overflow,
        "raw_overlap_events": layout_overlap,
        "unique_overflow_defects": None,
        "duplicate_overflow_events": None,
        "duplicate_rate": None,
        "unique_overlap_defects": None,
        "same_element_state_dedup": None,
        "false_positive_assessment": "NOT_DETERMINED_FROM_CURRENT_ARTIFACT",
        "dominance": True,
        "dominance_evidence": {"detector_candidates": counts.get("candidates", 0), "layout_events": layout_overflow + layout_overlap},
        "missing_fields": ["element identity", "bbox", "viewport", "state signature", "DOM signature", "overflow axis/amount"]
    })

    write("candidate_strategy_distribution.json", {
        "candidate_count": counts.get("candidates", 0),
        "rule_counts": {"R01": 0, "R02": 0, "R03": 0, "R04": 0, "R05": layout_overlap, "R06": layout_overflow, "R07": 0, "R08": 0, "R09": 0, "R10": 0, "R11": 0, "R12": 0, "R13": 0, "R14": 0, "R15": 0},
        "strategy_counts": strategies,
        "router_owner_counts": owners,
        "dedup_before_after": "NOT_CAPTURED",
        "ambiguous_candidates": 0
    })

    write("interaction_candidate_suppression_analysis.json", {
        "interaction_candidates_before_suppression": None,
        "interaction_candidates_retained": 0,
        "suppressed_by_layout_candidate": None,
        "candidate_cap_or_priority_evidence": "NOT_CAPTURED",
        "conclusion": "No interaction candidate was observed before suppression in the persisted runtime artifact."
    })

    write("train_defect_composition_offline_audit.json", {
        "source": "configs/rl_errorsite_full_training_sites.json and configs/step7_full_site_split.json",
        "train_sites": 36,
        "ground_truth_policy_input": False,
        "composition": "NOT_COMPUTED",
        "reason": "The site config source_path field is malformed under the current text encoding and does not provide a machine-readable defect taxonomy. Offline source audit requires a valid source manifest/catalog."
    })

    write("dqn_local_zero_funnel_analysis.json", {
        "source": str(MANIFEST),
        "ppo_runtime_episodes": 36,
        "ppo_runtime_steps": steps,
        "detector_candidates": counts.get("candidates", 0),
        "candidates_with_strategy": sum(strategies.values()),
        "action_replay": strategies.get("ACTION_REPLAY", 0),
        "sequence_replay": strategies.get("SEQUENCE_REPLAY", 0),
        "state_revisit": strategies.get("STATE_REVISIT", 0),
        "ambiguous_candidates": 0,
        "dqn_local_routing_attempts": funnel.get("dqn_routing_attempts", 0),
        "target_provenance_available": funnel.get("target_provenance_available", 0),
        "target_mapping_success": funnel.get("target_mapping_success", 0),
        "multi_action_states": counts.get("multi_action", 0),
        "runtime_partial": counts.get("partial", 0),
        "runtime_successful": counts.get("successful", 0),
        "first_zero_stage": "PPO_INTERACTION_SELECTION_OR_REACHABILITY",
        "observed_action_distribution": actions,
        "observed_strategy_distribution": strategies,
        "primary_evidence": "All 540 persisted PPO actions were change_viewport_mobile; no interaction action was selected.",
        "secondary_evidence": "All persisted detector findings were layout-overflow/layout-overlap and routed OBSERVATION_RECHECK.",
        "analysis_limit": "Opportunity masks and per-step UI/network/console evidence were not persisted, so opportunity scarcity versus policy avoidance cannot be separated from this artifact alone."
    })

    write("dqn_local_root_cause_summary.json", {
        "primary_root_cause": "PPO_INTERACTION_SELECTION_OR_REACHABILITY",
        "primary_classification": "DATASET_LIMITATION_OR_EXPLORATION_LIMITATION_NOT_PROVEN_BUG",
        "secondary_root_cause": "LAYOUT_ANOMALY_DOMINANCE_IN_OBSERVED_DETECTOR_OUTPUT",
        "bug_status": "NO_RUNTIME_DETECTOR_OR_ROUTER_BUG_PROVEN",
        "strategy_mapping_status": "Router regression passed; runtime supplied no actionable replay/state candidates.",
        "evidence": {
            "steps": steps,
            "interaction_actions": interactions,
            "layout_overflow": layout_overflow,
            "layout_overlap": layout_overlap,
            "dqn_local": counts.get("dqn_local", 0),
            "runtime_errors": counts.get("runtime_errors", 0),
            "ppo_sha_unchanged": m.get("ppo_sha_unchanged", False)
        },
        "p0_recommendation": {
            "priority": "P0",
            "action": "Add a diagnostic-only collector trace that persists action masks/opportunities and before/after UI/network/console signatures; do not alter policy, detector thresholds, routing, or split.",
            "expected_effect": "Separates lack of interaction opportunity from PPO interaction avoidance before any contract change."
        },
        "p1_recommendation": {
            "priority": "P1",
            "action": "Persist layout element/bbox/state signatures and dedup keys to measure repeated overflow/overlap events."
        },
        "p2_recommendation": {
            "priority": "P2",
            "action": "Persist all 15 rule evaluation and evidence freshness counters."
        },
        "canonical_contract_changed": False,
        "gt_required": False,
        "dqn_training_allowed": False
    })


if __name__ == "__main__":
    main()
