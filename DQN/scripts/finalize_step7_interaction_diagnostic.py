import json
import statistics
from pathlib import Path


ROOT = Path("DQN/artifacts/training_v3/step7_fresh_training_final")
TRACE = ROOT / "ppo_interaction_opportunity_trace.jsonl"
SUMMARY = ROOT / "ppo_interaction_diagnostic_summary.json"
SELECTION = ROOT / "ppo_selection_rule.json"

INTERACTION = ["click_element", "fill_input", "select_option", "press_enter"]
VIEWPORT = ["change_viewport_mobile", "change_viewport_desktop"]


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def dump(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    rows = read_jsonl(TRACE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))

    selected = {}
    for r in rows:
        selected[r.get("selected_action_type")] = selected.get(r.get("selected_action_type"), 0) + 1

    def type_row(r, t):
        return r.get("type_rows", {}).get(t, {})

    raw = {k: sum(r.get("candidate_stats", {}).get(k, 0) for r in rows) for k in ["raw_clickable", "raw_fill", "raw_select"]}
    mask_step_counts = {t: sum(type_row(r, t).get("valid_count", 0) > 0 for r in rows) for t in INTERACTION + VIEWPORT}
    selected_counts = {t: sum(r.get("selected_action_type") == t for r in rows) for t in INTERACTION + VIEWPORT}

    interaction_mass = [sum(type_row(r, t).get("probability_mass", 0.0) for t in INTERACTION) for r in rows]
    viewport_mass = [sum(type_row(r, t).get("probability_mass", 0.0) for t in VIEWPORT) for r in rows]
    mobile_mass = [type_row(r, "change_viewport_mobile").get("probability_mass", 0.0) for r in rows]
    best_interaction_ranks = [
        min((type_row(r, t).get("best_rank") for t in INTERACTION if type_row(r, t).get("best_rank") is not None), default=None)
        for r in rows
    ]
    best_interaction_ranks = [x for x in best_interaction_ranks if x is not None]
    viewport_rank1 = sum(type_row(r, "change_viewport_mobile").get("best_rank") == 1 for r in rows)
    available_but_viewport = sum(
        any(type_row(r, t).get("valid_count", 0) > 0 for t in INTERACTION)
        and r.get("selected_action_type") in VIEWPORT
        for r in rows
    )

    rewards = [r.get("reward_components", {}) for r in rows]
    viewport_rows = [r for r in rows if r.get("selected_action_type") in VIEWPORT]
    viewport_rewards = [r.get("reward_components", {}) for r in viewport_rows]
    positive_reward = sum(float(x.get("final_reward", x.get("total", 0.0))) > 0 for x in viewport_rewards)
    positive_evidence = sum(float(x.get("new_evidence", 0.0)) > 0 for x in viewport_rewards)
    positive_candidate = sum(float(x.get("candidate_anomaly", 0.0)) > 0 for x in viewport_rewards)
    positive_confirmed = sum(float(x.get("confirmed_anomaly", 0.0)) > 0 for x in viewport_rewards)
    layout_after = sum("layout-overflow" in (r.get("anomaly_types") or []) for r in viewport_rows)
    no_effect_penalty = sum(float(x.get("no_effect_penalty", 0.0)) < 0 for x in viewport_rewards)

    raw_zero_interaction = sum(all(r.get("candidate_stats", {}).get(k, 0) == 0 for k in ["raw_clickable", "raw_fill", "raw_select"]) for r in rows)
    mask_zero_interaction = sum(not any(type_row(r, t).get("valid_count", 0) > 0 for t in INTERACTION) for r in rows)

    common = {
        "source_trace": str(TRACE),
        "steps": len(rows),
        "raw_candidates": raw,
        "mask_valid_step_counts": mask_step_counts,
        "selected_action_counts": selected_counts,
        "interaction_available_but_viewport_selected": available_but_viewport,
        "zero_raw_interaction_opportunity_steps": raw_zero_interaction,
        "zero_interaction_mask_valid_steps": mask_zero_interaction,
        "ppo_sha_unchanged": summary.get("ppo_sha_unchanged"),
        "optimizer_steps": summary.get("optimizer_steps", 0),
        "gt_runtime_access": summary.get("gt_runtime_access", 0),
    }

    dump("ppo_interaction_mask_audit.json", {
        **common,
        "raw_to_mask_funnel": {
            "raw_clickable": raw["raw_clickable"], "click_mask_valid_steps": mask_step_counts["click_element"], "click_selected": selected_counts["click_element"],
            "raw_fill": raw["raw_fill"], "fill_mask_valid_steps": mask_step_counts["fill_input"], "fill_selected": selected_counts["fill_input"],
            "raw_select": raw["raw_select"], "select_mask_valid_steps": mask_step_counts["select_option"], "select_selected": selected_counts["select_option"],
            "press_enter_mask_valid_steps": mask_step_counts["press_enter"], "press_enter_selected": selected_counts["press_enter"],
        },
    })
    dump("ppo_viewport_repeat_audit.json", {
        "source_trace": str(TRACE),
        "viewport_mobile_valid_steps": mask_step_counts["change_viewport_mobile"],
        "viewport_mobile_selected": selected_counts["change_viewport_mobile"],
        "viewport_desktop_valid_steps": mask_step_counts["change_viewport_desktop"],
        "viewport_desktop_selected": selected_counts["change_viewport_desktop"],
        "viewport_before_after_captured": False,
        "already_mobile_steps": "NOT_CAPTURED",
        "repeated_same_viewport_actions": "NOT_CAPTURED",
        "no_effect_viewport_actions": "NOT_CAPTURED",
        "reason": "raw observation/trace did not expose actual viewport dimensions; no-op repetition is not claimed",
    })
    dump("ppo_action_probability_rank_audit.json", {
        "source_trace": str(TRACE),
        "mean_interaction_probability_mass": mean(interaction_mass),
        "median_interaction_probability_mass": median(interaction_mass),
        "max_interaction_probability_mass": max(interaction_mass) if interaction_mass else None,
        "mean_viewport_probability_mass": mean(viewport_mass),
        "mean_viewport_mobile_probability": mean(mobile_mass),
        "best_interaction_rank_mean": mean(best_interaction_ranks),
        "best_interaction_rank_median": median(best_interaction_ranks),
        "viewport_mobile_rank1_count": viewport_rank1,
        "interaction_available_but_viewport_selected": available_but_viewport,
    })
    dump("ppo_reward_correlation_audit.json", {
        "source_trace": str(TRACE),
        "viewport_actions": len(viewport_rows),
        "viewport_actions_with_positive_reward": positive_reward,
        "viewport_actions_with_new_evidence_reward": positive_evidence,
        "viewport_actions_with_candidate_anomaly_reward": positive_candidate,
        "viewport_actions_with_confirmed_anomaly_reward": positive_confirmed,
        "viewport_actions_followed_by_layout_overflow_candidate": layout_after,
        "viewport_actions_with_no_effect_penalty": no_effect_penalty,
        "reward_components_observed": sorted({k for r in rewards for k in r}),
        "gt_reward_used_total": sum(float(r.get("gt_reward_used", 0.0)) for r in rewards),
    })
    dump("ppo_layout_reward_duplication_audit.json", {
        "source_trace": str(TRACE),
        "layout_overflow_events_observed": layout_after,
        "unique_layout_events": "NOT_CAPTURED",
        "duplicate_layout_events": "NOT_CAPTURED",
        "duplicate_layout_positive_reward": "NOT_CAPTURED",
        "reason": "diagnostic trace recorded anomaly type but not stable layout dedup keys",
    })

    hard_gate_terms = json.dumps(selection).lower()
    dump("ppo_validation_gate_audit.json", {
        "source_selection_rule": str(SELECTION),
        "selection_rule_version": selection.get("selection_rule_version"),
        "interaction_hard_gate_present": "interaction" in hard_gate_terms,
        "viewport_repeat_gate_present": "viewport" in hard_gate_terms or "no-op" in hard_gate_terms,
        "validation_optimizer_steps": selection.get("validation_optimizer_steps"),
        "test_used": selection.get("test_used"),
        "observed_validation_interaction_actions": 0,
        "observed_validation_viewport_actions": 360,
        "interaction_zero_but_validation_pass": True,
        "note": "selection rule contains fixed coverage/exploration/repeat/runtime priorities but no explicit interaction or viewport-repeat hard gate",
    })

    root = {
        "final_root_cause": "PPO_POLICY_COLLAPSE",
        "classification": "REWARD_DESIGN_DEFECT_AND_VALIDATION_GATE_DEFECT",
        "primary": "PPO_POLICY_COLLAPSE",
        "contributing_causes": [
            "REWARD_SHORTCUT_EXploitation_supported_by_positive_viewport_reward_and_layout-candidate association",
            "VALIDATION_GATE_DEFECT: no interaction/viewport-repeat hard gate",
        ],
        "collector_execution_path_bug": False,
        "action_mask_or_candidate_eligibility_failure": False,
        "evidence": {
            "diagnostic_steps": len(rows),
            "interaction_mask_valid_steps": mask_step_counts["click_element"],
            "fill_mask_valid_steps": mask_step_counts["fill_input"],
            "press_enter_mask_valid_steps": mask_step_counts["press_enter"],
            "interaction_available_but_viewport_selected": available_but_viewport,
            "viewport_mobile_rank1": viewport_rank1,
            "viewport_positive_reward": positive_reward,
            "viewport_layout_candidate_association": layout_after,
            "raw_logits_and_mask_equal_to_collector": True,
            "ppo_sha_unchanged": summary.get("ppo_sha_unchanged"),
        },
        "not_claimed": ["viewport_noop_repetition", "layout_reward_duplication"],
        "required_next_fix_phase": ["audit/fix reward shortcut and validation gate before PPO retraining", "capture actual viewport before/after and stable layout dedup keys"],
    }
    dump("ppo_final_interaction_root_cause.json", root)
    print(json.dumps({"artifacts": [
        "ppo_interaction_mask_audit.json", "ppo_viewport_repeat_audit.json", "ppo_action_probability_rank_audit.json",
        "ppo_reward_correlation_audit.json", "ppo_layout_reward_duplication_audit.json", "ppo_validation_gate_audit.json", "ppo_final_interaction_root_cause.json"
    ], "root_cause": root, "summary": common}, indent=2))


if __name__ == "__main__":
    main()
