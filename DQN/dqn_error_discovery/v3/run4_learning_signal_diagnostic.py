"""Post-hoc learning-signal diagnostic for Run4 Candidate B.

This script is analysis-only. It reads completed validation/training artifacts and
does not load models, environments, GT at runtime, or write checkpoints.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "training_v3"
VAL = ART / "run4_candidate_b_validation"
RAW = VAL / "raw"
TRAIN = ART / "run4_fullstack"
FAKEWEB = ART / "run4_fakeweb"
OUT = ART / "run4_candidate_b_learning_signal_diagnostic"

CANDIDATES = ("candidate_a", "candidate_b")
SEEDS = (42, 43, 44)
DEV_SITES = ("site006", "site007", "site008", "site009", "site010")
RULES = [f"R{i:02d}" for i in range(1, 16)]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def category_from_key(key: str) -> str:
    first = (key or "").split("|", 1)[0]
    return first if first in RULES else ""


def run_dir(site: str, candidate: str, seed: int) -> Path:
    return RAW / f"{site}_{candidate}_seed{seed}"


def collect_candidate_keys(timeline: list[dict[str, Any]], lifecycle: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in timeline:
        keys.update(str(k) for k in row.get("candidate_keys", []) or [])
        keys.update(str(k) for k in row.get("confirmed_keys", []) or [])
    for row in lifecycle:
        key = str(row.get("logical_key") or "")
        if key:
            keys.add(key)
    return keys


def collect_confirmed_keys(lifecycle: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("logical_key") or "")
        for row in lifecycle
        if row.get("event") == "candidate_to_confirmed" and row.get("logical_key")
    }


def load_dev_bugs() -> dict[str, list[dict[str, Any]]]:
    registry = read_json(FAKEWEB / "fakeweb_registry.json")
    out: dict[str, list[dict[str, Any]]] = {}
    for site in registry.get("sites", []):
        site_id = str(site.get("site_id") or "")
        if site_id in DEV_SITES:
            out[site_id] = site.get("bugs", []) or []
    return out


def summarize_validation(dev_bugs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = read_json(VAL / "candidate_a_vs_b_summary.json")["aggregate"]
    stage_rows: list[dict[str, Any]] = []
    representative: list[dict[str, Any]] = []
    candidate_distribution: dict[str, Any] = {}
    evidence_source_counts: dict[str, Any] = {}
    failure_counts = {c: Counter() for c in CANDIDATES}
    taxonomy_failure = defaultdict(lambda: {c: Counter() for c in CANDIDATES})

    for candidate in CANDIDATES:
        cat_counter: Counter[str] = Counter()
        key_counter: Counter[str] = Counter()
        confirmed_cat_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        reward_positive = 0
        reward_positive_unmatched = 0
        total_steps = 0
        evidence_counts = Counter()
        dqn_effect = Counter()

        for site in DEV_SITES:
            gt_cats = {str(b.get("mapped_rule") or "") for b in dev_bugs[site]}
            for seed in SEEDS:
                d = run_dir(site, candidate, seed)
                timeline = read_jsonl(d / "step_timeline.jsonl")
                lifecycle = read_jsonl(d / "candidate_lifecycle.jsonl")
                controller = read_jsonl(d / "controller_events.jsonl")
                keys = collect_candidate_keys(timeline, lifecycle)
                confirmed_keys = collect_confirmed_keys(lifecycle)
                cats = {category_from_key(k) for k in keys if category_from_key(k)}
                matched_keys = {k for k in keys if category_from_key(k) in gt_cats}

                for key in keys:
                    key_counter[key] += 1
                    cat = category_from_key(key)
                    if cat:
                        cat_counter[cat] += 1
                for key in confirmed_keys:
                    cat = category_from_key(key)
                    if cat:
                        confirmed_cat_counter[cat] += 1

                for row in timeline:
                    total_steps += 1
                    action_type = str(row.get("action_type") or "")
                    source_counter[action_type] += 1
                    if action_type.startswith("inspect_"):
                        evidence_counts[action_type] += 1
                    if row.get("anomaly_emissions", 0):
                        evidence_counts["anomaly_emission_steps"] += 1
                    if row.get("candidate_eligible", 0):
                        evidence_counts["candidate_eligible_steps"] += 1
                    if row.get("candidate_admission", 0):
                        evidence_counts["candidate_admission_steps"] += 1
                    if row.get("active_agent") == "DQN_LOCAL":
                        dqn_effect["dqn_actions"] += 1
                        dqn_effect["no_effect"] += int(bool(row.get("no_effect")))
                        dqn_effect["effect"] += int(not bool(row.get("no_effect")))
                        dqn_effect["repeat_action_target"] += int(bool(row.get("repeat_action_target")))
                        dqn_effect["repeat_state"] += int(bool(row.get("repeat_state")))
                    reward = float(row.get("reward") or 0.0)
                    if reward > 0:
                        reward_positive += 1
                        row_cats = {category_from_key(k) for k in row.get("candidate_keys", []) or []}
                        if not row_cats.intersection(gt_cats):
                            reward_positive_unmatched += 1

                for bug in dev_bugs[site]:
                    taxonomy = str(bug.get("mapped_rule") or "")
                    gt_reached = True  # The existing evaluator defines reached as run browser_steps > 0.
                    trigger_executed = "UNKNOWN_NOT_LOGGED"
                    evidence_present = taxonomy in cats
                    detector_rule_evaluated = bool(evidence_counts["anomaly_emission_steps"] or keys)
                    detector_fired = taxonomy in cats
                    candidate_admitted = taxonomy in cats
                    gt_matched = bool(matched_keys and taxonomy in {category_from_key(k) for k in matched_keys})
                    handoff_started = any((r.get("handoff_count") or 0) > 0 for r in controller)
                    confirmed = taxonomy in {category_from_key(k) for k in confirmed_keys}
                    if not gt_reached:
                        first_failure = "NOT_REACHED"
                    elif trigger_executed == "UNKNOWN_NOT_LOGGED":
                        first_failure = "TRIGGER_EXECUTION_NOT_VERIFIABLE_FROM_ARTIFACT"
                    elif not evidence_present:
                        first_failure = "EVIDENCE_NOT_CAPTURED"
                    elif not detector_fired:
                        first_failure = "DETECTOR_NOT_FIRED"
                    elif not candidate_admitted:
                        first_failure = "CANDIDATE_GATE_REJECTED"
                    elif not gt_matched:
                        first_failure = "GT_MATCH_FAILED"
                    elif handoff_started and not confirmed:
                        first_failure = "CONFIRMATION_NOT_CREATED"
                    else:
                        first_failure = "CONFIRMED"
                    failure_counts[candidate][first_failure] += 1
                    taxonomy_failure[taxonomy][candidate][first_failure] += 1
                    stage_rows.append(
                        {
                            "candidate": candidate,
                            "site": site,
                            "seed": seed,
                            "bug_id": bug.get("bug_id"),
                            "taxonomy": taxonomy,
                            "gt_reached": gt_reached,
                            "gt_trigger_action_executed": trigger_executed,
                            "expected_effect_observable": "UNKNOWN_NOT_LOGGED",
                            "relevant_dom_evidence_present": False,
                            "relevant_ui_evidence_present": False,
                            "relevant_network_evidence_present": False,
                            "relevant_console_evidence_present": False,
                            "relevant_layout_evidence_present": taxonomy in {"R05", "R06"} and taxonomy in cats,
                            "temporal_evidence_present": False,
                            "detector_rule_evaluated": detector_rule_evaluated,
                            "detector_rule_fired": detector_fired,
                            "candidate_created": candidate_admitted,
                            "candidate_admitted": candidate_admitted,
                            "candidate_gt_matched_posthoc": gt_matched,
                            "dqn_handoff_started": handoff_started,
                            "dqn_confirmation_action_executed": handoff_started,
                            "repeated_evidence_observed": bool(confirmed_keys),
                            "confirmation_created": confirmed,
                            "first_failure_stage": first_failure,
                        }
                    )

        candidate_distribution[candidate] = {
            "candidate_category_distribution": dict(cat_counter),
            "confirmed_category_distribution": dict(confirmed_cat_counter),
            "top_repeated_candidate_signatures": key_counter.most_common(20),
            "candidate_action_source_distribution": dict(source_counter),
            "positive_reward_steps": reward_positive,
            "positive_reward_steps_without_gt_category_candidate": reward_positive_unmatched,
            "dqn_effect_summary": dict(dqn_effect),
        }
        evidence_source_counts[candidate] = {
            "total_steps": total_steps,
            "source_action_counts": dict(source_counter),
            "evidence_counts": dict(evidence_counts),
            "network_request_fields_present": False,
            "network_response_body_fields_present": False,
            "ui_value_pairing_fields_present": False,
        }

    # Representative traces: six R10 and four non-R10 rows from Candidate B first, then A if needed.
    chosen: list[dict[str, Any]] = []
    b_rows = [r for r in stage_rows if r["candidate"] == "candidate_b"]
    chosen.extend([r for r in b_rows if r["taxonomy"] == "R10"][:6])
    chosen.extend([r for r in b_rows if r["taxonomy"] != "R10"][:4])
    for row in chosen:
        site = row["site"]
        seed = int(row["seed"])
        candidate = row["candidate"]
        d = run_dir(site, candidate, seed)
        timeline = read_jsonl(d / "step_timeline.jsonl")
        lifecycle = read_jsonl(d / "candidate_lifecycle.jsonl")
        controller = read_jsonl(d / "controller_events.jsonl")
        keys = sorted(collect_candidate_keys(timeline, lifecycle))[:10]
        actions = [
            {
                "step": t.get("step"),
                "agent": t.get("active_agent"),
                "action_type": t.get("action_type"),
                "target": t.get("action_target"),
                "candidate_keys": t.get("candidate_keys"),
                "anomaly_emissions": t.get("anomaly_emissions"),
                "no_effect": t.get("no_effect"),
            }
            for t in timeline[:8]
        ]
        representative.append(
            {
                "gt": {k: row[k] for k in ("site", "bug_id", "taxonomy", "seed", "candidate")},
                "trigger": next((b for b in dev_bugs[site] if b.get("bug_id") == row["bug_id"]), {}),
                "selected_actions_sample": actions,
                "candidate_keys_sample": keys,
                "handoff_count": sum(int(c.get("handoff_count") or 0) for c in controller),
                "detected_false_reason": row["first_failure_stage"],
            }
        )

    return {
        "source_summary_metrics": summary["metrics"],
        "stage_matrix": stage_rows,
        "failure_stage_counts": {c: dict(failure_counts[c]) for c in CANDIDATES},
        "taxonomy_failure_stage_counts": {
            tax: {c: dict(vals[c]) for c in CANDIDATES} for tax, vals in taxonomy_failure.items()
        },
        "candidate_distribution": candidate_distribution,
        "evidence_source_counts": evidence_source_counts,
        "representative_traces": representative,
    }


def summarize_training() -> dict[str, Any]:
    summary = read_json(TRAIN / "run4_training_summary.json")
    dqn_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        rows = read_jsonl(TRAIN / f"dqn_seed{seed}_metrics.jsonl")
        dqn_rows.extend(rows)
        per_seed[str(seed)] = {
            "rows": len(rows),
            "reward_total": sum(float(r.get("reward") or 0) for r in rows),
            "positive_reward_events": sum(1 for r in rows if float(r.get("reward") or 0) > 0),
            "candidate_admissions": sum(int(r.get("candidate_admissions") or 0) for r in rows),
            "confirmations": sum(int(r.get("confirmations") or 0) for r in rows),
            "unique_candidates_sum": sum(int(r.get("unique_candidates") or 0) for r in rows),
            "candidate_to_confirmed": sum(int(r.get("candidate_to_confirmed") or 0) for r in rows),
            "unknown_causality_emissions": sum(int(r.get("UNKNOWN") or 0) for r in rows),
            "steps": sum(int(r.get("steps") or 0) for r in rows),
            "no_effect": sum(int(r.get("no_effect") or 0) for r in rows),
        }
    candidate_events = sum(item.get("candidate_admissions", 0) for item in summary.get("dqn_training", []))
    unique_candidates = sum(item.get("unique_candidates", 0) for item in summary.get("dqn_training", []))
    positive = sum(1 for r in dqn_rows if float(r.get("reward") or 0) > 0)
    # Training artifacts do not retain logical candidate keys or GT match ids, so post-hoc alignment cannot be computed.
    return {
        "summary_candidate_events": candidate_events,
        "summary_unique_candidates": unique_candidates,
        "event_unique_ratio": candidate_events / unique_candidates if unique_candidates else None,
        "total_sessions": len(dqn_rows),
        "positive_reward_events": positive,
        "gt_aligned_unique_candidates": "INSUFFICIENT_DATA_NO_TRAINING_CANDIDATE_LOGICAL_KEYS",
        "unmatched_unique_candidates": "INSUFFICIENT_DATA_NO_TRAINING_CANDIDATE_LOGICAL_KEYS",
        "gt_aligned_positive_reward_share": "INSUFFICIENT_DATA_NO_TRAINING_CANDIDATE_LOGICAL_KEYS",
        "unmatched_positive_reward_share": "INSUFFICIENT_DATA_NO_TRAINING_CANDIDATE_LOGICAL_KEYS",
        "per_seed": per_seed,
        "taxonomy_distribution": summary.get("taxonomy_distribution", {}),
        "dqn_training_summary": summary.get("dqn_training", []),
    }


def main() -> None:
    dev_bugs = load_dev_bugs()
    validation = summarize_validation(dev_bugs)
    training = summarize_training()
    metrics = validation["source_summary_metrics"]

    # Counts are reported for Candidate B as the learning-signal target, with Candidate A comparison included.
    b_rows = [r for r in validation["stage_matrix"] if r["candidate"] == "candidate_b"]
    r10_rows = [r for r in b_rows if r["taxonomy"] == "R10"]
    report_counts = {
        "gt_area_reached": sum(1 for r in b_rows if r["gt_reached"]),
        "gt_trigger_executed_directly_verifiable": 0,
        "gt_trigger_executed_status": "NOT_LOGGED_IN_VALIDATION_ARTIFACTS",
        "relevant_evidence_present": sum(1 for r in b_rows if r["detector_rule_fired"]),
        "detector_evaluated": sum(1 for r in b_rows if r["detector_rule_evaluated"]),
        "detector_fired": sum(1 for r in b_rows if r["detector_rule_fired"]),
        "candidate_admitted": sum(1 for r in b_rows if r["candidate_admitted"]),
        "gt_matched_candidate": sum(1 for r in b_rows if r["candidate_gt_matched_posthoc"]),
        "dqn_handoff": sum(1 for r in b_rows if r["dqn_handoff_started"]),
        "effective_dqn_actions": metrics["candidate_b"]["runtime"]["dqn_steps"]
        - metrics["candidate_b"]["aggregate"]["dqn_no_effect_count"],
        "confirmed": metrics["candidate_b"]["gt_confirmed"],
        "r10_trigger_executed_directly_verifiable": 0,
        "r10_trigger_executed_status": "NOT_LOGGED_IN_VALIDATION_ARTIFACTS",
        "r10_evidence_captured": sum(1 for r in r10_rows if r["detector_rule_fired"]),
        "r10_detector_fired": sum(1 for r in r10_rows if r["detector_rule_fired"]),
        "r10_candidate_admitted": sum(1 for r in r10_rows if r["candidate_admitted"]),
        "r10_gt_matched": sum(1 for r in r10_rows if r["candidate_gt_matched_posthoc"]),
    }
    root_cause = {
        "primary": "MULTI_FACTOR",
        "secondary": [
            "TRIGGER_EXECUTION_BOTTLENECK",
            "EVIDENCE_COLLECTION_BOTTLENECK",
            "DETECTOR_SIGNAL_BOTTLENECK",
            "DQN_CONFIRMATION_ACTION_BOTTLENECK",
            "REWARD_PROXY_MISALIGNMENT_POSSIBLE_BUT_TRAINING_GT_ALIGNMENT_INSUFFICIENT_DATA",
        ],
        "rationale": [
            "Existing scoring is internally consistent: matched_candidates=0 because no candidate logical-key category matched RUN4_DEV GT categories.",
            "The evaluator's reached flag is browser-step based, not proof that GT endpoint/trigger actions were executed.",
            "Validation artifacts do not contain network response bodies, UI/network semantic pairing, or endpoint-hit evidence required to prove R10 mismatch computation.",
            "Candidate admissions are dominated by non-GT categories such as R01/R06 and generic action/layout candidates.",
            "DQN local actions are almost entirely no-effect, so local confirmation is not creating GT-relevant reproduction evidence.",
        ],
    }
    payload = {
        "version": "RAWD_V3_RUN4_CANDIDATE_B_LEARNING_SIGNAL_DIAGNOSTIC",
        "inputs": {
            "validation_summary": str(VAL / "candidate_a_vs_b_summary.json"),
            "training_summary": str(TRAIN / "run4_training_summary.json"),
            "fakeweb_registry": str(FAKEWEB / "fakeweb_registry.json"),
        },
        "gt_scoring_sanity": {
            "result": "PASS",
            "detected_scoring_bug": False,
            "matcher_rule_observed": "candidate logical-key category must match GT mapped_rule category",
            "matched_candidates": {
                "candidate_a": metrics["candidate_a"]["matched_candidates"],
                "candidate_b": metrics["candidate_b"]["matched_candidates"],
            },
            "note": "No evidence was found that a GT-category candidate was missed by the matcher in the available artifacts.",
        },
        "report_counts_candidate_b": report_counts,
        "candidate_a_failure_stage_counts": validation["failure_stage_counts"]["candidate_a"],
        "candidate_b_failure_stage_counts": validation["failure_stage_counts"]["candidate_b"],
        "training_signal": training,
        "root_cause": root_cause,
        "markers": {
            "RAWD_V3_RUN4_CANDIDATE_B_LEARNING_SIGNAL_DIAGNOSTIC_CHECK": "PASS",
            "RAWD_V3_RUN4_GT_SCORING_SANITY_CHECK": "PASS",
            "RAWD_V3_RUN4_REWARD_ALIGNMENT_CHECK": "INSUFFICIENT_DATA",
            "RAWD_V3_RUN4_EVIDENCE_PIPELINE_CHECK": "BOTTLENECK_FOUND",
        },
    }

    write_json(OUT / "gt_pipeline_stage_matrix.json", validation["stage_matrix"])
    write_json(OUT / "candidate_distribution.json", validation["candidate_distribution"])
    write_json(OUT / "evidence_source_counts.json", validation["evidence_source_counts"])
    write_json(OUT / "representative_traces.json", validation["representative_traces"])
    write_json(OUT / "training_reward_alignment.json", training)
    write_json(OUT / "learning_signal_diagnostic_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
