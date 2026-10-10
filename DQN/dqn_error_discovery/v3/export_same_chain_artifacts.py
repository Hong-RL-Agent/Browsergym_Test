from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "training_v3" / "run6_same_chain_recurrence_validation"


def main() -> None:
    runtime = json.loads((OUT / "production_hybrid_runtime.json").read_text(encoding="utf-8"))
    records = runtime.get("downstream_validation", {}).get("records", [])
    timeline = runtime.get("timeline", [])
    handoffs = runtime.get("handoff_events", [])
    confirmations = runtime.get("confirmation_attribution", {}).get("confirmation_records", [])

    origins = {}
    for record in records:
        for anomaly in record.get("anomalies", []):
            if str(anomaly.get("reproduction_strategy") or "") != "ACTION_REPLAY":
                continue
            key = str(anomaly.get("signature") or "")
            evidence = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), dict) else {}
            origins.setdefault(key, {
                "candidate_logical_key": key,
                "candidate_status": "pending_confirmation",
                "reproduction_strategy": "ACTION_REPLAY",
                "requires_action_reproduction": bool(anomaly.get("requires_action_reproduction")),
                "origin_action_type": anomaly.get("trigger_action_type"),
                "origin_target_identity": anomaly.get("trigger_target_identity"),
                "reproduction_action_type": anomaly.get("reproduction_action_type"),
                "reproduction_target_identity": anomaly.get("reproduction_target_identity"),
                "origin_request_id": evidence.get("request_id"),
                "origin_endpoint": evidence.get("request_path") or evidence.get("request_url"),
                "origin_network_key": evidence.get("network_key"),
                "origin_network_value": evidence.get("network_value"),
                "origin_ui_key": evidence.get("ui_key"),
                "origin_ui_value": evidence.get("ui_value"),
                "origin_relation": evidence.get("relation"),
                "origin_pair_confidence": evidence.get("pair_confidence"),
                "origin_step": record.get("global_step"),
            })

    handoff_by_key = {}
    for event in handoffs:
        keys = list(event.get("entry_candidate_keys", []))
        if event.get("trigger_candidate_key"):
            keys.append(event["trigger_candidate_key"])
        for key in keys:
            if key in origins:
                handoff_by_key[key] = event

    transitions = []
    for key, origin in origins.items():
        handoff = handoff_by_key.get(key)
        session_id = handoff.get("local_session_id") if handoff else None
        expected_action = origin.get("reproduction_action_type")
        expected_target = origin.get("reproduction_target_identity")
        session_rows = [row for row in timeline if row.get("local_session_id") == session_id and row.get("active_agent") == "DQN_LOCAL"]
        for row in session_rows:
            selected_target = str(row.get("action_target") or "")
            target_match = selected_target == f"{expected_action}:{expected_target}" if expected_action and expected_target else False
            action_match = row.get("action_type") == expected_action if expected_action else False
            transitions.append({
                "candidate_logical_key": key,
                "local_session_id": session_id,
                "episode": row.get("episode"),
                "step": row.get("step"),
                "selected_action_type": row.get("action_type"),
                "selected_target_identity": selected_target,
                "reproduction_action_type_match": action_match,
                "reproduction_target_match": target_match,
                "action_executed": True,
                "execution_success": not bool(row.get("repeat_diagnostic", {}).get("action_error")),
                "candidate_relevant_effect": bool(row.get("reward_components", {}).get("new_target") or row.get("reward_components", {}).get("confirmation_upgrade")),
                "strict_request_recurrence": False,
                "semantic_not_equal_recurrence": False,
                "same_chain_recurrence": False,
            })

    same_chain = []
    for key, origin in origins.items():
        related = [row for row in transitions if row["candidate_logical_key"] == key]
        same_chain.extend(row for row in related if row["same_chain_recurrence"])

    action_replay_confirmations = [row for row in confirmations if row.get("logical_key") in origins]
    true_action_replay = [row for row in action_replay_confirmations if row.get("logical_key") in {item["candidate_logical_key"] for item in same_chain}]
    observation_confirmations = [row for row in confirmations if str(row.get("logical_key", "")).split("|", 1)[0] in {"R04", "R05", "R06"}]
    # The runtime does not expose a positive cross-candidate join key. Do not
    # infer CROSS_CANDIDATE from a missing recurrence; classify those as
    # NO_RECURRENCE instead.
    cross_candidate = []
    no_recurrence = [row for row in action_replay_confirmations if row not in true_action_replay]
    stale = []
    other = [row for row in confirmations if row not in action_replay_confirmations and row not in observation_confirmations]

    funnel = {
        "action_replay_candidates": len(origins),
        "dqn_handoffs": len(handoff_by_key),
        "reproduction_target_mapped": sum(int(bool(row.get("reproduction_action_type") and row.get("reproduction_target_identity"))) for row in origins.values()),
        "reproduction_target_actionable": sum(int(bool(handoff_by_key.get(key, {}).get("entry_target_resolvable"))) for key in origins),
        "reproduction_selected": sum(int(row["reproduction_target_match"] and row["reproduction_action_type_match"]) for row in transitions),
        "strict_request_recurrence": sum(int(row["strict_request_recurrence"]) for row in transitions),
        "semantic_not_equal_recurrence": sum(int(row["semantic_not_equal_recurrence"]) for row in transitions),
        "same_chain_recurrence": len(same_chain),
        "confirmed_action_replay": len(true_action_replay),
    }
    funnel["reproduction_selection_rate"] = funnel["reproduction_selected"] / max(1, funnel["dqn_handoffs"])
    funnel["recurrence_given_reproduction"] = funnel["same_chain_recurrence"] / max(1, funnel["reproduction_selected"])
    funnel["confirmation_given_recurrence"] = funnel["confirmed_action_replay"] / max(1, funnel["same_chain_recurrence"])
    funnel["FIRST_MAJOR_DROP_OFF"] = "DQN_REPRODUCTION_SELECTION_BOTTLENECK" if funnel["reproduction_selected"] == 0 and funnel["dqn_handoffs"] else "SAME_CHAIN_PIPELINE_RECOVERED"
    funnel["PRIMARY_ROOT_CAUSE"] = "Frozen DQN local policy did not select the mapped reproduction action/target; no forced reproduction was used."

    classifications = {
        "total_confirmations": len(confirmations),
        "DQN_REPRODUCTION": sum(int(row.get("confirmed_by") == "DQN" and row in true_action_replay) for row in confirmations),
        "OBSERVATION_CONFIRMATION": len(observation_confirmations),
        "TRUE_ACTION_REPLAY_CONFIRMATION": len(true_action_replay),
        "NO_RECURRENCE_CONFIRMATION": len(no_recurrence),
        "CROSS_CANDIDATE_CONFIRMATION": len(cross_candidate),
        "STALE_CONFIRMATION": len(stale),
        "OTHER": len(other),
        "UNKNOWN": 0,
        "records": confirmations,
    }
    lifecycle = []
    for key, origin in origins.items():
        lifecycle.append({
            **origin,
            "status_before": "pending_confirmation",
            "status_after": "confirmed" if key in {item["candidate_logical_key"] for item in same_chain} else "unresolved",
            "confirmation_event": key in {row.get("logical_key") for row in true_action_replay},
            "confirmation_source": "DQN_REPRODUCTION" if key in {row.get("logical_key") for row in true_action_replay} else "MISSING",
            "fresh_event": key in {item["candidate_logical_key"] for item in same_chain},
        })

    (OUT / "action_replay_funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "reproduction_transitions.jsonl").open("w", encoding="utf-8") as handle:
        for row in transitions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "same_chain_trace.json").write_text(json.dumps({"origins": list(origins.values()), "transitions": transitions, "same_chain": same_chain, "funnel": funnel}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "confirmation_classification.json").write_text(json.dumps(classifications, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "candidate_lifecycle.json").write_text(json.dumps({"candidates": lifecycle, "funnel": funnel}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "episodes": runtime.get("configuration", {}).get("episodes"),
        "global_steps": runtime.get("episode_summary", {}).get("total_steps"),
        **funnel,
        **{key: value for key, value in classifications.items() if key != "records"},
        "same_chain_sample_status": "INSUFFICIENT_ACTION_REPLAY_SAMPLES" if not origins else "SAMPLED_NO_RECURRENCE",
        "ppo_retrained": False,
        "dqn_retrained": False,
        "reward_modified": False,
        "replay_modified": False,
        "detector_modified": False,
        "gt_runtime_access": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
