"""Train/evaluate a DQN for ordinary BrowserGym UI and runtime error discovery.

This runner intentionally excludes infrastructure, authorization, and security
inspection actions. It records anomaly categories and evidence references only;
it does not attempt to exploit, bypass, or reproduce security weaknesses.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.dqn_agent import (
    FACTORIZED_DQN_ARCHITECTURE,
    FACTORIZED_DQN_CONFOBS_ARCHITECTURE,
    FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE,
    FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    FACTORIZED_SELECTION,
    FLAT_MAX_SELECTION,
    FLAT_DQN_ARCHITECTURE,
    FactorizedQNetwork,
    QNetwork,
    TYPE_LOGMEANEXP_SELECTION,
    TYPE_SCORE_NORMALIZED_LOGMEANEXP,
    ConfirmationAwareReplayBuffer,
    ReplayBuffer,
    Transition,
    balanced_expected_action_type_distribution,
    balanced_random_valid_action,
    double_dqn_next_actions,
    factorized_double_dqn_next_actions,
    factorized_q_for_actions,
    factorized_q_value_diagnostics,
    q_value_diagnostics,
    select_factorized_greedy_action,
    select_greedy_action,
    validate_selection_mode_consistency,
)
from dqn_error_discovery.agents.confirmation_observation import (
    CONFIRMATION_CONTEXT_DIM,
    CONFIRMATION_OBSERVATION_VERSION,
    augment_dqn_observation,
    build_dqn_confirmation_context,
)
from dqn_error_discovery.agents.history_observation import (
    DQN_HISTORY_WINDOW,
    HISTORY_CONTEXT_DIM,
    HISTORY_CONTEXT_V2_DIM,
    HISTORY_OBSERVATION_VERSION,
    HISTORY_OBSERVATION_V2_VERSION,
    PROSPECTIVE_REPEAT_RISK_FLAGS,
    append_history_transition,
    augment_dqn_history_observation,
    augment_dqn_history_observation_v2,
    build_dqn_history_context,
    build_dqn_history_context_v2,
)
from dqn_error_discovery.agents.repeat_semantics import cycle_repeat_length
from dqn_error_discovery.agents.confirmation_learning_diagnostics import (
    ConfirmationFunnelAggregator,
)
from dqn_error_discovery.agents.confirmation_reproduction import (
    ACTION_REPRODUCTION,
    OBSERVATION_REPRODUCTION,
    TARGET_ACTION_TYPES,
)
from dqn_error_discovery.agents.confirmation_reproduction import (
    ACTION_REPRODUCTION,
    OBSERVATION_REPRODUCTION,
    ExpectedReproduction,
    expected_reproduction_for_anomaly,
    reproduction_context_signature,
    same_reproduction_context,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from services.anomaly_detection_service import detect_anomalies


EXCLUDED_ACTIONS = {
    "inspect_server_health", "inspect_port_status", "inspect_latency",
    "inspect_server_logs", "inspect_runtime_metrics",
}

DQN_REWARD_CONFIG: dict[str, float] = {
    "new_local_state": 0.20,
    "new_local_action_target": 0.20,
    "flow_progress": 0.30,
    "candidate_anomaly": 0.70,
    "confirmed_anomaly": 2.00,
    "confirmation_upgrade": 1.30,
    "repeat_action_target": -0.30,
    "pending_reproduction_repeat": -0.05,
    "repeat_state": -0.15,
    "repeat_anomaly": -0.50,
    "no_effect": -0.30,
    "invalid_action": -0.50,
    "action_error": -0.50,
    "timeout": -0.50,
    "positive_cap": 3.00,
    "negative_floor": -1.50,
}

DQN_REPEAT_WINDOW = 4
DQN_REPEAT_MAX_CYCLE = 3


def _is_factorized_architecture(architecture: str) -> bool:
    return architecture in {
        FACTORIZED_DQN_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    }


def _dqn_state(env: BrowserGymJAWSEnv, raw: Mapping[str, Any], history: Mapping[str, Any], architecture: str) -> tuple[np.ndarray, dict[str, Any]]:
    base = env.observation_encoder.encode_observation(raw)
    if architecture not in {FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
        return base, {"pending_confirmation_exists": False, "matching_candidate_slots": []}
    context, metadata = build_dqn_confirmation_context(history, raw, env.action_space)
    if architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE:
        history_context, history_metadata = build_dqn_history_context(
            history, raw, env.action_space.action_types, env.action_space.max_candidates
        )
        metadata["history_context"] = history_metadata
        return augment_dqn_history_observation(base, context, history_context), metadata
    if architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE:
        history_context, history_metadata = build_dqn_history_context_v2(
            history, raw, env.action_space.action_types, env.action_space.max_candidates
        )
        metadata["history_context"] = history_metadata
        return augment_dqn_history_observation_v2(base, context, history_context), metadata
    return augment_dqn_observation(base, context), metadata


def safe_action_mask(action_space: ActionSpace, observation: dict) -> np.ndarray:
    mask = action_space.build_action_mask(observation)
    for action_type in EXCLUDED_ACTIONS:
        for index in range(action_space.max_candidates):
            mask[action_space.encode(action_type, index)] = 0.0
    if not mask.any():
        mask[action_space.encode("noop", 0)] = 1.0
    return mask


def anomaly_reward(anomalies: list[dict]) -> float:
    """Compatibility wrapper for tests/tools that only score anomaly novelty."""
    if not anomalies:
        return 0.0
    components = _score_dqn_anomalies(anomalies, {})
    positive = (
        components["candidate_anomaly"]
        + components["confirmed_anomaly"]
        + components["confirmation_upgrade"]
    )
    return min(DQN_REWARD_CONFIG["positive_cap"], positive)


def calculate_dqn_reward(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    anomalies: list[Mapping[str, Any]],
    history: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    history = history if history is not None else {}
    # Snapshot pending confirmation records before lifecycle scoring mutates
    # them.  Upgrade credit must be attributable to the candidate that was
    # pending at the start of this exact transition.
    pre_pending_records = {
        str(key): dict(value)
        for key, value in (history.get("pending_reproduction_records", {}) or {}).items()
        if isinstance(value, Mapping)
    }
    history["last_pre_pending_records"] = {key: dict(value) for key, value in pre_pending_records.items()}
    action_type = str(action.get("action_type") or "")
    candidate = _action_candidate(before_observation, action)
    local_state = _build_functional_local_state_signature(after_observation)
    local_action_target = _local_action_target_key(action_type, candidate, action)
    components: dict[str, float] = {
        "new_local_state": 0.0,
        "new_local_action_target": 0.0,
        "flow_progress": 0.0,
        "candidate_anomaly": 0.0,
        "confirmed_anomaly": 0.0,
        "confirmation_upgrade": 0.0,
        "repeat_action_penalty": 0.0,
        "repeat_state_penalty": 0.0,
        "repeat_anomaly_penalty": 0.0,
        "no_effect_penalty": 0.0,
        "invalid_penalty": 0.0,
        "action_error_penalty": 0.0,
        "timeout_penalty": 0.0,
    }
    history["last_confirmation_upgrade_eligible"] = False
    history["last_confirmation_upgrade_component_before_gate"] = 0.0
    history["last_confirmation_upgrade_component_after_gate"] = 0.0

    if _mark_first_seen(history, "local_seen_states", local_state):
        components["new_local_state"] = DQN_REWARD_CONFIG["new_local_state"]
    else:
        components["repeat_state_penalty"] = DQN_REWARD_CONFIG["repeat_state"]

    if _mark_first_seen(history, "local_seen_action_targets", local_action_target):
        components["new_local_action_target"] = DQN_REWARD_CONFIG["new_local_action_target"]

    flow_key = _flow_progress_key(before_observation, after_observation, action, candidate)
    if flow_key and _mark_first_seen(history, "local_seen_flows", flow_key):
        components["flow_progress"] = DQN_REWARD_CONFIG["flow_progress"]

    anomaly_components = _score_dqn_anomalies(anomalies, history, before_observation, after_observation, action)
    components["candidate_anomaly"] = anomaly_components["candidate_anomaly"]
    components["confirmed_anomaly"] = anomaly_components["confirmed_anomaly"]
    components["confirmation_upgrade"] = anomaly_components["confirmation_upgrade"]
    components["repeat_anomaly_penalty"] = anomaly_components["repeat_anomaly_penalty"]
    history["last_confirmation_upgrade_component_before_gate"] = float(components["confirmation_upgrade"])

    if components["confirmation_upgrade"] > 0.0 and not _confirmation_upgrade_transition_eligible(
        history=history,
        pre_pending_records=pre_pending_records,
        before_observation=before_observation,
        after_observation=after_observation,
        action=action,
        anomalies=anomalies,
    ):
        components["confirmation_upgrade"] = 0.0
        history["confirmation_reward_attribution_rejected"] = int(
            history.get("confirmation_reward_attribution_rejected", 0) or 0
        ) + 1
    elif components["confirmation_upgrade"] > 0.0:
        history["last_confirmation_upgrade_eligible"] = True
    history["last_confirmation_upgrade_component_after_gate"] = float(components["confirmation_upgrade"])

    no_effect_diagnostic = dqn_no_effect_diagnostic(before_observation, after_observation, action)
    repeat_diagnostic = dqn_repeat_diagnostic(
        history=history,
        before_observation=before_observation,
        after_observation=after_observation,
        action=action,
        anomalies=anomalies,
        reward_components=components,
        action_target_key=local_action_target,
        no_effect_diagnostic=no_effect_diagnostic,
    )
    if repeat_diagnostic["unproductive_repeat"] or repeat_diagnostic["useful_reproduction_repeat"]:
        components["repeat_action_penalty"] = (
            DQN_REWARD_CONFIG["pending_reproduction_repeat"]
            if repeat_diagnostic["useful_reproduction_repeat"]
            else DQN_REWARD_CONFIG["repeat_action_target"]
        )
    if repeat_diagnostic["useful_reproduction_repeat"]:
        components["repeat_state_penalty"] = 0.0

    if no_effect_diagnostic["no_effect"]:
        components["no_effect_penalty"] = DQN_REWARD_CONFIG["no_effect"]

    if action.get("invalid") or action.get("mask_violation") or action.get("candidate_missing"):
        components["invalid_penalty"] = DQN_REWARD_CONFIG["invalid_action"]

    if action.get("failed") or action.get("last_action_error") or _has_action_error(anomalies):
        components["action_error_penalty"] = DQN_REWARD_CONFIG["action_error"]

    if _has_timeout(action, after_observation):
        components["timeout_penalty"] = DQN_REWARD_CONFIG["timeout"]

    # A confirmed R10 ACTION_REPLAY is the productive transition the local
    # agent is explicitly meant to discover.  The detector evidence already
    # carries the runtime semantic recurrence and settled UI contract.  Do
    # not let generic repeat/no-effect accounting relabel that same evidence
    # as an unproductive loop.  This is deliberately scoped to R10 action
    # reproduction; ordinary exploration and R01/R06 semantics are unchanged.
    if _is_productive_r10_confirmation_reproduction(
        anomalies=anomalies,
        action=action,
        history=history,
        reward_components=components,
        no_effect_diagnostic=no_effect_diagnostic,
    ):
        components["repeat_action_penalty"] = 0.0
        components["repeat_state_penalty"] = 0.0
        components["repeat_anomaly_penalty"] = 0.0
        components["no_effect_penalty"] = 0.0
        repeat_record = history.get("last_repeat_diagnostic")
        if isinstance(repeat_record, dict):
            repeat_record["productive_confirmation_reproduction"] = True
            repeat_record["repeat_penalty_applied"] = False
            repeat_record["no_effect_suppressed_by_confirmation"] = True

    positive_total = sum(value for value in components.values() if value > 0)
    negative_total = sum(value for value in components.values() if value < 0)
    positive_capped = min(DQN_REWARD_CONFIG["positive_cap"], positive_total)
    negative_capped = max(DQN_REWARD_CONFIG["negative_floor"], negative_total)
    total = positive_capped + negative_capped
    components["positive_total"] = positive_total
    components["positive_capped"] = positive_capped
    components["negative_total"] = negative_total
    components["negative_capped"] = negative_capped
    components["final_reward"] = total
    _remember_reward_history(history, local_action_target, local_state)
    return total, components


def _confirmation_upgrade_transition_eligible(
    *,
    history: Mapping[str, Any],
    pre_pending_records: Mapping[str, Mapping[str, Any]],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    anomalies: list[Mapping[str, Any]],
) -> bool:
    """Check strict same-candidate, same-transition confirmation credit.

    The lifecycle scorer may observe confirmation events for several detector
    candidates in one post-action observation.  Reward credit is narrower:
    the event must upgrade a candidate that was pending before this action,
    and the action/evidence must belong to that candidate's reproduction
    chain.  This prevents cross-candidate and stale-event leakage.
    """
    event = history.get("last_confirmation_transition")
    if not isinstance(event, Mapping) or not bool(event.get("newly_confirmed")):
        return False
    logical_key = str(event.get("logical_key") or "")
    if not logical_key or str(event.get("previous_status") or "") != "candidate" or str(event.get("current_status") or "") != "confirmed":
        return False
    record = next(
        (row for row in pre_pending_records.values() if str(row.get("logical_key") or "") == logical_key),
        None,
    )
    if not isinstance(record, Mapping) or float(record.get("progress", 0.0) or 0.0) >= 1.0:
        return False
    # If this transition has an action-reproduction candidate pending, an
    # observation-recheck event for another candidate must not receive the
    # ACTION_REPLAY transition's upgrade credit.
    action_replay_keys = {
        str(row.get("logical_key") or "")
        for row in pre_pending_records.values()
        if isinstance(row, Mapping) and bool(row.get("requires_action_reproduction"))
    }
    if action_replay_keys and logical_key not in action_replay_keys:
        return False

    same_key = [
        anomaly for anomaly in anomalies or []
        if _candidate_reward_logical_key(anomaly) == logical_key
    ]
    if not same_key:
        return False

    mode = str(record.get("confirmation_mode") or "")
    if mode == ACTION_REPRODUCTION:
        expected_target = str(record.get("action_target") or "")
        actual_target = _local_action_target_key(
            str(action.get("action_type") or ""),
            _action_candidate(before_observation, action),
            action,
        )
        if not expected_target or actual_target != expected_target:
            return False
        # ACTION_REPLAY requires fresh, action-caused evidence for the same
        # candidate.  Generic detector confirmation alone is insufficient.
        if not any(
            (
                str(anomaly.get("status") or "") == "confirmed"
                or not bool(record.get("requires_action_reproduction"))
            )
            and str(anomaly.get("confirmation_source") or "") == "DQN_REPRODUCTION"
            and str((anomaly.get("evidence") or {}).get("relation") or "NOT_EQUAL") == "NOT_EQUAL"
            and float((anomaly.get("evidence") or {}).get("pair_confidence", 1.0) or 0.0) >= 0.9
            and _candidate_reward_attribution(anomaly, before_observation, after_observation, action)
            in {"ACTION_CAUSED", "ACTION_REVEALED"}
            for anomaly in same_key
        ):
            return False
        return True

    if mode == OBSERVATION_REPRODUCTION:
        return tuple(record.get("context_signature", ())) == reproduction_context_signature(same_key[0])

    return False


def _is_productive_r10_confirmation_reproduction(
    *,
    anomalies: list[Mapping[str, Any]],
    action: Mapping[str, Any],
    history: Mapping[str, Any],
    reward_components: Mapping[str, Any],
    no_effect_diagnostic: Mapping[str, Any],
) -> bool:
    """Return true only for runtime-confirmed R10 action replay evidence.

    This uses detector/runtime evidence only.  It intentionally does not
    inspect a site, endpoint, taxonomy catalog, or expected bug metadata.
    """
    if float(reward_components.get("confirmation_upgrade", 0.0) or 0.0) <= 0.0:
        return False
    if not bool(history.get("last_confirmation_confirmed")) and float(reward_components.get("confirmation_upgrade", 0.0) or 0.0) <= 0.0:
        return False
    for anomaly in anomalies or []:
        if str(anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id") or "") != "R10":
            continue
        if str(anomaly.get("status") or "") not in {"candidate", "confirmed"}:
            continue
        if str(anomaly.get("reproduction_strategy") or "") != "ACTION_REPLAY":
            continue
        if not bool(anomaly.get("requires_action_reproduction")):
            continue
        evidence = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), Mapping) else {}
        if str(evidence.get("relation") or "NOT_EQUAL") != "NOT_EQUAL":
            continue
        try:
            confidence = float(evidence.get("pair_confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.9 or not bool(evidence.get("ui_settled", True)):
            continue
        if not (bool(no_effect_diagnostic.get("meaningful_action_effect")) or evidence.get("source") == "network_ui_semantic"):
            continue
        return True
    return False


def build_replay_metadata(
    *,
    confirmation_context: bool,
    action_replay: bool,
    reproduction_attempt: bool,
    candidate_relevant_effect: bool,
    same_chain_recurrence: bool,
    confirmation_upgrade: bool,
) -> dict[str, bool]:
    """Classify replay strata from runtime lifecycle signals only."""
    successful = bool(
        action_replay
        and reproduction_attempt
        and candidate_relevant_effect
        and same_chain_recurrence
        and confirmation_upgrade
    )
    failed = bool(reproduction_attempt and not confirmation_upgrade)
    partial = bool(reproduction_attempt and not successful and (candidate_relevant_effect or same_chain_recurrence))
    return {
        "is_confirmation_context": bool(confirmation_context),
        "is_action_replay": bool(action_replay),
        "is_reproduction_attempt": bool(reproduction_attempt),
        "is_candidate_relevant_effect": bool(candidate_relevant_effect),
        "is_same_chain_recurrence": bool(same_chain_recurrence),
        "is_confirmation_upgrade": bool(confirmation_upgrade),
        "is_successful_reproduction": successful,
        "is_failed_reproduction": failed,
        "is_partial_reproduction": partial,
    }


def initialize_dqn_candidate_reward_baseline(
    history: dict[str, Any],
    anomalies: list[Mapping[str, Any]] | None,
) -> None:
    """Register candidate anomalies already present before the first action.

    This only affects DQN candidate-discovery reward eligibility. Detector
    signatures, confirmed rewards, and confirmation tracking remain unchanged.
    """
    keys = {
        _candidate_reward_logical_key(anomaly)
        for anomaly in anomalies or []
        if str(anomaly.get("status") or "candidate") == "candidate"
    }
    keys.discard("")
    history["pre_action_candidate_keys"] = set(keys)
    history["initial_preexisting_candidate_keys"] = set(keys)
    history.setdefault("seen_reward_candidate_keys", set()).update(keys)
    history.setdefault("pending_reproduction_records", {})
    records = history["pending_reproduction_records"]
    for anomaly in anomalies or []:
        if str(anomaly.get("status") or "candidate") != "candidate":
            continue
        if str(anomaly.get("confirmation_policy") or "") != "reproduce":
            continue
        signature = str(anomaly.get("signature") or "")
        instruction = expected_reproduction_for_anomaly(anomaly)
        if signature and instruction is not None:
            action_target = (
                f"{instruction.action_type}:{instruction.target_identity}"
                if instruction.mode == ACTION_REPRODUCTION
                else ""
            )
            records[signature] = {
                "signature": signature,
                "logical_key": _candidate_reward_logical_key(anomaly),
                "rule_id": str(anomaly.get("rule_id") or ""),
                "action_target": action_target,
                "confirmation_mode": instruction.mode,
                "expected_action_type": instruction.action_type,
                "expected_target_identity": instruction.target_identity,
                "evidence_target_identity": anomaly.get("evidence_target_identity"),
                "trigger_action_type": anomaly.get("trigger_action_type"),
                "trigger_target_identity": anomaly.get("trigger_target_identity"),
                "reproduction_strategy": anomaly.get("reproduction_strategy"),
                "reproduction_action_type": anomaly.get("reproduction_action_type"),
                "reproduction_target_identity": anomaly.get("reproduction_target_identity"),
                "requires_action_reproduction": bool(anomaly.get("requires_action_reproduction")),
                "context_requirements": list(instruction.context_requirements),
                "context_signature": list(reproduction_context_signature(anomaly)),
                "observation_count": 1,
                "reproduction_count": 0,
                "progress": 0.0,
                "attempt_count": 0,
                "created_step": 0,
            }
def _score_dqn_anomalies(
    anomalies: list[Mapping[str, Any]],
    history: dict[str, Any],
    before_observation: Mapping[str, Any] | None = None,
    after_observation: Mapping[str, Any] | None = None,
    action: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    components = {
        "candidate_anomaly": 0.0,
        "confirmed_anomaly": 0.0,
        "confirmation_upgrade": 0.0,
        "repeat_anomaly_penalty": 0.0,
    }
    seen_candidates = history.setdefault("seen_candidate_signatures", set())
    seen_confirmed = history.setdefault("seen_confirmed_signatures", set())
    seen_reward_candidates = history.setdefault("seen_reward_candidate_keys", set())
    pre_action_candidate_keys = set(history.get("pre_action_candidate_keys", set()) or set())
    current_candidate_keys: set[str] = set()
    pending = history.setdefault("pending_reproduction_signatures", set())
    pending_targets = history.setdefault("pending_reproduction_action_targets", set())
    pending_records = history.setdefault("pending_reproduction_records", {})
    rewarded_confirmed_logical_keys = history.setdefault("rewarded_confirmed_logical_keys", set())
    history["last_confirmation_progress"] = False
    history["last_confirmation_confirmed"] = False
    history["last_confirmation_mode"] = ""
    history["last_confirmation_transition"] = None
    history["last_confirmation_transition_events"] = []
    history["last_confirmation_observation_events"] = 0
    history["last_confirmation_reward_duplicate_attempts"] = 0
    history["last_confirmation_reward_duplicate_blocks"] = 0
    for anomaly in anomalies or []:
        signature = str(anomaly.get("signature") or "")
        if not signature:
            continue
        status = str(anomaly.get("status") or "candidate")
        instruction = expected_reproduction_for_anomaly(anomaly)
        action_target = (
            f"{instruction.action_type}:{instruction.target_identity}"
            if instruction is not None and instruction.mode == ACTION_REPRODUCTION
            else ""
        )
        logical_key = _candidate_reward_logical_key(anomaly)
        confirmation_key = logical_key or signature
        if status == "candidate" and confirmation_key in rewarded_confirmed_logical_keys:
            history["last_confirmation_observation_events"] += 1
            history["confirmed_observation_count"] = int(history.get("confirmed_observation_count", 0) or 0) + 1
            history["last_confirmation_reward_duplicate_attempts"] += 1
            history["last_confirmation_reward_duplicate_blocks"] += 1
        existing_record = next(
            (
                record for record in pending_records.values()
                if isinstance(record, Mapping)
                and record.get("logical_key") == logical_key
                and record.get("confirmation_mode") == (instruction.mode if instruction else None)
            ),
            None,
        )
        semantic_confirmation = False
        if status == "candidate" and instruction is not None and existing_record is not None:
            current_context = reproduction_context_signature(anomaly)
            action_matches = bool(
                instruction.mode == ACTION_REPRODUCTION
                and action_target
                and action_target == _local_action_target_key(
                    str((action or {}).get("action_type") or ""),
                    _action_candidate(before_observation or {}, action or {}),
                    action or {},
                )
            )
            context_matches = bool(
                instruction.mode == OBSERVATION_REPRODUCTION
                and tuple(existing_record.get("context_signature", ())) == current_context
            )
            if action_matches or context_matches:
                if action_matches:
                    existing_record["reproduction_count"] = int(existing_record.get("reproduction_count", 0) or 0) + 1
                if context_matches:
                    existing_record["observation_count"] = int(existing_record.get("observation_count", 1) or 1) + 1
                if int(existing_record.get("reproduction_count", 0) or 0) >= 1 or int(existing_record.get("observation_count", 0) or 0) >= 2:
                    progress_before = float(existing_record.get("progress", 0.0) or 0.0)
                    existing_record["progress"] = 1.0
                    existing_record["confirmed"] = True
                    semantic_confirmation = True
                    newly_confirmed = confirmation_key not in rewarded_confirmed_logical_keys
                    history["last_confirmation_progress"] = True
                    history["last_confirmation_confirmed"] = True
                    history["last_confirmation_mode"] = str(existing_record.get("confirmation_mode") or "")
                    history["confirmed_observation_count"] = int(history.get("confirmed_observation_count", 0) or 0) + 1
                    history["last_confirmation_observation_events"] += 1
                    history["newly_confirmed_transition_count"] = int(history.get("newly_confirmed_transition_count", 0) or 0) + int(newly_confirmed)
                    history["last_confirmation_transition"] = {
                        "logical_key": confirmation_key,
                        "rule_id": str(existing_record.get("rule_id") or ""),
                        "mode": str(existing_record.get("confirmation_mode") or ""),
                        "previous_status": "candidate",
                        "current_status": "confirmed",
                        "progress_before": progress_before,
                        "progress_after": 1.0,
                        "newly_confirmed": bool(newly_confirmed),
                    }
                    history["last_confirmation_transition_events"].append(dict(history["last_confirmation_transition"]))
                    if newly_confirmed:
                        rewarded_confirmed_logical_keys.add(confirmation_key)
                        components["confirmation_upgrade"] += DQN_REWARD_CONFIG["confirmation_upgrade"]
                        history["confirmed_reward_event_count"] = int(history.get("confirmed_reward_event_count", 0) or 0) + 1
                    pending.discard(signature)
                    if action_target:
                        pending_targets.discard(action_target)
                    for pending_signature, record in list(pending_records.items()):
                        if pending_signature == signature or record.get("logical_key") == existing_record.get("logical_key"):
                            pending_records.pop(pending_signature, None)
        if status == "confirmed":
            history["last_confirmation_observation_events"] += 1
            confirmation_key = logical_key or signature
            if confirmation_key in rewarded_confirmed_logical_keys:
                history["confirmed_observation_count"] = int(history.get("confirmed_observation_count", 0) or 0) + 1
                history["last_confirmation_reward_duplicate_attempts"] += 1
                history["last_confirmation_reward_duplicate_blocks"] += 1
                if signature in seen_confirmed:
                    components["repeat_anomaly_penalty"] += DQN_REWARD_CONFIG["repeat_anomaly"]
                else:
                    seen_confirmed.add(signature)
                continue
            if signature in seen_confirmed:
                components["repeat_anomaly_penalty"] += DQN_REWARD_CONFIG["repeat_anomaly"]
                continue
            seen_confirmed.add(signature)
            previous_status = "candidate" if existing_record is not None or signature in seen_candidates else "new"
            if previous_status == "candidate":
                components["confirmation_upgrade"] += DQN_REWARD_CONFIG["confirmation_upgrade"]
            else:
                components["confirmed_anomaly"] += DQN_REWARD_CONFIG["confirmed_anomaly"]
            rewarded_confirmed_logical_keys.add(confirmation_key)
            history["confirmed_observation_count"] = int(history.get("confirmed_observation_count", 0) or 0) + 1
            history["newly_confirmed_transition_count"] = int(history.get("newly_confirmed_transition_count", 0) or 0) + 1
            history["confirmed_reward_event_count"] = int(history.get("confirmed_reward_event_count", 0) or 0) + 1
            history["last_confirmation_transition"] = {
                "logical_key": confirmation_key,
                "rule_id": str(anomaly.get("rule_id") or ""),
                "mode": str(existing_record.get("confirmation_mode") or "") if existing_record else "",
                "previous_status": previous_status,
                "current_status": "confirmed",
                "progress_before": float(existing_record.get("progress", 0.0) or 0.0) if existing_record else 0.0,
                "progress_after": 1.0,
                "newly_confirmed": True,
            }
            history["last_confirmation_transition_events"].append(dict(history["last_confirmation_transition"]))
            pending.discard(signature)
            if action_target:
                pending_targets.discard(action_target)
            for pending_signature, record in list(pending_records.items()):
                if pending_signature == signature or (
                    existing_record is not None and record.get("logical_key") == existing_record.get("logical_key")
                ):
                    pending_records.pop(pending_signature, None)
            continue
        if logical_key:
            current_candidate_keys.add(logical_key)
        if signature not in seen_candidates:
            seen_candidates.add(signature)
        if (
            logical_key
            and logical_key not in pre_action_candidate_keys
            and logical_key not in seen_reward_candidates
            and _candidate_reward_attribution(
                anomaly,
                before_observation or {},
                after_observation or {},
                action or {},
            )
            in {"ACTION_CAUSED", "ACTION_REVEALED"}
        ):
            seen_reward_candidates.add(logical_key)
            components["candidate_anomaly"] += DQN_REWARD_CONFIG["candidate_anomaly"]
        if (
            str(anomaly.get("confirmation_policy") or "") == "reproduce"
            and instruction is not None
            and not semantic_confirmation
            and (logical_key or signature) not in rewarded_confirmed_logical_keys
        ):
            pending.add(signature)
            if action_target:
                pending_targets.add(action_target)
            pending_records[signature] = {
                "signature": signature,
                "logical_key": logical_key,
                "rule_id": str(anomaly.get("rule_id") or ""),
                "action_target": action_target,
                "confirmation_mode": instruction.mode,
                "expected_action_type": instruction.action_type,
                "expected_target_identity": instruction.target_identity,
                "context_requirements": list(instruction.context_requirements),
                "context_signature": list(reproduction_context_signature(anomaly)),
                "observation_count": 1,
                "reproduction_count": 0,
                "progress": 0.0,
                "attempt_count": int(pending_records.get(signature, {}).get("attempt_count", 0) or 0),
                "created_step": len(history.get("dqn_action_target_history", []) or []),
            }
    history["pre_action_candidate_keys"] = current_candidate_keys
    return components


def sanitize_anomalies(anomalies: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    blocked_keys = {
        "catalog_bug_id_matches",
        "data_bug_id",
        "clicked_data_bug_id",
        "selector_hint",
        "action_hints",
        "matched_bug_id",
    }
    for anomaly in anomalies:
        item = {key: value for key, value in anomaly.items() if key not in blocked_keys}
        evidence = item.get("evidence")
        if isinstance(evidence, dict):
            item["evidence"] = {key: value for key, value in evidence.items() if key not in blocked_keys}
        sanitized.append(item)
    return sanitized


def _mark_first_seen(history: dict[str, Any], key_name: str, value: str) -> bool:
    if not value:
        return False
    seen = history.setdefault(key_name, set())
    if value in seen:
        return False
    seen.add(value)
    return True


def _remember_reward_history(history: dict[str, Any], action_target_key: str, local_state: str) -> None:
    if action_target_key:
        history["last_dqn_action_target_key"] = action_target_key
        history.setdefault("dqn_action_target_history", []).append(action_target_key)
        for record in (history.get("pending_reproduction_records", {}) or {}).values():
            if isinstance(record, Mapping) and record.get("action_target") == action_target_key:
                record["attempt_count"] = int(record.get("attempt_count", 0) or 0) + 1
    if local_state:
        history["last_dqn_local_state"] = local_state


def _is_repeated_action_target(history: Mapping[str, Any], action_target_key: str) -> bool:
    return bool(action_target_key and history.get("last_dqn_action_target_key") == action_target_key)


def dqn_repeat_diagnostic(
    *,
    history: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    anomalies: list[Mapping[str, Any]],
    reward_components: Mapping[str, Any],
    action_target_key: str,
    no_effect_diagnostic: Mapping[str, Any] | None = None,
    window: int = DQN_REPEAT_WINDOW,
    max_cycle: int = DQN_REPEAT_MAX_CYCLE,
) -> dict[str, Any]:
    action_history = list(history.get("dqn_action_target_history", []) or [])
    current_sequence = action_history + ([action_target_key] if action_target_key else [])
    immediate = bool(action_target_key and action_history and action_history[-1] == action_target_key)
    recent = bool(action_target_key and action_target_key in action_history[max(0, len(action_history) - window):])
    cycle_length = _cycle_repeat_length(current_sequence, max_cycle=max_cycle)
    cycle = cycle_length > 0
    repeat_kinds: list[str] = []
    if immediate:
        repeat_kinds.append("IMMEDIATE")
    if cycle:
        repeat_kinds.append(f"CYCLE_{cycle_length}")
    if recent:
        repeat_kinds.append("RECENT_WINDOW")

    no_effect_diagnostic = no_effect_diagnostic or dqn_no_effect_diagnostic(before_observation, after_observation, action)
    productive_reasons = _productive_transition_reasons(
        action=action,
        reward_components=reward_components,
        no_effect_diagnostic=no_effect_diagnostic,
    )
    pending_reproduction = _pending_reproduction_for_action(history, anomalies, action_target_key)
    reproduction_target_match = _reproduction_target_matches(anomalies, action_target_key) or bool(
        action_target_key and action_target_key in set(history.get("pending_reproduction_action_targets", set()) or set())
    )
    confirmation_progress = _confirmation_progress(reward_components, anomalies)
    repeat_event = bool(immediate or recent or cycle)
    if repeat_event:
        productive_reasons = [
            reason
            for reason in productive_reasons
            if reason not in {"NEW_LOCAL_ACTION_TARGET", "NEW_FUNCTIONAL_LOCAL_STATE"}
        ]
    useful_reproduction = bool(
        repeat_event
        and reproduction_target_match
        and confirmation_progress
        and (pending_reproduction or _has_reproduce_policy_for_action(anomalies, action_target_key))
    )
    unproductive_repeat = bool(repeat_event and not useful_reproduction and not productive_reasons)

    attempt_count = 0
    if repeat_event and reproduction_target_match:
        attempts = history.setdefault("reproduction_attempt_counts", {})
        if isinstance(attempts, dict):
            attempt_count = int(attempts.get(action_target_key, 0) or 0) + 1
            attempts[action_target_key] = attempt_count

    diagnostic = {
        "action_identity": action_target_key,
        "meaningful_action_effect": bool(no_effect_diagnostic.get("meaningful_action_effect")),
        "no_effect": bool(no_effect_diagnostic.get("no_effect")),
        "recent_action_window": action_history[max(0, len(action_history) - window):],
        "immediate_repeat": immediate,
        "recent_window_repeat": recent,
        "cycle_repeat": cycle,
        "cycle_length": cycle_length,
        "repeat_kinds": repeat_kinds,
        "productive_transition": bool(productive_reasons),
        "productive_reasons": productive_reasons,
        "pending_reproduction": bool(pending_reproduction),
        "reproduction_target_match": bool(reproduction_target_match),
        "reproduction_attempt": bool(repeat_event and reproduction_target_match),
        "reproduction_attempt_count": attempt_count,
        "confirmation_progress": bool(confirmation_progress),
        "useful_reproduction_repeat": useful_reproduction,
        "unproductive_repeat": unproductive_repeat,
        "repeat_penalty_applied": bool(unproductive_repeat or useful_reproduction),
    }
    if isinstance(history, dict):
        history["last_repeat_diagnostic"] = diagnostic
    return diagnostic


def _cycle_repeat_length(action_targets: list[str], *, max_cycle: int = DQN_REPEAT_MAX_CYCLE) -> int:
    return cycle_repeat_length(action_targets, max_cycle=max_cycle)


def _productive_transition_reasons(
    *,
    action: Mapping[str, Any],
    reward_components: Mapping[str, Any],
    no_effect_diagnostic: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for component, label in (
        ("new_local_state", "NEW_FUNCTIONAL_LOCAL_STATE"),
        ("new_local_action_target", "NEW_LOCAL_ACTION_TARGET"),
        ("flow_progress", "FLOW_PROGRESS"),
        ("candidate_anomaly", "NEW_ELIGIBLE_CANDIDATE"),
        ("confirmed_anomaly", "NEW_CONFIRMED_ANOMALY"),
        ("confirmation_upgrade", "CANDIDATE_TO_CONFIRMED"),
    ):
        if float(reward_components.get(component, 0.0) or 0.0) > 0.0:
            reasons.append(label)
    for key, label in (
        ("url_changed", "URL_CHANGED"),
        ("functional_state_changed", "FUNCTIONAL_STATE_CHANGED"),
        ("feedback_changed", "FEEDBACK_CHANGED"),
        ("network_effect", "NETWORK_EFFECT"),
        ("target_state_changed", "TARGET_STATE_CHANGED"),
    ):
        if bool(no_effect_diagnostic.get(key)):
            reasons.append(label)
    action_type = str(action.get("action_type") or "")
    effect_reasons = set(str(item) for item in no_effect_diagnostic.get("effect_reasons", []) or [])
    if action_type in {"click_element", "fill_input", "press_enter"} and "MODAL_DIALOG_CHANGED" in effect_reasons:
        reasons.append("MODAL_DIALOG_CHANGED")
    return reasons


def _confirmation_progress(reward_components: Mapping[str, Any], anomalies: list[Mapping[str, Any]]) -> bool:
    if float(reward_components.get("confirmed_anomaly", 0.0) or 0.0) > 0.0:
        return True
    if float(reward_components.get("confirmation_upgrade", 0.0) or 0.0) > 0.0:
        return True
    return False


def _reproduction_target_matches(anomalies: list[Mapping[str, Any]], action_target_key: str) -> bool:
    if not action_target_key:
        return False
    return any(
        str(item.get("confirmation_policy") or "") == "reproduce"
        and _anomaly_action_target_key(item) == action_target_key
        for item in anomalies or []
    )


def _has_reproduce_policy_for_action(anomalies: list[Mapping[str, Any]], action_target_key: str) -> bool:
    return _reproduction_target_matches(anomalies, action_target_key)


def _pending_reproduction_for_action(
    history: Mapping[str, Any],
    anomalies: list[Mapping[str, Any]],
    action_target_key: str,
) -> bool:
    if not action_target_key:
        return False
    pending_targets = history.get("pending_reproduction_action_targets", set())
    if action_target_key in pending_targets:
        return True
    return any(
        str(item.get("status") or "") == "candidate"
        and str(item.get("confirmation_policy") or "") == "reproduce"
        and _anomaly_action_target_key(item) == action_target_key
        for item in anomalies or []
    )


def _action_candidate(observation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if action.get("action_type") not in {"click_element", "fill_input", "press_enter"}:
        return None
    candidates = observation.get("candidate_elements", []) or []
    try:
        index = int(action.get("candidate_index", 0) or 0)
    except (TypeError, ValueError):
        return None
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


FUNCTIONAL_LOCAL_STATE_KEYS = (
    "cart_count",
    "form_state",
    "modal_state",
    "validation_state",
    "selected_filter",
    "current_page",
    "quantity",
    "status",
    "selected_option",
    "selected_tab",
    "auth_state",
    "login_state",
    "entity_id",
    "product_id",
    "item_id",
    "post_id",
    "order_id",
    "user_id",
)


def _local_state_signature(observation: Mapping[str, Any]) -> str:
    return _build_functional_local_state_signature(observation)


def _build_functional_local_state_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    local_bits = []
    for key in FUNCTIONAL_LOCAL_STATE_KEYS:
        value = page_state.get(key)
        if value not in (None, ""):
            local_bits.append(f"{key}={_stable_value(value)}")
    return "|".join(
        [
            _url_path(observation),
            str(page_state.get("title") or ""),
            ",".join(local_bits),
        ]
    )


def _build_viewport_context_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    candidates = observation.get("candidate_elements", []) or []
    candidate_summary: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates[:8]:
            if isinstance(candidate, Mapping):
                bbox = candidate.get("bbox", [])
                bbox_key = ""
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    bbox_key = ",".join(str(int(float(value or 0.0))) for value in bbox[:4])
                candidate_summary.append(
                    "|".join(
                        [
                            _target_identity(candidate) or str(candidate.get("role") or candidate.get("tag") or ""),
                            str(candidate.get("visible", "")),
                            bbox_key,
                        ]
                    )
                )
    return "|".join(
        [
            _url_path(observation),
            str(page_state.get("viewport_width") or ""),
            str(page_state.get("viewport_height") or ""),
            str(page_state.get("viewport_type") or ""),
            _bucket_numeric(page_state.get("page_text_length"), 100),
            ",".join(candidate_summary),
        ]
    )


def _build_raw_local_state_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    candidates = observation.get("candidate_elements", []) or []
    candidate_summary: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates[:8]:
            if isinstance(candidate, Mapping):
                candidate_summary.append(_target_identity(candidate) or str(candidate.get("role") or candidate.get("tag") or ""))
    local_bits = []
    for key in FUNCTIONAL_LOCAL_STATE_KEYS:
        value = page_state.get(key)
        if value not in (None, ""):
            local_bits.append(f"{key}={_stable_value(value)}")
    return "|".join(
        [
            _url_path(observation),
            str(page_state.get("title") or ""),
            str(page_state.get("dom_node_count") or ""),
            str(page_state.get("page_text_length") or ""),
            ",".join(candidate_summary),
            ",".join(local_bits),
        ]
    )


def _stable_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _bucket_numeric(value: Any, bucket_size: int) -> str:
    try:
        numeric = int(float(value or 0.0))
    except (TypeError, ValueError):
        return ""
    if bucket_size <= 0:
        return str(numeric)
    return str((numeric // bucket_size) * bucket_size)


def _local_action_target_key(action_type: str, candidate: Mapping[str, Any] | None, action: Mapping[str, Any]) -> str:
    target = _target_identity(candidate)
    if target:
        return f"{action_type}:{target}"
    return f"{action_type}:{action.get('candidate_index', 0)}" if action_type else ""


def _target_identity(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    for key in ("bid", "entity_id", "product_id", "cart_item_id", "item_id", "post_id", "comment_id", "id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    href = _url_path_from_value(candidate.get("href"))
    role = str(candidate.get("role") or "")
    tag = str(candidate.get("tag") or "")
    name = _normalize_text(str(candidate.get("name") or candidate.get("aria_label") or ""))
    if href and (role or tag or name):
        return f"fingerprint:{role}:{tag}:{href}:{name}"
    return ""


def _anomaly_action_target_key(anomaly: Mapping[str, Any]) -> str:
    action_type = str(anomaly.get("action_type") or "")
    target_id = str(anomaly.get("target_id") or "")
    if not action_type or not target_id:
        return ""
    if target_id.startswith(("bid:", "entity_id:", "product_id:", "fingerprint:")):
        return f"{action_type}:{target_id}"
    return f"{action_type}:bid:{target_id}"


def _candidate_reward_logical_key(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    target = evidence.get("target", {}) if isinstance(evidence.get("target"), Mapping) else {}
    rule_id = str(anomaly.get("rule_id") or evidence.get("rule_id") or "")
    if not rule_id:
        return ""
    url_path = _url_path_from_value(anomaly.get("url") or anomaly.get("url_path") or evidence.get("url") or evidence.get("url_path"))
    issue_type = _normalize_text(str(anomaly.get("issue_type") or evidence.get("issue_type") or anomaly.get("type") or "candidate"))
    for source in (anomaly, evidence, target):
        if not isinstance(source, Mapping):
            continue
        for key in ("entity_id", "product_id", "item_id", "post_id", "order_id", "user_id", "cart_item_id"):
            value = source.get(key)
            if value not in (None, ""):
                return "|".join([rule_id, url_path, issue_type, f"{key}:{value}"])
    href = _url_path_from_value(evidence.get("clicked_href") or target.get("href") or anomaly.get("href"))
    if href:
        return "|".join([rule_id, url_path, issue_type, f"href:{href}"])
    target_id = str(
        anomaly.get("target_id")
        or evidence.get("target_id")
        or evidence.get("clicked_bid")
        or evidence.get("target_bid")
        or target.get("bid")
        or ""
    ).strip()
    if target_id:
        return "|".join([rule_id, url_path, issue_type, f"target:{target_id}"])
    text = _normalize_text(str(evidence.get("clicked_name") or evidence.get("clicked_text") or target.get("name") or target.get("text") or ""))
    role = _normalize_text(str(target.get("role") or evidence.get("clicked_role") or ""))
    tag = _normalize_text(str(target.get("tag") or evidence.get("clicked_tag") or ""))
    return "|".join([rule_id, url_path, issue_type, role, tag, text[:80]])


def _candidate_reward_attribution(
    anomaly: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
) -> str:
    rule_id = str(anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id") or "")
    action_type = str(action.get("action_type") or "")
    before_functional = _build_functional_local_state_signature(before_observation)
    after_functional = _build_functional_local_state_signature(after_observation)
    before_raw = _build_raw_local_state_signature(before_observation)
    after_raw = _build_raw_local_state_signature(after_observation)
    before_viewport = _build_viewport_context_signature(before_observation)
    after_viewport = _build_viewport_context_signature(after_observation)
    functional_changed = bool(before_functional != after_functional)
    raw_changed = bool(before_raw != after_raw)
    viewport_changed = bool(before_viewport != after_viewport)
    if rule_id == "R01" and action_type == "click_element" and _r01_targets_current_action(anomaly, before_observation, action):
        return "ACTION_CAUSED"
    if rule_id in {"R02", "R08", "R09", "R10", "R11", "R13", "R14", "R15"} and (functional_changed or raw_changed):
        return "ACTION_CAUSED"
    if rule_id in {"R04", "R05", "R06", "R07", "R12"} and _action_can_reveal_candidate(
        action_type,
        functional_changed=functional_changed,
        raw_changed=raw_changed,
        viewport_changed=viewport_changed,
        after_observation=after_observation,
    ):
        return "ACTION_REVEALED"
    return "UNKNOWN"


def _r01_targets_current_action(
    anomaly: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    action: Mapping[str, Any],
) -> bool:
    candidate = _action_candidate(before_observation, action)
    candidate_target = _target_identity(candidate)
    anomaly_target = _anomaly_action_target_key(anomaly)
    if candidate_target and anomaly_target == f"{action.get('action_type')}:{candidate_target}":
        return True
    target_id = str(anomaly.get("target_id") or "").strip()
    if candidate and target_id:
        raw_bid = str(candidate.get("bid") or "").strip()
        return target_id == raw_bid or target_id == f"bid:{raw_bid}"
    return False


def _action_can_reveal_candidate(
    action_type: str,
    *,
    functional_changed: bool,
    raw_changed: bool,
    viewport_changed: bool,
    after_observation: Mapping[str, Any],
) -> bool:
    if action_type in {"scroll_down", "scroll_up", "change_viewport_mobile", "change_viewport_desktop"}:
        return bool(viewport_changed or raw_changed)
    if action_type == "click_element":
        return bool(functional_changed or _has_dialog_or_modal(after_observation))
    return bool(functional_changed)


def _has_dialog_or_modal(observation: Mapping[str, Any]) -> bool:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    if page_state.get("modal_state") or page_state.get("dialog_open"):
        return True
    candidates = observation.get("candidate_elements", []) or []
    return any(
        isinstance(candidate, Mapping)
        and str(candidate.get("role") or "").lower() in {"dialog", "alertdialog"}
        and bool(candidate.get("visible", True))
        for candidate in candidates
        if isinstance(candidate, Mapping)
    )


def _feedback_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals"), Mapping) else {}
    values = []
    for source in (page_state, runtime):
        for key in (
            "feedback",
            "feedback_message",
            "toast",
            "toast_message",
            "alert",
            "error_message",
            "success_message",
            "validation_state",
            "status_message",
        ):
            value = source.get(key)
            if value not in (None, ""):
                values.append(f"{key}={_stable_value(value)}")
    return "|".join(values)


def _network_effect(observation: Mapping[str, Any]) -> bool:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals"), Mapping) else {}
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    for source in (runtime, page_state):
        for key in (
            "network_activity",
            "network_request_count",
            "request_count",
            "response_count",
            "last_response_status",
            "api_status",
            "runtime_error_count",
            "console_error_count",
        ):
            value = source.get(key)
            if isinstance(value, (int, float)) and float(value) != 0.0:
                return True
            if isinstance(value, (list, tuple, dict)) and bool(value):
                return True
            if isinstance(value, str) and value.strip():
                return True
    entries = runtime.get("network_entries") or runtime.get("requests") or page_state.get("network_entries")
    return bool(entries)


def _target_state_changed(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
) -> bool:
    candidate = _action_candidate(before_observation, action)
    target = _target_identity(candidate)
    if not target:
        return False
    after_candidate = _candidate_by_target(after_observation, target)
    if not after_candidate:
        return False
    return _candidate_state_signature(candidate) != _candidate_state_signature(after_candidate)


def _candidate_by_target(observation: Mapping[str, Any], target: str) -> Mapping[str, Any] | None:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, Mapping) and _target_identity(candidate) == target:
            return candidate
    return None


def _candidate_state_signature(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    keys = ("name", "text", "aria_label", "value", "checked", "selected", "disabled", "enabled", "expanded", "pressed")
    return "|".join(f"{key}={_stable_value(candidate.get(key))}" for key in keys if candidate.get(key) not in (None, ""))


def _viewport_mode_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return "|".join(
        str(page_state.get(key) or "")
        for key in ("viewport_type", "viewport_width", "viewport_height")
    )


def _scroll_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return "|".join(str(page_state.get(key) or "") for key in ("scroll_y", "scroll_x", "scroll_top", "scroll_left"))


def _dialog_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    dialogs = []
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and str(candidate.get("role") or "").lower() in {"dialog", "alertdialog"}:
                dialogs.append(_target_identity(candidate) or _normalize_text(str(candidate.get("name") or candidate.get("text") or "")))
    return "|".join([_stable_value(page_state.get("modal_state")), _stable_value(page_state.get("dialog_open")), ",".join(sorted(dialogs))])


def _flow_progress_key(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> str:
    for source in (action, after_observation.get("runtime_signals", {}), after_observation.get("page_state", {})):
        if not isinstance(source, Mapping):
            continue
        for key in ("flow_progress", "flow_transition", "semantic_transition"):
            value = source.get(key)
            if value:
                return str(value if not isinstance(value, bool) else f"{key}:{_url_path(after_observation)}")
    before_path = _url_path(before_observation)
    after_path = _url_path(after_observation)
    semantic = _candidate_semantic(candidate)
    if semantic and before_path != after_path:
        return f"{semantic}:{before_path}->{after_path}"
    return ""


def _candidate_semantic(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    text = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "aria_label", "role", "tag")).lower()
    if candidate.get("is_search_related") or "search" in text:
        return "search"
    if candidate.get("is_cart_related") or candidate.get("is_purchase_action") or "cart" in text or "buy" in text:
        return "cart"
    if candidate.get("is_login_related") or "login" in text or "sign in" in text:
        return "login"
    if candidate.get("is_submit_related") or "submit" in text or "save" in text:
        return "submit"
    if "detail" in text or "product" in text:
        return "detail"
    return ""


def _has_rule(anomalies: list[Mapping[str, Any]], rule_id: str) -> bool:
    return any(str(item.get("rule_id") or item.get("evidence", {}).get("rule_id") or "") == rule_id for item in anomalies or [])


def _has_action_error(anomalies: list[Mapping[str, Any]]) -> bool:
    return any(str(item.get("type") or "") == "action-error" for item in anomalies or [])


def _has_timeout(action: Mapping[str, Any], after_observation: Mapping[str, Any]) -> bool:
    if bool(action.get("timeout") or action.get("timed_out") or action.get("timeout_occurred")):
        return True
    for section_name in ("runtime_signals", "page_state"):
        section = after_observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("timeout") or section.get("timed_out") or section.get("timeout_occurred")):
            return True
    return False


def dqn_no_effect_diagnostic(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    if action.get("invalid") or action.get("failed"):
        return {
            "meaningful_action_effect": False,
            "no_effect": False,
            "effect_reasons": [],
            "no_effect_reasons": ["EXECUTION_ERROR_OR_INVALID"],
            "execution_error": True,
        }

    action_type = str(action.get("action_type") or "")
    before_path = _url_path(before_observation)
    after_path = _url_path(after_observation)
    functional_changed = _local_state_signature(before_observation) != _local_state_signature(after_observation)
    viewport_changed = _build_viewport_context_signature(before_observation) != _build_viewport_context_signature(after_observation)
    raw_changed = _build_raw_local_state_signature(before_observation) != _build_raw_local_state_signature(after_observation)
    feedback_changed = _feedback_signature(before_observation) != _feedback_signature(after_observation)
    network_effect = _network_effect(after_observation)
    target_changed = _target_state_changed(before_observation, after_observation, action)
    viewport_mode_changed = _viewport_mode_signature(before_observation) != _viewport_mode_signature(after_observation)
    scroll_moved = _scroll_signature(before_observation) != _scroll_signature(after_observation) or viewport_changed

    effect_reasons: list[str] = []
    if before_path != after_path:
        effect_reasons.append("URL_CHANGED")
    if functional_changed:
        effect_reasons.append("FUNCTIONAL_STATE_CHANGED")
    if feedback_changed:
        effect_reasons.append("FEEDBACK_CHANGED")
    if network_effect:
        effect_reasons.append("NETWORK_EFFECT")
    if target_changed:
        effect_reasons.append("TARGET_STATE_CHANGED")

    if action_type in {"click_element", "fill_input", "press_enter"}:
        meaningful = bool(effect_reasons)
        if _has_dialog_or_modal(after_observation) and _dialog_signature(before_observation) != _dialog_signature(after_observation):
            effect_reasons.append("MODAL_DIALOG_CHANGED")
            meaningful = True
    elif action_type in {"scroll_down", "scroll_up"}:
        meaningful = bool(scroll_moved)
        if meaningful:
            effect_reasons.append("SCROLL_POSITION_OR_VIEWPORT_CHANGED")
    elif action_type in {"change_viewport_mobile", "change_viewport_desktop"}:
        meaningful = bool(viewport_mode_changed)
        if meaningful:
            effect_reasons.append("VIEWPORT_MODE_CHANGED")
    else:
        # Preserve the previous policy for inspection/noop/finish actions: these
        # do not become meaningful solely because raw visible candidates changed.
        meaningful = bool(effect_reasons)

    no_effect_reasons: list[str] = []
    if not meaningful:
        if before_path == after_path:
            no_effect_reasons.append("NO_URL_CHANGE")
        if not functional_changed:
            no_effect_reasons.append("NO_FUNCTIONAL_CHANGE")
        if not feedback_changed:
            no_effect_reasons.append("NO_FEEDBACK")
        if not network_effect:
            no_effect_reasons.append("NO_NETWORK_EFFECT")
        if not target_changed:
            no_effect_reasons.append("NO_TARGET_CHANGE")
        if raw_changed and not functional_changed:
            no_effect_reasons.append("RAW_OBSERVATION_CHANGE_IGNORED")
    return {
        "meaningful_action_effect": meaningful,
        "no_effect": not meaningful,
        "effect_reasons": effect_reasons,
        "no_effect_reasons": no_effect_reasons,
        "functional_state_changed": functional_changed,
        "url_changed": before_path != after_path,
        "feedback_changed": feedback_changed,
        "network_effect": network_effect,
        "target_state_changed": target_changed,
        "viewport_context_changed": viewport_changed,
        "raw_state_changed": raw_changed,
        "execution_error": False,
    }


def _no_effect(before_observation: Mapping[str, Any], after_observation: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    return bool(dqn_no_effect_diagnostic(before_observation, after_observation, action).get("no_effect"))


def _url_path(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return _url_path_from_value(page_state.get("url"))


def _url_path_from_value(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    parsed = urlparse(text)
    return parsed.path or text.split("?", 1)[0] or "/"


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def train(args: argparse.Namespace) -> dict:
    if _is_factorized_architecture(args.dqn_architecture):
        args.greedy_selection_mode = FACTORIZED_SELECTION
        args.td_target_selection_mode = FACTORIZED_SELECTION
    else:
        validate_selection_mode_consistency(args.greedy_selection_mode, args.td_target_selection_mode)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    args.site_id = str(site["site_id"])
    args.base_url = str(site["base_url"])
    env = BrowserGymJAWSEnv(
        site_id=args.site_id, base_url=args.base_url, max_steps=args.max_steps,
        max_candidates=args.max_candidates, headless=args.headless,
    )
    action_space = env.action_space
    base_observation_dim = env.observation_encoder.get_obs_dim()
    has_confirmation_context = args.dqn_architecture in {
        FACTORIZED_DQN_CONFOBS_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    }
    history_dim = (
        HISTORY_CONTEXT_DIM if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE
        else HISTORY_CONTEXT_V2_DIM if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE
        else 0
    )
    observation_dim = base_observation_dim + (
        CONFIRMATION_CONTEXT_DIM if has_confirmation_context else 0
    ) + history_dim
    action_dim = action_space.get_action_dim()
    print("Observation dim:", observation_dim)
    print("Action dim:", action_dim)
    print("Action types:", action_space.action_types)
    if _is_factorized_architecture(args.dqn_architecture):
        online = FactorizedQNetwork(observation_dim, len(action_space.action_types), action_space.max_candidates)
        target = FactorizedQNetwork(observation_dim, len(action_space.action_types), action_space.max_candidates)
    else:
        online = QNetwork(observation_dim, action_dim)
        target = QNetwork(observation_dim, action_dim)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay = ConfirmationAwareReplayBuffer(args.replay_capacity)
    rewards: list[float] = []
    losses: list[float] = []
    findings: dict[str, int] = {}
    reward_component_totals: dict[str, float] = {}
    episode_summaries: list[dict[str, Any]] = []
    epsilon_trajectory: list[dict[str, float]] = []
    random_action_distribution: Counter[str] = Counter()
    greedy_action_distribution: Counter[str] = Counter()
    action_distribution: Counter[str] = Counter()
    q_diagnostics: list[dict[str, Any]] = []
    reward_values: list[float] = []
    action_reward_values: dict[str, list[float]] = defaultdict(list)
    valid_action_type_counts: list[int] = []
    last_balanced_expected_distribution: dict[str, float] = {}
    loss_history: list[dict[str, float]] = []
    confirmation_training_funnel = ConfirmationFunnelAggregator()
    random_expected_type_matches = 0
    greedy_expected_type_matches = 0
    random_full_reproduction_matches = 0
    greedy_full_reproduction_matches = 0
    action_reproduction_pending_steps = 0
    observation_reproduction_pending_steps = 0
    action_reproduction_opportunities = 0
    action_reproduction_executable_opportunities = 0
    action_reproduction_safe_mask_valid = 0
    random_action_reproduction_matches = 0
    greedy_action_reproduction_matches = 0
    random_action_full_matches = 0
    greedy_action_full_matches = 0
    observation_reproduction_events = 0
    confirmation_progress_count = 0
    confirmed_count = 0
    candidate_to_confirmed_count = 0
    confirmed_logical_candidates: set[str] = set()
    confirmed_observation_count = 0
    newly_confirmed_transition_count = 0
    confirmed_reward_event_count = 0
    confirmed_reward_duplicate_attempt_count = 0
    confirmed_reward_duplicate_blocked_count = 0
    progress_reward_values: list[float] = []
    confirmed_reward_values: list[float] = []
    progress_reward_component_totals: Counter[str] = Counter()
    confirmed_reward_component_totals: Counter[str] = Counter()
    checkpoint_episodes = _parse_episode_list(getattr(args, "checkpoint_episodes", ""))
    checkpoint_prefix = Path(args.checkpoint_prefix) if getattr(args, "checkpoint_prefix", "") else None

    try:
        for episode in range(args.episodes):
            print(f"[dqn-train] starting episode {episode + 1}/{args.episodes}")
            raw, _ = env.reset()
            mask = safe_action_mask(action_space, raw)
            episode_reward = 0.0
            done = False
            history: dict[str, Any] = {}
            epsilon = max(args.epsilon_end, args.epsilon_start - episode / max(1, args.episodes))
            episode_action_distribution: Counter[str] = Counter()
            episode_random_distribution: Counter[str] = Counter()
            episode_greedy_distribution: Counter[str] = Counter()
            episode_reward_components: Counter[str] = Counter()
            episode_reward_values: list[float] = []
            episode_action_reward_values: dict[str, list[float]] = defaultdict(list)
            episode_action_component_totals: dict[str, Counter[str]] = defaultdict(Counter)
            episode_action_component_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
            episode_repeat_event_counts: Counter[str] = Counter()
            episode_action_repeat_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
            episode_step_count = 0
            episode_random_count = 0
            episode_greedy_count = 0
            episode_valid_action_type_counts: list[int] = []
            initialize_dqn_candidate_reward_baseline(
                history,
                sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}})),
            )
            state, confirmation_metadata = _dqn_state(env, raw, history, args.dqn_architecture)
            epsilon_trajectory.append({"episode": float(episode + 1), "epsilon": float(epsilon)})
            while not done:
                mask_stats = action_space.build_action_mask_stats(raw, mask)
                valid_action_type_count = int(mask_stats.get("valid_action_type_count", 0) or 0)
                valid_action_type_counts.append(valid_action_type_count)
                episode_valid_action_type_counts.append(valid_action_type_count)
                last_balanced_expected_distribution = balanced_expected_action_type_distribution(mask, action_space)
                if random.random() < epsilon:
                    action = balanced_random_valid_action(mask, action_space)
                    selection_mode = "random"
                    episode_random_count += 1
                else:
                    if _is_factorized_architecture(args.dqn_architecture):
                        action = select_factorized_greedy_action(online, state, mask, action_space)
                    else:
                        action = select_greedy_action(
                            online,
                            state,
                            mask,
                            action_space,
                            mode=args.greedy_selection_mode,
                        )
                    selection_mode = "greedy"
                    episode_greedy_count += 1
                if _is_factorized_architecture(args.dqn_architecture):
                    q_diag = factorized_q_value_diagnostics(online, state, mask, action_space, action)
                else:
                    q_diag = q_value_diagnostics(online, state, mask, action_space, action, top_k=5)
                next_raw, _, done, info = env.step(action)
                decoded = action_space.decode(action)
                decoded_for_confirmation = dict(decoded)
                action_type = str(decoded.get("action_type") or "")
                decoded["failed"] = bool(info.get("last_action_error"))
                decoded["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": decoded}))
                for anomaly in anomalies:
                    kind = str(anomaly.get("type", "unknown"))
                    findings[kind] = findings.get(kind, 0) + 1
                reward, reward_components = calculate_dqn_reward(raw, next_raw, decoded, anomalies, history)
                repeat_diagnostic = history.get("last_repeat_diagnostic", {}) if isinstance(history, Mapping) else {}
                transition_events = history.get("last_confirmation_transition_events", [])
                confirmed_observation_count += int(history.get("last_confirmation_observation_events", 0) or 0)
                if isinstance(transition_events, list):
                    newly_events = [event for event in transition_events if event.get("newly_confirmed")]
                    newly_confirmed_transition_count += len(newly_events)
                    confirmed_reward_event_count += len(newly_events)
                    confirmed_count += len(newly_events)
                    candidate_to_confirmed_count += sum(1 for event in newly_events if event.get("previous_status") == "candidate")
                    for _event in newly_events:
                        confirmed_reward_values.append(float(reward))
                        for key, value in reward_components.items():
                            if isinstance(value, (int, float)):
                                confirmed_reward_component_totals[str(key)] += float(value)
                    for event in newly_events:
                        if event.get("logical_key"):
                            confirmed_logical_candidates.add(str(event["logical_key"]))
                confirmed_reward_duplicate_attempt_count += int(history.get("last_confirmation_reward_duplicate_attempts", 0) or 0)
                confirmed_reward_duplicate_blocked_count += int(history.get("last_confirmation_reward_duplicate_blocks", 0) or 0)
                mode = str(confirmation_metadata.get("confirmation_mode") or "")
                pending = bool(confirmation_metadata.get("pending_confirmation_exists"))
                action_pending = pending and mode == ACTION_REPRODUCTION
                observation_pending = pending and mode == OBSERVATION_REPRODUCTION
                if action_pending:
                    action_reproduction_pending_steps += 1
                if observation_pending:
                    observation_reproduction_pending_steps += 1
                expected_type = str(confirmation_metadata.get("expected_action_type") or "")
                requires_target = bool(confirmation_metadata.get("expected_action_requires_target"))
                visible = bool(confirmation_metadata.get("expected_target_visible"))
                opportunity = bool(action_pending and expected_type in TARGET_ACTION_TYPES and (not requires_target or visible))
                matching_slots = {int(slot) for slot in confirmation_metadata.get("matching_candidate_slots", []) or []}
                expected_slots_valid = {
                    slot for slot in matching_slots
                    if 0 <= slot < action_space.max_candidates
                    and float(mask[int(action_space.encode(expected_type, slot))]) > 0.0
                } if opportunity else set()
                if opportunity:
                    action_reproduction_opportunities += 1
                    if expected_slots_valid:
                        action_reproduction_safe_mask_valid += 1
                        action_reproduction_executable_opportunities += 1
                selected_type = str(decoded_for_confirmation.get("action_type") or "")
                type_match = bool(opportunity and selected_type == expected_type)
                full_match = bool(type_match and (not requires_target or int(decoded_for_confirmation.get("candidate_index", -1)) in matching_slots))
                confirmation_record = confirmation_training_funnel.record(
                    confirmation_metadata,
                    decoded_for_confirmation,
                    confirmation_progress=bool(history.get("last_confirmation_progress")) and str(history.get("last_confirmation_mode") or "") == mode,
                    candidate_to_confirmed=bool(history.get("last_confirmation_confirmed")) and str(history.get("last_confirmation_mode") or "") == mode,
                    confirmed=bool(history.get("last_confirmation_confirmed")) and str(history.get("last_confirmation_mode") or "") == mode,
                )
                if bool(confirmation_record.get("opportunity")) and bool(confirmation_record.get("type_match")):
                    if selection_mode == "random":
                        random_expected_type_matches += 1
                    else:
                        greedy_expected_type_matches += 1
                if bool(confirmation_record.get("opportunity")) and bool(confirmation_record.get("full_match")):
                    if selection_mode == "random":
                        random_full_reproduction_matches += 1
                    else:
                        greedy_full_reproduction_matches += 1
                if full_match:
                    if selection_mode == "random":
                        random_action_full_matches += 1
                    else:
                        greedy_action_full_matches += 1
                if type_match:
                    if selection_mode == "random":
                        random_action_reproduction_matches += 1
                    else:
                        greedy_action_reproduction_matches += 1
                progress = bool(history.get("last_confirmation_progress"))
                confirmed = bool(history.get("last_confirmation_confirmed"))
                confirmation_mode = str(history.get("last_confirmation_mode") or "")
                mode_progress = bool(progress and confirmation_mode == mode)
                mode_confirmed = bool(confirmed and confirmation_mode == mode)
                if observation_pending and mode_progress:
                    observation_reproduction_events += 1
                if mode_progress:
                    confirmation_progress_count += 1
                    progress_reward_values.append(float(reward))
                    for key, value in reward_components.items():
                        if isinstance(value, (int, float)):
                            progress_reward_component_totals[str(key)] += float(value)
                if mode_confirmed:
                    logical_key = str(confirmation_metadata.get("pending_logical_key") or confirmation_metadata.get("active_rule_id") or "")
                    if logical_key:
                        confirmed_logical_candidates.add(logical_key)
                replay_metadata = build_replay_metadata(
                    confirmation_context=bool(pending or mode_progress or mode_confirmed),
                    action_replay=bool(mode == ACTION_REPRODUCTION),
                    reproduction_attempt=bool(action_pending or full_match or mode_progress),
                    candidate_relevant_effect=bool(
                        repeat_diagnostic.get("meaningful_action_effect")
                        or mode_progress
                        or any(bool(item.get("evidence", {}).get("ui_settled")) for item in anomalies if isinstance(item.get("evidence"), Mapping))
                    ),
                    same_chain_recurrence=bool(mode_progress),
                    confirmation_upgrade=bool(float(reward_components.get("confirmation_upgrade", 0.0) or 0.0) > 0.0),
                )
                if args.dqn_architecture in {FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
                    append_history_transition(
                        history,
                        action_type=action_type,
                        candidate=_action_candidate(raw, decoded),
                        meaningful_action_effect=not bool(repeat_diagnostic.get("no_effect")),
                        productive_transition=bool(repeat_diagnostic.get("productive_transition")),
                        confirmation_progress=mode_progress,
                        unproductive_repeat=bool(repeat_diagnostic.get("unproductive_repeat")),
                    )
                if bool(confirmation_record.get("opportunity")):
                    episode_repeat_event_counts["reproduction_opportunity"] += 1
                    if selection_mode == "random":
                        episode_repeat_event_counts["random_reproduction_opportunity"] += 1
                    else:
                        episode_repeat_event_counts["greedy_reproduction_opportunity"] += 1
                if isinstance(repeat_diagnostic, Mapping):
                    for key in (
                        "immediate_repeat",
                        "recent_window_repeat",
                        "cycle_repeat",
                        "unproductive_repeat",
                        "useful_reproduction_repeat",
                        "confirmation_progress",
                        "repeat_penalty_applied",
                    ):
                        if bool(repeat_diagnostic.get(key)):
                            episode_repeat_event_counts[key] += 1
                            episode_action_repeat_event_counts[action_type][key] += 1
                    repeat_kinds = repeat_diagnostic.get("repeat_kinds", [])
                    if isinstance(repeat_kinds, list):
                        for repeat_kind in repeat_kinds:
                            event_key = f"repeat_kind:{repeat_kind}"
                            episode_repeat_event_counts[event_key] += 1
                            episode_action_repeat_event_counts[action_type][event_key] += 1
                for key, value in reward_components.items():
                    reward_component_totals[key] = reward_component_totals.get(key, 0.0) + float(value or 0.0)
                    episode_reward_components[key] += float(value or 0.0)
                    episode_action_component_totals[action_type][key] += float(value or 0.0)
                    if float(value or 0.0) != 0.0:
                        episode_action_component_event_counts[action_type][key] += 1
                next_state, next_confirmation_metadata = _dqn_state(env, next_raw, history, args.dqn_architecture)
                next_mask = safe_action_mask(action_space, next_raw)
                replay.add(Transition(state, action, reward, next_state, done, next_mask, mask, replay_metadata))
                episode_reward += reward
                episode_step_count += 1
                reward_values.append(float(reward))
                episode_reward_values.append(float(reward))
                episode_action_reward_values[action_type].append(float(reward))
                action_reward_values[action_type].append(float(reward))
                action_distribution[action_type] += 1
                episode_action_distribution[action_type] += 1
                if selection_mode == "random":
                    random_action_distribution[action_type] += 1
                    episode_random_distribution[action_type] += 1
                else:
                    greedy_action_distribution[action_type] += 1
                    episode_greedy_distribution[action_type] += 1
                q_diagnostics.append(
                    {
                        "episode": episode + 1,
                        "step": episode_step_count,
                        "selection_mode": selection_mode,
                        "selected_action_id": int(action),
                        **q_diag,
                    }
                )
                raw, state, mask = next_raw, next_state, next_mask
                confirmation_metadata = next_confirmation_metadata
                if len(replay) >= args.batch_size:
                    batch = replay.sample(args.batch_size)
                    states = torch.from_numpy(np.stack([x.state for x in batch])).float()
                    actions = torch.tensor([x.action for x in batch]).long().unsqueeze(1)
                    batch_rewards = torch.tensor([x.reward for x in batch]).float()
                    next_states = torch.from_numpy(np.stack([x.next_state for x in batch])).float()
                    dones = torch.tensor([x.done for x in batch]).float()
                    next_masks = torch.from_numpy(np.stack([x.next_mask for x in batch])).bool()
                    current_masks = torch.from_numpy(
                        np.stack([x.current_mask if x.current_mask is not None else x.next_mask for x in batch])
                    ).bool()
                    if _is_factorized_architecture(args.dqn_architecture):
                        q_values = factorized_q_for_actions(online, states, current_masks, actions.squeeze(1), action_space)
                    else:
                        q_values = online(states).gather(1, actions).squeeze(1)
                    with torch.no_grad():
                        if _is_factorized_architecture(args.dqn_architecture):
                            next_actions = factorized_double_dqn_next_actions(
                                online,
                                next_states,
                                next_masks,
                                action_space,
                            )
                            next_q = factorized_q_for_actions(target, next_states, next_masks, next_actions.squeeze(1), action_space)
                        else:
                            next_actions = double_dqn_next_actions(
                                online,
                                next_states,
                                next_masks,
                                action_space,
                                mode=args.td_target_selection_mode,
                            )
                            next_q = target(next_states).gather(1, next_actions).squeeze(1)
                        expected = batch_rewards + args.gamma * next_q * (1.0 - dones)
                    loss = nn.functional.smooth_l1_loss(q_values, expected)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    type_grad_norm = _head_grad_norm(getattr(online, "type_head", None))
                    candidate_grad_norm = _head_grad_norm(getattr(online, "candidate_head", None))
                    nn.utils.clip_grad_norm_(online.parameters(), 10.0); optimizer.step()
                    losses.append(float(loss.detach()))
                    loss_history.append(
                        {
                            "episode": float(episode + 1),
                            "loss": float(loss.detach()),
                            "td_error_abs_mean": float(torch.mean(torch.abs(expected - q_values)).detach()),
                            "type_head_grad_norm": type_grad_norm,
                            "candidate_head_grad_norm": candidate_grad_norm,
                            "finite": bool(torch.isfinite(loss.detach()) and torch.isfinite(expected).all() and torch.isfinite(q_values).all()),
                        }
                    )
            rewards.append(episode_reward)
            episode_summaries.append(
                {
                    "episode": episode + 1,
                    "epsilon": float(epsilon),
                    "steps": episode_step_count,
                    "episode_reward": float(episode_reward),
                    "action_distribution": dict(episode_action_distribution),
                    "random_action_count": episode_random_count,
                    "greedy_action_count": episode_greedy_count,
                    "random_action_ratio": _ratio(episode_random_count, episode_step_count),
                    "greedy_action_ratio": _ratio(episode_greedy_count, episode_step_count),
                    "random_action_distribution": dict(episode_random_distribution),
                    "greedy_action_distribution": dict(episode_greedy_distribution),
                    "unique_random_action_types": len(episode_random_distribution),
                    "random_action_type_entropy": _distribution_entropy(episode_random_distribution),
                    "valid_action_type_count_avg": float(np.mean(episode_valid_action_type_counts)) if episode_valid_action_type_counts else 0.0,
                    "reward_component_totals": dict(episode_reward_components),
                    "reward_distribution": _reward_distribution(episode_reward_values),
                    "action_type_reward_summary": _action_reward_summary(episode_action_reward_values),
                    "action_type_reward_component_totals": _nested_counter_dict(episode_action_component_totals),
                    "action_type_reward_component_event_counts": _nested_counter_dict(episode_action_component_event_counts),
                    "repeat_event_counts": dict(episode_repeat_event_counts),
                    "action_type_repeat_event_counts": _nested_counter_dict(episode_action_repeat_event_counts),
                }
            )
            print(
                f"[dqn-train] finished episode {episode + 1}/{args.episodes}: "
                f"reward={episode_reward:.3f}, steps={episode_step_count}, "
                f"epsilon={epsilon:.3f}, random={episode_random_count}, greedy={episode_greedy_count}"
            )
            if (episode + 1) % args.target_update == 0:
                target.load_state_dict(online.state_dict())
            if episode + 1 in checkpoint_episodes:
                _save_dqn_checkpoint(
                    online,
                    _training_manifest(
                        args=args,
                        observation_dim=observation_dim,
                        action_dim=action_dim,
                        rewards=rewards,
                        losses=losses,
                        findings=findings,
                        reward_component_totals=reward_component_totals,
                        episode_summaries=episode_summaries,
                        epsilon_trajectory=epsilon_trajectory,
                        action_distribution=action_distribution,
                        random_action_distribution=random_action_distribution,
                        greedy_action_distribution=greedy_action_distribution,
                        q_diagnostics=q_diagnostics,
                        reward_values=reward_values,
                        action_reward_values=action_reward_values,
                        valid_action_type_counts=valid_action_type_counts,
                        last_balanced_expected_distribution=last_balanced_expected_distribution,
                        loss_history=loss_history,
                        confirmation_training_summary={
                            **confirmation_training_funnel.summary(),
                            "random_expected_action_type_match_count": random_expected_type_matches,
                            "greedy_expected_action_type_match_count": greedy_expected_type_matches,
                            "random_full_reproduction_match_count": random_full_reproduction_matches,
                            "greedy_full_reproduction_match_count": greedy_full_reproduction_matches,
                            "action_reproduction_pending_steps": action_reproduction_pending_steps,
                            "observation_reproduction_pending_steps": observation_reproduction_pending_steps,
                            "action_reproduction_opportunities": action_reproduction_opportunities,
                            "action_reproduction_executable_opportunities": action_reproduction_executable_opportunities,
                            "action_reproduction_safe_mask_valid": action_reproduction_safe_mask_valid,
                            "random_action_reproduction_matches": random_action_reproduction_matches,
                            "greedy_action_reproduction_matches": greedy_action_reproduction_matches,
                            "random_action_full_matches": random_action_full_matches,
                            "greedy_action_full_matches": greedy_action_full_matches,
                            "observation_reproduction_events": observation_reproduction_events,
                            "confirmation_progress_count": confirmation_progress_count,
                            "candidate_to_confirmed_count": candidate_to_confirmed_count,
                            "confirmed_count": confirmed_count,
                            "confirmed_unique_logical_candidates": len(confirmed_logical_candidates),
                            "confirmed_observation_count": confirmed_observation_count,
                            "newly_confirmed_transition_count": newly_confirmed_transition_count,
                            "confirmed_reward_event_count": confirmed_reward_event_count,
                            "confirmed_reward_duplicate_attempt_count": confirmed_reward_duplicate_attempt_count,
                            "confirmed_reward_duplicate_blocked_count": confirmed_reward_duplicate_blocked_count,
                            "progress_reward_summary": _reward_distribution(progress_reward_values),
                            "confirmed_reward_summary": _reward_distribution(confirmed_reward_values),
                            "progress_reward_component_totals": dict(progress_reward_component_totals),
                            "confirmed_reward_component_totals": dict(confirmed_reward_component_totals),
                        },
                        replay_sampling_summary={
                            "strategy": "confirmation_stratified_uniform",
                            "confirmation_fraction": replay.confirmation_fraction,
                            "confirmation_capacity": replay.confirmation_capacity,
                            "retention": replay.retention_stats(),
                            "sampling": dict(replay.sample_telemetry),
                            "replacement_policy": "without_replacement_within_batch",
                            "empty_confirmation_fallback": "general_uniform_sampling",
                        },
                        episodes_completed=episode + 1,
                    ),
                    _checkpoint_path(args.output, checkpoint_prefix, episode + 1),
                )
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = _training_manifest(
        args=args,
        observation_dim=observation_dim,
        action_dim=action_dim,
        rewards=rewards,
        losses=losses,
        findings=findings,
        reward_component_totals=reward_component_totals,
        episode_summaries=episode_summaries,
        epsilon_trajectory=epsilon_trajectory,
        action_distribution=action_distribution,
        random_action_distribution=random_action_distribution,
        greedy_action_distribution=greedy_action_distribution,
        q_diagnostics=q_diagnostics,
        reward_values=reward_values,
        action_reward_values=action_reward_values,
        valid_action_type_counts=valid_action_type_counts,
        last_balanced_expected_distribution=last_balanced_expected_distribution,
        loss_history=loss_history,
        confirmation_training_summary={
            **confirmation_training_funnel.summary(),
            "random_expected_action_type_match_count": random_expected_type_matches,
            "greedy_expected_action_type_match_count": greedy_expected_type_matches,
            "random_full_reproduction_match_count": random_full_reproduction_matches,
            "greedy_full_reproduction_match_count": greedy_full_reproduction_matches,
            "action_reproduction_pending_steps": action_reproduction_pending_steps,
            "observation_reproduction_pending_steps": observation_reproduction_pending_steps,
            "action_reproduction_opportunities": action_reproduction_opportunities,
            "action_reproduction_executable_opportunities": action_reproduction_executable_opportunities,
            "action_reproduction_safe_mask_valid": action_reproduction_safe_mask_valid,
            "random_action_reproduction_matches": random_action_reproduction_matches,
            "greedy_action_reproduction_matches": greedy_action_reproduction_matches,
            "random_action_full_matches": random_action_full_matches,
            "greedy_action_full_matches": greedy_action_full_matches,
            "observation_reproduction_events": observation_reproduction_events,
            "confirmation_progress_count": confirmation_progress_count,
            "candidate_to_confirmed_count": candidate_to_confirmed_count,
            "confirmed_count": confirmed_count,
            "confirmed_unique_logical_candidates": len(confirmed_logical_candidates),
            "confirmed_observation_count": confirmed_observation_count,
            "newly_confirmed_transition_count": newly_confirmed_transition_count,
            "confirmed_reward_event_count": confirmed_reward_event_count,
            "confirmed_reward_duplicate_attempt_count": confirmed_reward_duplicate_attempt_count,
            "confirmed_reward_duplicate_blocked_count": confirmed_reward_duplicate_blocked_count,
            "progress_reward_summary": _reward_distribution(progress_reward_values),
            "confirmed_reward_summary": _reward_distribution(confirmed_reward_values),
            "progress_reward_component_totals": dict(progress_reward_component_totals),
            "confirmed_reward_component_totals": dict(confirmed_reward_component_totals),
        },
        replay_sampling_summary={
            "strategy": "confirmation_stratified_uniform",
            "confirmation_fraction": replay.confirmation_fraction,
            "confirmation_capacity": replay.confirmation_capacity,
            "retention": replay.retention_stats(),
            "sampling": dict(replay.sample_telemetry),
            "replacement_policy": "without_replacement_within_batch",
            "empty_confirmation_fallback": "general_uniform_sampling",
        },
        episodes_completed=args.episodes,
    )
    _save_dqn_checkpoint(online, manifest, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/browsergym_ui_dqn.pt"))
    parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=32); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=50000); parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0); parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument(
        "--dqn-architecture",
        choices=[FLAT_DQN_ARCHITECTURE, FACTORIZED_DQN_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE],
        default=FLAT_DQN_ARCHITECTURE,
    )
    parser.add_argument(
        "--greedy-selection-mode",
        choices=[FLAT_MAX_SELECTION, TYPE_LOGMEANEXP_SELECTION],
        default=TYPE_LOGMEANEXP_SELECTION,
    )
    parser.add_argument(
        "--td-target-selection-mode",
        choices=[FLAT_MAX_SELECTION, TYPE_LOGMEANEXP_SELECTION],
        default=TYPE_LOGMEANEXP_SELECTION,
    )
    parser.add_argument("--checkpoint-episodes", default="")
    parser.add_argument("--checkpoint-prefix", default="")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=True, indent=2))


def _load_site_config(config_path: str, site_id: str | None, base_url: str | None) -> dict[str, Any]:
    if base_url:
        return {"site_id": site_id or "site001", "base_url": base_url}
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    sites = config.get("sites", []) if isinstance(config, Mapping) else []
    for site in sites:
        if isinstance(site, Mapping) and str(site.get("site_id") or "") == str(site_id or ""):
            return {"site_id": site.get("site_id"), "base_url": site.get("base_url")}
    raise ValueError(f"site_id {site_id!r} not found in {config_path}")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _distribution_entropy(counts: Mapping[str, int]) -> float:
    total = sum(int(value or 0) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        probability = float(value or 0) / total
        if probability > 0.0:
            entropy -= probability * float(np.log(probability))
    return entropy


def _parse_episode_list(value: str | None) -> set[int]:
    episodes: set[int] = set()
    for item in str(value or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        episodes.add(int(item))
    return episodes


def _checkpoint_path(output: Path, checkpoint_prefix: Path | None, episode: int) -> Path:
    if checkpoint_prefix is not None:
        return checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_ep{episode}.pt")
    return output.with_name(f"{output.stem}_ep{episode}{output.suffix}")


def _save_dqn_checkpoint(network: QNetwork, manifest: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": network.state_dict(), "manifest": dict(manifest)}, output)
    output.with_suffix(".manifest.json").write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")


def _training_manifest(
    *,
    args: argparse.Namespace,
    observation_dim: int,
    action_dim: int,
    rewards: list[float],
    losses: list[float],
    findings: Mapping[str, int],
    reward_component_totals: Mapping[str, float],
    episode_summaries: list[Mapping[str, Any]],
    epsilon_trajectory: list[Mapping[str, float]],
    action_distribution: Counter[str],
    random_action_distribution: Counter[str],
    greedy_action_distribution: Counter[str],
    q_diagnostics: list[Mapping[str, Any]],
    reward_values: list[float],
    action_reward_values: Mapping[str, list[float]],
    valid_action_type_counts: list[int],
    last_balanced_expected_distribution: Mapping[str, float],
    loss_history: list[Mapping[str, float]],
    confirmation_training_summary: Mapping[str, Any],
    replay_sampling_summary: Mapping[str, Any],
    episodes_completed: int,
) -> dict[str, Any]:
    has_confirmation_context = args.dqn_architecture in {
        FACTORIZED_DQN_CONFOBS_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    }
    history_dim = (
        HISTORY_CONTEXT_DIM if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE
        else HISTORY_CONTEXT_V2_DIM if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE
        else 0
    )
    history_version = (
        HISTORY_OBSERVATION_VERSION if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE
        else HISTORY_OBSERVATION_V2_VERSION if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE
        else None
    )
    return {
        "schema_version": "browsergym-ui-dqn-v1",
        "algorithm": "masked-double-dqn",
        "dqn_architecture": args.dqn_architecture,
        "greedy_selection_mode": args.greedy_selection_mode,
        "td_target_selection_mode": args.td_target_selection_mode,
        "type_score_method": (
            TYPE_SCORE_NORMALIZED_LOGMEANEXP
            if args.greedy_selection_mode == TYPE_LOGMEANEXP_SELECTION
            else None
        ),
        "environment_action_dim": action_dim,
        "num_action_types": len(ActionSpace.action_types),
        "max_candidates": args.max_candidates,
        "type_head_dim": len(ActionSpace.action_types) if _is_factorized_architecture(args.dqn_architecture) else None,
        "candidate_head_shape": [len(ActionSpace.action_types), args.max_candidates] if _is_factorized_architecture(args.dqn_architecture) else None,
        "candidate_centering": "valid_mean" if _is_factorized_architecture(args.dqn_architecture) else None,
        "base_obs_dim": int(observation_dim - (CONFIRMATION_CONTEXT_DIM if has_confirmation_context else 0) - history_dim),
        "confirmation_context_dim": CONFIRMATION_CONTEXT_DIM if has_confirmation_context else 0,
        "history_context_dim": history_dim,
        "history_window": DQN_HISTORY_WINDOW if history_dim else 0,
        "history_observation_version": history_version,
        "prospective_repeat_risk_dim": len(ActionSpace.action_types) * len(PROSPECTIVE_REPEAT_RISK_FLAGS)
        if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE else 0,
        "prospective_repeat_risk_ordering": "action_type-major: immediate,recent,cycle"
        if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE else None,
        "replay_state_dim": int(observation_dim),
        "replay_next_state_dim": int(observation_dim),
        "dqn_obs_dim": int(observation_dim),
        "ppo_obs_dim": 635,
        "confirmation_observation_version": CONFIRMATION_OBSERVATION_VERSION if has_confirmation_context else None,
        "active_pending_policy": "resolvable_progress_age_key" if has_confirmation_context else None,
        "ground_truth": False,
        "episodes": args.episodes,
        "episodes_completed": episodes_completed,
        "seed": args.seed,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "excluded_action_types": sorted(EXCLUDED_ACTIONS),
        "mean_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "loss_history": list(loss_history),
        "finding_counts": dict(findings),
        "reward_config": DQN_REWARD_CONFIG,
        "exploration_sampling_mode": "action_type_balanced",
        "valid_action_type_count_avg": float(np.mean(valid_action_type_counts)) if valid_action_type_counts else 0.0,
        "reward_component_totals": dict(reward_component_totals),
        "episode_summaries": list(episode_summaries),
        "epsilon_schedule": {
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "decay_basis": "episode",
            "formula": "max(epsilon_end, epsilon_start - episode_index_zero_based / episodes)",
        },
        "epsilon_trajectory": list(epsilon_trajectory),
        "action_distribution": dict(action_distribution),
        "random_action_count": int(sum(random_action_distribution.values())),
        "greedy_action_count": int(sum(greedy_action_distribution.values())),
        "random_action_ratio": _ratio(sum(random_action_distribution.values()), sum(action_distribution.values())),
        "greedy_action_ratio": _ratio(sum(greedy_action_distribution.values()), sum(action_distribution.values())),
        "random_action_distribution": dict(random_action_distribution),
        "greedy_action_distribution": dict(greedy_action_distribution),
        "random_action_type_entropy": _distribution_entropy(random_action_distribution),
        "balanced_expected_action_type_distribution_last": dict(last_balanced_expected_distribution),
        "q_value_diagnostic_summary": _q_diagnostic_summary(q_diagnostics),
        "factorized_diagnostic_summary": _factorized_diagnostic_summary(q_diagnostics, loss_history),
        "confirmation_training_summary": dict(confirmation_training_summary),
        "replay_sampling_summary": dict(replay_sampling_summary),
        "replay_reward_distribution": _reward_distribution(reward_values),
        "action_type_reward_summary": _action_reward_summary(action_reward_values),
        "base_url_sha256": hashlib.sha256(args.base_url.encode()).hexdigest(),
    }


def _reward_distribution(values: list[float]) -> dict[str, Any]:
    buckets = {
        "< -0.5": 0,
        "-0.5 ~ 0": 0,
        "0": 0,
        "0 ~ 0.5": 0,
        "0.5 ~ 1.5": 0,
        "> 1.5": 0,
    }
    for value in values:
        if value < -0.5:
            buckets["< -0.5"] += 1
        elif value < 0:
            buckets["-0.5 ~ 0"] += 1
        elif value == 0:
            buckets["0"] += 1
        elif value < 0.5:
            buckets["0 ~ 0.5"] += 1
        elif value <= 1.5:
            buckets["0.5 ~ 1.5"] += 1
        else:
            buckets["> 1.5"] += 1
    return {
        "transition_count": len(values),
        "reward_mean": float(np.mean(values)) if values else 0.0,
        "reward_min": float(np.min(values)) if values else 0.0,
        "reward_max": float(np.max(values)) if values else 0.0,
        "positive_transition_count": sum(1 for value in values if value > 0),
        "zero_transition_count": sum(1 for value in values if value == 0),
        "negative_transition_count": sum(1 for value in values if value < 0),
        "histogram": buckets,
    }


def _action_reward_summary(values_by_type: Mapping[str, list[float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for action_type, values in values_by_type.items():
        summary[str(action_type)] = {
            "count": float(len(values)),
            "reward_mean": float(np.mean(values)) if values else 0.0,
            "reward_min": float(np.min(values)) if values else 0.0,
            "reward_max": float(np.max(values)) if values else 0.0,
            "positive_transition_count": float(sum(1 for value in values if value > 0)),
            "zero_transition_count": float(sum(1 for value in values if value == 0)),
            "negative_transition_count": float(sum(1 for value in values if value < 0)),
        }
    return summary


def _nested_counter_dict(values_by_type: Mapping[str, Counter[str]]) -> dict[str, dict[str, float]]:
    return {
        str(action_type): {str(key): float(value) for key, value in values.items()}
        for action_type, values in values_by_type.items()
    }


def _head_grad_norm(head: nn.Module | None) -> float | None:
    if head is None:
        return None
    total = 0.0
    found = False
    for parameter in head.parameters():
        if parameter.grad is None:
            continue
        found = True
        total += float(torch.sum(parameter.grad.detach() ** 2).item())
    return float(np.sqrt(total)) if found else None


def _q_diagnostic_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_keys = ("selected_action_q", "max_valid_q", "mean_valid_q", "min_valid_q")
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_min"] = float(np.min(values)) if values else None
        summary[f"{key}_max"] = float(np.max(values)) if values else None
    max_by_type: dict[str, float] = {}
    for row in rows:
        values = row.get("max_q_by_action_type", {})
        if not isinstance(values, Mapping):
            continue
        for action_type, value in values.items():
            numeric = float(value)
            if action_type not in max_by_type or numeric > max_by_type[action_type]:
                max_by_type[str(action_type)] = numeric
    top_counter: Counter[str] = Counter()
    top_examples: list[dict[str, Any]] = []
    for row in rows:
        top_actions = row.get("top_k_valid_actions", [])
        if not isinstance(top_actions, list) or not top_actions:
            continue
        top = top_actions[0]
        action_type = str(top.get("action_type") or "")
        top_counter[action_type] += 1
        if len(top_examples) < 10:
            top_examples.append(dict(top))
    summary["max_q_by_action_type"] = max_by_type
    summary["top_q_action_type_counts"] = dict(top_counter)
    summary["top_q_action_examples"] = top_examples
    return summary


def _factorized_diagnostic_summary(
    q_rows: list[Mapping[str, Any]],
    loss_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    type_values_by_type: dict[str, list[float]] = defaultdict(list)
    candidate_stats_by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in q_rows:
        values = row.get("type_value_by_action_type", {})
        if isinstance(values, Mapping):
            for action_type, value in values.items():
                type_values_by_type[str(action_type)].append(float(value))
        stats = row.get("candidate_advantage_stats", {})
        if isinstance(stats, Mapping):
            for action_type, stat in stats.items():
                if not isinstance(stat, Mapping):
                    continue
                for key in ("raw_advantage_mean", "raw_advantage_std", "centered_advantage_mean", "centered_advantage_std"):
                    if stat.get(key) is not None:
                        candidate_stats_by_type[str(action_type)][key].append(float(stat[key]))

    type_head_grads = [float(row["type_head_grad_norm"]) for row in loss_rows if row.get("type_head_grad_norm") is not None]
    candidate_head_grads = [
        float(row["candidate_head_grad_norm"]) for row in loss_rows if row.get("candidate_head_grad_norm") is not None
    ]
    return {
        "type_values_by_action_type": {
            action_type: {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }
            for action_type, values in type_values_by_type.items()
            if values
        },
        "candidate_advantage_stats": {
            action_type: {key: float(np.mean(values)) for key, values in stats.items() if values}
            for action_type, stats in candidate_stats_by_type.items()
        },
        "type_head_grad_norm_mean": float(np.mean(type_head_grads)) if type_head_grads else None,
        "candidate_head_grad_norm_mean": float(np.mean(candidate_head_grads)) if candidate_head_grads else None,
        "type_head_grad_nonzero": any(value > 0.0 for value in type_head_grads),
        "candidate_head_grad_nonzero": any(value > 0.0 for value in candidate_head_grads),
        "finite_updates": all(bool(row.get("finite", True)) for row in loss_rows),
    }


if __name__ == "__main__":
    main()
