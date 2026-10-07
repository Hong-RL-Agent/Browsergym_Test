"""Run PPO-only or DQN-only policy sanity checks and summarize behavior metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent
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
    factorized_q_value_diagnostics,
    factorized_flat_q_values,
    select_factorized_greedy_action,
    q_value_diagnostics,
    select_greedy_action,
)
from dqn_error_discovery.agents.confirmation_observation import (
    CONFIRMATION_CONTEXT_DIM,
    augment_dqn_observation,
    build_dqn_confirmation_context,
)
from dqn_error_discovery.agents.history_observation import (
    DQN_HISTORY_WINDOW,
    HISTORY_CONTEXT_DIM,
    HISTORY_CONTEXT_V2_DIM,
    HISTORY_OBSERVATION_VERSION,
    HISTORY_OBSERVATION_V2_VERSION,
    TARGET_REQUIRED_ACTION_TYPES,
    append_history_transition,
    augment_dqn_history_observation,
    augment_dqn_history_observation_v2,
    action_identity as history_action_identity,
    build_dqn_history_context,
    build_dqn_history_context_v2,
)
from dqn_error_discovery.agents.confirmation_learning_diagnostics import (
    ConfirmationFunnelAggregator,
    factorized_confirmation_step_diagnostic,
    is_reproduction_opportunity,
    real_vs_zero_context_diagnostic,
)
from dqn_error_discovery.agents.confirmation_reproduction import (
    ACTION_REPRODUCTION,
    OBSERVATION_REPRODUCTION,
    ExpectedReproduction,
    is_reproduction_instruction_executable,
)
from dqn_error_discovery.runners.train_dqn_error_detector import (
    calculate_dqn_reward,
    dqn_no_effect_diagnostic,
    initialize_dqn_candidate_reward_baseline,
    safe_action_mask,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv, normalize_action_error
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward


GROUND_TRUTH_KEYS = {
    "data_bug_id",
    "catalog_bug_id_matches",
    "known_bug",
    "expected_bug",
    "ground_truth",
    "site_profile",
    "matched_bug_id",
}

STRONG_EVIDENCE_RULE_IDS = {"R03", "R13", "R14", "R15"}


def _confirmation_state(env: BrowserGymJAWSEnv, observation: Mapping[str, Any], history: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    base = env.observation_encoder.encode_observation(observation)
    context, metadata = build_dqn_confirmation_context(history, observation, env.action_space)
    return augment_dqn_observation(base, context), metadata


def _dqn_state(env: BrowserGymJAWSEnv, observation: Mapping[str, Any], history: Mapping[str, Any], architecture: str) -> tuple[np.ndarray, dict[str, Any]]:
    base = env.observation_encoder.encode_observation(observation)
    if architecture not in {FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
        return base, {"pending_confirmation_exists": False, "matching_candidate_slots": []}
    context, metadata = build_dqn_confirmation_context(history, observation, env.action_space)
    if architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE:
        history_context, history_metadata = build_dqn_history_context(
            history, observation, env.action_space.action_types, env.action_space.max_candidates
        )
        metadata["history_context"] = history_metadata
        return augment_dqn_history_observation(base, context, history_context), metadata
    if architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE:
        history_context, history_metadata = build_dqn_history_context_v2(
            history, observation, env.action_space.action_types, env.action_space.max_candidates
        )
        metadata["history_context"] = history_metadata
        return augment_dqn_history_observation_v2(base, context, history_context), metadata
    return augment_dqn_observation(base, context), metadata


class SanityMetricsAggregator:
    def __init__(self, mode: str) -> None:
        if mode not in {"ppo", "dqn"}:
            raise ValueError("mode must be 'ppo' or 'dqn'")
        self.mode = mode
        self.total_steps = 0
        self.action_distribution: Counter[str] = Counter()
        self.reward_component_totals: Counter[str] = Counter()
        self.states: set[str] = set()
        self.urls: set[str] = set()
        self.targets: set[str] = set()
        self.action_targets: set[str] = set()
        self.local_states: set[str] = set()
        self.local_action_targets: set[str] = set()
        self.repeated_actions: Counter[str] = Counter()
        self.candidate_signatures: set[str] = set()
        self.confirmed_signatures: set[str] = set()
        self.candidate_to_confirmed_signatures: set[str] = set()
        self.strong_evidence_confirmed_signatures: set[str] = set()
        self.repeated_confirmed_signatures: Counter[str] = Counter()
        self.valid_action_count_total = 0
        self.valid_action_type_count_total = 0
        self.valid_actions_by_type: Counter[str] = Counter()
        self.selected_actions_by_type: Counter[str] = Counter()
        self.action_error_categories: Counter[str] = Counter()
        self.action_errors_by_type: Counter[str] = Counter()
        self.failed_target_ids: Counter[str] = Counter()
        self.synthetic_target_selection_count = 0
        self.non_resolvable_target_selection_count = 0
        self.empty_mask_count = 0
        self.fallback_action_count = 0
        self.q_diagnostics: list[dict[str, Any]] = []
        self.policy_entropies: list[float] = []
        self.action_error_details: list[dict[str, Any]] = []
        self.repeat_confirmed_anomaly_count = 0
        self.repeat_state_count = 0
        self.repeat_local_state_count = 0
        self.repeat_action_target_count = 0
        self.repeat_local_action_count = 0
        self.no_effect_count = 0
        self.invalid_action_count = 0
        self.action_error_count = 0
        self.timeout_count = 0
        self.flow_progress_count = 0
        self.repeat_breakdown: Counter[str] = Counter()
        self.action_type_repeat_breakdown: dict[str, Counter[str]] = {}
        self.step_summaries: list[dict[str, Any]] = []
        self.action_target_history: list[str] = []
        self.pending_state_step_count = 0
        self.reproduction_opportunity_count = 0
        self.expected_action_type_selected_count = 0
        self.expected_action_and_target_selected_count = 0
        self.confirmation_progress_count = 0
        self.confirmed_count = 0
        self.confirmation_funnel = ConfirmationFunnelAggregator()
        self.action_reproduction_pending_count = 0
        self.observation_reproduction_pending_count = 0
        self.action_reproduction_opportunity_count = 0
        self.action_reproduction_executable_count = 0
        self.action_reproduction_safe_mask_valid_count = 0
        self.action_reproduction_type_match_count = 0
        self.action_reproduction_full_match_count = 0
        self.observation_reproduction_event_count = 0
        self.rule_mode_funnel: dict[str, Counter[str]] = {}
        self.confirmed_observation_count = 0
        self.newly_confirmed_transition_count = 0
        self.confirmed_reward_event_count = 0
        self.confirmed_reward_duplicate_attempt_count = 0
        self.confirmed_reward_duplicate_blocked_count = 0
        self.confirmed_logical_keys: set[str] = set()
        self.reward_values_by_action_type: dict[str, list[float]] = {}
        self.history_steps_with_full_history = 0
        self.history_steps_with_cycle_feature = 0
        self.history_steps_with_streak_gt1 = 0
        self.history_distinct_recent_identity_counts: list[int] = []
        self.cycle_feature_active_count = 0
        self.cycle_feature_next_repeats_count = 0
        self.cycle_feature_next_breaks_count = 0
        self.previous_unproductive_repeat_count = 0
        self.previous_unproductive_next_changes_identity_count = 0
        self.previous_unproductive_next_same_identity_count = 0
        self.previous_unproductive_repeat_identity = ""
        self.previous_transition_was_unproductive_repeat = False
        self.prospective_repeat_validation: dict[str, Counter[str]] = {
            "immediate": Counter(),
            "recent": Counter(),
            "cycle": Counter(),
        }
        self.prospective_repeat_examples: list[dict[str, Any]] = []
        self.action_identity_sequence: list[str] = []
        self.action_type_sequence: list[str] = []
        self.selected_risk_counts: Counter[str] = Counter()
        self.candidate_independent_selected_count = 0
        self.cycle_risk_available_steps = 0
        self.cycle_risk_selected_count = 0
        self.cycle_risk_zero_selected_count = 0

    def record_step(
        self,
        *,
        step: int,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        action: Mapping[str, Any],
        reward: float,
        reward_components: Mapping[str, Any] | None = None,
        anomalies: list[Mapping[str, Any]] | None = None,
        info: Mapping[str, Any] | None = None,
        mask_stats: Mapping[str, Any] | None = None,
        q_diagnostic: Mapping[str, Any] | None = None,
    ) -> None:
        self.total_steps += 1
        reward_components = reward_components or {}
        anomalies = anomalies or []
        info = info or {}
        mask_stats = mask_stats or {}
        q_diagnostic = q_diagnostic or {}
        transition_events = action.get("confirmation_transition_events", [])
        if isinstance(transition_events, list):
            self.newly_confirmed_transition_count += sum(1 for event in transition_events if event.get("newly_confirmed"))
            self.confirmed_reward_event_count += sum(1 for event in transition_events if event.get("newly_confirmed"))
            self.confirmed_logical_keys.update(
                str(event.get("logical_key"))
                for event in transition_events
                if event.get("newly_confirmed") and event.get("logical_key")
            )
        self.confirmed_observation_count += int(action.get("confirmation_observation_events", 0) or 0)
        self.confirmed_reward_duplicate_attempt_count += int(action.get("confirmation_duplicate_attempts", 0) or 0)
        self.confirmed_reward_duplicate_blocked_count += int(action.get("confirmation_duplicate_blocks", 0) or 0)
        action_type = str(action.get("action_type") or "")
        confirmation_context = action.get("confirmation_context", {})
        history_context = confirmation_context.get("history_context", {}) if isinstance(confirmation_context, Mapping) else {}
        history_valid_count = 0
        active_cycle_length = 0
        same_identity_streak = 0
        distinct_recent_identity_count = 0
        cycle_continuation_identity = ""
        selected_action_identity = ""
        action_candidate_for_identity = _action_candidate(before_observation, action)
        if isinstance(history_context, Mapping):
            valid_mask = history_context.get("history_valid_mask", [])
            if isinstance(valid_mask, list):
                history_valid_count = int(sum(int(bool(item)) for item in valid_mask))
            active_cycle_length = int(history_context.get("active_cycle_length", 0) or 0)
            same_identity_streak = int(history_context.get("same_action_identity_streak", 0) or 0)
            distinct_recent_identity_count = int(history_context.get("distinct_action_identity_count", 0) or 0)
            recent_identities = history_context.get("recent_action_identities", [])
            if isinstance(recent_identities, list) and active_cycle_length > 0 and len(recent_identities) >= active_cycle_length:
                cycle_continuation_identity = str(recent_identities[active_cycle_length - 1] or "")
        if isinstance(confirmation_context, Mapping):
            pending = bool(confirmation_context.get("pending_confirmation_exists"))
            if pending:
                self.pending_state_step_count += 1
            mode = str(confirmation_context.get("confirmation_mode") or "")
            rule_id = str(confirmation_context.get("active_rule_id") or "UNKNOWN")
            rule_counter = self.rule_mode_funnel.setdefault(rule_id, Counter())
            if pending and mode == ACTION_REPRODUCTION:
                self.action_reproduction_pending_count += 1
                rule_counter["action_pending"] += 1
            elif pending and mode == OBSERVATION_REPRODUCTION:
                self.observation_reproduction_pending_count += 1
                rule_counter["observation_pending"] += 1
            expected_type = str(confirmation_context.get("expected_action_type") or "")
            requires_target = bool(confirmation_context.get("expected_action_requires_target"))
            visible = bool(confirmation_context.get("expected_target_visible"))
            opportunity = bool(mode == ACTION_REPRODUCTION and expected_type and (not requires_target or visible))
            if opportunity:
                self.action_reproduction_opportunity_count += 1
                rule_counter["opportunities"] += 1
                if bool(action.get("confirmation_action_executable")):
                    self.action_reproduction_executable_count += 1
                    rule_counter["executable"] += 1
                if bool(action.get("confirmation_action_mask_valid")):
                    self.action_reproduction_safe_mask_valid_count += 1
                    rule_counter["safe_mask_valid"] += 1
                self.reproduction_opportunity_count += 1
                if action_type == expected_type:
                    self.action_reproduction_type_match_count += 1
                    rule_counter["type_match"] += 1
                    self.expected_action_type_selected_count += 1
                    matching_slots = confirmation_context.get("matching_candidate_slots", [])
                    if not requires_target or int(action.get("candidate_index", -1)) in matching_slots:
                        self.expected_action_and_target_selected_count += 1
                        self.action_reproduction_full_match_count += 1
                        rule_counter["full_match"] += 1
            if bool(confirmation_context.get("confirmation_progress_after")):
                self.confirmation_progress_count += 1
                if mode == OBSERVATION_REPRODUCTION:
                    self.observation_reproduction_event_count += 1
                    rule_counter["observation_event"] += 1
                rule_counter["progress"] += 1
            if bool(confirmation_context.get("confirmation_confirmed_after")):
                self.confirmed_count += 1
                rule_counter["confirmed"] += 1
            self.confirmation_funnel.record(
                confirmation_context,
                action,
                confirmation_progress=bool(confirmation_context.get("confirmation_progress_after")),
                candidate_to_confirmed=_positive_component(reward_components, ("confirmation_upgrade",)),
                confirmed=_positive_component(reward_components, ("confirmed_anomaly",))
                or any(str(item.get("status") or "") == "confirmed" for item in anomalies),
                type_diagnostic=action.get("confirmation_type_diagnostic", {})
                if isinstance(action.get("confirmation_type_diagnostic"), Mapping)
                else {},
                context_diagnostic=action.get("confirmation_context_counterfactual", {})
                if isinstance(action.get("confirmation_context_counterfactual"), Mapping)
                else {},
            )
        action_candidate = _action_candidate(before_observation, action)
        target = _target_identity(action_candidate)
        action_target = f"{action_type}:{target or action.get('candidate_index', 0)}" if action_type else ""
        selected_action_identity = history_action_identity(
            action_type,
            action_candidate_for_identity,
            requires_target=action_type in {"click_element", "fill_input", "press_enter"},
        ) if action_type else ""
        state = _state_signature(after_observation)
        local_state = _local_state_signature(after_observation)
        url = _url_path(after_observation)
        local_action_target = action_target

        state_new = state not in self.states
        url_new = url not in self.urls
        target_new = bool(target and target not in self.targets)
        action_target_new = bool(action_target and action_target not in self.action_targets)
        local_state_new = local_state not in self.local_states
        local_action_target_new = bool(local_action_target and local_action_target not in self.local_action_targets)

        if not state_new:
            self.repeat_state_count += 1
        if not local_state_new:
            self.repeat_local_state_count += 1
        if action_target and not action_target_new:
            self.repeat_action_target_count += 1
            self.repeated_actions[action_target] += 1
        if local_action_target and not local_action_target_new:
            self.repeat_local_action_count += 1

        self.states.add(state)
        self.urls.add(url)
        if target:
            self.targets.add(target)
        if action_target:
            self.action_targets.add(action_target)
            self.action_target_history.append(action_target)
        self.local_states.add(local_state)
        if local_action_target:
            self.local_action_targets.add(local_action_target)
        self.action_distribution[action_type] += 1
        self.selected_actions_by_type[action_type] += 1
        self.action_type_sequence.append(action_type)
        self.action_identity_sequence.append(selected_action_identity or action_type)
        self.reward_values_by_action_type.setdefault(action_type, []).append(float(reward))
        if history_valid_count >= DQN_HISTORY_WINDOW:
            self.history_steps_with_full_history += 1
        if active_cycle_length > 0:
            self.history_steps_with_cycle_feature += 1
            self.cycle_feature_active_count += 1
            if selected_action_identity and selected_action_identity == cycle_continuation_identity:
                self.cycle_feature_next_repeats_count += 1
            else:
                self.cycle_feature_next_breaks_count += 1
        if same_identity_streak > 1:
            self.history_steps_with_streak_gt1 += 1
        if history_valid_count > 0:
            self.history_distinct_recent_identity_counts.append(distinct_recent_identity_count)
        if self.previous_transition_was_unproductive_repeat:
            self.previous_unproductive_repeat_count += 1
            if selected_action_identity and selected_action_identity != self.previous_unproductive_repeat_identity:
                self.previous_unproductive_next_changes_identity_count += 1
            else:
                self.previous_unproductive_next_same_identity_count += 1
        if isinstance(action.get("policy_entropy"), (int, float)):
            self.policy_entropies.append(float(action["policy_entropy"]))
        self._record_mask_stats(mask_stats)
        if q_diagnostic:
            self.q_diagnostics.append(dict(q_diagnostic))
        repeat_diagnostic = action.get("repeat_diagnostic", {}) if isinstance(action.get("repeat_diagnostic"), Mapping) else {}
        prospective_risk = (
            history_context.get("prospective_repeat_risk_by_action_type", {})
            if isinstance(history_context, Mapping)
            else {}
        )
        selected_risk = prospective_risk.get(action_type, {}) if isinstance(prospective_risk, Mapping) else {}
        if action_type and action_type not in TARGET_REQUIRED_ACTION_TYPES and isinstance(selected_risk, Mapping):
            self.candidate_independent_selected_count += 1
            comparisons = {
                "immediate": (bool(selected_risk.get("immediate")), bool(repeat_diagnostic.get("immediate_repeat"))),
                "recent": (bool(selected_risk.get("recent")), bool(repeat_diagnostic.get("recent_window_repeat"))),
                "cycle": (bool(selected_risk.get("cycle")), bool(repeat_diagnostic.get("cycle_repeat"))),
            }
            for name, (predicted, _) in comparisons.items():
                if predicted:
                    self.selected_risk_counts[name] += 1
            for name, (predicted, actual) in comparisons.items():
                if predicted and actual:
                    self.prospective_repeat_validation[name]["TP"] += 1
                elif predicted and not actual:
                    self.prospective_repeat_validation[name]["FP"] += 1
                elif not predicted and actual:
                    self.prospective_repeat_validation[name]["FN"] += 1
                else:
                    self.prospective_repeat_validation[name]["TN"] += 1
            if len(self.prospective_repeat_examples) < 12 and (
                action_type in {"scroll_up", "scroll_down"} or any(actual for _, actual in comparisons.values())
            ):
                self.prospective_repeat_examples.append({
                    "step": step,
                    "action_type": action_type,
                    "predicted": {key: value[0] for key, value in comparisons.items()},
                    "actual": {key: value[1] for key, value in comparisons.items()},
                    "recent_action_identities": history_context.get("recent_action_identities", [])
                    if isinstance(history_context, Mapping)
                    else [],
                })
        if isinstance(prospective_risk, Mapping):
            valid_by_type = mask_stats.get("valid_actions_by_type", {}) if isinstance(mask_stats, Mapping) else {}
            has_cycle_risk = False
            for risk_type, risk_flags in prospective_risk.items():
                if risk_type in TARGET_REQUIRED_ACTION_TYPES:
                    continue
                if int(valid_by_type.get(risk_type, 0) or 0) > 0 and isinstance(risk_flags, Mapping) and bool(risk_flags.get("cycle")):
                    has_cycle_risk = True
                    break
            if has_cycle_risk:
                self.cycle_risk_available_steps += 1
                if isinstance(selected_risk, Mapping) and bool(selected_risk.get("cycle")):
                    self.cycle_risk_selected_count += 1
                else:
                    self.cycle_risk_zero_selected_count += 1
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
                self.repeat_breakdown[key] += 1
                self.action_type_repeat_breakdown.setdefault(action_type, Counter())[key] += 1
        repeat_kinds = repeat_diagnostic.get("repeat_kinds", [])
        if isinstance(repeat_kinds, list):
            for repeat_kind in repeat_kinds:
                key = f"repeat_kind:{repeat_kind}"
                self.repeat_breakdown[key] += 1
                self.action_type_repeat_breakdown.setdefault(action_type, Counter())[key] += 1

        if self.mode == "dqn":
            no_effect = bool(
                dqn_no_effect_diagnostic(before_observation, after_observation, action).get("no_effect")
            )
        else:
            no_effect = _is_no_effect(before_observation, after_observation)
        if no_effect:
            self.no_effect_count += 1
        if action.get("invalid") or action.get("mask_violation") or action.get("candidate_missing"):
            self.invalid_action_count += 1
        has_action_error = bool(action.get("failed") or action.get("last_action_error") or any(str(item.get("type") or "") == "action-error" for item in anomalies))
        if has_action_error:
            self.action_error_count += 1
            self.action_errors_by_type[action_type] += 1
            category = str(info.get("action_error_category") or "")
            if not category:
                category = normalize_action_error(
                    info.get("action_error_summary")
                    or info.get("action_error")
                    or info.get("fallback_error")
                    or info.get("fill_error")
                    or info.get("press_error")
                    or info.get("viewport_error")
                    or info.get("network_error")
                    or info.get("console_error")
                )
            self.action_error_categories[category or "other-action-error"] += 1
            if target:
                self.failed_target_ids[target] += 1
            self.action_error_details.append(
                _action_error_detail(
                    step=step,
                    action=action,
                    action_type=action_type,
                    target=target,
                    candidate=action_candidate,
                    info=info,
                    mask_stats=mask_stats,
                    category=category or "other-action-error",
                )
            )
        if _has_timeout(action, after_observation) or any(str(item.get("rule_id") or item.get("evidence", {}).get("rule_id") or "") == "R15" for item in anomalies):
            self.timeout_count += 1
        if _flow_progress_key(before_observation, after_observation, action) or _positive_component(reward_components, ("flow_progress",)):
            self.flow_progress_count += 1
        if action_candidate and _is_synthetic_target(action_candidate):
            self.synthetic_target_selection_count += 1
        if action_candidate and not _is_candidate_resolvable_for_action(action_type, action_candidate):
            self.non_resolvable_target_selection_count += 1
        if bool(info.get("fallback_info") or info.get("fallback_error")):
            self.fallback_action_count += 1

        sanitized_anomalies = []
        for anomaly in anomalies:
            signature = str(anomaly.get("signature") or "")
            status = str(anomaly.get("status") or "")
            if not signature:
                continue
            if status == "confirmed":
                if signature in self.confirmed_signatures:
                    self.repeat_confirmed_anomaly_count += 1
                    self.repeated_confirmed_signatures[signature] += 1
                if signature in self.candidate_signatures:
                    self.candidate_to_confirmed_signatures.add(signature)
                if _is_strong_evidence_confirmed(anomaly):
                    self.strong_evidence_confirmed_signatures.add(signature)
                self.confirmed_signatures.add(signature)
            elif status == "candidate":
                self.candidate_signatures.add(signature)
            sanitized_anomalies.append(
                {
                    "rule_id": anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id"),
                    "status": status,
                    "signature": signature,
                }
            )

        for key, value in reward_components.items():
            if isinstance(value, (int, float)):
                self.reward_component_totals[str(key)] += float(value)

        self.step_summaries.append(
            {
                "step": step,
                "action_type": action_type,
                "target_id": target,
                "url_path": url,
                "state_new": state_new,
                "target_new": target_new,
                "action_target_new": action_target_new,
                "local_state_new": local_state_new,
                "local_action_target_new": local_action_target_new,
                "reward": float(reward),
                "action_identity": selected_action_identity,
                "mask_stats": dict(mask_stats),
                "action_error_category": info.get("action_error_category", ""),
                "action_error_summary": info.get("action_error_summary", ""),
                "selected_candidate": info.get("selected_candidate", {}),
                "selected_probability": action.get("selected_probability"),
                "action_type_probability_mass": action.get("action_type_probability_mass", {}),
                "q_diagnostic": dict(q_diagnostic),
                "confirmation_context": dict(action.get("confirmation_context", {})),
                "history_state_diagnostic": {
                    "history_valid_count": history_valid_count,
                    "active_cycle_length": active_cycle_length,
                    "same_action_identity_streak": same_identity_streak,
                    "distinct_action_identity_count": distinct_recent_identity_count,
                    "cycle_continuation_identity": cycle_continuation_identity,
                    "selected_action_identity": selected_action_identity,
                    "cycle_feature_next_repeats_cycle": bool(
                        active_cycle_length > 0
                        and selected_action_identity
                        and selected_action_identity == cycle_continuation_identity
                    ),
                    "previous_unproductive_repeat": bool(self.previous_transition_was_unproductive_repeat),
                    "previous_unproductive_repeat_identity": self.previous_unproductive_repeat_identity,
                    "previous_unproductive_next_changes_identity": bool(
                        self.previous_transition_was_unproductive_repeat
                        and selected_action_identity
                        and selected_action_identity != self.previous_unproductive_repeat_identity
                    ),
                },
                "confirmation_transition_events": list(action.get("confirmation_transition_events", [])),
                "anomalies": sanitized_anomalies,
            }
        )
        self.previous_transition_was_unproductive_repeat = bool(repeat_diagnostic.get("unproductive_repeat"))
        self.previous_unproductive_repeat_identity = (
            str(repeat_diagnostic.get("action_identity") or selected_action_identity)
            if self.previous_transition_was_unproductive_repeat
            else ""
        )

    def _record_mask_stats(self, mask_stats: Mapping[str, Any]) -> None:
        if not mask_stats:
            return
        valid_action_count = int(mask_stats.get("valid_action_count", 0) or 0)
        valid_type_count = int(mask_stats.get("valid_action_type_count", 0) or 0)
        self.valid_action_count_total += valid_action_count
        self.valid_action_type_count_total += valid_type_count
        if valid_action_count <= 0:
            self.empty_mask_count += 1
        by_type = mask_stats.get("valid_actions_by_type", {})
        if isinstance(by_type, Mapping):
            for key, value in by_type.items():
                self.valid_actions_by_type[str(key)] += int(value or 0)

    def summary(self) -> dict[str, Any]:
        total = self.total_steps
        actionable = max(1, sum(self.action_distribution.values()))
        metrics: dict[str, Any] = {
            "total_steps": total,
            "candidate_anomaly_count": len(self.candidate_signatures),
            "unique_candidate_signatures": len(self.candidate_signatures),
            "confirmed_anomaly_count": len(self.confirmed_signatures),
            "unique_confirmed_signatures": len(self.confirmed_signatures),
            "candidate_to_confirmed_count": len(self.candidate_to_confirmed_signatures),
            "candidate_confirmation_rate": _ratio(
                len(self.candidate_to_confirmed_signatures),
                len(self.candidate_signatures - self.strong_evidence_confirmed_signatures),
            ),
            "no_effect_count": self.no_effect_count,
            "invalid_action_count": self.invalid_action_count,
            "action_error_count": self.action_error_count,
            "timeout_count": self.timeout_count,
            "flow_progress_count": self.flow_progress_count,
            "valid_action_count_avg": _ratio(self.valid_action_count_total, total),
            "valid_action_type_count_avg": _ratio(self.valid_action_type_count_total, total),
            "empty_mask_count": self.empty_mask_count,
            "fallback_action_count": self.fallback_action_count,
            "synthetic_target_selection_count": self.synthetic_target_selection_count,
            "non_resolvable_target_selection_count": self.non_resolvable_target_selection_count,
            "action_type_count": len(self.action_distribution),
            "action_type_entropy": _counter_entropy(self.action_distribution),
            "policy_entropy_mean": float(np.mean(self.policy_entropies)) if self.policy_entropies else 0.0,
            "policy_entropy_min": float(np.min(self.policy_entropies)) if self.policy_entropies else 0.0,
            "policy_entropy_max": float(np.max(self.policy_entropies)) if self.policy_entropies else 0.0,
            "pending_state_step_count": self.pending_state_step_count,
            "reproduction_opportunity_count": self.reproduction_opportunity_count,
            "expected_action_type_selected_count": self.expected_action_type_selected_count,
            "expected_action_and_target_selected_count": self.expected_action_and_target_selected_count,
            "action_type_match_rate": _ratio(self.expected_action_type_selected_count, self.reproduction_opportunity_count),
            "full_reproduction_match_rate": _ratio(self.expected_action_and_target_selected_count, self.reproduction_opportunity_count),
            "confirmation_progress_count": self.confirmation_progress_count,
            "confirmed_count": self.confirmed_count,
            "confirmed_observation_count": self.confirmed_observation_count,
            "newly_confirmed_transition_count": self.newly_confirmed_transition_count,
            "confirmed_reward_event_count": self.confirmed_reward_event_count,
            "confirmed_reward_duplicate_attempt_count": self.confirmed_reward_duplicate_attempt_count,
            "confirmed_reward_duplicate_blocked_count": self.confirmed_reward_duplicate_blocked_count,
            "unique_confirmed_logical_count": len(self.confirmed_logical_keys),
            "action_reproduction_pending_count": self.action_reproduction_pending_count,
            "observation_reproduction_pending_count": self.observation_reproduction_pending_count,
            "action_reproduction_opportunity_count": self.action_reproduction_opportunity_count,
            "action_reproduction_executable_count": self.action_reproduction_executable_count,
            "action_reproduction_safe_mask_valid_count": self.action_reproduction_safe_mask_valid_count,
            "action_reproduction_type_match_count": self.action_reproduction_type_match_count,
            "action_reproduction_full_match_count": self.action_reproduction_full_match_count,
            "observation_reproduction_event_count": self.observation_reproduction_event_count,
            "action_reproduction_executable_ratio": _ratio(
                self.action_reproduction_executable_count,
                self.action_reproduction_opportunity_count,
            ),
            "rule_mode_funnel": {rule: dict(counter) for rule, counter in sorted(self.rule_mode_funnel.items())},
        }
        confirmation_learning_summary = self.confirmation_funnel.summary()
        metrics.update(
            {
                "expected_type_rank_mean": confirmation_learning_summary.get("expected_type_rank_mean"),
                "expected_target_rank_mean": confirmation_learning_summary.get("expected_target_rank_mean"),
                "expected_type_context_delta_mean": confirmation_learning_summary.get("expected_type_context_delta_mean"),
                "context_selection_flip_rate": confirmation_learning_summary.get("context_selection_flip_rate"),
            }
        )
        if self.mode == "ppo":
            metrics.update(
                {
                    "unique_states": len(self.states),
                    "state_novelty_ratio": _ratio(len(self.states), total),
                    "unique_urls": len(self.urls),
                    "url_novelty_ratio": _ratio(len(self.urls), total),
                    "unique_targets": len(self.targets),
                    "target_novelty_ratio": _ratio(len(self.targets), actionable),
                    "unique_action_targets": len(self.action_targets),
                    "action_target_novelty_ratio": _ratio(len(self.action_targets), actionable),
                    "unique_action_types": len(self.action_distribution),
                    "repeat_action_target_count": self.repeat_action_target_count,
                    "repeat_action_target_ratio": _ratio(self.repeat_action_target_count, actionable),
                    "repeat_state_count": self.repeat_state_count,
                    "repeat_state_ratio": _ratio(self.repeat_state_count, total),
                    "exploration_score": self._exploration_score(),
                }
            )
        else:
            repeat_patterns = _repeat_pattern_summary(self.action_target_history, window=4)
            scroll_summary = _scroll_cycle_summary(self.action_type_sequence)
            selected_count = self.candidate_independent_selected_count
            metrics.update(
                {
                    "unique_local_states": len(self.local_states),
                    "local_state_novelty_ratio": _ratio(len(self.local_states), total),
                    "unique_local_action_targets": len(self.local_action_targets),
                    "local_action_novelty_ratio": _ratio(len(self.local_action_targets), actionable),
                    "repeat_local_action_count": self.repeat_local_action_count,
                    "repeat_local_action_ratio": _ratio(self.repeat_local_action_count, actionable),
                    "immediate_repeat_count": repeat_patterns["immediate_repeat_count"],
                    "immediate_repeat_ratio": repeat_patterns["immediate_repeat_ratio"],
                    "recent_window_repeat_count": repeat_patterns["recent_window_repeat_count"],
                    "recent_window_repeat_ratio": repeat_patterns["recent_window_repeat_ratio"],
                    "cycle_repeat_count": repeat_patterns["cycle_repeat_count"],
                    "cycle_repeat_ratio": repeat_patterns["cycle_repeat_ratio"],
                    "unproductive_repeat_count": self.repeat_breakdown["unproductive_repeat"],
                    "unproductive_repeat_ratio": _ratio(self.repeat_breakdown["unproductive_repeat"], actionable),
                    "repeat_escape_rate": _ratio(
                        self.previous_unproductive_next_changes_identity_count,
                        self.previous_unproductive_repeat_count,
                    ),
                    "cycle_escape_rate": _ratio(
                        self.cycle_feature_next_breaks_count,
                        self.cycle_feature_active_count,
                    ),
                    "steps_with_full_history": self.history_steps_with_full_history,
                    "steps_with_cycle_feature": self.history_steps_with_cycle_feature,
                    "steps_with_streak_gt1": self.history_steps_with_streak_gt1,
                    "mean_distinct_recent_identities": (
                        float(np.mean(self.history_distinct_recent_identity_counts))
                        if self.history_distinct_recent_identity_counts
                        else 0.0
                    ),
                    "cycle_feature_next_repeats_count": self.cycle_feature_next_repeats_count,
                    "cycle_feature_next_breaks_count": self.cycle_feature_next_breaks_count,
                    "previous_unproductive_repeat_followup_count": self.previous_unproductive_repeat_count,
                    "previous_unproductive_next_changes_identity_count": self.previous_unproductive_next_changes_identity_count,
                    "previous_unproductive_next_same_identity_count": self.previous_unproductive_next_same_identity_count,
                    "useful_reproduction_repeat_count": self.repeat_breakdown["useful_reproduction_repeat"],
                    "repeat_penalty_count": self.repeat_breakdown["repeat_penalty_applied"],
                    "repeat_local_state_count": self.repeat_local_state_count,
                    "repeat_confirmed_anomaly_count": self.repeat_confirmed_anomaly_count,
                    "scroll_up_count": self.action_distribution.get("scroll_up", 0),
                    "scroll_down_count": self.action_distribution.get("scroll_down", 0),
                    **scroll_summary,
                    "selected_immediate_risk_rate": _ratio(self.selected_risk_counts["immediate"], selected_count),
                    "selected_recent_risk_rate": _ratio(self.selected_risk_counts["recent"], selected_count),
                    "selected_cycle_risk_rate": _ratio(self.selected_risk_counts["cycle"], selected_count),
                    "cycle_risk_available_steps": self.cycle_risk_available_steps,
                    "cycle_risk_selected_count": self.cycle_risk_selected_count,
                    "cycle_risk_zero_selected_count": self.cycle_risk_zero_selected_count,
                    "cycle_risk_avoidance_rate": _ratio(
                        self.cycle_risk_zero_selected_count,
                        self.cycle_risk_available_steps,
                    ),
                    "prospective_repeat_risk_validation": _prospective_validation_summary(
                        self.prospective_repeat_validation
                    ),
                }
            )
        return {
            "mode": self.mode,
            "metrics": metrics,
            "action_distribution": dict(self.action_distribution),
            "selected_actions_by_type": dict(self.selected_actions_by_type),
            "valid_actions_by_type": dict(self.valid_actions_by_type),
            "action_errors_by_type": dict(self.action_errors_by_type),
            "repeat_breakdown": dict(self.repeat_breakdown),
            "action_type_repeat_breakdown": {
                action_type: dict(counter)
                for action_type, counter in self.action_type_repeat_breakdown.items()
            },
            "action_type_repeat_summary": _action_type_repeat_summary(
                self.action_distribution,
                self.action_type_repeat_breakdown,
                self.reward_values_by_action_type,
            ),
            "action_error_categories": dict(self.action_error_categories),
            "error_rate_by_action_type": _error_rate_by_action_type(self.action_errors_by_type, self.action_distribution),
            "failed_target_ids": [
                {"target_id": key, "count": count}
                for key, count in self.failed_target_ids.most_common(10)
            ],
            "action_error_details": self.action_error_details,
            "q_value_diagnostic_summary": _q_diagnostic_summary(self.q_diagnostics),
            "confirmation_learning_summary": confirmation_learning_summary,
            "reward_component_totals": dict(self.reward_component_totals),
            "candidate_signatures": sorted(self.candidate_signatures),
            "confirmed_signatures": sorted(self.confirmed_signatures),
            "candidate_to_confirmed_signatures": sorted(self.candidate_to_confirmed_signatures),
            "strong_evidence_confirmed_signatures": sorted(self.strong_evidence_confirmed_signatures),
            "repeated_confirmed_signatures": [
                {"signature": key, "count": count}
                for key, count in self.repeated_confirmed_signatures.most_common()
            ],
            "top_repeated_actions": [
                {"action_target": key, "count": count}
                for key, count in self.repeated_actions.most_common(5)
            ],
            "step_summaries": self.step_summaries,
            "prospective_repeat_risk_examples": list(self.prospective_repeat_examples),
        }

    def _exploration_score(self) -> float:
        total = max(1, self.total_steps)
        return max(
            0.0,
            (
                _ratio(len(self.states), total)
                + _ratio(len(self.targets), total)
                + _ratio(len(self.action_targets), total)
                + _ratio(self.flow_progress_count, total)
                - _ratio(self.repeat_action_target_count + self.repeat_state_count, total)
            )
            / 4.0,
        )


def run_sanity_check(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    base_url = str(site["base_url"])
    site_id = str(site["site_id"])
    env = BrowserGymJAWSEnv(
        site_id=site_id,
        base_url=base_url,
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    aggregator = SanityMetricsAggregator(args.mode)
    commands = _commands(args, site_id, base_url)
    selection_mode = _selection_mode(args)
    reward_history: dict[str, Any] = {}
    checkpoint_relation = _checkpoint_reward_relation(args.model_path)
    history_counterfactuals: list[dict[str, Any]] = []
    histv2_cycle_q_rows: list[dict[str, Any]] = []
    try:
        if args.mode == "ppo":
            agent = PPOAgent(env.observation_encoder.get_obs_dim(), env.action_space.get_action_dim())
            if args.model_path and Path(args.model_path).exists():
                agent.load(Path(args.model_path))
            policy = agent
        else:
            if args.dqn_architecture in {FACTORIZED_DQN_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
                q_network = FactorizedQNetwork(
                    env.observation_encoder.get_obs_dim()
                    + (CONFIRMATION_CONTEXT_DIM if args.dqn_architecture in {FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE} else 0)
                    + (
                        HISTORY_CONTEXT_DIM if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE
                        else HISTORY_CONTEXT_V2_DIM if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE
                        else 0
                    ),
                    len(env.action_space.action_types),
                    env.action_space.max_candidates,
                )
            else:
                q_network = QNetwork(env.observation_encoder.get_obs_dim(), env.action_space.get_action_dim())
            if args.model_path and Path(args.model_path).exists():
                payload = torch.load(Path(args.model_path), map_location="cpu")
                _validate_checkpoint_architecture(payload, args.dqn_architecture)
                q_network.load_state_dict(payload.get("model", payload))
            q_network.eval()
            policy = q_network

        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            done = False
            reward_history = {}
            if args.mode == "dqn":
                initialize_dqn_candidate_reward_baseline(
                    reward_history,
                    _sanitize_anomalies(detect_anomalies(observation, observation, {"action": {"action_type": "noop", "candidate_index": 0}})),
                )
            for step in range(1, args.max_steps + 1):
                if args.dqn_architecture in {FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
                    obs_vector, confirmation_metadata = _dqn_state(env, observation, reward_history, args.dqn_architecture)
                else:
                    obs_vector = env.observation_encoder.encode_observation(observation)
                    confirmation_metadata = {"pending_confirmation_exists": False, "matching_candidate_slots": []}
                if args.mode == "ppo":
                    mask = env.action_space.build_action_mask(observation)
                    policy_context = _ppo_policy_context(policy, obs_vector, mask, env.action_space, observation)
                    if selection_mode == "greedy":
                        selected = policy.select_greedy_action(obs_vector, mask)
                    else:
                        selected = policy.select_action(obs_vector, mask)
                    action_id = int(selected["action_id"])
                else:
                    mask = safe_action_mask(env.action_space, observation)
                    if args.dqn_architecture in {FACTORIZED_DQN_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
                        action_id = select_factorized_greedy_action(policy, obs_vector, mask, env.action_space)
                    else:
                        action_id = select_greedy_action(
                            policy,
                            obs_vector,
                            mask,
                            env.action_space,
                            mode=args.dqn_greedy_selection_mode,
                        )
                if args.mode == "dqn" and args.dqn_architecture in {FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
                    base = env.observation_encoder.encode_observation(observation)
                    confirmation, _ = build_dqn_confirmation_context(reward_history, observation, env.action_space)
                    zero_state = (
                        augment_dqn_history_observation(base, confirmation, np.zeros(HISTORY_CONTEXT_DIM, dtype=np.float32))
                        if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE
                        else augment_dqn_history_observation_v2(base, confirmation, np.zeros(HISTORY_CONTEXT_V2_DIM, dtype=np.float32))
                    )
                    with torch.inference_mode():
                        actual_types = policy(torch.from_numpy(obs_vector).float().unsqueeze(0))["type_values"].squeeze(0)
                        zero_types = policy(torch.from_numpy(zero_state).float().unsqueeze(0))["type_values"].squeeze(0)
                    actual_action = env.action_space.decode(action_id)
                    zero_action = env.action_space.decode(select_factorized_greedy_action(policy, zero_state, mask, env.action_space))
                    actual_type = str(actual_action.get("action_type") or "")
                    zero_type = str(zero_action.get("action_type") or "")
                    history_metadata = confirmation_metadata.get("history_context", {})
                    recent_identities = history_metadata.get("recent_action_identities", []) if isinstance(history_metadata, Mapping) else []
                    active_cycle_length = int(history_metadata.get("active_cycle_length", 0) or 0) if isinstance(history_metadata, Mapping) else 0
                    streak = int(history_metadata.get("same_action_identity_streak", 0) or 0) if isinstance(history_metadata, Mapping) else 0
                    cycle_identity = (
                        str(recent_identities[active_cycle_length - 1] or "")
                        if isinstance(recent_identities, list) and active_cycle_length > 0 and len(recent_identities) >= active_cycle_length
                        else ""
                    )
                    streak_identity = str(recent_identities[0] or "") if isinstance(recent_identities, list) and recent_identities else ""
                    previous_unproductive_identity = ""
                    outcome_flags = history_metadata.get("outcome_flags", []) if isinstance(history_metadata, Mapping) else []
                    if isinstance(outcome_flags, list) and outcome_flags and isinstance(outcome_flags[0], list) and len(outcome_flags[0]) >= 4 and bool(outcome_flags[0][3]):
                        previous_unproductive_identity = streak_identity
                    real_candidate = _candidate_for_decoded_action(observation, actual_action)
                    zero_candidate = _candidate_for_decoded_action(observation, zero_action)
                    real_identity = history_action_identity(
                        actual_type,
                        real_candidate,
                        requires_target=actual_type in {"click_element", "fill_input", "press_enter"},
                    )
                    zero_identity = history_action_identity(
                        zero_type,
                        zero_candidate,
                        requires_target=zero_type in {"click_element", "fill_input", "press_enter"},
                    )
                    repeated_identities = {
                        identity
                        for identity in (cycle_identity, streak_identity if streak > 1 else "", previous_unproductive_identity)
                        if identity
                    }
                    history_counterfactuals.append({
                        "history_valid_count": int(sum(history_metadata.get("history_valid_mask", []))),
                        "active_cycle_length": active_cycle_length,
                        "same_action_identity_streak": streak,
                        "previous_unproductive_repeat": bool(previous_unproductive_identity),
                        "cycle_continuation_identity": cycle_identity,
                        "streak_identity": streak_identity if streak > 1 else "",
                        "previous_unproductive_repeat_identity": previous_unproductive_identity,
                        "selected_identity_real": real_identity,
                        "selected_identity_zero": zero_identity,
                        "real_repeats_relevant_identity": bool(real_identity in repeated_identities),
                        "zero_repeats_relevant_identity": bool(zero_identity in repeated_identities),
                        "real_breaks_cycle": bool(active_cycle_length > 0 and real_identity != cycle_identity),
                        "zero_breaks_cycle": bool(active_cycle_length > 0 and zero_identity != cycle_identity),
                        "selected_type_real": actual_type,
                        "selected_type_zero": zero_type,
                        "selected_type_flip": bool(actual_type != zero_type),
                        "type_value_l1": float(torch.sum(torch.abs(actual_types - zero_types)).item()),
                        "type_value_l2": float(torch.linalg.vector_norm(actual_types - zero_types).item()),
                        "type_value_max_abs": float(torch.max(torch.abs(actual_types - zero_types)).item()),
                    })
                    if args.dqn_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE:
                        q_row = _histv2_cycle_q_diagnostic(
                            policy,
                            obs_vector,
                            mask,
                            env.action_space,
                            history_metadata if isinstance(history_metadata, Mapping) else {},
                        )
                        if q_row:
                            histv2_cycle_q_rows.append(q_row)
                q_diagnostic = (
                    factorized_q_value_diagnostics(policy, obs_vector, mask, env.action_space, action_id)
                    if args.mode == "dqn" and args.dqn_architecture in {FACTORIZED_DQN_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}
                    else q_value_diagnostics(policy, obs_vector, mask, env.action_space, action_id, top_k=5)
                    if args.mode == "dqn"
                    else {}
                )
                mask_stats = env.action_space.build_action_mask_stats(observation, mask, action_id)
                action = env.action_space.decode(action_id)
                action["action_id"] = action_id
                if args.mode == "dqn":
                    mode = str(confirmation_metadata.get("confirmation_mode") or "")
                    expected_type = str(confirmation_metadata.get("expected_action_type") or "")
                    if mode == ACTION_REPRODUCTION and expected_type:
                        instruction = ExpectedReproduction(
                            action_type=expected_type,
                            target_identity=str(confirmation_metadata.get("expected_target_identity") or ""),
                            mode=ACTION_REPRODUCTION,
                            rule_id=str(confirmation_metadata.get("active_rule_id") or ""),
                        )
                        executable = is_reproduction_instruction_executable(
                            instruction,
                            action_space=env.action_space,
                            observation=observation,
                            mask=mask,
                        )
                        action["confirmation_action_executable"] = bool(executable.get("semantically_executable"))
                        action["confirmation_action_mask_valid"] = bool(executable.get("mask_valid"))
                if (
                    args.mode == "dqn"
                    and args.dqn_architecture in {FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}
                    and is_reproduction_opportunity(confirmation_metadata)
                ):
                    action["confirmation_type_diagnostic"] = factorized_confirmation_step_diagnostic(
                        policy,
                        obs_vector,
                        mask,
                        env.action_space,
                        action_id,
                        confirmation_metadata,
                    )
                    action["confirmation_context_counterfactual"] = real_vs_zero_context_diagnostic(
                        policy,
                        obs_vector,
                        mask,
                        env.action_space,
                        confirmation_metadata,
                    )
                if args.mode == "ppo":
                    action["policy_entropy"] = float(selected.get("entropy", 0.0))
                    action["selected_probability"] = _selected_probability(selected, policy_context, action_id)
                    action["action_type_probability_mass"] = policy_context.get("action_type_probability_mass", {})
                    action["top_valid_actions"] = policy_context.get("top_valid_actions", [])
                next_observation, _, done, info = env.step(action_id)
                action["failed"] = bool(info.get("last_action_error"))
                action["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = _sanitize_anomalies(detect_anomalies(observation, next_observation, {**info, "action": action}))
                if args.mode == "ppo":
                    reward, components = calculate_autonomous_reward(observation, next_observation, action, anomalies, [], reward_history, {})
                else:
                    reward, components = calculate_dqn_reward(observation, next_observation, action, anomalies, reward_history)
                    action["repeat_diagnostic"] = dict(reward_history.get("last_repeat_diagnostic", {}))
                    action["confirmation_transition_events"] = list(reward_history.get("last_confirmation_transition_events", []))
                    action["confirmation_observation_events"] = int(reward_history.get("last_confirmation_observation_events", 0) or 0)
                    action["confirmation_duplicate_attempts"] = int(reward_history.get("last_confirmation_reward_duplicate_attempts", 0) or 0)
                    action["confirmation_duplicate_blocks"] = int(reward_history.get("last_confirmation_reward_duplicate_blocks", 0) or 0)
                    action["confirmation_context"] = dict(confirmation_metadata)
                    confirmation_mode = str(confirmation_metadata.get("confirmation_mode") or "")
                    mode_progress = bool(
                        reward_history.get("last_confirmation_progress")
                        and str(reward_history.get("last_confirmation_mode") or "") == confirmation_mode
                    )
                    mode_confirmed = bool(
                        reward_history.get("last_confirmation_confirmed")
                        and str(reward_history.get("last_confirmation_mode") or "") == confirmation_mode
                    )
                    action["confirmation_context"]["confirmation_progress_after"] = bool(
                        action["repeat_diagnostic"].get("confirmation_progress")
                        or mode_progress
                    )
                    action["confirmation_context"]["confirmation_confirmed_after"] = mode_confirmed
                    if args.dqn_architecture in {FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE}:
                        repeat_diagnostic = action.get("repeat_diagnostic", {})
                        candidates = observation.get("candidate_elements", [])
                        candidate_index = int(action.get("candidate_index", -1))
                        candidate = candidates[candidate_index] if isinstance(candidates, list) and 0 <= candidate_index < len(candidates) else None
                        append_history_transition(
                            reward_history,
                            action_type=str(action.get("action_type") or ""),
                            candidate=candidate if isinstance(candidate, Mapping) else None,
                            meaningful_action_effect=not bool(repeat_diagnostic.get("no_effect", False)),
                            productive_transition=bool(repeat_diagnostic.get("productive_transition")),
                            confirmation_progress=mode_progress,
                            unproductive_repeat=bool(repeat_diagnostic.get("unproductive_repeat")),
                        )
                aggregator.record_step(
                    step=(episode - 1) * args.max_steps + step,
                    before_observation=observation,
                    after_observation=next_observation,
                    action=action,
                    reward=reward,
                    reward_components=components,
                    anomalies=anomalies,
                    info=info,
                    mask_stats=mask_stats,
                    q_diagnostic=q_diagnostic,
                )
                observation = next_observation
                if done:
                    break
    finally:
        env.close()
    summary = aggregator.summary()
    summary.update(
        {
            "site": site_id,
            "url": base_url,
            "model": args.model_path,
            "checkpoint": args.model_path,
            "checkpoint_created_before_or_after_reward_change": checkpoint_relation,
            "episode_count": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "max_candidates": args.max_candidates,
            "selection_mode": selection_mode,
            "dqn_architecture": args.dqn_architecture if args.mode == "dqn" else None,
            "dqn_greedy_selection_mode": args.dqn_greedy_selection_mode if args.mode == "dqn" else None,
            "history_counterfactual": {
                "count": len(history_counterfactuals),
                "valid_history_count": sum(item["history_valid_count"] > 0 for item in history_counterfactuals),
                "cycle_state_count": sum(item["active_cycle_length"] > 0 for item in history_counterfactuals),
                "selected_type_flip_rate": (
                    sum(item["selected_type_flip"] for item in history_counterfactuals) / len(history_counterfactuals)
                    if history_counterfactuals else 0.0
                ),
                "type_value_l1_mean": float(np.mean([item["type_value_l1"] for item in history_counterfactuals])) if history_counterfactuals else 0.0,
                "type_value_l2_mean": float(np.mean([item["type_value_l2"] for item in history_counterfactuals])) if history_counterfactuals else 0.0,
                "type_value_max_abs": float(max((item["type_value_max_abs"] for item in history_counterfactuals), default=0.0)),
                **_history_counterfactual_repeat_summary(history_counterfactuals),
                "rows": history_counterfactuals,
            },
            "histv2_cycle_q_diagnostic": _histv2_cycle_q_summary(histv2_cycle_q_rows),
            "type_score_method": (
                "normalized_logmeanexp"
                if args.mode == "dqn" and args.dqn_greedy_selection_mode == TYPE_LOGMEANEXP_SELECTION
                else None
            ),
            "commands": commands,
            "diagnostic": _diagnostic(summary),
        }
    )
    output = _write_summary(summary, args.output_dir, args.output_name)
    summary["summary_path"] = str(output)
    return summary


def _write_summary(summary: Mapping[str, Any], output_dir: str | Path, output_name: str = "") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = output_name or f"{summary.get('mode')}_{summary.get('site')}_sanity.json"
    output = path / filename
    output.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _sanitize_anomalies(anomalies: list[dict]) -> list[dict]:
    sanitized = []
    for anomaly in anomalies:
        item = {key: value for key, value in anomaly.items() if key not in GROUND_TRUTH_KEYS}
        evidence = item.get("evidence")
        if isinstance(evidence, dict):
            item["evidence"] = {key: value for key, value in evidence.items() if key not in GROUND_TRUTH_KEYS}
        sanitized.append(item)
    return sanitized


def _state_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return "|".join(
        [
            _url_path(observation),
            str(page_state.get("title") or ""),
            str(page_state.get("dom_node_count") or ""),
            str(page_state.get("page_text_length") or ""),
            _candidate_summary(observation),
        ]
    )


def _local_state_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    local_bits = []
    for key in (
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
    ):
        value = page_state.get(key)
        if value not in (None, ""):
            local_bits.append(f"{key}={_json_stable_value(value)}")
    return "|".join([_url_path(observation), str(page_state.get("title") or ""), ",".join(local_bits)])


def _candidate_summary(observation: Mapping[str, Any]) -> str:
    candidates = observation.get("candidate_elements", []) or []
    values = []
    if isinstance(candidates, list):
        for candidate in candidates[:12]:
            if isinstance(candidate, Mapping):
                values.append(_target_identity(candidate) or str(candidate.get("role") or candidate.get("tag") or ""))
    return ",".join(values)


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


def _candidate_for_decoded_action(observation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _action_candidate(observation, action)


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


def _url_path(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return _url_path_from_value(page_state.get("url"))


def _url_path_from_value(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    parsed = urlparse(text)
    return parsed.path or text.split("?", 1)[0] or "/"


def _flow_progress_key(before_observation: Mapping[str, Any], after_observation: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    for source in (action, after_observation.get("runtime_signals", {}), after_observation.get("page_state", {})):
        if isinstance(source, Mapping):
            for key in ("flow_progress", "flow_transition", "semantic_transition"):
                value = source.get(key)
                if value:
                    return str(value)
    return ""


def _is_no_effect(before_observation: Mapping[str, Any], after_observation: Mapping[str, Any]) -> bool:
    return _state_signature(before_observation) == _state_signature(after_observation)


def _has_timeout(action: Mapping[str, Any], after_observation: Mapping[str, Any]) -> bool:
    if action.get("timeout") or action.get("timed_out") or action.get("timeout_occurred"):
        return True
    for section_name in ("runtime_signals", "page_state"):
        section = after_observation.get(section_name, {})
        if isinstance(section, Mapping) and (section.get("timeout") or section.get("timed_out") or section.get("timeout_occurred")):
            return True
    return False


def _positive_component(components: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    return any(float(components.get(name, 0.0) or 0.0) > 0.0 for name in names)


def _is_synthetic_target(candidate: Mapping[str, Any]) -> bool:
    bid = str(candidate.get("bid") or "")
    if bid.startswith("-"):
        return True
    return str(candidate.get("source") or "") in {"bbox_fallback", "fullscreen_clickable_fallback", "emergency_bbox_fallback"}


def _is_candidate_resolvable_for_action(action_type: str, candidate: Mapping[str, Any]) -> bool:
    if action_type not in {"click_element", "fill_input", "press_enter"}:
        return True
    if not str(candidate.get("bid") or "").strip():
        return False
    if not bool(candidate.get("visible", True)) or not bool(candidate.get("enabled", True)):
        return False
    bbox = candidate.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        if float(bbox[2] or 0.0) <= 0.0 or float(bbox[3] or 0.0) <= 0.0:
            return False
    except (TypeError, ValueError):
        return False
    if action_type in {"fill_input", "press_enter"}:
        tag = str(candidate.get("tag") or "").lower()
        role = str(candidate.get("role") or "").lower()
        input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
        return input_type not in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"} and (
            bool(candidate.get("fillable")) or tag in {"input", "textarea"} or role in {"textbox", "searchbox", "combobox"}
        )
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return bool(
        candidate.get("clickable")
        or candidate.get("set_of_marks")
        or role in {"button", "link", "checkbox", "radio", "menuitem", "tab"}
        or tag in {"button", "a", "summary", "option", "label"}
    )


def _action_error_detail(
    *,
    step: int,
    action: Mapping[str, Any],
    action_type: str,
    target: str,
    candidate: Mapping[str, Any] | None,
    info: Mapping[str, Any],
    mask_stats: Mapping[str, Any],
    category: str,
) -> dict[str, Any]:
    selected_candidate = info.get("selected_candidate", {}) if isinstance(info.get("selected_candidate"), Mapping) else {}
    candidate_summary = selected_candidate if selected_candidate else _candidate_diagnostic_summary(candidate)
    action_type_mass = action.get("action_type_probability_mass", {})
    return {
        "step": int(step),
        "action_id": int(action.get("action_id", -1) or -1),
        "action_type": action_type,
        "candidate_index": int(action.get("candidate_index", 0) or 0),
        "target_id": target,
        "bid": str(candidate_summary.get("bid") or ""),
        "role": str(candidate_summary.get("role") or ""),
        "tag": str(candidate_summary.get("tag") or ""),
        "accessible_name": _sanitize_diagnostic_text(
            str((candidate or {}).get("name") or (candidate or {}).get("aria_label") or (candidate or {}).get("text") or "")
        ),
        "visible": bool(candidate_summary.get("visible", True)),
        "enabled": bool(candidate_summary.get("enabled", True)),
        "clickable": bool(candidate_summary.get("clickable", False)),
        "set_of_marks": bool(candidate_summary.get("set_of_marks", False)),
        "bbox": candidate_summary.get("bbox", []),
        "source": str(candidate_summary.get("source") or ""),
        "error_category": category,
        "error_summary": _short_diagnostic_error(
            info.get("action_error_summary")
            or info.get("action_error")
            or info.get("fallback_error")
            or info.get("fill_error")
            or info.get("press_error")
        ),
        "browsergym_raw_last_action_error": bool(info.get("browsergym_raw_last_action_error")),
        "execution_action_error": bool(info.get("execution_action_error", info.get("last_action_error"))),
        "selected_probability": action.get("selected_probability"),
        "action_type_probability_mass": action_type_mass.get(action_type) if isinstance(action_type_mass, Mapping) else None,
        "valid_action_count": int(mask_stats.get("valid_action_count", 0) or 0),
        "valid_action_type_count": int(mask_stats.get("valid_action_type_count", 0) or 0),
    }


def _error_rate_by_action_type(errors: Mapping[str, int], selected: Mapping[str, int]) -> dict[str, float]:
    return {
        str(action_type): _ratio(int(errors.get(action_type, 0) or 0), int(count or 0))
        for action_type, count in selected.items()
    }


def _candidate_diagnostic_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        return {}
    return {
        "bid": str(candidate.get("bid") or ""),
        "source": str(candidate.get("source") or ""),
        "role": str(candidate.get("role") or ""),
        "tag": str(candidate.get("tag") or ""),
        "visible": bool(candidate.get("visible", True)),
        "enabled": bool(candidate.get("enabled", True)),
        "clickable": bool(candidate.get("clickable", False)),
        "set_of_marks": bool(candidate.get("set_of_marks", False)),
        "bbox": list(candidate.get("bbox") or [])[:4] if isinstance(candidate.get("bbox"), list) else [],
    }


def _ppo_policy_context(
    policy: PPOAgent,
    obs_vector: np.ndarray,
    action_mask: np.ndarray,
    action_space,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        obs = torch.as_tensor(obs_vector, dtype=torch.float32, device=policy.device).unsqueeze(0)
        logits, _ = policy.policy(obs)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=policy.device).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask, -1e9)
        probabilities = torch.softmax(masked_logits, dim=-1).squeeze(0).detach().cpu().numpy()
        logits_np = masked_logits.squeeze(0).detach().cpu().numpy()
    return {
        "probabilities": probabilities,
        "action_type_probability_mass": _action_type_probability_mass(probabilities, action_mask, action_space),
        "top_valid_actions": _top_valid_actions(probabilities, logits_np, action_mask, action_space, observation, top_k=5),
    }


def _selected_probability(selected: Mapping[str, Any], policy_context: Mapping[str, Any], action_id: int) -> float:
    probabilities = policy_context.get("probabilities")
    if isinstance(probabilities, np.ndarray) and 0 <= int(action_id) < len(probabilities):
        return float(probabilities[int(action_id)])
    try:
        return float(np.exp(float(selected.get("log_prob", 0.0) or 0.0)))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _action_type_probability_mass(probabilities: np.ndarray, action_mask: np.ndarray, action_space) -> dict[str, float]:
    masses = {action_type: 0.0 for action_type in action_space.action_types}
    valid_total = 0.0
    for action_id in np.flatnonzero(action_mask > 0):
        decoded = action_space.decode(int(action_id))
        probability = float(probabilities[int(action_id)])
        masses[str(decoded["action_type"])] += probability
        valid_total += probability
    if valid_total > 0.0:
        for action_type in masses:
            masses[action_type] = float(masses[action_type] / valid_total)
    return masses


def _top_valid_actions(
    probabilities: np.ndarray,
    logits: np.ndarray,
    action_mask: np.ndarray,
    action_space,
    observation: Mapping[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    valid_ids = np.flatnonzero(action_mask > 0)
    ordered = sorted(valid_ids.tolist(), key=lambda action_id: float(probabilities[int(action_id)]), reverse=True)
    rows = []
    for action_id in ordered[:top_k]:
        decoded = action_space.decode(int(action_id))
        candidate = _action_candidate(observation, decoded)
        rows.append(
            {
                "action_id": int(action_id),
                "action_type": str(decoded.get("action_type") or ""),
                "candidate_index": int(decoded.get("candidate_index", 0) or 0),
                "target_id": _target_identity(candidate),
                "logit": float(logits[int(action_id)]),
                "probability": float(probabilities[int(action_id)]),
            }
        )
    return rows


def _short_diagnostic_error(error_text: Any, limit: int = 240) -> str:
    text = " ".join(str(error_text or "").split())
    return _sanitize_diagnostic_text(text[:limit], limit=limit)


def _sanitize_diagnostic_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    lowered = text.lower()
    for marker in ("password", "token", "authorization", "cookie"):
        if marker in lowered:
            return "[redacted]"
    return text[:limit]


def _is_strong_evidence_confirmed(anomaly: Mapping[str, Any]) -> bool:
    rule_id = str(anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id") or "")
    return rule_id in STRONG_EVIDENCE_RULE_IDS and str(anomaly.get("status") or "") == "confirmed"


def _diagnostic(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    mode = str(summary.get("mode") or "")
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), Mapping) else {}
    actions = summary.get("action_distribution", {}) if isinstance(summary.get("action_distribution"), Mapping) else {}
    total = int(metrics.get("total_steps", 0) or 0)
    dominant_ratio = 0.0
    dominant_action = ""
    if total and actions:
        dominant_action, dominant_count = max(((str(k), int(v or 0)) for k, v in actions.items()), key=lambda item: item[1])
        dominant_ratio = _ratio(dominant_count, total)
    if mode == "ppo":
        issues = []
        if float(metrics.get("state_novelty_ratio", 0.0) or 0.0) <= 0.2:
            issues.append("low_state_novelty")
        if float(metrics.get("repeat_action_target_ratio", 0.0) or 0.0) >= 0.6:
            issues.append("high_repeat_action_target_ratio")
        if dominant_ratio >= 0.7:
            issues.append(f"dominant_action:{dominant_action}")
        return {"PPO_ROLE_CHECK": "REVIEW" if issues else "PASS", "issues": issues}
    issues = []
    if float(metrics.get("local_action_novelty_ratio", 0.0) or 0.0) <= 0.2:
        issues.append("low_local_action_diversity")
    if int(metrics.get("repeat_confirmed_anomaly_count", 0) or 0) > int(metrics.get("unique_confirmed_signatures", 0) or 0):
        issues.append("confirmed_anomaly_repetition")
    if dominant_ratio >= 0.7:
        issues.append(f"dominant_action:{dominant_action}")
    return {"DQN_ROLE_CHECK": "REVIEW" if issues else "PASS", "issues": issues}


def _load_site_config(config_path: str, site_id: str | None, base_url: str | None) -> dict[str, Any]:
    if base_url:
        return {"site_id": site_id or "site001", "base_url": base_url}
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    sites = config.get("sites", []) if isinstance(config, Mapping) else []
    for site in sites:
        if isinstance(site, Mapping) and str(site.get("site_id") or "") == str(site_id or ""):
            return {"site_id": site.get("site_id"), "base_url": site.get("base_url")}
    raise ValueError(f"site_id {site_id!r} not found in {config_path}")


def _checkpoint_reward_relation(model_path: str) -> str:
    if not model_path:
        return "no checkpoint supplied"
    checkpoint = Path(model_path)
    if not checkpoint.exists():
        return "checkpoint missing"
    reward_files = [
        Path("services/autonomous_reward_service.py"),
        Path("dqn_error_discovery/runners/train_dqn_error_detector.py"),
    ]
    checkpoint_time = checkpoint.stat().st_mtime
    newest_reward_time = max((path.stat().st_mtime for path in reward_files if path.exists()), default=0.0)
    return "before_reward_change" if checkpoint_time < newest_reward_time else "after_reward_change_or_same_time"


def _validate_checkpoint_architecture(payload: Any, requested_architecture: str) -> None:
    if requested_architecture not in {
        FACTORIZED_DQN_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE,
        FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE,
    }:
        return
    if not isinstance(payload, Mapping):
        raise ValueError("factorized DQN checkpoint must include a manifest")
    manifest = payload.get("manifest", {})
    if not isinstance(manifest, Mapping) or manifest.get("dqn_architecture") != requested_architecture:
        raise ValueError(f"refusing to load checkpoint with architecture {manifest.get('dqn_architecture')!r} as {requested_architecture}")
    if requested_architecture == FACTORIZED_DQN_CONFOBS_ARCHITECTURE:
        if manifest.get("dqn_obs_dim") != 707 or manifest.get("confirmation_context_dim") != CONFIRMATION_CONTEXT_DIM:
            raise ValueError("factorized_v1_confobs checkpoint must declare dqn_obs_dim=707 and confirmation_context_dim=72")
    if requested_architecture == FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE:
        if manifest.get("dqn_obs_dim") != 945 or manifest.get("confirmation_context_dim") != CONFIRMATION_CONTEXT_DIM or manifest.get("history_context_dim") != HISTORY_CONTEXT_DIM:
            raise ValueError("factorized_v1_confobs_hist checkpoint must declare dqn_obs_dim=945, confirmation_context_dim=72, history_context_dim=238")
    if requested_architecture == FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE:
        if (
            manifest.get("dqn_obs_dim") != 976
            or manifest.get("confirmation_context_dim") != CONFIRMATION_CONTEXT_DIM
            or manifest.get("history_context_dim") != HISTORY_CONTEXT_V2_DIM
            or manifest.get("history_observation_version") != HISTORY_OBSERVATION_V2_VERSION
        ):
            raise ValueError("factorized_v1_confobs_histv2 checkpoint must declare dqn_obs_dim=976, confirmation_context_dim=72, history_context_dim=269, history_observation_version=v2")


def _commands(args: argparse.Namespace, site_id: str, base_url: str) -> Mapping[str, str]:
    del base_url
    script = "dqn_error_discovery\\runners\\run_policy_sanity_check.py"
    parts = [
        "python",
        script,
        "--mode",
        args.mode,
        "--config",
        args.config,
        "--site-id",
        site_id,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--max-candidates",
        str(args.max_candidates),
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
    ]
    if args.model_path:
        parts.extend(["--model-path", args.model_path])
    selection_mode = _selection_mode(args)
    if args.mode == "ppo":
        parts.extend(["--selection-mode", selection_mode])
    if args.mode == "dqn":
        parts.extend(["--dqn-architecture", args.dqn_architecture])
        parts.extend(["--dqn-greedy-selection-mode", args.dqn_greedy_selection_mode])
    return {"sanity_command": " ".join(parts)}


def _selection_mode(args: argparse.Namespace) -> str:
    requested = getattr(args, "selection_mode", "") or ""
    if requested:
        return str(requested)
    if getattr(args, "greedy", False):
        return "greedy"
    return "stochastic" if getattr(args, "mode", "") == "ppo" else "greedy"


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _counter_entropy(counter: Mapping[str, int]) -> float:
    total = sum(int(value or 0) for value in counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counter.values():
        count = int(value or 0)
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * float(np.log(probability))
    return entropy


def _repeat_pattern_summary(action_targets: list[str], window: int = 4) -> dict[str, Any]:
    if not action_targets:
        return {
            "immediate_repeat_count": 0,
            "immediate_repeat_ratio": 0.0,
            "recent_window_repeat_count": 0,
            "recent_window_repeat_ratio": 0.0,
            "cycle_repeat_count": 0,
            "cycle_repeat_ratio": 0.0,
        }
    immediate = 0
    recent = 0
    cycle = 0
    for index, target in enumerate(action_targets):
        if index > 0 and target == action_targets[index - 1]:
            immediate += 1
        if target in action_targets[max(0, index - window):index]:
            recent += 1
        if any(index >= span and target == action_targets[index - span] for span in (1, 2, 3)):
            cycle += 1
    total = len(action_targets)
    return {
        "immediate_repeat_count": immediate,
        "immediate_repeat_ratio": _ratio(immediate, total),
        "recent_window_repeat_count": recent,
        "recent_window_repeat_ratio": _ratio(recent, total),
        "cycle_repeat_count": cycle,
        "cycle_repeat_ratio": _ratio(cycle, total),
    }


def _scroll_cycle_summary(action_types: list[str], window: int = 4) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    active_start: int | None = None
    previous_scroll = ""
    for index, action_type in enumerate(action_types):
        is_scroll = action_type in {"scroll_up", "scroll_down"}
        alternates = is_scroll and previous_scroll in {"scroll_up", "scroll_down"} and action_type != previous_scroll
        if alternates and active_start is None:
            active_start = index - 1
        elif (not alternates) and active_start is not None:
            segments.append(
                {
                    "cycle_start_step": active_start + 1,
                    "cycle_end_step": index,
                    "cycle_length": index - active_start,
                    "unique_actions_inside_cycle": sorted(set(action_types[active_start:index])),
                }
            )
            active_start = None
        previous_scroll = action_type if is_scroll else ""
    if active_start is not None:
        segments.append(
            {
                "cycle_start_step": active_start + 1,
                "cycle_end_step": len(action_types),
                "cycle_length": len(action_types) - active_start,
                "unique_actions_inside_cycle": sorted(set(action_types[active_start:])),
            }
        )
    reentry_count = _cycle_reentry_count(segments, window=window)
    escape_count = len(segments)
    lengths = [int(segment["cycle_length"]) for segment in segments]
    return {
        "alternating_scroll_cycle_count": len(segments),
        "longest_alternating_scroll_cycle_length": max(lengths, default=0),
        "scroll_cycle_reentry_count": reentry_count,
        "scroll_cycle_reentry_rate": _ratio(reentry_count, escape_count),
        "scroll_cycle_duration_mean": float(np.mean(lengths)) if lengths else 0.0,
        "scroll_cycle_duration_max": max(lengths, default=0),
        "scroll_cycle_segments": segments,
    }


def _cycle_reentry_count(segments: list[Mapping[str, Any]], window: int = 4) -> int:
    reentries = 0
    for previous, current in zip(segments, segments[1:]):
        previous_end = int(previous.get("cycle_end_step", 0) or 0)
        current_start = int(current.get("cycle_start_step", 0) or 0)
        if current_start > previous_end and current_start - previous_end <= window:
            reentries += 1
    return reentries


def _action_type_repeat_summary(
    action_distribution: Mapping[str, int],
    repeat_breakdown_by_type: Mapping[str, Counter[str]],
    reward_values_by_type: Mapping[str, list[float]],
) -> dict[str, dict[str, Any]]:
    action_types = (
        ("click_element", "click_element"),
        ("press_enter", "press_enter"),
        ("fill_input", "fill_input"),
        ("scroll_down", "scroll_down"),
        ("scroll_up", "scroll_up"),
        ("inspect_network", "inspect_network"),
        ("inspect_layout", "inspect_layout"),
        ("inspect_console", "inspect_console"),
        ("viewport_mobile", "change_viewport_mobile"),
        ("viewport_desktop", "change_viewport_desktop"),
    )
    summary: dict[str, dict[str, Any]] = {}
    for label, action_type in action_types:
        repeat_counts = repeat_breakdown_by_type.get(action_type, Counter())
        values = reward_values_by_type.get(action_type, [])
        summary[label] = {
            "action_count": int(action_distribution.get(action_type, 0) or 0),
            "repeat_count": int(
                repeat_counts.get("immediate_repeat", 0)
                + repeat_counts.get("recent_window_repeat", 0)
                + repeat_counts.get("cycle_repeat", 0)
            ),
            "recent_repeat_count": int(repeat_counts.get("recent_window_repeat", 0)),
            "cycle_participation_count": int(repeat_counts.get("cycle_repeat", 0)),
            "unproductive_repeat_count": int(repeat_counts.get("unproductive_repeat", 0)),
            "mean_reward": float(np.mean(values)) if values else 0.0,
        }
    return summary


def _history_counterfactual_repeat_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    relevant = [
        row
        for row in rows
        if int(row.get("active_cycle_length", 0) or 0) > 0
        or int(row.get("same_action_identity_streak", 0) or 0) > 1
        or bool(row.get("previous_unproductive_repeat"))
    ]
    help_count = sum(
        1
        for row in relevant
        if bool(row.get("zero_repeats_relevant_identity")) and not bool(row.get("real_repeats_relevant_identity"))
    )
    hurt_count = sum(
        1
        for row in relevant
        if bool(row.get("real_repeats_relevant_identity")) and not bool(row.get("zero_repeats_relevant_identity"))
    )
    real_types = Counter(str(row.get("selected_type_real") or "") for row in relevant)
    zero_types = Counter(str(row.get("selected_type_zero") or "") for row in relevant)
    cycle_rows = [row for row in relevant if int(row.get("active_cycle_length", 0) or 0) > 0]
    return {
        "repeat_state_subset_count": len(relevant),
        "repeat_state_subset_flip_rate": _ratio(
            sum(1 for row in relevant if bool(row.get("selected_type_flip"))),
            len(relevant),
        ),
        "repeat_state_real_dominant_type": real_types.most_common(1)[0][0] if real_types else "",
        "repeat_state_zero_dominant_type": zero_types.most_common(1)[0][0] if zero_types else "",
        "repeat_state_expected_cycle_breaking_real_count": sum(1 for row in cycle_rows if bool(row.get("real_breaks_cycle"))),
        "repeat_state_expected_cycle_breaking_zero_count": sum(1 for row in cycle_rows if bool(row.get("zero_breaks_cycle"))),
        "history_repeat_help_count": help_count,
        "history_repeat_hurt_count": hurt_count,
        "history_repeat_help_ratio": _ratio(help_count, len(relevant)),
        "history_repeat_hurt_ratio": _ratio(hurt_count, len(relevant)),
    }


def _prospective_validation_summary(counters: Mapping[str, Counter[str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, counter in counters.items():
        tp = int(counter.get("TP", 0))
        fp = int(counter.get("FP", 0))
        tn = int(counter.get("TN", 0))
        fn = int(counter.get("FN", 0))
        summary[name] = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
        }
    return summary


HISTV2_RISK_OFFSET = 635 + 72 + 212
HISTV2_RISK_DIM = 57


def _histv2_risk_ablation_state(state: np.ndarray, group: str) -> np.ndarray:
    result = np.array(state, dtype=np.float32, copy=True)
    if group == "prospective_risk_full":
        result[HISTV2_RISK_OFFSET : HISTV2_RISK_OFFSET + HISTV2_RISK_DIM] = 0.0
        return result
    flag_offset = {"immediate_risk": 0, "recent_risk": 1, "cycle_risk": 2}.get(group)
    if flag_offset is None:
        return result
    for action_type_index in range(19):
        result[HISTV2_RISK_OFFSET + action_type_index * 3 + flag_offset] = 0.0
    return result


def _histv2_flat_q(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, action_space) -> np.ndarray:
    with torch.inference_mode():
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)
        values = factorized_flat_q_values(network(state_tensor), mask_tensor, action_space).squeeze(0)
    return values.detach().cpu().numpy().astype(np.float32)


def _histv2_best_action_for_types(
    flat_q: np.ndarray,
    mask: np.ndarray,
    action_space,
    action_types: set[str],
) -> tuple[int | None, float]:
    best_id: int | None = None
    best_q = float("-inf")
    for action_type in action_types:
        if action_type in TARGET_REQUIRED_ACTION_TYPES:
            continue
        try:
            action_id = action_space.encode(action_type, 0)
        except ValueError:
            continue
        if action_id >= len(mask) or float(mask[action_id]) <= 0.0:
            continue
        value = float(flat_q[action_id])
        if best_id is None or value > best_q:
            best_id = int(action_id)
            best_q = value
    return best_id, best_q


def _histv2_cycle_q_diagnostic(
    network: FactorizedQNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    action_space,
    history_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    prospective = history_metadata.get("prospective_repeat_risk_by_action_type", {})
    if not isinstance(prospective, Mapping):
        return {}
    risky_types = {
        str(action_type)
        for action_type, flags in prospective.items()
        if isinstance(flags, Mapping) and bool(flags.get("cycle"))
    }
    if not risky_types:
        return {}
    risk_zero_types = {
        str(action_type)
        for action_type, flags in prospective.items()
        if isinstance(flags, Mapping) and not bool(flags.get("cycle"))
    }
    real_q = _histv2_flat_q(network, state, mask, action_space)
    zero_history = np.array(state, dtype=np.float32, copy=True)
    zero_history[635 + 72 : 635 + 72 + HISTORY_CONTEXT_V2_DIM] = 0.0
    zero_q = _histv2_flat_q(network, zero_history, mask, action_space)
    risky_action, q_real_risky = _histv2_best_action_for_types(real_q, mask, action_space, risky_types)
    break_action, q_real_break = _histv2_best_action_for_types(real_q, mask, action_space, risk_zero_types)
    if risky_action is None or break_action is None:
        return {}
    q_zero_risky = float(zero_q[risky_action])
    q_zero_break = float(zero_q[break_action])
    margin_real = float(q_real_risky - q_real_break)
    margin_zero = float(q_zero_risky - q_zero_break)
    row: dict[str, Any] = {
        "risky_action_type": str(action_space.decode(risky_action).get("action_type") or ""),
        "break_action_type": str(action_space.decode(break_action).get("action_type") or ""),
        "Q_real_risky": float(q_real_risky),
        "Q_real_break": float(q_real_break),
        "Q_zero_risky": q_zero_risky,
        "Q_zero_break": q_zero_break,
        "cycle_margin_real": margin_real,
        "cycle_margin_zero": margin_zero,
        "history_v2_cycle_bias": margin_real - margin_zero,
        "ablations": {},
    }
    baseline_selected = int(np.argmax(real_q))
    for group in ("prospective_risk_full", "immediate_risk", "recent_risk", "cycle_risk"):
        ablated = _histv2_risk_ablation_state(state, group)
        q_ab = _histv2_flat_q(network, ablated, mask, action_space)
        selected_ab = int(np.argmax(q_ab))
        margin_ab = float(q_ab[risky_action] - q_ab[break_action])
        selected_type = str(action_space.decode(selected_ab).get("action_type") or "")
        row["ablations"][group] = {
            "selected_type_flip": bool(selected_ab != baseline_selected),
            "selected_action_type": selected_type,
            "risky_action_selected": bool(selected_ab == risky_action),
            "cycle_margin": margin_ab,
            "cycle_margin_delta": margin_ab - margin_real,
            "Q_risky_delta": float(q_ab[risky_action] - q_real_risky),
            "Q_break_delta": float(q_ab[break_action] - q_real_break),
        }
    return row


def _histv2_cycle_q_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "cycle_margin_real_mean": 0.0,
            "cycle_margin_zero_mean": 0.0,
            "history_v2_cycle_bias_mean": 0.0,
            "rows": [],
            "ablation_summary": {},
        }
    summary = {
        "count": len(rows),
        "cycle_margin_real_mean": float(np.mean([float(row.get("cycle_margin_real", 0.0) or 0.0) for row in rows])),
        "cycle_margin_zero_mean": float(np.mean([float(row.get("cycle_margin_zero", 0.0) or 0.0) for row in rows])),
        "history_v2_cycle_bias_mean": float(np.mean([float(row.get("history_v2_cycle_bias", 0.0) or 0.0) for row in rows])),
        "rows": rows,
        "ablation_summary": {},
    }
    for group in ("prospective_risk_full", "immediate_risk", "recent_risk", "cycle_risk"):
        group_rows = [
            row.get("ablations", {}).get(group, {})
            for row in rows
            if isinstance(row.get("ablations", {}), Mapping)
        ]
        summary["ablation_summary"][group] = {
            "selected_type_flip_rate": _ratio(sum(1 for row in group_rows if row.get("selected_type_flip")), len(group_rows)),
            "risky_action_selection_rate": _ratio(sum(1 for row in group_rows if row.get("risky_action_selected")), len(group_rows)),
            "cycle_margin_delta_mean": float(np.mean([float(row.get("cycle_margin_delta", 0.0) or 0.0) for row in group_rows])) if group_rows else 0.0,
            "Q_risky_delta_mean": float(np.mean([float(row.get("Q_risky_delta", 0.0) or 0.0) for row in group_rows])) if group_rows else 0.0,
            "Q_break_delta_mean": float(np.mean([float(row.get("Q_break_delta", 0.0) or 0.0) for row in group_rows])) if group_rows else 0.0,
        }
    return summary


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
        top_counter[str(top.get("action_type") or "")] += 1
        if len(top_examples) < 10:
            top_examples.append(dict(top))
    summary["max_q_by_action_type"] = max_by_type
    summary["top_q_action_type_counts"] = dict(top_counter)
    summary["top_q_action_examples"] = top_examples
    return summary


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _json_stable_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-dir", default="artifacts/sanity")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--selection-mode", choices=["greedy", "stochastic"], default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--dqn-architecture",
        choices=[FLAT_DQN_ARCHITECTURE, FACTORIZED_DQN_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE, FACTORIZED_DQN_CONFOBS_HISTV2_ARCHITECTURE],
        default=FLAT_DQN_ARCHITECTURE,
    )
    parser.add_argument(
        "--dqn-greedy-selection-mode",
        choices=[FLAT_MAX_SELECTION, TYPE_LOGMEANEXP_SELECTION, FACTORIZED_SELECTION],
        default=FLAT_MAX_SELECTION,
    )
    args = parser.parse_args()
    result = run_sanity_check(args)
    if not args.quiet:
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
