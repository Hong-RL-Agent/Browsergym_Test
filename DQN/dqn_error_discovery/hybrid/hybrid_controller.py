"""Deterministic RAWD Hybrid v1 controller.

The controller owns only orchestration state. It does not select BrowserGym
actions, calculate rewards, or mutate PPO/DQN learning buffers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from dqn_error_discovery.agents.confirmation_reproduction import ACTION_REPLAY, OBSERVATION_RULES, derive_reproduction_context
from dqn_error_discovery.hybrid.strategy_router import StrategyRoute, route_candidate


class AgentMode(str, Enum):
    PPO_GLOBAL = "PPO_GLOBAL"
    DQN_LOCAL = "DQN_LOCAL"


class HandoffReason(str, Enum):
    NEW_CANDIDATE = "NEW_CANDIDATE"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    FUNCTIONAL_REGION = "FUNCTIONAL_REGION"
    STALLED_FUNCTIONAL = "STALLED_FUNCTIONAL"


class ReturnReason(str, Enum):
    CONFIRMED = "CONFIRMED"
    CONTEXT_CHANGED = "CONTEXT_CHANGED"
    REPEAT_GUARD = "REPEAT_GUARD"
    NOVELTY_EXHAUSTED = "NOVELTY_EXHAUSTED"
    LOCAL_BUDGET = "LOCAL_BUDGET"
    EPISODE_DONE = "EPISODE_DONE"


@dataclass
class PendingHandoffState:
    logical_key: str
    last_handoff_global_step: int = -1
    last_handoff_ppo_step: int = -1
    last_return_reason: str = ""
    last_session_productive: bool = False
    last_confirmation_progress: int = 0
    last_context_fingerprint: str = ""
    last_target_resolvable: bool = False
    last_reproduction_requirement_fingerprint: str = ""
    handoff_count: int = 0
    suppressed_count: int = 0
    resolved: bool = False


@dataclass
class LocalSession:
    local_session_id: int
    handoff_step: int
    handoff_reason: str
    entry_url: str
    entry_functional_region: str
    entry_candidate_keys: set[str] = field(default_factory=set)
    entry_pending_confirmation_keys: set[str] = field(default_factory=set)
    trigger_candidate_key: str = ""
    trigger_candidate_rule_id: str = ""
    trigger_candidate_confirmation_mode: str = ""
    entry_context_fingerprint: str = ""
    entry_target_resolvable: bool = False
    entry_confirmation_progress: int = 0
    entry_reproduction_requirement_fingerprint: str = ""
    local_steps: int = 0
    novelty_miss_streak: int = 0
    repeat_guard_streak: int = 0
    session_productive: bool = False
    session_confirmation_progress: bool = False
    trigger_candidate_confirmed: bool = False
    resolved_candidate_key: str = ""
    confirmation_step: int = 0
    functional_region_key: str = ""


@dataclass
class FunctionalRegionState:
    region_key: str
    first_seen_step: int
    ppo_visit_count: int = 0
    dqn_handoff_count: int = 0
    last_seen_step: int = -1
    last_handoff_step: int = -1
    last_semantic_state_fingerprint: str = ""
    last_target_set_signature: str = ""
    last_action_target_signature: str = ""
    last_evidence_signature: str = ""
    last_actionability: bool = False
    last_local_outcome: str = ""
    last_local_strong_productive: bool = False


class HybridController:
    """Rule-based PPO <-> DQN handoff/return state machine."""

    handoff_priority = (
        HandoffReason.NEW_CANDIDATE,
        HandoffReason.PENDING_CONFIRMATION,
        HandoffReason.FUNCTIONAL_REGION,
        HandoffReason.STALLED_FUNCTIONAL,
    )
    return_priority = (
        ReturnReason.EPISODE_DONE,
        ReturnReason.CONFIRMED,
        ReturnReason.CONTEXT_CHANGED,
        ReturnReason.REPEAT_GUARD,
        ReturnReason.NOVELTY_EXHAUSTED,
        ReturnReason.LOCAL_BUDGET,
    )

    def __init__(
        self,
        *,
        max_local_steps: int = 5,
        cooldown_ppo_steps: int = 1,
        novelty_patience: int = 2,
        repeat_guard_patience: int = 2,
        pending_rehandoff_min_ppo_steps: int = 3,
        functional_region_novelty_gate_enabled: bool = False,
    ) -> None:
        if max_local_steps <= 0:
            raise ValueError("max_local_steps must be positive")
        self.max_local_steps = int(max_local_steps)
        self.cooldown_ppo_steps = int(max(0, cooldown_ppo_steps))
        self.novelty_patience = int(max(1, novelty_patience))
        self.repeat_guard_patience = int(max(1, repeat_guard_patience))
        self.pending_rehandoff_min_ppo_steps = int(max(0, pending_rehandoff_min_ppo_steps))
        self.functional_region_novelty_gate_enabled = bool(functional_region_novelty_gate_enabled)
        self.active_agent = AgentMode.PPO_GLOBAL
        self.local_session: LocalSession | None = None
        self.local_session_count = 0
        self.ppo_action_count = 0
        self.ppo_steps_since_return = self.cooldown_ppo_steps
        self.last_switch_step = -1
        self.pending_handoff_state: dict[str, PendingHandoffState] = {}
        self.suppression_events: list[dict[str, Any]] = []
        self.handoff_events: list[dict[str, Any]] = []
        self.return_events: list[dict[str, Any]] = []
        self.unrelated_confirmation_events: list[dict[str, Any]] = []
        self.handoff_reason_counts: Counter[str] = Counter()
        self.return_reason_counts: Counter[str] = Counter()
        self.suppression_reason_counts: Counter[str] = Counter()
        self.agent_step_counts: Counter[str] = Counter()
        self.local_session_lengths: list[int] = []
        self.guard_metric_counts: Counter[str] = Counter()
        self.functional_region_history: dict[str, FunctionalRegionState] = {}
        self.functional_region_opportunities: list[dict[str, Any]] = []

    def record_agent_step(self, agent: AgentMode | str) -> None:
        mode = AgentMode(agent) if not isinstance(agent, AgentMode) else agent
        self.agent_step_counts[mode.value] += 1
        if mode == AgentMode.PPO_GLOBAL:
            self.ppo_action_count += 1
            self.ppo_steps_since_return += 1

    def verification_route(self, candidate: Mapping[str, Any]) -> StrategyRoute:
        """Expose the canonical owner split to the production controller.

        The existing PPO/DQN state machine remains responsible for lifecycle
        orchestration; this method prevents callers from inferring DQN
        eligibility from a rule id or from a mere candidate presence.
        """
        return route_candidate(candidate)

    def dqn_local_eligible(self, candidate: Mapping[str, Any]) -> bool:
        return self.verification_route(candidate).dqn_eligible

    def maybe_handoff_after_ppo_step(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
        reward_components: Mapping[str, Any],
        confirmation_metadata: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> str:
        if self.active_agent != AgentMode.PPO_GLOBAL:
            return ""
        if step == self.last_switch_step:
            return ""
        if self.ppo_steps_since_return < self.cooldown_ppo_steps:
            return ""
        pending_decision = self._eligible_pending_candidate(
            step=step,
            observation=observation,
            confirmation_metadata=confirmation_metadata,
            history=history,
        )
        if pending_decision is None:
            self._record_stale_session_prevented(
                step=step,
                observation=observation,
                confirmation_metadata=confirmation_metadata,
                history=history,
            )
        functional_region = _functional_region(observation)
        functional_novelty = self._functional_region_novelty(
            step=step,
            observation=observation,
            confirmation_metadata=confirmation_metadata,
        ) if functional_region else _empty_functional_novelty()
        functional_trigger = bool(functional_region) and not self._same_context_pending_bypass_blocked(
            step=step,
            observation=observation,
            confirmation_metadata=confirmation_metadata,
            history=history,
            trigger="FUNCTIONAL_REGION",
        )
        if self.functional_region_novelty_gate_enabled and functional_trigger:
            functional_trigger = bool(functional_novelty["novelty_eligible"])
        stalled_trigger = _stalled_functional(reward_components, observation) and not self._same_context_pending_bypass_blocked(
            step=step,
            observation=observation,
            confirmation_metadata=confirmation_metadata,
            history=history,
            trigger="STALLED_FUNCTIONAL",
        )
        triggers = {
            HandoffReason.NEW_CANDIDATE: _has_new_candidate(anomalies, history),
            HandoffReason.PENDING_CONFIRMATION: pending_decision is not None,
            HandoffReason.FUNCTIONAL_REGION: functional_trigger,
            HandoffReason.STALLED_FUNCTIONAL: stalled_trigger,
        }
        reason = next((item for item in self.handoff_priority if triggers.get(item)), None)
        if reason is None:
            if functional_region:
                self._finalize_functional_region_opportunity(functional_novelty, handoff_performed=False, handoff_reason="")
            return ""
        self.local_session_count += 1
        trigger_candidate_key = ""
        if reason == HandoffReason.PENDING_CONFIRMATION and pending_decision:
            trigger_candidate_key = pending_decision["logical_key"]
        elif reason == HandoffReason.NEW_CANDIDATE:
            trigger_candidate_key = _new_candidate_key(anomalies, history)
        context_fingerprint = _context_fingerprint(observation, confirmation_metadata)
        target_resolvable = _target_resolvable(confirmation_metadata)
        confirmation_progress = _candidate_progress_value(history, confirmation_metadata, trigger_candidate_key)
        requirement_fingerprint = _reproduction_requirement_fingerprint(history, confirmation_metadata, trigger_candidate_key)
        session = LocalSession(
            local_session_id=self.local_session_count,
            handoff_step=step,
            handoff_reason=reason.value,
            entry_url=_url(observation),
            entry_functional_region=_functional_region(observation),
            entry_candidate_keys=set(_candidate_keys(anomalies)),
            entry_pending_confirmation_keys=set(_pending_keys(history)),
            trigger_candidate_key=trigger_candidate_key,
            trigger_candidate_rule_id=_rule_id_for_key(trigger_candidate_key, confirmation_metadata),
            trigger_candidate_confirmation_mode=str(confirmation_metadata.get("confirmation_mode") or ""),
            entry_context_fingerprint=context_fingerprint,
            entry_target_resolvable=target_resolvable,
            entry_confirmation_progress=confirmation_progress,
            entry_reproduction_requirement_fingerprint=requirement_fingerprint,
            functional_region_key=str(functional_novelty.get("region_key") or ""),
        )
        if trigger_candidate_key:
            state = self.pending_handoff_state.setdefault(trigger_candidate_key, PendingHandoffState(logical_key=trigger_candidate_key))
            state.last_handoff_global_step = step
            state.last_handoff_ppo_step = self.ppo_action_count
            state.last_context_fingerprint = context_fingerprint
            state.last_target_resolvable = target_resolvable
            state.last_confirmation_progress = confirmation_progress
            state.last_reproduction_requirement_fingerprint = requirement_fingerprint
            state.handoff_count += 1
            self.guard_metric_counts["pending_rehandoff_eligible_count"] += 1
            self.guard_metric_counts["pending_handoff_candidate_count"] += 1
            eligibility = pending_decision.get("reason", "FIRST_HANDOFF") if pending_decision else "FIRST_HANDOFF"
            self.guard_metric_counts[f"eligibility_{eligibility}"] += 1
        self.local_session = session
        self.active_agent = AgentMode.DQN_LOCAL
        self.last_switch_step = step
        event = {
            "step": step,
            "local_session_id": session.local_session_id,
            "reason": reason.value,
            "entry_url": session.entry_url,
            "entry_functional_region": session.entry_functional_region,
            "entry_candidate_keys": sorted(session.entry_candidate_keys),
            "entry_anomaly_keys": sorted(session.entry_candidate_keys),
            "entry_pending_confirmation_keys": sorted(session.entry_pending_confirmation_keys),
            "entry_tracker_candidate_keys": sorted(session.entry_pending_confirmation_keys),
            "trigger_candidate_key": session.trigger_candidate_key,
            "trigger_candidate_rule_id": session.trigger_candidate_rule_id,
            "trigger_candidate_confirmation_mode": session.trigger_candidate_confirmation_mode,
            "entry_context_fingerprint": session.entry_context_fingerprint,
            "entry_target_resolvable": session.entry_target_resolvable,
            "entry_confirmation_progress": session.entry_confirmation_progress,
            "entry_reproduction_requirement_fingerprint": session.entry_reproduction_requirement_fingerprint,
            "ppo_action_count": self.ppo_action_count,
            "eligibility_reason": pending_decision.get("reason", "") if pending_decision else "",
            "eligibility_reasons": pending_decision.get("eligibility_reasons", []) if pending_decision else [],
        }
        if functional_region:
            self._finalize_functional_region_opportunity(
                functional_novelty,
                handoff_performed=reason == HandoffReason.FUNCTIONAL_REGION,
                handoff_reason=reason.value,
            )
            event["functional_region_novelty"] = dict(functional_novelty)
        self.handoff_events.append(event)
        self.handoff_reason_counts[reason.value] += 1
        return reason.value

    def maybe_return_after_dqn_step(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
        reward_components: Mapping[str, Any],
        repeat_diagnostic: Mapping[str, Any],
        done: bool,
        confirmation_events: Sequence[Mapping[str, Any]] | None = None,
        tracker_history: Mapping[str, Any] | None = None,
        confirmation_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if self.active_agent != AgentMode.DQN_LOCAL or self.local_session is None:
            return ""
        session = self.local_session
        session.local_steps += 1
        if _local_novelty(reward_components):
            session.novelty_miss_streak = 0
            session.session_productive = True
        else:
            session.novelty_miss_streak += 1
        if _confirmation_progress(reward_components):
            session.session_confirmation_progress = True
            session.session_productive = True
        resolved = _resolved_trigger_candidate(
            session,
            anomalies,
            confirmation_events or (),
            tracker_history or {},
            confirmation_metadata or {},
        )
        if resolved:
            session.trigger_candidate_confirmed = True
            if not session.resolved_candidate_key:
                session.resolved_candidate_key = resolved
            if not session.confirmation_step:
                session.confirmation_step = session.local_steps
        unrelated_confirmed = _current_unrelated_confirmed_keys(
            session,
            anomalies,
            confirmation_events or (),
        )
        if unrelated_confirmed:
            self.guard_metric_counts["unrelated_confirmation_observed"] += len(unrelated_confirmed)
            self.unrelated_confirmation_events.append(
                {
                    "step": step,
                    "local_session_id": session.local_session_id,
                    "trigger_candidate_key": session.trigger_candidate_key,
                    "unrelated_confirmed_keys": sorted(unrelated_confirmed),
                    "caused_return": False,
                }
            )
        if bool(repeat_diagnostic.get("unproductive_repeat")) and bool(repeat_diagnostic.get("cycle_repeat")):
            session.repeat_guard_streak += 1
        else:
            session.repeat_guard_streak = 0
        checks = {
            ReturnReason.EPISODE_DONE: bool(done),
            ReturnReason.CONFIRMED: bool(
                resolved
                or (
                    not session.trigger_candidate_key
                    and _resolved_entry_candidate(session, anomalies)
                )
            ),
            ReturnReason.CONTEXT_CHANGED: _context_changed(session, observation),
            ReturnReason.REPEAT_GUARD: session.repeat_guard_streak >= self.repeat_guard_patience,
            ReturnReason.NOVELTY_EXHAUSTED: session.novelty_miss_streak >= self.novelty_patience,
            ReturnReason.LOCAL_BUDGET: session.local_steps >= self.max_local_steps,
        }
        reason = next((item for item in self.return_priority if checks.get(item)), None)
        if reason is None:
            return ""
        exit_metadata = confirmation_metadata or {}
        exit_history = tracker_history or {}
        exit_context = _context_fingerprint(observation, exit_metadata) if exit_metadata else session.entry_context_fingerprint
        exit_progress = (
            _candidate_progress_value(exit_history, exit_metadata, session.trigger_candidate_key)
            if exit_history or exit_metadata
            else session.entry_confirmation_progress
        )
        exit_resolvable = _target_resolvable(exit_metadata) if exit_metadata else session.entry_target_resolvable
        exit_requirement = (
            _reproduction_requirement_fingerprint(exit_history, exit_metadata, session.trigger_candidate_key)
            if exit_history or exit_metadata
            else session.entry_reproduction_requirement_fingerprint
        )
        self.active_agent = AgentMode.PPO_GLOBAL
        self.ppo_steps_since_return = 0
        self.last_switch_step = step
        self.local_session_lengths.append(session.local_steps)
        event = {
            "step": step,
            "local_session_id": session.local_session_id,
            "reason": reason.value,
            "local_steps": session.local_steps,
            "url": _url(observation),
            "functional_region": _functional_region(observation),
            "trigger_candidate_key": session.trigger_candidate_key,
            "session_productive": session.session_productive,
            "session_confirmation_progress": session.session_confirmation_progress,
            "context_fingerprint": exit_context,
            "ppo_action_count": self.ppo_action_count,
            "resolved_candidate_key": session.resolved_candidate_key,
            "trigger_candidate_confirmed": session.trigger_candidate_confirmed,
            "confirmation_step": session.confirmation_step,
            "confirmation_mode": session.trigger_candidate_confirmation_mode,
            "local_steps_before_confirmation": session.confirmation_step,
            "exit_confirmation_progress": exit_progress,
            "exit_target_resolvable": exit_resolvable,
            "exit_reproduction_requirement_fingerprint": exit_requirement,
        }
        self.return_events.append(event)
        self.return_reason_counts[reason.value] += 1
        if session.trigger_candidate_key:
            state = self.pending_handoff_state.setdefault(session.trigger_candidate_key, PendingHandoffState(logical_key=session.trigger_candidate_key))
            state.last_return_reason = reason.value
            state.last_session_productive = bool(session.session_productive)
            state.last_context_fingerprint = exit_context
            state.last_target_resolvable = exit_resolvable
            state.last_reproduction_requirement_fingerprint = exit_requirement
            state.last_confirmation_progress = max(state.last_confirmation_progress, exit_progress)
            if reason == ReturnReason.CONFIRMED:
                state.resolved = True
        if session.functional_region_key:
            region_state = self.functional_region_history.get(session.functional_region_key)
            if region_state is not None:
                region_state.last_local_outcome = reason.value
                region_state.last_local_strong_productive = bool(session.session_productive)
        self.local_session = None
        return reason.value

    def reset_episode_guard(self) -> None:
        self.pending_handoff_state.clear()
        self.functional_region_history.clear()

    def summary(self) -> dict[str, Any]:
        total_sessions = len(self.local_session_lengths)
        return {
            "active_agent": self.active_agent.value,
            "handoff_count": len(self.handoff_events),
            "return_count": len(self.return_events),
            "dqn_session_count": self.local_session_count,
            "mean_dqn_session_length": sum(self.local_session_lengths) / total_sessions if total_sessions else 0.0,
            "max_dqn_session_length": max(self.local_session_lengths, default=0),
            "handoff_reasons": dict(self.handoff_reason_counts),
            "return_reasons": dict(self.return_reason_counts),
            "pending_handoff_guard": {
                "pending_rehandoff_min_ppo_steps": self.pending_rehandoff_min_ppo_steps,
                "pending_handoff_candidate_count": int(self.guard_metric_counts.get("pending_handoff_candidate_count", 0)),
                "pending_handoff_suppressed_count": len(self.suppression_events),
                "pending_rehandoff_eligible_count": int(self.guard_metric_counts.get("pending_rehandoff_eligible_count", 0)),
                "pending_rehandoff_blocked_count": int(self.guard_metric_counts.get("pending_rehandoff_blocked_count", 0)),
                "same_candidate_rehandoff_blocked_count": int(self.guard_metric_counts.get("same_candidate_rehandoff_blocked_count", 0)),
                "same_context_functional_bypass_blocked_count": int(self.guard_metric_counts.get("same_context_functional_bypass_blocked_count", 0)),
                "same_context_stalled_bypass_blocked_count": int(self.guard_metric_counts.get("same_context_stalled_bypass_blocked_count", 0)),
                "rehandoff_due_to_progress": int(self.guard_metric_counts.get("rehandoff_due_to_progress", 0)),
                "rehandoff_due_to_context_change": int(self.guard_metric_counts.get("rehandoff_due_to_context_change", 0)),
                "rehandoff_due_to_target_resolvable": int(self.guard_metric_counts.get("rehandoff_due_to_target_resolvable", 0)),
                "rehandoff_due_to_reproduction_requirement": int(self.guard_metric_counts.get("rehandoff_due_to_reproduction_requirement", 0)),
                "stale_pending_evaluations": int(self.guard_metric_counts.get("stale_pending_evaluations", 0)),
                "stale_pending_handoff_blocked": int(self.guard_metric_counts.get("stale_pending_handoff_blocked", 0)),
                "stale_session_rehandoff_prevented": int(self.guard_metric_counts.get("stale_session_rehandoff_prevented", 0)),
                "eligibility_reasons": {
                    key.replace("eligibility_", ""): int(value)
                    for key, value in sorted(self.guard_metric_counts.items())
                    if str(key).startswith("eligibility_")
                },
                "suppression_reasons": dict(self.suppression_reason_counts),
                "suppression_events": list(self.suppression_events),
                "candidate_states": {key: dict(vars(value)) for key, value in sorted(self.pending_handoff_state.items())},
            },
            "functional_region_novelty": {
                "enabled": self.functional_region_novelty_gate_enabled,
                "opportunity_count": len(self.functional_region_opportunities),
                "suppressed_count": sum(not bool(item.get("handoff_performed")) for item in self.functional_region_opportunities),
                "handoff_count": sum(bool(item.get("handoff_performed")) for item in self.functional_region_opportunities),
                "suppression_reasons": dict(Counter(
                    reason
                    for item in self.functional_region_opportunities
                    for reason in item.get("suppression_reasons", [])
                )),
                "opportunities": list(self.functional_region_opportunities),
                "region_history": {key: dict(vars(value)) for key, value in sorted(self.functional_region_history.items())},
            },
            "agent_step_counts": dict(self.agent_step_counts),
            "handoff_events": list(self.handoff_events),
            "return_events": list(self.return_events),
            "zero_step_switch_loop": bool(
                {int(event["step"]) for event in self.handoff_events}.intersection(
                    int(event["step"]) for event in self.return_events
                )
            ),
        }

    def _functional_region_novelty(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        confirmation_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        region_key = _functional_region_key(observation, confirmation_metadata)
        state_fingerprint = _semantic_region_state_fingerprint(observation, confirmation_metadata)
        target_signature = _target_set_signature(observation)
        action_target_signature = _action_target_context_signature(confirmation_metadata)
        evidence_signature = _relevant_evidence_signature(confirmation_metadata)
        actionability = _target_resolvable(confirmation_metadata)
        prior = self.functional_region_history.get(region_key)
        first_time = prior is None
        semantic_changed = bool(prior and prior.last_semantic_state_fingerprint and prior.last_semantic_state_fingerprint != state_fingerprint)
        target_changed = bool(prior and prior.last_target_set_signature and prior.last_target_set_signature != target_signature)
        action_target_changed = bool(prior and prior.last_action_target_signature and prior.last_action_target_signature != action_target_signature)
        evidence_changed = bool(prior and evidence_signature and prior.last_evidence_signature != evidence_signature)
        actionability_changed = bool(prior and prior.last_actionability != actionability)
        novelty_eligible = first_time or semantic_changed or target_changed or action_target_changed or evidence_changed or actionability_changed
        suppression_reasons: list[str] = []
        if not novelty_eligible:
            suppression_reasons.extend(("NO_REGION_NOVELTY", "NO_NEW_ACTION_TARGET", "NO_NEW_EVIDENCE"))
            if prior and prior.ppo_visit_count > 0:
                suppression_reasons.append("PPO_SATURATED_NO_CHANGE")
            if prior and prior.last_local_outcome and not prior.last_local_strong_productive:
                suppression_reasons.append("PREVIOUS_UNPRODUCTIVE_NO_CHANGE")
        if prior is None:
            prior = FunctionalRegionState(region_key=region_key, first_seen_step=step)
            self.functional_region_history[region_key] = prior
        prior.ppo_visit_count += 1
        prior.last_seen_step = step
        prior.last_semantic_state_fingerprint = state_fingerprint
        prior.last_target_set_signature = target_signature
        prior.last_action_target_signature = action_target_signature
        if evidence_signature:
            prior.last_evidence_signature = evidence_signature
        prior.last_actionability = actionability
        return {
            "functional_region_detected": True,
            "region_key": region_key,
            "first_time_region": first_time,
            "semantic_state_changed": semantic_changed,
            "new_target_context": target_changed,
            "new_action_target_context": action_target_changed,
            "new_evidence_context": evidence_changed,
            "actionability_changed": actionability_changed,
            "previous_local_outcome": prior.last_local_outcome,
            "previous_local_strong_productive": prior.last_local_strong_productive,
            "ppo_visit_count": prior.ppo_visit_count,
            "novelty_eligible": novelty_eligible,
            "suppression_reasons": suppression_reasons,
            "handoff_performed": False,
            "handoff_reason": "",
        }

    def _finalize_functional_region_opportunity(
        self,
        novelty: dict[str, Any],
        *,
        handoff_performed: bool,
        handoff_reason: str,
    ) -> None:
        novelty["handoff_performed"] = handoff_performed
        novelty["handoff_reason"] = handoff_reason
        state = self.functional_region_history.get(str(novelty.get("region_key") or ""))
        if state is not None and handoff_performed and handoff_reason == HandoffReason.FUNCTIONAL_REGION.value:
            state.dqn_handoff_count += 1
            state.last_handoff_step = int(self.last_switch_step if self.last_switch_step >= 0 else state.last_seen_step)
        if not handoff_performed and novelty.get("suppression_reasons"):
            self.suppression_reason_counts["FUNCTIONAL_REGION_NO_NOVELTY"] += 1
        self.functional_region_opportunities.append(dict(novelty))

    def _eligible_pending_candidate(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        confirmation_metadata: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not _pending_confirmation_resolvable(confirmation_metadata):
            return None
        context = _context_fingerprint(observation, confirmation_metadata)
        target_resolvable = _target_resolvable(confirmation_metadata)
        for key in _candidate_priority_keys(history, confirmation_metadata):
            state = self.pending_handoff_state.get(key)
            if state is not None and state.resolved:
                continue
            if state is None:
                return {"logical_key": key, "reason": "FIRST_HANDOFF"}
            progress = _candidate_progress_value(history, confirmation_metadata, key)
            requirement = _reproduction_requirement_fingerprint(history, confirmation_metadata, key)
            reasons: list[str] = []
            if progress > state.last_confirmation_progress:
                reasons.append("PROGRESS_CHANGED")
            if target_resolvable and not state.last_target_resolvable:
                reasons.append("TARGET_BECAME_RESOLVABLE")
            if context != state.last_context_fingerprint:
                reasons.append("CONTEXT_CHANGED")
            if requirement != state.last_reproduction_requirement_fingerprint:
                reasons.append("REPRODUCTION_REQUIREMENT_CHANGED")
            if reasons:
                if "PROGRESS_CHANGED" in reasons:
                    self.guard_metric_counts["rehandoff_due_to_progress"] += 1
                if "CONTEXT_CHANGED" in reasons:
                    self.guard_metric_counts["rehandoff_due_to_context_change"] += 1
                if "TARGET_BECAME_RESOLVABLE" in reasons:
                    self.guard_metric_counts["rehandoff_due_to_target_resolvable"] += 1
                if "REPRODUCTION_REQUIREMENT_CHANGED" in reasons:
                    self.guard_metric_counts["rehandoff_due_to_reproduction_requirement"] += 1
                return {"logical_key": key, "reason": reasons[0], "eligibility_reasons": reasons}
            ppo_delta = self.ppo_action_count - state.last_handoff_ppo_step
            self._record_pending_suppression(
                step=step,
                logical_key=key,
                context_fingerprint=context,
                target_resolvable=target_resolvable,
                ppo_delta=ppo_delta,
                reasons=_suppression_reasons(state, context, target_resolvable, ppo_delta, self.pending_rehandoff_min_ppo_steps),
            )
        return None

    def _same_context_pending_bypass_blocked(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        confirmation_metadata: Mapping[str, Any],
        history: Mapping[str, Any],
        trigger: str,
    ) -> bool:
        if not _pending_confirmation_resolvable(confirmation_metadata):
            return False
        context = _context_fingerprint(observation, confirmation_metadata)
        target_resolvable = _target_resolvable(confirmation_metadata)
        for key in _candidate_priority_keys(history, confirmation_metadata):
            state = self.pending_handoff_state.get(key)
            if state is None:
                continue
            if state.resolved:
                continue
            ppo_delta = self.ppo_action_count - state.last_handoff_ppo_step
            progress = _candidate_progress_value(history, confirmation_metadata, key)
            requirement = _reproduction_requirement_fingerprint(history, confirmation_metadata, key)
            stale = (
                context == state.last_context_fingerprint
                and target_resolvable == state.last_target_resolvable
                and progress <= state.last_confirmation_progress
                and requirement == state.last_reproduction_requirement_fingerprint
            )
            if stale:
                metric = "same_context_functional_bypass_blocked_count" if trigger == "FUNCTIONAL_REGION" else "same_context_stalled_bypass_blocked_count"
                self.guard_metric_counts[metric] += 1
                self._record_pending_suppression(
                    step=step,
                    logical_key=key,
                    context_fingerprint=context,
                    target_resolvable=target_resolvable,
                    ppo_delta=ppo_delta,
                    reasons=["STALE_PENDING", "SAME_CONTEXT", "FUNCTIONAL_BYPASS_BLOCKED" if trigger == "FUNCTIONAL_REGION" else "STALLED_BYPASS_BLOCKED"],
                    bypass_trigger=trigger,
                )
                return True
        return False

    def _record_stale_session_prevented(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        confirmation_metadata: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> None:
        if not _pending_confirmation_resolvable(confirmation_metadata):
            return
        context = _context_fingerprint(observation, confirmation_metadata)
        target_resolvable = _target_resolvable(confirmation_metadata)
        for key in _candidate_priority_keys(history, confirmation_metadata):
            state = self.pending_handoff_state.get(key)
            if state is None or state.resolved:
                continue
            progress = _candidate_progress_value(history, confirmation_metadata, key)
            requirement = _reproduction_requirement_fingerprint(history, confirmation_metadata, key)
            if (
                context == state.last_context_fingerprint
                and target_resolvable == state.last_target_resolvable
                and progress <= state.last_confirmation_progress
                and requirement == state.last_reproduction_requirement_fingerprint
            ):
                self.guard_metric_counts["stale_session_rehandoff_prevented"] += 1
                return

    def _record_pending_suppression(
        self,
        *,
        step: int,
        logical_key: str,
        context_fingerprint: str,
        target_resolvable: bool,
        ppo_delta: int,
        reasons: list[str],
        bypass_trigger: str = "",
    ) -> None:
        state = self.pending_handoff_state.setdefault(logical_key, PendingHandoffState(logical_key=logical_key))
        state.suppressed_count += 1
        self.guard_metric_counts["pending_rehandoff_blocked_count"] += 1
        self.guard_metric_counts["same_candidate_rehandoff_blocked_count"] += 1
        self.guard_metric_counts["stale_pending_evaluations"] += 1
        self.guard_metric_counts["stale_pending_handoff_blocked"] += 1
        for reason in reasons:
            self.suppression_reason_counts[reason] += 1
        self.suppression_events.append(
            {
                "event": "PENDING_REHANDOFF_SUPPRESSED",
                "step": step,
                "logical_key": logical_key,
                "ppo_steps_since_candidate_handoff": ppo_delta,
                "context_fingerprint": context_fingerprint,
                "target_resolvable": target_resolvable,
                "reasons": list(reasons),
                "bypass_trigger": bypass_trigger,
            }
        )


def _candidate_keys(anomalies: Sequence[Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    for anomaly in anomalies or []:
        key = str(anomaly.get("logical_key") or anomaly.get("signature") or "")
        if key and str(anomaly.get("status") or "candidate") == "candidate":
            keys.append(key)
    return keys


def _new_candidate_key(anomalies: Sequence[Mapping[str, Any]], history: Mapping[str, Any]) -> str:
    seen = set(history.get("seen_candidate_signatures", set()) or set())
    rewarded = set(history.get("seen_reward_candidate_keys", set()) or set())
    eligible: list[Mapping[str, Any]] = []
    for anomaly in anomalies or []:
        key = _canonical_candidate_key(anomaly)
        if key and str(anomaly.get("status") or "candidate") == "candidate" and key not in seen and key not in rewarded and _eligible_for_action_replay_session(anomaly):
            eligible.append(anomaly)
    for anomaly in eligible:
        if bool(anomaly.get("pending_confirmation")):
            return _canonical_candidate_key(anomaly)
    for anomaly in eligible:
        if _candidate_requires_independent_confirmation(anomaly):
            return _canonical_candidate_key(anomaly)
    if eligible:
        return _canonical_candidate_key(eligible[0])
    return ""


def _has_new_candidate(anomalies: Sequence[Mapping[str, Any]], history: Mapping[str, Any]) -> bool:
    seen = set(history.get("seen_candidate_signatures", set()) or set())
    rewarded = set(history.get("seen_reward_candidate_keys", set()) or set())
    for anomaly in anomalies or []:
        if not _eligible_for_action_replay_session(anomaly):
            continue
        key = _canonical_candidate_key(anomaly)
        if key not in seen and key not in rewarded:
            return True
    return False


def _eligible_for_action_replay_session(anomaly: Mapping[str, Any]) -> bool:
    strategy = str(anomaly.get("reproduction_strategy") or "").upper()
    if strategy:
        return strategy == ACTION_REPLAY
    rule_id = str(anomaly.get("rule_id") or (anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}).get("rule_id") or "")
    return rule_id not in OBSERVATION_RULES


def _candidate_requires_independent_confirmation(anomaly: Mapping[str, Any]) -> bool:
    return bool(anomaly.get("pending_confirmation")) or str(anomaly.get("confirmation_policy") or "").lower() in {
        "reproduce",
        "independent_reproduction",
    }


def _canonical_candidate_key(item: Mapping[str, Any]) -> str:
    return str(item.get("logical_key") or item.get("signature") or "")


def _confirmed_event_keys(events: Sequence[Mapping[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for event in events or ():
        if not isinstance(event, Mapping):
            continue
        key = _canonical_candidate_key(event)
        if key and str(event.get("current_status") or "confirmed") == "confirmed":
            keys.add(key)
    return keys


def _tracker_confirmed_keys(history: Mapping[str, Any]) -> set[str]:
    keys = history.get("rewarded_confirmed_logical_keys", set()) or set()
    seen = history.get("seen_confirmed_signatures", set()) or set()
    return {str(key) for key in set(keys).union(set(seen)) if str(key)}


def _tracker_pending_keys(history: Mapping[str, Any]) -> set[str]:
    return set(_pending_keys(history))


def _rule_id_for_key(key: str, metadata: Mapping[str, Any]) -> str:
    if key.startswith("R") and "|" in key:
        return key.split("|", 1)[0]
    return str(metadata.get("active_rule_id") or "")


def _pending_keys(history: Mapping[str, Any]) -> list[str]:
    records = history.get("pending_reproduction_records", {})
    if not isinstance(records, Mapping):
        return []
    return [str(row.get("logical_key") or key) for key, row in records.items() if isinstance(row, Mapping)]


def _pending_record(history: Mapping[str, Any], logical_key: str) -> Mapping[str, Any]:
    records = history.get("pending_reproduction_records", {})
    if not isinstance(records, Mapping):
        return {}
    for key, row in records.items():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("logical_key") or key) == logical_key:
            return row
    return {}


def _pending_confirmation_resolvable(metadata: Mapping[str, Any]) -> bool:
    if not bool(metadata.get("pending_confirmation_exists")):
        return False
    mode = str(metadata.get("confirmation_mode") or "")
    if mode == "ACTION_REPRODUCTION":
        return bool(metadata.get("expected_action_type")) and (
            not bool(metadata.get("expected_action_requires_target"))
            or bool(metadata.get("expected_target_visible"))
        )
    if mode == "OBSERVATION_REPRODUCTION":
        return False
    return False


def _target_resolvable(metadata: Mapping[str, Any]) -> bool:
    mode = str(metadata.get("confirmation_mode") or "")
    if mode == "ACTION_REPRODUCTION":
        return bool(metadata.get("expected_action_type")) and (
            not bool(metadata.get("expected_action_requires_target"))
            or bool(metadata.get("expected_target_visible"))
        )
    if mode == "OBSERVATION_REPRODUCTION":
        return True
    return bool(metadata.get("pending_confirmation_exists"))


def _candidate_priority_keys(history: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    active = str(metadata.get("active_pending_key") or "")
    if active:
        keys.append(active)
    for key in _pending_keys(history):
        if key not in keys:
            keys.append(key)
    if not keys and bool(metadata.get("pending_confirmation_exists")):
        fallback = _metadata_pending_key(metadata)
        if fallback:
            keys.append(fallback)
    return keys


def _metadata_pending_key(metadata: Mapping[str, Any]) -> str:
    active = str(metadata.get("active_pending_key") or "")
    if active:
        return active
    mode = str(metadata.get("confirmation_mode") or "")
    rule = str(metadata.get("active_rule_id") or "")
    target = str(metadata.get("expected_target_identity") or "")
    action = str(metadata.get("expected_action_type") or "")
    if rule or target or action:
        return "|".join(part for part in (rule, mode, action, target) if part)
    return "PENDING_CONFIRMATION"


def _candidate_progress_value(history: Mapping[str, Any], metadata: Mapping[str, Any], logical_key: str) -> int:
    values: list[int] = []
    record = _pending_record(history, logical_key)
    if record:
        values.append(int(round(float(record.get("progress", 0.0) or 0.0) * 1000)))
    if logical_key and logical_key == str(metadata.get("active_pending_key") or ""):
        values.append(int(round(float(metadata.get("confirmation_progress", 0.0) or 0.0) * 1000)))
    return max(values, default=0)


def _reproduction_requirement_fingerprint(history: Mapping[str, Any], metadata: Mapping[str, Any], logical_key: str) -> str:
    record = _pending_record(history, logical_key)
    if record:
        fields = [
            str(record.get("confirmation_mode") or ""),
            str(record.get("rule_id") or ""),
            str(record.get("expected_action_type") or ""),
            str(record.get("expected_target_identity") or ""),
            ",".join(str(item) for item in record.get("context_requirements", []) or []),
            str(record.get("action_target") or ""),
        ]
        return "|".join(fields)
    fields = [
        str(metadata.get("confirmation_mode") or ""),
        str(metadata.get("active_rule_id") or ""),
        str(metadata.get("expected_action_type") or ""),
        str(metadata.get("expected_target_identity") or ""),
        ",".join(str(item) for item in metadata.get("context_requirements", []) or []),
    ]
    return "|".join(fields)


def _confirmation_progress_value(history: Mapping[str, Any], metadata: Mapping[str, Any]) -> int:
    progress = float(metadata.get("confirmation_progress", 0.0) or 0.0)
    attempts = float(metadata.get("reproduction_attempt_count", 0.0) or 0.0)
    last_progress = int(bool(history.get("last_confirmation_progress")))
    return int(round(progress * 1000)) + int(round(attempts * 100)) + last_progress


def _confirmation_progress(reward_components: Mapping[str, Any]) -> bool:
    return any(float(reward_components.get(key, 0.0) or 0.0) > 0.0 for key in ("confirmation_upgrade", "confirmed_anomaly"))


def _suppression_reasons(
    state: PendingHandoffState,
    context: str,
    target_resolvable: bool,
    ppo_delta: int,
    min_ppo_steps: int,
) -> list[str]:
    reasons = ["STALE_PENDING", "NO_PROGRESS"]
    if context == state.last_context_fingerprint:
        reasons.append("SAME_CONTEXT")
    if not target_resolvable:
        reasons.append("TARGET_STILL_UNRESOLVABLE")
    if ppo_delta < min_ppo_steps:
        reasons.append("PPO_STEPS_BELOW_MIN")
    return reasons


def _context_fingerprint(observation: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    fields = [
        _url_region(_url(observation)),
        _functional_region(observation),
        str(page.get("modal_state") or ""),
        str(page.get("form_state") or ""),
        str(page.get("cart_count") or ""),
        str(page.get("selected_filter") or ""),
        str(page.get("selected_tab") or ""),
        str(page.get("current_page") or ""),
        str(page.get("auth_state") or page.get("login_state") or ""),
        str(page.get("entity_id") or page.get("product_id") or page.get("item_id") or page.get("order_id") or ""),
        str(page.get("viewport") or page.get("viewport_mode") or observation.get("viewport_mode") or ""),
        str(metadata.get("confirmation_mode") or ""),
        str(metadata.get("active_rule_id") or ""),
        str(metadata.get("expected_action_type") or ""),
        str(metadata.get("expected_target_identity") or ""),
        str(int(bool(metadata.get("expected_target_visible")))),
    ]
    return "|".join(fields)


def _functional_region(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    url = _url(observation).lower()
    candidates = observation.get("candidate_elements", [])
    text = " ".join(str(page.get(key) or "") for key in ("title", "url", "form_state", "selected_filter", "selected_tab")).lower()
    if isinstance(candidates, list):
        for candidate in candidates[:32]:
            if not isinstance(candidate, Mapping):
                continue
            haystack = " ".join(str(candidate.get(key) or "") for key in ("text", "aria_label", "role", "tag", "name")).lower()
            if candidate.get("is_form_field") or any(token in haystack for token in ("search", "filter", "checkout", "login", "save", "quantity")):
                return "functional_target"
            if candidate.get("is_cart_related") or candidate.get("is_checkout_related") or candidate.get("is_search_related"):
                return "functional_target"
    if any(token in f"{url} {text}" for token in ("cart", "checkout", "search", "filter", "login", "modal", "detail", "quantity")):
        return "functional_region"
    return ""


def _stalled_functional(reward_components: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    no_effect = float(reward_components.get("no_effect_penalty", 0.0) or reward_components.get("no_effect", 0.0) or 0.0) < 0.0
    invalid = float(reward_components.get("invalid_penalty", 0.0) or 0.0) < 0.0
    return bool(no_effect and not invalid and _functional_region(observation))


def _local_novelty(reward_components: Mapping[str, Any]) -> bool:
    return any(float(reward_components.get(key, 0.0) or 0.0) > 0.0 for key in ("new_local_state", "new_local_action_target", "candidate_anomaly", "confirmation_upgrade", "confirmed_anomaly"))


def _resolved_trigger_candidate(
    session: LocalSession,
    anomalies: Sequence[Mapping[str, Any]],
    confirmation_events: Sequence[Mapping[str, Any]],
    tracker_history: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    trigger_key = str(session.trigger_candidate_key or "")
    if not trigger_key:
        return ""
    confirmed_keys = _confirmed_event_keys(confirmation_events)
    confirmed_keys.update(_tracker_confirmed_keys(tracker_history))
    for anomaly in anomalies or ():
        if str(anomaly.get("status") or "") == "confirmed":
            key = _canonical_candidate_key(anomaly)
            if key:
                confirmed_keys.add(key)
    if trigger_key in confirmed_keys:
        return trigger_key
    return ""


def _resolved_entry_candidate(session: LocalSession, anomalies: Sequence[Mapping[str, Any]]) -> bool:
    if not session.entry_candidate_keys:
        return False
    for anomaly in anomalies or []:
        key = str(anomaly.get("logical_key") or anomaly.get("signature") or "")
        if key in session.entry_candidate_keys and str(anomaly.get("status") or "") == "confirmed":
            return True
    return False


def _current_unrelated_confirmed_keys(
    session: LocalSession,
    anomalies: Sequence[Mapping[str, Any]],
    confirmation_events: Sequence[Mapping[str, Any]],
) -> set[str]:
    trigger_key = str(session.trigger_candidate_key or "")
    if not trigger_key:
        return set()
    keys = _confirmed_event_keys(confirmation_events)
    for anomaly in anomalies or []:
        if str(anomaly.get("status") or "") == "confirmed":
            key = _canonical_candidate_key(anomaly)
            if key:
                keys.add(key)
    return {key for key in keys if key and key != trigger_key}


def _context_changed(session: LocalSession, observation: Mapping[str, Any]) -> bool:
    current_url = _url(observation)
    if _url_region(current_url) != _url_region(session.entry_url):
        return True
    region = _functional_region(observation)
    return bool(region and session.entry_functional_region and region != session.entry_functional_region)


def _url(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    return str(page.get("url") or observation.get("url") or "")


def _url_region(url: str) -> str:
    parsed = urlparse(str(url or ""))
    parts = [part for part in parsed.path.split("/") if part]
    return "/" + "/".join(parts[:2]) if parts else "/"


def _normalize_region_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _target_set_signature(observation: Mapping[str, Any]) -> str:
    candidates = observation.get("candidate_elements", [])
    signatures: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            if not (
                candidate.get("is_form_field")
                or candidate.get("is_cart_related")
                or candidate.get("is_checkout_related")
                or candidate.get("is_search_related")
                or str(candidate.get("role") or "").lower() in {"button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox", "menuitem"}
            ):
                continue
            signatures.append("|".join(_normalize_region_text(candidate.get(key)) for key in ("role", "tag", "aria_label", "name", "text")))
    return ";".join(sorted(set(item for item in signatures if item.strip("|"))))


def _action_target_context_signature(metadata: Mapping[str, Any]) -> str:
    return "|".join(_normalize_region_text(metadata.get(key)) for key in ("expected_action_type", "expected_target_identity", "confirmation_mode"))


def _relevant_evidence_signature(metadata: Mapping[str, Any]) -> str:
    explicit = metadata.get("relevant_evidence_signature")
    if explicit:
        return _normalize_region_text(explicit)
    relevance = metadata.get("functional_relevance")
    if str(relevance or "").upper() not in {"RELEVANT", "POSSIBLY_RELEVANT"}:
        return ""
    return "|".join(_normalize_region_text(metadata.get(key)) for key in ("active_rule_id", "confirmation_mode", "expected_action_type", "expected_target_identity"))


def _functional_region_key(observation: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    return "|".join((
        _url_region(_url(observation)),
        _target_set_signature(observation) or _functional_region(observation),
        _normalize_region_text(metadata.get("expected_action_type") or "functional"),
    ))


def _semantic_region_state_fingerprint(observation: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    stable_page = {
        key: _normalize_region_text(page.get(key))
        for key in ("modal_state", "form_state", "cart_count", "selected_filter", "selected_tab", "current_page", "auth_state", "login_state", "entity_id", "product_id", "item_id", "order_id", "viewport", "viewport_mode")
        if page.get(key) not in (None, "")
    }
    return json.dumps({"page": stable_page, "targets": _target_set_signature(observation), "action": _action_target_context_signature(metadata)}, sort_keys=True)


def _empty_functional_novelty() -> dict[str, Any]:
    return {
        "functional_region_detected": False,
        "region_key": "",
        "first_time_region": False,
        "semantic_state_changed": False,
        "new_target_context": False,
        "new_action_target_context": False,
        "new_evidence_context": False,
        "actionability_changed": False,
        "previous_local_outcome": "",
        "previous_local_strong_productive": False,
        "ppo_visit_count": 0,
        "novelty_eligible": False,
        "suppression_reasons": [],
        "handoff_performed": False,
        "handoff_reason": "",
    }
