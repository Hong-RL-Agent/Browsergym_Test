"""Discrete J.A.W.S action space utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from services.action_opportunity_service import action_ids_for_opportunities


class ActionSpace:
    action_types = [
        "noop",
        "click_element",
        "click_submit",
        "scroll_down",
        "scroll_up",
        "inspect_dom",
        "inspect_layout",
        "inspect_network",
        "inspect_console",
        "inspect_cart",
        "inspect_server_health",
        "inspect_port_status",
        "inspect_latency",
        "inspect_server_logs",
        "inspect_runtime_metrics",
        "inspect_network_status",
        "inspect_api_response",
        "inspect_console_errors",
        "inspect_resource_loading",
        "inspect_alert_card",
        "inspect_metric_card",
        "inspect_timeline",
        "click_trigger_button",
        "click_retry_button",
        "click_recovery_button",
        "open_detail_panel",
        "fill_input",
        "press_enter",
        "login_as_user",
        "login_as_admin",
        "logout",
        "check_login_state",
        "inspect_last_api_response",
        "replay_last_api_request",
        "call_collected_endpoint",
        "check_api_ui_match",
        "take_postgres_snapshot_before",
        "take_postgres_snapshot_after",
        "take_mysql_snapshot_before",
        "take_mysql_snapshot_after",
        "compare_relational_db_state",
        "check_transaction_consistency",
        "check_foreign_key_consistency",
        "check_constraint_violation",
        "check_inventory_invariant",
        "check_payment_order_consistency",
        "check_partial_commit",
        "check_rollback_result",
        "check_deadlock_or_lock_timeout",
        "capture_trace_id_after_action",
        "capture_request_id_after_action",
        "match_request_with_server_trace",
        "match_api_response_with_backend_exception",
        "match_db_change_with_trace",
        "correlate_ui_action_api_log_db",
        "inspect_trace_span",
        "inspect_error_span",
        "calculate_correlation_confidence",
        "validate_last_api_schema",
        "validate_response_contract",
        "compare_openapi_schema",
        "check_required_fields",
        "check_response_type_consistency",
        "check_unexpected_fields",
        "check_status_code_contract",
        "check_error_response_format",
        "check_ui_contract_mismatch",
        "generate_login_scenario",
        "execute_login_scenario",
        "login_as_role",
        "logout_and_revisit",
        "switch_role_and_revisit",
        "expire_session_and_retry",
        "verify_post_login_redirect",
        "verify_role_based_navigation",
        "verify_logout_access_block",
        "verify_session_expiration_behavior",
        "change_viewport_mobile",
        "change_viewport_desktop",
        "finish_episode",
    ]

    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = max_candidates
        self._type_to_id = {name: idx for idx, name in enumerate(self.action_types)}

    def encode(self, action_type: str, candidate_index: int = 0) -> int:
        if action_type not in self._type_to_id:
            raise ValueError(f"Unknown action_type: {action_type}")
        if not 0 <= candidate_index < self.max_candidates:
            raise ValueError(f"candidate_index must be in [0, {self.max_candidates})")
        return self._type_to_id[action_type] * self.max_candidates + candidate_index

    def decode(self, action_id: int) -> Dict[str, Any]:
        action_id = int(action_id)
        if not 0 <= action_id < self.get_action_dim():
            raise ValueError(f"action_id must be in [0, {self.get_action_dim()})")
        action_type_id = action_id // self.max_candidates
        candidate_index = action_id % self.max_candidates
        return {
            "action_type_id": action_type_id,
            "action_type": self.action_types[action_type_id],
            "candidate_index": candidate_index,
        }

    def is_element_action(self, action_type: str) -> bool:
        return action_type in {
            "click_element",
            "click_submit",
            "fill_input",
            "press_enter",
            "click_trigger_button",
            "click_retry_button",
            "click_recovery_button",
            "open_detail_panel",
        }

    def is_click_action(self, action_type: str) -> bool:
        return action_type in {
            "click_element",
            "click_submit",
            "click_trigger_button",
            "click_retry_button",
            "click_recovery_button",
            "open_detail_panel",
        }

    def is_infra_action(self, action_type: str) -> bool:
        return action_type in {
            "inspect_server_health",
            "inspect_port_status",
            "inspect_latency",
            "inspect_server_logs",
            "inspect_runtime_metrics",
            "inspect_network_status",
            "inspect_api_response",
            "inspect_console_errors",
            "inspect_resource_loading",
            "inspect_alert_card",
            "inspect_metric_card",
            "inspect_timeline",
        }

    def is_login_action(self, action_type: str) -> bool:
        return action_type in {"login_as_user", "login_as_admin", "logout", "check_login_state"}

    def is_api_probe_action(self, action_type: str) -> bool:
        return action_type in {
            "inspect_last_api_response",
            "replay_last_api_request",
            "call_collected_endpoint",
            "check_api_ui_match",
        }

    def is_relational_db_hook_action(self, action_type: str) -> bool:
        return action_type in _RELATIONAL_DB_HOOK_ACTIONS

    def is_trace_correlation_action(self, action_type: str) -> bool:
        return action_type in _TRACE_CORRELATION_ACTIONS

    def is_api_contract_action(self, action_type: str) -> bool:
        return action_type in _API_CONTRACT_ACTIONS

    def is_login_scenario_action(self, action_type: str) -> bool:
        return action_type in _LOGIN_SCENARIO_ACTIONS

    def get_action_dim(self) -> int:
        return len(self.action_types) * self.max_candidates

    def build_action_mask(self, raw_observation: Mapping[str, Any]) -> np.ndarray:
        mask = np.zeros(self.get_action_dim(), dtype=np.float32)
        candidates = raw_observation.get("candidate_elements", []) if raw_observation else []
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        page_state = raw_observation.get("page_state", {}) if isinstance(raw_observation, Mapping) else {}
        runtime_signals = raw_observation.get("runtime_signals", {}) if isinstance(raw_observation, Mapping) else {}
        history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
        opportunity_summary = history.get("opportunity_summary", {}) if isinstance(history, Mapping) else {}
        summary_allows_finish = bool(opportunity_summary.get("finish_allowed", False)) if isinstance(opportunity_summary, Mapping) else False
        page_verification = history.get("page_verification", {}) if isinstance(history, Mapping) else {}
        if isinstance(runtime_signals, Mapping):
            runtime_signals = dict(runtime_signals)
            if isinstance(page_verification, Mapping):
                for key in ("page_type", "has_json_response"):
                    if page_verification.get(key) is not None:
                        runtime_signals.setdefault(key, page_verification.get(key))
        infra_signals = raw_observation.get("infra_signals", {}) if isinstance(raw_observation, Mapping) else {}
        site_id = str(page_state.get("site_id") or runtime_signals.get("site_id") or "")
        if not site_id:
            url = str(page_state.get("url") or "")
            site_id = "site003" if ":9221" in url else "site001" if ":9220" in url else ""
        infra_enabled = _is_infra_port(page_state, infra_signals)
        login_credentials_available = bool(runtime_signals.get("login_credentials_available"))
        logged_in = bool(runtime_signals.get("logged_in") or runtime_signals.get("login_success"))
        api_probe_available = _api_probe_available(runtime_signals)
        db_policy_state = _relational_db_policy_state(runtime_signals, infra_signals)
        trace_policy_state = _trace_policy_state(runtime_signals, infra_signals)
        contract_policy_state = _api_contract_policy_state(runtime_signals)
        login_scenario_state = _login_scenario_policy_state(runtime_signals)

        for action_type in self.action_types:
            if site_id and site_id not in {"site001", "site9800"} and action_type == "inspect_cart":
                continue
            if self.is_infra_action(action_type) and not infra_enabled:
                continue
            if self.is_login_action(action_type) and not _login_action_allowed(
                action_type,
                login_credentials_available=login_credentials_available,
                logged_in=logged_in,
            ):
                continue
            if self.is_api_probe_action(action_type) and not api_probe_available:
                continue
            if self.is_relational_db_hook_action(action_type) and not _relational_db_action_allowed(action_type, db_policy_state):
                continue
            if self.is_trace_correlation_action(action_type) and not _trace_action_allowed(action_type, trace_policy_state):
                continue
            if self.is_api_contract_action(action_type) and not _api_contract_action_allowed(action_type, contract_policy_state):
                continue
            if self.is_login_scenario_action(action_type) and not _login_scenario_action_allowed(action_type, login_scenario_state):
                continue
            if action_type == "finish_episode":
                # Observation-driven floor for candidates the adapter flagged as
                # worth interacting with (cart buttons, quantity controls, forum
                # write/submit/delete, etc.): this used to be a separate ad-hoc
                # check here (_has_unclicked_functional_priority_candidate) and a
                # second, blunter one for "any network activity was never
                # inspected" (_has_pending_network_or_console_verification).
                # Both are now folded into action_opportunity_service.py's own
                # required-opportunity generation (functional_priority_candidate
                # candidates are marked required=True there; console/network
                # verification opportunities already existed there with more
                # nuance -- e.g. network inspection is only required on API/JSON
                # pages, not on every page that merely loaded a script) so
                # summary_allows_finish below already accounts for them. Having
                # two independently-recomputed judgments of the same thing is
                # exactly what caused a real finish/reject infinite loop earlier
                # this session; opportunity_summary is now the single source of
                # truth for "is there still real work to do."
                #
                # Optional fixed floor (min_steps, default 0/off): a config-level
                # safety net for sites where the adapter finds zero
                # functional-priority candidates at all, so a page can't be declared
                # done in a single step just because nothing looked interactive.
                if int(history.get("step_index", 0) or 0) < int(history.get("min_steps", 0) or 0):
                    continue
                # _has_unprocessed_login_form and _has_required_page_verifications
                # are checked unconditionally, not only when the opportunity
                # summary itself claims finish isn't allowed yet: that summary
                # and the runner's own end-of-episode acceptance check are
                # separate systems that can disagree (e.g. a registration form
                # not tracked as a "required opportunity" but still unfilled) --
                # if summary_allows_finish alone gated this, finish_episode
                # stays permanently unmasked while the runner keeps rejecting
                # it every step, producing an infinite finish/reject loop.
                # (The broader _has_unprocessed_form_candidates -- which also
                # treats any incidental optional fillable field as blocking --
                # stays scoped to the summary_allows_finish=False branch below,
                # since applying it unconditionally would block finish on
                # purely optional fields too.)
                if _has_unprocessed_login_form(history):
                    continue
                if _has_required_page_verifications(history):
                    continue
                if not summary_allows_finish:
                    if _has_unprocessed_form_candidates(candidates, history):
                        continue
                    if _has_required_action_opportunities(history):
                        continue
            if action_type == "inspect_dom" and _inspect_dom_repeated_same_context(raw_observation, history):
                continue
            if self.is_element_action(action_type):
                for idx in range(min(candidate_count, self.max_candidates)):
                    candidate = candidates[idx] if isinstance(candidates, list) else {}
                    if not isinstance(candidate, Mapping):
                        continue
                    if action_type in {"fill_input", "press_enter"} and not _is_input_candidate(candidate):
                        continue
                    if action_type == "click_submit" and not _is_submit_candidate(candidate):
                        continue
                    if action_type == "click_submit" and _submit_candidate_already_attempted(candidate, history):
                        continue
                    if action_type in {"click_element", "click_submit", "fill_input", "press_enter"} and _candidate_previously_failed(candidate, history):
                        continue
                    # click_trigger_button/click_retry_button/click_recovery_button/open_detail_panel
                    # are heuristic infra-dashboard actions, not generic synonyms for click_element.
                    # Without gating on the matching semantic flag, they were enabled on every
                    # clickable candidate exactly like click_element, quadrupling the effective
                    # click-action mask surface on ordinary (non-infra) sites with candidates that
                    # were never actually retry/recovery/trigger/detail buttons -- diluting
                    # exploration and driving up debug-meta repeat penalties for no reason.
                    if action_type == "click_trigger_button" and not bool(candidate.get("is_api_trigger")):
                        continue
                    if action_type == "click_retry_button" and not bool(candidate.get("is_retry_button")):
                        continue
                    if action_type == "click_recovery_button" and not bool(candidate.get("is_recovery_button")):
                        continue
                    if action_type == "open_detail_panel" and not bool(candidate.get("is_detail_trigger")):
                        continue
                    if self.is_click_action(action_type) and not _is_clickable_candidate(candidate):
                        continue
                    mask[self.encode(action_type, idx)] = 1.0
            else:
                mask[self.encode(action_type, 0)] = 1.0

        _apply_login_form_priority(mask, self, candidates, raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {})
        _apply_api_json_priority(mask, self, raw_observation)
        _apply_action_opportunity_mask(mask, self, raw_observation)
        _restore_finish_when_allowed(mask, self, raw_observation)

        if not mask.any():
            mask[self.encode("noop", 0)] = 1.0
        return mask

    def build_action_mask_stats(
        self,
        raw_observation: Mapping[str, Any],
        action_mask: np.ndarray,
        selected_action_id: int,
    ) -> Dict[str, Any]:
        candidates = raw_observation.get("candidate_elements", []) if raw_observation else []
        if not isinstance(candidates, list):
            candidates = []
        selected = self.decode(int(selected_action_id))
        selected_index = int(selected.get("candidate_index", 0) or 0)
        selected_candidate = candidates[selected_index] if 0 <= selected_index < len(candidates) else {}
        if not isinstance(selected_candidate, Mapping):
            selected_candidate = {}
        history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
        login_flow = history.get("login_flow", {}) if isinstance(history, Mapping) else {}
        if not isinstance(login_flow, Mapping):
            login_flow = {}
        page_verification = history.get("page_verification", {}) if isinstance(history, Mapping) else {}
        if not isinstance(page_verification, Mapping):
            page_verification = {}
        opportunity_summary = history.get("opportunity_summary", {}) if isinstance(history, Mapping) else {}
        if not isinstance(opportunity_summary, Mapping):
            opportunity_summary = {}
        opportunities = history.get("action_opportunities", []) if isinstance(history, Mapping) else []
        if not isinstance(opportunities, list):
            opportunities = []
        selected_opportunity = _selected_opportunity_for_action(opportunities, str(selected.get("action_type") or ""), selected_index)

        valid_click_count = 0
        valid_fill_count = 0
        valid_submit_count = 0
        valid_global_action_count = 0
        submit_candidate_count = sum(
            1 for candidate in candidates if isinstance(candidate, Mapping) and _is_submit_candidate(candidate)
        )
        for action_type in self.action_types:
            if self.is_click_action(action_type):
                valid_click_count += self._valid_count_for_type(action_mask, action_type)
            elif action_type == "fill_input":
                valid_fill_count += self._valid_count_for_type(action_mask, action_type)
            if action_type == "click_submit":
                valid_submit_count += self._valid_count_for_type(action_mask, action_type)
            elif not self.is_element_action(action_type):
                action_id = self.encode(action_type, 0)
                if action_id < len(action_mask) and float(action_mask[action_id]) > 0.0:
                    valid_global_action_count += 1

        return {
            "candidate_count": min(len(candidates), self.max_candidates),
            "valid_action_count": int(np.count_nonzero(action_mask)),
            "valid_click_count": valid_click_count,
            "valid_fill_count": valid_fill_count,
            "submit_count": submit_candidate_count,
            "valid_submit_count": valid_submit_count,
            "valid_global_action_count": valid_global_action_count,
            "selected_action_type": str(selected.get("action_type") or ""),
            "selected_candidate_index": selected_index,
            "selected_candidate_role": str(selected_candidate.get("role") or ""),
            "selected_candidate_tag": str(selected_candidate.get("tag") or ""),
            "selected_candidate_visible": bool(selected_candidate.get("visible")),
            "selected_candidate_enabled": bool(selected_candidate.get("enabled", True)),
            "selected_candidate_clickable": bool(selected_candidate.get("clickable")),
            "selected_target_element_key": str(selected_candidate.get("element_key") or ""),
            "selected_target_text": str(selected_candidate.get("text") or selected_candidate.get("name") or ""),
            "selected_target_name": str(selected_candidate.get("name") or selected_candidate.get("text") or ""),
            "selected_target_role": str(selected_candidate.get("role") or ""),
            "selected_target_tag": str(selected_candidate.get("tag") or ""),
            "selected_target_type": str(selected_candidate.get("type") or selected_candidate.get("input_type") or ""),
            "fillable_count": sum(1 for candidate in candidates if isinstance(candidate, Mapping) and _is_input_candidate(candidate)),
            "clickable_count": sum(1 for candidate in candidates if isinstance(candidate, Mapping) and _is_clickable_candidate(candidate)),
            "password_input_count": sum(1 for candidate in candidates if isinstance(candidate, Mapping) and bool(candidate.get("is_password"))),
            "action_mask_enabled_actions": self.enabled_action_types(action_mask),
            "high_priority_candidate_count": sum(
                1
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority") or candidate.get("is_submit") or candidate.get("is_password"))
            ),
            "selected_target_semantic_type": str(selected_candidate.get("semantic_action_type") or ""),
            "action_priority_reason": _action_priority_reason(str(selected.get("action_type") or ""), selected_candidate, login_flow),
            "login_flow_status": str(login_flow.get("login_flow_status") or ""),
            "login_flow_status_after_step": str(login_flow.get("login_flow_status") or ""),
            "has_login_form": bool(login_flow.get("has_login_form", False)),
            "has_username_or_email_input": bool(login_flow.get("has_username_or_email_input") or login_flow.get("has_email_or_text_input")),
            "has_password_input": bool(login_flow.get("has_password_input", False)),
            "has_login_submit": bool(login_flow.get("has_login_submit") or login_flow.get("has_submit_button")),
            "username_or_email_filled": bool(login_flow.get("username_or_email_filled") or login_flow.get("email_input_filled")),
            "password_filled": bool(login_flow.get("password_filled") or login_flow.get("password_input_filled")),
            "login_submit_clicked": bool(login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked")),
            "login_result_checked": bool(login_flow.get("login_result_checked") or login_flow.get("submit_result_checked")),
            "login_flow_attempted": bool(login_flow.get("login_flow_attempted", False)),
            "login_flow_completed": bool(login_flow.get("login_flow_completed", False)),
            "required_actions_remaining": list(login_flow.get("required_actions_remaining") or []),
            "page_type": str(page_verification.get("page_type") or ""),
            "required_verifications_remaining": list(page_verification.get("required_verifications_remaining") or []),
            "required_verifications_completed": bool(page_verification.get("required_verifications_completed", False)),
            "api_response_checked": bool(page_verification.get("api_response_checked", False)),
            "json_parse_checked": bool(page_verification.get("json_parse_checked", False)),
            "schema_contract_checked": bool(page_verification.get("schema_contract_checked", False)),
            "network_checked": bool(page_verification.get("network_checked", False)),
            "console_checked": bool(page_verification.get("console_checked", False)),
            "detected_candidate_count": int(opportunity_summary.get("detected_candidate_count", len(candidates)) or 0),
            "generated_opportunity_count": int(opportunity_summary.get("generated_opportunity_count", 0) or 0),
            "required_opportunity_count": int(opportunity_summary.get("required_opportunity_count", 0) or 0),
            "optional_opportunity_count": int(opportunity_summary.get("optional_opportunity_count", 0) or 0),
            "executed_opportunity_count": int(opportunity_summary.get("executed_opportunity_count", 0) or 0),
            "verified_opportunity_count": int(opportunity_summary.get("verified_opportunity_count", 0) or 0),
            "skipped_opportunity_count": int(opportunity_summary.get("skipped_opportunity_count", 0) or 0),
            "failed_opportunity_count": int(opportunity_summary.get("failed_opportunity_count", 0) or 0),
            "remaining_opportunity_count": int(opportunity_summary.get("remaining_opportunity_count", 0) or 0),
            "pending_opportunity_count": int(opportunity_summary.get("pending_opportunity_count", opportunity_summary.get("remaining_opportunity_count", 0)) or 0),
            "remaining_blocking_opportunity_count": int(opportunity_summary.get("remaining_blocking_opportunity_count", 0) or 0),
            "pending_blocking_opportunity_count": int(opportunity_summary.get("pending_blocking_opportunity_count", opportunity_summary.get("remaining_blocking_opportunity_count", 0)) or 0),
            "remaining_required_opportunity_count": int(opportunity_summary.get("remaining_required_opportunity_count", 0) or 0),
            "pending_required_opportunity_count": int(opportunity_summary.get("pending_required_opportunity_count", opportunity_summary.get("remaining_required_opportunity_count", 0)) or 0),
            "action_opportunity_coverage_rate": float(opportunity_summary.get("action_opportunity_coverage_rate", 0.0) or 0.0),
            "action_coverage_rate": float(opportunity_summary.get("action_coverage_rate", opportunity_summary.get("action_opportunity_coverage_rate", 0.0)) or 0.0),
            "required_opportunity_completion_rate": float(opportunity_summary.get("required_opportunity_completion_rate", 0.0) or 0.0),
            "optional_opportunity_coverage_rate": float(opportunity_summary.get("optional_opportunity_coverage_rate", 0.0) or 0.0),
            "unverified_anomaly_count": int(opportunity_summary.get("unverified_anomaly_count", 0) or 0),
            "verified_finding_count": int(opportunity_summary.get("verified_finding_count", 0) or 0),
            "anomaly_verification_required": bool(opportunity_summary.get("anomaly_verification_required", False)),
            "anomaly_verification_completed": bool(opportunity_summary.get("anomaly_verification_completed", True)),
            "finish_allowed": bool(opportunity_summary.get("finish_allowed", False)),
            "finish_blocked_reason": str(opportunity_summary.get("finish_blocked_reason") or ""),
            "dom_exhaustive_action_mode": bool(opportunity_summary.get("dom_exhaustive_action_mode", False)),
            "opportunity_diagnostics": list(opportunity_summary.get("opportunity_diagnostics") or []),
            "selected_opportunity_id": str(selected_opportunity.get("opportunity_id") or ""),
            "selected_opportunity_type": str(selected_opportunity.get("opportunity_type") or ""),
            "opportunity_required": bool(selected_opportunity.get("required", False)),
            "opportunity_status_before": str(selected_opportunity.get("status") or ""),
            "opportunity_status_after_action": str(selected_opportunity.get("status") or ""),
            "next_recommended_verification_action": _next_recommended_verification_action(opportunities, opportunity_summary),
            "repeated_inspect_dom_count": int(history.get("repeated_inspect_dom_count", 0) or 0),
            "repeated_inspect_dom_penalty_applied": int(history.get("repeated_inspect_dom_penalty_applied", 0) or 0),
            "repeated_inspect_dom_blocked": int(history.get("repeated_inspect_dom_blocked", 0) or 0),
        }

    def enabled_action_types(self, action_mask: np.ndarray) -> list[str]:
        enabled: list[str] = []
        for action_type in self.action_types:
            if self._valid_count_for_type(action_mask, action_type) > 0:
                enabled.append(action_type)
        return enabled

    def _valid_count_for_type(self, action_mask: np.ndarray, action_type: str) -> int:
        start = self.encode(action_type, 0)
        end = start + self.max_candidates
        return int(np.count_nonzero(action_mask[start:end]))


_RELATIONAL_DB_HOOK_ACTIONS = {
    "take_postgres_snapshot_before",
    "take_postgres_snapshot_after",
    "take_mysql_snapshot_before",
    "take_mysql_snapshot_after",
    "compare_relational_db_state",
    "check_transaction_consistency",
    "check_foreign_key_consistency",
    "check_constraint_violation",
    "check_inventory_invariant",
    "check_payment_order_consistency",
    "check_partial_commit",
    "check_rollback_result",
    "check_deadlock_or_lock_timeout",
}

_TRACE_CORRELATION_ACTIONS = {
    "capture_trace_id_after_action",
    "capture_request_id_after_action",
    "match_request_with_server_trace",
    "match_api_response_with_backend_exception",
    "match_db_change_with_trace",
    "correlate_ui_action_api_log_db",
    "inspect_trace_span",
    "inspect_error_span",
    "calculate_correlation_confidence",
}

_API_CONTRACT_ACTIONS = {
    "validate_last_api_schema",
    "validate_response_contract",
    "compare_openapi_schema",
    "check_required_fields",
    "check_response_type_consistency",
    "check_unexpected_fields",
    "check_status_code_contract",
    "check_error_response_format",
    "check_ui_contract_mismatch",
}

_LOGIN_SCENARIO_ACTIONS = {
    "generate_login_scenario",
    "execute_login_scenario",
    "login_as_role",
    "logout_and_revisit",
    "switch_role_and_revisit",
    "expire_session_and_retry",
    "verify_post_login_redirect",
    "verify_role_based_navigation",
    "verify_logout_access_block",
    "verify_session_expiration_behavior",
}


def _is_infra_port(page_state: Mapping[str, Any], infra_signals: Any) -> bool:
    port = None
    if isinstance(infra_signals, Mapping):
        port = infra_signals.get("port")
    if port in (None, ""):
        url = str(page_state.get("url") or "")
        for token in url.split(":")[2:3]:
            digits = "".join(char for char in token if char.isdigit())
            if digits:
                port = digits
                break
    try:
        value = int(port)
    except (TypeError, ValueError):
        return False
    return 9000 <= value <= 9100


def _is_clickable_candidate(candidate: Mapping[str, Any]) -> bool:
    return _is_visible_enabled(candidate) and bool(candidate.get("clickable"))


def _is_submit_candidate(candidate: Mapping[str, Any]) -> bool:
    return _is_visible_enabled(candidate) and bool(candidate.get("is_submit")) and bool(candidate.get("clickable", True))


def _has_unprocessed_login_form(history: Mapping[str, Any]) -> bool:
    """Narrow, unconditional-safe check: is there a recognized login/registration
    form (login_flow.has_login_form) whose required steps aren't done yet?
    Unlike _has_unprocessed_form_candidates, this ignores incidental/optional
    fillable fields (e.g. a search box) so it's safe to enforce even when the
    opportunity summary already claims finish is allowed.
    """
    login_flow = history.get("login_flow", {}) if isinstance(history, Mapping) else {}
    if not isinstance(login_flow, Mapping) or not bool(login_flow.get("has_login_form")):
        return False
    if bool(login_flow.get("login_flow_completed")):
        return False
    remaining = login_flow.get("required_actions_remaining")
    if isinstance(remaining, list) and remaining:
        return True
    if bool(login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked")) and not bool(login_flow.get("login_result_checked") or login_flow.get("submit_result_checked")):
        return True
    return False


def _has_unprocessed_form_candidates(candidates: Any, history: Mapping[str, Any]) -> bool:
    if _has_unprocessed_login_form(history):
        return True
    if not isinstance(candidates, list):
        return False
    counts = history.get("action_type_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    fill_done = int(counts.get("fill_input", 0) or 0) > 0
    submit_done = int(counts.get("click_submit", 0) or 0) > 0 or int(counts.get("press_enter", 0) or 0) > 0
    has_input = any(
        isinstance(candidate, Mapping)
        and _is_visible_enabled(candidate)
        and (bool(candidate.get("fillable")) or bool(candidate.get("is_password")))
        for candidate in candidates
    )
    has_submit = any(
        isinstance(candidate, Mapping) and _is_visible_enabled(candidate) and bool(candidate.get("is_submit"))
        for candidate in candidates
    )
    return bool((has_input and not fill_done) or (has_submit and not submit_done))


def _has_required_page_verifications(history: Mapping[str, Any]) -> bool:
    page_verification = history.get("page_verification", {}) if isinstance(history, Mapping) else {}
    if not isinstance(page_verification, Mapping):
        return False
    remaining = page_verification.get("required_verifications_remaining")
    return isinstance(remaining, list) and bool(remaining)


def _has_required_action_opportunities(history: Mapping[str, Any]) -> bool:
    summary = history.get("opportunity_summary", {}) if isinstance(history, Mapping) else {}
    if not isinstance(summary, Mapping):
        return False
    if int(summary.get("remaining_required_opportunity_count", 0) or 0) > 0:
        return True
    if int(summary.get("unverified_anomaly_count", 0) or 0) > 0:
        return True
    blocked_reason = str(summary.get("finish_blocked_reason") or "")
    return blocked_reason in {
        "required_opportunity_remaining",
        "unverified_anomaly_remaining",
        "login_flow_incomplete",
        "observed_opportunity_remaining",
        "observed_actionable_opportunity_remaining",
    }


def _apply_login_form_priority(mask: np.ndarray, action_space: ActionSpace, candidates: Any, history: Mapping[str, Any]) -> None:
    if not isinstance(candidates, list) or not isinstance(history, Mapping):
        return
    login_flow = history.get("login_flow", {})
    if not isinstance(login_flow, Mapping):
        login_flow = {}
    has_login_form = bool(login_flow.get("has_login_form")) or _candidate_login_form_present(candidates)
    if not has_login_form:
        return
    email_done = bool(login_flow.get("username_or_email_filled") or login_flow.get("email_input_filled"))
    password_done = bool(login_flow.get("password_filled") or login_flow.get("password_input_filled"))
    submit_done = bool(login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked"))
    result_checked = bool(login_flow.get("login_result_checked") or login_flow.get("submit_result_checked"))

    preferred: list[int] = []
    if not email_done:
        preferred.extend(_candidate_action_ids(action_space, mask, candidates, "fill_input", _is_email_text_candidate))
    elif not password_done:
        preferred.extend(_candidate_action_ids(action_space, mask, candidates, "fill_input", _is_password_candidate))
    elif email_done and password_done and not submit_done:
        preferred.extend(_candidate_action_ids(action_space, mask, candidates, "click_submit", _is_submit_candidate))
    elif submit_done and not result_checked:
        for action_type in ("inspect_network", "inspect_console", "inspect_dom"):
            try:
                action_id = action_space.encode(action_type, 0)
            except ValueError:
                continue
            if action_id < len(mask) and float(mask[action_id]) > 0.0:
                preferred.append(action_id)
    if not preferred:
        return
    mask[:] = 0.0
    for action_id in preferred:
        if 0 <= action_id < len(mask):
            mask[action_id] = 1.0


def _apply_action_opportunity_mask(mask: np.ndarray, action_space: ActionSpace, raw_observation: Mapping[str, Any]) -> None:
    history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
    if not isinstance(history, Mapping):
        return
    opportunities = history.get("action_opportunities")
    summary = history.get("opportunity_summary", {})
    if not isinstance(opportunities, list) or not isinstance(summary, Mapping):
        return
    pending = [
        item for item in opportunities
        if isinstance(item, Mapping)
        and not bool(item.get("verified"))
        and not str(item.get("skipped_reason") or "")
    ]
    if not pending:
        return
    required_pending = [
        item
        for item in pending
        if bool(item.get("required")) and not _required_opportunity_satisfied(item)
    ]
    if required_pending:
        console_required = [
            item
            for item in required_pending
            if str(item.get("opportunity_type") or "") == "verify_runtime_error"
        ]
        selected = console_required or required_pending
    else:
        selected = [
            item for item in pending
            if not bool(item.get("executed")) and str(item.get("opportunity_type") or "") != "finish_episode"
        ]
    action_ids = action_ids_for_opportunities(selected, action_space)
    action_ids = [action_id for action_id in action_ids if 0 <= action_id < len(mask) and float(mask[action_id]) > 0.0]
    if not action_ids:
        return
    finish_id = action_space.encode("finish_episode", 0)
    if not bool(summary.get("finish_allowed", False)) and finish_id < len(mask):
        mask[finish_id] = 0.0
    if required_pending:
        kept = np.zeros_like(mask)
        for action_id in action_ids:
            kept[action_id] = 1.0
        mask[:] = kept


def _restore_finish_when_allowed(mask: np.ndarray, action_space: ActionSpace, raw_observation: Mapping[str, Any]) -> None:
    history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
    if not isinstance(history, Mapping):
        return
    summary = history.get("opportunity_summary", {})
    if not isinstance(summary, Mapping) or not bool(summary.get("finish_allowed", False)):
        return
    if int(history.get("step_index", 0) or 0) < int(history.get("min_steps", 0) or 0):
        return
    if _has_unprocessed_login_form(history):
        return
    if _has_required_page_verifications(history) or _has_required_action_opportunities(history):
        return
    finish_id = action_space.encode("finish_episode", 0)
    if 0 <= finish_id < len(mask):
        mask[finish_id] = 1.0


def _selected_opportunity_for_action(opportunities: list[Any], action_type: str, candidate_index: int) -> Mapping[str, Any]:
    for item in opportunities:
        if not isinstance(item, Mapping):
            continue
        mapped_actions = _opportunity_action_types(item)
        if action_type not in mapped_actions:
            continue
        if action_type in {"click_element", "click_submit", "fill_input", "press_enter"}:
            if int(item.get("candidate_index", 0) or 0) != int(candidate_index):
                continue
        return item
    return {}


def _opportunity_action_type(opportunity: Mapping[str, Any]) -> str:
    action_types = _opportunity_action_types(opportunity)
    return action_types[0] if action_types else ""


def _opportunity_action_types(opportunity: Mapping[str, Any]) -> tuple[str, ...]:
    value = {
        "fill_text_input": "fill_input",
        "fill_email_input": "fill_input",
        "fill_password_input": "fill_input",
        "fill_search_input": "fill_input",
        "fill_username_or_email": "fill_input",
        "fill_password": "fill_input",
        "click_submit": "click_submit",
        "submit_form": "click_submit",
        "click_login_submit": "click_submit",
        "click_button": "click_element",
        "click_link": "click_element",
        "inspect_network": "inspect_network",
        "inspect_console": "inspect_console",
        "inspect_dom": "inspect_dom",
        "inspect_layout": "inspect_layout",
        "check_api_response": ("inspect_last_api_response", "inspect_network", "inspect_api_response"),
        "check_api_endpoint_link": ("inspect_network", "inspect_dom"),
        "check_search_result_after_input": "inspect_dom",
        "check_json_schema": ("validate_last_api_schema", "inspect_last_api_response", "inspect_network"),
        "select_option": "click_element",
        "verify_state_change": "inspect_dom",
        "verify_anomaly": "inspect_dom",
        "verify_no_effect": "inspect_dom",
        "verify_anomaly_reproduction": "inspect_dom",
        "verify_duplicated_rendering": "inspect_dom",
        "verify_runtime_error": "inspect_console",
        "verify_form_no_feedback": "inspect_dom",
        "verify_action_result": ("inspect_network", "inspect_console", "inspect_dom"),
        "verify_login_result": ("inspect_network", "inspect_console", "inspect_dom", "check_login_state"),
        "change_viewport_once": "change_viewport_mobile",
        "restore_viewport": "change_viewport_desktop",
        "finish_episode": "finish_episode",
    }.get(str(opportunity.get("opportunity_type") or ""), ())
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _next_recommended_verification_action(opportunities: list[Any], summary: Mapping[str, Any]) -> str:
    if int(summary.get("unverified_anomaly_count", 0) or 0) > 0:
        for item in opportunities:
            if isinstance(item, Mapping) and str(item.get("opportunity_type") or "").startswith("verify_") and not bool(item.get("verified")):
                return _opportunity_action_type(item)
        return "inspect_dom"
    for item in opportunities:
        if not isinstance(item, Mapping) or bool(item.get("verified")) or bool(item.get("executed")):
            continue
        if bool(item.get("required")):
            return _opportunity_action_type(item)
    return ""


def _required_opportunity_satisfied(opportunity: Mapping[str, Any]) -> bool:
    if opportunity.get("skipped_reason"):
        return True
    if opportunity.get("verified"):
        return True
    if opportunity.get("failed"):
        return False
    return bool(opportunity.get("executed") and opportunity.get("success"))


def _inspect_dom_repeated_same_context(raw_observation: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    if not isinstance(raw_observation, Mapping) or not isinstance(history, Mapping):
        return False
    opportunities = history.get("action_opportunities")
    if isinstance(opportunities, list):
        for item in opportunities:
            if (
                isinstance(item, Mapping)
                and bool(item.get("required"))
                and not _required_opportunity_satisfied(item)
                and "inspect_dom" in _opportunity_action_types(item)
            ):
                return False
    inspected = history.get("inspect_dom_contexts_inspected")
    if isinstance(inspected, set):
        inspected_contexts = {str(item) for item in inspected}
    elif isinstance(inspected, (list, tuple)):
        inspected_contexts = {str(item) for item in inspected}
    else:
        inspected_contexts = set()
    repeated = _dom_context_key(raw_observation) in inspected_contexts
    if repeated and isinstance(history, dict):
        history["repeated_inspect_dom_blocked"] = int(history.get("repeated_inspect_dom_blocked", 0) or 0) + 1
    return repeated


def _dom_context_key(raw_observation: Mapping[str, Any]) -> str:
    import hashlib
    import json

    page_state = raw_observation.get("page_state", {}) if isinstance(raw_observation, Mapping) else {}
    payload = {
        "url": str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        "dom": str(page_state.get("dom_hash") or _fallback_dom_signature(raw_observation)) if isinstance(page_state, Mapping) else _fallback_dom_signature(raw_observation),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _fallback_dom_signature(raw_observation: Mapping[str, Any]) -> str:
    import hashlib
    import json

    page_state = raw_observation.get("page_state", {}) if isinstance(raw_observation, Mapping) else {}
    candidates = raw_observation.get("candidate_elements", []) if isinstance(raw_observation, Mapping) else []
    payload = {
        "url": str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        "text_len": int(page_state.get("page_text_length") or len(str(page_state.get("page_text") or ""))) if isinstance(page_state, Mapping) else 0,
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _submit_candidate_already_attempted(candidate: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    if not isinstance(history, Mapping):
        return False
    counts = history.get("action_type_counts", {})
    if isinstance(counts, Mapping) and int(counts.get("click_submit", 0) or 0) > 0:
        return True
    key = _candidate_identity_key(candidate)
    if not key:
        return False
    click_counts = history.get("click_target_counts", {})
    if isinstance(click_counts, Mapping) and int(click_counts.get(key, 0) or 0) > 0:
        return True
    element_counts = history.get("element_key_click_counts", {})
    if isinstance(element_counts, Mapping) and int(element_counts.get(key, 0) or 0) > 0:
        return True
    submit_keys = history.get("submitted_element_keys", set())
    if isinstance(submit_keys, set) and key in submit_keys:
        return True
    if isinstance(submit_keys, (list, tuple)) and key in {str(item) for item in submit_keys}:
        return True
    return False


def _candidate_previously_failed(candidate: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    if not isinstance(history, Mapping):
        return False
    failed_keys = history.get("failed_target_element_keys")
    if not failed_keys:
        return False
    key = _candidate_identity_key(candidate)
    return bool(key) and key in failed_keys


def _candidate_identity_key(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("element_key")
        or candidate.get("bid")
        or candidate.get("name")
        or candidate.get("text")
        or candidate.get("selector")
        or candidate.get("locator")
        or ""
    )


def _apply_api_json_priority(mask: np.ndarray, action_space: ActionSpace, raw_observation: Mapping[str, Any]) -> None:
    history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
    page_verification = history.get("page_verification", {}) if isinstance(history, Mapping) else {}
    if not isinstance(page_verification, Mapping):
        return
    page_type = str(page_verification.get("page_type") or "")
    if page_type not in {"api_json_page", "data_endpoint_page"}:
        return
    remaining = page_verification.get("required_verifications_remaining")
    if not isinstance(remaining, list) or not remaining:
        return
    preferred_by_requirement = {
        "api_response_check": ("inspect_last_api_response", "inspect_network", "inspect_api_response"),
        "json_parse_check": ("validate_last_api_schema", "check_required_fields", "inspect_last_api_response"),
        "schema_contract_check": ("validate_last_api_schema", "validate_response_contract", "check_required_fields"),
        "network_check": ("inspect_network",),
        "console_check": ("inspect_console", "inspect_console_errors"),
    }
    preferred: list[int] = []
    for requirement in remaining:
        for action_type in preferred_by_requirement.get(str(requirement), ()):
            try:
                action_id = action_space.encode(action_type, 0)
            except ValueError:
                continue
            if action_id < len(mask) and float(mask[action_id]) > 0.0:
                preferred.append(action_id)
        if preferred:
            break
    if not preferred:
        return
    mask[:] = 0.0
    for action_id in preferred:
        mask[action_id] = 1.0


def _candidate_action_ids(
    action_space: ActionSpace,
    mask: np.ndarray,
    candidates: list[Any],
    action_type: str,
    predicate: Any,
) -> list[int]:
    action_ids: list[int] = []
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping) or not predicate(candidate):
            continue
        try:
            action_id = action_space.encode(action_type, index)
        except ValueError:
            continue
        if action_id < len(mask) and float(mask[action_id]) > 0.0:
            action_ids.append(action_id)
    return action_ids


def _candidate_login_form_present(candidates: list[Any]) -> bool:
    return bool(
        any(isinstance(candidate, Mapping) and _is_email_text_candidate(candidate) for candidate in candidates)
        and any(isinstance(candidate, Mapping) and _is_password_candidate(candidate) for candidate in candidates)
        and any(isinstance(candidate, Mapping) and _is_submit_candidate(candidate) for candidate in candidates)
    )


def _is_password_candidate(candidate: Mapping[str, Any]) -> bool:
    return _is_input_candidate(candidate) and bool(candidate.get("is_password"))


def _is_email_text_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_input_candidate(candidate) or bool(candidate.get("is_password")):
        return False
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ("name", "text", "placeholder", "title", "id")
    ).lower()
    return bool(
        input_type in {"", "text", "email", "search"}
        or role in {"textbox", "searchbox"}
        or tag == "textarea"
        or any(token in text for token in ("email", "username", "user", "login"))
    )


def _action_priority_reason(action_type: str, candidate: Mapping[str, Any], login_flow: Mapping[str, Any]) -> str:
    if action_type == "fill_input" and candidate.get("is_password") and not (login_flow.get("password_filled") or login_flow.get("password_input_filled")):
        return "login_flow_password_required"
    if action_type == "fill_input" and _is_email_text_candidate(candidate) and not (login_flow.get("username_or_email_filled") or login_flow.get("email_input_filled")):
        return "login_flow_email_required"
    if action_type == "click_submit" and candidate.get("is_submit") and not (login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked")):
        return "login_flow_submit_required"
    if action_type in {"inspect_network", "inspect_console", "inspect_dom", "check_login_state"} and (login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked")) and not (login_flow.get("login_result_checked") or login_flow.get("submit_result_checked")):
        return "login_flow_result_verification_required"
    return ""


def _is_input_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_visible_enabled(candidate):
        return False
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
    if bool(candidate.get("fillable")):
        return input_type not in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"}
    if input_type in {"hidden", "button", "submit", "checkbox", "radio", "reset", "image", "file"}:
        return False
    if tag in {"input", "textarea"}:
        return True
    return role in {"textbox", "searchbox"}


def _is_visible_enabled(candidate: Mapping[str, Any]) -> bool:
    visibility = _safe_visibility(candidate)
    if not bool(candidate.get("visible", visibility > 0.0)):
        return False
    if not bool(candidate.get("enabled", True)):
        return False
    if visibility <= 0.0:
        return False
    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            width = float(bbox[2] or 0.0)
            height = float(bbox[3] or 0.0)
        except (TypeError, ValueError):
            return False
        if width <= 0.0 or height <= 0.0:
            source = str(candidate.get("source") or "")
            if source not in {"playwright_dom", "dom"} and not str(candidate.get("selector") or candidate.get("locator") or ""):
                return False
    return True


def _login_action_allowed(action_type: str, *, login_credentials_available: bool, logged_in: bool) -> bool:
    if action_type in {"login_as_user", "login_as_admin"}:
        return bool(login_credentials_available and not logged_in)
    if action_type == "logout":
        return bool(logged_in)
    if action_type == "check_login_state":
        return bool(login_credentials_available or logged_in)
    return False


def _api_probe_available(runtime_signals: Mapping[str, Any]) -> bool:
    if bool(runtime_signals.get("api_json_page") or runtime_signals.get("has_json_response")):
        return True
    page_type = str(runtime_signals.get("page_type") or "")
    if page_type in {"api_json_page", "data_endpoint_page"}:
        return True
    if bool(runtime_signals.get("api_probe_endpoint_available")):
        return True
    entries = runtime_signals.get("network_entries")
    if isinstance(entries, list):
        return any(_is_api_like_entry(entry) for entry in entries if isinstance(entry, Mapping))
    return False


def _is_api_like_entry(entry: Mapping[str, Any]) -> bool:
    url = str(entry.get("url") or entry.get("name") or entry.get("endpoint") or "").lower()
    resource_type = str(entry.get("resource_type") or entry.get("initiatorType") or "").lower()
    return bool("/api/" in url or url.endswith(".json") or resource_type in {"xhr", "fetch"})


def _relational_db_policy_state(runtime_signals: Mapping[str, Any], infra_signals: Mapping[str, Any]) -> Dict[str, Any]:
    merged = {}
    if isinstance(runtime_signals, Mapping):
        merged.update(runtime_signals)
    if isinstance(infra_signals, Mapping):
        merged.update(infra_signals)
    db_engine = str(merged.get("db_engine") or "").lower()
    return {
        "postgres_available": bool(merged.get("postgres_hook_available") or db_engine == "postgres" or db_engine == "postgresql"),
        "mysql_available": bool(merged.get("mysql_hook_available") or db_engine == "mysql"),
        "db_hook_available": bool(merged.get("relational_db_hook_available") or merged.get("db_hook_enabled") or merged.get("postgres_hook_available") or merged.get("mysql_hook_available")),
        "snapshot_available": bool(merged.get("db_snapshot_available") or merged.get("db_snapshot_before_available")),
        "invariant_available": bool(merged.get("db_invariant_available") or merged.get("domain_invariant_available")),
    }


def _trace_policy_state(runtime_signals: Mapping[str, Any], infra_signals: Mapping[str, Any]) -> Dict[str, Any]:
    merged = {}
    if isinstance(runtime_signals, Mapping):
        merged.update(runtime_signals)
    if isinstance(infra_signals, Mapping):
        merged.update(infra_signals)
    has_trace_id = bool(merged.get("trace_id") or merged.get("last_trace_id") or merged.get("request_id") or merged.get("last_request_id"))
    return {
        "trace_collector_available": bool(merged.get("trace_collector_available") or merged.get("trace_available") or has_trace_id),
        "trace_id_available": has_trace_id,
        "api_request_available": _api_probe_available(runtime_signals),
        "db_hook_available": bool(merged.get("db_hook_enabled") or merged.get("relational_db_hook_available")),
    }


def _api_contract_policy_state(runtime_signals: Mapping[str, Any]) -> Dict[str, Any]:
    page_type = str(runtime_signals.get("page_type") or "")
    api_page = page_type in {"api_json_page", "data_endpoint_page"} or bool(runtime_signals.get("has_json_response"))
    return {
        "contract_available": bool(api_page or runtime_signals.get("api_contract_available") or runtime_signals.get("schema_available")),
        "openapi_available": bool(runtime_signals.get("openapi_spec_available")),
        "api_response_available": bool(api_page or runtime_signals.get("api_probe_endpoint_available") or runtime_signals.get("api_probe_status") is not None or _api_probe_available(runtime_signals)),
    }


def _login_scenario_policy_state(runtime_signals: Mapping[str, Any]) -> Dict[str, Any]:
    roles = runtime_signals.get("available_roles")
    return {
        "credential_available": bool(runtime_signals.get("login_credentials_available") or runtime_signals.get("credential_available")),
        "logged_in": bool(runtime_signals.get("logged_in") or runtime_signals.get("login_success")),
        "roles_available": isinstance(roles, list) and bool(roles),
        "session_expiration_available": bool(runtime_signals.get("session_expiration_available")),
        "protected_resource_available": bool(runtime_signals.get("protected_resource_available") or runtime_signals.get("role_restricted_pages")),
    }


def _relational_db_action_allowed(action_type: str, state: Mapping[str, Any]) -> bool:
    if action_type.startswith("take_postgres"):
        return bool(state.get("postgres_available"))
    if action_type.startswith("take_mysql"):
        return bool(state.get("mysql_available"))
    if action_type == "compare_relational_db_state":
        return bool(state.get("db_hook_available") and state.get("snapshot_available"))
    if action_type in {"check_inventory_invariant", "check_payment_order_consistency"}:
        return bool(state.get("db_hook_available") and state.get("invariant_available"))
    return bool(state.get("db_hook_available"))


def _trace_action_allowed(action_type: str, state: Mapping[str, Any]) -> bool:
    if not bool(state.get("trace_collector_available")):
        return False
    if action_type in {"capture_trace_id_after_action", "capture_request_id_after_action", "inspect_trace_span", "inspect_error_span", "calculate_correlation_confidence"}:
        return True
    if action_type in {"match_request_with_server_trace", "match_api_response_with_backend_exception", "correlate_ui_action_api_log_db"}:
        return bool(state.get("trace_id_available") and state.get("api_request_available"))
    if action_type == "match_db_change_with_trace":
        return bool(state.get("trace_id_available") and state.get("db_hook_available"))
    return bool(state.get("trace_id_available"))


def _api_contract_action_allowed(action_type: str, state: Mapping[str, Any]) -> bool:
    if action_type == "compare_openapi_schema":
        return bool(state.get("openapi_available") and state.get("api_response_available"))
    if action_type in _API_CONTRACT_ACTIONS:
        return bool(state.get("contract_available") and state.get("api_response_available"))
    return False


def _login_scenario_action_allowed(action_type: str, state: Mapping[str, Any]) -> bool:
    if not bool(state.get("credential_available")):
        return False
    if action_type in {"generate_login_scenario", "execute_login_scenario", "login_as_role", "verify_post_login_redirect"}:
        return True
    if action_type in {"logout_and_revisit", "verify_logout_access_block"}:
        return bool(state.get("logged_in"))
    if action_type in {"switch_role_and_revisit", "verify_role_based_navigation"}:
        return bool(state.get("roles_available"))
    if action_type == "expire_session_and_retry":
        return bool(state.get("session_expiration_available"))
    if action_type == "verify_session_expiration_behavior":
        return bool(state.get("session_expiration_available") and state.get("logged_in"))
    return bool(state.get("protected_resource_available"))


def _safe_visibility(candidate: Mapping[str, Any]) -> float:
    if "visibility" in candidate and candidate.get("visibility") is not None:
        try:
            value = float(candidate.get("visibility") or 0.0)
        except (TypeError, ValueError):
            value = 1.0
    elif "visible" in candidate:
        value = 1.0 if bool(candidate.get("visible")) else 0.0
    else:
        value = 1.0
    if value != value:
        return 1.0
    return max(0.0, min(1.0, value))
