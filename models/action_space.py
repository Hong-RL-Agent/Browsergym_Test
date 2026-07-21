"""Discrete J.A.W.S action space utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


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
                history = raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {}
                if int(history.get("step_index", 0) or 0) < 5:
                    continue
                if _has_unprocessed_form_candidates(candidates, history):
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
                    if self.is_click_action(action_type) and not _is_clickable_candidate(candidate):
                        continue
                    mask[self.encode(action_type, idx)] = 1.0
            else:
                mask[self.encode(action_type, 0)] = 1.0

        _apply_login_form_priority(mask, self, candidates, raw_observation.get("history", {}) if isinstance(raw_observation, Mapping) else {})

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
            "required_actions_remaining": list(login_flow.get("required_actions_remaining") or []),
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


def _has_unprocessed_form_candidates(candidates: Any, history: Mapping[str, Any]) -> bool:
    login_flow = history.get("login_flow", {}) if isinstance(history, Mapping) else {}
    if isinstance(login_flow, Mapping) and bool(login_flow.get("has_login_form")):
        if bool(login_flow.get("login_flow_completed")):
            return False
        remaining = login_flow.get("required_actions_remaining")
        if isinstance(remaining, list) and remaining:
            return True
        if bool(login_flow.get("submit_clicked")) and not bool(login_flow.get("submit_result_checked")):
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


def _apply_login_form_priority(mask: np.ndarray, action_space: ActionSpace, candidates: Any, history: Mapping[str, Any]) -> None:
    if not isinstance(candidates, list) or not isinstance(history, Mapping):
        return
    login_flow = history.get("login_flow", {})
    if not isinstance(login_flow, Mapping):
        login_flow = {}
    has_login_form = bool(login_flow.get("has_login_form")) or _candidate_login_form_present(candidates)
    if not has_login_form:
        return
    email_done = bool(login_flow.get("email_input_filled"))
    password_done = bool(login_flow.get("password_input_filled"))
    submit_done = bool(login_flow.get("submit_clicked"))
    result_checked = bool(login_flow.get("submit_result_checked"))

    preferred: list[int] = []
    if not password_done:
        preferred.extend(_candidate_action_ids(action_space, mask, candidates, "fill_input", _is_password_candidate))
    if not email_done:
        preferred.extend(_candidate_action_ids(action_space, mask, candidates, "fill_input", _is_email_text_candidate))
    if email_done and password_done and not submit_done:
        preferred.extend(_candidate_action_ids(action_space, mask, candidates, "click_submit", _is_submit_candidate))
    if submit_done and not result_checked:
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
    if action_type == "fill_input" and candidate.get("is_password") and not login_flow.get("password_input_filled"):
        return "login_flow_password_required"
    if action_type == "fill_input" and _is_email_text_candidate(candidate) and not login_flow.get("email_input_filled"):
        return "login_flow_email_required"
    if action_type == "click_submit" and candidate.get("is_submit") and not login_flow.get("submit_clicked"):
        return "login_flow_submit_required"
    if action_type in {"inspect_network", "inspect_console", "inspect_dom"} and login_flow.get("submit_clicked") and not login_flow.get("submit_result_checked"):
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
    return {
        "contract_available": bool(runtime_signals.get("api_contract_available") or runtime_signals.get("schema_available")),
        "openapi_available": bool(runtime_signals.get("openapi_spec_available")),
        "api_response_available": bool(runtime_signals.get("api_probe_endpoint_available") or runtime_signals.get("api_probe_status") is not None or _api_probe_available(runtime_signals)),
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
