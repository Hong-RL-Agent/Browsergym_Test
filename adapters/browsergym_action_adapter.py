"""J.A.W.S action to BrowserGym action adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class BrowserGymAction:
    action: Any
    executable: bool
    action_type: str
    fallback_action: Any = "noop()"


class BrowserGymActionAdapter:
    def __init__(self, noop_action: Any = "noop()") -> None:
        self.noop_action = noop_action

    def adapt(
        self,
        decoded_action: Mapping[str, Any],
        raw_observation: Optional[Mapping[str, Any]] = None,
    ) -> BrowserGymAction:
        action_type = str(decoded_action.get("action_type", "noop"))
        candidate_index = int(decoded_action.get("candidate_index", 0) or 0)

        if action_type in {
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
        }:
            return BrowserGymAction(action=None, executable=False, action_type=action_type)

        if action_type in {"click_element", "click_submit", "click_trigger_button", "click_retry_button", "click_recovery_button", "open_detail_panel"}:
            candidate = self._candidate_at(raw_observation, candidate_index)
            if not candidate:
                return BrowserGymAction(
                    action=None,
                    executable=False,
                    action_type=action_type,
                    fallback_action=self.noop_action,
                )
            return BrowserGymAction(
                action=self.to_browsergym_action("click_element", candidate),
                executable=True,
                action_type=action_type,
                fallback_action=self.noop_action,
            )

        if action_type in {"fill_input", "press_enter"}:
            candidate = self._candidate_at(raw_observation, candidate_index)
            if not candidate:
                return BrowserGymAction(None, False, action_type, self.noop_action)
            return BrowserGymAction(
                action=self.to_browsergym_action(action_type, candidate, decoded_action),
                executable=True,
                action_type=action_type,
                fallback_action=self.noop_action,
            )

        if action_type == "scroll_down":
            return BrowserGymAction("scroll(0, 500)", True, action_type, self.noop_action)

        if action_type == "scroll_up":
            return BrowserGymAction("scroll(0, -500)", True, action_type, self.noop_action)

        return BrowserGymAction(self.noop_action, True, "noop", self.noop_action)

    def to_browsergym_action(
        self,
        action_type: str,
        candidate: Mapping[str, Any],
        decoded_action: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        if action_type == "click_element":
            bid = _escape_action_string(str(candidate.get("bid", "")))
            if not bid:
                return self.noop_action
            return f"click('{bid}')"
        if action_type == "fill_input":
            bid = _escape_action_string(str(candidate.get("bid", "")))
            value = _escape_action_string(str((decoded_action or {}).get("input_text") or "test"))
            if not bid:
                return self.noop_action
            return f"fill('{bid}', '{value}')"
        if action_type == "press_enter":
            bid = _escape_action_string(str(candidate.get("bid", "")))
            if not bid:
                return self.noop_action
            return f"press('{bid}', 'Enter')"
        return self.noop_action

    @staticmethod
    def _candidate_at(raw_observation: Optional[Mapping[str, Any]], index: int) -> Optional[Mapping[str, Any]]:
        if not raw_observation:
            return None
        candidates = raw_observation.get("candidate_elements", [])
        if not isinstance(candidates, list) or not (0 <= index < len(candidates)):
            return None
        candidate = candidates[index]
        return candidate if isinstance(candidate, Mapping) else None


def _escape_action_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")
