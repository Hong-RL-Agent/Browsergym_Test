"""BrowserGym wrapper that exposes J.A.W.S raw observations and actions."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

from adapters.browsergym_action_adapter import BrowserGymActionAdapter
from adapters.browsergym_observation_adapter import BrowserGymObservationAdapter
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.infra_observation_service import collect_infra_observation
from services.site_profile_service import build_site_profile


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

_PLAYWRIGHT_CANDIDATE_SCRIPT = r"""
() => {
  const selectorFor = (el, index) => {
    const tag = (el.tagName || 'element').toLowerCase();
    const esc = (value) => String(value || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    for (const attr of ['id', 'name', 'data-testid', 'aria-label', 'placeholder', 'type', 'href']) {
      const value = el.getAttribute && el.getAttribute(attr);
      if (value) return `${tag}[${attr}="${esc(value)}"]`;
    }
    return `${tag}:nth-of-type(${index + 1})`;
  };
  const textOf = (el) => {
    return [
      el.innerText,
      el.textContent,
      el.getAttribute && el.getAttribute('aria-label'),
      el.getAttribute && el.getAttribute('placeholder'),
      el.getAttribute && el.getAttribute('title'),
      el.getAttribute && el.getAttribute('value'),
      el.getAttribute && el.getAttribute('name')
    ].filter(Boolean).join(' ').trim().replace(/\s+/g, ' ').slice(0, 240);
  };
  const nodes = Array.from(document.querySelectorAll([
    'input[type="text"]',
    'input[type="password"]',
    'input[name]',
    'input:not([type])',
    'textarea',
    'button',
    'button[type="submit"]',
    'input[type="submit"]',
    'a[href]',
    'form input',
    'form button',
    '[role="button"]',
    '[role="textbox"]',
    '[role="searchbox"]',
    '[role="link"]'
  ].join(',')));
  const seen = new Set();
  return nodes.map((el, index) => {
    const rect = el.getBoundingClientRect ? el.getBoundingClientRect() : {x: 0, y: 0, width: 0, height: 0};
    const style = window.getComputedStyle ? window.getComputedStyle(el) : {};
    const tag = (el.tagName || '').toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'button' || type === 'submit' ? 'button' : tag === 'input' || tag === 'textarea' ? 'textbox' : '')).toLowerCase();
    const selector = selectorFor(el, index);
    if (seen.has(selector)) return null;
    seen.add(selector);
    const visible = !!(rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none');
    const disabled = !!(el.disabled || el.getAttribute('aria-disabled') === 'true');
    const fillable = ['input', 'textarea'].includes(tag) && !['hidden', 'button', 'submit', 'checkbox', 'radio', 'reset', 'image', 'file'].includes(type);
    const isSubmit = type === 'submit' || (tag === 'button' && ((el.getAttribute('type') || 'submit').toLowerCase() === 'submit'));
    const clickable = tag === 'button' || tag === 'a' || role === 'button' || role === 'link' || isSubmit;
    return {
      candidate_id: `pw-${index}`,
      bid: `pw-${index}`,
      selector,
      locator: selector,
      tag,
      role,
      type,
      input_type: type,
      name: el.getAttribute('name') || textOf(el),
      text: textOf(el),
      placeholder: el.getAttribute('placeholder') || '',
      href: el.getAttribute('href') || '',
      visible,
      visibility: visible ? 1.0 : 0.0,
      enabled: !disabled,
      clickable,
      fillable,
      is_password: type === 'password',
      is_submit: isSubmit,
      is_form_control: !!el.closest('form') || fillable || isSubmit,
      bounding_box: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}
    };
  }).filter(Boolean);
}
"""


class BrowserGymJAWSEnv:
    def __init__(
        self,
        site_id: Optional[str] = None,
        base_url: str = "",
        max_steps: int = 20,
        max_candidates: int = 32,
        headless: bool = True,
        site_profile: Optional[Dict[str, Any]] = None,
        exploration_profile: str | Dict[str, Any] | None = None,
        requires_login: bool = False,
        login_config: Optional[Mapping[str, Any]] = None,
        page_timeout_ms: int = 15000,
        navigation_timeout_ms: int = 15000,
    ) -> None:
        self.site_id = site_id
        if not base_url:
            raise ValueError("BrowserGymJAWSEnv requires an explicit base_url.")
        self.base_url = base_url
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.headless = headless
        self.site_profile = dict(site_profile or build_site_profile(site_id or "", exploration_profile=exploration_profile))
        self.requires_login = bool(requires_login)
        self.login_config = dict(login_config or {})
        self.page_timeout_ms = int(page_timeout_ms or 15000)
        self.navigation_timeout_ms = int(navigation_timeout_ms or 15000)

        self.observation_adapter = BrowserGymObservationAdapter(max_candidates=max_candidates, site_profile=self.site_profile)
        self.action_adapter = BrowserGymActionAdapter()
        self.action_space = ActionSpace(max_candidates=max_candidates)
        self.observation_encoder = ObservationEncoder(max_candidates=max_candidates)

        self.env = None
        self.step_index = 0
        self.no_change_steps = 0
        self.previous_action_type = "noop"
        self._browser_obs: Optional[Dict[str, Any]] = None
        self._jaws_obs: Optional[Dict[str, Any]] = None
        self._playwright_events: list[Dict[str, Any]] = []
        self._playwright_listener_page: Any = None
        self._playwright_listener_flags: Dict[str, bool] = {
            "playwright_console_listener_enabled": False,
            "playwright_pageerror_listener_enabled": False,
            "playwright_network_listener_enabled": False,
            "playwright_request_listener_enabled": False,
            "playwright_response_listener_enabled": False,
            "playwright_requestfailed_listener_enabled": False,
            "signal_collector_enabled": True,
        }
        self._playwright_listener_warning = ""
        self._db_snapshot_before: Optional[Dict[str, Any]] = None
        self._db_snapshot_after: Optional[Dict[str, Any]] = None

    def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        gym = _import_browsergym()
        if self.env is not None:
            self.close()

        self.env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": self.base_url},
            wait_for_user_message=False,
            headless=self.headless,
        )
        browser_obs, info = self.env.reset()
        self._apply_page_timeouts()
        self._install_playwright_listeners()
        login_info: Dict[str, Any] = {
            "login_required": self.requires_login,
            "login_attempted": False,
            "login_success": False,
            "post_login_url": str(browser_obs.get("url") or "") if isinstance(browser_obs, dict) else "",
        }
        if self.requires_login and self.login_config:
            login_info = self._perform_login(browser_obs)
            try:
                browser_obs, _, _, _, refresh_info = _normalize_step_result(self.env.step("noop()"))
                merged_refresh = dict(refresh_info or {})
                merged_refresh.update(login_info)
                info = dict(info or {})
                info.update(merged_refresh)
            except Exception as exc:
                login_info["login_warning"] = f"login refresh failed: {exc}"
                info = dict(info or {})
                info.update(login_info)
        else:
            info = dict(info or {})
            info.update(login_info)
        self.step_index = 0
        self.no_change_steps = 0
        self.previous_action_type = "noop"
        browser_obs = self._with_playwright_candidates(browser_obs)
        self._browser_obs = browser_obs
        self._jaws_obs = self.observation_adapter.convert(
            browser_obs,
            info,
            previous_obs=None,
            history=self._history(),
        )
        self._stamp_site_id(self._jaws_obs)
        self._stamp_login_info(self._jaws_obs, info)
        self._stamp_infra_info(self._jaws_obs, info)
        self._stamp_playwright_signals(self._jaws_obs, self._playwright_signal_counts(0))
        self._stamp_api_probe_info(self._jaws_obs, info)
        self._stamp_extended_policy_info(self._jaws_obs, info)
        return self._jaws_obs, self._info(info, self.previous_action_type)

    def step(self, action_id: int) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.env is None or self._jaws_obs is None:
            raise RuntimeError("Call reset() before step().")

        decoded_action = self.action_space.decode(action_id)
        browser_action = self.action_adapter.adapt(decoded_action, self._jaws_obs)
        event_start_index = len(self._playwright_events)
        action_error = False
        done = False
        info: Dict[str, Any] = {}
        browser_obs = self._browser_obs

        if browser_action.action_type == "finish_episode":
            done = True
            info["action_success"] = True
            info["success_reason"] = "episode finished by policy"
        elif browser_action.action_type == "inspect_network":
            info = self._inspect_network()
        elif browser_action.action_type == "inspect_console":
            info = self._inspect_console()
        elif browser_action.action_type in {"login_as_user", "login_as_admin"}:
            info = self._perform_login(browser_obs, role="admin" if browser_action.action_type == "login_as_admin" else "user")
            try:
                browser_obs, _, terminated, truncated, refresh_info = _normalize_step_result(self.env.step("noop()"))
                done = bool(terminated or truncated)
                info.update(dict(refresh_info or {}))
            except Exception as exc:
                info["login_refresh_error"] = str(exc)
        elif browser_action.action_type == "logout":
            info = self._logout()
            try:
                browser_obs, _, terminated, truncated, refresh_info = _normalize_step_result(self.env.step("noop()"))
                done = bool(terminated or truncated)
                info.update(dict(refresh_info or {}))
            except Exception as exc:
                info["logout_refresh_error"] = str(exc)
        elif browser_action.action_type == "check_login_state":
            info = self._check_login_state()
        elif browser_action.action_type == "inspect_last_api_response":
            info = self._inspect_last_api_response()
        elif browser_action.action_type in {"replay_last_api_request", "call_collected_endpoint"}:
            info = self._replay_last_api_request(browser_action.action_type)
        elif browser_action.action_type == "check_api_ui_match":
            info = self._check_api_ui_match(previous_observation=self._jaws_obs)
        elif browser_action.action_type in _RELATIONAL_DB_HOOK_ACTIONS:
            info = self._inspect_relational_db_hook(browser_action.action_type)
        elif browser_action.action_type in _TRACE_CORRELATION_ACTIONS:
            info = self._inspect_trace_correlation(browser_action.action_type)
        elif browser_action.action_type in _API_CONTRACT_ACTIONS:
            info = self._inspect_api_contract(browser_action.action_type)
        elif browser_action.action_type in _LOGIN_SCENARIO_ACTIONS:
            info = self._inspect_login_scenario(browser_action.action_type)
        elif browser_action.action_type in {
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
        }:
            info = self._inspect_infra(browser_action.action_type)
        elif browser_action.action_type in {"click_element", "click_submit"}:
            info = self._click_candidate(decoded_action)
            info["internal_action"] = browser_action.action_type
            if not info.get("clicked"):
                action_error = True
            try:
                browser_obs, _, terminated, truncated, refresh_info = _normalize_step_result(self.env.step("noop()"))
                done = bool(terminated or truncated)
                info.update(dict(refresh_info or {}))
            except Exception as exc:
                info["click_refresh_error"] = str(exc)
        elif browser_action.action_type == "fill_input":
            info = self._fill_input(decoded_action)
        elif browser_action.action_type == "press_enter":
            info = self._press_enter(decoded_action)
        elif browser_action.action_type == "change_viewport_mobile":
            info = self._set_viewport(375, 812, browser_action.action_type)
            if not info.get("action_success"):
                action_error = True
            try:
                browser_obs, _, terminated, truncated, refresh_info = _normalize_step_result(self.env.step("noop()"))
                done = bool(terminated or truncated)
                info["refresh_info"] = refresh_info
            except Exception as exc:
                info["viewport_refresh_error"] = str(exc)
        elif browser_action.action_type == "change_viewport_desktop":
            info = self._set_viewport(1440, 900, browser_action.action_type)
            if not info.get("action_success"):
                action_error = True
            try:
                browser_obs, _, terminated, truncated, refresh_info = _normalize_step_result(self.env.step("noop()"))
                done = bool(terminated or truncated)
                info["refresh_info"] = refresh_info
            except Exception as exc:
                info["viewport_refresh_error"] = str(exc)
        elif browser_action.executable:
            try:
                browser_obs, _, terminated, truncated, info = _normalize_step_result(
                    self.env.step(browser_action.action)
                )
                self._apply_page_timeouts()
                done = bool(terminated or truncated)
            except Exception as exc:
                action_error = True
                info = {
                    "action_error": str(exc),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "failure_reason": "browsergym action failed",
                    "retryable": True,
                    "action_success": False,
                }
                if browser_action.action != browser_action.fallback_action:
                    try:
                        browser_obs, _, terminated, truncated, fallback_info = _normalize_step_result(
                            self.env.step(browser_action.fallback_action)
                        )
                        done = bool(terminated or truncated)
                        info["fallback_info"] = fallback_info
                    except Exception as fallback_exc:
                        info["fallback_error"] = str(fallback_exc)
        else:
            info = {"internal_action": browser_action.action_type}
            if browser_action.action_type in {"click_element", "click_submit", "fill_input", "press_enter"}:
                action_error = True
                info["action_success"] = False
                info["invalid_action_reason"] = "target candidate not found"
                info["failure_reason"] = "target candidate not found"
            elif browser_action.action_type in {"inspect_dom", "inspect_layout"}:
                info["action_success"] = True
                info["success_reason"] = f"{browser_action.action_type} completed from current observation"
            if browser_action.action_type == "inspect_cart" and self._jaws_obs:
                page_state = self._jaws_obs.get("page_state", {})
                info.update(
                    {
                        "cart_count": page_state.get("cart_count"),
                        "cart_text": page_state.get("cart_text"),
                        "cart_count_detected": page_state.get("cart_count_detected"),
                    }
                )

        self.step_index += 1
        if _action_count_limit_reached(self.step_index, self.max_steps):
            done = True

        previous_jaws_obs = self._jaws_obs
        playwright_signal_info = self._playwright_signal_counts(event_start_index)
        info.update(playwright_signal_info)
        browser_obs = self._with_playwright_candidates(browser_obs)
        next_jaws_obs = self.observation_adapter.convert(
            browser_obs,
            info,
            previous_obs=previous_jaws_obs,
            history=self._next_history(browser_action.action_type, previous_jaws_obs, browser_obs),
        )
        self._stamp_site_id(next_jaws_obs)
        self._stamp_infra_info(next_jaws_obs, info)
        self._stamp_playwright_signals(next_jaws_obs, playwright_signal_info)
        self._stamp_api_probe_info(next_jaws_obs, info)
        self._stamp_extended_policy_info(next_jaws_obs, info)
        self._browser_obs = browser_obs
        self._jaws_obs = next_jaws_obs
        self.previous_action_type = browser_action.action_type

        step_info = self._info(info, browser_action.action_type)
        self._stamp_login_info(next_jaws_obs, step_info)
        self._stamp_api_probe_info(next_jaws_obs, step_info)
        self._stamp_extended_policy_info(next_jaws_obs, step_info)
        step_info.update(
            {
                "action": decoded_action,
                "browsergym_action": browser_action.action,
                "last_action_error": action_error or bool(next_jaws_obs["runtime_signals"]["last_action_error"]),
                "candidate_count": len(next_jaws_obs.get("candidate_elements", [])),
                "cart_count": next_jaws_obs.get("page_state", {}).get("cart_count"),
                "cart_text": next_jaws_obs.get("page_state", {}).get("cart_text"),
                "cart_count_detected": next_jaws_obs.get("page_state", {}).get("cart_count_detected"),
                "action_result": {
                    "failed": action_error,
                    "viewport_changed": bool(info.get("viewport_changed")),
                    "viewport_width": info.get("viewport_width"),
                    "viewport_height": info.get("viewport_height"),
                },
                "infra_signals": next_jaws_obs.get("infra_signals", {}),
                **playwright_signal_info,
            }
        )
        return next_jaws_obs, 0.0, done, step_info

    def _apply_page_timeouts(self) -> None:
        page = _active_page(self.env)
        if page is None:
            return
        try:
            page.set_default_timeout(self.page_timeout_ms)
            page.set_default_navigation_timeout(self.navigation_timeout_ms)
        except Exception:
            return

    def _install_playwright_listeners(self) -> None:
        page = _active_page(self.env)
        if page is None:
            self._playwright_listener_warning = "active Playwright page not found"
            return
        if page is self._playwright_listener_page:
            return
        self._playwright_listener_page = page
        self._playwright_events = []
        self._playwright_listener_flags = {
            "playwright_console_listener_enabled": False,
            "playwright_pageerror_listener_enabled": False,
            "playwright_network_listener_enabled": False,
            "playwright_request_listener_enabled": False,
            "playwright_response_listener_enabled": False,
            "playwright_requestfailed_listener_enabled": False,
            "signal_collector_enabled": True,
        }
        self._playwright_listener_warning = ""
        try:
            page.on("console", self._on_playwright_console)
            self._playwright_listener_flags["playwright_console_listener_enabled"] = True
        except Exception as exc:
            self._append_listener_warning(f"console listener failed: {exc}")
        try:
            page.on("pageerror", self._on_playwright_pageerror)
            self._playwright_listener_flags["playwright_pageerror_listener_enabled"] = True
        except Exception as exc:
            self._append_listener_warning(f"pageerror listener failed: {exc}")
        network_ok = False
        for event_name, handler in (
            ("request", self._on_playwright_request),
            ("response", self._on_playwright_response),
            ("requestfailed", self._on_playwright_requestfailed),
        ):
            try:
                page.on(event_name, handler)
                self._playwright_listener_flags[f"playwright_{event_name}_listener_enabled"] = True
                network_ok = True
            except Exception as exc:
                self._append_listener_warning(f"{event_name} listener failed: {exc}")
                continue
        self._playwright_listener_flags["playwright_network_listener_enabled"] = network_ok

    def _append_listener_warning(self, message: str) -> None:
        if not self._playwright_listener_warning:
            self._playwright_listener_warning = message
        elif message not in self._playwright_listener_warning:
            self._playwright_listener_warning += f"; {message}"

    def _on_playwright_console(self, message: Any) -> None:
        message_type = _safe_attr(message, "type", "")
        text = _safe_attr(message, "text", "")
        self._playwright_events.append(
            {
                "event": "console",
                "event_type": "console",
                "type": str(message_type or ""),
                "message": str(text or ""),
                "timestamp": _monotonic_timestamp(),
                "step_index": self.step_index,
                "action_signature": self.previous_action_type,
            }
        )

    def _on_playwright_pageerror(self, error: Any) -> None:
        self._playwright_events.append(
            {
                "event": "pageerror",
                "event_type": "pageerror",
                "type": error.__class__.__name__,
                "message": str(error or ""),
                "timestamp": _monotonic_timestamp(),
                "step_index": self.step_index,
                "action_signature": self.previous_action_type,
            }
        )

    def _on_playwright_request(self, request: Any) -> None:
        self._playwright_events.append(
            {
                "event": "request",
                "event_type": "request",
                "url": str(_safe_attr(request, "url", "") or ""),
                "method": str(_safe_attr(request, "method", "") or ""),
                "resource_type": str(_safe_attr(request, "resource_type", "") or ""),
                "timestamp": _monotonic_timestamp(),
                "step_index": self.step_index,
                "action_signature": self.previous_action_type,
            }
        )

    def _on_playwright_response(self, response: Any) -> None:
        request = _safe_attr(response, "request", None)
        self._playwright_events.append(
            {
                "event": "response",
                "event_type": "response",
                "url": str(_safe_attr(response, "url", "") or ""),
                "method": str(_safe_attr(request, "method", "") or ""),
                "status": int(_safe_attr(response, "status", 0) or 0),
                "resource_type": str(_safe_attr(request, "resource_type", "") or ""),
                "timestamp": _monotonic_timestamp(),
                "step_index": self.step_index,
                "action_signature": self.previous_action_type,
            }
        )

    def _on_playwright_requestfailed(self, request: Any) -> None:
        failure = _safe_attr(request, "failure", None)
        error_text = ""
        if callable(failure):
            try:
                failure = failure()
            except Exception:
                failure = None
        if isinstance(failure, dict):
            error_text = str(failure.get("errorText") or failure.get("error_text") or "")
        self._playwright_events.append(
            {
                "event": "requestfailed",
                "event_type": "requestfailed",
                "url": str(_safe_attr(request, "url", "") or ""),
                "method": str(_safe_attr(request, "method", "") or ""),
                "resource_type": str(_safe_attr(request, "resource_type", "") or ""),
                "message": error_text,
                "failure_text": error_text,
                "timestamp": _monotonic_timestamp(),
                "step_index": self.step_index,
                "action_signature": self.previous_action_type,
            }
        )

    def _playwright_signal_counts(self, start_index: int = 0) -> Dict[str, Any]:
        events = self._playwright_events[max(0, int(start_index or 0)) :]
        entries = [
            event for event in events if event.get("event") in {"request", "response", "requestfailed"}
        ]
        counts: Dict[str, Any] = {
            **self._playwright_listener_flags,
            "playwright_listener_warning": self._playwright_listener_warning,
            "playwright_event_count": len(events),
            "network_entries": entries[-50:],
            "console_error_count": 0,
            "console_warning_count": 0,
            "page_error_count": 0,
            "runtime_exception_count": 0,
            "unhandled_rejection_count": 0,
            "network_request_failed_count": 0,
            "api_4xx_count": 0,
            "api_5xx_count": 0,
            "api_timeout_count": 0,
            "static_asset_failure_count": 0,
            "auth_permission_anomaly_count": 0,
            "sensitive_data_exposure_signal_count": 0,
            "token_exposure_signal_count": 0,
        }
        for event in events:
            message = str(event.get("message") or "").lower()
            event_type = str(event.get("type") or "").lower()
            url = str(event.get("url") or "").lower()
            if event.get("event") == "console":
                if event_type == "error" or any(token in message for token in ("error", "exception", "failed")):
                    counts["console_error_count"] += 1
                elif event_type in {"warning", "warn"}:
                    counts["console_warning_count"] += 1
                if "unhandled" in message and "rejection" in message:
                    counts["unhandled_rejection_count"] += 1
            if event.get("event") == "pageerror":
                counts["page_error_count"] += 1
                counts["runtime_exception_count"] += 1
            if event.get("event") == "requestfailed":
                counts["network_request_failed_count"] += 1
                if "timeout" in message:
                    counts["api_timeout_count"] += 1
            status = int(event.get("status") or 0)
            if 400 <= status < 500:
                counts["api_4xx_count"] += 1
                if status in {401, 403}:
                    counts["auth_permission_anomaly_count"] += 1
            if status >= 500:
                counts["api_5xx_count"] += 1
            if any(url.endswith(suffix) for suffix in (".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".woff", ".woff2")) and (
                event.get("event") == "requestfailed" or status >= 400
            ):
                counts["static_asset_failure_count"] += 1
            if any(token in message for token in ("token", "bearer ", "authorization")):
                counts["token_exposure_signal_count"] += 1
            if any(token in message for token in ("password", "secret")):
                counts["sensitive_data_exposure_signal_count"] += 1
        for field in (
            "console_error_count",
            "console_warning_count",
            "page_error_count",
            "runtime_exception_count",
            "unhandled_rejection_count",
            "network_request_failed_count",
            "api_4xx_count",
            "api_5xx_count",
            "api_timeout_count",
            "static_asset_failure_count",
            "auth_permission_anomaly_count",
            "sensitive_data_exposure_signal_count",
            "token_exposure_signal_count",
        ):
            counts[f"delta_{field}"] = counts[field]
        return counts

    def _stamp_playwright_signals(self, observation: Dict[str, Any], signal_info: Mapping[str, Any]) -> None:
        runtime = observation.setdefault("runtime_signals", {})
        for key, value in signal_info.items():
            if key == "network_entries":
                runtime[key] = value
            elif key == "playwright_listener_warning":
                runtime[key] = str(value or "")
            elif isinstance(value, (bool, int, float, str)):
                runtime[key] = value

    def _stamp_api_probe_info(self, observation: Dict[str, Any], info: Mapping[str, Any]) -> None:
        runtime = observation.setdefault("runtime_signals", {})
        last_entry = self._last_api_entry()
        runtime["api_probe_endpoint_available"] = bool(info.get("api_probe_endpoint_available", bool(last_entry)))
        runtime["api_probe_action"] = bool(info.get("api_probe_action", False))
        runtime["api_probe_checked"] = bool(info.get("api_probe_checked", False))
        runtime["api_probe_replayed"] = bool(info.get("api_probe_replayed", False))
        runtime["api_ui_match_checked"] = bool(info.get("api_ui_match_checked", False))
        if info.get("api_probe_status") is not None:
            runtime["api_probe_status"] = int(info.get("api_probe_status") or 0)
        if info.get("api_probe_url"):
            runtime["api_probe_url"] = str(info.get("api_probe_url") or "")
        if info.get("api_probe_method"):
            runtime["api_probe_method"] = str(info.get("api_probe_method") or "")
        for key in (
            "api_probe_ok",
            "ui_changed_after_api",
            "expect_ui_change_after_api",
            "api_probe_skipped",
        ):
            if key in info:
                runtime[key] = bool(info.get(key))
        if info.get("api_probe_warning"):
            runtime["api_probe_warning"] = str(info.get("api_probe_warning") or "")
        if last_entry:
            runtime["last_api_entry"] = _api_entry_summary(last_entry)

    def _stamp_extended_policy_info(self, observation: Dict[str, Any], info: Mapping[str, Any]) -> None:
        runtime = observation.setdefault("runtime_signals", {})
        infra = observation.setdefault("infra_signals", {})
        profile_defaults = {
            "db_engine": self.site_profile.get("db_engine"),
            "relational_db_hook_available": bool(
                self.site_profile.get("relational_db_hook_available")
                or self.site_profile.get("postgres_hook_available")
                or self.site_profile.get("mysql_hook_available")
                or self.site_profile.get("database_paths")
            ),
            "postgres_hook_available": bool(self.site_profile.get("postgres_hook_available") or str(self.site_profile.get("db_engine") or "").lower() in {"postgres", "postgresql"}),
            "mysql_hook_available": bool(self.site_profile.get("mysql_hook_available") or str(self.site_profile.get("db_engine") or "").lower() == "mysql"),
            "db_invariant_available": bool(self.site_profile.get("db_invariant_available") or self.site_profile.get("domain_invariants")),
            "trace_collector_available": bool(self.site_profile.get("trace_collector_available")),
            "api_contract_available": bool(self.site_profile.get("api_contract_available") or self.site_profile.get("api_contract")),
            "schema_available": bool(self.site_profile.get("schema_available") or self.site_profile.get("api_contract")),
            "openapi_spec_available": bool(self.site_profile.get("openapi_spec_available") or self.site_profile.get("openapi_spec_path")),
            "available_roles": self.login_config.get("roles", []) if isinstance(self.login_config.get("roles"), list) else [],
            "credential_available": bool(self.login_config),
            "auth_scenario_available": bool(self.login_config),
            "protected_resource_available": bool(self.login_config.get("protected_pages")) if isinstance(self.login_config, Mapping) else False,
            "session_expiration_available": bool(self.login_config.get("session_expiration_available")) if isinstance(self.login_config, Mapping) else False,
        }
        for key, value in profile_defaults.items():
            if value not in (None, "", [], {}):
                runtime.setdefault(key, value)
                infra.setdefault(key, value)
        for key in (
            "db_engine",
            "relational_db_hook_available",
            "postgres_hook_available",
            "mysql_hook_available",
            "db_snapshot_available",
            "db_snapshot_before_available",
            "db_snapshot_after_available",
            "db_state_changed",
            "changed_tables",
            "changed_rows",
            "transaction_status",
            "foreign_key_violation",
            "constraint_violation",
            "rollback_detected",
            "partial_commit_detected",
            "deadlock_detected",
            "lock_timeout_detected",
            "db_invariant_violation",
            "trace_collector_available",
            "trace_available",
            "trace_id",
            "span_id",
            "request_id",
            "last_request_id",
            "server_exception_in_same_trace",
            "db_query_in_same_trace",
            "trace_error_count",
            "correlation_confidence",
            "api_contract_available",
            "schema_available",
            "openapi_spec_available",
            "schema_valid",
            "missing_required_fields",
            "type_mismatch_fields",
            "unexpected_fields",
            "contract_violation_count",
            "error_response_format_valid",
            "ui_contract_mismatch",
            "available_roles",
            "credential_available",
            "current_role",
            "visited_authenticated_pages",
            "role_restricted_pages",
            "auth_scenario_available",
            "login_redirect_mismatch",
            "logout_failed",
            "protected_page_access_after_logout",
            "role_based_access_violation",
            "session_expiration_failed",
            "sensitive_data_visible_after_switch",
            "unauthorized_authenticated_access",
        ):
            if key in info:
                runtime[key] = info[key]
                infra.setdefault(key, info[key])

    def _set_viewport(self, width: int, height: int, action_type: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "viewport_changed": False,
            "action_success": False,
            "attempted_viewport_width": width,
            "attempted_viewport_height": height,
            "viewport_change_supported": True,
        }
        page = _active_page(self.env)
        if page is None:
            info["viewport_error"] = "active page not found"
            info["failure_reason"] = "active page not found"
            info["exception_type"] = "NoActivePage"
            info["exception_message"] = "active page not found"
            info["retryable"] = True
            return info
        try:
            current_viewport = getattr(page, "viewport_size", None) or {}
            if isinstance(current_viewport, Mapping):
                info["current_viewport_width"] = current_viewport.get("width")
                info["current_viewport_height"] = current_viewport.get("height")
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(250)
            info.update(
                {
                    "viewport_changed": True,
                    "action_success": True,
                    "success_reason": "viewport changed",
                    "viewport_width": width,
                    "viewport_height": height,
                    "current_viewport_width": width,
                    "current_viewport_height": height,
                    "retryable": False,
                }
            )
        except Exception as exc:
            info["viewport_error"] = str(exc)
            info["failure_reason"] = "viewport change failed"
            info["exception_type"] = type(exc).__name__
            info["exception_message"] = str(exc)
            info["retryable"] = True
        return info

    def _perform_login(self, browser_obs: Optional[Dict[str, Any]], role: str = "user") -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "login_required": True,
            "login_attempted": True,
            "login_success": False,
            "login_role": role,
            "post_login_url": str(browser_obs.get("url") or "") if isinstance(browser_obs, dict) else "",
        }
        page = _active_page(self.env)
        if page is None:
            info["login_warning"] = "active page not found"
            return info
        role_config = self.login_config.get(role) if isinstance(self.login_config.get(role), Mapping) else {}
        email = str(role_config.get("email") or self.login_config.get("email") or self.login_config.get("username") or "")
        password = str(role_config.get("password") or self.login_config.get("password") or "")
        email_selectors = [str(item) for item in self.login_config.get("email_selector_candidates", []) or []]
        password_selectors = [str(item) for item in self.login_config.get("password_selector_candidates", []) or []]
        submit_selectors = [str(item) for item in self.login_config.get("submit_selector_candidates", []) or []]
        email_selector = self._first_visible_selector(page, email_selectors)
        password_selector = self._first_visible_selector(page, password_selectors)
        submit_selector = self._first_visible_selector(page, submit_selectors)
        if not email_selector:
            email_selector = self._fallback_email_selector(page)
        if not password_selector:
            password_selector = self._fallback_password_selector(page)
        if not submit_selector:
            submit_selector = self._fallback_submit_selector(page)
        if not email_selector or not password_selector or not submit_selector:
            info["login_warning"] = "login selectors not found"
            return info
        try:
            page.locator(email_selector).first.fill(email, timeout=2000)
            page.locator(password_selector).first.fill(password, timeout=2000)
            page.locator(submit_selector).first.click(timeout=2000)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=min(self.navigation_timeout_ms, 4000))
            except Exception:
                page.wait_for_timeout(800)
            current_url = page.url
            current_text = page.inner_text("body") if page else ""
            info["post_login_url"] = current_url
            info["login_success"] = self._login_success(current_url, current_text)
            info["logged_in"] = bool(info["login_success"])
            info["session_valid"] = bool(info["login_success"])
            info["user_role"] = role if info["login_success"] else ""
            if not info["login_success"]:
                info["login_warning"] = "still observing login page after login attempt"
        except Exception as exc:
            info["login_warning"] = str(exc)
        return info

    def _logout(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "logout", "logout_attempted": True, "logout_success": False}
        page = _active_page(self.env)
        if page is None:
            info["logout_warning"] = "active page not found"
            return info
        selectors = [str(item) for item in self.login_config.get("logout_selector_candidates", []) or []]
        selector = self._first_visible_selector(page, selectors) or self._first_visible_selector(
            page,
            ["button:has-text('Logout')", "button:has-text('Sign out')", "a:has-text('Logout')", "a:has-text('Sign out')"],
        )
        if not selector:
            info["logout_warning"] = "logout selector not found"
            return info
        try:
            page.locator(selector).first.click(timeout=2000)
            page.wait_for_timeout(500)
            text = page.inner_text("body") if page else ""
            info["logout_success"] = not self._login_success(page.url, text)
            info["logged_in"] = not bool(info["logout_success"])
            info["session_valid"] = not bool(info["logout_success"])
        except Exception as exc:
            info["logout_warning"] = str(exc)
        return info

    def _check_login_state(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "check_login_state", "login_state_checked": True}
        page = _active_page(self.env)
        if page is None:
            info["login_state_warning"] = "active page not found"
            return info
        try:
            text = page.inner_text("body")
            logged_in = self._login_success(page.url, text)
            info.update(
                {
                    "logged_in": bool(logged_in),
                    "session_valid": bool(logged_in),
                    "post_login_url": page.url if logged_in else "",
                    "login_success": bool(logged_in),
                }
            )
        except Exception as exc:
            info["login_state_warning"] = str(exc)
        return info

    def _first_visible_selector(self, page: Any, selectors: list[str]) -> str:
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if locator.count() > 0 and locator.is_visible(timeout=500):
                    return selector
            except Exception:
                continue
        return ""

    def _login_success(self, current_url: str, current_text: str) -> bool:
        lower_url = str(current_url or "").lower()
        lower_text = str(current_text or "").lower()
        for token in self.login_config.get("success_url_contains", []) or []:
            if str(token).lower() in lower_url:
                return True
        for token in self.login_config.get("success_text_contains", []) or []:
            if str(token).lower() in lower_text:
                return True
        return False

    def _fallback_email_selector(self, page: Any) -> str:
        for selector in ("input[type='email']", "input:not([type='password'])", "form input"):
            selected = self._first_visible_selector(page, [selector])
            if selected:
                return selected
        return ""

    def _fallback_password_selector(self, page: Any) -> str:
        return self._first_visible_selector(page, ["input[type='password']", "form input[type='password']"])

    def _fallback_submit_selector(self, page: Any) -> str:
        for selector in ("button:has-text('Sign in')", "button[type='submit']", "form button"):
            selected = self._first_visible_selector(page, [selector])
            if selected:
                return selected
        return ""

    def _inspect_network(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "internal_action": "inspect_network",
            "network_inspected": True,
            "network_capture_enabled": True,
            "network_log_available": False,
            "network_events_count": 0,
            "action_success": False,
        }
        page = _active_page(self.env)
        if page is None:
            info["network_error"] = "active page not found"
            info["failure_reason"] = "active page not found"
            info["exception_type"] = "ActivePageNotFound"
            info["retryable"] = True
            return info
        try:
            entries = page.evaluate(
                """() => performance.getEntriesByType('resource').slice(-50).map(e => ({
                    name: e.name || '',
                    initiatorType: e.initiatorType || '',
                    duration: e.duration || 0,
                    transferSize: e.transferSize || 0
                }))"""
            )
            info["network_entries"] = entries if isinstance(entries, list) else []
            info["network_events_count"] = len(info["network_entries"])
            info["network_log_available"] = bool(info["network_entries"])
            text = " ".join(str(item.get("name", "")) for item in info["network_entries"] if isinstance(item, dict)).lower()
            info["api_403_count"] = text.count("403") + text.count("forbidden")
            info["network_activity"] = bool(info["network_entries"])
            info["action_success"] = True
            info["success_reason"] = "network log inspected"
        except Exception as exc:
            info["network_error"] = str(exc)
            info["failure_reason"] = str(exc) or "network inspection failed"
            info["exception_type"] = exc.__class__.__name__
            info["exception_message"] = str(exc)
            info["playwright_error"] = str(exc)
            info["retryable"] = True
        return info

    def _inspect_last_api_response(self) -> Dict[str, Any]:
        entry = self._last_api_entry()
        info: Dict[str, Any] = {
            "internal_action": "inspect_last_api_response",
            "api_probe_action": True,
            "api_probe_checked": True,
            "api_probe_endpoint_available": bool(entry),
        }
        if entry:
            info.update(_api_entry_summary(entry))
            info["network_entries"] = [entry]
        return info

    def _replay_last_api_request(self, action_type: str) -> Dict[str, Any]:
        entry = self._last_api_entry()
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "api_probe_action": True,
            "api_probe_replayed": False,
            "api_probe_endpoint_available": bool(entry),
        }
        if not entry:
            info["api_probe_warning"] = "no collected API endpoint"
            return info
        method = str(entry.get("method") or "GET").upper()
        url = str(entry.get("url") or entry.get("name") or "")
        info.update(_api_entry_summary(entry))
        if method not in {"GET", "HEAD"}:
            info["api_probe_skipped"] = True
            info["api_probe_warning"] = f"unsafe replay method skipped: {method}"
            return info
        page = _active_page(self.env)
        if page is None:
            info["api_probe_warning"] = "active page not found"
            return info
        try:
            result = page.evaluate(
                """async ({url, method}) => {
                    const started = performance.now();
                    try {
                        const response = await fetch(url, {method, credentials: 'include'});
                        const text = await response.clone().text().catch(() => '');
                        return {
                            ok: response.ok,
                            status: response.status,
                            statusText: response.statusText,
                            url: response.url,
                            method,
                            duration: performance.now() - started,
                            bodySummary: text.slice(0, 500)
                        };
                    } catch (error) {
                        return {ok: false, status: 0, url, method, error: String(error), duration: performance.now() - started};
                    }
                }""",
                {"url": url, "method": method},
            )
            result = result if isinstance(result, dict) else {}
            info.update(
                {
                    "api_probe_replayed": True,
                    "api_probe_status": int(result.get("status") or 0),
                    "api_probe_ok": bool(result.get("ok")),
                    "api_probe_url": str(result.get("url") or url),
                    "api_probe_method": method,
                    "api_probe_duration_ms": float(result.get("duration") or 0.0),
                    "api_probe_body_summary": str(result.get("bodySummary") or result.get("error") or "")[:500],
                    "network_entries": [
                        {
                            "event": "response" if int(result.get("status") or 0) else "requestfailed",
                            "url": str(result.get("url") or url),
                            "method": method,
                            "status": int(result.get("status") or 0),
                            "message": str(result.get("statusText") or result.get("error") or ""),
                        }
                    ],
                }
            )
        except Exception as exc:
            info["api_probe_warning"] = str(exc)
        return info

    def _check_api_ui_match(self, previous_observation: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        entry = self._last_api_entry()
        current = self._jaws_obs or {}
        before_state = (previous_observation or {}).get("page_state", {}) if isinstance(previous_observation, Mapping) else {}
        current_state = current.get("page_state", {}) if isinstance(current, Mapping) else {}
        before_signature = (
            str(before_state.get("url") or ""),
            int(before_state.get("page_text_length") or 0),
            int(before_state.get("dom_node_count") or 0),
        )
        after_signature = (
            str(current_state.get("url") or ""),
            int(current_state.get("page_text_length") or 0),
            int(current_state.get("dom_node_count") or 0),
        )
        info: Dict[str, Any] = {
            "internal_action": "check_api_ui_match",
            "api_probe_action": True,
            "api_ui_match_checked": True,
            "api_probe_endpoint_available": bool(entry),
            "ui_changed_after_api": before_signature != after_signature,
        }
        if entry:
            info.update(_api_entry_summary(entry))
            status = int(entry.get("status") or 0)
            info["expect_ui_change_after_api"] = 200 <= status < 300
            info["network_entries"] = [entry]
        return info

    def _inspect_relational_db_hook(self, action_type: str) -> Dict[str, Any]:
        infra = self._collect_infra()
        db_engine = str(self.site_profile.get("db_engine") or infra.get("db_engine") or "").lower()
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "relational_db_hook_policy": True,
            "relational_db_hook_available": bool(infra.get("db_hook_enabled") or self.site_profile.get("relational_db_hook_available")),
            "db_engine": db_engine,
            "postgres_hook_available": bool(db_engine in {"postgres", "postgresql"} or self.site_profile.get("postgres_hook_available")),
            "mysql_hook_available": bool(db_engine == "mysql" or self.site_profile.get("mysql_hook_available")),
            "infra_signals": infra,
        }
        if action_type.endswith("snapshot_before"):
            self._db_snapshot_before = _db_snapshot(infra)
            info["db_snapshot_before_available"] = True
            info["db_snapshot_available"] = True
        elif action_type.endswith("snapshot_after"):
            self._db_snapshot_after = _db_snapshot(infra)
            info["db_snapshot_after_available"] = True
            info["db_snapshot_available"] = bool(self._db_snapshot_before)
        elif action_type == "compare_relational_db_state":
            before = self._db_snapshot_before or {}
            after = self._db_snapshot_after or _db_snapshot(infra)
            info.update(_compare_db_snapshots(before, after))
        elif action_type == "check_foreign_key_consistency":
            info["foreign_key_violation"] = bool(infra.get("foreign_key_violation"))
        elif action_type == "check_constraint_violation":
            info["constraint_violation"] = bool(infra.get("constraint_violation") or int(infra.get("db_integrity_error_count") or 0) > 0)
        elif action_type == "check_partial_commit":
            info["partial_commit_detected"] = bool(infra.get("partial_commit_detected"))
        elif action_type == "check_rollback_result":
            info["rollback_detected"] = bool(infra.get("rollback_detected") or infra.get("rollback_failed"))
        elif action_type == "check_deadlock_or_lock_timeout":
            text = str(infra.get("db_error_message") or infra.get("server_error_message") or "").lower()
            info["deadlock_detected"] = bool(infra.get("deadlock_detected") or "deadlock" in text)
            info["lock_timeout_detected"] = bool(infra.get("lock_timeout_detected") or "lock timeout" in text)
        elif action_type in {"check_inventory_invariant", "check_payment_order_consistency", "check_transaction_consistency"}:
            info["db_invariant_violation"] = bool(infra.get("db_invariant_violation") or int(infra.get("db_integrity_error_count") or 0) > 0)
            info["transaction_status"] = str(infra.get("transaction_status") or "")
        return info

    def _inspect_trace_correlation(self, action_type: str) -> Dict[str, Any]:
        entry = self._last_api_entry()
        trace_id = _trace_id_from_entry(entry or {}) or str(self.site_profile.get("trace_id") or "")
        infra = self._collect_infra()
        text = " ".join(str(item) for item in infra.get("server_log_tail_sample", []) if item).lower()
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "trace_correlation_policy": True,
            "trace_collector_available": bool(self.site_profile.get("trace_collector_available") or trace_id),
            "trace_available": bool(trace_id),
            "trace_id": trace_id,
            "last_request_id": trace_id,
            "related_endpoint": str((entry or {}).get("url") or ""),
            "related_http_method": str((entry or {}).get("method") or ""),
            "server_exception_in_same_trace": bool(trace_id and ("exception" in text or "traceback" in text or "error" in text)),
            "db_query_in_same_trace": bool(trace_id and int(infra.get("server_log_db_error_count") or 0) > 0),
            "trace_error_count": int(bool(trace_id and ("exception" in text or "error" in text))),
            "correlation_confidence": 0.85 if trace_id and entry else 0.0,
            "infra_signals": infra,
        }
        return info

    def _inspect_api_contract(self, action_type: str) -> Dict[str, Any]:
        entry = self._last_api_entry()
        contract = self.site_profile.get("api_contract") if isinstance(self.site_profile.get("api_contract"), Mapping) else {}
        required_fields = [str(item) for item in contract.get("required_fields", [])] if isinstance(contract, Mapping) else []
        body_summary = str((entry or {}).get("bodySummary") or (entry or {}).get("body") or "")
        missing = [field for field in required_fields if field and field not in body_summary]
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "api_contract_policy": True,
            "api_contract_available": bool(contract or self.site_profile.get("api_contract_available")),
            "schema_available": bool(contract or self.site_profile.get("schema_available")),
            "openapi_spec_available": bool(self.site_profile.get("openapi_spec_path") or self.site_profile.get("openapi_spec_available")),
            "endpoint": str((entry or {}).get("url") or ""),
            "method": str((entry or {}).get("method") or ""),
            "status_code": int((entry or {}).get("status") or 0),
            "missing_required_fields": missing,
            "type_mismatch_fields": [],
            "unexpected_fields": [],
            "contract_violation_count": len(missing),
            "schema_valid": not bool(missing),
            "error_response_format_valid": True,
            "ui_contract_mismatch": bool(self._jaws_obs and self._jaws_obs.get("runtime_signals", {}).get("api_ui_match_checked") and self._jaws_obs.get("runtime_signals", {}).get("ui_changed_after_api") is False),
        }
        if entry:
            info.update(_api_entry_summary(entry))
        return info

    def _inspect_login_scenario(self, action_type: str) -> Dict[str, Any]:
        runtime = self._jaws_obs.get("runtime_signals", {}) if self._jaws_obs else {}
        roles = self.login_config.get("roles") if isinstance(self.login_config.get("roles"), list) else []
        protected_pages = self.login_config.get("protected_pages") if isinstance(self.login_config.get("protected_pages"), list) else []
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "login_scenario_policy": True,
            "auth_scenario_available": bool(self.login_config),
            "credential_available": bool(self.login_config),
            "available_roles": roles,
            "current_role": str(runtime.get("user_role") or ""),
            "logged_in": bool(runtime.get("logged_in") or runtime.get("login_success")),
            "session_valid": bool(runtime.get("session_valid") or runtime.get("login_success")),
            "role_restricted_pages": protected_pages,
            "protected_resource_available": bool(protected_pages),
        }
        if action_type in {"logout_and_revisit", "verify_logout_access_block"}:
            logout_info = self._logout()
            info.update(logout_info)
            info["logout_failed"] = not bool(logout_info.get("logout_success"))
            info["protected_page_access_after_logout"] = bool(not logout_info.get("logout_success") and protected_pages)
        elif action_type in {"generate_login_scenario", "execute_login_scenario", "login_as_role", "verify_post_login_redirect"}:
            info["login_scenario_generated"] = True
            expected_redirect = str(self.login_config.get("expected_redirect_url") or "")
            info["expected_redirect_url"] = expected_redirect
            post_url = str(runtime.get("post_login_url") or "")
            info["login_redirect_mismatch"] = bool(expected_redirect and post_url and expected_redirect not in post_url)
        elif action_type in {"switch_role_and_revisit", "verify_role_based_navigation"}:
            info["role_based_access_violation"] = bool(runtime.get("role_based_access_violation"))
        elif action_type in {"expire_session_and_retry", "verify_session_expiration_behavior"}:
            info["session_expiration_failed"] = bool(runtime.get("session_valid") and runtime.get("session_expired"))
        return info

    def _last_api_entry(self) -> Optional[Dict[str, Any]]:
        for event in reversed(self._playwright_events):
            if not isinstance(event, Mapping):
                continue
            if _is_api_like_url(str(event.get("url") or event.get("name") or "")):
                return dict(event)
        if isinstance(self._jaws_obs, Mapping):
            runtime = self._jaws_obs.get("runtime_signals", {})
            if isinstance(runtime, Mapping):
                for entry in reversed(runtime.get("network_entries") if isinstance(runtime.get("network_entries"), list) else []):
                    if isinstance(entry, Mapping) and _is_api_like_url(str(entry.get("url") or entry.get("name") or "")):
                        return dict(entry)
        return None

    def _inspect_console(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "internal_action": "inspect_console",
            "console_inspected": True,
            "console_verification_performed": True,
            "action_success": False,
        }
        page = _active_page(self.env)
        if page is None:
            info["console_error"] = "active page not found"
            info["failure_reason"] = "active page not found"
            info["exception_type"] = "ActivePageNotFound"
            info["retryable"] = True
            return info
        try:
            info["console_error_count"] = page.evaluate(
                """() => {
                    const text = document.body ? document.body.innerText.toLowerCase() : '';
                    return (text.match(/error|forbidden|timeout|failed/g) || []).length;
                }"""
            )
            info["action_success"] = True
            info["success_reason"] = "console state inspected"
            info["console_verification_success"] = True
        except Exception as exc:
            info["console_error"] = str(exc)
            info["failure_reason"] = str(exc) or "console inspection failed"
            info["exception_type"] = exc.__class__.__name__
            info["exception_message"] = str(exc)
            info["playwright_error"] = str(exc)
            info["retryable"] = True
        return info

    def _inspect_infra(self, action_type: str) -> Dict[str, Any]:
        infra = self._collect_infra()
        return {
            "internal_action": action_type,
            "infra_inspected": True,
            "infra_action": action_type,
            "infra_signals": infra,
        }

    def _click_candidate(self, decoded_action: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "click_element", "clicked": False, "action_success": False}
        page = _active_page(self.env)
        candidate = self._candidate_for_decoded_action(decoded_action)
        action_type = str(decoded_action.get("action_type") or "click_element")
        if isinstance(candidate, Mapping):
            if action_type == "click_submit" and not bool(candidate.get("is_submit")):
                info["invalid_action_reason"] = "submit target candidate not found"
                info["failure_reason"] = "submit target candidate not found"
                info["retryable"] = False
                return info
            if action_type != "click_submit" and not bool(candidate.get("clickable")):
                info["invalid_action_reason"] = "clickable target candidate not found"
                info["failure_reason"] = "clickable target candidate not found"
                info["retryable"] = False
                return info
        locator = self._locator_for_candidate(page, candidate)
        if page is None or candidate is None or locator is None:
            info["invalid_action_reason"] = "target candidate not found"
            info["failure_reason"] = "target candidate not found"
            info["retryable"] = False
            return info
        try:
            before_url = str(getattr(page, "url", "") or "")
            locator.click(force=True, timeout=1500)
            page.wait_for_timeout(250)
            after_url = str(getattr(page, "url", "") or "")
            info.update(
                {
                    "clicked": True,
                    "action_success": True,
                    "success_reason": "target candidate clicked",
                    "clicked_element_key": candidate.get("element_key"),
                    "clicked_text": candidate.get("text") or candidate.get("name"),
                    "clicked_bid": candidate.get("bid"),
                    "new_url": after_url,
                    "url_changed": bool(before_url and after_url and before_url != after_url),
                }
            )
        except Exception as exc:
            info["click_error"] = str(exc)
            info["invalid_action_reason"] = "candidate click failed"
            info["failure_reason"] = "candidate click failed"
            info["exception_type"] = type(exc).__name__
            info["exception_message"] = str(exc)
            info["retryable"] = True
        return info

    def _fill_input(self, decoded_action: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "fill_input", "filled": False, "action_success": False}
        page = _active_page(self.env)
        candidate = self._candidate_for_decoded_action(decoded_action)
        bid = str((candidate or {}).get("bid") or "")
        value = str(decoded_action.get("input_text") or "test")
        if isinstance(candidate, Mapping) and not bool(candidate.get("fillable")):
            info["fill_error"] = "target candidate is not fillable"
            info["invalid_action_reason"] = "fillable target candidate not found"
            info["failure_reason"] = "fillable target candidate not found"
            info["retryable"] = False
            return info
        locator = self._locator_for_candidate(page, candidate)
        if page is None or candidate is None or locator is None:
            info["fill_error"] = "active page or target candidate not found"
            info["invalid_action_reason"] = "fillable target candidate not found"
            info["failure_reason"] = "fillable target candidate not found"
            info["retryable"] = False
            return info
        try:
            locator.fill(value, force=True, timeout=1500)
            page.wait_for_timeout(150)
            info.update(
                {
                    "filled": True,
                    "action_success": True,
                    "success_reason": "target input filled",
                    "filled_bid": bid,
                    "filled_element_key": candidate.get("element_key"),
                    "input_text": value,
                }
            )
        except Exception as exc:
            info["fill_error"] = str(exc)
            info["failure_reason"] = "candidate fill failed"
            info["exception_type"] = type(exc).__name__
            info["exception_message"] = str(exc)
            info["retryable"] = True
        return info

    def _press_enter(self, decoded_action: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "press_enter", "pressed": False, "action_success": False}
        page = _active_page(self.env)
        candidate = self._candidate_for_decoded_action(decoded_action)
        bid = str((candidate or {}).get("bid") or "")
        locator = self._locator_for_candidate(page, candidate)
        if page is None or candidate is None or locator is None:
            info["press_error"] = "active page or target candidate not found"
            info["invalid_action_reason"] = "press target candidate not found"
            info["failure_reason"] = "press target candidate not found"
            info["retryable"] = False
            return info
        try:
            locator.press("Enter", timeout=1500)
            page.wait_for_timeout(250)
            info.update({"pressed": True, "pressed_bid": bid, "action_success": True, "success_reason": "target input submitted with Enter"})
        except Exception as exc:
            info["press_error"] = str(exc)
            info["failure_reason"] = "candidate press failed"
            info["exception_type"] = type(exc).__name__
            info["exception_message"] = str(exc)
            info["retryable"] = True
        return info

    def _locator_for_candidate(self, page: Any, candidate: Optional[Mapping[str, Any]]) -> Any:
        if page is None or not isinstance(candidate, Mapping):
            return None
        selectors = []
        for key in ("selector", "locator", "selector_hint"):
            value = str(candidate.get(key) or "")
            if value:
                selectors.append(value)
        bid = str(candidate.get("bid") or "")
        if bid:
            selectors.append(f'[bid="{bid}"]')
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if locator.count() > 0:
                    return locator
            except Exception:
                continue
        return None

    def _with_playwright_candidates(self, browser_obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        obs = dict(browser_obs or {})
        page = _active_page(self.env)
        if page is None:
            obs.setdefault("playwright_candidates", [])
            return obs
        try:
            obs["playwright_candidates"] = page.evaluate(_PLAYWRIGHT_CANDIDATE_SCRIPT)
        except Exception as exc:
            obs.setdefault("playwright_candidates", [])
            obs["playwright_candidate_error"] = str(exc)
        return obs

    def _candidate_for_decoded_action(self, decoded_action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = self._jaws_obs.get("candidate_elements", []) if self._jaws_obs else []
        index = int(decoded_action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], dict):
            return candidates[index]
        return None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _history(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "previous_action_type": self.previous_action_type,
            "no_change_steps": self.no_change_steps,
        }

    def _next_history(
        self,
        action_type: str,
        previous_jaws_obs: Dict[str, Any],
        browser_obs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous_url = previous_jaws_obs.get("page_state", {}).get("url")
        next_url = ""
        if isinstance(browser_obs, dict):
            next_url = str(browser_obs.get("url") or "")
        if previous_url and next_url and previous_url == next_url:
            self.no_change_steps += 1
        else:
            self.no_change_steps = 0
        return {
            "step_index": self.step_index,
            "previous_action_type": action_type,
            "no_change_steps": self.no_change_steps,
        }

    def _info(self, info: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        merged = dict(info or {})
        merged.setdefault("site_id", self.site_id)
        merged.setdefault("site_profile", self.site_profile)
        merged.setdefault("action_type", action_type)
        merged.setdefault("candidate_count", len(self._jaws_obs.get("candidate_elements", [])) if self._jaws_obs else 0)
        return merged

    def _stamp_site_id(self, observation: Dict[str, Any]) -> None:
        observation.setdefault("page_state", {})["site_id"] = self.site_id
        observation.setdefault("runtime_signals", {})["site_id"] = self.site_id
        observation.setdefault("runtime_signals", {})["site_profile"] = self.site_profile

    def _stamp_login_info(self, observation: Dict[str, Any], info: Mapping[str, Any]) -> None:
        runtime_signals = observation.setdefault("runtime_signals", {})
        runtime_signals["login_required"] = bool(info.get("login_required", self.requires_login))
        runtime_signals["login_credentials_available"] = bool(self.login_config)
        runtime_signals["login_attempted"] = bool(info.get("login_attempted", False))
        runtime_signals["login_success"] = bool(info.get("login_success", False))
        runtime_signals["logged_in"] = bool(info.get("logged_in", info.get("login_success", False)))
        runtime_signals["session_valid"] = bool(info.get("session_valid", info.get("login_success", False)))
        runtime_signals["user_role"] = str(info.get("user_role") or info.get("login_role") or "")
        runtime_signals["post_login_url"] = str(info.get("post_login_url") or "")
        if info.get("login_warning"):
            runtime_signals["login_warning"] = str(info.get("login_warning"))

    def _stamp_infra_info(self, observation: Dict[str, Any], info: Optional[Mapping[str, Any]] = None) -> None:
        infra = (info or {}).get("infra_signals") if isinstance(info, Mapping) else None
        if not isinstance(infra, Mapping):
            infra = self._collect_infra()
        observation["infra_signals"] = dict(infra)
        observation.setdefault("runtime_signals", {})["infra_observed"] = True

    def _collect_infra(self) -> Dict[str, Any]:
        timeout_ms = int(self.site_profile.get("infra_timeout_ms") or self.site_profile.get("timeout_ms") or 750)
        log_paths = self.site_profile.get("server_log_paths", [])
        if not isinstance(log_paths, list):
            log_paths = []
        database_paths = self.site_profile.get("database_paths", self.site_profile.get("db_paths", []))
        if not isinstance(database_paths, list):
            database_paths = []
        infra = collect_infra_observation(
            site_id=str(self.site_id or ""),
            base_url=self.base_url,
            timeout_ms=timeout_ms,
            log_paths=[str(item) for item in log_paths],
            database_paths=[str(item) for item in database_paths],
        )
        for key in (
            "db_engine",
            "relational_db_hook_available",
            "postgres_hook_available",
            "mysql_hook_available",
            "db_invariant_available",
            "trace_collector_available",
            "api_contract_available",
            "schema_available",
            "openapi_spec_available",
        ):
            if self.site_profile.get(key) not in (None, "", [], {}):
                infra[key] = self.site_profile.get(key)
        return infra


def _import_browsergym():
    try:
        import browsergym.core  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "BrowserGymJAWSEnv requires gymnasium and browsergym. "
            "Install BrowserGym before running the environment wrapper."
        ) from exc
    return gym


def _normalize_step_result(result: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("env.step() returned a non-tuple result")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward or 0.0), bool(terminated), bool(truncated), dict(info or {})
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward or 0.0), bool(done), False, dict(info or {})
    raise ValueError(f"Unsupported env.step() result length: {len(result)}")


def _action_count_limit_reached(step_index: int, max_steps: int | None) -> bool:
    limit = int(max_steps or 0)
    return bool(limit > 0 and int(step_index or 0) >= limit)


def _active_page(env: Any) -> Any:
    for candidate in _walk_page_candidates(env):
        page = _page_from_candidate(candidate)
        if page is not None:
            return page
    return None


def _walk_page_candidates(root: Any, max_depth: int = 4) -> list[Any]:
    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(root, 0)]
    candidates: list[Any] = []
    attr_names = (
        "unwrapped",
        "env",
        "_env",
        "task",
        "_task",
        "browser",
        "_browser",
        "context",
        "_context",
        "browser_context",
        "playwright_context",
        "page",
        "_page",
        "pages",
    )
    while queue:
        obj, depth = queue.pop(0)
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        candidates.append(obj)
        if depth >= max_depth:
            continue
        for attr_name in attr_names:
            try:
                value = getattr(obj, attr_name, None)
            except Exception:
                continue
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                queue.extend((item, depth + 1) for item in value)
            else:
                queue.append((value, depth + 1))
    return candidates


def _page_from_candidate(candidate: Any) -> Any:
    if candidate is None:
        return None
    if callable(getattr(candidate, "on", None)) and (
        callable(getattr(candidate, "locator", None)) or callable(getattr(candidate, "evaluate", None))
    ):
        return candidate
    for attr_name in ("page", "_page"):
        try:
            page = getattr(candidate, attr_name, None)
        except Exception:
            page = None
        if page is not None and callable(getattr(page, "on", None)):
            return page
    try:
        pages = getattr(candidate, "pages", None)
    except Exception:
        pages = None
    if pages:
        for page in pages:
            if page is not None and callable(getattr(page, "on", None)):
                return page
    return None


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        value = getattr(obj, name, default)
        if callable(value):
            try:
                return value()
            except TypeError:
                return value
        return value
    except Exception:
        return default


def _monotonic_timestamp() -> float:
    return time.monotonic()


def _api_entry_summary(entry: Mapping[str, Any]) -> Dict[str, Any]:
    status = _safe_int(entry.get("status") or entry.get("status_code"))
    url = str(entry.get("url") or entry.get("name") or entry.get("endpoint") or "")
    method = str(entry.get("method") or "").upper()
    message = str(entry.get("message") or entry.get("status_text") or entry.get("error") or entry.get("failure") or "")
    return {
        "api_probe_status": status,
        "api_probe_url": url,
        "api_probe_method": method or "GET",
        "api_probe_ok": 200 <= status < 300,
        "api_probe_message": message,
    }


def _is_api_like_url(url: str) -> bool:
    lowered = str(url or "").lower()
    if not lowered:
        return False
    return any(token in lowered for token in ("/api/", "/api?", ".json", "/graphql", "/rest/", "typicode", "my-json-server"))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _db_snapshot(infra: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "db_engine": str(infra.get("db_engine") or ""),
        "db_table_count": _safe_int(infra.get("db_table_count")),
        "db_connection_ok_count": _safe_int(infra.get("db_connection_ok_count")),
        "db_integrity_error_count": _safe_int(infra.get("db_integrity_error_count")),
        "server_log_db_error_count": _safe_int(infra.get("server_log_db_error_count")),
    }


def _compare_db_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    changed_tables = []
    changed_rows = 0
    if _safe_int(before.get("db_table_count")) != _safe_int(after.get("db_table_count")):
        changed_tables.append("table_count")
    if _safe_int(before.get("db_integrity_error_count")) != _safe_int(after.get("db_integrity_error_count")):
        changed_tables.append("integrity_check")
    return {
        "db_snapshot_available": bool(before and after),
        "db_state_changed": bool(changed_tables or changed_rows),
        "changed_tables": changed_tables,
        "changed_rows": changed_rows,
        "db_invariant_violation": _safe_int(after.get("db_integrity_error_count")) > 0,
    }


def _trace_id_from_entry(entry: Mapping[str, Any]) -> str:
    headers = entry.get("headers") if isinstance(entry.get("headers"), Mapping) else {}
    for key in ("traceparent", "x-trace-id", "x-request-id", "request-id", "trace_id", "request_id"):
        value = entry.get(key) or headers.get(key)
        if value:
            return str(value)
    return ""
