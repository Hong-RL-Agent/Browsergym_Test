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
        elif browser_action.action_type == "inspect_network":
            info = self._inspect_network()
        elif browser_action.action_type == "inspect_console":
            info = self._inspect_console()
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
        elif browser_action.action_type == "fill_input":
            info = self._fill_input(decoded_action)
        elif browser_action.action_type == "press_enter":
            info = self._press_enter(decoded_action)
        elif browser_action.action_type == "change_viewport_mobile":
            info = self._set_viewport(375, 812, browser_action.action_type)
            try:
                browser_obs, _, terminated, truncated, refresh_info = _normalize_step_result(self.env.step("noop()"))
                done = bool(terminated or truncated)
                info["refresh_info"] = refresh_info
            except Exception as exc:
                info["viewport_refresh_error"] = str(exc)
        elif browser_action.action_type == "change_viewport_desktop":
            info = self._set_viewport(1440, 900, browser_action.action_type)
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
                info = {"action_error": str(exc)}
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
        if self.step_index >= self.max_steps:
            done = True

        previous_jaws_obs = self._jaws_obs
        playwright_signal_info = self._playwright_signal_counts(event_start_index)
        info.update(playwright_signal_info)
        next_jaws_obs = self.observation_adapter.convert(
            browser_obs,
            info,
            previous_obs=previous_jaws_obs,
            history=self._next_history(browser_action.action_type, previous_jaws_obs, browser_obs),
        )
        self._stamp_site_id(next_jaws_obs)
        self._stamp_infra_info(next_jaws_obs, info)
        self._stamp_playwright_signals(next_jaws_obs, playwright_signal_info)
        self._browser_obs = browser_obs
        self._jaws_obs = next_jaws_obs
        self.previous_action_type = browser_action.action_type

        step_info = self._info(info, browser_action.action_type)
        self._stamp_login_info(next_jaws_obs, step_info)
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

    def _set_viewport(self, width: int, height: int, action_type: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "internal_action": action_type,
            "viewport_changed": False,
        }
        page = _active_page(self.env)
        if page is None:
            info["viewport_error"] = "active page not found"
            return info
        try:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(250)
            info.update({"viewport_changed": True, "viewport_width": width, "viewport_height": height})
        except Exception as exc:
            info["viewport_error"] = str(exc)
        return info

    def _perform_login(self, browser_obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "login_required": True,
            "login_attempted": True,
            "login_success": False,
            "post_login_url": str(browser_obs.get("url") or "") if isinstance(browser_obs, dict) else "",
        }
        page = _active_page(self.env)
        if page is None:
            info["login_warning"] = "active page not found"
            return info
        email = str(self.login_config.get("email") or "")
        password = str(self.login_config.get("password") or "")
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
            if not info["login_success"]:
                info["login_warning"] = "still observing login page after login attempt"
        except Exception as exc:
            info["login_warning"] = str(exc)
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
        info: Dict[str, Any] = {"internal_action": "inspect_network", "network_inspected": True}
        page = _active_page(self.env)
        if page is None:
            info["network_error"] = "active page not found"
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
            text = " ".join(str(item.get("name", "")) for item in info["network_entries"] if isinstance(item, dict)).lower()
            info["api_403_count"] = text.count("403") + text.count("forbidden")
            info["network_activity"] = bool(info["network_entries"])
        except Exception as exc:
            info["network_error"] = str(exc)
        return info

    def _inspect_console(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "inspect_console", "console_inspected": True}
        page = _active_page(self.env)
        if page is None:
            info["console_error"] = "active page not found"
            return info
        try:
            info["console_error_count"] = page.evaluate(
                """() => {
                    const text = document.body ? document.body.innerText.toLowerCase() : '';
                    return (text.match(/error|forbidden|timeout|failed/g) || []).length;
                }"""
            )
        except Exception as exc:
            info["console_error"] = str(exc)
        return info

    def _inspect_infra(self, action_type: str) -> Dict[str, Any]:
        infra = self._collect_infra()
        return {
            "internal_action": action_type,
            "infra_inspected": True,
            "infra_action": action_type,
            "infra_signals": infra,
        }

    def _fill_input(self, decoded_action: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "fill_input", "filled": False}
        page = _active_page(self.env)
        candidate = self._candidate_for_decoded_action(decoded_action)
        bid = str((candidate or {}).get("bid") or "")
        value = str(decoded_action.get("input_text") or "test")
        if page is None or not bid:
            info["fill_error"] = "active page or bid not found"
            return info
        try:
            page.locator(f'[bid="{bid}"]').fill(value, force=True, timeout=1000)
            page.wait_for_timeout(150)
            info.update({"filled": True, "filled_bid": bid, "input_text": value})
        except Exception as exc:
            info["fill_error"] = str(exc)
        return info

    def _press_enter(self, decoded_action: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "press_enter", "pressed": False}
        page = _active_page(self.env)
        candidate = self._candidate_for_decoded_action(decoded_action)
        bid = str((candidate or {}).get("bid") or "")
        if page is None or not bid:
            info["press_error"] = "active page or bid not found"
            return info
        try:
            page.locator(f'[bid="{bid}"]').press("Enter", timeout=1000)
            page.wait_for_timeout(250)
            info.update({"pressed": True, "pressed_bid": bid})
        except Exception as exc:
            info["press_error"] = str(exc)
        return info

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
        runtime_signals["login_attempted"] = bool(info.get("login_attempted", False))
        runtime_signals["login_success"] = bool(info.get("login_success", False))
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
        return collect_infra_observation(
            site_id=str(self.site_id or ""),
            base_url=self.base_url,
            timeout_ms=timeout_ms,
            log_paths=[str(item) for item in log_paths],
        )


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
