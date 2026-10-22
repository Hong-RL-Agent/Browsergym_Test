"""BrowserGym wrapper that exposes J.A.W.S raw observations and actions."""

from __future__ import annotations

import json
import time
import re
import traceback
from typing import Any, Dict, Mapping, Optional

from adapters.browsergym_action_adapter import BrowserGymActionAdapter, input_value_for_candidate
from adapters.browsergym_observation_adapter import BrowserGymObservationAdapter
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.infra_observation_service import collect_infra_observation
from services.interactive_layout_evidence_service import enrich_observation_with_interactive_layout_evidence
from services.site_profile_service import build_site_profile
from services.temporal_evidence_service import (
    ActionEvidenceWindow,
    EvidenceSample,
    RawEvidenceCollector,
    TemporalEvidenceConfig,
    build_evidence_sample,
)


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
        self._action_diagnostic_context: Optional[Dict[str, Any]] = None
        self._temporal_page = None
        self._temporal_network_events: list[Dict[str, Any]] = []
        self._temporal_active_requests: Dict[str, Dict[str, Any]] = {}
        self._temporal_console_errors: list[Dict[str, Any]] = []
        self._temporal_action_origin = 0.0
        self._temporal_target_bid = ""

    def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        gym = _import_browsergym()
        if self.env is not None:
            self.close()

        gym_kwargs: Dict[str, Any] = {
            "task_kwargs": {"start_url": self.base_url},
            "wait_for_user_message": False,
            "headless": self.headless,
        }
        timeout_ms = self.site_profile.get("browsergym_timeout_ms") or self.site_profile.get("browser_timeout_ms")
        if timeout_ms:
            gym_kwargs["timeout"] = int(timeout_ms)
        self.env = gym.make("browsergym/openended", **gym_kwargs)
        browser_obs, info = self.env.reset()
        self._reset_temporal_runtime()
        self._ensure_temporal_hooks()
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
        self._stamp_external_runtime_state(self._jaws_obs)
        return self._jaws_obs, self._info(info, self.previous_action_type)

    def step(self, action_id: int) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.env is None or self._jaws_obs is None:
            raise RuntimeError("Call reset() before step().")

        decoded_action = self.action_space.decode(action_id)
        selected_candidate = self._candidate_for_decoded_action(decoded_action)
        browser_action = self.action_adapter.adapt(decoded_action, self._jaws_obs)
        diagnostic_context = dict(self._action_diagnostic_context or {})
        execution_diagnostic: Dict[str, Any] = {
            "enabled": bool(diagnostic_context),
            "action_trace_id": diagnostic_context.get("action_trace_id"),
            "primary_action_attempted": False,
            "primary_action_succeeded": False,
            "fallback_triggered": False,
            "fallback_attempted": False,
            "fallback_succeeded": False,
            "error_origin": None,
            "exception_module": None,
            "exception_class": None,
            "exception_message": None,
            "traceback": None,
            "target_resolved_at_execution": None,
            "target_resolution_method": None,
        }
        if diagnostic_context:
            execution_diagnostic.update(self._diagnostic_target_snapshot(selected_candidate))
        self._begin_temporal_action_window(
            selected_candidate,
            action_context=getattr(self, "_pending_action_context", None),
        )
        self._pending_action_context = None
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
        }:
            info = self._inspect_infra(browser_action.action_type)
        elif browser_action.action_type == "fill_input":
            info = self._fill_input(decoded_action)
        elif browser_action.action_type == "select_option":
            info = self._select_option(decoded_action)
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
            execution_diagnostic["primary_action_attempted"] = True
            try:
                browser_obs, _, terminated, truncated, info = _normalize_step_result(
                    self.env.step(browser_action.action)
                )
                done = bool(terminated or truncated)
                execution_diagnostic["primary_action_succeeded"] = True
            except Exception as exc:
                action_error = True
                execution_diagnostic.update(
                    {
                        "error_origin": "PYTHON_EXCEPTION",
                        "exception_module": type(exc).__module__,
                        "exception_class": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                info = {"action_error": str(exc)}
                if browser_action.action != browser_action.fallback_action:
                    execution_diagnostic["fallback_triggered"] = True
                    execution_diagnostic["fallback_attempted"] = True
                    try:
                        browser_obs, _, terminated, truncated, fallback_info = _normalize_step_result(
                            self.env.step(browser_action.fallback_action)
                        )
                        done = bool(terminated or truncated)
                        info["fallback_info"] = fallback_info
                        execution_diagnostic["fallback_succeeded"] = True
                    except Exception as fallback_exc:
                        info["fallback_error"] = str(fallback_exc)
                        execution_diagnostic.update(
                            {
                                "error_origin": "FALLBACK_FAILURE",
                                "fallback_exception_module": type(fallback_exc).__module__,
                                "fallback_exception_class": type(fallback_exc).__name__,
                                "fallback_exception_message": str(fallback_exc),
                                "fallback_traceback": traceback.format_exc(),
                            }
                        )
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
        explicit_action_error = bool(action_error or _has_internal_execution_error(info))
        next_jaws_obs = self.observation_adapter.convert(
            browser_obs,
            info,
            previous_obs=previous_jaws_obs,
            history=self._next_history(browser_action.action_type, previous_jaws_obs, browser_obs),
        )
        self._stamp_site_id(next_jaws_obs)
        self._stamp_infra_info(next_jaws_obs, info)
        raw_last_action_error = bool(next_jaws_obs.get("runtime_signals", {}).get("last_action_error"))
        next_jaws_obs.setdefault("runtime_signals", {})["browsergym_raw_last_action_error"] = raw_last_action_error
        next_jaws_obs["runtime_signals"]["last_action_error"] = explicit_action_error
        self._browser_obs = browser_obs
        self._jaws_obs = next_jaws_obs
        self.previous_action_type = browser_action.action_type

        step_info = self._info(info, browser_action.action_type)
        error_text = _first_error_text(info)
        error_category = normalize_action_error(error_text) if error_text else ""
        self._stamp_login_info(next_jaws_obs, step_info)
        self._stamp_external_runtime_state(next_jaws_obs)
        step_info.update(
            {
                "action": decoded_action,
                "browsergym_action": browser_action.action,
                "last_action_error": explicit_action_error,
                "execution_action_error": explicit_action_error,
                "browsergym_raw_last_action_error": raw_last_action_error,
                "action_error_category": error_category,
                "action_error_summary": _short_error_summary(error_text),
                "selected_candidate": _candidate_execution_summary(selected_candidate),
                "candidate_count": len(next_jaws_obs.get("candidate_elements", [])),
                "cart_count": next_jaws_obs.get("page_state", {}).get("cart_count"),
                "cart_text": next_jaws_obs.get("page_state", {}).get("cart_text"),
                "cart_count_detected": next_jaws_obs.get("page_state", {}).get("cart_count_detected"),
                "action_result": {
                    "failed": explicit_action_error,
                    "viewport_changed": bool(info.get("viewport_changed")),
                    "viewport_width": info.get("viewport_width"),
                    "viewport_height": info.get("viewport_height"),
                },
                "infra_signals": next_jaws_obs.get("infra_signals", {}),
            }
        )
        if browser_action.action_type == "select_option":
            action_context = dict(getattr(self, "_temporal_action_context", {}) or {})
            selected_option = info.get("selected_option") if isinstance(info.get("selected_option"), Mapping) else {}
            provenance = {
                "episode_id": action_context.get("episode_id") or self.site_id,
                "step": action_context.get("action_step") or self.step_index,
                "action_id": action_context.get("action_id") or action_context.get("action_trace_id") or f"{self.site_id}:step:{self.step_index}",
                "action_type": "select_option",
                "target_bid": info.get("selected_bid") or (selected_candidate or {}).get("bid"),
                "target_tag": str((selected_candidate or {}).get("tag") or "select"),
                "target_role": (selected_candidate or {}).get("role"),
                "option_identity": {
                    key: selected_option.get(key)
                    for key in ("value", "label", "text", "index")
                    if selected_option.get(key) is not None
                },
                "selected_value_before": info.get("current_value_before"),
                "selected_value_after": selected_option.get("value"),
                "execution_success": bool(info.get("selected")) and not explicit_action_error,
                "action_start_timestamp": action_context.get("action_start_timestamp"),
                "action_end_timestamp": action_context.get("action_end_timestamp") or _monotonic_ms(),
            }
            step_info["origin_action_provenance"] = provenance
            next_jaws_obs.setdefault("runtime_signals", {})["origin_action_provenance"] = provenance
        if diagnostic_context:
            if isinstance(info.get("diagnostic_exception"), Mapping):
                execution_diagnostic.update(dict(info["diagnostic_exception"]))
            execution_diagnostic.update(self._diagnostic_page_state())
            execution_diagnostic["last_action_error"] = explicit_action_error
            execution_diagnostic["failed"] = explicit_action_error
            execution_diagnostic["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
            step_info["action_error_diagnostic"] = execution_diagnostic
        return next_jaws_obs, 0.0, done, step_info

    def collect_temporal_evidence(
        self,
        action_context: Mapping[str, Any],
        config: Optional[TemporalEvidenceConfig] = None,
    ) -> ActionEvidenceWindow:
        if self.env is None or self._jaws_obs is None:
            raise RuntimeError("Call reset() before collecting temporal evidence.")
        temporal_config = config or TemporalEvidenceConfig()
        context = dict(action_context)
        context.setdefault("relative_origin_ms", 0)

        def sample() -> Mapping[str, Any]:
            self._ensure_temporal_hooks()
            self._stamp_temporal_runtime(self._jaws_obs)
            return self._jaws_obs

        def sleep_and_refresh(ms: int) -> None:
            page = _active_page(self.env)
            if page is not None:
                try:
                    page.wait_for_timeout(max(0, int(ms)))
                except Exception:
                    pass
            self._refresh_temporal_snapshot_from_page()

        collector = RawEvidenceCollector(
            sample,
            config=temporal_config,
            sleep_ms=sleep_and_refresh,
        )
        window = collector.collect(context)
        if isinstance(window.terminal_snapshot, dict):
            self._jaws_obs = window.terminal_snapshot
        return window

    def collect_interactive_layout_evidence(self, observation: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Enrich a stable detector snapshot with read-only browser layout evidence."""
        page = _active_page(self.env)
        enriched, summary = enrich_observation_with_interactive_layout_evidence(page, observation)
        if isinstance(enriched, dict):
            self._jaws_obs = enriched
        return enriched, summary

    def collect_post_stable_diagnostic_samples(
        self,
        temporal_window: ActionEvidenceWindow,
        action_context: Mapping[str, Any],
        offsets_ms: list[int],
        config: Optional[TemporalEvidenceConfig] = None,
    ) -> list[EvidenceSample]:
        if self.env is None or self._jaws_obs is None or not temporal_window.samples:
            return []
        stable_snapshot = self._jaws_obs
        stable_ms = int(temporal_window.samples[-1].relative_ms)
        samples: list[EvidenceSample] = []
        previous_offset = 0
        temporal_config = config or TemporalEvidenceConfig()
        for offset in offsets_ms:
            delay = max(0, int(offset) - previous_offset)
            previous_offset = int(offset)
            page = _active_page(self.env)
            if page is not None:
                try:
                    page.wait_for_timeout(delay)
                except Exception:
                    pass
            self._refresh_temporal_snapshot_from_page()
            sample_context = dict(action_context)
            previous_sample = samples[-1] if samples else temporal_window.samples[-1]
            sample_context["previous_sample"] = previous_sample
            now = time.time()
            samples.append(
                build_evidence_sample(
                    self._jaws_obs,
                    action_context=sample_context,
                    timestamp=now,
                    relative_ms=stable_ms + int(offset),
                    config=temporal_config,
                )
            )
        if isinstance(stable_snapshot, dict):
            self._jaws_obs = stable_snapshot
        return samples

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
                page.wait_for_load_state("networkidle", timeout=4000)
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
        value = input_value_for_candidate(candidate or {}, decoded_action)
        if page is None or not bid:
            info["fill_error"] = "active page or bid not found"
            return info
        try:
            page.locator(f'[bid="{bid}"]').fill(value, force=True, timeout=1000)
            page.wait_for_timeout(150)
            info.update({"filled": True, "filled_bid": bid, "input_text": value})
        except Exception as exc:
            info["fill_error"] = str(exc)
            info["diagnostic_exception"] = {
                "error_origin": "PLAYWRIGHT_EXCEPTION",
                "exception_module": type(exc).__module__,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        return info

    def _select_option(self, decoded_action: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {"internal_action": "select_option", "selected": False}
        page = _active_page(self.env)
        candidate = self._candidate_for_decoded_action(decoded_action)
        bid = str((candidate or {}).get("bid") or "")
        if page is None or not bid:
            info["select_error"] = "active page or bid not found"
            return info
        try:
            locator = page.locator(f'[bid="{bid}"]')
            if locator.count() != 1:
                info["select_error"] = "select target not uniquely resolvable"
                return info
            if str(locator.evaluate("el => String(el.tagName || '').toLowerCase()")) != "select":
                info["select_error"] = "target is not a native select"
                return info
            current_value = str(locator.input_value() or "")
            options = locator.locator("option")
            option_rows = []
            selected_value = None
            for index in range(options.count()):
                option = options.nth(index)
                value = str(option.get_attribute("value") or "")
                label = str(option.inner_text() or "")
                disabled = bool(option.is_disabled())
                selected = value == current_value
                option_rows.append({"value": value, "label": label, "index": index, "disabled": disabled, "selected": selected})
                if selected:
                    selected_value = value
            valid_options = [row for row in option_rows if not row["disabled"] and row["value"] != current_value]
            if not valid_options:
                info.update({"select_error": "no enabled option different from current value", "options": option_rows})
                return info
            chosen = valid_options[0]
            locator.select_option(value=chosen["value"], timeout=1000)
            page.wait_for_timeout(150)
            info.update({"selected": True, "selected_bid": bid, "selected_option": chosen, "current_value_before": current_value, "options": option_rows})
        except Exception as exc:
            info["select_error"] = str(exc)
            info["diagnostic_exception"] = {
                "error_origin": "PLAYWRIGHT_EXCEPTION",
                "exception_module": type(exc).__module__,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
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
            info["diagnostic_exception"] = {
                "error_origin": "PLAYWRIGHT_EXCEPTION",
                "exception_module": type(exc).__module__,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        return info

    def _candidate_for_decoded_action(self, decoded_action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = self._jaws_obs.get("candidate_elements", []) if self._jaws_obs else []
        index = int(decoded_action.get("candidate_index", 0) or 0)
        if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], dict):
            return candidates[index]
        return None

    def _reset_temporal_runtime(self) -> None:
        self._temporal_page = None
        self._temporal_network_events = []
        # Keep requests started by an earlier action alive so a late response
        # or failure retains its original start timestamp and is classified as
        # background rather than caused by the current action.
        self._temporal_active_requests = dict(getattr(self, "_temporal_active_requests", {}) or {})
        self._temporal_action_context = {}
        self._pending_action_context = None
        self._temporal_console_errors = []
        self._temporal_action_origin = 0.0

    def _begin_temporal_action_window(
        self,
        selected_candidate: Optional[Mapping[str, Any]] = None,
        action_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._ensure_temporal_hooks()
        self._temporal_action_origin = _monotonic_ms()
        self._temporal_target_bid = str((selected_candidate or {}).get("bid") or "")
        self._temporal_network_events = []
        self._temporal_active_requests = {}
        self._temporal_console_errors = []
        context = dict(action_context or {})
        context.setdefault("action_start_timestamp", _monotonic_ms())
        self._temporal_action_context = context

    def _ensure_temporal_hooks(self) -> None:
        page = _active_page(self.env)
        if page is None or page is self._temporal_page:
            return
        self._temporal_page = page
        try:
            page.on("request", self._on_temporal_request)
            page.on("response", self._on_temporal_response)
            page.on("requestfailed", self._on_temporal_request_failed)
            page.on("console", self._on_temporal_console)
            page.on("pageerror", self._on_temporal_page_error)
        except Exception:
            pass

    def _on_temporal_request(self, request: Any) -> None:
        request_id = _request_identity(request)
        event = {
            "event": "request",
            "request_id": request_id,
            "relative_ms": self._relative_temporal_ms(),
            "request_relative_ms": self._relative_temporal_ms(),
            "request_start_timestamp": _monotonic_ms(),
            "method": _safe_call(request, "method"),
            "url": _safe_attr(request, "url"),
            "resource_type": _safe_call(request, "resource_type"),
            "headers": _safe_call(request, "all_headers"),
            "request_body_summary": _bounded_text(_safe_call(request, "post_data"), limit=2048),
        }
        self._temporal_active_requests[request_id] = dict(event)
        self._temporal_network_events.append(event)

    def _on_temporal_response(self, response: Any) -> None:
        request = _safe_call(response, "request")
        request_id = _request_identity(request or response)
        event = {
            "event": "response",
            "request_id": request_id,
            "relative_ms": self._relative_temporal_ms(),
            "request_relative_ms": self._temporal_active_requests.get(request_id, {}).get("request_relative_ms"),
            "request_start_timestamp": self._temporal_active_requests.get(request_id, {}).get("request_start_timestamp"),
            "response_timestamp": _monotonic_ms(),
            "method": _safe_call(request, "method") if request is not None else "",
            "url": _safe_attr(response, "url") or (_safe_attr(request, "url") if request is not None else ""),
            "resource_type": _safe_call(request, "resource_type") if request is not None else "",
            "status": _safe_call(response, "status"),
            "headers": _safe_call(request, "all_headers") if request is not None else {},
            "response_headers": _safe_call(response, "all_headers"),
            "response_received": True,
            "network_failure": False,
        }
        event.update(_response_body_capture(response))
        self._temporal_active_requests.pop(request_id, None)
        self._temporal_network_events.append(event)

    def _on_temporal_request_failed(self, request: Any) -> None:
        request_id = _request_identity(request)
        failure = _safe_call(request, "failure")
        started = self._temporal_active_requests.get(request_id, {})
        event = {
            "event": "requestfailed",
            "request_id": request_id,
            "relative_ms": self._relative_temporal_ms(),
            "request_start_timestamp": started.get("request_start_timestamp") or _monotonic_ms(),
            "response_timestamp": None,
            "method": _safe_call(request, "method"),
            "url": _safe_attr(request, "url"),
            "resource_type": _safe_call(request, "resource_type"),
            "headers": _safe_call(request, "all_headers"),
            "status": None,
            "response_received": False,
            "network_failure": True,
            "failure_text": failure.get("errorText") if isinstance(failure, Mapping) else str(failure or ""),
        }
        self._temporal_active_requests.pop(request_id, None)
        self._temporal_network_events.append(event)

    def _on_temporal_console(self, message: Any) -> None:
        try:
            message_type = str(message.type)
        except Exception:
            message_type = ""
        if message_type.lower() != "error":
            return
        self._temporal_console_errors.append(
            {
                "relative_ms": self._relative_temporal_ms(),
                "type": message_type,
                "text": _safe_call(message, "text"),
            }
        )
    def _on_temporal_page_error(self, error: Any) -> None:
        self._temporal_console_errors.append(
            {
                "relative_ms": self._relative_temporal_ms(),
                "type": "pageerror",
                "text": str(error or ""),
            }
        )

    def _relative_temporal_ms(self) -> int:
        if self._temporal_action_origin <= 0:
            return 0
        return int(max(0.0, _monotonic_ms() - self._temporal_action_origin))

    def _stamp_temporal_runtime(self, observation: Dict[str, Any]) -> None:
        runtime = observation.setdefault("runtime_signals", {})
        events = list(self._temporal_network_events)
        requests = [event for event in events if event.get("event") == "request"]
        completed = [event for event in events if event.get("event") == "response"]
        failed = [event for event in events if event.get("event") == "requestfailed"]
        action_context = dict(getattr(self, "_temporal_action_context", {}) or {})
        action_context.setdefault("action_end_timestamp", _monotonic_ms())
        for event in events:
            event.setdefault("action_start_timestamp", action_context.get("action_start_timestamp"))
            event.setdefault("action_end_timestamp", action_context.get("action_end_timestamp"))
            event.setdefault("action_id", action_context.get("action_id"))
            event.setdefault("action_type", action_context.get("action_type"))
            event.setdefault("target_identity", action_context.get("target_identity"))
        runtime["temporal_network"] = {
            "request_events": requests,
            "active_requests": list(self._temporal_active_requests.values()),
            "completed_requests": completed,
            "failed_requests": failed,
            "active_request_count": len(self._temporal_active_requests),
            "completed_request_count": len(completed),
            "failed_request_count": len(failed),
            "action_context": action_context,
        }
        runtime["temporal_console_errors"] = list(self._temporal_console_errors)

    def _refresh_temporal_snapshot_from_page(self) -> None:
        page = _active_page(self.env)
        if page is None or self._jaws_obs is None:
            return
        snapshot = _shallow_observation_copy(self._jaws_obs)
        page_state = snapshot.setdefault("page_state", {})
        try:
            text = str(page.evaluate("() => document.body ? document.body.innerText : ''") or "")
        except Exception:
            text = str(page_state.get("page_text_sample") or "")
        try:
            page_state["url"] = str(page.url or page_state.get("url") or "")
        except Exception:
            pass
        try:
            page_state["title"] = str(page.title() or page_state.get("title") or "")
        except Exception:
            pass
        if text:
            page_state["page_text"] = text
            page_state["page_text_sample"] = text[:500]
            page_state["page_text_length"] = len(text)
            cart_count = _cart_count_from_text(text)
            if cart_count is not None:
                page_state["cart_count_detected"] = True
                page_state["cart_count"] = cart_count
                page_state["cart_text"] = f"Cart ({cart_count})"
        try:
            page_state["dom_node_count"] = int(page.evaluate("() => document.querySelectorAll('*').length") or 0)
        except Exception:
            pass
        try:
            page_state["ui_semantic_fields"] = _extract_ui_semantic_fields_from_page(page)
        except Exception:
            page_state.setdefault("ui_semantic_fields", [])
        candidates = snapshot.get("candidate_elements", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and str(candidate.get("bid") or "") == self._temporal_target_bid:
                    self._refresh_candidate_from_page(page, candidate)
        self._stamp_temporal_runtime(snapshot)
        self._jaws_obs = snapshot

    def _refresh_candidate_from_page(self, page: Any, candidate: Dict[str, Any]) -> None:
        bid = str(candidate.get("bid") or "")
        if not bid:
            return
        try:
            locator = page.locator(f'[bid="{bid}"]').first
            if locator.count() <= 0:
                candidate["visible"] = False
                candidate["visibility"] = 0.0
                return
            visible = bool(locator.is_visible(timeout=200))
            candidate["visible"] = visible
            candidate["visibility"] = 1.0 if visible else 0.0
            try:
                candidate["enabled"] = bool(locator.is_enabled(timeout=200))
            except Exception:
                pass
            try:
                text = str(locator.inner_text(timeout=200) or "").strip()
                if text:
                    candidate["text"] = text
                    candidate["name"] = text
            except Exception:
                pass
            try:
                bbox = locator.bounding_box(timeout=200)
                if isinstance(bbox, Mapping):
                    candidate["bbox"] = [
                        float(bbox.get("x") or 0.0),
                        float(bbox.get("y") or 0.0),
                        float(bbox.get("width") or 0.0),
                        float(bbox.get("height") or 0.0),
                    ]
            except Exception:
                pass
        except Exception:
            return

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

    def set_action_diagnostic_context(self, context: Optional[Mapping[str, Any]]) -> None:
        """Set per-step trace context; no-op for all behavior except opt-in logging."""
        self._action_diagnostic_context = dict(context or {}) or None
        self._pending_action_context = dict(context or {}) or None

    def runtime_candidate_compatibility(self, candidate: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Expose the same bid-based DOM probe used by execution diagnostics."""
        return self._diagnostic_target_snapshot(candidate)

    def _diagnostic_target_snapshot(self, candidate: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(candidate, Mapping):
            return {
                "target_present_at_selection": False,
                "target_resolved_at_execution": False,
                "target_resolution_method": None,
                "target_locator_count": 0,
            }
        bid = str(candidate.get("bid") or "")
        page = _active_page(self.env)
        resolved = False
        locator_count = 0
        runtime_snapshot: Dict[str, Any] = {}
        if page is not None and bid:
            try:
                locator = page.locator(f'[bid="{bid}"]')
                locator_count = int(locator.count())
                resolved = locator_count > 0
                if resolved:
                    target = locator.first
                    bbox = target.bounding_box() or {}
                    runtime_snapshot = {
                        "target_visible_runtime": bool(target.is_visible()),
                        "target_enabled_runtime": bool(target.is_enabled()),
                        "target_bbox_runtime": [
                            bbox.get("x", 0.0), bbox.get("y", 0.0),
                            bbox.get("width", 0.0), bbox.get("height", 0.0),
                        ] if isinstance(bbox, Mapping) else [],
                        "target_dom_attributes": target.evaluate(
                            """el => ({
                                tag: String(el.tagName || '').toLowerCase(),
                                type: String(el.getAttribute('type') || '').toLowerCase(),
                                role: String(el.getAttribute('role') || '').toLowerCase(),
                                contenteditable: String(el.getAttribute('contenteditable') || '').toLowerCase()
                            })"""
                        ),
                    }
                    dom = runtime_snapshot.get("target_dom_attributes") or {}
                    tag = str(dom.get("tag") or "")
                    role = str(dom.get("role") or "")
                    contenteditable = str(dom.get("contenteditable") or "")
                    runtime_snapshot["target_fillable_runtime"] = bool(
                        tag in {"input", "textarea"}
                        or tag != "select" and role in {"textbox", "searchbox"}
                        or contenteditable in {"true", "plaintext-only"}
                    )
                    runtime_snapshot["target_clickable_runtime"] = bool(runtime_snapshot.get("target_visible_runtime"))
            except Exception:
                resolved = False
        snapshot = {
            "target_present_at_selection": True,
            "target_resolved_at_execution": resolved,
            "target_resolution_method": "bid" if bid else None,
            "execution_target_bid": bid,
            "target_locator_count": locator_count,
            "target_candidate_role": str(candidate.get("role") or ""),
            "target_candidate_tag": str(candidate.get("tag") or ""),
            "target_candidate_type": str(candidate.get("type") or candidate.get("input_type") or ""),
            "target_candidate_visible": bool(candidate.get("visible", True)),
            "target_candidate_enabled": bool(candidate.get("enabled", True)),
            "target_candidate_fillable": bool(candidate.get("fillable", False)),
            "target_candidate_clickable": bool(candidate.get("clickable", False)),
            "target_candidate_bbox": list(candidate.get("bbox") or [])[:4] if isinstance(candidate.get("bbox"), list) else [],
        }
        snapshot.update(runtime_snapshot)
        return snapshot

    def _diagnostic_page_state(self) -> Dict[str, Any]:
        page = _active_page(self.env)
        connected = False
        page_closed = None
        context_available = False
        url = ""
        if page is not None:
            try:
                page_closed = bool(page.is_closed())
                url = str(page.url or "")
                context = page.context
                context_available = context is not None
                browser = context.browser if context is not None else None
                connected = bool(browser and browser.is_connected())
            except Exception:
                pass
        return {
            "browser_connected_after": connected,
            "page_closed_after": page_closed,
            "context_available_after": context_available,
            "current_url_after_execution": url,
        }

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

    def _stamp_external_runtime_state(self, observation: Dict[str, Any]) -> None:
        """Expose public runtime state for detector evidence attribution.

        This is deliberately generic and GT-blind.  It reads only state that
        the running page exposes to its own browser runtime (for example a
        public application-state object), never bug catalogs, labels, or
        expected answers.  The state is diagnostic evidence; the fixed PPO
        observation encoder remains unchanged.
        """
        page = _active_page(self.env)
        if page is None:
            return
        try:
            state = page.evaluate(
                """() => {
                    const clone = (value) => {
                      if (value === undefined || value === null) return null;
                      try { return JSON.parse(JSON.stringify(value)); } catch (_) { return null; }
                    };
                    const actions = Array.isArray(window.__compareLabActions)
                      ? window.__compareLabActions.slice(-20).map(clone)
                      : [];
                    return {
                      compare_lab_state: clone(window.__compareLabState),
                      compare_lab_actions: actions,
                      source: "browser_runtime_window_state",
                      freshness: "CURRENT_ACTION",
                      gt_blind: true
                    };
                }"""
            )
        except Exception:
            return
        if not isinstance(state, Mapping):
            return
        runtime = observation.setdefault("runtime_signals", {})
        runtime["external_runtime_state"] = dict(state)
        runtime["external_runtime_state_provenance"] = {
            "source": "browser_runtime_window_state",
            "freshness": "CURRENT_ACTION",
            "gt_blind": True,
        }

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
    candidates = [env, getattr(env, "unwrapped", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        page = getattr(candidate, "page", None)
        if page is not None:
            return page
        pages = getattr(candidate, "pages", None)
        if pages:
            return pages[0]
    return None


def _monotonic_ms() -> float:
    return time.monotonic() * 1000.0


def _safe_attr(obj: Any, name: str) -> str:
    if obj is None:
        return ""
    try:
        value = getattr(obj, name, "")
        if callable(value):
            value = value()
        return str(value or "")
    except Exception:
        return ""


def _safe_call(obj: Any, name: str) -> Any:
    if obj is None:
        return ""
    try:
        value = getattr(obj, name, "")
        return value() if callable(value) else value
    except Exception:
        return ""


def _request_identity(request: Any) -> str:
    if request is None:
        return ""
    for attr in ("_guid", "guid", "request_id"):
        try:
            value = getattr(request, attr, None)
            if value:
                return str(value)
        except Exception:
            pass
    try:
        impl = getattr(request, "_impl_obj", None)
        guid = getattr(impl, "_guid", None)
        if guid:
            return str(guid)
    except Exception:
        pass
    return str(id(request))


def _response_body_capture(response: Any, *, max_bytes: int = 65536) -> Dict[str, Any]:
    headers = _safe_call(response, "all_headers")
    headers = headers if isinstance(headers, Mapping) else {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    allowed = any(token in content_type for token in ("application/json", "application/problem+json", "text/plain"))
    payload: Dict[str, Any] = {
        "content_type": content_type.split(";", 1)[0],
        "response_body_available": False,
        "response_body_truncated": False,
    }
    if not allowed:
        return payload
    try:
        raw = _safe_call(response, "body")
    except Exception as exc:
        payload["response_body_error"] = str(exc)
        return payload
    if isinstance(raw, str):
        data = raw.encode("utf-8", errors="replace")
    elif isinstance(raw, bytes):
        data = raw
    else:
        return payload
    payload["response_body_available"] = True
    payload["response_body_size"] = len(data)
    if len(data) > max_bytes:
        payload["response_body_truncated"] = True
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    payload["response_body_summary"] = _bounded_text(text, limit=4096)
    if "json" in content_type:
        try:
            payload["response_body_json"] = json.loads(text)
        except Exception as exc:
            payload["response_body_parse_error"] = str(exc)
            payload["response_body_text"] = payload["response_body_summary"]
    else:
        payload["response_body_text"] = payload["response_body_summary"]
    return payload


def _bounded_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit]


def _extract_ui_semantic_fields_from_page(page: Any) -> list[Dict[str, Any]]:
    robust_script = """
    () => {
      const visible = (el) => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
      };
      const text = (el) => (el && (el.innerText || el.textContent || el.value || '') || '').trim().replace(/\\s+/g, ' ');
      const cssSafe = (value) => {
        if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(value);
        return String(value || '').replace(/["\\\\]/g, '\\\\$&');
      };
      const path = (el) => {
        const parts = [];
        let node = el;
        while (node && node.nodeType === 1 && parts.length < 5) {
          let part = node.tagName.toLowerCase();
          if (node.id) part += '#' + node.id;
          if (node.getAttribute('data-testid')) part += '[data-testid="' + node.getAttribute('data-testid') + '"]';
          parts.unshift(part);
          node = node.parentElement;
        }
        return parts.join('>');
      };
      const rows = [];
      const push = (key, value, el, method) => {
        key = String(key || '').trim();
        value = String(value || '').trim();
        if (!key || !value || key.length > 80 || value.length > 120) return;
        rows.push({
          key,
          value,
          display_text: key + ': ' + value,
          element_role: el.getAttribute('role') || el.tagName.toLowerCase(),
          element_id: el.id || el.getAttribute('name') || el.getAttribute('bid') || '',
          dom_path: path(el),
          visible: visible(el),
          extraction_method: method
        });
      };
      const inferTextKey = (value, el) => {
        const raw = String(value || '').trim();
        const low = raw.toLowerCase();
        if (!raw) return '';
        if (/[$￦₩€£]\\s*[-+]?\\d/.test(raw)) return 'price';
        if (/[-+]?\\d[\\d,.]*\\s*(pts?|points?)\\b/i.test(raw)) return 'points';
        if (/^[-+]?\\d[\\d,.]*$/.test(raw)) {
          const parentText = el && el.parentElement ? text(el.parentElement).toLowerCase() : '';
          if (/point|pts|balance|reward/.test(parentText)) return 'points';
          if (/price|cost|amount|total|payment|charge|fee/.test(parentText)) return 'price';
        }
        if (/^(physical|digital|basic|pro|premium|enterprise|free|paid)$/i.test(raw)) {
          return /physical|digital/i.test(raw) ? 'type' : 'plan';
        }
        if (/success|failed|error|granted|applied|completed|triggered/i.test(raw)) return 'message';
        if (/active|inactive|enabled|disabled|pending|complete/i.test(low)) return 'status';
        return '';
      };
      document.querySelectorAll('input, textarea, select').forEach((el) => {
        if (!visible(el)) return;
        const id = el.id;
        const label = (el.labels && el.labels.length ? el.labels[0] : null)
          || (id ? document.querySelector('label[for="' + cssSafe(id) + '"]') : null);
        const key = (label && text(label)) || el.getAttribute('aria-label') || el.getAttribute('name') || el.id || el.placeholder;
        const value = el.tagName.toLowerCase() === 'select'
          ? (el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : el.value)
          : el.value;
        push(key, value, el, 'FORM_VALUE');
      });
      document.querySelectorAll('[data-field], [data-key], [data-label]').forEach((el) => {
        if (!visible(el)) return;
        const key = el.getAttribute('data-field') || el.getAttribute('data-key') || el.getAttribute('data-label');
        const value = el.getAttribute('data-value') || text(el);
        push(key, value, el, 'DATA_FIELD');
      });
      document.querySelectorAll('tr').forEach((tr) => {
        const cells = Array.from(tr.querySelectorAll('th,td')).filter(visible);
        if (cells.length >= 2) push(text(cells[0]), text(cells[1]), cells[1], 'TABLE_ROW');
      });
      Array.from(document.querySelectorAll('body *')).slice(0, 800).forEach((el) => {
        if (!visible(el)) return;
        const children = Array.from(el.children || []).filter(visible);
        const value = text(el);
        if (!value || value.length > 120) return;
        const kv = value.match(/^([^:\\n]{1,40})\\s*[:：]\\s*(.{1,80})$/);
        if (kv) {
          push(kv[1], kv[2], el, 'TEXT_KEY_VALUE');
          return;
        }
        if (children.length > 0) return;
        const key = inferTextKey(value, el);
        if (key) push(key, value, el, 'TEXT_VALUE_HEURISTIC');
      });
      return rows.slice(0, 120);
    }
    """
    robust_rows = _safe_call_eval(page, robust_script)
    if isinstance(robust_rows, list):
        return [dict(row) for row in robust_rows if isinstance(row, Mapping)][:120]
    script = """
    () => {
      const visible = (el) => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
      };
      const text = (el) => (el && (el.innerText || el.textContent || el.value || '') || '').trim().replace(/\\s+/g, ' ');
      const path = (el) => {
        const parts = [];
        let node = el;
        while (node && node.nodeType === 1 && parts.length < 5) {
          let part = node.tagName.toLowerCase();
          if (node.id) part += '#' + node.id;
          if (node.getAttribute('data-testid')) part += '[data-testid="' + node.getAttribute('data-testid') + '"]';
          parts.unshift(part);
          node = node.parentElement;
        }
        return parts.join('>');
      };
      const rows = [];
      const push = (key, value, el, method) => {
        key = String(key || '').trim();
        value = String(value || '').trim();
        if (!key || !value || key.length > 80 || value.length > 120) return;
        rows.push({
          key,
          value,
          display_text: key + ': ' + value,
          element_role: el.getAttribute('role') || el.tagName.toLowerCase(),
          element_id: el.id || el.getAttribute('name') || el.getAttribute('bid') || '',
          dom_path: path(el),
          visible: visible(el),
          extraction_method: method
        });
      };
      document.querySelectorAll('input, textarea, select').forEach((el) => {
        if (!visible(el)) return;
        const id = el.id;
        const label = id ? document.querySelector('label[for="' + CSS.escape(id) + '"]') : null;
        const key = (label && text(label)) || el.getAttribute('aria-label') || el.getAttribute('name') || el.id || el.placeholder;
        const value = el.tagName.toLowerCase() === 'select'
          ? (el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : el.value)
          : el.value;
        push(key, value, el, 'FORM_VALUE');
      });
      document.querySelectorAll('[data-field], [data-key], [data-label]').forEach((el) => {
        if (!visible(el)) return;
        const key = el.getAttribute('data-field') || el.getAttribute('data-key') || el.getAttribute('data-label');
        const value = el.getAttribute('data-value') || text(el);
        push(key, value, el, 'DATA_FIELD');
      });
      document.querySelectorAll('tr').forEach((tr) => {
        const cells = Array.from(tr.querySelectorAll('th,td')).filter(visible);
        if (cells.length >= 2) push(text(cells[0]), text(cells[1]), cells[1], 'TABLE_ROW');
      });
      const body = document.body ? text(document.body) : '';
      body.split(/\\n+/).slice(0, 80).forEach((line) => {
        const m = line.match(/^([A-Za-z가-힣][A-Za-z0-9가-힣 _/-]{1,40})\\s*[:：=]\\s*(.{1,80})$/);
        if (m) rows.push({key: m[1], value: m[2], display_text: line, element_role: 'text', element_id: '', dom_path: 'body', visible: true, extraction_method: 'TEXT_KEY_VALUE'});
      });
      return rows.slice(0, 120);
    }
    """
    rows = _safe_call_eval(page, script)
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)][:120]


def _safe_call_eval(page: Any, script: str) -> Any:
    if page is None:
        return None
    try:
        return page.evaluate(script)
    except Exception:
        return None


def _shallow_observation_copy(observation: Mapping[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            copied[key] = dict(value)
        elif isinstance(value, list):
            copied[key] = [dict(item) if isinstance(item, dict) else item for item in value]
        else:
            copied[key] = value
    return copied


def _cart_count_from_text(text: str) -> Optional[int]:
    match = re.search(r"\bcart\s*\(?\s*(\d{1,4})\s*\)?", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def normalize_action_error(error_text: Any) -> str:
    text = str(error_text or "").lower()
    if not text:
        return ""
    if "active page or bid not found" in text or "active page not found" in text or "bid not found" in text:
        return "target-not-resolvable"
    if "waiting for selector" in text or "waiting for locator" in text or "not found" in text or "unable to locate" in text:
        return "target-not-found"
    if "not visible" in text or "element is not visible" in text or "hidden" in text:
        return "target-hidden"
    if "disabled" in text:
        return "target-disabled"
    if "not editable" in text or "not an input" in text or "not fillable" in text:
        return "input-target-not-editable"
    if "not clickable" in text or "intercepts pointer events" in text or "outside of the viewport" in text:
        return "target-not-clickable"
    if "detached" in text or "not attached" in text:
        return "target-detached"
    if "timeout" in text:
        return "playwright-timeout"
    if "syntaxerror" in text or "selector" in text and "error" in text:
        return "selector-error"
    if "unsupported" in text:
        return "unsupported-action"
    if "bid" in text and ("not found" in text or "invalid" in text):
        return "target-not-resolvable"
    return "other-action-error"


def _first_error_text(info: Mapping[str, Any]) -> str:
    for key in ("action_error", "fallback_error", "fill_error", "press_error", "viewport_error", "network_error", "console_error"):
        value = info.get(key)
        if value:
            return str(value)
    return ""


def _has_internal_execution_error(info: Mapping[str, Any]) -> bool:
    return bool(_first_error_text(info))


def _short_error_summary(error_text: Any, limit: int = 240) -> str:
    text = " ".join(str(error_text or "").split())
    return text[:limit]


def _candidate_execution_summary(candidate: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
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
