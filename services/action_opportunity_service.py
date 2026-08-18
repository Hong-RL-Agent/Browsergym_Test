"""Policy-safe action opportunity generation from observed browser state.

The opportunity engine defines what can still be explored or verified. It does
not execute actions and reads only observed browser state.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


OPPORTUNITY_COVERAGE_THRESHOLD_DEFAULT = 0.7
COMMERCE_TOKENS = (
    "cart",
    "basket",
    "checkout",
    "payment",
    "purchase",
    "order",
    "product",
    "detail",
    "details",
    "quantity",
    "qty",
    "subtotal",
    "total",
    "장바구니",
    "카트",
    "담기",
    "결제",
    "주문",
    "주문하기",
    "구매",
    "상품",
    "제품",
    "상세",
    "자세히",
    "수량",
    "증가",
    "감소",
    "소계",
    "합계",
    "총액",
    "장바구니",
    "카트",
    "결제",
    "주문",
    "상품",
    "상세",
    "수량",
    "합계",
)
FORUM_TOKENS = (
    # Deliberately narrow to vocabulary that's distinctly forum/community --
    # the previous list also included "post", "comment", "title", "content",
    # "write", "save", "delete" (and their Korean equivalents), all of which
    # show up on completely ordinary pages (any "Save"/"Delete" button, any
    # settings form) and made this file classify almost any site with a save
    # or delete control as forum context, forcing its click_submit/
    # submit_form opportunities to be treated as required.
    "forumworks",
    "community",
    "forum",
    "게시글",
    "글쓰기",
    "댓글",
    "커뮤니티",
)
FORUM_KO_TOKENS = (
    "게시글",  # forum post
    "글쓰기",  # write post
    "댓글",  # comment
    "커뮤니티",  # community
)
FORUM_MORE_TEXT = "\ub354\ubcf4\uae30"
FORUM_SEARCH_PLACEHOLDER = "\uc81c\ubaa9, \uc791\uc131\uc790, \uce74\ud14c\uace0\ub9ac\ub85c \uac80\uc0c9"


def build_action_opportunities(
    observation: Mapping[str, Any],
    history: Optional[Mapping[str, Any]] = None,
    anomalies: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    coverage_threshold: float = OPPORTUNITY_COVERAGE_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    history = history or {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    candidates = candidates if isinstance(candidates, list) else []
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    page_url = str(page_state.get("url") or "") if isinstance(page_state, Mapping) else ""
    dom_signature = _dom_signature(observation)
    executed = {str(item) for item in history.get("executed_opportunity_ids", set()) or set()}
    executed_signatures = {str(item) for item in history.get("executed_action_signatures", set()) or set()}
    failed_signature_counts = history.get("failed_action_signature_counts", {}) if isinstance(history.get("failed_action_signature_counts"), Mapping) else {}
    blacklisted_signatures = {str(item) for item in history.get("blacklisted_action_signatures", set()) or set()}
    verified = {str(item) for item in history.get("verified_opportunity_ids", set()) or set()}
    skipped = {str(item) for item in history.get("skipped_opportunity_ids", set()) or set()}
    failed = {str(item) for item in history.get("failed_opportunity_ids", set()) or set()}
    selected = {str(item) for item in history.get("selected_opportunity_ids", set()) or set()}
    enabled = {str(item) for item in history.get("enabled_opportunity_ids", set()) or set()}
    masked = {str(item) for item in history.get("masked_opportunity_ids", set()) or set()}
    duplicate_of = history.get("duplicate_opportunity_of", {}) if isinstance(history.get("duplicate_opportunity_of"), Mapping) else {}
    failure_reasons = history.get("opportunity_failure_reasons", {}) if isinstance(history.get("opportunity_failure_reasons"), Mapping) else {}
    skip_reasons = history.get("opportunity_skip_reasons", {}) if isinstance(history.get("opportunity_skip_reasons"), Mapping) else {}
    selected_steps = history.get("opportunity_selected_steps", {}) if isinstance(history.get("opportunity_selected_steps"), Mapping) else {}
    execution_steps = history.get("opportunity_execution_steps", {}) if isinstance(history.get("opportunity_execution_steps"), Mapping) else {}
    verified_steps = history.get("opportunity_verified_steps", {}) if isinstance(history.get("opportunity_verified_steps"), Mapping) else {}
    opportunities: List[Dict[str, Any]] = []
    login_flow = history.get("login_flow", {}) if isinstance(history.get("login_flow"), Mapping) else {}
    login_form_observed = _has_login_form(observation, history)
    login_flow_completed = bool(login_flow.get("login_flow_completed") or login_flow.get("login_result_checked") or login_flow.get("submit_result_checked"))
    username_or_email_done = bool(login_flow.get("username_or_email_filled") or login_flow.get("email_input_filled"))
    password_done = bool(login_flow.get("password_filled") or login_flow.get("password_input_filled"))
    submit_done = bool(login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked"))
    commerce_surface = _has_commerce_surface(observation)
    forum_surface = _has_forum_surface(observation)
    has_checkout_candidate = False

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        commerce_required = bool(commerce_surface and _is_required_commerce_candidate(candidate))
        commerce_priority_boost = _commerce_priority_boost(candidate)
        forum_required = bool(forum_surface and _is_required_forum_candidate(candidate))
        forum_priority_boost = _forum_priority_boost(candidate) if forum_surface else 0.0
        if _is_checkout_candidate(candidate):
            has_checkout_candidate = True
        if _is_fillable(candidate):
            input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
            if input_type == "password" or bool(candidate.get("is_password")):
                required = bool(login_form_observed and not login_flow_completed and not password_done)
                opportunities.append(_candidate_opportunity("fill_password_input", candidate, index, page_url, dom_signature, required=required, priority=0.995 if required else 0.91))
                opportunities.append(_candidate_opportunity("fill_password", candidate, index, page_url, dom_signature, required=required, priority=0.99 if required else 0.9))
            elif input_type == "email":
                required = bool(login_form_observed and not login_flow_completed and not username_or_email_done)
                opportunities.append(_candidate_opportunity("fill_email_input", candidate, index, page_url, dom_signature, required=required, priority=0.985 if required else 0.86))
                opportunities.append(_candidate_opportunity("fill_username_or_email", candidate, index, page_url, dom_signature, required=required, priority=0.98 if required else 0.85))
            elif (
                input_type == "search"
                or str(candidate.get("role") or "").lower() == "searchbox"
                # A search box frequently carries no type/role at all -- only a
                # placeholder like "찾고 싶은 상품을 검색하세요" -- which the checks
                # above miss entirely. Without this, such a field falls through
                # to _is_username_or_email_candidate below (which treats any
                # untyped text input as a login field by default) and gets
                # mislabeled fill_username_or_email instead of a search fill.
                or _candidate_text_has(candidate, ("search", "검색", "찾기", "찾아보기"))
            ):
                # Same reasoning as the generic fill_text_input branch below:
                # scarce enough per page that requiring it doesn't reproduce
                # the old over-blocking problem, and search-no-effect can
                # only ever be observed by actually using it.
                opportunities.append(_candidate_opportunity("fill_search_input", candidate, index, page_url, dom_signature, required=True, priority=0.82))
            elif _is_username_or_email_candidate(candidate):
                # required unconditionally, not just when a login form is
                # recognized: this heuristic can match a plain
                # username/email-shaped field outside any login context too,
                # and it's still worth an actual fill_input either way.
                login_required = bool(login_form_observed and not login_flow_completed and not username_or_email_done)
                opportunities.append(_candidate_opportunity("fill_username_or_email", candidate, index, page_url, dom_signature, required=True, priority=0.98 if login_required else 0.85))
                opportunities.append(_candidate_opportunity("fill_text_input", candidate, index, page_url, dom_signature, required=True, priority=0.84))
            else:
                # A generic fillable field was previously only required under
                # commerce/forum context, so on any other kind of site (an
                # admin dashboard, a settings page) the policy could finish
                # having never used fill_input at all -- confirmed on a real
                # scan where fillable_count stayed >=1 for all 60 steps but
                # fill_input was never once selected. functional_priority_
                # candidate doesn't help here (it requires clickable=True,
                # which plain <input> fields generally aren't), so just
                # require any fillable field reaching this branch directly --
                # unlike clickable elements, fillable fields are scarce
                # enough per page (typically 1-2) that this doesn't reproduce
                # the old "blocks finish on every incidental optional field"
                # problem.
                required = bool(commerce_required or forum_required)
                priority = (0.97 if forum_required else 0.93) + commerce_priority_boost + forum_priority_boost
                opportunities.append(_candidate_opportunity("fill_text_input", candidate, index, page_url, dom_signature, required=required, priority=priority))
        if _is_login_submit(candidate):
            required = bool(login_form_observed and not login_flow_completed and not submit_done)
            opportunities.append(_candidate_opportunity("click_submit", candidate, index, page_url, dom_signature, required=required, priority=0.975 if required else 0.91))
            opportunities.append(_candidate_opportunity("click_login_submit", candidate, index, page_url, dom_signature, required=required, priority=0.97 if required else 0.9))
            opportunities.append(_candidate_opportunity("submit_form", candidate, index, page_url, dom_signature, required=required, priority=0.9))
        elif _is_submit(candidate):
            required = bool((login_form_observed and not login_flow_completed and not submit_done) or commerce_required)
            opportunities.append(_candidate_opportunity("click_submit", candidate, index, page_url, dom_signature, required=required, priority=(0.96 if required else 0.92) + commerce_priority_boost))
            opportunities.append(_candidate_opportunity("submit_form", candidate, index, page_url, dom_signature, required=required, priority=(0.94 if required else 0.9) + commerce_priority_boost))
        elif forum_surface and _is_forum_submit_candidate(candidate):
            opportunities.append(_candidate_opportunity("click_submit", candidate, index, page_url, dom_signature, required=True, priority=0.99 + forum_priority_boost))
            opportunities.append(_candidate_opportunity("submit_form", candidate, index, page_url, dom_signature, required=True, priority=0.98 + forum_priority_boost))
        elif _is_select(candidate):
            opportunities.append(_candidate_opportunity("select_option", candidate, index, page_url, dom_signature, required=False, priority=0.66))
        elif _is_link(candidate):
            required = bool(commerce_required or forum_required or _is_functional_priority_candidate(candidate))
            opportunities.append(_candidate_opportunity("click_link", candidate, index, page_url, dom_signature, required=required, priority=(0.9 if required else 0.45) + commerce_priority_boost + forum_priority_boost))
        elif _is_clickable(candidate):
            # functional_priority_candidate is the observation adapter's own flag for
            # elements it judged worth interacting with (recognized semantic type:
            # add-to-cart, quantity control, etc.) that aren't already covered by the
            # commerce/forum surface heuristics above -- e.g. an admin dashboard or a
            # plain form with no commerce/forum vocabulary in its text. Without this,
            # such candidates were only ever enforced by a separate ad-hoc mask check
            # instead of flowing through this same required-opportunity pipeline.
            required = bool(commerce_required or forum_required or _is_functional_priority_candidate(candidate))
            opportunities.append(_candidate_opportunity("click_button", candidate, index, page_url, dom_signature, required=required, priority=(0.95 if forum_required else 0.92 if required else 0.6) + commerce_priority_boost + forum_priority_boost))

    page_type = _page_type(observation, history)
    has_network = _has_network(runtime)
    has_json = _has_json_response(observation, history)
    has_console = _console_error_count(runtime) > 0
    if page_type in {"api_json_page", "data_endpoint_page"} or has_network:
        opportunities.append(_global_opportunity("inspect_network", page_url, dom_signature, required=page_type in {"api_json_page", "data_endpoint_page"}, priority=0.9))
        opportunities.append(_global_opportunity("check_api_response", page_url, dom_signature, required=page_type in {"api_json_page", "data_endpoint_page"}, priority=0.95))
    if page_type in {"api_json_page", "data_endpoint_page"} or has_json:
        opportunities.append(_global_opportunity("check_json_schema", page_url, dom_signature, required=page_type in {"api_json_page", "data_endpoint_page"}, priority=0.85))
    if has_console or page_type in {"api_json_page", "data_endpoint_page"}:
        opportunities.append(_global_opportunity("inspect_console", page_url, dom_signature, required=has_console, priority=0.98 if has_console else 0.65))
    if has_console:
        opportunities.append(_global_opportunity("verify_runtime_error", page_url, dom_signature, required=True, priority=0.99))
    if _is_api_or_documentation_page(observation, history):
        opportunities.append(_global_opportunity("inspect_network", page_url, dom_signature, required=True, priority=0.92))
        opportunities.append(_global_opportunity("inspect_console", page_url, dom_signature, required=True, priority=0.82))
        opportunities.append(_global_opportunity("check_api_response", page_url, dom_signature, required=True, priority=0.96))
        if _has_endpoint_link(candidates):
            opportunities.append(_global_opportunity("check_api_endpoint_link", page_url, dom_signature, required=False, priority=0.72))
    if _has_search_input(candidates):
        opportunities.append(_global_opportunity("check_search_result_after_input", page_url, dom_signature, required=False, priority=0.74))
    if bool(login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked")) and not login_flow_completed:
        opportunities.append(_global_opportunity("verify_login_result", page_url, dom_signature, required=True, priority=0.97))
    if has_checkout_candidate:
        opportunities.append(_global_opportunity("inspect_network", page_url, dom_signature, required=False, priority=0.88))
    if history.get("last_action_failed"):
        context = str(history.get("last_action_signature") or dom_signature)
        opportunities.append(_global_opportunity("verify_action_result", page_url, dom_signature, required=True, priority=1.0, context_key=context))
        opportunities.append(_global_opportunity("inspect_network", page_url, dom_signature, required=True, priority=0.95))
        opportunities.append(_global_opportunity("inspect_console", page_url, dom_signature, required=True, priority=0.9))
    dom_context = _dom_context_key(observation)
    inspected_dom_contexts = {str(item) for item in history.get("inspect_dom_contexts_inspected", set()) or set()}
    if dom_context not in inspected_dom_contexts:
        opportunities.append(_global_opportunity("inspect_dom", page_url, dom_signature, required=False, priority=0.35, context_key=dom_context))
    layout_context = _layout_context_key(observation)
    if layout_context not in {str(item) for item in history.get("layout_contexts_inspected", set()) or set()}:
        opportunities.append(_global_opportunity("inspect_layout", page_url, dom_signature, required=False, priority=0.35, context_key=layout_context))
    viewport_context = _viewport_context_key(observation)
    if viewport_context not in {str(item) for item in history.get("viewport_contexts_changed", set()) or set()}:
        opportunities.append(_global_opportunity("change_viewport_once", page_url, dom_signature, required=commerce_surface, priority=0.96 if commerce_surface else 0.3, context_key=viewport_context))

    verified_anomaly_keys = {str(item) for item in history.get("verified_anomaly_keys", set()) or set()}
    verified_anomaly_type_contexts = {
        str(item) for item in history.get("verified_anomaly_type_contexts", set()) or set()
    }
    verified_anomaly_types = {str(item) for item in history.get("verified_anomaly_types", set()) or set()}
    inspected_dom_contexts_for_anomaly = {str(item) for item in history.get("inspect_dom_contexts_inspected", set()) or set()}
    dom_verification_already_performed = _dom_context_key(observation) in inspected_dom_contexts_for_anomaly
    for anomaly in anomalies or history.get("unverified_anomalies", []) or []:
        if isinstance(anomaly, Mapping):
            if not _anomaly_blocks_finish(anomaly):
                continue
            if _anomaly_key(anomaly) in verified_anomaly_keys:
                continue
            anomaly_type = str(anomaly.get("type") or "")
            if anomaly_type in verified_anomaly_types:
                continue
            if _anomaly_type_context_key(anomaly, page_url) in verified_anomaly_type_contexts:
                continue
            if dom_verification_already_performed and anomaly_type not in {"console-error", "runtime-exception", "network-error"}:
                continue
            if _dom_verification_loop_guard_active(history) and anomaly_type not in {"console-error", "runtime-exception", "network-error", "api-5xx", "api-4xx"}:
                continue
            if anomaly_type == "form-no-feedback":
                opportunities.append(_global_opportunity("verify_form_no_feedback", page_url, dom_signature, required=True, priority=1.0, context_key=_anomaly_key(anomaly)))
            elif anomaly_type == "action-error":
                opportunities.append(_global_opportunity("verify_action_result", page_url, dom_signature, required=True, priority=1.0, context_key=_anomaly_key(anomaly)))
            elif anomaly_type == "duplicated-rendering":
                opportunities.append(_global_opportunity("verify_duplicated_rendering", page_url, dom_signature, required=True, priority=1.0, context_key=_anomaly_key(anomaly)))
            else:
                opportunities.append(_global_opportunity("verify_anomaly_reproduction", page_url, dom_signature, required=True, priority=1.0, context_key=_anomaly_key(anomaly)))

    deduped = _cap_required_opportunities(_dedupe_opportunities(opportunities))
    for opportunity in deduped:
        oid = str(opportunity["opportunity_id"])
        signature = str(opportunity.get("action_signature") or "")
        signature_executed = bool(signature and signature in executed_signatures)
        signature_blacklisted = bool(signature and (signature in blacklisted_signatures or int(failed_signature_counts.get(signature, 0) or 0) >= 2))
        opportunity["executed"] = oid in executed or signature_executed
        opportunity["verified"] = oid in verified
        opportunity["failed"] = oid in failed
        opportunity["selected"] = oid in selected
        opportunity["enabled"] = oid in enabled or oid not in masked
        opportunity["masked"] = oid in masked
        opportunity["skipped"] = oid in skipped
        opportunity["skipped_reason"] = str(
            skip_reasons.get(oid)
            or ("failed_action_signature_blacklisted" if signature_blacklisted else "")
            or ("explicitly_skipped" if oid in skipped else "")
        )
        opportunity["skip_reason"] = opportunity["skipped_reason"]
        opportunity["failure_reason"] = str(failure_reasons.get(oid) or "")
        opportunity["duplicate_of"] = str(duplicate_of.get(oid) or "")
        opportunity["duplicate"] = bool(opportunity["duplicate_of"])
        opportunity["success"] = bool(opportunity["verified"])
        opportunity["status"] = _opportunity_status(opportunity)
        opportunity["selected_step"] = int(selected_steps.get(oid, 0) or 0)
        opportunity["executed_step"] = int(execution_steps.get(oid, opportunity.get("executed_step", 0)) or 0)
        opportunity["verified_step"] = int(verified_steps.get(oid, 0) or 0)

    required = [item for item in deduped if bool(item.get("required"))]
    optional = [item for item in deduped if not bool(item.get("required"))]
    executed_count = sum(1 for item in deduped if item.get("executed") or item.get("verified"))
    verified_count = sum(1 for item in deduped if item.get("verified"))
    skipped_count = sum(1 for item in deduped if item.get("skipped") or item.get("skipped_reason"))
    failed_count = sum(1 for item in deduped if item.get("failed"))
    remaining_required = [
        item for item in required if not _required_opportunity_satisfied(item)
    ]
    remaining = [
        item for item in deduped if not (item.get("executed") or item.get("verified") or item.get("skipped_reason"))
    ]
    blocking_remaining = [
        item
        for item in remaining
        if bool(item.get("required")) or _optional_opportunity_blocks_finish(item)
    ]
    coverage_rate = _safe_rate(executed_count, len(deduped))
    required_completion_rate = _safe_rate(len(required) - len(remaining_required), len(required)) if required else 1.0
    optional_completed_count = sum(1 for item in optional if item.get("executed") or item.get("verified") or item.get("skipped_reason"))
    optional_coverage_rate = _safe_rate(optional_completed_count, len(optional)) if optional else 1.0
    anomaly_verification_opportunities = [
        item
        for item in deduped
        if str(item.get("opportunity_type") or "").startswith("verify_")
    ]
    unverified_anomaly_count = sum(1 for item in anomaly_verification_opportunities if not item.get("verified"))
    verified_finding_count = sum(1 for item in anomaly_verification_opportunities if item.get("verified"))
    required_work_done = bool(
        not remaining_required
        and unverified_anomaly_count == 0
        and (not login_form_observed or bool(login_flow.get("login_flow_completed")))
    )
    has_historical_opportunity_action = bool(
        executed
        or verified
        or selected
        or history.get("last_action_type")
        or int(history.get("step_index", 0) or 0) > 0
    )
    has_any_opportunity_action = bool(
        executed_count > 0
        or verified_count > 0
        or skipped_count > 0
        or has_historical_opportunity_action
        or len(deduped) == 0
    )
    all_observed_opportunities_done = len(blocking_remaining) == 0
    finish_allowed = bool(required_work_done and has_any_opportunity_action and all_observed_opportunities_done)
    finish_blocked_reason = _finish_blocked_reason(
        remaining_required_count=len(remaining_required),
        remaining_opportunity_count=len(remaining),
        blocking_opportunity_count=len(blocking_remaining),
        coverage_rate=coverage_rate,
        coverage_threshold=coverage_threshold,
        unverified_anomaly_count=unverified_anomaly_count,
        has_any_opportunity_action=has_any_opportunity_action,
        finish_allowed=finish_allowed,
    )
    diagnostics = _opportunity_diagnostics(
        candidate_count=len(candidates),
        opportunities=deduped,
        has_console=has_console,
        anomalies=list(anomalies or history.get("unverified_anomalies", []) or []),
        history=history,
    )
    return {
        "opportunities": deduped,
        "summary": {
            "detected_candidate_count": len(candidates),
            "generated_opportunity_count": len(deduped),
            "required_opportunity_count": len(required),
            "optional_opportunity_count": len(optional),
            "executed_opportunity_count": executed_count,
            "verified_opportunity_count": verified_count,
            "skipped_opportunity_count": skipped_count,
            "failed_opportunity_count": failed_count,
            "remaining_opportunity_count": len(remaining),
            "pending_opportunity_count": len(remaining),
            "remaining_blocking_opportunity_count": len(blocking_remaining),
            "pending_blocking_opportunity_count": len(blocking_remaining),
            "remaining_required_opportunity_count": len(remaining_required),
            "pending_required_opportunity_count": len(remaining_required),
            "action_opportunity_coverage_rate": coverage_rate,
            "action_coverage_rate": coverage_rate,
            "required_opportunity_completion_rate": required_completion_rate,
            "optional_opportunity_coverage_rate": optional_coverage_rate,
            "unverified_anomaly_count": unverified_anomaly_count,
            "verified_finding_count": verified_finding_count,
            "anomaly_verification_required": unverified_anomaly_count > 0,
            "anomaly_verification_completed": unverified_anomaly_count == 0,
            "finish_allowed": finish_allowed,
            "finish_blocked_reason": finish_blocked_reason,
            "valid_scan_run": finish_allowed,
            "has_any_opportunity_action": has_any_opportunity_action,
            "all_observed_opportunities_done": all_observed_opportunities_done,
            "optional_opportunity_coverage_below_threshold": bool(
                coverage_rate < float(coverage_threshold) and len(deduped) > 2
            ),
            "dom_exhaustive_action_mode": True,
            "coverage_threshold": float(coverage_threshold),
            "opportunity_diagnostics": diagnostics,
            "page_type": page_type,
            **_login_flow_summary(login_flow, login_form_observed, observation),
        },
    }


def update_opportunity_history(
    history: Dict[str, Any],
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    step: int = 0,
    success: bool = False,
    anomalies: Optional[Iterable[Mapping[str, Any]]] = None,
) -> None:
    snapshot = build_action_opportunities(observation, history, anomalies)
    action_type = str(action.get("action_type") or "")
    candidate_index = int(action.get("candidate_index", 0) or 0)
    matched = [
        item for item in snapshot["opportunities"]
        if action_type in _opportunity_action_types(item) and int(item.get("candidate_index", 0) or 0) == candidate_index
    ]
    if action_type in {"inspect_network", "inspect_console", "inspect_dom", "inspect_layout", "inspect_last_api_response", "validate_last_api_schema", "change_viewport_mobile", "change_viewport_desktop"}:
        matched = [item for item in snapshot["opportunities"] if action_type in _opportunity_action_types(item)]
    history["last_action_failed"] = not bool(success)
    if action.get("action_signature"):
        history["last_action_signature"] = str(action.get("action_signature"))
        if bool(success):
            history.setdefault("executed_action_signatures", set()).add(str(action.get("action_signature")))
        else:
            counts = history.setdefault("failed_action_signature_counts", {})
            if isinstance(counts, dict):
                signature = str(action.get("action_signature"))
                counts[signature] = int(counts.get(signature, 0) or 0) + 1
                if counts[signature] >= 2:
                    history.setdefault("blacklisted_action_signatures", set()).add(signature)
    if action_type == "inspect_dom":
        dom_context = _dom_context_key(observation)
        inspected_dom_contexts = history.setdefault("inspect_dom_contexts_inspected", set())
        if isinstance(inspected_dom_contexts, set):
            if dom_context in inspected_dom_contexts:
                history["repeated_inspect_dom_count"] = int(history.get("repeated_inspect_dom_count", 0) or 0) + 1
                history["repeated_inspect_dom_penalty_applied"] = int(history.get("repeated_inspect_dom_penalty_applied", 0) or 0) + 1
            inspected_dom_contexts.add(dom_context)
    if anomalies:
        stored = history.setdefault("unverified_anomalies", [])
        if isinstance(stored, list):
            existing = {_anomaly_key(item) for item in stored if isinstance(item, Mapping)}
            verified_anomaly_keys = {str(item) for item in history.get("verified_anomaly_keys", set()) or set()}
            page_url_for_anomaly = str(observation.get("page_state", {}).get("url", "") if isinstance(observation.get("page_state", {}), Mapping) else "")
            verified_anomaly_type_contexts = {
                str(item) for item in history.get("verified_anomaly_type_contexts", set()) or set()
            }
            verified_anomaly_types = {str(item) for item in history.get("verified_anomaly_types", set()) or set()}
            for anomaly in anomalies:
                anomaly_key = _anomaly_key(anomaly) if isinstance(anomaly, Mapping) else ""
                anomaly_type_context = _anomaly_type_context_key(anomaly, page_url_for_anomaly) if isinstance(anomaly, Mapping) else ""
                anomaly_type = str(anomaly.get("type") or "") if isinstance(anomaly, Mapping) else ""
                if (
                    isinstance(anomaly, Mapping)
                    and _anomaly_blocks_finish(anomaly)
                    and anomaly_key not in existing
                    and anomaly_key not in verified_anomaly_keys
                    and anomaly_type_context not in verified_anomaly_type_contexts
                    and anomaly_type not in verified_anomaly_types
                ):
                    stored.append(dict(anomaly))
                    existing.add(anomaly_key)
    if bool(success) and action_type in {
        "inspect_dom",
        "inspect_layout",
        "inspect_network",
        "inspect_console",
        "inspect_last_api_response",
        "validate_last_api_schema",
    }:
        _mark_observed_anomalies_verified(history, anomalies or [], page_url=str(observation.get("page_state", {}).get("url", "") if isinstance(observation.get("page_state", {}), Mapping) else ""))
    executed_ids = history.setdefault("executed_opportunity_ids", set())
    verified_ids = history.setdefault("verified_opportunity_ids", set())
    selected_ids = history.setdefault("selected_opportunity_ids", set())
    failed_ids = history.setdefault("failed_opportunity_ids", set())
    for item in matched:
        oid = str(item.get("opportunity_id") or "")
        if not oid:
            continue
        selected_ids.add(oid)
        executed_ids.add(oid)
        current_step = int(step or history.get("step_index", 0) or 0)
        history.setdefault("opportunity_selected_steps", {})[oid] = current_step
        if success:
            verified_ids.add(oid)
            if item.get("action_signature"):
                history.setdefault("executed_action_signatures", set()).add(str(item.get("action_signature")))
            history.setdefault("opportunity_verified_steps", {})[oid] = current_step
            _mark_related_anomalies_verified(history, item)
        else:
            failed_ids.add(oid)
            history.setdefault("opportunity_failure_reasons", {})[oid] = _failure_reason(action)
            signature = str(item.get("action_signature") or action.get("action_signature") or "")
            if signature:
                counts = history.setdefault("failed_action_signature_counts", {})
                if isinstance(counts, dict):
                    counts[signature] = int(counts.get(signature, 0) or 0) + 1
                    if counts[signature] >= 2:
                        history.setdefault("blacklisted_action_signatures", set()).add(signature)
        history.setdefault("opportunity_execution_steps", {})[oid] = current_step
        if item.get("opportunity_type") == "inspect_layout" and item.get("element_context_key"):
            history.setdefault("layout_contexts_inspected", set()).add(str(item.get("element_context_key")))
        if item.get("opportunity_type") == "change_viewport_once":
            history.setdefault("viewport_contexts_changed", set()).add(str(item.get("element_context_key") or _viewport_context_key(observation)))


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


def action_ids_for_opportunities(opportunities: Iterable[Mapping[str, Any]], action_space: Any) -> List[int]:
    action_ids: List[int] = []
    for opportunity in opportunities:
        action_types = _opportunity_action_types(opportunity)
        if not action_types:
            continue
        if isinstance(action_types, str):
            action_types = (action_types,)
        for action_type in action_types:
            try:
                action_ids.append(action_space.encode(action_type, int(opportunity.get("candidate_index", 0) or 0)))
            except Exception:
                continue
    return action_ids


def _required_opportunity_satisfied(opportunity: Mapping[str, Any]) -> bool:
    if opportunity.get("skipped_reason"):
        return True
    if opportunity.get("verified"):
        return True
    if opportunity.get("failed"):
        return False
    return bool(opportunity.get("executed") and opportunity.get("success"))


def _candidate_opportunity(opportunity_type: str, candidate: Mapping[str, Any], index: int, page_url: str, dom_signature: str, *, required: bool, priority: float) -> Dict[str, Any]:
    element_key = _candidate_key(candidate) or f"candidate-{index}"
    opportunity = _base_opportunity(
        opportunity_type,
        page_url,
        dom_signature,
        required=required,
        priority=priority,
        candidate_index=index,
        target_element_key=element_key,
        target_selector=str(candidate.get("selector") or candidate.get("locator") or ""),
        target_role=str(candidate.get("role") or ""),
        target_text=str(candidate.get("text") or ""),
        target_name=str(candidate.get("name") or ""),
    )
    opportunity.update(
        {
            "target_tag": str(candidate.get("tag") or ""),
            "target_type": str(candidate.get("type") or candidate.get("input_type") or ""),
            "target_placeholder": str(candidate.get("placeholder") or ""),
            "visible": bool(candidate.get("visible", True)),
            "enabled": bool(candidate.get("enabled", True)),
            "clickable": bool(candidate.get("clickable", False)),
            "fillable": bool(candidate.get("fillable", False)),
            "is_password": bool(candidate.get("is_password", False)),
            "is_submit": bool(candidate.get("is_submit", False)),
            "is_form_control": bool(candidate.get("is_form_control", False)),
            "bounding_box": candidate.get("bounding_box", candidate.get("bbox", {})),
            "semantic_action_type": str(candidate.get("semantic_action_type") or ""),
        }
    )
    opportunity["action_signature"] = _opportunity_signature(opportunity)
    return opportunity


def _global_opportunity(opportunity_type: str, page_url: str, dom_signature: str, *, required: bool, priority: float, context_key: str = "") -> Dict[str, Any]:
    opportunity = _base_opportunity(opportunity_type, page_url, dom_signature, required=required, priority=priority, candidate_index=0, target_element_key="", target_selector="", target_role="", target_text="", target_name="", context_key=context_key)
    opportunity["action_signature"] = _opportunity_signature(opportunity)
    return opportunity


def _base_opportunity(opportunity_type: str, page_url: str, dom_signature: str, *, required: bool, priority: float, candidate_index: int, target_element_key: str, target_selector: str, target_role: str, target_text: str, target_name: str, context_key: str = "") -> Dict[str, Any]:
    context = context_key or target_element_key or dom_signature
    opportunity_id = _hash("|".join([opportunity_type, target_element_key, page_url, context]))
    return {
        "opportunity_id": opportunity_id,
        "opportunity_type": opportunity_type,
        "candidate_index": int(candidate_index),
        "target_element_key": target_element_key,
        "target_selector": target_selector,
        "target_role": target_role,
        "target_text": target_text,
        "target_name": target_name,
        "page_url": page_url,
        "dom_signature": dom_signature,
        "element_context_key": context,
        "priority": float(priority),
        "required": bool(required),
        "status": "created",
        "enabled": True,
        "masked": False,
        "selected": False,
        "executed": False,
        "success": False,
        "failed": False,
        "verified": False,
        "skipped": False,
        "skipped_reason": "",
        "skip_reason": "",
        "failure_reason": "",
        "duplicate": False,
        "duplicate_of": "",
        "created_step": 0,
        "selected_step": 0,
        "executed_step": 0,
        "verified_step": 0,
        "source_observation_id": dom_signature,
        "action_signature": "",
    }


def _dedupe_opportunities(opportunities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for opportunity in opportunities:
        oid = str(opportunity.get("opportunity_id") or "")
        if not oid:
            continue
        existing = seen.get(oid)
        if existing is None or float(opportunity.get("priority", 0.0) or 0.0) > float(existing.get("priority", 0.0) or 0.0):
            seen[oid] = dict(opportunity)
    return sorted(seen.values(), key=lambda item: (not bool(item.get("required")), -float(item.get("priority", 0.0) or 0.0), str(item.get("opportunity_type") or "")))


def _cap_required_opportunities(opportunities: List[Dict[str, Any]], *, max_required: int = 8) -> List[Dict[str, Any]]:
    required = [item for item in opportunities if bool(item.get("required"))]
    if len(required) <= max_required:
        return opportunities
    critical_types = {
        "fill_email_input",
        "fill_password_input",
        "fill_password",
        "click_login_submit",
        "verify_login_result",
        "verify_runtime_error",
        "verify_action_result",
        "verify_form_no_feedback",
        "verify_duplicated_rendering",
        "verify_anomaly_reproduction",
        "inspect_network",
        "inspect_console",
        "check_api_response",
        "check_json_schema",
    }
    ranked = sorted(
        required,
        key=lambda item: (
            str(item.get("opportunity_type") or "") not in critical_types,
            -float(item.get("priority", 0.0) or 0.0),
            str(item.get("action_signature") or item.get("opportunity_id") or ""),
        ),
    )
    keep_ids = {str(item.get("opportunity_id") or "") for item in ranked[:max_required]}
    capped: List[Dict[str, Any]] = []
    for item in opportunities:
        row = dict(item)
        if bool(row.get("required")) and str(row.get("opportunity_id") or "") not in keep_ids:
            row["required"] = False
            row["required_demoted_reason"] = "required_opportunity_cap"
        capped.append(row)
    return sorted(capped, key=lambda item: (not bool(item.get("required")), -float(item.get("priority", 0.0) or 0.0), str(item.get("opportunity_type") or "")))


def _finish_blocked_reason(*, remaining_required_count: int, remaining_opportunity_count: int, blocking_opportunity_count: int, coverage_rate: float, coverage_threshold: float, unverified_anomaly_count: int, has_any_opportunity_action: bool, finish_allowed: bool) -> str:
    if finish_allowed:
        return ""
    if unverified_anomaly_count > 0:
        return "unverified_anomaly_remaining"
    if remaining_required_count > 0:
        return "required_opportunity_remaining"
    if not has_any_opportunity_action:
        return "no_opportunity_action_executed"
    if blocking_opportunity_count > 0:
        return "observed_actionable_opportunity_remaining"
    return "finish_not_allowed"


def _optional_opportunity_blocks_finish(opportunity: Mapping[str, Any]) -> bool:
    opportunity_type = str(opportunity.get("opportunity_type") or "")
    if bool(opportunity.get("required")):
        return True
    if opportunity_type.startswith("verify_"):
        return True
    return False


def _anomaly_blocks_finish(anomaly: Mapping[str, Any]) -> bool:
    """Only strong, actionable anomalies should block episode completion.

    Weak needs-review observations are still reported by the evaluator, but if
    every low-confidence/noisy candidate creates a required verify_* action the
    agent never reaches a valid finish state. Keep the finish gate for findings
    with objective evidence or clear domain invariants.
    """
    if not isinstance(anomaly, Mapping):
        return False
    status = str(anomaly.get("human_review_status") or "")
    if status == "likely_false_positive":
        return False
    if "confidence" not in anomaly:
        return True
    anomaly_type = str(anomaly.get("type") or "")
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    if status == "likely_true_positive":
        return False
    if anomaly_type in {
        "api-5xx",
        "network-error",
        "runtime-exception",
        "console-error",
        "cart-quantity-mismatch",
        "cart-total-mismatch",
        "product-detail-mismatch",
        "api-ui-mismatch",
        "weak-password-validation",
        "duplicate-submission",
    }:
        return False
    if anomaly_type == "layout-overlap":
        return False
    if anomaly_type == "button-no-response":
        target = evidence.get("target") if isinstance(evidence.get("target"), Mapping) else {}
        semantic = str(target.get("semantic_action_type") or evidence.get("semantic_action_type") or "").lower()
        target_text = " ".join(
            str(value or "")
            for value in (
                evidence.get("clicked_text"),
                evidence.get("clicked_name"),
                target.get("text") if isinstance(target, Mapping) else "",
                target.get("name") if isinstance(target, Mapping) else "",
            )
        ).lower()
        meaningful_target = semantic in {"add", "cart", "checkout", "submit", "save", "detail", "details"} or any(
            token in target_text for token in ("add", "cart", "checkout", "submit", "save", "order", "buy", "detail")
        )
        no_observable_change = not any(
            bool(evidence.get(key))
            for key in ("route_changed", "cart_state_changed", "toast_visible", "modal_opened", "form_opened")
        )
        return False
    return False


def _opportunity_diagnostics(
    *,
    candidate_count: int,
    opportunities: list[Mapping[str, Any]],
    has_console: bool,
    anomalies: list[Any],
    history: Mapping[str, Any],
) -> list[str]:
    diagnostics: list[str] = []
    types = {str(item.get("opportunity_type") or "") for item in opportunities if isinstance(item, Mapping)}
    if candidate_count >= 10 and len(opportunities) < max(4, min(candidate_count // 4, 12)):
        diagnostics.append("candidates_exist_but_low_opportunity_generation")
    if has_console and "inspect_console" not in types:
        diagnostics.append("required_opportunity_not_created_for_console_error")
    anomaly_types = {str(item.get("type") or "") for item in anomalies if isinstance(item, Mapping)}
    if anomaly_types and not any(item.startswith("verify_") for item in types):
        diagnostics.append("anomaly_detected_but_no_verification_opportunity")
    if int(history.get("repeated_inspect_dom_count", 0) or 0) > 0:
        diagnostics.append("repeated_dom_inspection_without_new_signal")
    return diagnostics


def _is_fillable(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("fillable"))


def _is_submit(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("is_submit")) or str(candidate.get("type") or candidate.get("input_type") or "").lower() == "submit"


def _is_select(candidate: Mapping[str, Any]) -> bool:
    return str(candidate.get("tag") or "").lower() == "select" or str(candidate.get("role") or "").lower() in {"combobox", "listbox"}


def _is_login_submit(candidate: Mapping[str, Any]) -> bool:
    if not _is_submit(candidate):
        return False
    return _candidate_text_has(candidate, ("login", "log in", "sign in", "signin", "submit"))


def _is_link(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("href")) or str(candidate.get("role") or "").lower() == "link" or str(candidate.get("tag") or "").lower() == "a"


def _is_clickable(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("clickable"))


def _is_functional_priority_candidate(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority"))


def _candidate_text_has(candidate: Mapping[str, Any], tokens: Iterable[str]) -> bool:
    text = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "placeholder", "title", "id")).lower()
    return any(token in text for token in tokens)


def _has_commerce_surface(observation: Mapping[str, Any]) -> bool:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    url = str(page_state.get("url") or "").lower() if isinstance(page_state, Mapping) else ""
    title = str(page_state.get("title") or "").lower() if isinstance(page_state, Mapping) else ""
    if any(token in " ".join([url, title]) for token in ("fakeapi", "swagger", "api-docs", "documentation")):
        return False
    parts = [
        url,
        title,
        str(page_state.get("page_text_sample") or page_state.get("page_text") or "") if isinstance(page_state, Mapping) else "",
    ]
    if isinstance(candidates, list):
        for candidate in candidates[:32]:
            if isinstance(candidate, Mapping):
                parts.append(_candidate_text(candidate))
                parts.append(str(candidate.get("semantic_action_type") or ""))
    haystack = " ".join(parts).lower()
    return any(token.lower() in haystack for token in COMMERCE_TOKENS)


def _has_forum_surface(observation: Mapping[str, Any]) -> bool:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    parts = [
        str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        str(page_state.get("title") or "") if isinstance(page_state, Mapping) else "",
        str(page_state.get("page_text_sample") or page_state.get("page_text") or "") if isinstance(page_state, Mapping) else "",
    ]
    if isinstance(candidates, list):
        for candidate in candidates[:48]:
            if isinstance(candidate, Mapping):
                parts.append(_candidate_text(candidate))
                parts.append(str(candidate.get("semantic_action_type") or ""))
    haystack = " ".join(parts).lower()
    return any(token.lower() in haystack for token in FORUM_TOKENS + FORUM_KO_TOKENS)


def _is_required_forum_candidate(candidate: Mapping[str, Any]) -> bool:
    if _is_forum_noise_candidate(candidate):
        return False
    return bool(
        _is_forum_write_candidate(candidate)
        or _is_forum_submit_candidate(candidate)
        or _is_forum_delete_candidate(candidate)
        or _is_forum_fill_candidate(candidate)
        or _is_forum_detail_candidate(candidate)
    )


def _forum_priority_boost(candidate: Mapping[str, Any]) -> float:
    if _is_forum_submit_candidate(candidate):
        return 0.12
    if _is_forum_fill_candidate(candidate):
        return 0.1
    if _is_forum_write_candidate(candidate) or _is_forum_detail_candidate(candidate):
        return 0.08
    if _is_forum_delete_candidate(candidate):
        return 0.07
    return 0.0


def _is_forum_noise_candidate(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate)
    normalized = " ".join(text.split()).strip().lower()
    role = str(candidate.get("role") or "").lower()
    if role in {"searchbox", "textbox"} and "검색" in normalized:
        return True
    return normalized in {
        FORUM_MORE_TEXT,
        "more",
        "forumworks",
        f"forumworks {FORUM_KO_TOKENS[-1]}",
        f"forumworks {FORUM_KO_TOKENS[-1]} forumworks{FORUM_KO_TOKENS[-1]}",
        FORUM_KO_TOKENS[-1],
    } or FORUM_SEARCH_PLACEHOLDER in normalized


def _is_forum_write_candidate(candidate: Mapping[str, Any]) -> bool:
    return _candidate_text_has(candidate, ("\uae00\uc4f0\uae30", "\uc0c8 \uae00", "write post", "new post", "\uc791\uc131"))


def _is_forum_submit_candidate(candidate: Mapping[str, Any]) -> bool:
    return _candidate_text_has(candidate, ("\uc800\uc7a5", "\ub4f1\ub85d", "\uc791\uc131 \uc644\ub8cc", "\ub313\uae00 \uc791\uc131", "\ub313\uae00 \ub4f1\ub85d", "submit", "save", "post comment"))


def _is_forum_delete_candidate(candidate: Mapping[str, Any]) -> bool:
    return _candidate_text_has(candidate, ("\uc0ad\uc81c", "delete", "remove"))


def _is_forum_fill_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_fillable(candidate):
        return False
    if _is_forum_noise_candidate(candidate):
        return False
    return _candidate_text_has(candidate, ("\uac8c\uc2dc\uae00 \uc81c\ubaa9", "\uc81c\ubaa9", "\ub0b4\uc6a9", "\ub313\uae00", "comment", "content", "title"))


def _is_forum_detail_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_clickable(candidate) and not _is_link(candidate):
        return False
    if _is_forum_noise_candidate(candidate) or _is_forum_write_candidate(candidate) or _is_forum_submit_candidate(candidate) or _is_forum_delete_candidate(candidate):
        return False
    text = _candidate_text(candidate)
    if len(text.strip()) < 8:
        return False
    return not _candidate_text_has(candidate, ("\uac80\uc0c9", "search", "\uc54c\ub9bc", "notification", "\uce74\ud14c\uace0\ub9ac"))


def _is_required_commerce_candidate(candidate: Mapping[str, Any]) -> bool:
    semantic_type = str(candidate.get("semantic_action_type") or "").lower()
    if semantic_type in {"cart", "purchase", "add", "checkout", "detail", "details", "product_detail", "quantity"}:
        return True
    return bool(
        candidate.get("is_purchase_action")
        or candidate.get("is_quantity_control")
        or candidate.get("is_cart_quantity_related")
        or candidate.get("is_checkout_related")
        or _is_cart_candidate(candidate)
        or _is_detail_candidate(candidate)
        or _is_checkout_candidate(candidate)
        or _is_quantity_candidate(candidate)
    )


def _commerce_priority_boost(candidate: Mapping[str, Any]) -> float:
    boost = 0.0
    semantic_type = str(candidate.get("semantic_action_type") or "").lower()
    if semantic_type in {"add", "purchase", "checkout"}:
        boost += 0.2
    if semantic_type in {"cart", "quantity"}:
        boost += 0.16
    if semantic_type in {"detail", "details", "product_detail"}:
        boost += 0.14
    if _is_checkout_candidate(candidate):
        boost += 0.12
    if _is_quantity_candidate(candidate):
        boost += 0.1
    if _is_detail_candidate(candidate):
        boost += 0.09
    if bool(_is_cart_candidate(candidate) or candidate.get("is_purchase_action")):
        boost += 0.08
    return boost


def _is_cart_candidate(candidate: Mapping[str, Any]) -> bool:
    semantic_type = str(candidate.get("semantic_action_type") or "").lower()
    return semantic_type == "cart" or _candidate_text_has(
        candidate,
        ("cart", "basket", "add to cart", "장바구니", "카트", "담기", "추가"),
    )


def _is_quantity_candidate(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate).strip()
    return (
        bool(candidate.get("is_quantity_control") or candidate.get("is_cart_quantity_related"))
        or text in {"+", "-", "＋", "－"}
        or _candidate_text_has(
            candidate,
            ("quantity", "qty", "수량", "subtotal", "total", "합계", "소계", "총액", "plus", "minus", "증가", "감소"),
        )
    )


def _is_checkout_candidate(candidate: Mapping[str, Any]) -> bool:
    semantic_type = str(candidate.get("semantic_action_type") or "").lower()
    return semantic_type == "purchase" or bool(candidate.get("is_checkout_related")) or _candidate_text_has(
        candidate,
        ("checkout", "payment", "purchase", "order", "buy", "결제", "주문", "주문하기", "구매", "바로구매"),
    )


def _is_detail_candidate(candidate: Mapping[str, Any]) -> bool:
    if bool(candidate.get("is_detail_trigger")) or _candidate_text_has(
        candidate,
        ("detail", "details", "product", "상세", "상품", "제품", "자세히"),
    ):
        return True
    # A product card that opens its own detail view doesn't necessarily use
    # any of those words at all -- its identifying content is just the
    # product's own name/description (e.g. a card titled "라이트 러닝화" with
    # no literal "상세"/"product" anywhere), which the keyword check above
    # can't recognize. Only called when the page is already commerce_surface,
    # so a formatted price on an otherwise-unclassified clickable candidate
    # is a language-independent signal that this is a product listing entry
    # worth opening, not decorative chrome (nav tabs, a header icon).
    if bool(candidate.get("clickable")) and _has_price_pattern(_candidate_text(candidate)):
        return True
    return False


def _has_price_pattern(text: str) -> bool:
    return bool(re.search(r"[₩$]\s*\d[\d,]*|\d[\d,]{2,}\s*원", str(text or "")))


def _is_username_or_email_candidate(candidate: Mapping[str, Any]) -> bool:
    if not _is_fillable(candidate) or bool(candidate.get("is_password")):
        return False
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    if input_type in {"email", "text", ""}:
        return True
    return _candidate_text_has(candidate, ("email", "username", "user", "login"))


def _has_login_form(observation: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    login_flow = history.get("login_flow", {}) if isinstance(history, Mapping) else {}
    if isinstance(login_flow, Mapping) and bool(login_flow.get("has_login_form")):
        return True
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if not isinstance(candidates, list):
        candidates = []
    has_user = any(isinstance(candidate, Mapping) and _is_username_or_email_candidate(candidate) for candidate in candidates)
    has_password = any(isinstance(candidate, Mapping) and (bool(candidate.get("is_password")) or str(candidate.get("type") or candidate.get("input_type") or "").lower() == "password") for candidate in candidates)
    has_submit = any(isinstance(candidate, Mapping) and _is_login_submit(candidate) for candidate in candidates)
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    url = str(page_state.get("url") or "").lower() if isinstance(page_state, Mapping) else ""
    return bool((has_user and has_password and has_submit) or (has_password and any(token in url for token in ("login", "signin", "sign-in", "auth"))))


def _login_flow_summary(login_flow: Mapping[str, Any], has_login_form: bool, observation: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    candidates = candidates if isinstance(candidates, list) else []
    has_user = bool(login_flow.get("has_username_or_email_input") or login_flow.get("has_email_or_text_input") or any(isinstance(candidate, Mapping) and _is_username_or_email_candidate(candidate) for candidate in candidates))
    has_password = bool(login_flow.get("has_password_input") or any(isinstance(candidate, Mapping) and (bool(candidate.get("is_password")) or str(candidate.get("type") or candidate.get("input_type") or "").lower() == "password") for candidate in candidates))
    has_submit = bool(login_flow.get("has_login_submit") or login_flow.get("has_submit_button") or any(isinstance(candidate, Mapping) and _is_login_submit(candidate) for candidate in candidates))
    return {
        "has_login_form": bool(has_login_form),
        "has_username_or_email_input": has_user,
        "has_password_input": has_password,
        "has_login_submit": has_submit,
        "username_or_email_filled": bool(login_flow.get("username_or_email_filled") or login_flow.get("email_input_filled")),
        "password_filled": bool(login_flow.get("password_filled") or login_flow.get("password_input_filled")),
        "login_submit_clicked": bool(login_flow.get("login_submit_clicked") or login_flow.get("submit_clicked")),
        "login_result_checked": bool(login_flow.get("login_result_checked") or login_flow.get("submit_result_checked")),
        "login_flow_attempted": bool(login_flow.get("login_flow_attempted")),
        "login_flow_completed": bool(login_flow.get("login_flow_completed")),
        "login_flow_status": str(login_flow.get("login_flow_status") or ("input_detected" if has_login_form else "not_applicable")),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("element_key") or candidate.get("bid") or candidate.get("name") or candidate.get("text") or "")


def _opportunity_signature(opportunity: Mapping[str, Any], input_value: str = "") -> str:
    payload = {
        "page_url": str(opportunity.get("page_url") or ""),
        "opportunity_type": str(opportunity.get("opportunity_type") or ""),
        "target_element_key": str(opportunity.get("target_element_key") or opportunity.get("element_context_key") or ""),
        "normalized_input_value": _normalize_input_value(input_value),
    }
    return "::".join(payload.values())


def _normalize_input_value(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _page_type(observation: Mapping[str, Any], history: Mapping[str, Any]) -> str:
    page_verification = history.get("page_verification", {}) if isinstance(history, Mapping) else {}
    if isinstance(page_verification, Mapping) and page_verification.get("page_type"):
        return str(page_verification.get("page_type"))
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    url = str(page_state.get("url") or "").lower() if isinstance(page_state, Mapping) else ""
    if _has_json_response(observation, history) or "typicode" in url or "my-json-server" in url or "/api/" in url:
        return "api_json_page"
    if any(token in url for token in ("fakeapi", "swagger", "docs", "api-docs", "documentation")):
        return "documentation_page"
    return "ui_page"


def _is_api_or_documentation_page(observation: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    return _page_type(observation, history) in {"api_json_page", "data_endpoint_page", "documentation_page"}


def _has_endpoint_link(candidates: Any) -> bool:
    if not isinstance(candidates, list):
        return False
    return any(
        isinstance(candidate, Mapping)
        and ("/api" in str(candidate.get("href") or "").lower() or "/api" in _candidate_text(candidate))
        for candidate in candidates
    )


def _has_search_input(candidates: Any) -> bool:
    if not isinstance(candidates, list):
        return False
    return any(
        isinstance(candidate, Mapping)
        and bool(candidate.get("fillable"))
        and (
            str(candidate.get("role") or "").lower() == "searchbox"
            or str(candidate.get("type") or candidate.get("input_type") or "").lower() == "search"
            or "search" in _candidate_text(candidate)
        )
        for candidate in candidates
    )


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in ("name", "text", "placeholder", "href", "aria_label", "title")
    ).lower()


def _dom_verification_loop_guard_active(history: Mapping[str, Any]) -> bool:
    try:
        repeated = int(history.get("repeated_inspect_dom_count", 0) or 0)
    except Exception:
        repeated = 0
    return repeated >= 2


def _has_network(runtime: Any) -> bool:
    return isinstance(runtime, Mapping) and bool(
        runtime.get("network_entries")
        or runtime.get("network_request_count")
        or runtime.get("request_count")
        or runtime.get("api_probe_endpoint_available")
        or runtime.get("api_probe_status") is not None
    )


def _has_json_response(observation: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    page_verification = history.get("page_verification", {}) if isinstance(history, Mapping) else {}
    if isinstance(page_verification, Mapping) and bool(page_verification.get("has_json_response")):
        return True
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    text = str(page_state.get("page_text") or page_state.get("text") or "").strip() if isinstance(page_state, Mapping) else ""
    return text.startswith("{") or text.startswith("[")


def _console_error_count(runtime: Any) -> int:
    if not isinstance(runtime, Mapping):
        return 0
    return int(runtime.get("console_error_count", 0) or 0) + int(runtime.get("playwright_console_error_count", 0) or 0)


def _dom_signature(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    payload = {
        "url": str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        "text_len": int(page_state.get("page_text_length") or len(str(page_state.get("page_text") or ""))) if isinstance(page_state, Mapping) else 0,
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
    }
    return _hash(json.dumps(payload, sort_keys=True))[:12]


def _layout_context_key(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    payload = {
        "url": str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        "viewport": str(page_state.get("viewport_type") or "") if isinstance(page_state, Mapping) else "",
        "dom": _dom_signature(observation),
    }
    return _hash(json.dumps(payload, sort_keys=True))[:12]


def _dom_context_key(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    payload = {
        "url": str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "",
        "dom": str(page_state.get("dom_hash") or _dom_signature(observation)) if isinstance(page_state, Mapping) else _dom_signature(observation),
    }
    return _hash(json.dumps(payload, sort_keys=True))[:12]


def _viewport_context_key(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    return _hash(str(page_state.get("url") or "") if isinstance(page_state, Mapping) else "")[:12]


def _anomaly_key(anomaly: Mapping[str, Any]) -> str:
    return _hash(json.dumps({key: anomaly.get(key) for key in ("type", "signature", "evidence")}, sort_keys=True, default=str))[:12]


def _anomaly_type_context_key(anomaly: Mapping[str, Any], page_url: str = "") -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    url = str(page_url or evidence.get("after_url") or evidence.get("before_url") or "")
    return _hash(json.dumps({"type": str(anomaly.get("type") or ""), "url": url}, sort_keys=True))[:12]


def _hash(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _opportunity_status(opportunity: Mapping[str, Any]) -> str:
    if opportunity.get("duplicate"):
        return "duplicate"
    if opportunity.get("skipped") or opportunity.get("skipped_reason"):
        return "skipped"
    if opportunity.get("verified"):
        return "verified"
    if opportunity.get("failed"):
        return "failed"
    if opportunity.get("executed"):
        return "executed"
    if opportunity.get("selected"):
        return "selected"
    if opportunity.get("masked"):
        return "masked"
    if opportunity.get("enabled"):
        return "enabled"
    return "created"


def _failure_reason(action: Mapping[str, Any]) -> str:
    reason = str(
        action.get("failure_reason")
        or action.get("invalid_action_reason")
        or action.get("action_error")
        or ""
    )
    return reason or "action_failed_without_reason"


def _mark_related_anomalies_verified(history: Dict[str, Any], opportunity: Mapping[str, Any]) -> None:
    stored = history.get("unverified_anomalies")
    if not isinstance(stored, list) or not stored:
        return
    opportunity_type = str(opportunity.get("opportunity_type") or "")
    if opportunity_type not in {
        "check_api_response",
        "check_json_schema",
        "verify_anomaly_reproduction",
        "verify_duplicated_rendering",
        "verify_runtime_error",
        "verify_form_no_feedback",
        "verify_action_result",
        "check_search_result_after_input",
    }:
        return
    remaining = []
    verified_keys = history.setdefault("verified_anomaly_keys", set())
    if not isinstance(verified_keys, set):
        verified_keys = set(str(item) for item in verified_keys or [])
        history["verified_anomaly_keys"] = verified_keys
    verified_type_contexts = history.setdefault("verified_anomaly_type_contexts", set())
    if not isinstance(verified_type_contexts, set):
        verified_type_contexts = set(str(item) for item in verified_type_contexts or [])
        history["verified_anomaly_type_contexts"] = verified_type_contexts
    verified_types = history.setdefault("verified_anomaly_types", set())
    if not isinstance(verified_types, set):
        verified_types = set(str(item) for item in verified_types or [])
        history["verified_anomaly_types"] = verified_types
    page_url = str(opportunity.get("page_url") or "")

    def mark_verified(anomaly: Mapping[str, Any]) -> None:
        verified_keys.add(_anomaly_key(anomaly))
        verified_type_contexts.add(_anomaly_type_context_key(anomaly, page_url))
        verified_types.add(str(anomaly.get("type") or ""))

    for anomaly in stored:
        if not isinstance(anomaly, Mapping):
            continue
        anomaly_type = str(anomaly.get("type") or "")
        if opportunity_type == "verify_form_no_feedback" and anomaly_type == "form-no-feedback":
            mark_verified(anomaly)
            continue
        if opportunity_type == "verify_action_result" and anomaly_type == "action-error":
            mark_verified(anomaly)
            continue
        if opportunity_type == "verify_duplicated_rendering" and anomaly_type == "duplicated-rendering":
            mark_verified(anomaly)
            continue
        if opportunity_type == "verify_runtime_error" and anomaly_type in {"console-error", "runtime-exception"}:
            mark_verified(anomaly)
            continue
        if opportunity_type == "verify_anomaly_reproduction":
            mark_verified(anomaly)
            continue
        remaining.append(anomaly)
    history["unverified_anomalies"] = remaining


def _mark_observed_anomalies_verified(history: Dict[str, Any], anomalies: Iterable[Mapping[str, Any]], *, page_url: str = "") -> None:
    verified_keys = history.setdefault("verified_anomaly_keys", set())
    if not isinstance(verified_keys, set):
        verified_keys = set(str(item) for item in verified_keys or [])
        history["verified_anomaly_keys"] = verified_keys
    verified_type_contexts = history.setdefault("verified_anomaly_type_contexts", set())
    if not isinstance(verified_type_contexts, set):
        verified_type_contexts = set(str(item) for item in verified_type_contexts or [])
        history["verified_anomaly_type_contexts"] = verified_type_contexts
    verified_types = history.setdefault("verified_anomaly_types", set())
    if not isinstance(verified_types, set):
        verified_types = set(str(item) for item in verified_types or [])
        history["verified_anomaly_types"] = verified_types
    observed_keys = {
        _anomaly_key(anomaly)
        for anomaly in anomalies
        if isinstance(anomaly, Mapping)
    }
    observed_type_contexts = {
        _anomaly_type_context_key(anomaly, page_url)
        for anomaly in anomalies
        if isinstance(anomaly, Mapping)
    }
    if not observed_keys:
        return
    verified_keys.update(observed_keys)
    verified_type_contexts.update(observed_type_contexts)
    verified_types.update(str(anomaly.get("type") or "") for anomaly in anomalies if isinstance(anomaly, Mapping))
    stored = history.get("unverified_anomalies")
    if isinstance(stored, list):
        history["unverified_anomalies"] = [
            anomaly
            for anomaly in stored
            if not isinstance(anomaly, Mapping)
            or (
                _anomaly_key(anomaly) not in observed_keys
                and _anomaly_type_context_key(anomaly, page_url) not in observed_type_contexts
            )
        ]
