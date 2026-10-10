"""Observation-only anomaly heuristics for BrowserGym PPO training."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, List, Mapping
from urllib.parse import urljoin, urlparse

from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context
from services.r10_network_ui_semantic_evidence import semantic_pair_confidence


ACTION_KEYWORDS = (
    "구매",
    "구매하기",
    "장바구니 담기",
    "담기",
    "buy",
    "add to cart",
    "add",
    "purchase",
    "submit",
    "save",
    "login",
    "援щℓ",
    "?λ컮援щ땲",
)
PURCHASE_KEYWORDS = (
    "구매",
    "구매하기",
    "장바구니 담기",
    "담기",
    "buy",
    "add to cart",
    "add",
    "purchase",
    "援щℓ",
    "援щℓ?섍린",
    "?λ컮援щ땲",
)
CLICKABLE_ROLES = {"button", "link", "menuitem"}
FEEDBACK_TOKENS = (
    "toast",
    "modal",
    "alert",
    "success",
    "성공",
    "완료",
    "추가",
    "?깃났",
    "?꾨즺",
)
BOOK_CONTEXT_TOKENS = ("추천", "베스트셀러", "recommended", "bestseller", "rc-card", "book")


def detect_anomalies(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    site_profile: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    action = action_info.get("action", action_info)
    action_type = str(action.get("action_type", action_info.get("action_type", "")))
    site_id = str(action_info.get("site_id") or action.get("site_id") or "")
    site_profile = site_profile or action_info.get("site_profile") or before_observation.get("site_profile") or {}
    candidate_index = int(action.get("candidate_index", 0) or 0)

    before_candidates = before_observation.get("candidate_elements", []) or []
    after_candidates = after_observation.get("candidate_elements", []) or []
    before_url = _url(before_observation)
    after_url = _url(after_observation)
    last_action_error = bool(action_info.get("last_action_error")) or bool(
        after_observation.get("runtime_signals", {}).get("last_action_error")
    )
    no_url_change = before_url == after_url
    candidate_delta = abs(len(after_candidates) - len(before_candidates))
    no_state_change = _state_signature(before_observation) == _state_signature(after_observation)
    page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
    no_text_change = page_text_delta <= 20
    no_candidate_change = candidate_delta <= 1
    meaningful_page_text_change = page_text_delta > 40
    meaningful_candidate_change = candidate_delta > 2
    feedback_visible_before = _has_feedback_message(before_observation)
    feedback_visible_after = _has_feedback_message(after_observation)
    no_feedback = not feedback_visible_after or feedback_visible_before
    no_network_change = not _has_meaningful_network_change(after_observation, action_info)
    no_runtime_signal_delta = not _has_runtime_signal_delta(before_observation, after_observation)
    modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
    form_opened = _has_form_signal(after_observation) and not _has_form_signal(before_observation)

    clicked_candidate = _candidate_at(before_candidates, candidate_index)
    if action_type == "click_element" and clicked_candidate:
        visibility = _visibility(clicked_candidate)
        target_text = _candidate_text(clicked_candidate)
        is_purchase_action = _is_purchase_action(clicked_candidate)
        is_workout_add_action = _is_workout_add_action(clicked_candidate)
        cart_before = _cart_count(before_observation)
        cart_after = _cart_count(after_observation)
        has_cart_counts = cart_before is not None and cart_after is not None
        before_cart_text = _cart_text(before_observation)
        after_cart_text = _cart_text(after_observation)
        cart_state_changed = bool(before_cart_text != after_cart_text or (has_cart_counts and cart_before != cart_after))
        r01_no_response = _is_r01_no_response(
            no_url_change=no_url_change,
            no_state_change=no_state_change,
            no_text_change=no_text_change,
            no_candidate_change=no_candidate_change,
            no_feedback=no_feedback,
            no_network_change=no_network_change,
            no_runtime_signal_delta=no_runtime_signal_delta,
            modal_opened=modal_opened,
            form_opened=form_opened,
            target_state_changed=cart_state_changed,
            last_action_error=last_action_error,
        )
        general_r01_no_response = _is_r01_no_response(
            no_url_change=no_url_change,
            no_state_change=no_state_change,
            no_text_change=no_text_change,
            no_candidate_change=no_candidate_change,
            no_feedback=no_feedback,
            no_network_change=no_network_change,
            no_runtime_signal_delta=no_runtime_signal_delta,
            modal_opened=modal_opened,
            form_opened=form_opened,
            target_state_changed=False,
            last_action_error=last_action_error,
        )
        if (
            is_purchase_action
            and has_cart_counts
            and cart_before == cart_after
            and before_cart_text == after_cart_text
            and r01_no_response
        ):
            anomalies.append(
                {
                    "type": "button-no-response",
                    "confidence": 0.95,
                    "evidence": {
                        "clicked_bid": clicked_candidate.get("bid"),
                        "clicked_text": target_text,
                        "clicked_name": clicked_candidate.get("name"),
                        "cart_count_before": cart_before,
                        "cart_count_after": cart_after,
                        "cart_text_before": before_cart_text,
                        "cart_text_after": after_cart_text,
                        "before_url": before_url,
                        "after_url": after_url,
                        "cart_count_detected": True,
                        "target": _target_evidence(clicked_candidate),
                        "is_purchase_action": True,
                    },
                }
            )
        elif visibility <= 0.0:
            if last_action_error or no_state_change:
                anomalies.append(_low_visibility_anomaly(clicked_candidate, visibility, last_action_error))
        elif _is_clickable_target(clicked_candidate):
            if (
                is_purchase_action
                and r01_no_response
            ):
                anomalies.append(
                    {
                        "type": "button-no-response",
                        "confidence": 0.5,
                        "evidence": {
                            "before_url": before_url,
                            "after_url": after_url,
                            "candidate_delta": candidate_delta,
                            "page_text_delta": page_text_delta,
                            "no_state_change": no_state_change,
                            "cart_count_before": cart_before,
                            "cart_count_after": cart_after,
                            "cart_text_before": before_cart_text,
                            "cart_text_after": after_cart_text,
                            "cart_count_detected": has_cart_counts,
                            "clicked_text": target_text,
                            "clicked_name": clicked_candidate.get("name"),
                            "clicked_bid": clicked_candidate.get("bid"),
                            "target": _target_evidence(clicked_candidate),
                            "is_purchase_action": True,
                        },
                    }
                )
            elif (
                not is_purchase_action
                and not _is_openended_target(clicked_candidate)
                and general_r01_no_response
            ):
                confidence = 0.3 if _has_action_keyword(target_text) else 0.2
                anomalies.append(
                    {
                        "type": "button-no-response",
                        "confidence": confidence,
                        "evidence": {
                            "before_url": before_url,
                            "after_url": after_url,
                            "clicked_text": target_text,
                            "clicked_name": clicked_candidate.get("name"),
                            "clicked_bid": clicked_candidate.get("bid"),
                            "target": _target_evidence(clicked_candidate),
                            "is_purchase_action": False,
                        },
                    }
                )
            elif (
                _is_openended_target(clicked_candidate)
                and not meaningful_page_text_change
                and not meaningful_candidate_change
                and r01_no_response
            ):
                anomaly_type, confidence = _openended_interaction_type_and_confidence(clicked_candidate, before_observation)
                context_evidence = _openended_context_evidence(
                    clicked_candidate,
                    before_observation,
                    after_observation,
                    before_url,
                    after_url,
                )
                if _should_emit_openended_interaction_anomaly(
                    anomaly_type,
                    clicked_candidate,
                    before_observation,
                    after_observation,
                    context_evidence,
                ):
                    review = _human_review_metadata(anomaly_type, clicked_candidate, before_url, after_url, route_changed=False)
                    context_adjustments = _context_review_adjustments(anomaly_type, clicked_candidate, context_evidence)
                    if "confidence_cap" in context_adjustments:
                        confidence = min(confidence, float(context_adjustments.pop("confidence_cap") or confidence))
                    review.update(context_adjustments)
                    anomalies.append(
                        {
                            "type": anomaly_type,
                            "confidence": confidence,
                            "classification": "exploratory_anomaly",
                            **review,
                            "evidence": {
                                "clicked_text": target_text,
                                "clicked_name": clicked_candidate.get("name"),
                                "clicked_bid": clicked_candidate.get("bid"),
                                "clicked_role": clicked_candidate.get("role"),
                                "clicked_tag": clicked_candidate.get("tag"),
                                "openended_keyword_matches": clicked_candidate.get("openended_keyword_matches", []),
                                "before_url": before_url,
                                "after_url": after_url,
                                "page_text_delta": page_text_delta,
                                "candidate_delta": candidate_delta,
                                "modal_opened": modal_opened,
                                "form_opened": form_opened,
                                "route_changed": False,
                                "cart_state_changed": cart_state_changed,
                                "toast_visible": not no_feedback,
                                "target": _target_evidence(clicked_candidate),
                                **context_evidence,
                            },
                        }
                    )
        elif visibility < 0.3 and (last_action_error or no_state_change):
            anomalies.append(_low_visibility_anomaly(clicked_candidate, visibility, last_action_error))

        catalog_no_response = _detect_catalog_button_no_response(
            clicked_candidate,
            before_observation,
            after_observation,
            site_profile,
            before_url,
            after_url,
            candidate_delta,
            last_action_error,
            r01_no_response,
            page_text_delta,
            modal_opened,
            form_opened,
            feedback_visible_after,
        )
        if catalog_no_response:
            anomalies.append(catalog_no_response)

    layout_overflow = _detect_r06_layout_overflow(after_observation, action_type)
    if layout_overflow:
        anomalies.append(layout_overflow)
    layout_overlap = _detect_r05_layout_overlap(after_observation, action_type)
    if layout_overlap:
        anomalies.append(layout_overlap)
    anomalies.extend(_detect_r07_accessibility(after_observation, action_info, last_action_error))
    r08_validation = _detect_r08_validation_issue(
        before_observation,
        after_observation,
        action_info,
        action_type,
        clicked_candidate,
        last_action_error,
    )
    if r08_validation:
        anomalies.append(r08_validation)
    r09_result_issue = _detect_r09_result_issue(
        before_observation,
        after_observation,
        action_info,
        action_type,
        clicked_candidate,
        last_action_error,
    )
    if r09_result_issue:
        anomalies.append(r09_result_issue)
    r13_api_error = _detect_r13_api_error(after_observation, action_info, action_type, clicked_candidate)
    if r13_api_error:
        anomalies.append(r13_api_error)
    r15_timeout = _detect_r15_timeout(after_observation, action_info, action_type, clicked_candidate)
    if r15_timeout:
        anomalies.append(r15_timeout)
    r14_network_failure = _detect_r14_network_failure(after_observation, action_info, action_type, clicked_candidate)
    if r14_network_failure:
        anomalies.append(r14_network_failure)
    r10_ui_api_mismatch = _detect_r10_ui_api_mismatch(
        before_observation,
        after_observation,
        action_info,
        action_type,
        last_action_error,
    )
    if r10_ui_api_mismatch:
        anomalies.append(r10_ui_api_mismatch)
    r11_calculation_error = _detect_r11_calculation_error(
        before_observation,
        after_observation,
        action_info,
        action_type,
        last_action_error,
    )
    if r11_calculation_error:
        anomalies.append(r11_calculation_error)
    r12_persistence_issue = _detect_r12_persistence_issue(
        before_observation,
        after_observation,
        action_info,
        action_type,
        last_action_error,
    )
    if r12_persistence_issue:
        anomalies.append(r12_persistence_issue)

    error_message = _error_message(action_info)
    action_result = action_info.get("action_result", {})
    action_result_failed = isinstance(action_result, Mapping) and bool(action_result.get("failed"))
    if last_action_error and (error_message or action_result_failed):
        anomalies.append(
            {
                "type": "action-error",
                "confidence": 0.9,
                "evidence": {
                    "action_type": action_type,
                    "error": error_message,
                    "action_result_failed": action_result_failed,
                },
            }
        )
        if _r13_auth_error_expected(action_info) and ("403" in str(error_message).lower() or "forbidden" in str(error_message).lower()):
            anomalies.append(
                {
                    "type": "api-forbidden",
                    "confidence": 0.9,
                    "evidence": {
                        "action_type": action_type,
                        "error": error_message,
                        "before_url": before_url,
                        "after_url": after_url,
                        "target": _target_evidence(clicked_candidate) if clicked_candidate else {},
                    },
                }
            )

    page_text_after = _page_text(after_observation).lower()
    if action_type == "inspect_network":
        api_403_count = int(action_info.get("api_403_count", 0) or 0)
        network_entries = action_info.get("network_entries", []) or []
        network_text = " ".join(str(item) for item in network_entries).lower()
        if _r13_auth_error_expected(action_info) and (api_403_count > 0 or "403" in network_text or "forbidden" in network_text):
            anomalies.append(
                {
                    "type": "api-forbidden",
                    "confidence": 0.9,
                    "evidence": {
                        "action_type": action_type,
                        "api_403_count": api_403_count,
                        "network_status": 403,
                        "network_entries": network_entries[:5] if isinstance(network_entries, list) else [],
                        "target_keyword_matches": _keyword_matches(network_text, ["forbidden", "403", "api"]),
                        "before_url": before_url,
                        "after_url": after_url,
                    },
                }
            )
    if _r13_auth_error_expected(action_info) and ("403" in page_text_after or "forbidden" in page_text_after or "access denied" in page_text_after):
        anomalies.append(
            {
                "type": "api-forbidden",
                "confidence": 0.8,
                "evidence": {
                    "before_url": before_url,
                    "after_url": after_url,
                    "page_text_contains_forbidden": True,
                    "target": _target_evidence(clicked_candidate) if clicked_candidate else {},
                },
            }
        )

    runtime_exception = _detect_r03_runtime_exception(before_observation, after_observation, action_info, last_action_error)
    if runtime_exception:
        anomalies.append(runtime_exception)

    if (
        not r15_timeout
        and action_type in {"fill_input", "press_enter", "click_element"}
        and _has_explicit_async_or_hang_keyword(clicked_candidate or {})
        and _timeout_message_visible(after_observation)
    ):
        if _state_signature(before_observation) == _state_signature(after_observation) and not _has_feedback_message(after_observation):
            anomalies.append(
                {
                    "type": "async-hang",
                    "confidence": 0.82,
                    "evidence": {
                        "action_type": action_type,
                        "clicked_text": _candidate_text(clicked_candidate or {}),
                        "async_keyword_match": True,
                        "page_frozen_signal": True,
                        "loading_visible_after": _loading_visible(after_observation),
                        "timeout_message_visible": _timeout_message_visible(after_observation),
                        "before_url": before_url,
                        "after_url": after_url,
                    },
                }
            )

    login_redirect_mismatch = _detect_login_redirect_mismatch(
        before_observation,
        after_observation,
        action_info,
        before_url,
        after_url,
        last_action_error,
    )
    if login_redirect_mismatch:
        anomalies.append(login_redirect_mismatch)

    broken_navigation = _detect_broken_navigation(
        clicked_candidate,
        before_observation,
        after_observation,
        action_info,
        action_type,
        before_url,
        after_url,
        last_action_error,
    )
    if broken_navigation:
        anomalies.append(broken_navigation)

    duplicate_anomaly = _detect_duplicated_rendering(
        before_observation,
        after_observation,
        action_info,
        action_type,
        last_action_error,
    )
    if duplicate_anomaly:
        anomalies.append(duplicate_anomaly)

    rendering_failure = _detect_r03_rendering_failure(
        before_observation,
        after_observation,
        action_info,
        last_action_error,
    )
    if rendering_failure:
        anomalies.append(rendering_failure)

    return _standardize_anomalies(anomalies, before_observation, after_observation, action_info, action_type)


def compute_anomaly_score(anomalies: List[Dict[str, Any]]) -> float:
    if not anomalies:
        return 0.0
    return min(1.0, sum(float(item.get("confidence", 0.0) or 0.0) for item in anomalies) / len(anomalies))


class AnomalyConfirmationTracker:
    """Lightweight per-run confirmation state keyed by stable anomaly signature."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}
        self._auto_step = 0

    def next_step_id(self) -> int:
        self._auto_step += 1
        return self._auto_step

    def update(self, anomaly: Dict[str, Any], step_id: Any, action_id: Any = None, agent_mode: Any = None) -> Dict[str, Any]:
        signature = str(anomaly.get("signature") or "")
        if not signature:
            return {"seen_count": 0, "confirmed": False}
        action_identity = _confirmation_action_identity(action_id, step_id)
        record = self._records.setdefault(
            signature,
            {
                "seen_count": 0,
                "first_seen_step": step_id,
                "last_seen_step": step_id,
                "first_seen_action_id": action_identity,
                "last_seen_action_id": action_identity,
                "confirmed": False,
                "_seen_steps": set(),
                "_seen_action_ids": set(),
            },
        )
        seen_steps = record.setdefault("_seen_steps", set())
        if step_id not in seen_steps:
            seen_steps.add(step_id)
            record["seen_count"] = int(record.get("seen_count", 0) or 0) + 1
            record["last_seen_step"] = step_id
            record["last_seen_action_id"] = action_identity
            if record.get("first_seen_step") is None:
                record["first_seen_step"] = step_id
            if record.get("first_seen_action_id") is None:
                record["first_seen_action_id"] = action_identity
        if action_identity:
            record.setdefault("_seen_action_ids", set()).add(action_identity)
        r06_qualified = _r06_confirmation_evidence_qualified(anomaly)
        independent_r10_reproduction = (
            str(anomaly.get("rule_id") or _anomaly_rule_id(anomaly)) == "R10"
            and anomaly.get("confirmation_policy") == "reproduce"
            and _r10_confirmation_evidence_qualified(anomaly)
            and _r10_independent_reproduction_seen(record)
        )
        if (
            (anomaly.get("status") == "confirmed" or (
                anomaly.get("confirmation_policy") == "reproduce"
                and (
                    independent_r10_reproduction
                    if str(anomaly.get("rule_id") or _anomaly_rule_id(anomaly)) == "R10"
                    else int(record.get("seen_count", 0) or 0) >= 2
                )
            ))
            and r06_qualified
        ):
            record["confirmed"] = True
            anomaly["status"] = "confirmed"
            anomaly["lifecycle_status"] = "CONFIRMED"
            anomaly["pending_confirmation"] = False
            if str(anomaly.get("rule_id") or _anomaly_rule_id(anomaly)) == "R10":
                anomaly["confirmation_source"] = _r10_confirmation_source(agent_mode)
        anomaly["confirmation"] = self.record(signature)
        return anomaly["confirmation"]

    def record(self, signature: str) -> Dict[str, Any]:
        record = self._records.get(signature, {})
        return {
            "seen_count": int(record.get("seen_count", 0) or 0),
            "first_seen_step": record.get("first_seen_step"),
            "last_seen_step": record.get("last_seen_step"),
            "first_seen_action_id": record.get("first_seen_action_id"),
            "last_seen_action_id": record.get("last_seen_action_id"),
            "confirmed": bool(record.get("confirmed")),
        }


def _standardize_anomalies(
    anomalies: List[Dict[str, Any]],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
) -> List[Dict[str, Any]]:
    tracker = action_info.get("confirmation_tracker")
    if tracker is not None and not isinstance(tracker, AnomalyConfirmationTracker):
        tracker = None
    step_id = _anomaly_step_id(action_info)
    action_id = _anomaly_action_id(action_info)
    agent_mode = _anomaly_agent_mode(action_info)
    action = action_info.get("action", action_info)
    action = action if isinstance(action, Mapping) else {}
    trigger_action_type = str(action.get("action_type") or action_info.get("action_type") or action_type or "")
    before_candidates = before_observation.get("candidate_elements", []) if isinstance(before_observation, Mapping) else []
    before_candidates = before_candidates if isinstance(before_candidates, list) else []
    try:
        trigger_index = int(action.get("candidate_index", 0) or 0)
    except (TypeError, ValueError):
        trigger_index = 0
    trigger_candidate = before_candidates[trigger_index] if trigger_index < len(before_candidates) and isinstance(before_candidates[trigger_index], Mapping) else {}
    trigger_target = str(action.get("target_id") or trigger_candidate.get("bid") or trigger_candidate.get("id") or "").strip()
    if tracker is not None and step_id is None:
        step_id = tracker.next_step_id()
    standardized: List[Dict[str, Any]] = []
    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        evidence = anomaly.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            anomaly["evidence"] = evidence
        rule_id = _anomaly_rule_id(anomaly)
        issue_type = str(anomaly.get("issue_type") or evidence.get("issue_type") or anomaly.get("type") or "unknown")
        anomaly["rule_id"] = rule_id
        anomaly["issue_type"] = issue_type
        evidence.setdefault("rule_id", rule_id)
        evidence.setdefault("issue_type", issue_type)
        anomaly.setdefault("confidence", 0.0)
        anomaly["action_type"] = str(anomaly.get("action_type") or evidence.get("action_type") or action_type or "")
        anomaly["target_id"] = _anomaly_target_id(anomaly)
        evidence_target = str(anomaly.get("target_id") or "").strip()
        anomaly["evidence_target_identity"] = f"bid:{evidence_target}" if evidence_target and ":" not in evidence_target else (evidence_target or None)
        anomaly["trigger_action_type"] = trigger_action_type or None
        anomaly["trigger_target_identity"] = f"bid:{trigger_target}" if trigger_target and ":" not in trigger_target else (trigger_target or None)
        origin_provenance = action_info.get("origin_action_provenance")
        if isinstance(origin_provenance, Mapping):
            anomaly["origin_action_provenance"] = dict(origin_provenance)
            evidence.setdefault("origin_action_provenance", dict(origin_provenance))
            evidence.setdefault("source_action_id", origin_provenance.get("action_id"))
            evidence.setdefault("source_action_type", origin_provenance.get("action_type"))
            evidence.setdefault("source_target_bid", origin_provenance.get("target_bid"))
        selected_option = action_info.get("selected_option")
        if isinstance(selected_option, Mapping):
            option_value = selected_option.get("selected_option") if isinstance(selected_option.get("selected_option"), Mapping) else selected_option
            option_identity = {
                key: option_value.get(key)
                for key in ("value", "label", "text", "index")
                if option_value.get(key) is not None
            }
            if option_identity:
                anomaly["origin_option_identity"] = option_identity
                evidence.setdefault("origin_option_identity", option_identity)
                evidence.setdefault("reproduction_option_identity", option_identity)
        reproduction = derive_reproduction_context(anomaly)
        anomaly.update(reproduction.as_dict())
        evidence.setdefault("evidence_target_identity", anomaly.get("evidence_target_identity"))
        evidence.setdefault("trigger_action_type", anomaly.get("trigger_action_type"))
        evidence.setdefault("trigger_target_identity", anomaly.get("trigger_target_identity"))
        evidence.setdefault("reproduction_strategy", anomaly.get("reproduction_strategy"))
        anomaly["entity_type"] = str(anomaly.get("entity_type") or evidence.get("entity_type") or "")
        anomaly["entity_id"] = str(anomaly.get("entity_id") or evidence.get("entity_id") or evidence.get("expected_entity_id") or "")
        anomaly["url_path"] = _anomaly_url_path(anomaly, before_observation, after_observation)
        anomaly["signature"] = _build_anomaly_signature(anomaly)
        if rule_id == "R10":
            pair_confidence = semantic_pair_confidence(evidence)
            if pair_confidence is not None:
                anomaly["pair_confidence"] = pair_confidence
                anomaly["semantic_pair_confidence"] = pair_confidence
        policy = _anomaly_confirmation_policy(anomaly)
        anomaly["confirmation_policy"] = policy
        anomaly["status"] = "confirmed" if _anomaly_confirmed_by_policy(anomaly, policy) else "candidate"
        if rule_id == "R10":
            anomaly["lifecycle_status"] = "CONFIRMED" if anomaly["status"] == "confirmed" else "DETECTED"
            anomaly["pending_confirmation"] = anomaly["status"] != "confirmed"
            anomaly.setdefault("confirmation_source", "")
            anomaly.setdefault("originating_action_id", str(action_id or evidence.get("action_id") or ""))
            anomaly.setdefault("originating_step_id", step_id)
        if tracker is not None:
            tracker.update(anomaly, step_id, action_id=action_id, agent_mode=agent_mode)
        standardized.append(anomaly)
    return standardized


def _anomaly_step_id(action_info: Mapping[str, Any]) -> Any:
    action = action_info.get("action", action_info)
    for source in (action_info, action if isinstance(action, Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("step_id", "step", "action_step", "timestep"):
            if key in source:
                return source.get(key)
    return None


def _anomaly_action_id(action_info: Mapping[str, Any]) -> Any:
    action = action_info.get("action", action_info)
    for source in (action_info, action if isinstance(action, Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("action_id", "runtime_action_id", "action_identity"):
            if key in source:
                return source.get(key)
    return None


def _anomaly_agent_mode(action_info: Mapping[str, Any]) -> str:
    action = action_info.get("action", action_info)
    for source in (action_info, action if isinstance(action, Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        for key in ("agent_mode", "policy_mode", "active_agent"):
            value = source.get(key)
            if value:
                return str(value)
    return ""


def _anomaly_rule_id(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if isinstance(evidence, Mapping) and evidence.get("rule_id"):
        return str(evidence.get("rule_id"))
    anomaly_type = str(anomaly.get("type") or "")
    mapping = {
        "button-no-response": "R01",
        "broken-navigation": "R02",
        "login-redirect-mismatch": "R02",
        "runtime-exception": "R03",
        "empty-state-rendering": "R03",
        "duplicated-rendering": "R04",
        "forum-comment-duplicated": "R04",
        "layout-overlap": "R05",
        "layout-overflow": "R06",
        "accessibility-issue": "R07",
        "low-visibility-interaction": "R07",
        "weak-password-validation": "R08",
        "forum-empty-post-validation-missing": "R08",
        "filter-result-mismatch": "R09",
        "search-result-mismatch": "R09",
        "sort-order-mismatch": "R09",
        "pagination-mismatch": "R09",
        "filter-no-effect": "R09",
        "cart-quantity-mismatch": "R10",
        "product-detail-mismatch": "R10",
        "stale-data-rendering": "R10",
        "api-ui-mismatch": "R10",
        "cart-total-mismatch": "R11",
        "persistence-mismatch": "R12",
        "forum-comment-delete-failed": "R12",
        "api-server-error": "R13",
        "network-failure": "R14",
        "network-error": "R14",
        "timeout-error": "R15",
        "async-hang": "R15",
    }
    return mapping.get(anomaly_type, "")


def _anomaly_confirmation_policy(anomaly: Mapping[str, Any]) -> str:
    rule_id = str(anomaly.get("rule_id") or _anomaly_rule_id(anomaly))
    if rule_id in {"R01", "R02", "R04", "R05", "R06"}:
        return "reproduce"
    if rule_id == "R10":
        return "reproduce"
    if rule_id == "R12":
        return "followup"
    if rule_id in {"R03", "R07", "R08", "R09", "R11", "R13", "R14", "R15"}:
        return "strong-evidence"
    return "reproduce"


def _anomaly_confirmed_by_policy(anomaly: Mapping[str, Any], policy: str) -> bool:
    if policy == "reproduce":
        return False
    if policy == "followup":
        return _has_followup_confirmation_evidence(anomaly)
    if policy == "strong-evidence":
        return _has_strong_confirmation_evidence(anomaly)
    return False


def _has_followup_confirmation_evidence(anomaly: Mapping[str, Any]) -> bool:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return False
    return bool(
        evidence.get("followup_observation_present")
        or evidence.get("refetch_detected")
        or evidence.get("observed_persisted_value") is not None
        or evidence.get("observed_exists") is not None
    )


def _has_strong_confirmation_evidence(anomaly: Mapping[str, Any]) -> bool:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return False
    rule_id = str(anomaly.get("rule_id") or "")
    if rule_id == "R03":
        return bool(evidence.get("page_error") or evidence.get("runtime_error") or evidence.get("console_error"))
    if rule_id == "R07":
        return bool(evidence.get("issue_type") and evidence.get("target_role"))
    if rule_id == "R08":
        return bool(evidence.get("constraint_violated") and evidence.get("success_evidence"))
    if rule_id == "R09":
        return bool(evidence.get("requested_operation") and (evidence.get("violating_entity_ids") or evidence.get("issue_type")))
    if rule_id == "R10":
        return bool(evidence.get("entity_id") and (evidence.get("field") or evidence.get("mismatched_fields")))
    if rule_id == "R11":
        return bool(evidence.get("expected_value") is not None and evidence.get("actual_value") is not None)
    if rule_id == "R13":
        status = _safe_int(evidence.get("status_code"))
        return bool(evidence.get("response_received") and status is not None and 500 <= status < 600)
    if rule_id == "R14":
        return bool(evidence.get("response_received") is False and evidence.get("failure_type"))
    if rule_id == "R15":
        return bool(evidence.get("timeout") and evidence.get("operation_completed") is False)
    return False


def _r06_confirmation_evidence_qualified(anomaly: Mapping[str, Any]) -> bool:
    """Require functional clipping evidence before an R06 reproduction confirms."""
    if str(anomaly.get("rule_id") or _anomaly_rule_id(anomaly)) != "R06":
        return True
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return False
    return str(evidence.get("functional_impact") or "UNKNOWN").upper() in {"MEDIUM", "HIGH"}


def _r10_confirmation_evidence_qualified(anomaly: Mapping[str, Any]) -> bool:
    if str(anomaly.get("rule_id") or _anomaly_rule_id(anomaly)) != "R10":
        return True
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return False
    if str(evidence.get("relation") or "").upper() != "NOT_EQUAL":
        return False
    if str(evidence.get("pair_method") or "") not in {"EXACT_KEY", "FORM_NAME_VALUE", "LABEL_VALUE", "TABLE_HEADER_CELL"}:
        return False
    try:
        confidence = semantic_pair_confidence(evidence)
        if confidence is None or confidence < 0.85:
            return False
    except (TypeError, ValueError):
        return False
    return bool(evidence.get("entity_id") and (evidence.get("field") or evidence.get("mismatched_fields")))


def _confirmation_action_identity(action_id: Any, step_id: Any) -> str:
    if action_id is not None and str(action_id) != "":
        return str(action_id)
    if step_id is not None and str(step_id) != "":
        return f"step:{step_id}"
    return ""


def _r10_independent_reproduction_seen(record: Mapping[str, Any]) -> bool:
    if int(record.get("seen_count", 0) or 0) < 2:
        return False
    first_action = str(record.get("first_seen_action_id") or "")
    last_action = str(record.get("last_seen_action_id") or "")
    if first_action and last_action and first_action == last_action:
        return False
    return True


def _r10_confirmation_source(agent_mode: Any) -> str:
    mode = str(agent_mode or "").upper()
    if "DQN" in mode:
        return "DQN_REPRODUCTION"
    if "PPO" in mode:
        return "OTHER_INDEPENDENT_ACTION"
    return "OTHER_INDEPENDENT_ACTION"


def _build_anomaly_signature(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    rule_id = str(anomaly.get("rule_id") or "")
    anomaly_type = str(anomaly.get("type") or "")
    issue_type = str(anomaly.get("issue_type") or evidence.get("issue_type") or anomaly_type)
    parts: List[str] = [rule_id, anomaly_type, issue_type]
    if rule_id == "R01":
        parts.extend([str(anomaly.get("action_type") or ""), str(anomaly.get("target_id") or ""), str(anomaly.get("url_path") or "")])
    elif rule_id == "R02":
        parts.extend([_signature_value(evidence.get("expected_url") or evidence.get("clicked_href")), _signature_value(evidence.get("actual_url") or evidence.get("after_url"))])
    elif rule_id == "R03":
        parts.extend([_signature_value(evidence.get("error_type")), _normalize_error_identity(evidence.get("error_message"))])
    elif rule_id == "R04":
        parts.append(_signature_value(evidence.get("entity_key") or anomaly.get("entity_id")))
    elif rule_id == "R05":
        parts.extend([_signature_value(evidence.get("target_bid")), _signature_value(evidence.get("occluder_bid"))])
    elif rule_id == "R06":
        parts.extend([_signature_value(evidence.get("target_bid")), _viewport_category(evidence.get("viewport_bbox") or evidence.get("viewport"))])
    elif rule_id == "R07":
        parts.extend([issue_type, str(anomaly.get("target_id") or "")])
    elif rule_id == "R08":
        parts.extend([_signature_value(evidence.get("constraint_type")), str(anomaly.get("target_id") or "")])
    elif rule_id == "R09":
        parts.extend([_signature_value(evidence.get("requested_operation")), _signature_value(evidence.get("requested_key")), _signature_value(evidence.get("requested_value"))])
    elif rule_id == "R10":
        parts.extend([str(anomaly.get("entity_type") or ""), str(anomaly.get("entity_id") or ""), _r10_signature_field(evidence)])
    elif rule_id == "R11":
        parts.extend([_signature_value(evidence.get("calculation_type")), str(anomaly.get("url_path") or "")])
    elif rule_id == "R12":
        parts.extend([_signature_value(evidence.get("operation")), str(anomaly.get("entity_id") or ""), _signature_value(evidence.get("field"))])
    elif rule_id == "R13":
        parts.extend([_signature_value(evidence.get("method")), _normalize_signature_path(evidence.get("request_path")), _signature_value(evidence.get("status_code"))])
    elif rule_id == "R14":
        parts.extend([_signature_value(evidence.get("method")), _normalize_signature_path(evidence.get("request_path")), _signature_value(evidence.get("failure_type"))])
    elif rule_id == "R15":
        parts.extend([_signature_value(evidence.get("operation_type")), _normalize_signature_path(evidence.get("request_path")), _signature_value(evidence.get("timeout_type"))])
    else:
        parts.extend([str(anomaly.get("target_id") or ""), str(anomaly.get("entity_id") or ""), str(anomaly.get("url_path") or "")])
    return "|".join(_redact_signature_part(part) for part in parts if part is not None)


def _anomaly_target_id(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    for key in ("target_id", "target_bid", "clicked_bid", "bid"):
        value = anomaly.get(key) or evidence.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    target = evidence.get("target")
    if isinstance(target, Mapping):
        for key in ("bid", "id", "role", "text", "name"):
            value = target.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _anomaly_url_path(
    anomaly: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
) -> str:
    evidence = anomaly.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    for key in ("request_path", "url_path", "actual_url", "after_url", "url", "before_url"):
        value = anomaly.get(key) or evidence.get(key)
        if value:
            return _normalize_signature_path(value)
    return _normalize_signature_path(_url(after_observation) or _url(before_observation))


def _normalize_signature_path(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    parsed = urlparse(text)
    path = parsed.path or text.split("?", 1)[0]
    return _r13_redact(path or "/")


def _signature_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(_signature_value(item) for item in value)
    return str(value).strip()


def _redact_signature_part(value: Any) -> str:
    text = _r13_redact(str(value or ""))
    text = re.sub(r"(?i)(password|token|authorization|cookie|session|secret|api_key|key)[:=][^|&\s]+", r"\1=<redacted>", text)
    return text


def _normalize_error_identity(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    return _redact_signature_part(text[:160])


def _viewport_category(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        width = _safe_float(value[2]) or 0.0
        return "mobile" if width and width < 600 else "desktop"
    return ""


def _r10_signature_field(evidence: Mapping[str, Any]) -> str:
    field = evidence.get("field")
    if field:
        return str(field)
    mismatches = evidence.get("mismatched_fields")
    if isinstance(mismatches, list) and mismatches:
        first = mismatches[0]
        if isinstance(first, Mapping):
            return str(first.get("field") or "")
    return ""


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _url(observation: Mapping[str, Any]) -> str:
    return str(observation.get("page_state", {}).get("url", "") or "")


def _state_signature(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    page_state = observation.get("page_state", {})
    return (
        page_state.get("url"),
        page_state.get("title"),
        page_state.get("page_text_length"),
        page_state.get("dom_node_count"),
        len(observation.get("candidate_elements", []) or []),
        page_state.get("cart_count"),
    )


def _is_r01_no_response(
    *,
    no_url_change: bool,
    no_state_change: bool,
    no_text_change: bool,
    no_candidate_change: bool,
    no_feedback: bool,
    no_network_change: bool,
    no_runtime_signal_delta: bool,
    modal_opened: bool,
    form_opened: bool,
    target_state_changed: bool,
    last_action_error: bool,
) -> bool:
    return (
        not last_action_error
        and no_url_change
        and no_state_change
        and no_text_change
        and no_candidate_change
        and no_feedback
        and no_network_change
        and no_runtime_signal_delta
        and not modal_opened
        and not form_opened
        and not target_state_changed
    )


def _is_r02_wrong_navigation(
    *,
    expected_url: str | None,
    actual_url: str | None,
    expected_entity_id: str | None = None,
    actual_entity_id: str | None = None,
    last_action_error: bool = False,
    allowed_redirects: set[str] | None = None,
    base_url: str = "",
) -> bool:
    if last_action_error:
        return False
    expected_entity = _clean_entity_id(expected_entity_id)
    actual_entity = _clean_entity_id(actual_entity_id)
    if expected_entity and actual_entity:
        return expected_entity != actual_entity
    expected_route = _normalize_navigation_url(expected_url, base_url=base_url)
    actual_route = _normalize_navigation_url(actual_url, base_url=base_url)
    if not expected_route or not actual_route:
        return False
    if expected_route == actual_route:
        return False
    normalized_allowed = {
        route
        for route in (_normalize_navigation_url(item, base_url=base_url) for item in (allowed_redirects or set()))
        if route
    }
    if actual_route in normalized_allowed:
        return False
    return True


def _detect_r03_runtime_exception(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    last_action_error: bool,
) -> Dict[str, Any] | None:
    evidence = _r03_runtime_exception_evidence(before_observation, after_observation, action_info)
    if not evidence:
        return None
    return {
        "type": "runtime-exception",
        "confidence": 0.9,
        "evidence": {
            "rule_id": "R03",
            **evidence,
            "last_action_error": last_action_error,
        },
    }


def _has_r03_runtime_exception(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> bool:
    return bool(_r03_runtime_exception_evidence(before_observation, after_observation, action_info))


def _r03_runtime_exception_evidence(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> Dict[str, Any] | None:
    del before_observation
    page_error = _first_runtime_error_text(_runtime_values(after_observation, action_info, ("page_errors", "page_error", "pageerror")))
    runtime_error = _first_runtime_error_text(
        _runtime_values(after_observation, action_info, ("runtime_errors", "runtime_error", "uncaught_errors", "uncaught_error"))
    )
    console_error = _first_runtime_error_text(_console_error_values(after_observation, action_info))
    message = page_error or runtime_error or console_error
    if not message:
        return None
    return {
        "runtime_error": runtime_error,
        "page_error": page_error,
        "console_error": console_error,
        "error_type": _runtime_error_type(message),
        "error_message": message[:500],
    }


def _detect_r03_rendering_failure(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    last_action_error: bool,
) -> Dict[str, Any] | None:
    evidence = _r03_rendering_failure_evidence(before_observation, after_observation, action_info, last_action_error)
    if not evidence:
        return None
    return {
        "type": "empty-state-rendering",
        "confidence": 0.82 if evidence.get("invalid_visible_value") else 0.74,
        "evidence": {
            "rule_id": "R03",
            **evidence,
        },
    }


def _is_r03_rendering_failure(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> bool:
    return bool(_r03_rendering_failure_evidence(before_observation, after_observation, action_info, False))


def _r03_rendering_failure_evidence(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if last_action_error or _has_api_failure(action_info):
        return None
    before_text = _page_text(before_observation)
    after_text = _page_text(after_observation)
    before_len = _page_text_length(before_observation)
    after_len = _page_text_length(after_observation)
    before_count = len(before_observation.get("candidate_elements", []) or [])
    after_count = len(after_observation.get("candidate_elements", []) or [])
    normal_empty_state = _normal_empty_state_visible(after_observation)
    invalid_value = _invalid_rendered_value(after_text)
    runtime_error_present = _has_r03_runtime_exception(before_observation, after_observation, action_info)
    if invalid_value:
        return {
            "page_text_before": before_text[:500],
            "page_text_after": after_text[:500],
            "page_text_delta": abs(after_len - before_len),
            "candidate_count_before": before_count,
            "candidate_count_after": after_count,
            "invalid_visible_value": invalid_value,
            "normal_empty_state_detected": normal_empty_state,
            "runtime_error_present": runtime_error_present,
        }
    content_collapsed = (
        before_len >= 300
        and after_len <= max(80, int(before_len * 0.2))
        and before_count >= 5
        and after_count <= max(2, int(before_count * 0.25))
    )
    if content_collapsed and not normal_empty_state:
        return {
            "page_text_before": before_text[:500],
            "page_text_after": after_text[:500],
            "page_text_delta": abs(after_len - before_len),
            "candidate_count_before": before_count,
            "candidate_count_after": after_count,
            "invalid_visible_value": "",
            "normal_empty_state_detected": normal_empty_state,
            "runtime_error_present": runtime_error_present,
        }
    return None


def _detect_r05_layout_overlap(observation: Mapping[str, Any], action_type: str) -> Dict[str, Any] | None:
    if _has_modal_or_dialog(observation) or _loading_visible(observation):
        return None
    viewport = _viewport_bbox(observation)
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    best: Dict[str, Any] | None = None
    for target in candidates:
        if not isinstance(target, Mapping):
            continue
        target_bbox = _bbox(target)
        if not target_bbox or not _target_in_viewport(target_bbox, viewport):
            continue
        if not _is_actionable_candidate(target):
            continue
        target_area = _bbox_area(target_bbox)
        if target_area <= 0:
            continue
        for occluder in candidates:
            if not isinstance(occluder, Mapping) or occluder is target:
                continue
            occluder_bbox = _bbox(occluder)
            if not occluder_bbox:
                continue
            intersection_area = _bbox_intersection_area(target_bbox, occluder_bbox)
            if intersection_area <= 0:
                continue
            occlusion_ratio = intersection_area / max(1.0, target_area)
            same_element = _same_layout_element(target, occluder)
            parent_child_like = _parent_child_like_overlap(target, occluder, target_bbox, occluder_bbox)
            if not _is_r05_layout_overlap(
                target=target,
                occluder=occluder,
                viewport=viewport,
                intersection_area=intersection_area,
                target_area=target_area,
                occlusion_ratio=occlusion_ratio,
                same_element=same_element,
                parent_child_like=parent_child_like,
            ):
                continue
            evidence = {
                "rule_id": "R05",
                "target_bid": target.get("bid"),
                "target_role": target.get("role"),
                "target_tag": target.get("tag"),
                "target_bbox": target_bbox,
                "occluder_bid": occluder.get("bid"),
                "occluder_role": occluder.get("role"),
                "occluder_tag": occluder.get("tag"),
                "occluder_bbox": occluder_bbox,
                "intersection_area": round(intersection_area, 4),
                "target_area": round(target_area, 4),
                "occlusion_ratio": round(occlusion_ratio, 4),
                "target_actionable": True,
                "same_element": same_element,
                "parent_child_like": parent_child_like,
                "viewport": viewport,
                "action_type": action_type,
            }
            if best is None or float(evidence["occlusion_ratio"]) > float(best["evidence"]["occlusion_ratio"]):
                best = {
                    "type": "layout-overlap",
                    "confidence": min(0.95, 0.6 + occlusion_ratio * 0.35),
                    "evidence": evidence,
                }
    return best


def _is_r05_layout_overlap(
    *,
    target: Mapping[str, Any],
    occluder: Mapping[str, Any],
    viewport: List[float],
    intersection_area: float,
    target_area: float,
    occlusion_ratio: float,
    same_element: bool = False,
    parent_child_like: bool = False,
) -> bool:
    del intersection_area, target_area
    return bool(
        viewport
        and _is_actionable_candidate(target)
        and _candidate_visible(target)
        and _candidate_visible(occluder)
        and not same_element
        and not parent_child_like
        and not _normal_overlay_candidate(occluder)
        and occlusion_ratio >= 0.25
    )


def _detect_r06_layout_overflow(observation: Mapping[str, Any], action_type: str) -> Dict[str, Any] | None:
    viewport = _viewport_bbox(observation)
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    best: Dict[str, Any] | None = None
    for target in candidates:
        if not isinstance(target, Mapping):
            continue
        target_bbox = _bbox(target)
        if not target_bbox or _bbox_area(target_bbox) <= 0:
            continue
        target_area = _bbox_area(target_bbox)
        visible_area = _bbox_intersection_area(target_bbox, viewport)
        visible_ratio = visible_area / max(1.0, target_area)
        overflow_left = target_bbox[0] < viewport[0] - 2
        overflow_right = target_bbox[0] + target_bbox[2] > viewport[0] + viewport[2] + 2
        overflow_top = target_bbox[1] < viewport[1] - 2
        overflow_bottom = target_bbox[1] + target_bbox[3] > viewport[1] + viewport[3] + 2
        horizontal_overflow = bool(overflow_left or overflow_right)
        vertical_overflow = bool(overflow_top or overflow_bottom)
        functional = _r06_functional_evidence(
            observation=observation,
            target=target,
            target_bbox=target_bbox,
            viewport=viewport,
            visible_ratio=visible_ratio,
            horizontal_overflow=horizontal_overflow,
            vertical_overflow=vertical_overflow,
        )
        live_evidence = target.get("interactive_layout_evidence")
        if isinstance(live_evidence, dict):
            live_evidence.update({
                "functional_impact": functional["functional_impact"],
                "functional_impact_reasons": functional["functional_impact_reasons"],
                "container_bbox": functional["container_bbox"],
                "container_overflow": functional["container_overflow"],
                "container_clipping": functional["container_clipping"],
                "container_clipping_ratio": functional["container_clipping_ratio"],
                "intentional_scrollable": functional["intentional_scrollable"],
                "can_reach_by_scroll_x": functional["can_reach_by_scroll_x"],
                "can_reach_by_scroll_y": functional["can_reach_by_scroll_y"],
                "document_reachable_x": functional["document_reachable_x"],
                "document_reachable_y": functional["document_reachable_y"],
                "reachable_via_document_scroll": functional["reachable_via_document_scroll"],
                "reachability_source": functional["reachability_source"],
            })
        geometry_anomaly = bool(
            (visible_ratio < 0.75 and (horizontal_overflow or vertical_overflow))
            or functional["container_clipping"]
            or functional["container_overflow"]
        )
        if not geometry_anomaly or not _r06_relevant_content(target) or not _candidate_visible(target):
            continue
        if _r06_impact_rank(functional["functional_impact"]) < _r06_impact_rank("MEDIUM"):
            continue
        scroll_reachable = bool(functional["can_reach_by_scroll_y"])
        evidence = {
            "rule_id": "R06",
            "target_bid": target.get("bid"),
            "target_role": target.get("role"),
            "target_tag": target.get("tag"),
            "target_bbox": target_bbox,
            "viewport_bbox": viewport,
            "viewport_width": viewport[2],
            "viewport_height": viewport[3],
            "container_bbox": functional["container_bbox"],
            "target_area": round(target_area, 4),
            "visible_area": round(visible_area, 4),
            "visible_ratio": round(visible_ratio, 4),
            "clipped_ratio": round(max(0.0, 1.0 - visible_ratio), 4),
            "clipping_ratio": round(max(0.0, 1.0 - visible_ratio), 4),
            "overflow_left": overflow_left,
            "overflow_right": overflow_right,
            "overflow_top": overflow_top,
            "overflow_bottom": overflow_bottom,
            "horizontal_overflow": horizontal_overflow,
            "vertical_overflow": vertical_overflow,
            "viewport_overflow": horizontal_overflow or vertical_overflow,
            "container_overflow": functional["container_overflow"],
            "container_clipping": functional["container_clipping"],
            "container_clipping_ratio": functional["container_clipping_ratio"],
            "overflow_x": functional["overflow_x"],
            "overflow_y": functional["overflow_y"],
            "scroll_width": functional["scroll_width"],
            "client_width": functional["client_width"],
            "scroll_height": functional["scroll_height"],
            "client_height": functional["client_height"],
            "intentional_scrollable": functional["intentional_scrollable"],
            "can_reach_by_scroll_x": functional["can_reach_by_scroll_x"],
            "can_reach_by_scroll_y": functional["can_reach_by_scroll_y"],
            "document_scroll_width": functional["document_scroll_width"],
            "document_scroll_height": functional["document_scroll_height"],
            "document_client_width": functional["document_client_width"],
            "document_client_height": functional["document_client_height"],
            "document_can_scroll_x": functional["document_can_scroll_x"],
            "document_can_scroll_y": functional["document_can_scroll_y"],
            "target_document_bbox": functional["target_document_bbox"],
            "document_reachable_x": functional["document_reachable_x"],
            "document_reachable_y": functional["document_reachable_y"],
            "reachable_via_document_scroll": functional["reachable_via_document_scroll"],
            "reachability_source": functional["reachability_source"],
            "blocking_clipping_ancestor": functional["blocking_clipping_ancestor"],
            "scroll_reachable": scroll_reachable,
            "target_actionable": _is_actionable_candidate(target),
            "visible": _candidate_visible(target),
            "enabled": bool(target.get("enabled", True)),
            "clickable": _is_clickable_target(target),
            "keyboard_reachable": functional["keyboard_reachable"],
            "functional_priority": functional["functional_priority"],
            "hit_test_valid": functional["hit_test_valid"],
            "hit_test_blocked": functional["hit_test_blocked"],
            "hit_element": functional["hit_element"],
            "functional_impact": functional["functional_impact"],
            "functional_impact_reasons": functional["functional_impact_reasons"],
            "text_length": functional["text_length"],
            "action_type": action_type,
        }
        if best is None or float(evidence["clipped_ratio"]) > float(best["evidence"]["clipped_ratio"]):
            best = {
                "type": "layout-overflow",
                "confidence": min(
                    0.98,
                    0.65
                    + float(evidence["clipped_ratio"]) * 0.2
                    + (0.1 if functional["functional_impact"] == "HIGH" else 0.0),
                ),
                "evidence": evidence,
            }
    return best


def _is_r06_layout_overflow(
    *,
    target: Mapping[str, Any],
    viewport: List[float],
    visible_ratio: float,
    horizontal_overflow: bool,
    vertical_overflow: bool,
    scroll_reachable: bool,
) -> bool:
    if not viewport or not _is_actionable_candidate(target) or not _candidate_visible(target):
        return False
    if visible_ratio >= 0.75:
        return False
    if horizontal_overflow:
        return True
    if vertical_overflow and not scroll_reachable:
        return True
    return False


_R06_IMPORTANT_ROLES = {
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox",
    "menuitem", "tab", "option", "select", "input", "textarea",
}
_R06_SCROLL_VALUES = {"auto", "scroll"}


def _r06_functional_evidence(
    *,
    observation: Mapping[str, Any],
    target: Mapping[str, Any],
    target_bbox: List[float],
    viewport: List[float],
    visible_ratio: float,
    horizontal_overflow: bool,
    vertical_overflow: bool,
) -> Dict[str, Any]:
    live = target.get("interactive_layout_evidence")
    live = live if isinstance(live, Mapping) else {}
    evidence_source = dict(live)
    evidence_source.update({key: value for key, value in target.items() if value is not None})
    style = evidence_source.get("computed_style") or evidence_source.get("style") or {}
    if not isinstance(style, Mapping):
        style = {}
    overflow_x = str(evidence_source.get("overflow_x") or style.get("overflow-x") or style.get("overflowX") or "").lower() or None
    overflow_y = str(evidence_source.get("overflow_y") or style.get("overflow-y") or style.get("overflowY") or "").lower() or None
    overflow = str(evidence_source.get("overflow") or style.get("overflow") or "").lower()
    overflow_x = overflow_x or (overflow if overflow in _R06_SCROLL_VALUES | {"hidden", "clip", "visible"} else None)
    overflow_y = overflow_y or (overflow if overflow in _R06_SCROLL_VALUES | {"hidden", "clip", "visible"} else None)

    def number(*keys: str) -> float | None:
        for key in keys:
            value = evidence_source.get(key)
            if value is None and isinstance(style, Mapping):
                value = style.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                pass
        return None

    scroll_width = number("scroll_width", "scrollWidth")
    client_width = number("client_width", "clientWidth")
    scroll_height = number("scroll_height", "scrollHeight")
    client_height = number("client_height", "clientHeight")
    effective_ancestor = evidence_source.get("effective_clipping_ancestor")
    effective_ancestor = effective_ancestor if isinstance(effective_ancestor, Mapping) else {}
    container_bbox = _r06_optional_bbox(
        evidence_source.get("container_bbox")
        or evidence_source.get("clipping_container_bbox")
        or effective_ancestor.get("bbox")
    )
    container_overflow = bool(
        (scroll_width is not None and client_width is not None and scroll_width > client_width)
        or (scroll_height is not None and client_height is not None and scroll_height > client_height)
    )

    can_x = _r06_optional_bool(evidence_source, "can_reach_by_scroll_x", "canReachByScrollX", "reachable_via_scroll_x")
    can_y = _r06_optional_bool(evidence_source, "can_reach_by_scroll_y", "canReachByScrollY", "reachable_via_scroll_y")
    if can_x is None and overflow_x in _R06_SCROLL_VALUES and scroll_width is not None and client_width is not None:
        can_x = scroll_width > client_width
    if can_y is None and overflow_y in _R06_SCROLL_VALUES and scroll_height is not None and client_height is not None:
        can_y = scroll_height > client_height
    if can_y is None and vertical_overflow:
        can_y = _vertical_scroll_reachable(observation, target_bbox, viewport)
    intentional_scrollable = bool(
        (overflow_x in _R06_SCROLL_VALUES and can_x is True)
        or (overflow_y in _R06_SCROLL_VALUES and can_y is True)
    )

    hit_test_valid = _r06_optional_bool(evidence_source, "hit_test_valid", "hitTestValid")
    hit_test_blocked = _r06_optional_bool(evidence_source, "hit_test_blocked", "hitTestBlocked")
    hit_element = evidence_source.get("hit_element") or evidence_source.get("hitTestElement")
    keyboard_reachable = _r06_keyboard_reachable(evidence_source)
    functional_priority = _r06_functional_priority(target)
    text_length = len(_candidate_text(target).strip())
    clipped_ratio = max(0.0, 1.0 - visible_ratio)
    container_clipping = bool(
        container_bbox
        and _bbox_intersection_area(target_bbox, container_bbox) < _bbox_area(target_bbox)
    )
    container_clipping_ratio = (
        max(0.0, 1.0 - _bbox_intersection_area(target_bbox, container_bbox) / max(1.0, _bbox_area(target_bbox)))
        if container_bbox
        else 0.0
    )
    no_normal_access = (
        ((horizontal_overflow or container_clipping) and can_x is False)
        or (vertical_overflow and can_y is False)
        or (container_clipping and overflow_x in {"hidden", "clip"} and can_x is not True)
    )
    hidden_container = overflow_x in {"hidden", "clip"} or overflow_y in {"hidden", "clip"}
    effective_clipping_ratio = max(clipped_ratio, container_clipping_ratio)
    document_reachable_x = _r06_optional_bool(evidence_source, "document_reachable_x", "documentReachableX")
    document_reachable_y = _r06_optional_bool(evidence_source, "document_reachable_y", "documentReachableY")
    reachable_via_document_scroll = _r06_optional_bool(evidence_source, "reachable_via_document_scroll", "reachableViaDocumentScroll")
    reachability_source = evidence_source.get("reachability_source")
    blocking_clipping_ancestor = _r06_optional_bool(evidence_source, "blocking_clipping_ancestor", "blockingClippingAncestor")
    reasons: List[str] = []

    if hit_test_blocked is True:
        reasons.append("hit_test_blocked")
    if hidden_container:
        reasons.append("overflow_hidden_or_clip")
    if no_normal_access:
        reasons.append("not_reachable_by_normal_scroll")
    if functional_priority:
        reasons.append("important_interactive_or_functional_target")
    if text_length and clipped_ratio >= 0.25:
        reasons.append("meaningful_text_clipped")

    if hit_test_valid is True and hit_test_blocked is not True and not (
        text_length > 0 and container_clipping and hidden_container
    ):
        impact = "NONE"
        reasons.append("hit_test_valid")
    elif hit_test_blocked is True and (no_normal_access or functional_priority):
        impact = "HIGH"
    elif hidden_container and effective_clipping_ratio >= 0.5 and (functional_priority or text_length > 0) and no_normal_access:
        impact = "HIGH"
    elif (
        effective_clipping_ratio >= 0.25
        and (functional_priority or text_length > 0)
        and no_normal_access
    ):
        impact = "MEDIUM"
    elif intentional_scrollable or hit_test_valid is True or (
        _candidate_visible(target) and _is_clickable_target(target) and visible_ratio >= 0.5
    ):
        impact = "NONE"
        reasons.append("target_remains_usable")
    else:
        impact = "UNKNOWN"

    return {
        "container_bbox": container_bbox,
        "container_overflow": container_overflow,
        "container_clipping": container_clipping,
        "container_clipping_ratio": round(container_clipping_ratio, 4),
        "overflow_x": overflow_x,
        "overflow_y": overflow_y,
        "scroll_width": scroll_width,
        "client_width": client_width,
        "scroll_height": scroll_height,
        "client_height": client_height,
        "intentional_scrollable": intentional_scrollable,
        "can_reach_by_scroll_x": can_x,
        "can_reach_by_scroll_y": can_y,
        "document_scroll_width": number("document_scroll_width", "documentScrollWidth"),
        "document_scroll_height": number("document_scroll_height", "documentScrollHeight"),
        "document_client_width": number("document_client_width", "documentClientWidth"),
        "document_client_height": number("document_client_height", "documentClientHeight"),
        "document_can_scroll_x": _r06_optional_bool(evidence_source, "document_can_scroll_x", "documentCanScrollX"),
        "document_can_scroll_y": _r06_optional_bool(evidence_source, "document_can_scroll_y", "documentCanScrollY"),
        "target_document_bbox": evidence_source.get("target_document_bbox") or evidence_source.get("targetDocumentBBox"),
        "document_reachable_x": document_reachable_x,
        "document_reachable_y": document_reachable_y,
        "reachable_via_document_scroll": reachable_via_document_scroll,
        "reachability_source": reachability_source,
        "blocking_clipping_ancestor": blocking_clipping_ancestor,
        "hit_test_valid": hit_test_valid,
        "hit_test_blocked": hit_test_blocked,
        "hit_element": hit_element,
        "keyboard_reachable": keyboard_reachable,
        "functional_priority": functional_priority,
        "text_length": text_length,
        "functional_impact": impact,
        "functional_impact_reasons": reasons,
    }


def _r06_optional_bbox(value: Any) -> List[float] | None:
    if isinstance(value, Mapping):
        try:
            return [
                float(value.get("x", 0)),
                float(value.get("y", 0)),
                float(value.get("width", 0)),
                float(value.get("height", 0)),
            ]
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _r06_optional_bool(target: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key not in target:
            continue
        value = target.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() in {"true", "yes", "1"}:
                return True
            if value.lower() in {"false", "no", "0"}:
                return False
        return bool(value)
    return None


def _r06_keyboard_reachable(target: Mapping[str, Any]) -> bool | None:
    explicit = _r06_optional_bool(target, "keyboard_reachable", "keyboardReachable")
    if explicit is not None:
        return explicit
    if _r06_optional_bool(target, "disabled", "aria_disabled", "ariaDisabled") is True:
        return False
    if "tabindex" in target:
        try:
            return int(target.get("tabindex")) >= 0
        except (TypeError, ValueError):
            return False
    role = str(target.get("role") or "").lower()
    tag = str(target.get("tag") or "").lower()
    return True if role in _R06_IMPORTANT_ROLES or tag in {"button", "a", "input", "textarea", "select"} else None


def _r06_functional_priority(target: Mapping[str, Any]) -> bool:
    role = str(target.get("role") or "").lower()
    tag = str(target.get("tag") or "").lower()
    return bool(
        target.get("functional_priority")
        or target.get("functional_priority_candidate")
        or target.get("is_high_value_functional_candidate")
        or target.get("is_interactive")
        or role in _R06_IMPORTANT_ROLES
        or tag in {"button", "a", "input", "textarea", "select"}
    )


def _r06_relevant_content(target: Mapping[str, Any]) -> bool:
    """Keep actionable controls and meaningful text-bearing content in scope."""
    return bool(_is_actionable_candidate(target) or _candidate_text(target).strip())


def _r06_impact_rank(value: Any) -> int:
    return {"NONE": 0, "LOW": 1, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}.get(str(value or "UNKNOWN").upper(), 1)


def _detect_r07_accessibility(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    last_action_error: bool,
) -> List[Dict[str, Any]]:
    if last_action_error:
        return []
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return []
    anomalies: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        issue_type = _r07_accessibility_issue_type(candidate)
        if not issue_type:
            continue
        key = (str(candidate.get("bid") or ""), issue_type)
        if key in seen:
            continue
        seen.add(key)
        anomalies.append(
            {
                "type": "accessibility-issue",
                "confidence": 0.78,
                "evidence": _r07_accessibility_evidence(candidate, issue_type, action_info),
            }
        )
    return anomalies


def _r07_accessibility_issue_type(candidate: Mapping[str, Any]) -> str:
    if _candidate_hidden(candidate) or _candidate_disabled(candidate) or not _candidate_visible(candidate):
        return ""
    if _is_r07_image_without_alt(candidate):
        return "missing-image-alt"
    if _is_r07_form_control(candidate):
        if not _r07_accessible_name(candidate, include_visible_text=False):
            return "missing-form-label"
    if _is_r07_interactive_name_required(candidate):
        if not _r07_accessible_name(candidate, include_visible_text=True):
            return "missing-accessible-name"
    if _is_r07_keyboard_unreachable(candidate):
        return "keyboard-unreachable"
    return ""


def _r07_accessibility_evidence(
    candidate: Mapping[str, Any],
    issue_type: str,
    action_info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "rule_id": "R07",
        "issue_type": issue_type,
        "target_bid": candidate.get("bid"),
        "target_role": candidate.get("role"),
        "target_tag": candidate.get("tag"),
        "accessible_name": _r07_accessible_name(candidate, include_visible_text=True),
        "aria_label": _first_text(candidate, ("aria_label", "aria-label")),
        "aria_labelledby": _first_text(candidate, ("aria_labelledby", "aria-labelledby")),
        "label": _first_text(candidate, ("label", "associated_label")),
        "alt": candidate.get("alt"),
        "alt_present": candidate.get("alt_present"),
        "focusable": candidate.get("focusable", candidate.get("keyboard_focusable")),
        "tabindex": candidate.get("tabindex", candidate.get("tabIndex")),
        "disabled": _candidate_disabled(candidate),
        "hidden": _candidate_hidden(candidate),
        "actionable": _is_actionable_candidate(candidate),
        "action_type": str((action_info.get("action") or action_info).get("action_type") or ""),
    }


def _is_r07_accessibility_issue(candidate: Mapping[str, Any]) -> bool:
    return bool(_r07_accessibility_issue_type(candidate))


def _is_r07_form_control(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    input_type = str(candidate.get("type") or candidate.get("input_type") or "").lower()
    return bool(
        tag in {"input", "textarea", "select"}
        or role in {"textbox", "searchbox", "combobox"}
        or input_type in {"text", "email", "password", "search", "number", "tel", "url"}
    )


def _is_r07_interactive_name_required(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    if _is_r07_form_control(candidate):
        return False
    return bool(
        _is_actionable_candidate(candidate)
        and (
            role in {"button", "link", "checkbox", "radio", "menuitem", "tab"}
            or tag in {"button", "a", "summary", "option"}
        )
    )


def _is_r07_keyboard_unreachable(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    if role not in {"button", "link", "checkbox", "radio", "menuitem", "tab"}:
        return False
    if _is_native_keyboard_accessible(candidate):
        return False
    focusable = candidate.get("keyboard_focusable", candidate.get("focusable"))
    if focusable is not False:
        return False
    return not _tabindex_focusable(candidate.get("tabindex", candidate.get("tabIndex")))


def _is_native_keyboard_accessible(candidate: Mapping[str, Any]) -> bool:
    tag = str(candidate.get("tag") or "").lower()
    href = str(candidate.get("href") or "")
    if tag in {"button", "input", "textarea", "select", "option"}:
        return True
    if tag == "a" and href:
        return True
    return False


def _tabindex_focusable(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return int(str(value).strip()) >= 0
    except (TypeError, ValueError):
        return False


def _is_r07_image_without_alt(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    if tag != "img" and role != "img":
        return False
    if role in {"presentation", "none"} or _candidate_aria_hidden(candidate):
        return False
    if bool(candidate.get("alt_present")) and str(candidate.get("alt") or "") == "":
        return False
    return not _r07_accessible_name(candidate, include_visible_text=False, include_alt=True)


def _r07_accessible_name(
    candidate: Mapping[str, Any],
    *,
    include_visible_text: bool,
    include_alt: bool = True,
) -> str:
    keys = [
        "accessible_name",
        "computed_name",
        "name",
        "aria_label",
        "aria-label",
        "aria_labelledby",
        "aria-labelledby",
        "label",
        "associated_label",
    ]
    if include_visible_text:
        keys.extend(["text", "title"])
    else:
        keys.append("title")
    if include_alt:
        keys.append("alt")
    for key in keys:
        value = _meaningful_accessible_text(candidate.get(key))
        if value:
            return value
    return ""


def _first_text(candidate: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _meaningful_accessible_text(candidate.get(key))
        if value:
            return value
    return ""


def _meaningful_accessible_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null", "undefined", "nan"}:
        return ""
    if re.fullmatch(r"(element|node|dom)-?\d+", lowered):
        return ""
    return text


def _candidate_disabled(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("disabled") is True:
        return True
    if str(candidate.get("aria_disabled") or candidate.get("aria-disabled") or "").lower() == "true":
        return True
    return not bool(candidate.get("enabled", True))


def _candidate_hidden(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("hidden") is True:
        return True
    return _candidate_aria_hidden(candidate)


def _candidate_aria_hidden(candidate: Mapping[str, Any]) -> bool:
    return str(candidate.get("aria_hidden") or candidate.get("aria-hidden") or "").lower() == "true"


def _detect_r08_validation_issue(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if last_action_error or not _r08_submit_action(action_type, clicked_candidate, action_info):
        return None
    violated = _r08_violated_fields(before_observation, action_info)
    if not violated:
        return None
    validation_feedback = _r08_validation_feedback_detected(after_observation)
    success_evidence = _r08_success_evidence(before_observation, after_observation, action_info, violated)
    if not _is_r08_validation_failure(
        constraint_violated=True,
        submit_action=True,
        validation_feedback_detected=validation_feedback,
        success_evidence=success_evidence,
    ):
        return None
    field = violated[0]
    anomaly_type, issue_type = _r08_anomaly_type(field)
    return {
        "type": anomaly_type,
        "confidence": 0.86 if success_evidence else 0.7,
        "evidence": _r08_evidence(
            field,
            issue_type,
            action_type,
            validation_feedback,
            success_evidence,
        ),
    }


def _is_r08_validation_failure(
    *,
    constraint_violated: bool,
    submit_action: bool,
    validation_feedback_detected: bool,
    success_evidence: List[str],
) -> bool:
    strong_success = {"action_success", "success_message", "successful_write_response", "form_closed", "persisted_value"}
    return bool(
        constraint_violated
        and submit_action
        and not validation_feedback_detected
        and strong_success.intersection(set(success_evidence))
    )


def _r08_violated_fields(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return []
    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not _r08_form_candidate(candidate):
            continue
        violation = _r08_constraint_violation(candidate=candidate, action_info=action_info)
        if violation:
            results.append(violation)
    return results


def _r08_constraint_violation(
    *,
    candidate: Mapping[str, Any],
    action_info: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    del action_info
    value = _r08_candidate_value(candidate)
    constraints = _r08_candidate_constraints(candidate)
    if not _r08_has_constraint(constraints):
        return None
    constraint_type = ""
    expected: Any = None
    actual: Any = value
    if constraints["required"] and not value.strip():
        constraint_type = "required"
        expected = True
        actual = "empty"
    elif constraints["minlength"] is not None and len(value) < int(constraints["minlength"]):
        constraint_type = "minlength"
        expected = int(constraints["minlength"])
        actual = len(value)
    elif constraints["maxlength"] is not None and len(value) > int(constraints["maxlength"]):
        constraint_type = "maxlength"
        expected = int(constraints["maxlength"])
        actual = len(value)
    elif constraints["minimum"] is not None and _r08_number(value) is not None and _r08_number(value) < float(constraints["minimum"]):
        constraint_type = "min"
        expected = float(constraints["minimum"])
        actual = _r08_number(value)
    elif constraints["maximum"] is not None and _r08_number(value) is not None and _r08_number(value) > float(constraints["maximum"]):
        constraint_type = "max"
        expected = float(constraints["maximum"])
        actual = _r08_number(value)
    elif constraints["pattern"] and not _r08_pattern_matches(value, str(constraints["pattern"])):
        constraint_type = "pattern"
        expected = constraints["pattern"]
    elif constraints["input_type"] == "email" and value.strip() and not _r08_valid_email(value):
        constraint_type = "type"
        expected = "email"
    elif constraints["input_type"] == "url" and value.strip() and not _r08_valid_url(value):
        constraint_type = "type"
        expected = "url"
    if not constraint_type:
        return None
    return {
        "candidate": candidate,
        "value": value,
        "constraints": constraints,
        "constraint_type": constraint_type,
        "expected": expected,
        "actual": actual,
        "persisted_value": "",
    }


def _r08_candidate_constraints(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "required": _r08_bool(candidate.get("required")) or _r08_bool(candidate.get("aria_required") or candidate.get("aria-required")),
        "minlength": _r08_int(_r08_first_present(candidate, ("minlength", "min_length"))),
        "maxlength": _r08_int(_r08_first_present(candidate, ("maxlength", "max_length"))),
        "minimum": _r08_float(_r08_first_present(candidate, ("min", "minimum"))),
        "maximum": _r08_float(_r08_first_present(candidate, ("max", "maximum"))),
        "pattern": str(candidate.get("pattern") or "").strip(),
        "input_type": str(candidate.get("input_type") or candidate.get("type") or "").lower(),
    }


def _r08_has_constraint(constraints: Mapping[str, Any]) -> bool:
    return bool(
        constraints.get("required")
        or constraints.get("minlength") is not None
        or constraints.get("maxlength") is not None
        or constraints.get("minimum") is not None
        or constraints.get("maximum") is not None
        or constraints.get("pattern")
        or constraints.get("input_type") in {"email", "url"}
    )


def _r08_form_candidate(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return bool(tag in {"input", "textarea", "select"} or role in {"textbox", "searchbox", "combobox"} or candidate.get("is_form_field"))


def _r08_candidate_value(candidate: Mapping[str, Any]) -> str:
    for key in ("value", "current_value", "input_value", "submitted_value"):
        if key in candidate:
            return str(candidate.get(key) or "")
    return str(candidate.get("text") or "")


def _r08_submit_action(
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    action_info: Mapping[str, Any],
) -> bool:
    if action_type == "fill_input":
        return False
    if action_type == "press_enter":
        return True
    action = action_info.get("action", action_info)
    action_text = " ".join(str(action.get(key) or "") for key in ("action_type", "text", "name", "label")) if isinstance(action, Mapping) else ""
    if clicked_candidate and _looks_like_submit_button(clicked_candidate):
        return True
    if clicked_candidate:
        text = _candidate_text(clicked_candidate).lower()
        if any(token in text for token in ("submit", "save", "create", "post", "register", "apply", "send", "continue")):
            return True
    return any(token in action_text.lower() for token in ("submit", "save", "create", "post", "register", "apply", "send"))


def _r08_validation_feedback_detected(observation: Mapping[str, Any]) -> bool:
    if _validation_message_visible(observation) or _error_message_visible(observation):
        return True
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and str(candidate.get("aria_invalid") or candidate.get("aria-invalid") or "").lower() == "true":
                return True
    return False


def _r08_success_evidence(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    violated: List[Dict[str, Any]],
) -> List[str]:
    evidence: List[str] = []
    if _r08_action_success(action_info):
        evidence.append("action_success")
    if _r08_success_message_visible(after_observation):
        evidence.append("success_message")
    if _r08_successful_write_response(action_info):
        evidence.append("successful_write_response")
    if _url(before_observation) != _url(after_observation):
        evidence.append("route_transition")
    if _has_form_signal(before_observation) and not _has_form_signal(after_observation):
        evidence.append("form_closed")
    for item in violated:
        persisted = _r08_persisted_value(after_observation, item)
        if persisted:
            item["persisted_value"] = persisted
            evidence.append("persisted_value")
            break
    return sorted(set(evidence))


def _r08_action_success(action_info: Mapping[str, Any]) -> bool:
    for key in ("success", "submit_success", "save_success", "created", "creation_success", "applied", "api_success"):
        if bool(action_info.get(key)):
            return True
    action_result = action_info.get("action_result", {})
    if isinstance(action_result, Mapping) and bool(action_result.get("success")) and not bool(action_result.get("failed")):
        return True
    return False


def _r08_successful_write_response(action_info: Mapping[str, Any]) -> bool:
    for key in ("status_code", "response_status", "api_status", "network_status"):
        status = _r08_int(action_info.get(key))
        if status is not None and 200 <= status < 300:
            return True
    entries = action_info.get("network_entries", []) or action_info.get("network", []) or []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            method = str(entry.get("method") or "").upper()
            status = _r08_int(entry.get("status") or entry.get("status_code"))
            if method in {"POST", "PUT", "PATCH"} and status is not None and 200 <= status < 300:
                return True
    return False


def _r08_success_message_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    tokens = ("success", "saved", "created", "posted", "registered", "updated", "applied", "완료", "저장", "성공", "등록")
    return any(token in text for token in tokens)


def _r08_persisted_value(observation: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    candidate = item.get("candidate", {})
    if not isinstance(candidate, Mapping):
        return ""
    if _r08_is_password(candidate):
        return ""
    value = str(item.get("value") or "")
    if not value.strip():
        if _r08_empty_entity_created(observation):
            return "<empty>"
        return ""
    if value in _page_text(observation):
        return value[:80]
    return ""


def _r08_empty_entity_created(observation: Mapping[str, Any]) -> bool:
    state = observation.get("page_state", {})
    runtime = observation.get("runtime_signals", {})
    return bool(
        (isinstance(state, Mapping) and (state.get("created_entity") or state.get("empty_post_created")))
        or (isinstance(runtime, Mapping) and (runtime.get("created_entity") or runtime.get("empty_post_created")))
    )


def _is_r08_validation_failure_for_field(item: Mapping[str, Any]) -> bool:
    return bool(item.get("constraint_type"))


def _r08_anomaly_type(item: Mapping[str, Any]) -> tuple[str, str]:
    candidate = item.get("candidate", {})
    if isinstance(candidate, Mapping) and _r08_is_password(candidate):
        return "weak-password-validation", "weak-password-validation"
    if isinstance(candidate, Mapping) and _r08_is_forum_post(candidate) and str(item.get("value") or "").strip() == "":
        return "forum-empty-post-validation-missing", "empty-required-submission"
    return "validation-bypass", f"{item.get('constraint_type')}-constraint-bypass"


def _r08_evidence(
    item: Mapping[str, Any],
    issue_type: str,
    action_type: str,
    validation_feedback: bool,
    success_evidence: List[str],
) -> Dict[str, Any]:
    candidate = item.get("candidate", {})
    candidate = candidate if isinstance(candidate, Mapping) else {}
    constraints = item.get("constraints", {})
    constraints = constraints if isinstance(constraints, Mapping) else {}
    password = _r08_is_password(candidate)
    evidence: Dict[str, Any] = {
        "rule_id": "R08",
        "issue_type": issue_type,
        "target_bid": candidate.get("bid"),
        "target_role": candidate.get("role"),
        "target_tag": candidate.get("tag"),
        "input_type": constraints.get("input_type") or candidate.get("input_type") or candidate.get("type"),
        "required": constraints.get("required"),
        "minlength": constraints.get("minlength"),
        "maxlength": constraints.get("maxlength"),
        "min": constraints.get("minimum"),
        "max": constraints.get("maximum"),
        "pattern": constraints.get("pattern"),
        "constraint_type": item.get("constraint_type"),
        "constraint_violated": _is_r08_validation_failure_for_field(item),
        "submit_action": action_type,
        "validation_feedback_detected": validation_feedback,
        "success_evidence": success_evidence,
        "persisted_value": item.get("persisted_value") if not password else "",
    }
    if password:
        evidence["value_length"] = len(str(item.get("value") or ""))
    else:
        evidence["value"] = str(item.get("value") or "")[:120]
    if item.get("expected") is not None:
        evidence["expected"] = item.get("expected")
    if item.get("actual") is not None and not password:
        evidence["actual"] = item.get("actual")
    elif item.get("actual") is not None:
        evidence["actual_length"] = item.get("actual")
    return evidence


def _r08_is_password(candidate: Mapping[str, Any]) -> bool:
    text = " ".join(str(candidate.get(key) or "") for key in ("type", "input_type", "name", "label", "id", "autocomplete")).lower()
    return "password" in text


def _r08_is_forum_post(candidate: Mapping[str, Any]) -> bool:
    text = " ".join(str(candidate.get(key) or "") for key in ("role", "tag", "name", "label", "id", "class_name", "placeholder")).lower()
    return bool("textarea" in text or any(token in text for token in ("post", "comment", "forum", "content", "body", "message")))


def _r08_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "required"}
    return bool(value)


def _r08_first_present(candidate: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in candidate and candidate.get(key) is not None and candidate.get(key) != "":
            return candidate.get(key)
    return None


def _r08_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _r08_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _r08_number(value: str) -> float | None:
    return _r08_float(value.strip())


def _r08_pattern_matches(value: str, pattern: str) -> bool:
    try:
        return bool(re.fullmatch(pattern, value))
    except re.error:
        return True


def _r08_valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def _r08_valid_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return bool(parsed.scheme in {"http", "https"} and parsed.netloc)


def _detect_r09_result_issue(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if last_action_error or _has_api_failure(action_info):
        return None
    success = _r09_success_evidence(before_observation, after_observation, action_info)
    if not success:
        return None
    for detector in (
        _is_r09_filter_mismatch,
        _is_r09_search_mismatch,
        _is_r09_sort_mismatch,
        _is_r09_pagination_mismatch,
    ):
        anomaly = detector(before_observation, after_observation, action_info, action_type, clicked_candidate, success)
        if anomaly:
            return anomaly
    return None


def _is_r09_filter_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    success: List[str],
) -> Dict[str, Any] | None:
    condition = _r09_requested_condition(action_info, after_observation, clicked_candidate, operation="filter")
    if not condition:
        return None
    key, expected = condition
    entities = _r09_result_entities(after_observation)
    if not entities:
        return None
    violating = [entity for entity in entities if not _r09_entity_matches(entity, key, expected)]
    if not violating:
        return None
    return _r09_anomaly(
        "filter-result-mismatch",
        action_type,
        clicked_candidate,
        before_observation,
        after_observation,
        success,
        requested_operation="filter",
        requested_key=key,
        requested_value=expected,
        result_entities=entities,
        violating_entities=violating,
    )


def _is_r09_search_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    success: List[str],
) -> Dict[str, Any] | None:
    condition = _r09_requested_condition(action_info, after_observation, clicked_candidate, operation="search")
    if not condition:
        return None
    key, expected = condition
    entities = _r09_result_entities(after_observation)
    if not entities:
        return None
    violating = [entity for entity in entities if not _r09_entity_contains(entity, key, expected)]
    if not violating:
        return None
    return _r09_anomaly(
        "search-result-mismatch",
        action_type,
        clicked_candidate,
        before_observation,
        after_observation,
        success,
        requested_operation="search",
        requested_key=key,
        requested_value=expected,
        result_entities=entities,
        violating_entities=violating,
    )


def _is_r09_sort_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    success: List[str],
) -> Dict[str, Any] | None:
    sort_key = _r09_first_text(action_info, after_observation, ("sort_key", "requested_sort_key"))
    direction = _r09_first_text(action_info, after_observation, ("sort_direction", "requested_sort_direction", "direction")).lower()
    if not sort_key or direction not in {"asc", "ascending", "desc", "descending"}:
        return None
    entities = _r09_result_entities(after_observation)
    values = [_r09_sort_value(entity.get(sort_key)) for entity in entities if isinstance(entity, Mapping) and entity.get(sort_key) is not None]
    if len(values) < 2:
        return None
    descending = direction in {"desc", "descending"}
    ordered = all(values[index] >= values[index + 1] if descending else values[index] <= values[index + 1] for index in range(len(values) - 1))
    if ordered:
        return None
    return _r09_anomaly(
        "sort-order-mismatch",
        action_type,
        clicked_candidate,
        before_observation,
        after_observation,
        success,
        requested_operation="sort",
        requested_key=sort_key,
        requested_value=direction,
        result_entities=entities,
        violating_entities=[],
        sort_key=sort_key,
        sort_direction=direction,
    )


def _is_r09_pagination_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    success: List[str],
) -> Dict[str, Any] | None:
    before_page = _r09_page_value(before_observation, "current_page")
    actual_page = _r09_page_value(after_observation, "current_page")
    total_pages = _r09_page_value(before_observation, "total_pages") or _r09_page_value(after_observation, "total_pages")
    requested_page = _r09_requested_page(action_info, clicked_candidate, before_page)
    if before_page is None or actual_page is None or requested_page is None:
        return None
    if total_pages is not None and before_page >= total_pages and requested_page > before_page:
        return None
    before_ids = _r09_result_entity_ids(_r09_result_entities(before_observation))
    after_ids = _r09_result_entity_ids(_r09_result_entities(after_observation))
    mismatch = bool(actual_page != requested_page or (requested_page != before_page and before_ids and before_ids == after_ids))
    if not mismatch:
        return None
    return _r09_anomaly(
        "pagination-mismatch",
        action_type,
        clicked_candidate,
        before_observation,
        after_observation,
        success,
        requested_operation="pagination",
        requested_key="page",
        requested_value=requested_page,
        result_entities=_r09_result_entities(after_observation),
        violating_entities=[],
        requested_page=requested_page,
        actual_page=actual_page,
    )


def _r09_success_evidence(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[str]:
    evidence: List[str] = []
    if _r08_action_success(action_info):
        evidence.append("action_success")
    if _r09_successful_request(action_info):
        evidence.append("request_success")
    if _r09_selected_state_changed(before_observation, after_observation, action_info):
        evidence.append("selected_state_change")
    if _r09_results_refreshed(before_observation, after_observation):
        evidence.append("result_refresh")
    if str(after_observation.get("runtime_signals", {}).get("loading") or "").lower() in {"false", "complete", "done"}:
        evidence.append("loading_complete")
    return sorted(set(evidence))


def _r09_successful_request(action_info: Mapping[str, Any]) -> bool:
    for key in ("status_code", "response_status", "api_status", "network_status"):
        status = _r08_int(action_info.get(key))
        if status is not None and 200 <= status < 300:
            return True
    entries = action_info.get("network_entries", []) or action_info.get("network", []) or []
    if isinstance(entries, list):
        return any(
            isinstance(entry, Mapping)
            and _r08_int(entry.get("status") or entry.get("status_code")) is not None
            and 200 <= int(_r08_int(entry.get("status") or entry.get("status_code")) or 0) < 300
            for entry in entries
        )
    return False


def _r09_selected_state_changed(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> bool:
    del action_info
    keys = ("selected_filter", "filter_key", "filter_value", "sort_key", "sort_direction", "current_page")
    before_state = before_observation.get("page_state", {}) if isinstance(before_observation.get("page_state"), Mapping) else {}
    after_state = after_observation.get("page_state", {}) if isinstance(after_observation.get("page_state"), Mapping) else {}
    if any(before_state.get(key) != after_state.get(key) for key in keys if key in before_state or key in after_state):
        return True
    return False


def _r09_results_refreshed(before_observation: Mapping[str, Any], after_observation: Mapping[str, Any]) -> bool:
    before_entities = _r09_result_entities(before_observation)
    after_entities = _r09_result_entities(after_observation)
    if not after_entities:
        return False
    return _r09_result_entity_ids(before_entities) != _r09_result_entity_ids(after_entities) or len(before_entities) != len(after_entities)


def _r09_requested_condition(
    action_info: Mapping[str, Any],
    observation: Mapping[str, Any],
    clicked_candidate: Mapping[str, Any] | None,
    *,
    operation: str,
) -> tuple[str, Any] | None:
    if operation == "filter":
        key_names = ("filter_key", "requested_key", "selected_filter_key")
        value_names = ("filter_value", "requested_value", "selected_filter_value")
    else:
        key_names = ("search_key", "query_key", "requested_key")
        value_names = ("search_value", "query_value", "query", "requested_value")
    key = _r09_first_text(action_info, observation, key_names)
    value = _r09_first_value(action_info, observation, value_names)
    if key and value not in (None, ""):
        return key, value
    if clicked_candidate:
        key = _r09_first_text(clicked_candidate, {}, key_names)
        value = _r09_first_value(clicked_candidate, {}, value_names)
        if key and value not in (None, ""):
            return key, value
    return None


def _r09_result_entities(observation: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for container in (observation, observation.get("page_state", {}), observation.get("runtime_signals", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("result_entities", "results", "items", "entities"):
            values = container.get(key)
            if isinstance(values, list):
                return [item for item in values if isinstance(item, Mapping)]
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return []
    result_candidates = []
    entity_keys = {"entity_id", "product_id", "post_id", "item_id", "order_id", "user_id", "status", "category", "price", "role", "type"}
    for candidate in candidates:
        if isinstance(candidate, Mapping) and (candidate.get("is_result") or entity_keys.intersection(candidate.keys())):
            result_candidates.append(candidate)
    return result_candidates


def _r09_entity_matches(entity: Mapping[str, Any], key: str, expected: Any) -> bool:
    actual = entity.get(key)
    if isinstance(actual, list):
        return any(_r09_values_equal(item, expected) for item in actual)
    return _r09_values_equal(actual, expected)


def _r09_entity_contains(entity: Mapping[str, Any], key: str, expected: Any) -> bool:
    actual = entity.get(key)
    if actual is None:
        return False
    return str(expected).strip().lower() in str(actual).strip().lower()


def _r09_values_equal(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    return str(actual).strip().lower() == str(expected).strip().lower()


def _r09_sort_value(value: Any) -> Any:
    number = _r08_float(value)
    if number is not None:
        return number
    text = str(value or "").strip()
    date_match = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if date_match:
        return tuple(int(part) for part in date_match.groups())
    return text.lower()


def _r09_requested_page(action_info: Mapping[str, Any], clicked_candidate: Mapping[str, Any] | None, before_page: int | None) -> int | None:
    explicit = _r08_int(action_info.get("requested_page") or action_info.get("target_page"))
    if explicit is not None:
        return explicit
    operation = str(action_info.get("requested_operation") or action_info.get("pagination_action") or "").lower()
    if clicked_candidate and not operation:
        operation = _candidate_text(clicked_candidate).lower()
    if before_page is None:
        return None
    if "next" in operation:
        return before_page + 1
    if "prev" in operation or "previous" in operation:
        return max(1, before_page - 1)
    return None


def _r09_page_value(observation: Mapping[str, Any], key: str) -> int | None:
    for container in (observation, observation.get("page_state", {}), observation.get("runtime_signals", {})):
        if isinstance(container, Mapping):
            value = _r08_int(container.get(key))
            if value is not None:
                return value
    return None


def _r09_result_entity_ids(entities: List[Mapping[str, Any]]) -> List[str]:
    ids = []
    for entity in entities:
        value = ""
        for key in ("entity_id", "product_id", "post_id", "item_id", "order_id", "user_id", "id"):
            if entity.get(key) is not None and str(entity.get(key)).strip():
                value = str(entity.get(key)).strip()
                break
        if value:
            ids.append(value)
    return ids


def _r09_first_text(primary: Mapping[str, Any], observation: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    value = _r09_first_value(primary, observation, keys)
    return str(value).strip() if value not in (None, "") else ""


def _r09_first_value(primary: Mapping[str, Any], observation: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for container in (primary, observation.get("page_state", {}) if isinstance(observation, Mapping) else {}, observation):
        if isinstance(container, Mapping):
            for key in keys:
                if key in container and container.get(key) not in (None, ""):
                    return container.get(key)
    return None


def _r09_anomaly(
    issue_type: str,
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    success: List[str],
    *,
    requested_operation: str,
    requested_key: str,
    requested_value: Any,
    result_entities: List[Mapping[str, Any]],
    violating_entities: List[Mapping[str, Any]],
    sort_key: str = "",
    sort_direction: str = "",
    requested_page: int | None = None,
    actual_page: int | None = None,
) -> Dict[str, Any]:
    return {
        "type": issue_type,
        "confidence": 0.84,
        "evidence": {
            "rule_id": "R09",
            "issue_type": issue_type,
            "action_type": action_type,
            "target_bid": clicked_candidate.get("bid") if clicked_candidate else "",
            "requested_operation": requested_operation,
            "requested_key": requested_key,
            "requested_value": requested_value,
            "result_count_before": len(_r09_result_entities(before_observation)),
            "result_count_after": len(result_entities),
            "result_entity_ids": _r09_result_entity_ids(result_entities)[:20],
            "violating_entity_ids": _r09_result_entity_ids(violating_entities)[:20],
            "sort_key": sort_key,
            "sort_direction": sort_direction,
            "requested_page": requested_page,
            "actual_page": actual_page,
            "action_success": "action_success" in success,
            "request_success": "request_success" in success,
            "success_evidence": success,
        },
    }


def _detect_r13_api_error(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    failures = _r13_extract_api_failures(observation, action_info)
    if not failures:
        return None
    failure = failures[0]
    return {
        "type": "api-server-error",
        "confidence": 0.9,
        "evidence": {
            "rule_id": "R13",
            "issue_type": "server-error",
            "method": failure.get("method"),
            "request_path": failure.get("request_path"),
            "status_code": failure.get("status_code"),
            "resource_type": failure.get("resource_type"),
            "content_type": failure.get("content_type"),
            "request_succeeded": False,
            "response_received": True,
            "network_failure": False,
            "timeout": False,
            "action_type": action_type,
            "target_bid": clicked_candidate.get("bid") if clicked_candidate else "",
            "response_error_summary": failure.get("response_error_summary", ""),
        },
    }


def _is_r13_api_error(entry: Mapping[str, Any]) -> bool:
    status = _r13_status(entry)
    return bool(
        status is not None
        and _r13_is_server_error_status(status)
        and _r13_response_received(entry)
        and _r13_is_application_api_request(entry)
    )


def _r13_extract_api_failures(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    entries = _r13_network_entries(observation, action_info)
    failures: Dict[tuple[str, str, int, str], Dict[str, Any]] = {}
    for entry in entries:
        if not _is_r13_api_error(entry):
            continue
        status = int(_r13_status(entry) or 0)
        method = str(entry.get("method") or "GET").upper()
        path = _r13_request_path(entry.get("url") or entry.get("request_url") or entry.get("path") or "")
        request_id = str(entry.get("request_id") or entry.get("trace_id") or "")
        key = (method, path, status, request_id)
        failures.setdefault(
            key,
            {
                "method": method,
                "request_path": path,
                "status_code": status,
                "resource_type": str(entry.get("resource_type") or entry.get("type") or ""),
                "content_type": str(entry.get("content_type") or entry.get("response_content_type") or ""),
                "response_error_summary": _r13_response_summary(entry),
            },
        )
    return list(failures.values())


def _r13_network_entries(observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    entries: List[Mapping[str, Any]] = []
    for container in (action_info, observation.get("runtime_signals", {}), observation.get("page_state", {}), observation):
        if not isinstance(container, Mapping):
            continue
        for key in ("network_entries", "network_events", "api_requests", "requests"):
            value = container.get(key)
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, Mapping))
        if any(key in container for key in ("status", "status_code", "response_status", "network_status")):
            entries.append(container)
    return entries


def _r13_is_application_api_request(entry: Mapping[str, Any]) -> bool:
    semantic_class = str(entry.get("semantic_class") or "").upper()
    if semantic_class in {
        "STATIC_ASSET",
        "PREFETCH",
        "FRAMEWORK_NAVIGATION",
        "BACKGROUND_ACTIVITY",
        "DEVELOPMENT_INFRASTRUCTURE",
    }:
        return False
    if semantic_class == "USER_FUNCTIONAL_REQUEST":
        return True
    resource_type = str(entry.get("resource_type") or entry.get("type") or "").lower()
    content_type = str(entry.get("content_type") or entry.get("response_content_type") or "").lower()
    url = str(entry.get("url") or entry.get("request_url") or entry.get("path") or "")
    path = urlparse(url).path.lower() if url else ""
    if resource_type in {"image", "font", "stylesheet", "css", "script", "media"}:
        return False
    if path.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js", ".woff", ".woff2", ".ttf")):
        return False
    if resource_type in {"xhr", "fetch", "document"}:
        return True
    if "/api/" in path or path.startswith("/api/"):
        return True
    if "json" in content_type:
        return True
    return bool(entry.get("is_api") or entry.get("application_api"))


def _r13_is_server_error_status(status_code: int) -> bool:
    return 500 <= int(status_code) < 600


def _r13_contract_failure(entry: Mapping[str, Any]) -> bool:
    del entry
    return False


def _r13_status(entry: Mapping[str, Any]) -> int | None:
    for key in ("status", "status_code", "response_status", "network_status"):
        status = _r08_int(entry.get(key))
        if status is not None:
            return status
    return None


def _r13_response_received(entry: Mapping[str, Any]) -> bool:
    if "response_received" in entry:
        return bool(entry.get("response_received"))
    return _r13_status(entry) is not None


def _r13_entry_network_failure(entry: Mapping[str, Any]) -> bool:
    if bool(entry.get("network_failure")):
        return True
    text = " ".join(str(entry.get(key) or "") for key in ("network_error", "error", "failure_text")).lower()
    return any(token in text for token in ("connection_refused", "connection refused", "connection_reset", "connection reset", "dns", "failed to fetch", "socket"))


def _r13_entry_timeout(entry: Mapping[str, Any]) -> bool:
    if bool(entry.get("timeout") or entry.get("timeout_occurred")):
        return True
    text = " ".join(str(entry.get(key) or "") for key in ("error", "network_error", "failure_text")).lower()
    return "timeout" in text or "timed out" in text


def _r13_request_path(url: Any) -> str:
    text = str(url or "")
    if not text:
        return ""
    parsed = urlparse(text)
    path = parsed.path or text.split("?", 1)[0]
    return path or "/"


def _r13_response_summary(entry: Mapping[str, Any]) -> str:
    for key in ("error_summary", "status_text", "response_status_text", "message"):
        value = str(entry.get(key) or "").strip()
        if value:
            return _r13_redact(value)[:160]
    return ""


def _r13_redact(value: str) -> str:
    return re.sub(r"(?i)(token|api_key|key|secret|password|session)=([^&\\s]+)", r"\1=<redacted>", value)


def _r13_auth_error_expected(action_info: Mapping[str, Any]) -> bool:
    return bool(
        action_info.get("auth_expected")
        or action_info.get("permission_expected")
        or action_info.get("expected_authorized")
        or action_info.get("should_have_access")
    )


def _detect_r14_network_failure(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    failures = _r14_extract_network_failures(observation, action_info)
    if not failures:
        return None
    failure = failures[0]
    return {
        "type": "network-failure",
        "confidence": 0.88,
        "evidence": {
            "rule_id": "R14",
            "issue_type": failure.get("failure_type"),
            "method": failure.get("method"),
            "request_path": failure.get("request_path"),
            "resource_type": failure.get("resource_type"),
            "response_received": False,
            "status_code": None,
            "failure_type": failure.get("failure_type"),
            "failure_summary": failure.get("failure_summary"),
            "timeout": False,
            "cancelled": False,
            "action_type": action_type,
            "target_bid": clicked_candidate.get("bid") if clicked_candidate else "",
        },
    }


def _is_r14_network_failure(entry: Mapping[str, Any]) -> bool:
    return bool(
        _r13_status(entry) is None
        and not _r13_response_received(entry)
        and _r14_transport_failure_type(entry)
        and not _r13_entry_timeout(entry)
        and not _r14_is_timeout_failure(entry)
        and not _r14_is_expected_cancellation(entry)
        and _r13_is_application_api_request(entry)
    )


def _r14_extract_network_failures(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    failures: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for entry in _r13_network_entries(observation, action_info):
        if not _is_r14_network_failure(entry):
            continue
        method = str(entry.get("method") or "GET").upper()
        path = _r13_request_path(entry.get("url") or entry.get("request_url") or entry.get("path") or "")
        failure_type = _r14_transport_failure_type(entry)
        request_id = str(entry.get("request_id") or entry.get("trace_id") or "")
        key = _r14_request_identity(method, path, failure_type, request_id)
        failures.setdefault(
            key,
            {
                "method": method,
                "request_path": path,
                "resource_type": str(entry.get("resource_type") or entry.get("type") or ""),
                "failure_type": failure_type,
                "failure_summary": _r14_failure_summary(entry),
            },
        )
    return list(failures.values())


def _r14_transport_failure_type(entry: Mapping[str, Any]) -> str:
    text = _r14_failure_text(entry)
    if not text:
        return ""
    if "err_timed_out" in text or "timeout" in text or "timed out" in text:
        return ""
    if "err_aborted" in text or "cancelled" in text or "canceled" in text:
        return ""
    if "connection_refused" in text or "connection refused" in text:
        return "connection-refused"
    if "connection_reset" in text or "connection reset" in text:
        return "connection-reset"
    if "connection_aborted" in text or "connection aborted" in text:
        return "connection-aborted"
    if "name_not_resolved" in text or "dns" in text:
        return "dns-failure"
    if "socket" in text:
        return "socket-failure"
    if "failed to fetch" in text or "fetch failed" in text or "networkerror" in text or "network error" in text:
        return "fetch-failure"
    if "connection closed" in text or "closed before response" in text:
        return "connection-closed"
    if "network changed" in text or "err_network_changed" in text:
        return "network-changed"
    return "other-network-failure" if any(token in text for token in ("net::err_", "network", "connection")) else ""


def _r14_failure_text(entry: Mapping[str, Any]) -> str:
    return " ".join(str(entry.get(key) or "") for key in ("failure_text", "failure", "error_text", "request_error", "network_error", "error")).lower()


def _r14_is_timeout_failure(entry: Mapping[str, Any]) -> bool:
    if bool(entry.get("timeout") or entry.get("timeout_occurred")):
        return True
    text = _r14_failure_text(entry)
    return "err_timed_out" in text or "timeout" in text or "timed out" in text


def _r14_is_expected_cancellation(entry: Mapping[str, Any]) -> bool:
    text = _r14_failure_text(entry)
    if "err_aborted" not in text and "cancelled" not in text and "canceled" not in text:
        return False
    return True


def _r14_request_identity(method: str, path: str, failure_type: str, request_id: str) -> tuple[str, str, str, str]:
    return (request_id or "", method, path, failure_type)


def _r14_failure_summary(entry: Mapping[str, Any]) -> str:
    text = _r14_failure_text(entry)
    match = re.search(r"(err_[a-z_]+)", text)
    if match:
        return match.group(1).upper()
    failure_type = _r14_transport_failure_type(entry)
    return failure_type


def _detect_r15_timeout(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    clicked_candidate: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    events = _r15_extract_timeout_events(observation, action_info)
    if not events:
        return None
    event = events[0]
    issue_type = str(event.get("timeout_type") or "timeout")
    return {
        "type": "async-hang" if issue_type == "async-timeout" else "timeout-error",
        "confidence": 0.87,
        "evidence": {
            "rule_id": "R15",
            "issue_type": issue_type,
            "operation_type": event.get("operation_type"),
            "method": event.get("method"),
            "request_path": event.get("request_path"),
            "resource_type": event.get("resource_type"),
            "response_received": False,
            "status_code": None,
            "timeout": True,
            "timeout_type": issue_type,
            "timeout_summary": event.get("timeout_summary"),
            "configured_timeout_ms": event.get("configured_timeout_ms"),
            "elapsed_ms": event.get("elapsed_ms"),
            "operation_completed": False,
            "action_type": action_type,
            "target_bid": clicked_candidate.get("bid") if clicked_candidate else "",
        },
    }


def _is_r15_timeout(entry: Mapping[str, Any]) -> bool:
    if _r13_status(entry) is not None and _r13_response_received(entry):
        return False
    if _r15_operation_completed(entry):
        return False
    if _r14_is_expected_cancellation(entry):
        return False
    if _r15_is_static_resource(entry):
        return False
    timeout_type = _r15_timeout_type(entry)
    threshold_exceeded = _r15_threshold_exceeded(entry)
    performance_budget_exceeded = _r15_performance_budget_exceeded(entry)
    if not (timeout_type or threshold_exceeded or performance_budget_exceeded):
        return False
    if _r15_is_locator_action_timeout(entry) and not _r15_application_operation_started(entry):
        return False
    return _r15_operation_started(entry) or threshold_exceeded or performance_budget_exceeded


def _r15_extract_timeout_events(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    events: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for entry in _r15_timeout_entries(observation, action_info):
        if not _is_r15_timeout(entry):
            continue
        timeout_type = _r15_timeout_type(entry) or "request-timeout"
        method = str(entry.get("method") or "GET").upper() if _r15_entry_url(entry) else ""
        path = _r13_request_path(_r15_entry_url(entry))
        request_id = str(entry.get("request_id") or entry.get("trace_id") or entry.get("operation_id") or "")
        key = _r15_request_identity(_r15_operation_type(entry, timeout_type), method, path, timeout_type, request_id)
        events.setdefault(
            key,
            {
                "timeout_type": timeout_type,
                "operation_type": _r15_operation_type(entry, timeout_type),
                "method": method,
                "request_path": path,
                "resource_type": str(entry.get("resource_type") or entry.get("type") or ""),
                "timeout_summary": _r15_timeout_summary(entry, timeout_type),
                "configured_timeout_ms": _r15_configured_timeout_ms(entry),
                "elapsed_ms": _r15_elapsed_ms(entry),
            },
        )
    return list(events.values())


def _r15_timeout_entries(observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    entries: List[Mapping[str, Any]] = []
    entries.extend(_r13_network_entries(observation, action_info))
    action_result = action_info.get("action_result")
    for container in (
        action_info,
        action_result if isinstance(action_result, Mapping) else {},
        observation.get("runtime_signals", {}),
        observation.get("page_state", {}),
    ):
        if isinstance(container, Mapping) and _r15_container_has_timeout_signal(container):
            entries.append(container)
    return entries


def _r15_container_has_timeout_signal(container: Mapping[str, Any]) -> bool:
    return bool(
        _r15_timeout_type(container)
        or _r15_threshold_exceeded(container)
        or _r15_performance_budget_exceeded(container)
    )


def _r15_timeout_type(entry: Mapping[str, Any]) -> str:
    if bool(entry.get("navigation_timeout")):
        return "navigation-timeout"
    if bool(entry.get("request_timeout") or entry.get("response_timeout")):
        return "request-timeout"
    if bool(entry.get("async_timeout")):
        return "async-timeout"
    if bool(entry.get("action_timeout")):
        return "action-timeout"
    if _r15_performance_budget_exceeded(entry):
        return "performance-budget-exceeded"
    text = _r15_timeout_text(entry)
    if bool(entry.get("timeout") or entry.get("timeout_occurred") or entry.get("timed_out")):
        if bool(entry.get("navigation_started")):
            return "navigation-timeout"
        if bool(entry.get("loading") or entry.get("pending") or entry.get("request_pending") or entry.get("async_started")):
            return "async-timeout"
        if _r15_entry_url(entry) or entry.get("request_started") or entry.get("response_wait_started"):
            return "request-timeout"
        return "timeout"
    if "err_timed_out" in text:
        return "request-timeout"
    if any(token in text for token in ("timeouterror", "timeout exceeded", "timed out", "timeout 30000ms exceeded", "waiting for")):
        if "navigation" in text:
            return "navigation-timeout"
        if "request" in text or "response" in text or _r15_entry_url(entry):
            return "request-timeout"
        if bool(entry.get("loading") or entry.get("pending") or entry.get("async_started")):
            return "async-timeout"
        return "timeout"
    if _r15_threshold_exceeded(entry):
        return "request-timeout" if _r15_entry_url(entry) else "timeout"
    return ""


def _r15_timeout_text(entry: Mapping[str, Any]) -> str:
    return " ".join(
        str(entry.get(key) or "")
        for key in ("timeout_message", "failure_text", "failure", "error_text", "request_error", "network_error", "error", "message")
    ).lower()


def _r15_configured_timeout_ms(entry: Mapping[str, Any]) -> float | None:
    for key in (
        "configured_timeout_ms",
        "timeout_ms",
        "request_timeout_ms",
        "navigation_timeout_ms",
        "action_timeout_ms",
        "response_timeout_ms",
        "max_latency_ms",
        "performance_budget_ms",
        "sla_ms",
    ):
        value = _safe_float(entry.get(key))
        if value is not None:
            return value
    for key in ("timeout_seconds", "configured_timeout_seconds", "request_timeout_seconds", "navigation_timeout_seconds"):
        value = _safe_float(entry.get(key))
        if value is not None:
            return value * 1000.0
    return None


def _r15_elapsed_ms(entry: Mapping[str, Any]) -> float | None:
    for key in ("elapsed_ms", "duration_ms", "latency_ms", "response_time_ms", "actual_latency_ms"):
        value = _safe_float(entry.get(key))
        if value is not None:
            return value
    for key in ("elapsed_seconds", "duration_seconds", "latency_seconds"):
        value = _safe_float(entry.get(key))
        if value is not None:
            return value * 1000.0
    return None


def _r15_threshold_exceeded(entry: Mapping[str, Any]) -> bool:
    configured = _r15_configured_timeout_ms(entry)
    elapsed = _r15_elapsed_ms(entry)
    return bool(configured is not None and elapsed is not None and elapsed > configured and not _r15_operation_completed(entry))


def _r15_performance_budget_exceeded(entry: Mapping[str, Any]) -> bool:
    budget = _safe_float(entry.get("max_latency_ms") or entry.get("performance_budget_ms") or entry.get("sla_ms"))
    elapsed = _safe_float(entry.get("actual_latency_ms") or entry.get("latency_ms") or entry.get("duration_ms"))
    return bool(budget is not None and elapsed is not None and elapsed > budget)


def _r15_operation_completed(entry: Mapping[str, Any]) -> bool:
    if _r13_response_received(entry) and _r13_status(entry) is not None:
        return True
    for key in ("completed", "operation_completed", "request_completed", "navigation_complete", "success", "operation_done"):
        if bool(entry.get(key)):
            return True
    if str(entry.get("loading") or "").lower() in {"false", "complete", "done"}:
        return True
    if str(entry.get("pending") or "").lower() == "false":
        return True
    return False


def _r15_operation_started(entry: Mapping[str, Any]) -> bool:
    return bool(
        _r15_application_operation_started(entry)
        or _r15_entry_url(entry)
        or entry.get("request_started")
        or entry.get("navigation_started")
        or entry.get("async_started")
        or entry.get("operation_started")
        or entry.get("response_wait_started")
        or entry.get("operation_type")
    )


def _r15_application_operation_started(entry: Mapping[str, Any]) -> bool:
    return bool(
        _r15_entry_url(entry)
        or entry.get("request_started")
        or entry.get("navigation_started")
        or entry.get("async_started")
        or entry.get("response_wait_started")
    )


def _r15_is_locator_action_timeout(entry: Mapping[str, Any]) -> bool:
    text = _r15_timeout_text(entry)
    return bool(
        ("locator" in text or "selector" in text or "element" in text)
        and any(token in text for token in ("not found", "waiting for selector", "waiting for locator", "click"))
    )


def _r15_is_static_resource(entry: Mapping[str, Any]) -> bool:
    return not _r13_is_application_api_request(entry) and bool(_r15_entry_url(entry))


def _r15_entry_url(entry: Mapping[str, Any]) -> Any:
    return entry.get("url") or entry.get("request_url") or entry.get("path") or ""


def _r15_operation_type(entry: Mapping[str, Any], timeout_type: str) -> str:
    explicit = str(entry.get("operation_type") or "").strip()
    if explicit:
        return explicit
    if timeout_type == "navigation-timeout":
        return "navigation"
    if timeout_type == "async-timeout":
        return "async"
    if timeout_type == "performance-budget-exceeded":
        return "performance"
    if _r15_entry_url(entry):
        return "request"
    return "operation"


def _r15_request_identity(operation_type: str, method: str, path: str, timeout_type: str, request_id: str) -> tuple[str, str, str, str, str]:
    return (request_id or "", operation_type, method, path, timeout_type)


def _r15_timeout_summary(entry: Mapping[str, Any], timeout_type: str) -> str:
    text = _r15_timeout_text(entry)
    if "err_timed_out" in text:
        return "ERR_TIMED_OUT"
    if "timeouterror" in text:
        return "TimeoutError"
    if "timeout exceeded" in text:
        return "timeout exceeded"
    if "timed out" in text:
        return "timed out"
    return timeout_type


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_r10_ui_api_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if last_action_error or _has_api_failure(action_info) or _r10_network_failure(action_info) or not _r10_ui_settled(after_observation):
        return None
    semantic_mismatch = _detect_r10_network_ui_semantic_mismatch(before_observation, after_observation, action_info, action_type)
    if semantic_mismatch:
        return semantic_mismatch
    api_status = _r10_api_status(action_info)
    if api_status is None or not (200 <= api_status < 300):
        return None
    api_entities = _r10_extract_api_entities(action_info, after_observation)
    ui_entities = _r10_extract_ui_entities(after_observation)
    if not api_entities or not ui_entities:
        return None
    for api_entity in api_entities:
        api_identity = _r10_entity_identity(api_entity)
        if not api_identity:
            continue
        ui_entity = next((item for item in ui_entities if _r10_entity_identity(item) == api_identity), None)
        if not ui_entity:
            continue
        mismatches = _r10_compare_entity_fields(api_entity, ui_entity)
        if not mismatches:
            continue
        issue_type = _r10_issue_type(api_entity, ui_entity, mismatches, before_observation)
        return {
            "type": issue_type,
            "confidence": 0.86,
            "evidence": {
                "rule_id": "R10",
                "issue_type": issue_type,
                "entity_type": api_identity[0],
                "entity_id": api_identity[1],
                "field": mismatches[0]["field"],
                "api_value": mismatches[0]["api_value"],
                "ui_value": mismatches[0]["ui_value"],
                "normalized_api_value": mismatches[0]["normalized_api_value"],
                "normalized_ui_value": mismatches[0]["normalized_ui_value"],
                "mismatched_fields": mismatches,
                "api_status": api_status,
                "api_request_succeeded": True,
                "ui_settled": True,
                "identity_source": api_identity[2],
                "api_source": str(api_entity.get("_source") or "api"),
                "ui_source": str(ui_entity.get("_source") or "ui"),
                "action_type": action_type,
            },
        }
    return None


def _detect_r10_network_ui_semantic_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
) -> Dict[str, Any] | None:
    semantic = _r10_network_ui_semantic(after_observation, action_info)
    if not semantic:
        return None
    for pair in _r10_semantic_pairs(semantic):
        if not _r10_high_confidence_not_equal_pair(pair):
            continue
        network = pair.get("network_semantic_value") if isinstance(pair.get("network_semantic_value"), Mapping) else {}
        ui = pair.get("ui_semantic_value") if isinstance(pair.get("ui_semantic_value"), Mapping) else {}
        request_path = _normalize_signature_path(
            pair.get("request_path")
            or pair.get("normalized_url")
            or pair.get("url")
            or pair.get("request_url")
            or ""
        )
        field = str(pair.get("network_key") or pair.get("ui_key") or "")
        issue_type = "api-ui-mismatch"
        action = action_info.get("action", action_info)
        candidate_index = 0
        if isinstance(action, Mapping):
            try:
                candidate_index = int(action.get("candidate_index", 0) or 0)
            except (TypeError, ValueError):
                candidate_index = 0
        before_candidates = before_observation.get("candidate_elements", []) or []
        origin_candidate = _candidate_at(before_candidates, candidate_index)
        origin_bid = origin_candidate.get("bid") if origin_candidate else ""
        return {
            "type": issue_type,
            "confidence": 0.88,
            "target_id": origin_bid,
            "evidence": {
                "rule_id": "R10",
                "issue_type": issue_type,
                "source": "network_ui_semantic",
                "entity_type": "api_ui_semantic",
                "entity_id": request_path,
                "field": field,
                "api_value": pair.get("network_value"),
                "ui_value": pair.get("ui_value"),
                "normalized_api_value": network.get("normalized"),
                "normalized_ui_value": ui.get("normalized"),
                "mismatched_fields": [
                    {
                        "field": field,
                        "api_value": pair.get("network_value"),
                        "ui_value": pair.get("ui_value"),
                        "normalized_api_value": network.get("normalized"),
                        "normalized_ui_value": ui.get("normalized"),
                    }
                ],
                "action_id": semantic.get("action_id") or action_info.get("action_id"),
                "request_id": pair.get("request_id"),
                "request_url": pair.get("url") or pair.get("request_url"),
                "request_path": request_path,
                "request_method": pair.get("method") or pair.get("request_method") or "",
                "network_key": pair.get("network_key"),
                "ui_key": pair.get("ui_key"),
                "network_value": pair.get("network_value"),
                "ui_value": pair.get("ui_value"),
                "relation": "NOT_EQUAL",
                "pair_method": pair.get("pair_method"),
                "confidence": semantic_pair_confidence(pair),
                "pair_confidence": semantic_pair_confidence(pair),
                "match_basis": pair.get("match_basis"),
                "api_status": pair.get("status"),
                "api_request_succeeded": True,
                "ui_settled": True,
                "action_type": action_type,
                "candidate_index": candidate_index,
                "target_bid": origin_bid,
                "target_role": origin_candidate.get("role") if origin_candidate else "",
                "target_tag": origin_candidate.get("tag") if origin_candidate else "",
                "target_text": _candidate_text(origin_candidate) if origin_candidate else "",
                "reason": (
                    f"API/UI mismatch: network {pair.get('network_key')}={pair.get('network_value')} "
                    f"ui {pair.get('ui_key')}={pair.get('ui_value')} "
                    f"pair_method={pair.get('pair_method')}"
                ),
            },
        }
    return None


def _r10_network_ui_semantic(after_observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> Mapping[str, Any]:
    for container in (
        action_info,
        action_info.get("action_result", {}) if isinstance(action_info.get("action_result"), Mapping) else {},
        after_observation.get("runtime_signals", {}) if isinstance(after_observation.get("runtime_signals"), Mapping) else {},
        after_observation.get("page_state", {}) if isinstance(after_observation.get("page_state"), Mapping) else {},
    ):
        if isinstance(container, Mapping) and isinstance(container.get("network_ui_semantic"), Mapping):
            return container.get("network_ui_semantic")  # type: ignore[return-value]
    return {}


def _r10_semantic_pairs(semantic: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    settled = semantic.get("settled_pairs")
    if isinstance(settled, list) and settled:
        return [item for item in settled if isinstance(item, Mapping)]
    pairs = semantic.get("pairs")
    if isinstance(pairs, list):
        return [item for item in pairs if isinstance(item, Mapping)]
    return []


def _r10_high_confidence_not_equal_pair(pair: Mapping[str, Any]) -> bool:
    if str(pair.get("relation") or "") != "NOT_EQUAL":
        return False
    method = str(pair.get("pair_method") or "").upper()
    if method not in {"EXACT_KEY", "FORM_NAME_VALUE", "LABEL_VALUE", "TABLE_HEADER_CELL"}:
        return False
    confidence = semantic_pair_confidence(pair)
    if confidence is None or confidence < 0.85:
        return False
    if not pair.get("network_key") or not pair.get("ui_key"):
        return False
    if pair.get("network_value") in (None, "") or pair.get("ui_value") in (None, ""):
        return False
    if _r10_placeholder_semantic_value(pair.get("ui_value")) or _r10_placeholder_semantic_value(pair.get("network_value")):
        return False
    return True


def _r10_placeholder_semantic_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    return text in {"loading", "loading...", "n/a", "na", "-", "--", "pending..."} or "spinner" in text


def _is_r10_ui_api_mismatch(api_entity: Mapping[str, Any], ui_entity: Mapping[str, Any]) -> bool:
    return bool(_r10_entity_identity(api_entity) and _r10_entity_identity(api_entity) == _r10_entity_identity(ui_entity) and _r10_compare_entity_fields(api_entity, ui_entity))


def _r10_extract_api_entities(action_info: Mapping[str, Any], after_observation: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    entities: List[Mapping[str, Any]] = []
    for container in (action_info, action_info.get("action_result", {}), after_observation.get("runtime_signals", {}), after_observation.get("page_state", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("api_entities", "api_entity", "api_response", "response_body", "response_data", "data", "entity", "item", "cart_items"):
            entities.extend(_r10_entities_from_value(container.get(key), source=f"api:{key}"))
    network_entries = action_info.get("network_entries", []) or action_info.get("network", []) or []
    if isinstance(network_entries, list):
        for entry in network_entries:
            if not isinstance(entry, Mapping):
                continue
            status = _r08_int(entry.get("status") or entry.get("status_code"))
            if status is not None and not (200 <= status < 300):
                continue
            for key in ("response_body", "response_data", "body", "data", "json"):
                entities.extend(_r10_entities_from_value(entry.get(key), source=f"network:{key}"))
    return _r10_dedup_entities(entities)


def _r10_extract_ui_entities(observation: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    entities: List[Mapping[str, Any]] = []
    for container in (observation, observation.get("page_state", {}), observation.get("runtime_signals", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("ui_entities", "rendered_entities", "cart_items", "result_entities", "entities", "items"):
            entities.extend(_r10_entities_from_value(container.get(key), source=f"ui:{key}"))
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and _r10_entity_identity(candidate):
                enriched = dict(candidate)
                enriched["_source"] = "ui:candidate"
                entities.append(enriched)
    return _r10_dedup_entities(entities)


def _r10_entities_from_value(value: Any, *, source: str) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if _r10_entity_identity(value):
            item = dict(value)
            item["_source"] = source
            return [item]
        for key in ("items", "results", "entities", "data", "cart_items", "products"):
            nested = value.get(key)
            if isinstance(nested, list):
                results: List[Mapping[str, Any]] = []
                for child in nested:
                    results.extend(_r10_entities_from_value(child, source=f"{source}.{key}"))
                if results:
                    return results
        return []
    if isinstance(value, list):
        results = []
        for item in value:
            results.extend(_r10_entities_from_value(item, source=source))
        return results
    return []


def _r10_dedup_entities(entities: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    deduped: Dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for entity in entities:
        identity = _r10_entity_identity(entity)
        if identity and identity not in deduped:
            deduped[identity] = entity
    return list(deduped.values())


def _r10_entity_identity(entity: Mapping[str, Any]) -> tuple[str, str, str] | None:
    typed_keys = ("product_id", "cart_item_id", "item_id", "order_id", "user_id", "post_id", "comment_id", "entity_id")
    for key in typed_keys:
        value = entity.get(key)
        if value is not None and str(value).strip():
            entity_type = key[:-3] if key.endswith("_id") else key
            return (entity_type, str(value).strip(), key)
    value = entity.get("id")
    entity_type = str(entity.get("entity_type") or entity.get("type") or "").strip()
    if value is not None and str(value).strip() and entity_type:
        return (entity_type, str(value).strip(), "id")
    return None


def _r10_compare_entity_fields(api_entity: Mapping[str, Any], ui_entity: Mapping[str, Any]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    for field in _r10_comparable_fields(api_entity, ui_entity):
        api_value = api_entity.get(field)
        ui_value = ui_entity.get(field)
        api_norm = _r10_normalize_value(api_value, field)
        ui_norm = _r10_normalize_value(ui_value, field)
        if api_norm is None or ui_norm is None or api_norm == ui_norm:
            continue
        mismatches.append(
            {
                "field": field,
                "api_value": _r10_safe_value(api_value),
                "ui_value": _r10_safe_value(ui_value),
                "normalized_api_value": api_norm,
                "normalized_ui_value": ui_norm,
            }
        )
    return mismatches[:5]


def _r10_comparable_fields(api_entity: Mapping[str, Any], ui_entity: Mapping[str, Any]) -> List[str]:
    allowed = {
        "name",
        "title",
        "price",
        "quantity",
        "count",
        "status",
        "category",
        "total",
        "total_price",
        "stock",
        "email",
        "role",
        "date",
    }
    sensitive = {"password", "token", "authorization", "cookie", "secret", "access_token", "refresh_token"}
    fields = []
    for field in allowed.intersection(api_entity.keys()).intersection(ui_entity.keys()):
        lower = field.lower()
        if any(token in lower for token in sensitive):
            continue
        fields.append(field)
    return sorted(fields)


def _r10_normalize_value(value: Any, field: str) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ""
    lower = text.lower()
    if lower in {"true", "yes", "y", "active", "enabled"} and field in {"status", "role"}:
        return lower
    if lower in {"false", "no", "n", "inactive", "disabled"} and field in {"status", "role"}:
        return lower
    if field in {"price", "quantity", "count", "total", "total_price", "stock"}:
        numeric = _r10_numeric_value(text)
        return numeric if numeric is not None else lower
    if field == "date" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if field == "date":
        return None
    return " ".join(lower.split())


def _r10_numeric_value(text: str) -> float | None:
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _r10_safe_value(value: Any) -> Any:
    text = str(value)
    if any(token in text.lower() for token in ("password", "token", "authorization", "cookie")):
        return "<redacted>"
    return text[:120] if isinstance(value, str) else value


def _r10_issue_type(
    api_entity: Mapping[str, Any],
    ui_entity: Mapping[str, Any],
    mismatches: List[Mapping[str, Any]],
    before_observation: Mapping[str, Any],
) -> str:
    fields = {str(item.get("field") or "") for item in mismatches}
    identity = _r10_entity_identity(api_entity) or _r10_entity_identity(ui_entity)
    entity_type = identity[0] if identity else ""
    if "quantity" in fields and entity_type in {"product", "cart_item", "item", "entity"}:
        return "cart-quantity-mismatch"
    if _r10_stale_data_rendering(api_entity, ui_entity, mismatches, before_observation):
        return "stale-data-rendering"
    return "product-detail-mismatch"


def _r10_stale_data_rendering(
    api_entity: Mapping[str, Any],
    ui_entity: Mapping[str, Any],
    mismatches: List[Mapping[str, Any]],
    before_observation: Mapping[str, Any],
) -> bool:
    before_api = _r10_extract_api_entities({}, before_observation)
    before_ui = _r10_extract_ui_entities(before_observation)
    identity = _r10_entity_identity(api_entity)
    if not identity:
        return False
    old_api = next((item for item in before_api if _r10_entity_identity(item) == identity), None)
    old_ui = next((item for item in before_ui if _r10_entity_identity(item) == identity), None)
    if not old_api or not old_ui:
        return False
    for mismatch in mismatches:
        field = str(mismatch.get("field") or "")
        before_api_value = _r10_normalize_value(old_api.get(field), field)
        before_ui_value = _r10_normalize_value(old_ui.get(field), field)
        after_api_value = _r10_normalize_value(api_entity.get(field), field)
        after_ui_value = _r10_normalize_value(ui_entity.get(field), field)
        if before_api_value == before_ui_value and after_api_value != before_api_value and after_ui_value == before_ui_value:
            return True
    return False


def _r10_api_status(action_info: Mapping[str, Any]) -> int | None:
    for key in ("status_code", "response_status", "api_status", "network_status"):
        status = _r08_int(action_info.get(key))
        if status is not None:
            return status
    entries = action_info.get("network_entries", []) or action_info.get("network", []) or []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                status = _r08_int(entry.get("status") or entry.get("status_code"))
                if status is not None:
                    return status
    return None


def _r10_network_failure(action_info: Mapping[str, Any]) -> bool:
    text = " ".join(str(action_info.get(key) or "") for key in ("network_error", "error", "action_error", "fallback_error")).lower()
    return any(token in text for token in ("network error", "failed to fetch", "connection reset", "timeout", "dns"))


def _r10_ui_settled(observation: Mapping[str, Any]) -> bool:
    runtime = observation.get("runtime_signals", {})
    page_state = observation.get("page_state", {})
    for container in (runtime, page_state):
        if isinstance(container, Mapping):
            if bool(container.get("loading") or container.get("pending") or container.get("request_pending")):
                return False
    page_text = _page_text(observation).lower()
    if any(token in page_text for token in ("loading", "spinner", "loading...", "로딩")):
        return False
    return True


def _detect_r11_calculation_error(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    del before_observation
    if last_action_error or _has_api_failure(action_info) or _r10_network_failure(action_info) or _r12_timeout_failure(action_info) or not _r10_ui_settled(after_observation):
        return None
    calculations = _r11_extract_operands(after_observation)
    for calculation in calculations:
        if _is_r11_calculation_error(calculation):
            return _r11_anomaly(calculation, action_type)
    return None


def _is_r11_calculation_error(calculation: Mapping[str, Any]) -> bool:
    expected = _r11_number(calculation.get("expected_value"))
    actual = _r11_number(calculation.get("actual_value"))
    tolerance = _r11_tolerance(expected)
    return bool(expected is not None and actual is not None and not _r11_within_tolerance(expected, actual, tolerance))


def _r11_extract_operands(observation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    calculations: List[Dict[str, Any]] = []
    items = _r11_cart_items(observation)
    subtotal_actual = _r11_first_number(observation, ("subtotal", "sub_total"))
    total_actual = _r11_first_number(observation, ("total", "total_price", "grand_total", "amount"))
    if items:
        expected_subtotal = sum(item["unit_price"] * item["quantity"] for item in items)
        actual = subtotal_actual if subtotal_actual is not None else total_actual if not _r11_has_adjustments(observation) else None
        if actual is not None:
            calculations.append(
                {
                    "calculation_type": "cart_subtotal",
                    "operands": [
                        {"entity_id": item["entity_id"], "unit_price": item["unit_price"], "quantity": item["quantity"]}
                        for item in items
                    ],
                    "expected_value": expected_subtotal,
                    "actual_value": actual,
                    "item_count": len(items),
                    "entity_ids": [item["entity_id"] for item in items],
                    "subtotal": subtotal_actual,
                    "discount": None,
                    "tax": None,
                    "shipping": None,
                }
            )
    breakdown = _r11_breakdown_operands(observation)
    if breakdown:
        calculations.append(breakdown)
    return calculations


def _r11_cart_items(observation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_items = _r11_raw_items(observation)
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        price = _r11_first_item_number(item, ("unit_price", "price"))
        quantity = _r11_first_item_number(item, ("quantity", "qty", "count"))
        if price is None or quantity is None:
            continue
        identity = _r11_item_identity(item)
        if not identity or identity in deduped:
            continue
        deduped[identity] = {"entity_id": identity, "unit_price": price, "quantity": quantity}
    return list(deduped.values())


def _r11_raw_items(observation: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    for container in (observation, observation.get("page_state", {}), observation.get("runtime_signals", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("cart_items", "line_items", "items"):
            value = container.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, Mapping))
    return items


def _r11_item_identity(item: Mapping[str, Any]) -> str:
    for key in ("cart_item_id", "product_id", "item_id", "entity_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{str(value).strip()}"
    return ""


def _r11_breakdown_operands(observation: Mapping[str, Any]) -> Dict[str, Any] | None:
    subtotal = _r11_first_number(observation, ("subtotal", "sub_total"))
    total = _r11_first_number(observation, ("total", "total_price", "grand_total", "amount"))
    if subtotal is None or total is None:
        return None
    adjustments = _r11_adjustments(observation)
    if not adjustments["has_adjustment"]:
        return None
    expected = subtotal - adjustments["discount"] + adjustments["tax"] + adjustments["shipping"]
    return {
        "calculation_type": "total_breakdown",
        "operands": {
            "subtotal": subtotal,
            "discount": adjustments["discount"],
            "tax": adjustments["tax"],
            "shipping": adjustments["shipping"],
        },
        "expected_value": expected,
        "actual_value": total,
        "item_count": 0,
        "entity_ids": [],
        "subtotal": subtotal,
        "discount": adjustments["discount"],
        "tax": adjustments["tax"],
        "shipping": adjustments["shipping"],
    }


def _r11_adjustments(observation: Mapping[str, Any]) -> Dict[str, Any]:
    discount = _r11_first_number(observation, ("discount_amount", "discount"))
    tax = _r11_first_number(observation, ("tax_amount", "tax"))
    shipping = _r11_first_number(observation, ("shipping_fee", "shipping"))
    return {
        "discount": discount or 0.0,
        "tax": tax or 0.0,
        "shipping": shipping or 0.0,
        "has_adjustment": discount is not None or tax is not None or shipping is not None,
    }


def _r11_has_adjustments(observation: Mapping[str, Any]) -> bool:
    return bool(_r11_adjustments(observation)["has_adjustment"])


def _r11_first_number(observation: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for container in (observation.get("page_state", {}), observation.get("runtime_signals", {}), observation):
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if key in container:
                number = _r11_number(container.get(key))
                if number is not None:
                    return number
    return None


def _r11_first_item_number(item: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in item:
            number = _r11_number(item.get(key))
            if number is not None:
                return number
    return None


def _r11_number(value: Any) -> float | None:
    if value is None:
        return None
    return _r10_numeric_value(str(value))


def _r11_tolerance(expected: float | None) -> float:
    if expected is None:
        return 0.01
    return max(0.01, abs(expected) * 0.000001)


def _r11_within_tolerance(expected: float, actual: float, tolerance: float) -> bool:
    return abs(expected - actual) <= tolerance


def _r11_anomaly(calculation: Mapping[str, Any], action_type: str) -> Dict[str, Any]:
    expected = float(calculation.get("expected_value") or 0.0)
    actual = float(calculation.get("actual_value") or 0.0)
    tolerance = _r11_tolerance(expected)
    return {
        "type": "cart-total-mismatch",
        "confidence": 0.86,
        "evidence": {
            "rule_id": "R11",
            "issue_type": "calculation-mismatch",
            "calculation_type": calculation.get("calculation_type"),
            "operands": calculation.get("operands"),
            "expected_value": expected,
            "actual_value": actual,
            "difference": round(actual - expected, 6),
            "tolerance": tolerance,
            "item_count": calculation.get("item_count"),
            "entity_ids": calculation.get("entity_ids", []),
            "subtotal": calculation.get("subtotal"),
            "discount": calculation.get("discount"),
            "tax": calculation.get("tax"),
            "shipping": calculation.get("shipping"),
            "ui_settled": True,
            "action_type": action_type,
        },
    }


def _detect_r12_persistence_issue(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    del before_observation
    if last_action_error or _has_api_failure(action_info) or _r10_network_failure(action_info) or _r12_timeout_failure(action_info) or not _r10_ui_settled(after_observation):
        return None
    operation = _r12_write_operation(action_info, action_type)
    if not operation:
        return None
    success = _r12_write_success_evidence(action_info, after_observation)
    if not success:
        return None
    expected = _r12_expected_post_write_state(action_info, operation)
    if not expected:
        return None
    followup_entities, refetch_detected = _r12_observed_persisted_state(after_observation, action_info)
    if not refetch_detected:
        return None
    failure = _is_r12_persistence_failure(expected, followup_entities)
    if not failure:
        return None
    return _r12_anomaly(operation, expected, failure, success, action_type, refetch_detected)


def _is_r12_persistence_failure(
    expected: Mapping[str, Any],
    followup_entities: List[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    operation = str(expected.get("operation") or "")
    expected_entity = expected.get("entity", {})
    if not isinstance(expected_entity, Mapping):
        return None
    identity = _r10_entity_identity(expected_entity)
    if not identity:
        return None
    matched = next((entity for entity in followup_entities if _r10_entity_identity(entity) == identity), None)
    if operation == "delete":
        if matched:
            return {
                "issue_type": "delete-persistence-failed",
                "entity": matched,
                "expected_exists": False,
                "observed_exists": True,
            }
        return None
    if operation == "create":
        if matched is None and _r12_followup_has_entity_type(followup_entities, identity[0]):
            return {
                "issue_type": "create-persistence-missing",
                "entity": {},
                "expected_exists": True,
                "observed_exists": False,
            }
        return None
    if matched is None:
        return None
    fields = _r12_persisted_fields(expected_entity, matched)
    for field in fields:
        expected_value = _r12_normalize_value(expected_entity.get(field), field)
        observed_value = _r12_normalize_value(matched.get(field), field)
        if expected_value != observed_value:
            return {
                "issue_type": "update-persistence-mismatch",
                "entity": matched,
                "field": field,
                "expected_persisted_value": expected_entity.get(field),
                "observed_persisted_value": matched.get(field),
            }
    return None


def _r12_write_operation(action_info: Mapping[str, Any], action_type: str) -> Dict[str, Any]:
    method = _r12_http_method(action_info)
    action_text = _r12_action_text(action_info, action_type)
    operation = str(action_info.get("operation") or action_info.get("write_operation") or "").lower()
    if not operation:
        if method == "POST" or any(token in action_text for token in ("create", "post", "add", "register")):
            operation = "create"
        elif method in {"PUT", "PATCH"} or any(token in action_text for token in ("save", "update", "edit")):
            operation = "update"
        elif method == "DELETE" or any(token in action_text for token in ("delete", "remove")):
            operation = "delete"
    if operation not in {"create", "update", "delete"}:
        return {}
    return {"operation": operation, "http_method": method}


def _r12_http_method(action_info: Mapping[str, Any]) -> str:
    for key in ("method", "http_method", "request_method"):
        value = str(action_info.get(key) or "").upper()
        if value:
            return value
    entries = action_info.get("network_entries", []) or action_info.get("network", []) or []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                method = str(entry.get("method") or "").upper()
                if method in {"POST", "PUT", "PATCH", "DELETE"}:
                    return method
    return ""


def _r12_action_text(action_info: Mapping[str, Any], action_type: str) -> str:
    action = action_info.get("action", {})
    parts = [action_type]
    if isinstance(action, Mapping):
        parts.extend(str(action.get(key) or "") for key in ("action_type", "text", "name", "label"))
    parts.extend(str(action_info.get(key) or "") for key in ("action_name", "action_label", "operation", "write_operation"))
    return " ".join(parts).lower()


def _r12_write_success_evidence(action_info: Mapping[str, Any], after_observation: Mapping[str, Any]) -> List[str]:
    evidence: List[str] = []
    status = _r10_api_status(action_info)
    method = _r12_http_method(action_info)
    if method in {"POST", "PUT", "PATCH", "DELETE"} and status is not None and 200 <= status < 300:
        evidence.append("write_2xx_response")
    if _r08_action_success(action_info):
        evidence.append("action_success")
    if any(bool(action_info.get(key)) for key in ("save_success", "create_success", "delete_success", "update_success")):
        evidence.append("write_success_flag")
    if _r08_success_message_visible(after_observation):
        evidence.append("success_message")
    if _r12_response_entity(action_info):
        evidence.append("response_entity")
    return sorted(set(evidence))


def _r12_expected_post_write_state(action_info: Mapping[str, Any], operation: Mapping[str, Any]) -> Dict[str, Any]:
    op = str(operation.get("operation") or "")
    entity = _r12_expected_entity(action_info, op)
    if not entity:
        return {}
    return {"operation": op, "http_method": operation.get("http_method", ""), "write_status": _r10_api_status(action_info), "entity": entity}


def _r12_expected_entity(action_info: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    if operation == "delete":
        for key in ("deleted_entity", "target_entity", "entity", "submitted_entity"):
            value = action_info.get(key)
            if isinstance(value, Mapping) and _r10_entity_identity(value):
                return value
    for key in ("expected_post_write_state", "written_entity", "response_entity", "created_entity", "updated_entity", "submitted_entity", "entity"):
        value = action_info.get(key)
        if isinstance(value, Mapping) and _r10_entity_identity(value):
            return value
    response_entity = _r12_response_entity(action_info)
    if response_entity:
        return response_entity
    return {}


def _r12_response_entity(action_info: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("response_entity", "created_entity", "updated_entity"):
        value = action_info.get(key)
        if isinstance(value, Mapping) and _r10_entity_identity(value):
            return value
    entities = _r10_extract_api_entities(action_info, {})
    return entities[0] if entities else {}


def _r12_observed_persisted_state(
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> tuple[List[Mapping[str, Any]], bool]:
    entities: List[Mapping[str, Any]] = []
    refetch_detected = bool(
        action_info.get("refetch_detected")
        or action_info.get("followup_observation_present")
        or action_info.get("reload_detected")
        or action_info.get("refresh_detected")
    )
    for container in (action_info, after_observation, after_observation.get("page_state", {}), after_observation.get("runtime_signals", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("followup_entities", "refetched_entities", "refreshed_entities", "post_write_state", "persisted_entities", "current_entities"):
            value = container.get(key)
            if value is not None:
                refetch_detected = True
                entities.extend(_r10_entities_from_value(value, source=f"r12:{key}"))
    return _r10_dedup_entities(entities), refetch_detected


def _r12_followup_has_entity_type(entities: List[Mapping[str, Any]], entity_type: str) -> bool:
    return any((_r10_entity_identity(entity) or ("", "", ""))[0] == entity_type for entity in entities)


def _r12_persisted_fields(expected_entity: Mapping[str, Any], observed_entity: Mapping[str, Any]) -> List[str]:
    allowed = {"name", "title", "status", "quantity", "price", "role", "category", "email", "content", "text", "description", "password"}
    return sorted(field for field in allowed.intersection(expected_entity.keys()).intersection(observed_entity.keys()))


def _r12_normalize_value(value: Any, field: str) -> Any:
    if field in {"quantity", "price"}:
        return _r10_normalize_value(value, field)
    if field in {"status", "role", "category", "email"}:
        return str(value or "").strip().lower()
    if field == "password":
        return str(value or "")
    return str(value or "").strip()


def _r12_timeout_failure(action_info: Mapping[str, Any]) -> bool:
    if bool(action_info.get("timeout") or action_info.get("timeout_occurred")):
        return True
    text = " ".join(str(action_info.get(key) or "") for key in ("error", "action_error", "network_error")).lower()
    return "timeout" in text or "timed out" in text


def _r12_same_entity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(_r10_entity_identity(left) and _r10_entity_identity(left) == _r10_entity_identity(right))


def _r12_same_field(left: Mapping[str, Any], right: Mapping[str, Any], field: str) -> bool:
    return field in left and field in right


def _r12_anomaly(
    operation: Mapping[str, Any],
    expected: Mapping[str, Any],
    failure: Mapping[str, Any],
    success: List[str],
    action_type: str,
    refetch_detected: bool,
) -> Dict[str, Any]:
    expected_entity = expected.get("entity", {})
    expected_entity = expected_entity if isinstance(expected_entity, Mapping) else {}
    identity = _r10_entity_identity(expected_entity) or ("", "", "")
    field = str(failure.get("field") or "")
    anomaly_type = "forum-comment-delete-failed" if failure.get("issue_type") == "delete-persistence-failed" and identity[0] == "comment" else "persistence-mismatch"
    return {
        "type": anomaly_type,
        "confidence": 0.86,
        "evidence": {
            "rule_id": "R12",
            "issue_type": failure.get("issue_type"),
            "operation": operation.get("operation"),
            "http_method": operation.get("http_method"),
            "entity_type": identity[0],
            "entity_id": identity[1],
            "field": field,
            "before_value": _r12_safe_value(expected_entity.get("before_value"), field),
            "written_value": _r12_safe_value(expected_entity.get(field), field),
            "expected_persisted_value": _r12_safe_value(failure.get("expected_persisted_value", expected_entity.get(field)), field),
            "observed_persisted_value": _r12_safe_value(failure.get("observed_persisted_value"), field),
            "expected_exists": failure.get("expected_exists"),
            "observed_exists": failure.get("observed_exists"),
            "write_request_succeeded": bool(success),
            "write_status": expected.get("write_status") or "",
            "write_success_evidence": success,
            "followup_observation_present": True,
            "refetch_detected": refetch_detected,
            "ui_settled": True,
            "identity_source": identity[2],
            "action_type": action_type,
        },
    }


def _r12_safe_value(value: Any, field: str) -> Any:
    if field and any(token in field.lower() for token in ("password", "token", "authorization", "cookie", "secret")):
        text = str(value or "")
        return {"value_changed": bool(text), "value_length": len(text)}
    return _r10_safe_value(value)


def _detect_broken_navigation(
    clicked_candidate: Mapping[str, Any] | None,
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    before_url: str,
    after_url: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if action_type not in {"click_element", "press_enter"} or last_action_error or _has_api_failure(action_info):
        return None
    runtime = after_observation.get("runtime_signals", {})
    if bool(action_info.get("login_success")) or (isinstance(runtime, Mapping) and bool(runtime.get("login_success"))):
        return None
    candidate = clicked_candidate or {}
    expected_url = _expected_navigation_url(candidate, action_info)
    expected_entity_id = _expected_entity_id(candidate, expected_url)
    actual_entity_id = _actual_entity_id(after_observation, after_url)
    if not (expected_url or expected_entity_id):
        return None
    if before_url == after_url and not actual_entity_id:
        return None
    if not _is_r02_wrong_navigation(
        expected_url=expected_url,
        actual_url=after_url,
        expected_entity_id=expected_entity_id,
        actual_entity_id=actual_entity_id,
        last_action_error=last_action_error,
        allowed_redirects=_allowed_redirects(action_info),
        base_url=before_url,
    ):
        return None
    return {
        "type": "broken-navigation",
        "confidence": 0.85,
        "evidence": {
            "rule_id": "R02",
            "expected_url": expected_url,
            "actual_url": after_url,
            "expected_entity_id": expected_entity_id,
            "actual_entity_id": actual_entity_id,
            "clicked_href": _candidate_href(candidate),
            "clicked_text": _candidate_text(candidate),
            "clicked_bid": candidate.get("bid"),
            "route_changed": before_url != after_url,
            "last_action_error": last_action_error,
            "action_type": action_type,
        },
    }


def _detect_login_redirect_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    before_url: str,
    after_url: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if last_action_error or _has_api_failure(action_info):
        return None
    runtime = after_observation.get("runtime_signals", {})
    login_success = bool(action_info.get("login_success")) or (
        isinstance(runtime, Mapping) and bool(runtime.get("login_success"))
    )
    if not login_success:
        return None
    expected_url = _login_expected_url(action_info)
    if not expected_url:
        return None
    actual_url = str(action_info.get("post_login_url") or (runtime.get("post_login_url") if isinstance(runtime, Mapping) else "") or after_url)
    allowed_redirects = _allowed_redirects(action_info)
    allowed_redirects.update(_login_allowed_redirects(action_info))
    if not _is_r02_wrong_navigation(
        expected_url=expected_url,
        actual_url=actual_url,
        last_action_error=last_action_error,
        allowed_redirects=allowed_redirects,
        base_url=before_url,
    ):
        return None
    return {
        "type": "login-redirect-mismatch",
        "confidence": 0.82,
        "evidence": {
            "rule_id": "R02",
            "expected_url": expected_url,
            "actual_url": actual_url,
            "expected_entity_id": "",
            "actual_entity_id": "",
            "clicked_href": "",
            "clicked_text": "",
            "clicked_bid": "",
            "route_changed": before_url != actual_url,
            "last_action_error": last_action_error,
            "login_success": True,
        },
    }


def _page_text_length(observation: Mapping[str, Any]) -> int:
    return int(observation.get("page_state", {}).get("page_text_length", 0) or 0)


def _normalize_navigation_url(url: Any, base_url: str = "") -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    parsed = urlparse(urljoin(str(base_url or ""), text))
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return path.lower() or "/"


def _candidate_href(candidate: Mapping[str, Any]) -> str:
    for key in ("href", "target_url", "url", "to", "route"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _expected_navigation_url(candidate: Mapping[str, Any], action_info: Mapping[str, Any]) -> str:
    for key in ("expected_url", "target_url", "expected_route"):
        value = action_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _candidate_href(candidate)


def _clean_entity_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else ""


def _expected_entity_id(candidate: Mapping[str, Any], expected_url: str) -> str:
    for key in ("entity_id", "product_id", "post_id", "item_id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _entity_id_from_url(expected_url)


def _actual_entity_id(observation: Mapping[str, Any], actual_url: str) -> str:
    page_state = observation.get("page_state", {})
    runtime = observation.get("runtime_signals", {})
    for source in (page_state, runtime):
        if not isinstance(source, Mapping):
            continue
        for key in ("entity_id", "product_id", "post_id", "item_id"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return _entity_id_from_url(actual_url)


def _entity_id_from_url(url: Any) -> str:
    path = urlparse(str(url or "")).path
    if not path:
        return ""
    match = re.search(r"/(?:products?|items?|posts?|books?)/([^/?#]+)", path, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _allowed_redirects(action_info: Mapping[str, Any]) -> set[str]:
    redirects: set[str] = set()
    for key in ("allowed_redirects", "allowed_routes", "redirect_chain"):
        value = action_info.get(key)
        if isinstance(value, (list, tuple, set)):
            redirects.update(str(item) for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            redirects.add(str(value))
    return redirects


def _login_expected_url(action_info: Mapping[str, Any]) -> str:
    for key in ("expected_url", "expected_redirect_url", "login_expected_url", "expected_landing_url"):
        value = action_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _login_allowed_redirects(action_info: Mapping[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for key in ("success_url_contains", "login_success_url_contains"):
        value = action_info.get(key)
        if isinstance(value, (list, tuple, set)):
            allowed.update(str(item) for item in value if str(item).strip())
    login_config = action_info.get("login")
    if isinstance(login_config, Mapping):
        value = login_config.get("success_url_contains")
        if isinstance(value, (list, tuple, set)):
            allowed.update(str(item) for item in value if str(item).strip())
    return allowed


def _has_api_failure(action_info: Mapping[str, Any]) -> bool:
    for key in ("api_500_count", "api_5xx_count", "server_5xx_count"):
        try:
            if int(action_info.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    for key in ("response_status", "network_status", "status"):
        try:
            status = int(action_info.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if status >= 500:
            return True
    text = " ".join(str(action_info.get(key) or "") for key in ("network_error", "api_error", "error")).lower()
    return any(token in text for token in ("500", "5xx", "connection refused", "network failure", "failed to fetch"))


def _runtime_values(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    keys: tuple[str, ...],
) -> List[Any]:
    values: List[Any] = []
    runtime = observation.get("runtime_signals", {})
    page_state = observation.get("page_state", {})
    for source in (action_info, runtime, page_state):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if value:
                values.append(value)
    return values


def _console_error_values(observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> List[Any]:
    values = _runtime_values(observation, action_info, ("console_errors", "console_error", "console_messages", "console"))
    filtered: List[Any] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    level = str(item.get("type") or item.get("level") or "").lower()
                    if level and level != "error":
                        continue
                    filtered.append(item.get("text") or item.get("message") or item)
                else:
                    filtered.append(item)
        elif isinstance(value, Mapping):
            level = str(value.get("type") or value.get("level") or "").lower()
            if not level or level == "error":
                filtered.append(value.get("text") or value.get("message") or value)
        else:
            filtered.append(value)
    return filtered


def _first_runtime_error_text(values: List[Any]) -> str:
    for value in values:
        for text in _flatten_text_values(value):
            cleaned = " ".join(str(text).split())
            if _is_meaningful_runtime_error(cleaned):
                return cleaned
    return ""


def _flatten_text_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            str(value.get(key) or "")
            for key in ("message", "text", "error", "name", "stack")
            if str(value.get(key) or "").strip()
        ]
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            result.extend(_flatten_text_values(item))
        return result
    return [str(value)]


def _is_meaningful_runtime_error(text: str) -> bool:
    lower = text.lower()
    if not lower:
        return False
    if any(token in lower for token in ("console.warn", "warning:", "[warn]", "debug", "info:", "favicon", "dev server", "react devtools")):
        return False
    if any(token in lower for token in ("failed to fetch", "networkerror", "network error", " 404", " 500", "status 500")):
        return False
    strong_tokens = (
        "uncaught",
        "pageerror",
        "typeerror",
        "referenceerror",
        "syntaxerror",
        "rangeerror",
        "unhandled promise rejection",
        "cannot read properties",
        "is not defined",
    )
    return any(token in lower for token in strong_tokens)


def _runtime_error_type(text: str) -> str:
    lower = text.lower()
    for name in ("TypeError", "ReferenceError", "SyntaxError", "RangeError"):
        if name.lower() in lower:
            return name
    if "unhandled promise rejection" in lower:
        return "UnhandledPromiseRejection"
    if "uncaught" in lower:
        return "UncaughtError"
    if "pageerror" in lower:
        return "PageError"
    return "RuntimeError"


def _normal_empty_state_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    tokens = (
        "no results",
        "no result",
        "no data",
        "no posts yet",
        "no comments",
        "empty cart",
        "cart is empty",
        "검색 결과가 없습니다",
        "결과가 없습니다",
        "장바구니가 비어",
        "데이터가 없습니다",
    )
    return any(token in text for token in tokens)


def _invalid_rendered_value(text: str) -> str:
    patterns = (
        r"\b(?:price|total|amount|subtotal|name|username|title|quantity|qty|가격|총액|합계|사용자명|이름|제목)\s*[:：]\s*(undefined|nan|\[object object\])\b",
        r"\b(undefined|nan|\[object object\])\b\s*(?:원|usd|달러|개|items?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)[:120]
    return ""


def _candidate_at(candidates: Any, index: int) -> Mapping[str, Any] | None:
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _visibility(candidate: Mapping[str, Any]) -> float:
    if not bool(candidate.get("visible", True)):
        return 0.0
    try:
        return float(candidate.get("visibility", 1.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_clickable_target(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role", "")).lower()
    return bool(candidate.get("clickable")) or role in CLICKABLE_ROLES


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(str(candidate.get(key, "") or "") for key in ("name", "text", "label", "aria_label"))


def _is_actionable_candidate(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return bool(
        _candidate_visible(candidate)
        and bool(candidate.get("enabled", True))
        and (
            bool(candidate.get("clickable"))
            or bool(candidate.get("is_interactive"))
            or role in {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "menuitem", "tab"}
            or tag in {"button", "a", "input", "textarea", "select", "option"}
        )
    )


def _candidate_visible(candidate: Mapping[str, Any]) -> bool:
    if not bool(candidate.get("visible", True)):
        return False
    try:
        return float(candidate.get("visibility", 1.0) or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _target_evidence(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bid": candidate.get("bid"),
        "data_bug_id": candidate.get("data_bug_id"),
        "name": candidate.get("name"),
        "text": candidate.get("text"),
        "href": candidate.get("href"),
        "role": candidate.get("role"),
        "tag": candidate.get("tag"),
        "clickable": candidate.get("clickable"),
        "visible": candidate.get("visible"),
        "visibility": candidate.get("visibility"),
        "is_purchase_action": candidate.get("is_purchase_action"),
        "is_workout_add_action": candidate.get("is_workout_add_action"),
        "is_chart_related": candidate.get("is_chart_related"),
        "catalog_keyword_matches": candidate.get("catalog_keyword_matches", []),
        "catalog_selector_match": candidate.get("catalog_selector_match"),
        "catalog_bug_id_matches": candidate.get("catalog_bug_id_matches", []),
        "selector_hint": candidate.get("selector_hint"),
        "is_interactive": candidate.get("is_interactive"),
        "is_form_field": candidate.get("is_form_field"),
        "openended_keyword_matches": candidate.get("openended_keyword_matches", []),
    }


def _has_action_keyword(text: str) -> bool:
    haystack = text.lower()
    return any(keyword.lower() in haystack for keyword in ACTION_KEYWORDS)


def _is_openended_target(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return bool(
        candidate.get("openended_keyword_matches")
        or candidate.get("is_interactive")
        or candidate.get("is_login_related")
        or candidate.get("is_cart_related")
        or candidate.get("is_checkout_related")
        or candidate.get("is_search_related")
        or candidate.get("is_filter_related")
        or candidate.get("is_submit_related")
        or tag in {"button", "a", "input", "select", "textarea"}
        or role in {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio"}
    )


def _looks_like_form_submission(candidate: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    if candidate.get("is_submit_related") or candidate.get("is_login_related") or candidate.get("is_search_related"):
        return _has_form_signal(observation)
    text = _candidate_text(candidate).lower()
    return _has_form_signal(observation) and any(
        token in text for token in ("submit", "save", "continue", "next", "login", "sign in", "search", "checkout")
    )


def _openended_interaction_type_and_confidence(
    candidate: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> tuple[str, float]:
    if _has_explicit_async_or_hang_keyword(candidate):
        return "async-hang", 0.78
    if _is_same_page_cart_reclick(candidate, observation):
        return "button-no-response", 0.5
    if _looks_like_submit_button(candidate):
        return "form-no-feedback", 0.75 if _has_form_signal(observation) else 0.62
    if _looks_like_form_submission(candidate, observation) and not bool(candidate.get("is_form_field")):
        return "form-no-feedback", 0.65
    if bool(candidate.get("is_form_field")):
        return "form-no-feedback", 0.45
    return "button-no-response", 0.68


def _should_emit_openended_interaction_anomaly(
    anomaly_type: str,
    candidate: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    context_evidence: Mapping[str, Any],
) -> bool:
    if anomaly_type != "form-no-feedback":
        return True
    if _has_explicit_async_or_hang_keyword(candidate):
        return True

    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    plain_textbox_click = role in {"textbox", "searchbox"} or tag in {"input", "textarea"}
    if plain_textbox_click and not _looks_like_submit_button(candidate):
        return False

    if "query_text" in context_evidence:
        if bool(context_evidence.get("error_message_visible")) or bool(context_evidence.get("empty_state_visible")):
            return False
        if not str(context_evidence.get("query_text") or "").strip() and plain_textbox_click:
            return False

    if "checkout_step" in context_evidence:
        if bool(context_evidence.get("validation_message_visible")) or bool(context_evidence.get("error_message_visible")):
            return False
        if bool(context_evidence.get("next_step_reached")):
            return False

    return True


def _context_review_adjustments(
    anomaly_type: str,
    candidate: Mapping[str, Any],
    context_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    if anomaly_type != "form-no-feedback":
        return {}
    text = _candidate_text(candidate).strip() or str(candidate.get("role") or "")
    if "query_text" in context_evidence:
        query_text = str(context_evidence.get("query_text") or "").strip()
        if not query_text:
            return {
                "human_review_status": "likely_false_positive",
                "likely_false_positive_reason": "search was triggered without a query or explicit submit evidence",
                "review_question": "Search had no query text. Is an empty search expected to change results or show validation?",
                "confidence_cap": 0.4,
            }
        return {
            "human_review_status": "likely_true_positive",
            "review_question": "Search had a query but no result, empty-state, error, route, toast, or page update was detected.",
        }
    if "checkout_step" in context_evidence:
        if context_evidence.get("empty_required_fields") and not (
            context_evidence.get("validation_message_visible") or context_evidence.get("error_message_visible")
        ):
            return {
                "human_review_status": "likely_true_positive",
                "review_question": "Checkout had empty required fields but no validation or error feedback was visible.",
            }
        return {
            "human_review_status": "needs_review",
            "review_question": f"Verify whether {text} should advance checkout or display validation feedback for this form state.",
        }
    return {}


def _looks_like_submit_button(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    text = _candidate_text(candidate).lower()
    if bool(candidate.get("is_submit_related") or candidate.get("is_login_related") or candidate.get("is_checkout_related")):
        return True
    if role not in {"button", "link", "menuitem"} and tag not in {"button", "a"}:
        return False
    return any(token in text for token in ("submit", "save", "continue", "next", "login", "sign in", "search", "checkout"))


def _has_explicit_async_or_hang_keyword(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate).lower()
    return any(token in text for token in ("hang", "async", "stuck", "loading"))


def _is_same_page_cart_reclick(candidate: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate).lower()
    return "cart" in text and _url(observation).rstrip("/").endswith("/cart")


def _human_review_metadata(
    anomaly_type: str,
    candidate: Mapping[str, Any],
    before_url: str,
    after_url: str,
    route_changed: bool,
) -> Dict[str, Any]:
    text = _candidate_text(candidate).strip() or str(candidate.get("role") or "")
    lower_text = text.lower()
    if route_changed:
        return {
            "human_review_status": "likely_false_positive",
            "likely_false_positive_reason": "route or cart state changed after click",
            "review_question": f"{text} changed route from {before_url} to {after_url}; verify whether any no-response label is a false positive.",
        }
    if "cart" in lower_text and before_url.rstrip("/").endswith("/cart") and after_url.rstrip("/").endswith("/cart"):
        return {
            "human_review_status": "likely_false_positive",
            "likely_false_positive_reason": "already on cart page; same-page navigation click may be an expected no-op",
            "review_question": "Already on cart page. Should clicking Cart again trigger any UI change?",
        }
    if _has_explicit_async_or_hang_keyword(candidate):
        return {
            "human_review_status": "likely_true_positive",
            "review_question": f"{text} suggests async/hang behavior; verify whether clicking/input/submitting causes a stuck or missing-feedback state.",
        }
    if anomaly_type == "form-no-feedback":
        if "checkout" in lower_text:
            question = "Should Checkout advance to the next step or show validation feedback when clicked?"
        elif "search" in lower_text:
            question = "After entering a query, should Search update results or show validation/no-result feedback?"
        else:
            question = f"Should {text} produce validation, route, toast, or result changes?"
        return {"human_review_status": "needs_review", "review_question": question}
    return {
        "human_review_status": "needs_review",
        "review_question": f"Should clicking {text} change URL, modal, toast, page text, or component state?",
    }


def _openended_context_evidence(
    candidate: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    before_url: str,
    after_url: str,
) -> Dict[str, Any]:
    text = _candidate_text(candidate).lower()
    evidence: Dict[str, Any] = {}
    if "search" in text:
        query_text, search_input_found = _search_query_context(before_observation)
        evidence.update(
            {
                "query_text": query_text,
                "search_input_found": search_input_found,
                "search_result_count_before": _search_result_count(before_observation),
                "search_result_count_after": _search_result_count(after_observation),
                "empty_state_visible": _empty_state_visible(after_observation),
                "error_message_visible": _error_message_visible(after_observation),
            }
        )
    if "checkout" in text or "/checkout" in before_url or "/checkout" in after_url:
        required_fields, empty_required = _checkout_field_context(before_observation)
        evidence.update(
            {
                "checkout_step": _checkout_step(before_url),
                "required_fields_found": required_fields,
                "empty_required_fields": empty_required,
                "validation_message_visible": _validation_message_visible(after_observation),
                "next_step_reached": bool(before_url != after_url and _checkout_step(before_url) != _checkout_step(after_url)),
                "error_message_visible": _error_message_visible(after_observation),
            }
        )
    if _has_explicit_async_or_hang_keyword(candidate):
        evidence.update(
            {
                "loading_visible_before": _loading_visible(before_observation),
                "loading_visible_after": _loading_visible(after_observation),
                "timeout_message_visible": _timeout_message_visible(after_observation),
                "async_keyword_match": _keyword_matches(_candidate_text(candidate), ["hang", "async", "timeout", "loading"]),
                "page_frozen_signal": _state_signature(before_observation) == _state_signature(after_observation),
            }
        )
    return evidence


def _search_query_context(observation: Mapping[str, Any]) -> tuple[str, bool]:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return "", False
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        role = str(candidate.get("role") or "").lower()
        text = _candidate_text(candidate).strip()
        haystack = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "aria_label", "id", "class_name")).lower()
        if role in {"textbox", "searchbox"} or "search" in haystack:
            if text and "search" not in text.lower():
                return text, True
            return "", True
    return "", False


def _search_result_count(observation: Mapping[str, Any]) -> int:
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return 0
    count = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        text = _candidate_text(candidate).lower()
        if any(token in text for token in ("add", "details", "book", "product", "price", "cart")):
            count += 1
    return count


def _checkout_field_context(observation: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    fields: List[str] = []
    empty: List[str] = []
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return fields, empty
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not bool(candidate.get("is_form_field")):
            continue
        label = _candidate_text(candidate).strip() or str(candidate.get("bid") or "field")
        fields.append(label)
        if not str(candidate.get("text") or candidate.get("name") or "").strip():
            empty.append(label)
    return fields[:12], empty[:12]


def _checkout_step(url: str) -> str:
    parts = str(url or "").rstrip("/").split("/")
    if "step" in parts:
        index = parts.index("step")
        if index + 1 < len(parts):
            return parts[index + 1]
    if "checkout" in parts:
        return "checkout"
    return ""


def _validation_message_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in ("required", "invalid", "validation", "필수", "오류", "입력"))


def _error_message_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in ("error", "failed", "invalid", "오류", "실패"))


def _empty_state_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in ("no result", "no data", "empty", "not found", "없습니다"))


def _loading_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in ("loading", "pending", "spinner", "로딩", "처리 중"))


def _timeout_message_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in ("timeout", "timed out", "시간 초과"))


def _is_openended_profile(site_profile: Mapping[str, Any]) -> bool:
    return str(site_profile.get("mode") or "") == "openended_anomaly_discovery" or not bool(site_profile.get("bugs"))


def _is_purchase_action(candidate: Mapping[str, Any]) -> bool:
    if bool(candidate.get("is_purchase_action")):
        return True
    haystack = _candidate_text(candidate).lower()
    return any(keyword.lower() in haystack for keyword in PURCHASE_KEYWORDS)


def _is_workout_add_action(candidate: Mapping[str, Any]) -> bool:
    if bool(candidate.get("is_workout_add_action")):
        return True
    return bool(candidate.get("catalog_bug_id_matches") and candidate.get("catalog_keyword_matches"))


def _page_text(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {})
    runtime_signals = observation.get("runtime_signals", {})
    return " ".join(
        str(value or "")
        for value in (
            page_state.get("page_text"),
            page_state.get("page_text_sample"),
            runtime_signals.get("page_text"),
        )
    )


def _has_modal_or_dialog(observation: Mapping[str, Any]) -> bool:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("has_modal_or_dialog")):
            return True
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and str(candidate.get("role", "")).lower() in {"dialog", "alertdialog"}:
                return True
    return False


def _has_form_signal(observation: Mapping[str, Any]) -> bool:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("has_form")):
            return True
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            tag = str(candidate.get("tag", "")).lower()
            role = str(candidate.get("role", "")).lower()
            text = _candidate_text(candidate)
            if tag in {"form", "input", "textarea", "select"} or role in {"textbox", "combobox"}:
                return True
            if any(token in text for token in ("새 운동", "운동 기록", "운동명", "저장")):
                return True
    text = _page_text(observation)
    return any(token in text for token in ("새 운동", "운동 기록", "운동명"))


def _detect_catalog_button_no_response(
    clicked_candidate: Mapping[str, Any],
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    site_profile: Mapping[str, Any],
    before_url: str,
    after_url: str,
    candidate_delta: int,
    last_action_error: bool,
    r01_no_response: bool,
    page_text_delta: int,
    modal_opened: bool,
    form_opened: bool,
    feedback_visible_after: bool,
) -> Dict[str, Any] | None:
    bug_ids = _catalog_bug_ids(clicked_candidate)
    if not bug_ids:
        return None
    expected_change = _expected_changes_for_bugs(site_profile, bug_ids)
    if expected_change and not any(change in expected_change for change in ("modal_open", "form_visible", "route_change", "toast_visible")):
        return None
    route_changed = before_url != after_url
    expected_missing = {
        "modal_open": not modal_opened,
        "form_visible": not form_opened,
        "route_change": not route_changed,
        "toast_visible": not feedback_visible_after,
    }
    no_expected_change = not expected_change or all(expected_missing.get(change, False) for change in expected_change)
    if last_action_error or not r01_no_response or not no_expected_change or page_text_delta > 30 or candidate_delta > 2:
        return None
    return {
        "type": "button-no-response",
        "confidence": 0.9,
        "evidence": {
            "clicked_text": _candidate_text(clicked_candidate),
            "clicked_bid": clicked_candidate.get("bid"),
            "clicked_data_bug_id": clicked_candidate.get("data_bug_id"),
            "catalog_bug_id_matches": bug_ids,
            "catalog_keyword_matches": clicked_candidate.get("catalog_keyword_matches", []),
            "expected_change": expected_change,
            "modal_opened": modal_opened,
            "form_opened": form_opened,
            "route_changed": route_changed,
            "toast_visible": feedback_visible_after,
            "before_url": before_url,
            "after_url": after_url,
            "page_text_delta": page_text_delta,
            "candidate_delta": candidate_delta,
            "target": _target_evidence(clicked_candidate),
        },
    }


def _catalog_bug_ids(candidate: Mapping[str, Any]) -> List[str]:
    values = candidate.get("catalog_bug_id_matches", [])
    if isinstance(values, list):
        return [str(value) for value in values if value]
    return []


def _expected_changes_for_bugs(site_profile: Mapping[str, Any], bug_ids: List[str]) -> List[str]:
    expected: List[str] = []
    for bug in site_profile.get("bugs", []) or []:
        if not isinstance(bug, Mapping) or _bug_id(bug) not in set(bug_ids):
            continue
        expected.extend(_profile_list(bug.get("expected_change")))
    return sorted({item for item in expected if item})


def _keyword_matches(text: str, keywords: List[str]) -> List[str]:
    haystack = str(text or "").lower()
    return sorted({keyword for keyword in keywords if str(keyword).lower() in haystack})


def _profile_values(site_profile: Mapping[str, Any], field: str) -> List[str]:
    return _profile_list(site_profile.get(field))


def _profile_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)] if str(value) else []


def _bug_id(bug: Mapping[str, Any]) -> str:
    return str(bug.get("bug_id") or bug.get("id") or bug.get("name") or "")


def _cart_count(observation: Mapping[str, Any]) -> int | None:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping) and bool(section.get("cart_count_detected")):
            value = section.get("cart_count")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def _cart_text(observation: Mapping[str, Any]) -> str:
    for section_name in ("runtime_signals", "page_state"):
        section = observation.get(section_name, {})
        if isinstance(section, Mapping):
            value = section.get("cart_text")
            if value is not None:
                return str(value)
    return ""


def _has_feedback_message(observation: Mapping[str, Any]) -> bool:
    page_state = observation.get("page_state", {})
    text = str(page_state.get("page_text_sample", "") or "").lower()
    return any(token.lower() in text for token in FEEDBACK_TOKENS)


def _has_meaningful_network_change(observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> bool:
    signals = observation.get("runtime_signals", {})
    for source in (action_info, signals):
        normalized = source.get("normalized_network_events") if isinstance(source, Mapping) else None
        if isinstance(normalized, list):
            if any(str(item.get("functional_relevance") or "").upper() == "RELEVANT" for item in normalized if isinstance(item, Mapping)):
                return True
            # A present normalized list is authoritative.  Framework/static
            # traffic must not satisfy a functional-response check.
            continue
        for key in ("network_changed", "network_activity", "meaningful_request", "request_count_delta"):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, (int, float)) and value > 0:
                return True
            if isinstance(value, bool) and value:
                return True
        if isinstance(source, Mapping):
            for key in ("network_entries", "network_events", "api_requests", "requests"):
                value = source.get(key)
                if isinstance(value, list) and value:
                    return True
    return False


def _has_runtime_signal_delta(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
) -> bool:
    before = before_observation.get("runtime_signals", {})
    after = after_observation.get("runtime_signals", {})
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    ignored = {
        "elapsed_time",
        "page_text",
        "page_text_length",
        "visible_texts",
        "data_bug_ids_found",
        "section_texts",
        "last_action_error",
    }
    keys = set(before.keys()).union(after.keys()) - ignored
    for key in keys:
        if before.get(key) != after.get(key):
            return True
    return False


def _error_message(action_info: Mapping[str, Any]) -> str:
    for key in ("action_error", "fallback_error", "error", "last_action_error_message"):
        value = action_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _is_mobile_viewport(observation: Mapping[str, Any]) -> bool:
    width = int(observation.get("page_state", {}).get("viewport_width", 0) or 0)
    return bool(width and width <= 768)


def _viewport_type(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {})
    value = str(page_state.get("viewport_type") or observation.get("runtime_signals", {}).get("viewport_type") or "")
    if value:
        return value
    return "mobile" if _is_mobile_viewport(observation) else "desktop"


def _previous_action_type(observation: Mapping[str, Any]) -> str:
    history = observation.get("history", {})
    if isinstance(history, Mapping):
        return str(history.get("previous_action_type") or "")
    return ""


def _low_visibility_anomaly(candidate: Mapping[str, Any], visibility: float, last_action_error: bool) -> Dict[str, Any]:
    return {
        "type": "low-visibility-interaction",
        "confidence": 0.5 if last_action_error else 0.35,
        "evidence": {
            "target": _target_evidence(candidate),
            "bid": candidate.get("bid"),
            "visibility": visibility,
            "last_action_error": last_action_error,
        },
    }


def _detect_duplicated_rendering(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    action_type: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if last_action_error:
        return None
    before_counts, before_meta = _r04_identity_counts(before_observation)
    after_counts, after_meta = _r04_identity_counts(after_observation)
    for entity_key, after_count in after_counts.items():
        before_count = before_counts.get(entity_key, 0)
        meta = after_meta.get(entity_key, {})
        if not _is_r04_duplicated_rendering(
            entity_key=entity_key,
            before_count=before_count,
            after_count=after_count,
            expected_max_count=1,
            action_succeeded=True,
            legitimate_repetition=bool(meta.get("legitimate_repetition")),
        ):
            continue
        return {
            "type": "duplicated-rendering",
            "confidence": 0.85 if str(meta.get("identity_source")) != "candidate_fingerprint" else 0.72,
            "evidence": {
                "rule_id": "R04",
                "entity_key": entity_key,
                "entity_type": meta.get("entity_type", ""),
                "before_count": before_count,
                "after_count": after_count,
                "duplicate_count": max(0, after_count - 1),
                "expected_max_count": 1,
                "identity_source": meta.get("identity_source", ""),
                "identity_text": meta.get("text", ""),
                "href": meta.get("href", ""),
                "clicked_bid": _clicked_bid(action_info),
                "action_type": action_type,
            },
        }
    return None


def _is_r04_duplicated_rendering(
    *,
    entity_key: str | None,
    before_count: int,
    after_count: int,
    expected_max_count: int = 1,
    action_succeeded: bool = True,
    legitimate_repetition: bool = False,
) -> bool:
    return bool(
        action_succeeded
        and entity_key
        and not legitimate_repetition
        and before_count <= expected_max_count
        and after_count > expected_max_count
        and after_count > before_count
    )


def _r04_identity_counts(observation: Mapping[str, Any]) -> tuple[Counter[str], Dict[str, Dict[str, Any]]]:
    counts: Counter[str] = Counter()
    meta_by_key: Dict[str, Dict[str, Any]] = {}
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return counts, meta_by_key
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not bool(candidate.get("visible", True)):
            continue
        identity = _r04_candidate_identity(candidate)
        if not identity:
            continue
        key, meta = identity
        counts[key] += 1
        meta_by_key.setdefault(key, meta)
    return counts, meta_by_key


def _r04_candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, Dict[str, Any]] | None:
    for source in ("entity_id", "product_id", "post_id", "comment_id", "item_id", "data_id", "data-id"):
        value = candidate.get(source)
        if value is not None and str(value).strip():
            text = _normalize_title(_candidate_text(candidate))
            entity_value = str(value).strip()
            entity_type = source[:-3] if source.endswith("_id") else source
            return (
                f"{source}:{entity_value}",
                {
                    "identity_source": source,
                    "entity_type": entity_type,
                    "text": text,
                    "href": _candidate_href(candidate),
                    "legitimate_repetition": False,
                },
            )
    href = _candidate_href(candidate)
    text = _normalize_title(_candidate_text(candidate))
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    if href and text and (role or tag):
        normalized_href = _normalize_navigation_url(href)
        return (
            f"candidate_fingerprint:{role}:{tag}:{normalized_href}:{text}",
            {
                "identity_source": "candidate_fingerprint",
                "entity_type": tag or role,
                "text": text,
                "href": href,
                "legitimate_repetition": _is_common_repeated_action_title(text),
            },
        )
    return None


def _clicked_bid(action_info: Mapping[str, Any]) -> str:
    action = action_info.get("action", action_info)
    if isinstance(action, Mapping):
        for key in ("clicked_bid", "bid", "target_bid"):
            value = action.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    for key in ("clicked_bid", "bid", "target_bid"):
        value = action_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_title(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if len(normalized) < 2 or len(normalized) > 120:
        return ""
    return normalized.lower()


def _is_common_repeated_action_title(title: str) -> bool:
    normalized = _normalize_title(title)
    return normalized in {
        "add",
        "add add",
        "cart",
        "search",
        "checkout",
        "details",
        "view",
        "buy",
        "submit",
        "save",
        "continue",
        "next",
        "logs",
        "action logs",
        "action logs action logs",
        "logs logs",
    }


def _overlapping_duplicate_count(candidates: List[Mapping[str, Any]]) -> int:
    boxes = [_bbox(candidate) for candidate in candidates if _bbox(candidate)]
    overlap_count = 0
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if _bbox_overlap_ratio(left, right) >= 0.8:
                overlap_count += 1
    return overlap_count


def _bbox(candidate: Mapping[str, Any]) -> List[float]:
    raw = candidate.get("bbox") or []
    if not isinstance(raw, list) or len(raw) < 4:
        return []
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    except (TypeError, ValueError):
        return []


def _viewport_bbox(observation: Mapping[str, Any]) -> List[float]:
    page_state = observation.get("page_state", {})
    width = float(page_state.get("viewport_width", 0) or 0) if isinstance(page_state, Mapping) else 0.0
    height = float(page_state.get("viewport_height", 0) or 0) if isinstance(page_state, Mapping) else 0.0
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 4096.0, 4096.0]
    return [0.0, 0.0, width, height]


def _target_in_viewport(bbox: List[float], viewport: List[float]) -> bool:
    return _bbox_intersection_area(bbox, viewport) > 0


def _vertical_scroll_reachable(
    observation: Mapping[str, Any],
    bbox: List[float],
    viewport: List[float],
) -> bool:
    if len(bbox) < 4 or len(viewport) < 4:
        return False
    page_state = observation.get("page_state", {})
    document_height = 0.0
    if isinstance(page_state, Mapping):
        for key in ("document_height", "scroll_height", "page_height", "body_scroll_height"):
            try:
                document_height = max(document_height, float(page_state.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue
    bottom = float(bbox[1]) + float(bbox[3])
    viewport_bottom = float(viewport[1]) + float(viewport[3])
    if bbox[1] >= viewport_bottom and document_height >= bottom:
        return True
    if bottom > viewport_bottom and not _has_horizontal_clipping(bbox, viewport) and document_height >= bottom:
        return True
    return False


def _has_horizontal_clipping(bbox: List[float], viewport: List[float]) -> bool:
    return bool(bbox[0] < viewport[0] - 2 or bbox[0] + bbox[2] > viewport[0] + viewport[2] + 2)


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def _bbox_intersection_area(left: List[float], right: List[float]) -> float:
    lx, ly, lw, lh = left[:4]
    rx, ry, rw, rh = right[:4]
    if lw <= 0 or lh <= 0 or rw <= 0 or rh <= 0:
        return 0.0
    overlap_w = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    overlap_h = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    return overlap_w * overlap_h


def _same_layout_element(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in ("bid", "id", "href"):
        left_value = str(left.get(key) or "")
        right_value = str(right.get(key) or "")
        if left_value and left_value == right_value:
            return True
    return False


def _parent_child_like_overlap(
    target: Mapping[str, Any],
    occluder: Mapping[str, Any],
    target_bbox: List[float],
    occluder_bbox: List[float],
) -> bool:
    if _bbox_contains(target_bbox, occluder_bbox, tolerance=2.0):
        return not _overlay_like_candidate(occluder)
    if _bbox_contains(occluder_bbox, target_bbox, tolerance=2.0):
        return _container_like_candidate(occluder) and not _overlay_like_candidate(occluder)
    return False


def _bbox_contains(container: List[float], child: List[float], tolerance: float = 0.0) -> bool:
    cx, cy, cw, ch = container[:4]
    x, y, w, h = child[:4]
    return bool(
        cx <= x + tolerance
        and cy <= y + tolerance
        and cx + cw >= x + w - tolerance
        and cy + ch >= y + h - tolerance
    )


def _container_like_candidate(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return tag in {"div", "section", "article", "main", "form", "ul", "li"} or role in {"generic", "group", "region", "list", "listitem"}


def _overlay_like_candidate(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    text = " ".join(str(candidate.get(key) or "") for key in ("text", "name", "class_name", "id", "aria_label")).lower()
    return bool(
        role in {"dialog", "alertdialog", "tooltip", "menu"}
        or tag in {"dialog"}
        or any(token in text for token in ("overlay", "modal", "dialog", "tooltip", "dropdown", "drawer", "floating", "panel", "backdrop"))
    )


def _normal_overlay_candidate(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").lower()
    text = " ".join(str(candidate.get(key) or "") for key in ("text", "name", "class_name", "id", "aria_label")).lower()
    return bool(role in {"dialog", "alertdialog", "tooltip", "menu"} or any(token in text for token in ("tooltip", "dropdown", "cookie banner")))


def _bbox_overlap_ratio(left: List[float], right: List[float]) -> float:
    lx, ly, lw, lh = left[:4]
    rx, ry, rw, rh = right[:4]
    if lw <= 0 or lh <= 0 or rw <= 0 or rh <= 0:
        return 0.0
    l2x, l2y = lx + lw, ly + lh
    r2x, r2y = rx + rw, ry + rh
    overlap_w = max(0.0, min(l2x, r2x) - max(lx, rx))
    overlap_h = max(0.0, min(l2y, r2y) - max(ly, ry))
    overlap_area = overlap_w * overlap_h
    return overlap_area / max(1.0, min(lw * lh, rw * rh))
