"""Observation-only anomaly heuristics for BrowserGym PPO training."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping


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
        if (
            is_purchase_action
            and has_cart_counts
            and cart_before == cart_after
            and before_cart_text == after_cart_text
            and before_url == after_url
            and not last_action_error
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
            page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
            no_text_change = page_text_delta <= 20
            no_candidate_change = candidate_delta <= 1
            meaningful_page_text_change = page_text_delta > 40
            meaningful_candidate_change = candidate_delta > 2
            no_feedback = not _has_feedback_message(after_observation)
            no_network_change = not _has_meaningful_network_change(after_observation, action_info)
            modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
            form_opened = _has_form_signal(after_observation) and not _has_form_signal(before_observation)

            if (
                is_purchase_action
                and no_url_change
                and no_text_change
                and no_candidate_change
                and not last_action_error
                and no_feedback
                and no_network_change
                and not modal_opened
                and not form_opened
                and not cart_state_changed
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
                and no_url_change
                and no_state_change
                and not last_action_error
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
                and no_url_change
                and no_text_change
                and no_candidate_change
                and not meaningful_page_text_change
                and not meaningful_candidate_change
                and no_feedback
                and no_network_change
                and not modal_opened
                and not form_opened
                and not cart_state_changed
                and not last_action_error
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
        )
        if catalog_no_response:
            anomalies.append(catalog_no_response)

    overlap_count = int(after_observation.get("layout_signals", {}).get("layout_overlap_count", 0) or 0)
    overflow_count = int(after_observation.get("layout_signals", {}).get("layout_overflow_count", 0) or 0)
    overflow_details = after_observation.get("layout_signals", {}).get("layout_overflow_details", []) or []
    catalog_layout_anomaly = _detect_catalog_layout_overflow(after_observation, action_type, site_profile, overflow_count, overflow_details)
    if catalog_layout_anomaly:
        anomalies.append(catalog_layout_anomaly)
    if overlap_count >= 2:
        viewport_type = _viewport_type(after_observation)
        mobile_layout_action = action_type in {"inspect_layout", "change_viewport_mobile"} or _previous_action_type(
            after_observation
        ) == "change_viewport_mobile"
        if viewport_type == "mobile" and mobile_layout_action:
            confidence = 0.9
        elif viewport_type == "mobile":
            confidence = 0.75 if overlap_count >= 5 else 0.65
        else:
            confidence = 0.4
        anomalies.append(
            {
                "type": "layout-overlap",
                "confidence": confidence,
                "evidence": {
                    "layout_overlap_count": overlap_count,
                    "viewport_width": after_observation.get("page_state", {}).get("viewport_width", 0),
                    "viewport_height": after_observation.get("page_state", {}).get("viewport_height", 0),
                    "viewport_type": viewport_type,
                    "mobile_viewport": viewport_type == "mobile",
                    "action_type": action_type,
                },
            }
        )

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
        if "403" in str(error_message).lower() or "forbidden" in str(error_message).lower():
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
        if api_403_count > 0 or "403" in network_text or "forbidden" in network_text:
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
    if "403" in page_text_after or "forbidden" in page_text_after or "access denied" in page_text_after:
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

    if action_type == "inspect_cart":
        cart_before = _cart_count(before_observation)
        cart_after = _cart_count(after_observation)
        before_cart_text = _cart_text(before_observation)
        after_cart_text = _cart_text(after_observation)
        cart_text_changed = before_cart_text != after_cart_text
        if cart_before is not None and cart_after is not None and cart_before == cart_after and cart_text_changed:
            anomalies.append(
                {
                    "type": "cart-quantity-mismatch",
                    "confidence": 0.75,
                    "evidence": {
                        "action_type": action_type,
                        "cart_count_before": cart_before,
                        "cart_count_after": cart_after,
                        "cart_text_before": before_cart_text,
                        "cart_text_after": after_cart_text,
                        "cart_quantity_evidence": True,
                    },
                }
            )

    if action_type in {"fill_input", "press_enter", "click_element"} and _has_explicit_async_or_hang_keyword(clicked_candidate or {}):
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

    broken_tokens = ("404", "error", "not-found", "not_found", "page-not-found")
    title = str(after_observation.get("page_state", {}).get("title", "")).lower()
    url = after_url.lower()
    if any(token in title or token in url for token in broken_tokens):
        anomalies.append(
            {
                "type": "broken-navigation",
                "confidence": 0.85,
                "evidence": {"url": after_url, "title": title},
            }
        )

    duplicate_anomaly = _detect_duplicated_rendering(after_observation, site_profile)
    if duplicate_anomaly:
        if _is_openended_profile(site_profile):
            duplicate_anomaly.update(
                {
                    "classification": "exploratory_anomaly",
                    "human_review_status": "needs_review",
                    "review_question": "Duplicated text may come from the accessibility tree; verify whether the same component is duplicated in the visible UI.",
                }
            )
        anomalies.append(duplicate_anomaly)

    empty_state_anomaly = _detect_catalog_empty_state(after_observation, site_profile)
    if empty_state_anomaly:
        anomalies.append(empty_state_anomaly)

    return anomalies


def compute_anomaly_score(anomalies: List[Dict[str, Any]]) -> float:
    if not anomalies:
        return 0.0
    return min(1.0, sum(float(item.get("confidence", 0.0) or 0.0) for item in anomalies) / len(anomalies))


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


def _page_text_length(observation: Mapping[str, Any]) -> int:
    return int(observation.get("page_state", {}).get("page_text_length", 0) or 0)


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


def _target_evidence(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bid": candidate.get("bid"),
        "data_bug_id": candidate.get("data_bug_id"),
        "name": candidate.get("name"),
        "text": candidate.get("text"),
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
) -> Dict[str, Any] | None:
    bug_ids = _catalog_bug_ids(clicked_candidate)
    if not bug_ids:
        return None
    expected_change = _expected_changes_for_bugs(site_profile, bug_ids)
    if expected_change and not any(change in expected_change for change in ("modal_open", "form_visible", "route_change", "toast_visible")):
        return None
    page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
    modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
    form_opened = _has_form_signal(after_observation) and not _has_form_signal(before_observation)
    route_changed = before_url != after_url
    toast_visible = _has_feedback_message(after_observation)
    expected_missing = {
        "modal_open": not modal_opened,
        "form_visible": not form_opened,
        "route_change": not route_changed,
        "toast_visible": not toast_visible,
    }
    no_expected_change = not expected_change or all(expected_missing.get(change, False) for change in expected_change)
    if last_action_error or not no_expected_change or page_text_delta > 30 or candidate_delta > 2:
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
            "toast_visible": toast_visible,
            "before_url": before_url,
            "after_url": after_url,
            "page_text_delta": page_text_delta,
            "candidate_delta": candidate_delta,
            "target": _target_evidence(clicked_candidate),
        },
    }


def _detect_catalog_empty_state(
    observation: Mapping[str, Any],
    site_profile: Mapping[str, Any],
) -> Dict[str, Any] | None:
    bugs = [
        bug for bug in site_profile.get("bugs", []) or []
        if isinstance(bug, Mapping) and str(bug.get("type") or "") in {"empty-state-rendering", "component-rendering"}
    ]
    if not bugs:
        return None
    page_text = _page_text(observation)
    section_matches = _keyword_matches(page_text, _profile_values(site_profile, "section_keywords"))
    negative_matches = _keyword_matches(page_text, _profile_values(site_profile, "negative_keywords"))
    expected_elements = _profile_values(site_profile, "expected_elements")
    missing_expected = _missing_expected_elements(observation, expected_elements)
    if section_matches and negative_matches and not missing_expected:
        page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
        if bool(page_state.get("has_empty_state_text")):
            missing_expected = [
                item for item in expected_elements if str(item).lower() in {"chart", "chart-container", "bar", "svg", "canvas"}
            ]
    if not (section_matches and negative_matches and missing_expected):
        return None
    catalog_bug_ids = [_bug_id(bug) for bug in bugs if _bug_id(bug)]
    return {
        "type": "empty-state-rendering",
        "confidence": 0.8,
        "evidence": {
            "section_keyword_matches": section_matches,
            "negative_keyword_matches": negative_matches,
            "missing_expected_elements": missing_expected,
            "chart_visible": not any(item in {"chart", "chart-container", "bar", "svg", "canvas"} for item in missing_expected),
            "catalog_bug_id_matches": catalog_bug_ids,
            "page_text_sample": page_text[:800],
        },
    }


def _detect_catalog_layout_overflow(
    observation: Mapping[str, Any],
    action_type: str,
    site_profile: Mapping[str, Any],
    overflow_count: int,
    overflow_details: Any,
) -> Dict[str, Any] | None:
    if action_type != "inspect_layout":
        return None
    layout_bugs = [
        bug for bug in site_profile.get("bugs", []) or []
        if isinstance(bug, Mapping) and str(bug.get("layout_check") or "") == "child_overflows_parent"
    ]
    if not layout_bugs:
        return None
    signals = observation.get("layout_signals", {}) if isinstance(observation.get("layout_signals"), Mapping) else {}
    details = overflow_details if isinstance(overflow_details, list) else []
    candidates = signals.get("layout_overflow_candidates", []) if isinstance(signals.get("layout_overflow_candidates"), list) else []
    detail = _best_layout_detail(details, layout_bugs) or _best_layout_detail(candidates, layout_bugs) or _catalog_layout_fallback(observation, layout_bugs)
    if not detail and overflow_count <= 0 and not candidates:
        return None
    detail = detail or {}
    page_text = _page_text(observation)
    detail_text = " ".join(str(detail.get(key) or "") for key in ("text", "selector", "selector_hint", "data_bug_id"))
    section_keywords = _bug_values(layout_bugs, "section_keywords")
    target_keywords = _bug_values(layout_bugs, "target_keywords")
    expected_elements = _bug_values(layout_bugs, "expected_elements")
    section_matches = _keyword_matches(page_text, section_keywords)
    target_matches = sorted(set(_keyword_matches(detail_text, target_keywords) + _keyword_matches(page_text, target_keywords)))
    detail_bug_ids = _profile_list(detail.get("catalog_bug_id_matches"))
    layout_bug_ids = [_bug_id(bug) for bug in layout_bugs if _bug_id(bug)]
    detail_bug_ids = [bug_id for bug_id in detail_bug_ids if bug_id in set(layout_bug_ids)]
    if not detail_bug_ids:
        detail_bug_ids = layout_bug_ids
    overflow_right = bool(detail.get("overflow_right", overflow_count > 0))
    overflow_bottom = bool(detail.get("overflow_bottom", False))
    child_bbox = detail.get("child_bbox", [])
    parent_bbox = detail.get("parent_bbox", [])
    bbox_source = str(detail.get("bbox_source") or "")
    has_child_bbox = isinstance(child_bbox, list) and len(child_bbox) >= 4 and any(float(value or 0) for value in child_bbox[:4])
    has_parent_bbox = isinstance(parent_bbox, list) and len(parent_bbox) >= 4 and any(float(value or 0) for value in parent_bbox[:4])
    if not bbox_source:
        bbox_source = "actual_bbox" if has_child_bbox and has_parent_bbox else "viewport_fallback"
    confidence = 0.9 if (overflow_right or overflow_bottom) and has_child_bbox else 0.72 if bbox_source.endswith("fallback") else 0.75
    return {
        "type": "layout-overflow",
        "confidence": confidence,
        "evidence": {
            "site_id": site_profile.get("site_id") or "",
            "candidate_text": detail.get("text", ""),
            "data_bug_id": detail.get("data_bug_id") or "",
            "selector": detail.get("selector") or detail.get("selector_hint") or "",
            "selector_hint": detail.get("selector") or detail.get("selector_hint") or "",
            "catalog_bug_id_matches": detail_bug_ids,
            "section_keyword_matches": section_matches,
            "target_keyword_matches": target_matches,
            "expected_elements": expected_elements,
            "layout_check": "child_overflows_parent",
            "child_bbox": child_bbox,
            "parent_bbox": parent_bbox,
            "bbox_source": bbox_source,
            "overflow_right": overflow_right,
            "overflow_bottom": overflow_bottom,
            "layout_overflow_count": overflow_count,
            "viewport_width": observation.get("page_state", {}).get("viewport_width", 0),
            "viewport_height": observation.get("page_state", {}).get("viewport_height", 0),
        },
    }


def _best_layout_detail(details: List[Any], layout_bugs: List[Mapping[str, Any]]) -> Dict[str, Any] | None:
    layout_bug_ids = {_bug_id(bug) for bug in layout_bugs if _bug_id(bug)}
    best: Dict[str, Any] | None = None
    best_score = -1.0
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        detail_ids = set(_profile_list(detail.get("catalog_bug_id_matches")))
        text = " ".join(str(detail.get(key) or "") for key in ("text", "selector", "selector_hint", "data_bug_id")).lower()
        score = 0.0
        if detail_ids.intersection(layout_bug_ids):
            score += 5.0
        if bool(detail.get("catalog_selector_match")):
            score += 3.0
        if bool(detail.get("overflow_right")) or bool(detail.get("overflow_bottom")):
            score += 2.0
        if bool(detail.get("is_layout_target")):
            score += 1.0
        if bool(detail.get("is_chart_like")):
            score += 0.5
        for bug in layout_bugs:
            if _keyword_matches(text, _profile_list(bug.get("target_keywords")) + _profile_list(bug.get("expected_elements"))):
                score += 0.5
        if score > best_score:
            best_score = score
            best = dict(detail)
    return best


def _catalog_layout_fallback(observation: Mapping[str, Any], layout_bugs: List[Mapping[str, Any]]) -> Dict[str, Any] | None:
    viewport_width = float(observation.get("page_state", {}).get("viewport_width", 0) or 0)
    viewport_height = float(observation.get("page_state", {}).get("viewport_height", 0) or 0)
    candidates = observation.get("candidate_elements", []) or []
    if not isinstance(candidates, list):
        return None
    target_values = _bug_values(layout_bugs, "target_keywords") + _bug_values(layout_bugs, "expected_elements")
    section_values = _bug_values(layout_bugs, "section_keywords")
    fallback: Dict[str, Any] | None = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_text = " ".join(
            str(candidate.get(key) or "")
            for key in ("text", "name", "class_name", "id", "tag", "role", "selector_hint", "data_bug_id")
        )
        candidate_keyword_matches = {str(item).lower() for item in candidate.get("catalog_keyword_matches", []) or []}
        layout_target_matches = bool(candidate_keyword_matches.intersection({str(item).lower() for item in target_values}))
        if not (candidate.get("is_layout_target") or layout_target_matches or _keyword_matches(candidate_text, target_values)):
            continue
        bbox = candidate.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        right = float(bbox[0]) + float(bbox[2])
        bottom = float(bbox[1]) + float(bbox[3])
        overflow_right = bool(viewport_width and right > viewport_width + 6)
        overflow_bottom = bool(viewport_height and bottom > viewport_height + 6)
        detail = {
            "selector": candidate.get("selector_hint") or candidate.get("data_bug_id") or "",
            "data_bug_id": candidate.get("data_bug_id") or "",
            "catalog_bug_id_matches": candidate.get("catalog_bug_id_matches", []),
            "text": candidate_text[:240],
            "child_bbox": bbox,
            "parent_bbox": [0.0, 0.0, viewport_width, viewport_height],
            "bbox_source": "candidate_viewport_parent",
            "overflow_right": overflow_right,
            "overflow_bottom": overflow_bottom,
        }
        if overflow_right or overflow_bottom:
            return detail
        fallback = fallback or detail
    page_text = _page_text(observation)
    section_matches = _keyword_matches(page_text, section_values)
    if section_matches and (bool(observation.get("page_state", {}).get("has_chart_like_element")) or _keyword_matches(page_text, target_values)):
        layout_bug_ids = [_bug_id(bug) for bug in layout_bugs if _bug_id(bug)]
        return {
            "selector": "",
            "data_bug_id": "",
            "catalog_bug_id_matches": layout_bug_ids,
            "text": " ".join(section_matches),
            "child_bbox": [],
            "parent_bbox": [0.0, 0.0, viewport_width, viewport_height],
            "bbox_source": "viewport_fallback",
            "overflow_right": True,
            "overflow_bottom": False,
        }
    return fallback


def _bug_values(bugs: List[Mapping[str, Any]], field: str) -> List[str]:
    values: List[str] = []
    for bug in bugs:
        values.extend(_profile_list(bug.get(field)))
    return sorted({value for value in values if value})


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


def _missing_expected_elements(observation: Mapping[str, Any], expected_elements: List[str]) -> List[str]:
    if not expected_elements:
        return []
    haystack = _observation_element_text(observation)
    missing = []
    for element in expected_elements:
        element_text = str(element or "").lower()
        if not element_text:
            continue
        if element_text in {"chart", "chart-container", "bar", "svg", "canvas"}:
            if not _chart_like_visible(observation):
                missing.append(element_text)
        elif element_text not in haystack:
            missing.append(element_text)
    return sorted(set(missing))


def _chart_like_visible(observation: Mapping[str, Any]) -> bool:
    page_state = observation.get("page_state", {})
    if isinstance(page_state, Mapping) and bool(page_state.get("has_chart_like_element")):
        return True
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        return any(isinstance(candidate, Mapping) and bool(candidate.get("is_chart_related")) for candidate in candidates)
    return False


def _observation_element_text(observation: Mapping[str, Any]) -> str:
    parts = [_page_text(observation)]
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                parts.append(_candidate_text(candidate))
                parts.append(str(candidate.get("tag") or ""))
                parts.append(str(candidate.get("class_name") or ""))
                parts.append(str(candidate.get("data_bug_id") or ""))
    return " ".join(parts).lower()


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
        for key in ("network_changed", "network_activity", "meaningful_request", "request_count_delta"):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, (int, float)) and value > 0:
                return True
            if isinstance(value, bool) and value:
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
    observation: Mapping[str, Any],
    site_profile: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    candidates = observation.get("candidate_elements", []) or []
    titles = []
    candidates_by_title: Dict[str, List[Mapping[str, Any]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        text = _normalize_title(_candidate_text(candidate))
        if text:
            titles.append(text)
            candidates_by_title.setdefault(text, []).append(candidate)
    counts = Counter(titles)
    duplicated = {title: count for title, count in counts.items() if count >= 2}
    if not duplicated:
        return None
    visible_overlapping_duplicates = {
        title: _overlapping_duplicate_count(candidates_by_title.get(title, []))
        for title in duplicated
    }
    visible_overlapping_duplicates = {
        title: count for title, count in visible_overlapping_duplicates.items() if count > 0
    }
    common_repeated_titles = {title for title in duplicated if _is_common_repeated_action_title(title)}
    if common_repeated_titles and common_repeated_titles == set(duplicated) and not visible_overlapping_duplicates:
        return None
    page_text = str(observation.get("page_state", {}).get("page_text_sample", "") or "").lower()
    has_book_context = any(token.lower() in page_text for token in BOOK_CONTEXT_TOKENS)
    profile_bug_types = {
        str(item).lower()
        for item in (site_profile or {}).get("bug_types", []) or []
    }
    has_duplicate_catalog = bool(
        {"duplicated-rendering", "component-rendering"}.intersection(profile_bug_types)
    )
    if not has_duplicate_catalog and not visible_overlapping_duplicates:
        return None
    confidence = 0.55 + (0.15 if has_book_context else 0.0)
    if visible_overlapping_duplicates:
        confidence += 0.2
    elif common_repeated_titles and not has_book_context:
        confidence -= 0.25
    if max(duplicated.values()) >= 3:
        confidence += 0.1
    confidence = min(0.9, confidence)
    if confidence < 0.6:
        return None
    return {
        "type": "duplicated-rendering",
        "confidence": confidence,
        "evidence": {
            "duplicated_titles": duplicated,
            "visible_duplicate_candidates": visible_overlapping_duplicates,
            "book_context": has_book_context,
            "axtree_or_text_only": not bool(visible_overlapping_duplicates),
        },
    }


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
