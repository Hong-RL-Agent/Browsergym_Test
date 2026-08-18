"""Observation-only anomaly heuristics for BrowserGym PPO training."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Mapping

from services.extended_policy_oracle_service import detect_extended_policy_findings
from services.web_error_finding_service import detect_web_error_findings


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
HIGH_VALUE_SEMANTIC_ACTION_TYPES = {
    "workout_add",
    "cart",
    "add",
    "submit",
    "save",
    "login",
    "purchase",
    "enroll",
}
NON_EXECUTING_SEMANTIC_ACTION_TYPES = {"filter", "category", "tab", "search_input"}
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
FORUM_MORE_TEXT = "\ub354\ubcf4\uae30"
FORUM_COMMUNITY_TEXT = "\ucee4\ubba4\ub2c8\ud2f0"
FORUM_SEARCH_PLACEHOLDER = "\uc81c\ubaa9, \uc791\uc131\uc790, \uce74\ud14c\uace0\ub9ac\ub85c \uac80\uc0c9"
FORUM_BUG_TYPES = {
    "forum-comment-duplicated",
    "forum-comment-delete-failed",
    "forum-post-detail-not-opened",
    "forum-save-feedback-missing",
    "forum-empty-post-validation-missing",
}
# Deliberately narrow to tokens that are distinctly forum/community vocabulary.
# The previous list also included "post", "comment", and generic Korean UI
# words (title/content/save/delete) that show up on ordinary e-commerce pages
# too (a cart's "\uc0ad\uc81c" button, a review section's "comment"/"\ub313\uae00 X\uac1c") -- on a
# site with no explicit bug catalog those alone were enough to misclassify an
# e-commerce page as forum context and fire forum-* detectors on it.
FORUM_SURFACE_TOKENS = (
    "forumworks",
    "forum",
    "\uac8c\uc2dc\ud310",  # \uac8c\uc2dc\ud310
    "\ucee4\ubba4\ub2c8\ud2f0",  # \ucee4\ubba4\ub2c8\ud2f0
    "\uac8c\uc2dc\uae00",  # \uac8c\uc2dc\uae00
    "\uae00\uc4f0\uae30",  # \uae00\uc4f0\uae30
    "\ub313\uae00",  # \ub313\uae00
)


def detect_anomalies(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    site_profile: Mapping[str, Any] | None = None,
    history: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    # history is the same per-episode dict the training/eval loop already
    # threads through the reward and opportunity systems. It's optional here
    # (defaults to a throwaway dict) because most anomaly types only need the
    # immediate before/after diff -- but a few genuinely can't be judged from
    # one transition alone (a saved value only turns out stale once the page
    # is revisited later; a duplicate booking only shows up by comparing two
    # separate submit attempts), so those detectors read/write into it across
    # calls within the same episode.
    if history is None:
        history = {}
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
    visibility = _visibility(clicked_candidate)
    if action_type == "click_element" and clicked_candidate:
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
            semantic_type = str(clicked_candidate.get("semantic_action_type") or "")
            anomalies.append(
                {
                    "type": "button-no-response",
                    "confidence": 0.95,
                    "classification": "exploratory_anomaly",
                    "evidence": {
                        "clicked_bid": clicked_candidate.get("bid"),
                        "clicked_text": target_text,
                        "clicked_name": clicked_candidate.get("name"),
                        "semantic_action_type": semantic_type,
                        "functional_priority_candidate": bool(
                            clicked_candidate.get("functional_priority_candidate")
                            or clicked_candidate.get("functional_priority")
                        ),
                        "high_value_functional_candidate": bool(clicked_candidate.get("is_high_value_functional_candidate")),
                        "semantic_no_effect_click": bool(semantic_type),
                        "functional_no_effect_anomaly": bool(semantic_type),
                        "workout_add_no_effect": semantic_type == "workout_add",
                        "cart_no_effect": semantic_type == "cart",
                        "add_no_effect": semantic_type == "add",
                        "enroll_no_effect": semantic_type == "enroll",
                        "high_value_enroll_no_response": semantic_type == "enroll",
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
            feedback_visible_before = _has_feedback_message(before_observation)
            feedback_visible_after = _has_feedback_message(after_observation)
            no_feedback = not (feedback_visible_after and not feedback_visible_before)
            no_network_change = not _has_meaningful_network_change(after_observation, action_info)
            no_runtime_signal_delta = not _has_runtime_signal_delta(after_observation, action_info)
            modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
            form_opened = _has_form_signal(after_observation) and not _has_form_signal(before_observation)
            semantic_no_effect_emitted = False
            semantic_type = str(clicked_candidate.get("semantic_action_type") or "")

            if (
                is_purchase_action
                and semantic_type in {"cart", "add"}
                and before_url == after_url
                and not last_action_error
                and no_feedback
                and not modal_opened
                and not form_opened
                and not cart_state_changed
            ):
                anomalies.append(
                    {
                        "type": "button-no-response",
                        "confidence": 0.84,
                        "classification": "exploratory_anomaly",
                        **_human_review_metadata("button-no-response", clicked_candidate, before_url, after_url, route_changed=False),
                        "evidence": {
                            "clicked_text": target_text,
                            "clicked_name": clicked_candidate.get("name"),
                            "clicked_bid": clicked_candidate.get("bid"),
                            "semantic_action_type": semantic_type,
                            "semantic_no_effect_click": True,
                            "functional_no_effect_anomaly": True,
                            "cart_no_effect": semantic_type == "cart",
                            "add_no_effect": semantic_type == "add",
                            "before_url": before_url,
                            "after_url": after_url,
                            "page_text_delta": page_text_delta,
                            "candidate_delta": candidate_delta,
                            "cart_count_before": cart_before,
                            "cart_count_after": cart_after,
                            "cart_text_before": before_cart_text,
                            "cart_text_after": after_cart_text,
                            "cart_count_detected": has_cart_counts,
                            "cart_state_changed": cart_state_changed,
                            "toast_visible": False,
                            "network_request_delta": not no_network_change,
                            "runtime_signal_delta": not no_runtime_signal_delta,
                            "target": _target_evidence(clicked_candidate),
                            "is_purchase_action": True,
                        },
                    }
                )
                semantic_no_effect_emitted = True

            if not semantic_no_effect_emitted and _is_high_value_semantic_action(clicked_candidate) and _is_semantic_no_effect_click(
                before_observation=before_observation,
                after_observation=after_observation,
                before_url=before_url,
                after_url=after_url,
                candidate_delta=candidate_delta,
                page_text_delta=page_text_delta,
                no_text_change=no_text_change,
                no_candidate_change=no_candidate_change,
                no_feedback=no_feedback,
                no_network_change=no_network_change,
                no_runtime_signal_delta=no_runtime_signal_delta,
                modal_opened=modal_opened,
                form_opened=form_opened,
                cart_state_changed=cart_state_changed,
                last_action_error=last_action_error,
            ):
                semantic_type = str(clicked_candidate.get("semantic_action_type") or "")
                anomalies.append(
                    {
                        "type": "button-no-response",
                        "confidence": 0.82 if semantic_type == "workout_add" else 0.76,
                        "classification": "exploratory_anomaly",
                        **_human_review_metadata("button-no-response", clicked_candidate, before_url, after_url, route_changed=False),
                        "evidence": {
                            "clicked_text": target_text,
                            "clicked_name": clicked_candidate.get("name"),
                            "clicked_bid": clicked_candidate.get("bid"),
                            "clicked_role": clicked_candidate.get("role"),
                            "clicked_tag": clicked_candidate.get("tag"),
                            "semantic_action_type": semantic_type,
                            "functional_priority_candidate": bool(
                                clicked_candidate.get("functional_priority_candidate")
                                or clicked_candidate.get("functional_priority")
                            ),
                            "high_value_functional_candidate": bool(clicked_candidate.get("is_high_value_functional_candidate")),
                            "semantic_no_effect_click": True,
                            "functional_no_effect_anomaly": True,
                            "workout_add_no_effect": semantic_type == "workout_add",
                            "cart_no_effect": semantic_type == "cart",
                            "add_no_effect": semantic_type == "add",
                            "enroll_no_effect": semantic_type == "enroll",
                            "high_value_enroll_no_response": semantic_type == "enroll",
                            "before_url": before_url,
                            "after_url": after_url,
                            "page_text_delta": page_text_delta,
                            "candidate_delta": candidate_delta,
                            "modal_opened": modal_opened,
                            "form_opened": form_opened,
                            "route_changed": False,
                            "cart_state_changed": cart_state_changed,
                            "toast_visible": not no_feedback,
                            "network_request_delta": not no_network_change,
                            "runtime_signal_delta": not no_runtime_signal_delta,
                            "target": _target_evidence(clicked_candidate),
                        },
                    }
                )
                semantic_no_effect_emitted = True

            if (
                not semantic_no_effect_emitted
                and is_purchase_action
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
                semantic_type = str(clicked_candidate.get("semantic_action_type") or "")
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
                            "semantic_action_type": semantic_type,
                            "functional_priority_candidate": bool(
                                clicked_candidate.get("functional_priority_candidate")
                                or clicked_candidate.get("functional_priority")
                            ),
                            "high_value_functional_candidate": bool(clicked_candidate.get("is_high_value_functional_candidate")),
                            "semantic_no_effect_click": bool(semantic_type),
                            "functional_no_effect_anomaly": bool(semantic_type),
                            "workout_add_no_effect": semantic_type == "workout_add",
                            "cart_no_effect": semantic_type == "cart",
                            "add_no_effect": semantic_type == "add",
                            "enroll_no_effect": semantic_type == "enroll",
                            "high_value_enroll_no_response": semantic_type == "enroll",
                            "target": _target_evidence(clicked_candidate),
                            "is_purchase_action": True,
                        },
                    }
                )
            elif (
                not semantic_no_effect_emitted
                and not is_purchase_action
                and not _is_openended_target(clicked_candidate)
                and no_url_change
                and no_state_change
                and page_text_delta == 0
                and candidate_delta == 0
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
                not semantic_no_effect_emitted
                and _is_openended_target(clicked_candidate)
                and no_url_change
                and page_text_delta == 0
                and candidate_delta == 0
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
    elif action_type in {"press_enter", "submit_form"} and clicked_candidate:
        page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
        candidate_delta = abs(len(after_candidates) - len(before_candidates))
        no_feedback = not (
            _has_feedback_message(after_observation) and not _has_feedback_message(before_observation)
        )
        no_network_change = not _has_meaningful_network_change(after_observation, action_info)
        no_runtime_signal_delta = not _has_runtime_signal_delta(after_observation, action_info)
        modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
        form_opened = _has_form_signal(after_observation) and not _has_form_signal(before_observation)
        if (
            _is_search_input_candidate(clicked_candidate)
            and before_url == after_url
            and page_text_delta == 0
            and candidate_delta == 0
            and no_feedback
            and no_network_change
            and no_runtime_signal_delta
            and not modal_opened
            and not form_opened
            and not last_action_error
        ):
            anomalies.append(
                {
                    "type": "form-no-feedback",
                    "confidence": 0.58,
                    "classification": "exploratory_anomaly",
                    **_human_review_metadata("form-no-feedback", clicked_candidate, before_url, after_url, route_changed=False),
                    "evidence": {
                        "action_type": action_type,
                        "clicked_text": _candidate_text(clicked_candidate),
                        "clicked_name": clicked_candidate.get("name"),
                        "clicked_bid": clicked_candidate.get("bid"),
                        "clicked_role": clicked_candidate.get("role"),
                        "clicked_tag": clicked_candidate.get("tag"),
                        "semantic_action_type": "search_input",
                        "search_input_no_effect": True,
                        "functional_no_effect_anomaly": True,
                        "before_url": before_url,
                        "after_url": after_url,
                        "page_text_delta": page_text_delta,
                        "candidate_delta": candidate_delta,
                        "modal_opened": modal_opened,
                        "form_opened": form_opened,
                        "toast_visible": not no_feedback,
                        "network_request_delta": not no_network_change,
                        "runtime_signal_delta": not no_runtime_signal_delta,
                        "target": _target_evidence(clicked_candidate),
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
        layout_bugs = [
            bug for bug in site_profile.get("bugs", []) or []
            if isinstance(bug, Mapping) and str(bug.get("type") or "") in {"layout-overlap", "layout-overflow", "css-layout"}
        ]
        layout_detail = _best_layout_detail(after_candidates, layout_bugs) or _catalog_layout_fallback(after_observation, layout_bugs)
        layout_detail = layout_detail or {}
        detail_text = " ".join(str(layout_detail.get(key) or "") for key in ("text", "selector", "selector_hint", "data_bug_id"))
        target_keywords = _bug_values(layout_bugs, "target_keywords")
        target_matches = sorted(set(_keyword_matches(detail_text, target_keywords) + _keyword_matches(_page_text(after_observation), target_keywords)))
        specific_element_identified = bool(layout_detail)
        # layout_overlap_count only ever proved *how many* candidate pairs
        # intersect, not *which two* -- evidence used to report a single
        # "candidate_text"/"selector" (whichever element the catalog/priority
        # heuristics picked, e.g. the header cart icon) with no indication of
        # what it actually overlapped with. Surface the actual overlapping
        # pair (largest-area overlap, or the one matching target_keywords if a
        # catalog hint is available) so a reviewer can tell "price label
        # overlaps add-to-cart button" from "two unrelated header elements
        # happen to intersect."
        overlap_details_list = after_observation.get("layout_signals", {}).get("layout_overlap_details", []) or []
        best_overlap_pair: Dict[str, Any] = {}
        if target_keywords:
            for pair in overlap_details_list:
                combined = " ".join(
                    str(pair.get(key) or "")
                    for key in ("element1_text", "element1_selector", "element2_text", "element2_selector")
                )
                if _keyword_matches(combined, target_keywords):
                    best_overlap_pair = pair
                    break
        if not best_overlap_pair and overlap_details_list:
            best_overlap_pair = overlap_details_list[0]
        anomalies.append(
            {
                "type": "layout-overlap",
                "confidence": confidence,
                "classification": "exploratory_anomaly",
                "human_review_status": "needs_review",
                "ground_truth_match_status": "unconfirmed_layout_signal",
                "review_question": "Overlap count alone does not prove a specific overflow bug; verify the affected element and whether it overflows or overlaps.",
                "evidence": {
                    "layout_overlap_count": overlap_count,
                    "viewport_width": after_observation.get("page_state", {}).get("viewport_width", 0),
                    "viewport_height": after_observation.get("page_state", {}).get("viewport_height", 0),
                    "viewport_type": viewport_type,
                    "mobile_viewport": viewport_type == "mobile",
                    "action_type": action_type,
                    "specific_element_identified": specific_element_identified,
                    "candidate_text": layout_detail.get("text", ""),
                    "selector": layout_detail.get("selector") or layout_detail.get("selector_hint") or "",
                    "selector_hint": layout_detail.get("selector") or layout_detail.get("selector_hint") or "",
                    "data_bug_id": layout_detail.get("data_bug_id") or "",
                    "catalog_bug_id_matches": _profile_list(layout_detail.get("catalog_bug_id_matches")),
                    "target_keyword_matches": target_matches,
                    "overlapping_elements_identified": bool(best_overlap_pair),
                    "element1_text": best_overlap_pair.get("element1_text", ""),
                    "element1_selector": best_overlap_pair.get("element1_selector", ""),
                    "element2_text": best_overlap_pair.get("element2_text", ""),
                    "element2_selector": best_overlap_pair.get("element2_selector", ""),
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
                "confidence": 0.55,
                "classification": "action_execution_diagnostic",
                "human_review_status": "needs_verification",
                "evidence": {
                    "action_type": action_type,
                    "error": error_message,
                    "action_result_failed": action_result_failed,
                    "failure_reason": error_message or "action executor reported failure",
                    "infra_failure_possible": True,
                    "user_visible_failure_possible": False,
                    "requires_followup_verification": True,
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
    anomalies.extend(detect_web_error_findings(before_observation, after_observation, action_info))
    anomalies.extend(detect_extended_policy_findings(before_observation, after_observation, action_info))
    quantity_anomaly = _detect_invalid_quantity_allowed(
        before_observation,
        after_observation,
        action_info,
        clicked_candidate,
        action_type,
    )
    if quantity_anomaly:
        anomalies.append(quantity_anomaly)
    total_anomaly = _detect_cart_total_mismatch(after_observation, action_type)
    if total_anomaly:
        anomalies.append(total_anomaly)
    redirect_anomaly = _detect_login_redirect_mismatch(before_observation, after_observation, action_type)
    if redirect_anomaly:
        anomalies.append(redirect_anomaly)
    filter_anomaly = _detect_filter_no_effect(before_observation, after_observation, clicked_candidate, action_type, candidate_delta)
    if filter_anomaly:
        anomalies.append(filter_anomaly)
    delete_js_error_anomaly = _detect_js_error_on_destructive_action(before_observation, after_observation, clicked_candidate, action_type)
    if delete_js_error_anomaly:
        anomalies.append(delete_js_error_anomaly)
    stale_data_anomaly = _detect_stale_data_rendering(before_observation, after_observation, action_info, clicked_candidate, action_type, history)
    if stale_data_anomaly:
        anomalies.append(stale_data_anomaly)
    double_submit_anomaly = _detect_double_submit(before_observation, after_observation, action_info, clicked_candidate, action_type, history)
    if double_submit_anomaly:
        anomalies.append(double_submit_anomaly)
    weak_password_anomaly = _detect_weak_password_validation(before_observation, after_observation, action_info, clicked_candidate, action_type, history)
    if weak_password_anomaly:
        anomalies.append(weak_password_anomaly)
    detail_nav_anomaly = _detect_detail_navigation_no_effect(
        before_observation,
        after_observation,
        clicked_candidate,
        action_type,
        before_url,
        after_url,
        candidate_delta,
        last_action_error,
    )
    if detail_nav_anomaly:
        anomalies.append(detail_nav_anomaly)
    detail_mismatch_anomaly = _detect_product_detail_mismatch(
        before_observation,
        after_observation,
        clicked_candidate,
        action_type,
        before_url,
        after_url,
        last_action_error,
    )
    if detail_mismatch_anomaly:
        anomalies.append(detail_mismatch_anomaly)
    anomalies.extend(_detect_forum_findings(before_observation, after_observation, action_info, clicked_candidate, action_type, before_url, after_url, candidate_delta, last_action_error, site_profile))
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
        # Deliberately low confidence: this is a bare substring match against
        # whatever text happens to be visible on the page (which can include
        # unrelated content that merely mentions "forbidden"/"403"), with no
        # HTTP status code or request evidence behind it -- unlike the
        # network-entry-based api-forbidden finding above, which has real
        # response.status == 403 evidence. _is_reportable_anomaly requires
        # confidence >= 0.5 for this type, so this stays a low-confidence
        # diagnostic signal rather than an independently reportable finding.
        anomalies.append(
            {
                "type": "api-forbidden",
                "confidence": 0.35,
                "classification": "exploratory_anomaly",
                "human_review_status": "needs_verification",
                "evidence": {
                    "before_url": before_url,
                    "after_url": after_url,
                    "page_text_contains_forbidden": True,
                    "http_status_evidence": False,
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


def _dom_node_count(observation: Mapping[str, Any]) -> int:
    page_state = observation.get("page_state", {})
    runtime = observation.get("runtime_signals", {})
    for source in (page_state, runtime):
        if not isinstance(source, Mapping):
            continue
        try:
            return int(source.get("dom_node_count") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _candidate_at(candidates: Any, index: int) -> Mapping[str, Any] | None:
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _visibility(candidate: Mapping[str, Any] | None) -> float:
    if not isinstance(candidate, Mapping):
        return 1.0
    if "visible" in candidate and not bool(candidate.get("visible", True)):
        return 0.0
    if "visibility" not in candidate or candidate.get("visibility") is None:
        return 1.0
    try:
        value = float(candidate.get("visibility") or 0.0)
    except (TypeError, ValueError):
        return 1.0
    if value != value:
        return 1.0
    return max(0.0, min(1.0, value))


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
        "semantic_action_type": candidate.get("semantic_action_type"),
        "functional_priority_candidate": candidate.get("functional_priority_candidate") or candidate.get("functional_priority"),
        "is_high_value_functional_candidate": candidate.get("is_high_value_functional_candidate"),
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


def _is_high_value_semantic_action(candidate: Mapping[str, Any]) -> bool:
    semantic_type = str(candidate.get("semantic_action_type") or "")
    return bool(
        semantic_type in HIGH_VALUE_SEMANTIC_ACTION_TYPES
        and (candidate.get("functional_priority_candidate") or candidate.get("functional_priority") or candidate.get("is_high_value_functional_candidate"))
    )


def _is_search_input_candidate(candidate: Mapping[str, Any]) -> bool:
    semantic_type = str(candidate.get("semantic_action_type") or "")
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    input_type = str(candidate.get("input_type") or "").lower()
    return bool(
        semantic_type == "search_input"
        or candidate.get("is_search_related")
        or role in {"searchbox", "textbox"}
        or input_type == "search"
        or (tag in {"input", "textarea"} and "search" in _candidate_text(candidate).lower())
    )


def _is_semantic_no_effect_click(
    *,
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    before_url: str,
    after_url: str,
    candidate_delta: int,
    page_text_delta: int,
    no_text_change: bool,
    no_candidate_change: bool,
    no_feedback: bool,
    no_network_change: bool,
    no_runtime_signal_delta: bool,
    modal_opened: bool,
    form_opened: bool,
    cart_state_changed: bool,
    last_action_error: bool,
) -> bool:
    dom_delta = abs(_dom_node_count(after_observation) - _dom_node_count(before_observation))
    return bool(
        before_url == after_url
        and dom_delta <= 1
        and page_text_delta == 0
        and candidate_delta == 0
        and no_text_change
        and no_candidate_change
        and no_feedback
        and no_network_change
        and no_runtime_signal_delta
        and not modal_opened
        and not form_opened
        and not cart_state_changed
        and not last_action_error
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
    if _is_low_value_generic_candidate(candidate):
        return "button-no-response", 0.2
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
    if anomaly_type == "button-no-response" and _is_low_value_generic_candidate(candidate):
        return False
    if anomaly_type == "button-no-response" and _is_forum_noise_candidate(candidate):
        return False
    if anomaly_type == "button-no-response" and str(candidate.get("semantic_action_type") or "") in NON_EXECUTING_SEMANTIC_ACTION_TYPES:
        return False
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


def _is_low_value_generic_candidate(candidate: Mapping[str, Any]) -> bool:
    if bool(candidate.get("is_low_value_generic_candidate")):
        return True
    role = str(candidate.get("role") or "").lower()
    semantic_type = str(candidate.get("semantic_action_type") or "")
    functional = bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority"))
    high_value = bool(candidate.get("is_high_value_functional_candidate"))
    text = _candidate_text(candidate).strip().lower()
    return bool(
        role == "generic"
        and not semantic_type
        and not functional
        and not high_value
        and re.fullmatch(r"element-\d+", text or "")
    )


def _is_forum_noise_candidate(candidate: Mapping[str, Any]) -> bool:
    text = " ".join(_candidate_text(candidate).split()).strip().lower()
    return text in {
        FORUM_MORE_TEXT,
        f"{FORUM_MORE_TEXT} {FORUM_MORE_TEXT}",
        "more",
        "forumworks",
        f"forumworks {FORUM_COMMUNITY_TEXT}",
        f"forumworks {FORUM_COMMUNITY_TEXT} forumworks{FORUM_COMMUNITY_TEXT}",
        FORUM_COMMUNITY_TEXT,
    } or FORUM_SEARCH_PLACEHOLDER in text


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


def _detect_invalid_quantity_allowed(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
) -> Dict[str, Any] | None:
    if action_type not in {"fill_input", "press_enter", "click_element"}:
        return None
    target = candidate or {}
    target_text = _candidate_text(target)
    is_quantity_target = bool(
        target.get("is_quantity_control")
        # is_cart_quantity_related is deliberately NOT trusted here: the
        # observation adapter sets it on any element whose text merely
        # contains "cart"/"장바구니" (see COMMERCE priority-boost usage), which
        # includes a plain header cart *icon* with no quantity control at all
        # -- confirmed on a real scan where clicking the header cart button
        # fired this detector purely because its label contained "장바구니".
        or any(
            token in target_text
            for token in (
                "quantity", "qty", "수량", "subtotal", "total", "합계", "소계", "총액", "plus", "minus", "증가", "감소",
                # booking/reservation headcount uses the same "is this number
                # valid" shape as cart quantity (negative/zero/over-limit
                # accepted without validation), just different vocabulary.
                "인원", "인원수", "guest", "guests", "capacity", "참가", "참가자", "명",
            )
        )
        or target_text.strip() in {"+", "-", "＋", "－"}
    )
    action = action_info.get("action", action_info)
    input_text = " ".join(
        str(action.get(key) or action_info.get(key) or "")
        for key in ("input_value", "value", "text", "typed_text", "fill_value")
    )
    after_text_raw = " ".join([_cart_text(after_observation), _page_text(after_observation)])
    after_text = after_text_raw.lower()
    invalid_input = _contains_invalid_quantity(input_text)
    invalid_visible = _contains_invalid_quantity(after_text)
    capacity_exceeded = _capacity_exceeded(after_text_raw)
    if not is_quantity_target:
        return None
    if action_type == "click_element" and _is_low_value_generic_candidate(target):
        return None
    if not (invalid_input or invalid_visible or capacity_exceeded):
        return None
    if _state_signature(before_observation) == _state_signature(after_observation) and not (invalid_visible or capacity_exceeded):
        return None
    before_cart_text = _cart_text(before_observation)
    after_cart_text = _cart_text(after_observation)
    quantity_before = _extract_quantity_value(before_cart_text)
    quantity_after = _extract_quantity_value(after_cart_text)
    total_amounts = _currency_amounts(after_cart_text)
    expected_total = sum(total_amounts[:-1]) if len(total_amounts) >= 2 else None
    actual_total = total_amounts[-1] if total_amounts else None
    return {
        "type": "cart-quantity-mismatch",
        "confidence": 0.86 if (invalid_visible or capacity_exceeded) else 0.74,
        "classification": "exploratory_anomaly",
        "human_review_status": "likely_true_positive" if (invalid_visible or capacity_exceeded) else "needs_review",
        "evidence": {
            "action_type": action_type,
            "invalid_quantity_input": input_text.strip(),
            "invalid_quantity_visible": invalid_visible,
            "capacity_exceeded": capacity_exceeded,
            "quantity_before": quantity_before,
            "quantity_after": quantity_after,
            "expected_total": expected_total,
            "actual_total": actual_total,
            "cart_text_before": before_cart_text,
            "cart_text_after": after_cart_text,
            "target": _target_evidence(target) if target else {},
        },
    }


def _capacity_exceeded(text: str) -> bool:
    # Booking/reservation forms often display their own declared limit ("최대
    # 4명" / "max 10 guests") right next to the headcount control -- if the
    # accepted value exceeds that stated maximum, the validation is missing,
    # the same class of bug as a negative/zero cart quantity just phrased as
    # an upper bound instead of a lower one.
    lowered = str(text or "").lower()
    max_match = re.search(r"(?:최대|max(?:imum)?)\D{0,6}(\d+)", lowered)
    if not max_match:
        return False
    try:
        max_value = int(max_match.group(1))
    except ValueError:
        return False
    if max_value <= 0:
        return False
    for match in re.finditer(r"(?:인원수?|guests?|참가자?|명|quantity|qty|수량)\D{0,6}(\d+)", lowered):
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        if value > max_value:
            return True
    return False


def _extract_quantity_value(text: str) -> float | None:
    value = str(text or "").lower()
    patterns = (
        "(?:quantity|qty|\\uc218\\ub7c9|guests?)\\s*[:=]?\\s*(-?\\d+)",
        # A bare "N\uac1c"/"N items" is only trustworthy as a *cart* quantity when a
        # cart/quantity keyword sits close by -- matched with zero required
        # context, it grabs unrelated page copy just as readily (e.g. a "6\uac1c
        # \uc0c1\ud488" recommended-products counter), which is exactly what produced a
        # false cart-quantity-mismatch reading (quantity 6 -> 0) from a product
        # *listing* count on a real scan, not any actual cart state.
        "(?:cart|basket|\\uc7a5\\ubc14\\uad6c\\ub2c8|\\uce74\\ud2b8|\\uc218\\ub7c9)[^\\d-]{0,20}(-?\\d+)\\s*(?:items?|pcs?|ea|\\uac1c)?\\b",
        "(-?\\d+)\\s*(?:items?|pcs?|ea|\\uac1c)?[^\\d]{0,20}(?:\\uc7a5\\ubc14\\uad6c\\ub2c8|\\uce74\\ud2b8|cart|basket|\\uc218\\ub7c9)",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _contains_invalid_quantity(text: str) -> bool:
    value = str(text or "").lower()
    if not value:
        return False
    if re.search("(?:quantity|qty|\\uc218\\ub7c9)\\s*[:=]?\\s*(-\\d+|0)\\b", value):
        return True
    # Same proximity requirement as _extract_quantity_value above: a negative
    # number or a bare 0 only counts as an "invalid quantity visible" signal
    # when it sits next to an actual cart/quantity keyword, not merely
    # somewhere in the same broad cart_text blob as one.
    if re.search(
        "(?:cart|basket|\\uc7a5\\ubc14\\uad6c\\ub2c8|\\uce74\\ud2b8|\\uc218\\ub7c9)[^\\d-]{0,20}(-\\d+|0)\\s*(?:items?|pcs?|ea|\\uac1c)?\\b",
        value,
    ):
        return True
    if re.search(
        "(-\\d+|0)\\s*(?:items?|pcs?|ea|\\uac1c)?[^\\d]{0,20}(?:\\uc7a5\\ubc14\\uad6c\\ub2c8|\\uce74\\ud2b8|cart|basket|\\uc218\\ub7c9)",
        value,
    ):
        return True
    return bool(
        re.search("\\b(?:nan|null|undefined)\\b", value)
        and any(token in value for token in ("quantity", "qty", "\uc218\ub7c9", "cart", "basket", "\uc7a5\ubc14\uad6c\ub2c8"))
    )


def _detect_cart_total_mismatch(observation: Mapping[str, Any], action_type: str) -> Dict[str, Any] | None:
    if action_type not in {"inspect_cart", "inspect_dom", "click_element", "fill_input", "press_enter"}:
        return None
    cart_text = _cart_text(observation)
    text = (cart_text or _page_text(observation)).lower()
    # The underlying check (sum of visible line items vs. a displayed total)
    # is domain-agnostic -- it applies just as well to a booking headcount
    # summary or a stats dashboard's totals row as to a shopping cart, so the
    # trigger vocabulary isn't restricted to commerce-specific words.
    if not any(
        token in text
        for token in ("cart", "basket", "subtotal", "total", "장바구니", "카트", "소계", "합계", "총액", "통계", "stats", "statistics", "예약", "인원")
    ):
        return None
    amounts = _currency_amounts(text)
    if len(amounts) < 3:
        return None
    if len(amounts) > 8:
        return None
    if not cart_text and action_type not in {"inspect_cart", "inspect_dom"}:
        return None
    displayed_total = amounts[-1]
    expected_total = sum(amounts[:-1])
    if expected_total <= 0:
        return None
    if abs(displayed_total - expected_total) <= max(1.0, expected_total * 0.02):
        return None
    return {
        "type": "cart-total-mismatch",
        "confidence": 0.78,
        "classification": "exploratory_anomaly",
        "human_review_status": "likely_true_positive",
        "evidence": {
            "action_type": action_type,
            "line_item_amounts": amounts[:-1],
            "displayed_total": displayed_total,
            "expected_total_from_visible_amounts": expected_total,
            "cart_text": _cart_text(observation),
        },
    }


def _detect_login_redirect_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_type: str,
) -> Dict[str, Any] | None:
    if action_type not in {"click_submit", "submit_form", "press_enter", "click_element", "login_as_user", "login_as_admin"}:
        return None
    before_runtime = before_observation.get("runtime_signals", {}) if isinstance(before_observation, Mapping) else {}
    after_runtime = after_observation.get("runtime_signals", {}) if isinstance(after_observation, Mapping) else {}
    if not isinstance(before_runtime, Mapping):
        before_runtime = {}
    if not isinstance(after_runtime, Mapping):
        after_runtime = {}
    was_logged_in = bool(before_runtime.get("logged_in") or before_runtime.get("login_success"))
    is_logged_in = bool(after_runtime.get("logged_in") or after_runtime.get("login_success"))
    if not is_logged_in or was_logged_in:
        return None
    before_url = _url(before_observation)
    after_url = _url(after_observation)
    after_path = after_url.lower()
    still_on_auth_page = any(token in after_path for token in ("login", "signin", "sign-in", "auth", "logon"))
    no_navigation = before_url == after_url
    if not (still_on_auth_page or no_navigation):
        return None
    return {
        "type": "login-redirect-mismatch",
        "confidence": 0.7 if still_on_auth_page else 0.55,
        "classification": "exploratory_anomaly",
        "human_review_status": "needs_review",
        "evidence": {
            "before_url": before_url,
            "after_url": after_url,
            "still_on_auth_page": still_on_auth_page,
            "no_navigation_despite_login_success": no_navigation,
        },
    }


def _detect_filter_no_effect(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    candidate_delta: int,
) -> Dict[str, Any] | None:
    if action_type not in {"click_element", "select_option", "fill_input", "press_enter"}:
        return None
    target = candidate or {}
    if not target:
        return None
    semantic_type = str(target.get("semantic_action_type") or "").lower()
    if semantic_type == "search_input" and action_type == "click_element":
        return None
    is_filter_target = bool(
        target.get("is_filter_related")
        or semantic_type in {"filter", "category", "tab", "sort"}
        or any(token in _candidate_text(target).lower() for token in ("filter", "sort", "필터", "정렬", "카테고리"))
    )
    if not is_filter_target or _is_low_value_generic_candidate(target):
        return None
    page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
    no_state_change = _state_signature(before_observation) == _state_signature(after_observation)
    if not (no_state_change and candidate_delta == 0 and page_text_delta <= 5):
        return None
    return {
        "type": "filter-no-effect",
        "confidence": 0.72,
        "classification": "exploratory_anomaly",
        "human_review_status": "needs_review",
        "evidence": {
            "action_type": action_type,
            "target": _target_evidence(target),
            "candidate_delta": candidate_delta,
            "page_text_delta": page_text_delta,
        },
    }


def _detect_js_error_on_destructive_action(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
) -> Dict[str, Any] | None:
    if action_type != "click_element" or not candidate:
        return None
    target_text = _candidate_text(candidate).lower()
    is_destructive = any(token in target_text for token in ("삭제", "delete", "remove", "취소", "cancel"))
    if not is_destructive:
        return None
    before_runtime = before_observation.get("runtime_signals", {}) if isinstance(before_observation, Mapping) else {}
    after_runtime = after_observation.get("runtime_signals", {}) if isinstance(after_observation, Mapping) else {}
    if not isinstance(before_runtime, Mapping):
        before_runtime = {}
    if not isinstance(after_runtime, Mapping):
        after_runtime = {}

    def _count(source: Mapping[str, Any]) -> int:
        total = 0
        for key in ("playwright_console_error_count", "console_error_count", "page_error_count", "runtime_exception_count"):
            try:
                total += int(source.get(key) or 0)
            except (TypeError, ValueError):
                continue
        return total

    before_errors = _count(before_runtime)
    after_errors = _count(after_runtime)
    if after_errors <= before_errors:
        return None
    return {
        "type": "runtime-exception",
        "confidence": 0.82,
        "classification": "exploratory_anomaly",
        "human_review_status": "likely_true_positive",
        "evidence": {
            "action_type": action_type,
            "target": _target_evidence(candidate),
            "console_error_count_before": before_errors,
            "console_error_count_after": after_errors,
            "correlated_destructive_action": True,
            "destructive_action_text": _candidate_text(candidate).strip(),
        },
    }


def _field_identity_key(candidate: Mapping[str, Any]) -> str:
    # Prefer identifiers that survive a page reload/revisit (a bid usually
    # doesn't); fall back to visible text only as a last resort.
    for key in ("name", "aria_label", "placeholder", "label", "id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value.lower()
    text = _candidate_text(candidate).strip()
    return text.lower() if text else ""


def _detect_stale_data_rendering(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    history: Dict[str, Any],
) -> Dict[str, Any] | None:
    # Needs memory across steps: a saved value only turns out stale once the
    # field is seen again *after* a save was confirmed, not on the very next
    # transition (which would just be comparing the field to itself).
    entered = history.setdefault("entered_field_values", {})
    action = action_info.get("action", action_info)
    target = candidate or {}

    if action_type == "fill_input" and target:
        field_key = _field_identity_key(target)
        typed_value = _first_action_text(action, action_info)
        if field_key and typed_value:
            entered[field_key] = {"value": typed_value, "confirmed_saved": False, "resolved": False}
        return None

    if action_type in {"click_element", "click_submit", "submit_form"} and target and _is_forum_save_target(target):
        for record in entered.values():
            if isinstance(record, dict) and not record.get("confirmed_saved"):
                record["confirmed_saved"] = True
        return None

    candidates = after_observation.get("candidate_elements", []) if isinstance(after_observation, Mapping) else []
    if not isinstance(candidates, list):
        return None
    for other in candidates:
        if not isinstance(other, Mapping):
            continue
        field_key = _field_identity_key(other)
        record = entered.get(field_key) if field_key else None
        if not isinstance(record, dict) or not record.get("confirmed_saved") or record.get("resolved"):
            continue
        displayed_value = str(other.get("value") or other.get("input_value") or "").strip()
        if not displayed_value:
            continue
        record["resolved"] = True
        entered_value = str(record.get("value") or "").strip()
        if displayed_value != entered_value:
            return {
                "type": "stale-data-rendering",
                "confidence": 0.75,
                "classification": "exploratory_anomaly",
                "human_review_status": "needs_review",
                "evidence": {
                    "field": field_key,
                    "entered_value": entered_value,
                    "displayed_value": displayed_value,
                    "target": _target_evidence(other),
                },
            }
    return None


def _detect_double_submit(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    history: Dict[str, Any],
) -> Dict[str, Any] | None:
    if action_type not in {"click_element", "click_submit", "submit_form", "press_enter"}:
        return None
    target = candidate or {}
    if not target:
        return None
    text = _candidate_text(target).lower()
    is_submit_like = bool(
        target.get("is_submit")
        or any(token in text for token in ("예약", "reserve", "booking", "신청", "등록", "제출", "submit", "book now", "확정"))
    )
    if not is_submit_like:
        return None
    action = action_info.get("action", action_info)
    signature = str(action.get("action_signature") or "") or _field_identity_key(target) or text
    if not signature:
        return None
    # _state_signature is structural (DOM shape/candidate count) and won't
    # change for a booking counter that just increments its own text, so use
    # a content-level diff instead to confirm this click actually did
    # something rather than counting a no-effect click as a submission.
    no_content_change = (
        _state_signature(before_observation) == _state_signature(after_observation)
        and _page_text(before_observation) == _page_text(after_observation)
    )
    if no_content_change:
        # This click didn't actually do anything -- don't count it as a
        # successful (duplicate-creating) submission.
        return None
    counts = history.setdefault("submit_click_success_counts", {})
    counts[signature] = int(counts.get(signature, 0) or 0) + 1
    if counts[signature] < 2:
        return None
    return {
        "type": "duplicate-submission",
        "confidence": 0.7,
        "classification": "exploratory_anomaly",
        "human_review_status": "needs_review",
        "evidence": {
            "action_signature": signature,
            "submit_success_count_this_episode": counts[signature],
            "target": _target_evidence(target),
        },
    }


def _detect_weak_password_validation(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    history: Dict[str, Any],
) -> Dict[str, Any] | None:
    # Blind (no ground-truth credentials) so there's no way to know a
    # password is "wrong" from a single attempt. The only signal available
    # without a real password oracle is internal consistency: the same
    # account accepting two different passwords in the same episode is
    # unambiguously broken regardless of which (if either) was "correct".
    action = action_info.get("action", action_info)
    target = candidate or {}
    if action_type == "fill_input" and target:
        typed_value = _first_action_text(action, action_info)
        if typed_value:
            input_type = str(target.get("type") or target.get("input_type") or "").lower()
            role = str(target.get("role") or "").lower()
            if bool(target.get("is_password")) or input_type == "password":
                history["last_typed_password"] = typed_value
            elif input_type == "email" or "email" in (str(target.get("name") or "") + str(target.get("placeholder") or "")).lower():
                history["last_typed_email"] = typed_value.strip().lower()
        return None
    if action_type not in {"click_submit", "submit_form", "press_enter", "click_element"}:
        return None
    after_runtime = after_observation.get("runtime_signals", {}) if isinstance(after_observation, Mapping) else {}
    before_runtime = before_observation.get("runtime_signals", {}) if isinstance(before_observation, Mapping) else {}
    if not isinstance(after_runtime, Mapping):
        after_runtime = {}
    if not isinstance(before_runtime, Mapping):
        before_runtime = {}
    was_logged_in = bool(before_runtime.get("logged_in") or before_runtime.get("login_success"))
    is_logged_in = bool(after_runtime.get("logged_in") or after_runtime.get("login_success"))
    if not is_logged_in or was_logged_in:
        return None
    email = str(history.get("last_typed_email") or "").strip()
    password = str(history.get("last_typed_password") or "").strip()
    if not email or not password:
        return None
    attempts = history.setdefault("login_attempts_by_email", {})
    seen_passwords = attempts.setdefault(email, set())
    seen_passwords.add(password)
    if len(seen_passwords) < 2:
        return None
    return {
        "type": "weak-password-validation",
        "confidence": 0.8,
        "classification": "exploratory_anomaly",
        "human_review_status": "likely_true_positive",
        "evidence": {
            "email": email,
            "distinct_passwords_accepted_this_episode": len(seen_passwords),
            "password_length": len(password),
        },
    }


def _currency_amounts(text: str) -> List[float]:
    amounts: List[float] = []
    for match in re.finditer(r"(?:[$₩]\s*|krw\s*)?(-?\d[\d,]*(?:\.\d{1,2})?)\s*(?:원|usd|krw)?", str(text or ""), re.IGNORECASE):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if abs(value) >= 1:
            amounts.append(value)
    return amounts


def _currency_amounts(text: str) -> List[float]:
    amounts: List[float] = []
    value = str(text or "")
    patterns = (
        r"(?:[$\u20a9]\s*|krw\s+|usd\s+)(-?\d[\d,]*(?:\.\d{1,2})?)",
        r"(-?\d[\d,]*(?:\.\d{1,2})?)\s*(?:\uc6d0|won|usd|krw)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value, re.IGNORECASE):
            raw = match.group(1).replace(",", "")
            try:
                amount = float(raw)
            except ValueError:
                continue
            if abs(amount) >= 1:
                amounts.append(amount)
    if amounts:
        return amounts
    for match in re.finditer(r"\b(?:price|subtotal|total|amount)\D{0,12}(-?\d[\d,]*(?:\.\d{1,2})?)", value, re.IGNORECASE):
        raw = match.group(1).replace(",", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        if abs(amount) >= 1:
            amounts.append(amount)
    return amounts


def _detect_product_detail_mismatch(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    before_url: str,
    after_url: str,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if action_type != "click_element" or not candidate or last_action_error:
        return None
    target_text = _candidate_text(candidate)
    target_product = _product_name_from_text(target_text)
    if not target_product:
        return None
    role = str(candidate.get("role") or "").lower()
    href = str(candidate.get("href") or "").strip()
    semantic_type = str(candidate.get("semantic_action_type") or "").lower()
    detail_target = bool(candidate.get("is_detail_trigger")) or bool(href) or role == "link" or semantic_type in {"detail", "details", "product_detail", "navigation"}
    if not detail_target:
        return None
    after_text = _page_text(after_observation)
    actual_product = _product_name_from_text(after_text[:500])
    if not actual_product:
        return None
    if target_product == actual_product:
        return None
    before_text = _page_text(before_observation)
    if target_product in _product_tokens(before_text) and actual_product in _product_tokens(before_text) and target_product not in _product_tokens(after_text[:300]):
        confidence = 0.88 if before_url != after_url or "detail" in after_url.lower() or "product" in after_url.lower() else 0.76
        return {
            "type": "product-detail-mismatch",
            "confidence": confidence,
            "classification": "exploratory_anomaly",
            "human_review_status": "likely_true_positive" if confidence >= 0.8 else "needs_review",
            "evidence": {
                "before_url": before_url,
                "after_url": after_url,
                "clicked_text": target_text,
                "expected_product": target_product,
                "actual_product": actual_product,
                "detail_content_mismatch": True,
                "target": _target_evidence(candidate),
            },
        }
    return None


def _product_name_from_text(text: str) -> str:
    tokens = _product_tokens(text)
    return tokens[0] if tokens else ""


def _product_tokens(text: str) -> List[str]:
    lowered = " ".join(str(text or "").lower().split())
    products = (
        "wireless headphones",
        "smart watch",
        "smartwatch",
        "headphones",
        "watch",
        "\ubb34\uc120 \ud5e4\ub4dc\ud3f0",
        "\ud5e4\ub4dc\ud3f0",
        "\uc2a4\ub9c8\ud2b8\uc6cc\uce58",
    )
    return [product for product in products if product in lowered]


def _detect_detail_navigation_no_effect(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    before_url: str,
    after_url: str,
    candidate_delta: int,
    last_action_error: bool,
) -> Dict[str, Any] | None:
    if action_type != "click_element" or not candidate or last_action_error:
        return None
    target_text = _candidate_text(candidate)
    role = str(candidate.get("role") or "").lower()
    href = str(candidate.get("href") or "").strip()
    semantic_type = str(candidate.get("semantic_action_type") or "").lower()
    expects_navigation = bool(href) or role == "link" or semantic_type in {"detail", "details", "product_detail", "navigation"}
    if not expects_navigation:
        return None
    detail_target = bool(candidate.get("is_detail_trigger")) or any(
        token in target_text
        for token in ("detail", "details", "product", "상세", "상품", "제품", "자세히")
    )
    if not detail_target:
        return None
    page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
    modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
    if before_url == after_url and page_text_delta <= 20 and candidate_delta <= 1 and not modal_opened:
        return {
            "type": "broken-navigation",
            "confidence": 0.76,
            "classification": "exploratory_anomaly",
            "evidence": {
                "before_url": before_url,
                "after_url": after_url,
                "page_text_delta": page_text_delta,
                "candidate_delta": candidate_delta,
                "target": _target_evidence(candidate),
                "detail_click_no_effect": True,
            },
        }
    return None


def _detect_forum_findings(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    action_type: str,
    before_url: str,
    after_url: str,
    candidate_delta: int,
    last_action_error: bool,
    site_profile: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not _forum_context_allowed(before_observation, site_profile) and not _forum_context_allowed(after_observation, site_profile):
        return []
    target = candidate or {}
    target_text = _candidate_text(target).lower()
    before_text = _forum_visible_text(before_observation)
    after_text = _forum_visible_text(after_observation)
    findings: List[Dict[str, Any]] = []
    action = action_info.get("action", action_info)
    input_text = _first_action_text(action, action_info)

    if action_type == "click_element" and _is_forum_post_detail_target(target) and not last_action_error:
        page_text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
        modal_opened = _has_modal_or_dialog(after_observation) and not _has_modal_or_dialog(before_observation)
        if before_url == after_url and page_text_delta <= 20 and candidate_delta <= 1 and not modal_opened:
            findings.append(
                {
                    "type": "forum-post-detail-not-opened",
                    "confidence": 0.82,
                    "classification": "exploratory_anomaly",
                    "human_review_status": "likely_true_positive",
                    "ground_truth_match_status": "site003-post-detail-open-failure",
                    "evidence": {
                        "target": _target_evidence(target),
                        "before_url": before_url,
                        "after_url": after_url,
                        "page_text_delta": page_text_delta,
                        "candidate_delta": candidate_delta,
                        "modal_opened": modal_opened,
                    },
                }
            )

    if action_type in {"click_element", "submit_form", "press_enter"} and _is_forum_comment_submit_target(target):
        comment_text = _last_meaningful_input(input_text, before_observation)
        if comment_text:
            before_count = _visible_text_count(before_text, comment_text)
            after_count = _visible_text_count(after_text, comment_text)
            if after_count >= before_count + 2:
                findings.append(
                    {
                        "type": "forum-comment-duplicated",
                        "confidence": 0.94,
                        "classification": "exploratory_anomaly",
                        "human_review_status": "likely_true_positive",
                        "ground_truth_match_status": "site003-comment-duplicate",
                        "evidence": {
                            "comment_text": comment_text,
                            "comment_count_before": before_count,
                            "comment_count_after": after_count,
                            "target": _target_evidence(target),
                        },
                    }
                )

    if action_type == "click_element" and _is_forum_delete_target(target):
        comment_text = _deleted_forum_comment_text(target, action_info, before_observation)
        before_count = _visible_text_count(before_text, comment_text) if comment_text else 0
        after_count = _visible_text_count(after_text, comment_text) if comment_text else 0
        has_body_evidence = bool(comment_text and before_count > 0)
        no_state_delta = before_url == after_url and _state_signature(before_observation) == _state_signature(after_observation)
        if (has_body_evidence and after_count >= before_count) or (not has_body_evidence and no_state_delta and not _has_feedback_message(after_observation)):
            confidence = 0.88 if has_body_evidence else 0.49
            findings.append(
                {
                    "type": "forum-comment-delete-failed",
                    "confidence": confidence,
                    "classification": "exploratory_anomaly",
                    "human_review_status": "likely_true_positive" if has_body_evidence else "needs_review",
                    "ground_truth_match_status": "site003-comment-delete-failure" if has_body_evidence else "requires_deleted_comment_body_evidence",
                    "evidence": {
                        "comment_text": comment_text,
                        "comment_text_count_before": before_count,
                        "comment_text_count_after": after_count,
                        "comment_body_evidence": has_body_evidence,
                        "delete_button_text": _candidate_text(target).strip(),
                        "target": _target_evidence(target),
                    },
                }
            )

    if action_type in {"click_element", "submit_form", "press_enter"} and _is_forum_save_target(target):
        post_count_before = _forum_post_count(before_observation)
        post_count_after = _forum_post_count(after_observation)
        empty_title_or_content = _forum_empty_required_fields(before_observation)
        success_after = _success_message_visible(after_observation)
        validation_after = _validation_message_visible(after_observation)
        if post_count_after > post_count_before and empty_title_or_content and not validation_after:
            findings.append(
                {
                    "type": "forum-empty-post-validation-missing",
                    "confidence": 0.93,
                    "classification": "exploratory_anomaly",
                    "human_review_status": "likely_true_positive",
                    "ground_truth_match_status": "site003-empty-post-validation",
                    "evidence": {
                        "post_count_before": post_count_before,
                        "post_count_after": post_count_after,
                        "empty_required_fields": empty_title_or_content,
                        "validation_message_visible": validation_after,
                        "target": _target_evidence(target),
                    },
                }
            )
        elif not success_after and not validation_after and before_url == after_url:
            findings.append(
                {
                    "type": "forum-save-feedback-missing",
                    "confidence": 0.82,
                    "classification": "exploratory_anomaly",
                    "human_review_status": "likely_true_positive",
                    "ground_truth_match_status": "site003-save-feedback",
                    "evidence": {
                        "success_message_visible": success_after,
                        "validation_message_visible": validation_after,
                        "before_url": before_url,
                        "after_url": after_url,
                        "target": _target_evidence(target),
                    },
                }
            )

    return findings


def _has_forum_surface(observation: Mapping[str, Any]) -> bool:
    text = " ".join([_page_text(observation), _cart_text(observation)]).lower()
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        text += " " + " ".join(_candidate_text(candidate) for candidate in candidates if isinstance(candidate, Mapping)).lower()
    return any(token in text for token in FORUM_SURFACE_TOKENS)


def _forum_context_allowed(observation: Mapping[str, Any], site_profile: Mapping[str, Any] | None) -> bool:
    profile_bug_types = {
        str(item).lower()
        for item in (site_profile or {}).get("bug_types", []) or []
    }
    if profile_bug_types:
        # The site profile explicitly declares which bug types this site was
        # built with (e.g. SITE001 is a commerce catalog with no forum types
        # at all) -- trust that over guessing from page text, which is how a
        # commerce site's own "댓글"/"삭제" vocabulary (product review counts,
        # a cart line's remove button) previously got misclassified as a
        # forum page and triggered forum-post-detail-not-opened on SITE001.
        return bool(FORUM_BUG_TYPES.intersection(profile_bug_types))
    # No catalog configured (an arbitrary external site) -- fall back to the
    # narrower keyword heuristic.
    return _has_forum_surface(observation)


def _is_forum_post_detail_target(candidate: Mapping[str, Any]) -> bool:
    if _is_forum_noise_candidate(candidate):
        return False
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    if bool(candidate.get("fillable")) or role in {"textbox", "searchbox"} or tag in {"input", "textarea", "select"}:
        return False
    text = _candidate_text(candidate).lower()
    if _is_forum_comment_submit_target(candidate) or _is_forum_save_target(candidate) or _is_forum_delete_target(candidate):
        return False
    if any(token in text for token in ("\uae00\uc4f0\uae30", "\uac80\uc0c9", FORUM_MORE_TEXT, "forumworks")):
        return False
    return len(text.strip()) >= 8 and bool(candidate.get("clickable") or str(candidate.get("role") or "").lower() in CLICKABLE_ROLES)


def _is_forum_comment_submit_target(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate).lower()
    return any(token in text for token in ("\ub313\uae00 \uc791\uc131", "\ub313\uae00 \ub4f1\ub85d", "comment", "post comment"))


def _is_forum_delete_target(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate).lower()
    return any(token in text for token in ("\uc0ad\uc81c", "delete", "remove"))


def _is_forum_save_target(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_text(candidate).lower()
    return any(token in text for token in ("\uc800\uc7a5", "\ub4f1\ub85d", "\uc791\uc131 \uc644\ub8cc", "submit", "save"))


def _last_meaningful_input(input_text: str, observation: Mapping[str, Any]) -> str:
    value = str(input_text or "").strip()
    if value:
        return value
    return str(observation.get("runtime_signals", {}).get("last_input_text") or "").strip()


def _first_action_text(action: Mapping[str, Any], action_info: Mapping[str, Any]) -> str:
    for key in ("input_value", "value", "typed_text", "fill_value", "text"):
        value = action.get(key)
        if value:
            return str(value).strip()
        value = action_info.get(key)
        if value:
            return str(value).strip()
    return ""


def _visible_text_count(text: str, needle: str) -> int:
    needle = str(needle or "").strip()
    if not needle:
        return 0
    return str(text or "").count(needle)


def _deleted_forum_comment_text(candidate: Mapping[str, Any], action_info: Mapping[str, Any], observation: Mapping[str, Any] | None = None) -> str:
    action = action_info.get("action", action_info)
    for source in (action, action_info, candidate):
        if not isinstance(source, Mapping):
            continue
        for key in ("comment_text", "deleted_comment_text", "target_comment_text", "related_comment_text", "context_text"):
            value = str(source.get(key) or "").strip()
            if value and not _is_forum_delete_label(value):
                return value
    # Nothing upstream (observation adapter / action adapter) ever attaches those
    # keys, so the lookup above is dead in practice. Fall back to finding the
    # candidate positionally nearest the delete button (same row/card in a
    # typical comment-list layout) whose own text isn't just another
    # delete/save/submit control label -- that's the actual comment body being
    # deleted, not the button that deletes it.
    if isinstance(observation, Mapping):
        nearby = _nearest_forum_comment_body_text(candidate, observation)
        if nearby:
            return nearby
    return ""


def _nearest_forum_comment_body_text(target: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
    target_bbox = _bbox(target)
    if not target_bbox:
        return ""
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if not isinstance(candidates, list):
        return ""
    target_x, target_y, target_w, target_h = target_bbox
    target_center_x = target_x + target_w / 2.0
    target_center_y = target_y + target_h / 2.0
    best_text = ""
    best_distance: float | None = None
    for other in candidates:
        if not isinstance(other, Mapping) or other is target:
            continue
        text = " ".join(_candidate_text(other).split()).strip()
        if len(text) < 4 or _is_forum_delete_label(text):
            continue
        if _is_forum_delete_target(other) or _is_forum_comment_submit_target(other) or _is_forum_save_target(other) or _is_forum_noise_candidate(other):
            continue
        bbox = _bbox(other)
        if not bbox:
            continue
        center_x = bbox[0] + bbox[2] / 2.0
        center_y = bbox[1] + bbox[3] / 2.0
        # Same comment card/row only: close vertically, and not further below
        # than the delete button (the comment body sits above or beside its
        # own delete control, not several comments away).
        if abs(center_y - target_center_y) > 120:
            continue
        distance = ((center_x - target_center_x) ** 2 + (center_y - target_center_y) ** 2) ** 0.5
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_text = text
    return best_text


def _is_forum_delete_label(value: str) -> bool:
    text = " ".join(str(value or "").split()).strip().lower()
    return not text or text in {"\uc0ad\uc81c", f"\ub313\uae00 \uc0ad\uc81c", "delete", "remove", "comment delete"}


def _forum_visible_text(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {})
    if isinstance(page_state, Mapping):
        return str(page_state.get("page_text") or page_state.get("page_text_sample") or "")
    return _page_text(observation)


def _forum_post_count(observation: Mapping[str, Any]) -> int:
    candidates = observation.get("candidate_elements", []) or []
    count = 0
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and _is_forum_post_detail_target(candidate):
                count += 1
    if count:
        return count
    text = _page_text(observation)
    return max(0, len(re.findall(r"(?:운영|토론|공유|공지|post|게시글)", text, re.IGNORECASE)))


def _forum_empty_required_fields(observation: Mapping[str, Any]) -> List[str]:
    empty: List[str] = []
    candidates = observation.get("candidate_elements", []) or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            text = _candidate_text(candidate).lower()
            value = str(candidate.get("value") or candidate.get("input_value") or "").strip()
            if not value and any(token in text for token in ("\uac8c\uc2dc\uae00 \uc81c\ubaa9", "\uc81c\ubaa9", "\ub0b4\uc6a9", "title", "content")):
                empty.append(str(candidate.get("name") or candidate.get("text") or "field"))
    return empty[:4]


def _success_message_visible(observation: Mapping[str, Any]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in ("\uc800\uc7a5\ub418\uc5c8\uc2b5\ub2c8\ub2e4", "\ub4f1\ub85d\ub418\uc5c8\uc2b5\ub2c8\ub2e4", "\uc644\ub8cc", "\uc131\uacf5", "saved", "success"))


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


def _has_runtime_signal_delta(observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> bool:
    signals = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    for source in (action_info, signals):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "delta_console_error_count",
            "delta_console_warning_count",
            "delta_page_error_count",
            "delta_runtime_exception_count",
            "delta_unhandled_rejection_count",
            "delta_network_request_failed_count",
            "delta_api_4xx_count",
            "delta_api_5xx_count",
            "delta_api_timeout_count",
            "delta_static_asset_failure_count",
            "delta_auth_permission_anomaly_count",
            "delta_sensitive_data_exposure_signal_count",
            "delta_token_exposure_signal_count",
        ):
            try:
                if int(source.get(key, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
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
    if common_repeated_titles and common_repeated_titles == set(duplicated):
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
        "classification": "exploratory_anomaly",
        "human_review_status": "needs_review" if not visible_overlapping_duplicates else "",
        "ground_truth_match_status": "requires_component_evidence",
        "evidence": {
            "duplicated_titles": duplicated,
            "visible_duplicate_candidates": visible_overlapping_duplicates,
            "book_context": has_book_context,
            "axtree_or_text_only": not bool(visible_overlapping_duplicates),
            "empty_state_and_chart_both_visible": False,
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
        FORUM_MORE_TEXT,
        f"{FORUM_MORE_TEXT} {FORUM_MORE_TEXT}",
        FORUM_SEARCH_PLACEHOLDER,
        f"{FORUM_SEARCH_PLACEHOLDER} {FORUM_SEARCH_PLACEHOLDER}",
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
