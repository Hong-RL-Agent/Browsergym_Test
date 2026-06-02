"""BrowserGym observation adapter for J.A.W.S."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.site_profile_service import match_candidate_to_profile, profile_keyword_matches


INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "combobox",
    "checkbox",
    "radio",
    "menuitem",
    "tab",
}
RELAXED_AX_ROLES = INTERACTIVE_ROLES | {"generic", "group", "region"}
DOM_INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea", "summary", "option", "label"}
DOM_INTERACTIVE_ATTRS = {"onclick", "href", "role", "tabindex", "data-testid", "data-bug-id", "aria-label", "title"}
ACTION_TEXT_KEYWORDS = (
    "시작",
    "실행",
    "검색",
    "제출",
    "저장",
    "확인",
    "취소",
    "로그인",
    "회원가입",
    "추가",
    "삭제",
    "수정",
    "보기",
    "상세",
    "next",
    "submit",
    "save",
    "search",
    "login",
    "add",
    "delete",
    "edit",
    "view",
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
    "?λ컮援щ땲",
)
CART_KEYWORDS = ("장바구니", "카트", "cart", "basket", "?λ컮援щ땲", "移댄듃")
SOURCE_PRIORITY = {
    "axtree": 7,
    "dom": 6,
    "extra_props_clickable": 5,
    "text_fallback": 4,
    "bbox_fallback": 3,
    "fullscreen_clickable_fallback": 2,
    "emergency_bbox_fallback": 1,
}
GLOBAL_EMPTY_KEYWORDS = (
    "데이터가 없습니다",
    "기록이 없습니다",
    "결과가 없습니다",
    "no data",
    "no records",
    "empty",
    "nothing found",
)
CHART_LIKE_KEYWORDS = (
    "chart",
    "chart-container",
    "graph",
    "그래프",
    "차트",
    "bar",
    "svg",
    "canvas",
)


class BrowserGymObservationAdapter:
    def __init__(self, max_candidates: int = 32, site_profile: Optional[Mapping[str, Any]] = None) -> None:
        self.max_candidates = max_candidates
        self.site_profile = dict(site_profile or {})

    def adapt(
        self,
        obs: Optional[Mapping[str, Any]],
        info: Optional[Mapping[str, Any]] = None,
        previous_obs: Optional[Mapping[str, Any]] = None,
        history: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.convert(obs, info=info, previous_obs=previous_obs, history=history)

    def convert(
        self,
        raw_obs: Optional[Mapping[str, Any]],
        info: Optional[Mapping[str, Any]] = None,
        previous_obs: Optional[Mapping[str, Any]] = None,
        history: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        obs = raw_obs or {}
        info = info or {}
        history = history or {}
        site_profile = info.get("site_profile") if isinstance(info.get("site_profile"), Mapping) else self.site_profile

        site_id = _as_str(info.get("site_id"))
        url = _as_str(obs.get("url") or _active_page_value(obs, "open_pages_urls"))
        title = _as_str(obs.get("title") or _active_page_value(obs, "open_pages_titles"))
        screenshot = obs.get("screenshot")
        viewport_width, viewport_height = _viewport_size(screenshot, info)
        page_text = _page_text(obs)
        candidate_elements, candidate_debug = self._extract_candidates(
            obs,
            page_text,
            title,
            viewport_width,
            viewport_height,
            site_profile,
        )
        cart_state = _cart_state(page_text, obs, candidate_elements)
        dom_node_count = _dom_node_count(obs.get("dom_object"))
        elapsed_time = _as_float(obs.get("elapsed_time"), default=0.0)
        section_texts = _section_texts(page_text)
        visible_texts = _visible_texts(obs, candidate_elements, page_text)
        data_bug_ids_found = _data_bug_ids_found(obs, candidate_elements)
        has_empty_state_text = bool(
            profile_keyword_matches(page_text, site_profile, "negative_keywords")
            or _contains_keyword(page_text, GLOBAL_EMPTY_KEYWORDS)
        )
        has_chart_like_element = any(bool(candidate.get("is_chart_related")) for candidate in candidate_elements)
        has_modal_or_dialog = _has_modal_or_dialog(obs, candidate_elements, page_text)
        has_form = _has_form(obs, candidate_elements, page_text)
        dom_attributes_summary = _dom_attributes_summary(obs.get("dom_object"))
        layout_signals = _layout_signals(candidate_elements, viewport_width, viewport_height)

        previous_url = _nested_get(previous_obs, ("page_state", "url"), "")
        url_changed = bool(previous_url and previous_url != url)
        last_action_error = bool(obs.get("last_action_error")) or bool(info.get("action_error"))
        viewport_type = _viewport_type(viewport_width)
        raw_keys = _raw_observation_keys(obs)
        raw_summary = _compact_raw_observation(obs)

        return {
            "browsergym_raw_observation": raw_summary,
            "browsergym_raw_observation_keys": raw_keys,
            "browsergym_raw_observation_key_count": len(raw_keys),
            "browsergym_raw_observation_source": "env.reset/env.step",
            "page_state": {
                "site_id": site_id,
                "url": url,
                "title": title,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
                "viewport_type": viewport_type,
                "page_text_length": len(page_text),
                "page_text": page_text,
                "page_text_sample": page_text[:4000],
                "visible_texts": visible_texts,
                "data_bug_ids_found": data_bug_ids_found,
                "data_bug_id_counts": dict(Counter(data_bug_ids_found)),
                "dom_attributes_summary": dom_attributes_summary,
                "section_texts": section_texts,
                "has_empty_state_text": has_empty_state_text,
                "has_chart_like_element": has_chart_like_element,
                "has_modal": has_modal_or_dialog,
                "has_dialog": has_modal_or_dialog,
                "has_form": has_form,
                "has_modal_or_dialog": has_modal_or_dialog,
                "cart_text": cart_state["cart_text"],
                "cart_count": cart_state["cart_count"],
                "cart_count_detected": cart_state["cart_count_detected"],
                "cart_keywords_found": cart_state["cart_keywords_found"],
                "cart_related_texts": cart_state["cart_related_texts"],
                "dom_node_count": dom_node_count,
                "elapsed_time": elapsed_time,
                "browsergym_raw_observation_keys": raw_keys,
                "browsergym_raw_observation_key_count": len(raw_keys),
                "browsergym_text_source": _text_source(obs),
                "browsergym_has_axtree": "axtree_object" in obs,
                "browsergym_has_dom": "dom_object" in obs,
                "browsergym_has_screenshot": "screenshot" in obs,
            },
            "candidate_elements": candidate_elements,
            "candidate_debug": candidate_debug,
            "runtime_signals": {
                "site_id": site_id,
                "browsergym_raw_observation_keys": raw_keys,
                "browsergym_raw_observation_key_count": len(raw_keys),
                "url_changed": url_changed,
                "last_action_error": last_action_error,
                "elapsed_time": elapsed_time,
                "viewport_type": viewport_type,
                "cart_text": cart_state["cart_text"],
                "cart_count": cart_state["cart_count"],
                "cart_count_detected": cart_state["cart_count_detected"],
                "cart_keywords_found": cart_state["cart_keywords_found"],
                "cart_related_texts": cart_state["cart_related_texts"],
                "page_text": page_text,
                "page_text_length": len(page_text),
                "visible_texts": visible_texts,
                "data_bug_ids_found": data_bug_ids_found,
                "section_texts": section_texts,
                "has_empty_state_text": has_empty_state_text,
                "has_chart_like_element": has_chart_like_element,
                "has_modal": has_modal_or_dialog,
                "has_dialog": has_modal_or_dialog,
                "has_form": has_form,
                "has_modal_or_dialog": has_modal_or_dialog,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
            },
            "layout_signals": layout_signals,
            "history": {
                "step_index": int(history.get("step_index", 0) or 0),
                "previous_action_type": _as_str(history.get("previous_action_type", "noop")),
                "no_change_steps": int(history.get("no_change_steps", 0) or 0),
            },
        }

    def _extract_candidates(
        self,
        obs: Mapping[str, Any],
        page_text: str,
        page_title: str,
        viewport_width: int,
        viewport_height: int,
        site_profile: Optional[Mapping[str, Any]] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        properties = _index_extra_properties(obs.get("extra_element_properties"))
        source_counts: Counter[str] = Counter()
        rejected_counts: Counter[str] = Counter()
        candidates_by_key: Dict[str, Dict[str, Any]] = {}

        raw_debug = {
            "raw_axtree_nodes": len(list(_axtree_nodes(obs.get("axtree_object")))),
            "raw_extra_props": len(properties),
            "raw_dom_nodes": _dom_node_count(obs.get("dom_object")),
        }

        for node in _axtree_nodes(obs.get("axtree_object")):
            candidate = self._candidate_from_axtree(
                node,
                properties,
                page_text,
                page_title,
                viewport_width,
                viewport_height,
                rejected_counts,
                site_profile,
            )
            if candidate:
                _add_candidate(candidates_by_key, candidate)
                source_counts[candidate["source"]] += 1

        for key, prop in properties.items():
            candidate = self._candidate_from_extra_prop(
                key,
                prop,
                page_text,
                viewport_width,
                viewport_height,
                rejected_counts,
                site_profile=site_profile,
            )
            if candidate:
                _add_candidate(candidates_by_key, candidate)
                source_counts[candidate["source"]] += 1

        for dom_candidate in _dom_candidates(obs.get("dom_object"), page_text, site_profile):
            _add_candidate(candidates_by_key, dom_candidate)
            source_counts[dom_candidate["source"]] += 1

        candidates = list(candidates_by_key.values())
        if not candidates:
            for key, prop in properties.items():
                candidate = self._candidate_from_extra_prop(
                    key,
                    prop,
                    page_text,
                    viewport_width,
                    viewport_height,
                    rejected_counts,
                    allow_fullscreen=True,
                    source="fullscreen_clickable_fallback",
                    max_score=0.2,
                    site_profile=site_profile,
                )
                if candidate and (candidate.get("clickable") or candidate.get("set_of_marks")):
                    _add_candidate(candidates_by_key, candidate)
                    source_counts[candidate["source"]] += 1
                if source_counts["fullscreen_clickable_fallback"] >= 3:
                    break
            candidates = list(candidates_by_key.values())

        if not candidates:
            added_large = 0
            for key, prop in properties.items():
                bbox = _normalize_bbox(prop.get("bbox") or prop.get("bounding_box"))
                visibility = _as_float(prop.get("visibility"), default=0.0)
                if not _has_bbox(bbox) or visibility <= 0.0:
                    continue
                is_large = _is_fullscreen_container(bbox, viewport_width, viewport_height)
                if is_large and added_large >= 1:
                    continue
                added_large += 1 if is_large else 0
                candidate = _make_candidate(
                    bid=str(key),
                    text=_as_str(prop.get("text") or prop.get("aria-label") or prop.get("title") or f"element-{key}"),
                    name=_as_str(prop.get("text") or prop.get("aria-label") or prop.get("title") or f"element-{key}"),
                    role=_as_str(prop.get("role") or "generic"),
                    tag=_as_str(prop.get("tag") or prop.get("tagName")),
                    bbox=bbox,
                    visibility=visibility,
                    clickable=_as_bool(prop.get("clickable"), False),
                    enabled=True,
                    source="emergency_bbox_fallback",
                    page_text=page_text,
                    set_of_marks=_as_bool(prop.get("set_of_marks"), False),
                    clickable_score=0.1,
                    action_priority=0.1,
                    data_bug_id=_attr_value(prop, "data-bug-id", "data_bug_id"),
                    has_data_testid=_has_attr(prop, "data-testid", "data_testid"),
                    aria_label=_as_str(prop.get("aria-label") or prop.get("aria_label")),
                    title=_as_str(prop.get("title")),
                    element_id=_as_str(prop.get("id")),
                    class_name=_as_str(prop.get("class") or prop.get("className")),
                    site_profile=site_profile,
                )
                _add_candidate(candidates_by_key, candidate)
                source_counts[candidate["source"]] += 1
                if source_counts["emergency_bbox_fallback"] >= 5:
                    break
            candidates = list(candidates_by_key.values())

        candidates.sort(key=_candidate_sort_key, reverse=True)
        candidates = candidates[: self.max_candidates]
        debug = {
            **raw_debug,
            "candidate_source_counts": dict(source_counts),
            "rejected_counts": dict(rejected_counts),
            "catalog_candidate_count": sum(1 for candidate in candidates if candidate.get("catalog_bug_id_matches")),
            "catalog_keyword_match_count": sum(
                len(candidate.get("catalog_keyword_matches", []) or []) for candidate in candidates
            ),
            "catalog_selector_match_count": sum(1 for candidate in candidates if candidate.get("catalog_selector_match")),
            "openended_interactive_candidate_count": sum(1 for candidate in candidates if candidate.get("is_interactive")),
            "openended_keyword_match_count": sum(
                len(candidate.get("openended_keyword_matches", []) or []) for candidate in candidates
            ),
        }
        return candidates, debug

    def _candidate_from_axtree(
        self,
        node: Any,
        properties: Mapping[str, Mapping[str, Any]],
        page_text: str,
        page_title: str,
        viewport_width: int,
        viewport_height: int,
        rejected_counts: Counter[str],
        site_profile: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(node, Mapping):
            return None
        role = _normalize_role(node.get("role"))
        bid = _node_bid(node)
        backend_id = _as_str(node.get("backendDOMNodeId"))
        prop = properties.get(str(bid), {}) or properties.get(backend_id, {})
        name = _as_str(_ax_value(node.get("name")) or _ax_value(node.get("text")) or prop.get("text")).strip()
        text = _as_str(_ax_value(node.get("text")) or prop.get("text") or name).strip()
        focusable = _ax_property_bool(node, "focusable")
        ignored = bool(node.get("ignored"))
        has_backend = bool(backend_id)
        relaxed_candidate = (
            not ignored
            or role in RELAXED_AX_ROLES
            or bool(name)
            or focusable
            or has_backend
        )
        if not relaxed_candidate:
            rejected_counts["not_candidate_like"] += 1
            return None
        if not bid and backend_id:
            bid = backend_id
        if not bid:
            rejected_counts["missing_bid"] += 1
            return None

        bbox = _normalize_bbox(prop.get("bbox") or prop.get("bounding_box") or node.get("bbox"))
        visibility = max(0.0, min(1.0, _as_float(prop.get("visibility"), default=1.0 if _has_bbox(bbox) else 0.0)))
        clickable = _as_bool(prop.get("clickable"), role in INTERACTIVE_ROLES or focusable)
        set_of_marks = _as_bool(prop.get("set_of_marks"), False)
        meaningful_text = _meaningful_name(text or name, page_title)
        if not _has_bbox(bbox) and not meaningful_text:
            rejected_counts["no_bbox"] += 1
            return None
        is_fullscreen = _is_fullscreen_container(bbox, viewport_width, viewport_height)
        if is_fullscreen and role not in INTERACTIVE_ROLES and not meaningful_text:
            rejected_counts["too_large_container"] += 1
            return None
        clickable_score = 0.0
        clickable_score += 0.45 if clickable else 0.0
        clickable_score += 0.25 if set_of_marks else 0.0
        clickable_score += 0.2 if role in INTERACTIVE_ROLES else 0.0
        clickable_score += 0.15 if focusable else 0.0
        clickable_score += 0.15 if _contains_keyword(text or name, ACTION_TEXT_KEYWORDS) else 0.0
        clickable_score += 0.1 if meaningful_text else 0.0
        clickable_score += min(0.15, visibility * 0.15)
        if is_fullscreen:
            clickable_score -= 0.35
        source = "axtree" if (role in RELAXED_AX_ROLES or clickable or set_of_marks or focusable) else "text_fallback"
        if clickable_score < 0.15 and not meaningful_text:
            rejected_counts["low_score"] += 1
            return None
        return _make_candidate(
            bid=str(bid),
            text=text or name,
            name=name or text,
            role=role or "generic",
            tag=_as_str(_ax_value(node.get("tag")) or _ax_value(node.get("tagName")) or prop.get("tag") or prop.get("tagName")),
            bbox=bbox,
            visibility=visibility,
            clickable=clickable,
            enabled=not _as_bool(node.get("disabled") or prop.get("disabled"), False),
            source=source,
            page_text=page_text,
            set_of_marks=set_of_marks,
            clickable_score=max(0.0, min(1.0, clickable_score)),
            data_bug_id=_attr_value(prop, "data-bug-id", "data_bug_id"),
            has_data_testid=_has_attr(prop, "data-testid", "data_testid"),
            aria_label=_as_str(prop.get("aria-label") or prop.get("aria_label")),
            title=_as_str(prop.get("title")),
            element_id=_as_str(prop.get("id")),
            class_name=_as_str(prop.get("class") or prop.get("className")),
            site_profile=site_profile,
        )

    def _candidate_from_extra_prop(
        self,
        key: str,
        prop: Mapping[str, Any],
        page_text: str,
        viewport_width: int,
        viewport_height: int,
        rejected_counts: Counter[str],
        allow_fullscreen: bool = False,
        source: str = "extra_props_clickable",
        max_score: Optional[float] = None,
        site_profile: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        bbox = _normalize_bbox(prop.get("bbox") or prop.get("bounding_box"))
        visibility = max(0.0, min(1.0, _as_float(prop.get("visibility"), default=0.0)))
        clickable = _as_bool(prop.get("clickable"), False)
        set_of_marks = _as_bool(prop.get("set_of_marks"), False)
        if not (clickable or set_of_marks or visibility >= 0.5):
            rejected_counts["not_clickable_or_visible"] += 1
            return None
        if not _has_bbox(bbox):
            rejected_counts["no_bbox"] += 1
            return None
        if bbox[2] < 5 or bbox[3] < 5:
            rejected_counts["bbox_too_small"] += 1
            return None
        is_fullscreen = _is_fullscreen_container(bbox, viewport_width, viewport_height)
        if is_fullscreen and not allow_fullscreen:
            rejected_counts["too_large_container"] += 1
            return None
        text = _as_str(prop.get("text") or prop.get("aria-label") or prop.get("aria_label") or prop.get("title") or "")
        role = _as_str(prop.get("role") or "")
        tag = _as_str(prop.get("tag") or prop.get("tagName") or "")
        score = 0.0
        score += 0.45 if clickable else 0.0
        score += 0.25 if set_of_marks else 0.0
        score += min(0.25, visibility * 0.25)
        score += 0.1 if text else 0.0
        if is_fullscreen:
            score = min(score, 0.2)
        if max_score is not None:
            score = min(score, max_score)
        if score < 0.1:
            rejected_counts["low_score"] += 1
            return None
        actual_source = source
        if source == "extra_props_clickable" and not (clickable or set_of_marks):
            actual_source = "bbox_fallback"
        return _make_candidate(
            bid=str(key),
            text=text or f"element-{key}",
            name=text or f"element-{key}",
            role=role or "generic",
            tag=tag,
            bbox=bbox,
            visibility=visibility,
            clickable=clickable,
            enabled=not _as_bool(prop.get("disabled"), False),
            source=actual_source if not is_fullscreen else "fullscreen_clickable_fallback",
            page_text=page_text,
            set_of_marks=set_of_marks,
            clickable_score=score,
            action_priority=max(0.1, score),
            data_bug_id=_attr_value(prop, "data-bug-id", "data_bug_id"),
            has_data_testid=_has_attr(prop, "data-testid", "data_testid"),
            aria_label=_as_str(prop.get("aria-label") or prop.get("aria_label")),
            title=_as_str(prop.get("title")),
            element_id=_as_str(prop.get("id")),
            class_name=_as_str(prop.get("class") or prop.get("className")),
            site_profile=site_profile,
        )


def _make_candidate(
    bid: str,
    text: str,
    name: str,
    role: str,
    tag: str,
    bbox: List[float],
    visibility: float,
    clickable: bool,
    enabled: bool,
    source: str,
    page_text: str,
    set_of_marks: bool = False,
    clickable_score: float = 0.0,
    action_priority: Optional[float] = None,
    data_bug_id: str = "",
    has_data_testid: bool = False,
    aria_label: str = "",
    title: str = "",
    element_id: str = "",
    class_name: str = "",
    site_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    selector_hint = f'[data-bug-id="{data_bug_id}"]' if data_bug_id else ""
    keyword_text = " ".join([text, name, aria_label, title, element_id, class_name, data_bug_id])
    initial_candidate = {
        "text": text,
        "name": name,
        "aria_label": aria_label,
        "title": title,
        "id": element_id,
        "class_name": class_name,
        "data_bug_id": data_bug_id,
        "selector_hint": selector_hint,
        "tag": tag,
        "role": role,
    }
    catalog_matches = match_candidate_to_profile(initial_candidate, site_profile)
    catalog_keyword_matches = catalog_matches["keyword_matches"]
    catalog_bug_id_matches = catalog_matches["bug_id_matches"]
    catalog_selector_match = bool(catalog_matches["selector_match"])
    role_norm = role.lower() if role else "generic"
    tag_norm = tag.lower() if tag else ""
    is_interactive = bool(
        clickable
        or role_norm in INTERACTIVE_ROLES
        or tag_norm in DOM_INTERACTIVE_TAGS
        or _contains_keyword(keyword_text, ACTION_TEXT_KEYWORDS)
    )
    is_form_field = bool(tag_norm in {"input", "select", "textarea"} or role_norm in {"textbox", "searchbox", "combobox"})
    openended_keyword_matches = sorted(
        set(
            profile_keyword_matches(keyword_text, site_profile, "interaction_keywords")
            + profile_keyword_matches(keyword_text, site_profile, "form_keywords")
        )
    )
    openended_text = keyword_text.lower()
    is_login_related = _contains_keyword(openended_text, ("sign in", "login", "로그인"))
    is_cart_related_openended = _contains_keyword(openended_text, ("cart", "basket", "장바구니"))
    is_checkout_related = _contains_keyword(openended_text, ("checkout", "payment", "결제"))
    is_search_related = _contains_keyword(openended_text, ("search", "검색"))
    is_filter_related = _contains_keyword(openended_text, ("filter", "sort"))
    is_submit_related = _contains_keyword(openended_text, ("submit", "save", "continue", "next", "add", "buy", "purchase"))
    is_purchase_action = _contains_keyword(keyword_text, PURCHASE_KEYWORDS)
    action_hints = set(_profile_values(site_profile, "action_hints"))
    candidate_action_hints = _candidate_action_hints(site_profile, catalog_bug_id_matches)
    bug_types = set(str(item.get("type") or "") for item in (site_profile or {}).get("bugs", []) if isinstance(item, Mapping))
    lower_catalog_text = " ".join([keyword_text, " ".join(catalog_keyword_matches), " ".join(catalog_bug_id_matches)]).lower()
    is_sparse_related = _contains_keyword(lower_catalog_text, ("sparse", "sparse data"))
    is_forbidden_related = _contains_keyword(lower_catalog_text, ("forbidden", "403", "restricted", "access denied"))
    is_async_related = _contains_keyword(lower_catalog_text, ("async", "loading", "pending"))
    is_hang_related = _contains_keyword(lower_catalog_text, ("hang", "timeout", "stuck"))
    is_quantity_control = _contains_keyword(lower_catalog_text, ("quantity", "qty", "+", "-", "plus", "minus"))
    is_cart_quantity_related = _contains_keyword(lower_catalog_text, ("cart", "quantity", "qty", "subtotal", "total"))
    is_network_related = _contains_keyword(lower_catalog_text, ("api", "network", "forbidden", "403"))
    is_workout_add_action = bool(catalog_bug_id_matches and ("click" in action_hints or "button-no-response" in bug_types))
    is_weekly_stats_related = bool(catalog_keyword_matches and profile_keyword_matches(keyword_text, site_profile, "section_keywords"))
    is_empty_state_related = bool(
        profile_keyword_matches(" ".join([keyword_text, page_text]), site_profile, "negative_keywords")
        or _contains_keyword(" ".join([keyword_text, page_text]), GLOBAL_EMPTY_KEYWORDS)
    )
    layout_bug_ids = _layout_bug_ids(site_profile)
    layout_check_type = _layout_check_type(site_profile, catalog_bug_id_matches)
    layout_target_values = _layout_values(site_profile, "target_keywords") + _layout_values(site_profile, "expected_elements")
    layout_section_values = _layout_values(site_profile, "section_keywords")
    is_chart_related = _contains_keyword(" ".join([keyword_text, source, data_bug_id, class_name]), CHART_LIKE_KEYWORDS)
    is_chart_related = is_chart_related or any(
        _contains_keyword(keyword, CHART_LIKE_KEYWORDS) for keyword in catalog_keyword_matches
    )
    is_chart_related = is_chart_related or bool(
        _contains_keyword(keyword_text, layout_target_values)
    )
    is_layout_target = bool(
        layout_check_type
        or set(catalog_bug_id_matches).intersection(layout_bug_ids)
        or (catalog_selector_match and set(catalog_bug_id_matches).intersection(layout_bug_ids))
        or is_chart_related
        or _contains_keyword(keyword_text, layout_section_values)
    )
    is_cart_related = _contains_keyword(" ".join([keyword_text, page_text]), CART_KEYWORDS) or is_cart_related_openended
    priority = action_priority if action_priority is not None else clickable_score
    priority += 1.0 if is_purchase_action else 0.0
    priority += 2.0 if catalog_selector_match else 0.0
    priority += 1.5 if catalog_bug_id_matches else 0.0
    priority += min(1.0, 0.25 * len(catalog_keyword_matches))
    priority += 0.4 if candidate_action_hints else 0.0
    priority += 0.35 if any((is_sparse_related, is_forbidden_related, is_async_related, is_hang_related, is_cart_quantity_related, is_network_related)) else 0.0
    priority += 0.3 if clickable else 0.0
    openended_priority = 0.0
    openended_priority += 0.5 if is_interactive else 0.0
    openended_priority += 0.35 if clickable else 0.0
    openended_priority += min(1.0, 0.2 * len(openended_keyword_matches))
    openended_priority += 0.25 if is_form_field else 0.0
    openended_priority += 0.25 if any((is_login_related, is_cart_related_openended, is_checkout_related, is_search_related, is_filter_related, is_submit_related)) else 0.0
    priority += openended_priority
    priority += 0.25 if data_bug_id else 0.0
    priority += 0.2 if visibility > 0.2 else 0.0
    priority += 0.1 if _has_bbox(bbox) else 0.0
    if visibility <= 0.0:
        priority -= 0.2
    return {
        "bid": str(bid),
        "text": text,
        "name": name,
        "aria_label": aria_label,
        "title": title,
        "id": element_id,
        "class_name": class_name,
        "role": role_norm,
        "tag": tag_norm,
        "selector_hint": selector_hint,
        "visibility": max(0.0, min(1.0, visibility)),
        "visible": visibility > 0.0,
        "enabled": enabled,
        "clickable": clickable,
        "set_of_marks": set_of_marks,
        "bbox": bbox,
        "source": source,
        "clickable_score": round(max(0.0, min(1.0, clickable_score)), 4),
        "is_purchase_action": is_purchase_action,
        "is_interactive": is_interactive,
        "is_form_field": is_form_field,
        "is_login_related": is_login_related,
        "is_cart_related": is_cart_related,
        "is_quantity_control": is_quantity_control,
        "is_sparse_related": is_sparse_related,
        "is_forbidden_related": is_forbidden_related,
        "is_async_related": is_async_related,
        "is_hang_related": is_hang_related,
        "is_cart_quantity_related": is_cart_quantity_related,
        "is_network_related": is_network_related,
        "is_checkout_related": is_checkout_related,
        "is_search_related": is_search_related,
        "is_filter_related": is_filter_related,
        "is_submit_related": is_submit_related,
        "openended_keyword_matches": openended_keyword_matches,
        "openended_action_priority": round(max(0.0, openended_priority), 4),
        "is_workout_add_action": is_workout_add_action,
        "is_weekly_stats_related": is_weekly_stats_related,
        "is_empty_state_related": is_empty_state_related,
        "is_chart_related": is_chart_related,
        "is_chart_like": is_chart_related,
        "is_layout_target": is_layout_target,
        "layout_check_type": layout_check_type,
        "catalog_keyword_matches": catalog_keyword_matches,
        "catalog_selector_match": catalog_selector_match,
        "catalog_bug_id_matches": catalog_bug_id_matches,
        "action_hints": candidate_action_hints,
        "catalog_action_priority": round(0.5 * len(candidate_action_hints) + 0.25 * len(catalog_bug_id_matches), 4),
        "catalog_priority": round(float(bool(catalog_bug_id_matches)) + 0.25 * len(catalog_keyword_matches), 4),
        "action_priority": round(max(0.0, priority), 4),
        "has_text": bool(text or name),
        "text_length": len(text or name),
        "data_bug_id": data_bug_id,
        "has_data_bug_id": bool(data_bug_id),
        "has_data_testid": has_data_testid,
    }


def _add_candidate(candidates: Dict[str, Dict[str, Any]], candidate: Dict[str, Any]) -> None:
    key = str(candidate.get("bid") or "")
    if not key:
        return
    previous = candidates.get(key)
    if previous is None or _candidate_sort_key(candidate) > _candidate_sort_key(previous):
        candidates[key] = candidate


def _candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(item.get("catalog_bug_id_matches")),
        bool(item.get("has_data_bug_id")),
        bool(item.get("catalog_selector_match")),
        float(item.get("catalog_action_priority", 0.0) or 0.0),
        len(item.get("catalog_keyword_matches", []) or []),
        float(item.get("visibility", 0.0) or 0.0) > 0.0,
        bool(item.get("clickable")),
        bool(item.get("is_interactive")),
        len(item.get("openended_keyword_matches", []) or []),
        bool(item.get("is_form_field")),
        _has_bbox(item.get("bbox", [])),
        bool(item.get("is_layout_target")),
        bool(item.get("is_purchase_action")),
        str(item.get("role") or "") in INTERACTIVE_ROLES,
        float(item.get("openended_action_priority", 0.0) or 0.0),
        float(item.get("clickable_score", 0.0) or 0.0),
        bool(item.get("set_of_marks")),
        SOURCE_PRIORITY.get(str(item.get("source") or ""), 0),
        float(item.get("action_priority", 0.0) or 0.0),
    )


def _dom_candidates(
    dom_object: Any,
    page_text: str,
    site_profile: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not isinstance(dom_object, Mapping):
        return candidates
    strings = dom_object.get("strings") if isinstance(dom_object.get("strings"), list) else []
    documents = dom_object.get("documents")
    if not isinstance(documents, list):
        return candidates
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        nodes = document.get("nodes")
        if not isinstance(nodes, Mapping):
            continue
        node_names = nodes.get("nodeName", [])
        backend_ids = nodes.get("backendNodeId", [])
        attrs = nodes.get("attributes", [])
        node_values = nodes.get("nodeValue", [])
        if not isinstance(node_names, list):
            continue
        for index, raw_name in enumerate(node_names):
            tag = _string_at(strings, raw_name).lower()
            attr_map = _decode_attrs(strings, attrs[index] if isinstance(attrs, list) and index < len(attrs) else [])
            text = " ".join(
                value
                for key, value in attr_map.items()
                if key in {"aria-label", "title", "value", "alt", "data-testid", "data-bug-id"}
            ).strip()
            if not text and isinstance(node_values, list) and index < len(node_values):
                text = _string_at(strings, node_values[index]).strip()
            has_interactive_attr = any(key in attr_map for key in DOM_INTERACTIVE_ATTRS)
            has_catalog_attr = bool(attr_map.get("data-bug-id"))
            if tag not in DOM_INTERACTIVE_TAGS and not has_interactive_attr and not has_catalog_attr:
                continue
            bid = str(backend_ids[index]) if isinstance(backend_ids, list) and index < len(backend_ids) else f"dom-{index}"
            role = attr_map.get("role") or ("link" if tag == "a" else "button" if tag == "button" else tag or "generic")
            score = 0.45 if tag in DOM_INTERACTIVE_TAGS else 0.25
            score += 0.2 if has_interactive_attr else 0.0
            score += 0.1 if text else 0.0
            candidates.append(
                _make_candidate(
                    bid=bid,
                    text=text or tag or f"dom-{index}",
                    name=text or tag or f"dom-{index}",
                    role=role,
                    tag=tag,
                    bbox=[0.0, 0.0, 0.0, 0.0],
                    visibility=0.0,
                    clickable=True,
                    enabled=True,
                    source="dom",
                    page_text=page_text,
                    clickable_score=min(1.0, score),
                    data_bug_id=attr_map.get("data-bug-id", ""),
                    has_data_testid=bool(attr_map.get("data-testid")),
                    aria_label=attr_map.get("aria-label", ""),
                    title=attr_map.get("title", ""),
                    element_id=attr_map.get("id", ""),
                    class_name=attr_map.get("class", ""),
                    site_profile=site_profile,
                )
            )
    return candidates


def _active_page_value(obs: Mapping[str, Any], key: str) -> Any:
    values = obs.get(key)
    if not isinstance(values, (list, tuple)) or not values:
        return ""
    index = _as_int(obs.get("active_page_index"), default=0)
    return values[index] if 0 <= index < len(values) else values[0]


def _raw_observation_keys(obs: Mapping[str, Any]) -> List[str]:
    if not isinstance(obs, Mapping):
        return []
    return sorted(str(key) for key in obs.keys())


def _compact_raw_observation(obs: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(obs, Mapping):
        return {}
    return {str(key): _compact_raw_value(value, depth=0) for key, value in obs.items()}


def _compact_raw_value(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return _raw_value_summary(value)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= 120:
                result["__truncated__"] = f"{len(value) - index} more keys"
                break
            result[str(key)] = _compact_raw_value(child, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        limit = 80 if depth <= 1 else 30
        items = [_compact_raw_value(item, depth + 1) for item in list(value)[:limit]]
        if len(value) > limit:
            items.append({"__truncated__": f"{len(value) - limit} more items"})
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = value if not isinstance(value, str) else value[:4000]
        return text
    return _raw_value_summary(value)


def _raw_value_summary(value: Any) -> Dict[str, Any]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": [int(item) for item in tuple(shape)]}
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    return {"type": type(value).__name__, "repr": repr(value)[:1000]}


def _text_source(obs: Mapping[str, Any]) -> str:
    if obs.get("text"):
        return "text"
    if obs.get("axtree_object"):
        return "axtree_object"
    if obs.get("dom_object"):
        return "dom_object"
    if obs.get("goal"):
        return "goal"
    return ""


def _axtree_nodes(axtree_object: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(axtree_object, Mapping):
        nodes = axtree_object.get("nodes")
        if isinstance(nodes, list):
            return nodes
    if isinstance(axtree_object, list):
        return axtree_object
    return []


def _index_extra_properties(extra: Any) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if isinstance(value, Mapping):
                indexed[str(key)] = value
                for alias in ("bid", "nodeId", "backendDOMNodeId", "backendNodeId"):
                    if alias in value:
                        indexed[str(value[alias])] = value
        return indexed
    if isinstance(extra, list):
        for item in extra:
            if not isinstance(item, Mapping):
                continue
            for alias in ("bid", "nodeId", "backendDOMNodeId", "backendNodeId"):
                if alias in item:
                    indexed[str(item[alias])] = item
    return indexed


def _node_bid(node: Mapping[str, Any]) -> str:
    for key in ("browsergym_id", "bid", "nodeId", "backendDOMNodeId", "backendNodeId"):
        value = node.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def _normalize_role(value: Any) -> str:
    return _as_str(_ax_value(value)).lower()


def _normalize_bbox(value: Any) -> List[float]:
    if isinstance(value, Mapping):
        values = [value.get("x"), value.get("y"), value.get("width"), value.get("height")]
    elif isinstance(value, (list, tuple)):
        values = list(value[:4])
    else:
        values = []
    while len(values) < 4:
        values.append(0.0)
    return [_as_float(v, default=0.0) for v in values[:4]]


def _viewport_size(screenshot: Any, info: Mapping[str, Any]) -> tuple[int, int]:
    width = info.get("viewport_width") or info.get("width")
    height = info.get("viewport_height") or info.get("height")
    if width and height:
        return int(width), int(height)
    shape = getattr(screenshot, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return 0, 0


def _viewport_type(width: int) -> str:
    return "mobile" if width and width <= 768 else "desktop"


def _page_text(obs: Mapping[str, Any]) -> str:
    parts: List[str] = []
    raw_text = obs.get("text")
    if isinstance(raw_text, str) and raw_text.strip():
        parts.append(raw_text)
    elif isinstance(raw_text, list):
        parts.extend(str(item) for item in raw_text[:300] if str(item).strip())
    for node in _axtree_nodes(obs.get("axtree_object")):
        if isinstance(node, Mapping):
            text = _as_str(_ax_value(node.get("name")) or _ax_value(node.get("text"))).strip()
            if text:
                parts.append(text)
    goal = obs.get("goal")
    if isinstance(goal, str):
        parts.append(goal)
    parts.extend(_dom_text_parts(obs.get("dom_object")))
    return " ".join(parts)


def _visible_texts(obs: Mapping[str, Any], candidates: List[Mapping[str, Any]], page_text: str) -> List[str]:
    texts: List[str] = []
    for candidate in candidates:
        text = " ".join(_as_str(candidate.get(key)) for key in ("text", "name", "aria_label", "title")).strip()
        if text:
            texts.append(text[:240])
    if not texts:
        texts.extend(part[:240] for part in _dom_text_parts(obs.get("dom_object"), limit=80) if part.strip())
    if not texts and page_text:
        texts.append(page_text[:240])
    return _unique_strings(texts)[:120]


def _data_bug_ids_found(obs: Mapping[str, Any], candidates: List[Mapping[str, Any]]) -> List[str]:
    values = [str(candidate.get("data_bug_id") or "") for candidate in candidates if candidate.get("data_bug_id")]
    summary = _dom_attributes_summary(obs.get("dom_object"))
    for value in summary.get("data_bug_ids", []):
        values.append(str(value))
    return _unique_strings(values)


def _dom_attributes_summary(dom_object: Any) -> Dict[str, Any]:
    result = {
        "data_bug_ids": [],
        "ids": [],
        "classes": [],
        "roles": [],
        "aria_labels": [],
        "titles": [],
        "hrefs": [],
        "onclick_count": 0,
        "form_like_count": 0,
    }
    if not isinstance(dom_object, Mapping):
        return result
    strings = dom_object.get("strings") if isinstance(dom_object.get("strings"), list) else []
    documents = dom_object.get("documents")
    if not isinstance(documents, list):
        return result
    for document in documents:
        nodes = document.get("nodes") if isinstance(document, Mapping) else None
        if not isinstance(nodes, Mapping):
            continue
        node_names = nodes.get("nodeName", [])
        attrs = nodes.get("attributes", [])
        for index, raw_name in enumerate(node_names if isinstance(node_names, list) else []):
            tag = _string_at(strings, raw_name).lower()
            attr_map = _decode_attrs(strings, attrs[index] if isinstance(attrs, list) and index < len(attrs) else [])
            if tag in {"form", "input", "textarea", "select"}:
                result["form_like_count"] += 1
            if "onclick" in attr_map:
                result["onclick_count"] += 1
            field_map = {
                "data-bug-id": "data_bug_ids",
                "id": "ids",
                "class": "classes",
                "role": "roles",
                "aria-label": "aria_labels",
                "title": "titles",
                "href": "hrefs",
            }
            for attr_name, out_name in field_map.items():
                if attr_map.get(attr_name):
                    result[out_name].append(attr_map[attr_name])
    for key, value in list(result.items()):
        if isinstance(value, list):
            result[key] = _unique_strings(value)[:100]
    return result


def _has_form(obs: Mapping[str, Any], candidates: List[Mapping[str, Any]], page_text: str) -> bool:
    if any(str(candidate.get("tag") or "").lower() in {"form", "input", "select", "textarea"} for candidate in candidates):
        return True
    if any(str(candidate.get("role") or "").lower() in {"textbox", "combobox"} for candidate in candidates):
        return True
    summary = _dom_attributes_summary(obs.get("dom_object"))
    return int(summary.get("form_like_count", 0) or 0) > 0 or _contains_keyword(page_text, ("form", "input"))


def _profile_values(site_profile: Optional[Mapping[str, Any]], field: str) -> List[str]:
    if not site_profile:
        return []
    value = site_profile.get(field)
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if value:
        return [str(value)]
    return []


def _layout_bug_ids(site_profile: Optional[Mapping[str, Any]]) -> set[str]:
    if not site_profile:
        return set()
    ids: set[str] = set()
    for bug in site_profile.get("bugs", []) or []:
        if not isinstance(bug, Mapping):
            continue
        if str(bug.get("layout_check") or "") or str(bug.get("type") or "") in {"layout-overflow", "layout-overlap", "css-layout"}:
            bug_id = str(bug.get("bug_id") or bug.get("id") or "")
            if bug_id:
                ids.add(bug_id)
    return ids


def _candidate_action_hints(site_profile: Optional[Mapping[str, Any]], bug_ids: Iterable[str]) -> List[str]:
    if not site_profile:
        return []
    wanted = {str(value) for value in bug_ids if value}
    if not wanted:
        return []
    hints: List[str] = []
    for bug in site_profile.get("bugs", []) or []:
        if not isinstance(bug, Mapping):
            continue
        bug_id = str(bug.get("bug_id") or bug.get("id") or "")
        if wanted and bug_id not in wanted:
            continue
        for hint in bug.get("action_hints", []) or []:
            hint_text = str(hint)
            if hint_text and hint_text not in hints:
                hints.append(hint_text)
    return hints


def _layout_check_type(site_profile: Optional[Mapping[str, Any]], bug_ids: Iterable[str]) -> str:
    if not site_profile:
        return ""
    wanted = {str(value) for value in bug_ids if value}
    if not wanted:
        return ""
    for bug in site_profile.get("bugs", []) or []:
        if not isinstance(bug, Mapping):
            continue
        bug_type = str(bug.get("type") or "")
        layout_check = str(bug.get("layout_check") or "")
        if not layout_check and bug_type not in {"layout-overflow", "layout-overlap", "css-layout"}:
            continue
        bug_id = str(bug.get("bug_id") or bug.get("id") or "")
        if bug_id and bug_id in wanted:
            return layout_check or bug_type
    return ""


def _layout_values(site_profile: Optional[Mapping[str, Any]], field: str) -> List[str]:
    if not site_profile:
        return []
    values: set[str] = set()
    for bug in site_profile.get("bugs", []) or []:
        if not isinstance(bug, Mapping):
            continue
        bug_type = str(bug.get("type") or "")
        if not str(bug.get("layout_check") or "") and bug_type not in {"layout-overflow", "layout-overlap", "css-layout"}:
            continue
        for value in _profile_values(bug, field):
            values.add(value)
    return sorted(values)


def _unique_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = _as_str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _dom_text_parts(value: Any, limit: int = 300) -> List[str]:
    parts: List[str] = []

    def visit(item: Any) -> None:
        if len(parts) >= limit:
            return
        if isinstance(item, Mapping):
            strings = item.get("strings")
            if isinstance(strings, list):
                parts.extend(str(s) for s in strings[: max(0, limit - len(parts))] if isinstance(s, str) and s.strip())
                return
            for key, child in item.items():
                if key in {"text", "nodeValue", "value", "aria-label", "aria_label", "title", "alt", "innerText"}:
                    text = _as_str(child).strip()
                    if text:
                        parts.append(text)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item[:300]:
                visit(child)
        elif isinstance(item, str) and len(item) <= 200:
            parts.append(item)

    visit(value)
    return parts[:limit]


def _cart_state(page_text: str, obs: Mapping[str, Any], candidates: List[Mapping[str, Any]]) -> Dict[str, Any]:
    candidate_text = " ".join(" ".join(_as_str(candidate.get(key)) for key in ("text", "name")) for candidate in candidates)
    ax_text = " ".join(
        _as_str(_ax_value(node.get("name")) or _ax_value(node.get("text")))
        for node in _axtree_nodes(obs.get("axtree_object"))
        if isinstance(node, Mapping)
    )
    dom_text = " ".join(_dom_text_parts(obs.get("dom_object"), limit=300))
    combined_text = " ".join([page_text, ax_text, dom_text, candidate_text])
    related = _cart_related_texts(combined_text)
    cart_text = " ".join(related)
    cart_count = _extract_cart_count(cart_text) if cart_text else None
    if cart_count is None:
        cart_count = _extract_cart_count(combined_text)
    return {
        "cart_text": cart_text,
        "cart_count": cart_count,
        "cart_count_detected": cart_count is not None,
        "cart_keywords_found": _contains_keyword(combined_text, CART_KEYWORDS),
        "cart_related_texts": related,
    }


def _cart_related_texts(page_text: str) -> List[str]:
    parts = re.split(r"[\n\r|•·]+", page_text)
    related: List[str] = []
    for part in parts:
        text = " ".join(part.split())
        if text and _contains_keyword(text, CART_KEYWORDS):
            related.append(text[:240])
    if not related:
        lower_page = page_text.lower()
        for keyword in CART_KEYWORDS:
            index = lower_page.find(keyword.lower())
            if index >= 0:
                related.append(" ".join(page_text[max(0, index - 60) : min(len(page_text), index + 100)].split()))
                break
    return related[:8]


def _extract_cart_count(text: str) -> Optional[int]:
    patterns = (
        r"(?:장바구니|카트|cart|basket|移댄듃|\?λ컮援щ땲)\s*[:：]?\s*\(?\s*(\d+)\s*\)?\s*(?:개)?",
        r"(?:장바구니|카트|cart|basket|移댄듃|\?λ컮援щ땲)[^\d]{0,12}(\d+)\s*(?:개)?",
        r"(\d+)\s*개",
        r"\(\s*(\d+)\s*\)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _dom_node_count(dom_object: Any) -> int:
    if isinstance(dom_object, Mapping):
        documents = dom_object.get("documents")
        if isinstance(documents, list):
            total = 0
            for document in documents:
                if isinstance(document, Mapping):
                    nodes = document.get("nodes")
                    if isinstance(nodes, Mapping):
                        parent_index = nodes.get("parentIndex")
                        if isinstance(parent_index, list):
                            total += len(parent_index)
            if total:
                return total
        nodes = dom_object.get("nodes")
        if isinstance(nodes, list):
            return len(nodes)
        return len(dom_object)
    if isinstance(dom_object, list):
        return len(dom_object)
    return 0


def _layout_overlap_count(candidates: List[Mapping[str, Any]]) -> int:
    boxes = [c.get("bbox", [0, 0, 0, 0]) for c in candidates]
    count = 0
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a
        if aw <= 0 or ah <= 0:
            continue
        for b in boxes[i + 1 :]:
            bx, by, bw, bh = b
            if bw <= 0 or bh <= 0:
                continue
            if ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by:
                count += 1
    return count


def _layout_signals(candidates: List[Mapping[str, Any]], viewport_width: int, viewport_height: int) -> Dict[str, Any]:
    overlap_count = _layout_overlap_count(candidates)
    overflow_details: List[Dict[str, Any]] = []
    layout_candidates: List[Dict[str, Any]] = []
    chart_like_count = 0
    catalog_layout_target_count = 0
    overflow_right_count = 0
    overflow_bottom_count = 0
    tolerance = 6.0
    for child in candidates:
        if not isinstance(child, Mapping):
            continue
        child_bbox = child.get("bbox", [0, 0, 0, 0])
        if not _has_bbox(child_bbox):
            continue
        child_right = float(child_bbox[0]) + float(child_bbox[2])
        child_bottom = float(child_bbox[1]) + float(child_bbox[3])
        child_text = " ".join(
            _as_str(child.get(key))
            for key in ("text", "name", "tag", "role", "class_name", "id", "data_bug_id", "selector_hint", "source")
        )
        is_chart_like = bool(child.get("is_chart_like")) or _contains_keyword(child_text, CHART_LIKE_KEYWORDS)
        is_catalog_layout_target = bool(child.get("is_layout_target") or is_chart_like)
        if is_chart_like:
            chart_like_count += 1
        if is_catalog_layout_target:
            catalog_layout_target_count += 1
        overflow_right = bool(viewport_width and child_right > float(viewport_width) + tolerance)
        overflow_bottom = bool(viewport_height and child_bottom > float(viewport_height) + tolerance)

        parent_bbox = _best_parent_bbox(child, candidates)
        parent_overflow_right = False
        parent_overflow_bottom = False
        if parent_bbox:
            parent_overflow_right = child_right > float(parent_bbox[0]) + float(parent_bbox[2]) + tolerance
            parent_overflow_bottom = child_bottom > float(parent_bbox[1]) + float(parent_bbox[3]) + tolerance

        effective_overflow_right = overflow_right or parent_overflow_right
        effective_overflow_bottom = overflow_bottom or parent_overflow_bottom
        if effective_overflow_right:
            overflow_right_count += 1
        if effective_overflow_bottom:
            overflow_bottom_count += 1

        parent = parent_bbox or [0.0, 0.0, float(viewport_width or 0), float(viewport_height or 0)]
        detail = {
            "selector": _candidate_selector(child),
            "selector_hint": child.get("selector_hint") or _candidate_selector(child),
            "data_bug_id": child.get("data_bug_id"),
            "catalog_bug_id_matches": child.get("catalog_bug_id_matches", []),
            "catalog_keyword_matches": child.get("catalog_keyword_matches", []),
            "catalog_selector_match": bool(child.get("catalog_selector_match")),
            "layout_check_type": child.get("layout_check_type") or "",
            "is_chart_like": is_chart_like,
            "is_layout_target": is_catalog_layout_target,
            "child_bbox": child_bbox,
            "parent_bbox": parent,
            "overflow_right": effective_overflow_right,
            "overflow_bottom": effective_overflow_bottom,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "text": child_text[:240],
        }
        if is_catalog_layout_target:
            layout_candidates.append(detail)
        if is_catalog_layout_target and (effective_overflow_right or effective_overflow_bottom):
            overflow_details.append(detail)

    return {
        "layout_overlap_count": overlap_count,
        "layout_overflow_count": len(overflow_details),
        "layout_overflow_details": overflow_details[:8],
        "layout_overflow_candidates": layout_candidates[:16],
        "chart_like_candidate_count": chart_like_count,
        "catalog_layout_target_count": catalog_layout_target_count,
        "overflow_right_count": overflow_right_count,
        "overflow_bottom_count": overflow_bottom_count,
    }


def _best_parent_bbox(child: Mapping[str, Any], candidates: List[Mapping[str, Any]]) -> Optional[List[float]]:
    child_bbox = child.get("bbox", [0, 0, 0, 0])
    if not _has_bbox(child_bbox):
        return None
    cx, cy, cw, ch = [float(value or 0.0) for value in child_bbox[:4]]
    best: Optional[List[float]] = None
    best_area = float("inf")
    for candidate in candidates:
        if candidate is child or not isinstance(candidate, Mapping):
            continue
        bbox = candidate.get("bbox", [0, 0, 0, 0])
        if not _has_bbox(bbox):
            continue
        x, y, w, h = [float(value or 0.0) for value in bbox[:4]]
        if x <= cx + 2 and y <= cy + 2 and x + w >= cx + min(cw, 10) and y + h >= cy + min(ch, 10):
            area = w * h
            if area > cw * ch and area < best_area:
                best = list(bbox[:4])
                best_area = area
    return best


def _candidate_selector(candidate: Mapping[str, Any]) -> str:
    data_bug_id = _as_str(candidate.get("data_bug_id"))
    if data_bug_id:
        return f'[data-bug-id="{data_bug_id}"]'
    tag = _as_str(candidate.get("tag") or "*")
    bid = _as_str(candidate.get("bid"))
    return f"{tag}[bid='{bid}']" if bid else tag


def _section_texts(page_text: str) -> List[str]:
    normalized = " ".join(page_text.split())
    if not normalized:
        return []
    sections: List[str] = []
    for marker in ("dashboard", "section", "통계", "대시보드"):
        index = normalized.find(marker)
        if index >= 0:
            sections.append(normalized[max(0, index - 80) : min(len(normalized), index + 240)])
    if not sections:
        sections.append(normalized[:320])
    return sections[:8]


def _has_modal_or_dialog(obs: Mapping[str, Any], candidates: List[Mapping[str, Any]], page_text: str) -> bool:
    if any(str(candidate.get("role", "")).lower() in {"dialog", "alertdialog"} for candidate in candidates):
        return True
    if any(
        _contains_keyword(_as_str(candidate.get("text") or candidate.get("name")), ("modal", "dialog", "폼", "기록"))
        for candidate in candidates
    ):
        return True
    return _contains_keyword(page_text, ("modal", "dialog", "새 운동", "운동 기록", "폼"))


def _nested_get(mapping: Optional[Mapping[str, Any]], path: tuple[str, ...], default: Any) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key)
    return default if value is None else value


def _has_attr(mapping: Mapping[str, Any], *names: str) -> bool:
    return bool(_attr_value(mapping, *names))


def _attr_value(mapping: Mapping[str, Any], *names: str) -> str:
    attrs = mapping.get("attributes")
    for name in names:
        value = mapping.get(name)
        if value:
            return _as_str(value)
        if isinstance(attrs, Mapping) and attrs.get(name):
            return _as_str(attrs.get(name))
    return ""


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    haystack = _as_str(text).lower()
    return any(keyword.lower() in haystack for keyword in keywords if keyword)


def _has_bbox(bbox: Any) -> bool:
    return isinstance(bbox, list) and len(bbox) >= 4 and float(bbox[2] or 0.0) > 0.0 and float(bbox[3] or 0.0) > 0.0


def _is_fullscreen_container(bbox: List[float], viewport_width: int, viewport_height: int) -> bool:
    if not _has_bbox(bbox) or not viewport_width or not viewport_height:
        return False
    return bbox[2] >= viewport_width * 0.9 and bbox[3] >= viewport_height * 0.9


def _meaningful_name(text: str, page_title: str) -> bool:
    normalized = " ".join(_as_str(text).split())
    if len(normalized) < 2:
        return False
    if normalized.lower() == _as_str(page_title).strip().lower():
        return False
    if normalized.lower() in {"frontend", "root", "document"}:
        return False
    return True


def _ax_property_bool(node: Mapping[str, Any], name: str) -> bool:
    properties = node.get("properties")
    if not isinstance(properties, list):
        return False
    for prop in properties:
        if not isinstance(prop, Mapping) or prop.get("name") != name:
            continue
        value = prop.get("value")
        return _as_bool(_ax_value(value), False)
    return False


def _string_at(strings: List[Any], value: Any) -> str:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return _as_str(value)
    if 0 <= index < len(strings):
        return _as_str(strings[index])
    return ""


def _decode_attrs(strings: List[Any], attrs: Any) -> Dict[str, str]:
    if not isinstance(attrs, list):
        return {}
    decoded: Dict[str, str] = {}
    for i in range(0, len(attrs) - 1, 2):
        key = _string_at(strings, attrs[i]).lower()
        value = _string_at(strings, attrs[i + 1])
        if key:
            decoded[key] = value
    return decoded


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_int(value: Any, default: int = 0) -> int:
    return int(_as_float(value, default=float(default)))


def _ax_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value") or value.get("name") or value.get("text")
    return value
