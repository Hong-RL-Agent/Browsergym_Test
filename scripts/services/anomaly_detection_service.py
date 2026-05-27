"""Observation-only anomaly heuristics for BrowserGym PPO training.

The detector is intentionally conservative. It should surface likely UI bugs
without flooding PPO reward/evaluation with weak generic signals.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping


ACTION_KEYWORDS = (
    "구매",
    "추가",
    "저장",
    "제출",
    "확인",
    "로그인",
    "가입",
    "장바구니",
    "buy",
    "add",
    "submit",
    "save",
    "login",
)
SUCCESS_TOKENS = ("success", "saved", "added", "complete", "완료", "성공", "저장됨", "추가됨")
BOOK_CONTEXT_TOKENS = ("추천", "recommended", "rc-card", "도서", "책", "book")


def detect_anomalies(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    action = action_info.get("action", action_info)
    action_type = str(action.get("action_type", action_info.get("action_type", "")))
    candidate_index = int(action.get("candidate_index", 0) or 0)
    episode_seen = action_info.get("episode_seen_anomalies", set())

    before_candidates = before_observation.get("candidate_elements", []) or []
    after_candidates = after_observation.get("candidate_elements", []) or []
    target = _candidate_at(before_candidates, candidate_index)
    before_url = _url(before_observation)
    after_url = _url(after_observation)
    error_message = _error_message(action_info)
    action_failed = bool(action.get("failed") or action_info.get("action_result", {}).get("failed"))
    last_action_error = bool(error_message) or action_failed
    no_url_change = before_url == after_url
    candidate_delta = abs(len(after_candidates) - len(before_candidates))
    text_delta = abs(_page_text_length(after_observation) - _page_text_length(before_observation))
    no_state_change = _state_signature(before_observation) == _state_signature(after_observation)
    success_signal = _has_success_signal(before_observation, after_observation)

    if action_type == "click_element" and target:
        visibility = float(target.get("visibility", 1.0) or 0.0)
        if visibility <= 0.0:
            if last_action_error or no_state_change:
                anomalies.append(_low_visibility(target, visibility, last_action_error))
        else:
            maybe_button = bool(target.get("clickable")) or str(target.get("role", "")).lower() in {
                "button",
                "link",
                "menuitem",
            }
            visible_enough = bool(target.get("visible", True)) or visibility > 0.2
            target_text = _target_text(target)
            has_keyword = _has_action_keyword(target_text)
            stable_page = no_url_change and text_delta <= 8 and candidate_delta <= 1

            if visible_enough and maybe_button and stable_page and not last_action_error and not success_signal:
                confidence = 0.8 if has_keyword and (no_state_change or text_delta <= 2) else 0.35
                if confidence >= 0.6:
                    anomalies.append(
                        {
                            "type": "button-no-response",
                            "confidence": confidence,
                            "evidence": {
                                "before_url": before_url,
                                "after_url": after_url,
                                "candidate_delta": candidate_delta,
                                "text_delta": text_delta,
                                "no_state_change": no_state_change,
                                "target_bid": target.get("bid"),
                                "target_role": target.get("role"),
                                "target_name": target.get("name"),
                                "target_text": target_text,
                                "target_visibility": visibility,
                                "has_action_keyword": has_keyword,
                            },
                        }
                    )

            if visibility < 0.3 and (last_action_error or no_state_change):
                anomalies.append(_low_visibility(target, visibility, last_action_error))

    overlap_count = int(after_observation.get("layout_signals", {}).get("layout_overlap_count", 0) or 0)
    if overlap_count >= 2 and "layout-overlap" not in episode_seen:
        viewport_width = int(after_observation.get("page_state", {}).get("viewport_width", 0) or 0)
        confidence = 0.8 if overlap_count >= 5 else 0.65
        if viewport_width and viewport_width <= 768:
            confidence = min(1.0, confidence + 0.1)
        anomalies.append(
            {
                "type": "layout-overlap",
                "confidence": confidence,
                "evidence": {
                    "layout_overlap_count": overlap_count,
                    "viewport_width": viewport_width,
                },
            }
        )

    if last_action_error and (error_message or action_failed):
        anomalies.append(
            {
                "type": "action-error",
                "confidence": 0.9,
                "evidence": {
                    "action_type": action_type,
                    "error": error_message,
                    "failed": action_failed,
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

    duplicate = _duplicated_rendering(after_observation, episode_seen)
    if duplicate:
        anomalies.append(duplicate)

    return [item for item in anomalies if float(item.get("confidence", 0.0) or 0.0) >= _threshold(item)]


def compute_anomaly_score(anomalies: List[Dict[str, Any]]) -> float:
    if not anomalies:
        return 0.0
    return min(1.0, sum(float(item.get("confidence", 0.0) or 0.0) for item in anomalies) / len(anomalies))


def _duplicated_rendering(
    observation: Mapping[str, Any],
    episode_seen: Any,
) -> Dict[str, Any] | None:
    if "duplicated-rendering" in episode_seen:
        return None
    candidates = observation.get("candidate_elements", []) or []
    names = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        text = _normalize_text(candidate.get("name") or candidate.get("text") or "")
        if len(text) >= 3:
            names.append(text)
    counts = Counter(names)
    repeated = [(name, count) for name, count in counts.items() if count >= 2]
    if not repeated:
        return None
    repeated.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    context_text = " ".join(names).lower()
    has_book_context = any(token.lower() in context_text for token in BOOK_CONTEXT_TOKENS)
    confidence = 0.6 + min(0.2, (repeated[0][1] - 2) * 0.05)
    if has_book_context:
        confidence += 0.15
    confidence = min(0.95, confidence)
    if confidence < 0.6:
        return None
    return {
        "type": "duplicated-rendering",
        "confidence": confidence,
        "evidence": {
            "repeated_text": repeated[0][0],
            "repeat_count": repeated[0][1],
            "has_book_context": has_book_context,
            "candidate_repeated_texts": [{"text": name, "count": count} for name, count in repeated[:5]],
        },
    }


def _low_visibility(target: Mapping[str, Any], visibility: float, last_action_error: bool) -> Dict[str, Any]:
    return {
        "type": "low-visibility-interaction",
        "confidence": 0.7 if last_action_error else 0.55,
        "evidence": {
            "target_bid": target.get("bid"),
            "target_name": target.get("name"),
            "target_text": _target_text(target),
            "visibility": visibility,
            "last_action_error": last_action_error,
        },
    }


def _threshold(anomaly: Mapping[str, Any]) -> float:
    anomaly_type = anomaly.get("type")
    if anomaly_type in {"button-no-response", "duplicated-rendering"}:
        return 0.6
    if anomaly_type == "low-visibility-interaction":
        return 0.55
    return 0.5


def _has_success_signal(
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
) -> bool:
    before_text = _candidate_text_blob(before_observation)
    after_text = _candidate_text_blob(after_observation)
    if before_text == after_text:
        return False
    new_text = after_text.replace(before_text, "")
    return any(token in new_text.lower() for token in SUCCESS_TOKENS)


def _error_message(action_info: Mapping[str, Any]) -> str:
    for key in ("action_error", "fallback_error", "error", "last_action_error_message"):
        value = action_info.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "null":
            return str(value).strip()
    fallback = action_info.get("fallback_info")
    if isinstance(fallback, Mapping):
        value = fallback.get("action_error") or fallback.get("error")
        if value is not None and str(value).strip() and str(value).lower() != "null":
            return str(value).strip()
    return ""


def _url(observation: Mapping[str, Any]) -> str:
    return str(observation.get("page_state", {}).get("url", "") or "")


def _page_text_length(observation: Mapping[str, Any]) -> int:
    return int(observation.get("page_state", {}).get("page_text_length", 0) or 0)


def _state_signature(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    page_state = observation.get("page_state", {})
    return (
        page_state.get("url"),
        page_state.get("title"),
        page_state.get("page_text_length"),
        page_state.get("dom_node_count"),
        len(observation.get("candidate_elements", []) or []),
    )


def _candidate_at(candidates: Any, index: int) -> Mapping[str, Any] | None:
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _target_text(candidate: Mapping[str, Any]) -> str:
    return _normalize_text(candidate.get("name") or candidate.get("text") or candidate.get("tag") or "")


def _has_action_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in ACTION_KEYWORDS)


def _candidate_text_blob(observation: Mapping[str, Any]) -> str:
    candidates = observation.get("candidate_elements", []) or []
    return " ".join(_target_text(candidate) for candidate in candidates if isinstance(candidate, Mapping))


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
