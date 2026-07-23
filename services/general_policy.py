"""Site-independent exploration and fault-discovery policy overlay."""
from __future__ import annotations
from typing import Any, Mapping, MutableMapping
from models.action_space import ActionSpace

RISKY_KEYWORDS = ("delete", "remove", "pay", "purchase", "send", "withdraw", "탈퇴", "삭제", "결제", "송금")
CORE_KEYWORDS = ("login", "submit", "save", "search", "next", "confirm", "로그인", "제출", "저장", "검색", "다음", "확인")

def select_policy_action(action_space: ActionSpace, observation: Mapping[str, Any], history: MutableMapping[str, Any], model_action_id: int, mode: str = "coverage", allow_destructive_actions: bool = False) -> int:
    counts = history.setdefault("action_type_counts", {})
    step = int(history.get("step_index", 0) or 0)
    candidates = observation.get("candidate_elements", []) or []
    if step == 0 and not history.get("desktop_inspected"):
        history["desktop_inspected"] = True
        return action_space.encode("inspect_layout", 0)
    if int(counts.get("inspect_console", 0) or 0) == 0 and step >= 3:
        return action_space.encode("inspect_console", 0)
    if int(counts.get("inspect_network", 0) or 0) == 0 and step >= 4:
        return action_space.encode("inspect_network", 0)
    ranked: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping) or not bool(candidate.get("visible", True)):
            continue
        text = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "role", "type")).lower()
        stable_key = str(candidate.get("bid") or candidate.get("selector") or text or index)
        destructive = any(keyword in text for keyword in RISKY_KEYWORDS)
        if destructive and not allow_destructive_actions:
            continue
        role = str(candidate.get("role") or candidate.get("type") or "").lower()
        if mode in {"fault-discovery", "fault"} and role in {"textbox", "input", "combobox", "textarea"} and stable_key not in history.setdefault("fault_inputs", set()):
            ranked.append((120.0, index, "fill_input"))
        novelty = 60.0 if stable_key not in history.setdefault("visited_candidates", set()) else -25.0
        business = 25.0 if any(keyword in text for keyword in CORE_KEYWORDS) else 0.0
        ranked.append((novelty + business + (15.0 if destructive else 0.0), index, "click_element"))
    if ranked:
        _, index, action_type = max(ranked, key=lambda row: row[0])
        return action_space.encode(action_type, index)
    # The live exploration view represents a normal desktop browser window.
    # Do not allow a learned/fallback action to turn that preview into a
    # mobile-sized viewport.
    model_action = action_space.decode(model_action_id)
    if str(model_action.get("action_type") or "") == "change_viewport_mobile":
        return action_space.encode("inspect_layout", 0)
    return model_action_id

def update_general_history(history: MutableMapping[str, Any], observation: Mapping[str, Any], action: Mapping[str, Any]) -> None:
    action_type = str(action.get("action_type") or "")
    counts = history.setdefault("action_type_counts", {})
    counts[action_type] = int(counts.get(action_type, 0) or 0) + 1
    history["step_index"] = int(history.get("step_index", 0) or 0) + 1
    if action_type == "change_viewport_mobile": history["mobile_viewport_seen"] = True
    candidates = observation.get("candidate_elements", []) or []
    index = int(action.get("candidate_index", 0) or 0)
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        candidate = candidates[index]
        key = str(candidate.get("bid") or candidate.get("selector") or candidate.get("name") or candidate.get("text") or index)
        history.setdefault("visited_candidates", set()).add(key)
        if action_type == "fill_input": history.setdefault("fault_inputs", set()).add(key)
