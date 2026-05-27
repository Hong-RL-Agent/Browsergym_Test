"""BrowserGym observation adapter for J.A.W.S.

This module converts BrowserGym's raw observation dictionary into a stable
raw observation schema consumed by J.A.W.S models and encoders.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


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


class BrowserGymObservationAdapter:
    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = max_candidates

    def adapt(
        self,
        obs: Optional[Mapping[str, Any]],
        info: Optional[Mapping[str, Any]] = None,
        previous_obs: Optional[Mapping[str, Any]] = None,
        history: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        obs = obs or {}
        info = info or {}
        history = history or {}

        url = _as_str(obs.get("url") or _active_page_value(obs, "open_pages_urls"))
        title = _as_str(_active_page_value(obs, "open_pages_titles"))
        screenshot = obs.get("screenshot")
        viewport_width, viewport_height = _viewport_size(screenshot, info)
        page_text_length = _page_text_length(obs)
        dom_node_count = _dom_node_count(obs.get("dom_object"))
        elapsed_time = _as_float(obs.get("elapsed_time"), default=0.0)

        previous_url = _nested_get(previous_obs, ("page_state", "url"), "")
        url_changed = bool(previous_url and previous_url != url)
        last_action_error = bool(obs.get("last_action_error"))

        candidate_elements = self._extract_candidates(obs)

        return {
            "page_state": {
                "url": url,
                "title": title,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
                "page_text_length": page_text_length,
                "dom_node_count": dom_node_count,
                "elapsed_time": elapsed_time,
            },
            "candidate_elements": candidate_elements,
            "runtime_signals": {
                "url_changed": url_changed,
                "last_action_error": last_action_error,
                "elapsed_time": elapsed_time,
            },
            "layout_signals": {
                "layout_overlap_count": _layout_overlap_count(candidate_elements),
            },
            "history": {
                "step_index": int(history.get("step_index", 0) or 0),
                "previous_action_type": _as_str(history.get("previous_action_type", "noop")),
                "no_change_steps": int(history.get("no_change_steps", 0) or 0),
            },
        }

    def _extract_candidates(self, obs: Mapping[str, Any]) -> List[Dict[str, Any]]:
        properties = _index_extra_properties(obs.get("extra_element_properties"))
        nodes = _axtree_nodes(obs.get("axtree_object"))
        candidates: List[Dict[str, Any]] = []

        for node in nodes:
            if not isinstance(node, Mapping):
                continue

            role = _normalize_role(node.get("role"))
            if role not in INTERACTIVE_ROLES:
                continue

            bid = _node_bid(node)
            if not bid:
                continue

            prop = properties.get(str(bid), {})
            name = _as_str(_ax_value(node.get("name")) or _ax_value(node.get("text")) or prop.get("text"))
            tag = _as_str(_ax_value(node.get("tag")) or _ax_value(node.get("tagName")) or prop.get("tag") or prop.get("tagName"))
            bbox = _normalize_bbox(prop.get("bbox") or prop.get("bounding_box") or node.get("bbox"))
            default_visibility = 1.0 if prop and _as_bool(prop.get("visible"), True) else 0.0
            visibility = _as_float(prop.get("visibility"), default=default_visibility)
            visible = _as_bool(prop.get("visible"), visibility > 0.0)
            enabled = not _as_bool(node.get("disabled") or prop.get("disabled"), False)
            clickable = _as_bool(prop.get("clickable"), role in {"button", "link", "checkbox", "radio", "menuitem", "tab"})
            text_length = len(name)

            candidates.append(
                {
                    "bid": str(bid),
                    "role": role,
                    "name": name,
                    "tag": tag,
                    "visible": visible,
                    "enabled": enabled,
                    "clickable": clickable,
                    "visibility": max(0.0, min(1.0, visibility)),
                    "bbox": bbox,
                    "has_text": text_length > 0,
                    "text_length": text_length,
                    "has_data_bug_id": _has_attr(prop, "data-bug-id", "data_bug_id"),
                    "has_data_testid": _has_attr(prop, "data-testid", "data_testid"),
                }
            )

        candidates.sort(
            key=lambda item: (
                bool(item.get("clickable")),
                float(item.get("visibility", 0.0) or 0.0),
                bool(item.get("enabled")),
                bool(item.get("visible")),
            ),
            reverse=True,
        )
        return candidates[: self.max_candidates]


def _active_page_value(obs: Mapping[str, Any], key: str) -> Any:
    values = obs.get(key)
    if not isinstance(values, (list, tuple)) or not values:
        return ""
    index = _as_int(obs.get("active_page_index"), default=0)
    if 0 <= index < len(values):
        return values[index]
    return values[0]


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
                for alias in ("bid", "nodeId", "backendDOMNodeId"):
                    if alias in value:
                        indexed[str(value[alias])] = value
        return indexed

    if isinstance(extra, list):
        for item in extra:
            if not isinstance(item, Mapping):
                continue
            for alias in ("bid", "nodeId", "backendDOMNodeId"):
                if alias in item:
                    indexed[str(item[alias])] = item
    return indexed


def _node_bid(node: Mapping[str, Any]) -> str:
    for key in ("browsergym_id", "bid", "nodeId", "backendDOMNodeId"):
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
    for source in (info,):
        width = source.get("viewport_width") or source.get("width")
        height = source.get("viewport_height") or source.get("height")
        if width and height:
            return int(width), int(height)
    shape = getattr(screenshot, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return 0, 0


def _page_text_length(obs: Mapping[str, Any]) -> int:
    total = 0
    for node in _axtree_nodes(obs.get("axtree_object")):
        if isinstance(node, Mapping):
            total += len(_as_str(_ax_value(node.get("name")) or _ax_value(node.get("text"))))
    goal = obs.get("goal")
    if isinstance(goal, str):
        total += len(goal)
    return total


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


def _nested_get(mapping: Optional[Mapping[str, Any]], path: tuple[str, ...], default: Any) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key)
    return default if value is None else value


def _has_attr(mapping: Mapping[str, Any], *names: str) -> bool:
    attrs = mapping.get("attributes")
    for name in names:
        if mapping.get(name):
            return True
        if isinstance(attrs, Mapping) and attrs.get(name):
            return True
    return False


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
