"""Policy-safe DOM graph builder for future GNN encoders."""

from __future__ import annotations

from typing import Any, Mapping


FEATURE_DIM = 14


class DOMGraphBuilder:
    def __init__(self, max_nodes: int = 256) -> None:
        self.max_nodes = int(max_nodes)

    def build(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
        candidate_list = [item for item in candidates if isinstance(item, Mapping)]
        dom_nodes = self._dom_nodes(observation)
        if not dom_nodes:
            dom_nodes = [self._candidate_node(candidate, index) for index, candidate in enumerate(candidate_list)]
        nodes = dom_nodes[: self.max_nodes]
        if not nodes:
            nodes = [{"tag": "", "role": "", "depth": 0, "sibling_index": 0, "text": ""}]

        node_features = [self._node_features(node) for node in nodes]
        edge_index = self._edges(nodes)
        candidate_node_indices = self._candidate_indices(nodes, candidate_list)
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "candidate_node_indices": candidate_node_indices,
            "node_metadata": [self._node_metadata(node) for node in nodes],
            "graph_stats": {
                "node_count": len(nodes),
                "edge_count": len(edge_index),
                "candidate_count": len(candidate_list),
                "matched_candidate_count": sum(1 for index in candidate_node_indices if index is not None),
            },
        }

    def _dom_nodes(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        dom = observation.get("dom_object") or observation.get("dom") or {}
        roots = dom if isinstance(dom, list) else [dom] if isinstance(dom, Mapping) else []
        nodes: list[dict[str, Any]] = []
        for root in roots:
            self._walk(root, nodes, parent_index=None, depth=0, sibling_index=0)
        return nodes

    def _walk(
        self,
        node: Any,
        nodes: list[dict[str, Any]],
        *,
        parent_index: int | None,
        depth: int,
        sibling_index: int,
    ) -> None:
        if not isinstance(node, Mapping) or len(nodes) >= self.max_nodes:
            return
        current_index = len(nodes)
        row = {
            "tag": str(node.get("tag") or node.get("tagName") or node.get("nodeName") or "").lower(),
            "role": str(node.get("role") or node.get("aria_role") or "").lower(),
            "visible": bool(node.get("visible", True)),
            "enabled": bool(node.get("enabled", True)),
            "clickable": bool(node.get("clickable") or node.get("is_interactive")),
            "bbox": node.get("bbox") or node.get("bounding_box") or {},
            "text": str(node.get("text") or node.get("name") or node.get("aria_label") or ""),
            "input_type": str(node.get("input_type") or node.get("type") or ""),
            "href": str(node.get("href") or ""),
            "depth": depth,
            "sibling_index": sibling_index,
            "parent_index": parent_index,
        }
        nodes.append(row)
        children = node.get("children") or node.get("childNodes") or []
        if isinstance(children, list):
            for child_index, child in enumerate(children):
                self._walk(child, nodes, parent_index=current_index, depth=depth + 1, sibling_index=child_index)

    def _candidate_node(self, candidate: Mapping[str, Any], index: int) -> dict[str, Any]:
        return {
            "tag": str(candidate.get("tag") or "").lower(),
            "role": str(candidate.get("role") or "").lower(),
            "visible": bool(candidate.get("visible", True)),
            "enabled": bool(candidate.get("enabled", True)),
            "clickable": bool(candidate.get("clickable") or candidate.get("is_interactive")),
            "bbox": candidate.get("bbox") or {},
            "text": str(candidate.get("text") or candidate.get("name") or ""),
            "input_type": str(candidate.get("input_type") or candidate.get("type") or ""),
            "href": str(candidate.get("href") or ""),
            "depth": int(candidate.get("dom_depth", candidate.get("depth", 0)) or 0),
            "sibling_index": index,
            "parent_index": None,
        }

    def _node_features(self, node: Mapping[str, Any]) -> list[float]:
        x, y, w, h = _bbox(node.get("bbox"))
        tag = _stable_bucket(str(node.get("tag") or ""), 32)
        role = _stable_bucket(str(node.get("role") or ""), 32)
        text = str(node.get("text") or "")
        tag_name = str(node.get("tag") or "").lower()
        role_name = str(node.get("role") or "").lower()
        return [
            tag / 31.0,
            role / 31.0,
            _bool(node.get("visible", True)),
            _bool(node.get("enabled", True)),
            _bool(node.get("clickable", False)),
            min(1.0, x / 2000.0),
            min(1.0, y / 2000.0),
            min(1.0, w / 2000.0),
            min(1.0, h / 2000.0),
            min(1.0, len(text) / 512.0),
            _bool(text.strip()),
            _bool(tag_name in {"input", "textarea", "select"} or role_name in {"textbox", "searchbox"}),
            _bool(node.get("href")),
            min(1.0, float(node.get("depth", 0) or 0) / 64.0),
        ]

    def _edges(self, nodes: list[Mapping[str, Any]]) -> list[list[int]]:
        edges: list[list[int]] = []
        last_by_parent: dict[int | None, int] = {}
        for index, node in enumerate(nodes):
            parent = node.get("parent_index")
            if isinstance(parent, int) and 0 <= parent < len(nodes):
                edges.append([parent, index])
                edges.append([index, parent])
            previous = last_by_parent.get(parent)
            if previous is not None:
                edges.append([previous, index])
                edges.append([index, previous])
            last_by_parent[parent] = index
        return edges

    def _candidate_indices(self, nodes: list[Mapping[str, Any]], candidates: list[Mapping[str, Any]]) -> list[int | None]:
        result: list[int | None] = []
        for candidate_index, candidate in enumerate(candidates):
            key = _candidate_match_key(candidate)
            matched = None
            for node_index, node in enumerate(nodes):
                if key and key == _candidate_match_key(node):
                    matched = node_index
                    break
            if matched is None and candidate_index < len(nodes):
                matched = candidate_index
            result.append(matched)
        return result

    def _node_metadata(self, node: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tag": str(node.get("tag") or ""),
            "role": str(node.get("role") or ""),
            "depth": int(node.get("depth", 0) or 0),
            "sibling_index": int(node.get("sibling_index", 0) or 0),
            "text_length": len(str(node.get("text") or "")),
        }


def _candidate_match_key(value: Mapping[str, Any]) -> str:
    return str(value.get("bid") or value.get("backend_node_id") or value.get("node_id") or value.get("text") or value.get("name") or "")


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        return (
            _float(value.get("x")),
            _float(value.get("y")),
            _float(value.get("width", value.get("w"))),
            _float(value.get("height", value.get("h"))),
        )
    if isinstance(value, list) and len(value) >= 4:
        return _float(value[0]), _float(value[1]), _float(value[2]), _float(value[3])
    return 0.0, 0.0, 0.0, 0.0


def _stable_bucket(value: str, modulo: int) -> int:
    return sum(ord(char) for char in value) % modulo if value else 0


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
