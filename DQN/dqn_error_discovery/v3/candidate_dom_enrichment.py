"""Generic read-only BrowserGym candidate to DOM metadata enrichment."""

from __future__ import annotations

from typing import Any, Mapping


def enrich_candidates_with_dom(page: Any, candidates: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if page is None:
        return [dict(c, canonical_dom_resolved=False, canonical_resolution_method="UNRESOLVED", canonical_resolution_confidence="UNRESOLVED") for c in candidates]
    nodes = page.evaluate("""() => [...document.querySelectorAll('*')].map(n => { const r=n.getBoundingClientRect(); const s=getComputedStyle(n); return {bid:n.getAttribute('bid')||'',id:n.id||'',tag:n.tagName.toLowerCase(),role:n.getAttribute('role')||'',name:n.getAttribute('aria-label')||n.getAttribute('name')||n.id||n.innerText||'',bbox:[r.x,r.y,r.width,r.height],visible:!!(r.width&&r.height&&s.visibility!=='hidden'&&s.display!=='none'),enabled:!n.disabled}; })""")
    result = []
    for candidate in candidates:
        row = dict(candidate)
        matches = _resolve(row, nodes)
        if len(matches) == 1:
            node = matches[0]
            row.update({"canonical_dom_resolved": True, "canonical_dom_tag": node.get("tag", ""), "canonical_dom_id": node.get("id", ""), "canonical_dom_role": node.get("role", ""), "canonical_dom_name": node.get("name", ""), "canonical_dom_bbox": node.get("bbox", []), "canonical_native_control_kind": "native_select" if node.get("tag") == "select" else "", "canonical_target_key": node.get("id") or node.get("bid") or _geometry_key(node.get("bbox", [])), "canonical_resolution_method": node.get("_method", "generic_dom_match"), "canonical_resolution_confidence": "HIGH" if node.get("_method") == "bid" else "MEDIUM", "canonical_dom_bid": node.get("bid", "")})
        else:
            row.update({"canonical_dom_resolved": False, "canonical_dom_tag": "", "canonical_dom_id": "", "canonical_dom_role": "", "canonical_dom_name": "", "canonical_dom_bbox": [], "canonical_native_control_kind": "", "canonical_target_key": "", "canonical_resolution_method": "AMBIGUOUS" if matches else "UNRESOLVED", "canonical_resolution_confidence": "LOW" if matches else "UNRESOLVED", "canonical_dom_bid": ""})
        result.append(row)
    return result


def _resolve(candidate: Mapping[str, Any], nodes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    bid = str(candidate.get("bid") or "")
    exact = [dict(n, _method="bid") for n in nodes if bid and str(n.get("bid") or "") == bid]
    if len(exact) == 1:
        return exact
    bbox = candidate.get("bbox") or []
    name = str(candidate.get("name") or candidate.get("aria_label") or candidate.get("text") or "").strip().lower()
    matches = []
    for node in nodes:
        nb = node.get("bbox") or []
        close = len(bbox) == 4 and len(nb) == 4 and all(abs(float(bbox[i]) - float(nb[i])) < 2 for i in range(4))
        same_name = bool(name) and name == str(node.get("name") or "").strip().lower()
        if close and (node.get("visible") or same_name):
            matches.append(dict(node, _method="geometry_name"))
    return matches


def _geometry_key(bbox: list[Any]) -> str:
    try:
        return "bbox:" + ":".join(str(round(float(v), 2)) for v in bbox)
    except (TypeError, ValueError):
        return ""
