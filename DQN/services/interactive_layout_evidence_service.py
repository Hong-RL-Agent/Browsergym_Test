"""Side-effect-free live layout evidence extraction for the R06 detector.

The collector only reads DOM geometry and browser state.  It deliberately does
not click, scroll, focus, resize, navigate, or synthesize ground truth.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_LIVE_LAYOUT_SCRIPT = r"""
(payload) => {
  const d = payload || {};
  const descriptor = d.target || {};
  const rectOf = (el) => {
    const r = el.getBoundingClientRect();
    return {x: r.x, y: r.y, width: r.width, height: r.height,
      right: r.right, bottom: r.bottom};
  };
  const area = (r) => Math.max(0, r.width) * Math.max(0, r.height);
  const intersect = (a, b) => {
    const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
    const right = Math.min(a.right, b.right), bottom = Math.min(a.bottom, b.bottom);
    return {x, y, right, bottom, width: Math.max(0, right - x), height: Math.max(0, bottom - y)};
  };
  const boolAttr = (el, name) => el.hasAttribute(name) && el.getAttribute(name) !== 'false';
  const roleOf = (el) => (el.getAttribute('role') || el.tagName || '').toLowerCase();
  const textOf = (el) => (el.innerText || el.textContent || '').trim();
  const isClipStyle = (s) => ['hidden', 'clip', 'auto', 'scroll'].includes(s.overflowX) ||
    ['hidden', 'clip', 'auto', 'scroll'].includes(s.overflowY);
  const isScrollable = (s, el) => ({
    x: ['auto', 'scroll'].includes(s.overflowX) && el.scrollWidth > el.clientWidth,
    y: ['auto', 'scroll'].includes(s.overflowY) && el.scrollHeight > el.clientHeight,
  });
  const before = {
    href: location.href,
    active: document.activeElement,
    scrollX: window.scrollX,
    scrollY: window.scrollY,
  };

  const all = Array.from(document.querySelectorAll('*'));
  const wantedText = String(descriptor.text || descriptor.name || '').trim();
  const wantedRole = String(descriptor.role || '').toLowerCase();
  const wantedTag = String(descriptor.tag || '').toLowerCase();
  const wantedBox = descriptor.bbox || [];
  let target = null;
  const selector = String(descriptor.selector_hint || '').trim();
  if (selector && /^[#.[\w\-='": >]+$/.test(selector)) {
    try { target = document.querySelector(selector); } catch (_) {}
  }
  if (!target && descriptor.element_id) target = document.getElementById(String(descriptor.element_id));
  if (!target && descriptor.testid) {
    target = document.querySelector('[data-testid="' + CSS.escape(String(descriptor.testid)) + '"]');
  }
  if (!target && wantedBox.length >= 4) {
    const candidates = all.map((el) => {
      const r = el.getBoundingClientRect();
      const role = roleOf(el);
      const text = textOf(el);
      const boxDelta = Math.abs(r.x - Number(wantedBox[0])) + Math.abs(r.y - Number(wantedBox[1])) +
        Math.abs(r.width - Number(wantedBox[2])) + Math.abs(r.height - Number(wantedBox[3]));
      const roleBonus = wantedRole && role === wantedRole ? 0 : 40;
      const tagBonus = wantedTag && String(el.tagName || '').toLowerCase() === wantedTag ? 0 : 20;
      const textBonus = wantedText && text && (text === wantedText || text.includes(wantedText)) ? 0 : 30;
      return {el, score: boxDelta + roleBonus + tagBonus + textBonus};
    }).filter((item) => item.el.getBoundingClientRect().width > 0 && item.el.getBoundingClientRect().height > 0)
      .sort((a, b) => a.score - b.score);
    target = candidates.length ? candidates[0].el : null;
  }
  if (!target) {
    return {evidence_status: 'UNAVAILABLE', unavailable_reason: 'TARGET_NOT_RESOLVED', target_resolved: false,
      mutation_guard: {url_unchanged: true, scroll_unchanged: true, active_element_unchanged: true}};
  }

  const viewport = {x: 0, y: 0, width: window.innerWidth, height: window.innerHeight,
    right: window.innerWidth, bottom: window.innerHeight};
  const documentElement = document.documentElement;
  const body = document.body;
  const documentScrollWidth = Math.max(documentElement ? documentElement.scrollWidth : 0, body ? body.scrollWidth : 0);
  const documentScrollHeight = Math.max(documentElement ? documentElement.scrollHeight : 0, body ? body.scrollHeight : 0);
  const documentClientWidth = documentElement ? documentElement.clientWidth : window.innerWidth;
  const documentClientHeight = documentElement ? documentElement.clientHeight : window.innerHeight;
  const documentCanScrollX = documentScrollWidth > documentClientWidth;
  const documentCanScrollY = documentScrollHeight > documentClientHeight;
  const targetBox = rectOf(target);
  let visible = intersect(targetBox, viewport);
  const ancestors = [];
  let effective = null;
  let node = target.parentElement;
  while (node) {
    const style = getComputedStyle(node);
    const box = rectOf(node);
    const scrollable = isScrollable(style, node);
    const clipping = isClipStyle(style);
    ancestors.push({tag: node.tagName.toLowerCase(), role: roleOf(node), bbox: box,
      overflow: style.overflow, overflow_x: style.overflowX, overflow_y: style.overflowY,
      scroll_width: node.scrollWidth, client_width: node.clientWidth,
      scroll_height: node.scrollHeight, client_height: node.clientHeight,
      scroll_left: node.scrollLeft, scroll_top: node.scrollTop,
      position: style.position, clipping, scrollable});
    if (!effective && clipping && area(intersect(targetBox, box)) < area(targetBox)) effective = node;
    if (node === document.body) break;
    node = node.parentElement;
  }
  for (const ancestor of ancestors) {
    if (ancestor.clipping) visible = intersect(visible, ancestor.bbox);
  }
  const clipStyle = effective ? getComputedStyle(effective) : null;
  const clipBox = effective ? rectOf(effective) : null;
  const clip = clipBox || viewport;
  const blockingClip = !!effective && ((clipStyle.overflowX === 'hidden' || clipStyle.overflowX === 'clip' ||
    clipStyle.overflowY === 'hidden' || clipStyle.overflowY === 'clip') &&
    area(intersect(targetBox, clipBox)) < area(targetBox));
  const scrollNode = effective || target;
  const scrollStyle = getComputedStyle(scrollNode);
  const scrollable = isScrollable(scrollStyle, scrollNode);
  const scrollableAncestor = ancestors.find((ancestor) => ancestor.scrollable && (ancestor.scrollable.x || ancestor.scrollable.y));
  const ancestorReachableX = !!scrollableAncestor && !!scrollableAncestor.scrollable.x && !blockingClip;
  const ancestorReachableY = !!scrollableAncestor && !!scrollableAncestor.scrollable.y && !blockingClip;
  const targetDocumentBox = {
    left: targetBox.x + window.scrollX, top: targetBox.y + window.scrollY,
    right: targetBox.right + window.scrollX, bottom: targetBox.bottom + window.scrollY,
  };
  const documentReachableX = !blockingClip && documentCanScrollX &&
    targetDocumentBox.left >= 0 && targetDocumentBox.right <= documentScrollWidth;
  const documentReachableY = !blockingClip && documentCanScrollY &&
    targetDocumentBox.top >= 0 && targetDocumentBox.bottom <= documentScrollHeight;
  const viewportVisible = visible.width > 0 && visible.height > 0 &&
    targetBox.x >= viewport.x && targetBox.y >= viewport.y &&
    targetBox.right <= viewport.right && targetBox.bottom <= viewport.bottom;
  const reachabilitySource = viewportVisible ? 'CURRENT_VIEWPORT' :
    (blockingClip ? 'UNREACHABLE_CLIPPING_ANCESTOR' :
      (ancestorReachableX || ancestorReachableY ? 'SCROLLABLE_ANCESTOR' :
        (documentReachableX || documentReachableY ? 'DOCUMENT_SCROLL' : 'UNREACHABLE_DOCUMENT_BOUNDS')));
  const visibleRatio = area(targetBox) ? area(visible) / area(targetBox) : 0;
  const clippedRatio = Math.max(0, 1 - visibleRatio);
  const points = [];
  if (visible.width > 0 && visible.height > 0) {
    const ix = Math.min(4, Math.max(1, visible.width * 0.1));
    const iy = Math.min(4, Math.max(1, visible.height * 0.1));
    points.push([visible.x + visible.width / 2, visible.y + visible.height / 2]);
    points.push([visible.x + ix, visible.y + iy]);
    points.push([visible.right - ix, visible.y + iy]);
    points.push([visible.x + ix, visible.bottom - iy]);
    points.push([visible.right - ix, visible.bottom - iy]);
  }
  let validHits = 0, blockedHits = 0;
  const hitElements = [];
  for (const point of points) {
    const hit = document.elementFromPoint(point[0], point[1]);
    const valid = !!hit && (hit === target || target.contains(hit));
    if (valid) validHits += 1; else if (hit) blockedHits += 1;
    hitElements.push(hit ? {tag: hit.tagName.toLowerCase(), role: roleOf(hit), text: textOf(hit).slice(0, 120)} : null);
  }
  const style = getComputedStyle(target);
  const role = roleOf(target);
  const nativeInteractive = ['a', 'button', 'input', 'textarea', 'select', 'option'].includes(target.tagName.toLowerCase());
  const disabled = boolAttr(target, 'disabled') || boolAttr(target, 'aria-disabled');
  const tabindex = target.getAttribute('tabindex');
  const keyboardReachable = !disabled && (tabindex !== null ? Number(tabindex) >= 0 : nativeInteractive ||
    ['button', 'link', 'textbox', 'searchbox', 'checkbox', 'radio', 'combobox', 'menuitem', 'tab', 'option'].includes(role));
  const after = {href: location.href, active: document.activeElement, scrollX: window.scrollX, scrollY: window.scrollY};
  return {
    evidence_status: 'AVAILABLE', target_resolved: true,
    target_bbox: [targetBox.x, targetBox.y, targetBox.width, targetBox.height],
    visible_bbox: [visible.x, visible.y, visible.width, visible.height],
    viewport_width: viewport.width, viewport_height: viewport.height,
    window_scroll_x: window.scrollX, window_scroll_y: window.scrollY,
    document_scroll_width: documentScrollWidth, document_scroll_height: documentScrollHeight,
    document_client_width: documentClientWidth, document_client_height: documentClientHeight,
    document_can_scroll_x: documentCanScrollX, document_can_scroll_y: documentCanScrollY,
    target_document_bbox: [targetDocumentBox.left, targetDocumentBox.top,
      targetBox.width, targetBox.height],
    document_reachable_x: documentReachableX, document_reachable_y: documentReachableY,
    reachable_via_document_scroll: documentReachableX || documentReachableY,
    reachability_source: reachabilitySource,
    visible_ratio: Number(visibleRatio.toFixed(4)), clipping_ratio: Number(clippedRatio.toFixed(4)),
    clipped_left: targetBox.x < clip.x, clipped_right: targetBox.right > clip.right,
    clipped_top: targetBox.y < clip.y, clipped_bottom: targetBox.bottom > clip.bottom,
    clipped_left_px: Math.max(0, clip.x - targetBox.x), clipped_right_px: Math.max(0, targetBox.right - clip.right),
    clipped_top_px: Math.max(0, clip.y - targetBox.y), clipped_bottom_px: Math.max(0, targetBox.bottom - clip.bottom),
    viewport_overflow: targetBox.x < viewport.x || targetBox.y < viewport.y || targetBox.right > viewport.right || targetBox.bottom > viewport.bottom,
    container_overflow: !!effective && (targetBox.x < clip.x || targetBox.y < clip.y || targetBox.right > clip.right || targetBox.bottom > clip.bottom),
    effective_clipping_ancestor: effective ? {tag: effective.tagName.toLowerCase(), role: roleOf(effective), bbox: clipBox,
      overflow: clipStyle.overflow, overflow_x: clipStyle.overflowX, overflow_y: clipStyle.overflowY,
      scroll_width: effective.scrollWidth, client_width: effective.clientWidth,
      scroll_height: effective.scrollHeight, client_height: effective.clientHeight,
      scroll_left: effective.scrollLeft, scroll_top: effective.scrollTop, position: clipStyle.position} : null,
    ancestors,
    overflow: clipStyle ? clipStyle.overflow : style.overflow,
    overflow_x: clipStyle ? clipStyle.overflowX : style.overflowX,
    overflow_y: clipStyle ? clipStyle.overflowY : style.overflowY,
    scroll_width: scrollNode.scrollWidth, client_width: scrollNode.clientWidth,
    scroll_height: scrollNode.scrollHeight, client_height: scrollNode.clientHeight,
    scroll_left: scrollNode.scrollLeft, scroll_top: scrollNode.scrollTop,
    intentional_scrollable: !!(scrollable.x || scrollable.y), can_scroll_x: scrollable.x, can_scroll_y: scrollable.y,
    reachable_via_scroll_x: ancestorReachableX || documentReachableX,
    reachable_via_scroll_y: ancestorReachableY || documentReachableY,
    ancestor_scroll_reachable_x: ancestorReachableX, ancestor_scroll_reachable_y: ancestorReachableY,
    blocking_clipping_ancestor: blockingClip,
    safe_point_count: points.length, valid_hit_count: validHits, blocked_hit_count: blockedHits,
    hit_test_valid: validHits > 0, hit_test_blocked: blockedHits > 0 && validHits === 0, hit_test_partial: validHits > 0 && blockedHits > 0,
    hit_elements: hitElements, pointer_events: style.pointerEvents,
    display: style.display, visibility: style.visibility, opacity: style.opacity,
    rendered_visible: style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) > 0 && targetBox.width > 0 && targetBox.height > 0,
    enabled: !disabled, disabled, aria_disabled: target.getAttribute('aria-disabled') === 'true',
    semantic_clickable: !!descriptor.clickable || nativeInteractive || ['button', 'link', 'menuitem'].includes(role),
    browser_interactable_geometry: validHits > 0 && !disabled && style.pointerEvents !== 'none',
    tabindex, keyboard_reachable: keyboardReachable,
    text_content: textOf(target).slice(0, 2000), text_length: textOf(target).length,
    meaningful_text: textOf(target).length > 0, text_potentially_clipped: textOf(target).length > 0 && clippedRatio > 0,
    mutation_guard: {url_unchanged: before.href === after.href, scroll_unchanged: before.scrollX === after.scrollX && before.scrollY === after.scrollY,
      active_element_unchanged: before.active === after.active},
  };
}
"""


def collect_interactive_layout_evidence(page: Any, target: Mapping[str, Any]) -> dict[str, Any]:
    """Collect one target's live evidence without changing the page."""
    if page is None:
        return {"evidence_status": "UNAVAILABLE", "unavailable_reason": "PAGE_UNAVAILABLE", "target_resolved": False}
    try:
        result = page.evaluate(_LIVE_LAYOUT_SCRIPT, {"target": dict(target)})
        return result if isinstance(result, dict) else {"evidence_status": "UNAVAILABLE", "unavailable_reason": "INVALID_RESULT"}
    except Exception as exc:
        return {"evidence_status": "UNAVAILABLE", "unavailable_reason": type(exc).__name__, "target_resolved": False}


def enrich_observation_with_interactive_layout_evidence(page: Any, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a detector-only snapshot enriched with live evidence."""
    enriched = deepcopy(dict(observation))
    candidates = enriched.get("candidate_elements", [])
    if not isinstance(candidates, list):
        return enriched, {"status": "UNAVAILABLE", "count": 0, "available": 0}
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    viewport_width = float(page_state.get("viewport_width", 0) or 0)
    viewport_height = float(page_state.get("viewport_height", 0) or 0)
    available = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if not _likely_layout_relevant(candidate, viewport_width, viewport_height):
            evidence = {"evidence_status": "UNAVAILABLE", "unavailable_reason": "OUT_OF_SCOPE_CANDIDATE", "target_resolved": False}
        else:
            evidence = collect_interactive_layout_evidence(page, candidate)
        candidate["interactive_layout_evidence"] = evidence
        available += int(evidence.get("evidence_status") == "AVAILABLE")
    summary = {"status": "AVAILABLE" if available else "UNAVAILABLE", "count": len(candidates), "available": available,
               "unavailable": len(candidates) - available}
    enriched.setdefault("runtime_signals", {})["interactive_layout_evidence"] = summary
    return enriched, summary


def _likely_layout_relevant(candidate: Mapping[str, Any], viewport_width: float, viewport_height: float) -> bool:
    bbox = candidate.get("bbox") or []
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x, y, width, height = [float(value) for value in bbox[:4]]
            if x < 0 or y < 0 or x + width > viewport_width or y + height > viewport_height:
                return True
        except (TypeError, ValueError):
            pass
    role = str(candidate.get("role") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    return bool(
        candidate.get("clickable")
        or candidate.get("is_interactive")
        or candidate.get("is_layout_target")
        or candidate.get("is_chart_like")
        or role in {"button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox", "menuitem", "tab"}
        or tag in {"button", "a", "input", "textarea", "select", "option"}
        or len(str(candidate.get("text") or candidate.get("name") or "").strip()) >= 20
    )
