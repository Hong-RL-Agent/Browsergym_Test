"""Training-only productive-context classification; never imported by production controller."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def semantic_fingerprint(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state", {}), Mapping) else {}
    candidates = observation.get("candidate_elements", []) or []
    stable_candidates = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        stable_candidates.append((str(candidate.get("canonical_target_key") or candidate.get("bid") or candidate.get("id") or ""), str(candidate.get("role") or ""), str(candidate.get("tag") or "")))
    payload = {"url": page.get("url") or observation.get("url") or "", "route": page.get("route") or "", "text": str(page.get("page_text_sample") or "")[:300], "candidates": stable_candidates}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def target_fingerprint(observation: Mapping[str, Any]) -> frozenset[str]:
    values = set()
    for candidate in observation.get("candidate_elements", []) or []:
        if isinstance(candidate, Mapping):
            values.add(str(candidate.get("canonical_target_key") or candidate.get("bid") or candidate.get("id") or candidate.get("role") or ""))
    return frozenset(value for value in values if value)


def evidence_types(anomalies: Sequence[Mapping[str, Any]] | None, info: Mapping[str, Any] | None = None) -> list[str]:
    result: set[str] = set()
    for anomaly in anomalies or []:
        rule = str(anomaly.get("rule_id") or anomaly.get("rule") or anomaly.get("evidence", {}).get("rule_id") or "").upper()
        evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence", {}), Mapping) else {}
        text = json.dumps(anomaly, default=str).lower()
        if "console" in text or "r03" in rule or "r07" in rule:
            result.add("console_error")
        if "network" in text or "r13" in rule or "r14" in rule:
            result.add("network_failure")
        if "layout" in text or "r05" in rule or "r06" in rule:
            result.add("layout")
        if "no_effect" in text or "r01" in rule:
            result.add("no_effect")
        if rule or evidence:
            result.add(rule or "detector_evidence")
    if info and info.get("last_action_error"):
        result.add("action_error")
    return sorted(result)


def classify_context(before: Mapping[str, Any], after: Mapping[str, Any], anomalies: Sequence[Mapping[str, Any]] | None, info: Mapping[str, Any] | None = None, previous_zero_gain: bool = False) -> dict[str, Any]:
    before_state = semantic_fingerprint(before)
    after_state = semantic_fingerprint(after)
    before_targets = target_fingerprint(before)
    after_targets = target_fingerprint(after)
    evidence = evidence_types(anomalies, info)
    candidate_present = any(bool(item.get("candidate") or item.get("is_candidate") or str(item.get("status") or "") == "candidate") for item in anomalies or [] if isinstance(item, Mapping))
    confirmed = any(bool(item.get("confirmed")) or str(item.get("status") or "") == "confirmed" for item in anomalies or [] if isinstance(item, Mapping))
    state_changed = before_state != after_state
    new_targets = bool(after_targets - before_targets)
    tier = "TIER4_CONTROL"
    if candidate_present or confirmed:
        tier = "TIER1_CANDIDATE_CONTEXT"
    elif evidence:
        tier = "TIER2_DETECTOR_OPPORTUNITY"
    elif state_changed and after_targets:
        tier = "TIER3_PRODUCTIVE_FUNCTIONAL_REGION"
    if previous_zero_gain and not evidence and not candidate_present and not state_changed and not new_targets:
        return {"admit": False, "skip_reason": "REPEATED_ZERO_GAIN_NO_CHANGE", "context_tier": tier, "state_changed": state_changed, "new_target": new_targets, "evidence_types": evidence}
    return {"admit": tier != "TIER4_CONTROL", "skip_reason": "" if tier != "TIER4_CONTROL" else "NO_TIER1_TO_TIER3_SIGNAL", "context_tier": tier, "state_changed": state_changed, "new_target": new_targets, "evidence_types": evidence, "candidate_present": candidate_present, "confirmed": confirmed, "before_state": before_state, "after_state": after_state, "before_targets": sorted(before_targets), "after_targets": sorted(after_targets)}
