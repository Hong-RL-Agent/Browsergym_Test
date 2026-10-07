"""Diagnostic-only analysis for temporal stability and layout candidates."""

from __future__ import annotations

from collections import Counter
import statistics
from typing import Any, Dict, Iterable, List, Mapping

from services.anomaly_detection_service import detect_anomalies
from services.temporal_evidence_service import ActionEvidenceWindow, EvidenceSample


LAYOUT_RULES = {"R05", "R06"}


def analyze_temporal_stability_window(
    *,
    before_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    temporal_window: ActionEvidenceWindow,
    detector_anomalies: List[Mapping[str, Any]],
    post_stable_samples: List[EvidenceSample],
    episode: int,
    step: int,
) -> Dict[str, Any]:
    samples = temporal_window.samples
    stable_sample_index = max(0, len(samples) - 1)
    detector_sample_index = stable_sample_index
    all_samples = samples + post_stable_samples
    signal = _signal_diagnostics(samples)
    post_drift = _post_stable_drift(samples[-1], post_stable_samples) if samples else {}
    bbox = _bbox_delta_metrics(all_samples)
    movement = _movement_counts(all_samples)
    detector_layout = [_layout_candidate_payload(item) for item in detector_anomalies if _is_layout_candidate(item)]
    per_sample_layout = [
        {
            "sample_index": index,
            "relative_ms": sample.relative_ms,
            "phase": "pre_stable" if index < stable_sample_index else "stable" if index == stable_sample_index else "post_stable",
            "layout_candidates": [
                _layout_candidate_payload(item)
                for item in detect_anomalies(before_observation, sample.observation, action_info)
                if _is_layout_candidate(item)
            ],
        }
        for index, sample in enumerate(all_samples)
    ]
    detector_keys = {item["signature"] for item in detector_layout if item.get("signature")}
    post_keys = {
        item["signature"]
        for row in per_sample_layout[stable_sample_index + 1 :]
        for item in row["layout_candidates"]
        if item.get("signature")
    }
    stable_keys = {
        item["signature"]
        for item in per_sample_layout[stable_sample_index]["layout_candidates"]
        if item.get("signature")
    } if per_sample_layout else set()
    transient_keys = sorted(key for key in detector_keys if key not in post_keys)
    persistent_keys = sorted(key for key in detector_keys if key in post_keys or key in stable_keys)
    final_snapshot_mismatch = _sample_observation_fingerprint(samples[-1].observation if samples else {}) != _sample_observation_fingerprint(temporal_window.terminal_snapshot)
    early_stable_layout = bool(post_drift.get("post_stable_layout_drift"))
    return {
        "episode": episode,
        "step": step,
        "action_id": temporal_window.action_id,
        "agent_mode": temporal_window.agent_mode,
        "action_type": temporal_window.action_type,
        "target_id": temporal_window.target_id,
        "route_before": temporal_window.route_before,
        "stable_decision_ms": samples[-1].relative_ms if samples else 0,
        "stable_sample_index": stable_sample_index,
        "detector_sample_index": detector_sample_index,
        "detector_after_snapshot_equals_final_sample": not final_snapshot_mismatch,
        "final_snapshot_mismatch": final_snapshot_mismatch,
        "signal_stability_at_stable": signal["stable_flags"],
        "last_change_ms": signal["last_change_ms"],
        "time_from_last_layout_change_to_stable": signal["time_from_last_layout_change_to_stable"],
        "stability_decision_timeline": _decision_timeline(samples, post_stable_samples),
        "layout_components_at_stable": samples[-1].layout_summary if samples else {},
        "bbox_delta_metrics": bbox,
        "layout_movement_counts": movement,
        "post_stable_drift": post_drift,
        "early_stable_classification": "EARLY_STABLE_LAYOUT" if early_stable_layout else "NO_POST_STABLE_LAYOUT_DRIFT",
        "detector_layout_candidates": detector_layout,
        "layout_candidate_count": len(detector_layout),
        "per_sample_layout_candidates": per_sample_layout,
        "transient_layout_candidate_keys": transient_keys,
        "persistent_layout_candidate_keys": persistent_keys,
        "layout_candidate_classification": _layout_candidate_classification(detector_layout, transient_keys, persistent_keys),
        "post_stable_sample_count": len(post_stable_samples),
    }


def summarize_temporal_stability_diagnostics(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    stable_count = len(rows)
    post_layout_drift_count = sum(1 for row in rows if _nested_bool(row, "post_stable_drift", "post_stable_layout_drift"))
    post_any_drift_count = sum(1 for row in rows if any(bool(v) for v in (row.get("post_stable_drift") or {}).values()))
    layout_rows = [row for row in rows if int(row.get("layout_candidate_count", 0) or 0) > 0]
    transient = sum(len(row.get("transient_layout_candidate_keys", []) or []) for row in layout_rows)
    persistent = sum(len(row.get("persistent_layout_candidate_keys", []) or []) for row in layout_rows)
    bbox = _aggregate_bbox(rows)
    timing = [int(row.get("time_from_last_layout_change_to_stable", 0) or 0) for row in rows]
    drift_timing = Counter()
    for row in rows:
        bucket = str((row.get("post_stable_drift") or {}).get("first_layout_drift_bucket") or "")
        if bucket:
            drift_timing[bucket] += 1
    movement = Counter()
    for row in rows:
        movement.update(row.get("layout_movement_counts", {}) or {})
    return {
        "stable_count": stable_count,
        "post_stable_layout_drift_count": post_layout_drift_count,
        "early_stable_count": post_layout_drift_count,
        "early_stable_rate": post_layout_drift_count / max(1, stable_count),
        "stable_precision_proxy": (stable_count - post_any_drift_count) / max(1, stable_count),
        "layout_stable_precision_proxy": (stable_count - post_layout_drift_count) / max(1, stable_count),
        "layout_candidate_count": sum(int(row.get("layout_candidate_count", 0) or 0) for row in rows),
        "layout_candidate_windows": len(layout_rows),
        "transient_layout_candidates": transient,
        "persistent_layout_candidates": persistent,
        "layout_candidate_early_stable_rate": (
            sum(1 for row in layout_rows if _nested_bool(row, "post_stable_drift", "post_stable_layout_drift")) / max(1, len(layout_rows))
        ),
        "detector_stable_sample_mismatch_count": sum(1 for row in rows if bool(row.get("final_snapshot_mismatch"))),
        "bbox_delta_metrics": bbox,
        "layout_movement_counts": dict(movement),
        "time_last_layout_change_to_stable_distribution": _distribution(timing),
        "post_stable_drift_timing_distribution": dict(drift_timing),
    }


def _signal_diagnostics(samples: List[EvidenceSample]) -> Dict[str, Any]:
    last = {
        "semantic": 0,
        "layout": 0,
        "loading": 0,
        "network": 0,
        "url": 0,
    }
    if not samples:
        return {"stable_flags": {}, "last_change_ms": last, "time_from_last_layout_change_to_stable": 0}
    previous = samples[0]
    for sample in samples[1:]:
        if sample.semantic_dom_fingerprint != previous.semantic_dom_fingerprint:
            last["semantic"] = sample.relative_ms
        if sample.layout_fingerprint != previous.layout_fingerprint:
            last["layout"] = sample.relative_ms
        if sample.loading_state != previous.loading_state:
            last["loading"] = sample.relative_ms
        if sample.network_state.to_dict() != previous.network_state.to_dict():
            last["network"] = sample.relative_ms
        if sample.url != previous.url:
            last["url"] = sample.relative_ms
        previous = sample
    stable_ms = samples[-1].relative_ms
    return {
        "stable_flags": {
            "semantic_stable": last["semantic"] < stable_ms,
            "layout_stable": last["layout"] < stable_ms,
            "loading_stable": last["loading"] < stable_ms,
            "network_stable": last["network"] < stable_ms,
            "url_stable": last["url"] < stable_ms,
        },
        "last_change_ms": {
            "last_semantic_change_ms": last["semantic"],
            "last_layout_change_ms": last["layout"],
            "last_loading_change_ms": last["loading"],
            "last_network_change_ms": last["network"],
            "last_url_change_ms": last["url"],
        },
        "time_from_last_layout_change_to_stable": max(0, stable_ms - last["layout"]),
    }


def _post_stable_drift(stable: EvidenceSample, post: List[EvidenceSample]) -> Dict[str, Any]:
    result = {
        "post_stable_semantic_drift": False,
        "post_stable_layout_drift": False,
        "post_stable_url_drift": False,
        "post_stable_loading_drift": False,
        "post_stable_network_drift": False,
        "first_layout_drift_ms_after_stable": None,
        "first_layout_drift_bucket": "",
    }
    for sample in post:
        delta = max(0, sample.relative_ms - stable.relative_ms)
        if sample.semantic_dom_fingerprint != stable.semantic_dom_fingerprint:
            result["post_stable_semantic_drift"] = True
        if sample.layout_fingerprint != stable.layout_fingerprint:
            result["post_stable_layout_drift"] = True
            if result["first_layout_drift_ms_after_stable"] is None:
                result["first_layout_drift_ms_after_stable"] = delta
                result["first_layout_drift_bucket"] = _drift_bucket(delta)
        if sample.url != stable.url:
            result["post_stable_url_drift"] = True
        if sample.loading_state != stable.loading_state:
            result["post_stable_loading_drift"] = True
        if sample.network_state.to_dict() != stable.network_state.to_dict():
            result["post_stable_network_drift"] = True
    return result


def _bbox_delta_metrics(samples: List[EvidenceSample]) -> Dict[str, Any]:
    deltas: List[float] = []
    for before, after in zip(samples, samples[1:]):
        left = before.layout_summary.get("target_bbox") or []
        right = after.layout_summary.get("target_bbox") or []
        if len(left) < 4 or len(right) < 4:
            continue
        deltas.extend(abs(float(right[i]) - float(left[i])) for i in range(4))
    return {
        "mean_bbox_delta": statistics.mean(deltas) if deltas else 0.0,
        "p95_bbox_delta": _percentile(deltas, 95),
        "max_bbox_delta": max(deltas) if deltas else 0.0,
        "delta_count": len(deltas),
    }


def _movement_counts(samples: List[EvidenceSample]) -> Dict[str, int]:
    counts = Counter()
    for before, after in zip(samples, samples[1:]):
        left = before.layout_summary
        right = after.layout_summary
        if left.get("viewport") != right.get("viewport"):
            counts["VIEWPORT_CHANGE"] += 1
        lb = left.get("target_bbox") or []
        rb = right.get("target_bbox") or []
        if len(lb) >= 4 and len(rb) >= 4:
            if lb[:2] != rb[:2]:
                counts["TARGET_MOVE"] += 1
            if lb[2:4] != rb[2:4]:
                counts["TARGET_RESIZE"] += 1
        if int(left.get("overlap_count", 0) or 0) < int(right.get("overlap_count", 0) or 0):
            counts["OVERLAP_APPEARED"] += 1
        if int(left.get("overlap_count", 0) or 0) > int(right.get("overlap_count", 0) or 0):
            counts["OVERLAP_DISAPPEARED"] += 1
        if int(left.get("overflow_count", 0) or 0) < int(right.get("overflow_count", 0) or 0):
            counts["OVERFLOW_APPEARED"] += 1
        if int(left.get("overflow_count", 0) or 0) > int(right.get("overflow_count", 0) or 0):
            counts["OVERFLOW_DISAPPEARED"] += 1
        if left.get("visible_geometry") != right.get("visible_geometry") and not any(counts[key] for key in ("TARGET_MOVE", "TARGET_RESIZE", "VIEWPORT_CHANGE")):
            counts["OTHER_ELEMENT_MOVE"] += 1
    return dict(counts)


def _decision_timeline(samples: List[EvidenceSample], post: List[EvidenceSample]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    combined = samples + post
    stable_index = len(samples) - 1
    previous: EvidenceSample | None = None
    for index, sample in enumerate(combined):
        events: List[str] = []
        if previous is None:
            events.append("action")
        else:
            if sample.semantic_dom_fingerprint != previous.semantic_dom_fingerprint:
                events.append("semantic change")
            if sample.layout_fingerprint != previous.layout_fingerprint:
                events.append("layout change")
            if sample.loading_state != previous.loading_state:
                events.append("loading change")
            if sample.network_state.to_dict() != previous.network_state.to_dict():
                events.append("network change")
            if sample.url != previous.url:
                events.append("url change")
        if index == stable_index:
            events.append("STABLE")
        if index > stable_index:
            events.append("post-stable diagnostic")
        rows.append({"sample_index": index, "relative_ms": sample.relative_ms, "events": events})
        previous = sample
    return rows


def _layout_candidate_payload(item: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = item.get("evidence", {}) if isinstance(item.get("evidence"), Mapping) else {}
    return {
        "rule_id": str(item.get("rule_id") or evidence.get("rule_id") or ""),
        "type": str(item.get("type") or ""),
        "signature": str(item.get("logical_key") or item.get("signature") or ""),
        "target": evidence.get("target_bid") or item.get("target_id") or "",
        "target_bbox": evidence.get("target_bbox") or [],
        "other_element": evidence.get("occluder_bid") or "",
        "other_bbox": evidence.get("occluder_bbox") or [],
        "intersection_area": evidence.get("intersection_area"),
        "overlap_ratio": evidence.get("occlusion_ratio"),
        "overflow_dimension": _overflow_dimension(evidence),
        "viewport": evidence.get("viewport") or evidence.get("viewport_bbox") or [],
        "visibility": evidence.get("visible_ratio"),
        "occlusion": evidence.get("occlusion_ratio"),
        "click_blocked": bool(evidence.get("occlusion_ratio") and float(evidence.get("occlusion_ratio") or 0.0) >= 0.25),
        "target_partially_visible": bool(evidence.get("visible_ratio") is not None and 0.0 < float(evidence.get("visible_ratio") or 0.0) < 1.0),
        "target_fully_visible": bool(evidence.get("visible_ratio") is not None and float(evidence.get("visible_ratio") or 0.0) >= 1.0),
    }


def _layout_candidate_classification(candidates: List[Mapping[str, Any]], transient: List[str], persistent: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in candidates:
        key = str(item.get("signature") or "")
        if key in transient:
            result[key] = "TRANSIENT_LAYOUT_CANDIDATE"
        elif key in persistent:
            result[key] = "PERSISTENT_LAYOUT_CANDIDATE"
        else:
            result[key] = "UNCLASSIFIED_LAYOUT_CANDIDATE"
    return result


def _is_layout_candidate(item: Mapping[str, Any]) -> bool:
    return str(item.get("rule_id") or (item.get("evidence") or {}).get("rule_id") or "") in LAYOUT_RULES


def _overflow_dimension(evidence: Mapping[str, Any]) -> str:
    if evidence.get("horizontal_overflow"):
        return "horizontal"
    if evidence.get("vertical_overflow"):
        return "vertical"
    return ""


def _sample_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    candidates = observation.get("candidate_elements", []) or []
    return repr((page.get("url"), page.get("page_text_length"), page.get("dom_node_count"), len(candidates)))


def _aggregate_bbox(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    values = []
    for row in rows:
        metrics = row.get("bbox_delta_metrics") or {}
        if float(metrics.get("max_bbox_delta", 0.0) or 0.0) or int(metrics.get("delta_count", 0) or 0):
            values.append(metrics)
    max_values = [float(row.get("max_bbox_delta", 0.0) or 0.0) for row in values]
    mean_values = [float(row.get("mean_bbox_delta", 0.0) or 0.0) for row in values]
    return {
        "mean_bbox_delta": statistics.mean(mean_values) if mean_values else 0.0,
        "p95_bbox_delta": _percentile(max_values, 95),
        "max_bbox_delta": max(max_values) if max_values else 0.0,
    }


def _distribution(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": _percentile([float(value) for value in values], 95),
        "min": min(values),
        "max": max(values),
    }


def _drift_bucket(delta_ms: int) -> str:
    if delta_ms <= 100:
        return "+0~100ms"
    if delta_ms <= 300:
        return "100~300ms"
    if delta_ms <= 500:
        return "300~500ms"
    return ">500ms"


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def _nested_bool(row: Mapping[str, Any], container: str, key: str) -> bool:
    value = row.get(container)
    return bool(value.get(key)) if isinstance(value, Mapping) else False
