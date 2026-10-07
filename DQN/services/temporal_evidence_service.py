"""Framework-independent temporal evidence collection for detector input."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from services.framework_aware_evidence_normalizer import normalize_network_events
from services.r10_network_ui_semantic_evidence import build_network_ui_semantic_evidence


class TemporalState(str, Enum):
    OBSERVING = "OBSERVING"
    TRANSIENT = "TRANSIENT"
    STABLE = "STABLE"
    TIMED_OUT_UNSTABLE = "TIMED_OUT_UNSTABLE"


@dataclass(frozen=True)
class TemporalEvidenceConfig:
    sample_interval_ms: int = 100
    minimum_observation_ms: int = 250
    stable_quiet_window_ms: int = 350
    maximum_observation_ms: int = 1500
    layout_position_tolerance_px: float = 2.0
    long_lived_request_ms: int = 1000
    debug_timeline: bool = False


@dataclass
class NetworkState:
    active_request_count: int = 0
    blocking_active_request_count: int = 0
    completed_request_count: int = 0
    failed_request_count: int = 0
    active_requests: List[Dict[str, Any]] = field(default_factory=list)
    completed_requests: List[Dict[str, Any]] = field(default_factory=list)
    failed_requests: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_request_count": self.active_request_count,
            "blocking_active_request_count": self.blocking_active_request_count,
            "completed_request_count": self.completed_request_count,
            "failed_request_count": self.failed_request_count,
            "active_requests": list(self.active_requests),
            "completed_requests": list(self.completed_requests),
            "failed_requests": list(self.failed_requests),
        }


@dataclass
class EvidenceSample:
    timestamp: float
    relative_ms: int
    url: str
    semantic_dom_fingerprint: str
    layout_fingerprint: str
    loading_state: Dict[str, Any]
    network_state: NetworkState
    console_delta: List[Dict[str, Any]]
    target_state: Dict[str, Any]
    candidate_related_element_state: Dict[str, Any]
    dom_semantic_summary: Dict[str, Any]
    dom_mutation_summary: Dict[str, Any]
    layout_summary: Dict[str, Any]
    observation: Mapping[str, Any] = field(repr=False, default_factory=dict)

    def stability_signature(self) -> tuple[Any, ...]:
        return (
            self.url,
            self.semantic_dom_fingerprint,
            self.layout_fingerprint,
            _stable_loading_signature(self.loading_state),
            self.network_state.blocking_active_request_count,
        )

    def to_dict(self, include_observation: bool = False) -> Dict[str, Any]:
        payload = {
            "timestamp": self.timestamp,
            "relative_ms": self.relative_ms,
            "url": self.url,
            "semantic_dom_fingerprint": self.semantic_dom_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "loading_state": dict(self.loading_state),
            "network_state": self.network_state.to_dict(),
            "console_delta": list(self.console_delta),
            "target_state": dict(self.target_state),
            "candidate_related_element_state": dict(self.candidate_related_element_state),
            "dom_semantic_summary": dict(self.dom_semantic_summary),
            "dom_mutation_summary": dict(self.dom_mutation_summary),
            "layout_summary": dict(self.layout_summary),
        }
        if include_observation:
            payload["observation"] = self.observation
        return payload


@dataclass
class ActionEvidenceWindow:
    action_id: str
    agent_mode: str
    action_type: str
    target_id: str
    action_timestamp: float
    route_before: str
    started_at: float
    ended_at: float
    duration_ms: int
    samples: List[EvidenceSample]
    final_stability: TemporalState
    stability_reason: List[str]
    terminal_snapshot: Mapping[str, Any] = field(repr=False, default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_metadata(self, *, include_samples: bool = True) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "agent_mode": self.agent_mode,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "action_timestamp": self.action_timestamp,
            "route_before": self.route_before,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "sample_count": len(self.samples),
            "final_stability": self.final_stability.value,
            "stability_reason": list(self.stability_reason),
            "timeline": list(self.timeline),
            "samples": [sample.to_dict() for sample in self.samples] if include_samples else [],
        }


class TemporalStateTracker:
    def __init__(self, config: TemporalEvidenceConfig | None = None) -> None:
        self.config = config or TemporalEvidenceConfig()
        self.state = TemporalState.OBSERVING
        self.quiet_since_ms = 0
        self.reasons: List[str] = []
        self._last_signature: tuple[Any, ...] | None = None

    def observe(self, sample: EvidenceSample) -> TemporalState:
        signature = sample.stability_signature()
        changed = self._last_signature is not None and signature != self._last_signature
        unsettled = self._sample_unsettled(sample)
        if self._last_signature is None or changed or unsettled:
            self.quiet_since_ms = sample.relative_ms
        self._last_signature = signature
        if changed or unsettled:
            self.state = TemporalState.TRANSIENT
        elif (
            sample.relative_ms >= self.config.minimum_observation_ms
            and sample.relative_ms - self.quiet_since_ms >= self.config.stable_quiet_window_ms
        ):
            self.state = TemporalState.STABLE
            self.reasons = self._stable_reasons(sample)
        else:
            self.state = TemporalState.OBSERVING
        return self.state

    def timeout(self, sample: EvidenceSample | None = None) -> TemporalState:
        self.state = TemporalState.TIMED_OUT_UNSTABLE
        self.reasons = ["MAX_WINDOW_TIMEOUT"]
        if sample is not None:
            if sample.network_state.blocking_active_request_count:
                self.reasons.append("NETWORK_STILL_ACTIVE")
            if sample.loading_state.get("loading_visible"):
                self.reasons.append("LOADING_STILL_VISIBLE")
        return self.state

    def _sample_unsettled(self, sample: EvidenceSample) -> bool:
        return bool(
            sample.network_state.blocking_active_request_count > 0
            or sample.loading_state.get("loading_visible")
        )

    def _stable_reasons(self, sample: EvidenceSample) -> List[str]:
        reasons = ["STABLE_SEMANTIC_QUIET", "STABLE_LAYOUT_QUIET", "STABLE_URL_QUIET"]
        if sample.network_state.blocking_active_request_count == 0:
            reasons.append("STABLE_AFTER_NETWORK")
        if not sample.loading_state.get("loading_visible"):
            reasons.append("STABLE_AFTER_LOADING")
        return reasons


class RawEvidenceCollector:
    def __init__(
        self,
        sampler: Callable[[], Mapping[str, Any]],
        *,
        config: TemporalEvidenceConfig | None = None,
        sleep_ms: Callable[[int], None] | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.sampler = sampler
        self.config = config or TemporalEvidenceConfig()
        self.sleep_ms = sleep_ms or (lambda ms: time.sleep(max(0, ms) / 1000.0))
        self.time_func = time_func or time.time

    def collect(self, action_context: Mapping[str, Any]) -> ActionEvidenceWindow:
        started_at = self.time_func()
        tracker = TemporalStateTracker(self.config)
        samples: List[EvidenceSample] = []
        timeline: List[Dict[str, Any]] = []
        terminal = TemporalState.OBSERVING
        while True:
            now = self.time_func()
            relative_ms = int(round((now - started_at) * 1000.0))
            observation = self.sampler()
            sample_context = dict(action_context)
            if samples:
                sample_context["previous_sample"] = samples[-1]
            sample = build_evidence_sample(
                observation,
                action_context=sample_context,
                timestamp=now,
                relative_ms=relative_ms,
                config=self.config,
            )
            samples.append(sample)
            terminal = tracker.observe(sample)
            if self.config.debug_timeline:
                timeline.append(_timeline_row(sample, terminal))
            if terminal == TemporalState.STABLE:
                break
            if relative_ms >= self.config.maximum_observation_ms:
                terminal = tracker.timeout(sample)
                if self.config.debug_timeline and timeline:
                    timeline[-1]["state"] = terminal.value
                    timeline[-1]["reason"] = list(tracker.reasons)
                break
            remaining = self.config.maximum_observation_ms - relative_ms
            self.sleep_ms(min(self.config.sample_interval_ms, max(0, remaining)))
        ended_at = self.time_func()
        final_sample = samples[-1]
        return ActionEvidenceWindow(
            action_id=str(action_context.get("action_id") or ""),
            agent_mode=str(action_context.get("agent_mode") or ""),
            action_type=str(action_context.get("action_type") or ""),
            target_id=str(action_context.get("target_id") or ""),
            action_timestamp=float(action_context.get("action_timestamp") or started_at),
            route_before=str(action_context.get("route_before") or ""),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=int(round((ended_at - started_at) * 1000.0)),
            samples=samples,
            final_stability=terminal,
            stability_reason=list(tracker.reasons or ["OBSERVATION_WINDOW_ENDED"]),
            terminal_snapshot=final_sample.observation,
            timeline=timeline or [_timeline_row(sample, terminal if index == len(samples) - 1 else TemporalState.TRANSIENT) for index, sample in enumerate(samples)],
        )


def detect_anomalies_with_temporal_evidence(
    before_observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    temporal_evidence: ActionEvidenceWindow,
    site_profile: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    from services.anomaly_detection_service import detect_anomalies

    enriched_info = dict(action_info)
    metadata = temporal_evidence.to_metadata(include_samples=False)
    enriched_info["temporal_evidence"] = metadata
    raw_network_entries = _network_entries_for_detector(temporal_evidence)
    normalized_network_events = normalize_network_events(raw_network_entries, action_context=metadata)
    network_ui_semantic = build_network_ui_semantic_evidence(
        action_context=metadata,
        network_entries=raw_network_entries,
        observation=temporal_evidence.terminal_snapshot,
    )
    enriched_info["raw_network_entries"] = raw_network_entries
    enriched_info["normalized_network_events"] = normalized_network_events
    enriched_info["network_ui_semantic"] = network_ui_semantic
    # Keep the detector contract stable while enriching each entry with its
    # semantic classification and preserving the raw event under a nested key.
    enriched_info["network_entries"] = normalized_network_events
    terminal_snapshot = temporal_evidence.terminal_snapshot
    if isinstance(terminal_snapshot, dict):
        page_state = terminal_snapshot.setdefault("page_state", {})
        if isinstance(page_state, dict):
            page_state["network_ui_semantic"] = network_ui_semantic
            page_state.setdefault("api_entities", _entities_from_semantic_fields(network_ui_semantic.get("network_fields", [])))
            page_state.setdefault("ui_entities", _entities_from_semantic_fields(network_ui_semantic.get("ui_fields", [])))
    if temporal_evidence.final_stability == TemporalState.STABLE:
        enriched_info["network_changed"] = any(
            event.get("functional_relevance") == "RELEVANT"
            for event in normalized_network_events
        )
    anomalies = detect_anomalies(before_observation, temporal_evidence.terminal_snapshot, enriched_info, site_profile)
    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        anomaly["trigger_action_id"] = metadata["action_id"]
        anomaly["evidence_window_id"] = metadata["action_id"]
        anomaly["stability_state"] = metadata["final_stability"]
        evidence = anomaly.setdefault("evidence", {})
        if isinstance(evidence, dict):
            evidence.setdefault("temporal_evidence", metadata)
            evidence.setdefault("network_ui_semantic_summary", {
                "network_field_count": len(network_ui_semantic.get("network_fields", []) or []),
                "ui_field_count": len(network_ui_semantic.get("ui_fields", []) or []),
                "pair_count": len(network_ui_semantic.get("pairs", []) or []),
                "not_equal_pairs": network_ui_semantic.get("not_equal_pairs", 0),
            })
    return anomalies


def build_evidence_sample(
    observation: Mapping[str, Any],
    *,
    action_context: Mapping[str, Any],
    timestamp: float,
    relative_ms: int,
    config: TemporalEvidenceConfig | None = None,
) -> EvidenceSample:
    cfg = config or TemporalEvidenceConfig()
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    url = str(page_state.get("url") or observation.get("url") or "")
    candidates = observation.get("candidate_elements", []) or []
    candidates = candidates if isinstance(candidates, list) else []
    target = _target_candidate(candidates, action_context)
    semantic_summary = _semantic_summary(observation, candidates)
    layout_summary = _layout_summary(observation, candidates, target, cfg)
    loading_state = _loading_state(observation, candidates, target)
    network_state = _network_state(observation, int(action_context.get("relative_origin_ms", 0) or 0), relative_ms, cfg)
    console_delta = _console_delta(observation)
    target_state = _target_state(target)
    candidate_state = _candidate_related_state(candidates, target)
    previous = action_context.get("previous_sample")
    previous_semantic = previous.dom_semantic_summary if isinstance(previous, EvidenceSample) else {}
    return EvidenceSample(
        timestamp=timestamp,
        relative_ms=relative_ms,
        url=url,
        semantic_dom_fingerprint=_fingerprint(semantic_summary),
        layout_fingerprint=_fingerprint(layout_summary),
        loading_state=loading_state,
        network_state=network_state,
        console_delta=console_delta,
        target_state=target_state,
        candidate_related_element_state=candidate_state,
        dom_semantic_summary=semantic_summary,
        dom_mutation_summary=_dom_mutation_summary(previous_semantic, semantic_summary),
        layout_summary=layout_summary,
        observation=observation,
    )


def summarize_temporal_windows(windows: Iterable[ActionEvidenceWindow]) -> Dict[str, Any]:
    rows = list(windows)
    durations = [row.duration_ms for row in rows]
    if not durations:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean_evidence_window_ms": statistics.mean(durations),
        "median_evidence_window_ms": statistics.median(durations),
        "p95_evidence_window_ms": _percentile(durations, 95),
        "max_evidence_window_ms": max(durations),
        "stable_count": sum(1 for row in rows if row.final_stability == TemporalState.STABLE),
        "timed_out_unstable_count": sum(1 for row in rows if row.final_stability == TemporalState.TIMED_OUT_UNSTABLE),
        "average_samples_per_action": statistics.mean([len(row.samples) for row in rows]),
    }


def _semantic_summary(observation: Mapping[str, Any], candidates: List[Any]) -> Dict[str, Any]:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    normalized_candidates = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not bool(candidate.get("visible", True)):
            continue
        normalized_candidates.append(
            {
                "role": _clean_token(candidate.get("role")),
                "tag": _clean_token(candidate.get("tag")),
                "text": _normalize_text(_candidate_text(candidate))[:80],
                "enabled": bool(candidate.get("enabled", True)),
                "clickable": bool(candidate.get("clickable", False)),
            }
        )
    normalized_candidates.sort(key=lambda item: (item["role"], item["tag"], item["text"]))
    return {
        "title": _normalize_text(page_state.get("title")),
        "page_text_length_bucket": _bucket_int(page_state.get("page_text_length"), 20),
        "page_text_sample": _normalize_text(page_state.get("page_text_sample"))[:240],
        "dom_node_count_bucket": _bucket_int(page_state.get("dom_node_count"), 5),
        "candidate_count": len(normalized_candidates),
        "candidates": normalized_candidates[:32],
        "cart_count": page_state.get("cart_count"),
        "cart_text": _normalize_text(page_state.get("cart_text")),
        "modal": bool(page_state.get("has_modal_or_dialog")),
        "form": bool(page_state.get("has_form")),
    }


def _layout_summary(
    observation: Mapping[str, Any],
    candidates: List[Any],
    target: Mapping[str, Any] | None,
    config: TemporalEvidenceConfig,
) -> Dict[str, Any]:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    layout = page_state.get("layout_signals", {}) if isinstance(page_state.get("layout_signals"), Mapping) else {}
    visible = [candidate for candidate in candidates if isinstance(candidate, Mapping) and bool(candidate.get("visible", True))]
    return {
        "viewport": [
            _bucket_int(page_state.get("viewport_width"), 4),
            _bucket_int(page_state.get("viewport_height"), 4),
        ],
        "target_bbox": _rounded_bbox((target or {}).get("bbox"), config.layout_position_tolerance_px),
        "target_visible": bool((target or {}).get("visible", False)) if target else False,
        "target_enabled": bool((target or {}).get("enabled", True)) if target else None,
        "visible_geometry": _visible_geometry(visible, config.layout_position_tolerance_px),
        "overlap_count": int(layout.get("layout_overlap_count", 0) or 0),
        "overflow_count": int(layout.get("layout_overflow_count", 0) or 0),
    }


def _loading_state(observation: Mapping[str, Any], candidates: List[Any], target: Mapping[str, Any] | None) -> Dict[str, Any]:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals"), Mapping) else {}
    marker_count = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not bool(candidate.get("visible", True)):
            continue
        role = str(candidate.get("role") or "").lower()
        haystack = " ".join(str(candidate.get(key) or "") for key in ("name", "text", "aria_label", "title")).lower()
        normalized = " ".join(haystack.split())
        if role == "progressbar" or normalized in {"loading", "loading...", "pending", "please wait", "spinner", "처리 중", "로딩"}:
            marker_count += 1
    loading_visible = bool(
        str(runtime.get("loading") or "").lower() in {"true", "loading", "pending"}
        or marker_count > 0
    )
    target_disabled = bool(target is not None and not bool(target.get("enabled", True)))
    return {
        "loading_visible": loading_visible,
        "visible_loading_indicator_count": marker_count,
        "target_disabled_after_action": target_disabled,
    }


def _network_state(
    observation: Mapping[str, Any],
    relative_origin_ms: int,
    relative_ms: int,
    config: TemporalEvidenceConfig,
) -> NetworkState:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals"), Mapping) else {}
    temporal = runtime.get("temporal_network", {}) if isinstance(runtime.get("temporal_network"), Mapping) else {}
    active = [dict(item) for item in temporal.get("active_requests", []) or [] if isinstance(item, Mapping)]
    completed = [dict(item) for item in temporal.get("completed_requests", []) or [] if isinstance(item, Mapping)]
    failed = [dict(item) for item in temporal.get("failed_requests", []) or [] if isinstance(item, Mapping)]
    blocking_active = [
        item
        for item in active
        if not _background_like_active_request(item, relative_ms - relative_origin_ms, config.long_lived_request_ms)
    ]
    return NetworkState(
        active_request_count=len(active),
        blocking_active_request_count=len(blocking_active),
        completed_request_count=len(completed),
        failed_request_count=len(failed),
        active_requests=active[:20],
        completed_requests=completed[-20:],
        failed_requests=failed[-20:],
    )


def _console_delta(observation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    runtime = observation.get("runtime_signals", {}) if isinstance(observation.get("runtime_signals"), Mapping) else {}
    values = runtime.get("temporal_console_errors", []) or runtime.get("console_errors", []) or []
    return [dict(item) for item in values if isinstance(item, Mapping)][-20:]


def _target_candidate(candidates: List[Any], action_context: Mapping[str, Any]) -> Mapping[str, Any] | None:
    target_id = str(action_context.get("target_id") or "")
    if target_id:
        for candidate in candidates:
            if isinstance(candidate, Mapping) and target_id in {str(candidate.get("bid") or ""), str(candidate.get("candidate_id") or "")}:
                return candidate
    try:
        index = int(action_context.get("candidate_index", 0) or 0)
    except (TypeError, ValueError):
        index = 0
    if 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return None


def _target_state(target: Mapping[str, Any] | None) -> Dict[str, Any]:
    if target is None:
        return {"present": False}
    return {
        "present": True,
        "bid": str(target.get("bid") or ""),
        "visible": bool(target.get("visible", True)),
        "enabled": bool(target.get("enabled", True)),
        "bbox": _rounded_bbox(target.get("bbox"), 1.0),
    }


def _candidate_related_state(candidates: List[Any], target: Mapping[str, Any] | None) -> Dict[str, Any]:
    return {
        "candidate_count": len([candidate for candidate in candidates if isinstance(candidate, Mapping)]),
        "visible_candidate_count": len([candidate for candidate in candidates if isinstance(candidate, Mapping) and bool(candidate.get("visible", True))]),
        "target": _target_state(target),
    }


def _dom_mutation_summary(previous: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    if not previous:
        return {"semantic_changed": False, "candidate_count_delta": 0}
    return {
        "semantic_changed": _fingerprint(previous) != _fingerprint(current),
        "candidate_count_delta": int(current.get("candidate_count", 0) or 0) - int(previous.get("candidate_count", 0) or 0),
    }


def _network_entries_for_detector(window: ActionEvidenceWindow) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for sample in window.samples:
        entries.extend(sample.network_state.completed_requests)
        entries.extend(sample.network_state.failed_requests)
    deduped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("request_id") or entry.get("url") or entry)
        deduped[key] = entry
    return list(deduped.values())


def _entities_from_semantic_fields(fields: Any) -> List[Dict[str, Any]]:
    if not isinstance(fields, list):
        return []
    entity: Dict[str, Any] = {"_source": "network_ui_semantic"}
    for field in fields:
        if not isinstance(field, Mapping):
            continue
        key = str(field.get("key") or "")
        if not key:
            continue
        semantic = field.get("semantic_value") if isinstance(field.get("semantic_value"), Mapping) else {}
        entity[key] = semantic.get("normalized", field.get("value"))
    return [entity] if len(entity) > 1 else []


def _background_like_active_request(entry: Mapping[str, Any], age_ms: int, long_lived_request_ms: int) -> bool:
    resource_type = str(entry.get("resource_type") or entry.get("initiator_type") or "").lower()
    url = str(entry.get("url") or "").lower()
    if resource_type in {"websocket", "eventsource"}:
        return True
    if age_ms >= long_lived_request_ms and any(token in url for token in ("analytics", "telemetry", "beacon", "metrics")):
        return True
    return False


def _timeline_row(sample: EvidenceSample, state: TemporalState) -> Dict[str, Any]:
    return {
        "t": f"T+{sample.relative_ms}",
        "relative_ms": sample.relative_ms,
        "state": state.value,
        "url": sample.url,
        "semantic": sample.semantic_dom_fingerprint[:12],
        "layout": sample.layout_fingerprint[:12],
        "loading": bool(sample.loading_state.get("loading_visible")),
        "network_active": sample.network_state.active_request_count,
        "network_blocking_active": sample.network_state.blocking_active_request_count,
        "network_completed": sample.network_state.completed_request_count,
        "network_failed": sample.network_state.failed_request_count,
    }


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(str(candidate.get(key) or "") for key in ("name", "text", "aria_label", "title")).strip()


def _visible_geometry(candidates: List[Mapping[str, Any]], tolerance: float) -> List[Dict[str, Any]]:
    rows = []
    for candidate in candidates[:32]:
        rows.append(
            {
                "role": _clean_token(candidate.get("role")),
                "tag": _clean_token(candidate.get("tag")),
                "bbox": _rounded_bbox(candidate.get("bbox"), tolerance),
            }
        )
    rows.sort(key=lambda item: (item["role"], item["tag"], item["bbox"]))
    return rows


def _rounded_bbox(value: Any, tolerance: float) -> List[int]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    bucket = max(1.0, float(tolerance or 1.0))
    result = []
    for item in value[:4]:
        try:
            result.append(int(round(float(item) / bucket) * bucket))
        except (TypeError, ValueError):
            result.append(0)
    return result


def _stable_loading_signature(loading_state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(loading_state.get("loading_visible")),
        int(loading_state.get("visible_loading_indicator_count", 0) or 0),
        bool(loading_state.get("target_disabled_after_action")),
    )


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "<time>", text)
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", "<date>", text)
    text = re.sub(r"\b[a-f0-9]{8,}\b", "<hash>", text)
    text = re.sub(r"\bid[-_:]?\d+\b", "id<num>", text)
    return " ".join(text.split())


def _clean_token(value: Any) -> str:
    return _normalize_text(value)[:40]


def _bucket_int(value: Any, bucket: int) -> int:
    try:
        number = int(float(value or 0))
    except (TypeError, ValueError):
        number = 0
    if bucket <= 1:
        return number
    return int(round(number / bucket) * bucket)


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha1(repr(_sorted_jsonish(value)).encode("utf-8")).hexdigest()


def _sorted_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sorted_jsonish(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_jsonish(item) for item in value]
    return value


def _percentile(values: List[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])
