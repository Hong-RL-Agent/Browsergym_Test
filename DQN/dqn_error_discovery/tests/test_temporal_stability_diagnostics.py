from __future__ import annotations

from services.anomaly_detection_service import detect_anomalies
from services.temporal_evidence_service import ActionEvidenceWindow, EvidenceSample, NetworkState, TemporalState
from services.temporal_stability_diagnostic_service import (
    analyze_temporal_stability_window,
    summarize_temporal_stability_diagnostics,
)


def _candidate(bid: str = "submit", bbox: list[float] | None = None) -> dict:
    return {
        "bid": bid,
        "name": "Submit",
        "text": "Submit",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": bbox or [100, 100, 100, 40],
    }


def _panel() -> dict:
    return {
        "bid": "panel",
        "name": "Panel",
        "text": "Panel",
        "role": "generic",
        "tag": "div",
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "class_name": "floating-panel",
        "bbox": [105, 105, 100, 35],
    }


def _obs(candidates: list[dict], *, text: str = "Ready") -> dict:
    return {
        "page_state": {
            "url": "http://app.test/",
            "title": "App",
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": len(candidates),
            "viewport_width": 800,
            "viewport_height": 600,
        },
        "runtime_signals": {},
        "candidate_elements": candidates,
    }


def _sample(index: int, relative_ms: int, candidates: list[dict], *, layout: str | None = None, semantic: str = "s") -> EvidenceSample:
    target_bbox = candidates[0].get("bbox") if candidates else []
    overlap_count = 1 if len(candidates) > 1 else 0
    return EvidenceSample(
        timestamp=relative_ms / 1000.0,
        relative_ms=relative_ms,
        url="http://app.test/",
        semantic_dom_fingerprint=semantic,
        layout_fingerprint=layout or f"l{index}",
        loading_state={"loading_visible": False, "visible_loading_indicator_count": 0, "target_disabled_after_action": False},
        network_state=NetworkState(),
        console_delta=[],
        target_state={"present": True, "bbox": target_bbox},
        candidate_related_element_state={},
        dom_semantic_summary={},
        dom_mutation_summary={},
        layout_summary={
            "viewport": [800, 600],
            "target_bbox": target_bbox,
            "visible_geometry": [{"role": item.get("role"), "bbox": item.get("bbox")} for item in candidates],
            "overlap_count": overlap_count,
            "overflow_count": 0,
        },
        observation=_obs(candidates),
    )


def _window(samples: list[EvidenceSample]) -> ActionEvidenceWindow:
    return ActionEvidenceWindow(
        action_id="a1",
        agent_mode="PPO_GLOBAL",
        action_type="click_element",
        target_id="submit",
        action_timestamp=0.0,
        route_before="http://app.test/",
        started_at=0.0,
        ended_at=samples[-1].relative_ms / 1000.0,
        duration_ms=samples[-1].relative_ms,
        samples=samples,
        final_stability=TemporalState.STABLE,
        stability_reason=["STABLE_LAYOUT_QUIET"],
        terminal_snapshot=samples[-1].observation,
        timeline=[],
    )


def _action_info() -> dict:
    return {"action": {"action_type": "inspect_layout", "candidate_index": 0}}


def test_post_stable_drift_detected_and_early_stable_classified() -> None:
    stable = _sample(1, 500, [_candidate()], layout="stable")
    post = [_sample(2, 600, [_candidate(bbox=[120, 100, 100, 40])], layout="moved")]
    record = analyze_temporal_stability_window(
        before_observation=_obs([_candidate()]),
        action_info=_action_info(),
        temporal_window=_window([_sample(0, 0, [_candidate()], layout="a"), stable]),
        detector_anomalies=[],
        post_stable_samples=post,
        episode=1,
        step=1,
    )

    assert record["post_stable_drift"]["post_stable_layout_drift"] is True
    assert record["early_stable_classification"] == "EARLY_STABLE_LAYOUT"


def test_no_drift_stable_classification_and_detector_sample_index() -> None:
    stable = _sample(1, 500, [_candidate()], layout="stable")
    post = [_sample(2, 600, [_candidate()], layout="stable")]
    record = analyze_temporal_stability_window(
        before_observation=_obs([_candidate()]),
        action_info=_action_info(),
        temporal_window=_window([_sample(0, 0, [_candidate()], layout="a"), stable]),
        detector_anomalies=[],
        post_stable_samples=post,
        episode=1,
        step=1,
    )

    assert record["early_stable_classification"] == "NO_POST_STABLE_LAYOUT_DRIFT"
    assert record["stable_sample_index"] == record["detector_sample_index"]
    assert record["detector_after_snapshot_equals_final_sample"] is True


def test_final_snapshot_mismatch_detected() -> None:
    samples = [_sample(0, 0, [_candidate()], layout="a"), _sample(1, 500, [_candidate()], layout="stable")]
    window = _window(samples)
    window.terminal_snapshot = _obs([_candidate()], text="different")

    record = analyze_temporal_stability_window(
        before_observation=_obs([_candidate()]),
        action_info=_action_info(),
        temporal_window=window,
        detector_anomalies=[],
        post_stable_samples=[],
        episode=1,
        step=1,
    )

    assert record["final_snapshot_mismatch"] is True


def test_bbox_delta_metrics_correct() -> None:
    record = analyze_temporal_stability_window(
        before_observation=_obs([_candidate()]),
        action_info=_action_info(),
        temporal_window=_window([
            _sample(0, 0, [_candidate(bbox=[0, 0, 10, 10])], layout="a"),
            _sample(1, 500, [_candidate(bbox=[2, 4, 11, 13])], layout="b"),
        ]),
        detector_anomalies=[],
        post_stable_samples=[],
        episode=1,
        step=1,
    )

    assert record["bbox_delta_metrics"]["max_bbox_delta"] == 4
    assert record["bbox_delta_metrics"]["delta_count"] == 4


def test_transient_and_persistent_overlap_classification() -> None:
    before = _obs([_candidate()])
    stable = _sample(1, 500, [_candidate(), _panel()], layout="overlap")
    detector = detect_anomalies(before, stable.observation, _action_info())

    persistent = analyze_temporal_stability_window(
        before_observation=before,
        action_info=_action_info(),
        temporal_window=_window([_sample(0, 0, [_candidate()], layout="a"), stable]),
        detector_anomalies=detector,
        post_stable_samples=[_sample(2, 600, [_candidate(), _panel()], layout="overlap")],
        episode=1,
        step=1,
    )
    transient = analyze_temporal_stability_window(
        before_observation=before,
        action_info=_action_info(),
        temporal_window=_window([_sample(0, 0, [_candidate()], layout="a"), stable]),
        detector_anomalies=detector,
        post_stable_samples=[_sample(2, 600, [_candidate()], layout="resolved")],
        episode=1,
        step=1,
    )

    assert persistent["persistent_layout_candidate_keys"]
    assert transient["transient_layout_candidate_keys"]


def test_summary_metrics_and_runtime_isolation_shape() -> None:
    stable = _sample(1, 500, [_candidate()], layout="stable")
    record = analyze_temporal_stability_window(
        before_observation=_obs([_candidate()]),
        action_info=_action_info(),
        temporal_window=_window([_sample(0, 0, [_candidate()], layout="a"), stable]),
        detector_anomalies=[],
        post_stable_samples=[],
        episode=1,
        step=1,
    )

    summary = summarize_temporal_stability_diagnostics([record])

    assert summary["stable_count"] == 1
    assert summary["stable_precision_proxy"] == 1.0
    assert summary["layout_stable_precision_proxy"] == 1.0
