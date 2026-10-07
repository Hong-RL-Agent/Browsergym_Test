from __future__ import annotations

from services.temporal_evidence_service import (
    NetworkState,
    TemporalEvidenceConfig,
    TemporalState,
    TemporalStateTracker,
    EvidenceSample,
)


def _sample(relative_ms: int, *, semantic: str = "dom", layout: str = "layout", url: str = "http://app.test/", loading: bool = False, active: int = 0) -> EvidenceSample:
    return EvidenceSample(
        timestamp=relative_ms / 1000.0,
        relative_ms=relative_ms,
        url=url,
        semantic_dom_fingerprint=semantic,
        layout_fingerprint=layout,
        loading_state={"loading_visible": loading, "visible_loading_indicator_count": int(loading), "target_disabled_after_action": False},
        network_state=NetworkState(active_request_count=active, blocking_active_request_count=active),
        console_delta=[],
        target_state={"present": True, "enabled": True},
        candidate_related_element_state={},
        dom_semantic_summary={},
        dom_mutation_summary={},
        layout_summary={},
        observation={},
    )


def _config() -> TemporalEvidenceConfig:
    return TemporalEvidenceConfig(sample_interval_ms=100, minimum_observation_ms=100, stable_quiet_window_ms=200, maximum_observation_ms=600)


def test_tracker_reports_transient_while_semantic_dom_changes() -> None:
    tracker = TemporalStateTracker(_config())

    assert tracker.observe(_sample(0, semantic="a")) == TemporalState.OBSERVING
    assert tracker.observe(_sample(100, semantic="b")) == TemporalState.TRANSIENT
    assert tracker.observe(_sample(200, semantic="b")) == TemporalState.OBSERVING
    assert tracker.observe(_sample(300, semantic="b")) == TemporalState.STABLE


def test_tracker_reports_transient_while_layout_changes() -> None:
    tracker = TemporalStateTracker(_config())

    tracker.observe(_sample(0, layout="a"))
    assert tracker.observe(_sample(100, layout="b")) == TemporalState.TRANSIENT
    assert tracker.observe(_sample(300, layout="b")) == TemporalState.STABLE


def test_tracker_waits_for_loading_and_network_to_settle() -> None:
    tracker = TemporalStateTracker(_config())

    assert tracker.observe(_sample(0, loading=True)) == TemporalState.TRANSIENT
    assert tracker.observe(_sample(100, active=1)) == TemporalState.TRANSIENT
    assert tracker.observe(_sample(200)) == TemporalState.TRANSIENT
    assert tracker.observe(_sample(400)) == TemporalState.STABLE
    assert "STABLE_AFTER_NETWORK" in tracker.reasons
    assert "STABLE_AFTER_LOADING" in tracker.reasons


def test_tracker_timeout_is_not_a_defect_confirmation() -> None:
    tracker = TemporalStateTracker(_config())
    last = _sample(600, active=1)

    assert tracker.timeout(last) == TemporalState.TIMED_OUT_UNSTABLE
    assert "MAX_WINDOW_TIMEOUT" in tracker.reasons
    assert "NETWORK_STILL_ACTIVE" in tracker.reasons
