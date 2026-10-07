from __future__ import annotations

from copy import deepcopy

from services.temporal_evidence_service import RawEvidenceCollector, TemporalEvidenceConfig, TemporalState


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep_ms(self, ms: int) -> None:
        self.now += ms / 1000.0


def _candidate(enabled: bool = True) -> dict:
    return {
        "bid": "submit",
        "name": "Submit",
        "text": "Submit",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "enabled": enabled,
        "bbox": [100, 100, 120, 40],
    }


def _obs(text: str = "Ready", *, loading: bool = False, active_requests: list[dict] | None = None) -> dict:
    candidates = [_candidate(enabled=not loading)]
    if loading:
        candidates.append({"bid": "spinner", "text": "Loading", "role": "progressbar", "visible": True, "bbox": [0, 0, 20, 20]})
    return {
        "page_state": {
            "url": "http://app.test/form",
            "title": "Form",
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": len(candidates),
            "viewport_width": 800,
            "viewport_height": 600,
        },
        "runtime_signals": {
            "temporal_network": {
                "active_requests": active_requests or [],
                "completed_requests": [],
                "failed_requests": [],
            }
        },
        "candidate_elements": candidates,
    }


def _config() -> TemporalEvidenceConfig:
    return TemporalEvidenceConfig(
        sample_interval_ms=100,
        minimum_observation_ms=100,
        stable_quiet_window_ms=200,
        maximum_observation_ms=700,
        debug_timeline=True,
    )


def test_raw_evidence_collector_collects_multiple_samples_and_action_id() -> None:
    clock = FakeClock()
    collector = RawEvidenceCollector(lambda: _obs(), config=_config(), sleep_ms=clock.sleep_ms, time_func=clock.time)

    window = collector.collect({"action_id": "a-1", "agent_mode": "PPO_GLOBAL", "action_type": "click_element", "target_id": "submit", "route_before": "/form"})

    assert len(window.samples) > 1
    assert window.action_id == "a-1"
    assert window.agent_mode == "PPO_GLOBAL"
    assert window.action_type == "click_element"
    assert window.target_id == "submit"
    assert window.final_stability == TemporalState.STABLE
    assert "STABLE_SEMANTIC_QUIET" in window.stability_reason
    assert window.timeline[0]["t"].startswith("T+")


def test_delayed_dom_update_and_transient_loading_are_tracked_until_stable() -> None:
    clock = FakeClock()

    def sampler() -> dict:
        if clock.time() < 0.2:
            return _obs("Saving", loading=True)
        return _obs("Saved")

    collector = RawEvidenceCollector(sampler, config=_config(), sleep_ms=clock.sleep_ms, time_func=clock.time)
    window = collector.collect({"action_id": "a-2", "action_type": "click_element", "target_id": "submit"})

    assert any(sample.loading_state["loading_visible"] for sample in window.samples)
    assert window.final_stability == TemporalState.STABLE
    assert window.terminal_snapshot["page_state"]["page_text_sample"] == "Saved"


def test_long_lived_background_request_does_not_deadlock_tracker() -> None:
    clock = FakeClock()
    request = {"request_id": "ws-1", "url": "ws://app.test/socket", "resource_type": "websocket", "relative_ms": 0}
    collector = RawEvidenceCollector(lambda: _obs(active_requests=[request]), config=_config(), sleep_ms=clock.sleep_ms, time_func=clock.time)

    window = collector.collect({"action_id": "a-3", "action_type": "click_element", "target_id": "submit"})

    assert window.final_stability == TemporalState.STABLE
    assert window.samples[-1].network_state.active_request_count == 1
    assert window.samples[-1].network_state.blocking_active_request_count == 0


def test_maximum_window_timeout_works_without_fixed_sleep() -> None:
    clock = FakeClock()

    def sampler() -> dict:
        obs = _obs(f"Tick {clock.time()}")
        obs["runtime_signals"]["temporal_network"]["active_requests"] = [
            {"request_id": "api", "url": "http://app.test/api/save", "resource_type": "fetch"}
        ]
        return obs

    collector = RawEvidenceCollector(sampler, config=_config(), sleep_ms=clock.sleep_ms, time_func=clock.time)
    window = collector.collect({"action_id": "a-4", "action_type": "click_element", "target_id": "submit"})

    assert window.final_stability == TemporalState.TIMED_OUT_UNSTABLE
    assert "MAX_WINDOW_TIMEOUT" in window.stability_reason
    assert [sample.relative_ms for sample in window.samples] != [0, 2000]


def test_semantic_fingerprint_ignores_timestamp_like_text_noise() -> None:
    clock = FakeClock()
    first = _obs("Saved at 12:01:02 id-123")
    second = deepcopy(first)
    second["page_state"]["page_text_sample"] = "Saved at 12:01:03 id-456"
    samples = [first, second, second, second]

    collector = RawEvidenceCollector(lambda: samples.pop(0) if samples else second, config=_config(), sleep_ms=clock.sleep_ms, time_func=clock.time)
    window = collector.collect({"action_id": "a-5", "action_type": "click_element", "target_id": "submit"})

    assert window.samples[0].semantic_dom_fingerprint == window.samples[-1].semantic_dom_fingerprint
