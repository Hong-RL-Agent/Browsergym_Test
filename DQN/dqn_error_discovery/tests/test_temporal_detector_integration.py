from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from models.observation_encoder import ObservationEncoder
from models.action_space import ActionSpace
from services.anomaly_detection_service import detect_anomalies
from services.temporal_evidence_service import RawEvidenceCollector, TemporalEvidenceConfig, detect_anomalies_with_temporal_evidence


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep_ms(self, ms: int) -> None:
        self.now += ms / 1000.0


def _config() -> TemporalEvidenceConfig:
    return TemporalEvidenceConfig(sample_interval_ms=100, minimum_observation_ms=100, stable_quiet_window_ms=200, maximum_observation_ms=700)


def _button(bid: str = "buy-1", bbox: list[float] | None = None) -> dict:
    return {
        "bid": bid,
        "name": "Add to cart",
        "text": "Add to cart",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "is_purchase_action": True,
        "bbox": bbox or [100, 100, 120, 40],
    }


def _panel() -> dict:
    return {
        "bid": "floating",
        "name": "Floating",
        "text": "Floating",
        "role": "generic",
        "tag": "div",
        "clickable": False,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "class_name": "floating-panel",
        "bbox": [105, 105, 110, 35],
    }


def _observation(text: str = "Product Add to cart", *, cart_count: int = 0, candidates: list[dict] | None = None, network: dict | None = None) -> dict:
    return {
        "page_state": {
            "url": "http://example.test/product",
            "title": "Product",
            "page_text_length": len(text),
            "page_text_sample": text,
            "dom_node_count": len(candidates or [_button()]),
            "cart_count_detected": True,
            "cart_count": cart_count,
            "cart_text": f"Cart ({cart_count})",
            "has_modal_or_dialog": False,
            "has_form": False,
            "viewport_width": 800,
            "viewport_height": 600,
        },
        "runtime_signals": {"temporal_network": network or {"active_requests": [], "completed_requests": [], "failed_requests": []}},
        "candidate_elements": candidates or [_button()],
    }


def _window(observations: list[dict]):
    clock = FakeClock()
    index = {"value": 0}

    def sampler() -> dict:
        value = observations[min(index["value"], len(observations) - 1)]
        index["value"] += 1
        return value

    collector = RawEvidenceCollector(sampler, config=_config(), sleep_ms=clock.sleep_ms, time_func=clock.time)
    return collector.collect({"action_id": "act-1", "agent_mode": "PPO_GLOBAL", "action_type": "click_element", "target_id": "buy-1", "candidate_index": 0, "route_before": "http://example.test/product"})


def _types(anomalies: list[dict]) -> list[str]:
    return [item["type"] for item in anomalies]


def test_async_dom_update_uses_final_stable_snapshot_and_suppresses_r01_candidate() -> None:
    before = _observation()
    loading = _observation("Product Add to cart Loading")
    final = _observation("Product Added", cart_count=1)
    window = _window([loading, final, final, final])

    anomalies = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "click_element", "candidate_index": 0}}, window)

    assert "button-no-response" not in _types(anomalies)
    assert window.terminal_snapshot["page_state"]["cart_count"] == 1


def test_true_no_response_remains_candidate_and_canonical_key_is_unchanged() -> None:
    before = _observation()
    after = deepcopy(before)
    direct = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})
    window = _window([after, after, after, after])

    temporal = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "click_element", "candidate_index": 0}}, window)

    assert "button-no-response" in _types(temporal)
    assert temporal[0]["signature"] == direct[0]["signature"]
    assert temporal[0]["trigger_action_id"] == "act-1"
    assert temporal[0]["stability_state"] == "STABLE"


def test_transient_layout_overlap_resolves_before_detector() -> None:
    before = _observation(candidates=[_button()])
    transient = _observation(candidates=[_button(), _panel()])
    final = _observation(candidates=[_button()])
    window = _window([transient, final, final, final])

    anomalies = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "inspect_layout", "candidate_index": 0}}, window)

    assert "layout-overlap" not in _types(anomalies)


def test_stable_layout_defect_remains_detectable() -> None:
    before = _observation(candidates=[_button()])
    defect = _observation(candidates=[_button(), _panel()])
    window = _window([defect, defect, defect, defect])

    anomalies = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "inspect_layout", "candidate_index": 0}}, window)

    assert "layout-overlap" in _types(anomalies)


def test_network_starts_then_settles_blocks_immediate_no_response_fp() -> None:
    before = _observation()
    active = _observation(network={"active_requests": [{"request_id": "r1", "url": "http://example.test/api/cart", "resource_type": "fetch"}], "completed_requests": [], "failed_requests": []})
    complete = _observation(network={"active_requests": [], "completed_requests": [{"request_id": "r1", "url": "http://example.test/api/cart", "method": "POST", "status": 200}], "failed_requests": []})
    window = _window([active, complete, complete, complete])

    anomalies = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "click_element", "candidate_index": 0}}, window)

    assert "button-no-response" not in _types(anomalies)


def test_static_asset_does_not_count_as_functional_feedback() -> None:
    before = _observation()
    asset = _observation(network={"active_requests": [], "completed_requests": [{"request_id": "asset", "url": "http://example.test/app.js", "resource_type": "script", "status": 200}], "failed_requests": []})
    window = _window([asset, asset, asset, asset])

    anomalies = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "click_element", "candidate_index": 0}}, window)

    assert window.samples[-1].network_state.completed_requests
    assert not any(item.get("type") in {"api-server-error", "network-failure"} for item in anomalies)


def test_functional_http_failure_remains_failure_evidence() -> None:
    before = _observation()
    failed = _observation(network={"active_requests": [], "completed_requests": [{"request_id": "api", "url": "http://example.test/api/cart", "method": "POST", "resource_type": "fetch", "status": 500}], "failed_requests": []})
    window = _window([failed, failed, failed, failed])

    anomalies = detect_anomalies_with_temporal_evidence(before, {"action": {"action_type": "click_element", "candidate_index": 0}}, window)

    assert any(item.get("type") in {"api-server-error", "api-ui-mismatch"} for item in anomalies)


def test_detector_taxonomy_and_policy_dimensions_are_unchanged() -> None:
    assert ObservationEncoder().get_obs_dim() == 635
    assert ActionSpace().get_action_dim() == 608
    manifest = json.loads((Path(__file__).resolve().parents[2] / "artifacts" / "models" / "dqn_local_specialist_final_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dqn_obs_dim"] == 976
    before = _observation()
    after = deepcopy(before)
    anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

    assert {item["rule_id"] for item in anomalies} <= {f"R{index:02d}" for index in range(1, 16)}
