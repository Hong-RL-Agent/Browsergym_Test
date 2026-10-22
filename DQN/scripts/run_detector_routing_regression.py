"""Small dependency-free regression for detector evidence and routing contracts."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from services.anomaly_detection_service import detect_anomalies  # noqa: E402
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context  # noqa: E402
from dqn_error_discovery.hybrid.strategy_router import route_candidate  # noqa: E402

OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/detector_routing_regression.json"


def obs(*, url="http://example.test/page", candidates=None, runtime=None, page=None):
    return {
        "page_state": {"url": url, "page_text_length": 20, "page_text_sample": "page", **(page or {})},
        "runtime_signals": dict(runtime or {}),
        "candidate_elements": list(candidates or []),
    }


def run() -> dict:
    results = {}
    # R01: target action with no observable response.
    candidate = {"bid": "buy-1", "name": "Add to cart", "text": "Add to cart", "role": "button", "tag": "button", "clickable": True, "visible": True, "visibility": 1.0, "is_purchase_action": True}
    before = obs(candidates=[candidate], page={"cart_count_detected": True, "cart_count": 0, "cart_text": "Cart (0)"})
    r01 = detect_anomalies(before, copy.deepcopy(before), {"action": {"action_type": "click_element", "candidate_index": 0}, "action_type": "click_element"})
    results["R01"] = {"pass": any(x.get("rule_id") == "R01" for x in r01), "rules": [x.get("rule_id") for x in r01]}

    # R02: href-backed navigation to the wrong entity.
    nav_candidate = {"bid": "product-17", "name": "Product 17", "text": "Product 17", "role": "link", "tag": "a", "clickable": True, "visible": True, "visibility": 1.0, "href": "/products/17", "product_id": "17"}
    b02 = obs(url="http://shop.test/products", candidates=[nav_candidate])
    a02 = obs(url="http://shop.test/products/21", candidates=[nav_candidate], page={"entity_id": "21"})
    r02 = detect_anomalies(b02, a02, {"action": {"action_type": "click_element", "candidate_index": 0}, "action_type": "click_element"})
    results["R02"] = {"pass": any(x.get("rule_id") == "R02" for x in r02), "rules": [x.get("rule_id") for x in r02]}

    # R09: a filter request whose result entities violate the requested value.
    b09 = obs(candidates=[{"bid": "filter-1", "role": "button", "tag": "button", "clickable": True, "visible": True, "bbox": [0, 0, 10, 10]}])
    a09 = obs(candidates=[{"bid": "product-1", "is_result": True, "product_id": "1", "brand": "Other"}], runtime={"result_entities": [{"product_id": "1", "brand": "Other"}]})
    r09 = detect_anomalies(b09, a09, {"action": {"action_type": "click_element", "candidate_index": 0}, "action_type": "click_element", "requested_operation": "filter", "requested_key": "brand", "requested_value": "Expected"})
    results["R09"] = {"pass": any(x.get("rule_id") == "R09" for x in r09), "rules": [x.get("rule_id") for x in r09]}

    # R12: public runtime state shows a completed compare with dropped filters.
    before_state = {"compare": {"summary": {"count": 0}}, "filters": {"category": "laptop", "brands": [1]}, "lastSearch": None}
    after_state = {"compare": {"summary": {"count": 2}}, "filters": {"category": "laptop", "brands": []}, "lastSearch": {"category": "laptop", "brands": [1]}}
    b12 = obs(runtime={"external_runtime_state": {"compare_lab_state": before_state}}, candidates=[{"bid": "return-1", "role": "button", "clickable": True, "visible": True, "bbox": [0, 0, 10, 10]}])
    a12 = obs(runtime={"external_runtime_state": {"compare_lab_state": after_state}}, candidates=[{"bid": "return-1", "role": "button", "clickable": True, "visible": True, "bbox": [0, 0, 10, 10]}])
    r12 = detect_anomalies(b12, a12, {"action": {"action_type": "click_element", "candidate_index": 0, "action_id": 1}, "action_type": "click_element"})
    r12_item = next((x for x in r12 if x.get("rule_id") == "R12"), None)
    r12_ctx = derive_reproduction_context(r12_item).as_dict() if r12_item else {}
    results["R12"] = {"pass": r12_item is not None and r12_ctx.get("reproduction_strategy") == "SEQUENCE_REPLAY", "rules": [x.get("rule_id") for x in r12], "strategy": r12_ctx.get("reproduction_strategy")}

    # R07 must remain observation-owned even when a triggering click has a target.
    r07_ctx = derive_reproduction_context({"type": "accessibility-issue", "evidence": {"rule_id": "R07"}, "trigger_action_type": "click_element", "trigger_target_identity": "bid:1"}).as_dict()
    r07_route = route_candidate({"reproduction_strategy": r07_ctx.get("reproduction_strategy"), "requires_action_reproduction": False})
    results["R07_routing"] = {"pass": r07_route.owner == "OBSERVATION_RECHECK", "owner": r07_route.owner, "strategy": r07_route.strategy}
    output = {"artifact_type": "detector_and_routing_regression", "gt_leakage": 0, "pass": all(item.get("pass") for item in results.values()), "results": results}
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    raise SystemExit(0 if run().get("pass") else 1)
