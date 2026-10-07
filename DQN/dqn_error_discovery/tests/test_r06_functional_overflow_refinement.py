from services.anomaly_detection_service import detect_anomalies
from dqn_error_discovery.framework_dev.r06_refinement_evaluator import _n_d03_probe


def _observation(candidate):
    return {
        "page_state": {"viewport_width": 390, "viewport_height": 844, "document_height": 844},
        "runtime_signals": {},
        "candidate_elements": [candidate],
    }


def _detect(candidate):
    obs = _observation(candidate)
    return detect_anomalies(obs, obs, {"action": {"action_type": "inspect_layout", "candidate_index": 0}})


def test_visible_clickable_offscreen_target_without_functional_evidence_is_suppressed():
    assert not any(item.get("rule_id") == "R06" for item in _detect({
        "bid": "cta", "role": "button", "tag": "button", "text": "Continue",
        "visible": True, "visibility": 1.0, "enabled": True, "clickable": True,
        "bbox": [350, 100, 150, 40],
    }))


def test_intentional_scrollable_content_is_suppressed():
    assert not any(item.get("rule_id") == "R06" for item in _detect({
        "bid": "item", "role": "link", "tag": "a", "text": "More",
        "visible": True, "visibility": 1.0, "enabled": True, "clickable": True,
        "bbox": [350, 100, 150, 40], "overflow_x": "auto",
        "scroll_width": 800, "client_width": 390,
    }))


def test_hidden_clipped_functional_content_is_retained():
    items = _detect({
        "bid": "content", "role": "generic", "tag": "span",
        "text": "Important content cannot be read completely", "visible": True,
        "visibility": 1.0, "enabled": True, "clickable": False,
        "bbox": [0, 0, 360, 28], "container_bbox": [0, 0, 190, 28],
        "overflow_x": "hidden", "scroll_width": 360, "client_width": 190,
        "can_reach_by_scroll_x": False, "hit_test_blocked": True,
    })
    r06 = next(item for item in items if item.get("rule_id") == "R06")
    assert r06["evidence"]["functional_impact"] in {"MEDIUM", "HIGH"}


def test_hit_test_valid_usable_target_is_not_r06():
    assert not any(item.get("rule_id") == "R06" for item in _detect({
        "bid": "cta", "role": "button", "tag": "button", "text": "Continue",
        "visible": True, "visibility": 1.0, "enabled": True, "clickable": True,
        "bbox": [350, 100, 150, 40], "hit_test_valid": True,
        "overflow_x": "hidden", "can_reach_by_scroll_x": False,
    }))


def test_nd03_micro_probe_retains_r06_with_functional_evidence():
    probe = _n_d03_probe()
    assert probe["success"] is True
    assert probe["evidence"]["functional_impact"] in {"MEDIUM", "HIGH"}
    assert probe["evidence"]["overflow_x"] == "hidden"
    assert probe["evidence"]["can_reach_by_scroll_x"] is False


def test_live_collector_metadata_drives_existing_r06_gate():
    items = _detect({
        "bid": "live-content", "role": "generic", "tag": "span",
        "text": "Live clipped content", "visible": True, "visibility": 1.0,
        "enabled": True, "clickable": False, "bbox": [0, 0, 360, 28],
        "interactive_layout_evidence": {
            "evidence_status": "AVAILABLE",
            "container_bbox": [0, 0, 190, 28],
            "container_overflow": True,
            "container_clipping": True,
            "overflow_x": "hidden",
            "overflow_y": "hidden",
            "scroll_width": 360,
            "client_width": 190,
            "scroll_height": 28,
            "client_height": 28,
            "reachable_via_scroll_x": False,
            "reachable_via_scroll_y": False,
            "hit_test_valid": False,
            "hit_test_blocked": True,
            "keyboard_reachable": False,
        },
    })

    r06 = next(item for item in items if item.get("rule_id") == "R06")
    assert r06["evidence"]["functional_impact"] == "HIGH"


def test_document_scroll_reachable_viewport_target_is_not_r06():
    items = _detect({
        "bid": "below-fold", "role": "button", "tag": "button", "text": "Continue below",
        "visible": True, "visibility": 1.0, "enabled": True, "clickable": True,
        "bbox": [20, 1100, 140, 40],
        "interactive_layout_evidence": {
            "evidence_status": "AVAILABLE",
            "overflow_x": "visible", "overflow_y": "visible",
            "viewport_overflow": True, "container_overflow": False,
            "clipping_ratio": 1.0, "visible_ratio": 0.0,
            "document_can_scroll_y": True, "document_reachable_y": True,
            "reachable_via_document_scroll": True, "reachable_via_scroll_y": True,
            "reachability_source": "DOCUMENT_SCROLL",
            "effective_clipping_ancestor": None,
            "hit_test_valid": False, "hit_test_blocked": False,
            "keyboard_reachable": True,
        },
    })
    assert not any(item.get("rule_id") == "R06" for item in items)


def test_document_scroll_does_not_override_hidden_clipping_ancestor():
    items = _detect({
        "bid": "hidden-content", "role": "generic", "tag": "span",
        "text": "Important content permanently clipped", "visible": True,
        "visibility": 1.0, "enabled": True, "clickable": False,
        "bbox": [0, 0, 360, 28],
        "interactive_layout_evidence": {
            "evidence_status": "AVAILABLE",
            "container_bbox": [0, 0, 190, 28], "container_overflow": True,
            "container_clipping": True, "overflow_x": "hidden", "overflow_y": "hidden",
            "scroll_width": 360, "client_width": 190,
            "can_reach_by_scroll_x": False, "can_reach_by_scroll_y": False,
            "document_can_scroll_y": True, "document_reachable_y": True,
            "reachable_via_document_scroll": True,
            "blocking_clipping_ancestor": True,
            "hit_test_valid": False, "hit_test_blocked": True,
            "keyboard_reachable": False,
        },
    })
    r06 = next(item for item in items if item.get("rule_id") == "R06")
    assert r06["evidence"]["functional_impact"] == "HIGH"
