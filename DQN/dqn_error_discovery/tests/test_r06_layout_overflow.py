from services.anomaly_detection_service import detect_anomalies


def _candidate(
    bid: str,
    bbox: list[float],
    *,
    text: str = "Submit",
    role: str = "button",
    tag: str = "button",
    clickable: bool = True,
    visible: bool = True,
    visibility: float = 1.0,
    **extra,
) -> dict:
    item = {
        "bid": bid,
        "name": text,
        "text": text,
        "role": role,
        "tag": tag,
        "clickable": clickable,
        "visible": visible,
        "visibility": visibility,
        "bbox": bbox,
        "enabled": True,
    }
    item.update(extra)
    return item


def _observation(
    candidates: list[dict],
    *,
    width: int = 400,
    height: int = 800,
    document_height: int = 800,
    document_width: int = 400,
) -> dict:
    text = " ".join(candidate.get("text", "") for candidate in candidates)
    return {
        "page_state": {
            "url": "http://app.test/page",
            "title": "Page",
            "page_text": text,
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": len(candidates),
            "viewport_width": width,
            "viewport_height": height,
            "document_height": document_height,
            "document_width": document_width,
            "cart_count": 0,
        },
        "runtime_signals": {},
        "candidate_elements": candidates,
        "layout_signals": {},
    }


def _types(observation: dict) -> list[str]:
    return [
        item["type"]
        for item in detect_anomalies(
            observation,
            observation,
            {"action": {"action_type": "inspect_layout", "candidate_index": 0}},
        )
    ]


def test_r06_right_half_clipped_button_emits():
    obs = _observation([_candidate("cta", [350, 100, 150, 40], overflow_x="hidden", can_reach_by_scroll_x=False)])

    assert "layout-overflow" in _types(obs)


def test_r06_one_pixel_boundary_excess_does_not_emit():
    obs = _observation([_candidate("cta", [301, 100, 100, 40])])

    assert "layout-overflow" not in _types(obs)


def test_r06_normal_vertical_scroll_does_not_emit():
    obs = _observation([_candidate("below-fold", [100, 1500, 100, 40])], document_height=3000)

    assert "layout-overflow" not in _types(obs)


def test_r06_top_clipped_actionable_target_emits():
    obs = _observation([_candidate("top", [100, -80, 100, 100], overflow_y="hidden", can_reach_by_scroll_y=False)])

    assert "layout-overflow" in _types(obs)


def test_r06_decorative_image_clipping_does_not_emit():
    obs = _observation([_candidate("hero", [350, 100, 150, 100], role="img", tag="img", clickable=False)])

    assert "layout-overflow" not in _types(obs)


def test_r06_mobile_horizontal_overflow_emits():
    obs = _observation([_candidate("mobile-cta", [330, 100, 270, 50], overflow_x="hidden", can_reach_by_scroll_x=False)], width=390, height=800)

    assert "layout-overflow" in _types(obs)


def test_r06_desktop_ok_mobile_clipped_only_mobile_emits():
    desktop = _observation([_candidate("cta", [600, 100, 160, 50])], width=1024, height=800)
    mobile = _observation([_candidate("cta", [330, 100, 160, 50], overflow_x="hidden", can_reach_by_scroll_x=False)], width=390, height=800)

    assert "layout-overflow" not in _types(desktop)
    assert "layout-overflow" in _types(mobile)


def test_r06_r05_overlap_situation_does_not_emit_overflow():
    obs = _observation(
        [
            _candidate("submit", [100, 100, 100, 40]),
            _candidate("overlay", [95, 95, 110, 50], text="", role="generic", tag="div", clickable=False),
        ]
    )

    assert "layout-overflow" not in _types(obs)


def test_r06_invalid_bbox_does_not_emit():
    obs = _observation([_candidate("invalid", [100, 100, 0, 0])])

    assert "layout-overflow" not in _types(obs)


def test_r06_visible_ratio_high_does_not_emit():
    obs = _observation([_candidate("almost-visible", [305, 100, 100, 40])])

    assert "layout-overflow" not in _types(obs)


def test_r06_visible_ratio_low_emits():
    obs = _observation([_candidate("mostly-hidden", [380, 100, 100, 40], overflow_x="hidden", can_reach_by_scroll_x=False)])

    assert "layout-overflow" in _types(obs)


def test_r06_horizontal_page_overflow_alone_does_not_emit():
    obs = _observation([_candidate("visible", [100, 100, 100, 40])], document_width=850)

    assert "layout-overflow" not in _types(obs)


def test_r06_visible_clickable_partial_offscreen_without_impact_is_geometry_only():
    obs = _observation([_candidate("usable", [350, 100, 150, 40], hit_test_valid=True, can_reach_by_scroll_x=True)])

    assert "layout-overflow" not in _types(obs)


def test_r06_intentional_horizontal_scroll_container_is_not_a_defect():
    obs = _observation([_candidate(
        "carousel-item", [350, 100, 150, 40], role="link", tag="a",
        overflow_x="auto", scroll_width=800, client_width=400,
    )])

    assert "layout-overflow" not in _types(obs)


def test_r06_overflow_hidden_clipped_functional_content_emits():
    obs = _observation([_candidate(
        "important", [350, 100, 150, 40], text="Continue checkout",
        overflow_x="hidden", scroll_width=400, client_width=400,
        can_reach_by_scroll_x=False, hit_test_blocked=True,
    )])

    anomalies = detect_anomalies(obs, obs, {"action": {"action_type": "inspect_layout", "candidate_index": 0}})
    r06 = next(item for item in anomalies if item["rule_id"] == "R06")
    assert r06["evidence"]["functional_impact"] in {"MEDIUM", "HIGH"}


def test_r06_hit_test_valid_usable_target_is_not_a_strong_candidate():
    obs = _observation([_candidate(
        "usable", [350, 100, 150, 40], hit_test_valid=True,
        overflow_x="hidden", can_reach_by_scroll_x=False,
    )])

    assert "layout-overflow" not in _types(obs)


def test_r06_evidence_separates_viewport_and_container_overflow():
    obs = _observation([_candidate(
        "clipped", [350, 100, 150, 40], overflow_x="hidden",
        can_reach_by_scroll_x=False, container_bbox=[0, 0, 400, 800],
        scroll_width=600, client_width=400,
    )])
    r06 = next(item for item in detect_anomalies(obs, obs, {"action": {"action_type": "inspect_layout"}}) if item["rule_id"] == "R06")
    assert r06["evidence"]["viewport_overflow"]
    assert r06["evidence"]["container_overflow"]


def test_r06_keyboard_reachability_is_recorded():
    obs = _observation([_candidate(
        "keyboard-target", [350, 100, 150, 40], overflow_x="hidden",
        can_reach_by_scroll_x=False, tabindex=0,
    )])
    r06 = next(item for item in detect_anomalies(obs, obs, {"action": {"action_type": "inspect_layout"}}) if item["rule_id"] == "R06")
    assert r06["evidence"]["keyboard_reachable"] is True
