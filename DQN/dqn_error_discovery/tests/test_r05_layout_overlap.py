from services.anomaly_detection_service import detect_anomalies


def _candidate(
    bid: str,
    bbox: list[float],
    *,
    text: str = "",
    role: str = "generic",
    tag: str = "div",
    clickable: bool = False,
    visible: bool = True,
    visibility: float = 1.0,
    class_name: str = "",
) -> dict:
    return {
        "bid": bid,
        "name": text,
        "text": text,
        "role": role,
        "tag": tag,
        "clickable": clickable,
        "visible": visible,
        "visibility": visibility,
        "bbox": bbox,
        "class_name": class_name,
        "enabled": True,
    }


def _button(bid: str, bbox: list[float]) -> dict:
    return _candidate(bid, bbox, text="Submit", role="button", tag="button", clickable=True)


def _observation(candidates: list[dict], *, modal: bool = False, loading: bool = False) -> dict:
    text = "Loading" if loading else " ".join(candidate.get("text", "") for candidate in candidates)
    return {
        "page_state": {
            "url": "http://app.test/page",
            "title": "Page",
            "page_text": text,
            "page_text_sample": text,
            "page_text_length": len(text),
            "dom_node_count": len(candidates),
            "viewport_width": 800,
            "viewport_height": 600,
            "has_modal_or_dialog": modal,
            "cart_count": 0,
        },
        "runtime_signals": {"has_modal_or_dialog": modal},
        "candidate_elements": candidates,
        "layout_signals": {"layout_overlap_count": 0},
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


def test_r05_large_button_occlusion_emits_layout_overlap():
    obs = _observation(
        [
            _button("submit", [100, 100, 100, 40]),
            _candidate("panel", [90, 95, 120, 50], text="Panel", role="generic", tag="div", class_name="floating-panel"),
        ]
    )

    assert "layout-overlap" in _types(obs)


def test_r05_tiny_edge_overlap_does_not_emit():
    obs = _observation(
        [
            _button("submit", [100, 100, 100, 100]),
            _candidate("panel", [198, 100, 10, 100], text="Panel", role="generic", tag="div", class_name="floating-panel"),
        ]
    )

    assert "layout-overlap" not in _types(obs)


def test_r05_button_internal_span_does_not_emit():
    obs = _observation(
        [
            _button("submit", [100, 100, 120, 40]),
            _candidate("submit-label", [115, 110, 60, 20], text="Submit", role="generic", tag="span"),
        ]
    )

    assert "layout-overlap" not in _types(obs)


def test_r05_card_container_with_button_does_not_emit():
    obs = _observation(
        [
            _candidate("card", [80, 80, 220, 160], text="Product card", role="group", tag="div"),
            _button("add", [120, 180, 90, 36]),
        ]
    )

    assert "layout-overlap" not in _types(obs)


def test_r05_independent_floating_panel_covering_button_emits():
    obs = _observation(
        [
            _button("save", [100, 100, 120, 50]),
            _candidate("floating", [110, 105, 100, 45], text="Floating", role="generic", tag="div", class_name="floating-panel"),
        ]
    )

    assert "layout-overlap" in _types(obs)


def test_r05_non_actionable_text_overlap_does_not_emit():
    obs = _observation(
        [
            _candidate("text-a", [100, 100, 100, 40], text="Decorative", role="generic", tag="span"),
            _candidate("text-b", [105, 105, 100, 40], text="Icon", role="generic", tag="span"),
        ]
    )

    assert "layout-overlap" not in _types(obs)


def test_r05_same_bid_duplicate_candidate_does_not_emit():
    obs = _observation(
        [
            _button("submit", [100, 100, 100, 40]),
            _candidate("submit", [100, 100, 100, 40], text="Submit", role="generic", tag="div", class_name="floating-panel"),
        ]
    )

    assert "layout-overlap" not in _types(obs)


def test_r05_viewport_outside_button_without_overlap_does_not_emit():
    obs = _observation([_button("submit", [100, 700, 100, 40])])

    assert "layout-overlap" not in _types(obs)


def test_r05_normal_modal_overlay_does_not_emit():
    obs = _observation(
        [
            _button("background-submit", [100, 100, 100, 40]),
            _candidate("modal", [80, 80, 300, 200], text="Dialog", role="dialog", tag="div"),
        ],
        modal=True,
    )

    assert "layout-overlap" not in _types(obs)


def test_r05_transparent_stale_overlay_covering_button_emits():
    obs = _observation(
        [
            _button("submit", [100, 100, 100, 40]),
            _candidate("stale-overlay", [95, 95, 110, 50], text="", role="generic", tag="div", class_name="stale-overlay"),
        ]
    )

    assert "layout-overlap" in _types(obs)
