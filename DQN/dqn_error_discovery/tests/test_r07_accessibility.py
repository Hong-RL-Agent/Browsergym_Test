from services.anomaly_detection_service import detect_anomalies


def _candidate(
    bid: str,
    *,
    name: str = "",
    text: str = "",
    aria_label: str = "",
    label: str = "",
    alt: object = None,
    placeholder: str = "",
    role: str = "button",
    tag: str = "button",
    clickable: bool = True,
    visible: bool = True,
    visibility: float = 1.0,
    enabled: bool = True,
    disabled: bool = False,
    hidden: bool = False,
    aria_hidden: bool = False,
    focusable: object = None,
    tabindex: object = None,
    href: str = "",
    bbox: list[float] | None = None,
) -> dict:
    candidate = {
        "bid": bid,
        "name": name,
        "text": text,
        "aria_label": aria_label,
        "label": label,
        "placeholder": placeholder,
        "role": role,
        "tag": tag,
        "clickable": clickable,
        "visible": visible,
        "visibility": visibility,
        "enabled": enabled,
        "disabled": disabled,
        "hidden": hidden,
        "aria_hidden": aria_hidden,
        "href": href,
        "bbox": bbox or [100, 100, 100, 40],
    }
    if alt is not None:
        candidate["alt"] = alt
        candidate["alt_present"] = True
    if focusable is not None:
        candidate["focusable"] = focusable
    if tabindex is not None:
        candidate["tabindex"] = tabindex
    return candidate


def _observation(candidates: list[dict], *, width: int = 800, height: int = 600) -> dict:
    text = " ".join(str(candidate.get("text") or candidate.get("name") or "") for candidate in candidates)
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
            "cart_count": 0,
        },
        "runtime_signals": {},
        "candidate_elements": candidates,
        "layout_signals": {},
    }


def _r07_issues(observation: dict, *, last_action_error: bool = False) -> list[str]:
    anomalies = detect_anomalies(
        observation,
        observation,
        {
            "action": {"action_type": "inspect_accessibility", "candidate_index": 0},
            "last_action_error": last_action_error,
        },
    )
    return [
        anomaly.get("evidence", {}).get("issue_type")
        for anomaly in anomalies
        if anomaly.get("evidence", {}).get("rule_id") == "R07"
    ]


def test_r07_button_without_accessible_name_emits():
    obs = _observation([_candidate("icon-button")])

    assert "missing-accessible-name" in _r07_issues(obs)


def test_r07_icon_button_with_aria_label_does_not_emit():
    obs = _observation([_candidate("search", aria_label="Search")])

    assert not _r07_issues(obs)


def test_r07_button_with_visible_text_does_not_emit():
    obs = _observation([_candidate("save", name="Save", text="Save")])

    assert not _r07_issues(obs)


def test_r07_textbox_without_label_emits_form_label_issue():
    obs = _observation([_candidate("email", role="textbox", tag="input", clickable=True)])

    assert "missing-form-label" in _r07_issues(obs)


def test_r07_labeled_input_does_not_emit():
    obs = _observation([_candidate("email", name="Email", label="Email", role="textbox", tag="input")])

    assert not _r07_issues(obs)


def test_r07_placeholder_only_textbox_is_not_treated_as_label():
    obs = _observation([_candidate("email", placeholder="Enter email", role="textbox", tag="input")])

    assert "missing-form-label" in _r07_issues(obs)


def test_r07_native_button_without_tabindex_is_keyboard_accessible():
    obs = _observation([_candidate("native", name="Save", text="Save", role="button", tag="button")])

    assert "keyboard-unreachable" not in _r07_issues(obs)


def test_r07_div_role_button_not_focusable_emits_keyboard_issue():
    obs = _observation(
        [_candidate("custom", name="Open", text="Open", role="button", tag="div", focusable=False)]
    )

    assert "keyboard-unreachable" in _r07_issues(obs)


def test_r07_disabled_button_does_not_emit_by_itself():
    obs = _observation(
        [_candidate("disabled", role="button", tag="button", enabled=False, disabled=True)]
    )

    assert not _r07_issues(obs)


def test_r07_decorative_image_does_not_emit():
    obs = _observation(
        [_candidate("decorative", role="img", tag="img", clickable=False, alt="", aria_hidden=True)]
    )

    assert not _r07_issues(obs)


def test_r07_important_image_without_alt_emits():
    obs = _observation([_candidate("hero", role="img", tag="img", clickable=False)])

    assert "missing-image-alt" in _r07_issues(obs)


def test_r07_r05_overlap_situation_with_valid_name_does_not_emit_r07():
    obs = _observation(
        [
            _candidate("submit", name="Submit", text="Submit", role="button", tag="button"),
            _candidate("panel", name="Panel", text="Panel", role="generic", tag="div", clickable=False, bbox=[95, 95, 110, 50]),
        ]
    )

    assert not _r07_issues(obs)


def test_r07_r06_overflow_situation_with_valid_name_does_not_emit_r07():
    obs = _observation(
        [_candidate("submit", name="Submit", text="Submit", role="button", tag="button", bbox=[760, 100, 100, 40])]
    )

    assert not _r07_issues(obs)


def test_r07_action_error_without_accessibility_evidence_does_not_emit():
    obs = _observation([_candidate("save", name="Save", text="Save")])

    assert not _r07_issues(obs, last_action_error=True)
