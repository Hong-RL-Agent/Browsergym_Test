from services.anomaly_detection_service import detect_anomalies


def _field(
    bid: str,
    *,
    value: str = "",
    required: bool = False,
    minlength: object = None,
    maxlength: object = None,
    minimum: object = None,
    maximum: object = None,
    pattern: str = "",
    input_type: str = "text",
    role: str = "textbox",
    tag: str = "input",
    label: str = "Field",
) -> dict:
    candidate = {
        "bid": bid,
        "name": label,
        "label": label,
        "text": "",
        "value": value,
        "role": role,
        "tag": tag,
        "input_type": input_type,
        "type": input_type,
        "required": required,
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 100, 180, 36],
    }
    if minlength is not None:
        candidate["minlength"] = minlength
    if maxlength is not None:
        candidate["maxlength"] = maxlength
    if minimum is not None:
        candidate["min"] = minimum
    if maximum is not None:
        candidate["max"] = maximum
    if pattern:
        candidate["pattern"] = pattern
    return candidate


def _submit(text: str = "Save") -> dict:
    return {
        "bid": "submit",
        "name": text,
        "text": text,
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 160, 100, 40],
        "is_submit_related": True,
    }


def _observation(candidates: list[dict], *, text: str = "", url: str = "http://app.test/form") -> dict:
    page_text = text if text else " ".join(str(item.get("name") or item.get("text") or "") for item in candidates)
    return {
        "page_state": {
            "url": url,
            "title": "Form",
            "page_text": page_text,
            "page_text_sample": page_text,
            "page_text_length": len(page_text),
            "dom_node_count": len(candidates),
            "viewport_width": 800,
            "viewport_height": 600,
            "cart_count": 0,
        },
        "runtime_signals": {},
        "candidate_elements": candidates,
        "layout_signals": {},
    }


def _r08_anomalies(before: dict, after: dict, *, action_type: str = "click_element", **info) -> list[dict]:
    anomalies = detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": action_type, "candidate_index": 1},
            **info,
        },
    )
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R08"]


def test_r08_required_empty_submit_success_emits():
    before = _observation([_field("title", required=True, value=""), _submit()])
    after = _observation([_submit()], text="Saved")

    assert _r08_anomalies(before, after, success=True)


def test_r08_required_empty_with_validation_message_does_not_emit():
    before = _observation([_field("title", required=True, value=""), _submit()])
    after = _observation([_field("title", required=True, value=""), _submit()], text="required field")

    assert not _r08_anomalies(before, after, success=False)


def test_r08_minlength_violation_saved_success_emits():
    before = _observation([_field("username", value="abc", minlength=8), _submit()])
    after = _observation([_submit()], text="Saved")

    assert _r08_anomalies(before, after, success=True)[0]["type"] == "validation-bypass"


def test_r08_minlength_violation_rejected_does_not_emit():
    before = _observation([_field("username", value="abc", minlength=8), _submit()])
    after = _observation([_field("username", value="abc", minlength=8), _submit()], text="too short")

    assert not _r08_anomalies(before, after)


def test_r08_maxlength_violation_success_emits():
    before = _observation([_field("code", value="abcdef", maxlength=3), _submit()])
    after = _observation([_submit()], text="Saved")

    assert _r08_anomalies(before, after, success=True)


def test_r08_number_min_violation_success_emits():
    before = _observation([_field("qty", value="0", minimum=1, input_type="number"), _submit()])
    after = _observation([_submit()], text="Saved")

    assert _r08_anomalies(before, after, success=True)


def test_r08_valid_input_does_not_emit():
    before = _observation([_field("username", value="abcdefgh", minlength=8), _submit()])
    after = _observation([_submit()], text="Saved")

    assert not _r08_anomalies(before, after, success=True)


def test_r08_fill_input_only_does_not_emit():
    before = _observation([_field("username", value="abc", minlength=8)])
    after = _observation([_field("username", value="abc", minlength=8)])

    assert not _r08_anomalies(before, after, action_type="fill_input", success=True)


def test_r08_weak_password_without_policy_does_not_emit():
    before = _observation([_field("password", value="1234", input_type="password", label="Password"), _submit("Register")])
    after = _observation([_submit("Register")], text="Registered")

    assert not [item for item in _r08_anomalies(before, after, success=True) if item["type"] == "weak-password-validation"]


def test_r08_password_minlength_bypass_emits_without_plaintext_value():
    before = _observation(
        [_field("password", value="1234", input_type="password", minlength=8, label="Password"), _submit("Register")]
    )
    after = _observation([_submit("Register")], text="Registered")

    anomaly = _r08_anomalies(before, after, success=True)[0]
    assert anomaly["type"] == "weak-password-validation"
    assert "value" not in anomaly["evidence"]
    assert anomaly["evidence"]["value_length"] == 4


def test_r08_empty_forum_post_rejected_does_not_emit():
    before = _observation([_field("post", value="", required=True, tag="textarea", label="Post content"), _submit("Post")])
    after = _observation([_field("post", value="", required=True, tag="textarea", label="Post content"), _submit("Post")], text="required")

    assert not [item for item in _r08_anomalies(before, after) if item["type"] == "forum-empty-post-validation-missing"]


def test_r08_empty_forum_post_created_emits():
    before = _observation([_field("post", value="   ", required=True, tag="textarea", label="Post content"), _submit("Post")])
    after = _observation([_submit("Post")], text="Created")
    after["page_state"]["empty_post_created"] = True

    anomaly = _r08_anomalies(before, after, success=True)[0]
    assert anomaly["type"] == "forum-empty-post-validation-missing"


def test_r08_invalid_input_runtime_exception_without_success_does_not_emit():
    before = _observation([_field("username", value="abc", minlength=8), _submit()])
    after = _observation([_field("username", value="abc", minlength=8), _submit()])
    after["runtime_signals"]["page_errors"] = ["Uncaught TypeError"]

    assert not _r08_anomalies(before, after)


def test_r08_invalid_submit_no_response_does_not_emit():
    before = _observation([_field("username", value="abc", minlength=8), _submit()])
    after = _observation([_field("username", value="abc", minlength=8), _submit()])

    assert not _r08_anomalies(before, after)
