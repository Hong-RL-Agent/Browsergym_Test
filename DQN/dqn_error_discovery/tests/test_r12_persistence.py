from services.anomaly_detection_service import detect_anomalies


def _button(text: str = "Save") -> dict:
    return {
        "bid": "save",
        "name": text,
        "text": text,
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 100, 100, 40],
    }


def _observation(
    entities: list[dict] | None = None,
    *,
    followup_entities: list[dict] | None = None,
    loading: bool = False,
) -> dict:
    entities = entities or []
    text = "Loading" if loading else " ".join(str(value) for entity in entities for value in entity.values())
    page_state = {
        "url": "http://app.test/entity",
        "title": "Entity",
        "page_text": text,
        "page_text_sample": text,
        "page_text_length": len(text),
        "dom_node_count": len(entities) + 1,
        "viewport_width": 800,
        "viewport_height": 600,
        "ui_entities": entities,
        "cart_count": 0,
        "loading": loading,
    }
    if followup_entities is not None:
        page_state["refetched_entities"] = followup_entities
    return {
        "page_state": page_state,
        "runtime_signals": {"loading": loading},
        "candidate_elements": [_button()],
        "layout_signals": {},
    }


def _r12(before: dict, after: dict, **info) -> list[dict]:
    anomalies = detect_anomalies(
        before,
        after,
        {
            "action": {"action_type": "click_element", "candidate_index": 0},
            **info,
        },
    )
    return [item for item in anomalies if item.get("evidence", {}).get("rule_id") == "R12"]


def test_r12_update_normal_persistence_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "name": "Lee"}])

    assert not _r12(before, after, method="PATCH", response_status=200, submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_update_saved_value_lost_emits():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "name": "Kim"}])

    anomaly = _r12(before, after, method="PATCH", response_status=200, submitted_entity={"user_id": "1", "name": "Lee"})[0]
    assert anomaly["type"] == "persistence-mismatch"
    assert anomaly["evidence"]["issue_type"] == "update-persistence-mismatch"


def test_r12_create_normal_persistence_does_not_emit():
    before = _observation()
    after = _observation(followup_entities=[{"post_id": "100", "title": "Test"}])

    assert not _r12(before, after, method="POST", response_status=201, created_entity={"post_id": "100", "title": "Test"})


def test_r12_create_success_but_missing_in_followup_emits():
    before = _observation()
    after = _observation(followup_entities=[{"post_id": "101", "title": "Other"}])

    anomaly = _r12(before, after, method="POST", response_status=201, created_entity={"post_id": "100", "title": "Test"})[0]
    assert anomaly["evidence"]["issue_type"] == "create-persistence-missing"


def test_r12_delete_normal_persistence_does_not_emit():
    before = _observation([{"comment_id": "50", "content": "hello"}])
    after = _observation(followup_entities=[{"comment_id": "51", "content": "other"}])

    assert not _r12(before, after, method="DELETE", response_status=204, deleted_entity={"comment_id": "50"})


def test_r12_delete_success_but_entity_still_exists_emits_forum_delete_failed():
    before = _observation([{"comment_id": "50", "content": "hello"}])
    after = _observation(followup_entities=[{"comment_id": "50", "content": "hello"}])

    anomaly = _r12(before, after, method="DELETE", response_status=204, deleted_entity={"comment_id": "50"})[0]
    assert anomaly["type"] == "forum-comment-delete-failed"


def test_r12_save_complete_no_response_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation([{"user_id": "1", "name": "Kim"}])

    assert not _r12(before, after, operation="update", submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_save_api_500_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "name": "Kim"}])

    assert not _r12(before, after, method="PATCH", response_status=500, submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_network_failure_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "name": "Kim"}])

    assert not _r12(before, after, method="PATCH", response_status=200, network_error="connection reset", submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_timeout_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "name": "Kim"}])

    assert not _r12(before, after, method="PATCH", response_status=200, timeout=True, submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_current_api_ui_mismatch_without_refetch_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation([{"user_id": "1", "name": "Kim"}])

    assert not _r12(before, after, method="PATCH", response_status=200, api_entities=[{"user_id": "1", "name": "Lee"}])


def test_r12_loading_old_value_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "name": "Kim"}], loading=True)

    assert not _r12(before, after, method="PATCH", response_status=200, submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_unrelated_entity_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "2", "name": "Lee"}])

    assert not _r12(before, after, method="PATCH", response_status=200, submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_same_entity_unrelated_field_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation(followup_entities=[{"user_id": "1", "role": "user"}])

    assert not _r12(before, after, method="PATCH", response_status=200, submitted_entity={"user_id": "1", "name": "Lee"})


def test_r12_password_change_redacts_raw_value():
    before = _observation([{"user_id": "1", "password": "oldsecret"}])
    after = _observation(followup_entities=[{"user_id": "1", "password": "oldsecret"}])

    anomaly = _r12(before, after, method="PATCH", response_status=200, submitted_entity={"user_id": "1", "password": "newsecret"})[0]
    evidence_text = str(anomaly["evidence"])
    assert "newsecret" not in evidence_text
    assert "oldsecret" not in evidence_text
    assert anomaly["evidence"]["field"] == "password"


def test_r12_optimistic_ui_without_write_success_or_refetch_does_not_emit():
    before = _observation([{"user_id": "1", "name": "Kim"}])
    after = _observation([{"user_id": "1", "name": "Lee"}])

    assert not _r12(before, after, submitted_entity={"user_id": "1", "name": "Lee"})
