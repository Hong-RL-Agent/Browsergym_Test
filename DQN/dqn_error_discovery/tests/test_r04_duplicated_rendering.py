from copy import deepcopy

from services.anomaly_detection_service import detect_anomalies


def _candidate(
    text: str,
    *,
    bid: str,
    role: str = "link",
    tag: str = "a",
    href: str = "",
    product_id: str = "",
    post_id: str = "",
    comment_id: str = "",
) -> dict:
    candidate = {
        "bid": bid,
        "name": text,
        "text": text,
        "role": role,
        "tag": tag,
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
    }
    if href:
        candidate["href"] = href
    if product_id:
        candidate["product_id"] = product_id
    if post_id:
        candidate["post_id"] = post_id
    if comment_id:
        candidate["comment_id"] = comment_id
    return candidate


def _observation(candidates: list[dict]) -> dict:
    return {
        "page_state": {
            "url": "http://app.test/page",
            "title": "Page",
            "page_text": " ".join(item.get("text", "") for item in candidates),
            "page_text_sample": " ".join(item.get("text", "") for item in candidates),
            "page_text_length": sum(len(item.get("text", "")) for item in candidates),
            "dom_node_count": len(candidates),
            "cart_count": 0,
        },
        "runtime_signals": {},
        "candidate_elements": candidates,
    }


def _action(**extra: object) -> dict:
    result = {"action": {"action_type": "click_element", "candidate_index": 0, "clicked_bid": "submit-1"}}
    result.update(extra)
    return result


def _types(before: dict, after: dict, action_info: dict | None = None) -> list[str]:
    return [item["type"] for item in detect_anomalies(before, after, action_info or _action())]


def test_r04_same_entity_count_increases_from_one_to_two():
    before = _observation([_candidate("Product 10", bid="p10-a", product_id="10")])
    after = _observation(
        [
            _candidate("Product 10", bid="p10-a", product_id="10"),
            _candidate("Product 10", bid="p10-b", product_id="10"),
        ]
    )

    assert "duplicated-rendering" in _types(before, after)


def test_r04_different_entities_do_not_emit():
    before = _observation([_candidate("Product 10", bid="p10", product_id="10")])
    after = _observation(
        [
            _candidate("Product 10", bid="p10", product_id="10"),
            _candidate("Product 11", bid="p11", product_id="11"),
            _candidate("Product 12", bid="p12", product_id="12"),
        ]
    )

    assert "duplicated-rendering" not in _types(before, after)


def test_r04_same_text_different_entities_do_not_emit():
    before = _observation([_candidate("상품", bid="p10", product_id="10")])
    after = _observation(
        [
            _candidate("상품", bid="p10", product_id="10"),
            _candidate("상품", bid="p11", product_id="11"),
        ]
    )

    assert "duplicated-rendering" not in _types(before, after)


def test_r04_same_entity_count_unchanged_does_not_emit():
    before = _observation([_candidate("Product 10", bid="p10", product_id="10")])
    after = deepcopy(before)

    assert "duplicated-rendering" not in _types(before, after)


def test_r04_normal_list_increase_with_new_entity_does_not_emit():
    before = _observation(
        [
            _candidate("Post 1", bid="post-1", post_id="1"),
            _candidate("Post 2", bid="post-2", post_id="2"),
        ]
    )
    after = _observation(
        [
            _candidate("Post 1", bid="post-1", post_id="1"),
            _candidate("Post 2", bid="post-2", post_id="2"),
            _candidate("Post 3", bid="post-3", post_id="3"),
        ]
    )

    assert "duplicated-rendering" not in _types(before, after)


def test_r04_comment_entity_duplicate_emits():
    before = _observation([_candidate("Looks good", bid="c100-a", comment_id="100")])
    after = _observation(
        [
            _candidate("Looks good", bid="c100-a", comment_id="100"),
            _candidate("Looks good", bid="c100-b", comment_id="100"),
        ]
    )

    assert "duplicated-rendering" in _types(before, after)


def test_r04_same_text_header_and_main_without_identity_does_not_emit():
    before = _observation([_candidate("Home", bid="nav-home", role="link", tag="a")])
    after = _observation(
        [
            _candidate("Home", bid="nav-home", role="link", tag="a"),
            _candidate("Home", bid="main-home", role="heading", tag="h1"),
        ]
    )

    assert "duplicated-rendering" not in _types(before, after)


def test_r04_action_error_does_not_emit_new_duplicate():
    before = _observation([_candidate("Product 10", bid="p10-a", product_id="10")])
    after = _observation(
        [
            _candidate("Product 10", bid="p10-a", product_id="10"),
            _candidate("Product 10", bid="p10-b", product_id="10"),
        ]
    )

    assert "duplicated-rendering" not in _types(before, after, _action(last_action_error=True))


def test_r04_fallback_identity_uses_role_text_and_href():
    before = _observation([_candidate("Product 10", bid="p10-a", href="/products/10")])
    after = _observation(
        [
            _candidate("Product 10", bid="p10-a", href="/products/10"),
            _candidate("Product 10", bid="p10-b", href="/products/10"),
        ]
    )

    assert "duplicated-rendering" in _types(before, after)


def test_r04_text_only_identity_is_not_enough():
    before = _observation([_candidate("Product", bid="p-a")])
    after = _observation(
        [
            _candidate("Product", bid="p-a"),
            _candidate("Product", bid="p-b"),
        ]
    )

    assert "duplicated-rendering" not in _types(before, after)
