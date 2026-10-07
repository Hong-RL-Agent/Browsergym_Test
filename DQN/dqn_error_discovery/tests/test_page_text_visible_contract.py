from adapters.browsergym_observation_adapter import _page_text
from services.anomaly_detection_service import _r10_ui_settled


def _ax(*texts: str) -> dict:
    return {"nodes": [{"role": "StaticText", "name": text} for text in texts]}


def test_compiled_source_is_not_page_text_when_structured_ax_tree_exists() -> None:
    observation = {
        "text": "const [loading, setLoading] = useState(true);",
        "axtree_object": _ax("Points: 100", "Magic Buy"),
    }

    page_text = _page_text(observation)

    assert "Points: 100" in page_text
    assert "Magic Buy" in page_text
    assert "useState" not in page_text
    assert "loading" not in page_text


def test_source_only_raw_text_is_not_used_as_fallback() -> None:
    page_text = _page_text({"text": "const [loading, setLoading] = useState(true);"})

    assert page_text == ""


def test_plain_raw_text_remains_compatible_without_structured_data() -> None:
    assert _page_text({"text": "Points: 100"}) == "Points: 100"


def test_visible_loading_label_is_preserved() -> None:
    observation = {"axtree_object": _ax("Loading products...")}

    page_text = _page_text(observation)

    assert "Loading products..." in page_text
    assert _r10_ui_settled({"page_state": {"page_text": page_text}}) is False


def test_script_and_style_nodes_are_excluded_from_dom_text() -> None:
    observation = {
        "dom_object": {
            "nodeName": "body",
            "children": [
                {"nodeName": "script", "textContent": "const loading = true;"},
                {"nodeName": "style", "textContent": ".loading { display:block; }"},
                {"nodeName": "div", "innerText": "Point: 100"},
            ],
        }
    }

    page_text = _page_text(observation)

    assert page_text == "Point: 100"
    assert "loading" not in page_text.lower()


def test_visible_application_text_is_preserved() -> None:
    observation = {"axtree_object": _ax("Magic Buy", "Points", "1000", "Earn 0.1")}

    page_text = _page_text(observation)

    for text in ("Magic Buy", "Points", "1000", "Earn 0.1"):
        assert text in page_text
