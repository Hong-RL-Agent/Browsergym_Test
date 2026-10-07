from services.interactive_layout_evidence_service import (
    collect_interactive_layout_evidence,
    enrich_observation_with_interactive_layout_evidence,
)
import pytest


class _FakePage:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def evaluate(self, script, payload):
        self.calls.append((script, payload))
        return self.result


def _target():
    return {
        "bid": "cta",
        "role": "button",
        "tag": "button",
        "text": "Continue",
        "bbox": [350, 100, 150, 40],
        "clickable": True,
    }


def test_unavailable_page_fails_closed_without_crashing():
    result = collect_interactive_layout_evidence(None, _target())

    assert result["evidence_status"] == "UNAVAILABLE"
    assert result["target_resolved"] is False


def test_live_evidence_is_attached_to_detector_snapshot():
    page = _FakePage({
        "evidence_status": "AVAILABLE",
        "target_resolved": True,
        "overflow_x": "hidden",
        "scroll_width": 600,
        "client_width": 400,
        "reachable_via_scroll_x": False,
        "hit_test_valid": False,
        "hit_test_blocked": True,
        "keyboard_reachable": True,
        "mutation_guard": {
            "url_unchanged": True,
            "scroll_unchanged": True,
            "active_element_unchanged": True,
        },
    })
    observation = {"candidate_elements": [_target()], "runtime_signals": {}}

    enriched, summary = enrich_observation_with_interactive_layout_evidence(page, observation)

    assert summary == {"status": "AVAILABLE", "count": 1, "available": 1, "unavailable": 0}
    evidence = enriched["candidate_elements"][0]["interactive_layout_evidence"]
    assert evidence["overflow_x"] == "hidden"
    assert evidence["hit_test_blocked"] is True
    assert evidence["keyboard_reachable"] is True
    assert evidence["mutation_guard"]["scroll_unchanged"] is True
    assert observation["candidate_elements"][0].get("interactive_layout_evidence") is None
    assert len(page.calls) == 1


def test_collector_evaluate_payload_contains_no_mutating_commands():
    page = _FakePage({"evidence_status": "UNAVAILABLE", "target_resolved": False})

    collect_interactive_layout_evidence(page, _target())
    script = page.calls[0][0]

    assert "elementFromPoint" in script
    assert ".click(" not in script
    assert "scrollTo(" not in script
    assert "focus(" not in script
    assert "set_viewport" not in script


def test_observation_enrichment_handles_missing_candidates():
    page = _FakePage({"evidence_status": "AVAILABLE"})

    enriched, summary = enrich_observation_with_interactive_layout_evidence(page, {"candidate_elements": None})

    assert enriched["candidate_elements"] is None
    assert summary["status"] == "UNAVAILABLE"


def test_real_playwright_probe_is_read_only_when_browser_is_available():
    playwright = pytest.importorskip("playwright.sync_api")
    try:
        runner = playwright.sync_playwright().start()
    except Exception as exc:
        pytest.skip(f"Playwright runtime unavailable: {exc}")
    try:
        try:
            browser = runner.chromium.launch(headless=True)
        except Exception as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.set_content(
            '<div id="box" style="width:190px;height:28px;overflow:hidden">'
            '<span id="txt" style="display:inline-block;width:360px">Clipped text</span></div>'
        )
        before = page.evaluate("() => ({url: location.href, x: scrollX, y: scrollY, active: document.activeElement && document.activeElement.id})")
        evidence = collect_interactive_layout_evidence(page, {
            "element_id": "txt", "role": "generic", "tag": "span", "text": "Clipped text",
            "bbox": [0, 0, 360, 28],
        })
        after = page.evaluate("() => ({url: location.href, x: scrollX, y: scrollY, active: document.activeElement && document.activeElement.id})")
        browser.close()
        assert evidence["evidence_status"] == "AVAILABLE"
        assert evidence["overflow_x"] == "hidden"
        assert evidence["scroll_width"] > evidence["client_width"]
        assert evidence["container_overflow"] is True
        assert evidence["mutation_guard"]["scroll_unchanged"] is True
        assert before == after
    finally:
        runner.stop()
