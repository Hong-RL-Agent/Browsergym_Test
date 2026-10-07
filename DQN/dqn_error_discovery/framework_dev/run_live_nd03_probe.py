"""Evaluation-only live N-D03 probe; never imported by runtime paths."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from services.anomaly_detection_service import detect_anomalies
from services.interactive_layout_evidence_service import collect_interactive_layout_evidence


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.set_content(
            '<div id="box" style="width:190px;height:28px;overflow:hidden">'
            '<span id="txt" style="display:inline-block;width:360px">'
            "Important content cannot be read completely and is blocked"
            "</span></div>"
        )
        before = page.evaluate("() => ({url: location.href, x: scrollX, y: scrollY, active: document.activeElement && document.activeElement.id})")
        evidence = collect_interactive_layout_evidence(page, {
            "element_id": "txt", "role": "generic", "tag": "span",
            "text": "Important content cannot be read completely and is blocked",
            "bbox": [0, 0, 360, 28],
        })
        after = page.evaluate("() => ({url: location.href, x: scrollX, y: scrollY, active: document.activeElement && document.activeElement.id})")
        candidate = {
            "bid": "dev-next-nd03-content", "role": "generic", "tag": "span",
            "text": evidence.get("text_content"), "visible": True, "visibility": 1.0,
            "enabled": True, "clickable": False, "bbox": evidence.get("target_bbox"),
            "interactive_layout_evidence": evidence,
        }
        observation = {"page_state": {"viewport_width": 390, "viewport_height": 844, "document_height": 844}, "runtime_signals": {}, "candidate_elements": [candidate]}
        anomalies = detect_anomalies(observation, observation, {"action": {"action_type": "inspect_layout", "candidate_index": 0}})
        r06 = next((item for item in anomalies if item.get("rule_id") == "R06"), None)
        report = {
            "evaluation_only": True,
            "evidence": evidence,
            "functional_impact": (r06 or {}).get("evidence", {}).get("functional_impact"),
            "r06": r06 is not None,
            "mutation_guard": evidence.get("mutation_guard"),
            "before_after_same": before == after,
        }
        browser.close()
    path = Path(__file__).resolve().parents[2] / "artifacts" / "framework_dev" / "nd03_live_probe.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("functional_impact", "r06", "mutation_guard", "before_after_same")}, indent=2))


if __name__ == "__main__":
    main()
