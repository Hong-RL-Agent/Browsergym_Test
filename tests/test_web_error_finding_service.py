from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.web_error_finding_service import detect_web_error_findings


class WebErrorFindingServiceTests(unittest.TestCase):
    def test_api_5xx_network_entry_creates_verified_finding(self) -> None:
        after = _obs()
        after["runtime_signals"]["network_entries"] = [
            {"method": "POST", "url": "http://local/api/order", "status": 500, "status_text": "Internal Server Error"}
        ]

        findings = detect_web_error_findings(_obs(), after, {"action_type": "click_element", "action_signature": "click::order"})

        self.assertIn("api-5xx", [item["type"] for item in findings])
        self.assertTrue(all(item.get("signature") for item in findings))

    def test_database_error_text_creates_database_finding(self) -> None:
        after = _obs(text="SQLSTATE 23505 duplicate key value violates unique constraint")

        findings = detect_web_error_findings(_obs(), after, {"action_type": "submit_form"})

        self.assertIn("database-error", [item["type"] for item in findings])

    def test_security_token_text_creates_security_finding(self) -> None:
        after = _obs(text="Authorization: Bearer abcdefghijklmnopqrstuvwxyz")

        findings = detect_web_error_findings(_obs(), after, {"action_type": "inspect_dom"})

        self.assertIn("security-token-leak", [item["type"] for item in findings])

    def test_detect_anomalies_includes_web_error_findings(self) -> None:
        after = _obs()
        after["runtime_signals"]["console_error_count"] = 1
        after["runtime_signals"]["console_errors"] = ["ReferenceError: x is not defined"]

        anomalies = detect_anomalies(_obs(), after, {"action_type": "click_element"})

        self.assertIn("console-error", [item["type"] for item in anomalies])

    def test_verified_finding_reward_uses_signature_once(self) -> None:
        after = _obs()
        after["runtime_signals"]["network_entries"] = [{"method": "GET", "url": "http://local/api/items", "status": 500}]
        anomalies = detect_web_error_findings(_obs(), after, {"action_type": "click_element", "action_signature": "click::items"})

        first_reward, first_breakdown = calculate_autonomous_reward(_obs(), after, {"action_type": "click_element"}, anomalies, [], {}, {})
        seen = {item["signature"] for item in anomalies}
        second_reward, second_breakdown = calculate_autonomous_reward(
            _obs(),
            after,
            {"action_type": "click_element"},
            anomalies,
            [],
            {"seen_anomaly_keys": seen},
            {},
        )

        self.assertGreater(first_breakdown["verified_finding_reward_total"], 0.0)
        self.assertEqual(0.0, second_breakdown["verified_finding_reward_total"])
        self.assertGreater(first_reward, second_reward)

    def test_web_error_finding_does_not_use_bug_labels(self) -> None:
        clean = _obs()
        labeled = _obs()
        labeled["candidate_elements"][0].update(
            {
                "data-bug-id": "site001-bug01",
                "bug_id": "site001-bug01",
                "catalog_bug_id_matches": ["site001-bug01"],
                "known_bug_id": "site001-bug01",
            }
        )
        clean_after = _obs()
        labeled_after = _obs()
        for after in (clean_after, labeled_after):
            after["runtime_signals"]["network_entries"] = [{"status": 500, "url": "http://local/api"}]

        self.assertEqual(
            detect_web_error_findings(clean, clean_after, {"action_type": "click_element"}),
            detect_web_error_findings(labeled, labeled_after, {"action_type": "click_element"}),
        )


def _obs(url: str = "http://local", text: str = "") -> dict:
    return {
        "page_state": {
            "url": url,
            "title": "Local",
            "page_text_length": len(text),
            "page_text_sample": text,
            "dom_node_count": 5,
        },
        "candidate_elements": [
            {
                "bid": "b1",
                "element_key": "button|submit|main",
                "text": "Submit",
                "visible": True,
                "enabled": True,
                "clickable": True,
            }
        ],
        "runtime_signals": {},
        "layout_signals": {},
    }


if __name__ == "__main__":
    unittest.main()
