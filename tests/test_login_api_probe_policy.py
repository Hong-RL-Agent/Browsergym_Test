from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from services.autonomous_reward_service import calculate_autonomous_reward
from services.web_error_finding_service import detect_web_error_findings


class LoginApiProbePolicyTests(unittest.TestCase):
    def test_login_policy_action_mask_without_credentials(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        mask = action_space.build_action_mask(_obs(runtime={"login_credentials_available": False}))

        for action_type in ("login_as_user", "login_as_admin", "logout", "check_login_state"):
            self.assertEqual(0.0, mask[action_space.encode(action_type, 0)])

    def test_login_policy_action_mask_with_credentials(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        mask = action_space.build_action_mask(_obs(runtime={"login_credentials_available": True, "logged_in": False}))

        self.assertEqual(1.0, mask[action_space.encode("login_as_user", 0)])
        self.assertEqual(1.0, mask[action_space.encode("login_as_admin", 0)])
        self.assertEqual(1.0, mask[action_space.encode("check_login_state", 0)])
        self.assertEqual(0.0, mask[action_space.encode("logout", 0)])

    def test_logged_in_policy_action_mask_allows_logout(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        mask = action_space.build_action_mask(_obs(runtime={"login_credentials_available": True, "logged_in": True}))

        self.assertEqual(0.0, mask[action_space.encode("login_as_user", 0)])
        self.assertEqual(1.0, mask[action_space.encode("logout", 0)])

    def test_api_probe_action_mask_without_endpoint(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        mask = action_space.build_action_mask(_obs(runtime={}))

        for action_type in (
            "inspect_last_api_response",
            "replay_last_api_request",
            "call_collected_endpoint",
            "check_api_ui_match",
        ):
            self.assertEqual(0.0, mask[action_space.encode(action_type, 0)])

    def test_api_probe_action_mask_with_collected_endpoint(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        mask = action_space.build_action_mask(
            _obs(runtime={"network_entries": [{"url": "http://local/api/items", "method": "GET", "status": 200}]})
        )

        for action_type in (
            "inspect_last_api_response",
            "replay_last_api_request",
            "call_collected_endpoint",
            "check_api_ui_match",
        ):
            self.assertEqual(1.0, mask[action_space.encode(action_type, 0)])

    def test_api_probe_detects_api_ui_mismatch(self) -> None:
        after = _obs(
            runtime={
                "api_ui_match_checked": True,
                "expect_ui_change_after_api": True,
                "ui_changed_after_api": False,
                "api_probe_status": 200,
                "api_probe_url": "http://local/api/cart",
                "api_probe_method": "GET",
            }
        )

        findings = detect_web_error_findings(_obs(), after, {"action_type": "check_api_ui_match"})

        self.assertIn("api-ui-mismatch", [item["type"] for item in findings])

    def test_api_probe_does_not_treat_normal_403_as_error(self) -> None:
        after = _obs(runtime={"network_entries": [{"url": "http://local/api/admin", "method": "GET", "status": 403}]})

        findings = detect_web_error_findings(_obs(), after, {"action_type": "inspect_last_api_response"})

        self.assertNotIn("auth-permission-anomaly", [item["type"] for item in findings])
        self.assertNotIn("api-4xx", [item["type"] for item in findings])

    def test_auth_security_detects_unauthorized_success(self) -> None:
        after = _obs(runtime={"network_entries": [{"url": "http://local/api/admin", "method": "GET", "status": 200}]})

        findings = detect_web_error_findings(
            _obs(),
            after,
            {"action_type": "call_collected_endpoint", "request_without_auth": True},
        )

        self.assertIn("auth-unauthorized-access", [item["type"] for item in findings])

    def test_extended_policy_preserves_known_bug_reward_zero(self) -> None:
        after = _obs(runtime={"api_ui_match_checked": True, "expect_ui_change_after_api": True, "ui_changed_after_api": False, "api_probe_status": 200})
        anomalies = detect_web_error_findings(_obs(), after, {"action_type": "check_api_ui_match"})

        _, breakdown = calculate_autonomous_reward(
            _obs(),
            after,
            {"action_type": "check_api_ui_match"},
            anomalies,
            [{"bug_id": "site999-bug01"}],
            {},
            {"use_known_bug_reward": True},
        )

        self.assertEqual(0.0, breakdown["known_bug_reward_total"])
        self.assertEqual(0.0, breakdown["reward_known_bug_match"])

    def test_extended_policy_does_not_use_bug_id_or_catalog(self) -> None:
        clean_after = _obs(runtime={"api_probe_status": 500, "api_probe_url": "http://local/api"})
        labeled_after = _obs(runtime={"api_probe_status": 500, "api_probe_url": "http://local/api"})
        labeled_after["candidate_elements"][0].update(
            {
                "bug_id": "site001-bug01",
                "data-bug-id": "site001-bug01",
                "catalog_bug_id_matches": ["site001-bug01"],
                "site_group": "known",
                "target_signal_types": ["api"],
            }
        )

        self.assertEqual(
            detect_web_error_findings(_obs(), clean_after, {"action_type": "inspect_last_api_response"}),
            detect_web_error_findings(_obs(), labeled_after, {"action_type": "inspect_last_api_response"}),
        )


def _obs(*, runtime: dict | None = None) -> dict:
    runtime = dict(runtime or {})
    return {
        "page_state": {"url": "http://local", "page_text_length": 10, "dom_node_count": 4},
        "candidate_elements": [
            {
                "bid": "b1",
                "element_key": "button|submit|main",
                "role": "button",
                "tag": "button",
                "text": "Submit",
                "visible": True,
                "enabled": True,
                "clickable": True,
                "visibility": 1.0,
                "bbox": [0, 0, 80, 30],
            }
        ],
        "runtime_signals": runtime,
        "layout_signals": {},
    }


if __name__ == "__main__":
    unittest.main()
