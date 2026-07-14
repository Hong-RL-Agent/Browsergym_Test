from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.multisite_training_service import _attach_policy_execution_fields


class PolicyVsExecutedActionLoggingTests(unittest.TestCase):
    def test_action_log_records_policy_and_executed_action(self) -> None:
        action = {
            "action_type": "inspect_dom",
            "action_signature": "inspect_dom::global",
            "action_element_key": "",
        }
        policy_action = {
            "action_type": "click_element",
            "action_signature": "click_element::button|add",
            "action_element_key": "button|add",
        }

        _attach_policy_execution_fields(
            action,
            policy_action_id=5,
            policy_action=policy_action,
            executed_action_id=1,
            fallback_applied=True,
            fallback_reason="exploration_redirect",
        )

        self.assertEqual(5, action["policy_action_index"])
        self.assertEqual("click_element", action["policy_action_type"])
        self.assertEqual("click_element::button|add", action["policy_action_signature"])
        self.assertEqual("button|add", action["policy_action_element_key"])
        self.assertEqual(1, action["executed_action_index"])
        self.assertEqual("inspect_dom", action["executed_action_type"])
        self.assertEqual("inspect_dom::global", action["executed_action_signature"])
        self.assertTrue(action["fallback_applied"])

    def test_fallback_applied_records_reason(self) -> None:
        action = {"action_type": "inspect_network", "action_signature": "inspect_network::global"}
        _attach_policy_execution_fields(
            action,
            policy_action_id=2,
            policy_action={"action_type": "open_detail_panel", "action_signature": "open_detail_panel::global"},
            executed_action_id=3,
            fallback_applied=True,
            fallback_reason="policy_selected_meta_action",
        )
        self.assertEqual("policy_selected_meta_action", action["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
