from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.validate_training_sites import DEFAULT_OUTPUT, REQUIRED_RESULT_FIELDS, build_validation_summary


class ValidateTrainingSitesTests(unittest.TestCase):
    def test_validation_summary_path_is_v3_policy_safe(self) -> None:
        self.assertEqual(
            Path("artifacts/training_logs/v3_policy_safe/site_validation_summary.json"),
            DEFAULT_OUTPUT,
        )

    def test_failed_site_does_not_prevent_summary(self) -> None:
        summary = build_validation_summary(
            [
                {
                    "site_id": "site001",
                    "base_url": "http://localhost:9220",
                    "status": "ok",
                    "reset_ok": True,
                    "step_ok": True,
                },
                {
                    "site_id": "site-bad",
                    "base_url": "http://127.0.0.1:1",
                    "status": "failed",
                    "error_message": "connection refused",
                },
            ]
        )
        self.assertEqual("v3_policy_safe", summary["version"])
        self.assertEqual(2, summary["total_sites"])
        self.assertEqual(1, summary["ok_sites"])
        self.assertEqual(1, summary["failed_sites"])
        self.assertEqual(["site001"], summary["ok_site_ids"])
        self.assertEqual(["site-bad"], summary["failed_site_ids"])

    def test_validation_result_fields_are_complete(self) -> None:
        summary = build_validation_summary([{"site_id": "site-bad", "base_url": "", "status": "failed"}])
        result = summary["results"][0]
        for field in REQUIRED_RESULT_FIELDS:
            self.assertIn(field, result)


if __name__ == "__main__":
    unittest.main()
