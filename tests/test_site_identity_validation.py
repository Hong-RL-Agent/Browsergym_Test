from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.site_profile_service import validate_site_identity


class SiteIdentityValidationTests(unittest.TestCase):
    def test_title_hint_detects_wrong_site_mapping(self) -> None:
        result = validate_site_identity(
            "site5174",
            {"page_state": {"title": "Site001 Mini Shop", "data_bug_ids_found": []}},
        )

        self.assertFalse(result["site_identity_match"])
        self.assertEqual("site001", result["title_site_id_hint"])
        self.assertTrue(any("page title" in warning for warning in result["identity_warnings"]))


if __name__ == "__main__":
    unittest.main()
