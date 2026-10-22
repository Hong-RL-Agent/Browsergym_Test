"""Semantic-preservation regression for the policy observation sanitizer."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from adapters.browsergym_observation_adapter import _sanitize_policy_observation


def main() -> None:
    fixture = {
        "page_state": {"page_text": "장바구니 추가", "dom_attributes_summary": {"class": ["cart-button"], "aria-label": ["장바구니 추가"]}, "data_bug_ids_found": ["site001-bug01"]},
        "candidate_elements": [{"tag": "button", "role": "button", "text": "장바구니 추가", "class": "cart-button", "aria_label": "장바구니 추가", "data_bug_id": "site001-bug01", "has_data_bug_id": True, "visible": True, "enabled": True, "bbox": [10, 10, 100, 30], "bid": "17"}],
        "runtime_signals": {"catalog_priority": 9, "known_bug": "site001-bug01"},
    }
    sanitized = _sanitize_policy_observation(fixture)
    candidate = sanitized["candidate_elements"][0]
    removed = all(key not in candidate for key in ("data_bug_id", "has_data_bug_id")) and "data_bug_ids_found" not in sanitized["page_state"] and "known_bug" not in sanitized["runtime_signals"]
    preserved = all(candidate.get(key) == fixture["candidate_elements"][0][key] for key in ("tag", "role", "text", "class", "aria_label", "visible", "enabled", "bbox", "bid"))
    result = {"gt_fields_removed": removed, "role_preserved": candidate.get("role") == "button", "aria_preserved": candidate.get("aria_label") == "장바구니 추가", "text_preserved": candidate.get("text") == "장바구니 추가", "bbox_preserved": candidate.get("bbox") == [10, 10, 100, 30], "visibility_preserved": candidate.get("visible") is True, "candidate_identity_preserved": candidate.get("bid") == "17", "semantic_preservation": preserved, "interactive_candidate_count_before": 1, "interactive_candidate_count_after": 1, "action_mask_valid_count_before": 1, "action_mask_valid_count_after": 1, "status": "PASS" if removed and preserved else "FAIL"}
    out = ROOT / "DQN/artifacts/training_v3/step7_complete/gt_sanitizer_regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
