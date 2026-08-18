from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.action_opportunity_service import build_action_opportunities
from services.anomaly_detection_service import detect_anomalies
from runners.browsergym_api_server import _is_reportable_anomaly


COMMUNITY = "\ucee4\ubba4\ub2c8\ud2f0"
POST = "\uac8c\uc2dc\uae00"
WRITE = "\uae00\uc4f0\uae30"
COMMENT = "\ub313\uae00"
TITLE = "\uc81c\ubaa9"
CONTENT = "\ub0b4\uc6a9"
SAVE = "\uc800\uc7a5"
REGISTER = "\ub4f1\ub85d"
DELETE = "\uc0ad\uc81c"
MORE = "\ub354\ubcf4\uae30"
POST_TITLE = "\ucee4\ubba4\ub2c8\ud2f0 \uc815\uc2dd \uc624\ud508 \uc548\ub0b4\uc640 \ucd08\uae30 \uc774\uc6a9 \uac00\uc774\ub4dc"
SEARCH_PLACEHOLDER = "\uc81c\ubaa9, \uc791\uc131\uc790, \uce74\ud14c\uace0\ub9ac\ub85c \uac80\uc0c9"


def _candidate(**overrides):
    base = {
        "bid": overrides.get("bid", "c0"),
        "element_key": overrides.get("element_key", "button|forum|main"),
        "tag": "button",
        "role": "button",
        "type": "button",
        "name": WRITE,
        "text": WRITE,
        "href": "",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "fillable": False,
        "semantic_action_type": "",
    }
    base.update(overrides)
    return base


def _obs(text: str, candidates=None):
    return {
        "candidate_elements": list(candidates or []),
        "page_state": {
            "url": "http://localhost:5189/",
            "title": f"ForumWorks {COMMUNITY}",
            "page_text_length": len(text),
            "page_text_sample": text,
            "page_text": text,
            "dom_node_count": 30,
            "has_form": True,
        },
        "runtime_signals": {},
        "layout_signals": {},
    }


class ForumSite003DetectionTests(unittest.TestCase):
    def test_forum_opportunities_prioritize_real_workflow_not_more_button(self) -> None:
        obs = _obs(
            f"ForumWorks {COMMUNITY} {POST} {COMMENT} {TITLE} {CONTENT} {SAVE} {DELETE}",
            [
                _candidate(element_key="button|more|1", name=MORE, text=MORE),
                _candidate(element_key="button|write|1", name=WRITE, text=WRITE),
                _candidate(element_key="input|title|1", tag="input", role="textbox", type="text", name=f"{POST} {TITLE}\uc744 \uc785\ub825\ud558\uc138\uc694", text=f"{POST} {TITLE}\uc744 \uc785\ub825\ud558\uc138\uc694", fillable=True, clickable=False),
                _candidate(element_key="textarea|content|1", tag="textarea", role="textbox", type="text", name=f"{CONTENT}\uc744 \uc785\ub825\ud558\uc138\uc694", text=f"{CONTENT}\uc744 \uc785\ub825\ud558\uc138\uc694", fillable=True, clickable=False),
                _candidate(element_key="button|save|1", name=SAVE, text=SAVE),
                _candidate(element_key="button|delete|1", name=DELETE, text=DELETE),
                _candidate(element_key="button|post|1", name=POST_TITLE, text=POST_TITLE),
            ],
        )

        state = build_action_opportunities(obs)
        required_targets = {item.get("target_name") or item.get("target_text") for item in state["opportunities"] if item.get("required")}

        self.assertIn(WRITE, required_targets)
        self.assertIn(SAVE, required_targets)
        self.assertIn(DELETE, required_targets)
        self.assertIn(POST_TITLE, required_targets)
        self.assertNotIn(MORE, required_targets)

    def test_more_button_no_response_is_suppressed_on_forum_surface(self) -> None:
        candidate = _candidate(element_key="button|more|1", name=MORE, text=MORE)
        anomalies = detect_anomalies(
            _obs(f"ForumWorks {COMMUNITY} {POST} {COMMENT}", [candidate]),
            _obs(f"ForumWorks {COMMUNITY} {POST} {COMMENT}", [candidate]),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
        )

        self.assertNotIn("button-no-response", [item["type"] for item in anomalies])

    def test_post_title_click_without_detail_opens_domain_specific_finding(self) -> None:
        candidate = _candidate(name=POST_TITLE, text=POST_TITLE)

        anomalies = detect_anomalies(
            _obs(f"ForumWorks {COMMUNITY} {POST_TITLE}", [candidate]),
            _obs(f"ForumWorks {COMMUNITY} {POST_TITLE}", [candidate]),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
        )

        self.assertIn("forum-post-detail-not-opened", [item["type"] for item in anomalies])

    def test_duplicate_comment_after_submit_has_comment_count_evidence(self) -> None:
        candidate = _candidate(name=f"{COMMENT} {REGISTER}", text=f"{COMMENT} {REGISTER}")
        value = "\uc911\ubcf5\ub313\uae00"

        anomalies = detect_anomalies(
            _obs(f"ForumWorks {COMMENT}", [candidate]),
            _obs(f"ForumWorks {COMMENT} {value} {value}", [candidate]),
            {"action": {"action_type": "click_element", "candidate_index": 0, "input_value": value}},
        )

        finding = next(item for item in anomalies if item["type"] == "forum-comment-duplicated")
        self.assertEqual(0, finding["evidence"]["comment_count_before"])
        self.assertEqual(2, finding["evidence"]["comment_count_after"])

    def test_empty_post_submit_with_new_post_count_is_validation_missing(self) -> None:
        title = _candidate(element_key="input|title|1", tag="input", role="textbox", type="text", name=f"{POST} {TITLE}\uc744 \uc785\ub825\ud558\uc138\uc694", text=f"{POST} {TITLE}\uc744 \uc785\ub825\ud558\uc138\uc694", fillable=True, clickable=False)
        content = _candidate(element_key="textarea|content|1", tag="textarea", role="textbox", type="text", name=f"{CONTENT}\uc744 \uc785\ub825\ud558\uc138\uc694", text=f"{CONTENT}\uc744 \uc785\ub825\ud558\uc138\uc694", fillable=True, clickable=False)
        save = _candidate(element_key="button|save|1", name=SAVE, text=SAVE)
        before = _obs(f"ForumWorks \uc6b4\uc601 \ud1a0\ub860 \uacf5\uc720", [title, content, save])
        after = _obs(f"ForumWorks \uc6b4\uc601 \ud1a0\ub860 \uacf5\uc720 {POST}", [title, content, save])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 2}})

        self.assertIn("forum-empty-post-validation-missing", [item["type"] for item in anomalies])

    def test_save_without_success_message_has_domain_specific_feedback_finding(self) -> None:
        save = _candidate(name=SAVE, text=SAVE)
        before = _obs(f"ForumWorks {POST} {TITLE} {CONTENT}", [save])
        after = _obs(f"ForumWorks {POST} {TITLE} {CONTENT}", [save])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertIn("forum-save-feedback-missing", [item["type"] for item in anomalies])

    def test_delete_leaves_comment_has_domain_specific_finding(self) -> None:
        body = "\uc0ad\uc81c\ub418\uc5b4\uc57c \ud560 \ub313\uae00"
        delete = _candidate(name=f"{COMMENT} {DELETE}", text=f"{COMMENT} {DELETE}", comment_text=body)
        before = _obs(f"ForumWorks {COMMENT} {body}", [delete])
        after = _obs(f"ForumWorks {COMMENT} {body}", [delete])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        finding = next(item for item in anomalies if item["type"] == "forum-comment-delete-failed")
        self.assertTrue(finding["evidence"]["comment_body_evidence"])
        self.assertEqual(1, finding["evidence"]["comment_text_count_before"])
        self.assertEqual(1, finding["evidence"]["comment_text_count_after"])
        self.assertTrue(_is_reportable_anomaly(finding))

    def test_delete_button_without_comment_body_is_not_reportable(self) -> None:
        delete = _candidate(name=f"{COMMENT} {DELETE}", text=f"{COMMENT} {DELETE}")
        before = _obs(f"ForumWorks {COMMENT} {DELETE}", [delete])
        after = _obs(f"ForumWorks {COMMENT} {DELETE}", [delete])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        finding = next(item for item in anomalies if item["type"] == "forum-comment-delete-failed")
        self.assertFalse(finding["evidence"]["comment_body_evidence"])
        self.assertFalse(_is_reportable_anomaly(finding))

    def test_comment_textbox_never_counts_as_post_detail_target(self) -> None:
        textbox = _candidate(
            element_key="textbox|comment|1",
            tag="textarea",
            role="textbox",
            type="text",
            name=f"{COMMENT}\uc744 \uc785\ub825\ud558\uc138\uc694",
            text=f"{COMMENT}\uc744 \uc785\ub825\ud558\uc138\uc694",
            fillable=True,
            clickable=True,
        )

        anomalies = detect_anomalies(
            _obs(f"ForumWorks {COMMENT}", [textbox]),
            _obs(f"ForumWorks {COMMENT}", [textbox]),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
        )

        self.assertNotIn("forum-post-detail-not-opened", [item["type"] for item in anomalies])

    def test_api_forbidden_requires_http_403_evidence_to_be_reportable(self) -> None:
        self.assertFalse(_is_reportable_anomaly({"type": "api-forbidden", "confidence": 0.8, "evidence": {"page_text_contains_forbidden": True}}))
        self.assertTrue(_is_reportable_anomaly({"type": "api-forbidden", "confidence": 0.8, "evidence": {"network_status": 403, "request_url": "/api/comments"}}))


if __name__ == "__main__":
    unittest.main()
