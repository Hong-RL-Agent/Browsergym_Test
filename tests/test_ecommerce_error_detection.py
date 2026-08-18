from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.evaluate_multisite_browsergym_agent import _progress_policy_allows_finish
from services.action_opportunity_service import build_action_opportunities
from services.anomaly_detection_service import detect_anomalies
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs


def _candidate(**overrides):
    base = {
        "bid": "c0",
        "element_key": overrides.get("element_key", "button|cart|main"),
        "tag": "button",
        "role": "button",
        "type": "button",
        "name": "Add to cart",
        "text": "Add to cart",
        "href": "",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "fillable": False,
        "semantic_action_type": "cart",
        "is_cart_related": True,
        "is_purchase_action": True,
    }
    base.update(overrides)
    return base


def _obs(*, text: str = "", candidates=None, viewport_type: str = "desktop"):
    return {
        "candidate_elements": list(candidates or [_candidate()]),
        "page_state": {
            "url": "http://localhost:5174/",
            "title": "Mini Shop",
            "page_text_length": len(text),
            "page_text_sample": text,
            "page_text": text,
            "dom_node_count": 20,
            "viewport_type": viewport_type,
            "cart_count_detected": True,
            "cart_count": 1,
            "cart_text": text,
        },
        "runtime_signals": {
            "cart_count_detected": True,
            "cart_count": 1,
            "cart_text": text,
        },
        "layout_signals": {},
    }


class EcommerceErrorDetectionTests(unittest.TestCase):
    def test_commerce_surface_requires_mobile_viewport_opportunity(self) -> None:
        state = build_action_opportunities(_obs(text="Product cart total"))
        mobile = [item for item in state["opportunities"] if item["opportunity_type"] == "change_viewport_once"]

        self.assertEqual(1, len(mobile))
        self.assertTrue(mobile[0]["required"])
        self.assertFalse(state["summary"]["finish_allowed"])

    def test_cart_page_text_does_not_make_unrelated_button_required(self) -> None:
        obs = _obs(
            text="Product cart total",
            candidates=[
                _candidate(
                    element_key="button|sort|main",
                    name="Best sellers",
                    text="Best sellers",
                    semantic_action_type="",
                    is_cart_related=True,
                    is_purchase_action=False,
                )
            ],
        )

        state = build_action_opportunities(obs)
        click = [item for item in state["opportunities"] if item["opportunity_type"] == "click_button"]

        self.assertEqual(1, len(click))
        self.assertFalse(click[0]["required"])

    def test_repeated_dom_verification_guard_suppresses_stale_verify_opportunity(self) -> None:
        obs = _obs(text="Product cart total")
        history = {
            "step_index": 10,
            "repeated_inspect_dom_count": 3,
            "unverified_anomalies": [
                {"type": "duplicated-rendering", "confidence": 0.75, "evidence": {"duplicated_titles": {"Cart": 2}}},
            ],
        }

        state = build_action_opportunities(obs, history)

        self.assertNotIn("verify_anomaly_reproduction", {item["opportunity_type"] for item in state["opportunities"]})
        self.assertEqual(0, state["summary"]["unverified_anomaly_count"])

    def test_progress_policy_blocks_finish_until_mobile_checked_on_commerce_surface(self) -> None:
        obs = _obs(text="Product cart checkout total")
        history = {
            "opportunity_summary": {
                "generated_opportunity_count": 8,
                "remaining_opportunity_count": 1,
                "action_opportunity_coverage_rate": 0.9,
                "coverage_threshold": 0.7,
            },
            "action_type_counts": {"click_element": 3},
        }

        self.assertFalse(_progress_policy_allows_finish(obs, history))

    def test_invalid_quantity_visible_after_input_creates_cart_quantity_mismatch(self) -> None:
        before = _obs(text="Cart quantity 1 total $10", candidates=[_candidate(tag="input", role="textbox", type="number", text="Quantity", fillable=True, clickable=False, is_quantity_control=True, semantic_action_type="")])
        after = _obs(text="Cart quantity 0 total $0", candidates=before["candidate_elements"])

        anomalies = detect_anomalies(
            before,
            after,
            {"action": {"action_type": "fill_input", "candidate_index": 0, "input_value": "0"}},
        )

        self.assertIn("cart-quantity-mismatch", [item["type"] for item in anomalies])

    def test_korean_negative_quantity_visible_creates_cart_quantity_mismatch(self) -> None:
        candidate = _candidate(
            name="수량 감소",
            text="수량 감소",
            semantic_action_type="",
            is_quantity_control=True,
            is_purchase_action=False,
        )
        before = _obs(text="장바구니 수량 1 합계 10000원", candidates=[candidate])
        after = _obs(text="-1 -1 장바구니 합계 -10000원", candidates=[candidate])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertIn("cart-quantity-mismatch", [item["type"] for item in anomalies])

    def test_more_button_does_not_inherit_visible_cart_quantity_mismatch(self) -> None:
        candidate = _candidate(
            element_key="button|more|main",
            name="더보기",
            text="더보기",
            semantic_action_type="",
            is_detail_trigger=True,
            is_quantity_control=False,
            is_cart_quantity_related=False,
            is_cart_related=False,
            is_purchase_action=False,
        )
        before = _obs(text="장바구니 수량 -1", candidates=[candidate])
        after = _obs(text="장바구니 수량 -1", candidates=[candidate])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertNotIn("cart-quantity-mismatch", [item["type"] for item in anomalies])
        self.assertNotIn("broken-navigation", [item["type"] for item in anomalies])

    def test_generic_element_does_not_inherit_visible_cart_quantity_mismatch(self) -> None:
        candidate = _candidate(
            element_key="generic|element-108|main",
            tag="div",
            role="generic",
            name="element-108",
            text="element-108",
            semantic_action_type="",
            is_quantity_control=False,
            is_cart_quantity_related=False,
            is_cart_related=False,
            is_purchase_action=False,
        )
        before = _obs(text="장바구니 수량 -1", candidates=[candidate])
        after = _obs(text="장바구니 수량 -1", candidates=[candidate])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertNotIn("cart-quantity-mismatch", [item["type"] for item in anomalies])

    def test_cart_total_mismatch_from_visible_amounts(self) -> None:
        after = _obs(text="Cart item A $10 item B $20 Total $25")

        anomalies = detect_anomalies(_obs(text="Cart item A $10 item B $20 Total $30"), after, {"action_type": "inspect_dom"})

        self.assertIn("cart-total-mismatch", [item["type"] for item in anomalies])

    def test_korean_cart_total_mismatch_from_visible_won_amounts(self) -> None:
        after = _obs(text="장바구니 상품A 10,000원 상품B 20,000원 합계 25,000원")

        anomalies = detect_anomalies(_obs(text="장바구니 상품A 10,000원 상품B 20,000원 합계 30,000원"), after, {"action_type": "inspect_dom"})

        self.assertIn("cart-total-mismatch", [item["type"] for item in anomalies])

    def test_detail_click_with_no_page_change_is_broken_navigation(self) -> None:
        candidate = _candidate(
            element_key="link|product-detail|1",
            tag="a",
            role="link",
            name="Product details",
            text="Product details",
            href="/products/1",
            semantic_action_type="",
            is_detail_trigger=True,
            is_cart_related=False,
            is_purchase_action=False,
        )
        before = _obs(text="Product card", candidates=[candidate])
        after = _obs(text="Product card", candidates=[candidate])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertIn("broken-navigation", [item["type"] for item in anomalies])

    def test_korean_detail_click_with_no_page_change_is_broken_navigation(self) -> None:
        candidate = _candidate(
            element_key="link|product-detail|korean",
            tag="a",
            role="link",
            name="상품 상세 보기",
            text="상품 상세 보기",
            href="/products/1",
            semantic_action_type="",
            is_detail_trigger=True,
            is_cart_related=False,
            is_purchase_action=False,
        )
        before = _obs(text="상품 카드", candidates=[candidate])
        after = _obs(text="상품 카드", candidates=[candidate])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertIn("broken-navigation", [item["type"] for item in anomalies])

    def test_korean_commerce_candidates_are_required(self) -> None:
        obs = _obs(
            text="장바구니 합계 주문 상품 상세 수량",
            candidates=[
                _candidate(element_key="button|qty-minus|main", name="수량 감소", text="수량 감소", semantic_action_type="", is_quantity_control=True, is_purchase_action=False),
                _candidate(element_key="link|detail|main", name="상품 상세 보기", text="상품 상세 보기", semantic_action_type="", is_detail_trigger=True, is_cart_related=False, is_purchase_action=False),
                _candidate(element_key="button|order|main", name="주문하기", text="주문하기", semantic_action_type="purchase", is_checkout_related=True),
            ],
        )

        state = build_action_opportunities(obs)
        required_names = {
            item.get("target_name") or item.get("target_text")
            for item in state["opportunities"]
            if item.get("required") and item.get("target_name")
        }

        self.assertTrue({"수량 감소", "상품 상세 보기", "주문하기"}.issubset(required_names))

    def test_order_api_500_still_detected_through_anomaly_service(self) -> None:
        after = _obs(text="Checkout")
        after["runtime_signals"]["network_entries"] = [
            {"method": "POST", "url": "http://localhost:5174/api/order", "status": 500}
        ]

        anomalies = detect_anomalies(_obs(text="Checkout"), after, {"action_type": "click_element"})

        self.assertIn("api-5xx", [item["type"] for item in anomalies])

    def test_detail_add_to_cart_without_cart_count_is_button_no_response(self) -> None:
        candidate = _candidate(
            name="Add to cart",
            text="Add to cart",
            semantic_action_type="cart",
            is_purchase_action=True,
            is_cart_related=True,
        )
        before = _obs(text="Wireless Headphones detail Cart", candidates=[candidate])
        after = _obs(text="Wireless Headphones detail Cart", candidates=[candidate])
        before["runtime_signals"]["cart_count_detected"] = False
        before["runtime_signals"]["cart_count"] = None
        before["page_state"]["cart_count_detected"] = False
        before["page_state"]["cart_count"] = None
        after["runtime_signals"]["cart_count_detected"] = False
        after["runtime_signals"]["cart_count"] = None
        after["runtime_signals"]["network_activity"] = True
        after["page_state"]["cart_count_detected"] = False
        after["page_state"]["cart_count"] = None

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertIn("button-no-response", [item["type"] for item in anomalies])

    def test_invalid_cart_quantity_keeps_before_after_values(self) -> None:
        candidate = _candidate(
            name="Decrease quantity",
            text="Decrease quantity",
            semantic_action_type="",
            is_quantity_control=True,
            is_purchase_action=False,
        )
        before = _obs(text="Cart quantity 1 item $10 total $10", candidates=[candidate])
        after = _obs(text="Cart quantity 0 items $0 total $0", candidates=[candidate])

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})
        quantity = next(item for item in anomalies if item["type"] == "cart-quantity-mismatch")

        self.assertEqual(1.0, quantity["evidence"]["quantity_before"])
        self.assertEqual(0.0, quantity["evidence"]["quantity_after"])

    def test_cart_total_mismatch_uses_currency_amounts_not_quantities(self) -> None:
        after = _obs(text="Cart quantity 2 Item A $10 Item B $20 Total $36")

        anomalies = detect_anomalies(_obs(text="Cart quantity 2 Item A $10 Item B $20 Total $30"), after, {"action_type": "inspect_cart"})
        total = next(item for item in anomalies if item["type"] == "cart-total-mismatch")

        self.assertEqual([10.0, 20.0], total["evidence"]["line_item_amounts"])
        self.assertEqual(30.0, total["evidence"]["expected_total_from_visible_amounts"])
        self.assertEqual(36.0, total["evidence"]["displayed_total"])

    def test_wrong_product_detail_page_is_detected(self) -> None:
        candidate = _candidate(
            element_key="link|wireless-headphones|main",
            tag="a",
            role="link",
            name="Wireless Headphones",
            text="Wireless Headphones Details",
            href="/products/headphones",
            semantic_action_type="detail",
            is_detail_trigger=True,
            is_cart_related=False,
            is_purchase_action=False,
        )
        before = _obs(text="Wireless Headphones Smart Watch product list", candidates=[candidate])
        after = _obs(text="Smart Watch product detail Add to cart", candidates=[candidate])
        after["page_state"]["url"] = "http://localhost:5174/products/headphones"

        anomalies = detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}})

        self.assertIn("product-detail-mismatch", [item["type"] for item in anomalies])

    def test_mobile_layout_overlap_includes_specific_element_evidence(self) -> None:
        candidate = _candidate(
            name="Product price Add to cart",
            text="$20 Add to cart",
            is_purchase_action=False,
            is_layout_target=True,
            catalog_bug_id_matches=["SHOP-E06"],
        )
        after = _obs(text="Mobile product price add to cart", candidates=[candidate], viewport_type="mobile")
        after["layout_signals"]["layout_overlap_count"] = 3
        after["page_state"]["viewport_width"] = 390
        after["page_state"]["viewport_height"] = 844

        anomalies = detect_anomalies(
            _obs(text="Mobile product price add to cart", candidates=[candidate], viewport_type="mobile"),
            after,
            {"action_type": "inspect_layout"},
            site_profile={
                "bugs": [
                    {
                        "bug_id": "SHOP-E06",
                        "type": "layout-overlap",
                        "target_keywords": ["price", "add to cart"],
                        "section_keywords": ["product"],
                    }
                ]
            },
        )
        layout = next(item for item in anomalies if item["type"] == "layout-overlap")

        self.assertTrue(layout["evidence"]["specific_element_identified"])
        self.assertEqual(["SHOP-E06"], layout["evidence"]["catalog_bug_id_matches"])

    def test_site001_ground_truth_ids_match_new_ecommerce_types(self) -> None:
        known = load_known_bugs("site001")
        anomalies = [
            {"type": "api-5xx", "confidence": 0.9, "evidence": {"status": 500, "url": "http://localhost:5174/api/order", "method": "POST"}},
            {"type": "cart-total-mismatch", "confidence": 0.78, "evidence": {"line_item_amounts": [10, 20], "displayed_total": 36, "expected_total_from_visible_amounts": 30}},
            {"type": "product-detail-mismatch", "confidence": 0.88, "evidence": {"expected_product": "wireless headphones", "actual_product": "smart watch", "detail_content_mismatch": True}},
            {"type": "cart-quantity-mismatch", "confidence": 0.86, "evidence": {"quantity_before": 1, "quantity_after": 0, "invalid_quantity_visible": True}},
        ]

        matches = match_anomalies_to_known_bugs(anomalies, known, site_id="site001")

        self.assertTrue({"SHOP-E02", "SHOP-E03", "SHOP-E04", "SHOP-E05"}.issubset({item["matched_bug_id"] for item in matches}))


if __name__ == "__main__":
    unittest.main()
