# site9800 Open-ended Anomaly Review

## Current Decision Notes

The latest human review marked the previous open-ended candidates C1-C4 as not confirmed bugs:

- C1 `/dashboard` top search: search textbox click alone is close to normal Enter-driven behavior.
- C2 `/checkout/step/shipping`: checkout feedback has been fixed; validation/error feedback should suppress form-no-feedback.
- C3 `duplicated-rendering add add`: repeated Add buttons across product cards are normal UI unless overlapping inside the same parent/container.
- C4 `/shop/search`: search feedback has been fixed; error/empty/loading feedback should suppress form-no-feedback.

The confirmed evaluation path should use `datasets/site9800/bug_catalog.json` when available. That catalog is prepared for the ERRORS.md issue families: sparse data, forbidden/API mismatch, ASYNC/HANG, timeout/no-feedback, and cart quantity mismatch.

Open-ended output should now treat search-without-query, same-page cart clicks, and normal repeated product-card actions as likely false positives or filter them before they reach the main review list.

This document is generated from the latest open-ended BrowserGym evaluation. It is a review aid, not a confirmed bug catalog.

## Summary

- site_id: `site9800`
- base_url: `http://localhost:9800`
- evaluation_mode: `openended_anomaly_discovery`
- action_counts: `{"click_element": 42, "inspect_layout": 15, "change_viewport_mobile": 3, "inspect_dom": 15}`
- anomaly_types: `{"form-no-feedback": 3, "duplicated-rendering": 1}`
- unique_detected_candidates: `4`
- total_detected_candidates: `75`

## Candidate Review Table

| Candidate ID | Type | Target | URL | Confidence | Human Review Status | Evidence Summary | Review Question | Suggested Decision | Bug Catalog Candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | form-no-feedback | Search products, books, logistics status | http://localhost:9800/dashboard | 0.45 | needs_review | page_text_delta=0, candidate_delta=0, route_changed=False, cart_state_changed=False, query_text= | After entering a query, should Search update results or show validation/no-result feedback? | needs_human_review | review/low-priority |
| C2 | form-no-feedback | Checkout | http://localhost:9800/checkout/step/shipping | 0.75 | needs_review | page_text_delta=0, candidate_delta=0, route_changed=False, cart_state_changed=False, checkout_step=shipping, validation_message_visible=True | Should Checkout advance to the next step or show validation feedback when clicked? | needs_human_review | review/low-priority |
| C3 | duplicated-rendering |  |  | 0.80 | needs_review | duplicated_titles={'add add': 6} | Duplicated text may come from the accessibility tree; verify whether the same component is duplicated in the visible UI. | verify_visible_ui_before_promotion | review/low-priority |
| C4 | form-no-feedback | Search | http://localhost:9800/shop/search | 0.75 | needs_review | page_text_delta=0, candidate_delta=0, route_changed=False, cart_state_changed=False, query_text= | After entering a query, should Search update results or show validation/no-result feedback? | needs_human_review | review/low-priority |

## Review Instructions

1. Reproduce each candidate in a real browser against `http://localhost:9800`.
2. Confirm whether the issue is visible in the UI and not only in AXTree/page text.
3. Treat same-page navigation clicks, empty-input clicks, and clicks that change URL/cart state as likely false positives unless the product spec says otherwise.
4. Promote only confirmed true positives from `datasets/site9800/bug_catalog.candidates.json` to `datasets/site9800/bug_catalog.json`.
5. After promotion, set `has_bug_catalog` to `true` and add `bug_catalog_path` in `configs/training_sites.json`.

## True Positive Criteria

- Expected UI, URL, toast, modal, result list, or validation change is clear and does not occur.
- Validation feedback is absent when required user input is missing.
- Duplicated text/components are visible in the rendered UI.
- Async/hang behavior is reproducible with unresolved loading, timeout, or missing feedback.

## False Positive Criteria

- Already on the target page and clicking the current nav item is an expected no-op.
- Empty input submission is intentionally ignored.
- Duplicate only appears in AXTree/page_text but not the visible UI.
- Route, page text, modal, form, toast, cart state, or candidate set changed after the click.
