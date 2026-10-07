"""Direct Playwright probe for R10 network/UI semantic evidence.

This is a scoped evidence-production smoke test.  Runtime collection is
GT-blind; GT endpoints are loaded only after browser execution for post-hoc
trigger matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from envs.browsergym_jaws_env import _extract_ui_semantic_fields_from_page, _response_body_capture
from services.r10_network_ui_semantic_evidence import build_network_ui_semantic_evidence


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "training_v3" / "r10_evidence_production"
REGISTRY = ROOT / "artifacts" / "training_v3" / "run4_fakeweb" / "fakeweb_registry.json"
CASES = [
    ("site006", "http://localhost:9115/"),
    ("site007", "http://localhost:9116/"),
    ("site008", "http://localhost:9117/"),
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def clickable_texts(page: Any, limit: int = 12) -> list[tuple[int, str]]:
    rows: list[tuple[int, str, int]] = []
    locator = page.locator("button, a, [role=button], input[type=submit]")
    count = min(locator.count(), 80)
    for idx in range(count):
        item = locator.nth(idx)
        try:
            if not item.is_visible(timeout=200):
                continue
            text = " ".join((item.inner_text(timeout=200) or item.get_attribute("value") or item.get_attribute("aria-label") or "").split())
        except Exception:
            continue
        low = text.lower()
        score = int(any(token in low for token in ("buy", "order", "pay", "subscribe", "refund", "register", "upgrade", "구매", "결제", "변경", "예약")))
        rows.append((score, text, idx))
    return [(idx, text) for score, text, idx in sorted(rows, reverse=True)[:limit]]


def run_case(site: str, url: str) -> dict[str, Any]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        events: list[dict[str, Any]] = []
        request_started: dict[str, int] = {}

        def on_request(request: Any) -> None:
            request_started[str(id(request))] = len(events)
            events.append(
                {
                    "event": "request",
                    "request_id": str(id(request)),
                    "method": request.method,
                    "url": request.url,
                    "resource_type": request.resource_type,
                    "request_body_summary": (request.post_data or "")[:2048],
                }
            )

        def on_response(response: Any) -> None:
            request = response.request
            event = {
                "event": "response",
                "request_id": str(id(request)),
                "method": request.method,
                "url": response.url,
                "resource_type": request.resource_type,
                "status": response.status,
                "response_headers": response.all_headers(),
                "response_received": True,
                "network_failure": False,
            }
            event.update(_response_body_capture(response))
            events.append(event)

        page.on("request", on_request)
        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        actions = []
        try:
            for step, (idx, text) in enumerate(clickable_texts(page), start=1):
                start_len = len(events)
                try:
                    page.locator("button, a, [role=button], input[type=submit]").nth(idx).click(timeout=3000)
                    page.wait_for_timeout(800)
                except Exception as exc:
                    actions.append({"step": step, "action_type": "click", "action_target": text, "error": str(exc)})
                    continue
                step_events = [row for row in events[start_len:] if row.get("event") == "response"]
                observation = {"page_state": {"ui_semantic_fields": _extract_ui_semantic_fields_from_page(page)}}
                evidence = build_network_ui_semantic_evidence(
                    action_context={"action_id": f"{site}:step{step}", "action_type": "click", "action_target": text},
                    network_entries=step_events,
                    observation=observation,
                )
                actions.append(
                    {
                        "step": step,
                        "action_type": "click",
                        "action_target": text,
                        "observed_requests": evidence["requests"],
                        "network_field_count": len(evidence["network_fields"]),
                        "ui_field_count": len(evidence["ui_fields"]),
                        "pair_count": len(evidence["pairs"]),
                        "equal_pairs": evidence["equal_pairs"],
                        "not_equal_pairs": evidence["not_equal_pairs"],
                        "uncertain_pairs": evidence["uncertain_pairs"],
                        "pairs": evidence["pairs"][:12],
                    }
                )
                if evidence["pairs"]:
                    break
        finally:
            browser.close()
    all_requests = [req for action in actions for req in action.get("observed_requests", [])]
    return {
        "site": site,
        "url": url,
        "actions": actions,
        "observed_network_requests": len(all_requests),
        "response_bodies_captured": sum(1 for req in all_requests if req.get("response_body_available")),
        "network_semantic_fields": sum(action.get("network_field_count", 0) for action in actions),
        "ui_semantic_fields": sum(action.get("ui_field_count", 0) for action in actions),
        "network_ui_pairs": sum(action.get("pair_count", 0) for action in actions),
        "equal_pairs": sum(action.get("equal_pairs", 0) for action in actions),
        "not_equal_pairs": sum(action.get("not_equal_pairs", 0) for action in actions),
        "uncertain_pairs": sum(action.get("uncertain_pairs", 0) for action in actions),
        "gt_runtime_access": False,
    }


def annotate_posthoc(rows: list[dict[str, Any]]) -> None:
    registry = read_json(REGISTRY)
    endpoints = {
        item["site_id"]: [
            str(bug.get("affected_endpoint_or_ui") or "").split(" ", 1)[-1]
            for bug in item.get("bugs", [])
            if bug.get("mapped_rule") == "R10"
        ]
        for item in registry.get("sites", [])
    }
    for row in rows:
        matches = []
        for action in row["actions"]:
            for req in action.get("observed_requests", []):
                url = str(req.get("normalized_url") or req.get("url") or "")
                for endpoint in endpoints.get(row["site"], []):
                    if endpoint and endpoint in url:
                        matches.append({"endpoint": endpoint, "url": url, "action_target": action.get("action_target")})
        row["posthoc_gt_trigger_matches"] = matches
        row["posthoc_gt_trigger_matched"] = bool(matches)
        row["r10_relevant_mismatch_evidence_produced"] = bool(matches and row["not_equal_pairs"] > 0)


def main() -> None:
    rows = [run_case(site, url) for site, url in CASES]
    annotate_posthoc(rows)
    summary = {
        "version": "RAWD_V3_R10_DIRECT_PLAYWRIGHT_EVIDENCE_PROBE",
        "representative_cases": len(rows),
        "cases_trigger_executed": sum(1 for row in rows if row["observed_network_requests"] > 0),
        "observed_network_requests": sum(row["observed_network_requests"] for row in rows),
        "response_bodies_captured": sum(row["response_bodies_captured"] for row in rows),
        "network_semantic_fields": sum(row["network_semantic_fields"] for row in rows),
        "ui_semantic_fields": sum(row["ui_semantic_fields"] for row in rows),
        "network_ui_pairs": sum(row["network_ui_pairs"] for row in rows),
        "equal_pairs": sum(row["equal_pairs"] for row in rows),
        "not_equal_pairs": sum(row["not_equal_pairs"] for row in rows),
        "uncertain_pairs": sum(row["uncertain_pairs"] for row in rows),
        "gt_trigger_posthoc_match_count": sum(1 for row in rows if row.get("posthoc_gt_trigger_matched")),
        "r10_mismatch_evidence_success_count": sum(1 for row in rows if row.get("r10_relevant_mismatch_evidence_produced")),
        "gt_runtime_access": False,
        "cases": rows,
    }
    write_json(ART / "r10_direct_playwright_probe_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
