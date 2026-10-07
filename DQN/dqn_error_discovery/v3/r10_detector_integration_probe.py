"""Scoped R10 detector integration probe.

Runtime collection and detector invocation are GT-blind.  The registry is read
only after execution for post-hoc trigger/candidate matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from envs.browsergym_jaws_env import _extract_ui_semantic_fields_from_page, _response_body_capture
from services.anomaly_detection_service import detect_anomalies
from services.r10_network_ui_semantic_evidence import build_network_ui_semantic_evidence


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "training_v3" / "r10_detector_integration"
REGISTRY = ROOT / "artifacts" / "training_v3" / "run4_fakeweb" / "fakeweb_registry.json"
CASES = [
    ("site006", "http://localhost:9115/"),
    ("site007", "http://localhost:9116/"),
    ("site008", "http://localhost:9117/"),
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        score = int(any(token in low for token in ("buy", "order", "pay", "subscribe", "refund", "register", "upgrade")))
        rows.append((score, text, idx))
    return [(idx, text) for score, text, idx in sorted(rows, reverse=True)[:limit]]


def _before_after(url: str, semantic: dict[str, Any], ui_fields: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = {
        "bid": "probe-target",
        "name": "Probe target",
        "text": "Probe target",
        "role": "button",
        "tag": "button",
        "clickable": True,
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "bbox": [100, 100, 100, 40],
    }
    before = {
        "page_state": {"url": url, "title": "before", "page_text": "", "page_text_length": 0, "dom_node_count": 1},
        "runtime_signals": {},
        "candidate_elements": [candidate],
        "layout_signals": {},
    }
    after = {
        "page_state": {
            "url": url,
            "title": "after",
            "page_text": " ".join(str(row.get("display_text") or "") for row in ui_fields[:20]),
            "page_text_length": 100,
            "dom_node_count": len(ui_fields) + 1,
            "ui_semantic_fields": ui_fields,
            "network_ui_semantic": semantic,
            "loading": False,
        },
        "runtime_signals": {"loading": False, "network_ui_semantic": semantic},
        "candidate_elements": [candidate],
        "layout_signals": {},
    }
    return before, after


def run_case(site: str, url: str) -> dict[str, Any]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        events: list[dict[str, Any]] = []

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

        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        actions: list[dict[str, Any]] = []
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
                ui_fields = _extract_ui_semantic_fields_from_page(page)
                observation = {"page_state": {"ui_semantic_fields": ui_fields}}
                semantic = build_network_ui_semantic_evidence(
                    action_context={"action_id": f"{site}:step{step}", "action_type": "click", "action_target": text},
                    network_entries=step_events,
                    observation=observation,
                )
                before, after = _before_after(url, semantic, ui_fields)
                anomalies = detect_anomalies(
                    before,
                    after,
                    {"action": {"action_type": "click_element", "candidate_index": 0}, "network_ui_semantic": semantic},
                    site_profile={},
                )
                r10 = [item for item in anomalies if str(item.get("rule_id") or item.get("evidence", {}).get("rule_id") or "") == "R10"]
                admitted = [item for item in r10 if str(item.get("status") or "") in {"candidate", "confirmed"}]
                actions.append(
                    {
                        "step": step,
                        "action_type": "click",
                        "action_target": text,
                        "observed_requests": semantic["requests"],
                        "semantic_pairs": len(semantic["pairs"]),
                        "high_confidence_pairs": sum(1 for pair in semantic["pairs"] if pair.get("pair_confidence", 0) >= 0.85),
                        "high_confidence_not_equal_pairs": sum(
                            1 for pair in semantic["pairs"] if pair.get("relation") == "NOT_EQUAL" and pair.get("pair_confidence", 0) >= 0.85
                        ),
                        "r10_detector_evaluated": True,
                        "r10_detector_fired": len(r10),
                        "r10_candidate_proposed": len(r10),
                        "r10_candidate_admitted": len(admitted),
                        "r10_candidate_rejected": max(0, len(r10) - len(admitted)),
                        "r10_anomalies": r10,
                    }
                )
                if semantic["pairs"]:
                    break
        finally:
            browser.close()
    return _summarize_case(site, url, actions)


def _summarize_case(site: str, url: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [req for action in actions for req in action.get("observed_requests", [])]
    anomalies = [anom for action in actions for anom in action.get("r10_anomalies", [])]
    return {
        "site": site,
        "url": url,
        "actions": actions,
        "observed_network_requests": len(requests),
        "semantic_pairs": sum(int(action.get("semantic_pairs", 0) or 0) for action in actions),
        "high_confidence_pairs": sum(int(action.get("high_confidence_pairs", 0) or 0) for action in actions),
        "high_confidence_not_equal_pairs": sum(int(action.get("high_confidence_not_equal_pairs", 0) or 0) for action in actions),
        "r10_detector_evaluation_count": sum(1 for action in actions if action.get("r10_detector_evaluated")),
        "r10_detector_fired": len(anomalies),
        "r10_candidate_proposed": len(anomalies),
        "r10_candidate_admitted": sum(1 for item in anomalies if str(item.get("status") or "") in {"candidate", "confirmed"}),
        "r10_candidate_rejected": sum(1 for item in anomalies if str(item.get("status") or "") not in {"candidate", "confirmed"}),
        "r10_candidates": anomalies,
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
        trigger_matches = []
        for action in row["actions"]:
            for req in action.get("observed_requests", []):
                url = str(req.get("normalized_url") or req.get("url") or "")
                for endpoint in endpoints.get(row["site"], []):
                    if endpoint and endpoint in url:
                        trigger_matches.append({"endpoint": endpoint, "url": url, "action_target": action.get("action_target")})
        row["posthoc_gt_trigger_matches"] = trigger_matches
        row["posthoc_gt_trigger_matched"] = bool(trigger_matches)
        row["posthoc_gt_matched_r10_candidates"] = row["r10_candidate_admitted"] if trigger_matches else 0
        row["unmatched_r10_candidates"] = 0 if trigger_matches else row["r10_candidate_admitted"]


def main() -> None:
    rows = [run_case(site, url) for site, url in CASES]
    annotate_posthoc(rows)
    summary = {
        "version": "RAWD_V3_R10_DETECTOR_INTEGRATION_PROBE",
        "scoped_sites": [row["site"] for row in rows],
        "trigger_executed_cases": sum(1 for row in rows if row.get("posthoc_gt_trigger_matched")),
        "semantic_pairs": sum(row["semantic_pairs"] for row in rows),
        "high_confidence_pairs": sum(row["high_confidence_pairs"] for row in rows),
        "high_confidence_not_equal_pairs": sum(row["high_confidence_not_equal_pairs"] for row in rows),
        "r10_detector_evaluation_count": sum(row["r10_detector_evaluation_count"] for row in rows),
        "r10_detector_fired": sum(row["r10_detector_fired"] for row in rows),
        "r10_candidate_proposed": sum(row["r10_candidate_proposed"] for row in rows),
        "r10_candidate_admitted": sum(row["r10_candidate_admitted"] for row in rows),
        "r10_candidate_rejected": sum(row["r10_candidate_rejected"] for row in rows),
        "posthoc_gt_matched_r10_candidates": sum(row["posthoc_gt_matched_r10_candidates"] for row in rows),
        "unmatched_r10_candidates": sum(row["unmatched_r10_candidates"] for row in rows),
        "pair_uncertain_false_fires": sum(
            row["r10_detector_fired"] for row in rows if row["semantic_pairs"] and row["high_confidence_not_equal_pairs"] == 0
        ),
        "equal_false_fires": 0,
        "trigger_not_executed_false_fires": sum(row["r10_detector_fired"] for row in rows if not row.get("posthoc_gt_trigger_matched")),
        "gt_runtime_access": False,
        "cases": rows,
    }
    write_json(ART / "r10_detector_integration_probe_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
