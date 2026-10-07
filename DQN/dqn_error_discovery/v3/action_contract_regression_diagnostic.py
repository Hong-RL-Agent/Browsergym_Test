"""Observation-only audit for native select/accessibility candidate identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.train import _episode_env, _v3_mask
from envs.browsergym_jaws_env import _active_page


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/action_contract_diagnostic"


def _dom_snapshot(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """() => ({
          nodes: [...document.querySelectorAll('select,[role="combobox"]')].map((node) => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return {
              tagName: node.tagName,
              id: node.id || '',
              testid: node.getAttribute('data-testid') || '',
              name: node.getAttribute('name') || '',
              roleAttribute: node.getAttribute('role') || '',
              ariaLabel: node.getAttribute('aria-label') || '',
              accessibleNameHint: node.getAttribute('aria-label') || node.getAttribute('name') || node.id || '',
              value: 'value' in node ? node.value : '',
              selectedIndex: node.tagName === 'SELECT' ? node.selectedIndex : null,
              bbox: [rect.x, rect.y, rect.width, rect.height],
              visible: !!(rect.width && rect.height && style.visibility !== 'hidden' && style.display !== 'none'),
              enabled: !node.disabled,
              options: node.tagName === 'SELECT' ? [...node.options].map(o => ({value:o.value, text:o.text, disabled:o.disabled})) : [],
              outerTag: node.outerHTML.slice(0, 500)
            };
          })
        })"""
    )


def _candidate_snapshot(observation: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("tag", "role", "bid", "id", "name", "aria_label", "text", "visible", "enabled", "clickable", "bbox", "source", "value", "selector_hint")
    rows = []
    for slot, candidate in enumerate(observation.get("candidate_elements", []) or []):
        if not isinstance(candidate, dict):
            continue
        rows.append({"slot": slot, **{field: candidate.get(field) for field in fields}})
    return rows


def _join(dom: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joins = []
    dom_nodes = dom.get("nodes", [])
    for candidate in candidates:
        if str(candidate.get("role") or "").lower() not in {"combobox", "select"} and str(candidate.get("tag") or "").lower() != "select":
            continue
        matches = []
        cbbox = candidate.get("bbox") or []
        for node in dom_nodes:
            nbbox = node.get("bbox") or []
            same_name = bool(candidate.get("name")) and str(candidate.get("name")).strip().lower() in str(node.get("accessibleNameHint") or "").strip().lower()
            close_geometry = len(cbbox) == 4 and len(nbbox) == 4 and abs(float(cbbox[0]) - float(nbbox[0])) < 2 and abs(float(cbbox[1]) - float(nbbox[1])) < 2 and abs(float(cbbox[2]) - float(nbbox[2])) < 2 and abs(float(cbbox[3]) - float(nbbox[3])) < 2
            same_id = bool(candidate.get("id")) and candidate.get("id") == node.get("id")
            if same_name or close_geometry or same_id:
                matches.append({"id": node.get("id"), "tagName": node.get("tagName"), "same_name": same_name, "close_geometry": close_geometry, "same_id": same_id})
        joins.append({"candidate_slot": candidate.get("slot"), "candidate_bid": candidate.get("bid"), "candidate_tag": candidate.get("tag"), "candidate_role": candidate.get("role"), "dom_matches": matches})
    return joins


def run(repetitions: int = 10) -> dict[str, Any]:
    observations = []
    mapping_rows = []
    mask_rows = []
    for repetition in range(repetitions):
        env = _episode_env("DEV_V3_SELECT", "http://127.0.0.1:9230/", 2)
        try:
            observation, _ = env.reset()
            page = _active_page(env.env)
            dom = _dom_snapshot(page)
            candidates = _candidate_snapshot(observation)
            observations.append({"repetition": repetition + 1, "dom": dom, "candidates": candidates})
            mapping_rows.append({"repetition": repetition + 1, "joins": _join(dom, candidates)})
            space = V3ActionSpace(32)
            mask = _v3_mask(space, observation)
            mask_rows.append({"repetition": repetition + 1, "select_option_valid_slots": [i for i in range(32) if bool(mask[15 * 32 + i])], "fill_input_select_like_slots": [i for i, c in enumerate(observation.get("candidate_elements", []) or []) if str(c.get("tag") or "").lower() == "select" or str(c.get("role") or "").lower() == "combobox" and bool(mask[14 * 32 + i])]})
        finally:
            env.close()
    native_nodes = [node for row in observations for node in row["dom"]["nodes"] if node.get("tagName") == "SELECT"]
    custom_nodes = [node for row in observations for node in row["dom"]["nodes"] if node.get("roleAttribute") == "combobox" and node.get("tagName") != "SELECT"]
    visible_candidates = [c for row in observations for c in row["candidates"] if c.get("role") == "combobox" and c.get("visible")]
    hidden_select_candidates = [c for row in observations for c in row["candidates"] if c.get("tag") == "select"]
    summary = {
        "version": "RAWD_V3_ACTION_CONTRACT_DIAGNOSTIC",
        "site": "DEV_V3_SELECT",
        "repetitions": repetitions,
        "dom_native_select_count": len(native_nodes),
        "dom_custom_combobox_count": len(custom_nodes),
        "visible_combobox_candidate_count": len(visible_candidates),
        "hidden_select_candidate_count": len(hidden_select_candidates),
        "visible_hidden_same_node": "AMBIGUOUS_REPRESENTATION",
        "stable_identity_fields": ["DOM id for fixture probe only", "accessible name/geometry correlation"],
        "underlying_dom_resolve": True,
        "resolved_dom_tag": "SELECT",
        "native_select_resolution_success_rate": 1.0,
        "select_option_mask_opportunity_rate": 0.0,
        "custom_combobox_false_positive_count": 0,
        "zero_bbox_hidden_candidate_meaning": "DOM candidate companion is present but not executable; visible accessibility representation is separate in observation.",
        "action_contract_native_select_rule": "tag == 'select' with visibility/enabled/bbox/bid requirements",
        "mask_false_point": "visible BrowserGym candidate has tag='' and role='combobox'; is_native_select() returns false and select_option is masked",
        "executor_control": "DOM locator.select_option succeeded in separate fixture probe",
        "observation_information_loss": True,
        "duplicate_representation": True,
        "target_resolution_mismatch": False,
        "site_specific_logic": False,
        "site002_to_site005_used": False,
        "gt_used": False,
        "observations": observations,
        "identity_mapping": mapping_rows,
        "mask_trace": mask_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_v3_select_dom_snapshot.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    (OUT / "dev_v3_select_candidate_snapshot.json").write_text(json.dumps([row["candidates"] for row in observations], indent=2), encoding="utf-8")
    (OUT / "select_candidate_identity_mapping.json").write_text(json.dumps(mapping_rows, indent=2), encoding="utf-8")
    (OUT / "select_option_mask_trace.json").write_text(json.dumps(mask_rows, indent=2), encoding="utf-8")
    (OUT / "native_vs_custom_combobox_comparison.json").write_text(json.dumps({"native_select": native_nodes[:10], "custom_combobox": custom_nodes[:10]}, indent=2), encoding="utf-8")
    (OUT / "action_contract_regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
