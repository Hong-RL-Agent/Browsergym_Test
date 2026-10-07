"""Native select executor used by v3 readiness tests and future env wiring."""

from __future__ import annotations

from typing import Any, Mapping


def execute_select_option(page: Any, candidate: Mapping[str, Any], previously_selected: set[str] | None = None) -> dict[str, Any]:
    bid = str(candidate.get("canonical_dom_bid") or candidate.get("bid") or "")
    if not bid:
        raise ValueError("select_option requires a candidate bid")
    locator = page.locator(f'[bid="{bid}"]')
    current = str(locator.input_value() or "")
    blocked = previously_selected or set()
    options = locator.locator("option")
    selected = None
    for index in range(options.count()):
        option = options.nth(index)
        value = str(option.get_attribute("value") or "")
        text = str(option.text_content() or "").strip()
        disabled = option.get_attribute("disabled") is not None
        placeholder = not value and not text
        if disabled or placeholder or value == current or value in blocked:
            continue
        selected = {"value": value, "text": text, "index": index}
        break
    if selected is None:
        raise ValueError("native select has no selectable option")
    locator.select_option(value=selected["value"])
    return {"selected": True, "bid": bid, "previous_value": current, "selected_option": selected}
