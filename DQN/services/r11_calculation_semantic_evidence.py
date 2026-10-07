"""GT-blind numeric calculation evidence extracted from runtime UI observations."""

from __future__ import annotations

import re
from typing import Any, Mapping


NUMERIC_TOLERANCE = 0.01
HIGH_CONFIDENCE = 0.9

ROLE_ALIASES = {
    "unit_price": ("unit price", "price each", "unit cost", "단가", "개당", "현재가", "주문 가격"),
    "quantity": ("quantity", "qty", "count", "수량", "개수", "인원", "주문 수량"),
    "subtotal": ("subtotal", "sub total", "소계", "상품 합계", "식료품 소계"),
    "discount": ("discount", "coupon", "할인", "쿠폰"),
    "tax": ("tax", "vat", "세금", "부가세"),
    "shipping": ("shipping", "delivery fee", "배송비", "운임"),
    "total": ("grand total", "total", "amount due", "총액", "합계", "결제 금액", "최종가", "주문 금액"),
    "balance": ("balance", "잔액"),
    "usage": ("usage", "사용량", "사용 금액"),
    "rate": ("rate", "percentage", "비율", "요율", "할인율"),
}

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_PLACEHOLDER = re.compile(r"(?:loading|calculating|pending|로딩|계산\s*중|[-—]{2,})", re.IGNORECASE)


def parse_numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = " ".join(str(value).strip().split())
    if not text or _PLACEHOLDER.search(text):
        return None
    match = _NUMBER.search(text.replace("−", "-"))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def semantic_role(label: Any) -> str:
    text = " ".join(str(label or "").strip().lower().replace("_", " ").split())
    if not text:
        return ""
    # Prefer specific roles before the broad ``total`` aliases.
    for role in ("unit_price", "quantity", "subtotal", "discount", "tax", "shipping", "balance", "usage", "rate", "total"):
        if any(alias in text for alias in ROLE_ALIASES[role]):
            return role
    return ""


def extract_numeric_semantic_fields(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for row in page.get("ui_semantic_fields", []) or []:
        if isinstance(row, Mapping):
            _append_field(rows, row.get("key") or row.get("label"), row.get("value"), "dom_semantic")
    for candidate in observation.get("candidate_elements", []) or []:
        if not isinstance(candidate, Mapping) or not bool(candidate.get("visible", True)):
            continue
        label = candidate.get("label") or candidate.get("name") or candidate.get("aria_label") or candidate.get("text")
        value = candidate.get("value")
        if value not in (None, ""):
            _append_field(rows, label, value, "candidate")
        for text_key in ("text", "name", "aria_label"):
            text = candidate.get(text_key)
            if text:
                _append_labeled_text(rows, str(text), "candidate_text")
    text = str(page.get("page_text") or page.get("page_text_sample") or "")
    for line in re.split(r"[\r\n|]+", text):
        _append_labeled_text(rows, line, "visible_text")
    deduped: dict[tuple[str, float, str], dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault((row["role"], row["value"], row["source"]), row)
    return list(deduped.values())


def build_calculation_semantic_evidence(observation: Mapping[str, Any]) -> dict[str, Any]:
    fields = extract_numeric_semantic_fields(observation)
    by_role: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        by_role.setdefault(field["role"], []).append(field)
    calculations: list[dict[str, Any]] = []
    # High confidence requires one unambiguous value for every role.
    unit = _single(by_role, "unit_price")
    quantity = _single(by_role, "quantity")
    displayed = _single(by_role, "subtotal") or _single(by_role, "total")
    if unit and quantity and displayed:
        calculations.append(_calculation("MULTIPLY", [unit, quantity], displayed, unit["value"] * quantity["value"], "unit_price*quantity"))
    subtotal = _single(by_role, "subtotal")
    shipping = _single(by_role, "shipping")
    discount = _single(by_role, "discount")
    total = _single(by_role, "total")
    if subtotal and shipping and total:
        calculations.append(_calculation("ADD", [subtotal, shipping], total, subtotal["value"] + shipping["value"], "subtotal+shipping"))
    if subtotal and discount and total:
        calculations.append(_calculation("SUBTRACT", [subtotal, discount], total, subtotal["value"] - discount["value"], "subtotal-discount"))
    return {"numeric_fields": fields, "calculations": calculations, "gt_runtime_access": False}


def _append_field(rows: list[dict[str, Any]], label: Any, raw: Any, source: str) -> None:
    role = semantic_role(label)
    value = parse_numeric_value(raw)
    if role and value is not None:
        rows.append({"key": role, "role": role, "value": value, "raw_value": raw, "label": str(label or ""), "source": source})


def _append_labeled_text(rows: list[dict[str, Any]], text: str, source: str) -> None:
    compact = " ".join(text.strip().split())
    if not compact or _PLACEHOLDER.search(compact):
        return
    for role, aliases in ROLE_ALIASES.items():
        for alias in aliases:
            match = re.search(rf"{re.escape(alias)}\s*[:=]?\s*([^|;\n]{{0,40}})", compact, re.IGNORECASE)
            if match:
                value = parse_numeric_value(match.group(1))
                if value is not None:
                    rows.append({"key": role, "role": role, "value": value, "raw_value": match.group(1).strip(), "label": alias, "source": source})
                break


def _single(by_role: Mapping[str, list[dict[str, Any]]], role: str) -> dict[str, Any] | None:
    rows = by_role.get(role, [])
    values = {row["value"] for row in rows}
    return rows[0] if len(values) == 1 else None


def _calculation(operator: str, operands: list[dict[str, Any]], displayed: dict[str, Any], expected: float, formula: str) -> dict[str, Any]:
    tolerance = max(NUMERIC_TOLERANCE, abs(expected) * 0.000001)
    difference = displayed["value"] - expected
    relation = "EQUAL" if abs(difference) <= tolerance else "NOT_EQUAL"
    return {
        "operands": [{"key": row["role"], "value": row["value"]} for row in operands],
        "operator": operator,
        "formula": formula,
        "expected_value": expected,
        "displayed_key": displayed["role"],
        "displayed_value": displayed["value"],
        "relation": relation,
        "confidence": HIGH_CONFIDENCE,
        "tolerance": tolerance,
    }
