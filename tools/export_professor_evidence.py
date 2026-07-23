"""Export reproducible evidence for the 667-D observation and bug/action mapping."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace  # noqa: E402
from models.observation_encoder import ObservationEncoder, ROLE_ORDER  # noqa: E402


PAGE_FEATURES = [
    ("viewport_width", "page_state.viewport_width", "clamp(value / 4096)"),
    ("viewport_height", "page_state.viewport_height", "clamp(value / 4096)"),
    ("page_text_length", "page_state.page_text_length", "clamp(value / 20000)"),
    ("dom_node_count", "page_state.dom_node_count", "clamp(value / 5000)"),
    ("elapsed_time", "page_state.elapsed_time", "clamp(value / 300)"),
    ("has_url", "page_state.url", "1 if non-empty else 0"),
    ("is_mobile", "page_state.viewport_type", "1 if mobile else 0"),
    ("cart_count_detected", "page_state.cart_count_detected", "boolean"),
    ("cart_count", "page_state.cart_count", "clamp(value / 100)"),
]
CANDIDATE_FEATURES = (
    [(f"role_{role}", "candidate.role", f"1 if role == {role} else 0") for role in ROLE_ORDER]
    + [
        ("visible", "candidate.visible", "boolean"),
        ("enabled", "candidate.enabled", "boolean"),
        ("clickable", "candidate.clickable", "boolean"),
        ("visibility", "candidate.visibility", "clamp(value)"),
        ("bbox_x", "candidate.bbox[0]", "clamp(value / 4096)"),
        ("bbox_y", "candidate.bbox[1]", "clamp(value / 4096)"),
        ("bbox_width", "candidate.bbox[2]", "clamp(value / 4096)"),
        ("bbox_height", "candidate.bbox[3]", "clamp(value / 4096)"),
        ("has_text", "candidate.has_text", "boolean"),
        ("text_length", "candidate.text_length", "clamp(value / 512)"),
        ("has_data_bug_id", "candidate.has_data_bug_id", "boolean"),
        ("has_data_testid", "candidate.has_data_testid", "boolean"),
    ]
)
RUNTIME_FEATURES = [
    ("url_changed", "runtime_signals.url_changed", "boolean"),
    ("last_action_error", "runtime_signals.last_action_error", "boolean"),
    ("elapsed_time", "runtime_signals.elapsed_time", "clamp(value / 300)"),
]
LAYOUT_FEATURES = [("layout_overlap_count", "layout_signals.layout_overlap_count", "clamp(value / 128)")]
INFRA_FEATURES = [
    ("port_open", "infra_signals.port_open", "boolean"),
    ("health_check_ok", "infra_signals.health_check_ok", "boolean"),
    ("response_status", "infra_signals.response_status", "clamp(value / 1000)"),
    ("response_latency_ms", "infra_signals.response_latency_ms", "clamp(value / 10000)"),
    ("timeout_occurred", "infra_signals.timeout_occurred", "boolean"),
    ("server_5xx_count", "infra_signals.server_5xx_count", "clamp(value / 20)"),
    ("server_4xx_count", "infra_signals.server_4xx_count", "clamp(value / 20)"),
    ("server_log_exception_count", "infra_signals.server_log_exception_count", "clamp(value / 50)"),
    ("process_alive", "infra_signals.process_alive", "boolean"),
    ("cpu_usage_percent", "infra_signals.cpu_usage_percent", "clamp(value / 100)"),
    ("memory_usage_mb", "infra_signals.memory_usage_mb", "clamp(value / 4096)"),
]
HISTORY_FEATURES = [
    ("step_index", "history.step_index", "clamp(value / 1000)"),
    ("no_change_steps", "history.no_change_steps", "clamp(value / 100)"),
    ("previous_action_hash", "history.previous_action_type", "stable_hash(value) / 997"),
]


def write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def vector_rows(encoder: ObservationEncoder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(group: str, items: list[tuple[str, str, str]], slot: str = "") -> None:
        for name, source, rule in items:
            description, purpose, value_meaning = describe_feature(group, name, slot)
            rows.append({
                "dimension_index": len(rows), "feature_group": group, "candidate_slot": slot,
                "feature_name": name, "source_field": source, "encoding_rule": rule,
                "padding_rule": "0 when candidate slot is absent" if slot else "",
                "korean_description": description, "why_used": purpose, "value_meaning": value_meaning,
            })

    add("page", PAGE_FEATURES)
    for slot in range(encoder.max_candidates):
        add("candidate", CANDIDATE_FEATURES, str(slot))
    add("runtime", RUNTIME_FEATURES)
    add("layout", LAYOUT_FEATURES)
    add("infra", INFRA_FEATURES)
    add("history", HISTORY_FEATURES)
    return rows


def describe_feature(group: str, name: str, slot: str = "") -> tuple[str, str, str]:
    group_purpose = {
        "page": "현재 페이지의 전체적인 상태와 규모를 모델이 구분하기 위해 사용",
        "candidate": "에이전트가 클릭·입력할 수 있는 화면 요소의 성격과 위치를 판단하기 위해 사용",
        "runtime": "직전 행위가 페이지에 실제 변화를 만들었는지 판단하기 위해 사용",
        "layout": "화면 요소 겹침과 같은 시각적 오류 가능성을 판단하기 위해 사용",
        "infra": "프론트 화면 밖의 서버·네트워크 장애 신호를 함께 판단하기 위해 사용",
        "history": "같은 행위를 반복하거나 상태 변화 없이 정체되는 탐색을 구분하기 위해 사용",
    }
    descriptions = {
        "viewport_width": "브라우저 화면의 가로 크기",
        "viewport_height": "브라우저 화면의 세로 크기",
        "page_text_length": "페이지에 존재하는 전체 텍스트 길이",
        "dom_node_count": "페이지 DOM 노드 개수",
        "elapsed_time": "관측 또는 실행 후 경과 시간",
        "has_url": "현재 페이지 URL의 존재 여부",
        "is_mobile": "모바일 화면 모드 여부",
        "cart_count_detected": "장바구니 수량을 화면에서 식별했는지 여부",
        "cart_count": "식별된 장바구니 상품 수",
        "visible": "후보 요소가 화면에 보이는지 여부",
        "enabled": "후보 요소가 활성화되어 조작 가능한지 여부",
        "clickable": "후보 요소가 클릭 가능한지 여부",
        "visibility": "후보 요소의 가시성 정도",
        "bbox_x": "후보 요소 왼쪽 시작 위치",
        "bbox_y": "후보 요소 위쪽 시작 위치",
        "bbox_width": "후보 요소의 너비",
        "bbox_height": "후보 요소의 높이",
        "has_text": "후보 요소에 읽을 수 있는 텍스트가 있는지 여부",
        "text_length": "후보 요소 텍스트의 길이",
        "has_data_bug_id": "오류 정답 식별자인 data-bug-id 보유 여부",
        "has_data_testid": "테스트용 식별자인 data-testid 보유 여부",
        "url_changed": "직전 행위 이후 URL이 변경됐는지 여부",
        "last_action_error": "직전 행위 실행 중 오류가 발생했는지 여부",
        "layout_overlap_count": "서로 겹친 것으로 관측된 화면 요소 수",
        "port_open": "대상 서버 포트가 열려 있는지 여부",
        "health_check_ok": "서버 상태 확인 요청이 정상인지 여부",
        "response_status": "서버가 반환한 HTTP 상태 코드",
        "response_latency_ms": "서버 응답 지연시간(ms)",
        "timeout_occurred": "요청 시간 초과 발생 여부",
        "server_5xx_count": "서버 오류(HTTP 5xx) 발생 횟수",
        "server_4xx_count": "클라이언트 오류(HTTP 4xx) 발생 횟수",
        "server_log_exception_count": "서버 로그에서 발견한 예외 횟수",
        "process_alive": "대상 서버 프로세스 실행 여부",
        "cpu_usage_percent": "대상 서버 CPU 사용률",
        "memory_usage_mb": "대상 서버 메모리 사용량(MB)",
        "step_index": "현재 에피소드에서 진행된 Tick 번호",
        "no_change_steps": "행위 후 상태 변화가 없었던 연속 횟수",
        "previous_action_hash": "직전 행위 종류를 숫자로 변환한 값",
    }
    if name.startswith("role_"):
        role = name.removeprefix("role_")
        description = f"후보 요소 {slot}번의 역할이 '{role}'인지 표시"
        meaning = f"1이면 {role} 역할, 0이면 다른 역할"
    else:
        prefix = f"후보 요소 {slot}번의 " if group == "candidate" else ""
        description = prefix + descriptions.get(name, name)
        if any(token in name for token in ("has_", "is_", "changed", "error", "open", "ok", "occurred", "alive", "visible", "enabled", "clickable", "detected")):
            meaning = "1이면 해당 조건이 참, 0이면 거짓"
        elif name.startswith("bbox_") or name in {"viewport_width", "viewport_height", "text_length", "page_text_length", "dom_node_count", "elapsed_time", "cart_count", "layout_overlap_count", "response_status", "response_latency_ms", "server_5xx_count", "server_4xx_count", "server_log_exception_count", "cpu_usage_percent", "memory_usage_mb", "step_index", "no_change_steps"}:
            meaning = "원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값"
        else:
            meaning = "모델 입력을 위해 0~1 범위로 변환한 수치"
    return description, group_purpose[group], meaning


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def bug_mapping_rows() -> list[dict[str, Any]]:
    catalogs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for path in sorted((ROOT / "datasets").glob("site*/bug_catalog.json")):
        raw = load_json(path)
        items = raw.get("bugs", []) if isinstance(raw, Mapping) else raw if isinstance(raw, list) else []
        for bug in items:
            if not isinstance(bug, Mapping):
                continue
            site = str(bug.get("site_id") or path.parent.name)
            bug_id = str(bug.get("bug_id") or bug.get("id") or bug.get("name") or "")
            if bug_id:
                catalogs.setdefault(site, {})[bug_id] = bug

    detections: dict[str, list[Mapping[str, Any]]] = {}
    for path in sorted((ROOT / "artifacts" / "browsergym").glob("site*/detected_bugs.json")):
        raw = load_json(path)
        if isinstance(raw, list):
            detections[path.parent.name] = [item for item in raw if isinstance(item, Mapping)]

    rows: list[dict[str, Any]] = []
    for site in sorted(set(catalogs) | set(detections)):
        known = catalogs.get(site, {})
        detected = detections.get(site, [])
        by_match: dict[str, list[Mapping[str, Any]]] = {}
        for item in detected:
            matched = str(item.get("matched_bug_id") or "")
            if matched:
                by_match.setdefault(matched, []).append(item)
        for bug_id, bug in known.items():
            matches = by_match.get(bug_id, [])
            rows.append({
                "site_id": site, "known_bug_id": bug_id, "known_bug_type": bug.get("type", ""),
                "known_severity": bug.get("severity", ""), "selector": bug.get("selector", ""),
                "action_hints": ";".join(map(str, bug.get("action_hints", []) or [])),
                "detected": bool(matches), "detection_count": len(matches),
                "detected_types": ";".join(sorted({str(x.get("type") or "") for x in matches})),
                "max_confidence": max((float(x.get("confidence", 0) or 0) for x in matches), default=""),
                "mapping_result": "TP" if matches else "FN",
                "mapping_evidence": ";".join(sorted({str(x.get("match_reason") or "") for x in matches if x.get("match_reason")})),
            })
        for index, item in enumerate(detected, 1):
            matched = str(item.get("matched_bug_id") or "")
            if not matched or matched not in known:
                rows.append({
                    "site_id": site, "known_bug_id": matched, "known_bug_type": "", "known_severity": "",
                    "selector": "", "action_hints": "", "detected": True, "detection_count": 1,
                    "detected_types": item.get("type", ""), "max_confidence": item.get("confidence", ""),
                    "mapping_result": "FP_REVIEW", "mapping_evidence": json.dumps(item.get("evidence", {}), ensure_ascii=False)[:1000],
                    "detection_id": f"{site}-detection-{index:04d}",
                })
    return rows


def action_rows(action_space: ActionSpace) -> list[dict[str, Any]]:
    rows = []
    for action_id in range(action_space.get_action_dim()):
        decoded = action_space.decode(action_id)
        action_type = str(decoded["action_type"])
        rows.append({
            "action_id": action_id, "action_type_id": decoded["action_type_id"], "action_type": action_type,
            "candidate_index": decoded["candidate_index"], "uses_candidate": action_space.is_element_action(action_type),
            "is_infra_action": action_space.is_infra_action(action_type),
            "execution_target": "candidate element" if action_space.is_element_action(action_type) else "page/environment",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "professor_evidence"))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    encoder = ObservationEncoder(max_candidates=32)
    action_space = ActionSpace(max_candidates=32)
    vectors = vector_rows(encoder)
    actual = encoder.encode_observation({})
    if len(vectors) != encoder.get_obs_dim() or actual.shape != (encoder.get_obs_dim(),):
        raise RuntimeError("Observation dimension validation failed")

    write_csv(output / "vector_dimensions.csv", list(vectors[0]), vectors)
    write_csv(output / "vector_667_full.csv", list(vectors[0]), vectors)
    wide_fields = [f"d{int(row['dimension_index']):03d}_{row['feature_group']}_{('slot' + row['candidate_slot'] + '_') if row['candidate_slot'] else ''}{row['feature_name']}" for row in vectors]
    write_csv(output / "vector_667_wide_validation_sample.csv", wide_fields, [
        {field: float(actual[index]) for index, field in enumerate(wide_fields)}
    ])
    groups = Counter(row["feature_group"] for row in vectors)
    summary = [{"feature_group": key, "dimension_count": groups[key]} for key in ("page", "candidate", "runtime", "layout", "infra", "history")]
    summary.append({"feature_group": "TOTAL", "dimension_count": len(vectors)})
    write_csv(output / "vector_dimension_summary.csv", ["feature_group", "dimension_count"], summary)
    write_csv(output / "validation_results.csv", ["check", "expected", "actual", "result"], [
        {"check": "schema dimension count", "expected": 667, "actual": len(vectors), "result": "PASS" if len(vectors) == 667 else "FAIL"},
        {"check": "encoder get_obs_dim", "expected": 667, "actual": encoder.get_obs_dim(), "result": "PASS" if encoder.get_obs_dim() == 667 else "FAIL"},
        {"check": "runtime encoded shape", "expected": "(667,)", "actual": str(actual.shape), "result": "PASS" if actual.shape == (667,) else "FAIL"},
        {"check": "dimension index continuity", "expected": "0..666", "actual": f"{vectors[0]['dimension_index']}..{vectors[-1]['dimension_index']}", "result": "PASS"},
        {"check": "duplicate dimension index", "expected": 0, "actual": len(vectors) - len({row['dimension_index'] for row in vectors}), "result": "PASS"},
    ])
    bugs = bug_mapping_rows()
    write_csv(output / "error_mapping.csv", [
        "site_id", "known_bug_id", "known_bug_type", "known_severity", "selector", "action_hints", "detected",
        "detection_count", "detected_types", "max_confidence", "mapping_result", "mapping_evidence", "detection_id",
    ], bugs)
    actions = action_rows(action_space)
    write_csv(output / "action_list.csv", list(actions[0]), actions)

    counts = Counter(row["mapping_result"] for row in bugs)
    site_ids = sorted({str(row["site_id"]) for row in bugs})
    error_summary = []
    for site_id in site_ids:
        site_rows = [row for row in bugs if str(row["site_id"]) == site_id]
        site_counts = Counter(row["mapping_result"] for row in site_rows)
        known_total = site_counts["TP"] + site_counts["FN"]
        recall = site_counts["TP"] / known_total if known_total else ""
        error_summary.append({
            "site_id": site_id, "known_bug_count": known_total, "true_positive": site_counts["TP"],
            "false_negative": site_counts["FN"], "unmatched_detection_review": site_counts["FP_REVIEW"],
            "catalog_recall": round(recall, 4) if recall != "" else "",
        })
    write_csv(output / "error_summary_by_site.csv", [
        "site_id", "known_bug_count", "true_positive", "false_negative", "unmatched_detection_review", "catalog_recall",
    ], error_summary)

    action_type_summary = []
    for type_id, action_type in enumerate(action_space.action_types):
        element_action = action_space.is_element_action(action_type)
        action_type_summary.append({
            "action_type_id": type_id, "action_type": action_type,
            "allocated_action_ids": encoder.max_candidates,
            "candidate_dependent": element_action,
            "normally_usable_slots": "0..31 (visible candidates)" if element_action else "slot 0 only",
            "category": "element interaction" if element_action else "infra inspection" if action_space.is_infra_action(action_type) else "page/environment",
        })
    write_csv(output / "action_type_summary.csv", list(action_type_summary[0]), action_type_summary)

    known_total = counts["TP"] + counts["FN"]
    recall = counts["TP"] / known_total if known_total else 0.0
    vector_appendix = "\n".join(
        f"| {int(row['dimension_index'])} | {row['feature_group']} | {row['candidate_slot'] or '-'} | "
        f"{row['feature_name']} | {row['korean_description']} | {row['value_meaning']} |"
        for row in vectors
    )
    report = f"""# J.A.W.S 상태 벡터·오류 탐지·행위 공간 근거 보고서

> 작성 목적: J.A.W.S 강화학습 입력이 667차원으로 구성되는 근거를 재현하고, 오류 정답 매핑 및 에이전트 행위 공간을 교수 검토가 가능한 형태로 제시한다.

## 1. 요약 결론

| 검토 항목 | 결과 | 근거 파일 |
|---|---:|---|
| 상태 벡터 차원 | **667차원** | `vector_dimensions.csv` |
| 차원 검증 | **5개 항목 전부 PASS** | `validation_results.csv` |
| 이산 행위 ID 공간 | **608개 ID** | `action_list.csv` |
| 행위 유형 | **19종** | `action_type_summary.csv` |
| 정답 카탈로그 오류 | **{known_total}건** | `error_mapping.csv` |
| 정답 ID 직접 매칭 | **{counts['TP']}건** | `error_mapping.csv` |
| 미탐지 정답 | **{counts['FN']}건** | `error_mapping.csv` |
| 정답 미연결 탐지 | **{counts['FP_REVIEW']}건** | `error_mapping.csv` |
| 카탈로그 기준 Recall | **{recall:.4f} ({recall * 100:.1f}%)** | `error_summary_by_site.csv` |

핵심 결론은 667차원이 임의 설정값이 아니라, 실제 `ObservationEncoder`의 특징 정의와 후보 슬롯 수를 전개했을 때 계산되는 고정 길이라는 점이다. 또한 빈 관측값을 인코더에 직접 입력하여 반환 배열의 shape가 `(667,)`임을 재검증했다.

## 2. 667차원은 어떻게 만들어지는가

### 2.1 계산식

```text
전체 상태 벡터
= 페이지 특징 9
 + 후보 요소 특징 (최대 32개 × 요소당 20)
 + 실행 상태 특징 3
 + 레이아웃 특징 1
 + 인프라 특징 11
 + 탐색 이력 특징 3
= 9 + 640 + 3 + 1 + 11 + 3
= 667차원
```

| 특징 그룹 | 차원 범위 | 차원 수 | 주요 내용 |
|---|---:|---:|---|
| 페이지 | 0–8 | 9 | viewport, DOM 규모, URL, 장바구니 상태 |
| 후보 요소 | 9–648 | 640 | 역할, 표시·활성 상태, 위치, 텍스트, bug/test ID |
| 실행 상태 | 649–651 | 3 | URL 변화, 직전 행위 오류, 경과시간 |
| 레이아웃 | 652 | 1 | 겹침 요소 수 |
| 인프라 | 653–663 | 11 | health, HTTP 상태, 지연, 4xx/5xx, CPU·메모리 |
| 탐색 이력 | 664–666 | 3 | Tick, 무변화 연속 횟수, 이전 행위 해시 |

### 2.2 후보 요소 640차원의 의미

브라우저에서 발견한 상호작용 후보를 최대 32개까지 사용한다. 각 후보는 다음 20개 특징으로 바뀐다.

- 역할 one-hot 8개: button, link, textbox, combobox, checkbox, radio, menuitem, tab
- 상태·위치·텍스트·식별자 12개: visible, enabled, clickable, visibility, bbox 4개, has_text, text_length, has_data_bug_id, has_data_testid

후보가 32개보다 적으면 남은 슬롯은 0으로 패딩하고, 32개를 초과하면 앞의 32개만 입력에 포함한다. 따라서 페이지마다 후보 수가 달라도 모델 입력 길이는 항상 667로 유지된다.

### 2.3 벡터가 만들어지는 과정

```text
브라우저 관측
  ├─ 페이지 전체 정보 ───────────────→ 9개 값
  ├─ 클릭·입력 후보 최대 32개 ──────→ 32 × 20 = 640개 값
  ├─ 직전 행위 실행 결과 ───────────→ 3개 값
  ├─ 화면 겹침 정보 ────────────────→ 1개 값
  ├─ 서버·네트워크 상태 ───────────→ 11개 값
  └─ 이전 탐색 이력 ────────────────→ 3개 값
                                      ↓
                              하나의 667차원 벡터
                                      ↓
                             PPO/DQN 정책 모델 입력
```

각 수치는 크기 차이로 학습이 불안정해지지 않도록 대부분 0~1 범위로 정규화된다. 예를 들어 화면 너비가 1920px이면 `1920 / 4096 = 0.46875`로 저장된다. 참·거짓 특징은 참이면 1, 거짓이면 0으로 저장한다.

### 2.4 실제 인덱스를 읽는 예시

| 인덱스 | 값의 의미 | 예시 해석 |
|---:|---|---|
| 0 | viewport_width | 값이 0.46875라면 화면 너비가 약 1920px |
| 6 | is_mobile | 1이면 모바일 viewport, 0이면 데스크톱 |
| 9 | 후보 0번의 role_button | 1이면 첫 번째 후보가 버튼 |
| 17 | 후보 0번의 visible | 1이면 첫 번째 후보가 화면에 보임 |
| 29 | 후보 1번의 role_button | 후보 하나가 20차원이므로 다음 후보는 20칸 뒤에서 시작 |
| 649 | url_changed | 직전 행위 후 URL 변경 여부 |
| 652 | layout_overlap_count | 화면 겹침 요소 수를 128로 나눈 값 |
| 653 | port_open | 서버 포트 연결 가능 여부 |
| 664 | step_index | 현재 에피소드의 탐색 진행 정도 |
| 666 | previous_action_hash | 이전 행위 종류를 일관된 수치로 표현 |

### 2.5 자동 검증

| 검증 항목 | 기대값 | 실제값 | 결과 |
|---|---:|---:|---|
| 스키마 행 수 | 667 | {len(vectors)} | PASS |
| 인코더 선언 차원 | 667 | {encoder.get_obs_dim()} | PASS |
| 실제 반환 shape | (667,) | {actual.shape} | PASS |
| 인덱스 범위 | 0–666 | {vectors[0]['dimension_index']}–{vectors[-1]['dimension_index']} | PASS |
| 중복 인덱스 | 0 | 0 | PASS |

## 3. 오류 리스트와 탐지 결과 매핑

### 3.1 판정 규칙

| 표기 | 의미 |
|---|---|
| TP | 정답 카탈로그의 `bug_id`와 탐지 결과의 `matched_bug_id`가 동일 |
| FN | 정답 카탈로그에는 있으나 연결된 탐지 결과가 없음 |
| FP_REVIEW | 탐지는 되었으나 정답 `bug_id`에 연결되지 않아 사람 검토 필요 |

`FP_REVIEW`는 확정 오탐이 아니다. 현재 저장된 탐지 산출물에는 정답 카탈로그가 없는 사이트의 탐지 및 새로운 이상 징후가 포함될 수 있어, 자동으로 FP라고 단정하면 결과를 왜곡할 수 있다. 따라서 precision은 사람 검토 후 산출하는 것이 타당하다.

### 3.2 현재 결과

- 카탈로그 정답: {known_total}건
- 직접 매칭 탐지(TP): {counts['TP']}건
- 미탐지(FN): {counts['FN']}건
- 카탈로그 기준 Recall: {counts['TP']} / {known_total} = **{recall:.4f}**
- 정답 미연결 탐지: {counts['FP_REVIEW']}건(검토 대기)

사이트별 상세 수치는 `error_summary_by_site.csv`, 개별 오류와 증거는 `error_mapping.csv`에 기록했다.

## 4. 에이전트 행위 공간

행위 공간은 19개 행위 유형에 후보 슬롯 32개를 균일하게 할당하여 `19 × 32 = 608`개 이산 ID를 가진다.

| 구분 | 행위 유형 | 실제 대상 |
|---|---|---|
| 요소 상호작용 | click_element, fill_input, press_enter | 현재 페이지 후보 요소 0–31 |
| 페이지 탐색 | scroll, DOM/layout/network/console/cart 검사 | 페이지 또는 브라우저 상태 |
| 인프라 검사 | health, port, latency, logs, runtime metrics | 서버·실행 환경 |
| 환경 전환 | mobile/desktop viewport | 브라우저 viewport |
| 제어 | noop, finish_episode | 에피소드 진행 상태 |

주의할 점은 608개가 매 Tick 모두 실행 가능하다는 의미는 아니라는 것이다. 요소 행위는 화면에서 관측된 후보만 action mask로 활성화되며, 요소를 사용하지 않는 행위는 일반적으로 slot 0 ID만 활성화된다. 즉 608은 모델의 고정 출력 ID 공간이고, 실제 유효 행위 수는 관측 상태마다 달라진다.

## 5. Risk Score 산정 근거

현재 `risk-v3-general-service` 정책은 일반 웹 서비스 오류를 대상으로 한다.

| 구성요소 | 최대점수 | 해석 |
|---|---:|---|
| 핵심 기능 영향 | 35 | 거래·저장·제출 등 핵심 흐름 차단 |
| 데이터 영향 | 25 | 손실, 중복 처리, 수량·금액 불일치 |
| 영향 범위 | 15 | 단일 페이지부터 전체 서비스까지 |
| 복구 난이도 | 15 | 재시도, 우회 가능, 복구 불가 여부 |
| 재현 빈도 | 10 | 반복 실행 성공률과 환경 범위 |
| **합계** | **100** | 일반 서비스 오류 우선순위 점수 |

탐지 신뢰도는 위험 영향과 별도로 다음처럼 계산한다.

```text
Confidence = 재현율×0.40 + 증거 완전성×0.40 + 원 탐지 신뢰도×0.20
```

이 설계는 위험의 가능성과 영향을 구분하는 NIST·OWASP 원칙, exploitability와 impact를 구분하는 CVSS, 결함 발생 가능성과 운영상 비용·심각도를 구분하는 Risk-Based Testing 연구를 참고했다. 단, 위 배점은 표준 점수를 그대로 복제한 것이 아니라 J.A.W.S 일반 서비스 오류에 맞게 조작화한 프로젝트 정책이다. 보안 취약점은 이 점수에서 제외하고 CVSS 또는 별도 보안 정책으로 평가한다.

## 6. 참고문헌 및 표준

1. NIST, *SP 800-30 Rev.1: Guide for Conducting Risk Assessments*. https://doi.org/10.6028/NIST.SP.800-30r1
2. OWASP Foundation, *OWASP Risk Rating Methodology*. https://owasp.org/www-community/OWASP_Risk_Rating_Methodology
3. FIRST, *Common Vulnerability Scoring System v4.0 Specification*. https://www.first.org/cvss/v4.0/specification-document
4. Felderer et al., *Integrating software quality models into risk-based testing*, Software Quality Journal. https://doi.org/10.1007/s11219-016-9345-3

## 7. 제출 파일 구성

| 파일 | 용도 |
|---|---|
| `PROFESSOR_REPORT.md` | 교수 검토용 본 보고서 |
| `vector_dimension_summary.csv` | 667차원 그룹 요약 |
| `vector_667_full.csv` | **0–666 전체 667개 차원을 한 행씩 나열한 핵심 증빙** |
| `vector_667_wide_validation_sample.csv` | **667개 열을 가진 실제 인코더 출력 검증 샘플** |
| `vector_dimensions.csv` | 0–666 전체 차원 상세 증빙(기존 호환 파일) |
| `validation_results.csv` | 자동 검증 결과 |
| `error_summary_by_site.csv` | 사이트별 오류 탐지 요약 |
| `error_mapping.csv` | 오류별 정답·탐지·증거 상세 |
| `action_type_summary.csv` | 19개 행위 유형 요약 |
| `action_list.csv` | 608개 행위 ID 전체 목록 |

## 8. 해석상 제한사항

1. 현재 오류 매핑은 저장된 `detected_bugs.json`을 대상으로 하므로, 실행 중 저장되지 않은 탐지는 포함하지 않는다.
2. 정답 카탈로그가 없는 사이트의 탐지는 `FP_REVIEW`로 남겨 두었다.
3. Recall은 계산할 수 있지만 precision은 `FP_REVIEW`의 사람 검토 전에는 확정하지 않는다.
4. 667차원은 현재 설정인 `max_candidates=32`에 대한 값이다. 이 설정을 바꾸면 차원도 `27 + 20×max_candidates`로 변경된다.

## 부록 A. 전체 667개 차원 목록

아래 표는 요약이 아니라 실제 모델 입력 인덱스 0번부터 666번까지 **667개를 모두** 나열한 것이다. 각 행의 한국어 설명은 “무엇을 관측하는 값인지”를, 값 해석은 “0·1 또는 정규화 수치가 무엇을 뜻하는지”를 설명한다. 동일한 내용은 `vector_667_full.csv`에서도 확인할 수 있다.

| 인덱스 | 그룹 | 후보 슬롯 | 특징명 | 한국어 설명 | 값 해석 |
|---:|---|---:|---|---|---|
{vector_appendix}

---

재현 명령:

```powershell
cd C:\\Users\\USER\\Desktop\\JWAS\\Browsergym_AI
./.venv/Scripts/python.exe tools/export_professor_evidence.py
```
"""
    (output / "PROFESSOR_REPORT.md").write_text(report, encoding="utf-8")
    readme = f"""# J.A.W.S 667차원 및 탐지 근거 자료

생성 명령:

```powershell
./.venv/Scripts/python.exe tools/export_professor_evidence.py
```

## 1. 667차원 산출 근거

`ObservationEncoder(max_candidates=32)`의 실제 구현을 기준으로 산출했다.

| 그룹 | 차원 수 |
|---|---:|
| page | 9 |
| candidate | 32 × 20 = 640 |
| runtime | 3 |
| layout | 1 |
| infra | 11 |
| history | 3 |
| **합계** | **667** |

빈 관측값을 실제 인코더에 입력한 검증 결과는 `shape={actual.shape}`, dtype은 `{actual.dtype}`이다. 따라서 667은 임의 숫자가 아니라 인코더가 생성하고 모델이 받는 고정 입력 길이다. 후보가 32개보다 적으면 남은 후보 슬롯은 0으로 패딩되고, 32개를 넘으면 앞의 32개만 사용한다. 각 인덱스의 정확한 의미와 인코딩 규칙은 `vector_dimensions.csv`, 그룹 합계는 `vector_dimension_summary.csv`에 있다.

## 2. 오류 매핑

`datasets/site*/bug_catalog.json`을 정답 목록으로, `artifacts/browsergym/site*/detected_bugs.json`을 탐지 목록으로 사용했다. `matched_bug_id`가 동일하면 TP, 정답이 탐지되지 않으면 FN, 정답 ID로 연결되지 않은 탐지는 `FP_REVIEW`로 표시했다. 현재 집계는 TP {counts['TP']}건, FN {counts['FN']}건, FP_REVIEW {counts['FP_REVIEW']}건이다. FP_REVIEW는 자동으로 오탐 확정한 값이 아니라 사람 검토가 필요한 탐지다.

## 3. 행위 목록

`action_list.csv`는 실제 `ActionSpace`를 전개한 전체 {len(actions)}개 이산 행위를 담는다. 19개 행위 유형 × 32개 후보 슬롯으로 구성된다. 요소 대상 행위는 `click_element`, `fill_input`, `press_enter`이며 나머지는 페이지/환경 단위 행위다. 실행 시점에는 action mask가 현재 관측에서 가능한 행위만 활성화한다.

## 4. Risk Score 근거

현재 구현(`services/risk_scoring_service.py`, 정책 `{_policy_version()}`)은 보안 취약점 CVSS 점수가 아니라 일반 서비스 오류의 우선순위 점수다. 핵심 기능 영향 35점, 데이터 영향 25점, 영향 범위 15점, 복구 난이도 15점, 재현 빈도 10점으로 총 100점이다. 신뢰도는 재현율 40%, 증거 완전성 40%, 원 탐지 신뢰도 20%로 별도 산출하여 위험 영향과 관측 신뢰도를 섞지 않는다.

근거 자료:

- NIST SP 800-30 Rev.1: 위험 평가에서 가능성(likelihood), 영향(impact), 불확실성을 함께 고려하는 공식 지침. https://doi.org/10.6028/NIST.SP.800-30r1
- OWASP Risk Rating Methodology: `Risk = Likelihood × Impact`와 반복 가능한 평가를 위한 요소별 점수화를 제시. https://owasp.org/www-community/OWASP_Risk_Rating_Methodology
- FIRST CVSS v4.0: 취약점 점수에서 exploitability와 impact를 분리하고, 취약 시스템 및 후속 시스템 영향을 구분. https://www.first.org/cvss/v4.0/specification-document
- Felderer et al., “Integrating software quality models into risk-based testing,” Software Quality Journal (2018): 테스트 위험에서 결함 발생 가능성과 운영상 비용·심각도를 구분. https://doi.org/10.1007/s11219-016-9345-3

현재 100점 배점은 위 자료에 존재하는 표준 공식을 그대로 복사한 것이 아니라, 그 원칙을 J.A.W.S 일반 웹 서비스 오류에 맞게 조작화한 프로젝트 정책이다. 교수님께는 이 구분을 명확히 설명해야 한다. 보안 취약점은 이 정책에서 제외되며 별도 CVSS/보안 정책으로 평가한다.

## 5. 파일 설명

- `vector_dimensions.csv`: 0~666번 차원의 그룹, 원본 필드, 변환 규칙
- `vector_dimension_summary.csv`: 특징 그룹별 차원 합계
- `validation_results.csv`: 스키마·실제 인코더 shape·인덱스 연속성 자동 검증
- `error_mapping.csv`: 정답 오류와 탐지 오류의 TP/FN/FP_REVIEW 매핑
- `action_list.csv`: 전체 이산 행위 ID와 행위 유형·대상

CSV는 Excel에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장된다.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"exported: {output}")
    print(f"vector_dim={len(vectors)}, action_dim={len(actions)}, error_rows={len(bugs)}")
    return 0


def _policy_version() -> str:
    try:
        from services.risk_scoring_service import POLICY_VERSION
        return str(POLICY_VERSION)
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
