# J.A.W.S 상태 벡터·오류 탐지·행위 공간 근거 보고서

> 작성 목적: J.A.W.S 강화학습 입력이 667차원으로 구성되는 근거를 재현하고, 오류 정답 매핑 및 에이전트 행위 공간을 교수 검토가 가능한 형태로 제시한다.

## 1. 요약 결론

| 검토 항목 | 결과 | 근거 파일 |
|---|---:|---|
| 상태 벡터 차원 | **667차원** | `vector_dimensions.csv` |
| 차원 검증 | **5개 항목 전부 PASS** | `validation_results.csv` |
| 이산 행위 ID 공간 | **608개 ID** | `action_list.csv` |
| 행위 유형 | **19종** | `action_type_summary.csv` |
| 정답 카탈로그 오류 | **23건** | `error_mapping.csv` |
| 정답 ID 직접 매칭 | **8건** | `error_mapping.csv` |
| 미탐지 정답 | **15건** | `error_mapping.csv` |
| 정답 미연결 탐지 | **104건** | `error_mapping.csv` |
| 카탈로그 기준 Recall | **0.3478 (34.8%)** | `error_summary_by_site.csv` |

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
| 스키마 행 수 | 667 | 667 | PASS |
| 인코더 선언 차원 | 667 | 667 | PASS |
| 실제 반환 shape | (667,) | (667,) | PASS |
| 인덱스 범위 | 0–666 | 0–666 | PASS |
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

- 카탈로그 정답: 23건
- 직접 매칭 탐지(TP): 8건
- 미탐지(FN): 15건
- 카탈로그 기준 Recall: 8 / 23 = **0.3478**
- 정답 미연결 탐지: 104건(검토 대기)

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
| 0 | page | - | viewport_width | 브라우저 화면의 가로 크기 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 1 | page | - | viewport_height | 브라우저 화면의 세로 크기 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 2 | page | - | page_text_length | 페이지에 존재하는 전체 텍스트 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 3 | page | - | dom_node_count | 페이지 DOM 노드 개수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 4 | page | - | elapsed_time | 관측 또는 실행 후 경과 시간 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 5 | page | - | has_url | 현재 페이지 URL의 존재 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 6 | page | - | is_mobile | 모바일 화면 모드 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 7 | page | - | cart_count_detected | 장바구니 수량을 화면에서 식별했는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 8 | page | - | cart_count | 식별된 장바구니 상품 수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 9 | candidate | 0 | role_button | 후보 요소 0번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 10 | candidate | 0 | role_link | 후보 요소 0번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 11 | candidate | 0 | role_textbox | 후보 요소 0번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 12 | candidate | 0 | role_combobox | 후보 요소 0번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 13 | candidate | 0 | role_checkbox | 후보 요소 0번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 14 | candidate | 0 | role_radio | 후보 요소 0번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 15 | candidate | 0 | role_menuitem | 후보 요소 0번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 16 | candidate | 0 | role_tab | 후보 요소 0번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 17 | candidate | 0 | visible | 후보 요소 0번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 18 | candidate | 0 | enabled | 후보 요소 0번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 19 | candidate | 0 | clickable | 후보 요소 0번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 20 | candidate | 0 | visibility | 후보 요소 0번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 21 | candidate | 0 | bbox_x | 후보 요소 0번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 22 | candidate | 0 | bbox_y | 후보 요소 0번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 23 | candidate | 0 | bbox_width | 후보 요소 0번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 24 | candidate | 0 | bbox_height | 후보 요소 0번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 25 | candidate | 0 | has_text | 후보 요소 0번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 26 | candidate | 0 | text_length | 후보 요소 0번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 27 | candidate | 0 | has_data_bug_id | 후보 요소 0번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 28 | candidate | 0 | has_data_testid | 후보 요소 0번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 29 | candidate | 1 | role_button | 후보 요소 1번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 30 | candidate | 1 | role_link | 후보 요소 1번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 31 | candidate | 1 | role_textbox | 후보 요소 1번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 32 | candidate | 1 | role_combobox | 후보 요소 1번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 33 | candidate | 1 | role_checkbox | 후보 요소 1번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 34 | candidate | 1 | role_radio | 후보 요소 1번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 35 | candidate | 1 | role_menuitem | 후보 요소 1번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 36 | candidate | 1 | role_tab | 후보 요소 1번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 37 | candidate | 1 | visible | 후보 요소 1번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 38 | candidate | 1 | enabled | 후보 요소 1번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 39 | candidate | 1 | clickable | 후보 요소 1번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 40 | candidate | 1 | visibility | 후보 요소 1번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 41 | candidate | 1 | bbox_x | 후보 요소 1번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 42 | candidate | 1 | bbox_y | 후보 요소 1번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 43 | candidate | 1 | bbox_width | 후보 요소 1번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 44 | candidate | 1 | bbox_height | 후보 요소 1번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 45 | candidate | 1 | has_text | 후보 요소 1번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 46 | candidate | 1 | text_length | 후보 요소 1번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 47 | candidate | 1 | has_data_bug_id | 후보 요소 1번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 48 | candidate | 1 | has_data_testid | 후보 요소 1번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 49 | candidate | 2 | role_button | 후보 요소 2번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 50 | candidate | 2 | role_link | 후보 요소 2번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 51 | candidate | 2 | role_textbox | 후보 요소 2번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 52 | candidate | 2 | role_combobox | 후보 요소 2번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 53 | candidate | 2 | role_checkbox | 후보 요소 2번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 54 | candidate | 2 | role_radio | 후보 요소 2번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 55 | candidate | 2 | role_menuitem | 후보 요소 2번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 56 | candidate | 2 | role_tab | 후보 요소 2번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 57 | candidate | 2 | visible | 후보 요소 2번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 58 | candidate | 2 | enabled | 후보 요소 2번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 59 | candidate | 2 | clickable | 후보 요소 2번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 60 | candidate | 2 | visibility | 후보 요소 2번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 61 | candidate | 2 | bbox_x | 후보 요소 2번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 62 | candidate | 2 | bbox_y | 후보 요소 2번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 63 | candidate | 2 | bbox_width | 후보 요소 2번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 64 | candidate | 2 | bbox_height | 후보 요소 2번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 65 | candidate | 2 | has_text | 후보 요소 2번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 66 | candidate | 2 | text_length | 후보 요소 2번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 67 | candidate | 2 | has_data_bug_id | 후보 요소 2번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 68 | candidate | 2 | has_data_testid | 후보 요소 2번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 69 | candidate | 3 | role_button | 후보 요소 3번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 70 | candidate | 3 | role_link | 후보 요소 3번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 71 | candidate | 3 | role_textbox | 후보 요소 3번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 72 | candidate | 3 | role_combobox | 후보 요소 3번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 73 | candidate | 3 | role_checkbox | 후보 요소 3번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 74 | candidate | 3 | role_radio | 후보 요소 3번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 75 | candidate | 3 | role_menuitem | 후보 요소 3번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 76 | candidate | 3 | role_tab | 후보 요소 3번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 77 | candidate | 3 | visible | 후보 요소 3번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 78 | candidate | 3 | enabled | 후보 요소 3번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 79 | candidate | 3 | clickable | 후보 요소 3번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 80 | candidate | 3 | visibility | 후보 요소 3번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 81 | candidate | 3 | bbox_x | 후보 요소 3번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 82 | candidate | 3 | bbox_y | 후보 요소 3번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 83 | candidate | 3 | bbox_width | 후보 요소 3번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 84 | candidate | 3 | bbox_height | 후보 요소 3번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 85 | candidate | 3 | has_text | 후보 요소 3번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 86 | candidate | 3 | text_length | 후보 요소 3번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 87 | candidate | 3 | has_data_bug_id | 후보 요소 3번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 88 | candidate | 3 | has_data_testid | 후보 요소 3번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 89 | candidate | 4 | role_button | 후보 요소 4번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 90 | candidate | 4 | role_link | 후보 요소 4번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 91 | candidate | 4 | role_textbox | 후보 요소 4번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 92 | candidate | 4 | role_combobox | 후보 요소 4번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 93 | candidate | 4 | role_checkbox | 후보 요소 4번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 94 | candidate | 4 | role_radio | 후보 요소 4번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 95 | candidate | 4 | role_menuitem | 후보 요소 4번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 96 | candidate | 4 | role_tab | 후보 요소 4번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 97 | candidate | 4 | visible | 후보 요소 4번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 98 | candidate | 4 | enabled | 후보 요소 4번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 99 | candidate | 4 | clickable | 후보 요소 4번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 100 | candidate | 4 | visibility | 후보 요소 4번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 101 | candidate | 4 | bbox_x | 후보 요소 4번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 102 | candidate | 4 | bbox_y | 후보 요소 4번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 103 | candidate | 4 | bbox_width | 후보 요소 4번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 104 | candidate | 4 | bbox_height | 후보 요소 4번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 105 | candidate | 4 | has_text | 후보 요소 4번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 106 | candidate | 4 | text_length | 후보 요소 4번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 107 | candidate | 4 | has_data_bug_id | 후보 요소 4번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 108 | candidate | 4 | has_data_testid | 후보 요소 4번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 109 | candidate | 5 | role_button | 후보 요소 5번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 110 | candidate | 5 | role_link | 후보 요소 5번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 111 | candidate | 5 | role_textbox | 후보 요소 5번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 112 | candidate | 5 | role_combobox | 후보 요소 5번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 113 | candidate | 5 | role_checkbox | 후보 요소 5번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 114 | candidate | 5 | role_radio | 후보 요소 5번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 115 | candidate | 5 | role_menuitem | 후보 요소 5번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 116 | candidate | 5 | role_tab | 후보 요소 5번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 117 | candidate | 5 | visible | 후보 요소 5번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 118 | candidate | 5 | enabled | 후보 요소 5번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 119 | candidate | 5 | clickable | 후보 요소 5번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 120 | candidate | 5 | visibility | 후보 요소 5번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 121 | candidate | 5 | bbox_x | 후보 요소 5번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 122 | candidate | 5 | bbox_y | 후보 요소 5번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 123 | candidate | 5 | bbox_width | 후보 요소 5번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 124 | candidate | 5 | bbox_height | 후보 요소 5번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 125 | candidate | 5 | has_text | 후보 요소 5번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 126 | candidate | 5 | text_length | 후보 요소 5번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 127 | candidate | 5 | has_data_bug_id | 후보 요소 5번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 128 | candidate | 5 | has_data_testid | 후보 요소 5번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 129 | candidate | 6 | role_button | 후보 요소 6번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 130 | candidate | 6 | role_link | 후보 요소 6번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 131 | candidate | 6 | role_textbox | 후보 요소 6번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 132 | candidate | 6 | role_combobox | 후보 요소 6번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 133 | candidate | 6 | role_checkbox | 후보 요소 6번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 134 | candidate | 6 | role_radio | 후보 요소 6번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 135 | candidate | 6 | role_menuitem | 후보 요소 6번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 136 | candidate | 6 | role_tab | 후보 요소 6번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 137 | candidate | 6 | visible | 후보 요소 6번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 138 | candidate | 6 | enabled | 후보 요소 6번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 139 | candidate | 6 | clickable | 후보 요소 6번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 140 | candidate | 6 | visibility | 후보 요소 6번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 141 | candidate | 6 | bbox_x | 후보 요소 6번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 142 | candidate | 6 | bbox_y | 후보 요소 6번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 143 | candidate | 6 | bbox_width | 후보 요소 6번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 144 | candidate | 6 | bbox_height | 후보 요소 6번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 145 | candidate | 6 | has_text | 후보 요소 6번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 146 | candidate | 6 | text_length | 후보 요소 6번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 147 | candidate | 6 | has_data_bug_id | 후보 요소 6번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 148 | candidate | 6 | has_data_testid | 후보 요소 6번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 149 | candidate | 7 | role_button | 후보 요소 7번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 150 | candidate | 7 | role_link | 후보 요소 7번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 151 | candidate | 7 | role_textbox | 후보 요소 7번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 152 | candidate | 7 | role_combobox | 후보 요소 7번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 153 | candidate | 7 | role_checkbox | 후보 요소 7번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 154 | candidate | 7 | role_radio | 후보 요소 7번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 155 | candidate | 7 | role_menuitem | 후보 요소 7번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 156 | candidate | 7 | role_tab | 후보 요소 7번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 157 | candidate | 7 | visible | 후보 요소 7번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 158 | candidate | 7 | enabled | 후보 요소 7번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 159 | candidate | 7 | clickable | 후보 요소 7번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 160 | candidate | 7 | visibility | 후보 요소 7번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 161 | candidate | 7 | bbox_x | 후보 요소 7번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 162 | candidate | 7 | bbox_y | 후보 요소 7번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 163 | candidate | 7 | bbox_width | 후보 요소 7번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 164 | candidate | 7 | bbox_height | 후보 요소 7번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 165 | candidate | 7 | has_text | 후보 요소 7번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 166 | candidate | 7 | text_length | 후보 요소 7번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 167 | candidate | 7 | has_data_bug_id | 후보 요소 7번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 168 | candidate | 7 | has_data_testid | 후보 요소 7번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 169 | candidate | 8 | role_button | 후보 요소 8번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 170 | candidate | 8 | role_link | 후보 요소 8번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 171 | candidate | 8 | role_textbox | 후보 요소 8번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 172 | candidate | 8 | role_combobox | 후보 요소 8번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 173 | candidate | 8 | role_checkbox | 후보 요소 8번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 174 | candidate | 8 | role_radio | 후보 요소 8번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 175 | candidate | 8 | role_menuitem | 후보 요소 8번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 176 | candidate | 8 | role_tab | 후보 요소 8번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 177 | candidate | 8 | visible | 후보 요소 8번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 178 | candidate | 8 | enabled | 후보 요소 8번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 179 | candidate | 8 | clickable | 후보 요소 8번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 180 | candidate | 8 | visibility | 후보 요소 8번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 181 | candidate | 8 | bbox_x | 후보 요소 8번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 182 | candidate | 8 | bbox_y | 후보 요소 8번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 183 | candidate | 8 | bbox_width | 후보 요소 8번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 184 | candidate | 8 | bbox_height | 후보 요소 8번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 185 | candidate | 8 | has_text | 후보 요소 8번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 186 | candidate | 8 | text_length | 후보 요소 8번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 187 | candidate | 8 | has_data_bug_id | 후보 요소 8번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 188 | candidate | 8 | has_data_testid | 후보 요소 8번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 189 | candidate | 9 | role_button | 후보 요소 9번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 190 | candidate | 9 | role_link | 후보 요소 9번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 191 | candidate | 9 | role_textbox | 후보 요소 9번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 192 | candidate | 9 | role_combobox | 후보 요소 9번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 193 | candidate | 9 | role_checkbox | 후보 요소 9번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 194 | candidate | 9 | role_radio | 후보 요소 9번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 195 | candidate | 9 | role_menuitem | 후보 요소 9번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 196 | candidate | 9 | role_tab | 후보 요소 9번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 197 | candidate | 9 | visible | 후보 요소 9번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 198 | candidate | 9 | enabled | 후보 요소 9번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 199 | candidate | 9 | clickable | 후보 요소 9번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 200 | candidate | 9 | visibility | 후보 요소 9번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 201 | candidate | 9 | bbox_x | 후보 요소 9번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 202 | candidate | 9 | bbox_y | 후보 요소 9번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 203 | candidate | 9 | bbox_width | 후보 요소 9번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 204 | candidate | 9 | bbox_height | 후보 요소 9번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 205 | candidate | 9 | has_text | 후보 요소 9번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 206 | candidate | 9 | text_length | 후보 요소 9번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 207 | candidate | 9 | has_data_bug_id | 후보 요소 9번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 208 | candidate | 9 | has_data_testid | 후보 요소 9번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 209 | candidate | 10 | role_button | 후보 요소 10번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 210 | candidate | 10 | role_link | 후보 요소 10번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 211 | candidate | 10 | role_textbox | 후보 요소 10번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 212 | candidate | 10 | role_combobox | 후보 요소 10번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 213 | candidate | 10 | role_checkbox | 후보 요소 10번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 214 | candidate | 10 | role_radio | 후보 요소 10번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 215 | candidate | 10 | role_menuitem | 후보 요소 10번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 216 | candidate | 10 | role_tab | 후보 요소 10번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 217 | candidate | 10 | visible | 후보 요소 10번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 218 | candidate | 10 | enabled | 후보 요소 10번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 219 | candidate | 10 | clickable | 후보 요소 10번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 220 | candidate | 10 | visibility | 후보 요소 10번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 221 | candidate | 10 | bbox_x | 후보 요소 10번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 222 | candidate | 10 | bbox_y | 후보 요소 10번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 223 | candidate | 10 | bbox_width | 후보 요소 10번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 224 | candidate | 10 | bbox_height | 후보 요소 10번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 225 | candidate | 10 | has_text | 후보 요소 10번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 226 | candidate | 10 | text_length | 후보 요소 10번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 227 | candidate | 10 | has_data_bug_id | 후보 요소 10번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 228 | candidate | 10 | has_data_testid | 후보 요소 10번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 229 | candidate | 11 | role_button | 후보 요소 11번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 230 | candidate | 11 | role_link | 후보 요소 11번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 231 | candidate | 11 | role_textbox | 후보 요소 11번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 232 | candidate | 11 | role_combobox | 후보 요소 11번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 233 | candidate | 11 | role_checkbox | 후보 요소 11번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 234 | candidate | 11 | role_radio | 후보 요소 11번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 235 | candidate | 11 | role_menuitem | 후보 요소 11번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 236 | candidate | 11 | role_tab | 후보 요소 11번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 237 | candidate | 11 | visible | 후보 요소 11번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 238 | candidate | 11 | enabled | 후보 요소 11번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 239 | candidate | 11 | clickable | 후보 요소 11번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 240 | candidate | 11 | visibility | 후보 요소 11번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 241 | candidate | 11 | bbox_x | 후보 요소 11번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 242 | candidate | 11 | bbox_y | 후보 요소 11번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 243 | candidate | 11 | bbox_width | 후보 요소 11번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 244 | candidate | 11 | bbox_height | 후보 요소 11번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 245 | candidate | 11 | has_text | 후보 요소 11번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 246 | candidate | 11 | text_length | 후보 요소 11번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 247 | candidate | 11 | has_data_bug_id | 후보 요소 11번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 248 | candidate | 11 | has_data_testid | 후보 요소 11번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 249 | candidate | 12 | role_button | 후보 요소 12번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 250 | candidate | 12 | role_link | 후보 요소 12번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 251 | candidate | 12 | role_textbox | 후보 요소 12번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 252 | candidate | 12 | role_combobox | 후보 요소 12번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 253 | candidate | 12 | role_checkbox | 후보 요소 12번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 254 | candidate | 12 | role_radio | 후보 요소 12번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 255 | candidate | 12 | role_menuitem | 후보 요소 12번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 256 | candidate | 12 | role_tab | 후보 요소 12번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 257 | candidate | 12 | visible | 후보 요소 12번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 258 | candidate | 12 | enabled | 후보 요소 12번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 259 | candidate | 12 | clickable | 후보 요소 12번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 260 | candidate | 12 | visibility | 후보 요소 12번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 261 | candidate | 12 | bbox_x | 후보 요소 12번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 262 | candidate | 12 | bbox_y | 후보 요소 12번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 263 | candidate | 12 | bbox_width | 후보 요소 12번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 264 | candidate | 12 | bbox_height | 후보 요소 12번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 265 | candidate | 12 | has_text | 후보 요소 12번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 266 | candidate | 12 | text_length | 후보 요소 12번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 267 | candidate | 12 | has_data_bug_id | 후보 요소 12번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 268 | candidate | 12 | has_data_testid | 후보 요소 12번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 269 | candidate | 13 | role_button | 후보 요소 13번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 270 | candidate | 13 | role_link | 후보 요소 13번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 271 | candidate | 13 | role_textbox | 후보 요소 13번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 272 | candidate | 13 | role_combobox | 후보 요소 13번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 273 | candidate | 13 | role_checkbox | 후보 요소 13번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 274 | candidate | 13 | role_radio | 후보 요소 13번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 275 | candidate | 13 | role_menuitem | 후보 요소 13번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 276 | candidate | 13 | role_tab | 후보 요소 13번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 277 | candidate | 13 | visible | 후보 요소 13번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 278 | candidate | 13 | enabled | 후보 요소 13번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 279 | candidate | 13 | clickable | 후보 요소 13번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 280 | candidate | 13 | visibility | 후보 요소 13번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 281 | candidate | 13 | bbox_x | 후보 요소 13번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 282 | candidate | 13 | bbox_y | 후보 요소 13번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 283 | candidate | 13 | bbox_width | 후보 요소 13번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 284 | candidate | 13 | bbox_height | 후보 요소 13번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 285 | candidate | 13 | has_text | 후보 요소 13번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 286 | candidate | 13 | text_length | 후보 요소 13번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 287 | candidate | 13 | has_data_bug_id | 후보 요소 13번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 288 | candidate | 13 | has_data_testid | 후보 요소 13번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 289 | candidate | 14 | role_button | 후보 요소 14번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 290 | candidate | 14 | role_link | 후보 요소 14번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 291 | candidate | 14 | role_textbox | 후보 요소 14번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 292 | candidate | 14 | role_combobox | 후보 요소 14번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 293 | candidate | 14 | role_checkbox | 후보 요소 14번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 294 | candidate | 14 | role_radio | 후보 요소 14번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 295 | candidate | 14 | role_menuitem | 후보 요소 14번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 296 | candidate | 14 | role_tab | 후보 요소 14번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 297 | candidate | 14 | visible | 후보 요소 14번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 298 | candidate | 14 | enabled | 후보 요소 14번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 299 | candidate | 14 | clickable | 후보 요소 14번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 300 | candidate | 14 | visibility | 후보 요소 14번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 301 | candidate | 14 | bbox_x | 후보 요소 14번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 302 | candidate | 14 | bbox_y | 후보 요소 14번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 303 | candidate | 14 | bbox_width | 후보 요소 14번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 304 | candidate | 14 | bbox_height | 후보 요소 14번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 305 | candidate | 14 | has_text | 후보 요소 14번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 306 | candidate | 14 | text_length | 후보 요소 14번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 307 | candidate | 14 | has_data_bug_id | 후보 요소 14번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 308 | candidate | 14 | has_data_testid | 후보 요소 14번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 309 | candidate | 15 | role_button | 후보 요소 15번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 310 | candidate | 15 | role_link | 후보 요소 15번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 311 | candidate | 15 | role_textbox | 후보 요소 15번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 312 | candidate | 15 | role_combobox | 후보 요소 15번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 313 | candidate | 15 | role_checkbox | 후보 요소 15번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 314 | candidate | 15 | role_radio | 후보 요소 15번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 315 | candidate | 15 | role_menuitem | 후보 요소 15번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 316 | candidate | 15 | role_tab | 후보 요소 15번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 317 | candidate | 15 | visible | 후보 요소 15번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 318 | candidate | 15 | enabled | 후보 요소 15번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 319 | candidate | 15 | clickable | 후보 요소 15번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 320 | candidate | 15 | visibility | 후보 요소 15번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 321 | candidate | 15 | bbox_x | 후보 요소 15번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 322 | candidate | 15 | bbox_y | 후보 요소 15번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 323 | candidate | 15 | bbox_width | 후보 요소 15번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 324 | candidate | 15 | bbox_height | 후보 요소 15번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 325 | candidate | 15 | has_text | 후보 요소 15번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 326 | candidate | 15 | text_length | 후보 요소 15번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 327 | candidate | 15 | has_data_bug_id | 후보 요소 15번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 328 | candidate | 15 | has_data_testid | 후보 요소 15번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 329 | candidate | 16 | role_button | 후보 요소 16번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 330 | candidate | 16 | role_link | 후보 요소 16번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 331 | candidate | 16 | role_textbox | 후보 요소 16번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 332 | candidate | 16 | role_combobox | 후보 요소 16번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 333 | candidate | 16 | role_checkbox | 후보 요소 16번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 334 | candidate | 16 | role_radio | 후보 요소 16번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 335 | candidate | 16 | role_menuitem | 후보 요소 16번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 336 | candidate | 16 | role_tab | 후보 요소 16번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 337 | candidate | 16 | visible | 후보 요소 16번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 338 | candidate | 16 | enabled | 후보 요소 16번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 339 | candidate | 16 | clickable | 후보 요소 16번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 340 | candidate | 16 | visibility | 후보 요소 16번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 341 | candidate | 16 | bbox_x | 후보 요소 16번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 342 | candidate | 16 | bbox_y | 후보 요소 16번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 343 | candidate | 16 | bbox_width | 후보 요소 16번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 344 | candidate | 16 | bbox_height | 후보 요소 16번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 345 | candidate | 16 | has_text | 후보 요소 16번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 346 | candidate | 16 | text_length | 후보 요소 16번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 347 | candidate | 16 | has_data_bug_id | 후보 요소 16번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 348 | candidate | 16 | has_data_testid | 후보 요소 16번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 349 | candidate | 17 | role_button | 후보 요소 17번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 350 | candidate | 17 | role_link | 후보 요소 17번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 351 | candidate | 17 | role_textbox | 후보 요소 17번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 352 | candidate | 17 | role_combobox | 후보 요소 17번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 353 | candidate | 17 | role_checkbox | 후보 요소 17번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 354 | candidate | 17 | role_radio | 후보 요소 17번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 355 | candidate | 17 | role_menuitem | 후보 요소 17번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 356 | candidate | 17 | role_tab | 후보 요소 17번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 357 | candidate | 17 | visible | 후보 요소 17번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 358 | candidate | 17 | enabled | 후보 요소 17번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 359 | candidate | 17 | clickable | 후보 요소 17번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 360 | candidate | 17 | visibility | 후보 요소 17번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 361 | candidate | 17 | bbox_x | 후보 요소 17번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 362 | candidate | 17 | bbox_y | 후보 요소 17번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 363 | candidate | 17 | bbox_width | 후보 요소 17번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 364 | candidate | 17 | bbox_height | 후보 요소 17번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 365 | candidate | 17 | has_text | 후보 요소 17번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 366 | candidate | 17 | text_length | 후보 요소 17번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 367 | candidate | 17 | has_data_bug_id | 후보 요소 17번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 368 | candidate | 17 | has_data_testid | 후보 요소 17번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 369 | candidate | 18 | role_button | 후보 요소 18번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 370 | candidate | 18 | role_link | 후보 요소 18번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 371 | candidate | 18 | role_textbox | 후보 요소 18번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 372 | candidate | 18 | role_combobox | 후보 요소 18번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 373 | candidate | 18 | role_checkbox | 후보 요소 18번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 374 | candidate | 18 | role_radio | 후보 요소 18번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 375 | candidate | 18 | role_menuitem | 후보 요소 18번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 376 | candidate | 18 | role_tab | 후보 요소 18번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 377 | candidate | 18 | visible | 후보 요소 18번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 378 | candidate | 18 | enabled | 후보 요소 18번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 379 | candidate | 18 | clickable | 후보 요소 18번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 380 | candidate | 18 | visibility | 후보 요소 18번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 381 | candidate | 18 | bbox_x | 후보 요소 18번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 382 | candidate | 18 | bbox_y | 후보 요소 18번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 383 | candidate | 18 | bbox_width | 후보 요소 18번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 384 | candidate | 18 | bbox_height | 후보 요소 18번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 385 | candidate | 18 | has_text | 후보 요소 18번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 386 | candidate | 18 | text_length | 후보 요소 18번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 387 | candidate | 18 | has_data_bug_id | 후보 요소 18번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 388 | candidate | 18 | has_data_testid | 후보 요소 18번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 389 | candidate | 19 | role_button | 후보 요소 19번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 390 | candidate | 19 | role_link | 후보 요소 19번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 391 | candidate | 19 | role_textbox | 후보 요소 19번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 392 | candidate | 19 | role_combobox | 후보 요소 19번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 393 | candidate | 19 | role_checkbox | 후보 요소 19번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 394 | candidate | 19 | role_radio | 후보 요소 19번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 395 | candidate | 19 | role_menuitem | 후보 요소 19번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 396 | candidate | 19 | role_tab | 후보 요소 19번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 397 | candidate | 19 | visible | 후보 요소 19번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 398 | candidate | 19 | enabled | 후보 요소 19번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 399 | candidate | 19 | clickable | 후보 요소 19번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 400 | candidate | 19 | visibility | 후보 요소 19번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 401 | candidate | 19 | bbox_x | 후보 요소 19번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 402 | candidate | 19 | bbox_y | 후보 요소 19번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 403 | candidate | 19 | bbox_width | 후보 요소 19번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 404 | candidate | 19 | bbox_height | 후보 요소 19번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 405 | candidate | 19 | has_text | 후보 요소 19번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 406 | candidate | 19 | text_length | 후보 요소 19번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 407 | candidate | 19 | has_data_bug_id | 후보 요소 19번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 408 | candidate | 19 | has_data_testid | 후보 요소 19번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 409 | candidate | 20 | role_button | 후보 요소 20번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 410 | candidate | 20 | role_link | 후보 요소 20번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 411 | candidate | 20 | role_textbox | 후보 요소 20번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 412 | candidate | 20 | role_combobox | 후보 요소 20번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 413 | candidate | 20 | role_checkbox | 후보 요소 20번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 414 | candidate | 20 | role_radio | 후보 요소 20번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 415 | candidate | 20 | role_menuitem | 후보 요소 20번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 416 | candidate | 20 | role_tab | 후보 요소 20번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 417 | candidate | 20 | visible | 후보 요소 20번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 418 | candidate | 20 | enabled | 후보 요소 20번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 419 | candidate | 20 | clickable | 후보 요소 20번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 420 | candidate | 20 | visibility | 후보 요소 20번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 421 | candidate | 20 | bbox_x | 후보 요소 20번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 422 | candidate | 20 | bbox_y | 후보 요소 20번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 423 | candidate | 20 | bbox_width | 후보 요소 20번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 424 | candidate | 20 | bbox_height | 후보 요소 20번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 425 | candidate | 20 | has_text | 후보 요소 20번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 426 | candidate | 20 | text_length | 후보 요소 20번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 427 | candidate | 20 | has_data_bug_id | 후보 요소 20번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 428 | candidate | 20 | has_data_testid | 후보 요소 20번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 429 | candidate | 21 | role_button | 후보 요소 21번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 430 | candidate | 21 | role_link | 후보 요소 21번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 431 | candidate | 21 | role_textbox | 후보 요소 21번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 432 | candidate | 21 | role_combobox | 후보 요소 21번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 433 | candidate | 21 | role_checkbox | 후보 요소 21번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 434 | candidate | 21 | role_radio | 후보 요소 21번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 435 | candidate | 21 | role_menuitem | 후보 요소 21번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 436 | candidate | 21 | role_tab | 후보 요소 21번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 437 | candidate | 21 | visible | 후보 요소 21번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 438 | candidate | 21 | enabled | 후보 요소 21번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 439 | candidate | 21 | clickable | 후보 요소 21번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 440 | candidate | 21 | visibility | 후보 요소 21번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 441 | candidate | 21 | bbox_x | 후보 요소 21번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 442 | candidate | 21 | bbox_y | 후보 요소 21번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 443 | candidate | 21 | bbox_width | 후보 요소 21번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 444 | candidate | 21 | bbox_height | 후보 요소 21번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 445 | candidate | 21 | has_text | 후보 요소 21번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 446 | candidate | 21 | text_length | 후보 요소 21번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 447 | candidate | 21 | has_data_bug_id | 후보 요소 21번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 448 | candidate | 21 | has_data_testid | 후보 요소 21번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 449 | candidate | 22 | role_button | 후보 요소 22번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 450 | candidate | 22 | role_link | 후보 요소 22번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 451 | candidate | 22 | role_textbox | 후보 요소 22번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 452 | candidate | 22 | role_combobox | 후보 요소 22번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 453 | candidate | 22 | role_checkbox | 후보 요소 22번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 454 | candidate | 22 | role_radio | 후보 요소 22번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 455 | candidate | 22 | role_menuitem | 후보 요소 22번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 456 | candidate | 22 | role_tab | 후보 요소 22번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 457 | candidate | 22 | visible | 후보 요소 22번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 458 | candidate | 22 | enabled | 후보 요소 22번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 459 | candidate | 22 | clickable | 후보 요소 22번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 460 | candidate | 22 | visibility | 후보 요소 22번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 461 | candidate | 22 | bbox_x | 후보 요소 22번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 462 | candidate | 22 | bbox_y | 후보 요소 22번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 463 | candidate | 22 | bbox_width | 후보 요소 22번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 464 | candidate | 22 | bbox_height | 후보 요소 22번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 465 | candidate | 22 | has_text | 후보 요소 22번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 466 | candidate | 22 | text_length | 후보 요소 22번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 467 | candidate | 22 | has_data_bug_id | 후보 요소 22번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 468 | candidate | 22 | has_data_testid | 후보 요소 22번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 469 | candidate | 23 | role_button | 후보 요소 23번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 470 | candidate | 23 | role_link | 후보 요소 23번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 471 | candidate | 23 | role_textbox | 후보 요소 23번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 472 | candidate | 23 | role_combobox | 후보 요소 23번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 473 | candidate | 23 | role_checkbox | 후보 요소 23번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 474 | candidate | 23 | role_radio | 후보 요소 23번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 475 | candidate | 23 | role_menuitem | 후보 요소 23번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 476 | candidate | 23 | role_tab | 후보 요소 23번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 477 | candidate | 23 | visible | 후보 요소 23번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 478 | candidate | 23 | enabled | 후보 요소 23번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 479 | candidate | 23 | clickable | 후보 요소 23번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 480 | candidate | 23 | visibility | 후보 요소 23번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 481 | candidate | 23 | bbox_x | 후보 요소 23번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 482 | candidate | 23 | bbox_y | 후보 요소 23번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 483 | candidate | 23 | bbox_width | 후보 요소 23번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 484 | candidate | 23 | bbox_height | 후보 요소 23번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 485 | candidate | 23 | has_text | 후보 요소 23번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 486 | candidate | 23 | text_length | 후보 요소 23번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 487 | candidate | 23 | has_data_bug_id | 후보 요소 23번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 488 | candidate | 23 | has_data_testid | 후보 요소 23번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 489 | candidate | 24 | role_button | 후보 요소 24번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 490 | candidate | 24 | role_link | 후보 요소 24번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 491 | candidate | 24 | role_textbox | 후보 요소 24번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 492 | candidate | 24 | role_combobox | 후보 요소 24번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 493 | candidate | 24 | role_checkbox | 후보 요소 24번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 494 | candidate | 24 | role_radio | 후보 요소 24번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 495 | candidate | 24 | role_menuitem | 후보 요소 24번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 496 | candidate | 24 | role_tab | 후보 요소 24번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 497 | candidate | 24 | visible | 후보 요소 24번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 498 | candidate | 24 | enabled | 후보 요소 24번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 499 | candidate | 24 | clickable | 후보 요소 24번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 500 | candidate | 24 | visibility | 후보 요소 24번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 501 | candidate | 24 | bbox_x | 후보 요소 24번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 502 | candidate | 24 | bbox_y | 후보 요소 24번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 503 | candidate | 24 | bbox_width | 후보 요소 24번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 504 | candidate | 24 | bbox_height | 후보 요소 24번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 505 | candidate | 24 | has_text | 후보 요소 24번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 506 | candidate | 24 | text_length | 후보 요소 24번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 507 | candidate | 24 | has_data_bug_id | 후보 요소 24번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 508 | candidate | 24 | has_data_testid | 후보 요소 24번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 509 | candidate | 25 | role_button | 후보 요소 25번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 510 | candidate | 25 | role_link | 후보 요소 25번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 511 | candidate | 25 | role_textbox | 후보 요소 25번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 512 | candidate | 25 | role_combobox | 후보 요소 25번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 513 | candidate | 25 | role_checkbox | 후보 요소 25번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 514 | candidate | 25 | role_radio | 후보 요소 25번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 515 | candidate | 25 | role_menuitem | 후보 요소 25번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 516 | candidate | 25 | role_tab | 후보 요소 25번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 517 | candidate | 25 | visible | 후보 요소 25번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 518 | candidate | 25 | enabled | 후보 요소 25번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 519 | candidate | 25 | clickable | 후보 요소 25번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 520 | candidate | 25 | visibility | 후보 요소 25번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 521 | candidate | 25 | bbox_x | 후보 요소 25번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 522 | candidate | 25 | bbox_y | 후보 요소 25번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 523 | candidate | 25 | bbox_width | 후보 요소 25번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 524 | candidate | 25 | bbox_height | 후보 요소 25번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 525 | candidate | 25 | has_text | 후보 요소 25번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 526 | candidate | 25 | text_length | 후보 요소 25번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 527 | candidate | 25 | has_data_bug_id | 후보 요소 25번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 528 | candidate | 25 | has_data_testid | 후보 요소 25번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 529 | candidate | 26 | role_button | 후보 요소 26번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 530 | candidate | 26 | role_link | 후보 요소 26번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 531 | candidate | 26 | role_textbox | 후보 요소 26번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 532 | candidate | 26 | role_combobox | 후보 요소 26번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 533 | candidate | 26 | role_checkbox | 후보 요소 26번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 534 | candidate | 26 | role_radio | 후보 요소 26번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 535 | candidate | 26 | role_menuitem | 후보 요소 26번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 536 | candidate | 26 | role_tab | 후보 요소 26번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 537 | candidate | 26 | visible | 후보 요소 26번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 538 | candidate | 26 | enabled | 후보 요소 26번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 539 | candidate | 26 | clickable | 후보 요소 26번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 540 | candidate | 26 | visibility | 후보 요소 26번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 541 | candidate | 26 | bbox_x | 후보 요소 26번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 542 | candidate | 26 | bbox_y | 후보 요소 26번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 543 | candidate | 26 | bbox_width | 후보 요소 26번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 544 | candidate | 26 | bbox_height | 후보 요소 26번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 545 | candidate | 26 | has_text | 후보 요소 26번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 546 | candidate | 26 | text_length | 후보 요소 26번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 547 | candidate | 26 | has_data_bug_id | 후보 요소 26번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 548 | candidate | 26 | has_data_testid | 후보 요소 26번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 549 | candidate | 27 | role_button | 후보 요소 27번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 550 | candidate | 27 | role_link | 후보 요소 27번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 551 | candidate | 27 | role_textbox | 후보 요소 27번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 552 | candidate | 27 | role_combobox | 후보 요소 27번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 553 | candidate | 27 | role_checkbox | 후보 요소 27번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 554 | candidate | 27 | role_radio | 후보 요소 27번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 555 | candidate | 27 | role_menuitem | 후보 요소 27번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 556 | candidate | 27 | role_tab | 후보 요소 27번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 557 | candidate | 27 | visible | 후보 요소 27번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 558 | candidate | 27 | enabled | 후보 요소 27번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 559 | candidate | 27 | clickable | 후보 요소 27번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 560 | candidate | 27 | visibility | 후보 요소 27번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 561 | candidate | 27 | bbox_x | 후보 요소 27번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 562 | candidate | 27 | bbox_y | 후보 요소 27번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 563 | candidate | 27 | bbox_width | 후보 요소 27번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 564 | candidate | 27 | bbox_height | 후보 요소 27번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 565 | candidate | 27 | has_text | 후보 요소 27번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 566 | candidate | 27 | text_length | 후보 요소 27번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 567 | candidate | 27 | has_data_bug_id | 후보 요소 27번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 568 | candidate | 27 | has_data_testid | 후보 요소 27번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 569 | candidate | 28 | role_button | 후보 요소 28번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 570 | candidate | 28 | role_link | 후보 요소 28번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 571 | candidate | 28 | role_textbox | 후보 요소 28번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 572 | candidate | 28 | role_combobox | 후보 요소 28번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 573 | candidate | 28 | role_checkbox | 후보 요소 28번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 574 | candidate | 28 | role_radio | 후보 요소 28번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 575 | candidate | 28 | role_menuitem | 후보 요소 28번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 576 | candidate | 28 | role_tab | 후보 요소 28번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 577 | candidate | 28 | visible | 후보 요소 28번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 578 | candidate | 28 | enabled | 후보 요소 28번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 579 | candidate | 28 | clickable | 후보 요소 28번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 580 | candidate | 28 | visibility | 후보 요소 28번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 581 | candidate | 28 | bbox_x | 후보 요소 28번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 582 | candidate | 28 | bbox_y | 후보 요소 28번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 583 | candidate | 28 | bbox_width | 후보 요소 28번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 584 | candidate | 28 | bbox_height | 후보 요소 28번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 585 | candidate | 28 | has_text | 후보 요소 28번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 586 | candidate | 28 | text_length | 후보 요소 28번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 587 | candidate | 28 | has_data_bug_id | 후보 요소 28번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 588 | candidate | 28 | has_data_testid | 후보 요소 28번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 589 | candidate | 29 | role_button | 후보 요소 29번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 590 | candidate | 29 | role_link | 후보 요소 29번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 591 | candidate | 29 | role_textbox | 후보 요소 29번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 592 | candidate | 29 | role_combobox | 후보 요소 29번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 593 | candidate | 29 | role_checkbox | 후보 요소 29번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 594 | candidate | 29 | role_radio | 후보 요소 29번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 595 | candidate | 29 | role_menuitem | 후보 요소 29번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 596 | candidate | 29 | role_tab | 후보 요소 29번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 597 | candidate | 29 | visible | 후보 요소 29번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 598 | candidate | 29 | enabled | 후보 요소 29번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 599 | candidate | 29 | clickable | 후보 요소 29번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 600 | candidate | 29 | visibility | 후보 요소 29번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 601 | candidate | 29 | bbox_x | 후보 요소 29번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 602 | candidate | 29 | bbox_y | 후보 요소 29번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 603 | candidate | 29 | bbox_width | 후보 요소 29번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 604 | candidate | 29 | bbox_height | 후보 요소 29번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 605 | candidate | 29 | has_text | 후보 요소 29번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 606 | candidate | 29 | text_length | 후보 요소 29번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 607 | candidate | 29 | has_data_bug_id | 후보 요소 29번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 608 | candidate | 29 | has_data_testid | 후보 요소 29번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 609 | candidate | 30 | role_button | 후보 요소 30번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 610 | candidate | 30 | role_link | 후보 요소 30번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 611 | candidate | 30 | role_textbox | 후보 요소 30번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 612 | candidate | 30 | role_combobox | 후보 요소 30번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 613 | candidate | 30 | role_checkbox | 후보 요소 30번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 614 | candidate | 30 | role_radio | 후보 요소 30번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 615 | candidate | 30 | role_menuitem | 후보 요소 30번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 616 | candidate | 30 | role_tab | 후보 요소 30번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 617 | candidate | 30 | visible | 후보 요소 30번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 618 | candidate | 30 | enabled | 후보 요소 30번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 619 | candidate | 30 | clickable | 후보 요소 30번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 620 | candidate | 30 | visibility | 후보 요소 30번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 621 | candidate | 30 | bbox_x | 후보 요소 30번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 622 | candidate | 30 | bbox_y | 후보 요소 30번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 623 | candidate | 30 | bbox_width | 후보 요소 30번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 624 | candidate | 30 | bbox_height | 후보 요소 30번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 625 | candidate | 30 | has_text | 후보 요소 30번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 626 | candidate | 30 | text_length | 후보 요소 30번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 627 | candidate | 30 | has_data_bug_id | 후보 요소 30번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 628 | candidate | 30 | has_data_testid | 후보 요소 30번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 629 | candidate | 31 | role_button | 후보 요소 31번의 역할이 'button'인지 표시 | 1이면 button 역할, 0이면 다른 역할 |
| 630 | candidate | 31 | role_link | 후보 요소 31번의 역할이 'link'인지 표시 | 1이면 link 역할, 0이면 다른 역할 |
| 631 | candidate | 31 | role_textbox | 후보 요소 31번의 역할이 'textbox'인지 표시 | 1이면 textbox 역할, 0이면 다른 역할 |
| 632 | candidate | 31 | role_combobox | 후보 요소 31번의 역할이 'combobox'인지 표시 | 1이면 combobox 역할, 0이면 다른 역할 |
| 633 | candidate | 31 | role_checkbox | 후보 요소 31번의 역할이 'checkbox'인지 표시 | 1이면 checkbox 역할, 0이면 다른 역할 |
| 634 | candidate | 31 | role_radio | 후보 요소 31번의 역할이 'radio'인지 표시 | 1이면 radio 역할, 0이면 다른 역할 |
| 635 | candidate | 31 | role_menuitem | 후보 요소 31번의 역할이 'menuitem'인지 표시 | 1이면 menuitem 역할, 0이면 다른 역할 |
| 636 | candidate | 31 | role_tab | 후보 요소 31번의 역할이 'tab'인지 표시 | 1이면 tab 역할, 0이면 다른 역할 |
| 637 | candidate | 31 | visible | 후보 요소 31번의 후보 요소가 화면에 보이는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 638 | candidate | 31 | enabled | 후보 요소 31번의 후보 요소가 활성화되어 조작 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 639 | candidate | 31 | clickable | 후보 요소 31번의 후보 요소가 클릭 가능한지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 640 | candidate | 31 | visibility | 후보 요소 31번의 후보 요소의 가시성 정도 | 모델 입력을 위해 0~1 범위로 변환한 수치 |
| 641 | candidate | 31 | bbox_x | 후보 요소 31번의 후보 요소 왼쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 642 | candidate | 31 | bbox_y | 후보 요소 31번의 후보 요소 위쪽 시작 위치 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 643 | candidate | 31 | bbox_width | 후보 요소 31번의 후보 요소의 너비 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 644 | candidate | 31 | bbox_height | 후보 요소 31번의 후보 요소의 높이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 645 | candidate | 31 | has_text | 후보 요소 31번의 후보 요소에 읽을 수 있는 텍스트가 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 646 | candidate | 31 | text_length | 후보 요소 31번의 후보 요소 텍스트의 길이 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 647 | candidate | 31 | has_data_bug_id | 후보 요소 31번의 오류 정답 식별자인 data-bug-id 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 648 | candidate | 31 | has_data_testid | 후보 요소 31번의 테스트용 식별자인 data-testid 보유 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 649 | runtime | - | url_changed | 직전 행위 이후 URL이 변경됐는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 650 | runtime | - | last_action_error | 직전 행위 실행 중 오류가 발생했는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 651 | runtime | - | elapsed_time | 관측 또는 실행 후 경과 시간 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 652 | layout | - | layout_overlap_count | 서로 겹친 것으로 관측된 화면 요소 수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 653 | infra | - | port_open | 대상 서버 포트가 열려 있는지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 654 | infra | - | health_check_ok | 서버 상태 확인 요청이 정상인지 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 655 | infra | - | response_status | 서버가 반환한 HTTP 상태 코드 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 656 | infra | - | response_latency_ms | 서버 응답 지연시간(ms) | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 657 | infra | - | timeout_occurred | 요청 시간 초과 발생 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 658 | infra | - | server_5xx_count | 서버 오류(HTTP 5xx) 발생 횟수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 659 | infra | - | server_4xx_count | 클라이언트 오류(HTTP 4xx) 발생 횟수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 660 | infra | - | server_log_exception_count | 서버 로그에서 발견한 예외 횟수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 661 | infra | - | process_alive | 대상 서버 프로세스 실행 여부 | 1이면 해당 조건이 참, 0이면 거짓 |
| 662 | infra | - | cpu_usage_percent | 대상 서버 CPU 사용률 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 663 | infra | - | memory_usage_mb | 대상 서버 메모리 사용량(MB) | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 664 | history | - | step_index | 현재 에피소드에서 진행된 Tick 번호 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 665 | history | - | no_change_steps | 행위 후 상태 변화가 없었던 연속 횟수 | 원본 수치를 정해진 최대 기준으로 나누고 0~1 범위로 제한한 값 |
| 666 | history | - | previous_action_hash | 직전 행위 종류를 숫자로 변환한 값 | 모델 입력을 위해 0~1 범위로 변환한 수치 |

---

재현 명령:

```powershell
cd C:\Users\USER\Desktop\JWAS\Browsergym_AI
./.venv/Scripts/python.exe tools/export_professor_evidence.py
```
