# J.A.W.S 667차원 및 탐지 근거 자료

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

빈 관측값을 실제 인코더에 입력한 검증 결과는 `shape=(667,)`, dtype은 `float32`이다. 따라서 667은 임의 숫자가 아니라 인코더가 생성하고 모델이 받는 고정 입력 길이다. 후보가 32개보다 적으면 남은 후보 슬롯은 0으로 패딩되고, 32개를 넘으면 앞의 32개만 사용한다. 각 인덱스의 정확한 의미와 인코딩 규칙은 `vector_dimensions.csv`, 그룹 합계는 `vector_dimension_summary.csv`에 있다.

## 2. 오류 매핑

`datasets/site*/bug_catalog.json`을 정답 목록으로, `artifacts/browsergym/site*/detected_bugs.json`을 탐지 목록으로 사용했다. `matched_bug_id`가 동일하면 TP, 정답이 탐지되지 않으면 FN, 정답 ID로 연결되지 않은 탐지는 `FP_REVIEW`로 표시했다. 현재 집계는 TP 8건, FN 15건, FP_REVIEW 104건이다. FP_REVIEW는 자동으로 오탐 확정한 값이 아니라 사람 검토가 필요한 탐지다.

## 3. 행위 목록

`action_list.csv`는 실제 `ActionSpace`를 전개한 전체 608개 이산 행위를 담는다. 19개 행위 유형 × 32개 후보 슬롯으로 구성된다. 요소 대상 행위는 `click_element`, `fill_input`, `press_enter`이며 나머지는 페이지/환경 단위 행위다. 실행 시점에는 action mask가 현재 관측에서 가능한 행위만 활성화한다.

## 4. Risk Score 근거

현재 구현(`services/risk_scoring_service.py`, 정책 `risk-v3-general-service`)은 보안 취약점 CVSS 점수가 아니라 일반 서비스 오류의 우선순위 점수다. 핵심 기능 영향 35점, 데이터 영향 25점, 영향 범위 15점, 복구 난이도 15점, 재현 빈도 10점으로 총 100점이다. 신뢰도는 재현율 40%, 증거 완전성 40%, 원 탐지 신뢰도 20%로 별도 산출하여 위험 영향과 관측 신뢰도를 섞지 않는다.

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
