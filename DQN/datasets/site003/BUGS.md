# BUGS.md - site003

## 의도된 GUI 오류 목록

---

### bug01

| 항목 | 내용 |
|------|------|
| **bugId** | site003-bug01 |
| **type** | button-no-response |
| **화면 위치** | 대시보드 상단 우측 "운동 추가" 버튼 |
| **관련 파일** | `src/components/Dashboard.jsx` |
| **data-bug-id selector** | `[data-bug-id="site003-bug01"]` |
| **사용자가 경험하는 증상** | 사용자가 새 운동을 기록하기 위해 "운동 추가" 버튼을 클릭해도 폼이 열리거나 페이지가 이동하는 등의 아무런 반응이 없음. |
| **코드상 의도된 원인** | `<button>`의 `onClick` 이벤트 핸들러가 `() => {}` 빈 함수로 연결되어 있어서 기능이 동작하지 않음. |
| **PPO 에이전트 기대 행동** | 해당 버튼 클릭 후 화면 내 신규 모달 생성 여부, DOM 트리의 의미 있는 변화나 라우팅이 일어나지 않는 상태(no-response)를 탐지. |

---

### bug02

| 항목 | 내용 |
|------|------|
| **bugId** | site003-bug02 |
| **type** | component-rendering |
| **화면 위치** | 대시보드 메인 영역 좌측 "주간 칼로리 소모량" 통계 카드 |
| **관련 파일** | `src/components/WeeklyStats.jsx` |
| **data-bug-id selector** | `[data-bug-id="site003-bug02"]` |
| **사용자가 경험하는 증상** | 서버로부터 주간 통계 데이터가 정상적으로 전달되었음에도 불구하고, 차트가 그려지지 않고 "이번 주 기록된 데이터가 없습니다."라는 빈 상태(Empty State) 화면이 지속적으로 노출됨. |
| **코드상 의도된 원인** | 데이터 렌더링 조건문에 `stats.length > 0` 대신 강제로 `true` 또는 `!stats.length` 로직 무시 조건을 넣어 항상 데이터가 없는 상태 UI가 렌더링되게 하였음. |
| **PPO 에이전트 기대 행동** | API 응답은 200 OK와 함께 데이터 배열이 내려왔으나, 렌더링된 컴포넌트(`[data-bug-id="site003-bug02"]`)에는 차트가 없고 빈 데이터를 뜻하는 문구나 아이콘만 있는 모순 상태 탐지. |

---

### bug03

| 항목 | 내용 |
|------|------|
| **bugId** | site003-bug03 |
| **type** | css-layout |
| **화면 위치** | "주간 칼로리 소모량" 카드 내 차트 영역 전체 |
| **관련 파일** | `src/styles/main.css`, `src/components/WeeklyStats.jsx` |
| **data-bug-id selector** | `[data-bug-id="site003-bug03"]` |
| **사용자가 경험하는 증상** | 막대 그래프를 포함하는 차트 영역 컨테이너의 크기가 부모 카드 범위를 벗어나 우측으로 삐져나와 UI가 깨져 보임. |
| **코드상 의도된 원인** | `.chart-container`의 너비를 `120%`로 지정하고, 부모 컴포넌트에 `overflow: hidden`이 적용되지 않아 화면 밖으로 넘치게 설정함. |
| **PPO 에이전트 기대 행동** | `[data-bug-id="site003-bug03"]` 요소의 BoundingBox가 부모 컨테이너(`div.panel`)의 범위를 물리적으로 벗어났는지 계산하여 레이아웃 깨짐 버그로 탐지. |
