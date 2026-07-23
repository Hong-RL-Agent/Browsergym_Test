# J.A.W.S 모델 고도화 적용 보고서

작성일: 2026-07-15  
상태: 1차 적용 완료, 추가 학습 필요

## 목적

현재 PPO/Rainbow DQN은 평균 reward가 올라가도 특정 오류 경로를 놓치는 문제가 있었다. 특히 `site001-bug03`은 모바일 viewport와 레이아웃 검사 경로를 거쳐야 하므로 단일 정책이 모든 action을 직접 선택하는 구조에서 누락되기 쉬웠다.

이번 고도화는 PPO 대신 Rainbow DQN을 하위 정책으로 사용하고 다음 세 계층을 결합했다.

```text
상위 목표 정책
  → bug_reproduction / mobile_layout_inspection /
    security_surface_inspection / recovery / novel_state_exploration
        ↓
하위 Rainbow DQN + 기존 action mask
        ↓
실제 BrowserGym action
        ↓
failure taxonomy와 reward breakdown
```

## 적용 내용

### 1. 계층형 탐색 목표 정책

`_select_exploration_goal()`이 현재 viewport, 이미 발견한 bug, action count, security mode를 이용해 한 Tick의 상위 목표를 선택한다.

- `mobile_layout_inspection`: bug03 미탐지 상태에서 모바일 layout 검사
- `bug_reproduction`: known bug 재현 경로 우선
- `security_surface_inspection`: API/권한/입력 관련 보안 표면 검사
- `recovery`: 직전 action 실행 실패 후 DOM 재수집
- `novel_state_exploration`: 새로운 상태와 경로 탐색

선택된 목표는 `exploration_goal`로 transition log에 저장되며, 이후 학습 데이터에서 목표 조건 정책으로 사용할 수 있다.

### 2. 목표 조건 guided action

`mobile_layout_inspection` 목표가 선택되면 다음 조건을 강제한다.

```text
change_viewport_mobile
→ inspect_layout (최대 4회까지 우선)
→ 이후 Rainbow DQN action
```

목표가 끝나기 전에 일반 클릭/입력 action으로 이탈하지 않도록 목표 유지 조건을 강화했다.

### 3. 실패 recovery 정책

`ACTION_EXECUTION_FAILED`가 발생하면 다음 Tick에서 바로 mutation action을 반복하지 않고 `inspect_dom`을 먼저 수행한다. 이후 실패했던 action type을 유지하되 이전 candidate와 다른 visible/enabled candidate를 선택한다. recovery 시도 횟수와 원래 실패 action은 transition log에 남긴다.

### 4. 보안 reward v1 개선

- 같은 finding은 최초 확인 시에만 큰 보상 지급
- 이후 반복 Tick은 반복·무변화 패널티 적용
- reward 상한 포화 방지
- `layout-overlap`, `layout-overflow`도 보안 표면/증거 후보에 포함
- reward breakdown 검증이 `security_v1` 항목을 인식

### 5. Goal-conditioned imitation prior

transition JSONL에서 `exploration_goal → action_type` 빈도를 추출하는 도구를 추가했다. 이는 완전한 Behavioral Cloning 모델 전 단계의 검증 가능한 warm-start prior이며, 이후 성공 demonstration만 선별해 PPO 초기화에 사용할 수 있다.

```powershell
\.venv\Scripts\python.exe scripts\build_imitation_prior.py `
  artifacts\browsergym\site001\rl_transition_log.jsonl `
  --output artifacts\imitation\goal_action_prior.json
```

생성 결과: `artifacts/imitation/goal_action_prior.json`. Rainbow DQN 실행 시 모바일 레이아웃 목표에서 아직 시도하지 않은 검사 action을 우선 제안하도록 연결했다.

## 실행 결과

실행 조건:

| 항목 | 값 |
|---|---|
| Site | site001 / BookHaven |
| Algorithm | Rainbow DQN |
| Security mode | enabled |
| Seed | 42, 43, 44 일부 실험 및 seed 44 구조 검증 |
| Max steps | 12 |

### 기존 baseline

| Metric | 값 |
|---|---:|
| Recall | 0.3333 |
| Precision | 0.5000 |

### 보안 reward v1 초기 실행

| Metric | 값 |
|---|---:|
| Recall | 0.6667 |
| Precision | 1.0000 |
| Matched bugs | bug01, bug02 |
| Missed bug | bug03 |

### reward 포화 수정 후

| Metric | seed 42 | seed 43 |
|---|---:|---:|
| Recall | 0.6667 | 0.6667 |
| Precision | 1.0000 | 1.0000 |
| 주요 실패 | no-state 23, miss 12 | no-state 24, miss 4, execution 8 |

현재 결과는 precision은 안정적이지만 bug03 recall이 개선되지 않았으므로 모델 승격 기준을 충족하지 않는다.

## 발견된 병목

1. `mobile_layout_inspection` 목표를 선택해도 레이아웃 증거가 실제 finding으로 연결되지 않는 Tick이 있다.
2. `NO_STATE_CHANGE`가 여전히 많아 action 후보가 화면 상태를 충분히 바꾸지 못한다.
3. execution failure가 발생하면 selector 재선택이나 후보 교체가 아직 단순하다.
4. 3 episode는 정책 비교에 부족하므로 seed/site별 반복이 필요하다.

## 다음 고도화 단계

### A. 행동 시퀀스 정책

목표별 action sequence prior를 추가한다.

```text
mobile_layout_inspection:
viewport_mobile → inspect_layout → inspect_dom → candidate_click
security_surface_inspection:
inspect_network → inspect_console → fill_input → press_enter
```

### B. imitation learning

bug03 성공 경로를 5~10개 수집해 Behavioral Cloning으로 초기 policy를 만든 뒤 PPO fine-tuning을 수행한다.

### C. novelty/coverage encoder

URL뿐 아니라 DOM signature, action-target 조합, viewport, endpoint를 embedding해 미방문 상태 보상을 계산한다.

### D. 정책 승격 기준

- 3 seeds 평균 recall ≥ 0.67
- seed별 최저 recall ≥ 0.5
- precision ≥ 0.6
- false positive rate ≤ 10%
- execution failure rate ≤ 10%

## 재현 명령

```powershell
cd C:\Users\USER\Desktop\JWAS\Browsergym_AI
\.venv\Scripts\python.exe scripts\runners\train_browsergym_agent.py `
  --site-id site001 `
  --base-url http://127.0.0.1:9220 `
  --episodes 10 `
  --max-steps 20 `
  --headless true `
  --seed 44 `
  --security-mode `
  --algorithm rainbow-dqn
```

결과 파일:

- `artifacts/browsergym/site001/training_summary.json`
- `artifacts/browsergym/site001/rl_transition_log.jsonl`
- `artifacts/models/site001_browsergym_rainbow_dqn.pt`
- `artifacts/imitation/goal_action_prior.json`

## 결론

계층형 목표 정책, recovery 흐름, imitation action prior는 Rainbow DQN 코드와 로그에 적용되었다. 3 episode 검증에서 replay gradient update가 실제 수행되었고 precision은 1.0이었지만, bug03 누락이 반복되므로 현재 모델은 최종 승격하지 않는다.

## 다중 사이트 Fleet 실행

현재 저장소에서 발견된 90개 사이트를 순차적으로 실행·학습하는 fleet runner를 추가했다. 450개 사이트가 별도 디렉터리로 추가되면 동일한 config 생성 명령으로 확장된다. 각 사이트는 독립 server process와 artifact를 사용하므로 한 사이트 실패가 전체 fleet를 중단시키지 않는다.

```powershell
\.venv\Scripts\python.exe scripts\generate_local_sites_config.py `
  --sites-root ..\RL_Errorsite-frontend-errorsite `
  --start-port 9220 `
  --output configs\generated_local_sites.json

\.venv\Scripts\python.exe scripts\run_site_fleet_rainbow.py `
  --config configs\generated_local_sites.json `
  --episodes 3 `
  --max-steps 20 `
  --output artifacts\fleet\rainbow_summary.json
```

Fleet 실행은 사전 health/preflight를 거친 뒤 활성 사이트만 학습하는 것이 권장된다. 현재 세션에서는 90개 전체를 실행하지 않았으며, 대상 사이트 서버가 모두 준비된 환경에서 실행해야 한다.

Recovery 재학습(seed 45, 3 episodes) 결과는 recall 0.6667, precision 1.0, 평균 reward -4.245였다. 실패 분포는 no-state 14, detection-miss 12, execution-failed 10으로 recovery가 아직 충분히 작동하지 않았다. 다음에는 guided 선택이 아니라 환경 step 직후의 강제 fallback 재시도로 승격한다.

환경 레벨 fallback 적용 후(seed 46, 3 episodes)에는 execution-failed가 3건으로 감소했다. 다만 no-state가 30건으로 증가했고 recall은 0.6667로 동일했다. 따라서 recovery는 효과가 확인됐으며, 다음 병목은 상태 변화 없는 action을 사전에 차단하는 novelty/coverage 제약이다.
