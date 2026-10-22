# RAWD — Reinforcement-learning Autonomous Web Defect Detector

> **BrowserGym + Playwright + PPO·GRU·DQN 기반 자율 웹 오류 탐지 및 검증 프로젝트**

RAWD는 강화학습 에이전트가 웹사이트를 자율적으로 탐색하고, 사용자 행동 전후의 상태 변화를 분석하여 **오류 후보를 발견하고 재현·검증하는 웹 QA 자동화 시스템**입니다.

- **Repository:** [Hong-RL-Agent/Browsergym_Test](https://github.com/Hong-RL-Agent/Browsergym_Test)
- **Development Branch:** `RL_AI_RAWD`
- **Core:** Python · BrowserGym · Playwright · PyTorch · PPO · GRU · DQN

---

## 1. 프로젝트 소개

웹 애플리케이션의 오류는 특정 버튼 클릭, 입력, 화면 이동 또는 연속된 사용자 행동 이후에 나타나기도 합니다. 기존 수동 테스트와 시나리오 기반 자동화 테스트는 테스트 경로를 설계하고 반복 검증하는 데 시간이 필요합니다.

RAWD는 이러한 QA 작업의 부담을 줄이기 위해 **탐색 → 오류 후보 탐지 → 재현·검증 → 근거 기록** 과정을 자동화하는 것을 목표로 합니다. PPO가 다양한 화면과 기능을 탐색하고, DQN 및 규칙 기반 검증 모듈이 발견된 후보의 재현과 확인을 지원하도록 설계했습니다.

## 2. 주요 기능

- **자율 웹 탐색:** PPO 기반으로 페이지와 상호작용 요소를 탐색하고 유효 행동을 선택합니다.
- **상태 관찰 및 기억:** DOM, 화면 레이아웃, 후보 요소, Console·Network·Runtime 정보와 행동 이력을 활용합니다. GRU 기반 순차 정책을 실험합니다.
- **오류 후보 탐지:** 행동 전후 Observation을 비교해 기능·화면·데이터·실행 오류의 징후를 찾습니다.
- **오류 재현·검증:** 오류 특성에 따라 행동 재실행, 시퀀스 재현, 상태 재관찰 등 검증 전략을 선택합니다.
- **실험 및 진단:** PPO 학습, DQN 계약 검증, 보상 파이프라인 및 Hybrid 통합 진단 스크립트를 제공합니다.

## 3. 시스템 아키텍처

| 구성 요소 | 역할 |
|---|---|
| **BrowserGym + Playwright** | 브라우저 실행, 사용자 행동 수행, 웹 상태 수집 |
| **Observation / Action Adapter** | DOM·레이아웃·실행 신호를 모델 입력과 행동 공간에 연결 |
| **PPO (Global Explorer)** | 다양한 페이지와 사용자 행동 경로 탐색 |
| **GRU** | 이전 상태와 행동 이력을 순차 정책에 반영 |
| **Anomaly Detector** | 행동 전후 상태를 비교해 오류 후보 생성 |
| **Hybrid Controller** | 탐색과 국소 검증 사이의 제어 흐름 관리 |
| **DQN (Local Specialist)** | 후보 주변의 국소 행동 선택 및 재현 실험 |
| **Strategy Router / Confirmation** | 적합한 검증 전략 선택 및 근거 확인 |

> PPO·GRU 학습, Detector, DQN 및 Hybrid의 개별 코드와 진단 경로가 존재합니다. 모든 사이트에서 완성된 통합 시스템의 성능이 검증되었다는 의미는 아닙니다.

## 4. 시스템 동작 흐름

```text
대상 사이트(URL / 로컬 실행환경 지정)
                ↓
       BrowserGym + Playwright
                ↓
        Observation 수집
                ↓
        PPO·GRU 자율 탐색
                ↓
         사용자 행동 실행
                ↓
       행동 전후 상태 비교
                ↓
         Anomaly Detector
                ↓
    ┌─ 이상 징후 없음 → PPO 탐색 지속
    │
    └─ 오류 후보 발견
                ↓
         Hybrid Controller
                ↓
    DQN / Rule-based / 상태 재관찰
                ↓
          검증 결과 기록
                ↓
          PPO 탐색 복귀
```

탐지 결과는 우선 **오류 후보(Anomaly)** 로 다룹니다. 실제 결함 여부는 재현, 추가 근거 및 필요시 사람의 검토로 판단합니다. 정답(`bug_catalog`)이 있는 테스트 사이트의 사후 평가는 정답이 없는 일반 웹 탐색과 구분합니다.

## 5. 탐지 대상 오류

| 오류 범주 | 대표 사례 |
|---|---|
| 사용자 상호작용 | 버튼 무응답, 폼 제출 후 피드백 없음 |
| 화면 이동·라우팅 | 잘못된 화면 전환, 상세 페이지 미열림 |
| UI·레이아웃·접근성 | 요소 겹침, 화면 잘림, 중복 렌더링, 접근성 이름 누락 |
| 입력값 검증 | 필수 입력 검증 누락, 잘못된 입력 허용 |
| 검색·필터·정렬 | 검색 결과 불일치, 필터 미적용, 정렬 오류 |
| 데이터·비즈니스 정합성 | 장바구니 수량·합계 불일치, 저장 후 값 미반영 |
| Runtime·Network | JavaScript 예외, API 오류, 요청 실패, 시간 초과 |
| 상태·행동 시퀀스 | 화면 전환 후 상태 유실, 연속 행동 결과 불일치 |

탐지 가능 여부는 사이트와 수집 가능한 증거에 따라 달라집니다. 백엔드 내부 결함이나 보안 취약점은 브라우저 관찰만으로 확정할 수 없으며, 별도 검증 수단이 필요할 수 있습니다.

## 6. 프로젝트 폴더 구조

```text
browsergym/
├── README.md
├── .gitignore
├── DQN/
│   ├── adapters/                 # BrowserGym Observation 연결
│   ├── agents/                   # Recurrent Rollout Buffer
│   ├── envs/                     # BrowserGym 실행 환경
│   ├── models/                   # Observation, PPO·GRU, Action·DQN 계약
│   ├── services/                 # Anomaly Detector, Reward
│   ├── dqn_error_discovery/
│   │   ├── agents/               # 오류 재현·검증
│   │   ├── hybrid/               # Controller, Strategy Router, Replayer
│   │   ├── runners/              # PPO–DQN Hybrid 실행 경로
│   │   └── v3/                   # 학습·검증·진단 도구
│   └── scripts/                  # PPO 학습, DQN·Hybrid 검증, 진단
├── configs/
│   ├── ppo_full_training_v1.json
│   ├── step7_full_site_split.json
│   └── rl_errorsite_*.example.json
├── runners/                      # 기존 PPO 학습·평가 스크립트
└── scripts/                      # 기존 Batch 실행 도구
```

주요 디렉터리만 표시한 구조입니다. 기존 PPO 코드와 신규 RAWD 모듈이 함께 포함되어 있습니다.

## 7. 설치 및 실행 방법

아래 명령어는 **Windows PowerShell**, `RL_AI_RAWD` 브랜치 기준입니다.

### 7.1 저장소 클론

```powershell
git clone -b RL_AI_RAWD --single-branch https://github.com/Hong-RL-Agent/Browsergym_Test.git browsergym
cd browsergym
```

### 7.2 Python 환경 준비

```powershell
python -m venv .venv312
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch gymnasium browsergym-core playwright
python -m playwright install chromium
```

> 위 설치 명령은 핵심 라이브러리의 예시입니다. 현재 브랜치에 전체 의존성 버전을 고정한 `requirements.txt`가 확인되지 않아, 개별 실행 스크립트에 필요한 추가 패키지와 버전 정합성은 별도로 확인해야 합니다.

### 7.3 오류 사이트 및 로컬 설정 준비

실제 브라우저 탐색·학습에는 별도로 실행 중인 대상 웹사이트가 필요합니다. 테스트 사이트는 [RL_Errorsite](https://github.com/Hong-RL-Agent/RL_Errorsite) 저장소를 참고합니다.

```powershell
Copy-Item configs\rl_errorsite_runtime.example.json configs\rl_errorsite_runtime.json
Copy-Item configs\rl_errorsite_training_sites.example.json configs\rl_errorsite_training_sites.json
Copy-Item configs\rl_errorsite_full_training_sites.example.json configs\rl_errorsite_full_training_sites.json
```

복사한 JSON에서 `source_path`, `runtime_root`, URL 및 포트를 실제 실행환경에 맞게 수정하고 대상 서버를 실행합니다. `.example.json` 안의 경로는 실행 가능한 실제 경로가 아닌 예시입니다.

### 7.4 기본 진단 실행

```powershell
New-Item -ItemType Directory -Force DQN\artifacts\training_v3\step7_fresh_training_final | Out-Null

python DQN\scripts\run_step7_dqn_contract_smoke.py
python DQN\scripts\run_detector_routing_regression.py
python DQN\scripts\run_ppo_reward_pipeline_regression.py
```

위 명령은 DQN 입력·행동 계약, Detector 라우팅, PPO 보상 규칙 등을 점검하는 **개별 진단**입니다. 전체 웹사이트 탐색이나 Hybrid 성능 평가를 수행하는 명령은 아닙니다.

### 7.5 PPO 학습 예시 (추가 데이터 필요)

```powershell
python DQN\scripts\run_ppo_full_training_v1.py `
  --architecture gru `
  --seed 42 `
  --contract configs\ppo_full_training_v1.json `
  --split configs\step7_full_site_split.json `
  --sites configs\rl_errorsite_full_training_sites.json
```

`--architecture`는 `gru` 또는 `history_mlp`를 선택할 수 있습니다. 전체 학습 실행에는 사이트 서버와 설정 파일 외에 다음 로컬 클러스터 메타데이터가 필요합니다.

```text
DQN/artifacts/training_v3/step7_complete/structural_cluster_manifest.json
```

이 파일과 학습 결과·체크포인트는 GitHub 업로드 대상에서 제외되어 있으므로, 새 환경에서 즉시 전체 학습이나 체크포인트 기반 Hybrid 검증을 실행할 수는 없습니다.

## 8. 개발 현황 및 참고 사항

- **구현·진단 중:** Observation-v2, PPO·GRU 학습, 보상 설계, Anomaly Detector, DQN 계약, 오류 재현 전략, Hybrid Controller 및 시나리오 기반 통합 검증
- **추가 검증 과제:** PPO 탐색 Coverage, GRU 기억 효과, DQN Confirmation 성능, Hybrid 전환·복귀의 일반화 성능
- **결과 해석:** Anomaly 후보 수와 실제 확인된 결함 수를 구분합니다. 정답이 없는 사이트에서는 Precision/Recall을 바로 계산할 수 없습니다.
- **로컬 파일 관리:** `artifacts/`, `checkpoints/`, `*.pt`, `*.pth`, `.env`, 로컬 실행환경 JSON 등은 `.gitignore`로 제외합니다. 최신 모델 가중치는 저장소 클론만으로 제공되지 않습니다.
- **실험 재현성:** 학습·평가 시 사용한 설정, 사이트 분할, 시드, 모델 체크포인트 및 실행환경을 함께 관리해야 합니다.

---

**Repository:** https://github.com/Hong-RL-Agent/Browsergym_Test
**Branch:** `RL_AI_RAWD`