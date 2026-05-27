# J.A.W.S BrowserGym PPO Web Error Detection

> **BrowserGym + Playwright + PPO 기반 웹 GUI 오류 자동 탐지 및 학습/평가 자동화 프로젝트**

---

## 1. 프로젝트 개요

이 프로젝트는 **BrowserGym + Playwright 환경**에서 **PPO 강화학습 에이전트**를 실행하여 웹사이트를 자동 탐색하고,  
**Action 전후 상태 변화**를 비교해 GUI/Runtime 오류 후보를 탐지하는 시스템입니다.

### 핵심 탐지 흐름

```text
Action 전 상태 s_t 수집
→ Action 실행
→ Action 후 상태 s_t+1 수집
→ 전후 변화 비교
→ Anomaly 생성
→ bug_catalog와 매칭
→ matched_bug_ids / missed_bug_ids / exploratory_anomalies 생성
```

---

## 2. 주요 기능

- **BrowserGym 기반 웹사이트 자동 탐색**
- **Playwright 기반 브라우저 실행**
- **PPO Actor-Critic 모델 기반 Action 선택**
- **Action 전후 상태 비교 기반 Anomaly Detection**
- **Known Bug Catalog 기반 정답 매칭**
- **포트 범위 기반 Batch 학습/평가 자동화**
- **Git LFS 기반 공유 모델 관리**
- **학습/평가 결과 및 보고서 생성 연동**

---

## 3. 탐지 대상 오류 유형

| 오류 유형 | 설명 |
|---|---|
| `button-no-response` | 버튼 클릭 후 화면 변화나 피드백이 없음 |
| `form-no-feedback` | 폼 제출 후 성공/실패/검증 메시지가 없음 |
| `layout-overlap` | UI 요소 간 비정상적인 겹침 발생 |
| `layout-overflow` | 요소가 viewport 밖으로 넘침 |
| `duplicated-rendering` | 동일 UI가 비정상적으로 중복 출력됨 |
| `network-error` | API 요청 실패 또는 5xx 응답 발생 |
| `api-ui-mismatch` | API 응답과 UI 상태가 일치하지 않음 |
| `api-forbidden` | 403 Forbidden 또는 권한 관련 API 실패 |

---

## 4. 주요 폴더 구조

```text
adapters/          BrowserGym action/observation adapter
agents/            PPO agent, rollout buffer
configs/           학습/평가 설정 파일
datasets/          bug_catalog 및 site별 정답 데이터
envs/              BrowserGym J.A.W.S 환경
models/            action space, observation encoder, actor-critic model
runners/           train/evaluate 실행 스크립트
scripts/           포트 기반 batch 자동화 스크립트
services/          anomaly detection, reward, known bug matcher
reports/           보고서 생성 템플릿
artifacts/models/  공유 PPO 모델 파일
```

---

## 5. 설치 방법

### 5.1 프로젝트 클론

```powershell
cd C:\workspace
git clone https://github.com/Hong-RL-Agent/Browsergym_Test.git browsergym
cd browsergym
```

### 5.2 Git LFS 모델 파일 받기

공유 PPO 모델 파일은 **Git LFS**로 관리합니다.

```powershell
& "C:\Program Files\Git\cmd\git.exe" lfs pull
```

공유 모델 경로:

```text
artifacts/models/jaws_browsergym_shared_ppo.pt
```

### 5.3 Python 가상환경 생성

```powershell
python -m venv BrowserGym\.venv
```

### 5.4 가상환경 실행

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\BrowserGym\.venv\Scripts\Activate.ps1
```

### 5.5 패키지 설치

```powershell
.\BrowserGym\.venv\Scripts\python.exe -m pip install --upgrade pip
.\BrowserGym\.venv\Scripts\pip.exe install -r requirements.txt
```

### 5.6 Playwright Chromium 설치

```powershell
.\BrowserGym\.venv\Scripts\python.exe -m playwright install chromium
```

---

## 6. 단일 학습 실행

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\train_multisite_browsergym_agent.py --config configs\training_sites.json --total-updates 20 --episodes-per-site 1 --max-steps 25
```

---

## 7. 단일 평가 실행

```powershell
.\BrowserGym\.venv\Scripts\python.exe runners\evaluate_multisite_browsergym_agent.py --config configs\training_sites.json --model-path artifacts\models\jaws_browsergym_shared_ppo.pt --episodes 3 --max-steps 25
```

---

## 8. 포트 범위 기반 자동 학습/평가

여러 웹사이트가 `localhost` 포트에서 실행 중일 때, 포트 범위 기반으로 자동 학습/평가를 실행할 수 있습니다.

### 9220~9229번 사이트 학습/평가

```powershell
.\BrowserGym\.venv\Scripts\python.exe scripts\run_all_port_batches.py --start-port 9220 --end-port 9229 --batch-size 10 --total-updates 5 --episodes-per-site 1 --max-steps 15 --eval-episodes 1
```

### 9300~9309번 사이트 학습/평가

```powershell
.\BrowserGym\.venv\Scripts\python.exe scripts\run_all_port_batches.py --start-port 9300 --end-port 9309 --batch-size 10 --total-updates 5 --episodes-per-site 1 --max-steps 15 --eval-episodes 1
```

---

## 9. 자동화 파이프라인 순서

`run_all_port_batches.py`는 다음 작업을 자동으로 수행합니다.

```text
1. 포트 범위 기반 site config 생성
2. preflight 실행
3. active site / failed site 분리
4. batch config 생성
5. batch별 PPO 학습 실행
6. batch별 평가 실행
7. 평가 결과 merge
8. final summary 생성
```

---

## 10. 성공 여부 확인 기준

성공 로그 예시는 다음과 같습니다.

```text
generated_sites=10
active_sites=10
failed_sites=0
training success=true
evaluation success=true
evaluated_sites=10
failed_sites=0
```

> `average_recall`, `average_precision`이 `null`이어도 실행 실패는 아닙니다.  
> 정답 `bug_catalog.json`이 없는 사이트는 정답 기반 precision/recall을 계산할 수 없습니다.  
> 이 경우 **open-ended anomaly discovery** 방식으로 오류 후보를 탐지합니다.

---

## 11. 결과 파일 위치

예: 9220~9229 실행 결과

```text
artifacts/evaluations/ports_9220_9229/batch-001-result.json
artifacts/final/ports_9220_9229_summary.json
```

단, 위 실행 결과 폴더들은 GitHub에 올리지 않습니다.

---

## 12. 정답 기반 평가와 실제 사이트 평가 차이

정답 `bug_catalog.json`이 있는 웹사이트는 다음 지표를 계산할 수 있습니다.

```text
precision
recall
matched_bug_ids
missed_bug_ids
```

하지만 실제 웹사이트처럼 어떤 오류가 포함되어 있는지 모르는 경우에는 바로 정확도를 계산할 수 없습니다.

이 경우 모델이 탐지한 anomaly 후보를 QA/개발자가 검토하여 다음과 같이 분류해야 합니다.

```text
true positive
false positive
needs review
```

---

## 13. GitHub에 올리지 않는 파일

다음 파일/폴더는 GitHub에 올리지 않습니다.

```text
BrowserGym/
webarena-setup/
.venv/
__pycache__/
artifacts/evaluations/
artifacts/final/
artifacts/preflight/
artifacts/reports/
artifacts/scans/
artifacts/training/
configs/generated/
```

공유 모델은 예외적으로 **Git LFS**로 관리합니다.

```text
artifacts/models/jaws_browsergym_shared_ppo.pt
```

---

## 14. 팀원 실행 순서 요약

```powershell
cd C:\workspace
git clone https://github.com/Hong-RL-Agent/Browsergym_Test.git browsergym
cd browsergym

& "C:\Program Files\Git\cmd\git.exe" lfs pull

python -m venv BrowserGym\.venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\BrowserGym\.venv\Scripts\Activate.ps1

.\BrowserGym\.venv\Scripts\python.exe -m pip install --upgrade pip
.\BrowserGym\.venv\Scripts\pip.exe install -r requirements.txt
.\BrowserGym\.venv\Scripts\python.exe -m playwright install chromium

.\BrowserGym\.venv\Scripts\python.exe scripts\run_all_port_batches.py --start-port 9220 --end-port 9229 --batch-size 10 --total-updates 5 --episodes-per-site 1 --max-steps 15 --eval-episodes 1
```
