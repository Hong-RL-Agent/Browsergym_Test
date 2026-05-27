\# Browsergym\_Test



J.A.W.S BrowserGym 기반 PPO 웹 GUI 오류 탐지 및 자동 학습/평가 프로젝트입니다.



\## 1. 프로젝트 개요



이 프로젝트는 BrowserGym + Playwright 환경에서 PPO 강화학습 에이전트를 실행하여 웹사이트를 자동 탐색하고, Action 전후 상태 변화를 비교해 GUI/Runtime 오류 후보를 탐지합니다.



탐지 흐름은 다음과 같습니다.



```text

Action 전 상태 s\_t 수집

→ Action 실행

→ Action 후 상태 s\_t+1 수집

→ 전후 변화 비교

→ anomaly 생성

→ bug\_catalog와 매칭

→ matched\_bug\_ids / missed\_bug\_ids / exploratory\_anomalies 생성

```



\## 2. 탐지 대상 오류 유형



\- `button-no-response`: 버튼 클릭 후 화면 변화나 피드백 없음

\- `form-no-feedback`: 폼 제출 후 성공/실패 피드백 없음

\- `layout-overlap`: UI 요소 간 겹침

\- `layout-overflow`: 요소가 viewport 밖으로 넘침

\- `duplicated-rendering`: 동일 UI가 비정상적으로 중복 출력

\- `network-error`: API 요청 실패 또는 5xx 응답

\- `api-ui-mismatch`: API 응답과 UI 상태 불일치

\- `api-forbidden`: 403 Forbidden 또는 권한 관련 API 실패



\## 3. 주요 폴더 구조



```text

adapters/          BrowserGym action/observation adapter

agents/            PPO agent, rollout buffer

configs/           학습/평가 설정 파일

datasets/          bug\_catalog 및 site별 정답 데이터

envs/              BrowserGym J.A.W.S 환경

models/            action space, observation encoder, actor-critic model

runners/           train/evaluate 실행 스크립트

scripts/           포트 기반 batch 자동화 스크립트

services/          anomaly detection, reward, known bug matcher

reports/           보고서 생성 템플릿

artifacts/models/  공유 PPO 모델 파일

```



\## 4. 설치 방법



\### 4.1 프로젝트 클론



```powershell

cd C:\\workspace

git clone https://github.com/Hong-RL-Agent/Browsergym\_Test.git browsergym

cd browsergym

```



\### 4.2 Git LFS 모델 파일 받기



공유 모델 파일은 Git LFS로 관리합니다.



```powershell

\& "C:\\Program Files\\Git\\cmd\\git.exe" lfs pull

```



공유 모델 경로:



```text

artifacts/models/jaws\_browsergym\_shared\_ppo.pt

```



\### 4.3 Python 가상환경 생성



```powershell

python -m venv BrowserGym\\.venv

```



\### 4.4 가상환경 실행



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\\BrowserGym\\.venv\\Scripts\\Activate.ps1

```



\### 4.5 패키지 설치



```powershell

.\\BrowserGym\\.venv\\Scripts\\python.exe -m pip install --upgrade pip

.\\BrowserGym\\.venv\\Scripts\\pip.exe install -r requirements.txt

```



\### 4.6 Playwright Chromium 설치



```powershell

.\\BrowserGym\\.venv\\Scripts\\python.exe -m playwright install chromium

```



\## 5. 단일 학습 실행



```powershell

.\\BrowserGym\\.venv\\Scripts\\python.exe runners\\train\_multisite\_browsergym\_agent.py --config configs\\training\_sites.json --total-updates 20 --episodes-per-site 1 --max-steps 25

```



\## 6. 단일 평가 실행



```powershell

.\\BrowserGym\\.venv\\Scripts\\python.exe runners\\evaluate\_multisite\_browsergym\_agent.py --config configs\\training\_sites.json --model-path artifacts\\models\\jaws\_browsergym\_shared\_ppo.pt --episodes 3 --max-steps 25

```



\## 7. 포트 범위 기반 자동 학습/평가



여러 웹사이트가 localhost 포트에서 실행 중일 때, 포트 범위 기반으로 자동 학습/평가를 실행할 수 있습니다.



예: 9220\~9229번 사이트 학습/평가



```powershell

.\\BrowserGym\\.venv\\Scripts\\python.exe scripts\\run\_all\_port\_batches.py --start-port 9220 --end-port 9229 --batch-size 10 --total-updates 5 --episodes-per-site 1 --max-steps 15 --eval-episodes 1

```



예: 9300\~9309번 사이트 학습/평가



```powershell

.\\BrowserGym\\.venv\\Scripts\\python.exe scripts\\run\_all\_port\_batches.py --start-port 9300 --end-port 9309 --batch-size 10 --total-updates 5 --episodes-per-site 1 --max-steps 15 --eval-episodes 1

```



\## 8. 자동화 파이프라인 순서



`run\_all\_port\_batches.py`는 다음 작업을 자동으로 수행합니다.



```text

1\. 포트 범위 기반 site config 생성

2\. preflight 실행

3\. active site / failed site 분리

4\. batch config 생성

5\. batch별 PPO 학습 실행

6\. batch별 평가 실행

7\. 평가 결과 merge

8\. final summary 생성

```



\## 9. 성공 여부 확인 기준



성공 로그 예시는 다음과 같습니다.



```text

generated\_sites=10

active\_sites=10

failed\_sites=0

training success=true

evaluation success=true

evaluated\_sites=10

failed\_sites=0

```



`average\_recall`, `average\_precision`이 `null`이어도 실행 실패는 아닙니다.



정답 `bug\_catalog.json`이 없는 사이트는 정답 기반 precision/recall을 계산할 수 없습니다. 이 경우 open-ended anomaly discovery 방식으로 오류 후보를 탐지합니다.



\## 10. 결과 파일 위치



예: 9220\~9229 실행 결과



```text

artifacts/evaluations/ports\_9220\_9229/batch-001-result.json

artifacts/final/ports\_9220\_9229\_summary.json

```



단, 위 실행 결과 폴더들은 GitHub에 올리지 않습니다.



\## 11. 정답 기반 평가와 실제 사이트 평가 차이



정답 `bug\_catalog.json`이 있는 웹사이트는 다음 지표를 계산할 수 있습니다.



```text

precision

recall

matched\_bug\_ids

missed\_bug\_ids

```



하지만 실제 웹사이트처럼 어떤 오류가 포함되어 있는지 모르는 경우에는 바로 정확도를 계산할 수 없습니다. 이 경우 모델이 탐지한 anomaly 후보를 QA/개발자가 검토하여 다음과 같이 분류해야 합니다.



```text

true positive

false positive

needs review

```



\## 12. GitHub에 올리지 않는 파일



다음 파일/폴더는 GitHub에 올리지 않습니다.



```text

BrowserGym/

webarena-setup/

.venv/

\_\_pycache\_\_/

artifacts/evaluations/

artifacts/final/

artifacts/preflight/

artifacts/reports/

artifacts/scans/

artifacts/training/

configs/generated/

```



공유 모델은 예외적으로 Git LFS로 관리합니다.



```text

artifacts/models/jaws\_browsergym\_shared\_ppo.pt

```



\## 13. 팀원 실행 순서 요약



```powershell

cd C:\\workspace

git clone https://github.com/Hong-RL-Agent/Browsergym\_Test.git browsergym

cd browsergym



\& "C:\\Program Files\\Git\\cmd\\git.exe" lfs pull



python -m venv BrowserGym\\.venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\\BrowserGym\\.venv\\Scripts\\Activate.ps1



.\\BrowserGym\\.venv\\Scripts\\python.exe -m pip install --upgrade pip

.\\BrowserGym\\.venv\\Scripts\\pip.exe install -r requirements.txt

.\\BrowserGym\\.venv\\Scripts\\python.exe -m playwright install chromium



.\\BrowserGym\\.venv\\Scripts\\python.exe scripts\\run\_all\_port\_batches.py --start-port 9220 --end-port 9229 --batch-size 10 --total-updates 5 --episodes-per-site 1 --max-steps 15 --eval-episodes 1

```

