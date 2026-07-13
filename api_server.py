"""FastAPI job server for J.A.W.S BrowserGym exploration."""
from __future__ import annotations
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from services.exploration_service import ExplorationService

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "models" / "jaws_browsergym_shared_ppo.pt"
JOBS_ROOT = ROOT / "artifacts" / "jobs"
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="browsergym-job")
LOCK = threading.RLock()
JOBS: dict[str, dict[str, Any]] = {}
FUTURES: dict[str, Future[Any]] = {}
app = FastAPI(title="J.A.W.S BrowserGym AI", version="2.0.0")

class SafetyPolicy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    allowDestructiveActions: bool = False
    allowedHosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "::1"])

class ExplorationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sessionId: str | None = Field(default=None, validation_alias=AliasChoices("sessionId", "session_id"))
    targetUrl: str = Field(min_length=1, validation_alias=AliasChoices("targetUrl", "target_url"))
    mode: Literal["coverage", "fault-discovery", "ab-baseline"] = "fault-discovery"
    maxSteps: int = Field(default=25, ge=1, le=200, validation_alias=AliasChoices("maxSteps", "max_steps"))
    episodes: int = Field(default=3, ge=1, le=10)
    headless: bool = True
    modelPath: str | None = Field(default=None, validation_alias=AliasChoices("modelPath", "model_path"))
    referenceGraphPath: str | None = Field(default=None, validation_alias=AliasChoices("referenceGraphPath", "reference_graph_path"))
    safetyPolicy: SafetyPolicy = Field(default_factory=SafetyPolicy, validation_alias=AliasChoices("safetyPolicy", "safety_policy"))

class LegacyPredictRequest(BaseModel):
    console_errors: int = 0
    load_time: float = 0.0
    ui_overlap_score: float = 0.0
    is_new_path: bool = False
    is_action_success: bool = True

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "modelReady": DEFAULT_MODEL.exists(), "activeJobs": sum(1 for job in JOBS.values() if job["status"] in {"queued", "running"})}

@app.post("/explorations", status_code=202)
def create_exploration(request: ExplorationRequest) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    session_id = request.sessionId or job_id
    now = _now()
    with LOCK:
        JOBS[job_id] = {"jobId": job_id, "sessionId": session_id, "status": "queued", "progress": 0, "createdAt": now, "updatedAt": now, "events": [], "result": None, "error": None}
        FUTURES[job_id] = EXECUTOR.submit(_run_job, job_id, session_id, request)
    return _public_job(JOBS[job_id])

@app.post("/explorations/start", status_code=202)
def create_exploration_from_query(
    targetUrl: str,
    sessionId: str | None = None,
    mode: Literal["coverage", "fault-discovery", "ab-baseline"] = "fault-discovery",
    maxSteps: int = 25,
    episodes: int = 3,
    headless: bool = True,
) -> dict[str, Any]:
    """Body-free integration endpoint for clients/proxies that strip POST bodies."""
    target_host = urlsplit(targetUrl).hostname
    allowed_hosts = ["localhost", "127.0.0.1", "::1"]
    if target_host:
        allowed_hosts.append(target_host)
    return create_exploration(ExplorationRequest(
        sessionId=sessionId,
        targetUrl=targetUrl,
        mode=mode,
        maxSteps=maxSteps,
        episodes=episodes,
        headless=headless,
        safetyPolicy=SafetyPolicy(
            allowDestructiveActions=False,
            allowedHosts=sorted(set(allowed_hosts)),
        ),
    ))

@app.get("/explorations/{job_id}")
def get_exploration(job_id: str) -> dict[str, Any]:
    return _public_job(_job(job_id))

@app.get("/explorations/{job_id}/events")
def get_events(job_id: str, after: int = 0) -> dict[str, Any]:
    job = _job(job_id)
    events = job["events"][max(0, after):]
    return {"jobId": job_id, "next": max(0, after) + len(events), "events": events}

@app.get("/explorations/{job_id}/result")
def get_result(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    if job["status"] != "completed": raise HTTPException(409, detail=f"job is {job['status']}")
    return job["result"]

@app.get("/explorations/{job_id}/screenshots/{file_name}")
def get_screenshot(job_id: str, file_name: str) -> FileResponse:
    _job(job_id)
    if not file_name or "/" in file_name or "\\" in file_name:
        raise HTTPException(400, detail="invalid screenshot file name")
    directory = (JOBS_ROOT / job_id / "screenshots").resolve()
    screenshot = (directory / file_name).resolve()
    if directory not in screenshot.parents or not screenshot.exists():
        raise HTTPException(404, detail="screenshot not found")
    return FileResponse(screenshot, media_type="image/png")

@app.post("/explorations/{job_id}/stop")
def stop_exploration(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    future = FUTURES.get(job_id)
    if future and future.cancel():
        _update(job_id, status="cancelled", progress=job["progress"])
    elif job["status"] == "running":
        _update(job_id, status="stop-requested")
    return _public_job(_job(job_id))

@app.post("/predict")
def legacy_predict(request: LegacyPredictRequest) -> dict[str, Any]:
    probability = min(1.0, 0.1 + 0.45 * min(1, request.console_errors) + 0.25 * min(1.0, request.ui_overlap_score) + (0.15 if not request.is_action_success else 0.0))
    return {"action": "inspect_error" if probability >= 0.5 else "continue_exploration", "defect_probability": round(probability, 4), "reward": round((1.0 if request.is_new_path else -0.1) + probability, 4), "policy_version": "legacy-compat-v1"}

def _run_job(job_id: str, session_id: str, request: ExplorationRequest) -> None:
    _update(job_id, status="running", progress=1)
    try:
        service = ExplorationService(session_id=session_id, target_url=str(request.targetUrl), model_path=str(Path(request.modelPath) if request.modelPath else DEFAULT_MODEL), output_dir=str(JOBS_ROOT / job_id), mode=request.mode, episodes=request.episodes, max_steps=request.maxSteps, headless=request.headless, allowed_hosts=request.safetyPolicy.allowedHosts, allow_destructive_actions=request.safetyPolicy.allowDestructiveActions, reference_graph_path=request.referenceGraphPath, on_event=lambda event: _event(job_id, event))
        result = service.run()
        _update(job_id, status="completed", progress=100, result=result)
    except Exception as exc:
        traceback.print_exc()
        _event(job_id, {"type": "error", "message": str(exc)})
        _update(job_id, status="failed", error=str(exc))

def _event(job_id: str, event: dict[str, Any]) -> None:
    with LOCK:
        job = JOBS[job_id]
        payload = {"index": len(job["events"]), "timestamp": _now(), **event}
        job["events"].append(payload)
        if "progress" in event: job["progress"] = int(event["progress"])
        job["updatedAt"] = _now()

def _update(job_id: str, **values: Any) -> None:
    with LOCK:
        JOBS[job_id].update(values)
        JOBS[job_id]["updatedAt"] = _now()

def _job(job_id: str) -> dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, detail="job not found")
        return job

def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"events", "result"}}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
