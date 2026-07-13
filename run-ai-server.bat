@echo off
set "ROOT=%~dp0"
if exist "%ROOT%.venv\Scripts\python.exe" (
  set "PYTHON=%ROOT%.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
cd /d "%ROOT%"
"%PYTHON%" -m uvicorn api_server:app --host 127.0.0.1 --port 8080
