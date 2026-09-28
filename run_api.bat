@echo off
cd /d %~dp0
call venv\Scripts\activate
uvicorn api:app --host 0.0.0.0 --port 8000
