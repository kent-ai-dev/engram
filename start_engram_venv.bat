@echo off
cd /d C:\Users\Administrator\Documents\Github\engram
call .venv\Scripts\activate.bat
python -m uvicorn server:app --host 0.0.0.0 --port 5000
