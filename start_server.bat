@echo off
cd /d C:\Users\Administrator\Documents\Github\engram
set PYTHONIOENCODING=utf-8
py -3 -m uvicorn server:app --host 0.0.0.0 --port 5000 --reload
