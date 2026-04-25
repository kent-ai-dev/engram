"""
Self-contained Engram server launcher.
Run with: C:\...\engram\.venv\Scripts\python.exe run_engram_server.py
"""
import sys
import os

# Ensure we're running in the right dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Force the venv python for any subprocess spawning
import uvicorn
from server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        workers=1,
        loop="asyncio",
        # This ensures no subprocess spawn — single in-process server
    )
