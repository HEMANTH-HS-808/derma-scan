#!/usr/bin/env python
"""Run uvicorn and log all output."""
import subprocess
import sys

cmd = [sys.executable, '-m', 'uvicorn', 'server.app:app', '--host', '0.0.0.0', '--port', '8000', '--log-level', 'debug']

print(f"[RUN] Command: {' '.join(cmd)}")
print(f"[RUN] CWD: /d/derma/derma scan/derma-scan")

with open('server/logs/uvicorn.log', 'w') as logfile:
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, text=True)
    print(f"[RUN] Started PID {proc.pid}")
    proc.wait()
    print(f"[RUN] Exited with code {proc.returncode}")

print("[RUN] Logs written to server/logs/uvicorn.log")
