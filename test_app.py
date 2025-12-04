#!/usr/bin/env python
"""Test app startup without uvicorn."""
import sys
sys.path.insert(0, '.')

try:
    print("[TEST] Importing FastAPI app...")
    from server.app import app, JOBS
    print("[TEST] App imported successfully")
    print(f"[TEST] Routes: {len(app.routes)}")
    print("[TEST] All OK")
except Exception as e:
    print(f"[TEST] Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
