#!/usr/bin/env python
"""Simple script to run uvicorn with detailed logging."""
import uvicorn
import sys

if __name__ == '__main__':
    print("[BACKEND] Starting uvicorn...")
    try:
        uvicorn.run(
            "server.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"[BACKEND] Error: {e}", file=sys.stderr)
        sys.exit(1)
