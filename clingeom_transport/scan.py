#!/usr/bin/env python3
"""ClinGeom adapter for the public fail-closed prospective transport."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from transport import scan as base

base.ROOT = HERE
base.WATCHSET = HERE / "watchset.json"
base.SEEN = HERE / "seen.json"
base.INBOX = HERE / "inbox"

if __name__ == "__main__":
    try:
        raise SystemExit(base.main())
    except Exception as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise
