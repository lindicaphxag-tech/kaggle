#!/usr/bin/env python3
"""ClinGeom adapter for digest-authorized post-freeze resolution transport."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from transport import collect_resolution as base

base.ROOT = HERE
base.PRED = HERE / "predictions"
base.RES = HERE / "resolution"

if __name__ == "__main__":
    try:
        base.main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise
