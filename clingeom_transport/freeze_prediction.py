#!/usr/bin/env python3
"""ClinGeom adapter for tamper-evident prediction freezing."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from transport import freeze_prediction as base

base.ROOT = HERE
base.SEEN = HERE / "seen.json"
base.PRED = HERE / "predictions"

if __name__ == "__main__":
    base.main()
