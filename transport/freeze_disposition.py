#!/usr/bin/env python3
"""Freeze a tamper-evident exclusion/disposition for a first-seen public case."""

from __future__ import annotations
import argparse, datetime as dt, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEN = ROOT / "seen.json"
OUT = ROOT / "dispositions"
REQUIRED = ("case_id", "independence_class", "disposition", "reason")
HASH_FIELDS = (
    "case_id", "source_surface_sha256", "first_frozen_at",
    "prediction_deadline_at", "disposition_frozen_at",
    "independence_class", "disposition", "reason",
)

def parse(v):
    return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(dt.timezone.utc)

def fmt(v):
    return v.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def digest(doc):
    payload = {k: doc[k] for k in HASH_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

def safe(v):
    return v.replace("/", "__").replace("#", "__")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("draft", type=Path)
    p.add_argument("--at")
    a = p.parse_args()
    draft = json.loads(a.draft.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if not draft.get(k)]
    if missing:
        raise ValueError(f"missing disposition fields: {missing}")

    state = json.loads(SEEN.read_text(encoding="utf-8"))
    entries = {x.get("id"): x for x in state.get("seen", []) if isinstance(x, dict)}
    case = draft["case_id"]
    if case not in entries:
        raise ValueError(f"{case}: no public first-seen record")
    source = entries[case]
    if source.get("status") != "candidate-frozen-awaiting-prediction":
        raise ValueError(f"{case}: not awaiting prediction/disposition")

    now = parse(a.at) if a.at else dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if now > parse(source["prediction_deadline_at"]):
        raise ValueError(f"{case}: disposition deadline elapsed; record administrative expiry instead")
    frozen = {
        "schema_version": 1,
        "case_id": case,
        "source_surface_sha256": source["surface_sha256"],
        "first_frozen_at": source["first_frozen_at"],
        "prediction_deadline_at": source["prediction_deadline_at"],
        "disposition_frozen_at": fmt(now),
        "disposition_within_sla": now <= parse(source["prediction_deadline_at"]),
        "independence_class": draft["independence_class"],
        "disposition": draft["disposition"],
        "reason": draft["reason"],
        "information_barrier": (
            "Disposition uses only the already-frozen original issue title/body. "
            "Comments, linked workflow logs, linked PR diffs, later commits, "
            "maintainer diagnoses, and fixes are forbidden before this freeze."
        ),
        "claim_boundary": (
            "An exclusion/disposition record is not a prospective mechanism "
            "success, external adoption, or L8/L9 credit."
        ),
    }
    frozen["disposition_sha256"] = digest(frozen)

    OUT.mkdir(exist_ok=True)
    out = OUT / f"{safe(case)}.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for x in state["seen"]:
        if x.get("id") == case:
            x["status"] = frozen["disposition"]
            x["independence_class"] = frozen["independence_class"]
            x["disposition_frozen_at"] = frozen["disposition_frozen_at"]
            x["disposition_sha256"] = frozen["disposition_sha256"]
            x["disposition_source"] = str(out.relative_to(ROOT))
            break
    SEEN.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(frozen["disposition_sha256"])

if __name__ == "__main__":
    main()
