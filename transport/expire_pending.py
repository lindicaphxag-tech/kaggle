#!/usr/bin/env python3
"""Fail-closed administrative expiry of unpredicted prospective cases.

This is NOT I0/I1/I2 adjudication. A missed six-hour evidence deadline must be
visible and irreversible, rather than leaving an old case in a seemingly live
prediction queue. It never reads issue comments, PRs, fixes, or resolutions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
STATUS_PENDING = "candidate-frozen-awaiting-prediction"
STATUS_EXPIRED = "deadline_missed_unadjudicated"
HASH_FIELDS = (
    "case_id",
    "source_surface_sha256",
    "first_frozen_at",
    "prediction_deadline_at",
    "expiry_recorded_at",
    "reason",
)


def parse(value: str) -> dt.datetime:
    out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if out.tzinfo is None:
        raise ValueError("expiry timestamp must include a timezone")
    return out.astimezone(dt.timezone.utc)


def fmt(value: dt.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("expiry time must include a timezone")
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest(doc: dict) -> str:
    canonical = {field: doc[field] for field in HASH_FIELDS}
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def safe(case_id: str) -> str:
    return case_id.replace("/", "__").replace("#", "__")


def expire(root: Path, now: dt.datetime) -> dict:
    """Validate all expired candidates before changing any file."""
    now = parse(fmt(now))
    state_path = root / "seen.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidates: list[tuple[dict, Path, dict]] = []

    for entry in state.get("seen", []):
        if entry.get("status") != STATUS_PENDING:
            continue
        first = parse(entry["first_frozen_at"])
        deadline = parse(entry["prediction_deadline_at"])
        if first > deadline:
            raise ValueError(f"{entry['id']}: invalid first-seen/deadline ordering")
        if now <= deadline:
            continue

        case_id = entry["id"]
        filename = safe(case_id) + ".json"
        src_relative = Path(entry["source"])
        if src_relative.is_absolute() or ".." in src_relative.parts or src_relative.parts[0] != "inbox":
            raise ValueError(f"{case_id}: unsafe first-seen source path")
        source = json.loads((root / src_relative).read_text(encoding="utf-8"))
        matches = [c for c in source.get("new_candidates", []) if c.get("id") == case_id]
        if len(matches) != 1:
            raise ValueError(f"{case_id}: first-seen source cardinality is {len(matches)}")
        if matches[0]["surface_sha256"] != entry["surface_sha256"]:
            raise ValueError(f"{case_id}: first-seen source digest mismatch")
        if matches[0]["created_at"] != entry["created_at"]:
            raise ValueError(f"{case_id}: source created_at mismatch")

        for folder in ("predictions", "dispositions", "expirations"):
            if (root / folder / filename).exists():
                raise ValueError(f"{case_id}: incompatible pre-existing {folder} evidence")

        record = {
            "schema_version": 1,
            "case_id": case_id,
            "source_surface_sha256": entry["surface_sha256"],
            "first_frozen_at": entry["first_frozen_at"],
            "prediction_deadline_at": entry["prediction_deadline_at"],
            "expiry_recorded_at": fmt(now),
            "reason": "No prediction or original-surface disposition frozen before six-hour deadline.",
            "independence_class": "UNADJUDICATED",
            "primary_prospective_I2_eligible": False,
            "claim_boundary": (
                "Administrative missed-SLA evidence only. No retrospective mechanism "
                "prediction, I0 exclusion, independent hit, or external-adoption credit."
            ),
        }
        record["expiry_sha256"] = digest(record)
        candidates.append((entry, root / "expirations" / filename, record))

    if candidates:
        (root / "expirations").mkdir(exist_ok=True)
        for entry, path, record in candidates:
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            entry["status"] = STATUS_EXPIRED
            entry["expiry_recorded_at"] = record["expiry_recorded_at"]
            entry["expiry_sha256"] = record["expiry_sha256"]
            entry["expiry_source"] = str(path.relative_to(root))
            entry["independence_class"] = "UNADJUDICATED"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"status": "pass", "expired_count": len(candidates), "expired_cases": [x[0]["id"] for x in candidates]}


def self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "inbox").mkdir()
        case = "o/r#7"
        first = "2026-10-07T12:00:00Z"
        deadline = "2026-10-07T18:00:00Z"
        state = {"seen": [{
            "id": case, "status": STATUS_PENDING, "source": "inbox/first.json",
            "surface_sha256": "a" * 64, "created_at": "2026-10-07T11:00:00Z",
            "first_frozen_at": first, "prediction_deadline_at": deadline,
        }]}
        (root / "seen.json").write_text(json.dumps(state))
        (root / "inbox/first.json").write_text(json.dumps({"new_candidates": [{
            "id": case, "surface_sha256": "a" * 64, "created_at": "2026-10-07T11:00:00Z",
        }]}))
        assert expire(root, parse(deadline))["expired_count"] == 0
        result = expire(root, parse("2026-10-07T18:00:01Z"))
        assert result["expired_count"] == 1
        assert expire(root, parse("2026-10-07T19:00:00Z"))["expired_count"] == 0
        record = json.loads((root / "expirations/o__r__7.json").read_text())
        assert record["expiry_sha256"] == digest(record)
        assert record["primary_prospective_I2_eligible"] is False
        assert json.loads((root / "seen.json").read_text())["seen"][0]["status"] == STATUS_EXPIRED
    print("expiry self-test: pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--at", help="UTC test clock; never use to backdate production")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.at:
        raise ValueError("--at is disabled for live ledger writes; historical backdating is forbidden")
    now = dt.datetime.now(dt.timezone.utc)
    print(json.dumps(expire(ROOT, now), sort_keys=True))


if __name__ == "__main__":
    main()
