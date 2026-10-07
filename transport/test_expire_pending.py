"""Fail-closed SLA tests against frozen public cases; no live API calls."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from transport import expire_pending, validate_ledger


class ExpiryLedgerTests(unittest.TestCase):
    def test_real_public_snapshot_expiration_and_tamper_rejection(self) -> None:
        source = Path(__file__).resolve().parent
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(source / "seen.json", root / "seen.json")
            for name in ("inbox", "predictions", "dispositions", "expirations"):
                if (source / name).exists():
                    shutil.copytree(source / name, root / name)

            before = json.loads((root / "seen.json").read_text())["seen"]
            now = dt.datetime.now(dt.timezone.utc)
            overdue = {
                e["id"] for e in before
                if e["status"] == expire_pending.STATUS_PENDING
                and expire_pending.parse(e["prediction_deadline_at"]) < now
            }
            observed = expire_pending.expire(root, now)
            self.assertEqual(set(observed["expired_cases"]), overdue)
            self.assertEqual(expire_pending.expire(root, now)["expired_count"], 0)
            with patch.multiple(
                validate_ledger,
                ROOT=root,
                SEEN=root / "seen.json",
                PRED=root / "predictions",
                DISP=root / "dispositions",
                EXP=root / "expirations",
            ):
                validate_ledger.main()
                if observed["expired_cases"]:
                    target = root / "expirations" / (
                        expire_pending.safe(observed["expired_cases"][0]) + ".json"
                    )
                    doc = json.loads(target.read_text())
                    doc["reason"] = "altered retrospective explanation"
                    target.write_text(json.dumps(doc))
                    with self.assertRaisesRegex(ValueError, "expiration digest mismatch"):
                        validate_ledger.main()

    def test_inconsistent_snapshot_refuses_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inbox").mkdir()
            e = {
                "id": "o/r#1",
                "status": expire_pending.STATUS_PENDING,
                "source": "inbox/first.json",
                "surface_sha256": "a" * 64,
                "created_at": "2026-10-07T11:00:00Z",
                "first_frozen_at": "2026-10-07T12:00:00Z",
                "prediction_deadline_at": "2026-10-07T18:00:00Z",
            }
            (root / "seen.json").write_text(json.dumps({"seen": [e]}))
            (root / "inbox/first.json").write_text(json.dumps({"new_candidates": [{
                "id": e["id"], "surface_sha256": "b" * 64,
                "created_at": e["created_at"],
            }]}))
            with self.assertRaisesRegex(ValueError, "source digest mismatch"):
                expire_pending.expire(root, expire_pending.parse("2026-10-08T00:00:00Z"))
            self.assertFalse((root / "expirations").exists())
            self.assertEqual(
                json.loads((root / "seen.json").read_text())["seen"][0]["status"],
                expire_pending.STATUS_PENDING,
            )


if __name__ == "__main__":
    unittest.main()
