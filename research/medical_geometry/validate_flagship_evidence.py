"""Fail-closed, stdlib-only reviewer check for the frozen ClinGeom evidence ledger.

Offline mode checks schema, known necessary mathematical relationships,
evidence identity fields and nonclaims; live mode additionally verifies
that the exact recorded PR heads really have successful public CI runs.
The file records first-party results, NOT a reproduction of raw experiments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

REPO = "lindicaphxag-tech/kaggle"
DATA = Path(__file__).with_name("FLAGSHIP_EVIDENCE_2026.json")


def _github_json(endpoint: str) -> dict:
    req = Request(
        f"https://api.github.com/repos/{REPO}/{endpoint}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "clingeom-flagship-evidence-audit",
            **({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
               if os.environ.get("GITHUB_TOKEN") else {}),
        },
    )
    with urlopen(req, timeout=20) as r:
        return json.load(r)


def audit(*, live: bool = False) -> dict:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert data["schema"] == "clingeom.flagship.evidence.v1"
    assert data["evidence_class"].startswith("author-operated public")
    for key in ("upstream_maintainer_adoption", "independent_reproduction",
                "clinical_accuracy_gain", "arxiv_submission",
                "reviewer_acceptance"):
        assert data["attribution"][key] is False, f"unsupported claim: {key}"
    rows = data["entries"]
    assert {e["id"] for e in rows} == {"P1", "P2", "P3", "P4"}
    assert len({e["run_id"] for e in rows}) == len(rows)
    assert len({e["pr"] for e in rows}) == len(rows)
    for row in rows:
        assert row["head_sha"] and len(row["head_sha"]) == 40
        assert row["run_id"] > 0 and row["pr"] > 0
        assert row["source_revision"] and len(row["source_revision"]) == 40
        assert "NOT " in row["measurement"] or row["id"] in {"P1", "P2"}
        if live:
            pr = _github_json(f"pulls/{row['pr']}")
            run = _github_json(f"actions/runs/{row['run_id']}")
            if not (pr.get("merged_at") and pr["head"]["sha"] == row["head_sha"]):
                raise AssertionError(f"{row['id']}: PR merge/head differs from frozen evidence")
            if not (run.get("conclusion") == "success"
                    and run.get("status") == "completed"
                    and run.get("head_sha") == row["head_sha"]):
                raise AssertionError(f"{row['id']}: run is not successful for the declared exact head")
    p1, p2, p3, p4 = rows
    for plane in ("axial","coronal","sagittal"):
        assert p1["metrics"][plane]["repaired"] == 1.0
        assert 0.0 <= p1["metrics"][plane]["original"] <= 1.0
    assert p1["checkpoint"] and p1["real_mri"]
    assert p2["metrics"]["postmix_float32_max_absolute_difference"] == 0.0
    assert p2["metrics"]["minimax_postmix_lower_bound"] == (
        p2["metrics"]["physically_intended_target_difference"]/2.0
    )
    assert p2["metrics"]["premix_repair_max_absolute_error"] < 1e-6
    assert p3["metrics"]["original_nonaxial_expected_failures_per_version"] > 0
    assert p3["metrics"]["patched_test_passes_per_version"] > 0
    assert set(p3["metrics"]["python_versions"]) == {"3.10","3.12"}
    assert p4["metrics"]["affine_max_abs_error_before"] == 0
    assert p4["metrics"]["affine_max_abs_error_after"] == 0
    assert not p4["metrics"]["uid_auto_roundtrip_before"]
    assert p4["metrics"]["uid_auto_roundtrip_after"]
    return {
        "audit": "pass", "mode": "live PR-and-run API" if live else "offline",
        "verified_evidence_entries": len(rows),
        "third_party_adoption_claimed": False,
        "clinical_accuracy_claimed": False,
        "notice": ("CI validates public run metadata and consistency; it does"
                   " not independently rerun the underlying medical experiments."),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Also check exact PR/run SHA and success on GitHub")
    options = parser.parse_args()
    print(json.dumps(audit(live=options.live),indent=2,sort_keys=True))
