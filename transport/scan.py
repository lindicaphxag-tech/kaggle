#!/usr/bin/env python3
"""Public fail-closed first-seen transport for frozen embodied repositories."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
WATCHSET = ROOT / "watchset.json"
SEEN = ROOT / "seen.json"
INBOX = ROOT / "inbox"
API = "https://api.github.com"
PER_PAGE = 100
MAX_PAGES = 10


def parse_utc(value: str) -> dt.datetime:
    out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if out.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value}")
    return out.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def surface_sha(title: str, body: str) -> str:
    payload = json.dumps(
        {"title": title, "body": body},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def api_get(path: str, token: str):
    req = urllib.request.Request(
        API + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "semrepair-public-prospective-transport",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) < 100:
                raise RuntimeError(f"GitHub rate-limit safety floor reached: {remaining}")
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code} for {path}: {detail}") from exc


def scan_repo(repo: str, cutoff: dt.datetime, token: str) -> list[dict]:
    owner, name = repo.split("/", 1)
    found: list[dict] = []
    reached_cutoff = False
    for page in range(1, MAX_PAGES + 1):
        query = urllib.parse.urlencode({
            "state": "all",
            "sort": "created",
            "direction": "desc",
            "per_page": PER_PAGE,
            "page": page,
        })
        rows = api_get(f"/repos/{owner}/{name}/issues?{query}", token)
        if not isinstance(rows, list):
            raise RuntimeError(f"{repo}: unexpected issues payload")
        if not rows:
            reached_cutoff = True
            break
        oldest = None
        for row in rows:
            created = parse_utc(row["created_at"])
            oldest = created if oldest is None else min(oldest, created)
            if created <= cutoff or "pull_request" in row:
                continue
            title = row.get("title") or ""
            body = row.get("body") or ""
            found.append({
                "id": f"{repo}#{row['number']}",
                "repository": repo,
                "issue_number": int(row["number"]),
                "html_url": row["html_url"],
                "created_at": row["created_at"],
                "title": title,
                "body": body,
                "surface_sha256": surface_sha(title, body),
                "author_login": (row.get("user") or {}).get("login"),
                "labels": [
                    label.get("name")
                    for label in row.get("labels", [])
                    if isinstance(label, dict) and label.get("name")
                ],
            })
        if oldest is not None and oldest <= cutoff:
            reached_cutoff = True
            break
        if len(rows) < PER_PAGE:
            reached_cutoff = True
            break
        time.sleep(0.03)
    if not reached_cutoff:
        raise RuntimeError(f"{repo}: pagination safety limit reached before activation cutoff")
    return found


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    watch = json.loads(WATCHSET.read_text(encoding="utf-8"))
    state = json.loads(SEEN.read_text(encoding="utf-8"))
    repos = watch["repositories"]
    if len(repos) != watch["repository_count"] or len(repos) != len(set(repos)):
        raise RuntimeError("frozen repository identity/count mismatch")
    cutoff = parse_utc(watch["public_activation_cutoff"])
    seen_ids = {x["id"] for x in state.get("seen", []) if isinstance(x, dict) and x.get("id")}

    # Fetch all repositories before mutating disk: any transport error fails closed.
    all_rows: list[dict] = []
    for repo in repos:
        all_rows.extend(scan_repo(repo, cutoff, token))

    unique = {row["id"]: row for row in all_rows}
    new = [unique[k] for k in sorted(unique) if k not in seen_ids]
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    report = {
        "schema_version": 1,
        "scanned_at": fmt_utc(now),
        "public_activation_cutoff": watch["public_activation_cutoff"],
        "repositories_scanned": len(repos),
        "post_activation_issue_identities_observed": len(unique),
        "new_candidate_count": len(new),
        "new_candidates": new,
        "information_barrier": watch["information_barrier"],
        "claim_boundary": watch["claim_boundary"],
    }

    if new:
        INBOX.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        out = INBOX / f"FIRST_SEEN_{stamp}.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        deadline_delta = dt.timedelta(hours=int(watch["prediction_sla_hours"]))
        state.setdefault("seen", [])
        for row in new:
            deadline = now + deadline_delta
            state["seen"].append({
                "id": row["id"],
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "created_at": row["created_at"],
                "first_frozen_at": fmt_utc(now),
                "prediction_deadline_at": fmt_utc(deadline),
                "surface_sha256": row["surface_sha256"],
                "source": str(out.relative_to(ROOT)),
                "status": "candidate-frozen-awaiting-prediction",
            })
        SEEN.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise
