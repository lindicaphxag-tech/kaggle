# ClinGeom prospective transport

This directory is the public first-seen / prediction / resolution transport for
ClinGeom external evidence.

## Frozen identity

- activation commit: `e2021356533586d09192eb9afa42a374f4856acb`
- activation cutoff: `2026-10-06T03:10:20Z`
- repositories: exactly the 12 entries in `watchset.json`
- prediction SLA: 6 hours from the first frozen public surface

`seen.json` starts empty by design. Historical issues are never imported and
cannot be upgraded into prospective credit.

## Information barrier

Before a prediction/disposition is frozen, only the **original GitHub issue
endpoint surface** may be used: title, original body, creation time, author and
labels. Comments, linked pull requests, later commits, maintainer diagnoses and
fixes are forbidden.

The pipeline stores a SHA-256 of the original title/body. Any later resolution
is collected only after a prediction digest exists.

## What a scanner hit means

A newly frozen issue is only a **candidate**. Capture alone is not evidence that
ClinGeom predicted the mechanism, found a bug, or earned external adoption.

Within the SLA it must be given either:

1. a blind exclusion/disposition based only on the frozen original surface; or
2. a tamper-evident mechanism prediction that can later be falsified by upstream
   resolution evidence.

## Manual workflow triggers

The workflow also runs on pushes under `clingeom_transport/**`. Such a trigger
does **not** change the activation cutoff or watchset. The validation job checks
the frozen activation commit/cutoff and repository-count identity before any scan
is allowed to mutate `seen.json`.

## Claim boundary

- scanner success is not a prospective hit;
- candidate capture is not a mechanism match;
- self-authored triage is not external confirmation;
- only later upstream evidence can adjudicate a frozen prediction;
- clinical validation is never inferred from this transport.
