# ClinGeom prospective transport — v1 invalidation / v2 pre-activation anchor

## v1 status

ClinGeom public prospective transport v1 is **invalidated for prospective
evidence collection**. It produced no valid prospective hits.

Root cause discovered on 2026-10-08:

1. frozen v1 watchset contained the inaccessible/non-canonical repository
   `MIC-DKFZ/MedSAM3`, whose issues endpoint returned HTTP 404;
2. `transport/scan.py` correctly failed closed on that repository;
3. the GitHub Actions step invoked the scanner as
   `python ... | tee ...` without shell `pipefail`, so the Python failure was
   masked by `tee` returning success;
4. the workflow therefore displayed a successful scan job and later printed
   `No new ClinGeom first-seen cases` even though the scanner had aborted before
   completing the frozen repository set.

## Evidence consequence

- v1 `seen.json` remained empty;
- no v1 prediction was frozen;
- no v1 issue is counted as a prospective candidate, success, failure, I1/I2
  credit, or external validation;
- issues that existed before the clean v2 activation are **historical for v2**
  and will not be retroactively promoted.

## v2 rule

The **creation commit of this file** is the public pre-activation anchor for
v2. Its GitHub commit timestamp will be copied into the v2 watchset cutoff.

v2 must additionally:

- remove inaccessible/non-canonical repository identities rather than silently
  skipping them;
- run the scanner under `set -euo pipefail` so scanner failure makes the Actions
  step and workflow fail;
- retain fail-closed all-repository fetch semantics;
- restart `seen.json` empty at the new cutoff;
- preserve the same pre-prediction information barrier.

This invalidation is part of the research evidence. It must not be edited into
a claim that v1 successfully observed zero post-activation issues.
