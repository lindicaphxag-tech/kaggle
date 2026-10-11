"""Re-evaluate ClinGeom's previously frozen six-patient negative ROI evidence.

Reads the actual versioned source report instead of copying six numbers to
an isolated unit-test fixture. It does NOT download the underlying MRI/mask
or constitute an independent re-run of the pretrained network.
"""
from pathlib import Path
from statistics import mean
import re

FROZEN_REPORT = (Path(__file__).resolve().parent / "upstream_repros" /
                 "REAL_PROSPECTIVE_SIX_RESULT.md")


def verify_frozen_negative_source(path: Path = FROZEN_REPORT) -> dict:
    text = path.read_text(encoding="utf8")
    matches = [line for line in text.splitlines()
               if line.startswith("All six patient-level Dice deltas")]
    if len(matches) != 1:
        raise ValueError("missing/ambiguous six-patient frozen delta evidence")
    entries = re.findall(r"[−-]0\.\d+", matches[0])
    if len(entries) != 6:
        raise ValueError("patient delta count != 6")
    values = [float(t.replace("−", "-")) for t in entries]
    if not all(x < 0 for x in values):
        raise AssertionError("original source no longer supports all-negative patient deltas")
    mean_delta = mean(values)
    a = re.search(r"\| \*\*Canonical-RAS unpermuted first-view.*?\| \*\*(0\.\d+)\*\*", text)
    b = re.search(r"\|\s*\*{0,2}Label-free operator/view-risk gated probability fusion.*?\|\s*\*{0,2}(0\.\d+)", text)
    if not a or not b:
        raise ValueError("strong original first-view comparison omitted")
    source_gap = float(b.group(1)) - float(a.group(1))
    if abs(source_gap-mean_delta)>1e-6:
        raise AssertionError("strong baseline table and six per-patient deltas disagree")
    if "three predeclared 64³ regions per patient" not in text and "three fixed source-image-only 64³ ROIs" not in text:
        raise ValueError("ROI-dependent measurement unit is not disclosed")
    return {
        "source": str(path),
        "independent_subject_ids":6,
        "historical_paired_roi_patient_delta_mean":mean_delta,
        "strong_canonical_first_table_mean_dice":float(a.group(1)),
        "gated_table_mean_dice":float(b.group(1)),
        "all_paired_patient_dice_changes_negative":True,
        "claim_scope":"text-source consistency only, not new mask inference",
    }
