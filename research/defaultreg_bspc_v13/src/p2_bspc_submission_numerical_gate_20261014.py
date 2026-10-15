#!/usr/bin/env python3
"""Fail-closed arithmetic and source-claim checks for P2 BSPC author-review drafts.

This script does not download EEG, infer patient identity, or approve submission.
It checks reported deterministic descriptive quantities; original CI receipts
and owner/ethics approvals remain separately required.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
import math
import re
import sys

BASE = Path(__file__).resolve().parents[1]
FOCUS = BASE / "main_p2_BSPC_FOCUSED_AUTHOR_REVIEW_20261014.tex"
FIG = BASE / "P2_BSPC_native_matrix_and_attribution_20261014.tex"
SUPP = BASE / "P2_BSPC_SUPPLEMENTARY_EVIDENCE_AUTHOR_REVIEW_20261014.tex"
HIGHLIGHTS = BASE.parents[1] / "publication_packages" / "default_eeg" / "P2_BSPC_HIGHLIGHTS_AUTHOR_REVIEW.txt"


def main() -> int:
    main_tex = FOCUS.read_text(encoding="utf-8")
    fig_tex = FIG.read_text(encoding="utf-8")
    supp_tex = SUPP.read_text(encoding="utf-8")
    lines = [x.lstrip("•-* \t") for x in HIGHLIGHTS.read_text(encoding="utf-8").splitlines() if x.strip()]
    values = {
        "nominal_50": 0.594000,
        "AA": 0.588673, "AB": 0.587347,
        "BA": 0.580765, "BB": 0.582551,
        "one_affected_group": -0.2625,
        "other_48_mean": -0.00078125,
    }
    check = {
        "published_pairs_full": math.comb(50, 2) == 1225,
        "full_census_covers_all": 1222 + 3 == 1225,
        "edf_numerical_blocks": 29 * 320 == 9280,
        "edf_pair_4_5": "9280/9280" in fig_tex,
        "edf_pair_47_49": "6957/9280" in fig_tex,
        "edf_pair_11_15": "171/9280" in fig_tex,
        "frozen_AA": "0.588673" in main_tex and ".588673" in fig_tex,
        "frozen_BB": "0.582551" in main_tex and ".582551" in fig_tex,
        "denominator_only": abs(values["AA"] - values["nominal_50"] - (-0.005327)) < 1e-8,
        "matched_fit": abs(values["BB"] - values["AA"] - (-0.006122)) < 1e-8,
        "source_choice_interaction": abs((values["BB"] - values["BA"] - values["AB"] + values["AA"]) - 0.003112) < 1e-8,
        "full_contrast": abs(values["BB"] - values["nominal_50"] - (-0.011449)) < 1e-8,
        "localization": abs((values["one_affected_group"] + 48 * values["other_48_mean"]) / 49 - (values["BB"] - values["AA"])) < 1e-6,
        "highlights_3_to_5": 3 <= len(lines) <= 5,
        "highlights_max85": all(len(x) <= 85 for x in lines),
        "patient_identity_scope": "patient" in main_tex.lower() and "not" in main_tex.lower(),
        "custodian_uncertainty": "ancestry remains unresolved" in main_tex,
        "supplement_sources_preserved": supp_tex.count("\\input{") >= 18,
        "author_placeholders_preserved": "AUTHOR IDENTITIES AND AFFILIATIONS TO BE VERIFIED" in main_tex,
    }
    receipt = {
        "status": "REPRODUCIBLE_DESCRIPTIVE_NUMERICAL_PRECHECK_ONLY",
        "checks": check,
        "pass": all(check.values()),
        "reported_quantities": values,
        "figure": {
            "exact_4_5": 9280,
            "partial_47_49": 6957,
            "partial_11_15": 171,
            "block_denominator": 9280,
        },
        "sha256": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in (FOCUS, FIG, SUPP, HIGHLIGHTS)
        },
        "does_not_verify": [
            "clinical participant identity", "source MAT to EDF provenance",
            "independent external acquisition replication",
            "population-average causal leakage",
            "all-author publication rights and declarations",
        ],
        "submission_authorized": False,
    }
    output = BASE / "P2_BSPC_ARITHMETIC_RECEIPT_20261014.json"
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
