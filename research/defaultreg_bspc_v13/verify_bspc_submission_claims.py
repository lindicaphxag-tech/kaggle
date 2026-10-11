#!/usr/bin/env python3
"""Read-only claim consistency gate for Default Regularisation V12.

This is NOT raw EEG replication. It verifies manuscript figures against the
canonical, checked-in evidence JSON, checks declared hierarchy and explicitly
flags the independent Shin2017A data lineage as an external unresolved gate.

Run from any cwd:
    python papers/eeg-default-regularisation/paper_vnext/verify_bspc_submission_claims.py
    python .../verify_bspc_submission_claims.py --strict

Non-strict: report ALL findings and exit 0 after audit.
Strict: fail closed if any release blocker remains.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
EVIDENCE = P / "evidence" / "current_verified_results.json"
TABLE = P / "91_table_main_prospective_results.tex"
ABSTRACT = P / "00_abstract_evidence_locked.tex"
MANUSCRIPT = P / "main_bspc_v13_DRAFT.tex"
STATUS = P / "STATUS.md"
MAP = (
 ("EEGMMIDB_S021_S040", "EEGMMIDB S021--S040", 20),
 ("BNCI2014_002", r"BNCI2014\_002", 14),
 ("BNCI2015_001", r"BNCI2015\_001", 12),
)
HEAD = "source_macro_full_domain_LDA"
COV = "source_cov_nll_full_domain_LDA"
LW = "LedoitWolf_LDA"


def check():
    issues = []
    release_blockers = []
    verified = {}
    obj = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    tab = TABLE.read_text(encoding="utf-8")
    abstract = ABSTRACT.read_text(encoding="utf-8")
    main = MANUSCRIPT.read_text(encoding="utf-8")
    status = STATUS.read_text(encoding="utf-8")
    if obj.get("generated_from_verified_artifacts") is not True:
        issues.append("Evidence file not marked artifact-derived")
    if obj.get("manual_derived_effect_fields_forbidden") is not True:
        issues.append("Manual effect field policy is not true in evidence JSON")
    for key, label, n in MAP:
        cohort = obj["cohorts"][key]
        if cohort.get("n_participants") != n:
            issues.append(f"{key}: participant denominator disagrees")
        if cohort.get("target_information_budget") != "Z0":
            issues.append(f"{key}: target information budget not Z0")
        a = cohort["macro_auc"]
        macro, cov, lw = map(float, (a[HEAD], a[COV], a[LW]))
        i = cohort["primary_inference"]
        dc = i[f"{HEAD}__vs__{COV}"]
        dl = i[f"{HEAD}__vs__{LW}"]
        for suffix, expected, val in (
            ("cov", macro-cov, dc["mean_delta"]),
            ("lw", macro-lw, dl["mean_delta"]),
        ):
            if abs(expected-float(val)) > 1e-8:
                issues.append(f"{key}: arithmetic mismatch {suffix}={expected} vs stored {val}")
        if not (0 <= float(dc["holm_p"]) <= 1 and 0 <= float(dl["holm_p"]) <= 1):
            issues.append(f"{key}: invalid adjusted p-value")
        # Match the actual table row; N and macro, covariance and LW values are
        # the first four numeric fields. This prevents mixing data configurations.
        needle = re.escape(label) + r"\s*&\s*(\d+)\s*&\s*(\d+\.\d+)\s*&\s*(\d+\.\d+)\s*&\s*(\d+\.\d+)"
        match = re.search(needle, tab)
        if not match:
            issues.append(f"{key}: cannot locate table row with required numeric fields")
        else:
            got = (int(match[1]), *map(float, match.groups()[1:]))
            want = (n, macro, cov, lw)
            if got[0] != want[0] or any(abs(u-v)>0.00006 for u,v in zip(got[1:],want[1:])):
                issues.append(f"{key}: table vs artifact discrepancy: {got} vs {want}")
        # All rows must preserve adjusted nonsignificance if exact Holm > .05
        verified[key] = {
            "n":n, "macro_auc":macro, "cov_nll_auc":cov, "ledoit_wolf_auc":lw,
            "delta_cov":float(dc["mean_delta"]), "holm_cov":float(dc["holm_p"]),
            "delta_lw":float(dl["mean_delta"]), "holm_lw":float(dl["holm_p"]),
            "covariance_difference_significant":float(dc["holm_p"]) < .05,
        }
    for phrase in ("universal shrinkage multiplier", "hidden filter-bank advantage",
                   "98.3% of", "82% of the"):
        if phrase.lower() in abstract.lower():
            issues.append(f"Historic ungrounded claim revived in abstract: {phrase}")
    if not any(term in abstract.lower() for term in ("falsif", "refut", "contradict")):
        issues.append("Abstract omits prospective representation falsification")
    if "zero" not in status.lower():
        issues.append("Evidence status has lost population frontier zero-bound caveat")
    if "92_table_validity_frontier.tex" not in main or "91_table_main_prospective_results.tex" not in main:
        issues.append("Main assembly missing required canonical evidence tables")
    # The Shin PR #97 is in lindicaphxag-tech/kaggle, not in cs.
    # V13 records the successful ORIGINAL Kaggle run and its printed exact
    # statistics, while keeping score-file ZIP extraction a separate gate.
    receipt_file = P / "evidence/shin2017a_logged_receipt_v13.json"
    if not receipt_file.is_file():
        release_blockers.append("RELEASE BLOCKER: separate Shin Kaggle workflow receipt not mounted")
    else:
        rec = json.loads(receipt_file.read_text(encoding="utf-8"))
        run = rec.get("execution_summary", {})
        origin = rec.get("source", {})
        if (run.get("dataset") != "Shin2017A" or run.get("n_subjects") != 29 or
            run.get("trial_denominator") != 1740 or
            origin.get("repository") != "lindicaphxag-tech/kaggle" or
            origin.get("workflow_run_id") != 38065081504 or
            origin.get("job_outcome") != "success" or
            not re.fullmatch(r"[0-9a-f]{64}", run.get("score_manifest_sha256", ""))):
            issues.append("Shin2017A original workflow evidence identity mismatch")
        m = run.get("mean_auc_by_method", {})
        contrasts = run.get("primary_comparisons", [])
        if len(contrasts) != 2:
            issues.append("Shin2017A original outcome has wrong number of primary contrasts")
        else:
            for item in contrasts:
                base = item["baseline"]
                if abs(m[item["candidate"]] - m[base] - item["mean_delta_auc"]) > 1e-9:
                    issues.append("Shin2017A raw workflow AUC arithmetic mismatch: " + base)
                if item.get("signflip_mode") != "monte_carlo" or item.get("signflip_draws") != 200000:
                    issues.append("Shin2017A sign-flip method incorrectly classified as exact")
            verified["Shin2017A_workflow_logged"] = {
                "n":29,
                "macro_auc":m.get(HEAD),
                "cov_nll_auc":m.get(COV),
                "ledoit_wolf_auc":m.get(LW),
                "delta_cov":contrasts[0]["mean_delta_auc"],
                "holm_cov":contrasts[0]["holm_p_primary_family"],
                "delta_lw":contrasts[1]["mean_delta_auc"],
                "holm_lw":contrasts[1]["holm_p_primary_family"],
                "score_manifest_sha256":run.get("score_manifest_sha256"),
            }
    independent = P / "evidence/shin2017a_original_artifact_verified_v13.json"
    if not independent.is_file():
        release_blockers.append("RELEASE BLOCKER: no original 29-score ZIP checksum audit receipt")
    else:
        check = json.loads(independent.read_text(encoding="utf-8"))
        expected_manifest = "00d1aa5bb912aa7badcdc5f1c9d3f6e690551d4ffee4d6bff234c61c15582e9e"
        if (check.get("status") != "PASS_ORIGINAL_FROZEN_SCORE_ARTIFACT_INTEGRITY_AND_FOLD_AUC_ARITHMETIC" or
            check.get("n_subjects") != 29 or check.get("checked_score_npz_sha256") != 29 or
            check.get("score_manifest_sha256") != expected_manifest or
            check.get("original_artifact_zip_sha256") !=
            "cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842"):
            issues.append("Independent original Shin artifact audit receipt is inconsistent")
    # The independent integrity check is NOT a full rerun of raw EDF training.
    release_blockers.append("RELEASE BLOCKER: approved author identities, licensing, ethics, disclosures and final publisher-source format")
    return {"status":"CONSISTENCY_ERROR" if issues else "CONSISTENT_BUT_RELEASE_HOLD" if release_blockers else "STATIC_CHECK_ONLY",
            "checked_cohorts":verified, "consistency_errors":issues,
            "release_blockers":release_blockers,
            "method":"read-only canonical evidence and LaTeX reconciliation; not a new EEG experiment"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--check-consistency", action="store_true", help="Fail if manuscript and canonical evidence disagree, while separately reporting editorial release blockers")
    args=parser.parse_args()
    data=check()
    print(json.dumps(data,indent=2,ensure_ascii=False))
    if data["consistency_errors"] and (args.strict or args.check_consistency):
        return 2
    if args.strict and data["release_blockers"]:
        return 3
    return 0


if __name__=="__main__":
    sys.exit(main())
