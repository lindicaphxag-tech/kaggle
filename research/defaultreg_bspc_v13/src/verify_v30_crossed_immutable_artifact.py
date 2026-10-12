#!/usr/bin/env python3
"""Independent audit of the immutable V29 49-group EEG model x selector artifact.

Not a new experiment and not causal identification. Computes DESCRIPTIVE
recording-group paired contrasts from ALREADY EXPOSED frozen EEG outcomes.
No model fits or EEG data are read, no parameter policy is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

RECEIPT_SHA = "0e7bfdb73970fcff9231a51954bd2d548f472cb296624967afc80d9d241ce2be"
NAME = "CROSSED_MODEL_SELECTION_49_GROUPS.json"
POLICIES = ("source_macro", "proper_source_mean_gaussian", "historical_class_centred")
CELLS = ("AA_old_fit_old_lambda", "AB_old_fit_purged_lambda",
         "BA_purged_fit_old_lambda", "BB_purged_fit_purged_lambda")


def contrasts(matrix):
    """Pure 2x2 algebra: order [AA, AB, BA, BB], with a fixed matched target."""
    aa, ab, ba, bb = map(float, matrix)
    return {
        "lambda_change_old_fit": ab - aa,
        "refit_effect_old_lambda": ba - aa,
        "lambda_change_purged_fit": bb - ba,
        "refit_effect_purged_lambda": bb - ab,
        "model_by_lambda_interaction": bb - ba - ab + aa,
        "paired_full_change": bb - aa,
    }


def audit_archive(file: Path):
    with zipfile.ZipFile(io.BytesIO(file.read_bytes())) as z:
        names = [n for n in z.namelist() if n.endswith(NAME)]
        if len(names) != 1:
            raise AssertionError("missing or ambiguous frozen V29 cross-result")
        raw = z.read(names[0])
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECEIPT_SHA:
        raise AssertionError(f"immutable V29 receipt digest mismatch: {digest}")
    data = json.loads(raw)
    rows = data["held_record_crossed_responses"]
    if (data["status"] != "V29_RETROSPECTIVE_FROZEN_FOUR_CELL_REFIT_SELECTION_COMPLETE"
        or not data["all_49_original_and_purged_outcomes_reproduced"]
        or len(rows) != 49 or
        set(int(r["held_representative"]) for r in rows) != set(range(1, 51)) - {5}
        or set(data["method_summary"]) != set(POLICIES)):
        raise AssertionError("invalid frozen V29 study denominator or provenance")

    results = {}
    for policy in POLICIES:
        matrix = np.asarray([
            [r["method_crossed_AUC"][policy][key] for key in CELLS]
            for r in sorted(rows, key=lambda x: x["held_representative"])
        ], dtype=float)
        if matrix.shape != (49, 4) or not np.isfinite(matrix).all():
            raise AssertionError("invalid real four-cell EEG outcome array")
        if not (np.all(matrix >= 0) and np.all(matrix <= 1)):
            raise AssertionError("AUC outside [0,1]")
        c = [contrasts(line) for line in matrix]
        stats = data["method_summary"][policy]
        for k in CELLS:
            stored = stats["participant_macro_mean"][k]
            observed = float(matrix[:, CELLS.index(k)].mean())
            if abs(observed - stored) > 1e-12:
                raise AssertionError(f"macro AUC not independently reproduced {policy}/{k}")
        for source_key, ours in (
            ("selector_change_on_old_fit_AB_minus_AA","lambda_change_old_fit"),
            ("fit_change_at_old_lambda_BA_minus_AA","refit_effect_old_lambda"),
            ("selector_change_on_purged_fit_BB_minus_BA","lambda_change_purged_fit"),
            ("fit_change_at_new_lambda_BB_minus_AB","refit_effect_purged_lambda"),
            ("model_selector_interaction","model_by_lambda_interaction"),
            ("total_purged_BB_minus_original_AA","paired_full_change"),
        ):
            vals = np.asarray([a[ours] for a in c], dtype=float)
            stored = stats["participant_macro_mean"][source_key]
            if abs(float(vals.mean()) - stored) > 1e-12:
                raise AssertionError(f"factorial mean mismatch {policy}/{source_key}")
        for r, v in zip(sorted(rows,key=lambda x:x["held_representative"]), c):
            saved = r["method_crossed_AUC"][policy]
            if any(abs(saved[k] - v[new]) > 1e-12 for k,new in (
                ("selector_change_on_old_fit_AB_minus_AA", "lambda_change_old_fit"),
                ("fit_change_at_old_lambda_BA_minus_AA", "refit_effect_old_lambda"),
                ("selector_change_on_purged_fit_BB_minus_BA", "lambda_change_purged_fit"),
                ("fit_change_at_new_lambda_BB_minus_AB", "refit_effect_purged_lambda"),
                ("model_selector_interaction", "model_by_lambda_interaction"),
                ("total_purged_BB_minus_original_AA", "paired_full_change"),
            )):
                raise AssertionError("individual 2x2 calculation mismatch")
            if not saved["selected_lambda_changed"]:
                if (abs(v["lambda_change_old_fit"])>1e-10 or
                    abs(v["lambda_change_purged_fit"])>1e-10 or
                    abs(v["model_by_lambda_interaction"])>1e-10):
                    raise AssertionError("same lambdas unexpectedly changed AUC rank")

        # This conditional resampling preserves all four matched responses
        # of each recording group. It does NOT account for overlapping source
        # pools, so intervals are exploratory and do not have formal coverage.
        rng = np.random.default_rng(20261013)
        indices = rng.integers(0, 49, (16000,49))
        row_contrasts = np.asarray([list(v.values()) for v in c])
        boot = row_contrasts[indices].mean(axis=1)
        boot_q = np.quantile(boot, [0.025,0.975], axis=0)
        # A zero count of changed choices can arise from identical boundary
        # saturation, not from a proven robust prediction policy.
        old_lambdas = np.asarray([
            r["method_crossed_AUC"][policy]["lambda_A"] for r in rows
        ],dtype=float)
        purged_lambdas = np.asarray([
            r["method_crossed_AUC"][policy]["lambda_B"] for r in rows
        ],dtype=float)
        def histogram(v):
            unique, counts = np.unique(v,return_counts=True)
            return {format(float(key),".10g"):int(count)
                    for key,count in zip(unique,counts)}
        results[policy] = {
            "source_lambda_original_histogram":histogram(old_lambdas),
            "source_lambda_purged_histogram":histogram(purged_lambdas),
            "original_lambda_zero_count":int(np.sum(old_lambdas==0)),
            "original_lambda_one_count":int(np.sum(old_lambdas==1)),
            "purged_lambda_zero_count":int(np.sum(purged_lambdas==0)),
            "purged_lambda_one_count":int(np.sum(purged_lambdas==1)),
            "n_unique_processed_recording_groups": 49,
            "matched_mean_auc": {k:float(matrix[:,i].mean()) for i,k in enumerate(CELLS)},
            "mean_contrast": {key:float(np.mean([v[key] for v in c])) for key in c[0]},
            "descriptive_record_group_bootstrap_95_not_independent_fold_CI": {
                k:[float(boot_q[0,i]),float(boot_q[1,i])]
                for i,k in enumerate(c[0])
            },
            "n_lambda_changed":sum(r["method_crossed_AUC"][policy]["selected_lambda_changed"] for r in rows),
            "n_purged_fit_harmed_over_002":int(np.sum(matrix[:,3]-matrix[:,0]<-.020000000001)),
            "n_purged_fit_helped_over_002":int(np.sum(matrix[:,3]-matrix[:,0]>.020000000001)),
        }
        if results[policy]["n_lambda_changed"] != stats["n_changed_frozen_source_lambda"]:
            raise AssertionError("selector-change count mismatch")

    return {
        "status":"V30_PASS_IMMUTABLE_49_RECORD_CROSSED_REANALYSIS",
        "original_v29_sha256":digest,
        "no_new_target_data_or_CV_selection":True,
        "no_causal_leakage_or_independent_patients_claim":True,
        "n_groups":49,
        "results":results,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("original_v29_artifact_zip",type=Path)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    result=audit_archive(x.original_v29_artifact_zip)
    body=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(body)
    print(json.dumps({
        "status":result["status"],
        "original_v29_sha256":result["original_v29_sha256"],
        "macro_summary":{k:v["mean_contrast"] for k,v in result["results"].items()},
        "audit_sha256":hashlib.sha256(body.encode()).hexdigest(),
    },indent=2))

if __name__=="__main__":
    main()
