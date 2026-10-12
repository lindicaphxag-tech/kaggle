#!/usr/bin/env python3
"""Independent exact-artifact V18 post-outcome proper predictive EEG audit.

Takes two immutable GitHub Actions ZIPs:
  original Shin2017A archived score-grid artifact 11674238395;
  successfully rerun proper predictive comparator artifact 11722966174.
No model refitting or new independent cohort. The proper predictive
source-only lambda choices were SHA-locked before joining original outcomes.
Any bootstrap numbers are post-outcome descriptive conditional-on-source only.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ORIGINAL_ZIP_SHA="cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842"
PROPER_ZIP_SHA="d260a402bfbd9f4b8a49f6d0a47de766f71d7b399f1011039def302043d07467"
SOURCE_LOCK_SHA="fac20e7f1ec1e43f3fa01af8ce88e44869e80f624d9a10b6123a57a6687f9b5c"
ORIGINAL_RESULT="defaultreg_frontier_Shin2017A/Shin2017A_frontier_confirmatory.json"
CHOICES="SOURCE_CHOICES_FROZEN_BEFORE_TARGET_OUTCOME_JOIN.json"
PROPER_RESULT="POSTHOC_PROPER_PREDICTIVE_VS_FROZEN_RESULTS.json"


def read_zip(path,expected_hash):
    b=Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=expected_hash:
        raise AssertionError(f"ZIP SHA mismatch {path}")
    return zipfile.ZipFile(io.BytesIO(b))


def one_suffix(z,suffix):
    matches=[n for n in z.namelist() if n.endswith(suffix) and not n.startswith("__MACOSX/")]
    if len(matches)!=1:
        raise AssertionError(f"expected exactly 1 file ending {suffix}, got {matches}")
    return z.read(matches[0])


def audit(original_path,proper_path):
    with read_zip(original_path,ORIGINAL_ZIP_SHA) as original_zip, \
         read_zip(proper_path,PROPER_ZIP_SHA) as proper_zip:
        orig=json.loads(original_zip.read(ORIGINAL_RESULT))
        if orig["dataset"]!="Shin2017A" or orig["n_subjects"]!=29:
            raise AssertionError("Original dataset identity mismatch")
        fz={int(f["target_subject"]):f for f in orig["folds"]}
        locked_bytes=one_suffix(proper_zip,CHOICES)
        if hashlib.sha256(locked_bytes).hexdigest()!=SOURCE_LOCK_SHA:
            raise AssertionError("Executed source-only lambda lock digest mismatch")
        choices=json.loads(locked_bytes)
        report=json.loads(one_suffix(proper_zip,PROPER_RESULT))
        if report["status"]!="POST_OUTCOME_REAL_EEG_SENSITIVITY_NOT_CONFIRMATORY" or \
           report["source_only_lambda_lock_sha256"]!=SOURCE_LOCK_SHA or \
           report["original_archive_sha256"]!=ORIGINAL_ZIP_SHA:
            raise AssertionError("Postoutcome result metadata changed")
        if len(choices)!=29 or len(report["per_subject"])!=29:
            raise AssertionError("29 original subject decisions required")
        by_subject={int(z["subject"]):z for z in report["per_subject"]}
        source_lambdas={int(z["subject"]):float(z["new_source_proper_predictive_lambda"]) for z in choices}
        if set(by_subject)!=set(source_lambdas)!=set(range(1,30)):
            raise AssertionError("Missing subject row")
        if set(by_subject)!=set(range(1,30)) or set(source_lambdas)!=set(range(1,30)):
            raise AssertionError("Target subject set mismatch")
        if any(z["no_outer_target_info_for_selection"] is not True for z in choices):
            raise AssertionError("Source-information-budget flag missing")
        ds,dp,dc=[],[],[]
        # Independently compare the old and new *source validation* NLL
        # paths at exactly the SAME lambda and within the SAME grouped
        # training/validation geometry. Under the source-mean shift
        # identity the proper log score cannot be smaller than the
        # held validation person's class-centred covariance score.
        source_gap_at_proper_lambda=[]
        source_choice={int(z["subject"]):z for z in choices}
        for sid in range(1,30):
            z=by_subject[sid]
            f=fz[sid]
            lam=source_lambdas[sid]
            if abs(lam-.2)>1e-12:
                raise AssertionError(f"Shin proper lambda unexpectedly changed at subject {sid}")
            if abs(float(z["lambda_proper_predictive"])-lam)>1e-12:
                raise AssertionError("Source-choice/outcome lambda mismatch")
            if lam not in [float(v) for v in f["source_path"]["feasible_full_lambdas"]]:
                raise AssertionError("Source-feasibility violated")
            path_auc=float(f["target_auc_path_by_lambda"][str(float(lam))])
            if abs(float(z["auc_proper_predictive_FROM_FROZEN_PATH"])-path_auc)>1e-12:
                raise AssertionError("Target outcome mismatch")
            marco=float(f["auc"]["source_macro_full_domain_LDA"])
            cent=float(f["auc"]["source_cov_nll_full_domain_LDA"])
            if abs(float(z["auc_frozen_source_macro"])-marco)>1e-12 or \
               abs(float(z["auc_frozen_class_centered"])-cent)>1e-12:
                raise AssertionError("Old frozen comparator altered")
            if abs(float(z["delta_source_macro_minus_proper"])-(marco-path_auc))>1e-12:
                raise AssertionError("Paired effect arithmetic wrong")
            if abs(float(z["delta_class_centered_minus_proper"])-(cent-path_auc))>1e-12:
                raise AssertionError("Paired class-centered effect arithmetic wrong")
            grid_score=source_choice[sid]["source_predictive_log_score_by_lambda"]
            key=str(float(lam))
            assert key in grid_score and key in f["source_path"]["criteria_by_lambda"]
            proper_source=float(grid_score[key])
            centred_source=float(f["source_path"]["criteria_by_lambda"][key]["covariance_mean_nll"])
            diff_source=proper_source-centred_source
            if not np.isfinite(diff_source) or diff_source < -1e-8:
                raise AssertionError(f"Nonnegative exact source-mean shift difference violated in {sid}: {diff_source}")
            source_gap_at_proper_lambda.append(diff_source)
            ds.append(marco);dp.append(path_auc);dc.append(cent)
        ds,dp,dc=map(np.asarray,(ds,dp,dc))
        delta=ds-dp
        cent_delta=dc-dp
        assert abs(float(np.mean(delta))-report["mean_source_macro_minus_proper_AUC"])<1e-12
        assert abs(float(np.mean(cent_delta))-report["mean_class_centered_minus_proper_AUC"])<1e-12
        # Conditioned on original fixed source pools: post-outcome bootstrap
        # must NOT be interpreted as full-population or independent-site CI.
        rng=np.random.default_rng(20261012)
        draws=rng.integers(0,29,size=(40000,29))
        boot=delta[draws].mean(axis=1)
        leave=[np.delete(delta,k).mean() for k in range(29)]
        return {
            "status":"PASS_29_SUBJECT_REAL_PROPER_NLL_POSTOUTCOME_ARTIFACT_AUDIT",
            "n":29,"original_score_ZIP_sha256":ORIGINAL_ZIP_SHA,
            "new_proper_ZIP_sha256":PROPER_ZIP_SHA,
            "source_only_choice_lock_sha256":SOURCE_LOCK_SHA,
            "proper_lambda_counts":{"0.2":29},
            "original_source_macro_mean_AUC":float(ds.mean()),
            "new_source_proper_predictive_mean_AUC":float(dp.mean()),
            "original_class_centred_mean_AUC":float(dc.mean()),
            "macro_minus_proper_mean_delta_AUC":float(delta.mean()),
            "class_centred_minus_proper_mean_delta_AUC":float(cent_delta.mean()),
            "mean_proper_minus_class_centred_SOURCE_score_at_lambda_0_2":float(np.mean(source_gap_at_proper_lambda)),
            "min_proper_minus_class_centred_SOURCE_score":float(np.min(source_gap_at_proper_lambda)),
            "max_proper_minus_class_centred_SOURCE_score":float(np.max(source_gap_at_proper_lambda)),
            "source_validation_nonnegative_mean_shift_identity_29_of_29":True,
            "macro_minus_proper_subjects_positive":int(np.sum(delta>0)),
            "macro_minus_proper_subjects_negative":int(np.sum(delta<0)),
            "macro_minus_proper_subjects_tied":int(np.sum(np.abs(delta)<1e-12)),
            "source_macro_harm_gt_0_02_vs_proper":int(np.sum(delta<-.020000000001)),
            "proper_harm_gt_0_02_vs_source_macro":int(np.sum(delta>.020000000001)),
            "exploratory_subject_bootstrap_95_conditional":np.quantile(boot,[.025,.975]).tolist(),
            "exploratory_leave_one_person_out_mean_range":[float(min(leave)),float(max(leave))],
            "new_confirmatory_test_performed":False,
            "new_independent_EEG_cohort_acquired":False,
            "warning":"Post-outcome data, highly overlapping source pools and frozen target score reuse; descriptive sensitivity only, not statistical equivalence or proof of safety."
        }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_zip")
    p.add_argument("proper_zip")
    p.add_argument("--out")
    a=p.parse_args()
    result=audit(a.original_zip,a.proper_zip)
    data=json.dumps(result,indent=2)+"\n"
    if a.out:
        Path(a.out).write_text(data)
    print(data)
