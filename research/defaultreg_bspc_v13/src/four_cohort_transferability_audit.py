"""Descriptive four-cohort transportability sign audit, not meta-inference.

The four datasets were NOT a random prespecified sample of all EEG cohorts.
The Shin2017A family is separately frozen; all posthoc aggregations here are
exploratory. Do not pool within-family Holm p values, trials, or source fits.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
THREE=ROOT/"evidence/verified_three_cohort_paired_effect_vectors_v8.json"
SHIN=ROOT/"paper_vnext/evidence/shin2017a_logged_receipt_v13.json"


def audit(three_path=THREE,shin_path=SHIN):
    a=json.loads(Path(three_path).read_text())
    s=json.loads(Path(shin_path).read_text())["execution_summary"]
    cohort={}
    for key,v in a["cohorts"].items():
        vals=np.asarray(v["primary"]["vs_covariance_nll"],float)
        if len(vals)!=len(v["participant_ids"]):
            raise AssertionError("Participant ID count mismatch: "+key)
        expected=float(v["vs_covariance_nll_mean_from_archived_vector"])
        if abs(float(np.mean(vals))-expected)>1e-10:
            raise AssertionError("Archived participant mean mismatch: "+key)
        cohort[key]={"n_participants":len(vals),"mean_delta_auc":expected,
                     "evidence_tier":v["evidence_tier"]}
    contrasts=s["primary_comparisons"]
    c=next(x for x in contrasts if x["baseline"]=="source_cov_nll_full_domain_LDA")
    if s["dataset"]!="Shin2017A" or s["n_subjects"]!=29:
        raise AssertionError("Shin family/denominator mismatch")
    mean=float(s["mean_auc_by_method"]["source_macro_full_domain_LDA"]-
               s["mean_auc_by_method"]["source_cov_nll_full_domain_LDA"])
    if abs(mean-c["mean_delta_auc"])>1e-10:
        raise AssertionError("Shin archived AUC arithmetic mismatch")
    cohort["Shin2017A_separate_family"]={
        "n_participants":29,"mean_delta_auc":mean,
        "evidence_tier":"separate_frozen_family_original_job_log"}
    keys=list(cohort)
    means=np.array([cohort[k]["mean_delta_auc"] for k in keys],float)
    positive=int(np.sum(means>0))
    if np.any(means==0):raise AssertionError("Tied dataset mean; sign test requires a separate tie policy")
    # This p-value is descriptive under an ASSUMED i.i.d. symmetric sign
    # model; the observed benchmark set was not randomly sampled.
    from math import comb
    n=len(means)
    extreme=min(positive,n-positive)
    p=min(1.0,2.0*sum(comb(n,j) for j in range(extreme+1))/2**n)
    return {
        "status":"POSTHOC_EQUAL_DATASET_WEIGHT_REPORT_NOT_POPULATION_CONFIRMATION",
        "cohorts":cohort,
        "equal_dataset_mean_delta_auc":float(means.mean()),
        "per_dataset_lodo_equal_weight_mean":{
            keys[i]:float(np.delete(means,i).mean()) for i in range(n)},
        "sign_positive_datasets":positive,
        "sign_total_datasets":n,
        "two_sided_sign_p_under_hypothetical_symmetric_iid_cohorts":float(p),
        "validity_warning":"Not a prespecified random sample of datasets; p=.125 is illustrative, not independent confirmation of universal EEG transfer.",
        "statistical_family_warning":"Each cohort's original Holm tests remain separate; no new pooled inference.",
    }


if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
