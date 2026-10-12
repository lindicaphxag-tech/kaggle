#!/usr/bin/env python3
"""V29: 49-group factorial frozen model-fit x source-selector EEG diagnostic.

EXPOSED DATA, POST OUTCOME. Crossed SOFTWARE conditions do not imply a
patient-level causal effect. All source lambdas are reused from two frozen
artifacts; no target labels, outcomes or target moments choose lambdas.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import pathlib
import sys
import zipfile
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
sys.path.insert(0,str(ROOT/"research"/"defaultreg_bspc_v13"/"src"))
import original_frozen_runner as old
from run_liu2024_purged_lopo_v26 import load_and_group, get_representative_arrays

OLD_SRC="FIFTY_SOURCE_ONLY_CHOICES_LOCKED_BEFORE_ANY_TARGET_AUC.json"
OLD_OUT="FROZEN_AFTER_SOURCE_LOCK_50PATIENT_LOPO_RESULTS.json"
NEW_SRC="PURGED_RECORD_IDENTITY_SOURCE_POLICIES_BEFORE_TARGET_AUC.json"
NEW_OUT="PURGED_LIU2024_RECORD_INDEPENDENT_RETROSPECTIVE_RESULTS.json"
NEW_GROUP="FULL_COHORT_EXACT_SIGNAL_GROUPS_AFTER_GLOBAL_AUDIT.json"
ORIGINAL_SOURCE_SHA="d56f6f7169689ad80715fc94635a86fce19e936bc1f4536fc2f7845468323227"
PROTOCOL=ROOT/"research"/"defaultreg_bspc_v13"/"V29_FROZEN_CROSSED_FIT_SELECTOR_RETROSPECTIVE_PROTOCOL.md"
METHODS=("source_macro","proper_source_mean_gaussian","historical_class_centred")

def read_zip_member(path, ending):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(ending)]
        if len(names)!=1:raise AssertionError(f"missing/ambiguous {ending}")
        raw=z.read(names[0])
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def frozen_inputs(original_zip,purged_zip):
    so,sho=read_zip_member(original_zip,OLD_SRC)
    oo,hoo=read_zip_member(original_zip,OLD_OUT)
    sn,shn=read_zip_member(purged_zip,NEW_SRC)
    on,hnn=read_zip_member(purged_zip,NEW_OUT)
    gr,gh=read_zip_member(purged_zip,NEW_GROUP)
    if (sho!=ORIGINAL_SOURCE_SHA
        or oo["source_choices_SHA256_pre_target_evaluation"]!=sho
        or on["source_policy_SHA_before_held_record_auc"]!=shn
        or on["global_record_identity_SHA_before_selection"]!=gh
        or on["status"]!="RETROSPECTIVE_LIU2024_SIGNAL_IDENTITY_PURGED_LOPO_COMPLETE"
        or gr["duplicate_groups"]!=[[4,5]]
        or len(so)!=50 or len(sn)!=49):
        raise AssertionError("two independent source locks or EEG release identity not verified")
    by_old={int(k["outer_subject"]):k for k in so}
    by_new={int(k["target_recording_representative"]):k for k in sn}
    old_auc={int(k["held_patient"]):k["auc"] for k in oo["participants"]}
    new_auc={int(k["representative_recording"]):k["auc"] for k in on["target_outcomes"]}
    reps=sorted(map(int,gr["representative_ids"]))
    if (set(reps)!=set(range(1,51))-{5}
        or set(by_new)!=set(reps) or set(new_auc)!=set(reps)
        or set(by_old)!=set(range(1,51)) or set(old_auc)!=set(range(1,51))):
        raise AssertionError("source or target recording count drift")
    return (by_old,by_new,old_auc,new_auc,reps,
            {"old_source_SHA256":sho,"old_outcome_SHA256":hoo,
             "purged_source_SHA256":shn,"purged_outcome_SHA256":hnn,
             "purged_record_identity_SHA256":gh})

def fit_and_target_scores(C, y, g, target, source_mask, lambdas):
    test=g==target
    if len(np.unique(g[source_mask]))!=len(set(g[source_mask])):
        # np.unique is deterministic; sanity is tautological and retained
        # only as a shape/access assertion.
        raise AssertionError("source-ID array corrupted")
    if np.any(source_mask & test):
        raise AssertionError("outer target enters fit")
    inv=old.reference(C[source_mask])
    Xfit=old.tangent(C[source_mask],inv)
    Xtest=old.tangent(C[test],inv)
    spectral=old.spectral_state(Xfit,y[source_mask])
    if set(map(int,y[test]))!={0,1}:
        raise ValueError("target recording missing a class")
    return {k:float(roc_auc_score(y[test],old.scalar_scores(spectral,Xtest,float(v))))
            for k,v in lambdas.items()}

def run(original_zip,purged_zip):
    if not PROTOCOL.is_file():
        raise RuntimeError("fixed V29 retrospective protocol not present")
    by_old,by_new,original_auc,purged_auc,reps,sha=frozen_inputs(original_zip,purged_zip)
    identity,mapping,C,y,g=load_and_group()
    if identity["representative_ids"]!=reps or identity["duplicate_groups"]!=[[4,5]]:
        raise AssertionError("EEG release identity differs from both frozen artifacts")
    if len(y)!=2000 or len(C)!=2000 or len(set(g))!=50:
        raise AssertionError("dataset release denominator drift")
    reps_mask=np.isin(g,reps)
    lines=[]
    for target in reps:
        original_mask=g!=target
        purged_mask=reps_mask&(g!=target)
        if (original_mask.sum()!=1960 or purged_mask.sum()!=1920
            or len(set(g[purged_mask]))!=48):
            raise AssertionError("invalid paired source data scope")
        old_lam=by_old[target]["chosen_lambda"]
        new_lam=by_new[target]["chosen_lambdas"]
        lam_A={k:float(old_lam[k]) for k in METHODS}
        lam_B={k:float(new_lam[k]) for k in METHODS}
        if (by_new[target]["source_representative_subjects"]!=sorted(set(map(int,g[purged_mask])))
            or not by_new[target]["source_held_record_group_disjoint"]):
            raise AssertionError("purged training receipt inconsistent with refit")
        # Four test-condition cross products; one spectral fitting per
        # old/purged model, reused for BOTH already-frozen choices.
        old_score_lam={f"{k}_A":v for k,v in lam_A.items()}
        old_score_lam.update({f"{k}_B":v for k,v in lam_B.items()})
        new_score_lam=old_score_lam
        old_scores=fit_and_target_scores(C,y,g,target,original_mask,old_score_lam)
        new_scores=fit_and_target_scores(C,y,g,target,purged_mask,new_score_lam)
        crossed={}
        for k in METHODS:
            AA=old_scores[k+"_A"]
            AB=old_scores[k+"_B"]
            BA=new_scores[k+"_A"]
            BB=new_scores[k+"_B"]
            if abs(AA-float(original_auc[target][k]))>1e-10:
                raise AssertionError(f"original frozen 50-ID target AUC reproduction failed {target}/{k}")
            if abs(BB-float(purged_auc[target][k]))>1e-10:
                raise AssertionError(f"49-group refitted frozen target AUC reproduction failed {target}/{k}")
            interaction=BB-BA-AB+AA
            if abs((BB-AA)-((AB-AA)+(BB-AB)))>1e-12:
                raise AssertionError("factorial decomposition identity failed")
            if abs((BB-AA)-((BA-AA)+(BB-BA)))>1e-12:
                raise AssertionError("factorial decomposition identity failed")
            crossed[k]={
                "AA_old_fit_old_lambda":AA,
                "AB_old_fit_purged_lambda":AB,
                "BA_purged_fit_old_lambda":BA,
                "BB_purged_fit_purged_lambda":BB,
                "selected_lambda_changed":bool(abs(lam_A[k]-lam_B[k])>1e-12),
                "lambda_A":lam_A[k],"lambda_B":lam_B[k],
                "selector_change_on_old_fit_AB_minus_AA":AB-AA,
                "fit_change_at_old_lambda_BA_minus_AA":BA-AA,
                "selector_change_on_purged_fit_BB_minus_BA":BB-BA,
                "fit_change_at_new_lambda_BB_minus_AB":BB-AB,
                "model_selector_interaction":interaction,
                "total_purged_BB_minus_original_AA":BB-AA,
            }
        lines.append({"held_representative":int(target),
                      "original_fit_source_trials":int(original_mask.sum()),
                      "purged_fit_source_trials":int(purged_mask.sum()),
                      "target_trials":int((g==target).sum()),
                      "method_crossed_AUC":crossed})
        print("V29_FROZEN_FOUR_CELL_TARGET_VERIFIED",target,flush=True)
    if len(lines)!=49:raise AssertionError("49-group denominator lost")
    stats={}
    for method in METHODS:
        values=[v["method_crossed_AUC"][method] for v in lines]
        names=("AA_old_fit_old_lambda","AB_old_fit_purged_lambda",
               "BA_purged_fit_old_lambda","BB_purged_fit_purged_lambda",
               "selector_change_on_old_fit_AB_minus_AA",
               "fit_change_at_old_lambda_BA_minus_AA",
               "selector_change_on_purged_fit_BB_minus_BA",
               "fit_change_at_new_lambda_BB_minus_AB",
               "model_selector_interaction",
               "total_purged_BB_minus_original_AA")
        stats[method]={
            "participant_macro_mean":{n:float(np.mean([v[n] for v in values])) for n in names},
            "n_changed_frozen_source_lambda":sum(v["selected_lambda_changed"] for v in values),
            "n_BB_harmed_gt002_against_AA":sum(v["total_purged_BB_minus_original_AA"] < -.020000000001 for v in values),
            "n_BB_benefited_gt002_against_AA":sum(v["total_purged_BB_minus_original_AA"] > .020000000001 for v in values),
        }
    return {
        "status":"V29_RETROSPECTIVE_FROZEN_FOUR_CELL_REFIT_SELECTION_COMPLETE",
        "no_new_target_cohort_or_generalisation_confirmation":True,
        "not_causal_or_strict_physically_target_isolated_Z0":True,
        "all_49_original_and_purged_outcomes_reproduced":True,
        "n_content_disjoint_held_groups":49,
        "protocol_SHA256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "frozen_artifact_checksums":sha,
        "method_summary":stats,
        "held_record_crossed_responses":lines,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-zip",type=pathlib.Path,required=True)
    p.add_argument("--purged-zip",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    a=p.parse_args()
    result=run(a.original_zip,a.purged_zip)
    body=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(body)
    print(json.dumps({"status":result["status"],"n_groups":49,
        "summary":result["method_summary"],
        "report_sha256":hashlib.sha256(body.encode()).hexdigest()},indent=2),flush=True)

if __name__=="__main__":main()
