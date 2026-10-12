#!/usr/bin/env python3
"""NEW public independent acquisition EEG cohort: fixed original source-only
tangent-LDA, source macro-AUC vs held-source class-centred covariance and
genuinely predictive source-mean Gaussian validation. BNCI2014_004, 9 people.

The protocol in this directory was committed BEFORE first run. This is NOT
an assertion of independent public preregistration, accepted performance,
clinical external validity or world-wide absence of previous exposure.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from moabb.datasets import BNCI2014_004
import original_frozen_runner as old
from sklearn.metrics import roc_auc_score
from run_proper_predictive_posthoc import select_proper_source_nll

DATASET="BNCI2014_004"
SUBJECTS=tuple(range(1,10))
EVENTS=("left_hand","right_hand")
FULL_GRID=tuple(map(float,old.FULL_GRID))
PROTOCOL=Path(__file__).with_name("BNCI2014_004_GEOMETRY_SHIFT_PROTOCOL_20261012.md")
OUTPUT_CHOICES="SOURCE_ONLY_LAMBDA_CHOICES_SHA_LOCK_BEFORE_TARGET_EVALUATION.json"


def sha(b):return hashlib.sha256(b).hexdigest()


def load_public():
    paradigm=old.MotorImagery(n_classes=2,events=list(EVENTS),fmin=8,fmax=30,
                              resample=160,baseline=None,reject_by_annotation=False)
    ds=BNCI2014_004()
    covs=[];labels=[];grp=[];session_ids=[];receipt=[]
    expected_channels=None
    for sid in SUBJECTS:
        X,y,meta=paradigm.get_data(dataset=ds,subjects=[sid])
        y=np.asarray(y)
        if set(y)!=set(EVENTS):
            raise RuntimeError(f"Missing first-run frozen classes {sid}:{set(y)}")
        binary=np.asarray(y==EVENTS[1],int)
        if min(np.bincount(binary))<20:
            raise RuntimeError(f"Public EEG class balance failure {sid}")
        if expected_channels is None:expected_channels=int(X.shape[1])
        if X.shape[1]!=expected_channels:
            raise RuntimeError("Participant EEG channel mismatch")
        if not np.isfinite(X).all():
            raise RuntimeError("Non-finite new EEG features")
        if 'session' not in meta.columns or len(meta)!=len(binary):
            raise RuntimeError('No complete session labels; no per-session diagnostic')
        sess=np.asarray(meta['session'].astype(str))
        if len(set(sess))<2:raise RuntimeError('Expected multiple real acquisition sessions')
        session_ids.extend([str(sid)+'/'+q for q in sess])
        C=old.covariances(X)
        covs.append(C);labels.append(binary);grp.extend([sid]*len(binary))
        receipt.append({"subject":sid,"trials":len(binary),
                        "class0":int(np.sum(binary==0)),"class1":int(np.sum(binary==1)),
                        "eeg_channels":int(X.shape[1]),
                        "samples_per_trial":int(X.shape[-1]),
                        "sessions":{str(q):int(np.sum(sess==q)) for q in sorted(set(sess))}})
        print("PUBLIC_BNCI2014_004_RECEIPT",sid,"trials",len(binary),flush=True)
        del X
    # BNCI2014_004 BCI IV 2b uses C3,Cz,C4 bipolar EEG; exclude EOG.
    if expected_channels!=3:
        raise RuntimeError(f'Frozen THREE-EEG bipolar channel contract mismatch {expected_channels}')
    return np.concatenate(covs),np.concatenate(labels),np.asarray(grp),np.asarray(session_ids),receipt


def exact_first_change(grid,centred,proper,baseline):
    grid=np.asarray(grid,float)
    c0=np.asarray(centred,float)
    p=np.asarray(proper,float)-c0
    if np.min(p)<-1e-8:raise RuntimeError("Exact nonnegative penalty violated")
    at=np.where(np.isclose(grid,baseline,rtol=0,atol=1e-12))[0]
    if len(at)!=1:raise RuntimeError("Historical minimizer missing")
    i=at[0]
    roots=[]
    for j in range(len(grid)):
        den=p[i]-p[j]
        if den>1e-12:
            cross=(c0[j]-c0[i])/den
            if -1e-9<=cross<=1+1e-9:roots.append(max(0.,float(cross)))
    return min(roots) if roots else None


def run(outdir):
    if not PROTOCOL.exists():
        raise RuntimeError("Precommitted protocol not available")
    C,y,g,session_ids,participants=load_public()
    if set(np.unique(g))!=set(SUBJECTS):
        raise RuntimeError("Missing independent outer subject")
    source_choices=[]
    for sid in SUBJECTS:
        outer=(g==sid)
        source=~outer
        # CRITICAL: source_path and corrected predictor each accept ONLY
        # the eight source people's features/labels/subject IDs.
        oldpath=old.source_path(C[source],y[source],g[source])
        allowed=oldpath["feasible_full_lambdas"]
        lam_proper,proper_path=select_proper_source_nll(
            C[source],y[source],g[source],allowed)
        grid=sorted(set(map(float,allowed)))
        c0=[float(oldpath["criteria_by_lambda"][str(x)]["covariance_mean_nll"]) for x in grid]
        c1=[float(proper_path[str(x)]) for x in grid]
        threshold=exact_first_change(grid,c0,c1,oldpath["lambda_covariance_full"])
        chosen={"macro":float(oldpath["lambda_source_macro"]),
                "historical_class_centred":float(oldpath["lambda_covariance_full"]),
                "true_source_mean_predictive":float(lam_proper)}
        if not all(v in grid for v in chosen.values()):
            raise AssertionError("Source-only feasible grid selection mismatch")
        i=grid.index(float(lam_proper))
        shift=float(c1[i]-c0[i])
        if shift<0:raise AssertionError("Mean-shift penalty violated")
        source_choices.append({
            "outer_subject":sid,
            "source_people":list(map(int,sorted(set(g[source])))),
            "n_source_trials":int(np.sum(source)),
            "selected_lambda":chosen,
            "first_change_alpha_from_class_centred":threshold,
            "source_penalty_at_proper_lambda":shift,
            "source_feasible_lambda_grid":grid,
            "source_class_centred_log_scores":c0,
            "source_proper_predictive_log_scores":c1,
            "no_outer_target_data_in_source_selector":True})
        print("PRE_TARGET_SELECTION_LOCKING_SUBJECT",sid,chosen,flush=True)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    pretarget=(json.dumps(source_choices,sort_keys=True,indent=2)+"\n").encode()
    (out/OUTPUT_CHOICES).write_bytes(pretarget)
    source_sha=sha(pretarget)
    print("FROZEN_BEFORE_TARGET_LABEL_EVALUATION_SOURCE_MANIFEST_SHA256",source_sha,flush=True)
    # Only here use outer subject labels and evaluate their frozen scores.
    results=[]
    for record in source_choices:
        sid=record["outer_subject"];mask=(g==sid);tr=~mask
        inv=old.reference(C[tr])
        Xtr=old.tangent(C[tr],inv)
        Xte=old.tangent(C[mask],inv)
        state=old.spectral_state(Xtr,y[tr])
        scores={name:old.scalar_scores(state,Xte,lam) for name,lam in record["selected_lambda"].items()}
        for kind in ("LW","OAS"):
            z,shrinkage=old.analytic_scores(Xtr,y[tr],Xte,kind)
            scores[kind]=z
        auc={key:float(roc_auc_score(y[mask],v)) for key,v in scores.items()}
        held_sessions=session_ids[mask]
        by_session={}
        for ss in sorted(set(held_sessions)):
            mm=(held_sessions==ss)
            if len(set(y[mask][mm]))!=2:raise RuntimeError('Single-class held target session: '+ss)
            by_session[ss]={key:float(roc_auc_score(y[mask][mm],v[mm])) for key,v in scores.items()}
        results.append({"subject":sid,"test_trials":int(np.sum(mask)),
                        "per_session_auc":by_session,
                        "n_classes":int(len(set(y[mask]))),
                        "auc":auc,
                        "source_penalty_at_proper_lambda":record["source_penalty_at_proper_lambda"]})
    means={key:float(np.mean([row["auc"][key] for row in results])) for key in results[0]["auc"]}
    diff=np.asarray([x["auc"]["macro"]-x["auc"]["true_source_mean_predictive"] for x in results])
    rng=np.random.default_rng(20261012)
    ci=np.quantile(diff[rng.integers(0,len(diff),size=(30000,len(diff)))].mean(1),[.025,.975])
    summary={
      "status":"NEW_CROSS_GEOMETRY_BNCI2014_004_PROTOCOL_FIRST_EEG_RUN_COMPLETE",
      "dataset":DATASET,"n_outer_people":len(SUBJECTS),
      "all_participants_retained":True,
      "n_total_trials":int(len(y)),
      "n_channels":int(C.shape[1]),
      "target_information_track":"Z0",
      "source_lambda_choices_sha256_before_target_outcomes":source_sha,
      "pre_run_protocol_sha256":sha(PROTOCOL.read_bytes()),
      "github_source_commit":os.getenv("GITHUB_SHA","UNKNOWN_LOCAL"),
      "mean_AUC":means,
      "macro_minus_proper_mean_AUC":float(diff.mean()),
      "macro_minus_proper_person_bootstrap95_DESCRIPTIVE_CONDITIONAL":ci.tolist(),
      "macro_harmed_gt_0_02_vs_proper":int(np.sum(diff<-0.020000000001)),
      "proper_harmed_gt_0_02_vs_macro":int(np.sum(diff>0.020000000001)),
      "source_penalty_positive_n":sum(x["source_penalty_at_proper_lambda"]>0 for x in source_choices),
      "within_person_session_heterogeneity":{
        str(name):float(np.std([row["per_session_auc"][ss][name]
          for row in results for ss in row["per_session_auc"]]))
          for name in means},
      "n_session_cells":sum(len(row["per_session_auc"]) for row in results),
      "first_change_alpha_median":float(np.median([x["first_change_alpha_from_class_centred"] for x in source_choices
             if x["first_change_alpha_from_class_centred"] is not None])),
      "new_confirmatory_p_values":None,
      "no_global_preregistration_claim":True,
      "participant_receipts":participants,"per_subject_AUC":results,
      "caveat":"Protocol was committed before this run's results; no proof of global historical nonexposure. Only nine people, one acquisition site, overlapping outer SOURCE pools."
    }
    (out/"BNCI2014_004_SOURCE_ONLY_EXTERNAL_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("participant_receipts","per_subject_AUC")},indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    run(args.out)
