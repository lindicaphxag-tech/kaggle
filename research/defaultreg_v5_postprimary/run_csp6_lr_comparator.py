"""Post-primary CSP6+LR comparator for the completed v5 cohorts.

Evidence role:
    POST_PRIMARY_CLASSICAL_COMPARATOR_ONLY

The v5 target outcomes are already known. This script cannot create a new
prospective claim. It reuses the authoritative v5 loader/preprocessing and the
fixed covariance-form CSP6 + LogisticRegression(C=1) design that existed in
the closed, unexecuted PR #98.

No target statistic or target label is used for CSP/LR fitting.
Target score vectors are written and hashed before target AUC evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh as generalized_eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn
import moabb

HERE=Path(__file__).resolve()
V5_DIR=HERE.parents[1]/"defaultreg_v5"
sys.path.insert(0,str(V5_DIR))
import run_v5_predata_panel as v5  # noqa: E402

N_CSP=6


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def csp_fit(C,y,n_components=N_CSP):
    C=np.asarray(C,float);y=np.asarray(y,int)
    C0=np.mean(C[y==0],axis=0)
    C1=np.mean(C[y==1],axis=0)
    total=C0+C1
    eps=max(float(np.trace(total))/len(total),1e-20)*1e-8
    vals,W=generalized_eigh(C1,total+eps*np.eye(len(total)))
    order=np.argsort(np.abs(vals-.5))[::-1]
    k=min(int(n_components),len(order))
    idx=order[:k]
    return W[:,idx],vals[idx]


def csp_features(C,W):
    C=np.asarray(C,float);W=np.asarray(W,float)
    proj=np.einsum("pi,nij,jq->npq",W,C,W)
    var=np.diagonal(proj,axis1=1,axis2=2)
    var=np.maximum(var,np.finfo(float).tiny)
    var=var/np.sum(var,axis=1,keepdims=True)
    return np.log(var)


def freeze_csp_scores(C,y,g,sessions,subjects,out_file:Path):
    score_rows=[]
    row_indices=[]
    subject_rows=[]
    session_rows=[]
    fold_receipts=[]
    for sid in subjects:
        tr=g!=sid
        te=~tr
        if sid in set(g[tr]):
            raise AssertionError("outer target leakage")
        W,eigs=csp_fit(C[tr],y[tr],N_CSP)
        Ftr=csp_features(C[tr],W)
        Fte=csp_features(C[te],W)
        clf=LogisticRegression(
            C=1.0,solver="lbfgs",max_iter=3000
        )
        clf.fit(Ftr,y[tr])
        score=np.asarray(clf.decision_function(Fte),dtype=np.float64)
        idx=np.flatnonzero(te)
        score_rows.append(score)
        row_indices.append(idx)
        subject_rows.append(np.full(len(idx),sid,dtype=np.int64))
        session_rows.append(np.asarray(sessions[te],dtype="U32"))
        fold_receipts.append({
            "target_subject":int(sid),
            "source_subjects":[int(x) for x in sorted(set(g[tr]))],
            "source_trials":int(np.sum(tr)),
            "target_trials":int(np.sum(te)),
            "n_csp_components":int(W.shape[1]),
            "generalized_eigenvalues":list(map(float,eigs)),
            "target_information_used_for_fit":False,
        })

    score=np.concatenate(score_rows)
    row_index=np.concatenate(row_indices)
    subject=np.concatenate(subject_rows)
    session=np.concatenate(session_rows)
    np.savez_compressed(
        out_file,
        row_index=row_index,
        subject=subject,
        session=session,
        score=score,
        method=np.asarray(["CSP6_LogisticRegression"],dtype="U48"),
    )
    return fold_receipts,sha256(out_file)


def evaluate(score_file:Path,y,g,sessions,subjects):
    z=np.load(score_file,allow_pickle=False)
    row=np.asarray(z["row_index"],int)
    score=np.asarray(z["score"],float)
    ss=np.asarray(z["subject"],int)
    ses=np.asarray(z["session"]).astype(str)
    if len(row)!=len(score) or len(row)!=len(ss):
        raise ValueError("score bundle denominator drift")
    if len(set(row.tolist()))!=len(row):
        raise ValueError("duplicate target score row")
    yy=np.asarray(y)[row]
    gg=np.asarray(g)[row]
    if not np.array_equal(gg,ss):
        raise ValueError("subject identity drift in frozen score bundle")

    per=[]
    for sid in subjects:
        m=ss==sid
        if not np.any(m): raise ValueError(f"missing target {sid}")
        auc=float(roc_auc_score(yy[m],score[m]))
        sess={}
        for s in sorted(set(ses[m].tolist())):
            q=m&(ses==s)
            if len(set(yy[q].tolist()))==2:
                sess[str(s)]=float(roc_auc_score(yy[q],score[q]))
        per.append({
            "target_subject":int(sid),
            "target_trials":int(np.sum(m)),
            "auc":auc,
            "session_auc":sess,
        })
    return per


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",required=True,
        choices=["BNCI2014_002","BNCI2015_001"]
    )
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    start=time.monotonic()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ds,subjects,ch,C,y,g,sessions,runs,receipt,raw_files=v5.load_dataset(
        a.dataset,Path(a.data_root)
    )
    if set(np.unique(g))!=set(subjects):
        raise ValueError("participant denominator drift")

    score_path=out/"csp6_lr_target_scores_before_auc.npz"
    folds,score_sha=freeze_csp_scores(
        C,y,g,sessions,subjects,score_path
    )

    # Only after the immutable score bundle exists do target labels enter AUC.
    per=evaluate(score_path,y,g,sessions,subjects)
    macro=float(np.mean([r["auc"] for r in per]))
    result={
        "status":"POST_PRIMARY_CSP6_LR_COMPARATOR_COMPLETE",
        "evidence_role":"post_primary_classical_comparator_only",
        "dataset":a.dataset,
        "dataset_code":ds.code,
        "subjects":list(map(int,subjects)),
        "n_subjects":int(len(subjects)),
        "native_channels":list(map(str,ch)),
        "n_channels":int(len(ch)),
        "task":"right_hand_vs_feet",
        "band_hz":[8,30],
        "epoch_seconds":[0,5],
        "resample_hz":160.0,
        "target_information_track":"Z0",
        "target_labels_used_for_fit":False,
        "score_freeze_precedes_target_auc":True,
        "method":"CSP6_LogisticRegression",
        "n_csp_components":N_CSP,
        "log_normalized_variance_features":True,
        "logistic_regression_C":1.0,
        "participant_receipt":receipt,
        "raw_files":raw_files,
        "score_bundle_sha256":score_sha,
        "macro_auc":macro,
        "per_subject":per,
        "fold_receipts":folds,
        "environment":{
            "python":platform.python_version(),
            "numpy":np.__version__,
            "scipy":scipy.__version__,
            "sklearn":sklearn.__version__,
            "moabb":moabb.__version__,
        },
        "wall_seconds":round(time.monotonic()-start,2),
        "claim_boundary":"Comparator extension after v5 outcomes; no prospective claim upgrade.",
    }
    p=out/(a.dataset+"_postprimary_csp6_lr.json")
    p.write_text(json.dumps(result,indent=2,sort_keys=True))
    receipt_out={
        "dataset":a.dataset,
        "result_sha256":sha256(p),
        "score_bundle_sha256":score_sha,
        "n_subjects":len(subjects),
        "total_trials":int(len(y)),
        "macro_auc":macro,
        "status":"POST_PRIMARY_CSP_COMPARATOR_RECEIPT",
    }
    (out/"receipt.json").write_text(
        json.dumps(receipt_out,indent=2,sort_keys=True)
    )
    print(json.dumps(receipt_out,indent=2))


if __name__=="__main__":
    main()
