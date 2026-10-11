#!/usr/bin/env python3
"""Post-outcome comparison: true source-mean predictive NLL vs frozen EEG baselines.

All method choices use SOURCE labels/features only. Score-family target AUC
is read from the ORIGINAL frozen outcome JSON only AFTER all source choices
are written and SHA-256 hashed. NOT a new confirmatory experiment.
"""
import argparse,hashlib,io,json
from pathlib import Path
import zipfile
import numpy as np
import original_frozen_runner as old

ART_SHA="cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842"
RESULT="defaultreg_frontier_Shin2017A/Shin2017A_frontier_confirmatory.json"

def select_proper_source_nll(C,y,g,original_feasible):
    people=tuple(sorted(set(map(int,g))))
    allowed=tuple(sorted(set(map(float,original_feasible))))
    assert len(people)>=8 and len(allowed)>=2
    assert all(k in old.FULL_GRID for k in allowed)
    values={k:[] for k in allowed}
    for held in old._held_groups(people):
        va=np.isin(g,held)
        tr=~va
        if not va.any() or not tr.any():raise ValueError("Bad source split")
        inv=old.reference(C[tr])
        train=old.tangent(C[tr],inv)
        valid=old.tangent(C[va],inv)
        state=old.spectral_state(train,y[tr])
        _,_,d,mid,_=state
        means=np.stack([mid-d/2,mid+d/2])
        for sid in held:
            m=(g[va]==sid)
            yy=y[va][m]
            if set(np.unique(yy))!={0,1}:raise ValueError("Held SOURCE participant missing class")
            # Critical: means from inner TRAIN source; NEVER validation class means.
            R=valid[m]-means[yy]
            for lam in allowed:
                q=old.predictive_nll(state,R,lam)
                if not np.isfinite(q):raise ValueError("Nonfinite predictive source score")
                values[lam].append(float(q))
    if any(len(v)!=len(people) for v in values.values()):
        raise AssertionError("Missing held SOURCE participant scores")
    means={str(k):float(np.mean(values[k])) for k in allowed}
    selected=min(allowed,key=lambda k:(means[str(k)],k))
    return float(selected),means

def run(archive,out):
    blob=Path(archive).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=ART_SHA:raise AssertionError("Original score ZIP SHA failed")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        frozen=json.loads(z.read(RESULT))
    assert frozen["dataset"]=="Shin2017A" and frozen["n_subjects"]==29
    original={int(f["target_subject"]):f for f in frozen["folds"]}
    C,y,g,*_=old.load_dataset("Shin2017A")
    rows=[]
    for sid in range(1,30):
        source=g!=sid
        f=original[sid]
        lam,path=select_proper_source_nll(C[source],y[source],g[source],
                                         f["source_path"]["feasible_full_lambdas"])
        rows.append({"subject":sid,"new_source_proper_predictive_lambda":lam,
                     "source_predictive_log_score_by_lambda":path,
                     "original_class_centered_lambda":f["source_path"]["lambda_covariance_full"],
                     "no_outer_target_info_for_selection":True})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sel_bytes=json.dumps(rows,sort_keys=True,indent=2).encode()
    (out/"SOURCE_CHOICES_FROZEN_BEFORE_TARGET_OUTCOME_JOIN.json").write_bytes(sel_bytes)
    locked_sha=hashlib.sha256(sel_bytes).hexdigest()
    print("SOURCE_ONLY_PREDICTIVE_LAMBDA_VECTOR_LOCK",locked_sha,flush=True)
    # The original target path is read ONLY here, never during source selection.
    paired=[]
    for rec in rows:
        sid=rec["subject"]
        f=original[sid]
        lam=rec["new_source_proper_predictive_lambda"]
        target_path=f["target_auc_path_by_lambda"]
        key=str(float(lam))
        if key not in target_path:raise ValueError("Frozen target score path lacks chosen lambda")
        auc=float(target_path[key])
        a=f["auc"]
        paired.append({"subject":sid,"lambda_proper_predictive":lam,
                       "auc_proper_predictive_FROM_FROZEN_PATH":auc,
                       "auc_frozen_class_centered":float(a["source_cov_nll_full_domain_LDA"]),
                       "auc_frozen_source_macro":float(a["source_macro_full_domain_LDA"]),
                       "delta_source_macro_minus_proper":float(a["source_macro_full_domain_LDA"]-auc),
                       "delta_class_centered_minus_proper":float(a["source_cov_nll_full_domain_LDA"]-auc)})
    gains=np.array([r["delta_source_macro_minus_proper"] for r in paired])
    cent=np.array([r["delta_class_centered_minus_proper"] for r in paired])
    result={"status":"POST_OUTCOME_REAL_EEG_SENSITIVITY_NOT_CONFIRMATORY",
            "source_only_lambda_lock_sha256":locked_sha,
            "original_archive_sha256":ART_SHA,
            "n":29,"mean_source_macro_minus_proper_AUC":float(gains.mean()),
            "mean_class_centered_minus_proper_AUC":float(cent.mean()),
            "harm_gt_0_02_vs_proper":int(np.sum(gains<-.020000000001)),
            "proper_lambda_differs_from_original_class_centered_n":sum(
                r["new_source_proper_predictive_lambda"]!=r["original_class_centered_lambda"] for r in rows),
            "no_new_confirmatory_p_value":True,
            "per_subject":paired}
    (out/"POSTHOC_PROPER_PREDICTIVE_VS_FROZEN_RESULTS.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="per_subject"},indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-zip",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.original_zip,a.out)
