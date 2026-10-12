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

ART_SHA="15137db055b54983c297d28b93e714836ff3711888c35bcdd3640eba05b1a6e1"
RESULT="defaultreg_frontier_BNCI2014_002/BNCI2014_002_frontier_confirmatory.json"

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
    # Extract only SOURCE-generated feasible grids. Target AUC fields remain
    # inaccessible until *after* all 14 lambda choices have been locked.
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        receipt=json.loads(z.read(RESULT))
    assert receipt["dataset"]=="BNCI2014_002" and receipt["n_subjects"]==14
    assert receipt["target_information_track"]=="Z0"
    source_candidates={int(f["target_subject"]):tuple(
        f["source_path"]["feasible_full_lambdas"]) for f in receipt["folds"]}
    original_centered_lambdas={int(f["target_subject"]):float(
        f["source_path"]["lambda_covariance_full"]) for f in receipt["folds"]}
    frozen_counts={int(f["participant"]):int(f["trials"])
                   for f in receipt["participant_receipt"]}
    expected_trials=int(receipt["trial_denominator"])
    del receipt
    C,y,g,*_=old.load_dataset("BNCI2014_002")
    assert set(np.unique(g))==set(range(1,15)) and len(y)==expected_trials
    assert all(int(np.sum(g==sid))==frozen_counts[sid] for sid in range(1,15))
    rows=[]
    for sid in range(1,15):
        source=g!=sid
        lam,path=select_proper_source_nll(C[source],y[source],g[source],
                                         source_candidates[sid])
        print("SOURCE_ONLY_SUBJECT_SELECTION",sid,"lambda",lam,flush=True)
        rows.append({"subject":sid,"new_source_proper_predictive_lambda":lam,
                     "source_predictive_log_score_by_lambda":path,
                     "original_class_centered_lambda":original_centered_lambdas[sid],
                     "no_outer_target_info_for_selection":True})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sel_bytes=json.dumps(rows,sort_keys=True,indent=2).encode()
    (out/"SOURCE_CHOICES_FROZEN_BEFORE_TARGET_OUTCOME_JOIN.json").write_bytes(sel_bytes)
    locked_sha=hashlib.sha256(sel_bytes).hexdigest()
    print("SOURCE_ONLY_PREDICTIVE_LAMBDA_VECTOR_LOCK",locked_sha,flush=True)
    # Re-open the ORIGINAL ZIP only after all source lambdas have been locked.
    # The archived per-person target AUC paths were unavailable to selection.
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        frozen=json.loads(z.read(RESULT))
    original={int(f["target_subject"]):f for f in frozen["folds"]}
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
            "n":len(rows),"mean_source_macro_minus_proper_AUC":float(gains.mean()),
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
