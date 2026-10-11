"""Label-free frozen score-order necessary bound for ROC-AUC changes.

Two score vectors with identical pairwise trial ordering MUST have identical
ROC-AUC for every fixed binary labeling. More generally, for n0 and n1
trials, abs(AUC_D-AUC_C) <= min(1,K_all/(n0*n1)), where K_all counts
all unordered trial pairs whose score order (including ties) differs.
This is a NECESSARY bound only, not prospective predictive inference.
"""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
import numpy as np
from scipy.stats import spearmanr

ZIP_SHA="cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842"
RECORD="defaultreg_frontier_Shin2017A/Shin2017A_frontier_confirmatory.json"


def changed_pairs(x,y):
    a,b=np.asarray(x,float),np.asarray(y,float)
    if a.ndim!=1 or a.shape!=b.shape or not(np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Score arrays differ in shape or contain nonfinite values")
    i,j=np.triu_indices(len(a),1)
    return int(np.sum(np.sign(a[i]-a[j])!=np.sign(b[i]-b[j]))),int(len(i))


def audit(path):
    blob=open(path,"rb").read()
    if hashlib.sha256(blob).hexdigest()!=ZIP_SHA:
        raise AssertionError("Original Shin ZIP changed")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        j=json.loads(z.read(RECORD))
        folds=sorted(j["folds"],key=lambda x:int(x["target_subject"]))
        if len(folds)!=29:raise AssertionError("Frozen subject count drifted")
        rows=[]
        for fold in folds:
            src=fold["score_freeze"]
            arr=z.read(src["score_file"])
            if hashlib.sha256(arr).hexdigest()!=src["score_sha256"]:
                raise AssertionError("Frozen score bytes changed")
            with np.load(io.BytesIO(arr),allow_pickle=False) as npz:
                d=np.asarray(npz["source_macro_full_domain_LDA"],float)
                c=np.asarray(npz["source_cov_nll_full_domain_LDA"],float)
            k,allpairs=changed_pairs(d,c)
            n0=int(fold["target_label_counts"]["class0"])
            n1=int(fold["target_label_counts"]["class1"])
            if n0*n1<=0 or n0+n1!=len(d):
                raise AssertionError("Trial class denominators mismatch")
            effect=float(fold["auc"]["source_macro_full_domain_LDA"]-
                         fold["auc"]["source_cov_nll_full_domain_LDA"])
            upper=min(1.0,k/(n0*n1))
            if abs(effect)>upper+1e-10:
                raise AssertionError("Frozen AUC change exceeds label-free ordering bound")
            rows.append({"subject":int(fold["target_subject"]), "changed_pairs":k,
                         "all_pairs":allpairs, "rank_change_fraction":k/allpairs,
                         "observed_delta_auc":effect, "auc_absolute_change_upper_bound":upper,
                         "score_spearman_descriptive":float(spearmanr(d,c).statistic)})
        frac=np.array([row["rank_change_fraction"] for row in rows])
        effect=np.array([row["observed_delta_auc"] for row in rows])
        med=np.median([row["score_spearman_descriptive"] for row in rows])
        if abs(float(effect.mean())-j["primary_comparisons"][0]["mean_delta_auc"])>1e-12:
            raise AssertionError("Frozen 29-subject effect arithmetic changed")
        return {"status":"PASS_ORIGINAL_FROZEN_SCORE_ORDER_NECESSARY_AUC_BOUND",
                "n_participants":len(rows),
                "mean_trial_order_change_fraction":float(frac.mean()),
                "min_trial_order_change_fraction":float(frac.min()),
                "max_trial_order_change_fraction":float(frac.max()),
                "median_score_spearman":float(med),
                "all_29_auc_bounds_passed":True,
                "rows":rows,
                "scope":"Post-outcome, label-free rank verification of original frozen scores; no new classifier, transfer benefit or causality claim"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("original_shin_zip")
    parser.add_argument("--out")
    args=parser.parse_args()
    result=audit(args.original_shin_zip)
    if args.out:
        open(args.out,"w").write(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
