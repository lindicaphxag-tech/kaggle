#!/usr/bin/env python3
"""V17: archived 29-participant full-grid AUC FEASIBILITY envelope (post-outcome).

No EEG training, no new proper-predictive baseline, no target-safe policy.
The retrospective min/max are NOT trainable or prospective selectors; they
only bound potential AUC of methods that choose a source-feasible FULL_GRID
lambda for each participant using the exact archived fixed scoring family.
"""
import argparse,hashlib,io,json,pathlib,zipfile
import numpy as np

EXPECTED_ZIP="cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842"
NAME="defaultreg_frontier_Shin2017A/Shin2017A_frontier_confirmatory.json"

def audit(path):
    blob=pathlib.Path(path).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==EXPECTED_ZIP,"original immutable result archive changed"
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        j=json.loads(z.read(NAME))
    assert j["dataset"]=="Shin2017A" and j["n_subjects"]==29 and j["trial_denominator"]==1740
    assert j["target_information_track"]=="Z0"
    rows=[]
    for fold in sorted(j["folds"],key=lambda f:int(f["target_subject"])):
        sid=int(fold["target_subject"])
        source=fold["source_path"]
        grid=list(map(float,source["feasible_full_lambdas"]))
        assert len(grid)>=2 and len(set(grid))==len(grid)
        pathauc=fold["target_auc_path_by_lambda"]
        values=np.array([float(pathauc[str(float(lam))]) for lam in grid])
        macro=float(fold["auc"]["source_macro_full_domain_LDA"])
        cov=float(fold["auc"]["source_cov_nll_full_domain_LDA"])
        assert abs(float(pathauc[str(float(source["lambda_source_macro"]))])-macro)<1e-10
        assert abs(float(pathauc[str(float(source["lambda_covariance_full"]))])-cov)<1e-10
        rows.append({"subject":sid,"feasible_grid_candidates":len(grid),
                     "auc_source_macro_original":macro,
                     "auc_source_class_centered_original":cov,
                     "minimum_hindsight_auc":float(np.min(values)),
                     "maximum_hindsight_auc":float(np.max(values)),
                     "hindsight_oracle_lambda":float(grid[int(np.argmax(values))]),
                     "hindsight_oracle_gain_over_macro":float(np.max(values)-macro)})
    assert [r["subject"] for r in rows]==list(range(1,30))
    means={key:float(np.mean([r[key] for r in rows])) for key in
           ("auc_source_macro_original","auc_source_class_centered_original",
            "minimum_hindsight_auc","maximum_hindsight_auc")}
    assert abs(means["auc_source_macro_original"]-
               j["mean_auc_by_method"]["source_macro_full_domain_LDA"])<1e-11
    assert abs(means["auc_source_class_centered_original"]-
               j["mean_auc_by_method"]["source_cov_nll_full_domain_LDA"])<1e-11
    macro=means["auc_source_macro_original"]
    low,high=means["minimum_hindsight_auc"],means["maximum_hindsight_auc"]
    return {"status":"PASS_HINDSIGHT_FULL_GRID_ENVELOPE_NOT_EXECUTED_PREDICTIVE_EEG",
            "artifact_sha256":EXPECTED_ZIP,"subjects":29,"trials":1740,
            "source_selected_macro_auc":macro,
            "frozen_class_centred_covariance_auc":means["auc_source_class_centered_original"],
            "hindsight_any_selector_auc_lower":low,
            "hindsight_any_selector_auc_upper":high,
            "macro_minus_any_selector_delta_lower":macro-high,
            "macro_minus_any_selector_delta_upper":macro-low,
            "mean_hindsight_oracle_headroom_over_macro":high-macro,
            "subject_oracle_headroom_over_001_n":sum(
                r["hindsight_oracle_gain_over_macro"]>.01 for r in rows),
            "scope":"Only original fixed scoring family, same frozen folds and per-fold source-feasible candidate grid; posterior target AUC consulted to compute bounds. Not a proposed deployment method.",
            "prospective_pvalue":None,"proper_source_predictive_EEG_executed":False,
            "per_subject":rows}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("original_artifact_zip")
    ap.add_argument("--out")
    args=ap.parse_args()
    x=audit(args.original_artifact_zip)
    if args.out: pathlib.Path(args.out).write_text(json.dumps(x,indent=2)+"\n")
    print(json.dumps({k:v for k,v in x.items() if k!="per_subject"},indent=2))
