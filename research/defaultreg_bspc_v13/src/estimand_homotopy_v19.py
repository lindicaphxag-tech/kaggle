#!/usr/bin/env python3
"""DefaultReg V19: exact source-only *estimand homotopy* across lambda grid.

J_alpha(lambda)=C_cent(lambda)+alpha*(C_pred(lambda)-C_cent(lambda)).
alpha=0 is historical validation-person class-centred covariance scoring.
alpha=1 is true source-trained-mean Gaussian predictive scoring.

Compute exact piecewise-affine lower envelope from pre-existing immutable
SOURCE-only fold objective paths. Freeze all alpha/decision breakpoints to
a SHA-256 receipt BEFORE joining held-out target AUC. Any target AUC analysis
is EXPLICITLY POST-OUTCOME (not a newly blind cohort, nor source-safe
alpha tuning). No invented model improvement or statistical test.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json
from pathlib import Path
import zipfile
import numpy as np

DATASETS={
 "Shin2017A":{
    "n":29,
    "original_sha":"cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842",
    "proper_sha":"d260a402bfbd9f4b8a49f6d0a47de766f71d7b399f1011039def302043d07467"},
 "BNCI2014_002":{
    "n":14,
    "original_sha":"15137db055b54983c297d28b93e714836ff3711888c35bcdd3640eba05b1a6e1",
    "proper_sha":"b4f583d63997d69d2f405c915ca9ca8eef8e623a4443e98b1634399d7d4f421b"},
}
ALPHAS=[0.,.05,.1,.2,.4,.6,.8,1.]


def archive_member(path,expected_hash,suffix):
    buf=Path(path).read_bytes()
    if hashlib.sha256(buf).hexdigest()!=expected_hash:
        raise AssertionError(f"immutable artifact SHA-256 mismatch: {path}")
    with zipfile.ZipFile(io.BytesIO(buf)) as z:
        names=[x for x in z.namelist() if x.endswith(suffix)]
        if len(names)!=1: raise AssertionError((suffix,names))
        return json.loads(z.read(names[0]))


def lower_envelope(lambdas,centred,penalties):
    """Exact lower envelope of finite family of affine lines.

    At a crossing, select smaller candidate lambda (tie break).
    Report open-interval minimisers and alpha at their entry; do not
    incorrectly infer crossings by a coarse 11-point alpha grid.
    """
    l=np.asarray(lambdas,float)
    c=np.asarray(centred,float)
    p=np.asarray(penalties,float)
    if (len(l)<2 or len(set(l))!=len(l) or
        not np.isfinite(c).all() or not np.isfinite(p).all() or
        np.any(p< -1e-8)):
        raise ValueError("Source grid or nonnegative exact penalty invalid")
    if not np.all(np.diff(l)>0): raise ValueError("Source grid must ascend")
    def choice(alpha):
        z=c+float(alpha)*p
        return int(np.argmin(z))
    roots=[0.,1.]
    for i in range(len(l)):
        for j in range(i+1,len(l)):
            d=float(p[i]-p[j])
            if abs(d)<1e-14:continue
            a=float((c[j]-c[i])/d)
            if 1e-12<a<1-1e-12:roots.append(a)
    roots=sorted(set(roots))
    segments=[]
    for a,b in zip(roots[:-1],roots[1:]):
        if b-a<1e-12:continue
        k=choice((a+b)/2)
        if segments and segments[-1]["lambda"]==float(l[k]):
            segments[-1]["right"]=b
        else:
            segments.append({"left":a,"right":b,
                             "lambda":float(l[k]),
                             "source_penalty":float(p[k])})
    if not segments or segments[0]["left"]!=0 or segments[-1]["right"]!=1:
        raise AssertionError("Invalid exact segment coverage")
    for i in range(len(segments)-1):
        if segments[i+1]["source_penalty"]>segments[i]["source_penalty"]+1e-9:
            raise AssertionError("Concave minimum implies nonincreasing chosen slope")
    return segments,choice


def audited_source_trajectory(original,proper,n):
    old={int(f["target_subject"]):f["source_path"] for f in original["folds"]}
    newer={int(z["subject"]):z for z in proper}
    if set(old)!=set(newer) or set(old)!=set(range(1,n+1)):
        raise AssertionError("Source persons differ")
    rows=[]
    for sid in range(1,n+1):
        o=old[sid];q=newer[sid]
        if q["no_outer_target_info_for_selection"] is not True:
            raise AssertionError("New source information budget is not Z0")
        grid=np.asarray(o["feasible_full_lambdas"],float)
        c0=np.asarray([float(o["criteria_by_lambda"][str(float(x))]["covariance_mean_nll"]) for x in grid])
        c1=np.asarray([float(q["source_predictive_log_score_by_lambda"][str(float(x))]) for x in grid])
        segments,select=lower_envelope(grid,c0,c1-c0)
        if abs(grid[select(0)]-float(o["lambda_covariance_full"]))>1e-12:
            raise AssertionError("alpha0 fails to recover original frozen selector")
        if abs(grid[select(1)]-float(q["new_source_proper_predictive_lambda"]))>1e-12:
            raise AssertionError("alpha1 fails to recover new executed proper score")
        first=segments[1]["left"] if len(segments)>1 else None
        decisions={str(a):float(grid[select(a)]) for a in ALPHAS}
        rows.append({"subject":sid,
          "original_historical_lambda":float(grid[select(0)]),
          "new_true_predictive_lambda":float(grid[select(1)]),
          "critical_first_change_alpha":first,
          "segments":segments,
          "fixed_alpha_source_lambdas":decisions,
          "positive_penalty_at_proper":float((c1-c0)[select(1)]),
          "source_only_no_target_outcomes_in_selection":True})
    return rows


def audit(dataset,original_zip,proper_zip,outdir):
    spec=DATASETS[dataset]
    orig=archive_member(original_zip,spec["original_sha"],"_frontier_confirmatory.json")
    new=archive_member(proper_zip,spec["proper_sha"],
                       "SOURCE_CHOICES_FROZEN_BEFORE_TARGET_OUTCOME_JOIN.json")
    assert orig["dataset"]==dataset and orig["n_subjects"]==spec["n"]
    source=audited_source_trajectory(orig,new,spec["n"])
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(source,sort_keys=True,indent=2)+"\n").encode()
    (out/f"{dataset}_source_estimand_homotopy_LOCKED_BEFORE_TARGET_AUC.json").write_bytes(raw)
    lock=hashlib.sha256(raw).hexdigest()
    print("ESTIMAND_HOMOTOPY_SOURCE_ONLY_SHA256",dataset,lock,flush=True)
    # No target outcome is used above. POST-OUTCOME descriptive joining below.
    by_id={int(f["target_subject"]):f for f in orig["folds"]}
    auc_at_alpha={str(a):[] for a in ALPHAS}
    for rec in source:
        fold=by_id[rec["subject"]]
        for alpha in ALPHAS:
            la=rec["fixed_alpha_source_lambdas"][str(alpha)]
            auc_at_alpha[str(alpha)].append(float(fold["target_auc_path_by_lambda"][str(float(la))]))
    thresholds=[r["critical_first_change_alpha"] for r in source if r["critical_first_change_alpha"] is not None]
    summary={
      "status":"POSTOUTCOME_SOURCE_OBJECTIVE_ESTIMAND_HOMOTOPY_NOT_NEW_EEG_CONFIRMATION",
      "dataset":dataset,"n":spec["n"],
      "original_zip_sha256":spec["original_sha"],
      "proper_zip_sha256":spec["proper_sha"],
      "source_only_homotopy_manifest_sha256":lock,
      "number_changing_lambda_from_alpha0_to_alpha1":sum(
         x["original_historical_lambda"]!=x["new_true_predictive_lambda"] for x in source),
      "median_exact_first_change_alpha":float(np.median(thresholds)),
      "exact_first_change_alpha_min":float(min(thresholds)),
      "exact_first_change_alpha_max":float(max(thresholds)),
      "segment_count_distribution":dict(collections.Counter(str(len(x["segments"])) for x in source)),
      "source_lambda_penalty_monotonic_along_each_envelope":True,
      "postoutcome_mean_AUC_fixed_alpha":{a:float(np.mean(b)) for a,b in auc_at_alpha.items()},
      "no_new_hypothesis_test":True,
      "no_target_selected_alpha_policy":True,
      "caution":"Discrete parametrised SOURCE objective, hindsight AUC only; no novel classifier or confirmatory external cohort. Alpha was not tuned with target outcomes.",
    }
    (out/f"{dataset}_estimand_homotopy_POSTOUTCOME_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shin-original",required=True)
    ap.add_argument("--shin-proper",required=True)
    ap.add_argument("--bnci-original",required=True)
    ap.add_argument("--bnci-proper",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    result={key:audit(key,*paths,a.out) for key,paths in {
        "Shin2017A":(a.shin_original,a.shin_proper),
        "BNCI2014_002":(a.bnci_original,a.bnci_proper)}.items()}
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
