#!/usr/bin/env python3
"""V21 independent original low-rank BNCI2014_004 risk/CVaR EEG auditor."""
from __future__ import annotations
import argparse,collections,hashlib,io,json,pathlib,zipfile
import numpy as np
ARTIFACT_SHA="81fc7b2d8e88a3e65b88a73be4c8237cf605f854169f7cee232f9badc824bcfb"
CHOICE_SHA="b76594a0d908833897f2b428420a0f2d0b65af772177022b8c8adeb1d3bd30d4"
PROTOCOL_SHA="a499df2d4881be185bf74215652da1ee810e48e89fb124ce70d6da19125b5e8c"

def only(z,suffix):
    q=[k for k in z.namelist() if k.endswith(suffix)]
    if len(q)!=1:raise AssertionError((suffix,q))
    return z.read(q[0])

def verify(path):
    blob=pathlib.Path(path).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==ARTIFACT_SHA
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        choice_bytes=only(z,"SOURCE_ONLY_LAMBDA_CHOICES_SHA_LOCK_BEFORE_TARGET_EVALUATION.json")
        proto=only(z,"BNCI2014_004_RISK_LOW_DIM_FIRST_RUN_PROTOCOL_20261012.md")
        result=json.loads(only(z,"BNCI2014_004_RISK_SOURCE_ONLY_EXTERNAL_RESULTS.json"))
    assert hashlib.sha256(choice_bytes).hexdigest()==CHOICE_SHA
    assert hashlib.sha256(proto).hexdigest()==PROTOCOL_SHA
    choices=json.loads(choice_bytes)
    assert result["status"]=="NEW_EXTERNAL_BNCI2014_004_EEG_FIRST_RUN_COMPLETE_UNREGISTERED"
    assert result["target_information_track"]=="Z0" and result["no_global_preregistration_claim"] is True
    assert result["source_lambda_choices_sha256_before_target_outcomes"]==CHOICE_SHA
    assert result["pre_run_protocol_sha256"]==PROTOCOL_SHA
    assert result["n_outer_people"]==9 and result["n_channels"]==3
    assert result["n_total_trials"]==6520 and result["all_participants_retained"]
    assert result["source_CVaR_quarter_selector_primary_risk_comparator"] is True
    by={int(x["outer_subject"]):x for x in choices}
    outcome={int(x["subject"]):x for x in result["per_subject_AUC"]}
    receipts={int(x["subject"]):x for x in result["participant_receipts"]}
    assert set(by)==set(outcome)==set(receipts)==set(range(1,10))
    names={"macro","true_source_mean_predictive","historical_class_centred",
           "source_worst_quarter_CVaR_auc","LW","OAS"}
    scores={k:[] for k in names}
    diff=[]
    for sid in range(1,10):
        x=by[sid];o=outcome[sid];receipt=receipts[sid]
        assert x["no_outer_target_data_in_source_selector"] is True
        assert set(x["source_people"])==set(range(1,10))-{sid}
        assert receipt["class0"]+receipt["class1"]==receipt["trials"]==o["test_trials"]
        assert receipt["class0"]==receipt["class1"]>=20
        assert set(o["auc"])==names
        assert all(0<=v<=1 for v in o["auc"].values())
        grid=np.asarray(x["source_feasible_lambda_grid"],float)
        c=np.asarray(x["source_class_centred_log_scores"],float)
        p=np.asarray(x["source_proper_predictive_log_scores"],float)
        assert len(grid)==len(c)==len(p) and np.all(np.diff(grid)>0)
        assert np.min(p-c)>-1e-8
        assert abs(grid[np.argmin(c)]-x["selected_lambda"]["historical_class_centred"])<1e-12
        assert abs(grid[np.argmin(p)]-x["selected_lambda"]["true_source_mean_predictive"])<1e-12
        tail=x["source_worst_quarter_CVaR_auc_by_lambda"]
        assert set(tail)=={str(float(g)) for g in grid}
        risk=min(grid,key=lambda la:(-float(tail[str(float(la))]),la))
        assert risk==x["selected_lambda"]["source_worst_quarter_CVaR_auc"]
        li=list(grid).index(x["selected_lambda"]["true_source_mean_predictive"])
        assert abs(float(p[li]-c[li])-x["source_penalty_at_proper_lambda"])<1e-10
        for name in names:scores[name].append(o["auc"][name])
        diff.append(o["auc"]["source_worst_quarter_CVaR_auc"]-o["auc"]["macro"])
    means={k:float(np.mean(v)) for k,v in scores.items()}
    assert all(abs(means[k]-result["mean_AUC"][k])<1e-12 for k in names)
    a=np.asarray(diff)
    assert abs(float(a.mean())-result["source_CVaR_quarter_minus_macro_mean_AUC"])<1e-12
    rng=np.random.default_rng(20261012)
    conditional=np.quantile(a[rng.integers(0,9,size=(30000,9))].mean(1),[.025,.975])
    assert int(np.sum(a<-.020000000001))==result["source_CVaR_quarter_harm_gt_002_vs_macro"]
    assert int(np.sum(a>.020000000001))==result["macro_harm_gt_002_vs_CVaR_quarter"]
    return {
       "status":"PASS_V21_ALL_9_LOW_DIM_EEG_SOURCE_ONLY_RISK_AUDIT",
       "dataset":"BNCI2014_004","n":9,"trials":result["n_total_trials"],
       "tangent_feature_dimensions":6,
       "original_artifact_zip_sha256":ARTIFACT_SHA,
       "pre_execution_protocol_sha256":PROTOCOL_SHA,
       "source_choice_sha256_before_target":CHOICE_SHA,
       "mean_auc":means,
       "source_CVaR_minus_macro_auc":float(a.mean()),
       "source_CVaR_minus_macro_subject_bootstrap_95_DESCRIPTIVE":conditional.tolist(),
       "lambda_decision_diff_risk_vs_macro":sum(q["selected_lambda"]["source_worst_quarter_CVaR_auc"]!=q["selected_lambda"]["macro"] for q in choices),
       "risk_lambda_distribution":dict(collections.Counter(str(q["selected_lambda"]["source_worst_quarter_CVaR_auc"]) for q in choices)),
       "proper_lambda_distribution":dict(collections.Counter(str(q["selected_lambda"]["true_source_mean_predictive"]) for q in choices)),
       "source_class_centred_lambda_distribution":dict(collections.Counter(str(q["selected_lambda"]["historical_class_centred"]) for q in choices)),
       "average_source_mean_shift_penalty":float(np.mean([q["source_penalty_at_proper_lambda"] for q in choices])),
       "CVaR_subjects_harmed_gt002_against_macro":int(np.sum(a<-.020000000001)),
       "macro_subjects_harmed_gt002_against_CVaR":int(np.sum(a>.020000000001)),
       "scope":"New public source-only execution, not independent preregistration, not 9 independent sites; CVaR is prior-art risk criterion and did not improve mean AUC here.",
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("original_zip");ap.add_argument("--out")
    x=ap.parse_args();d=verify(x.original_zip)
    text=json.dumps(d,indent=2)+"\n"
    if x.out:pathlib.Path(x.out).write_text(text)
    print(text)
