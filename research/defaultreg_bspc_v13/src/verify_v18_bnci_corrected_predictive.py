"""Independent dual-artifact audit of retrospective BNCI2014_002 true
source-mean Gaussian predictive NLL versus ORIGINAL frozen scoring family.

The original 14-person pre-data experiment was performed earlier.
The new proper predictive comparator is POST-OUTCOME, not a new
independent confirmatory result. Training pools overlap.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,pathlib,zipfile
import numpy as np

O_SHA="15137db055b54983c297d28b93e714836ff3711888c35bcdd3640eba05b1a6e1"
P_SHA="b4f583d63997d69d2f405c915ca9ca8eef8e623a4443e98b1634399d7d4f421b"
LOCK="58c7df532bbc503f6ff9b76aa2030ef8ffa69ffec1aacb82d72425b3929674ab"
FROZEN="defaultreg_frontier_BNCI2014_002/BNCI2014_002_frontier_confirmatory.json"


def read_zip(path,digest):
    b=pathlib.Path(path).read_bytes()
    assert hashlib.sha256(b).hexdigest()==digest, "Immutable original/new zip SHA mismatch"
    return zipfile.ZipFile(io.BytesIO(b))


def get_suffix(z,suffix):
    matches=[name for name in z.namelist() if name.endswith(suffix)]
    assert len(matches)==1, (suffix,matches)
    return z.read(matches[0])


def audit(original_zip,proper_zip):
    with read_zip(original_zip,O_SHA) as zo,read_zip(proper_zip,P_SHA) as zp:
        old=json.loads(zo.read(FROZEN))
        frozen_bytes=get_suffix(zp,"SOURCE_CHOICES_FROZEN_BEFORE_TARGET_OUTCOME_JOIN.json")
        assert hashlib.sha256(frozen_bytes).hexdigest()==LOCK
        choices=json.loads(frozen_bytes)
        new=json.loads(get_suffix(zp,"POSTHOC_PROPER_PREDICTIVE_VS_FROZEN_RESULTS.json"))
        assert old["dataset"]=="BNCI2014_002" and old["n_subjects"]==14
        assert old["trial_denominator"]==2240
        assert new["n"]==14 and len(choices)==len(new["per_subject"])==14
        assert new["status"]=="POST_OUTCOME_REAL_EEG_SENSITIVITY_NOT_CONFIRMATORY"
        assert new["no_new_confirmatory_p_value"] is True
        assert new["source_only_lambda_lock_sha256"]==LOCK
        orig={int(f["target_subject"]):f for f in old["folds"]}
        selected={int(x["subject"]):x for x in choices}
        effects={int(x["subject"]):x for x in new["per_subject"]}
        assert set(orig)==set(selected)==set(effects)==set(range(1,15))
        records=[]
        for sid in range(1,15):
            f=orig[sid];c=selected[sid];v=effects[sid]
            assert c["no_outer_target_info_for_selection"] is True
            lam=float(c["new_source_proper_predictive_lambda"])
            key=str(lam)
            assert lam in set(map(float,f["source_path"]["feasible_full_lambdas"]))
            pred=float(f["target_auc_path_by_lambda"][key])
            macro=float(f["auc"]["source_macro_full_domain_LDA"])
            centred=float(f["auc"]["source_cov_nll_full_domain_LDA"])
            assert abs(pred-v["auc_proper_predictive_FROM_FROZEN_PATH"])<1e-12
            assert abs(macro-v["auc_frozen_source_macro"])<1e-12
            assert abs(centred-v["auc_frozen_class_centered"])<1e-12
            exact_source_gap=float(c["source_predictive_log_score_by_lambda"][key]) \
              -float(f["source_path"]["criteria_by_lambda"][key]["covariance_mean_nll"])
            assert np.isfinite(exact_source_gap) and exact_source_gap>=-1e-9
            records.append({"subject":sid,"proper_lambda":lam,
                "macro_auc":macro,"proper_auc":pred,"centred_auc":centred,
                "macro_minus_proper":macro-pred,
                "class_mean_shift_SOURCE_penalty":exact_source_gap})
        mean=lambda key:float(np.mean([r[key] for r in records]))
        delta=np.array([r["macro_minus_proper"] for r in records])
        assert abs(mean("macro_minus_proper")-new["mean_source_macro_minus_proper_AUC"])<1e-12
        assert abs((mean("centred_auc")-mean("proper_auc"))-new["mean_class_centered_minus_proper_AUC"])<1e-12
        rnd=np.random.default_rng(20261012)
        ci=np.quantile(delta[rnd.integers(0,14,size=(30000,14))].mean(axis=1),[.025,.975])
        return {
          "status":"PASS_BNCI14_ORIGINAL_AND_REAL_PROPER_EEG_SOURCE_SCORE_AUDIT",
          "dataset":"BNCI2014_002","participants":14,"trials":2240,
          "original_artifact_sha256":O_SHA,"new_artifact_sha256":P_SHA,
          "source_only_lambda_lock_sha256":LOCK,
          "lambda_distribution":dict(collections.Counter(str(r["proper_lambda"]) for r in records)),
          "source_macro_auc":mean("macro_auc"),
          "source_proper_predictive_auc":mean("proper_auc"),
          "historical_class_centered_auc":mean("centred_auc"),
          "source_macro_minus_proper_auc":mean("macro_minus_proper"),
          "proper_minus_centered_auc":mean("proper_auc")-mean("centred_auc"),
          "mean_SOURCE_class_mean_shift_penalty":mean("class_mean_shift_SOURCE_penalty"),
          "min_SOURCE_class_mean_shift_penalty":min(r["class_mean_shift_SOURCE_penalty"] for r in records),
          "max_SOURCE_class_mean_shift_penalty":max(r["class_mean_shift_SOURCE_penalty"] for r in records),
          "positive_SOURCE_penalties":sum(r["class_mean_shift_SOURCE_penalty"]>0 for r in records),
          "source_macro_harmed_gt002_vs_proper":int(np.sum(delta<-.020000000001)),
          "proper_harmed_gt002_vs_macro":int(np.sum(delta>.020000000001)),
          "postoutcome_conditional_subject_bootstrap95":ci.tolist(),
          "new_confirmatory_test":False,
          "old_BNCI_primary_family_unchanged":True,
          "rows":records,
          "scope":"Postoutcome only; source pools overlap. Do not pool p-values or infer new generalization."
        }


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("original_zip")
    a.add_argument("proper_zip")
    a.add_argument("--out")
    args=a.parse_args()
    res=audit(args.original_zip,args.proper_zip)
    raw=json.dumps(res,indent=2)+"\n"
    if args.out:pathlib.Path(args.out).write_text(raw)
    print(raw)
