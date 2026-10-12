#!/usr/bin/env python3
"""Independently audit frozen Lee2019 held A/B artifact data, no EEG loading."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

EXPECTED="27efc2565d5bc2bc22344b67ee994e42b985c16c34cabf6cdf55136270fa18f3"

def sha(b):return hashlib.sha256(b).hexdigest()
def signflip(x):
    if len(x)!=12:raise AssertionError("group unit denominator drift")
    bits=np.arange(4096,dtype=np.uint32)
    signs=1-2*((bits[:,None]>>np.arange(12,dtype=np.uint32))&1).astype(np.int8)
    return float(np.mean(np.abs(signs@x)>=abs(float(np.sum(x)))-1e-12))

def audit(src,block_paths):
    raw=Path(src).read_bytes()
    if sha(raw)!=EXPECTED:raise AssertionError("source original SHA not matching first run")
    s=json.loads(raw)
    assert s["source_ids"]==list(range(1,13))
    assert s["target_A_UNREAD"]==list(range(13,25))
    assert s["target_B_UNREAD"]==list(range(25,37))
    assert s["original_lambdas_exactly_reproduced"]
    assert len(s["source_trial_receipts"])==12
    audit=[]
    for label,p in zip("AB",block_paths):
        b=json.loads(Path(p).read_bytes())
        assert b["block"]==label
        assert b["status"]=="ONE_HELD_BLOCK_COMPLETED_WITHOUT_RETUNE"
        assert b["original_source_sha256"]==EXPECTED
        assert b["source_only_forecast_positive"]==s["positive_transfer_pre_prediction"]
        assert b["n_held_people"]==12 and b["n_source_people"]==12
        assert b["fixed_source_model_no_target_retrain"]
        assert abs(b["nonparallel_original_vs_gid_cosine"]-
                   s["final_operator_cosine"])<1e-12
        result=b["block_result"]
        persons=result["all_person_results"]
        ids=list(range(13,25) if label=="A" else range(25,37))
        assert [p["person_id"] for p in persons]==ids
        assert len({p["person_id"] for p in persons})==12
        keys=set(persons[0]["auc"])
        needed={"original","gid","within_only","balanced_only","LW",
                "OAS","fixed_full_shrink","tangent_LogReg",
                "riemannian_MDM","source_CSP3_LogReg"}
        assert keys==needed
        assert all(set(p["auc"])==needed and all(
                 0<=p["auc"][m]<=1 for m in needed)
                 for p in persons)
        macro={m:float(np.mean([p["auc"][m] for p in persons])) for m in needed}
        assert all(abs(macro[m]-result["macro_auc_per_method"][m])<1e-12 for m in needed)
        x=np.array([p["auc"]["gid"]-p["auc"]["original"] for p in persons])
        assert np.allclose(x,[p["gid_minus_original"] for p in persons])
        assert abs(x.mean()-result["gid_minus_original_mean_delta"])<1e-12
        pval=signflip(x)
        assert abs(pval-result["gid_minus_original_exact_two_sided_signflip_p"])<1e-12
        rng=np.random.default_rng(20261013)
        boot=x[rng.integers(0,12,(20000,12))].mean(axis=1)
        assert np.allclose(np.quantile(boot,[.025,.975]).tolist(),
             result["gid_minus_original_descriptive_bootstrap95"],atol=1e-12)
        assert int((x<-.020000000001).sum())==result["gid_losses_gt_002"]
        assert int((x>1e-12).sum())==result["gain_positive_persons"]
        assert int((x< -1e-12).sum())==result["gain_negative_persons"]
        gate=bool(s["positive_transfer_pre_prediction"]
           and s["final_operator_cosine"]<1-1e-5 and
           float(x.mean())>=.01 and pval<.05 and
           not (x<-.020000000001).any())
        assert gate==b["block_gate_passed"]
        audit.append({"block":label,
           "source_sha":EXPECTED,"n_held":12,
           "n_held_trials":int(sum(q["trials"] for q in persons)),
           "all_macro_auc":macro,
           "gid_minus_original_mean":float(x.mean()),
           "paired_exact_two_sided_p":pval,
           "harm_more_than002":int((x<-.020000000001).sum()),
           "positive_count":int((x>1e-12).sum()),
           "block_promotion_pass":gate,
           "source_forecast_positive":s["positive_transfer_pre_prediction"],
           "source_forecast_sign_consistent":bool((x.mean()>0)==s["positive_transfer_pre_prediction"]) if abs(x.mean())>1e-12 else None,
         })
    combined=np.mean([q["gid_minus_original_mean"] for q in audit])
    return {
      "audit_status":"IMMUTABLE_FIRST_LEE_S12_A12_B12_COMPLETE",
      "original_source_sha256":EXPECTED,
      "source_forecast_positive":s["positive_transfer_pre_prediction"],
      "source_original_lambda":s["chosen_source_only"]["decisions"]["original"]["lambda"],
      "source_gid_lambda":s["chosen_source_only"]["decisions"]["gid"]["lambda"],
      "source_nuisance_between_cov_fraction":s["nuisance_cov_between_fraction"],
      "source_direction_cosine":s["final_operator_cosine"],
      "both_held_block_results":audit,
      "combined_24_person_macro_delta_descriptive":float(combined),
      "research_promotion_all_gates_passed":bool(
         all(q["block_promotion_pass"] for q in audit)),
      "not_independent_acquisition_replication":True,
      "no_post_target_hyperparameter_update":True,
      "no_global_population_guarantee":True,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--a",type=Path,required=True)
    p.add_argument("--b",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    v=p.parse_args()
    out=audit(v.source,[v.a,v.b])
    v.out.parent.mkdir(parents=True,exist_ok=True)
    v.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("LEE2019_IMMUTABLE_TWO_BLOCK_INDEPENDENT_AUDIT_PASS",
          json.dumps(out,sort_keys=True),flush=True)
