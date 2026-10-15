"""P4 independent synthetic SPD stress profile, not independent real EEG.

Profiles p=2,3,6,9,12 over FIXED identical source-validation
spectral state and pairs; compares Bernstein whole-numerator
certificate against V23 Taylor/termwise certificate. Dense-grid AUC
is a sanity check, not formal interval-arithmetic proof.
"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from bernstein_rank_cells_v25 import bernstein_rank_path
from spectral_path_certificate_v23 import certified_path_rank_stability


def auc(state,X,y,lam):
    e,U,d,mid,nu=state
    scores=(X-mid)@U@((U.T@d)/((1.-lam)*e+lam*nu))
    positive=scores[y==1];negative=scores[y==0]
    dif=positive[:,None]-negative[None,:]
    return float(np.mean((dif>0)+.5*(dif==0)))


def trial(p,seed):
    r=np.random.default_rng(13171+p*101+seed)
    e=np.exp(np.linspace(-3.,3.,p))*np.exp(r.normal(0,.1,p))
    U,_=np.linalg.qr(r.normal(size=(p,p)))
    d=r.normal(size=p);nu=float(np.mean(e))
    state=(e,U,d,np.zeros(p),nu)
    X=r.normal(size=(36,p));y=np.array([0]*18+[1]*18)
    low,high=.02,.98
    start=time.perf_counter()
    old=certified_path_rank_stability(state,X,y,low,high,
                                      segments=12,block_size=8)
    t_old=time.perf_counter()-start
    start=time.perf_counter()
    new=bernstein_rank_path(state,X,y,low,high,
                            segments=12,block_size=8)
    t_new=time.perf_counter()-start
    if old["source_trial_pair_count"]!=324 or new["source_trial_pair_count"]!=324:
        raise AssertionError("pair count mismatch")
    same={0:old["observed_endpoint_auc_low"],1:old["observed_endpoint_auc_high"]}
    for key,value in same.items():
        actual=auc(state,X,y,low if key==0 else high)
        expected_new=new["observed_endpoint_auc_low" if key==0 else "observed_endpoint_auc_high"]
        if abs(actual-value)>1e-12 or abs(actual-expected_new)>1e-12:
            raise AssertionError("endpoint AUC arithmetic inconsistent")
    for lam in np.linspace(low,high,201):
        realized=auc(state,X,y,float(lam))
        for algorithm,result in (("old",old),("bernstein",new)):
            if not (result["uniform_path_auc_lower_bound"]-1e-10 <= realized
                    <=result["uniform_path_auc_upper_bound"]+1e-10):
                raise AssertionError("dense path AUC broke numerical envelope for "+algorithm)
    a=float(old["certified_stable_fraction"])
    b=float(new["certain_fixed_sign_fraction"])
    return {"p":int(p),"seed":int(seed),"n_pairs":324,
            "certified_old":a,"certified_bernstein":b,
            "certified_delta_pp":100*(b-a),
            "endpoint_auc_low":same[0],"endpoint_auc_high":same[1],
            "old_time_seconds_exploratory":float(t_old),
            "bernstein_time_seconds_exploratory":float(t_new),
            "dense_201_lambdas_passed":True,
            "no_formal_fp_outward_rounding":True}


def main():
    rows=[trial(p,k) for p in (2,3,6,9,12) for k in (0,1,2)]
    summary={}
    for p in (2,3,6,9,12):
        d=np.array([r["certified_delta_pp"] for r in rows if r["p"]==p])
        summary[str(p)]={"n_cases":int(len(d)),
          "mean_delta_coverage_pp":float(d.mean()),
          "min_delta_coverage_pp":float(d.min()),
          "max_delta_coverage_pp":float(d.max()),
          "n_new_more":int((d>1e-10).sum()),
          "n_new_less":int((d<-1e-10).sum()),
          "n_equal":int((np.abs(d)<=1e-10).sum())}
    result={"status":"SYNTHETIC_FIXED_STATE_SPECTRAL_CERTIFICATE_STRESS_PASS",
      "not_real_eeg_or_independent_replicated_dataset":True,
      "not_a_proven_runtime_improvement":True,
      "not_a_formally_rounding_verified_interval_certificate":True,
      "n_synthetic_cases":len(rows),"n_each_case_source_label_pairs":324,
      "dense_grid_checked_for_each_case":201,
      "dimension_summary":summary,"cases":rows}
    output=Path(sys.argv[1]) if len(sys.argv)>1 else Path("/tmp/p4_profile.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("P4_SYNTHETIC_DIMENSION_PROFILE",json.dumps(summary,sort_keys=True))
    print("P4_SYNTHETIC_FULL_NUMERIC_PROFILE",str(output))


if __name__=="__main__":
    main()
