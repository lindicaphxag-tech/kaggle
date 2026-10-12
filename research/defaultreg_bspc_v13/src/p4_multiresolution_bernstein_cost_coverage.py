"""P4 cost-coverage tradeoff on previously defined FIXED 15 synthetic SPD fixtures.
Compare Bernstein subinterval counts 12/24/48 under same scalar
shrinkage path, no extra EEG observations or target gain.
"""
from __future__ import annotations
import json,time,sys
from pathlib import Path
import numpy as np
from bernstein_rank_cells_v25 import bernstein_rank_path
from spectral_path_certificate_v23 import certified_path_rank_stability

def state_for(p,seed):
    r=np.random.default_rng(13171+p*101+seed)
    e=np.exp(np.linspace(-3.,3.,p))*np.exp(r.normal(0,.1,p))
    U,_=np.linalg.qr(r.normal(size=(p,p)))
    d=r.normal(size=p)
    X=r.normal(size=(36,p))
    return (e,U,d,np.zeros(p),float(np.mean(e))),X,np.array([0]*18+[1]*18)

def run():
    out=[]
    for p in (2,3,6,9,12):
      for seed in (0,1,2):
        state,X,y=state_for(p,seed)
        x={"p":p,"seed":seed,"segments":{}}
        for n in (12,24,48):
          start=time.perf_counter()
          b=bernstein_rank_path(state,X,y,.02,.98,segments=n,block_size=8)
          t_b=time.perf_counter()-start
          if n==12:
            start=time.perf_counter()
            o=certified_path_rank_stability(state,X,y,.02,.98,
                                             segments=n,block_size=8)
            t_o=time.perf_counter()-start
            x["baseline_old_taylor12_coverage"]=o["certified_stable_fraction"]
            x["baseline_old_taylor12_seconds"]=t_o
          x["segments"][str(n)]={
           "bernstein_certified_fraction":b["certain_fixed_sign_fraction"],
           "uncertain_pairs":b["uncertain_pairs"],
           "walltime_seconds_exploratory":t_b,
           "source_auc_lower":b["uniform_path_auc_lower_bound"],
           "source_auc_upper":b["uniform_path_auc_upper_bound"]}
        cs=[x["segments"][str(n)]["bernstein_certified_fraction"]
            for n in (12,24,48)]
        if np.any(np.diff(cs)<-1e-10):
            # Expected under exact real-arithmetic degree preservation;
            # report rather than silently delete floating counterexample.
            x["numerical_certification_nonmonotone"]=True
        else:
            x["numerical_certification_nonmonotone"]=False
        x["fraction_gain_48_vs12"]=float(cs[-1]-cs[0])
        out.append(x)
    res={"status":"P4_SYNTHETIC_BERNSTEIN_COST_COVERAGE_15_CASE_PROFILE",
     "real_EEG_acquisition_count":0,
     "independently_collected_EEG_groups":0,
     "not_outward_directed_rounded":True,
     "no_unseen_target_benefit_established":True,
     "n_cases":15,
     "results":out,
     "mean_pp_gain_48_vs12":float(np.mean([x["fraction_gain_48_vs12"]
                                          for x in out])*100),
     "n_48_more_than_12":int(sum(x["fraction_gain_48_vs12"]>1e-10 for x in out)),
     "n_nonmonotone_floating_bound":int(sum(x["numerical_certification_nonmonotone"]
                                          for x in out))}
    p=Path(sys.argv[1]) if len(sys.argv)>1 else Path("/tmp/p4_cost_coverage.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("P4_SPECTRAL_CERT_COST_COVERAGE",json.dumps({
        k:res[k] for k in ("mean_pp_gain_48_vs12","n_48_more_than_12",
                            "n_nonmonotone_floating_bound","n_cases")}))
if __name__=="__main__":run()
