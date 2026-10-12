#!/usr/bin/env python3
"""P2 REAL PUBLIC Liu2024 exact-49 vs high-cosine-47 group sensitivity.

Post-screen retrospective scenario analysis, NOT re-identification of
patients. The [4,5],[11,15],[47,49] groups are an ASSUMED conservative
split scenario because high record similarity does not establish
original acquisition identity. No target outcome enters group
discovery or model fitting. The grouping uses ALL target EEG signals
and therefore is not prospective Z0 information-isolated analysis.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from p2_real_model_family_sensitivity import fit_scores,METHODS
from record_identity_groups_v26 import assign_recording_groups
sys.path.insert(0,str(HERE.parents[1]/"defaultreg_proper_predictive"))
import original_frozen_runner as old

SOURCE_GROUPS=((4,5),(11,15),(47,49))
DROP_NEAR=(5,15,49)
DROP_EXACT=(5,)
HELD_IDS=tuple(i for i in range(1,51) if i not in DROP_NEAR)

def to_json(obj,path):
    body=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()

def run():
    ds=Liu2024()
    pm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                    fmin=8,fmax=30,resample=160,baseline=None,
                    reject_by_annotation=False)
    rec=[];cov=[];yy=[];gg=[]
    for sid in range(1,51):
        X,labels,_=pm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(np.asarray(labels)=="right_hand",dtype=np.uint8)
        if X.ndim!=3 or X.shape[0]!=40 or X.shape[1]!=29 or int(y.sum())!=20 or not np.isfinite(X).all():
            raise RuntimeError(f"Liu2024 pinned EEG schema drift at ID {sid}")
        rec.append({"subject":sid,"signal":X,"labels":y})
        cov.append(old.covariances(X));yy.append(y);gg.extend([sid]*len(y))
        print("P2_APPROX_GROUP_PUBLIC_EEG",sid,flush=True)
    audit,_=assign_recording_groups(rec)
    if audit["duplicate_groups"]!=[[4,5]]:
        raise AssertionError("exact-content group disagreement")
    groups_exact=set(audit["representative_ids"])
    groups_near=groups_exact-set((15,49))
    if len(groups_exact)!=49 or len(groups_near)!=47 or groups_near!=set(HELD_IDS):
        raise AssertionError("47 near-sensitivity representative mismatch")
    C=np.concatenate(cov);y=np.concatenate(yy);g=np.asarray(gg)
    del rec,cov,yy
    report={"status":"RETROSPECTIVE_HIGH_COSINE_47_VS_EXACT_49_GROUP_SCENARIO",
      "dataset":"MOABB1.7.2 Liu2024 processed 8–30Hz 160Hz EEG",
      "near_similarity_threshold":"aligned centered whole-record cosine >=0.9999",
      "near_group_pairs":[list(s) for s in SOURCE_GROUPS],
      "near_group_membership_assumed_for_sensitivity_only":True,
      "not_claimed_same_human_or_recording_session":True,
      "global_full_EEG_content_inspected_before_groups":True,
      "not_new_prospective_Z0_or_causal_bias_estimate":True,
      "same_held_ids_exact49_and_near47":list(HELD_IDS),
      "n_exact_unique_records":49,"n_conservative_similarity_groups":47,
      "fixed_model_families":list(METHODS),"rows":[]}
    for target in HELD_IDS:
        held=(g==target)
        src49=(g!=target)&np.isin(g,list(groups_exact))
        src47=(g!=target)&np.isin(g,list(groups_near))
        if int(held.sum())!=40 or int(src49.sum())!=48*40 or int(src47.sum())!=46*40:
            raise RuntimeError("representative source group leak")
        row={"held_record_id":target,"near_alias_group_target":target in (4,11,47),
             "outcomes":{}}
        for name in METHODS:
            a=fit_scores(C,y,src49,held,name)
            b=fit_scores(C,y,src47,held,name)
            x=float(roc_auc_score(y[held],a));z=float(roc_auc_score(y[held],b))
            row["outcomes"][name]={"exact49_source_auc":x,"near47_source_auc":z,
                                  "paired_near47_minus_exact49_auc":z-x}
        report["rows"].append(row)
        print("P2_47GROUP_MATCHED_HELD_OUTCOME",target,row["outcomes"],flush=True)
    report["summary"]={}
    for name in METHODS:
        rows=report["rows"]
        d=np.asarray([r["outcomes"][name]["paired_near47_minus_exact49_auc"] for r in rows])
        touched=np.asarray([r["near_alias_group_target"] for r in rows])
        exact=np.asarray([r["outcomes"][name]["exact49_source_auc"] for r in rows])
        near=np.asarray([r["outcomes"][name]["near47_source_auc"] for r in rows])
        report["summary"][name]={
          "exact49_fit_auc_over_same47_held":float(np.mean(exact)),
          "conservative_near47_fit_auc_over_same47_held":float(np.mean(near)),
          "mean_delta_all47":float(np.mean(d)),
          "mean_delta_3_candidate_group_representatives":float(np.mean(d[touched])),
          "mean_delta_44_others":float(np.mean(d[~touched])),
          "candidate_id11_delta":float(d[list(HELD_IDS).index(11)]),
          "candidate_id47_delta":float(d[list(HELD_IDS).index(47)]),
          "n_harmed_gt002":int(np.sum(d<-.020000001)),
          "n_helped_gt002":int(np.sum(d>.020000001))}
    return report
def main():
    arg=argparse.ArgumentParser();arg.add_argument("--out",type=Path,required=True)
    args=arg.parse_args();out=run();digest=to_json(out,args.out)
    print("P2_REAL_47_VS_49_GROUP_SCENARIO_RESULTS",json.dumps(out["summary"],sort_keys=True),flush=True)
    print("P2_47_VS_49_SHA256",digest,flush=True)
if __name__=="__main__":main()
