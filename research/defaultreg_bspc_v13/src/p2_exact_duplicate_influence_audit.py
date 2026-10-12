#!/usr/bin/env python3
"""P2 robust 49-record conditional influence audit from immutable ORIGINAL V29.

Key journal-method clarification: original aggregate correction is
dominated by the exact-overlap target group. Report leave-group-4-out
values without treating 48 overlapping source fits as independent.
Neither an independent dataset nor an intervention on true patients.
"""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

FROZEN_SHA="0e7bfdb73970fcff9231a51954bd2d548f472cb296624967afc80d9d241ce2be"
FROZEN_NAME="CROSSED_MODEL_SELECTION_49_GROUPS.json"
POLICIES=("source_macro","proper_source_mean_gaussian","historical_class_centred")
CELLS=("AA_old_fit_old_lambda","AB_old_fit_purged_lambda",
       "BA_purged_fit_old_lambda","BB_purged_fit_purged_lambda")

def audit(artifact:Path):
    with zipfile.ZipFile(artifact) as z:
        paths=[p for p in z.namelist() if p.endswith(FROZEN_NAME)]
        if len(paths)!=1:raise ValueError("not the unique frozen V29 record")
        raw=z.read(paths[0])
    h=hashlib.sha256(raw).hexdigest()
    if h!=FROZEN_SHA:raise AssertionError("V29 original 49-record result SHA mismatch")
    d=json.loads(raw)
    assert d["status"]=="V29_RETROSPECTIVE_FROZEN_FOUR_CELL_REFIT_SELECTION_COMPLETE"
    rows=sorted(d["held_record_crossed_responses"],key=lambda r:r["held_representative"])
    ids=np.array([int(r["held_representative"]) for r in rows])
    if (len(rows)!=49 or set(ids)!=(set(range(1,51))-{5})
        or not d["all_49_original_and_purged_outcomes_reproduced"]):
        raise AssertionError("wrong 49-group original result")
    gi=np.flatnonzero(ids==4)
    if len(gi)!=1:raise AssertionError("overlap affected group missing")
    k=int(gi[0])
    out={"status":"P2_FROZEN_SOURCE49_CONDITIONAL_INFLUENCE_REANALYSIS",
         "original_immutable_sha256":h,
         "n_unique_processed_recording_content_groups":49,
         "exact_duplicate_content_representative_id":4,
         "duplicate_ID5_excluded_from_49_group_target_denominator":True,
         "analysis_only_previously_exposed_outcomes":True,
         "not_patient_identity_or_causal_effect":True,
         "not_independent_paired_population_inference_due_source_overlap":True,
         "policies":{}}
    for name in POLICIES:
        a=np.array([[r["method_crossed_AUC"][name][col] for col in CELLS]
                    for r in rows],dtype=float)
        if a.shape!=(49,4) or not np.all(np.isfinite(a)):raise AssertionError("matrix shape")
        mu=a.mean(0)
        if max(abs(mu[i]-d["method_summary"][name]["participant_macro_mean"][col])
               for i,col in enumerate(CELLS))>1e-12:
            raise AssertionError("original immutable result mean mismatch")
        direct=a[:,3]-a[:,0]
        rest=np.delete(a,k,axis=0)
        rest_mean=rest.mean(0)
        impact_a=a[k]
        effect_by_row=direct
        change_selection=np.array([r["method_crossed_AUC"][name]["selected_lambda_changed"]
                                   for r in rows],bool)
        result={
            "all49_cells":dict(zip(CELLS,map(float,mu))),
            "id4_cells":dict(zip(CELLS,map(float,impact_a))),
            "remaining48_cells":dict(zip(CELLS,map(float,rest_mean))),
            "all49_delta_BB_minus_AA":float(mu[3]-mu[0]),
            "ID4_delta_BB_minus_AA":float(impact_a[3]-impact_a[0]),
            "remaining48_delta_BB_minus_AA":float(rest_mean[3]-rest_mean[0]),
            "id4_fit_only_delta_BA_minus_AA":float(impact_a[2]-impact_a[0]),
            "id4_lambda_effect_old_fit_AB_minus_AA":float(impact_a[1]-impact_a[0]),
            "id4_lambda_effect_purged_fit_BB_minus_BA":float(impact_a[3]-impact_a[2]),
            "all49_mean_conditional_change_quantiles":
                 dict(zip(("min","q10","q25","median","q75","q90","max"),
                          [float(v) for v in np.quantile(direct,[0,.1,.25,.5,.75,.9,1])])),
            "n_harmed_over_002":int(np.sum(direct < -.020000000001)),
            "n_helped_over_002":int(np.sum(direct > .020000000001)),
            "n_positive":int(np.sum(direct>0)),
            "n_negative":int(np.sum(direct<0)),
            "n_equal":int(np.sum(direct==0)),
            "n_changed_frozen_source_selection":int(change_selection.sum()),
            "n_changed_among_48_unaffected_target_groups":int(np.delete(change_selection,k).sum()),
            "largest_absolute_paired_impacts":
              [{"record_group":int(ids[i]),"delta_BB_AA":float(direct[i])}
                for i in np.argsort(np.abs(direct))[-5:][::-1]],
            "leave_any_single_group_out_mean_delta_extrema":
              [float(np.min((direct.sum()-direct)/48)),
               float(np.max((direct.sum()-direct)/48))],
        }
        # Additive decomposition, 49 weighted differences.
        assert np.isclose(result["all49_delta_BB_minus_AA"],
          (result["ID4_delta_BB_minus_AA"]+
           48*result["remaining48_delta_BB_minus_AA"])/49,
           atol=1e-14)
        out["policies"][name]=result
    a=out["policies"]["source_macro"]
    # Published nominal-50-ID macro AUC, from original 50-ID archive.
    a["published_original_nominal50_ID_macro_AUC"]=.594
    a["one_extra_repeated_nominal_ID_implied_AUC"]=float(
        50*.594-49*a["all49_cells"][CELLS[0]])
    assert abs(a["one_extra_repeated_nominal_ID_implied_AUC"]-.855)<1e-10
    assert abs(a["all49_delta_BB_minus_AA"]+.00612244897959184)<1e-11
    assert abs(a["remaining48_delta_BB_minus_AA"]+.00078125)<1e-11
    assert abs(a["ID4_delta_BB_minus_AA"]+.2625)<1e-11
    assert a["n_changed_frozen_source_selection"]==37
    out["source_macro_nominal50_to_record49_old_fit_delta"]=float(
        a["all49_cells"][CELLS[0]]-.594)
    out["source_macro_nominal50_to_record49_new_fit_delta"]=float(
        a["all49_cells"][CELLS[-1]]-.594)
    return out

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("original_v29_zip",type=Path)
    arg.add_argument("--out",type=Path,required=True)
    a=arg.parse_args()
    o=audit(a.original_v29_zip)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("P2_ALL49_AND_ID4_EXCLUDED_REPRODUCED",json.dumps({
      n:{k:v for k,v in o["policies"][n].items()
         if k in ("all49_delta_BB_minus_AA","ID4_delta_BB_minus_AA",
                  "remaining48_delta_BB_minus_AA",
                  "n_changed_among_48_unaffected_target_groups")}
      for n in POLICIES},sort_keys=True))
    print("P2_RECALCULATED_RECEIPT",str(a.out))
if __name__=="__main__":main()
