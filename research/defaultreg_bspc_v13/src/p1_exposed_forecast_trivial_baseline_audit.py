#!/usr/bin/env python3
"""Real but EXPOSED 9-subject forecast vs trivial source-blind baselines.
All LOPO source sets overlap. Paired sign-flip p is exploratory ONLY.
"""
import argparse,hashlib,itertools,json,math
from pathlib import Path

def audit(d):
    if not d.get("all_subject_outcomes_previously_exposed_projectwide"):
        raise ValueError("This audit is expressly retrospective; input provenance missing")
    rows=d["rows"]
    if len(rows)!=9 or len({r["target_id"] for r in rows})!=9:
        raise ValueError("expected nine unique original participants")
    y=[float(r["observed_macro_minus_predictive_auc"]) for r in rows]
    p=[float(r["source_forecast_macro_minus_predictive_auc"]) for r in rows]
    if not all(math.isfinite(x) for x in y+p):raise ValueError("non-finite AUC")
    pred_err=[abs(a-b) for a,b in zip(p,y)]
    zero_err=[abs(v) for v in y]
    signed=[u-v for u,v in zip(pred_err,zero_err)]
    obs=math.fsum(signed)/len(signed)
    all_signflips=[math.fsum(z*s for z,s in zip(signed,signs))/len(signed)
                   for signs in itertools.product((-1,1),repeat=len(signed))]
    pseudo_p=sum(abs(v)+1e-12>=abs(obs) for v in all_signflips)/len(all_signflips)
    return {
        "status":"EXPOSED_NINE_SUBJECT_NO_INDEPENDENT_CONFIRMATION",
        "n_subjects":len(y),"all_source_forecasts_positive":all(v>0 for v in p),
        "positive_actuals":sum(v>0 for v in y),
        "forecast_sign_correct":sum((a>0)==(b>0) for a,b in zip(p,y)),
        "always_positive_sign_correct":sum(v>0 for v in y),
        "model_mae":math.fsum(pred_err)/len(y),
        "zero_delta_forecast_mae":math.fsum(zero_err)/len(y),
        "paired_model_minus_zero_mae":obs,
        "exploratory_exact_two_sided_signflip_p":pseudo_p,
        "source_folds_overlap":True,"post_outcome_hypothesis":True,
        "p_is_not_acquisition_level_confirmatory":True,
        "no_new_unseen_acquisition_test":True}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    raw=args.input.read_bytes();r=audit(json.loads(raw))
    assert r["forecast_sign_correct"]==r["always_positive_sign_correct"]==7
    assert abs(r["exploratory_exact_two_sided_signflip_p"]-.3203125)<1e-10
    r["frozen_input_SHA256"]=hashlib.sha256(raw).hexdigest()
    text=json.dumps(r,indent=2,sort_keys=True)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    print("P1_EXPOSED_BASELINE_AUDIT",json.dumps(r,sort_keys=True))
