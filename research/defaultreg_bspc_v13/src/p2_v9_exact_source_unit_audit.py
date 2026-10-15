#!/usr/bin/env python3
"""Exact paired source-unit analysis for P2 V8 CHB-MIT evidence.
Read P2_V8_12_SOURCE_FIXED_CAR_CASES.json from GitHub artifact 11948092005.
Never treat 168 controlled variants as independent recordings.
"""
import sys,json
from pathlib import Path
from scipy.stats import binomtest
def audit(doc):
    x=doc["source_grouped"]
    if not x or len(x)!=len(set(r["record"] for r in x)):raise ValueError("duplicated/empty sources")
    differences=[r["residual_positive"]-r["car_positive"] for r in x]
    wins=sum(x>0 for x in differences);losses=sum(x<0 for x in differences)
    nonzero=wins+losses
    p2=binomtest(wins,nonzero,p=.5,alternative="two-sided").pvalue if nonzero else 1.
    p1=binomtest(wins,nonzero,p=.5,alternative="greater").pvalue if nonzero else 1.
    return dict(n_original_sources=len(differences),n_source_wins=wins,
      n_source_losses=losses,n_tied_sources=len(x)-nonzero,
      positive_detected_V5=doc["results"]["residual"]["tp"],
      positive_detected_tuned_CAR=doc["results"]["tuned_car"]["tp"],
      additional_controlled_positives=sum(differences),
      exact_two_sided_source_sign_p=p2,exact_one_sided_source_sign_p=p1,
      contributing_sources=[dict(source=x[i]["record"],gained_cases=d)
        for i,d in enumerate(differences) if d>0],
      inferential_unit="original EDF file (not each transformed trial)",
      claim="NOT significant at two-sided .05; NOT a SOTA algorithm comparison")
if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: python p2_v9_exact_source_unit_audit.py PATH_TO_V8_JSON")
    print(json.dumps(audit(json.loads(Path(sys.argv[1]).read_text())),indent=2))
