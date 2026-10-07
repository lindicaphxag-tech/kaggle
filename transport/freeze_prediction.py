#!/usr/bin/env python3
"""Freeze a tamper-evident mechanism prediction for a first-seen public case."""

from __future__ import annotations
import argparse, datetime as dt, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SEEN=ROOT/"seen.json"
PRED=ROOT/"predictions"
REQUIRED=("case_id","contract_id","independence_class","predicted_relation","predicted_mechanism","owner_boundary","discriminating_experiment","falsifier","repair_boundary")
HASH_FIELDS=("case_id","source_surface_sha256","contract_id","independence_class","predicted_relation","predicted_mechanism","owner_boundary","discriminating_experiment","falsifier","repair_boundary","prediction_frozen_at")

def parse(v): return dt.datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(dt.timezone.utc)
def fmt(v): return v.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
def digest(doc):
    payload={k:doc[k] for k in HASH_FIELDS}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def safe(v): return v.replace("/","__").replace("#","__")

def main():
    p=argparse.ArgumentParser(); p.add_argument("draft",type=Path); p.add_argument("--at"); a=p.parse_args()
    draft=json.loads(a.draft.read_text(encoding="utf-8"))
    missing=[k for k in REQUIRED if not draft.get(k)]
    if missing: raise ValueError(f"missing prediction fields: {missing}")
    state=json.loads(SEEN.read_text(encoding="utf-8"))
    entries={x.get("id"):x for x in state.get("seen",[]) if isinstance(x,dict)}
    case=draft["case_id"]
    if case not in entries: raise ValueError(f"{case}: no public first-seen record")
    source=entries[case]
    if source.get("status") != "candidate-frozen-awaiting-prediction":
        raise ValueError(f"{case}: already disposed, predicted, or expired")
    if a.at: raise ValueError("--at is disabled for live ledger writes; backdating is forbidden")
    now=dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if now > parse(source["prediction_deadline_at"]):
        raise ValueError(f"{case}: prediction deadline elapsed; no retrospective freeze allowed")
    frozen={k:draft[k] for k in REQUIRED}
    frozen.update({
        "schema_version":1,
        "source_surface_sha256":source["surface_sha256"],
        "first_frozen_at":source["first_frozen_at"],
        "prediction_deadline_at":source["prediction_deadline_at"],
        "prediction_frozen_at":fmt(now),
        "prediction_within_sla":now<=parse(source["prediction_deadline_at"]),
        "status":"prediction-frozen-awaiting-resolution",
        "claim_boundary":"A frozen prediction is not a correctness or I2 claim."
    })
    frozen["prediction_sha256"]=digest(frozen)
    PRED.mkdir(exist_ok=True)
    out=PRED/f"{safe(case)}.json"
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(frozen,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for x in state["seen"]:
        if x.get("id")==case:
            x["status"]="prediction-frozen-awaiting-resolution"
            x["prediction_sha256"]=frozen["prediction_sha256"]
            x["prediction_source"]=str(out.relative_to(ROOT))
            break
    SEEN.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(out); print(frozen["prediction_sha256"])
if __name__=="__main__": main()
