#!/usr/bin/env python3
"""Fail-closed integrity audit for the public prospective evidence ledger."""

from __future__ import annotations
import datetime as dt, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SEEN=ROOT/"seen.json"
PRED=ROOT/"predictions"
DISP=ROOT/"dispositions"
EXP=ROOT/"expirations"

PRED_HASH_FIELDS=("case_id","source_surface_sha256","contract_id","independence_class","predicted_relation","predicted_mechanism","owner_boundary","discriminating_experiment","falsifier","repair_boundary","prediction_frozen_at")
DISP_HASH_FIELDS=("case_id","source_surface_sha256","first_frozen_at","prediction_deadline_at","disposition_frozen_at","independence_class","disposition","reason")
EXP_HASH_FIELDS=("case_id","source_surface_sha256","first_frozen_at","prediction_deadline_at","expiry_recorded_at","reason")

def parse(v):
    return dt.datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(dt.timezone.utc)

def digest(doc, fields):
    payload={k:doc[k] for k in fields}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def safe(v):
    return v.replace("/","__").replace("#","__")

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def source_surface(entry):
    src=ROOT/entry["source"]
    doc=load(src)
    matches=[x for x in doc.get("new_candidates",[]) if x.get("id")==entry["id"]]
    if len(matches)!=1:
        raise ValueError(f"{entry['id']}: source snapshot cardinality={len(matches)}")
    return matches[0]

def main():
    state=load(SEEN)
    seen=state.get("seen",[])
    ids=[x.get("id") for x in seen]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate seen case id")

    counts={"pending":0,"prediction":0,"excluded":0,"expired":0}
    for entry in seen:
        case=entry["id"]
        surface=source_surface(entry)
        if surface["surface_sha256"]!=entry["surface_sha256"]:
            raise ValueError(f"{case}: seen/source surface hash mismatch")
        if surface["created_at"]!=entry["created_at"]:
            raise ValueError(f"{case}: seen/source created_at mismatch")

        pred=PRED/f"{safe(case)}.json"
        disp=DISP/f"{safe(case)}.json"
        exp=EXP/f"{safe(case)}.json"
        status=entry["status"]

        if status=="candidate-frozen-awaiting-prediction":
            if pred.exists() or disp.exists() or exp.exists():
                raise ValueError(f"{case}: pending case already has terminal artifact")
            if parse(entry["first_frozen_at"])>parse(entry["prediction_deadline_at"]):
                raise ValueError(f"{case}: impossible first-seen/deadline order")
            counts["pending"]+=1
            continue

        if status=="prediction-frozen-awaiting-resolution":
            if not pred.exists() or disp.exists() or exp.exists():
                raise ValueError(f"{case}: prediction state/artifact mismatch")
            doc=load(pred)
            if doc["case_id"]!=case or doc["source_surface_sha256"]!=entry["surface_sha256"]:
                raise ValueError(f"{case}: prediction source identity mismatch")
            if doc["prediction_sha256"]!=digest(doc,PRED_HASH_FIELDS):
                raise ValueError(f"{case}: prediction digest mismatch")
            if entry.get("prediction_sha256")!=doc["prediction_sha256"]:
                raise ValueError(f"{case}: seen/prediction digest mismatch")
            if bool(doc["prediction_within_sla"]) != (parse(doc["prediction_frozen_at"])<=parse(doc["prediction_deadline_at"])):
                raise ValueError(f"{case}: prediction SLA flag mismatch")
            if not doc["prediction_within_sla"]:
                raise ValueError(f"{case}: late mechanism prediction cannot count as prospective evidence")
            counts["prediction"]+=1
            continue

        if status=="deadline_missed_unadjudicated":
            if pred.exists() or disp.exists() or not exp.exists():
                raise ValueError(f"{case}: expired state/artifact mismatch")
            doc=load(exp)
            if doc["case_id"]!=case or doc["source_surface_sha256"]!=entry["surface_sha256"]:
                raise ValueError(f"{case}: expired source identity mismatch")
            if doc["expiry_sha256"]!=digest(doc,EXP_HASH_FIELDS):
                raise ValueError(f"{case}: expiration digest mismatch")
            if entry.get("expiry_sha256")!=doc["expiry_sha256"]:
                raise ValueError(f"{case}: seen/expiration digest mismatch")
            if entry.get("expiry_source")!=str(exp.relative_to(ROOT)):
                raise ValueError(f"{case}: expiration source path mismatch")
            if entry.get("expiry_recorded_at")!=doc["expiry_recorded_at"]:
                raise ValueError(f"{case}: expiration timestamp mismatch")
            if parse(doc["expiry_recorded_at"])<=parse(doc["prediction_deadline_at"]):
                raise ValueError(f"{case}: expiration was recorded before deadline")
            if doc.get("independence_class")!="UNADJUDICATED" or doc.get("primary_prospective_I2_eligible") is not False:
                raise ValueError(f"{case}: expired case improperly claims adjudication or I2 eligibility")
            counts["expired"]+=1
            continue

        if status.startswith("excluded_"):
            if pred.exists() or exp.exists() or not disp.exists():
                raise ValueError(f"{case}: exclusion state/artifact mismatch")
            doc=load(disp)
            if doc["case_id"]!=case or doc["source_surface_sha256"]!=entry["surface_sha256"]:
                raise ValueError(f"{case}: disposition source identity mismatch")
            if doc["disposition_sha256"]!=digest(doc,DISP_HASH_FIELDS):
                raise ValueError(f"{case}: disposition digest mismatch")
            if entry.get("disposition_sha256")!=doc["disposition_sha256"]:
                raise ValueError(f"{case}: seen/disposition digest mismatch")
            if entry.get("disposition_source")!=str(disp.relative_to(ROOT)):
                raise ValueError(f"{case}: disposition path mismatch")
            if bool(doc["disposition_within_sla"]) != (parse(doc["disposition_frozen_at"])<=parse(doc["prediction_deadline_at"])):
                raise ValueError(f"{case}: disposition SLA flag mismatch")
            counts["excluded"]+=1
            continue

        raise ValueError(f"{case}: unknown status {status!r}")

    print(json.dumps({"status":"pass","cases":len(seen),**counts},sort_keys=True))

if __name__=="__main__":
    main()
