#!/usr/bin/env python3
"""Collect post-freeze public resolution evidence only after prediction digest validation."""

from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, sys, urllib.error, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SEEN=ROOT/"seen.json"
PRED=ROOT/"predictions"; RES=ROOT/"resolution"; API="https://api.github.com"
HASH_FIELDS=("case_id","source_surface_sha256","contract_id","independence_class","predicted_relation","predicted_mechanism","owner_boundary","discriminating_experiment","falsifier","repair_boundary","prediction_frozen_at")

def parse(v): return dt.datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(dt.timezone.utc)
def fmt(v): return v.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
def pdigest(doc):
    return hashlib.sha256(json.dumps({k:doc[k] for k in HASH_FIELDS},ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def api(path,token):
    req=urllib.request.Request(API+path,headers={"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28","User-Agent":"semrepair-public-resolution"})
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub API {e.code} for {path}: {e.read().decode(errors='replace')}") from e
def paged(path,token):
    out=[]
    for page in range(1,21):
        sep="&" if "?" in path else "?"
        rows=api(f"{path}{sep}per_page=100&page={page}",token)
        if not isinstance(rows,list): raise RuntimeError("expected list payload")
        out.extend(x for x in rows if isinstance(x,dict))
        if len(rows)<100: return out
    raise RuntimeError("pagination safety limit")

def collect(path,token):
    pred=json.loads(path.read_text(encoding="utf-8"))
    if pred.get("prediction_sha256")!=pdigest(pred): raise ValueError(f"{path.name}: prediction digest mismatch")
    repo,num=pred["case_id"].rsplit("#",1); owner,name=repo.split("/",1); num=int(num); frozen=parse(pred["prediction_frozen_at"])
    issue=api(f"/repos/{owner}/{name}/issues/{num}",token)
    comments=[x for x in paged(f"/repos/{owner}/{name}/issues/{num}/comments",token) if x.get("created_at") and parse(x["created_at"])>=frozen]
    timeline=[x for x in paged(f"/repos/{owner}/{name}/issues/{num}/timeline",token) if x.get("created_at") and parse(x["created_at"])>=frozen]
    compact_comments=[{"id":x.get("id"),"created_at":x.get("created_at"),"updated_at":x.get("updated_at"),"author_login":(x.get("user") or {}).get("login"),"body":x.get("body") or "","html_url":x.get("html_url")} for x in comments]
    compact_timeline=[{"event":x.get("event"),"created_at":x.get("created_at"),"actor_login":(x.get("actor") or {}).get("login"),"commit_id":x.get("commit_id")} for x in timeline]
    record={"schema_version":1,"case_id":pred["case_id"],"prediction_sha256":pred["prediction_sha256"],"prediction_frozen_at":pred["prediction_frozen_at"],"collected_at":fmt(dt.datetime.now(dt.timezone.utc)),"issue":{"state":issue.get("state"),"updated_at":issue.get("updated_at"),"title_now":issue.get("title") or "","body_now":issue.get("body") or "","html_url":issue.get("html_url")},"comments":compact_comments,"timeline":compact_timeline,"adjudication":"not-performed","claim_boundary":"Post-freeze transport only; mechanism match requires separate adjudication."}
    payload=json.dumps({k:record[k] for k in ("case_id","prediction_sha256","issue","comments","timeline")},ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    record["resolution_evidence_sha256"]=hashlib.sha256(payload).hexdigest()
    RES.mkdir(exist_ok=True); out=RES/path.name
    prev=json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    if prev and prev.get("resolution_evidence_sha256")==record["resolution_evidence_sha256"]: return False
    out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return True

def self_test():
    x={"case_id":"o/r#1","source_surface_sha256":"a"*64,"contract_id":"c","independence_class":"I2","predicted_relation":"r","predicted_mechanism":"m","owner_boundary":"b","discriminating_experiment":"e","falsifier":"f","repair_boundary":"rb","prediction_frozen_at":"2026-10-06T02:00:00Z"}
    d=pdigest(x); y=dict(x); y["predicted_mechanism"]="changed"; assert pdigest(y)!=d; print("self-test: pass")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(); return
    token=os.environ.get("GITHUB_TOKEN")
    if not token: raise RuntimeError("GITHUB_TOKEN required")
    changed=[]
    state=json.loads(SEEN.read_text(encoding="utf-8"))
    finalized={
        item.get("id")
        for item in state.get("seen",[])
        if isinstance(item,dict) and str(item.get("status","")).startswith("adjudicated_")
    }
    if PRED.exists():
        for path in sorted(PRED.glob("*.json")):
            pred=json.loads(path.read_text(encoding="utf-8"))
            if pred.get("case_id") in finalized:
                continue
            if collect(path,token): changed.append(str(path))
    print(json.dumps({"changed":changed,"finalized_skipped":sorted(finalized)},sort_keys=True))
if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {exc}",file=sys.stderr); raise
