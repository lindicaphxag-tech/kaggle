"""Render Default Regularisation vNext tables from canonical absolute metrics.

Fail-closed design:
- refuses any key containing 'gain';
- derives every displayed delta from absolute AUC values;
- cross-checks stored primary mean_delta against candidate-baseline arithmetic;
- preserves evidence tier and target-information budget in output.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def reject_gain_fields(obj,path="root"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if "gain" in k.lower():
                raise ValueError(f"Manual gain field forbidden at {path}.{k}")
            reject_gain_fields(v,f"{path}.{k}")
    elif isinstance(obj,list):
        for i,v in enumerate(obj):reject_gain_fields(v,f"{path}[{i}]")

def fmt(x):return f"{float(x):.4f}"

def render(data):
    reject_gain_fields(data)
    rows=[]
    sources=data.get("evidence_sources",{})
    for cohort,c in data["cohorts"].items():
        source_key=c.get("evidence_source_key")
        if not isinstance(source_key,str) or source_key not in sources:
            raise ValueError(f"Missing/unknown evidence_source_key for {cohort}: {source_key}")
        source=sources[source_key]
        provenance_tokens=(
          source.get("artifact_digest"),source.get("artifact_zip_sha256"),
          source.get("result_json_sha256"),source.get("covariance_npz_sha256"),
          source.get("workflow_run"),source.get("receipt"))
        if not any(v not in (None,"",False) for v in provenance_tokens):
            raise ValueError(f"No machine provenance token for {cohort}/{source_key}")
        if "legacy" in source_key.lower() or "historical" in source_key.lower():
            raise ValueError(f"Historical/legacy evidence source forbidden in vNext table: {source_key}")
        auc=c["macro_auc"]
        base=c.get("table_reference_method")
        if not isinstance(base,str) or not base:
            raise ValueError(f"Missing explicit table_reference_method for {cohort}")
        if base not in auc:
            raise ValueError(f"table_reference_method absent from macro AUC for {cohort}: {base}")
        for method,value in auc.items():
            rows.append({
              "cohort":cohort,"evidence_source_key":source_key,
              "evidence_tier":c["evidence_tier"],
              "target_information_budget":c["target_information_budget"],
              "n":c["n_participants"],"method":method,
              "macro_auc":float(value),"baseline":base,
              "delta_vs_baseline":float(value)-float(auc[base])})
        for key,inf in c.get("primary_inference",{}).items():
            cand,base2=key.split("__vs__")
            if cand not in auc or base2 not in auc:
                raise ValueError(f"Inference methods absent from macro AUC for {cohort}: {key}")
            arithmetic=float(auc[cand])-float(auc[base2])
            # Macro-of-AUC arithmetic and mean paired delta are algebraically
            # identical when both are participant-macro on the same rows.
            if abs(arithmetic-float(inf["mean_delta"]))>5e-8:
                raise ValueError(
                    f"Primary delta mismatch {cohort}/{key}: "
                    f"absolute-AUC arithmetic={arithmetic} stored={inf['mean_delta']}")
    md=["| Cohort | Evidence | Budget | N | Method | Macro AUC | Delta vs table reference |",
        "|---|---|---:|---:|---|---:|---:|"]
    for r in rows:
        delta="--" if r["method"]==r["baseline"] else f"{r['delta_vs_baseline']:+.4f}"
        md.append(f"| {r['cohort']} | {r['evidence_tier']} | {r['target_information_budget']} | "
                  f"{r['n']} | {r['method']} | {fmt(r['macro_auc'])} | {delta} |")
    return rows,"\n".join(md)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--json-out",type=Path,required=True)
    ap.add_argument("--md-out",type=Path,required=True)
    a=ap.parse_args()
    data=json.loads(a.input.read_text())
    rows,md=render(data)
    a.json_out.parent.mkdir(parents=True,exist_ok=True)
    a.md_out.parent.mkdir(parents=True,exist_ok=True)
    a.json_out.write_text(json.dumps({"schema":"defaultreg_verified_table_v1","rows":rows},
                                     indent=2,sort_keys=True)+"\n")
    a.md_out.write_text(md)
    print(f"VERIFIED_TABLE rows={len(rows)} cohorts={len(data['cohorts'])}")

if __name__=="__main__":main()
