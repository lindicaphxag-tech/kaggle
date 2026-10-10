"""Fail-closed audit for DefaultReg v5 true PRE-DATA results."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

EXPECTED={
 "BNCI2014_002":{"n":14,"channels":15,"dim":120},
 "BNCI2015_001":{"n":12,"channels":13,"dim":91},
}
METHODS={
 "LedoitWolf_LDA","OAS_LDA","source_cov_nll_full_domain_LDA",
 "source_macro_full_domain_LDA","discriminability_first_LDA",
 "mean_tail_fidelity_LDA","TS_LogisticRegression","Riemannian_MDM",
}
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def audit(dataset,root):
 root=Path(root);e=EXPECTED[dataset]
 p=root/f"{dataset}_v5_result.json"
 x=json.loads(p.read_text())
 assert x["status"]=="V5_TRUE_PREDATA_EXTERNAL_COMPLETE"
 assert x["dataset"]==dataset and x["n_subjects"]==e["n"]
 assert x["subjects"]==list(range(1,e["n"]+1))
 assert x["n_channels"]==e["channels"] and x["tangent_dimension"]==e["dim"]
 assert x["target_information_track"]=="Z0"
 assert x["score_freeze_precedes_target_auc"] is True
 assert len(x["dense_grid"])>400
 assert set(x["methods"])==METHODS and set(x["macro_auc"])==METHODS
 assert len(x["per_subject"])==e["n"]
 assert all(set(r["auc"])==METHODS for r in x["per_subject"])
 for key in ("primary_H2_macro_vs_cov","primary_H3_macro_vs_LW"):
  q=x[key]
  for field in ("bootstrap95","exact_signflip_p","holm_p_H2_H3","cvar25","cvar10","worst","harm_rate_gt_0_02"):
   assert field in q,(key,field)
 score=root/"frozen_target_scores_before_auc.npz"
 h=sha(score)
 assert h==x["score_bundle_sha256"]
 receipt=json.loads((root/"receipt.json").read_text())
 assert receipt["score_bundle_sha256"]==h
 assert receipt["n_subjects"]==e["n"]
 assert sum(receipt["per_subject_trials"].values())==receipt["total_trials"]
 assert len(x["crossfitted_validity_frontier"]["envelope"])>400
 print("PASS_V5_TRUE_PREDATA_CONTRACT",dataset)
 print(json.dumps({
  "macro_auc":x["macro_auc"],
  "H2":x["primary_H2_macro_vs_cov"],
  "H3":x["primary_H3_macro_vs_LW"],
  "secondary":x["secondary_mean_tail_vs_macro"],
  "score_bundle_sha256":h},indent=2))
 return x

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--dataset",required=True,choices=sorted(EXPECTED))
 p.add_argument("--root",required=True);a=p.parse_args();audit(a.dataset,a.root)
