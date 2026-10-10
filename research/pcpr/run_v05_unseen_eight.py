"""PCPR v0.5 eight-repository first-score, frozen before reading PR patches.

Commit of v0.5 detector: f68acdb92d59b8cbeef9f317eef779a9569563d1.
All observations are syntax-derived HYPOTHESES, no original/native
executions, no correctness labels and zero whole-PR authorizations.
"""
import json
from source_effect_ir_v05 import assess_real_pr

PR_LIST=(
 ("vllm-project/vllm",59106),
 ("NVIDIA/structured-data-models",1052),
 ("jeshraghian/snntorch",451),
 ("huggingface/transformers",49315),
 ("Project-MONAI/MONAI",9103),
 ("pytorch/ao",4921),
 ("pytorch/tensordict",2011),
 ("kornia/kornia",5338),
)

def main():
  reports=[]
  for repo,pr in PR_LIST:
    try:
      reports.append(assess_real_pr(repo,pr))
    except Exception as e:
      reports.append({"url":f"https://github.com/{repo}/pull/{pr}",
                      "status":"FETCH_FAILURE","error":type(e).__name__,
                      "n_typed_changed_functions":0,"n_typed_facets":0,
                      "native_executions":0,"patch_approved":False})
  result={
    "label":"FROZEN_V05_FIRST_SCORE_NOT_PATCH_CORRECTNESS",
    "method_sha":"f68acdb92d59b8cbeef9f317eef779a9569563d1",
    "cohort_lock":"V05_PRE_INSPECTION_LOCK_20261010.md",
    "n_repositories":8,
    "n_prs":8,
    "n_typed_candidate_prs":sum(x["n_typed_changed_functions"]>0 for x in reports),
    "n_typed_changed_functions":sum(x["n_typed_changed_functions"] for x in reports),
    "n_typed_facets":sum(x["n_typed_facets"] for x in reports),
    "n_fetch_failures":sum(x["status"]=="FETCH_FAILURE" for x in reports),
    "n_native_source_replays":0,
    "n_independently_adjudicated_specifications":0,
    "n_patch_correctness_decisions":0,
    "n_baseline_cross_tool_reproductions":0,
    "cases":reports,
  }
  with open("pcpr_v05_unseen_eight_first_score.json","w") as f:
    json.dump(result,f,indent=2,sort_keys=True)
  print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
  for r in reports:print(r["url"],r["status"],r["n_typed_changed_functions"],
                         ",".join(r.get("facet_union",[])))
  return result

if __name__=="__main__":main()
