"""First score on eight repo-disjoint PRs. The actual v0.2 mechanism
detector/probes imported here are frozen in Git commit 92aeab877d341e8e923f451e2e45757c4b6e7516.
Do not modify typed_effects_v02.py or drop failures for this run.
"""
import json
from typed_effects_v02 import audit

UNSEEN=[
 ("pytorch/torchtitan",4474),
 ("NVIDIA/structured-data-models",637),
 ("pytorch/executorch",22994),
 ("torch-spyre/torch-spyre",4383),
 ("QwenAudio/SenseVoice",358),
 ("NVIDIA-NeMo/Automodel",3850),
 ("kornia/kornia",5026),
 ("applied-material-modeling/neml2",469),
]
def main():
 results=[]
 for repo,num in UNSEEN:
  try:results.append(audit(repo,num))
  except Exception as e:
   results.append({"url":f"https://github.com/{repo}/pull/{num}",
                   "status":"FETCH_FAILURE",
                   "reason":type(e).__name__,
                   "n_scoped_native_semantics":0})
 return {"status":"FIRST_PASS_AFTER_V02_METHOD_FREEZE",
         "design":"TITLE_SELECTED_EIGHT_DISTINCT_UPSTREAM_REPOS",
         "algorithm_commit":"92aeab877d341e8e923f451e2e45757c4b6e7516",
         "n_fixed":8,
         "n_scoped_semantic_prs":sum(bool(x["n_scoped_native_semantics"]) for x in results),
         "n_native_semantic_probe_cases":sum(x["n_scoped_native_semantics"] for x in results),
         "n_fetch_failures":sum(x.get("status")=="FETCH_FAILURE" for x in results),
         "n_real_full_package_validations":0,
         "n_patch_correctness_decisions":0,
         "results":results}
if __name__=="__main__":
 r=main()
 with open("pcpr_v02_unseen_eight_first_score.json","w") as f:json.dump(r,f,indent=2,sort_keys=True)
 print(json.dumps({k:v for k,v in r.items() if k!="results"},indent=2))
 for z in r["results"]:print(z["url"],z.get("matched_mechanisms",[]),z["n_scoped_native_semantics"])
