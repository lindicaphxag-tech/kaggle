"""First score v0.8 on NEW untouched 64-PR / 40-repository cohort.

Locked BEFORE patch access: V08_NEW_64_PR_PRE_INSPECTION_LOCK_20261010.md.
Frozen v0.8 implementation: 2fd8b21dcc2944d1703315b98767163e216c306b.
No requirement labels or patch correctness inferred from existing PRs or
patched tests; no whole-source execution. This is a conservative *operator
mechanism transfer* study with 8 file, 8 candidate and 3 dtype context budgets.
"""
from __future__ import annotations
import json,time
from run_v07_frozen_48_pr import raw,git_blob,api
from source_effect_ir_v05 import scan_pair as scan_v05
from native_delta_ir_v07 import score_file as score_v07
from symbolic_native_operator_v08 import candidate_pairs as pairs_v08,replay as replay_v08
from unchanged_consumer_graph_v08 import summarise_file

GROUPS={
"precision":[("kornia/kornia",5358),("braindecode/braindecode",1259),("vllm-project/vllm-ascend",18062),("jobovy/galpy",1613),("torch-spyre/torch-spyre",5152),("flagos-ai/FlagGems",6614),("pytorch/helion",3796),("SimplexLab/TorchJD",792)],
"state":[("NVIDIA/structured-data-models",642),("pese-git/llm-arch-research",32),("kornia/kornia",5559),("braindecode/braindecode",1235),("metatensor/metatensor",1165),("pytorch/torchtitan",5107),("vllm-project/vllm-omni",7061),("neuropoly/SmaugLab",95)],
"numerical":[("kornia/kornia",5329),("kornia/kornia",5579),("kornia/kornia",5484),("Grid-Raven/jammy_flows",12),("emdgroup/baybe",925),("knatarajan-lab/cehrgpt",156),("chrisfinlay/mri-generative-priors",12),("DeGrootResearchGroup/aquaflux",634)],
"layout":[("RBLN-SW/vllm-rbln",1242),("kornia/kornia",5058),("kornia/kornia",5558),("pytorch/helion",3789),("NVIDIA/structured-data-models",942),("MyoHub/myosuite",507),("tenstorrent/tt-metal",56069),("isaac-sim/IsaacLab",8052)],
"device":[("kornia/kornia",5142),("MooreThreads/vllm-musa",246),("unilabsim/unisim",351),("m87-labs/kestrel",261),("lvyufeng/relic-core",33),("kornia/kornia",4164),("flatppl/flatppl-python",18),("vllm-project/vllm",54213)],
"checkpoint":[("AI-Hypercomputer/maxtext",5508),("google/tunix",2585),("kozistr/pytorch_optimizer",579),("unslothai/unsloth",13092),("pytorch/torchtitan",5114),("OpenPipe/ART",1001),("rwth-i6/returnn",1921),("gke-labs/open-rl",322)],
"negative_and_features":[("kornia/kornia",5606),("jeshraghian/snntorch",455),("flagos-ai/FlagGems",6919),("tracel-ai/burn",5775),("NVIDIA/structured-data-models",941),("pytorch/torchtitan",4904),("google/tunix",2635),("huggingface/diffusers",14946)],
"gradient":[("kornia/kornia",5517),("kornia/kornia",5563),("onnxsim/onnxsim",2117),("braindecode/braindecode",1187),("NVIDIA/structured-data-models",910),("kornia/kornia",5409),("Grid-Raven/jammy_flows",11),("kornia/kornia",5076)]
}
UNSEEN=tuple((g,r,n) for g,targets in GROUPS.items() for r,n in targets)
assert len(UNSEEN)==64 and len({(r,n) for _,r,n in UNSEEN})==64

DTYPES=("float16","bfloat16","float32")
def one(group,repo,num,max_files=8,max_ops=8):
 t=time.monotonic()
 r={"stratum":group,"repo":repo,"pr":num,
    "url":f"https://github.com/{repo}/pull/{num}",
    "status":"PR_FETCH_FAILURE","fetch_error":None,
    "n_eligible_py_files":0,"n_checked_py_files":0,
    "n_file_failures":0,"n_budget_skipped_files":0,
    "v05_typed_functions":0,"v07_candidate_ops":0,
    "v07_native_contexts":0,
    "v08_candidate_ops":0,"v08_native_contexts":0,
    "v08_dtype_gain_contexts":0,"v08_preserving_contexts":0,
    "v08_before_only_consumer_paths":0,
    "n_independent_gold":0,"n_whole_PR_decisions":0,
    "files":[]}
 try:
  meta=api(f"https://api.github.com/repos/{repo}/pulls/{num}")
  listing=api(f"https://api.github.com/repos/{repo}/pulls/{num}/files?per_page=100")
  base=meta["base"]["sha"];head=meta["head"]["sha"]
  r["base"]=base;r["head"]=head
  if not isinstance(listing,list):raise ValueError("INVALID_FILE_LIST")
  py=[x["filename"] for x in listing if x["status"]=="modified" and x["filename"].endswith(".py")
       and "/tests/" not in x["filename"] and not x["filename"].startswith(("test/","tests/"))
       and not x["filename"].split("/")[-1].startswith("test_")]
  r["n_eligible_py_files"]=len(py)
  r["n_budget_skipped_files"]=max(0,len(py)-max_files)
  if len(listing)>=100:r["listing_may_be_truncated"]=True
  for path in py[:max_files]:
   try:
    oldbytes=raw(f"https://raw.githubusercontent.com/{repo}/{base}/{path}")
    newbytes=raw(f"https://raw.githubusercontent.com/{repo}/{head}/{path}")
    old,new=oldbytes.decode("utf8"),newbytes.decode("utf8")
    base_hints,_=scan_v05(path,old,new)
    v07=score_v07(path,old,new,max_candidates=max_ops)
    proposals,parse_status=pairs_v08(path,old,new)
    obs=[]
    for item in proposals[:max_ops]:
     try:obs.append(replay_v08(item,dtypes=DTYPES))
     except Exception as e:
      obs.append({"tier":"ABSTAIN_AST_UNSUPPORTED","reason":type(e).__name__,
                  "n_native_contexts":0,"n_dtype_gain_contexts":0,"n_preserving_contexts":0})
    graph=summarise_file(path,old,new)
    row={
      "path":path,"base_blob_sha1":git_blob(oldbytes),"head_blob_sha1":git_blob(newbytes),
      "v05_typed_functions":sum(bool(x["facets"]) for x in base_hints),
      "v07_candidate_ops":v07["candidate_count"],
      "v07_native_contexts":v07["n_native_contexts"],
      "v08_candidate_ops":len(proposals),
      "v08_native_contexts":sum(x["n_native_contexts"] for x in obs),
      "v08_dtype_gain_contexts":sum(x["n_dtype_gain_contexts"] for x in obs),
      "v08_preserving_contexts":sum(x["n_preserving_contexts"] for x in obs),
      "unchanged_consumer_paths":graph["n_syntactic_anchor_paths"],
      "consumer_roots":graph["roots"],
      "v08_operator_receipts":obs,
      "parse_status":parse_status,
      "n_skipped_candidates":max(0,len(proposals)-max_ops)
    }
    r["files"].append(row)
    r["n_checked_py_files"]+=1
    for key in ("v05_typed_functions","v07_candidate_ops","v07_native_contexts",
                "v08_candidate_ops","v08_native_contexts","v08_dtype_gain_contexts",
                "v08_preserving_contexts"):
     r[key]+=row[key]
    r["v08_before_only_consumer_paths"]+=row["unchanged_consumer_paths"]
   except Exception as e:
    r["n_file_failures"]+=1
    r["files"].append({"path":path,"status":"FETCH_OR_ANALYSIS_FAILURE",
                       "exception":type(e).__name__})
  r["status"]=("SOURCE_UNSUPPORTED" if not py else
              "ALL_SOURCE_FILES_FAILED" if r["n_file_failures"]==len(py[:max_files]) else
              "V08_NATIVE_OPERATOR_OBSERVED" if r["v08_native_contexts"] else
              "SOURCE_ANALYZED_NO_NATIVE")
 except Exception as e:
  r["fetch_error"]=type(e).__name__
 r["elapsed_seconds"]=round(time.monotonic()-t,3)
 return r

def main():
 rows=[one(g,repo,num) for g,repo,num in UNSEEN]
 out={"status":"NEW_64_V08_FROZEN_FIRST_SCORE_SOURCE_OPERATOR_ONLY",
      "method_v08_commit":"2fd8b21dcc2944d1703315b98767163e216c306b",
      "method_v07_commit":"deb2f16a660c352cb46d0f1113643f2623a7c784",
      "new_holdout_lock":"V08_NEW_64_PR_PRE_INSPECTION_LOCK_20261010.md",
      "dtype_contexts":DTYPES,
      "n_pr":64,"n_repos":len({r for _,r,_ in UNSEEN}),
      "n_pr_fetch_failures":sum(x["status"]=="PR_FETCH_FAILURE" for x in rows),
      "n_unsupported":sum(x["status"]=="SOURCE_UNSUPPORTED" for x in rows),
      "n_v05_candidate_prs":sum(x["v05_typed_functions"]>0 for x in rows),
      "n_v07_matched_prs":sum(x["v07_candidate_ops"]>0 for x in rows),
      "n_v07_native_prs":sum(x["v07_native_contexts"]>0 for x in rows),
      "n_v08_matched_prs":sum(x["v08_candidate_ops"]>0 for x in rows),
      "n_v08_native_prs":sum(x["v08_native_contexts"]>0 for x in rows),
      "n_v08_local_dtype_gain_prs":sum(x["v08_dtype_gain_contexts"]>0 for x in rows),
      "n_v08_native_contexts":sum(x["v08_native_contexts"] for x in rows),
      "n_v08_dtype_gain_contexts":sum(x["v08_dtype_gain_contexts"] for x in rows),
      "n_v08_preserving_contexts":sum(x["v08_preserving_contexts"] for x in rows),
      "n_pr_with_before_only_consumer_paths":sum(x["v08_before_only_consumer_paths"]>0 for x in rows),
      "n_independently_adjudicated_specs":0,
      "n_whole_patch_correctness_decisions":0,
      "n_matched_cost_strong_apca_baselines":0,
      "total_sum_pr_seconds":round(sum(x["elapsed_seconds"] for x in rows),2),
      "group_metrics":[{"name":g,"n":len(GROUPS[g]),
                        "n_v08_native":sum(x["v08_native_contexts"]>0 for x in rows if x["stratum"]==g),
                        "n_v05_candidates":sum(x["v05_typed_functions"]>0 for x in rows if x["stratum"]==g),
                        "n_fetch_failures":sum(x["status"]=="PR_FETCH_FAILURE" for x in rows if x["stratum"]==g)}
                       for g in GROUPS],
      "reports":rows}
 with open("pcpr_v08_new_64_frozen_first_score.json","w") as f:
  json.dump(out,f,sort_keys=True,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k!="reports"},indent=2))
 for x in rows:
  print(x["stratum"],x["url"],x["status"],
        "v05",x["v05_typed_functions"],"v07",x["v07_native_contexts"],
        "v08",x["v08_native_contexts"],"gain",x["v08_dtype_gain_contexts"])
 return out
if __name__=="__main__":main()
