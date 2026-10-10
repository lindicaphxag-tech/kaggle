"""Frozen 48-PR six-stratum first score for source-exact v0.7 native operators.

Cohort locked in V07_48_PR_SIX_STRATUM_PRE_INSPECTION_LOCK_20261010.md
BEFORE inspecting patches. Frozen v0.7 method: deb2f16a660c352cb46d0f1113643f2623a7c784.
DO NOT silently replace cases, modify v0.7 predicates, or infer correctness
from merged status, PR titles, changed files, or patched assertions.

GitHub token is used ONLY for metadata GET; no token is passed to untrusted
module code (only standalone whitelisted AST torch expressions are replayed).
"""
from __future__ import annotations
import json,os,time,hashlib,urllib.request
from source_effect_ir_v05 import scan_pair as v05_scan
from native_delta_ir_v07 import score_file
from run_v05_authenticated_retry import request_json_authorized as api

GROUPS={
"dtype":[("geomstats/geomstats",2133),("SimplexLab/TorchJD",821),("NVIDIA/Model-Optimizer",2651),("huggingface/lerobot",4786),("roboflow/rf-detr",1616),("fangwei123456/spikingjelly",769),("google-ai-edge/litert-torch",1318),("pytorch/tensordict",1983)],
"state":[("huggingface/peft",3816),("huggingface/datasets",8387),("pytorch/executorch",23266),("OpenRAL/openral",305),("decoderesearch/SAELens",749),("jjhwilkinson/iwpc",43),("verl-project/verl",7610),("pytorch/ao",4886)],
"device":[("tile-ai/tilelang",3457),("MrNeRF/LichtFeld-Studio",2959),("firnlabs/minirl",34),("tracel-ai/burn",5942),("pytorch/executorch",23547),("LMCache/LMCache",5541),("facebookresearch/spdl",1727),("RBLN-SW/vllm-rbln",1244)],
"gradient":[("ICB-DCM/pyPESTO",1776),("onnxsim/onnxsim",2120),("kornia/kornia",5642),("keras-team/keras-mlx",65),("PyAutoLabs/PyAutoLens",768),("araffin/sbx",95),("tensorflow/tensorflow",127742),("rwth-i6/returnn",1919)],
"shape":[("vllm-project/vllm",60529),("huggingface/transformers",49426),("tenstorrent/tt-metal",53891),("Genesis-Embodied-AI/genesis-world",3396),("isaac-sim/IsaacLab",7813),("pytorch/rl",4318),("open-edge-platform/anomalib",3834),("PriorLabs/TabPFN",1223)],
"checkpoint":[("AI-Hypercomputer/maxtext",5463),("OpenPipe/ART",992),("google/tunix",2630),("Tencent-Hunyuan/UniRL",579),("kozistr/pytorch_optimizer",576),("unslothai/unsloth",13051),("marin-community/marin",9460),("NVIDIA-NeMo/Megatron-Bridge",6260)]
}
FROZEN=tuple((group,repo,num) for group,targets in GROUPS.items() for repo,num in targets)
assert len(FROZEN)==48 and len({(repo,num) for _,repo,num in FROZEN})==48

def git_blob(b):
 return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def raw(url):
 if not url.startswith("https://raw.githubusercontent.com/"):
  raise ValueError("SOURCE_ORIGIN_NOT_GITHUB")
 request=urllib.request.Request(url,headers={"User-Agent":"pcpr-v07-cohort-original-source-only"})
 with urllib.request.urlopen(request,timeout=35) as response:return response.read()

def score_one(group,repo,num,max_pyfiles=8):
 start=time.monotonic()
 result={
  "stratum":group,"repo":repo,"pr":num,
  "url":f"https://github.com/{repo}/pull/{num}",
  "status":"FETCH_FAILURE","fetch_error":None,
  "n_modified_py":0,"n_eligible_py":0,"n_checked_py":0,
  "n_skipped_due_budget":0,"n_file_fetch_failures":0,
  "v05_typed_candidate_functions":0,
  "v07_matched_operator_candidates":0,
  "v07_native_contexts":0,"v07_positive_dtype_contexts":0,
  "v07_preserving_contexts":0,"n_files_with_native_contexts":0,
  "n_independent_truth_labels":0,"n_correctness_decisions":0,
  "files":[],"head":None,"base":None}
 try:
  meta=api(f"https://api.github.com/repos/{repo}/pulls/{num}")
  files=api(f"https://api.github.com/repos/{repo}/pulls/{num}/files?per_page=100")
  result["head"]=meta["head"]["sha"];result["base"]=meta["base"]["sha"]
  if not isinstance(files,list):raise ValueError("INVALID_PR_FILE_LIST")
  py=[f for f in files if f["filename"].endswith(".py") and f["status"]=="modified"]
  eligible=[f for f in py if not f["filename"].startswith(("test/","tests/"))
            and "/tests/" not in f["filename"] and
            not f["filename"].split("/")[-1].startswith("test_")]
  result["n_modified_py"]=len(py)
  result["n_eligible_py"]=len(eligible)
  result["n_skipped_due_budget"]=max(0,len(eligible)-max_pyfiles)
  if len(files)>=100:result["listing_may_be_truncated"]=True
  for f in eligible[:max_pyfiles]:
   path=f["filename"]
   try:
    base=raw(f"https://raw.githubusercontent.com/{repo}/{result['base']}/{path}")
    head=raw(f"https://raw.githubusercontent.com/{repo}/{result['head']}/{path}")
    old,new=base.decode("utf-8"),head.decode("utf-8")
    baseline,state=v05_scan(path,old,new)
    current=score_file(path,old,new,max_candidates=8)
    frow={"path":path,"base_blob":git_blob(base),"head_blob":git_blob(head),
          "before_sha256":hashlib.sha256(base).hexdigest(),
          "after_sha256":hashlib.sha256(head).hexdigest(),
          "v05_function_candidates":sum(bool(z["facets"]) for z in baseline),
          "v07_operator_candidates":current["candidate_count"],
          "v07_native_contexts":current["n_native_contexts"],
          "v07_positive_dtype_contexts":current["n_positive_dtype_contexts"],
          "v07_preserving_contexts":sum(w["n_preserving"] for w in current["native_results"]),
          "source_analysis":current}
    result["files"].append(frow)
    result["n_checked_py"]+=1
    result["v05_typed_candidate_functions"]+=frow["v05_function_candidates"]
    result["v07_matched_operator_candidates"]+=frow["v07_operator_candidates"]
    result["v07_native_contexts"]+=frow["v07_native_contexts"]
    result["v07_positive_dtype_contexts"]+=frow["v07_positive_dtype_contexts"]
    result["v07_preserving_contexts"]+=frow["v07_preserving_contexts"]
    result["n_files_with_native_contexts"]+=bool(frow["v07_native_contexts"])
   except Exception as e:
    result["files"].append({"path":path,"status":"FILE_FETCH_OR_ANALYSIS_ERROR",
                            "error_type":type(e).__name__})
    result["n_file_fetch_failures"]+=1
  result["status"]=("SOURCE_UNSUPPORTED" if not eligible else
                    "SOURCE_FILE_ERROR" if result["n_file_fetch_failures"]==len(eligible[:max_pyfiles]) else
                    "NATIVE_EFFECT_OBSERVED" if result["v07_native_contexts"] else
                    "SOURCE_ANALYZED_ABSTAIN")
 except Exception as e:
  result["fetch_error"]=type(e).__name__
 result["elapsed_seconds"]=round(time.monotonic()-start,3)
 return result

def main():
 reports=[score_one(g,r,n) for g,r,n in FROZEN]
 result={
  "status":"FROZEN_V07_FIRST_NATIVE_OPERATOR_TRANSFER_NOT_PATCH_CORRECTNESS",
  "method_commit":"deb2f16a660c352cb46d0f1113643f2623a7c784",
  "preinspection_lock":"V07_48_PR_SIX_STRATUM_PRE_INSPECTION_LOCK_20261010.md",
  "n_total":len(FROZEN),
  "n_unique_repositories":len({r for _,r,_ in FROZEN}),
  "n_api_fetch_failures":sum(x["status"]=="FETCH_FAILURE" for x in reports),
  "n_unsupported_py_or_language":sum(x["status"]=="SOURCE_UNSUPPORTED" for x in reports),
  "n_source_analyzed":sum(x["status"] in ("SOURCE_ANALYZED_ABSTAIN","NATIVE_EFFECT_OBSERVED","SOURCE_FILE_ERROR") for x in reports),
  "n_v05_semantic_candidate_prs":sum(x["v05_typed_candidate_functions"]>0 for x in reports),
  "n_v07_matched_factory_prs":sum(x["v07_matched_operator_candidates"]>0 for x in reports),
  "n_v07_native_operator_prs":sum(x["v07_native_contexts"]>0 for x in reports),
  "n_v07_positive_dtype_gain_prs":sum(x["v07_positive_dtype_contexts"]>0 for x in reports),
  "n_v07_native_contexts":sum(x["v07_native_contexts"] for x in reports),
  "n_v07_dtype_gain_contexts":sum(x["v07_positive_dtype_contexts"] for x in reports),
  "n_v07_preserving_contexts":sum(x["v07_preserving_contexts"] for x in reports),
  "n_independently_labelled_patch_correctness":0,
  "n_verified_patch_correctness_decisions":0,
  "total_wall_seconds_sum_pr":round(sum(x["elapsed_seconds"] for x in reports),2),
  "strata":[{"stratum":group,
             "n":len(GROUPS[group]),
             "n_native_pr":sum(x["v07_native_contexts"]>0 for x in reports if x["stratum"]==group),
             "n_candidate_pr":sum(x["v05_typed_candidate_functions"]>0 for x in reports if x["stratum"]==group),
             "n_fetch_failures":sum(x["status"]=="FETCH_FAILURE" for x in reports if x["stratum"]==group)}
            for group in GROUPS],
  "reports":reports}
 with open("pcpr_v07_48_pr_frozen_native_first_score.json","w") as f:
  json.dump(result,f,indent=2,sort_keys=True)
 print(json.dumps({k:v for k,v in result.items() if k!="reports"},indent=2))
 for x in reports:
  print(x["stratum"],x["url"],x["status"],"v05",x["v05_typed_candidate_functions"],
        "v07",x["v07_native_contexts"],"gain",x["v07_positive_dtype_contexts"],
        "error",x["fetch_error"])
 return result

if __name__=="__main__":main()
