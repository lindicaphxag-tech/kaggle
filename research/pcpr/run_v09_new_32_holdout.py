"""Frozen third-cohort first score: 32 untouched merged PRs, v0.9 cross-expression.

V09_NEW_32_PR_PRE_INSPECTION_LOCK_20261010.md fixed BEFORE patch access.
Algorithm v0.9 frozen at 471ea5d344b234defa6491c283395233750bcffd.
No independent requirement gold; no PR correctness decisions; count fetch
errors and unsupported languages as in-denominator, not method successes.
"""
from __future__ import annotations
import json,time
from run_v07_frozen_48_pr import raw,git_blob,api
from source_effect_ir_v05 import scan_pair as v05
from native_delta_ir_v07 import score_file as v07
from symbolic_native_operator_v08 import candidate_pairs as v08_pairs,replay as v08_replay
from cross_expression_native_v09 import score_file as v09
from unchanged_consumer_graph_v08 import summarise_file

GROUPS={
"cast_and_gradient":[("EleutherAI/bergson",488),("NVIDIA-NeMo/Automodel",4150),("tracel-ai/burn",5843),("meta-pytorch/flex_shard",23),("DIAGNijmegen/lightstream",346),("kozistr/pytorch_optimizer",560),("keras-team/keras",23593),("intel/auto-round",2232)],
"dtype_operator":[("kornia/kornia",5571),("kornia/kornia",5303),("kornia/kornia",5524),("kornia/kornia",4898),("NVIDIA/recsys-examples",495),("kornia/kornia",5377),("kozistr/pytorch_optimizer",599),("kornia/kornia",5353)],
"mixed_infrastructure":[("apple/coremltools",2833),("Boffee/piper-kernels",87),("flagos-ai/FlagGems",6039),("facebookresearch/spdl",1726),("rwth-i6/returnn",1891),("huggingface/pytorch-image-models",2841),("huggingface/sentence-transformers",4113),("ModelCloud/GPTQModel",3144)],
"negative_and_controls":[("scipy/scipy-stubs",2530),("pandas-dev/pandas",70909),("stdlib-js/stdlib",15747),("ml-explore/mlx",4626),("onnx/onnx",8506),("kornia/kornia",5526),("NVIDIA/physicsnemo",2072),("scikit-hep/uhi",280)]
}
FROZEN=tuple((g,r,n) for g,rows in GROUPS.items() for r,n in rows)
assert len(FROZEN)==32 and len({(r,n) for _,r,n in FROZEN})==32
DTYPES=("float16","bfloat16","float32")
def one(group,repo,number,max_files=8,op_budget=8):
 t=time.monotonic()
 row={"group":group,"repo":repo,"pr":number,
      "url":f"https://github.com/{repo}/pull/{number}",
      "status":"PR_FETCH_FAILURE","error_type":None,
      "n_source_files":0,"n_processed_files":0,"n_unsupported_files":0,
      "n_skipped_budget_files":0,
      "v05_typed_candidate_functions":0,
      "v07_native_contexts":0,"v08_native_contexts":0,
      "v09_cast_candidates":0,"v09_native_contexts":0,
      "v09_recovered_contexts":0,"v09_dtype_reference_gain_contexts":0,
      "v09_preserving_controls":0,
      "before_only_consumer_paths":0,
      "n_whole_patch_authorizations":0,
      "files":[]}
 try:
  meta=api(f"https://api.github.com/repos/{repo}/pulls/{number}")
  changed=api(f"https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=100")
  row["base_sha"]=meta["base"]["sha"];row["head_sha"]=meta["head"]["sha"]
  if not isinstance(changed,list):raise ValueError("NOT_FILE_LIST")
  py=[f["filename"] for f in changed if f["status"]=="modified"
      and f["filename"].endswith(".py")
      and "/tests/" not in f["filename"] and not f["filename"].startswith(("test/","tests/"))
      and not f["filename"].split("/")[-1].startswith("test_")]
  row["n_source_files"]=len(py)
  row["n_skipped_budget_files"]=max(0,len(py)-max_files)
  if len(changed)>=100:row["file_list_possibly_truncated"]=True
  for path in py[:max_files]:
   try:
    before_bytes=raw(f"https://raw.githubusercontent.com/{repo}/{row['base_sha']}/{path}")
    after_bytes=raw(f"https://raw.githubusercontent.com/{repo}/{row['head_sha']}/{path}")
    before=before_bytes.decode("utf8");after=after_bytes.decode("utf8")
    b05,_=v05(path,before,after)
    b07=v07(path,before,after,max_candidates=op_budget)
    candidates,_=v08_pairs(path,before,after)
    b08=[]
    for ca in candidates[:op_budget]:
     try:b08.append(v08_replay(ca,dtypes=DTYPES))
     except Exception:b08.append({"n_native_contexts":0})
    b09=v09(path,before,after,budget=op_budget)
    graph=summarise_file(path,before,after)
    vals={
      "path":path,"base_git_blob":git_blob(before_bytes),
      "head_git_blob":git_blob(after_bytes),
      "v05_typed_candidate_functions":sum(bool(z["facets"]) for z in b05),
      "v07_native_contexts":b07["n_native_contexts"],
      "v08_native_contexts":sum(z["n_native_contexts"] for z in b08),
      "v09_cast_candidates":b09["n_candidates"],
      "v09_native_contexts":b09["n_native_contexts"],
      "v09_recovered_contexts":b09["n_recovered_execution"],
      "v09_dtype_reference_gain_contexts":b09["n_dtype_reference_gains"],
      "v09_preserving_controls":b09["n_preserving_controls"],
      "before_only_consumer_paths":graph["n_syntactic_anchor_paths"],
      "v09_original_expression_receipts":b09["observations"]
    }
    row["files"].append(vals)
    row["n_processed_files"]+=1
    for key in ("v05_typed_candidate_functions","v07_native_contexts",
                "v08_native_contexts","v09_cast_candidates","v09_native_contexts",
                "v09_recovered_contexts","v09_dtype_reference_gain_contexts",
                "v09_preserving_controls","before_only_consumer_paths"):
     row[key]+=vals[key]
   except Exception as e:
    row["n_unsupported_files"]+=1
    row["files"].append({"path":path,"status":"FILE_FETCH_OR_ANALYSIS_FAILED",
                         "exception_type":type(e).__name__})
  row["status"]=("SOURCE_UNSUPPORTED" if not py else
                  "ALL_FILES_FAILED" if not row["n_processed_files"] else
                  "V09_NATIVE_EFFECT_OBSERVED" if row["v09_native_contexts"] else
                  "SOURCE_ANALYZED_ABSTAIN")
 except Exception as e:row["error_type"]=type(e).__name__
 row["elapsed_seconds"]=round(time.monotonic()-t,3)
 return row

def main():
 rows=[one(g,r,n) for g,r,n in FROZEN]
 result={
  "status":"NEW_32_V09_FROZEN_FIRST_SCORE_SOURCE_ONLY",
  "third_cohort_lock":"V09_NEW_32_PR_PRE_INSPECTION_LOCK_20261010.md",
  "method_v09_commit":"471ea5d344b234defa6491c283395233750bcffd",
  "number_pr":32,"unique_repositories":len({r for _,r,_ in FROZEN}),
  "dtype_contexts":DTYPES,
  "number_pr_fetch_failures":sum(x["status"]=="PR_FETCH_FAILURE" for x in rows),
  "number_unsupported_pr":sum(x["status"]=="SOURCE_UNSUPPORTED" for x in rows),
  "n_pr_v05_typed_candidates":sum(x["v05_typed_candidate_functions"]>0 for x in rows),
  "n_pr_v07_native":sum(x["v07_native_contexts"]>0 for x in rows),
  "n_pr_v08_native":sum(x["v08_native_contexts"]>0 for x in rows),
  "n_pr_v09_native":sum(x["v09_native_contexts"]>0 for x in rows),
  "n_pr_v09_recovered_execution":sum(x["v09_recovered_contexts"]>0 for x in rows),
  "n_pr_v09_dtype_reference_gain":sum(x["v09_dtype_reference_gain_contexts"]>0 for x in rows),
  "number_v09_native_contexts":sum(x["v09_native_contexts"] for x in rows),
  "number_v09_recovered_contexts":sum(x["v09_recovered_contexts"] for x in rows),
  "number_v09_dtype_gain_contexts":sum(x["v09_dtype_reference_gain_contexts"] for x in rows),
  "number_v09_preserving_controls":sum(x["v09_preserving_controls"] for x in rows),
  "n_pr_syntactic_before_only_consumers":sum(x["before_only_consumer_paths"]>0 for x in rows),
  "n_independent_correctness_labels":0,
  "n_whole_patch_correctness_decisions":0,
  "total_sum_PR_seconds":round(sum(x["elapsed_seconds"] for x in rows),2),
  "groups":[{"group":group,"n":8,
             "n_v09_native":sum(x["v09_native_contexts"]>0 for x in rows if x["group"]==group),
             "n_v05_candidates":sum(x["v05_typed_candidate_functions"]>0 for x in rows if x["group"]==group),
             "n_fetch_failure":sum(x["status"]=="PR_FETCH_FAILURE" for x in rows if x["group"]==group)}
            for group in GROUPS],
  "rows":rows}
 with open("pcpr_v09_new_32_frozen_first_score.json","w") as f:
  json.dump(result,f,indent=2,sort_keys=True)
 print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
 for row in rows:
  print(row["group"],row["url"],row["status"],"v09",row["v09_native_contexts"],
        "recover",row["v09_recovered_contexts"],"dtype_gain",row["v09_dtype_reference_gain_contexts"])
 return result
if __name__=="__main__":main()
