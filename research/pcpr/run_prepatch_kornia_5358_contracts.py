"""Real pinned BEFORE-only test contract extraction, Kornia upstream #5358.

The post-patch PR added new tests explicitly about scalar precision.
We MUST refuse those new assertions as independent evidence. The existing
pre-patch TestAdjustGamma.test_cardinality asserts an output shape invariant
and survives this patch unchanged. This run verifies PROVENANCE ONLY, not
the truth/completeness of the assertion nor native patch correctness.
"""
from __future__ import annotations
import hashlib,json,urllib.request
from baseline_test_contracts_v10 import scan

REPO="kornia/kornia"
PATH="tests/enhance/test_adjust.py"
BASE="d8b2f8dc252d20d969946a96fda743b2a8f2543c"
HEAD="d456787e8cd227f2d6ad91e2793633dc5b7d9cde"
BEFORE_BLOB="a6d623ca1d272f7ee8e8d0c05b0be591624ba8a9"
AFTER_BLOB="a32ca875805a1cd8b630f9ecf306761bdcad907c"
def gitblob(data):
 return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
def fetch(rev,blob):
 url=f"https://raw.githubusercontent.com/{REPO}/{rev}/{PATH}"
 request=urllib.request.Request(url,headers={"User-Agent":"pcpr-prepatch-spec-miner/1.0"})
 with urllib.request.urlopen(request,timeout=70) as r:data=r.read()
 if gitblob(data)!=blob:raise ValueError("GIT_BLOB_MISMATCH")
 return data.decode("utf8")

def main():
 before=fetch(BASE,BEFORE_BLOB);after=fetch(HEAD,AFTER_BLOB)
 evidence=scan(before,after,PATH,"kornia.enhance.adjust_gamma")
 qualifying=[e for e in evidence["contracts"] if
             e["source_test"].endswith("TestAdjustGamma.test_cardinality")
             and e["property"]=="shape"]
 if not qualifying:
  raise AssertionError("Unchanged BEFORE TestAdjustGamma shape contract missing")
 if any("TestFloatScalarPrecision" in e["source_test"] for e in evidence["contracts"]):
  raise AssertionError("Patch-added precision assertions leaked into independent gold")
 result={
  "status":"RETROSPECTIVE_PINNED_PREPATCH_TEST_PROVENANCE_ONLY",
  "upstream":"https://github.com/kornia/kornia/pull/5358",
  "test_before_git_blob":BEFORE_BLOB,
  "test_after_git_blob":AFTER_BLOB,
  "n_prepatch_unchanged_target_shape_assertions":len(qualifying),
  "n_total_target_contract_candidates":evidence["n_contracts"],
  "n_preexisting_tests":evidence["n_preexisting_test_functions"],
  "n_unchanged_preexisting_tests":evidence["n_unchanged_preexisting_tests"],
  "n_changed_preexisting_tests_excluded":evidence["n_removed_or_changed_tests"],
  "n_added_patch_test_assertions_admitted_as_gold":0,
  "native_test_executions":0,
  "independent_requirement_truth_verified":False,
  "whole_patch_correctness_decisions":0,
  "candidate_receipts":evidence["contracts"]}
 with open("pcpr_prepatch_kornia_5358_contracts.json","w") as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!="candidate_receipts"},indent=2))
 return result
if __name__=="__main__":main()
