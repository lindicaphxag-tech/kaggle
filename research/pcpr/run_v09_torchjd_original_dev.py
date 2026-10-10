"""Retrospective original-source TorchJD #821 v0.9 dtype-flow native replay.

This PR was inspected as part of the original 48-PR v0.7 diagnostics.
Any recovery is DEVELOPMENT evidence, not independent new-PR generalization.
Uses original Git blob-pinned entire file only for AST extraction; does not
import/execute the TorchJD package or use developer patch tests as gold.
"""
from __future__ import annotations
import hashlib,json,urllib.request
from cross_expression_native_v09 import score_file

REPO="SimplexLab/TorchJD"
PATH="src/torchjd/autojac/_jac_to_grad.py"
OLD="68cd21b65ee1c0c1aaee58b5e2a95766061e7a84"
NEW="4d899114a1bb3bfc7df9ecb6b1199073f2bd6a45"
OLD_BLOB="e8c67f104d2cd226c78ed6cdec54cac41966d146"
NEW_BLOB="8ceb5641b7a6bc003179012cf8cab5b0bf57df05"
def blob(data):
 return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
def fetch(commit,expected):
 url=f"https://raw.githubusercontent.com/{REPO}/{commit}/{PATH}"
 req=urllib.request.Request(url,headers={"User-Agent":"pcpr-v09-pinned-original-torchjd"})
 with urllib.request.urlopen(req,timeout=70) as r:data=r.read()
 if blob(data)!=expected:raise ValueError("GIT_BLOB_IDENTITY_MISMATCH")
 return data.decode("utf8")
def main():
 before=fetch(OLD,OLD_BLOB);after=fetch(NEW,NEW_BLOB)
 r=score_file(PATH,before,after)
 if r["n_candidates"]<2:
  raise AssertionError("TorchJD source original cast changes not extracted")
 if r["n_native_contexts"]<3:
  raise AssertionError("Native PyTorch cast operation did not replay")
 out={"status":"RETROSPECTIVE_DEVELOPMENT_TORCHJD_821_REAL_SOURCE_NATIVE",
      "upstream":"https://github.com/SimplexLab/TorchJD/pull/821",
      "base_blob":OLD_BLOB,"head_blob":NEW_BLOB,
      "n_justified_new_PRs":0,"n_whole_patch_correctness_authorizations":0,
      "n_changed_cast_patterns":r["n_candidates"],
      "n_native_operator_contexts":r["n_native_contexts"],
      "n_restored_executions":r["n_recovered_execution"],
      "n_dtype_reference_gains":r["n_dtype_reference_gains"],
      "n_preserving_controls":r["n_preserving_controls"],
      "source_result":r}
 with open("pcpr_v09_torchjd_821_original_source.json","w") as f:json.dump(out,f,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k!="source_result"},indent=2))
 return out
if __name__=="__main__":main()
