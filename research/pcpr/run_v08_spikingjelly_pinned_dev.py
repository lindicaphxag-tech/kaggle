"""Retrospective REAL upstream native-operator replay on spikingjelly #769.

This PR is part of the ALREADY INSPECTED first 48 cohort and was used to
diagnose the v0.7 failure. Therefore v0.8 success here is DEVELOPMENT ONLY,
not independent generalization. SHA-pinned whole original Python sources
are parsed but NEVER imported/executed; only whitelisted factory AST runs.
"""
import hashlib,json,urllib.request
from symbolic_native_operator_v08 import score_file

REPO="fangwei123456/spikingjelly"
FILE="spikingjelly/activation_based/neuron/psn.py"
BASE_COMMIT="c6cb8e46738bf6010cb94904fe5995e66d6451fc"
HEAD_COMMIT="f1271adcbeff6aef3b510b3ee565773be9a0d34f"
BASE_BLOB="5df72dccbc4d816018f79e6450f3c3dd2dc09303"
HEAD_BLOB="3cc629ec2b2228cbb2cd8d51ed4299353fb0ba70"

def git_blob(data):
 return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
def pinned(commit,expected):
 url=f"https://raw.githubusercontent.com/{REPO}/{commit}/{FILE}"
 req=urllib.request.Request(url,headers={"User-Agent":"pcpr-retrospective-spikingjelly-source-only"})
 with urllib.request.urlopen(req,timeout=80) as r:data=r.read()
 if git_blob(data)!=expected:raise ValueError("ORIGINAL_GIT_BLOB_MISMATCH")
 return data.decode("utf8")
def main():
 a=pinned(BASE_COMMIT,BASE_BLOB);b=pinned(HEAD_COMMIT,HEAD_BLOB)
 result=score_file(FILE,a,b)
 if result["n_native_contexts"]<4:
  raise AssertionError("Source-exact native operator not replayed in 4 contexts")
 if result["n_dtype_gain_contexts"]<3:
  raise AssertionError("Original source half/double mismatch not exposed")
 if result["n_preserving_contexts"]<1:
  raise AssertionError("Preserving FP32 context not exposed")
 output={
  "status":"RETROSPECTIVE_DEVELOPMENT_OPERATOR_ONLY",
  "upstream":f"https://github.com/{REPO}/pull/769",
  "pinned_base_blob":BASE_BLOB,
  "pinned_head_blob":HEAD_BLOB,
  "n_original_prs":1,"n_independently_unseen_prs":0,
  "n_native_contexts":result["n_native_contexts"],
  "n_dtype_gain_contexts":result["n_dtype_gain_contexts"],
  "n_preserving_contexts":result["n_preserving_contexts"],
  "n_whole_patch_correctness_decisions":0,
  "original_method_report":result}
 with open("pcpr_v08_spikingjelly_pinned_dev.json","w") as f:json.dump(output,f,indent=2)
 print(json.dumps({k:v for k,v in output.items() if k!="original_method_report"},indent=2))
 return output

if __name__=="__main__":main()
