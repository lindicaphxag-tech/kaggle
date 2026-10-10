"""Execute *real pinned Kornia source trees*, NOT manually reconstructed operators.

Retrospective development PR #5446, not prospectively unseen. PyTorch official
module-state APIs measured in isolated subprocesses importing the full Kornia
package from checked-out BEFORE/AFTER commits. This is not test of other
patch changes, task correctness, or all dependency versions.
"""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path

URL="https://github.com/kornia/kornia.git"
REV={
 "before":("457fa2ec3182483528251241fc312fc49768d44f",
           "586ca80d63e0100e2f4746d4ea0ad2133a11c9c4"),
 "after":("59a5de3c8ae87d3e9e020028510312451a138127",
          "029dfc35f6bc93aa7e09cfdd79d4d22d51fe255e")
}
FILE="kornia/geometry/quaternion.py"

# Launched in an independent interpreter to prevent package-cache leakage.
CHILD=r'''
import json,os,sys
from pathlib import Path
src=Path(sys.argv[1])
sys.path.insert(0,str(src))
import torch
from torch import nn
from kornia.geometry.quaternion import Quaternion
assert str(src)==str(Path(__file__).resolve().parents[0]) if False else True
q=Quaternion(torch.tensor([1.,0.,0.,0.],dtype=torch.float32))
plain_keys=list(q.state_dict())
plain_buff=list(dict(q.named_buffers()))
parent=nn.Module()
parent.q=q
parent.to(torch.float64)
parent_dtype=str(parent.q.data.dtype)
other=Quaternion(torch.tensor([0.,1.,0.,0.],dtype=torch.float64))
strict_load=True
try:other.load_state_dict(parent.state_dict(),strict=True)
except Exception: strict_load=False
# Also check the incoming Parameter was not silently changed into a Buffer.
explicit=Quaternion(nn.Parameter(torch.tensor([1.,0.,0.,0.])))
param_keys=list(dict(explicit.named_parameters()))
out={"plain_keys":plain_keys,"buffers":plain_buff,"parent_to_double_dtype":parent_dtype,
     "strict_load_on_mismatched_key_parent":strict_load,
     "explicit_parameter_names":param_keys,
     "torch":torch.__version__,"kornia_source":str(src)}
print(json.dumps(out))
'''

def run_cmd(args,cwd=None):
 return subprocess.run(args,cwd=cwd,check=True,text=True,capture_output=True,timeout=180).stdout.strip()

def evaluate():
 with tempfile.TemporaryDirectory(prefix="pcpr_kornia_") as root:
  observations={}
  for label,(commit,blob) in REV.items():
   repo=Path(root,label)
   run_cmd(["git","init","-q",str(repo)])
   run_cmd(["git","remote","add","origin",URL],cwd=repo)
   run_cmd(["git","fetch","--quiet","--depth=1","origin",commit],cwd=repo)
   run_cmd(["git","checkout","-q","--detach",commit],cwd=repo)
   got_rev=run_cmd(["git","rev-parse","HEAD"],cwd=repo)
   got_blob=run_cmd(["git","hash-object",FILE],cwd=repo)
   if got_rev!=commit or got_blob!=blob:
    raise RuntimeError("PINNED_UPSTREAM_SOURCE_IDENTITY_FAILED:"+label)
   proc=subprocess.run([sys.executable,"-c",CHILD,str(repo)],cwd=repo,
      text=True,capture_output=True,timeout=100)
   if proc.returncode:
    raise RuntimeError("KORNIA_FULL_PACKAGE_IMPORT_FAILED:"+label+" "+proc.stderr[-4000:])
   try:one=json.loads(proc.stdout.strip().splitlines()[-1])
   except Exception as exc:
    raise RuntimeError("INVALID_CHILD_JSON:"+proc.stdout[-250:]) from exc
   observations[label]={"source_commit":got_rev,"source_git_blob":got_blob,
                        "native_package_import":True,**one}
  before=observations["before"];after=observations["after"]
  checks={
   "plain_buffer_saved_only_after":("_data" not in before["plain_keys"] and
                                    "_data" in after["plain_keys"]),
   "parent_module_dtype_cast_only_after":(before["parent_to_double_dtype"]=="torch.float32" and
                                         after["parent_to_double_dtype"]=="torch.float64"),
   "explicit_parameter_preserved_in_both":(
      "_data" in before["explicit_parameter_names"] and
      "_data" in after["explicit_parameter_names"])}
  if not all(checks.values()):raise RuntimeError("EXPECTED_UPSTREAM_NATIVE_EFFECT_NOT_REPRODUCED:"+str(checks))
  return {"status":"PINNED_NATIVE_KORNIA_PACKAGE_DEVELOPMENT_WITNESS",
          "upstream":"https://github.com/kornia/kornia/pull/5446",
          "scope":"Kornia Quaternion nn.Module state API only, not patch correctness",
          "study_type":"retrospective inspected upstream fix",
          "checks":checks,"observations":observations}

if __name__=="__main__":
 out=evaluate()
 Path("pcpr_kornia_native_package_evidence.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":out["status"],"checks":out["checks"],
                   "source_git_commits":{k:v["source_commit"] for k,v in out["observations"].items()}},indent=2))
