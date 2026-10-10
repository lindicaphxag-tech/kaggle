"""Kornia #5005 exact-source *full-package* So2 pairwise geometry behavior.

RETROSPECTIVE inspected upstream case, NOT prospective transfer success or a
complete software patch correctness proof. Independent subprocess imports
full repository package from validated commit trees.
"""
import json,subprocess,sys,tempfile
from pathlib import Path
from run_native_kornia_5446 import URL,run_cmd

REVS={
 "before":("5f4271d84f4968af187857ceed047f250e9b451c","67f21a6c518aa16ac3743777f9222a0baa1c2b90"),
 "after":("668b7b2e021e0587d0d39f15350f5483985f132c","767ba8f272fee243edd3d34de8f1bbcf5e26b8c6")
}
FILE="kornia/geometry/liegroup/so2.py"

CODE=r'''
import json,sys,torch
sys.path.insert(0,sys.argv[1])
from kornia.geometry.liegroup import So2
theta=torch.tensor([.3,.5,.7])
pts=torch.tensor([[1.,0.],[0.,1.],[2.,3.]])
col=theta[:,None]
flat=So2.exp(theta)
column=So2.exp(col)
pair=flat*pts
out=column*pts
result={"flat_expected_shape":list(pair.shape),
        "column_observed_shape":list(out.shape),
        "flat_column_aligned":bool(torch.allclose(out,pair)) if out.shape==pair.shape else None,
        "native_class":str(type(column))}
print(json.dumps(result))
'''

def run():
 with tempfile.TemporaryDirectory(prefix="pcpr_so2_") as root:
  obs={}
  for label,(rev,blob) in REVS.items():
   p=Path(root,label)
   run_cmd(["git","init","-q",str(p)])
   run_cmd(["git","remote","add","origin",URL],cwd=p)
   run_cmd(["git","fetch","-q","--depth=1","origin",rev],cwd=p)
   run_cmd(["git","checkout","-q","--detach",rev],cwd=p)
   if run_cmd(["git","rev-parse","HEAD"],cwd=p)!=rev or run_cmd(["git","hash-object",FILE],cwd=p)!=blob:
    raise RuntimeError("SO2_SOURCE_IDENTITY_MISMATCH")
   proc=subprocess.run([sys.executable,"-c",CODE,str(p)],cwd=p,
                       capture_output=True,text=True,timeout=100)
   if proc.returncode:raise RuntimeError("SO2_NATIVE_PACKAGE_IMPORT_FAILED:"+label+":"+proc.stderr[-3000:])
   obs[label]={"commit":rev,"blob":blob,**json.loads(proc.stdout.strip().splitlines()[-1])}
  checks={"old_broadcasts_outer":obs["before"]["column_observed_shape"]==[3,3,2],
          "after_produces_pairwise":obs["after"]["column_observed_shape"]==[3,2],
          "after_matches_flat_layout":obs["after"]["flat_column_aligned"] is True}
  if not all(checks.values()):raise RuntimeError("SOURCE_NATIVE_SO2_EXPECTATION_FAIL:"+str(checks))
  return {"status":"KORNIA_SO2_TRUE_PACKAGE_NATIVE_DEV_CASE","upstream":"https://github.com/kornia/kornia/pull/5005",
          "checked":checks,"versions":obs,"scientific_scope":
          "one inspected geometry API pairwise broadcast transition, not general patch verification"}
if __name__=="__main__":
 result=run()
 Path("pcpr_kornia_so2_native.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":result["status"],"checks":result["checked"]},indent=2))
