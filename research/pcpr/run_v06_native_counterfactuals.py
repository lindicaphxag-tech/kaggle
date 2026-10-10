"""Native replay of GENERIC single-edit counterfactuals on pinned upstream source.

All 3 targets are previously inspected development patches, NOT held-out.
The intervention operators are generic; context inputs are manually chosen.
Full package correctness, automatic oracle discovery and prospective defect
detection remain UNPROVEN.
"""
import json
import torch
from native_ast_lift_v03 import fetch_ref,lift_function,lift_constructor
from semantic_interventions_v06 import mutate

def line_trial():
 src=fetch_ref("line","after")
 edited,manifest=mutate(src,"_fit_line_ols_2d","drop_tensor_factory_dtype")
 original=lift_function(src,"_fit_line_ols_2d")
 mutant=lift_function(edited,"_fit_line_ols_2d")
 outcomes=[]
 for dtype in (torch.float16,torch.bfloat16,torch.float32):
  p=torch.tensor([[[3.,0.],[3.,1.],[3.,2.]]],dtype=dtype)
  actual=original(p).direction
  bad=mutant(p).direction
  outcomes.append({"input":str(dtype),"real_after":str(actual.dtype),
                   "mutant":str(bad.dtype),
                   "negative_control_exposed":dtype in (torch.float16,torch.bfloat16)
                   and bad.dtype!=dtype and actual.dtype==dtype,
                   "preserving_context":dtype==torch.float32
                   and bad.dtype==actual.dtype})
 if not all(x["negative_control_exposed"] for x in outcomes[:2]):
  raise AssertionError("Original-source dtype-removal mutants not distinguished")
 if not outcomes[-1]["preserving_context"]:
  raise AssertionError("Expected safe-control dtype not preserved")
 return {"source":"kornia/kornia#5026","mutation":manifest,
         "contexts":outcomes,"exact_function_AST":True,
         "n_negative_contexts":2,"n_preserving_contexts":1}

def ctc_trial():
 src=fetch_ref("ctc","after")
 edited,manifest=mutate(src,"ctc_forced_align","drop_clone")
 original=lift_function(src,"ctc_forced_align")
 mutant=lift_function(edited,"ctc_forced_align")
 frames=torch.tensor([
  [-1.99,-1.09,-2.18,-1.83,-1.36],
  [-.29,-4.06,-4.18,-3.26,-1.72],
  [-.93,-1.8,-2.21,-3.37,-1.44],
  [-4.04,-4.09,-2.9,-2.12,-.23],
  [-3.59,-3.29,-.24,-2.61,-2.63],
 ],dtype=torch.float32).unsqueeze(0)
 def run(fn):
  tar=torch.tensor([[1,-1]])
  pre=tar.clone()
  out=fn(frames.clone(),tar,torch.tensor([5]),torch.tensor([2]))
  return {"mutated_input":bool(not torch.equal(tar,pre)),
          "alignment_length":list(out.shape)}
 good=run(original)
 wrong=run(mutant)
 if good["mutated_input"] or not wrong["mutated_input"]:
  raise AssertionError("Original-source clone-removal control not exposed")
 return {"source":"QwenAudio/SenseVoice#358","mutation":manifest,
         "actual":good,"counterfactual":wrong,"exact_function_AST":True,
         "negative_control_exposed":True}

def quaternion_trial():
 src=fetch_ref("quaternion","after")
 edited,manifest=mutate(src,"Quaternion.__init__","demote_buffer_registration")
 good=lift_constructor(src)
 bad=lift_constructor(edited)
 t=torch.tensor([1.,0.,0.,0.])
 def trace(cls):
  m=torch.nn.Module();m.quat=cls(t.clone())
  state=list(m.state_dict())
  m.to(dtype=torch.float64)
  return {"checkpoint_keys":state,
          "dtype_moved":m.quat._data.dtype==torch.float64}
 actual=trace(good);counter=trace(bad)
 if not actual["checkpoint_keys"] or counter["checkpoint_keys"] or not actual["dtype_moved"] or counter["dtype_moved"]:
  raise AssertionError("Original-source buffer removal not exposed")
 # Explicit Parameter input should remain a parameter in both.
 after_param=good(torch.nn.Parameter(t.clone()))
 wrong_param=bad(torch.nn.Parameter(t.clone()))
 preserving=("_data" in dict(after_param.named_parameters())
             and "_data" in dict(wrong_param.named_parameters()))
 if not preserving:raise AssertionError("Parameter preserving control failed")
 return {"source":"kornia/kornia#5446","mutation":manifest,
         "actual":actual,"counterfactual":counter,
         "explicit_Parameter_preserving_control":preserving,
         "negative_control_exposed":True}

def main():
 result={"status":"PINNED_DEVELOPMENT_ORIGINAL_AST_NATIVE_COUNTERFACTUAL",
         "independent_new_PRs":0,"whole_patch_correctness_claims":0,
         "native_package_end_to_end":0,
         "cases":{"line":line_trial(),"ctc":ctc_trial(),"quaternion":quaternion_trial()}}
 result["mutations_exposed"]=3
 with open("pcpr_v06_native_single_edit_counterfactuals.json","w") as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
 for case,val in result["cases"].items():
  print(case,val["mutation"]["operator"],val["mutation"]["output_sha256"],
        val.get("negative_control_exposed", val.get("n_negative_contexts")))
 return result
if __name__=="__main__":main()
