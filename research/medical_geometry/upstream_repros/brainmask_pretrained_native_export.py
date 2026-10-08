"""Native learned-inference and native patch-reconstruction test of MRI export.

Uses the original T1 Unet3dBneck checkpoint, the original source reader, native
TensorFlow patch extraction and inverse reconstruction, and original or
one-file-patched NIfTI exporters. This separates two claims:
(1) actual learned predictions are produced; (2) *given the very same
predictions*, exporting them without inverse plane reorientation is incorrect.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,logging,os,sys
from glob import glob
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

SOURCE="ca789fb26b70e359b61a8d52968813dd772fb4b1"
BLOBS={
 "source_predict":"d4b03f5b851928ead5ef014af75cd1dd5fbd64dc",
 "source_input":"1c2ce1a9f13f690cdaf28eac5b003e44ec06d192",
 "source_normalizer":"98eb8c4329ed2330b47905ab2e69ac1a5fdb5237",
 "source_config":"7dcc0cc0fa3b2b19e3b28eac47ab983c502b2022",
 "source_weights":"6ac54600739585f54d9f612d13264b74965ffd33",
 "source_mri":"81b1e5260ee4a579d8ac490cac113df9c1cb942f",
}
def blob(path):
 b=Path(path).read_bytes()
 return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def check(path,expected):
 actual=blob(path)
 if actual!=expected:raise ValueError(f"Unpinned source {path}: {actual}")
def compiled_export(source,util):
 tree=ast.parse(Path(source).read_text())
 defs=[node for node in tree.body if isinstance(node,ast.FunctionDef)
       and node.name=="predictions_2_nii"]
 if len(defs)!=1:raise RuntimeError("native export method not found")
 env={"nib":nib,"np":np,"os":os,"logging":logging,"glob":glob,
      "reconstruct_infer_patches_3d":util.reconstruct_infer_patches_3d,
      "__builtins__":__builtins__}
 exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),"exec"),env)
 return env["predictions_2_nii"]
def dice(a,b):
 a,b=np.asarray(a,bool),np.asarray(b,bool)
 n=int(a.sum()+b.sum())
 return 1. if not n else 2*int((a&b).sum())/n
def inverse_plane(x,plane):
 if plane=="ax":return np.asarray(x)
 if plane=="cor":return np.transpose(x,(0,2,1))
 if plane=="sag":return np.transpose(x,(2,0,1))
 raise ValueError(plane)
def main():
 p=argparse.ArgumentParser()
 for n in ("source-dir","original-export","image","checkpoint","config","out"):
  p.add_argument("--"+n,required=True,type=Path)
 a=p.parse_args()
 src=a.source_dir.resolve()
 for fn,key in (("utilities/input_fn_util.py","source_input"),
                ("utilities/normalizers.py","source_normalizer")):
  check(src/fn,BLOBS[key])
 for path,key in ((a.original_export,"source_predict"),
                  (a.image,"source_mri"),(a.checkpoint,"source_weights"),
                  (a.config,"source_config")):
  check(path,BLOBS[key])
 sys.path.insert(0,str(src))
 import tensorflow as tf
 from utilities import input_fn_util as util
 from model.networks import Networks
 tf.keras.backend.clear_session()
 tf.random.set_seed(17)
 params=SimpleNamespace(**yaml.safe_load(a.config.read_text()))
 params.batch_size=1
 params.train_dims=[64,64,64]
 params.infer_dims=[64,64,64]
 params.infer_patch_overlap=[1,1,1]
 params.policy=tf.keras.mixed_precision.Policy("float32")
 model=Networks(params)()
 before=[w.numpy().copy() for w in model.weights]
 model.load_weights(str(a.checkpoint))
 changed=sum(not np.array_equal(old,w.numpy()) for old,w in zip(before,model.weights))
 if changed<1:raise AssertionError("Official pretrained weights not loaded")
 weights_digest=hashlib.sha256(
  b"".join(np.ascontiguousarray(w.numpy()).tobytes() for w in model.weights)
 ).hexdigest()

 acquired=nib.load(str(a.image))
 image=np.asarray(acquired.dataobj,dtype=np.float32)
 corner=(np.array(image.shape)-64)//2
 sl=tuple(slice(int(v),int(v)+64) for v in corner)
 affine=np.asarray(acquired.affine)
 H=np.eye(4);H[:3,3]=corner
 A=affine@H
 a.out.mkdir(parents=True,exist_ok=True)
 study=a.out/"work"/"case_000"
 study.mkdir(parents=True,exist_ok=True)
 nib.save(nib.Nifti1Image(image[sl],A),str(study/"sample_t1.nii.gz"))
 native_export=compiled_export(a.original_export,util)
 patched_export=compiled_export(src/"predict.py",util)
 report={
 "upstream_source_commit":SOURCE,
 "source_sha256":hashlib.sha256(a.image.read_bytes()).hexdigest(),
 "checkpoint_sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
 "checkpoint_changed_parameter_variables":changed,
 "loaded_parameter_digest":weights_digest,
 "true_pretrained_model_inference":True,
 "actual_upstream_patch_extract_and_reconstruct":True,
 "original_export_function_from_pinned_upstream_ast":True,
 "source_example_is_anonymized":True,
 "clinical_ground_truth":False,
 "no_loss_repair_claim":False,
 "experimental_plane_change_changes_model_input_orientation":True,
 "cases":{},
 "tensorflow_version":tf.__version__,
 }
 figure_data={}
 for plane in ("ax","cor","sag"):
  # Original upstream reader with original upstream z-score normalizer
  x=util._load_single_study(str(study),["t1"],data_format="channels_last",
                            plane=plane,norm=True,norm_mode="zscore",
                            out_res=None,out_dims=None)
  if x.shape!=(64,64,64,1):raise AssertionError("Unexpected native loader shape")
  packed=util._tf_patch_3d(tf.convert_to_tensor(x[None,...],tf.float32),
                           [1,64,64,64,1],[1,64,64,64,1],padding="VALID")
  patches=tf.reshape(packed,[-1,64,64,64,1])
  y=model(patches,training=False)
  if tuple(y.shape)!=(1,64,64,64,1):raise AssertionError("Unexpected trained output")
  yval=np.asarray(y.numpy())
  if not np.isfinite(yval).all():raise AssertionError("Nonfinite learned inference")
  upstream_params=SimpleNamespace(
   data_prefix=["t1"],data_plane=plane,model_dir="original_t1",
   infer_dims=[64,64,64],infer_patch_overlap=[1,1,1],
   output_filters=1,data_format="channels_last",
   label_prefix=["brain_mask"],norm_data=True,norm_mode="zscore",
   resample_spacing=[],load_shape=[]
  )
  recon=util.reconstruct_infer_patches_3d(y,str(study),upstream_params)
  if not np.allclose(np.asarray(recon),yval[0,...,0],atol=1e-6):
   raise AssertionError("Original native TF reconstruction does not recover trained model output")
  expected=inverse_plane(recon,plane)
  learned_binary=expected>=.5
  if not learned_binary.any() or learned_binary.all():
   raise AssertionError("Predeclared .5 threshold makes learned prediction degenerate: "+plane)
  exports={}
  plane_vis={"reference":learned_binary}
  for label,fn in (("original",native_export),("patched",patched_export)):
   directory=a.out/"exports"/plane/label
   directory.mkdir(parents=True,exist_ok=True)
   path=fn(y,str(study),str(directory),upstream_params)
   saved=nib.load(str(path))
   saved_binary=np.asarray(saved.dataobj)>=.5
   if saved.shape!=(64,64,64) or not np.allclose(saved.affine,A,atol=1e-5):
    raise AssertionError("Unexpected model export grid")
   agreement=dice(saved_binary,learned_binary)
   plane_vis[label]=saved_binary
   if label=="patched" and agreement!=1:
    raise AssertionError("Patched native exporter did not preserve real learned output frame: "+plane)
   if label=="original" and plane=="ax" and agreement!=1:
    raise AssertionError("Native axial export negative control failed")
   if label=="original" and plane!="ax" and agreement>=1:
    raise AssertionError("Expected actual learned nonaxial export failure absent: "+plane)
   exports[label]={
    "dice_to_correctly_reoriented_same_model_prediction":agreement,
    "exported_shape":list(saved.shape),
    "source_affine_attached":True,
    "foreground_voxels":int(saved_binary.sum()),
    "saved_probability_sha256":hashlib.sha256(
     np.ascontiguousarray(np.asarray(saved.dataobj,dtype=np.float32)).tobytes()
    ).hexdigest(),
   }
  figure_data[plane]=plane_vis
  report["cases"][plane]={
   "actual_checkpoint_prediction_sha256":hashlib.sha256(
     np.ascontiguousarray(yval).tobytes()
   ).hexdigest(),
   "trained_model_foreground_count_in_model_plane":int((yval>=.5).sum()),
   "reconstruction_exact_to_real_model_output":True,
   "export_evidence":exports,
  }
 # Real-image 2x3 figure. All panels use the identical MRI grid, fixed
 # z=32, not an output-selected 'best looking' slice.
 fig,axes=plt.subplots(2,3,figsize=(13.8,8),constrained_layout=True)
 image_slice=np.asarray(image[sl][:,:,32],np.float64).T
 finite=image_slice[np.isfinite(image_slice)]
 lo,hi=np.percentile(finite,[1,99])
 background=np.clip((image_slice-lo)/max(hi-lo,1e-8),0,1)
 for row,plane in enumerate(("cor","sag")):
  vis=figure_data[plane]
  for col,(kind,label) in enumerate((
    ("reference","True model output in original grid"),
    ("original","Original upstream NIfTI export"),
    ("patched","Inverse-plane repaired export"),
  )):
   ax=axes[row,col]
   ax.imshow(background,cmap="gray",origin="lower",interpolation="nearest")
   for layer,color,line in (("reference","#15B0A8","solid"),
                            (kind,"#DB5A4A","dashed")):
    mask=vis[layer][:,:,32].astype(float).T
    if np.any(mask>.5) and np.any(mask<.5):
     ax.contour(mask,levels=[.5],colors=[color],
                linestyles=line,linewidths=1.25,origin="lower")
   ax.set_title(("Coronal" if row==0 else "Sagittal")+" | "+label,fontsize=9)
   ax.axis("off")
 fig.suptitle(
  "Authentic 3-D U-Net on acquired T1 MRI | same prediction, different export semantics",
  fontsize=12
 )
 fig.text(.02,.005,
  "Teal = same frozen learned prediction mapped into original patient grid; "
  "dashed red = exported mask. Original NIfTI affine fixed. "
  "Controlled plane selection; no clinical annotation.",
  fontsize=8.0,color="#333333")
 fig.savefig(a.out/"native_pretrained_export_figure.png",dpi=230,
             bbox_inches="tight",facecolor="white")
 plt.close(fig)
 report["visualization"]={
  "file":"native_pretrained_export_figure.png",
  "slice":"predeclared original MRI crop z=32",
  "real_input":"hash-pinned acquired T1 MRI",
  "model_output":"actual strictly loaded official HDF5 3D U-Net",
  "clinical_label":"none",
  "overlay_semantics":"same learned output vs original/patched export on one voxel/world grid"
 }
 (a.out/"native_pretrained_plane_export.json").write_text(
  json.dumps(report,indent=2,sort_keys=True)+"\n"
 )
 print(json.dumps({
  k:{v:round(c["export_evidence"][v]["dice_to_correctly_reoriented_same_model_prediction"],7)
    for v in ("original","patched")}
  for k,c in report["cases"].items()
 },sort_keys=True))
if __name__=="__main__":main()
