"""Real TensorFlow patch reconstruction in pinned brain_mask NIfTI export.

The existing real source mask is a transport sentinel, not learned output.
Uses original upstream TensorFlow patch extraction AND reconstruction.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,logging,os,sys
from glob import glob
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np

CHECKS={
"predict.py":"d4b03f5b851928ead5ef014af75cd1dd5fbd64dc",
"utilities/input_fn_util.py":"1c2ce1a9f13f690cdaf28eac5b003e44ec06d192",
"utilities/normalizers.py":"98eb8c4329ed2330b47905ab2e69ac1a5fdb5237",
"mri":"81b1e5260ee4a579d8ac490cac113df9c1cb942f",
"mask":"e0dd089b3544da4352e751a24dc27a828904bad7",
}
def blob(path):
 b=Path(path).read_bytes()
 return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def exporter_from_exact_ast(source,util):
 roots=ast.parse(Path(source).read_text())
 fs=[n for n in roots.body if isinstance(n,ast.FunctionDef)
     and n.name=="predictions_2_nii"]
 if len(fs)!=1:raise RuntimeError("Original exporter missing")
 namespace={"np":np,"nib":nib,"os":os,"logging":logging,"glob":glob,
            "reconstruct_infer_patches_3d":util.reconstruct_infer_patches_3d,
            "__builtins__":__builtins__}
 exec(compile(ast.Module(body=fs,type_ignores=[]),str(source),"exec"),namespace)
 return namespace["predictions_2_nii"]
def dice(a,b):
 a,b=np.asarray(a,bool),np.asarray(b,bool)
 d=int(a.sum()+b.sum())
 return 1. if not d else 2*int(np.logical_and(a,b).sum())/d
def main():
 ap=argparse.ArgumentParser()
 for name in ("source-dir","image","mask","out"):
  ap.add_argument("--"+name,type=Path,required=True)
 ap.add_argument("--stage",choices=["before","after"],required=True)
 a=ap.parse_args()
 import tensorflow as tf
 for file in ("predict.py","utilities/input_fn_util.py","utilities/normalizers.py"):
  if file=="predict.py" and a.stage=="after":continue
  if blob(a.source_dir/file)!=CHECKS[file]:
   raise ValueError("Unpinned original source module: "+file)
 if blob(a.image)!=CHECKS["mri"] or blob(a.mask)!=CHECKS["mask"]:
  raise ValueError("Unpinned real MRI or paired mask bytes")
 sys.path.insert(0,str(a.source_dir.resolve()))
 from utilities import input_fn_util as util
 mri=nib.load(str(a.image));lab=nib.load(str(a.mask))
 img=np.asarray(mri.dataobj);mask=np.asarray(lab.dataobj)>0
 if img.shape!=mask.shape or not np.allclose(mri.affine,lab.affine):
  raise ValueError("Real input NIfTI geometry mismatch")
 extent=np.array([48,56,40])
 corner=(np.array(img.shape)-extent)//2
 sl=tuple(slice(int(o),int(o+n)) for o,n in zip(corner,extent))
 H=np.eye(4);H[:3,3]=corner
 affine=mri.affine@H
 work=a.out/"case"/"sample"
 work.mkdir(parents=True,exist_ok=True)
 nib.save(nib.Nifti1Image(img[sl],affine),str(work/"sample_t1.nii.gz"))
 truth=mask[sl]
 nib.save(nib.Nifti1Image(truth.astype(np.float32),affine),
          str(work/"sample_brain_mask.nii.gz"))
 export=exporter_from_exact_ast(a.source_dir/"predict.py",util)
 outcomes={}
 for plane in ("ax","sag"):
  loaded=util._load_single_study(str(work),["brain_mask"],
            data_format="channels_last",plane=plane,norm=False)
  x=tf.convert_to_tensor(loaded[None,...],dtype=tf.float32)
  patch=[16,16,16];kernel=[1]+patch+[1];stride=[1]+patch+[1]
  packed=util._tf_patch_3d(x,kernel,stride,padding="VALID")
  patches=tf.reshape(packed,[-1]+patch+[1])
  params=SimpleNamespace(
   data_prefix=["t1"],data_plane=plane,model_dir="official_t1",
   infer_dims=patch,infer_patch_overlap=[1,1,1],output_filters=1,
   data_format="channels_last",label_prefix=["brain_mask"],
   norm_data=False,norm_mode="zscore",resample_spacing=[],load_shape=[]
  )
  recon=util.reconstruct_infer_patches_3d(patches,str(work),params)
  equal=np.array_equal(np.asarray(recon)>0,loaded[...,0]>0)
  if not equal:raise AssertionError("Native TensorFlow patch reconstruction failed: "+plane)
  outdir=a.out/a.stage/plane
  outdir.mkdir(parents=True,exist_ok=True)
  saved=nib.load(export(patches,str(work),str(outdir),params))
  v=np.asarray(saved.dataobj)>0
  shape_matches=saved.shape==tuple(extent)
  sim=dice(v,truth) if shape_matches else None
  if plane=="ax" and (not shape_matches or sim!=1.):
   raise AssertionError("Axial control unexpectedly changed")
  if plane=="sag" and a.stage=="before" and shape_matches:
   raise AssertionError("Original sagittal defect not reproduced")
  if plane=="sag" and a.stage=="after" and (not shape_matches or sim!=1.):
   raise AssertionError("Sagittal inverse repair failed")
  outcomes[plane]={
    "model_plane_shape":list(loaded.shape[:3]),
    "patch_tensor_shape":[int(n) for n in patches.shape],
    "native_tf_reconstruction_exact":equal,
    "exported_shape":list(saved.shape),
    "source_shape":[int(n) for n in extent],
    "source_shape_matches":shape_matches,
    "array_dice_when_shapes_match":sim,
    "source_affine_reused":bool(np.allclose(saved.affine,affine,atol=1e-5,rtol=0)),
  }
 report={"stage":a.stage,"upstream_commit":"ca789fb26b70e359b61a8d52968813dd772fb4b1",
         "tensor_flow_version":tf.__version__,
         "real_mri_blob":CHECKS["mri"],"real_existing_mask_blob":CHECKS["mask"],
         "uses_original_native_tf_patch_extraction":True,
         "uses_original_native_tf_patch_reconstruction":True,
         "exporter_is_original_unmodified":a.stage=="before",
         "model_inference_executed":False,"provided_mask_is_transport_sentinel":True,
         "crop_extent_voxels":[int(n) for n in extent],"outcomes":outcomes}
 a.out.mkdir(parents=True,exist_ok=True)
 (a.out/f"{a.stage}_native_tf.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(outcomes,sort_keys=True))
if __name__=="__main__":main()
