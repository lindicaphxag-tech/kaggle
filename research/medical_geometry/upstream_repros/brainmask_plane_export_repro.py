"""Exact-source, real-medical-image upstream brain_mask plane-export audit.

The audited functions come from AST-extracted original upstream source. A stub
ONLY replaces model patch reconstruction to isolate the exporter boundary:
the supplied existing brain mask is a transport sentinel, not a learned
prediction or expert label. Source and resulting file bytes are pinned.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from scipy.ndimage import affine_transform


SOURCE_COMMIT = "ca789fb26b70e359b61a8d52968813dd772fb4b1"
SOURCE_BLOBS = {
    "predict.py": "d4b03f5b851928ead5ef014af75cd1dd5fbd64dc",
    "input_fn_util.py": "1c2ce1a9f13f690cdaf28eac5b003e44ec06d192",
    "real_t1.nii.gz": "81b1e5260ee4a579d8ac490cac113df9c1cb942f",
    "existing_mask.nii.gz": "e0dd089b3544da4352e751a24dc27a828904bad7",
}


def blobid(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode("ascii")+b"\0"+b).hexdigest()


def compiled_function(path: Path, name: str, env: dict):
    """Compile the ORIGINAL one named function; no reimplementation of its logic."""
    root = ast.parse(path.read_text())
    matches=[node for node in root.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
             and node.name==name]
    if len(matches)!=1:
        raise RuntimeError(f"Upstream function {name} missing or ambiguous")
    code=compile(ast.Module(body=matches,type_ignores=[]),str(path),"exec")
    globenv={"__builtins__":__builtins__,**env}
    exec(code,globenv)
    return globenv[name]


def dsc(a,b):
    x,y=np.asarray(a,bool),np.asarray(b,bool)
    den=int(x.sum())+int(y.sum())
    return 1. if den==0 else 2*int(np.logical_and(x,y).sum())/den


def regrid_to_original(array, from_affine, original):
    t=np.linalg.solve(from_affine,np.asarray(original.affine))
    nearest=np.rint(t)
    t=np.where(abs(t-nearest)<1e-10,nearest,t)
    return affine_transform(np.asarray(array,np.uint8),t[:3,:3],
        offset=t[:3,3],output_shape=original.shape,order=0,
        cval=0,mode="constant",prefilter=False).astype(bool)


def center(mask,affine):
    idx=np.argwhere(mask)
    if not len(idx):
        raise ValueError("No foreground after world-grid reference alignment")
    return idx.mean(axis=0)@affine[:3,:3].T+affine[:3,3]


def render(image, source_mask, before, after, output):
    img=np.asarray(image.dataobj)
    z=int(np.argmax(source_mask.sum(axis=(0,1)))) # fixed from original label, no result selection
    sl=img[:,:,z].T.astype(float)
    lo,hi=np.percentile(sl[np.isfinite(sl)],[1,99])
    sl=np.clip((sl-lo)/max(float(hi-lo),1e-10),0,1)
    fig,axes=plt.subplots(1,5,figsize=(16.8,4.8),constrained_layout=True)
    seq=[
        ("a  Real MRI + provided mask",source_mask),
        ("b  Upstream cor export",before["cor"]["reference_mask"]),
        ("c  Corrected cor export",after["cor"]["reference_mask"]),
        ("d  Upstream sag export",before["sag"]["reference_mask"]),
        ("e  Corrected sag export",after["sag"]["reference_mask"]),
    ]
    for ax,(name,mask) in zip(axes,seq):
        ax.imshow(sl,origin="lower",cmap="gray")
        for m,col,style in ((source_mask,"#15B9B3","solid"),(mask,"#E35145","--")):
            plane=np.asarray(m[:,:,z],float)
            if plane.min()!=plane.max():
                ax.contour(plane.T,levels=[0.5],colors=[col],linestyles=style,
                           linewidths=1.3,origin="lower")
        ax.set_title(name,fontsize=9.1)
        ax.set_axis_off()
    fig.text(.02,.012,
      "Actual source code + existing example brain mask (transport sentinel, not model output);"
      " source MRI fixed across every panel; no clinical segmentation claim",
      fontsize=8.3)
    fig.savefig(output,dpi=240,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-dir",type=Path,required=True)
    ap.add_argument("--study-dir",type=Path,required=True)
    ap.add_argument("--stage",choices=["before","after"],required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    predict=args.source_dir/"predict.py"
    loader=args.source_dir/"utilities"/"input_fn_util.py"
    maskfile=args.study_dir/"example001_brain_mask.nii.gz"
    imagefile=args.study_dir/"example001_t1.nii.gz"
    if blobid(loader)!=SOURCE_BLOBS["input_fn_util.py"]:
        raise RuntimeError("Unpinned upstream loader; evidence refuses to run")
    for file,key in ((maskfile,"existing_mask.nii.gz"),
                     (imagefile,"real_t1.nii.gz")):
        if blobid(file)!=SOURCE_BLOBS[key]:
            raise RuntimeError("Real source bytes mismatch pinned Git object")
    if args.stage=="before" and blobid(predict)!=SOURCE_BLOBS["predict.py"]:
        raise RuntimeError("Baseline original exporter is not pinned source")

    env={"nib":nib,"np":np,"scipy":scipy,"os":os,"glob":glob,
         "logging":logging,
         # Identical logical prediction is passed through the boundary unchanged.
         # All loss is therefore attributable to the exporter, not a model.
         "reconstruct_infer_patches_3d":lambda tensor,*_ :tensor}
    load=compiled_function(loader,"_load_single_study",env)
    export=compiled_function(predict,"predictions_2_nii",env)
    real=nib.load(str(imagefile))
    mask_nii=nib.load(str(maskfile))
    if real.shape!=mask_nii.shape or not np.allclose(real.affine,mask_nii.affine):
        raise ValueError("Paired source image/mask geometry mismatched")
    truth=np.asarray(mask_nii.dataobj)>0

    cases={}
    for plane in ("ax","cor","sag"):
        permuted=load(str(args.study_dir),["brain_mask"],data_format="channels_last",
                      plane=plane,norm=False,out_res=None,out_dims=None)
        testmask=(permuted[...,0]>0).astype(np.float32)
        outdir=args.out/args.stage/plane
        outdir.mkdir(parents=True,exist_ok=True)
        params=SimpleNamespace(data_prefix=["t1"],model_dir="original_t1",
                               data_plane=plane,resample_spacing=[],load_shape=[])
        filename=export(testmask,str(args.study_dir),str(outdir),params)
        output=nib.load(filename)
        arr=np.asarray(output.dataobj)>0
        onref=regrid_to_original(arr,output.affine,real)
        dice=dsc(truth,onref)
        drift=float(np.linalg.norm(center(truth,real.affine)-
                                   center(onref,real.affine)))
        if not np.array_equal(output.affine,real.affine):
            raise AssertionError("Unexpected changed source affine")
        if args.stage=="before" and plane=="ax" and not np.array_equal(arr,truth):
            raise AssertionError("Axial negative-control source transport failed")
        if args.stage=="before" and plane in ("cor","sag") and dice>=1.:
            raise AssertionError("Expected upstream nonaxial semantic defect absent")
        if args.stage=="after" and (dice!=1. or arr.shape!=truth.shape or
                                     not np.array_equal(arr,truth)):
            raise AssertionError("Fixed source exporter still mislabels patient-space")
        cases[plane]={
            "model_plane":plane,"source_grid_shape":list(real.shape),
            "raw_export_shape":list(output.shape),
            "fixed_original_patient_grid_dice":float(dice),
            "physical_centroid_displacement_mm":drift,
            "source_affine_reused_without_transformation":True,
            "raw_export_sha256":hashlib.sha256(
                np.ascontiguousarray(arr).tobytes()).hexdigest(),
        }

    if args.stage=="after":
        params=SimpleNamespace(data_prefix=["t1"],model_dir="original_t1",
             data_plane="ax",resample_spacing=[2,2,2],load_shape=[])
        for mode in ("resample_spacing","load_shape","invalid_plane"):
            params.resample_spacing=([2,2,2] if mode=="resample_spacing" else [])
            params.load_shape=([64,64,64] if mode=="load_shape" else [])
            params.data_plane=("other" if mode=="invalid_plane" else "ax")
            try:
                export(truth.astype(np.float32),str(args.study_dir),
                       str(args.out/args.stage/"ax"),params)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsafe geometry mode {mode} was silently exported")
        saved=json.loads((args.out/"before_results.json").read_text())
        before={}
        after={}
        for plane in ("ax","cor","sag"):
            bp=Path(saved["baseline_files"][plane])
            bn=nib.load(str(bp))
            before[plane]={"reference_mask":regrid_to_original(
                np.asarray(bn.dataobj)>0,bn.affine,real)}
            current=Path(args.out/args.stage/plane/
                         "example001_predictions_original_t1.nii.gz")
            an=nib.load(str(current))
            after[plane]={"reference_mask":regrid_to_original(
                np.asarray(an.dataobj)>0,an.affine,real)}
        render(real,truth,before,after,args.out/"real_mri_before_after_upstream_export.png")
    result={
        "stage":args.stage,"audit_upstream":"ecalabr/brain_mask",
        "upstream_source_commit":SOURCE_COMMIT,
        "pinned_original_loader_blob":SOURCE_BLOBS["input_fn_util.py"],
        "original_exporter_blob":SOURCE_BLOBS["predict.py"],
        "current_exporter_sha256":hashlib.sha256(predict.read_bytes()).hexdigest(),
        "source_image_blob":SOURCE_BLOBS["real_t1.nii.gz"],
        "source_existing_mask_blob":SOURCE_BLOBS["existing_mask.nii.gz"],
        "exact_original_upstream_loader_used":True,
        "exact_original_upstream_exporter_used":args.stage=="before",
        "reconstruction_boundary_stubbed":True,
        "source_mask_is_only_transport_sentinel":True,
        "clinical_result":False,
        "learned_model_inference_in_this_assay":False,
        "cases":cases
    }
    if args.stage=="before":
        result["baseline_files"]={
            p:str(args.out/args.stage/p/
                 "example001_predictions_original_t1.nii.gz")
            for p in cases}
    (args.out/f"{args.stage}_results.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({x:{
        "dice":round(cases[x]["fixed_original_patient_grid_dice"],6),
        "centroid_mm":round(cases[x]["physical_centroid_displacement_mm"],3),
        "raw_shape":cases[x]["raw_export_shape"]
    } for x in cases},sort_keys=True))


if __name__=="__main__":
    main()
