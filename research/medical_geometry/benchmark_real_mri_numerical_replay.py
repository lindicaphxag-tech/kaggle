"""Executed CPU interpolation witness on real, hash-pinned NiBabel MRI.

The source is an actual acquired structural MRI from NiBabel docs. This proves
one execution's numerical identity, NOT clinical quality or learned inference.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha1,sha256
from pathlib import Path
import argparse,json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import torch

from .clingeom import FeatureLattice
from .repair_synthesis import synthesize_lattice_repair
from .torch_repair import LatticeRepair3d
from .executed_interpolation_witness import certify_executed_interpolation

SOURCE_REPO="https://github.com/nipy/nibabel"
SOURCE_COMMIT="d6a0ba0056445c06407afb421749a8ed5c939c56"
SOURCE_BLOB="b7e917e3f891c74fd99cdf82ec579ee254a195d0"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    data=args.image.read_bytes()
    blob=sha1(b"blob "+str(len(data)).encode("ascii")+b"\0"+data).hexdigest()
    if blob!=SOURCE_BLOB:
        raise ValueError("genuine NiBabel example MRI fingerprint mismatch")
    image=nib.load(str(args.image))
    volume=np.asarray(image.dataobj,dtype=np.float32)
    affine=np.asarray(image.affine,dtype=np.float64)
    if volume.ndim!=3 or min(volume.shape)<8 or not np.isfinite(volume).all():
        raise ValueError("unsupported real MRI shape/values")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-12:
        raise ValueError("invalid real MRI world affine")
    corner=tuple((int(n)-8)//2 for n in volume.shape)
    crop_slice=tuple(slice(v,v+8) for v in corner)
    patch=np.ascontiguousarray(volume[crop_slice],dtype=np.float32)
    shift=np.eye(4)
    shift[:3,3]=corner
    local_affine=affine@shift
    source=FeatureLattice((8,8,8),np.ones(3),np.zeros(3),np.ones(3))
    target=FeatureLattice((3,3,3),np.full(3,2.),np.full(3,.3),np.ones(3))
    plan=synthesize_lattice_repair(source,target)
    out={}
    tensors={}
    torch.set_num_threads(1)
    for dtype in (torch.float32,torch.float64):
        x=torch.from_numpy(patch.copy())[None,None].to(dtype=dtype)
        with torch.no_grad():
            prediction=LatticeRepair3d(plan)(x)
        witness=certify_executed_interpolation(
            x,source,target,plan,output=prediction,max_elements=27)
        key=str(dtype)
        out[key]=asdict(witness)
        tensors[key]=prediction[0,0].detach().cpu().numpy().astype(np.float64)
    difference=np.abs(tensors["torch.float32"]-tensors["torch.float64"])
    assert out["torch.float32"]["audited_elements"]==27
    assert out["torch.float64"]["audited_elements"]==27
    report={
        "source_image":"NiBabel genuine structural MRI: someone_anatomy",
        "source_repo":SOURCE_REPO,
        "source_commit":SOURCE_COMMIT,
        "source_git_blob":blob,
        "source_sha256":sha256(data).hexdigest(),
        "source_shape":list(volume.shape),
        "source_patient_affine":affine.tolist(),
        "crop_offset_voxels":list(corner),
        "crop_patient_affine":local_affine.tolist(),
        "crop_shape":[8,8,8],
        "sampling_target_shape":[3,3,3],
        "sampling_target_start_in_crop_voxels":[.3,.3,.3],
        "sampling_target_jump_in_crop_voxels":[2.,2.,2.],
        "compared_operands":"same stored MRI intensities, CPU grid_sample FP32 versus FP64",
        "float32_float64_max_output_discrepancy":float(difference.max()),
        "float32_float64_mean_output_discrepancy":float(difference.mean()),
        "witness":out,
        "externally_justified_hessian_bound":False,
        "claim_scope":"actual patient image input and stored-source-value numerical execution witness ONLY",
        "not_claimed":["expert segmentation","pretrained model inference",
                       "clinical accuracy","ideal patient coordinate rounding",
                       "independent replication","external adoption"]
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"real_mri_numerical_replay.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.3),constrained_layout=True)
    source_slice=volume[:,:,corner[2]+4].T
    vmin,vmax=np.percentile(source_slice,[1,99])
    axes[0].imshow(source_slice,cmap="gray",origin="lower",
                   vmin=vmin,vmax=vmax)
    axes[0].add_patch(Rectangle((corner[0],corner[1]),8,8,
                edgecolor="#E45C55",linewidth=1.5,fill=False))
    axes[0].set_title("(a) Acquired brain MRI + frozen crop",fontsize=9.5)
    axes[1].imshow(tensors["torch.float32"][:,:,1].T,
                   cmap="gray",origin="lower",interpolation="nearest")
    axes[1].set_title("(b) Actual FP32 trilinear output",fontsize=9.5)
    axes[2].imshow(difference[:,:,1].T,cmap="magma",
                   origin="lower",interpolation="nearest")
    axes[2].set_title("(c) |FP32 - FP64|, same MRI values",fontsize=9.5)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(
        "ClinGeom execution evidence | real MRI, exact-rational audit",
        fontsize=12,weight="semibold")
    fig.savefig(args.out/"real_mri_fp32_fp64_execution.png",dpi=270,
                bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(json.dumps({
       "shape":report["source_shape"],
       "fp32_numeric_bound":out["torch.float32"]["max_exact_barycentric_residual"],
       "fp64_numeric_bound":out["torch.float64"]["max_exact_barycentric_residual"],
       "fp32_vs_fp64_max":report["float32_float64_max_output_discrepancy"],
    },sort_keys=True))

if __name__=="__main__":
    main()
