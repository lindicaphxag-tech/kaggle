"""Pinned real pretrained 3-D MRI brain-mask inference + physical export witness.

Uses the exact public ecalabr/brain_mask T1 checkpoint and model source.
No pretrained model is fabricated. No clinical accuracy / performance uplift
is claimed; original example mask is not verified expert ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

UPSTREAM_SHA = "ca789fb26b70e359b61a8d52968813dd772fb4b1"
IMAGE_BLOB = "81b1e5260ee4a579d8ac490cac113df9c1cb942f"
EXISTING_MASK_BLOB = "e0dd089b3544da4352e751a24dc27a828904bad7"
WEIGHTS_BLOB = "6ac54600739585f54d9f612d13264b74965ffd33"
CONFIG_BLOB = "7dcc0cc0fa3b2b19e3b28eac47ab983c502b2022"


def source_identity(path: Path, blob: str) -> dict:
    raw = path.read_bytes()
    git_id = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()
    if git_id != blob:
        raise ValueError(f"Source {path} failed pinned Git blob identity: {git_id}")
    return {"git_blob": git_id, "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw)}


def project_mask_to_ref(mask: np.ndarray, mask_affine: np.ndarray,
                        target_affine: np.ndarray) -> np.ndarray:
    ijk = np.linalg.solve(mask_affine, target_affine)
    k = np.rint(ijk)
    ijk = np.where(np.abs(ijk - k) < 1e-10, k, ijk)
    return affine_transform(
        np.asarray(mask, np.uint8), ijk[:3, :3], offset=ijk[:3, 3],
        output_shape=mask.shape, order=0, prefilter=False,
        cval=0, mode="constant"
    ).astype(bool)


def dice(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    total = int(a.sum()+b.sum())
    return (2*int(np.logical_and(a,b).sum())/total) if total else 1.


def norm_zscore_nonzero(x: np.ndarray) -> np.ndarray:
    """Exact upstream Normalizers.zscore arithmetic, whole input before crop."""
    x = np.asarray(x, np.float64)
    m = x != 0.
    if not m.any():
        raise ValueError("Cannot zscore entirely zero MRI")
    mean = np.mean(x[m])
    std = np.std(x[m]) + 1e-10
    return np.where(m, (x-mean)/std, 0.).astype(np.float32)


def render(img, existing, learned, stale, repaired, filename,
           source_slice):
    z = source_slice
    plane = img[:,:,z].T
    lo,hi = np.percentile(plane[np.isfinite(plane)],[1,99])
    nrm = np.clip((plane-lo)/max(float(hi-lo),1e-10),0,1)
    fig,axes=plt.subplots(1,4,figsize=(15.0,4.3),constrained_layout=True)
    captions=[
        "(a) Real T1 MRI + provided mask",
        "(b) Real frozen 3-D U-Net output",
        "(c) Controlled stale-affine output",
        "(d) Known-transform repaired output"
    ]
    for ax,title,seg in zip(axes,captions,(existing,learned,stale,repaired)):
        ax.imshow(nrm,origin="lower",cmap="gray")
        for mask,color,style in ((existing,"#21C3B9","solid"),
                                 (seg,"#D95748","dashed")):
            sl=mask[:,:,z]
            if np.any(sl) and np.any(~sl):
                ax.contour(sl.T.astype(float),levels=[.5],
                           colors=[color],linestyles=style,linewidths=1.2,
                           origin="lower")
        ax.set_title(title,fontsize=9)
        ax.axis("off")
    fig.suptitle(
       "ClinGeom | actual checkpoint inference with measured spatial export",
       fontsize=12)
    fig.savefig(filename,dpi=210)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    for x in ("image","existing-mask","checkpoint","config","upstream-root","out"):
        p.add_argument("--"+x,required=True,type=Path)
    p.add_argument("--patch-size",type=int,default=64)
    args=p.parse_args()
    output=args.out
    output.mkdir(parents=True,exist_ok=True)
    im_id=source_identity(args.image,IMAGE_BLOB)
    mask_id=source_identity(args.existing_mask,EXISTING_MASK_BLOB)
    weights_id=source_identity(args.checkpoint,WEIGHTS_BLOB)
    config_id=source_identity(args.config,CONFIG_BLOB)

    nii=nib.load(str(args.image))
    img=np.asarray(nii.dataobj,dtype=np.float32)
    ref_nii=nib.load(str(args.existing_mask))
    full_provided=np.asarray(ref_nii.dataobj)>0
    a=np.asarray(nii.affine,dtype=np.float64)
    if img.ndim!=3 or img.shape!=full_provided.shape:
        raise ValueError("Source image and supplied mask not matching 3-D arrays")
    if not np.allclose(a,ref_nii.affine,atol=1e-5,rtol=0):
        raise ValueError("MRI / provided mask affines differ")
    if not np.all(np.isfinite(a)) or np.linalg.det(a[:3,:3])==0:
        raise ValueError("Malformed MRI affine")
    if args.patch_size!=64:
        raise ValueError("Frozen protocol uses exact 64^3 central crop")
    norm=norm_zscore_nonzero(img)
    origin=tuple(int((n-64)//2) for n in img.shape)
    slices=tuple(slice(x,x+64) for x in origin)
    crop=norm[slices]
    provided=full_provided[slices]
    if crop.shape!=(64,64,64):
        raise ValueError("MRI too small for predeclared crop")
    Hcrop=np.eye(4)
    Hcrop[:3,3]=origin
    crop_affine=a@Hcrop

    import tensorflow as tf
    import yaml
    from types import SimpleNamespace
    if str(args.upstream_root) not in sys.path:
        sys.path.insert(0,str(args.upstream_root))
    from model.networks import Networks

    tf.keras.backend.clear_session()
    tf.random.set_seed(17)
    data=yaml.safe_load(args.config.read_text())
    params=SimpleNamespace(**data)
    params.batch_size=1
    params.train_dims=[64,64,64]
    params.policy=tf.keras.mixed_precision.Policy("float32")
    model=Networks(params)()
    # Independent proof that the official HDF5 checkpoint actually populates
    # model parameters; simply calling load_weights is not enough evidence.
    before_weights=[
        np.asarray(w.numpy()).copy() for w in model.weights
    ]
    model.load_weights(str(args.checkpoint))
    after_weights=[np.asarray(w.numpy()) for w in model.weights]
    modified_variables=sum(
        not np.array_equal(x, y) for x, y in zip(before_weights, after_weights)
    )
    if modified_variables <= 0:
        raise AssertionError("No pretrained parameters changed after checkpoint load")
    model_parameters_sha256=hashlib.sha256(
        b"".join(np.ascontiguousarray(w).tobytes() for w in after_weights)
    ).hexdigest()
    data_in=np.ascontiguousarray(crop)[None,:,:,:,None]
    prediction=np.asarray(model(data_in,training=False).numpy())
    if prediction.shape!=(1,64,64,64,1):
        raise AssertionError("Frozen upstream T1 model produced unexpected tensor shape")
    probs=prediction[0,:,:,:,0]
    if not np.isfinite(probs).all():
        raise AssertionError("Real pretrained output contains nonfinite values")
    learned=probs>=0.5
    report={
        "checkpoint_origin":"ecalabr/brain_mask, frozen T1 3D U-Net",
        "upstream_commit":UPSTREAM_SHA,
        "weights":weights_id,
        "config":config_id,
        "real_mri":im_id,
        "provided_existing_mask":mask_id,
        "model_architecture":"Unet3dBneck",
        "tensorflow_version":tf.__version__,
        "cropping_policy":"predeclared center crop, 64^3, no use of labels to locate crop",
        "preprocessing":"upstream nonzero-whole-image zscore, followed by fixed crop",
        "crop_origin_vox":list(origin),
        "crop_affine":crop_affine.tolist(),
        "prediction_probability_range":[float(probs.min()),float(probs.max())],
        "prediction_probability_mean":float(probs.mean()),
        "prediction_probability_sha256":hashlib.sha256(
            np.ascontiguousarray(probs).tobytes()).hexdigest(),
        "prediction_foreground_count":int(learned.sum()),
        "provided_mask_foreground_count":int(provided.sum()),
        "prediction_vs_provided_mask_dice_not_expert_gt":float(dice(learned,provided)),
        "model_weight_load":"completed without skip_mismatch or by_name",
        "checkpoint_changed_variable_count":int(modified_variables),
        "model_loaded_parameters_sha256":model_parameters_sha256,
        "clinical_ground_truth":False,
        "task_performance_improvement":False,
        "independent_external_reproduction":False
    }
    if not learned.any() or learned.all():
        report["repair_comparison"]="REFUSED: learned output empty/full at frozen 0.5 threshold"
    else:
        H=np.eye(4)
        H[0,0]=-1
        H[0,3]=63
        flipped=np.flip(learned,axis=0).copy()
        stale=project_mask_to_ref(flipped,crop_affine,crop_affine)
        repaired=project_mask_to_ref(flipped,crop_affine@H,crop_affine)
        if not np.array_equal(repaired,learned):
            raise AssertionError("Patient-space exact reindex repair failed")
        native=nib.Nifti1Image(learned.astype(np.uint8),crop_affine).as_reoriented(
            np.array([[0.,-1.],[1.,1.],[2.,1.]])
        )
        if not np.array_equal(np.asarray(native.dataobj)>0,flipped):
            raise AssertionError("NiBabel independent voxel orientation oracle failed")
        if not np.allclose(native.affine,crop_affine@H,atol=1e-8,rtol=0):
            raise AssertionError("NiBabel independent affine oracle failed")
        report.update({
           "learned_mask_joint_wrong_reindex_dice":float(dice(flipped,flipped)),
           "learned_mask_stale_patient_grid_dice":float(dice(learned,stale)),
           "learned_mask_repaired_patient_grid_dice":float(dice(learned,repaired)),
           "ni_babel_orientation_agreement":True,
           "repair_comparison":"completed on genuine frozen-checkpoint inference",
        })
        # Display slice selection is only for illustration, never metric selection.
        z=int(np.argmax(learned.sum(axis=(0,1))))
        report["display_slice_rule"]="max frozen-prediction foreground occupancy only, no metric cherry-picking"
        report["display_z_in_crop"]=z
        render(img[slices],provided,learned,stale,repaired,
               output/"fig_actual_pretrained_t1_prediction.png",z)

    (output/"pretrained_t1_geometry.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "foreground":report["prediction_foreground_count"],
        "prediction_minmax":report["prediction_probability_range"],
        "provided_mask_dice_unverified":report["prediction_vs_provided_mask_dice_not_expert_gt"],
        "repair_state":report["repair_comparison"],
        "stale_patient_dice":report.get("learned_mask_stale_patient_grid_dice"),
        "repaired_patient_dice":report.get("learned_mask_repaired_patient_grid_dice")
    },sort_keys=True))


if __name__=="__main__":
    main()
