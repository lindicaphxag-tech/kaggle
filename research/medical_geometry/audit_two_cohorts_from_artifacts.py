"""Independent recomputation from two archived ClinGeom raw CI output JSONs.

No MRI volumes are downloaded; the previous CI produced anonymized patient ROI
scores and original checkpoint checksums. Fail if even one source artifact,
patient ID, model identity or printed patient-macro is inconsistent.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from json import loads,dumps
from math import isfinite
from pathlib import Path
from statistics import mean

HASHES=(
    "3e9c5fff2928543cb3ce04d4c951327bb1447ed626a45625fcf085cc18feba81",
    "22622d33cd0c9af8b028004fe234e07df9e79180be5080bbdce35737c89e9c1e",
)
ARMS=("native_axial","reliability_gated_fusion","standard_probability_ensemble")
FROZEN=("dataset_revision","source_train_checkpoint_git_blob",
        "real_original_pretrained_parameter_digest",
        "reliability_gate_frozen_thresholds","roi_edge",
        "roi_offsets_axis0","image_normalization")


def audit_artifacts(paths: tuple[Path,Path]) -> dict:
    documents=[]
    for i,path in enumerate(paths):
        source=path.read_bytes()
        if sha256(source).hexdigest()!=HASHES[i]:
            raise ValueError("archived outcome checksum mismatch")
        d=loads(source)
        if (d["patient_count"]!=6 or len(d["case_level"])!=6 or
                d["reserved_cohort_selection_offset"]!=(2 if i==0 else 8)):
            raise ValueError("original six-case cohort contract changed")
        if (not d["model_loaded_once_and_all_weights_frozen"] or
                not d["patient_level_unit_not_crop_level_pseudoreplication"] or
                not d["only_crop_pilot_not_whole_volume"] or
                d["training_or_hyperparameter_search_on_holdout"]):
            raise ValueError("model/ROI freeze not confirmed")
        if d["roi_edge"]!=64 or d["roi_offsets_axis0"]!=[0,-24,24]:
            raise ValueError("original ROI design altered")
        for arm in ARMS:
            if d["patient_macro_primary"][arm]["n_independent_patients"]!=6:
                raise ValueError("patient-level denominator changed")
            reproduced=mean(float(case["patient_unit_summary"][arm]
                                  ["mean_fixed_roi_dice"])
                            for case in d["case_level"])
            quoted=float(d["patient_macro_primary"][arm]
                         ["subject_macro_mean_fixed_roi_dice"])
            if abs(reproduced-quoted)>1e-12:
                raise ValueError("original published patient mean does not match cases")
        documents.append(d)
    if any(documents[0][name]!=documents[1][name] for name in FROZEN):
        raise ValueError("cohort model/data/method identity mismatch")
    subjects=set()
    deltas=[]; results={name:[] for name in ARMS}
    paired_hd=[]
    for d in documents:
        for case in d["case_level"]:
            unique=case["image_sha256"]
            if unique in subjects:
                raise ValueError("same MRI image appears twice across cohorts")
            subjects.add(unique)
            if len(case["roi_metrics"])!=3:
                raise ValueError("crop-level duplication or lost original ROI")
            scores=case["patient_unit_summary"]
            for arm in ARMS:
                v=float(scores[arm]["mean_fixed_roi_dice"])
                if not isfinite(v) or not 0<=v<=1:
                    raise ValueError("invalid patient macro Dice")
                results[arm].append(v)
            deltas.append(results["reliability_gated_fusion"][-1]
                          -results["native_axial"][-1])
            paired_hd.append(
                float(scores["reliability_gated_fusion"]
                      ["mean_defined_fixed_roi_hd95_mm"])
                -float(scores["native_axial"]
                       ["mean_defined_fixed_roi_hd95_mm"]))
    if len(subjects)!=12:
        raise ValueError("wrong unique patient count")
    return {
        "identity":"original GitHub Actions two frozen patient-source cohorts",
        "patient_count":len(subjects),
        "correlated_roi_count":len(subjects)*3,
        "patient_mean_dice":{arm:mean(vals) for arm,vals in results.items()},
        "gate_minus_native_paired_patient_dice_mean":mean(deltas),
        "gate_minus_native_paired_patient_hd95_mm_mean":mean(paired_hd),
        "patient_gate_better":sum(d>0 for d in deltas),
        "patient_gate_worse":sum(d<0 for d in deltas),
        "source_data_revision":documents[0]["dataset_revision"],
        "source_checkpoint":documents[0]["source_train_checkpoint_git_blob"],
        "pretrained_parameter_digest":
            documents[0]["real_original_pretrained_parameter_digest"],
        "result_sha256":HASHES,
        "disclosure":"12 source subjects; same site/dataset and checkpoint; fixed ROI, no full-volume clinical efficacy",
    }


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("original",type=Path)
    a.add_argument("reserved_next",type=Path)
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    result=audit_artifacts((x.original,x.reserved_next))
    payload=dumps(result,sort_keys=True,indent=2)+"\n"
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(payload)
    print(payload)
