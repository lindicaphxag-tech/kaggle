"""V26 fixed-data, THREE-estimand sensitivity of exact-record duplication.

A: original published 50-ID LOPO (not recording-independent).
B: merely drop second copied ID from original 50 target rows, WITHOUT
   changing original contaminated source fits (invalid repair, control).
C: exact-record-group-purged 49-representative source RE-FIT + evaluation.

C-A = (B-A) + (C-B) exactly for same representative IDs. Neither
component is a causal model-effect estimate, as training samples,
selection and target denominator are altered simultaneously.

Uses only immutable already-exposed first-run archives; no training or
parameter tuning here. No independent preregistration or new target test.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ORIG_SOURCE = "FIFTY_SOURCE_ONLY_CHOICES_LOCKED_BEFORE_ANY_TARGET_AUC.json"
ORIG_OUT = "FROZEN_AFTER_SOURCE_LOCK_50PATIENT_LOPO_RESULTS.json"
PURGED_SOURCE = "PURGED_RECORD_IDENTITY_SOURCE_POLICIES_BEFORE_TARGET_AUC.json"
PURGED_OUT = "PURGED_LIU2024_RECORD_INDEPENDENT_RETROSPECTIVE_RESULTS.json"
PURGED_GROUP = "FULL_COHORT_EXACT_SIGNAL_GROUPS_AFTER_GLOBAL_AUDIT.json"
METHOD_NAMES = {
    "source_macro": "source_macro",
    "proper": "proper_source_mean_gaussian",
    "historical_class_centred": "historical_class_centred",
    "LW": "LW",
    "OAS": "OAS",
}
ORIGINAL_FROZEN_CHOICES_SHA = (
    "d56f6f7169689ad80715fc94635a86fce19e936bc1f4536fc2f7845468323227"
)


def member(archive, ending):
    files = [x for x in archive.namelist() if x.endswith(ending)]
    if len(files) != 1:
        raise AssertionError(f"expected one {ending}, found {files}")
    return archive.read(files[0])


def parse_pair(original_zip, purged_zip):
    with zipfile.ZipFile(io.BytesIO(Path(original_zip).read_bytes())) as z:
        a_source = member(z, ORIG_SOURCE)
        a_out = member(z, ORIG_OUT)
    with zipfile.ZipFile(io.BytesIO(Path(purged_zip).read_bytes())) as z:
        b_source = member(z, PURGED_SOURCE)
        b_out = member(z, PURGED_OUT)
        b_group = member(z, PURGED_GROUP)
    if hashlib.sha256(a_source).hexdigest() != ORIGINAL_FROZEN_CHOICES_SHA:
        raise AssertionError("original source manifest digest mismatch")
    A = json.loads(a_out)
    B = json.loads(b_out)
    G = json.loads(b_group)
    SA = json.loads(a_source)
    SB = json.loads(b_source)
    if A["source_choices_SHA256_pre_target_evaluation"] != ORIGINAL_FROZEN_CHOICES_SHA:
        raise AssertionError("old 50-person source lock not verifiable")
    if B["source_policy_SHA_before_held_record_auc"] != hashlib.sha256(b_source).hexdigest():
        raise AssertionError("purged source policy lock not verifiable")
    if B["global_record_identity_SHA_before_selection"] != hashlib.sha256(b_group).hexdigest():
        raise AssertionError("purged group identity lock not verifiable")
    if B["status"] != "RETROSPECTIVE_LIU2024_SIGNAL_IDENTITY_PURGED_LOPO_COMPLETE":
        raise AssertionError("purged reanalysis not completed")
    assert G["duplicate_groups"] == [[4, 5]]
    assert G["independent_processed_record_count"] == 49
    assert B["no_strict_Z0_because_record_identity_used_target_signals"]
    assert len(SA) == 50 and len(SB) == 49
    assert len(A["participants"]) == 50 and len(B["target_outcomes"]) == 49
    reps = G["representative_ids"]
    assert set(reps) == set(range(1,51))-{5}
    old_targets = {i["held_patient"]: i for i in A["participants"]}
    new_targets = {i["representative_recording"]: i for i in B["target_outcomes"]}
    assert set(old_targets)==set(range(1,51))
    assert set(new_targets)==set(reps)
    old_source = {i["outer_subject"]:i for i in SA}
    new_source = {i["target_recording_representative"]:i for i in SB}

    decomposition={}
    for display, method in METHOD_NAMES.items():
        old = np.array([old_targets[i]["auc"][method] for i in range(1,51)])
        reweighted = np.array([old_targets[i]["auc"][method] for i in reps])
        properly_purged = np.array([new_targets[i]["auc"][method] for i in reps])
        Amean, Bmean, Cmean = map(float,(old.mean(),reweighted.mean(),properly_purged.mean()))
        denominator = Bmean-Amean
        source_retraining = Cmean-Bmean
        total = Cmean-Amean
        if abs(total-denominator-source_retraining)>1e-12:
            raise AssertionError("three-estimand arithmetic identity failed")
        record_deltas = properly_purged - reweighted
        decomposition[display]={
            "original_50_ID_macro_AUC":Amean,
            "same_models_delete_duplicate_target_only_49_macro_AUC":Bmean,
            "fully_record_purged_retrained_49_macro_AUC":Cmean,
            "target_denominator_reweighting_difference":denominator,
            "refit_and_policy_change_difference_on_same_49_target_IDs":source_retraining,
            "total_difference":total,
            "min_record_refit_delta":float(np.min(record_deltas)),
            "max_record_refit_delta":float(np.max(record_deltas)),
            "n_refit_harmed_more_than_002":int(np.sum(record_deltas<-.020000000001)),
            "n_refit_improved_more_than_002":int(np.sum(record_deltas>.020000000001)),
        }
        if method in ("source_macro","proper_source_mean_gaussian","historical_class_centred"):
            delta_lam=[]
            for i in reps:
                a = old_source[i]["chosen_lambda"][method]
                b = new_source[i]["chosen_lambdas"][method]
                delta_lam.append(float(b-a))
            decomposition[display]["n_source_lambda_decisions_changed"] = sum(
                abs(k)>1e-12 for k in delta_lam
            )
            decomposition[display]["median_absolute_selected_lambda_change"] = float(
                np.median(np.abs(delta_lam))
            )

    return {
        "status": "PASS_V26_THREE_ESTIMANDS_ID_REWEIGHT_VERSUS_RECORD_PURGED_REFIT",
        "original_source_manifest_sha256":ORIGINAL_FROZEN_CHOICES_SHA,
        "purged_source_manifest_sha256":hashlib.sha256(b_source).hexdigest(),
        "purged_record_group_sha256":hashlib.sha256(b_group).hexdigest(),
        "not_independent_pre_data_validation":True,
        "record_identity_preinspection_not_strict_Z0":True,
        "record_reweighting_is_not_a_training_contamination_repair":True,
        "retraining_delta_is_not_a_causal_effect_of_duplicates":True,
        "n_original_IDs":50,
        "n_unique_processed_EEG_records":49,
        "duplicate_group":[4,5],
        "comparisons":decomposition,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-zip",type=Path,required=True)
    parser.add_argument("--purged-zip",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    opts=parser.parse_args()
    d=parse_pair(opts.original_zip,opts.purged_zip)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    text=json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n"
    opts.out.write_text(text)
    print(json.dumps({
        "status":d["status"],
        "n_unique_processed_EEG_records":d["n_unique_processed_EEG_records"],
        "comparisons":{k:{
            "original_50":v["original_50_ID_macro_AUC"],
            "ID_deleted_only":v["same_models_delete_duplicate_target_only_49_macro_AUC"],
            "record_purged_refit":v["fully_record_purged_retrained_49_macro_AUC"],
            "n_source_choices_changed":v.get("n_source_lambda_decisions_changed"),
        } for k,v in d["comparisons"].items()},
        "receipt_sha256":hashlib.sha256(text.encode()).hexdigest(),
    },indent=2))


if __name__=="__main__":
    main()
