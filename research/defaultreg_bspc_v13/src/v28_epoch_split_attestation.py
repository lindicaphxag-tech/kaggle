"""V28: independently validate trial-content grouping and held-record splits.

Consumes only immutable, source-agnostic V27 overlap metadata. It never
sees classifier predictions or EEG samples. If a V26 purged-result artifact
is provided it checks that *every* source pool respected the independently
reconstructed trial-overlap components. This is a retrospective data-integrity
attestation, NOT a target-isolated Z0 or patient-identity verification.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def archive_member(raw_zip: bytes, suffix: str):
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        names = [n for n in z.namelist() if n.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"expected exactly one {suffix}; found {names}")
        b = z.read(names[0])
    return json.loads(b), hashlib.sha256(b).hexdigest()


def reconstruct(report):
    if report.get("status") != "V27_RETROSPECTIVE_EXACT_EEG_EPOCH_IDENTITY_COMPLETE":
        raise ValueError("not an executed V27 exact epoch report")
    ids = set(range(1, 51))
    counts = report["subject_trial_counts"]
    binary = report["subject_binary_class_counts"]
    if (set(map(int, counts)) != ids
        or set(map(int, binary)) != ids
        or any(int(counts[str(i)]) != 40 for i in ids)
        or any(binary[str(i)] != [20, 20] for i in ids)
        or report["n_original_subject_ids"] != 50
        or report["n_total_trial_instances"] != 2000
        or report["epoch_shape_channels_samples"][0] != 29
        or not report["global_EEG_content_access_for_grouping"]):
        raise AssertionError("complete 50-ID EEG input contract violated")
    parent = {i:i for i in ids}
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def join(a,b):
        a,b=root(a),root(b)
        parent[max(a,b)] = min(a,b)
    edge_map = {}
    for e in report["cross_subject_exact_overlap_edges"]:
        a,b=int(e["subject_a"]),int(e["subject_b"])
        if a not in ids or b not in ids or a >= b or (a,b) in edge_map:
            raise AssertionError("invalid or duplicated cross-ID exact-epoch edge")
        shared=int(e["shared_unique_epoch_contents"])
        pairs=int(e["cross_subject_matching_epoch_instance_pairs"])
        if shared < 1 or shared > 40 or pairs < shared or pairs > 1600:
            raise AssertionError("impossible exact-epoch overlap multiplicity")
        if int(e["trials_in_a"]) != 40 or int(e["trials_in_b"]) != 40:
            raise AssertionError("edge denominator drift")
        edge_map[(a,b)] = shared
        join(a,b)
    connected = {}
    for i in sorted(ids):
        connected.setdefault(root(i),[]).append(i)
    groups=sorted(connected.values(),key=lambda v:v[0])
    declared=sorted((sorted(map(int,g)) for g in
       report["exact_epoch_connected_components"]),key=lambda v:v[0])
    if groups != declared or len(groups) != report["n_exact_epoch_connected_groups"]:
        raise AssertionError("reported components disagree with actual epoch edges")
    if (sum(len(g) for g in groups)!=50
        or report["n_unique_exact_epoch_contents"] > 2000
        or report["n_unique_exact_epoch_contents"] < 1):
        raise AssertionError("impossible exact-epoch denominator")
    representatives=[g[0] for g in groups]
    folds=[]
    for target_group in groups:
        target=target_group[0]
        training=[r for r in representatives if r!=target]
        if any((a in target_group and b in training or b in target_group and a in training)
              for a,b in edge_map):
            raise AssertionError("exact epoch crosses a held group boundary")
        folds.append({"target_recording_representative":target,
                      "target_all_linked_ids":target_group,
                      "source_representatives":training,
                      "n_source_unique_groups":len(training),
                      "requires_global_signal_content_audit":True})
    return {
        "status":"PASS_V28_GRAPH_DERIVED_EPOCH_DISJOINT_GROUPS",
        "n_original_ids":50,
        "n_exact_epoch_unique_contents":report["n_unique_exact_epoch_contents"],
        "n_distinct_epoch_safe_recording_groups":len(groups),
        "epoch_components":groups,
        "shared_epoch_edges":[{"subject_a":a,"subject_b":b,"shared_unique_epoch_contents":n}
                              for (a,b),n in sorted(edge_map.items())],
        "recording_representatives":representatives,
        "all_held_group_source_splits":folds,
        "not_physically_target_isolated":True,
        "not_new_independent_generalisation_data":True,
        "cannot_infer_patient_identity":True,
    }


def verify_with_purged(v28, purged_zip):
    metadata, group_sha = archive_member(purged_zip,
                              "FULL_COHORT_EXACT_SIGNAL_GROUPS_AFTER_GLOBAL_AUDIT.json")
    choices, choice_sha = archive_member(purged_zip,
                              "PURGED_RECORD_IDENTITY_SOURCE_POLICIES_BEFORE_TARGET_AUC.json")
    outcomes, outcome_sha = archive_member(purged_zip,
                              "PURGED_LIU2024_RECORD_INDEPENDENT_RETROSPECTIVE_RESULTS.json")
    if (outcomes["global_record_identity_SHA_before_selection"] != group_sha
        or outcomes["source_policy_SHA_before_held_record_auc"] != choice_sha
        or outcomes["status"] != "RETROSPECTIVE_LIU2024_SIGNAL_IDENTITY_PURGED_LOPO_COMPLETE"):
        raise AssertionError("purged source/target chronology not hash-verified")
    if (metadata["exact_record_groups"] != v28["epoch_components"]
        or metadata["representative_ids"] != v28["recording_representatives"]):
        raise AssertionError("full-record groups miss partial exact-epoch links; must re-fit")
    by_fold={int(c["target_recording_representative"]):c for c in choices}
    outcomes_by_target={int(z["representative_recording"]):z for z in outcomes["target_outcomes"]}
    if not (len(by_fold) == len(outcomes_by_target) == len(v28["all_held_group_source_splits"])):
        raise AssertionError("inconsistent held recording denominator")
    if set(by_fold)!=set(v28["recording_representatives"]) or set(outcomes_by_target)!=set(by_fold):
        raise AssertionError("missing a held source group")
    for f in v28["all_held_group_source_splits"]:
        t=f["target_recording_representative"]
        c=by_fold[t]
        if (c["source_representative_subjects"]!=f["source_representatives"]
            or c["n_source_unique_recordings"]!=len(f["source_representatives"])
            or c["n_source_trials"]!=40*len(f["source_representatives"])
            or not c["source_held_record_group_disjoint"]
            or outcomes_by_target[t]["held_trials"]!=40):
            raise AssertionError("source or target fold contradicts epoch-safe grouping")
    return {
        "purged_source_choices_sha256":choice_sha,
        "purged_identity_manifest_sha256":group_sha,
        "purged_outcomes_sha256":outcome_sha,
        "all_source_held_epoch_groups_covered":True,
        "all_held_source_splits_exact_epoch_disjoint":True,
        "no_claim_of_independent_preregistered_replication":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v27-zip",type=Path,required=True)
    p.add_argument("--v26-zip",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r,sha=archive_member(args.v27_zip,"LIU2024_ALL2000_EPOCH_OVERLAP_REPORT.json")
    v=reconstruct(r)
    v["v27_original_report_sha256"]=sha
    if args.v26_zip:
        v["v26_purged_alignment"]=verify_with_purged(v,args.v26_zip.read_bytes())
    else:
        v["v26_purged_alignment"]="PENDING_IMMUTABLE_PURGED_OUTCOME_ARTIFACT"
    body=json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(body,encoding="utf-8")
    print(json.dumps({"status":v["status"],
           "n_epoch_safe_groups":v["n_distinct_epoch_safe_recording_groups"],
           "n_epoch_edges":len(v["shared_epoch_edges"]),
           "v26_cross_check":v["v26_purged_alignment"],
           "report_sha256":hashlib.sha256(body.encode()).hexdigest()},indent=2))
if __name__=="__main__":
    main()
