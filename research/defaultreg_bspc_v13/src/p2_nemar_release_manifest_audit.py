#!/usr/bin/env python3
"""Independent distributor's VERSIONED MANIFEST, not source MAT->EDF lineage.
Downloads JSON provenance metadata only, never any EEG signal file or
patient metadata. A recorded sha256 is not independently recomputed bytes.
"""
import argparse,hashlib,json,urllib.request
from pathlib import Path

COMMIT="11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83"
URL=(f"https://raw.githubusercontent.com/nemarDatasets/nm000158/{COMMIT}"
     "/sourcedata/sourcedata_provenance.json")

def audit(j):
    records=j.get("files",[])
    if len(records)!=50 or j.get("n_files")!=50 or j.get("n_subjects_recorded")!=50:
        raise ValueError("Incomplete 50-EDF manifest")
    ids=[int(x["subject"]) for x in records]
    if sorted(ids)!=list(range(1,51)):
        raise ValueError("Unexpected ID manifest")
    if j.get("moabb_version_used")!="1.5.0":
        raise ValueError("Unexpected distribution source version")
    if any(len(x["sha256"])!=64 or int(x["bytes"])<=0 for x in records):
        raise ValueError("Invalid digest record")
    byid={int(x["subject"]):x for x in records}
    pairs=[]
    for i,k in ((4,5),(11,15),(47,49)):
        pairs.append({"released_public_ids":[i,k],
            "different_whole_file_sha256_in_publisher_manifest":
                byid[i]["sha256"]!=byid[k]["sha256"]})
    return {"status":"DISTRIBUTOR_DIGEST_MANIFEST_REVIEW_ONLY",
        "distribution":"NEMAR nm000158 v1.0.3",
        "upstream_retrieved":"2026-08-11",
        "source_loader_version":j["moabb_version_used"],
        "recorded_file_count":len(records),
        "unique_recorded_whole_file_digests":
            len(set(x["sha256"] for x in records)),
        "pairs":pairs,
        "claims_same_underlying_Liu2024_acquisition":True,
        "manifest_hashes_not_independently_recomputed_from_signal_files":True,
        "no_RAW_MAT_to_EDF_transform_reconstructed":True,
        "not_independent_EEG_acquisition":True,
        "whole_EDF_hash_inequality_does_not_refute_digital_channel_alias":True,
        "no_patient_identity_inference":True,
        "no_signal_bytes_downloaded":True}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    z=p.parse_args()
    with urllib.request.urlopen(urllib.request.Request(
            URL,headers={"User-Agent":"EEG-public-provenance-metadata-audit"}),
            timeout=30) as r:
        raw=r.read(128000)
    j=json.loads(raw)
    x=audit(j)
    x["immutable_commit"]=COMMIT
    x["metadata_url"]=URL
    x["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    z.out.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(x,indent=2,sort_keys=True)+"\n"
    z.out.write_text(data)
    print("P2_INDEPENDENT_DISTRIBUTOR_MANIFEST",data)
