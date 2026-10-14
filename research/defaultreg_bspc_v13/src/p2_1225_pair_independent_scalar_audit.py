"""Recompute P2 50-ID / 1,225-pair exact-content histogram from immutable CI artifact.

Independent scalar-result arithmetic, NOT independent EDF decoding or a new
source collection. Never infer patient/acquisition identity from digital bytes.
"""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import argparse, hashlib, json

def census(original_zip):
    raw=Path(original_zip).read_bytes()
    with ZipFile(original_zip) as z:
        if z.namelist()!=["p2_nemar_50_edf_1225_content_pairs.json"]:
            raise ValueError("Unexpected immutable artifact files")
        d=json.loads(z.read(z.namelist()[0]))
    rows=d["pairwise_scalar_records"]
    seen={}; histogram=Counter(); channel_histogram=Counter()
    if len(rows)!=1225 or d["n_unordered_comparisons"]!=1225 or d["n_public_ids_rehashed"]!=50:
        raise ValueError("Incomplete source denominator")
    for r in rows:
        a,b=r["public_ids"]
        pair=(a,b)
        if (type(a)!=int or type(b)!=int or not 1<=a<b<=50
             or pair in seen or r["comparable"] is not True):
            raise ValueError("Invalid/duplicated pair")
        full=r["exact_time_varying_EEG_full_channels"]
        blocks=r["aligned_equal_native_EEG_storage_blocks"]
        if (r["labelled_native_EEG_storage_blocks"]!=9280
            or type(full)!=int or not 0<=full<=29 or type(blocks)!=int
            or not 0<=blocks<=9280
            or len(r["exact_time_varying_EEG_channel_labels"])!=full):
            raise ValueError("Invalid per-pair digital evidence")
        seen[pair]=(full,blocks)
        histogram[blocks]+=1
        channel_histogram[full]+=1
    if set(seen)!={(i,j) for i in range(1,51) for j in range(i+1,51)}:
        raise ValueError("Missing combination among 50 IDs")
    if dict(histogram)!={0:1222,171:1,6957:1,9280:1}:
        raise ValueError("Unexpected exact-block distribution")
    if dict(channel_histogram)!={0:1224,29:1}:
        raise ValueError("Unexpected full-channel distribution")
    if seen[(4,5)]!=(29,9280) or seen[(11,15)]!=(0,171) or seen[(47,49)]!=(0,6957):
        raise ValueError("Individual alias/partial-match count drift")
    return dict(sha256_original_github_artifact_zip=hashlib.sha256(raw).hexdigest(),
        count=1225,independent_scalar_count_recomputation=True,
        original_raw_edfs_not_redownloaded=True,
        patients_or_acquisitions_not_identified=True,
        full_channel_histogram=dict(sorted(channel_histogram.items())),
        exact_block_histogram=dict(sorted(histogram.items())),
        no_academic_outside_lab_replication=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("artifact_zip",type=Path)
    p.add_argument("--out",type=Path)
    o=p.parse_args()
    result=census(o.artifact_zip)
    if o.out:o.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))
