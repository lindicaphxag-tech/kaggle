"""V26 original released EDF channelwise source identity investigation.

Compares numerical recordings for subject IDs 4 and 5, and controls
47 and 49. Exports *only aggregate equality counts* and named channels.
No raw EEG, individual predictions, scores, or target labels are emitted.
Raw EDF channel equality does not establish human participant identity.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
import mne


def channel_equality(subject_a, subject_b):
    ds = Liu2024()
    paths = [Path(ds.data_path(s)[0]) for s in (subject_a,subject_b)]
    eegs = [
        mne.io.read_raw_edf(str(p), preload=True, verbose=False, stim_channel="")
        for p in paths
    ]
    if eegs[0].ch_names != eegs[1].ch_names:
        raise AssertionError("EDF channel ordering changed: cannot compare")
    if eegs[0].n_times != eegs[1].n_times:
        raise AssertionError("EDF acquisition duration changed")
    a, b = (raw.get_data().astype(np.float64, copy=False) for raw in eegs)
    eq = [bool(np.array_equal(a[k], b[k])) for k in range(a.shape[0])]
    maxdiff = [float(np.max(np.abs(a[k]-b[k]))) for k in range(a.shape[0])]
    original = [(p.name, int(p.stat().st_size)) for p in paths]
    del a,b,eegs
    return {
        "original_subject_ids":[subject_a,subject_b],
        "original_EDF_basenames":original,
        "n_raw_channels":len(eq),
        "equal_numeric_channels":int(sum(eq)),
        "different_numeric_channels":int(len(eq)-sum(eq)),
        "channelwise_numeric_equal":eq,
        "different_channel_indices":[i for i,v in enumerate(eq) if not v],
        "per_different_channel_max_abs":[maxdiff[i] for i,v in enumerate(eq) if not v],
        "different_EDF_binary_sources":paths[0].name!=paths[1].name,
        "does_not_identify_duplicate_human_participants":True,
    }


def main():
    result = {
        "status":"PUBLIC_LIU2024_EDF_NUMERIC_SOURCE_CHANNEL_IDENTITY",
        "forensic_not_performance_inference":True,
        "pairs":[channel_equality(4,5), channel_equality(47,49)],
    }
    p=Path("/tmp/defaultreg-v26-channel-forensics/LIU2024_EDF_ORIGIN_CHANNEL_IDENTITY.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    p.write_text(body)
    print(json.dumps({
        "status": result["status"],
        "pairs":[{
            "subjects":r["original_subject_ids"],
            "n_channels":r["n_raw_channels"],
            "n_exact_equal":r["equal_numeric_channels"],
            "n_distinct":r["different_numeric_channels"],
            "different_channel_indices":r["different_channel_indices"],
        } for r in result["pairs"]],
        "receipt_sha256":hashlib.sha256(body.encode()).hexdigest(),
    },indent=2))


if __name__=="__main__":
    main()
