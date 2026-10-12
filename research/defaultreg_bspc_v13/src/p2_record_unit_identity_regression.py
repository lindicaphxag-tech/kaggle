#!/usr/bin/env python3
"""P2 reproducible fingerprint/split audit logic, synthetic fixtures only.

The published Liu2024 numbers are *asserted as historical reported*
summary identities; this script does NOT independently read EDF signals.
"""
import hashlib,json,struct
from collections import defaultdict
from decimal import Decimal as D

def payload(values,shape,label,dtype="<f8"):
    if dtype!="<f8": raise ValueError("Pinned little-endian float64 only")
    if len(values)!=shape[0]*shape[1]: raise ValueError("invalid array shape")
    if any(not isinstance(z,(int,float)) for z in values):
        raise TypeError("numerical data required")
    import math
    if any(not math.isfinite(z) for z in values):
        raise ValueError("nonfinite samples must be handled explicitly")
    header=json.dumps({"shape":shape,"label":str(label),"dtype":dtype},
                      sort_keys=True,separators=(",",":")).encode()
    return len(header).to_bytes(4,"little")+header+struct.pack("<"+"d"*len(values),*values)

def cross_id_components(records):
    # Each (released_id, content_bytes) must be constructed AFTER
    # controlled epoch extraction, never from AUC vectors.
    first={}; parent={}; confirmed=[]
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for released_id,blob in records:
        parent.setdefault(released_id,released_id)
        key=hashlib.sha256(blob).digest()
        # Sha is a lookup accelerator only; compare BYTES to confirm.
        for old_id,old_blob in first.get(key,[]):
            if old_id!=released_id and old_blob==blob:
                join(released_id,old_id)
                confirmed.append((old_id,released_id))
        first.setdefault(key,[]).append((released_id,blob))
    groups=defaultdict(list)
    for a in parent: groups[find(a)].append(a)
    return sorted(tuple(sorted(g)) for g in groups.values()),confirmed

def maths():
    aa,ab,ba,bb=map(D,["0.588673","0.587347","0.580765","0.582551"])
    total=bb-aa
    interaction=bb-ba-ab+aa
    assert total==D("-0.006122")
    assert interaction==D("0.003112")
    local=D("-0.262500")
    others=D("-0.00078125")
    assert abs((local+48*others)/D(49)-total)<D("0.000001")
    return {"matched_49_AUC_delta":str(total),
            "selector_by_fit_interaction":str(interaction),
            "outlier_group_AUC_change":str(local),
            "other_48_mean_change":str(others)}

def test():
    a=payload([0.,1.,2.,3.],(2,2),"left")
    b=payload([0.,1.,2.,4.],(2,2),"left")
    c=payload([0.,1.,2.,3.],(2,2),"right")
    assert a!=b and a!=c
    grouping,edges=cross_id_components([(4,a),(5,a),(47,b),(49,c)])
    assert grouping==[(4,5),(47,),(49,)]
    assert edges==[(4,5)]
    # Equal released-ID scalar classifier performance is irrelevant
    # to whether actual signal-content fingerprints are equal.
    identical_auc={47:0.6,49:0.6}
    assert identical_auc[47]==identical_auc[49] and b!=c
    return {"group_components":[list(x) for x in grouping],
            "synthetic_edges":[list(x) for x in edges],
            "historical_summary_arithmetic":maths(),
            "EEG_raw_data_reloaded":False,
            "identity_of_humans_inferred":False,
            "new_population_effect_claimed":False}

if __name__=="__main__":
    print(json.dumps(test(),indent=2))
