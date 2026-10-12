"""Synthetic identifiability counterexample. NO empirical EEG transfer claim.

Two fixed source-learned policy score directions, SAME available
source data, two unseen target worlds that reverse their relative AUC
ordering. Cannot infer source-only target sign without assumptions.
"""
from __future__ import annotations
import json
import numpy as np

def auc(x,y,w):
    score=np.asarray(x,dtype=float)@np.asarray(w,dtype=float)
    y=np.asarray(y,dtype=int)
    d=score[y==1,None]-score[None,y==0]
    return float(np.mean((d>0)+.5*(d==0)))

def main():
    # Same public hypothetical "source" in both worlds.
    source_x=np.array([[.0,.0],[1.,0.],[.0,1.],[1.,1.]])
    source_y=np.array([0,0,1,1])
    w0=np.array([1.,0.]);w1=np.array([0.,1.])
    v=np.array([1.,-1.])
    x=np.vstack([np.tile(v,(4,1)),np.zeros((4,2))])
    y1=np.array([1]*4+[0]*4)
    y2=np.array([0]*4+[1]*4)
    a={k:auc(x,y,w) for k,y,w in [
        ("worldA_policy0",y1,w0),("worldA_policy1",y1,w1),
        ("worldB_policy0",y2,w0),("worldB_policy1",y2,w1)
    ]}
    assert a=={"worldA_policy0":1.,"worldA_policy1":0.,
              "worldB_policy0":0.,"worldB_policy1":1.}
    assert np.array_equal(source_x,source_x.copy())
    return {"same_sources":True,"unseen_target_reversal":True,
            "scope":"synthetic counterexample; no actual EEG target evidence",
            "no_universal_source_only_target_utility_guarantee":True,
            "claim_is_conditional_on_absence_of_source_to_target_assumptions":True,
            "source_observations":len(source_y),"target_conditions":a}

if __name__=="__main__":
    print(json.dumps(main(),indent=2,sort_keys=True))
