"""Minimax active provenance queries for bounded EEG electrode identity uncertainty.

A TRUSTED external answer identifies one observed row's physical electrode.
Before querying, all declared within-group permutations are possible. Exact
dynamic programming chooses the minimum worst-case number of identity queries
needed to make a frozen linear model's decision sign invariant across all
remaining mappings. This does not certify the true task label, and assumes the
real identity assignment lies in the finite declared hypothesis set.

This is a research implementation, not a claim that minmax decision trees are new.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
import math
import numpy as np

@dataclass(frozen=True)
class BeliefPlan:
    authorized: bool
    label: int | None
    worst_case_queries: int | None
    next_observed_row: int | None
    hypotheses_left: int

class ProvenanceQueryPlanner:
    def __init__(self, features, weights, intercept, groups, *, tol=1e-10,
                 max_hypotheses=720):
        x=np.asarray(features,dtype=float)
        w=np.asarray(weights,dtype=float)
        if x.ndim!=2 or x.shape!=w.shape or not np.isfinite(x).all() or not np.isfinite(w).all():
            raise ValueError("Nonfinite or mismatched signal and frozen weights")
        if not np.isfinite(intercept) or tol<0:
            raise ValueError("Invalid intercept or tolerance")
        n=len(x)
        covered=set()
        groups=tuple(tuple(int(j) for j in g) for g in groups)
        num=1
        for g in groups:
            if len(g)<2 or len(set(g))!=len(g) or covered.intersection(g) or any(j<0 or j>=n for j in g):
                raise ValueError("Invalid disjoint identity uncertainty groups")
            covered.update(g)
            num*=math.factorial(len(g))
        if num>max_hypotheses:
            raise ValueError("Hypothesis budget exceeded; fail closed")
        self.tol=tol
        self.rows=tuple(sorted(covered))
        mapping_list=[]
        margins=[]
        for assigned in product(*(permutations(g) for g in groups)):
            mapping=list(range(n))
            for g,p in zip(groups,assigned):
                for row,identity in zip(g,p):
                    mapping[row]=identity
            mapping=tuple(mapping)
            mapping_list.append(mapping)
            margins.append(float(intercept+np.sum(x*w[list(mapping)])))
        self.mappings=tuple(mapping_list)
        self.margins=tuple(margins)

    def _certified_label(self, indices):
        lo=min(self.margins[k] for k in indices)
        hi=max(self.margins[k] for k in indices)
        if lo>self.tol:
            return 1
        if hi < -self.tol:
            return 0
        return None

    @lru_cache(maxsize=None)
    def _minimax(self, indices):
        label=self._certified_label(indices)
        if label is not None:
            return 0,None,label
        winner=(math.inf,None,None)
        for row in self.rows:
            possible={}
            for i in indices:
                possible.setdefault(self.mappings[i][row],[]).append(i)
            if len(possible)<2:
                continue
            worst=max(self._minimax(tuple(v))[0] for v in possible.values())
            if not math.isfinite(worst):
                continue
            cost=1+worst
            if cost<winner[0] or (cost==winner[0] and (winner[1] is None or row<winner[1])):
                winner=(cost,row,None)
        return winner

    def propose(self, confirmations=()):
        confirmations=tuple((int(i),int(j)) for i,j in confirmations)
        if len({i for i,_ in confirmations})!=len(confirmations):
            raise ValueError("Duplicate queried electrode")
        if any(i not in self.rows for i,_ in confirmations):
            raise ValueError("Query outside the declared set")
        still=tuple(k for k,mapping in enumerate(self.mappings)
                    if all(mapping[i]==j for i,j in confirmations))
        if not still:
            raise ValueError("Trusted identity contradicts all hypotheses")
        depth,row,label=self._minimax(still)
        if not math.isfinite(depth):
            return BeliefPlan(False,None,None,None,len(still))
        return BeliefPlan(label is not None,label,int(depth),row,len(still))

    def interact(self, trusted_mapping, *, query_budget=None):
        """Test-only oracle, not data-derived inference of physical identity."""
        mapping=tuple(map(int,trusted_mapping))
        if mapping not in self.mappings:
            raise ValueError("True identity outside declared hypothesis set")
        seen=[]
        while True:
            p=self.propose(seen)
            if p.authorized or p.next_observed_row is None:
                return p,tuple(seen)
            if query_budget is not None and len(seen)>=query_budget:
                return p,tuple(seen)
            row=p.next_observed_row
            seen.append((row,mapping[row]))


class DecisionTableQueryPlanner(ProvenanceQueryPlanner):
    """Model-agnostic minimax query policy from a complete frozen-logit table.

    The provided table MUST include every within-group identity assignment,
    using the exact same frozen nonlinear or linear model and preprocessing.
    This does not certify a task label, hardware identity or out-of-set mapping.
    """

    def __init__(self, mappings, decision_margins, groups, *, tol=1e-10, max_hypotheses=720):
        mapping=tuple(tuple(int(j) for j in p) for p in mappings)
        margins=tuple(float(x) for x in decision_margins)
        if not mapping or len(set(mapping))!=len(mapping) or len(mapping)!=len(margins):
            raise ValueError("Missing or duplicate identity hypothesis")
        n=len(mapping[0])
        if any(len(p)!=n or set(p)!=set(range(n)) for p in mapping):
            raise ValueError("Every hypothesis must be a full channel bijection")
        groups=tuple(tuple(int(j) for j in g) for g in groups)
        covered=set()
        for g in groups:
            if len(g)<2 or len(set(g))!=len(g) or covered.intersection(g) or any(j<0 or j>=n for j in g):
                raise ValueError("Invalid disjoint channel uncertainty groups")
            covered.update(g)
        if math.prod(math.factorial(len(g)) for g in groups)>max_hypotheses:
            raise ValueError("Hypothesis budget exceeded; fail closed")
        expected=set()
        for assigns in product(*(permutations(g) for g in groups)):
            p=list(range(n))
            for g,assignment in zip(groups,assigns):
                for row,identity in zip(g,assignment):
                    p[row]=identity
            expected.add(tuple(p))
        if set(mapping)!=expected:
            raise ValueError("Incomplete or out-of-model decision table")
        if not np.isfinite(margins).all() or tol<0:
            raise ValueError("Untrusted or nonfinite model decision outputs")
        self.mappings=mapping
        self.margins=margins
        self.rows=tuple(sorted(covered))
        self.tol=tol
