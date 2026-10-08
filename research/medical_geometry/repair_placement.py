"""Choose a minimum-cost set of pre-authorized physical repair sites before fusion.

This is a graph planner, not a validator of lattice physics. Each admitted site
must be separately proven to repair *every* offending fiber reaching it.
Unknown ops or input-dependent normalization must not be silently admitted.
The cut certificate proves coverage of declared paths, not medical benefit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping


@dataclass(frozen=True)
class AdmittedSite:
    cost: int
    proof_digest: str  # provenance handle, NOT proof authentication


@dataclass(frozen=True)
class Placement:
    sites: tuple[str, ...]
    total_cost: int
    graph_digest: str
    source_to_fusion_paths_blocked: bool


class UnrepairablePath(ValueError):
    """Bad source reaches an irreversible op without an admitted repair."""

    def __init__(self, path: tuple[str, ...]):
        self.path = path
        super().__init__("no authorized pre-fusion repair site on path: " + " -> ".join(path))


def _unblocked_path(graph: Mapping[str, tuple[str, ...]],
                    sources: tuple[str, ...], hazards: set[str],
                    removed: set[str]) -> tuple[str, ...] | None:
    for source in sources:
        if source in removed:
            continue
        q = deque([(source, (source,))])
        seen = {source}
        while q:
            u, path = q.popleft()
            if u in hazards:
                return path
            for v in sorted(graph[u]):
                if v not in seen and v not in removed:
                    seen.add(v)
                    q.append((v, path + (v,)))
    return None


def choose_repair_sites(
    graph: Mapping[str, tuple[str, ...]],
    bad_sources: tuple[str, ...],
    irreversible_fusions: tuple[str, ...],
    admitted: Mapping[str, AdmittedSite],
) -> Placement:
    """Exact minimum integer-cost pre-fusion vertex cut of a feed-forward DAG.

    Graph nodes are tensor states AFTER an operator. Edges are data-flow
    dependencies. Admitted nodes are already independently authorized
    to repair all incoming offending patient-space channel fibers.
    No such authorization is inferred by this optimizer.
    """
    if not graph:
        raise ValueError("a nonempty graph is required")
    nodes = set(graph)
    if any(not isinstance(u, str) or not u or ":" in u or u.startswith("@") for u in nodes):
        raise ValueError("node IDs must be nonempty strings without reserved delimiters")
    for u, children in graph.items():
        if not isinstance(children, tuple):
            raise ValueError("graph successors must be tuples")
        if len(children) != len(set(children)) or any(v not in nodes for v in children):
            raise ValueError("unknown or duplicated successor")
    if not bad_sources or not irreversible_fusions:
        raise ValueError("bad sources and irreversible fusions must be explicit")
    sources = tuple(sorted(set(bad_sources)))
    hazards = set(irreversible_fusions)
    if not set(sources) <= nodes or not hazards <= nodes or set(sources) & hazards:
        raise ValueError("sources and fusion nodes must exist and be disjoint")
    if not set(admitted) <= nodes or set(admitted) & hazards:
        raise ValueError("fusion and unknown nodes cannot be repair sites")
    for site in admitted.values():
        if (not isinstance(site.cost, int) or isinstance(site.cost, bool)
                or site.cost <= 0 or len(site.proof_digest) != 64
                or any(c not in "0123456789abcdef" for c in site.proof_digest)):
            raise ValueError("site needs positive integer cost and 64-char proof digest")

    indegree = {u: 0 for u in nodes}
    for children in graph.values():
        for v in children:
            indegree[v] += 1
    q = deque(sorted(u for u, d in indegree.items() if d == 0))
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for v in sorted(graph[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
    if visited != len(nodes):
        raise ValueError("cyclic graph has no valid feedforward repair plan")

    # Feasibility witness independently checked on the ORIGINAL graph.
    witness = _unblocked_path(graph, sources, hazards, set(admitted))
    if witness is not None:
        raise UnrepairablePath(witness)

    # Node splitting with capacity cost at authorized repair sites. All other
    # vertices and data-flow edges have forbidden (effectively infinite) cost.
    infinity = sum(v.cost for v in admitted.values()) + 1
    residual: dict[str, dict[str, int]] = {}

    def arc(u: str, v: str, capacity: int) -> None:
        residual.setdefault(u, {})[v] = residual.setdefault(u, {}).get(v, 0) + capacity
        residual.setdefault(v, {}).setdefault(u, 0)

    for u in sorted(nodes):
        arc(u + ":in", u + ":out", admitted[u].cost if u in admitted else infinity)
    for u in sorted(nodes):
        for v in sorted(graph[u]):
            arc(u + ":out", v + ":in", infinity)
    super_source, super_sink = "@SOURCE", "@SINK"
    for u in sources:
        arc(super_source, u + ":in", infinity)
    for u in sorted(hazards):
        arc(u + ":out", super_sink, infinity)

    flow = 0
    while True:
        parents = {super_source: None}
        queue = deque([super_source])
        while queue and super_sink not in parents:
            u = queue.popleft()
            for v, cap in sorted(residual[u].items()):
                if cap > 0 and v not in parents:
                    parents[v] = u
                    queue.append(v)
        if super_sink not in parents:
            break
        path_cap = infinity
        v = super_sink
        while parents[v] is not None:
            u = parents[v]
            path_cap = min(path_cap, residual[u][v])
            v = u
        v = super_sink
        while parents[v] is not None:
            u = parents[v]
            residual[u][v] -= path_cap
            residual[v][u] += path_cap
            v = u
        flow += path_cap
    if flow >= infinity:
        raise RuntimeError("infinite repair cut despite admitted coverage")
    reachable = {super_source}
    q = deque([super_source])
    while q:
        u = q.popleft()
        for v, cap in residual[u].items():
            if cap > 0 and v not in reachable:
                reachable.add(v)
                q.append(v)
    chosen = tuple(sorted(
        u for u in admitted if u + ":in" in reachable and u + ":out" not in reachable
    ))
    if sum(admitted[u].cost for u in chosen) != flow:
        raise RuntimeError("repair cost does not match min-cut dual certificate")
    if _unblocked_path(graph, sources, hazards, set(chosen)) is not None:
        raise RuntimeError("selected repairs do not block all hazard paths")
    provenance = {
        "graph": {u: sorted(graph[u]) for u in sorted(nodes)},
        "bad_sources": sources, "fusions": sorted(hazards),
        "admitted": {u: (admitted[u].cost, admitted[u].proof_digest)
                     for u in sorted(admitted)}, "selected": chosen,
    }
    digest = sha256(json.dumps(
        provenance, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return Placement(chosen, flow, digest, True)
