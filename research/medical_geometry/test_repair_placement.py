"""Exhaustive independent oracle for minimum-cost pre-fusion repair placement."""
import itertools
import random
import pytest

from research.medical_geometry.repair_placement import (
    AdmittedSite, UnrepairablePath, choose_repair_sites,
)

PROOF = "a" * 64


def brute_force(graph, sources, hazards, admitted):
    keys = sorted(admitted)
    feasible = []
    for bits in itertools.product((False, True), repeat=len(keys)):
        removed = {k for k, bit in zip(keys, bits) if bit}
        reachable = set(sources) - removed
        todo = list(reachable)
        while todo:
            u = todo.pop()
            for v in graph[u]:
                if v not in removed and v not in reachable:
                    reachable.add(v)
                    todo.append(v)
        if not reachable.intersection(hazards):
            feasible.append(sum(admitted[k].cost for k in removed))
    return min(feasible) if feasible else None


def test_shared_premix_site_vs_two_branch_repairs():
    graph = {
        "s0": ("a",), "s1": ("b",),
        "a": ("cat",), "b": ("cat",),
        "cat": ("fusion",), "fusion": (),
    }
    admitted = {
        "a": AdmittedSite(3, PROOF),
        "b": AdmittedSite(3, PROOF),
        "cat": AdmittedSite(5, PROOF),
    }
    result = choose_repair_sites(graph, ("s0", "s1"), ("fusion",), admitted)
    assert result.sites == ("cat",) and result.total_cost == 5
    assert len(result.graph_digest) == 64
    assert result.source_to_fusion_paths_blocked
    admitted["cat"] = AdmittedSite(7, PROOF)
    result = choose_repair_sites(graph, ("s0", "s1"), ("fusion",), admitted)
    assert result.sites == ("a", "b") and result.total_cost == 6


def test_skip_connection_requires_two_interventions():
    graph = {
        "s": ("a", "skip"), "a": ("cat",),
        "skip": ("fusion",), "cat": ("fusion",), "fusion": (),
    }
    admitted = {
        "a": AdmittedSite(1, PROOF),
        "cat": AdmittedSite(2, PROOF),
        "skip": AdmittedSite(1, PROOF),
    }
    result = choose_repair_sites(graph, ("s",), ("fusion",), admitted)
    assert result.sites == ("a", "skip") and result.total_cost == 2


def test_postmix_only_is_refused_with_a_real_path():
    graph = {"s": ("fusion",), "fusion": ("postmix",), "postmix": ()}
    with pytest.raises(UnrepairablePath) as exc:
        choose_repair_sites(
            graph, ("s",), ("fusion",),
            {"postmix": AdmittedSite(1, PROOF)},
        )
    assert exc.value.path == ("s", "fusion")
    with pytest.raises(ValueError, match="fusion"):
        choose_repair_sites(
            graph, ("s",), ("fusion",),
            {"fusion": AdmittedSite(1, PROOF)},
        )


def test_cycles_malformed_provenance_cost_and_unknown_edges_refused():
    with pytest.raises(ValueError, match="cyclic"):
        choose_repair_sites(
            {"a": ("b",), "b": ("a",), "f": ()},
            ("a",), ("f",), {"b": AdmittedSite(1, PROOF)},
        )
    for cost in (True, 0, -1, 2.5):
        with pytest.raises(ValueError):
            choose_repair_sites(
                {"a": ("b",), "b": ("f",), "f": ()},
                ("a",), ("f",), {"b": AdmittedSite(cost, PROOF)},
            )
    with pytest.raises(ValueError, match="unknown"):
        choose_repair_sites(
            {"a": ("missing",), "f": ()}, ("a",), ("f",), {},
        )
    with pytest.raises(ValueError, match="digest"):
        choose_repair_sites(
            {"a": ("b",), "b": ("f",), "f": ()},
            ("a",), ("f",), {"b": AdmittedSite(1, "not-a-proof")},
        )
    with pytest.raises(ValueError, match="node IDs"):
        choose_repair_sites(
            {"a:in": ("f",), "f": ()},
            ("a:in",), ("f",), {"a:in": AdmittedSite(1, PROOF)},
        )


def test_multiple_fusions_and_unreachable_fusion():
    graph = {
        "s": ("a",), "a": ("f1", "f2"),
        "f1": (), "f2": (), "other": (),
    }
    assert choose_repair_sites(
        graph, ("s",), ("f1", "f2"), {"a": AdmittedSite(2, PROOF)}
    ).total_cost == 2
    graph = {"s": (), "f1": (), "f2": ()}
    result = choose_repair_sites(graph, ("s",), ("f1", "f2"), {})
    assert result.sites == () and result.total_cost == 0


def test_220_seeded_dags_vs_exhaustive_optimum():
    rng = random.Random(20261009)
    for _ in range(220):
        vertices = [f"n{i}" for i in range(7)]
        graph = {
            v: tuple(w for w in vertices[i+1:] if rng.random() < 0.35)
            for i, v in enumerate(vertices)
        }
        sites = {
            v: AdmittedSite(rng.randint(1, 8), PROOF)
            for v in vertices[:-1] if rng.random() < 0.8
        }
        gold = brute_force(graph, ("n0",), ("n6",), sites)
        if gold is None:
            with pytest.raises(UnrepairablePath):
                choose_repair_sites(graph, ("n0",), ("n6",), sites)
        else:
            found = choose_repair_sites(graph, ("n0",), ("n6",), sites)
            assert found.total_cost == gold
            assert brute_force(
                graph, ("n0",), ("n6",),
                {k: v for k, v in sites.items() if k in found.sites},
            ) == gold
