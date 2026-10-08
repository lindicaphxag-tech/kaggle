"""Conservative FX use-def extraction for pre-fusion repair placement.

This adapter does NOT infer which ops are dangerous or which sites are
physically admissible. Such claims require a separate sound semantics pass.
"""
from __future__ import annotations

from typing import Mapping

import torch.fx

from .repair_placement import AdmittedSite, Placement, choose_repair_sites


def plan_fx_repair_placement(
    model: torch.fx.GraphModule,
    *,
    bad_sources: tuple[str, ...],
    irreversible_fusions: tuple[str, ...],
    admitted: Mapping[str, AdmittedSite],
) -> Placement:
    """Plan on actual FX data dependencies, including skip and keyword paths.

    Names refer to FX node names, not Python module names. Unknown/opaque
    operations remain in the dependency graph, but must never be admitted
    without an external operation-specific proof of repairability.
    """
    if not isinstance(model, torch.fx.GraphModule):
        raise TypeError("expected a traced torch.fx.GraphModule")
    model.graph.lint()
    nodes = list(model.graph.nodes)
    graph = {n.name: tuple(sorted(u.name for u in n.users)) for n in nodes}
    lookup = {n.name: n for n in nodes}
    if len(lookup) != len(nodes):
        raise ValueError("FX node names are not unique")
    if any(
        name not in lookup or lookup[name].op in ("output", "get_attr")
        for name in irreversible_fusions
    ):
        raise ValueError("each declared irreversible fusion must be an FX computation node")
    if any(name not in lookup or lookup[name].op == "output" for name in admitted):
        raise ValueError("repair sites cannot be outputs or missing FX nodes")
    return choose_repair_sites(
        graph, bad_sources, irreversible_fusions, admitted
    )
