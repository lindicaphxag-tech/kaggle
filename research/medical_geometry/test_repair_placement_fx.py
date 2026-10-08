"""Actual CPU FX graph: even/odd patient-space phases -> dense mixer."""
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.repair_placement import (
    AdmittedSite, UnrepairablePath,
)
from research.medical_geometry.repair_placement_fx import (
    plan_fx_repair_placement,
)


class PhaseMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Conv3d(2, 1, 1, bias=False)

    def forward(self, x):
        even = x[:, :, 0::2, ::2, ::2]
        odd = x[:, :, 1::2, ::2, ::2]
        return self.mix(torch.cat((even, odd), dim=1))


def test_real_fx_graph_carries_both_phases_through_cat_and_conv():
    model = torch.fx.symbolic_trace(PhaseMix())
    assert tuple(model(torch.randn(1, 1, 8, 8, 8)).shape) == (
        1, 1, 4, 4, 4
    )
    nodes = {n.name: n for n in model.graph.nodes}
    assert {"getitem", "getitem_1", "cat", "mix"} <= nodes.keys()
    digest = "b" * 64  # test-only placeholder, NOT repair authorization
    admitted = {
        "getitem": AdmittedSite(3, digest),
        "getitem_1": AdmittedSite(3, digest),
        "cat": AdmittedSite(5, digest),
    }
    r = plan_fx_repair_placement(
        model, bad_sources=("getitem", "getitem_1"),
        irreversible_fusions=("mix",), admitted=admitted,
    )
    assert r.sites == ("cat",) and r.total_cost == 5
    admitted["cat"] = AdmittedSite(7, digest)
    r = plan_fx_repair_placement(
        model, bad_sources=("getitem", "getitem_1"),
        irreversible_fusions=("mix",), admitted=admitted,
    )
    assert r.sites == ("getitem", "getitem_1") and r.total_cost == 6


def test_fx_refuses_postmix_only_site_and_missing_premix_proof():
    model = torch.fx.symbolic_trace(PhaseMix())
    with pytest.raises(ValueError, match="repair sites cannot"):
        plan_fx_repair_placement(
            model, bad_sources=("getitem", "getitem_1"),
            irreversible_fusions=("mix",),
            admitted={"output": AdmittedSite(1, "b" * 64)},
        )
    with pytest.raises(UnrepairablePath):
        plan_fx_repair_placement(
            model, bad_sources=("getitem", "getitem_1"),
            irreversible_fusions=("mix",), admitted={},
        )


def test_fx_refuses_noncomputational_hazard_annotation():
    model = torch.fx.symbolic_trace(PhaseMix())
    with pytest.raises(ValueError, match="irreversible fusion"):
        plan_fx_repair_placement(
            model, bad_sources=("getitem",),
            irreversible_fusions=("output",), admitted={},
        )
