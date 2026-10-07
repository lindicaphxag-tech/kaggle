"""A repair is accepted only if the rewritten executable graph re-types cleanly."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.fx_geometry import (
    PatientSpaceMergeError, compile_fx_geometry,
)
from research.medical_geometry.fx_rewrite import rewrite_patient_space_merges
from research.medical_geometry.torch_geometry import UnsupportedGeometryOp


class ShiftedFeatureAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x) + self.pool(x)


def _rewrite():
    return rewrite_patient_space_merges(
        ShiftedFeatureAdd(),
        input_shape=(64, 64, 64),
        input_affine=np.diag([0.7, 0.7, 5.0, 1.0]),
    )


def test_rewritten_graph_is_retyped_with_zero_physical_merge_obligations():
    rewritten = _rewrite()
    assert len(rewritten.source_report.repair_obligations) == 1
    assert len(rewritten.rewrites) == 1
    assert rewritten.postcondition_report.repair_obligations == ()
    assert rewritten.postcondition_report.output_lattice.shape == (31, 31, 31)
    with torch.no_grad():
        output = rewritten.graph_module(torch.ones(1, 1, 64, 64, 64))
    assert output.shape == (1, 1, 31, 31, 31)


def test_postcondition_detects_tampered_physical_phase_after_rewrite():
    rewritten = _rewrite()
    repair = rewritten.graph_module.get_submodule(
        rewritten.rewrites[0].right_repair_module
    )
    with torch.no_grad():
        repair.target_to_source_index[0, 3] += 0.25
    with pytest.raises(PatientSpaceMergeError):
        compile_fx_geometry(
            rewritten.graph_module,
            input_shape=(64, 64, 64),
            input_affine=np.diag([0.7, 0.7, 5.0, 1.0]),
            strict=True,
        )


def test_postcondition_rejects_unrepresentable_off_axis_repair_map():
    rewritten = _rewrite()
    repair = rewritten.graph_module.get_submodule(
        rewritten.rewrites[0].right_repair_module
    )
    with torch.no_grad():
        repair.target_to_source_index[0, 1] = 0.01
    with pytest.raises(UnsupportedGeometryOp, match='off-axis'):
        compile_fx_geometry(
            rewritten.graph_module,
            input_shape=(64, 64, 64),
            input_affine=np.eye(4),
            strict=True,
        )
