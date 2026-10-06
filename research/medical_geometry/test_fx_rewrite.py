import numpy as np
import torch
import torch.nn as nn

from research.medical_geometry.fx_geometry import compile_fx_geometry
from research.medical_geometry.fx_rewrite import rewrite_patient_space_merges


class PhaseMismatchCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 1, 1, 1] = 1.0

    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], dim=1)


def linear_patient_field(size=64):
    d, h, w = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    return (d + 2.0 * h + 3.0 * w)[None, None]


def test_original_graph_is_tensor_valid_but_physical_samples_disagree():
    model = PhaseMismatchCat()
    x = linear_patient_field()
    conv = model.conv(x)
    pool = model.pool(x)
    assert conv.shape == pool.shape == (1, 1, 32, 32, 32)
    # Same tensor shape, different physical centers. For this linear field the
    # phase mismatch becomes a constant 3-unit value offset.
    assert torch.allclose(pool - conv, torch.full_like(conv, 3.0), atol=1e-5)

    report = compile_fx_geometry(
        model,
        input_shape=(64, 64, 64),
        input_affine=np.eye(4),
        strict=False,
        require_channel_aligned_cat=True,
    )
    assert len(report.repair_obligations) == 1


def test_auto_rewrite_aligns_branches_before_cat():
    model = PhaseMismatchCat()
    rewritten = rewrite_patient_space_merges(
        model,
        input_shape=(64, 64, 64),
        input_affine=np.eye(4),
        repair_channel_cat=True,
    )
    assert len(rewritten.rewrites) == 1

    obligation = rewritten.source_report.repair_obligations[0]
    assert obligation.bundle.target.shape == (31, 31, 31)
    assert obligation.bundle.left.kind == "exact_crop"
    assert obligation.bundle.right.kind == "resample"

    y = rewritten.graph_module(linear_patient_field())
    assert y.shape == (1, 2, 31, 31, 31)

    # The two channels now evaluate the same physical linear field on the same
    # target patient-space centers.
    error = torch.max(torch.abs(y[:, 0] - y[:, 1])).item()
    assert error < 2e-4


def test_rewrite_modules_are_bound_to_bundle_digest():
    model = PhaseMismatchCat()
    rewritten = rewrite_patient_space_merges(
        model,
        input_shape=(64, 64, 64),
        input_affine=np.eye(4),
        repair_channel_cat=True,
    )
    record = rewritten.rewrites[0]
    obligation = rewritten.source_report.repair_obligations[0]
    assert record.bundle_digest == obligation.bundle.bundle_digest

    left = rewritten.graph_module.get_submodule(record.left_repair_module)
    right = rewritten.graph_module.get_submodule(record.right_repair_module)
    assert left.certificate_digest == obligation.bundle.left.certificate_digest
    assert right.certificate_digest == obligation.bundle.right.certificate_digest
