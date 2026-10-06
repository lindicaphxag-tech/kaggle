"""Deterministic ClinGeom channel-fiber benchmark.

Reproduces DARTS/MONAI-style factorized reduction with torch.fx and quantifies
when a multi-channel feature tensor cannot be represented by one patient-space
affine lattice.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from research.medical_geometry.lattice_bundle import (
    bundle_summary,
    certify_single_lattice,
)
from research.medical_geometry.torch_fx_bundle import compile_fx_bundle_geometry


class FactorizedReduce3D(nn.Module):
    def __init__(self, c_in: int = 8, c_out: int = 16) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.conv_2 = nn.Conv3d(c_in, c_out - c_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm3d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return self.bn(
            torch.cat(
                [self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])],
                dim=1,
            )
        )


class ReduceThenDenseMix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce = FactorizedReduce3D()
        self.mix = nn.Conv3d(16, 24, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.reduce(x))


def make_affine(spacing: tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing)
    return affine


def one_case(spacing: tuple[float, float, float]) -> dict:
    reduced = compile_fx_bundle_geometry(
        FactorizedReduce3D(),
        input_shape=(32, 32, 32),
        input_channels=8,
    ).output_bundle
    mixed = compile_fx_bundle_geometry(
        ReduceThenDenseMix(),
        input_shape=(32, 32, 32),
        input_channels=8,
    ).output_bundle
    affine = make_affine(spacing)
    reduced_cert = certify_single_lattice(reduced, affine)
    mixed_cert = certify_single_lattice(mixed, affine)

    return {
        "input_spacing_mm": list(spacing),
        "factorized_reduce": {
            "single_lattice_valid": reduced_cert.valid,
            "unique_center_lattices": reduced_cert.unique_center_lattices,
            "phase_separation_mm": reduced_cert.max_intra_tensor_displacement_mm,
            "single_affine_minimax_error_lower_bound_mm": (
                reduced_cert.single_affine_minimax_lower_bound_mm
            ),
            "bundle": bundle_summary(reduced),
        },
        "after_dense_3x3x3_channel_mix": {
            "single_lattice_valid": mixed_cert.valid,
            "unique_center_lattices": mixed_cert.unique_center_lattices,
            "support_separation_mm": mixed_cert.max_intra_tensor_displacement_mm,
            "single_affine_minimax_error_lower_bound_mm": (
                mixed_cert.single_affine_minimax_lower_bound_mm
            ),
            "bundle": bundle_summary(mixed),
        },
    }


def main() -> None:
    report = {
        "schema_version": 1,
        "claim": (
            "A channel-concatenated feature tensor can carry multiple patient-space "
            "sampling lattices even though it has one ordinary tensor shape."
        ),
        "cases": [
            one_case((1.0, 1.0, 1.0)),
            one_case((0.7, 0.7, 5.0)),
            one_case((0.8, 0.8, 3.0)),
        ],
        "claim_boundary": (
            "This reproduces an architecture-level phase structure. It is not a claim "
            "that MONAI DiNTS is inaccurate or clinically unsafe."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
