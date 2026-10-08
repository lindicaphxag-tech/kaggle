"""Deterministic numeric experiment: real PyTorch execution versus grid storage.

No fabricated medical image, simulated clinical outcome or pretrained score.
The counterexample is an exact linear tensor-valued field: zero Hessian but
nonzero mixed-precision normalized-coordinate and execution deviations.
Outputs a reproducible raw JSON artifact with a patient-mm budget decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.grid_precision_certificate import (
    certify_grid_coordinate_precision,
)


def row(n: int, dtype: torch.dtype, mm_per_index: float, budget_mm: float) -> dict:
    x = torch.arange(n, dtype=dtype).reshape((1, 1, n, 1, 1))
    desired = torch.arange(n - 1, dtype=torch.float64) + 0.5
    normalized = 2.0 * desired / float(n - 1) - 1.0
    g = normalized.to(dtype).reshape(1, n - 1, 1, 1)
    grid = torch.stack((torch.zeros_like(g), torch.zeros_like(g), g), dim=-1)
    actual = F.grid_sample(x, grid, mode="bilinear", align_corners=True).flatten()
    measured = float(torch.max(torch.abs(actual.to(torch.float64) - desired)))
    source = FeatureLattice((n, 1, 1), np.ones(3), np.zeros(3), np.ones(3))
    target = FeatureLattice((n - 1, 1, 1), np.ones(3),
                            np.array([0.5, 0, 0], dtype=np.float64), np.ones(3))
    affine = np.diag([mm_per_index, 1., 1., 1.])
    cert = certify_grid_coordinate_precision(
        source, target, input_voxel_to_patient_mm=affine, input_dtype=dtype
    )
    return {
        "source_axis_cells": n,
        "dtype": str(dtype),
        "patient_mm_per_source_index": mm_per_index,
        "analytical_C2_linear_interpolation_error": 0.0,
        "measured_total_execution_max_abs_value_deviation": measured,
        "certified_coordinate_storage_max_patient_drift_mm":
            cert.max_world_coordinate_displacement_mm,
        "coordinate_storage_authorized_under_budget": (
            cert.max_world_coordinate_displacement_mm <= budget_mm
        ),
        "value_rounding_and_sampling_arithmetic_are_not_certified": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    budgets = [0.001, 0.01]
    out = {
        "experiment": "mixed precision normalized PyTorch grid and patient space",
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "field": "f(z,y,x)=z (exact linear field, Hessian identically zero)",
        "grid_encoding": "normalized align_corners=True, cast to input dtype",
        "claim_boundary": (
            "Derived certificate bounds only STORED normalized coordinate "
            "displacement; measured output also contains numerical kernel "
            "and value arithmetic. Not a clinical experiment."
        ),
        "budget_mm": budgets[0],
        "rows": [
            row(n, dtype, mm, budgets[0])
            for n in (32, 64, 128, 256)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
            for mm in (0.7, 5.0)
        ],
    }
    if not any(x["dtype"] == "torch.bfloat16" and
               not x["coordinate_storage_authorized_under_budget"]
               for x in out["rows"]):
        raise AssertionError("Counterexample vanished: all BF16 grids within strict mm budget")
    if not any(x["dtype"] == "torch.float32" and
               x["coordinate_storage_authorized_under_budget"]
               for x in out["rows"]):
        raise AssertionError("Negative control vanished: no FP32 grid meets budget")
    data = json.dumps(out, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(data)
    print(json.dumps({
        "rows": len(out["rows"]),
        "source_sizes": [32, 64, 128, 256],
        "mm_scales": [0.7, 5.0],
        "dtype_choices": ["FP32", "FP16", "BF16"],
        "sha256": hashlib.sha256(data.encode()).hexdigest(),
        "out": str(args.out)
    }))


if __name__ == "__main__":
    main()
