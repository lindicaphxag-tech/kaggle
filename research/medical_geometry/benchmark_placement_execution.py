"""Reproducible CPU forward comparison for two *executed* pre-mix placements.

This benchmark measures actual wall time, but does NOT calibrate its input
integer site costs from wall time and does not claim broad speed superiority.
It uses constructed phase-split input, NOT actual MRI or clinical labels.
"""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

from .clingeom import FeatureLattice
from .repair_placement_execute import rewrite_two_phase_fusion


class TwoPhaseMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Conv3d(2, 1, 1, bias=False)
        with torch.no_grad():
            self.mix.weight[0, 0, 0, 0, 0] = 1.0
            self.mix.weight[0, 1, 0, 0, 0] = -1.0

    def forward(self, x):
        first = x[:, :, 0::2, 0::2, 0::2]
        second = x[:, :, 1::2, 1::2, 1::2]
        return self.mix(torch.cat((first, second), dim=1))


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def benchmark(*, repetitions: int = 20) -> dict:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 3:
        raise ValueError("repetitions must be integer >=3")
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    model = TwoPhaseMix().eval().requires_grad_(False)
    kwargs = dict(
        input_shape=(8, 8, 8),
        target=FeatureLattice(
            (3, 3, 3), np.full(3, 2.0), np.full(3, 1.0), np.ones(3)
        ),
        input_patient_affine_mm=np.diag([0.7, 0.7, 5.0, 1.0]),
        patient_frame_uid="constructed-example-only",
        # Constructed affine-in-patient-mm field; this is NOT learned curvature.
        world_hessian_bounds_by_channel_start={0: 0., 1: 0.},
        maximum_output_value_error=1e-4,
    )
    phase = (torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1)
             .expand(1, 1, 8, 8, 8).clone() * 0.7)
    schemes = {
        "shared": {"cat": 5, "left": 3, "right": 3},
        "branchwise": {"cat": 7, "left": 3, "right": 3},
    }
    with torch.no_grad():
        original = model(phase)
        outputs = {}
        timing = {}
        configs = {}
        for name, costs in schemes.items():
            rewritten = rewrite_two_phase_fusion(
                model, **kwargs, authorized_site_costs=costs,
            )
            for _ in range(3):
                rewritten.graph_module(phase)
            elapsed = []
            y = None
            for _ in range(repetitions):
                start = time.perf_counter()
                y = rewritten.graph_module(phase)
                elapsed.append((time.perf_counter() - start) * 1000)
            assert y is not None
            outputs[name] = y
            configs[name] = {
                "chosen_sites": list(rewritten.selected_sites),
                "assigned_site_costs": costs,
                "conditional_analytic_value_error": list(
                    rewritten.conditional_per_output_value_error),
                "result_sha256": _sha(y),
            }
            timing[name] = {
                "median_ms": statistics.median(elapsed),
                "min_ms": min(elapsed),
                "max_ms": max(elapsed),
            }
        assert torch.allclose(outputs["shared"], outputs["branchwise"], atol=2e-6)
        assert torch.allclose(outputs["shared"], torch.zeros_like(outputs["shared"]), atol=2e-6)
        assert torch.allclose(original, torch.full_like(original, -0.7), atol=1e-6)
        assert torch.equal(original, model(phase))
    result = {
        "model": "constructed two-phase slice->cat->1x1 Conv3d",
        "clinical_or_pretrained_model": False,
        "learned_hessian_bound_proven": False,
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "seed": 20261009,
        "repetitions": repetitions,
        "original_output_shape": list(original.shape),
        "repaired_output_shape": list(outputs["shared"].shape),
        "original_mean": float(original.mean()),
        "shared_mean": float(outputs["shared"].mean()),
        "schemes": configs,
        "runtime_observations": timing,
        "maximum_paired_difference": float(
            torch.max(torch.abs(outputs["shared"] - outputs["branchwise"]))
        ),
        "proof_scope": (
            "conditional analytic field curvature, exact graph-site coverage, "
            "executed PyTorch forward; no model-independent FP rounding guarantee"
        ),
        "timing_scope": (
            "short noisy CPU microbenchmark on this machine; site costs are "
            "hypothetical integer planning inputs, not measured milliseconds"
        ),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
