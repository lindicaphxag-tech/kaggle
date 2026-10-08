"""Proof-carrying ACTUAL CPU 3D pre-mix execution, not just its chart.

A posteriori, bounded-volume witness for a single real PyTorch CPU execution:
  stored source float values
    -> EXACT rational interpolation at STORED lattice declarations
    -> EXACT rational 1x1 convolution with STORED model weights and bias
    -> comparison with the ACTUAL PyTorch align-then-mix output.

This checks numerical discrepancy relative to this declared-float computation.
It is neither a forward-error theorem for future data nor a clinical claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import math

import numpy as np
import torch
import torch.nn as nn

from .clingeom import FeatureLattice
from .lattice_bundle import LatticeBundle
from .pre_mix_compiler import physically_align_then_mix
from .executed_interpolation_witness import (
    _exact_barycentric,
    _exact_declared_chart_indices,
    _fingerprint,
    _round_up_exact,
)


@dataclass(frozen=True)
class ExecutedPreMixWitness:
    source_tensor_sha256: str
    output_tensor_sha256: str
    model_parameters_sha256: str
    compiler_provenance_sha256: str
    audited_output_elements: int
    exact_declared_float_fusion_residual: float
    maximum_audited_execution_error: float | None
    within_audited_execution_budget: bool | None
    proof_scope: str


def _hash_model_parameters(mixer: nn.Conv3d) -> str:
    h = sha256(b"executed-premix-fp-stored-model-v1")
    for parameter in (mixer.weight, mixer.bias):
        if parameter is None:
            h.update(b"NONE")
        else:
            data = parameter.detach().cpu().contiguous().numpy()
            h.update(repr((data.shape, data.dtype.str)).encode("ascii"))
            h.update(data.tobytes())
    return h.hexdigest()


def certify_executed_premix(
    x: torch.Tensor,
    bundle: LatticeBundle,
    target: FeatureLattice,
    mixer: nn.Module,
    *,
    output: torch.Tensor | None = None,
    maximum_audited_execution_error: float | None = None,
    max_exact_operations: int = 16384,
) -> ExecutedPreMixWitness:
    """Audit ONE true CPU PyTorch repair + 1x1 fusion against exact rationals.

    The mathematical reference treats every INPUT FLOAT and WEIGHT FLOAT as
    an exact binary rational, and every lattice jump/start as an exact stored
    binary rational. It captures rounding in float64 index calculations,
    grid_sample, multiply/accumulate and mixer.bias.

    Only authorizes an error budget for THIS EXACT input/model/execution.
    Changes to any input value or weight require a fresh witness.
    """
    if (
        x.device.type != "cpu"
        or x.ndim != 5
        or x.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(x).all())
    ):
        raise ValueError("expected finite CPU FP32/FP64 NCDHW inputs")
    if (
        type(mixer) is not nn.Conv3d
        or mixer.weight.device.type != "cpu"
        or mixer.weight.dtype != x.dtype
        or (mixer.bias is not None and mixer.bias.dtype != x.dtype)
    ):
        raise ValueError("expected CPU 1x1 mixer matching input scalar dtype")
    if (
        isinstance(max_exact_operations, bool)
        or not isinstance(max_exact_operations, int)
        or max_exact_operations <= 0
    ):
        raise ValueError("positive exact operation budget required")
    if maximum_audited_execution_error is not None:
        if (
            not math.isfinite(maximum_audited_execution_error)
            or maximum_audited_execution_error < 0
        ):
            raise ValueError("audited execution budget must be finite and nonnegative")
    num_output = int(x.shape[0]) * int(mixer.out_channels) * math.prod(target.shape)
    estimated_work = num_output * int(x.shape[1])
    if estimated_work > max_exact_operations:
        raise ValueError("exact fusion proof work budget exceeded")

    with torch.no_grad():
        actual = physically_align_then_mix(x, bundle, target, mixer)
    if (
        actual.tensor.device.type != "cpu"
        or not bool(torch.isfinite(actual.tensor).all())
    ):
        raise ValueError("executed fused tensor contains nonfinite values")
    if output is not None and (
        output.device.type != "cpu"
        or output.dtype != actual.tensor.dtype
        or tuple(output.shape) != tuple(actual.tensor.shape)
        or not torch.equal(output, actual.tensor)
    ):
        raise ValueError("supplied fused output fails exact CPU execution replay")
    result = actual.tensor.detach().cpu().numpy()
    source = x.detach().cpu().numpy()
    weights = mixer.weight.detach().cpu().numpy()
    bias = None if mixer.bias is None else mixer.bias.detach().cpu().numpy()

    # The compiler must have admitted one and ONLY one spatial chart for
    # every input channel; multi-support/dense pre-fused fibers are refused.
    channels: list[FeatureLattice | None] = [None] * int(x.shape[1])
    for fiber in bundle.fibers:
        if len(fiber.supports) != 1:
            raise ValueError("cannot audit already mixed multi-support fibers")
        for ch in range(fiber.channel_start, fiber.channel_stop):
            if channels[ch] is not None:
                raise ValueError("overlapping channel spatial provenance")
            channels[ch] = fiber.supports[0]
    if any(lattice is None for lattice in channels):
        raise ValueError("missing channel spatial provenance")

    # Avoid recomputing source chart indexes independently per batch/output
    # channel; however both the *indices* and barycentric sums remain exact.
    positions: list[tuple[tuple[Fraction, Fraction, Fraction], ...]] = []
    coords = list(np.ndindex(*target.shape))
    for lattice in channels:
        assert lattice is not None
        positions.append(tuple(
            _exact_declared_chart_indices(lattice, target, k)
            for k in coords
        ))

    weight_fractions = [
        [Fraction.from_float(float(weights[o, ch, 0, 0, 0]))
         for ch in range(int(x.shape[1]))]
        for o in range(mixer.out_channels)
    ]
    bias_fractions = [
        Fraction(0) if bias is None else Fraction.from_float(float(bias[o]))
        for o in range(mixer.out_channels)
    ]

    max_difference = Fraction(0)
    for b in range(int(x.shape[0])):
        for n, (i, j, k) in enumerate(coords):
            aligned_exact = [
                _exact_barycentric(source, positions[ch][n], b, ch)
                for ch in range(int(x.shape[1]))
            ]
            for o in range(mixer.out_channels):
                exact = bias_fractions[o] + sum(
                    (w * value for w, value in
                     zip(weight_fractions[o], aligned_exact)),
                    start=Fraction(0),
                )
                executed = Fraction.from_float(float(result[b, o, i, j, k]))
                max_difference = max(max_difference, abs(exact - executed))

    upper = _round_up_exact(max_difference)
    acceptable = (
        None if maximum_audited_execution_error is None
        else bool(upper <= maximum_audited_execution_error)
    )
    return ExecutedPreMixWitness(
        source_tensor_sha256=_fingerprint(x),
        output_tensor_sha256=_fingerprint(actual.tensor),
        model_parameters_sha256=_hash_model_parameters(mixer),
        compiler_provenance_sha256=actual.conv_weight_provenance_sha256,
        audited_output_elements=num_output,
        exact_declared_float_fusion_residual=upper,
        maximum_audited_execution_error=maximum_audited_execution_error,
        within_audited_execution_budget=acceptable,
        proof_scope=(
            "ONE frozen CPU PyTorch execution; exactly declared binary-float "
            "source lattice, input values and mixer parameters; full "
            "declared-grid trilinear+bias+dense-fusion arithmetic reference. "
            "No proof of source-acquisition coordinates, latent continuous "
            "field semantics, unmeasured hardware or future inputs. "
            "This witness attests observed numerical discrepancy only."
        ),
    )
