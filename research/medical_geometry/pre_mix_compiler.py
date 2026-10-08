"""Authorized patient-space alignment BEFORE irreversible channel mixing.

Post-fusion transport can be impossible: two spatially shifted fiber signals
can produce identical dense mixed tensors but different intended values
at the same patient coordinate. This compiler places a verified transport
upstream of a 1x1 convolution, instead of fabricating a mixed-feature affine.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from .channel_fiber_runtime import repair_channel_fibers
from .clingeom import FeatureLattice
from .lattice_bundle import LatticeBundle, bundle_from_lattice
from .support_envelope import premix_output_support_envelope


@dataclass(frozen=True)
class CertifiedPreMixResult:
    tensor: torch.Tensor
    output_bundle: LatticeBundle
    source_repair_digests: tuple[str, ...]
    per_output_value_error_bounds: tuple[float, ...] | None
    conv_weight_provenance_sha256: str
    output_receptive_field_envelope: tuple[float, float, float]


def _check_pointwise_dense_mix(mix: nn.Module, source_channels: int) -> nn.Conv3d:
    if type(mix) is not nn.Conv3d:
        raise ValueError("only exact Conv3d dense mixing is admitted")
    if (mix.in_channels != source_channels or mix.groups != 1
            or tuple(mix.kernel_size) != (1, 1, 1)
            or tuple(mix.stride) != (1, 1, 1)
            or tuple(mix.padding) != (0, 0, 0)
            or tuple(mix.dilation) != (1, 1, 1)):
        raise ValueError("expected pointwise dense 1x1 Conv3d without stride/padding")
    if not torch.isfinite(mix.weight.detach()).all():
        raise ValueError("non-finite mixer weights")
    if mix.bias is not None and not torch.isfinite(mix.bias.detach()).all():
        raise ValueError("non-finite mixer bias")
    return mix


def physically_align_then_mix(
    channel_tensor: torch.Tensor,
    source_bundle: LatticeBundle,
    target: FeatureLattice,
    mixer: nn.Module,
    *,
    input_voxel_to_patient_mm: np.ndarray | None = None,
    maximum_fiber_value_error: float | None = None,
    fiber_world_hessian_bounds: Mapping[int, float] | None = None,
    maximum_output_value_error: float | None = None,
    patient_frame_uid: str | None = None,
) -> CertifiedPreMixResult:
    """Move admitted dense fusion AFTER certified physical sampling.

    When a numeric budget is requested it first proves per-fiber interpolation
    error bounds, then transports them to output channels using |W|.
    This proves conditional error from sampling, not equivalence to the
    original phase-misaligned network or downstream model performance.
    """
    # Reject malformed physical charts even if a caller did not request a
    # numerical error bound: the physical identity is still part of the
    # semantic provenance of any returned "certified" output.
    if input_voxel_to_patient_mm is not None:
        patient_affine = np.asarray(input_voxel_to_patient_mm, dtype=np.float64)
        if (patient_affine.shape != (4, 4)
                or not np.all(np.isfinite(patient_affine))
                or not np.allclose(
                    patient_affine[3], [0.0, 0.0, 0.0, 1.0],
                    rtol=0, atol=1e-12,
                )
                or np.linalg.matrix_rank(patient_affine[:3, :3]) != 3):
            raise ValueError("patient affine must be finite, invertible 4x4")
    else:
        patient_affine = None
    if patient_frame_uid is not None and (
        not isinstance(patient_frame_uid, str)
        or not patient_frame_uid.strip()
    ):
        raise ValueError("patient frame UID must be nonblank if supplied")
    mixer = _check_pointwise_dense_mix(mixer, source_bundle.channels)
    if not channel_tensor.is_floating_point():
        raise ValueError("dense continuous-feature mixing requires a floating tensor")
    if maximum_output_value_error is not None:
        if maximum_fiber_value_error is None:
            raise ValueError("output-error authorization requires per-fiber bound")
        if not np.isfinite(maximum_output_value_error) or maximum_output_value_error < 0:
            raise ValueError("invalid maximum output error budget")
    output_support = premix_output_support_envelope(source_bundle, target)
    aligned = repair_channel_fibers(
        channel_tensor, source_bundle, target,
        role="feature",
        input_voxel_to_patient_mm=input_voxel_to_patient_mm,
        maximum_value_error=maximum_fiber_value_error,
        uniform_world_hessian_bound_by_channel_start=fiber_world_hessian_bounds,
    )
    per_output: tuple[float, ...] | None = None
    if aligned.conditional_value_error_bounds is not None:
        input_bounds = np.empty((source_bundle.channels,), dtype=np.float64)
        for fiber, bound in zip(source_bundle.fibers, aligned.conditional_value_error_bounds):
            input_bounds[fiber.channel_start:fiber.channel_stop] = bound
        abs_weights = np.abs(
            mixer.weight.detach().cpu().numpy().astype(np.float64, copy=True)
        ).reshape(mixer.out_channels, mixer.in_channels)
        estimates = abs_weights @ input_bounds
        per_output = tuple(float(x) for x in estimates)
        if (maximum_output_value_error is not None
                and any(x > maximum_output_value_error for x in per_output)):
            raise ValueError("post-fusion value-error bound exceeds output budget")
    if maximum_output_value_error is not None and per_output is None:
        raise ValueError("output-error budget needs trusted curvature evidence")
    output = mixer(aligned.tensor)
    if tuple(output.shape[-3:]) != tuple(target.shape):
        raise RuntimeError("pointwise mixer violated target shape")
    # The original digest committed to the mixer/voxel geometry but omitted
    # the patient-mm basis and input proof assumptions. Equal voxel tensors
    # with different physical spacing could receive the SAME digest even
    # though their authorized interpolation-value error bounds differed.
    #
    # v3 commits to every declared proof premise. This is a deterministic
    # provenance fingerprint, NOT a signed/trusted attestation of externally
    # supplied input bounds or patient identity.
    digest = sha256(b"clingeom-pre-mix-physical-semantic-provenance-v3")
    premises = {
        "patient_affine_mm": (
            None if patient_affine is None else patient_affine.tolist()
        ),
        "patient_frame_uid": patient_frame_uid,
        "declared_max_fiber_value_error": maximum_fiber_value_error,
        "declared_max_output_value_error": maximum_output_value_error,
        "fiber_world_hessian_bounds": (
            None if fiber_world_hessian_bounds is None
            else sorted(
                (int(key), float(value))
                for key, value in fiber_world_hessian_bounds.items()
            )
        ),
        "computed_per_output_value_error_bounds": per_output,
        "source_shape": tuple(int(n) for n in channel_tensor.shape),
        "target_shape": tuple(int(n) for n in output.shape),
    }
    digest.update(
        json.dumps(
            premises, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    for p in (mixer.weight, mixer.bias):
        if p is None:
            digest.update(b"NONE")
        else:
            a = p.detach().cpu().contiguous().numpy()
            digest.update(str((tuple(a.shape),str(a.dtype))).encode("ascii"))
            digest.update(a.tobytes())
    for s in aligned.plan_certificate_digests:
        digest.update(s.encode("ascii"))
    # The source footprints and derived output envelope are part of semantic
    # provenance: equal centers/weights do not imply equal physical support.
    for fiber in source_bundle.fibers:
        for support in fiber.supports:
            digest.update(np.asarray(
                support.receptive_field, dtype="<f8"
            ).tobytes())
    digest.update(np.asarray(
        output_support.receptive_field, dtype="<f8"
    ).tobytes())
    return CertifiedPreMixResult(
        tensor=output, output_bundle=bundle_from_lattice(mixer.out_channels,output_support),
        source_repair_digests=aligned.plan_certificate_digests,
        per_output_value_error_bounds=per_output,
        conv_weight_provenance_sha256=digest.hexdigest(),
        output_receptive_field_envelope=tuple(float(v) for v in output_support.receptive_field),
    )
