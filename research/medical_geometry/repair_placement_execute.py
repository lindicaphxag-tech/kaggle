"""Narrow, executable FX rewrite for verified two-phase pre-fusion repair.

This is intentionally a fail-closed vertical slice, not an arbitrary graph
rewriter. Exactly two phase-slice producers feed torch.cat, which feeds one
frozen dense 1x1 Conv3d, which feeds the graph output. Tensor geometry is
compiled from the *actual* FX operators and verified before any rewrite.

An external uniform world-space Hessian bound is a REQUIRED proof premise for
interpolation. A numeric certificate is conditional on this premise. It does
not prove model performance or account for GPU/float32 rounding.
"""
from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn

from .clingeom import FeatureLattice
from .interpolation_certificate import certify_interpolation_error
from .lattice_bundle import LatticeBundle, bundle_from_lattice, dense_conv_mix_bundle
from .repair_placement import AdmittedSite
from .repair_placement_fx import plan_fx_repair_placement
from .repair_synthesis import synthesize_lattice_repair, verify_repair_plan
from .support_envelope import premix_output_support_envelope
from .torch_fx_bundle import BundleGeometryError, compile_fx_bundle_geometry
from .torch_geometry import UnsupportedGeometryOp
from .torch_repair import LatticeRepair3d


class FixedFiberTransport(nn.Module):
    """Execute validated per-fiber repairs; never invent an affine for postmix data."""

    def __init__(self, source: LatticeBundle, target: FeatureLattice, plans):
        super().__init__()
        self.channels = source.channels
        self.source_shape = tuple(source.shape)
        self.target_shape = tuple(target.shape)
        self.ranges = tuple((f.channel_start, f.channel_stop) for f in source.fibers)
        if len(self.ranges) != len(plans):
            raise ValueError("one geometry plan required per fiber")
        self.repairs = nn.ModuleList(
            LatticeRepair3d(plan) for plan in plans
        )

    def forward(self, x):
        if (x.ndim != 5 or x.shape[1] != self.channels
                or tuple(x.shape[2:]) != self.source_shape):
            raise ValueError("source tensor violates certified NCDHW fiber contract")
        y = torch.cat(
            [repair(x[:, a:b]) for repair, (a, b) in zip(self.repairs, self.ranges)],
            dim=1,
        )
        if tuple(y.shape[2:]) != self.target_shape:
            raise RuntimeError("aligned FX fiber output violates target lattice")
        return y


@dataclass(frozen=True)
class ExecutedPlacementRewrite:
    graph_module: fx.GraphModule
    repair_node: str
    mixer_node: str
    selected_sites: tuple[str, ...]
    output_support: FeatureLattice
    conditional_per_output_value_error: tuple[float, ...]
    certificate_digest: str


def rewrite_two_phase_fusion(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    target: FeatureLattice,
    input_patient_affine_mm: np.ndarray,
    patient_frame_uid: str,
    world_hessian_bounds_by_channel_start: dict[int, float],
    maximum_output_value_error: float,
    authorized_site_costs: dict[str, int] | None = None,
) -> ExecutedPlacementRewrite:
    """Compile and execute exactly one pre-mix transport insertion.

    This supports only a two-phase graph with one unbranched cat -> 1x1
    pointwise mixer -> output. A min-cut chooses between a single pre-mix
    repair after cat and TWO independent repairs before cat, using caller-
    supplied positive integer planning costs. The physical plans and error
    obligations are verified before the minimum-cut/FX rewrite. Costs are
    user-supplied proxies, not measured latency or clinical utility.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("rewrite requires a frozen eval-mode model")
    if not isinstance(patient_frame_uid, str) or not patient_frame_uid.strip():
        raise ValueError("an explicit nonblank patient frame identifier is required")
    if not np.isfinite(maximum_output_value_error) or maximum_output_value_error < 0:
        raise ValueError("invalid output value error budget")
    if not isinstance(world_hessian_bounds_by_channel_start, dict):
        raise ValueError("per-fiber uniform curvature premises must be explicit")
    if authorized_site_costs is None:
        authorized_site_costs = {"cat": 1}
    if (not isinstance(authorized_site_costs, dict)
            or not authorized_site_costs
            or set(authorized_site_costs) - {"cat", "left", "right"}
            or any(type(v) is not int or v <= 0 for v in authorized_site_costs.values())):
        raise ValueError("authorized repair sites need positive integer costs")

    try:
        report = compile_fx_bundle_geometry(
            model, input_shape=input_shape, input_channels=1
        )
    except (BundleGeometryError, UnsupportedGeometryOp) as exc:
        raise ValueError(
            "unsupported topology: existing patient-space type checker refused the graph"
        ) from exc
    gm = report.graph_module
    gm.eval()
    nodes = list(gm.graph.nodes)
    # Deliberately allow *only* a topology whose every output dependency
    # is covered. Any other op/branch needs its own semantics rule.
    if (len(nodes) != 6
            or [n.op for n in nodes] != [
                "placeholder", "call_function", "call_function",
                "call_function", "call_module", "output",
            ]):
        raise ValueError("unsupported topology; expected phase0, phase1, cat, 1x1 mix")
    placeholder, left, right, cat, mixer_node, output = nodes
    if (left.target is not operator.getitem
            or right.target is not operator.getitem
            or left.args[0] is not placeholder
            or right.args[0] is not placeholder
            or cat.target is not torch.cat):
        raise ValueError("pre-mix repair requires real spatial phase slices and cat")
    if (cat.args[0] not in ((left, right), [left, right])
            or len(cat.users) != 1
            or mixer_node.args != (cat,)
            or output.args != (mixer_node,)):
        raise ValueError("untracked branch or nontrivial fusion dependency")
    dim = cat.kwargs.get("dim", cat.args[1] if len(cat.args) > 1 else 0)
    if dim != 1:
        raise ValueError("only channel-dimension concatenation is supported")
    mixer = gm.get_submodule(str(mixer_node.target))
    if (type(mixer) is not nn.Conv3d or mixer.in_channels != 2
            or mixer.groups != 1 or tuple(mixer.kernel_size) != (1, 1, 1)
            or tuple(mixer.stride) != (1, 1, 1)
            or tuple(mixer.padding) != (0, 0, 0)
            or tuple(mixer.dilation) != (1, 1, 1)):
        raise ValueError("only exact frozen dense pointwise 1x1 Conv3d admitted")
    if (not torch.isfinite(mixer.weight).all()
            or (mixer.bias is not None and not torch.isfinite(mixer.bias).all())):
        raise ValueError("nonfinite mixer weights")
    source = report.bundles[cat.name]
    if source.channels != 2 or len(source.fibers) != 2:
        raise ValueError("each phase must remain a separate patient-space fiber")
    if len(source.fibers[0].supports) != 1 or len(source.fibers[1].supports) != 1:
        raise ValueError("multi-support fibers have no single value-repair map")
    if (not np.any(
        np.asarray(source.fibers[0].supports[0].start)
        != np.asarray(source.fibers[1].supports[0].start)
    )):
        raise ValueError("expected physically offset phases for this experiment")

    plans = []
    input_bounds = np.zeros(source.channels, dtype=np.float64)
    for fiber in source.fibers:
        origin = fiber.supports[0]
        plan = synthesize_lattice_repair(origin, target, role="feature")
        if plan.kind == "refuse" or not verify_repair_plan(
            origin, target, plan, role="feature"
        ):
            raise ValueError("physical phase transport refused")
        curvature = world_hessian_bounds_by_channel_start.get(fiber.channel_start)
        cert = certify_interpolation_error(
            origin, target, plan,
            input_voxel_to_patient_mm=input_patient_affine_mm,
            maximum_allowed_error=maximum_output_value_error,
            world_hessian_operator_norm_bound=curvature,
            role="feature",
        )
        if not cert.authorized:
            raise ValueError("missing or over-budget physical interpolation proof: "
                             + cert.reason)
        input_bounds[fiber.channel_start:fiber.channel_stop] = (
            cert.max_absolute_error_bound
        )
        plans.append(plan)
    abs_weights = np.abs(
        mixer.weight.detach().cpu().numpy().astype(np.float64).reshape(
            mixer.out_channels, mixer.in_channels
        )
    )
    errors = tuple(float(v) for v in abs_weights @ input_bounds)
    if any(e > maximum_output_value_error for e in errors):
        raise ValueError("total postmix interpolation error exceeds declared budget")
    output_support = premix_output_support_envelope(source, target)
    mixed_bundle = dense_conv_mix_bundle(
        bundle_from_lattice(mixer.in_channels, output_support),
        out_channels=mixer.out_channels, kernel_size=1,
    )
    if (len(mixed_bundle.fibers) != 1
            or len(mixed_bundle.fibers[0].supports) != 1):
        raise RuntimeError("compiled corrected mixer still has mixed spatial supports")

    inputs = {
        "frame_uid": patient_frame_uid,
        "patient_affine_mm": np.asarray(input_patient_affine_mm).tolist(),
        "plans": [p.certificate_digest for p in plans],
        "world_hessian_premises": sorted(world_hessian_bounds_by_channel_start.items()),
        "max_output_error": maximum_output_value_error,
        "derived_error": errors,
        "authorized_site_costs": sorted(authorized_site_costs.items()),
        "mixer_weights": hashlib.sha256(
            mixer.weight.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    site_names = {"cat": cat.name, "left": left.name, "right": right.name}
    admitted = {
        site_names[key]: AdmittedSite(cost=cost, proof_digest=digest)
        for key, cost in authorized_site_costs.items()
    }
    placement = plan_fx_repair_placement(
        gm, bad_sources=(left.name, right.name),
        irreversible_fusions=(mixer_node.name,),
        admitted=admitted,
    )
    if not (placement.sites == (cat.name,)
            or placement.sites == tuple(sorted((left.name, right.name)))):
        raise RuntimeError("unexpected repair cut; refusing partial branch authority")

    # Graph modifications begin only after all candidate proofs and min-cut
    # postconditions succeeded. The original nn.Module remains unmodified.
    module_name = "_clingeom_certified_pre_mix"
    if hasattr(gm, module_name):
        raise ValueError("module name collision: refusing overwrite")
    if placement.sites == (cat.name,):
        gm.add_module(module_name, FixedFiberTransport(source, target, plans))
        with gm.graph.inserting_before(mixer_node):
            aligned = gm.graph.call_module(module_name, args=(cat,))
        mixer_node.args = (aligned,)
        repair_node_name = aligned.name
    else:
        # The repair plans are independently tied to each branch's real
        # patient-space phase; do not move a shared mixed tensor post-fusion.
        for i, (node, plan) in enumerate(((left, plans[0]), (right, plans[1]))):
            name = f"{module_name}_branch_{i}"
            if hasattr(gm, name):
                raise ValueError("repair module name collision")
            gm.add_module(name, LatticeRepair3d(plan))
            with gm.graph.inserting_before(cat):
                repaired = gm.graph.call_module(name, args=(node,))
            node.replace_all_uses_with(repaired)
            # replace_all_uses_with also rewires the new module's own input;
            # restore its original source explicitly.
            repaired.args = (node,)
        args = list(cat.args)
        if args[0] not in ((left, right), [left, right]):
            # Both cat inputs should now reference the two new repairs.
            node_args = args[0]
            if (not isinstance(node_args, (tuple, list))
                    or len(node_args) != 2
                    or any(not isinstance(n, fx.Node) for n in node_args)):
                raise RuntimeError("branch repair broke concatenation")
        repair_node_name = cat.name
    gm.graph.lint()
    gm.recompile()
    if (tuple(output_support.shape) != tuple(target.shape)
            or tuple(mixed_bundle.shape) != tuple(target.shape)):
        raise RuntimeError("post-rewrite patient-space lattice is inconsistent")
    return ExecutedPlacementRewrite(
        gm, repair_node_name, mixer_node.name, placement.sites,
        output_support, errors, digest,
    )
