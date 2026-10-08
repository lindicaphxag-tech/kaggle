"""Automatic torch.fx rewriting of patient-space-invalid feature merges."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Mapping

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn

from .fx_geometry import FXGeometryReport, compile_fx_geometry
from .interpolation_certificate import certify_interpolation_error
from .torch_repair import LatticeRepair3d
from .torch_geometry import UnsupportedGeometryOp


@dataclass(frozen=True)
class RewriteRecord:
    merge_node: str
    left_repair_module: str
    right_repair_module: str
    bundle_digest: str
    value_error_bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class RewrittenGeometryModel:
    graph_module: fx.GraphModule
    source_report: FXGeometryReport
    rewrites: tuple[RewriteRecord, ...]
    postcondition_report: FXGeometryReport


def _unique_module_name(gm: fx.GraphModule, base: str) -> str:
    name = base
    i = 0
    while hasattr(gm, name):
        i += 1
        name = f"{base}_{i}"
    return name


def rewrite_patient_space_merges(
    module: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    input_affine: np.ndarray,
    role: str = "feature",
    repair_channel_cat: bool = False,
    maximum_value_error: float | None = None,
    feature_hessian_bounds_by_node: Mapping[str, float] | None = None,
) -> RewrittenGeometryModel:
    """Insert certified repairs before binary pointwise/aligned merges.

    Pointwise add is eligible by default. Heterogeneous channel concatenation
    preserves channel-fiber identity and is only rewritten when
    repair_channel_cat=True explicitly declares an aligned-fusion contract.

    maximum_value_error enables an opt-in, curvature-conditional error gate.
    Each source feature branch that needs trilinear interpolation must have an
    *externally justified uniform patient-space Hessian bound* in the mapping
    keyed by its FX node name. Unknown bounds fail closed. This mode does NOT
    infer learned-feature curvature or guarantee equivalent model predictions.
    """

    report = compile_fx_geometry(
        module,
        input_shape=input_shape,
        input_affine=input_affine,
        role=role,
        strict=False,
        require_channel_aligned_cat=repair_channel_cat,
    )
    if maximum_value_error is not None and feature_hessian_bounds_by_node is None:
        raise UnsupportedGeometryOp(
            "value-error budget requires explicit per-node Hessian bounds"
        )
    typed_by_name = {typed.node_name: typed.lattice for typed in report.nodes}
    gm = report.graph_module
    graph = gm.graph
    records: list[RewriteRecord] = []

    for index, obligation in enumerate(report.repair_obligations):
        node_by_name = {node.name: node for node in graph.nodes}
        merge = node_by_name.get(obligation.node_name)
        left = node_by_name.get(obligation.left_node)
        right = node_by_name.get(obligation.right_node)
        if merge is None or left is None or right is None:
            raise UnsupportedGeometryOp(
                f"cannot resolve FX nodes for repair obligation {obligation.node_name!r}"
            )

        bundle = obligation.bundle
        if "refuse" in {bundle.left.kind, bundle.right.kind}:
            raise UnsupportedGeometryOp(
                f"repair obligation {merge.name!r} contains a refused component"
            )

        value_bounds: tuple[float, float] | None = None
        if maximum_value_error is not None:
            proofs = []
            for node_name, plan in (
                (obligation.left_node, bundle.left),
                (obligation.right_node, bundle.right),
            ):
                if node_name not in typed_by_name:
                    raise UnsupportedGeometryOp(
                        f"missing source geometry for repair input {node_name!r}"
                    )
                curvature = feature_hessian_bounds_by_node.get(node_name)
                cert = certify_interpolation_error(
                    typed_by_name[node_name],
                    bundle.target,
                    plan,
                    input_voxel_to_patient_mm=input_affine,
                    maximum_allowed_error=maximum_value_error,
                    world_hessian_operator_norm_bound=curvature,
                    role=role,
                )
                if not cert.authorized:
                    raise UnsupportedGeometryOp(
                        f"repair at {merge.name!r} cannot certify {node_name!r} "
                        f"within value-error budget: {cert.reason}"
                    )
                proofs.append(float(cert.max_absolute_error_bound))
            value_bounds = (proofs[0], proofs[1])

        left_name = _unique_module_name(gm, f"_clingeom_{index}_left")
        right_name = _unique_module_name(gm, f"_clingeom_{index}_right")
        gm.add_module(left_name, LatticeRepair3d(bundle.left))
        gm.add_module(right_name, LatticeRepair3d(bundle.right))

        with graph.inserting_before(merge):
            left_fixed = graph.call_module(left_name, args=(left,))
            right_fixed = graph.call_module(right_name, args=(right,))

        if merge.op == "call_function" and merge.target in {operator.add, torch.add}:
            args = list(merge.args)
            if len(args) < 2 or args[0] is not left or args[1] is not right:
                raise UnsupportedGeometryOp(
                    f"add rewrite expects binary direct inputs at {merge.name!r}"
                )
            args[0] = left_fixed
            args[1] = right_fixed
            merge.args = tuple(args)

        elif merge.op == "call_function" and merge.target is torch.cat:
            tensors = merge.args[0]
            if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
                raise UnsupportedGeometryOp(
                    "automatic ClinGeom rewrite currently supports binary torch.cat only"
                )
            if tensors[0] is not left or tensors[1] is not right:
                raise UnsupportedGeometryOp(
                    f"cat rewrite input identity changed at {merge.name!r}"
                )
            args = list(merge.args)
            args[0] = [left_fixed, right_fixed]
            merge.args = tuple(args)

        else:
            raise UnsupportedGeometryOp(
                f"unsupported merge target during rewrite: {merge.target!r}"
            )

        records.append(
            RewriteRecord(
                merge_node=merge.name,
                left_repair_module=left_name,
                right_repair_module=right_name,
                bundle_digest=bundle.bundle_digest,
                value_error_bounds=value_bounds,
            )
        )

    graph.lint()
    gm.recompile()

    # A compiled repair is not a proof merely because its plan was accepted:
    # independently re-type the ACTUAL rewritten graph and require no
    # unresolved patient-space fusion obligations.
    postcondition = compile_fx_geometry(
        gm,
        input_shape=input_shape,
        input_affine=input_affine,
        role=role,
        strict=True,
        require_channel_aligned_cat=repair_channel_cat,
    )
    if postcondition.repair_obligations:
        raise UnsupportedGeometryOp(
            "rewritten FX graph still has patient-space merge obligations"
        )

    return RewrittenGeometryModel(
        graph_module=gm,
        source_report=report,
        rewrites=tuple(records),
        postcondition_report=postcondition,
    )
