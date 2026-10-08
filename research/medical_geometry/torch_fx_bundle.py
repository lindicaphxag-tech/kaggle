"""torch.fx compiler for channel-fibered patient-space tensor semantics.

Unlike the single-lattice checker, this compiler permits channel concatenation
of different spatial phases and records the result as a LatticeBundle. It only
issues a global affine certificate when every channel fiber is geometrically
compatible.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node, symbolic_trace

from .lattice_bundle import (
    LatticeBundle,
    ChannelFiber,
    _deduplicate_supports,
    bundle_from_lattice,
    concat_channel_bundles,
    dense_conv_mix_bundle,
    map_bundle_supports,
    slice_bundle_spatial,
)
from .clingeom import FeatureLattice
from .torch_geometry import UnsupportedGeometryOp, _apply_supported_geometry
from .torch_fx_geometry import _PAD_MODULES, _apply_padding, _apply_upsample


class BundleGeometryError(RuntimeError):
    """Raised when a graph operation cannot preserve certified bundle semantics."""


@dataclass(frozen=True)
class GraphBundleResult:
    graph_module: GraphModule
    bundles: dict[str, LatticeBundle]
    output_bundle: LatticeBundle


def _node_bundle(value: Any, env: dict[Node, LatticeBundle]) -> LatticeBundle:
    if isinstance(value, Node) and value in env:
        return env[value]
    raise UnsupportedGeometryOp("expected a tensor-valued FX node with bundle geometry")


def _full_slice(value: Any) -> bool:
    return isinstance(value, slice) and value.start is None and value.stop is None and value.step is None


def _getitem_bundle(node: Node, env: dict[Node, LatticeBundle]) -> LatticeBundle:
    source = _node_bundle(node.args[0], env)
    index = node.args[1]
    if not isinstance(index, tuple) or len(index) != 5:
        raise UnsupportedGeometryOp(
            "bundle FX v1 certifies only explicit 5-D N,C,D,H,W slicing"
        )
    if not _full_slice(index[0]) or not _full_slice(index[1]):
        raise UnsupportedGeometryOp("batch/channel slicing is not certified by bundle FX v1")

    starts: list[int] = []
    stops: list[int | None] = []
    for item in index[2:]:
        if not isinstance(item, slice):
            raise UnsupportedGeometryOp("spatial integer indexing is not certified by bundle FX v1")
        if item.step not in (None, 1):
            raise UnsupportedGeometryOp("spatial slice steps other than 1 are not certified")
        starts.append(0 if item.start is None else int(item.start))
        stops.append(None if item.stop is None else int(item.stop))
    return slice_bundle_spatial(
        source,
        tuple(starts),
        tuple(stops),
    )


def _per_channel_module(
    module: nn.Module,
    bundle: LatticeBundle,
) -> LatticeBundle:
    if isinstance(module, _PAD_MODULES):
        return map_bundle_supports(
            bundle,
            lambda support: _apply_padding(module, support),
            provenance_suffix=type(module).__name__,
        )
    if isinstance(module, nn.Upsample):
        return map_bundle_supports(
            bundle,
            lambda support: _apply_upsample(module, support),
            provenance_suffix="Upsample",
        )
    return map_bundle_supports(
        bundle,
        lambda support: _apply_supported_geometry(module, support),
        provenance_suffix=type(module).__name__,
    )


def _module_bundle(module: nn.Module, bundle: LatticeBundle) -> LatticeBundle:
    if isinstance(module, nn.Conv3d):
        if module.groups != 1:
            raise UnsupportedGeometryOp(
                "bundle FX v1 certifies Conv3d channel mixing only for groups=1"
            )
        return dense_conv_mix_bundle(
            bundle,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )
    return _per_channel_module(module, bundle)


def _support_key(bundle: LatticeBundle) -> tuple:
    fibers = []
    for fiber in bundle.fibers:
        supports = []
        for support in fiber.supports:
            supports.append(
                (
                    support.shape,
                    tuple(np.round(support.jump, 12)),
                    tuple(np.round(support.start, 12)),
                )
            )
        fibers.append(
            (fiber.channel_start, fiber.channel_stop, tuple(sorted(supports)))
        )
    return (bundle.shape, bundle.channels, tuple(fibers))


def _pointwise_add(a: LatticeBundle, b: LatticeBundle) -> LatticeBundle:
    if _support_key(a) != _support_key(b):
        raise BundleGeometryError(
            "pointwise addition requires matching channel-fiber patient-space supports"
        )
    # Matching sampling CENTERS do not imply matching receptive supports.
    # Addition depends on both tensors. Returning only `a` silently erases
    # the right branch's dependency (e.g. same-center anisotropic AvgPool3d).
    # Keep the componentwise union of source neighborhoods for each fiber.
    merged_fibers = tuple(
        ChannelFiber(
            left.channel_start,
            left.channel_stop,
            _deduplicate_supports((*left.supports, *right.supports)),
            "pointwise sum of both spatial dependency supports",
        )
        for left, right in zip(a.fibers, b.fibers, strict=True)
    )
    return LatticeBundle(a.shape, a.channels, merged_fibers)


def _cat_bundle(node: Node, env: dict[Node, LatticeBundle]) -> LatticeBundle:
    seq = node.args[0]
    if not isinstance(seq, (tuple, list)):
        raise UnsupportedGeometryOp("torch.cat inputs must be an explicit tuple/list")
    dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
    if int(dim) != 1:
        raise UnsupportedGeometryOp(
            "bundle FX v1 certifies torch.cat only along the channel dimension"
        )
    bundles = [_node_bundle(value, env) for value in seq]
    return concat_channel_bundles(bundles, provenance=f"FX cat node={node.name}")


def compile_fx_bundle_geometry(
    module: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    input_channels: int,
) -> GraphBundleResult:
    """Trace a 3-D model and propagate channel-fibered spatial semantics."""

    if input_channels <= 0:
        raise ValueError("input_channels must be positive")
    gm = symbolic_trace(module)
    modules = dict(gm.named_modules())
    env: dict[Node, LatticeBundle] = {}
    input_bundle = bundle_from_lattice(
        input_channels,
        FeatureLattice.input_grid(input_shape),
        provenance="model input",
    )
    output_bundle: LatticeBundle | None = None

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if env:
                raise UnsupportedGeometryOp("bundle FX v1 supports one tensor input")
            env[node] = input_bundle
            continue

        if node.op == "get_attr":
            continue

        if node.op == "call_module":
            env[node] = _module_bundle(
                modules[str(node.target)],
                _node_bundle(node.args[0], env),
            )
            continue

        if node.op == "call_function":
            if node.target is operator.getitem:
                env[node] = _getitem_bundle(node, env)
                continue
            if node.target is torch.cat:
                env[node] = _cat_bundle(node, env)
                continue
            if node.target in (operator.add, torch.add):
                env[node] = _pointwise_add(
                    _node_bundle(node.args[0], env),
                    _node_bundle(node.args[1], env),
                )
                continue
            if node.target in (torch.relu, F.relu):
                env[node] = _node_bundle(node.args[0], env)
                continue
            raise UnsupportedGeometryOp(
                f"bundle FX function {node.target!r} has no certified rule"
            )

        if node.op == "call_method":
            if node.target == "relu":
                env[node] = _node_bundle(node.args[0], env)
                continue
            if node.target == "add":
                env[node] = _pointwise_add(
                    _node_bundle(node.args[0], env),
                    _node_bundle(node.args[1], env),
                )
                continue
            raise UnsupportedGeometryOp(
                f"bundle FX method {node.target!r} has no certified rule"
            )

        if node.op == "output":
            output_bundle = _node_bundle(node.args[0], env)
            continue

        raise UnsupportedGeometryOp(f"unsupported bundle FX node operation: {node.op}")

    if output_bundle is None:
        raise UnsupportedGeometryOp("traced graph did not produce a bundle output")

    return GraphBundleResult(
        graph_module=gm,
        bundles={node.name: bundle for node, bundle in env.items()},
        output_bundle=output_bundle,
    )
