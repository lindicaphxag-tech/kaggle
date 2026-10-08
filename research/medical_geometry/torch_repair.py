"""Executable ClinGeom lattice repairs for NCDHW tensors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .repair_synthesis import LatticeRepairPlan


class LatticeRepair3d(nn.Module):
    """Execute a previously synthesized and verified 3-D lattice repair.

    The module contains no learned parameters. It only realizes the exact crop or
    role-aware interpolation already authorized by the repair certificate.
    """

    def __init__(self, plan: LatticeRepairPlan):
        super().__init__()
        if plan.kind == "refuse":
            raise ValueError("cannot instantiate a refused lattice repair")
        self.source_shape = tuple(int(x) for x in plan.source_shape)
        if len(self.source_shape) != 3 or any(n <= 0 for n in self.source_shape):
            raise ValueError("invalid source shape in lattice repair plan")
        self.kind = plan.kind
        self.target_shape = tuple(int(x) for x in plan.target_shape)
        if len(self.target_shape) != 3 or any(n <= 0 for n in self.target_shape):
            raise ValueError("invalid target shape in lattice repair plan")
        # An independent execution-time guard: the affine index map may be
        # algebraically valid but still ask grid_sample to extrapolate from
        # beyond the source lattice. Never silently accept padded zeros.
        if self.kind == "resample":
            mapping = np.asarray(plan.target_to_source_index, dtype=np.float64)
            if mapping.shape != (4, 4) or not np.all(np.isfinite(mapping)):
                raise ValueError("invalid source index map in resampling plan")
            hi = np.asarray(self.target_shape, dtype=np.float64) - 1.0
            corners = np.array([
                [i, j, k, 1.0]
                for i in (0.0, hi[0])
                for j in (0.0, hi[1])
                for k in (0.0, hi[2])
            ], dtype=np.float64)
            mapped = (mapping @ corners.T).T[:, :3]
            source_hi = np.asarray(self.source_shape, dtype=np.float64) - 1.0
            if np.any(mapped < -1e-9) or np.any(mapped > source_hi + 1e-9):
                raise ValueError("resampling plan maps target points outside source lattice")
        self.interpolation = plan.interpolation
        self.crop_start = plan.crop_start
        self.crop_stop = plan.crop_stop
        self.certificate_digest = plan.certificate_digest
        self.register_buffer(
            "target_to_source_index",
            torch.as_tensor(plan.target_to_source_index, dtype=torch.float64),
            persistent=True,
        )

    def _source_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        """Compute physical-plan source indices in float64 before sampling."""
        d, h, w = self.target_shape
        td, th, tw = torch.meshgrid(
            torch.arange(d, device=x.device, dtype=torch.float64),
            torch.arange(h, device=x.device, dtype=torch.float64),
            torch.arange(w, device=x.device, dtype=torch.float64),
            indexing="ij",
        )
        target = torch.stack((td, th, tw, torch.ones_like(td)), dim=-1)
        mapping = self.target_to_source_index.to(device=x.device, dtype=torch.float64)
        return torch.einsum("ij,dhwj->dhwi", mapping, target)[..., :3]

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        source = self._source_coordinates(x)
        sd, sh, sw = x.shape[-3:]

        def normalize(coord: torch.Tensor, size: int) -> torch.Tensor:
            if size <= 1:
                return torch.zeros_like(coord)
            return 2.0 * coord / float(size - 1) - 1.0

        grid = torch.stack(
            (normalize(source[..., 2], sw),
             normalize(source[..., 1], sh),
             normalize(source[..., 0], sd)),
            dim=-1,
        )
        # grid_sample requires the grid's scalar dtype to match input dtype.
        # For int/bool labels use the lossless gather path below instead.
        return grid.to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)

    def _nearest_discrete(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest transport for integer/bool labels without any float value cast.

        Source coordinate math uses float64, but *pixel values* stay in their
        original discrete dtype, including int64 labels beyond 2**53.
        """
        coords = self._source_coordinates(x)
        indices = torch.round(coords).to(torch.int64)
        sd, sh, sw = self.source_shape
        # A plan is already domain-checked at construction; clamp only tiny
        # machine-precision excursions at an otherwise valid boundary.
        zi = indices[..., 0].clamp(0, sd - 1)
        yi = indices[..., 1].clamp(0, sh - 1)
        xi = indices[..., 2].clamp(0, sw - 1)
        return x[:, :, zi, yi, xi]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("LatticeRepair3d expects NCDHW tensors")
        if tuple(x.shape[-3:]) != self.source_shape:
            raise ValueError(
                f"tensor source shape {tuple(x.shape[-3:])} does not match "
                f"certified source shape {self.source_shape}"
            )

        if self.kind == "identity":
            return x

        if self.kind == "exact_crop":
            if self.crop_start is None or self.crop_stop is None:
                raise RuntimeError("exact_crop repair is missing crop bounds")
            s = self.crop_start
            e = self.crop_stop
            return x[:, :, s[0]:e[0], s[1]:e[1], s[2]:e[2]]

        if self.kind == "resample":
            if self.interpolation not in {"trilinear", "nearest"}:
                raise RuntimeError(f"unsupported interpolation: {self.interpolation!r}")
            if not torch.is_floating_point(x):
                if self.interpolation != "nearest":
                    raise RuntimeError("discrete tensors require nearest interpolation")
                return self._nearest_discrete(x)
            grid = self._grid(x)
            mode = "bilinear" if self.interpolation == "trilinear" else "nearest"
            return F.grid_sample(
                x,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )

        raise RuntimeError(f"unknown repair kind: {self.kind!r}")
