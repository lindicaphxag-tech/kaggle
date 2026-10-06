"""Executable ClinGeom lattice repairs for NCDHW tensors."""

from __future__ import annotations

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
        self.kind = plan.kind
        self.target_shape = tuple(int(x) for x in plan.target_shape)
        self.interpolation = plan.interpolation
        self.crop_start = plan.crop_start
        self.crop_stop = plan.crop_stop
        self.certificate_digest = plan.certificate_digest
        self.register_buffer(
            "target_to_source_index",
            torch.as_tensor(plan.target_to_source_index, dtype=torch.float64),
            persistent=True,
        )

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("LatticeRepair3d expects NCDHW tensors")
        d, h, w = self.target_shape
        device = x.device
        calc_dtype = torch.float64

        td, th, tw = torch.meshgrid(
            torch.arange(d, device=device, dtype=calc_dtype),
            torch.arange(h, device=device, dtype=calc_dtype),
            torch.arange(w, device=device, dtype=calc_dtype),
            indexing="ij",
        )
        ones = torch.ones_like(td)
        target = torch.stack((td, th, tw, ones), dim=-1)
        mapping = self.target_to_source_index.to(device=device, dtype=calc_dtype)
        source = torch.einsum("ij,dhwj->dhwi", mapping, target)[..., :3]

        sd, sh, sw = x.shape[-3:]

        def normalize(coord: torch.Tensor, size: int) -> torch.Tensor:
            if size <= 1:
                return torch.zeros_like(coord)
            return 2.0 * coord / float(size - 1) - 1.0

        # grid_sample expects x,y,z order, i.e. W,H,D.
        grid = torch.stack(
            (
                normalize(source[..., 2], sw),
                normalize(source[..., 1], sh),
                normalize(source[..., 0], sd),
            ),
            dim=-1,
        )
        return grid.to(dtype=torch.float32).unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("LatticeRepair3d expects NCDHW tensors")

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
            grid = self._grid(x)
            mode = "bilinear" if self.interpolation == "trilinear" else "nearest"

            original_dtype = x.dtype
            needs_cast = not torch.is_floating_point(x)
            sample = x.to(torch.float32) if needs_cast else x
            out = F.grid_sample(
                sample,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )
            if needs_cast:
                if mode == "nearest":
                    out = out.round().to(original_dtype)
                else:
                    raise RuntimeError("non-floating tensors require nearest interpolation")
            return out

        raise RuntimeError(f"unknown repair kind: {self.kind!r}")
