"""Public before/after negative control for discrete label value preservation."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.torch_repair import LatticeRepair3d


def main() -> None:
    src = FeatureLattice((8,8,8),np.full(3,2.0),np.full(3,0.5),np.ones(3))
    dst = FeatureLattice((7,7,7),np.full(3,2.0),np.full(3,2.0),np.ones(3))
    plan = synthesize_lattice_repair(src,dst,role="label")
    a,b,c=torch.meshgrid(
        torch.arange(8,dtype=torch.int64),
        torch.arange(8,dtype=torch.int64),
        torch.arange(8,dtype=torch.int64),indexing="ij")
    labels = (2**53+1000000*a+1000*b+c)[None,None]
    repair=LatticeRepair3d(plan)

    expected=labels[:,:,1:,1:,1:]
    safe=repair(labels)
    assert safe.dtype == torch.int64
    assert torch.equal(safe,expected)

    # Negative control exactly reproduces the former float32 value roundtrip.
    grid=repair._grid(labels.float())
    naive=F.grid_sample(labels.float(),grid,mode="nearest",
                        padding_mode="zeros",align_corners=True).round().long()
    corrupt=int((naive!=expected).sum().item())
    assert corrupt>0
    result={
        "input_dtype":str(labels.dtype),
        "min_label_id":int(labels.min().item()),
        "max_label_id":int(labels.max().item()),
        "num_target_voxels":int(expected.numel()),
        "naive_float32_corrupted_voxels":corrupt,
        "lossless_discrete_corrupted_voxels":int((safe!=expected).sum().item()),
        "all_original_label_values_preserved":True,
        "head_version":"CI checkout SHA; see workflow run",
        "limitation":"synthetic region identifiers; no clinical dataset or efficacy claim",
    }
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()