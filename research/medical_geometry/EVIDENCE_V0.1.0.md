# ClinGeom v0.1.0 — Reviewer Evidence Pack

Frozen public branch: `release/clingeom-v0.1.0`  
Frozen commit: `016b09d1d88f86ee20da5ddb7dab5fd471c50a27`  
Public core CI: run `37701895349` — **success**

## One-screen result

| Evidence | Frozen result |
| --- | ---: |
| NumPy core | 50 tests passed |
| Pinned MONAI runtime | 3 tests passed |
| Voxel Dice | 1.0 |
| Patient-space centroid error in the same wrong representation | **41.0 mm** |
| Factorized channel-phase separation at 0.7×0.7×5.0 mm spacing | **5.0970579749 mm** |
| Best possible single-affine minimax error lower bound | **2.5485289875 mm** |
| 45° box → tight AABB analytic fidelity | **0.5** |
| 45°→inverse with exact geometry deferred | **1.0 IoU** |
| Same round trip with eager intermediate AABB projection | **0.25 IoU** |
| Max analytic-vs-65³ sampled prompt-certificate error | **0.001226075** (<0.002 frozen gate) |

## T13 — minimal patient-lattice representation

The frozen artifact certifies a stronger result than “one affine is invalid”:

1. one global point-valued patient-space affine is exact **iff** every channel
   fiber has the same affine center map;
2. the number of distinct affine-map equivalence classes is the minimum number
   of patient-lattice semantic types;
3. with fixed channel order, the minimum number of contiguous fibers equals one
   plus the number of adjacent type transitions;
4. if a channel structurally depends on multiple patient-space supports, no
   point-affine channel partition is exact and the semantic representation must
   be lifted.

Public executable witnesses cover homogeneous, two-phase, A/B/A and dense
multi-support tensors.

## Immutable public artifacts

| Artifact | ID | SHA-256 |
| --- | ---: | --- |
| NumPy evidence | `11518215799` | `0c374be281798e689f9e263562b8b20f00db8d5b964954602f6517c3abfd463d` |
| Channel-fiber evidence | `11518585289` | `cd036830cd7c5c1faf692f9634cff30bec9649ff90d986c053b7dac6847e8585` |
| Pinned MONAI witness | `11518640050` | `f03abb657b8ec9da04014a4747f95847e756cb62d4929ae28d876d5a8cf2735b` |

## External-evidence state at freeze

- independent third-party reproduction: **0**;
- external maintainer adoption: **0**;
- prospective ClinGeom hit: **0**;
- clinical validation: **not claimed**.

This negative state is part of the evidence record. Public CI, self-authored
benchmarks and self-merged PRs must not be presented as external validation.

## Reproduce / falsify

Use `REPRODUCE_CLINGEOM.md` and public Issue #9. The reproduction surface is
pinned to `release/clingeom-v0.1.0`, not rolling `main`.
