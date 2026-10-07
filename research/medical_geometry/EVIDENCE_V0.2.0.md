# ClinGeom v0.2.0 — Frozen Evidence Pack

Frozen branch: `release/clingeom-v0.2.0`  
Frozen commit: `7e570b05071ae65dc1abc636701c5a75ca1424a3`  
Public CI run: `37703000016` — **success**  
NumPy core: **54 tests passed**

## What changed from v0.1.0

### T13 — representation minimality

ClinGeom no longer only says that one affine can be wrong. It certifies the
minimum exact point-affine representation complexity:

- one global affine iff all point-valued channel fibers share one patient-space
  affine center map;
- minimum number of distinct patient-lattice affine types;
- minimum number of channel-contiguous fibers under fixed channel ordering;
- explicit refusal when a channel is already multi-support and therefore cannot
  be represented by any point-affine channel partition.

### T14 — exact collapse radius

For parallel patient lattices

    F_g(i) = L i + t_g,

the best possible single-affine approximation has exact worst-case error

    r* = min_q max_g ||t_g - q||,

the minimum-enclosing-ball radius of the patient-space translations.

Frozen executable results:

| Case | Exact minimax radius | Optimal translation |
| --- | ---: | --- |
| two phases, spacing 0.7×0.7×5.0 mm | **2.5485289875 mm** | (0.35, 0.35, 2.5) mm |
| eight unit-cube phases | **0.8660254038 mm** | (0.5, 0.5, 0.5) |

This gives the compiler an exact authorization rule: collapse to one approximate
affine only when `r* <= declared_tolerance_mm`.

## Prior-art boundary

Polyphase components and shift-equivariant/invariant downsampling are established
prior art (APS, CVPR 2021; LPS, NeurIPS 2022). ClinGeom does **not** claim
discovery of sampling phases or a new anti-aliasing architecture.

The target contribution is the patient-space semantic layer: minimal affine-type
representation, multi-support detection/lifting, and exact millimetre-valued
collapse certificates for existing medical-AI graphs.

## Retained deterministic witnesses

- voxel Dice `1.0` with patient-space centroid error **41.0 mm**;
- two-phase separation **5.0970579749 mm** at 0.7×0.7×5.0 mm spacing;
- 45° tight-AABB prompt fidelity **0.5**;
- exact 45°→inverse deferred representation fidelity **1.0** vs eager AABB
  projection **0.25**;
- analytic-vs-sampled prompt certificate max error **0.001226075**.

## CI artifacts

| Artifact | ID | SHA-256 |
| --- | ---: | --- |
| NumPy evidence | `11518217903` | `607f6442c9d79719d0bee6b0d2975070249192cc3c467046c536fbbe910a3bad` |
| Channel-fiber evidence | `11518252178` | `5c582243e1fef5628779ced494876143673fdc645c7fd89dbcb4a52c77e99cd8` |
| Pinned MONAI witness | `11518606824` | `15c79f6742c89eabed53ae662a4030182af4aa32fe14999d7732303a5f67733c` |

## External state at freeze

- independent third-party reproduction: **0**;
- external maintainer adoption of ClinGeom: **0**;
- ClinGeom-specific prospective hit: **0**;
- clinical validation: **not claimed**;
- repository-wide explicit open-source license: **not yet declared**.

The license state matters for actual reuse. It is deliberately recorded rather
than silently treating public source visibility as permission for adoption.
