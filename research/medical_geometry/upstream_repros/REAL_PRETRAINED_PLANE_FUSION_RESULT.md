# Real frozen T1 U-Net: patient-grid fusion (author-operated evidence)

**Status:** source-pinned real-model CI **passed** on PR #68 head
`915093173b576bb00ed71783bc3ab8f586bbca97` (before this
documentation-only commit). **Not independent reproduction. Not expert
clinical validation. Not an externally accepted method contribution.**

## Primary immutable references

- Authenticated full original TensorFlow checkpoint + actual original/patched
  three-plane inference and ensemble computation:
  [Actions 37819051162](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37819051162)
- Real CPU job: `113455041241` (completed: success).
- Four-file artifact:
  [11568930798](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37819051162/artifacts/11568930798)
- Uploaded ZIP SHA-256:
  `0bb67d188557e1572fd4275fb9ec706841e62a34a01ba6971f93ff573efe610d`.
  Artifacts expire, so the source code and this numeric transcribed record
  remain permanent while independently rerunnable GitHub Actions can refresh.
- Upstream brain_mask Git source: `ca789fb26b70e359b61a8d52968813dd772fb4b1`.
  Source MRI Git blob: `81b1e5260ee4a579d8ac490cac113df9c1cb942f`.
  Provided example-mask Git blob: `e0dd089b3544da4352e751a24dc27a828904bad7`.
  Original T1 checkpoint Git blob:
  `6ac54600739585f54d9f612d13264b74965ffd33`.
- Exact three-plane fusion program:
  [brainmask_pretrained_plane_fusion.py](brainmask_pretrained_plane_fusion.py).
  Adversarial tests:
  [test_brainmask_pretrained_plane_fusion.py](test_brainmask_pretrained_plane_fusion.py).

## Original trained inference was actually run

The original unmodified and native repaired exporter were applied to the
**same inference output** for each source-loader plane. Same learned output
against its exactly reoriented self:

| Plane | Original exporter Dice | Native repair Dice |
|---|---:|---:|
| Axial negative control | 1.0000000 | 1.0000000 |
| Coronal | 0.0041172 | 1.0000000 |
| Sagittal | 0.3630826 | 1.0000000 |

The reference in **this** table is the same model prediction with the correct
known spatial-axis mapping, **not an expert segmentation label**.

## Predeclared frozen single-study three-output ensemble

All rows use the same MRI and publicly released T1 model checkpoint,
fixed 64³ center crop, fixed 0.5 threshold, fixed three-plane **unweighted
mean of probabilities**, and upstream-supplied existing **nonexpert mask**.
Only the ensemble's voxel correspondence changes:

| Weak-reference metric | Raw-position merge (wrong) | Declared inverse-plane merge |
|---|---:|---:|
| Dice | 0.0458168892 | 0.1700203499 |
| Physical HD95, mm | 50.9313263 | 40.9023227 |
| Physical centroid distance, mm | 32.0127615 | 28.6229273 |
| Foreground voxels | 6,109 | 22,591 |

Paired Dice delta on this *one source example*:
**+0.1242034608**. Crucially, this is **not** a general trained-model
improvement claim. The raw-position ensemble violates the known orientation
contract; ordinary native inverse orientation solves the same export step.

The full floating-point native patched fusion exactly matches independently
reoriented original exports in the author-run check. All six adversarial
regression tests pass in Python 3.10, and authentic upstream TF inference,
physical-space weak-reference scoring and complete artifact upload passed.

### Negative results matter

For individual plane-versus-provided-mask Dice, coronal improved
`0.068705→0.082254` while sagittal **decreased**
`0.374345→0.363077` after exporting with the correct patient grid.
This rules out simplistic "geometry fix universally increases Dice" claims;
the nonexpert reference is neither ground truth nor a spatial-semantic oracle.

## What this proves, and what it does not

- **Observed:** wrong coordinates can change downstream fusion of genuine
  frozen learned model probabilities; exact declared inverse transform restores
  agreement with a separate native source patch, and changes actual task-level
  *weak-reference* numbers.
- **Not observed:** patient-level generalization, held-out expert segmentation,
  trained internal feature-fusion repair, per-patient safety, external
  maintainer acceptance, or representation superiority over NiBabel/MONAI
  built-in orientation inversion.
- **Research requirement next:** evaluate genuinely channel-dependent *learned*
  feature phases inside a fusion architecture, compare against correct
  MONAI/NiBabel native transforms, and validate on independently annotated
  held-out patients. This simple output-export/fusion example does **not**
  establish that more ambitious claim.

Reproduce without changing weights, axis mapping, threshold, or patient subset:
run `.github/workflows/clingeom-brainmask-pretrained-planes.yml` and inspect
`pretrained_plane_fusion.json` and original job logs. Report contrary results.
