# ClinGeom: manually refined independent T1 MRI — real frozen model, strong negative result

**Scientific status:** primary author-executed *3 subject* pilot on 9 prespecified image-centered local 64³ ROIs (3 crops/subject). **Not** full-volume validation, clinical accuracy, proof of an improved segmentation method, 9 independent patients, or external reproduction.

## Live independently downloadable evidence

- Original study source [PR #72](https://github.com/lindicaphxag-tech/kaggle/pull/72).
- **Final data-source frozen + corrected HD95 TensorFlow CI [37826696442](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37826696442): SUCCESS**, with original public HDF5 official T1 3-D U-Net and its original source model; all 6 fixed methods on all 3 patients ran without retraining.
- GitHub execution revision `ecbb93cf8d6b0d1563aea76dbc81c0efcb3ca99c`. An earlier source-frozen execution [37826393592](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37826393592) also passed, before the explicit missing-HD95 correction.
- Public check artifacts: [independent MRI/mask SHA-only manifest](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37826696442/artifacts/11572131128) and [all case/ROI scalar metrics and numerical replay residuals](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37826696442/artifacts/11571099041). Artifacts have finite retention.
- Public upstream frozen [NFBS example patient](https://github.com/MehdiZouitine/pybrook/tree/14cd765ea07cf0ea50c9a7ffa56859401e9333a9/example), with raw image Git blob `f8afc7d0353461a5b94f36d2cd56e3ca265a4fc0` and matching source mask blob `65a0e8958f41be0f1802a0435e8f9f3dd85b2840`.
- Independent source [MedOtter/SynthStrip](https://huggingface.co/datasets/MedOtter/SynthStrip) **dataset revision fixed before scoring** to `d5a92c65c549b2a099bdb5a6e61ebeed6cb6eca3`; take the two lexicographically first unique-subject T1 acquisitions meeting predeclared source-file constraints, without reading masks or predictions. Dataset release *describes* its masks as manually refined; no independent rater audit has been conducted here.
- Original pretrained external HDF5 code checkpoint Git blob `6ac54600739585f54d9f612d13264b74965ffd33`, original `ecalabr/brain_mask` commit `ca789fb26b70e359b61a8d52968813dd772fb4b1`. All model weights frozen. Experimental Chart/Operator code remains connected to original pretrained TensorFlow computation and is individually reassembled and checked against original predictions in all three planes and every crop.

## Frozen ROI and method definitions

- Three independent **source subject IDs**: 1 NFBS + 2 SynthStrip. Patient-independent identifiers were checked; hidden cross-dataset identity overlap was not externally ruled out.
- Each source scan and brain mask must share native NIfTI shape and affine, and are both exactly reoriented into RAS; **no spatial label registration inferred or invented**.
- Fixed 64³ source scan center crop, and two additional crops with `axis0=-24,+24 voxels`, chosen solely from image dimensions, **never** brain masks, outputs, Dice or HD95. These overlapping crops are repeated measures nested within patients, not more patients.
- Original upstream Keras loader and z-score mode, model checkpoint, threshold 0.5 and original per-branch representations unchanged.
- Six methods fixed before looking at held-out labels: original pretraining-view axial only, corrected 3-view probability average, corrected 3-view logit average, correct spatial latent alignment then original pretrained nonlinear block, conjugated original trained spatial kernel+latent chart, and known erroneous array-index mixing.

## Results — patient-level macro averages, 3 fixed ROIs/patient

| Method | Source-mask Dice ↑ | Physical surface HD95 mm ↓ |
|---|---:|---:|
| **Original pretrained native axial (strongest baseline)** | **0.352365** | **30.5200** |
| Ordinary correctly oriented probability ensemble | 0.181565 | 44.3600 |
| Correctly oriented logit ensemble | 0.147658 | 47.7007 |
| Aligned latent features + untransported trained nonlinear spatial block | 0.136442 | 47.8691 |
| Aligned latent + properly transported pretrained nonlinear spatial operators | 0.137296 | 48.0037 |
| **Known geometrically incorrect raw-index feature mixture** | **0.019549** | **72.4446** |

**Mechanism-level negative result:** Correctly transporting physical coordinates and *even* the original learned anisotropic spatial kernels is insufficient for a trained model's three-plane representations to be interchangeable. On this small independent annotated cohort, every tested fusion was **worse than simply using the original model's pretrained acquisition-plane pathway**. The physical incorrect raw-index control is especially destructive, but that does **not** establish the other fused outputs as useful.

This result is stronger evidence for **fusion authorization/refusal** and cross-plane latent-semantic alignment being a meaningful research problem than for the tested fusion being an improved algorithm.

## Mandatory retrospective statistical disclosure

After the first full experimental output was seen, a methods audit found that physical HD95 was null if either surface was empty, and the original patient-level means omitted those ROIs in a potentially **method-specific** manner. This was an **after-the-fact statistical correction**, not preregistered; raw nullable HD95 is kept for every ROI.

The final documented version makes **all methods use the same ROIs**, scoring both-empty as 0 mm and one-empty as the farthest world-space corner-to-corner crop diameter (all sheared/rotated affine axis combinations); otherwise use actual symmetrized physical boundary HD95. This correction changes the known-invalid raw-index arm from ~62.72 mm to **72.44 mm**, and leaves the other five displayed arm averages numerically unchanged in this pilot. It must be reported, not hidden.

No bootstrap confidence interval or significance test is asserted from only 3 patients. No mask-driven crop optimization, post-hoc selected thresholds, pretraining weight updates or exclusion of unfavorable subjects.

## Next prospective **separate** experiment, not same-case "generalization"

The first three subjects are **discovery evidence**. [Stacked PR #73](https://github.com/lindicaphxag-tech/kaggle/pull/73) freezes a real original-trained-model-only *operator-disagreement/view-disagreement reliability gate* with completely explicit unsupervised fixed constants, **before** downloading/evaluating the next six independent SynthStrip T1 cases on the SAME frozen dataset revision and selecting the next six by deterministic metadata offset. That subsequent six-patient cohort must not be tuned after seeing it. Its gate might not outperform the native-axial baseline, and that null is a valid negative result.

**Publication gate:** an independently annotated/disjoint substantially larger cohort; whole-volume inference without cropping distribution artifacts; a truly stronger certified multi-branch training architecture or learned channel-gauge repair; clinically meaningful performance beyond the original pretrained native-view model, conservative conditional intervention with measured coverage and risk; external investigator reproduction. None is yet met here.

**Reproduction:** workflow [.github/workflows/clingeom-external-annotated.yml](../../../.github/workflows/clingeom-external-annotated.yml), downloader [fetch_external_expert_holdout.py](fetch_external_expert_holdout.py), inference [brainmask_external_annotated.py](brainmask_external_annotated.py). All raw medical imaging stays on transient CI runner; only nonimage provenance and per-case scalar metrics are archived.
