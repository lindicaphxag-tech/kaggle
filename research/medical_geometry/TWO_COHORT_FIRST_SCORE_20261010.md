# ClinGeom frozen two-cohort MRI first score — 2026-10-10

The two source-ID-disjoint six-subject experiments used the same authentic pretrained 3D U-Net, three fixed 64-cubed ROI per subject, seven frozen alternative arms and the same public dataset revision.

- Original patient cohort (offset 2): [CI #37827612217](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827612217), raw-result SHA256 `3e9c5fff2928543cb3ce04d4c951327bb1447ed626a45625fcf085cc18feba81`.
- New patient cohort (offset 8): [CI #37975340905](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37975340905), raw-result SHA256 `22622d33cd0c9af8b028004fe234e07df9e79180be5080bbdce35737c89e9c1e`.
- Independent verification of both raw outputs: 12 distinct input-image SHA fingerprints, same loaded original model checksum `f3924786285534591cf66d7d49483fe555df506a79f9a5805f338dc0da139b21`, same three crop positions, same fixed gate (.10/.015/.5), and identical reference data revision.

| Mean patient Dice | First six | Next six | Combined 12 |
|---|---:|---:|---:|
| Canonical-first strongest baseline | 0.186391697 | 0.115957955 | 0.151174826 |
| Proposed reliability-gated blend | 0.186314510 | 0.115891135 | 0.151102822 |
| Equal-probability fusion | 0.088261879 | 0.051992991 | 0.070127435 |
| Gate-minus-canonical | -0.000077187 | -0.000066820 | -0.000072003 |
| Patients with positive Dice gain | 0/6 | 0/6 | 0/12 |

**Outcome:** Negative efficacy against canonical-first in both frozen patient groups. The method avoids much of naive-fusion degradation but does NOT beat the strongest valid branch. Tiny paired differences do not establish medical harm, significance, clinical validity, or generalization. This is 12 patients and 36 correlated ROIs, not 36 independent patients or whole-volume multihospital evidence.

**Decision:** Do not claim positive TMI segmentation performance, retune thresholds on these exposed patients, or scan more offsets until a genuinely different method and independently validated full-volume study exist. Potential contribution is a rigorously scoped patient-space and learned-feature negative-transfer correctness mechanism.
