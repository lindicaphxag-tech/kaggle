# P4 whole-numerator Bernstein certificate synthetic p2–p12 stress — EXECUTED
- CI success: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312807723
- Raw 15-case machine-readable profile: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312807723/artifacts/11775293437
- Source: `research/defaultreg_bspc_v13/src/p4_multidimensional_source_certificate_stress.py`
- For each p in {2,3,6,9,12}, three deterministic SPD synthetic eigensystems,
  18+18 binary source-like features, 324 pair orderings per case, 12
  shrinkage subintervals on [.02,.98], 201 sampled lambda values
  whose direct AUC verified interval-envelope containment.
- Mean whole-numerator Bernstein minus V23 Taylor/termwise
  fraction of SOURCE PAIRS CERTIFIED STABLE (percentage points):
  p2: +0.000; p3: +4.321; p6: +13.169; p9: +16.667; p12: +16.152.
  3/3 cases tied in p2; Bernstein ahead in all 12/12 cases p>=3
  for this deliberately selected synthetic generator. These are
  NOT independent EEG recordings and NOT a clinical generalization result.
- Existing REAL EEG Weibo2014 source comparison is MUCH SMALLER:
  +0.146 pp coverage for full [0,1], +0.064 pp between selectors.
  NEVER replace those observed real-data values with the synthetic ones.
- No formally outward-rounded FP interval proof. No established wall-time
  advantage or target decoding accuracy improvement.
- P4 remains HOLD until independent higher-dimensional REAL EEG source
  geometries and robust numerics/runtime/publisher-novelty review.
