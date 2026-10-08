# Electrode identity: prospective real-data falsification protocol v1
**Status: protocol only. No model training, real EEG outcomes, new downstream gain, independent replication, or external adoption is claimed.**

## Hypothesis and what would refute it
Electrode-identity metadata and abstention may improve reliability under missing or mislabeled channels compared with naive positional tensor alignment. Merely reordering known labels exactly is a solved bookkeeping operation and **not** novel research. A new research method is justified only if it outperforms matched-information spatial interpolation, graph/montage-aware models and identity-conditioned masking on real held-out EEG subjects when sensor identity is uncertain or partly missing.

## Primary design
1. Use publicly redistributable or openly obtainable EEG with verified channel names, sensor locations and subject identifiers. Prefer PhysioNet EEG Motor Movement/Imagery (EEGMMIDB) with documented channel montage and subject-level splits. Do not silently merge incompatible montages, reference choices or sampling rates.
2. Hold out entire subjects before training or threshold calibration. Freeze one train/validation/test subject manifest, one source pretraining revision and one signal processing recipe before touching test outcomes.
3. Freeze one baseline EEG encoder and task head from training subjects. On each held-out trial, compare **identical signal bytes** under (a) original labels/order; (b) randomly permuted channel order with preserved true labels; (c) partially missing channels (10%, 25%, 40% of the declared montage); (d) injected incorrect metadata; (e) unknown or duplicate labels. Fix perturbation assignments per subject/trial before observing predictions.
4. Include source-channel exact reindexing oracle (for permutation only), positional tensor baseline, missing-channel zero-fill, spatial interpolation using known electrode coordinates, channel-dropout-trained baseline, and an identity-aware graph/geometry encoder. Every algorithm is given matched side information; no method may secretly receive true hidden electrode IDs unavailable to comparators.
5. Evaluate sensitivity of predictions and representation drift, classification balanced accuracy, macro F1, negative log-likelihood, selective risk/coverage curves and abstention false acceptances. Report actual latency and parameter counts. Subject-stratified paired uncertainty; raw per-subject scores; disclose subjects unable to be processed.
6. For *wrong* metadata, document an independently trusted source of physical electrode identity (external montage record / technician declaration). If no trusted provenance exists, a correctness certificate is impossible: mark as *uncertain* rather than silently repairing labels from predictions.
7. Freeze selective-classifier thresholds using validation subjects only. Test each method at equal target coverage and show risk-versus-coverage, not just rejection frequency. An always-refuse algorithm is not a successful solution.
8. Preserve negative findings. If exact reindexing completely resolves the only effect, classify as a data-contract regression, **not an innovative EEG algorithm**. If channel-dropout or spatial interpolation matches the proposed method, stop novelty claims.

## Proposed release gate
- Original seed/split/metadata manifest and public scripts, SHA-256 fingerprints, environment and test-level logs.
- Two **separate** EEG datasets or independently characterized montage changes are desirable before a generalization claim.
- A strong independent reviewer must be able to reproduce primary score and reconstruct a held-out subject from source data.
- Run the accepted Braindecode NeuroRVQ implementation unmodified as a starting baseline; keep the pending EEG-CLIP upstream PR unchanged unless the maintainer explicitly requests an integration test.
- Compare against methodological prior work before naming a new module, especially montage-invariant models, missing-channel robustness, interpolation and selective prediction.

## Explicit stop/go decision
GO toward a paper only if model adaptation gives a reproducible advantage over strongest matched-information baseline on disjoint subjects **and** has a genuine mechanism beyond channel ordering. Otherwise: retain the data-contract correctness finding as library engineering and prioritize a different original question.

## External-review boundary
- Upstream NeuroRVQ PRs #1218/#1223 were merged and model reproduction was recorded by maintainers; this protocol **is not** a maintainer-reviewed study.
- Braindecode EEG-CLIP #1200 remains an independent implementation proposal; do not present this protocol as a requirement imposed by its maintainers.
- No benchmark acceptance threshold or publication timeline is promised by this file.
