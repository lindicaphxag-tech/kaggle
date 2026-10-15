# P2 external distribution bridge (2026-10-13): Figshare v5 <-> NEMAR v1.0.3

Status: **IDENTIFIED RELEASE CHANNEL, NOT CONVERSION CHAIN VERIFIED.**
Do not claim an independent EEG acquisition, re-captured patients, or independent validation.

## Versioned public evidence
- Original paper: Liu et al., Scientific Data 11:131 (2024), DOI 10.1038/s41597-023-02787-8.
- Earlier release: Figshare, DOI 10.6084/m9.figshare.21679035.v5.
- Another distributor: NEMAR nm000158 v1.0.3 (2026-08-18),
  DOI 10.82901/nemar.nm000158.
- NEMAR describes 50 participants and MAT/EDF data. It explicitly cites
  the earlier Figshare release and the Scientific Data paper.
- NEMAR is another distribution of the SAME underlying study,
  not an independent acquisition cohort.

Metadata references:
- https://nemar.org/dataset/nm000158
- https://github.com/nemarDatasets/nm000158
- https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/liu2024.py

## Falsifiable bridge protocol (NOT EXECUTED)
1. Pin Figshare v5 and NEMAR v1.0.3 manifests to immutable hashes.
   Distinguish source from derivative and inspect whether original
   numeric product hashes match or transformation occurred.
2. Compare six MAT and EDF public-ID products, file digests and per-channel
   digital payload digests, recording channel labels, temporal alignment,
   digital calibration and exact version metadata.
3. Report byte equality, repackaged bytes and transformed correspondence
   separately; no acquisition provenance inference from filename,
   numerical correlation or BIDS participant ID.
4. Verify on genuinely unrelated known-origin acquisitions; hold them out
   from threshold fitting and use documented transformations.
5. Obtain versioned nonidentifying release-operator records or a
   custodian clarification before asserting acquisition-level ancestry.

## Claim barriers
- Persistent anomalies across Figshare and NEMAR would support
  distribution-layer persistence, not independent patient identity.
- A different or corrected NEMAR transformation must be reported rather
  than hidden or treated as a failed experiment.
- Do not redistribute EEG signals, patient metadata or identity guesses.
- External contact to the data custodian remains UNSENT; author
  verification and permission required.

This is a research gate, not new signal comparison results.
