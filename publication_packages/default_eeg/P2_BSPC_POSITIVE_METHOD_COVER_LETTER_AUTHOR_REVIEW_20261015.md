# BSPC — updated method-positive author-review COVER LETTER (not approved, not sent)

Dear Editor-in-Chief,

We respectfully submit for consideration, **subject to confirmation and approval by all human authors**, our candidate manuscript entitled **"Digital EEG recording provenance: exhaustive release-content audit and shared-clock multichannel verification."** Our target is *Biomedical Signal Processing and Control*.

Different public EEG participant identifiers can refer to the same numerical contents at a processed release layer, even if filenames differ. In the publicly distributed Liu2024 EEG data, we audited all 50 EDF files and all 1,225 unordered identifier pairs. The published digital content of identifiers 4 and 5 agreed across all 29 evaluated time-varying EEG channels; two further pairs showed partial aligned digital block overlap, without establishing the original patient or session identities. A retrospective 49-content-group refitting analysis separates changes due to evaluation-unit weighting from changes due to model fitting; its differences do not identify a causal population-level leakage effect.

Our **constructive method experiment** distinguishes arbitrary single-sensor sample reuse from a shared synchronized multichannel recording clock. Using publicly distributed EEGMMIDB digital waveforms and a separate CHB-MIT seizure EEG corpus, we applied frozen, known-content transformations and predeclared incoherent-channel negative scenarios. On four additional CHB-MIT source EDFs not used in the initial two-source check, shared-clock assignment detected 32 of 32 controlled reuse examples and rejected 20 of 20 constructed negatives. A fully informed independent-lag channel-assignment comparator detected the same 32 positives, but matched four of the 20 constructed clock-incoherent negatives. These outcomes demonstrate an operational benefit under this deliberately specified provenance question, **not** novelty over published synchronization algorithms or real-world clinical detection accuracy. The test outcomes and EDF prefix hashes are archived in public-source reproducible GitHub Actions.

The paper's practical contributions are a versioned release-content census, a careful separation of released identifiers, digital content groups and unknown acquisition/clinical identities, and a worked coherence-aware evaluation tool. Neither the separately distributed source MAT arrays nor the available transformation documentation establish the original MAT-to-EDF ancestry.

All manuscript claims, authorship/order, contributions, funding, data reuse and ethics conditions, competing interests, AI-assisted writing disclosure, and exclusive-submission declarations require final independent approval by the responsible human authors. The release-custodian clarification email is saved only as an unsent draft and is not represented as correspondence already made. This cover letter must be replaced with signed author information before use.

Sincerely,

[CORRESPONDING AUTHOR NAME, AFFILIATION, CONTACT — approval required]

Evidence: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38666330505
