# Public EEG data custodian technical clarification — DRAFT ONLY / NOT SENT
Study: *When EEG Participant IDs Share Recording Contents*
2026-10-13

**Subject:** Technical clarification request regarding numerical EEG content across released identifiers

Dear Dataset Curator / Corresponding Author,

We are preparing a methodological study of whether participant identifiers in the public Liu2024 acute-stroke motor-imagery EEG release always correspond to distinct numerical recordings for cross-subject machine-learning evaluation. Before making a public provenance interpretation, we would value your clarification of the release and any relevant data corrections. We do not infer that matching waveforms identify the same patient, acquisition session, or data-management intent.

Our reproducible *public digital signal* observations are:

1. **Released IDs 4 and 5:** under a fixed MOABB 1.7.2 pipeline, the complete 40 processed labelled EEG trials were numerically identical. When the EDF files were read as original little-endian **digital int16 payloads in the publisher-supplied preprocessed EDF files**, 32 of the 33 aligned channel streams were byte-for-byte equal. The remaining channel is an unnamed/different channel in the digital reader. The EDF files are a publisher-supplied processed signal layer, not a confirmed inspection of independently acquired source MAT files. Different released names cannot resolve their original provenance.
2. **IDs 47 and 49:** full aligned processed EEG arrays are exceptionally highly correlated (about 0.99999999166) but are not fully identical in the raw digital EDF data. In our raw-int16 analysis, 13 of 32 nonconstant channels exceeded 0.9999 Pearson correlation. We explicitly **do not call this pair an independent near-duplicate negative control**.
3. **IDs 11 and 15:** full aligned processed EEG arrays are similarly correlated (about 0.99999998340). In raw-int16 data, 13 of 32 nonconstant channels exceeded 0.9999 correlation, but the complete streams were not byte-identical.

A subsequent retrospective **record-block** analysis of the preprocessed EDF digital payload found exact equality in 6,957/9,280 aligned, nonconstant named EEG channel-record blocks for 47/49, including 245/320 native EDF records with at least 90% of named EEG channel blocks equal (longest contiguous run: 179 EDF records). In 11/15, exact equality was only 171/9,280 such blocks. These blocks are not independent physiological trials or clinical patient counts.

These correlations are post-discovery measurements of publicly released digital content, **not identification of a shared person**. A strict physical-unit reader reported a nonconforming EDF Physical Maximum header, so the digital payload audit is not an independent check of absolute physiological amplitude. The publicly described original MAT/\`sourcedata.zip\` layer has **not yet been compared**. We have no access to private clinical enrolment records and do not request them.

**Would you be able to clarify, without sharing identifiable patient data:**

- Were each of these pairs intended to denote distinct independent **acquisition sessions**, distinct raw signal exports, distinct participants, or release-specific identifiers with another definition?
- Might multiple released IDs intentionally reference a common digital recording, overlapping windows, different montages, transformations, or repeated/exported files?\n- Does the public Figshare v5 \`sourcedata.zip\` contain the original acquisition-stage MAT recordings corresponding to IDs 4/5, 11/15 and 47/49? Could you clarify their lineage relative to the preprocessed \`edffile\` release and whether either product may have been generated from an earlier shared signal?
- Are there public metadata, acquisition logs, anonymised lineage identifiers, checksums, channel maps, EDF channel-calibration conventions, or updated releases that establish the origin of the 4/5, 47/49 and 11/15 signals?
- How should the extra 33rd unnamed EDF channel and nonconforming Physical Maximum metadata be interpreted? Is there a documented export or conversion convention?
- Is a public correction or provenance note already available? If a relevant explanation exists, what source should be cited in the eventual publication?

We can provide scripts and nonidentifying aggregate fingerprints, preprocessing choices and GitHub Actions receipts. We will not redistribute EEG waveform contents, seek identifiable clinical data, or infer patient identity from numerical similarity.

The matched recording-content evaluation in our study is an explicitly retrospective comparison of software pipelines. Its reported AUC differences are not assumed to be causal patient-level leakage effects.

We would be grateful for any public or anonymised technical clarification you are comfortable sharing, including if an explanation is unavailable.

Sincerely,

[Actual corresponding author and affiliation, to be approved by all authors]

---
**STATUS:** Draft **not sent**; no dataset custodian has replied. Must verify author approval, public dataset terms, contact details, and appropriate respectful notification before external communication. Do not attribute any explanation to the custodians until it is received and verified. No clinical patient identity claim is authorised.
