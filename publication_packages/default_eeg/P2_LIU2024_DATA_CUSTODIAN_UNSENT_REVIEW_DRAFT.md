# Technical clarification request — UNSENT, author review required

**Intended recipient:** Corresponding author of the Liu2024 public EEG release, as identified by the MOABB dataset metadata (Xuanwu Hospital, Capital Medical University). Confirm the currently appropriate recipient/address against the original *Scientific Data* article before sending. **No contact has yet been made by this workflow.**

**Subject:** Technical clarification: released Liu2024 digital EEG records associated with identifiers 04 and 05

Dear Prof. Hao and colleagues,

We are conducting a reproducibility study of the publicly distributed motor-imagery EEG dataset described in your *Scientific Data* publication, DOI: 10.1038/s41597-023-02787-8 (Figshare release 10.6084/m9.figshare.21679035, with the repository version pinned for analysis).

In a strictly technical comparison of the publicly released EEG files, we observed that the 40 processed labelled motor-imagery epochs associated with released identifiers 04 and 05 are numerically identical under a fixed MOABB 1.7.2 EEG preprocessing pipeline (8–30 Hz, 160 Hz, 29 EEG channels). The same complete 40-trial match persists under a second processing configuration (6–35 Hz, 128 Hz). We additionally compared original EDF **digital sample payloads** using a minimal reader independent of MNE: the two released IDs match byte-for-byte in 32 of 33 channel sample streams, with the sole difference at an unnamed final channel. An unrelated control pair (47 and 49) matched in only 1 of 33 channel streams. A strict pyEDFlib physical-sample parser rejected an EDF header field reporting Physical Maximum; our byte-level result therefore concerns integer sample content, not independently validated physical amplitude calibration.

We emphasize that these observations **do not demonstrate** that two patients have the same clinical identity or establish any cause for the overlap. We would be grateful for your clarification on:
1. Whether released identifiers 04 and 05 are expected to contain independent EEG recording contents or whether any release, conversion, repackaging or indexing step could account for the digital match.
2. Whether a corrected public data release or an erratum is available, and which immutable file version should be used for independent analysis.
3. Whether the nonconforming EDF Physical Maximum header field reflects a known export issue and, if so, which reader or conversion pipeline you recommend.

We can provide the complete reproducible code, software versions, nonidentifying channel-by-channel equality reports and SHA-256 file receipts for your independent assessment. We have not redistributed raw clinical EEG or made any inference concerning identifiable participants.

We would appreciate the opportunity to reflect your technical explanation accurately in the submitted manuscript, including an explicit acknowledgment of your response where appropriate.

Sincerely,

[Verified corresponding author, affiliations, contact email, and coauthor approval — TO COMPLETE BEFORE SENDING]

Public reproduction links for technical inspection:
- Raw digital payload checker: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319951161
- Alternative processing audit: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319497449

**Status:** draft only, not sent or approved on behalf of any author.
