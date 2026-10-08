# Real public EEG verdict: zero usable correction coverage (predeclared source-anchored pilot)

**Decision: NOT an effective method yet. Do not merge as a claimed successful correction algorithm, benchmark result or flagship method.**

## Authentic unmodified source data / execution

[GitHub Actions original EDF CPU run 37830059962 — success](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37830059962) on original code head `d98aa0110b4a91218924b86a04a1079bc4ecd48f` loaded the original public
[PhysioNet EEGMMIDB v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)
`S001R01`, `S001R02`, `S002R01`, `S002R02` EDF files (two subjects, R01 eyes open and R02 eyes closed), accepted 64 actual EEG channels at 160 Hz and followed the already committed 1–35 Hz and source-role rules. The same public run passed **5/5 pure algorithmic unit controls** (those controls alone do not show real EEG utility). The complete 16-cell JSON, actual public raw-file SHA-256 provenance and outcomes are attached as [original Actions ZIP artifact 11572559912](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37830059962/artifacts/11572559912) (ZIP SHA-256 `121d49acf16e0fb1b985d78ff1f0d520f9a6bcfb721ff5ad37ed97f76b08ee8c`).

Public results **exactly under originally frozen gap >=0.012 per uncertain channel**:

| Primary quantity | Observed |
|---|---:|
| Unique original EEG participants | 2 |
| Independent source recording file hashes | 4 |
| Repeated nested subject × condition × injected permutation trials | 16 |
| Authorized physically correct repairs | **0** |
| Wrong authorized repairs | 0 |
| Refusals | **16** |
| Corrections on clean/no-corruption controls | 0 |
| Useful correction coverage | **0%** |

**Zero false corrections here is NOT validation of a safe correction method**, because the gate refuses **every** case, including clean no-change cases.

## Cause and untuned negative-control audit

The two temporal witness assignments agreed in 12/16 cells. All four S002 eyes-open-to-eyes-closed conditions failed witness agreement. But every single condition also failed the more fundamental numerical uniqueness threshold. The per-case minimum normalized global alternative-assignment gaps lay only in **[0.00047773, 0.00062107]**, over 19-fold below the predeclared 0.012 gate. Its currently chosen threshold refuses regardless of whether the physical channel identity information is sufficient for a weaker or calibrated criterion.

**Why all four injection severities show the same gap for each subject-condition cell:** each controlled injection simply permutes target rows of the same observed source matrix. A Hungarian cost matrix subjected to a row permutation has the same optimal assignment cost, and the **same globally best distinct-assignment cost**. Hence the globally normalized uniqueness gap is *exactly invariant* to the injected row permutation. Treating those four reindex injections as four independent statistical samples would be pseudoreplication. This equivalence is a mathematical property of bijective matching, not proof of model recovery.

The scope is an exploratory source-anchored identity inference; no original pretrained NeuroRVQ or clinical task score was measured. No naturally miswired electrodes were observed. The reference assumes correctly named trusted anchors and a same-person source session. No risk certificate or error bound was established.

## Scientific stop / resume decision

- **Stop the present fixed-threshold gating method as a candidate high-impact original paper.** Do not claim improvement or external adoption.
- **Do not post-hoc lower 0.012 to ~0.0005 using these four recordings, then present the same 16 cells as a new prospective success.** Such exploratory tuning would require a new subject-disjoint calibration/test split and explicit disclosure, followed by frozen inference on untouched subjects.
- If resuming, establish an actual new target subject/session cohort and independently calibrated selective-risk/coverage; use real fixed pretrained NeuroRVQ embeddings and a head/task outcome (not identity proxies alone), physically trusted and adversarially corrupted anchor controls, and a strong naive-montage/coordinate registration baseline.
- Prefer meanwhile the **already independently replicated and adopted upstream Braindecode NeuroRVQ implementation** or the stronger true closed-loop frozen-policy SemRepair experiments when deciding which flagship to show researchers.

This evidence pack captures a genuinely useful failed intervention and a structural equivalence control. Neither proves a new original technique nor contributes another externally accepted PR.
