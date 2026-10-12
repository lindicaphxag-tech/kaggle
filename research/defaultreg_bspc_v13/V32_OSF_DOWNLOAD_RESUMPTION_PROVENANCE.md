# V32 transport-only resumption disclosure (NOT a second independent first-run)
2026-10-13.

The FIRST genuinely new Dreyer2023A source-locked EEG workflow
38299747910 completed all 12 source IDs, wrote SHA256
9bfc60e1da10575315917aa6662532198a892dfa99f045341bb71adde58a04b9,
and subsequently evaluated 5 held target IDs 13..17.
At held ID 18 it was interrupted by the OSF file endpoint
files.de-1.osf.io ReadTimeout(timeout=30).
No full held-target results JSON was produced. This is neither
a negative experiment nor a result that may be silently omitted.

A transport-only rerun is permitted with ALL original protocol
parameters, source IDs, held IDs, channels, algorithms, objective,
source fold partition, alpha grid and original MOABB 1.7.2 unchanged.
Only the third-party HTTP file downloader receives an extended
read timeout and automatic retry for transient transport failure.

The resumed pipeline MUST recompute the SOURCE choice file byte-for-byte
and compare SHA256 against the original first-run digest ABOVE,
BEFORE opening any held target EEG. Mismatch aborts the rerun.
No observed target AUC values were used to make this decision;
the original source choices remain the only admissible ones.
The 13..17 signals were already read in a failed attempt, so the
follow-up is no longer physically target-unopened for those persons;
it is an exact-reproduction source-locked *continuation*, not a
new independent confirmation. All 12 held IDs must be evaluated.

The method claim is constrained by the unchanged V32 protocol:
only nonzero-alpha nonparallel matrix direction + true held-person
mean AUC >= +.01, exact 2-sided sign-flip p<.05 and no increased
individual harmed-person count qualify for method promotion.
If the resumed download fails or source SHA diverges, no target
performance claim may be made.
