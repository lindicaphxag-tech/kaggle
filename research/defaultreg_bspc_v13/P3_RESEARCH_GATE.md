# PAPER 3 — Source-only selector transfer limits (CONDITIONAL)
**NOT journal-ready. Do not submit concurrently with P1 in this state.**
File: `main_p3_source_only_optimisation_NEGATIVE_UNREADY.tex`.

Unlike P1 (Gaussian likelihood validation estimand anatomy), P3
would compare SOURCE-ONLY transfer policies and realized target
regret on the SAME EEG across methods. The current evidence uses
DIFFERENT algorithms and acquisition histories:
- Weibo2014 10 target people continuous v27-grid equal.
- Cho2017 40 people SCE selected fixed lambda1 and averaged
  delta -0.00005264, 2nd20 sequential after first outcomes.
- Dreyer2023A 12 people anisotropic alpha selected zero,
  -0.001088 AUC vs macro.
- Lee2019 GID pre-target source forecast negative, held A
  n=12 delta -0.003933; B failed BEFORE opening target on
  12/12 per-person covariance SHA drift between actions.
No successful newly proposed algorithm. Methods differ; don't
pool these as one predeclared hypothesis or aggregate p value.

## Hard gates for viable separate ORIGINAL manuscript
1. One frozen candidate family or common matrix of candidate
   methods evaluated on identical external datasets, group IDs,
   source persons, channel configuration and held-target budget.
2. A source-only, prospective question such as probability of
   >0.02 AUC participant harm or pairwise policy misranking
   with uncertainty and meaningful fair baseline comparator.
3. At least 2 separate truly unseen acquisitions with exact
   source artifact SHA lock and coherent failed-IO policy.
4. Fix Lee source feature reproducibility without observing B
   (serialize immutable source PREPROCESSED C or X arrays and
   verify original bytes rather than permit ambiguous tolerance
   solely to open target).
5. Exclusive contribution vs P1: previously published P1
   negative rows can only be cited as background, with clear
   data/method overlap disclosure; must NOT double count.
6. No universal no-free-lunch or claim of a new optimizer
   without an actual innovative selection mechanism.

Journal route only after these gates. Consider a brief negative
methods note if the journal explicitly accepts, but not
as a new algorithm performance paper.
