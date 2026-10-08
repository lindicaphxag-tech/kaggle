# Pre-outcome prospective EEG active provenance experiment, v1 — 2026-10-09

**No actual outcomes have been observed for test subjects 13–16.** Publication-grade claims remain blocked pending an independent review of the protocol and raw results.

## Frozen subject split
PhysioNet EEGMMIDB runs 4, 8, 12, two motor imagery labels T1/T2 only, 64-channel original EDF.
- Training: subjects 1–8
- Validation (hyperparameter selection): subjects 9–12
- Prospective test: **subjects 13–16 only**, unobserved for this method as of this freeze
- Input 160 Hz, CAR, epoch from 0.5 to 3.5 sec, bandpass 8–13 and 13–30 Hz, SPD covariance shrinkage=0.08, log-Euclidean vectorization. Frozen regularization grid [0.0001,0.001,0.01,0.1,1]. Only validation subjects choose C.

## Controller-like intervention with external provenance
Injected channel swaps ONLY within the two pairs C1/C3 and C2/C4. Each possible pair mapping is evaluated with the **same frozen** filter-bank log-covariance classifier. The algorithm sees the set of candidate decision margins and trusted ID answers only when it queries. It never reads the secret injected permutation for its query selection, model confidence or test label.

Optimize the minimum *worst-case* number of trusted electrode-identity questions required until all remaining permissible mappings yield the same classifier sign. An independent brute-force verifier checks exact decision-tree optimality on 100 three-electrode cases. An equal-information random-identity-query order is a weak cost comparator, **not** an assertion of expected-query optimality.

## Endpoints to publish regardless of sign
- Original trained classifier BAcc on unseen subject labels.
- Deliberately swapped-position input BAcc; post-query/canonical BAcc (not necessarily higher than original).
- Initial zero-query coverage, mean and worst-case number of human/metadata queries, comparison with fixed-seed random-order queries, fraction irreducibly refused.
- Every per-trial injected mapping and queried row, subject ID, original EDF SHA256, tool versions, no hidden test access.
- Test labels used **only** for outcome calculation; no threshold or model selection.
- Exact assignment invariance is **not** correctness of the clinical, cognitive or motor imagery label.

## Hard falsification criteria
If no improvement in query efficiency or useful initial coverage compared with baselines, report a negative method result. If a trusted physical identity oracle is unavailable in deployment, the algorithm cannot resolve ambiguity by querying and must abstain. Training/evaluating the frozen conventional classifier does not establish a benefit for a pretrained EEG foundation model.

No model benchmark result on this cohort exists at preregistration time, and no technical reviewer or journal has accepted this proposal.
