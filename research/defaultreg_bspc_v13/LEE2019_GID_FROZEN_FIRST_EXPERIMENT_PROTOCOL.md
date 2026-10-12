# Lee2019_MI PRE-TARGET frozen source-only model & dual held validation
2026-10-13. Commit before any Lee2019_MI EEG data read in this attempt.
No matching Lee2019_MI/OpenBMI code found in indexed default repo branch;
not evidence against earlier work outside this branch or other people.

## Data and budget
Original MOABB 1.7.2 Lee2019_MI, session [1], offline training run
(train_run=True,test_run=False), two events left_hand/right_hand,
right_hand positive; C3,Cz,C4, 8-30Hz, 160Hz, baseline=None,
reject_by_annotation=False. No alternate session, labels or
channel substitution after seeing target results.
SOURCE group released IDs 1..12, held A 13..24 (12 persons),
held B 25..36 (12 persons) without any algorithm change
between A and B. IDs 37..54 remain unused. Released-ID
denominators do not establish physical-person independence.
No target feature moments, target-fitted tangent geometry,
target validation labels or held-subject AUC used in tuning.

## Operator and exact baseline
Original source-person-grouped pooled-LDA with the original 27
scalar lambda grid 0..1, original fold partition and source-macro
AUC objective, exact numerical path equality required.
Source-group-invariant LDA (GID): within SOURCE person+class
scatter divided by N-2G; balanced contrast is the average
across groups of SOURCE within-person class differences.
The resulting linear classifier is invariant to any independent
class-shared group translation in FIXED source-tangent coordinates.
It is classical within-group LDA, not guaranteed mathematically
novel. GID has zero extra hyperparameters beyond original lambda.
Two structural ablations: within-class scatter only, balanced
source contrast only. All independently source-macro tune
lambda on EXACT same 5-fold group source CV, refitting geometry
and all model statistics from source training folds only.

## Source-only pre-outcome transfer forecast
Source group paired macro-AUC mean GID minus original >= +0.005
and >= ceil(.60*12)=8 group-source positives -> forecast POSITIVE,
else forecast NEGATIVE. The binary forecast is source-only and
locked in source manifest with SHA256 BEFORE held person I/O.
No guarantee from sources to an arbitrary new population is possible.
Even if negative, both held groups still evaluated to test
the signed forecast, never to select another policy.

## Comparisons, mandatory outcomes
Same 3 electrodes and source training budget: old pooled-LDA,
GID, its two ablations, analytic LW and OAS, fixed full shrinkage,
tangent LogisticRegression(C=1), Riemannian MDM, and fixed
CSP+LR n_components=3 when pinned version supports it.
Do not compare to high-channel deep systems as if same budget.
First A n12 and confirmatory B n12 evaluated with identical
frozen source model and choices. Each yields all 12 individual
AUC and paired changes, exact 2^12 signflip p (two-sided),
descriptive bootstrap CI with seed 20261013, >.02 harms,
classification direction cosine and baseline ratios.
NO participant- or trial-dropping after first target access.
Methods not executable due external file server errors receive
NOT COMPLETED, not an inferred result.

## Promotion HARD GATES (scientific not acceptance odds)
(i) GID direction nonparallel to original cos <1-1e-5;
(ii) source-only forecast POSITIVE;
(iii) BOTH held A and held B each mean paired AUC gain >=.01;
(iv) BOTH A and B exact 2-sided signflip p<.05;
(v) zero harmed (>0.02 AUC loss) people in each block;
(vi) ablations and comparators must not render supposed
GID mechanism superfluous.
If any fail, retain negative study and do NOT promote new
algorithm in BSPC abstract. Two held blocks are within
ONE public acquisition, so still not a separate-acquisition
replication. No independent registry of this protocol.

## Provenance
First run must write source choices SHA256 before target trial IO.
Transport-only retries require EXACT byte-identical SOURCE SHA,
before resuming. All source group outputs and both held blocks
must be made immutable and independently audited. No source
distribution shift guarantee from this study.
Main BSPC paper remains focused on source validation objective
mismatch and three-acquisition anatomy; V26-V30 recording identity
paper remains separate.
