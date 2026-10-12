# V31B definitive negative result: Cho2017 held-person two-block replication
2026-10-13. Verifiable REAL EEG, no result suppression; exploratory second block.

## Original frozen source and first-run provenance
- Source people: Cho2017 IDs 1..12, exactly C3/Cz/C4, 8-30 Hz,
  MOABB 1.7.2, 160 Hz, left/right MI, original fixed tangent LDA path.
- Immutable SOURCE choices SHA256 (original plus reproduced reserved run):
  677f848de0e3bc3122892e5bec55b20dc7808ab04093bdaeb33f1bd0a923f796
- Chosen lambdas always SCE-rho1=1.0, original grouped source macro=.5,
  SCE-rho0=.3, proper predictive source NLL=.1, historical NLL=0.
- First true EEG 20-person run: GitHub Actions 38295151137 (SUCCESS);
  independent frozen 20-person no-retraining audit 38295940609 (SUCCESS).
- Reserved 20-person NO RETUNE run: GitHub Actions 38296239584 (SUCCESS);
  runner required EXACT original source SHA before target IDs 33..52 read.
- Combined independent source-lock/individual-contrast audit:
  GitHub Actions 38296956773 (SUCCESS) downloaded both original artifacts,
  checked byte-identical source decisions and all 40 paired records,
  rederived paired AUC and exact 2^20 sign flips for each block.

## Definitive matched source-model real EEG target outcomes
| Strategy | First held 13-32 N20 | Reserved held 33-52 N20 | Total N40 |
| --- | ---: | ---: | ---: |
| Source Contrast Envelope rho1 | .618230 | .615761111 | .616995556 |
| Original source-group macro 27-grid | .614835 | .619261389 | .617048194 |
| SCE rho0 ablation | .610300 | .618996806 | .614648403 |
| Proper source-mean predictive Gaussian NLL | .601830 | .615486944 | .608658472 |
| Historical validation-class-centred score | .593355 | .609806667 | .601580833 |
| Ledoit-Wolf | .593535 | .609937778 | .601736389 |
| OAS | .593580 | .609953889 | .601766944 |
| SCE - source macro | +.003395 | -.003500278 | -.000052639 |
| Held trials | 4,000 | 4,040 | 8,040 |
| >.02 harmed target people | 0 | 1 | 1 |
| Two-sided all-2^20 signflip reference | .093084335 | .166742325 | not predeclared |

- Both 20-person blocks retained without any outcome-dependent selection.
- Total 40 people: 21 with positive individual contrast, 19 negative.
- Exact *descriptive* 40-person bootstrap 95% CI for SCE minus macro:
  [-0.00321796875, +0.00297268750], crossing zero.
- Exploratory Monte Carlo random-sign reference (200,000 signs):
  p about 0.97429; NOT a prespecified, confirmatory 40-subject p.
- Original first block was non-significant and positive by 0.003395;
  sequential reserved block was negative by 0.003500278. Neither supports
  reliable real transfer benefit; pooled performance approximately identical.
- One reserved subject (ID 48) lost .0296 AUC; loss means
  do NOT promote a zero-material-harm claim across the combined experiment.
- Both target blocks use *the same data acquisition* and source training
  pool. They are not two independent datasets or independently registered
  experiments. The reserved protocol was committed *after* seeing the first
  block, before reading reserved participants; disclose selection chronology.
- SCE source conditional dominance certificate was FALSE. The method
  selected lambda=1.0, consistent with an endpoint saturation concern.
  A sharp conditional worst-case margin theorem is not a target coverage
  or a transfer-generalisation theorem.

## Journal promotion decision
Paper A: preserve corrected Gaussian predictive vs class-centred NLL,
existing prospective and honest external negative evidence. Cho40 shows
SCE is not yet a superior new selector. Do not claim L8/9 or top-tier
performance from this attempt; do not filter away the 20 unfavorable
reserved people. The V31 study is an added falsification of
source-only mean/robust surrogate optimization.
Paper B: keep V26-V30 record identity and 2x2 model/selector auditing
independent as recording-provenance work.

## Follow-up standard
Any *new* method must explain WHY the first and reserved blocks reverse
BEFORE exposing a separate new dataset, and compare against the strong
source-macro, correct Gaussian NLL, LW/OAS, tangent logistic regression,
CSP+LR and DG baselines under matched channel, calibration and
computation budgets. A new posthoc radius is disallowed as
confirmatory evidence on these already inspected Cho40 participants.
