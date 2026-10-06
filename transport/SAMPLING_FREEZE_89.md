# Prospective Sampling Freeze — 89 Repository Frame

Status: **FROZEN sampling frame for the current SemRepair prospective study**

## Rule

The current public transport frame of **89 repositories** is frozen as the final repository sampling frame for the current RSS-2027 Stage-1 evidence cycle.

No additional repository may be added merely because the existing frame has not yet produced an eligible resolved I2 case.

## Rationale

Generation 5 froze 69 repositories. Generation 6 added 20 repositories selected from semantic-boundary-heavy infrastructure before their post-cutoff issue surfaces were inspected. The combined public frame now spans:

- VLA / policy runtimes;
- manipulation and simulators;
- ROS client/runtime/control layers;
- planning and kinematics;
- serialization / middleware / transport;
- camera / frame semantics;
- industrial robot interfaces;
- dataset / trajectory / replay infrastructure.

Continuing to expand the frame after observing zero or unfavorable cases would increase exposure but weaken the prospective interpretation by making repository selection outcome-adaptive.

## What may still change

Allowed:

- hourly transport scans over these same 89 repositories;
- first-seen capture of newly created issues;
- pre-deadline eligibility / independence triage using only the frozen issue surface;
- immutable prediction freeze for eligible I2 cases;
- post-freeze resolution collection;
- negative, unresolved, or contaminated outcomes.

Not allowed within this evidence cycle:

- adding repositories in response to a lack of I2 hits;
- removing repositories because they generate irrelevant issues;
- changing cutoffs retroactively;
- changing I0/I1/I2 definitions;
- excluding eligible negative cases from the denominator.

## Claim boundary

This freeze does not improve mechanism accuracy and does not count as L8/L9 evidence by itself. It exists to make future prospective evidence more credible and to prevent exposure optimization from becoming outcome-adaptive.
