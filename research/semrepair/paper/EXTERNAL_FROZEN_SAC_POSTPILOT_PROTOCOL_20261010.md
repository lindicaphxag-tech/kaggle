# External frozen pretrained SAC Reacher-v5 post-pilot benchmark

Status: committed BEFORE held-out scores. Post-pilot design: the initial 16-seed pilot was already inspected and found no fault effect. NOT a pure preregistration.

## Genuine public pretrained agent and official physics

Farama/Minari SAC expert repository: https://huggingface.co/farama-minari/Reacher-v5-SAC-expert . The exact 40-hex checkpoint source commit is ad8fadfbe3ab40ce9c1bae5b33641041056b3090. The actual loaded checkpoint SHA-256 from successful pilot CI is 095cd4cfe742a07c9e2c5587ca95fe1ce97360efaaa02c7e6569fe9ca00c6454. Reacher-v5 is the official Farama Gymnasium MuJoCo model, not our earlier hand-authored two-link arm. Runtime is pinned Gymnasium 1.1.1 and Stable-Baselines3 2.6.0. Zero new actor training steps.

Pilot: 16 paired seeds x five methods = 80 executed episodes. With planted source shear 0.8 and internal clip 1, each arm had identical original official mean reward **-2.593024** and 16/16 ever reaching 0.075 metres. This is a NEGATIVE fault-transfer result, preserved at https://github.com/lindicaphxag-tech/kaggle/actions/runs/38007320983 . All test seeds used below differ from the pilot.

## Fixed new evidence grid

All of the following were committed before seeing their held-out evaluations and MUST be reported, including zero and harmful cases:

- Independent from pilot seed identities: 32 environment seeds 23011 through 23042 inclusive.
- All five predeclared fault magnitudes: compensated inverse shear k=0.8,2.0,3.5 at clip L=1; k=2.0,3.5 at L=0.4. Source producer P=[[1,k],[0,1]], controller C=[[1,-k],[0,1]]. The fault includes a planted INTERMEDIATE software clip; correcting the faulty source returns to the native official action interface. This is an engineered software defect, not a malicious physical manufacturer.
- Six fully paired intervention arms: healthy, fault untouched, equal-information inverse-matrix blackbox, globally optimal instantaneous action projection across both the software and official actuator clips, a candidate atomic repair inferred from two numeric linear taps and one heldout tap, and a source-known atomic repair upper bound.
- Report official Reacher reward, average and terminal native fingertip distance, reached-goal count, actual action error, and number of requested controls whose producer interface triggers clipping. 32*5*6=960 episodes. Nested actions and multiple severities of the same 32 seeds are NOT independent hardware or robot-model observations.
- The numerical diagnostic reads generated interface tapes under small unsaturated inputs and never reads a symbolic fault label. Its selected SOURCE PATCH IS NOT ATTESTED and is only a simulator candidate, not genuine software installation.
- Full-information action oracle knows hidden P,C,L and the outer action clip; it is exact for each time-step Euclidean action tracking, but **not globally optimal cumulative task control**. The source algorithm does not gain an informational advantage over it.
- After results, no severity may be dropped on grounds of inconvenience. If actual real task return fails to improve, the paper must say so.

## Evidence availability and editorial position

Scripts: research/semrepair/external_reacher_sealed_holdout.py; oracle: research/semrepair/strong_clipped_actuator_oracle.py; tests and workflow semrepair-external-sealed-holdout.yml. The original pilot had confirmed public GitHub CI; this heldout experiment must not be claimed verified before CI passes and actual logs/raw files are examined.

Remaining large paper gaps are independent external model architectures, pre-trained VLA (SAC is **not** VLA), secure source-signing and actuator telemetry, real patch installation, multiple robot embodiments and strong full-horizon MPC recovery. A successful postpilot SAC case on one external model alone does not reach T-RO, RA-L, or high certainty CAS zone 2.
