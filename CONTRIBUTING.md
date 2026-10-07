# Contributing to the public research artifacts

This repository values **independent reproduction, falsification and narrow
upstream-quality fixes** more than feature volume.

## ClinGeom

Use the frozen branch named in the root README for independent reproduction.
Do not reproduce from rolling `main` unless the purpose is explicitly to test
unreleased work.

A useful report contains:

- exact repository commit / release branch;
- OS, Python and relevant package versions;
- exact commands;
- full pass/fail output;
- whether the result confirms, falsifies or only partially reproduces the
  claimed property;
- any numerical deviation without retuning the frozen threshold.

Negative results are first-class contributions. Please do not suppress a failure
because the expected public artifact says it should pass.

### High-value contribution types

1. independent reproduction on a clean machine;
2. minimal counterexample to T1–T14;
3. adapter/witness for a real medical framework that uses existing public APIs;
4. proof or counterexample tightening a theorem boundary;
5. a prospective upstream case that was frozen before reading its resolution.

### Low-value / excluded contributions

- stars/forks presented as validation;
- threshold tuning after observing a failure;
- historical bugs relabeled as prospective;
- broad refactors without a new falsifiable result;
- clinical-safety claims unsupported by clinical validation.

## External-upstream etiquette

Do not open an upstream issue merely to advertise ClinGeom. A useful upstream
report must stand alone as a minimal reproducer of a concrete physical-semantic
failure or ambiguity, state the expected behavior, and keep the proposed patch
smaller than the research framework whenever possible.

## License note

This repository currently has no repository-wide open-source license. Public
visibility does not by itself grant reuse rights. Until the repository owner
chooses an explicit license, reproduction reports can run the public artifact
but downstream redistribution/adoption must respect GitHub's access terms and
applicable law.
