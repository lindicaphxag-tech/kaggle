# Finite evidence redundancy for SemRepair (2026-10-10)

Status: author-run finite proof and synthetic experiment, not a physical robot result or journal publication.

## Constructive result

Given a finite catalog of hidden implementations, each implementation has a complete permitted patch class and a vector of expected categorical evidence. Suppose at most f independently delivered readings may be replaced. Different permitted patch classes can be distinguished with total coverage exactly when their selected signatures differ in at least 2f+1 channel positions. Otherwise a common received vector exists within f edits of both; a zero-error decision must sometimes refuse. This is the classical Hamming-distance condition, not a new coding theory result.

The implementation searches all eligible channel subsets of at most 18 elements, minimizing exact rational acquisition cost. The decoder permits a patch only if every compatible implementation agrees. Unknown readings are rejected.

## Executed fixed example

Two implementation classes with distinct patches and three independent binary channel signatures. One arbitrary substituted channel is permitted; minimum selected cost is 3 unit costs. In eight designed cases (two worlds, each with no change or one of three individual flips): naive one-channel guess = 6 correct / 2 wrong; two-channel selective rule = 4 correct / 4 refusals; three-channel rule = 8 correct / 0 wrong. These are not independent real experiments or empirical robot improvements.

Local eight unit tests passed. The public GitHub Actions workflow ran the newly added tests successfully on Python 3.10, 3.12 and 3.13: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37998562778

## Related accepted work and limits

Decision-region learning was studied in AISTATS 2014: https://proceedings.mlr.press/v33/javdani14.html . Cost-aware diagnosis was studied in ICML 2014: https://proceedings.mlr.press/v32/cicalese14.html . Actual VLA diagnosis and repair research in ICML 2026 is broader in robot-task scope: https://proceedings.mlr.press/v306/yan26r.html . We cannot claim invention of these existing optimization or coding principles.

The critical unresolved issue is evidence independence: three hashes issued by a single authority do not constitute three separate failure domains. This finite method does not authenticate software origin, measure field compromise rates, cover unknown controllers, prove task success or install repairs. All physical-system and top-journal claims remain contingent on independent authentic source checks and task-level blinded replication.
