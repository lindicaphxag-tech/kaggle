# ClinGeom — minimum-cost placement of authorized pre-fusion repairs

## What this adds

A *mechanism-level* planning subproblem, not a new abstract geometry law:
given a directed acyclic model graph, explicitly declared spatially defective
sources, irreversible fusion points, and a finite set of **independently
authorized** sites, choose the cheapest set of pre-fusion interventions that
intercepts **every** path from a bad source to destructive mixing.

The existing \`physically_align_then_mix\` repairs one explicit fiber bundle
before one dense 1×1 mixer. Multi-branch encoders also have skip connections,
concats and multiple hazardous fusions. Repairing the locally obvious branch
is insufficient if another branch still reaches a mixer uncorrected.

## Contract and interface

\`choose_repair_sites(graph, bad_sources, irreversible_fusions, admitted)\`

- Graph keys are unique **post-operator tensor states**; edges point forward
  through data dependencies. Cyclic/unresolved graphs are refused.
- A **site admission** consists of a strictly positive integer planning cost
  and a 64-hex provenance digest from a separate sound physical-repair proof.
  The digest is a handle, **not cryptographic verification of the proof**.
- An admitted site must be able to repair **all** offending channel fibers that
  can reach the site, including both sides of a concat or skip join. Never
  admit a site from shape equality alone.
- Cost is a caller-specified *planning proxy*, not automatically measured
  memory, wall-clock speed, diagnostic error, or resampling fidelity.
- Mixing nodes themselves may **never** be marked as repair sites.
- A graph that has a bad-source-to-fusion path with no admitted upstream
  intervention is refused with a concrete \`UnrepairablePath.path\`.
- For feasible instances, the selected sites block every bad-source-to-fusion
  route; the optimum total integer cost is exact relative to the supplied
  candidate sites and costs.

### Minimal concrete example

\`\`\`python
from research.medical_geometry.repair_placement import (
    AdmittedSite, choose_repair_sites
)

proof = "a" * 64  # TEST-ONLY PLACEHOLDER, NOT A REAL REPAIR CERTIFICATE
graph = {
    "phase0": ("branch0",), "phase1": ("branch1",),
    "branch0": ("concat",), "branch1": ("concat",),
    "concat": ("irreversible_mix",), "irreversible_mix": ()
}
admitted = {
    "branch0": AdmittedSite(3, proof),
    "branch1": AdmittedSite(3, proof),
    "concat": AdmittedSite(5, proof),
}
plan = choose_repair_sites(
    graph, ("phase0", "phase1"), ("irreversible_mix",), admitted
)
assert plan.sites == ("concat",) and plan.total_cost == 5
\`\`\`

The shared concat site is admissible **only if** an external channel-fiber
proof establishes that all of its branches can be physically aligned there.
Changing its cost to 7 selects both branch repairs at total cost 6. A bypass
skip path demands an extra intervention. An only-postmix candidate refuses.

## Algorithm and independent verification

The optimization is the **classical minimum-weight vertex cut**, implemented
using node splitting and integer max-flow/min-cut. **Neither graph cuts nor
max-flow/min-cut is claimed to be novel.** The methodological contribution
under construction is connecting real patient-space **repair admissions** to
this exact planner, then executing graph rewrites with auditable proof scope.

A preflight reachability check finds a concrete unrepairable path even if
every admitted repair is hypothetically inserted. Post-solve checks recompute
actual coverage of every dangerous path and verify the cut cost against
the max-flow optimum. A stable digest commits to the graph, source/hazard
sets, all candidate costs/provenance handles and the chosen site set.

Tests compare the result against a separate brute-force subset oracle on
220 seeded 7-node DAGs, plus shared-fusion, skip-bypass, multiple-hazard,
unreachable-hazard and fail-closed malformed-input cases. All tests are CPU-
only and require \`pytest\`, not medical scans, MONAI, or a GPU.

## Explicit limitations / next decisive research step

1. **This PR is a graph planner only.** It does not yet extract the graph
   automatically from arbitrary \`torch.fx\` nodes, insert repair operations,
   certify arbitrary channel joins, or measure any task-level gain.
2. The admission digest is externally supplied. Until verified native
   \`LatticeBundle\`/repair-synthesis certificates are bound to the graph
   locations, an untrusted caller could forge admissions. **Do not** call
   this a safe automatic medical-model rewrite.
3. No performance improvement is established by a smaller *estimated*
   planning cost. The next step is an actual executable MONAI branch graph
   with measured costs and a frozen held-out annotation set.
4. This is not independent external validation, clinical evidence, or an
   external upstream merge. Correct DICOM patient-frame identity must still
   be validated separately from voxel-array overlap.

The next flagship experiment should compare: ordinary MetaTensor/inverse
transforms; repair every eligible branch; minimum-cut pre-mix repair; and
forbidden post-mix-only recovery. Use identical **pretrained** models and
independently labeled patient volumes, with Dice/HD95 in common world
coordinates, error bounds, latency, memory, and refusals reported together.
