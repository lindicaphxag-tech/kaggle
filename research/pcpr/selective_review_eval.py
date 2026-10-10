"""PCPR v0.5: matched-risk evaluation that rejects vacuous abstention.

This protocol is designed for *future independently adjudicated* PR-level
results; it does not infer truth from a merged patch, an author-produced test,
commit text or a maintainer label. Zero known labels => NO EFFECTIVENESS SCORE.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,random,hashlib,collections,json

@dataclass(frozen=True)
class Judgment:
 pr:str
 repo:str
 truth:str                  # correct / incorrect / unadjudicated
 approved:bool             # explicit unqualified PR approval, not scoped local evidence
 score:float|None          # independent frozen model confidence/risk ranking
 cost_seconds:float
 rubric_version:str

def wilson(k,n,z=1.959963984540054):
 if n<=0:return None
 rate=k/n
 den=1+z*z/n
 center=(rate+z*z/(2*n))/den
 rad=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
 return (max(0.0,center-rad),min(1.0,center+rad))

def evaluate(rows,min_coverage=0.30):
 if not rows: return {"status":"NOT_EVALUABLE_EMPTY","reason":"No independent PR ground truth"}
 ids=[x.pr for x in rows]
 if len(ids)!=len(set(ids)):raise ValueError("Duplicate PR units")
 if any(x.truth not in ("correct","incorrect","unadjudicated") for x in rows):
  raise ValueError("Invalid truth; do not infer from merged status")
 if any(x.cost_seconds<0 or not math.isfinite(x.cost_seconds) for x in rows):
  raise ValueError("Invalid run cost")
 if not (0<min_coverage<=1):raise ValueError("Invalid coverage target")
 if any(not x.rubric_version for x in rows):raise ValueError("Missing frozen label rubric")
 total=len(rows)
 labelled=[x for x in rows if x.truth!="unadjudicated"]
 approved=[x for x in rows if x.approved]
 unknown_approved=[x for x in approved if x.truth=="unadjudicated"]
 wrong=[x for x in approved if x.truth=="incorrect"]
 right=[x for x in approved if x.truth=="correct"]
 denom_known=len(approved)-len(unknown_approved)
 coverage=len(approved)/total
 result={
   "status":"INSUFFICIENT_EVIDENCE",
   "n_pr":total,
   "n_repositories":len({x.repo for x in rows}),
   "n_independently_labeled":len(labelled),
   "n_unadjudicated":total-len(labelled),
   "n_approved":len(approved),
   "n_approved_unadjudicated":len(unknown_approved),
   "coverage_all_prs":coverage,
   "target_coverage":min_coverage,
   "false_authorizations_known":len(wrong),
   "right_authorizations_known":len(right),
   "known_authorization_denom":denom_known,
   "false_authorization_rate_known":len(wrong)/denom_known if denom_known else None,
   "wilson_interval_pr_iid_approximation":wilson(len(wrong),denom_known),
   "n_correct_refused":sum(x.truth=="correct" and not x.approved for x in rows),
   "cost_seconds_total":sum(x.cost_seconds for x in rows),
   "interpretation":"Descriptive PR-level figures; not a guarantee against project-cluster dependence",
 }
 if coverage<min_coverage: result["reason"]="COVERAGE_BELOW_PREDECLARED_TARGET"
 elif not labelled or unknown_approved:
  result["reason"]="NO_COMPLETE_INDEPENDENT_GOLD_FOR_APPROVED_CASES"
 elif len({x.repo for x in rows})<3:
  result["reason"]="TOO_FEW_REPOSITORY_CLUSTERS_FOR_GENERALIZATION"
 else:
  result["status"]="DESCRIPTIVE_ONLY_PENDING_STRONG_BASELINES"
  result["reason"]="No superiority claim without paired costs and source-equivalent comparisons"
 return result

def strict_paired(a,b):
 """Paired exact sign test of *wrong whole-PR approvals* on identical gold.
    
    DOES NOT permit a lower-error result at lower coverage to count as a win.
    """
 ar={x.pr:x for x in a};br={x.pr:x for x in b}
 if len(ar)!=len(a) or len(br)!=len(b) or set(ar)!=set(br):
  raise ValueError("Not the same PR cohort")
 if any(ar[k].truth!=br[k].truth or ar[k].truth=="unadjudicated" for k in ar):
  raise ValueError("Independent shared truth missing")
 ea=evaluate(a);eb=evaluate(b)
 if ea["coverage_all_prs"]!=eb["coverage_all_prs"]:
  return {"status":"NOT_MATCHED_COVERAGE","coverage_a":ea["coverage_all_prs"],
          "coverage_b":eb["coverage_all_prs"]}
 a_only=sum(ar[k].approved and not br[k].approved and ar[k].truth=="incorrect" for k in ar)
 b_only=sum(br[k].approved and not ar[k].approved and ar[k].truth=="incorrect" for k in ar)
 n=a_only+b_only
 p=min(1.0,2*sum(math.comb(n,k) for k in range(min(a_only,b_only)+1))/(2**n)) if n else 1.0
 return {"status":"MATCHED_DESCRIPTIVE_TEST","n_shared":len(ar),
         "n_clusters":len({x.repo for x in a}),
         "incorrect_a_only":a_only,"incorrect_b_only":b_only,
         "uncorrected_two_sided_exact_discordance_p":p,
         "note":"Exploratory exact paired test; not repository-cluster corrected"}

def cluster_bootstrap(rows,n_resamples=800,seed=17):
 """Project-level cluster bootstrap. Return None for tiny project count."""
 repos=sorted({x.repo for x in rows})
 if len(repos)<5:return None
 grouped={r:[x for x in rows if x.repo==r] for r in repos}
 rng=random.Random(seed);rates=[]
 for _ in range(n_resamples):
  selected=[rng.choice(repos) for _ in repos]
  items=[x for r in selected for x in grouped[r]]
  known=[x for x in items if x.approved and x.truth!="unadjudicated"]
  if known: rates.append(sum(x.truth=="incorrect" for x in known)/len(known))
 if len(rates)<n_resamples//2:return None
 rates.sort()
 return {"lower":rates[int(.025*(len(rates)-1))],
         "upper":rates[int(.975*(len(rates)-1))],
         "n_draws":len(rates),"unit":"repository"}

if __name__=="__main__":
 print(json.dumps(evaluate([]),indent=2))
