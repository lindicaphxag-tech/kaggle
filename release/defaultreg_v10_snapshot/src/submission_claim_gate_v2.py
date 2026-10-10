"""Evidence-conditioned manuscript claim gate for Default Regularisation.

The gate prevents wording from outrunning the evidence ledger. It is not a
style checker: prohibited claims are tied to concrete evidence requirements.

Input:
- evidence JSON containing boolean gates.
- one or more manuscript text/tex/md files.

The gate fails closed when a claim requiring a higher evidence tier appears
before its prerequisites are satisfied.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path

RULES=(
  {
    "id":"no_validation_free",
    "patterns":(r"\bvalidation[- ]free\b",r"\bwithout validation\b"),
    "always_forbidden":True,
    "reason":"Source-participant cross-fitting uses labelled source validation information; use source-only cross-fitted / zero-target-calibration."
  },
  {
    "id":"systematic_field_misspecification",
    "patterns":(r"systematically mis[- ]specified",r"systematic(?:ally)? wrong",
                r"field[- ]wide default(?:s)? (?:is|are) wrong"),
    "requires":("three_cohort_full_domain_confirmed","historical_table_integrity_resolved","true_predata_frontier_two_cohort_complete"),
    "reason":"Field-level/systematic language requires multi-cohort full-domain confirmation and clean evidence provenance."
  },
  {
    "id":"universal_law",
    "patterns":(r"\buniversal law\b",r"\bgeneral law\b",r"\balways improves\b",
                r"\bfor all (?:subjects|datasets|cohorts)\b"),
    "requires":("three_cohort_full_domain_confirmed","mechanism_counterfactuals_replicated"),
    "reason":"Universal/general-law wording requires replicated mechanism evidence; current p/n monotonic prediction was falsified."
  },
  {
    "id":"sota",
    "patterns":(r"\bstate[- ]of[- ]the[- ]art\b",r"\bSOTA\b",r"\boutperforms all\b"),
    "requires":("strong_same_budget_baselines_complete","external_benchmark_panel_complete"),
    "reason":"SOTA needs matched information-budget benchmark coverage; cross-paper metric comparisons are insufficient."
  },
  {
    "id":"tnsre_ready",
    "patterns":(r"\bTNSRE[- ]ready\b",r"\bclinical(?:ly)? validated\b",
                r"\bonline calibration[- ]free deployment\b"),
    "requires":("tnsre_bci_utility_complete","cross_session_or_device_complete"),
    "reason":"Offline LOSO public EEG does not establish online/clinical TNSRE utility."
  },
  {
    "id":"representation_conclusion_change",
    "patterns":(r"regularisation.{0,80}(?:changes|alters).{0,80}representation",
                r"representation.{0,80}(?:changes|depends on).{0,80}regularisation"),
    "requires":("prospective_representation_interaction_confirmed",),
    "reason":"Representation-conclusion language requires the frozen S041-S060 interaction."
  },
  {
    "id":"hidden_filterbank_gain_refuted",
    "patterns":(r"hidden spectral representation gains",
                r"hidden filter[- ]bank (?:gain|advantage)",
                r"regularisation (?:reveals|recovers).{0,80}filter[- ]bank advantage"),
    "always_forbidden":True,
    "reason":"The prospective S041-S060 interaction refuted the historical hidden-filter-bank direction; discrimination-oriented regularisation strengthened WIDE instead."
  },
  {
    "id":"fctr_prospective_superiority",
    "patterns":(r"\bFCTR (?:outperforms|improves|is superior|reduces harm)\b",
                r"\bFCTR .*negative[- ]transfer.*reduc"),
    "requires":("nested_source_fctr_audit_complete",
                "fctr_prospective_target_confirmed",
                "strong_same_budget_baselines_complete"),
    "reason":"FCTR is currently a source-only constrained hypothesis. Superiority/harm-reduction language requires nested source-policy audit, untouched target confirmation, and matched same-information baselines."
  },
  {
    "id":"decision_instability_generalisation",
    "patterns":(r"regularisation[- ]induced decision instability is (?:systematic|general)",
                r"covariance fidelity (?:systematically )?underdetermines"),
    "requires":("fidelity_regret_frontier_multicohort_complete",
                "three_cohort_full_domain_confirmed"),
    "reason":"General decision-instability language requires threshold-free fidelity-regret frontiers replicated across confirmatory cohorts."
  },
  {
    "id":"no_formal_equivalence_from_one_se",
    "patterns":(r"\bformally equivalent\b",r"\bstatistically equivalent\b",
                r"\bproven equivalent\b"),
    "always_forbidden":True,
    "reason":"The 1-SE covariance-fidelity set is an operational admissibility rule, not a formal equivalence test."
  },
  {
    "id":"fctr_robustness_claim",
    "patterns":(r"\bFCTR (?:is )?robust\b",
                r"\bFCTR .*stable across (?:hyperparameters|settings)\b"),
    "requires":("fctr_sensitivity_audit_complete",
                "nested_source_fctr_audit_complete",
                "fctr_prospective_target_confirmed"),
    "reason":"Robustness language requires predeclared selector-sensitivity audit, nested source-policy validation, and untouched target confirmation."
  },
  {
    "id":"multicohort_robustness_claim",
    "patterns":(r"\brobust across datasets\b",r"\bconsistent across cohorts\b"),
    "requires":("cross_cohort_lodo_stable",
                "fidelity_regret_frontier_multicohort_complete"),
    "reason":"Cross-cohort robustness language requires dataset-level replication plus leave-one-dataset-out direction stability."
  },
  {
    "id":"s021_chronology_overclaim",
    "patterns":(
      r"S021.{0,8}S040.{0,120}(?:prospectiv(?:e|ely)|preregistered).{0,120}(?:FCTR|Validity Frontier|critical fidelity regret)",
      r"(?:FCTR|Validity Frontier|critical fidelity regret).{0,120}(?:prospectiv(?:e|ely)|preregistered).{0,120}S021.{0,8}S040"
    ),
    "always_forbidden":True,
    "reason":"S021-S040 was opened before FCTR / Validity Frontier were developed; those constructs can only be post-primary/explanatory on this cohort."
  },
  {
    "id":"prospective_source_discriminability_consequence",
    "patterns":(
      r"prospectiv(?:e|ely).{0,100}source[- ](?:participant[- ]|macro[- ])?discriminability",
      r"source[- ]macro.{0,100}prospectiv(?:e|ely)"
    ),
    "requires":("prospective_source_discriminability_target_confirmed",),
    "reason":"Prospective source-discriminability consequence language requires the frozen S021-S040 evidence gate."
  },
  {
    "id":"true_predata_frontier_claim",
    "patterns":(
      r"(?:true |genuine )?pre[- ]data.{0,100}(?:Validity Frontier|decision fragility|regularisation[- ]validity)",
      r"(?:Validity Frontier|decision fragility).{0,100}(?:true |genuine )?pre[- ]data"
    ),
    "requires":("true_predata_frontier_one_cohort_complete",),
    "reason":"Pre-data frontier language requires at least one completed cohort whose protocol and score-family were frozen before any project data access."
  },
  {
    "id":"mean_tail_fidelity_target_benefit",
    "patterns":(
      r"mean.{0,8}tail.{0,8}fidelity.{0,120}(?:reduces harm|improves|outperforms|safer)",
      r"(?:reduces harm|safer).{0,120}mean.{0,8}tail.{0,8}fidelity"
    ),
    "requires":("mean_tail_fidelity_predata_two_cohort_safe",),
    "reason":"Mean-tail-fidelity target-benefit/safety language requires the frozen two-cohort promotion gate; the completed panel currently fails that gate."
  },
  {
    "id":"external_superiority_overclaim",
    "patterns":(
      r"(?:both|two).{0,80}(?:external|true pre[- ]data).{0,120}(?:significant|confirmed superiority|holm[- ]significant)",
      r"external.{0,100}(?:superiority|target benefit).{0,80}(?:confirmed|established)"
    ),
    "requires":("true_predata_external_primary_family_significant_two_cohort",),
    "reason":"The two TRUE PRE-DATA external datasets have positive target-effect directions, but neither frozen exact sign-flip + Holm family is individually significant."
  },
  {
    "id":"external_directional_replication",
    "patterns":(
      r"(?:target|H2|H3).{0,100}direction(?:ally)? replicated.{0,80}(?:external|cohort)",
      r"(?:external|cohort).{0,100}target[- ]effect direction.{0,60}(?:replicated|positive)"
    ),
    "requires":("true_predata_external_directionally_consistent_two_cohort",),
    "reason":"Directional external-replication wording requires both TRUE PRE-DATA cohorts."
  },
  {
    "id":"two_cohort_validity_frontier",
    "patterns":(
      r"(?:two|both).{0,60}(?:cohort|dataset).{0,100}(?:Validity Frontier|validity frontier|material[- ]fragility)",
      r"(?:Validity Frontier|material[- ]fragility).{0,100}(?:two|both).{0,60}(?:cohort|dataset)"
    ),
    "requires":("corrected_material_fragility_frontier_two_cohort_complete",),
    "reason":"Two-cohort validity-frontier wording requires the corrected material-fragility analysis with paired-path uncertainty in both v5 cohorts."
  },
  {
    "id":"universal_frontier_threshold",
    "patterns":(
      r"universal.{0,50}(?:fidelity|fragility|frontier).{0,50}(?:threshold|budget|delta)",
      r"single universal.{0,50}(?:delta|regret budget|fidelity tolerance)"
    ),
    "always_forbidden":True,
    "reason":"The two TRUE PRE-DATA cohorts have materially different critical raw fidelity budgets; no universal delta is supported."
  },
  {
    "id":"simple_pn_law_refuted",
    "patterns":(
      r"(?:mismatch|regularisation).{0,100}(?:monotonically|monotonic).{0,100}(?:p/n|dimension[- ]to[- ]sample|feature[- ]to[- ]sample)",
      r"(?:p/n|dimension[- ]to[- ]sample).{0,100}(?:law|predicts|explains).{0,80}(?:mismatch|regularisation)"
    ),
    "always_forbidden":True,
    "reason":"The prospective source-budget intervention did not show the frozen monotonic p/n prediction."
  },
  {
    "id":"calibration_auc_utility_overclaim",
    "patterns":(
      r"(?:U5|U10|U20|U-all|unlabeled calibration).{0,100}(?:improves|harms|benefits).{0,80}(?:AUC|performance|utility)",
      r"(?:calibration|unlabeled target).{0,100}(?:practical benefit|deployment benefit|utility)"
    ),
    "requires":("tnsre_bci_utility_complete",),
    "reason":"The completed U-k AUC experiment is algebraically non-identifiable for the tested constant score translation; practical utility needs a new threshold-sensitive endpoint."
  },
  {
    "id":"discriminability_first_superiority",
    "patterns":(
      r"discriminability[- ]first.{0,100}(?:outperforms|superior|improves target|safer)",
      r"lexicographic.{0,100}(?:outperforms|superior|reduces harm)"
    ),
    "always_forbidden":True,
    "reason":"The post-S021 lexicographic selector is descriptively below pure source-macro selection in both TRUE PRE-DATA external cohorts."
  },
  {
    "id":"new_algorithm_superiority",
    "patterns":(r"\bCFCTS (?:outperforms|improves|is superior)\b",
                r"\btail[- ]signed (?:outperforms|improves|is superior)\b",
                r"\bTMS (?:outperforms|is superior to) nested\b"),
    "always_forbidden":True,
    "reason":"Prospective BNCI/EEGMMIDB tests refuted tail-signed and CFCTS as primary superiority claims."
  }
)

REQUIRED_EVIDENCE_KEYS=(
 "historical_table_integrity_resolved",
 "s021_s040_full_domain_complete",
 "three_cohort_full_domain_confirmed",
 "mechanism_counterfactuals_replicated",
 "prospective_representation_interaction_confirmed",
 "strong_same_budget_baselines_complete",
 "external_benchmark_panel_complete",
 "tnsre_bci_utility_complete",
 "cross_session_or_device_complete",
 "nested_source_fctr_audit_complete",
 "fidelity_regret_frontier_multicohort_complete",
 "fctr_prospective_target_confirmed",
 "fctr_sensitivity_audit_complete",
 "validity_frontier_uncertainty_complete",
 "cross_cohort_lodo_stable",
 "prospective_source_discriminability_target_confirmed",
 "true_predata_frontier_one_cohort_complete",
 "true_predata_frontier_two_cohort_complete",
 "mean_tail_fidelity_predata_one_cohort_complete",
 "mean_tail_fidelity_predata_two_cohort_safe",
 "true_predata_external_directionally_consistent_two_cohort",
 "true_predata_external_primary_family_significant_two_cohort",
 "corrected_material_fragility_frontier_two_cohort_complete",
 "operator_path_geometry_implemented"
)

def audit_text(text,evidence,source="<memory>"):
    findings=[]
    for rule in RULES:
        blocked=bool(rule.get("always_forbidden",False))
        missing=[]
        if not blocked:
            missing=[k for k in rule.get("requires",())
                     if evidence.get(k) is not True]
            blocked=bool(missing)
        if not blocked:
            continue
        for pattern in rule["patterns"]:
            for m in re.finditer(pattern,text,flags=re.I):
                line=text.count("\n",0,m.start())+1
                findings.append({
                  "rule":rule["id"],"source":source,"line":line,
                  "match":m.group(0),"missing_evidence":missing,
                  "reason":rule["reason"]})
    return findings

def run(evidence_path,files,out=None):
    evidence=json.loads(Path(evidence_path).read_text())
    absent=[k for k in REQUIRED_EVIDENCE_KEYS if k not in evidence]
    if absent:
        raise ValueError("Evidence ledger missing explicit keys: "+", ".join(absent))
    findings=[]
    for file in files:
        p=Path(file)
        findings.extend(audit_text(p.read_text(errors="replace"),evidence,str(p)))
    report={
      "submission_claims_allowed":len(findings)==0,
      "blocked_findings":findings,
      "evidence":{k:bool(evidence[k]) for k in REQUIRED_EVIDENCE_KEYS},
      "status":"PASS" if not findings else "SCIENCE_HOLD_CLAIM_OVERREACH"}
    if out:Path(out).write_text(json.dumps(report,indent=2,sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",required=True)
    p.add_argument("--out")
    p.add_argument("files",nargs="+")
    a=p.parse_args()
    r=run(a.evidence,a.files,a.out)
    print(json.dumps(r,indent=2,sort_keys=True))
    raise SystemExit(0 if r["submission_claims_allowed"] else 2)

if __name__=="__main__":main()
