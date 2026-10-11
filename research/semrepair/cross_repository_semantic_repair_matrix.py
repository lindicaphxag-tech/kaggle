"""Cross-repository natural merged software defect case-study evidence ledger.

TWO publisher repositories, FOUR INDEPENDENT merged upstream fixes:
* LeRobot #3711: policy/checkpoint preprocessing provenance + registry/guard.
* LeRobot #4656: stacked-state frame-zero anchor + cross-episode reset.
* LeRobot #4452: sync/RTC action-name and state-feature permutations.
* ManiSkill #689: in-place mutation of caller-owned action tensor view.

All before/after publisher source references are pinned to immutable Git
commits/blobs by each originating runner. Weak-test plausible programs
are judged by deliberately WEAKER task-specific tests, and full-contract
programs by their distinct cross-boundary source semantics. The 34
candidate program variants are all CURATED retrospective templates, NOT
34 independent developer-submitted real software bugs. We DO NOT pool
hundreds of correlated action tensor cases into a statistical sample of
natural software faults or assert vendor authorship/independent adoption.

This is a reproducible small software ENGINEERING CASE STUDY, not a
general automated APR success rate on Defects4J or SWE-bench.
"""
from __future__ import annotations
import json

from .verified_multifile_source_synthesis import synthesize_merged_issue_contract
from .real_merged_lerobot_4656_multifile_patch_search import (
    run_real_4656_joint_temporal_source_repairs
)
from .real_merged_lerobot_action_order_regression import (
    run_real_merged_compensating_order_issue
)
from .real_merged_maniskill_alias_repair import run_real_publisher_alias_repair


def _record(key,repo,merged,weak,total,valid,metric,description):
    if not 0<=valid<=weak<=total:
        raise AssertionError("original source bug benchmark candidate denominator inconsistent")
    return {
        "independent_upstream_issue":key,
        "publisher_repository":repo,
        "actual_external_MERGED_patch":merged,
        "author_generated_program_variants":total,
        "weak_local_test_plausible":weak,
        "complete_cross_boundary_contract_valid":valid,
        "weakly_accepted_but_contract_invalid":weak-valid,
        "actual_original_source_numerical_tests":metric,
        "contract_semantics":description,
        "publishers_did_NOT_adopt_SemRepair":True,
    }


def run_small_natural_source_repair_corpus():
    version=synthesize_merged_issue_contract()
    temporal=run_real_4656_joint_temporal_source_repairs()
    ordering=run_real_merged_compensating_order_issue()
    alias=run_real_publisher_alias_repair()

    r1=_record(
        "LeRobot #3425/#3711","huggingface/lerobot",
        version["publisher_merge"],
        version["one_source_contract_test_plausible_count"],
        version["catalog_program_count"],
        version["complete_source_contract_candidate_count"],
        version["per_program_actual_train_source_conditions"],
        "pretrained policy/processor provenance, relative+absolute overrides, registry, fresh-map guard",
    )
    r2=_record(
        "LeRobot #4656","huggingface/lerobot",
        temporal["genuinely_publisher_MERGED_PR"],
        temporal["single_roundtrip_test_plausible_program_count"],
        temporal["typed_two_file_candidate_program_count"],
        temporal["full_crosscomponent_semantics_valid_count"],
        temporal["independent_num_source_cases_per_candidate"],
        "frame-zero model conditioning and pre/postprocessor cache reset across eval batches",
    )
    # Two original externally published source implementations, not
    # an AI search of arbitrary sync/RTC patches.
    # Both BEFORE and AFTER PASS an already-aligned local named-feature test,
    # but BEFORE fails under reversed checkpoint-vs-hardware joint order.
    if not ordering["already_aligned_original_remains_correct"] or \
       ordering["baseline_original_mean_action_error"]<=20 or \
       ordering["source_corrected_max_action_error"]>1e-9:
        raise AssertionError("published #4452 weak-vs-cross-order ordering test not reproduced")
    r3=_record(
        "LeRobot #4452","huggingface/lerobot",
        ordering["real_independently_MERGED_upstream_PR"],
        2,2,1,ordering["distinct_source_regression_samples"],
        "same feature names/values but reversed action/state joint permutation in sync vs RTC",
    )
    alias_records=alias["aggregate"]
    weak=sum(bool(x["action_output_equal_original_publisher_merged_all_192"])
             for x in alias_records.values())
    valid=sum(bool(x["action_output_equal_original_publisher_merged_all_192"])
              and x["violated_caller_input_immutability_high_norm"]==0
              and x["violated_caller_input_immutability_low_norm"]==0
              for x in alias_records.values())
    r4=_record(
        "ManiSkill #689","mani-skill/ManiSkill",
        alias["upstream_publisher_merged_PR"],
        weak,len(alias_records),valid,alias["tested_input_configs"],
        "caller-owned action tensor immutability under in-place normalized slice writes",
    )
    rows=(r1,r2,r3,r4)
    return {
        "protocol":"SemRepair-four-actually-merged-source-bugs-two-publishers-cross-boundary-contract-matrix-v1",
        "independently_merged_natural_defects":len(rows),
        "publisher_repository_count":len(set(r["publisher_repository"] for r in rows)),
        "authored_typed_program_variants_total":sum(
            r["author_generated_program_variants"] for r in rows),
        "weak_test_plausible_programs_total":sum(
            r["weak_local_test_plausible"] for r in rows),
        "full_semantics_valid_programs_total":sum(
            r["complete_cross_boundary_contract_valid"] for r in rows),
        "weak_local_test_false_accepts_total":sum(
            r["weakly_accepted_but_contract_invalid"] for r in rows),
        "cross_issue_pooled_percentage_is_NOT_generalizable":True,
        "upstream_fixes_are_INDEPENDENT_prior_art_not_our_merges":True,
        "all_original_source_runners_are_AST_extracted_with_some_stubbed_interfaces":True,
        "real_robot_hardware_or_full_neural_VLA_trials":0,
        "rows":rows,
        "limitations":"all patch grammars explicitly prepared after true publisher fixes; simple local tests deliberately weak; 4 natural bugs are insufficient for state-of-the-art generic automated program repair",
    }


if __name__=="__main__":
    print(json.dumps(run_small_natural_source_repair_corpus(),
                     indent=2,sort_keys=True))
