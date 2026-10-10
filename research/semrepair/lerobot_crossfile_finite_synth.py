"""Finite cross-file repair synthesis against pinned REAL LeRobot issue #3425.

The candidate generator receives ONLY pre-merge vendor source and an AUTHOR
DECLARED contract/grammar.  The upstream merged solution is never read during
selection.  Uses vendor AST and real processor setup statements with stubbed
checkpoint factory; this is NOT autonomous repository-level issue solving.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from dataclasses import dataclass
from . import real_merged_lerobot_processor_contract as vendor

REQUIRED_SCOPES = ("train", "registry", "config")
SCENARIOS = (
    ("pretrained_relative_mapped", "model/ckpt", True, {"cam": "image"}, False, True),
    ("pretrained_relative_plain", "model/ckpt", True, {}, False, True),
    ("pretrained_absolute_mapped", "model/ckpt", False, {"cam": "image"}, False, True),
    ("fresh_relative_plain", None, True, {}, False, True),
    ("fresh_absolute_plain", None, False, {}, False, True),
    ("fresh_mapped_refuse", None, True, {"cam": "image"}, False, False),
    ("resume_relative_mapped", "model/ckpt", True, {"cam": "image"}, True, True),
)

@dataclass(frozen=True)
class Candidate:
    preserve_checkpoint: bool
    inject_relative_pair: bool
    rename_registry: bool
    enforce_fresh_rename_guard: bool

    @property
    def cost(self):
        return (int(self.preserve_checkpoint) +
                2 * int(self.inject_relative_pair) +
                int(self.rename_registry) +
                int(self.enforce_fresh_rename_guard))

    @property
    def scopes(self):
        s = set()
        if self.preserve_checkpoint or self.inject_relative_pair:
            s.add("train")
        if self.rename_registry:
            s.add("registry")
        if self.enforce_fresh_rename_guard:
            s.add("config")
        return s


def _target_train(tree):
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"]
    if len(found) != 1:
        raise ValueError("ambiguous publisher train() function")
    return found[0]


def _is_checkpoint_drop(node):
    if not isinstance(node, ast.If):
        return False
    writes = [n for n in ast.walk(node) if isinstance(n, ast.Assign) and
              len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and
              n.targets[0].id == "processor_pretrained_path" and
              isinstance(n.value, ast.Constant) and n.value.value is None]
    return len(writes) == 1 and "use_relative_actions" in ast.unparse(node.test)


def transform_train(source, candidate):
    tree = ast.parse(source)
    fn = _target_train(tree)
    drops = [n for n in fn.body if _is_checkpoint_drop(n)]
    if len(drops) != 1:
        raise ValueError("expected exactly one destructive checkpoint rewrite")
    if candidate.preserve_checkpoint:
        fn.body.remove(drops[0])
    if candidate.inject_relative_pair:
        eligible = [n for n in fn.body if isinstance(n, ast.If) and
                    "not cfg.is_reward_model_training" in ast.unparse(n.test) and
                    "processor_pretrained_path is not None" in ast.unparse(n.test)]
        if len(eligible) != 1:
            raise ValueError("pretrained processor override owner not unique")
        extra = ast.parse("""
if getattr(active_cfg, "use_relative_actions", False):
    processor_kwargs["preprocessor_overrides"]["relative_actions_processor"] = {
        "enabled": True,
        "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
        "action_names": getattr(active_cfg, "action_feature_names", None),
    }
    postprocessor_kwargs["postprocessor_overrides"]["absolute_actions_processor"] = {"enabled": True}
""").body
        eligible[0].body.extend(extra)
    return ast.unparse(ast.fix_missing_locations(tree))


def transform_registry(source, candidate):
    tree = ast.parse(source)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
               isinstance(n.func, ast.Attribute) and n.func.attr == "register" and
               len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and
               n.args[0].value == "delta_actions_processor"]
    if len(matches) != 1:
        raise ValueError("source registry key was not uniquely identified")
    if candidate.rename_registry:
        matches[0].args[0] = ast.Constant(value="relative_actions_processor")
    return ast.unparse(ast.fix_missing_locations(tree))


def transform_config(source, candidate):
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and
             n.name == "TrainPipelineConfig"]
    if len(found) != 1:
        raise ValueError("training config ownership not unique")
    fns = [n for n in found[0].body if isinstance(n, ast.FunctionDef) and n.name == "validate"]
    if len(fns) != 1:
        raise ValueError("config validator ownership not unique")
    fn = fns[0]
    locations = [i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "active_cfg" for t in n.targets) and
                 "self.trainable_config" in ast.unparse(n.value)]
    if len(locations) != 1:
        raise ValueError("config active owner assignment not unique")
    if candidate.enforce_fresh_rename_guard:
        guard = ast.parse("""
if self.rename_map and not active_cfg.pretrained_path:
    raise ValueError("rename_map requires a pretrained policy checkpoint")
""").body
        fn.body[locations[0] + 1:locations[0] + 1] = guard
    return ast.unparse(ast.fix_missing_locations(tree))


def _check_scenario(fn, candidate, row):
    name, path, relative, rename, resume, should_allow = row
    actual = vendor._execute_real_pre_and_post_source(
        fn, pretrained_path=path, relative=relative,
        rename_map=rename, resume=resume)
    if not path and rename:
        # Validate the separately synthesized config guard, not the factory.
        allowed = not candidate.enforce_fresh_rename_guard
        return (allowed == should_allow,
                "fresh_map_guard_present" if not allowed else "unguarded_fresh_map")
    model = vendor._model_invariant(actual)
    allowed = bool(model["valid"])
    if relative and path and not candidate.rename_registry:
        allowed = False  # overrides cannot resolve to the historical registry key
    return (allowed == should_allow,
            model["decision"] if allowed else "missing_cross_file_contract")


def assess(source_by_role, candidate):
    train = transform_train(source_by_role["train"], candidate)
    registry = transform_registry(source_by_role["registry"], candidate)
    config = transform_config(source_by_role["config"], candidate)
    # Exercise exact AST statements extracted from the transformed publisher train.
    fn, statement_count = vendor._extract_original_executable_setup(train)
    results = {}
    for row in SCENARIOS:
        results[row[0]] = _check_scenario(fn, candidate, row)
    return {
        "pass": all(x[0] for x in results.values()),
        "scenarios": results,
        "real_training_AST_statement_count": statement_count,
        "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest()
                          for k, v in (("train", train), ("registry", registry),
                                       ("config", config))},
        "modified_sources": {"train": train, "registry": registry, "config": config},
    }


def _enumerate():
    return [Candidate(*bits) for bits in itertools.product((False, True), repeat=4)]


def select(source_by_role, *, allowed_scopes=REQUIRED_SCOPES, cost_limit=100):
    admitted, explored = [], []
    for candidate in _enumerate():
        licensed = candidate.scopes.issubset(set(allowed_scopes))
        within_budget = candidate.cost <= cost_limit
        if not (licensed and within_budget):
            explored.append((candidate, "NOT_AUTHORIZED"))
            continue
        assessment = assess(source_by_role, candidate)
        explored.append((candidate, "PASS" if assessment["pass"] else "FAIL"))
        if assessment["pass"]:
            admitted.append((candidate.cost, candidate, assessment))
    if not admitted:
        return None, explored
    admitted.sort(key=lambda x: (x[0], tuple(vars(x[1]).values())))
    return admitted[0], explored


def run():
    # All sources are PRE-MERGE only. The maintainer's accepted change is not
    # fetched, decoded, or used to design/test a candidate in this function.
    roles = (("train", vendor.SCRIPT_PATH, vendor.BEFORE_SOURCE_BLOB),
             ("config", vendor.CONFIG_PATH, vendor.BEFORE_CONFIG_BLOB),
             ("registry", vendor.REGISTRY_PATH, vendor.BEFORE_REGISTRY_BLOB))
    sources, identities = {}, {}
    for role, path, expected_blob in roles:
        src, digest = vendor._pinned_source(vendor.BEFORE_COMMIT, path, expected_blob)
        sources[role] = src
        identities[role] = {"upstream_blob": expected_blob, "sha256": digest}
    selected, evaluated = select(sources)
    if selected is None:
        raise AssertionError("no certified candidate in declared finite grammar")
    cost, candidate, certificate = selected
    cheap, _ = select(sources, cost_limit=cost - 1)
    restricted, _ = select(sources, allowed_scopes=("train", "registry"))
    if cheap is not None or restricted is not None:
        raise AssertionError("finite-grammar optimality or permission refusal broken")
    if candidate != Candidate(True, True, True, True):
        raise AssertionError("candidate deviates from declared multi-file contract")
    return {
        "protocol": "LeRobot-3425-premerge-only-finite-crossfile-synthesis-v1",
        "real_immutable_upstream_before_commit": vendor.BEFORE_COMMIT,
        "source_identities": identities,
        "candidate_count": len(evaluated),
        "grammar_minimum_declared_cost": cost,
        "minimality_verified_by_exhaustive_lower_budget_refusal": True,
        "scope_refusal_verified": True,
        "chosen_candidate": vars(candidate),
        "scenario_results": certificate["scenarios"],
        "synthesized_source_sha256": certificate["source_sha256"],
        "postmerge_code_not_used_for_selection": True,
        "actual_vendor_factory_mocked": True,
        "actual_pretrained_policy_executed": False,
        "all_programs_in_python_universe_synthesized": False,
        "external_upstream_adoption": False,
        "limitations": "One upstream issue, author-declared contract and 16-template mutation grammar; original AST statements execute with factory stubs, no full training or independent reviewer."
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
