#!/usr/bin/env python3
"""Freeze a post-resolution prospective adjudication without upgrading independence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEN = ROOT / "seen.json"
PRED = ROOT / "predictions"
RES = ROOT / "resolution"
ADJ = ROOT / "adjudications"

PRED_HASH_FIELDS = (
    "case_id",
    "source_surface_sha256",
    "contract_id",
    "independence_class",
    "predicted_relation",
    "predicted_mechanism",
    "owner_boundary",
    "discriminating_experiment",
    "falsifier",
    "repair_boundary",
    "prediction_frozen_at",
)
ADJ_HASH_FIELDS = (
    "case_id",
    "source_surface_sha256",
    "prediction_sha256",
    "resolution_evidence_sha256",
    "predicted_independence_class",
    "final_independence_class",
    "mechanism_match",
    "credit_decision",
    "reason",
    "adjudicated_at",
)
RANK = {"EXCLUDED": 0, "I0": 1, "I1": 2, "I2": 3}
MECHANISM = {"supported", "falsified", "unresolved"}
CREDIT = {"no_credit", "prospective_i2_credit"}


def parse(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


def fmt(value: dt.datetime) -> str:
    return (
        value.astimezone(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def digest_fields(doc: dict, fields: tuple[str, ...]) -> str:
    payload = {key: doc[key] for key in fields}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def resolution_digest(doc: dict) -> str:
    payload = {
        key: doc[key]
        for key in ("case_id", "prediction_sha256", "issue", "comments", "timeline")
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def safe(case_id: str) -> str:
    return case_id.replace("/", "__").replace("#", "__")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    parser.add_argument("--at")
    args = parser.parse_args()

    draft = load(args.draft)
    required = (
        "case_id",
        "final_independence_class",
        "mechanism_match",
        "credit_decision",
        "reason",
    )
    missing = [key for key in required if not draft.get(key)]
    if missing:
        raise ValueError(f"missing adjudication fields: {missing}")

    final_class = draft["final_independence_class"]
    mechanism_match = draft["mechanism_match"]
    credit_decision = draft["credit_decision"]
    if final_class not in RANK:
        raise ValueError(f"unknown final independence class: {final_class}")
    if mechanism_match not in MECHANISM:
        raise ValueError(f"unknown mechanism_match: {mechanism_match}")
    if credit_decision not in CREDIT:
        raise ValueError(f"unknown credit_decision: {credit_decision}")

    state = load(SEEN)
    entries = {
        item.get("id"): item
        for item in state.get("seen", [])
        if isinstance(item, dict)
    }
    case_id = draft["case_id"]
    if case_id not in entries:
        raise ValueError(f"{case_id}: no public first-seen record")
    entry = entries[case_id]
    if entry.get("status") != "prediction-frozen-awaiting-resolution":
        raise ValueError(f"{case_id}: not awaiting post-resolution adjudication")

    pred_path = PRED / f"{safe(case_id)}.json"
    res_path = RES / f"{safe(case_id)}.json"
    if not pred_path.exists() or not res_path.exists():
        raise ValueError(f"{case_id}: prediction/resolution evidence is incomplete")

    prediction = load(pred_path)
    resolution = load(res_path)
    expected_pred = digest_fields(prediction, PRED_HASH_FIELDS)
    if prediction.get("prediction_sha256") != expected_pred:
        raise ValueError(f"{case_id}: prediction digest mismatch")
    if entry.get("prediction_sha256") != expected_pred:
        raise ValueError(f"{case_id}: seen/prediction digest mismatch")
    if prediction.get("source_surface_sha256") != entry.get("surface_sha256"):
        raise ValueError(f"{case_id}: prediction/first-seen surface mismatch")

    expected_res = resolution_digest(resolution)
    if resolution.get("resolution_evidence_sha256") != expected_res:
        raise ValueError(f"{case_id}: resolution evidence digest mismatch")
    if resolution.get("prediction_sha256") != expected_pred:
        raise ValueError(f"{case_id}: resolution is bound to a different prediction")

    predicted_class = prediction["independence_class"]
    if predicted_class not in RANK:
        raise ValueError(f"{case_id}: unknown predicted independence class")
    if RANK[final_class] > RANK[predicted_class]:
        raise ValueError(
            f"{case_id}: post-hoc independence upgrade "
            f"{predicted_class}->{final_class} is forbidden"
        )

    qualifies_i2 = (
        predicted_class == "I2"
        and final_class == "I2"
        and bool(prediction.get("prediction_within_sla"))
        and mechanism_match == "supported"
    )
    if credit_decision == "prospective_i2_credit" and not qualifies_i2:
        raise ValueError(
            f"{case_id}: prospective I2 credit requires an in-SLA I2 prediction, "
            "final I2 adjudication, and supported mechanism match"
        )
    if credit_decision == "no_credit" and qualifies_i2:
        # A reviewer may still choose to withhold credit, so this is allowed.
        pass

    now = parse(args.at) if args.at else dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    )
    record = {
        "schema_version": 1,
        "case_id": case_id,
        "source_surface_sha256": entry["surface_sha256"],
        "prediction_sha256": expected_pred,
        "resolution_evidence_sha256": expected_res,
        "predicted_independence_class": predicted_class,
        "final_independence_class": final_class,
        "mechanism_match": mechanism_match,
        "credit_decision": credit_decision,
        "reason": draft["reason"],
        "adjudicated_at": fmt(now),
        "claim_boundary": (
            "Final adjudication is bound to one immutable first-seen surface, "
            "prediction digest, and resolution snapshot. Independence may be "
            "preserved or downgraded after resolution, never upgraded."
        ),
    }
    record["adjudication_sha256"] = digest_fields(record, ADJ_HASH_FIELDS)

    ADJ.mkdir(exist_ok=True)
    out = ADJ / f"{safe(case_id)}.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for item in state["seen"]:
        if item.get("id") == case_id:
            suffix = (
                "i2_credit"
                if credit_decision == "prospective_i2_credit"
                else "no_credit"
            )
            item["status"] = f"adjudicated_{final_class.lower()}_{suffix}"
            item["final_independence_class"] = final_class
            item["mechanism_match"] = mechanism_match
            item["credit_decision"] = credit_decision
            item["adjudication_source"] = str(out.relative_to(ROOT))
            item["adjudication_sha256"] = record["adjudication_sha256"]
            item["resolution_evidence_sha256"] = expected_res
            break
    SEEN.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(out)
    print(record["adjudication_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
