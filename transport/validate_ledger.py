#!/usr/bin/env python3
"""Fail-closed integrity audit for the public prospective evidence ledger."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEN = ROOT / "seen.json"
PRED = ROOT / "predictions"
RES = ROOT / "resolution"
DISP = ROOT / "dispositions"
EXP = ROOT / "expirations"
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
DISP_HASH_FIELDS = (
    "case_id",
    "source_surface_sha256",
    "first_frozen_at",
    "prediction_deadline_at",
    "disposition_frozen_at",
    "independence_class",
    "disposition",
    "reason",
)
EXP_HASH_FIELDS = (
    "case_id",
    "source_surface_sha256",
    "first_frozen_at",
    "prediction_deadline_at",
    "expiry_recorded_at",
    "reason",
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


def parse(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


def digest(doc, fields):
    payload = {key: doc[key] for key in fields}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def resolution_digest(doc):
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


def safe(value):
    return value.replace("/", "__").replace("#", "__")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_surface(entry):
    src = ROOT / entry["source"]
    doc = load(src)
    matches = [
        item
        for item in doc.get("new_candidates", [])
        if item.get("id") == entry["id"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{entry['id']}: source snapshot cardinality={len(matches)}"
        )
    return matches[0]


def validate_prediction(case, entry, pred_path):
    doc = load(pred_path)
    if (
        doc["case_id"] != case
        or doc["source_surface_sha256"] != entry["surface_sha256"]
    ):
        raise ValueError(f"{case}: prediction source identity mismatch")
    if doc["prediction_sha256"] != digest(doc, PRED_HASH_FIELDS):
        raise ValueError(f"{case}: prediction digest mismatch")
    if entry.get("prediction_sha256") != doc["prediction_sha256"]:
        raise ValueError(f"{case}: seen/prediction digest mismatch")
    if bool(doc["prediction_within_sla"]) != (
        parse(doc["prediction_frozen_at"]) <= parse(doc["prediction_deadline_at"])
    ):
        raise ValueError(f"{case}: prediction SLA flag mismatch")
    if not doc["prediction_within_sla"]:
        raise ValueError(
            f"{case}: late mechanism prediction cannot count as prospective evidence"
        )
    if doc["independence_class"] not in RANK:
        raise ValueError(f"{case}: unknown prediction independence class")
    return doc


def main():
    state = load(SEEN)
    seen = state.get("seen", [])
    ids = [item.get("id") for item in seen]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate seen case id")

    counts = {
        "pending": 0,
        "prediction": 0,
        "excluded": 0,
        "expired": 0,
        "adjudicated": 0,
        "prospective_i2_credit": 0,
    }

    for entry in seen:
        case = entry["id"]
        surface = source_surface(entry)
        if surface["surface_sha256"] != entry["surface_sha256"]:
            raise ValueError(f"{case}: seen/source surface hash mismatch")
        if surface["created_at"] != entry["created_at"]:
            raise ValueError(f"{case}: seen/source created_at mismatch")

        pred = PRED / f"{safe(case)}.json"
        res = RES / f"{safe(case)}.json"
        disp = DISP / f"{safe(case)}.json"
        exp = EXP / f"{safe(case)}.json"
        adj = ADJ / f"{safe(case)}.json"
        status = entry["status"]

        if status == "candidate-frozen-awaiting-prediction":
            if pred.exists() or disp.exists() or exp.exists() or adj.exists():
                raise ValueError(
                    f"{case}: pending case already has decision/terminal artifact"
                )
            if parse(entry["first_frozen_at"]) > parse(
                entry["prediction_deadline_at"]
            ):
                raise ValueError(f"{case}: impossible first-seen/deadline order")
            counts["pending"] += 1
            continue

        if status == "prediction-frozen-awaiting-resolution":
            if (
                not pred.exists()
                or disp.exists()
                or exp.exists()
                or adj.exists()
            ):
                raise ValueError(f"{case}: prediction state/artifact mismatch")
            pdoc = validate_prediction(case, entry, pred)
            if res.exists():
                rdoc = load(res)
                if rdoc.get("prediction_sha256") != pdoc["prediction_sha256"]:
                    raise ValueError(f"{case}: resolution/prediction mismatch")
                if (
                    rdoc.get("resolution_evidence_sha256")
                    != resolution_digest(rdoc)
                ):
                    raise ValueError(f"{case}: resolution evidence digest mismatch")
            counts["prediction"] += 1
            continue

        if status == "deadline_missed_unadjudicated":
            if pred.exists() or disp.exists() or adj.exists() or not exp.exists():
                raise ValueError(f"{case}: expired state/artifact mismatch")
            doc = load(exp)
            if (
                doc["case_id"] != case
                or doc["source_surface_sha256"] != entry["surface_sha256"]
            ):
                raise ValueError(f"{case}: expired source identity mismatch")
            if doc["expiry_sha256"] != digest(doc, EXP_HASH_FIELDS):
                raise ValueError(f"{case}: expiration digest mismatch")
            if entry.get("expiry_sha256") != doc["expiry_sha256"]:
                raise ValueError(f"{case}: seen/expiration digest mismatch")
            if entry.get("expiry_source") != str(exp.relative_to(ROOT)):
                raise ValueError(f"{case}: expiration source path mismatch")
            if entry.get("expiry_recorded_at") != doc["expiry_recorded_at"]:
                raise ValueError(f"{case}: expiration timestamp mismatch")
            if parse(doc["expiry_recorded_at"]) <= parse(
                doc["prediction_deadline_at"]
            ):
                raise ValueError(f"{case}: expiration was recorded before deadline")
            if (
                doc.get("independence_class") != "UNADJUDICATED"
                or doc.get("primary_prospective_I2_eligible") is not False
            ):
                raise ValueError(
                    f"{case}: expired case improperly claims adjudication or I2 eligibility"
                )
            counts["expired"] += 1
            continue

        if status.startswith("adjudicated_"):
            if (
                not pred.exists()
                or not res.exists()
                or not adj.exists()
                or disp.exists()
                or exp.exists()
            ):
                raise ValueError(f"{case}: adjudicated state/artifact mismatch")

            pdoc = validate_prediction(case, entry, pred)
            rdoc = load(res)
            if rdoc.get("prediction_sha256") != pdoc["prediction_sha256"]:
                raise ValueError(f"{case}: resolution/prediction mismatch")
            expected_res = resolution_digest(rdoc)
            if rdoc.get("resolution_evidence_sha256") != expected_res:
                raise ValueError(f"{case}: resolution evidence digest mismatch")

            adoc = load(adj)
            if adoc.get("case_id") != case:
                raise ValueError(f"{case}: adjudication case mismatch")
            if adoc.get("source_surface_sha256") != entry["surface_sha256"]:
                raise ValueError(f"{case}: adjudication source surface mismatch")
            if adoc.get("prediction_sha256") != pdoc["prediction_sha256"]:
                raise ValueError(f"{case}: adjudication prediction mismatch")
            if adoc.get("resolution_evidence_sha256") != expected_res:
                raise ValueError(f"{case}: adjudication resolution mismatch")
            if adoc.get("adjudication_sha256") != digest(adoc, ADJ_HASH_FIELDS):
                raise ValueError(f"{case}: adjudication digest mismatch")

            predicted = adoc["predicted_independence_class"]
            final = adoc["final_independence_class"]
            mechanism = adoc["mechanism_match"]
            credit = adoc["credit_decision"]
            if predicted != pdoc["independence_class"]:
                raise ValueError(f"{case}: adjudication predicted class mismatch")
            if predicted not in RANK or final not in RANK:
                raise ValueError(f"{case}: unknown adjudication independence class")
            if RANK[final] > RANK[predicted]:
                raise ValueError(
                    f"{case}: forbidden post-hoc independence upgrade "
                    f"{predicted}->{final}"
                )
            if mechanism not in MECHANISM:
                raise ValueError(f"{case}: invalid mechanism_match")
            if credit not in CREDIT:
                raise ValueError(f"{case}: invalid credit_decision")

            qualifies_i2 = (
                predicted == "I2"
                and final == "I2"
                and bool(pdoc["prediction_within_sla"])
                and mechanism == "supported"
            )
            if credit == "prospective_i2_credit" and not qualifies_i2:
                raise ValueError(f"{case}: invalid prospective I2 credit")

            expected_status = (
                f"adjudicated_{final.lower()}_"
                + (
                    "i2_credit"
                    if credit == "prospective_i2_credit"
                    else "no_credit"
                )
            )
            if status != expected_status:
                raise ValueError(
                    f"{case}: seen/adjudication terminal status mismatch"
                )
            if entry.get("final_independence_class") != final:
                raise ValueError(f"{case}: final class not mirrored in seen")
            if entry.get("mechanism_match") != mechanism:
                raise ValueError(f"{case}: mechanism result not mirrored in seen")
            if entry.get("credit_decision") != credit:
                raise ValueError(f"{case}: credit decision not mirrored in seen")
            if entry.get("adjudication_sha256") != adoc["adjudication_sha256"]:
                raise ValueError(f"{case}: seen/adjudication digest mismatch")
            if entry.get("resolution_evidence_sha256") != expected_res:
                raise ValueError(f"{case}: seen/resolution digest mismatch")
            if entry.get("adjudication_source") != str(adj.relative_to(ROOT)):
                raise ValueError(f"{case}: adjudication path mismatch")

            counts["adjudicated"] += 1
            if credit == "prospective_i2_credit":
                counts["prospective_i2_credit"] += 1
            continue

        if status.startswith("excluded_"):
            if pred.exists() or exp.exists() or adj.exists() or not disp.exists():
                raise ValueError(f"{case}: exclusion state/artifact mismatch")
            doc = load(disp)
            if (
                doc["case_id"] != case
                or doc["source_surface_sha256"] != entry["surface_sha256"]
            ):
                raise ValueError(f"{case}: disposition source identity mismatch")
            if doc["disposition_sha256"] != digest(doc, DISP_HASH_FIELDS):
                raise ValueError(f"{case}: disposition digest mismatch")
            if entry.get("disposition_sha256") != doc["disposition_sha256"]:
                raise ValueError(f"{case}: seen/disposition digest mismatch")
            if entry.get("disposition_source") != str(disp.relative_to(ROOT)):
                raise ValueError(f"{case}: disposition path mismatch")
            if bool(doc["disposition_within_sla"]) != (
                parse(doc["disposition_frozen_at"])
                <= parse(doc["prediction_deadline_at"])
            ):
                raise ValueError(f"{case}: disposition SLA flag mismatch")
            counts["excluded"] += 1
            continue

        raise ValueError(f"{case}: unknown status {status!r}")

    print(json.dumps({"status": "pass", "cases": len(seen), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
