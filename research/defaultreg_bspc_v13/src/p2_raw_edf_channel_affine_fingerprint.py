#!/usr/bin/env python3
"""Streaming, exact-moment raw EDF digital-channel affine-fingerprint audit.

Research purpose: test whether high similarity AFTER filtering is visible
BEFORE filtering as an integer-sample affine relationship. All samples are
read in EDF record/channel order; none are exported. This is a retrospective
technical comparison of the Liu2024 PUBLIC release, not a patient-identity
test, independent clinical cohort, or physical calibration check.

The sums of products are aggregated as arbitrary-precision Python integers.
The correlation and affine normalized residual follow from exact centered
integer moments (conversion to float occurs only for the final ratios).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
import numpy as np
from p2_edf_digital_payload_audit import header

PAIRS = ((4, 5), (47, 49), (11, 15), (2, 3), (6, 7))
DISCOVERY_PAIRS = {(4, 5), (47, 49), (11, 15)}

def raw_pair(a_path: Path, b_path: Path) -> dict:
    a_path, b_path = Path(a_path), Path(b_path)
    A, B = header(a_path), header(b_path)
    if any(A[k] != B[k] for k in ("n", "names", "samples", "records")):
        raise ValueError("EDF layout/channel/sample alignment differs: fail closed")
    n = A["n"]
    if A["records"] < 1:
        raise ValueError("no raw EDF records")
    # n samples, sum a, sum b, sum a*a, sum b*b, sum a*b (integer exact).
    moments = [[0] * 6 for _ in range(n)]
    is_exact = [True] * n
    with a_path.open("rb") as fa, b_path.open("rb") as fb:
        fa.seek(A["header"])
        fb.seek(B["header"])
        for _ in range(A["records"]):
            for j, count in enumerate(A["samples"]):
                size = 2 * count
                xa, xb = fa.read(size), fb.read(size)
                if len(xa) != size or len(xb) != size:
                    raise EOFError("truncated EDF digital samples")
                if xa != xb:
                    is_exact[j] = False
                aa = np.frombuffer(xa, dtype="<i2")
                bb = np.frombuffer(xb, dtype="<i2")
                # Bound dot-product size for int64 safety even on unusual EDFs.
                for start in range(0, count, 100000):
                    u = aa[start:start + 100000].astype(np.int64)
                    v = bb[start:start + 100000].astype(np.int64)
                    m = moments[j]
                    m[0] += len(u)
                    m[1] += int(u.sum())
                    m[2] += int(v.sum())
                    m[3] += int(np.dot(u, u))
                    m[4] += int(np.dot(v, v))
                    m[5] += int(np.dot(u, v))
    details = []
    for j, (N, sx, sy, sxx, syy, sxy) in enumerate(moments):
        if N < 2:
            raise ValueError("insufficient EDF samples")
        vx = N * sxx - sx * sx
        vy = N * syy - sy * sy
        cov = N * sxy - sx * sy
        if vx <= 0 or vy <= 0:
            r = None
            residual = None
        else:
            den_sq = vx * vy
            # Identity: var(X)*var(Y)-cov(X,Y)^2 >= 0, all exact integers.
            res_num = den_sq - cov * cov
            if res_num < 0:
                raise AssertionError("impossible negative exact Cauchy-Schwarz slack")
            r = float(max(-1, min(1, cov / math.sqrt(den_sq))))
            residual = float(math.sqrt(res_num / den_sq))
        details.append({"channel_index": j, "exact_raw_int16_samples": is_exact[j],
                        "n_samples": N, "pearson": r,
                        "relative_affine_fit_residual": residual,
                        "constant_channel": vx == 0 or vy == 0})
    finite = [q for q in details if q["pearson"] is not None]
    corr = sorted(q["pearson"] for q in finite)
    result = {
        "raw_edf_channels": n,
        "n_exact_digital_channels": sum(is_exact),
        "n_nonconstant_channels": len(finite),
        "n_raw_channel_pearson_ge_0p9999": sum(q["pearson"] >= .9999 for q in finite),
        "n_raw_channel_affine_residual_lt_0p001": sum(q["relative_affine_fit_residual"] < .001 for q in finite),
        "median_raw_channel_pearson": float(np.median(corr)) if corr else None,
        "min_raw_channel_pearson": corr[0] if corr else None,
        "max_raw_channel_affine_residual": max((q["relative_affine_fit_residual"] for q in finite), default=None),
        "nonmatching_raw_channel_indices": [j for j, value in enumerate(is_exact) if not value],
        "per_channel_aggregated_metrics": details,
        "aligned_raw_integer_content_only": True,
        "not_physical_units_patient_identity_or_acquisition_origin": True,
    }
    return result

def _make_small_edf(path, channels):
    """Construct fixed-schema 16-bit EDF solely for exact synthetic tests."""
    n = len(channels)
    records, samples = 2, len(channels[0]) // 2
    assert all(len(x) == records * samples for x in channels)
    h = bytearray(b" " * 256)
    h[184:192] = f"{256 + 256*n:<8}".encode()
    h[236:244] = f"{records:<8}".encode()
    h[252:256] = f"{n:<4}".encode()
    fields = [
        (16, [f"ch{j}" for j in range(n)]), (80, [""] * n),
        (8, ["uV"] * n), (8, ["-100"] * n), (8, ["100"] * n),
        (8, ["-32768"] * n), (8, ["32767"] * n), (80, [""] * n),
        (8, [str(samples)] * n), (32, [""] * n)
    ]
    with Path(path).open("wb") as f:
        f.write(h)
        for width, values in fields:
            for value in values:
                f.write(f"{value:<{width}}".encode())
        for rec in range(records):
            for v in channels:
                f.write(np.asarray(v[rec*samples:(rec+1)*samples], dtype="<i2").tobytes())

def self_test():
    with tempfile.TemporaryDirectory() as d:
        p = [Path(d) / f"t{i}.edf" for i in range(3)]
        orig = [[1, 2, 3, 4, 5, 6], [-2, 0, 2, -2, 0, 2]]
        affine = [[2 * x + 7 for x in orig[0]], [x + 5 for x in orig[1]]]
        other = [[1, 3, 2, 6, 4, 5], [-2, 2, 0, 0, -2, 2]]
        for file, chans in zip(p, (orig, affine, other)):
            _make_small_edf(file, chans)
        identical = raw_pair(p[0], p[0])
        changed = raw_pair(p[0], p[1])
        neg = raw_pair(p[0], p[2])
        assert identical["n_exact_digital_channels"] == 2
        assert changed["n_exact_digital_channels"] == 0
        assert changed["n_raw_channel_pearson_ge_0p9999"] == 2
        assert changed["n_raw_channel_affine_residual_lt_0p001"] == 2
        assert neg["n_raw_channel_pearson_ge_0p9999"] == 0
        assert not any(x["constant_channel"] for x in changed["per_channel_aggregated_metrics"])
    print("P2_RAW_DIGITAL_EXACT_AFFINE_NEGATIVE_SYNTHETIC_TEST_PASS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        ap.error("--out required for real public EDF audit")
    from moabb.datasets import Liu2024
    ds = Liu2024()
    rows = []
    for a, b in PAIRS:
        result = raw_pair(ds.data_path(a)[0], ds.data_path(b)[0])
        rows.append({
            "public_released_ids": [a, b],
            "selected_after_1225_processed_pair_screen": (a, b) in DISCOVERY_PAIRS,
            **result
        })
        print("P2_RAW_DIGITAL_AFFINE_PAIR_COMPLETE", a, b,
              result["n_exact_digital_channels"],
              result["n_raw_channel_pearson_ge_0p9999"], flush=True)
    report = {
        "status": "EXECUTED_RETROSPECTIVE_PUBLIC_RAW_EDF_AFFINE_DIAGNOSTIC",
        "input": "Liu2024 public EDF original signed little-endian int16 digital samples",
        "scope": "Three exposed candidate pairs plus two nominal negative-control pairs",
        "uses_second_acquisition": False,
        "reanalysis_after_primary_finding": True,
        "does_not_examine_or_infer_patient_identity": True,
        "no_classifier_target_results_read": True,
        "does_not_parse_physical_unit_calibration": True,
        "aligned_raw_channel_order_and_sample_counts_required": True,
        "pairs": rows
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.write_text(body, encoding="utf-8")
    print("P2_RAW_AFFINE_REPORT_SHA256", hashlib.sha256(body.encode()).hexdigest(), flush=True)

if __name__ == "__main__":
    main()
