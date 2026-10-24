#!/usr/bin/env python3
"""Independently audit committed derived records used in the PLOS paper.

This checks score/statistic arithmetic and record consistency. It does not
replace a fresh download-and-rerun of the original GEO expression matrices.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
REPO = ROOT.parent


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str):
    raise SystemExit("DERIVED-RECORD AUDIT FAILED: " + message)


def close(label, actual, expected, *, rel=5e-5, abs_tol=1e-12):
    if actual is None or expected is None:
        if actual is not expected:
            fail(f"{label}: got {actual!r}, expected {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol):
        fail(f"{label}: got {actual!r}, expected {expected!r}")


def succession(phases, order):
    """Mirror omictopo.cycles.circular_succession_score."""
    if len(order) < 3:
        return 0.0
    walked = [round(float(phases[i]), 6) for i in order]
    levels = sorted(set(walked))
    if len(levels) < 2:
        return 0.0
    lookup = {value: i for i, value in enumerate(levels)}
    group = [lookup[value] for value in walked]
    n_levels = len(levels)
    forward = backward = 0
    for i, current in enumerate(group):
        nxt = group[(i + 1) % len(group)]
        forward += int(nxt == current or nxt == (current + 1) % n_levels)
        backward += int(nxt == current or nxt == (current - 1) % n_levels)
    return max(forward, backward) / len(order)


def logcomb(n, k):
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact p-value from the upper hypergeometric tail."""
    total = a + b + c + d
    row1 = a + b
    col1 = a + c
    lo = max(0, row1 - (total - col1))
    hi = min(row1, col1)
    if a < lo or a > hi:
        fail(f"invalid 2x2 table {(a, b, c, d)}")
    denom = logcomb(total, row1)
    logs = [
        logcomb(col1, k) + logcomb(total - col1, row1 - k) - denom
        for k in range(a, hi + 1)
    ]
    peak = max(logs)
    return min(1.0, math.exp(peak) * sum(math.exp(x - peak) for x in logs))


def odds_ratio(a, b, c, d):
    if b * c == 0:
        if a * d == 0:
            return None
        return math.inf
    return (a * d) / (b * c)


def wald_ci(a, b, c, d):
    if min(a, b, c, d) <= 0:
        return None
    odds = odds_ratio(a, b, c, d)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return [math.exp(math.log(odds) - 1.96 * se),
            math.exp(math.log(odds) + 1.96 * se)]


def bh_adjust(pvalues):
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    q = [1.0] * n
    running = 1.0
    for rank0 in range(n - 1, -1, -1):
        i = order[rank0]
        running = min(running, pvalues[i] * n / (rank0 + 1))
        q[i] = min(1.0, running)
    return q


def table_counts(rec, keys):
    return tuple(int(rec[k]) for k in keys)


def check_enrichment(records, *, label, fields, family_key, expected_per_family):
    groups = {}
    for rec in records:
        a, b, c, d = table_counts(rec, fields["counts"])
        if a + b + c + d != int(rec[fields["n_mapped"]]):
            fail(f"{label}: mapped total mismatch in {rec}")
        expected_or = odds_ratio(a, b, c, d)
        stored_or = rec[fields["or"]]
        if math.isinf(expected_or):
            if stored_or not in (None, "inf", "Infinity"):
                fail(f"{label}: infinite odds ratio not represented for {rec}")
        else:
            close(f"{label} odds ratio", stored_or, expected_or, rel=1e-7, abs_tol=1e-10)
        expected_p = fisher_greater(a, b, c, d)
        close(f"{label} Fisher p", rec[fields["p"]], expected_p)
        expected_ci = wald_ci(a, b, c, d)
        stored_ci = rec[fields["ci"]]
        if expected_ci is None:
            if stored_ci is not None:
                fail(f"{label}: expected undefined Wald interval for zero cell")
        else:
            if not isinstance(stored_ci, list) or len(stored_ci) != 2:
                fail(f"{label}: missing Wald interval")
            close(f"{label} CI lower", stored_ci[0], expected_ci[0], rel=1e-6, abs_tol=1e-8)
            close(f"{label} CI upper", stored_ci[1], expected_ci[1], rel=1e-6, abs_tol=1e-8)
        groups.setdefault(rec[family_key], []).append((rec, expected_p))

    for family, members in groups.items():
        if len(members) != expected_per_family:
            fail(f"{label}: {family} has {len(members)} tests, expected {expected_per_family}")
        qs = bh_adjust([p for _, p in members])
        for (rec, _), q in zip(members, qs):
            close(f"{label} BH q ({family})", rec[fields["q"]], q, rel=5e-5, abs_tol=1e-12)


def main():
    order_data = load(DATA / "external_ordering_recomputed.json")
    download_hashes = load(DATA / "download_sha256.json")
    source_names = {
        "GSE11923": "GSE11923_series_matrix.txt.gz",
        "GSE50438": "GSE50438_series_matrix.txt.gz",
    }
    for accession, filename in source_names.items():
        if order_data["source_sha256"].get(accession) != download_hashes.get(filename):
            fail(f"source checksum mismatch for {accession}")

    order_count = 0
    for dataset in order_data["results"]:
        phases = dataset["phase_hours"]
        n = int(dataset["n_samples"])
        if len(phases) != n:
            fail(f"{dataset['name']}: phase vector length does not match n_samples")
        methods = dataset["methods"]
        if "H1 walk" not in methods:
            fail(f"{dataset['name']}: missing H1 representative walk")
        h1_support = set(methods["H1 walk"]["order"])
        for method_name, rec in methods.items():
            order = [int(i) for i in rec["order"]]
            if not order or any(i < 0 or i >= n for i in order):
                fail(f"{dataset['name']} / {method_name}: invalid sample index")
            close(f"{dataset['name']} / {method_name} succession",
                  rec["succession"], succession(phases, order), rel=1e-8, abs_tol=1e-10)
            coverage = len(set(order)) / n
            close(f"{dataset['name']} / {method_name} coverage",
                  rec["coverage"], coverage, rel=1e-8, abs_tol=1e-10)
            common = [i for i in order if i in h1_support]
            close(f"{dataset['name']} / {method_name} common-support succession",
                  rec["common_support_succession"],
                  succession(phases, common), rel=1e-8, abs_tol=1e-10)
            order_count += 1

    heat_records = load(DATA / "heat_enrichment_recomputed.json")
    check_enrichment(
        heat_records,
        label="primary enrichment",
        fields={
            "counts": ("a", "b", "c", "d"),
            "n_mapped": "n_mapped",
            "or": "odds_ratio",
            "p": "p",
            "ci": "ci95",
            "q": "q",
        },
        family_key="test",
        expected_per_family=4,
    )

    sensitivity = load(DATA / "candidate_cutoff_sensitivity.json")
    if len(sensitivity["results"]) != 24:
        fail(f"candidate-cutoff sensitivity has {len(sensitivity['results'])} rows, expected 24")
    check_enrichment(
        sensitivity["results"],
        label="cutoff sensitivity",
        fields={
            "counts": ("a_candidate_positive", "b_candidate_negative",
                       "c_non_candidate_positive", "d_non_candidate_negative"),
            "n_mapped": "n_mapped",
            "or": "odds_ratio",
            "p": "p_one_sided",
            "ci": "ci95_log_odds_wald",
            "q": "q_bh_12_tests_within_outcome_family",
        },
        family_key="outcome_family",
        expected_per_family=12,
    )

    candidate_counts = {
        "phase_candidates_TOT.json": (582, 117),
        "phase_candidates_TR.json": (689, 138),
    }
    for filename, (expected_n, expected_candidates) in candidate_counts.items():
        records = load(DATA / filename)
        n_candidates = sum(bool(r["candidate"]) for r in records)
        if len(records) != expected_n or n_candidates != expected_candidates:
            fail(f"{filename}: got {len(records)} records and {n_candidates} candidates")

    plan = load(DATA / "v2_validation_plan.json")
    real = load(DATA / "v2_real_validation.json")
    if plan["algorithm_sha256"] != real["plan"]["algorithm_sha256"]:
        fail("frozen plan algorithm digest differs from the real-validation record")
    if real["environment"].get("omictopo_source_version") != "0.2.0":
        fail("source version in the real-validation record is not 0.2.0")
    if "omictopo_version_note" not in real["environment"]:
        fail("version-metadata discrepancy is not documented")
    if len(real["results"]["GSE3431"]["design"]["samples"]) != 36:
        fail("GSE3431 design does not contain the expected 36 samples")

    print(
        "Derived-record audit passed: "
        f"{len(order_data['results'])} external matrices, {order_count} ordering records, "
        f"{len(heat_records)} primary enrichment tests, "
        f"{len(sensitivity['results'])} sensitivity tests, "
        "candidate counts, source checksums, and version provenance."
    )


if __name__ == "__main__":
    main()
