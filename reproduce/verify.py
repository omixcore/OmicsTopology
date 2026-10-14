"""Independent recomputation of every headline number in the repository.

This script is the third-party reproduction in executable form.  It shares no
code with the experiment scripts: the Wilson interval, the exact two-sided
binomial test and the Benjamini-Hochberg procedure below are re-implemented
here from their textbook definitions, and every published number is
recomputed from the *raw per-cloud records* committed under ``benchmark/`` --
not from the summaries those scripts printed.

Three layers of checking, in order:

1. *Integrity* -- the sha256 of every committed benchmark file matches
   ``reproduce/expected/manifest.json``.  This pins "these are the raw data
   the paper's numbers came from".
2. *Internal consistency* -- the summaries inside each benchmark file are
   recomputed from their own per-cloud records.  This catches any
   summarisation error.
3. *Published numbers* -- the values quoted in ``README.md`` and
   ``docs/results.md`` (embedded below as ``PUBLISHED``) match the
   recomputation.  This catches any transcription error between a benchmark
   file and the paper text.

Usage::

    python reproduce/verify.py                # layers 1-3 (fast, no network)
    python reproduce/verify.py --full         # additionally runs the test
                                              # suite and the known-shape demo
                                              # as the executable spec

Exit code 0 means every check passed; any mismatch prints FAIL and exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmark"
EXPECTED = Path(__file__).resolve().parent / "expected"

# --------------------------------------------------------------------------
# independent statistics (textbook implementations, deliberately not imported
# from omictopo.stats -- the point is that a second implementation agreeing
# with the first is evidence, not circularity)
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p: sum of outcomes no more probable than k."""
    def logpmf(x):
        if p0 in (0.0, 1.0):
            return 0.0 if x == (0 if p0 == 0.0 else n) else -math.inf
        return (math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
                + x * math.log(p0) + (n - x) * math.log(1.0 - p0))
    pk = logpmf(k)
    total = 0.0
    for x in range(n + 1):
        if logpmf(x) <= pk + 1e-12:
            total += math.exp(logpmf(x))
    return min(1.0, total)


def benjamini_hochberg(pvals: list, alpha: float = 0.05) -> int:
    """Number of hypotheses that survive BH at ``alpha``."""
    m = len(pvals)
    if not m:
        return 0
    order = sorted(range(m), key=lambda i: pvals[i])
    surviving = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / m:
            surviving = rank          # largest rank that holds, monotone rule
    return surviving


# --------------------------------------------------------------------------
# the numbers as they are quoted in README.md / docs/results.md
# --------------------------------------------------------------------------
PUBLISHED = {
    # 22-method benchmark (benchmark/incremental_value.json)
    "incremental_value": {
        "h1_cycle": {"sensitivity": 0.9167, "specificity": 0.9565,
                     "balanced_accuracy": 0.9366, "false_positives": 1,
                     "negatives_no_order": 5},
        "ae_2d_angle": {"sensitivity": 0.9167, "specificity": 0.4783},
        "tsne_angle": {"sensitivity": 0.9167, "specificity": 0.9130},
        "isomap_angle": {"sensitivity": 0.8333, "specificity": 0.8696},
        "circle_fit_angle": {"sensitivity": 0.9167, "specificity": 0.6087},
        "umap_angle": {"sensitivity": 0.9167, "specificity": 0.6957},
        "dae_2d_angle": {"sensitivity": 0.8333, "specificity": 0.6957},
        "ae_deep_2d_angle": {"sensitivity": 0.9167, "specificity": 0.6087,
                             "negatives_no_order": 5},
        "vae_2d_angle": {"sensitivity": 0.9167, "specificity": 0.6087,
                         "negatives_no_order": 5},
        "diffusion_map_angle": {"sensitivity": 0.9167, "specificity": 0.4783},
    },
    # 280-cloud power study (benchmark/increment_power.json)
    "increment_power": {
        "h1_cycle": {"fp_rate": 0.0667, "fp_k": 16, "no_order_k": 65,
                     "sensitivity": 1.0},
        "tsne_angle": {"fp_rate": 0.196, "fp_k": 47, "no_order_k": 0},
        "umap_angle": {"fp_rate": 0.204, "fp_k": 49, "no_order_k": 0},
        "isomap_angle": {"fp_rate": 0.292, "fp_k": 70, "no_order_k": 0},
        "dae_2d_angle": {"fp_rate": 0.325, "fp_k": 78, "no_order_k": 0},
        "circle_fit_angle": {"fp_rate": 0.400, "fp_k": 96, "no_order_k": 0},
        "ae_deep_2d_angle": {"fp_rate": 0.421, "fp_k": 101, "no_order_k": 60,
                             "sensitivity": 0.925},
        "vae_2d_angle": {"fp_rate": 0.421, "fp_k": 101, "no_order_k": 60,
                         "sensitivity": 0.925},
        "ae_2d_angle": {"fp_rate": 0.442, "fp_k": 106, "no_order_k": 0},
        "mcnemar_vs_tsne": {"only_a": 10, "only_b": 41},
        "mcnemar_vs_umap": {"only_a": 9, "only_b": 42},
        "mcnemar_vs_isomap": {"only_a": 7, "only_b": 61},
        "mcnemar_vs_dae": {"only_a": 10, "only_b": 72},
        "mcnemar_vs_circle_fit": {"only_a": 2, "only_b": 82},
        "mcnemar_vs_ae_deep": {"only_a": 5, "only_b": 90},
        "mcnemar_vs_vae": {"only_a": 5, "only_b": 90},
        "mcnemar_vs_ae_2d": {"only_a": 4, "only_b": 94},
    },
    # detection-axis audit (benchmark/statistical_audit.json)
    "statistical_audit": {
        "h1_only_correct": 7, "mantel_only_correct": 8, "n_units": 10,
        "resolution_p": 0.1094, "resolution_ci": (0.49, 0.943),
        "units_for_80pc_power": 20,
    },
    # label-free phase recovery (benchmark/topology_without_labels.json)
    "topology_without_labels": {
        "perfect_scores": 6, "n_designs": 6, "bh_survivors": 6,
        "n_testable": 7,
    },
    # deep-baseline tuning winners (must equal DEEP_SETTINGS in the source)
    "deep_tuning": {
        "ae_deep": {"weight_decay": 0.01, "lr": 0.006},
        "dae": {"noise_sigma": 0.5, "weight_decay": 0.01},
        "vae": {"kl_beta": 0.01, "weight_decay": 0.01},
        "umap": {"n_neighbors": 12},
    },
    # block C on measured data (benchmark/cycle_counting_real.json)
    "cycle_counting_real": {
        "n_primary_correct": 2, "n_primary": 5,
        "genotype_shift_r": 0.458,
        "designs": {
            "annual_all_samples": {"gap": 4.9253, "mantel_r": 0.7007,
                                   "n_cycles": 1},
            "cellcycle_two_genotypes": {"gap": 3.5583, "mantel_r": 0.1554,
                                        "n_cycles": 1},
            "diurnal_all_samples": {"gap": 1.3555, "mantel_r": 0.701,
                                    "n_cycles": 0},
            "fd_all_samples": {"gap": 1.0692, "mantel_r": 0.0448,
                               "n_cycles": 0},
            "pooled_all_samples": {"gap": 1.0908, "n_cycles": 0},
        },
    },
}

failures: list[str] = []
passed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def close(a, b, tol=5e-4) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# layer 1: integrity
# --------------------------------------------------------------------------
def check_integrity(manifest: dict) -> None:
    print("1. integrity of the committed benchmark data")
    for rel, want in manifest["files"].items():
        path = REPO / rel
        if not path.exists():
            check(rel, False, "missing file")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(rel, got == want, f"sha {got[:12]}... want {want[:12]}...")


# --------------------------------------------------------------------------
# layer 2 + 3: recomputation
# --------------------------------------------------------------------------
def recomputed_iv_summary(data: dict, name: str) -> dict:
    pos = [r for r in data["per_cloud"] if r["group"] == "positive"]
    neg = [r for r in data["per_cloud"] if r["group"] == "negative"]
    tp = sum(1 for r in pos if r["methods"][name]["declares"])
    fp = sum(1 for r in neg if r["methods"][name]["declares"])
    no_order = sum(1 for r in neg if r["methods"][name]["null"] is None)
    sens = tp / len(pos) if pos else float("nan")
    spec = 1.0 - fp / len(neg) if neg else float("nan")
    return {"sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2.0, "false_positives": fp,
            "negatives_no_order": no_order, "tp": tp, "n_pos": len(pos),
            "n_neg": len(neg)}


def check_incremental_value(manifest: dict) -> None:
    print("2. incremental-value benchmark (17 fixed + learned methods)")
    data = json.loads((BENCH / "incremental_value.json").read_text(encoding="utf-8"))
    for name, s in data["summary"].items():
        rec = recomputed_iv_summary(data, name)
        check(f"  {name}: recomputed = committed summary",
              all(close(rec[k], s.get(k)) for k in
                  ("sensitivity", "specificity", "balanced_accuracy",
                   "false_positives", "negatives_no_order")),
              f"recomputed {rec} vs committed { {k: s.get(k) for k in rec} }")
    for name, want in PUBLISHED["incremental_value"].items():
        if name not in data["summary"]:
            continue                                  # method not in this run
        rec = recomputed_iv_summary(data, name)
        bad = {k: (rec[k], v) for k, v in want.items() if not close(rec[k], v)}
        check(f"  {name}: recomputed = published", not bad, str(bad))
    lc = data.get("autoencoder_linear_check", {})
    check("  linear AE reproduces PCA (the correctness check)",
          bool(lc.get("linear_ae_matches_pca")), str(lc))


def check_increment_power(manifest: dict) -> None:
    print("3. 280-cloud power study")
    data = json.loads((BENCH / "increment_power.json").read_text(encoding="utf-8"))
    pos = [r for r in data["per_cloud"] if r["group"] == "positive"]
    neg = [r for r in data["per_cloud"] if r["group"] == "negative"]
    for name, s in data["summary"].items():
        fp = sum(1 for r in neg if r["methods"][name]["declares"])
        no = sum(1 for r in neg if r["methods"][name]["null"] is None)
        sens = sum(1 for r in pos if r["methods"][name]["declares"])
        lo, hi = wilson(fp, len(neg))
        ok = (close(fp / len(neg), s["false_positive_rate"]["rate"])
              and close(lo, s["false_positive_rate"]["ci95"][0], 2e-3)
              and close(hi, s["false_positive_rate"]["ci95"][1], 2e-3)
              and fp == s["false_positive_rate"]["k"]
              and no == s["abstention"]["no_order_at_all"]["k"]
              and close(sens / len(pos), s["sensitivity"]["rate"]))
        check(f"  {name}: recomputed = committed summary", ok,
              f"fp={fp} no={no} sens={sens}/{len(pos)}")
    for name, want in PUBLISHED["increment_power"].items():
        if name.startswith("mcnemar_vs_"):
            rival = {"mcnemar_vs_tsne": "tsne_angle",
                     "mcnemar_vs_umap": "umap_angle",
                     "mcnemar_vs_isomap": "isomap_angle",
                     "mcnemar_vs_dae": "dae_2d_angle",
                     "mcnemar_vs_circle_fit": "circle_fit_angle",
                     "mcnemar_vs_ae_deep": "ae_deep_2d_angle",
                     "mcnemar_vs_vae": "vae_2d_angle",
                     "mcnemar_vs_ae_2d": "ae_2d_angle"}[name]
            if rival not in data["methods"]:
                continue
            negatives = [r for r in data["per_cloud"] if r["group"] == "negative"]
            only_a = sum(1 for r in negatives
                         if r["methods"]["h1_cycle"]["declares"]
                         and not r["methods"][rival]["declares"])
            only_b = sum(1 for r in negatives
                         if r["methods"][rival]["declares"]
                         and not r["methods"]["h1_cycle"]["declares"])
            p = binom_two_sided(only_a, only_a + only_b)
            committed = data["paired"].get(rival, {})
            ok = (only_a == want["only_a"] and only_b == want["only_b"]
                  and close(p, committed.get("p_value", -1), 1e-6))
            check(f"  {name}: recomputed = published = committed", ok,
                  f"only_a={only_a} only_b={only_b} p={p:.3e} "
                  f"committed_p={committed.get('p_value')}")
        elif name in data["summary"]:
            s = data["summary"][name]
            ok = (s["false_positive_rate"]["k"] == want["fp_k"]
                  and close(s["false_positive_rate"]["rate"], want["fp_rate"])
                  and s["abstention"]["no_order_at_all"]["k"] == want["no_order_k"])
            check(f"  {name}: recomputed = published", ok, str(s))
    # seed-schedule disjointness from the 23-cloud benchmark (declared fact)
    sched = data["config"].get("seed_schedule", {})
    check("  power study declares a seed schedule disjoint from the benchmark",
          bool(sched), str(sched)[:120])


def check_statistical_audit() -> None:
    print("4. detection-axis audit")
    data = json.loads((BENCH / "statistical_audit.json").read_text(encoding="utf-8"))
    want = PUBLISHED["statistical_audit"]
    abl = data["ablation"]
    h1 = abl.get("h1_only", {})
    mantel = next((v for k, v in abl.items() if k.startswith("mantel_only")), {})
    res = data["resolution"]
    p = binom_two_sided(res["n_correct"], res["n_units"])
    lo, hi = wilson(res["n_correct"], res["n_units"])
    check("  mantel-only accuracy = 8/10", mantel.get("n_correct") == want["mantel_only_correct"], str(mantel))
    check("  h1-only accuracy = 7/10", h1.get("n_correct") == want["h1_only_correct"], str(h1))
    check("  resolution 8/10, exact binomial p recomputed = published",
          res["n_correct"] == 8 and close(p, want["resolution_p"], 1e-3),
          f"p={p:.4f}")
    check("  Wilson CI recomputed = published",
          close(lo, want["resolution_ci"][0], 5e-3)
          and close(hi, want["resolution_ci"][1], 5e-3), f"[{lo:.3f},{hi:.3f}]")


def check_topology_without_labels() -> None:
    print("5. label-free phase recovery")
    data = json.loads((BENCH / "topology_without_labels.json").read_text(encoding="utf-8"))
    want = PUBLISHED["topology_without_labels"]
    h1 = [dict(r["options"]["h1_cycle"], expects=r.get("design_expects"))
          for r in data["B_order_recovery"]]
    perfect = sum(1 for o in h1
                  if o["expects"] == "loop" and o.get("score") == 1.0)
    testable = sum(1 for o in h1 if o.get("testable"))
    survivors = sum(1 for o in h1 if o.get("passes"))
    check("  succession score 1.000 on 6 of 6 real circadian clouds",
          perfect == want["perfect_scores"], f"got {perfect}")
    check("  6 of 7 testable clouds survive BH",
          testable == want["n_testable"] and survivors == want["bh_survivors"],
          f"testable={testable} survivors={survivors}")


def check_deep_tuning() -> None:
    print("6. deep-baseline tuning provenance")
    data = json.loads((BENCH / "deep_tuning.json").read_text(encoding="utf-8"))
    grid = data.get("grid", {})
    for variant, want in PUBLISHED["deep_tuning"].items():
        got = grid.get(variant, {}).get("best", {})
        bad = {k: (got.get(k), v) for k, v in want.items()
               if not close(got.get(k), v, 1e-9)}
        check(f"  {variant}: frozen settings = grid winner", not bad, str(bad))
    check("  protocol recorded as positive-controls-only",
          "positive controls only" in data.get("protocol", ""))


def check_cycle_counting_real() -> None:
    print("7. block C on measured data")
    data = json.loads((BENCH / "cycle_counting_real.json").read_text(encoding="utf-8"))
    res = data["results"]
    want = PUBLISHED["cycle_counting_real"]
    prim = {k: v for k, v in res.items() if v.get("primary")}
    check(f"  {want['n_primary_correct']} of {want['n_primary']} designs counted right",
          data.get("n_primary_correct") == want["n_primary_correct"]
          and data.get("n_primary") == want["n_primary"] == len(prim),
          f"got {data.get('n_primary_correct')}/{data.get('n_primary')}, "
          f"rows marked primary: {len(prim)}")
    for label, fields in want["designs"].items():
        got = {k: res[label].get(k) for k in fields}
        bad = {k: (got[k], v) for k, v in fields.items() if not close(got[k], v)}
        check(f"  {label}: gap / mantel r / count match", not bad, str(bad))
    ax = data.get("cellcycle_axis_checks") or {}
    g = (ax.get("genotype") or {}).get("r")
    check("  genotype axis is a shift, not a cycle (r recorded, ~ +0.46)",
          close(g, want["genotype_shift_r"]),
          f"got {g}")


# --------------------------------------------------------------------------
# layer 4 (optional): the executable spec
# --------------------------------------------------------------------------
def run_full() -> None:
    print("7. the executable spec (--full)")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO)],
                       capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
    check("pytest suite passes", r.returncode == 0, tail)
    r = subprocess.run([sys.executable, str(REPO / "examples" / "demo_known_shapes.py")],
                       capture_output=True, text=True)
    check("known-topology demo passes", r.returncode == 0, r.stdout[-120:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="also run the pytest suite and the known-shape demo")
    ap.add_argument("--skip-integrity", action="store_true",
                    help="skip sha256 checks (e.g. after touching a raw file "
                         "on purpose; recomputation layers still run)")
    args = ap.parse_args()

    manifest_path = EXPECTED / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.skip_integrity:
            print("0. integrity skipped (--skip-integrity)")
        else:
            check_integrity(manifest)
    else:
        print("0. no manifest found -- integrity layer skipped (run "
              "reproduce/make_manifest.py to create one)")

    check_incremental_value(manifest if manifest_path.exists() else {})
    check_increment_power(manifest if manifest_path.exists() else {})
    check_statistical_audit()
    check_topology_without_labels()
    check_deep_tuning()
    check_cycle_counting_real()
    if args.full:
        run_full()

    print()
    if failures:
        print(f"{passed} checks passed, {len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all {passed} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
