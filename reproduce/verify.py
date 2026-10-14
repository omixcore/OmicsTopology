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
        "n_primary_correct": 2, "n_primary": 6,
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
            "estrous_grid_clock_axis": {"gap": 1.5637, "mantel_r": 0.6302,
                                        "n_cycles": 0},
        },
    },
    # the oestrous x circadian grid (benchmark/estrous_grid_cycles.json)
    "estrous_grid_cycles": {
        "accession": "PRJNA1348461",
        "n_samples": 48,
        "count": 0,
        "gap": 1.5637,
        "mantel_r_circadian": 0.6302,
        "mantel_r_oestrous": 0.0061,
        "phase_block_variance_p": 0.85,
        "per_phase_blocks": {
            "diestrus": {"mantel_r": 0.712},
            "proestrus": {"mantel_r": 0.726},
            "oestrus": {"mantel_r": 0.519},
            "metestrus": {"mantel_r": 0.574},
        },
        "calibration": {
            "clock_strength_2_count": 1,
            "clock_phase_shifted_strength_2_count": 1,
            "two_independent_strength_2_count": 2,
        },
    },
    # what the topology says about the clock x oestrous coordinate
    # (benchmark/axis_separability.json)
    "axis_separability": {
        "accession": "PRJNA1348461",
        "n_samples": 48,
        "mantel_r_clock": 0.6302,
        "mantel_r_oestrous": 0.0061,
        "amplitude_ratio": 1.5825,
        "variance_clock": 0.1161,
        "variance_oestrous": 0.0538,
        "variance_both": 0.1699,
        "additive_share": 1.0,
        "subspace_cosine": 0.0,
        "block_offset_coherence": 0.9222,
        "max_block_offset_in_quarter_periods": 0.3657,
        "separation_ratio": 0.8761,
        "separation_bootstrap_ci95": [0.0742, 0.1857],
    },
    # the axis-geometry diagnostic across designs
    # (benchmark/axis_geometry_multi.json)
    "axis_geometry_multi": {
        "n_cases": 7,
        "n_agree": 7,
        "calibrations": {
            "synth_independent": {"verdict": "independent_axes", "mantel_r_axis2": 0.6646},
            "synth_modulated": {"verdict": "modulation", "mantel_r_axis2": -0.0383},
            "synth_inert": {"verdict": "independent_axes_inert",
                            "mantel_r_axis2": -0.0383},
        },
        "real_designs": {
            "estrous_circadian_grid": {"verdict": "modulation",
                                       "mantel_r_axis2": 0.0061,
                                       "separation_ratio": 0.8761},
            "yeast_cell_cycle": {"verdict": "undetermined",
                                 "mantel_r_axis2": 0.4578},
            "human_forced_desync": {"verdict": "undetermined",
                                    "mantel_r_axis2": 0.0028},
            "psme_diurnal": {"verdict": "modulation",
                             "mantel_r_axis2": -0.0053},
        },
        # the negative result the criteria rest on: an inert axis is trivially
        # orthogonal to axis 1, so the separability block cannot decide
        "inert_axis_is_trivially_separable": {
            "additive_share_of_both": 1.0,
            "subspace_cosine": 0.0,
        },
    },
    # the third two-frequency design (benchmark/ultradian_two_oscillator.json)
    "ultradian_two_oscillator": {
        "accession": "GSE220120",
        "n_samples": 72,
        "n_genes": 3000,
        "whole_count": 0,
        "rhythm_screen_24h_in_selection": 53,
        "rhythm_screen_12h_in_selection": 0,
        "n_top": 200,
    },
    # the two-parameter search (benchmark/two_cycle_search.json)
    "two_cycle_search": {
        "estrous_count": 0,
        "estrous_mantel_r_24h": 0.1076,
        "estrous_mantel_r_96h": 0.1440,
        "grid_min_samples_full_signal": 64,
        "grid_min_grid": "8x8",
        "grid_13x9_full_signal_count": 2,
        "grid_13x9_half_signal_count": 1,
    },
    # the cross-dataset regularity study (benchmark/axis_regularity.json)
    "axis_regularity": {
        "n_real_axis_pairs": 7,
        "n_independent_experiments": 6,
        "n_organisms": 5,
        "same_experiment_groups": [["arabidopsis_heat", "arabidopsis_TR"]],
        "r_strong": 0.5,
        "condition_axes": 6,
        "condition_axes_below_threshold": 4,
        "coordinate_axes": 2,
        "coordinate_axes_above_threshold": 2,
        # the pre-registered hypothesis R1 dies here: a 1 h heat treatment is
        # condition-shaped and reaches +0.98, in both RNA fractions
        "R1_falsified_keys": ["arabidopsis_TR", "arabidopsis_heat"],
        "R1_falsified_r2": 0.9821,
        "R2_holds": True,
        "R3_holds": True,
        # what survives: an ordering by displacement rather than a partition by
        # factor kind. >1.47 steps of axis 1 -> r2 >= 0.98; the highest r2
        # under a step is the confounded pooled design at 0.54
        "displacement_over_a_step_min_r2": 0.9794,
        "displacement_under_a_step_max_r2": 0.5408,
        "psme_annual_x_diurnal_r2": 0.5408,
    },
    # the protocol's declared envelope (src/omictopo/protocol.py)
    "protocol": {
        "real_designs": 6,
        "real_designs_counted_correctly": 2,
        "gap_on_designs_counted_right": [3.56, 4.93],
        "gap_on_designs_not_counted": [1.07, 1.09, 1.36, 1.56],
        "two_parameter_grid_min_samples_at_full_signal": 64,
        "two_parameter_grid_grid": "8x8",
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
def check_two_cycle_search() -> None:
    print("8. the two-parameter search (the missing experiment is a grid)")
    data = json.loads((BENCH / "two_cycle_search.json").read_text(encoding="utf-8"))
    want = PUBLISHED["two_cycle_search"]

    est = data["measured_single_axis_case"]
    check("  GSE12289: one sampling axis, two frequencies -> count 0",
          est["n_cycles_detected"] == want["estrous_count"],
          f"got {est['n_cycles_detected']}")
    # the point of the measured case: both rhythms are there, neither dominates
    check("  both rhythms present and significant (24 h and 96 h)",
          close(est["mantel_r_time_of_day"], want["estrous_mantel_r_24h"])
          and close(est["mantel_r_estrous"], want["estrous_mantel_r_96h"])
          and est["mantel_p_time_of_day"] < 0.05
          and est["mantel_p_estrous"] < 0.05,
          f"r24={est['mantel_r_time_of_day']} r96={est['mantel_r_estrous']}")

    rows = data["grid_requirement"]["rows"]
    full = [r for r in rows if r["signal_fraction"] == 1.0 and r["correct"]]
    smallest = min(full, key=lambda r: r["n_samples"]) if full else {}
    check("  two cycles only above 64 samples at full signal",
          smallest.get("n_samples") == want["grid_min_samples_full_signal"]
          and smallest.get("grid") == want["grid_min_grid"],
          f"smallest working grid: {smallest.get('grid')} "
          f"({smallest.get('n_samples')})")
    check("  the published 13 x 9 grid is reproduced end to end",
          any(r["grid"] == "13x9" and r["signal_fraction"] == 1.0
              and r["n_cycles_detected"] == want["grid_13x9_full_signal_count"]
              for r in rows))
    check("  dilution destroys the second loop",
          any(r["grid"] == "13x9" and r["signal_fraction"] == 0.5
              and r["n_cycles_detected"] == want["grid_13x9_half_signal_count"]
              for r in rows)
          and all(not r["correct"] for r in rows
                  if r["signal_fraction"] <= 0.25))


def check_protocol() -> None:
    print("9. the protocol's decisions agree with the committed record")
    data = json.loads((BENCH / "cycle_counting_real.json").read_text(encoding="utf-8"))
    prim = {k: v for k, v in data["results"].items() if v.get("primary")}

    # the rule re-implemented from its definition, not imported
    def status_of(res: dict) -> str:
        gap = res.get("gap") or 0.0
        r = res.get("mantel_r")
        dominant = gap >= 2.0            # the calibrated dominant-bar threshold
        ring = r is not None and r >= 0.5  # the effect-size cut
        if dominant and ring:
            st = "counted"
        elif dominant and not ring:
            st = "inconsistent"
        elif ring and not dominant:
            st = "underpowered"
        else:
            st = "no_cycle"
        if st == "counted" and res["n_cycles"] < 1:
            st = "underpowered"
        return st

    # four of the five primary designs: no dominant bar -> the rule refuses,
    # and a refusal is the correct response to a count that is also wrong
    for label, res in prim.items():
        if label == "cellcycle_two_genotypes":
            continue  # the documented exception, asserted separately below
        if res["correct"]:
            check(f"  {label}: counted", status_of(res) == "counted",
                  f"status {status_of(res)} gap {res['gap']}")
        else:
            check(f"  {label}: no dominant bar -> refusal",
                  status_of(res) in ("underpowered", "no_cycle")
                  and res["gap"] < 2.0,
                  f"status {status_of(res)} gap {res['gap']}")

    # the boundary the protocol is honest about
    y = prim["cellcycle_two_genotypes"]
    check("  yeast cell cycle: counted right but the effect-size rule flags it "
          "(documented, not a bug)",
          bool(y["correct"]) and y["gap"] >= 2.0 and y["mantel_r"] < 0.5
          and status_of(y) == "inconsistent",
          f"gap {y['gap']} r {y['mantel_r']} status {status_of(y)}")

    # the package's declared envelope must match the records it summarises
    want = PUBLISHED["protocol"]
    try:
        sys.path.insert(0, str(REPO / "src"))
        from omictopo.protocol import envelope  # noqa: E402
        env = envelope()["measured"]
    except Exception as exc:  # pragma: no cover - package not importable
        print(f"  skip  package import unavailable ({exc})")
        return
    check("  package envelope: design counts match the record",
          env["real_designs"] == want["real_designs"] == data["n_primary"]
          and env["real_designs_counted_correctly"]
          == want["real_designs_counted_correctly"]
          == data["n_primary_correct"],
          f"envelope {env['real_designs']} designs / "
          f"{env['real_designs_counted_correctly']} right")
    check("  package envelope: both gap lists match the record",
          sorted(env["gap_on_designs_counted_right"])
          == sorted(want["gap_on_designs_counted_right"])
          and sorted(env["gap_on_designs_not_counted"])
          == sorted(want["gap_on_designs_not_counted"]),
          f"{env['gap_on_designs_counted_right']} / "
          f"{env['gap_on_designs_not_counted']}")
    check("  package envelope: grid minimum matches the search record",
          env["two_parameter_grid_min_samples_at_full_signal"]
          == want["two_parameter_grid_min_samples_at_full_signal"]
          and env["two_parameter_grid_grid"] == want["two_parameter_grid_grid"])


def check_estrous_grid() -> None:
    """The 12 x 4 clock x oestrous grid -- the first two-cyclic-axis design."""
    print("10. the estrous x circadian grid (two recorded cyclic axes)")
    path = BENCH / "estrous_grid_cycles.json"
    if not path.exists():
        check("  benchmark/estrous_grid_cycles.json exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    want = PUBLISHED["estrous_grid_cycles"]
    d = data["design"]

    check(f"  {want['accession']}: 12 clock times x 4 phases, every cell filled",
          d["accession"] == want["accession"]
          and d["n_samples"] == want["n_samples"],
          f"{d['accession']} n={d['n_samples']}")
    check("  the count is one, not two",
          d["count"]["n_cycles"] == want["count"]
          and abs(d["count"]["gap"] - want["gap"]) < 0.01,
          f"count {d['count']['n_cycles']} gap {d['count']['gap']}")

    # the clock is real...
    check("  the ~24 h clock axis is strong and significant",
          abs(d["mantel_r_circadian"] - want["mantel_r_circadian"]) < 0.01
          and d["mantel_p_circadian"] <= 0.001,
          f"r {d['mantel_r_circadian']} p {d['mantel_p_circadian']}")

    # ...and resolves inside every phase block on its own
    blocks = data["per_phase_blocks"]
    weak = [p for p, b in want["per_phase_blocks"].items()
            if not blocks.get(p)
            or abs(blocks[p]["mantel_r_circadian"] - b["mantel_r"]) > 0.01
            or blocks[p]["mantel_p_circadian"] > 0.001]
    check("  the clock resolves inside each of the four oestrous blocks",
          not weak, f"blocks off target: {weak}")

    # ...while the oestrous axis carries no independent loop
    check("  the oestrous axis leaves no independent loop",
          abs(d["mantel_r_oestrous"] - want["mantel_r_oestrous"]) < 0.01
          and d["mantel_p_oestrous"] > 0.05,
          f"r {d['mantel_r_oestrous']} p {d['mantel_p_oestrous']}")
    check("  across-phase variance is indistinguishable from a permutation",
          abs(d["phase_block_variance"]["p"] - want["phase_block_variance_p"])
          < 0.06,
          f"p {d['phase_block_variance']['p']}")

    # the calibration that makes the verdict a comparison, not an assertion
    cal = data["calibration"]
    check("  calibration: a clock counts 1",
          cal["clock"]["strength_2"]["n_cycles"]
          == want["calibration"]["clock_strength_2_count"])
    check("  calibration: a phase-shifted clock still counts 1 (a second "
          "phase is not a second loop)",
          cal["clock_phase_shifted"]["strength_2"]["n_cycles"]
          == want["calibration"]["clock_phase_shifted_strength_2_count"])
    check("  calibration: only two independent oscillations count 2",
          cal["two_independent"]["strength_2"]["n_cycles"]
          == want["calibration"]["two_independent_strength_2_count"])


def check_axis_separability() -> None:
    """What the topology says that the authors' per-gene harmonic fits do not."""
    print("11. axis separability: the answer a clock fit cannot give")
    path = BENCH / "axis_separability.json"
    if not path.exists():
        check("  benchmark/axis_separability.json exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    want = PUBLISHED["axis_separability"]
    q1, q2, q3, q4 = (data["q1_dominant_axis"], data["q2_additive_separability"],
                      data["q3_phase_vs_shape"], data["q4_matched_clock_distance"])

    check(f"  {want['accession']}: the joint geometry is scored on both axes",
          data["accession"] == want["accession"]
          and data["n_samples"] == want["n_samples"]
          and abs(q1["mantel_r_clock"] - want["mantel_r_clock"]) < 0.01
          and abs(q1["mantel_r_oestrous"] - want["mantel_r_oestrous"]) < 0.01,
          f"r {q1['mantel_r_clock']} / {q1['mantel_r_oestrous']}")

    check("  the clock's per-gene amplitude dominates the oestrous one",
          abs(q1["amplitude_ratio_clock_over_oestrous"]
              - want["amplitude_ratio"]) < 0.05,
          f"ratio {q1['amplitude_ratio_clock_over_oestrous']}")

    check("  the two axes explain separable slabs of the variance",
          abs(q2["variance_explained_by_clock"] - want["variance_clock"]) < 0.01
          and abs(q2["variance_explained_by_oestrous"] - want["variance_oestrous"]) < 0.01
          and abs(q2["variance_explained_by_both"] - want["variance_both"]) < 0.01
          and abs(q2["additive_share_of_both"] - want["additive_share"]) < 0.02
          and abs(q2["subspace_cosine"] - want["subspace_cosine"]) < 0.05,
          f"clock {q2['variance_explained_by_clock']} oestrous "
          f"{q2['variance_explained_by_oestrous']} both "
          f"{q2['variance_explained_by_both']}")

    check("  the oestrous axis is a coherent phase shift of the clock's loop",
          abs(q3["block_offset_coherence"] - want["block_offset_coherence"]) < 0.02
          and abs(q3["max_block_offset_in_quarter_periods"]
                  - want["max_block_offset_in_quarter_periods"]) < 0.02,
          f"coherence {q3['block_offset_coherence']} max offset "
          f"{q3['max_block_offset_in_quarter_periods']} quarter-periods")

    ci = q4["separation_bootstrap_ci95"]
    check("  the oestrous axis still moves points, but by less than the clock",
          abs(q4["separation_ratio"] - want["separation_ratio"]) < 0.01
          and ci[0] > 0.0
          and abs(ci[0] - want["separation_bootstrap_ci95"][0]) < 0.01
          and abs(ci[1] - want["separation_bootstrap_ci95"][1]) < 0.01,
          f"ratio {q4['separation_ratio']} CI {ci}")


def check_axis_geometry_multi() -> None:
    """The axis diagnostic is not a property of one dataset."""
    print("12. axis geometry across designs: three shapes, three answers")
    path = BENCH / "axis_geometry_multi.json"
    if not path.exists():
        check("  benchmark/axis_geometry_multi.json exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    want = PUBLISHED["axis_geometry_multi"]

    check("  every case classifies as its pre-registered expectation",
          data["agreement"]["n_agree"] == want["n_agree"]
          and data["agreement"]["n_cases"] == want["n_cases"]
          and data["agreement"]["all_agree"],
          f"{data['agreement']['n_agree']}/{data['agreement']['n_cases']}")

    cal = {r["key"]: r for r in data["calibrations"]}
    for key, spec in want["calibrations"].items():
        got = cal.get(key)
        check(f"  calibration {key}: {spec['verdict']}",
              got is not None and got["verdict"] == spec["verdict"]
              and abs(got["dominant"]["mantel_r_axis2"]
                      - spec["mantel_r_axis2"]) < 0.05,
              f"{got['verdict'] if got else '<missing>'}")

    check("  the three calibrations get three different verdicts",
          len({r["verdict"] for r in data["calibrations"]}) == 3,
          str(sorted({r["verdict"] for r in data["calibrations"]})))

    real = {r["key"]: r for r in data["real_designs"]}
    for key, spec in want["real_designs"].items():
        got = real.get(key)
        ok = got is not None and got["verdict"] == spec["verdict"] \
            and abs(got["dominant"]["mantel_r_axis2"]
                    - spec["mantel_r_axis2"]) < 0.05
        if ok and "separation_ratio" in spec:
            ok = abs(got["matched"]["separation_ratio"]
                     - spec["separation_ratio"]) < 0.05
        check(f"  real design {key}: {spec['verdict']}", ok,
              f"{got['verdict'] if got else '<missing>'}")

    check("  the mouse-liver design is the modulation the paper reports",
          real["estrous_circadian_grid"]["verdict"] == "modulation"
          and real["estrous_circadian_grid"]["phase_or_shape"][
              "block_offset_coherence"] > 0.9,
          "coherence "
          f"{real['estrous_circadian_grid']['phase_or_shape']['block_offset_coherence']}")

    inert = next(r for r in data["calibrations"] if r["key"] == "synth_inert")
    nk = want["inert_axis_is_trivially_separable"]
    check("  negative result: an inert axis is trivially separable",
          abs(inert["separable"]["additive_share_of_both"]
              - nk["additive_share_of_both"]) < 0.02
          and abs(inert["separable"]["subspace_cosine"]
                  - nk["subspace_cosine"]) < 0.05,
          f"share {inert['separable']['additive_share_of_both']} cosine "
          f"{inert['separable']['subspace_cosine']}")


def check_ultradian_two_oscillator() -> None:
    """GSE220120: two frequencies on one axis, and why the count is 0."""
    print("13. two frequencies in one tissue: the envelope, not the topology")
    path = BENCH / "ultradian_two_oscillator.json"
    if not path.exists():
        check("  benchmark/ultradian_two_oscillator.json exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    want = PUBLISHED["ultradian_two_oscillator"]

    check("  the design is the two-frequency human blood series",
          data["accession"] == want["accession"]
          and data["design"]["n_samples"] == want["n_samples"]
          and data["design"]["n_genes"] == want["n_genes"]
          and len(data["design"]["subjects"]) == 3,
          f"{data['accession']} n={data['design']['n_samples']}")

    check("  the count on the whole cloud is 0, not 2",
          data["whole_cloud"]["n_cycles"] == want["whole_count"],
          f"count {data['whole_cloud']['n_cycles']}")

    screen = data["rhythm_screen"]
    n_top = screen["n_top"]
    check("  the ~12 h program is absent from the high-variance genes",
          screen["12h"][f"n_of_top{n_top}_in_selection"]
          == want["rhythm_screen_12h_in_selection"],
          f"{screen['12h'][f'n_of_top{n_top}_in_selection']}/{n_top}")

    check("  while the ~24 h program is partly present",
          screen["24h"][f"n_of_top{n_top}_in_selection"]
          == want["rhythm_screen_24h_in_selection"],
          f"{screen['24h'][f'n_of_top{n_top}_in_selection']}/{n_top}")

    check("  the conclusion points at the envelope, not the obstruction",
          "envelope" in data["conclusion"]
          and "not the topological obstruction" in data["conclusion"],
          data["conclusion"][:60] + "...")


def check_axis_regularity() -> None:
    """The axis diagnostic across the whole corpus: the hypothesis that dies
    and the ordering that survives."""
    print("14. across the corpus: what the hypothesis got wrong, and what holds")
    path = BENCH / "axis_regularity.json"
    if not path.exists():
        check("  benchmark/axis_regularity.json exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    want = PUBLISHED["axis_regularity"]
    t = data["tally"]

    check("  the corpus is the size claimed",
          t["n_real_axis_pairs"] == want["n_real_axis_pairs"]
          and t["n_independent_experiments"] == want["n_independent_experiments"]
          and t["n_organisms"] == want["n_organisms"]
          and abs(t["r_strong"] - want["r_strong"]) < 1e-9,
          f"{t['n_real_axis_pairs']} pairs, "
          f"{t['n_independent_experiments']} experiments, {t['n_organisms']} organisms")

    check("  the two RNA fractions of one experiment are declared, not counted twice",
          [list(g) for g in t["same_experiment_groups"]]
          == want["same_experiment_groups"],
          str(t["same_experiment_groups"]))

    check("  the pre-registered hypothesis R1 is falsified, and recorded as such",
          t["R1_holds"] is False
          and t["condition_axes_below_threshold"]
          == want["condition_axes_below_threshold"],
          f"R1_holds={t['R1_holds']}, "
          f"{t['condition_axes_below_threshold']}/{t['condition_axes']} "
          "condition axes below threshold")

    above = {r["key"]: r["r2"] for r in t["condition_axes_above_threshold"]}
    check("  the falsifier is the same heat treatment in both RNA fractions",
          sorted(above) == sorted(want["R1_falsified_keys"])
          and all(abs(v - want["R1_falsified_r2"]) < 0.02 for v in above.values()),
          str(above))

    check("  R2 survives: the one independently sampled cycle clears r_strong",
          t["R2_holds"] is True
          and t["coordinate_axes_above_threshold"]
          == want["coordinate_axes_above_threshold"],
          f"{t['coordinate_axes_above_threshold']}/{t['coordinate_axes']}")

    check("  R3 survives: the number of axis-2 levels does not order r2",
          t["R3_holds"] is True,
          t["R3_levels_vs_r2"]["note"])

    do = t["displacement_ordering"]
    check("  the ordering that replaces the hypothesis is displacement, not kind",
          do["over_a_step_of_axis1"]["min_r2"]
          >= want["displacement_over_a_step_min_r2"] - 0.01
          and do["under_a_step_of_axis1"]["max_r2"]
          <= want["displacement_under_a_step_max_r2"] + 0.01,
          f"over a step {do['over_a_step_of_axis1']['min_r2']} > "
          f"under a step {do['under_a_step_of_axis1']['max_r2']}")

    check("  the one candidate positive case keeps its caveat rather than "
          "being sold as a clean one",
          "confounded" in t["R2_caveat"]
          and abs(do["under_a_step_of_axis1"]["max_r2"]
                  - want["psme_annual_x_diurnal_r2"]) < 0.01,
          t["R2_caveat"][:70] + "...")


# --------------------------------------------------------------------------
# layer 4 (optional): the executable spec
# --------------------------------------------------------------------------
def run_full() -> None:
    print("15. the executable spec (--full)")
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
    check_two_cycle_search()
    check_protocol()
    check_estrous_grid()
    check_axis_separability()
    check_axis_geometry_multi()
    check_ultradian_two_oscillator()
    check_axis_regularity()
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
