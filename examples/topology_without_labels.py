"""What a phase-based method cannot even be asked.

Motivation
----------
``examples/benchmark_audit.py`` found the uncomfortable thing: on the ten
benchmark clouds, the *circular Mantel* test alone scores exactly as well as
the full two-line verdict (8/10).  The topological statistic added no
measurable accuracy, and after a multiplicity correction none of its p-values
survived at all.  A referee is entitled to read that as "this is a Mantel test
with extra steps".

That audit measured *detection*: is there a loop?  This script measures three
things a phase-based test structurally cannot do, because all three require
dropping the labels rather than fitting them:

A. Side-information accounting
    ``C(t) = min(|dt|, P - |dt|)`` is built from the sampling times.  Permute
    the time labels and the Mantel statistic collapses to its null; the H1
    verdict is bit-identical, because it never saw a label.  Both numbers are
    measured rather than asserted.

B. Unlabelled cyclic-order recovery
    The reduction that produces the dominant H1 bar also produces the **cycle
    itself**, as a closed walk over sample indices
    (:mod:`omictopo.cycles`).  That walk orders the samples.  The clock was
    never used to build it, so the order is a *prediction* that can be scored
    against the clock afterwards — something a Mantel test cannot produce,
    since it needs the very order it is supposedly establishing.

    Scored with ``circular_succession_score``: the share of steps in which the
    recovered cycle either stays at the same time of day or advances to the
    next one.  Baselines, all label-free: the first principal coordinate, the
    angle of a two-dimensional embedding (the classical "PC1-PC2 phase"), and
    a greedy shortest tour of the distance matrix.

C. Counting independent cycles
    ``beta_1`` is an integer, so it can be used as a *model-selection* answer:
    how many independent cyclic parameters does this state space have?  The
    useful surprise here is that a single time axis answers "one", however
    many oscillators drive the cells: ``t -> (t/P1, t/P2)`` is a closed
    geodesic on the torus, not the torus.  A second independent cyclic
    parameter needs a two-dimensional design, and the script builds one
    (circadian time x an independent second cycle) with data simulated to the
    same shape as the real clouds, plus the 0-, 1- and 2-parameter controls.

Everything here is computed with the labels held out; the clock is used only
to score.

Run it with::

    python examples/prepare_datasets.py --outdir data/datasets
    python examples/topology_without_labels.py --data-root data/datasets

Outputs ``topology_without_labels.json`` and two figures under ``--outdir``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset_registry import SPECS, load_prepared  # noqa: E402
from multi_dataset_benchmark import covariance_matched_null, mantel  # noqa: E402

from omictopo import persistent_homology  # noqa: E402
from omictopo.cycles import (  # noqa: E402
    circular_succession_score,
    dominant_h1_cycles,
    h1_bar_lengths,
)
from omictopo.geometry import pairwise_distances, pca_embed, standardize  # noqa: E402

# The clouds this script reasons about, with the topology their design
# predicts.  Declared here rather than discovered, for the same reason the
# benchmark declares it in the registry.
CLOUDS = [
    "ath_diurnal", "ath_diurnal_heat", "ath_diurnal_ll", "ath_diurnal_ld",
    "rice_diurnal", "tomato_fruit_dev", "ath_recovery",
]
ALPHA = 0.05


# --------------------------------------------------------------------------
# label-free machinery
# --------------------------------------------------------------------------
def clock_phase(times: np.ndarray, period: float) -> np.ndarray:
    """Time of day as an angle in ``[0, 2 pi)``."""
    return 2.0 * np.pi * np.mod(np.asarray(times, dtype=float), period) / period


def greedy_tour(D: np.ndarray) -> np.ndarray:
    """A shortest-tour heuristic: repeatedly step to the nearest unvisited sample.

    A classical, completely label-free way to turn a point cloud into a cyclic
    order, and the reason a topological cycle has to earn its place: for
    samples lying on a ring, the shortest tour *is* the ring.
    """
    n = D.shape[0]
    used = np.zeros(n, dtype=bool)
    tour = [0]
    used[0] = True
    for _ in range(n - 1):
        remaining = np.flatnonzero(~used)
        tour.append(int(remaining[int(np.argmin(D[tour[-1], remaining]))]))
        used[tour[-1]] = True
    return np.asarray(tour, dtype=int)


def one_dimensional_order(Z: np.ndarray) -> np.ndarray:
    """Samples sorted along the first principal coordinate."""
    return np.argsort(pca_embed(Z, 1)[:, 0])


def two_dimensional_order(Z: np.ndarray) -> np.ndarray:
    """Samples sorted by their angle in the PC1-PC2 plane.

    This is the classical way to read a phase off an ordination, and it needs
    the analyst to decide in advance that two dimensions are the right number.
    """
    emb = pca_embed(Z, 3)
    x = emb[:, 0] - emb[:, 0].mean()
    y = emb[:, 1] - emb[:, 1].mean()
    return np.argsort(np.arctan2(y, x))


def top_bar_rel(D: np.ndarray) -> float:
    """Longest H1 bar of a distance matrix, in RMS-radius units (0 if none)."""
    bars = h1_bar_lengths(D)
    return float(bars[0]) if bars.size else 0.0


def permutation_null(phases: np.ndarray, order: np.ndarray, observed: float,
                     n_perm: int = 20_000, seed: int = 0) -> dict:
    """Null for a succession score: shuffle which sample carries which phase.

    The null is not centred on zero and it is not tight.  A cycle has no
    canonical direction, so ``circular_succession_score`` takes the better of
    the two, which hands a scrambled order credit for whichever half of its
    steps happens to move one way; the null therefore sits near 0.5 rather
    than near 0.  Quote it — and quote a permutation p-value from it — instead
    of comparing against a rule of thumb.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([circular_succession_score(rng.permutation(phases), order)
                      for _ in range(int(n_perm))], dtype=float)
    return {
        "n_perm": int(n_perm),
        "median": round(float(np.median(draws)), 4),
        "q95": round(float(np.quantile(draws, 0.95)), 4),
        "max": round(float(draws.max()), 4),
        "p_value": round(float((np.count_nonzero(draws >= observed) + 1)
                               / (int(n_perm) + 1)), 6),
    }


def benjamini_hochberg(p_values, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up, returning the set that survives.

    The order statistic is a per-cloud test, so the family has to be corrected
    before any "6 of 7" sentence is written.
    """
    p = np.asarray([np.nan if v is None else float(v) for v in p_values], dtype=float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    out = {"alpha": alpha, "n_tests": m, "n_surviving": 0,
           "surviving": [], "thresholds": [], "min_threshold": None}
    if m == 0:
        return out
    order = np.argsort(p[ok])
    sorted_p = p[ok][order]
    thresholds = alpha * (np.arange(1, m + 1)) / m
    passing = np.flatnonzero(sorted_p <= thresholds)
    out["thresholds"] = [round(float(t), 6) for t in thresholds]
    out["min_threshold"] = round(float(thresholds[0]), 6)
    if passing.size:
        keep = order[: int(passing[-1]) + 1]
        out["n_surviving"] = int(keep.size)
        out["surviving"] = [round(float(p[ok][k]), 6) for k in keep]
    return out


def best_cut_score(phases: np.ndarray, order: np.ndarray) -> float:
    """Best score a *linear* order can reach by choosing where to close it.

    A linear method is free to declare that the cycle closes anywhere, and
    this gives it every one of those choices.  If the score stays low under
    this oracle, the failure is in the ordering itself and not in the closure.
    """
    m = order.size
    if m < 3:
        return 0.0
    return float(max(circular_succession_score(phases, np.roll(order, -r))
                     for r in range(m)))


def bootstrap_cycle_scores(X: np.ndarray, phases: np.ndarray, metric: str,
                           n_draws: int = 40, rank: int = 1,
                           seed: int = 0) -> dict:
    """Stability of the recovered cycle under gene resampling.

    Resampling *features* leaves the sample size and the clock untouched, so
    it probes the cycle's dependence on which genes were assayed rather than
    on how many time points were collected.  This matters here: the audit's
    main complaint was statistical power at n = 16-26, and this is the one
    robustness axis that n does not set.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    scores, coverages, top_bars, orders = [], [], [], []
    for _ in range(int(n_draws)):
        cols = rng.integers(0, n_features, size=n_features)
        Z = standardize(X[:, cols])
        D = pairwise_distances(Z, metric=metric)
        cycles = dominant_h1_cycles(D=D, rank=rank)
        if not cycles or not cycles[0]["order"]:
            scores.append(0.0)
            coverages.append(0.0)
            top_bars.append(0.0)
            continue
        order = np.asarray(cycles[0]["order"], dtype=int)
        scores.append(circular_succession_score(phases, order))
        coverages.append(order.size / n_samples)
        top_bars.append(top_bar_rel(D))
        orders.append(order)
    return {
        "n_draws": int(n_draws),
        "score_median": round(float(np.median(scores)), 4),
        "score_q05": round(float(np.quantile(scores, 0.05)), 4),
        "score_q95": round(float(np.quantile(scores, 0.95)), 4),
        "fraction_perfect": round(float(np.mean(np.asarray(scores) >= 0.9999)), 4),
        "coverage_median": round(float(np.median(coverages)), 4),
        "top_bar_median": round(float(np.median(top_bars)), 4),
    }


# --------------------------------------------------------------------------
# A. side-information accounting
# --------------------------------------------------------------------------
def audit_side_information(cloud, X, times, period, metric="correlation",
                           n_perm=2000, n_label_draws=20, seed=0):
    """How much of each statistic is a function of the labels?"""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    period = float(period or 24.0)

    def circular_distance(t):
        phase = np.mod(t, period)[:, None]
        d = np.abs(phase - phase.T)
        return np.minimum(d, period - d)

    r_true, p_true = mantel(D, circular_distance(times), n_perm=n_perm, seed=seed)
    r_linear, p_linear = mantel(D, np.abs(times[:, None] - times[None, :]),
                                n_perm=n_perm, seed=seed)

    rng = np.random.default_rng(seed + 1)
    permuted = []
    h1_shift = []
    base_h1 = None
    for _ in range(int(n_label_draws)):
        t_perm = rng.permutation(times)
        r, _ = mantel(D, circular_distance(t_perm), n_perm=500, seed=seed)
        permuted.append(r)
        h1 = top_bar_rel(D)                # depends on D only, never on labels
        h1_shift.append(0.0 if base_h1 is None else abs(h1 - base_h1))
        base_h1 = h1

    permuted = np.asarray(permuted, dtype=float)
    return {
        "cloud": cloud,
        "n_samples": int(X.shape[0]),
        "mantel_r_circular_true_labels": round(float(r_true), 4),
        "mantel_p_circular_true_labels": float(p_true),
        "mantel_r_linear_true_labels": round(float(r_linear), 4),
        "mantel_r_circular_permuted_labels_median": round(float(np.median(permuted)), 4),
        "mantel_r_circular_permuted_labels_q95": round(float(np.quantile(permuted, 0.95)), 4),
        "mantel_r_drop_when_labels_are_masked": round(float(r_true - np.median(permuted)), 4),
        "h1_max_rel": round(float(base_h1), 4),
        "h1_max_rel_change_when_labels_are_masked": round(float(max(h1_shift)), 12),
        "labels_consumed": {"circular_mantel": 1, "h1": 0},
    }


# --------------------------------------------------------------------------
# B. unlabelled cyclic-order recovery
# --------------------------------------------------------------------------
def recover_order(cloud, X, times, period, expected, metric="correlation",
                  n_null=200, n_boot=40, n_label_draws=40,
                  n_score_perm=20_000, seed=0):
    """Score four label-free orders against a clock none of them used."""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    phases = clock_phase(times, float(period or 24.0))
    n = X.shape[0]

    cycles = dominant_h1_cycles(D=D, rank=1)
    if cycles and cycles[0]["order"]:
        cycle = cycles[0]
        ph_order = np.asarray(cycle["order"], dtype=int)
        ph_score = circular_succession_score(phases, ph_order)
        coverage = ph_order.size / n
        top_bar = cycle["persistence"] / float(np.sqrt(np.mean(D ** 2) / 2.0))
    else:
        cycle, ph_order, ph_score, coverage, top_bar = None, np.array([], int), 0.0, 0.0, 0.0

    orders = {
        "h1_cycle": ph_order,
        "pc1_order": one_dimensional_order(Z),
        "pc1_pc2_angle": two_dimensional_order(Z),
        "greedy_tour": greedy_tour(D),
    }
    scores = {}
    for name, order in orders.items():
        if order.size < 3:
            scores[name] = {"score": 0.0, "n_visited": int(order.size),
                            "coverage": round(order.size / n, 4), "null": None}
            continue
        observed = float(circular_succession_score(phases, order))
        scores[name] = {
            "score": round(observed, 4),
            "n_visited": int(order.size),
            "coverage": round(order.size / n, 4),
            "null": permutation_null(phases, order, observed,
                                     n_perm=n_score_perm, seed=seed),
        }
        if name in ("pc1_order", "greedy_tour"):
            # a linear order is free to close its cycle anywhere
            scores[name]["best_cut_score"] = round(
                float(best_cut_score(phases, order)), 4)

    # does the H1 bar itself beat a covariance-matched null?
    draws = covariance_matched_null(Z, metric=metric, n_draws=n_null, seed=seed)
    null_q95 = float(np.quantile(draws, 0.95))

    scores["h1_cycle"]["bar_exceeds_null_q95"] = bool(top_bar > null_q95)
    scores["h1_cycle"]["top_bar_rel"] = round(top_bar, 4)
    scores["h1_cycle"]["null_q95"] = round(null_q95, 4)
    scores["h1_cycle"]["testable"] = bool(
        scores["h1_cycle"]["null"] is not None
        and scores["h1_cycle"]["null"]["q95"] < 1.0
        and scores["h1_cycle"]["n_visited"] >= 6)
    if scores["h1_cycle"]["testable"]:
        scores["h1_cycle"]["passes"] = bool(
            scores["h1_cycle"]["score"] > scores["h1_cycle"]["null"]["q95"])
    else:
        scores["h1_cycle"]["passes"] = False

    # label-permutation score of the same cycle: the null the score really has
    rng = np.random.default_rng(seed + 2)
    shuffles = [circular_succession_score(rng.permutation(phases), ph_order)
                for _ in range(int(n_label_draws))] if ph_order.size >= 3 else []
    if shuffles:
        scores["h1_cycle"]["shuffled_label_q95"] = round(
            float(np.quantile(shuffles, 0.95)), 4)

    boot = (bootstrap_cycle_scores(X, phases, metric, n_draws=n_boot, seed=seed)
            if ph_order.size >= 3 else None)

    return {
        "cloud": cloud,
        "design_expects": expected,
        "n_samples": int(n),
        "n_distinct_times": int(np.unique(times).size),
        "period_hours": float(period or 24.0),
        "options": scores,
        "h1_bar": ({"birth": cycle["birth"], "death": cycle["death"],
                    "persistence": cycle["persistence"],
                    "n_edges": cycle["n_edges"]} if cycle else None),
        "gene_bootstrap": boot,
        "cycle_order": ph_order.tolist(),
        "times_in_cycle_order": [float(t) for t in times[ph_order]] if ph_order.size else [],
    }


# --------------------------------------------------------------------------
# C. counting independent cycles
# --------------------------------------------------------------------------
def simulate_design(design, n_genes, oscillators, noise_sd, seed=1,
                    amp_med=1.0, amp_sigma=1.0):
    """Simulate a gene x sample expression matrix for a given design.

    Parameters
    ----------
    design : np.ndarray, shape (n_samples, n_parameters)
        One column per *independent cyclic parameter* the experiment varies.
    oscillators : sequence of (int, float)
        ``(column, period)`` per oscillator.  The same column may be listed
        more than once with different periods: that is exactly the case of
        several oscillators riding on a single time axis, which produces a
        closed geodesic rather than a torus and therefore one loop, not
        several.
    """
    rng = np.random.default_rng(seed)
    k = max(len(oscillators), 1)
    amplitudes = amp_med * np.exp(rng.normal(0.0, amp_sigma, (n_genes, k)))
    phases = rng.uniform(0.0, 2.0 * np.pi, (n_genes, k))
    X = np.zeros((design.shape[0], n_genes))
    for j, (column, period) in enumerate(oscillators):
        angle = 2.0 * np.pi * np.outer(design[:, column],
                                       np.full(n_genes, 1.0 / period))
        X += np.cos(angle + phases[:, j][None, :]) * amplitudes[:, j][None, :]
    return X + rng.normal(0.0, noise_sd, X.shape)


def count_cycles(X, metric="correlation", n_null=120, seed=0):
    """How many H1 bars outrun the covariance-matched null?"""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    bars = h1_bar_lengths(D)
    draws = covariance_matched_null(Z, metric=metric, n_draws=n_null, seed=seed)
    q95 = float(np.quantile(draws, 0.95))
    return {
        "n_samples": int(X.shape[0]),
        "top_bars_rel": [round(float(v), 4) for v in bars[:5]],
        "null_q95": round(q95, 4),
        "n_bars_above_null_q95": int(np.count_nonzero(bars > q95)),
    }


def count_cycles_experiment(n_genes=3000, noise_sd=1.0, n_null=120, seed=0):
    """One time axis vs two: what does beta_1 count?"""
    # One time axis: 25 samples at 2 h intervals over two days.
    one_d = np.arange(0.0, 48.0 + 1e-9, 2.0)[:, None]
    # Two-dimensional design: a full circadian cycle times a full cycle of an
    # independent second parameter.  Both parameters must cover their whole
    # period, otherwise the second loop is truncated and cannot be found.
    t1 = np.arange(0.0, 24.0 + 1e-9, 2.0)
    t2 = np.arange(0.0, 16.0 + 1e-9, 2.0)
    grid = np.array([(a, b) for a in t1 for b in t2])        # 13 x 9 = 117 samples

    # (label, design, oscillators as (column, period), expected number of cycles)
    cases = [
        ("one_time_axis__0_oscillators", one_d, [], 0),
        ("one_time_axis__1_oscillator", one_d, [(0, 24.0)], 1),
        # two oscillators, one sampling axis: t -> (t/24, t/16) is a closed
        # geodesic *on* the torus, so the state space is still a circle
        ("one_time_axis__2_oscillators", one_d, [(0, 24.0), (0, 16.0)], 1),
        ("two_dimensional_grid__2_parameters", grid, [(0, 24.0), (1, 16.0)], 2),
        ("two_dimensional_grid__1_parameter", grid, [(0, 24.0)], 1),
        ("two_dimensional_grid__0_parameters", grid, [], 0),
    ]

    out = {}
    for label, design, oscillators, expected in cases:
        X = simulate_design(design, n_genes, oscillators, noise_sd, seed=seed)
        rec = count_cycles(X, n_null=n_null, seed=seed)
        driven = len({column for column, _ in oscillators})
        rec["design"] = ("one cyclic parameter" if design.shape[1] == 1
                         else f"{driven} driven parameter"
                              f"{'s' if driven != 1 else ''} on a 2-D grid")
        rec["oscillators"] = [f"col{c}@{int(p)}h" for c, p in oscillators]
        rec["n_cycles_expected"] = expected
        rec["n_cycles_detected"] = rec["n_bars_above_null_q95"]
        rec["correct"] = bool(rec["n_cycles_detected"] == expected)
        out[label] = rec
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(order_rows, counting, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # --- figure 1: order recovery ----------------------------------------
    methods = ["h1_cycle", "pc1_pc2_angle", "pc1_order", "greedy_tour"]
    pretty = {"h1_cycle": "H1 cycle (no labels)",
              "pc1_pc2_angle": "PC1-PC2 angle",
              "pc1_order": "PC1 order",
              "greedy_tour": "greedy tour"}
    colors = {"h1_cycle": "#c0392b", "pc1_pc2_angle": "#7b3f9d",
              "pc1_order": "#2c7fb8", "greedy_tour": "#e08a1e"}
    rows = [r for r in order_rows if r["n_samples"] >= 8]
    labels = [f"{r['cloud'].replace('_', ' ')}" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    ax = axes[0]
    x = np.arange(len(rows))
    width = 0.2
    for i, method in enumerate(methods):
        vals = [r["options"][method]["score"] for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, label=pretty[method],
               color=colors[method], zorder=3)
    nulls = [r["options"]["h1_cycle"]["null"]["q95"]
             if r["options"]["h1_cycle"]["null"] else np.nan for r in rows]
    ax.plot(x - 1.5 * width, nulls, "kv", markersize=6, zorder=5,
            label="H1 cycle's permutation null (95th pct)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=32, ha="right")
    ax.set_ylabel("succession score  (share of steps that follow the clock)")
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("A  Clock order recovered with the labels held out")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    demo = next((r for r in order_rows if r["cloud"] == "ath_diurnal:control"), None)
    if demo and demo["times_in_cycle_order"]:
        seq = np.asarray(demo["times_in_cycle_order"], dtype=float)
        idx = np.arange(seq.size)
        ax.plot(idx, seq, "o-", color=colors["h1_cycle"], lw=1.4, ms=5)
        ax.set_xlabel("position along the recovered cycle")
        ax.set_ylabel("true time of day (h)  -- hidden from the method")
        ax.set_title("B  The H1 cycle walks the clock\n"
                     "ath_diurnal control, no timestamps used")
        ax.grid(alpha=0.25)
        for k in range(seq.size - 1):
            if seq[k + 1] < seq[k]:
                ax.axvline(k + 0.5, color="#888", ls=":", lw=1.0)
                ax.annotate("cycle wraps", (k + 0.5, ax.get_ylim()[0]),
                            fontsize=7, rotation=90, va="bottom", ha="right",
                            color="#555", xytext=(2, 4), textcoords="offset points")
    else:
        ax.axis("off")
    fig.suptitle("Persistent homology recovers the phase of a biological clock "
                 "without being told what time it is", fontsize=12)
    path = os.path.join(outdir, "fig_labels_held_out.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # --- figure 2: counting independent cycles ----------------------------
    keys = list(counting)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 4.6),
                             layout="constrained", sharey=False)
    for ax, key in zip(np.atleast_1d(axes), keys):
        rec = counting[key]
        bars = rec["top_bars_rel"]
        xs = np.arange(len(bars))
        ax.bar(xs, bars, color="#2c7fb8", width=0.6, zorder=3)
        ax.axhline(rec["null_q95"], color="#c0392b", ls="--", lw=1.2,
                   label="95th pct of covariance-\nmatched null")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i + 1) for i in xs], fontsize=7.5)
        ax.set_ylim(0, max(max(bars), rec["null_q95"]) * 1.25 + 1e-6)
        title = key.replace("__", "\n").replace("_", " ")
        ax.set_title(f"{title}\ndetected {rec['n_cycles_detected']}, "
                     f"expected {rec['n_cycles_expected']}", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, axis="y")
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("H1 persistence / RMS radius")
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    fig.suptitle("How many independent cycles?  One time axis can only ever "
                 "produce one; two parameters produce two", fontsize=11)
    path = os.path.join(outdir, "fig_cycle_counting.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--metric", default="correlation")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(args.data_root)), "benchmark")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 96)
    print("OmicsTopology - what a phase-based method cannot be asked")
    print("=" * 96)
    print("Every statistic below is computed with the sampling times held out.")
    print("The clock is used only to score, never to compute.\n")

    audit, orders = [], []
    for key in CLOUDS:
        spec = SPECS[key]
        X, conditions, times, replicates, _ = load_prepared(
            os.path.join(args.data_root, key))
        for cond in sorted(set(conditions.tolist())):
            mask = conditions == cond
            cloud = f"{key}:{cond}"
            t0 = time.time()
            audit.append(audit_side_information(
                cloud, X[mask], times[mask], spec.period_hours,
                metric=args.metric, seed=args.seed))
            rec = recover_order(cloud, X[mask], times[mask], spec.period_hours,
                                spec.expected_topology, metric=args.metric,
                                n_null=args.n_null, n_boot=args.n_bootstrap,
                                seed=args.seed)
            rec["accession"] = spec.accession
            rec["organism"] = spec.organism
            rec["seconds"] = round(time.time() - t0, 1)
            orders.append(rec)
            o = rec["options"]
            print(f"  {cloud:<34} n={rec['n_samples']:<3} "
                  f"bar={o['h1_cycle']['top_bar_rel']:<6} "
                  f"(null95={o['h1_cycle']['null_q95']:<6})  "
                  f"H1cycle={o['h1_cycle']['score']:<6} "
                  f"PC1={o['pc1_order']['score']:<6} "
                  f"PC12={o['pc1_pc2_angle']['score']:<6} "
                  f"tour={o['greedy_tour']['score']:<6} "
                  f"p={o['h1_cycle']['null']['p_value'] if o['h1_cycle']['null'] else float('nan'):<9}"
                  f"[{rec['seconds']}s]")

    n_testable = sum(1 for r in orders if r["options"]["h1_cycle"]["testable"])
    n_pass = sum(1 for r in orders if r["options"]["h1_cycle"]["passes"])
    bh = benjamini_hochberg(
        [r["options"]["h1_cycle"]["null"]["p_value"]
         if r["options"]["h1_cycle"]["testable"] else None for r in orders])
    verdicts = {}
    for expect in ("loop", "line"):
        cells = [r for r in orders if r["design_expects"] == expect]
        testable = [r for r in cells if r["options"]["h1_cycle"]["testable"]]
        passed = [r for r in testable if r["options"]["h1_cycle"]["passes"]]
        verdicts[expect] = {
            "n_cells": len(cells), "n_testable": len(testable),
            "n_recovering_the_clock": len(passed),
            "clouds": [r["cloud"] for r in cells],
            "clouds_not_testable": [r["cloud"] for r in cells
                                    if r["cloud"] not in {t["cloud"] for t in testable}],
            "clouds_passing": [r["cloud"] for r in passed],
            "clouds_failing": [r["cloud"] for r in testable
                               if r["cloud"] not in {p_["cloud"] for p_ in passed}],
        }
    print(f"\n  H1 cycle order beats its permutation null in {n_pass}/{n_testable} "
          f"testable clouds")
    print(f"  Benjamini-Hochberg over the {bh['n_tests']} testable clouds "
          f"(alpha={bh['alpha']}): {bh['n_surviving']} survive, "
          f"smallest threshold {bh['min_threshold']}")
    for expect, v in verdicts.items():
        print(f"    design expects {expect:<5}: {v['n_recovering_the_clock']}"
              f"/{v['n_testable']} testable recover the clock "
              f"(not testable: {len(v['clouds_not_testable'])})"
              + (f"  failing: {v['clouds_failing']}" if v["clouds_failing"] else ""))

    print("\n" + "-" * 96)
    print("A. side-information accounting")
    print(f"{'cloud':<34}{'mantel r (true labels)':>24}{'mantel r (masked)':>20}"
          f"{'delta H1':>12}")
    for a in audit:
        print(f"{a['cloud']:<34}{a['mantel_r_circular_true_labels']:>24.3f}"
              f"{a['mantel_r_circular_permuted_labels_median']:>20.3f}"
              f"{a['h1_max_rel_change_when_labels_are_masked']:>12.1e}")
    r_true = np.array([a["mantel_r_circular_true_labels"] for a in audit])
    r_mask = np.array([a["mantel_r_circular_permuted_labels_median"] for a in audit])
    print(f"  mean Mantel r with the labels: {r_true.mean():+.3f};  "
          f"with the labels permuted: {r_mask.mean():+.3f};  "
          f"largest H1 change: "
          f"{max(a['h1_max_rel_change_when_labels_are_masked'] for a in audit):.1e}")

    print("\n" + "-" * 96)
    print("C. counting independent cycles (simulated to the real clouds' shape)")
    counting = count_cycles_experiment(n_null=max(60, args.n_null // 2),
                                       seed=args.seed)
    for label, rec in counting.items():
        print(f"  {label:<38} design={rec['design']:<34} "
              f"bars={rec['n_bars_above_null_q95']} expected={rec['n_cycles_expected']} "
              f"{'OK' if rec['correct'] else 'MISS'}   "
              f"top bars={rec['top_bars_rel'][:3]}")

    figures = []
    if not args.no_figures:
        try:
            figures = make_figures(orders, counting, outdir)
            print("\nfigures:")
            for f in figures:
                print("  " + f)
        except ImportError:
            print("\nmatplotlib unavailable; figures skipped")

    blob = {
        "config": {"metric": args.metric, "n_null": args.n_null,
                   "n_bootstrap": args.n_bootstrap, "seed": args.seed},
        "claim": {
            "question": "does persistent homology answer something a phase-"
                        "based test cannot?",
            "A": "circular Mantel consumes the phase; permuting the labels "
                 "collapses it, while H1 does not move.",
            "B": "the dominant H1 cycle predicts the clock order with no "
                 "labels; the score is a prediction, not a fit.",
            "C": "beta_1 counts independent cyclic parameters; a single time "
                 "axis can only ever give one, whatever the number of "
                 "oscillators.",
        },
        "A_side_information": audit,
        "B_order_recovery": orders,
        "C_cycle_counting": counting,
        "verdicts_by_design": verdicts,
        "multiplicity": bh,
        "summary": {
            "n_clouds": len(orders),
            "n_h1_order_pass": n_pass,
            "n_h1_order_testable": n_testable,
            "counting_correct": sum(1 for r in counting.values() if r["correct"]),
            "counting_total": len(counting),
        },
    }
    path = os.path.join(outdir, "topology_without_labels.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(f"\n[labels] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
