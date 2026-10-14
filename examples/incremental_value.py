"""Does the topology add anything the cheaper methods cannot?

The problem this script exists to answer
----------------------------------------
``examples/benchmark_audit.py`` left the project with an uncomfortable result:
on the ten benchmark clouds, a circular Mantel test alone scores exactly as
well as the full two-line topological verdict (8/10).  Read literally that
says *the topological statistic contributes no accuracy*, and a referee is
entitled to call the method "a Mantel test with extra steps".

That audit measured **detection** — is there a loop?  This script measures the
thing the method actually claims, and the thing a Mantel test cannot be asked
to do: **produce a cyclic order of the samples without being given the clock**.

The claim, stated so it can fail
--------------------------------
Every method below sees only the expression matrix (or the distance matrix
built from it).  None of them sees the sampling times.  Each returns a
*permutation of the sample indices*, which is then scored against the clock
that was held out, with ``circular_succession_score`` (share of steps that
stay at the current phase or advance to the next one).

A method **declares a cycle** when its score beats the 95th percentile of its
own phase-permutation null.  On clouds whose design predicts a cycle that is
the *sensitivity*; on clouds whose design predicts no cycle it is a *false
positive*.  The hypothesis under test is:

    the topological cycle matches the cheaper methods on sensitivity while
    producing fewer false positives, because it refuses to answer when there
    is no dominant H1 bar at all.

If that fails — if a cheap baseline is as good on both axes — the honest
conclusion is that the topology earns nothing, and this script will say so.

The competitors
---------------
Seven classical, fixed-transform baselines::

    pc1_order            sort by the first principal coordinate
    pc1_pc2_angle        angle in the PC1-PC2 plane (the classical ordination phase)
    circle_fit_angle     algebraic circle fitted to PC1-PC2, then angle around it
    mds_angle            classical MDS on the distance matrix, angle in top two
    hc_leaf_order        average-linkage leaf order (seriation)
    greedy_tour          nearest-neighbour tour of the distance matrix
    tsp_2opt             2-opt refinement of that tour (a genuine approximate TSP)

All are label-free.  Several are the standard way an analyst reads a phase off
an ordination, and the shortest tour is the reason a topological cycle has to
earn its place: for samples lying on a ring the shortest tour *is* the ring.

Nine *trained* baselines, from ``examples/learned_baselines.py``::

    ae_1d_order          autoencoder, 1-D bottleneck
    ae_2d_angle          autoencoder, 2-D bottleneck
    isomap_angle         Isomap (graph geodesics)
    lle_angle            locally linear embedding
    kernel_pca_angle     RBF kernel PCA
    spectral_angle       Laplacian eigenmaps
    tsne_angle           t-SNE
    som_chain_order      Kohonen self-organising map, chain topology
    principal_curve_order Hastie projection-and-smoothing principal curve

They exist because every classical rival is a *fixed* transform of the cloud,
which invites the reply that the topology only wins because it is weak: a
representation with enough capacity to learn the manifold would find the clock
too.  The learned group answers that objection with a measurement instead of an
argument -- its autoencoder carries ~850 trainable parameters on a 24-sample
cloud, its hyperparameters were tuned on the positive controls alone (so every
specificity number it earns below is out-of-sample), and its implementation is
checked against PCA with the nonlinearity switched off.  See the module
docstring for the full fairness argument.

Run with ``--classical-only`` to reproduce the earlier seven-baseline table.

Controls
--------
The three real "line" designs (fruit development, stress recovery) are
negative controls but there are only three of them, which is not enough to
estimate a false-positive rate.  Synthetic controls are therefore pre-declared
below, built at the same n, feature count and time grid as a real circadian
series so that the scoring is on the same footing, and **drawn several times
with independent noise seeds** so the specificity figure is an estimate rather
than one cloud's accident:

``synth_ring``      a genuine circadian ring            (positive control)
``synth_line``      an unbounded linear ramp             (negative)
``synth_noise``     i.i.d. Gaussian, no structure        (negative)
``synth_clusters``  three separated cell groups          (negative)
``synth_branch``    a bifurcating trajectory             (negative)

Run it with::

    python examples/prepare_datasets.py --outdir data/datasets
    python examples/incremental_value.py --data-root data/datasets

Outputs ``benchmark/incremental_value.json`` and ``docs/fig_incremental_value.png``.
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
from learned_baselines import (  # noqa: E402
    LEARNED_FAMILY,
    LEARNED_METHODS,
    autoencoder_linear_check,
    parameter_counts,
)

from omictopo.cycles import circular_succession_score, dominant_h1_cycles  # noqa: E402
from omictopo.geometry import pairwise_distances, pca_embed, standardize  # noqa: E402
from omictopo.stats import wilson_interval  # noqa: E402

REAL_CLOUDS = [
    "ath_diurnal", "ath_diurnal_heat", "ath_diurnal_ll", "ath_diurnal_ld",
    "rice_diurnal", "tomato_fruit_dev", "ath_recovery",
]

# Synthetic controls, declared before they are generated.  ``positive`` marks
# the one cloud that genuinely contains a clock; everything else must not be
# reported as a cycle by an honest method.
SYNTHETIC = [
    ("synth_ring", "positive", 1.0),
    ("synth_line", "negative", 0.0),
    ("synth_noise", "negative", 0.0),
    ("synth_clusters", "negative", 0.0),
    ("synth_branch", "negative", 0.0),
]

MIN_VISITED = 6          # a "cycle" of fewer samples is not an answer
N_GENES_SYNTH = 3000
N_SAMPLES_SYNTH = 24


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def clock_phase(times: np.ndarray, period: float) -> np.ndarray:
    """Time of day as an angle in ``[0, 2 pi)``."""
    return 2.0 * np.pi * np.mod(np.asarray(times, dtype=float), period) / period


def permutation_null(phases: np.ndarray, order: np.ndarray, observed: float,
                     n_perm: int = 5000, seed: int = 0) -> dict:
    """Null distribution of a succession score: shuffle which sample is which phase.

    The null is deliberately *not* centred on zero.  A cycle has no canonical
    direction and the score takes the better of the two, so a scrambled order
    keeps credit for whichever half of its steps happens to move one way; the
    null therefore sits around 0.4-0.6 and rises as the number of distinct
    phases falls.  Every method is judged against this same null.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([circular_succession_score(rng.permutation(phases), order)
                      for _ in range(int(n_perm))], dtype=float)
    return {
        "n_perm": int(n_perm),
        "median": round(float(np.median(draws)), 4),
        "q95": round(float(np.quantile(draws, 0.95)), 4),
        "p_value": round(float((np.count_nonzero(draws >= observed) + 1)
                               / (int(n_perm) + 1)), 6),
    }


def score_order(phases: np.ndarray, order: np.ndarray, observed: float,
                n_perm: int, seed: int) -> dict:
    """Bundle a method's score, its null, and whether it declares a cycle."""
    order = np.asarray(order, dtype=int)
    visited = int(order.size)
    rec = {
        "score": round(float(observed), 4),
        "n_visited": visited,
        "coverage": round(visited / max(phases.size, 1), 4),
        "null": None,
        "declares": False,
    }
    if visited < MIN_VISITED:
        return rec                      # too short to be an answer
    null = permutation_null(phases, order, observed, n_perm=n_perm, seed=seed)
    rec["null"] = null
    rec["declares"] = bool(null["q95"] < 1.0 and observed > null["q95"])
    return rec


# --------------------------------------------------------------------------
# the methods -- every one of them is label-free
# --------------------------------------------------------------------------
def _mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    """Classical (Torgerson) multidimensional scaling."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    L = np.clip(w[idx], 0.0, None)
    return V[:, idx] * np.sqrt(L)


def m_h1_cycle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """The dominant H1 cycle's vertex order.  Empty when there is no cycle."""
    cycles = dominant_h1_cycles(D=D, rank=1)
    if cycles and cycles[0]["order"]:
        return np.asarray(cycles[0]["order"], dtype=int), {
            "top_bar_rel": round(float(cycles[0]["persistence"])
                                 / float(np.sqrt(np.mean(D ** 2) / 2.0)), 4)}
    return np.array([], dtype=int), {"top_bar_rel": 0.0}


def m_pc1_order(Z: np.ndarray, D: np.ndarray) -> tuple:
    return np.argsort(pca_embed(Z, 1)[:, 0]), {}


def m_pc1_pc2_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    emb = pca_embed(Z, 3)
    x = emb[:, 0] - emb[:, 0].mean()
    y = emb[:, 1] - emb[:, 1].mean()
    return np.argsort(np.arctan2(y, x)), {}


def m_circle_fit_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Fit a circle to the PC1-PC2 scatter, then read the angle around it."""
    emb = pca_embed(Z, 2)
    x, y = emb[:, 0], emb[:, 1]
    A = np.c_[2.0 * x, 2.0 * y, np.ones(x.size)]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy, _ = sol
    return np.argsort(np.arctan2(y - cy, x - cx)), {}


def m_mds_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    emb = _mds(D, 2)
    x, y = emb[:, 0] - emb[:, 0].mean(), emb[:, 1] - emb[:, 1].mean()
    return np.argsort(np.arctan2(y, x)), {}


def m_hc_leaf_order(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Average-linkage seriation: the order in which the dendrogram leaves fall."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(D, checks=False)
    Zl = linkage(condensed, method="average")
    return np.asarray(leaves_list(Zl), dtype=int), {}


def m_greedy_tour(Z: np.ndarray, D: np.ndarray) -> tuple:
    n = D.shape[0]
    used = np.zeros(n, dtype=bool)
    tour = [0]
    used[0] = True
    for _ in range(n - 1):
        remaining = np.flatnonzero(~used)
        tour.append(int(remaining[int(np.argmin(D[tour[-1], remaining]))]))
        used[tour[-1]] = True
    return np.asarray(tour, dtype=int), {}


def _tour_length(D: np.ndarray, order: np.ndarray) -> float:
    o = np.asarray(order, dtype=int)
    return float(D[o, np.roll(o, -1)].sum())


def m_tsp_2opt(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Greedy tour improved by 2-opt until no swap helps (a real TSP heuristic)."""
    order = m_greedy_tour(Z, D)[0].copy()
    n = order.size
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                cand = order.copy()
                cand[i:j] = order[i:j][::-1]
                if _tour_length(D, cand) < _tour_length(D, order) - 1e-12:
                    order = cand
                    improved = True
    return order, {"tour_length": round(_tour_length(D, order), 4)}


CLASSICAL_METHODS = {
    "h1_cycle": m_h1_cycle,
    "pc1_order": m_pc1_order,
    "pc1_pc2_angle": m_pc1_pc2_angle,
    "circle_fit_angle": m_circle_fit_angle,
    "mds_angle": m_mds_angle,
    "hc_leaf_order": m_hc_leaf_order,
    "greedy_tour": m_greedy_tour,
    "tsp_2opt": m_tsp_2opt,
}

# identity of the topological method; everything else is a baseline
TOPOLOGICAL = "h1_cycle"

# ``METHODS`` is the full field the benchmark runs by default.  It stays a
# plain name -> callable dict so that ``run_cloud`` is agnostic about which
# family a method came from.
METHODS = {**CLASSICAL_METHODS, **LEARNED_METHODS}

CLASSICAL_FAMILY = {
    "h1_cycle": "topological",
    "pc1_order": "ordination",
    "pc1_pc2_angle": "ordination",
    "circle_fit_angle": "ordination",
    "mds_angle": "ordination",
    "hc_leaf_order": "seriation",
    "greedy_tour": "tour",
    "tsp_2opt": "tour",
}
FAMILY = {**CLASSICAL_FAMILY, **LEARNED_FAMILY}
LEARNED_NAMES = set(LEARNED_METHODS)


# --------------------------------------------------------------------------
# synthetic controls
# --------------------------------------------------------------------------
def make_synthetic(kind: str, n_genes: int = N_GENES_SYNTH,
                   n_samples: int = N_SAMPLES_SYNTH, noise: float = 1.0,
                   seed: int = 7) -> tuple:
    """A cloud at the same size and time grid as a real circadian series."""
    rng = np.random.default_rng(seed)
    hours = np.linspace(0.0, 23.0, n_samples)
    X = np.zeros((n_samples, n_genes))

    if kind == "synth_ring":
        amp = np.exp(rng.normal(0.0, 0.6, n_genes))
        ph = rng.uniform(0.0, 2.0 * np.pi, n_genes)
        X = np.cos(2.0 * np.pi * hours[:, None] / 24.0 + ph[None, :]) * amp[None, :]
    elif kind == "synth_line":
        slope = rng.normal(0.0, 1.0, n_genes)
        X = (hours[:, None] / 24.0) * slope[None, :]
    elif kind == "synth_noise":
        X = rng.standard_normal((n_samples, n_genes))
    elif kind == "synth_clusters":
        labels = np.arange(n_samples) % 3
        means = rng.normal(0.0, 3.0, (3, n_genes))
        X = means[labels]
    elif kind == "synth_branch":
        labels = (np.arange(n_samples) >= n_samples // 2).astype(int)
        axis = (hours[:, None] / 24.0) * rng.normal(0.0, 1.0, n_genes)[None, :]
        arm = labels[:, None] * rng.normal(0.0, 2.0, n_genes)[None, :]
        X = axis + arm
    else:
        raise ValueError(f"unknown synthetic kind {kind!r}")

    X = X + rng.normal(0.0, noise, X.shape)
    phases = clock_phase(hours, 24.0)
    return X, phases, hours


# --------------------------------------------------------------------------
# one cloud
# --------------------------------------------------------------------------
def run_cloud(label: str, X: np.ndarray, times: np.ndarray, period: float,
              metric: str = "correlation", n_perm: int = 5000,
              seed: int = 0, methods: dict | None = None) -> dict:
    """Score every method on one cloud with the clock held out.

    A method that raises is recorded as having produced no order rather than
    being allowed to end the run.  That is not free for the method: it is
    counted as a failure and reported, so a baseline cannot buy specificity by
    failing on the hard clouds.
    """
    registry = METHODS if methods is None else methods
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    phases = clock_phase(times, float(period or 24.0))

    out = {}
    for name, fn in registry.items():
        t0 = time.time()
        error = ""
        try:
            order, extra = fn(Z, D)
            order = np.asarray(order, dtype=int)
        except Exception as exc:
            order, extra = np.array([], dtype=int), {}
            error = f"{type(exc).__name__}: {exc}"[:160]
        observed = float(circular_succession_score(phases, order)) if order.size else 0.0
        rec = score_order(phases, order, observed, n_perm=n_perm, seed=seed)
        rec.update(extra)
        rec["seconds"] = round(time.time() - t0, 2)
        rec["failed"] = bool(error)
        if error:
            rec["error"] = error
        out[name] = rec
    return {"cloud": label, "n_samples": int(X.shape[0]),
            "n_distinct_times": int(np.unique(np.round(phases, 6)).size),
            "methods": out}


def _abstention_split(clouds: list, name: str) -> dict:
    """Separate *refusing to answer* from *answering and being unconvinced*.

    The project's structural claim is that the H1 cycle stays quiet on a
    non-cyclic cloud by declining to produce an order at all, whereas an
    embedding method always produces one and can only stay quiet by its score
    failing to clear its own null.  Those are different mechanisms and they are
    counted separately here rather than being lumped into one specificity
    figure.  ``margin`` is ``score - q95``: negative means silent, positive
    means it declared, and the largest negative margin says how close a method
    came to declaring on a cloud where it should not.
    """
    no_order = 0
    silent = 0
    declared = 0
    margins: list[float] = []
    for r in clouds:
        m = r["methods"][name]
        if m["null"] is None:
            no_order += 1
            continue
        margin = float(m["score"]) - float(m["null"]["q95"])
        margins.append(margin)
        if m["declares"]:
            declared += 1
        else:
            silent += 1
    return {"no_order": no_order, "silent_by_score": silent,
            "declared": declared, "margins": margins}


def summarise(rows: list, names=None) -> dict:
    """Sensitivity, specificity and balanced accuracy per method, with intervals."""
    pos = [r for r in rows if r["group"] == "positive"]
    neg = [r for r in rows if r["group"] == "negative"]
    names = list(METHODS) if names is None else list(names)
    table = {}
    for name in names:
        tp = sum(1 for r in pos if r["methods"][name]["declares"])
        fp = sum(1 for r in neg if r["methods"][name]["declares"])
        sens = tp / len(pos) if pos else 0.0
        spec = 1.0 - (fp / len(neg) if neg else 0.0)
        lo_s, hi_s = wilson_interval(tp, len(pos))
        lo_p, hi_p = wilson_interval(len(neg) - fp, len(neg))
        every = [r["methods"][name] for r in rows]
        pos_score = [r["methods"][name]["score"] for r in pos]
        neg_score = [r["methods"][name]["score"] for r in neg]
        neg_split = _abstention_split(neg, name)
        pos_split = _abstention_split(pos, name)
        table[name] = {
            "family": FAMILY.get(name, "baseline"),
            "sensitivity": round(sens, 4),
            "sensitivity_ci": [round(lo_s, 4), round(hi_s, 4)],
            "specificity": round(spec, 4),
            "specificity_ci": [round(lo_p, 4), round(hi_p, 4)],
            "balanced_accuracy": round(0.5 * (sens + spec), 4),
            "true_positives": int(tp), "n_positive": len(pos),
            "false_positives": int(fp), "n_negative": len(neg),
            "mean_score_positive": round(float(np.mean(pos_score)), 4) if pos_score else 0.0,
            "mean_score_negative": round(float(np.mean(neg_score)), 4) if neg_score else 0.0,
            "mean_seconds": round(float(np.mean([m["seconds"] for m in every])), 3),
            "method_failures": int(sum(1 for m in every if m["failed"])),
            "failed_clouds": [r["cloud"] for r in rows if r["methods"][name]["failed"]],
            # mechanism of the abstention, negative clouds
            "negatives_no_order": neg_split["no_order"],
            "negatives_silent_by_score": neg_split["silent_by_score"],
            "negatives_declared": neg_split["declared"],
            "negative_margin_worst": (round(max(neg_split["margins"]), 4)
                                      if neg_split["margins"] else None),
            "negative_margin_mean": (round(float(np.mean(neg_split["margins"])), 4)
                                     if neg_split["margins"] else None),
            # and on the clouds where it should answer
            "positives_no_order": pos_split["no_order"],
            "positive_margin_mean": (round(float(np.mean(pos_split["margins"])), 4)
                                     if pos_split["margins"] else None),
            "false_positive_clouds": [r["cloud"] for r in neg
                                      if r["methods"][name]["declares"]],
        }
    return table


FAMILY_COLOUR = {
    "topological": "#c0392b",
    "ordinate": "#2c7fb8",
}
LEARNED_COLOUR_POOL = ["#6a3d9a", "#8c6bb1", "#a679c9", "#5e3c99", "#7b3f9d",
                       "#9467bd", "#6b4e9e", "#8464a8", "#9d7bbd"]


def _family_style(table: dict, names: list) -> dict:
    """One colour per family; the topological method is always red."""
    families = []
    for n in names:
        f = table[n]["family"]
        if f not in families:
            families.append(f)
    learned = [f for f in families if f != "topological" and f not in FAMILY_COLOUR]
    style = {n: FAMILY_COLOUR.get(table[n]["family"], "#2c7fb8") for n in names}
    for i, f in enumerate(learned):
        col = LEARNED_COLOUR_POOL[i % len(LEARNED_COLOUR_POOL)]
        for n in names:
            if table[n]["family"] == f:
                style[n] = col
    return style


def make_figure(table: dict, rows: list, figdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = sorted(table, key=lambda k: (-table[k]["balanced_accuracy"],
                                         -table[k]["specificity"]))
    pretty = [k.replace("_", " ") for k in order]
    sens = [table[k]["sensitivity"] for k in order]
    spec = [table[k]["specificity"] for k in order]
    colour = _family_style(table, order)
    colors = [colour[k] for k in order]

    height = max(5.6, 0.40 * len(order) + 2.7)
    fig, axes = plt.subplots(1, 3, figsize=(18.2, height), layout="constrained",
                             gridspec_kw={"width_ratios": [1.18, 1.00, 0.80]})
    # panels A and C are aligned by construction below; sharing the y-axis here
    # would drag panel B's 0-1 score range onto A's 17 bars and squash them

    ax = axes[0]
    y = np.arange(len(order))
    ax.barh(y + 0.19, sens, height=0.36, color=colors, zorder=3,
            label="sensitivity (cyclic clouds)")
    ax.barh(y - 0.19, spec, height=0.36, color=colors, alpha=0.45, zorder=3,
            label="specificity (non-cyclic clouds)")
    for i, k in enumerate(order):
        ax.text(table[k]["sensitivity"] + 0.015, i + 0.19,
                f"{table[k]['true_positives']}/{table[k]['n_positive']}",
                va="center", fontsize=7.0)
        ax.text(table[k]["specificity"] + 0.015, i - 0.19,
                f"{table[k]['n_negative'] - table[k]['false_positives']}"
                f"/{table[k]['n_negative']}",
                va="center", fontsize=7.0)
    ax.set_yticks(y)
    ax.set_yticklabels(pretty, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.20)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of clouds")
    ax.set_title("A  a cycle must be found and not invented", fontsize=10)
    handles = [Patch(facecolor=colour[TOPOLOGICAL], label="topological"),
               Patch(facecolor="#2c7fb8", label="classical transform")]
    for f in sorted({table[n]["family"] for n in order}):
        if f not in ("topological",) and any(table[n]["family"] == f
                                             and colour[n] != "#2c7fb8" for n in order):
            sample = next(n for n in order if table[n]["family"] == f)
            handles.append(Patch(facecolor=colour[sample], label=f"learned: {f}"))
    ax.legend(handles=handles, fontsize=7.4, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    rivals = {k: v for k, v in table.items() if k != TOPOLOGICAL}
    best_classic = max((k for k in rivals if k not in LEARNED_NAMES),
                       key=lambda k: rivals[k]["balanced_accuracy"], default=None)
    best_learned = max((k for k in rivals if k in LEARNED_NAMES),
                       key=lambda k: rivals[k]["balanced_accuracy"], default=None)
    marks = [(TOPOLOGICAL, "o", "#c0392b")]
    if best_classic:
        marks.append((best_classic, "s", "#2c7fb8"))
    if best_learned:
        marks.append((best_learned, "^", "#6a3d9a"))
    for name, mk, col in marks:
        xs, ys = [], []
        for r in rows:
            m = r["methods"][name]
            if m["null"] is None:
                continue
            xs.append(m["null"]["q95"])
            ys.append(m["score"])
        ax.scatter(xs, ys, marker=mk, s=34, color=col, alpha=0.85,
                   label=name.replace("_", " "), zorder=4)
    lim = [0.0, 1.05]
    ax.plot(lim, lim, "k--", lw=1.0, zorder=2)
    ax.text(0.86, 0.80, "declares\na cycle", fontsize=8, color="#444")
    ax.text(0.30, 0.22, "silent", fontsize=8, color="#444")
    ax.set_xlim(*lim)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("95th pct of the method's own permutation null")
    ax.set_ylabel("succession score")
    ax.set_title("B  where each method sits relative to its own null", fontsize=10)
    ax.legend(fontsize=7.6, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    n_neg = int(table[order[0]]["n_negative"]) or 1
    no_order = np.array([table[k]["negatives_no_order"] for k in order], float)
    silent = np.array([table[k]["negatives_silent_by_score"] for k in order], float)
    declared = np.array([table[k]["negatives_declared"] for k in order], float)
    ax.barh(y, no_order / n_neg, color="#4a4a4a", zorder=3,
            label="no order produced at all")
    ax.barh(y, silent / n_neg, left=no_order / n_neg, color="#9ecae1", zorder=3,
            label="ordered, score below its null")
    ax.barh(y, declared / n_neg, left=(no_order + silent) / n_neg,
            color="#c0392b", zorder=3, label="declared a cycle (false positive)")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(axes[0].get_ylim())      # keep C lined up with A, row for row
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(f"share of the {n_neg} non-cyclic clouds")
    ax.set_title("C  refusing to answer vs. answering\n"
                 "without conviction", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25, axis="x")

    n_learned = len([k for k in order if k in LEARNED_NAMES])
    fig.suptitle("Reading a clock off an expression cloud: the topological cycle "
                 f"against {len(order) - 1 - n_learned} classical and "
                 f"{n_learned} learned baselines", fontsize=11.5)
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "fig_incremental_value.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def _print_cloud(label: str, group: str, rec: dict, note: str = "") -> None:
    m = rec["methods"]
    cells = []
    for name in ("h1_cycle", "pc1_pc2_angle", "greedy_tour", "ae_2d_angle",
                 "tsne_angle"):
        if name not in m:
            continue
        cells.append(f"{name.split('_')[0][:6]}={m[name]['score']:<6}"
                     f"{'D' if m[name]['declares'] else '.'}")
    print(f"  {label:<32} {group:<9} n={rec['n_samples']:<3} "
          + "  ".join(cells) + f"  [{rec.get('seconds', '?')}s]{note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--figdir", default="docs")
    ap.add_argument("--metric", default="correlation")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-synth-seeds", type=int, default=5,
                    help="independent draws of every synthetic control; the "
                         "specificity figure is an estimate and needs more than "
                         "one cloud per control type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--classical-only", action="store_true",
                    help="drop the learned baselines and reproduce the earlier "
                         "seven-baseline comparison")
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(args.data_root)), "benchmark")
    os.makedirs(outdir, exist_ok=True)

    registry = dict(CLASSICAL_METHODS) if args.classical_only else dict(METHODS)

    print("=" * 100)
    print("OmicsTopology - does the topology add anything the cheaper methods cannot?")
    print("=" * 100)
    print("Every method sees the expression matrix only.  The clock is used to "
          "score, never to compute.")
    n_cls = len([k for k in registry if k not in LEARNED_NAMES])
    n_lrn = len(registry) - n_cls
    print(f"field: {n_cls} classical + {n_lrn} learned = {len(registry)} methods\n")

    rows = []
    for key in REAL_CLOUDS:
        spec = SPECS[key]
        X, conditions, times, replicates, _ = load_prepared(
            os.path.join(args.data_root, key))
        for cond in sorted(set(conditions.tolist())):
            mask = conditions == cond
            label = f"{key}:{cond}"
            group = "positive" if spec.expected_topology == "loop" else "negative"
            t0 = time.time()
            rec = run_cloud(label, X[mask], times[mask], spec.period_hours,
                            metric=args.metric, n_perm=args.n_perm,
                            seed=args.seed, methods=registry)
            rec.update(group=group, source="real", accession=spec.accession,
                       organism=spec.organism, expects=spec.expected_topology)
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
            _print_cloud(label, group, rec)

    print(f"\n  synthetic controls (n={N_SAMPLES_SYNTH}, {N_GENES_SYNTH} genes, "
          f"same 24-h grid, {args.n_synth_seeds} independent draws each)")
    for kind, group, expects in SYNTHETIC:
        for s in range(int(args.n_synth_seeds)):
            X, phases, hours = make_synthetic(kind, seed=7 + 101 * s)
            label = kind if s == 0 else f"{kind}#{s}"
            t0 = time.time()
            rec = run_cloud(label, X, hours, 24.0, metric=args.metric,
                            n_perm=args.n_perm, seed=args.seed, methods=registry)
            rec.update(group=group, source="synthetic",
                       expects="loop" if group == "positive" else "line")
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
            _print_cloud(label, group, rec)

    table = summarise(rows, names=registry)
    print("\n" + "-" * 108)
    print(f"{'method':<22}{'family':<18}{'sensitivity':>16}{'specificity':>16}"
          f"{'balanced acc':>14}{'fails':>7}")
    for name in sorted(table, key=lambda k: -table[k]["balanced_accuracy"]):
        t = table[name]
        star = " *" if name == TOPOLOGICAL else "  "
        print(f"{name:<22}{t['family']:<18}"
              f"{t['sensitivity']:>8.3f} [{t['sensitivity_ci'][0]:.2f},"
              f"{t['sensitivity_ci'][1]:.2f}]{t['specificity']:>8.3f} "
              f"[{t['specificity_ci'][0]:.2f},{t['specificity_ci'][1]:.2f}]"
              f"{t['balanced_accuracy']:>12.3f}{t['method_failures']:>7}{star}")
    print("  * topological method; fails = clouds on which the method raised")

    n_neg = len([r for r in rows if r["group"] == "negative"])
    print(f"\n  how each method stays quiet on the {n_neg} non-cyclic clouds")
    print(f"  {'method':<22}{'no order':>10}{'silent by score':>18}"
          f"{'declared':>10}{'worst margin':>14}")
    for name in sorted(table, key=lambda k: -table[k]["negatives_no_order"]):
        t = table[name]
        worst = t["negative_margin_worst"]
        print(f"  {name:<22}{t['negatives_no_order']:>10}"
              f"{t['negatives_silent_by_score']:>18}{t['negatives_declared']:>10}"
              f"{(f'{worst:+.3f}' if worst is not None else 'n/a'):>14}")

    topo = table[TOPOLOGICAL]
    rivals = {k: v for k, v in table.items() if k != TOPOLOGICAL}
    best_rival = max(rivals, key=lambda k: rivals[k]["balanced_accuracy"])
    learned_idx = sorted((k for k in rivals if k in LEARNED_NAMES),
                         key=lambda k: -rivals[k]["balanced_accuracy"])
    classic_idx = sorted((k for k in rivals if k not in LEARNED_NAMES),
                         key=lambda k: -rivals[k]["balanced_accuracy"])
    print(f"\n  strongest classical rival: "
          f"{classic_idx[0]} {rivals[classic_idx[0]]['balanced_accuracy']:.3f} "
          f"(FP {rivals[classic_idx[0]]['false_positives']}/"
          f"{rivals[classic_idx[0]]['n_negative']})")
    if learned_idx:
        print(f"  strongest learned rival:   "
              f"{learned_idx[0]} {rivals[learned_idx[0]]['balanced_accuracy']:.3f} "
              f"(FP {rivals[learned_idx[0]]['false_positives']}/"
              f"{rivals[learned_idx[0]]['n_negative']}, "
              f"params {parameter_counts().get(learned_idx[0], 0)})")
    print(f"  topology:                  {topo['balanced_accuracy']:.3f} "
          f"(FP {topo['false_positives']}/{topo['n_negative']})")

    check = autoencoder_linear_check()
    print(f"\n  autoencoder sanity check (linear network must equal PCA): "
          f"1-D {check['linear_ae_1d_score']} vs {check['pca_1d_score']}, "
          f"2-D {check['linear_ae_2d_score']} vs {check['pca_2d_score']} "
          f"-> {'consistent' if check['linear_ae_matches_pca'] else 'INCONSISTENT'}")

    figures = []
    if not args.no_figures:
        try:
            figures.append(make_figure(table, rows, args.figdir))
        except ImportError:
            print("\nmatplotlib unavailable; figure skipped")

    compare = [TOPOLOGICAL]
    if learned_idx:
        compare.append(learned_idx[0])
    if classic_idx:
        compare.append(classic_idx[0])
    abstention_summary = {
        "note": "how each method stayed quiet on the non-cyclic clouds: "
                "producing no order at all, versus producing one whose score "
                "failed to clear its own permutation null",
        "how_each_method_stayed_quiet": {
            name: {k: table[name][k] for k in
                   ("negatives_no_order", "negatives_silent_by_score",
                    "negatives_declared", "negative_margin_worst",
                    "negative_margin_mean", "positives_no_order",
                    "positive_margin_mean")}
            for name in dict.fromkeys(compare)
        },
    }

    blob = {
        "config": {"metric": args.metric, "n_perm": args.n_perm, "seed": args.seed,
                   "min_visited": MIN_VISITED,
                   "n_synth_seeds": int(args.n_synth_seeds),
                   "classical_only": bool(args.classical_only)},
        "question": "does the topological cycle add sensitivity or specificity "
                    "over label-free baselines that cost less?",
        "methods": list(registry),
        "families": {k: FAMILY.get(k, "baseline") for k in registry},
        "learned_methods": sorted(LEARNED_NAMES & set(registry)),
        "trainable_parameters": {k: v for k, v in parameter_counts().items()
                                 if k in registry},
        "autoencoder_linear_check": check,
        "synthetic_controls_declared": [
            {"cloud": k, "group": g} for k, g, _ in SYNTHETIC],
        "per_cloud": rows,
        "summary": table,
        "headline": {
            "best_rival": best_rival,
            "best_classical": classic_idx[0] if classic_idx else None,
            "best_learned": learned_idx[0] if learned_idx else None,
            "topological": topo,
            "rival": rivals[best_rival],
            "abstention_mechanism": abstention_summary,
        },
        "figures": figures,
    }
    path = os.path.join(outdir, "incremental_value.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(f"\n[increment] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
