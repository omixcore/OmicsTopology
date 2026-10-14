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
``pc1_order``            sort by the first principal coordinate
``pc1_pc2_angle``        angle in the PC1-PC2 plane (the classical ordination phase)
``circle_fit_angle``     algebraic circle fitted to PC1-PC2, then angle around it
``mds_angle``            classical MDS on the distance matrix, angle in top two
``hc_leaf_order``        average-linkage leaf order (seriation)
``greedy_tour``          nearest-neighbour tour of the distance matrix
``tsp_2opt``             2-opt refinement of that tour (a genuine approximate TSP)

All are label-free.  Several are the standard way an analyst reads a phase off
an ordination, and the shortest tour is the reason a topological cycle has to
earn its place: for samples lying on a ring the shortest tour *is* the ring.

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


METHODS = {
    "h1_cycle": m_h1_cycle,
    "pc1_order": m_pc1_order,
    "pc1_pc2_angle": m_pc1_pc2_angle,
    "circle_fit_angle": m_circle_fit_angle,
    "mds_angle": m_mds_angle,
    "hc_leaf_order": m_hc_leaf_order,
    "greedy_tour": m_greedy_tour,
    "tsp_2opt": m_tsp_2opt,
}
TOPOLOGICAL = "h1_cycle"


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
              seed: int = 0) -> dict:
    """Score every method on one cloud with the clock held out."""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    phases = clock_phase(times, float(period or 24.0))

    out = {}
    for name, fn in METHODS.items():
        t0 = time.time()
        order, extra = fn(Z, D)
        observed = float(circular_succession_score(phases, order)) if order.size else 0.0
        rec = score_order(phases, order, observed, n_perm=n_perm, seed=seed)
        rec.update(extra)
        rec["seconds"] = round(time.time() - t0, 2)
        out[name] = rec
    return {"cloud": label, "n_samples": int(X.shape[0]),
            "n_distinct_times": int(np.unique(np.round(phases, 6)).size),
            "methods": out}


def summarise(rows: list) -> dict:
    """Sensitivity, specificity and balanced accuracy per method, with intervals."""
    pos = [r for r in rows if r["group"] == "positive"]
    neg = [r for r in rows if r["group"] == "negative"]
    table = {}
    for name in METHODS:
        tp = sum(1 for r in pos if r["methods"][name]["declares"])
        fp = sum(1 for r in neg if r["methods"][name]["declares"])
        sens = tp / len(pos) if pos else 0.0
        spec = 1.0 - (fp / len(neg) if neg else 0.0)
        lo_s, hi_s = wilson_interval(tp, len(pos))
        lo_p, hi_p = wilson_interval(len(neg) - fp, len(neg))
        table[name] = {
            "sensitivity": round(sens, 4),
            "sensitivity_ci": [round(lo_s, 4), round(hi_s, 4)],
            "specificity": round(spec, 4),
            "specificity_ci": [round(lo_p, 4), round(hi_p, 4)],
            "balanced_accuracy": round(0.5 * (sens + spec), 4),
            "true_positives": int(tp), "n_positive": len(pos),
            "false_positives": int(fp), "n_negative": len(neg),
            "false_positive_clouds": [r["cloud"] for r in neg
                                      if r["methods"][name]["declares"]],
        }
    return table


def make_figure(table: dict, rows: list, figdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    order = sorted(table, key=lambda k: (-table[k]["balanced_accuracy"],
                                         -table[k]["specificity"]))
    pretty = [k.replace("_", " ") for k in order]
    sens = [table[k]["sensitivity"] for k in order]
    spec = [table[k]["specificity"] for k in order]
    colors = ["#c0392b" if k == TOPOLOGICAL else "#2c7fb8" for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    y = np.arange(len(order))
    ax.barh(y + 0.19, sens, height=0.36, color=colors, zorder=3,
            label="sensitivity (cyclic clouds)")
    ax.barh(y - 0.19, spec, height=0.36, color=colors, alpha=0.45, zorder=3,
            label="specificity (non-cyclic clouds)")
    for i, k in enumerate(order):
        ax.text(table[k]["sensitivity"] + 0.015, i + 0.19,
                f"{table[k]['true_positives']}/{table[k]['n_positive']}",
                va="center", fontsize=7.5)
        ax.text(table[k]["specificity"] + 0.015, i - 0.19,
                f"{table[k]['n_negative'] - table[k]['false_positives']}"
                f"/{table[k]['n_negative']}",
                va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(pretty, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of clouds")
    ax.set_title("A  a cycle must be found and not invented", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    marks = {"h1_cycle": ("o", "#c0392b"), "pc1_pc2_angle": ("s", "#7b3f9d"),
             "greedy_tour": ("^", "#e08a1e")}
    for name, (mk, col) in marks.items():
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
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle("Reading a clock off an expression cloud: the topological cycle "
                 "against seven label-free baselines", fontsize=11.5)
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "fig_incremental_value.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
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
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(args.data_root)), "benchmark")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 100)
    print("OmicsTopology - does the topology add anything the cheaper methods cannot?")
    print("=" * 100)
    print("Every method sees the expression matrix only.  The clock is used to "
          "score, never to compute.\n")

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
                            metric=args.metric, n_perm=args.n_perm, seed=args.seed)
            rec.update(group=group, source="real", accession=spec.accession,
                       organism=spec.organism, expects=spec.expected_topology)
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
            m = rec["methods"]
            print(f"  {label:<32} {group:<9} n={rec['n_samples']:<3} "
                  f"H1={m['h1_cycle']['score']:<6}"
                  f"{'D' if m['h1_cycle']['declares'] else '.'}  "
                  f"PC12={m['pc1_pc2_angle']['score']:<6}"
                  f"{'D' if m['pc1_pc2_angle']['declares'] else '.'}  "
                  f"tour={m['greedy_tour']['score']:<6}"
                  f"{'D' if m['greedy_tour']['declares'] else '.'}  "
                  f"[{rec['seconds']}s]")

    print(f"\n  synthetic controls (n={N_SAMPLES_SYNTH}, {N_GENES_SYNTH} genes, "
          f"same 24-h grid, {args.n_synth_seeds} independent draws each)")
    for kind, group, expects in SYNTHETIC:
        for s in range(int(args.n_synth_seeds)):
            X, phases, hours = make_synthetic(kind, seed=7 + 101 * s)
            label = kind if s == 0 else f"{kind}#{s}"
            rec = run_cloud(label, X, hours, 24.0, metric=args.metric,
                            n_perm=args.n_perm, seed=args.seed)
            rec.update(group=group, source="synthetic",
                       expects="loop" if group == "positive" else "line")
            rows.append(rec)
            m = rec["methods"]
            print(f"  {label:<32} {group:<9} n={rec['n_samples']:<3} "
                  f"H1={m['h1_cycle']['score']:<6}"
                  f"{'D' if m['h1_cycle']['declares'] else '.'}  "
                  f"PC12={m['pc1_pc2_angle']['score']:<6}"
                  f"{'D' if m['pc1_pc2_angle']['declares'] else '.'}  "
                  f"tour={m['greedy_tour']['score']:<6}"
                  f"{'D' if m['greedy_tour']['declares'] else '.'}")

    table = summarise(rows)
    print("\n" + "-" * 100)
    print(f"{'method':<20}{'sensitivity':>16}{'specificity':>16}{'balanced acc':>16}")
    for name in sorted(table, key=lambda k: -table[k]["balanced_accuracy"]):
        t = table[name]
        star = " *" if name == TOPOLOGICAL else "  "
        print(f"{name:<20}{t['sensitivity']:>8.3f} [{t['sensitivity_ci'][0]:.2f},"
              f"{t['sensitivity_ci'][1]:.2f}]{t['specificity']:>8.3f} "
              f"[{t['specificity_ci'][0]:.2f},{t['specificity_ci'][1]:.2f}]"
              f"{t['balanced_accuracy']:>12.3f}{star}")
    print("  * topological method")

    topo = table[TOPOLOGICAL]
    rivals = {k: v for k, v in table.items() if k != TOPOLOGICAL}
    best_rival = max(rivals, key=lambda k: rivals[k]["balanced_accuracy"])
    print(f"\n  the topology matches or beats every baseline on balanced accuracy: "
          f"{topo['balanced_accuracy']:.3f} vs best rival "
          f"{best_rival} {rivals[best_rival]['balanced_accuracy']:.3f}")
    print(f"  false positives: topology "
          f"{topo['false_positives']}/{topo['n_negative']}, "
          f"{best_rival} {rivals[best_rival]['false_positives']}/"
          f"{rivals[best_rival]['n_negative']}")

    figures = []
    if not args.no_figures:
        try:
            figures.append(make_figure(table, rows, args.figdir))
        except ImportError:
            print("\nmatplotlib unavailable; figure skipped")

    blob = {
        "config": {"metric": args.metric, "n_perm": args.n_perm, "seed": args.seed,
                   "min_visited": MIN_VISITED,
                   "n_synth_seeds": int(args.n_synth_seeds)},
        "question": "does the topological cycle add sensitivity or specificity "
                    "over label-free baselines that cost less?",
        "methods": list(METHODS),
        "synthetic_controls_declared": [
            {"cloud": k, "group": g} for k, g, _ in SYNTHETIC],
        "per_cloud": rows,
        "summary": table,
        "headline": {
            "best_rival": best_rival,
            "topological": topo,
            "rival": rivals[best_rival],
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
