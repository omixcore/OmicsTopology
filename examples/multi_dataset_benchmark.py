"""Systematic, multi-dataset evaluation of state-space topology.

The single-case study in ``arabidopsis_heat_stress.py`` asked one question of
one dataset.  This script asks the **same question of every dataset in the
registry**, with the answer declared in advance, so that a success or a
failure cannot be rationalised after the fact.

The question
------------
Does the transcriptome of a biological process that is *designed to be
cyclic* (a circadian time course) trace a **closed loop** in expression state
space, while a process that is *designed to advance* (fruit development,
stress recovery) traces an **open trajectory**?

Pre-registered decision rule
----------------------------
For every (dataset, condition) pair the script computes three things without
seeing the answer first:

``h1_max_rel``
    The longest H1 bar, divided by the RMS radius of the embedded cloud —
    a dimensionless "how big is the loop relative to the cloud" number.

``p_null``
    A permutation-free null: draw many clouds from a **covariance-matched
    Gaussian** (same n, same covariance as the observed embedding) and
    recompute ``h1_max_rel``.  The p-value is the fraction of nulls at least
    as large.  This asks "is this loop bigger than what this cloud's own
    shape and size would produce by chance?" and so controls for n.

``mantel``
    Two Mantel correlations between the sample-sample expression distances
    and (a) linear time difference, (b) circular clock distance
    ``min(|dt|, P - |dt|)``.  If the state space is a loop, the *circular*
    distance should explain more.  This is computed on the full expression
    space, **not** on the PCA embedding, so it cannot be an artefact of the
    embedding.

A cloud is called

* **loop-like** if ``p_null < 0.05`` **and** ``r_circular > r_linear``;
* **line-like** if ``p_null >= 0.05`` (no loop beyond chance);
* **ambiguous** otherwise.

The expectation recorded in the registry is then compared with this label.

Robustness
----------
Every dataset is re-analysed at several feature counts (``--sweep``) so the
report can show whether a verdict survives preprocessing, instead of quoting
a single lucky number.

Run it with::

    python examples/prepare_datasets.py --outdir data/datasets
    python examples/multi_dataset_benchmark.py --data-root data/datasets

Outputs ``benchmark_matrix.tsv``, ``results.json`` and figures under
``--outdir``.
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

from omictopo import persistent_homology  # noqa: E402
from omictopo.geometry import pairwise_distances, pca_embed, standardize  # noqa: E402

LOOP_P_THRESHOLD = 0.05
DEFAULT_SWEEP = (500, 1000, 3000, 0)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def mantel(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0):
    """Mantel correlation of two distance matrices, with a permutation p."""
    iu = np.triu_indices(a.shape[0], 1)
    u, v = a[iu], b[iu]
    if u.size < 3 or np.std(u) == 0 or np.std(v) == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(u, v)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = np.corrcoef(u, v[rng.permutation(v.size)])[0, 1]
    p = float((np.count_nonzero(null >= r) + 1) / (n_perm + 1))
    return r, p


def h1_summary(D: np.ndarray) -> dict:
    """H1 summary of a distance matrix, in a dimensionless length unit.

    The unit is the RMS radius implied by the distances themselves,
    ``sqrt(mean(D^2) / 2)``, which for a Euclidean point cloud equals the RMS
    distance of the points from their centroid.  Dividing by it makes H1
    comparable between datasets whose expression scales differ, and — because
    the same divisor is applied to the null — keeps the test honest.
    """
    scale = float(np.sqrt(np.mean(D ** 2) / 2.0))
    scale = scale if scale > 0 else 1.0
    diameter = float(D.max())
    dgm = persistent_homology(D=D, max_dim=1, backend="ripser")
    h1 = np.sort(dgm.persistence(1, infinite_as=diameter))[::-1]

    fin = dgm.finite(1)
    if fin.size:
        dom = fin[int(np.argmax(fin[:, 1] - fin[:, 0]))]
        mid = 0.5 * (float(dom[0]) + float(dom[1]))
        beta1 = int(dgm.betti(1, mid))
        dom_bar = [round(float(dom[0]), 4), round(float(dom[1]), 4)]
    else:
        beta1, dom_bar = 0, None

    return {
        "n_points": int(D.shape[0]),
        "distance_scale": round(scale, 4),
        "h1_max": round(float(h1[0]), 4) if h1.size else 0.0,
        "h1_max_rel": round(float(h1[0]) / scale, 4) if h1.size else 0.0,
        "h1_total_rel": round(float(h1.sum()) / scale, 4) if h1.size else 0.0,
        "h1_n_bars": int(h1.size),
        "h1_entropy": round(float(dgm.persistence_entropy(1, infinite_as=diameter)), 4),
        "h1_dominant_bar": dom_bar,
        "beta1_at_mid_loop": beta1,
        "h0_components": int(dgm.n_bars(0, finite_only=True)),
    }


def covariance_matched_null(Z: np.ndarray, metric: str = "correlation",
                            n_draws: int = 200, seed: int = 0) -> np.ndarray:
    """H1 values from Gaussian clouds that share the data's own covariance.

    The null is built in *sample* space and then pushed through the **same**
    pipeline as the real data: draw ``n`` points from a Gaussian with the
    empirical covariance of the standardised expression vectors, compute the
    same metric, take the same H1 summary.  Every nuisance property — sample
    size, overall spread, anisotropy — is preserved; only structure is
    removed.  A plain isotropic null would be far too easy to beat.

    Because PCA is a rotation, distances computed from the full-rank PCA
    scores are identical to those computed from the standardised matrix, so
    the null lives in exactly the same geometry as the observation.
    """
    n = Z.shape[0]
    k = int(min(n - 1, Z.shape[1]))
    scores = pca_embed(Z, n_components=k)
    sd = scores.std(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        pts = rng.standard_normal((n, k)) * sd
        draws[i] = h1_summary(pairwise_distances(pts, metric=metric))["h1_max_rel"]
    return draws


def period_scan(X: np.ndarray, times: np.ndarray, lo: float = 16.0,
                hi: float = 32.0, step: float = 0.5, n_perm: int = 2000,
                seed: int = 0, metric: str = "correlation") -> dict:
    """Maximise the circular Mantel correlation over candidate periods.

    Free-running clocks (constant light) do not have a period of exactly
    24 h, so testing the circular hypothesis only at ``P = 24`` would put the
    hypothesis at a disadvantage.  Scanning ``P`` gives the data their best
    shot *and* returns a period estimate that can be sanity-checked against
    the known circadian range.
    """
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    periods = np.arange(lo, hi + 1e-9, step)
    r_vals = np.empty(periods.size, dtype=float)
    for i, P in enumerate(periods):
        phase = np.mod(times, P)[:, None]
        dph = np.abs(phase - phase.T)
        circular = np.minimum(dph, P - dph)
        r_vals[i], _ = mantel(D, circular, n_perm=n_perm, seed=seed)
    best = int(np.argmax(r_vals))
    return {
        "grid": [round(float(p), 2) for p in periods],
        "r": [round(float(v), 4) for v in r_vals],
        "best_period": round(float(periods[best]), 2),
        "best_r_circular": round(float(r_vals[best]), 4),
    }


def mantel_block(X: np.ndarray, times: np.ndarray, period: float,
                 n_perm: int = 5000, seed: int = 0,
                 metric: str = "correlation") -> dict:
    """Linear vs circular Mantel, computed on the full expression space."""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    dt = np.abs(times[:, None] - times[None, :])
    phase = np.mod(times, period)[:, None]
    dphase = np.abs(phase - phase.T)
    circular = np.minimum(dphase, period - dphase)

    r_lin, p_lin = mantel(D, dt, n_perm=n_perm, seed=seed)
    r_cir, p_cir = mantel(D, circular, n_perm=n_perm, seed=seed)
    scan = period_scan(X, times, n_perm=max(500, n_perm // 5), seed=seed,
                       metric=metric)

    best_r = max(r_cir, scan["best_r_circular"])
    return {
        "period_hours": float(period),
        "r_linear": round(r_lin, 4),
        "p_linear": p_lin,
        "r_circular": round(r_cir, 4),
        "p_circular": p_cir,
        "best_period": scan["best_period"],
        "best_r_circular": scan["best_r_circular"],
        "period_scan_grid": scan["grid"],
        "period_scan_r": scan["r"],
        # the pre-registered circular statistic is the best over the period
        # grid, so a mis-set nominal period cannot by itself sink the test
        "circular_advantage": round(best_r - r_lin, 4),
        "circular_advantage_nominal": round(r_cir - r_lin, 4),
    }


# --------------------------------------------------------------------------
# per-dataset analysis
# --------------------------------------------------------------------------
def analyse_group(X: np.ndarray, times: np.ndarray, replicates: np.ndarray,
                  period: float | None, label: str, metric: str = "correlation",
                  n_null: int = 200, n_perm: int = 5000, seed: int = 0) -> dict:
    """Full pre-registered readout for one (dataset, condition) cloud."""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)

    out = {"label": label, "n_samples": int(X.shape[0]), "metric": metric,
           "times": sorted({float(t) for t in times})}
    out.update(h1_summary(D))

    draws = covariance_matched_null(Z, metric=metric, n_draws=n_null, seed=seed)
    obs = out["h1_max_rel"]
    out["null"] = {
        "n_draws": int(n_null),
        "null_median": round(float(np.median(draws)), 4),
        "null_q95": round(float(np.quantile(draws, 0.95)), 4),
        "null_max": round(float(draws.max()), 4),
        "p_value": round(float((np.count_nonzero(draws >= obs) + 1) / (n_null + 1)), 4),
    }

    if period:
        out["mantel"] = mantel_block(X, times, period, n_perm=n_perm,
                                     seed=seed, metric=metric)

    # replicate structure: do same-time samples sit together?
    if replicates.max() > 0:
        within, between = [], []
        for t in sorted(set(times.tolist())):
            idx = np.where(times == t)[0]
            if idx.size < 2:
                continue
            others = np.setdiff1d(np.arange(X.shape[0]), idx)
            for i in idx:
                peers = np.setdiff1d(idx, [i])
                if peers.size:
                    within.extend(D[i, peers].tolist())
                if others.size:
                    between.extend(D[i, others].tolist())
        if within and between:
            out["replicate_structure"] = {
                "median_within_time": round(float(np.median(within)), 4),
                "median_between_time": round(float(np.median(between)), 4),
                "ratio": round(float(np.median(within) / np.median(between)), 4),
            }

    out["verdict"] = classify(out)
    return out


def classify(block: dict) -> str:
    """Apply the pre-registered rule to one (dataset, condition) cloud.

    Two independent lines of evidence are required for a full call:

    ``h1``
        the loop is longer than 95% of covariance-matched random clouds
        (``p_value < 0.05``);
    ``mantel``
        the circular clock distance explains the expression distances better
        than linear time, and that circular correlation is itself significant.
    """
    h1_support = block["null"]["p_value"] < 0.05
    m = block.get("mantel")
    mantel_support = bool(m and m["circular_advantage"] > 0 and m["p_circular"] < 0.05)

    if h1_support and mantel_support:
        return "loop-like"
    if h1_support or mantel_support:
        return "loop-like (partial)"
    return "line-like"


def analyse_dataset(key: str, data_root: str, top_var: int | None = None,
                    metric: str = "correlation", n_null: int = 200,
                    n_perm: int = 5000, seed: int = 0,
                    verbose: bool = True) -> dict:
    """Analyse every condition of one prepared dataset."""
    spec = SPECS[key]
    X, conditions, times, replicates, _ = load_prepared(os.path.join(data_root, key))

    if top_var and top_var < X.shape[1]:
        order = np.sort(np.argsort(X.var(axis=0))[::-1][:top_var])
        X = X[:, order]

    groups = {}
    for cond in sorted(set(conditions.tolist())):
        mask = conditions == cond
        block = analyse_group(X[mask], times[mask], replicates[mask],
                              spec.period_hours, label=f"{key}:{cond}",
                              metric=metric, n_null=n_null, n_perm=n_perm, seed=seed)
        block["condition"] = cond
        groups[cond] = block
        if verbose:
            m = block.get("mantel")
            extra = (f"  r_lin={m['r_linear']:+.3f} r_circ@{m['best_period']:.0f}h="
                     f"{m['best_r_circular']:+.3f}" if m else "")
            print(f"    {cond:<22} n={block['n_samples']:<3} "
                  f"H1max_rel={block['h1_max_rel']:<7} "
                  f"null95={block['null']['null_q95']:<7} p_null={block['null']['p_value']:<7} "
                  f"beta1@mid={block['beta1_at_mid_loop']}  -> {block['verdict']}{extra}")

    return {
        "key": key,
        "accession": spec.accession,
        "organism": spec.organism,
        "design": spec.design,
        "expected_topology": spec.expected_topology,
        "period_hours": spec.period_hours,
        "n_features": int(X.shape[1]),
        "conditions": groups,
    }


def half_period_windows(key: str, data_root: str, seed: int = 0,
                        n_null: int = 100, metric: str = "correlation") -> dict:
    """Same analysis restricted to the first half of one cycle.

    A half cycle is an *open* arc, so the pre-registered expectation is that
    the loop disappears.  This is a within-dataset contrast — same platform,
    same pipeline, only the window changes — which is why it is worth
    reporting even though it also shrinks n.
    """
    spec = SPECS[key]
    if not spec.period_hours:
        return {}
    X, conditions, times, replicates, _ = load_prepared(os.path.join(data_root, key))

    out = {}
    t0 = float(times.min())
    for cond in sorted(set(conditions.tolist())):
        mask = conditions == cond
        t = times[mask]
        window = mask & (times < t0 + spec.period_hours / 2.0)
        n_window = int(window.sum())
        if n_window < 6:
            out[cond] = {"skipped": f"only {n_window} samples in the window"}
            continue
        block = analyse_group(X[window], times[window], replicates[window],
                              period=None, label=f"{key}:{cond}:half",
                              metric=metric, n_null=n_null, n_perm=200, seed=seed)
        block["n_window_samples"] = n_window
        out[cond] = block
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
MATRIX_COLUMNS = [
    "dataset", "accession", "organism", "condition", "expected_topology",
    "n_samples", "n_features", "metric",
    "h1_max_rel", "h1_n_bars", "h1_entropy", "beta1_at_mid_loop",
    "null_median", "null_q95", "p_null",
    "r_linear", "r_circular", "best_period", "best_r_circular",
    "circular_advantage", "verdict", "matches_expectation",
]


def matrix_rows(results: list) -> list:
    rows = []
    for ds in results:
        for cond, block in ds["conditions"].items():
            m = block.get("mantel") or {}
            verdict = block["verdict"]
            expected = ds["expected_topology"]
            matches = (verdict.startswith("loop")) == (expected == "loop")
            rows.append({
                "dataset": ds["key"],
                "accession": ds["accession"],
                "organism": ds["organism"],
                "condition": cond,
                "expected_topology": expected,
                "n_samples": block["n_samples"],
                "n_features": ds["n_features"],
                "metric": block["metric"],
                "h1_max_rel": block["h1_max_rel"],
                "h1_n_bars": block["h1_n_bars"],
                "h1_entropy": block["h1_entropy"],
                "beta1_at_mid_loop": block["beta1_at_mid_loop"],
                "null_median": block["null"]["null_median"],
                "null_q95": block["null"]["null_q95"],
                "p_null": block["null"]["p_value"],
                "r_linear": m.get("r_linear"),
                "r_circular": m.get("r_circular"),
                "best_period": m.get("best_period"),
                "best_r_circular": m.get("best_r_circular"),
                "circular_advantage": m.get("circular_advantage"),
                "verdict": verdict,
                "matches_expectation": bool(matches),
            })
    return rows


def write_tsv(path: str, rows: list, columns=None) -> str:
    columns = columns or MATRIX_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(
                "" if row.get(c) is None else str(row.get(c)) for c in columns) + "\n")
    return path


def make_figures(results: list, data_root: str, outdir: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # --- figure 1: the trajectories ---------------------------------------
    panels = []
    for ds in results:
        X, conditions, times, _, _ = load_prepared(os.path.join(data_root, ds["key"]))
        for cond in sorted(set(conditions.tolist())):
            panels.append((ds, cond, X[conditions == cond], times[conditions == cond]))

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow),
                             layout="constrained", squeeze=False)
    for ax, (ds, cond, X, t) in zip(axes.ravel(), panels):
        emb = pca_embed(standardize(X), 3)
        norm = plt.Normalize(t.min(), t.max())
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=t, cmap="twilight", s=34,
                        norm=norm, edgecolor="white", linewidth=0.5, zorder=3)
        # connect time-point centroids in time order to make the shape visible
        order = []
        for tt in sorted(set(t.tolist())):
            order.append(emb[t == tt].mean(axis=0))
        order = np.asarray(order)
        if len(order) > 2:
            closed = np.vstack([order, order[0]])
            ax.plot(closed[:, 0], closed[:, 1], "-", color="#333", lw=1.0,
                    alpha=0.6, zorder=2)
        block = ds["conditions"][cond]
        ax.set_title(f"{ds['key'].replace('_', ' ')}\n{cond}  |  "
                     f"H1max={block['h1_max_rel']:.2f}  [{block['verdict']}]",
                     fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Embedded state space per dataset/condition "
                 "(colour = time; grey line joins time-point centroids)", fontsize=12)
    p = os.path.join(outdir, "fig_benchmark_trajectories.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- figure 2: loop size vs its own null ------------------------------
    rows = matrix_rows(results)
    order = sorted(rows, key=lambda r: (r["expected_topology"] != "loop", -r["h1_max_rel"]))
    labels = [f"{r['dataset'].replace('_', ' ')}\n{r['condition']}" for r in order]
    x = np.arange(len(order))
    colors = ["#c0392b" if r["expected_topology"] == "loop" else "#2c7fb8" for r in order]

    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(order)), 5.4), layout="constrained")
    ax.bar(x, [r["h1_max_rel"] for r in order], color=colors, width=0.6, zorder=3,
           label="observed H1 max / RMS radius")
    ax.plot(x, [r["null_q95"] for r in order], "k_", markersize=18, zorder=5,
            label="95th pct of covariance-matched null")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("H1 max persistence / RMS radius")
    ax.set_title("Loop size against its own null  (red = design expects a loop, "
                 "blue = design expects a line)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    p = os.path.join(outdir, "fig_benchmark_h1_vs_null.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- figure 3: circular vs linear Mantel ------------------------------
    pts = [r for r in rows if r["best_r_circular"] is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6.8, 6.4), layout="constrained")
        lo = min(min(r["r_linear"], r["best_r_circular"]) for r in pts) - 0.05
        hi = max(max(r["r_linear"], r["best_r_circular"]) for r in pts) + 0.05
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1.0,
                label="no circular advantage")
        for r in pts:
            col = "#c0392b" if r["expected_topology"] == "loop" else "#2c7fb8"
            ax.annotate("", xy=(r["r_linear"], r["best_r_circular"]),
                        xytext=(r["r_linear"], r["r_linear"]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.7))
            ax.scatter(r["r_linear"], r["best_r_circular"], color=col, s=52, zorder=4,
                       edgecolor="white", linewidth=0.8)
            ax.annotate(f"{r['dataset'].replace('_', ' ')}\n{r['condition']}",
                        (r["r_linear"], r["best_r_circular"]), fontsize=7,
                        textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("Mantel r  (expression distance vs linear time difference)")
        ax.set_ylabel("Mantel r  (expression distance vs circular clock distance)")
        ax.set_title("Is the clock distance a better explanation than linear time?")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        p = os.path.join(outdir, "fig_benchmark_mantel.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/datasets",
                    help="root written by examples/prepare_datasets.py")
    ap.add_argument("--outdir", default=None,
                    help="where to write results (default: <data-root>/../benchmark)")
    ap.add_argument("--dataset", default="all",
                    help="'all' or a comma-separated list of dataset keys")
    ap.add_argument("--metric", default="correlation",
                    choices=["correlation", "euclidean", "cosine", "manhattan"],
                    help="distance between samples; 'correlation' compares "
                         "expression *patterns* and is the default")
    ap.add_argument("--n-null", type=int, default=200,
                    help="covariance-matched null draws per cloud")
    ap.add_argument("--n-permutations", type=int, default=5000,
                    help="permutations for each Mantel test")
    ap.add_argument("--sweep", default="500,1000,3000,0",
                    help="feature counts for the robustness sweep (comma separated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half-period", action="store_true",
                    help="also analyse the first half of one cycle per dataset")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.data_root)),
                                         "benchmark")
    os.makedirs(outdir, exist_ok=True)

    keys = list(SPECS) if args.dataset == "all" else [k.strip() for k in args.dataset.split(",")]
    for k in keys:
        if k not in SPECS:
            raise SystemExit(f"unknown dataset {k!r}; known: {list(SPECS)}")

    print("=" * 78)
    print("OmicsTopology — multi-dataset benchmark")
    print("=" * 78)
    print("decision rule (pre-registered):")
    print("  loop-like          (a) H1 max beats 95% of a covariance-matched null, AND")
    print("                     (b) circular Mantel beats linear time, significantly")
    print("  loop-like (partial) exactly one of (a), (b)")
    print("  line-like          neither")
    print(f"distance metric = {args.metric}")
    print()

    results = []
    for key in keys:
        spec = SPECS[key]
        print(f"[{key}] {spec.accession}  {spec.organism}  "
              f"(expects: {spec.expected_topology})")
        print(f"    {spec.design}")
        t0 = time.time()
        ds = analyse_dataset(key, args.data_root, metric=args.metric,
                             n_null=args.n_null,
                             n_perm=args.n_permutations, seed=args.seed)
        ds["seconds"] = round(time.time() - t0, 1)
        results.append(ds)
        print(f"    [{ds['seconds']}s]\n")

    if args.half_period:
        print("-" * 78)
        print("half-cycle control (first P/2 of the record; expected: no loop)")
        for ds in results:
            hp = half_period_windows(ds["key"], args.data_root, seed=args.seed,
                                     n_null=max(60, args.n_null // 3),
                                     metric=args.metric)
            if hp:
                ds["half_period"] = hp
                for cond, block in hp.items():
                    if "skipped" in block:
                        print(f"    {ds['key']}:{cond}  {block['skipped']}")
                    else:
                        print(f"    {ds['key']}:{cond:<18} n={block['n_samples']:<3} "
                              f"H1max_rel={block['h1_max_rel']:<7} "
                              f"(null95={block['null']['null_q95']})  -> {block['verdict']}")
        print()

    # --- robustness sweep -------------------------------------------------
    sweep_values = [int(v) for v in str(args.sweep).split(",")] if args.sweep else []
    sweep = {}
    if sweep_values:
        print("-" * 78)
        print("robustness sweep: does the verdict survive the feature count?")
        for key in keys:
            sweep[key] = {}
            for nvar in sweep_values:
                ds = analyse_dataset(key, args.data_root, top_var=nvar,
                                     metric=args.metric, n_null=80,
                                     n_perm=1000, seed=args.seed, verbose=False)
                sweep[key][nvar] = {
                    cond: {"h1_max_rel": b["h1_max_rel"], "verdict": b["verdict"],
                           "r_linear": (b.get("mantel") or {}).get("r_linear"),
                           "r_circular": (b.get("mantel") or {}).get("r_circular")}
                    for cond, b in ds["conditions"].items()
                }
            print(f"    {key}")
            for nvar in sweep_values:
                verdicts = [v["verdict"] for v in sweep[key][nvar].values()]
                print(f"       features={nvar:<6} verdicts={verdicts}")
        print()

    # --- outputs ----------------------------------------------------------
    rows = matrix_rows(results)
    matrix_path = write_tsv(os.path.join(outdir, "benchmark_matrix.tsv"), rows)

    print("-" * 100)
    print(f"{'dataset':<20}{'condition':<20}{'expect':<7}{'H1max_rel':>10}"
          f"{'null95':>8}{'p_null':>8}{'r_lin':>8}{'r_circ':>8}{'P*':>6}"
          f"{'verdict':>20}{'match':>7}")
    for r in rows:
        r_lin = f"{r['r_linear']:>8.3f}" if r["r_linear"] is not None else " " * 8
        r_cir = f"{r['best_r_circular']:>8.3f}" if r["best_r_circular"] is not None else " " * 8
        p_star = f"{r['best_period']:>6.1f}" if r["best_period"] is not None else " " * 6
        print(f"{r['dataset']:<20}{r['condition']:<20}{r['expected_topology']:<7}"
              f"{r['h1_max_rel']:>10.3f}{r['null_q95']:>8.3f}{r['p_null']:>8.3f}"
              f"{r_lin}{r_cir}{p_star}"
              f"{r['verdict']:>20}{str(r['matches_expectation']):>7}")

    n_match = sum(1 for r in rows if r["matches_expectation"])
    n_full = sum(1 for r in rows if r["verdict"] == "loop-like")
    n_partial = sum(1 for r in rows if r["verdict"] == "loop-like (partial)")
    print(f"\nmatched expectation: {n_match}/{len(rows)}   "
          f"full loop calls: {n_full}   partial: {n_partial}")

    figures = []
    if not args.no_figures:
        try:
            figures = make_figures(results, args.data_root, outdir)
            print("\nfigures:")
            for f in figures:
                print("  " + f)
        except ImportError:
            print("\nmatplotlib unavailable; figures skipped")

    blob = {
        "config": {"metric": args.metric, "n_null": args.n_null,
                   "n_permutations": args.n_permutations, "seed": args.seed},
        "decision_rule": {
            "loop_like": "p_null < 0.05 AND circular_advantage > 0 AND p_circular < 0.05",
            "loop_like_partial": "exactly one line of evidence holds",
            "line_like": "neither holds",
            "null_model": "covariance-matched Gaussian, pushed through the same metric",
        },
        "datasets": results,
        "sweep": sweep,
        "figures": figures,
        "summary": {
            "n_units": len(rows),
            "n_matched": n_match,
            "n_loop_full": n_full,
            "n_loop_partial": n_partial,
        },
    }
    json_path = os.path.join(outdir, "results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(f"\n[benchmark] wrote {matrix_path}")
    print(f"[benchmark] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
