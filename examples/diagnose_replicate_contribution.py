"""Diagnosis: what do biological replicates actually contribute?

Context
-------
Two Arabidopsis circadian datasets were run through the benchmark:

* **GSE158444** — 3 biological replicates per time point over one 24 h cycle.
  Called **loop-like**.
* **GSE137732** — **one** sample per time point over a 45 h (two-cycle)
  record.  Called **line-like**, i.e. it "failed".

`diagnose_replicates.py` already showed that *degrading* GSE158444 to one
sample per time point does not remove its loop.  That is necessary but not
sufficient: it shows replicates are not required, not how much they
contribute.  This script answers the contribution question from five
independent angles, so the verdict does not rest on a single manipulation.

A. **Variance decomposition.**  On the dataset that *has* replicates, split
   each gene's variance into between-time and within-time components and
   compute the one-way random-effects ICC.  ICC is the share of variation
   that is signal; ``1 - ICC`` is the share that replication is averaging
   away.  Also report the correlation-distance analogue
   (within-time / between-time distance).

B. **De-noising ceiling.**  Average the 3 replicates per time point into
   centroids.  That cloud has *zero* within-time noise and the same number of
   points as a one-sample-per-time-point draw (n = #times).  If the loop
   survives at the ceiling *and* at n = #times singles, the loop lives in the
   time signal, not in the replicate structure.

C. **Pseudo-replicate rescue.**  Manufacture 3 replicates per time point for
   GSE137732 — which has none — two ways (gene bootstrap, and noise matched
   to GSE158444's own within-time scatter) and re-run the identical pipeline.
   If the whole 45 h record stays line-like *with* replicates, then
   replication cannot be the explanation for its failure.

D. **Noise titration.**  Push GSE158444 through increasing amounts of
   measurement noise (in units of its own within-time SD) and find the noise
   level at which its loop disappears.  That number is an upper bound on how
   much replication could ever have mattered.

E. **Split-half gene reliability.**  Split genes into two disjoint halves and
   correlate the resulting sample-by-sample distance matrices.  This measures
   per-sample reproducibility *without* needing replicates, so the two
   datasets become directly comparable: is GSE137732 noisier per sample?
   (If it is not, replicate absence is doubly implausible.)

Every number is written to ``data/benchmark/diagnosis_replicate_contribution.json``
plus a three-panel figure.

Run::

    python examples/diagnose_replicate_contribution.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_prepared  # noqa: E402
from diagnose_period_drift import cycle_windows  # noqa: E402
from multi_dataset_benchmark import (  # noqa: E402
    covariance_matched_null,
    h1_summary,
    mantel_block,
)

from omictopo.geometry import pairwise_distances, standardize  # noqa: E402

DATA_ROOT = "data/datasets"
WITH_REPS = "ath_diurnal"          # GSE158444 control arm, 3 reps / time point
WITH_REPS_HEAT = "ath_diurnal_heat"
SINGLETON_LL = "ath_diurnal_ll"    # GSE137732, 1 sample / time point
SINGLETON_LD = "ath_diurnal_ld"


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def evaluate(X: np.ndarray, times: np.ndarray, period: float | None = None,
             n_perm: int = 800, seed: int = 0, n_null: int = 0) -> dict:
    """One pipeline pass: H1 summary, optional null, optional Mantel block."""
    Z = standardize(X)
    D = pairwise_distances(Z, metric="correlation")
    hs = h1_summary(D)
    out = {"n": int(X.shape[0]),
           "h1_max_rel": hs["h1_max_rel"],
           "h1_n_bars": hs["h1_n_bars"],
           "beta1_at_mid_loop": hs["beta1_at_mid_loop"]}
    if n_null:
        draws = covariance_matched_null(Z, n_draws=n_null, seed=seed)
        obs = out["h1_max_rel"]
        out["null_q95"] = round(float(np.quantile(draws, 0.95)), 4)
        out["null_p"] = round(float((np.count_nonzero(draws >= obs) + 1)
                                    / (n_null + 1)), 4)
    if period and times.size >= 4:
        mb = mantel_block(X, times, float(period), n_perm=n_perm, seed=seed)
        out.update({"r_linear": mb["r_linear"], "r_circular": mb["r_circular"],
                    "p_circular": mb["p_circular"],
                    "best_period": mb["best_period"],
                    "best_r_circular": mb["best_r_circular"],
                    "circular_advantage": mb["circular_advantage"]})
    return out


def icc_oneway(Z: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """One-way random-effects ICC per gene (column of ``Z``).

    ``ICC = (MSB - MSW) / (MSB + (k0 - 1) MSW)`` with ``k0`` the
    group-size-corrected number of replicates.  ``ICC -> 1`` means the gene is
    pure time signal; ``ICC -> 0`` means its variation is mostly scatter that
    replication is averaging down.
    """
    uniq = sorted(set(groups.tolist()))
    k = len(uniq)
    n = Z.shape[0]
    if k < 2 or n <= k:
        return np.zeros(Z.shape[1])

    grand = Z.mean(axis=0)
    sizes = np.array([float((groups == g).sum()) for g in uniq])
    ss_between = np.zeros(Z.shape[1])
    ss_within = np.zeros(Z.shape[1])
    for g in uniq:
        m = groups == g
        blk = Z[m]
        ss_between += m.sum() * (blk.mean(axis=0) - grand) ** 2
        ss_within += ((blk - blk.mean(axis=0)) ** 2).sum(axis=0)

    msb = ss_between / (k - 1)
    msw = ss_within / max(n - k, 1)
    k0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    denom = msb + (k0 - 1.0) * msw
    icc = np.where(np.abs(denom) > 1e-12,
                   (msb - msw) / np.where(np.abs(denom) > 1e-12, denom, 1.0),
                   0.0)
    return np.clip(icc, -1.0, 1.0)


def within_time_sd(Z: np.ndarray, times: np.ndarray) -> float:
    """Median per-gene SD left over *inside* a time point (standardised units)."""
    resid = []
    for t in sorted(set(times.tolist())):
        m = times == t
        if m.sum() < 2:
            continue
        blk = Z[m]
        resid.append(blk - blk.mean(axis=0))
    if not resid:
        return 0.0
    R = np.vstack(resid)
    return float(np.median(np.sqrt((R ** 2).mean(axis=0))))


def centroid_cloud(X: np.ndarray, times: np.ndarray):
    """Average replicates within each time point -> one point per time."""
    ts = sorted(set(times.tolist()))
    return np.vstack([X[times == t].mean(axis=0) for t in ts]), np.asarray(ts)


def gene_bootstrap(X: np.ndarray, times: np.ndarray, k: int, rng):
    """``k`` pseudo-replicates per sample by resampling genes with replacement.

    Mimics the *structure* of replication (several noisy views of the same
    state) without inventing new biology.
    """
    n, g = X.shape
    rows, tt, rr = [], [], []
    for i in range(n):
        for j in range(k):
            rows.append(X[i, rng.integers(0, g, size=g)])
            tt.append(times[i])
            rr.append(j)
    return np.asarray(rows), np.asarray(tt), np.asarray(rr)


def noise_replicates(X: np.ndarray, times: np.ndarray, k: int, sigma: float,
                     rng):
    """``k`` pseudo-replicates per sample by adding matched Gaussian noise."""
    Z = standardize(X)
    n = Z.shape[0]
    rows, tt, rr = [], [], []
    for i in range(n):
        for j in range(k):
            rows.append(Z[i] + rng.standard_normal(Z.shape[1]) * sigma)
            tt.append(times[i])
            rr.append(j)
    return np.asarray(rows), np.asarray(tt), np.asarray(rr)


def split_half_reliability(X: np.ndarray, n_draws: int = 50, seed: int = 0,
                          metric: str = "correlation") -> dict:
    """Correlate distance matrices built from two disjoint gene halves.

    A replicates-free measure of how reproducible a single sample's geometry
    is.  Works for datasets that have no biological replicates at all, which
    is exactly the case that needs explaining.
    """
    Z = standardize(X)
    g = Z.shape[1]
    if g < 20:
        return {"n_draws": 0}
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(Z.shape[0], 1)
    vals = []
    for _ in range(n_draws):
        perm = rng.permutation(g)
        a, b = perm[: g // 2], perm[g // 2: 2 * (g // 2)]
        Da = pairwise_distances(Z[:, a], metric=metric)[iu]
        Db = pairwise_distances(Z[:, b], metric=metric)[iu]
        if Da.std() > 0 and Db.std() > 0:
            vals.append(float(np.corrcoef(Da, Db)[0, 1]))
    if not vals:
        return {"n_draws": 0}
    vals = np.asarray(vals)
    return {"n_draws": len(vals),
            "median": round(float(np.median(vals)), 4),
            "q25": round(float(np.quantile(vals, 0.25)), 4),
            "q75": round(float(np.quantile(vals, 0.75)), 4)}


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def block_a_variance_decomposition() -> dict:
    """How much of the variation is signal that replication cannot explain?"""
    out = {}
    for key, cond in ((WITH_REPS, "control"), (WITH_REPS_HEAT, "heat")):
        X, conditions, times, _, _ = load_prepared(os.path.join(DATA_ROOT, key))
        m = conditions == cond
        Z = standardize(X[m])
        t = times[m]

        icc = icc_oneway(Z, t)
        fin = icc[np.isfinite(icc)]
        D = pairwise_distances(Z, metric="correlation")
        within, between = [], []
        for tt in sorted(set(t.tolist())):
            idx = np.where(t == tt)[0]
            if idx.size < 2:
                continue
            others = np.setdiff1d(np.arange(Z.shape[0]), idx)
            for i in idx:
                peers = np.setdiff1d(idx, [i])
                within.extend(D[i, peers].tolist())
                between.extend(D[i, others].tolist())

        out[f"{key}:{cond}"] = {
            "n_samples": int(m.sum()),
            "n_times": int(len(set(t.tolist()))),
            "replicates_per_time": int(round(m.sum() / len(set(t.tolist())))),
            "icc_median": round(float(np.median(fin)), 4),
            "icc_q25": round(float(np.quantile(fin, 0.25)), 4),
            "icc_q75": round(float(np.quantile(fin, 0.75)), 4),
            "icc_frac_above_0.5": round(float((fin > 0.5).mean()), 4),
            "within_time_noise_share": round(1.0 - float(np.median(fin)), 4),
            "distance_within_over_between": round(
                float(np.median(within) / np.median(between)), 4),
            "within_time_sd_standardised": round(within_time_sd(Z, t), 4),
        }
    return out


def block_b_denoising_ceiling(n_trials: int = 60, seed: int = 0) -> dict:
    """Full cloud vs replicated-averaged centroids vs one-sample draws."""
    X, conditions, times, _, _ = load_prepared(os.path.join(DATA_ROOT, WITH_REPS))
    m = conditions == "control"
    Xc, tc = X[m], times[m]

    full = evaluate(Xc, tc, period=24.0, n_perm=1500, seed=seed, n_null=200)
    Xcen, tcen = centroid_cloud(Xc, tc)
    centroids = evaluate(Xcen, tcen, period=24.0, n_perm=1500, seed=seed)

    rng = np.random.default_rng(seed)
    draws = []
    for k in range(n_trials):
        idx = np.asarray([int(rng.choice(np.where(tc == tt)[0]))
                          for tt in sorted(set(tc.tolist()))])
        draws.append(evaluate(Xc[idx], tc[idx], period=24.0, n_perm=400,
                              seed=k))
    h1 = np.asarray([d["h1_max_rel"] for d in draws])
    adv = np.asarray([d.get("circular_advantage", np.nan) for d in draws])
    per = np.asarray([d.get("best_period", np.nan) for d in draws])

    return {
        "full_cloud": full,
        "centroids_zero_noise": centroids,
        "single_draws": {
            "n_trials": n_trials,
            "n_per_draw": int(len(set(tc.tolist()))),
            "h1_max_rel_median": round(float(np.median(h1)), 4),
            "h1_max_rel_q25": round(float(np.quantile(h1, 0.25)), 4),
            "h1_max_rel_q75": round(float(np.quantile(h1, 0.75)), 4),
            "circular_advantage_median": round(float(np.nanmedian(adv)), 4),
            "circular_advantage_fraction_positive": round(
                float(np.nanmean(adv > 0)), 4),
            "best_period_median": round(float(np.nanmedian(per)), 2),
        },
    }


def block_c_pseudo_replicate_rescue(sigma_ref: float, k: int = 3,
                                    seed: int = 0) -> dict:
    """Can manufactured replicates rescue GSE137732's verdict?"""
    out = {}
    for key in (SINGLETON_LL, SINGLETON_LD):
        X, conditions, times, _, _ = load_prepared(os.path.join(DATA_ROOT, key))
        cond = sorted(set(conditions.tolist()))[0]
        m = conditions == cond
        Xc, tc = X[m], times[m]

        blocks = {"whole_record": np.ones(tc.size, dtype=bool)}
        for kk, mask in cycle_windows(tc, 24.0):
            blocks[f"cycle_{kk}"] = mask

        rec = {}
        for idx, (name, mask) in enumerate(blocks.items()):
            Xw, tw = Xc[mask], tc[mask]
            if tw.size < 6:
                continue
            entry = {"n_times": int(tw.size)}
            entry["as_is"] = evaluate(Xw, tw, period=24.0, n_perm=1200,
                                      seed=seed, n_null=120)

            # seeds are index-derived, never ``hash()``: the latter is salted
            # per process and would make the run irreproducible
            rng = np.random.default_rng(seed + 101 * (idx + 1))
            Xg, tg, _ = gene_bootstrap(Xw, tw, k, rng)
            entry["gene_bootstrap_k3"] = evaluate(Xg, tg, period=24.0,
                                                  n_perm=1200, seed=seed,
                                                  n_null=120)

            rng = np.random.default_rng(seed + 211 * (idx + 1))
            Xn, tn, _ = noise_replicates(Xw, tw, k, sigma_ref, rng)
            entry["noise_matched_k3"] = evaluate(Xn, tn, period=24.0,
                                                 n_perm=1200, seed=seed,
                                                 n_null=120)
            rec[name] = entry
        out[f"{key}:{cond}"] = rec
    return out


def block_d_noise_titration(sigma_ref: float, seed: int = 0) -> dict:
    """How much noise does GSE158444's loop tolerate before it breaks?"""
    X, conditions, times, _, _ = load_prepared(os.path.join(DATA_ROOT, WITH_REPS))
    m = conditions == "control"
    Xc, tc = X[m], times[m]

    levels = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    rng = np.random.default_rng(seed)
    rows = []
    for lev in levels:
        sigma = lev * sigma_ref
        if sigma == 0:
            Xn = Xc
        else:
            Xn = standardize(Xc) + rng.standard_normal(Xc.shape) * sigma
        r = evaluate(Xn, tc, period=24.0, n_perm=800, seed=seed)
        r["noise_level_rel"] = lev
        r["sigma"] = round(float(sigma), 4)
        rows.append(r)

    break_level = None
    last_ok = None
    for r in rows:
        if r["noise_level_rel"] == 0:
            continue
        if r["circular_advantage"] <= 0 or r["h1_max_rel"] < 0.3:
            break_level = r["noise_level_rel"]
            break
        last_ok = r["noise_level_rel"]

    return {
        "sigma_ref_within_time_sd": round(float(sigma_ref), 4),
        "levels": rows,
        "breakdown_level_rel": break_level,
        "tolerated_level_rel": last_ok,
        "observed_replicate_noise_level_rel": 1.0,
    }


def block_e_split_half_reliability(seed: int = 0) -> dict:
    """Per-sample reproducibility without needing replicates."""
    out = {}
    for key in (WITH_REPS, SINGLETON_LL, SINGLETON_LD):
        X, conditions, times, _, _ = load_prepared(os.path.join(DATA_ROOT, key))
        for cond in sorted(set(conditions.tolist())):
            m = conditions == cond
            out[f"{key}:{cond}"] = split_half_reliability(X[m], seed=seed)
    return out


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def make_figure(a: dict, b: dict, c: dict, d: dict, e: dict, outdir: str) -> str:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#444", "axes.labelcolor": "#222",
        "xtick.color": "#333", "ytick.color": "#333",
        "text.color": "#222", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), layout="constrained")

    # -- panel A: noise titration ------------------------------------------
    ax = axes[0]
    lv = [r["noise_level_rel"] for r in d["levels"]]
    h1 = [r["h1_max_rel"] for r in d["levels"]]
    adv = [r["circular_advantage"] for r in d["levels"]]
    ax.plot(lv, h1, "o-", color="#c0392b", lw=1.6, ms=4.5,
            label="H1 max (RMS-normalised)")
    null95 = b["full_cloud"].get("null_q95")
    if null95 is not None:
        ax.axhline(null95, ls="--", lw=1.0, color="#888",
                   label="covariance-matched null q95")
    ax.set_xlabel("injected noise  (x within-time SD)")
    ax.set_ylabel("H1 max (relative)")
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(lv, adv, "s--", color="#2c7fb8", lw=1.4, ms=4,
             label="circular advantage")
    ax2.axhline(0.0, ls=":", lw=1.0, color="#2c7fb8", alpha=0.6)
    ax2.set_ylabel("circular advantage  $r_{circ}-r_{lin}$", color="#2c7fb8")
    ax2.tick_params(axis="y", colors="#2c7fb8")
    ax2.spines["right"].set_visible(True)
    if d["breakdown_level_rel"] is not None:
        ax.axvline(d["breakdown_level_rel"], ls=":", lw=1.4, color="#e67e22")
        ax.annotate(f"loop breaks\nat {d['breakdown_level_rel']:.0f}x",
                    (d["breakdown_level_rel"], max(h1) * 0.55),
                    fontsize=7.5, color="#b35c00", ha="left",
                    xytext=(6, 0), textcoords="offset points")
    h1l, l1 = ax.get_legend_handles_labels()
    h2l, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1l + h2l, l1 + l2, fontsize=6.8, loc="upper right",
              frameon=True, framealpha=0.92, edgecolor="none")
    ax.set_title("A  GSE158444: how much noise the loop tolerates", fontsize=9.5)

    # -- panel B: pseudo-replicate rescue ----------------------------------
    ax = axes[1]
    short_cond = {"constant light (LL)": "LL", "long day (LD)": "LD"}
    short_block = {"whole_record": "whole", "cycle_0": "cycle 0",
                   "cycle_1": "cycle 1"}
    labels, series = [], []
    for unit, blocks in c.items():
        cond_name = unit.split(":", 1)[1]
        for bname, entry in blocks.items():
            labels.append(f"{short_cond.get(cond_name, cond_name)}\n"
                          f"{short_block.get(bname, bname)}")
            series.append(entry)
    xs = np.arange(len(series))
    w = 0.26
    colors = ["#7f8c8d", "#2c7fb8", "#c0392b"]
    names = ["as_is", "gene_bootstrap_k3", "noise_matched_k3"]
    pretty = ["as is (no reps)", "+3 gene-bootstrap reps", "+3 noise-matched reps"]
    for j, (nm, pn, col) in enumerate(zip(names, pretty, colors)):
        vals = [s[nm].get("circular_advantage", 0.0) for s in series]
        ax.bar(xs + (j - 1) * w, vals, w, color=col, label=pn,
               edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("circular advantage")
    ax.set_ylim(-0.75, 0.55)
    ax.set_title("B  GSE137732: manufacturing replicates does not rescue it",
                 fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left")

    # -- panel C: split-half reliability -----------------------------------
    ax = axes[2]
    keys = list(e.keys())
    vals = np.asarray([e[k].get("median", np.nan) for k in keys])
    los = np.asarray([e[k].get("q25", np.nan) for k in keys])
    his = np.minimum(np.asarray([e[k].get("q75", np.nan) for k in keys]), 1.0)
    cols = ["#c0392b" if "ath_diurnal:" in k else "#2c7fb8" for k in keys]
    ys = np.arange(len(keys))
    ax.barh(ys, vals, color=cols, height=0.55, edgecolor="white")
    ax.errorbar(vals, ys, xerr=[vals - los, his - los],
                fmt="none", ecolor="#555", elinewidth=1.0, capsize=2.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=7)
    ax.set_xlabel("split-half distance-matrix correlation")
    ax.set_xlim(0, 1.0)
    ax.set_title("C  per-sample reliability (no replicates needed)",
                 fontsize=9.5)
    ax.invert_yaxis()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fig_replicate_contribution.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def main(data_root: str = DATA_ROOT, outdir: str = "data/benchmark",
         n_trials: int = 60, seed: int = 0) -> int:
    global DATA_ROOT
    DATA_ROOT = data_root

    print("=" * 78)
    print("replicate-contribution diagnosis")
    print("=" * 78)

    print("\n[A] variance decomposition (dataset that HAS replicates)")
    a = block_a_variance_decomposition()
    for k, v in a.items():
        print(f"  {k:<26} n={v['n_samples']:<3} {v['replicates_per_time']} reps/time"
              f"  ICC median={v['icc_median']:.3f} "
              f"[{v['icc_q25']:.3f}..{v['icc_q75']:.3f}] "
              f"frac>0.5={v['icc_frac_above_0.5']:.2f}"
              f"  within/between dist={v['distance_within_over_between']:.3f}")
    sigma_ref = np.mean([v["within_time_sd_standardised"] for v in a.values()
                         if v["replicates_per_time"] > 1])
    print(f"  -> reference within-time SD (standardised units): {sigma_ref:.4f}")

    print("\n[B] de-noising ceiling: centroids vs singles vs full")
    b = block_b_denoising_ceiling(n_trials=n_trials, seed=seed)
    print(f"  full n={b['full_cloud']['n']:<3} "
          f"H1max_rel={b['full_cloud']['h1_max_rel']:.3f} "
          f"circ_adv={b['full_cloud'].get('circular_advantage', float('nan')):+.3f}")
    print(f"  centroids (zero noise) n={b['centroids_zero_noise']['n']:<3} "
          f"H1max_rel={b['centroids_zero_noise']['h1_max_rel']:.3f} "
          f"circ_adv={b['centroids_zero_noise'].get('circular_advantage', float('nan')):+.3f}")
    sd = b["single_draws"]
    print(f"  singles n={sd['n_per_draw']:<3} "
          f"H1max_rel={sd['h1_max_rel_median']:.3f} "
          f"[{sd['h1_max_rel_q25']:.3f}..{sd['h1_max_rel_q75']:.3f}] "
          f"circ_adv={sd['circular_advantage_median']:+.3f} "
          f"positive {sd['circular_advantage_fraction_positive']*100:.0f}%")

    print("\n[C] pseudo-replicate rescue on GSE137732 (no real replicates)")
    c = block_c_pseudo_replicate_rescue(sigma_ref, k=3, seed=seed)
    for unit, blocks in c.items():
        print(f"  {unit}")
        for bname, entry in blocks.items():
            a0 = entry["as_is"]
            g0 = entry["gene_bootstrap_k3"]
            n0 = entry["noise_matched_k3"]
            print(f"    {bname:<12} n_times={entry['n_times']:<3} | "
                  f"as-is adv={a0.get('circular_advantage', float('nan')):+.3f} "
                  f"p_circ={a0.get('p_circular', float('nan')):.3f} "
                  f"H1={a0['h1_max_rel']:.3f} (p={a0.get('null_p', float('nan')):.3f})"
                  f" || geneboot adv={g0.get('circular_advantage', float('nan')):+.3f}"
                  f" H1={g0['h1_max_rel']:.3f} (p={g0.get('null_p', float('nan')):.3f})"
                  f" || noise adv={n0.get('circular_advantage', float('nan')):+.3f}"
                  f" H1={n0['h1_max_rel']:.3f} (p={n0.get('null_p', float('nan')):.3f})")

    print("\n[D] noise titration on GSE158444 control")
    d = block_d_noise_titration(sigma_ref, seed=seed)
    for r in d["levels"]:
        print(f"  noise x{r['noise_level_rel']:<4} H1max_rel={r['h1_max_rel']:<7} "
              f"circ_adv={r.get('circular_advantage', float('nan')):+.3f} "
              f"best_P={r.get('best_period', float('nan')):.1f}h")
    print(f"  -> loop breaks at {d['breakdown_level_rel']}x the within-time SD"
          if d["breakdown_level_rel"] is not None else
          "  -> loop did not break within the tested range")

    print("\n[E] split-half gene reliability (replicate-free noise probe)")
    e = block_e_split_half_reliability(seed=seed)
    for k, v in e.items():
        if v.get("n_draws"):
            print(f"  {k:<34} median r={v['median']:.3f} "
                  f"[{v['q25']:.3f}..{v['q75']:.3f}]  ({v['n_draws']} splits)")

    report = {
        "description": ("Replicate-contribution diagnosis for the GSE137732 "
                        "failure: variance decomposition, de-noising ceiling, "
                        "pseudo-replicate rescue, noise titration and "
                        "split-half reliability.  Post-hoc, not pre-registered."),
        "reference_within_time_sd_standardised": round(float(sigma_ref), 4),
        "A_variance_decomposition": a,
        "B_denoising_ceiling": b,
        "C_pseudo_replicate_rescue": c,
        "D_noise_titration": d,
        "E_split_half_reliability": e,
    }
    os.makedirs(outdir, exist_ok=True)
    fig = make_figure(a, b, c, d, e, outdir)
    report["figure"] = os.path.basename(fig)
    path = os.path.join(outdir, "diagnosis_replicate_contribution.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n[diagnosis] wrote {path}")
    print(f"[diagnosis] wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
