"""Real-data case study: is the Arabidopsis circadian state space a loop?

This script takes the public dataset prepared by
:mod:`examples.prepare_plant_dataset` (GEO **GSE158444**, Bonnot & Nagel 2021,
PMID 33871647) and asks a question the package was built for, but on real
data instead of a synthetic cloud:

    *Arabidopsis thaliana* seedlings were harvested at eight times of day
    spanning one circadian period, at 22 °C (control) or after 1 h at 37 °C
    (heat shock).  In expression state space, a functioning circadian clock
    should make the samples trace a **closed loop**, whereas a pure
    time-ordered response would trace a line.  That gives us a topology we
    can state in advance and then test — a ground truth the synthetic
    benchmarks cannot provide.

Five things are measured:

``A`` loop validation     — persistent H1 of each condition, plus a
                            PCA-free Mantel test: is the expression distance
                            between two samples better explained by their
                            *circular* clock distance than by their *linear*
                            time difference?
``B`` replicate structure — do biological replicates land closer to each
                            other than to other times of day?  (H0 sanity)
``C`` two-condition test  — control vs heat, permutation test.
``D`` preprocessing sweep — how much of the above depends on how many
                            features we keep?
``E`` power at matched n  — the same test on synthetic clouds with the same
                            number of points, so a negative result in (C) can
                            be read as "no difference" rather than "no power".

Run it with::

    python examples/prepare_plant_dataset.py --outdir data/arabidopsis_heat
    python examples/arabidopsis_heat_stress.py --data-dir data/arabidopsis_heat

Results are printed and written to ``--outdir`` as ``results.json`` plus
figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from omictopo import persistent_homology, run_states
from omictopo.datasets import two_state_omics
from omictopo.geometry import pca_embed, pairwise_distances, standardize
from omictopo.stats import permutation_test_pointclouds

TREATMENTS = ("control", "heat")


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_prepared(data_dir: str):
    """Load ``matrix.tsv`` + ``design.tsv`` written by prepare_plant_dataset."""
    matrix_path = os.path.join(data_dir, "matrix.tsv")
    design_path = os.path.join(data_dir, "design.tsv")
    if not (os.path.exists(matrix_path) and os.path.exists(design_path)):
        raise SystemExit(
            f"{data_dir} does not contain a prepared dataset.\n"
            "Run:  python examples/prepare_plant_dataset.py --outdir "
            f"{data_dir}"
        )
    with open(matrix_path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")[1:]
        samples, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            samples.append(parts[0])
            rows.append([float(v) for v in parts[1:]])
    groups, hours = [], []
    with open(design_path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            _, group, hour, _ = line.rstrip("\n").split("\t")
            groups.append(group)
            hours.append(int(hour))
    return np.asarray(rows, dtype=float), np.asarray(groups), np.asarray(hours), header


# --------------------------------------------------------------------------
# A. loop validation
# --------------------------------------------------------------------------
def mantel(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0):
    """Mantel correlation between two distance matrices, with a permutation p."""
    iu = np.triu_indices(a.shape[0], 1)
    u, v = a[iu], b[iu]
    r = float(np.corrcoef(u, v)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = np.corrcoef(u, v[rng.permutation(v.size)])[0, 1]
    p = float((np.count_nonzero(null >= r) + 1) / (n_perm + 1))
    return r, p


def loop_validation(X, groups, hours, backend="ripser"):
    """Per-condition embedding, persistent H1, and the PCA-free Mantel test."""
    out = {}
    for g in TREATMENTS:
        m = groups == g
        sub, h = X[m], hours[m].astype(float)

        Z = standardize(sub)
        emb = pca_embed(Z, 3)
        ev = emb.var(axis=0)
        dgm = persistent_homology(emb, max_dim=1, max_simplices=400_000,
                                  backend=backend)
        h1 = np.sort(dgm.persistence(1))[::-1]

        # PCA-free circularity: is expression distance explained better by the
        # distance *around* the clock than by the distance *along* it?
        D = pairwise_distances(Z)
        dt = np.abs(h[:, None] - h[None, :])
        r_lin, p_lin = mantel(D, dt)
        r_cir, p_cir = mantel(D, np.minimum(dt, 24.0 - dt))

        centroids = {}
        for hh in sorted(set(h.tolist())):
            centroids[hh] = emb[h == hh].mean(axis=0)

        out[g] = {
            "n": int(sub.shape[0]),
            "explained_variance": [round(float(v), 4) for v in ev / ev.sum()],
            "h1_max_persistence": round(float(h1[0]) if h1.size else 0.0, 4),
            "h1_spectrum": [round(float(v), 4) for v in h1[:6]],
            "h1_n_bars": int(h1.size),
            "mantel_linear_r": round(r_lin, 4),
            "mantel_linear_p": p_lin,
            "mantel_circular_r": round(r_cir, 4),
            "mantel_circular_p": p_cir,
            "embedding": emb,
            "centroids": centroids,
            "diagram": dgm,
            "distance_vs_circular": (dt, D),
        }
        # beta_1 evaluated at the midpoint of the dominant loop: 1 means
        # "exactly one loop is alive at that scale".  (Rips H1 bars have a
        # non-zero birth, so half the *persistence* is the wrong level.)
        if h1.size:
            fin = dgm.finite(1)
            dom = fin[int(np.argmax(fin[:, 1] - fin[:, 0]))]
            mid = 0.5 * (float(dom[0]) + float(dom[1]))
            beta1 = int(dgm.betti(1, mid))
        else:
            mid, beta1 = 0.0, 0
        out[g]["dominant_bar"] = [round(float(v), 4) for v in dom] if h1.size else None
        out[g]["beta1_at_mid_loop"] = beta1
    return out


# --------------------------------------------------------------------------
# B. replicate structure
# --------------------------------------------------------------------------
def replicate_structure(emb_by_condition, hours, groups):
    out = {}
    for g in TREATMENTS:
        m = groups == g
        emb = emb_by_condition[g]["embedding"]
        h = hours[m]
        D = pairwise_distances(emb)
        within, between = [], []
        for hh in sorted(set(h.tolist())):
            idx = np.where(h == hh)[0]
            others = np.setdiff1d(np.arange(emb.shape[0]), idx)
            for i in idx:
                peers = np.setdiff1d(idx, [i])
                within.extend(D[i, peers].tolist())
                between.extend(D[i, others].tolist())
        out[g] = {
            "median_within_hour": round(float(np.median(within)), 4),
            "median_between_hour": round(float(np.median(between)), 4),
            "ratio": round(float(np.median(within) / np.median(between)), 4),
        }
    return out


# --------------------------------------------------------------------------
# E. power at matched sample size
# --------------------------------------------------------------------------
def power_at_n(sample_sizes, n_datasets=12, n_perm=200, seed0=0):
    """Detection / false-positive rate of the permutation test at a given n."""
    out = []
    for n in sample_sizes:
        hits = {"cycle": 0, "shift": 0}
        for sd in range(n_datasets):
            for kind in ("cycle", "shift"):
                X, g, _ = two_state_omics(n_genes=600, n_per_state=int(n),
                                          state_b=kind, seed=seed0 + sd)
                g = np.asarray(g)
                emb = pca_embed(standardize(X), 3)
                res = permutation_test_pointclouds(
                    emb[g == 0], emb[g == 1], statistic="h1_max",
                    n_permutations=n_perm, backend="ripser", seed=sd,
                )
                if res["p_value"] <= 0.05:
                    hits[kind] += 1
        out.append({
            "n_per_state": int(n),
            "n_datasets": n_datasets,
            "detection_rate_true_loop": hits["cycle"] / n_datasets,
            "false_positive_rate_mean_shift": hits["shift"] / n_datasets,
        })
        print(f"    n={n:>3}: 真环检出 {hits['cycle']}/{n_datasets}"
              f"   假阳性 {hits['shift']}/{n_datasets}")
    return out


# --------------------------------------------------------------------------
# D. preprocessing sensitivity
# --------------------------------------------------------------------------
def sweep_top_var(values, cache, n_perm=300):
    """Re-prepare the dataset at several feature counts and redo the analysis."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    prepare = os.path.join(here, "prepare_plant_dataset.py")
    root = os.path.dirname(os.path.abspath(here))
    rows = []
    for nvar in values:
        out = os.path.join(root, "data", f"_sweep_topvar_{nvar}")
        subprocess.run([sys.executable, prepare, "--outdir", out,
                        "--top-var", str(nvar), "--cache", cache],
                       cwd=here, check=True, capture_output=True)
        Xs, gs, hs, _ = load_prepared(out)
        entry = {"top_var": int(nvar), "n_features": int(Xs.shape[1])}
        for g in TREATMENTS:
            emb = pca_embed(standardize(Xs[gs == g]), 3)
            dgm = persistent_homology(emb, max_dim=1, max_simplices=400_000,
                                      backend="ripser")
            h1 = np.sort(dgm.persistence(1))[::-1]
            entry[f"{g}_h1_max"] = round(float(h1[0]) if h1.size else 0.0, 3)
            entry[f"{g}_h1_bars"] = int(h1.size)
        rep = run_states(Xs, gs, labels=["control", "heat"], max_points=None,
                         max_simplices=400_000, backend="ripser",
                         n_permutations=n_perm, statistic="h1_max", seed=0)
        entry["p_value"] = round(float(rep.test["p_value"]), 4)
        rows.append(entry)
        print(f"  top-var={nvar:<6} features={entry['n_features']:<6} "
              f"control H1max={entry['control_h1_max']:<7} "
              f"heat H1max={entry['heat_h1_max']:<7} "
              f"p={entry['p_value']}")
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(loop, structure, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- figure 1: the loop, both conditions -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
    hours_sorted = sorted(loop[TREATMENTS[0]]["centroids"])
    colors = plt.cm.twilight(np.linspace(0, 1, len(hours_sorted)))
    for ax, g in zip(axes, TREATMENTS):
        info = loop[g]
        emb, cent, hrs = info["embedding"], info["centroids"], info["hours"]
        for col, hh in zip(colors, hours_sorted):
            pts = emb[hrs == hh]
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], s=34, color=col,
                           edgecolor="white", linewidth=0.6, zorder=3)
        present = [hh for hh in hours_sorted if hh in cent]
        ring = np.array([cent[hh] for hh in present] + [cent[present[0]]])
        ax.plot(ring[:, 0], ring[:, 1], "-", color="#333", lw=1.4, zorder=2,
                alpha=0.85)
        for hh in present:
            ax.annotate(f"{hh}h", (cent[hh][0], cent[hh][1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_title(f"{g}  (n={info['n']}, H1 max = {info['h1_max_persistence']:.2f})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)

    # shared colour bar for time of day
    sm = plt.cm.ScalarMappable(cmap="twilight",
                               norm=plt.Normalize(0, len(hours_sorted) - 1))
    cb = fig.colorbar(sm, ax=axes, ticks=range(len(hours_sorted)), pad=0.02)
    cb.ax.set_yticklabels([f"{hh}h" for hh in hours_sorted])
    cb.set_label("time of day")
    fig.suptitle("Arabidopsis circadian state space traces a loop (GSE158444)",
                 fontsize=13)
    p1 = os.path.join(outdir, "fig1_circadian_loop.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 2: circularity is PCA-free -------------------------------
    # A closed loop shows up as a *fold-back*: pairs of samples 18-21 h apart
    # on the clock are as close in expression space as pairs 3-6 h apart,
    # whereas a linear trajectory would keep increasing.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    for ax, g in zip(axes[:2], TREATMENTS):
        dt, D = loop[g]["distance_vs_circular"]
        iu = np.triu_indices(D.shape[0], 1)
        lin = dt[iu]
        circ = np.minimum(dt, 24.0 - dt)[iu]
        d = D[iu]
        sc = ax.scatter(lin, d, c=circ, cmap="twilight", s=16, alpha=0.75,
                        vmin=0, vmax=12)
        # median expression distance per clock-distance bin: turns back down
        bins = np.arange(0.5, 22, 1.0)
        med, ctr = [], []
        for lo in bins:
            m = (circ >= lo - 0.5) & (circ < lo + 0.5)
            if m.sum() >= 3:
                ctr.append(lo)
                med.append(np.median(d[m]))
        ax.plot(ctr, med, "-", color="#c0392b", lw=2.0, zorder=5,
                label="median at given clock distance")
        ax.set_title(g)
        ax.text(0.03, 0.97,
                f"Mantel r\ncircular {loop[g]['mantel_circular_r']:.2f}"
                f"\nlinear   {loop[g]['mantel_linear_r']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.85))
        ax.set_xlabel("linear time difference between samples (h)")
        ax.set_ylabel("expression distance")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=axes[:2], pad=0.015)
    cb.set_label("clock distance min(|Δt|, 24-|Δt|) (h)")

    ax = axes[2]
    for g, col in zip(TREATMENTS, ("#c0392b", "#8e44ad")):
        d = structure[g]
        ax.bar(g, d["median_within_hour"], color=col, width=0.5, label="within hour")
        ax.bar(g, d["median_between_hour"], bottom=d["median_within_hour"],
               color="#bdc3c7", width=0.5, label="gap to other hours")
        ax.text(g, d["median_within_hour"] / 2, f"×{d['ratio']:.2f}",
                ha="center", va="center", color="white", fontsize=9)
    ax.set_ylabel("median pairwise distance")
    ax.set_title("biological replicates cluster")
    ax.legend(fontsize=8)
    p2 = os.path.join(outdir, "fig2_circularity_and_replicates.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data/arabidopsis_heat")
    ap.add_argument("--outdir", default=None,
                    help="where to write results.json and figures "
                         "(default: <data-dir>/results)")
    ap.add_argument("--n-permutations", type=int, default=300)
    ap.add_argument("--power-n", default="12,23,40",
                    help="sample sizes for the matched-n power analysis")
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated --top-var values for the "
                         "preprocessing sensitivity sweep, e.g. 500,1000,3000,0")
    ap.add_argument("--cache", default=None,
                    help="GEO download cache to reuse in the --sweep re-runs")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(args.data_dir, "results")
    os.makedirs(outdir, exist_ok=True)

    X, groups, hours, features = load_prepared(args.data_dir)
    print(f"[case study] {X.shape[0]} samples x {X.shape[1]} features "
          f"({sum(groups == 'control')} control / {sum(groups == 'heat')} heat)")

    print("\n########## A. 状态空间是闭环还是直线？ ##########")
    loop = loop_validation(X, groups, hours)
    for g in TREATMENTS:
        i = loop[g]
        bar = i.get("dominant_bar")
        bar_txt = f"[{bar[0]:.2f}, {bar[1]:.2f}]" if bar else "-"
        print(f"  {g:<8} n={i['n']:<3} H1 max={i['h1_max_persistence']:.2f} "
              f"bars={i['h1_n_bars']} 主导条={bar_txt} "
              f"beta1@中点={i['beta1_at_mid_loop']}")
        print(f"           表达距离 vs 线性时间  r={i['mantel_linear_r']:+.3f} "
              f"(p={i['mantel_linear_p']:.4f})")
        print(f"           表达距离 vs 钟面距离  r={i['mantel_circular_r']:+.3f} "
              f"(p={i['mantel_circular_p']:.4f})")

    # attach the hour vector so the plotting helper can split points
    for g in TREATMENTS:
        loop[g]["hours"] = hours[groups == g]

    print("\n########## B. 生物学重复是否聚集？ ##########")
    structure = replicate_structure(loop, hours, groups)
    for g in TREATMENTS:
        s = structure[g]
        print(f"  {g:<8} 同小时={s['median_within_hour']:.2f}  "
              f"跨小时={s['median_between_hour']:.2f}  比值={s['ratio']:.3f}")

    print(f"\n########## C. 双条件检验 control vs heat ##########")
    report = run_states(X, groups, labels=["control", "heat"],
                        max_points=None, max_simplices=400_000,
                        backend="ripser", n_permutations=args.n_permutations,
                        statistic="h1_max", seed=0)
    print(report.summary_text())

    print("\n########## D. 预处理敏感性（保留多少高变基因）##########")
    sweep = []
    if args.sweep:
        values = [int(v) for v in str(args.sweep).split(",")]
        cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.data_dir)),
                                           "geo_cache")
        sweep = sweep_top_var(values, cache)
    else:
        print(f"  features={X.shape[1]}  H1max control="
              f"{report.summaries['control']['h1_max_persistence']:.3f} "
              f"heat={report.summaries['heat']['h1_max_persistence']:.3f} "
              f"p={report.test['p_value']:.3f}")
        print("  （加 --sweep 500,1000,3000,0 可扫描不同保留数）")

    power = []
    if not args.skip_power:
        print("\n########## E. 同等样本量下的检验功效 ##########")
        sizes = [int(v) for v in str(args.power_n).split(",")]
        power = power_at_n(sizes, n_perm=200)

    files = []
    if not args.no_figures:
        try:
            files = make_figures(loop, structure, outdir)
            print("\n[case study] figures:")
            for f in files:
                print("  " + f)
        except ImportError:
            print("\n[case study] matplotlib unavailable; skipped figures")

    results = {
        "dataset": {
            "accession": "GSE158444",
            "pmid": "33871647",
            "organism": "Arabidopsis thaliana",
            "design": "8 times of day x 24 h, 22C control vs 1 h at 37C heat shock",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "A_loop_validation": {
            g: {k: v for k, v in loop[g].items()
                if k not in ("embedding", "centroids", "diagram",
                             "distance_vs_circular", "hours")}
            for g in TREATMENTS
        },
        "B_replicate_structure": structure,
        "C_two_condition_test": {
            "statistic": report.test.get("statistic"),
            "p_value": float(report.test.get("p_value")),
            "n_permutations": int(report.test.get("n_permutations", 0)),
            "h1_max": {g: float(report.summaries[g]["h1_max_persistence"])
                       for g in TREATMENTS},
            "h1_total": {g: float(report.summaries[g]["h1_total_persistence"])
                         for g in TREATMENTS},
            "shared_threshold": float(report.diagnostics.get("shared_threshold", 0.0)),
        },
        "D_preprocessing_sweep": sweep,
        "E_power_matched_n": power,
        "figures": files,
    }
    json_path = os.path.join(outdir, "results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n[case study] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
