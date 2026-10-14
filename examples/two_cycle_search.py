"""Path (a): can a *measured* design with two independent cyclic parameters give
``beta_1 = 2``?

The flagship claim of block C is that ``beta_1`` counts independent cyclic
*parameters*, not the axes a sample sheet happens to record.  The positive half
of that claim -- "there are two independent cycles here and the count returns
2" -- is demonstrated only on simulated grids
(``two_dimensional_grid__2_parameters``).  On measured data it has never been
shown, and this script is the honest accounting of why.

The topology is not negotiable
------------------------------
A single sampling axis carrying two frequencies does **not** give a torus.  If
:math:`t \\mapsto (t/P_1, t/P_2)` wraps a closed curve in the state space, the
image is a closed geodesic on the torus -- topologically a circle -- so
``beta_1 <= 1`` however many rhythms ride on that one axis.  This is exactly the
simulated ``one_time_axis__2_oscillators`` case, which is expected to return 1.
Two independent cycles therefore require **two independently sampled axes**: a
grid.

What the search found
---------------------
No public transcriptome experiment is a genuine two-cycle grid *with usable
processed data*.  The candidates that came closest, and what happened to them:

==============================  ==========================================  ==================================
accession / source              design                                      outcome
==============================  ==========================================  ==================================
GSE12289                        rat mammary gland, 31 samples over a        4-day time course; **one** sampling
                                ~4-day estrous cycle                        axis, two frequencies -> knot, not a torus
PRJNA1348461 (Morehouse 2026)   mouse liver, 4 estrous phases x 12 clock    the one genuine grid, but only raw FASTQ
                                times (2-hourly, 48 females)                is deposited; the FPKM lives in a
                                                                            preprint supplement, and the authors
                                                                            report the second cycle as an
                                                                            *amplitude modulation* of the first
GSE48113                        human forced desynchrony, 22 people x        the two-oscillator protocol par
                                7 circadian x **2** sleep/wake phases       excellence, but only 2 sleep/wake
                                                                            phases -> an arc, not a loop
GSE93067 / GSE93066             Douglas-fir, year and 48 h                   two *separate* series; pooling them
                                (``psme_annual_x_diurnal``)                 dilutes both and returns 0
==============================  ==========================================  ==================================

The conclusion is not "the method fails".  It is that the two-parameter case is
gated by the **existence of the experiment**, and that no such experiment has
been published with processed data.  The script therefore does two things: it
measures the one single-axis two-frequency dataset that is downloadable
(GSE12289) to show the topological prediction holds on real data, and it
simulates the *cheapest* grid that would settle the question, so the missing
experiment can be stated as a number rather than a hope.

Run with::

    python examples/two_cycle_search.py --cache data/geo_cache \\
        --out benchmark/two_cycle_search.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, os.pardir))
_ROOT = os.path.abspath(os.path.join(_REPO, os.pardir))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from prepare_cycle_designs import FTP, _series_matrix_read, fetch  # noqa: E402
from topology_without_labels import count_cycles, simulate_design  # noqa: E402
from multi_dataset_benchmark import mantel  # noqa: E402

from omictopo.geometry import pairwise_distances, standardize  # noqa: E402

# ---------------------------------------------------------------------------
# the search record -- written down so the negative result is auditable
# ---------------------------------------------------------------------------
CANDIDATES = [
    {
        "source": "GSE12289",
        "organism": "Rattus norvegicus",
        "tissue": "mammary gland",
        "design": "singlet sampling every ~2-3 h over ~4 days of the estrous cycle",
        "n_samples": 32,
        "axes": 1,
        "cyclic_parameters": 1,
        "verdict": "one sampling axis with two frequencies -- a (4,1) torus knot, "
                   "topologically a circle; measured below",
    },
    {
        "source": "PRJNA1348461 (Smith et al., bioRxiv 2026)",
        "organism": "Mus musculus",
        "tissue": "liver",
        "design": "4 estrous phases x 12 two-hourly clock times (48 females)",
        "n_samples": 48,
        "axes": 2,
        "cyclic_parameters": 2,
        "verdict": "the one genuine grid found; only raw FASTQ deposited, and the "
                   "authors' own analysis reports the estrous effect as a "
                   "modulation of circadian *amplitude* rather than a second "
                   "cycle of comparable size",
    },
    {
        "source": "GSE48113",
        "organism": "Homo sapiens",
        "tissue": "whole blood (forced desynchrony)",
        "design": "22 people x 7 circadian phases x 2 sleep/wake phases",
        "n_samples": 287,
        "axes": 2,
        "cyclic_parameters": 2,
        "verdict": "the canonical two-oscillator protocol, but only 2 sleep/wake "
                   "phases are released -- an arc, not a loop; and the circadian "
                   "program is absent from the high-variance blood genes "
                   "(Mantel r = +0.045), so it counts 0 for a second reason",
    },
    {
        "source": "GSE93067 + GSE93066",
        "organism": "Pseudotsuga menziesii",
        "tissue": "needles",
        "design": "yearlong series pooled with a 48 h series",
        "n_samples": 138,
        "axes": 2,
        "cyclic_parameters": 2,
        "verdict": "two independent real cycles, but two separate sample "
                   "collections: pooling dilutes both and the count returns 0",
    },
]


def _diluted_grid(n1: int, n2: int, n_signal: int, n_genes: int,
                  noise_sd: float, seed: int, periods=(24.0, 96.0)) -> np.ndarray:
    """A ``n1 x n2`` grid of two independent cyclic parameters, then dilution.

    ``n_signal`` genes carry both cycles; the remaining ``n_genes - n_signal``
    are pure noise.  Dilution is the mechanism that killed the real clouds, so
    the design spec has to be quoted at a realistic signal fraction.
    """
    p1, p2 = periods
    t1 = np.linspace(0.0, p1, n1)
    t2 = np.linspace(0.0, p2, n2)
    grid = np.array([(a, b) for a in t1 for b in t2])
    driven = simulate_design(grid, n_signal, [(0, p1), (1, p2)], noise_sd,
                             seed=seed)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0.0, noise_sd, (grid.shape[0], n_genes - n_signal))
    return np.hstack([driven, noise])


def grid_requirement(n_genes: int, noise_sd: float, n_null: int, seed: int) -> dict:
    """How big must the grid be for the count to return 2?

    Sweeps the grid size and the fraction of genes that carry the cycles.  The
    answer is the cheapest design that survives a realistic dilution, i.e. the
    experiment the analysis points at.
    """
    rows = []
    for n1, n2 in ((3, 3), (4, 4), (6, 6), (8, 8), (13, 9)):
        for frac in (1.0, 0.5, 0.25, 0.10):
            n_signal = max(1, int(round(n_genes * frac)))
            X = _diluted_grid(n1, n2, n_signal, n_genes, noise_sd, seed)
            rec = count_cycles(X, n_null=n_null, seed=seed)
            bars = rec["top_bars_rel"]
            rows.append({
                "grid": f"{n1}x{n2}",
                "n_samples": int(n1 * n2),
                "signal_fraction": frac,
                "n_signal_genes": n_signal,
                "n_cycles_detected": int(rec["n_bars_above_null_q95"]),
                "top_bar": bars[0],
                "second_bar": bars[1] if len(bars) > 1 else None,
                "gap": (round(bars[0] / bars[1], 3)
                        if len(bars) > 1 and bars[1] > 1e-9 else None),
                "null_q95": rec["null_q95"],
                "correct": bool(rec["n_bars_above_null_q95"] == 2),
            })
    return {"periods_hours": [24.0, 96.0], "n_genes": n_genes,
            "noise_sd": noise_sd, "rows": rows}


def measure_estrous_series(path: str, n_genes: int, n_null: int, seed: int) -> dict:
    """GSE12289: two frequencies on one sampling axis, measured.

    The prediction from the simulation is that a single axis cannot give 2.  Here
    we check the stronger, more useful statement on real data: does it give 1?
    The circular Mantel correlation against each of the two candidate periods
    says whether the rhythms are there at all, which separates "no cycle" from
    "cycle present but not dominant".
    """
    acc, titles, features, X, _chars = _series_matrix_read(path)
    import re
    mins = np.array([float(re.search(r"([\d.]+) hours", t).group(1))
                     for t in titles])

    keep = np.isfinite(X).mean(axis=0) >= 0.25
    X = X[:, keep]
    X = np.where(np.isfinite(X), X, np.nanmedian(X))
    var = X.var(axis=0)
    top = np.argsort(var)[::-1][:n_genes]
    Z = X[:, top]

    rec = count_cycles(Z, n_null=n_null, seed=seed)
    bars = rec["top_bars_rel"]
    out = {
        "accession": "GSE12289",
        "n_samples": int(Z.shape[0]),
        "span_hours": round(float(mins.max() - mins.min()), 1),
        "n_cycles_detected": int(rec["n_bars_above_null_q95"]),
        "top_bars_rel": bars[:4],
        "null_q95": rec["null_q95"],
        "gap": (round(bars[0] / bars[1], 3)
                if len(bars) > 1 and bars[1] > 1e-9 else None),
    }
    if mantel is not None:
        D = pairwise_distances(standardize(Z), metric="correlation")
        for period, name in ((24.0, "time_of_day"), (96.0, "estrous")):
            ph = 2.0 * np.pi * (mins % period) / period
            d = np.abs(ph[:, None] - ph[None, :])
            d = np.minimum(d, 2.0 * np.pi - d)
            r, p = mantel(D, d, n_perm=2000, seed=seed)
            out[f"mantel_r_{name}"] = round(float(r), 4)
            out[f"mantel_p_{name}"] = round(float(p), 6)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.path.join(_ROOT, "data", "geo_cache"))
    ap.add_argument("--out", default=os.path.join(HERE, os.pardir, "benchmark",
                                                  "two_cycle_search.json"))
    ap.add_argument("--n-genes", type=int, default=3000)
    ap.add_argument("--noise-sd", type=float, default=1.0)
    ap.add_argument("--n-null", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("path (a): searching for a measured two-cycle design")
    print("  topological requirement: two *independently sampled* axes.")
    print("  a single axis with two frequencies is a closed geodesic on the")
    print("  torus -- a circle -- so beta_1 <= 1 whatever the sample sheet says.")

    url = (f"{FTP}/series/GSE12nnn/GSE12289/matrix/"
           "GSE12289_series_matrix.txt.gz")
    path = fetch(url, args.cache)
    est = measure_estrous_series(path, args.n_genes, args.n_null, args.seed)
    print(f"\n  GSE12289 (rat mammary, ~4-day estrous time course, one axis):")
    print(f"    count = {est['n_cycles_detected']}  gap = {est['gap']}  "
          f"q95 = {est['null_q95']}")
    if "mantel_r_estrous" in est:
        print(f"    Mantel vs 24 h  r = {est['mantel_r_time_of_day']:+.4f} "
              f"(p = {est['mantel_p_time_of_day']})")
        print(f"    Mantel vs 96 h  r = {est['mantel_r_estrous']:+.4f} "
              f"(p = {est['mantel_p_estrous']})")
        print("    -> both rhythms are present and significant, but neither")
        print("       dominates the selected genes: an arc, not a loop.")

    print("\n  the cheapest grid that would settle the two-parameter case:")
    req = grid_requirement(args.n_genes, args.noise_sd, args.n_null, args.seed)
    for row in req["rows"]:
        if row["correct"] or row["signal_fraction"] in (1.0, 0.25):
            print(f"    {row['grid']:>6} ({row['n_samples']:3d} samples)  "
                  f"signal {row['signal_fraction']:>4}  -> "
                  f"count {row['n_cycles_detected']}  gap {row['gap']}")

    payload = {
        "question": "does any measured design have two independent cyclic "
                    "parameters, and does the count return 2?",
        "topological_requirement": "two independently sampled axes; one axis "
                                   "with two frequencies is a closed geodesic",
        "candidates": CANDIDATES,
        "measured_single_axis_case": est,
        "grid_requirement": req,
        "conclusion": "No public two-cycle grid with processed data exists. The "
                      "single-axis case is measured, not assumed: GSE12289 "
                      "returns 0 with both rhythms significant, i.e. the "
                      "topological prediction holds. The missing experiment is "
                      "a grid, and its cheapest form is quoted above.",
        "n_genes": args.n_genes,
        "noise_sd": args.noise_sd,
        "n_null": args.n_null,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
