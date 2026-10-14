"""Can the cycle count be trusted, and can guiding it help?

Two questions, one script
-------------------------
The measured block-C result (``examples/cycle_counting_real.py``) was 1 of 4
designs counted correctly, with a diagnosed cause: on real time courses the
looping signal sat in a minority of the highest-variance genes, so the H1
spectrum decayed smoothly and no bar was dominant.  Two things follow, and
this script closes both.

**1. An identifiability rule.**  A count is meaningful only when one H1 bar
dominates, so ``gap = longest bar / second longest`` has to be checked before
any integer is reported.  :func:`omictopo.cycles.cycle_identifiability` turns
``(gap, circular Mantel r)`` into a verdict and a sentence for a results
table.

**2. A self-consistent fix, and its audit.**  If the diagnosis is right, then
keeping the genes that sweep around the cycle the data already suggest should
recover it.  :func:`omictopo.cycles.topology_guided_cycle` alternates between
finding the dominant cycle and keeping the first-harmonic genes, using no
clock at all.

The trap, and the test that catches it
--------------------------------------
Selecting genes so that they agree with a cycle you have just found is a way
to fool yourself: the selection is a free parameter and will manufacture
*some* structure.  Three tests are applied, weakest first.

*Bar against a covariance-matched Gaussian null.*  Each draw is a Gaussian
cloud with the observed covariance, and **the whole guided procedure is
re-run inside the null**, so the selection's optimism is present in both
arms.  Necessary but not sufficient — see the control below.

*Recovering the design's phase order, with the clock held out.*  Every case
here has a known cyclic design axis.  The guided procedure never sees it.  The
order it produces is scored against that axis afterwards with
``circular_succession_score``.  This is the claim that actually matters.

*The negative control.*  ``human_forced_desync`` is included although its
design axis is known not to explain its distances (circular Mantel
``r = +0.02`` on these genes; the circadian programme is not in the
top-variance genes of whole blood).  If guiding "recovers" a clock there, the
procedure manufactures cycles and none of its counts may be believed.

Run it with::

    python examples/prepare_cycle_designs.py --outdir data/cycle_designs
    python examples/cycle_identifiability.py --root data/cycle_designs

Outputs ``benchmark/cycle_identifiability.json``.
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

from cycle_counting_real import centre_by_group, collapse  # noqa: E402
from multi_dataset_benchmark import mantel  # noqa: E402
from prepare_cycle_designs import load  # noqa: E402

from omictopo.cycles import (  # noqa: E402
    circular_succession_score,
    cycle_identifiability,
    dominant_h1_cycles,
    h1_bar_lengths,
    topology_guided_cycle,
)
from omictopo.geometry import pairwise_distances, pca_embed, standardize  # noqa: E402

# Pre-registered cases.  The centre mode is chosen so no cloud exceeds a size
# the pure-Python representative-cycle reduction can handle.
CASES = [
    ("psme_annual", "annual_all_samples", "tree", 1),
    ("psme_diurnal", "diurnal_all_samples", "tree", 1),
    ("psme_annual_x_diurnal", "pooled_design_cells", "cell", 2),
    ("human_forced_desync", "fd_phase_cells", "phase", 1),
]
MAX_GUIDED_SAMPLES = 140
CONTROL = "fd_phase_cells"      # the negative control for the guided procedure


# --------------------------------------------------------------------------
# building the cloud
# --------------------------------------------------------------------------
def build_cloud(key: str, root: str, centres: str) -> tuple:
    """Return ``(X, phases_radians, info, note)`` for one design.

    ``axis1_value`` in ``coords.tsv`` is a phase in radians over one full
    period for every design, so it can be scored directly.

    ``centres`` picks what a row of the cloud is:

    ``tree``   every sample, after subtracting its replicate's mean vector;
    ``cell``   one centroid per (replicate, axis1, axis2) design cell;
    ``phase``  one centroid per (axis1, axis2) cell, averaged over replicates,
               which is what keeps a 287-sample protocol cloud small enough to
               find a representative cycle in.
    """
    X, coords, _, info = load(key, root)
    groups = [c["group"] for c in coords]
    axis1 = np.array([c["axis1_value"] for c in coords], dtype=float)
    axis2 = np.array([c["axis2_value"] for c in coords], dtype=float)

    if centres == "tree":
        X = centre_by_group(X, groups)
        phases = axis1
        note = "every sample, replicate means removed"
    elif centres == "cell":
        keys = list(zip(groups, np.round(axis1, 6), np.round(axis2, 6)))
        X, ordered = collapse(X, keys)
        phases = np.array([k[1] for k in ordered], dtype=float)
        note = "one centroid per (replicate, axis1, axis2) cell"
    elif centres == "phase":
        X = centre_by_group(X, groups)
        keys = list(zip(np.round(axis1, 6), np.round(axis2, 6)))
        X, ordered = collapse(X, keys)
        phases = np.array([k[0] for k in ordered], dtype=float)
        note = "replicate means removed, then one centroid per (axis1, axis2) cell"
    else:
        raise ValueError(f"unknown centre mode {centres!r}")
    return X, phases, info, note


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def circular_distance(phases: np.ndarray, period: float = 2.0 * np.pi) -> np.ndarray:
    v = np.mod(np.asarray(phases, dtype=float), period)[:, None]
    d = np.abs(v - v.T)
    return np.minimum(d, period - d)


def plain_diagnostics(X: np.ndarray, phases: np.ndarray, metric: str = "correlation",
                      n_perm: int = 2000) -> dict:
    """The unguided ``(top bar, gap, circular Mantel r)`` triple."""
    Z = standardize(X)
    D = pairwise_distances(Z, metric=metric)
    bars = h1_bar_lengths(D)
    top = float(bars[0]) if bars.size else 0.0
    gap = (float(bars[0] / bars[1])
           if bars.size >= 2 and bars[1] > 1e-12 else float("inf"))
    r, p = mantel(D, circular_distance(phases), n_perm=n_perm, seed=0)
    return {"top_bar_rel": round(top, 4), "gap": round(gap, 4),
            "mantel_r": round(float(r), 4), "mantel_p": round(float(p), 6),
            "top_bars_rel": [round(float(v), 4) for v in bars[:5]]}


def succession(phases: np.ndarray, order, n_perm: int = 5000, seed: int = 0) -> dict:
    """Score a recovered cyclic order against the held-out design phase."""
    order = np.asarray(order, dtype=int)
    if order.size < 3:
        return {"score": 0.0, "n_visited": int(order.size), "coverage": 0.0,
                "null_q95": None, "p_value": None, "passes": False}
    observed = float(circular_succession_score(phases, order))
    rng = np.random.default_rng(seed)
    draws = np.array([circular_succession_score(rng.permutation(phases), order)
                      for _ in range(int(n_perm))], dtype=float)
    q95 = float(np.quantile(draws, 0.95))
    return {
        "score": round(observed, 4),
        "n_visited": int(order.size),
        "coverage": round(order.size / phases.size, 4),
        "null_median": round(float(np.median(draws)), 4),
        "null_q95": round(q95, 4),
        "p_value": round(float((np.count_nonzero(draws >= observed) + 1)
                               / (n_perm + 1)), 6),
        "passes": bool(observed > q95),
    }


def guided_null_top_bar(Z: np.ndarray, n_keep: int, n_iter: int, n_draws: int,
                        seed: int) -> np.ndarray:
    """The guided statistic on covariance-matched Gaussian clouds.

    The entire procedure runs on every draw, so whatever bonus the selection
    confers on the observation it also confers on the null.
    """
    n = Z.shape[0]
    k = int(min(n - 1, Z.shape[1]))
    sd = pca_embed(Z, n_components=k).std(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        pts = rng.standard_normal((n, k)) * sd
        out = topology_guided_cycle(pts, n_keep=n_keep, n_iter=n_iter)
        draws[i] = out["top_bar_rel_selected"]
    return draws


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/cycle_designs")
    ap.add_argument("--outdir", default="benchmark")
    ap.add_argument("--metric", default="correlation")
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-iter", type=int, default=2)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 100)
    print("OmicsTopology - is the count identifiable, and does guiding recover the clock?")
    print("=" * 100)
    print("No clock is used to build a cycle.  The design phase is used only to score.\n")

    results = {}
    for key, label, centres, expected in CASES:
        t0 = time.time()
        X, phases, info, note = build_cloud(key, args.root, centres)
        col = X[:, :min(3000, X.shape[1])]
        n_keep = max(10, int(round(args.keep_frac * col.shape[1])))

        diag = plain_diagnostics(col, phases, metric=args.metric, n_perm=args.n_perm)
        verdict = cycle_identifiability(diag["gap"], diag["mantel_r"])

        unguided_order = np.array([], dtype=int)
        cycles = dominant_h1_cycles(D=pairwise_distances(standardize(col),
                                                         metric=args.metric), rank=1)
        if cycles and cycles[0]["order"]:
            unguided_order = np.asarray(cycles[0]["order"], dtype=int)
        s_un = succession(phases, unguided_order, n_perm=args.n_perm, seed=args.seed)

        guided = None
        s_g = None
        if col.shape[0] <= MAX_GUIDED_SAMPLES:
            g = topology_guided_cycle(col, n_keep=n_keep, n_iter=args.n_iter,
                                      metric=args.metric)
            guided_order = np.asarray(g["order"], dtype=int)
            s_g = succession(phases, guided_order, n_perm=args.n_perm, seed=args.seed)
            null = guided_null_top_bar(standardize(col), n_keep, args.n_iter,
                                       args.n_null, seed=args.seed)
            q95 = float(np.quantile(null, 0.95))
            guided = {
                "n_kept": int(len(g["selected"])),
                "top_bar_rel": round(float(g["top_bar_rel_selected"]), 4),
                "gap_selected": (round(float(g["gap_selected"]), 4)
                                 if g["gap_selected"] is not None else None),
                "null_median": round(float(np.median(null)), 4),
                "null_q95": round(q95, 4),
                "beats_gaussian_null": bool(g["top_bar_rel_selected"] > q95),
            }

        results[label] = {
            "design": key, "accession": info.get("accession"),
            "is_negative_control": bool(label == CONTROL),
            "n_samples": int(col.shape[0]), "n_features": int(col.shape[1]),
            "n_keep": int(n_keep), "expected_cycles": int(expected),
            "how_the_cloud_was_built": note,
            "unguided_diagnostics": diag,
            "verdict": verdict,
            "unguided_succession": s_un,
            "guided": guided,
            "guided_succession": s_g,
            "seconds": round(time.time() - t0, 1),
        }

        su = f"{s_un['score']:.3f}{'P' if s_un['passes'] else '.'}"
        sg = (f"{s_g['score']:.3f}{'P' if s_g['passes'] else '.'}"
              if s_g else "skipped")
        flag = "CONTROL" if label == CONTROL else "       "
        print(f"  {label:<24} {flag} n={col.shape[0]:<4} gap={diag['gap']:<8} "
              f"r={diag['mantel_r']:+.3f} -> {verdict['verdict']:<12} | "
              f"clock recovery: plain={su:<8} guided={sg:<8} "
              f"[{results[label]['seconds']}s]")

    print("\n" + "-" * 100)
    print("1. identifiability rule")
    for label, r in results.items():
        v = r["verdict"]
        print(f"  {label:<24} gap={r['unguided_diagnostics']['gap']:<8} "
              f"r={r['unguided_diagnostics']['mantel_r']:+.3f}  "
              f"-> {v['verdict']:<12} report a count: {v['report_count']}")
        print(f"      {v['advice']}")

    print("\n2. does guiding recover the clock?  (score against the held-out design phase)")
    improved = []
    for label, r in results.items():
        if r["guided_succession"] is None:
            print(f"  {label:<24} guided not attempted")
            continue
        a, b = r["unguided_succession"], r["guided_succession"]
        delta = b["score"] - a["score"]
        if b["passes"] and not a["passes"]:
            improved.append(label)
        print(f"  {label:<24} plain {a['score']:.3f} "
              f"(null95 {a['null_q95']}, n={a['n_visited']:<3})  "
              f"guided {b['score']:.3f} (null95 {b['null_q95']}, n={b['n_visited']:<3})"
              f"  delta {delta:+.3f}")
    print(f"  guided turns a non-detection into a detection in: "
          f"{improved if improved else 'no case'}")

    print("\n3. the control decides whether any of it may be believed")
    ctrl = results[CONTROL]
    if ctrl["guided"] is not None:
        cs = ctrl["guided_succession"]
        print(f"  {CONTROL}: the design axis does not explain these distances "
              f"(mantel r = {ctrl['unguided_diagnostics']['mantel_r']:+.3f}), "
              f"so no clock should be recoverable.")
        print(f"    guided bar beats the Gaussian null: "
              f"{ctrl['guided']['beats_gaussian_null']}")
        print(f"    guided order recovers the design phase: "
              f"score {cs['score']:.3f} vs null95 {cs['null_q95']} -> "
              f"{'PASSES (the procedure is manufacturing cycles)' if cs['passes'] else 'silent (consistent with an honest procedure)'}")
    trusted = not (ctrl["guided"] and ctrl["guided_succession"]["passes"])

    blob = {
        "config": {"metric": args.metric, "n_null": args.n_null,
                   "n_perm": args.n_perm, "n_iter": args.n_iter,
                   "keep_frac": args.keep_frac, "seed": args.seed},
        "question": "when is a beta_1 cycle count worth reporting, and does "
                    "topology-guided gene selection recover a cycle the plain "
                    "spectrum loses?",
        "cases": results,
        "headline": {
            "n_cases": len(results),
            "n_identifiable": sum(1 for r in results.values()
                                  if r["verdict"]["report_count"]),
            "guided_improves_clock_recovery": improved,
            "negative_control": CONTROL,
            "control_shows_manufactured_cycle": bool(
                ctrl["guided"] and ctrl["guided_succession"]["passes"]),
            "guided_procedure_trustworthy": bool(trusted),
        },
    }
    path = os.path.join(args.outdir, "cycle_identifiability.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(f"\n[identifiability] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
