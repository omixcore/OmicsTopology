"""When does counting cycles actually work?

``examples/cycle_counting_real.py`` runs block C's counting pipeline on four
measured designs and finds that only the densest one is counted correctly.  A
negative result is not useful on its own; what a reader needs is the condition
that separates the working case from the failing ones.

That condition is not the number of phases, and not the number of samples.  It
is **how large a share of the selected features actually carries the cycle**.

The pipeline standardises every gene and then measures the correlation distance
between whole transcriptomes.  If only a minority of the 3000 selected genes
follow the ring, the pairwise distances are dominated by the unrelated
majority, and the loop becomes a small perturbation of an almost-random cloud.
Simulation puts a number on it: this script sows a circadian programme into
``n_signal`` of ``p`` genes, leaves the rest unrelated, and sweeps ``n_signal``.

Two statistics are recorded for every cloud:

* the circular Mantel correlation ``r`` (which knows the phase, so it is the
  *upper bound* a phase-based method could reach), and
* the dominant-bar **gap** ``bar[0] / bar[1]``, the separation the counting
  criterion actually consumes.

The counting criterion needs a dominant bar, so it needs a gap far above one.
Both quantities are plotted against ``n_signal``; the four measured clouds are
placed on the same axes by their own gap and Mantel r.  The result is a
pre-registered, quantitative statement of the method's operating envelope
rather than a post-hoc excuse for the failures.

Usage::

    python examples/cycle_counting_power.py --outdir benchmark --figdir docs

Outputs ``benchmark/cycle_counting_power.json`` and
``docs/fig_cycle_counting_power.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from multi_dataset_benchmark import mantel  # noqa: E402
from topology_without_labels import count_cycles  # noqa: E402

from omictopo.cycles import h1_bar_lengths  # noqa: E402
from omictopo.geometry import pairwise_distances, standardize  # noqa: E402

P_TOTAL = 3000
N_SIGNAL_GRID = (30, 100, 300, 1000, 3000)
PHASE_GRID = (6, 16)
NOISE_SD = 1.0


def circular_distance(a: np.ndarray, period: float = 2.0 * np.pi) -> np.ndarray:
    d = np.abs(np.asarray(a, float)[:, None] - np.asarray(a, float)[None, :])
    return np.minimum(d, period - d)


def sow_ring(n_phases: int, n_reps: int, n_signal: int, p: int = P_TOTAL,
             noise_sd: float = NOISE_SD, seed: int = 0):
    """A ring sampled at ``n_phases`` positions, carried by ``n_signal`` genes."""
    rng = np.random.default_rng(seed)
    phases = np.tile(np.arange(n_phases) * (2.0 * np.pi / n_phases), n_reps)
    n = phases.size
    amps = np.exp(rng.normal(0.0, 1.0, n_signal))
    gene_phase = rng.uniform(0.0, 2.0 * np.pi, n_signal)
    X = np.zeros((n, p))
    X[:, :n_signal] = (np.cos(phases[:, None] + gene_phase[None, :])
                       * amps[None, :])
    X += rng.normal(0.0, noise_sd, (n, p))
    return X, phases


def measure(X: np.ndarray, phases: np.ndarray, n_null: int, seed: int) -> dict:
    Z = standardize(X)
    D = pairwise_distances(Z, metric="correlation")
    bars = h1_bar_lengths(D)
    top = float(bars[0]) if bars.size else 0.0
    second = float(bars[1]) if bars.size > 1 else 0.0
    gap = top / second if second > 1e-9 else float("inf")
    r, p = mantel(D, circular_distance(phases), n_perm=500, seed=seed)
    rec = count_cycles(X, n_null=n_null, seed=seed)
    return {
        "mantel_r": round(float(r), 4),
        "mantel_p": float(p),
        "top_bar": round(top, 4),
        "gap": round(float(gap), 4) if np.isfinite(gap) else None,
        "n_cycles": int(rec["n_bars_above_null_q95"]),
        "null_q95": rec["null_q95"],
    }


def sweep(n_null: int, n_seeds: int, seed0: int) -> dict:
    out: dict = {}
    for n_phases in PHASE_GRID:
        n_reps = max(1, round(72 / n_phases))
        for n_signal in N_SIGNAL_GRID:
            runs = []
            for s in range(n_seeds):
                X, ph = sow_ring(n_phases, n_reps, n_signal, seed=seed0 + 100 * s)
                runs.append(measure(X, ph, n_null, seed=seed0 + s))
            gaps = [r_["gap"] for r_ in runs if r_["gap"] is not None]
            key = f"phases{n_phases}__signal{n_signal}"
            out[key] = {
                "n_phases": n_phases,
                "n_signal": n_signal,
                "n_samples": int(n_reps * n_phases),
                "n_seeds": n_seeds,
                "mantel_r_median": round(float(np.median(
                    [r_["mantel_r"] for r_ in runs])), 4),
                "gap_median": round(float(np.median(gaps)), 4) if gaps else None,
                "n_cycles_median": float(np.median(
                    [r_["n_cycles"] for r_ in runs])),
                "fraction_counting_one": round(float(np.mean(
                    [r_["n_cycles"] == 1 for r_ in runs])), 4),
                "runs": runs,
            }
    return out


def real_operating_points() -> list:
    """The four measured clouds, for overlay.  Values from cycle_counting_real."""
    return [
        {"cloud": "psme_annual", "distinct_phases": 16,
         "mantel_r": 0.7007, "gap": 4.93, "n_cycles": 1, "expected": 1},
        {"cloud": "psme_diurnal", "distinct_phases": 6,
         "mantel_r": 0.7010, "gap": 1.36, "n_cycles": 0, "expected": 1},
        {"cloud": "human_forced_desync", "distinct_phases": 7,
         "mantel_r": 0.0448, "gap": 1.13, "n_cycles": 0, "expected": 1},
        {"cloud": "psme_annual_x_diurnal", "distinct_phases": None,
         "mantel_r": None, "gap": 1.06, "n_cycles": 0, "expected": 2},
    ]


def make_figure(sweep_out: dict, outdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

    colors = {6: "#c0504d", 16: "#1f5fa9"}
    ax = axes[0]
    for n_phases in PHASE_GRID:
        keys = [k for k in sweep_out if sweep_out[k]["n_phases"] == n_phases]
        keys.sort(key=lambda k: sweep_out[k]["n_signal"])
        xs = [sweep_out[k]["n_signal"] for k in keys]
        ys = [sweep_out[k]["gap_median"] for k in keys]
        ax.plot(xs, ys, "o-", color=colors[n_phases],
                label=f"{n_phases} distinct phases")
    ax.axhline(1.0, color="#888888", ls=":", lw=1.0)
    ax.text(35, 1.08, "no dominant bar", color="#666666", fontsize=7.5)
    for pt in real_operating_points():
        if pt["gap"] is None:
            continue
        ax.axhline(pt["gap"], color="#e08a1e", ls="--", lw=0.7, alpha=0.7)
    ax.annotate("measured clouds\n(gap 1.06 - 1.36)", xy=(60, 1.2),
                xytext=(60, 2.6), fontsize=7.5, color="#b0700f",
                arrowprops=dict(arrowstyle="->", color="#b0700f", lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("number of the 3000 selected genes that carry the ring")
    ax.set_ylabel(r"dominant-bar gap  $bar_1 / bar_2$")
    ax.set_title("A. the counting criterion needs a dominant bar", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for n_phases in PHASE_GRID:
        keys = [k for k in sweep_out if sweep_out[k]["n_phases"] == n_phases]
        keys.sort(key=lambda k: sweep_out[k]["mantel_r_median"])
        xs = [sweep_out[k]["mantel_r_median"] for k in keys]
        ys = [sweep_out[k]["fraction_counting_one"] for k in keys]
        ax.plot(xs, ys, "s-", color=colors[n_phases],
                label=f"{n_phases} distinct phases")
    for pt in real_operating_points():
        if pt["mantel_r"] is None:
            continue
        ok = pt["n_cycles"] == pt["expected"]
        ax.plot([pt["mantel_r"]], [1.0 if ok else 0.0], "k*", markersize=13,
                zorder=6)
    ax.plot([], [], "k*", markersize=11, label="measured clouds")
    ax.set_xlabel("circular Mantel correlation $r$  (phase is known here)")
    ax.set_ylabel("share of simulations counted as one cycle")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_title("B. a strong Mantel $r$ is necessary but not sufficient",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)

    fig.suptitle("The envelope of the cycle count: a phase-rich design is not "
                 "enough, a phase-rich *transcriptome* is needed", fontsize=11)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fig_cycle_counting_power.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="benchmark")
    ap.add_argument("--figdir", default="docs")
    ap.add_argument("--n-null", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 92)
    print("OmicsTopology - when does counting cycles work?")
    print("=" * 92)
    print(f"p={P_TOTAL} genes, {NOISE_SD=}, n_null={args.n_null}, "
          f"{args.n_seeds} seeds per cell\n")
    print(f"{'phases':>7}{'n_signal':>10}{'n':>5}{'mantel_r':>10}"
          f"{'gap':>9}{'counts one':>12}")
    sweep_out = sweep(args.n_null, args.n_seeds, args.seed)
    for key, rec in sorted(sweep_out.items()):
        print(f"{rec['n_phases']:>7}{rec['n_signal']:>10}{rec['n_samples']:>5}"
              f"{rec['mantel_r_median']:>+10.3f}"
              f"{(rec['gap_median'] if rec['gap_median'] is not None else float('nan')):>9.2f}"
              f"{rec['fraction_counting_one']:>12.2f}")

    payload = {
        "config": {"p_total": P_TOTAL, "noise_sd": NOISE_SD,
                   "n_null": args.n_null, "n_seeds": args.n_seeds,
                   "signal_grid": list(N_SIGNAL_GRID),
                   "phase_grid": list(PHASE_GRID)},
        "sweep": sweep_out,
        "measured_clouds": real_operating_points(),
    }
    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "cycle_counting_power.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nwrote {json_path}")
    if not args.no_figures:
        fig_path = make_figure(sweep_out, args.figdir)
        print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
