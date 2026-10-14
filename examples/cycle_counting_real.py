"""Block C, on measured data: what does ``beta_1`` actually count?

``examples/topology_without_labels.py`` answered the modelling question with
designs built to order: a single time axis gives one loop however many
oscillators ride on it, and only a genuinely two-dimensional design gives two.
The obvious objection is that the same person chose the design and wrote the
claim.  This script asks the same question of five **public, measured**
datasets whose designs were fixed by other people years ago.

The designs
----------------
===============================  ============  ==========================================
key                            accession     what varies
===============================  ============  ==========================================
``psme_annual``                GSE93067      5 Douglas-fir trees x 16 dates over a year
``psme_diurnal``               GSE93066      6 trees x 6 times of day x 2 days
``human_forced_desync``        GSE48113      22 people x 7 circadian phases x 2 sleep
``psme_annual_x_diurnal``      both          the two Douglas-fir series pooled
``yeast_cell_cycle``           GSE8799       2 genotypes x 2 elutriations x 30 times
``estrous_circadian_grid``     PRJNA1348461  48 females x 12 clock times x 4 oestrous
===============================  ============  ==========================================

What each one is for
--------------------
**One cyclic parameter on a different timescale.**  ``psme_annual`` returns to
the same tree every ~3 weeks for a year.  If the count is reading a cycle and
not a trajectory it must say *one* for a one-year period, and *zero* for any
sub-year window, because a partial year is an open arc into and out of
dormancy.  Sweeping the window length turns that into a curve rather than an
anecdote.

**Two axis labels, one cycle.**  ``psme_diurnal`` has a sample sheet with two
factors -- hour of day and day of the experiment -- and 24 apparent grid
cells, but day 2 repeats day 1's times of day.  ``human_forced_desync`` is the
same trap in a different costume: 7 circadian phases x 2 sleep phases, where
only the circadian axis is resolved as a cycle.  A count of *factors* gives 2;
a count of *cycles* gives 1.

**Two cyclic parameters.**  ``psme_annual_x_diurnal`` pools the yearlong and
the 48-h series.  Both come from one study, one protocol and one de-novo
transcript reference, and between them they carry two independently sampled
cycles -- a circannual one and a daily one.  This case has to come out as
**two**.

**A cycle that is not a clock.**  Every other design here turns on a *time*
axis -- a year, a day, a sleep schedule -- so a critic can say the method is
reading the calendar.  ``yeast_cell_cycle`` removes that escape: the
coordinate is the phase of the cell cycle, a cell-intrinsic programme that
nothing entrains, and the sample sheet records *three* axes -- genotype (wild
type vs a clb1-6 cyclin mutant), replicate elutriation, and time since release
-- of which exactly one is cyclic.  A criterion that counted axes would answer
three; the count must answer one.

The replicate problem, and what is done about it
------------------------------------------------
Real designs have biological replicates -- five trees, twenty-two people -- and
a naive cloud of every sample contains one loop *per replicate* whenever the
replicates sit far enough apart.  Two things are measured here.  The primary
cloud subtracts each replicate's own gene means, which is the standard way to
remove a subject effect and leaves the within-replicate time course intact.  As
a check, the same cells are also collapsed to one centroid per (replicate,
phase) point.  Both are reported; neither is assumed.

Everything goes through exactly the pipeline of block C: standardise,
correlation distance, H1 bar lengths, compare against a covariance-matched
Gaussian null, count the bars that outrun it.

What came out -- and it is not the 6/6 the simulation gave
---------------------------------------------------------
Two of the five designs are counted correctly, and both are the ones whose
cycle is carried by essentially the whole selected gene set: the yearlong
series (``gap`` ~ 4.9, count 1) and the yeast cell cycle (``gap`` ~ 3.6,
count 1 -- the one design whose cycle is not a clock).  The other three return
0.  Reporting only that number would be useless, so two diagnostics are
measured alongside it:

* the **dominant-bar gap**, ``bar[0] / bar[1]``, which the counting criterion
  silently requires to be far above 1, and which is 1.06-1.36 for every design
  the count misses; and
* the **circular Mantel correlation** against the design's own axis, which
  shows the cycles are real -- +0.70 for the diurnal cloud, the same value the
  yearlong series has -- so the failure belongs to the statistic, not to the
  data.

``examples/cycle_counting_power.py`` then sweeps the one variable that
explains both: how many of the 3000 selected genes actually carry the ring.
The count succeeds only when essentially all of them do.  That is the honest
answer to the label-free claim on measured data: the cycle is countable when
the transcriptome is dominated by it, and not otherwise.

Usage::

    python examples/prepare_cycle_designs.py --outdir data/cycle_designs
    python examples/cycle_counting_real.py --root data/cycle_designs
    python examples/cycle_counting_power.py

Outputs ``benchmark/cycle_counting_real.json`` and
``docs/fig_cycle_counting_real.png``.
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

from prepare_cycle_designs import DESIGNS, POOLED_KEY, load  # noqa: E402
from multi_dataset_benchmark import mantel  # noqa: E402
from topology_without_labels import count_cycles  # noqa: E402

from omictopo.geometry import pairwise_distances, standardize  # noqa: E402


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def centre_by_group(X: np.ndarray, groups: list) -> np.ndarray:
    """Subtract each replicate's own mean gene vector.

    This is the ordinary way to remove a subject/tree effect.  It matters for
    the cycle count because without it every well-separated replicate
    contributes its own loop: five trees = five rings, and ``beta_1`` would
    report the number of trees, not the number of cycles.
    """
    out = np.array(X, dtype=float, copy=True)
    idx: dict = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    for rows in idx.values():
        out[rows] -= X[rows].mean(axis=0, keepdims=True)
    return out


def collapse(X: np.ndarray, keys: list) -> tuple[np.ndarray, list]:
    """Average rows of ``X`` that share a key -- one centroid per design cell."""
    idx: dict = {}
    for k in keys:
        idx.setdefault(k, len(idx))
    acc = np.zeros((len(idx), X.shape[1]))
    cnt = np.zeros(len(idx))
    for row, k in zip(X, keys):
        j = idx[k]
        acc[j] += row
        cnt[j] += 1
    return acc / cnt[:, None], sorted(idx, key=lambda k: idx[k])


def circular_distance(a: np.ndarray, period: float = 2.0 * np.pi) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    d = np.abs(a[:, None] - a[None, :])
    return np.minimum(d, period - d)


def score(X: np.ndarray, n_null: int, seed: int, phase=None) -> dict:
    """Count the cycles, and record the two diagnostics that explain the count.

    ``gap`` is the ratio of the longest H1 bar to the next one.  A genuine
    cycle produces a *dominant* bar, so the criterion silently needs ``gap``
    to be far above one; when the bar spectrum decays smoothly there is no
    cycle to count however many samples were collected.

    ``mantel_r`` is the circular Mantel correlation against the design's own
    axis, and ``mantel_p`` its permutation p-value.  Both use the label, so
    they are **not** part of the label-free claim -- they are quoted here only
    to show that the cycles the count misses are really present in the data
    (a phase-based method can see them; the count cannot), and to keep the
    significance claim reproducible from the committed record.
    """
    rec = count_cycles(X, n_null=n_null, seed=seed)
    rec["n_cycles"] = int(rec.pop("n_bars_above_null_q95"))
    bars = rec["top_bars_rel"]
    rec["gap"] = (round(float(bars[0] / bars[1]), 4)
                  if len(bars) > 1 and bars[1] > 1e-9 else None)
    if phase is not None:
        Z = standardize(X)
        D = pairwise_distances(Z, metric="correlation")
        r, p = mantel(D, circular_distance(np.asarray(phase, dtype=float)),
                      n_perm=2000, seed=seed)
        rec["mantel_r"] = round(float(r), 4)
        rec["mantel_p"] = round(float(p), 6)
    return rec


def add(results: dict, label: str, X: np.ndarray, expected: int, n_null: int,
        seed: int, note: str = "", groups: list | None = None,
        phase=None, primary: bool = False) -> None:
    """Score one cloud, and (if replicates are given) the replicate-centred one.

    ``primary`` marks the one cloud per design that the headline count refers
    to.  The other rows are controls -- a single day, a single genotype, the
    centroids -- and answer different questions, so they are reported but not
    folded into the headline.
    """
    if groups is not None:
        raw = score(X, n_null, seed)
        X = centre_by_group(X, groups)
    else:
        raw = None
    rec = score(X, n_null, seed, phase=phase)
    rec.update(label=label, n_cycles_expected=int(expected),
               correct=bool(rec["n_cycles"] == int(expected)), note=note,
               centred=groups is not None, primary=bool(primary))
    if raw is not None:
        rec["n_cycles_raw"] = raw["n_cycles"]
        rec["top_bars_rel_raw"] = raw["top_bars_rel"]
        rec["null_q95_raw"] = raw["null_q95"]
    results[label] = rec
    flag = "OK " if rec["correct"] else "!! "
    extra = "" if raw is None else f" (raw: {raw['n_cycles']})"
    gap = rec["gap"]
    mant = rec.get("mantel_r")
    print(f"  {flag}{label:30} n={rec['n_samples']:4d} "
          f"bar={rec['top_bars_rel'][0]:.4f} null_q95={rec['null_q95']:.4f} "
          f"gap={gap if gap is not None else float('nan'):5.2f} "
          f"r={mant if mant is not None else float('nan'):+.3f} "
          f"-> {rec['n_cycles']}{extra} (expected {expected})")


# --------------------------------------------------------------------------
# the real designs
# --------------------------------------------------------------------------
def run_annual(root: str, results: dict, sweeps: dict, n_null: int, seed: int):
    X, coords, _, info = load("psme_annual", root)
    print(f"psme_annual  {info['accession']}  {info['title']}")
    groups = [c["group"] for c in coords]
    days = np.array([c["axis2_value"] for c in coords], dtype=float)
    phase = np.array([c["axis1_value"] for c in coords], dtype=float)

    add(results, "annual_all_samples", X, 1, n_null, seed,
        "66 samples, 5 trees x 16 dates", groups=groups, phase=phase,
        primary=True)
    Xc, _ = collapse(X, [(g, d) for g, d in zip(groups, days)])
    add(results, "annual_date_centroids", Xc, 1, n_null, seed,
        "one centroid per (tree, date)")

    uniq = sorted(set(days.tolist()))
    span = uniq[-1] - uniq[0]
    sweep = []
    for k in range(3, len(uniq) + 1):
        chosen = set(uniq[:k])
        keep = [i for i, d in enumerate(days) if d in chosen]
        rec = score(centre_by_group(X[keep], [groups[i] for i in keep]), n_null, seed)
        sweep.append({
            "k_dates": k,
            "covered_fraction": round(float((uniq[k - 1] - uniq[0]) / span), 4),
            "gap_fraction": round(float((span - (uniq[k - 1] - uniq[0])) / span), 4),
            "n_samples": int(rec["n_samples"]),
            "n_cycles": rec["n_cycles"],
            "top_bar_rel": rec["top_bars_rel"][0],
            "null_q95": rec["null_q95"],
        })
    sweeps["annual"] = sweep
    print(f"  sweep (k dates -> cycles): {[(s['k_dates'], s['n_cycles']) for s in sweep]}")


def run_diurnal(root: str, results: dict, sweeps: dict, n_null: int, seed: int):
    X, coords, _, info = load("psme_diurnal", root)
    print(f"psme_diurnal {info['accession']}  {info['title']}")
    groups = [c["group"] for c in coords]
    hours = np.array([c["axis2_value"] for c in coords], dtype=float)
    phase = np.array([c["axis1_value"] for c in coords], dtype=float)

    add(results, "diurnal_all_samples", X, 1, n_null, seed,
        "72 samples, 6 trees x 12 time points over 48 h", groups=groups,
        phase=phase, primary=True)
    for d in (0.0, 1.0):
        keep = [i for i, h in enumerate(hours) if h // 24 == d]
        add(results, f"diurnal_experimental_day{int(d) + 1}", X[keep], 1, n_null, seed,
            "one full 24-h cycle from a single day -- the two-axis trap",
            groups=[groups[i] for i in keep],
            phase=[phase[i] for i in keep])
    Xc, _ = collapse(X, [(g, h) for g, h in zip(groups, hours)])
    add(results, "diurnal_time_centroids", Xc, 1, n_null, seed,
        "one centroid per (tree, time point)")

    uniq = sorted(set(hours.tolist()))
    sweep = []
    for k in range(2, len(uniq) + 1):
        chosen = set(uniq[:k])
        keep = [i for i, h in enumerate(hours) if h in chosen]
        rec = score(centre_by_group(X[keep], [groups[i] for i in keep]), n_null, seed)
        sweep.append({
            "k_timepoints": k,
            "window_hours": float(uniq[k - 1] - uniq[0]),
            "n_samples": int(rec["n_samples"]),
            "n_cycles": rec["n_cycles"],
            "top_bar_rel": rec["top_bars_rel"][0],
            "null_q95": rec["null_q95"],
        })
    sweeps["diurnal"] = sweep
    print(f"  sweep (window h -> cycles): "
          f"{[(s['window_hours'], s['n_cycles']) for s in sweep]}")


def run_forced_desync(root: str, results: dict, n_null: int, seed: int):
    X, coords, _, info = load("human_forced_desync", root)
    print(f"human_forced_desync {info['accession']}  {info['title']}")
    groups = [c["group"] for c in coords]
    phase = np.array([c["axis1_value"] for c in coords], dtype=float)
    add(results, "fd_all_samples", X, 1, n_null, seed,
        "287 samples, 22 people x 7 circadian phases x 2 sleep phases",
        groups=groups, phase=phase, primary=True)
    for lab, val in (("in", 0.0), ("out", 1.0)):
        keep = [i for i, c in enumerate(coords) if c["axis2_value"] == val]
        add(results, f"fd_sleep_{lab}", X[keep], 1, n_null, seed,
            f"one full circadian cycle, sleep {lab} of phase",
            groups=[groups[i] for i in keep],
            phase=[phase[i] for i in keep])
    Xc, _ = collapse(X, [(c["axis1_value"], c["axis2_value"]) for c in coords])
    add(results, "fd_cell_centroids", Xc, 1, n_null, seed,
        "one centroid per (circadian phase, sleep phase) cell")


def run_pooled(root: str, results: dict, n_null: int, seed: int):
    X, coords, _, info = load(POOLED_KEY, root)
    print(f"psme_annual_x_diurnal  {info['accession']}  {info['title']}")
    groups = [c["group"] for c in coords]
    add(results, "pooled_all_samples", X, 2, n_null, seed,
        "138 samples: 66 from the yearlong series + 72 from the 48-h series",
        groups=groups, primary=True)
    keys = [(c["extra"], round(c["axis1_value"], 6), round(c["axis2_value"], 6))
            for c in coords]
    Xc, _ = collapse(X, keys)
    add(results, "pooled_design_cells", Xc, 2, n_null, seed,
        "16 annual cells + 6 daily cells: the two sampled cycles")


def run_yeast_cell_cycle(root: str, results: dict, sweeps: dict, n_null: int,
                         seed: int):
    """The one design whose cycle is not a clock.

    GSE8799 records three axes -- genotype (wild type vs a clb1-6 cyclin
    mutant), replicate elutriation, and time since release -- and only the last
    carries a cycle.  The design therefore asks the counting criterion the
    question it claims to answer: *how many independent cyclic parameters are
    here?*  The answer is one, and a criterion that counted axes would say
    three.
    """
    X, coords, _, info = load("yeast_cell_cycle", root)
    print(f"yeast_cell_cycle {info['accession']}  {info['title']}")
    groups = [c["group"] for c in coords]
    phase = np.array([c["axis1_value"] for c in coords], dtype=float)
    geno = [c["extra"].rsplit("@", 1)[0] for c in coords]
    mins = np.array([float(c["extra"].rsplit("@", 1)[1][:-3]) for c in coords])
    period_min = float(info["period_hours"]) * 60.0

    add(results, "cellcycle_two_genotypes", X, 1, n_null, seed,
        "60 samples: 2 genotypes x 2 elutriations x 30 time points on an 8-min "
        "grid -- three recorded axes, one cyclic parameter",
        groups=groups, phase=phase, primary=True)
    for key, g in (("wildtype", "WildType"), ("cyclin_mutant", "CyclinMutant")):
        keep = [i for i, name in enumerate(geno) if name == g]
        add(results, f"cellcycle_{key}", X[keep], 1, n_null, seed,
            f"one genotype alone ({g}), 30 samples",
            groups=[groups[i] for i in keep], phase=[phase[i] for i in keep])

    uniq = sorted(set(mins.tolist()))
    sweep = []
    for k in range(3, len(uniq) + 1):
        chosen = set(uniq[:k])
        keep = [i for i, m in enumerate(mins) if m in chosen]
        sub = [groups[i] for i in keep]
        sizes: dict = {}
        for g in sub:
            sizes[g] = sizes.get(g, 0) + 1
        if min(sizes.values()) < 2:
            # the two elutriations interleave on an 8-min grid, so a short
            # window leaves one replicate with a single sample; centring it
            # gives the zero vector, which has no correlation distance and hence
            # no H1 bars.  Such a window cannot be scored, so it is skipped.
            continue
        rec = score(centre_by_group(X[keep], sub), n_null, seed)
        bars = rec["top_bars_rel"]
        sweep.append({
            "k_timepoints": k,
            "minutes": float(uniq[k - 1] - uniq[0]),
            "covered_periods": round(float((uniq[k - 1] - uniq[0]) / period_min), 4),
            "n_samples": int(rec["n_samples"]),
            "n_cycles": rec["n_cycles"],
            "top_bar_rel": bars[0] if bars else None,
            "null_q95": rec["null_q95"],
        })
    sweeps["cellcycle"] = sweep
    print("  sweep (periods covered -> cycles): "
          f"{[(s['covered_periods'], s['n_cycles']) for s in sweep]}")

    # What each recorded axis actually carries.  Measured on the *uncentred*
    # cloud, so the genotype shift is still in the distances: genotype is a
    # strong *shift* (it takes two values, so it can carry no loop at all), and
    # the replicate is an offset.  Removing those two non-cyclic axis means is
    # what leaves the one countable cycle -- which is why the count is one and
    # not three, and why the same cloud without centring returns zero.
    geno_num = np.array([0.0 if g == "WildType" else 1.0 for g in geno])
    gdist = np.abs(np.subtract.outer(geno_num, geno_num))
    raw_d = pairwise_distances(standardize(X), metric="correlation")
    r_g, p_g = mantel(raw_d, gdist, n_perm=2000, seed=seed)
    r_p, p_p = mantel(raw_d, circular_distance(phase), n_perm=2000, seed=seed)
    checks = {
        "note": ("Mantel correlations on the uncentred cloud, where the "
                 "genotype shift is still present in the distances"),
        "genotype": {"r": round(float(r_g), 4), "p": round(float(p_g), 6)},
        "phase_uncentred": {"r": round(float(r_p), 4),
                            "p": round(float(p_p), 6)},
        "count_uncentred": int(results["cellcycle_two_genotypes"]["n_cycles_raw"]),
        "count_replicate_centred": int(results["cellcycle_two_genotypes"]["n_cycles"]),
    }
    print(f"  axis checks (uncentred): genotype r={r_g:+.3f}, phase "
          f"r={r_p:+.3f}; count {checks['count_uncentred']} -> "
          f"{checks['count_replicate_centred']} after removing axis means")
    return checks


def run_estrous_grid(root: str, results: dict, n_null: int, seed: int):
    """The first measured design with two recorded **cyclic** axes.

    PRJNA1348461 (Smith et al., bioRxiv 2026) is mouse liver from 48 females,
    one sample every two hours for 96 h, each animal staged into one of the four
    oestrous phases.  That is 12 clock times x 4 phases with every cell
    occupied: a grid, and the first measured design whose sample sheet records
    two axes that are *both* cyclic (a ~24 h clock and a ~96 h oestrous cycle),
    as opposed to the two-labels-one-cycle traps elsewhere in this script.

    The count is **one**, not two, and the two diagnostics say which axis
    survives.  Split at the phase boundaries, every 24 h block resolves the
    clock on its own (circular Mantel r = +0.52..+0.73, p <= 0.0005); the
    oestrous axis leaves no independent loop (r = +0.006, p = 0.43).  The
    calibration in ``examples/estrous_grid_cycles.py`` shows what that means:
    a clock with a per-phase *phase shift* counts exactly like an unshifted one,
    so a second phase of the same loop is not a second loop.
    """
    X, coords, _, info = load("estrous_circadian_grid", root)
    print(f"estrous_circadian_grid {info['accession']}  {info['title']}")
    clock = np.array([c["axis1_value"] for c in coords], dtype=float)
    groups = [c["group"] for c in coords]

    add(results, "estrous_grid_clock_axis", X, 1, n_null, seed,
        "48 samples, 12 clock times x 4 oestrous phases; the ~24 h axis is the "
        "one countable cycle",
        phase=clock, primary=True)

    # Each 24 h stretch on its own -- the clock must resolve inside every phase
    # if it is real, whatever the second axis is doing.
    block_scores = {}
    for ph in sorted(set(groups)):
        keep = [i for i, g in enumerate(groups) if g == ph]
        sub = X[keep]
        rec = score(sub, n_null, seed, phase=clock[keep])
        block_scores[ph] = rec
        add(results, f"estrous_grid_{ph}", sub, 1, n_null, seed,
            f"one oestrous phase alone ({ph}), 12 samples across the clock",
            phase=clock[keep])

    r_clock = results["estrous_grid_clock_axis"].get("mantel_r")
    print(f"  clock axis Mantel r={r_clock:+.3f}; per-phase blocks "
          f"{[(p, round(s['mantel_r'], 3)) for p, s in block_scores.items()]}")
    return {"per_phase_blocks": {p: {
        "n_samples": s["n_samples"], "n_cycles": s["n_cycles"],
        "mantel_r": s.get("mantel_r"), "mantel_p": s.get("mantel_p"),
    } for p, s in block_scores.items()}}


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def make_figure(results: dict, sweeps: dict, outdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))

    ax = axes[0]
    a = sweeps["annual"]
    ax.plot([s["covered_fraction"] for s in a], [s["n_cycles"] for s in a],
            "o-", color="#1f5fa9", label="circannual axis (GSE93067)")
    d = sweeps["diurnal"]
    ax.plot([s["window_hours"] / 24.0 for s in d], [s["n_cycles"] for s in d],
            "s-", color="#c0504d", label="daily axis (GSE93066)")
    c = sweeps["cellcycle"]
    ax.plot([s["covered_periods"] for s in c], [s["n_cycles"] for s in c],
            "^-", color="#2e7d32", label="cell cycle, no clock (GSE8799)")
    ax.axvline(1.0, color="#888888", ls=":", lw=1.0)
    ax.axvline(2.0, color="#cccccc", ls=":", lw=1.0)
    ax.text(1.015, 1.62, "window closes\nthe cycle", color="#666666", fontsize=7)
    ax.set_xlabel("sampling window (fraction of one period)")
    ax.set_ylabel(r"$\beta_1$ above the matched null")
    ax.set_yticks([0, 1, 2])
    ax.set_ylim(-0.25, 2.35)
    ax.set_title("A. a partial period is an arc, not a loop", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels = ["annual_all_samples", "diurnal_all_samples", "fd_all_samples",
              "pooled_all_samples", "cellcycle_two_genotypes"]
    pretty = ["annual\nyearlong\n1 cycle", "diurnal\n2 axes\n1 cycle",
              "forced desync\n2 axes\n1 cycle", "annual x daily\n2 cycles",
              "yeast cell cycle\n3 axes\n1 cycle"]
    got = [results[k]["n_cycles"] for k in labels]
    exp = [results[k]["n_cycles_expected"] for k in labels]
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.18, exp, width=0.34, color="#d8d8d8", edgecolor="#8a8a8a",
           label="design predicts")
    ax.bar(xpos + 0.18, got, width=0.34, color="#1f5fa9",
           label=r"$\beta_1$ counts")
    for i, (e, g) in enumerate(zip(exp, got)):
        ax.text(i - 0.18, e + 0.06, str(e), ha="center", fontsize=8)
        ax.text(i + 0.18, g + 0.06, str(g), ha="center", fontsize=8,
                fontweight="bold" if e == g else "normal")
    ax.set_xticks(xpos)
    ax.set_xticklabels(pretty, fontsize=8)
    ax.set_ylabel("independent cycles")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(0, 3.1)
    ax.set_title("B. axis labels vs cyclic parameters", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    # --- C: the diagnostic that explains panel B -------------------------
    ax = axes[2]
    gap_labels = ["annual_all_samples", "diurnal_all_samples", "fd_all_samples",
                  "pooled_all_samples", "cellcycle_two_genotypes"]
    short = ["annual", "diurnal", "forced\ndesync", "annual\nx daily",
             "yeast cell\ncycle"]
    gaps = [results[k].get("gap") or 0.0 for k in gap_labels]
    rs = [results[k].get("mantel_r") for k in gap_labels]
    correct = [results[k]["correct"] for k in gap_labels]
    bars = ax.bar(np.arange(len(gap_labels)), gaps,
                  color=["#2e7d32" if c else "#b03030" for c in correct],
                  width=0.6, zorder=3)
    ax.axhline(1.0, color="#888888", ls=":", lw=1.0)
    ax.text(len(gap_labels) - 0.55, 1.05, "no dominant\nbar", color="#666666",
            fontsize=7.5, ha="right", va="bottom")
    for bar, r in zip(bars, rs):
        if r is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2.4,
                    f"r={r:+.2f}", ha="center", fontsize=8, color="white",
                    fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 14.0)
    ax.set_xticks(np.arange(len(gap_labels)))
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(r"dominant-bar gap  $bar_1/bar_2$")
    ax.set_title("C. only two designs have a\ndominant bar  "
                 "(green = counted right)", fontsize=10)
    ax.grid(alpha=0.25, axis="y", which="both")

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fig_cycle_counting_real.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/cycle_designs")
    ap.add_argument("--outdir", default="benchmark")
    ap.add_argument("--figdir", default="docs")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    results: dict = {}
    sweeps: dict = {}
    run_annual(args.root, results, sweeps, args.n_null, args.seed)
    run_diurnal(args.root, results, sweeps, args.n_null, args.seed)
    run_forced_desync(args.root, results, args.n_null, args.seed)
    run_pooled(args.root, results, args.n_null, args.seed)
    axis_checks = run_yeast_cell_cycle(args.root, results, sweeps, args.n_null,
                                       args.seed)
    grid_blocks = run_estrous_grid(args.root, results, args.n_null, args.seed)
    n_ok = sum(1 for r in results.values() if r["correct"])
    prim = [r for r in results.values() if r.get("primary")]
    n_prim_ok = sum(1 for r in prim if r["correct"])
    print(f"\n{n_prim_ok}/{len(prim)} designs are counted right "
          f"({n_ok}/{len(results)} clouds overall)")
    print("  diagnostic: the yearlong series reaches a dominant-bar gap "
          f"{results['annual_all_samples']['gap']:.2f}, the yeast cell cycle "
          f"{results['cellcycle_two_genotypes']['gap']:.2f}; the designs the "
          "count misses sit near gap 1.1-1.4, i.e. their H1 spectrum has no "
          "dominant bar to count (see benchmark/cycle_counting_power.json).")

    os.makedirs(args.outdir, exist_ok=True)
    payload = {
        "n_null": args.n_null,
        "seed": args.seed,
        "designs": {k: dict(v) for k, v in DESIGNS.items()},
        "results": results,
        "window_sweeps": sweeps,
        "cellcycle_axis_checks": axis_checks,
        "estrous_grid_blocks": grid_blocks,
        "n_correct": n_ok,
        "n_cases": len(results),
        "n_primary_correct": n_prim_ok,
        "n_primary": len(prim),
        "finding": {
            "headline": (
                "On measured data the counting criterion returns the design's "
                "own number for 2 of 5 designs: the yearlong series and the "
                "yeast cell cycle. The cell cycle is the one design whose "
                "cyclic parameter is not a clock -- every other design here "
                "turns on a time axis -- and it is the design that separates "
                "counting axes from counting cycles: three recorded axes, one "
                "cycle, and the count is one."),
            "what_works": (
                "psme_annual: 5 Douglas-fir trees x 16 dates over a full year. "
                "Replicate-centred, the H1 spectrum has a dominant bar "
                "(gap ~4.9) and the count is 1, as the design predicts. The "
                "window sweep is the same statement from the other side: any "
                "sub-year window is an arc and counts 0. cellcycle_two_"
                "genotypes (GSE8799): 2 genotypes x 2 elutriations x 30 time "
                "points at 8-min resolution, i.e. *three* recorded design axes "
                "(genotype, replicate, cell-cycle time) and *one* cyclic "
                "parameter. Replicate-centred the dominant-bar gap is 3.6 and "
                "the count is 1. The two other recorded axes carry no cycle: "
                "genotype is a strong two-valued shift (Mantel r = +0.46, "
                "p <= 0.0005 on the uncentred cloud) and the elutriation is an "
                "offset, so the count is one and not three -- and without "
                "removing those two axis means the same cloud returns zero. The "
                "phase alignment is significant but modest: circular Mantel "
                "r = +0.16, p <= 0.0005, at the 110-130 min period the "
                "periodogram gives."),
            "what_fails_and_why": (
                "diurnal (GSE93066), forced desync (GSE48113) and the pooled "
                "annual x diurnal design all have a smoothly decaying H1 "
                "spectrum (gap 1.06-1.36), so there is no dominant bar and the "
                "count is 0. The cycles are nevertheless present: the circular "
                "Mantel correlation against the design's own axis is +0.70 for "
                "the diurnal cloud, i.e. the same strength as the yearlong "
                "series that *is* counted. The limitation is the power of the "
                "dominant-bar test, not the biology."),
            "why_the_single_genotypes_behave_differently": (
                "Split by genotype, the two 30-sample clouds do not behave "
                "alike. The wild type alone has no dominant bar (gap 1.35) and "
                "counts 0. The cyclin mutant alone has a dominant bar (gap "
                "5.15) and counts 1 - but its circular Mantel correlation "
                "against the cell-cycle phase is only +0.06, so the bar that "
                "dominates is not the one the design names, and the decision "
                "rule would call that cloud 'inconsistent: check the design "
                "before reporting'. The pooled 60-sample cloud is the one whose "
                "dominant bar does track the phase (r = +0.16, p = 0.0005), "
                "which is why the headline count is quoted on it. A large gap, "
                "in other words, is not by itself evidence that the integer is "
                "the design's - the same caveat the identifiability rule "
                "exists to enforce."),
            "why_the_forced_desync_cloud_cannot_test_the_claim": (
                "human_forced_desync has circular Mantel r = +0.045 against the "
                "circadian axis: the circadian programme is not in the 3000 "
                "highest-variance genes of whole blood at all, so no method "
                "restricted to those genes could count that cycle."),
            "envelope": (
                "examples/cycle_counting_power.py sweeps how many of the 3000 "
                "selected genes carry the ring. Across the time-axis designs "
                "the count only succeeds when essentially all of them do, and "
                "at the measured diurnal r ~ 0.70 the simulated gap is 1.2-1.4 "
                "- the failing clouds sit exactly on that curve. This is a real "
                "limitation of the criterion and it is measured, not assumed."),
            "where_the_envelope_does_not_explain_everything": (
                "The power sweep explains the diurnal and forced-desync "
                "failures but not the yeast success: there the circular Mantel "
                "r is only +0.16, far below the +0.70 the failing diurnal cloud "
                "has, and yet the H1 spectrum still shows a dominant bar. A "
                "large gap and a large r are therefore two different kinds of "
                "evidence - the ring is present geometrically even where the "
                "design axis explains little of the distance variance - and it "
                "is the first kind that the count needs. The same asymmetry is "
                "what the identifiability rule in examples/cycle_"
                "identifiability.py is built to expose."),
        },
    }
    payload["designs"][POOLED_KEY] = {
        "accession": "GSE93067+GSE93066",
        "organism": "Pseudotsuga menziesii",
        "title": "Douglas-fir: yearlong series + 48-h diurnal series, pooled",
        "n_cyclic_parameters": 2,
    }
    json_path = os.path.join(args.outdir, "cycle_counting_real.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    fig_path = make_figure(results, sweeps, args.figdir)
    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
