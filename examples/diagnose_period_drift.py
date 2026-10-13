"""Diagnosis: cross-period drift explains why a 2-cycle record can look linear.

When the benchmark ran on GSE137732 (a 45 h record covering nearly two
circadian cycles, sampled once per time point), both light conditions were
called *line-like* — the only circadian failures in the whole evaluation.
This script asks whether the failure is a property of the method or of the
state space itself, by splitting the record into single-cycle windows and
re-running the same pipeline on each window.

Why this matters: a clock that keeps time for 45 h does **not** guarantee that
the expression state space is a single closed loop.  If the transcriptome is
also drifting — development, ageing, Run-to-run effects — then the second
cycle does not close onto the first, and the 45 h cloud is a **helix**, not a
loop.  A helix has no closed loop at the scale of the whole record, and the
circular Mantel test with ``P = 24`` correctly refuses it, because the phase
of cycle 2 is displaced relative to cycle 1.  Splitting into single-cycle
windows removes the drift between cycles while keeping the loop within one.

This is a post-hoc diagnosis, not part of the pre-registered benchmark; it
was written *after* the benchmark flagged the two failures.

Run::

    python examples/diagnose_period_drift.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import SPECS, load_prepared  # noqa: E402
from multi_dataset_benchmark import h1_summary, mantel_block  # noqa: E402
from omictopo.geometry import pairwise_distances, standardize  # noqa: E402


def analyse_window(X, times, period, n_perm=1500, seed=0) -> dict:
    Z = standardize(X)
    out = {"n": int(X.shape[0]),
           "h1_max_rel": h1_summary(pairwise_distances(Z, metric="correlation"))["h1_max_rel"]}
    if times.size >= 4 and period:
        mb = mantel_block(X, times, period, n_perm=n_perm, seed=seed)
        out.update({"r_linear": mb["r_linear"],
                    "r_circular": mb["r_circular"],
                    "best_period": mb["best_period"],
                    "best_r_circular": mb["best_r_circular"],
                    "circular_advantage": mb["circular_advantage"]})
    return out


def cycle_windows(times: np.ndarray, period: float) -> list:
    """Consecutive half-open windows ``[k*P, (k+1)*P)`` that contain >= 6 samples."""
    windows = []
    lo = float(times.min())
    hi = float(times.max())
    k = 0
    while lo + k * period <= hi:
        mask = (times >= lo + k * period) & (times < lo + (k + 1) * period)
        if int(mask.sum()) >= 6:
            windows.append((k, mask))
        k += 1
    return windows


def diagnose(key: str, data_root: str, period: float | None,
             n_perm: int = 1500) -> dict:
    X, conditions, times, _, _ = load_prepared(os.path.join(data_root, key))
    out = {"key": key, "period_hours": period, "whole_record": {}, "cycles": []}
    for cond in sorted(set(conditions.tolist())):
        m = conditions == cond
        whole = analyse_window(X[m], times[m], period)
        out["whole_record"][cond] = whole

        if not period:
            continue
        for k, mask in cycle_windows(times[m], period):
            block = analyse_window(X[m][mask], times[m][mask], period)
            block["cycle"] = k
            block["t_from"] = float(times[m][mask].min())
            block["t_to"] = float(times[m][mask].max())
            out["cycles"].append({"condition": cond, **block})
    return out


def main(data_root: str = "data/datasets", outdir: str = "data/benchmark",
         n_perm: int = 1500) -> int:
    keys = [k for k, s in SPECS.items() if s.period_hours]
    report = {}
    for key in keys:
        print(f"\n### {key}  (period = {SPECS[key].period_hours} h)")
        rep = diagnose(key, data_root, SPECS[key].period_hours, n_perm=n_perm)
        report[key] = rep
        for cond, whole in rep["whole_record"].items():
            adv = whole.get("circular_advantage")
            adv_txt = f"circular_advantage={adv:+.3f}" if adv is not None else "-"
            print(f"  whole record  {cond:<20} n={whole['n']:<3} "
                  f"H1max_rel={whole['h1_max_rel']:<8} {adv_txt}")
        for cyc in rep["cycles"]:
            print(f"  cycle {cyc['cycle']}        {cyc['condition']:<20} "
                  f"n={cyc['n']:<3} [{cyc['t_from']:.0f}-{cyc['t_to']:.0f} h]  "
                  f"r_lin={cyc['r_linear']:+.3f}  r_circ@{cyc['best_period']:.0f}h="
                  f"{cyc['best_r_circular']:+.3f}  advantage={cyc['circular_advantage']:+.3f}")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "diagnosis_period_drift.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n[diagnosis] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
