"""Statistical audit of the benchmark: is the headline claim actually supported?

A pre-registered benchmark that "matches expectation in 8 of 10 clouds" still
has to survive the questions a referee will ask.  This script answers four of
them directly from the committed results in ``benchmark/results.json``:

1. **Ablation.**  How much of the accuracy comes from the *topological*
   statistic (H1 against a covariance-matched null) and how much from the
   classical circular-vs-linear Mantel test?  If the Mantel line alone scores
   the same, persistent homology has not demonstrated value on this benchmark.
2. **Multiplicity.**  Ten clouds, two tests each.  Benjamini-Hochberg FDR at
   0.05 across the family: how many H1 p-values survive?  A raw p = 0.02 in a
   family of ten needs comparison against 0.005, not 0.05.
3. **Resolution.**  The headline is a success count on 10 units.  Exact
   binomial against a 50 % baseline, plus a Wilson score interval, says how
   much that count is actually worth.
4. **Per-unit table.**  Everything above, unit by unit, so nothing is hidden
   behind an aggregate.

This is an audit, not a pre-registered analysis: it was written after the
benchmark ran and it exists to *limit* the claims that can be made.

Run::

    python examples/benchmark_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import os

DEFAULT_RESULTS = "benchmark/results.json"
DEFAULT_OUT = "benchmark/statistical_audit.json"
ALPHA = 0.05


# --------------------------------------------------------------------------
def collect_units(results: dict) -> list:
    """Flatten ``results.json`` into one record per (dataset, condition) cloud."""
    units = []
    for ds in results["datasets"]:
        for cond, blk in ds["conditions"].items():
            mantel = blk.get("mantel") or {}
            units.append({
                "unit": f"{ds['key']}:{cond}",
                "accession": ds["accession"],
                "expected_topology": ds["expected_topology"],
                "p_h1_null": blk["null"]["p_value"],
                "h1_max_rel": blk.get("h1_max_rel"),
                "circular_advantage": mantel.get("circular_advantage"),
                "p_circular": mantel.get("p_circular"),
                "verdict": blk["verdict"],
            })
    units.sort(key=lambda u: u["unit"])
    return units


def bh_adjust(pvalues: list) -> list:
    """Benjamini-Hochberg adjusted p-values (NaN-safe, order preserving)."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adj = [float("nan")] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        val = min(prev, pvalues[i] * n / k)
        adj[i] = val
        prev = val
    return adj


def wilson_interval(k: int, n: int, z: float = 1.959963985):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (sum of all outcomes no more likely)."""
    from math import comb

    def pmf(i):
        return comb(n, i) * p ** i * (1 - p) ** (n - i)

    obs = pmf(k)
    tol = 1e-12
    return float(min(1.0, sum(pmf(i) for i in range(n + 1)
                              if pmf(i) <= obs + tol)))


def accuracy(units: list, predict_loop) -> dict:
    """Accuracy of a rule ``predict_loop(unit) -> bool`` against the design."""
    hits = [u for u in units
            if predict_loop(u) == (u["expected_topology"] == "loop")]
    return {"n_correct": len(hits), "n_units": len(units),
            "accuracy": round(len(hits) / len(units), 4),
            "misses": [u["unit"] for u in units if u not in hits]}


# --------------------------------------------------------------------------
def audit(results: dict) -> dict:
    units = collect_units(results)
    n = len(units)

    # 1. ablation -----------------------------------------------------------
    # ``benchmark_verdict`` replays the decision rule actually used by the
    # benchmark, where both ``loop-like`` and ``loop-like (partial)`` count as
    # a loop call.  ``strict_and`` is the conservative variant that demands
    # both evidence lines at once.
    def _h1_ok(u):
        return u["p_h1_null"] < ALPHA

    def _mantel_ok(u):
        return (u["circular_advantage"] is not None
                and u["circular_advantage"] > 0
                and u["p_circular"] is not None
                and u["p_circular"] < ALPHA)

    rules = {
        "h1_only": _h1_ok,
        "mantel_only": _mantel_ok,
        "strict_and_(both_lines)": lambda u: _h1_ok(u) and _mantel_ok(u),
        "benchmark_verdict_(either_line)": lambda u: (
            u["verdict"].startswith("loop-like")),
    }
    ablation = {name: accuracy(units, rule) for name, rule in rules.items()}

    # 2. multiplicity -------------------------------------------------------
    p_h1 = [u["p_h1_null"] for u in units]
    p_h1_bh = bh_adjust(p_h1)
    p_h1_bonf = [min(1.0, p * n) for p in p_h1]
    h1_survivors_bh = [u["unit"] for u, a in zip(units, p_h1_bh) if a < ALPHA]

    with_mantel = [u for u in units if u["p_circular"] is not None]
    m = len(with_mantel)
    p_circ = [u["p_circular"] for u in with_mantel]
    p_circ_bh = bh_adjust(p_circ)
    p_circ_bonf = [min(1.0, p * m) for p in p_circ]
    circ_survivors_bh = [u["unit"] for u, a in zip(with_mantel, p_circ_bh)
                         if a < ALPHA]
    circ_survivors_bonf = [u["unit"] for u, p in zip(with_mantel, p_circ_bonf)
                           if p < ALPHA]

    # 3. resolution of the headline ----------------------------------------
    k = ablation["benchmark_verdict_(either_line)"]["n_correct"]
    lo, hi = wilson_interval(k, n)
    resolution = {
        "n_correct": k, "n_units": n,
        "exact_binomial_two_sided_p_vs_0.5": round(
            exact_binomial_two_sided(k, n), 4),
        "wilson_95_ci": [round(lo, 3), round(hi, 3)],
        "units_needed_for_80pct_power": _units_needed_for_80pct_power(k / n),
    }

    # 4. per-unit table -----------------------------------------------------
    table = []
    for u, a_bh, a_bonf in zip(units, p_h1_bh, p_h1_bonf):
        table.append({
            **u,
            "p_h1_null_bh": round(a_bh, 4),
            "p_h1_null_bonferroni": round(a_bonf, 4),
            "matches_expectation": (u["verdict"].startswith("loop-like")
                                    == (u["expected_topology"] == "loop")),
        })

    return {
        "note": ("Post-hoc statistical audit of the pre-registered benchmark: "
                 "ablation, multiplicity and resolution.  It exists to bound "
                 "the claims, not to add new ones."),
        "n_units": n,
        "alpha": ALPHA,
        "ablation": ablation,
        "multiplicity": {
            "h1_family": {
                "n": n,
                "min_p": round(min(p_h1), 4),
                "smallest_bh_threshold": round(ALPHA / n, 4),
                "survivors_bh": h1_survivors_bh,
                "survivors_bonferroni": [u["unit"] for u in units
                                         if min(1.0, u["p_h1_null"] * n) < ALPHA],
            },
            "mantel_family": {
                "n": m,
                "min_p": round(min(p_circ), 4),
                "smallest_bh_threshold": round(ALPHA / m, 4),
                "survivors_bh": circ_survivors_bh,
                "survivors_bonferroni": circ_survivors_bonf,
            },
        },
        "resolution": resolution,
        "per_unit": table,
    }


def _units_needed_for_80pct_power(phat: float) -> int:
    """Rough n for 80 % power detecting ``phat`` against a 50 % baseline."""
    if not 0.5 < phat < 1.0:
        return -1
    z_a, z_b = 1.959963985, 0.841621234
    p0 = 0.5
    num = (z_a * math.sqrt(p0 * (1 - p0))
           + z_b * math.sqrt(phat * (1 - phat))) ** 2
    return int(math.ceil(num / (phat - p0) ** 2))


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    with open(args.results, encoding="utf-8") as fh:
        results = json.load(fh)

    rep = audit(results)

    print("=" * 78)
    print("benchmark statistical audit")
    print("=" * 78)

    print("\n[1] ablation - which evidence line carries the accuracy?")
    for name, a in rep["ablation"].items():
        print(f"  {name:<24} {a['n_correct']}/{a['n_units']} "
              f"({a['accuracy']*100:.0f}%)"
              + (f"   misses: {', '.join(a['misses'])}" if a["misses"] else ""))

    print("\n[2] multiplicity across the family of clouds")
    for fam, d in rep["multiplicity"].items():
        print(f"  {fam:<14} n={d['n']:<3} min p={d['min_p']:<7} "
              f"smallest BH threshold={d['smallest_bh_threshold']:<7} "
              f"BH survivors={len(d['survivors_bh'])} "
              f"Bonferroni survivors={len(d['survivors_bonferroni'])}")

    r = rep["resolution"]
    print("\n[3] resolution of the headline success count")
    print(f"  {r['n_correct']}/{r['n_units']} correctly classified")
    print(f"  exact binomial vs 50% baseline : p = "
          f"{r['exact_binomial_two_sided_p_vs_0.5']}")
    print(f"  Wilson 95% CI                  : "
          f"[{r['wilson_95_ci'][0]}, {r['wilson_95_ci'][1]}]")
    print(f"  units for 80% power            : "
          f"{r['units_needed_for_80pct_power']}")

    print("\n[4] per unit")
    print(f"  {'unit':<36}{'expect':<7}{'p_H1':>8}{'p_H1_BH':>9}"
          f"{'adv':>8}{'p_circ':>9}  match")
    for u in rep["per_unit"]:
        adv = (f"{u['circular_advantage']:>8.3f}"
               if u["circular_advantage"] is not None else " " * 8)
        pc = (f"{u['p_circular']:>9.4f}"
              if u["p_circular"] is not None else " " * 9)
        print(f"  {u['unit']:<36}{u['expected_topology']:<7}"
              f"{u['p_h1_null']:>8.4f}{u['p_h1_null_bh']:>9.4f}{adv}{pc}"
              f"  {u['matches_expectation']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"\n[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
