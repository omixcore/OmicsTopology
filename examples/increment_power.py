"""Is the topological cycle's edge real, or is 1 false positive against 2 an accident?

Why this script exists
----------------------
``examples/incremental_value.py`` runs 17 methods over 23 non-cyclic clouds and
finds the H1 cycle first (1 false positive) with t-SNE second (2) and Isomap
third (3).  An honest reading of that table is *not* "the topology is the most
specific method" — with 23 clouds, 1 and 2 and 3 are the same number.  The
benchmark was sized to ask whether the topology beats the seven fixed
transforms, and on that question the answer was not close (1 against 9).  It was
never sized to separate it from the best manifold learner.

This script sizes that comparison properly.  The negative controls are already
declared — the same four families, a ramp, pure noise, three clusters and a
branching trajectory — so drawing more of them is a power increase, not a new
hypothesis.  240 draws give a false-positive rate to about ±3 points, which is
enough to see a 4 % against 9 % difference if it is there, and enough to say
plainly that it is not there if it is not.

Draws here are independent of the benchmark's: the benchmark uses seeds
``7 + 101 s`` and this uses ``1000003 + 7919 i``, so the two negative sets share
no cloud.

Two things are measured at this scale, not one:

* the false-positive rate of each method, with a Wilson interval, pooled and
  broken down by negative family; and
* **how** each method stays quiet — by producing no order at all, or by
  producing one whose score fails to clear its own permutation null.  That
  distinction is the project's structural claim, and it is the part that should
  sharpen rather than blur as the number of controls grows.

Usage::

    python examples/increment_power.py --n-neg-per-family 60 --n-pos 40

Outputs ``benchmark/increment_power.json`` and ``docs/fig_increment_power.png``.
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

import incremental_value as iv  # noqa: E402
from learned_baselines import LEARNED_METHODS  # noqa: E402

from omictopo.stats import binom_test_two_sided, wilson_interval  # noqa: E402

# The nine methods worth spending 240 clouds on: the topological one, the two
# neighbour-graph learners that came close, the best fixed transform, the
# autoencoder the original objection was actually about, and the four deep
# variants the "give it real capacity" version of that objection asks for.
FIELD = {
    "h1_cycle": iv.CLASSICAL_METHODS["h1_cycle"],
    "tsne_angle": LEARNED_METHODS["tsne_angle"],
    "isomap_angle": LEARNED_METHODS["isomap_angle"],
    "circle_fit_angle": iv.CLASSICAL_METHODS["circle_fit_angle"],
    "ae_2d_angle": LEARNED_METHODS["ae_2d_angle"],
    "ae_deep_2d_angle": LEARNED_METHODS["ae_deep_2d_angle"],
    "dae_2d_angle": LEARNED_METHODS["dae_2d_angle"],
    "vae_2d_angle": LEARNED_METHODS["vae_2d_angle"],
    "umap_angle": LEARNED_METHODS["umap_angle"],
}

# already-declared negative families, unchanged from the benchmark
NEGATIVE_KINDS = ["synth_line", "synth_noise", "synth_clusters", "synth_branch"]
POSITIVE_KIND = "synth_ring"

NEG_SEED0, NEG_STEP = 1_000_003, 7_919
POS_SEED0, POS_STEP = 2_000_003, 7_919


def _draw(kind: str, i: int) -> tuple:
    return iv.make_synthetic(kind, seed=NEG_SEED0 + NEG_STEP * i)


def _draw_positive(i: int) -> tuple:
    return iv.make_synthetic(POSITIVE_KIND, seed=POS_SEED0 + POS_STEP * i)


def _rate(successes: int, n: int) -> dict:
    lo, hi = wilson_interval(successes, n)
    return {"k": int(successes), "n": int(n),
            "rate": round(successes / n, 4) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def _required_n(p1: float, p2: float, alpha: float = 0.05,
                power: float = 0.8) -> int | None:
    """Clouds per group for an unpaired two-proportion test.

    Reported so the reader can see what a *confirmed* difference of this size
    would cost; it is not a claim about the difference actually measured.
    """
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0) or abs(p1 - p2) < 1e-9:
        return None
    from scipy.stats import norm
    pbar = 0.5 * (p1 + p2)
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    num = (z_a * np.sqrt(2.0 * pbar * (1.0 - pbar))
           + z_b * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def _summarise(records: list, name: str) -> dict:
    """Rate, interval and abstention mechanism for one method over many clouds."""
    declared = [r for r in records if r["methods"][name]["declares"]]
    no_order = [r for r in records if r["methods"][name]["null"] is None]
    return {
        "rate": _rate(len(declared), len(records)),
        "abstention": {
            "no_order_at_all": _rate(len(no_order), len(records)),
            "silent_by_score": _rate(
                len(records) - len(declared) - len(no_order), len(records)),
        },
    }


def _mcnemar(records: list, a: str, b: str) -> dict:
    """Paired comparison on the same clouds: who declared and who did not."""
    only_a = sum(1 for r in records
                 if r["methods"][a]["declares"] and not r["methods"][b]["declares"])
    only_b = sum(1 for r in records
                 if r["methods"][b]["declares"] and not r["methods"][a]["declares"])
    both = sum(1 for r in records
               if r["methods"][a]["declares"] and r["methods"][b]["declares"])
    discordant = only_a + only_b
    p = binom_test_two_sided(only_a, discordant) if discordant else 1.0
    return {"a": a, "b": b, "only_a": int(only_a), "only_b": int(only_b),
            "both": int(both), "discordant": int(discordant),
            "p_value": float(f"{float(p):.6e}")}      # 7 significant digits


def make_figure(table: dict, by_family: dict, figdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(table, key=lambda k: table[k]["false_positive_rate"]["rate"])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.15, 0.85]})

    ax = axes[0]
    x = np.arange(len(names))
    rate = [table[n]["false_positive_rate"]["rate"] for n in names]
    lo = [table[n]["false_positive_rate"]["rate"]
          - table[n]["false_positive_rate"]["ci95"][0] for n in names]
    hi = [table[n]["false_positive_rate"]["ci95"][1]
          - table[n]["false_positive_rate"]["rate"] for n in names]
    colors = ["#c0392b" if n == "h1_cycle" else
              ("#2c7fb8" if "angle" in n and n == "circle_fit_angle" else "#6a3d9a")
              for n in names]
    ax.bar(x, rate, yerr=[lo, hi], capsize=4, color=colors, zorder=3)
    for i, n in enumerate(names):
        t = table[n]["false_positive_rate"]
        ax.text(i, t["rate"] + hi[i] + 0.012, f"{t['k']}/{t['n']}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("false positives on non-cyclic clouds")
    ax.set_ylim(0, max(rate) * 1.45 + 0.05)
    ax.set_title("A  the same number, measured properly", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    fams = sorted(by_family)
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [by_family[f][n]["rate"] for f in fams]
        ax.bar(np.arange(len(fams)) + i * w - 0.4 + w / 2, vals, width=w,
               label=n.replace("_", " "),
               color="#c0392b" if n == "h1_cycle" else None,
               zorder=3)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([f.replace("synth_", "") for f in fams], fontsize=8)
    ax.set_ylabel("false positives")
    ax.set_title("B  split by negative family", fontsize=10)
    ax.legend(fontsize=7.2, ncol=2)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    for i, n in enumerate(names):
        t, ab = table[n], table[n]["abstention"]
        refused = ab["no_order_at_all"]["rate"]
        silent = ab["silent_by_score"]["rate"]
        ax.barh(i, refused, color="#4a4a4a", zorder=3,
                label="no order produced at all" if i == 0 else None)
        ax.barh(i, silent, left=refused, color="#9ecae1", zorder=3,
                label="ordered, score below its null" if i == 0 else None)
        ax.barh(i, t["false_positive_rate"]["rate"], left=refused + silent,
                color="#c0392b", zorder=3,
                label="declared (false positive)" if i == 0 else None)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of non-cyclic clouds")
    ax.set_title("C  refusing, versus answering\nwithout conviction", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("The topology against the best learned baselines, at a sample "
                 "size that can tell them apart", fontsize=11.5)
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "fig_increment_power.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-neg-per-family", type=int, default=60)
    ap.add_argument("--n-pos", type=int, default=40)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--outdir", default=os.path.join("benchmark"))
    ap.add_argument("--figdir", default="docs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 100)
    print("OmicsTopology - is the topology's edge over the best learned baseline real?")
    print("=" * 100)
    n_neg = args.n_neg_per_family * len(NEGATIVE_KINDS)
    print(f"{len(FIELD)} methods, {n_neg} non-cyclic clouds "
          f"({len(NEGATIVE_KINDS)} declared families x {args.n_neg_per_family}), "
          f"{args.n_pos} cyclic clouds, {args.n_perm} permutations.")
    print("Draws are independent of the benchmark's (different seed schedule).\n")

    records: list[dict] = []
    by_family: dict = {}

    t0 = time.time()
    for kind in NEGATIVE_KINDS:
        fam = []
        for i in range(int(args.n_neg_per_family)):
            X, phases, hours = _draw(kind, i)
            rec = iv.run_cloud(f"{kind}#{i}", X, hours, 24.0,
                               n_perm=args.n_perm, seed=args.seed, methods=FIELD)
            rec.update(group="negative", kind=kind)
            fam.append(rec)
        records.extend(fam)
        by_family[kind] = {n: _summarise(fam, n)["rate"] for n in FIELD}
        done = len(records)
        print(f"  {kind:<16} {args.n_neg_per_family:3d} clouds   "
              f"[{done}/{n_neg}]  {time.time() - t0:5.0f}s")

    positives = []
    for i in range(int(args.n_pos)):
        X, phases, hours = _draw_positive(i)
        rec = iv.run_cloud(f"synth_ring#{i}", X, hours, 24.0,
                           n_perm=args.n_perm, seed=args.seed, methods=FIELD)
        rec.update(group="positive", kind=POSITIVE_KIND)
        positives.append(rec)

    table = {}
    for n in FIELD:
        rec = _summarise(records, n)
        table[n] = {
            "false_positive_rate": rec["rate"],
            "abstention": rec["abstention"],
            "sensitivity": _rate(
                sum(1 for r in positives if r["methods"][n]["declares"]),
                len(positives)),
        }

    print("\n" + "-" * 96)
    print(f"{'method':<20}{'sensitivity':>24}{'false positives':>26}"
          f"{'fp k/n':>10}{'no order':>10}{'silent':>9}")
    for n in sorted(table, key=lambda k: table[k]["false_positive_rate"]["rate"]):
        t = table[n]
        s, f = t["sensitivity"], t["false_positive_rate"]
        star = " *" if n == "h1_cycle" else "  "
        ab = t["abstention"]
        print(f"{n:<20}{s['rate']:>8.3f} [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]"
              f"{f['rate']:>10.3f} [{f['ci95'][0]:.2f},{f['ci95'][1]:.2f}]"
              f"{f['k']:>5}/{f['n']:<4}"
              f"{ab['no_order_at_all']['rate']:>10.3f}"
              f"{ab['silent_by_score']['rate']:>9.3f}{star}")
    print("  * topological method")

    print("\n  paired comparison on the same non-cyclic clouds "
          "(McNemar, exact)")
    pairs = {}
    for rival in [n for n in FIELD if n != "h1_cycle"]:
        mc = _mcnemar(records, "h1_cycle", rival)
        pairs[rival] = mc
        verdict = ("indistinguishable" if mc["p_value"] > 0.05
                   else f"significantly fewer for {mc['a']}")
        print(f"  h1_cycle vs {rival:<18} only-h1={mc['only_a']:>3}  "
              f"only-{rival[:9]}={mc['only_b']:>3}  "
              f"p={mc['p_value']:.2e}  {verdict}")

    fp_topo = table["h1_cycle"]["false_positive_rate"]
    fp_best = table["tsne_angle"]["false_positive_rate"]
    need = _required_n(fp_topo["rate"], fp_best["rate"])
    print(f"\n  to confirm a difference of {fp_best['rate'] - fp_topo['rate']:+.3f} "
          f"in false-positive rate as an unpaired comparison would take "
          f"{need if need else 'n/a'} clouds per method at 80% power.")

    figures = []
    if not args.no_figures:
        try:
            figures.append(make_figure(table, by_family, args.figdir))
        except ImportError:
            print("\nmatplotlib unavailable; figure skipped")

    blob = {
        "config": {"n_neg_per_family": int(args.n_neg_per_family),
                   "n_pos": int(args.n_pos), "n_perm": int(args.n_perm),
                   "seed": args.seed,
                   "negative_families": NEGATIVE_KINDS,
                   "positive_kind": POSITIVE_KIND,
                   "seed_schedule": {"negative": [NEG_SEED0, NEG_STEP],
                                     "positive": [POS_SEED0, POS_STEP]}},
        "question": "is the false-positive gap between the H1 cycle and the best "
                    "learned baseline larger than sampling noise?",
        "methods": list(FIELD),
        "summary": table,
        "by_negative_family": by_family,
        "paired": pairs,
        "clouds_per_method_for_80pc_power": need,
        "per_cloud": records + positives,
        "figures": figures,
    }
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "increment_power.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(f"\n[increment-power] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
