"""Statistical power of the topology-change test, measured across datasets.

**Why this exists.**  A single p-value from a single synthetic dataset means
very little — with 100 permutations, p = 0.03 is the 4th smallest value
attainable, and it can happen by luck.  What matters is:

* **detection rate** — how often the test fires when the topology really did
  change (a genuine ring appears);
* **false-positive rate** — how often it fires when only the mean shifted.

Both are estimated here by regenerating the data under many different random
seeds and re-running the whole pipeline each time.  Run this before quoting
any single p-value in a paper.

The statistic matters enormously.  ``h1_total`` sums *all* H1 bars, so the
short noise loops dilute a single genuine ring; ``h1_max`` looks only at the
longest bar, and ``bottleneck`` at the worst-matched feature.  On the default
settings the difference is the difference between an unusable test and a
clean one — see the table at the end of this script's output.

Usage::

    python examples/power_analysis.py
    python examples/power_analysis.py --seeds 20 --n-permutations 200
    python examples/power_analysis.py --statistics h1_total,h1_max --backend builtin
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from omictopo import __version__  # noqa: E402
from omictopo.datasets import two_state_omics  # noqa: E402
from omictopo.geometry import pca_embed, standardize  # noqa: E402
from omictopo.stats import permutation_test_pointclouds  # noqa: E402

# Bottleneck and Wasserstein cost real time because the null distribution
# rebuilds every diagram; the plain persistence summaries are cheap.
DEFAULT_STATISTICS = ["h1_total", "h1_max", "h1_count", "bottleneck", "wasserstein"]


def p_values(
    kind: str,
    statistic: str,
    seeds,
    n_genes: int,
    n_per_state: int,
    n_permutations: int,
    max_simplices: int,
    backend: str,
    n_components: int,
) -> np.ndarray:
    """Run the full test once per seed and collect the p-values."""
    out = []
    for seed in seeds:
        X, groups, _ = two_state_omics(
            n_genes=n_genes, n_per_state=n_per_state, state_b=kind, seed=seed
        )
        groups = np.asarray(groups)
        emb = pca_embed(standardize(X), n_components)
        result = permutation_test_pointclouds(
            emb[groups == 0],
            emb[groups == 1],
            statistic=statistic,
            dim=1,
            n_permutations=n_permutations,
            seed=0,
            max_points=None,
            max_dim=1,
            max_simplices=max_simplices,
            backend=backend,
        )
        out.append(result["p_value"])
    return np.asarray(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--statistics", default=",".join(DEFAULT_STATISTICS),
                        help="comma-separated list of test statistics")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of independent datasets per case")
    parser.add_argument("--n-genes", type=int, default=600)
    parser.add_argument("--n-per-state", type=int, default=40)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--max-simplices", type=int, default=80_000)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "builtin", "ripser"])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    stats = [s.strip() for s in args.statistics.split(",") if s.strip()]
    seeds = list(range(int(args.seeds)))
    cases = [("cycle", "ring is real", "detection"), ("shift", "mean shift only", "false positive")]

    print(f"omictopo {__version__} — statistical power of the permutation test")
    print(f"  datasets per case : {len(seeds)} (seeds 0..{len(seeds)-1})")
    print(f"  each dataset      : {args.n_genes} genes x {args.n_per_state} samples/state")
    print(f"  permutations      : {args.n_permutations}")
    print(f"  backend           : {args.backend}")
    print(f"  significance      : p < {args.alpha}")
    print()

    collected: dict = {}
    t0 = time.time()
    for kind, desc, _label in cases:
        for statistic in stats:
            ps = p_values(kind, statistic, seeds, args.n_genes, args.n_per_state,
                          args.n_permutations, args.max_simplices, args.backend,
                          args.n_components)
            collected[(kind, statistic)] = ps
            print(f"  [{time.time()-t0:6.1f}s] {kind:<6} {statistic:<12} "
                  f"median p = {np.median(ps):.3f}   fired {int((ps < args.alpha).sum())}/{len(ps)}")
    print()

    width = 78
    print("=" * width)
    header = f"{'statistic':<14}{'detection (ring)':>22}{'false positive (shift)':>26}"
    print(header)
    print("-" * width)
    for statistic in stats:
        det = collected[("cycle", statistic)]
        fp = collected[("shift", statistic)]
        det_rate = float((det < args.alpha).mean())
        fp_rate = float((fp < args.alpha).mean())
        print(f"{statistic:<14}"
              f"{f'{(det<args.alpha).sum()}/{len(det)} = {det_rate:.0%}':>22}"
              f"{f'{(fp<args.alpha).sum()}/{len(fp)} = {fp_rate:.0%}':>26}")
    print("=" * width)
    print("A usable statistic wants detection near 100% and false positives near the")
    print(f"nominal {args.alpha:.0%}.  `run_states` defaults to 'h1_max'.")
    print(f"total time {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
