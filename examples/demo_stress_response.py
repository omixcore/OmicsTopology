"""Does salt stress change the *shape* of the transcriptome state space?

This is the headline example.  It builds a synthetic two-condition expression
matrix whose latent geometry is known:

* ``control``  — a single homogeneous basin;
* ``salt_stress`` — a ring, i.e. cells progressively moving through a cycle of
  transcriptional programmes.

A ring carries a real H1 class (a loop), a single basin does not, so the
topological test should fire.  The companion run with ``--control-shape shift``
replaces the ring with a mere mean shift: the same marginal difference, no
topological change, and the test should *not* fire.  Running both is the point
— a topology test that fires on everything is worthless.

Usage::

    python examples/demo_stress_response.py
    python examples/demo_stress_response.py --control-shape shift
    python examples/demo_stress_response.py --outdir generated --n-permutations 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from omictopo import __version__, run_states  # noqa: E402
from omictopo.datasets import two_state_omics  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-genes", type=int, default=600)
    parser.add_argument("--n-per-state", type=int, default=40)
    parser.add_argument("--control-shape", choices=["cycle", "shift"], default="cycle",
                        help="'cycle' is the positive case, 'shift' the negative control")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--outdir", default=None, help="write tables and figures here")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    X, groups, labels = two_state_omics(
        n_genes=args.n_genes,
        n_per_state=args.n_per_state,
        state_b=args.control_shape,
        seed=args.seed,
    )
    print(f"omictopo {__version__}")
    print(f"expression matrix: {X.shape[0]} samples x {X.shape[1]} genes")
    print(f"condition B shape : {args.control_shape}")
    if args.control_shape == "shift":
        print("(negative control: same mean shift, no topological change expected)")
    print()

    report = run_states(
        X,
        groups,
        labels=labels,
        n_components=3,
        max_points=args.n_per_state,
        max_simplices=200_000,
        n_permutations=args.n_permutations,
        plots=not args.no_plots,
        outdir=args.outdir,
        seed=args.seed,
        verbose=True,
    )
    print()
    print(report.summary_text())

    control = report.summaries[labels[0]]
    stress = report.summaries[labels[1]]
    ratio = stress["h1_max_persistence"] / max(control["h1_max_persistence"], 1e-12)
    print()
    print(f"strongest loop, {labels[1]} vs {labels[0]}: {ratio:.1f}x")
    print("interpretation: a loop in the stress state means the samples are "
          "arranged around a cycle, not merely shifted along one axis.")

    if args.outdir:
        print()
        print("written:")
        for path in report.files:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
