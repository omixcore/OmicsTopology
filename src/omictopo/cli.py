"""Command-line interface.

Examples
--------
::

    omictopo run --matrix expression.tsv --groups groups.tsv \\
                 --labels control,salt_stress --outdir results --plots

    omictopo version
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .io import read_groups, read_matrix
from .pipeline import run_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omictopo",
        description=(
            "Topological data analysis of omics state spaces: persistent "
            "homology of expression state spaces, and a permutation test for "
            "whether the topology changed between two conditions."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full state comparison")
    run.add_argument("--matrix", required=True,
                     help="expression matrix (TSV/CSV), rows = samples, columns = features")
    run.add_argument("--groups", required=True,
                     help="group labels: one column, or two columns sample<TAB>group")
    run.add_argument("--labels", default=None,
                     help="comma-separated state names, in order of sorted group codes")
    run.add_argument("--outdir", default=None, help="directory for tables and figures")
    run.add_argument("--delimiter", default=None, help="override the delimiter")
    run.add_argument("--n-components", type=int, default=3, help="PCA components (default 3)")
    run.add_argument("--whiten", action="store_true", help="whiten the PCA embedding")
    run.add_argument("--metric", default="euclidean",
                     choices=["euclidean", "cosine", "correlation", "manhattan"])
    run.add_argument("--max-points", type=int, default=120,
                     help="per-state subsample cap (default 120)")
    run.add_argument("--max-dim", type=int, default=1, choices=[0, 1, 2],
                     help="highest homology dimension (default 1)")
    run.add_argument("--max-simplices", type=int, default=80000,
                     help="simplex budget for the automatic threshold")
    run.add_argument("--backend", default="builtin",
                     choices=["builtin", "ripser", "auto"],
                     help="persistent-homology backend (default builtin; "
                          "'ripser' is ~100x faster but ignores the threshold)")
    run.add_argument("--statistic", default="h1_max",
                     choices=["h1_max", "h1_total", "h1_count",
                              "bottleneck", "wasserstein"],
                     help="permutation-test statistic (default h1_max)")
    run.add_argument("--n-permutations", type=int, default=100,
                     help="permutations for the significance test (default 100)")
    run.add_argument("--no-test", action="store_true", help="skip the permutation test")
    run.add_argument("--plots", action="store_true", help="write figures into --outdir")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--verbose", action="store_true")

    sub.add_parser("version", help="print the version and exit")
    return parser


def _align(matrix, rownames, group_ids, labels):
    """Reorder matrix rows so they follow the order of the group file."""
    if not group_ids:
        return matrix, labels
    if len(group_ids) != matrix.shape[0]:
        raise SystemExit(
            f"group file has {len(group_ids)} rows but the matrix has "
            f"{matrix.shape[0]}; they must match"
        )
    if group_ids == list(rownames):
        return matrix, labels
    index = {name: i for i, name in enumerate(rownames)}
    missing = [g for g in group_ids if g not in index]
    if missing:
        raise SystemExit(f"group file references unknown samples: {missing[:5]}")
    order = [index[g] for g in group_ids]
    return matrix[order], labels


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(f"omictopo {__version__}")
        return 0

    matrix, rownames, _ = read_matrix(args.matrix, delimiter=args.delimiter)
    group_ids, labels = read_groups(args.groups, delimiter=args.delimiter)
    matrix, labels = _align(matrix, rownames, group_ids, labels)

    if not labels or any(l == "" for l in labels):
        raise SystemExit("the group file contains empty labels")

    names = args.labels.split(",") if args.labels else None
    if names is not None:
        names = [n.strip() for n in names]

    groups = np.asarray(labels)
    unique = list(dict.fromkeys(groups.tolist()))
    if len(unique) != 2:
        print(
            f"[omictopo] note: {len(unique)} states detected "
            f"({', '.join(map(str, unique))}); the permutation test needs exactly 2",
            file=sys.stderr,
        )

    report = run_states(
        matrix,
        groups,
        labels=names,
        n_components=args.n_components,
        whiten=args.whiten,
        metric=args.metric,
        max_points=None if args.max_points <= 0 else args.max_points,
        max_dim=args.max_dim,
        max_simplices=args.max_simplices,
        backend=args.backend,
        statistic=args.statistic,
        n_permutations=args.n_permutations,
        run_test=not args.no_test,
        plots=args.plots,
        outdir=args.outdir,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(report.summary_text())
    if args.outdir:
        print("\nwritten:")
        for path in report.files:
            print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
