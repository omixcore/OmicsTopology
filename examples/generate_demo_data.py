"""Write the two-condition demo dataset to disk for the command-line workflow.

Produces an expression matrix and a matching group file, so the CLI can be
exercised without hand-building inputs::

    python examples/generate_demo_data.py
    omictopo run --matrix examples/data/expression.tsv \\
                 --groups examples/data/groups.tsv \\
                 --labels control,salt_stress --outdir generated --plots
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from omictopo.datasets import two_state_omics  # noqa: E402
from omictopo.io import write_matrix  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.join("examples", "data"))
    parser.add_argument("--n-genes", type=int, default=600)
    parser.add_argument("--n-per-state", type=int, default=40)
    parser.add_argument("--shape", choices=["cycle", "shift"], default="cycle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    X, groups, labels = two_state_omics(
        n_genes=args.n_genes,
        n_per_state=args.n_per_state,
        state_b=args.shape,
        seed=args.seed,
    )
    n_rows = X.shape[0]
    samples = [f"{labels[g][:4]}_{i:03d}" for i, g in enumerate(groups)]

    matrix_path = os.path.join(args.outdir, "expression.tsv")
    group_path = os.path.join(args.outdir, "groups.tsv")

    write_matrix(matrix_path, X, rownames=samples,
                 colnames=[f"gene_{j:04d}" for j in range(X.shape[1])])
    with open(group_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("sample\tgroup\n")
        for name, g in zip(samples, groups):
            fh.write(f"{name}\t{labels[g]}\n")

    print(f"wrote {matrix_path}  ({n_rows} x {X.shape[1]})")
    print(f"wrote {group_path}    ({n_rows} labels)")
    print()
    print("run it with:")
    print(f"  omictopo run --matrix {matrix_path} --groups {group_path} \\")
    print(f"                --labels {','.join(labels)} --outdir generated --plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
