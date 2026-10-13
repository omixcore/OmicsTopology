"""Validate the implementation against point clouds with known topology.

Every case here has an answer you can write down before running the code, so
this script doubles as an executable correctness check::

    python examples/demo_known_shapes.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from omictopo import bottleneck_distance, pairwise_distances, persistent_homology  # noqa: E402
from omictopo.datasets import blob, circle, torus, two_clusters  # noqa: E402

CASES = [
    ("circle(24)", circle(24), "one loop", lambda d: (d.betti(1, 0.5), 1)),
    ("two_clusters()", two_clusters(), "two basins", lambda d: (d.betti(0, 0.2), 2)),
    ("blob(30)", blob(30), "no structure", lambda d: (d.max_persistence(1) < 0.1, True)),
    ("torus()", torus(), "loops in two directions",
     lambda d: (np.count_nonzero(np.sort(d.persistence(1))[::-1] > 0.5) >= 2, True)),
]


def main() -> int:
    print(f"{'point cloud':<18}{'expected':<28}{'measured':<24}{'ok'}")
    print("-" * 76)
    ok = True

    for name, points, expectation, check in CASES:
        dgm = persistent_homology(points, max_dim=1)
        measured, expected = check(dgm)
        passed = measured == expected
        ok &= passed
        extra = ""
        if name.startswith("circle"):
            extra = f" (loop persistence {dgm.max_persistence(1):.3f})"
        print(f"{name:<18}{expectation:<28}{str(measured):<24}{'PASS' if passed else 'FAIL'}{extra}")

    print()
    print("H0 from two independent implementations (must be identical):")
    D = pairwise_distances(circle(24))
    mst = persistent_homology(D=D, max_dim=0, h0="mst")
    red = persistent_homology(D=D, max_dim=0, h0="reduction", threshold=float(D.max()))
    same = np.allclose(np.sort(mst.finite(0)[:, 1]), np.sort(red.finite(0)[:, 1]))
    print(f"  MST vs boundary-matrix reduction: {'identical' if same else 'MISMATCH'}")
    ok &= bool(same)

    print()
    with_loop = persistent_homology(circle(24), max_dim=1)
    without = persistent_homology(blob(24, dims=2), max_dim=1)
    print("H1 bottleneck distance, circle vs blob: "
          f"{bottleneck_distance(with_loop, without, dim=1):.4f} (should be large)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
