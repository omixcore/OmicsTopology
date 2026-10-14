"""Determinism audit: the same commands, run twice in separate processes.

A reproduction claim rests on the runs being deterministic.  This script
demonstrates rather than asserts it: it executes the same seeded pipeline in
two fresh interpreters and compares the sha256 of their outputs.

    python tools/determinism_audit.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# A seeded end-to-end slice of the pipeline: build a synthetic circadian
# cloud, score it, and print the numbers.  Run twice, hash the stdout.
PROBE = r"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from omictopo.geometry import standardize, pairwise_distances
from omictopo import persistent_homology
from omictopo.cycles import circular_succession_score

angles = np.sort(np.random.default_rng(11).uniform(0.0, 2.0 * np.pi, 24))
pattern = np.random.default_rng(12).normal(size=(300, 2))
X = np.c_[np.cos(angles), np.sin(angles)] @ pattern.T
X = standardize(X + np.random.default_rng(13).normal(0.0, 0.35, X.shape))
D = pairwise_distances(X)
dgm = persistent_homology(D=D, max_dim=1)
order = np.argsort(np.arctan2(*(np.random.default_rng(14).normal(size=(2, 24)))))
score = circular_succession_score(angles, order)
print(json.dumps({
    "h1_bars": [[round(float(b), 9), round(float(d), 9)]
                for b, d in sorted(np.asarray(dgm.finite(1))[:, :2].tolist())],
    "score": round(float(score), 9),
}))
"""


def run_probe() -> str:
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return r.stdout.strip()


def main() -> int:
    print("running the seeded probe twice in separate interpreters ...")
    first = run_probe()
    second = run_probe()
    h1 = hashlib.sha256(first.encode()).hexdigest()
    h2 = hashlib.sha256(second.encode()).hexdigest()
    ok = h1 == h2
    print(f"  run 1 sha256 {h1[:24]}...")
    print(f"  run 2 sha256 {h2[:24]}...")
    print(f"  identical: {ok}")

    # the whole demo, twice: exercises the public pipeline end to end
    outs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                [sys.executable, str(REPO / "examples" / "demo_known_shapes.py")],
                cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                print("  demo_known_shapes FAILED:", r.stdout[-300:])
                return 1
            outs.append(hashlib.sha256(
                "\n".join(l for l in r.stdout.splitlines()
                          if "persistence" in l or "PASS" in l
                          or "identical" in l).encode()).hexdigest())
    ok2 = outs[0] == outs[1]
    print(f"  demo_known_shapes stable across runs: {ok2}")

    print("\ndeterminism audit:", "PASS" if (ok and ok2) else "FAIL")
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
