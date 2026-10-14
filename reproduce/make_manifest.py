"""Regenerate reproduce/expected/manifest.json from the committed benchmark data.

The manifest pins the sha256 of every file under ``benchmark/`` that a
published number depends on.  Re-run this *only* when a benchmark has been
genuinely re-computed and re-reviewed -- the manifest is the "these are the
data behind the paper" pin, so changing it is part of changing the paper, not
part of running code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmark"
OUT = Path(__file__).resolve().parent / "expected" / "manifest.json"

# every raw file a published number is recomputed from, plus the tables the
# text quotes directly
INCLUDE = [
    "incremental_value.json",
    "increment_power.json",
    "statistical_audit.json",
    "topology_without_labels.json",
    "cycle_counting_real.json",
    "cycle_counting_power.json",
    "estrous_grid_cycles.json",
    "axis_separability.json",
    "axis_geometry_multi.json",
    "ultradian_two_oscillator.json",
    "cycle_identifiability.json",
    "deep_tuning.json",
    "diagnosis_replicate_contribution.json",
    "diagnosis_period_drift.json",
    "results.json",
]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    files = {}
    missing = []
    for name in INCLUDE:
        path = BENCH / name
        if not path.exists():
            missing.append(name)
            continue
        files[f"benchmark/{name}"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    if missing:
        print("missing (skipped):", ", ".join(missing))
    manifest = {
        "note": "sha256 of the committed raw benchmark data; regenerate only "
                "when a benchmark is genuinely recomputed",
        "git_head": git_head(),
        "files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(files)} files (git head {manifest['git_head'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
