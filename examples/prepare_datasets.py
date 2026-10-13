"""Prepare all benchmark datasets into a single, uniform on-disk format.

Each dataset becomes ``<outdir>/<key>/`` containing

``matrix.tsv``
    samples x features, ``log2(expression + 1)``, header row of feature ids;
``design.tsv``
    ``sample``, ``condition``, ``time``, ``replicate`` — everything the
    benchmark needs to slice a dataset by time or treatment.

Examples
--------
::

    python examples/prepare_datasets.py --outdir data/datasets
    python examples/prepare_datasets.py --outdir data/datasets --dataset rice_diurnal
    python examples/prepare_datasets.py --outdir data/datasets --top-var 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset_registry import SPECS, prepare  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="data/datasets",
                    help="root directory; one sub-directory per dataset")
    ap.add_argument("--dataset", default="all",
                    help="'all' or a comma-separated list of dataset keys")
    ap.add_argument("--cache", default=None,
                    help="download cache (default: <outdir>/../geo_cache)")
    ap.add_argument("--min-expr", type=float, default=1.0,
                    help="a feature must reach this value (CPM/FPKM/TPM) in --min-frac samples")
    ap.add_argument("--min-frac", type=float, default=0.25)
    ap.add_argument("--top-var", type=int, default=3000,
                    help="keep the N most variable features (0 = keep all)")
    args = ap.parse_args(argv)

    keys = list(SPECS) if args.dataset == "all" else [k.strip() for k in args.dataset.split(",")]
    unknown = [k for k in keys if k not in SPECS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}; known: {list(SPECS)}")

    cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.outdir)),
                                       "geo_cache")
    manifest = {}
    for key in keys:
        outdir = os.path.join(args.outdir, key)
        print(f"\n=== {key} ({SPECS[key].accession}, {SPECS[key].organism}) ===")
        t0 = time.time()
        try:
            meta = prepare(key, outdir, cache=cache, min_expr=args.min_expr,
                           min_frac=args.min_frac, top_var=args.top_var)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            manifest[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        meta["seconds"] = round(time.time() - t0, 1)
        manifest[key] = meta
        print(f"  samples={meta['n_samples']}  features={meta['n_features_kept']} "
              f"(of {meta['n_features_expressed']} expressed)")
        print(f"  conditions={meta['conditions']}")
        print(f"  times={meta['times']}")
        print(f"  expected topology = {meta['expected_topology']}  "
              f"(period={meta['period_hours']})  [{meta['seconds']}s]")

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\n[prepare] wrote {manifest_path}")
    failed = [k for k, v in manifest.items() if "error" in v]
    if failed:
        print(f"[prepare] {len(failed)} dataset(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
