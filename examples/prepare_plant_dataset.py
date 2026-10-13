"""Prepare a real, public plant expression dataset for OmicsTopology.

The dataset is **GSE158444** (Bonnot & Nagel, PMID 33871647): *Arabidopsis
thaliana* seedlings sampled at eight times of day across a 24 h time course,
either kept at 22 °C (control) or shifted to 37 °C for 1 h (heat stress).
Two RNA fractions were sequenced — total mRNA (``TOT``) and ribosome-bound
"translating" mRNA (``TR``).  The processed count table is small enough to
work with directly, which makes the experiment a good first non-synthetic
test case for the package.

Why this experiment is a useful benchmark:

* it is real data, with real batch structure, real noise and real outliers;
* the sampling is a **time course over roughly one circadian period**, so the
  control state space ought to trace a closed loop — a topology we can
  predict in advance rather than one we discover and then rationalise;
* the heat arm gives a clean two-condition comparison for the permutation
  test, on samples collected at the same times of day.

The script downloads the processed counts, normalises them, filters
unreliable features and writes the three files ``omictopo run`` expects.

Examples
--------
::

    python examples/prepare_plant_dataset.py --outdir data/arabidopsis_heat
    python examples/prepare_plant_dataset.py --fraction TR --top-var 0

Notes
-----
Raw data are downloaded from NCBI GEO and are **not** redistributed with this
repository.  Re-running the script re-downloads them.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import urllib.request

import numpy as np

GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158444/suppl/"
    "GSE158444_Counts_Time.txt.gz"
)
ACCESSION = "GSE158444"
SAMPLE_RE = re.compile(r"^(TOT|TR)\.T(\d+)\.([CH])(\d)$")

# ``T<n>`` in the supplementary file maps onto the ZT clock time of that
# sampling; ZT48 is the first time point of the experiment.
HOUR_OF_TIME_POINT = {
    "0": 0, "3": 3, "6": 6, "9": 9, "12": 12, "15": 15, "18": 18, "21": 21,
    "01": 24,
}
TREATMENT = {"C": "control", "H": "heat"}


def download(cache_dir: str) -> str:
    """Fetch the processed count table, reusing a cached copy when present."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(GEO_URL))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[prepare] using cached {path}")
        return path
    print(f"[prepare] downloading {GEO_URL}")
    tmp = path + ".part"
    with urllib.request.urlopen(GEO_URL, timeout=300) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1 << 16)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp, path)
    print(f"[prepare] saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def read_counts(path: str):
    """Return ``(genes, samples, counts)`` with ``counts`` shaped genes x samples."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        samples = [s.strip() for s in header[1:]]
        genes: list = []
        rows: list = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            genes.append(parts[0].strip())
            rows.append([float(v) if v not in ("", "NA") else 0.0 for v in parts[1:]])
    counts = np.asarray(rows, dtype=float)
    print(f"[prepare] read {counts.shape[0]} features x {counts.shape[1]} libraries")
    return genes, samples, counts


def parse_design(samples):
    """Split sample names into (fraction, hour, treatment, replicate)."""
    design = []
    for name in samples:
        m = SAMPLE_RE.match(name)
        if not m:
            design.append(None)
            continue
        fraction, time_point, treat, rep = m.groups()
        design.append({
            "fraction": fraction,
            "time_point": time_point,
            "hour": HOUR_OF_TIME_POINT.get(time_point, -1),
            "treatment": TREATMENT[treat],
            "replicate": int(rep),
        })
    return design


def cpm_log(counts: np.ndarray) -> np.ndarray:
    """Counts-per-million followed by ``log2(x + 1)``."""
    lib = counts.sum(axis=0, keepdims=True)
    lib = np.where(lib > 0, lib, 1.0)
    return np.log2(counts / lib * 1e6 + 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="data/arabidopsis_heat",
                    help="where to write matrix.tsv / groups.tsv / design.tsv")
    ap.add_argument("--cache", default=None,
                    help="download cache (default: <outdir>/../geo_cache)")
    ap.add_argument("--fraction", default="TOT", choices=["TOT", "TR"],
                    help="TOT = total mRNA (default), TR = translating mRNA")
    ap.add_argument("--min-cpm", type=float, default=1.0,
                    help="a feature must reach this CPM in --min-frac of samples")
    ap.add_argument("--min-frac", type=float, default=0.25,
                    help="fraction of samples that must pass --min-cpm")
    ap.add_argument("--top-var", type=int, default=3000,
                    help="keep the N most variable features (0 = keep all)")
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="also keep time points that lack one treatment arm")
    args = ap.parse_args(argv)

    cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.outdir)),
                                       "geo_cache")
    path = download(cache)
    genes, samples, counts = read_counts(path)
    design = parse_design(samples)

    unparsed = [s for s, d in zip(samples, design) if d is None]
    if unparsed:
        print(f"[prepare] warning: {len(unparsed)} unparsed sample names, e.g. "
              f"{unparsed[:3]}", file=sys.stderr)

    # --- pick the samples we want ------------------------------------------
    keep = []
    for i, d in enumerate(design):
        if d is None or d["fraction"] != args.fraction:
            continue
        keep.append(i)

    hours_with_both = {}
    for i in keep:
        d = design[i]
        hours_with_both.setdefault(d["hour"], set()).add(d["treatment"])

    if not args.keep_incomplete:
        complete = {h for h, t in hours_with_both.items() if t == {"control", "heat"}}
        dropped = sorted(h for h, t in hours_with_both.items() if t != {"control", "heat"})
        if dropped:
            print(f"[prepare] dropping time points without both arms: {dropped}")
        keep = [i for i in keep if design[i]["hour"] in complete]

    keep.sort(key=lambda i: (design[i]["hour"], design[i]["treatment"], design[i]["replicate"]))
    if len(keep) < 6:
        print("[prepare] not enough samples selected; aborting", file=sys.stderr)
        return 2

    sub_counts = counts[:, keep]
    sub_cpm = (sub_counts / np.where(sub_counts.sum(axis=0, keepdims=True) > 0,
                                     sub_counts.sum(axis=0, keepdims=True), 1.0)) * 1e6

    n_keep_genes = max(1, int(np.ceil(args.min_frac * len(keep))))
    expressed = (sub_cpm >= args.min_cpm).sum(axis=1) >= n_keep_genes
    print(f"[prepare] expressed features: {int(expressed.sum())} / {len(genes)} "
          f"(CPM >= {args.min_cpm} in >= {n_keep_genes} of {len(keep)} samples)")

    genes_f = [g for g, m in zip(genes, expressed) if m]
    counts_f = sub_counts[expressed]
    X = cpm_log(counts_f)

    if args.top_var and args.top_var < X.shape[0]:
        var = X.var(axis=1)
        order = np.argsort(var)[::-1][:args.top_var]
        order = np.sort(order)
        X = X[order]
        genes_f = [genes_f[i] for i in order]
        print(f"[prepare] kept the {args.top_var} most variable features")

    matrix = X.T  # samples x features, which is what omictopo expects
    sample_names = [samples[i] for i in keep]
    labels = [design[i]["treatment"] for i in keep]
    hours = [design[i]["hour"] for i in keep]

    os.makedirs(args.outdir, exist_ok=True)
    matrix_path = os.path.join(args.outdir, "matrix.tsv")
    groups_path = os.path.join(args.outdir, "groups.tsv")
    design_path = os.path.join(args.outdir, "design.tsv")

    with open(matrix_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join([""] + genes_f) + "\n")
        for name, row in zip(sample_names, matrix):
            fh.write(name + "\t" + "\t".join(f"{v:.4f}" for v in row) + "\n")

    with open(groups_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("sample\tgroup\n")
        for name, label in zip(sample_names, labels):
            fh.write(f"{name}\t{label}\n")

    with open(design_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("sample\tgroup\thour\treplicate\n")
        for i in keep:
            fh.write(f"{samples[i]}\t{design[i]['treatment']}\t"
                     f"{design[i]['hour']}\t{design[i]['replicate']}\n")

    print()
    print(f"[prepare] dataset ready in {args.outdir}")
    print(f"  matrix : {matrix.shape[0]} samples x {matrix.shape[1]} features")
    print(f"  groups : " + ", ".join(
        f"{k}={labels.count(k)}" for k in sorted(set(labels))))
    print(f"  hours  : " + ", ".join(str(h) for h in sorted(set(hours))))
    for p in (matrix_path, groups_path, design_path):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
