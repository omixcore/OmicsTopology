"""Prepare the three real datasets that put block C on measured data.

Block C of ``examples/topology_without_labels.py`` asks a modelling question --
*how many independent cyclic parameters does this state space have?* -- and
answered it with designs built to order.  That is the right place to start,
because only a simulation can guarantee that a grid is full and that the ground
truth is known.  But a claim about real biology has to survive real data, so
this module fetches and normalises the three public series on which the same
question can actually be asked.

``psme_annual`` -- GSE93067
    Douglas-fir needles.  5 trees x 16 collection dates spread over one full
    year (27-Oct-2010 .. 26-Oct-2011, roughly every three weeks).  Every
    sample was taken in the same early-morning window (ZT 05:00-07:00), so the
    design varies **one cyclic parameter only** -- but its period is a *year*.

``psme_diurnal`` -- GSE93066
    Douglas-fir needles from the same study.  6 trees x 6 times of day x 2
    consecutive days (48 h at 4-h resolution, 7-8 September).  The sample sheet
    has **two axis labels** -- hour of day and day of the experiment -- but the
    second day simply repeats the first day's times of day, so there is still
    exactly one cyclic parameter.

``human_forced_desync`` -- GSE48113
    Human blood, 22 subjects.  Each subject was sampled at 7 phases of their
    own circadian (melatonin) cycle on two protocol days: one day with the
    sleep-wake schedule *in phase* with the clock, and one after a 28-h
    forced-desynchrony protocol had shifted it *out of phase*.  Again two axis
    labels; the sleep-wake axis is sampled at two phases only.

Everything is written in the same layout the rest of the repository uses --
``matrix.tsv`` (samples x features, log2 CPM) plus ``design.tsv`` -- with one
extra file, ``coords.tsv``, carrying the design coordinates block C needs.

Usage::

    python examples/prepare_cycle_designs.py --outdir data/cycle_designs

Raw files are never redistributed; they are downloaded into ``--cache`` and
skipped if already present.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import urllib.request

import numpy as np

FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

# ---------------------------------------------------------------------------
# design registry -- written before the data are looked at
# ---------------------------------------------------------------------------
DESIGNS: dict[str, dict] = {
    "psme_annual": {
        "accession": "GSE93067",
        "organism": "Pseudotsuga menziesii",
        "title": "Douglas-fir needles, 5 trees x 16 dates over one year",
        "kind": "counts",
        "urls": {
            "counts": f"{FTP}/series/GSE93nnn/GSE93067/suppl/"
                      "GSE93067_psme_annual_DESeq_counts.txt.gz",
            "series": f"{FTP}/series/GSE93nnn/GSE93067/matrix/"
                      "GSE93067_series_matrix.txt.gz",
        },
        "period_hours": 365.25 * 24.0,
        "axes": ("annual_phase", "tree"),
        "n_cyclic_parameters": 1,
        "design_note": "one sampling window per date; the year is the cycle",
    },
    "psme_diurnal": {
        "accession": "GSE93066",
        "organism": "Pseudotsuga menziesii",
        "title": "Douglas-fir needles, 6 trees x 6 times of day x 2 days",
        "kind": "counts",
        "urls": {
            "counts": f"{FTP}/series/GSE93nnn/GSE93066/suppl/"
                      "GSE93066_psme_diurnal_DESeq_counts.txt.gz",
            "series": f"{FTP}/series/GSE93nnn/GSE93066/matrix/"
                      "GSE93066_series_matrix.txt.gz",
        },
        "period_hours": 24.0,
        "axes": ("daily_phase", "day"),
        "n_cyclic_parameters": 1,
        "design_note": "day 2 repeats the times of day of day 1",
    },
    "human_forced_desync": {
        "accession": "GSE48113",
        "organism": "Homo sapiens",
        "title": "Human blood, 22 subjects x 7 circadian phases x 2 sleep phases",
        "kind": "array",
        "urls": {
            "series": f"{FTP}/series/GSE48nnn/GSE48113/matrix/"
                      "GSE48113_series_matrix.txt.gz",
        },
        "period_hours": 24.0,
        "axes": ("circadian_phase", "sleep_timing"),
        "n_cyclic_parameters": 1,
        "design_note": "circadian cycle resolved; sleep-wake sampled at 2 phases",
    },
}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def fetch(url: str, cache: str, name: str | None = None) -> str:
    """Download ``url`` into ``cache`` unless it is already there."""
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, name or os.path.basename(url))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "omictopo/1.0"})
    with urllib.request.urlopen(req, timeout=900) as src, open(tmp, "wb") as dst:
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            dst.write(block)
    os.replace(tmp, dest)
    return dest


def _open(path: str):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") \
        else open(path, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------
def _read_table(path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Read a feature x sample text table; return (samples, features, samples x features)."""
    with _open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[1:]
        features, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(samples) + 1:
                continue
            features.append(parts[0])
            rows.append(parts[1:len(samples) + 1])
    mat = np.asarray(rows, dtype=float)          # features x samples
    return samples, features, mat.T              # samples x features


def _day_of_year(year: int, month: int, day: int) -> int:
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    doy = sum(_DAYS_IN_MONTH[:month - 1]) + day
    if leap and month > 2:
        doy += 1
    return doy


def _parse_annual_names(samples: list[str]) -> list[dict]:
    """``CCTO44_2010_27_oct`` -> tree, monotone day index, annual phase (radians)."""
    meta = []
    for s in samples:
        m = re.match(r"^(CCTO\d+)_(\d{4})_(\d{1,2})_([a-z]{3})$", s)
        if not m:
            raise ValueError(f"unexpected annual sample name: {s!r}")
        tree, year, day, mon = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        doy = _day_of_year(year, MONTHS[mon], day)
        meta.append({
            "sample": s, "group": tree,
            "axis1_name": "annual_phase",
            "axis1_value": 2.0 * np.pi * doy / 365.25,
            # monotone: never wraps, so windows can be taken in order
            "axis2_name": "day_index",
            "axis2_value": float(365 * (year - 2010) + doy),
        })
    return meta


def _parse_diurnal_names(samples: list[str]) -> list[dict]:
    """``1A04`` -> tree 1A, hour 04 (hours run 00..44 over the 48-h window)."""
    meta = []
    for s in samples:
        m = re.match(r"^(\d[A-B])(\d{2})$", s)
        if not m:
            raise ValueError(f"unexpected diurnal sample name: {s!r}")
        tree, hour = m.group(1), int(m.group(2))
        meta.append({
            "sample": s, "group": tree,
            "axis1_name": "daily_phase",
            "axis1_value": 2.0 * np.pi * (hour % 24) / 24.0,
            "axis2_name": "hour_index", "axis2_value": float(hour),
        })
    return meta


# ---------------------------------------------------------------------------
# the three parsers
# ---------------------------------------------------------------------------
def _prep_psme(key: str, counts_path: str, meta_txt: dict, outdir: str,
               min_expr: float, min_frac: float, top_var: int) -> dict:
    samples, features, X = _read_table(counts_path)
    cpm = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0) * 1e6
    logx = np.log2(cpm + 1.0)
    if key == "psme_annual":
        meta = _parse_annual_names(samples)
    else:
        meta = _parse_diurnal_names(samples)
    order = {m["sample"]: i for i, m in enumerate(meta)}
    assert set(order) == set(samples), "sample-name parsing lost samples"
    return _finalize(meta, features, logx, outdir, min_expr, min_frac, top_var)


def _clock_hours(text: str) -> float | None:
    """``"22:15"`` -> 22.25; anything unparseable (including ``"NA"``) -> None."""
    hh, _, mm = (text or "").partition(":")
    try:
        return (int(hh) % 24) + int(mm or 0) / 60.0
    except ValueError:
        return None


def _parse_fd_series_matrix(path: str):
    """Human Agilent series matrix -> (samples, features, samples x features, meta)."""
    chars: dict[str, list[str]] = {}
    acc: list[str] = []
    table: list[list[str]] = []
    in_table = False
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                acc = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                key = vals[0].split(":")[0].strip() if vals else ""
                chars.setdefault(key, []).extend(v.split(":", 1)[1].strip() for v in vals)
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table.append(line.rstrip("\n").split("\t"))

    if not table:
        raise ValueError(f"no data table in {path}")
    rows = table[1:]                       # first row is the header (ID_REF, GSMs)
    features = [r[0].strip('"') for r in rows]
    raw = np.full((len(rows), len(acc)), np.nan)
    for i, r in enumerate(rows):
        for j in range(min(len(acc), len(r) - 1)):
            v = r[j + 1].strip('"')
            if v:
                try:
                    raw[i, j] = float(v)
                except ValueError:
                    pass
    X = raw.T                              # samples x features

    subjects = chars.get("subject", [""] * len(acc))
    sleep = chars.get("sleep timing", [""] * len(acc))
    clock = chars.get("time sample taken", [""] * len(acc))
    phase = chars.get("time point", [""] * len(acc))
    day = chars.get("day sample was taken in protocol", [""] * len(acc))

    meta = []
    for i, s in enumerate(acc):
        meta.append({
            "sample": s,
            "group": subjects[i] if i < len(subjects) else "",
            "sleep_timing": sleep[i] if i < len(sleep) else "",
            "clock": clock[i] if i < len(clock) else "",
            "time_point": phase[i] if i < len(phase) else "",
            "protocol_day": day[i] if i < len(day) else "",
        })

    # circadian phase: each subject's 7 clock times span their own 24-h cycle.
    # Anchor on the earliest clock time of protocol day 1 for that subject.
    by_subject: dict[str, list[float]] = {}
    for m in meta:
        h = _clock_hours(m["clock"])
        if h is not None:
            by_subject.setdefault(m["group"], []).append(h)
    anchors = {k: min(v) for k, v in by_subject.items() if v}

    for m in meta:
        hour = _clock_hours(m["clock"])
        anchor = anchors.get(m["group"], 0.0)
        phase_h = float(np.mod((hour if hour is not None else anchor) - anchor, 24.0))
        m["circadian_phase"] = 2.0 * np.pi * phase_h / 24.0
        m["sleep_label"] = ("in" if "in phase" in m["sleep_timing"].lower()
                            else "out" if "out" in m["sleep_timing"].lower()
                            else "?")
        m["axis1_name"] = "circadian_phase"
        m["axis1_value"] = m["circadian_phase"]
        m["axis2_name"] = "sleep_phase"
        m["axis2_value"] = 0.0 if m["sleep_label"] == "in" else 1.0
    return acc, features, X, meta


def _prep_fd(path: str, outdir: str, min_expr: float, min_frac: float,
             top_var: int) -> dict:
    samples, features, X, meta = _parse_fd_series_matrix(path)
    # keep probes seen in most samples, then centre the missing ones
    seen = np.mean(~np.isnan(X), axis=0)
    keep = seen >= 0.9
    X = X[:, keep]
    features = [f for f, k in zip(features, keep) if k]
    col_mean = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])
    return _finalize(meta, features, X, outdir, min_expr, min_frac, top_var,
                     already_log=True)


# ---------------------------------------------------------------------------
# shared filtering / writing
# ---------------------------------------------------------------------------
def _finalize(meta, features, logx, outdir, min_expr, min_frac, top_var,
              already_log: bool = False):
    """Filter to expressed, then high-variance, features and write the bundle."""
    features = list(features)
    if already_log:
        keep = np.ones(len(features), dtype=bool)
    else:
        det = logx >= (min_expr if min_expr > 1.0 else np.log2(min_expr + 1.0))
        keep = det.mean(axis=0) >= min_frac
    logx = logx[:, keep]
    features = [f for f, k in zip(features, keep) if k]
    if logx.shape[1] > top_var:
        var = logx.var(axis=0)
        sel = np.argsort(var)[::-1][:top_var]
        sel.sort()
        logx = logx[:, sel]
        features = [features[i] for i in sel]

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "matrix.tsv"), "w", encoding="utf-8") as fh:
        fh.write("sample\t" + "\t".join(features) + "\n")
        for i, m in enumerate(meta):
            fh.write(m["sample"] + "\t"
                     + "\t".join(f"{v:.6g}" for v in logx[i]) + "\n")
    with open(os.path.join(outdir, "design.tsv"), "w", encoding="utf-8") as fh:
        fh.write("sample\tcondition\ttime\treplicate\n")
        for i, m in enumerate(meta):
            fh.write(f"{m['sample']}\t{m['group']}\t{m['axis1_value']:.6g}\t{i}\n")
    with open(os.path.join(outdir, "coords.tsv"), "w", encoding="utf-8") as fh:
        fh.write("sample\tgroup\taxis1_name\taxis1_value\taxis2_name\taxis2_value\textra\n")
        for m in meta:
            extra = (m.get("series") or m.get("sleep_label")
                     or m.get("protocol_day") or "")
            fh.write(f"{m['sample']}\t{m['group']}\t{m['axis1_name']}\t"
                     f"{m['axis1_value']:.6g}\t{m['axis2_name']}\t"
                     f"{m['axis2_value']:.6g}\t{extra}\n")
    return {
        "n_samples": len(meta),
        "n_features": len(features),
        "n_groups": len({m["group"] for m in meta}),
        "axis1_name": meta[0]["axis1_name"],
        "axis2_name": meta[0]["axis2_name"],
    }


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
def prepare(key: str, cache: str, outdir: str, min_expr: float = 1.0,
            min_frac: float = 0.25, top_var: int = 3000) -> dict:
    spec = DESIGNS[key]
    paths = {name: fetch(url, cache) for name, url in spec["urls"].items()}
    dest = os.path.join(outdir, key)
    if spec["kind"] == "counts":
        info = _prep_psme(key, paths["counts"], {}, dest, min_expr, min_frac, top_var)
    else:
        info = _prep_fd(paths["series"], dest, min_expr, min_frac, top_var)
    info.update({
        "key": key, "accession": spec["accession"], "organism": spec["organism"],
        "title": spec["title"], "period_hours": spec["period_hours"],
        "n_cyclic_parameters": spec["n_cyclic_parameters"],
        "design_note": spec["design_note"],
    })
    return info


POOLED_KEY = "psme_annual_x_diurnal"

# The annual series was sampled in an early-morning window, ZT 05:00-07:00;
# the diurnal series was a 48-h block on 7-8 September 2011.  Those two facts
# give every sample a coordinate on *both* axes, so the pooled set is a real
# two-parameter design rather than a concatenation.
ANNUAL_ZT_HOURS = 6.0
DIURNAL_DOY = _day_of_year(2011, 9, 7)


def prepare_pooled(cache: str, outdir: str, min_expr: float = 1.0,
                   min_frac: float = 0.25, top_var: int = 3000) -> dict:
    """Pool the annual and diurnal Douglas-fir series into one 2-parameter design.

    Both series come from the same study, the same trees' species, the same
    RNA-seq protocol and the same de-novo transcript reference, which is what
    makes pooling defensible at all.  Read counts are re-normalised *jointly*
    over the shared transcripts so no library-size step is done twice.
    """
    a_path = fetch(DESIGNS["psme_annual"]["urls"]["counts"], cache)
    d_path = fetch(DESIGNS["psme_diurnal"]["urls"]["counts"], cache)
    a_s, a_f, a_X = _read_table(a_path)
    d_s, d_f, d_X = _read_table(d_path)

    shared = sorted(set(a_f) & set(d_f))
    if len(shared) < 10_000:
        raise SystemExit(f"only {len(shared)} shared transcripts; refusing to pool")
    ai = {f: i for i, f in enumerate(a_f)}
    di = {f: i for i, f in enumerate(d_f)}
    a_cols = [ai[f] for f in shared]
    d_cols = [di[f] for f in shared]
    counts = np.vstack([a_X[:, a_cols], d_X[:, d_cols]])
    samples = list(a_s) + list(d_s)

    meta = _parse_annual_names(a_s)
    for m in meta:
        doy = np.rad2deg(m["axis1_value"]) * 365.25 / 360.0
        m["group"] = f"annual:{m['group']}"
        m["axis1_name"] = "annual_phase"
        m["axis1_value"] = 2.0 * np.pi * doy / 365.25
        m["axis2_name"] = "daily_phase"
        m["axis2_value"] = 2.0 * np.pi * ANNUAL_ZT_HOURS / 24.0
        m["series"] = "annual"

    for src in _parse_diurnal_names(d_s):
        meta.append({
            "sample": src["sample"],
            "group": f"diurnal:{src['group']}",
            "axis1_name": "annual_phase",
            "axis1_value": 2.0 * np.pi * DIURNAL_DOY / 365.25,
            "axis2_name": "daily_phase",
            "axis2_value": src["axis1_value"],
            "series": "diurnal",
        })
    assert len(meta) == len(samples)

    cpm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6
    logx = np.log2(cpm + 1.0)
    info = _finalize(meta, shared, logx, os.path.join(outdir, POOLED_KEY),
                     min_expr, min_frac, top_var)
    info.update({
        "key": POOLED_KEY, "accession": "GSE93067+GSE93066",
        "organism": DESIGNS["psme_annual"]["organism"],
        "title": "Douglas-fir: yearlong series + 48-h diurnal series, pooled",
        "period_hours": None,
        "n_cyclic_parameters": 2,
        "design_note": "two independently sampled cycles: circannual and daily",
    })
    return info


def load(key: str, outdir: str):
    """Return ``(X, coords, info)`` for a prepared design.

    ``X`` has shape (samples, features); ``coords`` is a list of dicts with the
    design coordinates in the same sample order.
    """
    d = os.path.join(outdir, key)
    mpath = os.path.join(d, "matrix.tsv")
    cpath = os.path.join(d, "coords.tsv")
    if not (os.path.exists(mpath) and os.path.exists(cpath)):
        raise SystemExit(f"{d} is not prepared -- run prepare_cycle_designs.py first")
    with open(mpath, encoding="utf-8") as fh:
        features = fh.readline().rstrip("\n").split("\t")[1:]
        rows = []
        for line in fh:
            rows.append([float(v) for v in line.rstrip("\n").split("\t")[1:]])
    X = np.asarray(rows, dtype=float)
    coords = []
    with open(cpath, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            vals = line.rstrip("\n").split("\t")
            coords.append(dict(zip(header, vals)))
    for c in coords:
        c["axis1_value"] = float(c["axis1_value"])
        c["axis2_value"] = float(c["axis2_value"])
    info = DESIGNS.get(key) or {
        "key": POOLED_KEY, "accession": "GSE93067+GSE93066",
        "organism": DESIGNS["psme_annual"]["organism"],
        "title": "Douglas-fir: yearlong series + 48-h diurnal series, pooled",
        "period_hours": None, "n_cyclic_parameters": 2,
        "design_note": "two independently sampled cycles: circannual and daily",
    }
    return X, coords, features, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="data/cycle_designs")
    ap.add_argument("--cache", default="data/geo_cache")
    ap.add_argument("--keys", nargs="*",
                    default=list(DESIGNS) + [POOLED_KEY])
    ap.add_argument("--top-var", type=int, default=3000)
    args = ap.parse_args(argv)

    for key in args.keys:
        if key == POOLED_KEY:
            info = prepare_pooled(args.cache, args.outdir, top_var=args.top_var)
        else:
            info = prepare(key, args.cache, args.outdir, top_var=args.top_var)
        print(f"{key:22} {info['accession']:16} n={info['n_samples']:4d} "
              f"features={info['n_features']:5d} groups={info['n_groups']:3d}  "
              f"axes={info['axis1_name']} x {info['axis2_name']}  "
              f"-> {os.path.join(args.outdir, key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
