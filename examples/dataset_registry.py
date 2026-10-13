"""Registry of real, public expression time courses used by the benchmark.

The point of a *systematic* evaluation is that the topology is declared
**before** the data are looked at.  Each dataset below therefore carries an
``expected_topology`` field, written down from the experimental design rather
than from the result:

``loop``
    The samples were collected across one or more full circadian cycles while
    the clock kept running.  A functioning clock returns the transcriptome to
    (approximately) where it started, so the state space should trace a
    **closed loop** — a genuine H1 feature.  Note this is a statement about
    the *design*, not about the outcome: whether the loop survives in
    expression space is exactly what the benchmark measures.

``line``
    The samples follow a monotonically advancing process — fruit development,
    stress recovery.  There is no reason for the transcriptome to come back
    to its starting point, so the state space should trace an **open
    trajectory** without a persistent loop.

Every dataset here is a real GEO series with a downloadable processed
expression matrix, so the benchmark can be re-run end to end.  Raw data are
never redistributed with this repository.

Datasets
--------
====================  =========  ==========  ==========================  ==========
key                   accession  organism    design                      expected
====================  =========  ==========  ==========================  ==========
``ath_diurnal``       GSE158444  A. thaliana 8 times of day / 24 h       loop
``ath_recovery``      GSE158444  A. thaliana 4 recovery times            line
``rice_diurnal``      GSE138193  rice        17 times / 48 h, 3 lines    loop
``ath_diurnal_ll``    GSE137732  A. thaliana 16 times / 45 h, constant   loop
``ath_diurnal_ld``    GSE137732  A. thaliana 16 times / 45 h, long day   loop
``tomato_fruit_dev``  GSE128739  tomato      9 stages of fruit growth    line
====================  =========  ==========  ==========================  ==========
"""

from __future__ import annotations

import gzip
import importlib.util
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
USER_AGENT = "OmicsTopology-benchmark/0.1 (research; contact: see repository)"


# --------------------------------------------------------------------------
# specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Everything the benchmark needs to know about a dataset up front."""

    key: str
    accession: str
    organism: str
    common_name: str
    pmid: str
    citation: str
    design: str
    expected_topology: str          # "loop" or "line"
    period_hours: float | None      # circadian period, None when aperiodic
    units: str                      # CPM / FPKM / TPM
    filename: str
    loader: str
    notes: str = ""

    def url(self) -> str:
        n = self.accession[3:]
        stub = "GSE" + n[:-3] + "nnn"
        return f"{GEO_BASE}/{stub}/{self.accession}/suppl/{self.filename}"


SPECS: dict = {}


def _register(spec: DatasetSpec) -> DatasetSpec:
    SPECS[spec.key] = spec
    return spec


_register(DatasetSpec(
    key="ath_diurnal",
    accession="GSE158444",
    organism="Arabidopsis thaliana",
    common_name="thale cress",
    pmid="33871647",
    citation="Bonnot & Nagel, Plant Cell Environ 2021",
    design="8 times of day over one 24 h cycle (ZT48-ZT69), 22 C control",
    expected_topology="loop",
    period_hours=24.0,
    units="CPM",
    filename="GSE158444_Counts_Time.txt.gz",
    loader="genotype_matrix",  # resolved to the GSE158444 parser below
    notes="control arm of the heat-shock time course; 23 libraries",
))

_register(DatasetSpec(
    key="ath_diurnal_heat",
    accession="GSE158444",
    organism="Arabidopsis thaliana",
    common_name="thale cress",
    pmid="33871647",
    citation="Bonnot & Nagel, Plant Cell Environ 2021",
    design="8 times of day over one 24 h cycle, 1 h at 37 C heat shock",
    expected_topology="loop",
    period_hours=24.0,
    units="CPM",
    filename="GSE158444_Counts_Time.txt.gz",
    loader="genotype_matrix",
    notes="heat arm of the same time course; an independent replicate of the "
          "circadian-loop prediction within one experiment",
))

_register(DatasetSpec(
    key="ath_recovery",
    accession="GSE158444",
    organism="Arabidopsis thaliana",
    common_name="thale cress",
    pmid="33871647",
    citation="Bonnot & Nagel, Plant Cell Environ 2021",
    design="heat-shock recovery sampled at 4 times (T6, T7, T9, T12); 22 C control and 37 C arm",
    expected_topology="line",
    period_hours=None,
    units="CPM",
    filename="GSE158444_Counts_Recovery.txt.gz",
    loader="genotype_matrix",
    notes="same lab, platform and pipeline as ath_diurnal - a paired contrast",
))

_register(DatasetSpec(
    key="rice_diurnal",
    accession="GSE138193",
    organism="Oryza sativa",
    common_name="rice",
    pmid="33705889",
    citation="Li et al. 2019 (super-hybrid rice diurnal transcriptome)",
    design="17 times over two 24 h cycles, three genotypes (SE21s, LY2186, MH86)",
    expected_topology="loop",
    period_hours=24.0,
    units="FPKM",
    filename="GSE138193_AllSamples_FPKM.txt.gz",
    loader="rice_fpkm",
    notes="each genotype is a separate 17-point time course",
))

_register(DatasetSpec(
    key="ath_diurnal_ll",
    accession="GSE137732",
    organism="Arabidopsis thaliana",
    common_name="thale cress",
    pmid="32303634",
    citation="Oztas et al., J Biol Chem 2020",
    design="16 times over 45 h under constant light (CT2-CT47)",
    expected_topology="loop",
    period_hours=24.0,
    units="FPKM",
    filename="GSE137732_LL_CT2_47_transcript_expr.txt.gz",
    loader="ath_transcript_fpkm",
    notes="no biological replicates; circadian rhythm free-runs under constant light",
))

_register(DatasetSpec(
    key="ath_diurnal_ld",
    accession="GSE137732",
    organism="Arabidopsis thaliana",
    common_name="thale cress",
    pmid="32303634",
    citation="Oztas et al., J Biol Chem 2020",
    design="16 times over 45 h under long day 16 h light / 8 h dark (ZT2-ZT47)",
    expected_topology="loop",
    period_hours=24.0,
    units="FPKM",
    filename="GSE137732_LD_ZT2_47_transcript_expr.txt.gz",
    loader="ath_transcript_fpkm",
    notes="same study as ath_diurnal_ll, but the light/dark cycle entrains the clock",
))

_register(DatasetSpec(
    key="tomato_fruit_dev",
    accession="GSE128739",
    organism="Solanum lycopersicum",
    common_name="tomato",
    pmid="31724073",
    citation="Belouah et al., Physiol Plant 2019 (Data in Brief)",
    design="9 developmental stages of fruit pericarp (8-53 days post-anthesis), 3 replicates",
    expected_topology="line",
    period_hours=None,
    units="TPM",
    filename="GSE128739_devTom_tpm.xlsx",
    loader="tomato_xlsx",
    notes="monotonic developmental progression; the aperiodic contrast arm",
))


# --------------------------------------------------------------------------
# download helpers
# --------------------------------------------------------------------------
def download(spec: DatasetSpec, cache_dir: str) -> str:
    """Fetch the supplementary file for ``spec``, reusing a cached copy."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, spec.filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = spec.url()
    print(f"[registry] downloading {url}")
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1 << 16)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp, path)
    print(f"[registry] saved {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


_NA_TOKENS = {"", "NA", "N/A", "NaN", "nan", "None", "#N/A", ".", "null"}


def _num(value, default: float = 0.0) -> float:
    """Best-effort float conversion that tolerates the usual NA spellings."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in {t.lower() for t in _NA_TOKENS}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _load_prepare_module():
    """Import ``prepare_plant_dataset`` by path (the examples dir is not a package)."""
    path = os.path.join(_HERE, "prepare_plant_dataset.py")
    spec = importlib.util.spec_from_file_location("_omictopo_prepare_plant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# loaders: each returns (samples, features, matrix_genes_x_samples, design)
# --------------------------------------------------------------------------
def _load_ath_diurnal(path, spec, arm: str = "control"):
    """GSE158444 time course: 8 times of day, 3 replicates, one treatment arm."""
    prep = _load_prepare_module()
    genes, samples, counts = prep.read_counts(path)
    design_all = prep.parse_design(samples)

    keep = [i for i, d in enumerate(design_all)
            if d is not None and d["fraction"] == "TOT" and d["treatment"] == arm]
    keep.sort(key=lambda i: (design_all[i]["hour"], design_all[i]["replicate"]))
    if len(keep) < 6:
        raise SystemExit(f"ath_diurnal[{arm}]: fewer than 6 samples parsed")

    sub = counts[:, keep]
    lib = np.where(sub.sum(axis=0, keepdims=True) > 0, sub.sum(axis=0, keepdims=True), 1.0)
    cpm = sub / lib * 1e6

    design = [{"sample": samples[i],
               "condition": arm,
               "time": float(design_all[i]["hour"]),
               "replicate": int(design_all[i]["replicate"])}
              for i in keep]
    return [samples[i] for i in keep], list(genes), cpm, design


def _load_ath_recovery(path, spec):
    """GSE158444 recovery series: 4 times x {control, heat} x 3 replicates."""
    prep = _load_prepare_module()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        raw_samples = [s.strip() for s in header[1:]]
        genes, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            genes.append(parts[0].strip())
            rows.append([_num(v) for v in parts[1:]])
    counts = np.asarray(rows, dtype=float)

    pat = re.compile(r"^R_(TOT|TR)\.T(\d+)\.([CH])(\d)$")
    treat = {"C": "control", "H": "heat"}
    keep, design = [], []
    for i, name in enumerate(raw_samples):
        m = pat.match(name)
        if not m:
            continue
        fraction, t, arm, rep = m.groups()
        if fraction != "TOT":
            continue
        keep.append(i)
        design.append({"sample": name,
                       "condition": treat[arm],
                       "time": float(t),
                       "replicate": int(rep)})
    if len(keep) < 6:
        raise SystemExit("ath_recovery: fewer than 6 TOT samples parsed")

    order = sorted(range(len(keep)), key=lambda k: (design[k]["time"],
                                                    design[k]["condition"],
                                                    design[k]["replicate"]))
    keep = [keep[k] for k in order]
    design = [design[k] for k in order]

    sub = counts[:, keep]
    lib = np.where(sub.sum(axis=0, keepdims=True) > 0, sub.sum(axis=0, keepdims=True), 1.0)
    cpm = sub / lib * 1e6
    return [raw_samples[i] for i in keep], list(genes), cpm, design


def _load_rice_fpkm(path, spec):
    """GSE138193: one column per genotype x time point, FPKM."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = [c.strip() for c in header[1:]]
        genes, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            genes.append(parts[0].strip())
            rows.append([_num(v) for v in parts[1:]])
    mat = np.asarray(rows, dtype=float)

    pat = re.compile(r"^(SE|LY|MH)(\d+)_FPKM$")
    genotype_name = {"SE": "SE21s", "LY": "LY2186", "MH": "MH86"}
    samples, design, keep = [], [], []
    for i, col in enumerate(cols):
        m = pat.match(col)
        if not m:
            continue
        code, t = m.groups()
        keep.append(i)
        samples.append(col)
        design.append({"sample": col,
                       "condition": genotype_name[code],
                       "time": float(t),
                       "replicate": 0})
    order = sorted(range(len(keep)), key=lambda k: (design[k]["condition"], design[k]["time"]))
    keep = [keep[k] for k in order]
    samples = [samples[k] for k in order]
    design = [design[k] for k in order]
    return samples, list(genes), mat[:, keep], design


def _load_ath_transcript_fpkm(path, spec):
    """GSE137732: transcript-level FPKM, 16 columns CT2..CT47 / ZT2..ZT47."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = [c.strip() for c in header]
        fpkm_cols = [i for i, c in enumerate(cols) if c.startswith("FPKM.")]
        id_col = cols.index("t_name") if "t_name" in cols else 0
        labels = [cols[i].split(".", 1)[1] for i in fpkm_cols]

        features, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(fpkm_cols):
                continue
            features.append(parts[id_col].strip())
            rows.append([_num(parts[i]) for i in fpkm_cols])
    mat = np.asarray(rows, dtype=float)

    phase = "constant light (LL)" if labels[0].startswith("CT") else "long day (LD)"
    design = [{"sample": lab,
               "condition": phase,
               "time": float(re.sub(r"^[A-Z]+", "", lab)),
               "replicate": 0}
              for lab in labels]
    order = np.argsort([d["time"] for d in design])
    mat = mat[:, order]
    return [labels[i] for i in order], features, mat, [design[i] for i in order]


def _load_tomato_xlsx(path, spec):
    """GSE128739: TPM xlsx, columns ``F<days post anthesis>-<replicate>``."""
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "reading GSE128739 needs openpyxl:  pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    names = [str(v).strip() if v is not None else "" for v in header[1:]]

    features, values = [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        rid = row[0]
        if rid is None:
            continue
        rid = str(rid).strip()
        if not rid or rid.startswith("RNA_SPIKE"):
            continue
        features.append(rid)
        values.append([_num(v) for v in row[1:len(names) + 1]])
    wb.close()
    mat = np.asarray(values, dtype=float)

    pat = re.compile(r"^F(\d+)-(\d+)$")
    design, keep = [], []
    for i, name in enumerate(names):
        m = pat.match(name)
        if not m:
            continue
        dpa, rep = m.groups()
        keep.append(i)
        design.append({"sample": name,
                       "condition": "fruit",
                       "time": float(dpa),
                       "replicate": int(rep)})
    order = sorted(range(len(keep)), key=lambda k: (design[k]["time"], design[k]["replicate"]))
    keep = [keep[k] for k in order]
    return ([names[i] for i in keep], features, mat[:, keep],
            [design[k] for k in order])


_LOADERS = {
    "prepare_time": _load_ath_diurnal,
    "prepare_recovery": _load_ath_recovery,
    "rice_fpkm": _load_rice_fpkm,
    "ath_transcript_fpkm": _load_ath_transcript_fpkm,
    "tomato_xlsx": _load_tomato_xlsx,
}


# --------------------------------------------------------------------------
# normalisation + persistence to disk
# --------------------------------------------------------------------------
def finalize(samples, features, mat, design, outdir,
             min_expr: float = 1.0, min_frac: float = 0.25, top_var: int = 3000,
             log2: bool = True) -> dict:
    """Filter, log-transform and write ``matrix.tsv`` + ``design.tsv``.

    Parameters
    ----------
    mat : np.ndarray, shape (n_features, n_samples)
        Values on a linear scale (CPM / FPKM / TPM).
    """
    mat = np.asarray(mat, dtype=float)
    if mat.shape[1] != len(samples):
        raise ValueError(f"matrix has {mat.shape[1]} columns but {len(samples)} samples")

    need = max(1, int(np.ceil(min_frac * mat.shape[1])))
    expressed = (mat >= min_expr).sum(axis=1) >= need
    features = [f for f, m in zip(features, expressed) if m]
    mat = mat[expressed]

    X = np.log2(mat + 1.0) if log2 else mat
    if top_var and top_var < X.shape[0]:
        order = np.sort(np.argsort(X.var(axis=1))[::-1][:top_var])
        X = X[order]
        features = [features[i] for i in order]

    os.makedirs(outdir, exist_ok=True)
    matrix_path = os.path.join(outdir, "matrix.tsv")
    design_path = os.path.join(outdir, "design.tsv")

    with open(matrix_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join([""] + features) + "\n")
        for name, row in zip(samples, X.T):
            fh.write(name + "\t" + "\t".join(f"{v:.4f}" for v in row) + "\n")

    with open(design_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("sample\tcondition\ttime\treplicate\n")
        for name, d in zip(samples, design):
            fh.write(f"{d['sample']}\t{d['condition']}\t{d['time']}\t{d['replicate']}\n")

    return {
        "n_samples": len(samples),
        "n_features_expressed": int(expressed.sum()),
        "n_features_kept": int(X.shape[0]),
        "matrix_path": matrix_path,
        "design_path": design_path,
        "conditions": sorted({d["condition"] for d in design}),
        "times": sorted({d["time"] for d in design}),
    }


def prepare(key: str, outdir: str, cache: str | None = None,
            min_expr: float = 1.0, min_frac: float = 0.25, top_var: int = 3000) -> dict:
    """Download (if needed), parse, filter and write one dataset."""
    spec = SPECS[key]
    cache = cache or os.path.join(os.path.dirname(os.path.abspath(outdir)), "geo_cache")
    path = download(spec, cache)

    if key in ("ath_diurnal", "ath_diurnal_heat"):
        arm = "heat" if key.endswith("_heat") else "control"
        samples, features, mat, design = _load_ath_diurnal(path, spec, arm=arm)
    elif key == "ath_recovery":
        samples, features, mat, design = _load_ath_recovery(path, spec)
    else:
        loader = _LOADERS[spec.loader]
        samples, features, mat, design = loader(path, spec)

    meta = finalize(samples, features, mat, design, outdir,
                    min_expr=min_expr, min_frac=min_frac, top_var=top_var)
    meta.update({"key": key, "accession": spec.accession,
                 "organism": spec.organism, "expected_topology": spec.expected_topology,
                 "period_hours": spec.period_hours})
    return meta


def load_prepared(data_dir: str):
    """Read ``matrix.tsv`` + ``design.tsv`` written by :func:`prepare`."""
    matrix_path = os.path.join(data_dir, "matrix.tsv")
    design_path = os.path.join(data_dir, "design.tsv")
    if not (os.path.exists(matrix_path) and os.path.exists(design_path)):
        raise SystemExit(f"{data_dir} does not contain a prepared dataset")

    with open(matrix_path, encoding="utf-8") as fh:
        features = fh.readline().rstrip("\n").split("\t")[1:]
        samples, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            samples.append(parts[0])
            rows.append([float(v) for v in parts[1:]])
    X = np.asarray(rows, dtype=float)

    conditions, times, replicates = [], [], []
    with open(design_path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            _, cond, t, rep = line.rstrip("\n").split("\t")
            conditions.append(cond)
            times.append(float(t))
            replicates.append(int(rep))
    return (X, np.asarray(conditions), np.asarray(times, dtype=float),
            np.asarray(replicates, dtype=int), features)


if __name__ == "__main__":  # tiny self-check
    for k, s in SPECS.items():
        print(f"{k:20} {s.accession:10} {s.common_name:16} {s.expected_topology:5} "
              f"{s.design}")
