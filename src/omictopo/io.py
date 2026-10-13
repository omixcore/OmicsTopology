"""Reading expression matrices, sample groups and writing result tables.

Deliberately pandas-free: numpy plus the standard library keep the package
installable anywhere an R-free, compiler-free analysis box exists.
"""

from __future__ import annotations

import csv
import os

import numpy as np

__all__ = ["read_matrix", "read_groups", "write_table", "write_matrix"]


def _sniff_delimiter(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tsv", ".tab", ".txt"):
        return "\t"
    if ext == ".csv":
        return ","
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","


def read_matrix(path: str, delimiter: str | None = None, has_header: bool = True,
                has_rownames: bool = True):
    """Read a numeric matrix from TSV/CSV.

    Rows are samples (or cells) and columns are features (genes, proteins,
    metabolites).

    Parameters
    ----------
    path : str
    delimiter : str or None
        Guessed from the file extension when omitted.
    has_header : bool
        First line holds column names.
    has_rownames : bool
        First column holds sample identifiers.

    Returns
    -------
    (np.ndarray, list, list)
        ``(matrix, rownames, colnames)``
    """
    delimiter = delimiter or _sniff_delimiter(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path} is empty")

    colnames: list = []
    if has_header:
        header = rows.pop(0)
        colnames = header[1:] if has_rownames else header

    rownames = []
    data = []
    for row in rows:
        if not row:
            continue
        if has_rownames:
            rownames.append(row[0])
            row = row[1:]
        data.append([float(v) if v not in ("", "NA", "NaN", "nan") else np.nan for v in row])

    X = np.asarray(data, dtype=float)
    if not colnames:
        colnames = [f"f{i + 1}" for i in range(X.shape[1])]
    if not rownames:
        rownames = [f"s{i + 1}" for i in range(X.shape[0])]
    return X, rownames, colnames


def read_groups(path: str, delimiter: str | None = None, column: int = 1):
    """Read per-sample group labels.

    Accepts either a single column of labels or a two-column
    ``sample<TAB>group`` table.

    Returns
    -------
    (list, list)
        ``(sample_ids, labels)``; ``sample_ids`` is empty for a single-column
        file.
    """
    delimiter = delimiter or _sniff_delimiter(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh, delimiter=delimiter) if r]

    if not rows:
        return [], []

    first = rows[0][0].strip().lower()
    if first in ("sample", "samples", "id", "sample_id", "sampleid"):
        rows = rows[1:]

    if all(len(r) == 1 for r in rows):
        return [], [r[0].strip() for r in rows]

    ids = [r[0].strip() for r in rows]
    labels = [r[column].strip() if len(r) > column else "" for r in rows]
    return ids, labels


def write_table(path: str, rows: list, delimiter: str = "\t") -> str:
    """Write a list of dicts to a delimited file.

    Returns
    -------
    str
        The path written.
    """
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return path

    keys: list = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(keys)
        for row in rows:
            writer.writerow([_fmt(row.get(k)) for k in keys])
    return path


def write_matrix(path: str, X: np.ndarray, rownames=None, colnames=None,
                 delimiter: str = "\t") -> str:
    """Write a numeric matrix with optional row and column names."""
    X = np.asarray(X)
    rownames = list(rownames) if rownames is not None else [f"s{i + 1}" for i in range(X.shape[0])]
    colnames = list(colnames) if colnames is not None else [f"f{i + 1}" for i in range(X.shape[1])]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow([""] + colnames)
        for name, row in zip(rownames, X):
            writer.writerow([name] + [_fmt(v) for v in row])
    return path


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)
