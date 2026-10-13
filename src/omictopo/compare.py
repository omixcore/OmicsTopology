"""Comparing persistence diagrams between biological states.

Two distances are implemented, both with the standard convention that a bar
may be matched to the diagonal, so diagrams of different sizes are
comparable:

* **Bottleneck distance** — the smallest ``delta`` such that a perfect
  matching exists with every matched pair at most ``delta`` apart.  It is the
  strictest "did any single feature move" measure.  Computed exactly by
  binary search over candidate values plus a maximum bipartite matching.
* **Wasserstein distance** — the p-norm of the optimal matching cost.  It is
  sensitive to many small changes, where bottleneck only sees the worst one.

Essential (infinite) bars carry no meaningful distance and are excluded;
their counts are reported separately so the loss is visible.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .diagram import PersistenceDiagram

__all__ = [
    "bottleneck_distance",
    "wasserstein_distance",
    "compare_diagrams",
    "select_bars",
]


def select_bars(diagram: PersistenceDiagram, dim: int | None = None) -> np.ndarray:
    """Finite bars as an ``(n, 2)`` array, optionally restricted to a dimension."""
    bars = diagram.finite(dim)
    return bars[:, :2] if bars.size else np.empty((0, 2), dtype=float)


def _ground_matrix(a: np.ndarray, b: np.ndarray, metric: str = "inf") -> np.ndarray:
    """Pairwise ground distance between two sets of diagram points."""
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=float)
    if metric == "inf":
        return np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    if metric == "l2":
        return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    raise ValueError("metric must be 'inf' or 'l2'")


def _diag_cost(bars: np.ndarray) -> np.ndarray:
    """Distance from each bar to the diagonal: ``(death - birth) / 2``."""
    if bars.size == 0:
        return np.empty(0, dtype=float)
    return 0.5 * (bars[:, 1] - bars[:, 0])


def _scaled(bars: np.ndarray, scale: float | None) -> np.ndarray:
    if scale is None or bars.size == 0:
        return bars
    return bars / float(scale)


def bottleneck_distance(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
    dim: int | None = None,
    scale: float | None = None,
) -> float:
    """Exact bottleneck distance between the finite parts of two diagrams.

    Parameters
    ----------
    dgm1, dgm2 : PersistenceDiagram
    dim : int or None
        Restrict to one homology dimension; ``None`` compares all dimensions
        together as one multiset.
    scale : float or None
        Divide both diagrams' coordinates by this value first.  Because
        birth/death coordinates carry the units of the embedding, and two
        states may sit at different overall scales, comparing normalised
        diagrams is usually the fairer test.  Use
        :func:`compare_diagrams` with ``normalize=True`` to do this
        automatically.

    Returns
    -------
    float
    """
    a = _scaled(select_bars(dgm1, dim), scale)
    b = _scaled(select_bars(dgm2, dim), scale)
    n1, n2 = a.shape[0], b.shape[0]
    n = n1 + n2
    if n == 0:
        return 0.0

    diag_a = _diag_cost(a)
    diag_b = _diag_cost(b)
    cross = _ground_matrix(a, b, metric="inf")

    candidates = [0.0]
    if cross.size:
        candidates.extend(cross.ravel().tolist())
    candidates.extend(diag_a.tolist())
    candidates.extend(diag_b.tolist())
    candidates = np.unique(np.asarray(candidates, dtype=float))

    def feasible(delta: float) -> bool:
        allowed = np.zeros((n, n), dtype=bool)
        # rows: P (n1) then placeholder diagonals for B (n2)
        # cols: placeholder diagonals for A (n1) then Q (n2)
        if n1:
            allowed[:n1, :n1] = diag_a[:, None] <= delta
        if n1 and n2:
            allowed[:n1, n1:] = cross <= delta
        if n2:
            allowed[n1:, :n1] = True
            allowed[n1:, n1:] = diag_b[None, :] <= delta
        graph = csr_matrix(allowed.astype(np.int8))
        perm = maximum_bipartite_matching(graph, perm_type="column")
        return int(np.count_nonzero(perm != -1)) == n

    if not feasible(float(candidates[-1])):
        return float("inf")

    lo, hi = 0, candidates.size - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(float(candidates[mid])):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def wasserstein_distance(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
    dim: int | None = None,
    p: float = 1.0,
    metric: str = "inf",
    scale: float | None = None,
) -> float:
    """Wasserstein distance between the finite parts of two diagrams.

    Bars are matched to each other or to the diagonal; the returned value is
    the ``p``-norm of the optimal matching cost.

    Parameters
    ----------
    dgm1, dgm2 : PersistenceDiagram
    dim : int or None
    p : float
    metric : {"inf", "l2"}
        Ground metric between diagram points.
    scale : float or None
        Divide both diagrams' coordinates by this value first.

    Returns
    -------
    float
    """
    a = _scaled(select_bars(dgm1, dim), scale)
    b = _scaled(select_bars(dgm2, dim), scale)
    n1, n2 = a.shape[0], b.shape[0]
    n = n1 + n2
    if n == 0:
        return 0.0

    diag_a = _diag_cost(a)
    diag_b = _diag_cost(b)
    C = np.zeros((n, n), dtype=float)
    if n1:
        C[:n1, :n1] = diag_a[:, None]
    if n2:
        C[n1:, n1:] = diag_b[None, :]
    if n1 and n2:
        C[:n1, n1:] = _ground_matrix(a, b, metric=metric)

    rows, cols = linear_sum_assignment(C)
    cost = C[rows, cols]
    return float(np.sum(cost ** p) ** (1.0 / p))


def compare_diagrams(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
    max_dim: int = 1,
    p: float = 1.0,
    metric: str = "inf",
    normalize: bool = True,
) -> dict:
    """Bottleneck and Wasserstein distances per dimension and overall.

    Parameters
    ----------
    dgm1, dgm2 : PersistenceDiagram
    max_dim : int
        Highest homology dimension to compare.
    p : float
        Wasserstein exponent.
    metric : {"inf", "l2"}
    normalize : bool
        When True (default) both diagrams are divided by the larger of the
        two truncation thresholds before distances are computed.  Birth and
        death coordinates carry the units of the embedding, so raw distances
        between two separately embedded states mix "how far apart the states
        sit" with "how different their shapes are"; normalising keeps only
        the shape.  The scale used is reported as ``normalization_scale``.

    Returns
    -------
    dict
        Keys ``h{d}_bottleneck``, ``h{d}_wasserstein``, ``all_bottleneck``,
        ``all_wasserstein``, ``normalization_scale``, plus
        ``h{d}_n_essential_1`` / ``_2`` so that classes excluded from the
        distance stay visible.
    """
    scale = None
    if normalize:
        candidates = [float(v) for v in (dgm1.threshold, dgm2.threshold)
                      if np.isfinite(v) and v > 0]
        scale = max(candidates) if candidates else None

    out: dict = {}
    for d in range(0, int(max_dim) + 1):
        out[f"h{d}_bottleneck"] = bottleneck_distance(dgm1, dgm2, dim=d, scale=scale)
        out[f"h{d}_wasserstein"] = wasserstein_distance(
            dgm1, dgm2, dim=d, p=p, metric=metric, scale=scale
        )
        out[f"h{d}_n_essential_1"] = int(dgm1.essential(d).shape[0])
        out[f"h{d}_n_essential_2"] = int(dgm2.essential(d).shape[0])
    out["all_bottleneck"] = bottleneck_distance(dgm1, dgm2, dim=None, scale=scale)
    out["all_wasserstein"] = wasserstein_distance(
        dgm1, dgm2, dim=None, p=p, metric=metric, scale=scale
    )
    out["normalization_scale"] = float(scale) if scale else 1.0
    return out
