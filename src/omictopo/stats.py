"""Permutation tests: is the topological change bigger than chance?

Two designs are provided, because they answer subtly different questions.

* :func:`permutation_test_pointclouds` — **one cloud per state**.  All points
  from both states are pooled and randomly re-split into groups of the same
  sizes.  This asks: "is the observed difference between the two state spaces
  larger than what a random partition of the same data produces?"  It assumes
  the points are exchangeable between states, which is a strong assumption;
  it is the right test when each state is a single snapshot.
* :func:`permutation_test_replicates` — **several replicates per state**, each
  with its own diagram.  Pairwise distances between diagrams are computed
  once and group labels are permuted, giving an energy-style statistic.  This
  is the more defensible test when you have biological replicates, because it
  respects the sampling unit.

Neither test is a substitute for a design with adequate replication.
"""

from __future__ import annotations

import numpy as np

from .compare import bottleneck_distance, wasserstein_distance
from .diagram import PersistenceDiagram
from .filtration import choose_threshold
from .geometry import pairwise_distances, subsample
from .homology import persistent_homology

__all__ = [
    "permutation_test_pointclouds",
    "permutation_test_replicates",
    "adjust_pvalues",
]


def adjust_pvalues(pvalues, method: str = "bh") -> np.ndarray:
    """Multiple-testing correction.

    Parameters
    ----------
    pvalues : sequence of float
    method : {"bh", "bonferroni"}
        Benjamini-Hochberg FDR or Bonferroni family-wise error.

    Returns
    -------
    np.ndarray
    """
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return p
    if method == "bonferroni":
        return np.clip(p * p.size, 0.0, 1.0)
    if method != "bh":
        raise ValueError("method must be 'bh' or 'bonferroni'")
    order = np.argsort(p)
    ranked = p[order] * p.size / (np.arange(p.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def _statistic(
    name: str,
    dgm_a: PersistenceDiagram,
    dgm_b: PersistenceDiagram,
    dim: int,
) -> float:
    # Bars that outlive the (shared) filtration are counted as dying at the
    # threshold.  Without this a ring whose loop never closes inside the
    # threshold would contribute nothing, and the test would lose its signal.
    cap_a = dgm_a.threshold if np.isfinite(dgm_a.threshold) else None
    cap_b = dgm_b.threshold if np.isfinite(dgm_b.threshold) else None

    if name == "h1_total":
        a = dgm_a.total_persistence(dim, infinite_as=cap_a)
        b = dgm_b.total_persistence(dim, infinite_as=cap_b)
        return abs(a - b)
    if name == "h1_count":
        return abs(float(dgm_a.n_bars(dim, finite_only=True) - dgm_b.n_bars(dim, finite_only=True)))
    if name == "h1_max":
        return abs(dgm_a.max_persistence(dim, infinite_as=cap_a)
                   - dgm_b.max_persistence(dim, infinite_as=cap_b))
    if name == "bottleneck":
        return bottleneck_distance(dgm_a, dgm_b, dim=dim)
    if name == "wasserstein":
        return wasserstein_distance(dgm_a, dgm_b, dim=dim)
    raise ValueError(
        f"unknown statistic {name!r}; expected one of "
        "'h1_total', 'h1_count', 'h1_max', 'bottleneck', 'wasserstein'"
    )


def _prepare_cloud(points: np.ndarray, max_points, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points, _ = subsample(points, max_points=max_points, seed=seed, return_index=True)
    return points


def permutation_test_pointclouds(
    points_a: np.ndarray,
    points_b: np.ndarray,
    statistic: str = "h1_total",
    dim: int = 1,
    n_permutations: int = 100,
    seed: int = 0,
    max_points: int | None = None,
    max_dim: int = 1,
    max_simplices: int = 80_000,
    threshold: float | None = None,
    backend: str = "builtin",
) -> dict:
    """Permutation test on two single-snapshot state clouds.

    Parameters
    ----------
    points_a, points_b : array-like, shape (n_points, n_features)
        Embedded state clouds.
    statistic : str
        One of ``'h1_total'``, ``'h1_count'``, ``'h1_max'``,
        ``'bottleneck'``, ``'wasserstein'``.
    dim : int
        Homology dimension the statistic refers to.
    n_permutations : int
        Number of label permutations.  100-1000 is usual; cost is
        proportional to this number, because homology is recomputed for every
        permutation.
    seed : int
    max_points : int or None
        Per-state subsample cap.
    max_dim, max_simplices : int
        Passed to :func:`omictopo.persistent_homology`.
    threshold : float or None
        A single Rips threshold shared by both states and every permutation,
        so the comparisons are on a common scale.  When ``None`` it is
        chosen once from the pooled cloud.
    backend : {"builtin", "auto", "ripser"}
        Homology backend.  Defaults to ``"builtin"`` on purpose: the ripser
        path ignores ``threshold``/``max_simplices``, so ``"auto"`` would make
        the p-value depend on whether ripser happens to be installed.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``n_permutations``, ``null`` (array of
        permuted statistics), ``threshold``, ``observed_a``, ``observed_b``.
    """
    rng = np.random.default_rng(seed)
    A = _prepare_cloud(points_a, max_points, seed)
    B = _prepare_cloud(points_b, max_points, seed + 1)
    n_a = A.shape[0]

    if threshold is None:
        pooled = np.vstack([A, B])
        D = pairwise_distances(pooled)
        threshold = choose_threshold(D, max_dim=max_dim, max_simplices=max_simplices)

    def diagrams(X: np.ndarray, Y: np.ndarray):
        da = persistent_homology(
            X, max_dim=max_dim, threshold=threshold, max_simplices=max_simplices,
            backend=backend,
        )
        db = persistent_homology(
            Y, max_dim=max_dim, threshold=threshold, max_simplices=max_simplices,
            backend=backend,
        )
        return da, db

    dgm_a, dgm_b = diagrams(A, B)
    observed = _statistic(statistic, dgm_a, dgm_b, dim)

    pooled = np.vstack([A, B])
    null = np.empty(int(n_permutations), dtype=float)
    for i in range(int(n_permutations)):
        idx = rng.permutation(pooled.shape[0])
        X = pooled[idx[:n_a]]
        Y = pooled[idx[n_a:]]
        da, db = diagrams(X, Y)
        null[i] = _statistic(statistic, da, db, dim)

    exceed = int(np.count_nonzero(null >= observed))
    p_value = (1.0 + exceed) / (1.0 + int(n_permutations))

    return {
        "statistic": statistic,
        "dim": int(dim),
        "observed": float(observed),
        "observed_a": dgm_a,
        "observed_b": dgm_b,
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "null": null,
        "threshold": float(threshold),
        "n_points_a": int(A.shape[0]),
        "n_points_b": int(B.shape[0]),
    }


def _diagram_distance_matrix(
    diagrams: list,
    dim: int | None = None,
    metric: str = "bottleneck",
) -> np.ndarray:
    n = len(diagrams)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "bottleneck":
                d = bottleneck_distance(diagrams[i], diagrams[j], dim=dim)
            elif metric == "wasserstein":
                d = wasserstein_distance(diagrams[i], diagrams[j], dim=dim)
            else:
                raise ValueError("metric must be 'bottleneck' or 'wasserstein'")
            if not np.isfinite(d):
                d = 0.0
            D[i, j] = D[j, i] = d
    return D


def _energy_statistic(D: np.ndarray, groups: np.ndarray) -> float:
    """Between-group mean distance minus within-group mean distance.

    A positive value means diagrams from different states sit further apart
    than diagrams within a state, which is exactly the signal we care about.
    """
    a = groups == 0
    b = ~a
    if a.sum() < 2 or b.sum() < 2:
        return 0.0
    within_a = D[np.ix_(a, a)].sum() / (a.sum() * (a.sum() - 1))
    within_b = D[np.ix_(b, b)].sum() / (b.sum() * (b.sum() - 1))
    cross = D[np.ix_(a, b)].mean()
    return float(2.0 * cross - within_a - within_b)


def permutation_test_replicates(
    diagrams_a: list,
    diagrams_b: list,
    n_permutations: int = 1000,
    seed: int = 0,
    dim: int | None = 1,
    metric: str = "bottleneck",
) -> dict:
    """Energy-style permutation test across replicate diagrams.

    Parameters
    ----------
    diagrams_a, diagrams_b : list of PersistenceDiagram
        One diagram per replicate.
    n_permutations : int
        Cheap here: pairwise diagram distances are computed once and only the
        labels are permuted.
    seed : int
    dim : int or None
        Homology dimension used for the distances; ``None`` uses all bars.
    metric : {"bottleneck", "wasserstein"}

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``n_permutations``, ``null``,
        ``distance_matrix``, ``within_a``, ``within_b``, ``cross``.
    """
    diagrams_a = list(diagrams_a)
    diagrams_b = list(diagrams_b)
    if len(diagrams_a) < 2 or len(diagrams_b) < 2:
        raise ValueError("need at least two replicate diagrams per state")

    diagrams = diagrams_a + diagrams_b
    groups = np.array([0] * len(diagrams_a) + [1] * len(diagrams_b))
    D = _diagram_distance_matrix(diagrams, dim=dim, metric=metric)
    observed = _energy_statistic(D, groups)

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_permutations), dtype=float)
    for i in range(int(n_permutations)):
        null[i] = _energy_statistic(D, rng.permutation(groups))

    exceed = int(np.count_nonzero(null >= observed))
    p_value = (1.0 + exceed) / (1.0 + int(n_permutations))

    a, b = groups == 0, groups == 1
    return {
        "statistic": "energy",
        "metric": metric,
        "dim": dim if dim is not None else "all",
        "observed": float(observed),
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "null": null,
        "distance_matrix": D,
        "within_a": float(D[np.ix_(a, a)].sum() / (a.sum() * (a.sum() - 1))),
        "within_b": float(D[np.ix_(b, b)].sum() / (b.sum() * (b.sum() - 1))),
        "cross": float(D[np.ix_(a, b)].mean()),
        "n_replicates_a": int(a.sum()),
        "n_replicates_b": int(b.sum()),
    }
