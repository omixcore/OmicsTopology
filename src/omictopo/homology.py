"""Persistent homology of a point cloud or distance matrix.

Algorithm
---------
* **H0** is computed exactly from the minimum spanning tree of the full
  distance matrix (equivalent to single-linkage clustering).  The finite bars
  are the sorted MST edge lengths, plus one essential class per connected
  component.
* **H1** and above come from the standard reduction of the boundary matrix
  over GF(2), with columns held as Python big-integer bitmasks so that the
  XOR column operations stay cheap for the complex sizes typical of omics
  state spaces.

Truncation
----------
Because the Rips complex is built up to a distance threshold ``T``, the
computed complex coincides with the full Rips complex at every filtration
value ``<= T``.  Bars whose death is ``<= T`` are therefore exact; classes
that would die later are reported as essential with ``death = inf`` and
``diagram.threshold == T``.
"""

from __future__ import annotations

import numpy as np

from .diagram import PersistenceDiagram
from .filtration import build_rips
from .geometry import pairwise_distances

__all__ = ["persistent_homology", "reduce_boundary", "h0_from_mst"]


def h0_from_mst(D: np.ndarray) -> np.ndarray:
    """H0 bars from the minimum spanning tree of a distance matrix.

    Returns
    -------
    np.ndarray, shape (n, 2)
        ``(birth, death)`` pairs; births are 0 and the last bar is essential
        (``death = inf``).
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return np.array([[0.0, np.inf]])

    mst = minimum_spanning_tree(D)
    deaths = np.sort(np.asarray(mst.data, dtype=float))
    bars = np.empty((deaths.size + 1, 2), dtype=float)
    bars[: deaths.size, 0] = 0.0
    bars[: deaths.size, 1] = deaths
    bars[-1] = (0.0, np.inf)
    return bars


def reduce_boundary(boundary: list) -> tuple:
    """Standard GF(2) boundary matrix reduction.

    Parameters
    ----------
    boundary : list of int
        Boundary of simplex ``j`` as a bitmask over simplex indices.

    Returns
    -------
    pairs : list of (int, int)
        ``(birth_simplex, death_simplex)`` index pairs.
    essential : list of int
        Indices of simplices that create unpaired (essential) classes: their
        boundary reduces to zero and they never appear as the pivot of
        another column.
    """
    reduced = list(boundary)
    pivot_to_col: dict = {}
    pairs: list = []

    for j, col in enumerate(reduced):
        while col:
            pivot = col.bit_length() - 1
            other = pivot_to_col.get(pivot)
            if other is None:
                break
            col ^= reduced[other]
        reduced[j] = col
        if col:
            pivot = col.bit_length() - 1
            pivot_to_col[pivot] = j
            pairs.append((pivot, j))

    # Creators that never die: zero boundary, and never acting as a pivot
    # (pivots are the negative/death simplices of their column).
    essential = [
        j for j in range(len(reduced)) if reduced[j] == 0 and j not in pivot_to_col
    ]
    return pairs, essential


def _persistent_homology_builtin(
    D: np.ndarray,
    max_dim: int,
    threshold: float | None,
    max_simplices: int,
    h0: str,
    verbose: bool,
) -> tuple:
    if h0 not in ("mst", "reduction"):
        raise ValueError("h0 must be 'mst' or 'reduction'")

    bars: list = []

    # H0 only, computed from the MST: no filtration needs to be built at all.
    if max_dim == 0 and h0 == "mst":
        for b, d in h0_from_mst(D):
            bars.append((float(b), float(d), 0.0))
        info = {
            "threshold": float("inf"),
            "truncated": False,
            "n_simplices": int(D.shape[0]),
            "count_by_dim": {0: int(D.shape[0])},
            "backend": "builtin",
            "h0_method": "mst",
        }
        return bars, info

    filtration = build_rips(
        D, max_dim=max_dim, threshold=threshold, max_simplices=max_simplices
    )

    if h0 == "mst":
        for b, d in h0_from_mst(D):
            bars.append((float(b), float(d), 0.0))

    pairs, essential = reduce_boundary(filtration.boundary)
    vals = filtration.values
    dims = filtration.dims

    for p, j in pairs:
        d = int(dims[p])
        if d > max_dim:
            continue
        if d == 0:
            if h0 == "reduction":
                bars.append((float(vals[p]), float(vals[j]), 0.0))
        else:
            bars.append((float(vals[p]), float(vals[j]), float(d)))

    for j in essential:
        d = int(dims[j])
        if d > max_dim:
            continue
        if d == 0:
            if h0 == "reduction":
                bars.append((float(vals[j]), float("inf"), 0.0))
        else:
            bars.append((float(vals[j]), float("inf"), float(d)))

    if verbose:
        print(
            f"[omictopo] threshold={filtration.threshold:.4g} "
            f"simplices={filtration.n_simplices} {filtration.count_by_dim()}"
        )

    info = {
        "threshold": filtration.threshold,
        "truncated": filtration.truncated,
        "n_simplices": filtration.n_simplices,
        "count_by_dim": filtration.count_by_dim(),
        "backend": "builtin",
        "h0_method": h0,
    }
    return bars, info


def persistent_homology(
    X: np.ndarray | None = None,
    D: np.ndarray | None = None,
    max_dim: int = 1,
    threshold: float | None = None,
    max_simplices: int = 80_000,
    h0: str = "mst",
    backend: str = "auto",
    verbose: bool = False,
) -> PersistenceDiagram:
    """Persistent homology of a point cloud or a distance matrix.

    Parameters
    ----------
    X : array-like, shape (n_points, n_features), optional
        Point cloud.  Distances are computed internally.
    D : array-like, shape (n_points, n_points), optional
        Precomputed distance matrix.  Takes precedence over ``X``.
    max_dim : int
        Highest homology dimension to report (0, 1 or 2).
    threshold : float or None
        Rips distance cut-off; ``None`` picks the largest value that fits
        ``max_simplices``.
    max_simplices : int
        Simplex budget for the automatic threshold and the hard cap.
    h0 : {"mst", "reduction"}
        ``"mst"`` (default) computes H0 exactly from the minimum spanning
        tree; ``"reduction"`` takes it from the boundary matrix instead,
        which is mainly useful for cross-validation of the implementation.
    backend : {"auto", "builtin", "ripser"}
        ``"auto"`` uses :mod:`ripser` when it is installed and falls back to
        the built-in implementation otherwise.
    verbose : bool
        Print the chosen threshold and complex size.

    Returns
    -------
    PersistenceDiagram
    """
    if D is None:
        if X is None:
            raise ValueError("provide either X or D")
        D = pairwise_distances(X)
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix")
    if backend not in ("auto", "builtin", "ripser"):
        raise ValueError(
            f"unknown backend {backend!r}; expected 'auto', 'builtin' or 'ripser'"
        )
    if max_dim not in (0, 1, 2):
        raise ValueError("max_dim must be 0, 1 or 2")

    n_points = int(D.shape[0])

    use_ripser = backend == "ripser" or (backend == "auto" and _ripser_available())
    if use_ripser and h0 == "mst" and backend == "auto":
        try:
            bars, info = _persistent_homology_ripser(D, max_dim)
        except Exception as exc:  # pragma: no cover - defensive fallback
            if verbose:
                print(f"[omictopo] ripser backend failed ({exc}), using builtin")
            bars, info = _persistent_homology_builtin(
                D, max_dim, threshold, max_simplices, h0, verbose
            )
    else:
        bars, info = _persistent_homology_builtin(
            D, max_dim, threshold, max_simplices, h0, verbose
        )

    meta = dict(info)
    meta["n_points"] = n_points
    return PersistenceDiagram(
        bars=np.asarray(bars, dtype=float).reshape(-1, 3),
        threshold=float(info.get("threshold", np.inf)),
        meta=meta,
    )


def _ripser_available() -> bool:
    try:
        import ripser  # noqa: F401

        return True
    except Exception:
        return False


def _persistent_homology_ripser(D: np.ndarray, max_dim: int) -> tuple:
    """Optional fast backend.  Requires ``pip install ripser``."""
    from ripser import ripser

    out = ripser(D, distance_matrix=True, maxdim=max_dim)
    bars = []
    for dim, dgm in enumerate(out["dgms"]):
        for birth, death in dgm:
            bars.append((float(birth), float(death), float(dim)))
    info = {
        "threshold": float("inf"),
        "truncated": False,
        "backend": "ripser",
        "h0_method": "ripser",
    }
    return bars, info
