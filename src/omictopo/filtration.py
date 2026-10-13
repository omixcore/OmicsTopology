"""Vietoris-Rips filtration with automatic size control.

The Vietoris-Rips complex of a point cloud at scale ``t`` contains every
simplex whose pairwise distances are all at most ``t``.  Sweeping ``t`` from
0 to infinity produces the filtration in which persistent homology is
computed.

Two practical guards are built in, because a naive Rips complex is
combinatorially explosive (O(n^3) triangles):

* ``choose_threshold`` picks the largest distance threshold whose simplex
  count still fits a budget, so the complex is as rich as possible without
  blowing up.
* ``build_rips`` refuses to allocate beyond ``max_simplices`` and tells the
  caller what to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Filtration", "build_rips", "choose_threshold", "count_simplices"]


def simplex_dim_for(max_dim: int, simplex_dim: int | None = None) -> int:
    """Simplex dimension needed so that classes up to ``max_dim`` can die.

    To read persistence in dimension ``k`` the filtration must also contain
    ``(k + 1)``-simplices, otherwise every ``k``-class looks essential.  The
    cap at 2 keeps a request for H2 from silently exploding into tetrahedra.
    """
    if simplex_dim is not None:
        return int(simplex_dim)
    return int(min(int(max_dim) + 1, 2))


def _count_triangles(adjacency: np.ndarray) -> float:
    """Number of triangles in a boolean adjacency matrix.

    The diagonal must be cleared first: leaving self-loops in makes every
    length-3 closed walk count, which inflates the triangle count and makes
    the automatic threshold needlessly conservative.
    """
    B = np.array(adjacency, dtype=np.float64, copy=True)
    np.fill_diagonal(B, 0.0)
    return float(np.sum(B * (B @ B))) / 6.0


def count_simplices(D: np.ndarray, threshold: float, simplex_dim: int = 1) -> int:
    """Number of vertices + edges (+ triangles) at a given threshold.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Symmetric distance matrix.
    threshold : float
        Filtration value.
    simplex_dim : int
        Highest simplex dimension to include.

    Returns
    -------
    int
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    B = D <= threshold
    n_edges = int(np.count_nonzero(np.triu(B, 1)))
    n_tri = int(round(_count_triangles(B))) if simplex_dim >= 2 else 0
    return n + n_edges + n_tri


def choose_threshold(
    D: np.ndarray,
    max_dim: int = 1,
    max_simplices: int = 80_000,
    max_candidates: int = 400,
    simplex_dim: int | None = None,
) -> float:
    """Largest distance threshold whose complex fits inside a simplex budget.

    The simplex count is monotone non-decreasing in the threshold, so the
    largest feasible value is found by binary search over the observed
    pairwise distances.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
    max_dim : int
        Highest homology dimension you intend to read; the budget accounts
        for the ``max_dim + 1`` simplices needed to kill those classes, so a
        request for H1 pays for triangles.
    max_simplices : int
        Budget for vertices + edges + (triangles).
    max_candidates : int
        Cap on the number of candidate thresholds probed, so that the search
        stays cheap for large clouds.
    simplex_dim : int or None
        Override the derived simplex dimension.

    Returns
    -------
    float
        The chosen threshold.  Use :func:`count_simplices` to inspect it.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n < 2:
        return 0.0
    simplex_dim = simplex_dim_for(max_dim, simplex_dim)

    iu, ju = np.triu_indices(n, k=1)
    vals = np.unique(D[iu, ju])
    if vals.size == 0:
        return 0.0
    if vals.size > max_candidates:
        picks = np.unique(np.linspace(0, vals.size - 1, max_candidates).astype(int))
        vals = vals[picks]

    def feasible(t: float) -> bool:
        return count_simplices(D, t, simplex_dim=simplex_dim) <= max_simplices

    if feasible(float(vals[0])):
        lo, hi = 0, vals.size - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(float(vals[mid])):
                lo = mid
            else:
                hi = mid - 1
        return float(vals[lo])

    # Even the smallest observed distance is over budget: return it and let
    # build_rips decide whether to raise or warn.
    return float(vals[0])


@dataclass
class Filtration:
    """An ordered simplex list with its boundary columns.

    Attributes
    ----------
    values : np.ndarray, shape (n_simplices,)
        Filtration value (birth scale) of each simplex.
    dims : np.ndarray, shape (n_simplices,)
        Simplex dimension.
    vertices : list of tuple
        Vertex indices of each simplex, in filtration order.
    boundary : list of int
        Boundary of simplex ``j`` as a bitmask over simplex indices; bit
        ``i`` set means simplex ``i`` is a face of simplex ``j``.  Bit masks
        keep the reduction in :mod:`omictopo.homology` fast and small.
    threshold : float
        Distance threshold used to build the complex.
    truncated : bool
        True when the complex stops before the full diameter, i.e. bars that
        would die later are reported as essential (alive at ``threshold``).
    """

    values: np.ndarray
    dims: np.ndarray
    vertices: list
    boundary: list
    threshold: float
    truncated: bool

    @property
    def n_simplices(self) -> int:
        return int(self.values.size)

    def count_by_dim(self) -> dict:
        """Number of simplices per dimension."""
        return {int(d): int(np.count_nonzero(self.dims == d)) for d in np.unique(self.dims)}


def build_rips(
    D: np.ndarray,
    max_dim: int = 1,
    threshold: float | None = None,
    max_simplices: int = 80_000,
    simplex_dim: int | None = None,
) -> Filtration:
    """Build a Vietoris-Rips filtration from a distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Symmetric distance matrix with zero diagonal.
    max_dim : int
        Highest homology dimension that will be *read*.  Simplices are built
        up to ``min(max_dim + 1, 2)`` so that classes in dimension ``max_dim``
        can actually die; a request for H1 therefore pays for triangles.
    threshold : float or None
        Distance cut-off.  ``None`` selects it automatically so the complex
        fits ``max_simplices``.
    max_simplices : int
        Hard budget; exceeding it raises :class:`ValueError`.
    simplex_dim : int or None
        Override the derived simplex dimension (use 1 to build a bare
        1-skeleton).

    Returns
    -------
    Filtration
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    if max_dim not in (0, 1, 2):
        raise ValueError("max_dim must be 0, 1 or 2")
    build_dim = simplex_dim_for(max_dim, simplex_dim)
    n = D.shape[0]

    if threshold is None:
        threshold = choose_threshold(
            D, max_dim=max_dim, max_simplices=max_simplices,
            simplex_dim=build_dim,
        )
    threshold = float(threshold)

    # Simplex keys must be unique *across dimensions*: a naive encoding such
    # as ``i * n + j`` for edges collides with the vertex keys 0..n-1.  Each
    # dimension therefore gets its own disjoint key range, which the boundary
    # construction below relies on.
    base_edge = n * n
    base_tri = 2 * n * n

    # --- 0-simplices -----------------------------------------------------
    keys = [np.arange(n, dtype=np.int64)]
    values = [np.zeros(n, dtype=float)]
    dims = [np.zeros(n, dtype=np.int8)]
    verts = [(int(i),) for i in range(n)]

    # --- 1-simplices -----------------------------------------------------
    iu, ju = np.triu_indices(n, k=1)
    edge_val = D[iu, ju]
    keep = edge_val <= threshold
    ei, ej, ew = iu[keep].astype(np.int64), ju[keep].astype(np.int64), edge_val[keep]

    keys.append(base_edge + ei * n + ej)
    values.append(ew)
    dims.append(np.ones(ei.size, dtype=np.int8))
    verts.extend((int(a), int(b)) for a, b in zip(ei, ej))

    # --- 2-simplices -----------------------------------------------------
    if build_dim >= 2:
        adj = D <= threshold
        ti, tj, tk, tw = [], [], [], []
        for a, b in zip(ei.tolist(), ej.tolist()):
            tail = adj[a, b + 1 :] & adj[b, b + 1 :]
            for off in np.nonzero(tail)[0]:
                c = b + 1 + int(off)
                tri = np.array(
                    [D[a, b], D[a, c], D[b, c]], dtype=float
                )
                ti.append(a)
                tj.append(b)
                tk.append(c)
                tw.append(float(tri.max()))
        if ti:
            ti = np.asarray(ti, dtype=np.int64)
            tj = np.asarray(tj, dtype=np.int64)
            tk = np.asarray(tk, dtype=np.int64)
            tw = np.asarray(tw, dtype=float)
            keys.append(base_tri + (ti * n + tj) * n + tk)
            values.append(tw)
            dims.append(np.full(ti.size, 2, dtype=np.int8))
            verts.extend(zip(ti.tolist(), tj.tolist(), tk.tolist()))

    keys = np.concatenate(keys)
    values = np.concatenate(values)
    dims = np.concatenate(dims)

    if keys.size > max_simplices:
        raise ValueError(
            f"complex has {keys.size} simplices, over the budget of {max_simplices}. "
            "Lower `threshold`, reduce `max_dim`, or subsample the point cloud "
            "with omictopo.subsample()."
        )

    # Order by (filtration value, dimension, vertex key).  A face always
    # precedes its cofaces because it has a smaller or equal value and, on
    # ties, a smaller dimension.
    order = np.lexsort((keys, dims, values))
    values = values[order]
    dims = dims[order]
    keys = keys[order]

    key_to_pos = {int(k): i for i, k in enumerate(keys.tolist())}
    ordered_verts = [verts[int(i)] for i in order.tolist()]
    boundary: list = [0] * values.size
    for j, (d, simplex) in enumerate(zip(dims.tolist(), ordered_verts)):
        if d == 0:
            continue
        if d == 1:
            # Boundary of an edge is its two vertices, whose keys are the
            # vertex indices themselves.
            a, b = simplex
            boundary[j] = (1 << key_to_pos[a]) | (1 << key_to_pos[b])
        else:
            # Boundary of a triangle is its three edges.
            a, b, c = simplex
            boundary[j] = (
                (1 << key_to_pos[base_edge + a * n + b])
                | (1 << key_to_pos[base_edge + a * n + c])
                | (1 << key_to_pos[base_edge + b * n + c])
            )

    truncated = bool(threshold < float(D.max()))
    return Filtration(
        values=values,
        dims=dims,
        vertices=ordered_verts,
        boundary=boundary,
        threshold=threshold,
        truncated=truncated,
    )
