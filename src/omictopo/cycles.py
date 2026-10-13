"""Representative cycles of H1 classes, and the cyclic order they induce.

Why this module exists
----------------------
A persistent-homology bar is a *number*: this loop lives between scale ``b``
and scale ``d``.  A number is easy to compare with a null model and hard to
do anything biological with.  But the reduction that produced the bar also
produced the **cycle itself** — a closed walk over sample indices — and that
is an object you can interpret:

* it names the samples that lie on the loop, in the order the loop visits
  them;
* therefore it induces a **cyclic order** on those samples, without using any
  label, timestamp or phase;
* that order is a *prediction* the geometry makes on its own, which can be
  scored against an external periodic variable that the statistic never saw.

The last point is the reason this module exists at all.  A circular Mantel
test, a cosinor fit, or any other phase-based method takes the phase as an
**input** — it can only tell you whether the phase you already have explains
the data.  The cycle below takes no input beyond the distance matrix, so the
order it produces can be checked against the clock rather than assumed from
it.  That is a different, and strictly weaker, set of assumptions.

How the cycle is obtained
-------------------------
The standard GF(2) reduction of the boundary matrix is run while also
tracking, for each column, which original columns were added into it.  When a
triangle ``j`` reduces to a column whose pivot is the edge ``i``, the pair
``(i, j)`` is a bar, and the reduced column itself is a 1-cycle: the sum of
triangle boundaries, with all interior edges cancelled in pairs and the
surviving edge set closed (every vertex has even degree).  Walking that edge
set gives the loop.

Caveat, stated plainly: an H1 class that is still alive at the truncation
threshold is *essential* and has no reduced column, hence no representative.
Use a threshold large enough for the loop to die (the default here builds the
full complex whenever the simplex budget allows).
"""

from __future__ import annotations

import numpy as np

from .filtration import build_rips
from .geometry import pairwise_distances

__all__ = [
    "reduce_boundary_columns",
    "cycles_from_edges",
    "dominant_h1_cycles",
    "h1_bar_lengths",
    "circular_succession_score",
]


# --------------------------------------------------------------------------
# reduction with column provenance
# --------------------------------------------------------------------------
def reduce_boundary_columns(boundary: list) -> tuple:
    """GF(2) boundary reduction that also returns the reduced columns.

    Parameters
    ----------
    boundary : list of int
        Boundary of simplex ``j`` as a bitmask over simplex indices.

    Returns
    -------
    pairs : list of (int, int)
        ``(birth_simplex, death_simplex)`` index pairs.
    essential : list of int
        Indices of simplices creating unpaired (essential) classes.
    reduced : list of int
        The reduced columns.  For a death simplex ``j`` these are the columns
        that remain after the reduction; in dimension 1 they are 1-cycles.

    Notes
    -----
    The reduction is identical to :func:`omictopo.homology.reduce_boundary`
    (and the two are cross-checked by the test suite); this variant simply
    keeps the intermediate columns instead of discarding them.
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

    essential = [
        j for j in range(len(reduced)) if reduced[j] == 0 and j not in pivot_to_col
    ]
    return pairs, essential, reduced


# --------------------------------------------------------------------------
# edge set -> closed walk
# --------------------------------------------------------------------------
def cycles_from_edges(edges, tie_break=None) -> tuple:
    """Decompose an Eulerian edge set into closed walks.

    The edge set produced by :func:`reduce_boundary_columns` is a 1-cycle, so
    every vertex has even degree and the edges decompose into closed walks.
    Usually there is exactly one; when a vertex has degree 4 the walk is not
    unique and the remaining edges are collected in further passes.

    Parameters
    ----------
    edges : iterable of (int, int)
        Undirected vertex pairs.
    tie_break : callable or None
        Key used to choose among several unvisited edges at a vertex.  The
        default is the smallest vertex index, which makes the result
        independent of set iteration order (and therefore reproducible across
        processes).

    Returns
    -------
    list of (list of int, bool)
        ``(walk, closed)`` per component; ``walk`` lists the vertices in
        visit order and does not repeat the starting vertex.
    """
    key = tie_break or (lambda v: v)
    adjacency: dict = {}
    for a, b in edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    unused = {(min(a, b), max(a, b)) for a, b in edges if a != b}
    walks: list = []
    while unused:
        start, nxt = min(unused)
        unused.discard((start, nxt))
        walk = [start]
        cur = nxt
        closed = False
        for _ in range(len(edges) + 2):
            if cur == start:
                closed = True
                break
            walk.append(cur)
            options = [w for w in adjacency.get(cur, [])
                       if (min(cur, w), max(cur, w)) in unused]
            if not options:
                break
            w = min(options, key=key)
            unused.discard((min(cur, w), max(cur, w)))
            cur = w
        if cur == start:
            closed = True
        walks.append((walk, closed))
    walks.sort(key=lambda t: (-len(t[0]), tuple(t[0])))
    return walks


# --------------------------------------------------------------------------
# dominant H1 cycles
# --------------------------------------------------------------------------
def dominant_h1_cycles(D=None, X=None, metric: str = "correlation",
                       rank: int | None = None, threshold: float | None = None,
                       max_simplices: int = 1_000_000) -> list:
    """Representative cycles of the longest H1 bars, longest bar first.

    Parameters
    ----------
    D : np.ndarray, optional
        Precomputed square distance matrix.  Takes precedence over ``X``.
    X : np.ndarray, optional
        Point cloud, shape ``(n_samples, n_features)``.
    metric : str
        Metric used when ``D`` is derived from ``X``.
    rank : int or None
        Return only the first ``rank`` cycles.  ``None`` returns all bars that
        die inside the complex.
    threshold : float or None
        Rips cut-off.  ``None`` lets the budget pick it, which for clouds up
        to a few dozen points means the full complex.
    max_simplices : int
        Simplex budget.

    Returns
    -------
    list of dict
        One entry per finite H1 bar, sorted by decreasing persistence, with
        keys ``birth``, ``death``, ``persistence``, ``n_edges``,
        ``n_vertices``, ``order`` (cyclic vertex order), ``closed``,
        ``mean_edge_length`` and ``max_edge_length``.
    """
    if D is None:
        if X is None:
            raise ValueError("provide either X or D")
        D = pairwise_distances(X, metric=metric)
    D = np.asarray(D, dtype=float)

    filtration = build_rips(D, max_dim=1, threshold=threshold,
                            max_simplices=max_simplices)
    pairs, essential, reduced = reduce_boundary_columns(filtration.boundary)
    values = filtration.values
    dims = filtration.dims
    vertices = filtration.vertices

    found: list = []
    for birth_simplex, death_simplex in pairs:
        if int(dims[birth_simplex]) != 1 or int(dims[death_simplex]) != 2:
            continue
        column = reduced[death_simplex]
        edges, lengths = [], []
        while column:
            k = column.bit_length() - 1
            column ^= (1 << k)
            if int(dims[k]) == 1:
                a, b = vertices[k]
                edges.append((int(a), int(b)))
                lengths.append(float(values[k]))
        walk, closed = cycles_from_edges(edges)[0] if edges else ([], False)
        found.append({
            "birth": round(float(values[birth_simplex]), 6),
            "death": round(float(values[death_simplex]), 6),
            "persistence": round(float(values[death_simplex]
                                      - values[birth_simplex]), 6),
            "n_edges": len(edges),
            "n_vertices": len(walk),
            "closed": bool(closed),
            "order": walk,
            "mean_edge_length": (round(float(np.mean(lengths)), 6)
                                 if lengths else 0.0),
            "max_edge_length": (round(float(np.max(lengths)), 6)
                                if lengths else 0.0),
        })

    found.sort(key=lambda r: (-r["persistence"], -r["n_edges"]))
    if rank is not None:
        return found[: int(rank)]
    return found


def h1_bar_lengths(D: np.ndarray, normalized: bool = True,
                   backend: str = "ripser") -> np.ndarray:
    """All H1 bar lengths of a distance matrix, longest first.

    Normalisation is the same RMS radius used by the benchmark,
    ``sqrt(mean(D^2) / 2)``, so bar lengths are dimensionless and comparable
    between datasets.  ``backend`` defaults to ``"ripser"`` to match
    ``examples/multi_dataset_benchmark.py``; it silently falls back to the
    built-in implementation when ripser is unavailable.
    """
    from .homology import persistent_homology

    D = np.asarray(D, dtype=float)
    scale = float(np.sqrt(np.mean(D ** 2) / 2.0))
    scale = scale if scale > 0 else 1.0
    diameter = float(D.max())
    diagram = persistent_homology(D=D, max_dim=1, backend=backend)
    lengths = np.sort(diagram.persistence(1, infinite_as=diameter))[::-1]
    return lengths / scale if normalized else lengths


# --------------------------------------------------------------------------
# scoring a recovered cyclic order against an external periodic variable
# --------------------------------------------------------------------------
def circular_succession_score(phases: np.ndarray, visit_order,
                              decimals: int = 6) -> float:
    """Fraction of cycle steps that follow the external clock.

    Parameters
    ----------
    phases : array-like, shape (n_samples,)
        The *external* phase (e.g. time of day, in radians) of every sample.
    visit_order : array-like of int
        The sample indices in the order the method visits them.  The order is
        treated as a **cycle**: the step from the last sample back to the
        first is scored too.
    decimals : int
        Rounding used to group samples that share a phase (replicate time
        points).

    Returns
    -------
    float
        Share of the ``len(visit_order)`` steps that either stay at the
        current phase (another replicate at the same time of day) or move to
        the phase immediately next in the cyclic schedule, taking the better
        of the two traversal directions.

        * ``1.0`` — every step stays or advances: the cycle walks the clock
          forwards.
        * ``~0.5`` — half the steps jump backwards or skip.
        * random order — a few percent to a few tens of percent, set by the
          number of distinct phases and by how many samples share each of
          them; quote the permutation null rather than a rule of thumb.

    Why this rather than a correlation
    ----------------------------------
    The statistic is invariant to the cycle's arbitrary starting vertex and
    direction, and to the phase origin, because it only ever compares
    *successors* (and it is maximised over the two directions).  It is also
    deliberately not a circular correlation or a rank correlation against an
    unwrapped angle: the recovered positions of a cycle are near-uniformly
    spread (an undefined circular mean), and an unwrapped zig-zag still drifts
    outwards because of the folding, so both of those reward orders that
    double back.  Counting successor hits does not.
    """
    ph = np.asarray(phases, dtype=float)
    order = np.asarray(visit_order, dtype=int)
    if order.size < 3:
        return 0.0
    walked = np.round(ph[order], decimals)
    levels = np.unique(walked)
    if levels.size < 2:
        return 0.0
    group = np.searchsorted(levels, walked)
    index = np.arange(group.size)
    successor = group[np.roll(index, -1)]
    next_level = np.roll(np.arange(levels.size), -1)
    prev_level = np.roll(np.arange(levels.size), 1)

    forward = np.count_nonzero((successor == group) | (successor == next_level[group]))
    backward = np.count_nonzero((successor == group) | (successor == prev_level[group]))
    return float(max(forward, backward)) / float(order.size)

