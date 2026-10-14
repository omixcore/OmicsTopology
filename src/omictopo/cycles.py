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
from .geometry import pairwise_distances, standardize

__all__ = [
    "reduce_boundary_columns",
    "cycles_from_edges",
    "dominant_h1_cycles",
    "h1_bar_lengths",
    "circular_succession_score",
    "cycle_identifiability",
    "topology_guided_cycle",
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


# --------------------------------------------------------------------------
# is the count even identifiable?
# --------------------------------------------------------------------------
def cycle_identifiability(gap: float, mantel_r: float | None = None,
                          gap_threshold: float = 2.0,
                          r_threshold: float = 0.5) -> dict:
    """Decide whether a ``beta_1``-based cycle count is worth reporting.

    Counting cycles assumes the longest H1 bar is *dominant*: the count is the
    number of bars that outrun a null, and that number is only meaningful when
    one bar stands far above the rest of the spectrum.  When the spectrum
    decays smoothly there is no dominant bar, and any integer returned is an
    artefact of where the null happened to fall.

    Measured on four real designs (see ``examples/cycle_counting_real.py``),
    the working case had ``gap = 4.9`` and every failing case had
    ``gap = 1.06-1.36``.  ``gap`` alone, however, says nothing about *why* the
    spectrum is flat, and the reason matters for the reader — which is what
    ``mantel_r`` is for.  Two flat cases with very different diagnoses were
    observed: a ring that is genuinely present in the samples but invisible to
    H1 (``r = +0.70``), and no ring in the selected genes at all
    (``r = +0.045``).

    Parameters
    ----------
    gap : float
        Dominant-bar gap: longest H1 bar divided by the second longest.
    mantel_r : float or None
        Circular Mantel correlation between the expression distances and the
        design's own cyclic axis.  ``None`` skips the second half of the
        diagnosis.
    gap_threshold : float
        Below this the spectrum counts as flat.  ``2.0`` is deliberately
        conservative: it sits between the observed 1.36 and 4.93.
    r_threshold : float
        Mantel correlation taken as evidence that a ring is present in the
        samples whether or not H1 sees it.

    Returns
    -------
    dict
        ``report_count`` (bool), ``verdict`` (one of ``"identifiable"``,
        ``"underpowered"``, ``"no_ring"``, ``"inconsistent"``) and a
        one-sentence ``advice`` written for a reader of a results table.

        ``report_count`` means "the spectrum contains a dominant bar, so the
        counting criterion is applicable".  It is **not** a claim that the
        integer is right: the count is the number of bars that outrun a null,
        and a design with two independent cycles can still return one when the
        second loop is too weak to resolve (measured: ``gap = 9.3`` and a count
        of 1 against a design that contains 2).  Quote the gap and the count
        together, never the count alone.
    """
    gap = float(gap)
    dominant = gap >= float(gap_threshold)
    if mantel_r is None:
        return {
            "report_count": bool(dominant),
            "verdict": "identifiable" if dominant else "underpowered",
            "advice": ("a dominant cycle is present; the count is interpretable"
                       if dominant else
                       "the H1 spectrum is flat; do not report a count"),
        }

    r = float(mantel_r)
    ring = r >= float(r_threshold)
    if dominant and ring:
        verdict = "identifiable"
        advice = "a dominant cycle is present and the design axis explains the distances"
    elif dominant and not ring:
        verdict = "inconsistent"
        advice = ("a dominant loop exists but it does not follow the design's "
                  "cyclic axis; check the design before reporting")
    elif ring and not dominant:
        verdict = "underpowered"
        advice = ("the cycle is real (the design axis explains the distances) "
                  "but H1 cannot resolve it in this gene set; report the "
                  "failure, not a count")
    else:
        verdict = "no_ring"
        advice = ("no cycle in the design axis and none in the spectrum; this "
                  "cloud is not a test of the count")
    return {"report_count": bool(dominant and ring), "verdict": verdict,
            "advice": advice}


# --------------------------------------------------------------------------
# topology-guided feature selection
# --------------------------------------------------------------------------
def _harmonic_power(Z: np.ndarray, order) -> np.ndarray:
    """How much of each gene's variation follows one lap of a cyclic order.

    With ``m`` samples visited in the given order, sample ``k`` is assigned
    the angle ``2 pi k / m`` and every gene is projected onto the fundamental
    cosine and sine of that angle.  The squared projection is the share of the
    gene's variance explained by the cycle's first harmonic, so selecting the
    genes with the largest power keeps the ones whose expression actually
    sweeps once around the loop.
    """
    m = len(order)
    theta = 2.0 * np.pi * np.arange(m) / float(m)
    Y = np.asarray(Z, dtype=float)[np.asarray(order, dtype=int)]
    c = np.cos(theta) @ Y
    s = np.sin(theta) @ Y
    return c * c + s * s


def topology_guided_cycle(X: np.ndarray, n_keep: int | None = None,
                          n_iter: int = 2, rank: int = 1,
                          metric: str = "correlation",
                          max_simplices: int = 1_000_000) -> dict:
    """Find a cycle, keep the genes that follow it, and look again.

    The failure mode measured in ``examples/cycle_counting_real.py`` is not
    the design but the gene set: on real time courses the looping signal was
    carried by a minority of the highest-variance genes, and the remaining
    noise flattened the H1 spectrum until no bar was dominant.  If that
    diagnosis is right, then a **self-consistent** subset should recover the
    loop: use the cycle the data already suggest, keep the genes whose
    expression sweeps around it, rebuild the distance matrix and re-derive
    the cycle.

    No clock is used.  The cycle comes from the distance matrix and the
    selection criterion is a function of that cycle and the expression matrix
    alone, so the procedure can be validated afterwards against a held-out
    periodic variable — and, more importantly, it can be **calibrated**: apply
    exactly the same alternation to a covariance-matched Gaussian cloud and
    the selection bonus cancels out.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    n_keep : int or None
        Number of genes to keep.  ``None`` keeps half.
    n_iter : int
        Number of select-and-recount rounds.  ``2`` is the usual choice;
        more rounds risk locking onto noise.
    rank : int
        Which H1 bar's cycle to follow (1 = the longest bar).
    metric : str
        Distance metric, as elsewhere in the package.
    max_simplices : int
        Simplex budget for the filtration.

    Returns
    -------
    dict
        ``selected`` (column indices), ``order`` (sample indices in cycle
        order), ``history`` (one entry per round, with the bar gap and the
        number of genes kept), ``X_selected`` and ``D_selected``.

    Warning
    -------
    **Measured on four real designs, this procedure does not work, and it is
    kept here because the failure is informative rather than because it is
    usable** (``examples/cycle_identifiability.py``).

    * It produced no improvement over the plain spectrum: the recovered cycle
      scored against the held-out clock moved by ``-0.06`` to ``+0.08`` across
      the four designs, and turned no non-detection into a detection.
    * Its natural calibration — the longest bar against a covariance-matched
      Gaussian null with the whole procedure re-run inside the null — **fires
      on a cloud that has no ring at all**.  ``human_forced_desync`` (circular
      Mantel ``r = +0.10``) beat that null.  Any cloud with coherent sample
      structure, cyclic or not, gets sharpened into an apparent cycle by the
      selection step.

    So: do not use ``beats_gaussian_null`` as evidence for a cycle, and do not
    pair this with :func:`cycle_identifiability` as if it validated a count.
    The surviving decision rule is ``gap`` plus ``mantel_r`` from the plain
    pipeline.
    """
    Z = standardize(X)
    n_features = Z.shape[1]
    keep = int(n_keep) if n_keep is not None else max(10, n_features // 2)
    keep = min(keep, n_features)

    columns = np.arange(n_features)
    order = np.array([], dtype=int)
    history: list = []

    for it in range(max(1, int(n_iter))):
        D = pairwise_distances(Z[:, columns], metric=metric)
        cycles = dominant_h1_cycles(D=D, rank=int(rank),
                                    max_simplices=max_simplices)
        if not cycles or not cycles[0]["order"]:
            history.append({"round": it, "n_kept": int(columns.size),
                            "n_visited": 0, "gap": None})
            break
        order = np.asarray(cycles[0]["order"], dtype=int)
        bars = h1_bar_lengths(D)
        gap = float(bars[0] / bars[1]) if bars.size >= 2 and bars[1] > 1e-12 else None
        history.append({"round": it, "n_kept": int(columns.size),
                        "n_visited": int(order.size), "gap": gap,
                        "top_bar_rel": round(float(bars[0]), 4) if bars.size else 0.0})

        power = _harmonic_power(Z, order)
        columns = np.argsort(power)[::-1][:keep]
        columns = np.sort(columns)

    D_sel = pairwise_distances(Z[:, columns], metric=metric)
    bars = h1_bar_lengths(D_sel)
    gap = None
    if bars.size >= 2 and bars[1] > 1e-12:
        gap = float(bars[0] / bars[1])
    history.append({"round": int(max(1, int(n_iter))),
                    "n_kept": int(columns.size),
                    "n_visited": int(order.size), "gap": gap,
                    "top_bar_rel": round(float(bars[0]), 4) if bars.size else 0.0})
    return {
        "selected": columns.tolist(),
        "order": order.tolist(),
        "history": history,
        "X_selected": X[:, columns],
        "D_selected": D_sel,
        "gap_selected": gap,
        "top_bar_rel_selected": round(float(bars[0]), 4) if bars.size else 0.0,
    }

