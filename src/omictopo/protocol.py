"""A calibrated inference protocol for cyclic structure in omics state spaces.

Why a *protocol* rather than a *method*
---------------------------------------
The usual framing of a topology paper is "our statistic wins".  The measured
record of this one does not support that framing and is more interesting
without it.  Against twenty-four baselines on 280 clouds the topological count
is not the most sensitive detector; what it does that nothing else does is
**decline to answer** when the geometry does not support an answer, and its
declines are tied to the geometry of the cloud rather than to a threshold on a
score.  A count that can be wrong silently is worth less than a count that says
when it is unusable.

So the contribution is stated as a *decision procedure with an epistemic
status*:

1. **Declare the design first.**  The cyclic parameter(s) and the sampling
   layout are fixed before the data are looked at.  Nothing here is selected
   after seeing the answer.
2. **Count** the H1 bars that outrun a covariance-matched null
   (:func:`omictopo.cycles.count_cycles`).  This is label-free.
3. **Diagnose with two orthogonal quantities.**  ``gap`` — the longest bar over
   the second longest — says whether a loop *dominates* the spectrum.  A
   circular Mantel ``r`` against the design's own axis says whether that loop
   is the one the design sampled.  They disagree in both directions and both
   disagreements are informative.
4. **Classify** into one of four statuses (:data:`STATUSES`) and, when the
   status is not ``"counted"``, **return the refusal instead of the integer**.
5. **Declare the envelope** (:func:`envelope`): the conditions under which the
   count is trustworthy, measured from six real public designs rather than
   assumed.

When a design records *two* axes, the question "is the second one a coordinate
of its own?" is answered by :mod:`omictopo.axes`, which classifies the pair as
``independent_axes``, ``independent_axes_inert``, ``modulation`` or
``undetermined`` on pre-registered criteria.  The two modules are complementary:
``protocol`` counts loops, ``axes`` says whether an extra design axis adds a
coordinate.

The rule is deliberately conservative.  ``gap >= 2`` sits between the smallest
gap ever measured on a design that was counted right (3.56) and the largest on
one that was not (1.36).

What the protocol refuses to pretend
------------------------------------
* A dominant bar does **not** mean the integer is right.  A design with two
  independent cycles returned 1 at ``gap = 9.3`` because the second loop was
  too weak to resolve.  Quote the gap and the count together.
* ``r`` is an *effect size*, not a significance level, and on large clouds it
  under-flags real rings.  The yeast cell-cycle design has ``r = +0.16`` at
  ``p = 0.0005`` on 60 samples *and* a dominant bar; the rule returns
  ``"inconsistent"`` — correctly, in the sense that it asks for a design check,
  but the count itself is sound.  Read the status as advice to look, not as a
  ruling on the integer.
* The two-parameter case needs an experiment that does not yet exist publicly:
  a *grid*, not a longer time course.  See
  ``examples/two_cycle_search.py``; the cheapest grid measured is 64 samples at
  full signal.

Usage
-----
>>> from omictopo.protocol import analyse, summarise, envelope
>>> rec = analyse(X)                          # label-free: geometry only
>>> rec = analyse(X, axis=phase, period=2*np.pi)   # with the design axis
>>> print(summarise(rec))
"""
from __future__ import annotations

import numpy as np

from .cycles import count_cycles, cycle_identifiability
from .geometry import pairwise_distances, standardize
from .stats import circular_distance, mantel

__all__ = ["STATUSES", "analyse", "envelope", "summarise"]

#: The four epistemic statuses the protocol can return.
#:
#: ``"counted"``        the spectrum has a dominant bar and the design axis
#:                      explains the distances — the integer may be quoted.
#: ``"underpowered"``   a cycle is present but the selected genes do not let H1
#:                      resolve it; report the failure, not a count.
#: ``"inconsistent"``   a dominant loop exists but does not follow the design's
#:                      axis; check the design before reporting.
#: ``"no_cycle"``       no cycle in the axis and none in the spectrum; the cloud
#:                      is not a test of the count.
STATUSES = ("counted", "underpowered", "inconsistent", "no_cycle")

_VERDICT_TO_STATUS = {
    "identifiable": "counted",
    "underpowered": "underpowered",
    "inconsistent": "inconsistent",
    "no_ring": "no_cycle",
}

# ---------------------------------------------------------------------------
# the declared envelope -- measured, not assumed
# ---------------------------------------------------------------------------
_ENVELOPE = {
    "claim": (
        "beta_1 counts independent cyclic *parameters* of the sampled state "
        "space, not the axes a sample sheet records, and the protocol returns "
        "a status rather than a bare integer."
    ),
    "requires": [
        "the selected gene set is dominated by the cyclic signal, not merely "
        "contains it -- the count succeeds only when essentially all of the "
        "high-variance genes carry the loop (examples/cycle_counting_power.py)",
        "the design samples the whole period; a sub-period window is an open "
        "arc and cannot close a loop (examples/cycle_counting_real.py, the "
        "yearlong window sweep)",
        "for two independent parameters, two *independently sampled* axes: a "
        "grid, not a longer time course (examples/two_cycle_search.py)",
    ],
    "measured": {
        "real_designs": 6,
        "real_designs_counted_correctly": 2,
        "gap_on_designs_counted_right": [3.56, 4.93],
        "gap_on_designs_not_counted": [1.07, 1.09, 1.36, 1.56],
        "gap_threshold": 2.0,
        "two_parameter_grid_min_samples_at_full_signal": 64,
        "two_parameter_grid_grid": "8x8",
        "two_parameter_at_half_signal": (
            "117 samples returns 1, i.e. one loop is lost"),
        "two_cyclic_axes_measured": (
            "PRJNA1348461, 12 clock times x 4 oestrous phases: the ~24 h clock "
            "is real (circular Mantel r = +0.63) but the oestrous axis carries "
            "no independent loop (r = +0.006, p = 0.43), so the count is 1 and "
            "not 2 -- see examples/estrous_grid_cycles.py"),
        "two_frequencies_one_axis_measured": (
            "GSE220120, human blood, 72 samples (3 subjects x 2 days x 12 "
            "two-hourly draws) carrying a published ~24 h clock *and* a ~12 h "
            "ultradian program: the count is 0, and the rhythm screen shows the "
            "~12 h program is not carried by the selected high-variance genes "
            "(0 of the 200 strongest 12 h genes survive the selection, against "
            "53 of 200 for the clock) -- the envelope condition below, not a "
            "topological obstruction. See examples/ultradian_two_oscillator.py"),
        "axis_geometry_designs": 7,
        "axis_geometry_verdicts_agree": 7,
    },
    "cannot_do": [
        "recover a loop whose signal sits in a minority of the selected genes",
        "resolve two cycles from a single sampling axis, however many samples "
        "that axis carries -- the image is a closed geodesic, not a torus",
        "quote an integer when the H1 spectrum has no dominant bar",
    ],
    "not_label_free": [
        "the Mantel r and its p-value use the design axis; they are reported "
        "to keep the label-free count honest, not as part of the label-free "
        "claim",
    ],
}


def envelope() -> dict:
    """The declared applicability conditions of the protocol.

    Every number in ``measured`` comes from a committed benchmark record, so
    the boundary can be recomputed rather than trusted: ``verify.py`` checks
    the design counts against ``benchmark/cycle_counting_real.json`` and the
    grid requirement against ``benchmark/two_cycle_search.json``.
    """
    import copy

    return copy.deepcopy(_ENVELOPE)


def analyse(X: np.ndarray, axis=None, *, period: float | None = None,
            n_null: int = 120, n_perm: int = 2000, seed: int = 0,
            metric: str = "correlation", gap_threshold: float = 2.0,
            r_threshold: float = 0.5) -> dict:
    """Run the whole protocol on one cloud and return a status, not just a count.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The state cloud.  Replicate structure must already be removed by the
        caller (see ``centre_by_group`` in ``examples/cycle_counting_real.py``):
        every well-separated group contributes its own loop, so an uncentred
        cloud counts groups.
    axis : sequence of float, optional
        The design's own cyclic coordinate, one value per sample.  Supplying it
        turns on the second diagnostic; without it the protocol runs
        label-free and can still say ``"counted"`` or ``"underpowered"``, but
        it cannot tell a loop that follows the axis from one that does not.
    period : float, optional
        Period of ``axis``.  Defaults to :math:`2\\pi`, i.e. ``axis`` is an
        angle.  Pass the period in the axis's own units otherwise.
    n_null, n_perm, seed : int
        Null draws, permutations and the seed; all three are recorded.
    metric : str
        Distance metric, ``"correlation"`` to match the benchmark.
    gap_threshold, r_threshold : float
        Thresholds of the identifiability rule; defaults are the calibrated
        ones.

    Returns
    -------
    dict
        ``n_cycles`` (the raw count — *only* quote it when ``status ==
        "counted"``), ``status``, ``advice``, the two diagnostics ``gap`` and
        ``mantel_r`` (with ``mantel_p``), the raw spectrum ``top_bars_rel``
        and ``null_q95``, and the settings used.
    """
    rec = count_cycles(X, metric=metric, n_null=n_null, seed=seed)
    bars = rec["top_bars_rel"]
    # A perfectly clean loop can produce a *single* H1 bar, with no second bar to
    # divide by.  That is the most dominant spectrum there is, so the gap is
    # infinite rather than undefined; reporting it as 0 would turn the cleanest
    # possible ring into "underpowered".
    if len(bars) >= 2 and bars[1] > 1e-9:
        gap_used = float(bars[0] / bars[1])
        gap = round(gap_used, 4)
    elif bars and bars[0] > 0:
        gap_used = float("inf")
        gap = None
    else:
        gap_used = 0.0
        gap = None

    mantel_r = mantel_p = None
    if axis is not None:
        ax = np.asarray(axis, dtype=float)
        if ax.shape[0] != X.shape[0]:
            raise ValueError("axis must have one entry per sample")
        scale = 2.0 * np.pi if period is None else float(period)
        phase = 2.0 * np.pi * (np.mod(ax, scale) / scale)
        D = pairwise_distances(standardize(X), metric=metric)
        mantel_r, mantel_p = mantel(D, circular_distance(phase), n_perm=n_perm,
                                    seed=seed)

    verdict = cycle_identifiability(gap_used, mantel_r,
                                    gap_threshold=gap_threshold,
                                    r_threshold=r_threshold)
    status = _VERDICT_TO_STATUS[verdict["verdict"]]
    if status == "counted" and rec["n_bars_above_null_q95"] < 1:
        # a dominant bar that does not even clear the null is not a count
        status = "underpowered"

    return {
        "n_samples": int(X.shape[0]),
        "n_cycles": int(rec["n_bars_above_null_q95"]),
        "status": status,
        "advice": verdict["advice"],
        "gap": gap,
        "top_bars_rel": bars,
        "null_q95": rec["null_q95"],
        "mantel_r": None if mantel_r is None else round(float(mantel_r), 4),
        "mantel_p": None if mantel_p is None else round(float(mantel_p), 6),
        "evidence": "geometry + design axis" if axis is not None
                    else "geometry only (label-free)",
        "settings": {"metric": metric, "n_null": int(n_null),
                     "n_perm": int(n_perm), "seed": int(seed),
                     "gap_threshold": float(gap_threshold),
                     "r_threshold": float(r_threshold)},
    }


def summarise(rec: dict) -> str:
    """One line for a results table, in the protocol's voice."""
    status = rec["status"]
    n = rec["n_cycles"]
    gap = rec["gap"]
    r = rec.get("mantel_r")
    tail = f"gap {gap}" if gap is not None else "single H1 bar"
    if r is not None:
        tail += f", Mantel r {r:+.2f}"
    if status == "counted":
        return f"counted {n} cycle(s)  ({tail})"
    if status == "underpowered":
        return f"no count — underpowered  ({tail})"
    if status == "inconsistent":
        return f"no count — inconsistent with the design axis  ({tail})"
    return f"no count — no cycle  ({tail})"
