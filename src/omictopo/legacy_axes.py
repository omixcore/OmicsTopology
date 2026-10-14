"""Historical version-1 implementation retained for record reproduction only.

The following documentation is historical and contains claims corrected in v2;
do not use it to interpret new analyses.

Is a second design axis a new coordinate, or a modulation of the first?

The question
------------
Every method in the rhythmic-omics toolbox -- JTK_CYCLE, COSINOR, RAIN, the
harmonic fits that follow them -- is **per gene and one-dimensional in time**.
Each one answers "does *this gene* have a rhythm?" and the answer is a call plus
an amplitude and a phase.  Put two design axes in the sample sheet and the
toolbox still answers per gene: it fits a harmonic on each axis, counts rhythm
genes per axis, and compares amplitudes between conditions.  Nothing in that
pipeline represents the *joint* coordinate, so nothing in it can say whether the
second axis contributes a new direction to the state space or merely modulates
the first axis's own loop.

That distinction is what a topology is defined on, so it can be asked directly.
The answer is a small number of geometric quantities, and the whole point of
this module is that they are *not special to any one dataset*: the original
analysis lived in ``examples/axis_separability.py`` and was written for one
design (mouse clock x oestrous).  Everything here is the same arithmetic with
the design lifted out into arguments.

What is measured
----------------
:func:`axis_geometry` returns four blocks, in the order the questions arise:

``dominant``
    Which axis carries the geometry?  Circular Mantel ``r`` of the joint
    distance matrix against each axis, plus each axis's per-gene harmonic
    amplitude.  A Mantel ``r`` that is large on axis 1 and near zero on axis 2
    is the first sign that axis 2 may not be a coordinate.

``separable``
    Do the two axes explain *separate slabs* of the variance?  Write the
    centred matrix as ``L = C + O + R`` for the parts explained by each axis and
    the rest.  If the joint coordinate is a product of two independent axes,
    ``C`` and ``O`` are close to orthogonal: the variance shares add up
    (``additive_share_of_both`` near 1), the fitted subspaces are nearly
    orthogonal (``subspace_cosine`` near 0), and removing either axis leaves the
    other's distances intact.

``phase_or_shape``
    If axis 2 is not separable, *how* does it act on axis 1's loop?  Split each
    gene's axis-1 phase into the component explained by the axis-2 block and the
    residual.  A pure phase shift moves every gene's phase by the *same* offset
    -- one direction, ``block_offset_coherence`` near 1 -- while a change of
    programme scatters the offsets across genes.  The offset is also reported as
    a fraction of a quarter period, because a coherent but tiny offset is a real
    phase shift that is nonetheless too small to close a second loop.

``matched``
    The direct, model-free version of the same question: at a *fixed* value of
    axis 1, how far apart are samples that differ only in axis 2?  Compare that
    with the ordinary across-axis-1 distance.  A ratio near 1 means axis 2 is a
    coordinate of comparable size; a ratio well below 1 means it moves the point
    by less than a step of axis 1 does.  This one needs no harmonic model at
    all.  On a fully occupied grid (one sample per cell) the natural within-cell
    permutation is degenerate -- every permutation is the same labelling -- so
    the effect size is quoted with a *sample-level bootstrap* interval instead
    of a permutation p, and ``within_cell_permutation_degenerate`` says so
    explicitly.  That is the design telling you it has no replicates per cell,
    not a failure of the test.

The pre-registered criteria
---------------------------
The criteria are fixed here, before any dataset is scored, and
``reproduce/verify.py`` re-checks the numbers they are evaluated on.  Let

* ``r1``, ``r2`` be the Mantel correlations against axis 1 and axis 2;
* ``share`` be ``additive_share_of_both`` and ``cos`` the ``subspace_cosine``;
* ``coh`` be ``block_offset_coherence`` and ``shift_qp`` the largest block
  offset in quarter-periods;
* ``ratio`` be ``separation_ratio`` (matched distance over across-axis-1
  distance) and ``ci`` its bootstrap interval.

Then:

**``"independent_axes"``** -- ``r2 >= 0.5``.  Axis 2 carries a strong effect of
its own.  The separable block is reported alongside it and is expected to show
``share`` near 1 with ``cos`` near 0, but it is *not* allowed to overrule ``r2``:
see the negative result below.

**``"independent_axes_inert"``** -- ``r2 < 0.1`` and ``ratio < 0.2``.  An axis
that neither produces a Mantel effect nor moves points at a fixed axis-1 value:
it is orthogonal to axis 1 by construction, because it does nothing.

**``"modulation"``** -- ``r2 < 0.1``, ``coh >= 0.8``, ``shift_qp < 0.5`` and
``ratio >= 0.2``.  Axis 2 has no Mantel effect on its own, yet it demonstrably
moves points at a fixed axis-1 value, and it does so by one *common* phase
offset smaller than half a quarter-period.  This is the measured mouse-liver
case (``ratio`` 0.88), and the synthetic calibration of the same shape lands at
0.25.

**``"undetermined"``** -- anything else, including every ``r2`` in the band
``[0.1, 0.5)`` and any coherent shift at or above half a quarter-period.  A
coherent large shift is close to what a second loop would look like, so the
design has to be examined rather than classified.

A negative result about the criteria (measured, not assumed)
------------------------------------------------------------
The obvious discriminators do **not** work, and the module reports them anyway
because the failure constrains the claim:

* ``additive_share_of_both`` is **1.00 and ``subspace_cosine`` is 0.00 for an
  inert axis too** -- an axis that does nothing is trivially orthogonal to axis
  1.  The separable block therefore cannot distinguish "independent" from
  "inert"; it only rules out collinearity.
* ``separation_ratio`` is **lower for a genuinely independent second axis
  (~0.57) than for the modulation case (~0.88)**.  The ratio compares
  matched-across-axis-2 pairs against *all* across-axis-1 pairs, and on a grid
  the latter includes diagonal offsets, so it is not monotone in the size of
  axis 2.  It is used only as a coarse "resolved vs not" indicator, never as
  the discriminator.

What does discriminate, on both the measured design and the synthetic
calibration in ``examples/axis_geometry_multi.py``, is ``mantel_r_axis2``:
+0.70 for two independent axes against -0.04 for a modulated or inert one.  The
criteria above are therefore written so that ``r2`` carries the decision and the
geometric quantities describe the shape of the effect.

* **``mantel_r_axis2`` measures displacement, not cyclicity** -- the third
  negative result, and the one with the widest consequence.  The corpus in
  ``examples/axis_regularity.py`` (seven axis pairs, six independent
  experiments, five organisms) was scored against the natural hypothesis that a
  *condition*-shaped second axis never carries an independent effect, and the
  hypothesis is **falsified**: a 1 h 37 C heat treatment reaches
  ``r2 = +0.982`` and does so in both RNA fractions of the same experiment,
  while the oestrous cycle -- four sampled phases on a real 96 h cycle --
  reaches ``+0.006``.  The mechanism is that a coherent phase shift is an
  *isometry* of axis 1's own loop: it relabels the points without changing their
  distances, so it leaves no Mantel effect however large the shift, whereas a
  condition that relocates samples to a different region leaves a large one.
  ``independent_axes`` therefore means **"axis 2 carries a displaced copy of the
  cloud"**, which is *not* the same as "axis 2 is a second cycle".  Read the
  verdict with that in mind, and read :mod:`omictopo.protocol` for the cyclic
  question: the two modules answer two different questions and this is the
  boundary between them.

Reference: on the one measured grid where the question has a published answer
(PRJNA1348461, mouse liver clock x oestrous) this returns ``"modulation"`` with
``r2 = +0.006``, ``coh = 0.92``, ``shift_qp = 0.37``, ``ratio = 0.88``.  The
corpus that puts that number in context is in
``examples/axis_regularity.py``; its falsified hypothesis and its surviving
ordering are recorded there rather than here.
"""
from __future__ import annotations

import numpy as np

from .geometry import pairwise_distances, standardize
from .stats import circular_distance, legacy_mantel as mantel

__all__ = [
    "AXIS_VERDICTS",
    "axis_geometry",
    "classify_axis",
    "harmonic_amplitude",
    "summarise_axis",
]

#: The four verdicts :func:`classify_axis` can return.
AXIS_VERDICTS = ("independent_axes", "independent_axes_inert", "modulation",
                 "undetermined")

#: Pre-registered thresholds, calibrated on the synthetic grid in
#: ``examples/axis_geometry_multi.py`` *and* then applied unchanged to the real
#: designs.  The calibration values (8x8 grid, 300 genes, noise 0.3) are:
#:
#: ============ ========= ======= ======= =====
#: case         r2        ratio   coh     shift
#: ============ ========= ======= ======= =====
#: independent  +0.665    0.557   0.999   0.035
#: modulated    -0.038    0.249   0.999   0.040
#: inert        -0.038    0.159   0.999   0.046
#: ============ ========= ======= ======= =====
#:
#: ``axis_effect`` (``r_strong`` / ``r_inert``) carries the first split: does
#: axis 2 produce a Mantel effect at all.  ``ratio_modulated`` carries the
#: second: among axes with no Mantel effect of their own, one that still moves
#: points *and* has a resolved bootstrap interval is a modulation, while one
#: that barely moves them is inert.  ``share_separable`` / ``cos_separable`` are
#: reported but deliberately **not** used to overrule either split: an inert
#: axis is trivially orthogonal to axis 1 (measured share 1.00, cosine 0.00) and
#: would otherwise be labelled independent.  See the module docstring.
CRITERIA = {
    "r_strong": 0.5,          # Mantel r for "axis 2 is an effect of its own"
    "r_inert": 0.1,           # Mantel r below which axis 2 does nothing
    "ratio_modulated": 0.2,   # matched/across ratio separating modulation
                              # (0.249) from inert (0.159) on the calibration
    "coherence_common": 0.8,  # coherence of the per-block phase offsets
    "shift_quarter_periods": 0.5,   # shift size at which a modulation is suspect
    # reported only, never used to overrule the two splits above
    "share_separable": 0.8,
    "cos_separable": 0.5,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _phase(axis, period: float | None) -> np.ndarray:
    """Design coordinate -> angle in [0, 2*pi)."""
    a = np.asarray(axis, dtype=float)
    scale = 2.0 * np.pi if period is None else float(period)
    return 2.0 * np.pi * (np.mod(a, scale) / scale)


def harmonic_amplitude(y: np.ndarray, theta: np.ndarray) -> tuple[float, float]:
    """Least-squares amplitude and phase of a fixed-period harmonic.

    ``theta`` is the design coordinate *already in radians*.  Returns
    ``(amplitude, phase)``; this is the per-gene quantity a COSINOR fit reports,
    reproduced here so the geometric answer can be compared against it.
    """
    y = np.asarray(y, dtype=float)
    th = np.asarray(theta, dtype=float)
    A = np.c_[np.cos(th), np.sin(th)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(coef[0], coef[1])), float(np.arctan2(coef[1], coef[0]))


def _block_design(labels, n_levels: int | None = None) -> tuple[np.ndarray, list]:
    """Indicator design for a discrete axis (one column per level)."""
    levels = sorted(set(np.asarray(labels).tolist()))
    if n_levels is not None and len(levels) != n_levels:
        raise ValueError(f"expected {n_levels} levels, found {len(levels)}")
    A = np.zeros((len(labels), len(levels)))
    index = {lv: i for i, lv in enumerate(levels)}
    for i, lv in enumerate(labels):
        A[i, index[lv]] = 1.0
    return A, levels


def _residual_share(L: np.ndarray, V: np.ndarray) -> float:
    """Fraction of total variance *not* explained by the design ``V``."""
    P = V @ np.linalg.pinv(V.T @ V) @ V.T
    fitted = P @ L
    denom = float((L ** 2).sum())
    return float(1.0 - (fitted ** 2).sum() / denom) if denom > 0 else 1.0


def _project(L: np.ndarray, V: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(V, L, rcond=None)
    return V @ coef


# ---------------------------------------------------------------------------
# the four blocks
# ---------------------------------------------------------------------------
def _dominant(X, theta1, theta2, period1, period2, groups, seed, n_perm) -> dict:
    Z = standardize(X)
    D = pairwise_distances(Z, metric="correlation")
    r1, p1 = mantel(D, circular_distance(theta1, period1), n_perm=n_perm, seed=seed)
    r2, p2 = mantel(D, circular_distance(theta2, period2), n_perm=n_perm, seed=seed)

    n_genes = X.shape[1]
    amp1 = np.empty(n_genes)
    amp2 = np.empty(n_genes)
    for j in range(n_genes):
        amp1[j], _ = harmonic_amplitude(X[:, j], theta1)
        amp2[j], _ = harmonic_amplitude(X[:, j], theta2)

    return {
        "n_genes": int(n_genes),
        "mantel_r_axis1": round(float(r1), 4),
        "mantel_p_axis1": round(float(p1), 6),
        "mantel_r_axis2": round(float(r2), 4),
        "mantel_p_axis2": round(float(p2), 6),
        "mean_amplitude_axis1": round(float(amp1.mean()), 5),
        "mean_amplitude_axis2": round(float(amp2.mean()), 5),
        "amplitude_ratio_axis1_over_axis2": round(
            float(amp1.mean() / amp2.mean()), 4) if amp2.mean() > 0 else None,
        "reading": (
            "circular Mantel r of the joint distance matrix against each "
            "design axis, alongside each axis's mean per-gene harmonic "
            "amplitude. Large on axis 1 and near zero on axis 2 is the first "
            "sign that axis 2 carries no geometry of its own"),
    }


def _separable(X, theta1, theta2, groups, seed, n_perm) -> dict:
    L = X - X.mean(axis=0)
    n = L.shape[0]
    A1 = np.c_[np.ones(n), np.cos(theta1), np.sin(theta1)]
    A2, levels = _block_design(groups)
    both = np.c_[A1, A2[:, 1:]]

    v1 = 1.0 - _residual_share(L, A1)
    v2 = 1.0 - _residual_share(L, A2)
    vb = 1.0 - _residual_share(L, both)

    D0 = pairwise_distances(standardize(X), metric="correlation")
    X_no1 = X - _project(L, A1)
    X_no2 = X - _project(L, A2)
    D_no1 = pairwise_distances(standardize(X_no1), metric="correlation")
    D_no2 = pairwise_distances(standardize(X_no2), metric="correlation")
    r_no1, _ = mantel(D0, D_no1, n_perm=n_perm, seed=seed)
    r_no2, _ = mantel(D0, D_no2, n_perm=n_perm, seed=seed)

    P1 = A1 @ np.linalg.pinv(A1.T @ A1) @ A1.T
    P2 = A2 @ np.linalg.pinv(A2.T @ A2) @ A2.T
    overlap = float(np.trace(P1 @ P2) ** 2 / (np.trace(P1) * np.trace(P2)))
    C1, C2 = P1 @ L, P2 @ L
    denom = float(np.linalg.norm(C1) * np.linalg.norm(C2))
    cos = float(np.sum(C1 * C2) / denom) if denom > 0 else 0.0

    return {
        "variance_explained_by_axis1": round(float(v1), 4),
        "variance_explained_by_axis2": round(float(v2), 4),
        "variance_explained_by_both": round(float(vb), 4),
        "additive_share_of_both": round(float((v1 + v2) / vb), 4) if vb > 0 else None,
        "subspace_overlap": round(overlap, 4),
        "subspace_cosine": round(cos, 4),
        "distances_after_removing_axis1": round(float(r_no1), 4),
        "distances_after_removing_axis2": round(float(r_no2), 4),
        "levels_axis2": [str(lv) for lv in levels],
        "reading": (
            "additive_share_of_both near 1 with a small subspace_cosine means "
            "the two axes explain separate slabs of the same geometry; a large "
            "overlap means one axis is a modulation of the other"),
    }


def _phase_or_shape(X, theta1, groups, seed) -> dict:
    """Split each gene's axis-1 phase into block-explained and residual parts."""
    L = X - X.mean(axis=0)
    labels = np.asarray(groups)
    levels = sorted(set(labels.tolist()))
    n_genes = X.shape[1]

    phase = np.zeros((len(levels), n_genes))
    for b, lv in enumerate(levels):
        m = labels == lv
        for j in range(n_genes):
            _, phase[b, j] = harmonic_amplitude(X[m, j], theta1[m])

    mu = phase.mean(axis=0, keepdims=True)
    dev = phase - mu
    total_var = float((dev ** 2).mean())
    if total_var <= 0:
        return {"n_genes": int(n_genes), "between_block_phase_variance": 0.0,
                "common_shift_share": None, "gene_specific_share": None,
                "phase_deviation_effective_rank": None,
                "block_offset_coherence": None, "max_block_offset_rad": None,
                "max_block_offset_in_quarter_periods": None,
                "block_offsets_rad": [0.0] * len(levels),
                "reading": "the per-gene axis-1 phase does not vary between "
                           "blocks, so there is no phase shift to size"}

    offset = dev.mean(axis=1, keepdims=True)
    common = np.repeat(offset, n_genes, axis=1)
    common_var = float((common ** 2).mean())
    resid_var = float(((dev - common) ** 2).mean())

    s = np.linalg.svd(dev, compute_uv=False)
    s = s / (s.sum() if s.sum() > 0 else 1.0)
    eff_rank = float(1.0 / np.sum(s ** 2))

    flat = offset.ravel()
    coherence = float(np.abs(np.exp(1j * flat).mean()))
    max_offset = float(np.abs(offset).max())

    return {
        "n_genes": int(n_genes),
        "between_block_phase_variance": round(total_var, 5),
        "common_shift_share": round(float(common_var / total_var), 4),
        "gene_specific_share": round(float(resid_var / total_var), 4),
        "phase_deviation_effective_rank": round(eff_rank, 3),
        "block_offset_coherence": round(coherence, 4),
        "max_block_offset_rad": round(max_offset, 4),
        "max_block_offset_in_quarter_periods": round(
            float(max_offset / (np.pi / 2.0)), 4),
        "block_offsets_rad": [round(float(o), 4) for o in flat],
        "levels_axis2": [str(lv) for lv in levels],
        "reading": (
            "block_offset_coherence near 1 says axis 2 shifts every gene's "
            "axis-1 phase by the *same* amount -- a pure phase shift of axis "
            "1's own loop rather than a second loop. "
            "max_block_offset_in_quarter_periods says whether that shift is "
            "large enough to resolve as a separate circle; a coherent but tiny "
            "shift is a modulation, not a coordinate. gene_specific_share is "
            "the size of the residual (phase-noise) component"),
    }


def _matched(X, theta1, groups, seed, n_boot) -> dict:
    Z = standardize(X)
    D = pairwise_distances(Z, metric="correlation")
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    t1 = np.asarray(theta1, dtype=float)
    labels = np.asarray(groups)

    same1 = (np.abs(np.subtract.outer(t1, t1)) < 1e-9)[iu]
    diff2 = np.not_equal.outer(labels, labels)[iu]
    matched = same1 & diff2
    other = ~same1

    if not matched.any() or not other.any():
        return {"mean_distance_matched": None,
                "mean_distance_across_axis1": None,
                "separation_ratio": None, "separation_absolute": None,
                "separation_bootstrap_ci95": [None, None],
                "within_cell_permutation_degenerate": None,
                "separation_p": None,
                "reading": "no pair of samples shares an axis-1 value while "
                           "differing in axis 2, so the matched comparison is "
                           "not defined on this design"}

    d_matched = float(D[iu][matched].mean())
    d_other = float(D[iu][other].mean())

    key1 = np.round(t1, 9)
    order = {v: i for i, v in enumerate(sorted(set(key1.tolist())))}
    cell = np.array([order[v] for v in key1])
    groups_arr = labels
    # a cell is degenerate when it cannot shuffle: every occupant already
    # carries a different axis-2 label, so a within-cell permutation is the
    # identity relabelling and p is pinned at 1.0
    degenerate = all(len(set(groups_arr[cell == c].tolist())) ==
                     int((cell == c).sum()) for c in np.unique(cell))

    rng = np.random.default_rng(seed)
    obs = d_other - d_matched
    perm_null = np.empty(400)
    for t in range(400):
        perm = groups_arr.copy()
        for c in np.unique(cell):
            rows = np.flatnonzero(cell == c)
            perm[rows] = groups_arr[rng.permutation(rows)]
        dp = np.not_equal.outer(perm, perm)[iu]
        sub = same1 & dp
        perm_null[t] = d_other - float(D[iu][sub].mean()) if sub.any() else 0.0
    p_sep = (float((np.count_nonzero(perm_null >= obs) + 1) / (perm_null.size + 1))
             if not degenerate else None)

    boot = np.empty(int(n_boot))
    for t in range(int(n_boot)):
        pick = rng.integers(0, n, size=n)
        Db = D[np.ix_(pick, pick)]
        iub = np.triu_indices(n, 1)
        s1 = (np.abs(np.subtract.outer(t1[pick], t1[pick])) < 1e-9)[iub]
        d2 = np.not_equal.outer(groups_arr[pick], groups_arr[pick])[iub]
        boot[t] = (Db[iub][~s1].mean() - Db[iub][s1 & d2].mean()
                   if (s1 & d2).any() else np.nan)
    boot = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(boot, [2.5, 97.5]).tolist()
              if boot.size > 10 else [None, None])

    return {
        "mean_distance_matched": round(d_matched, 4),
        "mean_distance_across_axis1": round(d_other, 4),
        "separation_ratio": round(float(d_matched / d_other), 4) if d_other > 0
                            else None,
        "separation_absolute": round(obs, 4),
        "separation_bootstrap_ci95": [None if lo is None else round(lo, 4),
                                      None if hi is None else round(hi, 4)],
        "within_cell_permutation_degenerate": bool(degenerate),
        "separation_p": None if p_sep is None else round(p_sep, 4),
        "reading": (
            "separation_ratio < 1 means that at a fixed axis-1 value axis 2 "
            "still moves the point, but by less than a step of axis 1 does, so "
            "it cannot be an axis of comparable size. On a full grid (one "
            "sample per cell) the within-cell permutation is degenerate, so "
            "the effect size is quoted with a sample-level bootstrap interval "
            "instead of a permutation p"),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def axis_geometry(X: np.ndarray, axis1, axis2, groups=None, *,
                  period1: float | None = None, period2: float | None = None,
                  seed: int = 0, n_perm: int = 2000, n_boot: int = 2000,
                  n_perm_sep: int = 1000) -> dict:
    """Ask whether ``axis2`` is a new coordinate or a modulation of ``axis1``.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The state cloud.  Replicate structure must already be handled by the
        caller: well-separated groups each contribute their own loop, so an
        uncentred cloud measures the grouping instead of the axes.
    axis1, axis2 : sequence of float
        The design's two coordinates, one value per sample.  ``axis1`` is the
        reference axis (the one expected to carry the geometry); ``axis2`` is
        the axis under test.  ``axis2`` may be cyclic (a phase, a clock time) or
        categorical (a treatment block) -- it is only ever used as a *blocking*
        label in the separable and phase blocks, and as a circle in the
        dominant block, so a categorical axis simply gets its ``r`` and its
        offsets computed on its own set of levels.
    groups : sequence, optional
        Blocking labels for axis 2.  Defaults to ``axis2`` itself, which is the
        right choice when axis 2 is a small set of repeated levels.
    period1, period2 : float, optional
        Periods of the two axes in their own units.  ``None`` means the axis is
        already an angle on a :math:`2\\pi` circle.
    seed, n_perm, n_boot, n_perm_sep : int
        Seeds and repetition counts.  All are recorded in the result.

    Returns
    -------
    dict
        ``dominant``, ``separable``, ``phase_or_shape``, ``matched`` blocks,
        each with a ``reading`` string, plus ``verdict``, ``reason`` and the
        ``criteria`` used.  Quote ``verdict`` together with the four blocks: the
        verdict is a summary of them, not a substitute.

    Notes
    -----
    The function is deliberately blind to what the axes *are*.  The same call
    scores a clock against an oestrous phase (mouse liver), a cell-cycle phase
    against a genotype (yeast), or a circadian phase against a sleep-wake
    schedule (human forced desynchrony).  The verdict is only as good as the
    design: see :func:`classify_axis` for what each verdict licenses.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    n = X.shape[0]
    if len(axis1) != n or len(axis2) != n:
        raise ValueError("each axis must have one entry per sample")

    theta1 = _phase(axis1, period1)
    theta2 = _phase(axis2, period2)
    p1 = float(period1) if period1 else 2.0 * np.pi
    p2 = float(period2) if period2 else 2.0 * np.pi
    if groups is None:
        groups = np.asarray(axis2)

    dominant = _dominant(X, theta1, theta2, p1, p2, groups, seed, n_perm)
    separable = _separable(X, theta1, theta2, groups, seed, n_perm_sep)
    phase_or_shape = _phase_or_shape(X, theta1, groups, seed)
    matched = _matched(X, theta1, groups, seed, n_boot)

    verdict, reason = classify_axis(dominant, separable, phase_or_shape, matched)

    return {
        "n_samples": int(n),
        "n_genes": int(X.shape[1]),
        "n_axis2_levels": len(set(np.asarray(groups).tolist())),
        "dominant": dominant,
        "separable": separable,
        "phase_or_shape": phase_or_shape,
        "matched": matched,
        "verdict": verdict,
        "reason": reason,
        "criteria": dict(CRITERIA),
        "settings": {"seed": int(seed), "n_perm": int(n_perm),
                     "n_boot": int(n_boot), "n_perm_sep": int(n_perm_sep)},
    }


def classify_axis(dominant: dict, separable: dict, phase_or_shape: dict,
                  matched: dict, criteria: dict | None = None) -> tuple:
    """Apply the pre-registered criteria; return ``(verdict, reason)``.

    The thresholds are in :data:`CRITERIA` and are documented in the module
    docstring.  ``None`` anywhere it is needed refuses rather than guesses.
    """
    c = dict(CRITERIA if criteria is None else criteria)

    r2 = dominant.get("mantel_r_axis2")
    coh = phase_or_shape.get("block_offset_coherence")
    shift = phase_or_shape.get("max_block_offset_in_quarter_periods")
    ratio = matched.get("separation_ratio")
    share = separable.get("additive_share_of_both")
    cos = separable.get("subspace_cosine")
    if r2 is None:
        return "undetermined", "axis 2 carries no measurable geometry to classify"

    # -- an effect of its own ---------------------------------------------
    if r2 >= c["r_strong"]:
        extra = ""
        if share is not None and cos is not None:
            extra = (f"; the two axes also sit in separate subspaces (additive "
                     f"share {share:.2f}, cosine {cos:+.2f})")
        return "independent_axes", (
            f"axis 2 carries a strong effect of its own (r = {r2:+.3f})"
            + extra)

    # -- a modulation of axis 1 -------------------------------------------
    if (r2 < c["r_inert"] and coh is not None
            and coh >= c["coherence_common"] and shift is not None
            and shift < c["shift_quarter_periods"] and ratio is not None
            and ratio >= c["ratio_modulated"]):
        return "modulation", (
            f"axis 2 has no Mantel effect of its own (r = {r2:+.3f}) but it "
            f"still moves points at a fixed axis-1 value (ratio {ratio:.2f}) "
            f"and it does so by shifting every gene's axis-1 phase by one "
            f"common offset (coherence {coh:.2f}) smaller than half a "
            f"quarter-period ({shift:.2f} quarter-periods). Axis 2 modulates "
            f"axis 1's loop rather than opening a second coordinate")

    # -- present but too weak to act --------------------------------------
    if (r2 < c["r_inert"] and ratio is not None
            and ratio < c["ratio_modulated"]):
        return "independent_axes_inert", (
            f"axis 2 carries no Mantel effect (r = {r2:+.3f}) and barely moves "
            f"points even at a fixed axis-1 value (ratio {ratio:.2f}), so it "
            f"is orthogonal to axis 1 by construction rather than by acting on "
            f"it")

    # -- refuse ------------------------------------------------------------
    bits = []
    if r2 is not None:
        bits.append(f"r = {r2:+.3f}")
    if share is not None and cos is not None:
        bits.append(f"additive share {share:.2f}, cosine {cos:+.2f}")
    if coh is not None and shift is not None:
        bits.append(f"coherence {coh:.2f}, shift {shift:.2f} quarter-periods")
    if ratio is not None:
        bits.append(f"matched ratio {ratio:.2f}")
    return "undetermined", (
        "the design does not separate the possibilities ("
        + "; ".join(bits) + ") -- in particular a coherent shift at or above "
        "half a quarter-period is close to what a second loop would look like, "
        "so the axis needs examining rather than classifying")


def summarise_axis(rec: dict) -> str:
    """One line for a results table, in the same voice as the protocol."""
    v = rec["verdict"]
    m = rec["matched"]
    d = rec["dominant"]
    bits = [f"r2 {d['mantel_r_axis2']:+.2f}"]
    if m.get("separation_ratio") is not None:
        bits.append(f"ratio {m['separation_ratio']:.2f}")
    if rec["phase_or_shape"].get("block_offset_coherence") is not None:
        bits.append(f"coh {rec['phase_or_shape']['block_offset_coherence']:.2f}")
    return f"{v}  ({', '.join(bits)})"
