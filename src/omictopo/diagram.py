"""Persistence diagrams and the statistics that summarise them.

A persistence diagram is a multiset of intervals ``[birth, death)`` tagged by
homological dimension.  Reading the dimensions:

* ``H0`` — connected components of the state space.  Several long H0 bars
  mean the samples occupy several well-separated basins.
* ``H1`` — loops.  A loop means states are arranged around a cycle, i.e. the
  transition is *cyclic* rather than a simple drift from A to B.
* ``H2`` — cavities.  Requires a dense cloud and a generous threshold.

An ``inf`` death marks a class still alive at the truncation threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PersistenceDiagram"]


@dataclass
class PersistenceDiagram:
    """Multiset of birth-death pairs, grouped by homology dimension.

    Attributes
    ----------
    bars : np.ndarray, shape (n_bars, 3)
        Columns are ``birth``, ``death`` and ``dim``.  ``death`` may be
        ``inf`` for classes alive at the truncation threshold.
    threshold : float
        Distance threshold the filtration was truncated at.
    meta : dict
        Free-form provenance (number of points, backend, ...).
    """

    bars: np.ndarray
    threshold: float = float("inf")
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bars = np.asarray(self.bars, dtype=float).reshape(-1, 3)
        order = np.lexsort((self.bars[:, 1], self.bars[:, 0], self.bars[:, 2]))
        self.bars = self.bars[order]

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.bars.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = ", ".join(
            f"H{d}={self.n_bars(d)}" for d in self.dims
        )
        return f"<PersistenceDiagram {parts}, threshold={self.threshold:.4g}>"

    @property
    def dims(self) -> list:
        """Dimensions present, ascending."""
        return [int(d) for d in np.unique(self.bars[:, 2])] if len(self) else []

    def by_dim(self, dim: int) -> np.ndarray:
        """Bars of one dimension as an ``(n, 2)`` array of (birth, death)."""
        if len(self) == 0:
            return np.empty((0, 2), dtype=float)
        return self.bars[self.bars[:, 2] == dim][:, :2]

    def n_bars(self, dim: int, finite_only: bool = False) -> int:
        """Number of bars in a dimension."""
        b = self.by_dim(dim)
        if finite_only:
            b = b[np.isfinite(b[:, 1])]
        return int(b.shape[0])

    def finite(self, dim: int | None = None) -> np.ndarray:
        """Bars with a finite death, optionally restricted to one dimension."""
        bars = self.bars if dim is None else self.bars[self.bars[:, 2] == dim]
        if bars.size == 0:
            return np.empty((0, 3), dtype=float)
        return bars[np.isfinite(bars[:, 1])]

    def essential(self, dim: int | None = None) -> np.ndarray:
        """Bars alive at the truncation threshold."""
        bars = self.bars if dim is None else self.bars[self.bars[:, 2] == dim]
        if bars.size == 0:
            return np.empty((0, 3), dtype=float)
        return bars[~np.isfinite(bars[:, 1])]

    # ------------------------------------------------------------------ #
    # statistics
    # ------------------------------------------------------------------ #
    def persistence(self, dim: int | None = None, infinite_as: float | None = None) -> np.ndarray:
        """Bar lengths ``death - birth``.

        Parameters
        ----------
        dim : int or None
        infinite_as : float or None
            Length to assign to bars that never die.  Without it they are
            skipped; with it they are counted as if they died at that value.
            Passing the truncation threshold is the usual choice, and it is
            what makes summaries robust to a loop outliving the filtration:
            an essential class then contributes ``threshold - birth`` instead
            of silently contributing nothing.
        """
        bars = self.finite(dim)
        lengths = list(bars[:, 1] - bars[:, 0]) if bars.size else []
        if infinite_as is not None:
            ess = self.essential(dim)
            if ess.size:
                lengths.extend(float(infinite_as) - ess[:, 0])
        return np.asarray(lengths, dtype=float)

    def max_persistence(self, dim: int, infinite_as: float | None = None) -> float:
        """Longest bar in a dimension (0.0 when there is none)."""
        p = self.persistence(dim, infinite_as=infinite_as)
        return float(p.max()) if p.size else 0.0

    def total_persistence(self, dim: int, infinite_as: float | None = None) -> float:
        """Sum of bar lengths.  The main "how much topology" number."""
        p = self.persistence(dim, infinite_as=infinite_as)
        return float(p.sum()) if p.size else 0.0

    def persistence_entropy(self, dim: int, infinite_as: float | None = None) -> float:
        """Shannon entropy of the normalised bar-length distribution.

        Low entropy means a few dominant features; high entropy means many
        comparable features.  Useful to tell "one strong cycle" apart from
        "lots of small noise loops".  Zero-length bars (the common Rips
        artefacts) contribute nothing and are excluded.
        """
        p = self.persistence(dim, infinite_as=infinite_as)
        if p.size == 0:
            return 0.0
        total = p.sum()
        if total <= 0:
            return 0.0
        q = p / total
        nz = q > 1e-12
        if not np.any(nz):
            return 0.0
        q = q[nz]
        return float(-np.sum(q * np.log(q)))

    def betti(self, dim: int, t: float) -> int:
        """Betti number ``beta_dim`` at filtration value ``t``."""
        b = self.by_dim(dim)
        if b.size == 0:
            return 0
        alive = (b[:, 0] <= t) & (t < b[:, 1])
        return int(np.count_nonzero(alive))

    def n_components(self, t: float = 0.0) -> int:
        """Connected components at scale ``t`` (``beta_0`` at that scale).

        With the single-linkage H0 convention at ``t = 0`` this equals the
        number of points; the interesting values are at larger ``t``, where
        it counts how many well-separated basins survive.
        """
        return self.betti(0, t)

    def summary(self) -> dict:
        """Flat dictionary of summary statistics, handy for tables."""
        out = {
            "threshold": float(self.threshold),
            "n_points": int(self.meta.get("n_points", 0)),
            "n_bars_total": len(self),
        }
        for d in self.dims:
            out[f"h{d}_n_finite"] = self.n_bars(d, finite_only=True)
            out[f"h{d}_n_essential"] = int(self.essential(d).shape[0])
            out[f"h{d}_total_persistence"] = self.total_persistence(d)
            out[f"h{d}_max_persistence"] = self.max_persistence(d)
            out[f"h{d}_entropy"] = self.persistence_entropy(d)
        return out

    def to_records(self) -> list:
        """List of ``dict`` rows, ready to be written as a table."""
        return [
            {"dim": int(d), "birth": float(b), "death": float(dd),
             "persistence": float(dd - b) if np.isfinite(dd) else float("inf"),
             "essential": not np.isfinite(dd)}
            for b, dd, d in self.bars
        ]
