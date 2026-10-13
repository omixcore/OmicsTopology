"""Vector-valued summaries of persistence diagrams.

Betti curves and persistence landscapes turn a diagram into a function of the
filtration scale, which makes diagrams from different states comparable even
when they have different numbers of bars.
"""

from __future__ import annotations

import numpy as np

from .diagram import PersistenceDiagram

__all__ = [
    "betti_curve",
    "persistence_landscape",
    "landscape_norm",
    "topological_summary",
    "summary_table",
]


def _default_grid(diagram: PersistenceDiagram, n_points: int) -> np.ndarray:
    finite = diagram.finite()
    hi = float(finite[:, 1].max()) if finite.size else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    return np.linspace(0.0, hi, int(n_points))


def betti_curve(
    diagram: PersistenceDiagram,
    dim: int,
    grid: np.ndarray | None = None,
    n_points: int = 128,
) -> tuple:
    """Betti number as a function of the filtration scale.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension.
    grid : array-like, optional
        Scale values; defaults to ``n_points`` values spanning the finite
        death range.
    n_points : int
        Grid size when ``grid`` is not given.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(grid, betti)``
    """
    if grid is None:
        grid = _default_grid(diagram, n_points)
    grid = np.asarray(grid, dtype=float)
    bars = diagram.by_dim(dim)
    if bars.size == 0:
        return grid, np.zeros_like(grid)
    birth = bars[:, 0][None, :]
    death = bars[:, 1][None, :]
    alive = (birth <= grid[:, None]) & (grid[:, None] < death)
    return grid, alive.sum(axis=1).astype(float)


def persistence_landscape(
    diagram: PersistenceDiagram,
    dim: int,
    grid: np.ndarray | None = None,
    n_points: int = 128,
    k_max: int = 3,
) -> tuple:
    """Persistence landscape of a diagram.

    The landscape is the collection of tent functions
    ``max(0, min(t - b, d - t))`` sorted in decreasing order at every scale.
    ``lambda_1`` is the strongest feature at each scale; higher ``lambda_k``
    capture secondary structure.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
    grid : array-like, optional
    n_points : int
    k_max : int
        Number of landscape functions to return.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(grid, landscape)`` with ``landscape`` of shape
        ``(k_max, len(grid))``.  Missing ranks are filled with zeros.
    """
    if grid is None:
        grid = _default_grid(diagram, n_points)
    grid = np.asarray(grid, dtype=float)
    lam = np.zeros((int(k_max), grid.size), dtype=float)

    bars = diagram.finite(dim)
    if bars.size == 0:
        return grid, lam

    birth = bars[:, 0][:, None]
    death = bars[:, 1][:, None]
    tents = np.minimum(grid[None, :] - birth, death - grid[None, :])
    np.maximum(tents, 0.0, out=tents)
    tents = -np.sort(-tents, axis=0)
    k = min(int(k_max), tents.shape[0])
    lam[:k] = tents[:k]
    return grid, lam


def landscape_norm(grid: np.ndarray, landscape: np.ndarray, p: int = 2) -> float:
    """L^p norm of a landscape, a compact scalar summary of its shape."""
    grid = np.asarray(grid, dtype=float)
    landscape = np.atleast_2d(np.asarray(landscape, dtype=float))
    if grid.size < 2:
        return float(np.abs(landscape).max())
    values = np.abs(landscape) ** p
    integral = np.trapezoid(values, grid, axis=1) if hasattr(np, "trapezoid") else np.trapz(values, grid, axis=1)
    return float(np.sum(integral) ** (1.0 / p))


def topological_summary(
    diagram: PersistenceDiagram,
    rel: bool = False,
    max_dim: int = 1,
    infinite_as: float | None = None,
    scale: float | None = None,
) -> dict:
    """Summary statistics of a diagram.

    Parameters
    ----------
    diagram : PersistenceDiagram
    rel : bool
        Normalise persistence values by ``scale``, which makes summaries
        comparable between states whose embeddings live at different overall
        scales.
    max_dim : int
        Highest dimension to summarise.
    infinite_as : float or None
        Length assigned to bars that never die.  Defaults to the diagram's
        truncation threshold, so a loop that outlives the filtration still
        counts as ``threshold - birth`` rather than disappearing from the
        statistics.
    scale : float or None
        Divisor for ``rel``.  When omitted, the largest observed death is
        used; pass the shared threshold to put several states on one scale.

    Returns
    -------
    dict
    """
    if infinite_as is None and np.isfinite(diagram.threshold):
        infinite_as = float(diagram.threshold)

    out = {
        "threshold": float(diagram.threshold),
        "n_points": int(diagram.meta.get("n_points", 0)),
    }

    used_scale = scale
    if used_scale is None:
        p = diagram.persistence(max_dim, infinite_as=infinite_as)
        used_scale = float(p.max()) if p.size else 0.0
    if rel:
        out["scale"] = float(used_scale)
    norm = float(used_scale) if (rel and used_scale and used_scale > 0) else 1.0

    for d in range(0, int(max_dim) + 1):
        out[f"h{d}_n_finite"] = diagram.n_bars(d, finite_only=True)
        out[f"h{d}_n_essential"] = int(diagram.essential(d).shape[0])
        out[f"h{d}_total_persistence"] = diagram.total_persistence(d, infinite_as=infinite_as) / norm
        out[f"h{d}_max_persistence"] = diagram.max_persistence(d, infinite_as=infinite_as) / norm
        out[f"h{d}_entropy"] = diagram.persistence_entropy(d, infinite_as=infinite_as)

    half = 0.5 * float(infinite_as) if infinite_as else 0.0
    out["h0_components_at_mean"] = diagram.betti(0, half)
    return out


def summary_table(summaries: dict) -> list:
    """Turn ``{state_name: summary_dict}`` into rows for a table."""
    rows = []
    keys = []
    for stats in summaries.values():
        for k in stats:
            if k not in keys:
                keys.append(k)
    for name, stats in summaries.items():
        row = {"state": name}
        row.update({k: stats.get(k) for k in keys})
        rows.append(row)
    return rows
