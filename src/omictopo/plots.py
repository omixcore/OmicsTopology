"""Publication-oriented figures for state-space topology.

matplotlib is imported lazily so that the core package stays usable in a
headless analysis environment with only numpy and scipy installed.

Figure labels are in English on purpose: it keeps the code free of
font-provisioning issues across platforms.
"""

from __future__ import annotations

import numpy as np

from .diagram import PersistenceDiagram
from .summary import betti_curve, persistence_landscape

__all__ = [
    "plot_persistence_diagram",
    "plot_barcode",
    "plot_betti_curves",
    "plot_landscape",
    "plot_point_cloud",
    "plot_state_comparison",
]

_DIM_COLORS = {0: "#185FA5", 1: "#D85A30", 2: "#0F6E56"}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "matplotlib is required for plotting; install it with "
            "`pip install omictopo[plot]`"
        ) from exc


def _finish(fig, path, show):
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:  # pragma: no cover - interactive only
        plt = _plt()
        plt.show()
    else:
        plt = _plt()
        plt.close(fig)
    return path


def plot_persistence_diagram(
    diagram: PersistenceDiagram,
    path: str | None = None,
    title: str = "Persistence diagram",
    max_dim: int = 1,
    show: bool = False,
):
    """Scatter of birth vs death, one colour per dimension."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    finite = diagram.finite()
    hi = float(finite[:, 1].max()) if finite.size else 1.0
    hi = hi if hi > 0 else 1.0

    ax.plot([0, hi], [0, hi], color="#B4B2A9", lw=0.8, ls="--", zorder=1)
    for d in range(0, int(max_dim) + 1):
        bars = diagram.by_dim(d)
        fin = bars[np.isfinite(bars[:, 1])]
        if fin.size:
            ax.scatter(
                fin[:, 0], fin[:, 1], s=22, alpha=0.85, zorder=3,
                color=_DIM_COLORS.get(d, "#5F5E5A"), label=f"H{d}",
            )
        ess = bars[~np.isfinite(bars[:, 1])]
        for b, _ in ess:
            ax.plot([b, hi], [hi, hi], color=_DIM_COLORS.get(d, "#5F5E5A"),
                    lw=1.2, alpha=0.7, zorder=2)
        if ess.size:
            ax.scatter(ess[:, 0], np.full(ess.shape[0], hi), marker="^", s=30,
                       color=_DIM_COLORS.get(d, "#5F5E5A"), zorder=4,
                       label=f"H{d} essential")

    ax.set_xlabel("Birth (distance scale)")
    ax.set_ylabel("Death (distance scale)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, hi * 1.05)
    ax.set_ylim(0, hi * 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_barcode(
    diagram: PersistenceDiagram,
    path: str | None = None,
    title: str = "Persistence barcode",
    max_dim: int = 1,
    max_bars: int = 40,
    show: bool = False,
):
    """Horizontal bars sorted by length: the most persistent features on top."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 0.28 * max_bars * 0.5 + 1.6))

    y = 0
    yticks, ylabels = [], []
    finite = diagram.finite()
    longest = float(finite[:, 1].max()) if finite.size else 1.0

    for d in range(0, int(max_dim) + 1):
        bars = diagram.finite(d)[:, :2]
        if bars.size == 0:
            continue
        bars = bars[np.argsort(-(bars[:, 1] - bars[:, 0]))][:max_bars]
        for b, dd in bars:
            ax.plot([b, dd], [y, y], lw=2.2, solid_capstyle="butt",
                    color=_DIM_COLORS.get(d, "#5F5E5A"))
            yticks.append(y)
            ylabels.append(f"H{d}")
            y += 1
        ess = diagram.essential(d)[:, :2]
        for b, _ in ess:
            ax.plot([b, longest], [y, y], lw=2.2, alpha=0.55,
                    color=_DIM_COLORS.get(d, "#5F5E5A"))
            yticks.append(y)
            ylabels.append(f"H{d} inf")
            y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Distance scale")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_betti_curves(
    diagrams: dict,
    path: str | None = None,
    dim: int = 1,
    title: str | None = None,
    show: bool = False,
):
    """Betti curves of several states on one axis.

    Parameters
    ----------
    diagrams : dict
        ``{state_name: PersistenceDiagram}``
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["#185FA5", "#D85A30", "#0F6E56", "#534AB7", "#854F0B", "#993556"]
    for i, (name, dgm) in enumerate(diagrams.items()):
        grid, beta = betti_curve(dgm, dim=dim)
        ax.plot(grid, beta, lw=1.6, color=colors[i % len(colors)], label=str(name))
    ax.set_xlabel("Distance scale")
    ax.set_ylabel(f"beta_{dim} (number of {('components' if dim == 0 else 'loops')})")
    ax.set_title(title or f"Betti curve, H{dim}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_landscape(
    diagrams: dict,
    path: str | None = None,
    dim: int = 1,
    k_max: int = 2,
    title: str | None = None,
    show: bool = False,
):
    """First ``k_max`` landscape functions per state."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["#185FA5", "#D85A30", "#0F6E56", "#534AB7"]
    for i, (name, dgm) in enumerate(diagrams.items()):
        grid, lam = persistence_landscape(dgm, dim=dim, k_max=k_max)
        for k in range(lam.shape[0]):
            ax.plot(
                grid, lam[k], lw=1.5, color=colors[i % len(colors)],
                alpha=1.0 - 0.28 * k,
                label=str(name) if k == 0 else None,
            )
    ax.set_xlabel("Distance scale")
    ax.set_ylabel(f"landscape lambda, H{dim}")
    ax.set_title(title or f"Persistence landscape, H{dim}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_point_cloud(
    clouds: dict,
    path: str | None = None,
    title: str = "State space (2-D embedding)",
    show: bool = False,
    annotate_bars: bool = True,
):
    """Side-by-side scatter of the embedded state clouds."""
    plt = _plt()
    n = len(clouds)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    colors = ["#185FA5", "#D85A30", "#0F6E56", "#534AB7"]
    for i, (name, pts) in enumerate(clouds.items()):
        ax = axes[0][i]
        pts = np.asarray(pts, dtype=float)
        xy = pts[:, :2] if pts.shape[1] >= 2 else np.column_stack([pts[:, 0], np.zeros(pts.shape[0])])
        ax.scatter(xy[:, 0], xy[:, 1], s=26, alpha=0.8,
                   color=colors[i % len(colors)], edgecolors="white", linewidths=0.4)
        ax.set_title(str(name), fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if pts.shape[1] >= 2 else "")
        ax.grid(alpha=0.15, lw=0.5)
        if annotate_bars:
            ax.text(0.03, 0.97, f"n = {pts.shape[0]}", transform=ax.transAxes,
                    va="top", fontsize=9, color="#5F5E5A")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_state_comparison(
    clouds: dict,
    diagrams: dict,
    path: str | None = None,
    dim: int = 1,
    max_dim: int = 1,
    title: str = "State-space topology compared",
    show: bool = False,
):
    """Four-panel overview: clouds, diagrams, Betti curves, landscapes."""
    plt = _plt()
    fig = plt.figure(figsize=(11.0, 7.6))
    ax_cloud = fig.add_subplot(2, 2, 1)
    ax_diag = fig.add_subplot(2, 2, 2)
    ax_beta = fig.add_subplot(2, 2, 3)
    ax_land = fig.add_subplot(2, 2, 4)
    colors = ["#185FA5", "#D85A30", "#0F6E56", "#534AB7"]

    for i, (name, pts) in enumerate(clouds.items()):
        pts = np.asarray(pts, dtype=float)
        xy = pts[:, :2] if pts.shape[1] >= 2 else np.column_stack([pts[:, 0], np.zeros(pts.shape[0])])
        ax_cloud.scatter(xy[:, 0], xy[:, 1], s=24, alpha=0.8,
                         color=colors[i % len(colors)], label=str(name),
                         edgecolors="white", linewidths=0.4)
    ax_cloud.set_xlabel("PC1")
    ax_cloud.set_ylabel("PC2")
    ax_cloud.set_title("Embedded state space", fontsize=11)
    ax_cloud.legend(frameon=False, fontsize=9)
    ax_cloud.grid(alpha=0.15, lw=0.5)

    hi = 1.0
    for dgm in diagrams.values():
        finite = dgm.finite()
        if finite.size:
            hi = max(hi, float(finite[:, 1].max()))
    ax_diag.plot([0, hi], [0, hi], color="#B4B2A9", lw=0.8, ls="--")
    for i, (name, dgm) in enumerate(diagrams.items()):
        for d in range(0, int(max_dim) + 1):
            bars = dgm.finite(d)[:, :2]
            if bars.size == 0:
                continue
            ax_diag.scatter(bars[:, 0], bars[:, 1], s=20, alpha=0.8,
                            color=colors[i % len(colors)], marker="o" if d == 0 else "s",
                            label=str(name) if d == max_dim else None)
    ax_diag.set_xlabel("Birth")
    ax_diag.set_ylabel("Death")
    ax_diag.set_title(f"Persistence diagram (H{max_dim})", fontsize=11)
    ax_diag.legend(frameon=False, fontsize=9)
    ax_diag.grid(alpha=0.15, lw=0.5)

    for i, (name, dgm) in enumerate(diagrams.items()):
        grid, beta = betti_curve(dgm, dim=dim)
        ax_beta.plot(grid, beta, lw=1.6, color=colors[i % len(colors)], label=str(name))
    ax_beta.set_xlabel("Distance scale")
    ax_beta.set_ylabel(f"beta_{dim}")
    ax_beta.set_title(f"Betti curve (H{dim})", fontsize=11)
    ax_beta.legend(frameon=False, fontsize=9)
    ax_beta.grid(alpha=0.15, lw=0.5)

    for i, (name, dgm) in enumerate(diagrams.items()):
        grid, lam = persistence_landscape(dgm, dim=dim, k_max=2)
        for k in range(lam.shape[0]):
            ax_land.plot(grid, lam[k], lw=1.5, color=colors[i % len(colors)],
                         alpha=1.0 - 0.3 * k, label=str(name) if k == 0 else None)
    ax_land.set_xlabel("Distance scale")
    ax_land.set_ylabel(f"lambda, H{dim}")
    ax_land.set_title(f"Landscape (H{dim})", fontsize=11)
    ax_land.legend(frameon=False, fontsize=9)
    ax_land.grid(alpha=0.15, lw=0.5)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, path, show)
