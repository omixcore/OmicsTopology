"""Synthetic point clouds and expression matrices with *known* topology.

Having ground truth is what makes a topology pipeline trustworthy: you cannot
tell whether a long H1 bar is signal or an artefact unless you have a case
where you already know the answer.

Point-cloud generators
----------------------
===================  ==========================================
``circle``           one loop, so H1 has exactly one long bar
``two_clusters``     two blobs, so H0 has one long bar
``blob``             no structure, so no persistent H1
``torus``            two independent loops, so H1 has two bars
===================  ==========================================

Expression-matrix generators
----------------------------
:func:`two_state_omics` builds a gene-expression matrix whose *latent* state
space differs between two conditions while the marginal gene distributions
stay broadly comparable.  That is exactly the regime where a mean-difference
test sees little but a topological test sees a lot.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "circle",
    "two_clusters",
    "blob",
    "torus",
    "cyclic_state",
    "clustered_state",
    "two_state_omics",
]


# --------------------------------------------------------------------- #
# point clouds with known topology
# --------------------------------------------------------------------- #
def circle(n: int = 24, radius: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """``n`` points on a circle in the plane.  H1: one long bar."""
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    if noise > 0:
        pts = pts + np.random.default_rng(seed).normal(scale=noise, size=pts.shape)
    return pts


def two_clusters(
    n: int = 20,
    separation: float = 6.0,
    spread: float = 0.12,
    seed: int = 0,
) -> np.ndarray:
    """Two tight blobs ``separation`` apart.  H0: one bar of that length."""
    rng = np.random.default_rng(seed)
    left = rng.normal(scale=spread, size=(int(n), 2))
    right = rng.normal(scale=spread, size=(int(n), 2)) + np.array([separation, 0.0])
    return np.vstack([left, right])


def blob(n: int = 30, spread: float = 0.5, dims: int = 3, seed: int = 0) -> np.ndarray:
    """A single isotropic cloud with no persistent topology."""
    return np.random.default_rng(seed).normal(scale=spread, size=(int(n), int(dims)))


def torus(
    n_theta: int = 14,
    n_phi: int = 9,
    major: float = 2.0,
    minor: float = 0.7,
    noise: float = 0.0,
    seed: int = 0,
    structured: bool = True,
) -> np.ndarray:
    """Points on a torus in 3-D.  H1: two loops in principle.

    ``structured=True`` (default) lays the samples on a regular
    ``n_theta x n_phi`` lattice instead of scattering them randomly, which
    makes the major loop come out cleanly (persistence ~1.8 for the default
    lattice).  The minor loop is genuinely harder: it needs a dense sample
    around each small circle, and below that density the simplex budget cuts
    the filtration before the minor classes separate.  Expect the major loop
    plus a family of shorter bars, not a tidy two-bar diagram — this is a
    property of Rips sampling, not of the implementation, and it is a useful
    reminder that topology detection is bounded by sampling.
    """
    rng = np.random.default_rng(seed)
    if structured:
        theta = np.repeat(np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False), int(n_phi))
        phi = np.tile(np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False), int(n_theta))
    else:
        n = int(n_theta)
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        phi = rng.uniform(0.0, 2.0 * np.pi, n)

    ring = major + minor * np.cos(phi)
    pts = np.column_stack([ring * np.cos(theta), ring * np.sin(theta), minor * np.sin(phi)])
    if noise > 0:
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return pts


# --------------------------------------------------------------------- #
# latent biological states
# --------------------------------------------------------------------- #
def cyclic_state(
    n: int = 40,
    radius: float = 1.0,
    noise: float = 0.12,
    n_latent: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """A state space that is a ring: cells step through a cycle of programs.

    Biologically this is a developmental progression, a cell cycle, or a
    stress response that goes out and comes back.
    """
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, int(n)))
    lat = np.zeros((int(n), int(n_latent)), dtype=float)
    lat[:, 0] = radius * np.cos(theta)
    lat[:, 1] = radius * np.sin(theta)
    if n_latent > 2:
        lat[:, 2:] = rng.normal(scale=0.05, size=(int(n), int(n_latent) - 2))
    if noise > 0:
        lat = lat + rng.normal(scale=noise, size=lat.shape)
    return lat


def clustered_state(
    n: int = 40,
    spread: float = 0.35,
    n_latent: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """A single basin: one homogeneous population, no loops."""
    return np.random.default_rng(seed).normal(scale=spread, size=(int(n), int(n_latent)))


def two_state_omics(
    n_genes: int = 600,
    n_per_state: int = 40,
    state_b: str = "cycle",
    noise: float = 0.15,
    n_latent: int = 3,
    seed: int = 0,
    scale: float = 1.0,
) -> tuple:
    """Build a two-condition expression matrix with a known latent geometry.

    Condition A (``"control"``) is a single homogeneous basin.  Condition B
    (``"salt_stress"``) is either a ring (``state_b="cycle"``), which carries a
    real H1 class, or another basin (``state_b="shift"``), which does not.
    Both are projected through the same gene loadings, so the two conditions
    are directly comparable and only their *shape* differs.

    Parameters
    ----------
    n_genes : int
    n_per_state : int
    state_b : {"cycle", "shift"}
    noise : float
        Latent noise within each state.
    n_latent : int
        Number of latent dimensions.
    seed : int
    scale : float
        Global multiplier for expression values.

    Returns
    -------
    (np.ndarray, list, list)
        ``(matrix, groups, labels)`` where ``matrix`` has shape
        ``(2 * n_per_state, n_genes)``, ``groups`` is a list of condition
        codes and ``labels`` is ``["control", "salt_stress"]``.
    """
    rng = np.random.default_rng(seed)
    n_genes = int(n_genes)
    n_latent = int(n_latent)

    z_control = clustered_state(n_per_state, spread=0.35, n_latent=n_latent, seed=seed + 1)
    if state_b == "cycle":
        z_stress = cyclic_state(n_per_state, radius=1.0, noise=noise,
                                n_latent=n_latent, seed=seed + 2)
    elif state_b == "shift":
        z_stress = clustered_state(n_per_state, spread=0.35, n_latent=n_latent,
                                   seed=seed + 3)
        z_stress[:, 0] += 1.7
    else:
        raise ValueError("state_b must be 'cycle' or 'shift'")

    Z = np.vstack([z_control, z_stress])

    # Shared gene loadings and per-gene baselines: both conditions are
    # generated by exactly the same gene space, so the difference is purely
    # geometric.
    loadings = rng.normal(scale=1.0, size=(n_latent, n_genes))
    baseline = rng.uniform(4.0, 8.0, size=n_genes)
    X = Z @ loadings + baseline
    X += rng.normal(scale=0.35, size=X.shape)
    X = np.clip(X * scale, 0.0, None)

    groups = [0] * int(n_per_state) + [1] * int(n_per_state)
    labels = ["control", "salt_stress"]
    return X, groups, labels
