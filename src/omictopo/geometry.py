"""Distance computation, standardisation and embeddings for state spaces.

The omics state space is the space in which each biological sample (or cell,
or time point) is one point.  Feature vectors are far too high dimensional to
build a simplicial complex directly, so we first embed to a low-dimensional
Euclidean space and then let the topology work in that space.

Only numpy is required: the embedding is a plain SVD-based PCA so that the
package stays dependency-light and installable without a compiler.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "standardize",
    "pairwise_distances",
    "pca_embed",
    "subsample",
]


def standardize(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Z-score each feature (column), leaving zero-variance features as zeros.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    ddof : int, optional
        Delta degrees of freedom for the standard deviation.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=ddof)
    safe = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / safe
    Z[:, sd <= 0] = 0.0
    return Z


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Pairwise distance matrix.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    metric : {"euclidean", "cosine", "correlation", "manhattan"}
        The correlation/cosine metrics are useful when samples should be
        compared by expression *pattern* rather than by absolute level.

    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        Symmetric matrix with a zero diagonal.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")

    if metric == "euclidean":
        sq = np.einsum("ij,ij->i", X, X)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(d2, 0.0, out=d2)
        D = np.sqrt(d2)
    elif metric == "manhattan":
        D = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    elif metric in ("cosine", "correlation"):
        Y = X - X.mean(axis=1, keepdims=True) if metric == "correlation" else X
        norm = np.linalg.norm(Y, axis=1)
        norm = np.where(norm > 0, norm, 1.0)
        Y = Y / norm[:, None]
        D = 1.0 - np.clip(Y @ Y.T, -1.0, 1.0)
    else:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of "
            "'euclidean', 'cosine', 'correlation', 'manhattan'"
        )

    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D


def pca_embed(X: np.ndarray, n_components: int = 3, whiten: bool = False) -> np.ndarray:
    """Principal component embedding via SVD.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int
        Number of retained components (clipped to the matrix rank).
    whiten : bool
        Scale each component to unit variance.  Useful when you want the
        embedding to weight rare-but-sharp modes of variation equally with
        broad ones.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    n_samples = X.shape[0]
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(n_components, U.shape[1]))
    scores = U[:, :k] * S[:k]
    if whiten:
        scale = S[:k] / np.sqrt(max(n_samples - 1, 1))
        scale = np.where(scale > 0, scale, 1.0)
        scores = scores / scale
    return scores


def subsample(
    X: np.ndarray,
    max_points: int | None = None,
    seed: int = 0,
    return_index: bool = False,
):
    """Randomly subsample rows to keep the simplicial complex tractable.

    Persistent homology of a Vietoris-Rips complex grows roughly as
    O(n^3) in the number of points, so clouds should be capped before the
    filtration is built.

    Parameters
    ----------
    X : array-like, shape (n_samples, ...)
    max_points : int or None
        If None or >= n_samples, the input is returned unchanged.
    seed : int
        Seed for the random generator, so runs are reproducible.
    return_index : bool
        Also return the selected row indices.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
    """
    X = np.asarray(X)
    n = X.shape[0]
    if max_points is None or max_points >= n:
        if return_index:
            return X, np.arange(n)
        return X

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    if return_index:
        return X[idx], idx
    return X[idx]
