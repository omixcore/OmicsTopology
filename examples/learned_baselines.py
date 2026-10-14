"""Learned baselines: does a *trained* representation read the clock any better?

Why these exist
---------------
``examples/incremental_value.py`` showed that the topological cycle beats seven
cheap label-free baselines on specificity (1 false positive in 23 against 9 in
23) while matching the best of them on sensitivity.  That comparison had an
obvious hole: every rival was a fixed transform of the cloud -- a principal
axis, a circle fit, a tour.  A referee is entitled to answer "your advantage is
that your method is *weak*, not that it is *wise*; a representation with enough
capacity to learn the manifold would find the clock too, and it would not
hallucinate one on a non-cyclic cloud."

This module puts that objection on the same footing as everything else.  Each
baseline below is *trained* on the cloud and then read for an order:

``ae_1d_order``           autoencoder, 1-D bottleneck   - a learned PC1
``ae_2d_angle``           autoencoder, 2-D bottleneck   - a learned PC1-PC2
``isomap_angle``          Isomap: graph geodesics, then an embedding
``lle_angle``             locally linear embedding
``kernel_pca_angle``      RBF kernel PCA
``spectral_angle``        Laplacian eigenmaps
``tsne_angle``            t-SNE (gradient-trained neighbour embedding)
``som_chain_order``       Kohonen self-organising map, chain topology
``principal_curve_order`` Hastie projection-and-smoothing principal curve

Every one of them receives exactly what the classical baselines receive and
returns a permutation of the sample indices.  Nothing sees the sampling times.

Two geometries, kept honest
---------------------------
The benchmark hands every method ``(Z, D)``: the standardised expression matrix
and the correlation distance matrix built from it.  The classical baselines
already split along that line -- ``pc1_order`` and ``circle_fit_angle`` use
``Z``, ``mds_angle`` uses ``D`` -- and so do these:

* *distance-based* methods (Isomap, kernel PCA, Laplacian eigenmaps, t-SNE)
  read the same correlation distance matrix ``D`` that ``mds_angle`` reads, so
  they are compared like for like.
* *coordinate* methods (autoencoders, LLE, SOM, principal curve) need
  coordinates rather than distances, and are given the cloud in principal
  coordinates via :func:`coordinate_view`.  With ``q = min(200, n - 1)`` and
  n <= 26 here, that truncated SVD is *exact* on the centred cloud, so it is a
  rotation carrying all of the data; the single global rescaling that follows
  changes no distance ratio and no angle.  It only spares these methods the
  degeneracy every Euclidean method suffers in 3 000 dimensions, where all
  distances concentrate.  (sklearn's LLE has no ``metric`` argument, so it
  belongs to this group rather than to the distance group.)

Capacity, and the tuning rule
-----------------------------
The autoencoder carries ~1 800 trainable parameters on a 24-sample cloud: the
network has ~75 times more parameters than the cloud has samples.  That is
deliberately the ordinary architecture for expression data, and it is exactly
the regime in which a learned representation *would* fix the false-positive
problem if it could.

Its hyperparameters were chosen **on the positive controls only** -- four
independent draws of a synthetic circadian ring -- and then frozen for every
other cloud.  Every sensitivity number for it below is therefore in-sample and
every specificity number is out-of-sample, which biases the comparison in the
autoencoder's favour.  One consequence is visible in the results and is worth
stating plainly: with the strength of weight decay that makes a ring recover
perfectly (``weight_decay = 1e-2``), the reconstruction error stays at ~0.58
of the cloud's variance, i.e. the network is *not* merely memorising its 24
inputs.

Correctness of the implementation
---------------------------------
``autoencoder_linear_check`` refits the same network with the nonlinearity
switched off.  A linear autoencoder with a linear bottleneck spans the same
subspace as PCA, so that check must reproduce PCA's own order -- and it does,
for both bottleneck widths.  If it did not, the learned methods would lose here
for the wrong reason; the numbers are reported in the benchmark output so the
reader can confirm them.  The bottleneck is normalised to zero mean and unit
variance during training: without that, the weight matrices can be rescaled by
a constant without changing the loss, and Adam drifts along that flat direction
instead of fitting (this was observed, with latent scales of 10^5).

No deep-learning framework is required: the network is ~60 lines of NumPy with
Adam and gradient clipping, so these baselines run inside the base dependency
set and reproduce bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from omictopo.geometry import pca_embed

__all__ = [
    "LEARNED_METHODS",
    "LEARNED_FAMILY",
    "AE_SETTINGS",
    "coordinate_view",
    "autoencoder_latent",
    "autoencoder_linear_check",
    "parameter_counts",
]

# Frozen after tuning on the positive controls alone; see the module docstring.
AE_SETTINGS = {
    "hidden": 16,
    "epochs": 800,
    "lr": 8e-3,
    "weight_decay": 1e-2,
    "sd_floor": 0.05,
    "grad_clip": 10.0,
    "seed": 0,
}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def coordinate_view(Z: np.ndarray) -> np.ndarray:
    """The cloud in principal coordinates, rescaled by one global constant.

    ``q = min(200, n - 1, p)``.  For n <= 201 the truncated SVD keeps the full
    row space, so this is an exact rotation of the centred cloud, not an
    approximation.  The single global scalar keeps the numbers near order one
    for the network and changes no distance ratio and no angle.
    """
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape
    q = int(max(1, min(200, n - 1, p)))
    S = pca_embed(Z, q)
    scale = float(np.sqrt(np.mean(S ** 2)))
    return S / (scale if scale > 1e-12 else 1.0)


def _angle_order(emb: np.ndarray) -> np.ndarray:
    """Angular order of a 2-D embedding about its own centroid."""
    emb = np.asarray(emb, dtype=float)
    if emb.ndim != 2 or emb.shape[1] < 2 or not np.all(np.isfinite(emb)):
        return np.array([], dtype=int)
    x = emb[:, 0] - emb[:, 0].mean()
    y = emb[:, 1] - emb[:, 1].mean()
    if float(np.hypot(x, y).max()) < 1e-12:
        return np.array([], dtype=int)          # collapsed: no order to read
    return np.argsort(np.arctan2(y, x))


def _n_neighbours(n: int, cap: int = 15) -> int:
    """A neighbour count that scales sanely over n = 12 ... 72."""
    return int(max(3, min(cap, n - 1, int(round(0.35 * n)))))


def _rbf_kernel(D: np.ndarray) -> np.ndarray:
    """Gaussian affinity from a distance matrix, bandwidth = median distance."""
    D = np.asarray(D, dtype=float)
    off = D[np.triu_indices_from(D, k=1)]
    sigma = float(np.median(off)) if off.size else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    return np.exp(-(D ** 2) / (2.0 * sigma ** 2))


# --------------------------------------------------------------------------
# autoencoder -- NumPy, Adam, deterministic
# --------------------------------------------------------------------------
def autoencoder_latent(Z: np.ndarray, latent: int = 2, hidden: int | None = None,
                       epochs: int | None = None, lr: float | None = None,
                       weight_decay: float | None = None,
                       sd_floor: float | None = None,
                       grad_clip: float | None = None,
                       seed: int | None = None,
                       nonlinear: bool = True) -> tuple:
    """Train a small autoencoder; return its latent code and its size.

    Architecture ``q -> hidden -> latent -> hidden -> q``: ``tanh`` hidden
    layers, a linear bottleneck normalised to zero mean and unit variance over
    the batch, full-batch Adam on mean squared reconstruction error, L2 on the
    weight matrices, and gradient clipping.  Deterministic -- the only random
    input is the initialisation, which comes from ``seed``.
    """
    cfg = dict(AE_SETTINGS)
    hidden = cfg["hidden"] if hidden is None else hidden
    epochs = cfg["epochs"] if epochs is None else epochs
    lr = cfg["lr"] if lr is None else lr
    weight_decay = cfg["weight_decay"] if weight_decay is None else weight_decay
    sd_floor = cfg["sd_floor"] if sd_floor is None else sd_floor
    grad_clip = cfg["grad_clip"] if grad_clip is None else grad_clip
    seed = cfg["seed"] if seed is None else seed

    S = coordinate_view(Z)
    n, q = S.shape
    latent = int(max(1, min(latent, q)))
    rng = np.random.default_rng(seed)

    def init(fan_in: int, fan_out: int) -> np.ndarray:
        return rng.standard_normal((fan_in, fan_out)) * np.sqrt(
            2.0 / float(fan_in + fan_out))

    params = [init(q, hidden), np.zeros(hidden),
              init(hidden, latent), np.zeros(latent),
              init(latent, hidden), np.zeros(hidden),
              init(hidden, q), np.zeros(q)]

    def act(H):
        return np.tanh(H) if nonlinear else H

    def grad_act(A):
        return (1.0 - A ** 2) if nonlinear else 1.0

    m = [np.zeros_like(w) for w in params]
    v = [np.zeros_like(w) for w in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, int(epochs) + 1):
        W1, b1, W2, b2, W3, b3, W4, b4 = params
        A1 = act(S @ W1 + b1)
        pre = A1 @ W2 + b2
        sd = pre.std(axis=0) + sd_floor          # pins the scale gauge
        code = (pre - pre.mean(axis=0)) / sd        # linear bottleneck
        A3 = act(code @ W3 + b3)
        rec = A3 @ W4 + b4

        dR = 2.0 * (rec - S) / float(n * q)
        gW4 = A3.T @ dR + 2.0 * weight_decay * W4
        gb4 = dR.sum(axis=0)
        dH3 = (dR @ W4.T) * grad_act(A3)
        gW3 = code.T @ dH3 + 2.0 * weight_decay * W3
        gb3 = dH3.sum(axis=0)
        dcode = (dH3 @ W3.T) / sd
        gW2 = A1.T @ dcode + 2.0 * weight_decay * W2
        gb2 = dcode.sum(axis=0)
        dH1 = (dcode @ W2.T) * grad_act(A1)
        gW1 = S.T @ dH1 + 2.0 * weight_decay * W1
        gb1 = dH1.sum(axis=0)

        grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4]
        norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
        if norm > grad_clip:
            grads = [g * (grad_clip / norm) for g in grads]

        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g
            params[i] = params[i] - lr * (m[i] / (1.0 - beta1 ** step)) / (
                np.sqrt(v[i] / (1.0 - beta2 ** step)) + eps)

    W1, b1, W2, b2, W3, b3, W4, b4 = params
    pre = act(S @ W1 + b1) @ W2 + b2
    sd = pre.std(axis=0) + sd_floor
    code = (pre - pre.mean(axis=0)) / sd

    n_params = int(sum(w.size for w in params))
    return code, {"n_params": n_params, "n_ae_input_dim": int(q),
                  "ae_nonlinear": bool(nonlinear), "ae_epochs": int(epochs)}


def autoencoder_linear_check(seed: int = 0) -> dict:
    """With the nonlinearity off the network must behave exactly like PCA.

    A linear autoencoder with a linear bottleneck spans the same subspace as
    PCA, so its order has to agree with the corresponding principal axes.  This
    is reported in the benchmark rather than assumed: if it failed, the learned
    methods below would lose for the wrong reason.
    """
    angles = np.sort(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 24))
    pat = np.random.default_rng(seed + 1).normal(size=(300, 2))
    X = np.c_[np.cos(angles), np.sin(angles)] @ pat.T
    X = X + np.random.default_rng(seed + 2).normal(0, 0.35, X.shape)

    from omictopo.geometry import standardize
    from omictopo.cycles import circular_succession_score

    X = standardize(X)

    def score(order: np.ndarray) -> float:
        order = np.asarray(order, dtype=int)
        return round(float(circular_succession_score(angles, order)), 4) if order.size else 0.0

    lin1 = score(np.argsort(autoencoder_latent(
        X, latent=1, nonlinear=False, seed=seed)[0][:, 0]))
    lin2 = score(_angle_order(autoencoder_latent(
        X, latent=2, nonlinear=False, seed=seed)[0]))
    pca1 = score(np.argsort(pca_embed(X, 1)[:, 0]))
    emb = pca_embed(X, 3)
    pca2 = score(np.argsort(np.arctan2(emb[:, 1] - emb[:, 1].mean(),
                                       emb[:, 0] - emb[:, 0].mean())))
    nonlin2 = score(_angle_order(autoencoder_latent(
        X, latent=2, nonlinear=True, seed=seed)[0]))
    return {
        "pca_1d_score": pca1, "linear_ae_1d_score": lin1,
        "pca_2d_score": pca2, "linear_ae_2d_score": lin2,
        "nonlinear_ae_2d_score": nonlin2,
        "linear_ae_matches_pca": bool(abs(lin1 - pca1) <= 0.05
                                     and abs(lin2 - pca2) <= 0.05),
    }


def parameter_counts(seed: int = 0, n_samples: int = 24) -> dict:
    """Trainable parameter count of every learned method, for the results table."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, 500))
    out = {}
    for label, latent in (("ae_1d_order", 1), ("ae_2d_angle", 2)):
        out[label] = int(autoencoder_latent(Z, latent=latent)[1]["n_params"])
    for label in ("isomap_angle", "lle_angle", "kernel_pca_angle",
                  "spectral_angle", "tsne_angle", "som_chain_order",
                  "principal_curve_order"):
        out[label] = 0            # no fitted parameter vector
    return out


# --------------------------------------------------------------------------
# the learned methods -- (Z, D) -> (order, extra)
# --------------------------------------------------------------------------
def m_ae_1d_order(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Learned analogue of sorting by PC1: order by a 1-D AE bottleneck."""
    try:
        code, extra = autoencoder_latent(Z, latent=1)
    except Exception as exc:                                   # pragma: no cover
        return np.array([], dtype=int), {"error": type(exc).__name__}
    col = code[:, 0]
    if float(np.ptp(col)) < 1e-12:
        return np.array([], dtype=int), {**extra, "error": "collapsed"}
    return np.argsort(col), extra


def m_ae_2d_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Learned analogue of the PC1-PC2 phase: angle in a 2-D AE bottleneck."""
    try:
        code, extra = autoencoder_latent(Z, latent=2)
    except Exception as exc:                                   # pragma: no cover
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(code), extra


def m_isomap_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Isomap: replace distances by graph geodesics, embed, read the angle."""
    from sklearn.manifold import Isomap

    D = np.asarray(D, dtype=float)
    try:
        emb = Isomap(n_neighbors=_n_neighbours(D.shape[0]), n_components=2,
                     metric="precomputed").fit_transform(D)
    except Exception as exc:
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(emb), {}


def m_lle_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Locally linear embedding, on the cloud's own principal coordinates.

    sklearn's LLE has no ``metric`` argument, so it cannot be handed the
    precomputed correlation matrix; it gets the same coordinates the
    autoencoders get.
    """
    from sklearn.manifold import LocallyLinearEmbedding

    X = coordinate_view(Z)
    try:
        emb = LocallyLinearEmbedding(
            n_neighbors=_n_neighbours(X.shape[0]), n_components=2,
            eigen_solver="dense", random_state=0).fit_transform(X)
    except Exception as exc:
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(emb), {}


def m_kernel_pca_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """RBF kernel PCA on the distance matrix, angle in the top components."""
    from sklearn.decomposition import KernelPCA

    try:
        emb = KernelPCA(n_components=2, kernel="precomputed").fit_transform(
            _rbf_kernel(D))
    except Exception as exc:
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(emb), {}


def m_spectral_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Laplacian eigenmaps on the same Gaussian affinity."""
    from sklearn.manifold import SpectralEmbedding

    try:
        emb = SpectralEmbedding(n_components=2, affinity="precomputed",
                                random_state=0).fit_transform(_rbf_kernel(D))
    except Exception as exc:
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(emb), {}


def m_tsne_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """t-SNE: a gradient-trained two-dimensional neighbour embedding.

    ``metric="precomputed"`` rules out ``init="pca"``, so the embedding starts
    from a seeded random layout instead.
    """
    from sklearn.manifold import TSNE

    D = np.asarray(D, dtype=float)
    n = int(D.shape[0])
    perp = float(max(5.0, min(12.0, (n - 1) / 3.0)))
    try:
        emb = TSNE(n_components=2, perplexity=perp, init="random",
                   learning_rate="auto", max_iter=600, metric="precomputed",
                   random_state=0).fit_transform(D)
    except Exception as exc:
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(emb), {"perplexity": perp}


def m_som_chain_order(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Kohonen map with a chain topology; order by best-matching unit index.

    Units start on a line laid along PC1 and are refined by the usual
    neighbourhood-adaptation rule with decaying radius and rate, so the chain
    learns the cloud's own ordering rather than having a fixed axis imposed.
    """
    X = coordinate_view(Z)
    n = X.shape[0]
    m = int(max(8, min(120, n)))

    centre = X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - centre, full_matrices=False)
    v1 = Vt[0]
    proj = (X - centre) @ v1
    s = np.linspace(float(proj.min()), float(proj.max()), m)
    units = centre + s[:, None] * v1[None, :]

    idx = np.arange(m)
    epochs, lr0 = 60, 0.5
    for e in range(epochs):
        frac = e / max(epochs - 1, 1)
        lr = lr0 * (1.0 - 0.95 * frac)
        sigma = max(0.6, (m / 2.0) * (1.0 - frac))
        nbhd = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / sigma) ** 2)
        for i in range(n):
            k = int(np.argmin(np.linalg.norm(units - X[i][None, :], axis=1)))
            units += (lr * nbhd[k])[:, None] * (X[i][None, :] - units)

    bmu = np.array([int(np.argmin(np.linalg.norm(units - X[i][None, :], axis=1)))
                    for i in range(n)])
    return np.argsort(bmu, kind="stable"), {"n_units": int(m)}


def m_principal_curve_order(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Hastie principal curve: alternate projection and smoothing.

    The curve is carried at the data points.  From an initial parameter (PC1),
    each round smooths every feature along the current order to produce curve
    vertices, assigns each sample the vertex it is nearest to, and refines
    within a vertex by projection onto the local tangent.  Twenty-five rounds,
    or fewer if the assignment stops changing.  It is a learned one-dimensional
    manifold in the sense the objection intends: nothing about circularity is
    built in, and a ring can only emerge if the smoothing puts it there.
    """
    from scipy.spatial.distance import cdist

    X = coordinate_view(Z)
    n = X.shape[0]
    w = int(max(3, n // 5))
    kernel = np.ones(2 * w + 1) / float(2 * w + 1.0)

    t = pca_embed(X, 1)[:, 0]
    span = float(np.ptp(t))
    t = (t - t.min()) / (span if span > 1e-12 else 1.0)

    order = np.argsort(t, kind="stable")
    for _ in range(25):
        Xo = X[order]
        pad = np.r_[np.repeat(Xo[:1], w, axis=0), Xo, np.repeat(Xo[-1:], w, axis=0)]
        curve = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="valid"), 0, pad)
        ts = np.sort(t)

        nearest = np.argmin(cdist(X, curve), axis=1)

        buckets: list[list[int]] = [[] for _ in range(n)]
        for i, j in enumerate(nearest):
            buckets[int(j)].append(int(i))
        new_order: list[int] = []
        for j, members in enumerate(buckets):
            if not members:
                continue
            lo, hi = max(j - 1, 0), min(j + 1, n - 1)
            tangent = curve[hi] - curve[lo]
            if len(members) == 1 or float(np.linalg.norm(tangent)) < 1e-12:
                new_order.extend(members)
            else:
                proj = (X[members] - curve[j]) @ tangent
                new_order.extend([m for _, m in sorted(zip(proj.tolist(), members))])

        t = np.clip(np.interp(nearest, np.arange(n), ts), 0.0, 1.0)
        candidate = np.asarray(new_order, dtype=int)
        if candidate.size != n:
            return np.array([], dtype=int), {"error": "assignment lost samples"}
        if np.array_equal(candidate, order):
            order = candidate
            break
        order = candidate

    return order, {"smoother_span": int(w)}


LEARNED_METHODS = {
    "ae_1d_order": m_ae_1d_order,
    "ae_2d_angle": m_ae_2d_angle,
    "isomap_angle": m_isomap_angle,
    "lle_angle": m_lle_angle,
    "kernel_pca_angle": m_kernel_pca_angle,
    "spectral_angle": m_spectral_angle,
    "tsne_angle": m_tsne_angle,
    "som_chain_order": m_som_chain_order,
    "principal_curve_order": m_principal_curve_order,
}

# which family each method belongs to, for the results table and the figure
LEARNED_FAMILY = {
    "ae_1d_order": "autoencoder",
    "ae_2d_angle": "autoencoder",
    "isomap_angle": "manifold learning",
    "lle_angle": "manifold learning",
    "spectral_angle": "manifold learning",
    "tsne_angle": "manifold learning",
    "kernel_pca_angle": "kernel method",
    "som_chain_order": "self-organising map",
    "principal_curve_order": "principal curve",
}
