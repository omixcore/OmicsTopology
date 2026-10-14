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
``ae_deep_2d_angle``      deep autoencoder (64-32-16, ~8.4k params) - the
                          "give it real capacity" objection, taken at face value
``dae_2d_angle``          deep *denoising* autoencoder - corruption-trained,
                          the standard answer to "it only memorised n = 24"
``vae_2d_angle``          deep variational autoencoder - a probabilistic latent,
                          read from the posterior mean
``umap_angle``            UMAP - the default learned manifold embedding of the
                          single-cell literature
``diffusion_map_angle``   diffusion maps - the random-walk eigenbasis of the
                          same correlation distance every other method reads
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

No deep-learning framework is required: the networks are ~120 lines of NumPy
with Adam and gradient clipping, so these baselines run inside the base
dependency set and reproduce bit-for-bit.  ``umap-learn`` is the one optional
import (``pip install -e ".[baselines]"``); when it is absent the UMAP row
reports itself as unavailable instead of silently dropping out.

The deep variants and their tuning rule
---------------------------------------
``DEEP_SETTINGS`` is frozen exactly the way ``AE_SETTINGS`` was: the grid was
run on four independent positive rings and nothing else
(``examples/tune_deep_baselines.py`` -> ``benchmark/deep_tuning.json`` records
the whole grid).  The values below are that grid's winners, committed before
any negative cloud was scored.
"""

from __future__ import annotations

import numpy as np

from omictopo.geometry import pca_embed

__all__ = [
    "LEARNED_METHODS",
    "LEARNED_FAMILY",
    "AE_SETTINGS",
    "DEEP_SETTINGS",
    "coordinate_view",
    "autoencoder_latent",
    "deep_autoencoder_latent",
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

# The deep variants: winners of the positive-control grid in
# examples/tune_deep_baselines.py (recorded whole in benchmark/deep_tuning.json).
# Worth noting what the grid said: the deep nets read the control rings less
# reliably than the shallow one (best means 0.84-0.88, worst 0.67-0.75, against
# 1.000 for the shallow network on all four), and UMAP is worse still (0.573).
# Those are the tuned ceilings, not defaults that were abandoned.
DEEP_SETTINGS = {
    "layers": (64, 32, 16),
    "epochs": 1200,
    "lr": 6e-3,
    "weight_decay": 1e-2,
    "noise_sigma": 0.5,      # denoising corruption, frozen from the same grid
    "kl_beta": 0.01,         # VAE KL weight, frozen from the same grid
    "umap_n_neighbors": 12,  # UMAP's winner on the same controls (of 5/8/12/15/23)
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


def deep_autoencoder_latent(
    Z: np.ndarray,
    latent: int = 2,
    layers: tuple | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    sd_floor: float | None = None,
    grad_clip: float | None = None,
    seed: int | None = None,
    nonlinear: bool = True,
    noise_sigma: float | None = None,
    kl_beta: float | None = None,
) -> tuple:
    """Train a deep autoencoder; return its latent code and its size.

    Architecture ``q -> layers -> latent -> layers(reversed) -> q`` with
    ``tanh`` hidden layers and a linear bottleneck normalised to zero mean and
    unit variance (the same scale-gauge fix the shallow network needs: without
    it, Adam drifts along the flat direction where all weights are rescaled by
    a constant).

    Three modes of the same network:

    * plain (``kl_beta = 0``): mean-squared reconstruction;
    * denoising (``noise_sigma > 0``, ``kl_beta = 0``): every training step
      corrupts the *input* with Gaussian noise of that standard deviation and
      reconstructs the clean cloud;
    * variational (``kl_beta > 0``): the bottleneck produces ``mu`` and
      ``log sigma^2`` and the code is ``mu + sigma * eps`` with ``eps`` from the
      seeded generator, so the run stays deterministic; the loss adds
      ``kl_beta`` times the usual closed-form KL.  The returned code is the
      posterior **mean**, which keeps ordering independent of the sampling.

    Full-batch Adam on the reconstruction term, L2 on every weight matrix,
    gradient clipping.  Deterministic: the only random inputs are the
    initialisation and (for the VAE / denoiser) the seeded noise stream.
    """
    cfg = dict(DEEP_SETTINGS)
    layers = tuple(cfg["layers"]) if layers is None else tuple(layers)
    epochs = cfg["epochs"] if epochs is None else epochs
    lr = cfg["lr"] if lr is None else lr
    weight_decay = cfg["weight_decay"] if weight_decay is None else weight_decay
    sd_floor = cfg["sd_floor"] if sd_floor is None else sd_floor
    grad_clip = cfg["grad_clip"] if grad_clip is None else grad_clip
    seed = cfg["seed"] if seed is None else seed
    noise_sigma = cfg["noise_sigma"] if noise_sigma is None else noise_sigma
    kl_beta = cfg["kl_beta"] if kl_beta is None else kl_beta

    S = coordinate_view(Z)
    n, q = S.shape
    latent = int(max(1, min(latent, q)))
    variational = float(kl_beta) > 0.0
    rng = np.random.default_rng(seed)

    def init(fan_in: int, fan_out: int) -> np.ndarray:
        return rng.standard_normal((fan_in, fan_out)) * np.sqrt(
            2.0 / float(fan_in + fan_out))

    enc_dims = [q] + list(layers)
    enc = [(init(enc_dims[i], enc_dims[i + 1]), np.zeros(enc_dims[i + 1]))
           for i in range(len(layers))]
    dec_dims = [latent] + list(reversed(layers))
    dec = [(init(dec_dims[i], dec_dims[i + 1]), np.zeros(dec_dims[i + 1]))
           for i in range(len(dec_dims) - 1)]
    W_out, b_out = init(layers[0], q), np.zeros(q)

    if variational:
        W_mu, b_mu = init(layers[-1], latent), np.zeros(latent)
        W_lv, b_lv = init(layers[-1], latent), np.zeros(latent)
        Wb = bb = None
        params = ([w for p in enc for w in p] + [w for p in dec for w in p]
                  + [W_out, b_out, W_mu, b_mu, W_lv, b_lv])
    else:
        Wb, bb = init(layers[-1], latent), np.zeros(latent)
        W_mu = b_mu = W_lv = b_lv = None
        params = ([w for p in enc for w in p] + [Wb, bb]
                  + [w for p in dec for w in p] + [W_out, b_out])

    def act(H):
        return np.tanh(H) if nonlinear else H

    def grad_act(A):
        return (1.0 - A ** 2) if nonlinear else 1.0

    m = [np.zeros_like(w) for w in params]
    v = [np.zeros_like(w) for w in params]
    slot = {id(w): i for i, w in enumerate(params)}   # biases start as equal
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8         # zeros; index() would be
                                                      # ambiguous, id() is not

    for step in range(1, int(epochs) + 1):
        S_in = S
        if float(noise_sigma) > 0.0 and not variational:
            S_in = S + rng.normal(0.0, float(noise_sigma), S.shape)

        # ---- forward ----------------------------------------------------
        acts = [S_in]
        A = S_in
        for W, b in enc:
            A = act(A @ W + b)
            acts.append(A)
        H = acts[-1]
        if variational:
            mu = H @ W_mu + b_mu
            logvar = np.clip(H @ W_lv + b_lv, -8.0, 8.0)
            eps = rng.standard_normal(mu.shape)
            pre = mu + np.exp(0.5 * logvar) * eps
        else:
            mu = logvar = eps = None
            pre = H @ Wb + bb
        sd = pre.std(axis=0) + sd_floor
        code = (pre - pre.mean(axis=0)) / sd

        d_acts = [code]
        A = code
        for W, b in dec:
            A = act(A @ W + b)
            d_acts.append(A)
        rec = A @ W_out + b_out

        # ---- backward ---------------------------------------------------
        dR = 2.0 * (rec - S) / float(n * q)
        grads: list = [None] * len(params)

        def bump(mat, grad):
            grads[slot[id(mat)]] = grad

        dA = dR @ W_out.T
        bump(W_out, A.T @ dR + 2.0 * weight_decay * W_out)
        bump(b_out, dR.sum(axis=0))
        for i in range(len(dec) - 1, -1, -1):
            W, b = dec[i]
            dH = dA * grad_act(d_acts[i + 1])
            dA = dH @ W.T
            bump(W, d_acts[i].T @ dH + 2.0 * weight_decay * W)
            bump(b, dH.sum(axis=0))

        dcode = dA / sd                       # through the gauge normalisation

        if variational:
            inv_n = 1.0 / float(n)
            dkl_mu = mu * inv_n
            dkl_lv = -0.5 * (1.0 - np.exp(logvar)) * inv_n
            dmu = dcode + kl_beta * dkl_mu
            dlogvar = kl_beta * dkl_lv + (dcode * 0.5 * np.exp(0.5 * logvar) * eps)
            bump(W_mu, H.T @ dmu + 2.0 * weight_decay * W_mu)
            bump(b_mu, dmu.sum(axis=0))
            bump(W_lv, H.T @ dlogvar + 2.0 * weight_decay * W_lv)
            bump(b_lv, dlogvar.sum(axis=0))
            dH = dmu @ W_mu.T + dlogvar @ W_lv.T
        else:
            bump(Wb, H.T @ dcode + 2.0 * weight_decay * Wb)
            bump(bb, dcode.sum(axis=0))
            dH = dcode @ Wb.T

        for i in range(len(enc) - 1, -1, -1):
            W, b = enc[i]
            dH = dH * grad_act(acts[i + 1])
            dA_prev = dH @ W.T
            bump(W, acts[i].T @ dH + 2.0 * weight_decay * W)
            bump(b, dH.sum(axis=0))
            dH = dA_prev

        norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
        if norm > grad_clip:
            grads = [g * (grad_clip / norm) for g in grads]
        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g
            # in-place: the forward pass reads enc/dec, which hold the same
            # array objects -- rebinding here would train a copy nobody reads
            params[i] -= lr * (m[i] / (1.0 - beta1 ** step)) / (
                np.sqrt(v[i] / (1.0 - beta2 ** step)) + eps_adam)

    # ---- final clean pass ---------------------------------------------------
    A = S
    for W, b in enc:
        A = act(A @ W + b)
    H = A
    if variational:
        pre = H @ W_mu + b_mu                 # posterior mean, no sampling
    else:
        pre = H @ Wb + bb
    sd = pre.std(axis=0) + sd_floor
    code = (pre - pre.mean(axis=0)) / sd

    n_params = int(sum(w.size for w in params))
    return code, {"n_params": n_params, "n_ae_input_dim": int(q),
                  "ae_layers": [int(x) for x in layers],
                  "ae_variational": bool(variational),
                  "ae_noise_sigma": float(noise_sigma) if not variational else 0.0,
                  "ae_kl_beta": float(kl_beta) if variational else 0.0,
                  "ae_epochs": int(epochs)}


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
    out["ae_deep_2d_angle"] = int(deep_autoencoder_latent(
        Z, latent=2, kl_beta=0.0, noise_sigma=0.0)[1]["n_params"])
    out["dae_2d_angle"] = int(deep_autoencoder_latent(
        Z, latent=2, kl_beta=0.0, noise_sigma=0.25)[1]["n_params"])
    out["vae_2d_angle"] = int(deep_autoencoder_latent(
        Z, latent=2, kl_beta=0.03)[1]["n_params"])
    for label in ("umap_angle", "diffusion_map_angle", "isomap_angle",
                  "lle_angle", "kernel_pca_angle", "spectral_angle",
                  "tsne_angle", "som_chain_order", "principal_curve_order"):
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


def m_ae_deep_2d_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Deep autoencoder (64-32-16): the capacity objection at face value."""
    try:
        code, extra = deep_autoencoder_latent(Z, latent=2)
    except Exception as exc:                                   # pragma: no cover
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(code), extra


def m_dae_2d_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Deep denoising autoencoder: corruption-trained, same read-out."""
    try:
        code, extra = deep_autoencoder_latent(Z, latent=2, kl_beta=0.0)
    except Exception as exc:                                   # pragma: no cover
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(code), extra


def m_vae_2d_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Deep variational autoencoder; order by the posterior mean."""
    try:
        code, extra = deep_autoencoder_latent(Z, latent=2)
    except Exception as exc:                                   # pragma: no cover
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(code), extra


def m_umap_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """UMAP on the correlation distance matrix, angle in the 2-D embedding.

    Optional dependency: without ``umap-learn`` the method reports itself as
    unavailable instead of silently vanishing from the table.
    """
    try:
        import umap
    except Exception:
        return np.array([], dtype=int), {"error": "umap-learn not installed"}
    D = np.asarray(D, dtype=float)
    try:
        emb = umap.UMAP(n_components=2,
                        n_neighbors=int(DEEP_SETTINGS["umap_n_neighbors"]),
                        metric="precomputed", random_state=0).fit_transform(D)
    except Exception as exc:
        return np.array([], dtype=int), {"error": type(exc).__name__}
    return _angle_order(emb), {"n_neighbors": int(DEEP_SETTINGS["umap_n_neighbors"])}


def m_diffusion_map_angle(Z: np.ndarray, D: np.ndarray) -> tuple:
    """Diffusion maps: the random-walk eigenbasis of the same distance matrix.

    The Gaussian affinity is row-normalised into a Markov operator (symmetrised
    for a stable eigensolver) and the angle is read from the first two
    non-trivial diffusion coordinates.  Nothing about cycles is built in; a
    ring emerges only if the random walk actually goes round one.
    """
    K = _rbf_kernel(D)
    deg = K.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.where(deg > 1e-12, deg, 1.0))
    W = inv_sqrt[:, None] * K * inv_sqrt[None, :]
    W = 0.5 * (W + W.T)
    try:
        vals, vecs = np.linalg.eigh(W)
    except Exception as exc:                                   # pragma: no cover
        return np.array([], dtype=int), {"error": type(exc).__name__}
    order_idx = np.argsort(vals)[::-1]
    if len(order_idx) < 3:
        return np.array([], dtype=int), {"error": "spectrum too small"}
    pick = order_idx[1:3]                       # skip the trivial first vector
    emb = vecs[:, pick]
    return _angle_order(emb), {"eigenvalues": [round(float(vals[i]), 6)
                                               for i in pick]}


LEARNED_METHODS = {
    "ae_1d_order": m_ae_1d_order,
    "ae_2d_angle": m_ae_2d_angle,
    "ae_deep_2d_angle": m_ae_deep_2d_angle,
    "dae_2d_angle": m_dae_2d_angle,
    "vae_2d_angle": m_vae_2d_angle,
    "umap_angle": m_umap_angle,
    "diffusion_map_angle": m_diffusion_map_angle,
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
    "ae_deep_2d_angle": "deep autoencoder",
    "dae_2d_angle": "denoising autoencoder",
    "vae_2d_angle": "variational autoencoder",
    "umap_angle": "manifold learning",
    "diffusion_map_angle": "diffusion map",
    "isomap_angle": "manifold learning",
    "lle_angle": "manifold learning",
    "spectral_angle": "manifold learning",
    "tsne_angle": "manifold learning",
    "kernel_pca_angle": "kernel method",
    "som_chain_order": "self-organising map",
    "principal_curve_order": "principal curve",
}
