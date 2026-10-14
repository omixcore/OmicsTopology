"""Surrogate-calibrated H1 detection with explicit resampling stability.

Counts describe persistent features under a specified distance and Gaussian
reference model. They are not estimates of biological oscillator number.
"""
from __future__ import annotations

import importlib.util
import math
import numpy as np

from .geometry import standardize, pairwise_distances
from .homology import persistent_homology
from .stats import wilson_interval

__all__ = ["spectral_count", "stable_cycle_count", "signal_spectrum"]


def _matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 6 or X.shape[1] < 2:
        raise ValueError("X requires at least six samples and two features")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite; impute missing values explicitly")
    varying = np.std(X, axis=0) > 1e-12
    if varying.sum() < 2:
        raise ValueError("at least two varying features are required")
    return X[:, varying]


def _backend(backend):
    if backend not in ("ripser", "builtin", "auto"):
        raise ValueError("backend must be ripser, builtin or auto")
    available = importlib.util.find_spec("ripser") is not None
    if backend == "ripser" and not available:
        raise ImportError("install omictopo[fast] for the pinned ripser backend")
    return "ripser" if backend == "auto" and available else (
        "builtin" if backend == "auto" else backend)


def _bars(D, backend):
    scale = float(np.sqrt(np.mean(D ** 2) / 2.0))
    if scale <= 1e-12:
        return np.empty(0)
    if backend == "ripser":
        # Call the requested backend directly: never change algorithms silently.
        from ripser import ripser
        bars = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    else:
        diagram = persistent_homology(D=D, max_dim=1, backend="builtin")
        if diagram.meta.get("truncated"):
            raise ValueError("builtin filtration truncated; use backend='ripser'")
        bars = diagram.by_dim(1)
    if not len(bars):
        return np.empty(0)
    if not np.isfinite(bars).all():
        raise ValueError("incomplete H1 filtration cannot calibrate a count")
    return np.sort((bars[:, 1] - bars[:, 0]) / scale)[::-1]


def signal_spectrum(lengths, null_maxima, alpha=0.05):
    """Compare *each* H1 length with one maximum-statistic surrogate null.

The Monte Carlo tail is (1 + exceedances)/(1 + draws). The same maximum
null is used for every observed bar; comparable signal bars are not penalised
by dividing one signal bar by another. Error control is conditional on the
adequacy of the fitted Gaussian surrogate, not a distribution-free guarantee.
"""
    lengths = np.asarray(lengths, float)
    maxima = np.asarray(null_maxima, float)
    if lengths.ndim != 1 or maxima.ndim != 1 or maxima.size < 1:
        raise ValueError("lengths and nonempty null_maxima must be vectors")
    if not np.isfinite(lengths).all() or not np.isfinite(maxima).all():
        raise ValueError("spectra must be finite")
    if (lengths < 0).any() or (maxima < 0).any() or not 0 < alpha < 1:
        raise ValueError("nonnegative spectra and 0 < alpha < 1 required")
    p = (1 + (maxima[:, None] >= lengths[None, :]).sum(axis=0)) / (1 + maxima.size)
    selected = p <= alpha
    return {"candidate_count": int(selected.sum()),
            "bar_pvalues_max_null": p.tolist(),
            "selected": selected.tolist(),
            "null_threshold": float(np.quantile(maxima, 1 - alpha)),
            "minimum_p": 1.0 / (maxima.size + 1)}


def spectral_count(X, *, metric="correlation", n_null=199, alpha=0.05,
                   seed=0, backend="ripser"):
    """Count H1 bars against Gaussian draws reconstructed in feature space.

SVD loadings are retained, so both observed and surrogate matrices pass
through the same feature standardisation and distance calculation. Principal
scores alone are not substituted for features when using correlation distance.
"""
    X = _matrix(X)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if int(n_null) != n_null or n_null < math.ceil(1 / alpha) - 1:
        raise ValueError("n_null is too small for the requested alpha")
    resolved = _backend(backend)
    Z = standardize(X)
    _, singular, loadings = np.linalg.svd(Z, full_matrices=False)
    keep = singular > max(Z.shape) * np.finfo(float).eps * singular[0]
    sd = singular[keep] / np.sqrt(X.shape[0] - 1)
    basis = loadings[keep]
    D = pairwise_distances(Z, metric=metric)
    observed = _bars(D, resolved)
    rng = np.random.default_rng(seed)
    maxima = np.empty(int(n_null))
    for i in range(int(n_null)):
        surrogate = (rng.normal(size=(X.shape[0], len(sd))) * sd) @ basis
        bars = _bars(pairwise_distances(standardize(surrogate), metric=metric), resolved)
        maxima[i] = bars[0] if bars.size else 0.0
    rec = signal_spectrum(observed, maxima, alpha)
    rec.update({"n_samples": X.shape[0], "n_features": X.shape[1],
                "bar_lengths": observed.tolist(), "null_maxima": maxima.tolist(),
                "null_model": "feature_covariance_gaussian_restandardised",
                "settings": {"metric": metric, "n_null": int(n_null),
                             "alpha": alpha, "seed": int(seed),
                             "backend_requested": backend, "backend_used": resolved}})
    return rec


def _subsample_indices(n, rng, fraction, units=None, strata=None):
    if units is not None and strata is not None:
        raise ValueError("use independent units OR strata, not both")
    if units is not None:
        u = np.asarray(units)
        if u.shape != (n,):
            raise ValueError("units requires one label per sample")
        levels = np.unique(u)
        if len(levels) < 3:
            return None
        k = min(len(levels) - 1, max(2, int(np.floor(fraction * len(levels)))))
        return np.flatnonzero(np.isin(u, rng.choice(levels, k, replace=False)))
    if strata is None:
        return np.sort(rng.choice(n, max(6, int(np.floor(fraction * n))), replace=False))
    s = np.asarray(strata)
    if s.shape != (n,):
        raise ValueError("strata requires one label per sample")
    picks = []
    for level in np.unique(s):
        idx = np.flatnonzero(s == level)
        k = max(1, int(np.floor(len(idx) * fraction)))
        picks.extend(rng.choice(idx, k, replace=False).tolist())
    picks = np.sort(picks)
    return picks if len(picks) >= 6 and len(picks) < n else None


def stable_cycle_count(X, *, metric="correlation", n_null=199, alpha=0.05,
                       seed=0, backend="ripser", n_resamples=12,
                       resample_fraction=0.9, stability_threshold=0.75,
                       resample_n_null=99, units=None, strata=None):
    """Return a stable detected count or abstain, without a dominant-bar gate.

Subsampling estimates robustness of a detection, not posterior confidence in
the true Betti number. Every reduced cloud refits its own surrogate null.
Whole independent units may be retained together. With no units supplied,
row subsampling is a geometric sensitivity check, not biological inference.
"""
    X = _matrix(X)
    if not 0 < resample_fraction < 1 or not 0 < stability_threshold <= 1:
        raise ValueError("invalid resampling fraction or stability threshold")
    if int(n_resamples) != n_resamples or n_resamples < 0:
        raise ValueError("n_resamples must be a nonnegative integer")
    if units is not None and strata is not None:
        raise ValueError("use independent units OR strata, not both")
    children = np.random.SeedSequence(seed).spawn(int(n_resamples) + 1)
    seed_for = lambda child: int(child.generate_state(1)[0])
    rec = spectral_count(X, metric=metric, n_null=n_null, alpha=alpha,
                         seed=seed_for(children[0]), backend=backend)
    counts = []
    for child in children[1:]:
        sub_seed = seed_for(child)
        idx = _subsample_indices(len(X), np.random.default_rng(sub_seed),
                                 resample_fraction, units, strata)
        if idx is None or len(idx) < 6:
            continue
        sub = spectral_count(X[idx], metric=metric, n_null=resample_n_null,
                             alpha=alpha, seed=sub_seed, backend=backend)
        counts.append(sub["candidate_count"])
    k = rec["candidate_count"]
    successes = sum(v == k for v in counts)
    rate = successes / len(counts) if counts else None
    stable = rate is not None and rate >= stability_threshold
    rec.update({"protocol_version": "2", "n_cycles": k if stable else None,
                "status": ("counted" if k else "no_cycle") if stable else "unresolved",
                "count_interpretation": "detected H1 features, not biological oscillator number",
                "stability": {"resampled_counts": counts, "agreement": rate,
                              "agreement_ci95": list(wilson_interval(successes, len(counts)))
                                  if counts else [None, None],
                              "n_completed": len(counts),
                              "sampling_unit": "whole_unit" if units is not None else
                                  "stratified_rows" if strata is not None else "rows",
                              "interpretation": "subsampling robustness, not count coverage"}})
    rec["settings"].update({"seed": int(seed), "n_resamples": int(n_resamples),
                             "resample_fraction": resample_fraction,
                             "resample_n_null": int(resample_n_null),
                             "stability_threshold": stability_threshold})
    return rec
