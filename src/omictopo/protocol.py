"""Version 2 H1 inference: surrogate detection, stability and design association.

The topology count is never gated by the ratio of two signal bars or by a
Mantel effect-size threshold. Design association is reported separately.
Historical version-1 calls are available from omictopo.legacy_protocol.
"""
from __future__ import annotations
import numpy as np
from .inference import stable_cycle_count
from .geometry import standardize, pairwise_distances
from .stats import circular_distance, mantel

__all__ = ["STATUSES", "analyse", "envelope", "summarise"]
STATUSES = ("counted", "no_cycle", "unresolved")


def analyse(X, axis=None, *, period=None, n_null=199, n_perm=2000,
            seed=0, metric="correlation", alpha=0.05, backend="ripser",
            n_resamples=12, resample_fraction=0.9, stability_threshold=0.75,
            resample_n_null=99, units=None, strata=None,
            permutation="none", permutation_strata=None):
    """Detect robust H1 features and separately describe design alignment.

    ``n_cycles`` is None on abstention. ``candidate_count`` is always retained.
    Design p-values are omitted by default: enable sample permutation only
    where exchangeability is justified, optionally within declared strata.
    ``units`` governs topology subsampling, not the Mantel permutation scheme.
    """
    X = np.asarray(X, float)
    rec = stable_cycle_count(X, metric=metric, n_null=n_null, alpha=alpha,
        seed=seed, backend=backend, n_resamples=n_resamples,
        resample_fraction=resample_fraction, stability_threshold=stability_threshold,
        resample_n_null=resample_n_null, units=units, strata=strata)
    rec["design_association"] = None
    if axis is not None:
        axis = np.asarray(axis, float)
        if axis.shape != (len(X),) or not np.isfinite(axis).all():
            raise ValueError("axis must have one finite entry per sample")
        p = 2 * np.pi if period is None else float(period)
        D = pairwise_distances(standardize(X), metric=metric)
        r, pv = mantel(D, circular_distance(axis, p), n_perm=n_perm,
                       seed=seed, permutation=permutation, strata=permutation_strata)
        rec["design_association"] = {"mantel_r": r, "mantel_p": pv,
            "period": p, "permutation": permutation,
            "restricted": permutation_strata is not None,
            "interpretation": "distance association, not a test of independent cyclic coordinates"}
    rec["settings"].update({"n_perm": n_perm, "permutation": permutation})
    return rec


def envelope():
    """Declare current assumptions; no universal sample-size minimum is asserted."""
    return {"protocol_version": "2", "requires": [
        "finite samples-by-features data and an explicitly specified distance",
        "adequate cycle coverage under the observation map",
        "an adequate feature-covariance Gaussian reference model",
        "an explicitly pinned, untruncated persistence backend",
        "a declared resampling unit and sufficient resampling agreement"],
        "cannot_do": ["equate H1 features with biological oscillators",
                      "prove absence of a weak loop from a zero detection",
                      "establish a torus from two H1 bars alone",
                      "establish axis independence from a Mantel effect"],
        "not_label_free": ["design association and stratification use supplied labels"],
        "validation": "see benchmark/v2; historical records are not v2 validation"}


def summarise(rec):
    """Brief count and uncertainty statement with no integer on abstention."""
    if rec["status"] == "unresolved":
        return "unresolved H1 count; examine candidate spectrum and resampling"
    if rec["status"] == "no_cycle":
        return "0 H1 features detected under the specified reference model"
    return f"counted {rec['n_cycles']} persistent H1 feature(s)"
