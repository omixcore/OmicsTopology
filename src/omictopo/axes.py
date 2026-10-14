"""Second-axis effects and circular phase diagnostics, without independence claims.

Version 2 separates axis-associated distances, circular phase structure and
supplied topological evidence. Historical numeric outputs are in legacy_axes.
"""
from __future__ import annotations
import numpy as np
from .geometry import standardize, pairwise_distances
from .stats import mantel, circular_distance

__all__ = ["AXIS_VERDICTS", "CRITERIA", "harmonic_amplitude",
           "circular_phase_decomposition", "axis_geometry", "classify_axis", "summarise_axis"]
AXIS_VERDICTS = ("strong_axis_effect", "phase_modulation", "shape_change",
                 "no_resolved_axis_effect", "undetermined")
CRITERIA = {"r_strong": 0.5, "r_weak": 0.1, "gene_coherence": 0.8,
            "common_shift_share": 0.8, "min_phase_r2": 0.2,
            "min_phase_genes": 3, "min_shift_radians": 0.1,
            "ratio_small": 0.2}


def _phase(axis, period):
    period = 2 * np.pi if period is None else float(period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite")
    return np.mod(np.asarray(axis, float), period) * (2 * np.pi / period)


def harmonic_amplitude(y, theta):
    """Intercept-inclusive harmonic amplitude and phase in radians."""
    y, theta = np.asarray(y, float), np.asarray(theta, float)
    A = np.c_[np.ones(len(theta)), np.cos(theta), np.sin(theta)]
    if y.shape != theta.shape or not np.isfinite(y).all() or not np.isfinite(theta).all():
        raise ValueError("matching finite phase and expression vectors required")
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.hypot(coef[1], coef[2])), float(np.arctan2(coef[2], coef[1]))


def circular_phase_decomposition(X, theta, blocks, *, min_phase_r2=0.2):
    """Fit block-specific phases and decompose *circular* residual dispersion.

    Phase offsets are circular means across reliably fitted genes, not means
    of signed angles. Coherence measures gene agreement about the offset in
    each block. Weak harmonic fits are excluded before interpreting phases.
    Shares partition circular dispersion; they are not linear variance shares.
    """
    X, theta, blocks = np.asarray(X, float), np.asarray(theta, float), np.asarray(blocks)
    n, p = X.shape
    if theta.shape != (n,) or blocks.shape != (n,):
        raise ValueError("one entry per sample required for phases and blocks")
    levels = np.unique(blocks)
    phase, quality = [], []
    for lv in levels:
        m = blocks == lv
        A = np.c_[np.ones(m.sum()), np.cos(theta[m]), np.sin(theta[m])]
        if m.sum() < 6 or np.linalg.matrix_rank(A) < 3:
            return {"status": "unresolved", "reason": "at least six samples and full harmonic rank per block required",
                    "n_reliable_genes": 0, "levels_axis2": levels.tolist()}
        beta = np.linalg.lstsq(A, X[m], rcond=None)[0]
        residual = X[m] - A @ beta
        total = ((X[m] - X[m].mean(axis=0)) ** 2).sum(axis=0)
        r2 = np.divide(total - (residual ** 2).sum(axis=0), total,
                       out=np.zeros(p), where=total > 1e-12)
        phase.append(np.arctan2(beta[2], beta[1]))
        quality.append(r2)
    phase, quality = np.asarray(phase), np.asarray(quality)
    reliable = np.all(quality >= min_phase_r2, axis=0)
    if reliable.sum() < CRITERIA["min_phase_genes"]:
        return {"status": "unresolved", "reason": "too few reliably phased genes across all blocks",
                "n_reliable_genes": int(reliable.sum()), "levels_axis2": levels.tolist()}
    phase = phase[:, reliable]
    gene_mean_vectors = np.exp(1j * phase).mean(axis=0)
    # A zero resultant (e.g. evenly spread phase shifts) has no unique centre.
    # Fix one observed block as the circular reference in that case.
    centre = np.where(np.abs(gene_mean_vectors) > 1e-10,
                      np.angle(gene_mean_vectors), phase[0])
    dev = np.angle(np.exp(1j * (phase - centre)))
    vectors = np.exp(1j * dev).mean(axis=1)
    offsets = np.angle(vectors)
    residual = np.angle(np.exp(1j * (dev - offsets[:, None])))
    total_disp = float(np.mean(1 - np.cos(dev)))
    resid_disp = float(np.mean(1 - np.cos(residual)))
    share = float(np.clip(1 - resid_disp / total_disp, 0, 1)) if total_disp > 1e-12 else 0.0
    span = float(max(np.abs(np.angle(np.exp(1j * (a - b))))
                     for a in offsets for b in offsets))
    return {"status": "resolved", "n_reliable_genes": int(reliable.sum()),
            "n_genes": p, "levels_axis2": levels.tolist(),
            "block_offsets_rad": offsets.tolist(),
            "per_block_gene_coherence": np.abs(vectors).tolist(),
            "gene_offset_coherence": float(np.mean(np.abs(vectors))),
            "common_shift_share": share, "gene_specific_share": 1 - share,
            "total_circular_dispersion": total_disp,
            "residual_circular_dispersion": resid_disp,
            "max_pairwise_block_shift_rad": span,
            "median_harmonic_r2": float(np.median(quality[:, reliable])),
            "min_phase_r2": min_phase_r2,
            "interpretation": "descriptive circular phase agreement in reliably phased genes"}


def _matched(D, theta, labels, rng, n_boot, units=None):
    n = len(D)
    def contrast(idx):
        sub = D[np.ix_(idx, idx)]
        i, j = np.triu_indices(len(idx), 1)
        same = np.abs(np.angle(np.exp(1j * (theta[idx[i]] - theta[idx[j]])))) < 1e-8
        different = labels[idx[i]] != labels[idx[j]]
        matched, other = same & different, ~same
        if not matched.any() or not other.any():
            return None
        a, b = float(sub[i[matched], j[matched]].mean()), float(sub[i[other], j[other]].mean())
        return a, b, b - a, a / b if b > 1e-12 else None
    observed = contrast(np.arange(n))
    if observed is None:
        return {"separation_ratio": None, "absolute_difference": None,
                "absolute_difference_ci95": [None, None], "ratio_ci95": [None, None],
                "reason": "no matched phase pairs with different second-axis labels"}
    draws = []
    if units is not None:
        units = np.asarray(units)
        if units.shape != (n,):
            raise ValueError("units must have one entry per sample")
        levels = np.unique(units)
    else:
        levels = np.arange(n)
    if len(levels) >= 3:
        for _ in range(n_boot):
            chosen = rng.choice(levels, len(levels), replace=True)
            idx = np.concatenate([np.flatnonzero(units == lv) for lv in chosen]) if units is not None else chosen
            result = contrast(idx)
            if result is not None and result[3] is not None:
                draws.append(result[2:])
    valid = len(draws) >= 20
    interval = np.quantile(draws, [0.025, 0.975], axis=0) if valid else np.full((2, 2), np.nan)
    return {"mean_distance_matched": observed[0], "mean_distance_across_axis1": observed[1],
            "absolute_difference": observed[2], "separation_ratio": observed[3],
            "absolute_difference_ci95": interval[:, 0].tolist() if valid else [None, None],
            "ratio_ci95": interval[:, 1].tolist() if valid else [None, None],
            "bootstrap_unit": "whole_unit" if units is not None else "rows",
            "n_valid_bootstraps": len(draws),
            "interpretation": "row bootstrap is descriptive if samples are temporally dependent"}


def _phase_uncertainty(X, theta, labels, base, rng, n_boot, units):
    """Bootstrap circular offsets by whole units or within phase-fit blocks.

    Intervals on offsets use wrapped deviations from the original offset;
    endpoints may exceed pi and must be interpreted modulo 2pi. Raw-row
    intervals require exchangeable biological observations to be inferential.
    """
    if base.get("status") != "resolved":
        return {"status": "unresolved", "n_valid_bootstraps": 0}
    levels = np.unique(labels)
    offsets, shares = [], []
    unit_levels = np.unique(units) if units is not None else None
    if unit_levels is not None and len(unit_levels) < 3:
        return {"status": "unresolved", "reason": "fewer than three independent units",
                "n_valid_bootstraps": 0}
    for _ in range(n_boot):
        if units is None:
            idx = np.concatenate([rng.choice(np.flatnonzero(labels == lv),
                 np.count_nonzero(labels == lv), replace=True) for lv in levels])
        else:
            chosen = rng.choice(unit_levels, len(unit_levels), replace=True)
            idx = np.concatenate([np.flatnonzero(units == lv) for lv in chosen])
        rec = circular_phase_decomposition(X[idx], theta[idx], labels[idx])
        if rec.get("status") != "resolved" or rec["levels_axis2"] != levels.tolist():
            continue
        offsets.append(rec["block_offsets_rad"])
        shares.append(rec["common_shift_share"])
    if len(offsets) < 20:
        return {"status": "unresolved", "n_valid_bootstraps": len(offsets)}
    origin = np.asarray(base["block_offsets_rad"])
    delta = np.angle(np.exp(1j * (np.asarray(offsets) - origin)))
    interval = np.quantile(delta, [.025, .975], axis=0) + origin
    return {"status": "resolved", "block_offset_ci95_rad": interval.T.tolist(),
            "common_shift_share_ci95": np.quantile(shares, [.025, .975]).tolist(),
            "n_valid_bootstraps": len(offsets),
            "sampling_unit": "whole_unit" if units is not None else "within_axis2_block_rows",
            "interpretation": "conditional fitted-phase bootstrap; descriptive for dependent time series"}


def classify_axis(dominant, separable, phase_or_shape, matched, criteria=None):
    """Assign descriptive labels, never an independent-coordinate verdict."""
    c = CRITERIA if criteria is None else criteria
    p = phase_or_shape
    if p.get("status") == "resolved":
        if (p["common_shift_share"] >= c["common_shift_share"]
                and p["gene_offset_coherence"] >= c["gene_coherence"]
                and p["max_pairwise_block_shift_rad"] >= c["min_shift_radians"]):
            return "phase_modulation", "reliable genes agree on a nonzero circular phase shift"
        if p["gene_specific_share"] >= 0.5 and p["total_circular_dispersion"] >= 0.01:
            return "shape_change", "phase differences are predominantly gene specific"
    r = dominant.get("mantel_r_axis2")
    ratio = matched.get("separation_ratio")
    if r is not None and r >= c["r_strong"]:
        return "strong_axis_effect", "axis 2 is associated with distances; no independent cycle is established"
    if r is not None and abs(r) < c["r_weak"] and ratio is not None and ratio < c["ratio_small"]:
        return "no_resolved_axis_effect", "small association and small matched distance under this analysis"
    return "undetermined", "the descriptive quantities do not identify the type of effect"


def axis_geometry(X, axis1, axis2, groups=None, *, period1=None, period2=None,
                  axis2_kind="categorical", seed=0, n_perm=2000, n_boot=2000,
                  n_perm_sep=1000, units=None, permutation="none",
                  permutation_strata=None, cycle_result=None):
    """Separate distance effects, phase structure and optional H1 evidence.

    A categorical axis uses equal distance between different labels, so its
    result does not depend on arbitrary numeric coding. Cyclic axes use the
    supplied period. No p-values are produced by default for dependent series.
    ``groups`` labels the blocks used for phase fits; it is not a bootstrap unit.
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite matrix")
    n = len(X)
    axis1, axis2 = np.asarray(axis1), np.asarray(axis2)
    if axis1.shape != (n,) or axis2.shape != (n,):
        raise ValueError("each axis must have one entry per sample")
    if axis2_kind not in ("categorical", "cyclic"):
        raise ValueError("axis2_kind must be categorical or cyclic")
    theta = _phase(axis1, period1)
    if not np.isfinite(theta).all():
        raise ValueError("axis1 must be finite")
    labels = axis2 if groups is None else np.asarray(groups)
    if labels.shape != (n,):
        raise ValueError("groups must have one entry per sample")
    D = pairwise_distances(standardize(X), metric="correlation")
    ref = circular_distance(theta)
    second = (axis2[:, None] != axis2[None, :]).astype(float) if axis2_kind == "categorical" else circular_distance(_phase(axis2, period2))
    r1, p1 = mantel(D, ref, n_perm=n_perm, seed=seed,
                    permutation=permutation, strata=permutation_strata)
    r2, p2 = mantel(D, second, n_perm=n_perm, seed=seed,
                    permutation=permutation, strata=permutation_strata)
    phase = circular_phase_decomposition(X, theta, labels,
                                        min_phase_r2=CRITERIA["min_phase_r2"])
    matched = _matched(D, theta, labels, np.random.default_rng(seed), int(n_boot), units)
    phase["uncertainty"] = _phase_uncertainty(X, theta, labels, phase,
        np.random.default_rng(seed + 1), min(int(n_boot), 200),
        np.asarray(units) if units is not None else None)
    A = np.c_[np.ones(n), np.cos(theta), np.sin(theta)]
    levels, codes = np.unique(labels, return_inverse=True)
    blocks = np.eye(len(levels))[codes]
    designs = [A, blocks, np.c_[A, blocks[:, 1:]]]
    L = X - X.mean(axis=0)
    total = float((L ** 2).sum())
    shares = []
    for V in designs:
        residual = L - V @ np.linalg.lstsq(V, L, rcond=None)[0]
        shares.append(1 - float((residual ** 2).sum()) / total if total else 0.0)
    separable = {"variance_explained_by_axis1": shares[0],
                 "variance_explained_by_axis2": shares[1],
                 "variance_explained_by_both": shares[2],
                 "levels_axis2": levels.tolist(),
                 "interpretation": "descriptive projection shares, not independence tests"}
    dominant = {"mantel_r_axis1": r1, "mantel_p_axis1": p1,
                "mantel_r_axis2": r2, "mantel_p_axis2": p2,
                "axis2_kind": axis2_kind, "permutation": permutation}
    verdict, reason = classify_axis(dominant, separable, phase, matched)
    topo = {"status": "not_tested", "independent_second_cycle": "not_established"}
    if cycle_result is not None:
        topo.update({"status": cycle_result.get("status"),
                     "detected_h1_features": cycle_result.get("n_cycles"),
                     "interpretation": "even two stable H1 bars do not assign a particular bar to axis 2"})
    return {"protocol_version": "2", "n_samples": n, "n_genes": X.shape[1],
            "dominant": dominant, "separable": separable, "phase_or_shape": phase,
            "matched": matched, "verdict": verdict, "reason": reason,
            "axis_effect": "strong_association" if r2 >= CRITERIA["r_strong"] else "weak_or_undetermined",
            "cyclic_evidence": topo, "criteria": dict(CRITERIA),
            "settings": {"seed": seed, "n_perm": n_perm, "n_boot": n_boot,
                         "axis2_kind": axis2_kind, "permutation": permutation,
                         "bootstrap_unit": "whole_unit" if units is not None else "rows"}}


def summarise_axis(rec):
    """Brief descriptive label and the separate cycle-evidence limit."""
    return f"{rec['verdict']} (axis 2 r={rec['dominant']['mantel_r_axis2']:+.3f}; second cycle not established)"
