"""One-call workflow: from an expression matrix to a topology verdict.

    from omictopo import run_states

    report = run_states(matrix, groups, labels=["control", "salt_stress"])
    print(report.summary_text())
    report.write("results/")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .compare import compare_diagrams
from .diagram import PersistenceDiagram
from .filtration import choose_threshold
from .geometry import pairwise_distances, pca_embed, standardize, subsample
from .homology import persistent_homology
from .io import write_table
from .stats import permutation_test_pointclouds
from .summary import topological_summary

__all__ = ["run_states", "StateTopologyReport"]


@dataclass
class StateTopologyReport:
    """Container for everything a state comparison produced.

    Attributes
    ----------
    states : dict
        ``{state_name: PersistenceDiagram}``
    clouds : dict
        ``{state_name: embedded point cloud}``
    summaries : dict
        ``{state_name: summary statistics}``
    comparison : dict
        Pairwise diagram distances, keyed ``"a|b"``.
    test : dict or None
        Permutation test result when exactly two states were given.
    diagnostics : dict
        Thresholds, embedding parameters, group sizes.
    files : list
        Paths written by :meth:`write`.
    """

    states: dict = field(default_factory=dict)
    clouds: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    test: dict | None = None
    diagnostics: dict = field(default_factory=dict)
    files: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def state_names(self) -> list:
        return list(self.states.keys())

    def summary_rows(self) -> list:
        rows = []
        for name, stats in self.summaries.items():
            row = {"state": name}
            row.update(stats)
            rows.append(row)
        return rows

    def comparison_rows(self) -> list:
        rows = []
        for pair, stats in self.comparison.items():
            left, right = pair.split("|", 1)
            row = {"state_a": left, "state_b": right}
            row.update(stats)
            rows.append(row)
        return rows

    def diagram_rows(self) -> list:
        rows = []
        for name, dgm in self.states.items():
            for rec in dgm.to_records():
                rec = {"state": name, **rec}
                rows.append(rec)
        return rows

    def summary_text(self, width: int = 78) -> str:
        """Human-readable report, suitable for a log or a console."""
        lines = ["=" * width, "OmicsTopology — state-space topology report", "=" * width]

        diag = self.diagnostics
        lines.append(
            f"embedding: PCA({diag.get('n_components')}, whiten={diag.get('whiten')}) "
            f"on {diag.get('n_features')} features, "
            f"metric={diag.get('metric')}, points/state cap={diag.get('max_points')}"
        )
        lines.append(
            f"shared Rips threshold = {diag.get('shared_threshold', float('nan')):.4g} "
            "(persistence values below are relative to it)"
        )
        lines.append("")

        header = f"{'state':<18}{'n':>5}{'H0':>6}{'H1':>6}{'H1 total':>11}{'H1 max':>9}"
        lines.append(header)
        lines.append("-" * width)
        for name, stats in self.summaries.items():
            n = stats.get("n_points", 0)
            lines.append(
                f"{str(name):<18}{n:>5}{stats.get('h0_n_finite', 0):>6}"
                f"{stats.get('h1_n_finite', 0):>6}"
                f"{stats.get('h1_total_persistence', 0.0):>11.4f}"
                f"{stats.get('h1_max_persistence', 0.0):>9.4f}"
            )
        lines.append("")

        if self.comparison:
            lines.append("pairwise diagram distances (normalised by the shared threshold):")
            for pair, stats in self.comparison.items():
                lines.append(
                    f"  {pair}: bottleneck={stats.get('all_bottleneck', float('nan')):.4f}  "
                    f"wasserstein={stats.get('all_wasserstein', float('nan')):.4f}  "
                    f"(H1 bottleneck={stats.get('h1_bottleneck', float('nan')):.4f})"
                )
            lines.append("")

        if self.test:
            t = self.test
            verdict = "significant" if t["p_value"] < 0.05 else "not significant"
            lines.append(
                f"permutation test ({t['statistic']}, {t['n_permutations']} permutations): "
                f"statistic={t['observed']:.4f}, p={t['p_value']:.4f} -> {verdict} at alpha=0.05"
            )
        else:
            lines.append("permutation test: not run (needs exactly two states)")
        lines.append("=" * width)
        return "\n".join(lines)

    def write(self, outdir: str, delimiter: str = "\t") -> list:
        """Write summaries, diagrams, comparisons and diagnostics to ``outdir``."""
        os.makedirs(outdir, exist_ok=True)
        paths = []
        paths.append(write_table(os.path.join(outdir, "topology_summary.tsv"),
                                 self.summary_rows(), delimiter=delimiter))
        paths.append(write_table(os.path.join(outdir, "persistence_diagrams.tsv"),
                                 self.diagram_rows(), delimiter=delimiter))
        if self.comparison:
            paths.append(write_table(os.path.join(outdir, "diagram_distances.tsv"),
                                     self.comparison_rows(), delimiter=delimiter))
        if self.test:
            null = np.asarray(self.test["null"], dtype=float)
            paths.append(write_table(
                os.path.join(outdir, "permutation_null.tsv"),
                [{"permutation": i + 1, "statistic": float(v)} for i, v in enumerate(null)],
                delimiter=delimiter,
            ))
            paths.append(write_table(
                os.path.join(outdir, "permutation_test.tsv"),
                [{
                    "statistic": self.test["statistic"],
                    "observed": self.test["observed"],
                    "p_value": self.test["p_value"],
                    "n_permutations": self.test["n_permutations"],
                    "threshold": self.test["threshold"],
                }],
                delimiter=delimiter,
            ))
        paths.append(write_table(
            os.path.join(outdir, "diagnostics.tsv"),
            [{"key": k, "value": v} for k, v in self.diagnostics.items()],
            delimiter=delimiter,
        ))
        self.files = paths
        return paths


def run_states(
    matrix: np.ndarray,
    groups,
    labels=None,
    n_components: int = 3,
    whiten: bool = False,
    metric: str = "euclidean",
    max_points: int | None = 120,
    max_dim: int = 1,
    max_simplices: int = 80_000,
    backend: str = "builtin",
    n_permutations: int = 100,
    statistic: str = "h1_max",
    run_test: bool = True,
    plots: bool = False,
    outdir: str | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> StateTopologyReport:
    """Run the full state-space topology workflow.

    Parameters
    ----------
    matrix : array-like, shape (n_samples, n_features)
        Expression matrix: rows are samples/cells, columns are genes.
    groups : sequence
        Group label or integer code per row of ``matrix``.
    labels : sequence, optional
        Names for the groups in the order of their sorted unique values.
        Defaults to ``"state0"``, ``"state1"``, ...
    n_components : int
        PCA components retained for the point cloud.
    whiten : bool
        Whiten the embedding.
    metric : str
        Distance metric for the point cloud.
    max_points : int or None
        Per-state subsample cap, because Rips complexes grow as O(n^3).
    max_dim : int
        Highest homology dimension.
    max_simplices : int
        Simplex budget for the shared threshold.
    backend : {"builtin", "auto", "ripser"}
        Homology backend.  Defaults to ``"builtin"`` so that the shared
        threshold is actually honoured and results do not change depending on
        whether ripser is installed.  ``"ripser"`` is much faster on large
        clouds but ignores the threshold, so persistence values are then on a
        different scale.
    n_permutations : int
        Permutations for the point-cloud test; set 0 or ``run_test=False`` to
        skip it.
    statistic : str
        Test statistic.  Defaults to ``"h1_max"`` — the longest surviving
        loop.  This matters a lot: ``"h1_total"`` sums *all* H1 bars
        including the short noise loops, and in simulation it detects a real
        ring only ~20% of the time while still firing on a pure mean shift
        ~10% of the time.  ``"h1_max"`` and ``"bottleneck"`` both reach 100%
        detection with 0% false positives on the same data.  See
        ``examples/power_analysis.py``.  Other options: ``"h1_total"``,
        ``"h1_count"``, ``"bottleneck"``, ``"wasserstein"``.
    run_test : bool
        Run the permutation test when there are exactly two states.
    plots : bool
        Write figures into ``outdir`` when matplotlib is available.
    outdir : str, optional
        Directory for tables and figures.
    seed : int
    verbose : bool

    Returns
    -------
    StateTopologyReport
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2-D: rows are samples, columns are features")
    groups = np.asarray(list(groups))
    if groups.shape[0] != X.shape[0]:
        raise ValueError(
            f"got {groups.shape[0]} group labels for {X.shape[0]} matrix rows"
        )

    unique = list(dict.fromkeys(groups.tolist()))
    if labels is None:
        labels = [f"state{i}" for i in range(len(unique))]
    if len(labels) != len(unique):
        raise ValueError(f"expected {len(unique)} labels, got {len(labels)}")
    name_of = dict(zip(unique, labels))

    # Embed once, on the pooled data, so that both states live in the same
    # coordinate system and their topologies are directly comparable.
    Z = standardize(X)
    if not np.isfinite(Z).all():
        raise ValueError(
            "standardised matrix contains NaN or inf; impute or drop missing values first"
        )
    emb = pca_embed(Z, n_components=n_components, whiten=whiten)

    clouds: dict = {}
    for value, name in name_of.items():
        pts, _ = subsample(emb[groups == value], max_points=max_points,
                           seed=seed, return_index=True)
        clouds[name] = pts

    # A single shared Rips threshold keeps Betti numbers comparable.  It is
    # taken as the most permissive value any single state can afford: the
    # budget must cover triangles, which grow as O(n^3), so judging the
    # threshold on the pooled cloud of 2n points would be needlessly
    # conservative and could truncate a real loop.
    state_distances = {name: pairwise_distances(pts, metric=metric)
                       for name, pts in clouds.items()}
    shared_threshold = max(
        choose_threshold(D_state, max_dim=max_dim, max_simplices=max_simplices)
        for D_state in state_distances.values()
    )
    if verbose:
        print(f"[omictopo] shared threshold = {shared_threshold:.4g}")

    diagrams: dict = {}
    for name, D_state in state_distances.items():
        diagrams[name] = persistent_homology(
            D=D_state, max_dim=max_dim, threshold=shared_threshold,
            max_simplices=max_simplices, backend=backend,
        )

    # `scale` puts every state on the same ruler: the shared threshold is what
    # the essential bars are capped at, so it is the natural divisor too.
    summaries = {
        name: topological_summary(d, rel=True, max_dim=max_dim,
                                  scale=shared_threshold)
        for name, d in diagrams.items()
    }

    comparison: dict = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            comparison[f"{a}|{b}"] = compare_diagrams(diagrams[a], diagrams[b], max_dim=max_dim)

    test = None
    if run_test and len(labels) == 2 and int(n_permutations) > 0:
        test = permutation_test_pointclouds(
            clouds[labels[0]], clouds[labels[1]],
            statistic=statistic, dim=1,
            n_permutations=int(n_permutations), seed=seed,
            max_points=None, max_dim=max_dim, max_simplices=max_simplices,
            threshold=shared_threshold, backend=backend,
        )

    diagnostics = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_components": int(n_components),
        "whiten": bool(whiten),
        "metric": metric,
        "max_points": max_points,
        "max_dim": int(max_dim),
        "max_simplices": int(max_simplices),
        "backend": str(backend),
        "shared_threshold": float(shared_threshold),
        "n_permutations": int(n_permutations),
        "statistic": str(statistic),
        "seed": int(seed),
    }
    for name, pts in clouds.items():
        diagnostics[f"n_points_{name}"] = int(pts.shape[0])

    report = StateTopologyReport(
        states=diagrams,
        clouds=clouds,
        summaries=summaries,
        comparison=comparison,
        test=test,
        diagnostics=diagnostics,
    )

    if outdir:
        report.write(outdir)
        if plots:
            report.files.extend(_write_figures(report, outdir, max_dim))
    return report


def _write_figures(report: StateTopologyReport, outdir: str, max_dim: int) -> list:
    from .plots import (plot_betti_curves, plot_landscape, plot_point_cloud,
                        plot_state_comparison)

    paths = []
    paths.append(plot_point_cloud(
        report.clouds, path=os.path.join(outdir, "state_space.png"),
        title="Embedded state space",
    ))
    paths.append(plot_state_comparison(
        report.clouds, report.states,
        path=os.path.join(outdir, "topology_comparison.png"), max_dim=max_dim,
    ))
    paths.append(plot_betti_curves(
        report.states, path=os.path.join(outdir, "betti_curves.png"), dim=max_dim,
    ))
    paths.append(plot_landscape(
        report.states, path=os.path.join(outdir, "landscapes.png"),
        dim=1 if max_dim >= 1 else 0,
    ))
    return paths
