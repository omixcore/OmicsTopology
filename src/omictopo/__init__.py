"""Topological data analysis of omics state spaces.

`omictopo` answers a question that mean-difference methods cannot: did the
*shape* of the sample/cell state space change between two biological
conditions?

Pipeline::

    expression matrix  ->  standardise + embed  ->  point cloud
                       ->  Vietoris-Rips filtration
                       ->  persistent homology (H0, H1, H2)
                       ->  diagrams / summaries / group comparison

Typical use::

    from omictopo import run_states

    report = run_states(matrix, groups, labels=["normal", "salt_stress"])
    print(report.summary_text())
"""

from .diagram import PersistenceDiagram
from .geometry import pairwise_distances, pca_embed, standardize, subsample
from .filtration import build_rips, choose_threshold
from .homology import persistent_homology
from .cycles import (
    circular_succession_score,
    cycles_from_edges,
    dominant_h1_cycles,
    h1_bar_lengths,
)
from .summary import betti_curve, persistence_landscape, topological_summary
from .compare import bottleneck_distance, compare_diagrams, wasserstein_distance
from .stats import permutation_test_pointclouds, permutation_test_replicates
from .pipeline import run_states, StateTopologyReport
from . import datasets

__version__ = "0.1.0"

__all__ = [
    "PersistenceDiagram",
    "pairwise_distances",
    "pca_embed",
    "standardize",
    "subsample",
    "build_rips",
    "choose_threshold",
    "persistent_homology",
    "dominant_h1_cycles",
    "cycles_from_edges",
    "h1_bar_lengths",
    "circular_succession_score",
    "betti_curve",
    "persistence_landscape",
    "topological_summary",
    "bottleneck_distance",
    "wasserstein_distance",
    "compare_diagrams",
    "permutation_test_pointclouds",
    "permutation_test_replicates",
    "run_states",
    "StateTopologyReport",
    "datasets",
    "__version__",
]
