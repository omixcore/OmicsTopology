"""End-to-end workflow: does a real topological change get reported?"""

import os

import numpy as np
import pytest

from omictopo import run_states
from omictopo.datasets import two_state_omics

FAST = dict(max_points=40, max_simplices=40_000, n_permutations=60, seed=0)


def test_cyclic_stress_state_is_detected():
    X, groups, labels = two_state_omics(n_genes=400, n_per_state=40,
                                        state_b="cycle", seed=0)
    report = run_states(X, groups, labels=labels, **FAST)

    control = report.summaries["control"]
    stress = report.summaries["salt_stress"]

    # A ring carries far more H1 than a single basin.
    assert stress["h1_total_persistence"] > 2.0 * control["h1_total_persistence"]
    assert stress["h1_max_persistence"] > 3.0 * control["h1_max_persistence"]
    assert report.test is not None
    assert report.test["p_value"] < 0.05


def test_negative_control_is_not_significant():
    """Same two conditions but only a mean shift: no topology change."""
    X, groups, labels = two_state_omics(n_genes=400, n_per_state=40,
                                        state_b="shift", seed=0)
    report = run_states(X, groups, labels=labels, **FAST)

    control = report.summaries["control"]
    stress = report.summaries["salt_stress"]
    assert stress["h1_total_persistence"] == pytest.approx(
        control["h1_total_persistence"], rel=1.0
    )
    assert report.test["p_value"] > 0.2


def test_report_text_and_tables(tmp_path):
    X, groups, labels = two_state_omics(n_genes=300, n_per_state=25,
                                        state_b="cycle", seed=1)
    report = run_states(X, groups, labels=labels, outdir=str(tmp_path),
                        plots=False, **FAST)

    text = report.summary_text()
    assert "control" in text and "salt_stress" in text
    assert "permutation test" in text
    assert "bottleneck" in text

    for name in ("topology_summary.tsv", "persistence_diagrams.tsv",
                 "diagram_distances.tsv", "permutation_test.tsv",
                 "permutation_null.tsv", "diagnostics.tsv"):
        assert os.path.exists(tmp_path / name), name

    rows = report.summary_rows()
    assert {r["state"] for r in rows} == {"control", "salt_stress"}
    assert report.comparison_rows()[0]["state_a"] == "control"
    assert report.diagram_rows()


def test_three_states_compare_pairwise_without_a_test():
    X, groups, labels = two_state_omics(n_genes=200, n_per_state=20, seed=2)
    groups = [0] * 20 + [1] * 20 + [2] * 20
    X = np.vstack([X, X[:20] * 1.5])

    report = run_states(X, groups, labels=["a", "b", "c"],
                        max_points=20, max_simplices=20_000, seed=0)

    assert set(report.state_names()) == {"a", "b", "c"}
    assert set(report.comparison) == {"a|b", "a|c", "b|c"}
    assert report.test is None
    assert "not run" in report.summary_text()


def test_group_length_mismatch_is_rejected():
    X, groups, labels = two_state_omics(n_genes=50, n_per_state=10, seed=3)
    with pytest.raises(ValueError, match="group labels"):
        run_states(X, groups[:-2], labels=labels)


def test_wrong_number_of_labels_is_rejected():
    X, groups, _ = two_state_omics(n_genes=50, n_per_state=10, seed=4)
    with pytest.raises(ValueError, match="labels"):
        run_states(X, groups, labels=["only_one"])


def test_missing_values_are_rejected():
    X, groups, labels = two_state_omics(n_genes=50, n_per_state=10, seed=5)
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_states(X, groups, labels=labels)


def test_plots_are_written_when_available(tmp_path):
    pytest.importorskip("matplotlib")
    X, groups, labels = two_state_omics(n_genes=200, n_per_state=20, seed=6)
    report = run_states(X, groups, labels=labels, outdir=str(tmp_path),
                        plots=True, max_points=20, max_simplices=20_000,
                        n_permutations=0, seed=0)
    for name in ("state_space.png", "topology_comparison.png",
                 "betti_curves.png", "landscapes.png"):
        assert os.path.exists(tmp_path / name), name
    assert any(p.endswith(".png") for p in report.files)
