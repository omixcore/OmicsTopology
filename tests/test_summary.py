"""Vector summaries: Betti curves, landscapes, summary dictionaries."""

import numpy as np
import pytest

from omictopo import (
    betti_curve,
    persistence_landscape,
    persistent_homology,
    topological_summary,
)
from omictopo.datasets import blob, circle
from omictopo.summary import landscape_norm, summary_table


def test_betti_curve_agrees_with_pointwise_betti():
    dgm = persistent_homology(circle(24), max_dim=1)
    grid, beta = betti_curve(dgm, dim=1, n_points=40)
    assert grid.shape == beta.shape
    for t, value in zip(grid, beta):
        assert value == dgm.betti(1, float(t))


def test_betti_curve_of_an_empty_dimension_is_zero():
    dgm = persistent_homology(blob(12), max_dim=1)
    grid, beta = betti_curve(dgm, dim=2)
    assert beta.shape == grid.shape
    assert np.allclose(beta, 0.0)


def test_landscape_is_ordered_and_non_negative():
    dgm = persistent_homology(circle(24), max_dim=1)
    grid, lam = persistence_landscape(dgm, dim=1, k_max=3, n_points=64)
    assert lam.shape == (3, 64)
    assert np.all(lam >= -1e-12)
    assert np.all(lam[0] >= lam[1] - 1e-12)
    assert np.all(lam[1] >= lam[2] - 1e-12)
    assert lam[0].max() > 0.0


def test_landscape_ranks_are_zero_when_there_are_no_bars():
    dgm = persistent_homology(blob(10), max_dim=0)
    grid, lam = persistence_landscape(dgm, dim=2, k_max=2)
    assert np.allclose(lam, 0.0)


def test_landscape_norm_separates_a_loop_from_a_blob():
    loop = persistent_homology(circle(24), max_dim=1)
    flat = persistent_homology(blob(24), max_dim=1)
    grid_a, lam_a = persistence_landscape(loop, dim=1)
    grid_b, lam_b = persistence_landscape(flat, dim=1)
    assert landscape_norm(grid_a, lam_a) > 5 * landscape_norm(grid_b, lam_b)


def test_topological_summary_contents():
    dgm = persistent_homology(circle(24), max_dim=1)
    stats = topological_summary(dgm, rel=False, max_dim=1)
    for key in ("threshold", "n_points", "h0_n_finite", "h0_n_essential",
                "h1_total_persistence", "h1_max_persistence", "h1_entropy"):
        assert key in stats
    assert stats["n_points"] == 24
    assert stats["h0_n_essential"] == 1
    assert stats["h1_total_persistence"] == pytest.approx(dgm.total_persistence(1))


def test_relative_summary_divides_by_the_scale():
    dgm = persistent_homology(circle(20, radius=3.0), max_dim=1)
    absolute = topological_summary(dgm, rel=False, max_dim=1)
    relative = topological_summary(dgm, rel=True, max_dim=1)
    assert "scale" in relative
    assert relative["h1_total_persistence"] < absolute["h1_total_persistence"]
    assert relative["scale"] > 1.0


def test_summary_table_unions_keys():
    a = topological_summary(persistent_homology(circle(12), max_dim=1), max_dim=1)
    b = topological_summary(persistent_homology(circle(12), max_dim=0), max_dim=0)
    rows = summary_table({"a": a, "b": b})
    assert [r["state"] for r in rows] == ["a", "b"]
    assert all(set(r) == set(rows[0]) for r in rows)
