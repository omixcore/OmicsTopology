"""Permutation tests: the null is respected and real change is detected."""

import numpy as np
import pytest

from omictopo import persistent_homology
from omictopo.datasets import blob, circle
from omictopo.stats import permutation_test_pointclouds, permutation_test_replicates


def _replicates(generator, n_reps=4, **kwargs):
    return [persistent_homology(generator(**{**kwargs, "seed": s}), max_dim=1)
            for s in range(n_reps)]


def test_replicate_test_detects_a_shape_change():
    loops = _replicates(circle, n=20, noise=0.05)
    flat = _replicates(blob, n=20, dims=2)
    result = permutation_test_replicates(loops, flat, n_permutations=500, seed=0)

    assert result["p_value"] < 0.05
    assert result["observed"] > 0
    assert result["cross"] > 0.5 * max(result["within_a"], result["within_b"])
    assert result["null"].shape == (500,)


def test_replicate_test_respects_the_null():
    """Two samples of the *same* shape must not look different."""
    a = _replicates(circle, n=20, noise=0.05)
    b = [persistent_homology(circle(20, noise=0.05, seed=100 + s), max_dim=1)
         for s in range(4)]
    result = permutation_test_replicates(a, b, n_permutations=500, seed=1)
    assert result["p_value"] > 0.05


def test_replicate_test_needs_two_replicates():
    one = _replicates(circle, n=12, n_reps=1)
    with pytest.raises(ValueError, match="at least two"):
        permutation_test_replicates(one, one, n_permutations=10)


def test_pointcloud_test_reports_a_well_formed_result():
    result = permutation_test_pointclouds(
        circle(18), blob(18, dims=2), n_permutations=20, seed=0, max_dim=1
    )
    assert result["null"].shape == (20,)
    assert 0.0 < result["p_value"] <= 1.0
    assert result["observed"] >= 0.0
    assert result["n_points_a"] == 18
    assert result["n_points_b"] == 18
    assert result["threshold"] > 0
    assert isinstance(result["observed_a"], type(result["observed_b"]))


def test_pointcloud_test_detects_a_loop():
    result = permutation_test_pointclouds(
        circle(20), blob(20, dims=2), statistic="h1_max",
        n_permutations=40, seed=0, max_dim=1,
    )
    assert result["p_value"] < 0.1


def test_pointcloud_test_respects_the_null_for_identical_shapes():
    result = permutation_test_pointclouds(
        circle(18, noise=0.15, seed=1), circle(18, noise=0.15, seed=2),
        n_permutations=30, seed=3, max_dim=1,
    )
    assert result["p_value"] > 0.1


def test_unknown_statistic_is_rejected():
    with pytest.raises(ValueError, match="unknown statistic"):
        permutation_test_pointclouds(
            circle(10), blob(10, dims=2), statistic="nonsense", n_permutations=5
        )
