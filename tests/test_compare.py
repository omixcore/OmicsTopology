"""Diagram distances and the statistical helpers around them."""

import numpy as np
import pytest

from omictopo import (
    PersistenceDiagram,
    bottleneck_distance,
    compare_diagrams,
    persistent_homology,
    wasserstein_distance,
)
from omictopo.datasets import blob, circle, two_clusters
from omictopo.stats import adjust_pvalues


def bar(birth, death, dim=1.0):
    return PersistenceDiagram(bars=np.array([[birth, death, dim]], dtype=float))


def test_distance_to_self_is_zero():
    dgm = persistent_homology(circle(20), max_dim=1)
    assert bottleneck_distance(dgm, dgm) == pytest.approx(0.0)
    assert wasserstein_distance(dgm, dgm) == pytest.approx(0.0)


def test_distances_are_symmetric():
    a = persistent_homology(circle(20), max_dim=1)
    b = persistent_homology(blob(20), max_dim=1)
    assert bottleneck_distance(a, b) == pytest.approx(bottleneck_distance(b, a))
    assert wasserstein_distance(a, b) == pytest.approx(wasserstein_distance(b, a))


def test_bottleneck_of_a_shifted_bar_equals_the_shift():
    """Two single-bar diagrams offset by (0.5, 0.5) in L-infinity.

    Both the direct match and the two diagonal matches cost 0.5, so the
    bottleneck distance is exactly 0.5 and the Wasserstein-1 distance picks
    the cheaper diagonal route, also 0.5.
    """
    a = bar(0.0, 1.0)
    b = bar(0.5, 1.5)
    assert bottleneck_distance(a, b, dim=1) == pytest.approx(0.5)
    assert wasserstein_distance(a, b, dim=1) == pytest.approx(0.5)


def test_bottleneck_never_exceeds_wasserstein():
    a = persistent_homology(circle(24), max_dim=1)
    b = persistent_homology(two_clusters(), max_dim=1)
    for dim in (0, 1):
        assert bottleneck_distance(a, b, dim=dim) <= wasserstein_distance(a, b, dim=dim) + 1e-9


def test_shape_with_a_loop_is_far_from_a_shape_without_one():
    with_loop = persistent_homology(circle(24), max_dim=1)
    without = persistent_homology(blob(24), max_dim=1)
    assert bottleneck_distance(with_loop, without, dim=1) > 0.5


def test_normalization_removes_overall_scale():
    """Scaling a diagram should not change a normalised comparison."""
    small = persistent_homology(circle(20, radius=1.0), max_dim=1)
    large = persistent_homology(circle(20, radius=4.0), max_dim=1)

    raw = compare_diagrams(small, large, normalize=False)
    normalized = compare_diagrams(small, large, normalize=True)
    assert raw["all_bottleneck"] > 5 * normalized["all_bottleneck"]
    assert normalized["normalization_scale"] > 0


def test_compare_diagrams_reports_expected_keys():
    a = persistent_homology(circle(16), max_dim=1)
    b = persistent_homology(blob(16), max_dim=1)
    stats = compare_diagrams(a, b, max_dim=1)
    for key in ("h0_bottleneck", "h1_bottleneck", "h0_wasserstein",
                "h1_wasserstein", "all_bottleneck", "all_wasserstein",
                "normalization_scale", "h1_n_essential_1", "h1_n_essential_2"):
        assert key in stats
    assert all(np.isfinite(v) for v in stats.values())


def test_empty_diagrams_give_zero_distance():
    empty = PersistenceDiagram(bars=np.empty((0, 3)))
    assert bottleneck_distance(empty, empty) == 0.0
    assert wasserstein_distance(empty, empty) == 0.0


def test_adjust_pvalues_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.5, 0.9])
    bh = adjust_pvalues(p, "bh")
    bonf = adjust_pvalues(p, "bonferroni")
    assert np.all((bh >= p - 1e-12) & (bh <= 1.0))
    assert np.all(bh <= bonf + 1e-12)
    assert np.all(np.diff(bh) >= -1e-12)  # sorted input stays sorted
    with pytest.raises(ValueError):
        adjust_pvalues(p, "nonsense")
