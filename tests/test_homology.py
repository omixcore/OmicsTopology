"""Persistent homology on point clouds whose topology is known by construction."""

import numpy as np
import pytest

from omictopo import pairwise_distances, persistent_homology
from omictopo.datasets import blob, circle, torus, two_clusters


def test_circle_has_exactly_one_persistent_loop():
    dgm = persistent_homology(circle(24), max_dim=1)

    assert dgm.max_persistence(1) > 1.0
    # A clean circle produces one dominant loop; after it, only the
    # degenerate zero-persistence Rips pairs remain.
    finite = np.sort(dgm.persistence(1))[::-1]
    assert finite[0] > 1.0
    assert np.all(finite[1:] < 1e-9), finite[:5]


def test_circle_betti_one_across_the_loop_lifetime():
    dgm = persistent_homology(circle(24), max_dim=1)
    for scale in (0.3, 0.6, 1.0, 1.5):
        assert dgm.betti(1, scale) == 1, scale
        assert dgm.betti(0, scale) == 1, scale


def test_two_clusters_produce_one_long_h0_bar():
    separation = 6.0
    dgm = persistent_homology(two_clusters(n=20, separation=separation), max_dim=1)

    assert dgm.max_persistence(0) == pytest.approx(separation, rel=0.15)
    assert dgm.betti(0, 0.3) == 2
    assert dgm.betti(1, 0.3) == 0


def test_unstructured_blob_has_no_persistent_loop():
    dgm = persistent_homology(blob(30), max_dim=1)
    assert dgm.max_persistence(1) < 0.1
    assert dgm.max_persistence(0) < 2.0


def test_single_point():
    dgm = persistent_homology(np.array([[0.0, 0.0]]), max_dim=1)
    assert len(dgm.essential(0)) == 1
    assert dgm.n_bars(1, finite_only=True) == 0
    assert len(dgm) == 1


def test_all_persistence_values_are_non_negative():
    """A valid filtration can never produce a bar that dies before it is born."""
    for points in (circle(24), two_clusters(), blob(30), torus(n_theta=10, n_phi=7)):
        dgm = persistent_homology(points, max_dim=1)
        for dim in (0, 1):
            p = dgm.persistence(dim)
            assert np.all(p >= -1e-9), (dim, p.min())


def test_h1_bar_count_matches_euler_characteristic():
    """For the complete complex, #H1 bars = #edges - (#vertices - 1)."""
    n = 24
    dgm = persistent_homology(circle(n), max_dim=1, max_simplices=10 ** 6)
    n_edges = n * (n - 1) // 2
    assert dgm.threshold == pytest.approx(2.0, rel=1e-9)  # the full diameter
    assert dgm.n_bars(1, finite_only=True) == n_edges - (n - 1)
    assert dgm.n_bars(1, finite_only=True) + len(dgm.essential(1)) <= n_edges


def test_torus_shows_a_dominant_loop():
    dgm = persistent_homology(torus(), max_dim=1)
    finite = np.sort(dgm.persistence(1))[::-1]
    assert finite[0] > 1.5
    # The minor loop is sampling-limited, but more than one loop-shaped
    # feature should still be visible.
    assert np.count_nonzero(finite > 0.5) >= 2


def test_mst_and_matrix_reduction_agree_on_h0():
    """Two independent H0 implementations must produce identical bars."""
    for points in (circle(24), two_clusters(n=15), blob(20)):
        D = pairwise_distances(points)
        mst = persistent_homology(D=D, max_dim=0, h0="mst")
        reduction = persistent_homology(D=D, max_dim=0, h0="reduction", threshold=float(D.max()))

        a = np.sort(mst.finite(0)[:, 1])
        b = np.sort(reduction.finite(0)[:, 1])
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-9)


def test_truncation_marks_far_bars_as_essential():
    """Bars that would die beyond the threshold are reported as essential."""
    D = pairwise_distances(circle(20))
    cutoff = 0.4 * float(D.max())
    dgm = persistent_homology(D=D, max_dim=1, threshold=cutoff)
    assert dgm.threshold == pytest.approx(cutoff)
    assert len(dgm.essential(1)) >= 1
    assert np.all(dgm.finite(1)[:, 1] <= cutoff + 1e-9)


def test_diagram_accessors_and_records():
    dgm = persistent_homology(circle(20), max_dim=1)
    records = dgm.to_records()
    assert records and {"state" not in records[0]}
    assert {"dim", "birth", "death", "persistence", "essential"} <= set(records[0])
    summary = dgm.summary()
    assert summary["h1_n_finite"] == dgm.n_bars(1, finite_only=True)
    assert summary["threshold"] == pytest.approx(dgm.threshold)
    assert "H1=" in repr(dgm)


def test_unknown_backend_and_dimension_are_rejected():
    with pytest.raises(ValueError):
        persistent_homology(circle(8), max_dim=1, h0="nonsense")
    with pytest.raises(ValueError):
        persistent_homology(circle(8), max_dim=1, backend="nonsense")
