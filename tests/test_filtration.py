"""Filtration construction: ordering invariant and budget control."""

import numpy as np
import pytest

from omictopo import build_rips, choose_threshold, pairwise_distances
from omictopo.filtration import count_simplices
from shapes import circle


def test_face_always_precedes_coface():
    """Every face must appear earlier in the filtration than its cofaces.

    This is the invariant the boundary reduction depends on: without it the
    persistence pairs come out with negative or nonsensical lengths.
    """
    D = pairwise_distances(circle(12))
    filtration = build_rips(D, max_dim=1)

    position = {v: i for i, v in enumerate(filtration.vertices)}
    for j, simplex in enumerate(filtration.vertices):
        if len(simplex) < 2:
            continue
        for drop in range(len(simplex)):
            face = tuple(v for k, v in enumerate(simplex) if k != drop)
            if len(face) == 1:
                # vertices are the first n simplices in the order
                assert position[face] < j
            else:
                assert position[face] < j, (face, j)


def test_simplex_counts_for_three_points():
    """Three points spanning a triangle: 3 vertices + 3 edges + 1 triangle."""
    D = pairwise_distances(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]))
    at_full = count_simplices(D, float(D.max()), simplex_dim=2)
    assert at_full == 7
    assert count_simplices(D, 0.0, simplex_dim=2) == 3


def test_count_simplices_is_monotone_in_threshold():
    D = pairwise_distances(circle(16))
    values = [count_simplices(D, t, simplex_dim=2) for t in np.linspace(0.05, D.max(), 8)]
    assert all(a <= b for a, b in zip(values, values[1:])), values


def test_choose_threshold_respects_budget():
    D = pairwise_distances(circle(24))
    budget = 4000
    threshold = choose_threshold(D, max_dim=1, max_simplices=budget)
    assert count_simplices(D, threshold, simplex_dim=2) <= budget


def test_build_rips_refuses_to_explode():
    D = pairwise_distances(circle(30))
    with pytest.raises(ValueError, match="over the budget"):
        build_rips(D, max_dim=1, threshold=float(D.max()), max_simplices=500)


def test_explicit_threshold_is_honoured():
    D = pairwise_distances(circle(10))
    cutoff = 0.5 * float(D.max())
    filtration = build_rips(D, max_dim=1, threshold=cutoff)
    assert filtration.threshold == pytest.approx(cutoff)
    assert filtration.truncated
    assert filtration.values.max() <= cutoff


def test_bare_one_skeleton_can_be_requested():
    D = pairwise_distances(circle(10))
    filtration = build_rips(D, max_dim=1, simplex_dim=1)
    assert set(filtration.count_by_dim()) == {0, 1}
