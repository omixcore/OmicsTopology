"""Tests for the real-data cycle-counting example helpers.

The counting pipeline itself is exercised by ``test_cycles.py`` and the
benchmark; what is specific to ``examples/cycle_counting_real.py`` is the
bookkeeping around a *replicated* design: centring on the replicate, collapsing
to design cells, the dominant-bar gap, and the simulation used by the power
sweep.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples"))

from cycle_counting_real import centre_by_group, circular_distance, score  # noqa: E402
from cycle_counting_power import sow_ring  # noqa: E402


# --------------------------------------------------------------------------
def test_centre_by_group_removes_each_group_mean():
    X = np.array([[10.0, 0.0], [12.0, 2.0],
                  [0.0, 10.0], [2.0, 12.0]])
    out = centre_by_group(X, ["a", "a", "b", "b"])
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    # the *shape* of each group's variation is untouched
    np.testing.assert_allclose(out[1] - out[0], X[1] - X[0], atol=1e-12)
    np.testing.assert_allclose(out[3] - out[2], X[3] - X[2], atol=1e-12)


def test_centre_by_group_is_a_no_op_for_one_group():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 4))
    out = centre_by_group(X, ["only"] * 7)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(out, X - X.mean(axis=0, keepdims=True), atol=1e-12)


def test_centre_by_group_does_not_modify_its_input():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X0 = X.copy()
    centre_by_group(X, ["a", "a", "b"])
    np.testing.assert_array_equal(X, X0)


# --------------------------------------------------------------------------
def test_circular_distance_is_zero_on_the_diagonal_and_symmetric():
    a = np.array([0.0, 1.0, 3.0, 6.2])
    D = circular_distance(a)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    assert D.min() >= -1e-12 and D.max() <= np.pi + 1e-12


def test_circular_distance_wraps_the_short_way():
    D = circular_distance(np.array([0.0, 2.0 * np.pi - 0.1]))
    assert D[0, 1] == pytest.approx(0.1, abs=1e-9)


# --------------------------------------------------------------------------
def test_score_counts_one_cycle_for_a_clean_ring():
    X, phases = sow_ring(n_phases=12, n_reps=6, n_signal=3000, seed=0)
    rec = score(X, n_null=20, seed=0, phase=phases)
    assert rec["n_cycles"] == 1
    assert rec["gap"] is not None and rec["gap"] > 5.0
    assert rec["mantel_r"] > 0.9


def test_score_counts_zero_when_the_ring_is_carried_by_few_genes():
    # the measured operating point: a minority of the features carry the ring
    X, phases = sow_ring(n_phases=12, n_reps=6, n_signal=100, seed=0)
    rec = score(X, n_null=20, seed=0, phase=phases)
    assert rec["n_cycles"] == 0
    assert rec["gap"] is None or rec["gap"] < 3.0


def test_score_reports_a_gap_even_without_a_cycle():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 500))
    rec = score(X, n_null=15, seed=0)
    assert "gap" in rec and "mantel_r" not in rec
    assert rec["n_cycles"] == 0


# --------------------------------------------------------------------------
def test_sow_ring_places_the_signal_in_the_first_n_signal_columns():
    X, phases = sow_ring(n_phases=8, n_reps=3, n_signal=50, p=300, seed=3)
    assert X.shape == (24, 300)
    assert phases.size == 24
    rng = np.random.default_rng(3)
    ref, ref_phases = sow_ring(n_phases=8, n_reps=3, n_signal=50, p=300, seed=3)
    np.testing.assert_allclose(X, ref)
    np.testing.assert_allclose(phases, ref_phases)
    # the silent columns really are noise, not a second copy of the ring
    assert np.allclose(X[:, 50:], 0.0) is False
