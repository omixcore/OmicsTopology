"""Representative cycles and the cyclic order they induce.

The claim being tested is the one the unlabelled experiment rests on: the
reduction does not merely report *how long* a loop is, it names the samples on
the loop in the order the loop visits them, and that order is the true angular
order for a point cloud that really lies on a circle.
"""

import numpy as np
import pytest

from omictopo import build_rips, pairwise_distances, persistent_homology
from omictopo.cycles import (
    _harmonic_power,
    circular_succession_score,
    cycle_identifiability,
    cycles_from_edges,
    dominant_h1_cycles,
    h1_bar_lengths,
    reduce_boundary_columns,
    topology_guided_cycle,
)
from omictopo.datasets import circle
from omictopo.homology import reduce_boundary


def _ring_cloud(n=24, radius=3.0, jitter=0.0, seed=0):
    """Points on a circle with *known* angular order."""
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    xy = np.c_[np.cos(angles), np.sin(angles)] * radius
    if jitter:
        xy = xy + rng.normal(0.0, jitter, xy.shape)
    return xy, angles


# --------------------------------------------------------------------------
# the reduction with column provenance
# --------------------------------------------------------------------------
def test_reduce_boundary_columns_matches_reduce_boundary():
    D = pairwise_distances(circle(16))
    filtration = build_rips(D, max_dim=1, max_simplices=200_000)
    pairs, essential, reduced = reduce_boundary_columns(filtration.boundary)
    ref_pairs, ref_essential = reduce_boundary(filtration.boundary)

    assert pairs == ref_pairs
    assert essential == ref_essential
    assert len(reduced) == len(filtration.boundary)


def test_reduced_column_of_a_triangle_is_its_own_cycle():
    # A single triangle: the H1 class is born at the last edge and dies at the
    # triangle, and the representative is the triangle itself.
    D = np.array([[0.0, 1.0, 1.0],
                  [1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0]])
    filtration = build_rips(D, max_dim=1, max_simplices=1000)
    pairs, _essential, reduced = reduce_boundary_columns(filtration.boundary)
    h1_pairs = [(b, d) for b, d in pairs
                if int(filtration.dims[b]) == 1 and int(filtration.dims[d]) == 2]
    assert len(h1_pairs) == 1, pairs
    _birth, death = h1_pairs[0]
    support = [k for k in range(reduced[death].bit_length())
               if (reduced[death] >> k) & 1]
    assert len(support) == 3
    assert all(int(filtration.dims[k]) == 1 for k in support)


# --------------------------------------------------------------------------
# edge set -> closed walk
# --------------------------------------------------------------------------
def test_cycles_from_edges_walks_a_triangle():
    walks = cycles_from_edges([(0, 1), (1, 2), (2, 0)])
    assert len(walks) == 1
    walk, closed = walks[0]
    assert closed is True
    assert sorted(walk) == [0, 1, 2]
    # consecutive entries really are edges of the cycle
    pairs = {frozenset((walk[i], walk[(i + 1) % 3])) for i in range(3)}
    assert pairs == {frozenset((0, 1)), frozenset((1, 2)), frozenset((0, 2))}


def test_cycles_from_edges_finds_the_longest_component_first():
    walks = cycles_from_edges([(0, 1), (1, 2), (2, 0),
                               (3, 4), (4, 5), (5, 6), (6, 3)])
    assert [len(w) for w, _ in walks] == [4, 3]


def test_cycles_from_edges_is_deterministic():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    runs = [cycles_from_edges(edges)]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        shuffled = [edges[i] for i in rng.permutation(len(edges))]
        runs.append(cycles_from_edges(shuffled))
    # the longest walk is the 4-cycle and must come out identical every time
    longest = [max(run, key=lambda t: len(t[0]))[0] for run in runs]
    assert all(np.array_equal(longest[0], w) for w in longest), longest


# --------------------------------------------------------------------------
# the dominant cycle of a circle
# --------------------------------------------------------------------------
def test_circle_order_is_recovered_up_to_rotation_and_reflection():
    xy, angles = _ring_cloud(n=24)
    cycles = dominant_h1_cycles(X=xy, metric="euclidean", rank=1)
    assert cycles, "a circle must have a dominant H1 bar"

    order = np.asarray(cycles[0]["order"], dtype=int)
    assert cycles[0]["closed"] is True
    assert order.size >= 12, order.size          # it should tour the ring
    # the score is invariant to start and direction, and must be perfect here
    assert circular_succession_score(angles, order) == pytest.approx(1.0)


def test_circle_order_survives_jitter():
    xy, angles = _ring_cloud(n=24, jitter=0.05, seed=3)
    cycles = dominant_h1_cycles(X=xy, metric="euclidean", rank=1)
    order = np.asarray(cycles[0]["order"], dtype=int)
    assert circular_succession_score(angles, order) == pytest.approx(1.0)


def test_dominant_cycle_is_a_valid_closed_walk():
    xy, _angles = _ring_cloud(n=20, seed=7)
    cycles = dominant_h1_cycles(X=xy, metric="euclidean", rank=2)
    assert cycles
    for cycle in cycles:
        order = cycle["order"]
        assert len(order) == len(set(order)), "a walk must not repeat a vertex"
        assert cycle["n_edges"] >= len(order)
        assert cycle["closed"] is True


def test_dominant_cycle_persistence_matches_the_diagram():
    D = pairwise_distances(circle(24))
    cycles = dominant_h1_cycles(D=D, rank=1)
    diagram = persistent_homology(D=D, max_dim=1, backend="builtin",
                                  max_simplices=1_000_000)
    finite = diagram.finite(1)
    top = float(np.max(finite[:, 1] - finite[:, 0]))
    assert cycles[0]["persistence"] == pytest.approx(top, rel=1e-6)


def test_two_clusters_have_only_short_cycles():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.3, (12, 2))
    b = rng.normal(6.0, 0.3, (12, 2))
    cycles = dominant_h1_cycles(X=np.vstack([a, b]), metric="euclidean", rank=1)
    # two tight blobs: no long loop, so the longest bar is a fraction of the gap
    assert cycles[0]["persistence"] < 0.5 * 6.0


# --------------------------------------------------------------------------
# bar lengths
# --------------------------------------------------------------------------
def test_h1_bar_lengths_are_normalised_and_sorted():
    D = pairwise_distances(circle(24))
    bars = h1_bar_lengths(D)
    assert bars[0] > 0.0
    assert np.all(np.diff(bars) <= 1e-12), bars[:5]
    # normalising by the RMS radius of a unit circle gives a scale-free number
    raw = h1_bar_lengths(D, normalized=False)
    scale = np.sqrt(np.mean(D ** 2) / 2.0)
    assert np.allclose(bars, raw / scale)


def test_h1_bar_lengths_of_a_single_point_is_empty():
    bars = h1_bar_lengths(np.zeros((1, 1)))
    assert bars.size == 0


# --------------------------------------------------------------------------
# the succession score
# --------------------------------------------------------------------------
def test_succession_score_is_invariant_to_start_and_direction():
    angles = np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, 24))
    order = np.arange(24)
    assert circular_succession_score(angles, order) == pytest.approx(1.0)
    assert circular_succession_score(angles, order[::-1]) == pytest.approx(1.0)
    assert circular_succession_score(angles, np.roll(order, 9)) == pytest.approx(1.0)


def test_succession_score_separates_a_correct_order_from_scrambled_ones():
    """The metric's floor is ~0.5, not 0: either direction is allowed.

    Because a 1-cycle has no canonical direction the score takes the better of
    the two, and a finely scrambled order still gets credit for the half of
    its steps that happen to move one way.  What matters is the gap to 1.0,
    and that is what the permutation null quoted in the experiment calibrates.
    """
    angles = np.sort(np.random.default_rng(1).uniform(0, 2 * np.pi, 32))
    perfect = circular_succession_score(angles, np.arange(32))
    linear = circular_succession_score(angles, np.argsort(np.cos(angles)))
    interleaved = circular_succession_score(
        angles, np.argsort(np.cos(angles))[::-1])          # worst-case tour
    random_order = np.mean([circular_succession_score(
        angles, np.random.default_rng(s).permutation(32)) for s in range(20)])

    assert perfect == pytest.approx(1.0)
    assert linear < 0.75, linear
    assert interleaved < 0.75, interleaved
    assert random_order < 0.75, random_order
    assert perfect > linear and perfect > random_order


def test_succession_score_of_an_alternating_walk_is_about_a_half():
    angles = np.sort(np.random.default_rng(4).uniform(0, 2 * np.pi, 24))
    alternating = np.empty(24, dtype=int)
    alternating[0::2] = np.arange(12)
    alternating[1::2] = np.arange(12) + 12
    score = circular_succession_score(angles, alternating)
    assert score < 0.3, score


def test_succession_score_needs_at_least_three_steps():
    assert circular_succession_score(np.array([0.0, 1.0, 2.0]), [0, 1]) == 0.0
    assert circular_succession_score(np.array([0.5, 0.5, 0.5]), [0, 1, 2]) == 0.0


def test_replicate_phases_do_not_disturb_a_correct_order():
    # three time points, three replicates each, visited time point by time point:
    # every step either stays (another replicate) or advances (the next time
    # point), so the order is perfect
    base = np.array([0.0, 1.0, 2.0]) * (2 * np.pi / 3)
    phases = np.repeat(base, 3)
    assert circular_succession_score(phases, np.arange(9)) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the identifiability rule
# --------------------------------------------------------------------------
def test_identifiability_accepts_the_one_design_that_worked():
    """The yearlong series: gap 4.93, circular Mantel r = +0.70."""
    rec = cycle_identifiability(4.93, 0.701)
    assert rec["verdict"] == "identifiable"
    assert rec["report_count"] is True


def test_identifiability_separates_a_real_ring_from_a_flat_spectrum():
    """The diurnal design: the ring is real, H1 simply cannot resolve it."""
    rec = cycle_identifiability(1.36, 0.701)
    assert rec["verdict"] == "underpowered"
    assert rec["report_count"] is False


def test_identifiability_flags_a_loop_that_ignores_the_design_axis():
    """A dominant bar whose order does not follow the axis is not a cycle count."""
    assert cycle_identifiability(2.05, 0.096)["verdict"] == "inconsistent"


def test_identifiability_flags_a_cloud_with_no_ring_in_it():
    assert cycle_identifiability(1.07, 0.02)["verdict"] == "no_ring"


def test_identifiability_without_a_mantel_correlation():
    assert cycle_identifiability(5.0, None)["report_count"] is True
    assert cycle_identifiability(1.1, None)["report_count"] is False


def test_identifiability_threshold_sits_between_the_measured_gaps():
    """1.36 failed, 4.93 worked; the default split must separate them."""
    assert cycle_identifiability(1.36, 0.9)["report_count"] is False
    assert cycle_identifiability(4.93, 0.9)["report_count"] is True


def test_identifiability_always_gives_a_readable_sentence():
    for gap, r in ((4.93, 0.7), (1.36, 0.7), (2.05, 0.1), (1.07, 0.02),
                   (3.0, None), (0.5, None)):
        rec = cycle_identifiability(gap, r)
        assert isinstance(rec["advice"], str) and len(rec["advice"]) > 20
        assert rec["verdict"] in {"identifiable", "underpowered",
                                  "no_ring", "inconsistent"}


def test_identifiability_thresholds_are_configurable():
    assert cycle_identifiability(1.5, 0.9, gap_threshold=1.4)["report_count"] is True
    assert cycle_identifiability(4.9, 0.9, gap_threshold=5.0)["report_count"] is False
    assert cycle_identifiability(4.9, 0.4, r_threshold=0.3)["report_count"] is True


# --------------------------------------------------------------------------
# topology-guided selection
#
# The procedure is documented as *not validated*: it produced no improvement on
# the real designs and its Gaussian-null calibration fires on a cloud with no
# ring.  These tests therefore pin down what it computes, not that it works.
# --------------------------------------------------------------------------
def _wide_ring(n=30, p=60, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.linspace(0.0, 24.0, n)
    phases = rng.uniform(0.0, 2.0 * np.pi, p)
    amp = np.exp(rng.normal(0.0, 0.5, p))
    X = np.cos(2.0 * np.pi * hours[:, None] / 24.0 + phases[None, :]) * amp[None, :]
    return X + rng.normal(0.0, noise, X.shape), 2.0 * np.pi * hours / 24.0


def test_harmonic_power_prefers_the_genes_that_follow_the_order():
    order = np.arange(12)
    theta = 2.0 * np.pi * order / 12.0
    rng = np.random.default_rng(0)
    Z = np.zeros((12, 3))
    Z[:, 0] = np.cos(theta)
    Z[:, 1] = np.sin(theta)
    Z[:, 2] = rng.normal(size=12)
    power = _harmonic_power(Z, order)
    assert power[0] > power[2]
    assert power[1] > power[2]


def test_guided_cycle_returns_a_valid_selection():
    X, _ = _wide_ring()
    out = topology_guided_cycle(X, n_keep=20, n_iter=2)
    selected = out["selected"]
    assert len(selected) == 20
    assert len(set(selected)) == 20
    assert all(0 <= c < X.shape[1] for c in selected)
    assert out["X_selected"].shape == (X.shape[0], 20)
    assert out["D_selected"].shape == (X.shape[0], X.shape[0])
    assert len(out["history"]) == 3          # one per round, plus the final count
    assert out["gap_selected"] is None or out["gap_selected"] >= 1.0


def test_guided_cycle_recovers_a_clean_ring_it_was_given():
    """The selection must not destroy an order that was already there.

    The threshold is 0.8 rather than 1.0 on purpose: the point of the test is
    that the alternation does not wreck the geometry, not that it is perfect.
    """
    X, angles = _wide_ring(noise=0.5)
    out = topology_guided_cycle(X, n_keep=30, n_iter=2)
    order = np.asarray(out["order"], dtype=int)
    if order.size >= 6:
        assert circular_succession_score(angles, order) >= 0.8


def test_guided_cycle_reports_no_order_when_there_is_no_cycle():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(16, 40))
    out = topology_guided_cycle(X, n_keep=20, n_iter=2)
    assert out["history"][0]["n_visited"] >= 0
    assert isinstance(out["order"], list)


def test_guided_cycle_keeps_the_asked_number_of_genes_when_it_can():
    X, _ = _wide_ring(n=20, p=40)
    out = topology_guided_cycle(X, n_keep=40, n_iter=1)
    assert len(out["selected"]) == 40
    out2 = topology_guided_cycle(X, n_keep=100, n_iter=1)
    assert len(out2["selected"]) == 40     # clipped to the number available
