"""The example scripts are part of the claim, so their pure parts are tested.

Everything under ``examples/`` used to be untested, which is a real gap: those
scripts produced every number quoted in the evaluation documents.  The
cloud-loading halves need GEO downloads and are therefore only exercised by
running the scripts, but the pieces that do the reasoning — the ordering
methods, the synthetic controls, the succession score, the name parsers — are
pure functions and are checked here without touching the network.
"""

import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import axis_separability as axsep  # noqa: E402
import cycle_identifiability as ci  # noqa: E402
import cycle_counting_real as ccr  # noqa: E402
import incremental_value as iv  # noqa: E402
import prepare_cycle_designs as pcd  # noqa: E402

from omictopo.cycles import dominant_h1_cycles, h1_bar_lengths  # noqa: E402
from omictopo.geometry import pairwise_distances, standardize  # noqa: E402


def _ring(n=24, p=200, noise=1.0, seed=0):
    """A ring across many features, built exactly like ``synth_ring``.

    Two cautions are baked into this helper.  First, the cloud must have more
    than two features: for a two-feature ring the correlation metric collapses
    to rank one, because centring each *row* of a two-column matrix removes
    exactly one dimension and every sample's residual becomes parallel to
    ``(1, -1)``.  Second, a handful of features is not enough either — the
    correlation distance between two samples is set by the component they
    *share*, so a ring has to be carried by many genes with independent noise
    before the shared sweep dominates.  This is the same construction the
    benchmark uses, so the tests exercise what the experiments exercise.
    """
    rng = np.random.default_rng(seed)
    hours = np.linspace(0.0, 23.0, n)
    phases = rng.uniform(0.0, 2.0 * np.pi, p)
    amp = np.exp(rng.normal(0.0, 0.6, p))
    X = (np.cos(2.0 * np.pi * hours[:, None] / 24.0 + phases[None, :])
         * amp[None, :])
    X = X + rng.normal(0.0, noise, X.shape)
    angles = 2.0 * np.pi * hours / 24.0
    return standardize(X), angles


def _order_score(angles, order):
    return ci.succession(angles, order, n_perm=200)["score"]


# --------------------------------------------------------------------------
# incremental_value: the ordering methods
# --------------------------------------------------------------------------
def test_synthetic_controls_have_the_declared_shape():
    for kind in ("synth_ring", "synth_line", "synth_noise",
                 "synth_clusters", "synth_branch"):
        X, phases, hours = iv.make_synthetic(kind, n_genes=200, n_samples=12)
        assert X.shape == (12, 200)
        assert phases.shape == hours.shape == (12,)
        assert np.all(np.isfinite(X))
        assert np.all((phases >= 0) & (phases < 2 * np.pi))


def test_synthetic_controls_are_seed_reproducible():
    a = iv.make_synthetic("synth_ring", n_genes=50, n_samples=8, seed=3)[0]
    b = iv.make_synthetic("synth_ring", n_genes=50, n_samples=8, seed=3)[0]
    c = iv.make_synthetic("synth_ring", n_genes=50, n_samples=8, seed=4)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_synthetic_controls_reject_an_unknown_kind():
    with pytest.raises(ValueError):
        iv.make_synthetic("synth_torus")


def test_pc1_pc2_angle_reads_a_ring():
    """The classical ordination phase: on a genuine ring it is exactly right."""
    Z, angles = _ring(24)
    order, _ = iv.m_pc1_pc2_angle(Z, pairwise_distances(Z, metric="correlation"))
    assert _order_score(angles, order) >= 0.9


def test_mds_angle_reads_a_ring():
    Z, angles = _ring(24)
    order, _ = iv.m_mds_angle(Z, pairwise_distances(Z, metric="correlation"))
    assert _order_score(angles, order) >= 0.9


def test_circle_fit_angle_reads_a_ring():
    Z, angles = _ring(24, noise=0.5)
    order, _ = iv.m_circle_fit_angle(Z, pairwise_distances(Z, metric="correlation"))
    assert _order_score(angles, order) >= 0.9


def test_h1_cycle_reads_a_ring():
    """The topological order, on a ring, is the ring."""
    X, phases, _ = iv.make_synthetic("synth_ring", n_genes=800, seed=11)
    Z = standardize(X)
    order, _ = iv.m_h1_cycle(Z, pairwise_distances(Z, metric="correlation"))
    assert order.size >= 12
    assert _order_score(phases, order) >= 0.9


def test_h1_cycle_returns_nothing_for_a_pure_noise_cloud():
    X, _, _ = iv.make_synthetic("synth_noise", n_genes=300, seed=5)
    Z = standardize(X)
    order, _ = iv.m_h1_cycle(Z, pairwise_distances(Z, metric="correlation"))
    assert order.size < X.shape[0]


def test_two_opt_never_lengthens_the_greedy_tour():
    rng = np.random.default_rng(0)
    Z = standardize(rng.normal(size=(14, 6)))
    D = pairwise_distances(Z)
    greedy, _ = iv.m_greedy_tour(Z, D)
    refined, info = iv.m_tsp_2opt(Z, D)
    assert iv._tour_length(D, refined) <= iv._tour_length(D, greedy) + 1e-9
    assert info["tour_length"] == pytest.approx(iv._tour_length(D, refined), abs=1e-3)
    assert sorted(refined.tolist()) == list(range(14))


def test_classical_mds_reproduces_a_euclidean_distance_matrix():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(12, 3))
    D = pairwise_distances(pts)
    emb = iv._mds(D, 3)
    assert np.allclose(pairwise_distances(emb), D, atol=1e-8)


def test_permutation_null_is_ordered_and_in_range():
    _, angles = _ring(12)
    order = np.arange(12)
    null = iv.permutation_null(angles, order, 1.0, n_perm=400, seed=0)
    assert null["median"] <= null["q95"] <= 1.0
    assert 0.0 <= null["median"]
    assert null["p_value"] <= 1.0


def test_score_order_refuses_a_cycle_that_is_too_short():
    _, angles = _ring(12)
    rec = iv.score_order(angles, np.arange(3), 1.0, n_perm=100, seed=0)
    assert rec["declares"] is False
    assert rec["null"] is None
    assert rec["n_visited"] == 3


def test_score_order_declares_a_perfect_cycle():
    _, angles = _ring(12)
    rec = iv.score_order(angles, np.arange(12), 1.0, n_perm=500, seed=0)
    assert rec["declares"] is True
    assert rec["null"]["q95"] < 1.0


def test_every_registered_method_returns_a_valid_order():
    """Every method must name real samples exactly once -- except the H1 cycle,
    which names the subset of samples that lie on the loop."""
    X, _, _ = iv.make_synthetic("synth_ring", n_genes=120, n_samples=10, seed=2)
    Z = standardize(X)
    D = pairwise_distances(Z, metric="correlation")
    for name, fn in iv.METHODS.items():
        order, extra = fn(Z, D)
        idx = np.asarray(order, dtype=int)
        assert isinstance(extra, dict), name
        assert np.all((idx >= 0) & (idx < 10)), name
        if name == "h1_cycle":
            assert idx.size <= 10, name
        else:
            assert sorted(idx.tolist()) == list(range(10)), name


# --------------------------------------------------------------------------
# cycle_counting_real: the cloud-building helpers
# --------------------------------------------------------------------------
def test_centre_by_group_removes_each_group_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 5))
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    out = ccr.centre_by_group(X, groups)
    for g in ("a", "b", "c"):
        rows = [i for i, k in enumerate(groups) if k == g]
        assert np.allclose(out[rows].mean(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(X, out)


def test_centre_by_group_leaves_input_untouched():
    X = np.ones((4, 3))
    before = X.copy()
    ccr.centre_by_group(X, ["a", "a", "b", "b"])
    assert np.array_equal(X, before)


def test_collapse_returns_one_centroid_per_key():
    X = np.array([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [14.0, 14.0]])
    out, keys = ccr.collapse(X, ["p", "p", "q", "q"])
    assert out.shape == (2, 2)
    assert keys == ["p", "q"]
    assert np.allclose(out[0], [2.0, 2.0])
    assert np.allclose(out[1], [12.0, 12.0])


def test_collapse_keeps_rows_that_share_no_key():
    X = np.arange(12.0).reshape(4, 3)
    out, keys = ccr.collapse(X, list("abcd"))
    assert out.shape == (4, 3)
    assert np.array_equal(out, X)


def test_circular_distance_wraps_at_the_period():
    a = np.array([0.0, np.pi])
    D = ccr.circular_distance(a, 2.0 * np.pi)
    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(np.pi)
    # 0 and 2*pi - eps are neighbours on the circle, not a full period apart
    D2 = ccr.circular_distance(np.array([0.0, 2.0 * np.pi - 0.01]), 2.0 * np.pi)
    assert D2[0, 1] == pytest.approx(0.01, abs=1e-9)


# --------------------------------------------------------------------------
# cycle_identifiability: the decision rule and its scoring
# --------------------------------------------------------------------------
def test_succession_is_one_for_the_true_cyclic_order():
    phases = 2.0 * np.pi * np.arange(12) / 12.0
    assert ci.succession(phases, np.arange(12), n_perm=500)["score"] == 1.0


def test_succession_is_worse_for_a_scrambled_order():
    phases = 2.0 * np.pi * np.arange(12) / 12.0
    good = ci.succession(phases, np.arange(12), n_perm=500)["score"]
    bad = ci.succession(phases, np.random.default_rng(0).permutation(12),
                        n_perm=500)["score"]
    assert good > bad


def test_succession_refuses_an_order_that_is_too_short():
    phases = 2.0 * np.pi * np.arange(12) / 12.0
    rec = ci.succession(phases, np.array([0, 1]))
    assert rec["score"] == 0.0 and rec["passes"] is False


# --------------------------------------------------------------------------
# prepare_cycle_designs: the sample-sheet parsers
# --------------------------------------------------------------------------
def test_day_of_year_matches_the_calendar():
    assert pcd._day_of_year(2011, 9, 7) == 250
    assert pcd._day_of_year(2011, 1, 1) == 1
    assert pcd._day_of_year(2012, 3, 1) == 61        # 2012 is a leap year
    assert pcd._day_of_year(2011, 3, 1) == 60


def test_clock_hours_parses_and_survives_junk():
    assert pcd._clock_hours("22:15") == pytest.approx(22.25)
    assert pcd._clock_hours("05:00") == pytest.approx(5.0)
    assert pcd._clock_hours("7") == pytest.approx(7.0)
    assert pcd._clock_hours("NA") is None
    assert pcd._clock_hours("") is None
    assert pcd._clock_hours("ZT05") is None


def test_annual_names_give_a_monotone_index_across_a_year_boundary():
    meta = pcd._parse_annual_names(["CCTO44_2010_27_oct", "CCTO44_2011_15_jan"])
    assert [m["group"] for m in meta] == ["CCTO44", "CCTO44"]
    idx = [m["axis2_value"] for m in meta]
    assert idx[1] > idx[0], "the index must not wrap when the year changes"
    assert 0.0 <= meta[0]["axis1_value"] < 2.0 * np.pi


def test_annual_names_reject_an_unexpected_format():
    with pytest.raises(ValueError):
        pcd._parse_annual_names(["not-a-sample"])


def test_diurnal_names_map_the_second_day_onto_the_same_clock():
    meta = pcd._parse_diurnal_names(["1A04", "1A28"])
    assert meta[0]["group"] == meta[1]["group"] == "1A"
    # 04:00 and 28:00 are the same time of day, but different positions
    assert meta[0]["axis1_value"] == pytest.approx(meta[1]["axis1_value"])
    assert meta[1]["axis2_value"] - meta[0]["axis2_value"] == 24.0


def test_diurnal_names_reject_an_unexpected_format():
    with pytest.raises(ValueError):
        pcd._parse_diurnal_names(["ZZ99"])


def test_the_design_registry_declares_every_axis_before_the_data():
    """The registry is written before the data are seen, so it must be complete."""
    for key, spec in pcd.DESIGNS.items():
        # GEO series for most designs; the oestrous x circadian grid is an
        # INSDC BioProject (PRJNA...) because that is how the authors released
        # it -- both are public archival accessions, so accept either prefix.
        assert (spec["accession"].startswith("GSE")
                or spec["accession"].startswith("PRJNA")), key
        assert spec["n_cyclic_parameters"] >= 1, key
        assert len(spec["axes"]) >= 1, key
        assert spec["period_hours"] > 0, key
        assert isinstance(spec["design_note"], str) and spec["design_note"], key
    # the pooled design is built from the other two rather than downloaded, so
    # it lives beside the registry instead of inside it
    assert isinstance(pcd.POOLED_KEY, str) and pcd.POOLED_KEY
    assert pcd.POOLED_KEY not in pcd.DESIGNS
    assert "human_forced_desync" in pcd.DESIGNS


def test_cell_cycle_names_give_a_phase_a_genotype_and_a_replicate():
    """GSE8799's coordinate is a phase of the cell cycle, not a clock reading."""
    period = pcd.DESIGNS["yeast_cell_cycle"]["period_hours"] * 60.0
    meta = pcd._parse_yeast_cc_names(
        ["GSM1", "GSM2", "GSM3", "GSM4"],
        ["Cerevisiae_WildType_30min_rep1",
         "Cerevisiae_WildType_70min_rep1",
         "Cerevisiae_CyclinMutant_30min_rep2",
         "Cerevisiae_WildType_150min_rep1"],
        period)
    assert all(0.0 <= m["axis1_value"] < 2.0 * np.pi for m in meta)
    assert meta[0]["axis1_value"] == pytest.approx(0.0)       # earliest time, phase 0
    assert meta[1]["axis1_value"] != pytest.approx(meta[0]["axis1_value"])  # phase moves
    # the coordinate is a *phase*: 30 and 30 + 120 min are the same point on the circle,
    # so a criterion that read the clock would see two times where the circle has one
    assert meta[3]["axis1_value"] == pytest.approx(meta[0]["axis1_value"])
    # genotype and replicate ride alongside the phase without touching it
    assert meta[2]["axis1_value"] == pytest.approx(meta[0]["axis1_value"])
    assert [m["axis2_value"] for m in meta] == [0.0, 0.0, 1.0, 0.0]
    assert meta[0]["group"] == "WildType_rep1"
    assert meta[2]["group"] == "CyclinMutant_rep2"
    assert meta[0]["series"] == "WildType@30min"


def test_cell_cycle_names_reject_an_unexpected_format():
    with pytest.raises(ValueError):
        pcd._parse_yeast_cc_names(["x"], ["GSM1_WildType_30min_rep1"], 120.0)


def test_the_cell_cycle_design_records_three_axes_and_one_cycle():
    """The whole point of this design: axis count and cycle count differ."""
    spec = pcd.DESIGNS["yeast_cell_cycle"]
    assert spec["n_cyclic_parameters"] == 1
    assert spec["axes"] == ("cell_cycle_phase", "genotype")
    assert spec["period_hours"] == pytest.approx(2.0)
    # the third recorded axis (replicate) shows up in the sample grouping
    meta = pcd._parse_yeast_cc_names(
        ["a", "b"], ["Cerevisiae_WildType_30min_rep1",
                     "Cerevisiae_WildType_30min_rep2"], 120.0)
    assert len({m["group"] for m in meta}) == 2


# ---------------------------------------------------------------------------
# the oestrous x circadian grid: axis bookkeeping and the separability reading
# ---------------------------------------------------------------------------
def test_grid_phase_zt_maps_every_hour_onto_the_right_phase_and_day():
    """The released KEY sheet is the authority for the phase mapping."""
    for hour, phase in ((1, "diestrus"), (23, "diestrus"),
                        (25, "proestrus"), (47, "proestrus"),
                        (49, "oestrus"), (71, "oestrus"),
                        (73, "metestrus"), (96, "metestrus")):
        name, zt = pcd._grid_phase_zt(hour)
        assert name == phase, hour
        assert 1 <= zt <= 24, hour
    # the 96 h record is four consecutive 24 h days, so +24 h is the same ZT
    assert pcd._grid_phase_zt(5)[1] == pcd._grid_phase_zt(29)[1]
    assert pcd._grid_phase_zt(5)[0] != pcd._grid_phase_zt(29)[0]


def test_grid_phase_zt_rejects_hours_outside_the_recorded_window():
    for hour in (0, 97, 120):
        with pytest.raises(ValueError):
            pcd._grid_phase_zt(hour)


def test_the_grid_design_records_two_cyclic_axes_but_one_countable_cycle():
    """The design a count of axes would call 2 -- and a count of cycles calls 1."""
    spec = pcd.DESIGNS["estrous_circadian_grid"]
    assert spec["accession"].startswith("PRJNA")
    assert spec["axes"] == ("circadian_time", "oestrous_phase")
    assert len(spec["axes"]) == 2
    assert spec["n_cyclic_parameters"] == 1
    assert spec["period_hours"] == pytest.approx(24.0)
    assert spec["cache_names"]["fpkm"].endswith(".xlsx")


def test_grid_rows_are_one_continuous_96h_trajectory():
    """Odd hours 1..95 are sampled; each phase covers four consecutive such hours."""
    hours = list(range(1, 96, 2))            # the released grid is 2-hourly
    blocks = [pcd._grid_phase_zt(h)[0] for h in hours]
    # four phases, 24 hours each, in the order the KEY sheet gives
    assert blocks[:12] == ["diestrus"] * 12
    assert blocks[12:24] == ["proestrus"] * 12
    assert blocks[24:36] == ["oestrus"] * 12
    assert blocks[36:] == ["metestrus"] * 12
    # and the ZT position walks the day exactly once per phase
    for i in range(4):
        zts = [pcd._grid_phase_zt(h)[1] for h in hours[12 * i:12 * (i + 1)]]
        assert zts == list(range(1, 24, 2)), zts


def test_circular_distance_wraps_around_the_period():
    a = np.array([0.0, 1.0, 3.0])
    D = axsep.circular_distance(a, period=4.0)
    assert D[0, 2] == pytest.approx(1.0)      # |0-3| = 3, wrapped -> 1
    assert D[0, 1] == pytest.approx(1.0)
    assert np.allclose(np.diag(D), 0.0)


def test_axis_removal_drops_the_variance_it_explains():
    """Centring out an axis's own additive part must shrink the total variance."""
    rng = np.random.default_rng(0)
    n, p = 24, 50
    clock = np.repeat(np.arange(4), 6) * (np.pi / 2.0)
    groups = np.tile(["diestrus", "proestrus", "oestrus"], 8)
    X = rng.normal(size=(n, p))
    X[:, :20] += 3.0 * np.cos(clock)[:, None]          # a strong shared clock
    rec = axsep.axis_separability(X, clock,
                                 (np.arange(n) % 3).astype(float), groups, 0)
    assert rec["variance_explained_by_clock"] > 0.2
    assert 0.0 <= rec["subspace_cosine"] <= 1.0
    assert rec["additive_share_of_both"] > 1.0 - 1e-6


def test_matched_clock_distance_flags_a_degenerate_full_grid():
    """One sample per (clock, phase) cell makes the within-cell shuffle a no-op."""
    clock = np.tile(np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False), 4)
    groups = np.repeat(["d", "p", "e", "m"], 4)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 30))
    rec = axsep.matched_clock_distances(X, clock, groups, 0)
    assert rec["within_cell_permutation_degenerate"] is True
    assert rec["separation_p"] is None
    ci = rec["separation_bootstrap_ci95"]
    assert ci[0] is not None and ci[1] is not None and ci[0] <= ci[1]
