"""The learned baselines decide whether the increment is real, so they are tested.

``examples/learned_baselines.py`` exists to close the strongest objection to the
project's headline claim — that the topological cycle only beats the baselines
because the baselines are weak transforms of the cloud.  If the learned methods
in there were quietly broken, they would lose for the wrong reason and the
comparison would be worthless, so the things that make them fair are asserted
here rather than asserted in prose: the coordinate view is lossless, the
autoencoder is deterministic and bounded, its linear special case really does
reproduce PCA, and every method returns an honest permutation on the cloud
sizes the benchmark actually uses.
"""

import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_EXAMPLES = _HERE.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import incremental_value as iv  # noqa: E402
import learned_baselines as lb  # noqa: E402

from omictopo.geometry import pairwise_distances, pca_embed, standardize  # noqa: E402


def _cloud(kind="synth_ring", n=24, p=400, seed=7):
    """A clock cloud in the benchmark's own geometry: standardised, correlation."""
    X, phases, hours = iv.make_synthetic(kind, n_genes=p, n_samples=n, seed=seed)
    Z = standardize(X)
    return Z, pairwise_distances(Z, metric="correlation"), phases


# --------------------------------------------------------------------------
# the coordinate view has to be lossless, or the learned methods are crippled
# --------------------------------------------------------------------------
def test_coordinate_view_preserves_every_distance():
    """It must be a rotation plus one global scale, nothing else."""
    Z, _, _ = _cloud(n=24, p=400)
    V = lb.coordinate_view(Z)
    D_z = pairwise_distances(Z)
    D_v = pairwise_distances(V)
    ratio = D_v[np.triu_indices_from(D_v, k=1)] / D_z[np.triu_indices_from(D_z, k=1)]
    assert np.allclose(ratio, ratio[0], rtol=1e-8)
    assert np.all(np.isfinite(V))


def test_coordinate_view_keeps_the_whole_row_space():
    for n in (12, 24, 40):
        Z, _, _ = _cloud(n=n, p=500)
        V = lb.coordinate_view(Z)
        assert V.shape == (n, min(200, n - 1, 500))


# --------------------------------------------------------------------------
# the autoencoder: the one method here with a large parameter count
# --------------------------------------------------------------------------
def test_linear_autoencoder_reproduces_pca():
    """The fairness check that is reported in the benchmark output.

    With the nonlinearity removed, a linear bottleneck spans the same subspace
    as PCA, so both the 1-D order and the 2-D angle must match the principal
    axes.  If this fails the learned baselines are broken, not honest.
    """
    check = lb.autoencoder_linear_check()
    assert check["linear_ae_matches_pca"] is True
    assert check["linear_ae_1d_score"] == pytest.approx(check["pca_1d_score"], abs=0.05)
    assert check["linear_ae_2d_score"] == pytest.approx(check["pca_2d_score"], abs=0.05)


def test_autoencoder_is_deterministic():
    Z, _, _ = _cloud(n=24, p=300)
    a = lb.autoencoder_latent(Z, latent=2, seed=0)[0]
    b = lb.autoencoder_latent(Z, latent=2, seed=0)[0]
    assert np.array_equal(a, b)


def test_autoencoder_code_stays_bounded():
    """Regression: the scale of the bottleneck is a free gauge.

    Before the bottleneck was normalised, the weight matrices could be rescaled
    by a constant without changing the loss, and Adam drifted along that flat
    direction — observed latent values of order 10^5 and no convergence.  The
    code must come out standardised and finite on every kind of cloud.
    """
    for kind in ("synth_ring", "synth_line", "synth_noise", "synth_clusters"):
        Z, _, _ = _cloud(kind=kind, n=24, p=400)
        code, info = lb.autoencoder_latent(Z, latent=2)
        assert np.all(np.isfinite(code)), kind
        assert float(np.abs(code).max()) < 50.0, kind
        assert np.allclose(code.mean(axis=0), 0.0, atol=1e-8), kind
        # the variance floor shrinks the code slightly below unit scale; it can
        # never inflate it, and it can never collapse it either
        std = code.std(axis=0)
        assert np.all(std <= 1.0 + 1e-9) and np.all(std > 0.5), (kind, std)
        assert info["n_params"] > 0


def test_autoencoder_reads_a_ring_and_not_a_line():
    """The learned 2-D phase is a real competitor, so it has to work.

    The hyperparameters were tuned on the positive controls, so this asserts
    exactly that tuning succeeded — a learned baseline that could not read the
    clock would make the comparison meaningless rather than favourable.
    """
    ring_scores, line_scores = [], []
    for seed in (7, 108, 209):
        Z, D, ph = _cloud("synth_ring", seed=seed)
        order, _ = lb.m_ae_2d_angle(Z, D)
        ring_scores.append(iv.circular_succession_score(ph, order))
        Z, D, ph = _cloud("synth_line", seed=seed)
        order, _ = lb.m_ae_2d_angle(Z, D)
        line_scores.append(iv.circular_succession_score(ph, order))
    assert min(ring_scores) >= 0.85, ring_scores
    assert max(line_scores) < min(ring_scores), (line_scores, ring_scores)


def test_angle_order_refuses_a_collapsed_embedding():
    assert lb._angle_order(np.zeros((10, 2))).size == 0
    assert lb._angle_order(np.zeros((10, 1))).size == 0
    assert lb._angle_order(np.array([[np.nan, 1.0]])).size == 0


# --------------------------------------------------------------------------
# every learned method, on the cloud sizes the benchmark uses
# --------------------------------------------------------------------------
def test_every_learned_method_returns_a_permutation():
    Z, D, _ = _cloud(n=24, p=400)
    for name, fn in lb.LEARNED_METHODS.items():
        order, extra = fn(Z, D)
        assert isinstance(extra, dict), name
        assert not extra.get("error"), (name, extra)
        assert sorted(np.asarray(order, dtype=int).tolist()) == list(range(24)), name


def test_every_learned_method_survives_the_smallest_cloud():
    """Twelve samples is the smallest cloud in the benchmark.

    A method is allowed to return nothing there — ``score_order`` treats that
    as silence and counts it, so it cannot buy specificity by crashing — but it
    may not raise, and what it does return must be a valid permutation.
    """
    Z, D, _ = _cloud(n=12, p=400)
    for name, fn in lb.LEARNED_METHODS.items():
        try:
            order, extra = fn(Z, D)
        except Exception as exc:                               # pragma: no cover
            pytest.fail(f"{name} raised on n=12: {type(exc).__name__}: {exc}")
        order = np.asarray(order, dtype=int)
        assert np.array_equal(np.sort(order), np.arange(12)), name


def test_principal_curve_keeps_every_sample():
    Z, D, _ = _cloud(n=24, p=400)
    order, extra = lb.m_principal_curve_order(Z, D)
    assert not extra.get("error")
    assert sorted(order.tolist()) == list(range(24))


def test_som_chain_length_tracks_the_cloud():
    Z, D, _ = _cloud(n=17, p=400)
    order, extra = lb.m_som_chain_order(Z, D)
    assert extra["n_units"] == 17
    assert sorted(order.tolist()) == list(range(17))


def test_neighbour_count_stays_usable_across_sizes():
    for n in (5, 12, 24, 72):
        k = lb._n_neighbours(n)
        assert 3 <= k <= min(15, n - 1)


# --------------------------------------------------------------------------
# wiring into the benchmark
# --------------------------------------------------------------------------
def test_families_cover_every_learned_method():
    assert set(lb.LEARNED_FAMILY) == set(lb.LEARNED_METHODS)
    assert set(lb.LEARNED_METHODS) <= set(iv.FAMILY)
    assert set(iv.METHODS) == set(iv.CLASSICAL_METHODS) | set(iv.LEARNED_METHODS)
    assert iv.FAMILY[iv.TOPOLOGICAL] == "topological"


def test_the_classical_registry_is_unchanged():
    """``--classical-only`` must still reproduce the published eight-method table."""
    assert set(iv.CLASSICAL_METHODS) == {
        "h1_cycle", "pc1_order", "pc1_pc2_angle", "circle_fit_angle",
        "mds_angle", "hc_leaf_order", "greedy_tour", "tsp_2opt"}


def test_only_the_autoencoders_carry_trainable_parameters():
    counts = lb.parameter_counts()
    assert counts["ae_1d_order"] > 0 and counts["ae_2d_angle"] > 0
    for name in lb.LEARNED_METHODS:
        if name.startswith("ae_"):
            continue
        assert counts[name] == 0, name


def test_a_raising_method_is_recorded_not_fatal():
    """Failure must be recorded, so a baseline cannot hide behind an exception."""
    Z, D, ph = _cloud(n=16, p=200)

    def boom(_Z, _D):
        raise RuntimeError("deliberate")

    registry = {"h1_cycle": iv.CLASSICAL_METHODS["h1_cycle"], "boom": boom}
    rec = iv.run_cloud("fake", Z, ph * 24.0 / (2.0 * np.pi), 24.0,
                       n_perm=50, methods=registry)
    assert rec["methods"]["boom"]["failed"] is True
    assert "deliberate" in rec["methods"]["boom"]["error"]
    assert rec["methods"]["boom"]["declares"] is False
    assert rec["methods"]["h1_cycle"]["failed"] is False

    rows = [dict(rec, group="negative", cloud="fake")]
    table = iv.summarise(rows, names=registry)
    assert table["boom"]["method_failures"] == 1
    assert table["boom"]["failed_clouds"] == ["fake"]


def test_summary_reports_family_and_mean_scores():
    Z, D, ph = _cloud(n=24, p=200)
    registry = {"h1_cycle": iv.CLASSICAL_METHODS["h1_cycle"],
                "ae_2d_angle": lb.LEARNED_METHODS["ae_2d_angle"]}
    times = ph * 24.0 / (2.0 * np.pi)
    rows = [dict(iv.run_cloud("pos", Z, times, 24.0, n_perm=50, methods=registry),
                 group="positive", cloud="pos"),
            dict(iv.run_cloud("neg", Z, times, 24.0, n_perm=50, methods=registry),
                 group="negative", cloud="neg")]
    table = iv.summarise(rows, names=registry)
    assert table["ae_2d_angle"]["family"] == "autoencoder"
    assert table["h1_cycle"]["family"] == "topological"
    assert table["h1_cycle"]["n_positive"] == 1
    assert table["ae_2d_angle"]["mean_seconds"] >= 0.0
    assert set(table["h1_cycle"]) >= {"sensitivity", "specificity",
                                      "balanced_accuracy", "method_failures"}
