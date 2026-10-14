"""The axis-geometry diagnostic: pre-registered criteria and their calibration.

These tests pin three things, in increasing order of importance:

1. the four blocks reproduce the published ``benchmark/axis_separability.json``
   numbers bit-for-bit -- so lifting the analysis out of
   ``examples/axis_separability.py`` into ``omictopo.axes`` cannot have changed
   the result the repository already reports;
2. the pre-registered criteria classify a synthetic grid of *known* shape
   correctly, for all three shapes (independent / modulated / inert) -- the
   calibration is the only place the ground truth exists;
3. the criteria refuse instead of guessing when the design does not separate
   the possibilities, and the negative result about which quantities *cannot*
   carry the decision is asserted rather than left implicit.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from omictopo.legacy_axes import (  # noqa: E402
    AXIS_VERDICTS,
    CRITERIA,
    axis_geometry,
    classify_axis,
    harmonic_amplitude,
    summarise_axis,
)

_ROOT = os.path.join(os.path.dirname(__file__), "..", "benchmark")


def _grid(n1, n2):
    a1 = np.repeat(np.arange(n1), n2)
    a2 = np.tile(np.arange(n2), n1)
    return (2.0 * np.pi * a1 / n1, 2.0 * np.pi * a2 / n2, a1, a2)


def _synth(kind, n1=8, n2=8, n_genes=200, noise=0.3, seed=0):
    """The same three constructions the calibration script uses."""
    rng = np.random.default_rng(seed)
    t1, t2, _a1, a2 = _grid(n1, n2)
    n = t1.size
    X = np.zeros((n, n_genes))
    if kind == "independent":
        for j in range(n_genes):
            X[:, j] = (np.cos(t1) * rng.normal() + np.sin(t1) * rng.normal()
                       + np.cos(t2) * rng.normal() + np.sin(t2) * rng.normal())
    else:
        off = (np.linspace(-0.5, 0.5, n2)[a2] if kind == "modulated"
               else np.zeros(n))
        for j in range(n_genes):
            ph = t1 + off
            X[:, j] = np.cos(ph) * rng.normal() + np.sin(ph) * rng.normal()
    X += rng.standard_normal(X.shape) * noise
    return X, t1, t2, a2


# ---------------------------------------------------------------------------
# 1. harmonic_amplitude is the COSINOR quantity
# ---------------------------------------------------------------------------
def test_harmonic_amplitude_recovers_a_known_sine():
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    amp, phase = harmonic_amplitude(3.0 * np.cos(th - 0.7), th)
    assert amp == pytest.approx(3.0, abs=1e-6)
    assert phase == pytest.approx(0.7, abs=1e-6)


def test_harmonic_amplitude_of_a_constant_is_zero():
    th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    amp, _ = harmonic_amplitude(np.full(32, 5.0), th)
    assert amp == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. the lifted code reproduces the published numbers
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.path.exists(os.path.join(_ROOT, "axis_separability.json")),
    reason="published axis_separability.json not present")
def test_published_axis_separability_numbers_are_reproduced():
    """Every block that has a published counterpart must match it exactly.

    The published record was produced by ``examples/axis_separability.py``,
    which hard-codes this one design.  The generic module must return the same
    four blocks on the same input -- otherwise the refactor changed a result
    that is already in the paper.
    """
    try:
        from prepare_cycle_designs import load
    except Exception:                                    # pragma: no cover
        pytest.skip("examples/prepare_cycle_designs.py unavailable")
    # the prepared designs live next to the repository, in the session
    # workspace that ran prepare_cycle_designs.py; allow both layouts
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data",
                     "cycle_designs"),
        os.path.join(os.path.dirname(__file__), "..", "data", "cycle_designs"),
    ]
    env_root = os.environ.get("OMICTOPO_DESIGNS")
    if env_root:
        candidates.insert(0, env_root)
    root = next((c for c in candidates
                 if os.path.isdir(os.path.join(c, "estrous_circadian_grid"))),
                None)
    if root is None:
        pytest.skip("estrous_circadian_grid data not prepared on this machine")

    X, coords, _features, _info = load("estrous_circadian_grid", root)
    clock = np.array([c["axis1_value"] for c in coords], dtype=float)
    oes = np.array([c["axis2_value"] for c in coords], dtype=float)
    groups = [c["group"] for c in coords]
    rec = axis_geometry(X, clock, oes, groups, seed=0)

    with open(os.path.join(_ROOT, "axis_separability.json"), encoding="utf-8") as fh:
        pub = json.load(fh)

    q1, q2 = pub["q1_dominant_axis"], pub["q2_additive_separability"]
    q3, q4 = pub["q3_phase_vs_shape"], pub["q4_matched_clock_distance"]

    assert rec["dominant"]["mantel_r_axis1"] == q1["mantel_r_clock"]
    assert rec["dominant"]["mantel_r_axis2"] == q1["mantel_r_oestrous"]
    assert (rec["dominant"]["amplitude_ratio_axis1_over_axis2"]
            == q1["amplitude_ratio_clock_over_oestrous"])
    assert rec["separable"]["additive_share_of_both"] == q2["additive_share_of_both"]
    assert rec["separable"]["subspace_cosine"] == q2["subspace_cosine"]
    assert rec["phase_or_shape"]["block_offset_coherence"] == q3["block_offset_coherence"]
    assert (rec["phase_or_shape"]["max_block_offset_in_quarter_periods"]
            == q3["max_block_offset_in_quarter_periods"])
    assert rec["matched"]["separation_ratio"] == q4["separation_ratio"]
    assert (rec["matched"]["separation_bootstrap_ci95"]
            == q4["separation_bootstrap_ci95"])


# ---------------------------------------------------------------------------
# 3. the three known-shape calibrations classify correctly
# ---------------------------------------------------------------------------
def test_independent_axes_calibration_is_called_independent():
    X, t1, t2, a2 = _synth("independent", seed=0)
    rec = axis_geometry(X, t1, t2, a2, n_perm=500, seed=0)
    assert rec["verdict"] == "independent_axes"
    assert rec["dominant"]["mantel_r_axis2"] > CRITERIA["r_strong"]


def test_modulated_clock_calibration_is_called_a_modulation():
    X, t1, t2, a2 = _synth("modulated", seed=0)
    rec = axis_geometry(X, t1, t2, a2, n_perm=500, seed=0)
    assert rec["verdict"] == "modulation"
    # the axis produces no Mantel effect of its own, yet it moves the points
    assert rec["dominant"]["mantel_r_axis2"] < CRITERIA["r_inert"]
    assert (rec["matched"]["separation_ratio"]
            >= CRITERIA["ratio_modulated"])


def test_inert_axis_calibration_is_not_called_independent():
    X, t1, t2, a2 = _synth("inert", seed=0)
    rec = axis_geometry(X, t1, t2, a2, n_perm=500, seed=0)
    assert rec["verdict"] != "independent_axes"
    assert rec["verdict"] == "independent_axes_inert"


def test_the_three_calibrations_get_three_different_verdicts():
    """The whole point: three shapes, three different answers."""
    got = {kind: axis_geometry(*_synth(kind, seed=0), n_perm=500, seed=0)["verdict"]
           for kind in ("independent", "modulated", "inert")}
    assert set(got.values()) == {"independent_axes", "modulation",
                                 "independent_axes_inert"}


# ---------------------------------------------------------------------------
# 4. the negative result about the separability block
# ---------------------------------------------------------------------------
def test_separability_block_cannot_distinguish_independent_from_inert():
    """An inert axis is *trivially* orthogonal, so share/cosine cannot decide.

    This is a documented negative result, not a bug: it is the reason the
    criteria let the Mantel r carry the decision.  If this ever starts failing
    the criteria can be simplified -- until then, asserting it keeps the
    docstring honest.
    """
    sep = {}
    for kind in ("independent", "inert"):
        rec = axis_geometry(*_synth(kind, seed=0), n_perm=300, seed=0)
        sep[kind] = rec["separable"]
    assert sep["independent"]["additive_share_of_both"] == pytest.approx(1.0)
    assert sep["inert"]["additive_share_of_both"] == pytest.approx(1.0)
    assert abs(sep["independent"]["subspace_cosine"]) < 0.2
    assert abs(sep["inert"]["subspace_cosine"]) < 0.2


# ---------------------------------------------------------------------------
# 5. the criteria refuse rather than guess
# ---------------------------------------------------------------------------
def test_an_axis_in_the_band_between_the_cuts_is_undetermined():
    """r2 between r_inert and r_strong must not be forced into a verdict."""
    dominant = {"mantel_r_axis2": 0.3}
    verdict, reason = classify_axis(dominant, {}, {}, {})
    assert verdict == "undetermined"
    assert "0.300" in reason


def test_a_large_coherent_shift_is_undetermined_not_a_modulation():
    """A shift near half a quarter-period could be a second loop; refuse."""
    dominant = {"mantel_r_axis2": 0.0}
    phase = {"block_offset_coherence": 0.95,
             "max_block_offset_in_quarter_periods": 0.8}
    verdict, _ = classify_axis(dominant, {}, phase,
                              {"separation_ratio": 0.5})
    assert verdict == "undetermined"


def test_a_missing_axis_is_undetermined():
    verdict, reason = classify_axis({"mantel_r_axis2": None}, {}, {}, {})
    assert verdict == "undetermined"
    assert "no measurable geometry" in reason


def test_every_verdict_is_one_of_the_declared_four():
    for kind in ("independent", "modulated", "inert"):
        rec = axis_geometry(*_synth(kind, seed=0), n_perm=300, seed=0)
        assert rec["verdict"] in AXIS_VERDICTS
        assert summarise_axis(rec)


# ---------------------------------------------------------------------------
# 6. input validation and the categorical-axis path
# ---------------------------------------------------------------------------
def test_axis_must_have_one_entry_per_sample():
    X, t1, t2, a2 = _synth("inert", n1=4, n2=4, n_genes=20, seed=0)
    with pytest.raises(ValueError, match="one entry per sample"):
        axis_geometry(X, t1[:-1], t2, a2)
    with pytest.raises(ValueError, match="one entry per sample"):
        axis_geometry(X, t1, t2[:-1], a2)


def test_a_categorical_axis_is_accepted_as_the_axis_under_test():
    """axis2 may be a block label rather than a circle; it must still run."""
    X, t1, _t2, a2 = _synth("inert", n1=8, n2=4, n_genes=40, seed=0)
    labels = np.array(["a", "b", "c", "d"])[a2]
    rec = axis_geometry(X, t1, np.arange(X.shape[0]), labels, n_perm=200,
                        seed=0)
    assert rec["n_axis2_levels"] == 4
    assert rec["separable"]["levels_axis2"] == ["a", "b", "c", "d"]


def test_the_record_carries_its_criteria_so_the_cut_can_be_moved():
    X, t1, t2, a2 = _synth("independent", n1=6, n2=6, n_genes=40, seed=0)
    rec = axis_geometry(X, t1, t2, a2, n_perm=200, seed=0)
    assert rec["criteria"]["r_strong"] == CRITERIA["r_strong"]
    assert set(rec["settings"]) >= {"seed", "n_perm", "n_boot"}
