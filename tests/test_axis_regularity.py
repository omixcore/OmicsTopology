"""The cross-dataset corpus: what it settled, and what it refuted.

``examples/axis_regularity.py`` turns the axis diagnostic from "works on one
design" into "here is what second axes are like across the corpus".  The corpus
disagreed with the hypothesis it was pre-registered against, and that
disagreement is one of the results -- so the tests here are written to make it
**hard to lose**:

1. the record is internally consistent (its counts match its rows);
2. the refuted hypothesis is still on the record, with its counterexample, and
   the counterexample is still above the threshold -- a future edit that quietly
   drops the heat design, or that retunes ``r_strong`` past 0.98, fails here;
3. the two RNA fractions of one experiment are still counted as one experiment;
4. the surviving ordering (displacement, not level count) still holds on the
   recorded numbers.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

_BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmark")
_RECORD = os.path.join(_BENCH, "axis_regularity.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_RECORD),
    reason="benchmark/axis_regularity.json not present")


def _record():
    with open(_RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def _real_by_key(rec):
    return {r["key"]: r for r in rec["real_designs"]}


def _plant_root():
    """Where prepare_plant_dataset.py wrote its output, if it did."""
    here = os.path.dirname(__file__)
    candidates = [
        os.environ.get("OMICTOPO_PLANT"),
        os.path.join(here, "..", "..", "data"),
        os.path.join(here, "..", "data"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "arabidopsis_heat")):
            return c
    return None


# ---------------------------------------------------------------------------
# 1. the record is consistent with itself
# ---------------------------------------------------------------------------
def test_the_tally_counts_the_rows_it_has():
    rec = _record()
    t = rec["tally"]
    assert t["n_real_axis_pairs"] == len(rec["real_designs"])
    assert (t["condition_axes"] + len([r for r in rec["real_designs"]
                                       if r["expectation"] == "coordinate"])
            == len(rec["real_designs"]))
    assert (t["condition_axes_below_threshold"]
            + len(t["condition_axes_above_threshold"]) == t["condition_axes"])
    assert t["n_organisms"] >= 4
    assert t["n_independent_experiments"] <= t["n_real_axis_pairs"]


def test_every_row_carries_the_four_blocks_and_its_criteria():
    rec = _record()
    for row in rec["real_designs"] + rec["calibrations"]:
        assert set(row) >= {"dominant", "separable", "phase_or_shape", "matched",
                            "verdict", "criteria", "expectation"}
        assert row["verdict"] in ("independent_axes", "independent_axes_inert",
                                  "modulation", "undetermined")


# ---------------------------------------------------------------------------
# 2. the refuted hypothesis stays refuted, and stays visible
# ---------------------------------------------------------------------------
def test_the_condition_axis_hypothesis_is_recorded_as_refuted():
    """R1 said a condition-shaped axis never reaches r_strong.  It is false.

    The assertion is deliberately written so that the *interesting* outcome --
    the refutation -- is what fails if it disappears.  If a future corpus makes
    R1 true, this test must be rewritten on purpose, not deleted by accident.
    """
    t = _record()["tally"]
    assert t["R1_holds"] is False
    assert "falsified" in t["R1_verdict"].lower()
    keys = {c["key"] for c in t["condition_axes_above_threshold"]}
    assert keys == {"arabidopsis_heat", "arabidopsis_TR"}


def test_the_counterexample_is_still_far_above_the_threshold():
    """A heat shock reaches r2 ~ 0.98; the threshold is 0.5, not 0.95."""
    rec = _record()
    rows = _real_by_key(rec)
    for key in ("arabidopsis_heat", "arabidopsis_TR"):
        r2 = rows[key]["dominant"]["mantel_r_axis2"]
        assert r2 > 0.9
        assert r2 > rec["criteria"]["r_strong"]
    # and it must stay a *condition* in the pre-registered table, not be
    # relabelled after the fact
    for key in ("arabidopsis_heat", "arabidopsis_TR"):
        assert rows[key]["expectation"] == "condition"


def test_the_refutation_is_not_an_artifact_of_the_blocking_label():
    """r2 comes from the dominant block, which never sees ``groups``."""
    rec = _record()
    rows = _real_by_key(rec)
    for key in ("arabidopsis_heat", "arabidopsis_TR"):
        sens = rows[key]["dominant"]["mantel_p_axis2"]
        assert sens <= 0.01
        assert rows[key]["dominant"]["mantel_r_axis2"] > 0.9


# ---------------------------------------------------------------------------
# 3. one experiment, two fractions
# ---------------------------------------------------------------------------
def test_the_two_rna_fractions_are_counted_as_one_experiment():
    t = _record()["tally"]
    assert ["arabidopsis_heat", "arabidopsis_TR"] in t["same_experiment_groups"]
    assert t["n_independent_experiments"] == t["n_real_axis_pairs"] - 1


# ---------------------------------------------------------------------------
# 4. the surviving ordering: displacement, not level count
# ---------------------------------------------------------------------------
def test_r2_is_not_ordered_by_the_number_of_axis2_levels():
    """Two levels span almost the whole range; a 16-level index is in between."""
    rec = _record()
    rows = rec["real_designs"]
    by_levels = {}
    for r in rows:
        by_levels.setdefault(len(r["axis2_levels"]), []).append(
            r["dominant"]["mantel_r_axis2"])
    two = by_levels[2]
    assert min(two) < 0.01 and max(two) > 0.9


def test_every_axis_that_moves_points_more_than_a_step_beats_every_axis_that_does_not():
    """The measured ordering, read straight off the displacement block."""
    rec = _record()
    disp = rec["tally"]["displacement_ordering"]
    over = [r["r2"] for r in disp["rows"] if r["separation_ratio"] >= 1.2]
    under = [r["r2"] for r in disp["rows"] if r["separation_ratio"] < 1.2]
    assert over and under
    assert min(over) > max(under)


def test_the_pooled_two_cycle_design_is_flagged_as_confounded():
    """The corpus's only candidate positive case is not a clean one."""
    t = _record()["tally"]
    assert "confounded" in t["R2_caveat"]
    rows = _real_by_key(_record())
    pooled = rows["psme_annual_x_diurnal"]
    assert pooled["verdict"] == "independent_axes"
    assert pooled["dominant"]["mantel_r_axis2"] >= _record()["criteria"]["r_strong"]


# ---------------------------------------------------------------------------
# 5. the counterexample reproduces from the data, when the data is here
# ---------------------------------------------------------------------------
def test_a_heat_treatment_reproduces_a_high_mantel_effect():
    """Recompute the counterexample rather than trusting the record."""
    root = _plant_root()
    if root is None:
        pytest.skip("arabidopsis_heat not prepared on this machine")
    from axis_regularity import _load_plant, _plant_axes
    from omictopo.legacy_axes import axis_geometry

    for key in ("arabidopsis_heat", "arabidopsis_TR"):
        X, hours, treat = _load_plant(root, key)
        a1, a2, groups, _levels = _plant_axes(hours, treat, "hour", "treatment")
        rec = axis_geometry(X, a1, a2, groups, n_perm=500, seed=0)
        assert rec["dominant"]["mantel_r_axis2"] > 0.9
        assert rec["verdict"] == "independent_axes"
