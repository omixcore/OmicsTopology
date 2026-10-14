"""The powered comparison has to be as honest as the benchmark it extends.

``examples/increment_power.py`` exists because 23 non-cyclic clouds cannot
separate 1 false positive from 2.  Its own arithmetic — rates with intervals,
the paired McNemar test, the required sample size, and the split between
producing no order and producing an unconvincing one — is checked here on
hand-built records, without touching the network or the data directory.
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
import increment_power as ip  # noqa: E402
import learned_baselines as lb  # noqa: E402


def _record(*, declares, ordered=True):
    return {"declares": bool(declares),
            "null": {"q95": 0.5} if ordered else None,
            "score": 0.9 if declares else 0.2}


def _cloud(name, **by_method):
    return {"cloud": name, "methods": by_method}


# --------------------------------------------------------------------------
# the field is the five methods worth spending 280 clouds on
# --------------------------------------------------------------------------
def test_the_field_is_the_five_methods_that_matter():
    assert set(ip.FIELD) == {"h1_cycle", "tsne_angle", "isomap_angle",
                             "circle_fit_angle", "ae_2d_angle"}
    for name, fn in ip.FIELD.items():
        assert callable(fn), name
    assert ip.FIELD["h1_cycle"] is iv.CLASSICAL_METHODS["h1_cycle"]
    assert ip.FIELD["tsne_angle"] is lb.LEARNED_METHODS["tsne_angle"]
    assert ip.FIELD["ae_2d_angle"] is lb.LEARNED_METHODS["ae_2d_angle"]


def test_the_control_families_are_the_benchmark_ones_unchanged():
    declared = [kind for kind, group, _ in iv.SYNTHETIC if group == "negative"]
    assert ip.NEGATIVE_KINDS == declared
    assert ip.POSITIVE_KIND == "synth_ring"


def test_its_draws_share_no_cloud_with_the_benchmark():
    """Independence is the reason 240 further clouds are more power, not more of
    the same 23."""
    benchmark_seeds = {7 + 101 * s for s in range(60)}
    power_seeds = {ip.NEG_SEED0 + ip.NEG_STEP * i for i in range(240)}
    assert not (benchmark_seeds & power_seeds)
    assert len(power_seeds) == 240


# --------------------------------------------------------------------------
# rates, intervals and the required sample size
# --------------------------------------------------------------------------
def test_rate_reports_the_count_and_a_valid_interval():
    r = ip._rate(3, 10)
    assert r["k"] == 3 and r["n"] == 10 and r["rate"] == 0.3
    lo, hi = r["ci95"]
    assert 0.0 <= lo <= 0.3 <= hi <= 1.0


def test_rate_survives_an_empty_group():
    r = ip._rate(0, 0)
    assert r["k"] == 0 and r["n"] == 0 and r["rate"] == 0.0


def test_required_n_is_positive_for_a_real_gap_and_none_for_a_tie():
    assert ip._required_n(0.04, 0.09) >= 100
    assert ip._required_n(0.04, 0.04) is None
    assert ip._required_n(0.0, 0.5) is None      # a boundary rate is not measurable
    assert ip._required_n(0.5, 1.0) is None


def test_required_n_shrinks_as_the_gap_grows():
    small = ip._required_n(0.04, 0.09)
    large = ip._required_n(0.04, 0.40)
    assert large < small


# --------------------------------------------------------------------------
# the paired test
# --------------------------------------------------------------------------
def test_mcnemar_counts_the_discordant_pairs():
    clouds = [
        _cloud("a", h1_cycle=_record(declares=True),
               tsne_angle=_record(declares=False)),
        _cloud("b", h1_cycle=_record(declares=False),
               tsne_angle=_record(declares=True)),
        _cloud("c", h1_cycle=_record(declares=True),
               tsne_angle=_record(declares=True)),
        _cloud("d", h1_cycle=_record(declares=False),
               tsne_angle=_record(declares=False)),
    ]
    mc = ip._mcnemar(clouds, "h1_cycle", "tsne_angle")
    assert mc["only_a"] == 1 and mc["only_b"] == 1
    assert mc["both"] == 1 and mc["discordant"] == 2
    assert mc["p_value"] == pytest.approx(1.0)     # 1 vs 1 is perfectly balanced


def test_mcnemar_calls_a_one_sided_run_significant():
    clouds = [_cloud(f"c{i}",
                     h1_cycle=_record(declares=False),
                     tsne_angle=_record(declares=True))
              for i in range(10)]
    mc = ip._mcnemar(clouds, "h1_cycle", "tsne_angle")
    assert mc["discordant"] == 10 and mc["only_b"] == 10
    assert mc["p_value"] < 0.01


# --------------------------------------------------------------------------
# the abstention split, which is the structural claim
# --------------------------------------------------------------------------
def test_summarise_separates_refusal_from_unconvincing_answers():
    records = [
        _cloud("refused", m=_record(declares=False, ordered=False)),
        _cloud("unconvincing", m=_record(declares=False)),
        _cloud("declared", m=_record(declares=True)),
    ]
    out = ip._summarise(records, "m")
    ab = out["abstention"]
    assert ab["no_order_at_all"]["k"] == 1
    assert ab["silent_by_score"]["k"] == 1
    assert out["rate"]["k"] == 1 and out["rate"]["n"] == 3


def test_summarise_counts_everything_when_a_method_always_answers():
    records = [_cloud(f"c{i}", m=_record(declares=(i % 2 == 0)))
               for i in range(6)]
    out = ip._summarise(records, "m")
    assert out["abstention"]["no_order_at_all"]["k"] == 0
    assert out["abstention"]["silent_by_score"]["k"] == 3
    assert out["rate"]["k"] == 3


def test_sensitivity_is_scored_on_the_positive_clouds_only():
    records = [_cloud("p1", m=_record(declares=True)),
               _cloud("p2", m=_record(declares=False))]
    out = ip._summarise(records, "m")
    assert out["rate"]["k"] == 1 and out["rate"]["n"] == 2
    assert out["rate"]["rate"] == 0.5
