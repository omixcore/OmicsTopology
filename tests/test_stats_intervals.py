"""Intervals and exact tests: the small-count machinery the claims rest on.

Every headline number in this project is a proportion over a handful of
clouds — eight of ten, six of seven, one of four.  A proportion quoted without
an interval reads as far more certain than those counts support, so the
interval and the exact test are part of the result, not decoration, and they
are checked here against values that can be worked out by hand.
"""

import numpy as np
import pytest

from omictopo.stats import binom_test_two_sided, wilson_interval


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------
def test_wilson_matches_hand_computed_values():
    # 5/10: symmetric about 0.5.  The Wilson interval is wider than the
    # textbook normal interval but nowhere near [0, 1].
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=5e-4)
    assert high == pytest.approx(0.7634, abs=5e-4)
    assert low < 0.5 < high


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert 0.0 < high < 0.35
    low, high = wilson_interval(10, 10)
    assert high == 1.0
    assert 0.65 < low < 1.0


def test_wilson_narrows_as_n_grows():
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_is_monotone_in_the_count():
    lows = [wilson_interval(k, 20)[0] for k in range(21)]
    assert all(b >= a for a, b in zip(lows, lows[1:]))


def test_wilson_handles_zero_trials():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_tighter_at_lower_confidence():
    wide = wilson_interval(5, 10, confidence=0.99)
    narrow = wilson_interval(5, 10, confidence=0.90)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------
# exact binomial test
# --------------------------------------------------------------------------
def test_exact_binomial_matches_the_hand_computation():
    # 8 of 10 against a coin flip: 2 * (C(10,8)+C(10,9)+C(10,10)) / 2**10
    assert binom_test_two_sided(8, 10) == pytest.approx(0.109375, abs=1e-9)
    # 10 of 10: 2 / 2**10
    assert binom_test_two_sided(10, 10) == pytest.approx(0.001953125, abs=1e-9)
    assert binom_test_two_sided(0, 10) == pytest.approx(0.001953125, abs=1e-9)


def test_exact_binomial_is_one_at_the_centre():
    assert binom_test_two_sided(5, 10) == pytest.approx(1.0, abs=1e-9)
    assert binom_test_two_sided(50, 100) == pytest.approx(1.0, abs=1e-9)


def test_exact_binomial_is_capped_at_one():
    for k in range(11):
        assert binom_test_two_sided(k, 10) <= 1.0


def test_exact_binomial_decreases_with_the_count():
    p = [binom_test_two_sided(k, 12) for k in range(13)]
    assert p[6] > p[9] > p[12]
    assert p[6] > p[3] > p[0]


def test_exact_binomial_supports_a_non_half_null():
    # 5 of 5 against p = 0.9 is not surprising; against p = 0.1 it is
    assert binom_test_two_sided(5, 5, p=0.1) < 1e-4
    assert binom_test_two_sided(5, 5, p=0.9) > 0.5


def test_exact_binomial_handles_degenerate_inputs():
    assert binom_test_two_sided(0, 0) == 1.0
    assert binom_test_two_sided(3, 10, p=0.0) == 1.0
    assert binom_test_two_sided(3, 10, p=1.0) == 1.0


def test_the_headline_audit_number_is_not_significant():
    """8 of 10 sounds decisive and is not — this is why the interval is quoted."""
    assert binom_test_two_sided(8, 10) > 0.05
    low, high = wilson_interval(8, 10)
    assert low < 0.5 < high


def test_wilson_interval_accepts_numpy_integers():
    assert wilson_interval(np.int64(8), np.int64(10)) == wilson_interval(8, 10)
