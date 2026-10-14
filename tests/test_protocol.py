"""The protocol is a claim about *decisions*, so the decisions are tested.

Three things are checked here and nowhere else:

1. the protocol's counting primitives are **bit-identical** to the ones the
   published benchmark numbers came from, so a protocol verdict can never
   drift away from the results it claims to summarise;
2. the four statuses fire for the clouds they are meant to fire for, including
   the ones where the answer is a refusal;
3. the declared envelope still matches the committed benchmark records, so the
   boundary cannot silently become stale.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_EXAMPLES = _ROOT / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import topology_without_labels as twl  # noqa: E402

from omictopo import protocol  # noqa: E402
from omictopo.cycles import count_cycles, covariance_matched_null  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ring_cloud(n=48, n_genes=1200, noise=0.35, signal_frac=1.0, seed=0):
    """A clean ring: ``n`` samples around one full period."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    n_sig = max(1, int(round(n_genes * signal_frac)))
    X = rng.normal(0.0, noise, (n, n_genes))
    X[:, :n_sig] += np.cos(t[:, None] + rng.uniform(0.0, 2.0 * np.pi, n_sig))
    return X, t


# ---------------------------------------------------------------------------
# 1. identical to the published primitives
# ---------------------------------------------------------------------------
def test_package_count_cycles_is_bit_identical_to_the_benchmark_one():
    """The protocol must not re-derive the number it is summarising."""
    X, _ = ring_cloud(n=36, n_genes=400, seed=7)
    mine = count_cycles(X, metric="correlation", n_null=25, seed=3)
    theirs = twl.count_cycles(X, metric="correlation", n_null=25, seed=3)
    assert mine == theirs


def test_package_null_is_bit_identical_to_the_benchmark_one():
    import multi_dataset_benchmark as mdb

    X, _ = ring_cloud(n=30, n_genes=300, seed=11)
    from omictopo.geometry import standardize

    Z = standardize(X)
    mine = covariance_matched_null(Z, metric="correlation", n_draws=15, seed=5)
    theirs = mdb.covariance_matched_null(Z, metric="correlation", n_draws=15,
                                         seed=5)
    assert np.array_equal(mine, theirs)


# ---------------------------------------------------------------------------
# 2. the statuses
# ---------------------------------------------------------------------------
def test_a_clean_ring_that_follows_its_axis_is_counted():
    X, t = ring_cloud(n=48, n_genes=1200, seed=1)
    rec = protocol.analyse(X, axis=t, period=2.0 * np.pi, n_null=60, n_perm=400,
                           seed=0)
    assert rec["status"] == "counted"
    assert rec["n_cycles"] >= 1
    # a perfectly clean loop has a *single* H1 bar, so the gap is infinite and
    # is reported as None -- the most dominant spectrum there is
    assert rec["gap"] is None or rec["gap"] >= 2.0
    assert len(rec["top_bars_rel"]) == 1 or rec["gap"] >= 2.0
    assert rec["mantel_r"] > 0.5


def test_a_diluted_ring_is_refused_not_counted():
    """Exactly the measured failure mode: the loop is real, H1 cannot see it."""
    X, t = ring_cloud(n=48, n_genes=1200, noise=1.0, signal_frac=0.02, seed=2)
    rec = protocol.analyse(X, axis=t, period=2.0 * np.pi, n_null=60, n_perm=400,
                           seed=0)
    assert rec["status"] in ("underpowered", "no_cycle")
    assert protocol.summarise(rec).startswith("no count")


def test_a_dominant_loop_that_ignores_the_axis_is_inconsistent():
    X, t = ring_cloud(n=48, n_genes=1200, seed=3)
    rng = np.random.default_rng(99)
    rec = protocol.analyse(X, axis=rng.uniform(0.0, 2.0 * np.pi, X.shape[0]),
                           period=2.0 * np.pi, n_null=60, n_perm=400, seed=0)
    assert rec["status"] == "inconsistent"
    assert protocol.summarise(rec).startswith("no count")


def test_label_free_analysis_can_never_be_inconsistent():
    """Without an axis there is nothing for the loop to be inconsistent with."""
    X, _ = ring_cloud(n=36, n_genes=800, seed=4)
    seen = set()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        noisy = X + rng.normal(0.0, 1.0, X.shape)
        rec = protocol.analyse(noisy, n_null=40, seed=seed)
        assert rec["evidence"] == "geometry only (label-free)"
        assert rec["mantel_r"] is None
        seen.add(rec["status"])
    assert "inconsistent" not in seen


def test_no_status_other_than_counted_ever_quotes_a_number():
    X, t = ring_cloud(n=48, n_genes=1200, noise=1.0, signal_frac=0.02, seed=5)
    for axis in (None, t):
        rec = protocol.analyse(X, axis, period=2.0 * np.pi, n_null=40,
                               n_perm=200, seed=0)
        line = protocol.summarise(rec)
        if rec["status"] == "counted":
            assert line.startswith("counted")
        else:
            assert not line.startswith("counted")
            assert str(rec["n_cycles"]) not in line.split("(")[0]


def test_analyse_records_every_setting_it_used():
    X, t = ring_cloud(n=30, n_genes=300, seed=6)
    rec = protocol.analyse(X, axis=t, period=2.0 * np.pi, n_null=20, n_perm=100,
                           seed=13)
    assert rec["settings"] == {"metric": "correlation", "n_null": 20,
                               "n_perm": 100, "seed": 13,
                               "gap_threshold": 2.0, "r_threshold": 0.5}
    assert rec["status"] in protocol.STATUSES


def test_axis_length_is_checked():
    X, t = ring_cloud(n=20, n_genes=100, seed=8)
    with pytest.raises(ValueError):
        protocol.analyse(X, axis=t[:-3], period=2.0 * np.pi, n_null=10)


def test_envelope_is_a_copy_not_a_handle():
    e1 = protocol.envelope()
    e1["measured"]["real_designs"] = 999
    assert protocol.envelope()["measured"]["real_designs"] == 6


# ---------------------------------------------------------------------------
# 3. the declared envelope still matches the committed records
# ---------------------------------------------------------------------------
def test_envelope_matches_the_cycle_counting_record():
    data = json.loads((_ROOT / "benchmark" / "cycle_counting_real.json")
                      .read_text(encoding="utf-8"))
    env = protocol.envelope()["measured"]
    assert env["real_designs"] == data["n_primary"]

    ok = sorted(round(data["results"][k]["gap"], 2)
                for k in data["results"]
                if data["results"][k].get("primary")
                and data["results"][k]["correct"])
    assert ok == env["gap_on_designs_counted_right"]

    bad = sorted(round(data["results"][k]["gap"], 2)
                 for k in data["results"]
                 if data["results"][k].get("primary")
                 and not data["results"][k]["correct"])
    assert bad == sorted(round(g, 2) for g in env["gap_on_designs_not_counted"])


def test_envelope_matches_the_two_cycle_search_record():
    data = json.loads((_ROOT / "benchmark" / "two_cycle_search.json")
                      .read_text(encoding="utf-8"))
    rows = data["grid_requirement"]["rows"]
    full = [r for r in rows if r["signal_fraction"] == 1.0 and r["correct"]]
    smallest = min(full, key=lambda r: r["n_samples"])
    env = protocol.envelope()["measured"]
    assert smallest["n_samples"] == env["two_parameter_grid_min_samples_at_full_signal"]
    assert smallest["grid"] == env["two_parameter_grid_grid"]


def test_envelope_lists_its_own_limits():
    env = protocol.envelope()
    assert env["requires"] and env["cannot_do"] and env["not_label_free"]
    assert any("grid" in s for s in env["requires"])
