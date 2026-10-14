# Evidence

Every number on this page is produced by a script in `examples/` and committed
under `benchmark/`. None of it is a summary written by hand.

## 1. Does the topology add anything the cheaper methods cannot?

`examples/incremental_value.py` → `benchmark/incremental_value.json`

Eight label-free methods are given the same expression matrix and each must
produce a **cyclic order of the samples** without seeing the clock. The order is
then scored against the clock that was withheld
(`circular_succession_score`: the share of steps that stay at the current phase
or advance to the next one). A method *declares a cycle* when its score beats
the 95th percentile of its own phase-permutation null.

Twelve positive clouds (seven real circadian designs plus five draws of the
synthetic ring) and twenty-three negative clouds (three real "line" designs plus
five draws each of a linear ramp, pure noise, three clusters and a branching
trajectory).

| method | sensitivity | specificity | balanced accuracy |
|---|---|---|---|
| **h1 cycle** | **0.917** [0.65, 0.99] | **0.957** [0.79, 0.99] | **0.937** |
| circle fit angle | 0.917 | 0.609 | 0.763 |
| average-linkage leaf order | 0.917 | 0.609 | 0.763 |
| PC1-PC2 angle | 0.917 | 0.565 | 0.741 |
| classical MDS angle | 0.917 | 0.478 | 0.698 |
| greedy tour | 0.833 | 0.478 | 0.656 |
| 2-opt tour | 0.833 | 0.435 | 0.634 |
| PC1 order | 0.667 | 0.478 | 0.573 |

The topological cycle matches the best baseline on sensitivity and produces
**1 false positive in 23** against the best baseline's **9 in 23**. The
mechanism is not subtle: the ordination-based methods always return an order,
so they always return a score, and on a cloud with no cycle that score is
noise that clears its null often enough to matter. The topological method
returns an order only when there is a dominant H1 bar, and declines otherwise.

Two honest blemishes are visible in the same table: the topological method's
one false positive is a **branching** cloud (1 of 5 draws), and its one miss is
`ath_diurnal_ld`, the known 45-hour drift case.

## 2. When is a cycle count worth reporting?

`examples/cycle_counting_real.py` → `benchmark/cycle_counting_real.json`

Four real designs, all publicly available and all designed by other people
years ago, run through the identical pipeline as the simulated block:

| design | cycling parameter(s) | gap | circular Mantel r | count | expected |
|---|---|---|---|---|---|
| Douglas-fir yearlong (GSE93067) | annual | 4.93 | +0.70 | **1** | 1 |
| Douglas-fir 48 h (GSE93066) | daily | 1.36 | +0.70 | 0 | 1 |
| human forced desynchrony (GSE48113) | circadian | 1.07 | +0.05 | 0 | 1 |
| yearlong × daily, pooled | annual + daily | 1.09 | — | 0 | 2 |

One of four. The cause was then located rather than guessed:
`examples/cycle_counting_power.py` → `benchmark/cycle_counting_power.json`
sweeps how many of the 3000 selected genes actually carry the ring and shows
that the count succeeds only when essentially all of them do. At the measured
`r ≈ 0.70` the simulated gap is 1.2–1.4 and the measured diurnal gap is 1.36 —
the failing clouds sit exactly on the curve.

The yearlong series also supplies the cleanest positive result in the project:
sweeping the sampling window, every window short of a full year counts **0**,
and only windows that close the cycle count 1. "A partial period is an arc, not
a loop" therefore rests on a real 365-day cycle rather than on simulated data.

## 3. The decision rule

`examples/cycle_identifiability.py` → `benchmark/cycle_identifiability.json`

`omictopo.cycles.cycle_identifiability(gap, mantel_r)` turns the two diagnostics
into a verdict. Applied to the four designs it separates them correctly:

| design | gap | r | verdict |
|---|---|---|---|
| yearlong | 4.93 | +0.70 | **identifiable** — report the count |
| diurnal | 1.36 | +0.70 | underpowered — the ring is real, H1 cannot resolve it in this gene set |
| pooled | 1.22 | +0.59 | underpowered |
| forced desynchrony | 2.05 | +0.10 | inconsistent — a loop exists that does not follow the design axis |

`gap` alone says the spectrum is flat; `mantel_r` says *why*, and the two flat
cases have opposite diagnoses. That distinction is the useful part: it tells a
reader whether to go back to the experiment or back to the feature selection.

Two caveats, both measured. A large `gap` means one bar dominates, not that the
integer is right — the pooled design showed `gap = 9.3` and returned 1 against
a design containing 2. And the unguided dominant cycle recovered the diurnal
clock order **perfectly** (score 1.000) even though its bar was not dominant, so
the *ordering* claim survives in places where the *counting* claim does not.

## 4. A fix that was tried and rejected

`omictopo.cycles.topology_guided_cycle` alternates between finding the dominant
cycle and keeping the genes whose expression sweeps around it — no clock, fully
self-consistent. It was tested on the same four designs with a **negative
control built in**: the forced-desynchrony cloud, whose design axis is known not
to explain its distances (`r = +0.10`).

It failed on both counts.

* No improvement. Scored against the held-out clock, the recovered cycle moved
  by −0.06 to +0.08 across the four designs and turned no non-detection into a
  detection.
* Its calibration is invalid. The guided statistic beats a covariance-matched
  Gaussian null *with the whole procedure re-run inside the null* — and it beats
  it on the control cloud, where no cycle exists. Any cloud with coherent sample
  structure gets sharpened into an apparent cycle by the selection step.

The function stays in the package with that warning in its docstring, because
a documented failure is more useful than a missing one. The decision rule in
§3 does not depend on it.
