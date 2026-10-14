# Evidence

Every number on this page is produced by a script in `examples/` and committed
under `benchmark/`. None of it is a summary written by hand.

## 1. Does the topology add anything the cheaper methods cannot?

`examples/incremental_value.py` → `benchmark/incremental_value.json`

Seventeen label-free methods are given the same expression matrix and each must
produce a **cyclic order of the samples** without seeing the clock. The order is
then scored against the clock that was withheld
(`circular_succession_score`: the share of steps that stay at the current phase
or advance to the next one). A method *declares a cycle* when its score beats
the 95th percentile of its own phase-permutation null.

Twelve positive clouds (seven real circadian designs plus five draws of the
synthetic ring) and twenty-three negative clouds (three real "line" designs plus
five draws each of a linear ramp, pure noise, three clusters and a branching
trajectory). Seven baselines are fixed transforms; nine are *trained* — a
two-layer autoencoder at two bottleneck widths (~850 parameters on a 24-sample
cloud, tuned on the positive controls alone), Isomap, LLE, kernel PCA, Laplacian
eigenmaps, t-SNE, a self-organising map and a principal curve.

| method | family | sensitivity | specificity | balanced accuracy |
|---|---|---|---|---|
| **h1 cycle** | topological | **0.917** [0.65, 0.99] | **0.957** [0.79, 0.99] | **0.937** |
| t-SNE angle | learned, neighbour | 0.917 | 0.913 | 0.915 |
| Isomap angle | learned, neighbour | 0.833 | 0.870 | 0.851 |
| circle fit angle | fixed transform | 0.917 | 0.609 | 0.763 |
| average-linkage leaf order | fixed transform | 0.917 | 0.609 | 0.763 |
| PC1-PC2 angle | fixed transform | 0.917 | 0.565 | 0.741 |
| self-organising map | learned | 0.833 | 0.609 | 0.721 |
| classical MDS angle | fixed transform | 0.917 | 0.478 | 0.698 |
| **autoencoder, 2-D** | learned, reconstruction | 0.917 | 0.478 | 0.698 |
| kernel PCA angle | learned, reconstruction | 0.917 | 0.478 | 0.698 |
| Laplacian eigenmaps | learned, reconstruction | 0.917 | 0.478 | 0.698 |
| principal curve | learned, 1-D manifold | 0.833 | 0.522 | 0.677 |
| greedy tour | fixed transform | 0.833 | 0.478 | 0.656 |
| 2-opt tour | fixed transform | 0.833 | 0.435 | 0.634 |
| locally linear embedding | learned, reconstruction | 0.833 | 0.435 | 0.634 |
| PC1 order | fixed transform | 0.667 | 0.478 | 0.573 |
| autoencoder, 1-D | learned, reconstruction | 0.583 | 0.435 | 0.509 |

Three things this table says, none of them flattering to the method proposed
here:

1. **Learning the manifold bought the autoencoder nothing.** Its specificity,
   0.478 with 12 of 23 false positives, is identical to a circle fitted to two
   principal axes. Its nonlinearity, its 857 parameters and its tuning on the
   positive controls changed neither axis of the comparison. Capacity was not
   what was missing.
2. **What separates the learned methods is fitting the cloud versus fitting its
   neighbourhoods.** Every reconstruction method — autoencoders, kernel PCA,
   Laplacian eigenmaps, LLE, principal curve, and fixed MDS — invents a ring on
   11-13 of the 23 non-cyclic clouds. The two that only preserve local
   neighbourhoods — t-SNE and Isomap — invent one on 2 and 3.
3. **The topology is first, but 1 against 2 is a tie at this sample size.**
   What the table *does* resolve is the mechanism of the abstention: on **5 of
   the 23** non-cyclic clouds the H1 cycle produces **no order at all**, while
   every baseline produces an order on all 23 and stays quiet only because its
   score missed its own null — t-SNE by **+0.042** on its worst cloud.

Two honest blemishes are visible in the same table: the topological method's
one false positive is a **branching** cloud (1 of 5 draws), and its one miss is
`ath_diurnal_ld`, the known 45-hour drift case.

### 1b. Is 1 against 2 real?  240 further clouds say yes

`examples/increment_power.py` → `benchmark/increment_power.json`

Twenty-three clouds cannot separate 1 false positive in 23 from 2 in 23 — the
two rates behind the ordering of the table. The four declared negative families
were therefore redrawn — 60 independent clouds each, on a seed schedule that
shares nothing with the benchmark's — and the five methods that mattered were
rerun on all 240, with 40 further rings for sensitivity.

| method | sensitivity (40 rings) | false positives (240) | produced no order |
|---|---|---|---|
| **h1 cycle** | 1.000 [0.91, 1.00] | **0.067** [0.04, 0.11] (16) | **0.271** [0.22, 0.33] (65) |
| t-SNE angle | 1.000 | 0.196 [0.15, 0.25] (47) | 0.000 (0) |
| Isomap angle | 1.000 | 0.292 [0.24, 0.35] (70) | 0.000 (0) |
| circle fit angle | 1.000 | 0.400 [0.34, 0.46] (96) | 0.000 (0) |
| autoencoder, 2-D | 1.000 | 0.442 [0.38, 0.50] (106) | 0.000 (0) |

Paired on the same clouds (McNemar, exact): the H1 cycle declares on **10**
clouds where t-SNE stays quiet and t-SNE declares on **41** where the H1 cycle
stays quiet, p = 1.5 × 10⁻⁵; against Isomap 7 / 61, p = 7.4 × 10⁻¹²; against the
circle fit 2 / 82, p = 3.7 × 10⁻²²; against the autoencoder 4 / 94,
p = 2.4 × 10⁻²³. The observed gap of 12.9 points would need 107 clouds per
method for 80 % power; this used 240.

Where the false positives come from matters as much as how many:

| false-positive rate | branch | clusters | line | noise |
|---|---|---|---|---|
| **h1 cycle** | **0.233** | 0.000 | 0.000 | 0.033 |
| t-SNE angle | 0.450 | 0.000 | 0.267 | 0.067 |
| Isomap angle | 0.583 | 0.000 | 0.533 | 0.050 |
| circle fit angle | 1.000 | 0.000 | 0.600 | 0.000 |
| autoencoder, 2-D | 0.850 | 0.000 | 0.900 | 0.017 |

A monotone ramp is where the reconstruction methods come apart — the
autoencoder invents a ring on 90 % of them and the circle fit on 60 %, which is
what "always returns an order" means when the cloud has a line in it. Separated
cell groups are trivial for everyone. The branching trajectory is the one family
that costs the topological method anything, and it costs the others twice as
much.

And the mechanism, now measured at scale instead of asserted: the H1 cycle
**refuses to answer** on 65 of the 240 non-cyclic clouds — 27.1 % [21.9, 33.0] —
while t-SNE refuses on **0 of 240** [0, 1.6]. Every other method's silence is a
score that happened to miss a null; the topological method's silence is the
absence of a dominant bar, which is a property of the cloud rather than of the
draw.


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
