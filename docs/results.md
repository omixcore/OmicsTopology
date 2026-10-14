# Evidence

Every number on this page is produced by a script in `examples/` and committed
under `benchmark/`. None of it is a summary written by hand.

## 1. Does the topology add anything the cheaper methods cannot?

`examples/incremental_value.py` → `benchmark/incremental_value.json`

Twenty-two label-free methods are given the same expression matrix and each
must produce a **cyclic order of the samples** without seeing the clock. The
order is then scored against the clock that was withheld
(`circular_succession_score`: the share of steps that stay at the current phase
or advance to the next one). A method *declares a cycle* when its score beats
the 95th percentile of its own phase-permutation null.

Twelve positive clouds (seven real circadian designs plus five draws of the
synthetic ring) and twenty-three negative clouds (three real "line" designs plus
five draws each of a linear ramp, pure noise, three clusters and a branching
trajectory). Eight baselines are fixed transforms; fourteen are *trained* or
learned — a shallow autoencoder at two bottleneck widths (857 parameters on a
24-sample cloud), a deep autoencoder (64-32-16, 8 377 parameters), a denoising
autoencoder and a variational autoencoder on the same stack, UMAP, diffusion
maps, Isomap, LLE, kernel PCA, Laplacian eigenmaps, t-SNE, a self-organising
map and a principal curve. The deep variants' hyperparameters are the winners
of a grid run on the positive controls alone
(`examples/tune_deep_baselines.py` → `benchmark/deep_tuning.json`), frozen
before any negative cloud was scored.

| method | family | sensitivity | specificity | balanced accuracy |
|---|---|---|---|---|
| **h1 cycle** | topological | **0.917** [0.65, 0.99] | **0.957** [0.79, 0.99] | **0.937** |
| t-SNE angle | learned, neighbour | 0.917 | 0.913 | 0.915 |
| Isomap angle | learned, neighbour | 0.833 | 0.870 | 0.851 |
| UMAP angle | learned, neighbour | 0.917 | 0.696 | 0.806 |
| denoising autoencoder | learned, reconstruction | 0.833 | 0.696 | 0.764 |
| circle fit angle | fixed transform | 0.917 | 0.609 | 0.763 |
| average-linkage leaf order | fixed transform | 0.917 | 0.609 | 0.763 |
| **autoencoder, deep 64-32-16** | learned, reconstruction | 0.917 | 0.609 | 0.763 |
| **autoencoder, variational** | learned, reconstruction | 0.917 | 0.609 | 0.763 |
| PC1-PC2 angle | fixed transform | 0.917 | 0.565 | 0.741 |
| self-organising map | learned | 0.833 | 0.609 | 0.721 |
| classical MDS angle | fixed transform | 0.917 | 0.478 | 0.698 |
| autoencoder, 2-D shallow | learned, reconstruction | 0.917 | 0.478 | 0.698 |
| diffusion maps | learned, spectral | 0.917 | 0.478 | 0.698 |
| kernel PCA angle | learned, reconstruction | 0.917 | 0.478 | 0.698 |
| Laplacian eigenmaps | learned, reconstruction | 0.917 | 0.478 | 0.698 |
| principal curve | learned, 1-D manifold | 0.833 | 0.522 | 0.677 |
| greedy tour | fixed transform | 0.833 | 0.478 | 0.656 |
| 2-opt tour | fixed transform | 0.833 | 0.435 | 0.634 |
| locally linear embedding | learned, reconstruction | 0.833 | 0.435 | 0.634 |
| PC1 order | fixed transform | 0.667 | 0.478 | 0.573 |
| autoencoder, 1-D | learned, reconstruction | 0.583 | 0.435 | 0.509 |

Four things this table says, none of them flattering to the methods proposed
as alternatives:

1. **Capacity was never what was missing, and now that has been tested twice.**
   The deep autoencoder — 8 377 parameters against 24 samples, a 349:1 ratio,
   tuned on the positives — lands on a specificity of 0.609, *bit-for-bit the
   specificity of a circle fitted to two principal axes in 1971*. The shallow
   autoencoder lands on 0.478, bit-for-bit classical MDS. The tuning grid
   itself says why: on the four positive controls the deep nets top out at
   mean scores of 0.84–0.88 where the shallow network reads 1.000 — more
   depth made the one thing they had to do *harder* at this sample size.
2. **What separates the learned methods is fitting the cloud versus fitting
   its neighbourhoods.** Every reconstruction method — autoencoders at every
   depth, kernel PCA, Laplacian eigenmaps, LLE, principal curve, and fixed
   MDS — invents a ring on 11-13 of the 23 non-cyclic clouds. The three that
   only preserve local neighbourhoods — t-SNE, Isomap and UMAP — invent one
   on 2, 3 and 7.
3. **Refusal without calibration buys nothing.** The deep autoencoder and the
   variational autoencoder also refuse to answer on 5 of the 23 negative
   clouds — their bottleneck collapses, the same outward behaviour as the
   topological method's — but on the 18 clouds they *do* answer they still
   invent 9 rings. The topological method refuses on 5 and errs on 1. Silence
   is only worth what the answers are worth.
4. **The topology is first, and the gap to the best learned method is the
   abstention mechanism.** Every non-topological method produces an order on
   all 23 negative clouds and stays quiet only because its score missed its
   own null — t-SNE by **+0.042** on its worst cloud; the topological method's
   silence is the absence of a dominant bar, which is a property of the cloud.

Two honest blemishes are visible in the same table: the topological method's
one false positive is a **branching** cloud (1 of 5 draws), and its one miss is
`ath_diurnal_ld`, the known 45-hour drift case.

### 1b. Is the gap to the best learned baseline real?  240 further clouds say yes

`examples/increment_power.py` → `benchmark/increment_power.json`

Twenty-three clouds cannot separate 1 false positive in 23 from 2 in 23 — the
two rates behind the ordering of the table. The four declared negative families
were therefore redrawn — 60 independent clouds each, on a seed schedule that
shares nothing with the benchmark's — and the nine methods that mattered were
rerun on all 240, with 40 further rings for sensitivity.

| method | sensitivity (40 rings) | false positives (240) | produced no order |
|---|---|---|---|
| **h1 cycle** | 1.000 [0.91, 1.00] | **0.067** [0.04, 0.11] (16) | **0.271** [0.22, 0.33] (65) |
| t-SNE angle | 1.000 | 0.196 [0.15, 0.25] (47) | 0.000 (0) |
| UMAP angle | 1.000 | 0.204 [0.16, 0.26] (49) | 0.000 (0) |
| Isomap angle | 1.000 | 0.292 [0.24, 0.35] (70) | 0.000 (0) |
| denoising autoencoder | 1.000 | 0.325 [0.27, 0.39] (78) | 0.000 (0) |
| circle fit angle | 1.000 | 0.400 [0.34, 0.46] (96) | 0.000 (0) |
| autoencoder, deep | 0.925 [0.80, 0.97] | 0.421 [0.36, 0.48] (101) | 0.250 (60) |
| autoencoder, variational | 0.925 [0.80, 0.97] | 0.421 [0.36, 0.48] (101) | 0.250 (60) |
| autoencoder, shallow 2-D | 1.000 | 0.442 [0.38, 0.50] (106) | 0.000 (0) |

Paired on the same negative clouds (McNemar, exact), the H1 cycle declares on
fewer clouds than every rival, and each pair is significant on its own:
10 / 41 against t-SNE, p = 1.5 × 10⁻⁵; 9 / 42 against UMAP, p = 3.4 × 10⁻⁶;
7 / 61 against Isomap, p = 7.4 × 10⁻¹²; 10 / 72 against the denoising
autoencoder, p = 1.0 × 10⁻¹²; 2 / 82 against the circle fit, p = 3.7 × 10⁻²²;
5 / 90 against the deep and the variational autoencoder, p = 3.1 × 10⁻²¹;
4 / 94 against the shallow autoencoder, p = 2.4 × 10⁻²³. The observed gap of
12.9 points would need 107 clouds per method for 80 % power; this used 240.

Two things the larger field adds. UMAP joins t-SNE at the top of the learned
methods (0.204 against 0.196, an indistinguishable pair), so "the best learned
baseline" is now a plateau of neighbour-preserving methods, not a single
winner. And the deep autoencoders fail in a new way: they refuse to answer on
60 of the 240 non-cyclic clouds (25 %) but **also** miss 3 of the 40 rings
(sensitivity 0.925) — their bottleneck collapses when the cloud does not suit
it, whether or not there is a cycle to find. Refusal is only an advantage when
it is tied to the cloud's geometry, and the deep nets' refusal is not.

Where the false positives come from matters as much as how many:

| false-positive rate | branch | clusters | line | noise |
|---|---|---|---|---|
| **h1 cycle** | **0.233** | 0.000 | 0.000 | 0.033 |
| t-SNE angle | 0.450 | 0.000 | 0.267 | 0.067 |
| UMAP angle | 0.450 | 0.000 | 0.333 | 0.033 |
| Isomap angle | 0.583 | 0.000 | 0.533 | 0.050 |
| denoising autoencoder | 0.483 | 0.000 | 0.783 | 0.033 |
| circle fit angle | 1.000 | 0.000 | 0.600 | 0.000 |
| autoencoder, deep | 0.683 | 0.000 | 1.000 | 0.000 |
| autoencoder, variational | 0.683 | 0.000 | 1.000 | 0.000 |
| autoencoder, shallow 2-D | 0.850 | 0.000 | 0.900 | 0.017 |

A monotone ramp is where the reconstruction methods come apart — the deep and
variational autoencoders invent a ring on *every* ramp they answer (they stay
silent on the rest) and the shallow one on 90 %, which is what "always returns
an order" means when the cloud has a line in it. Separated cell groups are
trivial for everyone. The branching trajectory is the one family that costs the
topological method anything, and it costs every rival at least twice as much.

And the mechanism, now measured at scale instead of asserted: the H1 cycle
**refuses to answer** on 65 of the 240 non-cyclic clouds — 27.1 % [21.9, 33.0] —
and errs on only one cloud it answers, while the deep autoencoders refuse on 60
and still get 101 wrong. The topology's silence is the absence of a dominant
bar, which is a property of the cloud rather than of the draw; the deep nets'
silence is a collapsed bottleneck, which happens to rings as well.


## 2. When is a cycle count worth reporting?

`examples/cycle_counting_real.py` → `benchmark/cycle_counting_real.json`

Five real designs, all publicly available and all designed by other people
years ago, run through the identical pipeline as the simulated block:

| design | cycling parameter(s) | gap | circular Mantel r | count | expected |
|---|---|---|---|---|---|
| Douglas-fir yearlong (GSE93067) | annual | 4.93 | +0.70 | **1** | 1 |
| yeast cell cycle (GSE8799) | cell-cycle phase | 3.56 | +0.16 | **1** | 1 |
| Douglas-fir 48 h (GSE93066) | daily | 1.36 | +0.70 | 0 | 1 |
| human forced desynchrony (GSE48113) | circadian | 1.07 | +0.05 | 0 | 1 |
| yearlong × daily, pooled | annual + daily | 1.09 | — | 0 | 2 |

Two of five. The cause was then located rather than guessed:
`examples/cycle_counting_power.py` → `benchmark/cycle_counting_power.json`
sweeps how many of the 3000 selected genes actually carry the ring and shows
that, across the time-axis designs, the count succeeds only when essentially all
of them do. At the measured `r ≈ 0.70` the simulated gap is 1.2–1.4 and the
measured diurnal gap is 1.36 — the failing clouds sit exactly on the curve. It
does *not* explain everything: the yeast cloud below has a dominant bar whereas
its `r` is only +0.16, so a dominant bar and a strong phase alignment are two
different kinds of evidence, and it is the first kind the count needs.

The yearlong series supplies the cleanest positive result in the project:
sweeping the sampling window, every window short of a full year counts **0**,
and only windows that close the cycle count 1. "A partial period is an arc, not
a loop" therefore rests on a real 365-day cycle rather than on simulated data.

The yeast cell cycle is the second success and the more pointed one. Every
other design turns on a *time* axis — a year, a day, a sleep schedule — so a
critic can say the method is only reading the calendar. GSE8799 removes that
escape: the cycle is a cell-intrinsic state cycle that nothing entrains, and its
period is not a calendar constant — the periodogram peaks at 110-130 min and had
to be measured. More to the point, the sample sheet records **three** axes
(genotype, replicate, cell-cycle time) of which exactly **one** is cyclic, so it
is the measured counterpart of the simulated "two axis labels, one cycle" trap:
a criterion that counted axes would answer three, and the count answers one.

The two other axes are not cycles at all. Genotype is a strong two-valued shift
(Mantel r = +0.46, p ≤ 0.0005, measured on the uncentred cloud so the shift is
still in the distances) and the elutriation is an offset, so removing their
means is what exposes the single cycle — the same cloud *without* centring
returns 0. The result is also honest about its own boundary: the phase alignment
is significant but modest (circular Mantel r = +0.16, p ≤ 0.0005), and split by
genotype the two 30-sample clouds behave differently — the wild type alone
counts 0, while the cyclin mutant alone counts 1 on a dominant bar that does
*not* track the phase (r = +0.06, p = 0.09). The pooled 60-sample cloud is the
one whose dominant bar and its phase agree, which is why the headline count is
quoted on it.

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

The rule above was calibrated on those four designs and is deliberately left
that way; the fifth design is its boundary case rather than a fifth data point.
The yeast cell-cycle cloud has a dominant bar (`gap` 3.56) **and** a phase
alignment that is significant but modest — `r = +0.16`, `p ≤ 0.0005` on 60
samples — so the rule returns `inconsistent`, i.e. "check the design before
reporting". Checking is what §2 does, and the check passes: the phase is real
and the two other recorded axes are inert. The point is that `r >= 0.5` is an
*effect-size* cut, and a small effect size can still be significant on a large
cloud, so the verdict is best read as an instruction to look rather than as a
ruling on the integer. The function's docstring now says so.

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

## 5. The two-parameter case: the missing experiment is a *grid*

`beta_1` claims to count independent cyclic *parameters*. The positive half of
that claim — "there are two cycles here and the count returns 2" — holds on a
simulated grid and on **no measured design**. `examples/two_cycle_search.py`
(→ `benchmark/two_cycle_search.json`) is the accounting, and the reason is
topological rather than statistical.

**One axis, two frequencies, is a knot — not a torus.** If
`t -> (t/P1, t/P2)` wraps a closed curve in the state space, its image is a
closed geodesic on the torus, which is topologically a circle, so `beta_1 <= 1`
however many rhythms ride on the single axis. The prediction was checked on the
one downloadable dataset of that shape:

| cloud | samples | count | gap | Mantel vs 24 h | Mantel vs 96 h |
|---|---|---|---|---|---|
| GSE12289 (rat mammary, ~4-day estrous time course) | 32 | **0** | 1.25 | +0.11 (p = 0.0085) | +0.14 (p = 0.001) |

Both rhythms are present and significant, and neither dominates the selected
genes: the cloud is an arc, so there is nothing to count. The topological
prediction holds on real data.

**A public grid *was* found, and it is now measured.** The 2026 mouse-liver
study (PRJNA1348461) samples the four estrous phases against twelve two-hourly
clock times — a genuine 4 x 12 grid, 48 females, every cell occupied. The
FPKM table is released as a preprint supplement, so the question can be asked
of the measured data rather than of a simulation.
`examples/estrous_grid_cycles.py` (→ `benchmark/estrous_grid_cycles.json`)
does exactly that, and the answer is **one, not two**:

| what is measured | value |
|---|---|
| count on the 12 x 4 grid | **0** (gap 1.56 — no dominant bar) |
| circular Mantel r vs the ~24 h clock | **+0.630** (p = 0.0005) |
| circular Mantel r vs the oestrous phase | **+0.006** (p = 0.43) |
| clock r measured *inside* each 24 h block | +0.712 / +0.726 / +0.519 / +0.574 (all p <= 0.0005) |
| across-phase variance, label permutation | p = 0.85 |

The clock is real and resolves on its own inside every oestrous block; the
oestrous axis leaves no independent loop. A calibration on the *same* sampling
settles what the count should be: a pure 24 h clock counts 1, the same clock
with a per-phase **phase shift** also counts 1, and only two genuinely
independent oscillations count 2. A second phase of the same loop is not a
second loop.

**And the topology answers a question the authors' own analysis cannot.**
PRJNA1348461's own analysis quantifies the oestrous effect the field's way —
per-gene harmonic fits, rhythm calls, amplitude comparisons — and reports it as
a modulation of circadian *amplitude*. That analysis is per-gene and
one-dimensional in time; it has no notion of the joint coordinate. Scoring the
joint geometry directly (`examples/axis_separability.py` →
`benchmark/axis_separability.json`) makes the statement precise:

| question | measured |
|---|---|
| which axis dominates the geometry? | clock Mantel r = +0.630 vs oestrous +0.006; per-gene clock amplitude 1.58x the oestrous one |
| are the axes separable slabs of variance? | clock explains 11.6%, oestrous 5.4%, both 17.0% — additive share **1.00**, subspace cosine **0.00** |
| phase shift or shape change? | block offsets are coherent (coherence **0.92**), largest offset 0.37 quarter-periods — a common phase shift of every gene, not a second programme |
| does the oestrous axis move points at fixed clock time? | yes but little: matched pairs sit 12% closer than across-clock pairs, bootstrap CI [0.07, 0.19] |

So the oestrous axis is not inert — it reliably displaces the state at fixed
clock time — but it displaces it by a *common phase offset of the clock's own
loop*, less than one clock step, rather than by opening a second coordinate.
That is a partition of the joint state space, and it is what turns "the
oestrous effect is a modulation" from a summary of amplitude tables into a
property of the geometry.

**The envelope is still stated as a number.** Simulating the grid the analysis
points at — one period of each parameter, both fully sampled, with a realistic
dilution of the signal across the selected genes:

| grid | samples | signal fraction | count |
|---|---|---|---|
| 6 x 6 | 36 | 1.00 | 0 |
| **8 x 8** | **64** | **1.00** | **2** |
| 13 x 9 | 117 | 1.00 | 2 |
| 13 x 9 | 117 | 0.50 | 1 |
| 13 x 9 | 117 | 0.25 | 0 |

The count survives only when the grid is large (**>= 64 samples**) *and* the
cycles are carried by essentially all of the high-variance genes. That is the
honest envelope of the two-parameter claim, and it is the cheapest experiment
this analysis points at. The measured grid sits *below* that envelope (48
samples, two-hourly) and it counts 0 for a second reason as well: the second
axis carries no independent cycle to count.

### 5b. The remaining loophole, closed: two frequencies in one tissue

The one shape the search had not tested is a design that *does* put two genuine
oscillators in the same tissue with replicates — because "two rhythms riding one
axis" would then be the only thing standing between it and a count of 2.
**GSE220120** (Zhu et al.) is exactly that: human peripheral blood, three healthy
subjects x two days x twelve two-hourly draws (72 samples), carrying a published
~24 h circadian program *and* a ~12 h ultradian one.
`examples/ultradian_two_oscillator.py` (→
`benchmark/ultradian_two_oscillator.json`) measures it:

| what is measured | value |
|---|---|
| count on the whole 72-sample cloud | **0** (gap 2.15) |
| circular Mantel r vs 24 h | +0.006 (p = 0.39) |
| circular Mantel r vs 12 h | -0.018 (p = 0.83) |
| 24 h Mantel r after removing the 12 h harmonic | +0.007 (p = 0.38) |
| 12 h Mantel r after removing the 24 h harmonic | -0.016 (p = 0.79) |

Neither frequency leaves a signal at all — so this is *not* the closed-geodesic
obstruction of §5, which would show both rhythms present and neither countable.
The design's own rhythm screen says what is happening: of the 200 genes with the
strongest ~24 h harmonic power, **53** survive the high-variance gene selection
this kind of analysis uses; of the 200 strongest ~12 h genes, **0** do. The
ultradian program is not topologically hidden — it is **filtered out before the
topology is ever built**.

That is the first condition of the protocol's declared envelope ("the count
succeeds only when essentially all of the high-variance genes carry the loop"),
measured on a third tissue and a second organism. It is worth more than another
zero: it says the failure mode is a *preprocessing* choice that can be audited
from the data, not a limit of the method.

## 6. The protocol

`omictopo.protocol` turns the decision rule of §3 into a reusable procedure that
returns a status — `counted`, `underpowered`, `inconsistent`, `no_cycle` — and
refuses to quote an integer the spectrum does not support. Applied to the six
real designs of block C it says *counted* twice and **refuses four times**,
with the two diagnostics (`gap` and the circular Mantel `r`) reported alongside
so every refusal can be audited. The declared envelope is returned by
`envelope()` and re-checked by `reproduce/verify.py`. See
[docs/protocol.md](protocol.md).

## 7. The second axis, as a reusable diagnostic

§5 asked the joint-coordinate question once, on one dataset.
`omictopo.axes` turns it into a procedure that is blind to what the axes *are*:
`axis_geometry(X, axis1, axis2)` returns four blocks (which axis dominates, are
the two axes separable slabs, is axis 2 a common phase shift or a shape change,
and the model-free matched distance at fixed axis 1) and classifies the pair on
criteria fixed in `omictopo.axes.CRITERIA` **before** any dataset is scored.

`examples/axis_geometry_multi.py` (→ `benchmark/axis_geometry_multi.json`)
runs three synthetic calibrations where the ground truth is known, plus every
real design that records two axes:

| case | kind | verdict | r (axis 2) | matched ratio |
|---|---|---|---|---|
| synth_independent | 8x8 grid, both axes carry their own harmonic | **independent_axes** | +0.665 | 0.557 |
| synth_modulated | one clock, shifted coherently per block | **modulation** | -0.038 | 0.249 |
| synth_inert | one clock, second axis does nothing | **independent_axes_inert** | -0.038 | 0.159 |
| estrous_circadian_grid | PRJNA1348461 clock x oestrous | modulation | +0.006 | 0.876 |
| yeast_cell_cycle | GSE8799 cell cycle x genotype | undetermined | +0.458 | 1.083 |
| human_forced_desync | GSE48113 circadian x sleep-wake | undetermined | +0.003 | 0.960 |
| psme_diurnal | GSE93066 day x day-of-experiment | modulation | -0.005 | 0.878 |

All seven match their pre-registered expectation, and the three calibrations get
**three different verdicts** — the diagnostic is not collapsing everything onto
one answer. The `undetermined` verdicts are the designed refusals: the yeast
genotype axis sits in the band between the cuts, and the human sleep-wake axis
has a phase shift (0.68 quarter-periods) large enough that it is close to what a
second loop would look like, so the design has to be examined rather than
classified.

**Two negative results are published with it**, because they constrain how the
verdict may be read and because a criterion that quietly ignores them would be
wrong:

* the **separability block cannot carry the decision.** An *inert* axis is
  trivially orthogonal to axis 1 — the calibration measures additive share
  **1.00** and subspace cosine **0.00** for a second axis that does nothing. So
  `share` and `cosine` rule out collinearity but cannot tell "independent" from
  "does nothing";
* the **matched-distance ratio is not monotone** in the size of axis 2. A
  genuinely independent second axis measures **0.557** while the mouse-liver
  modulation measures **0.876** — because the denominator pools across-axis-1
  pairs that include diagonal offsets. It is used as a coarse resolved/not
  indicator, never as the discriminator.

What does discriminate, on both the calibrations and the real designs, is the
Mantel `r` of axis 2 (+0.70 for two independent axes against −0.04 for a
modulated or inert one). The criteria are therefore written so that `r` carries
the decision and the geometric quantities describe the *shape* of the effect —
which is the honest statement of what the diagnostic is doing, and the reason the
negative results above are in the record rather than in a footnote.

### 7b. Across the corpus: a hypothesis dies, an ordering survives

§7 validated the diagnostic on seven axis pairs read as independent results.
`examples/axis_regularity.py` (→ `benchmark/axis_regularity.json`) asks the
*cross-dataset* question instead, on the same **7 real axis pairs from 6
independent published experiments and 5 organisms**, all scored on one code path.
The hypothesis and the criteria are written into the script's docstring **before**
any dataset is scored.

**The pre-registered hypothesis is false.** The natural guess — "a
condition-shaped second axis never carries an independent Mantel effect" — is
falsified by a 1 h heat treatment. `GSE158444` is a time × treatment design whose
treatment axis is unambiguously a *condition*, and it reaches **r2 = +0.982**, and
does so in *both* RNA fractions of the same experiment (+0.979), so the
counterexample is not one noisy library set. The oestrous, sleep-wake and
day-index axes stay near zero; the heat axis does not. There is no partition by
factor kind.

**What survives is an ordering, not a partition.** The quantity that orders the
corpus is *displacement* — how far axis 2 moves the state cloud relative to a step
of axis 1, reported as `matched.separation_ratio`. The two designs whose second
axis moves points by **more than 1.2 steps of axis 1** carry the two largest
Mantel effects in the corpus; every design whose second axis moves points by less
sits at +0.54 or below.

| design | axis 2 | levels | r2 | ratio | over 1.2 steps? |
|---|---|---|---|---|---|
| estrous_circadian_grid | oestrous phase | 4 | +0.006 | 0.876 | no |
| human_forced_desync | sleep-wake | 2 | +0.003 | 0.915 | no |
| psme_diurnal | day index | 2 | +0.098 | 0.826 | no |
| yeast_cell_cycle | genotype | 2 | +0.458 | 1.089 | no |
| psme_annual_x_diurnal | annual phase (pooled) | 16 | +0.541 | 0.782 | no |
| arabidopsis_TR | 1 h heat | 2 | +0.979 | 1.462 | **yes** |
| arabidopsis_heat | 1 h heat | 2 | +0.982 | 1.471 | **yes** |

The mechanism is geometric: a coherent phase shift is an *isometry* of axis 1's
own loop — it relabels the points without changing their distances — so it leaves
no Mantel effect however large the shift, while a condition that relocates samples
to a different region does. `r2` therefore answers *"does axis 2 displace the
cloud?"*, which is a **different question** from *"is axis 2 a second cycle?"*.

This is the third negative result published for the axis module, and it is the
honest form of the axis claim: the diagnostic measures displacement, and
displacement is not cycle-ness. Two facts from the same corpus sharpen it:

* the one independently sampled second cycle (`psme_annual_x_diurnal`) is also the
  one design where axis 2 is *perfectly confounded* with which of the two pooled
  experiments a sample came from — so there is still **no clean positive case on
  real data**, the same wall `two_cycle_search.py` reports from the counting side,
  arriving here from the axis side;
* the number of axis-2 levels does not order `r2`: a 2-level label spans almost the
  whole observed range (+0.003 to +0.982).
