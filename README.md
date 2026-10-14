# OmicsTopology

Topological data analysis (TDA) for omics state spaces: does the **shape** of
the data change between two biological conditions?

Mean-difference methods ask "is gene X higher in condition B?".  They cannot
ask the question this package asks:

> did the overall state space change its topology — did a new basin appear,
> did the samples start moving around a cycle?

That is the question behind stress responses, developmental transitions and
cell differentiation, where a population can move through a *sequence* of
states without any single gene showing a large average shift.

![State-space topology compared](docs/topology_comparison.png)

In the figure above, the stress condition (orange) holds a stable loop
(`beta_1 = 1`) across a wide range of scales while the control (blue) has none.
No gene-level test would surface that; it is a property of the whole space.

**Read [docs/results.md](docs/results.md) before quoting anything here.**  The
project publishes its negative results next to its positive ones, and they
constrain how far each claim reaches: the topological verdict ties a plain
circular Mantel test at *detection*; the cycle *count* works on one of four
real designs, for a reason that is measured rather than guessed; and a
topology-guided gene-selection fix was implemented, tested against a
pre-registered negative control and **rejected**.  What does hold up is measured
too — against eight cheaper label-free baselines the topological cycle reaches
the same sensitivity with **1 false positive in 23** instead of 9 in 23.

---

## What it computes

| Dimension | Meaning in an omics state space |
|---|---|
| **H0** | Connected components — how many separate basins the samples occupy |
| **H1** | Loops — the population moves *around* a cycle rather than from A to B |
| **H2** | Cavities — a shell of states surrounding a hole (needs a dense sample) |

For each condition you get a **persistence diagram**, a **barcode**, **Betti
curves** and **persistence landscapes**; between two conditions you get exact
**bottleneck** and **Wasserstein** distances plus a **permutation test**.

## Install

```bash
pip install git+https://gitee.com/omixcore/OmicsTopology.git

# with plotting
pip install "OmicsTopology[plot] @ git+https://gitee.com/omixcore/OmicsTopology.git"
```

Only `numpy` and `scipy` are required. `matplotlib` is optional and used only
for figures. There is no compiler, no Java, no Bioconductor.

## Quickstart

### Python

```python
from omictopo import run_states

# matrix: rows = samples, columns = genes;  groups: condition label per row
report = run_states(matrix, groups, labels=["control", "salt_stress"],
                    n_components=3, max_points=40, n_permutations=200)

print(report.summary_text())
report.test["p_value"]          # is the shape change significant?
report.states["salt_stress"]    # that condition's persistence diagram
report.write("results/")        # TSV tables
```

### Command line

```bash
omictopo run --matrix expression.tsv --groups groups.tsv \
             --labels control,salt_stress --outdir results --plots
```

### Try the built-in examples

```bash
python examples/demo_known_shapes.py      # validate against known topology
python examples/demo_stress_response.py   # the salt-stress case above
python examples/demo_stress_response.py --control-shape shift   # negative control
```

## Reading the numbers

* **`h1_max_persistence`** — the longest surviving loop. A ring that is real
  gives one bar far longer than the rest.
* **`h1_total_persistence`** — total amount of loop structure, including
  noise-level loops. Sensitive but *not* specific — do not use it as a test
  statistic; see the power table below.
* **`h1_entropy`** — one dominant loop gives low entropy; many equal small
  loops give high entropy. Use it to tell signal from a noise floor.
* **`bottleneck`** — the single worst-matched feature. Strict: fires only if
  *some* feature genuinely moved.
* **`wasserstein`** — the overall matching cost. Sensitive to many small
  changes.

Persistence values are reported **relative to the shared Rips threshold**, so
states with different embedding scales are comparable.

## A negative control is built in

```python
from omictopo.datasets import two_state_omics

X, groups, labels = two_state_omics(state_b="cycle")   # loop  -> test fires
X, groups, labels = two_state_omics(state_b="shift")   # shift -> test does not fire
```

Both conditions are generated through the same gene loadings, so the marginal
gene distributions are comparable and only the *geometry* differs.

### A single p-value is not evidence

With 100 permutations, `p = 0.030` is only the 4th smallest value attainable,
and it can happen by luck. What matters is how the test behaves across
*independent* datasets. `examples/power_analysis.py` measures exactly that —
10 datasets per case, 100 permutations each:

| statistic | detection (a real ring) | false positive (mean shift) |
|---|---|---|
| `h1_total` | 2/10 = 20% | 1/10 = 10% |
| **`h1_max`** ← default | **10/10 = 100%** | **0/10 = 0%** |
| `bottleneck` | 10/10 = 100% | 0/10 = 0% |
| `wasserstein` | 7/10 = 70% | 1/10 = 10% |
| `h1_count` | 4/10 = 40% | 0/10 = 0% |

The first row is the lesson. `h1_total` sums *every* H1 bar, so a crowd of
short noise loops dilutes one genuine ring until it is indistinguishable from
a plain mean shift — 20% detection against 10% false positives is barely
better than guessing. `run_states` defaults to `h1_max` (the longest surviving
loop) and exposes `statistic=` so you can pick another.

```python
report = run_states(matrix, groups, labels=["control", "salt_stress"],
                    statistic="h1_max",       # default; "bottleneck" also works well
                    n_permutations=200)
```

## Correctness

`examples/demo_known_shapes.py` checks the implementation against shapes whose
topology is known before running anything:

| point cloud | expected | measured |
|---|---|---|
| `circle(24)` | exactly one loop | `beta_1 = 1`, loop persistence 1.47 |
| `two_clusters()` | two basins | `beta_0 = 2` |
| `blob(30)` | no persistent loop | H1 max < 0.1 |
| `torus()` | loops in two directions | ≥ 2 H1 bars > 0.5 |

H0 is computed twice by independent routes — minimum spanning tree, and the
full GF(2) boundary-matrix reduction — and the two agree to machine precision
on every test cloud. The full pytest suite is in `tests/`.

## Real data: the circadian clock is a loop

Synthetic checks can only go so far, so the package ships a case study on a
public plant dataset: **GSE158444** (Bonnot & Nagel 2021,
[PMID 33871647](https://pubmed.ncbi.nlm.nih.gov/33871647/)) — *Arabidopsis
thaliana* seedlings sampled at eight times of day across one circadian
period, kept at 22 °C or given a 1 h 37 °C heat shock. The processed count
table is small, so the whole thing runs in about a minute.

```bash
python examples/prepare_plant_dataset.py --outdir data/arabidopsis_heat
python examples/arabidopsis_heat_stress.py --data-dir data/arabidopsis_heat
```

A working circadian clock should make the samples trace a **closed loop** in
state space; a purely linear response would trace a line. That is a topology
stated in advance, then tested — not one discovered and rationalised
afterwards.

![circadian loop](docs/fig_circadian_loop.png)

Both conditions recover the loop: a single dominant H1 class
(persistence 9.7 control, 8.2 heat; β₁ = 1 at its midpoint). The evidence does
not depend on the embedding — a Mantel test finds sample-to-sample expression
distance is better explained by the distance *around* the clock than by the
distance *along* it (r = 0.74 vs 0.58 control, 0.69 vs 0.60 heat, both
p = 0.0002):

![circularity and replicates](docs/fig_circularity_replicates.png)

Biological replicates also behave (same-hour samples are ~0.25× as far apart
as different-hour ones), so the loop is not an artefact of noisy replicates.

The permutation test finds **no topological difference** between control and
heat (p = 0.96). That negative result is informative rather than merely
inconclusive: at the same sample size (n = 23/24) the test recovers a true
loop difference in 11/12 synthetic datasets with 0/12 false positives, so it
had power here and found nothing. A 1 h heat shock does not rewrite the
topology of the diurnal program.

Two caveats the case study measures rather than hides:

* **Shared-embedding compression.** `run_states` embeds all samples together;
  here PC1 is 85 % of the variance and is the treatment axis, so the
  circadian structure survives only in PC2/PC3. A shared threshold calibrated
  on that spread makes the H1 numbers look small.
* **p-values move with preprocessing.** Sweeping the number of retained
  features (500 → all 21 015) keeps the loop in every setting but swings the
  control-vs-heat p-value between 0.07 and 0.96. When two conditions share a
  dominant structure, `h1_max` has wide permutation nulls. Report the sweep,
  not a single number.


## A systematic benchmark: does the loop survive across species and designs?

One dataset is an anecdote. `examples/multi_dataset_benchmark.py` asks the
same pre-registered question of every dataset in
`examples/dataset_registry.py` — with the answer declared **before** the data
are looked at:

* **loop** — a circadian time course; a running clock should close the loop;
* **line** — a monotonically advancing process (fruit development, stress
  recovery); there is no reason for the transcriptome to return to start.

Seven public datasets, four species, ten dataset × condition clouds:

| key | accession | organism | design | expected |
|---|---|---|---|---|
| `ath_diurnal` / `ath_diurnal_heat` | GSE158444 | *A. thaliana* | 8 times of day / 24 h, control & heat arm | loop |
| `ath_recovery` | GSE158444 | *A. thaliana* | heat-shock recovery, 4 times | line |
| `rice_diurnal` | GSE138193 | rice | 17 times / 48 h, three genotypes | loop |
| `ath_diurnal_ll` / `ath_diurnal_ld` | GSE137732 | *A. thaliana* | 16 times / 45 h, constant light & long day | loop |
| `tomato_fruit_dev` | GSE128739 | tomato | 9 stages of fruit development | line |

Two independent, pre-registered lines of evidence per cloud:

1. **H1 vs its own null.** The longest H1 bar (normalised by the RMS radius
   implied by the distance matrix) is compared with clouds drawn from a
   *covariance-matched Gaussian* — same n, same covariance, no structure —
   pushed through the identical pipeline.
2. **Circular vs linear Mantel.** In full expression space (no PCA), is the
   expression distance between two samples better explained by the distance
   *around* the clock than the distance *along* it?  The period is scanned
   over 16–32 h so a free-running clock is not penalised for not being
   exactly 24 h.

```bash
python examples/prepare_datasets.py --outdir data/datasets
python examples/multi_dataset_benchmark.py --data-root data/datasets
```

**8 of 10 clouds match the design's expectation.**  The three rice genotypes
are called loop-like (H1 p = 0.02–0.03, circular Mantel r ≈ 0.92 vs linear
r ≈ 0.05), both GSE158444 arms are loop-like, and all three aperiodic clouds
(tomato development, stress recovery × 2) are line-like.  Verdicts survive a
feature-count sweep (500 → all genes): rice 12/12, aperiodic clouds 12/12.

![benchmark trajectories](docs/fig_benchmark_trajectories.png)
![H1 vs null](docs/fig_benchmark_h1_vs_null.png)
![circular vs linear Mantel](docs/fig_benchmark_mantel.png)

Three things the benchmark measures rather than hides:

* **The two GSE137732 failures are the method behaving correctly.**  Both
  conditions were sampled once per time point over 45 h — nearly two cycles.
  Split the record into single-cycle windows
  (`examples/diagnose_period_drift.py`) and the loop reappears in every
  window (circular advantage +0.46 / +0.41 for constant light, +0.04 / +0.17
  for long day): within a cycle the clock closes the loop, but the second
  cycle does **not** close onto the first.  A 45 h cloud under drift is a
  helix, not a loop, and the circular test refuses a model that does not
  hold.
* **Distance metric matters more than the embedding.**  With Euclidean
  distances no cloud clears its null; with correlation distances (comparing
  expression *patterns* rather than absolute levels) the rice genotypes
  clear theirs.  `--metric` makes the comparison easy to rerun.
* **Biological replicates are not what carries the loop.**  Five independent
  checks (`examples/diagnose_replicate_contribution.py`): (i) a variance
  decomposition puts only 28–37 % of each gene's variation inside a time
  point (median ICC 0.72 / 0.63) — the share replication could ever average
  away; (ii) replicate-averaged centroids (zero within-time noise) detect the
  loop *better* than the full cloud (H1 max 0.73 vs 0.61); (iii) sampling one
  library per time point (60 draws) leaves H1 max at 0.61 with the circular
  advantage positive in 100 % of draws; (iv) **manufacturing** three
  pseudo-replicates per time point for GSE137732 — which has none — does not
  flip its whole-record verdict (LL: −0.10 → +0.03/+0.04, LD: −0.69 →
  +0.01/−0.37, H1 p = 1.0 in every case); (v) GSE137732's loop would only
  break at ~3–4× the within-time scatter GSE158444 actually has, yet its
  split-half gene reliability (0.985 for long day) is nearly identical to
  GSE158444's (0.997).  Replicates help a little (the circular advantage
  rises by ~0.13), but far too little to matter for the verdict — which is
  geometry (drift), not statistics.

Full result tables and diagnosis JSON live in [`benchmark/`](benchmark/).

**What this benchmark does *not* yet show.**  `examples/benchmark_audit.py`
(→ `benchmark/statistical_audit.json`) audits the headline number, and the
audit is deliberately unflattering:

* **The topological statistic has no demonstrated edge over a classical
  test.**  Dropping H1 entirely and keeping only the circular-vs-linear
  Mantel test still scores **8/10**; H1 alone scores 7/10 and demanding both
  scores 7/10.  On these ten clouds the Mantel line carries the accuracy, so
  "TDA was needed here" is not yet supported.
* **Nothing survives multiplicity correction on the H1 line.**  With ten
  clouds the smallest BH threshold is 0.005 while the smallest H1 p-value is
  0.0199 — **0/10** survive BH or Bonferroni.  The Mantel p-values (0.0002)
  do survive (6/7).
* **8/10 is not distinguishable from chance at this sample size.**  Exact
  two-sided binomial against a 50 % baseline gives p = 0.109 and a Wilson
  95 % CI of [0.49, 0.94]; about 20 units would be needed for 80 % power.

The per-unit table in the audit JSON is the honest version of the summary.


## What TDA does that a phase-based test cannot

The audit says the topological statistic has no demonstrated edge at
*detection*, and that stands.  `examples/topology_without_labels.py`
(→ `benchmark/topology_without_labels.json`) asks a different question: what
happens when the sampling times are **held out** instead of fitted?

**A. The labels are the Mantel test's input, not the topology's.**  Permuting
the time labels drops the circular Mantel correlation from a mean of **+0.65**
to **−0.03** across the ten clouds, while the H1 verdict does not move at all
(largest change 0.0).  A phase-based test is a statement about the data
*given* the labels; the cycle below is a statement about the data.

**B. The H1 cycle predicts the clock.**  The reduction that produces the
dominant H1 bar also produces the loop itself, as a closed walk over sample
indices (`omictopo.cycles`).  Scored by how many of its steps follow the clock
(`circular_succession_score`), that walk is **exact on all six real circadian
clouds** (control, heat, constant light, three rice genotypes) and **6/7
testable clouds survive Benjamini-Hochberg at α = 0.05** (p = 5·10⁻⁵ to
1.8·10⁻³).  Label-free baselines fall short: the first principal coordinate
scores 0.19-0.65, and the PC1-PC2 angle — the classical way to read a phase
off an ordination — ties the cycle on five clouds, collapses to 0.38 on the
constant-light cloud, and returns a *perfect* 1.00 on two clouds that contain
no loop at all.  Resampling genes leaves the score at 1.00 in 40/40 draws on
four of the six clouds.  The single failure is the long-day cloud whose 45 h
record drifts — the failure the benchmark already documents, not a new one.

**C. β₁ counts independent cyclic parameters, not oscillators.**  Two
oscillators riding a single sampling axis still give one loop, because
`t ↦ (t/P₁, t/P₂)` is a closed geodesic *on* the torus: the state space is a
circle whatever the number of oscillators.  Give the design a second
independent parameter (a 13 × 9 grid, data simulated to the real clouds'
shape) and two H1 bars clear the covariance-matched null; one parameter gives
one bar and none gives none.  **6/6 designs are read correctly.**

**C, on measured data.**  A claim built and tested by the same person is not
worth much, so `examples/prepare_cycle_designs.py` +
`examples/cycle_counting_real.py` (→ `benchmark/cycle_counting_real.json`)
run the identical pipeline on five public designs whose sample sheets were
fixed years ago: 5 Douglas-fir trees × 16 dates across a full year (GSE93067),
6 trees × 12 time points over 48 h (GSE93066), 22 people × 7 circadian phases
× 2 sleep phases (GSE48113), the two Douglas-fir series pooled into an
annual × daily design, and — the one design whose cycle is *not* a clock —
budding yeast released from an elutriation, wild type vs a `clb1-6` cyclin
mutant (GSE8799).

Exactly **two of five** designs are counted correctly, and both are the ones
whose cycle dominates their transcriptome.  The yearlong series reaches a
dominant-bar gap of 4.9 and is counted as 1, and sweeping the window shows the
same thing from the other side — any sub-year window is an open arc and counts
0.  The yeast cell cycle reaches a gap of 3.6 and is counted as 1 while its
sample sheet records **three** axes (genotype, replicate, cell-cycle time) of
which only one is cyclic — the other two carry no cycle at all: genotype is a
strong two-valued shift (Mantel r = +0.46, p ≤ 0.0005), the elutriation an
offset, and removing their means is what exposes the single cycle, whereas the
same cloud without centring returns 0.  Its phase alignment is significant but
modest (circular Mantel r = +0.16, p ≤ 0.0005), which is why the headline count
is quoted on the pooled 60-sample cloud rather than on either genotype alone.
The other three return 0, and two diagnostics say why: their H1 spectra decay
smoothly (gap 1.06-1.36, i.e. *no dominant bar to count*) while the cycles are
nevertheless present — the circular Mantel correlation against the design's own
axis is +0.70 for the diurnal cloud, the same value the yearlong series has.
`examples/cycle_counting_power.py` (→ `benchmark/cycle_counting_power.json`)
then measures the variable that separates the time-axis working cases from the
failing ones: how many of the 3000 selected genes actually carry the ring.  The
count succeeds only when essentially all of them do.  That is the honest envelope
of the claim, and it is measured rather than asserted — and it is an envelope
about *time-axis* designs, because the yeast cloud shows a dominant bar at
`r = +0.16`, so a dominant bar and a strong phase alignment are two different
kinds of evidence.

**D. Incremental value, measured against twenty-one baselines — eight fixed transforms and thirteen learned or trained representations.**
`examples/incremental_value.py` (→ `benchmark/incremental_value.json`) puts the
topological cycle on a level field with twenty-one cheaper, label-free ways to read
a cyclic order off an expression cloud, over **12 positive clouds** (7 real
circadian designs plus 5 synthetic rings) and **23 negative clouds** (3 real
"line" designs plus 5 independent draws each of a ramp, pure noise, three
clusters and a branching trajectory).  Every method must beat the 95th
percentile of its own phase-permutation null to count as *declaring* a cycle.

The eight fixed transforms are the classical answers — PC1 order, the PC1-PC2
angle, a circle fitted to PC1-PC2, classical MDS, average-linkage seriation, a
greedy tour and a 2-opt tour (diffusion maps sit with the learned spectral rows).
The thirteen learned ones
(``examples/learned_baselines.py``) exist because a fixed transform invites an
obvious reply: *your method wins because the others are weak, not because it is
wise — a representation with enough capacity to learn the manifold would find
the clock too, and would not invent one.*  That reply is now taken at face
value twice.  The field includes a shallow two-layer autoencoder (1-D and 2-D
bottleneck, 857 trainable parameters on a 24-sample cloud) **and a deep
64-32-16 autoencoder, a denoising autoencoder and a variational autoencoder on
the same stack (8 377-8 411 parameters, a 349:1 ratio)**, plus UMAP, Isomap,
locally linear embedding, RBF kernel PCA, Laplacian eigenmaps, t-SNE, a Kohonen
self-organising map, a Hastie principal curve and diffusion maps.  Every deep
hyperparameter is the winner of a grid run on the positive controls alone
(`examples/tune_deep_baselines.py` → `benchmark/deep_tuning.json`, the whole
grid committed) and frozen before any negative cloud was scored, so every
specificity number the deep nets earn is out-of-sample.  With the nonlinearity
switched off the autoencoders reproduce PCA's order exactly, in both bottleneck
widths, so they are not losing here because they are broken.

| method | family | sensitivity | specificity | balanced accuracy |
|---|---|---|---|---|
| **H1 cycle** | topological | **0.917** | **0.957** | **0.937** |
| t-SNE angle | learned, neighbour embedding | 0.917 | 0.913 | 0.915 |
| Isomap angle | learned, neighbour embedding | 0.833 | 0.870 | 0.851 |
| UMAP angle | learned, neighbour embedding | 0.917 | 0.696 | 0.806 |
| denoising autoencoder | learned, reconstruction | 0.833 | 0.696 | 0.764 |
| circle-fit angle | fixed transform | 0.917 | 0.609 | 0.763 |
| average-linkage seriation | fixed transform | 0.917 | 0.609 | 0.763 |
| **autoencoder, deep 64-32-16** | learned, reconstruction | 0.917 | 0.609 | 0.763 |
| **autoencoder, variational** | learned, reconstruction | 0.917 | 0.609 | 0.763 |
| PC1-PC2 angle | fixed transform | 0.917 | 0.565 | 0.741 |
| self-organising map | learned, topology preserving | 0.833 | 0.609 | 0.721 |
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

Four conclusions, none of them flattering to the methods being proposed as
alternatives:

1. **Capacity was never what was missing, and that has now been tested twice.**
   The deep 64-32-16 autoencoder — 8 377 parameters against 24 samples, tuned
   on the positive controls alone — lands on a specificity of 0.609,
   *bit-for-bit the specificity of a circle fitted to two principal axes*, the
   same fixed transform from 1971.  The shallow autoencoder lands on 0.478,
   bit-for-bit classical MDS.  The tuning grid itself says why: on the four
   positive controls the deep nets top out at mean scores of 0.84-0.88 where
   the shallow network reads 1.000 — more depth made the one thing they had to
   do *harder* at this sample size.

2. **What separates the learned methods is whether they fit the cloud or only
   its neighbourhoods.**  Every reconstruction method — autoencoders at every
   depth, kernel PCA, Laplacian eigenmaps, LLE, principal curve, and the fixed
   classical MDS — invents a ring on 11-13 of the 23 non-cyclic clouds.  The
   methods that only preserve local neighbourhoods — t-SNE, Isomap and UMAP —
   invent one on 2, 3 and 7.  The distinction is mechanistic, not a tuning
   accident: a reconstruction objective is satisfied by any smooth 2-D
   arrangement, including a circular one, and a non-cyclic cloud has a
   circular arrangement available.

3. **Refusal without calibration buys nothing.**  The deep and the variational
   autoencoders also refuse to answer on 5 of the 23 negative clouds — the
   bottleneck collapses, outwardly the same behaviour as the topological
   method's — but on the 18 clouds they *do* answer they still invent 9 rings.
   The topological method refuses on 5 and errs on 1.  Silence is only worth
   what the answers are worth.

4. **The topological cycle is still first, and the mechanism is measured, not
   asserted.**  Every non-topological method produces an order on every
   negative cloud it does not collapse on, and stays quiet only because its
   score happened to miss its own null — t-SNE by a margin of **+0.042** on
   its worst cloud.

`examples/increment_power.py` sizes the comparison that the 23-cloud benchmark
cannot resolve: 240 further non-cyclic clouds drawn from the same four declared
families (60 each, on a seed schedule sharing nothing with the benchmark's) and
40 further rings, with the nine methods that mattered rerun on all of them.
The 1-against-2 was an artefact of sample size.  At 240 clouds the H1 cycle's
false-positive rate is **0.067** [0.04, 0.11]; the best learned methods are a
plateau of neighbour-preserving embeddings — t-SNE **0.196** [0.15, 0.25] and
UMAP **0.204** [0.16, 0.26] — with the reconstruction methods far behind (deep
autoencoder 0.421, shallow 0.442).  Paired on the same clouds, the H1 cycle
declares on fewer than every rival and each pair is significant on its own:
10 / 41 against t-SNE, p = 1.5 × 10⁻⁵; 9 / 42 against UMAP, p = 3.4 × 10⁻⁶;
7 / 61 against Isomap, p = 7.4 × 10⁻¹²; 2 / 82 against the circle fit,
p = 3.7 × 10⁻²²; 4 / 94 against the shallow autoencoder, p = 2.4 × 10⁻²³.
Confirming a gap of this size unpaired would take 107 clouds per method at 80 %
power; this used 240.

And the mechanism, now measured at scale: the H1 cycle **produces no order at
all** on 65 of the 240 non-cyclic clouds — 27.1 % — and errs on one cloud it
answers, while the deep autoencoders produce no order on 60 (their bottleneck
collapses) *and still* get 101 wrong, and also miss 3 of the 40 rings.  Refusal
tied to the cloud's geometry is an advantage; refusal that fires on rings too
is just failure by another name.

The blemishes are in the same table, not hidden: the topological cycle's false
positives concentrate entirely in the branching family (0.233 there, 0.000 on
ramps, 0.000 on clusters, 0.033 on noise), and its one miss is the long-day
drift case already documented above.

**E. When a count is worth reporting at all.**
`examples/cycle_identifiability.py` (→
`benchmark/cycle_identifiability.json`) turns the two diagnostics into a
decision rule, `omictopo.cycles.cycle_identifiability(gap, mantel_r)`, and it
sorts the four measured designs correctly: yearlong → *identifiable*; diurnal
and pooled → *underpowered* (the ring is real, H1 cannot resolve it in that gene
set); forced desynchrony → *inconsistent* (a loop exists that does not follow
the design axis, `r = +0.10`).  `gap` says the spectrum is flat; `mantel_r` says
**why**, and the two flat cases call for opposite responses — back to the
experiment, or back to the feature selection.  The rule also refuses to
over-claim: a large `gap` means one bar dominates, not that the integer is
right, and the pooled design showed `gap = 9.3` while returning 1 against a
design that contains 2.

**A fix that was tried and rejected.**  `omictopo.cycles.topology_guided_cycle`
alternates between finding the dominant cycle and keeping the genes whose
expression sweeps around it, using no clock at all.  Tested on the same four
designs with the forced-desynchrony cloud as a pre-registered negative control,
it failed twice.  It moved the recovered-order score by only −0.06 to +0.08 and
turned no non-detection into a detection.  And decisively, its calibration — the
guided statistic against a covariance-matched Gaussian null **with the whole
procedure re-run inside the null**, so the selection's optimism is present in
both arms — fires on the control cloud, where there is no cycle to find.  The
function ships with that warning in its docstring instead of being quietly
deleted, because a documented failure is more useful than a missing one.

Limits, stated plainly: the *scoring* needs the clock even though the
*computation* does not; the recovered cycle visits only 44-76 % of a cloud's
samples; the *two-parameter* result (β₁ = 2) is demonstrated on simulated data
only, because even the pooled annual × daily design has no dominant bar once
the two series' signals dilute each other — what measured data now do show is
the *axis-count* trap: the yeast cell-cycle design records three axes and one
cycle and is counted as one; the counting criterion needs a transcriptome
dominated by the cycle; and the identifiability rule is a two-threshold
heuristic calibrated on five designs, not a test with a p-value.  The design
that would settle the two-parameter case — time against an independently
cycling factor, sampled densely in both — remains the cheapest experiment this
analysis points at.

**F. Third-party reproduction.**  `reproduce/` makes the check executable for
someone who does not trust us: `sh reproduce/run_all.sh` from a fresh clone
installs the package, runs the test suite, runs the known-topology demo, and
recomputes every published headline number from the committed raw benchmark
records with code that re-implements the Wilson interval, the exact binomial
test (including every McNemar p-value) and Benjamini-Hochberg from their
textbook definitions rather than importing them — so a bug in the package's own
statistics module cannot certify itself.  `reproduce/expected/manifest.json`
pins the sha256 of every benchmark file a published number depends on, and
`tools/determinism_audit.py` demonstrates the runs are deterministic by
executing the same seeded pipeline twice in separate interpreters and comparing
hashes.  A container definition is included.

![Clock order recovered with the labels held out](docs/fig_labels_held_out.png)

![The envelope of the cycle count](docs/fig_cycle_counting_power.png)

![The topological cycle against eight fixed transforms and thirteen learned baselines](docs/fig_incremental_value.png)

![The same comparison, at a sample size that separates all nine](docs/fig_increment_power.png)


## API tour

```python
from omictopo import (
    standardize, pca_embed, pairwise_distances,   # geometry
    persistent_homology,                          # H0/H1/H2
    betti_curve, persistence_landscape,           # summaries
    bottleneck_distance, wasserstein_distance,    # diagram distances
    permutation_test_pointclouds,                 # significance
    run_states,                                   # the whole workflow
)
from omictopo import datasets                   # shapes with known topology
```

Lower-level pieces (`build_rips`, `reduce_boundary`, `choose_threshold`) are
importable too if you want the filtration itself.

## Method notes and honest limitations

* **Embedding first.** The filtration is built on a PCA embedding (default 3
  components), not on the raw feature space — a Rips complex in 20 000
  dimensions is meaningless. Change `n_components` if you have a reason to.
* **Sample size, and what the threshold really costs.** Rips complexes grow as
  O(n³), so clouds are subsampled to `max_points` per state and the complex is
  held inside `max_simplices` (default 80 000) by *lowering the distance
  threshold*. That keeps the runtime flat — but it is not free. Measured on
  random clouds at the default budget, the threshold the backend ends up using
  is:

  | points | threshold as % of the cloud's full diameter |
  |---|---|
  | 40 | 100% |
  | 200 | 23% |
  | 800 | 10% |
  | 3000 | **4%** |

  Past a few hundred points the built-in backend is looking at a small-scale
  neighbourhood only, and any large-scale structure — including the global
  loop that H1 exists to find — falls outside the filtration. For clouds that
  large use `backend="ripser"`, which never truncates.
* **Truncation.** Bars dying beyond the threshold are reported as essential
  (`death = inf`) and, in the summaries, counted as dying *at* the threshold.
  That keeps a loop which outlives the filtration visible instead of silently
  dropping it.
* **Backend choice is not cosmetic.** `backend="builtin"` (the default)
  honours `threshold` and `max_simplices`, and gives the same answer on every
  machine. `"ripser"` and `"auto"` ignore the threshold, so
  `diagram.threshold` becomes `inf` and more bars survive; they are also
  roughly 100× faster. On the examples here both backends return identical
  p-values, but pin the backend explicitly when reproducibility matters.
* **H2 is indicative only.** To resolve cavities the complex would need
  tetrahedra, which are deliberately not built. For full H2 persistence use a
  dedicated library; `persistent_homology(..., backend="ripser")` will use
  [ripser](https://github.com/scikit-tda/ripser.py) when it is installed.
* **A torus needs samples.** `datasets.torus()` documents this honestly: the
  major loop resolves cleanly, the minor one needs a denser sample. Topology
  detection is bounded by sampling, not by the algorithm.
* **The point-cloud permutation test assumes exchangeability** between
  conditions. With biological replicates, prefer
  `permutation_test_replicates`, which treats each replicate as the sampling
  unit.

## Repository layout

```
src/omictopo/
  geometry.py      distances, standardisation, PCA embedding
  filtration.py    Vietoris-Rips filtration with automatic size control
  homology.py      H0 via MST, H1/H2 via GF(2) boundary reduction
  diagram.py       persistence diagrams and their statistics
  summary.py       Betti curves, persistence landscapes
  compare.py       exact bottleneck and Wasserstein distances
  stats.py         permutation tests, FDR correction, Wilson intervals, exact binomial
  cycles.py        representative cycles, cyclic order, identifiability rule
  plots.py         publication-oriented figures
  pipeline.py      run_states(): matrix -> verdict
  cli.py           omictopo command
  datasets.py      synthetic data with known topology
examples/          runnable demos; see the files
  demo_known_shapes.py        correctness against shapes with known topology
  demo_stress_response.py     positive / negative two-condition controls
  power_analysis.py           detection and false-positive rates across datasets
  prepare_plant_dataset.py    download + normalise the GSE158444 Arabidopsis data
  arabidopsis_heat_stress.py  real-data case study: the circadian clock is a loop
  dataset_registry.py         the benchmark's public datasets + their loaders
  prepare_datasets.py         prepare every benchmark dataset in one command
  multi_dataset_benchmark.py  pre-registered multi-dataset topology benchmark
  diagnose_period_drift.py    why a 2-cycle record under drift is a helix, not a loop
  diagnose_replicates.py      does dropping replicates hurt detection? (no)
  diagnose_replicate_contribution.py
                              how much do replicates contribute? (measured 5 ways)
  benchmark_audit.py          ablation, multiplicity and resolution of the headline
  topology_without_labels.py  what a phase-based test cannot be asked (labels held out)
  prepare_cycle_designs.py    download + normalise five real cyclic designs (GSE93067/93066/48113/8799)
  cycle_counting_real.py      block C on measured data: 2 of 5 designs are counted correctly
  cycle_counting_power.py     when does the count work? (measured, not assumed)
  incremental_value.py        the H1 cycle against 21 baselines, fixed and learned
  learned_baselines.py        the thirteen learned representations and why they are fair
  tune_deep_baselines.py      the positive-control grid behind the frozen deep settings
  increment_power.py          is the gap to the best learned baseline real? (280 clouds)
  cycle_identifiability.py    when is a count reportable, and a fix that was rejected
tools/
  ci.sh                       the checks the hosted CI runs, runnable locally
  gen_api_docs.py             regenerate docs/api.md; --check fails on drift
  determinism_audit.py        the same seeded pipeline twice, hashes compared
reproduce/
  README.md                   the third-party reproduction protocol
  run_all.sh                  fresh clone -> install -> tests -> verify, one command
  verify.py                   recomputes every published number from the raw records
  make_manifest.py            regenerate expected/manifest.json (sha256 pins)
  Dockerfile                  the same checks in a container
docs/
  index.md results.md api.md  guide, evidence table, generated API reference
  fig_*.png                   every figure the scripts produce
benchmark/         committed result tables of every experiment
tests/             194 pytest cases, no network needed
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest                 # 194 tests, no network and no benchmark data required
sh tools/ci.sh         # install + tests + generated-docs check, as CI runs them
sh reproduce/run_all.sh   # the third-party reproduction, from this tree or a fresh clone
```

Continuous integration is configured for both hosts and does the same work:

* `.workflow/ci.yml` — Gitee Go (must be switched on under 服务 → Gitee Go)
* `.github/workflows/ci.yml` — GitHub Actions, across three Python versions
  and two operating systems

## Documentation

* [docs/index.md](docs/index.md) — what the package is for and what to read first
* [docs/results.md](docs/results.md) — every headline number, with the script
  that produced it
* [docs/api.md](docs/api.md) — generated reference; CI fails if it drifts from
  the source (`python tools/gen_api_docs.py --check`)

Build the docs site with `pip install -e ".[docs]"` then `mkdocs serve`.

## License

MIT — see [LICENSE](LICENSE).
