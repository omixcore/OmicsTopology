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
* **Dropping biological replicates does not destroy the loop.**  Sampling
  one library per time point from GSE158444 (60 independent draws) leaves
  H1 max at 0.61 — the same as the full data — with the circular advantage
  positive in 100 % of draws (`examples/diagnose_replicates.py`).

Full result tables and diagnosis JSON live in [`benchmark/`](benchmark/).


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
  stats.py         permutation tests, FDR correction
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
  diagnose_replicates.py      how much biological replicates contribute to detection
benchmark/         committed result tables of the multi-dataset benchmark
tests/             68 pytest cases
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
