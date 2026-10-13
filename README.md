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
  noise-level loops. Sensitive but less specific.
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
gene distributions are comparable and only the *geometry* differs. Measured on
the default setting (600 genes, 40 samples per condition, 100 permutations):

| case | H1 max (stress / control) | p-value |
|---|---|---|
| `cycle` (a real loop) | 0.361 / 0.062 (≈ 6×) | **0.030** |
| `shift` (mean shift only) | 0.095 / 0.100 (≈ 1×) | 0.693 |

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
* **Sample size.** Rips complexes grow as O(n³). Clouds are subsampled to
  `max_points` (default 120) per state, and a distance threshold is chosen
  automatically to keep the complex inside `max_simplices`.
* **Truncation.** Bars dying beyond the threshold are reported as essential
  (`death = inf`) and, in the summaries, counted as dying *at* the threshold.
  That keeps a loop which outlives the filtration visible instead of silently
  dropping it.
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
tests/             63 pytest cases
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
