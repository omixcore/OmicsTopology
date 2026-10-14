# OmicsTopology

Topological data analysis of omics state spaces.

The usual question asked of transcriptomic time courses is *which genes differ*.
This package asks a different one: **what shape does the state space have, and
does that shape change?** A circadian time course should trace a closed loop in
expression space; a developmental series should trace an open trajectory. The
tools here measure that distinction directly, with persistent homology, and
say how far each result can be pushed before it stops being evidence.

## What is here

* **Persistent homology** with two backends — a dependency-free built-in
  implementation and an optional `ripser` path — for H0 and H1, plus bottleneck
  and Wasserstein distances between diagrams.
* **Permutation tests** that respect the sampling unit, with Benjamini-Hochberg
  and Bonferroni corrections, Wilson intervals and exact binomial tests, because
  every headline number here is a proportion over a handful of clouds.
* **Representative cycles**: the reduction that produces an H1 bar also produces
  the *cycle* — a closed walk over sample indices. That walk orders the samples
  without using any timestamp, which makes it a prediction that can be scored
  against a clock held out of the computation.
* **A cycle count**: `beta_1` read as a model-selection answer, "how many
  independent cyclic parameters does this state space have?", together with the
  decision rule that says when the count is worth reporting at all.
* **A calibrated protocol** (`omictopo.protocol`): the decision rule wrapped as a
  reusable procedure that returns a *status* — `counted`, `underpowered`,
  `inconsistent`, `no_cycle` — and refuses to quote an integer the spectrum does
  not support. See [The protocol](protocol.md).
* **An axis-geometry diagnostic** (`omictopo.axes`): given a design with two
  recorded axes, decides whether the second is a coordinate of its own, a
  coherent modulation of the first, or inert — on criteria fixed before the data
  are scored, and refusing when the design does not separate the cases. It is
  the joint-coordinate question that per-gene harmonic fits cannot pose. See
  [Axis geometry](protocol.md#axis-geometry).

## Quick start

```bash
pip install -e ".[dev]"          # numpy, scipy, pytest, matplotlib
python -m pytest                 # 239 tests, no network needed
python tools/gen_api_docs.py     # regenerate the API reference
sh tools/ci.sh                   # the same checks the hosted CI runs
```

The analysis scripts need data that is downloaded, not committed:

```bash
python examples/prepare_datasets.py --outdir data/datasets
python examples/multi_dataset_benchmark.py --data-root data/datasets

python examples/prepare_cycle_designs.py --outdir data/cycle_designs
python examples/cycle_counting_real.py --root data/cycle_designs
python examples/cycle_counting_power.py
python examples/incremental_value.py --data-root data/datasets
python examples/cycle_identifiability.py --root data/cycle_designs
python examples/two_cycle_search.py --cache data/geo_cache
python examples/estrous_grid_cycles.py --root data/cycle_designs
python examples/axis_separability.py --root data/cycle_designs
python examples/axis_geometry_multi.py --root data/cycle_designs
python examples/axis_regularity.py --designs data/cycle_designs --plant data
python examples/ultradian_two_oscillator.py --cache data/geo_cache
```

## Read this before quoting a number

The project publishes its negative results alongside its positive ones, because
they constrain what the method may be used for:

* On ten benchmark clouds the topological verdict and a plain circular Mantel
  test score the same for **detection** (8/10). The audit is in
  `benchmark/statistical_audit.json` and it is why the claims here are about
  ordering rather than about detection.
* The cycle count works on **two of six** real designs. The other four fail
  for a measured reason, not an unknown one, and the decision rule that
  separates them is part of the package.
* The two-parameter claim — "the count returns 2" — holds on a simulated grid
  and on **no measured design**: a single sampling axis with two frequencies is
  a closed geodesic, not a torus. The one measured design that records two
  *cyclic* axes (PRJNA1348461: 12 clock times × 4 oestrous phases, 48 females,
  every cell occupied) is now measured and counts **1**: the second axis is a
  coherent phase offset of the clock's own loop (coherence 0.92), not an
  independent coordinate. `examples/two_cycle_search.py` states the grid that
  *would* count 2 as a number (>= 64 samples, at full signal).
* The axis-geometry diagnostic does **not** partition axes by kind. Across 7 real
  axis pairs from 6 independent experiments and 5 organisms, the pre-registered
  guess — "a condition-shaped second axis never carries an independent Mantel
  effect" — is falsified by a 1 h heat treatment at `r2 = +0.98`. What the corpus
  supports is an *ordering by displacement*, not a rule about factor kind: `r2`
  measures whether axis 2 moves the cloud, which is a different question from
  whether it is a second cycle. The diagnostic publishes this as its third
  negative result (`benchmark/axis_regularity.json`).
* A topology-guided gene-selection procedure was implemented, tested on four
  real designs and **rejected**: it improved nothing and its calibration fired
  on a cloud with no cycle in it. It remains in the package with that warning
  attached, rather than quietly working around.

[Evidence](results.md) collects the numbers behind each of those statements.
