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

## Quick start

```bash
pip install -e ".[dev]"          # numpy, scipy, pytest, matplotlib
python -m pytest                 # 151 tests, no network needed
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
```

## Read this before quoting a number

The project publishes its negative results alongside its positive ones, because
they constrain what the method may be used for:

* On ten benchmark clouds the topological verdict and a plain circular Mantel
  test score the same for **detection** (8/10). The audit is in
  `benchmark/statistical_audit.json` and it is why the claims here are about
  ordering rather than about detection.
* The cycle count works on **one of four** real designs. The other three fail
  for a measured reason, not an unknown one, and the decision rule that
  separates them is part of the package.
* A topology-guided gene-selection procedure was implemented, tested on four
  real designs and **rejected**: it improved nothing and its calibration fired
  on a cloud with no cycle in it. It remains in the package with that warning
  attached, rather than quietly working around.

[Evidence](results.md) collects the numbers behind each of those statements.
