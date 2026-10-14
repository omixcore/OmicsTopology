# OmicsTopology

Python tools for persistent homology and design-associated geometry in omics
state spaces. The current protocol detects **H1 features under a specified
reference model**; it does not count biological oscillators or establish
independent latent coordinates.

## Install

```bash
pip install 'omictopo[fast] @ git+https://github.com/omixcore/OmicsTopology.git@fix/validated-topology-protocol'
```

Python ≥3.9, NumPy and SciPy are required. The v2 counting protocol pins ripser
and fails explicitly if it is unavailable. Install `[dev]` for the complete
test and reproduction environment, including optional baseline dependencies.

## Detect H1 features and report uncertainty

```python
from omictopo import analyse, summarise

# X: samples × features. Units may identify repeated subjects or trees.
result = analyse(X, axis=phase, period=24, backend='ripser', seed=0,
                 units=subject_ids)
print(summarise(result))
result['candidate_count']       # bars passing the maximum-surrogate test
result['n_cycles']              # stable detected count, or None on abstention
result['stability']             # counts and agreement across reduced clouds
result['design_association']    # separate Mantel effect; p omitted by default
```

Every observed bar is compared with a Gaussian surrogate **maximum** H1
statistic. Surrogate draws are reconstructed in feature space and pass through
the same standardisation and distance calculation as the observations. Each
subsample refits its own surrogate distribution. A two-loop result is not
rejected just because its two bars have similar lengths.

A stable count of zero means **no H1 features detected under this analysis**;
it is not proof that a biological rhythm is absent. Resampling agreement is
robustness, not posterior count confidence or nominal Betti-number coverage.
`units` retains complete independent units during topology subsampling. Without
units, row subsampling is a geometric sensitivity check. The design Mantel
statistic never decides how many features are reported.

## Describe a second axis

```python
from omictopo.axes import axis_geometry

record = axis_geometry(X, clock_hours, treatment,
                       period1=24, axis2_kind='categorical', units=subject_ids)
record['axis_effect']
record['phase_or_shape']        # circular offsets, reliable fits and residuals
record['matched']               # separate intervals for ratio and difference
record['cyclic_evidence']       # second cycle is not established by Mantel r
```

Categories use equal distances between different labels, rather than an
arbitrary ordering on a circle. Harmonic fits include an intercept. Phase
residuals wrap on the circle, and gene coherence measures agreement across
genes. Mean block offsets alone are not evidence for a common shift. Optional
phase uncertainty resamples whole supplied units or rows within fit blocks.
Raw-row intervals are descriptive when observations are temporally dependent.

Mantel permutations now exchange sample labels by simultaneously permuting
matrix rows and columns. Supply `permutation='samples'` only when justified;
`permutation_strata` restricts exchanges within a declared block. A degenerate
scheme returns no p-value. Never treat a significant axis effect as evidence
for another independent loop.

## Evidence and limitations

Read [docs/results.md](docs/results.md) for the frozen v2 validation and fresh
public-source analyses. Validation seeds are separate from development seeds;
the configuration and algorithm hash are fixed before scoring. This is an
internal frozen plan, not an externally registered or independently conducted
study. GSE3431 is a new source study held out from parameter fitting; it remains
one retrospective continuous-culture experiment.

The tests include equal-strength and unequal-strength double loops, a sparse
three-loop manifold, Gaussian noise, an open arc, drift, clusters and a branch.
Undercounts and unresolved real cases are reported alongside successes.

Version-1 JSON records remain unchanged in `benchmark/`. Their old protocol
and phase classifier are available explicitly as `omictopo.legacy_protocol`
and `omictopo.legacy_axes` to reproduce historical records. They are not v2
validation. See [docs/migration_v2.md](docs/migration_v2.md).

## Reproduce

```bash
pip install -e '.[dev]'
python reproduce/reproduce_v2.py              # tests, integrity, v2 smoke check
python reproduce/reproduce_v2.py --synthetic  # rerun the frozen 60-cloud study
python reproduce/reproduce_v2.py --real       # download, hash-check, reanalyse sources
```

The real route downloads processed expression matrices for GSE3431 and
GSE158444; raw files are not redistributed. Source and derived-record hashes,
all settings, sample identifiers and environment metadata accompany results.
The real two-fraction comparison is one study, not two independent replications.
GitHub CI executes both historical record verification and a v2 smoke check.
A [container recipe](reproduce/Dockerfile) and a reference environment are
provided. A fresh real-data run is different from checking committed summaries.

## General two-condition workflow

The existing `run_states` and `omictopo run` workflow generates diagrams,
barcodes, Betti curves and permutation comparisons between groups. It uses a
shared PCA embedding, unlike the full-feature distance geometry of v2 counting.
These outputs must not be conflated. Full H2 persistence requires ripser; the
built-in implementation does not construct tetrahedra and can truncate at its
simplex budget.

[API](docs/api.md) · [Protocol](docs/protocol.md) ·
[Validation](docs/results.md) · [MIT license](LICENSE)
