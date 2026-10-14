# Version 2 evidence

These are current records from the frozen v2 protocol, not a reinterpretation of v1 summaries.

## Frozen simulated validation

| Generator | Reference H1 rank | Stable detected counts | Answers |
|---|---:|---|---:|
| gaussian | 0 | [0, 0, 0, 0, 0, 0] | 6/6 |
| ramp | 0 | [0, 0, 0, 0, 0, 0] | 6/6 |
| clusters | 0 | [0, 0, 0, 0, 0, 0] | 6/6 |
| branch | 0 | [0, 0, 0, 0, 0, 0] | 6/6 |
| arc | 0 | [0, 0, 0, 0, 0, 0] | 6/6 |
| drift | 0 | [0, 0, 0, 0, 0, 0] | 6/6 |
| one_cycle | 1 | [1, 1, 1, 1, 1, 1] | 6/6 |
| two_cycles | 2 | [2, 2, 2, 2, 2, 2] | 6/6 |
| unequal_two | 2 | [1, 1, 1, 1, 1, 1] | 6/6 |
| three_cycles | 3 | [0, 0, None, 0, 0, 0] | 5/6 |

Non-cyclic clouds: 0/36 positive detections; Wilson 95% interval [0.0, 0.09641862996887933]. This small benchmark does not establish a zero population error rate.

Single loops and equal-strength double loops recover their reference counts. Weak double loops are undercounted; sparse triple-cycle manifolds are not resolved. Stability does not guarantee completeness of the count.

The 30 axis simulations separate common phase shifts, gene-specific changes and small effects. Both independently varying phases and discrete shifts can receive strong_axis_effect; this is deliberately not an independence label.

## Fresh public-source analyses

GSE3431: 36 samples and 1000 selected features. Candidate count 1; resampling agreement 0.500; final status unresolved; circular Mantel r=0.811. No permutation p is assigned to the dependent time series.

The 500-feature analysis is also unresolved; the all-feature analysis detects no H1 features. This is one retrospective held-out source study, not an independent experimental replication or a confirmed topological positive.

GSE158444 TOT: control no_cycle, heat no_cycle; heat distance r=0.986; common-shift share=0.024.
GSE158444 TR: control unresolved, heat no_cycle; heat distance r=0.985; common-shift share=0.009.

Both fractions are one source study. The treatment association and gene-specific phase changes are descriptive reanalyses. No stable detected loop comparison supports a claim that heat destroys a cycle, and no novel mechanism is asserted.

## Reproduce and inspect

- `benchmark/v2/validation_plan.json`: frozen settings, algorithm digest, disjoint seed schedule.
- `benchmark/v2/validation.json`: per-cloud spectra, surrogate maxima, resampling counts and environment.
- `benchmark/v2/real_validation.json`: sample identities, feature IDs, source hashes and fresh analyses.
- `reproduce/reproduce_v2.py`: verification, optional simulation rerun, optional download-to-results rerun.
- `docs/results_v1.md`: historical records and claims only; use legacy APIs for their reproduction.
