# The protocol

`omictopo.protocol` is the part of this package that is meant to be reused by
someone who does not care about persistent homology. It takes a cloud of samples
and returns a **decision with an epistemic status**, not a bare number.

The reason to lead with a protocol rather than with a statistic is measured. On
280 clouds against 24 baselines the topological count is *not* the most sensitive
detector of a cycle. What it does that none of the baselines do is decline to
answer when the geometry does not support an answer — and its refusals are tied
to the shape of the cloud, not to a threshold on a score. A count that can be
silently wrong is worth less than a count that says when it is unusable.

## The five steps

1. **Declare the design first.** Fix the cyclic parameter(s) and the sampling
   layout before looking at the data. Nothing downstream is chosen after seeing
   the answer.
2. **Count.** `count_cycles` counts the H1 bars of the Rips filtration that
   outrun a *covariance-matched* null — Gaussian clouds carrying the data's own
   covariance, pushed through the same distance and the same H1 summary. This
   step is label-free.
3. **Diagnose with two orthogonal quantities.** `gap` (longest bar over second
   longest) says whether a loop *dominates* the spectrum. A circular Mantel `r`
   against the design's own axis says whether that loop is the one the design
   sampled. They disagree in both directions, and both disagreements matter.
4. **Classify.** Map the two diagnostics onto one of four statuses, and when the
   status is not `counted`, return the refusal instead of the integer.
5. **Declare the envelope.** State the conditions under which the count is
   trustworthy, measured from the real designs rather than assumed.

## The four statuses

| status | meaning | what to report |
|---|---|---|
| `counted` | a dominant loop exists **and** it follows the design axis | the integer, with the gap beside it |
| `underpowered` | a cycle is present (the axis explains the distances) but H1 cannot resolve it in this gene set | the failure, not a count |
| `inconsistent` | a dominant loop exists but it does **not** follow the design axis | check the design before reporting |
| `no_cycle` | no cycle in the axis and none in the spectrum | the cloud is not a test of the count |

Two rules are deliberately conservative:

* the dominant-bar threshold is `gap >= 2`, which sits between the smallest gap
  ever measured on a design that was counted right (**3.56**) and the largest on
  one that was not (**1.56** — the mouse-liver oestrous × circadian grid);
* `r` is an **effect-size** cut, not a significance level. The yeast cell-cycle
  design has `r = +0.16` at `p = 0.0005` on 60 samples *and* a dominant bar; the
  rule returns `inconsistent`, which is the right *instruction* (go and check the
  design) even though the count itself is sound. Read the status as advice to
  look, not as a ruling on the integer.

## The envelope

`envelope()` returns the declared boundary. Every number in it comes from a
committed benchmark record, and `reproduce/verify.py` re-checks those numbers, so
the boundary cannot quietly go stale.

**The count is trustworthy when**

* the selected gene set is *dominated* by the cyclic signal, not merely contains
  it — the count succeeds only when essentially all of the high-variance genes
  carry the loop (`examples/cycle_counting_power.py`);
* the design samples the whole period — a sub-period window is an open arc and
  cannot close a loop (`examples/cycle_counting_real.py`, the yearlong window
  sweep);
* for two independent parameters, there are two *independently sampled* axes — a
  **grid**, not a longer time course (`examples/two_cycle_search.py`). On the one
  measured design that satisfies the shape (PRJNA1348461, 12 clock times × 4
  oestrous phases, every cell occupied) the second axis turns out to be a
  *coherent phase offset of the first axis's loop* rather than an independent
  coordinate, so the count is 1 and not 2 — `examples/estrous_grid_cycles.py`
  and `examples/axis_separability.py`.

A third design closes the last loophole. **GSE220120** (Zhu et al.) is human
blood carrying *two genuine oscillators* in the same tissue — a published ~24 h
circadian program and a ~12 h ultradian one — with three subjects × two days ×
twelve two-hourly draws. If the only obstacle were "two rhythms riding one axis",
this is the design that would break it. It does not, and the measurement
(`examples/ultradian_two_oscillator.py`) says why in a way that is more useful
than another zero: the ~12 h program is **absent from the high-variance genes
the selection keeps**. Of the 200 strongest 12 h genes, **0** survive the
high-variance selection, against **53** of 200 for the 24 h program. The second
cycle is not topologically hidden here; it is filtered out before the topology
ever sees it — which is precisely the first envelope condition above, now
confirmed on a third tissue and a second organism.

**The protocol cannot**

* recover a loop whose signal sits in a minority of the selected genes;
* resolve two cycles from a single sampling axis, however many samples that axis
  carries — the image of a one-parameter family is a closed geodesic on the
  torus, topologically a circle, so `beta_1 <= 1`;
* quote an integer when the H1 spectrum has no dominant bar.

**Not label-free.** The Mantel `r` and its p-value use the design axis. They are
reported to keep the label-free count honest, never as part of the label-free
claim.

## Axis geometry

`omictopo.axes` answers the question the protocol cannot: when a design records
*two* axes, is the second one a coordinate of its own? This is a joint-geometry
question, so per-gene harmonic fits cannot pose it — a COSINOR-style analysis
produces a rhythm call and an amplitude per gene and has no representation of the
joint coordinate at all.

`axis_geometry(X, axis1, axis2)` returns four blocks — which axis dominates the
geometric Mantel correlation, whether the two axes explain separate slabs of the
variance, whether the second axis acts as a common phase shift of the first or a
change of shape, and the model-free matched distance at a fixed value of axis 1 —
then classifies the pair into one of four pre-registered verdicts:

| verdict | meaning |
|---|---|
| `independent_axes` | axis 2 carries a Mantel effect of its own (`r >= 0.5`) |
| `modulation` | axis 2 has no Mantel effect but shifts every gene's axis-1 phase coherently by less than half a quarter-period |
| `independent_axes_inert` | axis 2 neither produces an effect nor moves points at a fixed axis-1 value |
| `undetermined` | the design does not separate the possibilities — refuse |

The criteria are fixed in `omictopo.axes.CRITERIA` before any dataset is scored,
and `reproduce/verify.py` re-checks the numbers they are evaluated on.
`examples/axis_geometry_multi.py` scores three synthetic calibrations where the
ground truth is known (independent / modulated / inert) plus every real design
that has two axes, and gets **three different verdicts for the three
calibrations** and agreement with the published reading on all real designs.

Two negative results are reported rather than hidden, because they constrain how
the verdict may be read:

* the **separability block cannot carry the decision** — an *inert* axis is
  trivially orthogonal to axis 1 (measured additive share 1.00, cosine 0.00), so
  `share`/`cosine` rule out collinearity but cannot tell "independent" from
  "does nothing";
* the **matched-distance ratio is not monotone** in the size of axis 2 — a
  genuinely independent second axis measures ~0.57 while the mouse-liver
  modulation measures ~0.88, because the denominator pools diagonal offsets.

What does discriminate, on both the calibrations and the real designs, is the
Mantel `r` of axis 2 (+0.70 for two independent axes against −0.04 for a
modulated or inert one), which is why the criteria let `r` carry the decision and
use the geometric quantities to describe the *shape* of the effect.

### The corpus-scale reading

`examples/axis_regularity.py` (→ `benchmark/axis_regularity.json`) asks the
cross-dataset question on **7 real axis pairs from 6 independent published
experiments and 5 organisms**, on one code path, with the hypothesis and the
criteria written into the script docstring *before* any dataset is scored. Its
pre-registered hypothesis — "a condition-shaped second axis never carries an
independent Mantel effect" — is **falsified**: a 1 h heat treatment reaches
`r2 = +0.982` in both RNA fractions of the same experiment. What survives is an
*ordering* rather than a partition: `r2` is set by displacement
(`matched.separation_ratio`), and every design whose second axis moves points by
more than 1.2 steps of axis 1 carries the largest effects. So `r2` answers *"does
axis 2 displace the cloud?"*, which is a **different question** from *"is axis 2 a
second cycle?"* — the third negative result published for this module. The one
independently sampled second cycle in the corpus is also the one design where axis
2 is confounded with the experiment, so there is still no clean positive case on
real data. See [Evidence §7b](results.md#7b-across-the-corpus-a-hypothesis-dies-an-ordering-survives).

## Usage

```python
import numpy as np
from omictopo.protocol import analyse, summarise, envelope

X, phase = ...                      # samples x features, and the design axis

rec = analyse(X, axis=phase, period=2.0 * np.pi)
print(rec["status"])                # 'counted' | 'underpowered' | ...
print(summarise(rec))               # one line for a results table

if rec["status"] == "counted":
    print(rec["n_cycles"], "cycle(s), gap", rec["gap"])
else:
    print(rec["advice"])            # why the protocol is refusing

print(envelope()["requires"])       # the conditions it is claiming under
```

Label-free use (no design axis) is allowed and still distinguishes `counted`
from `underpowered`, but it can never return `inconsistent` — with no axis there
is nothing for a loop to be inconsistent *with*.

## Implementation notes

* `analyse` requires the caller to have removed replicate structure first (see
  `centre_by_group` in `examples/cycle_counting_real.py`). Every well-separated
  group contributes its own loop, so an uncentred cloud counts groups.
* A perfectly clean loop produces a **single** H1 bar, with no second bar to
  divide by. That is the most dominant spectrum there is, so the internal gap is
  infinite and the reported `gap` is `None` rather than `0`.
* `omictopo.cycles.count_cycles` and `omictopo.stats.mantel` are bit-identical to
  the versions the published benchmark numbers came from, and
  `tests/test_protocol.py` pins that equivalence — the protocol can never drift
  away from the results it summarises.
