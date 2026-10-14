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
  one that was not (**1.36**);
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
  **grid**, not a longer time course (`examples/two_cycle_search.py`).

**The protocol cannot**

* recover a loop whose signal sits in a minority of the selected genes;
* resolve two cycles from a single sampling axis, however many samples that axis
  carries — the image of a one-parameter family is a closed geodesic on the
  torus, topologically a circle, so `beta_1 <= 1`;
* quote an integer when the H1 spectrum has no dominant bar.

**Not label-free.** The Mantel `r` and its p-value use the design axis. They are
reported to keep the label-free count honest, never as part of the label-free
claim.

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
