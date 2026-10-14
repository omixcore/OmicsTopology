# Third-party reproduction, from a fresh clone to a yes/no answer

This directory exists so that someone who does not trust us can check us.
Everything a published number depends on is either committed raw data under
`benchmark/` or synthetic data generated with pinned seeds, and the check
below re-derives the numbers from the raw records with code that shares
nothing with the experiment scripts.

## The one-command version

```bash
git clone https://gitee.com/omixcore/OmicsTopology.git
cd OmicsTopology
sh reproduce/run_all.sh            # add BASELINES=1 to include umap-learn
```

It installs the package, runs the full test suite, runs the known-topology
demo (whose expected answers are written down before the code runs), and then
recomputes every headline number.  Exit code 0 means every check passed.

## What "reproduced" means here

`reproduce/verify.py` performs four layers of checking:

1. **Integrity.**  The sha256 of each committed benchmark file must match
   `reproduce/expected/manifest.json`.  This pins the raw data; the manifest
   names the commit it was made at.  Regenerating the manifest is part of
   changing the paper, not part of running code.
2. **Internal consistency.**  Every method summary inside
   `incremental_value.json` and `increment_power.json` (sensitivity,
   specificity, balanced accuracy, false-positive counts, abstention counts,
   Wilson intervals) is recomputed from that file's own per-cloud records.
3. **Published numbers.**  The values quoted in `README.md` and
   `docs/results.md` are embedded in `verify.py` as constants
   (`PUBLISHED`) and compared against the recomputation.  A transcription
   error between a benchmark file and the paper text fails here.
4. **Independent statistics.**  The Wilson interval, the exact two-sided
   binomial test (including every McNemar p-value) and the
   Benjamini-Hochberg rule are re-implemented in `verify.py` from their
   textbook definitions rather than imported, so a bug in the package's own
   statistics module cannot certify itself.

With `--full`, the script additionally runs the pytest suite and the
known-topology demo as the executable specification.

```bash
python reproduce/verify.py --full
```

## What a third party can and cannot reproduce without downloads

Re-runnable offline, in full, from code and pinned seeds alone: the 280-cloud
power study (`examples/increment_power.py`), the deep-baseline tuning grid
(`examples/tune_deep_baselines.py`), the known-topology demo and the entire
test suite.

Recomputable offline from committed raw records: every number in the
17-method incremental-value benchmark, the detection-axis audit, the
label-free phase-recovery results, the cycle-count results and the
two-parameter search.  The per-cloud records behind them are committed under
`benchmark/`, and `verify.py` re-derives all of them without the network.  The
two-parameter section additionally checks the *protocol's own declared
envelope* (`omictopo.protocol.envelope()`) against those records, so the
boundary the package publishes about itself cannot go stale.

Not re-runnable offline: re-running the incremental-value benchmark end to
end, which needs the ten public GEO series.  The `data/` caches are
deliberately not committed (they are large and the providers ask for
bulk-download courtesy); `examples/prepare_*.py` refetches them.  The chain
from raw record to printed number stays closed either way.

## Determinism

* Every benchmark script pins its seeds; the same commands give byte-identical
  JSON on re-runs.  `tools/determinism_audit.py` demonstrates this by running
  the core pipeline twice in separate processes and comparing hashes.
* The neural baselines are ~120 lines of NumPy with a seeded generator: no
  framework, no nondeterministic kernel, no GPU.
* UMAP is seeded via `random_state=0` and is an optional dependency; without
  it the row reports itself as unavailable instead of vanishing silently.

## Recording a replication

If you run this, the useful report is five lines: the commit you cloned, the
Python version, whether `run_all.sh` exited 0, the output of
`python reproduce/verify.py` (it prints each check), and anything that failed.
The checks are intentionally granular so that a single disagreement -- a
Wilson bound off by more than the tolerance, one method's specificity drifting
-- is attributable rather than a blanket "did not reproduce".
