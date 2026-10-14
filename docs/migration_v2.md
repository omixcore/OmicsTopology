# Migrating to version 2

| API | Version 1 | Version 2 |
|---|---|---|
| `analyse` count | L1/L2 gate and optional Mantel r≥0.5 | maximum-surrogate detection and resampling agreement |
| Missing count | integer retained with a refusal status | `n_cycles=None`, raw `candidate_count` retained |
| Status | counted / underpowered / inconsistent / no_cycle | counted / unresolved / no_cycle |
| Design | can determine count status | separate `design_association` |
| Null geometry | PCA scores substituted for features | scores reconstructed through feature loadings |
| Backend | implicit fallback possible | pinned backend; failures are explicit |
| Axis label | independent_axes | strong_axis_effect, with no independence claim |
| Categories | arbitrary numeric cyclic coding | equal distance between different labels |
| Phase residuals | subtract arithmetic mean angles | wrapped circular offsets and residuals |
| Phase coherence | concentration of block means | agreement across reliably phased genes |
| Mantel permutation | independent distance-entry shuffle | joint sample row/column permutation |
| Uncertainty | matched difference interval misread as ratio | separate ratio and difference, optional whole-unit bootstrap |

Use `omictopo.legacy_protocol` and `omictopo.legacy_axes` only to reproduce v1
records. Legacy `count_cycles` also remains unchanged for historical results;
new work should use `analyse`, `spectral_count` or `stable_cycle_count`.
Historical benchmark scripts import legacy APIs explicitly. Their metadata and
numbers do not constitute an independent assessment of the corrected methods.

Defaults require the fast backend. Old `gap_threshold` and `r_threshold`
arguments are deliberately not accepted by `analyse`; they no longer define
the estimand. Independent-unit labels for resampling are separate from category
labels used for phase fitting and from strata used to restrict permutation.
