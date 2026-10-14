# Version 2 inference protocol

## Detection

Standardise the supplied sample-by-feature matrix. Compute the requested
feature-space distance or dissimilarity matrix. Run an explicitly pinned H1
backend without filtration truncation. Normalise each persistence length by
`sqrt(mean(D**2)/2)`.

For the surrogate model, retain the full-rank SVD loadings of the standardised
matrix, draw independent Gaussian principal scores scaled by empirical score
standard deviations, and reconstruct feature vectors through those loadings.
Re-standardise the reconstructed features and apply exactly the same distance
and H1 computation. This fixes the v1 mismatch that replaced features with
principal scores even when using correlation distances.

Each observed bar receives a Monte Carlo tail against the **maximum** H1 length
in each surrogate draw: `(1 + exceedances)/(1 + draws)`. Bars with tail ≤alpha
are detected. This is maximum-statistic calibration conditional on the fitted
Gaussian reference, not a distribution-free or biological error guarantee.
Comparable signal bars never exclude one another through an L1/L2 gate.

## Stability and abstention

Each reduced cloud refits its surrogate distribution. Retain whole supplied
independent units, stratify rows if requested, or report ordinary row removal
as a geometric sensitivity check. Defaults use 12 resamples and require 75%
agreement with the full-cloud detected count. Too few units to resample, no
resampling, or insufficient agreement returns `unresolved` and `n_cycles=None`.

`counted` reports a stable positive detection. `no_cycle` reports a stable zero
detection under the chosen analysis, **not absence of a biological process**.
Count agreement and its Wilson interval describe variability of this resampling
procedure, not confidence coverage for the true Betti number. Use the full
spectrum and the committed validation to assess limitations.

## Design and axis diagnostics

Design association is separate from the topology count. It never vetoes a
significant but modest phase effect or imposes a single-axis model on two loops.
Mantel p-values are omitted by default because exchangeability is a design
assumption. When enabled, permutations move labels via joint row/column
permutations and may be restricted by strata. If the design coordinate cannot
change under those exchanges, no p-value is reported.

Second-axis labels describe a strong distance effect, common circular phase
modulation, gene-specific phase changes, a small unresolved effect, or an
undetermined result. They do not establish latent-axis independence. Categories
use equal between-category distance; a cyclic axis requires an explicit period.
Harmonic fits include an intercept and phase fits with inadequate coverage or
weak explained variance are withheld. Phase offsets and residuals are circular,
and common-shift shares use circular dispersion, not ordinary squared angles.

Matched-distance ratios compare pairs at equal reference phase and different
second-axis labels with all pairs at different reference phases. The denominator
is not one adjacent step. Ratio and absolute-difference bootstrap intervals are
computed separately, and supplied independent units are resampled intact.

## What remains unvalidated

Two stable H1 features do not establish a torus, do not assign a particular
feature to a specific recorded axis, and do not enumerate biological oscillators.
Weak second loops can be missed. A sparse three-cycle product can remain
undetected. An adequately sampled true biological double-cycle state space and
independent experimental series are still needed for broader interpretation.
