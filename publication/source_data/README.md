Derived source records for the OmicsTopology PLOS Computational Biology manuscript.

These files contain numerical and gene-level records used to create the current figures and supplementary tables. They are derived from public Gene Expression Omnibus datasets; the original expression matrices are not redistributed here. Original accessions and checksums are listed in download_sha256.json and Supporting Information Table S3.

Key files:
- Figure3_*_PCA.tsv and FigureS2_trajectory_coordinates.json: coordinates used in trajectory/PCA visualizations.
- phase_candidates_TOT.json and phase_candidates_TR.json: selected candidate gene records for total RNA and translating RNA.
- external_gene_matches_*.json: gene-level matches to published heat-response and genotype contrasts.
- external_ordering_recomputed.json: matched-input ordering scores and sample coverage.
- heat_enrichment_recomputed.json and candidate_cutoff_sensitivity.json: enrichment counts and threshold-sensitivity results.
- design.json, environment.json, delivery_checks.json, v2_validation_plan.json, v2_real_validation.json, v2_verification.txt: settings, environment, and verification metadata.

Version provenance: the frozen v2 record preserves the original installed-distribution metadata value (0.1.0) and separately records that the analyzed source revision declares OmicsTopology 0.2.0. The source revision and algorithm digest identify the code used; the metadata mismatch is documented rather than silently overwritten.

The candidate-cutoff sensitivity record includes 95% log-odds Wald intervals for the heat-response tests plotted in S11 Fig; genotype-under-heat sensitivity rows retain odds ratios and adjusted tests but omit intervals because they are not plotted.
