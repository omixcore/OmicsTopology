# Cover letter draft — PLOS Computational Biology

> **Author review required before submission.** Confirm that all authors approve the final manuscript, that it is not under consideration elsewhere, and that the corresponding author’s ORCID is entered in the submission system. Remove this note before uploading the letter.

Dear Editors of *PLOS Computational Biology*,

Please consider our manuscript, “OmicsTopology: calibrated topology and phase diagnostics for transcriptomic state spaces,” as a Research Article for *PLOS Computational Biology*.

The manuscript presents OmicsTopology, an open-source workflow for evaluating persistent loop-like structure in transcriptomic sample spaces while keeping several distinct questions separate: candidate H1 detection under a fitted reference, stability under sample removal, association with experimental axes, and gene-level circular-phase diagnostics. The workflow is assessed using frozen simulations and public transcriptome datasets, with source records, preprocessing details, numerical summaries, and figure sources supplied for inspection and reproduction.

A central result is that visible time-associated geometry does not, by itself, establish a stable positive H1 detection or identify an independent biological oscillator. The simulations quantify recovery limits for weak, unequal, and sparsely sampled structures; the public-data analyses show no general ordering advantage for the H1-based method. A separate cross-experiment gene-set analysis identifies an association with published heat-responsive genes, while the manuscript explicitly avoids interpreting this as proof of clock-specific regulation or a causal clock-resetting mechanism.

We believe the work is suitable for *PLOS Computational Biology* because it combines a computational method with explicit calibration, failure-boundary analysis, comparisons against established geometric baselines, and a biologically motivated application using public data. The intended contribution is a reproducible framework for hypothesis generation and methodological assessment, not an unsupported claim that persistent loops count biological oscillators.

The OmicsTopology source code, configuration, derived numerical records, and figure sources for this submission are publicly available under the MIT License at https://github.com/omixcore/OmicsTopology/tree/submission/plos-computational-biology. The manuscript identifies the public GEO accessions used in the analyses, and the accompanying Supporting Information provides numerical tables, source-file checksums, and sensitivity results.

Thank you for considering this manuscript.

Sincerely,

Fei Liang  
Corresponding author  
Email: fyliangfei@163.com
