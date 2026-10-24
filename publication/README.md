# OmicsTopology — PLOS Computational Biology submission package

This folder contains the PLOS-formatted manuscript source, review copies, Supporting Information, editable vector figures, and a reference/DOI audit.

## Manuscript and Supporting Information

- `OmicsTopology_PLOS_Comp_Biol_manuscript.docx` — double-spaced Word manuscript with continuous line-number settings, Author Summary, figure captions near first citation, and Vancouver-style references.
- `OmicsTopology_PLOS_Comp_Biol_manuscript.pdf` — rendered review copy.
- `OmicsTopology_PLOS_Comp_Biol_manuscript.md` — reproducible manuscript text source.
- `OmicsTopology_PLOS_Comp_Biol_supporting_information.docx` / `.pdf` / `.md` — four supplementary tables and eleven supplementary figures.
- `reference_audit.md` — 42-reference citation and DOI audit.
- `verify_derived_records.py` — standard-library audit that recalculates the committed ordering scores, enrichment statistics, confidence intervals, and multiple-testing corrections from the archived derived records.
- `submission_metadata.md` — funding disclosure for the online PLOS form and remaining author confirmations.
- `cover_letter.md` — journal-fit and impact letter draft; confirm author declarations before submission.

## Figures

- [Combined main-figure PDF](Main_figures.pdf)
- [Combined supplementary-figure PDF](Supplementary_figures.pdf)

`figures/` contains six main figures and `supplementary_figures/` contains eleven supplementary figures. SVG files are editable vector sources; each has PDF and PNG renderings. Each main figure also has a 600-ppi RGB TIFF (`Fig1.tif`–`Fig6.tif`) for PLOS figure upload.

## Rebuild

The GitHub Actions workflow uses the committed Markdown and SVG files as the sources of record. It builds the manuscript and Supporting Information DOCX files, exports review PDFs with LibreOffice, renders each SVG to PDF/PNG, creates 600-ppi TIFF uploads for all six main figures, merges the six main figures and eleven supplementary figures into combined PDFs, validates the sequential reference list and in-text citation coverage, checks TIFF resolution and size, and runs verify_derived_records.py. The verifier checks the arithmetic and internal consistency of committed derived records; it does not claim to replace a fresh download-and-rerun of every original GEO matrix analysis. Generated outputs are committed as `omixcore`. The one-time transfer payload and legacy `asset_sources` chunks are removed after the first build. Original public GEO expression matrices are not redistributed; use the accessions in the manuscript to retrieve them.

## Submission status

Prepared for PLOS Computational Biology as a Research Article draft. PLOS directs authors to enter funding details in the online Financial Disclosure form. Exact funder URLs and funder roles require author confirmation. An archival DOI has not yet been assigned to the fixed publication snapshot.
