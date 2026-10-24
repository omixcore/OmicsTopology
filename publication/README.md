# OmicsTopology — PLOS Computational Biology submission package

This folder contains the PLOS-formatted manuscript source, review copies, Supporting Information, editable vector figures, and a reference/DOI audit.

## Manuscript and Supporting Information

- `OmicsTopology_PLOS_Comp_Biol_manuscript.docx` — double-spaced Word manuscript with continuous line-number settings, Author Summary, figure captions near first citation, and Vancouver-style references.
- `OmicsTopology_PLOS_Comp_Biol_manuscript.pdf` — rendered review copy.
- `OmicsTopology_PLOS_Comp_Biol_manuscript.md` — reproducible manuscript text source.
- `OmicsTopology_PLOS_Comp_Biol_supporting_information.docx` / `.pdf` / `.md` — four supplementary tables and eleven supplementary figures.
- `reference_audit.md` — 42-reference citation and DOI audit.
- `submission_metadata.md` — funding disclosure for the online PLOS form and remaining author confirmations.
- `cover_letter.md` — journal-fit and impact letter draft; confirm author declarations before submission.

## Figures

`figures/` contains six main figures and `supplementary_figures/` contains eleven supplementary figures. SVG files are editable vector sources; each has PDF and PNG renderings.

## Rebuild

The GitHub Actions workflow expands compressed SVG/Markdown sources under `asset_sources/`, renders SVG/PDF/PNG artwork, builds Word documents from Markdown, and exports PDFs with LibreOffice. Original public GEO expression matrices are not redistributed; use the accessions in the manuscript to retrieve them.

## Submission status

Prepared for PLOS Computational Biology as a Research Article draft. PLOS directs authors to enter funding details in the online Financial Disclosure form. Exact funder URLs and funder roles require author confirmation. An archival DOI has not yet been assigned to the fixed publication snapshot.
