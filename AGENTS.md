# Project Agent Guide

## Purpose

This repository contains the GART 2.0 research code and the LaTeX manuscript for distribution-independent TSP tour-length estimation. Use the local corpus in `research_sources/` before searching the web.

## Skill routing

| Work | Skill | Project-specific focus |
| --- | --- | --- |
| Draft, revise, outline, abstract, or format the paper | `academic-paper` | Preserve the existing LaTeX structure and author voice; use `citation-check` for bibliography audits. |
| Literature search, source verification, and evidence synthesis | `deep-research` | Start from `paper_reference/references.bib`; verify every source independently and record inaccessible items. |
| ML/model/data-pipeline code | `mle-workflow` | Keep split policy, leakage checks, reproducibility, artifact provenance, and size/dimension slices explicit. |
| Any material code or paper change | `verification-loop` | Run targeted tests/lint plus the project’s benchmark/table and LaTeX checks where applicable. |
| PDF acquisition, extraction, or inspection | `pdf` | Keep the source PDF, extract with `pdftotext -layout`, and retain provenance/licensing notes. |

## Repository facts

- Manuscript: `paper_reference/Area_Free_Main.tex`; bibliography: `paper_reference/references.bib`.
- Paper workflow: after manuscript edits, compile from `paper_reference/` with `latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error Area_Free_Main.tex` and report page count and PDF size.
- Core estimators and features: `classical_region_estimators.py`, `baselines_calibrated.py`, `feature_creator_v3.py`, `mst_utils.py`, and `lgbm_model_v3/`.
- Bounds/solvers: `held_karp_1tree.py` and `solvers/`.
- Benchmark and paper tooling: `run_benchmark_2D_all.py`, `run_benchmark_ND_final.py`, `tsplib_benchmark/`, and `paper_tooling/`.
- Use `C:\Users\catst\AppData\Local\Python\pythoncore-3.14-64\python.exe`; bare `python` is a Windows Store stub.
- Do not read large CSVs, binary model artifacts, instance/solution directories, or images into context. Query them with headers, row counts, or project scripts.
- The worktree may contain user changes. Preserve them and make surgical edits only.

## Research corpus

- `research_sources/pdf/` stores downloaded PDFs.
- `research_sources/text/` stores layout-aware `.txt` and provenance-headed `.md` transcriptions.
- `research_sources/metadata/` stores per-source JSON metadata.
- `research_sources/SOURCES_MANIFEST.csv` records every bibliography entry, acquisition status, URL, checksum, and reason when unavailable.
- Run `C:\Users\catst\AppData\Local\Python\pythoncore-3.14-64\python.exe paper_tooling/fetch_research_sources.py` to refresh the corpus. It only uses explicitly curated official/open-access URLs and optional OpenAlex OA discovery; it does not bypass paywalls.
- Do not add a citation to the manuscript from a source marked `title_mismatch`, `download_failed`, or `not_obtainable` without resolving the record first.
