# Bibliography and acquisition audit

Generated during the local source-corpus setup on 2026-08-15.

## Inventory

- The parser found 71 keyed bibliography entries in `paper_reference/references.bib`.
- 17 original PDFs are preserved under `pdf/`; 16 have non-empty `pdftotext -layout` transcriptions under `text/`.
- 11 entries have an exact or author-posted primary/official copy; 5 are explicitly marked `downloaded_related` because the local file is a dissertation or documentation source containing the cited material.
- 50 entries have no known openly downloadable copy in the curated pass; 4 candidate URLs failed; 1 downloaded PDF has no text layer and is marked `extraction_failed`.

## Records needing care

- `steele1988growth`: the DOI currently in the bibliography (`10.1214/aop/1176991590`) resolves to a different title in the metadata service. Search evidence identifies `10.1214/aop/1176991596` as the DOI associated with *Growth Rates of Euclidean Minimal Spanning Trees with Power Weighted Edges*. Do not change the `.bib` without checking the original journal record.
- `smithmiles2010`: the DOI currently in the bibliography resolved to *Time-Bounded Sequential Parameter Optimization* in OpenAlex, while the bibliography title is *Understanding TSP difficulty by learning from evolved instances*. Verify the proceedings record before relying on the DOI.
- `harris2020numpy`: the keyed NumPy entry is valid, but an unkeyed orphan block follows it in `references.bib` beginning with a title about the Euclidean TSP tour-length distribution. The source downloader intentionally ignores that orphan because it is not a valid citable entry.
- `cavdar2015distribution`: the local file is the author’s/institution’s dissertation copy containing the distribution-free estimator chapter, not the published EJOR version.
- `kou2022standard` and `kou2023sammon`: the local files are chapters in an institutional dissertation, not the publisher versions of the two journal articles.
- `numba2015`: the local file is the official Numba documentation PDF because the ACM paper was not openly served; it is marked `downloaded_related`.
- `chazelle1993`: an author-hosted Princeton technical-report predecessor was obtainable, but it has no extractable text layer in this environment; it is retained as a PDF and marked `extraction_failed`.

The authoritative machine-readable status for every entry is `SOURCES_MANIFEST.csv`; use `SOURCE_INDEX.md` to navigate successful transcriptions.
