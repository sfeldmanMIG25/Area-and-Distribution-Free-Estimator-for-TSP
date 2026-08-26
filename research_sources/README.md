# Local research source corpus

This directory is a local, provenance-preserving reading corpus for the paper. It is built from the bibliography in `paper_reference/references.bib` and from a small set of official/open-access URLs selected for this project.

## Layout

- `pdf/`: original downloaded PDF bytes, named by bibliography key.
- `text/`: `pdftotext -layout` extraction as `.txt` plus a `.md` wrapper containing the citation key, title, source URL, checksum, and extraction notes.
- `metadata/`: one JSON record per attempted source.
- `SOURCES_MANIFEST.csv`: complete inventory, including sources that were not legally or technically obtainable.

## Refresh

```powershell
& 'C:\Users\catst\AppData\Local\Python\pythoncore-3.14-64\python.exe' paper_tooling\fetch_research_sources.py
```

Add `--auto-openalex` only when a fresh OA discovery pass is wanted. The downloader accepts only PDF responses, checks the PDF signature, extracts text with `pdftotext -layout`, and records title mismatches instead of silently treating an unrelated document as the cited source.

## Use policy

Prefer these local transcriptions for agent work. The PDFs remain under their original publisher/repository terms; this corpus is for local research and reproducibility. Closed or unavailable primary sources are listed with their DOI/landing page and a reason rather than being obtained through a paywall bypass.
