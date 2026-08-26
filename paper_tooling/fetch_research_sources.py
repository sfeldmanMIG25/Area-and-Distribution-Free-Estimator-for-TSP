"""Download and transcribe obtainable sources cited by the manuscript.

The script is intentionally conservative: it uses curated official/open-access
URLs, optionally asks OpenAlex for an OA PDF candidate, validates the PDF magic
bytes and extracted title, and records every attempted bibliography entry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper_reference" / "references.bib"
OUT = ROOT / "research_sources"
PDF_DIR = OUT / "pdf"
TEXT_DIR = OUT / "text"
META_DIR = OUT / "metadata"


# These URLs were selected from publisher, repository, proceedings, or project
# pages. Related dissertations are explicitly labelled in the manifest.
CURATED_URLS: dict[str, tuple[str, str]] = {
    "aaai2023": (
        "https://ojs.aaai.org/index.php/AAAI/article/download/26120/25892",
        "official AAAI open-access PDF",
    ),
    "cavdar2014dissertation": (
        "https://repository.gatech.edu/bitstreams/6c212776-27a6-4d47-b520-ffa3367b6e1e/download",
        "institutional repository dissertation; contains the cited distribution-free estimator chapter",
    ),
    "cavdar2015distribution": (
        "https://repository.gatech.edu/bitstreams/6c212776-27a6-4d47-b520-ffa3367b6e1e/download",
        "institutional repository dissertation; related open copy of the cited distribution-free estimator",
    ),
    "chazelle1993": (
        "https://www.cs.princeton.edu/techreports/1991/336.pdf",
        "author-hosted Princeton technical-report predecessor to the cited published paper",
    ),
    "edmonds1965": (
        "https://www.cambridge.org/core/services/aop-cambridge-core/content/view/08B492B72322C4130AE800C0610E0E21/S0008414X00039419a.pdf/div-class-title-paths-trees-and-flowers-div.pdf",
        "official Cambridge Journal PDF",
    ),
    "frontiers2021": (
        "https://www.frontiersin.org/articles/10.3389/frobt.2021.689908/pdf",
        "official Frontiers open-access PDF",
    ),
    "freeman1975determining": (
        "https://dl.acm.org/doi/pdf/10.1145/360881.360919",
        "official ACM PDF reported as open access by OpenAlex",
    ),
    "harris2020numpy": (
        "https://arxiv.org/pdf/2006.10256",
        "author-posted arXiv version of the official open-access Nature paper",
    ),
    "kruskal1956": (
        "https://www.ams.org/proc/1956-007-01/S0002-9939-1956-0078686-7/S0002-9939-1956-0078686-7.pdf",
        "official AMS open-access PDF",
    ),
    "ke2017lightgbm": (
        "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
        "official NeurIPS proceedings PDF",
    ),
    "numba2015": (
        "https://numba.readthedocs.io/_/downloads/en/0.51.2/pdf/",
        "official Numba documentation PDF; fallback implementation reference, not the cited ACM paper",
    ),
    "akiba2019optuna": (
        "https://arxiv.org/pdf/1907.10902",
        "author-posted arXiv preprint of the cited Optuna work",
    ),
    "percus1996finite": (
        "https://scholar.cgu.edu/allon-percus/wp-content/uploads/sites/11/2013/08/tspprl.pdf",
        "author-hosted copy of the official APS paper",
    ),
    "agarwala2000fast": (
        "https://europepmc.org/articles/PMC311427?pdf=render",
        "Europe PMC open-access render of the official Genome Research paper",
    ),
    "bertsimas1990asymptotic": (
        "https://web.mit.edu/dbertsim/www/papers/AppliedProbability/An%20asymptotic%20determination%20of%20the%20minimum%20spanning%20tree%20and%20minimum%20matching%20constants%20in%20geometrical%20probability.pdf",
        "author-hosted MIT working-paper copy of the cited result",
    ),
    "scipy2020": (
        "https://arxiv.org/pdf/1907.10121",
        "author-posted arXiv preprint of the cited SciPy work",
    ),
    "sklearn2011": (
        "https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf",
        "official JMLR open-access PDF",
    ),
    "kou2022standard": (
        "https://api.drum.lib.umd.edu/server/api/core/bitstreams/3aa339e3-5e9f-46e4-9890-3485a9ad6f4c/content",
        "institutional repository dissertation; contains the standard-deviation estimator chapter",
    ),
    "kou2023sammon": (
        "https://api.drum.lib.umd.edu/server/api/core/bitstreams/3aa339e3-5e9f-46e4-9890-3485a9ad6f4c/content",
        "institutional repository dissertation; contains the Sammon-map estimator chapter",
    ),
    "varol2023neural": (
        "https://pubsonline.informs.org/doi/pdf/10.1287/trsc.2022.0015",
        "official INFORMS PDF if openly served",
    ),
}


def parse_bibtex(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    entries: list[dict[str, str]] = []
    for index, match in enumerate(starts):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        body = text[match.end() : cursor - 1 if depth == 0 else len(text)]
        fields: dict[str, str] = {}
        for line in body.splitlines():
            field = re.match(r"\s*([A-Za-z]+)\s*=\s*(.*?)(?:,\s*)?$", line)
            if not field:
                continue
            value = field.group(2).strip()
            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fields[field.group(1).lower()] = re.sub(r"\s+", " ", value).strip()
        fields.update({"type": match.group(1), "key": match.group(2)})
        entries.append(fields)
    return entries


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def title_matches(expected: str, extracted: str) -> bool:
    if not expected or not extracted:
        return True
    expected_words = [w for w in re.findall(r"[a-z0-9]+", expected.lower()) if len(w) > 3]
    haystack = normalized(extracted[:16000])
    hits = sum(normalized(word) in haystack for word in expected_words[:12])
    if len(expected_words) <= 3:
        return hits >= max(1, len(expected_words) - 1)
    return hits >= max(3, int(0.45 * min(12, len(expected_words))))


def openalex_pdf(doi: str) -> tuple[str, str]:
    if not doi:
        return "", ""
    url = "https://api.openalex.org/works/https://doi.org/" + doi
    request = Request(url, headers={"User-Agent": "local-research-corpus/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            record = json.load(response)
    except Exception:
        return "", ""
    location = record.get("best_oa_location") or {}
    return location.get("pdf_url") or "", record.get("title") or ""


def download(url: str, destination: Path) -> tuple[int, str, str]:
    request = Request(url, headers={"User-Agent": "local-research-corpus/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read()
            status = getattr(response, "status", 200)
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        return 0, "", str(error)
    if not data.startswith(b"%PDF"):
        return status, final_url, "response is not a PDF"
    destination.write_bytes(data)
    return status, final_url, ""


def extract_pdf(pdf: Path, text_path: Path, markdown_path: Path, entry: dict[str, str], url: str, checksum: str) -> tuple[bool, str]:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return False, "pdftotext is not available on PATH"
    raw_path = text_path.with_suffix(".raw.txt")
    result = subprocess.run(
        [pdftotext, "-layout", str(pdf), str(raw_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not raw_path.exists():
        return False, result.stderr.strip() or "pdftotext failed"
    extracted = raw_path.read_text(encoding="utf-8", errors="replace")
    if not extracted.strip():
        raw_path.unlink(missing_ok=True)
        return False, "PDF has no extractable text layer; OCR is not available in this environment"
    extracted = extracted.replace("\f", "\n\n--- PAGE BREAK ---\n\n")
    text_path.write_text(extracted, encoding="utf-8")
    raw_path.unlink(missing_ok=True)
    markdown = (
        f"# {entry.get('title', entry['key'])}\n\n"
        f"> Citation key: `{entry['key']}`  \n"
        f"> DOI: {entry.get('doi', '') or 'n/a'}  \n"
        f"> Download URL: {url}  \n"
        f"> SHA-256: `{checksum}`  \n\n"
        "---\n\n"
        f"{extracted}\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-openalex", action="store_true", help="try OpenAlex OA PDF discovery for unmapped DOI entries")
    parser.add_argument("--delay", type=float, default=0.15, help="delay between OpenAlex requests")
    args = parser.parse_args()

    for directory in (PDF_DIR, TEXT_DIR, META_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for entry in parse_bibtex(BIB):
        key = entry["key"]
        url, notes = CURATED_URLS.get(key, ("", ""))
        discovery_title = ""
        if not url and args.auto_openalex:
            url, discovery_title = openalex_pdf(entry.get("doi", ""))
            if url:
                notes = "OpenAlex best OA PDF candidate; title checked locally"
            time.sleep(args.delay)

        row = {
            "key": key,
            "type": entry.get("type", ""),
            "title": entry.get("title", ""),
            "author": entry.get("author", ""),
            "year": entry.get("year", ""),
            "doi": entry.get("doi", ""),
            "bib_url": entry.get("url", ""),
            "candidate_url": url,
            "status": "not_obtainable",
            "notes": notes or "no known openly downloadable copy was found",
            "pdf_path": "",
            "text_path": "",
            "markdown_path": "",
            "sha256": "",
            "bytes": "",
            "http_status": "",
            "final_url": "",
        }
        if url:
            pdf = PDF_DIR / f"{key}.pdf"
            text = TEXT_DIR / f"{key}.txt"
            markdown = TEXT_DIR / f"{key}.md"
            status, final_url, error = download(url, pdf)
            row["http_status"] = str(status)
            row["final_url"] = final_url
            if error:
                row["status"] = "download_failed"
                row["notes"] = error
            else:
                data = pdf.read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                row["sha256"] = checksum
                row["bytes"] = str(len(data))
                row["pdf_path"] = str(pdf.relative_to(ROOT))
                extraction_ok, extraction_error = extract_pdf(pdf, text, markdown, entry, final_url or url, checksum)
                if extraction_ok:
                    extracted = text.read_text(encoding="utf-8", errors="replace")
                    related = "dissertation" in notes or "documentation" in notes
                    if not title_matches(entry.get("title", ""), extracted) and not related:
                        row["status"] = "title_mismatch"
                        row["notes"] = "PDF downloaded, but extracted title did not match the bibliography record"
                    else:
                        row["status"] = "downloaded_related" if related else "downloaded"
                        row["text_path"] = str(text.relative_to(ROOT))
                        row["markdown_path"] = str(markdown.relative_to(ROOT))
                        if discovery_title and discovery_title.lower() != entry.get("title", "").lower():
                            row["notes"] += "; OpenAlex metadata title differs slightly"
                else:
                    row["status"] = "extraction_failed"
                    row["notes"] = extraction_error
        metadata_path = META_DIR / f"{key}.json"
        metadata_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(row)

    fieldnames = list(rows[0]) if rows else []
    with (OUT / "SOURCES_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    index_lines = [
        "# Downloaded source index",
        "",
        "Generated from `SOURCES_MANIFEST.csv`. Use the Markdown transcriptions first; the original PDFs remain in `pdf/`.",
        "",
        "| Key | Status | Markdown | PDF | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["status"].startswith("downloaded"):
            markdown_path = row["markdown_path"].replace("\\", "/")
            pdf_path = row["pdf_path"].replace("\\", "/")
            index_lines.append(
                f"| `{row['key']}` | `{row['status']}` | `{markdown_path}` | `{pdf_path}` | {row['notes']} |"
            )
    (OUT / "SOURCE_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"entries": len(rows), "status_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
