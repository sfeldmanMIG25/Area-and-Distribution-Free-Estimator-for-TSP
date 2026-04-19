"""Chunked archiver for large data folders.

Packs each source folder into ~45 MiB DEFLATE zip chunks so every part
stays under GitHub's 50 MB push-warning line. Archives are tracked as
regular git blobs (no LFS - see .gitattributes).

Guarantees:
    * Originals are never deleted, moved, or modified.
    * Each file is SHA-256 hashed before it goes into a chunk, and the
      hash is re-verified by streaming it back out of the zip after close.
    * A manifest CSV per source folder makes each archive self-describing.
    * Unpack refuses to overwrite existing files unless --force is given.

Parallelism:
    * Hashing, verification, and extraction use a ThreadPoolExecutor
      (hashlib / zip I/O release the GIL).
    * Compression (DEFLATE) uses a ProcessPoolExecutor - each worker owns
      its own chunk files (named part_wNN_NNNN.zip), so no shared-zip
      contention. Work is balanced across workers via greedy bin-packing
      by uncompressed bytes.

Usage:
    python chunker.py pack                     # archive default sources
    python chunker.py pack --only instances
    python chunker.py pack --dry-run
    python chunker.py pack --workers 4
    python chunker.py unpack                   # restore default sources
    python chunker.py unpack --only instances
    python chunker.py unpack --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = REPO_ROOT / "data_archives"

# 45 MiB - under GitHub's 50 MB non-LFS warning threshold.
CHUNK_LIMIT_BYTES = 45 * 1024 * 1024

# Folders to pack. Missing folders are skipped with a notice.
# Visuals / visualizations are intentionally excluded.
DEFAULT_SOURCES = [
    "instances",
    "solutions",
    "Generalized_TSP_Analysis/instances",
    "Generalized_TSP_Analysis/solutions",
    "tsplib_benchmark/instances",
    "tsplib_benchmark/ground_truth",
]

HASH_BUF = 1024 * 1024


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _mangle(src_rel: str) -> str:
    return src_rel.replace("/", "__").replace("\\", "__")


def _out_dir(src_rel: str) -> Path:
    return ARCHIVE_ROOT / _mangle(src_rel)


def sha256_of(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(HASH_BUF):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def sha256_of_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with zf.open(name, "r") as f:
        while chunk := f.read(HASH_BUF):
            h.update(chunk)
    return h.hexdigest()


def iter_files(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.is_file())


# ---------- parallel helpers ----------

def _parallel_hash(
    files: list[Path], n_workers: int
) -> dict[Path, tuple[str, int]]:
    result: dict[Path, tuple[str, int]] = {}
    if not files:
        return result
    total = len(files)
    done = 0
    # Threads: sha256 releases the GIL during its C-level digest loop.
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(sha256_of, str(p)): p for p in files}
        for fut in as_completed(futs):
            p = futs[fut]
            result[p] = fut.result()
            done += 1
            if done % 10000 == 0 or done == total:
                print(f"    hashed {done:,}/{total:,}")
    return result


def _partition_by_size(
    files: list[Path], pre: dict[Path, tuple[str, int]], n_parts: int
) -> list[list[Path]]:
    """Greedy bin-packing: assign each file to the currently-smallest bin."""
    n_parts = max(1, min(n_parts, len(files)))
    bins: list[list[Path]] = [[] for _ in range(n_parts)]
    totals = [0] * n_parts
    for p in sorted(files, key=lambda q: pre[q][1], reverse=True):
        idx = totals.index(min(totals))
        bins[idx].append(p)
        totals[idx] += pre[p][1]
    return [b for b in bins if b]


def _pack_worker(
    worker_idx: int,
    partition: list[tuple[str, str, str, int]],
    out_dir_str: str,
    chunk_limit: int,
) -> list[dict]:
    """Write one partition's chunks. Runs in a separate process.

    `partition` entries: (abs_path, arc_rel, sha256, size_bytes).
    """
    out_dir = Path(out_dir_str)
    rows: list[dict] = []
    chunk_idx = 0
    zf: zipfile.ZipFile | None = None
    zf_path: Path | None = None
    zf_bytes = 0

    def open_new() -> None:
        nonlocal zf, zf_path, zf_bytes, chunk_idx
        chunk_idx += 1
        zf_path = out_dir / f"part_w{worker_idx:02d}_{chunk_idx:04d}.zip"
        zf = zipfile.ZipFile(
            zf_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        )
        zf_bytes = 0

    open_new()
    assert zf is not None and zf_path is not None

    for abs_path, rel, sha, size in partition:
        if zf_bytes > 0 and zf_bytes + size > chunk_limit:
            zf.close()
            open_new()
        zf.write(abs_path, arcname=rel)
        zf_bytes += size
        rows.append(
            {
                "relative_path": rel,
                "sha256": sha,
                "size_bytes": size,
                "chunk": zf_path.name,
            }
        )
    zf.close()
    return rows


def _verify_chunk(
    chunk_path_str: str, rows: list[dict]
) -> tuple[str, list[str]]:
    errors: list[str] = []
    chunk_path = Path(chunk_path_str)
    with zipfile.ZipFile(chunk_path, "r") as zf_r:
        names = set(zf_r.namelist())
        for row in rows:
            rel = row["relative_path"]
            if rel not in names:
                errors.append(f"{chunk_path.name} missing {rel}")
                continue
            got = sha256_of_zip_member(zf_r, rel)
            if got != row["sha256"]:
                errors.append(f"{chunk_path.name}:{rel} hash mismatch")
    return chunk_path.name, errors


def _extract_chunk(
    chunk_path_str: str,
    rows: list[dict],
    dst_str: str,
    force: bool,
) -> tuple[int, int, list[str]]:
    chunk_path = Path(chunk_path_str)
    dst = Path(dst_str)
    extracted = 0
    skipped = 0
    errors: list[str] = []
    with zipfile.ZipFile(chunk_path, "r") as zf_r:
        for row in rows:
            rel = row["relative_path"]
            target = dst / rel
            if target.exists() and not force:
                existing_hash, _ = sha256_of(str(target))
                if existing_hash == row["sha256"]:
                    skipped += 1
                    continue
                errors.append(
                    f"{rel} exists with different hash (use --force)"
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            with zf_r.open(rel, "r") as src_f, tmp.open("wb") as dst_f:
                while chunk := src_f.read(HASH_BUF):
                    dst_f.write(chunk)
            got, _ = sha256_of(str(tmp))
            if got != row["sha256"]:
                tmp.unlink(missing_ok=True)
                errors.append(f"hash mismatch on extract: {rel}")
                continue
            tmp.replace(target)
            extracted += 1
    return extracted, skipped, errors


# ---------- pack ----------

def pack_folder(
    src_rel: str, workers: int, dry_run: bool = False
) -> bool:
    src = REPO_ROOT / src_rel
    if not src.is_dir():
        print(f"[skip] {src_rel} - not found")
        return True

    files = iter_files(src)
    if not files:
        print(f"[skip] {src_rel} - empty")
        return True

    out_dir = _out_dir(src_rel)
    manifest_path = out_dir / "manifest.csv"

    total_bytes = sum(f.stat().st_size for f in files)
    print(
        f"[plan] {src_rel}: {len(files):,} files, "
        f"{total_bytes / 1024 / 1024:.1f} MiB -> {out_dir}  "
        f"(workers={workers})"
    )
    if dry_run:
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  hashing {len(files):,} source files (parallel)...")
    pre = _parallel_hash(files, workers)

    partitions = _partition_by_size(files, pre, workers)
    print(f"  writing chunks across {len(partitions)} worker(s)...")

    manifest_rows: list[dict] = []
    if len(partitions) == 1:
        # Avoid process-spawn overhead for tiny folders.
        rows = _pack_worker(
            0,
            [
                (
                    str(p),
                    p.relative_to(src).as_posix(),
                    pre[p][0],
                    pre[p][1],
                )
                for p in partitions[0]
            ],
            str(out_dir),
            CHUNK_LIMIT_BYTES,
        )
        manifest_rows.extend(rows)
    else:
        args_list = []
        for i, part in enumerate(partitions):
            payload = [
                (
                    str(p),
                    p.relative_to(src).as_posix(),
                    pre[p][0],
                    pre[p][1],
                )
                for p in part
            ]
            args_list.append((i, payload, str(out_dir), CHUNK_LIMIT_BYTES))
        with ProcessPoolExecutor(max_workers=len(partitions)) as ex:
            futs = [ex.submit(_pack_worker, *a) for a in args_list]
            for fut in as_completed(futs):
                manifest_rows.extend(fut.result())

    # Warn on oversized files (single file > chunk limit ends up alone).
    for row in manifest_rows:
        if row["size_bytes"] > CHUNK_LIMIT_BYTES:
            print(
                f"  [warn] {row['relative_path']} is "
                f"{row['size_bytes'] / 1024 / 1024:.1f} MiB - exceeds "
                "chunk limit; its zip will too."
            )

    # Verify every chunk in parallel.
    by_chunk: dict[str, list[dict]] = {}
    for row in manifest_rows:
        by_chunk.setdefault(row["chunk"], []).append(row)

    print(f"  verifying {len(by_chunk)} chunk(s)...")
    ok = True
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_verify_chunk, str(out_dir / cn), rows)
            for cn, rows in by_chunk.items()
        ]
        for fut in as_completed(futs):
            _, errors = fut.result()
            for e in errors:
                print(f"  [FAIL] {e}")
                ok = False

    if not ok:
        print(f"[FAIL] {src_rel} - verification failed. Originals untouched.")
        return False

    # Stable manifest order (by relative path).
    manifest_rows.sort(key=lambda r: r["relative_path"])
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(
            mf, fieldnames=["relative_path", "sha256", "size_bytes", "chunk"]
        )
        w.writeheader()
        w.writerows(manifest_rows)

    archive_bytes = sum((out_dir / c).stat().st_size for c in by_chunk)
    print(
        f"[ok]   {src_rel} -> {len(by_chunk)} chunk(s), "
        f"{archive_bytes / 1024 / 1024:.1f} MiB on disk "
        f"(src {total_bytes / 1024 / 1024:.1f} MiB, "
        f"ratio {total_bytes / max(1, archive_bytes):.2f}x)"
    )
    return True


# ---------- unpack ----------

def unpack_folder(
    src_rel: str, workers: int, force: bool = False, dry_run: bool = False
) -> bool:
    dst = REPO_ROOT / src_rel
    out_dir = _out_dir(src_rel)
    manifest_path = out_dir / "manifest.csv"

    if not manifest_path.exists():
        print(f"[skip] {src_rel} - no archive at {out_dir}")
        return True

    rows: list[dict] = []
    with manifest_path.open(newline="", encoding="utf-8") as mf:
        for row in csv.DictReader(mf):
            rows.append(row)
    total_bytes = sum(int(r["size_bytes"]) for r in rows)
    by_chunk: dict[str, list[dict]] = {}
    for r in rows:
        by_chunk.setdefault(r["chunk"], []).append(r)

    print(
        f"[plan] unpack {src_rel}: {len(rows):,} files, "
        f"{total_bytes / 1024 / 1024:.1f} MiB from {len(by_chunk)} chunk(s) "
        f"-> {dst}  (workers={workers})"
    )
    if dry_run:
        return True

    dst.mkdir(parents=True, exist_ok=True)

    total_extracted = 0
    total_skipped = 0
    ok = True
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for chunk_name, chunk_rows in by_chunk.items():
            chunk_path = out_dir / chunk_name
            if not chunk_path.exists():
                print(f"  [FAIL] missing chunk {chunk_name}")
                ok = False
                continue
            futs.append(
                ex.submit(
                    _extract_chunk,
                    str(chunk_path),
                    chunk_rows,
                    str(dst),
                    force,
                )
            )
        for fut in as_completed(futs):
            extracted, skipped, errors = fut.result()
            total_extracted += extracted
            total_skipped += skipped
            for e in errors:
                print(f"  [FAIL] {e}")
                ok = False

    if not ok:
        print(f"[FAIL] {src_rel} - unpack had errors.")
        return False

    print(
        f"[ok]   {src_rel} - extracted {total_extracted:,}, "
        f"skipped {total_skipped:,} already-present"
    )
    return True


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    for name, helptxt in (
        ("pack", "archive sources into chunked zips"),
        ("unpack", "restore sources from chunked zips"),
    ):
        sp = sub.add_parser(name, help=helptxt)
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--only", nargs="+", metavar="PATH")
        sp.add_argument(
            "--workers",
            type=int,
            default=_default_workers(),
            help=f"parallel workers (default: {_default_workers()})",
        )
        if name == "unpack":
            sp.add_argument(
                "--force",
                action="store_true",
                help="overwrite existing files (default: keep + verify hash)",
            )

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return 2

    sources = args.only if args.only else DEFAULT_SOURCES
    ARCHIVE_ROOT.mkdir(exist_ok=True)
    workers = max(1, args.workers)

    all_ok = True
    if args.cmd == "pack":
        for src in sources:
            if not pack_folder(src, workers=workers, dry_run=args.dry_run):
                all_ok = False
    else:
        for src in sources:
            if not unpack_folder(
                src,
                workers=workers,
                force=args.force,
                dry_run=args.dry_run,
            ):
                all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
