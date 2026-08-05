#!/usr/bin/env python3
"""Optimized, version-isolated Tiara TF-IDF training.

This workload is gzip/FASTA/k-mer document-frequency bound, not dense matrix
training. GPU transfer and variable-length string handling provide little
benefit, so this implementation uses persistent CPU workers + numba and
computes all k values for a stage in one pass.

Unlike the original TfidfWeighter.fit bug, each k-mer is counted once per
FASTA record (document), not only the first encountered k-mer.

For v2b, records are preserved at their actual 3-15 kb lengths by default,
matching hyperparameter_search_gpu feature construction. Use
--split-long-records only to reproduce fixed-fragment document semantics.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Iterator

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
try:
    from numba import njit
except ImportError:  # functional fallback; production tiara environment has numba
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda func: func

FIRST_FILES = ("mitochondria", "plastids", "bacteria", "eukarya", "archaea")
SECOND_FILES = ("mitochondria", "plastids")
FIRST_K = (4, 5, 6)
SECOND_K = (4, 5, 6, 7)


@njit(cache=True)
def add_document_frequency(raw: np.ndarray, k: int, counts: np.ndarray,
                           seen: np.ndarray, stamp: int) -> None:
    mask = (1 << (2 * k)) - 1
    code = 0
    valid = 0
    for base in raw:
        if base == 65:      # A
            value = 0
        elif base == 67:    # C
            value = 1
        elif base == 71:    # G
            value = 2
        elif base == 84:    # T
            value = 3
        else:
            code = 0
            valid = 0
            continue
        code = ((code << 2) | value) & mask
        valid += 1
        if valid >= k and seen[code] != stamp:
            seen[code] = stamp
            counts[code] += 1


def process_batch(payload: tuple[list[str], tuple[int, ...]]) -> tuple[int, dict[int, np.ndarray]]:
    sequences, kmers = payload
    totals = {k: np.zeros(4 ** k, dtype=np.uint64) for k in kmers}
    seen = {k: np.zeros(4 ** k, dtype=np.int32) for k in kmers}
    for doc_index, sequence in enumerate(sequences, 1):
        raw = np.frombuffer(sequence.upper().encode("ascii", errors="ignore"), dtype=np.uint8)
        for k in kmers:
            add_document_frequency(raw, k, totals[k], seen[k], doc_index)
    return len(sequences), totals


def simple_fasta_parser(handle) -> Iterator[tuple[str, str]]:
    header = None
    sequence: list[str] = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(sequence)
            header = line[1:].strip()
            sequence = []
        else:
            sequence.append(line)
    if header is not None:
        yield header, "".join(sequence)


def iter_fasta_records(paths: list[Path], fragment_len: int,
                       split_long_records: bool) -> Iterator[str]:
    for path in paths:
        with path.open() as handle:
            for _, sequence in simple_fasta_parser(handle):
                sequence = sequence.upper()
                if split_long_records and len(sequence) > fragment_len:
                    for start in range(0, len(sequence) - fragment_len + 1, fragment_len):
                        yield sequence[start:start + fragment_len]
                else:
                    yield sequence


def batches(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_stage(paths: list[Path], kmers: tuple[int, ...], workers: int,
                  batch_size: int, fragment_len: int,
                  split_long_records: bool, label: str) -> tuple[int, dict[int, np.ndarray]]:
    totals = {k: np.zeros(4 ** k, dtype=np.uint64) for k in kmers}
    n_documents = 0
    max_in_flight = max(2, workers * 2)
    pending: set[cf.Future] = set()
    started = time.time()

    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        for batch in batches(
            iter_fasta_records(paths, fragment_len, split_long_records), batch_size
        ):
            pending.add(executor.submit(process_batch, (batch, kmers)))
            if len(pending) >= max_in_flight:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    n, partial = future.result()
                    n_documents += n
                    for k in kmers:
                        totals[k] += partial[k]
                if n_documents and n_documents % 50000 < batch_size * max_in_flight:
                    print(f"[{label}] documents={n_documents:,} elapsed={(time.time()-started)/60:.1f}m", flush=True)
        for future in cf.as_completed(pending):
            n, partial = future.result()
            n_documents += n
            for k in kmers:
                totals[k] += partial[k]

    if n_documents == 0:
        raise RuntimeError(f"No FASTA records found for {label}")
    print(f"[{label}] complete: documents={n_documents:,} elapsed={(time.time()-started)/60:.1f}m", flush=True)
    return n_documents, totals


def save_tfidf_model(root: Path, stage: str, k: int, n_documents: int,
                     document_frequency: np.ndarray, fragment_len: int,
                     data_names: list[str]) -> None:
    model_dir = root / f"k{k}-{stage}-stage"
    model_dir.mkdir(parents=True, exist_ok=False)
    idf = (np.log((n_documents + 1) / (document_frequency.astype(np.float64) + 1)) + 1).astype(np.float32)
    np.save(model_dir / "model.npy", idf)
    with (model_dir / "params.txt").open("w") as handle:
        handle.write(f"k:{k}\n")
        handle.write(f"fragment_len:{fragment_len}\n")
        handle.write("verbose:True\n")
        handle.write("smooth:True\n")
        handle.write(f"N:{n_documents}\n")
        handle.write(f"data_names:{','.join(data_names)}")


def validate_inputs(data_dir: Path) -> dict[str, Path]:
    paths = {name: data_dir / f"{name}.fasta" for name in FIRST_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing/empty training FASTA:\n" + "\n".join(missing))
    return paths


def _canonical_config(config: dict) -> dict:
    """Subset of config that actually changes the TF-IDF OUTPUT.

    ``workers`` and ``batch_size`` are pure parallelism knobs -- the document
    frequencies they compute are identical regardless of how many processes or
    how large a batch is used. They are deliberately EXCLUDED from the signature
    so bumping the thread count (e.g. 16 -> 56) never invalidates a completed
    model and forces a multi-hour rebuild. Only inputs and k-mer/fragment
    semantics matter.
    """
    return {
        "data_dir": config["data_dir"],
        "fragment_len": config["fragment_len"],
        "split_long_records": config["split_long_records"],
        "first_k": list(config.get("first_k", FIRST_K)),
        "second_k": list(config.get("second_k", SECOND_K)),
    }


def config_signature(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(_canonical_config(config), sort_keys=True).encode()
    ).hexdigest()[:24]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="canonical train directory with five FASTA files")
    parser.add_argument("output_dir", type=Path, help="versioned TF-IDF root, e.g. tfidf_v2b")
    default_workers = min(16, max(1, (os.cpu_count() or 8) // 8))
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--fragment-len", type=int, default=5000,
                        help="saved compatibility metadata; also split length with --split-long-records")
    parser.add_argument("--k-first", type=int, nargs="+", default=list(FIRST_K),
                        help="first-stage k values (config-driven; default: 4 5 6)")
    parser.add_argument("--k-second", type=int, nargs="+", default=list(SECOND_K),
                        help="second-stage k values (config-driven; default: 4 5 6 7)")
    parser.add_argument("--split-long-records", action="store_true",
                        help="split records longer than fragment-len before document-frequency counting")
    parser.add_argument("--force", action="store_true",
                        help="back up an existing output and rebuild")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.fragment_len < 1:
        raise ValueError("workers, batch-size and fragment-len must be positive")
    first_k, second_k = tuple(args.k_first), tuple(args.k_second)
    for name, values in (("--k-first", first_k), ("--k-second", second_k)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique k values")
        if any(not 1 <= int(k) <= 8 for k in values):
            raise ValueError(f"{name} values must be integers in [1, 8]")
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = validate_inputs(data_dir)
    config = {
        "data_dir": str(data_dir),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "fragment_len": args.fragment_len,
        "split_long_records": args.split_long_records,
        "first_k": first_k,
        "second_k": second_k,
    }
    signature = config_signature(config)

    if output_dir.exists():
        manifest = output_dir / "tfidf_manifest.json"
        if manifest.is_file():
            try:
                existing = json.loads(manifest.read_text())
            except json.JSONDecodeError:
                existing = {}
            if existing.get("status") == "complete":
                stored_cfg = existing.get("config") or {}
                # Recompute the signature from the STORED config so a completed
                # model stays valid across (a) signature-formula changes and
                # (b) parallelism-only differences like workers/batch_size.
                # Fall back to the stored signature string for very old
                # manifests that predate this check.
                matches = (
                    bool(stored_cfg) and config_signature(stored_cfg) == signature
                ) or existing.get("signature") == signature
                if matches:
                    print(f"[resume] TF-IDF already complete: {output_dir}")
                    return 0
        if not args.force:
            raise FileExistsError(f"Output exists but is not matching/complete: {output_dir}; use --force")
        backup = output_dir.with_name(output_dir.name + ".backup_" + time.strftime("%Y%m%d_%H%M%S"))
        print(f"[safety] backup {output_dir} -> {backup}")
        os.replace(output_dir, backup)

    tmp = output_dir.with_name(output_dir.name + f".tmp.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    started = time.time()
    try:
        first_paths = [paths[name] for name in FIRST_FILES]
        second_paths = [paths[name] for name in SECOND_FILES]
        first_n, first_df = compute_stage(
            first_paths, first_k, args.workers, args.batch_size,
            args.fragment_len, args.split_long_records, "first",
        )
        for k in first_k:
            save_tfidf_model(tmp, "first", k, first_n, first_df[k],
                             args.fragment_len, list(FIRST_FILES))

        second_n, second_df = compute_stage(
            second_paths, second_k, args.workers, args.batch_size,
            args.fragment_len, args.split_long_records, "second",
        )
        for k in second_k:
            save_tfidf_model(tmp, "second", k, second_n, second_df[k],
                             args.fragment_len, list(SECOND_FILES))

        manifest = {
            "status": "complete",
            "signature": signature,
            "config": config,
            "first_documents": first_n,
            "second_documents": second_n,
            "seconds": time.time() - started,
            "implementation": "CPU multiprocessing + numba document frequency",
            "gpu_used": False,
        }
        (tmp / "tfidf_manifest.json").write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, output_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(f"TF-IDF complete -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
