#!/usr/bin/env python3
"""Publish trained params from the /data working tree back into the CODE package.

Closed loop:  code lives at ~/ (repo)  <->  data/runs live at /data/...

The train stage writes TF-IDF + NNet params under /data (BASE). When training is
fully done, this script snapshots those params into the package itself
(``tiara/models/...``) so the repo carries a runnable copy. Later, when tiara2 is
packaged, inference can point straight at these in-package params.

This generalizes the user's two manual scripts (copy_v2_models_to_git.sh +
copy_v2_tfidf_to_git.sh) into ONE config-driven, validated, resumable copy:

  * validates completeness BEFORE touching the destination
      - NNet:  exactly <first_count> first_*.pkl + <second_count> second_*.pkl
               + a non-empty training_manifest.json
      - TF-IDF: tfidf_manifest.json status == "complete"; every k-stage folder
                has model.npy whose shape == (4**k,) and a params.txt
  * timestamped backup if the destination already exists
  * atomic publish (copy into a temp dir, then rename into place)
  * writes + verifies SHA256SUMS in each destination
  * pure standard library (reads .npy headers directly; no numpy needed), so it
    runs in ANY env, incl. base Python -- it never re-pins your ML stack
  * NO git add/commit/push (same policy as your originals)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(msg, flush=True)


def npy_shape(path: Path) -> tuple:
    """Read an .npy header WITHOUT numpy and return its shape tuple.

    .npy format: b'\\x93NUMPY' + ver(2) + header_len + ascii dict literal that
    contains e.g. {'descr':'<f8','fortran_order':False,'shape':(256,)}.
    """
    with open(path, "rb") as fh:
        magic = fh.read(6)
        if magic != b"\x93NUMPY":
            fail(f"not a .npy file: {path}")
        major, _minor = fh.read(1), fh.read(1)
        if major == b"\x01":
            hlen = int.from_bytes(fh.read(2), "little")
        else:  # v2/v3 use 4-byte header length
            hlen = int.from_bytes(fh.read(4), "little")
        header = fh.read(hlen).decode("latin1")
    meta = ast.literal_eval(header.strip())
    return tuple(meta["shape"])


def write_and_verify_checksums(dest: Path, names: list[str]) -> None:
    """Write SHA256SUMS (relative paths) for the given files, then verify."""
    lines = []
    for rel in sorted(names):
        h = hashlib.sha256()
        with open(dest / rel, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {rel}")
    (dest / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    # verify
    for line in lines:
        digest, rel = line.split("  ", 1)
        h = hashlib.sha256()
        with open(dest / rel, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            fail(f"checksum mismatch after publish: {rel}")
    info(f"  SHA256SUMS written + verified ({len(lines)} files)")


def backup_if_present(dest: Path, do_backup: bool) -> None:
    if dest.exists() and any(dest.iterdir()):
        if not do_backup:
            fail(f"destination exists and --no-backup given: {dest}")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = dest.with_name(f"{dest.name}.backup_{stamp}")
        info(f"  [safety] existing destination -> {bak}")
        shutil.move(str(dest), str(bak))


def atomic_publish(staged: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(staged), str(dest))  # same filesystem: atomic rename


# --------------------------------------------------------------------------- #
# NNet models
# --------------------------------------------------------------------------- #
def publish_nnet(src: Path, dest: Path, first_count: int, second_count: int,
                 do_backup: bool, dry_run: bool) -> None:
    info(f"\n== NNet models ==\n  src : {src}\n  dest: {dest}")
    if not src.is_dir():
        fail(f"NNet source dir not found: {src}")
    first = sorted(p for p in src.glob("first_*.pkl") if p.is_file())
    second = sorted(p for p in src.glob("second_*.pkl") if p.is_file())
    manifest = src / "training_manifest.json"
    if len(first) != first_count:
        fail(f"first-stage models: found {len(first)}, expected {first_count}")
    if len(second) != second_count:
        fail(f"second-stage models: found {len(second)}, expected {second_count}")
    if not (manifest.is_file() and manifest.stat().st_size > 0):
        fail(f"missing/empty training_manifest.json in {src}")
    info(f"  validated: first={len(first)} second={len(second)} + manifest")

    files = first + second + [manifest]
    if dry_run:
        for f in files:
            info(f"  [dry-run] would copy {f.name}")
        return

    staged = dest.with_name(f".{dest.name}.staging.{os.getpid()}")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    try:
        for f in files:
            shutil.copy2(f, staged / f.name)
        write_and_verify_checksums(staged, [f.name for f in files] )
        backup_if_present(dest, do_backup)
        atomic_publish(staged, dest)
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    info(f"  published {len(files)} files -> {dest}")


# --------------------------------------------------------------------------- #
# TF-IDF models
# --------------------------------------------------------------------------- #
def publish_tfidf(src: Path, dest: Path, k_first: list[int], k_second: list[int],
                  do_backup: bool, dry_run: bool) -> None:
    info(f"\n== TF-IDF models ==\n  src : {src}\n  dest: {dest}")
    if not src.is_dir():
        fail(f"TF-IDF source dir not found: {src}")
    manifest = src / "tfidf_manifest.json"
    if not manifest.is_file():
        fail(f"missing manifest: {manifest}")
    try:
        status = json.loads(manifest.read_text()).get("status")
    except Exception as exc:
        fail(f"cannot parse {manifest}: {exc}")
    if status != "complete":
        fail(f"TF-IDF not complete: status={status!r}")

    folders = [f"k{k}-first-stage" for k in k_first] + \
              [f"k{k}-second-stage" for k in k_second]
    ks = {f"k{k}-first-stage": k for k in k_first}
    ks.update({f"k{k}-second-stage": k for k in k_second})
    for folder in folders:
        fdir = src / folder
        model, params = fdir / "model.npy", fdir / "params.txt"
        if not (model.is_file() and params.is_file()):
            fail(f"missing model.npy/params.txt in {fdir}")
        shape = npy_shape(model)
        expected = (4 ** ks[folder],)
        if shape != expected:
            fail(f"{model}: shape={shape}, expected={expected}")
    info(f"  validated: {len(folders)} folders + manifest (status=complete)")

    if dry_run:
        for folder in folders:
            info(f"  [dry-run] would copy {folder}/")
        return

    staged = dest.with_name(f".{dest.name}.staging.{os.getpid()}")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    checksum_names: list[str] = []
    try:
        for folder in folders:
            shutil.copytree(src / folder, staged / folder)
            checksum_names += [f"{folder}/model.npy", f"{folder}/params.txt"]
        shutil.copy2(manifest, staged / manifest.name)
        checksum_names.append(manifest.name)
        write_and_verify_checksums(staged, checksum_names)
        backup_if_present(dest, do_backup)
        atomic_publish(staged, dest)
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    info(f"  published {len(folders)} folders -> {dest}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publish trained params into the package")
    ap.add_argument("--which", choices=["both", "nnet", "tfidf"], default="both")
    ap.add_argument("--nnet-src"); ap.add_argument("--nnet-dest")
    ap.add_argument("--tfidf-src"); ap.add_argument("--tfidf-dest")
    ap.add_argument("--first-count", type=int, default=3)
    ap.add_argument("--second-count", type=int, default=4)
    ap.add_argument("--k-first", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--k-second", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    do_backup = not a.no_backup

    if a.which in ("both", "nnet"):
        if not (a.nnet_src and a.nnet_dest):
            fail("--nnet-src and --nnet-dest are required for nnet")
        publish_nnet(Path(a.nnet_src), Path(a.nnet_dest),
                     a.first_count, a.second_count, do_backup, a.dry_run)
    if a.which in ("both", "tfidf"):
        if not (a.tfidf_src and a.tfidf_dest):
            fail("--tfidf-src and --tfidf-dest are required for tfidf")
        publish_tfidf(Path(a.tfidf_src), Path(a.tfidf_dest),
                      a.k_first, a.k_second, do_backup, a.dry_run)
    info("\npublish complete (no git add/commit/push performed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
