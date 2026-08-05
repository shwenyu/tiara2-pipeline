"""Provenance manifests: what ran, on what inputs, with which params.

Each stage writes ``<work>/<stage>/manifest.json`` after a successful run.
The manifest is also the resume key: if the recomputed fingerprint matches the
stored one, the stage is skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def file_fingerprint(path: str | os.PathLike) -> dict[str, Any]:
    """Cheap fingerprint: size + mtime (avoids hashing multi-TiB FASTA)."""
    st = Path(path).stat()
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def fingerprint(inputs: list[str], params: dict) -> str:
    """Stable SHA over input file stats + resolved params."""
    h = hashlib.sha256()
    for path in sorted(inputs):
        try:
            fp = file_fingerprint(path)
        except FileNotFoundError:
            fp = {"path": str(path), "missing": True}
        h.update(json.dumps(fp, sort_keys=True).encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()


def tool_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "version"], capture_output=True,
                             text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "unknown"
    except Exception:
        return "unavailable"


def write(manifest_path: str | os.PathLike, *, stage: str, fp: str,
          params: dict, inputs: list[str], outputs: list[str],
          counts: dict | None = None, versions: dict | None = None) -> dict:
    payload = {
        "stage": stage,
        "fingerprint": fp,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": params,
        "inputs": inputs,
        "outputs": outputs,
        "counts": counts or {},
        "versions": versions or {},
    }
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    return payload


def read(manifest_path: str | os.PathLike) -> dict | None:
    p = Path(manifest_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
