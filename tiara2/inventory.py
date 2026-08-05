"""Fast, SHALLOW disk inventory.

Why "shallow" is a hard requirement
-----------------------------------
The data tree lives on a 135 T spinning ext4 volume holding multi-TB FASTA and
a multi-TB feature cache. A naive ``du -sh`` over it takes MINUTES and hammers
the very disk the pipeline is reading from. Every helper here therefore answers
"what have I got?" with as few stat() calls as possible:

  * known single files (per split x class FASTA)  -> ~15 stat() calls, instant
  * derived directories (cache / pack / flat)     -> capped recursive walk that
    STOPS at ``max_entries`` and reports its byte count as a LOWER BOUND
  * unknown trees (data_home)                     -> ONE level of scandir, entry
    counts only, never a recursive size

Nothing in this module raises. An inventory of a half-populated tree must still
print: missing paths come back as ``exists=False`` instead of an exception, and
unreadable directories are skipped rather than aborting the walk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Past this many directory entries a walk stops and marks itself truncated.
# The binned-shard tree can hold hundreds of thousands of small FASTA files and
# nobody needs its size to four significant figures -- ">= 1.2 TiB" answers the
# only real question ("is this worth deleting?") just as well.
DEFAULT_MAX_ENTRIES = 200_000


def human(num: float) -> str:
    """Bytes -> short human string, binary units."""
    if abs(num) < 1024:
        return f"{int(num):,} B"
    value = float(num)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024.0
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:,.1f} {unit}"
    return f"{value:,.1f} TiB"


@dataclass
class DirStat:
    """Result of one (possibly capped) size probe."""
    path: str
    exists: bool = False
    bytes: int = 0
    files: int = 0
    truncated: bool = False
    mtime: float = 0.0

    @property
    def size_h(self) -> str:
        """Human size, prefixed with '>=' when the walk hit the entry cap."""
        return (">=" if self.truncated else "") + human(self.bytes)

    def as_dict(self) -> dict:
        return {
            "path": self.path, "exists": self.exists, "bytes": self.bytes,
            "files": self.files, "truncated": self.truncated,
            "mtime": self.mtime,
        }


def file_stat(path: str | os.PathLike) -> DirStat:
    """Single-file probe: exactly one stat() call, never walks."""
    out = DirStat(path=str(path))
    try:
        st = Path(path).stat()
    except OSError:
        return out
    out.exists = True
    out.bytes = st.st_size
    out.files = 1
    out.mtime = st.st_mtime
    return out


def dir_stat(path: str | os.PathLike, *,
             max_entries: int = DEFAULT_MAX_ENTRIES) -> DirStat:
    """Recursive size of ``path``, capped at ``max_entries`` directory entries.

    Iterative (explicit stack) rather than recursive so a deep tree cannot blow
    the Python stack, and ``entry.stat(follow_symlinks=False)`` reuses the
    dirent data scandir already fetched, which is what makes this many times
    faster than ``os.walk`` + ``os.path.getsize``.
    """
    p = Path(path)
    out = DirStat(path=str(p))
    try:
        st = p.stat()
    except OSError:
        return out
    out.exists = True
    out.mtime = st.st_mtime
    if p.is_file():
        out.bytes = st.st_size
        out.files = 1
        return out

    seen = 0
    stack = [str(p)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    seen += 1
                    if seen > max_entries:
                        out.truncated = True
                        return out
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            out.files += 1
                            out.bytes += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return out


def shallow_children(path: str | os.PathLike, *, limit: int = 60) -> list[dict]:
    """ONE level of scandir: name, kind, immediate child count, mtime.

    Deliberately no sizes -- this is used on ``data_home``, whose subtrees are
    terabytes each. Immediate child count is a good enough "how much is in
    here?" signal and costs one extra scandir per directory.
    """
    rows: list[dict] = []
    try:
        with os.scandir(str(path)) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError:
        return rows
    for entry in entries[:limit]:
        row = {"name": entry.name, "is_dir": False, "children": 0, "mtime": 0.0}
        try:
            row["is_dir"] = entry.is_dir(follow_symlinks=False)
            row["mtime"] = entry.stat(follow_symlinks=False).st_mtime
        except OSError:
            pass
        if row["is_dir"]:
            try:
                with os.scandir(entry.path) as sub:
                    row["children"] = sum(1 for _ in sub)
            except OSError:
                pass
        rows.append(row)
    return rows


def corpus_inventory(cfg: dict) -> dict:
    """Per split x class FASTA inventory of ``source_ready``.

    This is the cheap answer to "what is actually downloaded and prepared?":
    len(splits) x len(classes) stat() calls (15 with the default config), no
    directory walking at all.
    """
    root = str(cfg.get("source_ready") or "")
    splits = list(cfg.get("splits", []))
    classes = list(cfg.get("classes", []))
    rows: list[dict] = []
    for split in splits:
        for cls in classes:
            st = file_stat(Path(root) / split / f"{cls}.fasta")
            rows.append({
                "split": split, "class": cls,
                "exists": st.exists, "bytes": st.bytes, "mtime": st.mtime,
            })
    total = sum(r["bytes"] for r in rows)
    present = sum(1 for r in rows if r["exists"])
    return {
        "root": root, "rows": rows, "total_bytes": total,
        "present": present, "expected": len(rows),
    }


def render_corpus(inv: dict) -> str:
    """Fixed-width table of the corpus inventory (class rows x split columns)."""
    rows = inv["rows"]
    if not rows:
        return "  (no splits/classes configured)"
    splits: list[str] = []
    classes: list[str] = []
    for row in rows:
        if row["split"] not in splits:
            splits.append(row["split"])
        if row["class"] not in classes:
            classes.append(row["class"])
    by_key = {(r["split"], r["class"]): r for r in rows}

    width = max([len(c) for c in classes] + [8])
    head = "  " + "class".ljust(width) + "".join(s.rjust(14) for s in splits)
    lines = [head, "  " + "-" * (width + 14 * len(splits))]
    for cls in classes:
        cells = []
        for split in splits:
            row = by_key.get((split, cls))
            cells.append((human(row["bytes"]) if row and row["exists"]
                          else "-").rjust(14))
        lines.append("  " + cls.ljust(width) + "".join(cells))
    totals = []
    for split in splits:
        sub = sum(by_key[(split, c)]["bytes"] for c in classes
                  if (split, c) in by_key)
        totals.append(human(sub).rjust(14))
    lines.append("  " + "-" * (width + 14 * len(splits)))
    lines.append("  " + "TOTAL".ljust(width) + "".join(totals))
    missing = inv["expected"] - inv["present"]
    if missing:
        lines.append(f"  !! {missing} of {inv['expected']} class/split files "
                     f"are MISSING")
    return "\n".join(lines)


def disk_free(path: str | os.PathLike) -> dict:
    """statvfs on the filesystem holding ``path`` (never raises)."""
    try:
        usage = os.statvfs(str(path))
    except OSError:
        return {"ok": False}
    total = usage.f_blocks * usage.f_frsize
    free = usage.f_bavail * usage.f_frsize
    return {
        "ok": True, "total": total, "free": free, "used": total - free,
        "pct_used": (100.0 * (total - free) / total) if total else 0.0,
    }
