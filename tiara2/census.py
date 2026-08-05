"""Per-class data census -- "how much do I actually have, right now?"

Why this exists when inventory.py already reports sizes
-------------------------------------------------------
``inventory.corpus_inventory`` answers "how many BYTES are on disk" with 15
stat() calls. That is the right tool before a download. It is the wrong tool
before a CURATION decision, because bytes are not the quantity the training
budget is denominated in. The budget is denominated in FRAGMENTS, and the map
from bytes to fragments runs through record count, sequence length and header
overhead -- none of which a stat() can see. A class can be large in bytes and
small in usable fragments (few long records that chop into little) or the
reverse.

This module therefore adds three things on top of inventory:

1. RECORDS and BASE PAIRS per split x class, so the current corpus can be
   compared directly against the target class priors.
2. A projection into fragments at the configured chop length, and the DELTA
   against the Tiara S3 priors -- which is the number that actually decides
   whether a class is over- or under-represented.
3. A DOWNLOAD-ACTIVITY GUARD.

About that guard
----------------
A download is running while this is being written. Any census taken mid-flight
is a moving target, and freezing a curation plan against a moving target is how
you end up with a selection that references assemblies that were still being
written. So ``download_activity`` looks for partial-file suffixes, recent
mtimes and live acquire logs, and ``curate`` refuses to write a plan when it
finds them, unless explicitly forced. Measuring is always safe; FREEZING is
what is gated.

Cost control: the default mode reads only the first ``sample_mb`` of each FASTA
and extrapolates by byte ratio. On this corpus a full scan is a multi-TB read.
The sample is taken from the head of the file, so its bias is stated openly in
the output (``mode=fast``) rather than hidden -- use ``--exact`` when the
number has to be exact.

Nothing here raises. A census of a half-downloaded tree must still print.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import inventory
from .progress import Progress

#: Suffixes that mean "this file is still being written". Covers wget/curl,
#: rsync and ncbi_pipeline.py's own temp naming.
PARTIAL_SUFFIXES = (".part", ".tmp", ".temp", ".filepart", ".download",
                    ".crdownload", ".partial", ".aria2")

#: A file touched within this many seconds counts as "being written now".
DEFAULT_ACTIVE_WINDOW = 900.0

#: Bytes of each FASTA read in fast mode.
DEFAULT_SAMPLE_MB = 64

#: Tiara Supplementary Table S3, verbatim: the 5 kb fragment composition of the
#: published first-stage training set. Used as the default target prior because
#: it is the only class balance with a measured F1 attached to it.
TIARA_S3_PRIORS = {
    "bacteria": 0.42438644,
    "eukarya": 0.31667691,
    "archaea": 0.17336867,
    "plastids": 0.05765327,
    "mitochondria": 0.02791470,
}


def sample_fasta(path, sample_bytes=None, progress=None):
    """Record count / bp / mean length, from the head of a FASTA.

    Returns a dict with ``mode`` set to ``exact`` when the whole file was read
    and ``fast`` when it was sampled and extrapolated. Extrapolation is by byte
    ratio: bp scales with file size far more reliably than record count does,
    because record LENGTH varies but the header-to-sequence byte ratio within
    one class does not vary much.

    Binary-safe and encoding-agnostic (the file is opened in binary), because a
    stray non-UTF8 byte in a header must not abort a census.
    """
    out = {"path": str(path), "exists": False, "records": 0, "bp": 0,
           "mean_len": 0.0, "mode": "fast", "sampled_bytes": 0,
           "total_bytes": 0}
    try:
        size = os.path.getsize(str(path))
    except OSError:
        return out
    out["exists"] = True
    out["total_bytes"] = size
    if size == 0:
        out["mode"] = "exact"
        return out

    limit = size if not sample_bytes else min(int(sample_bytes), size)
    records = 0
    bp = 0
    read = 0
    pending_bytes = 0
    pending_records = 0
    progress_batch_bytes = 8 * 1024 * 1024
    try:
        with open(str(path), "rb") as fh:
            for line in fh:
                line_bytes = len(line)
                read += line_bytes
                pending_bytes += line_bytes
                is_record = line.startswith(b">")
                if is_record:
                    records += 1
                    pending_records += 1
                else:
                    bp += len(line.strip())
                # Avoid one Python method call per FASTA line on multi-TiB
                # corpora. Eight-MiB accounting batches preserve a responsive
                # progress display without measurably slowing the census.
                if progress is not None and pending_bytes >= progress_batch_bytes:
                    progress.tick(records=pending_records, nbytes=pending_bytes)
                    pending_bytes = 0
                    pending_records = 0
                if read >= limit:
                    break
    except OSError:
        return out
    finally:
        if progress is not None and pending_bytes:
            progress.tick(records=pending_records, nbytes=pending_bytes)

    out["sampled_bytes"] = read
    if read >= size:
        out["mode"] = "exact"
        out["records"], out["bp"] = records, bp
    else:
        # Drop the final, probably truncated record before extrapolating.
        records = max(0, records - 1)
        scale = size / float(read) if read else 1.0
        out["records"] = int(round(records * scale))
        out["bp"] = int(round(bp * scale))
    if out["records"]:
        out["mean_len"] = out["bp"] / float(out["records"])
    return out


def class_census(cfg, *, exact=False, sample_mb=DEFAULT_SAMPLE_MB,
                 progress=None):
    """Records / bp / bytes for every split x class file in ``source_ready``."""
    root = Path(str(cfg.get("source_ready") or "."))
    splits = list(cfg.get("splits", []))
    classes = list(cfg.get("classes", []))
    sample_bytes = None if exact else int(sample_mb) * 1024 * 1024
    rows = []
    paths = [(split, cls, root / split / (cls + ".fasta"))
             for split in splits for cls in classes]
    if progress is not None:
        units = []
        for _split, _cls, path in paths:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            read_size = size if exact or sample_bytes is None else min(size, sample_bytes)
            units.append((str(path), read_size))
        progress.plan(units)
    for split, cls, path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        read_size = size if exact or sample_bytes is None else min(size, sample_bytes)
        if progress is not None:
            progress.start_unit(str(path), read_size)
        stats = sample_fasta(path, sample_bytes, progress=progress)
        stats.update({"split": split, "class": cls})
        rows.append(stats)
        if progress is not None:
            progress.finish_unit(f"{stats['bp']:,} bp")
    if progress is not None:
        progress.finish()
    return {"root": str(root), "rows": rows,
            "mode": "exact" if exact else "fast",
            "sample_mb": None if exact else int(sample_mb)}


def fragment_projection(census, cfg):
    """Project bp into fragments and compare against the target class priors.

    The comparison is the whole point. A raw bp table cannot tell you whether a
    class is a problem; the SHARE of the fragment budget versus the target
    share can. v2.0's failure was exactly a prior mismatch, so this table is
    the pre-flight check that would have caught it.
    """
    cur = (cfg.get("curate", {}) or {})
    budget = (cur.get("budget", {}) or {})
    mean_bp = float(budget.get("mean_fragment_bp", 4200)) or 4200.0
    priors = dict(cur.get("target_priors") or TIARA_S3_PRIORS)
    split = str(budget.get("census_split", "train"))

    per_class = {}
    for row in census["rows"]:
        if row["split"] != split:
            continue
        entry = per_class.setdefault(row["class"], {"bp": 0, "records": 0,
                                                    "bytes": 0})
        entry["bp"] += row["bp"]
        entry["records"] += row["records"]
        entry["bytes"] += row["total_bytes"]

    total_frags = sum(v["bp"] for v in per_class.values()) / mean_bp
    out = []
    for cls, entry in sorted(per_class.items()):
        frags = entry["bp"] / mean_bp
        share = (frags / total_frags) if total_frags else 0.0
        target = float(priors.get(cls, 0.0))
        out.append({
            "class": cls, "records": entry["records"], "bp": entry["bp"],
            "bytes": entry["bytes"], "fragments": int(frags),
            "share": share, "target_share": target,
            "delta_pp": (share - target) * 100.0,
            # Fragments this class must gain (+) or shed (-) to hit its target
            # at the CURRENT total. Directly actionable as a chop/subsample cap.
            "fragments_to_target": int(target * total_frags - frags),
        })
    return {"split": split, "mean_fragment_bp": mean_bp,
            "total_fragments": int(total_frags), "classes": out}


def download_activity(cfg, *, window_s=DEFAULT_ACTIVE_WINDOW, max_entries=40000):
    """Detect an in-progress download. Cheap, capped, never raises.

    Three independent signals, because any one alone gives false negatives:

    * partial-suffix files anywhere under the watched roots;
    * any file modified inside ``window_s`` (a download that just wrote a
      complete chunk leaves no .part behind);
    * an acquire step log whose mtime is inside the window (catches a download
      that is stalled on the network but has NOT finished).
    """
    roots = []
    for key in ("source_ready", "corpus_ready", "work_root"):
        val = cfg.get(key)
        if val:
            roots.append(str(val))
    extra = ((cfg.get("census", {}) or {}).get("watch_dirs") or [])
    roots.extend(str(x) for x in extra)

    now = time.time()
    partials, recent = [], []
    seen = 0
    for root in roots:
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        seen += 1
                        if seen > max_entries:
                            stack = []
                            break
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                                continue
                            name = entry.name.lower()
                            if name.endswith(PARTIAL_SUFFIXES):
                                partials.append(entry.path)
                            st = entry.stat(follow_symlinks=False)
                            if now - st.st_mtime <= window_s:
                                recent.append({"path": entry.path,
                                               "age_s": now - st.st_mtime})
                        except OSError:
                            continue
            except OSError:
                continue

    logs = []
    work_root = cfg.get("work_root")
    if work_root:
        log_dir = Path(str(work_root)) / "acquire" / "logs"
        try:
            for entry in sorted(log_dir.glob("acquire_*.log")):
                age = now - entry.stat().st_mtime
                if age <= window_s:
                    logs.append({"path": str(entry), "age_s": age})
        except OSError:
            pass

    recent.sort(key=lambda r: r["age_s"])
    return {
        "active": bool(partials or recent or logs),
        "window_s": window_s,
        "partial_files": partials[:20],
        "partial_count": len(partials),
        "recent_files": recent[:20],
        "recent_count": len(recent),
        "active_logs": logs,
        "scanned_entries": seen,
        "truncated": seen > max_entries,
    }


def render_census(census, projection=None, activity=None):
    """Fixed-width report, in the style of inventory.render_corpus."""
    lines = [f"corpus census ({census['mode']} mode"
             + (f", {census['sample_mb']} MiB sample/file"
                if census["mode"] == "fast" else "") + ")",
             f"  root: {census['root']}"]
    header = ("  " + "split".ljust(12) + "class".ljust(15)
              + "size".rjust(12) + "records".rjust(14) + "bp".rjust(18)
              + "mean len".rjust(11))
    lines += [header, "  " + "-" * (len(header) - 2)]
    for row in census["rows"]:
        if not row["exists"]:
            lines.append("  " + row["split"].ljust(12) + row["class"].ljust(15)
                         + "MISSING".rjust(12))
            continue
        lines.append(
            "  " + row["split"].ljust(12) + row["class"].ljust(15)
            + inventory.human(row["total_bytes"]).rjust(12)
            + f"{row['records']:,}".rjust(14)
            + f"{row['bp']:,}".rjust(18)
            + f"{row['mean_len']:,.0f}".rjust(11))

    if projection:
        lines += ["", f"fragment projection @ {projection['mean_fragment_bp']:.0f} "
                      f"bp/fragment  (split={projection['split']})"]
        head = ("  " + "class".ljust(15) + "fragments".rjust(14)
                + "share".rjust(9) + "target".rjust(9) + "delta".rjust(10)
                + "to target".rjust(15))
        lines += [head, "  " + "-" * (len(head) - 2)]
        for row in projection["classes"]:
            lines.append(
                "  " + row["class"].ljust(15)
                + f"{row['fragments']:,}".rjust(14)
                + f"{row['share'] * 100:.2f}%".rjust(9)
                + f"{row['target_share'] * 100:.2f}%".rjust(9)
                + f"{row['delta_pp']:+.2f}pp".rjust(10)
                + f"{row['fragments_to_target']:+,}".rjust(15))
        lines.append(f"  total fragments: {projection['total_fragments']:,}")

    if activity:
        lines.append("")
        if activity["active"]:
            lines.append("  !! DOWNLOAD IN PROGRESS -- census is a moving "
                         "target, do not freeze a selection yet")
            if activity["partial_count"]:
                lines.append(f"     partial files: {activity['partial_count']}")
                for path in activity["partial_files"][:5]:
                    lines.append(f"       {path}")
            if activity["recent_count"]:
                lines.append(f"     modified in the last "
                             f"{activity['window_s']:.0f}s: "
                             f"{activity['recent_count']} files")
                for row in activity["recent_files"][:5]:
                    lines.append(f"       {row['path']} "
                                 f"({row['age_s']:.0f}s ago)")
            for row in activity["active_logs"]:
                lines.append(f"     live acquire log: {row['path']} "
                             f"({row['age_s']:.0f}s ago)")
        else:
            lines.append(f"  no write activity in the last "
                         f"{activity['window_s']:.0f}s -- corpus looks settled")
    return "\n".join(lines)


def run_census(cfg, *, exact=False, sample_mb=DEFAULT_SAMPLE_MB,
               window_s=DEFAULT_ACTIVE_WINDOW, progress=None):
    """Everything at once: census + projection + activity guard."""
    census = class_census(cfg, exact=exact, sample_mb=sample_mb,
                          progress=progress)
    return {
        "census": census,
        "projection": fragment_projection(census, cfg),
        "activity": download_activity(cfg, window_s=window_s),
    }
