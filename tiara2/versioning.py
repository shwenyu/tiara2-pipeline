"""Version tags, the run registry, and resolved-config snapshots.

Two tags, on purpose
--------------------
``corpus_tag``   identifies the CORPUS layer:   acquire -> chop_bin -> dedup -> regroup
``version_tag``  identifies the TRAINING layer: tfidf/features -> HP -> models -> publish

They are separate because they change at completely different rates and cost.
Every pipeline path is templated off one tag or the other, and a stage is
skipped when its manifest fingerprint (inputs + params) is unchanged. If a
single tag drove every path, then bumping it to record a new training run would
relocate ``work_root`` too, and the tens of hours of mmseqs dedup behind it
would silently re-run from scratch for no reason at all.

So: bump ``version_tag`` for a new training run (new subsampling, new k, new
hyper-parameters). Bump ``corpus_tag`` only when the corpus itself changes --
chop parameters, dedup thresholds, or a re-partition of train/validation/test.

Interactive prompting is TTY-GATED
----------------------------------
Long runs are started under nohup, where stdin is not a terminal. A blocking
``input()`` there does not ask anybody anything -- it raises EOFError or hangs
the entire pipeline silently. Every prompt in this module therefore returns its
default immediately unless stdin AND stdout are both a TTY.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# A tag becomes a directory name, so keep it filesystem-safe and boring.
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class VersionError(ValueError):
    pass


def validate_tag(tag: str) -> str:
    tag = (tag or "").strip()
    if not TAG_RE.match(tag):
        raise VersionError(
            f"invalid tag {tag!r}: use letters, digits, dot, dash or underscore "
            f"(1-64 chars), because the tag becomes a directory name")
    return tag


def corpus_tag(cfg: dict) -> str:
    return str(cfg.get("corpus_tag") or cfg.get("input_tag") or "untagged")


def version_tag(cfg: dict) -> str:
    return str(cfg.get("version_tag") or cfg.get("output_tag") or "untagged")


def registry_path(cfg: dict) -> Path:
    configured = (cfg.get("versioning", {}) or {}).get("registry")
    if configured:
        return Path(str(configured))
    return Path(str(cfg.get("base", "."))) / "versions.json"


def config_sha(cfg: dict) -> str:
    """Stable digest of the fully resolved config -- the run's identity."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def read_registry(cfg: dict) -> dict:
    """Load the run registry; a corrupt or missing file is not fatal."""
    path = registry_path(cfg)
    if not path.exists():
        return {"runs": []}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"runs": []}
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        return {"runs": []}
    return data


def known_versions(cfg: dict) -> list[str]:
    """Distinct version tags seen before, most recent last."""
    out: list[str] = []
    for run in read_registry(cfg).get("runs", []):
        tag = run.get("version_tag")
        if tag and tag not in out:
            out.append(tag)
    return out


def last_run(cfg: dict) -> dict | None:
    runs = read_registry(cfg).get("runs", [])
    return runs[-1] if runs else None


def register_run(cfg: dict, *, stages: list[str], note: str = "") -> dict:
    """Append one entry to the registry. Best-effort: never fails a run.

    Written atomically (tmp + os.replace) because a pipeline launched twice by
    accident must not be able to truncate the history of every previous run.
    """
    entry = {
        "version_tag": version_tag(cfg),
        "corpus_tag": corpus_tag(cfg),
        "model_tag": str(cfg.get("model_tag", "")),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stages": list(stages),
        "config_sha": config_sha(cfg),
        "note": note or "",
    }
    try:
        path = registry_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = read_registry(cfg)
        data.setdefault("runs", []).append(entry)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
    except OSError:
        pass  # telemetry must never break a multi-day run
    return entry


def snapshot_config(cfg: dict, *, dest: str | os.PathLike | None = None) -> Path | None:
    """Freeze the RESOLVED config next to the run's results.

    The YAML on disk keeps changing between runs; this snapshot is what makes a
    finished version reproducible and diffable months later. Best-effort.
    """
    target = dest or (cfg.get("versioning", {}) or {}).get("snapshot")
    if not target:
        results = cfg.get("results_root")
        if not results:
            return None
        target = Path(str(results)) / "config_snapshot.json"
    path = Path(str(target))
    payload = {
        "version_tag": version_tag(cfg),
        "corpus_tag": corpus_tag(cfg),
        "model_tag": str(cfg.get("model_tag", "")),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_sha": config_sha(cfg),
        "config": cfg,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def interactive() -> bool:
    """True only when a human can actually answer (never under nohup)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def confirm(question: str, *, default: bool = False,
            assume_yes: bool = False) -> bool:
    """Yes/no prompt that degrades to ``default`` when non-interactive."""
    if assume_yes:
        return True
    if not interactive():
        return default
    suffix = " [y/N] " if not default else " [Y/n] "
    try:
        answer = input(question + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def suggest_next(tag: str) -> str:
    """Bump the last integer group in a tag: v2_0_hybrid -> v2_1_hybrid."""
    matches = list(re.finditer(r"\d+", tag))
    if not matches:
        return tag + "_2"
    last = matches[-1]
    return tag[:last.start()] + str(int(last.group()) + 1) + tag[last.end():]


def prompt_for_version(cfg: dict, *, assume_yes: bool = False) -> str | None:
    """Ask for the version tag of a fresh full run.

    Returns the NEW tag, or ``None`` to keep what the config already says.
    Shows the current tag and the previous run so the choice is informed --
    which was the whole point of asking.
    """
    current = version_tag(cfg)
    previous = last_run(cfg)
    print("")
    print("  current version_tag : %s" % current)
    print("  corpus_tag          : %s" % corpus_tag(cfg))
    if previous:
        print("  last recorded run   : %s  (%s, corpus %s)"
              % (previous.get("version_tag", "?"),
                 previous.get("started_at", "?"),
                 previous.get("corpus_tag", "?")))
    else:
        print("  last recorded run   : (none yet)")
    if not interactive() or assume_yes:
        return None
    suggestion = suggest_next(current)
    try:
        answer = input(
            f"  new version_tag (blank = keep {current}, "
            f"suggestion {suggestion}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not answer:
        return None
    return validate_tag(answer)


def render_registry(cfg: dict, *, limit: int = 20) -> str:
    """Compact table of recorded runs, newest last."""
    runs = read_registry(cfg).get("runs", [])
    if not runs:
        return ("  (no runs recorded yet -- versions.json is created on the "
                "first non-dry run)")
    lines = ["  %-26s %-22s %-10s %s" % ("version_tag", "started", "corpus",
                                         "stages")]
    lines.append("  " + "-" * 88)
    for run in runs[-limit:]:
        lines.append("  %-26s %-22s %-10s %s" % (
            str(run.get("version_tag", "?"))[:26],
            str(run.get("started_at", "?"))[:22],
            str(run.get("corpus_tag", "?"))[:10],
            ",".join(run.get("stages", []))[:34],
        ))
    return "\n".join(lines)
