#!/usr/bin/env python3
"""Explicit progress output for the long-running corpus scripts.

Why this module exists
----------------------
Every script here streams multi-TiB FASTA in a single pass and prints ONE report
at the end. On this corpus that means hours of total silence, during which a
run that is merely slow is indistinguishable from a run that is wedged in D
state. This module makes the running state explicit, and it does so under four
conventions that are the actual design content:

1. **Progress goes to stderr, results go to stdout.**
   ``script ... > report.txt`` and ``script ... | jq`` must stay clean, while
   the heartbeat still shows up on the terminal. Asserted in
   ``tests/test_progress.py``.

2. **Throttle on the WALL CLOCK, never on a record-count modulo.**
   ``if n % 100000 == 0`` prints thousands of lines a second on 1 kb organelle
   fragments and stays silent for an hour on 10 kb eukaryotic ones. Neither is
   useful. One line every ``seconds`` is useful in both cases.

3. **Percent complete is computed from BYTES, not records.**
   The record count of a FASTA is unknown until it has been read, so a
   record-based percentage cannot exist on the first pass. ``st_size`` is known
   before a single byte is read, so the total work is stated UP FRONT -- that is
   the difference between "ran for 4 hours" and "4 hours into an estimated 6".

4. **When the ETA cannot be computed, print ``?``. Never invent one.**
   And use the cumulative mean rate rather than a sliding window: with one
   class holding ~48% of the corpus, a windowed rate swings wildly at every
   file boundary while the mean converges.

Two further properties that are easy to get wrong:

* ``--progress-seconds 0`` means UNTHROTTLED (one line per record), not "off".
  Making 0 mean "off" would duplicate ``--quiet`` and leave no way to ask for
  unthrottled tracing when investigating a hang.
* **Writing progress can never kill the job.** ``| head`` closes the pipe; a
  six-hour relabel must not die of a BrokenPipeError raised by a heartbeat.

Usage::

    from progress import Progress, add_progress_args, progress_from_args

    prog = progress_from_args(args, label="relabel")
    prog.plan([(str(p), p.stat().st_size) for p in files])
    for path in files:
        prog.start_unit(str(path), path.stat().st_size)
        for header, chunk in iter_records(path):
            prog.tick(nbytes=...)
        prog.finish_unit("relocated 0")
    prog.finish()

Environment overrides (for nohup runs where editing the command line is
awkward)::

    TIARA2_PROGRESS=0            # off, same as --quiet
    TIARA2_PROGRESS_SECONDS=60   # default throttle when no flag is given
"""

from __future__ import annotations

import os
import sys
import time

DEFAULT_SECONDS = 30.0
_UNITS = ((1024.0 ** 4, "TiB"), (1024.0 ** 3, "GiB"),
          (1024.0 ** 2, "MiB"), (1024.0, "KiB"))


def human_bytes(n: float) -> str:
    """1234567890 -> '1.15 GiB'. Binary units, because df/ls report binary."""
    n = float(n or 0)
    for scale, name in _UNITS:
        if n >= scale:
            return f"{n / scale:.2f} {name}"
    return f"{int(n)} B"


def human_duration(seconds: float | None) -> str:
    """Coarse, human-readable duration. ``None`` -> ``'?'`` (never invented)."""
    if seconds is None or seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def env_enabled() -> bool:
    """``TIARA2_PROGRESS`` in {0,false,no,off} disables progress entirely."""
    raw = os.environ.get("TIARA2_PROGRESS")
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def env_seconds(default: float = DEFAULT_SECONDS) -> float:
    raw = os.environ.get("TIARA2_PROGRESS_SECONDS")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


class Progress:
    """Wall-clock throttled, byte-based progress reporter writing to stderr.

    Parameters
    ----------
    label:
        Prefix for every line, e.g. ``relabel`` -> ``[relabel] ...``.
    seconds:
        Minimum wall-clock gap between progress lines. ``0`` = every tick.
    enabled:
        ``False`` silences everything except nothing at all -- no lines, and
        ``tick`` becomes a couple of integer adds.
    stream:
        Defaults to ``sys.stderr``. Only redirected by the tests.
    """

    def __init__(self, label: str, *, seconds: float = DEFAULT_SECONDS,
                 enabled: bool = True, stream=None, clock=time.monotonic):
        self.label = label
        self.seconds = max(0.0, float(seconds))
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self.started = clock()
        self._last_emit = 0.0
        self._emitted = False
        # totals across the whole run
        self.records = 0
        self.nbytes = 0
        self.total_bytes = 0          # 0 = unknown -> percent and ETA are '?'
        self.units_total = 0
        self.units_done = 0
        # current unit
        self.unit = ""
        self.unit_started = self.started
        self.unit_records = 0
        self.unit_bytes = 0
        self.unit_total_bytes = 0

    # ---- output ----
    def write(self, message: str) -> None:
        """Emit one line. Never raises: a closed pipe must not kill the job."""
        if not self.enabled:
            return
        try:
            self.stream.write(f"[{self.label}] {message}\n")
            self.stream.flush()
        except (OSError, ValueError):
            # BrokenPipeError (`| head`), or a stream closed at interpreter
            # shutdown. Progress is diagnostics; it is never worth an abort.
            self.enabled = False

    # ---- plan ----
    def plan(self, units) -> None:
        """State the whole workload before reading a byte.

        ``units`` is an iterable of ``(name, size_bytes)``. This is what makes
        percent and ETA possible on a first pass.
        """
        units = [(str(name), int(size or 0)) for name, size in units]
        self.units_total = len(units)
        self.total_bytes = sum(size for _n, size in units)
        self.write(f"plan: {self.units_total} file(s), "
                   f"{human_bytes(self.total_bytes)} to read")
        for name, size in sorted(units, key=lambda kv: -kv[1])[:8]:
            self.write(f"    {name:<48} {human_bytes(size):>12}")

    # ---- per unit ----
    def start_unit(self, name: str, total_bytes: int = 0) -> None:
        self.unit = str(name)
        self.unit_started = self._clock()
        self.unit_records = 0
        self.unit_bytes = 0
        self.unit_total_bytes = int(total_bytes or 0)
        index = self.units_done + 1
        counter = f"[{index}/{self.units_total}] " if self.units_total else ""
        size = (f"  ({human_bytes(self.unit_total_bytes)})"
                if self.unit_total_bytes else "")
        self.write(f"{counter}start {self.unit}{size}")

    def finish_unit(self, extra: str = "") -> None:
        """Close the current unit and report ITS result immediately.

        Per-unit results are printed here rather than only in the final table:
        a routing rule that is wrong shows up within minutes instead of after
        the whole corpus has been rewritten.
        """
        self.units_done += 1
        elapsed = self._clock() - self.unit_started
        parts = [f"done  {self.unit}", human_bytes(self.unit_bytes),
                 f"in {human_duration(elapsed)}",
                 f"{self.unit_records:,} rec"]
        if extra:
            parts.append(str(extra))
        self.write("  ".join(parts))

    # ---- ticking ----
    def tick(self, records: int = 1, nbytes: int = 0) -> None:
        """Account for work done and emit a line if the throttle allows it."""
        self.records += records
        self.unit_records += records
        self.nbytes += nbytes
        self.unit_bytes += nbytes
        if not self.enabled:
            return
        now = self._clock()
        if self.seconds and (now - self._last_emit) < self.seconds:
            return
        self._last_emit = now
        self.write(self._status(now))

    def _status(self, now: float) -> str:
        elapsed = max(now - self.started, 1e-9)
        rate = self.nbytes / elapsed                      # bytes/second, mean
        pct = (f"{100.0 * self.nbytes / self.total_bytes:.1f}%"
               if self.total_bytes else "?")
        eta = None
        if self.total_bytes and rate > 0:
            remaining = self.total_bytes - self.nbytes
            eta = remaining / rate if remaining > 0 else 0.0
        return (f"{self.records:,} rec  {human_bytes(self.nbytes)}  {pct}  "
                f"eta {human_duration(eta)}  {human_bytes(rate)}/s  "
                f"elapsed {human_duration(elapsed)}")

    # ---- end ----
    def finish(self, extra: str = "") -> None:
        elapsed = self._clock() - self.started
        message = (f"finished: {self.records:,} records, "
                   f"{human_bytes(self.nbytes)} in {human_duration(elapsed)}")
        if extra:
            message += f"  {extra}"
        self.write(message)


class Counter(Progress):
    """Record-count reporter for row-oriented inputs (XLSX / TSV / CSV).

    Convention 3 says percentages come from bytes -- but a spreadsheet read
    through ``openpyxl`` or ``csv.DictReader`` exposes no byte offset, so there
    is nothing to divide by. Rather than fabricate a percentage from a guessed
    row total, this reporter prints the row count and the elapsed time and
    leaves percent and ETA as ``?`` unless the caller genuinely knows ``total``
    (e.g. ``ws.max_row``).

    Same wall-clock throttle and same stderr destination as :class:`Progress`.
    """

    def __init__(self, label: str, *, total: int | None = None,
                 noun: str = "row", **kwargs):
        super().__init__(label, **kwargs)
        self.total = int(total) if total else 0
        self.noun = noun

    def count(self, n: int = 1) -> None:
        self.records += n
        if not self.enabled:
            return
        now = self._clock()
        if self.seconds and (now - self._last_emit) < self.seconds:
            return
        self._last_emit = now
        elapsed = max(now - self.started, 1e-9)
        rate = self.records / elapsed
        if self.total:
            pct = f"{100.0 * self.records / self.total:.1f}%"
            eta = (self.total - self.records) / rate if rate > 0 else None
            eta = max(eta, 0.0) if eta is not None else None
        else:
            pct = "?"
            eta = None
        self.write(f"{self.records:,} {self.noun}  {pct}  "
                   f"eta {human_duration(eta)}  {rate:,.0f} {self.noun}/s  "
                   f"elapsed {human_duration(elapsed)}")

    # Same wording as Progress.finish, but counted in rows rather than bytes.
    def finish(self, extra: str = "") -> None:
        elapsed = self._clock() - self.started
        message = (f"finished: {self.records:,} {self.noun}(s) in "
                   f"{human_duration(elapsed)}")
        if extra:
            message += f"  {extra}"
        self.write(message)


# Historical name kept because import_tiara_corpus.py refers to it.
ProgressCounter = Counter


class NullProgress(Progress):
    """Disabled reporter. Same interface, so callers never branch on None."""

    def __init__(self, label: str = "", **kwargs):
        kwargs["enabled"] = False
        super().__init__(label, **kwargs)


def add_progress_args(parser) -> None:
    """Add the shared ``--progress-seconds`` / ``--quiet`` pair."""
    parser.add_argument(
        "--progress-seconds", type=float, default=None,
        help=("seconds between progress lines on stderr "
              f"(default {int(DEFAULT_SECONDS)}; 0 = every record, for "
              "investigating a hang). Use --quiet to switch progress off."))
    parser.add_argument("--quiet", action="store_true",
                        help="no progress output (results still go to stdout)")


def progress_from_args(args, label: str, *, stream=None) -> Progress:
    """Build a reporter from parsed args + environment.

    Precedence: ``--quiet`` / ``TIARA2_PROGRESS=0`` off; then
    ``--progress-seconds``; then ``TIARA2_PROGRESS_SECONDS``; then the default.
    """
    if getattr(args, "quiet", False) or not env_enabled():
        return NullProgress(label, stream=stream)
    seconds = getattr(args, "progress_seconds", None)
    if seconds is None:
        seconds = env_seconds()
    return Progress(label, seconds=seconds, enabled=True, stream=stream)


def counter_from_args(args, label: str, *, total: int | None = None,
                     noun: str = "row", stream=None) -> Counter:
    """Same precedence as :func:`progress_from_args`, but row-counting."""
    if getattr(args, "quiet", False) or not env_enabled():
        return Counter(label, total=total, noun=noun, enabled=False,
                       stream=stream)
    seconds = getattr(args, "progress_seconds", None)
    if seconds is None:
        seconds = env_seconds()
    return Counter(label, total=total, noun=noun, seconds=seconds,
                   enabled=True, stream=stream)


def file_size(path) -> int:
    """``st_size`` or 0. Never raises: a vanished file is not a crash here."""
    try:
        return int(os.stat(str(path)).st_size)
    except OSError:
        return 0


def record_bytes(header: str, chunk) -> int:
    """Bytes a FASTA record occupied, including the newlines that were split off."""
    return len(header) + 1 + sum(len(line) + 1 for line in chunk)
