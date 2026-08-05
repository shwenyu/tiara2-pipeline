#!/usr/bin/env python3
"""Tests for the explicit progress output (scripts/progress.py).

These lock down the four conventions, because each one is a decision that a
later edit could plausibly undo:

1. progress on stderr, results on stdout
2. wall-clock throttling, not record-count modulo
3. percent from bytes
4. unknown ETA prints '?'

plus two safety properties: a broken pipe must not kill the job, and every
long-running script must actually be wired up.

Run directly:  python3 tests/test_progress.py
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import progress as P  # noqa: E402


class FakeClock:
    """Manually advanced monotonic clock, so throttling is tested exactly."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ExplodingStream(io.StringIO):
    """Stands in for a pipe closed by `| head`."""

    def write(self, _data):
        raise BrokenPipeError(32, "Broken pipe")


def lines(buf):
    return [ln for ln in buf.getvalue().splitlines() if ln.strip()]


class TestDestination(unittest.TestCase):
    """Convention 1: progress on stderr, results on stdout."""

    def test_default_stream_is_stderr(self):
        self.assertIs(P.Progress("x").stream, sys.stderr)
        self.assertIs(P.Counter("x").stream, sys.stderr)

    def test_label_prefix(self):
        buf = io.StringIO()
        P.Progress("relabel", stream=buf).write("hello")
        self.assertEqual(lines(buf), ["[relabel] hello"])


class TestThrottle(unittest.TestCase):
    """Convention 2: throttle on the wall clock, not on a record modulo."""

    def test_one_line_per_interval_regardless_of_record_count(self):
        clock = FakeClock()
        buf = io.StringIO()
        prog = P.Progress("t", seconds=30, stream=buf, clock=clock)
        # 100k tiny records inside one interval must not print 100k lines.
        for _ in range(100_000):
            prog.tick(nbytes=10)
        self.assertEqual(len(lines(buf)), 1, "first tick emits, rest throttled")
        clock.advance(30.0)
        prog.tick(nbytes=10)
        self.assertEqual(len(lines(buf)), 2)
        # Records still accumulate while throttled.
        self.assertEqual(prog.records, 100_001)

    def test_zero_seconds_means_unthrottled_not_off(self):
        buf = io.StringIO()
        prog = P.Progress("t", seconds=0, stream=buf, clock=FakeClock())
        for _ in range(5):
            prog.tick(nbytes=1)
        self.assertEqual(len(lines(buf)), 5)


class TestBytesAndEta(unittest.TestCase):
    """Conventions 3 and 4: percent from bytes; unknown ETA is '?'."""

    def test_percent_uses_bytes_not_records(self):
        clock = FakeClock()
        buf = io.StringIO()
        prog = P.Progress("t", seconds=0, stream=buf, clock=clock)
        prog.plan([("a", 400), ("b", 600)])          # 1000 bytes total
        clock.advance(1.0)
        prog.tick(records=1, nbytes=250)             # 1 record, 25% of bytes
        self.assertIn("25.0%", lines(buf)[-1])

    def test_plan_states_total_up_front(self):
        buf = io.StringIO()
        P.Progress("t", stream=buf).plan([("a", 2 * 1024 ** 4)])
        self.assertIn("2.00 TiB", lines(buf)[0])
        self.assertIn("1 file(s)", lines(buf)[0])

    def test_unknown_total_gives_question_marks(self):
        clock = FakeClock()
        buf = io.StringIO()
        prog = P.Progress("t", seconds=0, stream=buf, clock=clock)
        clock.advance(1.0)
        prog.tick(nbytes=100)                        # no plan() -> no total
        self.assertIn("?", lines(buf)[-1])
        self.assertIn("eta ?", lines(buf)[-1])

    def test_eta_is_never_invented(self):
        self.assertEqual(P.human_duration(None), "?")
        self.assertEqual(P.human_duration(-1), "?")
        self.assertEqual(P.human_duration(float("nan")), "?")

    def test_eta_from_cumulative_mean_rate(self):
        clock = FakeClock()
        buf = io.StringIO()
        prog = P.Progress("t", seconds=0, stream=buf, clock=clock)
        prog.plan([("a", 1000)])
        clock.advance(10.0)
        prog.tick(nbytes=500)        # 50 B/s mean -> 500 B left -> 10s
        self.assertIn("eta 10s", lines(buf)[-1])

    def test_human_bytes_is_binary(self):
        self.assertEqual(P.human_bytes(1024), "1.00 KiB")
        self.assertEqual(P.human_bytes(1024 ** 3), "1.00 GiB")
        self.assertEqual(P.human_bytes(512), "512 B")


class TestPerUnitReporting(unittest.TestCase):
    def test_start_and_finish_report_the_unit_result(self):
        buf = io.StringIO()
        prog = P.Progress("relabel", seconds=30, stream=buf, clock=FakeClock())
        prog.plan([("train/bacteria.fasta", 1024)])
        prog.start_unit("train/bacteria.fasta", 1024)
        prog.tick(nbytes=1024)
        prog.finish_unit("relocated 7")
        prog.finish()
        text = buf.getvalue()
        self.assertIn("[1/1] start train/bacteria.fasta", text)
        # The per-file verdict must appear as the file closes, not only at the
        # end of the run.
        self.assertIn("relocated 7", text)
        self.assertIn("done  train/bacteria.fasta", text)
        self.assertIn("finished:", text)


class TestNeverKillsTheJob(unittest.TestCase):
    def test_broken_pipe_is_swallowed(self):
        prog = P.Progress("t", seconds=0, stream=ExplodingStream())
        prog.tick(nbytes=1)          # must not raise
        prog.finish()
        self.assertFalse(prog.enabled, "progress disables itself after a write error")


class TestSwitches(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k)
                     for k in ("TIARA2_PROGRESS", "TIARA2_PROGRESS_SECONDS")}
        for key in self._env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    class Args:
        def __init__(self, quiet=False, progress_seconds=None):
            self.quiet = quiet
            self.progress_seconds = progress_seconds

    def test_quiet_disables(self):
        prog = P.progress_from_args(self.Args(quiet=True), "t")
        self.assertFalse(prog.enabled)

    def test_env_zero_disables(self):
        os.environ["TIARA2_PROGRESS"] = "0"
        self.assertFalse(P.progress_from_args(self.Args(), "t").enabled)
        self.assertFalse(P.counter_from_args(self.Args(), "t").enabled)

    def test_flag_beats_env(self):
        os.environ["TIARA2_PROGRESS_SECONDS"] = "60"
        self.assertEqual(P.progress_from_args(self.Args(), "t").seconds, 60.0)
        self.assertEqual(
            P.progress_from_args(self.Args(progress_seconds=5), "t").seconds, 5.0)

    def test_default_seconds(self):
        self.assertEqual(P.progress_from_args(self.Args(), "t").seconds,
                         P.DEFAULT_SECONDS)

    def test_disabled_progress_still_counts(self):
        prog = P.NullProgress("t")
        prog.tick(records=3, nbytes=30)
        self.assertEqual((prog.records, prog.nbytes), (3, 30))


class TestRowCounter(unittest.TestCase):
    def test_no_total_means_question_mark(self):
        clock = FakeClock()
        buf = io.StringIO()
        counter = P.Counter("import", seconds=0, stream=buf, clock=clock)
        clock.advance(1.0)
        counter.count()
        self.assertIn("eta ?", lines(buf)[-1])
        self.assertIn("row", lines(buf)[-1])

    def test_known_total_gives_percent(self):
        clock = FakeClock()
        buf = io.StringIO()
        counter = P.Counter("import", total=200, seconds=0, stream=buf, clock=clock)
        clock.advance(1.0)
        counter.count(50)
        self.assertIn("25.0%", lines(buf)[-1])

    def test_progress_counter_alias(self):
        self.assertIs(P.ProgressCounter, P.Counter)


WIRED_SCRIPTS = (
    "repair_corpus_labels.py",
    "audit_corpus_labels.py",
    "import_tiara_corpus.py",
    "bin_by_length.py",
    "regroup_by_metadata.py",
)


class TestScriptsAreWired(unittest.TestCase):
    """Every script that streams the corpus must expose progress."""

    def test_scripts_import_and_expose_progress(self):
        for name in WIRED_SCRIPTS:
            source = (ROOT / "scripts" / name).read_text()
            with self.subTest(script=name):
                self.assertIn("from progress import", source)
                self.assertIn("add_progress_args", source)

    def test_progress_seconds_flag_is_documented(self):
        self.assertIn("--progress-seconds", (ROOT / "README.md").read_text())


class TestEndToEndStreams(unittest.TestCase):
    """The whole point, on a real run: report on stdout, heartbeat on stderr."""

    HEADERS = (
        ("bacteria", "GCA_000000001.1", "Bacteria", "prok"),
        ("archaea", "GCA_000000002.1", "Archaea", "prok"),
        ("eukarya", "GCA_000000003.1", "Eukaryota", "euk"),
    )

    def _corpus(self, root):
        split = root / "train"
        split.mkdir(parents=True)
        for cls, acc, sg, label in self.HEADERS:
            (split / f"{cls}.fasta").write_text(
                f">{acc}|0 sg={sg} label={label} epoch=train\nACGT\n")

    def test_repair_splits_streams(self):
        import repair_corpus_labels as R
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "in"
            self._corpus(root)
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                R.main(["--root", str(root), "--out", str(Path(tmp) / "out"),
                        "--splits", "train", "--allow-class-drop",
                        "--progress-seconds", "0"])
            self.assertIn("[relabel]", err.getvalue())
            self.assertNotIn("[relabel]", out.getvalue(),
                             "progress must not pollute stdout")
            self.assertIn("relabel:", out.getvalue(), "report belongs on stdout")

    def test_repair_quiet_is_silent_on_stderr(self):
        import repair_corpus_labels as R
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "in"
            self._corpus(root)
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                R.main(["--root", str(root), "--out", str(Path(tmp) / "out"),
                        "--splits", "train", "--allow-class-drop", "--quiet"])
            self.assertEqual(err.getvalue().strip(), "")
            self.assertIn("relabel:", out.getvalue())

    def test_audit_splits_streams(self):
        import audit_corpus_labels as A
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "in"
            self._corpus(root)
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                A.main(["--root", str(root), "--splits", "train",
                        "--progress-seconds", "0"])
            self.assertIn("[audit]", err.getvalue())
            self.assertNotIn("[audit]", out.getvalue())
            # Per-file mismatch rate is reported as each file finishes.
            self.assertIn("mismatched", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
