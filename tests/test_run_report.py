"""Tests for scripts/build_run_report.py (the publish-stage run record).

Run with either:
    python3 -m unittest tests.test_run_report
    python3 tests/test_run_report.py

The harvester is a standalone, stdlib-only script (so it can run in any env),
hence it is loaded by path rather than imported as a package module.

The SOFT contract is the important part of this suite: the record documents a
run that already succeeded, so missing artifacts -- above all missing TRAINING
TIME, which is only recorded from this version onward -- must degrade to "-"
rather than raise.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "build_run_report.py"
    spec = importlib.util.spec_from_file_location("build_run_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load_module()

CLASSES = ["bacteria", "archaea"]
SPLITS = ["train", "validation"]


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data))
    else:
        path.write_text(str(data))


def _write_bytes(path: Path, nbytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * nbytes)


class RunReportBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="runreport_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base = self.tmp / "Tiara2"

    def cfg(self) -> dict:
        base = self.base
        return {
            "base": str(base),
            "data_home": str(self.tmp),
            "input_tag": "vTest",
            "model_tag": "v9.9",
            "output_tag": "vTest_out",
            "splits": list(SPLITS),
            "classes": list(CLASSES),
            "work_root": str(base / ".work"),
            "results_root": str(base / "results"),
            "dedup": {"enabled": True, "within_split": False,
                      "min_seq_id": 0.95, "min_cov": 0.95, "cov_mode": 0,
                      "varlen_bin": "VAR",
                      "varlen": {"min_seq_id": 0.95, "cov_mode": 5}},
            "chop": {"preset": "T6", "seed": 42},
            "mag": {"include_mags": False},
            "train": {
                "backend": "tiara",
                "train_ready": str(base / "train_ready"),
                "tfidf_dir": str(base / "tfidf"),
                "log_dir": str(base / "log"),
                "feature_cache": str(base / "feat"),
                "seq_pack": str(base / "pack"),
                "k_first": [4],
                "k_second": [4, 5],
                "tfidf_workers": 8,
                "tfidf_batch": 1000,
                "tfidf_fragment_len": 5000,
                "subsample": {
                    "enabled": True, "seed": 42,
                    "train": {"rate": 0.05, "min_acgt_purity": 0.9,
                              "min_len": 1000},
                    "validation": {"rate": 1.0, "min_acgt_purity": 0.0,
                                   "min_len": 0},
                    "test": {"rate": 1.0, "min_acgt_purity": 0.0,
                             "min_len": 0},
                },
                "gpu": {
                    "hp_epochs": 8, "hp_batch": 512,
                    "hp_max_train_rows": 4000000,
                    "hp_max_val_rows": 1000000,
                    "hp_maxpar": 16, "hp_gpus": "0,1",
                    "model_epochs": 50, "model_batch": 8192,
                },
            },
        }

    # ---- artifact builders ---------------------------------------------- #
    def make_corpus(self):
        ready = self.base / "train_ready"
        for split in SPLITS:
            for i, cls in enumerate(CLASSES):
                _write_bytes(ready / split / f"{cls}.fasta", 100 + i)

    def make_pack(self):
        for stage in ("first", "second"):
            for split in SPLITS:
                pdir = self.base / "pack" / stage / split
                spec = ({"rate": 0.05, "min_purity": 0.9, "min_len": 1000,
                         "seed": 42} if split == "train" else
                        {"rate": 1.0, "min_purity": 0.0, "min_len": 0,
                         "seed": 42})
                _write(pdir / "meta.json",
                       {"n": 1000 if split == "train" else 2500,
                        "inputs_sig": "sig", "subsample": spec,
                        "stage": stage, "split": split})
                _write_bytes(pdir / "codes.bin", 64)

    def make_features(self):
        # first/k4 both tags; second/k4 train only (a partially built cache)
        kdir = self.base / "feat" / "first" / "k4"
        _write(kdir / "meta.json", {
            "train": {"n": 1000, "dim": 256, "inputs_sig": "s"},
            "val": {"n": 2500, "dim": 256, "inputs_sig": "s"},
        })
        _write_bytes(kdir / "train_X.f32", 400)
        _write_bytes(kdir / "train_y.i64", 100)
        _write_bytes(kdir / "val_X.f32", 500)
        _write_bytes(kdir / "val_y.i64", 100)

        kdir2 = self.base / "feat" / "second" / "k4"
        _write(kdir2 / "meta.json",
               {"train": {"n": 700, "dim": 256, "inputs_sig": "s"}})
        _write_bytes(kdir2 / "train_X.f32", 200)
        _write_bytes(kdir2 / "train_y.i64", 100)

    def make_tfidf(self):
        _write(self.base / "tfidf" / "tfidf_manifest.json", {
            "status": "complete", "signature": "abc123", "seconds": 14365,
            "workers": 16, "config": {"fragment_len": 5000},
        })

    def make_hp(self):
        log = self.base / "log"
        _write(log / "hp_first_k4.json", {
            "status": "complete", "signature": "sig1", "stage": "first",
            "k": 4, "labels": ["a", "b"],
            "budget": {"epochs": 8, "hp_max_train_rows": 4000000,
                       "hp_max_val_rows": 1000000, "architectures": 2},
            "results": [
                {"mean_f1": 0.80, "accuracy": 0.9, "hid1": 128, "hid2": 64,
                 "learning_rate": 0.001, "dropout": 0.2, "seconds": 100.0,
                 "f1": [0.8, 0.8]},
                {"mean_f1": 0.91, "accuracy": 0.95, "hid1": 256, "hid2": 128,
                 "learning_rate": 0.01, "dropout": 0.1, "seconds": 300.0,
                 "f1": [0.9, 0.92]},
            ],
        })

    def make_timings(self):
        log = self.base / "log"
        log.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"module": "m.tfidf", "log": "10_tfidf.log",
                        "status": "ok", "started_at": "t0",
                        "ended_at": "t1", "seconds": 10.0}),
            "   ",                      # blank -> skipped
            "{not json at all",         # garbage -> skipped, must not raise
            json.dumps({"module": "m.hp", "log": "11_hp.log",
                        "status": "ok", "started_at": "t2",
                        "ended_at": "t3", "seconds": 20.0}),
            json.dumps({"module": "m.feat", "log": "09_feat.log",
                        "status": "failed", "started_at": "t4",
                        "ended_at": "t5", "seconds": 5.0}),
        ]
        (log / "step_timings.jsonl").write_text("\n".join(lines) + "\n")

    def make_stage_manifests(self):
        for name in ("train", "publish"):
            _write(self.base / ".work" / name / "manifest.json",
                   {"stage": name, "fingerprint": "f" * 64,
                    "finished_at": "2026-07-27T20:00:00+0800", "counts": {}})

    def make_all(self):
        self.make_corpus()
        self.make_pack()
        self.make_features()
        self.make_tfidf()
        self.make_hp()
        self.make_timings()
        self.make_stage_manifests()

    def report(self, cfg=None):
        return rr.build_report(cfg or self.cfg(), repo_root=str(ROOT))


class TestHarvest(RunReportBase):
    def test_selection_strategy_is_recorded(self):
        self.make_all()
        sel = self.report()["selection"]
        self.assertEqual(sel["subsample"]["train"]["rate"], 0.05)
        self.assertEqual(sel["subsample"]["train"]["min_acgt_purity"], 0.9)
        self.assertEqual(sel["dedup"]["min_seq_id"], 0.95)
        self.assertEqual(sel["k_second"], [4, 5])
        self.assertEqual(sel["hp_budget"]["max_train_rows"], 4000000)
        self.assertEqual(sel["hp_budget"]["epochs"], 8)

    def test_volumes(self):
        self.make_all()
        vol = self.report()["volumes"]
        # source corpus
        self.assertEqual(vol["train_ready"]["train"]["total_bytes"], 100 + 101)
        # pack records + the spec that produced them
        node = vol["seqpack"]["first/train"]
        self.assertEqual(node["records"], 1000)
        self.assertEqual(node["subsample"]["rate"], 0.05)
        self.assertEqual(vol["seqpack"]["second/validation"]["records"], 2500)
        # features: rows x dim and real bytes (X + y)
        first_k4 = vol["feature_cache"]["first"]["k4"]
        self.assertEqual(first_k4["train"]["rows"], 1000)
        self.assertEqual(first_k4["train"]["dim"], 256)
        self.assertEqual(first_k4["train"]["bytes"], 500)
        self.assertEqual(first_k4["val"]["bytes"], 600)
        # a partially built cache records only the tag that exists
        self.assertNotIn("val", vol["feature_cache"]["second"]["k4"])
        # k=5 was configured but never built -> absent, not an error
        self.assertNotIn("k5", vol["feature_cache"]["second"])
        self.assertEqual(vol["feature_cache_total_bytes"], 500 + 600 + 300)

    def test_timing_and_best_architecture(self):
        self.make_all()
        tim = self.report()["timing"]
        self.assertEqual(tim["tfidf"]["seconds"], 14365)
        hp = tim["hp_search"]["first_k4"]
        self.assertEqual(hp["candidates"], 2)
        self.assertEqual(hp["gpu_seconds"], 400.0)
        self.assertEqual(hp["slowest_candidate_seconds"], 300.0)
        # best = highest mean_f1, not the first entry
        self.assertAlmostEqual(hp["best"]["mean_f1"], 0.91)
        self.assertEqual(hp["best"]["hid1"], 256)
        # garbage lines skipped; failed step kept but excluded from wall total
        self.assertEqual(len(tim["steps"]), 3)
        self.assertEqual(tim["totals"]["recorded_wall_seconds"], 30.0)
        self.assertEqual(tim["totals"]["hp_gpu_seconds"], 400.0)
        self.assertIn("train", tim["stage_manifests"])


class TestSoftDegradation(RunReportBase):
    """The requirement: never raise when something (esp. timing) is absent."""

    def test_missing_training_time_does_not_raise(self):
        # everything EXCEPT step_timings.jsonl
        self.make_corpus()
        self.make_pack()
        self.make_features()
        self.make_tfidf()
        self.make_hp()
        report = self.report()
        self.assertIsNone(report["timing"]["totals"]["recorded_wall_seconds"])
        self.assertEqual(report["timing"]["steps"], [])
        # still renders, and says so explicitly instead of showing a wrong 0
        md = rr.render_markdown(report)
        self.assertIn("step_timings.jsonl", md)
        # other timings still present
        self.assertIn("14365", json.dumps(report["timing"]["tfidf"]))

    def test_nothing_on_disk_at_all(self):
        report = self.report()          # no artifacts created
        self.assertEqual(report["volumes"]["seqpack"], {})
        self.assertEqual(report["volumes"]["feature_cache"], {})
        self.assertIsNone(report["volumes"]["feature_cache_total_bytes"])
        self.assertIsNone(report["timing"]["tfidf"])
        self.assertEqual(report["timing"]["hp_search"], {})
        md = rr.render_markdown(report)
        self.assertIn("Run record", md)
        row = rr.build_csv_row(report, "x.md")
        self.assertIsNone(row["best_f1_first"])

    def test_minimal_config(self):
        report = rr.build_report({}, repo_root=str(ROOT))
        md = rr.render_markdown(report)
        self.assertIn("Run record", md)
        rr.build_csv_row(report, "x.md")   # must not raise

    def test_corrupt_artifacts_are_ignored(self):
        self.make_all()
        # truncate / corrupt every json the harvester reads
        (self.base / "tfidf" / "tfidf_manifest.json").write_text("{oops")
        (self.base / "log" / "hp_first_k4.json").write_text("not json")
        (self.base / "feat" / "first" / "k4" / "meta.json").write_text("[")
        report = self.report()
        self.assertIsNone(report["timing"]["tfidf"])
        self.assertEqual(report["timing"]["hp_search"], {})
        rr.render_markdown(report)     # must not raise


class TestOutputs(RunReportBase):
    def test_markdown_sections(self):
        self.make_all()
        md = rr.render_markdown(self.report())
        for heading in ("## 1. Run identity",
                        "## 2. Data selection strategy",
                        "## 3. Data volume actually used",
                        "## 4. Time spent",
                        "## 5. Best architecture per search",
                        "## 6. Published artifacts",
                        "## 8. Resolved configuration snapshot"):
            self.assertIn(heading, md)
        self.assertIn("vTest_out", md)

    def test_csv_index_appends_one_row_per_run(self):
        self.make_all()
        path = self.base / "runs_index.csv"
        report = self.report()
        rr.append_csv(path, rr.build_csv_row(report, "a.md"))
        rr.append_csv(path, rr.build_csv_row(report, "b.md"))
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["output_tag"], "vTest_out")
        self.assertEqual(rows[0]["sub_train_rate"], "0.05")
        self.assertEqual(rows[0]["first_train_records"], "1000")
        self.assertEqual(rows[0]["hp_epochs"], "8")
        self.assertEqual(rows[0]["best_f1_first"], "0.91")
        self.assertEqual(rows[1]["report_md"], "b.md")
        # header written exactly once
        self.assertEqual(path.read_text().count("generated_at"), 1)

    def test_cli_end_to_end(self):
        self.make_all()
        cfg_path = self.tmp / "cfg.json"
        cfg_path.write_text(json.dumps(self.cfg()))
        out_md = self.base / "results" / "r.md"
        out_json = self.base / "results" / "r.json"
        index = self.base / "idx.csv"
        code = rr.main(["--config-json", str(cfg_path),
                        "--out-md", str(out_md),
                        "--out-json", str(out_json),
                        "--index-csv", str(index),
                        "--repo-root", str(ROOT)])
        self.assertEqual(code, 0)
        self.assertTrue(out_md.is_file())
        self.assertTrue(index.is_file())
        payload = json.loads(out_json.read_text())
        self.assertEqual(payload["report_version"], rr.REPORT_VERSION)
        self.assertEqual(payload["identity"]["model_tag"], "v9.9")

    def test_cli_missing_config_returns_error(self):
        self.assertEqual(rr.main(["--config-json", str(self.tmp / "nope.json"),
                                  "--out-md", str(self.tmp / "a.md"),
                                  "--out-json", str(self.tmp / "a.json")]), 1)

    def test_dry_run_writes_nothing(self):
        self.make_all()
        cfg_path = self.tmp / "cfg.json"
        cfg_path.write_text(json.dumps(self.cfg()))
        out_md = self.base / "results" / "dry.md"
        code = rr.main(["--config-json", str(cfg_path),
                        "--out-md", str(out_md),
                        "--out-json", str(self.base / "results" / "dry.json"),
                        "--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse(out_md.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
