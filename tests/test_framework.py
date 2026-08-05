"""Unit tests for the framework core (config, binning, gpu policy).

Run: python3 tests/test_framework.py   (no pytest dependency required)
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import LengthBinner, dedup_params_for_bin, can_be_duplicate  # noqa: E402


def test_binning_discrete_and_varlen():
    b = LengthBinner(mode="discrete", discrete_lengths=(1000, 2000, 5000))
    assert b.bin_key(1000) == "L1000"
    assert b.bin_key(5000) == "L5000"
    # non-fixed length -> single VAR bin
    assert b.bin_key(3333) == "VAR"
    assert b.bin_key(12000) == "VAR"
    # fixed + VAR bins are self-contained
    assert b.bins_to_search("L1000") == ["L1000"]
    assert b.bins_to_search("VAR") == ["VAR"]
    print("ok: discrete + single VAR bin")


def test_dedup_params_cov_mode():
    cfg = {"dedup": {"min_seq_id": 0.95, "min_cov": 0.95, "cov_mode": 0,
                     "varlen_bin": "VAR",
                     "varlen": {"min_seq_id": 0.95, "min_cov": 0.95, "cov_mode": 5}}}
    assert dedup_params_for_bin("L1000", cfg)["cov_mode"] == 0
    assert dedup_params_for_bin("VAR", cfg)["cov_mode"] == 5
    print("ok: fixed bins cov-mode 0, VAR bin cov-mode 5 (short-cov)")


def test_length_invariant():
    # T6 fixed lengths: no cross-bin double-95 duplicates possible
    lengths = [1000, 2000, 3000, 5000, 10000]
    for i, a in enumerate(lengths):
        for j, c in enumerate(lengths):
            if i != j:
                assert not can_be_duplicate(a, c, 0.95), (a, c)
    assert can_be_duplicate(1000, 1000, 0.95)
    print("ok: T6 fixed lengths are cross-bin duplicate-free")


def test_config_load_merge_env_template():
    import tiara2.config as C
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text(
            "base: /data/X\n"
            "input_tag: v1\n"
            "classes: [bacteria, archaea]\n"
            "source_ready: '{base}/ready_{input_tag}'\n"
            "dedup: {length_mode: discrete, min_seq_id: 0.95, min_cov: 0.95}\n"
        )
        os.environ["TIARA2_dedup__min_seq_id"] = "0.97"
        cfg = C.load(p, overrides=["dedup.max_seqs=50"])
        del os.environ["TIARA2_dedup__min_seq_id"]
    assert cfg["source_ready"] == "/data/X/ready_v1", cfg["source_ready"]
    assert cfg["dedup"]["min_seq_id"] == 0.97   # env override
    assert cfg["dedup"]["max_seqs"] == 50       # --set override
    print("ok: config merge + template + env + --set precedence")


def test_config_validation_rejects_bad():
    import tiara2.config as C
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text("base: /x\nsplits: [train, test]\n"
                     "split_priority: [train]\nclasses: [a]\n")
        try:
            C.load(p)
        except C.ConfigError:
            print("ok: validation rejects mismatched split_priority")
            return
    raise AssertionError("expected ConfigError")


def test_paths_are_relative_and_portable():
    import tiara2.paths as P
    # repo root is discovered from __file__, not hardcoded
    assert (P.repo_root() / "tiara2").is_dir()
    assert (P.repo_root() / "tiara").is_dir()   # vendored model package present
    # relative code path resolves under repo; absolute data path passes through
    assert P.resolve("scripts") == P.repo_root() / "scripts"
    assert str(P.resolve("/data/shouhanyu/Tiara2")) == "/data/shouhanyu/Tiara2"
    env = P.pythonpath_with_repo({})
    assert str(P.repo_root()) in env["PYTHONPATH"]
    print("ok: code paths relative/portable, data paths absolute pass-through")


def test_model_backend_contract_and_registry():
    import tiara2.model_backend as M
    assert "tiara" in M.available_backends()
    b = M.get_backend("tiara")
    for verb in ("train_tfidf", "hp_search", "train_models"):
        assert callable(getattr(b, verb))

    # a new algorithm can register without touching the pipeline
    @M.register_backend("dummy")
    class _Dummy(M.ModelBackend):
        def train_tfidf(self, cfg, ctx): return "tfidf"
        def hp_search(self, cfg, ctx, *, stage, k): return f"{stage}{k}"
        def train_models(self, cfg, ctx): return "models"
    assert "dummy" in M.available_backends()
    print("ok: ModelBackend contract + swappable registry (reserved interface)")


def test_train_config_paths_resolve():
    import tiara2.config as C
    cfg = C.load(ROOT / "config" / "config.yaml")
    t = cfg["train"]
    # Training consumes the DEDUPED corpus (regroup's output), not the raw
    # source. These used to be the same path, which meant the dedup stage's
    # product was built and then read by nobody.
    # Tier-aware since v2.2b: the corpus lives on the HOT tier (fast_base, NVMe)
    # because featurize re-reads it once per (stage, k); published models and
    # results stay on the COLD tier because losing them costs a retrain.
    # Asserted structurally, not as literals, so retagging a run cannot break it.
    hot, cold = cfg["fast_base"], cfg["base"]
    assert cfg["source_ready"] == f"{hot}/train_ready_{cfg['corpus_tag']}", cfg["source_ready"]
    assert cfg["corpus_ready"] == f"{hot}/corpus_ready_{cfg['corpus_tag']}", cfg["corpus_ready"]
    # Since v2.1.3 training consumes the BP-BALANCED corpus, never the raw
    # corpus_ready. v2.1.2 collapsed because the split it actually trained on
    # was 98.6% eukarya, so this identity is now the load-bearing assertion.
    fb = cfg["fragment_bp_balance"]
    if fb.get("enabled"):
        assert t["train_ready"] == fb["output_root"], t["train_ready"]
        assert t["train_ready"] != cfg["corpus_ready"]
    else:
        assert t["train_ready"] == cfg["corpus_ready"]
    assert t["train_ready"] != cfg["source_ready"]
    # Both splits that feed a fit must be balanced, and test must stay raw.
    assert set(fb["balance_splits"]) == {"train", "validation"}, fb["balance_splits"]
    assert "test" not in fb["balance_splits"]
    # Everything downstream of train must carry version_tag, not corpus_tag:
    # reusing a corpus-tagged tfidf/feature cache across model versions is how
    # a stale vocabulary gets published next to fresh weights.
    for key in ("tfidf_dir", "feature_cache", "seq_pack"):
        assert t[key].endswith(cfg["version_tag"]), (key, t[key])
    assert cfg["publish"]["tfidf_src"] == t["tfidf_dir"]
    assert t["feature_cache"].startswith(hot), t["feature_cache"]
    assert t["seq_pack"].startswith(hot), t["seq_pack"]
    assert t["out_models"] == f"{cold}/models_{cfg['model_tag']}_optimizedHP_gpu", t["out_models"]
    assert t["k_first"] == [5, 6, 7] and t["k_second"] == [5, 6, 7, 8]
    # tracks config.yaml: 40/36 was counter-productive (disk thrash), so the
    # HP scheduler was retuned to 8 GPUs x 2 tasks = 16.
    assert 1 <= t["gpu"]["hp_maxpar"] <= 40, t["gpu"]["hp_maxpar"]
    # anti-collapse knobs must stay switched on
    assert t["gpu"]["hp_val_balance"] == "equal"
    assert t["gpu"]["class_weight"] == "balanced"
    assert 0 < t["gpu"]["sanity_max_pred_share"] <= 1
    assert 0 < t["gates"]["min_first_mean_f1"] <= 1
    # never refit on validation unless validation itself is balanced
    assert (not t["final_include_validation"]
            or "validation" in fb["balance_splits"])
    # the run-report step must stay wired up in publish
    assert cfg["publish"]["report"]["enabled"] is True
    assert cfg["publish"]["report"]["out_md"].startswith(
        f"{cold}/results_"), cfg["publish"]["report"]["out_md"]
    print("ok: train paths template to the existing /data/shouhanyu layout")


def test_gpu_scheduler_places_jobs_without_gpu():
    # No nvidia-smi in sandbox -> query_gpus() empty -> scheduler must not hang;
    # with fake handles that finish immediately and a monkeypatched picker.
    import tiara2.resources as R
    R.query_gpus = lambda: [{"index": 0, "free_mib": 20000, "util": 0},
                            {"index": 1, "free_mib": 20000, "util": 0}]
    launched = []

    class Handle:
        def poll(self):
            return 0

    sched = R.GpuScheduler(R.GpuPolicy(allowed=[0, 1], launch_delay=0, poll_seconds=0))
    jobs = [(lambda gpu, i=i: (launched.append((i, gpu)), Handle())[1]) for i in range(4)]
    sched.run(jobs, wait_fn=lambda h: None)
    assert len(launched) == 4, launched
    print(f"ok: gpu scheduler placed {len(launched)} jobs across cards")


if __name__ == "__main__":
    test_binning_discrete_and_varlen()
    test_dedup_params_cov_mode()
    test_length_invariant()
    test_config_load_merge_env_template()
    test_config_validation_rejects_bad()
    test_paths_are_relative_and_portable()
    test_model_backend_contract_and_registry()
    test_train_config_paths_resolve()
    test_gpu_scheduler_places_jobs_without_gpu()
    print("\nALL FRAMEWORK TESTS PASSED")
