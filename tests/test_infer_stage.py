"""Tests for the infer stage -- the pipeline step added after publish.

Deliberately stdlib-only. ``tiara2.stages.infer`` imports torch/skorch/numba
lazily (inside methods), so the parts that actually decide WHAT runs -- input
resolution, output naming, the resume fingerprint, and config validation --
are testable without a deep-learning stack. That is the whole point of keeping
those imports local.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiara2 import cli as _cli
from tiara2.config import ConfigError, validate
from tiara2.stage import StageContext, registry
from tiara2.stages import infer as _infer  # noqa: F401  (registers the stage)


def _ctx(cfg, tmp_path):
    class _Log:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    return StageContext(cfg=cfg, work_dir=Path(tmp_path) / ".work", log=_Log())


def _stage():
    return registry()["infer"]()


def _base_cfg(**infer):
    cfg = {
        "base": "/data/x",
        "splits": ["train", "validation", "test"],
        "split_priority": ["test", "validation", "train"],
        # validate() runs against a merged config; mirror the defaults the
        # loader would have supplied so this fixture exercises infer only.
        "genus_balance": {"scope": "global", "max_species_per_genus": 1},
        "train": {"k_first": [5, 6, 7], "k_second": [5, 6, 7, 8]},
        "publish": {"model_tag": "v2.2.0", "first_count": 3, "second_count": 4},
    }
    if infer:
        cfg["infer"] = infer
    return cfg


def _must_fail(cfg, text):
    try:
        validate(cfg)
    except ConfigError as exc:
        assert text in str(exc), f"expected {text!r} in:\n{exc}"
        return
    raise AssertionError(f"validate() accepted a config it should reject: {text}")


# --------------------------------------------------------------------------- #

def test_infer_runs_after_publish_in_the_canonical_order():
    """Order lives in exactly one place; infer must be the terminal step."""
    order = _cli.DEFAULT_ORDER
    assert "infer" in order, "infer stage is not registered in DEFAULT_ORDER"
    assert order.index("infer") == order.index("publish") + 1
    assert order[-1] == "infer"
    assert "infer" in _cli.TRAINING_STAGES
    assert "infer" in registry()
    print("OK infer is the last stage, immediately after publish")


def test_inputs_accept_file_dir_and_glob_without_duplicates(tmp_path):
    """One key takes a file, a whole panel directory, or a glob."""
    panel = tmp_path / "panel"
    panel.mkdir()
    for name in ("a.fasta", "b.fna", "c.fa.gz", "notes.txt"):
        (panel / name).write_text(">x\nACGT\n")
    single = tmp_path / "single.fasta"
    single.write_text(">y\nACGT\n")

    cfg = _base_cfg(inputs=[str(single), str(panel),
                            str(panel / "*.fna"), str(single)])
    found = _stage()._resolve_inputs(_ctx(cfg, tmp_path))

    assert str(single) in found
    assert str(panel / "a.fasta") in found
    assert str(panel / "c.fa.gz") in found
    # non-fasta files in a directory are ignored
    assert str(panel / "notes.txt") not in found
    # the glob and the directory both yield b.fna -- it must appear once
    assert found.count(str(panel / "b.fna")) == 1
    assert len(found) == len(set(found))
    print(f"OK resolved {len(found)} inputs from file + dir + glob, deduplicated")


def test_outputs_are_named_per_input(tmp_path):
    out_dir = tmp_path / "preds"
    fasta = tmp_path / "fungi_main.fasta"
    fasta.write_text(">x\nACGT\n")
    cfg = _base_cfg(inputs=[str(fasta)], out_dir=str(out_dir))
    outs = _stage().outputs(_ctx(cfg, tmp_path))
    assert outs == [str(out_dir / "fungi_main.tsv")], outs
    print("OK output TSV is named after its input")


def test_fingerprint_covers_the_published_pack(tmp_path):
    """Re-publishing must invalidate old predictions, not resume past them."""
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">x\nACGT\n")
    cfg = _base_cfg(inputs=[str(fasta)])
    cfg["publish"]["nnet_dest"] = "tiara/models/nnet-models-v2.2.0"
    cfg["publish"]["tfidf_dest"] = "tiara/models/tfidf-models-v2.2.0"
    inputs = _stage().inputs(_ctx(cfg, tmp_path))
    assert any("nnet-models-v2.2.0" in p for p in inputs), inputs
    assert any("tfidf-models-v2.2.0" in p for p in inputs), inputs
    assert str(fasta) in inputs
    print("OK resume fingerprint spans both the inputs and the published pack")


def test_disabled_by_default_is_a_no_op(tmp_path):
    """Publish succeeding must not silently start classifying a 600 GiB panel."""
    cfg = _base_cfg(enabled=False, inputs=[])
    result = _stage().run(_ctx(cfg, tmp_path))
    assert result["counts"]["status"] == "disabled"
    print("OK infer is opt-in; disabled is a clean no-op")


def test_enabled_without_inputs_fails_loudly(tmp_path):
    _must_fail(_base_cfg(enabled=True, inputs=[]),
               "infer.enabled is true but infer.inputs is empty")
    print("OK enabling inference with no input is rejected at config load")


def test_untrained_k_cannot_be_requested_for_inference():
    """k=4 was trained in v2.1.3 but not in v2.2.0; asking for it must fail."""
    _must_fail(_base_cfg(k_first=4), "is not in train.k_first")
    _must_fail(_base_cfg(k_second=4), "is not in train.k_second")
    # k=8 IS trained in v2.2.0 and must be accepted
    validate(_base_cfg(k_second=8))
    print("OK inference k is checked against the k that were actually trained")


def test_threshold_and_class_typos_are_rejected():
    _must_fail(_base_cfg(prob_cutoff={"first": 1.5}),
               "infer.prob_cutoff.first must be in (0, 1)")
    _must_fail(_base_cfg(prob_cutoff={"third": 0.5}),
               "infer.prob_cutoff.third is not a stage")
    _must_fail(_base_cfg(to_fasta=["mit", "mitochondria"]),
               "unknown class 'mitochondria'")
    _must_fail(_base_cfg(device="gpu"),
               "infer.device must be 'cpu', 'cuda' or null")
    validate(_base_cfg(prob_cutoff={"first": 0.938553, "second": 0.999999},
                       to_fasta=["mit", "pla", "pro"], device="cuda"))
    print("OK thresholds, classes and device are validated before a run starts")


def test_shipped_config_is_valid_and_inference_is_off():
    """The config we ship must load, and must not auto-classify."""
    try:
        import yaml
    except ImportError:
        print("SKIP shipped config check (PyYAML not installed)")
        return
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    assert cfg["infer"]["enabled"] is False, "shipped config must not auto-infer"
    assert cfg["infer"]["min_len"] == 3000
    assert cfg["infer"]["prob_cutoff"] == {"first": None, "second": None}, \
        "shipped thresholds must stay unset so they get re-calibrated"
    print("OK shipped config: infer present, disabled, thresholds unset")


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
