"""Unified inference entry point.

WHY THIS FILE EXISTS
--------------------
Until now the inference layer was addressed by hand: someone had to know which
``first_k-*.pkl`` pairs with which ``k*-first-stage`` folder, retype the
architecture into a params dict, and remember the prob_cutoff. That is exactly
the class of mistake that let the collapsed v2.1.2 pack reach a benchmark, and
it becomes worse in v2.2.0 where the k lists are configurable.

So the model set is DISCOVERED, never typed:

  training_manifest.json  ->  which .pkl exists, for which stage and k, with
                              which hidden_1/hidden_2/dropout/dim_out
  tfidf_manifest.json     ->  which k<k>-<stage>-stage folder exists
  config / --model-tag    ->  which published pack to read

The two are then joined into a strict (stage, k, weights, tfidf, cutoff) triple
per stage and handed to tiara.src.classification.Classification, which itself
re-validates the triple before loading anything.

The resolved set can be dumped to, or loaded from, an INFERENCE MANIFEST so a
benchmark run is reproducible byte for byte:

    python -m tiara2.cli classify --emit-manifest infer.json
    python -m tiara2.cli classify --manifest infer.json -i contigs.fa -o out.tsv
"""
from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

from . import paths as _paths

STAGES = ("first", "second")

# Default thresholds. These are the v1.4-era values and are ONLY safe for a
# smoke test: every prob_cutoff must be re-calibrated per model x task x panel
# x contig length after a new pack is trained.
DEFAULT_CUTOFF = {"first": 0.65, "second": 0.65}

_WEIGHT_RE = re.compile(
    r"^(?P<stage>first|second)_k-(?P<k>\d+)_"
    r"hidden_1-(?P<hidden_1>[^_]+)_hidden_2-(?P<hidden_2>[^_]+)_"
    r"lr-(?P<lr>[^_]+)_dropout-(?P<dropout>[^_]+)_epochs-(?P<epochs>\d+)\.pkl$"
)

# Stage output widths are fixed by the label maps in tiara/src/prediction.py:
# first stage indexes 0..4 (organelle/bacteria/unused/archaea/eukarya),
# second stage indexes 0..2 (plastid/unused/mitochondrion).
DIM_OUT = {"first": 5, "second": 3}


class InferenceError(RuntimeError):
    pass


@dataclass
class StageSpec:
    """One fully resolved stage: net + tf-idf + k + threshold."""

    stage: str
    k: int
    weights: str
    tfidf: str
    prob_cutoff: float
    fragment_len: int = 5000
    hidden_1: int = 0
    hidden_2: Optional[int] = None
    dropout: float = 0.0
    dim_out: int = 0
    mean_f1: Optional[float] = None

    def params(self) -> Dict[str, object]:
        """The dict Classification expects for this stage."""
        return {
            "k": int(self.k),
            "fragment_len": int(self.fragment_len),
            "prob_cutoff": float(self.prob_cutoff),
            "hidden_1": int(self.hidden_1),
            "hidden_2": self.hidden_2,
            "dropout": float(self.dropout),
            "dim_out": int(self.dim_out or DIM_OUT[self.stage]),
        }


@dataclass
class InferencePlan:
    model_tag: str
    nnet_dir: str
    tfidf_dir: str
    stages: Dict[str, StageSpec] = field(default_factory=dict)
    min_len: int = 3000
    threads: int = 1
    batch_records: int = 512
    device: Optional[str] = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["stages"] = {name: asdict(spec) for name, spec in self.stages.items()}
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "InferencePlan":
        payload = json.loads(text)
        stages = {
            name: StageSpec(**spec)
            for name, spec in (payload.pop("stages", {}) or {}).items()
        }
        plan = cls(**payload)
        plan.stages = stages
        return plan

    def ordered(self) -> List[StageSpec]:
        missing = [s for s in STAGES if s not in self.stages]
        if missing:
            raise InferenceError(
                "inference plan is missing stage(s): " + ", ".join(missing))
        return [self.stages[s] for s in STAGES]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise InferenceError(f"missing {path}")
    except ValueError as exc:
        raise InferenceError(f"{path} is not valid JSON: {exc}")


def _walk(node, key):
    """Yield every value stored under `key` anywhere in a nested structure."""
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                yield value
            else:
                yield from _walk(value, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, key)


def _index_manifest(manifest: dict) -> Dict[tuple, dict]:
    """Map (stage, k) -> the manifest record describing that trained model."""
    out: Dict[tuple, dict] = {}

    def visit(node):
        if isinstance(node, dict):
            stage = node.get("stage")
            k = node.get("k")
            if stage in STAGES and k is not None:
                out[(stage, int(k))] = node
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(manifest)
    return out


def discover_models(nnet_dir: Path, tfidf_dir: Path) -> Dict[str, List[dict]]:
    """List the (stage, k) combinations actually present on disk.

    Nothing is hardcoded here: v2.1.3 shipped first k=4,5,6 and second
    k=4,5,6,7; v2.2.0 ships first k=5,6,7 and second k=5,6,7,8. Both are read,
    not assumed.
    """
    nnet_dir = Path(nnet_dir)
    tfidf_dir = Path(tfidf_dir)
    if not nnet_dir.is_dir():
        raise InferenceError(f"nnet model dir not found: {nnet_dir}")
    if not tfidf_dir.is_dir():
        raise InferenceError(f"tfidf model dir not found: {tfidf_dir}")

    found: Dict[str, List[dict]] = {"first": [], "second": []}
    for path in sorted(nnet_dir.glob("*.pkl")):
        match = _WEIGHT_RE.match(path.name)
        if not match:
            continue
        stage = match.group("stage")
        k = int(match.group("k"))
        tfidf_sub = tfidf_dir / f"k{k}-{stage}-stage"
        if not (tfidf_sub / "model.npy").is_file():
            # A net without its matching idf vector is unusable; skip it here
            # and let select() report it if the user asked for that k.
            continue
        hidden_2 = match.group("hidden_2")
        found[stage].append(
            {
                "stage": stage,
                "k": k,
                "weights": str(path),
                "tfidf": str(tfidf_sub),
                "hidden_1": int(match.group("hidden_1")),
                "hidden_2": None if hidden_2.lower() == "none" else int(hidden_2),
                "dropout": float(match.group("dropout")),
            }
        )
    for stage in STAGES:
        found[stage].sort(key=lambda item: item["k"])
    return found


def build_plan(
    nnet_dir,
    tfidf_dir,
    model_tag: str = "",
    k_first: Optional[int] = None,
    k_second: Optional[int] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    min_len: int = 3000,
    threads: int = 1,
    batch_records: int = 512,
    device: Optional[str] = None,
    fragment_len: int = 5000,
) -> InferencePlan:
    """Resolve exactly one model per stage into an InferencePlan.

    Classification loads ONE net per stage -- the seven published .pkl files are
    not an ensemble and were never averaged. If k is not pinned, the model with
    the best recorded mean_f1 wins, falling back to the largest k.
    """
    nnet_dir = Path(nnet_dir)
    tfidf_dir = Path(tfidf_dir)
    found = discover_models(nnet_dir, tfidf_dir)
    manifest_index: Dict[tuple, dict] = {}
    manifest_path = nnet_dir / "training_manifest.json"
    if manifest_path.is_file():
        manifest_index = _index_manifest(_read_json(manifest_path))

    cutoffs = dict(cutoffs or {})
    wanted = {"first": k_first, "second": k_second}
    plan = InferencePlan(
        model_tag=model_tag or nnet_dir.name.replace("nnet-models-", ""),
        nnet_dir=str(nnet_dir),
        tfidf_dir=str(tfidf_dir),
        min_len=int(min_len),
        threads=int(threads),
        batch_records=int(batch_records),
        device=device,
    )

    for stage in STAGES:
        candidates = found[stage]
        if not candidates:
            raise InferenceError(
                f"no usable {stage}-stage model in {nnet_dir} paired with a "
                f"tf-idf folder in {tfidf_dir}")
        pinned = wanted[stage]
        if pinned is not None:
            picked = [c for c in candidates if c["k"] == int(pinned)]
            if not picked:
                raise InferenceError(
                    f"{stage} stage: k={pinned} requested but available k are "
                    + ", ".join(str(c["k"]) for c in candidates))
            chosen = picked[0]
        else:
            def score(item):
                record = manifest_index.get((stage, item["k"])) or {}
                mean_f1 = record.get("mean_f1")
                return (float(mean_f1) if mean_f1 is not None else -1.0, item["k"])

            chosen = max(candidates, key=score)
        record = manifest_index.get((stage, chosen["k"])) or {}
        mean_f1 = record.get("mean_f1")
        plan.stages[stage] = StageSpec(
            stage=stage,
            k=chosen["k"],
            weights=chosen["weights"],
            tfidf=chosen["tfidf"],
            prob_cutoff=float(cutoffs.get(stage, DEFAULT_CUTOFF[stage])),
            fragment_len=int(record.get("fragment_len", fragment_len) or fragment_len),
            hidden_1=int(record.get("hidden_1", chosen["hidden_1"])),
            hidden_2=record.get("hidden_2", chosen["hidden_2"]),
            dropout=float(record.get("dropout", chosen["dropout"])),
            dim_out=int(record.get("dim_out", DIM_OUT[stage]) or DIM_OUT[stage]),
            mean_f1=float(mean_f1) if mean_f1 is not None else None,
        )
    return plan


def plan_from_config(cfg: dict, **overrides) -> InferencePlan:
    """Build a plan from the pipeline config, using the published pack.

    The published pack is the one publish.py wrote into the package:
    ``tiara/models/nnet-models-<model_tag>`` and ``tfidf-models-<model_tag>``.
    """
    train = cfg.get("train", {}) or {}
    publish = cfg.get("publish", {}) or {}
    model_tag = str(publish.get("model_tag") or train.get("model_tag") or "").strip()
    if not model_tag:
        raise InferenceError(
            "config does not define publish.model_tag; cannot locate a "
            "published model pack")
    root = Path(_paths.TIARA_PKG) / "models"
    overrides.setdefault("fragment_len",
                         int(train.get("tfidf_fragment_len", 5000) or 5000))
    return build_plan(
        root / f"nnet-models-{model_tag}",
        root / f"tfidf-models-{model_tag}",
        model_tag=model_tag,
        **overrides,
    )


def make_classifier(plan: InferencePlan):
    """Instantiate Classification from a plan.

    Imported lazily: tiara.src pulls in torch, skorch, numba and Bio, which a
    config-only command should not have to pay for.
    """
    from tiara.src.classification import Classification

    specs = plan.ordered()
    return Classification(
        min_len=plan.min_len,
        nnet_weights=[spec.weights for spec in specs],
        params=[spec.params() for spec in specs],
        tfidf=[spec.tfidf for spec in specs],
        threads=plan.threads,
        batch_records=plan.batch_records,
        device=plan.device,
    )


# Column order is fixed by tiara.src.utilities.classes_list -- the original
# tool's output contract. Keep these in lockstep with generate_line().
HEADER = ["sequence_id", "class_fst_stage", "class_snd_stage"]
PROB_HEADER = [
    "p_organelle", "p_bacteria", "p_archaea", "p_eukarya", "p_unknown",
    "p_plastid", "p_unknown_snd", "p_mitochondrion",
]

# The original tiara `--to_fasta / --tf` vocabulary, unchanged so existing
# habits and downstream pipelines keep working.
FASTA_ALIASES = {
    "mit": "mitochondrion",
    "pla": "plastid",
    "bac": "bacteria",
    "arc": "archaea",
    "euk": "eukarya",
    "unk": "unknown",
    "pro": "prokarya",
    "org": "organelle",
}
FASTA_CHOICES = tuple(FASTA_ALIASES) + ("all",)


def resolve_to_fasta(values) -> List[str]:
    """Normalise --to-fasta arguments into concrete class names."""
    if not values:
        return []
    wanted: List[str] = []
    for raw in values:
        token = str(raw).strip().lower()
        if not token:
            continue
        if token == "all":
            return ["all"]
        name = FASTA_ALIASES.get(token, token)
        if name not in FASTA_ALIASES.values():
            raise InferenceError(
                f"unknown --to-fasta class {raw!r}; choose from "
                + ", ".join(FASTA_CHOICES))
        if name not in wanted:
            wanted.append(name)
    return wanted


def _bucket(result) -> str:
    """Which fasta bucket a record belongs to (mirrors utilities.sort_type)."""
    first, second = result.cls
    return second if first == "organelle" else first


class _FastaWriter:
    """Lazily opens one fasta per class, so empty classes leave no stray file."""

    def __init__(self, wanted: List[str], out_dir: str, prefix: str,
                 gzip_output: bool = False):
        self.wanted = set(wanted)
        self.all = "all" in self.wanted
        self.out_dir = out_dir
        self.prefix = prefix
        self.gzip_output = gzip_output
        self.handles: Dict[str, object] = {}
        self.counts: Dict[str, int] = {}

    def _handle(self, name: str):
        handle = self.handles.get(name)
        if handle is None:
            suffix = ".fasta.gz" if self.gzip_output else ".fasta"
            path = os.path.join(self.out_dir, f"{self.prefix}{name}{suffix}")
            opener = gzip.open if self.gzip_output else open
            handle = opener(path, "wt")
            self.handles[name] = handle
        return handle

    def write(self, result) -> None:
        if not self.wanted:
            return
        name = _bucket(result)
        if not (self.all or name in self.wanted):
            return
        handle = self._handle(name)
        handle.write(">" + result.desc + "\n")
        handle.write(result.seq + "\n")
        self.counts[name] = self.counts.get(name, 0) + 1

    def close(self) -> Dict[str, int]:
        for handle in self.handles.values():
            handle.close()
        return dict(self.counts)

    def paths(self) -> List[str]:
        suffix = ".fasta.gz" if self.gzip_output else ".fasta"
        return [os.path.join(self.out_dir, f"{self.prefix}{name}{suffix}")
                for name in sorted(self.counts)]


def run(plan: InferencePlan, fasta: str, out_path: str,
        probabilities: bool = False, verbose: bool = False,
        to_fasta=None, gzip_output: bool = False,
        log_path: Optional[str] = None) -> dict:
    """Classify one fasta and stream results out.

    Mirrors the original tool's output contract:
      * ``out_path``  -- TSV, header sequence_id / first stage / second stage,
                         plus the eight probability columns with --probabilities
      * ``log_path``  -- model parameters + per-class summary (the old
                         ``log_<output>`` file)
      * ``to_fasta``  -- per-class fasta files next to the output

    Returns a dict of counts; nothing is buffered, so a 600 GiB panel costs the
    same RAM as a 6 MiB one.
    """
    wanted = resolve_to_fasta(to_fasta)
    if gzip_output and not out_path.endswith(".gz"):
        out_path = out_path + ".gz"
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    classifier = make_classifier(plan)

    prefix = os.path.basename(out_path)
    for ext in (".gz", ".tsv", ".txt"):
        if prefix.endswith(ext):
            prefix = prefix[: -len(ext)]
    writer = _FastaWriter(wanted, out_dir or ".", prefix + "_", gzip_output)

    per_class: Dict[str, int] = {}
    written = 0
    tmp_path = out_path + ".partial"
    opener = gzip.open if gzip_output else open
    try:
        with opener(tmp_path, "wt") as handle:
            header = list(HEADER) + (list(PROB_HEADER) if probabilities else [])
            handle.write("\t".join(header) + "\n")
            for result in classifier.classify_iter(fasta, verbose=verbose):
                handle.write(result.generate_line(prob=probabilities) + "\n")
                name = _bucket(result)
                per_class[name] = per_class.get(name, 0) + 1
                writer.write(result)
                written += 1
        os.replace(tmp_path, out_path)
    finally:
        fasta_counts = writer.close()

    counts = {
        "input": fasta,
        "output": out_path,
        "records": written,
        "per_class": per_class,
        "fasta_written": fasta_counts,
        "fasta_paths": writer.paths(),
    }
    if log_path:
        write_log(log_path, plan, counts)
        counts["log"] = log_path
    return counts


def write_log(log_path: str, plan: InferencePlan, counts: dict) -> None:
    """Write the model-parameters + summary log the original tool produced.

    Deliberately verbose about WHICH models answered: a result file with no
    record of the k, the weights file and the threshold that produced it is
    exactly how a collapsed pack once got benchmarked without anyone noticing.
    """
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    lines = [
        f"model_tag: {plan.model_tag}",
        f"nnet_dir:  {plan.nnet_dir}",
        f"tfidf_dir: {plan.tfidf_dir}",
        f"min_len:   {plan.min_len}",
        f"threads:   {plan.threads}",
        f"batch_records: {plan.batch_records}",
        f"device:    {plan.device or 'auto'}",
        "",
        "models used",
        "-----------",
    ]
    for name in STAGES:
        spec = plan.stages[name]
        lines.append(
            f"  {name}: k={spec.k} prob_cutoff={spec.prob_cutoff:.6f} "
            f"hidden_1={spec.hidden_1} hidden_2={spec.hidden_2} "
            f"dropout={spec.dropout} fragment_len={spec.fragment_len}"
        )
        lines.append(f"    weights: {spec.weights}")
        lines.append(f"    tfidf:   {spec.tfidf}")
        if spec.mean_f1 is not None:
            lines.append(f"    recorded validation mean_f1: {spec.mean_f1:.6f}")
    total = int(counts.get("records", 0))
    lines += ["", "classification summary", "----------------------",
              f"  input:   {counts.get('input')}",
              f"  output:  {counts.get('output')}",
              f"  records: {total}"]
    for name, value in sorted((counts.get("per_class") or {}).items(),
                              key=lambda kv: -kv[1]):
        share = (value / total * 100.0) if total else 0.0
        lines.append(f"    {name:<14} {value:>12,}  ({share:6.2f}%)")
    for path in counts.get("fasta_paths") or []:
        lines.append(f"  wrote fasta: {path}")
    Path(log_path).write_text("\n".join(lines) + "\n")
