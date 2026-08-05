#!/usr/bin/env python3
"""Tiara v1.2 unified data preparation pipeline (03 + 04).

Pipeline
--------
raw genomes -> deterministic parallel chopping -> resumable per-genome shards
-> class mapping / genome-level validation / deterministic euk downsampling
-> canonical train_ready + final-training flat directory + benchmark subset.

This is intentionally CPU/I/O optimized rather than GPU based: gzip decoding,
FASTA parsing, string slicing and file writing do not benefit from CUDA.
```
BASE=/data/shouhanyu/Tiara2

PYTHONUNBUFFERED=1 \
nice -n 10 \
ionice -c2 -n7 \
python3 03_04_prepare_dataset_v2_evaluate.py \
  --base "$BASE" \
  --workflow evaluation-only \
  --evaluation-mode virus \
  --tag v2_0_hybrid_virus_eval \
  --work-subdir .work_03_04_v2_0_hybrid_virus_eval \
  --evaluation-out-subdir evaluation_ready_v2_0_hybrid_virus \
  --preset T6-v2c-equal-v3ov \
  --workers 2 \
  --max-in-flight 3 \
  --seed 42 \
  2>&1 | tee "$BASE/index/virus_evaluation_prepare.log"
```
Default v1.2 run (fixed 5 kb, 32 safe workers on a 128-core shared host)::

    python3 03_04_prepare_dataset_v1_2.py \
      --base /data/shouhanyu/Tiara2 \
      --tag v1.2 --mode fixed --frag-len 5000 --n-chops 1 \
      --workers 32 --seed 42 --drop-train-species-in-test

Outputs::

    train_ready_v1.2/
      train|validation|test/{archaea,bacteria,eukarya,mitochondria,plastids}.fasta
      train|validation/{organelle,archea}.fasta  # legacy search compatibility
      test_benchmark/                            # deterministic capped euk test
      fragments.tsv
      prepare_manifest.json

    train_ready_flat_v1.2/
      {archaea,bacteria,eukarya,mitochondria}_fr.fasta
      plast_fr.fasta
      prepare_manifest.json

Resume is shard-level. A worker writes temporary files and atomically publishes
its shard plus a done JSON. Interrupted final merges are rebuilt from shards.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiara2 import labels as _labels
_labels.self_check()  # fail fast if class routing regresses

CLASSES = ("bacteria", "archaea", "eukarya", "mitochondria", "plastids")
FLAT_NAMES = {
    "bacteria": "bacteria_fr.fasta",
    "archaea": "archaea_fr.fasta",
    "eukarya": "eukarya_fr.fasta",
    "mitochondria": "mitochondria_fr.fasta",
    "plastids": "plast_fr.fasta",
}
GROUP_META_DEFAULT = dict(_labels.GROUP_META_DEFAULT)
_NON_ATGC = str.maketrans("", "", "ATGC")
BUFFER_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class Task:
    group: str
    split: str
    accession: str
    source: str
    label: str
    supergroup: str
    species_taxid: str
    class_name: str
    task_id: str


def log(message: str) -> None:
    print(message, flush=True)


def stable_hex(text: str, size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def stable_fraction(text: str) -> float:
    value = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
    return value / float((1 << 64) - 1)


def stable_seed(seed: int, *parts: str) -> int:
    raw = ":".join([str(seed), *parts])
    return int.from_bytes(hashlib.blake2b(raw.encode(), digest_size=8).digest(), "big")


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("r", buffering=BUFFER_SIZE)


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    seq: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def is_clean(seq: str) -> bool:
    return len(seq.translate(_NON_ATGC)) == 0


def fragment_digest(seq: str) -> bytes:
    return hashlib.blake2b(seq.encode("ascii"), digest_size=12).digest()


def chop_fixed(seq: str, rng: random.Random, length: int,
               n_chops: int, overlap: int, phase_mode: str = "random") -> Iterator[str]:
    """Fixed-length chopping: zero phase for T0, random phase for T1/T2/T3."""
    n=len(seq)
    if n<length: return
    step=length-overlap; seen=set()
    for _ in range(n_chops):
        if phase_mode=="zero":
            pos=0
        else:
            max_offset=min(length-1,n-length)
            pos=rng.randint(0,max_offset) if max_offset>0 else 0
        while pos+length<=n:
            frag=seq[pos:pos+length]; pos+=step
            if not is_clean(frag): continue
            digest=fragment_digest(frag)
            if digest in seen: continue
            seen.add(digest); yield frag



def chop_varlen(seq: str, rng: random.Random, len_min: int, len_max: int,
                coverage: float, log_uniform: bool) -> Iterator[str]:
    n = len(seq)
    if n < len_min:
        return
    target_bp = int(coverage * n)
    got_bp = 0
    tries = 0
    hi = min(len_max, n)
    max_tries = max(100, target_bp // len_min * 6 + 100)
    seen: set[bytes] = set()
    while got_bp < target_bp and tries < max_tries:
        tries += 1
        if log_uniform:
            length = int(math.exp(rng.uniform(math.log(len_min), math.log(hi))))
        else:
            length = rng.randint(len_min, hi)
        start = rng.randint(0, n - length)
        frag = seq[start:start + length]
        if not is_clean(frag):
            continue
        digest = fragment_digest(frag)
        if digest in seen:
            continue
        seen.add(digest)
        got_bp += length
        yield frag



def chop_discrete(seq: str, rng: random.Random, lengths: list[int], weights: list[float],
                  overlap_ratio: float, min_tail: int) -> Iterator[str]:
    """Sequential weighted discrete lengths (legacy-v3/v2c), with ATGC filtering and dedup."""
    n=len(seq); pos=0; seen=set()
    while pos<n:
        length=rng.choices(lengths,weights=weights,k=1)[0]
        remaining=n-pos
        actual=min(length,remaining)
        if actual<min_tail: break
        frag=seq[pos:pos+actual]
        step=max(1,int(length*(1.0-overlap_ratio)))
        pos+=step
        if not is_clean(frag): continue
        digest=fragment_digest(frag)
        if digest in seen: continue
        seen.add(digest); yield frag


PRESETS = {
    "T0-original5k": {"mode":"fixed","phase_mode":"zero","frag_len":5000,"n_chops":1,"overlap":0},
    "T1-fixed5k-rphase": {"mode":"fixed","phase_mode":"random","frag_len":5000,"n_chops":1,"overlap":0},
    "T2-fixed5k-3phase": {"mode":"fixed","phase_mode":"random","frag_len":5000,"n_chops":3,"overlap":0},
    "T3-fixed5k-ov1k": {"mode":"fixed","phase_mode":"random","frag_len":5000,"n_chops":1,"overlap":1000},
    "T4-v2b-U3-15-cov2": {"mode":"varlen","len_min":3000,"len_max":15000,"coverage":2.0,"log_uniform":False},
    "T5-v2b-logU3-15-cov2": {"mode":"varlen","len_min":3000,"len_max":15000,"coverage":2.0,"log_uniform":True},
    "T6-v2c-equal-v3ov": {"mode":"discrete","discrete_lengths":"1000,2000,3000,5000,10000","discrete_weights":"1,1,1,1,1","discrete_overlap":0.2,"min_tail":500},
    "T7-v2c-equal-cov2": {"mode":"discrete","discrete_lengths":"1000,2000,3000,5000,10000","discrete_weights":"1,1,1,1,1","discrete_overlap":0.5,"min_tail":500},
    "v3-legacy-weighted": {"mode":"discrete","discrete_lengths":"1000,2000,3000,5000,10000","discrete_weights":"0.30,0.30,0.20,0.15,0.05","discrete_overlap":0.2,"min_tail":500},
}


def apply_preset(args: argparse.Namespace) -> None:
    if not args.preset: return
    for key,value in PRESETS[args.preset].items():
        setattr(args,key,value)


def parse_number_list(text: str, cast) -> list:
    values=[cast(x.strip()) for x in text.split(",") if x.strip()]
    if not values: raise ValueError("empty discrete list")
    return values


def publish_fragment_dataset(tasks: list[Task], eval_tasks: list[Task], shard_dir: Path,
                             target: Path, force: bool, config: dict[str,Any]) -> dict[str,Any]:
    prepare_publish_target(target,force)
    tmp=target.with_name(target.name+f".tmp.{os.getpid()}"); shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    handles={}; counts=defaultdict(int); bp=defaultdict(int); length_hist=defaultdict(int)
    try:
        for task in tasks+eval_tasks:
            if task.split=="evaluation": name=f"evaluation.{task.group}.fna"
            else: name=f"{task.split}.{task.class_name}.fna"
            if name not in handles: handles[name]=(tmp/name).open("w",buffering=BUFFER_SIZE)
            fasta,_,_=shard_paths(shard_dir,task)
            for header,sequence in iter_fasta(fasta):
                handles[name].write(f">{header}\n{sequence}\n")
                counts[name]+=1; bp[name]+=len(sequence); length_hist[(task.split,task.class_name,len(sequence))]+=1
    finally:
        for h in handles.values(): h.close()
    copy_tsv_shards(tasks+eval_tasks,shard_dir,tmp/"fragments.tsv")
    with (tmp/"length_distribution.tsv").open("w",newline="") as fh:
        w=csv.writer(fh,delimiter="\t"); w.writerow(["split","class","length","fragments","bp"])
        for (sp,c,l),n in sorted(length_hist.items()): w.writerow([sp,c,l,n,l*n])
    manifest={"status":"complete","workflow":"chop-only","config":config,"counts":dict(counts),"bp":dict(bp)}
    (tmp/"chop_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    os.replace(tmp,target); return manifest

def class_from_meta(label: str, supergroup: str) -> str | None:
    """Normalize canonical and legacy groups.tsv spellings into five classes.

    Delegates to tiara2.labels, which is the single source of truth.  The old
    inline version substring-matched the supergroup name, so "Archaeplastida"
    matched 'plastid' and every plant genome was written into plastids.fasta.
    Do not reintroduce substring matching on taxon names here.
    """
    return _labels.class_from_meta(label, supergroup)


def accession_from_path(path: Path) -> str:
    """Return canonical entity accession from a raw link or store filename.

    v2 store filenames may include an assembly name after the accession, e.g.
    GCF_000001405.40_GRCh38.p14_genomic.fna.gz.  Downstream taxid/split tables
    are keyed by GCA/GCF accession, so never use the entire store stem.
    """
    name = path.name
    match = re.search(r"(?:^|[^A-Z0-9])(GC[AF]_\d{9}\.\d+)(?:[^0-9]|$)", name)
    if match:
        return match.group(1)
    if name.startswith("JGI:"):
        return name.split("_genomic", 1)[0]
    for suffix in ("_genomic.fna.gz", "_genomic.fna", ".fna.gz", ".fna"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def load_group_meta(select_dir: Path) -> dict[str, tuple[str, str]]:
    meta = dict(GROUP_META_DEFAULT)
    path = select_dir / "groups.tsv"
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3:
                    meta[fields[0]] = (fields[1], fields[2])
    return meta


def load_taxids(select_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = select_dir / "taxid.tsv"
    if path.exists():
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2:
                    result[fields[0]] = fields[1]
    return result


def collect_tasks(raw_dir: Path, group_meta: dict[str, tuple[str, str]],
                  taxids: dict[str, str]) -> tuple[list[Task], list[str]]:
    tasks: list[Task] = []
    unknown: list[str] = []
    patterns = ("*_genomic.fna.gz", "*_genomic.fna", "*.fna.gz", "*.fna")
    for group_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        group = group_dir.name
        if group not in group_meta:
            unknown.append(group)
            continue
        label, supergroup = group_meta[group]
        class_name = class_from_meta(label, supergroup)
        if class_name is None:
            unknown.append(group)
            continue
        for split in ("train", "test"):
            split_dir = group_dir / split
            if not split_dir.is_dir():
                continue
            # Preserve the raw symlink name for accession identity, while reading
            # from the resolved canonical-store target.  Deduplicate targets only
            # within the same group/split.
            links: list[Path] = []
            for pattern in patterns:
                links.extend(p for p in split_dir.glob(pattern) if p.is_file())
            by_source: dict[Path, Path] = {}
            for link in sorted(set(links), key=str):
                source = link.resolve()
                by_source.setdefault(source, link)
            for source, link in sorted(by_source.items(), key=lambda item: str(item[0])):
                accession = accession_from_path(link)
                key = f"{group}\t{split}\t{accession}\t{source}"
                tasks.append(Task(
                    group=group,
                    split=split,
                    accession=accession,
                    source=str(source),
                    label=label,
                    supergroup=supergroup,
                    species_taxid=taxids.get(accession, "NA"),
                    class_name=class_name,
                    task_id=stable_hex(key, 12),
                ))
    tasks.sort(key=lambda t: (t.group, t.split, t.accession, t.source))
    return tasks, sorted(set(unknown))


def apply_species_leakage_filter(tasks: list[Task], taxids: dict[str, str],
                                 enabled: bool) -> tuple[list[Task], int]:
    if not enabled:
        return tasks, 0
    if not taxids:
        raise RuntimeError("--drop-train-species-in-test requires select/taxid.tsv")
    train_species = {
        task.species_taxid for task in tasks
        if task.split == "train" and task.species_taxid != "NA"
    }
    filtered = [
        task for task in tasks
        if not (task.split == "test" and task.species_taxid in train_species)
    ]
    return filtered, len(tasks) - len(filtered)


def source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def shard_paths(shard_dir: Path, task: Task) -> tuple[Path, Path, Path]:
    stem = f"{task.split}.{task.class_name}.{task.accession}.{task.task_id}"
    return shard_dir / f"{stem}.fna", shard_dir / f"{stem}.tsv", shard_dir / f"{stem}.done.json"


def shard_is_complete(shard_dir: Path, task: Task, fingerprint: str) -> bool:
    fasta, tsv, done = shard_paths(shard_dir, task)
    if not (fasta.exists() and tsv.exists() and done.exists()):
        return False
    try:
        meta = json.loads(done.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("status") == "complete"
        and meta.get("fingerprint") == fingerprint
        and meta.get("source_signature") == source_signature(Path(task.source))
    )


def process_task(payload: dict[str, Any]) -> dict[str, Any]:
    task = Task(**payload["task"])
    cfg = payload["config"]
    shard_dir = Path(payload["shard_dir"])
    fingerprint = payload["fingerprint"]
    fasta_path, tsv_path, done_path = shard_paths(shard_dir, task)
    if shard_is_complete(shard_dir, task, fingerprint):
        meta = json.loads(done_path.read_text())
        meta["resumed"] = True
        return meta

    fasta_tmp = fasta_path.with_suffix(fasta_path.suffix + f".tmp.{os.getpid()}")
    tsv_tmp = tsv_path.with_suffix(tsv_path.suffix + f".tmp.{os.getpid()}")
    done_tmp = done_path.with_suffix(done_path.suffix + f".tmp.{os.getpid()}")
    for path in (fasta_tmp, tsv_tmp, done_tmp):
        path.unlink(missing_ok=True)

    rng = random.Random(stable_seed(cfg["seed"], task.group, task.split,
                                    task.accession, task.source))
    count = 0
    total_bp = 0
    started = time.time()
    try:
        with fasta_tmp.open("w", buffering=BUFFER_SIZE) as fasta_out, \
             tsv_tmp.open("w", buffering=BUFFER_SIZE) as tsv_out:
            for _, sequence in iter_fasta(Path(task.source)):
                sequence = sequence.upper()
                if cfg["mode"] == "fixed":
                    fragments = chop_fixed(sequence, rng, cfg["frag_len"], cfg["n_chops"],
                                           cfg["overlap"], cfg["phase_mode"])
                elif cfg["mode"] == "varlen":
                    fragments = chop_varlen(sequence, rng, cfg["len_min"], cfg["len_max"],
                                            cfg["coverage"], cfg["log_uniform"])
                else:
                    fragments = chop_discrete(sequence, rng, cfg["discrete_lengths"],
                                              cfg["discrete_weights"], cfg["discrete_overlap"],
                                              cfg["min_tail"])
                for fragment in fragments:
                    fid = f"{task.accession}|{count}"
                    fasta_out.write(
                        f">{fid} sg={task.supergroup} label={task.label} "
                        f"epoch={task.split}\n{fragment}\n"
                    )
                    tsv_out.write(
                        f"{fid}\t{task.accession}\t{task.species_taxid}\t"
                        f"{task.supergroup}\t{task.label}\t{task.split}\t"
                        f"{len(fragment)}\n"
                    )
                    count += 1
                    total_bp += len(fragment)

        os.replace(fasta_tmp, fasta_path)
        os.replace(tsv_tmp, tsv_path)
        meta = {
            "status": "complete",
            "fingerprint": fingerprint,
            "task": asdict(task),
            "source_signature": source_signature(Path(task.source)),
            "count": count,
            "total_bp": total_bp,
            "seconds": time.time() - started,
            "resumed": False,
        }
        done_tmp.write_text(json.dumps(meta, indent=2))
        os.replace(done_tmp, done_path)
        return meta
    finally:
        fasta_tmp.unlink(missing_ok=True)
        tsv_tmp.unlink(missing_ok=True)
        done_tmp.unlink(missing_ok=True)


def validation_split(accession: str, val_pct: int, seed: int) -> bool:
    return stable_fraction(f"validation:{seed}:{accession}") < val_pct / 100.0


def read_shard_meta(shard_dir: Path, task: Task) -> dict[str, Any]:
    _, _, done = shard_paths(shard_dir, task)
    return json.loads(done.read_text())


def prepare_publish_target(target: Path, force: bool) -> None:
    if not target.exists():
        return
    if not force:
        raise FileExistsError(
            f"Output exists: {target}. Use --force to back it up and replace it."
        )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".backup_{stamp}")
    log(f"[safety] backing up {target} -> {backup}")
    os.replace(target, backup)


def copy_tsv_shards(tasks: list[Task], shard_dir: Path, output: Path) -> None:
    with output.open("w", buffering=BUFFER_SIZE) as out:
        out.write("frag_id\taccession\tspecies_taxid\tsupergroup\tlabel\tepoch\tlength\n")
        for task in tasks:
            _, tsv, _ = shard_paths(shard_dir, task)
            with tsv.open("r", buffering=BUFFER_SIZE) as handle:
                shutil.copyfileobj(handle, out, length=BUFFER_SIZE)


def create_benchmark_subset(test_dir: Path, benchmark_dir: Path,
                            test_counts: dict[str, int], euk_cap: int) -> dict[str, int]:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for class_name in CLASSES:
        source = test_dir / f"{class_name}.fasta"
        target = benchmark_dir / f"{class_name}.fasta"
        total = test_counts.get(class_name, 0)
        rate = 1.0
        if class_name == "eukarya" and euk_cap > 0 and total > euk_cap:
            rate = euk_cap / total
        written = 0
        with target.open("w", buffering=BUFFER_SIZE) as out:
            for header, sequence in iter_fasta(source):
                fid = header.split()[0]
                if rate < 1.0 and stable_fraction("benchmark:" + fid) >= rate:
                    continue
                out.write(f">{header}\n{sequence}\n")
                written += 1
        counts[class_name] = written
    return counts



ORIGINAL_TIARA_TARGETS = {
    "bacteria": 439077, "archaea": 179370, "eukarya": 327639,
    "mitochondria": 28881, "plastids": 59649,
}


def load_policy(path: str) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def scaled_ratios(ratios: dict[str, float], total: int) -> dict[str, int]:
    values = {c: float(ratios.get(c, 0)) for c in CLASSES}
    denom = sum(values.values())
    if denom <= 0:
        raise ValueError("ratios must sum to > 0")
    raw = {c: total * values[c] / denom for c in CLASSES}
    out = {c: int(math.floor(raw[c])) for c in CLASSES}
    remainder = int(total - sum(out.values()))
    order = sorted(CLASSES, key=lambda c: raw[c] - out[c], reverse=True)
    for c in order[:remainder]:
        out[c] += 1
    return out


def derive_targets(strategy: str, available: dict[str, int], target_total: int,
                   euk_mult: float, policy: dict[str, Any]) -> tuple[dict[str,int],dict[str,int],dict[str,int]]:
    if strategy == "preserve":
        requested = dict(available)
    elif strategy == "original-tiara":
        requested = (scaled_ratios(ORIGINAL_TIARA_TARGETS, target_total)
                     if target_total else dict(ORIGINAL_TIARA_TARGETS))
    elif strategy == "balanced":
        total = target_total or 5 * min(available.values())
        requested = scaled_ratios({c: 1 for c in CLASSES}, total)
    elif strategy == "legacy":
        requested = dict(available)
        requested["eukarya"] = int(euk_mult * (available.get("bacteria",0)+available.get("archaea",0)))
    elif strategy == "custom":
        if policy.get("targets"):
            requested = {c: int(policy["targets"].get(c,0)) for c in CLASSES}
        elif policy.get("ratios"):
            total = int(policy.get("target_total") or target_total or 0)
            if total <= 0:
                raise ValueError("custom ratios require target_total")
            requested = scaled_ratios(policy["ratios"], total)
        else:
            raise ValueError("custom policy requires targets or ratios")
    else:
        raise ValueError("unknown strategy: "+strategy)
    effective={c:min(max(0,int(requested.get(c,0))),int(available.get(c,0))) for c in CLASSES}
    shortages={c:max(0,int(requested.get(c,0))-int(available.get(c,0))) for c in CLASSES}
    return requested,effective,shortages


def allocate_rates(counts: dict[str,int], target: int, mode: str,
                   floor: int=0, cap: int=0) -> tuple[dict[str,float],dict[str,float]]:
    counts={a:int(n) for a,n in counts.items() if int(n)>0}
    if not counts:
        return {},{}
    capacities={a:min(n,cap) if cap else n for a,n in counts.items()}
    target=min(int(target),sum(capacities.values()))
    quota={a:0.0 for a in counts}
    if floor and target>0:
        base={a:min(floor,capacities[a]) for a in counts}
        scale=min(1.0,target/max(1,sum(base.values())))
        quota={a:base[a]*scale for a in counts}
    remaining=max(0.0,target-sum(quota.values()))
    active={a for a in counts if quota[a]<capacities[a]}
    while remaining>1e-9 and active:
        weights={a:(counts[a] if mode=="proportional" else math.sqrt(counts[a]) if mode=="sqrt" else 1.0) for a in active}
        denom=sum(weights.values())
        progressed=0.0
        for a in list(active):
            add=min(remaining*weights[a]/denom,capacities[a]-quota[a])
            quota[a]+=add; progressed+=add
            if quota[a]>=capacities[a]-1e-9:
                active.discard(a)
        if progressed<=1e-9: break
        remaining=max(0.0,remaining-progressed)
    rates={a:min(1.0,quota[a]/counts[a]) for a in counts}
    return quota,rates


def _evaluation_class(collection: str) -> tuple[str,str,str]:
    if collection.startswith("mag_bacteria"):
        return "prok","Bacteria","bacteria"
    if collection.startswith("mag_archaea"):
        return "prok","Archaea","archaea"
    if collection.startswith("mag_nuclear_euk") or collection.startswith("mag_euk"):
        return "euk","MAG-Eukaryota","eukarya"
    if collection.startswith("virus") or collection.startswith("mag_virus"):
        return "evaluation","Virus","virus"
    return "evaluation",collection,collection


def collect_evaluation_tasks(evaluation_dir: Path, mode: str) -> list[Task]:
    if mode == "none" or not evaluation_dir.is_dir():
        return []
    metadata: dict[str, dict[str,str]] = {}
    meta_path=evaluation_dir/"metadata.tsv"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8",newline="") as fh:
            for row in csv.DictReader(fh,delimiter="\t"):
                for key in (row.get("path",""),Path(row.get("path","")).name,row.get("entity_id","")):
                    if key: metadata[str(key)]=row
    tasks=[]; patterns=("*_genomic.fna.gz","*_genomic.fna","*.fna.gz","*.fna")
    for collection_dir in sorted(p for p in evaluation_dir.iterdir() if p.is_dir()):
        collection=collection_dir.name
        is_mag = collection.startswith("mag_") and not collection.startswith("mag_virus")
        is_virus = collection.startswith("virus") or collection.startswith("mag_virus")
        if mode == "mag" and not is_mag:
            continue
        if mode == "virus" and not is_virus:
            continue
        links=[]
        for pat in patterns:
            links.extend(x for x in collection_dir.glob(pat) if x.is_file())
        by_source={}
        for link in sorted(set(links),key=str):
            by_source.setdefault(link.resolve(),link)
        label,supergroup,class_name=_evaluation_class(collection)
        for source,link in sorted(by_source.items(),key=lambda item:str(item[0])):
            acc=accession_from_path(link)
            row=(metadata.get(str(link)) or metadata.get(link.name) or
                 metadata.get(str(source)) or metadata.get(source.name) or metadata.get(acc) or {})
            key=f"{collection}\tevaluation\t{acc}\t{source}"
            tasks.append(Task(collection,"evaluation",acc,str(source),label,supergroup,
                              row.get("species_taxid") or "NA",class_name,stable_hex(key,12)))
    return tasks


def merge_evaluation_outputs(tasks: list[Task], shard_dir: Path, output: Path,
                             cfg: dict[str,Any]) -> dict[str,Any]:
    output.mkdir(parents=True,exist_ok=True)
    handles={}; counts=defaultdict(int); bp=defaultdict(int)
    cap=int(cfg.get("evaluation_fragment_cap_per_genome",0))
    try:
        for task in tasks:
            fasta,_,_=shard_paths(shard_dir,task)
            total=int(read_shard_meta(shard_dir,task).get("count",0))
            rate=min(1.0,cap/total) if cap and total else 1.0
            if task.group not in handles:
                handles[task.group]=(output/f"{task.group}.fasta").open("w",buffering=BUFFER_SIZE)
            for header,sequence in iter_fasta(fasta):
                fid=header.split()[0]
                if rate<1 and stable_fraction("evaluation:"+fid)>=rate:
                    continue
                handles[task.group].write(f">{header}\n{sequence}\n")
                counts[task.group]+=1; bp[task.group]+=len(sequence)
    finally:
        for h in handles.values(): h.close()
    copy_tsv_shards(tasks,shard_dir,output/"fragments.tsv")
    manifest={"status":"complete","dataset_role":"test_only","training_included":False,
              "mode":cfg.get("evaluation_mode"),"counts":dict(counts),"bp":dict(bp)}
    (output/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    return manifest

def merge_outputs(tasks: list[Task], shard_dir: Path, ready_tmp: Path,
                  flat_tmp: Path, cfg: dict[str, Any], fingerprint: str,
                  raw_counts: dict[str, int]) -> dict[str, Any]:
    for split in ("train", "validation", "test"):
        (ready_tmp / split).mkdir(parents=True, exist_ok=True)
    flat_tmp.mkdir(parents=True, exist_ok=True)
    canonical={}; organelle={}; flat={}
    for split in ("train","validation","test"):
        for class_name in CLASSES:
            canonical[(split,class_name)]=(ready_tmp/split/f"{class_name}.fasta").open("w",buffering=BUFFER_SIZE)
        organelle[split]=(ready_tmp/split/"organelle.fasta").open("w",buffering=BUFFER_SIZE)
    for class_name,filename in FLAT_NAMES.items():
        flat[class_name]=(flat_tmp/filename).open("w",buffering=BUFFER_SIZE)

    policy=load_policy(cfg.get("policy_json",""))
    requested,targets,shortages=derive_targets(cfg["strategy"],raw_counts,cfg["target_total"],cfg["euk_target_mult"],policy)
    genome_counts={c:{} for c in CLASSES}
    for task in tasks:
        if task.split=="train":
            genome_counts[task.class_name][task.accession]=int(read_shard_meta(shard_dir,task)["count"])
    rates={}; quotas={}
    for c in CLASSES:
        quotas[c],rates[c]=allocate_rates(genome_counts[c],targets[c],cfg["allocation"],cfg["per_genome_floor"],cfg["per_genome_cap"])
    selected=defaultdict(int)
    try:
        for task in tasks:
            fasta,_,_=shard_paths(shard_dir,task)
            final_split=("test" if task.split=="test" else
                         "validation" if validation_split(task.accession,cfg["val_pct"],cfg["seed"]) else "train")
            rate=rates.get(task.class_name,{}).get(task.accession,1.0) if task.split=="train" else 1.0
            for header,sequence in iter_fasta(fasta):
                fid=header.split()[0]
                if rate<1 and stable_fraction("keep:"+task.class_name+":"+fid)>=rate:
                    continue
                record=f">{header}\n{sequence}\n"
                canonical[(final_split,task.class_name)].write(record)
                selected[(final_split,task.class_name)]+=1
                if task.class_name in ("mitochondria","plastids"):
                    organelle[final_split].write(record)
                if task.split=="train":
                    flat[task.class_name].write(record)
    finally:
        for h in canonical.values(): h.close()
        for h in organelle.values(): h.close()
        for h in flat.values(): h.close()
    for split in ("train","validation","test"):
        (ready_tmp/split/"archea.fasta").symlink_to("archaea.fasta")
    copy_tsv_shards(tasks,shard_dir,ready_tmp/"fragments.tsv")
    test_counts={c:selected[("test",c)] for c in CLASSES}
    benchmark_counts=create_benchmark_subset(ready_tmp/"test",ready_tmp/"test_benchmark",test_counts,cfg["benchmark_euk_cap"])
    plan=[{"class":c,"available":raw_counts.get(c,0),"requested":requested[c],
           "effective_target":targets[c],"shortage":shortages[c],"genomes":len(genome_counts[c]),
           "allocation":cfg["allocation"],"expected_quota":sum(quotas[c].values())} for c in CLASSES]
    with (ready_tmp/"adjustment_plan.tsv").open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(plan[0]),delimiter="\t"); w.writeheader(); w.writerows(plan)
    manifest={"pipeline":"03_04_prepare_dataset_v1_2.py","status":"complete","fingerprint":fingerprint,
              "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"config":cfg,
              "raw_train_counts":raw_counts,"adjustment_plan":plan,
              "selected_counts":{sp:{c:selected[(sp,c)] for c in CLASSES} for sp in ("train","validation","test")},
              "benchmark_counts":benchmark_counts,"task_count":len(tasks)}
    text=json.dumps(manifest,indent=2,ensure_ascii=False)
    (ready_tmp/"prepare_manifest.json").write_text(text); (flat_tmp/"prepare_manifest.json").write_text(text)
    return manifest



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("/data/shouhanyu/Tiara2"))
    parser.add_argument("--tag", default="v2_0_hybrid")
    parser.add_argument("--raw-subdir", default="raw")
    parser.add_argument("--select-subdir", default="select")
    parser.add_argument("--out-subdir", default=None,
                        help="default: train_ready_<tag>")
    parser.add_argument("--flat-out-subdir", default=None,
                        help="default: train_ready_flat_<tag>")
    parser.add_argument("--work-subdir", default=None,
                        help="default: .work_03_04_<tag>")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="", help="T0-T7 or legacy-v3 named chopping preset")
    parser.add_argument("--mode", choices=("fixed", "varlen", "discrete"), default="fixed")
    parser.add_argument("--phase-mode", choices=("zero", "random"), default="random")
    parser.add_argument("--frag-len", type=int, default=5000)
    parser.add_argument("--n-chops", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--len-min", type=int, default=3000)
    parser.add_argument("--len-max", type=int, default=15000)
    parser.add_argument("--coverage", "--cov", dest="coverage", type=float, default=2.0)
    parser.add_argument("--log-uniform", action="store_true")
    parser.add_argument("--discrete-lengths", default="1000,2000,3000,5000,10000")
    parser.add_argument("--discrete-weights", default="1,1,1,1,1")
    parser.add_argument("--discrete-overlap", type=float, default=0.2)
    parser.add_argument("--min-tail", type=int, default=500)
    parser.add_argument("--workflow", choices=("all", "chop-only", "evaluation-only"), default="all",
                        help="evaluation-only skips raw Train/Test and writes only evaluation_ready output")
    parser.add_argument("--frag-out-subdir", default=None, help="chop-only default: frag_<tag>")
    parser.add_argument("--val-pct", type=int, default=10)
    parser.add_argument("--strategy", choices=("preserve", "original-tiara", "balanced", "legacy", "custom"), default="legacy")
    parser.add_argument("--policy-json", default="")
    parser.add_argument("--target-total", type=int, default=0)
    parser.add_argument("--euk-target-mult", type=float, default=1.0)
    parser.add_argument("--allocation", choices=("proportional", "sqrt", "equal"), default="sqrt")
    parser.add_argument("--per-genome-floor", type=int, default=0)
    parser.add_argument("--per-genome-cap", type=int, default=0)
    parser.add_argument("--benchmark-euk-cap", type=int, default=200000)
    parser.add_argument("--evaluation-mode", choices=("none", "mag", "virus", "all"), default="mag",
                        help="evaluation collection filter; virus selects virus_* and mag_virus only")
    parser.add_argument("--evaluation-subdir", default="evaluation")
    parser.add_argument("--evaluation-out-subdir", default=None, help="default: evaluation_ready_<tag>")
    parser.add_argument("--evaluation-fragment-cap-per-genome", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    auto_workers = min(32, max(1, (os.cpu_count() or 4) // 4))
    parser.add_argument("--workers", type=int, default=auto_workers,
                        help=f"parallel genome workers (default: {auto_workers})")
    parser.add_argument("--max-in-flight", type=int, default=0,
                        help="bounded submitted tasks; default 2*workers. Execution-only, does not change resume fingerprint")
    parser.add_argument("--drop-train-species-in-test", action="store_true",
                        help="remove test genomes whose species_taxid occurs in train")
    parser.add_argument("--force", action="store_true",
                        help="back up and replace completed output directories")
    parser.add_argument("--cleanup-work", action="store_true",
                        help="delete resumable shards after successful publication")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers<1: raise ValueError("--workers must be >=1")
    if args.max_in_flight<0: raise ValueError("--max-in-flight must be >=0")
    if not (0<=args.val_pct<100): raise ValueError("--val-pct must be in [0,100)")
    if args.mode=="fixed":
        if args.frag_len<1 or args.n_chops<1: raise ValueError("fixed lengths/chops must be positive")
        if not (0<=args.overlap<args.frag_len): raise ValueError("0 <= overlap < frag_len required")
    elif args.mode=="varlen":
        if args.len_min<1 or args.len_max<args.len_min or args.coverage<=0: raise ValueError("invalid varlen parameters")
    else:
        args.discrete_lengths_list=parse_number_list(args.discrete_lengths,int)
        args.discrete_weights_list=parse_number_list(args.discrete_weights,float)
        if len(args.discrete_lengths_list)!=len(args.discrete_weights_list): raise ValueError("discrete lengths/weights size mismatch")
        if any(x<=0 for x in args.discrete_lengths_list) or any(x<0 for x in args.discrete_weights_list) or sum(args.discrete_weights_list)<=0: raise ValueError("invalid discrete values")
        if not (0<=args.discrete_overlap<1) or args.min_tail<1: raise ValueError("invalid discrete overlap/min-tail")
    if args.target_total<0 or args.per_genome_floor<0 or args.per_genome_cap<0: raise ValueError("target/floor/cap must be >=0")
    if args.evaluation_fragment_cap_per_genome<0: raise ValueError("evaluation cap must be >=0")
    if args.strategy=="custom" and not args.policy_json: raise ValueError("custom strategy requires policy JSON")



def main() -> int:
    args=parse_args(); apply_preset(args); validate_args(args)
    base=args.base.expanduser().resolve(); raw_dir=base/args.raw_subdir; select_dir=base/args.select_subdir
    evaluation_dir=base/args.evaluation_subdir
    out_name=args.out_subdir or f"train_ready_{args.tag}"; flat_name=args.flat_out_subdir or f"train_ready_flat_{args.tag}"
    eval_name=args.evaluation_out_subdir or f"evaluation_ready_{args.tag}"
    work_name=args.work_subdir or f".work_03_04_{args.tag}"
    ready_out=base/out_name; flat_out=base/flat_name; eval_out=base/eval_name; work_root=base/work_name
    if args.workflow != "evaluation-only" and not raw_dir.is_dir():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")
    if args.workflow == "evaluation-only" and not evaluation_dir.is_dir():
        raise FileNotFoundError(f"evaluation directory not found: {evaluation_dir}")
    config={"tag":args.tag,"workflow":args.workflow,"preset":args.preset,"mode":args.mode,"phase_mode":args.phase_mode,
            "frag_len":args.frag_len,"n_chops":args.n_chops,"overlap":args.overlap,
            "len_min":args.len_min,"len_max":args.len_max,"coverage":args.coverage,"log_uniform":args.log_uniform,
            "discrete_lengths":getattr(args,"discrete_lengths_list",[]),
            "discrete_weights":getattr(args,"discrete_weights_list",[]),
            "discrete_overlap":args.discrete_overlap,"min_tail":args.min_tail,"val_pct":args.val_pct,"strategy":args.strategy,
            "policy_json":str(Path(args.policy_json).expanduser().resolve()) if args.policy_json else "",
            "target_total":args.target_total,"euk_target_mult":args.euk_target_mult,
            "allocation":args.allocation,"per_genome_floor":args.per_genome_floor,
            "per_genome_cap":args.per_genome_cap,"benchmark_euk_cap":args.benchmark_euk_cap,
            "seed":args.seed,"drop_train_species_in_test":args.drop_train_species_in_test,
            "evaluation_mode":args.evaluation_mode,
            "evaluation_fragment_cap_per_genome":args.evaluation_fragment_cap_per_genome}
    fingerprint=stable_hex(json.dumps(config,sort_keys=True),12); shard_dir=work_root/fingerprint/"shards"; shard_dir.mkdir(parents=True,exist_ok=True)
    group_meta=load_group_meta(select_dir); taxids=load_taxids(select_dir)
    if args.workflow == "evaluation-only":
        tasks=[]; unknown=[]; removed=0
    else:
        tasks,unknown=collect_tasks(raw_dir,group_meta,taxids)
        if not tasks: raise RuntimeError(f"No FASTA tasks found under {raw_dir}/<group>/train,test")
        tasks,removed=apply_species_leakage_filter(tasks,taxids,args.drop_train_species_in_test)
    eval_tasks=collect_evaluation_tasks(evaluation_dir,args.evaluation_mode)
    if args.workflow == "evaluation-only" and not eval_tasks:
        raise RuntimeError(f"No evaluation tasks matched mode={args.evaluation_mode!r} under {evaluation_dir}")
    all_tasks=tasks+eval_tasks
    log(f"== Tiara {args.tag} unified 03+04 ==")
    log(f"raw_tasks={len(tasks)} evaluation_tasks={len(eval_tasks)} all_shards={len(all_tasks)} workers={args.workers} fingerprint={fingerprint}")
    log("[note] shard progress denominator = raw Train/Test tasks + evaluation tasks; evaluation never enters training outputs")
    if removed: log(f"[leakage] removed {removed} test genomes sharing species with train")
    if unknown: log(f"[warning] ignored unknown/unmapped groups: {unknown}")
    # Keep only a bounded number of ProcessPool futures in memory.  Submitting
    # 100k+ tasks at once causes avoidable coordinator RAM growth and worsens
    # swap/I/O pressure.  This execution setting intentionally stays outside
    # the data fingerprint, so a restart with fewer workers resumes old shards.
    max_in_flight=args.max_in_flight or max(2,args.workers*2)
    max_in_flight=max(args.workers,max_in_flight)
    completed=resumed=0; started=time.time(); task_iter=iter(all_tasks)
    def submit_one(executor, task):
        payload={"task":asdict(task),"config":config,"shard_dir":str(shard_dir),"fingerprint":fingerprint}
        return executor.submit(process_task,payload)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
        in_flight=set()
        for _ in range(min(max_in_flight,len(all_tasks))):
            try: in_flight.add(submit_one(executor,next(task_iter)))
            except StopIteration: break
        log(f"[executor] workers={args.workers} max_in_flight={max_in_flight}")
        while in_flight:
            done,in_flight=cf.wait(in_flight,return_when=cf.FIRST_COMPLETED)
            for future in done:
                result=future.result(); completed+=1; resumed+=int(bool(result.get("resumed")))
                try: in_flight.add(submit_one(executor,next(task_iter)))
                except StopIteration: pass
                if completed==1 or completed%25==0 or completed==len(all_tasks):
                    elapsed=time.time()-started; eta=(elapsed/completed)*(len(all_tasks)-completed)
                    log(f"[shards] {completed}/{len(all_tasks)} resumed={resumed} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")
    if args.workflow == "evaluation-only":
        prepare_publish_target(eval_out,args.force)
        eval_tmp=eval_out.with_name(eval_out.name+f".tmp.{os.getpid()}")
        shutil.rmtree(eval_tmp,ignore_errors=True)
        try:
            eval_manifest=merge_evaluation_outputs(eval_tasks,shard_dir,eval_tmp,config)
            os.replace(eval_tmp,eval_out)
        except BaseException:
            shutil.rmtree(eval_tmp,ignore_errors=True)
            raise
        log("== evaluation-only publication complete ==")
        log(json.dumps(eval_manifest.get("counts",{}),indent=2))
        log(f"evaluation={eval_out}")
        if args.cleanup_work: shutil.rmtree(work_root/fingerprint,ignore_errors=True)
        else: log(f"[resume] kept shards at {shard_dir}")
        return 0

    raw_counts=defaultdict(int)
    for task in tasks:
        if task.split=="train": raw_counts[task.class_name]+=int(read_shard_meta(shard_dir,task)["count"])
    missing=[c for c in CLASSES if raw_counts.get(c,0)==0]
    if missing: raise RuntimeError("No train fragments for classes: "+", ".join(missing)+". Check raw/ and select/groups.tsv")
    if args.workflow=="chop-only":
        frag_out=base/(args.frag_out_subdir or f"frag_{args.tag}")
        manifest=publish_fragment_dataset(tasks,eval_tasks,shard_dir,frag_out,args.force,config)
        log(f"chop-only complete: {frag_out}"); log(json.dumps(manifest["counts"],indent=2)); return 0
    prepare_publish_target(ready_out,args.force); prepare_publish_target(flat_out,args.force)
    if eval_tasks: prepare_publish_target(eval_out,args.force)
    ready_tmp=ready_out.with_name(ready_out.name+f".tmp.{os.getpid()}"); flat_tmp=flat_out.with_name(flat_out.name+f".tmp.{os.getpid()}")
    eval_tmp=eval_out.with_name(eval_out.name+f".tmp.{os.getpid()}")
    for x in (ready_tmp,flat_tmp,eval_tmp): shutil.rmtree(x,ignore_errors=True)
    try:
        manifest=merge_outputs(tasks,shard_dir,ready_tmp,flat_tmp,config,fingerprint,dict(raw_counts))
        eval_manifest=merge_evaluation_outputs(eval_tasks,shard_dir,eval_tmp,config) if eval_tasks else {}
        manifest["evaluation_output"]={"path":str(eval_out),"manifest":eval_manifest}
        text=json.dumps(manifest,indent=2,ensure_ascii=False); (ready_tmp/"prepare_manifest.json").write_text(text); (flat_tmp/"prepare_manifest.json").write_text(text)
        required=[ready_tmp/sp/f"{c}.fasta" for sp in ("train","validation","test") for c in CLASSES]+[flat_tmp/n for n in FLAT_NAMES.values()]
        missing_files=[str(x) for x in required if not x.exists()]
        if missing_files: raise RuntimeError("Required outputs missing:\n"+"\n".join(missing_files))
        os.replace(ready_tmp,ready_out); os.replace(flat_tmp,flat_out)
        if eval_tasks: os.replace(eval_tmp,eval_out)
    except BaseException:
        for x in (ready_tmp,flat_tmp,eval_tmp): shutil.rmtree(x,ignore_errors=True)
        raise
    log("== publication complete =="); log(json.dumps(manifest["selected_counts"],indent=2)); log(f"ready={ready_out}"); log(f"flat={flat_out}")
    if eval_tasks: log(f"evaluation={eval_out}")
    if args.cleanup_work: shutil.rmtree(work_root/fingerprint,ignore_errors=True)
    else: log(f"[resume] kept shards at {shard_dir}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
