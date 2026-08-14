#!/usr/bin/env python3
"""Unified v2.5.1 generalist + v2.6.1 lightweight BioSignal residual inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tiara.hierarchical.data import fasta
from tiara.hierarchical.model import HierarchicalClassifier, probabilities
from tiara.hierarchical.schema import HierarchySchema
from tiara.src.transformations import TfidfWeighter
from tiara.training.featurize_cache import featurize_block

STOPS = {"TAA", "TAG", "TGA"}
STARTS = {"ATG", "GTG", "TTG"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entropy(values) -> float:
    total = float(sum(values))
    return 0.0 if not total else -sum((x / total) * math.log2(x / total) for x in values if x)


def _reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _frame_features(seq: str, offset: int) -> list[float]:
    valid = [seq[i : i + 3] for i in range(offset, len(seq) - 2, 3) if set(seq[i : i + 3]) <= set("ACGT")]
    n = max(len(valid), 1)
    stops = [i for i, codon in enumerate(valid) if codon in STOPS]
    segments, previous = [], -1
    for position in stops + [len(valid)]:
        segments.append(position - previous - 1); previous = position
    longest = max(segments, default=0); mean_orf = sum(segments) / max(len(segments), 1)
    p90 = float(np.percentile(segments, 90)) if segments else 0.0
    return [len(stops) / n, sum(codon in STARTS for codon in valid) / n, longest * 3 / max(len(seq), 1), mean_orf * 3 / max(len(seq), 1), p90 * 3 / max(len(seq), 1), sum(x >= 100 for x in segments) / max(n / 100.0, 1.0)]


def _match_fraction(seq: str, lag: int) -> float:
    valid = matches = 0
    for a, b in zip(seq[:-lag], seq[lag:]):
        if a in "ACGT" and b in "ACGT": valid += 1; matches += a == b
    return matches / max(valid, 1)


def biosignal_features(seq: str) -> np.ndarray:
    length = len(seq); counts = [seq.count(base) for base in "ACGTN"]; acgt = max(sum(counts[:4]), 1)
    max_run = run = 0; previous = ""
    for char in seq:
        run = run + 1 if char == previous else 1; previous = char; max_run = max(max_run, run)
    tri = Counter(seq[i : i + 3] for i in range(max(length - 2, 0)))
    vector = [math.log1p(length), (counts[1] + counts[2]) / acgt, counts[4] / max(length, 1), _entropy(counts[:4]) / 2.0, _entropy(tri.values()) / 6.0, max_run / max(length, 1)]
    rows = []
    for strand in (seq, _reverse_complement(seq)):
        for offset in range(3):
            row = _frame_features(strand, offset); rows.append(row); vector.extend(row[:4])
    matrix = np.asarray([row[:4] for row in rows], dtype=np.float32)
    for column in range(4):
        values = matrix[:, column]; vector.extend([float(values.min()), float(values.max()), float(values.std())])
    for row in rows: vector.extend(row[4:])
    phases = []
    for offset in range(3):
        phase = seq[offset::3]; valid = sum(phase.count(base) for base in "ACGT")
        phases.append((phase.count("G") + phase.count("C")) / max(valid, 1))
    vector.extend(phases); vector.extend([float(np.std(phases)), float(max(phases) - min(phases))])
    vector.append(_match_fraction(seq, 3) - 0.5 * (_match_fraction(seq, 1) + _match_fraction(seq, 2)))
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (60,): raise RuntimeError(f"unexpected BioSignal feature shape: {result.shape}")
    return result


def classify(bundle, input_fasta, output, batch=512, device=None, min_len=1000, max_records=None):
    import joblib

    bundle_path = Path(bundle).resolve(); manifest = json.loads(bundle_path.read_text())
    if manifest.get("format") != "tiara2-biosignal-residual-v1": raise ValueError("invalid BioSignal manifest")
    resolve = lambda value: Path(value) if Path(value).is_absolute() else (bundle_path.parent / value).resolve()
    base_path, expert_path = resolve(manifest["base"]["checkpoint"]), resolve(manifest["expert"]["model"])
    for name, path in (("base", base_path), ("expert", expert_path)):
        if _sha256(path) != manifest[name]["sha256"]: raise ValueError(f"{name} model hash mismatch")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(base_path, map_location=dev, weights_only=False)
    schema = HierarchySchema.from_dict(checkpoint["schema"]); model = HierarchicalClassifier(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"]); model.to(dev).eval()
    expert = joblib.load(expert_path)
    if int(getattr(expert, "n_features_in_", -1)) != 60: raise ValueError("BioSignal expert feature dimension mismatch")
    tfidf = TfidfWeighter.load_params(str(resolve(manifest["tfidf"])))
    router = manifest["router"]; temperatures = manifest.get("temperatures"); thresholds = manifest.get("thresholds", {})

    def flush(records, writer):
        if not records: return
        sequences = [seq for _, seq in records]
        matrix = featurize_block(sequences, tfidf.k, np.asarray(tfidf.idfs, dtype=np.float32), 4 ** tfidf.k)
        with torch.inference_mode():
            logits = model(torch.from_numpy(matrix).to(dev)); base_probs = probabilities(logits, temperatures)
        root_probability = base_probs["root"].detach().cpu().numpy()
        expert_probability = expert.predict_proba(np.stack([biosignal_features(seq) for seq in sequences]))[:, 1]
        p = np.clip(expert_probability, 1e-7, 1 - 1e-7)
        evidence = np.maximum(0.0, np.log(p / (1 - p)) - np.log(router["tau"] / (1 - router["tau"])))
        lengths = np.asarray([len(seq) for seq in sequences]); weights = np.ones(len(lengths), dtype=np.float32)
        weights[(lengths >= 2500) & (lengths < 5000)] = router["middle_weight"]; weights[lengths >= 5000] = router["long_weight"]
        boost = np.minimum(router["alpha"] * weights * evidence, router["delta_max"])
        eligible = root_probability[:, 0] >= router["base_euk_min"]
        adjusted_logits = np.log(np.clip(root_probability, 1e-12, 1.0)); adjusted_logits[:, 0] += boost * eligible
        adjusted_root = np.exp(adjusted_logits - adjusted_logits.max(1, keepdims=True)); adjusted_root /= adjusted_root.sum(1, keepdims=True)
        base_choice, adjusted_choice = root_probability.argmax(1), adjusted_root.argmax(1)
        promoted = (base_choice != 0) & (adjusted_choice == 0) & eligible
        for i, (header, seq) in enumerate(records):
            root_index = int(adjusted_choice[i]); root = schema.profile.root[root_index]; root_p = float(adjusted_root[i, root_index])
            leaf, leaf_p = root, root_p; branch = {"euk_nuclear": "euk", "prok": "prok", "organelle": "organelle"}.get(root)
            if branch:
                branch_probs = base_probs[branch][i]; leaf_index = int(branch_probs.argmax()); leaf = schema.classes(branch)[leaf_index]; leaf_p = float(branch_probs[leaf_index])
            if root_p < float(thresholds.get("root", 0)) or (branch and leaf_p < float(thresholds.get(branch, 0))): leaf = "unknown"
            writer.writerow([header, len(seq), "base+biosignal" if promoted[i] else "base", root, leaf, f"{root_p:.8f}", f"{leaf_p:.8f}"])

    with Path(output).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t"); writer.writerow(["record_id", "length_bp", "expert", "root", "leaf", "root_probability", "leaf_probability"])
        records, accepted = [], 0
        for header, sequence in fasta(Path(input_fasta)):
            if len(sequence) < min_len: continue
            records.append((header, sequence)); accepted += 1
            if len(records) >= batch: flush(records, writer); records = []
            if max_records is not None and accepted >= max_records: break
        flush(records, writer)
